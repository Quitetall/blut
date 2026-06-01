#!/usr/bin/env python3
"""SNN event-level evaluation — the metric that actually matters for the
CR-controller, alongside the per-timestep number.

Per-timestep CRIT recall undersells the controller: a seizure event spans many
timesteps, and escalating ANY of them flags the event for high-fidelity coding.
This reports per-WINDOW event recall (a window containing a true seizure -> did
the controller escalate it to CRITICAL?) at several persistence thresholds k,
plus the false-escalation cost (background windows wrongly escalated = CR loss).

Reconstructs the model architecture from the checkpoint's state_dict (robust to
config drift) and mirrors train_4state_controller.validate() exactly.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

# Evaluate the production model on its native input (21-ch L3); never inject
# detail bands here.
os.environ.pop("SNN_DETAIL_BANDS", None)

from lamquant.snn.lma_dataset import LmaDataset  # noqa: E402
from lamquant.snn.train_4state_controller import derive_batch_targets, _confusion  # noqa: E402
from lamquant_neural.models.mamba_ssm_minimal import MambaSNN  # noqa: E402
from lamquant_neural.models.heads import build_head  # noqa: E402

CRIT = 3
L3_T = 313


def infer_arch(sd):
    in_ch = int(sd["spatial_mix.weight"].shape[1])
    d_model = int(sd["spatial_mix.weight"].shape[0])
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("ssm_blocks.")})
    n_layers = (max(layers) + 1) if layers else 2
    a_keys = [k for k in sd if k.endswith("fwd.A_log")]
    d_state = int(sd[a_keys[0]].shape[-1]) if a_keys else 8
    return in_ch, d_model, d_state, n_layers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lma-root", nargs="+", required=True, type=Path)
    p.add_argument("--split-manifest", required=True, type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--max-windows", type=int, default=4000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--persist", type=int, nargs="+", default=[1, 3, 8])
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd_m, sd_h = ck["model"], ck["head"]
    in_ch, d_model, d_state, n_layers = infer_arch(sd_m)
    use_spectral = in_ch != 21
    head_kind = ck.get("head_kind", "attention_softmax")
    quiet_thr = float(ck["quiet_rms_threshold"])
    logged = ck.get("metrics", {}).get("critical_recall")
    print(f"[event] arch in_ch={in_ch} d_model={d_model} d_state={d_state} "
          f"n_layers={n_layers} head={head_kind} spectral={use_spectral} "
          f"quiet_thr={quiet_thr:.4g} (logged CRIT_rec={logged})")

    dev = args.device if torch.cuda.is_available() else "cpu"
    model = MambaSNN(in_channels=in_ch, d_model=d_model, d_state=d_state,
                     n_layers=n_layers, use_subband=True).to(dev)
    head = build_head(head_kind, K=4).to(dev)
    # strict=False: the auxiliary seizure_head arch may have drifted across runs
    # (single Linear -> MLP); it is NOT on the 4-state path
    # (model -> activity_logits -> head), so a mismatch there is harmless. Assert
    # the 4-state-relevant keys (spatial_mix, ssm_blocks, activity head) all load.
    missing, unexpected = model.load_state_dict(sd_m, strict=False)
    bad = [k for k in missing if not k.startswith("seizure_head")]
    if bad:
        raise RuntimeError(f"4-state-path keys missing from checkpoint: {bad}")
    print(f"[event] loaded backbone (strict=False); "
          f"{len(missing)} seizure_head keys fresh-init (unused on 4-state path)")
    head.load_state_dict(sd_h)
    model.eval()
    head.eval()

    paths = []
    for r in args.lma_root:
        paths += sorted(Path(r).glob("*/*.lma")) or sorted(Path(r).glob("*.lma"))
    ds = LmaDataset(lma_paths=paths, split=args.split,
                    split_manifest_path=args.split_manifest)
    from torch.utils.data import Subset, DataLoader
    rng = np.random.default_rng(1337)
    idxs = rng.permutation(len(ds))[:min(args.max_windows, len(ds))].tolist()
    loader = DataLoader(Subset(ds, [int(i) for i in idxs]), batch_size=64,
                        num_workers=8, collate_fn=lambda b: b)

    if use_spectral:
        from lamquant.snn.spectral import build_augmented_input

    cm = np.zeros((4, 4), dtype=np.int64)
    sz_windows = 0
    bg_windows = 0
    fired = {k: 0 for k in args.persist}
    false_fired = {k: 0 for k in args.persist}
    n = 0
    for batch in loader:
        l3s, labs = [], []
        for sig, lab in batch:
            sig = np.asarray(sig, dtype=np.float32)
            if sig.shape[0] != in_ch:
                continue
            l3s.append(sig)
            labs.append(np.asarray(lab))
        if not l3s:
            continue
        l3 = torch.from_numpy(np.stack(l3s)).to(dev)
        labels = torch.from_numpy(np.stack(labs)).to(dev)
        x = build_augmented_input(l3) if use_spectral else l3
        with torch.no_grad():
            act, _, _ = model(x)
            _s, class_logits = head(act, L3_T)
        pred = class_logits.argmax(dim=1).cpu().numpy()        # [B, 313]
        target = derive_batch_targets(labels, l3, quiet_thr, L3_T).cpu().numpy()
        cm += _confusion(pred.ravel(), target.ravel())
        for b in range(pred.shape[0]):
            n_pred_crit = int((pred[b] == CRIT).sum())
            if (target[b] == CRIT).any():
                sz_windows += 1
                for k in args.persist:
                    if n_pred_crit >= k:
                        fired[k] += 1
            else:
                bg_windows += 1
                for k in args.persist:
                    if n_pred_crit >= k:
                        false_fired[k] += 1
        n += pred.shape[0]

    ts_rec = cm[CRIT, CRIT] / max(cm[CRIT, :].sum(), 1)
    print(f"\n[event] windows={n}  seizure_windows={sz_windows}  bg_windows={bg_windows}")
    print(f"[event] per-TIMESTEP CRIT recall = {ts_rec:.3f}")
    print(f"\n[event] per-WINDOW event recall  (seizure window escalated to >=k CRITICAL timesteps):")
    print(f"   {'k':>4s}  {'event_recall':>13s}  {'false_escalation':>17s}")
    for k in args.persist:
        er = fired[k] / max(sz_windows, 1)
        fe = false_fired[k] / max(bg_windows, 1)
        print(f"   {k:>4d}  {er:>13.3f}  {fe:>17.3f}")
    print("\n[event] event_recall = fraction of true-seizure windows the controller "
          "escalates (the safety metric); false_escalation = background windows "
          "escalated (the CR cost). k=1 lenient, k=8 sustained.")


if __name__ == "__main__":
    main()
