#!/usr/bin/env python3
"""Reconstruction-difficulty probe — does codec-MEASURED per-window difficulty
agree with the energy+seizure 4-state target, and is it cleaner?

The SNN picks one of 4 FSQ presets (levels 2/3/4/5) per window. The RIGHT label
is the MINIMUM-sufficient FSQ level for the decoder to reconstruct that window
within tolerance — a reconstruction-difficulty quantity the codec MEASURES, not
the seizure annotation (which paints whole windows CRITICAL via max3, the
dominant label noise).

For a window sample: encode -> latent; for L in {2,3,4,5}: FSQ-quantize the
latent to L levels -> decode -> measure reconstruction R(recon, L3). The
min-sufficient level = lowest L clearing the tolerance. Cross-tab that against
the current derive_4state_target to see (a) do they agree, (b) is the codec
label more separable.

This is the first turn of the codec eval loop / the D[y,a] harness.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.pop("SNN_DETAIL_BANDS", None)

ROOT = Path(__file__).resolve().parent.parent.parent  # blut/python
import sys
sys.path.insert(0, str(ROOT / "lamquant" / "student"))

LEVELS = (2, 3, 4, 5)
L3_T = 313


def fsq_quantize(latent: torch.Tensor, L: int) -> torch.Tensor:
    """Per-window FSQ to L levels: scale to [-1,1] by max-abs, round to L
    levels, scale back. The relative difficulty (which windows need more
    levels) is what the label cares about."""
    m = latent.abs().amax(dim=tuple(range(1, latent.dim())), keepdim=True).clamp(min=1e-6)
    x = (latent / m).clamp(-1.0, 1.0)
    idx = torch.round((x * 0.5 + 0.5) * (L - 1))
    q = (idx / (L - 1) * 2.0 - 1.0) * m
    return q


def pearson(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--tier", type=int, default=3)
    p.add_argument("--encoder-width", type=int, default=256,
                   help="must match the encoder ckpt (joint_fast = 256)")
    p.add_argument("--lma-root", type=Path, nargs="+", required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--quiet-thr", type=float, default=3.57)
    p.add_argument("--n-windows", type=int, default=400)
    p.add_argument("--rel-tol", type=float, default=0.98,
                   help="min-sufficient level = lowest L whose R >= rel_tol * R@L5")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    from joint_codec import build_default_joint
    from lamquant.common.utils import safe_torch_load as _safe
    enc_ck = _safe(str(args.encoder), map_location=dev)
    dec = _safe(str(args.decoder), map_location=dev)
    # Auto-read the exact arch from the encoder checkpoint's training_config so
    # we build the matching JointCodec (width/blocks/kernels/tier vary per ckpt).
    tc = enc_ck.get("training_config", {}) if isinstance(enc_ck, dict) else {}
    ek = tc.get("encoder_kernels", "3,5,7")
    ek = tuple(int(x) for x in ek.split(",")) if isinstance(ek, str) else tuple(ek)
    arch = dict(latent_dim=tc.get("latent_dim", 32),
                encoder_width=tc.get("encoder_width", args.encoder_width),
                vocos_tier=tc.get("vocos_tier", args.tier),
                encoder_blocks=tc.get("encoder_blocks", 3),
                encoder_kernels=ek)
    print(f"[probe] codec arch from ckpt: {arch}  (best_val_r={tc.get('best_val_r','?') or enc_ck.get('best_val_r','?')})")
    codec = build_default_joint(**arch).to(dev).eval()
    enc = enc_ck.get("state_dict", enc_ck) if isinstance(enc_ck, dict) else enc_ck
    dec = dec.get("state_dict", dec) if isinstance(dec, dict) else dec
    me, ue = codec.encoder.load_state_dict(enc, strict=False)
    md, ud = codec.decoder.load_state_dict(dec, strict=False)
    print(f"[probe] encoder miss={len(me)} unexp={len(ue)} | decoder miss={len(md)} unexp={len(ud)}")

    from lamquant.snn.lma_dataset import LmaDataset
    from lamquant.snn.four_state import derive_4state_target, STATE_NAMES
    paths = []
    for r in args.lma_root:
        paths += sorted(Path(r).glob("*/*.lma")) or sorted(Path(r).glob("*.lma"))
    ds = LmaDataset(lma_paths=paths, split="val", split_manifest_path=args.split_manifest)
    rng = np.random.default_rng(1337)
    idxs = rng.permutation(len(ds))[:args.n_windows].tolist()

    # per-window: min-sufficient codec level (per-TIMESTEP R) vs 4-state target mode
    rows = []          # (codec_min_level_mode, target_mode)
    perstep_pairs = [] # (codec_min_level[t], target[t]) for confusion
    R_by_L = {L: [] for L in LEVELS}
    for i in idxs:
        try:
            l3, lab = ds[int(i)]
        except Exception:
            continue
        l3 = np.asarray(l3, dtype=np.float32)
        if l3.shape[0] != 21:
            continue
        lab = np.asarray(lab)
        x = torch.from_numpy(l3).unsqueeze(0).to(dev)
        with torch.no_grad():
            latent = codec.encoder.encode(x, quantize=True)
            recons = {}
            for L in LEVELS:
                r = codec.decoder(fsq_quantize(latent, L)).squeeze(0).cpu().numpy()
                recons[L] = r[..., :L3_T] if r.shape[-1] >= L3_T else r
        # per-timestep R per level (windowed корреляция over the 21 channels at t)
        per_t_R = {}
        for L in LEVELS:
            rc = recons[L]
            num = (l3 * rc).sum(0) - l3.sum(0) * rc.sum(0) / 21
            dl = np.sqrt(((l3**2).sum(0) - l3.sum(0)**2 / 21).clip(1e-9))
            dr = np.sqrt(((rc**2).sum(0) - rc.sum(0)**2 / 21).clip(1e-9))
            per_t_R[L] = (num / (dl * dr + 1e-9)).clip(-1, 1)
            R_by_L[L].append(float(np.nanmean(per_t_R[L])))
        # min-sufficient level per timestep: lowest L with R >= rel_tol * R@L5
        thr = args.rel_tol * per_t_R[5]
        min_lvl = np.full(per_t_R[5].shape, 5, dtype=np.int64)
        for L in (4, 3, 2):
            min_lvl[per_t_R[L] >= thr] = L
        codec_state = min_lvl - 2          # level 2->state0 ... level5->state3
        tgt = derive_4state_target(lab, l3, args.quiet_thr)
        # align lengths
        T = min(len(codec_state), len(tgt))
        perstep_pairs.append((codec_state[:T], tgt[:T]))
        rows.append((int(np.bincount(codec_state[:T]).argmax()),
                     int(np.bincount(tgt[:T]).argmax())))

    if not perstep_pairs:
        print("[probe] no windows"); return
    cs = np.concatenate([a for a, _ in perstep_pairs])
    tg = np.concatenate([b for _, b in perstep_pairs])
    print(f"\n[probe] windows={len(rows)} timesteps={cs.size}")
    print("[probe] mean R by FSQ level (the difficulty curve):")
    for L in LEVELS:
        print(f"   L={L} (CR {[525,134,82,63][L-2]}): R={np.mean(R_by_L[L]):.4f}")
    print(f"\n[probe] codec-min-level state distribution: {np.bincount(cs, minlength=4).tolist()} {STATE_NAMES}")
    print(f"[probe] current 4-state target distribution:  {np.bincount(tg, minlength=4).tolist()}")
    # agreement
    agree = float((cs == tg).mean())
    print(f"\n[probe] per-timestep agreement codec-vs-target: {agree:.3f}")
    print("[probe] confusion (rows=codec-min-level state, cols=4-state target):")
    cm = np.zeros((4, 4), dtype=np.int64)
    for c, t in zip(cs, tg):
        cm[c, t] += 1
    for r in range(4):
        print("   " + STATE_NAMES[r][:4] + " " + " ".join(f"{cm[r,c]:7d}" for c in range(4)))
    print("\n[probe] READ: low agreement = the codec measures something DIFFERENT from "
          "the energy+seizure heuristic (the reframe's premise). Check if codec-min-level "
          "tracks signal complexity while the target tracks seizure annotation.")


if __name__ == "__main__":
    main()
