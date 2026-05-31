#!/usr/bin/env python3
"""Clinical event-level seizure FPR/h eval for the Mamba SNN.

Runs a trained checkpoint over the VAL split, ONE RECORDING AT A TIME,
collects the seizure-head probability stream + ground-truth seizure mask
per recording, then calls
:func:`lamquant.snn.event_scoring.calibrate_event_operating_point` to find
the clinical operating point (event-sensitivity, event-FPR/h, threshold +
post-processing params).

WHY PER-RECORDING
-----------------
Seizure EVENTS must be extracted within a contiguous recording. If the
val probabilities were flattened across recordings, an event could "span"
the boundary between two unrelated EEGs — an artifact. So this driver
reconstructs each recording's contiguous L3 timeline by grouping the
``LmaDataset`` window index by ``stem`` and concatenating the windows in
``win_idx`` order.

REUSE, DON'T REINVENT
---------------------
The model class (``MambaSNN``), the dataset (``LmaDataset``), and the
geometry constants are imported from the training module / dataset module
that the trainer already uses, so this eval can never drift from training:
  - model:   lamquant_neural.models.mamba_ssm_minimal.MambaSNN
  - dataset: lamquant.snn.lma_dataset.LmaDataset
  - L3 geometry: lamquant.snn.lma_dataset.{L3_T, LABEL_PER_WINDOW}

The seizure target per timestep matches the trainer's ``validate()``:
``(labels == 2).amax(dim=0)`` — seizure if ANY of the 8 groups is class 2.

NOTE: this module is import-clean and py_compile-safe with no checkpoint
present. ``main()`` only touches torch / the model / the dataset when
actually invoked with ``--checkpoint``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Package root (blut/python). Mirror train_mamba_snn.py's sys.path setup so
# the bare cross-area imports (snn_training_config, lma_dataset) resolve the
# same way whether run as a module or a script.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _sub in ("snn", "dataset", "common"):
    _p = os.path.join(ROOT_DIR, "lamquant", _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from event_scoring import (  # noqa: E402  (after sys.path setup)
    SEC_PER_STEP_L3,
    calibrate_event_operating_point,
)


def _build_recording_seqs(
    model,
    dataset,
    device,
    batch_size: int = 32,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Run ``model`` over ``dataset`` grouped by recording.

    Returns a list of per-recording ``(probs_1d, target_mask_1d)`` tuples.

    The dataset's ``index`` is a list of
    ``(lma_path, stem, win_idx, lml_internal, label_internal)`` tuples.
    Windows are grouped by ``stem`` and ordered by ``win_idx`` so each
    recording's L3 timeline is contiguous. For each window we run the model
    once, take ``sigmoid(seizure_logits)`` (the dedicated seizure head) as
    the per-timestep probability, and ``(labels == 2).any(group axis)`` as
    the per-timestep seizure target, then concatenate the windows of a
    recording end-to-end.
    """
    import torch

    # Group dataset row indices by stem, preserving win_idx order.
    by_stem: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for row_idx, entry in enumerate(dataset.index):
        # entry: (lma_path, stem, win_idx, lml_internal, label_internal)
        stem = entry[1]
        win_idx = entry[2]
        by_stem[stem].append((win_idx, row_idx))

    model.eval()
    seqs: List[Tuple[np.ndarray, np.ndarray]] = []

    with torch.no_grad():
        for stem, pairs in by_stem.items():
            pairs.sort(key=lambda wp: wp[0])  # contiguous by win_idx
            row_indices = [row_idx for _wi, row_idx in pairs]

            probs_chunks: List[np.ndarray] = []
            target_chunks: List[np.ndarray] = []

            # Mini-batch the recording's windows through the model.
            for b in range(0, len(row_indices), batch_size):
                batch_rows = row_indices[b:b + batch_size]
                signals = []
                labels = []
                for r in batch_rows:
                    sig, lab = dataset[r]          # ([21, L3_T], [8, L3_T])
                    signals.append(sig)
                    labels.append(lab)
                signal = torch.stack(signals).to(device)
                label = torch.stack(labels).to(device)

                _activity, _spike, seizure_logits = model(signal)
                # seizure_logits: [B, 1, T]. Per-timestep seizure probability.
                probs = torch.sigmoid(seizure_logits)[:, 0, :]   # [B, T]
                # Per-timestep seizure target: ANY group == 2 (matches trainer
                # validate()). label: [B, 8, T] -> [B, T].
                target = (label == 2).any(dim=1)                 # [B, T] bool

                # Flatten each window into the recording timeline, in order.
                for bi in range(probs.shape[0]):
                    probs_chunks.append(probs[bi].cpu().numpy())
                    target_chunks.append(
                        target[bi].cpu().numpy().astype(bool))

            if probs_chunks:
                rec_probs = np.concatenate(probs_chunks)
                rec_target = np.concatenate(target_chunks)
                seqs.append((rec_probs, rec_target))

    return seqs


def _load_model(checkpoint_path: Path, subband: bool, device):
    """Construct MambaSNN with the checkpoint's config and load weights."""
    import torch
    from lamquant_neural.models.mamba_ssm_minimal import MambaSNN

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    d_model = int(cfg.get("d_model", 32))
    d_state = int(cfg.get("d_state", 16))
    n_layers = int(cfg.get("n_layers", 2))

    model = MambaSNN(
        in_channels=21, d_model=d_model, d_state=d_state,
        n_layers=n_layers, use_subband=subband,
    ).to(device)
    # Non-strict so a checkpoint that predates a head still loads its core.
    model.load_state_dict(ckpt["model"], strict=False)
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clinical event-level seizure FPR/h eval for the Mamba SNN")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained MambaSNN checkpoint (.pt)")
    parser.add_argument("--split-manifest", type=Path, required=True,
                        help="Subject-grouped split manifest JSON "
                             "(from build_snn_train_val_split.py)")
    parser.add_argument("--lma-root", type=Path, required=True, nargs="+",
                        help="One or more LMA roots (per-corpus .lma files "
                             "and/or directories of .lma). Unioned, so per-corpus "
                             "source-of-truth archives and legacy per-recording "
                             "dirs can be mixed — must cover the val split.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sens-floor", type=float, default=0.85,
                        help="Required event-sensitivity (PCCP floor 0.85)")
    parser.add_argument("--spec-target", type=float, default=0.90,
                        help="Clinical time-specificity target (default 0.90). "
                             "Reported as OK/LOW; does not gate the operating "
                             "point (selection still minimizes FPR/h).")
    parser.add_argument("--sec-per-step", type=float, default=SEC_PER_STEP_L3,
                        help="Seconds per L3 timestep (default 10/313)")
    parser.add_argument("--max-windows-per-file", type=int, default=100000,
                        help="Per-recording window cap. Event-level FPR/h needs "
                             "FULL recordings, so the default is effectively "
                             "uncapped (100000). The 5-window training cap would "
                             "truncate each recording to ~50 s and produce a "
                             "meaningless FPR/h denominator — do not lower this "
                             "for a clinical eval.")
    args = parser.parse_args()

    import torch
    from lma_dataset import LmaDataset

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[*] Event-FPR eval on {device}")
    print(f"[*] checkpoint   = {args.checkpoint}")
    print(f"[*] split-manifest = {args.split_manifest}")
    print(f"[*] lma-root     = {args.lma_root}")

    # The LMA-direct path is always L3 subband (stride-1), matching training.
    model, ckpt = _load_model(args.checkpoint, subband=True, device=device)
    print(f"[*] loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"saved sens={ckpt.get('sensitivity', '?')})")

    ds_kwargs: dict = {}
    if args.max_windows_per_file is not None:
        ds_kwargs["max_windows_per_file"] = args.max_windows_per_file
    # Union every root into an explicit .lma list (mirrors train_mamba_snn):
    # a root that is itself a .lma file is used directly; a directory is
    # globbed two-then-one level deep.
    roots = args.lma_root if isinstance(args.lma_root, (list, tuple)) \
        else [args.lma_root]
    lma_paths: list = []
    for r in roots:
        r = Path(r)
        if r.is_file() and r.suffix == ".lma":
            lma_paths.append(r)
            continue
        found = sorted(r.glob("*/*.lma")) or sorted(r.glob("*.lma"))
        if not found:
            raise RuntimeError(f"no .lma archives under {r}")
        lma_paths.extend(found)
    _seen: set = set()
    lma_paths = [p for p in lma_paths if not (str(p) in _seen or _seen.add(str(p)))]
    print(f"[*] {len(roots)} root(s) -> {len(lma_paths)} .lma archive(s)")
    val_ds = LmaDataset(
        lma_paths=lma_paths, split="val",
        split_manifest_path=args.split_manifest,
        **ds_kwargs,
    )
    print(f"[*] val windows = {len(val_ds)}")

    seqs = _build_recording_seqs(
        model, val_ds, device, batch_size=args.batch_size)
    n_rec = len(seqs)
    total_steps = sum(len(p) for p, _t in seqs)
    total_seconds = total_steps * args.sec_per_step
    print(f"[*] reconstructed {n_rec} recordings "
          f"({total_steps} timesteps, {total_seconds/3600:.2f} h total)")
    if total_seconds < 3600.0:
        print(f"[!] WARNING: only {total_seconds/3600:.2f} h of recording "
              f"reconstructed — FPR/h denominator is tiny and the rate will be "
              f"unreliable. This usually means --max-windows-per-file is capping "
              f"each recording (event-level eval needs FULL recordings). "
              f"Current cap = {args.max_windows_per_file}.")

    op = calibrate_event_operating_point(
        seqs, sens_floor=args.sens_floor, sec_per_step=args.sec_per_step)

    time_spec = float(op.get("time_specificity", float("nan")))
    step_spec = float(op.get("timestep_specificity", float("nan")))
    print("\n[*] Clinical operating point (NEDC OVLP, event-level):")
    print(f"    threshold       = {op['threshold']:.3f}")
    print(f"    min_event_sec   = {op['min_event_sec']:.1f}")
    print(f"    merge_gap_sec   = {op['merge_gap_sec']:.1f}")
    print(f"    refractory_sec  = {op['refractory_sec']:.1f}")
    print(f"    event_sens      = {op['event_sens']:.4f}")
    print(f"    time_spec       = {time_spec:.4f}  "
          f"(non-seizure time left un-flagged; clinical specificity)")
    print(f"    timestep_spec   = {step_spec:.4f}  (raw per-step, pre-postproc)")
    print(f"    event_FPR/h     = {op['event_fpr_per_h']:.4f}")
    print(f"    meets_floor     = {op['meets_floor']} "
          f"(sens_floor={args.sens_floor})")
    # Clinical target: sens ~1.0 AND specificity >= 0.90.
    sens_ok = op["event_sens"] >= args.sens_floor
    spec_ok = time_spec >= args.spec_target
    print(f"    TARGET sens~1.0 & spec>={args.spec_target:.2f}: "
          f"sens={'OK' if sens_ok else 'LOW'} "
          f"time_spec={'OK' if spec_ok else 'LOW'}")


if __name__ == "__main__":
    main()
