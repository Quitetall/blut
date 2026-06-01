#!/usr/bin/env python3
"""Oracle ceiling probe — does >15 Hz detail-band content carry recoverable
4-state (esp. CRITICAL/seizure) discrimination signal that L3 alone lacks?

This is the CHEAP, NO-TRAINING gate for the L3-vs-detail-bands question
(ADR 0027 revisit lever / ADR 0029 input study). It fits a high-capacity,
unconstrained classifier (HistGradientBoosting) per-timestep on each INPUT
REGIME and reports the per-class recall CEILING. It measures "is the signal
IN the input", isolating that from "can the 57K SSM extract it".

Regimes (channel subsets of the SNN_DETAIL_BANDS-augmented stack):
  L3            = ch[0:21]   <=15.6 Hz (current SNN input)
  L3+l3_detail  = ch[0:42]   + 15.6-31.25 Hz (LVFA band; the DEPLOYABLE arm)
  L3+all        = ch[0:84]   + l3/l2/l1 detail (>15 Hz off-device ceiling)
  L3+spectral   = L3 + spectral band-powers (the existing --spectral arm, the
                  REAL baseline to beat — verifier-mandated comparison)

Decision rule: if the CRITICAL-recall ceiling does NOT rise from L3 -> L3+detail
(and not over L3+spectral), the input is NOT the lever and the SSM A/B is not
worth GPU. If it rises, green-light the L3+l3_detail SSM training arm.

CAVEAT (printed in the report): the split is window-level random, so absolute
ceilings are OPTIMISTIC (intra-subject leakage). The valid signal is the
DELTA between regimes — all regimes share the identical split, so the
comparison is fair even though the absolute number is inflated.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Detail bands must be captured by the dataset. Set BEFORE the dataset reads it
# (it reads os.environ at call time, so setting here is sufficient).
os.environ.setdefault("SNN_DETAIL_BANDS", "l3_detail,l2_detail,l1_detail")

from lamquant.snn.lma_dataset import LmaDataset, _detail_bands_cfg  # noqa: E402
from lamquant.snn.four_state import (  # noqa: E402
    derive_4state_target, calibrate_quiet_threshold, STATE_NAMES,
)
from lamquant.snn.train_4state_controller import _pool_states_to_T  # noqa: E402

L3_CH = 21
L3_T = 313


def contextize(win_chT: np.ndarray, ctx: int) -> np.ndarray:
    """[C, T] -> [T, C*(2*ctx+1)] with edge-replicated context (no cross-window
    bleed: called per window)."""
    C, T = win_chT.shape
    if ctx == 0:
        return win_chT.T.copy()
    padded = np.pad(win_chT, ((0, 0), (ctx, ctx)), mode="edge")  # [C, T+2ctx]
    frames = [padded[:, i:i + T] for i in range(2 * ctx + 1)]    # 2ctx+1 x [C,T]
    stacked = np.stack(frames, axis=2)                            # [C, T, 2ctx+1]
    return stacked.transpose(1, 0, 2).reshape(T, C * (2 * ctx + 1))


def per_class_recall(y_true, y_pred, n_states=4):
    rec = np.zeros(n_states)
    for k in range(n_states):
        m = (y_true == k)
        rec[k] = float((y_pred[m] == k).mean()) if m.any() else float("nan")
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lma-root", type=Path, nargs="+", required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--max-windows", type=int, default=4000,
                   help="cap windows sampled (each ~313 timesteps)")
    p.add_argument("--ctx", type=int, default=4, help="+/- temporal context frames")
    p.add_argument("--quiet-thr", type=float, default=3.57,
                   help="QUIET/BASELINE RMS cut (stable ~3.57 across runs; only "
                        "affects QUIET vs BASELINE, NOT CRITICAL recall). Avoids "
                        "re-calibration, which needs bare 21-ch L3.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-iter", type=int, default=300, help="HistGBM boosting rounds")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    bands = _detail_bands_cfg()
    exp_ch = L3_CH * (1 + len(bands))
    print(f"[oracle] detail bands = {bands}; expected stack channels = {exp_ch}")

    ds = LmaDataset(lma_dir=None, lma_paths=sum(
        ([*Path(r).glob('*/*.lma')] or [*Path(r).glob('*.lma')] for r in args.lma_root), []),
        split=args.split, split_manifest_path=args.split_manifest)
    print(f"[oracle] {args.split} dataset: {len(ds)} windows")

    thr = args.quiet_thr
    print(f"[oracle] quiet_rms_threshold = {thr:.6g} (fixed; CRITICAL recall is "
          f"quiet-thr-invariant)")

    # ---- Collect flat per-timestep aug values + targets (parallel decode). ----
    import torch
    from torch.utils.data import Subset, DataLoader
    n = min(args.max_windows, len(ds))
    idxs = rng.permutation(len(ds))[:n].tolist()
    sub = Subset(ds, [int(i) for i in idxs])
    loader = DataLoader(sub, batch_size=64, num_workers=8,
                        collate_fn=lambda b: b)  # list of (sig, labels)
    aug_rows, tgt_rows = [], []
    t0 = time.time()
    kept = 0
    for batch in loader:
        for sig, labels in batch:
            sig = np.asarray(sig, dtype=np.float32)
            labels = np.asarray(labels)
            if sig.shape[0] != exp_ch:
                continue
            tgt = derive_4state_target(labels, sig[:L3_CH], thr)
            tgt = _pool_states_to_T(tgt, L3_T).astype(np.int8)
            aug_rows.append(sig.T.astype(np.float16))   # [313, 84]
            tgt_rows.append(tgt)                         # [313]
            kept += 1
        if kept and kept % 512 < 64:
            print(f"[oracle]   collected {kept} windows ({time.time()-t0:.0f}s)", flush=True)
    if kept == 0:
        print("[oracle] FATAL: no windows collected"); sys.exit(1)

    aug = np.concatenate(aug_rows, axis=0)   # [N*313, 84]
    tgt = np.concatenate(tgt_rows, axis=0)   # [N*313]
    win_of_row = np.repeat(np.arange(kept), L3_T)
    print(f"[oracle] collected {kept} windows -> {aug.shape[0]} timesteps, "
          f"class counts = {np.bincount(tgt, minlength=4).tolist()} "
          f"({STATE_NAMES})")

    crit_frac = (tgt == 3).mean()
    if crit_frac < 1e-4:
        print(f"[oracle] WARNING: CRITICAL fraction {crit_frac:.5f} tiny — "
              f"recall estimate noisy. Increase --max-windows or seizure data.")

    # ---- Window-level 70/30 split (shared across all regimes). ----
    win_perm = rng.permutation(kept)
    n_tr = int(0.7 * kept)
    tr_wins = set(win_perm[:n_tr].tolist())
    tr_mask = np.array([w in tr_wins for w in win_of_row])

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    regimes = {
        "L3 (<=15.6Hz)":        list(range(0, 21)),
        "L3+l3_detail (LVFA)":  list(range(0, 42)),
        "L3+all_details":       list(range(0, 84)),
    }
    # Spectral arm: derive from L3 channels (the existing --spectral features).
    spectral_cols = None
    try:
        from lamquant.snn.spectral import l3_spectral_features_np

        def _spectral_aug(a_chT):  # [21,313] -> [105,313]
            feats = l3_spectral_features_np(a_chT[None])[0]  # [84,313]
            return np.concatenate([a_chT, feats], axis=0)
        spectral_cols = _spectral_aug
    except Exception as e:
        print(f"[oracle] spectral arm unavailable ({e}); skipping")

    def build_X(channels, spectral=False):
        """Contextized features per window for the given channel subset."""
        Xs = []
        for w in range(kept):
            a = aug[w * L3_T:(w + 1) * L3_T].T.astype(np.float32)  # [84,313]
            if spectral:
                a = spectral_cols(a[:L3_CH])                        # [105,313]
                sub = a
            else:
                sub = a[channels]                                   # [Cs,313]
            Xs.append(contextize(sub, args.ctx))                    # [313, Cs*(2ctx+1)]
        return np.concatenate(Xs, axis=0)

    report = {"meta": {"windows": kept, "timesteps": int(aug.shape[0]),
                       "ctx": args.ctx, "bands": list(bands),
                       "class_counts": np.bincount(tgt, minlength=4).tolist(),
                       "split": "window-level 70/30 (absolute optimistic; "
                                "regime DELTA is the valid signal)"}}
    arms = list(regimes.items())
    if spectral_cols is not None:
        arms.append(("L3+spectral (baseline)", None))

    print("\n[oracle] === per-regime ceiling (HistGBM, per-timestep) ===")
    hdr = f"{'regime':24s} {'CRIT_rec':>9s} {'INT_rec':>8s} {'BASE_rec':>9s} {'QUIET_rec':>10s} {'macro':>7s}"
    print(hdr); print("-" * len(hdr))
    for name, chans in arms:
        spectral = chans is None
        Xf = build_X(chans if chans else [], spectral=spectral)
        Xtr, ytr = Xf[tr_mask], tgt[tr_mask]
        Xte, yte = Xf[~tr_mask], tgt[~tr_mask]
        sw = compute_sample_weight("balanced", ytr)
        clf = HistGradientBoostingClassifier(
            max_iter=args.max_iter, learning_rate=0.1, max_depth=None,
            l2_regularization=1.0, random_state=args.seed)
        clf.fit(Xtr, ytr, sample_weight=sw)
        pred = clf.predict(Xte)
        rec = per_class_recall(yte, pred)
        macro = float(np.nanmean(rec))
        report.setdefault("regimes", {})[name] = {
            "n_features": int(Xf.shape[1]),
            "recall": {STATE_NAMES[k]: (None if np.isnan(rec[k]) else round(float(rec[k]), 4))
                       for k in range(4)},
            "macro_recall": round(macro, 4)}
        print(f"{name:24s} {rec[3]:9.3f} {rec[2]:8.3f} {rec[1]:9.3f} {rec[0]:10.3f} {macro:7.3f}")

    # Verdict.
    rg = report["regimes"]
    base = rg["L3 (<=15.6Hz)"]["recall"]["CRITICAL"] or 0.0
    lvfa = rg["L3+l3_detail (LVFA)"]["recall"]["CRITICAL"] or 0.0
    allb = rg["L3+all_details"]["recall"]["CRITICAL"] or 0.0
    d_lvfa, d_all = lvfa - base, allb - base
    print(f"\n[oracle] CRIT-recall ceiling: L3={base:.3f}  "
          f"+l3_detail={lvfa:.3f} (Δ{d_lvfa:+.3f})  +all={allb:.3f} (Δ{d_all:+.3f})")
    if "L3+spectral (baseline)" in rg:
        sp = rg["L3+spectral (baseline)"]["recall"]["CRITICAL"] or 0.0
        print(f"[oracle]   vs L3+spectral={sp:.3f} — detail must beat THIS to matter: "
              f"l3_detail {'BEATS' if lvfa > sp else 'does NOT beat'} spectral")
    verdict = ("GREEN: >15Hz detail bands raise the CRIT ceiling -> train the SSM arm"
               if d_lvfa >= 0.02 else
               "RED: detail bands do NOT raise the CRIT ceiling -> input is not the lever, "
               "L3(+spectral) stands, do not spend GPU on the SSM arm")
    report["verdict"] = verdict
    print(f"\n[oracle] VERDICT: {verdict}")
    print("[oracle] NOTE: window-level split — absolute ceilings optimistic; "
          "the Δ between regimes is the decision signal.")

    if args.out:
        import json
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[oracle] wrote {args.out}")


if __name__ == "__main__":
    main()
