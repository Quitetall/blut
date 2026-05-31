#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Discrimination diagnostic: the sens-vs-time_spec tradeoff curve for ONE
checkpoint on a val split. Answers the load-bearing question for the 100/90
goal — is a (sens~1.0, time_spec>=0.90) operating point REACHABLE with these
weights (a calibration problem), or does the seizure head fundamentally fail to
separate seizure from non-seizure time (must retrain)?

For each threshold, with the best post-processing per threshold (min_event /
merge / refractory swept), reports pooled event-sensitivity and time-specificity
so the full Pareto frontier is visible. Also prints the seizure-head probability
distribution on true-negative vs true-positive timesteps — the raw separability.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _sub in ("snn", "dataset", "common"):
    _p = os.path.join(ROOT_DIR, "lamquant", _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from event_scoring import (  # noqa: E402
    SEC_PER_STEP_L3, events_from_probs, ovlp_score, events_from_binary,
    _predicted_positive_mask,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--lma-root", nargs="+", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sec-per-step", type=float, default=SEC_PER_STEP_L3)
    args = ap.parse_args()

    import torch
    from eval_event_fpr import _build_recording_seqs, _load_model
    from lma_dataset import LmaDataset

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else (args.device if args.device != "auto" else "cpu"))
    model, ckpt = _load_model(__import__("pathlib").Path(args.checkpoint), True, dev)
    print(f"[*] {args.checkpoint} (epoch {ckpt.get('epoch','?')})")

    roots = []
    for r in args.lma_root:
        from pathlib import Path
        r = Path(r)
        if r.is_file() and r.suffix == ".lma":
            roots.append(r); continue
        roots += sorted(r.glob("*/*.lma")) or sorted(r.glob("*.lma"))
    seen = set(); roots = [p for p in roots if not (str(p) in seen or seen.add(str(p)))]
    ds = LmaDataset(lma_paths=roots, split="val",
                    split_manifest_path=args.split_manifest,
                    max_windows_per_file=100000)
    seqs = _build_recording_seqs(model, ds, dev, batch_size=32)
    print(f"[*] {len(seqs)} recordings")

    # Pre-extract true events + total / negative time.
    sec = args.sec_per_step
    true_ev = []
    total_steps = 0
    neg_total = 0
    pos_probs, neg_probs = [], []
    for p, t in seqs:
        true_ev.append(events_from_binary(t, sec_per_step=sec, min_event_sec=0.0))
        total_steps += len(p)
        neg = (t == 0)
        neg_total += int(neg.sum())
        pos_probs.append(p[~neg]); neg_probs.append(p[neg])
    total_h = total_steps * sec / 3600.0
    pos_p = np.concatenate([x for x in pos_probs if len(x)]) if any(len(x) for x in pos_probs) else np.array([])
    neg_p = np.concatenate([x for x in neg_probs if len(x)])
    print(f"[*] {total_h:.1f}h, neg-time={neg_total*sec/3600:.1f}h")
    if len(pos_p):
        print(f"[*] seizure-head prob  TRUE-POS timesteps: "
              f"mean={pos_p.mean():.3f} p10={np.percentile(pos_p,10):.3f} "
              f"p50={np.percentile(pos_p,50):.3f}")
    print(f"[*] seizure-head prob  TRUE-NEG timesteps: "
          f"mean={neg_p.mean():.3f} p50={np.percentile(neg_p,50):.3f} "
          f"p90={np.percentile(neg_p,90):.3f} p99={np.percentile(neg_p,99):.3f}")

    print(f"\n{'thr':>6}{'sens':>8}{'time_spec':>11}{'FPR/h':>9}  (best post-proc/thr)")
    best_spec_at_full_sens = 0.0
    best_sens_at_90_spec = 0.0
    for thr in np.linspace(0.05, 0.98, 32):
        best = None
        for min_ev in (1.0, 2.0, 5.0, 10.0):
            for merge in (0.0, 2.0, 5.0):
                for refr in (0.0, 5.0):
                    pt = pd = pf = fp_time = 0
                    for (p, t), te in zip(seqs, true_ev):
                        pe = events_from_probs(p, threshold=float(thr), sec_per_step=sec,
                                               min_event_sec=min_ev, merge_gap_sec=merge,
                                               refractory_sec=refr)
                        sc = ovlp_score(pe, te)
                        pt += sc["n_true"]; pd += sc["n_true_detected"]; pf += sc["n_false_pred"]
                        neg = (t == 0)
                        pm = _predicted_positive_mask(pe, len(p), sec)
                        fp_time += int((pm & neg).sum())
                    sens = pd / pt if pt else 0.0
                    tspec = 1.0 - fp_time / neg_total if neg_total else 1.0
                    fpr = pf * 3600.0 / (total_steps * sec)
                    # best post-proc at this thr = max time_spec among those
                    # keeping the HIGHEST sensitivity.
                    if best is None or (sens, tspec) > (best[0], best[1]):
                        best = (sens, tspec, fpr)
        sens, tspec, fpr = best
        print(f"{thr:>6.2f}{sens:>8.3f}{tspec:>11.3f}{fpr:>9.2f}")
        if sens >= 0.99:
            best_spec_at_full_sens = max(best_spec_at_full_sens, tspec)
        if tspec >= 0.90:
            best_sens_at_90_spec = max(best_sens_at_90_spec, sens)

    print(f"\n=== VERDICT ===")
    print(f"max time_spec @ sens>=0.99 : {best_spec_at_full_sens:.3f}  (target 0.90)")
    print(f"max sens     @ time_spec>=0.90: {best_sens_at_90_spec:.3f}  (target ~1.0)")
    if best_spec_at_full_sens >= 0.90:
        print("=> REACHABLE with these weights — a calibration/op-selection fix.")
    elif best_sens_at_90_spec >= 0.95:
        print("=> 90% spec reachable but only by dropping some sensitivity — "
              "tradeoff/threshold question.")
    else:
        print("=> NOT reachable — fundamental discrimination gap. Must retrain "
              "(loss/sampler rebalance, arch capacity, more data).")


if __name__ == "__main__":
    main()
