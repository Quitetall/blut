#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rank SNN checkpoints by the weighted clinical cost J.

J = w_fn * (1 - event_sens) + event_FPR_per_h   (lower = better)

w_fn is how many false-alarms/hour a missed seizure is worth (default 100).
Parses the per-model event-eval logs written by eval_event_fpr.py and prints a
ranked table. Robust to logs that predate the in-eval COST line — it recomputes
J from the reported event_sens + event_FPR/h.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_PATS = {
    "event_sens": re.compile(r"event_sens\s*=\s*([0-9.]+)"),
    "fpr": re.compile(r"event_FPR/h\s*=\s*([0-9.]+)"),
    "time_spec": re.compile(r"time_spec\s*=\s*([0-9.]+)"),
    "hours": re.compile(r"reconstructed\s+\d+\s+recordings\s+\([0-9]+\s+timesteps,\s+([0-9.]+)\s+h"),
}


def _parse(path: Path) -> dict | None:
    txt = path.read_text(errors="replace")
    out: dict = {}
    for k, pat in _PATS.items():
        m = pat.search(txt)
        if m:
            out[k] = float(m.group(1))
    if "event_sens" not in out or "fpr" not in out:
        return None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path,
                    default=Path("/mnt/4tb/data/Training/eval"))
    ap.add_argument("--fn-weight", type=float, default=100.0)
    args = ap.parse_args()

    rows = []
    for log in sorted(args.eval_dir.glob("eval_*.log")):
        name = log.stem[len("eval_"):]
        d = _parse(log)
        if d is None:
            rows.append((name, None))
            continue
        fnr = 1.0 - d["event_sens"]
        # J = w_fn*FNR + (1 - time_spec). Falls back to FPR/h only if time_spec
        # is missing from an older log (then the number is not comparable).
        if "time_spec" in d:
            J = args.fn_weight * fnr + (1.0 - d["time_spec"])
        else:
            J = args.fn_weight * fnr + d["fpr"]
            d["J_note"] = "no time_spec (FPR/h fallback — not comparable)"
        rows.append((name, {**d, "fnr": fnr, "J": J}))

    done = [(n, d) for n, d in rows if d is not None]
    done.sort(key=lambda r: r[1]["J"])

    print(f"=== Leaderboard (J = {args.fn_weight:g}*FNR + FPR/h, lower=better) ===")
    print(f"{'rank':<5}{'model':<22}{'sens':>7}{'FNR':>8}{'FPR/h':>9}"
          f"{'time_spec':>11}{'hours':>8}{'J':>10}")
    for i, (name, d) in enumerate(done, 1):
        ts = d.get("time_spec", float("nan"))
        hr = d.get("hours", float("nan"))
        print(f"{i:<5}{name:<22}{d['event_sens']:>7.4f}{d['fnr']:>8.4f}"
              f"{d['fpr']:>9.3f}{ts:>11.4f}{hr:>8.1f}{d['J']:>10.3f}")
    pending = [n for n, d in rows if d is None]
    if pending:
        print(f"\npending/unparsed: {', '.join(pending)}")
    if done:
        best = done[0]
        print(f"\nBEST by J: {best[0]}  (sens={best[1]['event_sens']:.4f}, "
              f"FPR/h={best[1]['fpr']:.3f}, "
              f"time_spec={best[1].get('time_spec', float('nan')):.4f})")


if __name__ == "__main__":
    main()
