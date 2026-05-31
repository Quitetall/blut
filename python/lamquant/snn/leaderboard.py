#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rank SNN checkpoints by the weighted clinical cost J.

J = w_fn * (1 - event_sens) + (1 - time_specificity)   (lower = better)

Both terms are fractions; w_fn (default 100) weights a missed seizure ~w_fn x a
unit of false-positive TIME. (1 - time_spec) is used, NOT event-FPR/h, because
FPR/h is gameable by one long flooding event. Parses the per-model event-eval
logs from eval_event_fpr.py. Logs that predate time_spec fall back to FPR/h and
are flagged as not-comparable (their J is on a different scale).
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

    print(f"=== Leaderboard (J = {args.fn_weight:g}*FNR + (1-time_spec), "
          f"lower=better) ===")
    print(f"{'rank':<5}{'model':<22}{'sens':>7}{'FNR':>8}{'FPR/h':>9}"
          f"{'time_spec':>11}{'hours':>8}{'J':>10}  note")
    for i, (name, d) in enumerate(done, 1):
        ts = d.get("time_spec", float("nan"))
        hr = d.get("hours", float("nan"))
        note = d.get("J_note", "")
        print(f"{i:<5}{name:<22}{d['event_sens']:>7.4f}{d['fnr']:>8.4f}"
              f"{d['fpr']:>9.3f}{ts:>11.4f}{hr:>8.1f}{d['J']:>10.3f}  {note}")
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
