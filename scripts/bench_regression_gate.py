#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail when a Criterion comparison shows a confident slowdown.

`cargo bench -- --baseline NAME` prints "Performance has regressed" and still
exits 0, so on its own it is not a gate. This reads every
`target/criterion/**/change/estimates.json` written by that comparison and
flags a benchmark when the MEDIAN change's lower confidence bound is above the
threshold: it is slower by more than the threshold even in the most favourable
reading of its samples. Medians resist the outliers a busy host produces.

One run is not enough to fail on. On a whole idle machine, unchanged code put
one benchmark 17% over its baseline in one run and level with it in the next.
So a comparison is run in two passes:

  first pass   --suspects-out FILE   writes the flagged benchmarks to FILE and
                                     exits 1 if there are any;
  second pass  --confirm FILE        fails only for benchmarks flagged in BOTH
                                     passes.

Without either option a single pass decides, as before.

Usage: bench_regression_gate.py CRITERION_DIR MAX_REGRESSION_FRACTION
           [--suspects-out FILE | --confirm FILE]
"""

import argparse
import glob
import json
import os
import sys


def flagged(root: str, limit: float) -> tuple[list[tuple[str, float, float]], list[str]]:
    """Every compared benchmark as (name, median change, lower bound), and the
    names whose lower bound is above `limit`."""
    rows, over = [], []
    pattern = os.path.join(glob.escape(root), "**", "change", "estimates.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        bench = os.path.relpath(os.path.dirname(os.path.dirname(path)), root)
        with open(path, encoding="utf-8") as f:
            median = json.load(f)["median"]
        lower = median["confidence_interval"]["lower_bound"]
        rows.append((bench, median["point_estimate"], lower))
        if lower > limit:
            over.append(bench)
    return rows, over


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(usage=__doc__.strip().split("Usage: ")[-1])
    parser.add_argument("criterion_dir")
    parser.add_argument("max_regression", type=float)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--suspects-out")
    mode.add_argument("--confirm")
    args = parser.parse_args(argv)

    rows, over = flagged(args.criterion_dir, args.max_regression)
    if not rows:
        print(
            f"no Criterion change estimates under {args.criterion_dir}: nothing was compared",
            file=sys.stderr,
        )
        return 2

    earlier = None
    if args.confirm:
        with open(args.confirm, encoding="utf-8") as f:
            earlier = {line.rstrip("\n") for line in f if line.strip()}

    failed = []
    for bench, point, lower in rows:
        if bench not in over:
            verdict = "ok"
        elif earlier is not None and bench not in earlier:
            verdict = "unconfirmed"
        else:
            verdict = "REGRESSED"
            failed.append(bench)
        print(f"{verdict:11} {point:+8.1%}  (lower bound {lower:+.1%})  {bench}")
    print(
        f"{len(rows)} benchmarks compared; limit +{args.max_regression:.0%} "
        "on the median's lower bound"
    )

    if args.suspects_out:
        with open(args.suspects_out, "w", encoding="utf-8") as f:
            f.writelines(f"{bench}\n" for bench in failed)
    if earlier is not None:
        cleared = sorted(earlier - set(over))
        if cleared:
            print(f"not confirmed by this pass: {', '.join(cleared)}")
    if failed:
        what = "regressed in both passes" if earlier is not None else "over the limit"
        print(f"{len(failed)} {what}: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
