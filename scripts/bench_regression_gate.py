#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail when a Criterion comparison shows a confident slowdown.

`cargo bench -- --baseline NAME` prints "Performance has regressed" and still
exits 0, so on its own it is not a gate. This reads every
`target/criterion/**/change/estimates.json` written by that comparison and
fails when the MEDIAN change's lower confidence bound is above the threshold:
the benchmark is slower by more than the threshold even in the most favourable
reading of its samples. Medians resist the outliers a busy host produces.

Usage: bench_regression_gate.py CRITERION_DIR MAX_REGRESSION_FRACTION
"""

import glob
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    root, limit = sys.argv[1], float(sys.argv[2])
    files = sorted(glob.glob(os.path.join(root, "**", "change", "estimates.json"), recursive=True))
    if not files:
        print(f"no Criterion change estimates under {root}: nothing was compared", file=sys.stderr)
        return 2
    failed = []
    for path in files:
        bench = os.path.relpath(os.path.dirname(os.path.dirname(path)), root)
        median = json.load(open(path))["median"]
        point = median["point_estimate"]
        lower = median["confidence_interval"]["lower_bound"]
        verdict = "REGRESSED" if lower > limit else "ok"
        print(f"{verdict:9} {point:+8.1%}  (lower bound {lower:+.1%})  {bench}")
        if lower > limit:
            failed.append(bench)
    print(f"{len(files)} benchmarks compared; limit +{limit:.0%} on the median's lower bound")
    if failed:
        print(f"{len(failed)} regressed beyond the limit: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
