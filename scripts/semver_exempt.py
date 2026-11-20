#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decide whether a failed cargo-semver-checks run consists only of recorded,
reviewed exemptions.

Usage: semver_exempt.py CRATE BASELINE LOG EXEMPTIONS_TSV
Exit 0 when the log has at least one finding and every finding
(lint, item) is listed for CRATE at BASELINE; exit 1 otherwise, naming
each finding that is not exempt. A log with no parseable finding is never
accepted: a failure we cannot read is a failure.
"""

import re
import sys


def findings(log_text):
    """(lint, item) pairs, and the lints whose items could not be read."""
    lint = None
    found = []
    per_lint = {}
    for line in log_text.splitlines():
        header = re.match(r"^--- failure ([a-z0-9_]+):", line)
        if header:
            lint = header.group(1)
            per_lint.setdefault(lint, 0)
            continue
        # "  enum X in path:1" or, for removed items,
        # "  method X, previously in file path:1"
        item = re.match(r"^  (\S.*?)(?:,\s+previously)? in (?:file )?\S+:\d+\s*$", line)
        if lint and item:
            found.append((lint, item.group(1)))
            per_lint[lint] += 1
        elif line.startswith("--- ") or line.startswith("     Summary"):
            lint = None
    unreadable = [l for l, n in per_lint.items() if n == 0]
    return found, unreadable


def main():
    if len(sys.argv) != 5:
        print(__doc__.strip().splitlines()[3], file=sys.stderr)
        return 2
    crate, baseline, log_path, tsv = sys.argv[1:]
    allowed = set()
    for row in open(tsv, encoding="utf-8"):
        if row.startswith("#") or not row.strip():
            continue
        c, b, lint, item = row.rstrip("\n").split("\t")
        if c == crate and b == baseline:
            allowed.add((lint, item))
    found, unreadable = findings(open(log_path, encoding="utf-8", errors="replace").read())
    if unreadable:
        print(f"semver_exempt: could not read the findings of {', '.join(unreadable)}; "
              "not accepting", file=sys.stderr)
        return 1
    if not found:
        print(f"semver_exempt: no findings could be read for {crate}; not accepting", file=sys.stderr)
        return 1
    unexempt = [f for f in found if f not in allowed]
    for lint, item in found:
        print(f"semver_exempt: {crate} {lint}: {item}: {'exempt' if (lint, item) not in unexempt else 'NOT EXEMPT'}")
    return 1 if unexempt else 0


if __name__ == "__main__":
    sys.exit(main())
