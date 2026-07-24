#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADR 0139 P5 runtime fault-containment producer.\n\nEmits ONE fault-injection receipt as deterministic JSON on stdout. The realm\nunder fault is this file's stem; all fault producers are byte-identical. Each\ndrives an always-failing kernel through the real executor and records that\nthe failure was contained: no step completed and nothing was committed, so a\nfault can never leave a half-applied graph behind."""

import json
import subprocess
import sys
from pathlib import Path

PRODUCER_CONTRACT = "runtime-fault"

_CASES = {
    "host-stream-fault": "host-stream",
    "mcu-aot-fault": "mcu-aot",
}
_SCHEMA = "lamquant.adr0139.runtime-fault-receipt/v1"


def _execute(realm, inject_fault):
    """Run the real graph runtime for one realm and return its evidence."""
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "blut-graph-core",
        "--example",
        "runtime_execution_probe",
        "--",
        realm,
    ]
    if inject_fault:
        command.append("--inject-fault")
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False
    )
    if completed.returncode != 0:
        raise SystemExit(f"runtime execution failed for {realm}")
    return json.loads(completed.stdout)


def produce_evidence():
    """Inject a kernel fault on this realm and return its containment receipt."""
    case = Path(__file__).stem
    if case not in _CASES:
        raise SystemExit(f"unknown case: {case}")
    realm = _CASES[case]
    measured = _execute(realm, True)
    contained = measured["contained"] is True and measured["completed_steps"] == 0
    receipt = {}
    receipt["schema"] = _SCHEMA
    receipt["case_id"] = case
    receipt["realm"] = measured["realm"]
    receipt["status"] = "pass" if contained else "fail"
    receipt["contained"] = contained
    receipt["completed_steps"] = measured["completed_steps"]
    receipt["terminal_values"] = measured["terminal_values"]
    return receipt


def main():
    rendered = json.dumps(produce_evidence(), indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
