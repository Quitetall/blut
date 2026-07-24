#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADR 0139 P5 runtime graph-execution producer.\n\nEmits ONE executed-graph receipt as deterministic JSON on stdout. The\nexecution realm is this file's stem; all realm producers are byte-identical.\nEach compiles the conformance graph for its realm and runs it there, then\nrecords whether ordered step evidence survived, whether declared policy\npropagated into the compiled plan, whether transactional receipts stayed\nintact, and whether any step exceeded its declared resource envelope."""

import json
import subprocess
import sys
from pathlib import Path

PRODUCER_CONTRACT = "runtime-graph"

_CASES = {
    "host-stream": "host-stream",
    "mcu-aot": "mcu-aot",
}
_SCHEMA = "lamquant.adr0139.runtime-graph-receipt/v1"


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
    """Execute the conformance graph on this realm and return its receipt."""
    case = Path(__file__).stem
    if case not in _CASES:
        raise SystemExit(f"unknown case: {case}")
    realm = _CASES[case]
    measured = _execute(realm, False)
    complete = measured["clock_and_gap_evidence_preserved"] is True
    receipt = {}
    receipt["schema"] = _SCHEMA
    receipt["case_id"] = case
    receipt["realm"] = measured["realm"]
    receipt["status"] = "pass" if complete else "fail"
    receipt["clock_and_gap_evidence_preserved"] = complete
    receipt["policy_bypass_count"] = measured["policy_bypass_count"]
    receipt["transactional_receipt_failures"] = measured["transactional_receipt_failures"]
    receipt["resource_bound_violations"] = measured["resource_bound_violations"]
    receipt["completed_steps"] = measured["completed_steps"]
    return receipt


def main():
    rendered = json.dumps(produce_evidence(), indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
