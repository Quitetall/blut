#!/usr/bin/env python3
"""DPO trainer subprocess.

v2 commit 7: stub. Full DPO implementation lands in a follow-up
once the chosen library (trl.DPOTrainer) is integrated. For now
this script accepts the same TrainSpec JSON shape as trainer.py
but emits a `Failed` status and exits non-zero. The Rust-side
typed contract (DpoTrain stage + dpo_from_preferences recipe) is
in place and ready; only the actual gradient step is missing.
"""

from __future__ import annotations

import json
import sys


def emit_failed(error: str) -> None:
    print(json.dumps({"kind": "failed", "error": error}), flush=True)


def read_spec(argv: list[str]) -> dict | None:
    """Best-effort parse of the TrainSpec JSON passed as an argv token.

    The Rust backend invokes us as `trainer_dpo.py <spec_json>`; the
    `--self-check` smoke path may also carry a spec. Return the first token
    that parses as a JSON object, else None.
    """
    for tok in argv[1:]:
        tok = tok.strip()
        if tok.startswith("{"):
            try:
                obj = json.loads(tok)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def self_check(spec: dict | None = None) -> int:
    # Echo the DPO temperature we received so the end-to-end thread
    # (Rust Args -> TrainSpec.dpo_beta -> JSON -> here) is testable.
    beta = spec.get("dpo_beta") if spec else None
    print(
        json.dumps(
            {
                "kind": "step",
                "step": 1,
                "total": 2,
                "loss": 1.0,
                "lr": 0.0,
                "vram_mb": 0,
                "dpo_beta": beta,
            }
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "kind": "done",
                "final_loss": 1.0,
                "checkpoint_dir": "/tmp/lamu-dpo-self-check",
                "dpo_beta": beta,
            }
        ),
        flush=True,
    )
    return 0


def main(argv: list[str]) -> int:
    spec = read_spec(argv)
    if len(argv) >= 2 and argv[1] == "--self-check":
        return self_check(spec)
    # Stub: no gradient step yet (trl.DPOTrainer integration pending), but the
    # DPO temperature is now threaded through and surfaced so the contract is
    # observably wired end-to-end.
    beta = spec.get("dpo_beta") if spec else None
    emit_failed(
        "trainer_dpo.py is a stub. Full DPO implementation pending "
        f"(received dpo_beta={beta}). Use --self-check for protocol smoke."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
