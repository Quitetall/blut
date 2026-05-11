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


def self_check() -> int:
    print(
        json.dumps(
            {
                "kind": "step",
                "step": 1,
                "total": 2,
                "loss": 1.0,
                "lr": 0.0,
                "vram_mb": 0,
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
            }
        ),
        flush=True,
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--self-check":
        return self_check()
    emit_failed(
        "trainer_dpo.py is a stub. Full DPO implementation pending. "
        "Track progress in unified-launching-quill.md commit 7 follow-up. "
        "Use --self-check for protocol smoke."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
