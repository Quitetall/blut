#!/usr/bin/env python3
"""Knowledge distillation trainer subprocess.

v2 commit 7: stub. Full distillation lands in a follow-up.
Same shape as trainer_dpo.py — the Rust typed contract
(DistillTrain stage + distill_from_teacher recipe) is in place.
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
                "checkpoint_dir": "/tmp/lamu-distill-self-check",
            }
        ),
        flush=True,
    )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--self-check":
        return self_check()
    emit_failed(
        "trainer_distill.py is a stub. Full distillation implementation pending. "
        "Use --self-check for protocol smoke."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
