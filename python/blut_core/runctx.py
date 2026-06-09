"""runctx — resolve a training run's identity + filesystem anchors from the
BLUT environment, ONCE (ADR 0044 P10).

BLUT's runner sets ``BLUT_JOB_DIR`` + ``BLUT_STAGE_NAME`` (+ ``BLUT_CONTAINED``)
across the systemd ``--setenv`` boundary when a run launches as a contained
stage (blut-lamquant runner). Standalone runs (raw script, E1-style) have none
→ fall back to a repo-local ``training_logs``. Every primitive that needs
"where do metrics / logs / checkpoints go" resolves it HERE, so the convention
is implemented once instead of re-derived per trainer.

stdlib-only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["job_dir", "stage_name", "is_contained", "RunContext", "resolve"]


def _repo_training_logs() -> Path:
    # blut_core/runctx.py → parents[1] == blut/python
    return Path(__file__).resolve().parents[1] / "training_logs"


def job_dir(fallback: Optional[Path] = None) -> Path:
    """The per-job anchor for logs / metrics / checkpoints.

    ``$BLUT_JOB_DIR`` when set (a BLUT stage), else ``fallback``, else the
    repo-local ``blut/python/training_logs``.
    """
    env = os.environ.get("BLUT_JOB_DIR")
    if env:
        return Path(env)
    return Path(fallback) if fallback is not None else _repo_training_logs()


def stage_name() -> Optional[str]:
    """``$BLUT_STAGE_NAME`` if this run is a BLUT stage, else None."""
    return os.environ.get("BLUT_STAGE_NAME") or None


def is_contained() -> bool:
    """True iff launched under BLUT's systemd containment (``BLUT_CONTAINED=1``)."""
    return os.environ.get("BLUT_CONTAINED") == "1"


@dataclass(frozen=True)
class RunContext:
    """Immutable snapshot of a run's identity + placement."""
    run_id: str
    job_dir: Path
    stage_name: Optional[str]
    contained: bool


def resolve(run_id: str, *, fallback_log_dir: Optional[Path] = None) -> RunContext:
    """Resolve the full run context for ``run_id`` from the environment."""
    if not run_id or not isinstance(run_id, str):
        raise ValueError(f"run_id must be a non-empty str, got {run_id!r}")
    return RunContext(
        run_id=run_id,
        job_dir=job_dir(fallback_log_dir),
        stage_name=stage_name(),
        contained=is_contained(),
    )
