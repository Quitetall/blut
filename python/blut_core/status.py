"""status — emit/read the BLUT StatusUpdate wire protocol (``blut/src/protocol.rs``).

One JSON line per event, ``"kind"``-tagged (step/eval/saved/done/failed),
printed to stdout **with an explicit flush** — this defeats Python's
block-buffered stdout so journald + the live monitor see events as they happen
(ADR 0044 P4; the block-buffering is why ``read_metric --unit`` on a codec/SNN
run sees only the launch line today). The same line is appended to
``<job_dir>/status.jsonl`` for post-hoc reads. Mirror ``protocol.rs`` exactly.

stdlib-only. Emitting must never crash a run (the status.jsonl append is guarded).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import runctx

__all__ = ["emit", "step", "eval_pass", "saved", "done", "failed", "read", "VALID_KINDS"]

VALID_KINDS = ("step", "eval", "saved", "done", "failed")
TERMINAL_KINDS = ("done", "failed")


def emit(kind: str, *, job_dir: Optional[Path] = None, to_stdout: bool = True,
         **fields: Any) -> Dict[str, Any]:
    """Emit one StatusUpdate: a flushed JSON line on stdout + an append to
    ``<job_dir>/status.jsonl``. Returns the record. Never raises on the file
    append (logging must not crash a run); a bad ``kind`` IS rejected (caller bug)."""
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown StatusUpdate kind {kind!r}; one of {list(VALID_KINDS)}")
    rec: Dict[str, Any] = {"kind": kind, **fields}
    line = json.dumps(rec)
    if to_stdout:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except OSError:
            pass  # stdout closed (broken pipe in a detached pipeline); the file append still runs
    jd = Path(job_dir) if job_dir is not None else runctx.job_dir()
    try:
        jd.mkdir(parents=True, exist_ok=True)
        with open(jd / "status.jsonl", "a") as f:
            f.write(line + "\n")
    except Exception as e:  # pragma: no cover - defensive
        sys.stderr.write(f"[status] status.jsonl append failed (non-fatal): {e}\n")
    return rec


def step(step: int, total: int, loss: float, lr: float, vram_mb: int = 0,
         **kw: Any) -> Dict[str, Any]:
    return emit("step", step=step, total=total, loss=loss, lr=lr, vram_mb=vram_mb, **kw)


def eval_pass(step: int, eval_loss: float, **kw: Any) -> Dict[str, Any]:
    # `eval` is a Python builtin; the JSON kind is still "eval".
    return emit("eval", step=step, eval_loss=eval_loss, **kw)


def saved(path, **kw: Any) -> Dict[str, Any]:
    return emit("saved", path=str(path), **kw)


def done(final_loss: float, checkpoint_dir, **kw: Any) -> Dict[str, Any]:
    return emit("done", final_loss=final_loss, checkpoint_dir=str(checkpoint_dir), **kw)


def failed(error, **kw: Any) -> Dict[str, Any]:
    return emit("failed", error=str(error), **kw)


def read(path: Path) -> List[Dict[str, Any]]:
    """Parse a ``status.jsonl`` into a list of records. Malformed lines are
    skipped (not guessed). For richer selection use ``read_metric --status``."""
    out: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
