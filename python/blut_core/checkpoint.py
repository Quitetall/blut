"""checkpoint — corruption-safe torch checkpoint I/O + a resume payload contract
(ADR 0044 P7, "durability"; the single most important reliability gap).

- ``save(payload, path)`` — free-space preflight → write to a unique tmp in the
  same dir → ``fsync`` → atomic ``os.replace`` → write a sidecar ``<path>.sha256``.
  A mid-write kill leaves the PRIOR checkpoint intact, never a truncated file
  (the failure mode that silently corrupts on a disk-full ``torch.save``).
  Optional ``keep_last_n`` rotation of ``<stem>_ep*.pt`` siblings.
- ``load(path, validate=True)`` — verify the sidecar SHA (or, absent a sidecar,
  a dry ``torch.load`` of the bytes) BEFORE returning → a corrupt/truncated
  checkpoint raises ``CheckpointError``, never silent garbage.
- ``validate_payload(payload, required)`` — assert the resume contract keys are
  present so a resume fails fast at SAVE, not 3 epochs into a relaunch (prior
  bites: optimizer-state, QAT-alpha). Default contract = the training-resume set.

torch is lazy-imported (only on save/load), so importing ``blut_core`` stays cheap.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

__all__ = ["save", "load", "validate_payload", "RESUME_CONTRACT", "CheckpointError"]

# ADR 0044 P7 resume payload contract. A trainer that wants resumability MUST
# put these in the checkpoint; validate_payload(strict=True) enforces it.
RESUME_CONTRACT = (
    "model", "optimizer", "lr_scheduler", "rng_state", "step",
)


class CheckpointError(RuntimeError):
    """A checkpoint is missing, corrupt, or violates the resume contract."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_payload(payload: Dict[str, Any], required: Iterable[str] = RESUME_CONTRACT,
                     *, strict: bool = True) -> list:
    """Return the list of missing contract keys. ``strict`` → raise on any missing.
    Use before save so a resume can't silently lack optimizer/RNG/step."""
    if not isinstance(payload, dict):
        raise CheckpointError(f"payload must be a dict, got {type(payload).__name__}")
    missing = [k for k in required if k not in payload]
    if missing and strict:
        raise CheckpointError(
            f"checkpoint payload missing resume-contract keys {missing}; "
            f"present: {sorted(payload)}")
    return missing


def _free_bytes(directory: Path) -> int:
    return shutil.disk_usage(directory).free


def save(payload: Dict[str, Any], path, *, contract: Optional[Iterable[str]] = None,
         keep_last_n: Optional[int] = None, min_free_mb: int = 256) -> Path:
    """Atomically save ``payload`` to ``path`` with a sidecar SHA.

    ``contract`` (default None = skip) → validate_payload(strict) first.
    ``keep_last_n`` → after writing, keep only the newest N ``<stem>_ep*.pt``.
    Raises ``CheckpointError`` on insufficient free space or a contract miss."""
    import torch  # lazy
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if contract is not None:
        validate_payload(payload, contract, strict=True)

    # Free-space preflight: a disk-full torch.save silently yields garbage.
    if _free_bytes(path.parent) < min_free_mb * (1 << 20):
        raise CheckpointError(
            f"insufficient free space in {path.parent} (<{min_free_mb} MiB) — refusing to save")

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.stem + ".",
                                    suffix=path.suffix + ".tmp")
    tmp = Path(tmp_name)
    try:
        os.close(fd)  # torch.save writes by path
        torch.save(payload, tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        sha = _sha256(tmp)
        os.replace(tmp, path)
        # fsync the sidecar too — the durability contract above promises the
        # SHA is persisted, but write_text() leaves it in the page cache.
        sha_path = path.with_suffix(path.suffix + ".sha256")
        sfd = os.open(str(sha_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(sfd, (sha + "\n").encode("utf-8"))
            os.fsync(sfd)
        finally:
            os.close(sfd)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass
        raise

    if keep_last_n is not None and keep_last_n > 0:
        _rotate(path, keep_last_n)
    return path


def _rotate(path: Path, keep_last_n: int) -> None:
    # Rotates the EPOCH-SNAPSHOT siblings `{stem}_ep<N>{suffix}` (the trainer
    # convention, e.g. snn_4state_best_ep10.pt), keeping the newest N. It does
    # NOT touch the base `{stem}{suffix}` (the canonical/best checkpoint) — that
    # is intentional: you keep best + the last N epoch snapshots.
    sibs = sorted(path.parent.glob(f"{path.stem}_ep*{path.suffix}"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for old in sibs[keep_last_n:]:
        for victim in (old, old.with_suffix(old.suffix + ".sha256")):
            try:
                victim.unlink()
            except OSError:
                pass


def load(path, *, validate: bool = True, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint, verifying integrity first. Raises ``CheckpointError``
    on a missing/corrupt file rather than returning silent garbage."""
    import torch  # lazy
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"no checkpoint: {path}")
    if validate:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.exists():
            want = sidecar.read_text().strip()
            got = _sha256(path)
            if got != want:
                raise CheckpointError(
                    f"checkpoint SHA mismatch for {path}: sidecar {want[:12]}… got {got[:12]}…")
        else:
            # No sidecar → integrity is UNVERIFIED (e.g. a crash between the
            # checkpoint rename and the sidecar write; the sidecar is not part
            # of the atomic unit). Surface it; don't silently accept-as-clean.
            import sys as _sys
            _sys.stderr.write(
                f"[checkpoint] WARNING: no .sha256 sidecar for {path}; integrity unverified\n")
    try:
        # NOT weights_only=True: the resume payload carries optimizer/LR/RNG/
        # scheduler objects (not plain tensors), which weights_only rejects.
        # Trust model: checkpoints are own-produced under the job dir, not loaded
        # from untrusted sources — the sidecar SHA guards corruption, not supply chain.
        return torch.load(path, map_location=map_location)
    except Exception as e:
        raise CheckpointError(f"failed to load {path}: {type(e).__name__}: {e}") from e
