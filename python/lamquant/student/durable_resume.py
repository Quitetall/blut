"""Durable resume (BLUT-API Phase D) — the TRAINER half (checkpoint mechanics).

The orchestrator half (the crash-gated decision) lives in the Rust engine
(`blut::framework::resume`). This module is the Python side it talks to: it
writes the `state.json` run-state marker + atomic recovery checkpoints into the
stable resume directory the stage passes via ``--resume-dir``, and restores
model + optimizer + RNG on ``--resume``. The whole contract between the two is:
this directory, the ``state.json`` schema, and the recovery checkpoint payload.

Key design points:

* **Daemon heartbeat thread.** ``start()`` spawns a daemon thread that rewrites
  ``state.json.heartbeat_unix`` every ``HEARTBEAT_INTERVAL`` seconds. It is
  decoupled from the training loop entirely, so a *live* run always looks alive
  (no false "crashed" detection mid-long-epoch), and on a hard kill (SIGKILL /
  OOM) the thread dies with the process and the heartbeat freezes — the
  orchestrator then sees a stale heartbeat and treats the dir as resumable.
  ``HEARTBEAT_INTERVAL`` MUST match ``blut`` ``resume::HEARTBEAT_INTERVAL_SECS``.

* **Atomic, rotated checkpoints.** ``save_recovery`` writes to a ``.tmp`` then
  ``os.replace``s it into place (atomic publish), rotating the prior good file
  to ``<name>.prev.ckpt`` first — so a kill mid-write can corrupt at most the
  ``.tmp``, never both the latest and the prev.

* **Config-hash guard.** Every checkpoint embeds the ``resume_key``; a load that
  finds a checkpoint whose key differs from the current one rejects it (never
  resume onto a foreign config's checkpoint — belt-and-suspenders on top of the
  per-config resume directory).
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

# Must equal blut `resume::HEARTBEAT_INTERVAL_SECS` (the Rust stale window is
# 3× this). Time-based, NOT tied to the epoch/validation cadence.
HEARTBEAT_INTERVAL = 60

# The recovery-checkpoint keys the trainer (`train_joint.py`) reads
# UNCONDITIONALLY on resume — absence of any one ⇒ a `KeyError` that aborts the
# resume, OR (worse) a silent cold-start. This tuple is the SINGLE SOURCE OF
# TRUTH for "a recovery checkpoint is structurally complete", referenced by both
# `save_recovery` (sanity-check before write) and `load_recovery` (fail-fast
# validate after read). Derived from the read sites in `train_joint.py`:
#   * ``ckpt['encoder']`` / ``ckpt['decoder']``     (state-dict loads)
#   * ``ckpt['epoch']`` / ``ckpt['phase']``         (resume position)
#   * ``_qat_ckpt['optimizer']``                    (continuous-optimizer resume)
# plus the two keys `save_recovery` itself always embeds:
#   * ``rng``         (DurableResume.restore_rng — continuous data/RNG stream)
#   * ``resume_key``  (foreign-config guard)
# Keys read via ``.get(...)`` in the trainer (``scheduler``, ``seizure_head``,
# ``best_val_r``, ``best_val_prd``, ``scaler``) are OPTIONAL and deliberately
# NOT listed here — a checkpoint without them resumes correctly.
REQUIRED_RECOVERY_KEYS = (
    "encoder",
    "decoder",
    "epoch",
    "phase",
    "optimizer",
    "rng",
    "resume_key",
)

# Free-space margin for a recovery-checkpoint write, mirroring the statvfs
# disk-fill guard used by the fullband / L3 caches
# (`lma_typed_adapter.py::_fb_win_save`, `warm_fb_cache.py`,
# `snn/lma_dataset.py`): free bytes = ``st.f_bavail * st.f_frsize``, threshold
# from an env override × 1e9. The cache guards default to 40 GB because they
# write hundreds of GB of windows; a recovery checkpoint is a single
# encoder+decoder+optimizer blob (sub-GB to a few GB), so the default margin is
# smaller — just enough headroom that the ``.tmp`` can't truncate mid-write.
RECOVERY_MIN_FREE_GB_DEFAULT = 2.0


def _recovery_min_free_bytes() -> int:
    """Free-space floor (bytes) required before writing a recovery `.tmp`.

    Reads ``RECOVERY_MIN_FREE_GB`` (GB, default ``RECOVERY_MIN_FREE_GB_DEFAULT``)
    and scales by 1e9 — same idiom as the cache disk-fill guards. A malformed
    env value (non-numeric, NaN) falls back to the default rather than raising
    ``ValueError`` mid-save: a typo'd knob must not crash a long training run,
    and the default margin is the safe floor."""
    try:
        gb = float(os.environ.get("RECOVERY_MIN_FREE_GB", RECOVERY_MIN_FREE_GB_DEFAULT))
        # math.isfinite rejects NaN AND ±inf in one shot — guards the int(gb*1e9)
        # below from OverflowError on inf (or an inf-overflowing finite value).
        if not math.isfinite(gb) or gb < 0:
            gb = RECOVERY_MIN_FREE_GB_DEFAULT
    except (ValueError, TypeError, OverflowError):
        # OverflowError: some libc builds raise on float("1e400") instead of inf.
        gb = RECOVERY_MIN_FREE_GB_DEFAULT
    try:
        return int(gb * 1e9)
    except (OverflowError, ValueError):  # belt-and-suspenders for any residual overflow
        return int(RECOVERY_MIN_FREE_GB_DEFAULT * 1e9)


def _free_bytes(path: Any) -> int:
    """Free bytes available at ``path`` via ``os.statvfs`` (``f_bavail *
    f_frsize``). FAIL-CLOSED: an unknowable free-space state returns 0 so the
    caller's ``< min_free`` guard trips (skip the write) rather than risking a
    truncated checkpoint on a genuinely full/inaccessible disk — mirrors
    ``warm_fb_cache._free_gb``."""
    try:
        st = os.statvfs(os.fspath(path))
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def _default_loader(path: Any, map_location: Any = "cpu") -> dict:
    """Load a recovery checkpoint with ``weights_only=False``. These are the
    trainer's OWN checkpoints (we wrote them this process/last process), and
    they carry non-tensor state — the optimizer state_dict + the RNG blobs
    (numpy arrays / python tuples) — which PyTorch 2.6's ``weights_only=True``
    default refuses to unpickle. The files live under the operator's data root,
    same trust level as the dataset itself."""
    return torch.load(path, map_location=map_location, weights_only=False)


class DurableResume:
    """Owns the resume dir: the ``state.json`` marker + recovery checkpoints."""

    def __init__(self, resume_dir: str, run_id: str = "", resume_key: str = ""):
        self.dir = Path(resume_dir)
        self.run_id = run_id or ""
        self.key = resume_key or ""
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    # ---- run-state marker -------------------------------------------------

    def _write_state(self, status: str) -> None:
        body = json.dumps(
            {
                "status": status,
                "run_id": self.run_id,
                "pid": self.pid,
                "heartbeat_unix": int(time.time()),
            }
        )
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(body)
        os.replace(tmp, self.dir / "state.json")  # atomic publish

    def start(self) -> "DurableResume":
        """Mark the run ``running`` and start the daemon heartbeat."""
        self._write_state("running")

        def _loop() -> None:
            # wait() returns True the instant _stop is set → the loop exits
            # without another write. finish() sets _stop, join()s this thread,
            # THEN writes "finished" — so the heartbeat can never clobber the
            # final "finished" marker (no inner re-check needed).
            while not self._stop.wait(HEARTBEAT_INTERVAL):
                try:
                    self._write_state("running")
                except Exception:  # noqa: BLE001 — a heartbeat hiccup must never crash training
                    pass

        self._thr = threading.Thread(target=_loop, name="durable-heartbeat", daemon=True)
        self._thr.start()
        return self

    def finish(self) -> None:
        """Mark the run ``finished`` (a clean completion is NEVER resumed)."""
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2)  # ensure the heartbeat can't write after us
        try:
            self._write_state("finished")
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "DurableResume":
        return self.start()

    def __exit__(self, exc_type: Any, *_exc: Any) -> bool:
        # finish() (status "finished") ONLY on a clean exit. On an exception the
        # run did NOT complete — stop the heartbeat but leave the marker
        # "running", so it goes stale and the orchestrator treats the dir as a
        # resumable crash. (train_joint.py calls start()/finish() explicitly
        # rather than using this CM, but keep the protocol crash-correct.)
        if exc_type is None:
            self.finish()
        else:
            self._stop.set()
            if self._thr is not None:
                self._thr.join(timeout=2)
        return False

    # ---- RNG capture / restore -------------------------------------------

    @staticmethod
    def capture_rng() -> dict:
        st = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            st["cuda"] = torch.cuda.get_rng_state_all()
        return st

    @staticmethod
    def restore_rng(st: Optional[dict]) -> None:
        if not st:
            return
        try:
            random.setstate(st["python"])
            np.random.set_state(st["numpy"])
            torch.set_rng_state(st["torch"])
            if "cuda" in st and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(st["cuda"])
        except Exception as e:  # noqa: BLE001 — a stale/foreign RNG blob must not abort resume
            print(f"[durable] RNG restore skipped ({e})")

    # ---- recovery checkpoints --------------------------------------------

    def save_recovery(
        self,
        name: str,
        payload: dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Any = None,
    ) -> None:
        """Atomically write ``<name>.ckpt`` (rotating the prior good copy to
        ``<name>.prev.ckpt``), embedding the resume_key + RNG (+ optimizer /
        scaler state if the caller's ``payload`` doesn't already carry them)."""
        full = dict(payload)
        full["resume_key"] = self.key
        full.setdefault("rng", self.capture_rng())
        if optimizer is not None and "optimizer" not in full:
            full["optimizer"] = optimizer.state_dict()
        if scaler is not None and "scaler" not in full:
            full["scaler"] = scaler.state_dict()

        # Completeness sanity-check against the single-source-of-truth key set.
        # The trainer reads these UNCONDITIONALLY on resume; a checkpoint missing
        # one would resume-abort (or cold-start). This is a non-fatal warning,
        # not a hard raise: a missing recovery is recoverable (the orchestrator
        # falls back to a fresh start / the prev rotation), and we never want a
        # heartbeat/save hiccup to crash a long training run. The matching
        # fail-fast lives on the LOAD side (`load_recovery` rejects an incomplete
        # checkpoint), where a cold-optimizer silent resume is the real hazard.
        _missing = [k for k in REQUIRED_RECOVERY_KEYS if k not in full]
        if _missing:
            warnings.warn(
                f"[durable] save_recovery({name!r}) payload missing required "
                f"key(s) {_missing}; this checkpoint will be REJECTED on resume "
                f"(REQUIRED_RECOVERY_KEYS={REQUIRED_RECOVERY_KEYS})",
                RuntimeWarning,
                stacklevel=2,
            )

        # Free-space preflight (statvfs) — mirrors the cache disk-fill guards.
        # A near-full disk produces a truncated `.tmp`; `os.replace` would then
        # publish a CORRUPT latest, and on the next save rotate it over the last
        # GOOD `.prev` — destroying the rotation. Skip the save loudly instead:
        # recovery is best-effort, a MISSING recovery is recoverable, a TRUNCATED
        # one corrupts the chain. The good `.prev` is left untouched (we never
        # delete it to make room).
        min_free = _recovery_min_free_bytes()
        free = _free_bytes(self.dir)
        if free < min_free:
            warnings.warn(
                f"[durable] save_recovery({name!r}) skipped: only "
                f"{free / 1e9:.2f} GB free at {self.dir}, need >= "
                f"{min_free / 1e9:.2f} GB to write without risking a truncated "
                f"checkpoint (set RECOVERY_MIN_FREE_GB to override). Existing "
                f"recovery checkpoint + prev rotation left intact.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        dst = self.dir / f"{name}.ckpt"
        if dst.exists():
            try:
                os.replace(dst, self.dir / f"{name}.prev.ckpt")  # keep the last good one
            except OSError:
                pass
        tmp = self.dir / f"{name}.ckpt.tmp"
        torch.save(full, tmp)
        os.replace(tmp, dst)  # atomic publish

    def detect(self) -> Optional[str]:
        """Auto-pick the freshest resumable checkpoint base name (qat over
        warm). Returns ``None`` if the dir holds no recovery checkpoint."""
        for name in ("qat_latest", "warm_latest"):
            if (self.dir / f"{name}.ckpt").exists() or (self.dir / f"{name}.prev.ckpt").exists():
                return name
        return None

    @staticmethod
    def _missing_required(ck: Any) -> list:
        """Return the ``REQUIRED_RECOVERY_KEYS`` absent from ``ck`` (the whole
        set if ``ck`` is not a dict). Empty ⇒ structurally complete."""
        if not isinstance(ck, dict):
            return list(REQUIRED_RECOVERY_KEYS)
        return [k for k in REQUIRED_RECOVERY_KEYS if k not in ck]

    def load_recovery(
        self,
        name: str,
        map_location: Any = "cpu",
        loader: Callable[..., dict] = _default_loader,
    ) -> Optional[dict]:
        """Load ``<name>.ckpt``, falling back to ``<name>.prev.ckpt`` on a
        corrupt, foreign-key, or STRUCTURALLY INCOMPLETE latest. Returns the
        checkpoint dict, or ``None`` if neither file exists OR every present
        candidate is a FOREIGN-config checkpoint (the caller then starts fresh).

        Validate-on-load (Phase D): a half-written checkpoint that loads but is
        missing a ``REQUIRED_RECOVERY_KEYS`` member (e.g. no ``optimizer``) is
        treated EXACTLY like a corrupt file — it falls through to the prev
        rotation. If a candidate is OURS (matching/empty resume_key) yet
        unreadable or incomplete, and no usable candidate is found, this RAISES
        ``RuntimeError`` rather than returning ``None``: a present-but-broken
        recovery dir means the crash-gated orchestrator chose to resume, and
        silently cold-starting the optimizer/scheduler would corrupt the loss
        curve. A loud refusal beats a silent cold resume.

        A READABLE checkpoint with a FOREIGN ``resume_key`` is NOT a
        broken-checkpoint case — it is a leftover from a different config (the
        resume dir is per-config). It returns ``None`` (start fresh), the
        existing belt-and-suspenders guard, never a raise. (An UNREADABLE file,
        by contrast, has no inspectable ``resume_key`` — in a per-config dir it
        is almost certainly our own torn write, so it is treated as ours-broken
        and contributes to the loud refusal; a corrupt latest still falls
        through to the prev rotation first.)"""
        saw_broken = False  # an unreadable, or OURS-but-incomplete, candidate
        broken_detail = ""  # last observed failure reason → threaded into the raise
        for cand in (self.dir / f"{name}.ckpt", self.dir / f"{name}.prev.ckpt"):
            if not cand.exists():
                continue
            try:
                ck = loader(cand, map_location=map_location)
            except Exception as e:  # noqa: BLE001 — a corrupt latest falls through to prev
                # Unreadable ⇒ key uninspectable. In a per-config resume dir this
                # is almost certainly our own torn write, so flag it broken (the
                # loop still tries the prev before any raise fires).
                print(f"[durable] {cand.name} unreadable ({e}); trying prev")
                saw_broken = True
                broken_detail = f"{cand.name} unreadable ({e})"
                continue
            ck_key = ck.get("resume_key", "") if isinstance(ck, dict) else ""
            if self.key and ck_key and ck_key != self.key:
                print(f"[durable] {cand.name} resume_key mismatch (foreign checkpoint); skipping")
                continue
            # Validate-on-load: a structurally incomplete checkpoint (half-written
            # — missing optimizer/encoder/etc.) must NOT be returned, or the
            # trainer resumes with a cold optimizer / KeyErrors mid-restore.
            missing = self._missing_required(ck)
            if missing:
                print(
                    f"[durable] {cand.name} incomplete — missing required "
                    f"key(s) {missing}; trying prev"
                )
                saw_broken = True
                broken_detail = f"{cand.name} missing required key(s) {missing}"
                continue
            return ck
        if saw_broken:
            # A checkpoint was on disk but unusable (unreadable, or ours-but-
            # incomplete) and no good fallback was found. Refuse loudly rather
            # than silently cold-start (see docstring). A READABLE foreign-key
            # checkpoint does NOT set saw_broken — it falls through to the None
            # below (start fresh).
            raise RuntimeError(
                f"[durable] recovery checkpoint(s) for {name!r} present in "
                f"{self.dir} but none is usable (last failure: {broken_detail}; "
                f"required keys {list(REQUIRED_RECOVERY_KEYS)}). Refusing to "
                f"cold-start a resume — inspect or remove the recovery dir."
            )
        return None
