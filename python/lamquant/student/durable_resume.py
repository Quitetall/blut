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
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

# Must equal blut `resume::HEARTBEAT_INTERVAL_SECS` (the Rust stale window is
# 3× this). Time-based, NOT tied to the epoch/validation cadence.
HEARTBEAT_INTERVAL = 60


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
            # wait() returns True the instant _stop is set → exit without a
            # further write (so finish()'s "finished" is never clobbered).
            while not self._stop.wait(HEARTBEAT_INTERVAL):
                if self._stop.is_set():
                    break
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

    def __exit__(self, *_exc: Any) -> bool:
        # NOTE: __exit__ marks finished on ANY exit including an exception. The
        # trainer therefore calls finish() explicitly only on a CLEAN completion
        # (an exception must leave the marker "running" → stale → resumable), so
        # train_joint.py does NOT use this context manager around the loop.
        self.finish()
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

    def load_recovery(
        self,
        name: str,
        map_location: Any = "cpu",
        loader: Callable[..., dict] = _default_loader,
    ) -> Optional[dict]:
        """Load ``<name>.ckpt``, falling back to ``<name>.prev.ckpt`` on a
        corrupt or foreign-key latest. Returns the checkpoint dict, or ``None``
        if neither file is usable (the caller then starts fresh)."""
        for cand in (self.dir / f"{name}.ckpt", self.dir / f"{name}.prev.ckpt"):
            if not cand.exists():
                continue
            try:
                ck = loader(cand, map_location=map_location)
            except Exception as e:  # noqa: BLE001 — a corrupt latest falls through to prev
                print(f"[durable] {cand.name} unreadable ({e}); trying prev")
                continue
            ck_key = ck.get("resume_key", "") if isinstance(ck, dict) else ""
            if self.key and ck_key and ck_key != self.key:
                print(f"[durable] {cand.name} resume_key mismatch (foreign checkpoint); skipping")
                continue
            return ck
        return None
