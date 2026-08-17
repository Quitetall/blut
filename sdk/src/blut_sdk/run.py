# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""Submit / monitor / control a plan by shelling the CLI (ADR 0093 exec-bridge).

The SDK is a client of the ``blut`` binary in the same sense ``kubectl`` is a
client of an API server — except there is no server.  It opens no socket and
embeds no runtime, because ADR 0034 forbids an in-engine HTTP surface and ADR
0078 forbids any front-end shipping executable stage bodies.  Everything here
is a subprocess call plus reading files the engine already writes.

That constraint is load-bearing rather than incidental: it is what keeps the
clinical hard-block intact.  Submission goes through the same
``recipe declare`` path a human uses, so the fail-closed tenant gate fires
identically no matter which front-end authored the plan.  A control channel
that bypassed the CLI would be a second door into the engine with its own
policy surface — the drift ADR 0111's Validation section says to watch for.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .plan import PlanSpec

__all__ = ["Run", "RunError", "JobState", "submit"]

#: A job id as the engine mints it: `20260619-082149-230733595`.
_JOB_ID = re.compile(r"\b(\d{8}-\d{6}-\d{9})\b")


class RunError(Exception):
    """A CLI invocation failed, or its output could not be understood."""


@dataclass
class JobState:
    """One row of ``blut jobs --json``."""

    id: str
    state: str
    pid: int | None = None
    output_name: str | None = None
    last_loss: float | None = None
    last_step: int | None = None
    final_loss: float | None = None

    @property
    def finished(self) -> bool:
        # Anything that is not actively running is terminal for polling
        # purposes. Listing the live states rather than the terminal ones is
        # deliberate: a new terminal state added engine-side would otherwise
        # make `wait()` loop forever, whereas a new *running* state at worst
        # makes it return early and report that state verbatim.
        return self.state not in ("running", "queued", "pending", "starting")

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "JobState":
        return cls(
            id=data["id"],
            state=data.get("state", "unknown"),
            pid=data.get("pid"),
            output_name=data.get("output_name"),
            last_loss=data.get("last_loss"),
            last_step=data.get("last_step"),
            final_loss=data.get("final_loss"),
        )


def _run_cli(blut: str, args: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [blut, *args], capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        raise RunError(
            f"'{blut}' is not on PATH; blut-sdk drives the engine through its CLI, "
            "so a blut binary (or a cookbook binary such as lqt) must be installed"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RunError(f"'{blut} {' '.join(args)}' timed out after {timeout}s") from exc


class Run:
    """A plan, and the job it becomes once submitted."""

    def __init__(self, spec: PlanSpec, *, blut: str = "blut") -> None:
        self.spec = spec
        self.blut = blut
        self.job_id: str | None = None

    # -- submission -------------------------------------------------------

    def declare(self, *, timeout: float = 300.0, extra_args: list[str] | None = None) -> str:
        """Compile and kind-check WITHOUT executing; return the CLI's output.

        This is the engine's dry run.  Note the spelling: ``recipe declare``
        renders the DAG and executes nothing *by default* — there is no
        ``--dry-run`` flag on it (that flag lives on ``recipe run``).  Passing
        one is an "unexpected argument" error, so a gate written against the
        wrong spelling fails for a reason unrelated to the plan.
        """
        return self._declare(run=False, timeout=timeout, extra_args=extra_args)

    def submit(
        self,
        *,
        timeout: float = 300.0,
        tenant: str | None = None,
        experiment: str | None = None,
        shared_cache: bool = False,
        no_cache: bool = False,
        extra_args: list[str] | None = None,
    ) -> str:
        """LAUNCH the plan; return the job id.

        Goes through ``recipe declare --run``, i.e. the same admission-gated,
        cgroup-contained, cache-honouring path as a hand-run recipe.
        """
        args = list(extra_args or [])
        if tenant:
            args += ["--tenant", tenant]
        if experiment:
            args += ["--experiment", experiment]
        if shared_cache:
            args.append("--shared-cache")
        if no_cache:
            args.append("--no-cache")
        output = self._declare(run=True, timeout=timeout, extra_args=args)
        match = _JOB_ID.search(output)
        if not match:
            raise RunError(
                "the launch succeeded but no job id appeared in its output, so this "
                "Run cannot be monitored; check `blut jobs` for the new job.\n"
                f"--- output ---\n{output.strip()[:1000]}"
            )
        self.job_id = match.group(1)
        return self.job_id

    def _declare(self, *, run: bool, timeout: float, extra_args: list[str] | None) -> str:
        # A NamedTemporaryFile that deletes on close would race the subprocess
        # on some platforms; write, close, invoke, then remove in `finally` so
        # the file cannot outlive a failure either.
        handle, path = tempfile.mkstemp(prefix="blut_sdk_plan_", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as f:
                f.write(self.spec.to_json_text())
            args = ["recipe", "declare", path]
            if run:
                args.append("--run")
            args += list(extra_args or [])
            completed = _run_cli(self.blut, args, timeout=timeout)
            if completed.returncode != 0:
                raise RunError(
                    f"`{self.blut} recipe declare` failed with exit {completed.returncode}.\n"
                    f"--- stderr ---\n{completed.stderr.strip()[:2000]}\n"
                    f"--- stdout ---\n{completed.stdout.strip()[:2000]}"
                )
            return completed.stdout
        finally:
            Path(path).unlink(missing_ok=True)

    # -- monitoring -------------------------------------------------------

    def status(self, *, timeout: float = 60.0) -> JobState:
        """This run's row from ``blut jobs --json``."""
        job_id = self._require_job()
        completed = _run_cli(self.blut, ["jobs", "--json"], timeout=timeout)
        if completed.returncode != 0:
            raise RunError(f"`{self.blut} jobs --json` failed: {completed.stderr.strip()[:400]}")
        try:
            rows = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RunError(f"`{self.blut} jobs --json` did not emit JSON: {exc}") from exc
        for row in rows:
            if row.get("id") == job_id:
                return JobState.from_json(row)
        raise RunError(f"job {job_id} is not in the job list")

    def wait(self, *, poll: float = 5.0, timeout: float | None = None) -> JobState:
        """Poll until the job leaves its running state."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            state = self.status()
            if state.finished:
                return state
            if deadline is not None and time.monotonic() >= deadline:
                raise RunError(f"job {state.id} still {state.state} after {timeout}s")
            time.sleep(poll)

    def status_path(self) -> Path:
        """Path to this job's ``status.jsonl``.

        Mirrors ``paths::jobs_dir`` — ``$LAMU_TRAIN_JOBS_DIR`` if set, else
        ``~/.local/share/lamu/train-jobs``.  The engine's own
        ``paths::job_dir`` *creates* the directory; this deliberately does not,
        because an SDK that conjured an empty job directory would make a typo'd
        id look like a real job with no events yet.
        """
        job_id = self._require_job()
        base = os.environ.get("LAMU_TRAIN_JOBS_DIR")
        root = Path(base) if base else Path.home() / ".local/share/lamu/train-jobs"
        return root / job_id / "status.jsonl"

    def events(self) -> list[dict[str, Any]]:
        """Every ``StatusUpdate`` written so far (empty if none yet)."""
        path = self.status_path()
        if not path.exists():
            return []
        events = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # The engine appends this file live, so the last line can be
                    # a partial write. Skipping it is correct — it will be
                    # complete on the next read — whereas raising would make
                    # `stream()` die at a random point in a healthy run.
                    continue
        return events

    def stream(self, *, poll: float = 2.0, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        """Yield ``StatusUpdate`` events as they are appended, until the job ends."""
        seen = 0
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            events = self.events()
            for event in events[seen:]:
                yield event
            seen = len(events)
            state = self.status()
            if state.finished:
                # One last read: the terminal events may land between the
                # previous read and the state flip.
                for event in self.events()[seen:]:
                    yield event
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise RunError(f"job {state.id} still {state.state} after {timeout}s")
            time.sleep(poll)

    # -- control ----------------------------------------------------------

    def cancel(self, *, grace: str | None = None, timeout: float = 60.0) -> str:
        """SIGTERM the job (``blut cancel <id>``)."""
        job_id = self._require_job()
        args = ["cancel", job_id]
        if grace:
            args += ["--grace", grace]
        completed = _run_cli(self.blut, args, timeout=timeout)
        if completed.returncode != 0:
            raise RunError(f"`{self.blut} cancel {job_id}` failed: {completed.stderr.strip()[:400]}")
        return completed.stdout

    def _require_job(self) -> str:
        if not self.job_id:
            raise RunError("this Run has not been submitted yet (call submit() first)")
        return self.job_id


def submit(spec: PlanSpec, *, blut: str = "blut", **kwargs: Any) -> Run:
    """Convenience: build a :class:`Run`, submit it, and hand it back."""
    run = Run(spec, blut=blut)
    run.submit(**kwargs)
    return run
