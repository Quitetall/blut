# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
"""The decorator sugar and the exec-bridge.

The decorators are the layer most likely to be misread as "write a task in
Python and BLUT runs it", so most of these tests are about the refusal that
makes the misreading loud.  The Run tests cover the failure paths, since the
happy path needs a real engine and is exercised by the ADR gate instead.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from blut_sdk import PlanBuilder, PlanSpec, Run, RunError, stage_ref, workflow
from blut_sdk.decorators import current_builder


def test_decorated_workflow_builds_the_same_graph_as_the_builder():
    @stage_ref("prepare_data")
    def prepare(corpus: str): ...

    @stage_ref("train_model")
    def train(tier: int): ...

    @stage_ref("compare_report")
    def compare(): ...

    @workflow("nightly")
    def build():
        root = prepare(corpus="tuh")
        left = train(tier=1, after=root)
        right = train(tier=2, after=root)
        compare(after=[left, right])

    manual = PlanBuilder("nightly")
    root = manual.add("prepare_data", {"corpus": "tuh"})
    a = manual.add("train_model", {"tier": 1}, after=root)
    b = manual.add("train_model", {"tier": 2}, after=root)
    manual.add("compare_report", after=[a, b])

    assert build().canonical_bytes() == manual.build().canonical_bytes()


def test_a_stage_ref_with_a_real_body_is_refused_at_definition_time():
    # The whole point: a body cannot be shipped into the engine (ADR 0078), so
    # writing one must fail loudly rather than be silently ignored.
    with pytest.raises(ValueError, match="will NEVER run"):

        @stage_ref("train_model")
        def train(tier: int):
            return tier * 2


def test_docstring_and_ellipsis_bodies_are_accepted():
    @stage_ref("a")
    def with_doc(x: int):
        """Just a docstring."""

    @stage_ref("b")
    def with_ellipsis(x: int): ...

    @stage_ref("c")
    def with_pass(x: int):
        pass

    assert with_doc.stage_name == "a"
    assert with_ellipsis.stage_name == "b"
    assert with_pass.stage_name == "c"


def test_bare_decorator_uses_the_function_name_as_the_stage():
    @stage_ref
    def prepare_data(corpus: str): ...

    @workflow
    def nightly():
        prepare_data(corpus="x")

    spec = nightly()
    assert spec.name == "nightly"
    assert spec.nodes[0].stage == "prepare_data"


def test_a_stage_ref_outside_a_workflow_says_so():
    @stage_ref("a")
    def thing(): ...

    with pytest.raises(RuntimeError, match="no active workflow"):
        thing()


def test_unknown_keyword_is_caught_against_the_declared_signature():
    @stage_ref("train_model")
    def train(tier: int): ...

    @workflow("w")
    def build():
        train(teir=1)  # typo

    with pytest.raises(TypeError):
        build()


def test_positional_arguments_are_refused():
    @stage_ref("train_model")
    def train(tier: int): ...

    @workflow("w")
    def build():
        train(1)

    with pytest.raises(TypeError, match="by keyword"):
        build()


def test_a_raising_workflow_does_not_leak_its_builder():
    @stage_ref("a")
    def thing(): ...

    @workflow("boom")
    def bad():
        thing()
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        bad()
    # If the builder leaked, this call would land in the dead workflow's
    # builder instead of raising.
    with pytest.raises(RuntimeError, match="no active workflow"):
        thing()


def test_current_builder_is_the_active_workflow():
    @workflow("w")
    def build():
        assert current_builder().name == "w"
        current_builder().add("a")

    assert len(build().nodes) == 1


# -- Run -----------------------------------------------------------------


def _spec() -> PlanSpec:
    b = PlanBuilder("p")
    b.add("a")
    return b.build()


def test_a_missing_binary_explains_the_exec_bridge():
    run = Run(_spec(), blut="definitely-not-a-real-binary-xyz")
    with pytest.raises(RunError, match="not on PATH"):
        run.declare()


def test_status_before_submit_is_refused():
    with pytest.raises(RunError, match="not been submitted"):
        Run(_spec()).status()


def test_declare_surfaces_the_engine_stderr(tmp_path):
    fake = tmp_path / "blut"
    fake.write_text("#!/bin/sh\necho 'kind mismatch at node 2' >&2\nexit 3\n")
    fake.chmod(0o755)
    run = Run(_spec(), blut=str(fake))
    with pytest.raises(RunError, match="kind mismatch at node 2"):
        run.declare()


def test_declare_writes_a_plan_the_engine_can_read(tmp_path):
    seen = tmp_path / "seen.json"
    fake = tmp_path / "blut"
    fake.write_text(f'#!/bin/sh\ncp "$3" {seen}\nexit 0\n')
    fake.chmod(0o755)
    Run(_spec(), blut=str(fake)).declare()
    written = json.loads(seen.read_text())
    assert written["name"] == "p"
    assert written["nodes"][0]["stage"] == "a"


def test_the_temp_plan_file_does_not_outlive_the_call(tmp_path):
    captured = tmp_path / "path.txt"
    fake = tmp_path / "blut"
    fake.write_text(f'#!/bin/sh\necho "$3" > {captured}\nexit 0\n')
    fake.chmod(0o755)
    Run(_spec(), blut=str(fake)).declare()
    import pathlib

    assert not pathlib.Path(captured.read_text().strip()).exists()


def test_submit_without_a_job_id_in_the_output_is_an_error(tmp_path):
    # Launching and then being unable to monitor is worse than a clean failure:
    # the job is running and the caller has no handle to it.
    fake = tmp_path / "blut"
    fake.write_text("#!/bin/sh\necho 'launched, good luck'\nexit 0\n")
    fake.chmod(0o755)
    with pytest.raises(RunError, match="no job id"):
        Run(_spec(), blut=str(fake)).submit()


def test_submit_extracts_the_job_id(tmp_path):
    fake = tmp_path / "blut"
    fake.write_text("#!/bin/sh\necho 'started job 20260619-082149-230733595'\nexit 0\n")
    fake.chmod(0o755)
    run = Run(_spec(), blut=str(fake))
    assert run.submit() == "20260619-082149-230733595"


def test_status_reads_the_job_row(tmp_path):
    rows = json.dumps([{"id": "20260619-082149-230733595", "state": "done", "final_loss": 1.5}])
    fake = tmp_path / "blut"
    fake.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = jobs ]; then\n"
        f"  echo '{rows}'\n"
        "else\n"
        "  echo 'started job 20260619-082149-230733595'\n"
        "fi\n"
        "exit 0\n"
    )
    fake.chmod(0o755)
    run = Run(_spec(), blut=str(fake))
    run.submit()
    state = run.status()
    assert state.state == "done" and state.finished and state.final_loss == 1.5


def test_events_tolerates_a_partial_trailing_line(tmp_path, monkeypatch):
    monkeypatch.setenv("LAMU_TRAIN_JOBS_DIR", str(tmp_path))
    job = tmp_path / "20260619-082149-230733595"
    job.mkdir()
    # The engine appends this file live, so the last line can be half-written.
    (job / "status.jsonl").write_text('{"kind":"a"}\n{"kind":"b"}\n{"kind":"par')
    run = Run(_spec())
    run.job_id = "20260619-082149-230733595"
    assert [e["kind"] for e in run.events()] == ["a", "b"]


def test_status_path_does_not_create_the_directory(tmp_path, monkeypatch):
    # The engine's own `job_dir` creates on demand; an SDK that did the same
    # would make a typo'd id look like a real job with no events yet.
    monkeypatch.setenv("LAMU_TRAIN_JOBS_DIR", str(tmp_path))
    run = Run(_spec())
    run.job_id = "20260101-000000-000000001"
    path = run.status_path()
    assert not path.parent.exists()
    assert run.events() == []
