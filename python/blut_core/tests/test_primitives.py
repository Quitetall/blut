"""Tests for the blut_core primitives: runctx, status, sysgauge, checkpoint.

Contracts pinned: runctx resolves BLUT_JOB_DIR-or-fallback; status emits valid
kinds + appends status.jsonl + rejects bad kinds; sysgauge never fabricates and
never raises; checkpoint is atomic + SHA-validated + enforces the resume contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # blut/python
from blut_core import runctx, status, sysgauge, checkpoint  # noqa: E402


# ---- runctx ----
def test_runctx_job_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BLUT_JOB_DIR", str(tmp_path))
    assert runctx.job_dir() == tmp_path
    monkeypatch.delenv("BLUT_JOB_DIR", raising=False)
    assert runctx.job_dir(tmp_path / "fb") == tmp_path / "fb"          # explicit fallback
    assert runctx.job_dir().name == "training_logs"                    # repo default


def test_runctx_resolve(monkeypatch, tmp_path):
    monkeypatch.setenv("BLUT_JOB_DIR", str(tmp_path))
    monkeypatch.setenv("BLUT_STAGE_NAME", "snn_train")
    monkeypatch.setenv("BLUT_CONTAINED", "1")
    ctx = runctx.resolve("run42")
    assert ctx.run_id == "run42" and ctx.job_dir == tmp_path
    assert ctx.stage_name == "snn_train" and ctx.contained is True
    with pytest.raises(ValueError):
        runctx.resolve("")


# ---- status ----
def test_status_emit_writes_jsonl_and_returns(tmp_path, capsys):
    rec = status.step(1, 100, loss=5.0, lr=1e-3, vram_mb=8000, job_dir=tmp_path)
    assert rec["kind"] == "step" and rec["loss"] == 5.0
    out = capsys.readouterr().out.strip()
    assert json.loads(out)["kind"] == "step"                          # flushed to stdout
    lines = (tmp_path / "status.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["step"] == 1                          # appended


def test_status_eval_uses_eval_kind_and_bad_kind_rejected(tmp_path):
    rec = status.eval_pass(5, eval_loss=4.2, job_dir=tmp_path, to_stdout=False)
    assert rec["kind"] == "eval" and rec["eval_loss"] == 4.2
    with pytest.raises(ValueError):
        status.emit("telemetry", job_dir=tmp_path, to_stdout=False)   # not a valid kind


def test_status_read_skips_malformed(tmp_path):
    p = tmp_path / "status.jsonl"
    p.write_text(json.dumps({"kind": "step", "step": 1}) + "\n{bad\n"
                 + json.dumps({"kind": "done", "final_loss": 0.1}) + "\n")
    recs = status.read(p)
    assert [r["kind"] for r in recs] == ["step", "done"]


# ---- sysgauge ----
def test_sysgauge_returns_dict_never_raises():
    snap = sysgauge.snapshot()
    assert isinstance(snap, dict)
    # On Linux the host gauges are available; all values are floats (no fabricated strings).
    assert all(isinstance(v, float) for v in snap.values())


# ---- checkpoint ----
def test_checkpoint_validate_payload():
    assert checkpoint.validate_payload({"model": 1, "optimizer": 2, "lr_scheduler": 3,
                                        "rng_state": 4, "step": 5}) == []
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.validate_payload({"model": 1}, strict=True)
    assert "step" in checkpoint.validate_payload({"model": 1}, strict=False)


def test_checkpoint_roundtrip_and_corruption(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "ckpt.pt"
    payload = {"model": {"w": torch.tensor([1.0, 2.0])}, "step": 7}
    checkpoint.save(payload, path)
    assert path.exists() and path.with_suffix(".pt.sha256").exists()
    loaded = checkpoint.load(path)
    assert loaded["step"] == 7
    # corrupt the file → SHA mismatch → CheckpointError (not silent garbage)
    path.write_bytes(b"garbage")
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.load(path)
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.load(tmp_path / "absent.pt")


def test_checkpoint_contract_enforced_on_save(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.save({"model": 1}, tmp_path / "c.pt",
                        contract=checkpoint.RESUME_CONTRACT)
