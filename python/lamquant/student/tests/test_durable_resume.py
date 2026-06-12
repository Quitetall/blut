"""Unit tests for the durable-resume TRAINER half (durable_resume.DurableResume).

CPU-only, no full trainer / GPU needed — exercises the checkpoint mechanics +
the state.json marker the Rust orchestrator (`blut::framework::resume`) reads.
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn

from durable_resume import HEARTBEAT_INTERVAL, DurableResume


def _read_state(d):
    return json.loads((d / "state.json").read_text())


def test_recovery_roundtrip_restores_optimizer_epoch_rng(tmp_path):
    dur = DurableResume(tmp_path, run_id="run-A", resume_key="cfgkey1")
    model = nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Take a step so the optimizer accrues state (Adam moment buffers).
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()
    payload = {
        "encoder": model.state_dict(),
        "decoder": {},
        "optimizer": opt.state_dict(),
        "epoch": 7,
        "phase": "warm",
    }
    dur.save_recovery("warm_latest", payload)

    # A fresh process would build a fresh optimizer and load the saved state.
    model2 = nn.Linear(4, 4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    ck = dur.load_recovery("warm_latest")
    assert ck is not None
    assert ck["epoch"] == 7
    assert ck["phase"] == "warm"
    assert ck["resume_key"] == "cfgkey1"
    model2.load_state_dict(ck["encoder"])
    opt2.load_state_dict(ck["optimizer"])
    # The Adam step-count carried over (optimizer state truly restored).
    st = next(iter(opt2.state.values()))
    assert int(st["step"]) >= 1
    # Weights round-tripped exactly.
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)
    # RNG blob present.
    assert "rng" in ck and "torch" in ck["rng"]


def test_rng_restore_reproduces_stream():
    rng = DurableResume.capture_rng()
    a_py = [__import__("random").random() for _ in range(3)]
    a_np = np.random.rand(3)
    a_t = torch.rand(3)
    DurableResume.restore_rng(rng)
    b_py = [__import__("random").random() for _ in range(3)]
    b_np = np.random.rand(3)
    b_t = torch.rand(3)
    assert a_py == b_py
    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_t, b_t)


def test_corrupt_latest_falls_back_to_prev(tmp_path):
    dur = DurableResume(tmp_path, resume_key="k")
    dur.save_recovery("warm_latest", {"epoch": 1, "phase": "warm"})  # becomes prev on next save
    dur.save_recovery("warm_latest", {"epoch": 2, "phase": "warm"})  # latest=ep2, prev=ep1
    # Corrupt the latest (a kill mid-write).
    (tmp_path / "warm_latest.ckpt").write_bytes(b"\x00\x00 not a torch file")
    ck = dur.load_recovery("warm_latest")
    assert ck is not None and ck["epoch"] == 1, "must fall back to the prev rotation"


def test_foreign_resume_key_is_rejected(tmp_path):
    dur = DurableResume(tmp_path, resume_key="mine")
    dur.save_recovery("warm_latest", {"epoch": 3, "phase": "warm"})
    # A loader keyed on a DIFFERENT config must not load this checkpoint.
    other = DurableResume(tmp_path, resume_key="theirs")
    assert other.load_recovery("warm_latest") is None


def test_detect_prefers_qat_over_warm(tmp_path):
    dur = DurableResume(tmp_path, resume_key="k")
    assert dur.detect() is None
    dur.save_recovery("warm_latest", {"epoch": 1, "phase": "warm"})
    assert dur.detect() == "warm_latest"
    dur.save_recovery("qat_latest", {"epoch": 2, "phase": "qat"})
    assert dur.detect() == "qat_latest"


def test_state_marker_running_then_finished(tmp_path):
    dur = DurableResume(tmp_path, run_id="run-XYZ", resume_key="k")
    dur.start()
    s = _read_state(tmp_path)
    assert s["status"] == "running"
    assert s["run_id"] == "run-XYZ"
    assert s["pid"] > 0
    assert abs(s["heartbeat_unix"] - int(time.time())) < 5
    dur.finish()
    assert _read_state(tmp_path)["status"] == "finished"


def test_heartbeat_constant_matches_rust_contract():
    # The Python heartbeat cadence must match the Rust resume policy's constant
    # (blut resume::HEARTBEAT_INTERVAL_SECS = 60; stale window = 3×).
    assert HEARTBEAT_INTERVAL == 60
