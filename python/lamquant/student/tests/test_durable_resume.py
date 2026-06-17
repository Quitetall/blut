"""Unit tests for the durable-resume TRAINER half (durable_resume.DurableResume).

CPU-only, no full trainer / GPU needed — exercises the checkpoint mechanics +
the state.json marker the Rust orchestrator (`blut::framework::resume`) reads.
"""

import json
import time

import numpy as np
import pytest
import torch
import torch.nn as nn

from durable_resume import (
    HEARTBEAT_INTERVAL,
    REQUIRED_RECOVERY_KEYS,
    DurableResume,
)


def _read_state(d):
    return json.loads((d / "state.json").read_text())


def _complete_payload(epoch=1, phase="warm"):
    """A structurally COMPLETE recovery payload (every unconditional key the
    trainer reads on resume). `save_recovery` adds `rng` + `resume_key`, so the
    caller need only supply encoder/decoder/optimizer/epoch/phase."""
    return {
        "encoder": {},
        "decoder": {},
        "optimizer": {},
        "epoch": epoch,
        "phase": phase,
    }


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
    dur.save_recovery("warm_latest", _complete_payload(epoch=1))  # becomes prev on next save
    dur.save_recovery("warm_latest", _complete_payload(epoch=2))  # latest=ep2, prev=ep1
    # Corrupt the latest (a kill mid-write).
    (tmp_path / "warm_latest.ckpt").write_bytes(b"\x00\x00 not a torch file")
    ck = dur.load_recovery("warm_latest")
    assert ck is not None and ck["epoch"] == 1, "must fall back to the prev rotation"


def test_foreign_resume_key_is_rejected(tmp_path):
    dur = DurableResume(tmp_path, resume_key="mine")
    dur.save_recovery("warm_latest", _complete_payload(epoch=3))
    # A loader keyed on a DIFFERENT config must not load this checkpoint.
    other = DurableResume(tmp_path, resume_key="theirs")
    assert other.load_recovery("warm_latest") is None


def test_detect_prefers_qat_over_warm(tmp_path):
    dur = DurableResume(tmp_path, resume_key="k")
    assert dur.detect() is None
    dur.save_recovery("warm_latest", _complete_payload(epoch=1, phase="warm"))
    assert dur.detect() == "warm_latest"
    dur.save_recovery("qat_latest", _complete_payload(epoch=2, phase="qat"))
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


# ---------------------------------------------------------------------------
# Phase D hardening: required-keys validation + validate-on-load + preflight
# ---------------------------------------------------------------------------


def test_required_keys_is_the_verified_unconditional_set():
    # Single source of truth — these are exactly the keys train_joint.py reads
    # unconditionally on resume (encoder/decoder/epoch/phase/optimizer) plus the
    # two save_recovery always embeds (rng/resume_key).
    assert set(REQUIRED_RECOVERY_KEYS) == {
        "encoder",
        "decoder",
        "epoch",
        "phase",
        "optimizer",
        "rng",
        "resume_key",
    }


def test_complete_checkpoint_loads_normally(tmp_path):
    # Happy path is byte-identical to before: a complete checkpoint loads.
    dur = DurableResume(tmp_path, resume_key="k")
    dur.save_recovery("warm_latest", _complete_payload(epoch=5, phase="warm"))
    ck = dur.load_recovery("warm_latest")
    assert ck is not None
    assert ck["epoch"] == 5 and ck["phase"] == "warm"
    assert all(key in ck for key in REQUIRED_RECOVERY_KEYS)


def test_incomplete_latest_falls_back_to_complete_prev(tmp_path):
    # A complete prev + an INCOMPLETE latest (missing optimizer, e.g. a
    # half-written kill) → load rejects the latest and returns the good prev.
    dur = DurableResume(tmp_path, resume_key="k")
    dur.save_recovery("warm_latest", _complete_payload(epoch=1))  # latest=ep1
    dur.save_recovery("warm_latest", _complete_payload(epoch=1))  # ep1→prev, latest=ep1
    assert (tmp_path / "warm_latest.prev.ckpt").exists()
    # Overwrite the latest with an incomplete object (bypass save_recovery's
    # key-injection) — a torn write that still happens to be a loadable torch
    # object, missing `optimizer` + `rng`. The good prev must catch it.
    torn = {"encoder": {}, "decoder": {}, "epoch": 2, "phase": "warm", "resume_key": "k"}
    torch.save(torn, tmp_path / "warm_latest.ckpt")
    ck = dur.load_recovery("warm_latest")
    assert ck is not None and ck["epoch"] == 1, "incomplete latest must fall to the complete prev"
    assert "optimizer" in ck


def test_both_incomplete_raises_not_silent_coldstart(tmp_path):
    # BOTH latest and prev are ours-but-incomplete → load_recovery RAISES rather
    # than returning a keyless dict (a silent cold-optimizer resume is worse than
    # a loud refusal).
    dur = DurableResume(tmp_path, resume_key="k")
    incomplete = {"encoder": {}, "epoch": 1, "phase": "warm", "resume_key": "k"}  # no optimizer/decoder/rng
    torch.save(incomplete, tmp_path / "warm_latest.ckpt")
    torch.save(incomplete, tmp_path / "warm_latest.prev.ckpt")
    with pytest.raises(RuntimeError, match="missing required key"):
        dur.load_recovery("warm_latest")


def test_missing_optimizer_alone_is_rejected(tmp_path):
    # The exact audit scenario: a checkpoint missing ONLY `optimizer` (everything
    # else present) is still rejected — that single absence is what cold-starts
    # SOAP/Adam. With no prev to fall back to, it raises.
    dur = DurableResume(tmp_path, resume_key="k")
    ck = dict(_complete_payload(epoch=4))
    ck["rng"] = DurableResume.capture_rng()
    ck["resume_key"] = "k"
    del ck["optimizer"]
    torch.save(ck, tmp_path / "warm_latest.ckpt")
    with pytest.raises(RuntimeError, match="optimizer"):
        dur.load_recovery("warm_latest")


def test_readable_foreign_key_only_returns_none_not_raise(tmp_path):
    # A READABLE checkpoint with a foreign resume_key is a different-config
    # leftover, not a crash to resume — load_recovery returns None (start fresh),
    # never raises, even though it is the only candidate on disk.
    owner = DurableResume(tmp_path, resume_key="mine")
    owner.save_recovery("warm_latest", _complete_payload(epoch=9))  # complete, key=mine
    other = DurableResume(tmp_path, resume_key="theirs")
    assert other.load_recovery("warm_latest") is None  # must NOT raise


def test_malformed_min_free_env_falls_back_to_default(tmp_path, monkeypatch):
    # A typo'd RECOVERY_MIN_FREE_GB must not crash save_recovery mid-run — it
    # falls back to the default margin (fail-safe, matches _free_bytes's
    # fail-closed contract). Ample disk so the (default) preflight passes.
    import durable_resume as dr

    class _AmpleStat:
        f_bavail = 10**9
        f_frsize = 4096

    monkeypatch.setattr(dr.os, "statvfs", lambda _p: _AmpleStat())
    monkeypatch.setenv("RECOVERY_MIN_FREE_GB", "not-a-number")
    dur = DurableResume(tmp_path, resume_key="k")
    dur.save_recovery("warm_latest", _complete_payload(epoch=1))  # must not raise
    assert dur.load_recovery("warm_latest")["epoch"] == 1


@pytest.mark.parametrize("bad", ["not-a-number", "inf", "-inf", "nan", "-5", "1e400"])
def test_recovery_min_free_bytes_never_raises_on_bad_env(monkeypatch, bad):
    # The byte-floor helper must NEVER raise — not on garbage, not on inf
    # (int(inf*1e9) would OverflowError), not on NaN/negative — it falls back to
    # the default margin. This is the load-bearing fail-safe for the hot save path.
    import durable_resume as dr

    monkeypatch.setenv("RECOVERY_MIN_FREE_GB", bad)
    val = dr._recovery_min_free_bytes()
    assert val == int(dr.RECOVERY_MIN_FREE_GB_DEFAULT * 1e9)


def test_recovery_min_free_bytes_honors_valid_env(monkeypatch):
    # A valid override scales GB → bytes via the documented 1e9 idiom.
    import durable_resume as dr

    monkeypatch.setenv("RECOVERY_MIN_FREE_GB", "8")
    assert dr._recovery_min_free_bytes() == int(8 * 1e9)


def test_no_recovery_files_returns_none(tmp_path):
    # A genuinely empty recovery dir (first run, no crash) still returns None so
    # the trainer starts fresh — the raise is ONLY for present-but-broken.
    dur = DurableResume(tmp_path, resume_key="k")
    assert dur.load_recovery("warm_latest") is None


def test_low_disk_skips_save_and_preserves_prev(tmp_path, monkeypatch):
    # Free-space preflight: with the disk reporting near-zero free, save_recovery
    # must SKIP the write (+ warn) and leave the existing prev rotation intact —
    # never truncate a `.tmp` that os.replace would publish over the good prev.
    dur = DurableResume(tmp_path, resume_key="k")
    # Establish a good latest (ep1) + a second save so a prev exists (ep1→prev).
    dur.save_recovery("warm_latest", _complete_payload(epoch=1))
    dur.save_recovery("warm_latest", _complete_payload(epoch=2))  # latest=ep2, prev=ep1
    prev = tmp_path / "warm_latest.prev.ckpt"
    latest = tmp_path / "warm_latest.ckpt"
    assert prev.exists() and latest.exists()
    prev_bytes_before = prev.read_bytes()
    latest_bytes_before = latest.read_bytes()

    # Mock statvfs to report ~0 free (f_bavail=0). Patch on the durable_resume
    # module's os reference so only THIS save sees the full disk.
    import durable_resume as dr

    class _FullStat:
        f_bavail = 0
        f_frsize = 4096

    monkeypatch.setattr(dr.os, "statvfs", lambda _p: _FullStat())

    with pytest.warns(RuntimeWarning, match="skipped"):
        dur.save_recovery("warm_latest", _complete_payload(epoch=3))

    # The write was skipped: latest is STILL ep2, and the prev (ep1) is untouched
    # — no rotation occurred, the good chain survives.
    assert prev.read_bytes() == prev_bytes_before, "prev rotation must be left intact"
    assert latest.read_bytes() == latest_bytes_before, "latest must be unchanged (no truncated publish)"
    assert not (tmp_path / "warm_latest.ckpt.tmp").exists(), "no stray .tmp left behind"
    # load_recovery does not call statvfs, so the still-mocked full disk is
    # irrelevant here — the good ep2 latest is read back unchanged.
    assert dur.load_recovery("warm_latest")["epoch"] == 2


def test_ample_disk_save_writes_and_rotates_as_before(tmp_path, monkeypatch):
    # With the preflight reporting ample free space, save behaves byte-identically
    # to the pre-guard path: it writes the latest and rotates the prior to prev.
    dur = DurableResume(tmp_path, resume_key="k")
    import durable_resume as dr

    class _AmpleStat:
        f_bavail = 10**9  # ~4 PB at 4K blocks — comfortably over any margin
        f_frsize = 4096

    monkeypatch.setattr(dr.os, "statvfs", lambda _p: _AmpleStat())
    dur.save_recovery("warm_latest", _complete_payload(epoch=1))
    dur.save_recovery("warm_latest", _complete_payload(epoch=2))
    assert (tmp_path / "warm_latest.ckpt").exists()
    assert (tmp_path / "warm_latest.prev.ckpt").exists()
    assert dur.load_recovery("warm_latest")["epoch"] == 2
    # prev carries the earlier epoch (rotation happened).
    prev = torch.load(tmp_path / "warm_latest.prev.ckpt", weights_only=False)
    assert prev["epoch"] == 1


def test_save_warns_on_incomplete_payload_but_still_writes(tmp_path, monkeypatch):
    # The save-side completeness check is a non-fatal WARNING (a save hiccup must
    # never crash training); the hard fail-fast is on load. Ample disk so the
    # preflight doesn't also fire.
    dur = DurableResume(tmp_path, resume_key="k")
    import durable_resume as dr

    class _AmpleStat:
        f_bavail = 10**9
        f_frsize = 4096

    monkeypatch.setattr(dr.os, "statvfs", lambda _p: _AmpleStat())
    with pytest.warns(RuntimeWarning, match="missing required key"):
        dur.save_recovery("warm_latest", {"epoch": 1, "phase": "warm"})  # no encoder/decoder/optimizer
    # It still wrote the file (best-effort), but load will reject it (no prev).
    assert (tmp_path / "warm_latest.ckpt").exists()
    with pytest.raises(RuntimeError):
        dur.load_recovery("warm_latest")
