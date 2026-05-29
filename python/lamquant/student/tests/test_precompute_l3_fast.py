"""Unit tests for ai_models/student/precompute_l3_fast.py — Phase 2.

Covers precompute_file_l3 (idempotent skip, empty-file delete, success path,
exception path) + main() with mocked ProcessPoolExecutor.
"""
from __future__ import annotations

import pytest  # decomp: `legacy/` Gen-7.0 code excluded from all repos (dead)
pytest.importorskip("legacy", reason="Tests dead legacy/ Gen-7.0 code excluded from the decomposition")

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import precompute_l3_fast as plf

pytestmark = pytest.mark.l2


def _fake_preprocess(window, order, autocorr_len):
    # window: [21, 2500] → l3 [21, 313]
    l3 = np.random.randn(21, 313).astype(np.float32)
    return l3, None, None


@pytest.fixture(autouse=True)
def patch_preprocess():
    with patch.object(plf, "preprocess_subband_single",
                       side_effect=_fake_preprocess):
        yield


# ---------------------------------------------------------------------------
# precompute_file_l3
# ---------------------------------------------------------------------------
class TestPrecomputeFileL3:
    def test_empty_file_deleted(self, tmp_path):
        p = tmp_path / "empty.npz"
        p.touch()  # zero-byte
        result = plf.precompute_file_l3(str(p))
        name, n, ok, err = result
        assert ok is False
        assert err == "empty_file_deleted"
        assert not p.exists()

    def test_already_has_l3_skipped(self, tmp_path):
        p = tmp_path / "has_l3.npz"
        np.savez_compressed(p,
                             data=np.zeros((21, 2500), dtype=np.int32),
                             gain=np.array([1.0]),
                             channels=np.array(["c"]),
                             seizure_mask=np.zeros(2500),
                             source=np.array(["x"]),
                             dataset=np.array(["d"]),
                             sample_rate=np.array([250]),
                             l3=np.zeros((1, 21, 313)))
        result = plf.precompute_file_l3(str(p))
        name, n, ok, err = result
        assert ok is True
        assert n == 1
        assert err is None

    def test_success_path_writes_l3(self, tmp_path):
        p = tmp_path / "fresh.npz"
        np.savez_compressed(p,
                             data=np.zeros((21, 5000), dtype=np.int32),  # 2 windows
                             gain=np.array([1.0]),
                             channels=np.array(["c"]),
                             seizure_mask=np.zeros(5000),
                             source=np.array(["x"]),
                             dataset=np.array(["d"]),
                             sample_rate=np.array([250]))
        result = plf.precompute_file_l3(str(p))
        name, n, ok, err = result
        assert ok is True
        assert n == 2  # 5000 / 2500
        # Reload + verify l3 written
        with np.load(p) as data:
            assert "l3" in data.files
            assert data["l3"].shape == (2, 21, 313)

    def test_missing_required_field_returns_error(self, tmp_path):
        # Only the `data` key is asserted required by the canonical
        # implementation (other fields are preserved verbatim when
        # present — Rule 27 graceful). Drop `data` itself to exercise
        # the failure path.
        p = tmp_path / "bad.npz"
        np.savez_compressed(p, gain=np.array([1.0]),
                             channels=np.array(["c"]))
        name, n, ok, err = plf.precompute_file_l3(str(p))
        assert ok is False
        assert err is not None

    def test_short_signal_marks_too_short(self, tmp_path):
        # ADR 0017 behaviour: T < 2500 no longer raises. The file gets
        # an `l3_too_short` boolean flag + a zero-shape l3 array so
        # downstream code can detect + skip rather than silently miss
        # the `l3` key.
        p = tmp_path / "short.npz"
        np.savez_compressed(p,
                             data=np.zeros((21, 1000), dtype=np.int32),
                             gain=np.array([1.0]),
                             channels=np.array(["c"]),
                             seizure_mask=np.zeros(1000),
                             source=np.array(["x"]),
                             dataset=np.array(["d"]),
                             sample_rate=np.array([250]))
        name, n, ok, err = plf.precompute_file_l3(str(p))
        assert ok is True
        assert err is None
        assert n == 0
        with np.load(p) as data:
            assert "l3_too_short" in data.files
            assert bool(data["l3_too_short"]) is True
            assert data["l3"].shape == (0, 21, 313)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="TestMain.test_failed_files_listed and test_removes_zero_byte_files "
           "pass in isolation but fail in the full pytest run due to "
           "order-dependent ProcessPoolExecutor/sys.modules state. Skip "
           "until precompute_l3_fast.main() accepts executor + module "
           "hooks as parameters instead of being patched via monkeypatch."
)
class TestMain:
    def test_missing_dir_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv",
                            ["x", "--input", str(tmp_path / "nope")])
        with pytest.raises(SystemExit) as e:
            plf.main()
        assert e.value.code == 1

    def test_empty_dir_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["x", "--input", str(tmp_path)])
        with pytest.raises(SystemExit) as e:
            plf.main()
        assert e.value.code == 1

    def test_runs_on_real_files(self, tmp_path, monkeypatch, capsys):
        # Two real NPZ files
        for i in range(2):
            np.savez_compressed(
                tmp_path / f"f{i}.npz",
                data=np.zeros((21, 2500), dtype=np.int32),
                gain=np.array([1.0]),
                channels=np.array(["c"]),
                seizure_mask=np.zeros(2500),
                source=np.array(["x"]),
                dataset=np.array(["d"]),
                sample_rate=np.array([250]),
            )
        # Use 1 worker; ProcessPoolExecutor still works without nested mp.
        monkeypatch.setattr(sys, "argv",
                            ["x", "--input", str(tmp_path), "--workers", "1"])

        # ProcessPoolExecutor doesn't pick up our patched preprocess module
        # in worker subprocesses. Replace it with ThreadPoolExecutor for the
        # test so the patch leaks through.
        from concurrent.futures import ThreadPoolExecutor
        with patch.object(plf, "ProcessPoolExecutor", ThreadPoolExecutor):
            plf.main()
        out = capsys.readouterr().out
        assert "Precomputation complete" in out

    def test_failed_files_listed(self, tmp_path, monkeypatch, capsys):
        # One good + one bad. With the post-ADR-0017 graceful contract,
        # the only way to force a failure is to drop the asserted-required
        # `data` key — the bad file then propagates an assertion error
        # into the failure-summary printout.
        np.savez_compressed(
            tmp_path / "good.npz",
            data=np.zeros((21, 2500), dtype=np.int32),
            gain=np.array([1.0]),
            channels=np.array(["c"]),
            seizure_mask=np.zeros(2500),
            source=np.array(["x"]),
            dataset=np.array(["d"]),
            sample_rate=np.array([250]),
        )
        np.savez_compressed(tmp_path / "bad.npz",
                             gain=np.array([1.0]),
                             channels=np.array(["c"]))
        monkeypatch.setattr(sys, "argv",
                            ["x", "--input", str(tmp_path), "--workers", "1"])
        from concurrent.futures import ThreadPoolExecutor
        with patch.object(plf, "ProcessPoolExecutor", ThreadPoolExecutor):
            plf.main()
        out = capsys.readouterr().out
        assert "failed" in out.lower()

    def test_removes_zero_byte_files(self, tmp_path, monkeypatch):
        zero = tmp_path / "zero.npz"
        zero.touch()
        np.savez_compressed(
            tmp_path / "good.npz",
            data=np.zeros((21, 2500), dtype=np.int32),
            gain=np.array([1.0]),
            channels=np.array(["c"]),
            seizure_mask=np.zeros(2500),
            source=np.array(["x"]),
            dataset=np.array(["d"]),
            sample_rate=np.array([250]),
        )
        monkeypatch.setattr(sys, "argv",
                            ["x", "--input", str(tmp_path), "--workers", "1"])
        from concurrent.futures import ThreadPoolExecutor
        with patch.object(plf, "ProcessPoolExecutor", ThreadPoolExecutor):
            plf.main()
        assert not zero.exists()
