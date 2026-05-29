"""Unit tests for ai_models/student/launch_production.py — Phase 3.

preflight + launch + main(). All heavy I/O (data_types, /mnt/4tb,
torch.cuda, subprocess) stubbed for CPU-only smoke testing.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import launch_production as lp

pytestmark = pytest.mark.l2


@pytest.fixture
def fake_args():
    class _Args:
        config = "production"
        tier = 7
        fullband_mode = "memmap"
        foreground = False
        preflight_only = False
        skip_preflight = False
    return _Args()


# ---------------------------------------------------------------------------
# _check helper
# ---------------------------------------------------------------------------
class TestCheck:
    def test_returns_true_on_ok(self, capsys):
        assert lp._check("foo", True, "ok") is True
        out = capsys.readouterr().out
        assert "✓" in out
        assert "foo" in out

    def test_returns_false_on_not_ok(self, capsys):
        assert lp._check("foo", False) is False
        out = capsys.readouterr().out
        assert "✗" in out


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
class TestPreflight:
    def test_manifest_load_error_aborts(self, fake_args, monkeypatch):
        fake_dt = types.ModuleType("data_types")
        class _BadManifest:
            @classmethod
            def load(cls, _p):
                raise RuntimeError("schema fail")
        fake_dt.DatasetManifest = _BadManifest
        fake_dt.Split = MagicMock()
        monkeypatch.setitem(sys.modules, "data_types", fake_dt)
        assert lp.preflight(fake_args) is False

    def test_full_path_passes_when_all_ok(self, fake_args, monkeypatch, tmp_path):
        # Patch everything so preflight completes
        fake_dt = types.ModuleType("data_types")
        class _Manifest:
            train_files = 100
            val_files = 20
            train_windows = 200
            val_windows = 40
            @classmethod
            def load(cls, _p): return cls()
        fake_dt.DatasetManifest = _Manifest
        fake_dt.Split = MagicMock()
        monkeypatch.setitem(sys.modules, "data_types", fake_dt)

        # Create fake fullband files
        ds_dir = lp._REPO / "ai_models" / "dataset_sim"
        # Use real path; create temp files
        train_dat = ds_dir / "fullband_train.dat"
        train_meta = ds_dir / "fullband_train.meta.json"
        val_dat = ds_dir / "fullband_val.dat"
        val_meta = ds_dir / "fullband_val.meta.json"
        # Skip if real path is not writable
        try:
            ds_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pytest.skip("cannot write to dataset_sim")

        # Preserve any pre-existing real files
        backup = {}
        for p in (train_dat, train_meta, val_dat, val_meta):
            if p.exists():
                backup[p] = p.read_bytes()

        try:
            # 200 windows × 21 channels × 2500 samples × 2 bytes
            train_size = 200 * 21 * 2500 * 2
            val_size = 40 * 21 * 2500 * 2
            train_dat.write_bytes(b"\x00" * train_size)
            train_meta.write_text(json.dumps({
                "n_windows": 200, "n_channels": 21, "window_samples": 2500
            }))
            val_dat.write_bytes(b"\x00" * val_size)
            val_meta.write_text(json.dumps({
                "n_windows": 40, "n_channels": 21, "window_samples": 2500
            }))

            # Stub disk_usage + torch.cuda + psutil
            with patch.object(lp.shutil, "disk_usage") as mock_disk, \
                 patch("psutil.virtual_memory") as mock_vm:
                mock_disk.return_value = MagicMock(free=100 * 1024**3)
                mock_vm.return_value = MagicMock(available=64 * 1024**3)

                # Patch torch.cuda
                import torch
                with patch.object(torch.cuda, "is_available", return_value=True), \
                     patch.object(torch.cuda, "get_device_properties") as mp, \
                     patch.object(torch.cuda, "mem_get_info",
                                   return_value=(20 * 1024**3, 24 * 1024**3),
                                   create=True), \
                     patch.object(torch.cuda, "get_device_name",
                                   return_value="MockGPU"):
                    mp.return_value = MagicMock(total_memory=24 * 1024**3)
                    ok = lp.preflight(fake_args)
                    assert ok is True
        finally:
            # Cleanup or restore
            for p in (train_dat, train_meta, val_dat, val_meta):
                if p.exists():
                    p.unlink()
                if p in backup:
                    p.write_bytes(backup[p])

    def test_no_cuda_fails(self, fake_args, monkeypatch):
        fake_dt = types.ModuleType("data_types")
        class _M:
            train_files = val_files = 1
            train_windows = val_windows = 1
            @classmethod
            def load(cls, _p): return cls()
        fake_dt.DatasetManifest = _M
        fake_dt.Split = MagicMock()
        monkeypatch.setitem(sys.modules, "data_types", fake_dt)

        import torch
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(lp.shutil, "disk_usage",
                           return_value=MagicMock(free=100 * 1024**3)), \
             patch("psutil.virtual_memory",
                    return_value=MagicMock(available=64 * 1024**3)):
            ok = lp.preflight(fake_args)
            assert ok is False  # no CUDA → fail


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------
class TestLaunch:
    def test_foreground_invokes_subprocess_call(self, fake_args, tmp_path):
        fake_args.foreground = True
        with patch.object(lp.subprocess, "call", return_value=0) as call_mock, \
             patch.object(lp, "_REPO", tmp_path):
            (tmp_path / "outputs").mkdir(exist_ok=True)
            (tmp_path / "ai_models" / "student").mkdir(parents=True, exist_ok=True)
            (tmp_path / "ai_models" / "student" / "train_joint.py").touch()
            rc = lp.launch(fake_args)
            assert rc == 0
            call_mock.assert_called_once()

    def test_background_invokes_popen(self, fake_args, tmp_path):
        fake_args.foreground = False
        mock_proc = MagicMock(pid=12345)
        with patch.object(lp.subprocess, "Popen", return_value=mock_proc) as popen, \
             patch.object(lp, "_REPO", tmp_path):
            (tmp_path / "outputs").mkdir(exist_ok=True)
            (tmp_path / "ai_models" / "student").mkdir(parents=True, exist_ok=True)
            (tmp_path / "ai_models" / "student" / "train_joint.py").touch()
            rc = lp.launch(fake_args)
            assert rc == 0
            popen.assert_called_once()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
class TestMain:
    def test_preflight_fail_returns_1(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lp"])
        with patch.object(lp, "preflight", return_value=False):
            assert lp.main() == 1

    def test_preflight_only_returns_0(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lp", "--preflight-only"])
        with patch.object(lp, "preflight", return_value=True):
            assert lp.main() == 0

    def test_skip_preflight_launches(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lp", "--skip-preflight"])
        with patch.object(lp, "launch", return_value=0) as launch_mock:
            assert lp.main() == 0
            launch_mock.assert_called_once()

    def test_full_pipeline_preflight_passes_then_launch(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lp"])
        with patch.object(lp, "preflight", return_value=True), \
             patch.object(lp, "launch", return_value=42) as launch_mock:
            assert lp.main() == 42
            launch_mock.assert_called_once()
