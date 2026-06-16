"""Unit tests for ai_models/student/experiment_runner.py — Phase 3.

Covers EXPERIMENTS preset dict, _load_data file scanner, _make_augmentor,
_make_multiscale_fsq, _get_tau (cosine + progressive + unknown),
and main(--list + smoke). run_experiment is exercised end-to-end via
stubs for TernaryMobileNetV5_Subband to keep CPU-only smoke fast.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.l2


_MODULE_PATH = (Path(__file__).resolve().parents[1]
                / "experiment_runner.py")


@pytest.fixture
def er_mod(monkeypatch):
    """Load student experiment_runner with stubbed encoder import."""
    # Provide a fake TernaryMobileNetV5_Subband so import-time call succeeds
    # even on CPU-only machines.
    class _FakeEncoder(nn.Module):
        def __init__(self, in_ch=21, latent_dim=32):
            super().__init__()
            self.encoder_proj = nn.Conv1d(in_ch, latent_dim, kernel_size=1)
            self.decoder_proj = nn.Conv1d(latent_dim, in_ch, kernel_size=1)

        def ensure_initialized(self): pass

        def encode(self, x, quantize=False):
            return torch.nn.functional.adaptive_avg_pool1d(
                self.encoder_proj(x), 79)

        def forward(self, x, quantize=False):
            l = self.encode(x, quantize)
            up = torch.nn.functional.interpolate(l, size=x.shape[-1],
                                                  mode="linear",
                                                  align_corners=False)
            return self.decoder_proj(up)

    encoder_pkg = types.ModuleType("lamquant_codec")
    encoder_models = types.ModuleType("lamquant_neural.models")
    encoder_mod = types.ModuleType("lamquant_neural.models.encoder")
    encoder_mod.TernaryMobileNetV5_Subband = _FakeEncoder
    encoder_pkg.models = encoder_models
    encoder_models.encoder = encoder_mod
    monkeypatch.setitem(sys.modules, "lamquant_codec", encoder_pkg)
    monkeypatch.setitem(sys.modules, "lamquant_neural.models", encoder_models)
    monkeypatch.setitem(sys.modules, "lamquant_neural.models.encoder", encoder_mod)

    # Stub training_guard so it doesn't import the real ai_models.student
    name = "ai_models.student.experiment_runner_under_test"
    monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# EXPERIMENTS dict
# ---------------------------------------------------------------------------
class TestExperiments:
    def test_baseline_present(self, er_mod):
        assert "baseline" in er_mod.EXPERIMENTS

    def test_all_have_required_keys(self, er_mod):
        required = {"desc", "augmentor", "multiscale_fsq",
                     "tau_schedule", "wd_schedule", "epochs"}
        for name, cfg in er_mod.EXPERIMENTS.items():
            assert required <= set(cfg.keys()), f"{name} missing keys"

    def test_each_desc_nonempty(self, er_mod):
        for name, cfg in er_mod.EXPERIMENTS.items():
            assert cfg["desc"]


# ---------------------------------------------------------------------------
# _make_augmentor
# ---------------------------------------------------------------------------
class TestMakeAugmentor:
    def test_none_returns_none(self, er_mod):
        assert er_mod._make_augmentor(None) is None

    def test_selfeeg_returns_eeg_augmentor(self, er_mod):
        out = er_mod._make_augmentor(("selfeeg", "moderate"))
        assert out is not None
        # Has __call__
        assert callable(out)

    def test_builtin_returns_builtin(self, er_mod):
        out = er_mod._make_augmentor(("builtin", "moderate"))
        assert out is not None
        assert callable(out)

    def test_unknown_type_returns_none(self, er_mod):
        assert er_mod._make_augmentor(("bogus", "x")) is None


# ---------------------------------------------------------------------------
# _make_multiscale_fsq
# ---------------------------------------------------------------------------
class TestMakeMultiscaleFsq:
    def test_none(self, er_mod):
        assert er_mod._make_multiscale_fsq(None) is None

    def test_preset(self, er_mod):
        out = er_mod._make_multiscale_fsq("balanced")
        assert out is not None


# ---------------------------------------------------------------------------
# _get_tau
# ---------------------------------------------------------------------------
class TestGetTau:
    def test_cosine_at_start_is_max(self, er_mod):
        # cos(0) = 1 → tau = 0.1
        assert er_mod._get_tau(0, 100, "cosine") == pytest.approx(0.1)

    def test_cosine_at_end_is_zero(self, er_mod):
        # cos(pi) = -1 → tau = 0
        assert er_mod._get_tau(100, 100, "cosine") == pytest.approx(0.0, abs=1e-6)

    def test_cosine_at_mid_is_half(self, er_mod):
        assert er_mod._get_tau(50, 100, "cosine") == pytest.approx(0.05, abs=1e-6)

    def test_progressive_before_boundary(self, er_mod):
        # 80% boundary at epoch=80 → before that, tau = 0.1
        assert er_mod._get_tau(50, 100, "progressive") == 0.1

    def test_progressive_after_boundary_anneal(self, er_mod):
        # At boundary, tau = 0.1; at end, tau = 0
        boundary = int(100 * 0.8)
        assert er_mod._get_tau(boundary, 100, "progressive") == pytest.approx(0.1)
        assert er_mod._get_tau(99, 100, "progressive") == pytest.approx(0.005)

    def test_unknown_schedule_constant(self, er_mod):
        assert er_mod._get_tau(50, 100, "bogus") == 0.1


# ---------------------------------------------------------------------------
# _load_data
# ---------------------------------------------------------------------------
class TestLoadData:
    def test_no_files_returns_empty(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        # Directory exists but empty
        (tmp_path / "ai_models" / "dataset_sim" / "q31_events").mkdir(parents=True)
        windows = er_mod._load_data(max_windows=10)
        assert windows == []

    def test_loads_3d_l3(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        d = tmp_path / "ai_models" / "dataset_sim" / "q31_events"
        d.mkdir(parents=True)
        np.savez_compressed(d / "a.npz",
                             l3=np.random.randn(5, 21, 313).astype(np.float32))
        windows = er_mod._load_data(max_windows=10)
        assert len(windows) == 5

    def test_loads_2d_l3(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        d = tmp_path / "ai_models" / "dataset_sim" / "q31_events"
        d.mkdir(parents=True)
        np.savez_compressed(d / "b.npz",
                             l3=np.random.randn(21, 313).astype(np.float32))
        windows = er_mod._load_data(max_windows=10)
        assert len(windows) == 1

    def test_skips_unreadable(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        d = tmp_path / "ai_models" / "dataset_sim" / "q31_events"
        d.mkdir(parents=True)
        (d / "broken.npz").write_text("corrupt")
        windows = er_mod._load_data(max_windows=10)
        assert windows == []

    def test_caps_at_max_windows(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        d = tmp_path / "ai_models" / "dataset_sim" / "q31_events"
        d.mkdir(parents=True)
        np.savez_compressed(d / "c.npz",
                             l3=np.random.randn(50, 21, 313).astype(np.float32))
        windows = er_mod._load_data(max_windows=10)
        assert len(windows) == 10


# ---------------------------------------------------------------------------
# run_experiment - exercise abort path (too few windows)
# ---------------------------------------------------------------------------
class TestRunExperimentAbort:
    def test_unknown_experiment_raises(self, er_mod):
        with pytest.raises(ValueError, match="Unknown experiment"):
            er_mod.run_experiment("does_not_exist")

    def test_too_few_data_returns_none(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        # No data dir → _load_data returns 0 windows → run_experiment None
        (tmp_path / "ai_models" / "dataset_sim" / "q31_events").mkdir(parents=True)
        result = er_mod.run_experiment("baseline", device="cpu")
        assert result is None


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
class TestMain:
    def test_list_lists_experiments(self, er_mod, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["er", "--list"])
        er_mod.main()
        out = capsys.readouterr().out
        assert "baseline" in out
        assert "selfeeg_moderate" in out
