"""Deep coverage for ai_models/student/experiment_runner.py.

Builds on the existing ``test_student_experiment_runner.py`` which
already covers EXPERIMENTS dict, _make_augmentor, _make_multiscale_fsq,
_get_tau, _load_data, the unknown-experiment branch and ``main --list``.

This file extends to:
  - ``run_experiment`` happy path (tiny epochs, CPU, stubbed encoder)
  - Different tau schedule branches inside the loop
  - Two-stage WD schedule branch
  - main() driving a real experiment + json artifact

Per ``feedback_futureproof_tests``: pin shape/type/keys, not numeric
training losses (which drift across torch versions).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.l2


_MODULE_PATH = (Path(__file__).resolve().parents[1]
                / "experiment_runner.py")


class _StubEncoder(nn.Module):
    """Minimal encoder that mimics the TernaryMobileNetV5_Subband API.

    Just enough to let run_experiment() complete on CPU with tiny inputs.
    The contract under test is the orchestration loop, not the encoder.
    """
    def __init__(self, in_ch=21, latent_dim=32):
        super().__init__()
        self.in_ch = in_ch
        self.latent_dim = latent_dim
        self.enc = nn.Conv1d(in_ch, latent_dim, kernel_size=1)
        self.dec = nn.Conv1d(latent_dim, in_ch, kernel_size=1)

    def ensure_initialized(self):
        return

    def encode(self, x, quantize=False):
        # Encoder must downsample to 79 timesteps to match the codec
        # latent contract used by the runner's q2d2 / msfsq branches.
        z = self.enc(x)
        return torch.nn.functional.adaptive_avg_pool1d(z, 79)

    def forward(self, x, quantize=False):
        z = self.encode(x, quantize)
        up = torch.nn.functional.interpolate(z, size=x.shape[-1],
                                              mode="linear",
                                              align_corners=False)
        return self.dec(up)


@pytest.fixture
def er_mod(monkeypatch, tmp_path):
    """Load student experiment_runner with stubbed encoder + small data dir."""
    encoder_pkg = types.ModuleType("lamquant_codec")
    encoder_models = types.ModuleType("lamquant_neural.models")
    encoder_mod = types.ModuleType("lamquant_neural.models.encoder")
    encoder_mod.TernaryMobileNetV5_Subband = _StubEncoder
    encoder_pkg.models = encoder_models
    encoder_models.encoder = encoder_mod
    monkeypatch.setitem(sys.modules, "lamquant_codec", encoder_pkg)
    monkeypatch.setitem(sys.modules, "lamquant_neural.models", encoder_models)
    monkeypatch.setitem(sys.modules, "lamquant_neural.models.encoder", encoder_mod)

    # Load module fresh
    name = "ai_models.student.experiment_runner_deep_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_data_dir(repo_root: Path, n_windows: int = 200) -> Path:
    """Plant a single NPZ with N L3 windows in the runner-expected dir."""
    d = repo_root / "ai_models" / "dataset_sim" / "q31_events"
    d.mkdir(parents=True, exist_ok=True)
    np.random.seed(7)
    l3 = np.random.randn(n_windows, 21, 313).astype(np.float32)
    np.savez_compressed(d / "windows.npz", l3=l3)
    return d


# ============================================================
# run_experiment — happy path (tiny epochs, tiny data)
# ============================================================


class TestRunExperimentHappyPath:
    def test_baseline_returns_dict(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        _seed_data_dir(tmp_path, n_windows=200)

        # Override the 'baseline' epochs to a tiny count for the test.
        original_epochs = er_mod.EXPERIMENTS["baseline"]["epochs"]
        er_mod.EXPERIMENTS["baseline"] = {
            **er_mod.EXPERIMENTS["baseline"], "epochs": 5,
        }
        try:
            result = er_mod.run_experiment("baseline", device="cpu")
        finally:
            er_mod.EXPERIMENTS["baseline"]["epochs"] = original_epochs

        # Pinned contract: result is a dict with the documented keys.
        assert isinstance(result, dict)
        for k in ("experiment", "description", "epochs",
                   "final_train_loss", "final_val_r", "final_val_prd",
                   "best_val_r", "rans_compressed_bytes", "fsq_utilization_pct",
                   "per_channel_r_variance", "loss_slope_at_50",
                   "guard_warnings", "best_val_prd", "history", "config",
                   "timestamp"):
            assert k in result, f"missing key {k}"
        # Pearson R bounded in [-1, 1].
        assert -1.0 <= result["final_val_r"] <= 1.0
        # Train loss is finite.
        assert np.isfinite(result["final_train_loss"])
        # History dict has the three time series.
        for series in ("train_loss", "val_r", "val_prd"):
            assert series in result["history"]
            assert isinstance(result["history"][series], list)
        # Config preserved.
        assert "augmentor" in result["config"]

    def test_progressive_tau_branch(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        _seed_data_dir(tmp_path, n_windows=200)
        orig = er_mod.EXPERIMENTS["progressive_tau"]["epochs"]
        er_mod.EXPERIMENTS["progressive_tau"] = {
            **er_mod.EXPERIMENTS["progressive_tau"], "epochs": 5,
        }
        try:
            result = er_mod.run_experiment("progressive_tau", device="cpu")
        finally:
            er_mod.EXPERIMENTS["progressive_tau"]["epochs"] = orig
        assert result is not None
        assert result["experiment"] == "progressive_tau"

    def test_two_stage_wd_branch(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        _seed_data_dir(tmp_path, n_windows=200)
        orig = er_mod.EXPERIMENTS["two_stage_wd"]["epochs"]
        # Need enough epochs to exceed the 2/3 boundary (so the
        # branch flips weight_decay → 0).
        er_mod.EXPERIMENTS["two_stage_wd"] = {
            **er_mod.EXPERIMENTS["two_stage_wd"], "epochs": 5,
        }
        try:
            result = er_mod.run_experiment("two_stage_wd", device="cpu")
        finally:
            er_mod.EXPERIMENTS["two_stage_wd"]["epochs"] = orig
        assert result is not None
        assert result["experiment"] == "two_stage_wd"


# ============================================================
# run_experiment — q2d2 branch
# ============================================================


class TestRunExperimentQ2D2:
    def test_q2d2_loss_path(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        _seed_data_dir(tmp_path, n_windows=200)
        orig = er_mod.EXPERIMENTS["q2d2_l5"]["epochs"]
        er_mod.EXPERIMENTS["q2d2_l5"] = {
            **er_mod.EXPERIMENTS["q2d2_l5"], "epochs": 5,
        }
        try:
            result = er_mod.run_experiment("q2d2_l5", device="cpu")
        finally:
            er_mod.EXPERIMENTS["q2d2_l5"]["epochs"] = orig
        assert result is not None


# ============================================================
# main() — drive a real experiment
# ============================================================


class TestMainRunsExperiment:
    def test_main_baseline_writes_json(self, er_mod, tmp_path,
                                        monkeypatch, capsys):
        # Pin the runner to a clean tmp _REPO and seed minimal data.
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        _seed_data_dir(tmp_path, n_windows=200)

        # Trim baseline epochs for speed.
        orig = er_mod.EXPERIMENTS["baseline"]["epochs"]
        er_mod.EXPERIMENTS["baseline"] = {
            **er_mod.EXPERIMENTS["baseline"], "epochs": 5,
        }
        monkeypatch.setattr(sys, "argv",
                            ["er", "--experiment", "baseline",
                             "--device", "cpu"])
        try:
            er_mod.main()
        finally:
            er_mod.EXPERIMENTS["baseline"]["epochs"] = orig

        # An experiments output dir should exist with a JSON in it.
        out_dir = tmp_path / "outputs" / "experiments"
        assert out_dir.is_dir()
        jsons = sorted(out_dir.glob("baseline_*.json"))
        assert jsons, f"no JSON written to {out_dir}"
        # Validate basic JSON shape.
        loaded = json.loads(jsons[0].read_text())
        assert loaded["experiment"] == "baseline"
        assert "history" in loaded


# ============================================================
# _get_tau edge: unknown schedule falls through to default 0.1
# ============================================================


class TestGetTauEdge:
    def test_cosine_decreasing(self, er_mod):
        # Cosine schedule should be monotonically non-increasing.
        taus = [er_mod._get_tau(e, 10, "cosine") for e in range(11)]
        # NOT strictly monotonic in general due to cosine but at the
        # endpoints we expect tau(0) > tau(T).
        assert taus[0] > taus[-1]

    def test_progressive_at_zero(self, er_mod):
        # epoch=0 in progressive → tau=0.1 (before boundary).
        assert er_mod._get_tau(0, 100, "progressive") == pytest.approx(0.1)


# ============================================================
# _load_data — additional shapes
# ============================================================


class TestLoadDataExtras:
    def test_wrong_shape_l3_skipped(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        d = tmp_path / "ai_models" / "dataset_sim" / "q31_events"
        d.mkdir(parents=True)
        # Wrong second-axis size: 10 channels instead of 21.
        np.savez_compressed(
            d / "wrong.npz",
            l3=np.random.randn(3, 10, 313).astype(np.float32))
        windows = er_mod._load_data(max_windows=10)
        # All windows should be skipped — wrong channel count.
        assert windows == []

    def test_missing_l3_key_skipped(self, er_mod, tmp_path, monkeypatch):
        monkeypatch.setattr(er_mod, "_REPO", str(tmp_path))
        d = tmp_path / "ai_models" / "dataset_sim" / "q31_events"
        d.mkdir(parents=True)
        # File has 'signal' but no 'l3'.
        np.savez_compressed(d / "no_l3.npz", signal=np.zeros(10))
        windows = er_mod._load_data(max_windows=10)
        assert windows == []
