"""Unit tests for ai_models/student/sweep_noise_bits.py — Phase 1 trivial.

CLI smoke: argparse + mocked ExperimentRunner.sweep().
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.l2


_MODULE_PATH = (Path(__file__).resolve().parents[1]
                / "sweep_noise_bits.py")


@pytest.fixture
def snb_mod(monkeypatch):
    # Stub heavy import chain BEFORE module load. monkeypatch.setitem
    # auto-restores prior sys.modules state at function teardown so the
    # real ai_models.experiment_runner stays untouched for other tests.
    fake_runner = MagicMock()
    fake_runner.return_value.sweep.return_value = [
        {"best_val_r": 0.51, "best_val_prd": 24.5,
         "best_val_loss": 0.003, "best_epoch": 50,
         "config": {"train_noise_bits": 0}},
        {"best_val_r": 0.55, "best_val_prd": 23.0,
         "best_val_loss": 0.002, "best_epoch": 40,
         "config": {"train_noise_bits": 1}},
        None,  # exercise the None-skip path
    ]
    er_mod = types.ModuleType("ai_models.experiment_runner")
    er_mod.ExperimentRunner = fake_runner
    monkeypatch.setitem(sys.modules, "ai_models.experiment_runner", er_mod)

    name = "ai_models.student.sweep_noise_bits_under_test"
    monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module, fake_runner


class TestSweepNoiseBitsMain:
    def test_main_default_runs(self, snb_mod, monkeypatch, capsys):
        mod, fake_runner = snb_mod
        monkeypatch.setattr(sys, "argv", ["snb"])
        mod.main()
        # ExperimentRunner constructed once, .sweep called once
        fake_runner.assert_called_once()
        fake_runner.return_value.sweep.assert_called_once()
        out = capsys.readouterr().out
        assert "Noise-Bits Ablation" in out
        assert "RESULTS" in out

    def test_main_max_bits(self, snb_mod, monkeypatch):
        mod, fake_runner = snb_mod
        monkeypatch.setattr(sys, "argv", ["snb", "--max-bits", "3"])
        mod.main()
        # grid was {'train_noise_bits': [0,1,2,3]}
        _, kwargs = fake_runner.return_value.sweep.call_args
        assert kwargs["grid"]["train_noise_bits"] == [0, 1, 2, 3]

    def test_main_tier_and_preset(self, snb_mod, monkeypatch):
        mod, fake_runner = snb_mod
        monkeypatch.setattr(sys, "argv",
                            ["snb", "--preset", "medium", "--tier", "5"])
        mod.main()
        _, kwargs = fake_runner.return_value.sweep.call_args
        assert kwargs["preset"] == "medium"
        assert kwargs["tier"] == 5

    def test_main_multi_seed(self, snb_mod, monkeypatch):
        mod, fake_runner = snb_mod
        monkeypatch.setattr(sys, "argv",
                            ["snb", "--seeds", "0", "42", "99"])
        mod.main()
        _, kwargs = fake_runner.return_value.sweep.call_args
        assert kwargs["seeds"] == [0, 42, 99]
