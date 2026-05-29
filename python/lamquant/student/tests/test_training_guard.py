"""Unit tests for ai_models/student/training_guard.py — Phase 1.

Covers all 10 guards: R plateau / collapse / minimum, loss explosion,
dead layer, alpha explosion/collapse, gradient vanish/explode,
NaN/Inf, latent collapse, DW-sep imbalance, summary printer.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from training_guard import GUARD_PRESETS, GuardConfig, TrainingGuard

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# GuardConfig / GUARD_PRESETS
# ---------------------------------------------------------------------------
class TestPresets:
    def test_presets_exist(self):
        assert {"v1", "v2", "fast"} <= set(GUARD_PRESETS.keys())

    def test_fast_short_patience(self):
        assert GUARD_PRESETS["fast"].check_every < GUARD_PRESETS["v2"].check_every

    def test_v2_wider_alpha(self):
        assert GUARD_PRESETS["v2"].alpha_ceiling >= GUARD_PRESETS["v1"].alpha_ceiling


# ---------------------------------------------------------------------------
# TrainingGuard.__init__
# ---------------------------------------------------------------------------
class TestInit:
    def test_string_config_resolves(self):
        m = nn.Linear(4, 4)
        g = TrainingGuard(m, config="v2")
        assert g.cfg is GUARD_PRESETS["v2"]

    def test_unknown_string_falls_back_to_v2(self):
        m = nn.Linear(4, 4)
        g = TrainingGuard(m, config="bogus")
        assert g.cfg is GUARD_PRESETS["v2"]

    def test_custom_config_used_as_is(self):
        m = nn.Linear(4, 4)
        c = GuardConfig(r_plateau_patience=5)
        g = TrainingGuard(m, config=c)
        assert g.cfg is c

    def test_initial_state(self):
        g = TrainingGuard(nn.Linear(4, 4))
        assert g.best_r == 0.0
        assert g.total_warnings == 0
        assert g.history == []
        assert g.initial_loss is None


# ---------------------------------------------------------------------------
# check() — basic schedule
# ---------------------------------------------------------------------------
class TestCheckSchedule:
    def test_skips_non_check_epoch(self):
        g = TrainingGuard(nn.Linear(4, 4), config=GuardConfig(check_every=5))
        # epoch 3 → not a check epoch and epoch > 0 → skip
        out = g.check(epoch=3, val_r=0.5)
        assert out == []
        assert g.best_r == 0.0  # state untouched

    def test_runs_on_zero_epoch(self):
        g = TrainingGuard(nn.Linear(4, 4))
        out = g.check(epoch=0, val_r=0.1)
        assert isinstance(out, list)
        assert g.best_r == 0.1

    def test_tracks_history(self):
        g = TrainingGuard(nn.Linear(4, 4), config=GuardConfig(check_every=1))
        g.check(epoch=0, val_r=0.1, train_loss=1.0)
        g.check(epoch=1, val_r=0.2, train_loss=0.5)
        assert len(g.history) == 2


# ---------------------------------------------------------------------------
# Guard 1+2+3: R plateau / collapse / minimum
# ---------------------------------------------------------------------------
class TestRGuards:
    def _g(self):
        return TrainingGuard(
            nn.Linear(4, 4),
            config=GuardConfig(check_every=1, r_plateau_patience=5,
                                r_collapse_threshold=0.1, r_minimum=0.05),
        )

    def test_plateau_warning(self):
        g = self._g()
        g.check(0, val_r=0.5)
        out = g.check(10, val_r=0.5)
        assert any("R PLATEAU" in w for w in out)

    def test_collapse_warning(self):
        g = self._g()
        g.check(0, val_r=0.8)
        out = g.check(1, val_r=0.6)
        assert any("R COLLAPSE" in w for w in out)

    def test_minimum_broken_warning(self):
        g = self._g()
        # epoch > 20 with R < r_minimum
        out = g.check(25, val_r=0.01)
        assert any("R BROKEN" in w for w in out)

    def test_no_warning_when_improving(self):
        g = self._g()
        g.check(0, val_r=0.5)
        out = g.check(1, val_r=0.6)
        assert all("PLATEAU" not in w for w in out)


# ---------------------------------------------------------------------------
# Guard 4: loss explosion
# ---------------------------------------------------------------------------
class TestLossExplosion:
    def test_explosion_warning(self):
        g = TrainingGuard(nn.Linear(4, 4),
                          config=GuardConfig(check_every=1, loss_explosion_factor=5.0))
        g.check(0, train_loss=1.0)
        out = g.check(1, train_loss=10.0)  # 10x initial
        assert any("LOSS EXPLOSION" in w for w in out)

    def test_no_warning_below_factor(self):
        g = TrainingGuard(nn.Linear(4, 4),
                          config=GuardConfig(check_every=1, loss_explosion_factor=10.0))
        g.check(0, train_loss=1.0)
        out = g.check(1, train_loss=2.0)
        assert all("EXPLOSION" not in w for w in out)


# ---------------------------------------------------------------------------
# Guard 5+6: dead layer + alpha explosion/collapse
# ---------------------------------------------------------------------------
class _StubLayer(nn.Module):
    def __init__(self, weight_shape, alpha_val):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(weight_shape))
        self.lsq_alpha = nn.Parameter(torch.tensor([alpha_val]))


class TestAlphaDeadLayer:
    def test_dead_layer_triggered(self):
        # Weights below |alpha| → dead. Set alpha large.
        layer = _StubLayer((8, 8), alpha_val=10.0)
        layer.weight.data.fill_(0.0)  # all zero → all dead
        g = TrainingGuard(layer, config=GuardConfig(check_every=1,
                                                      dead_layer_sparsity=0.5))
        out = g.check(0)
        assert any("DEAD LAYER" in w for w in out)

    def test_alpha_explosion(self):
        layer = _StubLayer((4, 4), alpha_val=20.0)
        g = TrainingGuard(layer, config=GuardConfig(check_every=1,
                                                      alpha_ceiling=5.0))
        out = g.check(0)
        assert any("ALPHA EXPLOSION" in w for w in out)

    def test_alpha_collapse(self):
        layer = _StubLayer((4, 4), alpha_val=1e-6)
        g = TrainingGuard(layer, config=GuardConfig(check_every=1,
                                                      alpha_floor=1e-3))
        out = g.check(0)
        assert any("ALPHA COLLAPSE" in w for w in out)


# ---------------------------------------------------------------------------
# Guard 7+8: gradient vanish/explode + NaN
# ---------------------------------------------------------------------------
class TestGradientGuards:
    def test_gradient_explode(self):
        m = nn.Linear(4, 4)
        # Manually set giant grad
        m.weight.grad = torch.full_like(m.weight.data, 1e3)
        g = TrainingGuard(m, config=GuardConfig(check_every=1,
                                                  grad_explode_threshold=10.0))
        out = g.check(0)
        assert any("GRADIENT EXPLODE" in w for w in out)

    def test_gradient_vanish(self):
        m = nn.Linear(4, 4)
        m.weight.grad = torch.full_like(m.weight.data, 1e-10)
        g = TrainingGuard(m, config=GuardConfig(check_every=1,
                                                  grad_vanish_threshold=1e-5))
        out = g.check(0)
        assert any("GRADIENT VANISH" in w for w in out)

    def test_nan_gradient_detected(self):
        m = nn.Linear(4, 4)
        m.weight.grad = torch.full_like(m.weight.data, float("nan"))
        g = TrainingGuard(m, config=GuardConfig(check_every=1))
        out = g.check(0)
        assert any("NaN/Inf GRADIENT" in w for w in out)

    def test_nan_weight_detected(self):
        m = nn.Linear(4, 4)
        m.weight.data.fill_(float("inf"))
        g = TrainingGuard(m, config=GuardConfig(check_every=1))
        out = g.check(0)
        assert any("NaN/Inf WEIGHT" in w for w in out)


# ---------------------------------------------------------------------------
# Guard 9: latent collapse
# ---------------------------------------------------------------------------
class TestLatentCollapse:
    def test_collapse_triggered(self):
        m = nn.Linear(4, 4)
        g = TrainingGuard(m, config=GuardConfig(check_every=1,
                                                  latent_std_minimum=0.5))
        out = g.check(0, latent=torch.zeros(4, 16))
        assert any("LATENT COLLAPSE" in w for w in out)

    def test_no_collapse_when_std_high(self):
        m = nn.Linear(4, 4)
        g = TrainingGuard(m, config=GuardConfig(check_every=1,
                                                  latent_std_minimum=0.001))
        out = g.check(0, latent=torch.randn(4, 16))
        assert all("LATENT" not in w for w in out)


# ---------------------------------------------------------------------------
# Guard 10: DW-sep imbalance
# ---------------------------------------------------------------------------
class _DwPwModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.dw = nn.Linear(4, 4)
        self.block.pw = nn.Linear(4, 4)


class TestDwSepImbalance:
    def test_imbalance_triggered(self):
        m = _DwPwModel()
        # DW grad large, PW grad small
        m.block.dw.weight.grad = torch.full_like(m.block.dw.weight.data, 100.0)
        m.block.pw.weight.grad = torch.full_like(m.block.pw.weight.data, 0.1)
        g = TrainingGuard(m, config=GuardConfig(check_every=1))
        out = g.check(0)
        assert any("DW-SEP IMBALANCE" in w for w in out)

    def test_no_imbalance_when_balanced(self):
        m = _DwPwModel()
        m.block.dw.weight.grad = torch.full_like(m.block.dw.weight.data, 0.1)
        m.block.pw.weight.grad = torch.full_like(m.block.pw.weight.data, 0.1)
        g = TrainingGuard(m, config=GuardConfig(check_every=1))
        out = g.check(0)
        assert all("DW-SEP" not in w for w in out)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
class TestSummary:
    def test_summary_contains_counts(self):
        g = TrainingGuard(nn.Linear(4, 4), config=GuardConfig(check_every=1))
        g.check(0, val_r=0.5, train_loss=1.0)
        s = g.summary()
        assert "Total warnings" in s
        assert "Best R" in s
        assert "Epochs tracked" in s

    def test_summary_with_history_includes_final(self):
        g = TrainingGuard(nn.Linear(4, 4), config=GuardConfig(check_every=1))
        g.check(0, val_r=0.5)
        g.check(1, val_r=0.3)
        s = g.summary()
        assert "Final R" in s
