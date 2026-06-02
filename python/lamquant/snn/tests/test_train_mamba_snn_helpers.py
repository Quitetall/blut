"""Unit tests for lamquant/snn/train_mamba_snn.py helpers — Phase 3."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Import via the canonical path. train_mamba_snn was ARCHIVED (legacy
# seizure trainer; SOT is train_4state_controller.py) — it now lives at
# blut/python/lamquant/snn/archive/train_mamba_snn.py (area dirs also on
# sys.path via blut/python/conftest.py; this keeps the file self-contained).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lamquant" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lamquant" / "snn" / "archive"))

import train_mamba_snn as tms

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# _dwb_weight
# ---------------------------------------------------------------------------
class TestDwbWeight:
    def test_shape_matches_target(self):
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        logits = torch.tensor([-0.5, 0.5, 0.2, -0.2])
        w = tms._dwb_weight(target, logits)
        assert w.shape == target.shape

    def test_positive_samples_weighted_more(self):
        target = torch.tensor([0.0, 1.0])
        logits = torch.tensor([0.0, 0.0])  # equal difficulty
        w = tms._dwb_weight(target, logits, base_pos_weight=3.0)
        # Positive sample should weigh more than negative
        assert w[1] > w[0]

    def test_normalized_to_mean_one(self):
        target = torch.zeros(10)
        logits = torch.zeros(10)
        w = tms._dwb_weight(target, logits)
        assert w.mean().item() == pytest.approx(1.0, abs=1e-6)

    def test_returns_no_grad(self):
        # _dwb_weight uses torch.no_grad — result should not require grad
        target = torch.zeros(4, requires_grad=False)
        logits = torch.zeros(4, requires_grad=True)
        w = tms._dwb_weight(target, logits)
        assert not w.requires_grad


# ---------------------------------------------------------------------------
# _rss_gb
# ---------------------------------------------------------------------------
class TestRssGb:
    def test_returns_float(self):
        out = tms._rss_gb()
        assert isinstance(out, float)
        # On Linux this is > 0; on other OSes returns 0
        assert out >= 0.0


# ---------------------------------------------------------------------------
# _augment_eeg
# ---------------------------------------------------------------------------
class TestAugmentEeg:
    def test_preserves_shape(self):
        torch.manual_seed(0)
        x = torch.randn(2, 21, 2500)
        out = tms._augment_eeg(x)
        assert out.shape == x.shape

    def test_no_aug_when_all_zero_prob(self):
        x = torch.randn(2, 21, 2500)
        x_orig = x.clone()
        out = tms._augment_eeg(x.clone(), p_channel_drop=0.0,
                                p_amplitude=0.0, p_noise=0.0,
                                p_time_shift=0.0)
        # All aug disabled → output unchanged
        assert torch.allclose(out, x_orig)

    def test_channel_dropout_zeros_channels(self):
        torch.manual_seed(42)
        x = torch.ones(2, 21, 100)
        out = tms._augment_eeg(x.clone(), p_channel_drop=1.0,
                                p_amplitude=0.0, p_noise=0.0,
                                p_time_shift=0.0)
        # At least 1 channel should be zero
        zero_channels = (out.abs().sum(dim=-1) == 0).sum().item()
        assert zero_channels >= 1


# ---------------------------------------------------------------------------
# _float_to_q31 / _float_to_q15
# ---------------------------------------------------------------------------
class TestQuantHelpers:
    def test_q31_full_range(self):
        assert tms._float_to_q31(1.0) == 2**31 - 1
        assert tms._float_to_q31(-1.0) == -(2**31 - 1)
        assert tms._float_to_q31(0.0) == 0

    def test_q31_clip(self):
        assert tms._float_to_q31(2.0) == 2**31 - 1
        assert tms._float_to_q31(-2.0) == -(2**31)

    def test_q15_full_range(self):
        assert tms._float_to_q15(1.0) == 32767
        assert tms._float_to_q15(-1.0) == -32767
        assert tms._float_to_q15(0.0) == 0

    def test_q15_clip(self):
        assert tms._float_to_q15(2.0) == 32767
        assert tms._float_to_q15(-2.0) == -32768


# ---------------------------------------------------------------------------
# _emit_int8_array
# ---------------------------------------------------------------------------
class TestEmitInt8Array:
    def test_emits_header_and_body(self):
        lines = []
        tensor = torch.randn(4, 4)
        n = tms._emit_int8_array(lines, "test_w", tensor)
        assert n == 16
        text = "\n".join(lines)
        assert "static const int8_t test_w" in text
        assert "static const float test_w_scale" in text
        assert "/* scale=" in text

    def test_handles_zero_tensor(self):
        lines = []
        n = tms._emit_int8_array(lines, "zero", torch.zeros(4))
        assert n == 4
        # Scale clamped at 1e-8
        text = "\n".join(lines)
        assert "0," in text  # All zeros


# ---------------------------------------------------------------------------
# _checkpoint_score
# ---------------------------------------------------------------------------
class TestCheckpointScore:
    def test_below_spec_floor_zero(self):
        assert tms._checkpoint_score(sens=1.0, acc=1.0, spec=0.5,
                                       min_spec=0.6) == 0.0

    def test_high_sens_tier1_uses_accuracy(self):
        # sens >= 0.99 → tier 1: returns 1.0 + acc
        out = tms._checkpoint_score(sens=0.99, acc=0.85, spec=0.95)
        assert out == pytest.approx(1.85)

    def test_low_sens_tier2_uses_sensitivity(self):
        # sens < 0.99 → tier 2: returns sens
        out = tms._checkpoint_score(sens=0.85, acc=0.95, spec=0.95)
        assert out == pytest.approx(0.85)

    def test_tier1_always_beats_tier2(self):
        tier1 = tms._checkpoint_score(sens=0.99, acc=0.0, spec=0.95)
        tier2 = tms._checkpoint_score(sens=0.98, acc=1.0, spec=0.95)
        assert tier1 > tier2
