"""Unit tests for ai_models/student/augmentations.py — Phase 2.

Covers _estimate_scale, EEGAugmentor (all 5 ops + dispatch + presets),
BuiltinAugmentor (all 3 ops + dispatch). Force `use_selfeeg=False`
so the fallback paths are deterministic.
"""
from __future__ import annotations

import pytest
import torch

from augmentations import BuiltinAugmentor, EEGAugmentor, _estimate_scale

pytestmark = pytest.mark.l2


class TestEstimateScale:
    def test_returns_per_batch(self):
        x = torch.randn(4, 21, 100)
        s = _estimate_scale(x)
        assert s.shape == (4, 1, 1)

    def test_clamps_to_min(self):
        x = torch.zeros(2, 4, 8)
        s = _estimate_scale(x)
        assert (s > 0).all()


class TestEEGAugmentorPresets:
    def test_default_moderate(self):
        a = EEGAugmentor(use_selfeeg=False)
        assert a.cfg["p"] == 0.5

    def test_light(self):
        a = EEGAugmentor(mode="light", use_selfeeg=False)
        assert a.cfg["p"] == 0.3

    def test_aggressive(self):
        a = EEGAugmentor(mode="aggressive", use_selfeeg=False)
        assert a.cfg["p"] == 0.7

    def test_unknown_falls_back_to_moderate(self):
        a = EEGAugmentor(mode="bogus", use_selfeeg=False)
        assert a.cfg["p"] == 0.5

    def test_override_p(self):
        a = EEGAugmentor(mode="moderate", p=0.99, use_selfeeg=False)
        assert a.cfg["p"] == 0.99


class TestEEGAugmentorCall:
    def test_skip_when_below_p(self, monkeypatch):
        a = EEGAugmentor(p=0.0, use_selfeeg=False)
        x = torch.randn(2, 21, 100)
        y = a(x)
        assert torch.equal(x, y)  # not augmented → identity

    def test_always_applies_when_p_one(self, monkeypatch):
        # p=1.0 → augment every time
        torch.manual_seed(0)
        a = EEGAugmentor(p=1.0, use_selfeeg=False)
        x = torch.randn(2, 21, 100)
        # Just verify it runs without error; output shape preserved
        y = a(x)
        assert y.shape == x.shape

    def test_additive_noise_fallback(self):
        a = EEGAugmentor(use_selfeeg=False)
        x = torch.randn(2, 4, 100)
        y = a._additive_noise(x)
        assert y.shape == x.shape
        assert not torch.equal(x, y)  # noise was added

    def test_channel_dropout_fallback(self):
        a = EEGAugmentor(use_selfeeg=False)
        # Set p high so dropout actually kicks in deterministically.
        a.cfg["channel_drop_p"] = 1.0
        x = torch.ones(2, 4, 8)
        y = a._channel_dropout(x)
        # With p=1.0 all channels should be dropped (zero)
        assert (y == 0).all()

    def test_temporal_mask(self):
        a = EEGAugmentor(use_selfeeg=False)
        x = torch.ones(2, 4, 100)
        y = a._temporal_mask(x)
        # Some samples should be zeroed
        assert (y == 0).any()
        assert y.shape == x.shape

    def test_amplitude_scale_fallback(self):
        a = EEGAugmentor(use_selfeeg=False)
        x = torch.ones(2, 4, 100)
        y = a._amplitude_scale(x)
        assert y.shape == x.shape
        # Not equal — scaled by uniform in [0.9, 1.1] (moderate)
        assert not torch.allclose(x, y, atol=1e-6)

    def test_temporal_shift_zero_noop(self):
        a = EEGAugmentor(use_selfeeg=False)
        a.cfg["shift_samples"] = 0
        x = torch.randn(2, 4, 100)
        y = a._temporal_shift(x)
        assert torch.equal(x, y)

    def test_temporal_shift_nonzero(self):
        a = EEGAugmentor(use_selfeeg=False)
        torch.manual_seed(42)
        a.cfg["shift_samples"] = 10
        x = torch.arange(100).float().reshape(1, 1, 100)
        y = a._temporal_shift(x)
        # Shape preserved; values rolled (most differ if shift != 0)
        assert y.shape == x.shape


class TestEEGAugmentorSelfEEGPath:
    """Exercise the selfeeg branches if the lib is installed.

    Each fallback wraps a try/except so even on selfeeg API drift the
    branch lights up via the except path.
    """
    def test_additive_noise_selfeeg(self):
        pytest.importorskip("selfeeg")
        a = EEGAugmentor(use_selfeeg=True)
        if not a.use_selfeeg:
            pytest.skip("selfeeg disabled in environment")
        x = torch.randn(2, 4, 100)
        y = a._additive_noise(x)
        assert y.shape == x.shape

    def test_channel_dropout_selfeeg(self):
        pytest.importorskip("selfeeg")
        a = EEGAugmentor(use_selfeeg=True)
        if not a.use_selfeeg:
            pytest.skip("selfeeg disabled in environment")
        x = torch.randn(2, 4, 100)
        y = a._channel_dropout(x)
        assert y.shape == x.shape

    def test_amplitude_scale_selfeeg(self):
        pytest.importorskip("selfeeg")
        a = EEGAugmentor(use_selfeeg=True)
        if not a.use_selfeeg:
            pytest.skip("selfeeg disabled in environment")
        x = torch.randn(2, 4, 100)
        y = a._amplitude_scale(x)
        assert y.shape == x.shape


class TestBuiltinAugmentor:
    def test_construction(self):
        a = BuiltinAugmentor()
        assert a.p == 0.5

    def test_custom_params(self):
        a = BuiltinAugmentor(noise_snr=10.0, channel_drop_p=0.5, p=0.99)
        assert a.noise_snr == 10.0
        assert a.channel_drop_p == 0.5

    def test_skip_when_p_zero(self):
        a = BuiltinAugmentor(p=0.0)
        x = torch.randn(2, 4, 8)
        y = a(x)
        assert torch.equal(x, y)

    def test_applies_when_p_one(self):
        torch.manual_seed(0)
        a = BuiltinAugmentor(p=1.0)
        x = torch.randn(2, 4, 100)
        y = a(x)
        assert y.shape == x.shape

    def test_each_branch_runs(self):
        # Force each branch via seeded random choice
        a = BuiltinAugmentor(p=1.0)
        x = torch.ones(2, 4, 100)
        for seed in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
            torch.manual_seed(seed)
            y = a(x)
            assert y.shape == x.shape
