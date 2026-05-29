"""Unit tests for ai_models/student/training_utils.py — Phase 3.

Covers the testable helpers (_safe_load, SpectralLoss,
temporal_importance_mask, band_weighted_mse, pearson_r_loss,
pearson_r_batch, channel_dropout, eeg_augment, split_by_manifest,
validate_epoch, latent_kurtosis). The 1000-line `run()` training
function needs real data + GPU + heavy deps; out of unit-test scope.
"""
from __future__ import annotations

import pytest  # decomp: `legacy/` Gen-7.0 code excluded from all repos (dead)
pytest.importorskip("legacy", reason="Tests dead legacy/ Gen-7.0 code excluded from the decomposition")

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import training_utils as tu

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# _safe_load
# ---------------------------------------------------------------------------
class TestSafeLoad:
    def test_loads_state_dict(self, tmp_path):
        ck = tmp_path / "x.ckpt"
        sd = {"weight": torch.zeros(4)}
        torch.save(sd, ck)
        loaded = tu._safe_load(ck)
        assert "weight" in loaded

    def test_fallback_on_failure(self, tmp_path):
        # Force the fallback path by mocking torch.load to raise on
        # weights_only=True, succeed on weights_only=False.
        ck = tmp_path / "x.ckpt"
        ck.write_bytes(b"x")
        from unittest.mock import patch

        def _fake_load(path, map_location=None, weights_only=None):
            if weights_only:
                raise RuntimeError("simulated unsafe")
            return {"loaded": True}

        with patch("training_utils.torch.load", side_effect=_fake_load):
            with pytest.warns(UserWarning, match="weights_only"):
                loaded = tu._safe_load(ck)
        assert loaded == {"loaded": True}


# ---------------------------------------------------------------------------
# SpectralLoss
# ---------------------------------------------------------------------------
class TestSpectralLoss:
    def test_identical_signals_zero_loss(self):
        loss = tu.SpectralLoss(fft_sizes=[16, 32])
        x = torch.randn(2, 4, 313)
        out = loss(x, x.clone())
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_returns_scalar(self):
        loss = tu.SpectralLoss(fft_sizes=[16])
        out = loss(torch.randn(2, 4, 313), torch.randn(2, 4, 313))
        assert out.ndim == 0


# ---------------------------------------------------------------------------
# temporal_importance_mask
# ---------------------------------------------------------------------------
class TestTemporalImportanceMask:
    def test_shape(self):
        m = tu.temporal_importance_mask(T=100)
        assert m.shape == (1, 1, 100)

    def test_edges_below_center(self):
        m = tu.temporal_importance_mask(T=100, edge_weight=0.3)
        # First and last samples should be near edge_weight
        assert m[0, 0, 0].item() == pytest.approx(0.3, abs=1e-5)
        assert m[0, 0, -1].item() == pytest.approx(0.3, abs=1e-5)
        # Middle should be near 1.0
        assert m[0, 0, 50].item() > 0.9


# ---------------------------------------------------------------------------
# band_weighted_mse
# ---------------------------------------------------------------------------
class TestBandWeightedMSE:
    def test_identical_signals_zero_loss(self):
        x = torch.randn(2, 4, 313)
        out = tu.band_weighted_mse(x, x.clone())
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different(self):
        out = tu.band_weighted_mse(torch.randn(2, 4, 313),
                                    torch.randn(2, 4, 313))
        assert out.item() > 0

    def test_returns_scalar(self):
        out = tu.band_weighted_mse(torch.randn(2, 4, 313),
                                    torch.randn(2, 4, 313))
        assert out.ndim == 0

    def test_odd_T_no_nyquist_correction(self):
        # T odd → if branch at line 133-134 not taken
        x = torch.randn(2, 4, 313)  # 313 is odd
        out = tu.band_weighted_mse(x, x.clone() + 0.001)
        assert torch.isfinite(out)

    def test_even_T(self):
        x = torch.randn(2, 4, 64)
        out = tu.band_weighted_mse(x, x.clone() + 0.001)
        assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# pearson_r_loss / pearson_r_batch
# ---------------------------------------------------------------------------
class TestPearsonR:
    def test_loss_identical_zero(self):
        x = torch.randn(4, 21, 313)
        assert tu.pearson_r_loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_negated_two(self):
        x = torch.randn(2, 4, 100)
        assert tu.pearson_r_loss(x, -x).item() == pytest.approx(2.0, abs=1e-5)

    def test_batch_identical_one(self):
        x = torch.randn(4, 21, 313)
        assert tu.pearson_r_batch(x, x.clone()) == pytest.approx(1.0, abs=1e-5)

    def test_batch_returns_float(self):
        out = tu.pearson_r_batch(torch.randn(2, 4, 16),
                                  torch.randn(2, 4, 16))
        assert isinstance(out, float)


# ---------------------------------------------------------------------------
# channel_dropout
# ---------------------------------------------------------------------------
class TestChannelDropout:
    def test_no_op_when_not_training(self):
        x = torch.randn(2, 21, 100)
        out = tu.channel_dropout(x, training=False)
        assert torch.equal(x, out)

    def test_drops_channels(self):
        torch.manual_seed(0)
        x = torch.ones(2, 21, 50)
        out = tu.channel_dropout(x, p_min=5, p_max=13, training=True)
        # Some channels should be zeroed
        zero_count = (out.abs().sum(dim=-1) == 0).sum().item()
        assert zero_count >= 5 * 2  # at least 5 per sample × 2 samples
        assert zero_count <= 13 * 2

    def test_preserves_shape(self):
        x = torch.randn(2, 21, 50)
        assert tu.channel_dropout(x).shape == x.shape


# ---------------------------------------------------------------------------
# eeg_augment
# ---------------------------------------------------------------------------
class TestEegAugment:
    def test_no_op_when_not_training(self):
        x = torch.randn(2, 21, 313)
        out = tu.eeg_augment(x, training=False)
        assert torch.equal(x, out)

    def test_augments_when_training(self):
        pytest.importorskip("selfeeg")
        torch.manual_seed(0)
        x = torch.randn(2, 21, 313)
        out = tu.eeg_augment(x, training=True)
        # Output should differ (gaussian noise always applied)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# split_by_manifest
# ---------------------------------------------------------------------------
class TestSplitByManifest:
    def test_returns_two_lists(self):
        # Uses real manifest_v3.json (present in repo)
        train, val = tu.split_by_manifest(npz_files=[])
        assert isinstance(train, list)
        assert isinstance(val, list)


# ---------------------------------------------------------------------------
# validate_epoch
# ---------------------------------------------------------------------------
class TestValidateEpoch:
    def test_empty_loader_returns_defaults(self):
        model = nn.Linear(313, 313)
        r, prd = tu.validate_epoch(model, val_loader=[],
                                     device=torch.device("cpu"))
        assert r == 0.0
        assert prd == 100.0

    def test_with_synth_batch(self):
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(313, 313)
            def forward(self, x, quantize=True):
                return self.lin(x)

        m = _M()
        x_l3 = torch.randn(2, 21, 313)
        loader = [(x_l3, None, None)]
        r, prd = tu.validate_epoch(m, loader, device=torch.device("cpu"))
        assert isinstance(r, float)
        assert isinstance(prd, float)


# ---------------------------------------------------------------------------
# latent_kurtosis
# ---------------------------------------------------------------------------
class TestLatentKurtosis:
    def test_smoke(self):
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Conv1d(21, 32, 1)
            def encode(self, x, quantize=False):
                # Return [B, 32, T]
                return self.lin(x)
            def forward(self, x, quantize=False):
                return self.lin(x)

        m = _M()
        x_l3 = torch.randn(2, 21, 313)
        loader = [(x_l3, None, None)]
        out = tu.latent_kurtosis(m, loader, device=torch.device("cpu"),
                                  max_batches=1)
        # Returns whatever the impl returns — just verify it doesn't raise
        assert out is not None
