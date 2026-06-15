#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# Coverage for the training primitives extracted into common/ (losses,
# augment, metrics.pearson_r_batch) out of the now-archived per-arch trainers.
# Pure-torch — runs everywhere (the legacy training_utils tests skip-first on
# importorskip("legacy")).

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lamquant.common.augment import (  # noqa: E402
    apply_montage_permutation,
    channel_dropout,
    clinical_augmentation,
)
from lamquant.common.losses import (  # noqa: E402
    SpectralLoss,
    band_weighted_mse,
    distillation_loss,
    pearson_r_loss,
    temporal_importance_mask,
)
from lamquant.common.metrics import pearson_r_batch  # noqa: E402


# ── losses ──────────────────────────────────────────────────────────
class TestSpectralLoss:
    def test_identical_signals_near_zero(self):
        x = torch.randn(2, 4, 313)
        assert SpectralLoss()(x, x.clone()).item() == pytest.approx(0.0, abs=1e-5)

    def test_grad_flows(self):
        x = torch.randn(2, 4, 313, requires_grad=True)
        SpectralLoss()(x, torch.randn(2, 4, 313)).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestTemporalImportanceMask:
    def test_shape_and_edges_below_center(self):
        m = temporal_importance_mask(T=100, edge_weight=0.3)
        assert m.shape == (1, 1, 100)
        assert m[0, 0, 0] < m[0, 0, 50]
        assert m.min() >= 0.3 - 1e-6 and m.max() <= 1.0 + 1e-6


class TestBandWeightedMse:
    def test_identical_zero(self):
        x = torch.randn(2, 4, 313)
        assert band_weighted_mse(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_and_scalar_for_diff(self):
        out = band_weighted_mse(torch.randn(2, 4, 313), torch.randn(2, 4, 313))
        assert out.dim() == 0 and out.item() > 0.0

    def test_even_and_odd_T(self):
        for T in (312, 313):
            x = torch.randn(1, 2, T)
            assert torch.isfinite(band_weighted_mse(x, x + 0.01)).all()


class TestPearsonRLoss:
    def test_identical_zero(self):
        x = torch.randn(2, 4, 16)
        assert pearson_r_loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-5)

    def test_negated_two(self):
        x = torch.randn(2, 4, 16)
        assert pearson_r_loss(x, -x).item() == pytest.approx(2.0, abs=1e-5)


class TestDistillationLoss:
    def test_returns_triple_and_identical_r_one(self):
        x = torch.randn(2, 4, 16)
        mse_a, mse_b, r = distillation_loss(x, x.clone())
        assert mse_a.item() == pytest.approx(0.0, abs=1e-6)
        assert mse_b.item() == pytest.approx(0.0, abs=1e-6)
        assert r.item() == pytest.approx(1.0, abs=1e-5)


# ── metrics ─────────────────────────────────────────────────────────
class TestPearsonRBatch:
    def test_identical_one(self):
        x = torch.randn(2, 4, 16)
        assert pearson_r_batch(x, x.clone()) == pytest.approx(1.0, abs=1e-5)

    def test_returns_python_float(self):
        out = pearson_r_batch(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        assert isinstance(out, float)


# ── augment ─────────────────────────────────────────────────────────
class TestChannelDropout:
    def test_no_op_when_not_training(self):
        x = torch.randn(2, 21, 50)
        assert torch.equal(channel_dropout(x, training=False), x)

    def test_zeros_some_channels_when_training(self):
        torch.manual_seed(0)
        x = torch.ones(1, 21, 50)
        out = channel_dropout(x, p_min=5, p_max=13, training=True)
        zeroed = (out.abs().sum(dim=-1)[0] == 0).sum().item()
        assert 5 <= zeroed <= 13


class TestMontagePermutation:
    def test_shape_preserved(self):
        x = torch.randn(2, 21, 50)
        assert apply_montage_permutation(x).shape == x.shape


class TestClinicalAugmentation:
    def test_shape_preserved_and_changes_signal(self):
        torch.manual_seed(0)
        x = torch.zeros(2, 8, 100)
        out = clinical_augmentation(x, fs=250.0)
        assert out.shape == x.shape
        assert not torch.equal(out, x)
