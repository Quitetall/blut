"""Unit tests for ai_models/student/muon_optimizer.py — Phase 1 quick win.

Covers Newton-Schulz orthogonalization, muon/adam updates, the Muon
optimizer class (matrix + scalar param groups), and split_params_for_muon.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lamquant.ingredients.optimizers.muon_optimizer import (
    Muon,
    adam_update,
    muon_update,
    split_params_for_muon,
    zeropower_via_newtonschulz5,
)

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# zeropower_via_newtonschulz5
# ---------------------------------------------------------------------------
class TestNewtonSchulz:
    def test_square_matrix_shape(self):
        G = torch.randn(8, 8)
        out = zeropower_via_newtonschulz5(G, steps=3)
        assert out.shape == G.shape

    def test_tall_matrix_shape(self):
        G = torch.randn(16, 8)
        out = zeropower_via_newtonschulz5(G, steps=3)
        assert out.shape == G.shape

    def test_wide_matrix_shape(self):
        G = torch.randn(8, 16)
        out = zeropower_via_newtonschulz5(G, steps=3)
        assert out.shape == G.shape

    def test_assert_2d_min(self):
        with pytest.raises(AssertionError):
            zeropower_via_newtonschulz5(torch.randn(8), steps=1)

    def test_dtype_bfloat16(self):
        # NS5 internally upcasts to bf16
        G = torch.randn(4, 4)
        out = zeropower_via_newtonschulz5(G, steps=1)
        assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# muon_update / adam_update
# ---------------------------------------------------------------------------
class TestMuonUpdate:
    def test_returns_same_outer_size(self):
        grad = torch.randn(8, 16)
        momentum = torch.zeros_like(grad)
        out = muon_update(grad, momentum, ns_steps=2)
        assert out.shape[0] == 8

    def test_conv_4d_flattened(self):
        grad = torch.randn(8, 4, 3, 3)  # conv weight
        momentum = torch.zeros_like(grad)
        out = muon_update(grad, momentum, ns_steps=2)
        # After view(len(update), -1): [8, 36]
        assert out.shape[0] == 8


class TestAdamUpdate:
    def test_shape_preserved(self):
        grad = torch.randn(4, 8)
        buf1 = torch.zeros_like(grad)
        buf2 = torch.zeros_like(grad)
        out = adam_update(grad, buf1, buf2, step=1,
                          betas=(0.9, 0.999), eps=1e-8)
        assert out.shape == grad.shape

    def test_first_step_nonzero_with_nonzero_grad(self):
        grad = torch.ones(4)
        buf1 = torch.zeros_like(grad)
        buf2 = torch.zeros_like(grad)
        out = adam_update(grad, buf1, buf2, step=1,
                          betas=(0.9, 0.999), eps=1e-8)
        assert (out != 0).any()


# ---------------------------------------------------------------------------
# Muon class
# ---------------------------------------------------------------------------
class TestMuonOptimizer:
    def _make_model(self):
        return nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))

    def test_construction_dual_group(self):
        m = self._make_model()
        matrix = [p for p in m.parameters() if p.ndim >= 2]
        scalar = [p for p in m.parameters() if p.ndim < 2]
        opt = Muon([
            dict(params=matrix, use_muon=True),
            dict(params=scalar, use_muon=False),
        ])
        assert len(opt.param_groups) == 2

    def test_default_lr_for_muon_group(self):
        m = self._make_model()
        opt = Muon([dict(params=list(m.parameters()), use_muon=True)])
        assert opt.param_groups[0]['lr'] == 0.02

    def test_default_lr_for_adamw_group(self):
        m = self._make_model()
        opt = Muon([dict(params=list(m.parameters()), use_muon=False)])
        assert opt.param_groups[0]['lr'] == 3e-4

    def test_missing_use_muon_raises(self):
        m = self._make_model()
        with pytest.raises(AssertionError):
            Muon([dict(params=list(m.parameters()))])

    def test_step_updates_params(self):
        torch.manual_seed(0)
        m = self._make_model()
        matrix = [p for p in m.parameters() if p.ndim >= 2]
        scalar = [p for p in m.parameters() if p.ndim < 2]
        before = [p.detach().clone() for p in m.parameters()]
        opt = Muon([
            dict(params=matrix, use_muon=True, lr=0.01),
            dict(params=scalar, use_muon=False, lr=0.01),
        ])
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        opt.step()
        for b, p in zip(before, m.parameters()):
            assert not torch.allclose(b, p)

    def test_step_returns_loss_from_closure(self):
        m = self._make_model()
        matrix = [p for p in m.parameters() if p.ndim >= 2]
        scalar = [p for p in m.parameters() if p.ndim < 2]
        opt = Muon([
            dict(params=matrix, use_muon=True),
            dict(params=scalar, use_muon=False),
        ])

        def closure():
            opt.zero_grad()
            x = torch.randn(2, 8)
            loss = m(x).sum()
            loss.backward()
            return loss

        out = opt.step(closure)
        assert out is not None

    def test_step_skips_none_grad(self):
        m = self._make_model()
        opt = Muon([
            dict(params=list(m.parameters())[:1], use_muon=True),
            dict(params=list(m.parameters())[1:], use_muon=False),
        ])
        # No backward → all grads are None; step should be a no-op.
        opt.step()


# ---------------------------------------------------------------------------
# split_params_for_muon
# ---------------------------------------------------------------------------
class TestSplitParams:
    def test_splits_by_ndim(self):
        m = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 4))
        matrix, scalar = split_params_for_muon(m)
        # Each linear has weight (2D) + bias (1D)
        assert len(matrix) == 2
        assert len(scalar) == 2

    def test_skips_frozen(self):
        m = nn.Linear(8, 4)
        m.weight.requires_grad_(False)
        matrix, scalar = split_params_for_muon(m)
        assert len(matrix) == 0  # weight frozen
        assert len(scalar) == 1  # bias still trainable
