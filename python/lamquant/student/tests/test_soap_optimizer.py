"""Unit tests for ai_models/student/soap_optimizer.py — Phase 1 quick win.

Covers the SOAP optimizer class: construction with all defaults, single +
multi-step training loop on a small model (so init_preconditioner,
update_preconditioner, and the post-skip projection path all run),
merge_dims tensor reshaping, and the precondition_1d=True branch.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from soap_optimizer import SOAP

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# SOAP construction
# ---------------------------------------------------------------------------
class TestSOAPConstruction:
    def test_defaults(self):
        m = nn.Linear(8, 16)
        opt = SOAP(m.parameters())
        g = opt.param_groups[0]
        assert g["lr"] == 3e-3
        assert g["betas"] == (0.95, 0.95)
        assert g["eps"] == 1e-8
        assert g["weight_decay"] == 0.01
        assert g["precondition_frequency"] == 10
        assert g["merge_dims"] is False
        assert g["precondition_1d"] is False
        assert g["correct_bias"] is True
        assert opt._data_format == "channels_first"

    def test_custom_lr_and_betas(self):
        m = nn.Linear(4, 8)
        opt = SOAP(m.parameters(), lr=1e-4, betas=(0.9, 0.99))
        g = opt.param_groups[0]
        assert g["lr"] == 1e-4
        assert g["betas"] == (0.9, 0.99)


# ---------------------------------------------------------------------------
# merge_dims tensor reshaping
# ---------------------------------------------------------------------------
class TestMergeDims:
    def test_no_merge_when_within_cap(self):
        opt = SOAP(nn.Linear(4, 4).parameters())
        g = torch.randn(4, 4)
        out = opt.merge_dims(g, max_precond_dim=16)
        # 4*4 = 16 fits → single dim
        assert out.numel() == 16

    def test_merges_small_dims(self):
        opt = SOAP(nn.Linear(4, 4).parameters())
        g = torch.randn(2, 2, 2, 2)  # numel=16
        out = opt.merge_dims(g, max_precond_dim=16)
        assert out.numel() == 16

    def test_preserves_large_dim_alone(self):
        opt = SOAP(nn.Linear(4, 4).parameters())
        g = torch.randn(16, 4)
        out = opt.merge_dims(g, max_precond_dim=20)
        # 16 stays separate, then 4 appended
        assert 16 in out.shape


# ---------------------------------------------------------------------------
# SOAP.step — end-to-end training loop
# ---------------------------------------------------------------------------
class TestSOAPStep:
    def _make_model(self):
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def test_single_step_updates_state(self):
        m = self._make_model()
        opt = SOAP(m.parameters(), lr=1e-3)
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        opt.step()
        # State should now have entries for each param
        assert len(opt.state) == len(list(m.parameters()))

    def test_multi_step_runs_preconditioner_update(self):
        m = self._make_model()
        opt = SOAP(m.parameters(), lr=1e-3, precondition_frequency=2)
        for _ in range(5):
            x = torch.randn(4, 8)
            opt.zero_grad()
            loss = m(x).sum()
            loss.backward()
            opt.step()
        # After 5 steps with freq=2, preconditioner must have rotated
        # (at least one param has step >= 2).
        steps = [s.get("step", 0) for s in opt.state.values()]
        assert max(steps) >= 2

    def test_skips_none_grad(self):
        m = self._make_model()
        opt = SOAP(m.parameters(), lr=1e-3)
        # No backward — all grads None
        opt.step()  # must not raise
        assert opt.state == {}

    def test_step_returns_closure_loss(self):
        m = self._make_model()
        opt = SOAP(m.parameters(), lr=1e-3)

        def closure():
            # SOAP.step is decorated @torch.no_grad — must re-enable for
            # the forward + backward inside the closure.
            with torch.enable_grad():
                x = torch.randn(2, 8)
                loss = m(x).sum()
                opt.zero_grad()
                loss.backward()
                return loss

        out = opt.step(closure)
        assert out is not None

    def test_precondition_1d_branch(self):
        # 1D params (biases) — turn on precondition_1d
        m = self._make_model()
        opt = SOAP(m.parameters(), lr=1e-3, precondition_1d=True)
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        opt.step()
        opt.step()

    def test_merge_dims_branch(self):
        m = self._make_model()
        opt = SOAP(m.parameters(), lr=1e-3, merge_dims=True)
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        opt.step()
        opt.step()

    def test_zero_weight_decay(self):
        m = self._make_model()
        opt = SOAP(m.parameters(), lr=1e-3, weight_decay=0.0)
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        opt.step()
        opt.step()

    def test_no_correct_bias_branch(self):
        m = self._make_model()
        opt = SOAP(m.parameters(), lr=1e-3, correct_bias=False)
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        opt.step()
        opt.step()

    def test_explicit_shampoo_beta(self):
        m = self._make_model()
        opt = SOAP(m.parameters(), lr=1e-3, shampoo_beta=0.9)
        x = torch.randn(4, 8)
        loss = m(x).sum()
        loss.backward()
        opt.step()
        opt.step()

    def test_params_actually_change(self):
        m = self._make_model()
        before = [p.detach().clone() for p in m.parameters()]
        opt = SOAP(m.parameters(), lr=1e-2)
        for _ in range(3):
            x = torch.randn(4, 8)
            opt.zero_grad()
            loss = m(x).sum()
            loss.backward()
            opt.step()
        # At least one param should have changed
        any_changed = any(
            not torch.allclose(b, p)
            for b, p in zip(before, m.parameters())
        )
        assert any_changed
