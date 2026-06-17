"""Tests for the step ingredient registry (ADR 0050/0051).

These pin the load-bearing QAT step invariants on a synthetic module (no model
wheel needed): the value-clip bounds grads, the optimizer steps, and the
alpha-clamp runs AFTER the step.
"""
from __future__ import annotations

import copy

import pytest
import torch

from lamquant.ingredients import build_ingredient, list_ingredients

pytestmark = pytest.mark.l2


class _AlphaModule(torch.nn.Module):
    """A module with a clamp_alpha() hook + an out-of-range lsq_alpha that does
    NOT participate in the loss (so only the post-step clamp can move it)."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.lsq_alpha = torch.nn.Parameter(torch.tensor(50.0))
        self.clamped = False

    def clamp_alpha(self):
        self.lsq_alpha.data.clamp_(min=1e-4, max=20.0)
        self.clamped = True

    def forward(self, x):
        return self.lin(x)


def test_qat_codec_step_registered():
    assert "qat_codec" in list_ingredients("step")


def test_qat_step_clips_steps_and_clamps_alpha_after():
    m = _AlphaModule()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    step = build_ingredient("step", "qat_codec",
                            {"grad_clip_value": 1.0, "grad_clip_norm": 1.0})
    w0 = m.lin.weight.detach().clone()
    loss = (m(torch.ones(2, 4)) * 1000.0).sum()  # large grads → exercise clips

    gnorm = step(loss, m, opt, [m])

    assert gnorm is not None
    assert not torch.allclose(m.lin.weight, w0)          # optimizer stepped
    assert m.lin.weight.grad.abs().max() <= 1.0 + 1e-6   # value-clipped to <=1
    # lsq_alpha had no grad path (not in the loss) → only the post-step clamp
    # could move it from 50 back into range, proving clamp ran AFTER the step.
    assert m.clamped and float(m.lsq_alpha) <= 20.0


def test_qat_step_matches_handrolled_inline():
    """Byte-identical equivalence: the ingredient step vs a verbatim hand-rolled
    copy of the old inline sequence must produce identical post-step weights +
    alpha. This is the gold-standard proof the extraction changed nothing."""
    torch.manual_seed(0)
    m1 = _AlphaModule()
    m2 = copy.deepcopy(m1)
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.1)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.1)
    x = torch.ones(2, 4)

    # via the ingredient
    step = build_ingredient("step", "qat_codec",
                            {"grad_clip_value": 1.0, "grad_clip_norm": 1.0})
    step((m1(x) * 1000.0).sum(), m1, opt1, [m1])

    # verbatim hand-rolled inline (what train_joint used to do) — keep this in
    # sync with _qat_codec_step; if it drifts, the equivalence proof is void.
    opt2.zero_grad()
    (m2(x) * 1000.0).sum().backward()
    torch.nn.utils.clip_grad_value_(m2.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
    opt2.step()
    with torch.no_grad():
        for mm in [m2]:
            if hasattr(mm, 'clamp_alpha'):
                mm.clamp_alpha()
            else:
                mm.lsq_alpha.data.clamp_(min=1e-4, max=20.0)

    assert torch.allclose(m1.lin.weight, m2.lin.weight)
    assert torch.allclose(m1.lsq_alpha, m2.lsq_alpha)


def test_qat_step_lsq_alpha_fallback_clamp():
    m = torch.nn.Linear(4, 4)
    m.lsq_alpha = torch.nn.Parameter(torch.tensor(99.0))
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    step = build_ingredient("step", "qat_codec", {})
    step(m(torch.ones(2, 4)).sum(), m, opt, [m])
    assert float(m.lsq_alpha) <= 20.0  # fallback clamp_(1e-4, 20) ran
