"""Tests for the snn_ssm step ingredient (ADR 0050/0051) — the shared SNN step.

Uses a plain Linear (no SelectiveSSM, so clamp_ssm_params is a no-op) — that is
enough to pin the step mechanics + nan-skip + byte-identical equivalence vs the
verbatim inline sequence both SNN trainers used. Needs the neural wheel (the
clamp import); skips when absent.
"""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("lamquant_neural")

import torch  # noqa: E402

from lamquant.ingredients import build_ingredient, list_ingredients  # noqa: E402

pytestmark = pytest.mark.l2


def test_snn_ssm_registered():
    assert "snn_ssm" in list_ingredients("step")


def test_snn_ssm_finite_matches_handrolled_inline():
    """Byte-identical: the ingredient step vs a verbatim copy of the old inline
    nan-skip / backward / clip / step / clamp sequence."""
    from lamquant_neural.models.mamba_ssm_minimal import clamp_ssm_params
    torch.manual_seed(0)
    m1 = torch.nn.Linear(4, 4)
    m2 = copy.deepcopy(m1)
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.1)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.1)
    x = torch.ones(2, 4)

    step = build_ingredient("step", "snn_ssm", {"grad_clip_norm": 1.0})
    did = step((m1(x) * 1000.0).sum(), opt1, m1.parameters(), m1)

    loss2 = (m2(x) * 1000.0).sum()
    assert torch.isfinite(loss2)
    loss2.backward()
    torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
    opt2.step()
    with torch.no_grad():
        clamp_ssm_params(m2)

    assert did is True
    assert torch.allclose(m1.weight, m2.weight)


def test_snn_ssm_nan_skips_without_stepping():
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    w0 = m.weight.detach().clone()
    step = build_ingredient("step", "snn_ssm", {})
    did = step((m(torch.ones(2, 4)) * float("inf")).sum(), opt, m.parameters(), m)
    assert did is False
    assert torch.allclose(m.weight, w0)  # non-finite loss → no optimizer step
