"""Tests for the scheduler ingredient registry (ADR 0050/0051)."""
from __future__ import annotations

import pytest
import torch

from lamquant.ingredients import build_ingredient, list_ingredients
from lamquant.ingredients.schedules.wsd import WSDScheduler

pytestmark = pytest.mark.l2


def _opt():
    m = torch.nn.Linear(4, 4)
    return torch.optim.SGD(m.parameters(), lr=0.1)


def test_wsd_registered():
    assert "wsd" in list_ingredients("scheduler")


def test_build_wsd_returns_scheduler_and_steps():
    opt = _opt()
    sched = build_ingredient(
        "scheduler", "wsd",
        {"total_epochs": 100, "peak_lr": 1e-3, "warmup_frac": 0.1,
         "decay_frac": 0.1, "min_lr": 1e-6},
        optimizer=opt)
    assert isinstance(sched, WSDScheduler)
    # warmup: lr ramps from ~0 toward peak
    sched.step()
    assert 0 < opt.param_groups[0]["lr"] <= 1e-3
    # advance into the stable phase → peak lr
    for _ in range(20):
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-9
    assert sched.phase in ("stable", "stable∞")


def test_wsd_infinite_when_decay_zero():
    sched = build_ingredient(
        "scheduler", "wsd",
        {"total_epochs": 50, "peak_lr": 1e-3, "decay_frac": 0.0},
        optimizer=_opt())
    assert sched._infinite


def test_wsd_missing_required_fails_closed():
    with pytest.raises(ValueError, match="invalid config"):
        # peak_lr is required (no default)
        build_ingredient("scheduler", "wsd", {"total_epochs": 10},
                         optimizer=_opt())
