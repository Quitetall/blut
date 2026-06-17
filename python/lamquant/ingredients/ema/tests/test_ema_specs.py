"""Tests for the EMA ingredient registry (ADR 0050/0051)."""
from __future__ import annotations

import torch

from lamquant.ingredients import build_ingredient, list_ingredients


def test_avg_model_registered():
    assert "avg_model" in list_ingredients("ema")


def test_enabled_wraps_model():
    m = torch.nn.Linear(4, 4)
    ema = build_ingredient("ema", "avg_model", {"decay": 0.99}, model=m)
    from torch.optim.swa_utils import AveragedModel
    assert isinstance(ema, AveragedModel)
    # The averaged module mirrors the source params (same shapes).
    assert ema.module.weight.shape == m.weight.shape


def test_disabled_returns_none():
    m = torch.nn.Linear(4, 4)
    assert build_ingredient("ema", "avg_model", {"enabled": False}, model=m) is None


def test_decay_out_of_range_fails_closed():
    import pytest
    m = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="decay must be in"):
        build_ingredient("ema", "avg_model", {"decay": 1.5}, model=m)


def test_update_averages_after_first_copy():
    m = torch.nn.Linear(2, 2)
    ema = build_ingredient("ema", "avg_model", {"decay": 0.5}, model=m)
    # torch's AveragedModel COPIES the live weights on the first update
    # (n_averaged == 0) and only EMA-averages from the second update on.
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update_parameters(m)
    assert torch.allclose(ema.module.weight, torch.ones_like(ema.module.weight))
    # second update with new weights → EMA-averaged, strictly between 1 and 3.
    with torch.no_grad():
        m.weight.fill_(3.0)
    ema.update_parameters(m)
    w = ema.module.weight
    assert (w > 1).all() and (w < 3).all()
