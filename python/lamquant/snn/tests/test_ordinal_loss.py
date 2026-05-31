# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# Unit tests for ordinal_loss — the ordinal (CORAL) + constrained
# (Lagrangian CRITICAL-recall floor) objective for the 4-state SNN
# CR-controller. Tiny synthetic logits/targets, deterministic.

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lamquant" / "snn"))

from lamquant.snn.ordinal_loss import (  # noqa: E402
    ordinal_loss,
    constrained_loss,
    constrained_loss_components,
    NUM_STATES,
    CRITICAL_STATE,
)

torch.manual_seed(1337)


def _one_hot_logits(target: torch.Tensor, margin: float = 8.0) -> torch.Tensor:
    """Build [B, K, T] logits that argmax to `target` (sharp one-hot)."""
    B, T = target.shape
    oh = F.one_hot(target, num_classes=NUM_STATES).float()  # [B, T, K]
    return (oh * margin).permute(0, 2, 1).contiguous()       # [B, K, T]


# ----------------------------------------------------------------------
# Deliverable 1 — ordinal loss penalises off-by-3 >> off-by-1.
# ----------------------------------------------------------------------

def test_ordinal_offby3_costs_more_than_offby1():
    """Predicting QUIET for a true CRITICAL (off-by-3) must cost strictly more
    than predicting INTERESTING for a true CRITICAL (off-by-1)."""
    target = torch.full((1, 1), CRITICAL_STATE, dtype=torch.long)  # true=3
    # Off-by-1: confident prediction of INTERESTING=2.
    logits_offby1 = _one_hot_logits(torch.full((1, 1), 2, dtype=torch.long))
    # Off-by-3: confident prediction of QUIET=0.
    logits_offby3 = _one_hot_logits(torch.full((1, 1), 0, dtype=torch.long))

    l1 = ordinal_loss(logits_offby1, target)
    l3 = ordinal_loss(logits_offby3, target)
    assert l3 > l1, f"off-by-3 ({l3}) must exceed off-by-1 ({l1})"
    # And both exceed a correct prediction.
    l0 = ordinal_loss(_one_hot_logits(target), target)
    assert l1 > l0


def test_ordinal_monotone_in_distance():
    """Loss grows monotonically with tier distance from the true CRITICAL."""
    target = torch.full((1, 1), CRITICAL_STATE, dtype=torch.long)
    losses = [
        float(ordinal_loss(
            _one_hot_logits(torch.full((1, 1), p, dtype=torch.long)), target))
        for p in range(NUM_STATES)
    ]
    # distance 3,2,1,0 -> losses should be decreasing as pred approaches true.
    assert losses[0] > losses[1] > losses[2] > losses[3]


def test_ordinal_correct_is_low():
    """A confident correct prediction yields a small loss for every tier."""
    for s in range(NUM_STATES):
        target = torch.full((2, 3), s, dtype=torch.long)
        logits = _one_hot_logits(target)
        assert float(ordinal_loss(logits, target)) < 0.1


def test_ordinal_weight_emphasis():
    """Per-class weight scales the term for the weighted tier.

    Matches F.cross_entropy weighted semantics: under ``sum`` reduction the
    CRITICAL timestep's term is multiplied by its class weight, so up-weighting
    CRITICAL raises the total. Under weighted-``mean`` the CRITICAL error also
    pulls the (weight-normalised) mean up vs a flat weight, because the heavy
    CRITICAL term dominates the weighted average."""
    # Batch with a mispredicted CRITICAL and a correct QUIET timestep.
    target = torch.tensor([[CRITICAL_STATE, 0]], dtype=torch.long)
    logits = _one_hot_logits(torch.tensor([[0, 0]], dtype=torch.long))  # crit->QUIET
    w_flat = torch.ones(NUM_STATES)
    w_crit = torch.tensor([1.0, 1.0, 1.0, 5.0])
    # sum reduction: absolute scaling of the CRITICAL term.
    base_sum = ordinal_loss(logits, target, weight=w_flat, reduction="sum")
    heavy_sum = ordinal_loss(logits, target, weight=w_crit, reduction="sum")
    assert heavy_sum > base_sum
    # mean reduction: weighted mean is pulled toward the heavy CRITICAL error.
    base_mean = ordinal_loss(logits, target, weight=w_flat)
    heavy_mean = ordinal_loss(logits, target, weight=w_crit)
    assert heavy_mean > base_mean


def test_ordinal_reduction_modes_and_grad():
    target = torch.randint(0, NUM_STATES, (2, 5))
    logits = torch.randn(2, NUM_STATES, 5, requires_grad=True)
    none = ordinal_loss(logits, target, reduction="none")
    assert none.shape == (2, 5)
    s = ordinal_loss(logits, target, reduction="sum")
    m = ordinal_loss(logits, target, reduction="mean")
    assert torch.isfinite(s) and torch.isfinite(m)
    m.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ----------------------------------------------------------------------
# Deliverable 2 — constrained loss increases when CRIT-recall < floor.
# ----------------------------------------------------------------------

def test_constrained_increases_when_crit_recall_below_floor():
    """Same batch, two predictions: when the model's CRITICAL recall is BELOW
    the floor the constrained loss must be strictly larger than when it is
    ABOVE the floor (the hinge fires)."""
    # 8 timesteps, 2 of them true CRITICAL.
    target = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)

    # Good: confident-correct everywhere -> recall ~ 1.0 (>= floor).
    good = _one_hot_logits(target)
    # Bad: CRITICAL timesteps mispredicted as QUIET -> recall ~ 0 (< floor).
    bad_target = target.clone()
    bad_pred = target.clone()
    bad_pred[bad_pred == CRITICAL_STATE] = 0
    bad = _one_hot_logits(bad_pred)

    l_good = constrained_loss(good, target, crit_floor=0.88)
    l_bad = constrained_loss(bad, target, crit_floor=0.88)
    assert l_bad > l_good, f"below-floor ({l_bad}) must exceed above-floor ({l_good})"


def test_constrained_hinge_scales_with_floor():
    """Raising the floor (tighter constraint) on a low-recall batch raises the
    constrained loss (the shortfall, hence the ramp, grows)."""
    target = torch.tensor([[3, 3, 0, 1]], dtype=torch.long)
    # Predict CRITICAL with moderate confidence so recall sits mid-range.
    logits = torch.zeros(1, NUM_STATES, 4)
    logits[:, CRITICAL_STATE, :2] = 1.0   # weak CRITICAL pref on the crit steps
    low_floor = constrained_loss(logits, target, crit_floor=0.30)
    high_floor = constrained_loss(logits, target, crit_floor=0.95)
    assert high_floor > low_floor


def test_constrained_escalation_penalty_charges_overescalation():
    """Escalating true-LOW timesteps to HIGH tiers costs more than keeping them
    low, holding the CRITICAL constraint satisfied."""
    target = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)  # all LOW, no CRIT
    keep_low = _one_hot_logits(target)                        # predict low
    escalate = _one_hot_logits(torch.full_like(target, CRITICAL_STATE))
    l_low = constrained_loss(keep_low, target, escalation_weight=0.5)
    l_high = constrained_loss(escalate, target, escalation_weight=0.5)
    assert l_high > l_low


def test_constrained_no_critical_in_batch_is_finite():
    """A CRITICAL-free batch must not fire the hinge (recall defaults to 1.0)
    and must stay finite + differentiable."""
    target = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    logits = torch.randn(1, NUM_STATES, 5, requires_grad=True)
    loss = constrained_loss(logits, target, crit_floor=0.88)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_constrained_is_drop_in_shape_compatible():
    """Accepts exactly the trainer's (class_logits [B,K,T], target [B,T]) and
    an optional [K] weight, returns a scalar — like F.cross_entropy."""
    B, T = 3, 7
    logits = torch.randn(B, NUM_STATES, T, requires_grad=True)
    target = torch.randint(0, NUM_STATES, (B, T))
    cw = torch.tensor([1.0, 1.2, 1.5, 4.0])
    loss = constrained_loss(logits, target, weight=cw)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_components_helper_keys_and_total():
    target = torch.tensor([[3, 0, 1, 2]], dtype=torch.long)
    logits = torch.randn(1, NUM_STATES, 4)
    comp = constrained_loss_components(logits, target)
    assert set(comp) == {"base", "soft_crit_recall", "hinge", "escalation", "total"}
    assert 0.0 <= comp["soft_crit_recall"] <= 1.0
    assert comp["hinge"] >= 0.0 and comp["escalation"] >= 0.0


# ----------------------------------------------------------------------
# Contract rejection.
# ----------------------------------------------------------------------

def test_contract_rejects_bad_shapes():
    good_t = torch.zeros((2, 4), dtype=torch.long)
    with pytest.raises(AssertionError):
        ordinal_loss(torch.randn(2, 3, 4), good_t)            # K != 4
    with pytest.raises(AssertionError):
        ordinal_loss(torch.randn(2, 4, 4), torch.zeros((2, 5), dtype=torch.long))
    with pytest.raises(AssertionError):
        bad = torch.zeros((2, 4), dtype=torch.long); bad[0, 0] = 9  # out of range
        ordinal_loss(torch.randn(2, 4, 4), bad)
    with pytest.raises(AssertionError):
        ordinal_loss(torch.randn(2, 4, 4), torch.zeros((2, 4)))   # float target
