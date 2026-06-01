"""ADR-0029 gate + feasibility-first selection — frozen-expectation tests.

Covers the slide-killer (lexicographic ``selection_key``) and the deterministic
high-amplitude fail-safe override (``apply_energy_failsafe``). Values are pinned
against hand-computed expectations, not implementation-derived, so a behaviour
change must update the constant deliberately.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant.snn.four_state import apply_energy_failsafe, STATE_NAMES
from lamquant.snn.train_4state_controller import selection_key

CRIT = len(STATE_NAMES) - 1  # 3


# ----------------------------------------------------------------------
# selection_key — lexicographic feasibility-first (ADR 0029)
# ----------------------------------------------------------------------

def test_selection_key_tuple_shape_and_type():
    k = selection_key({"critical_recall": 0.96, "quiet_specificity": 0.50}, 0.95)
    assert isinstance(k, tuple) and len(k) == 2
    assert k[0] in (0, 1)


def test_feasible_always_outranks_infeasible():
    # A feasible checkpoint with the WORST possible specificity (0.0) must still
    # outrank an infeasible one with the BEST possible recall just below alpha.
    feasible_worst = selection_key({"critical_recall": 0.95, "quiet_specificity": 0.0}, 0.95)
    infeasible_best = selection_key({"critical_recall": 0.9499, "quiet_specificity": 1.0}, 0.95)
    assert feasible_worst == (1, 0.0)
    assert infeasible_best == (0, 0.9499)
    assert feasible_worst > infeasible_best


def test_the_slide_is_killed():
    # The exact failure ADR 0029 records: a slid checkpoint (CRIT_rec 0.49,
    # QB_spec 0.85) scored HIGHER than a safe one under combined_score. Under
    # feasibility-first it can NEVER be selected over the high-recall epoch.
    slid = selection_key({"critical_recall": 0.49, "quiet_specificity": 0.85}, 0.95)
    safe_lowcr = selection_key({"critical_recall": 0.90, "quiet_specificity": 0.16}, 0.95)
    assert slid == (0, 0.49)
    assert safe_lowcr == (0, 0.90)
    assert safe_lowcr > slid  # least-slid wins when none feasible


def test_among_feasible_max_quiet_specificity():
    a = selection_key({"critical_recall": 0.97, "quiet_specificity": 0.60}, 0.95)
    b = selection_key({"critical_recall": 0.99, "quiet_specificity": 0.55}, 0.95)
    assert a == (1, 0.60) and b == (1, 0.55)
    assert a > b  # higher compression at guaranteed safety wins


def test_boundary_equality_is_feasible():
    # CRIT_rec == alpha is feasible (>=, not >).
    assert selection_key({"critical_recall": 0.95, "quiet_specificity": 0.3}, 0.95)[0] == 1


# ----------------------------------------------------------------------
# apply_energy_failsafe — deterministic high-amplitude override (ADR 0029 §3)
# ----------------------------------------------------------------------

def test_failsafe_forces_loud_to_critical():
    T = 8
    l3 = np.ones((21, 16), dtype=np.float64) * 0.1
    l3[:, 8:] = 10.0  # second half loud (pools to second half of T)
    out = apply_energy_failsafe(np.zeros(T, dtype=np.int64), l3, hi_rms_threshold=1.0)
    assert out.shape == (T,) and out.dtype == np.int64
    assert (out[T // 2:] == CRIT).all()
    assert (out[:T // 2] == 0).all()


def test_failsafe_never_lowers_a_tier():
    T = 8
    quiet_l3 = np.ones((21, 16), dtype=np.float64) * 0.01
    already_crit = np.full(T, CRIT, dtype=np.int64)
    out = apply_energy_failsafe(already_crit, quiet_l3, hi_rms_threshold=1.0)
    assert (out == CRIT).all()  # input copy untouched, never demoted


def test_failsafe_returns_copy_not_mutate():
    T = 4
    states = np.zeros(T, dtype=np.int64)
    l3 = np.ones((21, 8), dtype=np.float64) * 10.0
    out = apply_energy_failsafe(states, l3, hi_rms_threshold=1.0)
    assert (states == 0).all()      # input not mutated
    assert (out == CRIT).all()


def test_failsafe_rejects_bad_shapes():
    with pytest.raises(AssertionError):
        apply_energy_failsafe(np.zeros((2, 2), dtype=np.int64), np.ones((21, 8)), 1.0)
    with pytest.raises(AssertionError):
        apply_energy_failsafe(np.zeros(4, dtype=np.int64), np.ones((8, 8)), 1.0)  # wrong C
    with pytest.raises(AssertionError):
        apply_energy_failsafe(np.zeros(4, dtype=np.int64), np.ones((21, 8)), np.nan)
