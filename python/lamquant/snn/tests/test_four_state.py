# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# Unit tests for four_state.derive_4state_target — the pure 4-state
# CR-controller target derivation. Tiny synthetic tensors, deterministic.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lamquant" / "snn"))

from lamquant.snn.four_state import (  # noqa: E402
    derive_4state_target,
    _l3_rms_pooled,
    NUM_STATES,
    LEVEL_TABLE_4,
    CR_TABLE_4,
    STATE_NAMES,
)
from lamquant_neural.models.heads import LEVEL_TABLE_4 as HEAD_LEVEL_TABLE_4  # noqa: E402


def test_basic_four_state_mapping():
    """seizure->3, active->2, high-rms-quiet->1, low-rms-quiet->0."""
    labels = np.zeros((8, 4), dtype=np.int64)
    labels[3, 2] = 1  # active at t2
    labels[5, 3] = 2  # seizure at t3
    l3 = np.zeros((21, 4), dtype=np.float32)
    l3[:, 0] = 0.001  # low rms  -> QUIET
    l3[:, 1] = 1.0    # high rms -> BASELINE
    l3[:, 2] = 0.5
    l3[:, 3] = 0.5
    tgt = derive_4state_target(labels, l3, quiet_rms_threshold=0.1)
    assert tgt.tolist() == [0, 1, 2, 3]
    assert tgt.dtype == np.int64


def test_seizure_beats_active_same_timestep():
    """CRITICAL has precedence over INTERESTING within one timestep."""
    labels = np.zeros((8, 1), dtype=np.int64)
    labels[0, 0] = 1  # one group active
    labels[1, 0] = 2  # another group seizure
    l3 = np.ones((21, 1), dtype=np.float32) * 0.5
    assert derive_4state_target(labels, l3, 0.1)[0] == 3


def test_active_overrides_rms():
    """An active timestep is INTERESTING regardless of L3 energy."""
    labels = np.zeros((8, 2), dtype=np.int64)
    labels[0, :] = 1
    l3 = np.zeros((21, 2), dtype=np.float32)  # zero energy
    out = derive_4state_target(labels, l3, 0.1)
    assert out.tolist() == [2, 2]


def test_resolution_mismatch_pools_l3():
    """l3 with T_l3 != T must be pooled to the label resolution."""
    labels = np.zeros((8, 3), dtype=np.int64)  # all quiet
    l3 = np.ones((21, 6), dtype=np.float32) * 2.0  # high energy, 6 timesteps
    out = derive_4state_target(labels, l3, 0.1)
    assert out.shape == (3,)
    assert (out == 1).all()  # high-rms quiet -> BASELINE


def test_threshold_boundary_is_strict_gt():
    """rms == threshold stays QUIET (split is rms > threshold)."""
    labels = np.zeros((8, 1), dtype=np.int64)
    l3 = np.ones((21, 1), dtype=np.float32) * 0.5  # rms == 0.5
    assert derive_4state_target(labels, l3, 0.5)[0] == 0   # equal -> QUIET
    assert derive_4state_target(labels, l3, 0.49)[0] == 1  # above -> BASELINE


def test_rms_pool_identity_when_matching():
    """_l3_rms_pooled is a no-op resolution-wise when T_l3 == target_T."""
    l3 = np.random.default_rng(0).standard_normal((21, 10)).astype(np.float32)
    rms = _l3_rms_pooled(l3, 10)
    expect = np.sqrt(np.mean(l3.astype(np.float64) ** 2, axis=0))
    assert rms.shape == (10,)
    np.testing.assert_allclose(rms, expect, rtol=1e-6)


def test_contract_rejects_bad_shapes():
    l3 = np.ones((21, 4), dtype=np.float32)
    with pytest.raises(AssertionError):
        derive_4state_target(np.zeros((4, 4), dtype=np.int64), l3, 0.1)  # not 8 groups
    with pytest.raises(AssertionError):
        derive_4state_target(np.zeros((8, 4), dtype=np.int64),
                             np.ones((10, 4), dtype=np.float32), 0.1)  # not 21 ch
    with pytest.raises(AssertionError):
        bad = np.zeros((8, 4), dtype=np.int64); bad[0, 0] = 3  # out-of-range label
        derive_4state_target(bad, l3, 0.1)


def test_tables_aligned_with_heads():
    """four_state level table must match heads.LEVEL_TABLE_4 (state->FSQ)."""
    assert tuple(LEVEL_TABLE_4) == tuple(int(x) for x in HEAD_LEVEL_TABLE_4)
    assert len(LEVEL_TABLE_4) == NUM_STATES == 4
    assert len(CR_TABLE_4) == 4 and len(STATE_NAMES) == 4
    # Lower state = higher CR (more compression).
    assert list(CR_TABLE_4) == sorted(CR_TABLE_4, reverse=True)
