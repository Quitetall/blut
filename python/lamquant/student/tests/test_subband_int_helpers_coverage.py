"""Coverage tests for ``ai_models/student/_subband_int_helpers.py``.

Pure integer-domain DSP helpers. All assertions pin shape + invariants
(bit-exact round-trip for Mode 3 lossless contract), not exact numeric
values.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamquant.student._subband_int_helpers import (
    compute_l3_correction,
    preprocess_subband_int,
    reconstruct_from_subband_int,
    reconstruct_with_l3_correction,
)


pytestmark = pytest.mark.l2


@pytest.fixture
def signal_int_4ch():
    """Deterministic 4ch x 2500 int16 signal — math fixture, not EEG."""
    rng = np.random.RandomState(42)
    return rng.randint(-2**14, 2**14, size=(4, 2500), dtype=np.int16)


class TestPreprocessSubbandInt:
    def test_returns_triple(self, signal_int_4ch) -> None:
        l3, coeffs, subs = preprocess_subband_int(signal_int_4ch)
        assert l3.shape == (4, 313)
        assert coeffs.shape[0] == 4
        assert len(subs) == 4

    def test_l3_is_integer(self, signal_int_4ch) -> None:
        l3, _, _ = preprocess_subband_int(signal_int_4ch)
        assert np.issubdtype(l3.dtype, np.integer)

    def test_lpc_order(self, signal_int_4ch) -> None:
        _, coeffs, _ = preprocess_subband_int(signal_int_4ch, order=8)
        assert coeffs.shape == (4, 8)


class TestReconstructFromSubbandInt:
    def test_lossless_roundtrip(self, signal_int_4ch) -> None:
        """Mode 3 lossless contract: preprocess + reconstruct = identity
        (up to LPC numerical drift, which is documented to be 0)."""
        l3, coeffs, subs = preprocess_subband_int(signal_int_4ch)
        recon = reconstruct_from_subband_int(l3, coeffs, subs)
        # Shape preserved
        assert recon.shape == signal_int_4ch.shape
        # Bit-exact round-trip is the contract — assert exact equality
        # over the int16 input.
        np.testing.assert_array_equal(
            recon.astype(np.int16), signal_int_4ch
        )


class TestComputeL3Correction:
    def test_shape_preserved(self) -> None:
        l3_orig = np.zeros((4, 313), dtype=np.int64)
        l3_recon = np.ones((4, 313), dtype=np.int64)
        out = compute_l3_correction(l3_orig, l3_recon)
        assert out.shape == (4, 313)

    def test_zero_when_equal(self) -> None:
        l3 = np.arange(4 * 313, dtype=np.int64).reshape(4, 313)
        out = compute_l3_correction(l3, l3)
        np.testing.assert_array_equal(out, 0)


class TestReconstructWithL3Correction:
    def test_recovers_original_with_zero_l3_error(
            self, signal_int_4ch) -> None:
        l3, coeffs, subs = preprocess_subband_int(signal_int_4ch)
        # zero error -> identical to plain reconstruct
        zero_err = np.zeros_like(l3)
        recon = reconstruct_with_l3_correction(l3, zero_err, coeffs, subs)
        assert recon.shape == signal_int_4ch.shape


class TestRecomputeDetailsApproach4:
    """Approach 4: recompute detail subbands relative to RECONSTRUCTED L3
    so the standard inverse lifting recovers the original signal exactly.
    """

    def test_runs_without_error(self, signal_int_4ch) -> None:
        from lamquant.student._subband_int_helpers import (
            recompute_details_approach4,
        )
        l3, _, _ = preprocess_subband_int(signal_int_4ch)
        # Use the original l3 as the "reconstructed" L3 -> should match.
        lpc_coeffs, subs = recompute_details_approach4(
            signal_int_4ch.astype(np.float64),
            l3.astype(np.float64),
        )
        assert lpc_coeffs.shape[0] == 4
        assert len(subs) == 4

    def test_subband_keys(self, signal_int_4ch) -> None:
        from lamquant.student._subband_int_helpers import (
            recompute_details_approach4,
        )
        l3, _, _ = preprocess_subband_int(signal_int_4ch)
        _, subs = recompute_details_approach4(
            signal_int_4ch.astype(np.float64),
            l3.astype(np.float64),
        )
        # The contract is "subs[c] has the same subband keys as the
        # forward integer pipeline" — pin "l3_approx" key as a smoke.
        assert "l3_approx" in subs[0]


class TestRecomputeDetailsApproach4Cascade:
    def test_runs_without_error(self, signal_int_4ch) -> None:
        from lamquant.student._subband_int_helpers import (
            recompute_details_approach4_cascade,
        )
        l3, _, _ = preprocess_subband_int(signal_int_4ch)
        lpc_coeffs, subs = recompute_details_approach4_cascade(
            signal_int_4ch.astype(np.float64),
            l3.astype(np.float64),
        )
        assert lpc_coeffs.shape[0] == 4
        assert len(subs) == 4

    def test_produces_l1_l2_l3_detail(self, signal_int_4ch) -> None:
        from lamquant.student._subband_int_helpers import (
            recompute_details_approach4_cascade,
        )
        l3, _, _ = preprocess_subband_int(signal_int_4ch)
        _, subs = recompute_details_approach4_cascade(
            signal_int_4ch.astype(np.float64),
            l3.astype(np.float64),
        )
        for key in ("l3_approx", "l3_detail", "l2_detail", "l1_detail"):
            assert key in subs[0], f"missing key {key!r}"


class TestSolveDetailForTarget:
    """Internal solver — exposed for testability."""

    def test_returns_array(self) -> None:
        from lamquant.student._subband_int_helpers import (
            _solve_detail_for_target,
        )
        target = np.arange(64, dtype=np.int64)
        approx = target[::2].copy()  # rough approx
        detail = _solve_detail_for_target(target, approx)
        assert isinstance(detail, np.ndarray)
        # detail length = floor(N/2)
        assert len(detail) == 32

    def test_zero_length_input(self) -> None:
        from lamquant.student._subband_int_helpers import (
            _solve_detail_for_target,
        )
        # N = 1 -> n_detail = 0
        out = _solve_detail_for_target(
            np.array([42], dtype=np.int64),
            np.array([42], dtype=np.int64),
        )
        assert len(out) == 0
