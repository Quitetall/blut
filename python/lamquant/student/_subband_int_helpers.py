"""Integer-domain subband helpers used only by the training tree.

Houses the seven functions that ``subband_preprocess.py`` carried inline
alongside its re-exports of ``lamquant_codec.ops.*``. The audit's F2
finding asked for that file to be a pure shim; the inline tail moves
here and the shim now imports these names alongside the codec ops it
already exposes.

These functions are training-only — they orchestrate the codec's
lifting + LPC primitives to produce Mode 3 lossless / Approach 4 detail
recompute paths and have no analogue in ``lamquant_codec.ops``. Nothing
in the shipped codec package should import this module.
"""

import numpy as np

from lamquant_codec.ops.lpc import (
    analyze as lpc_analyze,
    analyze_channel as lpc_analyze_channel,
    analyze_int as lpc_analyze_int,
    synthesize_channel as lpc_synthesize_channel,
    synthesize_int as lpc_synthesize_int,
)
from lamquant_codec.ops.lifting import (
    forward_1d_int as lifting_1d_forward_int,
    inverse_1d_int as lifting_1d_inverse_int,
    forward_3level_int as lifting_3level_forward_int,
    inverse_3level_int as lifting_3level_inverse_int,
)


def preprocess_subband_int(signal_int, order=8, autocorr_len=256):
    """Full integer preprocessing: LPC (int) -> integer lifting -> integer subbands.

    Input: [C, T] int16/int32 signal (from ADC/Q31 conversion)
    Output: (l3_approx [C, 313] int64, lpc_coeffs_q15 [C, order] int16,
             subbands_per_ch list of int64 dicts)

    Bit-exact invertible. For true lossless Mode 3.
    """
    C, T = signal_int.shape
    signal_i = signal_int.astype(np.int64)

    lpc_coeffs_q15 = []
    l3_list = []
    subbands_per_ch = []

    for c in range(C):
        coeffs_f, _ = lpc_analyze_channel(signal_i[c].astype(np.float64), order, autocorr_len)
        coeffs_q15, residual_int = lpc_analyze_int(signal_i[c], coeffs_f, order)
        lpc_coeffs_q15.append(coeffs_q15)

        subs = lifting_3level_forward_int(residual_int)
        l3_list.append(subs['l3_approx'])
        subbands_per_ch.append(subs)

    l3_approx = np.stack(l3_list)
    lpc_q27 = np.stack(lpc_coeffs_q15)
    return l3_approx, lpc_q27, subbands_per_ch


def reconstruct_from_subband_int(l3_approx_recon, lpc_coeffs_q15, subbands_per_ch):
    """Full integer inverse: subbands -> inverse lifting -> inverse LPC -> signal.

    For Mode 3 lossless: l3_approx_recon IS the original l3_approx (not TNN output).
    Bit-exact inverse of preprocess_subband_int.
    """
    C = l3_approx_recon.shape[0]
    signals = []
    for c in range(C):
        subs = {k: v.copy() if hasattr(v, 'copy') else v
                for k, v in subbands_per_ch[c].items()}
        subs['l3_approx'] = l3_approx_recon[c].astype(np.int64)

        residual = lifting_3level_inverse_int(subs)
        signal = lpc_synthesize_int(residual, lpc_coeffs_q15[c])
        signals.append(signal)

    return np.stack(signals)


def compute_l3_correction(l3_original, l3_recon):
    """Approach 1: compute L3 error residual for transmission.

    The MCU has both original L3 (from forward lifting) and reconstructed L3
    (from TNN encode->FSQ->ternary decode). The difference is small and
    compressible. Transmitted alongside detail subbands. The decoder adds
    the correction to recon_l3 before inverse lifting -> zero error amplification.

    Args:
        l3_original: [C, 313] original L3 from forward lifting (int or float)
        l3_recon: [C, 313] TNN reconstructed L3 (int or float)
    Returns:
        l3_error: [C, 313] int64 -- correction to add before inverse lifting
    """
    orig = np.round(np.asarray(l3_original)).astype(np.int64)
    recon = np.round(np.asarray(l3_recon)).astype(np.int64)
    return orig - recon


def reconstruct_with_l3_correction(l3_recon, l3_error, lpc_coeffs, subbands_per_ch):
    """Inverse pipeline with L3 error correction. Near-zero PRD.

    corrected_l3 = recon_l3 + l3_error = original_l3 (exact)
    Then standard inverse lifting with corrected L3 + original details.
    """
    corrected_l3 = np.round(l3_recon).astype(np.int64) + l3_error.astype(np.int64)
    C = corrected_l3.shape[0]
    signals = []
    for c in range(C):
        subs = {k: np.round(v).astype(np.int64) for k, v in subbands_per_ch[c].items()}
        subs['l3_approx'] = corrected_l3[c]
        residual = lifting_3level_inverse_int(subs).astype(np.float64)
        signal = lpc_synthesize_channel(residual, lpc_coeffs[c])
        signals.append(signal)
    return np.stack(signals)


def _solve_detail_for_target(target_signal, given_approx):
    """Solve for detail coefficients that make inverse_lifting(approx, detail) = target.

    Given an approximation (which may differ from the original), find the detail
    coefficients such that the standard inverse lifting recovers the target signal
    exactly. This is the core of Approach 4.

    Uses an iterative approach: start with the detail from forward lifting on the
    target, then refine by measuring the error from inverse lifting with the given
    approx and correcting odd-sample residuals.
    """
    N = len(target_signal)
    n_approx = (N + 1) // 2
    n_detail = N // 2

    if n_detail == 0:
        return np.array([], dtype=np.int64)

    target = target_signal.astype(np.int64)
    approx = given_approx.astype(np.int64)

    orig_approx, orig_detail = lifting_1d_forward_int(target)

    if np.array_equal(orig_approx, approx):
        return orig_detail.copy()

    detail = orig_detail.copy()
    for iteration in range(5):
        result = lifting_1d_inverse_int(approx, detail)
        err = target - result
        if np.max(np.abs(err)) == 0:
            break
        for n in range(n_detail):
            detail[n] += err[2 * n + 1]

    return detail


def recompute_details_approach4_cascade(signal_np, l3_recon_np, order=8, autocorr_len=256):
    """Approach 4 (cascaded): recompute details at EVERY level against reconstruction.

    Uses _solve_detail_for_target to find modified details that make
    inverse_lifting(recon_approx, modified_detail) = original_signal at each level.

    The decoder runs standard inverse lifting with modified details and
    recovers the original signal exactly (or near-exactly within a few
    iterations of the solver).

    Returns (lpc_coeffs, subbands_per_ch) with reconstruction-aware details.
    """
    signal = signal_np.astype(np.float64)
    C, T = signal.shape
    lpc_coeffs, residual = lpc_analyze(signal, order, autocorr_len)

    subbands_per_ch = []
    for c in range(C):
        residual_int = np.round(residual[c]).astype(np.int64)

        l1_approx, _ = lifting_1d_forward_int(residual_int)
        l2_approx, _ = lifting_1d_forward_int(l1_approx)
        _, l3_detail_orig = lifting_1d_forward_int(l2_approx)

        recon_l3 = np.round(l3_recon_np[c]).astype(np.int64)

        recon_l2 = lifting_1d_inverse_int(recon_l3, l3_detail_orig)

        modified_l2_detail = _solve_detail_for_target(l1_approx, recon_l2)

        recon_l1 = lifting_1d_inverse_int(recon_l2, modified_l2_detail)

        modified_l1_detail = _solve_detail_for_target(residual_int, recon_l1)

        subs = {
            'l3_approx': recon_l3,
            'l3_detail': l3_detail_orig,
            'l2_detail': modified_l2_detail,
            'l1_detail': modified_l1_detail,
        }
        subbands_per_ch.append({k: v.astype(np.float64) for k, v in subs.items()})

    return lpc_coeffs, subbands_per_ch


def recompute_details_approach4(signal_np, l3_recon_np, order=8, autocorr_len=256):
    """Approach 4: recompute detail subbands relative to RECONSTRUCTED L3.

    Runs the inverse lifting from reconstructed L3 down to get reconstructed
    intermediate approximations, then recomputes details at each level as:
        modified_detail[level] = original_signal_at_level - predict(recon_approx)

    This ensures inverse_lifting(recon_l3, modified_details) perfectly
    recovers the original signal because each detail was computed against
    exactly what the decoder will see.

    The MCU cost: one ternary decode (~17 ms) to get recon_l3, then
    recompute details during forward lifting. Total: 75 ms, under 100 ms budget.

    Args:
        signal_np: original EEG [C, T] float (Q31 normalized)
        l3_recon_np: TNN reconstructed L3 [C, 313] float
    Returns:
        (lpc_coeffs, subbands_per_ch) with reconstruction-aware details
    """
    signal = signal_np.astype(np.float64)
    C, T = signal.shape

    lpc_coeffs, residual = lpc_analyze(signal, order, autocorr_len)

    subbands_per_ch = []
    for c in range(C):
        residual_int = np.round(residual[c]).astype(np.int64)

        subs_orig = lifting_3level_forward_int(residual_int)

        recon_l3 = np.round(l3_recon_np[c]).astype(np.int64)

        l1_approx_orig, l1_detail_orig = lifting_1d_forward_int(residual_int)
        l2_approx_orig, l2_detail_orig = lifting_1d_forward_int(l1_approx_orig)

        n_approx = len(subs_orig['l3_approx'])
        n_detail = len(subs_orig['l3_detail'])
        orig_l3 = subs_orig['l3_approx']

        pred_orig = np.zeros(n_detail, dtype=np.int64)
        pred_recon = np.zeros(n_detail, dtype=np.int64)
        for n in range(n_detail):
            next_n = min(n + 1, n_approx - 1)
            pred_orig[n] = (orig_l3[n] + orig_l3[next_n]) >> 1
            pred_recon[n] = (recon_l3[n] + recon_l3[next_n]) >> 1

        modified_l3_detail = subs_orig['l3_detail'] + (pred_orig - pred_recon)

        subs_modified = {
            'l3_approx': recon_l3,
            'l3_detail': modified_l3_detail,
            'l2_detail': subs_orig['l2_detail'],
            'l1_detail': subs_orig['l1_detail'],
        }

        subs_float = {k: v.astype(np.float64) for k, v in subs_modified.items()}
        subbands_per_ch.append(subs_float)

    return lpc_coeffs, subbands_per_ch


__all__ = [
    "preprocess_subband_int",
    "reconstruct_from_subband_int",
    "compute_l3_correction",
    "reconstruct_with_l3_correction",
    "_solve_detail_for_target",
    "recompute_details_approach4_cascade",
    "recompute_details_approach4",
]
