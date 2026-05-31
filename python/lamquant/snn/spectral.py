# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# spectral.py — band-power input features for the 4-state CR controller.
#
# ORACLE FINDING (ADR 0027): QUIET/BASELINE are trivially separable by L3
# energy (the CR side is loss-limited). CRITICAL/INTERESTING are *temporal*
# and *spectral* — per-timestep energy oracles fail them; the events live in
# the band structure (theta/alpha rhythmicity, delta slowing). The fix is to
# hand the MambaSNN backbone explicit per-band power so the SSM can reason
# about *which* band carries the activity, not just total RMS.
#
# This module derives short-time band-power features from the L3 subband and
# returns them as EXTRA input channels. L3 (the level-3 DWT approximation,
# ~0.5-15.6 Hz) is split into K=4 clinical sub-bands:
#
#       delta    0.5 - 4   Hz
#       theta    4   - 8   Hz
#       alpha    8   - 13  Hz
#       low-beta 13  - 15.6 Hz
#
# Per channel, per L3 timestep, we compute a sliding-window short-time power
# in each band, aligned 1:1 to the 313 latent timesteps. Output shape is
# [21*K, T] (or [B, 21*K, T]) — concatenated onto the 21 raw L3 channels by
# the trainer so spatial_mix becomes Linear(21 + 21*K -> 40).
#
# L3 timing: a 10 s window (2500 samples @ 250 Hz) → T=313 L3 timesteps, so
# the L3 *sequence* has an effective sample rate of ~31.3 Hz (Nyquist
# ~15.65 Hz), which lines up with L3's 0.5-15.6 Hz content. Band edges are
# therefore resolved directly on the L3 sequence's own FFT — no resampling.
#
# Two paths, identical math + identical band binning:
#   * l3_spectral_features_np  — numpy, dataset-side (DataLoader worker).
#   * l3_spectral_features_torch — torch, on-device (autograd-safe, GPU).
# `l3_spectral_features` dispatches on input type.
#
# Programming-Bible style: contract assertions, no silent fallback, typed.

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

# ----------------------------------------------------------------------
# Geometry + band table — single source of truth for this module.
# ----------------------------------------------------------------------

L3_CHANNELS = 21           # MambaSNN in_channels (must match spatial_mix).
L3_T = 313                 # preprocess_subband_single output time dim.
WINDOW_SECONDS = 10.0      # one training window = 10 s.

# Effective sample rate of the L3 *sequence* (timesteps per second). 313
# timesteps over a 10 s window. Nyquist ≈ 15.65 Hz ≈ L3 band ceiling, so the
# four clinical bands below sit inside the resolvable range.
L3_SEQ_FS = L3_T / WINDOW_SECONDS   # ≈ 31.3 Hz

# K=4 clinical sub-bands within L3's 0.5-15.6 Hz band. (lo_hz, hi_hz] per band.
BAND_EDGES_HZ: Tuple[Tuple[float, float], ...] = (
    (0.5, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha
    (13.0, 15.6),  # low-beta
)
BAND_NAMES = ("delta", "theta", "alpha", "low_beta")
K_BANDS = len(BAND_EDGES_HZ)
assert len(BAND_NAMES) == K_BANDS

# Short-time analysis window over the L3 sequence, in timesteps. 32 L3
# timesteps ≈ 1.02 s at L3_SEQ_FS — long enough to resolve the 0.5 Hz lower
# edge of delta (period 2 s would need 2 s; we accept that delta's lowest
# bin folds into the window's DC-adjacent bins) while staying short enough to
# localise transients. Stride 1: one band-power vector PER L3 timestep so the
# feature is aligned 1:1 with the 313 output timesteps (centre-aligned,
# reflect-padded at the edges — no silent truncation).
STFT_WINDOW = 32
assert STFT_WINDOW % 2 == 0, "STFT_WINDOW must be even for symmetric centring"


def _band_bin_mask(n_fft: int, fs: float) -> np.ndarray:
    """Boolean [K_BANDS, n_rfft] mask: which rFFT bins fall in each band.

    rFFT bin f maps to frequency ``f * fs / n_fft``. A bin is assigned to band
    ``b`` iff ``lo < freq <= hi`` for that band's (lo, hi]. Bins outside every
    band (e.g. DC, or > 15.6 Hz) belong to no band and are dropped — this is
    intentional, not a silent fallback: only the four clinical bands carry
    feature energy.
    """
    assert isinstance(n_fft, int) and n_fft > 0, f"n_fft must be positive int, got {n_fft!r}"
    assert np.isfinite(fs) and fs > 0, f"fs must be positive finite, got {fs!r}"
    n_rfft = n_fft // 2 + 1
    freqs = np.arange(n_rfft, dtype=np.float64) * (fs / n_fft)  # [n_rfft]
    mask = np.zeros((K_BANDS, n_rfft), dtype=np.float64)
    for b, (lo, hi) in enumerate(BAND_EDGES_HZ):
        mask[b] = ((freqs > lo) & (freqs <= hi)).astype(np.float64)
    # Contract: every band must claim at least one bin at the resolution we
    # use, else the feature for that band is identically zero (a silent dead
    # channel). With n_fft=STFT_WINDOW=32 and fs≈31.3 Hz the bin spacing is
    # ~0.98 Hz, so each band gets ≥2 bins; assert it.
    per_band = mask.sum(axis=1)
    assert (per_band > 0).all(), (
        f"a band has zero rFFT bins at n_fft={n_fft} fs={fs:.3f}: "
        f"per-band bin counts {per_band.tolist()} — widen the band or window"
    )
    return mask


# Precompute the band→bin mask once for the default geometry (numpy path).
_BAND_MASK_NP = _band_bin_mask(STFT_WINDOW, L3_SEQ_FS)  # [K_BANDS, n_rfft]


def _sliding_windows_np(x: np.ndarray, win: int) -> np.ndarray:
    """Centre-aligned sliding windows along the last axis (reflect-padded).

    Args:
        x: ``[..., T]`` float array.
        win: window length (timesteps).

    Returns:
        ``[..., T, win]`` — one length-``win`` window centred on each of the T
        positions. Reflect padding at the edges keeps every output position a
        real window (no zero-padding bias, no truncation of the T axis).
    """
    assert x.ndim >= 1 and x.shape[-1] > 0, f"x must have a non-empty last axis, got {x.shape}"
    T = x.shape[-1]
    half = win // 2
    # Centre window t spans [t-half, t-half+win). Pad `half` on the left and
    # `win-half-1` on the right so the first/last centres stay in-bounds.
    pad_l, pad_r = half, win - half - 1
    pad_width = [(0, 0)] * (x.ndim - 1) + [(pad_l, pad_r)]
    # reflect needs pad < T; for short recordings fall back to edge padding
    # (still real samples, no zeros) — but flag it, don't hide it.
    mode = "reflect" if (pad_l < T and pad_r < T) else "edge"
    xp = np.pad(x, pad_width, mode=mode)
    win_view = np.lib.stride_tricks.sliding_window_view(xp, win, axis=-1)
    assert win_view.shape[-2] == T and win_view.shape[-1] == win, (
        f"sliding-window shape {win_view.shape} != expected (..., {T}, {win})"
    )
    return win_view


def l3_spectral_features_np(l3: np.ndarray) -> np.ndarray:
    """Numpy band-power features (dataset-side path).

    Args:
        l3: ``[21, 313]`` or ``[B, 21, 313]`` float32 L3 subband signal.

    Returns:
        ``[21*K, 313]`` or ``[B, 21*K, 313]`` float32 — per-channel,
        per-band, per-timestep short-time power. K=4 (delta/theta/alpha/
        low-beta). Channel layout is band-major within each EEG channel:
        ``[ch0_delta, ch0_theta, ch0_alpha, ch0_lowbeta, ch1_delta, ...]``.
    """
    assert isinstance(l3, np.ndarray), f"l3 must be ndarray, got {type(l3).__name__}"
    assert l3.dtype.kind == "f", f"l3 must be float, got dtype {l3.dtype}"
    squeeze = False
    if l3.ndim == 2:
        l3 = l3[None]            # [1, 21, T]
        squeeze = True
    assert l3.ndim == 3, f"l3 must be [21,T] or [B,21,T], got shape {l3.shape}"
    B, C, T = l3.shape
    assert C == L3_CHANNELS, f"l3 must have {L3_CHANNELS} channels, got {C}"
    assert T > 0, "l3 has zero time dimension"

    x = l3.astype(np.float64, copy=False)
    # Sliding short-time windows: [B, C, T, win].
    wins = _sliding_windows_np(x, STFT_WINDOW)
    # Remove per-window DC so band power is not contaminated by the running
    # mean (delta's lowest bins otherwise swamp everything).
    wins = wins - wins.mean(axis=-1, keepdims=True)
    # Hann taper to cut spectral leakage between adjacent bands.
    taper = np.hanning(STFT_WINDOW).astype(np.float64)
    wins = wins * taper
    # rFFT over the window axis → power spectrum [B, C, T, n_rfft].
    spec = np.fft.rfft(wins, n=STFT_WINDOW, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2)
    # Band-pool: [B, C, T, n_rfft] x [K, n_rfft] → [B, C, T, K].
    band_pow = np.einsum("bctf,kf->bctk", power, _BAND_MASK_NP)
    # log1p compresses the heavy-tailed power dist → friendlier input scale.
    band_pow = np.log1p(band_pow)
    # Reorder to band-major-within-channel [B, C, K, T] then flatten C*K.
    band_pow = np.transpose(band_pow, (0, 1, 3, 2))      # [B, C, K, T]
    out = band_pow.reshape(B, C * K_BANDS, T).astype(np.float32)
    assert out.shape == (B, C * K_BANDS, T), f"bad output shape {out.shape}"
    if squeeze:
        out = out[0]
    return out


def l3_spectral_features_torch(l3: torch.Tensor) -> torch.Tensor:
    """Torch band-power features (on-device path; autograd-safe).

    Same math + same band binning as :func:`l3_spectral_features_np`, but
    runs on ``l3.device`` (CPU/CUDA) and is differentiable end-to-end so it
    can sit inside the model's forward without a graph break.

    Args:
        l3: ``[21, 313]`` or ``[B, 21, 313]`` float L3 subband signal.

    Returns:
        ``[21*K, 313]`` or ``[B, 21*K, 313]`` float — same layout as the
        numpy path (band-major within channel), same dtype as ``l3``.
    """
    assert isinstance(l3, torch.Tensor), f"l3 must be torch.Tensor, got {type(l3).__name__}"
    assert l3.is_floating_point(), f"l3 must be float, got dtype {l3.dtype}"
    squeeze = False
    if l3.dim() == 2:
        l3 = l3.unsqueeze(0)     # [1, 21, T]
        squeeze = True
    assert l3.dim() == 3, f"l3 must be [21,T] or [B,21,T], got shape {tuple(l3.shape)}"
    B, C, T = l3.shape
    assert C == L3_CHANNELS, f"l3 must have {L3_CHANNELS} channels, got {C}"
    assert T > 0, "l3 has zero time dimension"

    device, dtype = l3.device, l3.dtype
    half = STFT_WINDOW // 2
    pad_l, pad_r = half, STFT_WINDOW - half - 1
    mode = "reflect" if (pad_l < T and pad_r < T) else "replicate"
    # F.pad reflect/replicate operate on the last dim of a [B, C, T] tensor.
    xp = torch.nn.functional.pad(l3, (pad_l, pad_r), mode=mode)  # [B, C, T+win-1]
    # Centre-aligned sliding windows via unfold → [B, C, T, win].
    wins = xp.unfold(dimension=-1, size=STFT_WINDOW, step=1)
    assert wins.shape[-2] == T and wins.shape[-1] == STFT_WINDOW, (
        f"unfold shape {tuple(wins.shape)} != (B,C,{T},{STFT_WINDOW})"
    )
    wins = wins - wins.mean(dim=-1, keepdim=True)            # remove per-window DC
    taper = torch.hann_window(STFT_WINDOW, periodic=False,
                              device=device, dtype=dtype)     # match np.hanning
    wins = wins * taper
    spec = torch.fft.rfft(wins, n=STFT_WINDOW, dim=-1)        # [B,C,T,n_rfft]
    power = spec.real ** 2 + spec.imag ** 2
    band_mask = torch.from_numpy(_BAND_MASK_NP).to(device=device, dtype=dtype)
    band_pow = torch.einsum("bctf,kf->bctk", power, band_mask)  # [B,C,T,K]
    band_pow = torch.log1p(band_pow)
    band_pow = band_pow.permute(0, 1, 3, 2).contiguous()     # [B,C,K,T]
    out = band_pow.reshape(B, C * K_BANDS, T)
    assert out.shape == (B, C * K_BANDS, T), f"bad output shape {tuple(out.shape)}"
    if squeeze:
        out = out.squeeze(0)
    return out


def l3_spectral_features(l3):
    """Dispatch to the numpy or torch path based on input type.

    Args:
        l3: ``np.ndarray`` or ``torch.Tensor``, shape ``[21,313]`` or
            ``[B,21,313]``.

    Returns:
        Same container type as the input, shape ``[21*K,313]`` /
        ``[B,21*K,313]``.
    """
    if isinstance(l3, torch.Tensor):
        return l3_spectral_features_torch(l3)
    if isinstance(l3, np.ndarray):
        return l3_spectral_features_np(l3)
    raise TypeError(
        f"l3 must be np.ndarray or torch.Tensor, got {type(l3).__name__}"
    )


# ----------------------------------------------------------------------
# Integration helper — build the augmented input the trainer feeds the
# backbone. Kept here so the channel layout (raw L3 first, then band-power)
# is defined in exactly ONE place.
# ----------------------------------------------------------------------

# in_channels the trainer must pass to MambaSNN(in_channels=...) /
# nn.Linear(in_channels, d_model) when spectral features are concatenated.
AUGMENTED_IN_CHANNELS = L3_CHANNELS + L3_CHANNELS * K_BANDS   # 21 + 84 = 105


def build_augmented_input(l3: torch.Tensor) -> torch.Tensor:
    """Concat raw L3 with its band-power features along the channel axis.

    The augmented tensor is what the trainer feeds the backbone once
    ``spatial_mix`` is widened to ``Linear(AUGMENTED_IN_CHANNELS -> d_model)``.

    Args:
        l3: ``[B, 21, 313]`` (or ``[21, 313]``) float L3 input.

    Returns:
        ``[B, 105, 313]`` (or ``[105, 313]``) — channels 0:21 are the raw L3,
        channels 21:105 are the 21*K band-power features (band-major within
        channel). Same dtype/device as ``l3``.
    """
    assert isinstance(l3, torch.Tensor), f"l3 must be torch.Tensor, got {type(l3).__name__}"
    feats = l3_spectral_features_torch(l3)
    cat_dim = -2  # channel axis (works for both [C,T] and [B,C,T])
    out = torch.cat([l3, feats], dim=cat_dim)
    exp_c = AUGMENTED_IN_CHANNELS
    assert out.shape[cat_dim] == exp_c, (
        f"augmented channel count {out.shape[cat_dim]} != {exp_c}"
    )
    return out
