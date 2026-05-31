# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# Unit + smoke tests for spectral.l3_spectral_features — band-power input
# features for the 4-state CR controller. Deterministic synthetic signals.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lamquant" / "snn"))

from lamquant.snn.spectral import (  # noqa: E402
    l3_spectral_features,
    l3_spectral_features_np,
    l3_spectral_features_torch,
    build_augmented_input,
    BAND_EDGES_HZ,
    BAND_NAMES,
    K_BANDS,
    L3_CHANNELS,
    L3_T,
    L3_SEQ_FS,
    WINDOW_SECONDS,
    AUGMENTED_IN_CHANNELS,
)

ALPHA_BAND = BAND_NAMES.index("alpha")   # band index for the 10 Hz test


def _band_slice(feat: np.ndarray, ch: int, band: int) -> np.ndarray:
    """Return [T] feature for (channel ch, band) from a [21*K, T] feat array.

    Layout is band-major within channel: row = ch*K_BANDS + band.
    """
    return feat[ch * K_BANDS + band]


# ----------------------------------------------------------------------
# Shape / dtype contract.
# ----------------------------------------------------------------------

def test_shape_unbatched_np():
    l3 = np.zeros((L3_CHANNELS, L3_T), dtype=np.float32)
    out = l3_spectral_features_np(l3)
    assert out.shape == (L3_CHANNELS * K_BANDS, L3_T)
    assert out.dtype == np.float32


def test_shape_batched_np():
    l3 = np.zeros((5, L3_CHANNELS, L3_T), dtype=np.float32)
    out = l3_spectral_features_np(l3)
    assert out.shape == (5, L3_CHANNELS * K_BANDS, L3_T)


def test_shape_unbatched_torch():
    l3 = torch.zeros(L3_CHANNELS, L3_T)
    out = l3_spectral_features_torch(l3)
    assert tuple(out.shape) == (L3_CHANNELS * K_BANDS, L3_T)


def test_shape_batched_torch():
    l3 = torch.zeros(3, L3_CHANNELS, L3_T)
    out = l3_spectral_features_torch(l3)
    assert tuple(out.shape) == (3, L3_CHANNELS * K_BANDS, L3_T)


def test_dispatch_by_type():
    l3_np = np.zeros((L3_CHANNELS, L3_T), dtype=np.float32)
    l3_pt = torch.zeros(L3_CHANNELS, L3_T)
    assert isinstance(l3_spectral_features(l3_np), np.ndarray)
    assert isinstance(l3_spectral_features(l3_pt), torch.Tensor)
    with pytest.raises(TypeError):
        l3_spectral_features([1, 2, 3])


# ----------------------------------------------------------------------
# THE headline unit test: a 10 Hz sine in one channel ⇒ ALPHA band power
# dominant for THAT channel (10 Hz ∈ alpha 8-13 Hz), and that channel's
# alpha dominates the alpha feature across all channels.
# ----------------------------------------------------------------------

def _make_sine_l3(freq_hz: float, ch: int) -> np.ndarray:
    """[21, 313] L3 with a pure sine at `freq_hz` in channel `ch`, else zero.

    The sine is sampled at the L3 sequence rate L3_SEQ_FS so `freq_hz` is the
    real frequency the feature extractor sees on the L3 timestep axis.
    """
    t = np.arange(L3_T, dtype=np.float64) / L3_SEQ_FS   # seconds
    l3 = np.zeros((L3_CHANNELS, L3_T), dtype=np.float32)
    l3[ch] = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return l3


def test_10hz_sine_alpha_dominant():
    """10 Hz sine in channel 7 → alpha band is the dominant band for ch7."""
    ch = 7
    l3 = _make_sine_l3(10.0, ch)
    feat = l3_spectral_features_np(l3)   # [21*K, 313]

    # Within channel 7, average each band's power over the (stable interior of
    # the) time axis and confirm alpha wins.
    interior = slice(L3_T // 4, 3 * L3_T // 4)
    band_means = np.array([
        _band_slice(feat, ch, b)[interior].mean() for b in range(K_BANDS)
    ])
    assert int(band_means.argmax()) == ALPHA_BAND, (
        f"expected alpha (idx {ALPHA_BAND}) dominant, got band means "
        f"{dict(zip(BAND_NAMES, band_means.round(3)))}"
    )
    # Alpha should beat the next-strongest band by a clear margin.
    sorted_means = np.sort(band_means)
    assert sorted_means[-1] > 2.0 * max(sorted_means[-2], 1e-6), (
        f"alpha not clearly dominant: {dict(zip(BAND_NAMES, band_means.round(3)))}"
    )


def test_10hz_sine_localised_to_its_channel():
    """The alpha energy concentrates in the channel carrying the 10 Hz sine."""
    ch = 7
    l3 = _make_sine_l3(10.0, ch)
    feat = l3_spectral_features_np(l3)
    interior = slice(L3_T // 4, 3 * L3_T // 4)
    alpha_per_ch = np.array([
        _band_slice(feat, c, ALPHA_BAND)[interior].mean()
        for c in range(L3_CHANNELS)
    ])
    assert int(alpha_per_ch.argmax()) == ch, (
        f"alpha power should peak at channel {ch}, peaked at {alpha_per_ch.argmax()}"
    )
    # Every other channel is silent (zero input) → ~zero alpha power.
    others = np.delete(alpha_per_ch, ch)
    assert alpha_per_ch[ch] > 10.0 * max(others.max(), 1e-6)


def test_delta_sine_delta_dominant():
    """A 2 Hz sine (delta band) → delta dominant — guards band assignment."""
    ch = 3
    l3 = _make_sine_l3(2.0, ch)
    feat = l3_spectral_features_np(l3)
    interior = slice(L3_T // 4, 3 * L3_T // 4)
    band_means = np.array([
        _band_slice(feat, ch, b)[interior].mean() for b in range(K_BANDS)
    ])
    assert int(band_means.argmax()) == BAND_NAMES.index("delta"), (
        f"expected delta dominant for 2 Hz, got "
        f"{dict(zip(BAND_NAMES, band_means.round(3)))}"
    )


# ----------------------------------------------------------------------
# Numpy and torch paths agree (the two paths are the same math).
# ----------------------------------------------------------------------

def test_np_torch_parity():
    rng = np.random.default_rng(0)
    l3 = rng.standard_normal((2, L3_CHANNELS, L3_T)).astype(np.float32)
    out_np = l3_spectral_features_np(l3)
    out_pt = l3_spectral_features_torch(torch.from_numpy(l3)).numpy()
    assert out_np.shape == out_pt.shape
    # Both go through float64 FFT internally; agree to tight tolerance.
    np.testing.assert_allclose(out_np, out_pt, rtol=1e-4, atol=1e-4)


def test_zero_input_zero_features():
    """Silent L3 ⇒ all-zero band power (log1p(0) == 0). No NaNs."""
    l3 = np.zeros((L3_CHANNELS, L3_T), dtype=np.float32)
    out = l3_spectral_features_np(l3)
    assert np.all(out == 0.0)
    assert np.isfinite(out).all()


def test_no_nan_on_random():
    rng = np.random.default_rng(7)
    l3 = (rng.standard_normal((4, L3_CHANNELS, L3_T)) * 100).astype(np.float32)
    out_np = l3_spectral_features_np(l3)
    out_pt = l3_spectral_features_torch(torch.from_numpy(l3))
    assert np.isfinite(out_np).all()
    assert torch.isfinite(out_pt).all()


# ----------------------------------------------------------------------
# Integration helper: augmented input + autograd.
# ----------------------------------------------------------------------

def test_augmented_input_shape_and_layout():
    l3 = torch.randn(6, L3_CHANNELS, L3_T)
    aug = build_augmented_input(l3)
    assert tuple(aug.shape) == (6, AUGMENTED_IN_CHANNELS, L3_T)
    assert AUGMENTED_IN_CHANNELS == 21 + 21 * K_BANDS == 105
    # First 21 channels are the untouched raw L3.
    torch.testing.assert_close(aug[:, :L3_CHANNELS], l3)


def test_torch_path_is_differentiable():
    l3 = torch.randn(2, L3_CHANNELS, L3_T, requires_grad=True)
    feat = l3_spectral_features_torch(l3)
    feat.sum().backward()
    assert l3.grad is not None
    assert torch.isfinite(l3.grad).all()


def test_feeds_widened_spatial_mix():
    """End-to-end: augmented input flows through a widened spatial_mix and
    the rest of a real MambaSNN, producing the same output contract."""
    from lamquant_neural.models.mamba_ssm_minimal import MambaSNN
    model = MambaSNN(in_channels=AUGMENTED_IN_CHANNELS, d_model=40,
                     d_state=16, n_layers=2, use_subband=True)
    l3 = torch.randn(2, L3_CHANNELS, L3_T)
    aug = build_augmented_input(l3)
    logits, spike_rate, seizure = model(aug)
    assert logits.shape == (2, MambaSNN.NUM_GROUPS, L3_T)
    assert seizure.shape == (2, 1, L3_T)
    assert torch.isfinite(logits).all()


def test_band_table_covers_l3_band():
    """Sanity: bands are contiguous, ordered, inside L3's 0.5-15.6 Hz band."""
    assert K_BANDS == 4
    assert WINDOW_SECONDS == 10.0
    los = [lo for lo, _ in BAND_EDGES_HZ]
    his = [hi for _, hi in BAND_EDGES_HZ]
    assert los == sorted(los) and his == sorted(his)
    assert los[0] == 0.5 and his[-1] <= 15.65
    # Contiguous: each band's hi == next band's lo.
    for i in range(K_BANDS - 1):
        assert his[i] == los[i + 1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
