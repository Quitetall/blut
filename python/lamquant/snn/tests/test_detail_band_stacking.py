"""Regression tests for detail-band stacking (ADR 0031).

Pins the two invariants the ADR-0031 input-limitation experiment depends on:

  1. `detail_stack_in_channels` must equal the actual channel count produced by
     `_stack_detail_bands` for every (bands, mode) — otherwise the encoder is
     sized wrong and the run crashes on the first batch.
  2. FOLD mode must be exactly lossless: every detail coefficient is recoverable
     from the stacked blocks. A null result under fold must be attributable to
     the information, not to the stacking (the whole point of the mode).

Legacy INTERP mode is pinned to its prior channel counts so the default path is
provably unchanged.
"""
import math
import os

import numpy as np
import pytest

from lamquant.snn.lma_dataset import (
    _BAND_NATIVE_LEN,
    _stack_detail_bands,
    detail_stack_in_channels,
)

try:
    # lamquant_codec lives in the sibling Lossless checkout under
    # reference_implementations/python_codec/ — the blut conftest only adds the
    # repo root, so resolve the codec package dir here too.
    import sys
    from pathlib import Path
    _CODEC_DIR = (
        Path(__file__).resolve().parents[5]
        / "LamQuant-Lossless" / "reference_implementations" / "python_codec"
    )
    if _CODEC_DIR.is_dir() and str(_CODEC_DIR) not in sys.path:
        sys.path.insert(0, str(_CODEC_DIR))
    from lamquant_codec.ops.lifting import forward_3level_int
    _HAVE_LIFTING = True
except Exception:
    _HAVE_LIFTING = False

C, T = 21, 313
ALL_BANDS = ["l3_detail", "l2_detail", "l1_detail"]


def _make_subs(seed=0):
    rng = np.random.RandomState(seed)
    subs = []
    for _ in range(C):
        r = np.round(rng.randn(2500) * 1000).astype(np.int64)
        s = forward_3level_int(r)
        subs.append({k: np.asarray(v, dtype=np.float64) for k, v in s.items()})
    l3 = rng.randn(C, T).astype(np.float32)
    return l3, subs


@pytest.fixture(autouse=True)
def _clean_env():
    old = {k: os.environ.get(k) for k in ("SNN_DETAIL_BANDS", "SNN_DETAIL_STACK_MODE")}
    yield
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---- pure-config tests (no lifting needed) ----------------------------------

def test_in_channels_interp_legacy_counts():
    assert detail_stack_in_channels([], mode="interp") == 21
    assert detail_stack_in_channels(["l3_detail"], mode="interp") == 42
    assert detail_stack_in_channels(ALL_BANDS, mode="interp") == 84  # legacy


def test_in_channels_fold_counts():
    # fold adds ceil(native_len/313) blocks per band: l3_d=1, l2_d=2, l1_d=4
    assert detail_stack_in_channels([], mode="fold") == 21
    assert detail_stack_in_channels(["l3_detail"], mode="fold") == 42
    assert detail_stack_in_channels(["l2_detail"], mode="fold") == 21 * (1 + 2)
    assert detail_stack_in_channels(["l1_detail"], mode="fold") == 21 * (1 + 4)
    assert detail_stack_in_channels(ALL_BANDS, mode="fold") == 168


def test_band_native_lengths_sum_to_window():
    # l3_approx(313) + the three details must reconstruct the 2500 residual.
    assert 313 + sum(_BAND_NATIVE_LEN.values()) == 2500


# ---- behavioural tests (need the integer lifting primitive) ------------------

@pytest.mark.skipif(not _HAVE_LIFTING, reason="lamquant_codec lifting unavailable")
@pytest.mark.parametrize("mode", ["interp", "fold"])
@pytest.mark.parametrize("bands", [[], ["l3_detail"], ALL_BANDS])
def test_stack_shape_matches_in_channels(mode, bands):
    l3, subs = _make_subs()
    os.environ["SNN_DETAIL_BANDS"] = ",".join(bands)
    os.environ["SNN_DETAIL_STACK_MODE"] = mode
    out = _stack_detail_bands(l3, subs)
    assert out.shape == (detail_stack_in_channels(bands, mode=mode), T)
    assert out.dtype == np.float32


@pytest.mark.skipif(not _HAVE_LIFTING, reason="lamquant_codec lifting unavailable")
@pytest.mark.parametrize("band", ALL_BANDS)
def test_fold_is_lossless(band):
    """Every coefficient of `band` is recoverable from its fold blocks."""
    l3, subs = _make_subs(seed=1)
    os.environ["SNN_DETAIL_BANDS"] = band
    os.environ["SNN_DETAIL_STACK_MODE"] = "fold"
    out = _stack_detail_bands(l3, subs)
    native = _BAND_NATIVE_LEN[band]
    m = math.ceil(native / T)
    assert out.shape[0] == C * (1 + m)
    for c in range(C):
        blocks = [out[C * (1 + g) + c] for g in range(m)]
        recon = np.concatenate(blocks)[:native]
        true = subs[c][band][:native].astype(np.float32)
        assert np.abs(recon - true).max() == 0.0, f"fold lossy for {band} ch{c}"


@pytest.mark.skipif(not _HAVE_LIFTING, reason="lamquant_codec lifting unavailable")
def test_empty_bands_returns_bare_l3_both_modes():
    l3, subs = _make_subs()
    for mode in ("interp", "fold"):
        os.environ["SNN_DETAIL_BANDS"] = ""
        os.environ["SNN_DETAIL_STACK_MODE"] = mode
        out = _stack_detail_bands(l3, subs)
        assert out.shape == (C, T)
        assert np.array_equal(out, l3)
