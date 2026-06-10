"""Regression tests for the 2026-06-10 audit fixes in snn_to_nedc_eval.

Pins the window-count + 1 Hz pooling off-by-ones that silently dropped the
trailing samples of a recording (undercounting NEDC sensitivity).
"""

import numpy as np

from snn_to_nedc_eval import _count_windows, _pool_probs_to_1hz


def test_count_windows_covers_trailing_partial():
    # 5100 samples, 2500-wide non-overlapping windows: two full windows leave
    # a 100-sample tail. floor gave 2 (tail dropped); ceil gives 3 so the
    # caller's slide-back clamp scores the tail.
    assert _count_windows(5100, 2500, 2500) == 3
    # Exact multiples are unchanged (no spurious extra window).
    assert _count_windows(5000, 2500, 2500) == 2
    assert _count_windows(2500, 2500, 2500) == 1
    # Shorter-than-window signal still yields one (padded) window.
    assert _count_windows(1000, 2500, 2500) == 1


def test_pool_to_1hz_keeps_every_sample_and_no_empty_bins():
    # 257 steps into 256 one-second bins. A plain len//n block (=1) drops the
    # 257th sample; a ceil block (=2) would empty the tail bins (max() raises).
    probs = np.linspace(0.0, 1.0, 257)
    out = _pool_probs_to_1hz(probs, 256)

    assert out.shape == (256,)
    assert np.isfinite(out).all()
    # The maximum value (the last sample) must survive into the final bin —
    # proof the trailing sample was not dropped.
    assert out[-1] == probs.max()


def test_pool_to_1hz_interpolates_when_fewer_steps_than_seconds():
    probs = np.array([0.0, 1.0])
    out = _pool_probs_to_1hz(probs, 5)
    assert out.shape == (5,)
    assert np.isfinite(out).all()
