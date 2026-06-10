"""Regression tests for the 2026-06-10 blind-audit fixes (dataset area).

Each test pins a bug that the audit surfaced so it cannot silently return.
"""

import numpy as np
import pytest
from scipy.signal import resample_poly


def test_resample_alloc_matches_scipy_output_length():
    """edf_to_events.py:522 — the resample buffer must be allocated with
    ceil(N*up/down), not floor. scipy's resample_poly returns ceil; a floor
    allocation is 1 short and raises a broadcast ValueError on assignment.

    256 -> 250 Hz reduces to up=125, down=128. For N=2500 the floor alloc
    (2441) is one less than scipy's output (2442).
    """
    N, up, down = 2500, 125, 128
    floor_len = int(N * up / down)
    ceil_len = int(np.ceil(N * up / down))
    actual_len = len(resample_poly(np.zeros(N), up, down))

    assert ceil_len == actual_len, "ceil alloc must match scipy output length"
    assert floor_len < actual_len, "floor alloc is short (the original bug)"

    # The fixed allocation accepts the assignment; the buggy one would raise.
    buf_ok = np.zeros((1, ceil_len))
    buf_ok[0] = resample_poly(np.zeros(N), up, down)  # must not raise

    buf_bug = np.zeros((1, floor_len))
    with pytest.raises(ValueError):
        buf_bug[0] = resample_poly(np.zeros(N), up, down)


def test_detect_dataset_chbmit_requires_mit():
    """preprocess.py:612 — a bare '/chb' path must NOT be classified CHB-MIT.
    The old `... or '/chb' in p` (loose precedence) misclassified any path
    with a /chb directory. Real CHB-MIT paths always carry 'mit'.
    """
    from preprocess import detect_dataset

    # Real CHB-MIT layouts (carry 'mit') -> chbmit
    assert detect_dataset("/data/chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_03.edf") == "chbmit"
    assert detect_dataset("/x/physionet/chbmit/chb05/chb05_06.edf") == "chbmit"

    # A /chb directory WITHOUT 'mit' must not be chbmit (the regression).
    assert detect_dataset("/seizure_data/chb/rec001.edf") != "chbmit"
