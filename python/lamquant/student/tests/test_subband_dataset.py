"""Unit tests for ai_models/student/subband_dataset.py — Phase 1 trivial."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from subband_dataset import SubbandDataset

pytestmark = pytest.mark.l2


class _BaseDataset:
    """Synthetic base dataset returning [21, 2500] eeg + mask."""
    def __init__(self, n=4, with_l3=False):
        self.n = n
        self.with_l3 = with_l3

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        eeg = torch.randn(21, 2500)
        target = torch.randn(21, 2500)
        mask = torch.zeros(2500)
        if self.with_l3:
            l3 = torch.randn(21, 313)
            return eeg, target, mask, l3
        return eeg, target, mask


def _fake_preprocess(eeg_np, order, autocorr_len):
    """Stub preprocess_subband_single — returns [21, 313] L3."""
    return np.random.randn(21, 313).astype(np.float32), None, None


class TestSubbandDataset:
    def test_len_passes_through(self):
        ds = SubbandDataset(_BaseDataset(n=7))
        assert len(ds) == 7

    def test_init_stores_params(self):
        ds = SubbandDataset(_BaseDataset(), lpc_order=12, autocorr_len=128)
        assert ds.lpc_order == 12
        assert ds.autocorr_len == 128

    def test_getitem_3tuple_runs_preprocess(self):
        ds = SubbandDataset(_BaseDataset(with_l3=False))
        with patch("subband_dataset.preprocess_subband_single",
                   side_effect=_fake_preprocess):
            l3, eeg, mask = ds[0]
        assert l3.shape == (21, 313)
        assert eeg.shape == (21, 2500)
        assert mask.shape == (2500,)

    def test_getitem_4tuple_with_precomp_skips_preprocess(self):
        ds = SubbandDataset(_BaseDataset(with_l3=True))
        # If precomputed l3 is present, we should NOT call preprocess.
        with patch("subband_dataset.preprocess_subband_single") as mocked:
            l3, eeg, mask = ds[0]
            mocked.assert_not_called()
        assert l3.shape == (21, 313)

    def test_getitem_4tuple_with_none_falls_back(self):
        class _Base4None:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return torch.randn(21, 2500), torch.randn(21, 2500), torch.zeros(2500), None

        ds = SubbandDataset(_Base4None())
        with patch("subband_dataset.preprocess_subband_single",
                   side_effect=_fake_preprocess) as mocked:
            l3, _, _ = ds[0]
            mocked.assert_called_once()
        assert l3.shape == (21, 313)

    def test_getitem_returns_l3_as_torch_tensor(self):
        ds = SubbandDataset(_BaseDataset())
        with patch("subband_dataset.preprocess_subband_single",
                   side_effect=_fake_preprocess):
            l3, _, _ = ds[0]
        assert isinstance(l3, torch.Tensor)
