"""
LamQuant — Subband-Preprocessed Dataset
========================================
Wraps any EEG dataset and applies LPC + lifting DWT preprocessing
inside DataLoader workers, so the GPU never waits for CPU preprocessing.

The preprocessing (LPC order-8 + 3-level lifting DWT) takes ~90ms per
sample on CPU. With 8 workers, 8 samples preprocess in parallel while
the GPU processes the current batch. This eliminates the 77x bottleneck
where the GPU was idle 98.7% of the time.

Usage:
    from subband_dataset import SubbandDataset
    base_dataset = HybridQ31Dataset(files, ...)
    dataset = SubbandDataset(base_dataset)
    loader = DataLoader(dataset, batch_size=32, num_workers=8,
                        pin_memory=True, persistent_workers=True)
    for l3, target_eeg, mask in loader:
        # l3: [B, 21, 313] — ready for TNN, already on CPU
        # target_eeg: [B, 21, 2500] — original signal for loss
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from subband_preprocess import preprocess_subband_single


class SubbandDataset(Dataset):
    """Wraps a base EEG dataset and applies subband preprocessing per-worker.

    Each __getitem__ call:
      1. Gets (eeg [21, 2500], target [21, 2500], mask [2500]) from base
      2. Runs LPC + lifting on CPU (in the DataLoader worker)
      3. Returns (l3 [21, 313], eeg [21, 2500], mask [2500])

    The l3 tensor goes directly to the TNN. The original eeg is kept
    for full-chain loss computation if needed.
    """

    def __init__(self, base_dataset, lpc_order=8, autocorr_len=256):
        self.base = base_dataset
        self.lpc_order = lpc_order
        self.autocorr_len = autocorr_len

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]

        # Handle 4-tuple from Q31Dataset (with precomputed L3)
        if len(item) == 4:
            eeg, target, mask, l3_precomp = item
            if l3_precomp is not None:
                return (l3_precomp, eeg, mask)
        else:
            eeg, target, mask = item[:3]

        # eeg is a torch tensor [21, 2500] — convert to numpy for preprocessing
        eeg_np = eeg.numpy() if isinstance(eeg, torch.Tensor) else eeg

        # LPC + lifting (runs in DataLoader worker process)
        l3, _coeffs, _subs = preprocess_subband_single(
            eeg_np, order=self.lpc_order, autocorr_len=self.autocorr_len)

        return (torch.from_numpy(l3),      # [21, 313] — TNN input (computed)
                eeg,                         # [21, 2500] — original for loss
                mask)                        # [2500] — seizure mask
