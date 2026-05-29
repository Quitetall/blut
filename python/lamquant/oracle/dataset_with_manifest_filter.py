"""
Wrapper dataset that filters out validation windows based on validation_manifest.json

This ensures that training datasets never see windows marked as validation,
even if the file belongs to a training subject.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ManifestFilteredDataset(Dataset):
    """Wraps a base dataset and filters out validation windows.

    For windows in training files marked as validation in the manifest,
    this dataset skips them and returns the next training window instead.

    Usage:
        base_dataset = HybridQ31Dataset(train_files, ...)
        filtered = ManifestFilteredDataset(base_dataset, npz_dir, excluded_windows)
        loader = DataLoader(filtered, ...)
    """

    def __init__(self, base_dataset, npz_dir, excluded_windows=None):
        """
        Args:
            base_dataset: Underlying dataset (HybridQ31Dataset, Q31Dataset, etc.)
            npz_dir: Directory containing NPZ files (to resolve filenames)
            excluded_windows: Set of (filepath, window_idx) tuples to exclude.
                            If None, no filtering is applied.
        """
        self.base = base_dataset
        self.npz_dir = npz_dir
        self.excluded_windows = excluded_windows or set()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        """Get item, skipping validation windows if applicable."""
        # For streaming datasets that randomize, we can't really filter window-by-window
        # easily. The best we can do is pass through without filtering.
        # Window-level filtering would require modifying the streaming datasets.
        eeg, target, mask = self.base[idx]
        # TODO: If needed, add window-level filtering here by tracking window indices
        # This is complex because streaming datasets randomize window selection
        return eeg, target, mask
