"""Raw EEG window dataset for Route B decoder training.

Loads both L3 [21, 313] and raw [21, 2500] windows from Q31 NPZ files.
Stores in float16 to save RAM. The raw signal is the decoder's training
target — the decoder learns to reconstruct full [21, 2500] from latent
[32, 79], bypassing the lifting DWT entirely.

Memory: 50K windows × (21×313 + 21×2500) × 2 bytes = ~5.5 GB float16.
"""

import os
import glob
import gc
import numpy as np
import torch
from torch.utils.data import Dataset


def peek_shape(path, key):
    """Read array shape from NPZ without loading data."""
    try:
        with np.load(path) as d:
            if key in d.files:
                return d[key].shape
    except Exception:
        pass
    return None


class RawWindowDataset(Dataset):
    """Paired L3 + raw EEG windows for Route B decoder training."""

    def __init__(self, file_paths, windows_per_epoch=50000, max_windows=50000):
        if not file_paths:
            raise ValueError("No NPZ files")

        self.windows_per_epoch = windows_per_epoch

        # Pass 1: count windows
        file_windows = []
        total = 0
        for f in file_paths:
            shape = peek_shape(f, 'l3')
            if shape is not None and len(shape) >= 3:
                n = shape[0]
                file_windows.append((f, n))
                total += n

        # Cap
        if max_windows and total > max_windows:
            import random
            random.Random(42).shuffle(file_windows)
            capped = []
            count = 0
            for f, n in file_windows:
                if count + n > max_windows:
                    break
                capped.append((f, n))
                count += n
            file_windows = capped
            total = count

        # Pass 2: load L3 and raw windows
        self.l3_data = torch.empty(total, 21, 313, dtype=torch.float16)
        self.raw_data = torch.empty(total, 21, 2500, dtype=torch.float16)
        idx = 0
        for f, n in file_windows:
            try:
                with np.load(f) as d:
                    l3 = d['l3']  # [N, 21, 313]
                    raw = d['data']  # [21, T] int32
                    spw = raw.shape[1] // n
                    self.l3_data[idx:idx + n] = torch.from_numpy(l3).half()
                    for w in range(n):
                        start = w * spw
                        window = (raw[:, start:start + 2500].astype(np.float32)
                                  / 2147483647.0) * 1000.0
                        self.raw_data[idx + w] = torch.from_numpy(window).half()
                    idx += n
            except Exception:
                continue
        gc.collect()

        if idx < total:
            self.l3_data = self.l3_data[:idx].contiguous()
            self.raw_data = self.raw_data[:idx].contiguous()

        self.n_windows = self.l3_data.shape[0]
        l3_gb = self.l3_data.nelement() * 2 / 1e9
        raw_gb = self.raw_data.nelement() * 2 / 1e9
        print(f"[*] RawWindowDataset: {len(file_windows)} files, "
              f"{self.n_windows:,} windows (L3={l3_gb:.1f} GB + raw={raw_gb:.1f} GB, float16)")

    def __len__(self):
        return self.windows_per_epoch

    def __getitem__(self, idx):
        i = torch.randint(0, self.n_windows, ()).item()
        return self.l3_data[i].float(), self.raw_data[i].float()
