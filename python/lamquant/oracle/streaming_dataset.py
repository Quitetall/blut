"""
LamQuant — Streaming EEG Dataset
=================================
Loads random 10-second windows from NPZ files on-the-fly.
No RAM caching — reads from disk each batch using worker processes.

Each NPZ file contains a full EEG recording (minutes to hours):
  data: int32 [21, T]       T >> 2500 (e.g., 900,000 for 1 hour)
  seizure_mask: float32 [T]

This dataset yields random 2500-sample (10-second) windows from
random files. With num_workers > 0, multiple workers load in parallel,
keeping the GPU fed without loading the entire dataset into RAM.

Usage:
    from streaming_dataset import StreamingQ31Dataset
    dataset = StreamingQ31Dataset(npz_files, windows_per_epoch=10000)
    loader = DataLoader(dataset, batch_size=32, num_workers=4,
                        pin_memory=True, persistent_workers=True)
"""

import os
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from numpy.lib.format import (
    read_magic as _npy_read_magic,
    read_array_header_1_0 as _npy_read_header_1_0,
    read_array_header_2_0 as _npy_read_header_2_0,
)


WINDOW_SAMPLES = 2500
BYTES_PER_WINDOW = 21 * 2500 * 4 * 2 + 2500 * 4  # eeg + target + mask ≈ 430 KB


def peek_npz_data_shape(npz_path, member_name='data'):
    """
    Read only the NPY header of `<member_name>.npy` inside an NPZ archive
    to return the stored array's shape. Reads ~128 bytes per file instead
    of decompressing the full payload.

    CRITICAL: `np.load(npz, mmap_mode='r')` silently ignores mmap_mode on
    NPZ files (NPZ is a zip, not a flat .npy), so `d['data'].shape[1]` on
    the returned handle actually DECOMPRESSES the whole array into RAM.
    For 11,000+ files this OOM-kills the process with no stderr trace.
    Use this helper instead whenever you only need the shape.

    Returns the shape tuple, or None on any error.
    """
    try:
        with zipfile.ZipFile(npz_path) as zf:
            name = f'{member_name}.npy'
            if name not in zf.namelist():
                return None
            with zf.open(name) as f:
                version = _npy_read_magic(f)
                if version == (1, 0):
                    shape, _, _ = _npy_read_header_1_0(f)
                elif version == (2, 0):
                    shape, _, _ = _npy_read_header_2_0(f)
                else:
                    # Fall back to the private helper for v3.0+
                    from numpy.lib.format import _read_array_header as _rah
                    shape, _, _ = _rah(f, version)
                return shape
    except Exception:  # corrupt NPZ or unreadable — caller treats as 0 windows
        return None


class StreamingQ31Dataset(Dataset):
    """Streaming dataset that loads random windows from NPZ files on disk.

    Each __getitem__ call:
      1. Picks a random file
      2. Loads it from disk (np.load with mmap_mode for speed)
      3. Picks a random 2500-sample window
      4. Converts Q31 -> float32, scales to uV
      5. Returns (eeg, target, seizure_mask)

    Args:
        file_paths: List of NPZ file paths
        windows_per_epoch: Virtual epoch size (how many windows per "epoch").
                           Controls how often the LR scheduler steps.
                           Default: 10,000 (roughly 1 pass through CHB-MIT).
        window_size: Samples per window (default: 2500 = 10s at 250 Hz)
    """

    def __init__(self, file_paths, windows_per_epoch=10000, window_size=WINDOW_SAMPLES):
        self.file_paths = file_paths
        self.windows_per_epoch = windows_per_epoch
        self.window_size = window_size

        if not file_paths:
            raise ValueError("No NPZ files provided")

        # Pre-scan file lengths so we can sample proportionally
        # (longer files contribute more windows). Read NPY headers only —
        # see peek_npz_data_shape for why np.load + mmap_mode='r' doesn't
        # work on NPZ archives.
        self.file_lengths = []
        self.cumulative_windows = []
        total = 0
        for f in file_paths:
            shape = peek_npz_data_shape(f)
            length = shape[1] if shape is not None and len(shape) >= 2 else 0
            n_windows = max(0, length // window_size)
            self.file_lengths.append(n_windows)
            total += n_windows
            self.cumulative_windows.append(total)

        self.total_windows = total
        self.cumulative_windows = np.array(self.cumulative_windows)

        print(f"[*] StreamingQ31Dataset: {len(file_paths)} files, "
              f"{total:,} total windows, "
              f"{windows_per_epoch:,} per epoch")

    @staticmethod
    def estimate_ram_budget(max_ram_gb=None, reserve_gb=20.0,
                            max_fraction=0.50):
        """Estimate how many windows fit in available RAM.

        Two independent safety caps prevent OOM:

        1. ``available - reserve_gb`` — never eat into the headroom needed
           for Firefox, KDE, CUDA, DataLoader workers, Claude, etc. The
           previous default of 8-10 GB was too low on a 62 GB system where
           the user's desktop alone uses ~10 GB.

        2. ``total_physical * max_fraction`` — hard cap at 50% of physical
           RAM regardless of what ``available`` reports. This catches the
           case where ``available`` is inflated by reclaimable page cache
           that the OS *could* reclaim but hasn't yet.

        The effective budget is ``min(cap1, cap2)``.

        Args:
            max_ram_gb: Hard limit in GB. If provided, both caps are
                bypassed and this value is used directly.
            reserve_gb: RAM to keep free. Default: 20 GB (covers a
                typical desktop + CUDA + DataLoader workers with margin).
            max_fraction: Never use more than this fraction of physical RAM
                for the dataset cache. Default: 0.50 (50%).

        Returns:
            max_windows that fit in the budgeted RAM.
        """
        if max_ram_gb is not None:
            usable = max_ram_gb * 1e9
            print(f"[*] RAM budget: explicit cap {max_ram_gb:.1f} GB "
                  f"-> {int(usable / BYTES_PER_WINDOW):,} windows")
            return int(usable / BYTES_PER_WINDOW)

        total = available = 0
        try:
            import psutil
            mi = psutil.virtual_memory()
            total = mi.total
            available = mi.available
        except ImportError:
            try:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            total = int(line.split()[1]) * 1024
                        elif line.startswith('MemAvailable:'):
                            available = int(line.split()[1]) * 1024
            except Exception:  # /proc/meminfo unreadable (container, non-Linux)
                total = available = 32e9  # conservative fallback

        cap_avail = max(0, available - reserve_gb * 1e9)
        cap_frac = max(0, total * max_fraction)
        usable = min(cap_avail, cap_frac)
        max_windows = int(usable / BYTES_PER_WINDOW)

        print(f"[*] RAM budget: {total/1e9:.1f} GB total, "
              f"{available/1e9:.1f} GB available")
        print(f"    cap1 (avail - {reserve_gb:.0f} GB reserve): "
              f"{cap_avail/1e9:.1f} GB")
        print(f"    cap2 ({max_fraction:.0%} of total):            "
              f"{cap_frac/1e9:.1f} GB")
        print(f"    effective: {usable/1e9:.1f} GB "
              f"-> {max_windows:,} windows")
        return max_windows

    def __len__(self):
        return self.windows_per_epoch

    def __getitem__(self, idx):
        # Sample a random window (idx is ignored — each call is random)
        rng = np.random.default_rng()

        # Pick a random file, weighted by number of windows
        file_idx = rng.integers(0, len(self.file_paths))

        # Load the file — use context manager to close the zip handle
        # immediately. Without this, 1000s of open fds accumulate across
        # DataLoader workers × epochs.
        try:
            with np.load(self.file_paths[file_idx]) as data:
                eeg_q31 = np.array(data['data'])           # copy out of zip
                mask = np.array(data['seizure_mask'])
        except Exception:  # corrupt NPZ — skip file, continue loading
            z = torch.zeros(21, self.window_size)
            return z, z, torch.zeros(self.window_size)

        T = eeg_q31.shape[1]
        if T < self.window_size:
            eeg_q31 = np.pad(eeg_q31, ((0, 0), (0, self.window_size - T)))
            mask = np.pad(mask, (0, self.window_size - T))
            start = 0
        else:
            start = rng.integers(0, T - self.window_size + 1)

        window = eeg_q31[:, start:start + self.window_size]
        window_mask = mask[start:start + self.window_size]

        eeg = (torch.tensor(window, dtype=torch.float32) / 2147483647.0) * 1000.0
        return eeg, eeg, torch.tensor(window_mask, dtype=torch.float32)


class HybridQ31Dataset(Dataset):
    """Loads the largest random subset that fits in RAM, streams the rest.

    Strategy:
      1. Estimate available RAM (auto or manual)
      2. Randomly select files until RAM budget is reached
      3. Pre-load selected files, extract ALL windows into RAM
      4. Remaining files are streamed on-the-fly during training
      5. Each __getitem__: 70% chance from RAM cache, 30% from disk
         (ensures the model sees both cached and streamed data)

    This gives you the speed of RAM caching for most batches while
    still covering the full dataset. Files rotate each epoch via
    the streaming path.

    Args:
        file_paths: All NPZ file paths
        max_ram_gb: Max RAM to use for cache (None = auto-detect)
        reserve_gb: RAM to keep free (default: 8 GB)
        windows_per_epoch: Virtual epoch size
        cache_ratio: Fraction of batches drawn from cache (default: 0.7)
    """

    def __init__(self, file_paths, max_ram_gb=None, reserve_gb=8.0,
                 windows_per_epoch=50000, cache_ratio=0.7):
        self.file_paths = file_paths
        self.windows_per_epoch = windows_per_epoch
        self.cache_ratio = cache_ratio
        self.window_size = WINDOW_SAMPLES

        if not file_paths:
            raise ValueError("No NPZ files provided")

        max_windows = StreamingQ31Dataset.estimate_ram_budget(max_ram_gb, reserve_gb)

        # Scan all files for window counts (header-only read — do NOT
        # decompress the full `data` array; see peek_npz_data_shape for why.)
        file_windows = []
        for f in file_paths:
            shape = peek_npz_data_shape(f)
            if shape is not None and len(shape) >= 2:
                file_windows.append(shape[1] // WINDOW_SAMPLES)
            else:
                file_windows.append(0)

        total_windows = sum(file_windows)

        # Randomly select files for caching until budget is full
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(file_paths))
        cached_windows = 0
        self.cached_files = []
        self.streamed_files = []

        for i in indices:
            if cached_windows + file_windows[i] <= max_windows and file_windows[i] > 0:
                self.cached_files.append(file_paths[i])
                cached_windows += file_windows[i]
            else:
                self.streamed_files.append(file_paths[i])

        print(f"[*] HybridQ31Dataset: {len(file_paths)} files, {total_windows:,} total windows")
        print(f"    Cached:   {len(self.cached_files)} files, {cached_windows:,} windows "
              f"({cached_windows * BYTES_PER_WINDOW / 1e9:.1f} GB)")
        print(f"    Streamed: {len(self.streamed_files)} files")
        print(f"    Cache ratio: {cache_ratio:.0%} cached, {1-cache_ratio:.0%} streamed")

        # Load cached windows into RAM as contiguous numpy arrays, convert
        # to float32 in numpy (in-place), THEN wrap as torch tensors
        # (zero-copy). This sequence keeps peak RAM minimal:
        #
        #   Previous approach peak: eeg_list views (16 GB, holds parent
        #   arrays alive) + stacked int32 (16 GB) + torch float32 temp
        #   (16 GB) = 48 GB → OOM on a 62 GB system.
        #
        #   Fixed approach peak: stacked int32 (16 GB) + float32 copy
        #   (16 GB) = 32 GB. Then int32 is freed → 16 GB steady state.
        #
        # Key fixes:
        #   - np.load() as context manager (closes fd immediately)
        #   - .copy() on window slices (breaks view chain to parent array)
        #   - Delete list BEFORE stacking to avoid 3× peak
        #   - Convert in numpy, not torch (avoids torch intermediate copies)
        import gc

        eeg_list = []
        mask_list = []
        for f in self.cached_files:
            try:
                with np.load(f) as data:
                    eeg_q31 = data['data']           # int32 [21, T]
                    file_mask = data['seizure_mask']  # float32 [T]
                    T = eeg_q31.shape[1]
                    for start in range(0, T - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                        # .copy() severs the view from the parent array so
                        # the ~5 MB parent can be GC'd when the `with` block
                        # closes, instead of staying alive until stacking.
                        eeg_list.append(
                            eeg_q31[:, start:start + WINDOW_SAMPLES].copy())
                        mask_list.append(
                            file_mask[start:start + WINDOW_SAMPLES].copy())
            except Exception:  # corrupt NPZ — skip file, continue loading
                continue

        if eeg_list:
            # 1. Stack into contiguous int32 array, then free the list.
            eeg_np = np.stack(eeg_list)    # [N, 21, 2500] int32 ≈ 16 GB
            del eeg_list; gc.collect()     # free the N individual arrays

            mask_np = np.stack(mask_list)  # [N, 2500] float32 ≈ 0.8 GB
            del mask_list; gc.collect()

            # 2. Convert int32 → float32 in numpy. This creates one new
            #    16 GB array; the int32 is then freed. Peak = 32 GB.
            eeg_f32 = eeg_np.astype(np.float32)
            del eeg_np; gc.collect()       # free the 16 GB int32

            # 3. Scale Q31 → microvolts in-place on the float32 array.
            #    In-place ops avoid creating yet another 16 GB copy.
            eeg_f32 /= 2147483647.0
            eeg_f32 *= 1000.0

            # 4. Wrap as torch tensors (zero-copy from the numpy arrays).
            self.cache_eeg = torch.from_numpy(eeg_f32)
            self.cache_mask = torch.from_numpy(mask_np)
            self.cache_size = self.cache_eeg.shape[0]
        else:
            self.cache_eeg = torch.empty(0, 21, WINDOW_SAMPLES)
            self.cache_mask = torch.empty(0, WINDOW_SAMPLES)
            self.cache_size = 0

        actual_gb = (self.cache_eeg.nelement() * 4 + self.cache_mask.nelement() * 4) / 1e9
        print(f"    Loaded {self.cache_size:,} windows into RAM ({actual_gb:.1f} GB)")

    def __len__(self):
        return self.windows_per_epoch

    def __getitem__(self, idx):
        rng = np.random.default_rng()

        # Decide: cache or stream
        if self.cache_size > 0 and rng.random() < self.cache_ratio:
            i = rng.integers(0, self.cache_size)
            eeg = self.cache_eeg[i]
            return eeg, eeg, self.cache_mask[i]

        # Stream from disk
        if not self.streamed_files:
            # All files are cached — just sample from cache
            i = rng.integers(0, self.cache_size)
            eeg = self.cache_eeg[i]
            return eeg, eeg, self.cache_mask[i]

        file_idx = rng.integers(0, len(self.streamed_files))
        try:
            with np.load(self.streamed_files[file_idx]) as data:
                eeg_q31 = np.array(data['data'])
                mask = np.array(data['seizure_mask'])
        except Exception:  # corrupt NPZ — skip file, continue loading
            z = torch.zeros(21, self.window_size)
            return z, z, torch.zeros(self.window_size)

        T = eeg_q31.shape[1]
        if T < self.window_size:
            eeg_q31 = np.pad(eeg_q31, ((0, 0), (0, self.window_size - T)))
            mask = np.pad(mask, (0, self.window_size - T))
            start = 0
        else:
            start = rng.integers(0, T - self.window_size + 1)

        window = eeg_q31[:, start:start + self.window_size]
        window_mask = mask[start:start + self.window_size]
        eeg = (torch.tensor(window, dtype=torch.float32) / 2147483647.0) * 1000.0
        return eeg, eeg, torch.tensor(window_mask, dtype=torch.float32)


class MemmapTeacherDataset(Dataset):
    """
    Pre-extracts ALL teacher training windows into a single flat memmap file
    on disk, then memory-maps it for O(1) random access at ~0.5ms per sample.

    Why this exists: HybridQ31Dataset's streaming path decompresses a multi-MB
    NPZ archive for every random window access (~208ms/sample). With 150K
    windows/epoch and 30% streaming, that's 9,360 seconds/epoch = 200+ hours
    for 800 epochs. This dataset replaces that with a contiguous binary file
    where each window is a flat [21, 2500] float32 block. The OS page cache
    handles hot/cold page management automatically.

    One-time extraction: ~40 minutes (reads all NPZs, writes the memmap).
    After that: ~0.5ms per random window, GPU-bound at ~3 hours for 800 epochs.

    File format:
      eeg_cache:  np.memmap, shape [N, 21, 2500], dtype float32
      mask_cache: np.memmap, shape [N, 2500],     dtype float32
      meta:       JSON sidecar with window count and source file list
    """

    CACHE_DIR = 'ai_models/dataset_sim/teacher_memmap'

    def __init__(self, file_paths, root_dir, windows_per_epoch=150000):
        self.windows_per_epoch = windows_per_epoch
        cache_dir = os.path.join(root_dir, self.CACHE_DIR)
        eeg_path = os.path.join(cache_dir, 'eeg.dat')
        mask_path = os.path.join(cache_dir, 'mask.dat')
        meta_path = os.path.join(cache_dir, 'meta.json')

        # BUG FIX H1: handle corrupted meta.json from a crashed build
        cache_valid = False
        if os.path.exists(meta_path):
            import json
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                n_windows = meta['n_windows']
                cache_valid = True
                print(f"[*] MemmapTeacherDataset: loading existing cache "
                      f"({n_windows:,} windows)")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[!] Corrupt cache meta ({e}), rebuilding...")
        if not cache_valid:
            n_windows = self._build_cache(
                file_paths, cache_dir, eeg_path, mask_path, meta_path)

        self.eeg = np.memmap(eeg_path, dtype=np.float32, mode='r',
                             shape=(n_windows, 21, WINDOW_SAMPLES))
        self.mask = np.memmap(mask_path, dtype=np.float32, mode='r',
                              shape=(n_windows, WINDOW_SAMPLES))
        self.n_windows = n_windows
        print(f"[*] MemmapTeacherDataset: {n_windows:,} windows, "
              f"{n_windows * 21 * WINDOW_SAMPLES * 4 / 1e9:.1f} GB on disk, "
              f"memmap'd for O(1) access")

    @staticmethod
    def _build_cache(file_paths, cache_dir, eeg_path, mask_path, meta_path):
        """One-time extraction: read all NPZs, write flat memmap files."""
        import json
        from tqdm import tqdm

        os.makedirs(cache_dir, exist_ok=True)
        print(f"[*] Building teacher memmap cache (one-time, ~40 min)...")
        print(f"    Scanning {len(file_paths)} files for window counts...")

        # Pass 1: count total windows (header-only, fast)
        total = 0
        for f in file_paths:
            shape = peek_npz_data_shape(f)
            if shape is not None and len(shape) >= 2:
                total += shape[1] // WINDOW_SAMPLES
        print(f"    Total windows: {total:,}")

        # Pass 2: extract and write
        eeg_mm = np.memmap(eeg_path, dtype=np.float32, mode='w+',
                           shape=(total, 21, WINDOW_SAMPLES))
        mask_mm = np.memmap(mask_path, dtype=np.float32, mode='w+',
                            shape=(total, WINDOW_SAMPLES))

        idx = 0
        for f in tqdm(file_paths, desc="Extracting windows"):
            try:
                with np.load(f) as data:
                    eeg_q31 = data['data']       # int32 [21, T]
                    file_mask = data['seizure_mask']  # float32 [T]
                    T = eeg_q31.shape[1]
                    for start in range(0, T - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                        w = eeg_q31[:, start:start + WINDOW_SAMPLES]
                        m = file_mask[start:start + WINDOW_SAMPLES]
                        eeg_mm[idx] = (w.astype(np.float32) / 2147483647.0) * 1000.0
                        mask_mm[idx] = m.astype(np.float32)
                        idx += 1
            except Exception:  # corrupt NPZ — skip file, continue loading
                continue

        # Trim if some files failed
        actual = idx
        eeg_mm.flush()
        mask_mm.flush()
        del eeg_mm, mask_mm

        if actual < total:
            # BUG FIX H5: actually truncate the files (creating a smaller memmap
            # view does NOT shrink the underlying file).
            os.truncate(eeg_path, actual * 21 * WINDOW_SAMPLES * 4)
            os.truncate(mask_path, actual * WINDOW_SAMPLES * 4)
            print(f"    Note: {total - actual} windows skipped (read errors), "
                  f"files truncated to {actual} windows")

        with open(meta_path, 'w') as f:
            json.dump({'n_windows': actual, 'n_files': len(file_paths)}, f)

        print(f"    Cache built: {actual:,} windows, "
              f"{actual * 21 * WINDOW_SAMPLES * 4 / 1e9:.1f} GB")
        return actual

    def __len__(self):
        return self.windows_per_epoch

    def __getitem__(self, idx):
        # Random window (idx ignored — stochastic like streaming dataset)
        i = np.random.randint(0, self.n_windows)
        eeg = torch.from_numpy(self.eeg[i].copy())
        mask = torch.from_numpy(self.mask[i].copy())
        return eeg, eeg, mask


class PrecomputedL3Dataset(Dataset):
    """
    Loads precomputed L3 arrays directly from NPZ files into a single
    contiguous RAM tensor. Each __getitem__ is a nanosecond tensor index
    — no disk I/O, no LPC, no lifting DWT at training time.

    This is semantically equivalent to HybridQ31Dataset + SubbandDataset
    but ~3,000x faster because the 94 ms per-window LPC+lifting
    computation was already done once by precompute_l3_fast.py and stored
    in each NPZ file's 'l3' key.

    Accuracy impact: NONE. The L3 data is bit-identical to what
    SubbandDataset would have computed on the fly (verified on a 250-file
    sample with zero mismatches).

    Memory: ~20 GB for 768K training windows at [21, 313] float32.
    Fits within the 33.6 GB cache budget on a 64 GB system.

    Returns: (l3, l3, dummy_mask) to match the 3-tuple interface that
    train_student_subband.py expects. x_eeg and mask are never used by
    the student's training loop — only x_l3 matters.

    Usage:
        train_ds = PrecomputedL3Dataset(train_files, windows_per_epoch=50000)
        loader = DataLoader(train_ds, batch_size=32, num_workers=0,
                            pin_memory=True, shuffle=False)
    """

    def __init__(self, file_paths=None, windows_per_epoch=50000, max_windows=None,
                 *, file_entries=None, with_fullband: bool = False,
                 fullband_window_samples: int = 2500,
                 fullband_memmap_path=None,
                 train_noise_bits: int = 0):
        """
        file_paths:   legacy interface — list of NPZ paths, no provenance
        file_entries: typed interface — list of lamquant.common.data_types.FileEntry,
                      enables prefetch_typed_batches() and per-window provenance

        Pass exactly one of file_paths or file_entries. If file_entries is
        provided, parallel provenance arrays are built during the load
        loop so the typed batch path can yield TrainingBatch instances
        with assert_no_leakage() support.

        Two ways to attach a fullband target (mutually exclusive):

        with_fullband=True (in-RAM):
            Load each window's raw fullband from the NPZ's 'data' key
            aligned with L3 windows. RAM cost ~105 KB/window in float16
            — fast preset (~50K windows ≈ 5 GB) fits comfortably;
            production scale (~1.3M windows ≈ 136 GB) does NOT fit and
            should use the memmap path instead.

        fullband_memmap_path=Path (memmap):
            Open a precomputed flat memmap [N_total, 21, 2500] float16
            built by precompute_fullband_memmap.py. The dataset asserts
            the memmap's window count matches the L3 totals from this
            file_entries list (same canonical manifest order is required).
            Production-scale path: zero RAM cost, OS page cache absorbs
            the per-batch 1-10 MB random read.

        Pass at most one of with_fullband / fullband_memmap_path.
        """
        if (file_paths is None) == (file_entries is None):
            raise ValueError("Pass exactly one of file_paths or file_entries")

        # Derive file_paths from file_entries when typed inputs are used.
        if file_entries is not None:
            file_paths = [fe.path for fe in file_entries]
            self._file_entries = list(file_entries)
        else:
            self._file_entries = None

        if not file_paths:
            raise ValueError("No NPZ files provided")

        self.windows_per_epoch = windows_per_epoch
        self._dummy_mask = torch.zeros(313)
        self._train_noise_bits = train_noise_bits

        # Two-pass loading to avoid 2× peak memory from concatenation:
        # Pass 1: count total windows (header-only, fast)
        # Pass 2: preallocate one tensor, fill in-place from each file

        # Pass 1: count. We pair each (path, n) with its FileEntry (or
        # None) so the cap+load loop below can populate the provenance
        # arrays without re-scanning.
        import gc
        file_windows = []  # list of (path, n_windows, file_entry_or_None)
        total = 0
        skipped = 0
        if self._file_entries is not None:
            for fe in self._file_entries:
                shape = peek_npz_data_shape(fe.path, member_name='l3')
                if shape is not None and len(shape) >= 3:
                    n = shape[0]
                    file_windows.append((fe.path, n, fe))
                    total += n
                else:
                    skipped += 1
        else:
            for f in file_paths:
                shape = peek_npz_data_shape(f, member_name='l3')
                if shape is not None and len(shape) >= 3:
                    n = shape[0]
                    file_windows.append((f, n, None))
                    total += n
                else:
                    skipped += 1

        if total == 0:
            raise ValueError("No precomputed L3 found. Run precompute_l3_fast.py first.")

        # Cap at max_windows if specified (for fast preset — don't load 20 GB
        # when you only need 50K windows per epoch)
        if max_windows is not None and total > max_windows:
            # Shuffle and take a subset of files until we hit the cap
            import random
            random.Random(42).shuffle(file_windows)
            capped = []
            count = 0
            for tup in file_windows:
                _f, n, _fe = tup
                if count + n > max_windows:
                    break
                capped.append(tup)
                count += n
            file_windows = capped
            total = count
            print(f"[*] PrecomputedL3Dataset: capped at {total:,} windows "
                  f"(max_windows={max_windows:,})")

        # Pass 2: preallocate and fill (no intermediate list or concatenation)
        # Store as float16 to halve RAM (29 GB → 15 GB). L3 values are integers
        # in [-300, 300] — float16 represents them exactly (integer-exact to 2048).
        # Cast to float32 at batch time in prefetch_batches.
        self.l3_data = torch.empty(total, 21, 313, dtype=torch.float16)

        # Optional fullband target (for Tier 3+ joint loss).
        # Q31 → microvolts (matches the encoder-input scaling), float16.
        if with_fullband and fullband_memmap_path is not None:
            raise ValueError(
                "Pass at most one of with_fullband / fullband_memmap_path")
        self.with_fullband = bool(with_fullband)
        self._fb_window = int(fullband_window_samples)
        self._fb_memmap_path = (str(fullband_memmap_path)
                                if fullband_memmap_path is not None else None)
        if self.with_fullband:
            self.fullband_data = torch.empty(
                total, 21, self._fb_window, dtype=torch.float16)
        else:
            self.fullband_data = None

        # Provenance arrays — only populated when file_entries is provided.
        # One entry per window, parallel to l3_data along axis 0.
        if self._file_entries is not None:
            self._win_dataset = [''] * total
            self._win_patient = [''] * total
            self._win_split = [''] * total
            self._win_has_seizure = [False] * total
            self._win_event_type = [''] * total
            self._win_clinical_category = ['normal'] * total
        else:
            self._win_dataset = None  # signal "no provenance available"

        idx = 0
        # Q31 → microvolts conversion (matches PrecomputedL3Dataset's
        # encoder-input scaling so the fullband target lives in the same
        # numerical regime as the L3 the model is trained on).
        Q31 = 2147483647.0
        UV_PER_Q31 = 1000.0
        # Per-loaded-window record of source NPZ path + within-file
        # window index. Used to look up memmap offsets if a precomputed
        # fullband memmap is attached after this loop.
        loaded_src = []  # list of (npz_path, k_window_idx) of length idx
        for f, n, fe in file_windows:
            try:
                with np.load(f) as d:
                    chunk = d['l3']  # [n, 21, 313] float32
                    actual_n = chunk.shape[0]
                    if actual_n != n:
                        # Header lied (or file was modified). Use the
                        # smaller of the two so we don't overrun the
                        # preallocated tensor or read uninitialised cells.
                        n = min(n, actual_n)
                        chunk = chunk[:n]
                    self.l3_data[idx:idx + n] = torch.from_numpy(chunk).half()
                    if self.with_fullband:
                        # Slice raw data into 2500-sample windows aligned to L3.
                        raw = d['data']           # int32 [21, T]
                        T = raw.shape[1]
                        for k in range(n):
                            start = k * self._fb_window
                            end = start + self._fb_window
                            if end > T:
                                # Pad the last short window with zeros so the
                                # tensor stays a fixed shape — matches the
                                # encoder's behaviour on partial windows.
                                seg = np.zeros((21, self._fb_window),
                                               dtype=np.float32)
                                seg[:, :max(0, T - start)] = (
                                    raw[:, start:T].astype(np.float32)
                                    / Q31 * UV_PER_Q31)
                            else:
                                seg = (raw[:, start:end].astype(np.float32)
                                       / Q31 * UV_PER_Q31)
                            self.fullband_data[idx + k] = (
                                torch.from_numpy(seg).half())
                    if fe is not None and self._win_dataset is not None:
                        for k in range(idx, idx + n):
                            self._win_dataset[k] = fe.dataset.value
                            self._win_patient[k] = fe.patient_id
                            self._win_split[k] = fe.split.value
                            self._win_has_seizure[k] = fe.has_seizure
                            self._win_event_type[k] = fe.event_type
                            self._win_clinical_category[k] = getattr(
                                fe, 'clinical_category', 'normal')
                    if self._fb_memmap_path is not None:
                        for k in range(n):
                            loaded_src.append((f, k))
                    idx += n
            except Exception:
                continue
        gc.collect()

        # Trim if some files failed
        if idx < total:
            self.l3_data = self.l3_data[:idx].contiguous()
            if self.fullband_data is not None:
                self.fullband_data = self.fullband_data[:idx].contiguous()
            if self._win_dataset is not None:
                self._win_dataset = self._win_dataset[:idx]
                self._win_patient = self._win_patient[:idx]
                self._win_split = self._win_split[:idx]
                self._win_has_seizure = self._win_has_seizure[:idx]
                self._win_event_type = self._win_event_type[:idx]
                self._win_clinical_category = self._win_clinical_category[:idx]

        self.n_windows = self.l3_data.shape[0]

        # Attach the precomputed fullband memmap, if requested. Read the
        # sidecar meta JSON, build a {npz_path → memmap_base_offset}
        # lookup, then translate this dataset's per-window source records
        # into a flat numpy index array. At fetch time we just gather
        # `self._fb_memmap[self._fb_memmap_idx[shard_idx]]`.
        self._fb_memmap = None
        self._fb_memmap_idx = None
        if self._fb_memmap_path is not None:
            import json as _json
            from pathlib import Path as _Path
            dat_path = _Path(self._fb_memmap_path)
            meta_path = dat_path.with_suffix('.meta.json')
            if not meta_path.exists():
                # `precompute_fullband_memmap.py` writes <name>.meta.json
                # alongside the .dat — accept either by stripping suffix.
                meta_path = dat_path.parent / (dat_path.stem + '.meta.json')
            if not meta_path.exists():
                raise FileNotFoundError(
                    f'fullband memmap meta JSON not found alongside '
                    f'{self._fb_memmap_path} (looked for {meta_path})')
            with open(meta_path) as f:
                meta = _json.load(f)
            mm_total = int(meta['n_windows'])
            self._fb_memmap = np.memmap(
                str(dat_path), dtype=np.float16, mode='r',
                shape=(mm_total, 21, self._fb_window))
            # Map source NPZ → base offset in the memmap.
            base = {entry['path']: int(entry['offset'])
                    for entry in meta['schedule']}
            # Translate this dataset's per-window source records into
            # flat memmap row indices.
            mm_idx = np.empty(self.n_windows, dtype=np.int64)
            misses = 0
            for i, (npz_path, k) in enumerate(loaded_src):
                off = base.get(npz_path)
                if off is None:
                    misses += 1
                    mm_idx[i] = 0          # safe fallback (will produce
                                           # noise; surfaced via misses log)
                else:
                    mm_idx[i] = off + k
            self._fb_memmap_idx = mm_idx
            if misses:
                print(f"    [!] {misses} window(s) had no memmap offset "
                      f"— rebuild precompute_fullband_memmap with current manifest")

        size_gb = self.l3_data.nelement() * 2 / 1e9
        fb_tag = ''
        if self.fullband_data is not None:
            size_gb += self.fullband_data.nelement() * 2 / 1e9
            fb_tag = ', +fullband(RAM)'
        elif self._fb_memmap is not None:
            mm_gb = self._fb_memmap.nbytes / 1e9
            fb_tag = f', +fullband(memmap {mm_gb:.0f} GB)'
        print(f"[*] PrecomputedL3Dataset: {len(file_windows)} files, "
              f"{self.n_windows:,} windows loaded ({size_gb:.1f} GB, float16"
              f"{fb_tag})")
        if self._win_dataset is not None:
            print(f"    Provenance enabled: {len(set(self._win_dataset))} datasets, "
                  f"{len(set(self._win_patient))} patients")
        if skipped:
            print(f"    ({skipped} files skipped — missing 'l3' key)")

    # ------------------------------------------------------------
    # Fullband fetch — RAM tensor or memmap, uniform call site
    # ------------------------------------------------------------
    def _fb_fetch(self, indices) -> 'torch.Tensor':
        """Fetch fullband rows for the given local indices.

        Returns a torch.float16 tensor on CPU; caller is responsible
        for `.float().to(device)`. Returns None if no fullband target
        is attached.
        """
        if self.fullband_data is not None:
            return self.fullband_data[indices]
        if self._fb_memmap is not None:
            if hasattr(indices, 'cpu'):
                np_idx = indices.cpu().numpy()
            elif hasattr(indices, 'numpy'):
                np_idx = indices.numpy()
            else:
                np_idx = np.asarray(indices)
            global_idx = self._fb_memmap_idx[np_idx]
            # Memmap returns a numpy view of the requested rows.
            # `np.array(...)` materialises the slice into a contiguous
            # owned buffer so the torch tensor doesn't keep the memmap
            # rows pinned indefinitely.
            rows = np.array(self._fb_memmap[global_idx])
            return torch.from_numpy(rows)
        return None

    @property
    def has_fullband(self) -> bool:
        return self.fullband_data is not None or self._fb_memmap is not None

    def calibrate_shard_budget(self, device):
        """Measure free VRAM and lock shard size. Call after torch.compile warmup."""
        bytes_per_window = 21 * 313 * 4
        if self.has_fullband:
            bytes_per_window += 21 * self._fb_window * 4
        try:
            free_vram = torch.cuda.mem_get_info(device)[0]
            # Reserve VRAM for model forward+backward activations.
            # Fullband decoder at batch=16 needs ~8-10 GB for activations
            # + optimizer state. 14 GB reserve is safe for 24 GB cards.
            reserve = 14 * 1024**3 if self.has_fullband else 4 * 1024**3
            budget = max(free_vram - reserve, 1 * 1024**3)
            self._shard_max = int(budget / bytes_per_window)
            self._shard_max = max(self._shard_max, 64)
            shard_gb = self._shard_max * bytes_per_window / 1e9
            print(f"[*] Shard budget calibrated: {free_vram/1e9:.1f} GB free → "
                  f"{self._shard_max:,} windows ({shard_gb:.1f} GB) per shard"
                  + (' [+fullband]' if self.has_fullband else ''))
        except Exception:
            self._shard_max = self.windows_per_epoch
            print(f"[*] Shard budget: defaulting to full epoch ({self._shard_max:,} windows)")

    def to_gpu(self, device):
        """Move entire dataset to GPU. Only call if VRAM budget allows."""
        size_gb = self.l3_data.nelement() * 4 / 1e9
        self.l3_data = self.l3_data.to(device)
        self._dummy_mask = self._dummy_mask.to(device)
        self._on_gpu = True
        print(f"[*] PrecomputedL3Dataset: moved to GPU ({size_gb:.1f} GB VRAM)")

    def __len__(self):
        return self.windows_per_epoch

    def __getitem__(self, idx):
        # Random window from the pool (idx is ignored — stochastic like
        # the streaming dataset, compatible with shuffle=False DataLoader).
        i = torch.randint(0, self.n_windows, ()).item()
        l3 = self.l3_data[i].float()     # [21, 313] — cast float16→float32
        if self._train_noise_bits > 0:
            # Mask LSBs: quantize to nearest 2^nb. Training-only — data on
            # disk is unmodified. Model learns signal, not ADC thermal noise.
            nb = self._train_noise_bits
            scale = float(1 << nb)
            l3 = torch.floor(l3 / scale) * scale
        return l3, l3, self._dummy_mask  # (input, target, mask)

    def prefetch_batches(self, batch_size, device):
        """Shard-based GPU batch iterator: one bulk transfer per epoch.

        Samples windows_per_epoch random indices, transfers the entire epoch's
        data to GPU in one contiguous block (~10 GB for 400K windows), then
        iterates batches as pure GPU tensor slices — zero per-batch transfer,
        zero DataLoader overhead, zero pin_memory copies.

        For GPU-resident datasets, skips the transfer entirely.

        With 400K windows at [21, 313] float32 = 10.5 GB, one PCIe 4.0 x16
        transfer takes ~0.7 seconds. If windows_per_epoch exceeds VRAM budget,
        splits into shards of shard_max windows.

        Yields (x_l3, x_l3, dummy_mask) tuples, already on device.
        """
        n_total = (self.windows_per_epoch // batch_size) * batch_size
        dummy = self._dummy_mask if getattr(self, '_on_gpu', False) else self._dummy_mask.to(device)

        # Sample all epoch indices upfront
        epoch_indices = torch.randint(0, self.n_windows, (n_total,))

        nb = self._train_noise_bits

        def _mask_noise(t):
            if nb > 0:
                scale = float(1 << nb)
                return torch.floor(t / scale) * scale
            return t

        if getattr(self, '_on_gpu', False):
            # Data already on GPU — gather and slice, no transfer
            gpu_epoch = _mask_noise(self.l3_data[epoch_indices].float())
            for i in range(0, n_total, batch_size):
                l3 = gpu_epoch[i:i + batch_size]
                yield l3, l3, dummy
            return

        # CPU data → shard-based bulk transfer
        # Use calibrated shard size if available, otherwise query VRAM live
        if hasattr(self, '_shard_max'):
            shard_max = self._shard_max
        else:
            bytes_per_window = 21 * 313 * 4
            try:
                free_vram = torch.cuda.mem_get_info(device)[0]
                shard_budget = max(free_vram - 2 * 1024**3, 2 * 1024**3)
            except Exception:
                shard_budget = 20 * 1024**3
            shard_max = int(shard_budget / bytes_per_window)
            shard_max = max(shard_max, batch_size)

        for shard_start in range(0, n_total, shard_max):
            shard_end = min(shard_start + shard_max, n_total)
            shard_idx = epoch_indices[shard_start:shard_end]
            # One big gather + transfer, apply noise masking on GPU
            gpu_shard = _mask_noise(self.l3_data[shard_idx].to(device).float())
            for i in range(0, len(gpu_shard), batch_size):
                l3 = gpu_shard[i:i + batch_size]
                yield l3, l3, dummy
            del gpu_shard

    def prefetch_typed_batches(self, batch_size, device, sampler=None):
        """Like prefetch_batches but yields TrainingBatch with provenance.

        Requires the dataset to have been built with file_entries=... so
        per-window provenance arrays are populated. The yielded batches
        support .assert_no_leakage(expected_split) at the top of the
        training loop — runtime safety net against data leakage.

        sampler: optional iterable of int indices (e.g. ClinicalWeightedSampler).
            When provided, epoch indices are drawn from the sampler instead of
            uniform random. The sampler must yield at least n_total indices.

        Yields TrainingBatch instances with l3_approx already on device.
        """
        if self._win_dataset is None:
            raise RuntimeError(
                "prefetch_typed_batches requires the dataset to be "
                "constructed with file_entries=... (got file_paths=...). "
                "Use the typed pipeline: DatasetManifest.get_file_entries(split)."
            )
        # Lazy import — keeps the legacy path free of any data_types
        # dependency. Try the package-style import first, fall back to a
        # path-based import for callers who run scripts directly without
        # the repo on sys.path as a package.
        try:
            from lamquant.common.data_types import TrainingBatch
        except ImportError:
            import os, sys as _sys
            # MOVE-B: common DTOs live in lamquant/common (sibling area).
            _common = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', 'common'))
            if _common not in _sys.path:
                _sys.path.insert(0, _common)
            from data_types import TrainingBatch

        n_total = (self.windows_per_epoch // batch_size) * batch_size

        # Epoch index generation: sampler (clinical-balanced) or uniform random.
        if sampler is not None:
            sampler_iter = iter(sampler)
            idx_list = [next(sampler_iter) for _ in range(n_total)]
            epoch_indices = torch.tensor(idx_list, dtype=torch.long)
            # Clamp to valid range (sampler may return indices for the full
            # manifest but the dataset may be capped at max_windows).
            epoch_indices.clamp_(0, self.n_windows - 1)
        else:
            epoch_indices = torch.randint(0, self.n_windows, (n_total,))

        # Use the same shard-transfer machinery as prefetch_batches.
        if getattr(self, '_on_gpu', False):
            gpu_epoch = self.l3_data[epoch_indices].float()
            # When the dataset is fully GPU-resident the fullband path is
            # only ever in-RAM (it would not fit on a 24 GB GPU). Memmap
            # callers should not use _on_gpu=True.
            gpu_fb_epoch = (self.fullband_data[epoch_indices].float()
                            if self.fullband_data is not None else None)
            for i in range(0, n_total, batch_size):
                end = i + batch_size
                idx_slice = epoch_indices[i:end].tolist()
                fb = gpu_fb_epoch[i:end] if gpu_fb_epoch is not None else None
                yield TrainingBatch(
                    l3_approx=gpu_epoch[i:end],
                    fullband_target=fb,
                    datasets=[self._win_dataset[k] for k in idx_slice],
                    patient_ids=[self._win_patient[k] for k in idx_slice],
                    splits=[self._win_split[k] for k in idx_slice],
                    has_seizure=[self._win_has_seizure[k] for k in idx_slice],
                    event_types=[self._win_event_type[k] for k in idx_slice],
                    clinical_categories=([self._win_clinical_category[k] for k in idx_slice]
                                         if self._win_clinical_category is not None else []),
                )
            return

        if hasattr(self, '_shard_max'):
            shard_max = self._shard_max
        else:
            # L3 (21×313×4) + fullband (21×2500×4 if present). Without
            # the fullband term, large shards OOM at the .to(device)
            # transfer when with_fullband=True (~9× more memory).
            bytes_per_window = 21 * 313 * 4
            if self.has_fullband:
                bytes_per_window += 21 * self._fb_window * 4
            try:
                free_vram = torch.cuda.mem_get_info(device)[0]
                reserve = 14 * 1024**3 if self.has_fullband else 4 * 1024**3
                shard_budget = max(free_vram - reserve, 1 * 1024**3)
            except Exception:
                shard_budget = 8 * 1024**3
            shard_max = int(shard_budget / bytes_per_window)
            shard_max = max(shard_max, batch_size)

        for shard_start in range(0, n_total, shard_max):
            shard_end = min(shard_start + shard_max, n_total)
            shard_idx = epoch_indices[shard_start:shard_end]
            gpu_shard = self.l3_data[shard_idx].to(device).float()  # transfer float16, cast on GPU
            # _fb_fetch handles both in-RAM tensor and memmap-backed paths.
            cpu_fb = self._fb_fetch(shard_idx)
            gpu_fb_shard = (cpu_fb.to(device).float()  # transfer then cast on GPU
                            if cpu_fb is not None else None)
            shard_idx_list = shard_idx.tolist()
            for i in range(0, len(gpu_shard), batch_size):
                end = i + batch_size
                idx_slice = shard_idx_list[i:end]
                fb = gpu_fb_shard[i:end] if gpu_fb_shard is not None else None
                yield TrainingBatch(
                    l3_approx=gpu_shard[i:end],
                    fullband_target=fb,
                    datasets=[self._win_dataset[k] for k in idx_slice],
                    patient_ids=[self._win_patient[k] for k in idx_slice],
                    splits=[self._win_split[k] for k in idx_slice],
                    has_seizure=[self._win_has_seizure[k] for k in idx_slice],
                    event_types=[self._win_event_type[k] for k in idx_slice],
                    clinical_categories=([self._win_clinical_category[k] for k in idx_slice]
                                         if self._win_clinical_category is not None else []),
                )
            del gpu_shard
            if gpu_fb_shard is not None:
                del gpu_fb_shard
