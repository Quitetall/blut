#!/usr/bin/env python3
"""precompute_fullband_memmap.py — flatten manifest fullband targets into a memmap.

The production training run (Tier 7, full manifest) needs the raw
fullband EEG window for every L3 token in the dataset so the joint
loss can compare decoder output against fullband ground truth (Mode 1
neural-only path). RAM math:

    1.29M train windows × 21 × 2500 × 2 bytes (fp16) = 136 GB

That doesn't fit on a 64 GB system. The MemmapTeacherDataset pattern
already in streaming_dataset.py solves the same shape of problem for
the teacher pipeline — flatten everything into a single contiguous
.dat file, mmap it, let the OS page cache handle hot/cold management.

This script is the precompute side. It iterates DatasetManifest entries
in canonical order, reads each NPZ's raw `data` array, slices into
2500-sample windows aligned with the file's L3 windows, converts Q31 →
microvolts (matching the encoder-input scaling), writes float16 into
the memmap, and records a sidecar JSON with the (file, n_windows)
schedule so the dataset can verify the order.

Usage:
    # Default: train + val splits → fullband_train.dat, fullband_val.dat
    python ai_models/dataset_sim/precompute_fullband_memmap.py

    # One split only:
    python ai_models/dataset_sim/precompute_fullband_memmap.py --splits train

    # Different output dir:
    python ai_models/dataset_sim/precompute_fullband_memmap.py --out /scratch/

The output is idempotent: re-running on the same manifest produces a
byte-identical .dat (modulo the meta JSON `created` timestamp).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np

# MOVE-B (2026-05-29): now at blut/python/lamquant/dataset/. parents[2]
# is the blut/python package root; putting it on sys.path lets
# `from lamquant.common.data_types import ...` resolve as a package.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from lamquant.common.data_types import DatasetManifest, FileEntry, Split  # rewritten 2026-05-16 for legacy/ relocation

WINDOW = 2500
N_CHANNELS = 21
DTYPE = np.float16
Q31 = 2147483647.0
UV_PER_Q31 = 1000.0


def _split_name(s: Split) -> str:
    return s.value


def _file_window_count_from_npz(npz_path: str) -> int:
    """Return the number of L3 windows in the NPZ (header read only)."""
    import zipfile
    from numpy.lib.format import (
        read_magic, read_array_header_1_0, read_array_header_2_0,
    )
    try:
        with zipfile.ZipFile(npz_path) as zf:
            if 'l3.npy' not in zf.namelist():
                return 0
            with zf.open('l3.npy') as f:
                ver = read_magic(f)
                if ver == (1, 0):
                    shape, _, _ = read_array_header_1_0(f)
                elif ver == (2, 0):
                    shape, _, _ = read_array_header_2_0(f)
                else:
                    from numpy.lib.format import _read_array_header
                    shape, _, _ = _read_array_header(f, ver)
                return int(shape[0]) if shape else 0
    except Exception:
        return 0


def precompute_split(file_entries: List[FileEntry],
                     dat_path: Path, meta_path: Path,
                     verbose: bool = True) -> dict:
    """Build the flattened memmap for one split.

    Returns a meta dict with ordered file schedule + total window count.
    """
    # Pass 1: count windows per file (header read).
    if verbose:
        print(f'[*] Pass 1/2: counting windows in {len(file_entries):,} files...')
    schedule = []
    total = 0
    for fe in file_entries:
        n = _file_window_count_from_npz(fe.path)
        if n <= 0:
            if verbose:
                print(f'    [skip] {os.path.basename(fe.path)}: no l3 windows')
            continue
        schedule.append({
            'path': fe.path,
            'dataset': fe.dataset.value,
            'patient_id': fe.patient_id,
            'n_windows': n,
            'offset': total,
        })
        total += n
    if total == 0:
        raise RuntimeError(f'no usable windows in {len(file_entries)} entries')

    gb = total * N_CHANNELS * WINDOW * 2 / 1e9
    if verbose:
        print(f'    {total:,} windows total → {gb:.1f} GB on disk')

    # Pass 2: write the memmap.
    if verbose:
        print(f'[*] Pass 2/2: writing {dat_path} ...')
    dat_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(dat_path, dtype=DTYPE, mode='w+',
                   shape=(total, N_CHANNELS, WINDOW))

    t0 = time.perf_counter()
    written = 0
    skipped = 0
    actual_total = 0   # windows actually written (vs scheduled)
    scale = np.float32(UV_PER_Q31 / Q31)   # one multiply, baked once
    for i, entry in enumerate(schedule):
        try:
            with np.load(entry['path']) as d:
                data = d['data']    # int32 [21, T]
                T = data.shape[1]
                n = entry['n_windows']
                offset = entry['offset']
                # Vectorise: process the full-window prefix in one shot
                # (single reshape + dtype convert + memmap write). The
                # previous per-k Python loop was ~20× slower on TUH-sized
                # files because each window cost a fresh
                # NPZ-decompressed-slice + float32 convert + 105 KB
                # memmap write. Batching the full-window prefix amortises
                # all three.
                n_full = min(n, T // WINDOW)
                if n_full > 0:
                    block = (data[:, :n_full * WINDOW]
                             .reshape(N_CHANNELS, n_full, WINDOW)
                             .transpose(1, 0, 2))                 # [n_full, 21, 2500]
                    block_uv = (block.astype(np.float32) * scale).astype(DTYPE)
                    mm[offset:offset + n_full] = block_uv
                # Handle the tail (zero-padded last window) if any.
                for k in range(n_full, n):
                    start = k * WINDOW
                    seg = np.zeros((N_CHANNELS, WINDOW), dtype=np.float32)
                    end = min(start + WINDOW, T)
                    if end > start:
                        seg[:, :end - start] = (
                            data[:, start:end].astype(np.float32) * scale)
                    mm[offset + k] = seg.astype(DTYPE)
                actual_total += n
        except Exception as e:
            skipped += 1
            if verbose:
                print(f'    [error] {os.path.basename(entry["path"])}: {e}')
            continue
        written += 1
        if verbose and (i + 1) % 500 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            eta = (len(schedule) - i - 1) / rate
            print(f'    {i+1:,}/{len(schedule):,} files  '
                  f'({rate:.0f}/s, ETA {eta/60:.0f} min)')

    mm.flush()
    del mm

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f'[*] Wrote {dat_path} in {elapsed/60:.1f} min '
              f'({written}/{len(schedule)} files OK, {skipped} skipped)')

    meta = {
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'window_samples': WINDOW,
        'n_channels': N_CHANNELS,
        'dtype': 'float16',
        'unit': 'microvolts',
        'q31_scale': Q31,
        'uv_per_q31': UV_PER_Q31,
        'n_windows': total,
        'n_files': len(schedule),
        'dat_path': str(dat_path),
        'dat_bytes': total * N_CHANNELS * WINDOW * 2,
        'schedule': schedule,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f'[*] Wrote {meta_path}')
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Flatten manifest fullband targets into a memmap.')
    parser.add_argument('--manifest', type=Path,
                        default=_REPO / 'lamquant' / 'dataset' / 'manifest_v3.json',
                        help='manifest_v3.json path')
    parser.add_argument('--out', type=Path,
                        default=_REPO / 'lamquant' / 'dataset',
                        help='output directory for fullband_{split}.dat + .meta.json')
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'val'],
                        help="splits to precompute (default: train val)")
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    manifest = DatasetManifest.load(args.manifest)
    print(f'[*] Loaded manifest: {manifest.train_files:,} train files, '
          f'{manifest.val_files:,} val files')
    args.out.mkdir(parents=True, exist_ok=True)

    name_to_split = {'train': Split.TRAIN, 'val': Split.VAL,
                     'test': Split.TEST, 'holdout': Split.HOLDOUT}

    for split_name in args.splits:
        if split_name not in name_to_split:
            print(f'[!] Unknown split {split_name!r}; skipping')
            continue
        s = name_to_split[split_name]
        entries = manifest.get_file_entries(s)
        if not entries:
            print(f'[*] Split {split_name}: no entries; skipping')
            continue
        print(f'\n[*] === Split: {split_name} ({len(entries):,} files) ===')
        dat_path = args.out / f'fullband_{split_name}.dat'
        meta_path = args.out / f'fullband_{split_name}.meta.json'
        precompute_split(entries, dat_path, meta_path, verbose=not args.quiet)

    print('\n[*] Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
