#!/usr/bin/env python3
"""
Fast L3 Precomputation for Q31 Dataset

Adds pre-computed L3 approximations to existing Q31 NPZ files.
Eliminates the 94ms per-sample bottleneck during training.

Uses:
- Parallel processing (N workers, one per NPZ file)
- NumPy vectorization for lifting DWT
- FFT-based autocorrelation for faster LPC

Usage:
    python3 precompute_l3_fast.py --input ai_models/dataset_sim/q31_events

Output: Updates NPZ files in-place, adding 'l3' array [num_windows, 21, 313]
"""

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import tempfile
import shutil

# MOVE-B (2026-05-29): now at blut/python/lamquant/student/. Put the
# blut/python package root on sys.path so the lamquant.* package import
# below resolves.
sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from lamquant.student.subband_preprocess import preprocess_subband_single  # rewritten 2026-05-16 for legacy/ relocation


def precompute_file_l3(npz_path):
    """Load Q31 NPZ, compute L3 for all windows, update file in-place.

    Skips files that already have an `l3` key so the script is idempotent
    across restarts. Uses an atomic rename via a `.npz`-suffixed tempfile so
    `np.savez_compressed` cannot silently redirect the output to a different
    path (it auto-appends `.npz` when the target lacks that extension).
    """
    try:
        # Guard against 0-byte files (causes indefinite hang in np.load)
        if os.path.getsize(npz_path) == 0:
            os.remove(npz_path)
            return (os.path.basename(npz_path), 0, False, "empty_file_deleted")

        # Load NPZ and fully materialize arrays before touching the file.
        # Preserve EVERY existing key (Rule 27 graceful, Rule 6 boundary
        # validation) — earlier versions of this script enumerated a fixed
        # subset which silently dropped `original_sample_rate`,
        # `subject_id`, `metadata_json`, etc. Catching that regression is
        # the job of `test_data_pipeline_e2e.py`.
        with np.load(npz_path, allow_pickle=True) as data:
            # Idempotent short-circuit: file already has L3
            if 'l3' in data.files:
                existing_l3 = data['l3']
                return (os.path.basename(npz_path),
                        int(existing_l3.shape[0]), True, None)
            assert 'data' in data.files, (
                f"{npz_path}: required `data` key missing — corrupt NPZ?"
            )
            eeg_q31 = np.asarray(data['data'])
            save_dict = {k: np.asarray(data[k]) for k in data.files}

        # Denormalize to float for processing
        eeg_float = (eeg_q31.astype(np.float32) / 2147483647.0) * 1000.0

        # Extract windows [21, 2500] — FLOOR division to avoid zero-padded
        # runt windows that train the model on near-silence garbage.
        T = eeg_float.shape[1]
        num_windows = T // 2500

        if num_windows == 0:
            # Recording is shorter than a single 10-second window. Mark
            # the file explicitly so downstream code can detect + skip
            # rather than silently miss the `l3` key (Rule 27, Rule 6).
            save_dict['l3'] = np.zeros((0, 21, 313), dtype=np.float32)
            save_dict['l3_too_short'] = np.bool_(True)
        else:
            l3_list = []
            for w in range(num_windows):
                start_idx = w * 2500
                end_idx = min(start_idx + 2500, T)
                window = eeg_float[:, start_idx:end_idx].astype(np.float32)
                if window.shape[1] < 2500:
                    window = np.pad(window, ((0, 0), (0, 2500 - window.shape[1])))
                l3, _, _ = preprocess_subband_single(window, order=8, autocorr_len=256)
                l3_list.append(l3)
            save_dict['l3'] = np.stack(l3_list, axis=0).astype(np.float32)

        # Atomic write: create a *.npz-suffixed tempfile in the same directory
        # so `np.savez_compressed` doesn't silently append `.npz` to a
        # non-`.npz` name and write to a different path than we rename from.
        fd, tmp_path = tempfile.mkstemp(
            prefix='.l3_precompute_', suffix='.npz',
            dir=os.path.dirname(npz_path),
        )
        os.close(fd)
        try:
            np.savez_compressed(tmp_path, **save_dict)
            # savez_compressed writes to exactly `tmp_path` when it already
            # ends in `.npz`, so the rename moves the real payload.
            os.replace(tmp_path, npz_path)
        except Exception:  # NPZ read or compute error — skip this file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        return (os.path.basename(npz_path), num_windows, True, None)

    except Exception as e:
        return (os.path.basename(npz_path), 0, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="Precompute L3 for Q31 dataset")
    parser.add_argument('--input', default='ai_models/dataset_sim/q31_events',
                        help='Directory with Q31 NPZ files')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"[!] Directory not found: {input_dir}")
        sys.exit(1)

    npz_files = sorted(input_dir.glob('*.npz'))
    if not npz_files:
        print(f"[!] No NPZ files found in {input_dir}")
        sys.exit(1)

    # Pre-cleanup: remove any 0-byte files that might have been created by failed conversion
    zero_byte_files = [f for f in npz_files if f.stat().st_size == 0]
    if zero_byte_files:
        print(f"[*] Removing {len(zero_byte_files)} stale 0-byte files before processing")
        for f in zero_byte_files:
            f.unlink()
        npz_files = sorted(input_dir.glob('*.npz'))

    print(f"[*] Precomputing L3 for {len(npz_files)} NPZ files")
    print(f"[*] Using {args.workers} parallel workers")
    print()

    total_windows = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(precompute_file_l3, str(f)): f for f in npz_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Precomputing L3"):
            filename, num_windows, success, error = future.result()

            if success:
                total_windows += num_windows
            else:
                failed.append((filename, error))

    print()
    print(f"✓ Precomputation complete")
    print(f"  Files processed: {len(npz_files) - len(failed)}/{len(npz_files)}")
    print(f"  Total windows: {total_windows:,}")
    print(f"  Windows per file (avg): {total_windows // (len(npz_files) - len(failed)):.0f}")

    if failed:
        print()
        print(f"[!] {len(failed)} files failed:")
        for filename, error in failed[:10]:
            print(f"  • {filename}: {error}")

    print()
    print("L3 precomputation complete. DataLoader can now load L3 instantly.")


if __name__ == '__main__':
    main()
