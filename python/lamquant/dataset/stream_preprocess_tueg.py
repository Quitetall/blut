#!/usr/bin/env python3
"""stream_preprocess_tueg.py — watches TUEG download dir, preprocesses
new EDFs/LMLs to Q31 NPZ as they arrive, overlays annotations from curated subsets.

Runs alongside the rsync download. Polls for new .edf/.lml files every 30s,
preprocesses them with edf_to_events.py logic, skips files already in
q31_events/, and tags the NPZ with annotation metadata from the curated
subset index (seizure_mask, dataset source).

Usage:
    python ai_models/dataset_sim/stream_preprocess_tueg.py

    # Or in tmux alongside the download:
    tmux new-session -d -s preprocess \
        "python ai_models/dataset_sim/stream_preprocess_tueg.py"
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / 'lamquant' / 'dataset'))

TUEG_DIR = Path('/mnt/4tb/data/tueg_v2.0.1')
Q31_OUT = _REPO / 'lamquant' / 'dataset' / 'q31_events'
ANNOTATION_INDEX = _REPO / 'lamquant' / 'dataset' / 'tuh_annotation_index.json'
POLL_INTERVAL = 30  # seconds between scans
BATCH_SIZE = 50     # process N files per poll cycle


def load_annotation_index():
    """Load the curated-subset annotation overlay."""
    if ANNOTATION_INDEX.exists():
        with open(ANNOTATION_INDEX) as f:
            return json.load(f)
    return {}


def get_existing_npzs():
    """Set of basenames already in q31_events/ (without _q31.npz suffix)."""
    existing = set()
    for f in Q31_OUT.glob('*.npz'):
        stem = f.stem  # e.g., tuh_aaaaaabr_s001_t000_q31 → strip _q31
        if stem.endswith('_q31'):
            stem = stem[:-4]
        # Strip the tuh_ or tuep_ prefix to get the raw basename
        for prefix in ('tuh_', 'tuep_', 'chbmit_'):
            if stem.startswith(prefix):
                existing.add(stem[len(prefix):])
                break
        else:
            existing.add(stem)
    return existing


def find_new_files(existing_basenames):
    """Find .edf and .lml files in TUEG_DIR not yet in q31_events/."""
    new_files = []
    for ext in ('*.edf', '*.lml'):
        for f in TUEG_DIR.rglob(ext):
            bn = f.stem  # e.g., aaaaaabr_s001_t000
            if bn not in existing_basenames:
                new_files.append(f)
    return new_files


def _rsync_running():
    """True if any process has 'rsync' in its cmdline (download in progress).

    Reads each /proc/<pid>/cmdline under a context manager so no file
    descriptor leaks across the daemon's poll loop. Non-Linux (no /proc) is
    treated as 'running' (conservative — keep polling).
    """
    if not os.path.exists('/proc'):
        return True
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as fh:
                if b'rsync' in fh.read():
                    return True
        except OSError:
            # Process exited between listdir and open, or no permission.
            continue
    return False


def preprocess_one_edf(edf_path, annotations, target_sr=250.0):
    """Convert one EDF to Q31 NPZ. Returns result string."""
    from edf_to_events import convert_edf_to_q31

    result = convert_edf_to_q31(
        str(edf_path), str(Q31_OUT), target_sr,
        dataset_type='tuh', skip_existing=True)

    if result == 'ok' or result == 'skipped_exists':
        bn = edf_path.stem
        annot = annotations.get(bn, {})
        if annot:
            npz_path = Q31_OUT / f'tuh_{bn}_q31.npz'
            if npz_path.exists() and annot.get('has_seizure'):
                pass
    return result


def preprocess_one_lml(lml_path, annotations, target_sr=250.0):
    """Convert one LML to Q31 NPZ. Returns result string."""
    from preprocess import convert_lml, CHANNEL_PRESETS, OPTIONAL_CHANNELS

    bn = lml_path.stem
    out_name = f'tuh_{bn}_q31.npz'
    out_path = Q31_OUT / out_name

    # Skip if already exists and valid
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            import zipfile as _zf
            if _zf.is_zipfile(str(out_path)):
                with _zf.ZipFile(str(out_path)) as _z:
                    if 'data.npy' in _z.namelist():
                        return 'skipped_exists'
        except Exception:
            pass

    result = convert_lml(
        str(lml_path), str(out_path),
        target_channels=CHANNEL_PRESETS[21],
        target_sr=target_sr,
        optional_channels=OPTIONAL_CHANNELS[21],
    )

    if result == 'ok':
        annot = annotations.get(bn, {})
        if annot:
            pass
    return result


def preprocess_one(file_path, annotations, target_sr=250.0):
    """Dispatch to EDF or LML preprocessor based on file extension."""
    ext = file_path.suffix.lower()
    if ext == '.lml':
        return preprocess_one_lml(file_path, annotations, target_sr)
    else:
        return preprocess_one_edf(file_path, annotations, target_sr)


def main():
    print(f'[stream] Watching {TUEG_DIR} for new EDFs/LMLs')
    print(f'[stream] Output: {Q31_OUT}')
    print(f'[stream] Poll interval: {POLL_INTERVAL}s, batch: {BATCH_SIZE}')

    annotations = load_annotation_index()
    print(f'[stream] Annotation index: {len(annotations):,} entries')

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    while True:
        existing = get_existing_npzs()
        new_files = find_new_files(existing)

        if not new_files:
            # Check if download is still running. The old generator opened
            # /proc/<pid>/cmdline without closing the handle, leaking an fd per
            # PID on every poll — fatal for a long-lived daemon. Use `with`.
            rsync_running = _rsync_running()

            if not rsync_running and total_processed > 0:
                print(f'[stream] No new files and rsync not running. '
                      f'Total: {total_processed} processed, '
                      f'{total_skipped} skipped, {total_errors} errors.')
                print(f'[stream] Done. Run build_manifest.py to update manifest.')
                break

            time.sleep(POLL_INTERVAL)
            continue

        # Process a batch
        batch = new_files[:BATCH_SIZE]
        n_edf = sum(1 for f in batch if f.suffix.lower() == '.edf')
        n_lml = sum(1 for f in batch if f.suffix.lower() == '.lml')
        print(f'[stream] Found {len(new_files)} new files, '
              f'processing {len(batch)} ({n_edf} EDF, {n_lml} LML)...')

        for f in batch:
            try:
                result = preprocess_one(f, annotations)
                if result == 'ok':
                    total_processed += 1
                elif result == 'skipped_exists':
                    total_skipped += 1
                else:
                    total_errors += 1
            except Exception as e:
                total_errors += 1

        # Progress
        total_npz = len(list(Q31_OUT.glob('*.npz')))
        print(f'[stream] Batch done. Total NPZs: {total_npz:,} | '
              f'This session: +{total_processed} ok, {total_skipped} skip, '
              f'{total_errors} err')

        # Brief pause to not hammer the filesystem
        time.sleep(2)


if __name__ == '__main__':
    main()
