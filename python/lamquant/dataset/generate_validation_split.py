#!/usr/bin/env python3
"""
LamQuant Validation Split Generator
=====================================
Generates a deterministic, reproducible validation manifest from
downloaded EEG datasets. The manifest specifies exactly which files
and windows are held out from training.

This script runs ONCE before training. The output manifest is committed
to the git repo so that:
  1. Everyone trains on the same data
  2. Everyone validates on the same data
  3. Nobody accidentally trains on validation data
  4. Results are reproducible across machines

Usage:
  # Just run it. No arguments needed.
  python generate_validation_split.py

  # Output goes to: ./validation_manifest/validation_manifest.json
  # Datasets expected at: ./datasets/

  # Optional overrides:
  python generate_validation_split.py --ratio 0.10 --seed 42
  python generate_validation_split.py --verify validation_manifest/validation_manifest.json

The manifest does NOT contain any signal data — only file paths,
window indices, and checksums. It is safe and legal to commit to
a public repo regardless of dataset license terms.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# MOVE-B (2026-05-29): now at blut/python/lamquant/dataset/. Put the
# blut/python package root on sys.path so the lazy
# `from lamquant.dataset.channel_resolver import ...` (below) resolves
# as a package regardless of cwd / launch style.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    import pyedflib
    HAS_PYEDF = True
except ImportError:
    HAS_PYEDF = False


# ============================================================
# Constants
# ============================================================

WINDOW_SAMPLES = 2500       # 10 seconds at 250 Hz
TARGET_FS = 250             # Resample everything to this
MIN_CHANNELS = 8            # Skip files with fewer usable channels
MANIFEST_VERSION = "2.0.0"

# Dataset configurations
DATASETS = {
    "chbmit": {
        "name": "CHB-MIT Scalp EEG Database",
        "source": "s3://physionet-open/chbmit/1.0.0/",
        "license": "Open Data Commons Attribution License v1.0",
        "doi": "10.13026/C2K01R",
        "glob_pattern": "**/chb*/chb*.edf",
        "train_subjects": [f"chb{i:02d}" for i in range(1, 15)],
        "holdout_subjects": [f"chb{i:02d}" for i in range(15, 25)],
        "fs_native": 256,
        "channels_expected": 23,
        "has_seizure_annotations": True,
    },
    "siena": {
        "name": "Siena Scalp EEG Database",
        "source": "s3://physionet-open/siena-scalp-eeg/1.0.0/",
        "license": "Open Data Commons Attribution License v1.0",
        "doi": "10.13026/5d4a-j060",
        "glob_pattern": "**/PN*/*.edf",
        "train_subjects": [],  # ALL subjects are validation (cross-site)
        "holdout_subjects": "ALL",
        "fs_native": 512,
        "channels_expected": 31,
        "has_seizure_annotations": True,
    },
    "eegmmidb": {
        "name": "EEG Motor Movement/Imagery Dataset",
        "source": "s3://physionet-open/eegmmidb/1.0.0/",
        "license": "Open Data Commons Attribution License v1.0",
        "doi": "10.13026/C28G6P",
        "glob_pattern": "**/S*/S*R*.edf",
        "train_subjects": [f"S{i:03d}" for i in range(1, 80)],
        "holdout_subjects": [f"S{i:03d}" for i in range(80, 110)],
        "fs_native": 160,
        "channels_expected": 64,
        "has_seizure_annotations": False,
    },
    "mental_arithmetic": {
        "name": "Mental Arithmetic EEG Dataset",
        "source": "s3://physionet-open/eeg-during-mental-arithmetic-tasks/1.0.0/",
        "license": "Open Data Commons Attribution License v1.0",
        "doi": "10.13026/C2JQ1M",
        "glob_pattern": "**/*.edf",
        "train_subjects": [],  # ALL subjects are validation
        "holdout_subjects": "ALL",
        "fs_native": 500,
        "channels_expected": 19,
        "has_seizure_annotations": False,
    },
    "tuh_seizure": {
        "name": "TUH Seizure Corpus v2.0.6",
        "source": "https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml",
        "license": "TUH EEG EULA (research use)",
        "doi": "",
        "glob_pattern": "**/*.edf",
        "train_subjects": "ALL",
        "holdout_subjects": [],
        "fs_native": 250,
        "channels_expected": 33,
        "has_seizure_annotations": True,
    },
    "tuh_artifact": {
        "name": "TUH EEG Artifact Corpus v3.0.1",
        "source": "https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml",
        "license": "TUH EEG EULA (research use)",
        "doi": "",
        "glob_pattern": "**/*.edf",
        "train_subjects": "ALL",
        "holdout_subjects": [],
        "fs_native": 250,
        "channels_expected": 36,
        "has_seizure_annotations": False,
    },
    "tuh_epilepsy": {
        "name": "TUH EEG Epilepsy Corpus v3.0.0",
        "source": "https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml",
        "license": "TUH EEG EULA (research use)",
        "doi": "",
        "glob_pattern": "**/*.edf",
        "train_subjects": [],  # ALL subjects are validation
        "holdout_subjects": "ALL",
        "fs_native": 250,
        "channels_expected": 33,
        "has_seizure_annotations": False,
    },
    "tuh_events": {
        "name": "TUH EEG Events Corpus v2.0.1",
        "source": "https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml",
        "license": "TUH EEG EULA (research use)",
        "doi": "",
        "glob_pattern": "**/*.edf",
        "train_subjects": [],  # ALL subjects are validation
        "holdout_subjects": "ALL",
        "fs_native": 250,
        "channels_expected": 33,
        "has_seizure_annotations": False,
    },
}


# ============================================================
# Channel name normalization (delegates to shared channel_resolver)
# ============================================================

from lamquant.dataset.channel_resolver import resolve as _cr_resolve  # rewritten 2026-05-16 for legacy/ relocation

def normalize_channel(name: str) -> Optional[str]:
    """Resolve channel name to canonical form. Delegates to channel_resolver."""
    return _cr_resolve(name)


# ============================================================
# File scanning and metadata extraction
# ============================================================

def get_file_hash(filepath: str) -> str:
    """SHA256 of first 4096 bytes (fast, identifies the file uniquely)."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        h.update(f.read(4096))
    return h.hexdigest()[:16]


def get_edf_metadata(filepath: str) -> Optional[dict]:
    """Extract metadata from an EDF file without loading all data."""
    try:
        if HAS_MNE:
            raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
            ch_names = raw.ch_names
            fs = int(raw.info['sfreq'])
            duration_sec = raw.n_times / fs
            n_samples = raw.n_times
        elif HAS_PYEDF:
            f = pyedflib.EdfReader(filepath)
            ch_names = f.getSignalLabels()
            fs = int(f.getSampleFrequency(0))
            n_samples = f.getNSamples()[0]
            duration_sec = n_samples / fs
            f.close()
        else:
            return None

        # Count usable channels
        usable = [ch for ch in ch_names if normalize_channel(ch) is not None]

        # Compute windows at target sample rate
        target_samples = int(n_samples * TARGET_FS / fs)
        num_windows = target_samples // WINDOW_SAMPLES

        return {
            'channels_total': len(ch_names),
            'channels_usable': len(usable),
            'channel_names': usable,
            'sample_rate': fs,
            'duration_sec': duration_sec,
            'num_windows': num_windows,
            'file_hash': get_file_hash(filepath),
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
        }
    except Exception as e:
        return None


def extract_subject_id(filepath: str, dataset_key: str) -> str:
    """Extract subject identifier from file path."""
    parts = Path(filepath).parts

    if dataset_key == 'chbmit':
        for part in parts:
            # Match 'chb01' through 'chb24' — NOT 'chbmit' (the dataset dir)
            if part.startswith('chb') and len(part) <= 5 and any(c.isdigit() for c in part):
                return part
    elif dataset_key == 'siena':
        for part in parts:
            if part.startswith('PN'):
                return part
    elif dataset_key == 'eegmmidb':
        for part in parts:
            if part.startswith('S') and len(part) == 4 and part[1:].isdigit():
                return part
    elif dataset_key == 'mental_arithmetic':
        for part in parts:
            if part.startswith('Subject'):
                return part
    elif dataset_key in ('tuh_seizure', 'tuh_epilepsy', 'tuh_artifact',
                          'tuh_events'):
        # TUH datasets: extract subject from filename prefix.
        # Filename: aaaaaajy_s001_t000.edf -> subject "aaaaaajy"
        # Filename: bckg_000_a_.edf -> subject "bckg_000"
        # This is more reliable than path parsing because TUH path
        # structures vary across corpus versions.
        fname = os.path.splitext(os.path.basename(filepath))[0]
        # TUH Seizure/Epilepsy: "aaaaaajy_s001_t000" -> "aaaaaajy"
        # TUH Artifact: "aaaaaaju_s005_t000" -> "aaaaaaju"
        # TUH Events: "bckg_000_a_" -> "bckg_000"
        parts_fname = fname.split('_')
        if len(parts_fname) >= 2 and len(parts_fname[0]) >= 4:
            # Check if first part looks like a TUH patient hash (8 lowercase alpha)
            if len(parts_fname[0]) == 8 and parts_fname[0].isalpha():
                return parts_fname[0]
            # Otherwise use first two parts as subject ID
            return '_'.join(parts_fname[:2])

    return os.path.basename(os.path.dirname(filepath))


# ============================================================
# Stratified sampling
# ============================================================

def stratified_window_sample(
    file_entries: List[dict],
    ratio: float,
    rng: np.random.Generator,
    ensure_seizure_coverage: bool = True,
) -> List[dict]:
    """
    Sample validation windows from a list of file entries.
    Stratifies by subject to ensure every subject contributes
    proportionally to the validation set.
    """
    # Group by subject
    by_subject = {}
    for entry in file_entries:
        subj = entry['subject']
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append(entry)

    sampled = []

    for subject, entries in sorted(by_subject.items()):
        # Collect all available windows across this subject's files
        all_windows = []
        for entry in entries:
            for w in range(entry['num_windows']):
                all_windows.append({
                    'file': entry['relative_path'],
                    'window_idx': w,
                    'subject': subject,
                    'file_hash': entry['file_hash'],
                    'has_seizure': entry.get('has_seizure', False),
                })

        if not all_windows:
            continue

        # Number of validation windows for this subject
        n_val = max(1, int(len(all_windows) * ratio))

        # If seizure annotations exist, ensure at least one seizure window
        if ensure_seizure_coverage:
            seizure_windows = [w for w in all_windows if w.get('has_seizure')]
            non_seizure_windows = [w for w in all_windows if not w.get('has_seizure')]

            if seizure_windows:
                # Sample at least 1 seizure window
                n_seizure = max(1, int(n_val * len(seizure_windows) / len(all_windows)))
                n_non_seizure = n_val - n_seizure

                seizure_idx = rng.choice(len(seizure_windows),
                                          size=min(n_seizure, len(seizure_windows)),
                                          replace=False)
                non_seizure_idx = rng.choice(len(non_seizure_windows),
                                              size=min(n_non_seizure, len(non_seizure_windows)),
                                              replace=False)

                for i in seizure_idx:
                    sampled.append(seizure_windows[i])
                for i in non_seizure_idx:
                    sampled.append(non_seizure_windows[i])
                continue

        # Random sample without seizure stratification
        indices = rng.choice(len(all_windows), size=min(n_val, len(all_windows)),
                              replace=False)
        for i in indices:
            sampled.append(all_windows[i])

    return sampled


# ============================================================
# Seizure annotation parsing
# ============================================================

def parse_chbmit_seizures(data_dir: str) -> dict:
    """Parse CHB-MIT summary files to identify which files contain seizures."""
    seizure_files = set()
    chbmit_dir = os.path.join(data_dir, 'chbmit')

    # Read RECORDS-WITH-SEIZURES
    records_file = os.path.join(chbmit_dir, 'RECORDS-WITH-SEIZURES')
    if os.path.exists(records_file):
        with open(records_file) as f:
            for line in f:
                line = line.strip()
                if line and line.endswith('.edf'):
                    seizure_files.add(line)

    return seizure_files


# ============================================================
# Main manifest generation
# ============================================================

def scan_dataset(
    data_dir: str,
    dataset_key: str,
    config: dict,
    verbose: bool = True
) -> List[dict]:
    """Scan a dataset directory and extract file metadata."""
    dataset_dir = os.path.join(data_dir, dataset_key)
    if not os.path.isdir(dataset_dir):
        if verbose:
            print(f"  Dataset not found: {dataset_dir}")
        return []

    pattern = os.path.join(dataset_dir, config['glob_pattern'])
    edf_files = sorted(glob.glob(pattern, recursive=True))

    if not edf_files:
        # Try simpler pattern
        edf_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.edf'),
                                      recursive=True))

    if verbose:
        print(f"  Found {len(edf_files)} EDF files in {dataset_key}")

    # Get seizure annotations for CHB-MIT
    seizure_files = set()
    if dataset_key == 'chbmit':
        seizure_files = parse_chbmit_seizures(data_dir)

    entries = []
    skipped = 0

    for filepath in edf_files:
        meta = get_edf_metadata(filepath)
        if meta is None or meta['channels_usable'] < MIN_CHANNELS:
            skipped += 1
            continue

        relative_path = os.path.relpath(filepath, dataset_dir)
        subject = extract_subject_id(filepath, dataset_key)

        # Check if this subject is in holdout
        holdout_subjects = config['holdout_subjects']
        if holdout_subjects == "ALL":
            is_holdout = True
        elif subject in holdout_subjects:
            is_holdout = True
        else:
            is_holdout = False

        # Check seizure annotation
        has_seizure = False
        if dataset_key == 'chbmit':
            for sf in seizure_files:
                if relative_path.endswith(sf) or sf in relative_path:
                    has_seizure = True
                    break

        entries.append({
            'relative_path': relative_path,
            'subject': subject,
            'is_holdout': is_holdout,
            'has_seizure': has_seizure,
            'num_windows': meta['num_windows'],
            'channels_usable': meta['channels_usable'],
            'sample_rate': meta['sample_rate'],
            'duration_sec': meta['duration_sec'],
            'file_hash': meta['file_hash'],
            'file_size_mb': meta['file_size_mb'],
        })

    if verbose:
        print(f"  Usable: {len(entries)} files, skipped: {skipped}")
        subjects = set(e['subject'] for e in entries)
        print(f"  Subjects: {len(subjects)}")
        total_windows = sum(e['num_windows'] for e in entries)
        print(f"  Total windows: {total_windows}")
        seizure_count = sum(1 for e in entries if e['has_seizure'])
        if seizure_count > 0:
            print(f"  Files with seizures: {seizure_count}")

    return entries


def generate_manifest(
    data_dir: str,
    ratio: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Generate the complete validation manifest."""
    rng = np.random.default_rng(seed)

    manifest = {
        'manifest_version': MANIFEST_VERSION,
        'created': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'generator': 'generate_validation_split.py',
        'seed': seed,
        'sample_ratio': ratio,
        'window_samples': WINDOW_SAMPLES,
        'target_sample_rate': TARGET_FS,
        'min_channels': MIN_CHANNELS,
        'description': (
            f"Deterministic {ratio*100:.0f}% stratified validation split across "
            f"all EEG datasets. DO NOT train on any window listed in this manifest. "
            f"Regenerate with the same seed to get identical splits."
        ),
        'instructions': {
            'for_training': (
                "Before training, load this manifest and exclude all listed "
                "file/window pairs from your training DataLoader. The helper "
                "function is_validation_window() in this repo handles this."
            ),
            'for_validation': (
                "Run validate_cross_dataset.py with --manifest flag to validate "
                "only on the windows specified in this file."
            ),
        },
        'datasets': {},
        'summary': {},
    }

    total_train_windows = 0
    total_val_windows = 0
    total_files = 0
    total_subjects = set()

    for dataset_key, config in DATASETS.items():
        if verbose:
            print(f"\nScanning {config['name']}...")

        entries = scan_dataset(data_dir, dataset_key, config, verbose)
        if not entries:
            continue

        # Split: holdout subjects go entirely to validation
        # Training subjects get ratio% of windows sampled for validation
        holdout_entries = [e for e in entries if e['is_holdout']]
        train_entries = [e for e in entries if not e['is_holdout']]

        # Sample from training subjects
        train_val_windows = []
        if train_entries:
            train_val_windows = stratified_window_sample(
                train_entries, ratio, rng,
                ensure_seizure_coverage=config['has_seizure_annotations']
            )

        # Sample from holdout subjects (all are validation, but still sample
        # a subset to keep validation runtime reasonable)
        holdout_val_windows = []
        if holdout_entries:
            holdout_ratio = min(0.20, ratio * 4)  # More aggressive sampling for holdout
            holdout_val_windows = stratified_window_sample(
                holdout_entries, holdout_ratio, rng,
                ensure_seizure_coverage=config['has_seizure_annotations']
            )

        all_val_windows = train_val_windows + holdout_val_windows

        # Organize by file
        files_dict = {}
        for w in all_val_windows:
            fpath = w['file']
            if fpath not in files_dict:
                files_dict[fpath] = {
                    'subject': w['subject'],
                    'file_hash': w['file_hash'],
                    'has_seizure': w.get('has_seizure', False),
                    'windows': [],
                    'split': 'holdout' if w['subject'] in [
                        e['subject'] for e in holdout_entries
                    ] else 'train_val',
                }
            files_dict[fpath]['windows'].append(w['window_idx'])

        # Sort windows within each file
        for f in files_dict.values():
            f['windows'] = sorted(set(f['windows']))

        # Compute stats
        total_dataset_windows = sum(e['num_windows'] for e in entries)
        val_window_count = sum(len(f['windows']) for f in files_dict.values())
        subjects_in_val = set(f['subject'] for f in files_dict.values())

        manifest['datasets'][dataset_key] = {
            'name': config['name'],
            'source': config['source'],
            'license': config['license'],
            'doi': config.get('doi', ''),
            'native_sample_rate': config['fs_native'],
            'total_files': len(entries),
            'total_subjects': len(set(e['subject'] for e in entries)),
            'total_windows': total_dataset_windows,
            'validation_files': len(files_dict),
            'validation_windows': val_window_count,
            'validation_subjects': len(subjects_in_val),
            'validation_ratio_actual': val_window_count / max(1, total_dataset_windows),
            'holdout_subjects': (
                "ALL" if config.get('holdout_subjects') == "ALL" else
                list(set(e['subject'] for e in holdout_entries))
                if holdout_entries else []
            ),
            'files': files_dict,
        }

        total_train_windows += total_dataset_windows - val_window_count
        total_val_windows += val_window_count
        total_files += len(entries)
        total_subjects.update(e['subject'] for e in entries)

        if verbose:
            print(f"  Validation: {val_window_count}/{total_dataset_windows} windows "
                  f"({val_window_count/max(1,total_dataset_windows)*100:.1f}%) "
                  f"from {len(subjects_in_val)} subjects across {len(files_dict)} files")

    manifest['summary'] = {
        'total_datasets': len(manifest['datasets']),
        'total_files_scanned': total_files,
        'total_subjects': len(total_subjects),
        'total_train_windows': total_train_windows,
        'total_validation_windows': total_val_windows,
        'effective_validation_ratio': total_val_windows / max(1, total_train_windows + total_val_windows),
    }

    return manifest


# ============================================================
# Helper function for training scripts to use
# ============================================================

HELPER_CODE = '''
# ============================================================
# Copy this into your training script or import from this module
# ============================================================

import json
import os
from typing import Set, Tuple

def load_validation_manifest(manifest_path: str = None) -> dict:
    """Load the validation manifest. Auto-detects path if not specified."""
    if manifest_path is None:
        # Look relative to the repo root
        candidates = [
            "validation_manifest/validation_manifest.json",
            os.path.join(os.path.dirname(__file__), "validation_manifest", "validation_manifest.json"),
            os.path.join(os.path.dirname(__file__), "..", "validation_manifest", "validation_manifest.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                manifest_path = c
                break
        if manifest_path is None:
            raise FileNotFoundError(
                "Validation manifest not found. Run: python generate_validation_split.py"
            )
    with open(manifest_path) as f:
        return json.load(f)

def get_excluded_windows(manifest: dict) -> Set[Tuple[str, int]]:
    """
    Get set of (filepath, window_idx) tuples that must be excluded from training.
    
    Usage in your DataLoader:
        manifest = load_validation_manifest()
        excluded = get_excluded_windows(manifest)
        
        for file_path, window_idx in your_training_data:
            relative = os.path.relpath(file_path, dataset_dir)
            if (relative, window_idx) in excluded:
                continue  # Skip this window — it's validation data
            # ... train on this window
    """
    excluded = set()
    for dataset_key, dataset in manifest.get('datasets', {}).items():
        for filepath, file_info in dataset.get('files', {}).items():
            for window_idx in file_info.get('windows', []):
                excluded.add((filepath, window_idx))
    return excluded

def is_validation_window(manifest: dict, dataset_key: str, filepath: str, window_idx: int) -> bool:
    """Check if a specific window is in the validation set."""
    dataset = manifest.get('datasets', {}).get(dataset_key, {})
    file_info = dataset.get('files', {}).get(filepath, {})
    return window_idx in file_info.get('windows', [])

def is_holdout_subject(manifest: dict, dataset_key: str, subject: str) -> bool:
    """Check if a subject is in the holdout set (never train on ANY of their data)."""
    dataset = manifest.get('datasets', {}).get(dataset_key, {})
    holdout = dataset.get('holdout_subjects', [])
    if holdout == "ALL":
        return True
    return subject in holdout
'''


# ============================================================
# Importable helper functions (mirrored from HELPER_CODE above)
# ============================================================

from typing import Set, Tuple


def load_validation_manifest(manifest_path: str = None) -> dict:
    """Load the validation manifest. Auto-detects path if not specified."""
    if manifest_path is None:
        # Look relative to the repo root
        candidates = [
            "validation_manifest/validation_manifest.json",
            os.path.join(os.path.dirname(__file__), "validation_manifest", "validation_manifest.json"),
            os.path.join(os.path.dirname(__file__), "..", "validation_manifest", "validation_manifest.json"),
            os.path.join(os.path.dirname(__file__), "..", "dataset_sim", "validation_manifest", "validation_manifest.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                manifest_path = c
                break
        if manifest_path is None:
            raise FileNotFoundError(
                "Validation manifest not found. Run: python generate_validation_split.py"
            )
    with open(manifest_path) as f:
        return json.load(f)


def get_excluded_windows(manifest: dict) -> Set[Tuple[str, int]]:
    """
    Get set of (filepath, window_idx) tuples that must be excluded from training.

    Usage in your DataLoader:
        manifest = load_validation_manifest()
        excluded = get_excluded_windows(manifest)

        for file_path, window_idx in your_training_data:
            relative = os.path.relpath(file_path, dataset_dir)
            if (relative, window_idx) in excluded:
                continue  # Skip this window — it's validation data
            # ... train on this window
    """
    excluded = set()
    for dataset_key, dataset in manifest.get('datasets', {}).items():
        for filepath, file_info in dataset.get('files', {}).items():
            for window_idx in file_info.get('windows', []):
                excluded.add((filepath, window_idx))
    return excluded


def is_validation_window(manifest: dict, dataset_key: str, filepath: str, window_idx: int) -> bool:
    """Check if a specific window is in the validation set."""
    dataset = manifest.get('datasets', {}).get(dataset_key, {})
    file_info = dataset.get('files', {}).get(filepath, {})
    return window_idx in file_info.get('windows', [])


def is_holdout_subject(manifest: dict, dataset_key: str, subject: str) -> bool:
    """Check if a subject is in the holdout set (never train on ANY of their data)."""
    dataset = manifest.get('datasets', {}).get(dataset_key, {})
    holdout = dataset.get('holdout_subjects', [])
    if holdout == "ALL":
        return True
    return subject in holdout


# ============================================================
# Verification
# ============================================================

def verify_manifest(manifest_path: str, data_dir: str) -> bool:
    """Verify that a manifest matches the local dataset files."""
    print(f"\nVerifying manifest: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_ok = True

    for dataset_key, dataset in manifest.get('datasets', {}).items():
        dataset_dir = os.path.join(data_dir, dataset_key)
        if not os.path.isdir(dataset_dir):
            print(f"  ⚠ Dataset not found: {dataset_key}")
            continue

        files = dataset.get('files', {})
        checked = 0
        mismatched = 0

        for filepath, file_info in files.items():
            full_path = os.path.join(dataset_dir, filepath)
            if not os.path.exists(full_path):
                print(f"  ✗ File missing: {filepath}")
                mismatched += 1
                continue

            actual_hash = get_file_hash(full_path)
            if actual_hash != file_info.get('file_hash', ''):
                print(f"  ✗ Hash mismatch: {filepath}")
                print(f"    Expected: {file_info['file_hash']}")
                print(f"    Actual:   {actual_hash}")
                mismatched += 1
            checked += 1

        if mismatched > 0:
            all_ok = False
            print(f"  {dataset_key}: {mismatched}/{checked + mismatched} files mismatched")
        else:
            print(f"  ✓ {dataset_key}: {checked} files verified")

    if all_ok:
        print("\n✓ Manifest verification PASSED")
    else:
        print("\n✗ Manifest verification FAILED — files may have been modified")

    return all_ok


# ============================================================
# Main
# ============================================================

def main():
    # Auto-detect paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, 'datasets')
    default_output_dir = os.path.join(script_dir, 'validation_manifest')
    default_output = os.path.join(default_output_dir, 'validation_manifest.json')

    parser = argparse.ArgumentParser(
        description='Generate deterministic validation split for LamQuant training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELPER_CODE
    )
    parser.add_argument('--data-dir', type=str, default=default_data_dir,
                        help=f'Root directory containing downloaded datasets (default: {default_data_dir})')
    parser.add_argument('--output', type=str, default=default_output,
                        help=f'Output manifest path (default: {default_output})')
    parser.add_argument('--ratio', type=float, default=0.05,
                        help='Fraction of windows to hold out (default: 0.05 = 5%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--config', type=str, default=None,
                        help='Split config JSON (e.g., official_split_config.json). '
                             'When provided, overrides --seed, --ratio, and dataset '
                             'train/holdout assignments with locked values.')
    parser.add_argument('--verify', type=str, default=None,
                        help='Verify an existing manifest against local data')
    parser.add_argument('--print-helper', action='store_true',
                        help='Print the helper code for training scripts')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    if args.print_helper:
        print(HELPER_CODE)
        return

    if args.verify:
        ok = verify_manifest(args.verify, args.data_dir)
        sys.exit(0 if ok else 1)

    # Load split config if provided
    config_source = "defaults"
    if args.config:
        with open(args.config) as f:
            split_config = json.load(f)
        config_source = os.path.basename(args.config)

        # Override seed and ratio from config
        args.seed = split_config.get('seed', args.seed)
        args.ratio = split_config.get('ratio', args.ratio)

        # Override per-dataset train/holdout subjects
        for ds_key, ds_override in split_config.get('datasets', {}).items():
            if ds_key in DATASETS:
                if 'train_subjects' in ds_override:
                    DATASETS[ds_key]['train_subjects'] = ds_override['train_subjects']
                if 'holdout_subjects' in ds_override:
                    DATASETS[ds_key]['holdout_subjects'] = ds_override['holdout_subjects']
            else:
                # New dataset from config — add a minimal entry so it gets scanned
                DATASETS[ds_key] = {
                    "name": ds_key,
                    "source": "",
                    "license": "",
                    "glob_pattern": "**/*.edf",
                    "train_subjects": ds_override.get('train_subjects', []),
                    "holdout_subjects": ds_override.get('holdout_subjects', []),
                    "fs_native": 256,
                    "channels_expected": 21,
                    "has_seizure_annotations": False,
                }

        # Mark validation-only datasets (never train on them)
        for ds_key in split_config.get('validation_only_datasets', []):
            if ds_key in DATASETS:
                DATASETS[ds_key]['train_subjects'] = []
                DATASETS[ds_key]['holdout_subjects'] = "ALL"

        print(f"  Config: {args.config}")
        if 'description' in split_config:
            print(f"  {split_config['description']}")

    # Generate manifest
    print("=" * 60)
    print("  LamQuant Validation Split Generator")
    print(f"  Config: {config_source}")
    print(f"  Seed: {args.seed}")
    print(f"  Ratio: {args.ratio*100:.0f}%")
    print(f"  Data dir: {args.data_dir}")
    print("=" * 60)

    manifest = generate_manifest(
        args.data_dir,
        ratio=args.ratio,
        seed=args.seed,
        verbose=not args.quiet,
    )

    # Record which config produced this manifest
    manifest['config_source'] = config_source
    if args.config:
        manifest['config_file'] = os.path.basename(args.config)

    # Write manifest
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=False)

    # Print summary
    summary = manifest['summary']
    print("\n" + "=" * 60)
    print("  MANIFEST GENERATED")
    print("=" * 60)
    print(f"  Output:              {args.output}")
    print(f"  Datasets:            {summary['total_datasets']}")
    print(f"  Total subjects:      {summary['total_subjects']}")
    print(f"  Training windows:    {summary['total_train_windows']}")
    print(f"  Validation windows:  {summary['total_validation_windows']}")
    print(f"  Effective ratio:     {summary['effective_validation_ratio']*100:.1f}%")
    print()
    print("  NEXT STEPS:")
    print(f"  1. git add {args.output}")
    print(f"  2. git commit -m 'Add validation manifest (seed={args.seed}, ratio={args.ratio})'")
    print(f"  3. In your training script, call:")
    print(f"     manifest = load_validation_manifest('{args.output}')")
    print(f"     excluded = get_excluded_windows(manifest)")
    print(f"     # Skip excluded windows in your DataLoader")
    print("=" * 60)


if __name__ == '__main__':
    main()
