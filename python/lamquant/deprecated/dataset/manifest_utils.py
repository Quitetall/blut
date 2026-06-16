"""
Manifest-based Data Split Utilities
====================================
Loads the official_split_config.json and validation_manifest.json to enforce
the canonical training/validation split across all datasets.

This ensures that:
1. Training never sees validation subject data
2. Training never sees validation window data (marked in manifest)
3. All training scripts use the same split, reproducing results
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_official_config(config_path: str = None) -> dict:
    """Load the official split configuration."""
    if config_path is None:
        # Auto-detect relative to this file
        script_dir = Path(__file__).parent
        config_path = script_dir / "official_split_config.json"

    with open(config_path) as f:
        return json.load(f)


def load_validation_manifest(manifest_path: str = None) -> dict:
    """Load the validation manifest."""
    if manifest_path is None:
        # Auto-detect relative to this file
        script_dir = Path(__file__).parent
        manifest_path = script_dir / "validation_manifest" / "validation_manifest.json"

    with open(manifest_path) as f:
        return json.load(f)


def get_training_files(
    npz_files: List[str],
    official_config: dict = None,
    manifest: dict = None,
) -> List[str]:
    """
    Filter NPZ files to only include training files.

    Uses official_split_config.json to determine which subjects are training.
    Files from holdout subjects are entirely excluded.

    Args:
        npz_files: All NPZ file paths
        official_config: official_split_config.json (auto-loaded if None)
        manifest: validation_manifest.json (auto-loaded if None)

    Returns:
        List of NPZ files that belong to training subjects
    """
    if official_config is None:
        official_config = load_official_config()
    if manifest is None:
        manifest = load_validation_manifest()

    # Collect all holdout subject IDs from official config
    holdout_subjects = set()
    for ds_key, ds_config in official_config.get('datasets', {}).items():
        h = ds_config.get('holdout_subjects', [])
        if h == "ALL":
            # Validation-only dataset — all subjects are holdout
            ds_manifest = manifest.get('datasets', {}).get(ds_key, {})
            for fpath, finfo in ds_manifest.get('files', {}).items():
                holdout_subjects.add(finfo.get('subject', ''))
        elif isinstance(h, list):
            holdout_subjects.update(h)

    # Filter files
    training_files = []
    for f in npz_files:
        basename = os.path.basename(f)
        # Extract patient ID: "chbmit_chb01_01_q31.npz" -> "chb01"
        parts = basename.replace('_q31.npz', '').split('_')
        patient_id = parts[1] if len(parts) > 1 else parts[0]

        # Only include files from training subjects
        if patient_id not in holdout_subjects:
            training_files.append(f)

    return training_files


def get_validation_files(
    npz_files: List[str],
    official_config: dict = None,
    manifest: dict = None,
) -> List[str]:
    """
    Filter NPZ files to only include validation files.

    Files from holdout subjects are entirely validation.

    Args:
        npz_files: All NPZ file paths
        official_config: official_split_config.json (auto-loaded if None)
        manifest: validation_manifest.json (auto-loaded if None)

    Returns:
        List of NPZ files that belong to validation subjects
    """
    if official_config is None:
        official_config = load_official_config()
    if manifest is None:
        manifest = load_validation_manifest()

    # Collect all holdout subject IDs
    holdout_subjects = set()
    for ds_key, ds_config in official_config.get('datasets', {}).items():
        h = ds_config.get('holdout_subjects', [])
        if h == "ALL":
            ds_manifest = manifest.get('datasets', {}).get(ds_key, {})
            for fpath, finfo in ds_manifest.get('files', {}).items():
                holdout_subjects.add(finfo.get('subject', ''))
        elif isinstance(h, list):
            holdout_subjects.update(h)

    # Filter files
    validation_files = []
    for f in npz_files:
        basename = os.path.basename(f)
        parts = basename.replace('_q31.npz', '').split('_')
        patient_id = parts[1] if len(parts) > 1 else parts[0]

        # Only include files from validation subjects
        if patient_id in holdout_subjects:
            validation_files.append(f)

    return validation_files


def get_excluded_windows(manifest: dict = None) -> Set[Tuple[str, int]]:
    """
    Get all (filepath, window_idx) tuples that must be excluded from training.

    These are windows marked as validation in the manifest, even if their
    file is in a training subject.

    Args:
        manifest: validation_manifest.json (auto-loaded if None)

    Returns:
        Set of (filepath, window_idx) tuples to exclude from training
    """
    if manifest is None:
        manifest = load_validation_manifest()

    excluded = set()
    for dataset_key, dataset in manifest.get('datasets', {}).items():
        for filepath, file_info in dataset.get('files', {}).items():
            for window_idx in file_info.get('windows', []):
                excluded.add((filepath, window_idx))

    return excluded


def is_validation_window(
    filepath: str,
    window_idx: int,
    manifest: dict = None,
) -> bool:
    """
    Check if a specific window is validation data.

    Args:
        filepath: Relative path as stored in manifest (e.g., "chb01/chb01_03.edf")
        window_idx: Window index
        manifest: validation_manifest.json (auto-loaded if None)

    Returns:
        True if this window is validation, False if training
    """
    if manifest is None:
        manifest = load_validation_manifest()

    # Search all datasets for the file
    for dataset_key, dataset in manifest.get('datasets', {}).items():
        file_info = dataset.get('files', {}).get(filepath, {})
        if filepath in dataset.get('files', {}):
            return window_idx in file_info.get('windows', [])

    # Not found in manifest — assume training
    return False


def is_holdout_subject(
    subject_id: str,
    official_config: dict = None,
    manifest: dict = None,
) -> bool:
    """
    Check if a subject is entirely held out from training.

    Args:
        subject_id: Subject identifier (e.g., "chb01", "S001")
        official_config: official_split_config.json (auto-loaded if None)
        manifest: validation_manifest.json (auto-loaded if None)

    Returns:
        True if this subject should never appear in training data
    """
    if official_config is None:
        official_config = load_official_config()
    if manifest is None:
        manifest = load_validation_manifest()

    # Check all datasets for this subject
    for ds_key, ds_config in official_config.get('datasets', {}).items():
        h = ds_config.get('holdout_subjects', [])
        if h == "ALL":
            # Validation-only dataset — check if subject appears in manifest
            ds_manifest = manifest.get('datasets', {}).get(ds_key, {})
            for fpath, finfo in ds_manifest.get('files', {}).items():
                if finfo.get('subject', '') == subject_id:
                    return True
        elif isinstance(h, list) and subject_id in h:
            return True

    return False


def print_split_summary(official_config: dict = None, manifest: dict = None) -> None:
    """Pretty-print the canonical split configuration."""
    if official_config is None:
        official_config = load_official_config()
    if manifest is None:
        manifest = load_validation_manifest()

    print("\n" + "=" * 70)
    print("CANONICAL TRAINING/VALIDATION SPLIT")
    print("=" * 70)

    for ds_key, ds_config in official_config.get('datasets', {}).items():
        ds_manifest = manifest.get('datasets', {}).get(ds_key, {})

        train_s = ds_config.get('train_subjects', [])
        holdout_s = ds_config.get('holdout_subjects', [])

        total_w = ds_manifest.get('total_windows', 0)
        val_w = ds_manifest.get('validation_windows', 0)
        train_w = total_w - val_w

        print(f"\n{ds_key}:")
        if holdout_s == "ALL":
            print(f"  Status: Validation-only dataset")
        elif isinstance(train_s, list) and len(train_s) > 0:
            print(f"  Train subjects: {len(train_s)} ({train_s[0]}...)")
        if isinstance(holdout_s, list) and len(holdout_s) > 0:
            print(f"  Holdout subjects: {holdout_s}")

        print(f"  Windows: {train_w:,} training, {val_w:,} validation")

    total_train = sum(
        manifest.get('datasets', {}).get(ds_key, {}).get('total_windows', 0) -
        manifest.get('datasets', {}).get(ds_key, {}).get('validation_windows', 0)
        for ds_key in manifest.get('datasets', {})
    )
    total_val = sum(
        manifest.get('datasets', {}).get(ds_key, {}).get('validation_windows', 0)
        for ds_key in manifest.get('datasets', {})
    )

    print(f"\n{'TOTAL':20} {total_train:,} training, {total_val:,} validation")
    ratio = total_val / (total_train + total_val) * 100
    print(f"{'':20} {ratio:.1f}% validation split")
    print("=" * 70 + "\n")
