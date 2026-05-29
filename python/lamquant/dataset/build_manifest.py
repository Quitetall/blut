#!/usr/bin/env python3
"""build_manifest.py — produce manifest_v3.json from preprocessed q31_events/.

Run ONCE after any change to the preprocessed dataset (new TUEP ingestion,
re-run of edf_to_events.py, etc.). Produces a typed `DatasetManifest`
that every training script reads. Replaces the old dual-source split
logic (official_split_config.json + validation_manifest.json) with a
single canonical artefact.

The build steps:

  1. Scan every NPZ in q31_events/ — get its window count (header read).
  2. For each NPZ, determine canonical (dataset, patient_id):
       - If the v2 manifest covers it (basename match) → use v2's
         dataset_key + subject directly. This is authoritative because
         v2 was generated from the original EDF directory layout, which
         knows the difference between tuh_seizure / tuh_artifact /
         tuh_epilepsy that the flattened NPZ filenames lost.
       - Else → use parse_npz_filename() as a deterministic fallback.
         This covers tuh_epilepsy (TUEP, never had a v2 entry generated)
         plus any future datasets added to q31_events/ before the
         maintainer extends v2.
  3. Per dataset, decide which patients are held out:
       - validation_only datasets (siena/eegmmidb/mental) → every
         patient is implicit HOLDOUT.
       - chbmit → preserve the established chb21-24 holdouts (matches
         every published baseline against this dataset).
       - tuh_seizure / tuh_artifact / tuh_events / tuh_epilepsy → pick
         val_fraction (default 5 %) of unique patients deterministically
         (seeded). Subject-disjoint guarantee: every file from a held-out
         patient is VAL; every other file is TRAIN.
  4. Validate. Save to manifest_v3.json.

Usage:
    python ai_models/dataset_sim/build_manifest.py
    python ai_models/dataset_sim/build_manifest.py --val-fraction 0.05 --seed 42
    python ai_models/dataset_sim/build_manifest.py --dry-run     # print summary, don't write
    python ai_models/dataset_sim/build_manifest.py --output ai_models/dataset_sim/manifest_v3.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _stable_str_hash(s: str) -> int:
    """Deterministic 32-bit hash of a string. Python's built-in hash() is
    randomized per process (PYTHONHASHSEED), which would silently produce
    a different manifest on every run.
    """
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], 'big')

# Repo path setup so we can import from ai_models/.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'lamquant'))

from lamquant.common.data_types import (
    Dataset, Split, VALIDATION_ONLY_DATASETS, CLINICAL_CATEGORIES,
    FileEntry, DatasetEntry, DatasetManifest, MANIFEST_VERSION,
    parse_npz_filename, ParsedFilename,
)


# ============================================================
# Subject holdouts that are NOT auto-computed
# ============================================================
# These take precedence over the random per-dataset selection. Use this
# for datasets where the holdout subjects are established by literature
# (CHB-MIT chb21-24 is the canonical baseline split) or where we want
# to lock specific patients across training runs for comparison.

EXPLICIT_HOLDOUT_PATIENTS: Dict[Dataset, List[str]] = {
    Dataset.CHBMIT: ['chb21', 'chb22', 'chb23', 'chb24'],
}


# ============================================================
# Window count — header-only NPZ read (no decompression)
# ============================================================

def _peek_l3_window_count(npz_path: str) -> int:
    """Read the L3 array shape from an NPZ header. Returns 0 if missing.

    Uses the same byte-level header parse that streaming_dataset.py uses
    (np.load with mmap silently decompresses NPZ archives, OOM-killing
    on 11 K files — see streaming_dataset.peek_npz_data_shape).
    """
    try:
        from numpy.lib.format import (
            read_magic, read_array_header_1_0, read_array_header_2_0,
        )
    except ImportError:
        return 0

    try:
        with zipfile.ZipFile(npz_path) as zf:
            # Prefer 'l3' (the precomputed subband). If absent, fall back
            # to inferring from the raw 'data' shape (T // 2500 windows).
            for member, divisor in (('l3.npy', 1), ('data.npy', 2500)):
                if member not in zf.namelist():
                    continue
                with zf.open(member) as f:
                    version = read_magic(f)
                    if version == (1, 0):
                        shape, _, _ = read_array_header_1_0(f)
                    elif version == (2, 0):
                        shape, _, _ = read_array_header_2_0(f)
                    else:
                        from numpy.lib.format import _read_array_header
                        shape, _, _ = _read_array_header(f, version)
                    if member == 'l3.npy':
                        # L3 array shape is [n_windows, 21, 313]
                        return int(shape[0]) if shape else 0
                    # data array shape is [21, T] — derive windows
                    return int(shape[1] // divisor) if len(shape) >= 2 else 0
            return 0
    except Exception:
        return 0


# ============================================================
# Clinical category assignment — runs once per file at build time
# ============================================================

# Datasets that are inherently pediatric.
_PEDIATRIC_DATASETS = {Dataset.CHBMIT}  # HBN added when dataset exists

# TUEV event types that are spike/sharp-wave.
_SPIKE_EVENT_TYPES = frozenset({'spsw', 'gped', 'pled'})

# TUEV event types that are artifact.
_ARTIFACT_EVENT_TYPES = frozenset({'eyem', 'eyeb', 'musc', 'chew', 'shiv', 'elpp', 'elec'})


def _peek_has_seizure(npz_path: str) -> bool:
    """Check if an NPZ has any seizure annotations (seizure_mask > 0)."""
    try:
        with zipfile.ZipFile(npz_path) as zf:
            if 'seizure_mask.npy' not in zf.namelist():
                return False
            import numpy as np
            with zf.open('seizure_mask.npy') as f:
                from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
                version = read_magic(f)
                if version == (1, 0):
                    shape, _, dtype = read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, dtype = read_array_header_2_0(f)
                else:
                    return False
                # If the mask has any nonzero, there's seizure. Read a small
                # chunk to check without decompressing the whole thing.
                data = np.frombuffer(f.read(), dtype=dtype)
                return bool(np.any(data != 0))
    except Exception:
        return False


def assign_clinical_category(entry: FileEntry) -> str:
    """Assign primary clinical category to a FileEntry.

    Priority order (rarest first):
      seizure > spike_event > artifact > epilepsy_patient > sleep > pediatric > normal

    Called during manifest build; result stored in entry.clinical_category
    so the training sampler never needs to re-classify at runtime.
    """
    # 1. Seizure — from TUSZ seizure_mask
    if entry.has_seizure:
        return 'seizure'

    # 2. Spike/sharp-wave — from TUEV event annotations
    if entry.event_type in _SPIKE_EVENT_TYPES:
        return 'spike_event'

    # 3. Artifact — from TUEV/TUAR event annotations
    if entry.event_type in _ARTIFACT_EVENT_TYPES:
        return 'artifact'

    # 4. Epilepsy / seizure-corpus patient — TUEP or TUSZ cohort
    #    Even recordings without a labeled seizure are from patients
    #    referred for seizure evaluation → clinically relevant.
    if entry.dataset in (Dataset.TUH_EPILEPSY, Dataset.TUH_SEIZURE):
        return 'epilepsy_patient'

    # 5. Sleep — sleep datasets (future: sleep-edf, HBN sleep)
    # Currently no sleep dataset in manifest; placeholder for TUEG overlay
    ds_name = entry.dataset.value.lower()
    if 'sleep' in ds_name:
        return 'sleep'

    # 6. Pediatric — CHB-MIT, HBN
    if entry.dataset in _PEDIATRIC_DATASETS:
        return 'pediatric'

    # 7. Normal awake baseline
    return 'normal'


# ============================================================
# v2 manifest — the canonical (dataset, subject) lookup
# ============================================================

def _load_v2_basename_index(
    v2_path: Path,
) -> Dict[str, Tuple[Dataset, str]]:
    """Return {edf_basename_without_ext → (Dataset, subject)} from v2 manifest.

    Used as the primary lookup for files that were covered by the
    canonical v2 split. Falls back to parse_npz_filename() for files
    not present (typically tuh_epilepsy / TUEP).
    """
    if not v2_path.exists():
        print(f'[!] v2 manifest not found at {v2_path}; using parser fallback for everything')
        return {}

    import json
    with open(v2_path) as f:
        v2 = json.load(f)

    # Maps v2's dataset key to our Dataset enum.
    KEY_TO_DATASET = {
        'chbmit': Dataset.CHBMIT,
        'tuh_seizure': Dataset.TUH_SEIZURE,
        'tuh_artifact': Dataset.TUH_ARTIFACT,
        'tuh_events': Dataset.TUH_EVENTS,
        'tuh_epilepsy': Dataset.TUH_EPILEPSY,
        'siena': Dataset.SIENA,
        'eegmmidb': Dataset.EEGMMIDB,
        'mental_arithmetic': Dataset.MENTAL_ARITHMETIC,
    }

    index: Dict[str, Tuple[Dataset, str]] = {}
    for key, ds in v2.get('datasets', {}).items():
        ds_enum = KEY_TO_DATASET.get(key)
        if ds_enum is None:
            print(f'[!] v2 manifest references unknown dataset {key!r}; skipping')
            continue
        for fpath, finfo in ds.get('files', {}).items():
            bn = os.path.basename(fpath)
            if bn.endswith('.edf'):
                bn = bn[:-len('.edf')]
            subj = finfo.get('subject', '')
            if subj:
                index[bn] = (ds_enum, subj)
    return index


def _resolve_npz(
    npz_path: str,
    v2_index: Dict[str, Tuple[Dataset, str]],
    edf_corpus: Optional[Dict[str, str]] = None,
) -> ParsedFilename:
    """Determine the canonical (dataset, patient_id, session_id, ...) for an NPZ.

    Three-level lookup:
      1. v2 manifest (authoritative for files it covers)
      2. EDF directory scan (ground-truth corpus for files v2 missed)
      3. Parser fallback (deterministic but defaults tuh_* to TUH_SEIZURE)

    The EDF lookup (level 2) fixes the 6,206 files that v2 never indexed
    and the parser would incorrectly bucket as TUH_SEIZURE.
    """
    bn = os.path.basename(npz_path)
    if bn.endswith('_q31.npz'):
        stem = bn[:-len('_q31.npz')]
    elif bn.endswith('.npz'):
        stem = bn[:-len('.npz')]
    else:
        stem = bn

    # Strip the dataset prefix to match v2's basename keys (which are
    # original EDF basenames without dataset prefix).
    for prefix in ('chbmit_', 'tuh_', 'siena_', 'eegmmidb_',
                   'mental_arithmetic_'):
        if stem.startswith(prefix):
            edf_bn = stem[len(prefix):]
            break
    else:
        edf_bn = stem

    parsed = parse_npz_filename(bn)
    v2_hit = v2_index.get(edf_bn)
    if v2_hit:
        ds, subj = v2_hit
        return ParsedFilename(
            dataset=ds,
            patient_id=subj,
            session_id=parsed.session_id,
            event_type=parsed.event_type,
            segment=parsed.segment,
        )
    # Level 2: EDF directory ground truth (fixes 6,206 unmatched tuh_* files)
    if edf_corpus and edf_bn in edf_corpus:
        corpus_name = edf_corpus[edf_bn]
        ds = _CORPUS_TO_DATASET.get(corpus_name, parsed.dataset)
        return ParsedFilename(
            dataset=ds,
            patient_id=parsed.patient_id,
            session_id=parsed.session_id,
            event_type=parsed.event_type,
            segment=parsed.segment,
        )
    return parsed


# ============================================================
# Holdout selection
# ============================================================

def _select_holdout_patients(
    patients: List[str],
    val_fraction: float,
    seed: int,
    explicit: Optional[List[str]] = None,
) -> List[str]:
    """Return the subset of patients to hold out for validation.

    If `explicit` is given and non-empty, those patients are the holdouts
    verbatim (filtered to those that actually exist in `patients`).
    Otherwise pick val_fraction of patients deterministically.
    """
    patients_sorted = sorted(set(patients))
    if explicit:
        actual = set(patients_sorted)
        return sorted(p for p in explicit if p in actual)
    if not patients_sorted:
        return []
    n_holdout = max(1, round(len(patients_sorted) * val_fraction))
    rng = random.Random(seed)
    shuffled = list(patients_sorted)
    rng.shuffle(shuffled)
    return sorted(shuffled[:n_holdout])


# ============================================================
# Build
# ============================================================

def _build_edf_corpus_index(datasets_dir: Path) -> Dict[str, str]:
    """Scan raw EDF directories to build {basename → corpus_name} mapping.

    Used as a SECOND lookup for NPZs not covered by the v2 manifest.
    The EDF directory structure is the authoritative source of which
    corpus a recording belongs to.
    """
    index: Dict[str, str] = {}
    for corpus in ('tuh_seizure', 'tuh_artifact', 'tuh_events'):
        corpus_dir = datasets_dir / corpus
        if not corpus_dir.exists():
            continue
        for edf in corpus_dir.rglob('*.edf'):
            bn = edf.stem  # basename without .edf
            index.setdefault(bn, corpus)  # first-seen wins for collisions
        for edf in corpus_dir.rglob('*.EDF'):
            bn = edf.stem
            index.setdefault(bn, corpus)
    return index


# Map corpus directory names to Dataset enums.
_CORPUS_TO_DATASET = {
    'tuh_seizure': Dataset.TUH_SEIZURE,
    'tuh_artifact': Dataset.TUH_ARTIFACT,
    'tuh_events': Dataset.TUH_EVENTS,
    'tuh_epilepsy': Dataset.TUH_EPILEPSY,
}


def build_manifest(
    q31_dir: Path,
    *,
    val_fraction: float = 0.05,
    seed: int = 42,
    v2_path: Optional[Path] = None,
) -> DatasetManifest:
    """Construct a DatasetManifest from a directory of preprocessed NPZs."""
    npz_files = sorted(glob.glob(str(q31_dir / '*.npz')))
    if not npz_files:
        raise RuntimeError(f'no NPZ files found in {q31_dir}')

    if v2_path is None:
        v2_path = q31_dir.parent / 'validation_manifest' / 'validation_manifest.json'
    v2_index = _load_v2_basename_index(v2_path)

    # Build ground-truth EDF→corpus mapping as a second lookup for
    # NPZs not in the v2 manifest.
    datasets_dir = q31_dir.parent / 'datasets'
    edf_corpus = _build_edf_corpus_index(datasets_dir)
    if edf_corpus:
        print(f'[*] EDF corpus index: {len(edf_corpus):,} basenames from '
              f'{datasets_dir}')

    # Pass 1: parse + window-count every file. Group by dataset.
    print(f'[*] Scanning {len(npz_files):,} NPZs (parser + header read)...')
    by_dataset: Dict[Dataset, List[FileEntry]] = {}
    parser_errors = 0
    zero_window_files = 0

    for f in npz_files:
        try:
            parsed = _resolve_npz(f, v2_index, edf_corpus=edf_corpus)
        except ValueError as e:
            parser_errors += 1
            print(f'    parser error: {os.path.basename(f)}: {e}')
            continue

        n_win = _peek_l3_window_count(f)
        if n_win <= 0:
            zero_window_files += 1
            # Skip — no L3 means the file was never finalised by precompute_l3.
            continue

        has_sz = _peek_has_seizure(f)

        entry = FileEntry(
            path=str(Path(f).resolve()),
            dataset=parsed.dataset,
            patient_id=parsed.patient_id,
            session_id=parsed.session_id,
            split=Split.TRAIN,             # placeholder; set per-dataset below
            n_windows=n_win,
            event_type=parsed.event_type,
            segment=parsed.segment,
            has_seizure=has_sz,
        )
        entry.clinical_category = assign_clinical_category(entry)
        by_dataset.setdefault(parsed.dataset, []).append(entry)

    if parser_errors:
        print(f'[!] {parser_errors} files failed to parse — they will not appear in manifest_v3')
    if zero_window_files:
        print(f'[!] {zero_window_files} files skipped (no l3 array — re-run precompute_l3)')

    # Pass 2: UNIFIED cross-corpus holdout selection.
    #
    # TUH subcorpora SHARE patients — the same patient can appear in
    # tuh_seizure, tuh_artifact, tuh_events, and tuh_epilepsy. Selecting
    # holdouts independently per corpus causes leakage: patient X held
    # out from tuh_artifact but training in tuh_seizure.
    #
    # Fix: collect ALL TUH patients into one pool, select 5% as holdout
    # ONCE, apply that set to every TUH subcorpus. Non-TUH datasets
    # (chbmit, siena, etc.) keep independent holdouts.
    TUH_DATASETS = {Dataset.TUH_SEIZURE, Dataset.TUH_ARTIFACT,
                    Dataset.TUH_EVENTS, Dataset.TUH_EPILEPSY, Dataset.TUEG}
    all_tuh_patients = set()
    for ds_enum in TUH_DATASETS:
        if ds_enum in by_dataset:
            all_tuh_patients.update(f.patient_id for f in by_dataset[ds_enum])
    unified_tuh_holdouts = set(_select_holdout_patients(
        sorted(all_tuh_patients),
        val_fraction=val_fraction,
        seed=seed + _stable_str_hash('tuh_unified') % 10000,
    )) if all_tuh_patients else set()
    print(f'[*] Unified TUH holdout: {len(unified_tuh_holdouts)} of '
          f'{len(all_tuh_patients)} patients across '
          f'{sum(1 for d in TUH_DATASETS if d in by_dataset)} subcorpora')

    manifest = DatasetManifest(
        version=MANIFEST_VERSION,
        seed=seed,
        val_fraction=val_fraction,
    )

    for ds_enum, files in sorted(by_dataset.items(), key=lambda x: x[0].value):
        is_val_only = ds_enum in VALIDATION_ONLY_DATASETS
        patients = sorted({f.patient_id for f in files})

        if is_val_only:
            holdouts = patients
            for f in files:
                f.split = Split.HOLDOUT
        elif ds_enum in TUH_DATASETS:
            # Use the UNIFIED TUH holdout — same patients held out
            # from every TUH subcorpus. No per-corpus independence.
            holdouts = sorted(unified_tuh_holdouts & set(patients))
            holdout_set = unified_tuh_holdouts
            for f in files:
                f.split = Split.VAL if f.patient_id in holdout_set else Split.TRAIN
        else:
            holdouts = _select_holdout_patients(
                patients,
                val_fraction=val_fraction,
                seed=seed + _stable_str_hash(ds_enum.value) % 10000,
                explicit=EXPLICIT_HOLDOUT_PATIENTS.get(ds_enum),
            )
            holdout_set = set(holdouts)
            for f in files:
                f.split = Split.VAL if f.patient_id in holdout_set else Split.TRAIN

        entry = DatasetEntry(
            name=ds_enum.value,
            holdout_patients=holdouts,
            validation_only=is_val_only,
            files=files,
        )
        entry.recompute_aggregates()
        manifest.datasets[ds_enum.value] = entry

    manifest.recompute_aggregates()

    # Print clinical category distribution for the training set.
    from collections import Counter
    cat_counts = Counter()
    cat_windows = Counter()
    for f in manifest.all_files():
        if f.split == Split.TRAIN:
            cat_counts[f.clinical_category] += 1
            cat_windows[f.clinical_category] += f.n_windows
    total_w = sum(cat_windows.values())
    if cat_counts:
        print(f'\n[*] Clinical categories (train set, {sum(cat_counts.values()):,} files, '
              f'{total_w:,} windows):')
        for cat in CLINICAL_CATEGORIES:
            n = cat_counts.get(cat, 0)
            w = cat_windows.get(cat, 0)
            pct = 100 * w / max(total_w, 1)
            print(f'    {cat:20} {n:>6,} files  {w:>9,} windows  ({pct:5.1f}%)')

    # Validate before returning. validate() catches subject-overlap,
    # validation_only-in-train, window-count drift.
    issues = manifest.validate()
    if issues:
        raise RuntimeError(
            'manifest failed validation:\n  - ' + '\n  - '.join(issues)
        )
    return manifest


# ============================================================
# CLI
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        prog='build_manifest',
        description='Build typed DatasetManifest (manifest_v3.json) from q31_events.',
    )
    parser.add_argument('--q31-dir', type=Path,
                        default=_REPO / 'lamquant' / 'dataset' / 'q31_events',
                        help='Directory containing precomputed *_q31.npz files')
    parser.add_argument('--output', type=Path,
                        default=_REPO / 'lamquant' / 'dataset' / 'manifest_v3.json',
                        help='Output JSON path (default: ai_models/dataset_sim/manifest_v3.json)')
    parser.add_argument('--v2', type=Path, default=None,
                        help='v2 validation_manifest.json (default: alongside q31_dir)')
    parser.add_argument('--val-fraction', type=float, default=0.05,
                        help='Per-dataset val patient fraction (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for holdout patient selection (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print summary, do not write the JSON file')
    args = parser.parse_args()

    manifest = build_manifest(
        q31_dir=args.q31_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        v2_path=args.v2,
    )

    print()
    print(manifest.summary_str())
    print()

    if args.dry_run:
        print('[*] --dry-run: not writing JSON.')
    else:
        out = manifest.save(args.output)
        print(f'[*] Wrote manifest to: {out}')
        print(f'    Reload it with `DatasetManifest.load({out!r})` — '
              f'validate() runs automatically.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
