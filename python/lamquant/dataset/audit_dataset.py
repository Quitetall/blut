#!/usr/bin/env python3
"""
LamQuant — Dataset Distribution Audit
======================================
Scans all .npz files in the Q31 events directory and reports:
  - Total samples, total hours
  - Seizure vs interictal ratio (per dataset and overall)
  - Per-patient breakdown
  - Warnings for imbalanced data

Usage:
  python audit_dataset.py --dir ./q31_events
"""
import os
import glob
import argparse
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Audit LamQuant Q31 dataset distribution")
    parser.add_argument("--dir", required=True, help="Directory containing *_q31.npz files")
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.dir, "*_q31.npz")))
    if not npz_files:
        print(f"[!] No *_q31.npz files found in {args.dir}")
        return

    print(f"[*] Scanning {len(npz_files)} files in {args.dir}\n")

    # Accumulators
    dataset_stats = defaultdict(lambda: {
        'files': 0, 'total_samples': 0, 'seizure_samples': 0,
        'patients': set(), 'min_len': float('inf'), 'max_len': 0,
    })
    patient_stats = defaultdict(lambda: {
        'files': 0, 'total_samples': 0, 'seizure_samples': 0,
    })
    total_files = 0
    total_samples = 0
    total_seizure = 0
    bad_files = []

    for fpath in npz_files:
        fname = os.path.basename(fpath)
        try:
            with np.load(fpath, allow_pickle=True) as f:
                data = f['data']
                mask = f['seizure_mask']
                dataset = str(f.get('dataset', 'unknown'))
                sr = float(f.get('sample_rate', 250.0))
        except Exception as e:
            bad_files.append((fname, str(e)))
            continue

        # Validate shape
        if data.shape[0] != 21:
            bad_files.append((fname, f"wrong channels: {data.shape[0]}"))
            continue
        if data.dtype != np.int32:
            bad_files.append((fname, f"wrong dtype: {data.dtype}"))
            continue

        T = data.shape[1]
        seizure_samps = int(np.sum(mask > 0.5))

        # Extract patient ID from filename heuristics
        # chbmit_chb01_03_q31.npz → chb01
        # siena_PN00_q31.npz → PN00
        parts = fname.replace('_q31.npz', '').split('_')
        if len(parts) >= 2:
            patient = f"{parts[0]}_{parts[1]}"
        else:
            patient = parts[0]

        total_files += 1
        total_samples += T
        total_seizure += seizure_samps

        ds = dataset_stats[dataset]
        ds['files'] += 1
        ds['total_samples'] += T
        ds['seizure_samples'] += seizure_samps
        ds['patients'].add(patient)
        ds['min_len'] = min(ds['min_len'], T)
        ds['max_len'] = max(ds['max_len'], T)

        ps = patient_stats[patient]
        ps['files'] += 1
        ps['total_samples'] += T
        ps['seizure_samples'] += seizure_samps

    # ======================== REPORT ========================
    sr = 250.0  # Assumed
    total_hours = total_samples / sr / 3600
    seizure_hours = total_seizure / sr / 3600
    interictal_hours = (total_samples - total_seizure) / sr / 3600
    seizure_pct = (total_seizure / total_samples * 100) if total_samples > 0 else 0

    print(f"{'='*70}")
    print(f" DATASET DISTRIBUTION REPORT")
    print(f"{'='*70}")
    print(f"  Total files:    {total_files}")
    print(f"  Total hours:    {total_hours:.1f}h")
    print(f"  Interictal:     {interictal_hours:.1f}h ({100-seizure_pct:.2f}%)")
    print(f"  Seizure:        {seizure_hours:.2f}h ({seizure_pct:.2f}%)")

    # Warnings
    print(f"\n  DISTRIBUTION CHECKS:")
    if seizure_pct > 10:
        print(f"  [WARN] Seizure data is {seizure_pct:.1f}% — model may overfit to seizure patterns")
    elif seizure_pct > 5:
        print(f"  [WARN] Seizure data is {seizure_pct:.1f}% — slightly high, consider adding more interictal")
    elif seizure_pct < 0.01:
        print(f"  [WARN] Seizure data is {seizure_pct:.4f}% — very low, seizure augmentation may be needed")
    else:
        print(f"  [OK]   Seizure fraction {seizure_pct:.2f}% is in healthy range (0.1-5%)")

    if total_hours < 10:
        print(f"  [WARN] Only {total_hours:.1f}h of data — more data will improve generalization")
    elif total_hours < 50:
        print(f"  [OK]   {total_hours:.1f}h is reasonable for initial training")
    else:
        print(f"  [OK]   {total_hours:.1f}h is a strong dataset")

    n_patients = len(patient_stats)
    if n_patients < 10:
        print(f"  [WARN] Only {n_patients} patients — cross-patient generalization may suffer")
    else:
        print(f"  [OK]   {n_patients} patients provides good diversity")

    # Per-dataset breakdown
    print(f"\n{'='*70}")
    print(f" PER-DATASET BREAKDOWN")
    print(f"{'='*70}")
    print(f"  {'Dataset':<15} {'Files':>6} {'Hours':>8} {'Patients':>9} "
          f"{'Seizure%':>9} {'Min/Max len':>15}")
    print(f"  {'-'*65}")
    for ds_name in sorted(dataset_stats.keys()):
        ds = dataset_stats[ds_name]
        hours = ds['total_samples'] / sr / 3600
        spct = (ds['seizure_samples'] / ds['total_samples'] * 100) if ds['total_samples'] > 0 else 0
        min_sec = ds['min_len'] / sr
        max_sec = ds['max_len'] / sr
        print(f"  {ds_name:<15} {ds['files']:>6} {hours:>7.1f}h {len(ds['patients']):>9} "
              f"{spct:>8.2f}% {min_sec:>6.0f}s/{max_sec:<6.0f}s")

    # Per-patient (top 20 by hours, or all if < 20)
    print(f"\n{'='*70}")
    print(f" PER-PATIENT (sorted by recording hours)")
    print(f"{'='*70}")
    sorted_patients = sorted(patient_stats.items(),
                              key=lambda x: x[1]['total_samples'], reverse=True)
    print(f"  {'Patient':<25} {'Files':>6} {'Hours':>8} {'Seizure%':>9}")
    print(f"  {'-'*52}")
    for patient, ps in sorted_patients[:30]:
        hours = ps['total_samples'] / sr / 3600
        spct = (ps['seizure_samples'] / ps['total_samples'] * 100) if ps['total_samples'] > 0 else 0
        flag = " ⚠" if spct > 20 else ""
        print(f"  {patient:<25} {ps['files']:>6} {hours:>7.1f}h {spct:>8.2f}%{flag}")
    if len(sorted_patients) > 30:
        print(f"  ... and {len(sorted_patients) - 30} more patients")

    # Bad files
    if bad_files:
        print(f"\n{'='*70}")
        print(f" BAD FILES ({len(bad_files)})")
        print(f"{'='*70}")
        for fname, reason in bad_files[:20]:
            print(f"  {fname}: {reason}")

    # Training window count (10s windows at 250Hz = 2500 samples)
    window_size = 2500
    n_windows = total_samples // window_size
    print(f"\n{'='*70}")
    print(f" TRAINING BUDGET")
    print(f"{'='*70}")
    print(f"  Window size: {window_size} samples ({window_size/sr:.0f}s)")
    print(f"  Total windows: {n_windows:,}")
    print(f"  Current cache: check q31_cache_v1.pt size")


if __name__ == "__main__":
    main()
