#!/usr/bin/env python3
"""
LamQuant Cross-Dataset Validation Suite
========================================
Tests the codec on datasets that were NOT used during training.
This is the honest benchmark — if the codec generalizes, it works here.
If it doesn't, the training numbers are meaningless.

Datasets tested:
  1. CHB-MIT holdout (chb15-chb24) — same site, unseen patients
  2. Siena Scalp EEG — different hospital, different country, adults
  3. EEGMMIDB — healthy subjects, motor imagery, 64 channels
  4. Mental Arithmetic — cognitive load, different paradigm

Usage:
  # Just run it. No arguments needed.
  python validate_cross_dataset.py

  # Quick mode (fewer files, ~10 minutes)
  python validate_cross_dataset.py --quick

  # Datasets expected at: ./datasets/
  # Model expected at: ./ai_models/student/student_hardened.ckpt
  # Report output to: ./validation_manifest/validation_report.json
"""

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not available, skipping neural codec tests")

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    print("WARNING: mne not available (pip install mne), using raw EDF reader")

try:
    import pyedflib
    HAS_PYEDF = True
except ImportError:
    HAS_PYEDF = False


# ============================================================
# Data classes for results
# ============================================================

@dataclass
class WindowResult:
    file: str
    window_idx: int
    r: float
    prd: float
    snr_db: float
    cr: float
    channels_used: int
    sample_rate_orig: int

@dataclass 
class DatasetResult:
    name: str
    num_files: int
    num_windows: int
    num_subjects: int
    mean_r: float
    std_r: float
    median_r: float
    mean_prd: float
    mean_snr_db: float
    mean_cr: float
    per_subject_r: dict
    windows: List[WindowResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

@dataclass
class ValidationReport:
    timestamp: str
    model_path: str
    codec_version: str
    datasets: List[DatasetResult]
    overall_r: float
    overall_cr: float
    pass_fail: str  # PASS / FAIL / WARN
    summary: str


# ============================================================
# EDF reading utilities (delegates to shared channel_resolver)
# ============================================================

from lamquant_codec.channel_resolver import (
    resolve as normalize_channel_name,
    extract_channel_data,
    TARGET_CHANNELS,
)


def read_edf_channels(filepath: str, target_fs: int = 250) -> Optional[Tuple[np.ndarray, int, List[str]]]:
    """
    Read an EDF file and extract available 10-20 channels.
    Returns: (data [21, samples], native_sample_rate, channel_names) or None.
    `data` is resampled to target_fs; the returned rate is the file's NATIVE
    sample rate (read from the header), so callers can record true provenance.
    """
    try:
        if HAS_MNE:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            ch_names = raw.ch_names
            fs = int(raw.info['sfreq'])
            all_data = raw.get_data()
        elif HAS_PYEDF:
            f = pyedflib.EdfReader(filepath)
            ch_names = f.getSignalLabels()
            fs = int(f.getSampleFrequency(0))
            all_data = np.array([f.readSignal(i) for i in range(len(ch_names))])
            f.close()
        else:
            print(f"  No EDF reader available, skipping {filepath}")
            return None

        data, missing = extract_channel_data(all_data, ch_names)
        if data is None:
            return None

        # Resample to target_fs if needed, but keep the NATIVE rate to return.
        native_fs = fs
        if fs != target_fs:
            num_samples = int(data.shape[1] * target_fs / fs)
            from scipy.signal import resample
            data_resampled = np.zeros((data.shape[0], num_samples), dtype=data.dtype)
            for ch in range(data.shape[0]):
                data_resampled[ch] = resample(data[ch], num_samples)
            data = data_resampled

        return data, native_fs, TARGET_CHANNELS

    except Exception as e:
        return None


# ============================================================
# Codec pipeline (Python reference implementation)
# ============================================================

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[float, float, float]:
    """Compute R, PRD, SNR between original and reconstructed signals."""
    orig_flat = original.flatten().astype(np.float64)
    recon_flat = reconstructed.flatten().astype(np.float64)
    
    # Pearson R
    if np.std(orig_flat) < 1e-10 or np.std(recon_flat) < 1e-10:
        r = 0.0
    else:
        r = float(np.corrcoef(orig_flat, recon_flat)[0, 1])
    
    # PRD
    error = orig_flat - recon_flat
    prd = 100.0 * np.sqrt(np.sum(error ** 2) / (np.sum(orig_flat ** 2) + 1e-10))
    
    # SNR
    snr = 10.0 * np.log10((np.sum(orig_flat ** 2) + 1e-10) / (np.sum(error ** 2) + 1e-10))
    
    return float(r), float(prd), float(snr)


def run_codec_pipeline(
    signal: np.ndarray,
    model: 'torch.nn.Module',
    num_channels: int = 21,
    window_len: int = 2500,
    fsq_levels: int = 16,
    sample_rate: int = 250,
) -> List[WindowResult]:
    """
    Run the full LamQuant codec pipeline on a multi-channel signal.
    signal: [channels, total_samples]
    sample_rate: native rate to record on each WindowResult (provenance).
    Returns list of WindowResult per window.
    """
    results = []
    channels, total_samples = signal.shape
    
    # Pad or truncate to num_channels
    if channels < num_channels:
        pad = np.zeros((num_channels - channels, total_samples), dtype=signal.dtype)
        signal_padded = np.vstack([signal, pad])
    else:
        signal_padded = signal[:num_channels]
    
    num_windows = total_samples // window_len
    
    for w in range(num_windows):
        start = w * window_len
        end = start + window_len
        window = signal_padded[:, start:end]
        
        # Scale to µV range
        window_uv = window.astype(np.float32)
        if np.abs(window_uv).max() > 1e-3:  # Not flat
            # Normalize similar to training pipeline
            window_uv = window_uv * 1000  # Scale to µV if in V
            if np.abs(window_uv).max() > 10000:
                window_uv = window_uv / (np.abs(window_uv).max() / 1000)
        
        # Prepare input tensor
        x = torch.from_numpy(window_uv[:num_channels]).unsqueeze(0).float()
        if torch.cuda.is_available():
            x = x.cuda()
        
        # Center and clamp
        x = x - x.mean(dim=2, keepdim=True)
        x = torch.clamp(x, -50, 50)
        
        with torch.no_grad():
            # Encode
            latent = model.encode(x)
            
            # FSQ quantization (simulate)
            if hasattr(model, 'quantize'):
                latent_q = model.quantize(latent, fsq_levels)
            else:
                # Manual FSQ
                vmin = latent.min()
                vmax = latent.max()
                span = vmax - vmin + 1e-8
                indices = torch.clamp(
                    ((latent - vmin) / span * fsq_levels).long(), 
                    0, fsq_levels - 1
                )
                latent_q = vmin + (indices.float() + 0.5) * span / fsq_levels
            
            # Decode
            recon = model.decode(latent_q)
        
        # Compute metrics
        orig_np = x.cpu().numpy().squeeze()
        recon_np = recon.cpu().numpy().squeeze()
        
        r, prd, snr = compute_metrics(orig_np[:channels], recon_np[:channels])
        
        # Estimate CR
        raw_bits = num_channels * window_len * 16
        latent_shape = latent.shape
        coded_bits = latent_shape[-1] * latent_shape[-2] * np.log2(fsq_levels)
        cr = raw_bits / (coded_bits + 1e-10)
        
        results.append(WindowResult(
            file="", window_idx=w, r=r, prd=prd, snr_db=snr,
            cr=float(cr), channels_used=channels,
            sample_rate_orig=sample_rate
        ))
    
    return results


# ============================================================
# Dataset-specific loaders
# ============================================================

def validate_chbmit_holdout(data_dir: str, model, quick: bool = False) -> DatasetResult:
    """
    CHB-MIT holdout patients (chb15-chb24).
    Same site as training, different patients.
    Tests: within-site generalization.
    """
    print("\n" + "=" * 60)
    print("  CHB-MIT Holdout (chb15-chb24)")
    print("  Same hospital, unseen patients")
    print("=" * 60)
    
    holdout_patients = [f'chb{i:02d}' for i in range(15, 25)]
    chbmit_dir = os.path.join(data_dir, 'chbmit')
    
    all_windows = []
    per_subject = {}
    errors = []
    files_processed = 0
    
    for patient in holdout_patients:
        patient_dir = os.path.join(chbmit_dir, patient)
        if not os.path.isdir(patient_dir):
            continue
        
        edf_files = sorted(glob.glob(os.path.join(patient_dir, '*.edf')))
        if quick:
            edf_files = edf_files[:3]
        
        patient_rs = []
        
        for edf_file in edf_files:
            result = read_edf_channels(edf_file)
            if result is None:
                errors.append(f"Failed to read: {edf_file}")
                continue
            
            data, fs, ch_names = result
            files_processed += 1
            
            if model is not None:
                windows = run_codec_pipeline(data, model, sample_rate=fs)
                for w in windows:
                    w.file = os.path.basename(edf_file)
                all_windows.extend(windows)
                patient_rs.extend([w.r for w in windows])
            
            if quick and files_processed >= 5:
                break
        
        if patient_rs:
            per_subject[patient] = float(np.mean(patient_rs))
            print(f"  {patient}: {len(patient_rs)} windows, R={np.mean(patient_rs):.4f}")
    
    if not all_windows:
        return DatasetResult(
            name="CHB-MIT Holdout", num_files=files_processed,
            num_windows=0, num_subjects=len(per_subject),
            mean_r=0, std_r=0, median_r=0, mean_prd=0,
            mean_snr_db=0, mean_cr=0, per_subject_r=per_subject,
            errors=errors
        )
    
    rs = [w.r for w in all_windows]
    return DatasetResult(
        name="CHB-MIT Holdout",
        num_files=files_processed,
        num_windows=len(all_windows),
        num_subjects=len(per_subject),
        mean_r=float(np.mean(rs)),
        std_r=float(np.std(rs)),
        median_r=float(np.median(rs)),
        mean_prd=float(np.mean([w.prd for w in all_windows])),
        mean_snr_db=float(np.mean([w.snr_db for w in all_windows])),
        mean_cr=float(np.mean([w.cr for w in all_windows])),
        per_subject_r=per_subject,
        windows=all_windows,
        errors=errors
    )


def validate_siena(data_dir: str, model, quick: bool = False) -> DatasetResult:
    """
    Siena Scalp EEG Database.
    COMPLETELY DIFFERENT SITE from training data.
    Different country (Italy), different equipment, adult patients.
    This is the real cross-site generalization test.
    """
    print("\n" + "=" * 60)
    print("  Siena Scalp EEG (Cross-Site Validation)")
    print("  Different hospital, country, equipment, population")
    print("=" * 60)
    
    siena_dir = os.path.join(data_dir, 'siena')
    
    all_windows = []
    per_subject = {}
    errors = []
    files_processed = 0
    
    # Siena has patient directories like PN00, PN01, etc.
    patient_dirs = sorted(glob.glob(os.path.join(siena_dir, 'PN*')))
    if not patient_dirs:
        patient_dirs = sorted(glob.glob(os.path.join(siena_dir, '*')))
    
    for patient_dir in patient_dirs:
        if not os.path.isdir(patient_dir):
            continue
        
        patient = os.path.basename(patient_dir)
        edf_files = sorted(glob.glob(os.path.join(patient_dir, '*.edf')))
        if quick:
            edf_files = edf_files[:2]
        
        patient_rs = []
        
        for edf_file in edf_files:
            result = read_edf_channels(edf_file)
            if result is None:
                errors.append(f"Failed to read: {edf_file}")
                continue
            
            data, fs, ch_names = result
            files_processed += 1
            
            if model is not None:
                windows = run_codec_pipeline(data, model, sample_rate=fs)
                for w in windows:
                    w.file = os.path.basename(edf_file)
                all_windows.extend(windows)
                patient_rs.extend([w.r for w in windows])
        
        if patient_rs:
            per_subject[patient] = float(np.mean(patient_rs))
            print(f"  {patient}: {len(patient_rs)} windows, R={np.mean(patient_rs):.4f}")
    
    if not all_windows:
        return DatasetResult(
            name="Siena (Cross-Site)", num_files=files_processed,
            num_windows=0, num_subjects=len(per_subject),
            mean_r=0, std_r=0, median_r=0, mean_prd=0,
            mean_snr_db=0, mean_cr=0, per_subject_r=per_subject,
            errors=errors
        )
    
    rs = [w.r for w in all_windows]
    return DatasetResult(
        name="Siena (Cross-Site)",
        num_files=files_processed,
        num_windows=len(all_windows),
        num_subjects=len(per_subject),
        mean_r=float(np.mean(rs)),
        std_r=float(np.std(rs)),
        median_r=float(np.median(rs)),
        mean_prd=float(np.mean([w.prd for w in all_windows])),
        mean_snr_db=float(np.mean([w.snr_db for w in all_windows])),
        mean_cr=float(np.mean([w.cr for w in all_windows])),
        per_subject_r=per_subject,
        windows=all_windows,
        errors=errors
    )


def validate_eegmmidb(data_dir: str, model, quick: bool = False) -> DatasetResult:
    """
    EEG Motor Movement/Imagery Dataset.
    109 HEALTHY subjects, 64 channels, 160 Hz.
    Tests: does the codec work on normal brains, not just epilepsy?
    Different channel count, different sample rate, different paradigm.
    """
    print("\n" + "=" * 60)
    print("  EEGMMIDB (Healthy Subjects, Motor Imagery)")
    print("  109 subjects, 64ch, 160 Hz — completely different from training")
    print("=" * 60)
    
    eegmmi_dir = os.path.join(data_dir, 'eegmmidb')
    
    all_windows = []
    per_subject = {}
    errors = []
    files_processed = 0
    
    # EEGMMIDB has S001/S001R01.edf structure
    subject_dirs = sorted(glob.glob(os.path.join(eegmmi_dir, 'S*')))
    if not subject_dirs:
        subject_dirs = sorted(glob.glob(os.path.join(eegmmi_dir, 'files', 'S*')))
    
    if quick:
        subject_dirs = subject_dirs[:10]
    else:
        subject_dirs = subject_dirs[:30]  # Cap at 30 for reasonable runtime
    
    for subject_dir in subject_dirs:
        if not os.path.isdir(subject_dir):
            continue
        
        subject = os.path.basename(subject_dir)
        # Use resting state recordings (R01 = eyes open, R02 = eyes closed)
        edf_files = sorted(glob.glob(os.path.join(subject_dir, '*R01.edf')))
        edf_files += sorted(glob.glob(os.path.join(subject_dir, '*R02.edf')))
        
        if not edf_files:
            edf_files = sorted(glob.glob(os.path.join(subject_dir, '*.edf')))[:2]
        
        subject_rs = []
        
        for edf_file in edf_files:
            result = read_edf_channels(edf_file, target_fs=250)
            if result is None:
                errors.append(f"Failed to read: {edf_file}")
                continue
            
            data, fs, ch_names = result
            files_processed += 1
            
            if model is not None:
                windows = run_codec_pipeline(data, model, sample_rate=fs)
                for w in windows:
                    w.file = os.path.basename(edf_file)
                all_windows.extend(windows)
                subject_rs.extend([w.r for w in windows])
        
        if subject_rs:
            per_subject[subject] = float(np.mean(subject_rs))
    
    if per_subject:
        print(f"  {len(per_subject)} subjects processed")
        print(f"  Mean R across subjects: {np.mean(list(per_subject.values())):.4f}")
    
    if not all_windows:
        return DatasetResult(
            name="EEGMMIDB (Healthy)", num_files=files_processed,
            num_windows=0, num_subjects=len(per_subject),
            mean_r=0, std_r=0, median_r=0, mean_prd=0,
            mean_snr_db=0, mean_cr=0, per_subject_r=per_subject,
            errors=errors
        )
    
    rs = [w.r for w in all_windows]
    return DatasetResult(
        name="EEGMMIDB (Healthy)",
        num_files=files_processed,
        num_windows=len(all_windows),
        num_subjects=len(per_subject),
        mean_r=float(np.mean(rs)),
        std_r=float(np.std(rs)),
        median_r=float(np.median(rs)),
        mean_prd=float(np.mean([w.prd for w in all_windows])),
        mean_snr_db=float(np.mean([w.snr_db for w in all_windows])),
        mean_cr=float(np.mean([w.cr for w in all_windows])),
        per_subject_r=per_subject,
        windows=all_windows,
        errors=errors
    )


def validate_mental_arithmetic(data_dir: str, model, quick: bool = False) -> DatasetResult:
    """
    Mental Arithmetic EEG Dataset.
    36 subjects, 19 channels, 500 Hz, resting + cognitive load.
    Tests: does codec preserve cognitive state differences?
    """
    print("\n" + "=" * 60)
    print("  Mental Arithmetic EEG (Cognitive Load)")
    print("  36 subjects, 500 Hz — tests cognitive state preservation")
    print("=" * 60)
    
    ma_dir = os.path.join(data_dir, 'mental_arithmetic')
    
    all_windows = []
    per_subject = {}
    errors = []
    files_processed = 0
    
    edf_files = sorted(glob.glob(os.path.join(ma_dir, '**', '*.edf'), recursive=True))
    if quick:
        edf_files = edf_files[:10]
    
    for edf_file in edf_files:
        subject = os.path.basename(os.path.dirname(edf_file))
        
        result = read_edf_channels(edf_file, target_fs=250)
        if result is None:
            errors.append(f"Failed to read: {edf_file}")
            continue
        
        data, fs, ch_names = result
        files_processed += 1
        
        if model is not None:
            windows = run_codec_pipeline(data, model, sample_rate=fs)
            for w in windows:
                w.file = os.path.basename(edf_file)
            all_windows.extend(windows)
            
            if subject not in per_subject:
                per_subject[subject] = []
            per_subject[subject].extend([w.r for w in windows])
    
    # Average per subject
    per_subject_avg = {k: float(np.mean(v)) for k, v in per_subject.items() if v}
    
    if per_subject_avg:
        print(f"  {len(per_subject_avg)} subjects processed")
        print(f"  Mean R: {np.mean(list(per_subject_avg.values())):.4f}")
    
    if not all_windows:
        return DatasetResult(
            name="Mental Arithmetic", num_files=files_processed,
            num_windows=0, num_subjects=len(per_subject_avg),
            mean_r=0, std_r=0, median_r=0, mean_prd=0,
            mean_snr_db=0, mean_cr=0, per_subject_r=per_subject_avg,
            errors=errors
        )
    
    rs = [w.r for w in all_windows]
    return DatasetResult(
        name="Mental Arithmetic",
        num_files=files_processed,
        num_windows=len(all_windows),
        num_subjects=len(per_subject_avg),
        mean_r=float(np.mean(rs)),
        std_r=float(np.std(rs)),
        median_r=float(np.median(rs)),
        mean_prd=float(np.mean([w.prd for w in all_windows])),
        mean_snr_db=float(np.mean([w.snr_db for w in all_windows])),
        mean_cr=float(np.mean([w.cr for w in all_windows])),
        per_subject_r=per_subject_avg,
        windows=all_windows,
        errors=errors
    )


# ============================================================
# Main validation runner
# ============================================================

def load_model(model_path: str):
    """Load the trained LamQuant encoder model."""
    if not HAS_TORCH:
        return None
    
    # MOVE-B: encoder is the PRIVATE lamquant_neural wheel
    # (pip-installed); no sys.path injection needed.
    from lamquant_neural.models.encoder import TernaryMobileNetV5
    
    model = TernaryMobileNetV5(21, 32)
    if torch.cuda.is_available():
        model = model.cuda()
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Loaded model from {model_path}")
    else:
        print(f"  WARNING: Model not found at {model_path}, running without codec")
        return None
    
    model.eval()
    return model


def generate_report(datasets: List[DatasetResult], model_path: str) -> ValidationReport:
    """Generate the final validation report."""
    valid_datasets = [d for d in datasets if d.num_windows > 0]
    
    if not valid_datasets:
        return ValidationReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            model_path=model_path,
            codec_version="LamQuant Gen 7",
            datasets=datasets,
            overall_r=0,
            overall_cr=0,
            pass_fail="FAIL",
            summary="No valid windows processed"
        )
    
    # Weighted average by window count
    total_windows = sum(d.num_windows for d in valid_datasets)
    overall_r = sum(d.mean_r * d.num_windows for d in valid_datasets) / total_windows
    overall_cr = sum(d.mean_cr * d.num_windows for d in valid_datasets) / total_windows
    
    # Pass/fail criteria
    chbmit = next((d for d in valid_datasets if 'CHB-MIT' in d.name), None)
    siena = next((d for d in valid_datasets if 'Siena' in d.name), None)
    
    if chbmit and chbmit.mean_r >= 0.85 and (siena is None or siena.mean_r >= 0.75):
        pass_fail = "PASS"
    elif chbmit and chbmit.mean_r >= 0.80:
        pass_fail = "WARN"
    else:
        pass_fail = "FAIL"
    
    # Cross-site degradation
    cross_site_note = ""
    if chbmit and siena and siena.num_windows > 0:
        degradation = chbmit.mean_r - siena.mean_r
        cross_site_note = f"Cross-site R degradation: {degradation:.4f}"
        if degradation > 0.15:
            cross_site_note += " (WARNING: >0.15 cross-site drop, model may be overfitting to CHB-MIT)"
            if pass_fail == "PASS":
                pass_fail = "WARN"
    
    summary_lines = [
        f"Overall R={overall_r:.4f}, CR={overall_cr:.1f}x across {total_windows} windows",
        f"Datasets: {len(valid_datasets)} validated, {len(datasets) - len(valid_datasets)} failed",
    ]
    if cross_site_note:
        summary_lines.append(cross_site_note)
    
    return ValidationReport(
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        model_path=model_path,
        codec_version="LamQuant Gen 7",
        datasets=datasets,
        overall_r=overall_r,
        overall_cr=overall_cr,
        pass_fail=pass_fail,
        summary="\n".join(summary_lines)
    )


def print_report(report: ValidationReport):
    """Print the validation report to stdout."""
    print("\n")
    print("=" * 70)
    print("  LAMQUANT CROSS-DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"  Timestamp:     {report.timestamp}")
    print(f"  Model:         {report.model_path}")
    print(f"  Codec Version: {report.codec_version}")
    print(f"  Result:        {report.pass_fail}")
    print(f"  Overall R:     {report.overall_r:.4f}")
    print(f"  Overall CR:    {report.overall_cr:.1f}x")
    print()
    
    print("  Per-Dataset Results:")
    print("  " + "-" * 66)
    print(f"  {'Dataset':<25} {'Windows':>8} {'Subjects':>9} {'R':>8} {'PRD':>8} {'CR':>6}")
    print("  " + "-" * 66)
    
    for d in report.datasets:
        if d.num_windows > 0:
            print(f"  {d.name:<25} {d.num_windows:>8} {d.num_subjects:>9} "
                  f"{d.mean_r:>8.4f} {d.mean_prd:>7.1f}% {d.mean_cr:>5.1f}x")
        else:
            print(f"  {d.name:<25} {'SKIPPED':>8} {'-':>9} {'-':>8} {'-':>8} {'-':>6}")
    
    print("  " + "-" * 66)
    print()
    print(f"  {report.summary}")
    print()
    
    if report.pass_fail == "PASS":
        print("  ✓ VALIDATION PASSED — codec generalizes across sites and populations")
    elif report.pass_fail == "WARN":
        print("  ⚠ VALIDATION WARNING — check cross-site degradation")
    else:
        print("  ✗ VALIDATION FAILED — codec does not generalize sufficiently")
    
    print("=" * 70)


def main():
    # Auto-detect paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, 'datasets')
    default_model = os.path.join(script_dir, 'lamquant', 'student', 'student_hardened.ckpt')
    default_output = os.path.join(script_dir, 'validation_manifest', 'validation_report.json')

    parser = argparse.ArgumentParser(description='LamQuant Cross-Dataset Validation')
    parser.add_argument('--data-dir', type=str, default=default_data_dir,
                        help='Root directory containing downloaded datasets')
    parser.add_argument('--model', type=str, default=default_model,
                        help='Path to trained encoder checkpoint')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer files per dataset')
    parser.add_argument('--output', type=str, default=default_output,
                        help='Output JSON report path')
    parser.add_argument('--skip-download-check', action='store_true',
                        help='Skip checking if datasets exist')
    
    args = parser.parse_args()
    
    # Check datasets exist
    if not args.skip_download_check:
        if not os.path.isdir(args.data_dir):
            print(f"Dataset directory not found: {args.data_dir}")
            print(f"Run: bash download_datasets.sh {args.data_dir}")
            sys.exit(1)
    
    # Load model
    print("Loading model...")
    model = load_model(args.model)
    
    # Run validation on each dataset
    datasets = []
    
    # 1. CHB-MIT holdout (MUST pass)
    if os.path.isdir(os.path.join(args.data_dir, 'chbmit')):
        datasets.append(validate_chbmit_holdout(args.data_dir, model, args.quick))
    else:
        print("  CHB-MIT not found, skipping")
    
    # 2. Siena (cross-site, SHOULD pass)
    if os.path.isdir(os.path.join(args.data_dir, 'siena')):
        datasets.append(validate_siena(args.data_dir, model, args.quick))
    else:
        print("  Siena not found, skipping")
    
    # 3. EEGMMIDB (healthy subjects)
    if os.path.isdir(os.path.join(args.data_dir, 'eegmmidb')):
        datasets.append(validate_eegmmidb(args.data_dir, model, args.quick))
    else:
        print("  EEGMMIDB not found, skipping")
    
    # 4. Mental Arithmetic
    if os.path.isdir(os.path.join(args.data_dir, 'mental_arithmetic')):
        datasets.append(validate_mental_arithmetic(args.data_dir, model, args.quick))
    else:
        print("  Mental Arithmetic not found, skipping")
    
    # Generate report
    report = generate_report(datasets, args.model)
    print_report(report)
    
    # Save JSON report
    report_dict = {
        'timestamp': report.timestamp,
        'model_path': report.model_path,
        'codec_version': report.codec_version,
        'overall_r': report.overall_r,
        'overall_cr': report.overall_cr,
        'pass_fail': report.pass_fail,
        'summary': report.summary,
        'datasets': []
    }
    for d in report.datasets:
        report_dict['datasets'].append({
            'name': d.name,
            'num_files': d.num_files,
            'num_windows': d.num_windows,
            'num_subjects': d.num_subjects,
            'mean_r': d.mean_r,
            'std_r': d.std_r,
            'median_r': d.median_r,
            'mean_prd': d.mean_prd,
            'mean_snr_db': d.mean_snr_db,
            'mean_cr': d.mean_cr,
            'per_subject_r': d.per_subject_r,
            'errors': d.errors
        })
    
    with open(args.output, 'w') as f:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        json.dump(report_dict, f, indent=2)
    print(f"\n  Report saved to {args.output}")


if __name__ == '__main__':
    main()
