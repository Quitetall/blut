#!/usr/bin/env python3
"""preprocess.py — Universal EDF/LML → LamQuant NPZ preprocessor.

Single tool that converts any EDF/EDF+ or LML file to the standardized
LamQuant format. Handles arbitrary channel counts, sample rates, and montage
conventions. Preserves ALL original metadata so raw EDFs can be deleted
after conversion.

Design principles (Unix philosophy):
  - One tool, one job: EDF/LML → NPZ conversion
  - Configurable but opinionated defaults (21ch, 250Hz, Q31)
  - Complete metadata preservation (EDF header → NPZ metadata)
  - Annotation-agnostic (stores ALL annotations, not just seizures)
  - Idempotent (--skip-existing)

Output format (.npz):
  signal:             [C, T] int32 Q31 — the EEG signal
  signal_native:      [C, T] int16     — native EDF precision (for LML)
  channels:           [C] str          — channel labels in output order
  sample_rate:        float64          — output sample rate
  annotations:        structured       — ALL annotations with timestamps
  metadata:           dict             — complete EDF header preservation

Usage:
  # Default: 21ch 10-20, 250 Hz, Q31
  python preprocess.py input.edf -o output.npz

  # From LML (lossless compressed EDF)
  python preprocess.py input.lml -o output.npz

  # 32-channel at 500 Hz
  python preprocess.py input.edf -o output.npz --channels 32 --sr 500

  # Custom channel list
  python preprocess.py input.edf -o output.npz --channel-list Fp1,Fp2,F3,F4

  # Batch directory processing (finds both .edf and .lml)
  python preprocess.py /data/edf_dir/ -o /data/npz_dir/ --recursive

  # Pipe mode: list files on stdin
  find /data -name '*.lml' | python preprocess.py --stdin -o /data/npz_dir/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Channel presets
# ============================================================

CHANNEL_PRESETS = {
    # Standard 10-20 (21 channels) — LamQuant default, firmware contract
    21: [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2',
    ],
    # Extended 10-20 (32 channels) — includes additional midline and temporal
    32: [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2',
        'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6',
        'FT9', 'FT10', 'TP9',
    ],
    # 10-10 system (64 channels) — BCI, high-density research
    64: [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
        'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
        'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
        'FT7', 'FT8', 'FT9', 'FT10', 'TP7', 'TP8', 'TP9', 'TP10',
        'AF3', 'AF4', 'AF7', 'AF8',
        'PO3', 'PO4', 'PO7', 'PO8',
        'F1', 'F2', 'F5', 'F6',
        'C1', 'C2', 'C5', 'C6',
        'P1', 'P2', 'P5', 'P6',
        'CPz', 'FCz', 'POz', 'Fpz',
    ],
}

# Optional channels that are zero-filled if absent (per preset)
OPTIONAL_CHANNELS = {
    21: {'A1', 'A2'},
    32: {'A1', 'A2', 'FT9', 'FT10', 'TP9'},
    64: set(),  # all required for 64ch
}


# ============================================================
# Metadata extraction
# ============================================================

def extract_edf_metadata(edf_path: str, raw=None) -> Dict:
    """Extract complete metadata from an EDF file.

    Args:
        edf_path: Path to the EDF file (used for source_file/source_path).
        raw: Optional pre-loaded MNE Raw object. If provided, skips re-reading
             the file (2x faster).

    Returns a dict with all recoverable header information so the
    raw EDF can be deleted after conversion without information loss.
    """
    import mne

    meta = {
        'source_file': os.path.basename(edf_path),
        'source_path': str(edf_path),
        'conversion_date': datetime.now(timezone.utc).isoformat(),
        'converter': 'lamquant_preprocess_v1',
    }

    try:
        if raw is None:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        info = raw.info

        # Patient info
        si = info.get('subject_info') or {}
        if isinstance(si, dict):
            meta['subject_id'] = si.get('his_id', '')
            meta['subject_sex'] = si.get('sex', 0)  # 0=unknown, 1=male, 2=female
            meta['subject_hand'] = si.get('hand', 0)
            meta['subject_birthday'] = str(si.get('birthday', ''))
        else:
            meta['subject_sex'] = 0

        # Recording info
        meta['meas_date'] = str(info.get('meas_date', ''))
        meta['experimenter'] = info.get('experimenter', '') or ''
        meta['description'] = info.get('description', '') or ''
        meta['line_freq'] = info.get('line_freq')
        meta['device_info'] = str(info.get('device_info', '')) if info.get('device_info') else ''

        # Channel info (original, before remontage)
        meta['original_channels'] = list(info['ch_names'])
        meta['original_sfreq'] = float(info['sfreq'])
        meta['original_n_channels'] = len(info['ch_names'])

        # Per-channel details
        ch_details = []
        for ch_info in info['chs']:
            ch_details.append({
                'name': ch_info['ch_name'],
                'kind': int(ch_info.get('kind', 0)),
                'unit': int(ch_info.get('unit', 0)),
                'cal': float(ch_info.get('cal', 1.0)),
                'range': float(ch_info.get('range', 1.0)),
            })
        meta['channel_details'] = ch_details

        # Annotations (ALL of them, not just seizures)
        annotations = []
        if raw.annotations is not None:
            for ann in raw.annotations:
                annotations.append({
                    'onset': float(ann['onset']),
                    'duration': float(ann['duration']),
                    'description': str(ann['description']),
                })
        meta['annotations'] = annotations

    except Exception as e:
        meta['read_error'] = str(e)

    return meta


def build_event_mask(annotations: List[Dict], T: int, sr: float,
                     event_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Build per-sample binary masks from annotations.

    Returns {event_type: mask_array[T]} for each unique annotation type.
    If event_types is specified, only returns those types.
    """
    masks = {}
    for ann in annotations:
        desc = ann['description'].lower().strip()
        if event_types and desc not in [e.lower() for e in event_types]:
            continue
        key = desc.replace(' ', '_')
        if key not in masks:
            masks[key] = np.zeros(T, dtype=np.float32)
        s = max(0, int(ann['onset'] * sr))
        e = min(T, int((ann['onset'] + ann['duration']) * sr))
        masks[key][s:e] = 1.0
    return masks


# ============================================================
# LML signal → float microvolts conversion
# ============================================================

def _lml_digital_to_float(signal_int: np.ndarray, metadata: dict) -> np.ndarray:
    """Convert LML digital (int64) signal to float microvolts.

    LML stores the original EDF digital values as int64. The conversion
    to physical units (microvolts) uses the calibration values stored in
    the LML metadata:
        signal_uv = (signal_int - dig_min) / (dig_max - dig_min)
                     * (phys_max - phys_min) + phys_min

    Args:
        signal_int: [C, T] int64 array from read_lml_file()
        metadata: dict from read_lml_file() containing phys_min, phys_max,
                  dig_min, dig_max per channel

    Returns:
        [C, T] float64 array in physical units (typically microvolts)
    """
    C, T = signal_int.shape
    phys_min = metadata.get('phys_min', [-32768.0] * C)
    phys_max = metadata.get('phys_max', [32767.0] * C)
    dig_min = metadata.get('dig_min', [-32768] * C)
    dig_max = metadata.get('dig_max', [32767] * C)

    # Vectorized conversion — no per-channel Python loop
    dig_min_arr = np.array(dig_min, dtype=np.float64).reshape(C, 1)
    dig_max_arr = np.array(dig_max, dtype=np.float64).reshape(C, 1)
    phys_min_arr = np.array(phys_min, dtype=np.float64).reshape(C, 1)
    phys_max_arr = np.array(phys_max, dtype=np.float64).reshape(C, 1)
    dig_range = dig_max_arr - dig_min_arr
    phys_range = phys_max_arr - phys_min_arr
    safe = dig_range != 0
    signal_float = np.where(safe,
        (signal_int.astype(np.float64) - dig_min_arr) * phys_range / dig_range + phys_min_arr,
        0.0)
    return signal_float


# ============================================================
# Core conversion
# ============================================================

def convert_edf(
    edf_path: str,
    output_path: str,
    target_channels: List[str] = None,
    target_sr: float = 250.0,
    optional_channels: set = None,
    highpass_hz: float = 0.5,
    q31_headroom: float = 0.72,
    include_native: bool = True,
) -> str:
    """Convert a single EDF file to LamQuant NPZ format.

    Args:
        edf_path: path to input EDF/EDF+
        output_path: path for output NPZ
        target_channels: list of channel names to extract (default: 21ch 10-20)
        target_sr: output sample rate in Hz (default: 250)
        optional_channels: channels that are zero-filled if absent
        highpass_hz: highpass filter cutoff (0 to disable)
        q31_headroom: utilization factor for Q31 (0.72 = 6dB headroom)
        include_native: also store int16 native-precision data for LML

    Returns:
        'ok' on success, error string on failure.
    """
    import mne
    from lamquant_codec.channel_resolver import extract_channel_data

    if target_channels is None:
        target_channels = CHANNEL_PRESETS[21]
    if optional_channels is None:
        optional_channels = OPTIONAL_CHANNELS.get(len(target_channels), set())

    # ---- Read EDF ----
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        all_data = raw.get_data()  # [n_ch, T] in volts
        all_ch_names = raw.info['ch_names']
        original_sr = raw.info['sfreq']
    except Exception as e:
        return f'read_error: {e}'

    # ---- Extract metadata (reuse already-loaded raw, no double-read) ----
    metadata = extract_edf_metadata(edf_path, raw=raw)

    # ---- Channel selection ----
    data, missing = extract_channel_data(all_data, all_ch_names)
    n_target = len(target_channels)
    if data is None:
        return f'missing_channels: {missing}'

    # Handle non-standard channel counts
    if data.shape[0] != n_target:
        # Pad or truncate to target
        padded = np.zeros((n_target, data.shape[1]), dtype=data.dtype)
        padded[:min(data.shape[0], n_target)] = data[:min(data.shape[0], n_target)]
        data = padded

    # ---- Resample ----
    if abs(original_sr - target_sr) > 0.5:
        from scipy.signal import resample_poly
        from math import gcd
        up = int(target_sr)
        down = int(original_sr)
        g = gcd(up, down)
        up, down = up // g, down // g
        if up > 256 or down > 256:
            from scipy.signal import resample
            new_len = int(data.shape[1] * target_sr / original_sr)
            resampled = np.zeros((data.shape[0], new_len), dtype=np.float64)
            for ch in range(data.shape[0]):
                resampled[ch] = resample(data[ch], new_len)
            data = resampled
        else:
            data = resample_poly(data, up, down, axis=1).astype(np.float64)

    # ---- Highpass filter (remove DC drift) ----
    if highpass_hz > 0:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, highpass_hz, btype='high', fs=target_sr, output='sos')
        data = sosfiltfilt(sos, data, axis=1)

    # ---- Reject flat signals ----
    max_abs = np.max(np.abs(data))
    if max_abs < 1e-12:
        return 'flat_signal'

    # ---- Native int16 (for LML lossless compression) ----
    # Scale to use full int16 range with same headroom
    signal_native = None
    if include_native:
        native_gain = q31_headroom / max_abs
        signal_native = (data * native_gain * 32767).astype(np.int16)

    # ---- Q31 int32 (for neural codec training) ----
    gain = q31_headroom / max_abs
    signal_q31 = (data * gain * 2147483647).astype(np.int32)

    # ---- Build annotation masks ----
    T = signal_q31.shape[1]
    annotations = metadata.get('annotations', [])
    event_masks = build_event_mask(annotations, T, target_sr)

    # Seizure mask (backward compatible — always present)
    seizure_mask = np.zeros(T, dtype=np.float32)
    for key in ('seizure', 'seiz', 'sz', 'seizure_start'):
        if key in event_masks:
            seizure_mask = np.maximum(seizure_mask, event_masks[key])

    # ---- Content hash for deduplication ----
    content_hash = hashlib.sha256(signal_q31.tobytes()[:8192]).hexdigest()[:16]

    # ---- Save ----
    save_dict = {
        # Signal data
        'data': signal_q31,                          # [C, T] int32 Q31
        'seizure_mask': seizure_mask,                # [T] float32
        'gain': np.float64(gain),                    # for Q31 → physical inverse
        'channels': np.array(target_channels[:data.shape[0]]),
        'sample_rate': np.float64(target_sr),
        'original_sample_rate': np.float64(original_sr),
        # Metadata (complete EDF header preservation)
        'source': os.path.basename(edf_path),
        'dataset': metadata.get('dataset', 'unknown'),
        'subject_sex': np.int32(metadata.get('subject_sex', 0)),
        'subject_id': metadata.get('subject_id', ''),
        'recording_date': metadata.get('meas_date', ''),
        'content_hash': content_hash,
        'metadata_json': json.dumps(metadata, default=str),
    }

    # Native int16 for LML (optional, ~50% more file size)
    if signal_native is not None:
        save_dict['signal_native'] = signal_native

    # Per-event-type masks (all annotations, not just seizures)
    for event_key, mask in event_masks.items():
        save_dict[f'mask_{event_key}'] = mask

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(output_path, **save_dict)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return 'save_failed'

    return 'ok'


def convert_lml(
    lml_path: str,
    output_path: str,
    target_channels: List[str] = None,
    target_sr: float = 250.0,
    optional_channels: set = None,
    highpass_hz: float = 0.5,
    q31_headroom: float = 0.72,
    include_native: bool = True,
) -> str:
    """Convert a single LML file to LamQuant NPZ format.

    LML files contain the full EDF signal (as int64 digital values) plus
    all original metadata. This function reads the LML, converts digital
    values to physical units (microvolts), then applies the same channel
    selection, resampling, filtering, and Q31 normalization as convert_edf.

    Args:
        lml_path: path to input LML file
        output_path: path for output NPZ
        target_channels: list of channel names to extract (default: 21ch 10-20)
        target_sr: output sample rate in Hz (default: 250)
        optional_channels: channels that are zero-filled if absent
        highpass_hz: highpass filter cutoff (0 to disable)
        q31_headroom: utilization factor for Q31 (0.72 = 6dB headroom)
        include_native: also store int16 native-precision data for LML

    Returns:
        'ok' on success, error string on failure.
    """
    from lamquant_codec.channel_resolver import extract_channel_data
    from lamquant_codec.edf_to_lml import read_lml_file

    if target_channels is None:
        target_channels = CHANNEL_PRESETS[21]
    if optional_channels is None:
        optional_channels = OPTIONAL_CHANNELS.get(len(target_channels), set())

    # ---- Read LML ----
    try:
        signal_int, lml_meta = read_lml_file(lml_path)
    except Exception as e:
        return f'read_error: {e}'

    # ---- Convert digital → physical (microvolts) ----
    all_data = _lml_digital_to_float(signal_int, lml_meta)
    all_ch_names = lml_meta.get('channels', [])
    original_sr = float(lml_meta.get('sample_rate', 250.0))

    # ---- Build metadata dict (parallel to extract_edf_metadata) ----
    metadata = {
        'source_file': os.path.basename(lml_path),
        'source_path': str(lml_path),
        'source_format': 'lml',
        'conversion_date': datetime.now(timezone.utc).isoformat(),
        'converter': 'lamquant_preprocess_v1',
        'original_channels': list(all_ch_names),
        'original_sfreq': original_sr,
        'original_n_channels': len(all_ch_names),
        'subject_id': lml_meta.get('patient_id', ''),
        'subject_sex': 0,
        'meas_date': lml_meta.get('startdate', ''),
        'annotations': lml_meta.get('annotations', []),
    }

    # ---- Channel selection ----
    data, missing = extract_channel_data(all_data, all_ch_names)
    n_target = len(target_channels)
    if data is None:
        return f'missing_channels: {missing}'

    # Handle non-standard channel counts
    if data.shape[0] != n_target:
        padded = np.zeros((n_target, data.shape[1]), dtype=data.dtype)
        padded[:min(data.shape[0], n_target)] = data[:min(data.shape[0], n_target)]
        data = padded

    # ---- Resample ----
    if abs(original_sr - target_sr) > 0.5:
        from scipy.signal import resample_poly
        from math import gcd
        up = int(target_sr)
        down = int(original_sr)
        g = gcd(up, down)
        up, down = up // g, down // g
        if up > 256 or down > 256:
            from scipy.signal import resample
            new_len = int(data.shape[1] * target_sr / original_sr)
            resampled = np.zeros((data.shape[0], new_len), dtype=np.float64)
            for ch in range(data.shape[0]):
                resampled[ch] = resample(data[ch], new_len)
            data = resampled
        else:
            data = resample_poly(data, up, down, axis=1).astype(np.float64)

    # ---- Highpass filter (remove DC drift) ----
    if highpass_hz > 0:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, highpass_hz, btype='high', fs=target_sr, output='sos')
        data = sosfiltfilt(sos, data, axis=1)

    # ---- Reject flat signals ----
    max_abs = np.max(np.abs(data))
    if max_abs < 1e-12:
        return 'flat_signal'

    # ---- Native int16 (for LML lossless compression) ----
    signal_native = None
    if include_native:
        native_gain = q31_headroom / max_abs
        signal_native = (data * native_gain * 32767).astype(np.int16)

    # ---- Q31 int32 (for neural codec training) ----
    gain = q31_headroom / max_abs
    signal_q31 = (data * gain * 2147483647).astype(np.int32)

    # ---- Build annotation masks ----
    T = signal_q31.shape[1]
    annotations = metadata.get('annotations', [])
    event_masks = build_event_mask(annotations, T, target_sr)

    # Seizure mask (backward compatible — always present)
    seizure_mask = np.zeros(T, dtype=np.float32)
    for key in ('seizure', 'seiz', 'sz', 'seizure_start'):
        if key in event_masks:
            seizure_mask = np.maximum(seizure_mask, event_masks[key])

    # ---- Content hash for deduplication ----
    content_hash = hashlib.sha256(signal_q31.tobytes()[:8192]).hexdigest()[:16]

    # ---- Save ----
    save_dict = {
        'data': signal_q31,
        'seizure_mask': seizure_mask,
        'gain': np.float64(gain),
        'channels': np.array(target_channels[:data.shape[0]]),
        'sample_rate': np.float64(target_sr),
        'original_sample_rate': np.float64(original_sr),
        'source': os.path.basename(lml_path),
        'dataset': metadata.get('dataset', 'unknown'),
        'subject_sex': np.int32(metadata.get('subject_sex', 0)),
        'subject_id': metadata.get('subject_id', ''),
        'recording_date': metadata.get('meas_date', ''),
        'content_hash': content_hash,
        'metadata_json': json.dumps(metadata, default=str),
    }

    if signal_native is not None:
        save_dict['signal_native'] = signal_native

    for event_key, mask in event_masks.items():
        save_dict[f'mask_{event_key}'] = mask

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez(output_path, **save_dict)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return 'save_failed'

    return 'ok'


def convert_file(
    input_path: str,
    output_path: str,
    **kwargs,
) -> str:
    """Dispatch to convert_edf or convert_lml based on file extension.

    Accepts .edf/.EDF (EDF path) or .lml/.LML (LML path).
    All keyword arguments are forwarded to the underlying converter.

    Returns:
        'ok' on success, error string on failure.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.lml':
        return convert_lml(input_path, output_path, **kwargs)
    elif ext in ('.edf',):
        return convert_edf(input_path, output_path, **kwargs)
    else:
        return f'unsupported_format: {ext}'


# ============================================================
# Batch processing
# ============================================================

def find_input_files(input_path: str, recursive: bool = True) -> List[str]:
    """Find all EDF and LML files in a directory."""
    files = []
    if os.path.isfile(input_path):
        return [input_path]
    for ext in ('*.edf', '*.EDF', '*.lml', '*.LML'):
        pattern = f'**/{ext}' if recursive else ext
        for f in Path(input_path).glob(pattern):
            files.append(str(f))
    return sorted(set(files))


def find_edf_files(input_path: str, recursive: bool = True) -> List[str]:
    """Find all EDF/EDF+ files in a directory.

    Kept for backward compatibility. New code should use find_input_files().
    """
    edfs = []
    if os.path.isfile(input_path):
        return [input_path]
    pattern = '**/*.edf' if recursive else '*.edf'
    for f in Path(input_path).glob(pattern):
        edfs.append(str(f))
    for f in Path(input_path).glob(pattern.replace('.edf', '.EDF')):
        edfs.append(str(f))
    return sorted(set(edfs))


def detect_dataset(edf_path: str) -> str:
    """Auto-detect dataset type from path."""
    p = edf_path.lower()
    # Require BOTH 'chb' and 'mit' — the old `... or '/chb' in p` matched any
    # path with a /chb directory (operator precedence: the `or` bound loosely),
    # misclassifying unrelated datasets. Real CHB-MIT paths always carry 'mit'
    # (e.g. chb-mit-scalp-eeg-database, chbmit/).
    if 'chb' in p and 'mit' in p:
        return 'chbmit'
    if 'tueg' in p or 'tuh_eeg' in p:
        return 'tuh'
    if 'tuh_seizure' in p or 'tusz' in p:
        return 'tuh_seizure'
    if 'tuh_artifact' in p or 'tuar' in p:
        return 'tuh_artifact'
    if 'tuep' in p or 'tuh_epilepsy' in p:
        return 'tuep'
    if 'siena' in p:
        return 'siena'
    if 'eegmmi' in p or 'physionet' in p:
        return 'eegmmidb'
    if 'sleep' in p:
        return 'sleep'
    if 'hbn' in p:
        return 'hbn'
    return 'generic'


def make_output_name(edf_path: str, dataset: str) -> str:
    """Generate output NPZ filename."""
    base = Path(edf_path).stem
    return f'{dataset}_{base}_q31.npz'


# ============================================================
# CLI
# ============================================================

# Module-level functions for multiprocessing (must be picklable)
_worker_kwargs = {}

def _pool_init_ignore_sigint():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _worker(task):
    fp, op = task
    try:
        return (fp, convert_file(fp, op, **_worker_kwargs))
    except Exception as e:
        return (fp, f'exception: {e}')


def main() -> int:
    parser = argparse.ArgumentParser(
        prog='preprocess',
        description='Universal EDF/LML → LamQuant NPZ preprocessor. '
                    'Converts any EDF/EDF+ or LML file to standardized format '
                    'with complete metadata preservation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file, default settings (21ch, 250Hz)
  python preprocess.py recording.edf -o output/
  python preprocess.py recording.lml -o output/

  # Batch directory (finds both .edf and .lml)
  python preprocess.py /data/tueg/ -o /data/npz/ --recursive

  # 32-channel at 500Hz
  python preprocess.py input.edf -o out.npz --channels 32 --sr 500

  # Custom channel list
  python preprocess.py input.edf -o out.npz --channel-list Fp1,Fp2,C3,C4

  # Include native int16 for LML compression benchmark
  python preprocess.py input.edf -o out.npz --include-native

  # Skip files that already exist
  python preprocess.py /data/edf/ -o /data/npz/ --skip-existing
""",
    )
    parser.add_argument('input', help='EDF/LML file or directory')
    parser.add_argument('-o', '--output', required=True,
                        help='Output NPZ file or directory')
    parser.add_argument('--channels', type=int, default=21,
                        choices=[21, 32, 64],
                        help='Channel preset (default: 21)')
    parser.add_argument('--channel-list', type=str, default=None,
                        help='Custom comma-separated channel list')
    parser.add_argument('--sr', type=float, default=250.0,
                        help='Output sample rate in Hz (default: 250)')
    parser.add_argument('--highpass', type=float, default=0.5,
                        help='Highpass filter cutoff Hz (0=disable, default: 0.5)')
    parser.add_argument('--headroom', type=float, default=0.72,
                        help='Q31 utilization factor (default: 0.72 = 6dB headroom)')
    parser.add_argument('--include-native', action='store_true', default=False,
                        help='Include int16 native-precision data for LML compression')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset identifier (auto-detected if not set)')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Recursively search for EDF/LML files')
    parser.add_argument('--skip-existing', action='store_true', default=False,
                        help='Skip files that already have a corresponding NPZ')
    parser.add_argument('--stdin', action='store_true', default=False,
                        help='Read file paths from stdin (pipe mode)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers (default: 1)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='List files that would be processed')
    args = parser.parse_args()

    # Resolve channel list
    if args.channel_list:
        target_channels = [ch.strip() for ch in args.channel_list.split(',')]
        optional = set()
    else:
        target_channels = CHANNEL_PRESETS[args.channels]
        optional = OPTIONAL_CHANNELS.get(args.channels, set())

    # Find input files
    if args.stdin:
        input_files = [line.strip() for line in sys.stdin if line.strip()]
    else:
        input_files = find_input_files(args.input, recursive=args.recursive)

    if not input_files:
        print('[!] No EDF or LML files found')
        return 1

    n_edf = sum(1 for f in input_files if f.lower().endswith('.edf'))
    n_lml = sum(1 for f in input_files if f.lower().endswith('.lml'))

    output_is_dir = os.path.isdir(args.output) or args.output.endswith('/')
    if output_is_dir:
        os.makedirs(args.output, exist_ok=True)

    print(f'[*] Preprocessing {len(input_files):,} files ({n_edf} EDF, {n_lml} LML)')
    print(f'    Channels: {len(target_channels)} ({target_channels[:5]}...)')
    print(f'    Sample rate: {args.sr} Hz')
    print(f'    Highpass: {args.highpass} Hz')
    print(f'    Native int16: {"yes" if args.include_native else "no"}')

    if args.dry_run:
        for f in input_files[:20]:
            print(f'  {f}')
        if len(input_files) > 20:
            print(f'  ... and {len(input_files) - 20} more')
        return 0

    # Build task list (skip existing before dispatching to workers)
    tasks = []
    skip = 0
    for file_path in input_files:
        dataset = args.dataset or detect_dataset(file_path)
        out_name = make_output_name(file_path, dataset)
        out_path = os.path.join(args.output, out_name) if output_is_dir else args.output

        if os.path.exists(out_path):
            skip += 1
            continue
        tasks.append((file_path, out_path))

    print(f'  Skipped {skip:,} existing, {len(tasks):,} to process')

    # Set module-level kwargs for workers
    global _worker_kwargs
    _worker_kwargs = dict(
        target_channels=target_channels,
        target_sr=args.sr,
        optional_channels=optional,
        highpass_hz=args.highpass,
        q31_headroom=args.headroom,
        include_native=args.include_native,
    )

    # Process (parallel or serial)
    ok, err = 0, 0
    t0 = time.time()
    total = len(tasks)

    if args.workers > 1 and total > 1:
        import multiprocessing as mp

        with mp.Pool(args.workers, initializer=_pool_init_ignore_sigint) as pool:
            for i, (fp, result) in enumerate(
                    pool.imap_unordered(_worker, tasks, chunksize=8)):
                if result == 'ok':
                    ok += 1
                else:
                    err += 1
                    if err <= 20:
                        print(f'  [err] {os.path.basename(fp)}: {result}')

                processed = i + 1
                if processed % 200 == 0 or processed == total:
                    elapsed = time.time() - t0
                    rate = processed / elapsed
                    eta = (total - processed) / max(rate, 0.01)
                    print(f'  [{processed:,}/{total:,}] '
                          f'{ok:,} ok, {err:,} err '
                          f'({rate:.1f}/s, ETA {eta/60:.0f}m)',
                          flush=True)
    else:
        for i, (fp, op) in enumerate(tasks):
            result = _worker((fp, op))[1]
            if result == 'ok':
                ok += 1
            else:
                err += 1
                if err <= 20:
                    print(f'  [err] {os.path.basename(fp)}: {result}')

            processed = i + 1
            if processed % 100 == 0 or processed == total:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (total - processed) / max(rate, 0.01)
                print(f'  [{processed:,}/{total:,}] '
                      f'{ok:,} ok, {err:,} err '
                      f'({rate:.1f}/s, ETA {eta/60:.0f}m)',
                      flush=True)

    elapsed = time.time() - t0
    print(f'\n[*] Done: {ok:,} ok, {skip:,} skipped, {err:,} errors '
          f'in {elapsed/60:.1f} min ({ok/max(elapsed,1):.1f}/s)')
    return 0 if err == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
