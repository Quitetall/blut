#!/usr/bin/env python3
"""
LamQuant — Universal EDF to Q31 Event Converter
================================================
Converts EDF/EDF+ files from multiple public EEG datasets into the
Q31 .npz format used by the LamQuant training pipeline.

Supported datasets:
  - CHB-MIT Scalp EEG (PhysioNet) — 23ch, 256Hz, seizure annotations
  - Siena Scalp EEG (PhysioNet) — 29ch, 512Hz, seizure annotations
  - EEG Motor Movement/Imagery (PhysioNet) — 64ch, 160Hz, no seizures
  - TUH EEG Corpus (Temple Univ) — variable channels/rates
  - Any EDF with ≥21 channels matching the 10-20 montage

Output format (.npz):
  data:         int32 [21, T] — Q31-scaled EEG, 250Hz
  seizure_mask: float32 [T] — binary mask (1.0 = seizure)
  gain:         float64 — scaling factor applied
  channels:     str[21] — channel names in order
  source:       str — original filename
  dataset:      str — dataset identifier

Channel selection is BY NAME using the 10-20 standard montage.
Files missing required channels are skipped, not silently misaligned.

Usage:
  python edf_to_events.py --input ./dataset --output ./q31_events
  python edf_to_events.py --input ./siena_eeg --output ./q31_events --dataset siena
  python edf_to_events.py --input ./eegmmidb --output ./q31_events --dataset eegmmi
"""
import os
import re
import glob
import argparse
import numpy as np
import mne
from tqdm import tqdm

mne.set_log_level('WARNING')

# =====================================================================
# Canonical 21-channel 10-20 montage (the firmware contract)
# =====================================================================

TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2'
]

# 10-20 alternative names (T7/T8/P7/P8 are the modern names for T3/T4/T5/T6)
CHANNEL_ALIASES = {
    # Standard
    'Fp1': 'Fp1', 'Fp2': 'Fp2', 'F3': 'F3', 'F4': 'F4',
    'C3': 'C3', 'C4': 'C4', 'P3': 'P3', 'P4': 'P4',
    'O1': 'O1', 'O2': 'O2', 'F7': 'F7', 'F8': 'F8',
    'T3': 'T3', 'T4': 'T4', 'T5': 'T5', 'T6': 'T6',
    'Fz': 'Fz', 'Cz': 'Cz', 'Pz': 'Pz', 'A1': 'A1', 'A2': 'A2',
    # Modern 10-20 renames
    'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6',
    # Case variants
    'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz',
    'OZ': 'Oz',
    # PhysioNet EEG prefix variants
    'EEG Fp1-REF': 'Fp1', 'EEG Fp1-LE': 'Fp1', 'EEG FP1-REF': 'Fp1',
    'EEG Fp2-REF': 'Fp2', 'EEG Fp2-LE': 'Fp2', 'EEG FP2-REF': 'Fp2',
    'EEG F3-REF': 'F3', 'EEG F3-LE': 'F3',
    'EEG F4-REF': 'F4', 'EEG F4-LE': 'F4',
    'EEG C3-REF': 'C3', 'EEG C3-LE': 'C3',
    'EEG C4-REF': 'C4', 'EEG C4-LE': 'C4',
    'EEG P3-REF': 'P3', 'EEG P3-LE': 'P3',
    'EEG P4-REF': 'P4', 'EEG P4-LE': 'P4',
    'EEG O1-REF': 'O1', 'EEG O1-LE': 'O1',
    'EEG O2-REF': 'O2', 'EEG O2-LE': 'O2',
    'EEG F7-REF': 'F7', 'EEG F7-LE': 'F7',
    'EEG F8-REF': 'F8', 'EEG F8-LE': 'F8',
    'EEG T3-REF': 'T3', 'EEG T3-LE': 'T3',
    'EEG T4-REF': 'T4', 'EEG T4-LE': 'T4',
    'EEG T5-REF': 'T5', 'EEG T5-LE': 'T5',
    'EEG T6-REF': 'T6', 'EEG T6-LE': 'T6',
    'EEG T7-REF': 'T3', 'EEG T7-LE': 'T3',
    'EEG T8-REF': 'T4', 'EEG T8-LE': 'T4',
    'EEG P7-REF': 'T5', 'EEG P7-LE': 'T5',
    'EEG P8-REF': 'T6', 'EEG P8-LE': 'T6',
    'EEG Fz-REF': 'Fz', 'EEG Fz-LE': 'Fz', 'EEG FZ-REF': 'Fz',
    'EEG Cz-REF': 'Cz', 'EEG Cz-LE': 'Cz', 'EEG CZ-REF': 'Cz',
    'EEG Pz-REF': 'Pz', 'EEG Pz-LE': 'Pz', 'EEG PZ-REF': 'Pz',
    'EEG A1-REF': 'A1', 'EEG A1-LE': 'A1',
    'EEG A2-REF': 'A2', 'EEG A2-LE': 'A2',
    # CHB-MIT uses "FP1-F7" bipolar style — strip to first electrode
    # (handled by the strip function below)
    # Siena uses standard names
    # EEGMMIDB uses "Fc5.", "C3.." dot-padded names
}


def normalize_channel_name(raw_name):
    """Resolve any EDF channel name to our canonical 10-20 name.
    Delegates to the shared channel_resolver module.
    """
    return _shared_resolve(raw_name)


def _normalize_channel_name_legacy(raw_name):
    """Legacy implementation kept for reference — not called.
    All resolution now goes through channel_resolver.resolve().
    """
    name = raw_name.strip().rstrip('.')

    # Direct lookup
    if name in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[name]

    # Strip "EEG " prefix (Siena uses "EEG Fp1", "EEG F3", etc.)
    stripped_eeg = re.sub(r'^EEG\s+', '', name).strip()
    if stripped_eeg != name and stripped_eeg in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[stripped_eeg]

    # Case-insensitive lookup on original
    name_upper = name.upper()
    for alias, canonical in CHANNEL_ALIASES.items():
        if name_upper == alias.upper():
            return canonical

    # Case-insensitive lookup on EEG-stripped version
    if stripped_eeg != name:
        stripped_upper = stripped_eeg.upper()
        for alias, canonical in CHANNEL_ALIASES.items():
            if stripped_upper == alias.upper():
                return canonical

    # CHB-MIT bipolar: "FP1-F7" → try first electrode, then second
    if '-' in name:
        parts = name.split('-')
        first = re.sub(r'^EEG\s+', '', parts[0].strip())
        result = normalize_channel_name(first)
        if result is not None:
            return result
        # Try second electrode (e.g., "P3-O1" → "O1", "CZ-PZ" → "PZ")
        if len(parts) >= 2:
            second = parts[1].strip()
            # Strip trailing duplicate index (e.g., "P8-0" from MNE dedup)
            second = re.sub(r'-\d+$', '', second)
            return normalize_channel_name(second)
        return None

    # EEGMMIDB dot-padded: "Fc5." → "Fc5", "C3.." → "C3"
    stripped_dots = name.replace('.', '').strip()
    if stripped_dots != name and stripped_dots in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[stripped_dots]
    # Case-insensitive on dot-stripped
    if stripped_dots != name:
        stripped_dots_upper = stripped_dots.upper()
        for alias, canonical in CHANNEL_ALIASES.items():
            if stripped_dots_upper == alias.upper():
                return canonical

    return None


def select_channels(raw):
    """Select and reorder channels to match TARGET_CHANNELS.
    Delegates to the shared channel_resolver module.
    Returns (data_array [21, T], missing_list) or (None, missing_list).
    """
    all_data = raw.get_data()
    data, missing = _shared_extract_channel_data(all_data, raw.ch_names)
    return data, missing


# =====================================================================
# Seizure annotation parsers (per-dataset)
# =====================================================================

def parse_chb_summary(summary_path, edf_name):
    """Parse CHB-MIT summary.txt for seizure times."""
    seizures = []
    if not os.path.exists(summary_path):
        return seizures
    try:
        with open(summary_path, 'r') as f:
            lines = f.readlines()
        current_file = None
        start_sec = None
        for line in lines:
            if line.startswith("File Name:"):
                current_file = line.split(":", 1)[1].strip()
            if current_file == edf_name:
                if "Seizure" in line and "Start" in line:
                    m = re.search(r'(\d+)\s*seconds', line)
                    if m:
                        start_sec = int(m.group(1))
                if "Seizure" in line and "End" in line:
                    m = re.search(r'(\d+)\s*seconds', line)
                    if m and start_sec is not None:
                        seizures.append((start_sec, int(m.group(1))))
                        start_sec = None
    except Exception:  # annotation file missing or malformed — return what we found so far
        pass
    return seizures


def parse_siena_annotations(edf_path):
    """Parse Siena dataset seizure annotations from accompanying .tsv or summary."""
    seizures = []
    # Siena uses BIDS-style events.tsv or channel-level annotations
    tsv_path = edf_path.replace('.edf', '_events.tsv')
    if os.path.exists(tsv_path):
        try:
            with open(tsv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3 and 'seizure' in parts[2].lower():
                        onset = float(parts[0])
                        duration = float(parts[1])
                        seizures.append((int(onset), int(onset + duration)))
        except Exception:  # CSV parse error — return what we found so far
            pass
    return seizures


def find_seizure_annotations(edf_path, dataset_type):
    """Route to the correct annotation parser."""
    edf_name = os.path.basename(edf_path)
    dir_path = os.path.dirname(edf_path)

    if dataset_type == 'chbmit':
        folder = os.path.basename(dir_path)
        summary = os.path.join(dir_path, f"{folder}-summary.txt")
        return parse_chb_summary(summary, edf_name)
    elif dataset_type == 'siena':
        return parse_siena_annotations(edf_path)
    else:
        # Try CHB-MIT style first, then Siena style
        folder = os.path.basename(dir_path)
        summary = os.path.join(dir_path, f"{folder}-summary.txt")
        seizures = parse_chb_summary(summary, edf_name)
        if not seizures:
            seizures = parse_siena_annotations(edf_path)
        return seizures


# =====================================================================
# Shared channel resolver (single source of truth)
# =====================================================================
from lamquant_codec.channel_resolver import (
    resolve as _shared_resolve,
    select_channels as _shared_select_channels,
    extract_channel_data as _shared_extract_channel_data,
    TARGET_CHANNELS as _SHARED_TARGET_CHANNELS,
    OPTIONAL_CHANNELS as _SHARED_OPTIONAL_CHANNELS,
)


# =====================================================================
# Binary EDF reader for TUH multi-rate files
# =====================================================================

def _read_edf_binary(edf_path):
    """
    Minimal binary EDF/EDF+ reader for files that MNE chokes on.

    Handles EDF+ continuous mode (num_records = -1) and multi-rate channels
    via mode-based filtering (keeps only channels matching the most common
    sample rate).

    Args:
        edf_path: Path to EDF file.

    Returns:
        (signal_dict, sfreq) where signal_dict is {label: np.ndarray [T]}
        with physical units and sfreq is sample rate in Hz.
        Returns (None, None) on failure.
    """
    from collections import Counter

    try:
        with open(edf_path, 'rb') as f:
            # --- General header (256 bytes) ---
            header_bytes = f.read(256)
            if len(header_bytes) < 256:
                return None, None

            # BDF (BioSemi) detection: header starts with 0xFF "BIOSEMI"
            # rather than "0       ". int24-LE little-endian samples
            # instead of int16-LE; otherwise the layout is identical.
            is_bdf = (
                header_bytes[0] == 0xFF
                and header_bytes[1:8] == b'BIOSEMI'
            )
            bytes_per_sample = 3 if is_bdf else 2

            version     = header_bytes[0:8].decode('latin-1', errors='replace').strip()
            hdr_size    = int(header_bytes[184:192].decode('latin-1').strip())
            num_records = int(header_bytes[236:244].decode('latin-1').strip())
            rec_dur     = float(header_bytes[244:252].decode('latin-1').strip())
            ns          = int(header_bytes[252:256].decode('latin-1').strip())

            if ns <= 0 or rec_dur <= 0:
                return None, None

            # --- Per-channel headers ---
            field_widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]
            field_names  = ['labels', 'transducer', 'phys_dim',
                            'phys_min', 'phys_max', 'dig_min', 'dig_max',
                            'prefilter', 'samples_per_record', 'reserved']

            sig_header_size = hdr_size - 256
            sig_header = f.read(sig_header_size)

            fields = {}
            offset = 0
            for fname, fwidth in zip(field_names, field_widths):
                block = sig_header[offset : offset + fwidth * ns]
                fields[fname] = [
                    block[i*fwidth:(i+1)*fwidth].decode('latin-1').strip()
                    for i in range(ns)
                ]
                offset += fwidth * ns

            labels   = fields['labels']
            phys_min = [float(x) for x in fields['phys_min']]
            phys_max = [float(x) for x in fields['phys_max']]
            dig_min  = [float(x) for x in fields['dig_min']]
            dig_max  = [float(x) for x in fields['dig_max']]
            spr      = [int(x)   for x in fields['samples_per_record']]

            bytes_per_record = bytes_per_sample * sum(spr)

            # Handle EDF+ continuous (num_records = -1)
            if num_records < 0:
                file_size = os.path.getsize(edf_path)
                if bytes_per_record == 0:
                    return None, None
                num_records = (file_size - hdr_size) // bytes_per_record

            if num_records <= 0:
                return None, None

            # --- Channel filtering: ANNOTATION + mode-based multi-rate ---
            # Step 1: exclude ANNOTATION channels
            base_idx = [i for i, lbl in enumerate(labels)
                        if 'ANNOTATION' not in lbl.upper()]

            if not base_idx:
                return None, None

            # Step 2: mode-based multi-rate filter (keep only channels at most common rate)
            mode_spr = Counter(spr[i] for i in base_idx).most_common(1)[0][0]
            sel_idx  = [i for i in base_idx if spr[i] == mode_spr]

            if not sel_idx:
                return None, None

            # Sample rate inferred from mode channel
            sfreq = float(mode_spr) / float(rec_dur)

            # --- Read all signal data ---
            T_total = mode_spr * num_records
            arrays  = {i: np.empty(T_total, dtype=np.float64) for i in sel_idx}

            # Byte offsets of each channel within a record. For BDF this is
            # 3 bytes per sample; for EDF, 2.
            chan_byte_offsets = [bytes_per_sample * sum(spr[:i]) for i in range(ns)]

            write_ptr = 0
            for rec in range(num_records):
                record_buf = f.read(bytes_per_record)
                if len(record_buf) < bytes_per_record:
                    # Truncated file — trim arrays
                    for i in sel_idx:
                        arrays[i] = arrays[i][:write_ptr]
                    break
                for i in sel_idx:
                    offset_b = chan_byte_offsets[i]
                    if is_bdf:
                        # int24 little-endian → int32 with sign extension.
                        # numpy has no native int24 dtype; unpack manually.
                        raw_u8 = np.frombuffer(
                            record_buf, dtype=np.uint8,
                            count=spr[i] * 3, offset=offset_b,
                        ).reshape(-1, 3)
                        # Combine bytes: low | mid<<8 | high<<16
                        raw_int = (
                            raw_u8[:, 0].astype(np.int32)
                            | (raw_u8[:, 1].astype(np.int32) << 8)
                            | (raw_u8[:, 2].astype(np.int32) << 16)
                        )
                        # Sign-extend negative values (top bit of byte 2 set).
                        sign_mask = (raw_int & 0x800000) != 0
                        raw_int[sign_mask] -= 0x1000000
                        raw_int = raw_int.astype(np.float64)
                    else:
                        raw_int = np.frombuffer(
                            record_buf, dtype='<i2',
                            count=spr[i], offset=offset_b
                        ).astype(np.float64)
                    arrays[i][write_ptr:write_ptr + mode_spr] = raw_int
                write_ptr += mode_spr

            # --- Digital to physical scaling ---
            signal_dict = {}
            for i in sel_idx:
                lbl = labels[i]
                dig_range  = dig_max[i] - dig_min[i]
                phys_range = phys_max[i] - phys_min[i]
                if dig_range == 0:
                    signal_dict[lbl] = np.zeros_like(arrays[i])
                else:
                    signal_dict[lbl] = (
                        (arrays[i] - dig_min[i]) * phys_range / dig_range
                    ) + phys_min[i]

            return signal_dict, sfreq

    except Exception:  # corrupt/truncated EDF file — return None so caller tries binary fallback
        return None, None


# =====================================================================
# Core conversion
# =====================================================================


def detect_dataset_type(input_dir):
    """Auto-detect which dataset we're processing.

    TUEP and TUH-seizure look identical at the file naming level; only
    the parent directory name disambiguates. We check the input path
    itself before falling through to the per-file heuristics.
    """
    input_norm = input_dir.lower().replace('\\', '/')
    if 'tuh_epilepsy' in input_norm or input_norm.endswith('tuep'):
        return 'tuep'
    listing = os.listdir(input_dir)
    if any('chb' in f.lower() for f in listing):
        return 'chbmit'
    if any('siena' in f.lower() for f in listing):
        return 'siena'
    if any('S0' in f for f in listing):
        return 'eegmmi'
    return 'generic'


def convert_edf_to_q31(edf_path, output_dir, target_sr, dataset_type,
                       skip_existing=False):
    """Convert one EDF file to Q31 .npz format.

    If skip_existing=True, files that already exist as non-empty, loadable
    NPZ outputs are left alone. This makes re-running the conversion step
    idempotent so downstream L3 precompute work isn't wiped out.
    """
    # Compute the target output path early so we can support skip_existing
    base_name = os.path.splitext(os.path.basename(edf_path))[0]
    planned_out_name = f"{dataset_type}_{base_name}_q31.npz"
    planned_out_path = os.path.join(output_dir, planned_out_name)

    if skip_existing and os.path.exists(planned_out_path) and os.path.getsize(planned_out_path) > 0:
        # Lightweight validity check: the NPZ is just a zip archive; check
        # that it parses as one and contains a `data.npy` member. We
        # DELIBERATELY do not open `data` with np.load — for savez_compressed
        # outputs that would decompress the whole ~5 MB array just to read a
        # shape, turning the "fast skip" path into minutes of wasted I/O
        # across thousands of files.
        try:
            import zipfile as _zf
            if _zf.is_zipfile(planned_out_path):
                with _zf.ZipFile(planned_out_path) as _z:
                    names = _z.namelist()
                    if 'data.npy' in names:
                        return "skipped_exists"
        except Exception:
            # Corrupt file — fall through and re-convert to fix it
            pass

    data = None
    original_sr = None
    mne_error = None

    # --- Try MNE first (primary) ---
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        data, missing = select_channels(raw)
        if data is None:
            # Don't try binary fallback for missing channels — it won't help
            return f"missing_channels: {missing[:5]}..."
        original_sr = raw.info['sfreq']
    except Exception as e:
        mne_error = str(e)

    # --- Binary fallback for TUH multi-rate / EDF+ issues ---
    if data is None:
        signal_dict, sfreq = _read_edf_binary(edf_path)
        if signal_dict is None:
            return f"read_error: {mne_error} (binary fallback also failed)"
        ch_names = list(signal_dict.keys())
        all_data = np.array([signal_dict[ch] for ch in ch_names], dtype=np.float64)
        data, missing = _shared_extract_channel_data(all_data, ch_names)
        if data is None:
            return f"missing_channels (binary): {missing[:5]}..."
        original_sr = sfreq

    # Resample if needed
    if abs(original_sr - target_sr) > 0.5:
        # Use MNE's resampler on the raw object for anti-aliasing
        # But we already extracted data, so resample manually
        from scipy.signal import resample_poly
        from math import gcd
        up = int(target_sr)
        down = int(original_sr)
        g = gcd(up, down)
        up, down = up // g, down // g
        # Cap resampling ratio to prevent memory explosion
        if up > 256 or down > 256:
            # Fall back to scipy resample
            from scipy.signal import resample
            new_len = int(data.shape[1] * target_sr / original_sr)
            data_resampled = np.zeros((21, new_len), dtype=np.float64)
            for ch in range(21):
                data_resampled[ch] = resample(data[ch], new_len)
            data = data_resampled
        else:
            # resample_poly returns ceil(N*up/down) samples; allocating with
            # floor (int(...)) leaves the buffer 1 sample short whenever
            # (N*up) % down != 0 (e.g. 256->250 Hz), raising a broadcast error.
            out_len = int(np.ceil(data.shape[1] * up / down))
            data_resampled = np.zeros((21, out_len), dtype=np.float64)
            for ch in range(21):
                data_resampled[ch] = resample_poly(data[ch], up, down)
            data = data_resampled

    # Highpass at 0.5Hz to remove DC drift (matches firmware biquad)
    # Using a simple 2nd-order Butterworth via scipy
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, 0.5, btype='high', fs=target_sr, output='sos')
    for ch in range(21):
        data[ch] = sosfiltfilt(sos, data[ch])

    # Q31 normalization
    max_abs = np.max(np.abs(data))
    if max_abs < 1e-12:
        return "flat_signal"

    # 72% utilization — 6dB IIR headroom for biquad transients
    gain = 0.72 / max_abs
    data_q31 = (data * gain * 2147483647).astype(np.int32)

    # Seizure mask
    T = data_q31.shape[1]
    seizure_mask = np.zeros(T, dtype=np.float32)
    seizures = find_seizure_annotations(edf_path, dataset_type)
    for start_sec, end_sec in seizures:
        s = int(start_sec * target_sr)
        e = int(end_sec * target_sr)
        if s < T:
            seizure_mask[s:min(e, T)] = 1.0

    # Save
    base_name = os.path.splitext(os.path.basename(edf_path))[0]
    # Prefix with dataset type to avoid name collisions
    out_name = f"{dataset_type}_{base_name}_q31.npz"
    output_path = os.path.join(output_dir, out_name)

    # Delete any stale 0-byte output file to force re-creation
    if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
        os.remove(output_path)

    # Extract EDF metadata so raw EDFs can be safely deleted.
    # This makes the NPZ strictly better than the EDF for our purposes:
    #   - Signal: Q31 int32 (31-bit) > EDF int16 (16-bit) precision
    #   - Channels: standardized 10-20 montage, 250 Hz
    #   - Metadata: patient sex, recording date, original sample rate, gain
    #   - Annotations: per-sample seizure mask
    #   - Provenance: source filename, dataset, original sample rate
    subject_sex = 0  # 0=unknown, 1=male, 2=female (MNE convention)
    recording_date = ''
    try:
        _raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        si = _raw.info.get('subject_info', {})
        if isinstance(si, dict):
            subject_sex = si.get('sex', 0)
        md = _raw.info.get('meas_date')
        if md is not None:
            recording_date = str(md)
    except Exception:
        pass

    np.savez_compressed(
        output_path,
        data=data_q31,
        gain=gain,
        channels=TARGET_CHANNELS,
        seizure_mask=seizure_mask,
        source=os.path.basename(edf_path),
        dataset=dataset_type,
        sample_rate=target_sr,
        original_sample_rate=original_sr,
        subject_sex=subject_sex,
        recording_date=recording_date,
    )

    # Verify output is non-empty (catch disk full, permission issues, etc.)
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return "savez_failed_zero_byte"

    return "ok"


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert EDF files to LamQuant Q31 format")
    parser.add_argument("--input", required=True, help="Directory containing EDF files (searched recursively)")
    parser.add_argument("--output", required=True, help="Output directory for .npz files")
    parser.add_argument("--dataset", default="auto",
                        choices=["auto", "chbmit", "siena", "eegmmi", "tuh", "tuep",
                                 "generic"],
                        help="Dataset type (default: auto-detect). 'tuep' emits "
                             "tuep_* prefix for TUH Epilepsy Corpus to avoid "
                             "filename collision with tuh_seizure NPZs.")
    parser.add_argument("--sr", type=float, default=250.0, help="Target sample rate (default: 250)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files whose Q31 output already exists (idempotent re-runs)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    edf_files = glob.glob(os.path.join(args.input, "**/*.edf"), recursive=True)
    edf_files += glob.glob(os.path.join(args.input, "**/*.EDF"), recursive=True)
    edf_files = sorted(set(edf_files))

    if not edf_files:
        print(f"[!] No EDF files found in {args.input}")
        return

    dataset_type = args.dataset
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(args.input)
    print(f"[*] Dataset type: {dataset_type}")
    print(f"[*] Found {len(edf_files)} EDF files in {args.input}")
    print(f"[*] Target sample rate: {args.sr} Hz")
    print(f"[*] Output: {args.output}")

    stats = {'ok': 0, 'skip': 0, 'resumed': 0, 'errors': {}}

    pbar = tqdm(edf_files, desc="Converting to Q31")
    for edf_path in pbar:
        result = convert_edf_to_q31(
            edf_path, args.output, args.sr, dataset_type,
            skip_existing=args.skip_existing,
        )
        if result == "ok":
            stats['ok'] += 1
        elif result == "skipped_exists":
            stats['resumed'] += 1
        else:
            stats['skip'] += 1
            reason = result.split(':')[0]
            stats['errors'][reason] = stats['errors'].get(reason, 0) + 1
        pbar.set_postfix(ok=stats['ok'], resumed=stats['resumed'],
                         skip=stats['skip'])

    print(f"\n[*] Conversion complete:")
    print(f"    Converted:         {stats['ok']}")
    print(f"    Already existed:   {stats['resumed']}")
    print(f"    Skipped (errors):  {stats['skip']}")
    if stats['errors']:
        print(f"    Reasons:")
        for reason, count in sorted(stats['errors'].items(), key=lambda x: -x[1]):
            print(f"      {reason}: {count}")

    # Remind to clear cache
    print(f"\n[*] Delete q31_cache_v1.pt to rebuild the training cache with new data.")


if __name__ == "__main__":
    main()
