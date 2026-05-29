#!/usr/bin/env python3
"""Convert Mamba SNN predictions to NEDC CSV annotation format for evaluation.

This script runs the Mamba SNN on EDF files and produces per-file CSV
annotation files in the NEDC format, compatible with nedc_eeg_eval (OVLP
metric). This enables direct apples-to-apples comparison with NEDC's
ResNet-18 baseline on TUSZ v2.0.5.

NEDC CSV format:
    # version = csv_v1.0.0
    # bname = filename_without_extension
    # duration = 345.0000 secs
    # montage_file = lamquant_mamba_snn.txt
    #
    channel,start_time,stop_time,label,confidence
    TERM,0.0000,7.1237,bckg,1.0000
    TERM,7.1237,33.5162,seiz,1.0000
    ...

Usage:
    # Generate hypothesis files from SNN
    python scripts/snn_to_nedc_eval.py \\
        --edf-list /path/to/eval_edfs.list \\
        --checkpoint weights/snn/mamba_snn_best.pt \\
        --out-dir /tmp/snn_hyp \\
        --device cuda

    # Score against ground truth using NEDC eval
    python reference_software/nedc_eeg_eval/v6.0.0/src/nedc_eeg_eval/nedc_eeg_eval.py \\
        ref.list hyp.list

    # Where ref.list and hyp.list are text files listing the annotation files,
    # one per line, in the same order.
"""

import argparse
import glob
import os
import numpy as np
import torch

# MambaSNN is the inference-time model definition, now owned by the
# private LamQuant-Neural wheel (lamquant_neural.models). This eval
# driver lives in BLUT (training/eval tooling) per the Neural/BLUT/
# Lossless boundary migration (2026-05-29).
from lamquant_neural.models.mamba_ssm_minimal import MambaSNN

# Default checkpoint root. The trained SNN weights live in the Neural
# repo (LamQuant-Neural/weights/snn/). After MOVE-B this driver lives in
# blut/, so resolve weights via $LAMQUANT_NEURAL_ROOT (set by BLUT when
# launching the recipe) or the sibling LamQuant-Neural checkout, falling
# back to the cwd so an explicit --checkpoint always wins.
ROOT_DIR = os.environ.get("LAMQUANT_NEURAL_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "LamQuant-Neural")
)

# Target EEG channels (21-channel 10-20 montage)
TARGET_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ', 'A1', 'A2',
]

# NEDC parameters matching their ResNet-18 baseline (from Joe's email)
DEFAULT_SEIZ_THRESHOLD = 0.90   # logit threshold for seizure
DEFAULT_MIN_BCKG = 40.0         # minimum bckg segment duration (seconds)
DEFAULT_MIN_SEIZ = 20.0         # minimum seiz segment duration (seconds)


def load_edf_signal(edf_path, target_fs=250.0):
    """Load EDF, extract 21 channels, resample to target_fs.

    Returns (signal [21, T], duration_sec, fs).
    """
    import mne
    mne.set_log_level('ERROR')

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs = raw.info['sfreq']

    # Channel resolution: match to our 21-channel standard
    ch_names_upper = [ch.upper().replace('.', '').replace('-REF', '').replace('-LE', '')
                      for ch in raw.ch_names]
    ch_map = {}
    for i, name in enumerate(ch_names_upper):
        for target in TARGET_CHANNELS:
            if target in name and target not in ch_map:
                ch_map[target] = i
                break

    # Build 21-channel array
    data = raw.get_data()
    signal = np.zeros((21, data.shape[1]), dtype=np.float32)
    for ci, target in enumerate(TARGET_CHANNELS):
        if target in ch_map:
            signal[ci] = data[ch_map[target]].astype(np.float32)

    # Resample if needed
    if abs(fs - target_fs) > 0.5:
        from scipy.signal import resample
        n_out = int(data.shape[1] * target_fs / fs)
        signal = resample(signal, n_out, axis=1).astype(np.float32)
        fs = target_fs

    duration = signal.shape[1] / fs
    return signal, duration, fs


def run_snn_on_signal(model, signal, fs, device, window_samples=2500,
                      stride_samples=2500):
    """Run Mamba SNN on a full-length signal.

    Returns per-timestep activity probability [T_seconds] at 1 Hz.
    """
    model.eval()
    C, T = signal.shape
    n_windows = max(1, (T - window_samples) // stride_samples + 1)

    all_probs = []
    with torch.no_grad():
        for wi in range(n_windows):
            t0 = wi * stride_samples
            t1 = t0 + window_samples
            if t1 > T:
                t0 = max(0, T - window_samples)
                t1 = T
            chunk = signal[:, t0:t1]
            if chunk.shape[1] < window_samples:
                pad = np.zeros((C, window_samples - chunk.shape[1]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=1)

            x = torch.from_numpy(chunk).unsqueeze(0).to(device)  # [1, 21, 2500]
            logits, _ = model(x)  # [1, 8, T_out]

            # Max activity across 8 spatial groups → single probability
            prob = torch.sigmoid(logits.max(dim=1).values)  # [1, T_out]
            all_probs.append(prob.cpu().numpy().flatten())

    # Concatenate and resample to 1 Hz (one value per second)
    if not all_probs:
        return np.zeros(max(1, int(T / fs)))
    probs = np.concatenate(all_probs)
    duration_sec = T / fs
    n_seconds = max(1, int(duration_sec))
    # Pool to 1 Hz
    if len(probs) >= n_seconds:
        block = len(probs) // n_seconds
        probs_1hz = np.array([probs[i * block:(i + 1) * block].max()
                              for i in range(n_seconds)])
    else:
        probs_1hz = np.interp(
            np.arange(n_seconds), np.linspace(0, n_seconds, len(probs)), probs)
    return probs_1hz


def probs_to_segments(probs_1hz, duration_sec, seiz_threshold=0.90,
                      min_bckg=40.0, min_seiz=20.0):
    """Convert per-second probabilities to SEIZ/BCKG segments.

    Applies the same post-processing as NEDC's ResNet-18:
    - Threshold at seiz_threshold
    - Merge short gaps (< min_bckg seconds) between seizures
    - Drop short seizure segments (< min_seiz seconds)
    """
    n = len(probs_1hz)

    # Binary mask: 1 = seizure, 0 = background
    mask = (probs_1hz >= seiz_threshold).astype(int)

    # Merge short background gaps between seizures
    in_gap = False
    gap_start = 0
    for i in range(n):
        if mask[i] == 0 and not in_gap:
            in_gap = True
            gap_start = i
        elif mask[i] == 1 and in_gap:
            gap_len = i - gap_start
            if gap_len < min_bckg:
                mask[gap_start:i] = 1  # fill the gap
            in_gap = False

    # Extract segments
    segments = []
    current_label = 'seiz' if mask[0] else 'bckg'
    seg_start = 0.0

    for i in range(1, n):
        label = 'seiz' if mask[i] else 'bckg'
        if label != current_label:
            segments.append((seg_start, float(i), current_label))
            seg_start = float(i)
            current_label = label
    segments.append((seg_start, duration_sec, current_label))

    # Drop short seizure segments
    filtered = []
    for start, stop, label in segments:
        if label == 'seiz' and (stop - start) < min_seiz:
            label = 'bckg'  # reclassify as background
        filtered.append((start, stop, label))

    # Merge adjacent same-label segments
    merged = [filtered[0]]
    for start, stop, label in filtered[1:]:
        if label == merged[-1][2]:
            merged[-1] = (merged[-1][0], stop, label)
        else:
            merged.append((start, stop, label))

    return merged


def write_nedc_csv(segments, bname, duration_sec, out_path):
    """Write segments in NEDC CSV annotation format."""
    lines = [
        "# version = csv_v1.0.0",
        f"# bname = {bname}",
        f"# duration = {duration_sec:.4f} secs",
        "# montage_file = lamquant_mamba_snn.txt",
        "#",
        "channel,start_time,stop_time,label,confidence",
    ]
    for start, stop, label in segments:
        lines.append(f"TERM,{start:.4f},{stop:.4f},{label},1.0000")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Run Mamba SNN on EDFs and produce NEDC-format annotations')
    parser.add_argument('--edf-list', required=True,
                        help='Text file listing EDF paths (one per line), '
                             'or a directory to glob *.edf')
    parser.add_argument('--checkpoint', default=os.path.join(ROOT_DIR, 'weights', 'snn', 'mamba_snn_best.pt'),
                        help='Mamba SNN checkpoint')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for hypothesis CSV files')
    parser.add_argument('--ref-dir', default=None,
                        help='If provided, also write ref.list and hyp.list for nedc_eeg_eval')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seiz-threshold', type=float, default=DEFAULT_SEIZ_THRESHOLD)
    parser.add_argument('--min-bckg', type=float, default=DEFAULT_MIN_BCKG)
    parser.add_argument('--min-seiz', type=float, default=DEFAULT_MIN_SEIZ)
    parser.add_argument('--d-model', type=int, default=40)
    parser.add_argument('--d-state', type=int, default=16)
    parser.add_argument('--n-layers', type=int, default=2)
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Load model
    model = MambaSNN(in_channels=21, d_model=args.d_model, d_state=args.d_state,
                     n_layers=args.n_layers).to(device)
    # Contains non-tensor metadata (model, sensitivity, accuracy)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"[*] Loaded {args.checkpoint}")
    if 'sensitivity' in ckpt:
        print(f"    sens={ckpt['sensitivity']:.4f}  acc={ckpt.get('accuracy', '?')}")

    # EDF list
    if os.path.isdir(args.edf_list):
        edf_files = sorted(glob.glob(os.path.join(args.edf_list, '**', '*.edf'),
                                     recursive=True))
    elif os.path.isfile(args.edf_list):
        with open(args.edf_list) as f:
            edf_files = [line.strip() for line in f if line.strip()]
    else:
        raise FileNotFoundError(f"Not found: {args.edf_list}")
    print(f"[*] {len(edf_files)} EDF files")

    # Process
    os.makedirs(args.out_dir, exist_ok=True)
    hyp_paths = []
    ref_paths = []
    n_seiz = 0

    for i, edf_path in enumerate(edf_files):
        bname = os.path.splitext(os.path.basename(edf_path))[0]
        out_path = os.path.join(args.out_dir, f"{bname}.csv_bi")

        try:
            signal, duration, fs = load_edf_signal(edf_path)
            probs = run_snn_on_signal(model, signal, fs, device)
            segments = probs_to_segments(probs, duration,
                                         seiz_threshold=args.seiz_threshold,
                                         min_bckg=args.min_bckg,
                                         min_seiz=args.min_seiz)
            write_nedc_csv(segments, bname, duration, out_path)
            hyp_paths.append(out_path)

            has_seiz = any(label == 'seiz' for _, _, label in segments)
            if has_seiz:
                n_seiz += 1

            # Look for matching reference annotation
            if args.ref_dir:
                for ext in ('.csv_bi', '.tse_bi', '.tse'):
                    ref_path = os.path.join(args.ref_dir, f"{bname}{ext}")
                    if os.path.exists(ref_path):
                        ref_paths.append(ref_path)
                        break

            if (i + 1) % 50 == 0 or i == len(edf_files) - 1:
                print(f"  [{i+1}/{len(edf_files)}] {n_seiz} files with seizure detections")

        except Exception as e:
            print(f"  [!] {bname}: {e}")

    # Write list files for nedc_eeg_eval
    hyp_list = os.path.join(args.out_dir, 'hyp.list')
    with open(hyp_list, 'w') as f:
        f.write('\n'.join(hyp_paths) + '\n')
    print(f"\n[*] Hypothesis list: {hyp_list} ({len(hyp_paths)} files)")

    if ref_paths:
        ref_list = os.path.join(args.out_dir, 'ref.list')
        with open(ref_list, 'w') as f:
            f.write('\n'.join(ref_paths) + '\n')
        print(f"[*] Reference list: {ref_list} ({len(ref_paths)} files)")
        print(f"\n[*] Run evaluation:")
        print(f"    python reference_software/nedc_eeg_eval/v6.0.0/src/"
              f"nedc_eeg_eval/nedc_eeg_eval.py {ref_list} {hyp_list}")

    print(f"\n[*] Done. {len(hyp_paths)} files processed, {n_seiz} with seizure detections.")


if __name__ == '__main__':
    main()
