#!/usr/bin/env python3
"""
Mamba-based SNN activity detector for LamQuant.

Bidirectional selective state space model for seizure / activity
detection from EEG. Drop-in replacement for train_dlif_run.py.

Advantages over dLIF:
  - Captures long-range temporal dependencies (Mamba's selective scan)
  - Complex-valued state tracking (transient spike detection)
  - Linear memory scaling (O(T) vs O(T²) for attention)
  - Same output shape [B, 8, T_out] for codec compatibility

Usage:
    python train_mamba_snn.py --data ai_models/snn/labels --epochs 500
    python train_mamba_snn.py --data ai_models/snn/labels --subband --eeg-dir /mnt/4tb/data/lml/tueg_super

Firmware size: ~4-8 KB at W2A8 (vs 64 KB budget).
"""

import os
import sys
import gc
import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# MOVE-B (2026-05-29): now at blut/python/lamquant/snn/. ROOT_DIR is
# the blut/python package root; sibling area dirs go on sys.path for
# the bare cross-area imports below. MambaSNN is the PRIVATE
# lamquant_neural model def (pip-installed).
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'snn'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'dataset'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'common'))

from lamquant_neural.models.mamba_ssm_minimal import MambaSNN
from snn_training_config import SNN_CONFIGS, SNNConfig

# Geometry constants: previously read from the legacy
# ai_models/architectures/snn.py (a different, frozen lineage that
# stays in the Neural repo and is NOT shipped to BLUT). These are the
# stable production values; the legacy module is unavailable here so we
# use them directly (matching the prior ImportError fallback).
NUM_CHANNELS, NUM_GROUPS, STRIDE_8 = 21, 8, 8
T_INPUT, T_LATENT = 2500, 312
SPATIAL_GROUPS = None

try:
    from generate_validation_split import load_validation_manifest, get_excluded_windows
except ImportError:
    load_validation_manifest = None
    get_excluded_windows = None


# ============================================================
# Dataset — real EEG only, no synthetic fallback
# ============================================================

Q31_PREFIXES = (
    'tuh_seizure_', 'tuh_artifact_', 'tuh_epilepsy_', 'tueg_',
    'chbmit_', 'tuh_', 'siena_', 'eegmmidb_', 'mental_arithmetic_',
)

# 21-channel 10-20 montage targets
_TARGET_CH = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'FZ', 'CZ', 'PZ', 'A1', 'A2',
]


def _load_edf_signal(edf_path, window_size):
    """Load EDF, extract 21 channels, return [21, window_size] float32."""
    import mne
    mne.set_log_level('ERROR')
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs = raw.info['sfreq']
    data = raw.get_data().astype(np.float32)
    # Channel matching
    ch_upper = [ch.upper().replace('.', '').replace('-REF', '').replace('-LE', '')
                for ch in raw.ch_names]
    signal = np.zeros((NUM_CHANNELS, data.shape[1]), dtype=np.float32)
    for ci, target in enumerate(_TARGET_CH):
        for i, name in enumerate(ch_upper):
            if target in name:
                signal[ci] = data[i]
                break
    # Resample to 250 Hz if needed
    if abs(fs - 250.0) > 0.5:
        from scipy.signal import resample
        n_out = int(data.shape[1] * 250.0 / fs)
        signal = resample(signal, n_out, axis=1).astype(np.float32)
    # Truncate/pad to window_size
    T = signal.shape[1]
    if T > window_size:
        signal = signal[:, :window_size]
    elif T < window_size:
        signal = np.pad(signal, ((0, 0), (0, window_size - T)))
    return signal


class ActivityLabelDataset(Dataset):
    """Activity labels paired with real EEG data, pre-loaded into RAM.

    Two-pass loading like SubbandActivityDataset: scan → pre-allocate →
    fill. All signals loaded at init time. No per-batch I/O.

    Supports Q31 .npz files (fast) and raw .edf files (slower init, uses MNE).
    """

    def __init__(self, data_dir, eeg_dir, window_size=T_INPUT, excluded_windows=None,
                 max_windows_per_file=10):
        self.window_size = window_size
        label_files = sorted(glob.glob(os.path.join(data_dir, '*_labels.npz')))
        if not label_files:
            raise ValueError(f"No label files found in {data_dir}")

        # Normalize eeg_dir to list
        eeg_dirs = eeg_dir if isinstance(eeg_dir, (list, tuple)) else [eeg_dir]

        # Build EEG lookup: stem -> path (supports Q31 .npz and raw .edf)
        eeg_map = {}
        is_edf = False
        for d in eeg_dirs:
            for f in glob.glob(os.path.join(d, '*_q31.npz')):
                stem = os.path.basename(f).replace('_q31.npz', '')
                for prefix in Q31_PREFIXES:
                    if stem.startswith(prefix):
                        stem = stem[len(prefix):]
                        break
                eeg_map[stem] = f
        if not eeg_map:
            is_edf = True
            for d in eeg_dirs:
                for f in glob.glob(os.path.join(d, '**', '*.edf'),
                                   recursive=True):
                    stem = os.path.splitext(os.path.basename(f))[0]
                    eeg_map[stem] = f
            if eeg_map:
                print(f"  Found {len(eeg_map)} EDF files from {len(eeg_dirs)} dirs (pre-loading into RAM)")

        # Pass 1: match labels to EEG files
        excluded = excluded_windows or set()
        matched = []
        n_excluded = n_no_eeg = 0
        for lf in label_files:
            with np.load(lf) as labels:
                activity = np.array(labels['activity_labels'])
                source = str(labels.get('source', ''))
            if excluded and source:
                if any((source, w) in excluded
                       for w in range(activity.shape[1])):
                    n_excluded += 1
                    continue
            src_stem = source.replace('.edf', '')
            eeg_path = eeg_map.get(src_stem)
            if eeg_path is None:
                n_no_eeg += 1
                continue
            matched.append((eeg_path, activity, source))
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} validation files")
        if n_no_eeg > 0:
            print(f"  Skipped {n_no_eeg} files (no matching EEG)")
        if not matched:
            raise ValueError(
                f"No samples with matching EEG. Provide --eeg-dir "
                f"with Q31 .npz or .edf files (found {len(eeg_map)})")

        # Pass 2: count total windows across all files
        T_lat = window_size // STRIDE_8
        n = len(matched)
        total_windows = 0
        n_seizure_windows = 0
        file_windows = []  # (eeg_path, is_edf, n_windows, activity)
        for eeg_path, activity, source in matched:
            label_T = activity.shape[1]
            n_total = max(1, label_T // T_lat)
            # Find seizure windows
            sz_wins = []
            bg_wins = []
            for wi in range(n_total):
                lbl_start = wi * T_lat
                lbl_end = min(lbl_start + T_lat, label_T)
                if lbl_end > lbl_start and np.any(activity[:, lbl_start:lbl_end] == 2):
                    sz_wins.append(wi)
                else:
                    bg_wins.append(wi)
            # Always include seizure windows, fill rest with evenly-spaced bg
            must_include = set(sz_wins)
            remaining = max(0, max_windows_per_file - len(must_include))
            if remaining > 0 and bg_wins:
                n_bg = min(remaining, len(bg_wins))
                bg_idx = np.linspace(0, len(bg_wins) - 1, n_bg, dtype=int)
                selected_bg = [bg_wins[i] for i in bg_idx]
            else:
                selected_bg = []
            selected = sorted(must_include | set(selected_bg))
            if not selected:
                selected = [0]
            file_windows.append((eeg_path, selected, activity))
            total_windows += len(selected)
            n_seizure_windows += len(must_include)

        print(f"  Pre-loading {n} files ({total_windows} windows, "
              f"{n_seizure_windows} with seizures) into RAM...")
        self.signals = torch.empty(total_windows, NUM_CHANNELS, window_size,
                                   dtype=torch.float32)
        self.labels = torch.zeros(total_windows, NUM_GROUPS, T_lat,
                                  dtype=torch.long)

        loaded = 0
        for i, (eeg_path, selected, activity) in enumerate(file_windows):
            try:
                if is_edf:
                    max_sample = (max(selected) + 1) * window_size
                    full_signal = _load_edf_signal(eeg_path, max_sample)
                else:
                    with np.load(eeg_path) as eeg_data:
                        if 'data' in eeg_data:
                            full_signal = np.array(eeg_data['data'], dtype=np.float32)
                        elif 'l3' in eeg_data:
                            l3 = np.array(eeg_data['l3'], dtype=np.float32)
                            full_signal = l3[0] if l3.ndim == 3 else l3
                        else:
                            continue
                    if full_signal.max() > 100:
                        full_signal = full_signal / (2**31)
            except Exception:
                continue

            for wi in selected:
                sig_start = wi * window_size
                sig_end = sig_start + window_size
                if sig_end > full_signal.shape[-1]:
                    continue
                chunk = full_signal[..., sig_start:sig_end]
                if chunk.shape[-1] < window_size:
                    chunk = np.pad(chunk, ((0, 0),) * (chunk.ndim - 1) + ((0, window_size - chunk.shape[-1]),))

                lbl_start = wi * T_lat
                lbl_end = min(lbl_start + T_lat, activity.shape[1])
                lbl_len = lbl_end - lbl_start
                if lbl_len < 1:
                    continue

                self.signals[loaded] = torch.from_numpy(chunk)
                self.labels[loaded, :, :lbl_len] = torch.from_numpy(
                    activity[:, lbl_start:lbl_end].astype(np.int64))
                if lbl_len < T_lat:
                    self.labels[loaded, :, lbl_len:] = self.labels[loaded, :, lbl_len - 1:lbl_len]
                loaded += 1

            del full_signal
            if (i + 1) % 500 == 0:
                ram = _rss_gb()
                print(f"    [{i+1}/{n}] loaded={loaded} windows  RAM={ram:.1f} GB")

        self.signals = self.signals[:loaded]
        self.labels = self.labels[:loaded]
        del file_windows
        gc.collect()

        sig_gb = self.signals.nbytes / 1e9
        lbl_gb = self.labels.nbytes / 1e9
        print(f"  Loaded {loaded} windows from {n} files: "
              f"signals {sig_gb:.2f} GB + labels {lbl_gb:.2f} GB  "
              f"(RSS {_rss_gb():.1f} GB)")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def _dwb_weight(target, pred_logits, base_pos_weight=3.0):
    """Dynamically Weighted Balanced per-sample loss weighting.

    DWB self-adapts based on class frequency AND prediction difficulty.
    For each sample: w = class_weight * (1 + |p - y|)^gamma
    """
    gamma = 2.0
    with torch.no_grad():
        p = torch.sigmoid(pred_logits)
        difficulty = (p - target).abs()
        class_w = torch.where(target > 0.5, base_pos_weight, 1.0)
        dwb_w = class_w * (1.0 + difficulty).pow(gamma)
        dwb_w = dwb_w / (dwb_w.mean() + 1e-8)
    return dwb_w


def _rss_gb():
    """Current process RSS in GB (0 if unavailable)."""
    try:
        with open(f'/proc/{os.getpid()}/statm') as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE') / 1e9
    except Exception:
        return 0.0


class SubbandActivityDataset(Dataset):
    """L3 subband windows from q31 .npz files matched to activity labels.

    Two-pass loading: count windows -> pre-allocate contiguous tensors -> fill.
    No list-append fragmentation, explicit cleanup, zero memory leaks.

    Each q31 file has l3: [N_windows, 21, 313].  We take up to
    `max_windows_per_file` evenly-spaced windows and pair them with
    the corresponding stride-8 label segment.
    """

    Q31_PREFIXES = (
    'tuh_seizure_', 'tuh_artifact_', 'tuh_epilepsy_', 'tueg_',
    'chbmit_', 'tuh_', 'siena_', 'eegmmidb_', 'mental_arithmetic_',
)
    L3_T = 313
    LABEL_PER_WINDOW = 312   # 2500 // 8

    def __init__(self, label_dir, eeg_dir, max_windows_per_file=10,
                 excluded_windows=None):
        excluded = excluded_windows or set()

        # Build q31 lookup: stripped stem -> path. `eeg_dir` may be a single
        # path (legacy) OR a list of paths (CLI uses nargs='+'). Glob across
        # all entries so multi-corpus training works.
        if isinstance(eeg_dir, (str, os.PathLike)):
            eeg_dirs = [str(eeg_dir)]
        else:
            eeg_dirs = [str(d) for d in eeg_dir]
        q31_map = {}
        for d in eeg_dirs:
            for f in glob.glob(os.path.join(d, '*.npz')):
                stem = os.path.basename(f).replace('_q31.npz', '')
                for prefix in self.Q31_PREFIXES:
                    if stem.startswith(prefix):
                        stem = stem[len(prefix):]
                        break
                q31_map[stem] = f

        # -- Pass 1: scan labels, match to q31, pick windows --
        label_files = sorted(glob.glob(os.path.join(label_dir, '*_labels.npz')))
        entries = []
        total_windows = 0
        n_matched = n_skipped = n_seizure_wins = 0

        for lf in label_files:
            with np.load(lf) as data:
                src = str(data.get('source', '')).replace('.edf', '')
                activity = np.array(data['activity_labels'])

            if src not in q31_map:
                n_skipped += 1
                del activity
                continue

            if excluded and any((src + '.edf', w) in excluded
                                for w in range(min(10, activity.shape[1]))):
                n_skipped += 1
                del activity
                continue

            n_total = max(1, activity.shape[1] // self.LABEL_PER_WINDOW)

            # Find windows containing seizure labels
            seizure_wins = []
            for wi in range(n_total):
                lbl_start = wi * self.LABEL_PER_WINDOW
                lbl_end = min(lbl_start + self.L3_T, activity.shape[1])
                if lbl_end > lbl_start and np.any(activity[:, lbl_start:lbl_end] == 2):
                    seizure_wins.append(wi)

            must_include = set(seizure_wins)
            n_seizure_wins += len(must_include)
            remaining_budget = max(0, max_windows_per_file - len(must_include))
            if remaining_budget > 0 and n_total > len(must_include):
                candidates = [i for i in range(n_total) if i not in must_include]
                n_bg = min(remaining_budget, len(candidates))
                bg_idx = np.linspace(0, len(candidates) - 1, n_bg, dtype=int)
                bg_wins = [candidates[i] for i in bg_idx]
            else:
                bg_wins = []

            selected = sorted(must_include | set(bg_wins))
            if not selected:
                selected = [0]

            entries.append((q31_map[src], activity, selected))
            total_windows += len(selected)
            n_matched += 1

        print(f"  [SubbandDS] {n_matched} files matched, {n_skipped} skipped, "
              f"loading {total_windows} windows ({n_seizure_wins} seizure)")

        # -- Pass 2: allocate contiguous tensors and fill --
        self.signals = torch.empty(total_windows, 21, self.L3_T, dtype=torch.float32)
        self.labels = torch.zeros(total_windows, 8, self.L3_T, dtype=torch.long)

        idx = 0
        for q31_path, activity, selected in entries:
            try:
                with np.load(q31_path) as data:
                    if 'l3' not in data:
                        continue
                    l3 = np.array(data['l3'])
            except Exception:
                continue  # corrupt or missing l3

            for wi in selected:
                if wi >= l3.shape[0]:
                    continue
                self.signals[idx] = torch.from_numpy(l3[wi])

                lbl_start = wi * self.LABEL_PER_WINDOW
                lbl_end = min(lbl_start + self.L3_T, activity.shape[1])
                lbl_len = lbl_end - lbl_start
                if lbl_len > 0:
                    self.labels[idx, :, :lbl_len] = torch.from_numpy(
                        activity[:, lbl_start:lbl_end].astype(np.int64))
                    if lbl_len < self.L3_T:
                        self.labels[idx, :, lbl_len:] = self.labels[idx, :, lbl_len - 1:lbl_len]
                idx += 1

            del l3, activity

        self.signals = self.signals[:idx]
        self.labels = self.labels[:idx]

        del entries
        gc.collect()

        sig_gb = self.signals.nbytes / 1e9
        lbl_gb = self.labels.nbytes / 1e9
        print(f"  [SubbandDS] {idx} windows loaded: "
              f"signals {sig_gb:.2f} GB + labels {lbl_gb:.2f} GB = "
              f"{sig_gb + lbl_gb:.2f} GB  (RSS {_rss_gb():.1f} GB)")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


# ============================================================
# Training + Validation
# ============================================================

def _augment_eeg(signal, p_channel_drop=0.15, p_amplitude=0.5, p_noise=0.5,
                  p_time_shift=0.3):
    """In-place EEG augmentation for SNN training.

    Applied per-batch on GPU. Does not modify labels (label-preserving).
      - Channel dropout: zero out 1-3 random channels (simulates bad electrodes)
      - Amplitude scaling: per-channel scale 0.7-1.3x (simulates gain variation)
      - Gaussian noise: additive noise at 5% signal std (simulates ADC noise)
      - Time shift: circular shift ±50 samples (simulates onset jitter)
    """
    B, C, T = signal.shape

    # Channel dropout: zero 1-3 channels per sample
    if torch.rand(1).item() < p_channel_drop:
        n_drop = torch.randint(1, 4, (1,)).item()
        drop_idx = torch.randperm(C)[:n_drop]
        signal[:, drop_idx, :] = 0.0

    # Amplitude scaling: per-channel, per-sample
    if torch.rand(1).item() < p_amplitude:
        scale = 0.7 + 0.6 * torch.rand(B, C, 1, device=signal.device)
        signal = signal * scale

    # Gaussian noise
    if torch.rand(1).item() < p_noise:
        std = signal.std() * 0.05
        signal = signal + torch.randn_like(signal) * std

    # Time shift: circular shift ±50 samples
    if torch.rand(1).item() < p_time_shift:
        shift = torch.randint(-50, 51, (1,)).item()
        signal = torch.roll(signal, shifts=shift, dims=-1)

    return signal


def train_epoch(model, loader, optimizer, device, lambda_spike=0.01,
                pos_weight=3.0, augment=True):
    """Training loop with DWB loss + EEG augmentation.

    DWB (Dynamically Weighted Balanced) handles class imbalance via
    per-sample weighting based on class frequency + prediction difficulty.
    pos_weight should be computed from data distribution (neg/pos ratio).
    """
    model.train()
    total_loss = total_correct = total_samples = 0
    total_seizure_tp = total_seizure_fn = 0
    nan_skips = 0
    n_steps = 0

    _zero = torch.tensor(0.0, device=device)

    for signal, labels in loader:
        signal, labels = signal.to(device), labels.to(device)
        if augment:
            signal = _augment_eeg(signal)
        optimizer.zero_grad()

        activity_logits, spike_rate = model(signal)
        # Logit scaling: 3.0 gives moderate sigmoid sharpness.
        # 5.0 was too sharp — near-binary sigmoid killed gradient signal
        # on borderline samples and made DWB difficulty weighting erratic.
        scaled_logits = activity_logits * 3.0
        target_event = (labels >= 1).float()

        # DWB loss: per-sample dynamic weighting (class freq + difficulty)
        dwb_w = _dwb_weight(target_event, scaled_logits, pos_weight)
        bce_unreduced = F.binary_cross_entropy_with_logits(
            scaled_logits, target_event, reduction='none')
        bce_loss = (bce_unreduced * dwb_w).mean()

        # Seizure-specific loss: extra penalty for missing seizures.
        # Kept at 1.0x — DWB's pos_weight already upweights positive
        # samples, and the data-derived pos_weight captures the true
        # class imbalance. The old 2.0x + class_weights(17x) triple-
        # stacked and caused the model to predict everything as active.
        seizure_mask = (labels == 2).float()
        if seizure_mask.sum() > 0:
            seizure_loss_unreduced = F.binary_cross_entropy_with_logits(
                scaled_logits, seizure_mask, reduction='none')
            seizure_loss = (seizure_loss_unreduced * seizure_mask).sum() / (seizure_mask.sum() + 1e-8)
        else:
            seizure_loss = _zero

        loss = bce_loss + seizure_loss + lambda_spike * spike_rate

        # Non-finite-loss guard. A divergent Mamba SSM scan can emit inf/nan
        # activations as weights grow during the LR warmup ramp; doing
        # backward()+step() on a non-finite loss permanently poisons EVERY
        # weight to NaN — the exact failure that collapsed the seizure head
        # from val[S]=0.888 (ep10) to 0.000 (ep16) and left L=nan for the rest
        # of the run. Skip the offending batch entirely so params stay clean
        # and the optimizer trajectory is never stepped on garbage gradients.
        if not torch.isfinite(loss):
            nan_skips += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_steps += 1

        total_loss += loss.item()
        with torch.no_grad():
            pred = (activity_logits > 0).long()
            actual = (labels >= 1).long()
            total_correct += (pred == actual).sum().item()
            total_samples += labels.numel()
            sz = (labels == 2)
            if sz.any():
                total_seizure_tp += ((pred == 1) & sz).sum().item()
                total_seizure_fn += ((pred == 0) & sz).sum().item()

    if nan_skips:
        print(f"  [train_epoch] skipped {nan_skips}/{len(loader)} non-finite-loss "
              f"batch(es) via SSM-divergence guard; stepped {n_steps}")
    acc = total_correct / max(total_samples, 1)
    sens = total_seizure_tp / max(total_seizure_tp + total_seizure_fn, 1)
    avg_loss = total_loss / max(n_steps, 1)
    sr = spike_rate.item() if torch.isfinite(spike_rate).all() else float('nan')
    return avg_loss, acc, sens, sr


def validate(model, loader, device):
    """Validation with accuracy, sensitivity, specificity, FNR."""
    model.eval()
    total_correct = total_samples = 0
    total_seizure_tp = total_seizure_fn = total_quiet_tn = total_quiet_fp = 0
    with torch.no_grad():
        for signal, labels in loader:
            signal, labels = signal.to(device), labels.to(device)
            logits, _ = model(signal)
            pred = (logits > 0).long()
            actual = (labels >= 1).long()
            total_correct += (pred == actual).sum().item()
            total_samples += labels.numel()
            sz = (labels == 2)
            if sz.any():
                total_seizure_tp += (pred[sz] == 1).sum().item()
                total_seizure_fn += (pred[sz] == 0).sum().item()
            qt = (labels == 0)
            if qt.any():
                total_quiet_tn += (pred[qt] == 0).sum().item()
                total_quiet_fp += (pred[qt] == 1).sum().item()
    acc = total_correct / max(total_samples, 1)
    sens = total_seizure_tp / max(total_seizure_tp + total_seizure_fn, 1)
    spec = total_quiet_tn / max(total_quiet_tn + total_quiet_fp, 1)
    fnr = total_seizure_fn / max(total_seizure_tp + total_seizure_fn, 1)
    return acc, sens, spec, fnr


# ============================================================
# Firmware export — INT8 quantized Mamba weights to C header
# ============================================================

def _float_to_q31(val):
    return int(np.clip(val * (2**31 - 1), -(2**31), 2**31 - 1))


def _float_to_q15(val):
    return int(np.clip(val * (2**15 - 1), -(2**15), 2**15 - 1))


def _emit_int8_array(lines, name, tensor, per_row=16):
    """Quantize a float tensor to INT8 and emit as C array."""
    w = tensor.detach().cpu().float().numpy().flatten()
    scale = max(abs(w.max()), abs(w.min()), 1e-8) / 127.0
    w_q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    lines.append(f"/* scale={scale:.8e} */")
    lines.append(f"static const int8_t {name}[{len(w_q)}] = {{")
    for i in range(0, len(w_q), per_row):
        chunk = w_q[i:i + per_row]
        lines.append("    " + ', '.join(f'{v:4d}' for v in chunk) + ",")
    lines.append("};")
    lines.append(f"static const float {name}_scale = {scale:.8e}f;")
    lines.append("")
    return len(w_q)


def export_mamba_weights(model, output_path):
    """Export MambaSNN weights to C header for firmware.

    All weights quantized to INT8 with per-tensor scale factors.
    The firmware multiplies: output = (int8_weight * scale) * input.

    Architecture on device:
      spatial_mix:  INT8 Linear(21 → d_model)
      ssm_blocks:   INT8 SSM (A_log, conv1d, projections)
      readout:      INT8 Linear(d_model → 8)
    """
    print(f"\n[*] Exporting Mamba SNN weights to {output_path}")
    d_model = model.spatial_mix.in_features
    out_dim = d_model  # spatial_mix output
    n_layers = len(model.ssm_blocks)
    n_groups = model.NUM_GROUPS

    lines = [
        "/* Auto-generated by train_mamba_snn.py — Mamba SNN for LamQuant */",
        "#ifndef MAMBA_SNN_WEIGHTS_H",
        "#define MAMBA_SNN_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define MAMBA_SNN_IN_CHANNELS   {model.spatial_mix.in_features}",
        f"#define MAMBA_SNN_D_MODEL       {out_dim}",
        f"#define MAMBA_SNN_N_LAYERS      {n_layers}",
        f"#define MAMBA_SNN_NUM_GROUPS    {n_groups}",
        f"#define MAMBA_SNN_STRIDE        {model.stride}",
        "",
    ]

    total_bytes = 0

    # Spatial mix: Linear(21, d_model)
    total_bytes += _emit_int8_array(lines, "mamba_spatial_mix_w",
                                    model.spatial_mix.weight)
    total_bytes += _emit_int8_array(lines, "mamba_spatial_mix_b",
                                    model.spatial_mix.bias)

    # SSM blocks
    for li, block in enumerate(model.ssm_blocks):
        for direction, ssm in [("fwd", block.fwd), ("bwd", block.bwd)]:
            prefix = f"mamba_l{li}_{direction}"
            # A_log: kept as Q15 (log-space decay, not quantized to INT8)
            a_log = ssm.A_log.detach().cpu().float().numpy().flatten()
            lines.append(f"static const int16_t {prefix}_a_log_q15[{len(a_log)}] = {{")
            for i in range(0, len(a_log), 8):
                chunk = a_log[i:i + 8]
                lines.append("    " + ', '.join(f'{_float_to_q15(v)}' for v in chunk) + ",")
            lines.append("};")
            lines.append("")
            total_bytes += len(a_log) * 2

            # D (skip connection)
            total_bytes += _emit_int8_array(lines, f"{prefix}_D", ssm.D)
            # dt_bias
            total_bytes += _emit_int8_array(lines, f"{prefix}_dt_bias", ssm.dt_bias)
            # Projections
            total_bytes += _emit_int8_array(lines, f"{prefix}_in_proj_w",
                                            ssm.in_proj.weight)
            total_bytes += _emit_int8_array(lines, f"{prefix}_conv1d_w",
                                            ssm.conv1d.weight)
            total_bytes += _emit_int8_array(lines, f"{prefix}_conv1d_b",
                                            ssm.conv1d.bias)
            total_bytes += _emit_int8_array(lines, f"{prefix}_x_proj_w",
                                            ssm.x_proj.weight)
            total_bytes += _emit_int8_array(lines, f"{prefix}_out_proj_w",
                                            ssm.out_proj.weight)

        # LayerNorm (kept as float — 2 * d_model values, tiny)
        norm = block.norm
        norm_w = norm.weight.detach().cpu().float().numpy()
        norm_b = norm.bias.detach().cpu().float().numpy()
        prefix_n = f"mamba_l{li}_norm"
        lines.append(f"static const float {prefix_n}_w[{len(norm_w)}] = {{")
        lines.append("    " + ', '.join(f'{v:.6f}f' for v in norm_w))
        lines.append("};")
        lines.append(f"static const float {prefix_n}_b[{len(norm_b)}] = {{")
        lines.append("    " + ', '.join(f'{v:.6f}f' for v in norm_b))
        lines.append("};")
        lines.append("")
        total_bytes += len(norm_w) * 4 + len(norm_b) * 4

    # Readout: Linear(d_model, 8)
    total_bytes += _emit_int8_array(lines, "mamba_readout_w",
                                    model.readout.weight)
    total_bytes += _emit_int8_array(lines, "mamba_readout_b",
                                    model.readout.bias)

    lines.append(f"/* Total firmware footprint: {total_bytes} bytes ({total_bytes/1024:.1f} KB) */")
    lines.append("")
    lines.append("#endif /* MAMBA_SNN_WEIGHTS_H */")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Total firmware footprint: {total_bytes/1024:.1f} KB (INT8)")
    return total_bytes


def _checkpoint_score(sens, acc, spec, min_spec=0.60):
    """Tiered checkpoint metric: sensitivity first, accuracy second.

    The SNN drives SNAC compression preset selection — false negatives
    (missing activity) waste clinical signal. False positives (calling
    quiet windows active) just cost bandwidth, not safety.

    Tier 1: sensitivity >= 0.99 — pick by accuracy (no false negatives).
    Tier 2: sensitivity < 0.99 — pick by sensitivity (still learning).

    Specificity floor prevents the degenerate "everything is active"
    solution that would lock SNAC on the lowest compression ratio.
    """
    if spec < min_spec:
        return 0.0
    if sens >= 0.99:
        return 1.0 + acc   # [1.0, 2.0] — always beats tier 2
    return sens             # [0.0, 0.99)


# ============================================================
# Async checkpoint saver (O5)
# ============================================================
#
# `torch.save` blocks the main thread for ~1-5 s on the SNN's
# 57 K-parameter ckpt (small file but optimizer-state pickling +
# disk fsync still cost). End-of-epoch latency matters because
# the next epoch's DataLoader prefetch can't start until the save
# completes, so the GPU sits idle. Hand the save off to a
# single-worker thread pool; the next epoch begins immediately.
#
# Safety: state_dict is cloned + moved to CPU before handoff so
# the worker can serialize without racing the continued training
# pass. atexit drains the executor so pending saves are durable.

import threading as _threading
import atexit as _atexit
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

_SAVE_EXECUTOR = None
_SAVE_LOCK = _threading.Lock()


def _ensure_save_executor():
    global _SAVE_EXECUTOR
    with _SAVE_LOCK:
        if _SAVE_EXECUTOR is None:
            _SAVE_EXECUTOR = _ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="snn-ckpt-saver"
            )
            _atexit.register(_SAVE_EXECUTOR.shutdown, wait=True)
    return _SAVE_EXECUTOR


def _async_save(payload: dict, path: str) -> None:
    """Schedule torch.save off the main thread.

    Expects `payload['model']` and (optionally) `payload['optimizer']`
    to be ALREADY moved to CPU + detached/cloned; passing live GPU
    tensors here invites a race with the next training step that
    mutates the same parameters.
    """
    ex = _ensure_save_executor()
    ex.submit(torch.save, payload, path)


def _state_dict_to_cpu(sd):
    """Clone a state_dict to CPU so the async saver can serialize it
    without racing the live training pass."""
    out = {}
    for k, v in sd.items():
        if hasattr(v, "detach"):
            out[k] = v.detach().to("cpu", copy=True)
        else:
            out[k] = v
    return out


# ============================================================
# pos_weight on-disk cache (O2)
# ============================================================
#
# Computing pos_weight from the full training set is expensive (one
# DataLoader pass over ~295 K windows). On subsequent runs with the
# same split manifest, the value won't have changed — cache it.
#
# Cache key: sha256(split_manifest_path bytes + split name). Survives
# manifest edits because the SHA changes when the file does.

def _pos_weight_cache_path(args, train_ds):
    """Return the on-disk path for the cached pos_weight, or None if
    we can't compute a stable key (LMA-direct only)."""
    try:
        from pathlib import Path as _Path
        import hashlib as _hl
        if not getattr(args, "split_manifest", None):
            return None
        manifest = _Path(args.split_manifest)
        if not manifest.exists():
            return None
        h = _hl.sha256()
        h.update(manifest.read_bytes())
        h.update(b"|train")  # only train split is scanned for pos_weight
        key = h.hexdigest()[:16]
        return manifest.parent / f".pos_weight_cache_{key}.json"
    except Exception:
        return None


def _pos_weight_cache_load(args, train_ds):
    p = _pos_weight_cache_path(args, train_ds)
    if p is None or not p.exists():
        return None
    try:
        import json as _json
        data = _json.loads(p.read_text())
        v = float(data.get("pos_weight"))
        if v > 0.0:
            return v
    except Exception:
        pass
    return None


def _pos_weight_cache_save(args, train_ds, pos_weight: float) -> None:
    p = _pos_weight_cache_path(args, train_ds)
    if p is None:
        return
    try:
        import json as _json
        p.write_text(_json.dumps({
            "pos_weight": pos_weight,
            "n_train_windows": len(train_ds),
            "manifest": str(args.split_manifest),
        }, indent=2))
        print(f"[*] pos_weight cached at {p}")
    except Exception as e:
        print(f"[!] pos_weight cache write failed: {e}")


# ============================================================
# Main
# ============================================================

def main():
    # G4 (2026-05-18): force spawn for DataLoader workers. The default
    # forkserver/fork start method hangs on the LMA-direct path —
    # workers inherit a half-initialised CUDA context + the lamquant_core
    # PyO3 .so, then deadlock on first .so call from the child. Spawn
    # rebuilds a clean Python process per worker (~5 s startup cost
    # amortised across the epoch).
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set (e.g. when imported under pytest)

    parser = argparse.ArgumentParser(description='Train Mamba SNN')
    parser.add_argument('--data', default=None,
                        help='Directory with _labels.npz files (legacy Q31 path)')
    parser.add_argument('--eeg-dir', default=None,
                        help='Directory with Q31 .npz or .edf files (legacy path)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Validation manifest (legacy random-split path)')
    parser.add_argument('--lma-root', type=Path, default=None,
                        help='LMA-direct: per-stem .lma archive root (one .lma '
                             'per recording bundling LML + sidecars + labels). '
                             'When set with --split-manifest, takes precedence '
                             'over --data/--eeg-dir.')
    parser.add_argument('--split-manifest', type=Path, default=None,
                        help='LMA-direct: subject-grouped split manifest from '
                             'build_snn_train_val_split.py. Required with '
                             '--lma-root.')
    parser.add_argument('--subband', action='store_true',
                        help='Use L3 subband input [21, 313] instead of raw [21, 2500]')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--config', choices=list(SNN_CONFIGS.keys()),
                        default='production', help='Training preset')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lambda-spike', type=float, default=None)
    parser.add_argument('--d-model', type=int, default=None)
    parser.add_argument('--d-state', type=int, default=None)
    parser.add_argument('--n-layers', type=int, default=None)
    parser.add_argument('--max-windows-per-file', type=int, default=None)
    parser.add_argument('--checkpoint', default=None,
                        help='Path to save best checkpoint (default: weights/snn/mamba_snn_best.pt)')
    parser.add_argument('--export', default=None,
                        help='Export best weights to C header for firmware')
    parser.add_argument('--infinite-lr', action='store_true', default=False,
                        help='WSD∞: warmup then constant peak LR forever. '
                             'Every checkpoint is shippable. Decay triggered '
                             'manually. For continual training.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (loads model + optimizer)')
    parser.add_argument('--optimizer', choices=['adamw', 'soap', 'cosmos'],
                        default='adamw',
                        help='Optimizer for the COSMOS/SOAP A/B (#71). '
                             "'adamw' (default) preserves current behavior. "
                             "'soap' uses the vendored SOAP "
                             '(ai_models/student/soap_optimizer.py). '
                             "'cosmos' is license-gated and not yet vendored.")
    parser.add_argument('--seed', type=int, default=1337,
                        help='RNG seed (torch+numpy+random) set before model '
                             'construction so A/B arms differ only in optimizer.')
    args = parser.parse_args()

    # Load preset, then apply CLI overrides
    cfg = SNN_CONFIGS[args.config]
    overrides = {}
    for field in ('epochs', 'lr', 'batch_size', 'lambda_spike',
                  'd_model', 'd_state', 'n_layers', 'max_windows_per_file'):
        val = getattr(args, field, None)
        if val is not None:
            overrides[field] = val
    if overrides:
        cfg = cfg.replace(**overrides)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"[*] Mamba SNN on {device} — preset '{cfg.name}'")
    print(cfg)

    # Seed all RNGs before model construction so A/B optimizer arms
    # (#71) differ ONLY in the optimizer — identical init weights,
    # identical data shuffling. random is imported here (rarely used
    # elsewhere in this module) to keep the seeding self-contained.
    import random as _random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[*] Seed: {args.seed} (torch + numpy + random)")

    model = MambaSNN(
        in_channels=21, d_model=cfg.d_model, d_state=cfg.d_state,
        n_layers=cfg.n_layers, use_subband=args.subband
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Parameters: {n_params:,}")
    print(f"[*] Firmware size: {model.param_size_kb(bits=8):.1f} KB (INT8), "
          f"{model.param_size_kb(bits=2):.1f} KB (W2A8)")
    assert model.param_size_kb(bits=8) <= 64, \
        f"Model exceeds 64 KB budget at INT8: {model.param_size_kb(bits=8):.1f} KB"

    # ---- Train/val split ----
    excluded = None
    manifest = None

    # LMA-direct path (preferred when --lma-root + --split-manifest provided)
    if args.lma_root is not None and args.split_manifest is not None:
        if not args.subband:
            raise ValueError(
                "LMA-direct path requires --subband (LmaDataset always "
                "returns L3 [21, 313], not raw 2500-sample windows)"
            )
        from lamquant.snn.lma_dataset import LmaDataset
        print(f"[*] LMA-direct path: lma_root={args.lma_root}")
        print(f"[*] split_manifest={args.split_manifest}")
        train_ds = LmaDataset(
            lma_root=args.lma_root, split="train",
            split_manifest_path=args.split_manifest,
            max_windows_per_file=cfg.max_windows_per_file,
        )
        val_ds = LmaDataset(
            lma_root=args.lma_root, split="val",
            split_manifest_path=args.split_manifest,
            max_windows_per_file=cfg.max_windows_per_file,
        )
        # LmaDataset is streaming (worker LRU on decoded signal).
        # Default 4 workers when an L3 disk cache is active (cache-hit
        # path uses ~150 MB/worker — multi-GB RSS pressure only on cache
        # MISS which decodes a TUEG file). Without the cache, drop to 2
        # because every __getitem__ can spike to ~12 GB peak per worker
        # on a worst-case TUEG decode.
        _default_workers = "4" if os.environ.get("L3_CACHE_DIR") else "2"
        num_workers = int(os.environ.get("LMA_NUM_WORKERS", _default_workers))
    elif args.lma_root is not None or args.split_manifest is not None:
        raise ValueError(
            "--lma-root and --split-manifest must be used together"
        )
    else:
        if args.data is None or args.eeg_dir is None:
            raise ValueError(
                "must provide either (--lma-root + --split-manifest) for "
                "LMA-direct training OR (--data + --eeg-dir) for the legacy "
                "Q31+L3 path"
            )
        if args.manifest and load_validation_manifest is not None:
            try:
                manifest = load_validation_manifest(args.manifest)
                excluded = get_excluded_windows(manifest)
                print(f"[*] Validation manifest: {len(excluded)} excluded windows")
            except FileNotFoundError as e:
                print(f"[!] {e} — falling back to random split")
        elif args.manifest and load_validation_manifest is None:
            print("[!] generate_validation_split not available, falling back to random split")

    if args.lma_root is not None:
        pass  # train_ds, val_ds, num_workers set above
    elif args.subband and args.eeg_dir:
        if manifest is not None and excluded is not None:
            train_ds = SubbandActivityDataset(
                args.data, args.eeg_dir,
                max_windows_per_file=cfg.max_windows_per_file,
                excluded_windows=excluded)
            val_ds = SubbandActivityDataset(
                args.data, args.eeg_dir,
                max_windows_per_file=cfg.max_windows_per_file)
        else:
            full_ds = SubbandActivityDataset(
                args.data, args.eeg_dir,
                max_windows_per_file=cfg.max_windows_per_file)
            n = len(full_ds)
            n_val = max(1, n // 5)
            n_train = n - n_val
            train_ds, val_ds = torch.utils.data.random_split(
                full_ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(42))
        num_workers = 0
    else:
        if manifest is not None and excluded is not None:
            train_ds = ActivityLabelDataset(args.data, eeg_dir=args.eeg_dir,
                                           excluded_windows=excluded,
                                           max_windows_per_file=cfg.max_windows_per_file)
            val_all = ActivityLabelDataset(args.data, eeg_dir=args.eeg_dir,
                                          max_windows_per_file=cfg.max_windows_per_file)
            train_sources = {s['source'] for s in train_ds.samples}
            val_all.samples = [s for s in val_all.samples
                               if s['source'] not in train_sources]
            val_ds = val_all
        else:
            full_ds = ActivityLabelDataset(args.data, eeg_dir=args.eeg_dir,
                                          max_windows_per_file=cfg.max_windows_per_file)
            n = len(full_ds)
            n_val = max(1, n // 5)
            n_train = n - n_val
            train_ds, val_ds = torch.utils.data.random_split(
                full_ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(42))
        num_workers = 2

    # LMA-direct path: prefer LmaGroupedSampler over shuffle=True so the
    # ~5 windows per LMA hit the same on-disk cache file in a row. Random
    # shuffle re-mmaps a 1.5 MB cache file per window (page-cache bound,
    # ~50 min/epoch floor); grouped order amortizes one mmap over 5 reads.
    train_sampler = None
    if args.lma_root is not None:
        try:
            from lamquant.snn.lma_dataset import LmaGroupedSampler
            train_sampler = LmaGroupedSampler(train_ds, shuffle=True, seed=42)
        except Exception as e:
            print(f"[!] LmaGroupedSampler unavailable, falling back to shuffle: {e}")

    # Worker lifecycle: persistent_workers=True keeps spawned worker
    # subprocesses alive across epochs (re-imports cost ~5-10 s/worker
    # on cold start; 4 workers × 5 epochs = wasted minutes otherwise).
    # prefetch_factor=4 (default 2) widens the pipeline buffer so GPU
    # doesn't starve when decode jitter > GPU step time.
    _dl_kwargs = {}
    if num_workers > 0:
        _dl_kwargs["persistent_workers"] = True
        _dl_kwargs["prefetch_factor"] = int(os.environ.get("LMA_PREFETCH_FACTOR", "4"))

    if train_sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  pin_memory=(device.type == 'cuda' and num_workers > 0),
                                  **_dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=(device.type == 'cuda' and num_workers > 0),
                                  **_dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=num_workers,
                            pin_memory=(device.type == 'cuda' and num_workers > 0),
                            **_dl_kwargs)
    print(f"[*] Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Compute pos_weight from actual class distribution (not hardcoded).
    # pos_weight = neg_count / pos_count tells DWB the true imbalance ratio.
    # SNN_POS_WEIGHT_SCAN_N caps how many batches are scanned (default: full
    # pass; set e.g. 200 to bound at ~25 K windows). SNN_POS_WEIGHT_OVERRIDE
    # bypasses the scan entirely with a hardcoded value (e.g. 3.0).
    _pw_override = os.environ.get("SNN_POS_WEIGHT_OVERRIDE")
    _pw_cached = None if _pw_override else _pos_weight_cache_load(args, train_ds)
    if _pw_override:
        pos_weight = float(_pw_override)
        print(f"[*] Class distribution: SCAN SKIPPED, pos_weight override={pos_weight:.2f}")
    elif _pw_cached is not None:
        pos_weight = _pw_cached
        print(f"[*] Class distribution: CACHED pos_weight={pos_weight:.2f}")
    else:
        _scan_cap = int(os.environ.get("SNN_POS_WEIGHT_SCAN_N", "0") or "0")
        pos_count = neg_count = 0
        # LMA-direct path uses iter_labels_only — skips L3 cache load
        # entirely (one NPZ extract per LMA, not per window). On 295 K
        # windows / 64 K LMAs that's ~5x fewer disk hits than the full
        # __getitem__ loop. SCAN_N cap is window-counted here.
        if args.lma_root is not None:
            from lamquant.snn.lma_dataset import iter_labels_only as _ilo
            import numpy as _np
            import time as _t
            t0 = _t.time()
            for _wi, labels_window in enumerate(_ilo(train_ds)):
                target = (labels_window >= 1).astype(_np.int64)
                pos_count += int(target.sum())
                neg_count += int((1 - target).sum())
                if _scan_cap and _wi + 1 >= _scan_cap:
                    print(f"[*] pos_weight scan capped at {_scan_cap} windows "
                          f"(SNN_POS_WEIGHT_SCAN_N) — {(_t.time()-t0):.0f}s")
                    break
        else:
            for _bi, (signal, labels) in enumerate(train_loader):
                target = (labels >= 1).long()
                pos_count += target.sum().item()
                neg_count += (1 - target).sum().item()
                if _scan_cap and _bi + 1 >= _scan_cap:
                    print(f"[*] pos_weight scan capped at {_scan_cap} batches "
                          f"(SNN_POS_WEIGHT_SCAN_N)")
                    break
        if pos_count > 0:
            pos_weight = neg_count / pos_count
            print(f"[*] Class distribution: {neg_count:,} quiet, {pos_count:,} active "
                  f"(pos_weight={pos_weight:.2f})")
        else:
            pos_weight = 3.0
            print(f"[!] No positive examples — using default pos_weight=3.0")
        # Persist the result so re-runs skip the scan. Only when the
        # scan saw the full dataset — capped scans (SNN_POS_WEIGHT_SCAN_N)
        # produce an estimate that would poison the cache on reload.
        if not _scan_cap:
            _pos_weight_cache_save(args, train_ds, pos_weight)
        else:
            print(f"[*] pos_weight cache NOT written (scan was capped at {_scan_cap})")

    # Optimizer factory (#71 COSMOS/SOAP A/B). The A/B arms keep the
    # SAME cfg.lr / cfg.weight_decay so the comparison isolates the
    # optimizer — SOAP's own default lr (3e-3) is deliberately NOT used.
    # WSDScheduler wraps whichever optimizer is built (below).
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=cfg.weight_decay)
    elif args.optimizer == 'soap':
        sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'student'))
        from soap_optimizer import SOAP
        optimizer = SOAP(model.parameters(), lr=cfg.lr,
                         weight_decay=cfg.weight_decay)
    elif args.optimizer == 'cosmos':
        raise NotImplementedError(
            "COSMOS optimizer is license-gated and NOT vendored. To enable "
            "the COSMOS arm: (1) vendor the source from "
            "https://github.com/lliu606/COSMOS into ai_models/student/, "
            "(2) confirm its OSI-approved license is compatible with this "
            "repo (GPL-3.0-or-later), and (3) pin the exact upstream commit "
            "SHA in pccp/registry.yaml before use. See MEMORY: "
            "project_cosmos_optimizer_ab.md.")
    else:
        # argparse 'choices' guards this; defensive for future edits.
        raise ValueError(f"Unknown --optimizer {args.optimizer!r}")
    print(f"[*] Optimizer: {args.optimizer} "
          f"(lr={cfg.lr:.0e}, weight_decay={cfg.weight_decay:.0e})")
    # LR schedule (W3, 2026-05-21): cosine warmup → WSD stable → cosine decay.
    # Single continuous schedule; same shape used by train_joint.py so the
    # SNN and joint pair share one curve.
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'student'))
    from train_joint import WSDScheduler
    if args.infinite_lr:
        # WSD∞ — manual decay trigger; production-style continual training.
        scheduler = WSDScheduler(
            optimizer, total_epochs=cfg.epochs, peak_lr=cfg.lr,
            warmup_frac=0.05, decay_frac=0.0, min_lr=cfg.lr_min,
            warmup_kind="cosine")
        print(f"[*] Schedule: cosine-warmup → WSD∞ stable "
              f"(warmup={scheduler.warmup_epochs}ep, stable=∞)")
    else:
        # Standard WSD with fixed cosine decay at the tail.
        scheduler = WSDScheduler(
            optimizer, total_epochs=cfg.epochs, peak_lr=cfg.lr,
            warmup_frac=0.05, decay_frac=0.10, min_lr=cfg.lr_min,
            warmup_kind="cosine")
        print(f"[*] Schedule: cosine-warmup → WSD stable → cosine decay "
              f"(warmup={scheduler.warmup_epochs}ep, "
              f"decay={scheduler.decay_epochs}ep, "
              f"{cfg.lr:.0e} → {cfg.lr_min:.0e})")

    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        # Contains non-tensor metadata
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"[*] Resumed from {args.resume} (epoch {start_epoch})")
        try:
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                print(f"    Optimizer state restored")
        except Exception:
            pass

    best_score = 0.0
    best_sens = 0.0
    best_acc = 0.0
    best_spec = 0.0
    best_epoch = 0
    save_dir = os.path.join(ROOT_DIR, 'weights', 'snn')
    os.makedirs(save_dir, exist_ok=True)
    save_path = args.checkpoint or os.path.join(save_dir, 'mamba_snn_best.pt')

    import time as _time
    n_batches = len(train_loader)
    print(f"[*] Training: {cfg.epochs} epochs x {n_batches} batches (bs={cfg.batch_size})")
    train_start = _time.time()

    for epoch in range(start_epoch, cfg.epochs):
        ep_start = _time.time()
        loss, acc, sens, sr = train_epoch(
            model, train_loader, optimizer, device, cfg.lambda_spike,
            pos_weight=pos_weight)
        if hasattr(scheduler, 'step'):
            scheduler.step()
        ep_sec = _time.time() - ep_start

        # Validation every epoch
        val_acc, val_sens, val_spec, val_fnr = validate(model, val_loader, device)

        score = _checkpoint_score(val_sens, val_acc, val_spec)
        improved = ''
        if score > best_score:
            best_score = score
            best_sens = val_sens
            best_acc = val_acc
            best_spec = val_spec
            best_epoch = epoch + 1
            improved = ' *BEST*'
            _async_save({
                'model': _state_dict_to_cpu(model.state_dict()),
                'optimizer': _state_dict_to_cpu(optimizer.state_dict()),
                'epoch': epoch + 1,
                'sensitivity': val_sens,
                'accuracy': val_acc,
                'specificity': val_spec,
                'fnr': val_fnr,
                'score': score,
                'config': cfg.to_dict(),
            }, save_path)

        elapsed = _time.time() - train_start
        epochs_done = epoch - start_epoch + 1
        remaining = elapsed / max(epochs_done, 1) * (cfg.epochs - epoch - 1)
        eta_h, eta_m = divmod(int(remaining), 3600)
        eta_m //= 60

        ram = _rss_gb()
        gpu_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        print(f"E{epoch+1:3d}/{cfg.epochs}  L={loss:.4f}  "
              f"train[A={acc:.3f} S={sens:.3f}]  "
              f"val[A={val_acc:.3f} S={val_sens:.3f} Sp={val_spec:.3f} FNR={val_fnr:.4f}]  "
              f"sr={sr:.4f}  {ep_sec:.0f}s  RAM={ram:.1f}G  GPU={gpu_mb:.0f}M  "
              f"ETA={eta_h}h{eta_m:02d}m{improved}")

    # Save final with metrics
    final_path = os.path.join(save_dir, f'mamba_snn_{cfg.name}_{cfg.epochs}_completed.pt')
    final_acc, final_sens, final_spec, final_fnr = validate(model, val_loader, device)
    torch.save({
        'model': model.state_dict(),
        'epoch': cfg.epochs,
        'sensitivity': final_sens,
        'accuracy': final_acc,
        'specificity': final_spec,
        'fnr': final_fnr,
        'config': cfg.to_dict(),
    }, final_path)

    total_h = (_time.time() - train_start) / 3600
    print(f"\n[*] Training complete in {total_h:.1f}h")
    print(f"[*] Best (epoch {best_epoch}):  "
          f"acc={best_acc:.4f}  sens={best_sens:.4f}  spec={best_spec:.4f}  "
          f"score={best_score:.4f}")
    print(f"[*] Final (epoch {cfg.epochs}):  "
          f"acc={final_acc:.4f}  sens={final_sens:.4f}  spec={final_spec:.4f}")
    print(f"[*] Saved: {save_path} (best), {final_path} (final)")

    # Auto-export to firmware C header when sensitivity target met
    export_path = args.export or os.path.join(
        ROOT_DIR, 'firmware', 'firmware_export', 'mamba_snn_weights.h')
    def _load_best(p):
        try:
            return torch.load(p, map_location='cpu', weights_only=True)
        except Exception:
            return torch.load(p, map_location='cpu', weights_only=False)

    if best_sens >= 0.99:
        # Reload best checkpoint for export
        best_state = _load_best(save_path)
        model.load_state_dict(best_state['model'])
        export_mamba_weights(model, export_path)
        print(f"[*] Sensitivity >= 0.99 — auto-exported to {export_path}")
    elif args.export:
        # Explicit --export flag: export regardless of sensitivity
        best_state = _load_best(save_path)
        model.load_state_dict(best_state['model'])
        export_mamba_weights(model, export_path)


if __name__ == '__main__':
    main()
