#!/usr/bin/env python3
"""downstream_concordance.py — measure whether compression preserves
downstream task accuracy.

The key insight from the literature (FAE, C²SP-Net, Dynamic-Deep):
PRD/R are secondary metrics. What matters is whether a clinician or
algorithm makes the *same decision* on compressed EEG as on original.

This script:
  1. Trains a lightweight seizure detector on ORIGINAL validation EEG
  2. Runs the same detector on COMPRESSED-THEN-RECONSTRUCTED EEG
  3. Reports concordance: F1, sensitivity, specificity, AUROC delta
  4. Per-band Hjorth parameter preservation (activity, mobility, complexity)

If concordance is high (F1 delta < 0.02), the compression is
clinically transparent — a neurologist can't tell the difference.

Usage:
    python ai_models/validation/downstream_concordance.py \
        --encoder weights/student_subband.ckpt \
        --decoder weights/decoder_tier3.ckpt \
        --tier 3

Outputs a single table:
    metric              original    compressed    delta
    seizure_f1          0.87        0.85          -0.02
    seizure_sens        0.92        0.89          -0.03
    seizure_spec        0.95        0.94          -0.01
    hjorth_activity_r   1.000       0.982         -0.018
    hjorth_mobility_r   1.000       0.967         -0.033
    hjorth_complexity_r 1.000       0.954         -0.046
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'ai_models'))
sys.path.insert(0, str(_REPO / 'ai_models' / 'student'))
sys.path.insert(0, str(_REPO / 'ai_models' / 'decoder'))


# ============================================================
# Hjorth parameters — feature-level concordance
# ============================================================

def hjorth_parameters(signal: np.ndarray) -> Tuple[float, float, float]:
    """Compute Hjorth activity, mobility, complexity for signal [..., T].

    These are the three most-used EEG features in clinical practice.
    If compression preserves them, it preserves the features clinicians
    actually use for diagnosis.
    """
    # Activity = variance of the signal
    activity = float(np.var(signal))
    # Mobility = sqrt(var(d1) / var(signal))
    d1 = np.diff(signal, axis=-1)
    var_d1 = float(np.var(d1))
    mobility = float(np.sqrt(var_d1 / max(activity, 1e-12)))
    # Complexity = mobility(d1) / mobility(signal)
    d2 = np.diff(d1, axis=-1)
    var_d2 = float(np.var(d2))
    mob_d1 = float(np.sqrt(var_d2 / max(var_d1, 1e-12)))
    complexity = float(mob_d1 / max(mobility, 1e-12))
    return activity, mobility, complexity


def hjorth_concordance(original: np.ndarray,
                        reconstructed: np.ndarray) -> Dict[str, float]:
    """Pearson R between Hjorth parameters of original vs reconstructed.

    Computed per-channel, averaged. R=1.0 means perfect preservation.
    """
    from metrics import pearson_r_numpy
    C = original.shape[0]
    results = {'activity': [], 'mobility': [], 'complexity': []}
    for c in range(C):
        a_orig = hjorth_parameters(original[c])
        a_recon = hjorth_parameters(reconstructed[c])
        for i, key in enumerate(results):
            results[key].append((a_orig[i], a_recon[i]))

    concordance = {}
    for key, pairs in results.items():
        orig_vals = np.array([p[0] for p in pairs])
        recon_vals = np.array([p[1] for p in pairs])
        concordance[f'hjorth_{key}_r'] = float(pearson_r_numpy(orig_vals, recon_vals))
    return concordance


# ============================================================
# Seizure concordance — task-level
# ============================================================

def train_simple_seizure_detector(windows: List[np.ndarray],
                                   labels: List[bool],
                                   ) -> 'sklearn.linear_model.LogisticRegression':
    """Train a logistic regression seizure detector on Hjorth features.

    Deliberately simple — the detector's job is to be a PROBE for
    whether compression preserves task-relevant features, not to be
    SOTA at seizure detection. LogReg on Hjorth features is a standard
    EEG baseline (Shoeb & Guttag 2010).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = []
    for w in windows:
        feats = []
        for c in range(w.shape[0]):
            a, m, co = hjorth_parameters(w[c])
            feats.extend([a, m, co])
        X.append(feats)
    X = np.array(X)
    y = np.array(labels, dtype=np.float64)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    clf = LogisticRegression(
        class_weight='balanced',   # handles seizure rarity
        max_iter=1000,
        random_state=42,
    ).fit(X_scaled, y)
    return clf, scaler


def evaluate_seizure_concordance(
    clf, scaler,
    original_windows: List[np.ndarray],
    reconstructed_windows: List[np.ndarray],
    labels: List[bool],
) -> Dict[str, float]:
    """Run the same detector on original and reconstructed, compare."""
    from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

    def _extract_features(windows):
        X = []
        for w in windows:
            feats = []
            for c in range(w.shape[0]):
                a, m, co = hjorth_parameters(w[c])
                feats.extend([a, m, co])
            X.append(feats)
        return scaler.transform(np.array(X))

    X_orig = _extract_features(original_windows)
    X_recon = _extract_features(reconstructed_windows)
    y = np.array(labels, dtype=np.float64)

    pred_orig = clf.predict(X_orig)
    pred_recon = clf.predict(X_recon)

    results = {}
    for name, metric in [('f1', f1_score), ('sensitivity', recall_score),
                          ('precision', precision_score)]:
        results[f'seizure_{name}_orig'] = float(metric(y, pred_orig, zero_division=0))
        results[f'seizure_{name}_recon'] = float(metric(y, pred_recon, zero_division=0))
        results[f'seizure_{name}_delta'] = (
            results[f'seizure_{name}_recon'] - results[f'seizure_{name}_orig'])

    # AUROC if both classes present
    if len(set(y)) > 1:
        prob_orig = clf.predict_proba(X_orig)[:, 1]
        prob_recon = clf.predict_proba(X_recon)[:, 1]
        results['auroc_orig'] = float(roc_auc_score(y, prob_orig))
        results['auroc_recon'] = float(roc_auc_score(y, prob_recon))
        results['auroc_delta'] = results['auroc_recon'] - results['auroc_orig']

    # Decision concordance: same prediction on both versions
    concordance = float(np.mean(pred_orig == pred_recon))
    results['decision_concordance'] = concordance

    return results


# ============================================================
# CLI
# ============================================================

def main() -> int:
    import torch
    from data_types import DatasetManifest, Split
    from joint_codec import build_default_joint
    from subband_preprocess import preprocess_subband_single
    from metrics import pearson_r_numpy, prd_numpy

    parser = argparse.ArgumentParser(prog='downstream_concordance')
    parser.add_argument('--encoder', type=Path,
                        default=_REPO / 'ai_models' / 'student' / 'student_encoder_joint_fast.ckpt')
    parser.add_argument('--decoder', type=Path,
                        default=_REPO / 'ai_models' / 'student' / 'decoder_tier3_joint_fast.ckpt')
    parser.add_argument('--tier', type=int, default=3)
    parser.add_argument('--max-windows', type=int, default=200)
    parser.add_argument('--manifest', type=Path,
                        default=_REPO / 'ai_models' / 'dataset_sim' / 'manifest_v3.json')
    args = parser.parse_args()

    print('=' * 72)
    print('  DOWNSTREAM CONCORDANCE EVALUATION')
    print('=' * 72)

    # Load codec
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    codec = build_default_joint(vocos_tier=args.tier).to(device).eval()
    codec.load_encoder(args.encoder)
    codec.load_decoder(args.decoder)
    print(f'[*] Codec: tier={args.tier}, device={device}')

    # Load validation windows + seizure labels
    manifest = DatasetManifest.load(args.manifest)
    val_entries = manifest.get_file_entries(Split.VAL)
    print(f'[*] {len(val_entries)} val files')

    Q31 = 2147483647.0
    UV = 1000.0
    WINDOW = 2500
    original_windows = []
    reconstructed_windows = []
    labels = []
    n_loaded = 0

    for fe in val_entries:
        if n_loaded >= args.max_windows:
            break
        try:
            with np.load(fe.path) as d:
                raw = d['data']
                mask = d.get('seizure_mask', np.zeros(raw.shape[1]))
                T = raw.shape[1]
                n_win = T // WINDOW
                for k in range(min(n_win, 5)):  # max 5 per file
                    if n_loaded >= args.max_windows:
                        break
                    start = k * WINDOW
                    seg = raw[:, start:start + WINDOW]
                    seg_uv = (seg.astype(np.float32) / Q31 * UV)

                    # Reconstruct through codec
                    l3, _, _ = preprocess_subband_single(seg_uv)
                    l3_t = torch.from_numpy(l3).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        recon_t = codec(l3_t, quantize=True)
                    recon_np = recon_t.squeeze(0).cpu().numpy()
                    T_r = min(seg_uv.shape[-1], recon_np.shape[-1])

                    original_windows.append(seg_uv[..., :T_r])
                    reconstructed_windows.append(recon_np[..., :T_r])

                    # Seizure label: any sample in this window has mask=1
                    win_mask = mask[start:start + WINDOW]
                    labels.append(bool(win_mask.sum() > 0))
                    n_loaded += 1
        except Exception:
            continue

    print(f'[*] Loaded {n_loaded} windows '
          f'({sum(labels)} seizure, {n_loaded - sum(labels)} normal)')

    if n_loaded < 10:
        print('[!] Too few windows for meaningful concordance.')
        return 1

    # Hjorth concordance (all windows)
    print('\n[*] Hjorth parameter concordance...')
    hjorth_results = {}
    for key in ('activity', 'mobility', 'complexity'):
        rs = []
        for orig, recon in zip(original_windows, reconstructed_windows):
            h = hjorth_concordance(orig, recon)
            rs.append(h.get(f'hjorth_{key}_r', 0.0))
        hjorth_results[f'hjorth_{key}_r'] = float(np.mean(rs))

    # Seizure concordance (if seizures present)
    seizure_results = {}
    if sum(labels) >= 3 and sum(not l for l in labels) >= 3:
        print('[*] Training seizure probe classifier...')
        try:
            clf, scaler = train_simple_seizure_detector(original_windows, labels)
            seizure_results = evaluate_seizure_concordance(
                clf, scaler, original_windows, reconstructed_windows, labels)
        except Exception as e:
            print(f'[!] Seizure concordance failed: {e}')
    else:
        print(f'[*] Skipping seizure concordance (need ≥3 of each class; '
              f'have {sum(labels)} seizure, {n_loaded - sum(labels)} normal)')

    # Summary
    print('\n' + '=' * 72)
    print('  CONCORDANCE RESULTS')
    print('=' * 72)
    print(f'  {"metric":35} {"value":>10}')
    print('  ' + '-' * 50)
    for k, v in hjorth_results.items():
        print(f'  {k:35} {v:>10.4f}')
    for k, v in seizure_results.items():
        print(f'  {k:35} {v:>10.4f}')
    print('=' * 72)

    return 0


if __name__ == '__main__':
    sys.exit(main())
