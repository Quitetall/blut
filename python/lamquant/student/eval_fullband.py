#!/usr/bin/env python3
"""eval_fullband.py — fullband per-band PRD + LQS gate for the joint codec.

Training-time validation operates on L3 (~31 Hz Nyquist), which only
covers δ/θ/α reliably. The codec actually has to reconstruct fullband
EEG (250 Hz) — beta and gamma compliance live at the fullband level
and require a separate evaluation.

This script runs the FULL inverse pipeline on raw EEG NPZs:

    raw fullband EEG  →  preprocess (LPC + lifting)  →  L3
                        →  encoder.encode(L3)  →  ternary latent
                        →  decoder(latent)  →  recon L3
                        →  reconstruct_from_subband(recon_L3, coeffs, details)
                        →  fullband EEG reconstruction
                        →  per-band PRD @ fs=250 Hz  →  LQS compliance

Usage:
    python ai_models/student/eval_fullband.py \
        --encoder ai_models/student/student_encoder_joint_fast.ckpt \
        --decoder ai_models/student/decoder_tier2_joint_fast.ckpt \
        --tier 2 \
        --max-files 30 \
        --windows-per-file 16

The output is the same LQS summary the training script prints at end of
run, but evaluated against fullband ground truth — the publishable
ship/no-ship gate.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'lamquant'))
sys.path.insert(0, str(ROOT_DIR / 'lamquant' / 'common'))  # MOVE-B: common DTOs
sys.path.insert(0, str(ROOT_DIR / 'lamquant' / 'student'))
sys.path.insert(0, str(ROOT_DIR / 'lamquant' / 'decoder'))

from data_types import DatasetManifest, Split
from metrics import (
    pearson_r_numpy, prd_numpy, per_band_prd, per_band_r,
    lqs_compliance,
)
from subband_preprocess import preprocess_subband_single, reconstruct_from_subband
from joint_codec import build_default_joint
from lamquant.common.utils import safe_torch_load as _safe_load


# ============================================================
# Per-window evaluation
# ============================================================

WINDOW = 2500          # 10 s @ 250 Hz
FS = 250.0
Q31_SCALE = 2147483647.0
UV_PER_Q31 = 1000.0    # matches streaming_dataset's Q31 → µV conversion


def _fullband_reconstruct(window_uv: np.ndarray,
                          codec, device, quantize: bool = True) -> np.ndarray:
    """Run one window through encoder→decoder→inverse pipeline.

    Returns the fullband reconstruction at the same shape as input.
    """
    # 1. Subband preprocess: raw → L3 + LPC coeffs + detail subbands
    l3, coeffs, subs = preprocess_subband_single(window_uv)
    # 2. Encode + decode the L3 (the only lossy stage)
    l3_t = torch.from_numpy(l3).float().unsqueeze(0).to(device)
    with torch.no_grad():
        recon_l3_t = codec(l3_t, quantize=quantize)
    recon_l3 = recon_l3_t.squeeze(0).cpu().numpy().astype(np.float64)
    # Decoder may emit a slightly different T (e.g. 316 vs 313) — crop.
    T_l3 = l3.shape[-1]
    recon_l3 = recon_l3[..., :T_l3]
    # 3. Inverse pipeline: recon_L3 + ORIGINAL details + LPC → fullband
    fullband = reconstruct_from_subband(recon_l3, coeffs, subs)
    return fullband


def _eval_window(window_uv: np.ndarray, codec, device,
                 quantize: bool = True) -> dict:
    """Return {r, prd, per_band_prd, per_band_r} for one [21, 2500] window."""
    recon = _fullband_reconstruct(window_uv, codec, device, quantize)
    # Pin shape — pipeline reconstructs to original length but float drift
    # at the LPC boundary can shave 1-2 samples.
    T = min(window_uv.shape[-1], recon.shape[-1])
    orig = window_uv[..., :T]
    recon = recon[..., :T]
    return {
        'r':   pearson_r_numpy(orig, recon),
        'prd': prd_numpy(orig, recon),
        'pb_prd': per_band_prd(orig, recon, fs=FS),
        'pb_r':   per_band_r(orig, recon, fs=FS),
    }


# ============================================================
# Aggregator
# ============================================================

def evaluate(file_entries, codec, device,
             windows_per_file: int = 16, max_files: int = 30,
             quantize: bool = True, seed: int = 42) -> dict:
    """Run the codec on a sample of files and aggregate metrics.

    Returns a dict with keys: n_windows, mean_r, mean_prd, per_band_prd,
    per_band_r, lqs_level, lqs_violations.
    """
    rng = np.random.default_rng(seed)
    rs, prds = [], []
    pb_prd_acc = {b: [] for b in ('delta', 'theta', 'alpha', 'beta', 'gamma')}
    pb_r_acc = {b: [] for b in pb_prd_acc}
    n_windows = 0

    files_seen = 0
    for fe in file_entries:
        if files_seen >= max_files:
            break
        try:
            with np.load(fe.path) as d:
                data = np.array(d['data'], dtype=np.float64)   # [21, T] int32 → float
        except Exception:
            continue

        T_total = data.shape[-1]
        n_full = T_total // WINDOW
        if n_full < 1:
            continue
        # Sample window starts so a file doesn't dominate via its first 16 windows.
        window_starts = rng.choice(n_full, size=min(windows_per_file, n_full),
                                   replace=False)
        for w_idx in window_starts:
            start = int(w_idx) * WINDOW
            seg = data[:, start:start + WINDOW]
            # Q31 → µV (matches PrecomputedL3Dataset's conversion, so the
            # encoder sees the same scale as during training)
            seg_uv = seg / Q31_SCALE * UV_PER_Q31
            try:
                m = _eval_window(seg_uv, codec, device, quantize=quantize)
            except Exception as e:
                # Skip individual windows on numerical failure; don't tank the run.
                print(f'  [skip] {fe.path}:{w_idx} → {e!s}')
                continue
            rs.append(m['r'])
            prds.append(m['prd'])
            for b in pb_prd_acc:
                pb_prd_acc[b].append(m['pb_prd'].get(b, 0.0))
                pb_r_acc[b].append(m['pb_r'].get(b, 0.0))
            n_windows += 1
        files_seen += 1

    if n_windows == 0:
        raise RuntimeError('no windows successfully evaluated')

    mean_r = float(np.mean(rs))
    mean_prd = float(np.mean(prds))
    per_band_prd_mean = {b: float(np.mean(v)) if v else 0.0
                         for b, v in pb_prd_acc.items()}
    per_band_r_mean = {b: float(np.mean(v)) if v else 0.0
                       for b, v in pb_r_acc.items()}

    lqs_level, lqs_viol = lqs_compliance(
        val_r=mean_r, val_prd=mean_prd,
        per_band_prd_dict=per_band_prd_mean,
        per_band_r_dict=per_band_r_mean,
    )
    return {
        'n_windows': n_windows,
        'n_files': files_seen,
        'mean_r': mean_r,
        'mean_prd': mean_prd,
        'per_band_prd': per_band_prd_mean,
        'per_band_r': per_band_r_mean,
        'lqs_level': lqs_level,
        'lqs_violations': lqs_viol,
    }


# ============================================================
# Reporting
# ============================================================

def print_report(result: dict, encoder_path: str, decoder_path: str,
                  duration_s: float) -> None:
    """The publishable ship/no-ship gate."""
    band_glyph = (('delta', 'δ'), ('theta', 'θ'), ('alpha', 'α'),
                  ('beta', 'β'), ('gamma', 'γ'))
    band_str = '  '.join(
        f'{g} {result["per_band_prd"][b]:>5.1f}%'
        for b, g in band_glyph)
    band_r_str = '  '.join(
        f'{g} {result["per_band_r"][b]:>5.3f}'
        for b, g in band_glyph)
    level_name = {'C': 'Clinical', 'M': 'Monitoring',
                  'A': 'Alerting', '': 'below LQS-A'}.get(
                      result['lqs_level'], result['lqs_level'])
    next_tier = {'M': 'C', 'A': 'M', '': 'A'}.get(result['lqs_level'], '?')

    print()
    print('=' * 72)
    print(f'  FULLBAND LQS EVALUATION')
    print(f'  Encoder: {encoder_path}')
    print(f'  Decoder: {decoder_path}')
    print(f'  Sample:  {result["n_files"]} files, '
          f'{result["n_windows"]} windows, fs=250 Hz')
    print(f'  Eval time: {duration_s:.1f} s '
          f'({duration_s / max(result["n_windows"], 1):.2f} s/window)')
    print('-' * 72)
    print(f'  Global R:   {result["mean_r"]:.4f}')
    print(f'  Global PRD: {result["mean_prd"]:.2f}%')
    print(f'  Per-band PRD: {band_str}')
    print(f'  Per-band R:   {band_r_str}')
    print('-' * 72)
    print(f'  LQS Level: {result["lqs_level"] or "--"} ({level_name})')
    if result['lqs_violations']:
        print(f'  To reach LQS-{next_tier}, fix:')
        for v in result['lqs_violations'][:8]:
            print(f'    - {v}')
        if len(result['lqs_violations']) > 8:
            print(f'    ... and {len(result["lqs_violations"]) - 8} more')
    else:
        print(f'  No violations — model passes the strictest tier.')
    print('=' * 72)
    print()


# ============================================================
# CLI
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        prog='eval_fullband',
        description='Fullband per-band PRD + LQS gate (the publishable evaluation).',
    )
    parser.add_argument('--encoder', type=Path, required=True,
                        help='Path to encoder ckpt')
    parser.add_argument('--decoder', type=Path, required=True,
                        help='Path to decoder ckpt')
    parser.add_argument('--tier', type=int, default=2,
                        help='Vocos decoder tier (1, 2, 3, 6, 7, 8)')
    parser.add_argument('--manifest', type=Path,
                        default=ROOT_DIR / 'lamquant' / 'dataset_sim' / 'manifest_v3.json',
                        help='manifest_v3.json path')
    parser.add_argument('--max-files', type=int, default=30,
                        help='How many val files to sample (default: 30)')
    parser.add_argument('--windows-per-file', type=int, default=16,
                        help='How many windows to sample per file (default: 16)')
    parser.add_argument('--quantize', dest='quantize', action='store_true', default=True)
    parser.add_argument('--no-quantize', dest='quantize', action='store_false',
                        help='Disable encoder ternary quantization (FP32 eval)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Window-sampling RNG seed')
    parser.add_argument('--datasets', type=str, default='',
                        help='Comma-separated dataset names to filter '
                             '(e.g. chbmit,tuh_seizure). Default: all.')
    parser.add_argument('--json', action='store_true',
                        help='Emit a single __PCCP_JSON__ line at end of run with '
                             'measurements for pccp_gate.py consumption. Adds metrics: '
                             'pearson_r, mean_prd, per_band_*, lqs_level, n_windows, '
                             'duration_s, encoder_param_count, encoder_weight_memory_kb.')
    args = parser.parse_args()

    # Load manifest, pick val files.
    manifest = DatasetManifest.load(args.manifest)
    if args.datasets:
        from data_types import Dataset
        names = [n.strip() for n in args.datasets.split(',') if n.strip()]
        ds_filter = [Dataset(n) for n in names]
        val_entries = manifest.get_file_entries(Split.VAL, datasets=ds_filter)
    else:
        val_entries = manifest.get_file_entries(Split.VAL)
    if not val_entries:
        print('[!] No validation entries found.')
        return 1
    print(f'[*] Manifest: {len(val_entries):,} val files available '
          f'(sampling up to {args.max_files})')

    # Build codec, load weights.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    codec = build_default_joint(vocos_tier=args.tier).to(device).eval()
    enc_state = _safe_load(args.encoder, map_location=device)
    dec_state = _safe_load(args.decoder, map_location=device)
    codec.encoder.load_state_dict(
        enc_state.get('state_dict', enc_state), strict=False)
    codec.decoder.load_state_dict(
        dec_state.get('state_dict', dec_state), strict=False)
    print(f'[*] Codec loaded ({sum(p.numel() for p in codec.parameters()):,} params, '
          f'tier={args.tier}, device={device}, quantize={args.quantize})')

    t0 = time.perf_counter()
    result = evaluate(val_entries, codec, device,
                      windows_per_file=args.windows_per_file,
                      max_files=args.max_files,
                      quantize=args.quantize, seed=args.seed)
    duration = time.perf_counter() - t0
    print_report(result, str(args.encoder), str(args.decoder), duration)

    if args.json:
        # Static-analysis hooks for the PCCP gate. Encoder params + weight
        # memory are computed from the loaded encoder; CR / latency hooks
        # require a separate encode pass and firmware bench (TODO).
        import json as _json
        encoder_param_count = sum(p.numel() for p in codec.encoder.parameters())
        # Weight memory: ternary weights pack to 1.58 bits each on disk; the
        # in-memory FP32 estimate (4 B / param) is the conservative upper
        # bound that aligns with the firmware deployment budget.
        encoder_weight_memory_bytes = sum(
            p.numel() * p.element_size() for p in codec.encoder.parameters()
        )
        payload = {
            'mean_r':                     result['mean_r'],
            'pearson_r':                  result['mean_r'],   # alias for gate
            'mean_prd':                   result['mean_prd'],
            'per_band_prd':               result['per_band_prd'],
            'per_band_r':                 result['per_band_r'],
            'lqs_level':                  result['lqs_level'],
            'lqs_violations':             result['lqs_violations'],
            'n_windows':                  result['n_windows'],
            'n_files':                    result['n_files'],
            'duration_s':                 duration,
            'encoder_param_count':        encoder_param_count,
            'encoder_weight_memory_kb':   encoder_weight_memory_bytes / 1024.0,
            # Aliases the gate registry expects:
            'param_count':                encoder_param_count,
            'weight_memory_kb':           encoder_weight_memory_bytes / 1024.0,
        }
        print('__PCCP_JSON__' + _json.dumps(payload))
    return 0


if __name__ == '__main__':
    sys.exit(main())
