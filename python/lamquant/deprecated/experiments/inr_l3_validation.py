#!/usr/bin/env python3
"""
INR L3 Subband Validation: SIREN on the lifting DWT approximation subband.

Tests the hybrid architecture: SIREN for smooth L3 base + existing entropy
coding for detail subbands. This is the viable path identified from the
full-bandwidth experiment (single SIREN fails on beta/gamma).

Default: 1 kHz sample rate (10,000 samples/ch, L3 = [21, 1250] after stride-8).
Also tests 250 Hz (clinical), 500 Hz, 2 kHz (research).

The L3 subband at 1 kHz captures 0-62.5 Hz (delta through low gamma).
Detail subbands carry 62.5-500 Hz content separately via lossless coding.

Usage:
    python ai_models/experiments/inr_l3_validation.py
    python ai_models/experiments/inr_l3_validation.py --sample-rate 250
    python ai_models/experiments/inr_l3_validation.py --sample-rate 2000 --epochs 2000
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# Lifting DWT (3-level Le Gall 5/3) — matches production codec
# ============================================================

def lifting_forward_1level(signal: np.ndarray) -> tuple:
    """Single-level Le Gall 5/3 lifting DWT.

    Input:  [C, T]
    Output: (approx [C, ceil(T/2)], detail [C, floor(T/2)])
    """
    C, T = signal.shape
    # Pad to even length
    if T % 2 == 1:
        signal = np.pad(signal, ((0, 0), (0, 1)), mode='reflect')
        T += 1

    even = signal[:, 0::2]  # [C, T/2]
    odd = signal[:, 1::2]   # [C, T/2]
    N = min(even.shape[1], odd.shape[1])
    even = even[:, :N]
    odd = odd[:, :N]

    # Predict: detail = odd - average of even neighbors
    even_left = even
    even_right = np.pad(even[:, 1:], ((0, 0), (0, 1)), mode='edge')
    detail = odd - (even_left + even_right) / 2.0

    # Update: approx = even + average of detail neighbors
    detail_left = np.pad(detail[:, :-1], ((0, 0), (1, 0)), mode='edge')
    approx = even + (detail_left + detail) / 4.0

    return approx, detail


def lifting_3level(signal: np.ndarray) -> tuple:
    """3-level lifting DWT. Returns (L3_approx, [L1_detail, L2_detail, L3_detail]).

    Input:  [C, T]
    Output: L3 approx [C, ~T/8], list of details at each level
    """
    details = []

    # Level 1
    approx, d1 = lifting_forward_1level(signal)
    details.append(d1)

    # Level 2
    approx, d2 = lifting_forward_1level(approx)
    details.append(d2)

    # Level 3
    approx, d3 = lifting_forward_1level(approx)
    details.append(d3)

    return approx, details


# ============================================================
# SIREN (from inr_validation.py, adapted for variable T)
# ============================================================

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SIREN(nn.Module):
    """SIREN for L3 subband reconstruction. Maps time -> all channels."""

    def __init__(self, hidden_dim=64, n_layers=2, n_channels=21, omega_0=30.0):
        super().__init__()
        layers = []
        layers.append(SirenLayer(1, hidden_dim, omega=omega_0, is_first=True))
        for _ in range(n_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega=omega_0))
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, n_channels)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.output.weight.uniform_(-bound, bound)

    def forward(self, coords):
        return self.output(self.net(coords))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def build_siren_for_budget(target_params, n_channels=21, n_layers=2, omega_0=30.0):
    """Build SIREN with approximately target_params parameters."""
    lo, hi = 4, 512
    best_dim = lo
    for _ in range(20):
        mid = (lo + hi) // 2
        model = SIREN(hidden_dim=mid, n_layers=n_layers, n_channels=n_channels,
                      omega_0=omega_0)
        if model.param_count() <= target_params:
            best_dim = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return SIREN(hidden_dim=best_dim, n_layers=n_layers, n_channels=n_channels,
                 omega_0=omega_0)


# ============================================================
# Signal Generation (multi-rate)
# ============================================================

def generate_eeg(sample_rate: int = 1000, duration: float = 10.0,
                 n_channels: int = 21, seed: int = None) -> np.ndarray:
    """Generate realistic multi-rate EEG.

    Returns [n_channels, T] float32 where T = sample_rate * duration.
    Includes all clinical bands up to Nyquist/2.
    """
    rng = np.random.RandomState(seed)
    T = int(sample_rate * duration)
    t = np.linspace(0, duration, T)
    nyquist = sample_rate / 2.0
    sig = np.zeros((n_channels, T), dtype=np.float32)

    for ch in range(n_channels):
        # Delta (0.5-4 Hz) — always present
        for _ in range(rng.randint(1, 3)):
            f = rng.uniform(0.5, 4)
            sig[ch] += rng.uniform(10, 30) * np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))

        # Theta (4-8 Hz)
        for _ in range(rng.randint(1, 3)):
            f = rng.uniform(4, 8)
            sig[ch] += rng.uniform(5, 15) * np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))

        # Alpha (8-13 Hz) — posterior dominant
        if ch in [14, 15, 18, 19, 20]:  # occipital channels
            sig[ch] += rng.uniform(15, 30) * np.sin(2*np.pi*rng.uniform(9, 11)*t + rng.uniform(0, 2*np.pi))
        else:
            sig[ch] += rng.uniform(3, 8) * np.sin(2*np.pi*rng.uniform(8, 13)*t + rng.uniform(0, 2*np.pi))

        # Beta (13-30 Hz)
        for _ in range(rng.randint(1, 4)):
            f = rng.uniform(13, 30)
            sig[ch] += rng.uniform(2, 8) * np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))

        # Gamma (30-100 Hz) — only if sample rate supports it
        if nyquist > 60:
            for _ in range(rng.randint(0, 3)):
                f = rng.uniform(30, min(100, nyquist * 0.8))
                sig[ch] += rng.uniform(1, 4) * np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))

        # High-frequency oscillations (80-500 Hz) — research rates only
        if nyquist > 250:
            if rng.random() < 0.3:
                f = rng.uniform(80, min(500, nyquist * 0.8))
                sig[ch] += rng.uniform(0.5, 2) * np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi))

        # Broadband noise (pink spectrum)
        noise = rng.randn(T)
        # Simple pink noise: attenuate high frequencies
        freqs = np.fft.rfftfreq(T, 1.0/sample_rate)
        spectrum = np.fft.rfft(noise)
        spectrum[1:] /= np.sqrt(freqs[1:])  # 1/f
        sig[ch] += np.fft.irfft(spectrum, T).astype(np.float32) * 3

        # Sharp transients (10% chance per channel)
        if rng.random() < 0.1:
            n_spikes = rng.randint(1, 4)
            for _ in range(n_spikes):
                spike_t = rng.randint(T // 10, T - T // 10)
                spike_amp = rng.uniform(30, 100) * rng.choice([-1, 1])
                # Sharp spike: narrow Gaussian (1-5ms width)
                width_samples = int(rng.uniform(0.001, 0.005) * sample_rate)
                width_samples = max(2, width_samples)
                gaussian = np.exp(-0.5 * ((np.arange(T) - spike_t) / width_samples)**2)
                sig[ch] += spike_amp * gaussian.astype(np.float32)

    return sig


# ============================================================
# Fitting & Evaluation
# ============================================================

@dataclass
class L3FitResult:
    recording_idx: int
    sample_rate: int
    l3_samples: int          # T of L3 subband
    target_params: int
    actual_params: int
    hidden_dim: int
    pearson_r: float
    prd_percent: float
    fit_time_s: float
    # Per-band PRD (relative to L3 effective bandwidth)
    prd_delta: float = 0.0   # 0.5-4 Hz
    prd_theta: float = 0.0   # 4-8 Hz
    prd_alpha: float = 0.0   # 8-13 Hz
    prd_beta: float = 0.0    # 13-30 Hz (only if L3 bandwidth allows)
    prd_gamma: float = 0.0   # 30+ Hz (only if L3 bandwidth allows)
    # Compression ratios
    cr_raw_vs_siren_int4: float = 0.0
    cr_l3_vs_siren_int4: float = 0.0


def pearson_r(pred, target):
    p = pred.flatten()
    t = target.flatten()
    pc = p - p.mean()
    tc = t - t.mean()
    num = np.sum(pc * tc)
    denom = np.sqrt(np.sum(pc**2) * np.sum(tc**2))
    return float(num / max(denom, 1e-12))


def prd(pred, target):
    diff = pred - target
    sig_power = np.sum(target**2)
    return float(np.sqrt(np.sum(diff**2) / max(sig_power, 1e-12)) * 100)


def per_band_prd_l3(pred, target, l3_sample_rate):
    """Per-band PRD limited to L3's effective bandwidth."""
    from scipy.signal import butter, sosfiltfilt

    nyq = l3_sample_rate / 2.0
    bands = {}
    for name, (lo, hi) in [('delta', (0.5, 4)), ('theta', (4, 8)),
                            ('alpha', (8, 13)), ('beta', (13, 30)),
                            ('gamma', (30, 100))]:
        if hi >= nyq or lo >= nyq:
            bands[name] = 0.0
            continue
        hi_clamped = min(hi, nyq - 1)
        try:
            sos = butter(4, [lo/nyq, hi_clamped/nyq], btype='bandpass', output='sos')
            p_band = sosfiltfilt(sos, pred.flatten())
            t_band = sosfiltfilt(sos, target.flatten())
            power = np.sum(t_band**2)
            if power < 1e-12:
                bands[name] = 0.0
            else:
                bands[name] = float(np.sqrt(np.sum((p_band - t_band)**2) / power) * 100)
        except Exception:
            bands[name] = 0.0
    return bands


def fit_siren_to_l3(
    l3_signal: np.ndarray,
    target_params: int,
    sample_rate: int,
    n_layers: int = 3,
    omega_0: float = 60.0,
    epochs: int = 1500,
    lr: float = 5e-4,
    device: str = 'cuda',
    recording_idx: int = 0,
) -> L3FitResult:
    """Fit SIREN to L3 subband of one recording.

    l3_signal: [21, T_l3] float32
    """
    C, T = l3_signal.shape
    l3_fs = sample_rate / 8.0  # L3 effective sample rate

    # Normalize to [-1, 1]
    vmin, vmax = l3_signal.min(), l3_signal.max()
    vrange = max(vmax - vmin, 1e-8)
    sig_norm = (l3_signal - vmin) / vrange * 2.0 - 1.0

    # Build SIREN
    model = build_siren_for_budget(target_params, n_channels=C, n_layers=n_layers,
                                   omega_0=omega_0)
    model = model.to(device)
    actual_params = model.param_count()

    # Coordinates
    t_coords = torch.linspace(-1, 1, T, device=device).unsqueeze(1)
    target = torch.from_numpy(sig_norm).float().to(device).T  # [T, C]

    # Optimizer: Adam with warm restarts for better convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epochs//3, T_mult=1, eta_min=lr*0.01)

    t0 = time.perf_counter()
    for epoch in range(epochs):
        pred = model(t_coords)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    fit_time = time.perf_counter() - t0

    # Evaluate
    with torch.no_grad():
        pred_final = model(t_coords).cpu().numpy().T  # [C, T]

    # Denormalize
    pred_denorm = (pred_final + 1.0) / 2.0 * vrange + vmin

    # Metrics
    r = pearson_r(pred_denorm, l3_signal)
    p = prd(pred_denorm, l3_signal)
    band_prd = per_band_prd_l3(pred_denorm, l3_signal, l3_fs)

    # Compression ratios
    raw_bytes = C * (sample_rate * 10) * 2  # raw signal bytes (16-bit)
    l3_bytes = C * T * 2  # L3 bytes (16-bit)
    siren_bytes_int4 = actual_params * 0.5

    return L3FitResult(
        recording_idx=recording_idx,
        sample_rate=sample_rate,
        l3_samples=T,
        target_params=target_params,
        actual_params=actual_params,
        hidden_dim=model.net[0].linear.out_features,
        pearson_r=r,
        prd_percent=p,
        fit_time_s=fit_time,
        prd_delta=band_prd.get('delta', 0.0),
        prd_theta=band_prd.get('theta', 0.0),
        prd_alpha=band_prd.get('alpha', 0.0),
        prd_beta=band_prd.get('beta', 0.0),
        prd_gamma=band_prd.get('gamma', 0.0),
        cr_raw_vs_siren_int4=raw_bytes / max(siren_bytes_int4, 1),
        cr_l3_vs_siren_int4=l3_bytes / max(siren_bytes_int4, 1),
    )


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(
    n_recordings: int = 50,
    sample_rate: int = 1000,
    param_budgets: List[int] = None,
    n_layers: int = 3,
    omega_0: float = 60.0,
    epochs: int = 1500,
    lr: float = 5e-4,
    device: str = None,
    output_path: str = None,
):
    if param_budgets is None:
        param_budgets = [1000, 2500, 5000, 10000, 25000]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    l3_fs = sample_rate / 8.0
    l3_T = int(sample_rate * 10 / 8)  # 10-second window

    print(f"{'=' * 70}")
    print(f"  INR L3 SUBBAND VALIDATION")
    print(f"  SIREN on lifting DWT L3 approximation — smooth base signal")
    print(f"{'=' * 70}")
    print(f"  Sample rate:     {sample_rate} Hz")
    print(f"  Window:          10 seconds = {sample_rate * 10:,} samples")
    print(f"  L3 subband:      [21, {l3_T}] @ {l3_fs:.1f} Hz effective")
    print(f"  L3 bandwidth:    0 - {l3_fs/2:.1f} Hz")
    print(f"  Param budgets:   {param_budgets}")
    print(f"  SIREN layers:    {n_layers}")
    print(f"  Omega_0:         {omega_0}")
    print(f"  Epochs/fit:      {epochs}")
    print(f"  Device:          {device}")
    print(f"{'=' * 70}")

    summary_by_budget = {}

    for budget in param_budgets:
        print(f"\n{'─' * 70}")
        print(f"  Parameter budget: {budget:,}")
        print(f"{'─' * 70}")

        results: List[L3FitResult] = []

        for i in range(n_recordings):
            # Generate EEG at target sample rate
            sig = generate_eeg(sample_rate=sample_rate, duration=10.0,
                               n_channels=21, seed=42 + i)
            # Apply 3-level lifting DWT
            l3, _ = lifting_3level(sig)

            result = fit_siren_to_l3(
                l3, target_params=budget, sample_rate=sample_rate,
                n_layers=n_layers, omega_0=omega_0, epochs=epochs,
                lr=lr, device=device, recording_idx=i,
            )
            results.append(result)

            if (i + 1) % 10 == 0 or i == 0:
                recent = results[-min(10, len(results)):]
                avg_r = np.mean([r.pearson_r for r in recent])
                print(f"    [{i+1:3d}/{n_recordings}]  R={avg_r:.4f}  "
                      f"params={result.actual_params:,}  L3=[21,{result.l3_samples}]")

        # Summary
        rs = [r.pearson_r for r in results]
        summary = {
            'target_params': budget,
            'actual_params': results[0].actual_params,
            'hidden_dim': results[0].hidden_dim,
            'l3_samples': results[0].l3_samples,
            'r_mean': float(np.mean(rs)),
            'r_median': float(np.median(rs)),
            'r_p10': float(np.percentile(rs, 10)),
            'r_p90': float(np.percentile(rs, 90)),
            'prd_mean': float(np.mean([r.prd_percent for r in results])),
            'prd_delta': float(np.mean([r.prd_delta for r in results])),
            'prd_theta': float(np.mean([r.prd_theta for r in results])),
            'prd_alpha': float(np.mean([r.prd_alpha for r in results])),
            'prd_beta': float(np.mean([r.prd_beta for r in results])),
            'prd_gamma': float(np.mean([r.prd_gamma for r in results])),
            'cr_raw_int4': float(np.mean([r.cr_raw_vs_siren_int4 for r in results])),
            'cr_l3_int4': float(np.mean([r.cr_l3_vs_siren_int4 for r in results])),
            'time_mean_s': float(np.mean([r.fit_time_s for r in results])),
        }
        summary_by_budget[budget] = summary

        print(f"\n  BUDGET {budget:,}:")
        print(f"    R:         mean={summary['r_mean']:.4f}  median={summary['r_median']:.4f}")
        print(f"    PRD:       {summary['prd_mean']:.1f}%")
        print(f"    Per-band:  δ={summary['prd_delta']:.1f}%  θ={summary['prd_theta']:.1f}%  "
              f"α={summary['prd_alpha']:.1f}%  β={summary['prd_beta']:.1f}%  "
              f"γ={summary['prd_gamma']:.1f}%")
        print(f"    CR vs raw: {summary['cr_raw_int4']:.0f}:1 (INT4)")
        print(f"    CR vs L3:  {summary['cr_l3_int4']:.0f}:1 (INT4)")
        print(f"    Time:      {summary['time_mean_s']:.2f}s/recording")

    # Verdict
    print(f"\n{'=' * 70}")
    print(f"  VERDICT (L3 subband @ {sample_rate} Hz, L3 bandwidth 0-{l3_fs/2:.0f} Hz)")
    print(f"{'=' * 70}")

    for budget, s in summary_by_budget.items():
        status = "PASS" if s['r_median'] > 0.90 else ("MARGINAL" if s['r_median'] > 0.82 else "FAIL")
        beta_ok = "OK" if s['prd_beta'] < 30 else ("WARN" if s['prd_beta'] < 60 else "FAIL")
        print(f"    {budget:>6,} params:  R={s['r_median']:.4f}  "
              f"β={s['prd_beta']:.0f}%[{beta_ok}]  "
              f"CR={s['cr_raw_int4']:.0f}:1  [{status}]")

    print(f"\n  Acceptance criteria:")
    print(f"    - R_median > 0.90 for L3 subband")
    print(f"    - Per-band PRD < 30% for all bands within L3 bandwidth")
    print(f"    - Beta specifically < 30% (critical for clinical use)")
    print(f"{'=' * 70}")

    # Save
    output = {
        'experiment': 'inr_l3_validation',
        'config': {
            'sample_rate': sample_rate,
            'l3_effective_rate': l3_fs,
            'l3_bandwidth_hz': l3_fs / 2,
            'l3_samples': l3_T,
            'n_recordings': n_recordings,
            'param_budgets': param_budgets,
            'n_layers': n_layers,
            'omega_0': omega_0,
            'epochs': epochs,
        },
        'summary_by_budget': summary_by_budget,
    }
    if output_path is None:
        output_path = str(ROOT / 'outputs' / f'inr_l3_validation_{sample_rate}hz.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="INR L3 Subband Validation")
    parser.add_argument('--sample-rate', type=int, default=1000,
                        help='EEG sample rate in Hz (default: 1000)')
    parser.add_argument('--n-recordings', type=int, default=50,
                        help='Number of recordings (default: 50)')
    parser.add_argument('--budgets', type=str, default='1000,2500,5000,10000,25000',
                        help='Parameter budgets (comma-separated)')
    parser.add_argument('--layers', type=int, default=3,
                        help='SIREN layers (default: 3)')
    parser.add_argument('--omega', type=float, default=60.0,
                        help='SIREN omega_0 (default: 60.0)')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Epochs per fit (default: 1500)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        n_recordings=args.n_recordings,
        sample_rate=args.sample_rate,
        param_budgets=[int(x) for x in args.budgets.split(',')],
        n_layers=args.layers,
        omega_0=args.omega,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
