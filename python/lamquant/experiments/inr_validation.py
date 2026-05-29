#!/usr/bin/env python3
"""
INR Validation Experiment: Can SIREN represent EEG at MCU-feasible parameter counts?

Fits SIREN networks at varying parameter budgets to real EEG recordings and measures
reconstruction quality. This determines whether implicit neural representations are
viable for on-device EEG decompression.

Stop criterion: if 5K params doesn't hit R > 0.82 median, kill the approach.

Usage:
    python ai_models/experiments/inr_validation.py
    python ai_models/experiments/inr_validation.py --n-recordings 200 --epochs 2000
    python ai_models/experiments/inr_validation.py --output results/inr_validation.json
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# SIREN Architecture
# ============================================================

class SirenLayer(nn.Module):
    """Single SIREN layer: Linear + sin(omega * x).

    Initialization follows Sitzmann et al. 2020:
    - First layer: uniform(-1/in, 1/in)
    - Hidden layers: uniform(-sqrt(6/in)/omega, sqrt(6/in)/omega)
    """

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
    """SIREN: Sinusoidal Representation Network for EEG signals.

    Maps continuous coordinates (t, channel) -> signal value.

    Input modes:
      - 'time_only': input is [t] (1D), output is [21] (all channels)
      - 'coord': input is [t, ch_embed] (1+ch_dim), output is [1]

    For EEG compression, 'time_only' is more parameter-efficient:
    one forward pass gives all 21 channels simultaneously.
    """

    def __init__(self, hidden_dim=64, n_layers=2, n_channels=21,
                 omega_0=30.0, mode='time_only'):
        super().__init__()
        self.mode = mode
        self.n_channels = n_channels

        if mode == 'time_only':
            in_dim = 1
            out_dim = n_channels
        else:  # coord mode
            in_dim = 1 + 4  # time + channel Fourier features
            out_dim = 1

        layers = []
        layers.append(SirenLayer(in_dim, hidden_dim, omega=omega_0, is_first=True))
        for _ in range(n_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega=omega_0))
        self.net = nn.Sequential(*layers)

        # Final linear layer (no sin activation)
        self.output = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.output.weight.uniform_(-bound, bound)

    def forward(self, coords):
        """coords: [N, in_dim] -> [N, out_dim]"""
        h = self.net(coords)
        return self.output(h)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def build_siren(target_params, n_channels=21, n_layers=2, omega_0=30.0):
    """Build a SIREN with approximately target_params parameters.

    Solves for hidden_dim given:
      params ≈ (1 * hidden) + (n_layers-1) * hidden^2 + hidden * n_channels + biases
    """
    # Binary search for hidden_dim
    lo, hi = 4, 512
    best_dim = lo
    for _ in range(20):
        mid = (lo + hi) // 2
        model = SIREN(hidden_dim=mid, n_layers=n_layers, n_channels=n_channels,
                      omega_0=omega_0)
        n = model.param_count()
        if n <= target_params:
            best_dim = mid
            lo = mid + 1
        else:
            hi = mid - 1

    model = SIREN(hidden_dim=best_dim, n_layers=n_layers, n_channels=n_channels,
                  omega_0=omega_0)
    return model


# ============================================================
# Signal Loading
# ============================================================

def load_recordings(manifest_path: str, n_recordings: int = 100,
                    seed: int = 42) -> List[np.ndarray]:
    """Load n_recordings from the dataset manifest.

    Returns list of [21, T] int16 arrays (T varies by recording).
    Falls back to synthetic data if manifest not available.
    """
    recordings = []
    rng = np.random.RandomState(seed)

    # Try loading real data from precomputed L3 or raw EDF
    q31_dir = ROOT / 'ai_models' / 'dataset_sim' / 'q31_events'
    if q31_dir.exists():
        npz_files = sorted(q31_dir.glob('*.npz'))
        if len(npz_files) > n_recordings:
            indices = rng.choice(len(npz_files), n_recordings, replace=False)
            npz_files = [npz_files[i] for i in indices]
        for f in npz_files[:n_recordings]:
            try:
                data = np.load(f)
                # Q31 events: 'signal' key, shape [21, 2500]
                if 'signal' in data:
                    sig = data['signal']
                elif 'samples' in data:
                    sig = data['samples']
                else:
                    continue
                if sig.shape[0] == 21 and sig.shape[1] >= 2500:
                    recordings.append(sig[:, :2500].astype(np.float32))
            except Exception:
                continue

    if len(recordings) < 10:
        print(f"[!] Only {len(recordings)} real recordings found. "
              f"Generating synthetic EEG for remaining {n_recordings - len(recordings)}.")
        # Synthetic EEG: sum of band-limited oscillations + noise
        for _ in range(n_recordings - len(recordings)):
            t = np.linspace(0, 10, 2500)
            sig = np.zeros((21, 2500), dtype=np.float32)
            for ch in range(21):
                # Delta (0.5-4 Hz)
                sig[ch] += rng.randn() * 20 * np.sin(2 * np.pi * rng.uniform(0.5, 4) * t + rng.uniform(0, 2*np.pi))
                # Theta (4-8 Hz)
                sig[ch] += rng.randn() * 10 * np.sin(2 * np.pi * rng.uniform(4, 8) * t + rng.uniform(0, 2*np.pi))
                # Alpha (8-13 Hz)
                sig[ch] += rng.randn() * 15 * np.sin(2 * np.pi * rng.uniform(8, 13) * t + rng.uniform(0, 2*np.pi))
                # Beta (13-30 Hz)
                sig[ch] += rng.randn() * 5 * np.sin(2 * np.pi * rng.uniform(13, 30) * t + rng.uniform(0, 2*np.pi))
                # Broadband noise
                sig[ch] += rng.randn(2500) * 3
                # Occasional sharp transient (20% chance per channel)
                if rng.random() < 0.2:
                    spike_t = rng.randint(100, 2400)
                    spike_amp = rng.uniform(30, 80) * rng.choice([-1, 1])
                    spike_width = rng.randint(5, 20)
                    gaussian = np.exp(-0.5 * ((np.arange(2500) - spike_t) / spike_width) ** 2)
                    sig[ch] += spike_amp * gaussian
            recordings.append(sig)

    print(f"[*] Loaded {len(recordings)} recordings "
          f"({'real' if q31_dir.exists() else 'synthetic'})")
    return recordings


# ============================================================
# Fitting & Evaluation
# ============================================================

@dataclass
class FitResult:
    """Result of fitting a SIREN to one recording."""
    recording_idx: int
    target_params: int
    actual_params: int
    hidden_dim: int
    n_layers: int
    pearson_r: float
    prd_percent: float
    mse: float
    fit_time_s: float
    # Per-band PRD
    prd_delta: float = 0.0
    prd_theta: float = 0.0
    prd_alpha: float = 0.0
    prd_beta: float = 0.0
    prd_gamma: float = 0.0


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Global Pearson R across all channels and timepoints."""
    p = pred.flatten()
    t = target.flatten()
    p_centered = p - p.mean()
    t_centered = t - t.mean()
    num = np.sum(p_centered * t_centered)
    denom = np.sqrt(np.sum(p_centered**2) * np.sum(t_centered**2))
    if denom < 1e-12:
        return 0.0
    return float(num / denom)


def prd(pred: np.ndarray, target: np.ndarray) -> float:
    """Percent Root-mean-square Difference."""
    diff = pred - target
    sig_power = np.sum(target**2)
    if sig_power < 1e-12:
        return 100.0
    return float(np.sqrt(np.sum(diff**2) / sig_power) * 100)


def per_band_prd(pred: np.ndarray, target: np.ndarray, fs=250.0) -> Dict[str, float]:
    """Per-band PRD using FFT filtering."""
    from scipy.signal import butter, sosfiltfilt

    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, min(100.0, fs/2 - 1)),
    }
    results = {}
    nyq = fs / 2.0

    for name, (lo, hi) in bands.items():
        if hi >= nyq:
            results[name] = 0.0
            continue
        try:
            sos = butter(4, [lo/nyq, hi/nyq], btype='bandpass', output='sos')
            pred_band = sosfiltfilt(sos, pred.flatten())
            target_band = sosfiltfilt(sos, target.flatten())
            sig_power = np.sum(target_band**2)
            if sig_power < 1e-12:
                results[name] = 0.0
            else:
                results[name] = float(np.sqrt(np.sum((pred_band - target_band)**2) / sig_power) * 100)
        except Exception:
            results[name] = 0.0
    return results


def fit_siren_to_recording(
    signal: np.ndarray,
    target_params: int,
    n_layers: int = 2,
    omega_0: float = 30.0,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: str = 'cuda',
    recording_idx: int = 0,
) -> FitResult:
    """Fit a SIREN to a single EEG recording.

    signal: [21, 2500] float32 array
    target_params: approximate parameter budget
    Returns: FitResult with quality metrics
    """
    C, T = signal.shape
    assert C == 21 and T == 2500

    # Normalize signal to [-1, 1] for SIREN stability
    sig_min = signal.min()
    sig_max = signal.max()
    sig_range = sig_max - sig_min
    if sig_range < 1e-8:
        sig_range = 1.0
    signal_norm = (signal - sig_min) / sig_range * 2.0 - 1.0

    # Build SIREN
    model = build_siren(target_params, n_channels=C, n_layers=n_layers, omega_0=omega_0)
    model = model.to(device)
    actual_params = model.param_count()

    # Coordinates: normalized time [0, 1] -> [-1, 1]
    t_coords = torch.linspace(-1, 1, T, device=device).unsqueeze(1)  # [T, 1]
    target = torch.from_numpy(signal_norm).float().to(device).T  # [T, C]

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*0.01)

    t0 = time.perf_counter()

    for epoch in range(epochs):
        pred = model(t_coords)  # [T, C]
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    fit_time = time.perf_counter() - t0

    # Evaluate
    with torch.no_grad():
        pred_final = model(t_coords).cpu().numpy()  # [T, C]

    # Denormalize
    pred_denorm = (pred_final.T + 1.0) / 2.0 * sig_range + sig_min  # [C, T]
    target_denorm = signal  # [C, T]

    # Metrics
    r = pearson_r(pred_denorm, target_denorm)
    p = prd(pred_denorm, target_denorm)
    mse_val = float(np.mean((pred_denorm - target_denorm)**2))

    # Per-band PRD
    band_prd = per_band_prd(pred_denorm, target_denorm, fs=250.0)

    return FitResult(
        recording_idx=recording_idx,
        target_params=target_params,
        actual_params=actual_params,
        hidden_dim=model.net[0].linear.out_features,
        n_layers=n_layers,
        pearson_r=r,
        prd_percent=p,
        mse=mse_val,
        fit_time_s=fit_time,
        prd_delta=band_prd.get('delta', 0.0),
        prd_theta=band_prd.get('theta', 0.0),
        prd_alpha=band_prd.get('alpha', 0.0),
        prd_beta=band_prd.get('beta', 0.0),
        prd_gamma=band_prd.get('gamma', 0.0),
    )


# ============================================================
# Experiment Runner
# ============================================================

def run_experiment(
    n_recordings: int = 100,
    param_budgets: List[int] = None,
    n_layers: int = 2,
    omega_0: float = 30.0,
    epochs: int = 1000,
    lr: float = 1e-3,
    device: str = None,
    output_path: str = None,
):
    """Run the full INR validation experiment."""
    if param_budgets is None:
        param_budgets = [1000, 2500, 5000, 10000, 25000]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"=" * 70)
    print(f"  INR VALIDATION EXPERIMENT")
    print(f"  SIREN fitness for EEG compression at MCU-feasible parameter counts")
    print(f"=" * 70)
    print(f"  Recordings:   {n_recordings}")
    print(f"  Param budgets: {param_budgets}")
    print(f"  Layers:        {n_layers}")
    print(f"  Omega_0:       {omega_0}")
    print(f"  Epochs/fit:    {epochs}")
    print(f"  Device:        {device}")
    print(f"=" * 70)

    # Load recordings
    recordings = load_recordings(
        str(ROOT / 'ai_models/dataset_sim/manifest_v3.json'),
        n_recordings=n_recordings,
    )

    all_results: List[Dict] = []
    summary_by_budget: Dict[int, Dict] = {}

    for budget in param_budgets:
        print(f"\n{'─' * 70}")
        print(f"  Parameter budget: {budget:,}")
        print(f"{'─' * 70}")

        budget_results: List[FitResult] = []

        for i, sig in enumerate(recordings):
            result = fit_siren_to_recording(
                sig, target_params=budget, n_layers=n_layers,
                omega_0=omega_0, epochs=epochs, lr=lr,
                device=device, recording_idx=i,
            )
            budget_results.append(result)
            all_results.append(asdict(result))

            if (i + 1) % 10 == 0 or i == 0:
                recent = budget_results[-min(10, len(budget_results)):]
                avg_r = np.mean([r.pearson_r for r in recent])
                avg_prd = np.mean([r.prd_percent for r in recent])
                avg_time = np.mean([r.fit_time_s for r in recent])
                print(f"    [{i+1:3d}/{len(recordings)}]  "
                      f"R={avg_r:.4f}  PRD={avg_prd:.1f}%  "
                      f"time={avg_time:.2f}s  params={result.actual_params:,}")

        # Summary stats for this budget
        rs = [r.pearson_r for r in budget_results]
        prds = [r.prd_percent for r in budget_results]
        times = [r.fit_time_s for r in budget_results]

        summary = {
            'target_params': budget,
            'actual_params': budget_results[0].actual_params if budget_results else 0,
            'hidden_dim': budget_results[0].hidden_dim if budget_results else 0,
            'n_recordings': len(budget_results),
            'r_mean': float(np.mean(rs)),
            'r_median': float(np.median(rs)),
            'r_p10': float(np.percentile(rs, 10)),
            'r_p90': float(np.percentile(rs, 90)),
            'prd_mean': float(np.mean(prds)),
            'prd_median': float(np.median(prds)),
            'time_mean_s': float(np.mean(times)),
            'prd_delta_mean': float(np.mean([r.prd_delta for r in budget_results])),
            'prd_theta_mean': float(np.mean([r.prd_theta for r in budget_results])),
            'prd_alpha_mean': float(np.mean([r.prd_alpha for r in budget_results])),
            'prd_beta_mean': float(np.mean([r.prd_beta for r in budget_results])),
            'prd_gamma_mean': float(np.mean([r.prd_gamma for r in budget_results])),
            # Compression ratio at various quantization levels
            'cr_fp16': 21 * 2500 * 2 / (budget_results[0].actual_params * 2) if budget_results else 0,
            'cr_int8': 21 * 2500 * 2 / (budget_results[0].actual_params * 1) if budget_results else 0,
            'cr_int4': 21 * 2500 * 2 / (budget_results[0].actual_params * 0.5) if budget_results else 0,
            'cr_ternary': 21 * 2500 * 2 / (budget_results[0].actual_params * 0.25) if budget_results else 0,
        }
        summary_by_budget[budget] = summary

        print(f"\n  BUDGET {budget:,} SUMMARY:")
        print(f"    Actual params:  {summary['actual_params']:,} (hidden={summary['hidden_dim']})")
        print(f"    Pearson R:      mean={summary['r_mean']:.4f}  median={summary['r_median']:.4f}  "
              f"p10={summary['r_p10']:.4f}  p90={summary['r_p90']:.4f}")
        print(f"    PRD:            mean={summary['prd_mean']:.1f}%  median={summary['prd_median']:.1f}%")
        print(f"    Per-band PRD:   δ={summary['prd_delta_mean']:.1f}%  θ={summary['prd_theta_mean']:.1f}%  "
              f"α={summary['prd_alpha_mean']:.1f}%  β={summary['prd_beta_mean']:.1f}%  γ={summary['prd_gamma_mean']:.1f}%")
        print(f"    Compression:    FP16={summary['cr_fp16']:.0f}:1  INT8={summary['cr_int8']:.0f}:1  "
              f"INT4={summary['cr_int4']:.0f}:1  ternary={summary['cr_ternary']:.0f}:1")
        print(f"    Fit time:       {summary['time_mean_s']:.2f}s/recording")

    # ---- Final verdict ----
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}")

    viable = False
    for budget, s in summary_by_budget.items():
        status = "PASS" if s['r_median'] > 0.82 else "FAIL"
        if budget == 5000 and s['r_median'] > 0.82:
            viable = True
        print(f"    {budget:>6,} params:  R_median={s['r_median']:.4f}  [{status}]"
              f"  CR(INT4)={s['cr_int4']:.0f}:1")

    if viable:
        print(f"\n  *** VIABLE: 5K params achieves R > 0.82 median. ***")
        print(f"  *** Proceed to Phase 2: hypernetwork prototype. ***")
    else:
        print(f"\n  *** NOT VIABLE at 5K params. ***")
        if any(s['r_median'] > 0.82 for s in summary_by_budget.values()):
            viable_at = min(b for b, s in summary_by_budget.items() if s['r_median'] > 0.82)
            print(f"  *** Viable at {viable_at:,} params (CR={summary_by_budget[viable_at]['cr_int4']:.0f}:1 INT4). ***")
            print(f"  *** Consider if that compression ratio justifies the approach. ***")
        else:
            print(f"  *** SIREN does not achieve R > 0.82 at any tested budget. ***")
            print(f"  *** Kill the INR approach for this signal type. ***")
    print(f"{'=' * 70}")

    # Save results
    output = {
        'experiment': 'inr_validation',
        'config': {
            'n_recordings': n_recordings,
            'param_budgets': param_budgets,
            'n_layers': n_layers,
            'omega_0': omega_0,
            'epochs': epochs,
            'lr': lr,
            'device': device,
        },
        'summary_by_budget': summary_by_budget,
        'all_results': all_results,
    }

    if output_path is None:
        output_path = str(ROOT / 'outputs' / 'inr_validation.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    return output


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="INR Validation: SIREN fitness for EEG compression")
    parser.add_argument('--n-recordings', type=int, default=100,
                        help='Number of recordings to test (default: 100)')
    parser.add_argument('--budgets', type=str, default='1000,2500,5000,10000,25000',
                        help='Comma-separated parameter budgets')
    parser.add_argument('--layers', type=int, default=2,
                        help='SIREN hidden layers (default: 2)')
    parser.add_argument('--omega', type=float, default=30.0,
                        help='SIREN omega_0 frequency (default: 30.0)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Optimization epochs per recording (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: cuda if available)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(',')]

    run_experiment(
        n_recordings=args.n_recordings,
        param_budgets=budgets,
        n_layers=args.layers,
        omega_0=args.omega,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
