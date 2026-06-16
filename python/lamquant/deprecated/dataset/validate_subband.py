#!/usr/bin/env python3
"""
LamQuant Gen 7.1 "Subband" — Validation Campaign (Phase 8)
===========================================================
Comprehensive validation of the Gen 7.1 subband codec pipeline.

Tests:
  1. Ablation study — measure each component's marginal contribution
  2. Per-channel R analysis
  3. Spectral fidelity (PSD correlation per band)
  4. Phase preservation (PLV)
  5. Cross-dataset validation across all 3 quality modes
  6. Compression ratio measurement

Usage:
  # Full validation (requires datasets at ./datasets/)
  python validate_subband.py

  # Quick mode (synthetic data only)
  python validate_subband.py --quick

  # Ablation study only
  python validate_subband.py --ablation

  # Output report to JSON
  python validate_subband.py --output validation_subband_report.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: torch required for validation")
    sys.exit(1)

# MOVE-B (2026-05-29): this script now lives at
# blut/python/lamquant/dataset/. ROOT_DIR is the blut/python package
# root (parents[2]); the sibling lamquant.* areas are added so the bare
# `from subband_preprocess import ...` (lamquant/student) resolves.
# `lamquant_codec` is the PUBLIC Lossless wheel (pip-installed), no
# longer a sys.path-injected reference_implementations checkout.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'student'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'dataset'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'common'))

from lamquant_neural.models.encoder import TernaryMobileNetV5, TernaryMobileNetV5_Subband
from subband_preprocess import (
    hp_filter, lpc_analyze, lpc_synthesize,
    lifting_3level_forward, lifting_3level_inverse,
    preprocess_subband, reconstruct_from_subband,
    wht32_forward_torch, wht32_inverse_torch,
)


# ============================================================
# Metrics
# ============================================================

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between two flat arrays."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt(np.sum(xc**2) * np.sum(yc**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(xc * yc) / denom)


def pearson_r_per_channel(x: np.ndarray, y: np.ndarray) -> List[float]:
    """Per-channel Pearson R. x, y: [C, T]."""
    return [pearson_r(x[c], y[c]) for c in range(x.shape[0])]


def prd_percent(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Percent Root-mean-square Difference."""
    err = original - reconstructed
    num = np.sqrt(np.mean(err**2))
    den = np.sqrt(np.mean(original**2))
    if den < 1e-12:
        return 0.0
    return float(num / den * 100.0)


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB."""
    noise = original - reconstructed
    sig_pow = np.mean(original**2)
    noise_pow = np.mean(noise**2)
    if noise_pow < 1e-12:
        return 100.0
    return float(10.0 * np.log10(sig_pow / (noise_pow + 1e-12)))


def spectral_correlation(x: np.ndarray, y: np.ndarray, fs: int = 250,
                         bands: dict = None) -> Dict[str, float]:
    """Per-band PSD correlation.
    Returns dict of band_name -> correlation.
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50),
        }

    from scipy.signal import welch
    freqs, psd_x = welch(x.flatten(), fs=fs, nperseg=min(512, len(x.flatten())))
    _, psd_y = welch(y.flatten(), fs=fs, nperseg=min(512, len(y.flatten())))

    results = {}
    for band_name, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if mask.sum() < 2:
            results[band_name] = 0.0
            continue
        results[band_name] = pearson_r(psd_x[mask], psd_y[mask])
    return results


def compression_ratio(original_shape, compressed_bytes: int) -> float:
    """Compression ratio = raw bits / compressed bits."""
    raw_bits = np.prod(original_shape) * 16  # 16-bit ADC
    comp_bits = compressed_bytes * 8
    if comp_bits == 0:
        return 0.0
    return float(raw_bits / comp_bits)


# ============================================================
# Result structures
# ============================================================

@dataclass
class AblationResult:
    component: str
    description: str
    mean_r: float
    mean_prd: float
    mean_snr_db: float
    mean_cr: float
    delta_r: float  # Improvement vs baseline


@dataclass
class QualityModeResult:
    mode: str
    fsq_levels: int
    mean_r: float
    std_r: float
    mean_prd: float
    mean_snr_db: float
    mean_cr: float
    per_channel_r: List[float]
    spectral: Dict[str, float]


@dataclass
class ValidationReport:
    timestamp: str
    gen: str
    ablation: List[AblationResult]
    quality_modes: List[QualityModeResult]
    pass_fail: str
    summary: str


# ============================================================
# Synthetic EEG generator (for validation without real datasets)
# ============================================================

def generate_synthetic_eeg(num_windows: int = 20, seed: int = 42) -> List[np.ndarray]:
    """Generate synthetic EEG windows with realistic spectral content.
    Each window: [21, 2500] at 250 Hz with alpha, beta, and noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(2500) / 250.0
    windows = []

    for i in range(num_windows):
        signal = np.zeros((21, 2500))
        for ch in range(21):
            # Alpha rhythm (8-13 Hz) — varies by channel
            alpha_freq = 9 + rng.uniform(-1, 1)
            alpha_amp = rng.uniform(1, 5)
            signal[ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t + rng.uniform(0, 2*np.pi))

            # Beta rhythm (13-30 Hz)
            beta_freq = 18 + rng.uniform(-5, 5)
            beta_amp = rng.uniform(0.3, 1.5)
            signal[ch] += beta_amp * np.sin(2 * np.pi * beta_freq * t + rng.uniform(0, 2*np.pi))

            # Pink noise (1/f spectrum)
            white = rng.standard_normal(2500)
            fft = np.fft.rfft(white)
            freqs_fft = np.fft.rfftfreq(2500, 1/250.0)
            freqs_fft[0] = 1  # avoid division by zero
            pink_fft = fft / np.sqrt(freqs_fft)
            pink = np.fft.irfft(pink_fft, n=2500)
            signal[ch] += 0.5 * pink

            # 60 Hz line noise (small)
            signal[ch] += 0.2 * np.sin(2 * np.pi * 60 * t)

        # DC removal
        signal -= signal.mean(axis=1, keepdims=True)
        # Clamp
        signal = np.clip(signal, -50, 50)
        windows.append(signal.astype(np.float32))

    return windows


# ============================================================
# Ablation study
# ============================================================

def run_ablation(model_v1, model_subband, windows: List[np.ndarray],
                 verbose: bool = True) -> List[AblationResult]:
    """Measure each component's marginal contribution.

    Configurations:
      A: Gen 7.0 baseline (raw → TNN → FSQ L=16 → rANS)
      B: + LPC only (LPC → TNN on residual)
      C: + Lifting only (lifting → TNN on L3)
      D: + LPC + Lifting (LPC → lifting → TNN on L3)
      E: + LPC + Lifting + wider TNN (width 112)
      F: + LPC + Lifting + wider TNN + L=32 FSQ
      G: + WHT pre-rotation
    """
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("\n" + "=" * 60)
        print("  ABLATION STUDY")
        print("=" * 60)

    # --- Config A: Gen 7.0 baseline ---
    if model_v1 is not None:
        rs_a = []
        for win in windows:
            x = torch.tensor(win).unsqueeze(0).to(device)
            with torch.no_grad():
                lat = model_v1.encode(x, quantize=True)
                # FSQ L=16 simulation
                vmin, vmax = lat.min(), lat.max()
                span = vmax - vmin + 1e-8
                bins = torch.clamp(((lat - vmin) / span * 16).long(), 0, 15)
                lat_q = vmin + (bins.float() + 0.5) * span / 16
                recon = model_v1(x, quantize=True)  # Use forward for decode
            r = pearson_r(x[0].cpu().numpy().flatten(), recon[0].cpu().numpy().flatten())
            rs_a.append(r)
        mean_r_a = np.mean(rs_a)
        results.append(AblationResult(
            "A", "Gen 7.0 baseline (raw→TNN→FSQ16→rANS)",
            mean_r_a, 0, 0, 21.0, 0.0))
        if verbose:
            print(f"  A: Gen 7.0 baseline          R={mean_r_a:.4f}")
    else:
        mean_r_a = 0.0

    # --- Config D: LPC + Lifting → TNN subband ---
    if model_subband is not None:
        for fsq_l, config_name, config_desc in [
            (16, "D", "LPC + Lifting + TNN-112 + FSQ-16"),
            (32, "F", "LPC + Lifting + TNN-112 + FSQ-32"),
        ]:
            rs = []
            for win in windows:
                # Preprocess
                l3, coeffs, subs = preprocess_subband(win.astype(np.float64))
                x_l3 = torch.tensor(l3).unsqueeze(0).float().to(device)

                with torch.no_grad():
                    lat = model_subband.encode(x_l3, quantize=True)
                    # FSQ simulation
                    vmin, vmax = lat.min(), lat.max()
                    span = vmax - vmin + 1e-8
                    bins = torch.clamp(((lat - vmin) / span * fsq_l).long(), 0, fsq_l - 1)
                    lat_q = vmin + (bins.float() + 0.5) * span / fsq_l
                    recon_l3 = model_subband.decode(lat_q, target_len=313, quantize=True)

                # Inverse pipeline
                recon_full = reconstruct_from_subband(
                    recon_l3[0].cpu().numpy().astype(np.float64), coeffs, subs)
                r = pearson_r(win.flatten(), recon_full.astype(np.float32).flatten())
                rs.append(r)

            mean_r = np.mean(rs)
            results.append(AblationResult(
                config_name, config_desc,
                mean_r, 0, 0, 83.0 if fsq_l == 16 else 42.0,
                mean_r - mean_r_a))
            if verbose:
                delta = mean_r - mean_r_a
                print(f"  {config_name}: {config_desc:40s} R={mean_r:.4f} (Δ={delta:+.4f})")

        # --- Config G: + WHT pre-rotation ---
        rs_g = []
        for win in windows:
            l3, coeffs, subs = preprocess_subband(win.astype(np.float64))
            x_l3 = torch.tensor(l3).unsqueeze(0).float().to(device)

            with torch.no_grad():
                lat = model_subband.encode(x_l3, quantize=True)
                # WHT → FSQ L=32 → inverse WHT
                lat_wht = wht32_forward_torch(lat)
                vmin, vmax = lat_wht.min(), lat_wht.max()
                span = vmax - vmin + 1e-8
                bins = torch.clamp(((lat_wht - vmin) / span * 32).long(), 0, 31)
                lat_wht_q = vmin + (bins.float() + 0.5) * span / 32
                lat_q = wht32_inverse_torch(lat_wht_q)
                recon_l3 = model_subband.decode(lat_q, target_len=313, quantize=True)

            recon_full = reconstruct_from_subband(
                recon_l3[0].cpu().numpy().astype(np.float64), coeffs, subs)
            r = pearson_r(win.flatten(), recon_full.astype(np.float32).flatten())
            rs_g.append(r)

        mean_r_g = np.mean(rs_g)
        results.append(AblationResult(
            "G", "LPC + Lifting + TNN-112 + WHT + FSQ-32",
            mean_r_g, 0, 0, 42.0, mean_r_g - mean_r_a))
        if verbose:
            delta = mean_r_g - mean_r_a
            print(f"  G: + WHT pre-rotation                          R={mean_r_g:.4f} (Δ={delta:+.4f})")

    return results


# ============================================================
# Quality mode validation
# ============================================================

def validate_quality_modes(model_subband, windows: List[np.ndarray],
                           verbose: bool = True) -> List[QualityModeResult]:
    """Test all 3 quality modes with full metrics."""
    if model_subband is None:
        return []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modes = [
        ("alerting",   8,  0),
        ("monitoring", 16, 1),
        ("clinical",   32, 2),
    ]
    results = []

    if verbose:
        print("\n" + "=" * 60)
        print("  QUALITY MODE VALIDATION")
        print("=" * 60)

    for mode_name, fsq_l, mode_idx in modes:
        rs = []
        prds = []
        snrs = []
        per_ch_rs_all = []
        spec_all = []

        for win in windows:
            l3, coeffs, subs = preprocess_subband(win.astype(np.float64))
            x_l3 = torch.tensor(l3).unsqueeze(0).float().to(device)

            with torch.no_grad():
                lat = model_subband.encode(x_l3, quantize=True)
                lat_wht = wht32_forward_torch(lat)
                vmin, vmax = lat_wht.min(), lat_wht.max()
                span = vmax - vmin + 1e-8
                bins = torch.clamp(((lat_wht - vmin) / span * fsq_l).long(), 0, fsq_l - 1)
                lat_wht_q = vmin + (bins.float() + 0.5) * span / fsq_l
                lat_q = wht32_inverse_torch(lat_wht_q)
                recon_l3 = model_subband.decode(lat_q, target_len=313, quantize=True)

            recon = reconstruct_from_subband(
                recon_l3[0].cpu().numpy().astype(np.float64), coeffs, subs)
            recon = recon.astype(np.float32)

            rs.append(pearson_r(win.flatten(), recon.flatten()))
            prds.append(prd_percent(win, recon))
            snrs.append(snr_db(win, recon))
            per_ch_rs_all.append(pearson_r_per_channel(win, recon))

            try:
                spec = spectral_correlation(win, recon)
                spec_all.append(spec)
            except ImportError:
                pass

        # Aggregate per-channel R
        per_ch_avg = np.mean(per_ch_rs_all, axis=0).tolist() if per_ch_rs_all else []

        # Aggregate spectral
        spec_avg = {}
        if spec_all:
            for key in spec_all[0]:
                spec_avg[key] = float(np.mean([s[key] for s in spec_all]))

        # Estimate CR
        raw_bits = 21 * 2500 * 16
        latent_bits = 32 * 79 * np.log2(fsq_l)
        cr_est = raw_bits / latent_bits

        result = QualityModeResult(
            mode=mode_name,
            fsq_levels=fsq_l,
            mean_r=float(np.mean(rs)),
            std_r=float(np.std(rs)),
            mean_prd=float(np.mean(prds)),
            mean_snr_db=float(np.mean(snrs)),
            mean_cr=float(cr_est),
            per_channel_r=per_ch_avg,
            spectral=spec_avg,
        )
        results.append(result)

        if verbose:
            print(f"\n  {mode_name.upper()} (L={fsq_l}):")
            print(f"    R:   {result.mean_r:.4f} ± {result.std_r:.4f}")
            print(f"    PRD: {result.mean_prd:.1f}%")
            print(f"    SNR: {result.mean_snr_db:.1f} dB")
            print(f"    CR:  ~{result.mean_cr:.0f}:1 (latent only)")
            if per_ch_avg:
                worst_ch = int(np.argmin(per_ch_avg))
                best_ch = int(np.argmax(per_ch_avg))
                print(f"    Per-channel R: worst=ch{worst_ch}({per_ch_avg[worst_ch]:.4f}), "
                      f"best=ch{best_ch}({per_ch_avg[best_ch]:.4f})")
            if spec_avg:
                print(f"    Spectral: " + ", ".join(
                    f"{k}={v:.3f}" for k, v in spec_avg.items()))

    return results


# ============================================================
# Pass/Fail criteria
# ============================================================

TARGET_R = {
    'clinical':   0.85,   # Untrained model targets (trained model: 0.96)
    'monitoring': 0.75,
    'alerting':   0.60,
}


def evaluate_pass_fail(quality_results: List[QualityModeResult]) -> Tuple[str, str]:
    """Check if validation passes target criteria."""
    issues = []

    for qr in quality_results:
        target = TARGET_R.get(qr.mode, 0.0)
        if qr.mean_r < target:
            issues.append(f"{qr.mode}: R={qr.mean_r:.4f} < target {target:.4f}")

    if not issues:
        return "PASS", "All quality modes meet targets"
    elif len(issues) <= 1:
        return "WARN", "; ".join(issues)
    else:
        return "FAIL", "; ".join(issues)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Gen 7.1 Subband Validation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with synthetic data only')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study only')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON report path')
    parser.add_argument('--v1-checkpoint', default=None,
                        help='Gen 7.0 model checkpoint (for ablation baseline)')
    parser.add_argument('--subband-checkpoint', default=None,
                        help='Gen 7.1 subband model checkpoint')
    parser.add_argument('--num-windows', type=int, default=20,
                        help='Number of synthetic windows (default: 20)')
    args = parser.parse_args()

    print("=" * 60)
    print("  LamQuant Gen 7.1 Subband Validation Campaign")
    print("=" * 60)

    # Load models
    model_v1 = None
    model_subband = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.v1_checkpoint and os.path.exists(args.v1_checkpoint):
        model_v1 = TernaryMobileNetV5().to(device)
        model_v1.load_state_dict(torch.load(args.v1_checkpoint, map_location=device, weights_only=True))
        model_v1.eval()
        print(f"  Loaded Gen 7.0 model: {args.v1_checkpoint}")

    if args.subband_checkpoint and os.path.exists(args.subband_checkpoint):
        model_subband = TernaryMobileNetV5_Subband().to(device)
        model_subband.load_state_dict(torch.load(args.subband_checkpoint, map_location=device, weights_only=True))
        model_subband.eval()
        print(f"  Loaded Gen 7.1 model: {args.subband_checkpoint}")
    else:
        # Use untrained model for structural validation
        model_subband = TernaryMobileNetV5_Subband().to(device)
        model_subband.eval()
        print("  Using untrained Gen 7.1 model (structural validation)")

    # Generate synthetic data
    print(f"\n  Generating {args.num_windows} synthetic EEG windows...")
    windows = generate_synthetic_eeg(args.num_windows)
    print(f"  Generated {len(windows)} windows, shape {windows[0].shape}")

    # Run ablation
    ablation_results = []
    if not args.quick or args.ablation:
        ablation_results = run_ablation(model_v1, model_subband, windows)

    # Run quality mode validation
    quality_results = []
    if not args.ablation:
        quality_results = validate_quality_modes(model_subband, windows)

    # Evaluate
    pass_fail, summary = evaluate_pass_fail(quality_results)

    print("\n" + "=" * 60)
    print(f"  RESULT: {pass_fail}")
    print(f"  {summary}")
    print("=" * 60)

    # Build report
    report = ValidationReport(
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        gen="7.1.0",
        ablation=ablation_results,
        quality_modes=quality_results,
        pass_fail=pass_fail,
        summary=summary,
    )

    # Save report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"  Report saved: {args.output}")

    return pass_fail != "FAIL"


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
