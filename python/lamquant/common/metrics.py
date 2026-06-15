"""ai_models/metrics.py — quality metrics for the LamQuant training pipeline.

Single source of truth for the two co-equal primary metrics:

    R   — Pearson correlation. Measures shape preservation.
    PRD — Percentage Root-mean-square Difference. Measures magnitude
          preservation: PRD = sqrt(sum((x - x_hat)**2) / sum(x**2)) * 100.

R and PRD answer different questions. A reconstruction can have R = 0.95
(shape excellent) and PRD = 25 % (signal scaled wrong) — clinicians
would see the right patterns at the wrong amplitudes, which fails
LQS-Clinical compliance. Both metrics gate ship/no-ship.

The functions split by domain:

    prd_torch(x, x_hat)         — differentiable, used in training loss
    prd_numpy(x, x_hat)         — pure numpy, used at validation/CSV time
    per_band_prd(x, x_hat, fs)  — bandpass + PRD per EEG frequency band
    lqs_compliance(R, PRD, ...) — return highest LQS level passed + violations

The per-band PRD only runs at validation: bandpass filtering is
~5 ms/window/band — fine every 10 epochs, too expensive every batch.
For batch-level PRD use the global `prd_torch`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Bands — match lamquant_codec.lqs.LQS_LEVELS["C"].band_fidelity ranges
# ============================================================

EEG_BANDS: Dict[str, Tuple[float, float]] = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 50.0),
}


# ============================================================
# PRD — co-equal with R as primary metric
# ============================================================

def prd_numpy(original: np.ndarray, reconstructed: np.ndarray,
              eps: float = 1e-12) -> float:
    """Percentage Root-mean-square Difference. Lower is better.

    PRD = 100 * sqrt(sum((x - x_hat)**2) / sum(x**2))

    Returns a single scalar over the entire array. eps prevents
    divide-by-zero on all-zero signals (returns 0 instead of NaN).
    """
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    noise = original - reconstructed
    num = float(np.sum(noise ** 2))
    den = float(np.sum(original ** 2))
    if den < eps:
        return 0.0 if num < eps else 100.0
    return 100.0 * (num / den) ** 0.5


def pearson_r_torch(pred, target, eps: float = 1e-8):
    """Differentiable batch Pearson R. Returns a torch scalar tensor.

    Both pred and target should be [B, ...] tensors; each sample is
    flattened to 1-D for the correlation. Returns mean R across batch.

    Use as a training loss: `l_r = 1.0 - pearson_r_torch(recon, target)`

    BUG FIX (2026-04-17): the prior code used `pearson_r_batch` which
    calls `.item()` and returns a Python float — zero gradient. This
    function stays on the computation graph.
    """
    import torch
    p = pred.flatten(1)         # [B, D]
    t = target.flatten(1)       # [B, D]
    pc = p - p.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    num = (pc * tc).sum(dim=-1)                              # [B]
    den = torch.sqrt((pc ** 2).sum(dim=-1)) * torch.sqrt((tc ** 2).sum(dim=-1)) + eps
    r = num / den
    return r.mean()  # scalar tensor, differentiable


def pearson_r_batch(pred, target, eps: float = 1e-8) -> float:
    """Batch Pearson R as a plain Python float — for monitoring/logging.

    The non-differentiable companion of [`pearson_r_torch`]: identical math,
    `.item()`-ed. Use this for metric reporting; use `pearson_r_torch` (or
    `1.0 - pearson_r_torch`) anywhere that needs a gradient — calling THIS in
    a loss path silently zeroes the gradient (the 2026-04-17 bug). Single
    source of truth: delegates to `pearson_r_torch` so there is exactly one
    correlation implementation.
    """
    return float(pearson_r_torch(pred, target, eps=eps))


def prd_torch(original, reconstructed, eps: float = 1e-12,
              max_prd: float = 200.0):
    """Differentiable PRD. Returns a torch scalar tensor.

    Same formula as prd_numpy but stays on GPU and supports backprop.
    Use as a training loss: `l_prd = prd_torch(target, recon) / 100`
    so the loss is in the same scale as the other terms.

    Clamped at max_prd to prevent gradient explosion on all-zero or
    near-zero original signals (flat-line EEG segments).
    """
    import torch
    noise = original - reconstructed
    num = torch.sum(noise ** 2)
    den = torch.sum(original ** 2) + eps
    prd = 100.0 * torch.sqrt(num / den)
    return torch.clamp(prd, max=max_prd)


def masked_pearson_r_torch(pred, target, ch_mask=None, eps: float = 1e-8):
    """Channel-masked differentiable Pearson R (scalar tensor).

    Identical to `pearson_r_torch(pred, target)` when ch_mask is None — same
    flatten-and-pool-per-sample semantics — so existing call sites are a no-op.

    With ch_mask [B, N] (True = real channel), padded channels are excluded
    from BOTH the per-sample mean and the correlation sums. This is the fix for
    the variable-N validation bug: zero-padded channels otherwise pollute R
    (a flat-zero channel correlates as NaN/0 against any target).

    pred/target: [B, N, T]. ch_mask: [B, N] bool/float or None.
    """
    import torch
    p = pred.flatten(1)         # [B, N*T]
    t = target.flatten(1)
    if ch_mask is None:
        pc = p - p.mean(dim=-1, keepdim=True)
        tc = t - t.mean(dim=-1, keepdim=True)
    else:
        T = pred.shape[-1]
        m = ch_mask.to(p.dtype).unsqueeze(-1).expand(-1, -1, T).flatten(1)  # [B,N*T]
        cnt = m.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pm = (p * m).sum(dim=-1, keepdim=True) / cnt
        tm = (t * m).sum(dim=-1, keepdim=True) / cnt
        pc = (p - pm) * m          # padded elements → exactly 0
        tc = (t - tm) * m
    num = (pc * tc).sum(dim=-1)
    den = torch.sqrt((pc ** 2).sum(dim=-1)) * torch.sqrt((tc ** 2).sum(dim=-1)) + eps
    return (num / den).mean()


def masked_pearson_r_batch(pred, target, ch_mask=None) -> float:
    """Float (non-grad) channel-masked Pearson R for monitoring.

    `.item()` of `masked_pearson_r_torch`; matches `pearson_r_batch`
    (training_utils, the validate path) bit-for-bit when ch_mask is None.
    """
    return float(masked_pearson_r_torch(pred, target, ch_mask=ch_mask).item())


def masked_prd_torch(original, reconstructed, ch_mask=None,
                     eps: float = 1e-12, max_prd: float = 200.0):
    """Channel-masked differentiable PRD (scalar tensor).

    Identical to `prd_torch(original, reconstructed)` when ch_mask is None.
    With ch_mask [B, N], padded channels are excluded from the noise/signal
    energy sums (m ∈ {0,1} so m² == m).
    """
    import torch
    noise = original - reconstructed
    if ch_mask is None:
        num = torch.sum(noise ** 2)
        den = torch.sum(original ** 2) + eps
    else:
        m = ch_mask.to(original.dtype).unsqueeze(-1)        # [B,N,1] broadcast over T
        num = torch.sum((noise ** 2) * m)
        den = torch.sum((original ** 2) * m) + eps
    prd = 100.0 * torch.sqrt(num / den)
    return torch.clamp(prd, max=max_prd)


def pearson_r_numpy(original: np.ndarray, reconstructed: np.ndarray,
                     eps: float = 1e-12) -> float:
    """Pearson R over the whole array, single scalar. For diagnostic use."""
    a = np.asarray(original, dtype=np.float64).ravel()
    b = np.asarray(reconstructed, dtype=np.float64).ravel()
    am = a - a.mean()
    bm = b - b.mean()
    den = float(np.sqrt(np.sum(am ** 2) * np.sum(bm ** 2)))
    if den < eps:
        return 0.0
    return float(np.sum(am * bm) / den)


# ============================================================
# Per-band PRD — bandpass + PRD per EEG band
# ============================================================

def per_band_prd(original: np.ndarray, reconstructed: np.ndarray,
                 fs: float = 250.0, order: int = 4) -> Dict[str, float]:
    """Compute PRD per EEG frequency band. Returns {band_name: prd_percent}.

    Inputs are arrays of shape [..., T]. Bandpass is sosfiltfilt (zero-phase,
    no temporal shift). Order 4 → 8th-order zero-phase response, sharp
    enough to isolate adjacent bands without ringing.

    The Nyquist guard: bands above fs/2 are clamped. At fs=250 Hz,
    Nyquist=125 Hz, so all five canonical bands fit comfortably.
    """
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2.0
    out: Dict[str, float] = {}
    for name, (lo, hi) in EEG_BANDS.items():
        hi = min(hi, nyq * 0.99)
        if lo >= hi:
            out[name] = 0.0
            continue
        sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
        try:
            orig_b = sosfiltfilt(sos, original, axis=-1)
            recon_b = sosfiltfilt(sos, reconstructed, axis=-1)
            out[name] = prd_numpy(orig_b, recon_b)
        except ValueError:  # signal too short for filter padlen — skip
            out[name] = 0.0
    return out


def per_band_r(original: np.ndarray, reconstructed: np.ndarray,
                fs: float = 250.0, order: int = 4) -> Dict[str, float]:
    """Pearson R per EEG band. Companion to per_band_prd."""
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2.0
    out: Dict[str, float] = {}
    for name, (lo, hi) in EEG_BANDS.items():
        hi = min(hi, nyq * 0.99)
        if lo >= hi:
            out[name] = 0.0
            continue
        sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
        try:
            orig_b = sosfiltfilt(sos, original, axis=-1)
            recon_b = sosfiltfilt(sos, reconstructed, axis=-1)
            out[name] = pearson_r_numpy(orig_b, recon_b)
        except ValueError:
            out[name] = 0.0
    return out


# ============================================================
# LQS compliance — the ship/no-ship gate
# ============================================================
#
# An LQS level passes iff every threshold is met:
#   global R   ≥ level.min_r
#   global PRD ≤ level.max_prd
#   per-band:  R   ≥ band.min_r
#              PRD ≤ band.max_prd
#
# Returns the highest level passed (strictest first: C → M → A).
# An empty string means even LQS-A failed — the model is below the
# alerting floor and isn't fit for any deployment tier.

_LEVELS_BY_STRICTNESS: List[str] = ['C', 'M', 'A']


def _check_level(level_obj, val_r: float, val_prd: float,
                 per_band_r_dict: Dict[str, float],
                 per_band_prd_dict: Dict[str, float]) -> List[str]:
    """Return a list of violation strings for one LQS level. Empty = passed."""
    violations: List[str] = []

    if val_r < level_obj.min_r:
        violations.append(
            f'global R {val_r:.4f} < {level_obj.min_r:.4f}')
    if val_prd > level_obj.max_prd:
        violations.append(
            f'global PRD {val_prd:.2f}% > {level_obj.max_prd:.2f}%')

    for band_name, req in level_obj.band_fidelity.items():
        bp = per_band_prd_dict.get(band_name)
        br = per_band_r_dict.get(band_name)
        if bp is not None and bp > req.max_prd:
            violations.append(
                f'{band_name} PRD {bp:.2f}% > {req.max_prd:.2f}%')
        if br is not None and br < req.min_r:
            violations.append(
                f'{band_name} R {br:.4f} < {req.min_r:.4f}')

    return violations


def lqs_compliance(val_r: float, val_prd: float,
                   per_band_prd_dict: Optional[Dict[str, float]] = None,
                   per_band_r_dict: Optional[Dict[str, float]] = None,
                   ) -> Tuple[str, List[str]]:
    """Determine the highest LQS level the metrics pass.

    Returns (level, violations). `level` is one of:
        'C' — Clinical (highest neural quality tier; lossless 'L' is
              reserved for the bit-exact lossless pipeline)
        'M' — Monitoring (default ambulatory tier)
        'A' — Alerting (lowest acceptable tier)
        ''  — below LQS-A; the model is not deployable

    `violations` are the constraints that BLOCKED reaching the next
    higher level. So if the function returns ('M', [...]), the
    violations list explains why LQS-C failed — that's the to-do list
    for hitting clinical grade.

    Per-band dicts default to {} if not provided. Without them only
    the global R / PRD are checked, which is enough to identify the
    coarse tier but won't catch a band-localised regression.
    """
    # Robust to call sites where the repo root isn't on sys.path
    # (training scripts insert ai_models/ but not the parent — the
    # lamquant_codec package lives one level up).
    try:
        from lamquant_codec.lqs import LQS_LEVELS
    except ImportError:
        import os, sys as _sys
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from lamquant_codec.lqs import LQS_LEVELS

    pb_prd = per_band_prd_dict or {}
    pb_r = per_band_r_dict or {}

    passed_level = ''
    blocking_violations: List[str] = []

    # Walk from strictest to loosest. The first level that PASSES is the
    # one we report; the violations from the FAILED level above it are
    # the to-do list.
    for L in _LEVELS_BY_STRICTNESS:
        level_obj = LQS_LEVELS[L]
        violations = _check_level(level_obj, val_r, val_prd, pb_r, pb_prd)
        if not violations:
            passed_level = L
            break
        # Remember the violations that blocked the strictest level we tried.
        # When we eventually pass M (say), the blocking_violations are the
        # ones that prevented C — exactly the to-do list the user wants.
        if not passed_level:
            blocking_violations = violations

    return passed_level, blocking_violations


def lqs_pretty(val_r: float, val_prd: float,
                per_band_prd_dict: Dict[str, float]) -> str:
    """Pretty-print the per-band PRD inline: 'δ 3.7%  θ 5.2%  α 5.6%  β 12.9%  γ 25.8%'"""
    glyph = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α',
             'beta': 'β', 'gamma': 'γ'}
    parts = []
    for band in ('delta', 'theta', 'alpha', 'beta', 'gamma'):
        v = per_band_prd_dict.get(band, 0.0)
        parts.append(f'{glyph[band]} {v:>4.1f}%')
    return '  '.join(parts)


# ============================================================
# Asymmetric / clinically weighted loss (training-side only)
# ============================================================
#
# Flat MSE treats every sample equally. A 70-µV epileptiform spike has
# 100× the clinical value of an equally-magnitude background fluctuation,
# but L2 doesn't know that. The asymmetric loss penalises errors in
# proportion to a "clinical importance" weight derived from the original
# signal envelope — peaks get up-weighted, flat background down-weighted.
#
# Two variants:
#
#   asymmetric_eeg_loss(orig, recon, ...)
#     Naive envelope weighting (the user's original proposal). Clamped
#     [floor, ceil] so a single 500-µV EMG burst doesn't completely
#     dominate the gradient. Confound: the envelope can't tell spikes
#     from EMG; both produce high amplitude.
#
#   band_aware_asymmetric_loss(orig, recon, ...)
#     Per-band envelope weighting, with positive bias on δ/θ/α (clinical
#     content lives there) and NEGATIVE bias on β/γ (high amplitude in
#     these bands is mostly EMG/artifact). Costs ~5 FFTs per batch but
#     addresses the EMG confound directly.

def _hilbert_envelope_torch(x: 'torch.Tensor', eps: float = 1e-8) -> 'torch.Tensor':
    """FFT-based analytic-signal magnitude. Differentiable.

    Input  : [..., T] real
    Output : [..., T] non-negative real (the analytic-signal magnitude)
    """
    import torch
    n = x.shape[-1]
    X = torch.fft.fft(x, dim=-1)
    h = torch.zeros(n, device=x.device, dtype=X.dtype)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = torch.fft.ifft(X * h, dim=-1)
    return analytic.abs() + eps


def asymmetric_eeg_loss(original, reconstructed,
                         floor: float = 0.5, ceiling: float = 3.0,
                         eps: float = 1e-8):
    """Envelope-weighted MSE. Penalises missing peaks more than reproducing noise.

    Per-sample weight = clamp(|hilbert(original)| / mean(...), [floor, ceiling]).
    Returns a torch scalar (mean weighted squared error).

    The user's original proposal — implemented as-is for the A/B test.
    Known issue: envelope conflates spikes (clinical) and EMG bursts
    (artifact); see `band_aware_asymmetric_loss` for the band-discriminating
    variant.
    """
    import torch
    env = _hilbert_envelope_torch(original.float(), eps=eps)
    # Per-channel mean-normalise so 21 channels with different baseline
    # amplitudes don't get accidental cross-channel reweighting.
    mean = env.mean(dim=-1, keepdim=True).clamp(min=eps)
    weight = (env / mean).clamp(min=floor, max=ceiling)
    err = (original - reconstructed) ** 2
    return (err * weight).mean()


# Per-band asymmetric weights for the band-aware variant. Positive →
# up-weight high-amplitude regions in this band (clinical content).
# Negative → down-weight high-amplitude (treat as noise/artifact).
_BAND_ASYM_BIAS = {
    'delta': 3.0,    # high amp = K-complex / slow wave / encephalopathy
    'theta': 2.5,    # high amp = drowsiness / temporal pathology
    'alpha': 2.5,    # high amp = posterior dominant rhythm
    'beta': -0.5,    # high amp likely EMG (frontalis, temporalis)
    'gamma': -0.8,   # high amp almost certainly EMG / powerline
}


def band_aware_asymmetric_loss(original, reconstructed,
                                fs: float = 250.0,
                                base: float = 1.0,
                                eps: float = 1e-8):
    """Per-band envelope weighting that addresses the EMG/spike confound.

    For each EEG band: bandpass the original, compute envelope, multiply
    by the band's polarity (+ for clinical bands, − for artifact-prone
    bands), and accumulate into a per-sample weight tensor. The result
    up-weights high-amplitude *clinical* content while down-weighting
    high-amplitude *artifact* content.

    Costs ~5 forward FFTs + 5 envelope calculations per batch, but each
    is over a fullband window so the absolute time is small (~1-2 ms on
    a 4090 at batch=16).

    Returns a torch scalar (mean weighted squared error).
    """
    import torch
    o = original.float()
    n = o.shape[-1]
    # Build a frequency mask once per call (cached on device for the
    # current shape — autograd sees this as a constant).
    freqs = torch.fft.fftfreq(n, d=1.0 / fs, device=o.device).abs()

    weight = torch.full_like(o, fill_value=base)
    for band_name, (lo, hi) in EEG_BANDS.items():
        bias = _BAND_ASYM_BIAS.get(band_name, 0.0)
        if bias == 0.0 or hi >= fs / 2:
            continue
        mask = ((freqs >= lo) & (freqs < min(hi, fs / 2))).to(o.dtype)
        # Bandpass via FFT mask; envelope from analytic signal.
        # We reuse _hilbert_envelope_torch on the band-passed signal.
        X = torch.fft.fft(o, dim=-1) * mask
        band_signal = torch.fft.ifft(X, dim=-1).real
        env = _hilbert_envelope_torch(band_signal, eps=eps)
        env_norm = env / env.mean(dim=-1, keepdim=True).clamp(min=eps)
        # Linear blend: weight += bias * (env_norm - 1)  →  bands at
        # baseline amplitude leave weight unchanged; high-envelope
        # samples shift weight by `bias` per unit of normalised envelope.
        weight = weight + bias * (env_norm - 1.0)
    # Clamp the final weight so no single sample blows up the gradient.
    weight = weight.clamp(min=0.1, max=5.0)

    err = (o - reconstructed.float()) ** 2
    return (err * weight).mean()


def per_band_relative_loss(reconstructed, original, fs: float = 250.0,
                           eps: float = 1e-6):
    """Per-band RELATIVE reconstruction loss — the allocation fix.

    Splits recon + target into the canonical EEG bands (differentiable FFT
    mask), computes the relative L2 error ``||recon_b − orig_b|| / ||orig_b||``
    per band, and averages over bands with **EQUAL weight**. Because each
    band's error is normalised by THAT band's own energy, the low-amplitude
    high-frequency bands (beta / gamma — the >15 Hz detail the encoder is
    otherwise blind to) drive as much gradient as the high-amplitude
    low-frequency bulk.

    Without this, a global time-domain / MSE / R loss on ~1/f EEG is
    low-frequency-dominated, so the network UNDER-ALLOCATES capacity to the
    fast morphology (spikes, LVFA) the detail bands were added to preserve —
    a width sweep would then plateau on global R and be misread as
    "latent too small" when the real failure is allocation, not capacity.

    Mirrors `per_band_prd` (the validation metric) so train and eval agree
    in DIRECTION. NOTE the band split here is a rectangular FFT mask
    (differentiable) whereas `per_band_prd` uses scipy `sosfiltfilt`
    (Butterworth); the rectangular mask has edge leakage, so the training
    loss VALUE will not numerically track the validation per-band PRD — only
    the gradient direction (allocate to the low-energy high-freq bands) is
    shared. The per-band ratio is clamped so a near-silent band cannot
    explode the gradient.

    Returns a torch scalar in ``[0, ~RATIO_CAP]`` (lower = better).
    """
    import torch
    r = reconstructed.float()
    o = original.float()
    n = o.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / fs, device=o.device)
    R = torch.fft.rfft(r, dim=-1)
    O = torch.fft.rfft(o, dim=-1)
    nyq = fs / 2.0
    terms = []
    for _, (lo, hi) in EEG_BANDS.items():
        hi = min(hi, nyq)
        if lo >= hi:
            continue
        mask = ((freqs >= lo) & (freqs < hi)).to(O.dtype)
        r_b = torch.fft.irfft(R * mask, n=n, dim=-1)
        o_b = torch.fft.irfft(O * mask, n=n, dim=-1)
        num = torch.sqrt(((r_b - o_b) ** 2).sum(dim=-1) + eps)
        den = torch.sqrt((o_b ** 2).sum(dim=-1) + eps)
        # Clamp the relative error so a near-silent target band (den ≈ √eps)
        # cannot blow up the gradient (cf. band_aware's weight clamp).
        terms.append((num / den).clamp(max=4.0).mean())
    if not terms:
        return r.new_zeros(())
    return torch.stack(terms).mean()


__all__ = [
    'EEG_BANDS',
    'prd_numpy', 'prd_torch', 'pearson_r_numpy', 'pearson_r_torch',
    'per_band_prd', 'per_band_r',
    'lqs_compliance', 'lqs_pretty',
    'asymmetric_eeg_loss', 'band_aware_asymmetric_loss',
    'per_band_relative_loss',
]
