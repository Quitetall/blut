"""
Frequency-Weighted MSE Loss for EEG Neural Codecs.

Based on FEMBA's physiologically-aware pre-training objective (2026):
errors in clinically relevant frequency bands (delta, theta, alpha, beta)
are penalized more heavily than errors in artifact-prone high-frequency
bands (gamma, muscle). This teaches the codec to prioritize faithful
reconstruction of neural oscillations over noise.

Usage:
    loss_fn = FrequencyWeightedMSE(sample_rate=250.0)
    loss = loss_fn(pred, target)  # [B, C, T] tensors

The band weights are derived from clinical EEG interpretation priorities:
  - Delta (0.5-4 Hz):  3.0× — critical for seizure, sleep staging
  - Theta (4-8 Hz):    2.5× — drowsiness, memory, temporal lobe
  - Alpha (8-13 Hz):   2.0× — resting state, posterior dominant rhythm
  - Beta (13-30 Hz):   1.5× — active cognition, motor planning
  - Gamma (30-70 Hz):  0.5× — often contaminated by EMG artifact
  - HF (70-125 Hz):    0.2× — mostly muscle artifact, minimal neural content
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Clinical EEG frequency band definitions and weights
# (low_hz, high_hz, weight)
DEFAULT_BAND_WEIGHTS = [
    (0.5,   4.0,  3.0),   # delta
    (4.0,   8.0,  2.5),   # theta
    (8.0,  13.0,  2.0),   # alpha
    (13.0, 30.0,  1.5),   # beta
    (30.0, 70.0,  0.5),   # gamma
    (70.0, 125.0, 0.2),   # high-frequency (artifact)
]


class FrequencyWeightedMSE(nn.Module):
    """MSE loss weighted by EEG frequency band clinical importance.

    Computes the error spectrum via STFT, applies per-band weights,
    and returns the weighted mean squared error. This is additive with
    the existing EventWeightedMSELoss (seizure mask weighting) and
    SpectralConvergenceLoss.
    """

    def __init__(self, sample_rate=250.0, n_fft=256, hop_length=64,
                 band_weights=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bands = band_weights or DEFAULT_BAND_WEIGHTS

        # Precompute frequency bin → weight mapping
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        weights = torch.ones_like(freqs)
        for low_hz, high_hz, w in self.bands:
            mask = (freqs >= low_hz) & (freqs < high_hz)
            weights[mask] = w
        # Normalize so mean weight = 1 (preserves loss scale)
        weights = weights / weights.mean()
        self.register_buffer('freq_weights', weights)

    def forward(self, pred, target):
        """
        pred, target: [B, C, T] float tensors (EEG in microvolts)
        returns: scalar loss
        """
        error = pred - target  # [B, C, T]
        B, C, T = error.shape

        # STFT of the error signal per channel
        # Reshape to [B*C, T] for batched STFT
        error_flat = error.reshape(B * C, T)
        window = torch.hann_window(self.n_fft, device=error.device, dtype=error.dtype)
        spec = torch.stft(
            error_flat, n_fft=self.n_fft, hop_length=self.hop_length,
            window=window, return_complex=True
        )  # [B*C, n_fft//2+1, n_frames]

        # Power spectrum of the error
        power = spec.abs().pow(2)  # [B*C, F, T_frames]

        # Apply frequency weights: [F] broadcast to [B*C, F, T_frames]
        weighted_power = power * self.freq_weights.unsqueeze(0).unsqueeze(-1)

        # Mean over all dimensions
        return weighted_power.mean()
