"""
EEG-adapted adversarial discriminators for Vocos decoder training.

Two discriminator families adapted from audio codec literature:

1. Multi-Period Discriminator (MPD) — captures EEG rhythm periodicities
   Periods [5, 7, 13, 25, 41] target gamma through theta at 250 Hz.

2. Multi-Scale STFT Discriminator (MS-STFT) — captures spectral structure
   Windows [32, 64, 128, 256] give 3.9-0.98 Hz resolution across EEG bands.

Total ensemble: ~1-3M params (proportionate to EEG's 50 Hz bandwidth,
vs 10-50M for audio's 20 kHz bandwidth).

Usage:
    from discriminator import EEGDiscriminator
    disc = EEGDiscriminator().to(device)

    # In training loop:
    real_scores, real_feats = disc(real_eeg)
    fake_scores, fake_feats = disc(fake_eeg.detach())
    d_loss = disc.discriminator_loss(real_scores, fake_scores)
    g_loss, feat_loss = disc.generator_loss(real_scores, fake_scores,
                                             real_feats, fake_feats)

References:
    - HiFi-GAN (Kong et al., 2020): MPD + MSD design
    - EnCodec (Défossez et al., 2023): MS-STFT discriminator with LayerNorm
    - BigVGAN-v2 (2024): CQT discriminator, anti-aliased activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Period Sub-Discriminator (one per period)
# ============================================================

class PeriodSubDiscriminator(nn.Module):
    """Single sub-discriminator for one period.

    Reshapes 1D signal into 2D (period × time/period) and applies
    2D convolutions to detect periodic patterns at that period.

    For EEG at 250 Hz:
      period=5  → 50 Hz (gamma)
      period=7  → 35.7 Hz (high beta)
      period=13 → 19.2 Hz (beta)
      period=25 → 10 Hz (alpha)
      period=41 → 6.1 Hz (theta)
    """

    def __init__(self, period, channels=None):
        super().__init__()
        self.period = period
        # Smaller channel progression than audio (32→128 vs 64→1024)
        if channels is None:
            channels = [1, 32, 64, 128, 128]
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            stride = (3, 1) if i < len(channels) - 2 else (3, 1)
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], (5, 1),
                          stride=stride, padding=(2, 0)),
                nn.GroupNorm(1, channels[i + 1]),  # Instance norm (GroupNorm(1))
                nn.LeakyReLU(0.1),
            ))
        self.output = nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0))

    def forward(self, x):
        """x: [B, 1, T] → score + feature list."""
        B, C, T = x.shape
        # Pad to multiple of period
        if T % self.period != 0:
            pad = self.period - (T % self.period)
            x = F.pad(x, (0, pad), mode='reflect')
            T = T + pad
        # Reshape: [B, 1, T] → [B, 1, T/period, period]
        x = x.view(B, C, T // self.period, self.period)

        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        score = self.output(x)
        return score, features


# ============================================================
# Multi-Period Discriminator
# ============================================================

class MultiPeriodDiscriminator(nn.Module):
    """MPD with EEG-adapted periods targeting neural oscillation bands."""

    # Primes/near-primes spanning gamma (50Hz) through theta (6Hz) at 250 Hz
    DEFAULT_PERIODS = [5, 7, 13, 25, 41]

    def __init__(self, periods=None):
        super().__init__()
        periods = periods or self.DEFAULT_PERIODS
        self.discriminators = nn.ModuleList([
            PeriodSubDiscriminator(p) for p in periods
        ])

    def forward(self, x):
        """x: [B, C, T] → (scores_list, features_list)."""
        # Flatten channels: [B, 21, T] → [B*21, 1, T]
        B, C, T = x.shape
        x_flat = x.reshape(B * C, 1, T)

        all_scores = []
        all_features = []
        for disc in self.discriminators:
            score, feats = disc(x_flat)
            all_scores.append(score)
            all_features.append(feats)
        return all_scores, all_features


# ============================================================
# Multi-Scale STFT Sub-Discriminator
# ============================================================

class STFTSubDiscriminator(nn.Module):
    """Single STFT discriminator at one resolution.

    Operates on complex STFT (real + imaginary concatenated as 2 channels).
    Uses LayerNorm (EnCodec finding: only normalization that works reliably).
    """

    def __init__(self, n_fft, hop_length=None, channels=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        n_bins = n_fft // 2 + 1

        if channels is None:
            channels = [2, 32, 64, 128]  # 2 input channels (real + imag)

        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], (3, 3),
                          stride=(1, 1), padding=(1, 1)),
                nn.GroupNorm(1, channels[i + 1]),  # Instance norm
                nn.LeakyReLU(0.1),
            ))
        self.output = nn.Conv2d(channels[-1], 1, (3, 3), padding=(1, 1))

    def forward(self, x):
        """x: [B, 1, T] → score + feature list."""
        B = x.shape[0]
        # STFT
        x_1d = x.squeeze(1)  # [B, T]
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        stft = torch.stft(x_1d, self.n_fft, hop_length=self.hop_length,
                          window=window, return_complex=True)  # [B, n_bins, n_frames]
        # Stack real + imag as 2 channels: [B, 2, n_bins, n_frames]
        x_2d = torch.stack([stft.real, stft.imag], dim=1)

        features = []
        for conv in self.convs:
            x_2d = conv(x_2d)
            features.append(x_2d)
        score = self.output(x_2d)
        return score, features


# ============================================================
# Multi-Scale STFT Discriminator
# ============================================================

class MultiScaleSTFTDiscriminator(nn.Module):
    """MS-STFT discriminator at multiple resolutions covering EEG bands.

    Window sizes [32, 64, 128, 256] at 250 Hz give frequency resolutions
    from 7.8 Hz (coarse temporal) to 0.98 Hz (fine spectral).
    """

    DEFAULT_FFT_SIZES = [32, 64, 128, 256]

    def __init__(self, fft_sizes=None):
        super().__init__()
        fft_sizes = fft_sizes or self.DEFAULT_FFT_SIZES
        self.discriminators = nn.ModuleList([
            STFTSubDiscriminator(n_fft=n) for n in fft_sizes
        ])

    def forward(self, x):
        """x: [B, C, T] → (scores_list, features_list)."""
        B, C, T = x.shape
        x_flat = x.reshape(B * C, 1, T)

        all_scores = []
        all_features = []
        for disc in self.discriminators:
            score, feats = disc(x_flat)
            all_scores.append(score)
            all_features.append(feats)
        return all_scores, all_features


# ============================================================
# Combined EEG Discriminator
# ============================================================

class EEGDiscriminator(nn.Module):
    """Combined MPD + MS-STFT discriminator for EEG adversarial training.

    Total ~1-2M parameters. Proportionate to EEG's 50 Hz bandwidth
    (vs 10-50M for audio's 20 kHz bandwidth).

    Training recipe (from audio codec literature):
      1. Warm up decoder 200-300 epochs with reconstruction loss only
      2. Add discriminator with ramped adversarial weight (0.1 → 1.0)
      3. Train 100-200 more epochs with full loss suite
      4. Generator:discriminator LR ratio = 2:1
    """

    def __init__(self, periods=None, fft_sizes=None):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods)
        self.ms_stft = MultiScaleSTFTDiscriminator(fft_sizes)

    def forward(self, x):
        """Returns (all_scores, all_features) from both discriminator families."""
        mpd_scores, mpd_feats = self.mpd(x)
        stft_scores, stft_feats = self.ms_stft(x)
        return mpd_scores + stft_scores, mpd_feats + stft_feats

    @staticmethod
    def discriminator_loss(real_scores, fake_scores):
        """Hinge loss for discriminator update."""
        loss = 0.0
        for real_s, fake_s in zip(real_scores, fake_scores):
            loss += F.relu(1.0 - real_s).mean() + F.relu(1.0 + fake_s).mean()
        return loss / len(real_scores)

    @staticmethod
    def generator_loss(real_scores, fake_scores, real_feats, fake_feats,
                       feat_weight=2.0):
        """Hinge adversarial + feature matching loss for generator update.

        Returns (adv_loss, feat_match_loss).
        """
        # Adversarial: generator wants fake scores to be positive
        adv_loss = 0.0
        for fake_s in fake_scores:
            adv_loss += -fake_s.mean()
        adv_loss /= len(fake_scores)

        # Feature matching: L1 between discriminator intermediates
        feat_loss = 0.0
        n_feats = 0
        for real_f_list, fake_f_list in zip(real_feats, fake_feats):
            for real_f, fake_f in zip(real_f_list, fake_f_list):
                feat_loss += F.l1_loss(fake_f, real_f.detach())
                n_feats += 1
        feat_loss = feat_weight * feat_loss / max(n_feats, 1)

        return adv_loss, feat_loss
