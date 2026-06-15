#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# losses.py — shared training LOSS primitives for the LamQuant codec.
#
# Extracted from the (now-archived) per-architecture trainers so that EVERY
# trainer imports its loss/objective helpers from ONE shared home instead of
# reaching into a sibling trainer file. Pairs with:
#   - common/metrics.py     (R / PRD / LQS metrics, incl. pearson_r_batch)
#   - common/augment.py     (channel dropout, EEG/clinical augmentations)
#
# All functions are pure torch (no trainer/dataset deps) so they import
# cheaply and are safe to use from any trainer (student/oracle/snn/decoder).

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """Multi-resolution STFT loss tuned for L3 subband (313 samples).

    Window sizes {16, 32} catch fast spike components (2-8 samples in L3
    domain = 20-70ms at the original 250 Hz). {64, 128, 256} catch slow
    waves. No 512 — at 313 input samples it's mostly zero-pad interpolation.
    """
    def __init__(self, fft_sizes=None):
        super().__init__()
        # Avoid a shared mutable default (B006); same default values as before.
        self.fft_sizes = fft_sizes if fft_sizes is not None else [16, 32, 64, 128, 256]

    def forward(self, pred, target):
        loss = 0.0
        for n_fft in self.fft_sizes:
            hop = max(n_fft // 4, 1)
            win = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
            p = torch.stft(pred.reshape(-1, pred.shape[-1]).float(),
                           n_fft=n_fft, hop_length=hop, window=win,
                           return_complex=True).abs() + 1e-8
            t = torch.stft(target.reshape(-1, target.shape[-1]).float(),
                           n_fft=n_fft, hop_length=hop, window=win,
                           return_complex=True).abs() + 1e-8
            loss += F.mse_loss(torch.log10(p), torch.log10(t))
        return loss / len(self.fft_sizes)


def temporal_importance_mask(T, edge_weight=0.3, device=None):
    """Raised cosine mask: 1.0 at center, tapers to edge_weight at boundaries.

    L3 window edges carry less useful information due to DWT boundary effects.
    The inverse lifting will mostly overwrite edge samples with detail subbands.
    Concentrates encoder capacity on the clinically relevant center.
    """
    t = torch.linspace(0, 1, T, device=device)
    mask = edge_weight + (1.0 - edge_weight) * 0.5 * (1 - torch.cos(2 * torch.pi * t))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T] for broadcasting


def band_weighted_mse(recon, target, sample_rate=31.25):
    """Frequency-weighted MSE: emphasizes clinically important EEG bands.

    The L3 approximation covers 0-15.6 Hz (sample rate 31.25 Hz).
    Clinical EEG reading depends primarily on:
      Delta (0-4 Hz):   2.0x — seizures, encephalopathy, sleep staging
      Theta (4-8 Hz):   1.5x — drowsiness, temporal lobe epilepsy
      Alpha (8-13 Hz):  1.5x — posterior dominant rhythm, consciousness
      Upper (13-15.6):  1.0x — edge of L3 band, less diagnostic weight

    Uses Parseval's theorem: weighted MSE in frequency domain = weighted
    time-domain MSE after band decomposition. Normalized so that uniform
    weights reproduce standard F.mse_loss.
    """
    error = recon - target                          # [B, C, T]
    T = error.shape[-1]
    E = torch.fft.rfft(error, dim=-1)               # [B, C, T//2+1]
    n_freq = E.shape[-1]

    freqs = torch.linspace(0, sample_rate / 2, n_freq, device=error.device)
    w = torch.ones(n_freq, device=error.device)
    w[freqs < 4] = 2.0                              # delta
    w[(freqs >= 4) & (freqs < 8)] = 1.5             # theta
    w[(freqs >= 8) & (freqs < 13)] = 1.5            # alpha
    # 13-15.6 Hz stays 1.0

    # |E(f)|^2, corrected for one-sided spectrum
    power = E.real.pow(2) + E.imag.pow(2)            # [B, C, n_freq]
    scale = torch.full((n_freq,), 2.0, device=error.device)
    scale[0] = 1.0
    if T % 2 == 0:
        scale[-1] = 1.0

    weighted_power = power * w * scale
    # Parseval: sum(|x|^2) = (1/T)*sum(scale*|X|^2), so divide by T*numel
    return weighted_power.sum() / (T * error.numel())


def pearson_r_loss(pred, target):
    """Differentiable Pearson R loss: 1 - R, averaged over batch.

    Directly optimizes waveform shape correlation on L3 reconstructions.
    R is computed per-sample (flattened across channels × time), then
    averaged over the batch. Returns a scalar loss in [0, 2].

    NB: this is the *loss* form (returns a differentiable tensor). For the
    plain monitoring metric (a float), use
    ``lamquant.common.metrics.pearson_r_batch``; for a masked / channel-
    agnostic variant use ``masked_pearson_r_torch``.
    """
    p = pred.flatten(1)
    t = target.flatten(1)
    pc = p - p.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    r = torch.sum(pc * tc, dim=-1) / (
        torch.sqrt(torch.sum(pc ** 2, dim=-1)) *
        torch.sqrt(torch.sum(tc ** 2, dim=-1)) + 1e-8
    )
    return (1.0 - r).mean()


def distillation_loss(s_recon, t_recon, kl_weight=1.0):
    """Student↔teacher reconstruction distillation.

    Returns ``(mse, mse, r_mean)`` — the duplicated mse preserves the historic
    call signature (callers unpack three values). ``r_mean`` is the batch-mean
    Pearson R between student and teacher reconstructions (a monitoring scalar
    tensor). ``kl_weight`` is accepted for signature compatibility.
    """
    mse = F.mse_loss(s_recon, t_recon)
    s = s_recon.flatten(1)
    t = t_recon.flatten(1)
    sc = s - s.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    r = torch.sum(sc * tc, dim=-1) / (
        torch.sqrt(torch.sum(sc ** 2, dim=-1)) *
        torch.sqrt(torch.sum(tc ** 2, dim=-1)) + 1e-8
    )
    return mse, mse, r.mean()
