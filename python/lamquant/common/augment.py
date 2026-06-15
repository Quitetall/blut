#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# augment.py — shared EEG data-augmentation primitives for LamQuant training.
#
# Extracted from the (now-archived) per-architecture trainers so every trainer
# pulls augmentations from ONE shared home rather than importing from a sibling
# trainer file. Pairs with common/losses.py + common/metrics.py.
#
# ``eeg_augment`` lazily imports ``selfeeg`` inside the function so importing
# this module never hard-requires the selfeeg dependency.

import numpy as np
import torch


def channel_dropout(x, p_min=5, p_max=13, training=True):
    """Randomly zero out 5-13 channels per batch element during training.

    Critical for deployment across 8/24/32-channel SKUs. The ADS1299
    provides 8 physical channels with remaining channels zero-filled.
    Training on full 21 channels and deploying on 8 is a domain shift
    that channel dropout directly addresses.

    Args:
        x: [B, 21, T] input tensor
        p_min/p_max: range of channels to zero (uniform random per sample)
        training: only apply during training
    """
    if not training:
        return x
    B, C, T = x.shape
    x_out = x.clone()
    for b in range(B):
        n_drop = torch.randint(p_min, p_max + 1, (1,)).item()
        drop_idx = torch.randperm(C)[:n_drop]
        x_out[b, drop_idx, :] = 0.0
    return x_out


def eeg_augment(x, training=True):
    """GPU EEG augmentations from selfeeg (applied after channel dropout).

    Conservative rates: Gaussian noise always (subtle), band noise 30%,
    temporal flip 10%. All operate on GPU tensors in-place.
    """
    if not training:
        return x
    from selfeeg import augmentation as seeg_aug
    x = seeg_aug.add_gaussian_noise(x, std=0.02)
    if torch.rand(1).item() < 0.3:
        x = seeg_aug.add_band_noise(x, bandwidth=2.0, samplerate=31.25)
    if torch.rand(1).item() < 0.1:
        x = seeg_aug.flip_horizontal(x)
    return x


def apply_montage_permutation(x):
    """Randomly permute channel order (montage-agnostic augmentation)."""
    B, C, T = x.shape
    perm = torch.randperm(C)
    return x[:, perm, :]


def clinical_augmentation(x, fs=250.0):
    """Inject clinically realistic artifacts: 60 Hz mains hum, EMG, electrode pop."""
    B, C, T = x.shape
    device = x.device
    t = torch.arange(T, device=device).float() / fs

    hum_amp = torch.rand(B, C, 1, device=device) * 2.0
    hum = hum_amp * torch.sin(2 * np.pi * 60.0 * t)
    x = x + hum

    emg = torch.randn_like(x) * (torch.rand(B, C, 1, device=device) * 1.5)
    x = x + emg

    if torch.rand(1) < 0.3:
        pop_ch = torch.randint(0, C, (1,)).item()
        pop_t = torch.randint(0, T, (1,)).item()
        pop_amp = (torch.rand(1).item() - 0.5) * 50.0
        x[:, pop_ch, pop_t:] += pop_amp

    return x
