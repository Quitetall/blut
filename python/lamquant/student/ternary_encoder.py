"""Train-only helpers for the ternary student (MOVE-B, 2026-05-29).

The architecture classes (TernaryMobileNetV5*, TernaryConv1d, blocks,
etc.) live in the PRIVATE ``lamquant_neural.models`` package and are
imported directly from there by the trainers. This file used to ALSO
re-export those classes for backward compatibility; that re-export was
redundant (no importer pulled the classes through here — only the
train-only functions below) and is dropped in the Neural<->BLUT
boundary migration. Import the model defs from ``lamquant_neural.models``.
"""
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Training-only functions
# ============================================================

def apply_montage_permutation(x):
    B, C, T = x.shape
    perm = torch.randperm(C)
    return x[:, perm, :]


def clinical_augmentation(x, fs=250.0):
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


def distillation_loss(s_recon, t_recon, kl_weight=1.0):
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
