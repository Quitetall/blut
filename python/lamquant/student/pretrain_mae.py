#!/usr/bin/env python3
"""pretrain_mae.py — Masked Autoencoder pretraining for the EEG encoder.

Self-supervised pretraining on TUH + HBN + OpenNeuro before codec training.
The encoder learns EEG structure (oscillatory patterns, cross-channel
relationships, artifact signatures) without any compression objective.
Then the pretrained encoder initialises the joint training pipeline.

Architecture: the same TernaryMobileNetV5_Subband encoder, but with a
lightweight prediction head that reconstructs masked L3 patches. After
pretraining, the prediction head is discarded and the encoder weights
seed the joint codec.

Reference: He et al. MAE (CVPR 2022) adapted for 1D EEG.
Also: CBraMod (Wang ICLR 2025), LaBraM (Jiang ICLR 2024).

Usage:
    # Pretrain on the full training set
    python ai_models/student/pretrain_mae.py --epochs 100 --mask-ratio 0.5

    # Then use the pretrained encoder for joint training
    python ai_models/student/train_joint.py --config fast --tier 3 \\
        --encoder-init ai_models/student/pretrained_mae.ckpt

Pipeline:
    1. Load L3 windows from manifest (same dataset, no labels needed)
    2. Mask 50% of the L3 time-patches
    3. Encode the visible patches
    4. Predict the masked patches from the encoder output
    5. Loss = MSE on masked patches only
    6. Save encoder state_dict for downstream joint training
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = str(_REPO)
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'lamquant'))
sys.path.insert(0, str(_REPO / 'lamquant' / 'common'))  # MOVE-B: common DTOs
sys.path.insert(0, str(_REPO / 'lamquant' / 'student'))
sys.path.insert(0, str(_REPO / 'lamquant' / 'oracle'))


class MAEPredictionHead(nn.Module):
    """Lightweight head that predicts masked L3 patches.

    Input:  encoder output [B, latent_dim, T_latent] (e.g., [B, 32, 79])
    Output: predicted L3 patches [B, 21, 313] (same shape as input L3)

    Discarded after pretraining — only the encoder weights transfer.
    """

    def __init__(self, latent_dim: int = 32, n_channels: int = 21,
                 l3_len: int = 313):
        super().__init__()
        self.l3_len = l3_len
        self.n_channels = n_channels
        # Project latent → L3 shape via transpose conv
        self.proj = nn.Sequential(
            nn.Conv1d(latent_dim, 128, 1),
            nn.GELU(),
            nn.ConvTranspose1d(128, n_channels, kernel_size=4, stride=4),
            # Output: [B, 21, 316] — crop to 313
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.proj(latent)
        return x[..., :self.l3_len]


def create_mask(batch_size: int, n_channels: int, l3_len: int,
                mask_ratio: float = 0.5, patch_size: int = 16,
                device='cpu') -> torch.Tensor:
    """Create a binary mask for L3 patches. 1 = masked (to predict).

    Patches are contiguous blocks of `patch_size` timesteps across
    all channels simultaneously (same mask for all channels within
    a sample, different mask per sample in the batch).
    """
    n_patches = l3_len // patch_size
    n_masked = int(n_patches * mask_ratio)
    mask = torch.zeros(batch_size, 1, l3_len, device=device)
    for b in range(batch_size):
        # Random patch indices to mask
        masked_idx = torch.randperm(n_patches, device=device)[:n_masked]
        for idx in masked_idx:
            start = idx * patch_size
            end = min(start + patch_size, l3_len)
            mask[b, :, start:end] = 1.0
    return mask.expand(-1, n_channels, -1)


def run_pretraining(
    epochs: int = 100,
    mask_ratio: float = 0.5,
    patch_size: int = 16,
    lr: float = 1e-3,
    batch_size: int = 64,
    windows_per_epoch: int = 50000,
    max_windows: int = None,
    seed: int = 42,
    output_path: str = None,
    lma_root: str = None,
    split_manifest: str = None,
):
    """Run MAE pretraining on the L3 dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] MAE pretraining on {device}')
    print(f'    epochs={epochs}, mask_ratio={mask_ratio}, '
          f'patch_size={patch_size}, lr={lr}')

    # ---- Load data — LMA-direct (BLUT canonical) when --lma-root set,
    # else fall through to the deprecated NPZ + L3 precompute path. ----
    if lma_root is not None and split_manifest is not None:
        from lamquant_codec.training import LmaL3Dataset, load_split_stems
        train_stems, _ = load_split_stems(split_manifest, "train")
        print(f'[*] LMA-direct: root={lma_root}, train_stems={len(train_stems)}')
        train_ds = LmaL3Dataset(
            lma_root=lma_root, file_stems=train_stems,
            windows_per_epoch=windows_per_epoch,
            max_windows=max_windows,
            seed=seed,
        )
    else:
        from data_types import DatasetManifest, Split
        from streaming_dataset import PrecomputedL3Dataset

        manifest = DatasetManifest.load(
            os.path.join(ROOT_DIR, 'lamquant', 'dataset', 'manifest_v3.json'))
        train_entries = manifest.get_file_entries(Split.TRAIN)
        print(f'[*] Loaded manifest: {len(train_entries):,} train files')

        train_ds = PrecomputedL3Dataset(
            file_entries=train_entries,
            windows_per_epoch=windows_per_epoch,
            max_windows=max_windows,
        )

    # ---- Build encoder + prediction head ----
    from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
    encoder = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
    pred_head = MAEPredictionHead(latent_dim=32).to(device)
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in pred_head.parameters())
    print(f'[*] Encoder: {n_enc:,} params')
    print(f'[*] Prediction head: {n_head:,} params (discarded after pretraining)')

    # ADR 0050/0051 ingredient registry (uniform optimizer construction).
    from lamquant.ingredients import build_ingredient
    optimizer = build_ingredient(
        "optimizer", "adamw",
        {"lr": lr, "weight_decay": 1e-4, "betas": (0.9, 0.999),
         "fused": (device.type == 'cuda')},
        named_params=list(encoder.named_parameters())
        + list(pred_head.named_parameters()))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    # ---- Training loop ----
    best_loss = float('inf')
    t0 = time.time()

    for ep in range(1, epochs + 1):
        encoder.train()
        pred_head.train()
        loss_acc, n = 0.0, 0

        for batch in train_ds.prefetch_batches(batch_size=batch_size,
                                                device=device):
            l3 = batch[0]  # [B, 21, 313]
            B, C, T = l3.shape

            # Create mask: 1 = masked (to predict)
            mask = create_mask(B, C, T, mask_ratio, patch_size, device)

            # Zero out masked regions in the input
            l3_masked = l3 * (1.0 - mask)

            # Encode the visible patches
            latent = encoder.encode(l3_masked, quantize=False)

            # Predict full L3 from latent
            l3_pred = pred_head(latent)

            # Loss only on masked regions
            loss = F.mse_loss(l3_pred * mask, l3 * mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(pred_head.parameters()), 5.0)
            optimizer.step()
            loss_acc += float(loss.detach())
            n += 1

        scheduler.step()
        avg_loss = loss_acc / max(n, 1)

        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f'  [MAE] ep {ep:>4}/{epochs}  loss={avg_loss:.4f}  '
                  f'lr={scheduler.get_last_lr()[0]:.2e}  '
                  f'elapsed={elapsed/60:.1f}min')

        if avg_loss < best_loss:
            best_loss = avg_loss

    # ---- Save encoder weights ----
    out = Path(output_path or os.path.join(
        ROOT_DIR, 'lamquant', 'student', 'pretrained_mae.ckpt'))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': encoder.state_dict(),
        'pretraining': 'mae',
        'epochs': epochs,
        'mask_ratio': mask_ratio,
        'best_loss': best_loss,
        'seed': seed,
    }, out)
    wall = time.time() - t0
    print(f'\n[*] Done. Best loss={best_loss:.4f}  wall={wall/60:.1f}min')
    print(f'    Saved: {out}')
    print(f'    Use: train_joint.py --encoder-init {out}')
    return str(out)


def main() -> int:
    parser = argparse.ArgumentParser(prog='pretrain_mae')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mask-ratio', type=float, default=0.5)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--windows-per-epoch', type=int, default=50000)
    parser.add_argument('--max-windows', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    # ---- LMA-direct training (BLUT canonical, ADR 0017) ----
    parser.add_argument('--lma-root', type=str, default=None,
                        help='Directory of per-recording .lma archives. When set '
                             'with --split-manifest, training reads LMA directly.')
    parser.add_argument('--split-manifest', type=str, default=None,
                        help='JSON split manifest. Required when --lma-root is set.')
    args = parser.parse_args()

    run_pretraining(
        epochs=args.epochs, mask_ratio=args.mask_ratio,
        patch_size=args.patch_size, lr=args.lr,
        batch_size=args.batch_size,
        windows_per_epoch=args.windows_per_epoch,
        max_windows=args.max_windows,
        seed=args.seed, output_path=args.output,
        lma_root=args.lma_root,
        split_manifest=args.split_manifest,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
