#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — L3-Native Teacher Training
==============================================
Trains a high-capacity FP32 autoencoder on L3 subband approximation [21, 313].
Produces [32, 79] latents matching the student TNN exactly.

Purpose: distillation target for student hardening. The teacher's only advantage
over the student is FP32 precision — same input, same latent shape, same task.
Hardening then directly measures the cost of ternarization.

Architecture:
  8.3M FP32 params (18x student capacity), width=512
  StridedFocalBlock encoder (stride 4 total = 2x2)
  UpsampleFocalBlock decoder (symmetric)

Usage:
  python train_l3_teacher.py --epochs 300
  python train_l3_teacher.py --epochs 300 --batch-size 64 --lr 2e-3
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'oracle'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'student'))

from train_teacher import L3Teacher
from streaming_dataset import PrecomputedL3Dataset


def pearson_r_batch(pred, target):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    pc = p - p.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    r = (pc * tc).sum(dim=-1) / (
        torch.sqrt((pc ** 2).sum(dim=-1)) * torch.sqrt((tc ** 2).sum(dim=-1)) + 1e-8)
    return r.mean().item()


def main():
    parser = argparse.ArgumentParser(description='Train L3-native teacher')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr-min', type=float, default=1e-5)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--windows-per-epoch', type=int, default=400000)
    parser.add_argument('--max-windows', type=int, default=500000,
                        help='Cap total L3 windows loaded into RAM (default 500K ≈ 13 GB)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', action='store_true')
    # ---- LMA-direct training (BLUT canonical, ADR 0017) ----
    parser.add_argument('--lma-root', type=str, default=None,
                        help='Directory of per-recording .lma archives. When set '
                             'with --split-manifest, training reads LMA directly '
                             'instead of NPZ + L3 precompute.')
    parser.add_argument('--split-manifest', type=str, default=None,
                        help='JSON split manifest. Required when --lma-root is set.')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ADR 0050/0051 ingredient registry (uniform dataset/optimizer/loss build).
    from lamquant.ingredients import build_ingredient

    # Load L3 data — LMA-direct (BLUT canonical) when --lma-root set,
    # else fall through to the deprecated NPZ + L3 precompute path.
    if args.lma_root is not None and args.split_manifest is not None:
        print(f"[*] LMA-direct: root={args.lma_root}, manifest={args.split_manifest}")
        dataset = build_ingredient(
            "data", "lma_l3",
            {"lma_root": args.lma_root, "split_manifest": args.split_manifest,
             "windows_per_epoch": args.windows_per_epoch,
             "max_windows": args.max_windows})
    else:
        q31_dir = os.path.join(ROOT_DIR, 'ai_models/dataset_sim/q31_events')
        train_files = sorted(glob.glob(os.path.join(q31_dir, '*.npz')))
        if not train_files:
            print(f"[!] No Q31 files in {q31_dir}")
            sys.exit(1)
        dataset = PrecomputedL3Dataset(train_files, windows_per_epoch=args.windows_per_epoch,
                                        max_windows=args.max_windows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == 'cuda'))

    # Model
    model = L3Teacher(width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] L3 Teacher (width={args.width}) on {device}")
    print(f"    Params: {n_params:,} ({n_params * 4 / 1e6:.1f} MB FP32)")
    print(f"    Dataset: {len(dataset)} windows/epoch, bs={args.batch_size}")
    print(f"    Training: {args.epochs} epochs, lr={args.lr:.0e}")

    # Verify shapes
    with torch.no_grad():
        sample = torch.randn(1, 21, 313, device=device)
        lat = model.encode(sample)
        out = model(sample)
    print(f"    Shapes: input {list(sample.shape)} → latent {list(lat.shape)} → output {list(out.shape)}")
    assert lat.shape == torch.Size([1, 32, 79]), f"Latent shape mismatch: {lat.shape}"

    # ADR 0050/0051 ingredient registry (uniform optimizer construction).
    # build_ingredient was imported above (dataset build).
    optimizer = build_ingredient(
        "optimizer", "adamw",
        {"lr": args.lr, "weight_decay": 1e-4, "betas": (0.9, 0.999)},
        named_params=list(model.named_parameters()))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # Resume
    ckpt_dir = os.path.join(ROOT_DIR, 'ai_models/oracle')
    best_path = os.path.join(ckpt_dir, 'l3_teacher_best.ckpt')
    start_epoch = 0
    best_r = 0.0

    if args.resume and os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            best_r = ckpt.get('best_r', 0.0)
            print(f"    Resumed from epoch {start_epoch}, best R={best_r:.4f}")

    # ADR 0050/0051 loss ingredient — byte-identical F.mse_loss(recon, x_l3).
    loss_fn = build_ingredient("loss", "teacher_mse", {})

    # Training loop
    n_batches = len(loader)
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_start = time.time()
        losses, rs = [], []

        for batch_idx, (x_l3, _, _) in enumerate(loader):
            x_l3 = x_l3.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, dtype=torch.bfloat16,
                                     enabled=(device.type == 'cuda')):
                recon = model(x_l3)
                loss = loss_fn(recon, x_l3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())
            with torch.no_grad():
                r = pearson_r_batch(recon, x_l3)
                rs.append(r)

        scheduler.step()
        ep_sec = time.time() - ep_start
        mean_loss = np.mean(losses)
        mean_r = np.mean(rs)

        improved = ''
        if mean_r > best_r:
            best_r = mean_r
            improved = ' *BEST*'
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'epoch': epoch + 1,
                'best_r': best_r,
                'width': args.width,
            }, best_path)

        elapsed = time.time() - train_start
        remaining = elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch - 1)
        eta_h, eta_m = divmod(int(remaining), 3600)
        eta_m //= 60

        print(f"E{epoch+1:3d}/{args.epochs}  L={mean_loss:.6f}  R={mean_r:.4f}  "
              f"best={best_r:.4f}  {ep_sec:.0f}s  "
              f"LR={scheduler.get_last_lr()[0]:.2e}  "
              f"ETA={eta_h}h{eta_m:02d}m{improved}")

    # Save final
    final_path = os.path.join(ckpt_dir, f'l3_teacher_{args.epochs}_completed.ckpt')
    torch.save(model.state_dict(), final_path)

    # Also save encoder-only for hardening (same format as old teacher_best.ckpt)
    enc_path = os.path.join(ckpt_dir, 'l3_teacher_encoder.ckpt')
    torch.save(model.encoder.state_dict(), enc_path)

    total_h = (time.time() - train_start) / 3600
    print(f"\n[*] Training complete in {total_h:.1f}h. Best R: {best_r:.4f}")
    print(f"[*] Saved: {best_path} (best), {final_path} (final), {enc_path} (encoder)")


if __name__ == '__main__':
    main()
