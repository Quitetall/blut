#!/usr/bin/env python3
"""
LamQuant Gen 7.5 — Combined L3 Teacher + Vocos Decoder Training
================================================================
Trains both models simultaneously on the same data with one DataLoader
and one frozen student forward pass per batch. Saves ~40% time vs
training them separately (12h instead of 20h).

The teacher can stop early while the decoder continues training.

Usage:
  python train_combined.py --teacher-epochs 200 --decoder-epochs 300 --decoder-tier 3
  python train_combined.py --teacher-epochs 200 --decoder-epochs 300 --batch-size 64
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'decoder'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'common'))  # MOVE-B: common DTOs
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'student'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'oracle'))

from lamquant_neural.models.vocos_decoder import VocosDecoder
from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
from train_teacher import L3Teacher
from lamquant.common.losses import pearson_r_loss
from streaming_dataset import PrecomputedL3Dataset
from raw_window_dataset import RawWindowDataset
from auraloss.freq import MultiResolutionSTFTLoss


def pearson_r_batch(pred, target):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    pc = p - p.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    r = (pc * tc).sum(dim=-1) / (
        torch.sqrt((pc ** 2).sum(dim=-1)) * torch.sqrt((tc ** 2).sum(dim=-1)) + 1e-8)
    return r.mean().item()


def prd_batch(pred, target):
    """Percentage Root-mean-square Difference: 100 × ||x - x̂|| / ||x||.
    Standard EEG compression quality metric. Lower is better.
    PRD < 1%: clinically indistinguishable. < 3%: good. < 9%: acceptable.
    """
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    diff_norm = torch.sqrt(((p - t) ** 2).sum(dim=-1))
    sig_norm = torch.sqrt((t ** 2).sum(dim=-1)).clamp(min=1e-8)
    return (100.0 * diff_norm / sig_norm).mean().item()


def validate_teacher(teacher, val_loader, device):
    """Validation R and PRD on holdout data."""
    teacher.eval()
    rs, prds = [], []
    with torch.no_grad():
        for x_l3, _, _ in val_loader:
            x_l3 = x_l3.to(device)
            recon = teacher(x_l3)
            rs.append(pearson_r_batch(recon, x_l3))
            prds.append(prd_batch(recon, x_l3))
    return (np.mean(rs) if rs else 0.0, np.mean(prds) if prds else 0.0)


def main():
    parser = argparse.ArgumentParser(description='Combined L3 Teacher + Vocos Decoder Training')
    parser.add_argument('--teacher-epochs', type=int, default=200)
    parser.add_argument('--decoder-epochs', type=int, default=300)
    parser.add_argument('--decoder-tier', type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument('--teacher-width', type=int, default=512)
    parser.add_argument('--teacher-strides', type=str, default='1,2,2',
                        help='Comma-separated stride pattern (e.g. 1,1,1,2,2 for 5 blocks)')
    parser.add_argument('--channel-attn', action='store_true',
                        help='Enable channel-aware spatial attention in teacher encoder')
    parser.add_argument('--bottleneck-attn', action='store_true',
                        help='Enable multi-head self-attention before latent projection')
    parser.add_argument('--teacher-r-loss', type=float, default=0.0,
                        help='Weight on Pearson R loss for teacher (0=off, 0.5=refinement)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--teacher-lr', type=float, default=1e-3)
    parser.add_argument('--decoder-lr', type=float, default=5e-4)
    parser.add_argument('--lr-min', type=float, default=1e-6)
    parser.add_argument('--windows-per-epoch', type=int, default=100000)
    parser.add_argument('--max-windows', type=int, default=500000)
    parser.add_argument('--student-checkpoint', type=str, default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--teacher-init', type=str, default=None,
                        help='Warm-start teacher from checkpoint')
    parser.add_argument('--decoder-init', type=str, default=None,
                        help='Warm-start decoder from checkpoint')
    # ---- LMA-direct training (BLUT canonical, ADR 0017) ----
    parser.add_argument('--lma-root', type=str, default=None,
                        help='Directory of per-recording .lma archives. When set '
                             'with --split-manifest, training reads LMA directly.')
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

    total_epochs = max(args.teacher_epochs, args.decoder_epochs)

    # --- Data (LMA-direct when supplied, else legacy NPZ manifest) ---
    if args.lma_root is not None and args.split_manifest is not None:
        from lamquant_codec.training import LmaL3Dataset, LmaSignalDataset, load_split_stems
        train_stems, _ = load_split_stems(args.split_manifest, "train")
        val_stems, _ = load_split_stems(args.split_manifest, "val")
        print(f"[*] LMA-direct: root={args.lma_root}, "
              f"train_stems={len(train_stems)}, val_stems={len(val_stems)}")
        dataset = LmaL3Dataset(
            lma_root=args.lma_root, file_stems=train_stems,
            windows_per_epoch=args.windows_per_epoch,
            max_windows=args.max_windows,
        )
        val_dataset = LmaL3Dataset(
            lma_root=args.lma_root, file_stems=val_stems,
            windows_per_epoch=5000,
        )
        decoder_raw_dataset = None
        if args.decoder_tier >= 3:
            decoder_raw_dataset = LmaSignalDataset(
                lma_root=args.lma_root, file_stems=train_stems,
                windows_per_epoch=min(args.max_windows or 50000, 50000),
            )
            decoder_raw_loader = torch.utils.data.DataLoader(
                decoder_raw_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=(device.type == 'cuda'))
    else:
        q31_dir = os.path.join(ROOT_DIR, 'lamquant/dataset/q31_events')
        all_files = sorted(glob.glob(os.path.join(q31_dir, '*.npz')))
        if not all_files:
            print(f"[!] No Q31 files in {q31_dir}"); sys.exit(1)

        sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant'))
        from data_types import DatasetManifest, Split
        manifest = DatasetManifest.load(os.path.join(
            ROOT_DIR, 'lamquant', 'dataset', 'manifest_v3.json'))
        train_files = [str(p) for p in manifest.get_files(Split.TRAIN)]
        val_files = [str(p) for p in manifest.get_files(Split.VAL)]
        print(f"[*] Split: {len(train_files)} train, {len(val_files)} val (holdout)")

        dataset = PrecomputedL3Dataset(train_files, windows_per_epoch=args.windows_per_epoch,
                                        max_windows=args.max_windows)
        val_dataset = PrecomputedL3Dataset(val_files, windows_per_epoch=5000)

        # Route B decoder dataset: paired L3 + raw [21, 2500] for full-signal training
        decoder_raw_dataset = None
        if args.decoder_tier >= 3:
            decoder_raw_dataset = RawWindowDataset(
                train_files, windows_per_epoch=args.windows_per_epoch,
                max_windows=min(args.max_windows or 50000, 50000))
            decoder_raw_loader = torch.utils.data.DataLoader(
                decoder_raw_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=(device.type == 'cuda'))

    # Shard-based GPU batching
    class _PrefetchLoader:
        def __init__(self, ds, bs, dev):
            self._ds, self._bs, self._dev = ds, bs, dev
        def __len__(self):
            return self._ds.windows_per_epoch // self._bs
        def __iter__(self):
            return self._ds.prefetch_batches(self._bs, self._dev)

    loader = _PrefetchLoader(dataset, args.batch_size, device)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda'))

    # --- Frozen student encoder ---
    student_ckpt = args.student_checkpoint
    if student_ckpt is None:
        for p in [
            os.path.join(ROOT_DIR, 'ai_models/student/student_hardened.ckpt'),
            os.path.join(ROOT_DIR, 'ai_models/student/student_subband_hardened.ckpt'),
            os.path.join(ROOT_DIR, 'weights/student_subband.ckpt'),
            os.path.join(ROOT_DIR, 'weights/student_subband_fast.ckpt'),
        ]:
            if os.path.exists(p):
                student_ckpt = p; break
    if student_ckpt is None:
        print("[!] No student checkpoint found"); sys.exit(1)

    student = TernaryMobileNetV5_Subband.from_checkpoint(student_ckpt, device=device).eval()
    for p in student.parameters():
        p.requires_grad = False

    # --- Models ---
    strides = [int(s) for s in args.teacher_strides.split(',')]
    teacher = L3Teacher(width=args.teacher_width, strides=strides,
                        channel_attn=args.channel_attn,
                        bottleneck_attn=args.bottleneck_attn).to(device)
    decoder = VocosDecoder(tier=args.decoder_tier).to(device)

    # Warm-start from checkpoints
    # Attention module names — excluded from warm-start so zero-init is preserved
    _attn_prefixes = ('channel_encoding', 'bn_attn')

    def _load_ckpt(model, path, label):
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        # Filter shape mismatches AND attention modules (preserve zero-init)
        model_sd = model.state_dict()
        compatible = {k: v for k, v in sd.items()
                      if k in model_sd and v.shape == model_sd[k].shape
                      and not any(ap in k for ap in _attn_prefixes)}
        new_params = [k for k in model_sd if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        print(f"[*] {label} warm-start from: {path} (R={ckpt.get('best_r', '?')})")
        print(f"    Loaded {len(compatible)}/{len(model_sd)} params, "
              f"{len(new_params)} new (zero-init attention)")
        if new_params:
            print(f"    New: {', '.join(new_params[:5])}{'...' if len(new_params) > 5 else ''}")

    if args.teacher_init:
        _load_ckpt(teacher, args.teacher_init, "Teacher")
    if args.decoder_init:
        _load_ckpt(decoder, args.decoder_init, "Decoder")

    # torch.compile both models
    if device.type == 'cuda':
        try:
            teacher = torch.compile(teacher, mode="default", dynamic=False)
            decoder = torch.compile(decoder, mode="default", dynamic=False)
            print(f"[*] Teacher + Decoder compiled (mode=default)")
        except Exception as e:
            print(f"[*] Compile failed: {e}")

    t_params = sum(p.numel() for p in teacher.parameters())
    d_params = sum(p.numel() for p in decoder.parameters())
    s_params = sum(p.numel() for p in student.parameters())

    print(f"[*] Combined Training on {device}")
    print(f"    Teacher: {t_params:,} params (width={args.teacher_width}), {args.teacher_epochs} epochs")
    print(f"    Decoder: {d_params:,} params (Tier {args.decoder_tier}), {args.decoder_epochs} epochs")
    print(f"    Student: frozen, {s_params:,} params")
    print(f"    Dataset: {len(dataset)} windows/epoch, bs={args.batch_size}")

    # Compile warmup + shard calibration
    if device.type == 'cuda':
        with torch.no_grad():
            _dummy = torch.randn(args.batch_size, 21, 313, device=device)
            try:
                teacher(_dummy)
            except Exception:
                pass
        torch.cuda.empty_cache()
        dataset.calibrate_shard_budget(device)
    print(f"    Total: {total_epochs} epochs (teacher stops at {args.teacher_epochs})")

    # --- Optimizers ---
    # Separate LR for attention modules (new, random init) vs conv blocks (pretrained)
    # Attention gets full LR, conv blocks get 10× lower if warm-starting
    _teacher_raw = getattr(teacher, '_orig_mod', teacher)
    _attn_names = {'channel_encoding', 'bn_attn'}
    _attn_params = [p for n, p in _teacher_raw.named_parameters()
                    if any(a in n for a in _attn_names)]
    _conv_params = [p for n, p in _teacher_raw.named_parameters()
                    if not any(a in n for a in _attn_names)]
    if args.teacher_init and (args.channel_attn or args.bottleneck_attn):
        # Warm-start: conv blocks already converged, lower LR
        conv_lr = args.teacher_lr / 10
        print(f"    Teacher LR: attention={args.teacher_lr:.0e}, conv={conv_lr:.0e} (10× lower, pretrained)")
        t_opt = torch.optim.AdamW([
            {'params': _attn_params, 'lr': args.teacher_lr},
            {'params': _conv_params, 'lr': conv_lr},
        ], weight_decay=1e-4)
    else:
        t_opt = torch.optim.AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=1e-4)
    d_opt = torch.optim.AdamW(decoder.parameters(), lr=args.decoder_lr, weight_decay=1e-4)
    t_sched = torch.optim.lr_scheduler.CosineAnnealingLR(t_opt, args.teacher_epochs, eta_min=args.lr_min)
    d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(d_opt, args.decoder_epochs, eta_min=args.lr_min)

    # --- Loss ---
    fft_sizes = [16, 32, 64, 128, 256]
    spec_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes,
        hop_sizes=[max(n // 4, 1) for n in fft_sizes],
        win_lengths=fft_sizes,
    ).to(device)

    # --- Checkpoints ---
    ckpt_dir = os.path.join(ROOT_DIR, 'lamquant')
    t_best_path = os.path.join(ckpt_dir, 'oracle/l3_teacher_best.ckpt')
    d_best_path = os.path.join(ckpt_dir, 'decoder/vocos_tier{}_best.ckpt'.format(args.decoder_tier))
    best_t_r = 0.0
    best_d_r = 0.0

    train_start = time.time()
    teacher_active = True

    for epoch in range(1, total_epochs + 1):
        if epoch > args.teacher_epochs and teacher_active:
            teacher_active = False
            print(f"\n[*] Teacher training complete at epoch {epoch-1}. Decoder continues.")

        teacher.train() if teacher_active else teacher.eval()
        decoder.train()
        ep_start = time.time()

        t_losses, d_losses, t_rs, d_rs = [], [], [], []

        for batch_idx, (x_l3, _, _) in enumerate(loader):
            x_l3 = x_l3.to(device, non_blocking=True)

            # Shared: frozen student encode (done once)
            with torch.no_grad():
                latent = student.encode(x_l3, quantize=True)

            # --- Teacher branch ---
            if teacher_active:
                t_opt.zero_grad()
                with torch.amp.autocast(device.type, dtype=torch.bfloat16,
                                         enabled=(device.type == 'cuda')):
                    t_recon = teacher(x_l3)
                    t_loss = F.mse_loss(t_recon, x_l3)
                    if args.teacher_r_loss > 0:
                        t_loss = t_loss + args.teacher_r_loss * pearson_r_loss(t_recon, x_l3)
                    if batch_idx % 4 == 0:  # spectral every 4th batch (expensive)
                        t_loss = t_loss + spec_loss_fn(t_recon, x_l3)
                t_loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
                t_opt.step()
                t_losses.append(t_loss.item())
                with torch.no_grad():
                    t_rs.append(pearson_r_batch(t_recon, x_l3))

            # --- Decoder branch ---
            # Route B (Tier 3+): get raw [21, 2500] target from paired dataset
            if decoder_raw_dataset is not None:
                try:
                    d_l3, d_raw = next(_decoder_iter)
                except (StopIteration, NameError):
                    _decoder_iter = iter(decoder_raw_loader)
                    d_l3, d_raw = next(_decoder_iter)
                d_raw = d_raw.to(device, non_blocking=True)
                d_l3 = d_l3.to(device, non_blocking=True)
                with torch.no_grad():
                    d_latent = student.encode(d_l3, quantize=True)
            else:
                d_raw = None
                d_latent = latent

            d_opt.zero_grad()
            with torch.amp.autocast(device.type, dtype=torch.bfloat16,
                                     enabled=(device.type == 'cuda')):
                d_recon = decoder(d_latent)
                if d_raw is not None:
                    target = d_raw  # Route B: real full-signal [21, 2500]
                elif decoder.output_mode == 'direct':
                    target = x_l3
                else:
                    target = F.interpolate(x_l3, size=d_recon.shape[2], mode='linear', align_corners=False)
                d_loss = F.l1_loss(d_recon, target) + 0.5 * pearson_r_loss(d_recon, target)
                if batch_idx % 4 == 0:
                    d_loss = d_loss + spec_loss_fn(d_recon, target)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            d_opt.step()
            d_losses.append(d_loss.item())
            with torch.no_grad():
                d_rs.append(pearson_r_batch(d_recon, target))

        if teacher_active:
            t_sched.step()
        if epoch <= args.decoder_epochs:
            d_sched.step()

        ep_sec = time.time() - ep_start
        elapsed = time.time() - train_start
        remaining = elapsed / epoch * (total_epochs - epoch)
        eta_h, eta_m = divmod(int(remaining), 3600)
        eta_m //= 60

        t_r = np.mean(t_rs) if t_rs else 0.0
        d_r = np.mean(d_rs) if d_rs else 0.0

        # Save best checkpoints
        t_tag = d_tag = ''
        if teacher_active and t_r > best_t_r:
            best_t_r = t_r
            t_tag = ' *T*'
            torch.save({
                'model_state_dict': teacher.state_dict(),
                'encoder_state_dict': teacher.encoder.state_dict(),
                'epoch': epoch, 'best_r': best_t_r, 'width': args.teacher_width,
            }, t_best_path)
        if d_r > best_d_r:
            best_d_r = d_r
            d_tag = ' *D*'
            torch.save({
                'model_state_dict': decoder.state_dict(),
                'epoch': epoch, 'best_r': best_d_r, 'tier': args.decoder_tier,
            }, d_best_path)

        # Validation every 10 epochs
        val_str = ""
        if epoch % 10 == 0 and teacher_active:
            val_t_r, val_t_prd = validate_teacher(teacher, val_loader, device)
            val_str = f"  valR={val_t_r:.4f} PRD={val_t_prd:.2f}%"

        t_str = f"T:L={np.mean(t_losses):.4f} R={t_r:.4f}" if teacher_active else "T:done"
        print(f"E{epoch:3d}/{total_epochs}  {t_str}  D:L={np.mean(d_losses):.4f} R={d_r:.4f}  "
              f"best_T={best_t_r:.4f} best_D={best_d_r:.4f}  {ep_sec:.0f}s  "
              f"ETA={eta_h}h{eta_m:02d}m{val_str}{t_tag}{d_tag}")

    # Save finals
    torch.save(teacher.state_dict(),
               os.path.join(ckpt_dir, f'oracle/l3_teacher_{args.teacher_epochs}_completed.ckpt'))
    torch.save(decoder.state_dict(),
               os.path.join(ckpt_dir, f'decoder/vocos_tier{args.decoder_tier}_{args.decoder_epochs}_completed.ckpt'))

    total_h = (time.time() - train_start) / 3600
    print(f"\n[*] Combined training complete in {total_h:.1f}h")
    print(f"    Teacher: best R={best_t_r:.4f} ({t_best_path})")
    print(f"    Decoder: best R={best_d_r:.4f} ({d_best_path})")


if __name__ == '__main__':
    main()
