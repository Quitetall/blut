#!/usr/bin/env python3
"""
LamQuant Gen 7.5 -- Vocos Decoder Training
===========================================
Trains a Vocos-style ConvNeXt decoder to reconstruct L3 subband EEG [21, 313]
from FSQ-dequantized latent [32, 79].

Two-phase training:
  Phase 1 (reconstruction): L1 + multi-res STFT + Pearson R
  Phase 2 (adversarial):    + EEG-adapted MPD + MS-STFT discriminator
                            + feature matching loss (ramped 0.1 -> 1.0)

Usage:
  # Phase 1 only (reconstruction):
  python train_vocos_decoder.py --tier 3 --epochs 300

  # Phase 1 + Phase 2 (with adversarial from epoch 200):
  python train_vocos_decoder.py --tier 3 --epochs 500 --adversarial --adv-start-epoch 300

  # Resume Phase 2 on a converged Phase 1 checkpoint:
  python train_vocos_decoder.py --tier 3 --epochs 500 --adversarial --adv-start-epoch 0 --resume
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
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'student'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'oracle'))

from lamquant_neural.models.vocos_decoder import VocosDecoder
from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
from lamquant.student.training_utils import pearson_r_loss
from auraloss.freq import MultiResolutionSTFTLoss
from streaming_dataset import PrecomputedL3Dataset
from flow_postfilter import CFMPostfilter
from perceptual_losses import MultiTeacherPerceptualLoss


def pearson_r_batch(pred, target):
    """Batch Pearson R for monitoring (returns float, not tensor)."""
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    pc = p - p.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    r = (pc * tc).sum(dim=-1) / (
        torch.sqrt((pc ** 2).sum(dim=-1)) * torch.sqrt((tc ** 2).sum(dim=-1)) + 1e-8)
    return r.mean().item()


def discover_student_checkpoint(explicit_path=None):
    """Auto-discover the best available student checkpoint."""
    if explicit_path is not None:
        if os.path.exists(explicit_path):
            return explicit_path
        print(f"[!] Specified checkpoint not found: {explicit_path}")
        sys.exit(1)

    candidates = [
        os.path.join(ROOT_DIR, 'ai_models/student/student_hardened.ckpt'),
        os.path.join(ROOT_DIR, 'ai_models/student/student_subband_hardened.ckpt'),
        os.path.join(ROOT_DIR, 'weights/student_subband.ckpt'),
        os.path.join(ROOT_DIR, 'weights/student_subband_fast.ckpt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    print("[!] No student checkpoint found. Searched:")
    for p in candidates:
        print(f"    {p}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train Vocos decoder')
    parser.add_argument('--tier', type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-min', type=float, default=1e-6)
    parser.add_argument('--windows-per-epoch', type=int, default=400000)
    parser.add_argument('--max-windows', type=int, default=500000,
                        help='Cap total L3 windows loaded into RAM (default 500K)')
    parser.add_argument('--student-checkpoint', type=str, default=None,
                        help='Path to student checkpoint (auto-discover if not set)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', action='store_true')
    # Phase 2: adversarial training
    parser.add_argument('--adversarial', action='store_true',
                        help='Enable Phase 2 adversarial training (MPD + MS-STFT)')
    parser.add_argument('--adv-start-epoch', type=int, default=200,
                        help='Epoch to start adversarial loss (warm-up reconstruction first)')
    parser.add_argument('--adv-ramp-epochs', type=int, default=50,
                        help='Epochs to ramp adversarial weight from 0.1 to 1.0')
    parser.add_argument('--disc-lr', type=float, default=None,
                        help='Discriminator LR (default: half of generator LR)')
    # P2 features
    parser.add_argument('--cfm-postfilter', action='store_true',
                        help='Train CFM postfilter after decoder converges (Phase 3)')
    parser.add_argument('--cfm-start-epoch', type=int, default=None,
                        help='Epoch to start CFM training (default: 80%% of total)')
    parser.add_argument('--perceptual-loss', action='store_true',
                        help='Use multi-teacher perceptual loss (LaBraM+DAC+FEMBA)')
    parser.add_argument('--perceptual-weight', type=float, default=0.1,
                        help='Weight on perceptual loss term')
    parser.add_argument('--dac-init', action='store_true',
                        help='Initialize decoder Conv1d layers from DAC encoder weights')
    # ---- LMA-direct training (BLUT canonical, ADR 0017) ----
    parser.add_argument('--lma-root', type=str, default=None,
                        help='Directory of per-recording .lma archives. When set '
                             'with --split-manifest, training reads LMA directly.')
    parser.add_argument('--split-manifest', type=str, default=None,
                        help='JSON split manifest. Required when --lma-root is set.')
    args = parser.parse_args()

    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Load L3 data — LMA-direct (BLUT canonical) when --lma-root set,
    # else fall through to the deprecated NPZ path.
    if args.lma_root is not None and args.split_manifest is not None:
        from lamquant_codec.training import LmaL3Dataset, load_split_stems
        train_stems, _ = load_split_stems(args.split_manifest, "train")
        print(f"[*] LMA-direct: root={args.lma_root}, train_stems={len(train_stems)}")
        dataset = LmaL3Dataset(
            lma_root=args.lma_root, file_stems=train_stems,
            windows_per_epoch=args.windows_per_epoch,
            max_windows=args.max_windows,
        )
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

    # Load frozen student encoder
    student_ckpt = discover_student_checkpoint(args.student_checkpoint)
    student = TernaryMobileNetV5_Subband.from_checkpoint(student_ckpt, device=device).eval()
    for p in student.parameters():
        p.requires_grad = False
    s_params = sum(p.numel() for p in student.parameters())

    # Create Vocos decoder
    decoder = VocosDecoder(tier=args.tier).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())

    # --- P2 features initialization ---
    # CFM postfilter (Phase 3: train after decoder converges)
    cfm = None
    if args.cfm_postfilter:
        cfm = CFMPostfilter(
            channels=decoder.n_channels,
            dim=min(decoder.dim, 256),  # cap at 256 for memory
        ).to(device)
        cfm_start = args.cfm_start_epoch or int(args.epochs * 0.8)
        print(f"[*] CFM postfilter enabled (Phase 3 from epoch {cfm_start})")

    # Multi-teacher perceptual loss
    perceptual_loss_fn = None
    if args.perceptual_loss:
        perceptual_loss_fn = MultiTeacherPerceptualLoss(device=str(device))
        print(f"[*] Multi-teacher perceptual loss: LaBraM + DAC + FEMBA")

    # DAC pretrained weight initialization
    if args.dac_init:
        try:
            import dac
            dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
            # Transfer Conv1d weights where dimensions match
            n_transferred = 0
            for (name_d, p_d), (name_dac, p_dac) in zip(
                    decoder.named_parameters(), dac_model.encoder.named_parameters()):
                if p_d.shape == p_dac.shape and 'conv' in name_d.lower():
                    p_d.data.copy_(p_dac.data)
                    n_transferred += 1
            print(f"[*] DAC weight init: transferred {n_transferred} layers")
            del dac_model
        except Exception as e:
            print(f"[!] DAC init failed: {e}. Training from scratch.")

    print(f"[*] Vocos Decoder Tier {args.tier} on {device}")
    print(f"    Decoder: {n_params:,} params")
    print(f"    Student: frozen, {s_params:,} params")
    print(f"    Input: latent [32, 79] -> output [21, 313]")
    print(f"    Dataset: {len(dataset)} windows/epoch, bs={args.batch_size}")
    print(f"    Training: {args.epochs} epochs, lr={args.lr:.0e}")
    print(f"    Student ckpt: {student_ckpt}")

    # Determine expected output length based on tier output mode
    output_mode = decoder.output_mode
    if output_mode == 'direct':
        expected_out_len = 313
    else:
        # iSTFT tiers reconstruct full waveform [21, 2500] — L3 target is
        # upsampled to match during loss computation
        expected_out_len = 2500

    # Verify shapes
    with torch.no_grad():
        sample_l3 = torch.randn(1, 21, 313, device=device)
        latent = student.encode(sample_l3, quantize=True)
        recon = decoder(latent)
    print(f"    Shapes: L3 {list(sample_l3.shape)} -> latent {list(latent.shape)} "
          f"-> recon {list(recon.shape)}")
    print(f"    Output mode: {output_mode}, expected length: {expected_out_len}")
    assert latent.shape == torch.Size([1, 32, 79]), f"Latent shape mismatch: {latent.shape}"
    assert recon.shape[1] == 21, f"Output channels mismatch: {recon.shape}"
    assert recon.shape[2] == expected_out_len, f"Output length mismatch: {recon.shape}"

    # Optimizer and scheduler (generator)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # Loss functions
    fft_sizes = [16, 32, 64, 128, 256]
    spectral_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes,
        hop_sizes=[max(n // 4, 1) for n in fft_sizes],
        win_lengths=fft_sizes,
    ).to(device)

    # Phase 2: adversarial setup (discriminator created but only used after adv_start_epoch)
    discriminator = None
    disc_optimizer = None
    if args.adversarial:
        from discriminator import EEGDiscriminator
        discriminator = EEGDiscriminator().to(device)
        disc_lr = args.disc_lr or (args.lr / 2.0)  # 2:1 gen:disc LR ratio
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(),
                                            lr=disc_lr, weight_decay=1e-4)
        d_params = sum(p.numel() for p in discriminator.parameters())
        print(f"    Discriminator: {d_params:,} params, lr={disc_lr:.0e}")
        print(f"    Adversarial: starts epoch {args.adv_start_epoch}, "
              f"ramps over {args.adv_ramp_epochs} epochs")

    # Resume
    ckpt_dir = os.path.join(ROOT_DIR, 'ai_models/decoder')
    best_path = os.path.join(ckpt_dir, f'vocos_tier{args.tier}_best.ckpt')
    start_epoch = 0
    best_r = 0.0

    if args.resume and os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            decoder.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            best_r = ckpt.get('best_r', 0.0)
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if discriminator is not None and 'disc_state_dict' in ckpt:
                discriminator.load_state_dict(ckpt['disc_state_dict'])
                if 'disc_optimizer_state_dict' in ckpt:
                    disc_optimizer.load_state_dict(ckpt['disc_optimizer_state_dict'])
            print(f"    Resumed from epoch {start_epoch}, best R={best_r:.4f}")

    # Training loop
    n_batches = len(loader)
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        ep_start = time.time()
        losses_l1, losses_spec, losses_r, losses_total = [], [], [], []
        rs = []

        # Determine adversarial state for this epoch
        adv_active = (discriminator is not None and epoch >= args.adv_start_epoch)
        if adv_active:
            ramp_progress = min(1.0, (epoch - args.adv_start_epoch) / max(args.adv_ramp_epochs, 1))
            adv_weight = 0.1 + 0.9 * ramp_progress  # 0.1 → 1.0
        else:
            adv_weight = 0.0

        for batch_idx, (x_l3, _, _) in enumerate(loader):
            x_l3 = x_l3.to(device, non_blocking=True)

            # Forward: frozen student encode -> Vocos decode
            with torch.no_grad():
                latent = student.encode(x_l3, quantize=True)  # [B, 32, 79]

            # For iSTFT tiers, upsample x_l3 to match decoder output length
            if output_mode != 'direct':
                target = F.interpolate(x_l3, size=expected_out_len,
                                       mode='linear', align_corners=False)
            else:
                target = x_l3

            # ---- Discriminator update (if adversarial active) ----
            if adv_active:
                discriminator.train()
                disc_optimizer.zero_grad()
                with torch.no_grad():
                    fake = decoder(latent).float().detach()
                real_scores, _ = discriminator(target.float())
                fake_scores, _ = discriminator(fake)
                d_loss = discriminator.discriminator_loss(real_scores, fake_scores)
                d_loss.backward()
                disc_optimizer.step()

            # ---- Generator (decoder) update ----
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, dtype=torch.bfloat16,
                                     enabled=(device.type == 'cuda')):
                recon = decoder(latent)

                # Reconstruction losses
                l1_loss = F.l1_loss(recon, target)
                spec_loss = spectral_loss_fn(recon, target)
                r_loss = pearson_r_loss(recon, target)
                loss = l1_loss + 1.0 * spec_loss + 0.5 * r_loss

            # Adversarial + feature matching (float32, outside autocast)
            if adv_active:
                recon_f32 = recon.float()
                target_f32 = target.float()
                real_scores, real_feats = discriminator(target_f32)
                fake_scores, fake_feats = discriminator(recon_f32)
                g_adv_loss, feat_loss = discriminator.generator_loss(
                    real_scores, fake_scores, real_feats, fake_feats)
                loss = loss.float() + adv_weight * g_adv_loss + feat_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()

            losses_l1.append(l1_loss.item())
            losses_spec.append(spec_loss.item())
            losses_r.append(r_loss.item())
            losses_total.append(loss.item())
            with torch.no_grad():
                r = pearson_r_batch(recon, target)
                rs.append(r)

        scheduler.step()
        ep_sec = time.time() - ep_start
        mean_l1 = np.mean(losses_l1)
        mean_spec = np.mean(losses_spec)
        mean_r_loss = np.mean(losses_r)
        mean_total = np.mean(losses_total)
        mean_r = np.mean(rs)

        improved = ''
        if mean_r > best_r:
            best_r = mean_r
            improved = ' *BEST*'
            save_dict = {
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_r': best_r,
                'tier': args.tier,
            }
            if discriminator is not None:
                save_dict['disc_state_dict'] = discriminator.state_dict()
                save_dict['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
            torch.save(save_dict, best_path)

        elapsed = time.time() - train_start
        remaining = elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch - 1)
        eta_h, eta_m = divmod(int(remaining), 3600)
        eta_m //= 60

        adv_tag = f"  adv={adv_weight:.2f}" if adv_active else ""
        print(f"E{epoch+1:3d}/{args.epochs}  "
              f"L={mean_total:.6f} (l1={mean_l1:.4f} spec={mean_spec:.4f} r={mean_r_loss:.4f})  "
              f"R={mean_r:.4f}  best={best_r:.4f}  {ep_sec:.0f}s  "
              f"LR={scheduler.get_last_lr()[0]:.2e}{adv_tag}  "
              f"ETA={eta_h}h{eta_m:02d}m{improved}")

    # Save final checkpoint
    final_path = os.path.join(ckpt_dir, f'vocos_tier{args.tier}_{args.epochs}_completed.ckpt')
    torch.save({
        'model_state_dict': decoder.state_dict(),
        'epoch': args.epochs,
        'best_r': best_r,
        'tier': args.tier,
    }, final_path)

    total_h = (time.time() - train_start) / 3600
    print(f"\n[*] Training complete in {total_h:.1f}h. Best R: {best_r:.4f}")
    print(f"[*] Saved: {best_path} (best), {final_path} (final)")


if __name__ == '__main__':
    main()
