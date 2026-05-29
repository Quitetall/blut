#!/usr/bin/env python3
"""Production Route B decoder training — Tiers 5/6/7 (100M/400M/837M).

All tiers: latent [32, 79] → fullband [21, 2500] via iSTFT.
No detail conditioning. No FiLM. Tokens only, 274:1 CR.

Usage:
  python run_decoder_tier.py --tier 5 --epochs 300 --batch-size 32   # 100M, any GPU
  python run_decoder_tier.py --tier 6 --epochs 200 --batch-size 16   # 400M, 3060+
  python run_decoder_tier.py --tier 7 --epochs 200 --batch-size 8    # 837M, 4090/cloud
"""
import os, sys, time, glob, argparse, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader

# MOVE-B (2026-05-29): now at blut/python/lamquant/decoder/. Resolve
# ROOT_DIR from __file__ (the blut/python package root) so the sibling
# area dirs are on sys.path for the bare imports below, regardless of
# cwd. vocos_decoder/encoder are PRIVATE lamquant_neural defs.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'decoder'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'student'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'oracle'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'dataset'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'common'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant'))

from lamquant_neural.models.vocos_decoder import VocosDecoder, anti_wrapping_phase_loss
from discriminator import EEGDiscriminator
from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband, TernaryMobileNetV5_Subband_V2
from raw_window_dataset import RawWindowDataset
from data_types import DatasetManifest, Split
from auraloss.freq import MultiResolutionSTFTLoss


def pearson_r_loss(pred, target):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    pc = p - p.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    r = (pc * tc).sum(dim=-1) / (
        torch.sqrt((pc ** 2).sum(dim=-1)) * torch.sqrt((tc ** 2).sum(dim=-1)) + 1e-8)
    return 1.0 - r.mean()


def pearson_r_batch(pred, target):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    pc = p - p.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    r = (pc * tc).sum(dim=-1) / (
        torch.sqrt((pc ** 2).sum(dim=-1)) * torch.sqrt((tc ** 2).sum(dim=-1)) + 1e-8)
    return r.mean().item()


def prd_batch(pred, target):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    return (100.0 * torch.sqrt(((p - t) ** 2).sum(dim=-1)) /
            torch.sqrt((t ** 2).sum(dim=-1)).clamp(min=1e-8)).mean().item()


def main():
    parser = argparse.ArgumentParser(description='Production Route B decoder training')
    parser.add_argument('--tier', type=int, required=True, choices=[5, 6, 7])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-windows', type=int, default=50000)
    parser.add_argument('--student-ckpt', type=str,
                        default='weights/backups/gen76_20260414/student_subband_runD_v4_best.ckpt')
    parser.add_argument('--init-from', type=str, default=None,
                        help='Warm-start decoder from checkpoint')
    parser.add_argument('--adversarial', action='store_true',
                        help='Enable progressive adversarial training (Phase B at 50%%, Phase C at 80%%)')
    parser.add_argument('--gradient-checkpoint', action='store_true')
    args = parser.parse_args()

    # Default batch sizes per tier
    if args.batch_size is None:
        args.batch_size = {5: 32, 6: 16, 7: 8}[args.tier]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Frozen student encoder — auto-detect V1 or V2 from checkpoint
    student = TernaryMobileNetV5_Subband.from_checkpoint(args.student_ckpt, device=str(device))
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    # Decoder
    use_gc = args.gradient_checkpoint or args.tier >= 6
    decoder = VocosDecoder(tier=args.tier, gradient_checkpointing=use_gc).to(device)

    if args.init_from and os.path.exists(args.init_from):
        try:
            ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        d_sd = ckpt.get('model_state_dict', ckpt)
        d_sd = {k.replace('_orig_mod.', ''): v for k, v in d_sd.items()}
        loaded = {k: v for k, v in d_sd.items()
                  if k in decoder.state_dict() and v.shape == decoder.state_dict()[k].shape}
        decoder.load_state_dict(loaded, strict=False)
        print(f"[*] Warm-start from {args.init_from} ({len(loaded)}/{len(decoder.state_dict())} params)")

    # P1: Progressive adversarial training
    discriminator = None
    disc_opt = None
    if args.adversarial:
        discriminator = EEGDiscriminator().to(device)
        disc_opt = torch.optim.AdamW(discriminator.parameters(),
                                      lr=args.lr * 0.5, weight_decay=1e-4)
        disc_n = sum(p.numel() for p in discriminator.parameters())
        print(f"[*] Adversarial training enabled: discriminator {disc_n:,} params")
        print(f"    Phase A (0-50%): reconstruction only")
        print(f"    Phase B (50-80%): + adversarial (ramp 0→1)")
        print(f"    Phase C (80-100%): full adversarial")

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"[*] Tier {args.tier} decoder: {n_params:,} params ({n_params/1e6:.0f}M)")
    print(f"[*] Gradient checkpointing: {use_gc}")
    print(f"[*] Batch size: {args.batch_size}")

    # Data — manifest_v3 (typed pipeline, single source of truth)
    manifest = DatasetManifest.load('ai_models/dataset_sim/manifest_v3.json')
    train_files = [str(p) for p in manifest.get_files(Split.TRAIN)]
    val_files = [str(p) for p in manifest.get_files(Split.VAL)]

    dataset = RawWindowDataset(train_files, windows_per_epoch=args.max_windows,
                                max_windows=args.max_windows)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    # Spectral loss with tier-appropriate FFT sizes
    tier_fft = {5: [32, 64, 128, 256], 6: [64, 128, 256, 512], 7: [128, 256, 512, 1024]}
    fft_sizes = tier_fft[args.tier]
    spec_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes,
        hop_sizes=[max(n // 4, 1) for n in fft_sizes],
        win_lengths=fft_sizes,
    ).to(device)

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    best_r = 0.0
    ckpt_path = f'ai_models/decoder/vocos_tier{args.tier}_best.ckpt'

    print(f"[*] Training: {args.epochs} epochs, {len(dataset)} windows/epoch")
    print(f"[*] FFT sizes: {fft_sizes}")

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        losses, rs = [], []
        t0 = time.time()

        for l3_batch, raw_batch in loader:
            l3_batch = l3_batch.to(device, non_blocking=True)
            raw_batch = raw_batch.to(device, non_blocking=True)

            with torch.no_grad():
                latent = student.encode(l3_batch, quantize=True)

            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                recon = decoder(latent)
                loss_recon = F.l1_loss(recon, raw_batch) + 0.5 * pearson_r_loss(recon, raw_batch)
                if epoch > 10:
                    loss_recon = loss_recon + 0.1 * anti_wrapping_phase_loss(
                        recon.float(), raw_batch.float(), n_fft=64, hop_length=8)
                if epoch % 4 == 0:
                    loss_recon = loss_recon + 0.03 * spec_loss_fn(recon.float(), raw_batch.float())

            # P1-6: Progressive adversarial training
            loss_adv = torch.tensor(0.0, device=device)
            phase_pct = epoch / args.epochs
            if discriminator is not None and phase_pct >= 0.5:
                disc_opt.zero_grad()
                with torch.no_grad():
                    recon_det = recon.detach().float()
                d_real, d_real_feats = discriminator(raw_batch.float())
                d_fake, d_fake_feats = discriminator(recon_det)
                loss_d = discriminator.discriminator_loss(d_real, d_fake)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
                disc_opt.step()
                # Generator update
                d_fake_g, d_fake_feats_g = discriminator(recon.float())
                loss_g, loss_fm = discriminator.generator_loss(
                    d_real, d_fake_g, d_real_feats, d_fake_feats_g)
                w_adv = min(1.0, (phase_pct - 0.5) / 0.3) if phase_pct < 0.8 else 1.0
                loss_adv = w_adv * (loss_g + loss_fm)

            loss = loss_recon + loss_adv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

            with torch.no_grad():
                rs.append(pearson_r_batch(recon, raw_batch))

        sched.step()
        ep_sec = time.time() - t0
        r = np.mean(rs)
        tag = ''

        if r > best_r:
            best_r = r
            torch.save({
                'model_state_dict': decoder.state_dict(),
                'epoch': epoch, 'best_r': best_r, 'tier': args.tier,
            }, ckpt_path)
            tag = ' *BEST*'

        if epoch % 10 == 0 or epoch <= 10:
            print(f"E{epoch:3d}/{args.epochs}  L={np.mean(losses):.4f} R={r:.4f} "
                  f"best={best_r:.4f} {ep_sec:.0f}s{tag}")

    print(f"\n[*] Tier {args.tier} done. Best R={best_r:.4f}")
    print(f"[*] Saved: {ckpt_path}")


if __name__ == '__main__':
    main()
