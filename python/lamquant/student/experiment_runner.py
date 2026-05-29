"""Controlled experiment runner for research ablations.

Each experiment modifies ONE variable against the current best baseline.
Uses the standardized holdout benchmark for apples-to-apples comparison.

Usage:
    python experiment_runner.py --experiment baseline
    python experiment_runner.py --experiment selfeeg_moderate
    python experiment_runner.py --experiment multiscale_balanced
    python experiment_runner.py --experiment all  # run all experiments sequentially

Experiment list:
    baseline           Current best config (SEQ + SubLN + unified QAT)
    selfeeg_light      selfEEG augmentation, light preset
    selfeeg_moderate   selfEEG augmentation, moderate preset
    selfeeg_aggressive selfEEG augmentation, aggressive preset
    builtin_aug        Built-in augmentation (no selfEEG dependency)
    multiscale_compact SNAC multi-scale FSQ, compact preset (122:1)
    multiscale_balanced SNAC multi-scale FSQ, balanced preset (82:1)
    multiscale_quality SNAC multi-scale FSQ, quality preset (63:1)
    progressive_tau    Progressive ternary: soft 80%, anneal 20%
    two_stage_wd       Two-stage weight decay: normal 2/3, remove 1/3
    combined_best      Best of each category combined

Results are written to outputs/experiments/<experiment_name>_<timestamp>.json
"""

import argparse
import json
import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "lamquant", "student"))
# MOVE-B: lamquant_codec is the pip-installed PUBLIC wheel; add common DTOs dir.
sys.path.insert(0, os.path.join(_REPO, "lamquant", "common"))

from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband


# ============================================================
# Experiment definitions
# ============================================================

EXPERIMENTS = {
    # Exp 1: Reference — V1 architecture, no augmentation, standard schedule
    'baseline': {
        'desc': 'V1 baseline (w=128, 3 focal, full conv, SEQ+SubLN, DitheredFSQ)',
        'augmentor': None,
        'multiscale_fsq': None,
        'tau_schedule': 'cosine',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,  # V1 as reference
    },
    # Exp 2-4: selfEEG augmentation at three intensities
    'selfeeg_light': {
        'desc': 'selfEEG augmentation, light (SNR=30dB, p=0.3)',
        'augmentor': ('selfeeg', 'light'),
        'multiscale_fsq': None,
        'tau_schedule': 'cosine',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,
    },
    'selfeeg_moderate': {
        'desc': 'selfEEG augmentation, moderate (SNR=20dB, p=0.5)',
        'augmentor': ('selfeeg', 'moderate'),
        'multiscale_fsq': None,
        'tau_schedule': 'cosine',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,
    },
    'selfeeg_aggressive': {
        'desc': 'selfEEG augmentation, aggressive (SNR=15dB, p=0.7)',
        'augmentor': ('selfeeg', 'aggressive'),
        'multiscale_fsq': None,
        'tau_schedule': 'cosine',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,
    },
    # Exp 5: Built-in augmentation (no selfEEG dependency)
    'builtin_aug': {
        'desc': 'Built-in augmentation (noise+dropout+mask, p=0.5)',
        'augmentor': ('builtin', 'moderate'),
        'multiscale_fsq': None,
        'tau_schedule': 'cosine',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,
    },
    # Exp 6-8: SNAC multi-scale FSQ at three presets
    'snac_balanced': {
        'desc': 'SNAC multi-scale FSQ, balanced (strides=[8,4,2,1], L=[3,3,5,5], 82:1)',
        'augmentor': None,
        'multiscale_fsq': 'balanced',
        'tau_schedule': 'cosine',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,
    },
    'snac_compact': {
        'desc': 'SNAC multi-scale FSQ, compact (strides=[8,4,2,1], L=[2,2,3,3], 122:1)',
        'augmentor': None,
        'multiscale_fsq': 'compact',
        'tau_schedule': 'cosine',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,
    },
    'snac_flat': {
        'desc': 'SNAC flat FSQ (stride=[1], L=[5], 143:1 — current behavior)',
        'augmentor': None,
        'multiscale_fsq': 'flat',
        'tau_schedule': 'cosine',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,
    },
    # Exp 9-10: Training schedule variants
    'progressive_tau': {
        'desc': 'Progressive ternary: tau=0.1 soft 80%%, anneal 20%%',
        'augmentor': None,
        'multiscale_fsq': None,
        'tau_schedule': 'progressive',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,
    },
    'two_stage_wd': {
        'desc': 'Two-stage WD: normal 2/3, zero final 1/3',
        'augmentor': None,
        'multiscale_fsq': None,
        'tau_schedule': 'cosine',
        'wd_schedule': 'two_stage',
        'epochs': 150,
        'use_v1': True,
    },

    # === P1 experiments ===

    'q2d2_l5': {
        'desc': 'Q2D2 pairwise channel quantization (L=5, 25 joint codes per pair)',
        'augmentor': None,
        'multiscale_fsq': None,
        'tau_schedule': 'cosine',
        'wd_schedule': 'constant',
        'epochs': 150,
        'use_v1': True,
        'q2d2': True,
    },
    'snac_balanced_long': {
        'desc': 'SNAC balanced at 400 epochs (tests latent reorganization hypothesis)',
        'augmentor': None,
        'multiscale_fsq': 'balanced',
        'tau_schedule': 'cosine',
        'wd_schedule': 'two_stage',
        'epochs': 400,
        'use_v1': True,
    },
    'snac_compact_long': {
        'desc': 'SNAC compact at 400 epochs (baseline for balanced comparison)',
        'augmentor': None,
        'multiscale_fsq': 'compact',
        'tau_schedule': 'cosine',
        'wd_schedule': 'two_stage',
        'epochs': 400,
        'use_v1': True,
    },
    'combined_winners': {
        'desc': 'Combined: two-stage WD + SNAC compact (stack both winners)',
        'augmentor': None,
        'multiscale_fsq': 'compact',
        'tau_schedule': 'cosine',
        'wd_schedule': 'two_stage',
        'epochs': 150,
        'use_v1': True,
    },
}


def _load_data(max_windows=5000):
    """Load L3 training data from Q31 .npz files.

    Each .npz has l3 shape [N_windows, 21, 313]. We extract individual
    windows up to max_windows total.
    """
    eeg_dir = os.path.join(_REPO, "ai_models/dataset_sim/q31_events")
    files = sorted(glob.glob(os.path.join(eeg_dir, "*.npz")))
    windows = []
    for f in files:
        if len(windows) >= max_windows:
            break
        try:
            d = np.load(f)
            l3 = d['l3']  # [N, 21, 313]
            if l3.ndim == 3 and l3.shape[1] == 21 and l3.shape[2] == 313:
                for i in range(min(l3.shape[0], max_windows - len(windows))):
                    windows.append(l3[i].astype(np.float32))
            elif l3.ndim == 2 and l3.shape == (21, 313):
                windows.append(l3.astype(np.float32))
        except Exception:
            continue
    print(f"[*] Loaded {len(windows)} L3 windows from {len(files)} files")
    return windows


def _make_augmentor(spec):
    """Create augmentor from experiment spec."""
    if spec is None:
        return None
    aug_type, mode = spec
    if aug_type == 'selfeeg':
        from augmentations import EEGAugmentor
        return EEGAugmentor(mode=mode)
    elif aug_type == 'builtin':
        from augmentations import BuiltinAugmentor
        return BuiltinAugmentor()
    return None


def _make_multiscale_fsq(preset):
    """Create multi-scale FSQ from preset name."""
    if preset is None:
        return None
    from multiscale_fsq import make_multiscale_fsq
    return make_multiscale_fsq(preset)


def _get_tau(epoch, total_epochs, schedule):
    """Compute ternary temperature tau."""
    if schedule == 'cosine':
        return 0.1 * (1 + np.cos(np.pi * epoch / total_epochs)) / 2
    elif schedule == 'progressive':
        # Soft for 80%, anneal in final 20%
        boundary = int(total_epochs * 0.8)
        if epoch < boundary:
            return 0.1
        else:
            progress = (epoch - boundary) / (total_epochs - boundary)
            return 0.1 * (1 - progress)
    return 0.1


def run_experiment(name: str, device='cuda'):
    """Run a single experiment and return results."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(EXPERIMENTS)}")

    cfg = EXPERIMENTS[name]
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"  {cfg['desc']}")
    print(f"{'='*60}\n")

    # Load data
    windows = _load_data(max_windows=5000)
    if len(windows) < 100:
        print("[!] Not enough data. Need at least 100 windows.")
        return None

    # Split 90/10
    n_val = max(50, len(windows) // 10)
    val_data = windows[-n_val:]
    train_data = windows[:-n_val]

    # Create model (fresh for each experiment)
    if cfg.get('use_v1', False):
        from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
        model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
    else:
        # V1 baseline (w=128, 3 focal, full conv) for all experiments.
        # Compare relative deltas — winners get rolled into V2 production config.
        model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
    model.ensure_initialized()

    # Training guard for automated problem detection
    from training_guard import TrainingGuard
    guard = TrainingGuard(model, config='fast')

    # Setup
    augmentor = _make_augmentor(cfg['augmentor'])
    msfsq = _make_multiscale_fsq(cfg['multiscale_fsq'])
    if msfsq is not None:
        msfsq = msfsq.to(device)

    # Cosine annealing LR — stable for both 150ep and 400ep runs.
    # OneCycleLR collapsed at 400ep (R dropped from 0.84 to 0.64).
    epochs = cfg['epochs']
    batch_size = 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    # Training loop
    history = {'train_loss': [], 'val_r': [], 'val_prd': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        # Shuffle
        np.random.shuffle(train_data)

        tau = _get_tau(epoch, epochs, cfg['tau_schedule'])

        # Weight decay schedule
        if cfg['wd_schedule'] == 'two_stage' and epoch > epochs * 2 // 3:
            for pg in optimizer.param_groups:
                pg['weight_decay'] = 0.0

        for i in range(0, len(train_data) - batch_size, batch_size):
            batch = torch.stack([torch.from_numpy(w) for w in train_data[i:i+batch_size]])
            batch = batch.to(device)

            # Apply augmentation
            if augmentor is not None:
                batch = augmentor(batch)

            # Forward
            recon = model(batch, quantize=True)
            loss = F.mse_loss(recon, batch)

            # Multi-scale FSQ loss (added to main loss)
            if msfsq is not None:
                with torch.no_grad():
                    latent = model.encode(batch, quantize=True)
                _, _, qloss = msfsq(latent)
                loss = loss + 0.1 * qloss

            # Q2D2 pairwise quantization loss
            if cfg.get('q2d2', False):
                with torch.no_grad():
                    latent = model.encode(batch, quantize=True)
                # Pair adjacent channels, quantize on L×L grid, measure reconstruction error
                L = 5
                lat_np = latent[0].detach()
                vmin, vmax = lat_np.min(), lat_np.max()
                span = vmax - vmin + 1e-8
                norm = (lat_np - vmin) / span
                # Pair [0,1], [2,3], ..., [30,31] → 16 pairs
                n_pairs = lat_np.shape[0] // 2
                q2d2_recon = torch.zeros_like(lat_np)
                for p in range(n_pairs):
                    a = torch.clamp((norm[2*p] * L).long(), 0, L-1)
                    b = torch.clamp((norm[2*p+1] * L).long(), 0, L-1)
                    q2d2_recon[2*p] = vmin + (a.float() + 0.5) / L * span
                    q2d2_recon[2*p+1] = vmin + (b.float() + 0.5) / L * span
                q2d2_loss = F.mse_loss(q2d2_recon, lat_np)
                loss = loss + 0.05 * q2d2_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()  # cosine annealing: per-epoch, not per-step
        avg_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            val_rs = []
            val_prds = []
            with torch.no_grad():
                for w in val_data[:100]:
                    x = torch.from_numpy(w).unsqueeze(0).to(device)
                    recon = model(x, quantize=True)
                    o = w.flatten().astype(np.float64)
                    r = recon[0].cpu().numpy().flatten().astype(np.float64)
                    # Pearson R
                    corr = np.corrcoef(o, r)[0, 1] if o.std() > 1e-8 else 0
                    val_rs.append(float(corr))
                    # PRD
                    norm = np.sqrt(np.sum(o**2))
                    prd = 100 * np.sqrt(np.sum((o - r)**2)) / max(norm, 1e-8)
                    val_prds.append(float(prd))

            mean_r = np.mean(val_rs)
            mean_prd = np.mean(val_prds)
            history['val_r'].append(mean_r)
            history['val_prd'].append(mean_prd)

            # Training guard
            warnings = guard.check(epoch, val_r=mean_r, train_loss=avg_loss)
            warn_str = f"  [{len(warnings)} warnings]" if warnings else ""
            print(f"  ep {epoch+1:3d}/{epochs}  loss={avg_loss:.6f}  R={mean_r:.4f}  PRD={mean_prd:.1f}%  tau={tau:.4f}{warn_str}")
            for w in warnings:
                print(f"    [GUARD] {w}")

    # Secondary metrics on final model
    model.eval()
    per_ch_rs = []
    rans_sizes = []
    fsq_codes_used = set()
    with torch.no_grad():
        for w in val_data[:50]:
            x = torch.from_numpy(w).unsqueeze(0).to(device)
            recon = model(x, quantize=True)
            latent = model.encode(x, quantize=True)

            # Per-channel R
            for c in range(min(21, w.shape[0])):
                o = w[c].astype(np.float64)
                r = recon[0, c].cpu().numpy().astype(np.float64)
                if o.std() > 1e-8:
                    per_ch_rs.append(float(np.corrcoef(o, r)[0, 1]))

            # rANS compressed size (use codec compress)
            try:
                # MOVE-B: lamquant_codec is the pip-installed PUBLIC wheel; add common DTOs dir.
                sys.path.insert(0, os.path.join(_REPO, "lamquant", "common"))
                from codec import _rans_encode_symbols
                lat_np = latent[0].cpu().numpy()
                # FSQ quantize to L=5 for consistent comparison
                L = 5
                norm = (lat_np - lat_np.min()) / (lat_np.max() - lat_np.min() + 1e-8)
                syms = np.clip((norm * L).astype(np.int64), 0, L - 1).flatten()
                rb, _ = _rans_encode_symbols(syms)
                rans_sizes.append(len(rb))
                fsq_codes_used.update(syms.tolist())
            except Exception:
                pass

    # Loss curve slope at epoch 50 (steeper = faster convergence)
    loss_slope_50 = 0.0
    if len(history['train_loss']) >= 50:
        # Linear regression slope on epochs 40-50
        y = np.array(history['train_loss'][40:50])
        x_range = np.arange(10)
        if y.std() > 1e-10:
            loss_slope_50 = float(np.polyfit(x_range, y, 1)[0])

    per_ch_r_var = float(np.var(per_ch_rs)) if per_ch_rs else 0
    mean_rans = float(np.mean(rans_sizes)) if rans_sizes else 0
    fsq_util = len(fsq_codes_used) / 5 * 100 if fsq_codes_used else 0  # % of L=5 codes used

    print(guard.summary())
    print(f"  Secondary: rANS={mean_rans:.0f}B, FSQ_util={fsq_util:.0f}%, ch_R_var={per_ch_r_var:.6f}, loss_slope@50={loss_slope_50:.6f}")

    # Final results
    result = {
        'experiment': name,
        'description': cfg['desc'],
        'epochs': epochs,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
        'final_val_r': history['val_r'][-1] if history['val_r'] else 0,
        'final_val_prd': history['val_prd'][-1] if history['val_prd'] else 0,
        'best_val_r': max(history['val_r']) if history['val_r'] else 0,
        # Secondary metrics
        'rans_compressed_bytes': mean_rans,
        'fsq_utilization_pct': fsq_util,
        'per_channel_r_variance': per_ch_r_var,
        'loss_slope_at_50': loss_slope_50,
        'guard_warnings': guard.total_warnings,
        'best_val_prd': min(history['val_prd']) if history['val_prd'] else float('inf'),
        'history': history,
        'config': {k: str(v) for k, v in cfg.items()},
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
    }

    if msfsq is not None:
        result['multiscale_info'] = msfsq.token_count()
        result['estimated_cr'] = msfsq.estimated_cr()

    return result


def main():
    parser = argparse.ArgumentParser(description='LamQuant experiment runner')
    parser.add_argument('--experiment', '-e', default='baseline',
                        help='Experiment name or "all" for all experiments')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--list', action='store_true', help='List available experiments')
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name:25s}  {cfg['desc']}")
        return

    os.makedirs(os.path.join(_REPO, 'outputs', 'experiments'), exist_ok=True)

    if args.experiment == 'all':
        experiments = list(EXPERIMENTS.keys())
    else:
        experiments = [args.experiment]

    all_results = []
    for exp_name in experiments:
        result = run_experiment(exp_name, device=args.device)
        if result is not None:
            all_results.append(result)
            # Save individual result
            out_path = os.path.join(_REPO, 'outputs', 'experiments',
                                     f'{exp_name}_{result["timestamp"]}.json')
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Saved: {out_path}")

    # Summary table
    if all_results:
        baseline_r = next((r['best_val_r'] for r in all_results if r['experiment'] == 'baseline'), 0)
        print(f"\n{'='*100}")
        print(f"{'Exp':25s} {'R':>7s} {'delta':>7s} {'PRD':>7s} {'rANS':>6s} {'FSQ%':>5s} {'chR_var':>8s} {'slope50':>8s} {'warns':>5s}")
        print(f"{'-'*100}")
        for r in sorted(all_results, key=lambda x: -x['best_val_r']):
            delta = r['best_val_r'] - baseline_r
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            print(f"{r['experiment']:25s} {r['best_val_r']:7.4f} {delta_str:>7s} {r['best_val_prd']:6.1f}% "
                  f"{r.get('rans_compressed_bytes',0):5.0f}B {r.get('fsq_utilization_pct',0):4.0f}% "
                  f"{r.get('per_channel_r_variance',0):8.6f} {r.get('loss_slope_at_50',0):8.6f} "
                  f"{r.get('guard_warnings',0):5d}")
        print(f"{'='*100}")
        print(f"Baseline R: {baseline_r:.4f}")

        # Save combined results
        combined_path = os.path.join(_REPO, 'outputs', 'experiments',
                                      f'summary_{time.strftime("%Y%m%d_%H%M%S")}.json')
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Combined results: {combined_path}")


if __name__ == '__main__':
    main()
