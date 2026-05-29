#!/usr/bin/env python3
"""
LamQuant Gen 7.1 "Subband" — Student Training
==============================================
Trains the 112-wide subband TNN autoencoder on L3 approximation input.

Key difference from Gen 7.0 (train_student.py):
  - Input: L3 approximation [B, 21, 313] (from LPC + 3-level lifting)
  - Reconstruction target: original HP-filtered signal [B, 21, 2500]
  - Loss: MSE(inverse_lifting(LPC_synthesis(decode(encode(L3)))), original)
  - The full inverse chain ensures the encoder learns representations
    that reconstruct well through the complete signal path

Schedule (same 3-phase as Gen 7.0):
  Phase 1: Warm-up (50 ep, lr=2e-3, no quantize)
  Phase 2: Quantize-aware (200 ep, lr=1e-3, quantize=True)
  Phase 3: Fine-tune (250 ep, lr=2e-4, quantize=True, + spectral loss)

Total: 500 epochs
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import glob
import argparse


def _safe_load(path, map_location='cpu'):
    """Load checkpoint preferring weights_only=True for security."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError):
        warnings.warn(
            f"weights_only=True failed for {path}; using weights_only=False",
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(ROOT_DIR, 'lamquant', 'oracle'))
sys.path.append(os.path.join(ROOT_DIR, 'lamquant', 'student'))

from train_teacher import Q31Dataset
from lamquant_neural.models.encoder import (
    TernaryMobileNetV5_Subband,
    TernaryMobileNetV5_Subband_V2,
)
from ternary_encoder import (
    apply_montage_permutation,
    clinical_augmentation,
)
from subband_preprocess import (
    hp_filter,
    preprocess_subband_torch,
    reconstruct_subband_torch,
)
from subband_dataset import SubbandDataset


class SpectralLoss(nn.Module):
    """Multi-resolution STFT loss tuned for L3 subband (313 samples).

    Window sizes {16, 32} catch fast spike components (2-8 samples in L3
    domain = 20-70ms at the original 250 Hz). {64, 128, 256} catch slow
    waves. No 512 — at 313 input samples it's mostly zero-pad interpolation.
    """
    def __init__(self, fft_sizes=[16, 32, 64, 128, 256]):
        super().__init__()
        self.fft_sizes = fft_sizes

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


def pearson_r_batch(pred, target):
    """Batch Pearson R for monitoring (returns float, not tensor)."""
    p = pred.flatten(1)
    t = target.flatten(1)
    pc = p - p.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    r = torch.sum(pc * tc, dim=-1) / (
        torch.sqrt(torch.sum(pc ** 2, dim=-1)) *
        torch.sqrt(torch.sum(tc ** 2, dim=-1)) + 1e-8
    )
    return r.mean().item()


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


def split_by_manifest(npz_files, manifest_path=None):
    """Split NPZ files into train/val using the validation manifest + official_split_config.json.

    Reads the canonical split from manifest_v3.json (single source of truth).
    Files from holdout subjects go entirely into the validation set.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_types import DatasetManifest, Split

    manifest = DatasetManifest.load(
        Path(__file__).parent.parent / 'dataset_sim' / 'manifest_v3.json')
    train_files = [str(p) for p in manifest.get_files(Split.TRAIN)]
    val_files = [str(p) for p in manifest.get_files(Split.VAL)]
    return train_files, val_files


def validate_epoch(model, val_loader, device, quantize=True):
    """Run validation on holdout data. Returns (mean R, mean PRD).

    PRD is computed alongside R so the dashboard / CheckpointManager
    Option B tiebreak can use both. Per-band PRD is left to the
    end-of-run fullband evaluator (eval_fullband.py).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from metrics import prd_torch as _prd_torch

    model.eval()
    r_scores = []
    prd_scores = []
    with torch.no_grad():
        for x_l3, x_eeg, mask in val_loader:
            x_l3 = x_l3.to(device)
            recon_l3 = model(x_l3, quantize=quantize)
            r_scores.append(pearson_r_batch(recon_l3, x_l3))
            prd_scores.append(float(_prd_torch(x_l3, recon_l3)))
    if not r_scores:
        return 0.0, 100.0
    return float(np.mean(r_scores)), float(np.mean(prd_scores))


def latent_kurtosis(model, val_loader, device, max_batches=10):
    """Compute per-channel kurtosis of post-tanh latent distribution.

    Target trajectory: 5.2 (peaky) → 0 (Gaussian) → -1.2 (uniform/optimal).
    Returns (mean_kurtosis, per_channel_kurtosis_array).
    """
    model.eval()
    latents = []
    with torch.no_grad():
        for i, (x_l3, x_eeg, mask) in enumerate(val_loader):
            if i >= max_batches:
                break
            x_l3 = x_l3.to(device)
            lat = model.encode(x_l3, quantize=True)  # post-tanh, post-FSQ
            latents.append(lat.cpu())
    if not latents:
        return 0.0, np.zeros(32)
    all_lat = torch.cat(latents, dim=0).numpy()  # [N, 32, 79]
    per_ch_kurt = np.zeros(all_lat.shape[1])
    for c in range(all_lat.shape[1]):
        ch = all_lat[:, c, :].flatten()
        mu, sigma = ch.mean(), ch.std()
        if sigma > 1e-8:
            per_ch_kurt[c] = float(np.mean(((ch - mu) / sigma) ** 4)) - 3.0
        else:
            per_ch_kurt[c] = 999.0  # collapsed channel
    return float(per_ch_kurt.mean()), per_ch_kurt


def run(cfg=None, epochs_warmup=50, epochs_quant=200, epochs_fine=300, batch_size=32):
    """Train the Gen 7.1 subband student model.

    Args:
        cfg: TrainingConfig dataclass. If provided, overrides all other args.
             If None, uses the individual epoch/batch args (legacy interface).
    """
    from training_config import TrainingConfig, CONFIGS
    if cfg is None:
        cfg = TrainingConfig(
            epochs_warmup=epochs_warmup, epochs_quant=epochs_quant,
            epochs_fine=epochs_fine, batch_size=cfg.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Gen 7.1 Subband Student Training on {device}")
    print(cfg)

    # Pure-upside perf: TF32 + cudnn benchmark (teacher already has these)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # AMP dtype: bfloat16 on Ampere+, float16 otherwise
    amp_dtype = torch.float32  # default: no AMP on CPU
    if device.type == 'cuda':
        amp_dtype = (torch.bfloat16
                     if torch.cuda.get_device_capability(device)[0] >= 8
                     else torch.float16)

    # Model — V1 (w=128, 3 focal, full conv) is production.
    # V2 DW-sep tested and killed: -0.026 R. Ternary pointwise too coarse.
    _cdf_entries = getattr(args, 'cdf_entries', 32) or 32
    student = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32, cdf_entries=_cdf_entries).to(device)
    print(f"[*] V1 architecture (w=128, 3 focal blocks, 226K encoder params)")
    # Training guard: automated alarm system catches problems mid-training
    from training_guard import TrainingGuard
    guard = TrainingGuard(student, config='v2' if not getattr(args, 'v1', False) else 'v1')

    def _run_guards(epoch, val_r=None, train_loss=None):
        """Run training guards and print any warnings."""
        warnings = guard.check(epoch, val_r=val_r, train_loss=train_loss)
        for w in warnings:
            print(f"  [GUARD] {w}")

    # SNAC multi-scale FSQ (experiment winner: +0.0009 R with compact preset)
    _snac_fsq = None
    if cfg.snac_preset and cfg.snac_preset != 'none':
        from multiscale_fsq import make_multiscale_fsq
        _snac_fsq = make_multiscale_fsq(cfg.snac_preset).to(device)
        print(f"[*] SNAC multi-scale FSQ: preset={cfg.snac_preset}, CR={_snac_fsq.estimated_cr():.0f}:1")

    # Preset tag for checkpoint filenames
    PRESET_TAGS = {'fast': 'fast', 'standard': 'std', 'production': 'gold', 'custom': 'custom'}
    preset_tag = PRESET_TAGS.get(cfg.name, 'custom')
    s_path = os.path.join(ROOT_DIR, f"ai_models/student/student_subband_{preset_tag}.ckpt")

    if args.init_from:
        # Warm start: load weights but train from epoch 0 (fresh optimizer/scheduler)
        # strict=False allows architectural changes (e.g. ternary→INT8 bneck_v,
        # tanh→erf reshaping). New buffers/params use their default init.
        init_ckpt = args.init_from
        if not os.path.isabs(init_ckpt):
            init_ckpt = os.path.join(ROOT_DIR, init_ckpt)
        _ckpt_sd = _safe_load(init_ckpt, map_location=device)
        # Filter out shape-mismatched keys (e.g. cdf_breakpoints 32→64 upgrade)
        _model_sd = student.state_dict()
        _filtered = {k: v for k, v in _ckpt_sd.items()
                     if k in _model_sd and v.shape == _model_sd[k].shape}
        _shape_skipped = [k for k, v in _ckpt_sd.items()
                          if k in _model_sd and v.shape != _model_sd[k].shape]
        missing, unexpected = student.load_state_dict(_filtered, strict=False)
        if _shape_skipped:
            missing = list(set(missing) | set(_shape_skipped))
            print(f"    Shape mismatch (default init): {_shape_skipped}")
        print(f"[*] Warm start from: {init_ckpt}")
        if missing:
            print(f"    New params (default init): {missing}")
        if unexpected:
            print(f"    Skipped old params: {unexpected}")
        print(f"    Training from epoch 0 with fresh optimizer state.")

        # Reinitialize structurally damaged layers:
        # 1. Manual: --reinit-layers names layers with known damage
        # 2. Auto: any ternary layer with >75% zeros is functionally dead
        _manual_reinit = set()
        if args.reinit_layers:
            _manual_reinit = {s.strip() for s in args.reinit_layers.split(',')}
        _reinit_threshold = 0.75
        for name, m in student.named_modules():
            should_reinit = name in _manual_reinit
            if not should_reinit and hasattr(m, 'lsq_alpha') and hasattr(m, 'weight') and m.weight.dim() >= 2:
                with torch.no_grad():
                    alpha = m.lsq_alpha.abs().clamp(min=1e-8)
                    w_ternary = torch.clamp(torch.round(m.weight / alpha), -1, 1)
                    dead_frac = (w_ternary == 0).float().mean().item()
                should_reinit = dead_frac > _reinit_threshold
            if should_reinit and hasattr(m, 'weight') and m.weight.dim() >= 2:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                    if hasattr(m, 'lsq_alpha'):
                        with torch.no_grad():
                            mean_abs = m.weight.abs().mean(dim=(1, 2), keepdim=True)
                            m.lsq_alpha.data.copy_((2.0 / 3.0) * mean_abs.clamp(min=0.001))
                    print(f"    REINIT: {name} — {dead_frac:.0%} dead, fresh Kaiming + alpha reset")

    elif os.path.exists(s_path):
        try:
            student.load_state_dict(torch.load(s_path, map_location=device, weights_only=True), strict=True)
            print(f"[*] Resumed from existing checkpoint.")
        except Exception:  # model load or import failure — train from scratch
            print(f"[*] Architecture mismatch — training from scratch.")

    total_params = sum(p.numel() for p in student.parameters())
    encoder_params = sum(p.numel() for n, p in student.named_parameters()
                         if not n.startswith('expand') and not n.startswith('output'))
    print(f"[*] Model: {student.focal2.conv.weight.shape[0]}-wide subband, {total_params:,} total params, "
          f"{encoder_params:,} encoder params")

    # --- Empirical CDF quantile table from pre-reshape latent statistics ---
    # 32 breakpoints per channel, computed from training set latents.
    # Maps any distribution → uniform [-1,1]. Handles kurtosis 0 to 700+.
    # Frozen forever — encoder adapts to fixed target. 2 KB firmware cost.
    if hasattr(student, 'cdf_breakpoints') and args.init_from:
        print(f"[*] Computing empirical CDF quantile table from pre-reshape latent distribution...")
        # Temporarily set breakpoints to wide linear ramp so encode ≈ identity
        N_CDF = student.cdf_breakpoints.shape[1]
        student.cdf_breakpoints.copy_(
            torch.linspace(-100, 100, N_CDF).unsqueeze(0).expand_as(student.cdf_breakpoints))
        _sample_files = glob.glob(os.path.join(
            ROOT_DIR, "ai_models/dataset_sim/q31_events/*.npz"))[:16]
        if _sample_files:
            _latents = []
            for _sf in _sample_files:
                _sd = np.load(_sf)
                if 'l3' in _sd.files:
                    _l3_sample = torch.from_numpy(_sd['l3'][:64]).float().to(device)
                    with torch.no_grad():
                        _lat = student.encode(_l3_sample, quantize=False)
                    _latents.append(_lat.cpu())
                    del _l3_sample
                del _sd
            if _latents:
                _all_lat = torch.cat(_latents, dim=0)  # [N, 32, 79]
                C = _all_lat.shape[1]
                quantile_fracs = torch.linspace(0, 1, N_CDF)
                for c in range(C):
                    ch_vals = _all_lat[:, c, :].flatten().sort().values
                    # Compute quantiles at evenly spaced fractions
                    indices = (quantile_fracs * (len(ch_vals) - 1)).long()
                    student.cdf_breakpoints[c] = ch_vals[indices]
                bp = student.cdf_breakpoints
                print(f"    CDF table: {C} channels × {N_CDF} breakpoints = "
                      f"{C * N_CDF * 2} bytes (INT16)")
                print(f"    Breakpoint ranges: min=[{bp[:, 0].min():.3f}, {bp[:, 0].max():.3f}], "
                      f"max=[{bp[:, -1].min():.3f}, {bp[:, -1].max():.3f}]")
                del _all_lat, _latents

    # Verify shapes
    with torch.no_grad():
        test = torch.randn(1, 21, 313).to(device)
        lat = student.encode(test, quantize=False)
        out = student(test, quantize=False)
        print(f"[*] Input {list(test.shape)} -> Latent {list(lat.shape)} -> Output {list(out.shape)}")
        assert out.shape == test.shape, f"Shape mismatch: {out.shape} != {test.shape}"

    # Dataset — manifest-based train/val split
    npz_dir = os.path.join(ROOT_DIR, "ai_models/dataset_sim/q31_events")
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz"))) if os.path.isdir(npz_dir) else []

    manifest_path = os.path.join(ROOT_DIR, "ai_models/dataset_sim/validation_manifest/validation_manifest.json")
    if os.path.exists(manifest_path):
        train_files, val_files = split_by_manifest(npz_files, manifest_path)
        print(f"[*] Manifest split: {len(train_files)} train, {len(val_files)} val (holdout)")
    else:
        # Fallback: random 90/10 split
        n_val = max(1, len(npz_files) // 10)
        val_files = npz_files[:n_val]
        train_files = npz_files[n_val:]
        print(f"[*] Random split (no manifest): {len(train_files)} train, {len(val_files)} val")

    # Auto-select dataset strategy. If ALL training files have precomputed
    # L3 (from precompute_l3_fast.py), we load them directly into a single
    # contiguous RAM tensor — no disk I/O, no LPC+lifting at training time.
    # This is ~3000x faster than the SubbandDataset path that recomputes the
    # 94 ms per-window LPC+lifting on every access, and produces bit-identical
    # results (verified on a 250-file sample with zero mismatches).
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'oracle'))
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'common'))  # MOVE-B: common DTOs
    from streaming_dataset import (
        HybridQ31Dataset, StreamingQ31Dataset, PrecomputedL3Dataset,
        peek_npz_data_shape,
    )

    # Check if L3 is precomputed in the training files
    has_l3 = True
    for f in train_files[:20]:  # spot-check first 20
        try:
            with np.load(f) as d:
                if 'l3' not in d.files:
                    has_l3 = False
                    break
        except Exception:  # model load or import failure — train from scratch
            has_l3 = False
            break

    # AdaBatch: DataLoader factory for per-phase batch size changes.
    # Recreating a DataLoader is cheap (no data copy — same underlying dataset).
    _loader_is_precomputed = False

    if has_l3:
        print(f"[*] Precomputed L3 detected — loading directly into RAM (fast path)")
        train_dataset = PrecomputedL3Dataset(
            train_files, windows_per_epoch=cfg.windows_per_epoch,
            max_windows=cfg.max_windows)
        _loader_is_precomputed = True

        # Move to GPU if it fits (fast/standard configs with max_windows cap)
        _dataset_vram_gb = train_dataset.l3_data.nelement() * 4 / 1e9
        _vram_budget_gb = 20.0  # leave ~4 GB for model + activations + compile overhead
        if device.type == 'cuda' and _dataset_vram_gb < _vram_budget_gb:
            train_dataset.to_gpu(device)
        elif device.type == 'cuda':
            print(f"[*] Dataset {_dataset_vram_gb:.1f} GB > {_vram_budget_gb:.0f} GB VRAM budget — using prefetch")

        class _PrefetchLoader:
            """Reusable wrapper around prefetch_batches — fresh generator per epoch."""
            def __init__(self, dataset, bs, dev):
                self._dataset = dataset
                self._bs = bs
                self._dev = dev
            def __len__(self):
                return self._dataset.windows_per_epoch // self._bs
            def __iter__(self):
                return self._dataset.prefetch_batches(self._bs, self._dev)

        def _make_loader(bs):
            return _PrefetchLoader(train_dataset, bs, device)

        loader = _make_loader(cfg.batch_size_warmup)
        print(f"[*] Workers: 0 (in-memory tensor indexing, zero I/O)")
    else:
        print(f"[*] No precomputed L3 — falling back to on-the-fly LPC+lifting")
        n_workers = min(8, os.cpu_count() or 4)
        if len(train_files) > 500:
            print(f"[*] Large dataset ({len(train_files)} files) — hybrid RAM+streaming")
            base_train = HybridQ31Dataset(train_files, reserve_gb=20.0, windows_per_epoch=cfg.windows_per_epoch)
        else:
            print(f"[*] Small dataset ({len(train_files)} files) — RAM cache")
            cache_path = os.path.join(ROOT_DIR, "ai_models/dataset_sim/q31_cache_v1.pt")
            base_train = Q31Dataset(train_files, headless=True, cache_path=cache_path)

        train_dataset = SubbandDataset(base_train)
        _n_workers_train = n_workers
        _shuffle_train = (len(train_files) <= 500)

        def _make_loader(bs):
            return torch.utils.data.DataLoader(
                train_dataset, batch_size=bs,
                shuffle=_shuffle_train,
                num_workers=_n_workers_train,
                pin_memory=(device.type == 'cuda'),
                persistent_workers=True)

        loader = _make_loader(cfg.batch_size_warmup)
        print(f"[*] Workers: {n_workers} (LPC+lifting runs in parallel)")

    val_dataset = None
    val_loader = None
    if val_files:
        # Val set is small enough to always fit in RAM as precomputed L3.
        val_has_l3 = True
        for _vf in val_files[:5]:
            with np.load(_vf) as _vd:
                if 'l3' not in _vd.files:
                    val_has_l3 = False
                    break

        if val_has_l3:
            val_dataset = PrecomputedL3Dataset(
                val_files, windows_per_epoch=cfg.val_windows)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=cfg.batch_size,
                shuffle=False, num_workers=0,
                pin_memory=(device.type == 'cuda'))
        elif len(val_files) > 200:
            base_val = StreamingQ31Dataset(val_files, windows_per_epoch=cfg.val_windows)
            val_dataset = SubbandDataset(base_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=cfg.batch_size,
                shuffle=False, num_workers=min(4, os.cpu_count() or 4),
                pin_memory=(device.type == 'cuda'),
                persistent_workers=True)
        else:
            val_cache = os.path.join(ROOT_DIR, "ai_models/dataset_sim/q31_cache_val.pt")
            base_val = Q31Dataset(val_files, headless=True, cache_path=val_cache)
            val_dataset = SubbandDataset(base_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=cfg.batch_size,
                shuffle=False, num_workers=0,
                pin_memory=(device.type == 'cuda'))
        print(f"[*] Train: {len(train_dataset)} windows/epoch, Val: {len(val_dataset)} windows/epoch, "
              f"bs={cfg.batch_size_warmup}→{cfg.batch_size_quant}→{cfg.batch_size_fine} (AdaBatch)")
    else:
        print(f"[*] Train: {len(train_dataset)} windows/epoch (no val set), "
              f"bs={cfg.batch_size_warmup}→{cfg.batch_size_quant}→{cfg.batch_size_fine} (AdaBatch)")

    # torch.compile: dynamic=True avoids per-shape recompilation that caused
    # the previous 30 GB memory blowup (temporal dims 313/157/79 each triggered
    # a separate compiled kernel). mode="default" — not "reduce-overhead" which
    # uses CUDA graphs that conflict with the cached Hadamard matrix.
    # `student` stays unwrapped for introspection (.modules(), .state_dict(),
    # .cdf_breakpoints, etc.). `student_fwd` is used only in training loops.
    # Eagerly initialize all quantization parameters (alpha, INT8 scale)
    # and pre-populate Hadamard cache BEFORE torch.compile, so no lazy
    # construction happens inside the compiled/CUDA-graph-captured path.
    from lamquant_neural.models.blocks import warmup_hadamard_cache
    warmup_hadamard_cache(device)
    with torch.no_grad():
        dummy = torch.randn(1, 21, 313, device=device)
        student(dummy, quantize=True)
    student.ensure_initialized()

    # torch.compile: mode="default" with dynamic=True.
    # "reduce-overhead" (CUDA graphs) conflicts with the Hadamard matrix cache
    # in _quantize_activation — Dynamo traces the cache-miss path and tries to
    # capture tensor construction inside the graph. Not worth the complexity
    # for this small model where shard-based GPU batching already eliminated
    # the data transfer bottleneck.
    if device.type == 'cuda':
        try:
            import torch._dynamo.config as dynamo_config
            dynamo_config.recompile_limit = 32
            student_fwd = torch.compile(student, mode="default", dynamic=True)
            print(f"[*] torch.compile enabled (mode=default, dynamic=True)")
        except Exception as e:
            student_fwd = student
            print(f"[*] torch.compile failed, using eager mode: {e}")
    else:
        student_fwd = student

    # Trigger compile warmup, then calibrate shard budget with actual free VRAM
    if device.type == 'cuda' and _loader_is_precomputed and student_fwd is not student:
        with torch.no_grad():
            _ = student_fwd(torch.randn(cfg.batch_size_quant, 21, 313, device=device), quantize=True)
        torch.cuda.empty_cache()
        train_dataset.calibrate_shard_budget(device)

    # auraloss: spectral convergence + log-magnitude STFT (replaces hand-rolled SpectralLoss)
    from auraloss.freq import MultiResolutionSTFTLoss
    fft_sizes = list(cfg.spectral_fft_sizes)
    spectral_loss_fn = MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes,
        hop_sizes=[max(n // 4, 1) for n in fft_sizes],
        win_lengths=fft_sizes,
    ).to(device)
    start_time = time.time()
    best_r = 0.0
    best_val_r = 0.0
    best_epoch = 0
    completed_epoch = 0
    total_epochs = cfg.total_epochs
    epochs_warmup = cfg.epochs_warmup
    epochs_quant = cfg.epochs_quant
    epochs_fine = cfg.epochs_fine

    # Resume checkpoint path (separate from best-weights checkpoint)
    resume_path = os.path.join(ROOT_DIR, f"ai_models/student/student_resume_{preset_tag}.ckpt")

    def _save_resume(phase, phase_epoch, optimizer, scheduler):
        """Save full training state for crash recovery."""
        torch.save({
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'phase': phase,              # 1, 2, or 3
            'phase_epoch': phase_epoch,  # epoch within current phase
            'best_val_r': best_val_r,
            'best_r': best_r,
            'best_epoch': best_epoch,
            'completed_epoch': completed_epoch,
            'preset': cfg.name,
        }, resume_path)

    # --start-phase: skip directly to phase N (for warm starts)
    resume_phase = 0
    resume_phase_epoch = 0
    if args.start_phase is not None:
        resume_phase = args.start_phase
        print(f"[*] Skipping to Phase {resume_phase} (--start-phase)")

    # --resume: reload full training state if available (skip if --init-from warm start)
    if hasattr(cfg, '_resume') and cfg._resume and not args.init_from and os.path.exists(resume_path):
        rk = _safe_load(resume_path, map_location=device)
        student.load_state_dict(rk['model_state_dict'])
        resume_phase = rk.get('phase', 0)
        resume_phase_epoch = rk.get('phase_epoch', 0)
        best_val_r = rk.get('best_val_r', 0.0)
        best_r = rk.get('best_r', 0.0)
        best_epoch = rk.get('best_epoch', 0)
        completed_epoch = rk.get('completed_epoch', 0)
        print(f"[*] RESUME: Phase {resume_phase}, epoch {resume_phase_epoch}, "
              f"best ValR={best_val_r:.4f}")
        del rk

    # Tequila deadzone tau annealing: starts at 0.1 (Phase 2), decays to 0
    # by end of Phase 3. This lets trapped boundary weights contribute soft
    # corrections early in training, then gradually enforces pure ternary.
    # torch.compile wraps the model — `student` stays unwrapped for .modules()
    # introspection, `student_fwd` is the compiled wrapper for training loops.
    def set_deadzone_tau(model, tau):
        target = getattr(model, '_orig_mod', model)
        for m in target.modules():
            if hasattr(m, 'deadzone_tau'):
                m.deadzone_tau = tau

    # Standard dashboard — shared across all training scripts
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant'))
    from training_dashboard import TrainingDashboard

    dash = TrainingDashboard(
        model_name='Student Subband',
        gen='7.5',
        preset=cfg.name,
        total_epochs=total_epochs,
        device=device,
        emit_interval=50,
    )

    def _progress(phase, epoch, total_ep, batch_idx, n_batches, loss_val,
                  r_val=None, loss_parts=None, grad_norm=None):
        # Update ternary stats every 200 batches
        if batch_idx % 200 == 0:
            with torch.no_grad():
                dash.update_ternary_stats(student)
        dash.step(
            phase=phase, epoch=epoch, batch=batch_idx,
            n_batches=n_batches, loss=loss_val, r=r_val,
            loss_parts=loss_parts, grad_norm=grad_norm,
            extra={'lr': f"{optimizer.param_groups[0]['lr']:.2e}"},
        )

    # --- Training plotter (4-panel diagnostic figure) ---
    from training_plotter import TrainingPlotter
    plotter = TrainingPlotter(
        output_dir=os.path.join(ROOT_DIR, 'outputs'),
        plot_interval=cfg.val_interval,
    )

    # --- CheckpointManager — per-layer alpha CSV log + smoke check ---
    # Runs alongside the existing save logic (not replacing it). Added
    # 2026-04-16 to diagnose the QAT-degrades-warm phenomenon. The CSV
    # log lets us see when/why per-layer alpha drifts.
    try:
        from checkpoint_manager import CheckpointManager, GuardConfig
        _cm_alpha_csv = os.path.join(
            ROOT_DIR, 'training_logs',
            f'alpha_trajectory_{preset_tag}_{int(time.time())}.csv'
        )
        # Smoke-input: a tiny random tensor matched to the encoder shape.
        _cm_smoke = lambda: torch.randn(1, 21, 313, device=device)
        # Use generous bounds so CM doesn't halt the run while we're
        # diagnosing. The training script's own clamp + early-stop
        # logic remains in charge of stopping decisions.
        _ckpt_mgr = CheckpointManager(
            model=student,
            ckpt_path=os.path.join(
                ROOT_DIR, f'ai_models/student/student_cm_{preset_tag}_best.ckpt'),
            ckpt_dir=os.path.join(ROOT_DIR, 'ai_models/student/recovery'),
            device=device,
            smoke_input=_cm_smoke,
            alpha_log_csv=_cm_alpha_csv,
            guard=GuardConfig(
                r_plateau_patience=10**6,   # disabled — main script handles this
                alpha_max_safe=10**6,       # disabled — diagnostic mode
                alpha_min_safe=0,
                smoke_check_tolerance=0.5,  # very loose — smoke is informational
            ),
        )
        print(f"[*] CheckpointManager active: alpha CSV → {_cm_alpha_csv}")
    except Exception as e:
        print(f"[!] CheckpointManager could not start ({e}); continuing without alpha log")
        _ckpt_mgr = None

    # --- Ternary sparsity helpers (L1 reg + alpha clamping) ---
    # Decoder layer names for split L1 lambda
    _decoder_layers = {'expand1.conv', 'expand1.shortcut', 'expand2.conv',
                       'expand2.shortcut', 'expand3.conv', 'expand3.shortcut',
                       'output'}

    def _l1_ternary_loss():
        """L1 on fractional residual after ternary rounding.

        Split lambda: encoder uses l1_lambda, decoder uses l1_lambda_decoder
        (defaults to l1_lambda if not set). Decoder needs less sparsity
        pressure to maintain reconstruction capacity.
        """
        if cfg.l1_lambda <= 0:
            return torch.tensor(0.0, device=device)
        l1_enc = torch.tensor(0.0, device=device)
        l1_dec = torch.tensor(0.0, device=device)
        for name, m in student.named_modules():
            if hasattr(m, 'lsq_alpha') and hasattr(m, 'weight'):
                alpha = m.lsq_alpha.abs().clamp(min=1e-8)
                w_scaled = m.weight / alpha
                w_rounded = torch.clamp(torch.round(w_scaled), -1, 1)
                residual = w_scaled - w_rounded
                if name in _decoder_layers:
                    l1_dec = l1_dec + torch.mean(torch.abs(residual))
                else:
                    l1_enc = l1_enc + torch.mean(torch.abs(residual))
        lam_dec = cfg.l1_lambda_decoder if cfg.l1_lambda_decoder > 0 else cfg.l1_lambda
        return cfg.l1_lambda * l1_enc + lam_dec * l1_dec

    # Per-layer alpha floors (populated at Phase 2 start)
    _alpha_floors = {}

    # Decoder layers with elevated floor multiplier (1.0× instead of 0.8×).
    # expand2/expand3 are 131K weights (30% of network) stuck at 16-17% sparsity
    # because 0.8× floor is the binding constraint. 1.0× gives more headroom.
    _elevated_floor_layers = {'expand2.conv', 'expand2.shortcut',
                              'expand3.conv', 'expand3.shortcut'}

    def _compute_alpha_floors():
        """Snapshot σ_W per ternary layer, set floor = mult × σ_W.

        Default mult=0.8, elevated to 1.0 for large decoder layers.
        Anchored to weight scale before co-adaptation can collapse α and σ_W
        together. For Gaussian weights quantized by round(clamp(W/α, -1, 1)):
          α_optimal ≈ 1.22 × σ_W  (balanced trimodal: P(0) ≈ 1/3)
          α_floor   = 0.80 × σ_W  (leaves room for optimizer to reach optimum)
        """
        for name, m in student.named_modules():
            if hasattr(m, 'lsq_alpha') and hasattr(m, 'weight'):
                sigma_w = m.weight.data.std().item()
                mult = 1.0 if name in _elevated_floor_layers else 0.8
                floor = mult * sigma_w
                _alpha_floors[name] = floor
        if _alpha_floors:
            ceil_str = f", absolute ceiling={cfg.alpha_ceiling}" if cfg.alpha_ceiling > 0 else ""
            print(f"[*] Per-layer alpha floors (0.8×σ_W, 1.0× for expand2/3){ceil_str}:")
            for name, floor in sorted(_alpha_floors.items()):
                mult = 1.0 if name in _elevated_floor_layers else 0.8
                print(f"    {name:30s}  σ_W={floor/mult:.4f}  floor={floor:.4f}  ({mult:.1f}×)")

    def _clamp_alphas():
        """Clamp all ternary layer alphas.

        Five mechanisms (applied in order):
          0. Hard safety net (ALWAYS): α ∈ [1e-4, 20]  — prevents NaN /
             pathological runaway. Unconditional safety, not config-gated.
             Was missing before 2026-04-16; the production preset shipped
             with no clamping at all and alpha exploded to 144 on
             expand3.conv before training collapsed.
          1. Relative clamp: [0.5×std(W), 2×std(W)]  (--alpha-clamp)
          2. Per-layer init floor: 0.8×σ_W_init       (--alpha-floor-init)
          3. Uniform absolute floor                    (--alpha-floor 0.05)
          4. Absolute ceiling: α ≤ N                    (--alpha-ceiling 3.0)
        """
        for name, m in student.named_modules():
            if not hasattr(m, 'lsq_alpha'):
                continue
            # 0. UNCONDITIONAL safety: prevent runaway / NaN. Cheap.
            with torch.no_grad():
                m.lsq_alpha.data.clamp_(min=1e-4, max=20.0)
            # 1. Relative clamp (existing behavior)
            if cfg.alpha_clamp and hasattr(m, 'clamp_alpha'):
                m.clamp_alpha()
            # 2. Per-layer initialization-anchored floor
            if name in _alpha_floors:
                with torch.no_grad():
                    m.lsq_alpha.data.clamp_(min=_alpha_floors[name])
            # 3. Uniform absolute floor (fallback / override)
            if cfg.alpha_floor > 0:
                with torch.no_grad():
                    m.lsq_alpha.data.clamp_(min=cfg.alpha_floor)
            # 4. Absolute ceiling (prevents runaway alpha from over-sparsifying)
            if cfg.alpha_ceiling > 0:
                with torch.no_grad():
                    m.lsq_alpha.data.clamp_(max=cfg.alpha_ceiling)

    # --- Decoder norm gradient clipping ---
    # expand1.norm has gradient norms 143× larger than encoder convs, driving
    # alpha inflation on downstream layers. Clip decoder norm gradients to
    # prevent them from dominating the optimization landscape.
    _decoder_norm_params = [p for n, p in student.named_parameters()
                           if '.norm.' in n and any(d in n for d in ('expand1', 'expand2', 'expand3'))]

    def _clip_decoder_norm_grads(max_norm=1.0):
        for p in _decoder_norm_params:
            if p.grad is not None:
                p.grad.clamp_(-max_norm, max_norm)

    # --- Optimizer (single group — erf_scale is a frozen buffer, not a parameter) ---
    def _make_optimizer(base_lr, weight_decay):
        return torch.optim.AdamW(student.parameters(), lr=base_lr, weight_decay=weight_decay)

    # --- Temporal importance mask (raised cosine, edges=0.3) ---
    _temporal_mask = temporal_importance_mask(313, edge_weight=0.3, device=device)

    # --- MSE function selection ---
    def _mse_fn(recon, target):
        """MSE with temporal importance weighting + optional band weighting."""
        # Apply temporal importance mask: center=1.0, edges=0.3
        weighted_recon = recon * _temporal_mask
        weighted_target = target * _temporal_mask
        if cfg.band_weighted_mse:
            return band_weighted_mse(weighted_recon, weighted_target)
        return F.mse_loss(weighted_recon, weighted_target)

    mse_fn = _mse_fn
    if cfg.band_weighted_mse:
        print(f"[*] Using band-weighted MSE (delta 2.0x, theta 1.5x, alpha 1.5x)")
    print(f"[*] Temporal importance mask: center=1.0, edges=0.3 (raised cosine)")

    # ================================================================
    # PHASE 1: WARM-UP (no quantization, MSE + Pearson R + channel dropout)
    # ================================================================
    phase1_start = resume_phase_epoch + 1 if resume_phase == 1 else 1
    if resume_phase > 1:
        print(f"\n[*] Phase 1: SKIPPED (resume at Phase {resume_phase})")
    else:
        print(f"\n[*] Phase 1: Warm-up ({epochs_warmup} ep, lr={cfg.lr_warmup:.0e}, quantize=OFF, +R, +ch dropout)")
        if phase1_start > 1:
            print(f"    Resuming from epoch {phase1_start}")
    optimizer = _make_optimizer(cfg.lr_warmup, cfg.wd_warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_warmup, eta_min=cfg.lr_warmup_min)

    # Restore optimizer/scheduler if resuming within Phase 1
    if resume_phase == 1 and os.path.exists(resume_path):
        rk = _safe_load(resume_path, map_location=device)
        if 'optimizer_state_dict' in rk:
            optimizer.load_state_dict(rk['optimizer_state_dict'])
        if 'scheduler_state_dict' in rk:
            scheduler.load_state_dict(rk['scheduler_state_dict'])
        del rk

    n_batches = len(loader)
    for epoch in range(phase1_start if resume_phase <= 1 else epochs_warmup + 1, epochs_warmup + 1):
        student.train()
        losses, r_scores = [], []

        for batch_idx, (x_l3, x_eeg, mask) in enumerate(loader):
            x_l3 = x_l3.to(device, non_blocking=True)
            x_l3_aug = channel_dropout(x_l3, p_min=cfg.channel_dropout_min,
                                       p_max=cfg.channel_dropout_max, training=True)
            # x_l3_aug = eeg_augment(x_l3_aug, training=True)  # disabled: A/B testing auraloss alone

            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                recon_l3 = student_fwd(x_l3_aug, quantize=False)
                mse = mse_fn(recon_l3, x_l3)
                r_loss = pearson_r_loss(recon_l3, x_l3)
                loss = mse + cfg.pearson_r_weight * r_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip_warmup)
            _clip_decoder_norm_grads()
            optimizer.step()
            losses.append(loss.item())

            with torch.no_grad():
                r_val = pearson_r_batch(recon_l3, x_l3)
                r_scores.append(r_val)

            _progress("Warm", epoch, total_epochs, batch_idx, n_batches, loss.item(), r_val)

        scheduler.step()

        if epoch % 10 == 0:
            val_r, val_prd = validate_epoch(student, val_loader, device, quantize=False) if val_loader else (0.0, 100.0)
            if val_r > best_val_r:
                best_val_r = val_r
                best_r = float(np.mean(r_scores)) if r_scores else 0.0
                best_epoch = epoch
                # BUG FIX (2026-04-16): the previous version updated best_val_r
                # but never persisted the weights — so when the model peaked
                # during warmup at epoch 40 (the "0.8855 incident") and then
                # regressed during QAT, the best ckpt was lost forever and
                # the broken final state shipped. Save on every new best
                # like the QAT loop already does.
                torch.save(student.state_dict(), s_path)
                print(f"  [WARM] saved new best ValR={val_r:.4f} at ep {epoch} → {s_path}")
            dash.update_val(val_r, best_val_r)
            _run_guards(epoch, val_r=val_r, train_loss=loss.item() if 'loss' in dir() else None)
            _save_resume(1, epoch, optimizer, scheduler)
            if _ckpt_mgr is not None:
                try:
                    _ckpt_mgr.on_validation(epoch=epoch, val_r=val_r,
                                             val_prd=val_prd,
                                             train_r=float(np.mean(r_scores)) if r_scores else 0.0,
                                             raise_on_halt=False)
                except Exception:
                    pass   # diagnostic — never let CM crash the run
            elapsed = time.time() - start_time
            print(f"  [Warm] Ep {epoch:>3}/{epochs_warmup} | "
                  f"MSE: {np.mean(losses):.4f} | "
                  f"R(L3): {np.mean(r_scores):.4f} | "
                  f"ValR: {val_r:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"{elapsed:.0f}s")

    # ================================================================
    # PHASE 2: QUANTIZATION-AWARE (STE active, Tequila deadzone τ=0.1)
    # ================================================================
    # Save warm-best to a separate diagnostic path BEFORE resetting the
    # best tracker. Then reset best_val_r so QAT can save its own best
    # to the production path.
    #
    # Why this matters:
    #   - Warm phase validates with quantize=False (FP32). R values are
    #     naturally higher than QAT (e.g., 0.85).
    #   - QAT validates with quantize=True (ternary). R values are
    #     naturally lower (e.g., 0.78).
    #   - Comparing them with the same `best_val_r` means QAT-best never
    #     beats warm-best, so QAT NEVER overwrites the production ckpt.
    #   - Production then ships warm-phase FP32 weights running in
    #     ternary mode → R drops 30 points at deployment.
    #
    # Fix: separate the two scales.
    if best_val_r > 0 and best_epoch > 0:
        warm_diag = os.path.join(ROOT_DIR,
                                  f"ai_models/student/student_warm_best_{preset_tag}.ckpt")
        # The production path s_path currently holds warm weights — copy
        # them to the diagnostic path before they get overwritten by QAT.
        if os.path.exists(s_path):
            import shutil
            shutil.copy2(s_path, warm_diag)
            print(f"  [WARM→QAT] warm-best ValR={best_val_r:.4f} archived to {warm_diag}")
        # Reset trackers so QAT compares ternary R against ternary R only.
        best_val_r = 0.0
        best_r = 0.0
        best_epoch = 0

    set_deadzone_tau(student, cfg.deadzone_tau_initial)
    # Compute per-layer alpha floors from Phase 1 weight distributions
    if cfg.alpha_floor_init:
        _compute_alpha_floors()
    # AdaBatch: rebuild loader with Phase 2 batch size
    if cfg.batch_size_quant != cfg.batch_size_warmup:
        loader = _make_loader(cfg.batch_size_quant)
        n_batches = len(loader)
        print(f"[*] AdaBatch: bs {cfg.batch_size_warmup} → {cfg.batch_size_quant}")

    phase2_start = resume_phase_epoch + 1 if resume_phase == 2 else 1
    if resume_phase > 2:
        print(f"\n[*] Phase 2: SKIPPED (resume at Phase {resume_phase})")
    else:
        print(f"\n[*] Phase 2: QAT ({epochs_quant} ep, lr={cfg.lr_quant:.0e}, τ={cfg.deadzone_tau_initial}, bs={cfg.batch_size_quant})")
        if resume_phase == 2:
            print(f"    Resuming from epoch {phase2_start}")
    plotter.mark_phase(epochs_warmup, 'QAT')
    optimizer = _make_optimizer(cfg.lr_quant, cfg.wd_quant)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_quant, eta_min=cfg.lr_quant_min)

    if resume_phase == 2 and not args.init_from and os.path.exists(resume_path):
        rk = _safe_load(resume_path, map_location=device)
        if 'optimizer_state_dict' in rk:
            optimizer.load_state_dict(rk['optimizer_state_dict'])
        if 'scheduler_state_dict' in rk:
            scheduler.load_state_dict(rk['scheduler_state_dict'])
        del rk

    # Early-stop tracking for QAT.
    # BUG FIX (2026-04-16): the previous version saw 360 epochs of zero
    # improvement and kept running anyway (the guard warned but nothing
    # acted on it). Default patience: 60 val intervals (= 60 * val_interval
    # epochs).  Disabled with --no-early-stop / cfg.early_stop_patience=0.
    qat_patience = getattr(cfg, 'early_stop_patience', 60)
    qat_no_improve = 0
    qat_best_at_entry = best_val_r

    for epoch in range(phase2_start if resume_phase <= 2 else epochs_quant + 1, epochs_quant + 1):
        # Progressive ternary: soft for 80%, harden in final 20%
        # (P1: gives optimizer more time with soft weights before hard quantization)
        harden_start = int(epochs_quant * 0.8)
        if epoch <= harden_start:
            tau = cfg.deadzone_tau_initial  # constant soft ternary
        else:
            # Linear anneal in final 20%
            progress = (epoch - harden_start) / max(epochs_quant - harden_start, 1)
            tau = cfg.deadzone_tau_initial * (1.0 - progress)
        set_deadzone_tau(student, tau)

        # Reduce LR at hardening boundary for stability
        if epoch == harden_start + 1:
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.5
            print(f"  [Harden] τ anneal begins at epoch {epoch}, LR halved")

        # Two-stage weight decay (BitNet b1.58). Experiment result: +0.0007 R.
        # Normal WD for first 2/3, remove entirely for final 1/3.
        # High WD causes frequent ternary flips late in training.
        if cfg.wd_two_stage and epoch == int(epochs_quant * 2 / 3) + 1:
            for pg in optimizer.param_groups:
                pg['weight_decay'] = 0.0
            print(f"  [WD] Weight decay → 0 at epoch {epoch}/{epochs_quant} (2/3 point)")

        student.train()
        losses, r_scores = [], []

        for batch_idx, (x_l3, x_eeg, mask) in enumerate(loader):
            x_l3 = x_l3.to(device, non_blocking=True)
            x_l3_aug = channel_dropout(x_l3, p_min=cfg.channel_dropout_min,
                                       p_max=cfg.channel_dropout_max, training=True)

            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                recon_l3 = student_fwd(x_l3_aug, quantize=True)
                mse = mse_fn(recon_l3, x_l3)
                r_loss = pearson_r_loss(recon_l3, x_l3)
                # Unified loss: MSE + R + spectral + L1 + SNAC (all from epoch 1)
                loss = mse + cfg.pearson_r_weight * r_loss + _l1_ternary_loss()
                if cfg.spectral_weight > 0:
                    spec = spectral_loss_fn(recon_l3, x_l3)
                    loss = loss + cfg.spectral_weight * spec
                if _snac_fsq is not None:
                    with torch.no_grad():
                        latent = student.encode(x_l3_aug, quantize=True)
                    _, _, snac_qloss = _snac_fsq(latent)
                    loss = loss + 0.1 * snac_qloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip_quant)
            _clip_decoder_norm_grads()
            optimizer.step()
            _clamp_alphas()
            # erf_scale is a frozen buffer — no clamping needed
            losses.append(loss.item())

            with torch.no_grad():
                r_val = pearson_r_batch(recon_l3, x_l3)
                r_scores.append(r_val)

            _progress("QAT", epochs_warmup + epoch, total_epochs, batch_idx, n_batches, loss.item(), r_val)

        scheduler.step()

        if epoch % cfg.val_interval == 0:
            mean_r = np.mean(r_scores)
            val_r, val_prd = validate_epoch(student, val_loader, device) if val_loader else (0.0, 100.0)
            elapsed = time.time() - start_time
            ep_total = epochs_warmup + epoch

            completed_epoch = ep_total
            if val_r > best_val_r:
                best_val_r = val_r
                best_r = mean_r
                best_epoch = ep_total
                torch.save(student.state_dict(), s_path)
                qat_no_improve = 0   # reset patience counter on any improvement
            else:
                qat_no_improve += 1
            dash.update_val(val_r, best_val_r)
            _run_guards(ep_total, val_r=val_r, train_loss=mean_r)
            _save_resume(2, epoch, optimizer, scheduler)
            if _ckpt_mgr is not None:
                try:
                    _ckpt_mgr.on_validation(epoch=ep_total, val_r=val_r,
                                             val_prd=val_prd,
                                             train_r=float(mean_r), raise_on_halt=False)
                except Exception:
                    pass

            # Early stop on R plateau. The training_guard module already
            # warns about this — we now ACT on it instead of just logging.
            if qat_patience > 0 and qat_no_improve >= qat_patience:
                print(f"\n  [EARLY STOP] No ValR improvement for {qat_no_improve} "
                      f"validation intervals (≈{qat_no_improve * cfg.val_interval} epochs). "
                      f"Best ValR={best_val_r:.4f} at ep {best_epoch}. "
                      f"Stopping QAT phase early at ep {ep_total}/{total_epochs}.")
                break

            # Latent kurtosis tracking (tanh distribution reshaping diagnostic)
            mean_kurt, per_ch_kurt = latent_kurtosis(student, val_loader, device)
            erf_str = " | CDF-LUT"

            # Per-channel kurtosis: flag worst offenders (esp. ch5)
            top5_idx = np.argsort(per_ch_kurt)[-5:][::-1]
            top5_str = ", ".join(f"ch{i}={per_ch_kurt[i]:.1f}" for i in top5_idx)
            ch5_kurt = per_ch_kurt[5] if len(per_ch_kurt) > 5 else 0.0

            print(f"  [QAT] Ep {epoch:>3}/{epochs_quant} ({ep_total}/{total_epochs}) | "
                  f"MSE: {np.mean(losses):.4f} | "
                  f"R(L3): {mean_r:.4f} | "
                  f"ValR: {val_r:.4f} | "
                  f"Best: {best_val_r:.4f} | "
                  f"Kurt: {mean_kurt:.2f} (ch5={ch5_kurt:.1f}){erf_str} | "
                  f"τ: {tau:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"{elapsed:.0f}s")
            print(f"    Kurt top5: {top5_str}")
            with torch.no_grad():
                dash.update_ternary_stats(student)
            dash.log_per_layer_sparsity()
            # Plot diagnostics
            pls = {k: v['sparsity'] for k, v in getattr(dash, '_per_layer_sparse', {}).items()}
            pla = {k: v['alpha'] for k, v in getattr(dash, '_per_layer_sparse', {}).items()}
            plotter.log_epoch(
                epoch=ep_total, train_r=mean_r, val_r=val_r,
                global_sparsity=dash._last_sparse,
                per_layer_sparsity=pls, per_layer_alpha=pla,
                loss_total=np.mean(losses), loss_mse=np.mean(losses),
                loss_r=cfg.pearson_r_weight * (1 - mean_r),
            )

    # ================================================================
    # PHASE 3: FINE-TUNE (MSE + R + spectral, τ anneals 0.1→0, 300 epochs)
    # Extended by 50 epochs vs original 250 to give the combined loss
    # (MSE + R + spectral) time to settle before τ forces pure ternary.
    # ================================================================
    # AdaBatch: rebuild loader with Phase 3 batch size
    if cfg.batch_size_fine != cfg.batch_size_quant:
        loader = _make_loader(cfg.batch_size_fine)
        n_batches = len(loader)
        print(f"[*] AdaBatch: bs {cfg.batch_size_quant} → {cfg.batch_size_fine}")

    phase3_start = resume_phase_epoch + 1 if resume_phase == 3 else 1
    print(f"\n[*] Phase 3: Fine-tune ({epochs_fine} ep, lr={cfg.lr_fine:.0e}, +spectral+R, τ→0, bs={cfg.batch_size_fine})")
    plotter.mark_phase(epochs_warmup + epochs_quant, 'Fine')
    if resume_phase == 3:
        print(f"    Resuming from epoch {phase3_start}")
    optimizer = _make_optimizer(cfg.lr_fine, cfg.wd_fine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_fine, eta_min=cfg.lr_fine_min)

    if resume_phase == 3 and not args.init_from and os.path.exists(resume_path):
        rk = _safe_load(resume_path, map_location=device)
        if 'optimizer_state_dict' in rk:
            optimizer.load_state_dict(rk['optimizer_state_dict'])
        if 'scheduler_state_dict' in rk:
            scheduler.load_state_dict(rk['scheduler_state_dict'])
        del rk

    _p3_no_improve = 0  # early stopping counter
    for epoch in range(phase3_start, epochs_fine + 1):
        # Anneal deadzone τ linearly over Phase 3
        tau = cfg.deadzone_tau_initial * (1.0 - epoch / epochs_fine)
        set_deadzone_tau(student, tau)

        student.train()
        losses, r_scores = [], []

        for batch_idx, (x_l3, x_eeg, mask) in enumerate(loader):
            x_l3 = x_l3.to(device, non_blocking=True)
            x_l3_aug = channel_dropout(x_l3, p_min=cfg.channel_dropout_min,
                                       p_max=cfg.channel_dropout_max, training=True)
            # x_l3_aug = eeg_augment(x_l3_aug, training=True)  # disabled: A/B testing auraloss alone

            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
                recon_l3 = student_fwd(x_l3_aug, quantize=True)
                mse = mse_fn(recon_l3, x_l3)
                r_loss = pearson_r_loss(recon_l3, x_l3)
                spec = spectral_loss_fn(recon_l3, x_l3)
                loss = mse + cfg.pearson_r_weight * r_loss + cfg.spectral_weight * spec + _l1_ternary_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip_fine)
            _clip_decoder_norm_grads()
            optimizer.step()
            _clamp_alphas()
            # erf_scale is a frozen buffer — no clamping needed
            losses.append(loss.item())

            with torch.no_grad():
                r_val = pearson_r_batch(recon_l3, x_l3)
                r_scores.append(r_val)

            _progress("Fine", epochs_warmup + epochs_quant + epoch, total_epochs,
                      batch_idx, n_batches, loss.item(), r_val)

        scheduler.step()

        if epoch % cfg.val_interval == 0:
            mean_r = np.mean(r_scores)
            val_r, val_prd = validate_epoch(student, val_loader, device) if val_loader else (0.0, 100.0)
            elapsed = time.time() - start_time
            ep_total = epochs_warmup + epochs_quant + epoch

            completed_epoch = ep_total
            _run_guards(ep_total, val_r=val_r, train_loss=mean_r)
            if val_r > best_val_r:
                best_val_r = val_r
                best_r = mean_r
                best_epoch = ep_total
                torch.save(student.state_dict(), s_path)
                _p3_no_improve = 0
            else:
                _p3_no_improve += 1
                if _p3_no_improve >= 20:
                    print(f"  [Fine] Early stop: no improvement for 20 val checks "
                          f"(best={best_val_r:.4f} at ep {best_epoch})")
                    break
            dash.update_val(val_r, best_val_r)
            _save_resume(3, epoch, optimizer, scheduler)
            if _ckpt_mgr is not None:
                try:
                    _ckpt_mgr.on_validation(epoch=ep_total, val_r=val_r,
                                             val_prd=val_prd,
                                             train_r=float(mean_r), raise_on_halt=False)
                except Exception:
                    pass

            mean_kurt, per_ch_kurt = latent_kurtosis(student, val_loader, device)
            erf_str = " | CDF-LUT"

            # Per-channel kurtosis: flag worst offenders (esp. ch5)
            top5_idx = np.argsort(per_ch_kurt)[-5:][::-1]
            top5_str = ", ".join(f"ch{i}={per_ch_kurt[i]:.1f}" for i in top5_idx)
            ch5_kurt = per_ch_kurt[5] if len(per_ch_kurt) > 5 else 0.0

            print(f"  [Fine] Ep {epoch:>3}/{epochs_fine} ({ep_total}/{total_epochs}) | "
                  f"Loss: {np.mean(losses):.4f} | "
                  f"R(L3): {mean_r:.4f} | "
                  f"ValR: {val_r:.4f} | "
                  f"Best: {best_val_r:.4f} | "
                  f"Kurt: {mean_kurt:.2f} (ch5={ch5_kurt:.1f}){erf_str} | "
                  f"τ: {tau:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"{elapsed:.0f}s")
            print(f"    Kurt top5: {top5_str}")
            with torch.no_grad():
                dash.update_ternary_stats(student)
            dash.log_per_layer_sparsity()
            # Plot diagnostics
            pls = {k: v['sparsity'] for k, v in getattr(dash, '_per_layer_sparse', {}).items()}
            pla = {k: v['alpha'] for k, v in getattr(dash, '_per_layer_sparse', {}).items()}
            plotter.log_epoch(
                epoch=ep_total, train_r=mean_r, val_r=val_r,
                global_sparsity=dash._last_sparse,
                per_layer_sparsity=pls, per_layer_alpha=pla,
                loss_total=np.mean(losses), loss_mse=np.mean(losses),
                loss_r=cfg.pearson_r_weight * (1 - mean_r),
                loss_spectral=cfg.spectral_weight * np.mean(losses),
                loss_l1=cfg.l1_lambda if cfg.l1_lambda > 0 else 0,
            )

    # Training complete — clean up resume checkpoint
    if os.path.exists(resume_path):
        os.remove(resume_path)
        print(f"[*] Removed resume checkpoint (training complete)")

    # Final validation on best checkpoint
    if os.path.exists(s_path):
        student.load_state_dict(torch.load(s_path, map_location=device, weights_only=True), strict=False)
    final_val_r, val_prd = validate_epoch(student, val_loader, device) if val_loader else (0.0, 100.0)
    print(f"\n{guard.summary()}")

    # Save with preset tag + epoch info (best_epoch of total_epochs attempted)
    completed_name = f"student_{preset_tag}_{best_epoch}of{total_epochs}_completed.ckpt"
    completed_path = os.path.join(ROOT_DIR, "ai_models/student", completed_name)
    torch.save(student.state_dict(), completed_path)

    # Distributable copy tagged with preset grade
    dist_path = os.path.join(ROOT_DIR, "weights", f"student_subband_{preset_tag}.ckpt")
    os.makedirs(os.path.dirname(dist_path), exist_ok=True)
    torch.save(student.state_dict(), dist_path)

    # Also save as the canonical name for the best available grade.
    # Gold > std > fast. Only overwrite if this preset is equal or higher grade.
    GRADE_RANK = {'fast': 0, 'std': 1, 'gold': 2, 'custom': 1}
    canonical_path = os.path.join(ROOT_DIR, "weights", "student_subband.ckpt")
    should_overwrite = True
    if os.path.exists(canonical_path):
        # Check if existing canonical is a higher grade
        for tag, rank in GRADE_RANK.items():
            existing_tagged = os.path.join(ROOT_DIR, "weights", f"student_subband_{tag}.ckpt")
            if os.path.exists(existing_tagged) and rank > GRADE_RANK.get(preset_tag, 1):
                should_overwrite = False
                break
    if should_overwrite:
        torch.save(student.state_dict(), canonical_path)

    elapsed = time.time() - start_time
    print(f"\n[*] Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"[*] Preset: {cfg.name} ({preset_tag})")
    print(f"[*] Best epoch: {best_epoch}/{total_epochs}")
    print(f"[*] Best Train R(L3): {best_r:.4f}")
    print(f"[*] Best Val R(L3):   {best_val_r:.4f}")
    print(f"[*] Final Val R(L3):  {final_val_r:.4f}")
    print(f"[*] Saved -> {completed_path}")
    print(f"[*] Saved -> {dist_path} ({preset_tag} grade)")
    if should_overwrite:
        print(f"[*] Saved -> {canonical_path} (canonical — highest grade)")


if __name__ == "__main__":
    from training_config import CONFIGS, TrainingConfig

    parser = argparse.ArgumentParser(
        description="Gen 7.1 Subband TNN Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Presets:\n" + "\n".join(
            f"  {name:12s} {c.total_epochs} ep, bs={c.batch_size}  {c.description[:60]}"
            for name, c in CONFIGS.items()
        ))
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default='production',
                        help="Training preset (default: production)")
    parser.add_argument("--config-yaml", type=str, default=None,
                        help="Path to YAML config file (overrides --config)")
    parser.add_argument("--activation-bits", type=int, default=None, choices=[8, 16],
                        help="Override activation bit width (default: from config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved checkpoint (phase + epoch + optimizer)")
    parser.add_argument("--quantization-mode", choices=['ternary', 'binary'],
                        default='ternary',
                        help="Weight quantization: ternary {-1,0,+1} or binary {-1,+1}")
    parser.add_argument("--l1-lambda", type=float, default=None,
                        help="L1 sparsity penalty (overrides config)")
    parser.add_argument("--alpha-clamp", action="store_true", default=None,
                        help="Enable alpha clamping (overrides config)")
    parser.add_argument("--band-weighted-mse", action="store_true", default=None,
                        help="Use frequency-weighted MSE (2x delta, 1.5x theta/alpha)")
    parser.add_argument("--alpha-floor", type=float, default=None,
                        help="Uniform absolute minimum alpha per layer (e.g. 0.05)")
    parser.add_argument("--alpha-floor-init", action="store_true", default=None,
                        help="Per-layer alpha floor anchored to 0.8×σ_W at Phase 2 start")
    parser.add_argument("--alpha-ceiling", type=float, default=None,
                        help="Absolute alpha ceiling for all layers (e.g. 3.0)")
    parser.add_argument("--l1-lambda-decoder", type=float, default=None,
                        help="Separate L1 lambda for decoder layers (less sparsity pressure)")
    parser.add_argument("--epochs-quant", type=int, default=None,
                        help="Override QAT epoch count")
    parser.add_argument("--epochs-fine", type=int, default=None,
                        help="Override fine-tune epoch count")
    parser.add_argument("--spectral-weight", type=float, default=None,
                        help="Override spectral loss weight")
    parser.add_argument("--cdf-entries", type=int, default=32,
                        help="CDF-LUT entries per channel (32=2KB, 64=4KB firmware)")
    parser.add_argument("--fp32-ratio", type=float, default=None,
                        help="Fraction of training as FP32 before QAT (e.g. 0.85 = 85%% FP32)")
    parser.add_argument("--reinit-layers", type=str, default=None,
                        help="Force reinit of named layers (comma-separated)")
    parser.add_argument("--val-interval", type=int, default=None,
                        help="Validate every N epochs (overrides config)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Load weights from checkpoint but train from epoch 0 (warm start)")
    parser.add_argument("--v1", action="store_true",
                        help="Use V1 architecture (w=128, 3 focal) instead of V2 (w=216, 4 DW-sep)")
    parser.add_argument("--start-phase", type=int, choices=[1, 2, 3], default=None,
                        help="Skip to phase N (1=warmup, 2=QAT, 3=fine-tune)")
    args = parser.parse_args()

    # Load config
    if args.config_yaml:
        cfg = TrainingConfig.from_yaml(args.config_yaml)
        print(f"[*] Loaded config from {args.config_yaml}")
    else:
        cfg = CONFIGS[args.config]

    # CLI overrides
    if args.activation_bits is not None:
        cfg = cfg.replace(activation_bits=args.activation_bits)

    # Set activation bit width and quantization mode before model construction
    from lamquant_neural.models.blocks import set_activation_bits, set_quantization_mode
    set_activation_bits(cfg.activation_bits)
    set_quantization_mode(args.quantization_mode)
    if args.quantization_mode == 'binary':
        print(f"[*] BINARY MODE: weights are {{-α, +α}} only, no zeros")

    # CLI overrides for sparsity controls
    if args.l1_lambda is not None:
        cfg = cfg.replace(l1_lambda=args.l1_lambda)
    if args.alpha_clamp is not None:
        cfg = cfg.replace(alpha_clamp=args.alpha_clamp)
    if args.band_weighted_mse is not None:
        cfg = cfg.replace(band_weighted_mse=args.band_weighted_mse)
    if args.alpha_floor is not None:
        cfg = cfg.replace(alpha_floor=args.alpha_floor)
    if args.alpha_floor_init is not None:
        cfg = cfg.replace(alpha_floor_init=args.alpha_floor_init)
    if args.val_interval is not None:
        cfg = cfg.replace(val_interval=args.val_interval)
    if args.alpha_ceiling is not None:
        cfg = cfg.replace(alpha_ceiling=args.alpha_ceiling)
    if args.l1_lambda_decoder is not None:
        cfg = cfg.replace(l1_lambda_decoder=args.l1_lambda_decoder)
    if args.epochs_quant is not None:
        cfg = cfg.replace(epochs_quant=args.epochs_quant)
    if args.epochs_fine is not None:
        cfg = cfg.replace(epochs_fine=args.epochs_fine)
    if args.spectral_weight is not None:
        cfg = cfg.replace(spectral_weight=args.spectral_weight)

    # Pass resume flag via non-frozen attribute (TrainingConfig is frozen dataclass)
    object.__setattr__(cfg, '_resume', args.resume)

    run(cfg=cfg)
