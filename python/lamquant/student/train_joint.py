#!/usr/bin/env python3
"""train_joint.py — joint encoder + Vocos decoder training.

Replaces the encoder-solo / mini-decoder pattern. Trains:

  TernaryMobileNetV5_Subband.encode  (ships to MCU, ternary QAT)
                  +
  VocosDecoder Tier 3                (ships to base station, FP32)

end-to-end, with the loss measured on the ACTUAL deployed reconstruction
path. After training, the encoder and decoder are saved separately:

  weights/student_encoder_joint.ckpt   ← firmware export target
  weights/decoder_tier3_joint.ckpt     ← base station target

Optimization:
  * Two parameter groups via CheckpointManager.make_param_groups():
      group 0: encoder (non-alpha) + decoder, weight_decay = cfg.wd_quant
      group 1: encoder LSQ alphas, weight_decay = 1e-3 (BitNet-style)
  * Encoder ternary QAT identical to train_student_subband (quantize=True
    after warmup).
  * Decoder stays FP32 throughout.

Safety:
  * CheckpointManager owns saves: best-on-improvement (with smoke-check)
    + every-50-epoch recovery snapshot + per-layer alpha CSV log.
  * Hard early-stop on R-plateau and alpha-explosion.

This is a fast-preset-only script. Long production runs are intentionally
gated until joint training proves out repeatedly on fast presets.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Optional


def _nullctx():
    """No-op context manager for the amp=False path."""
    return contextlib.nullcontext()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE (#255): No donated_buffer=False workaround here. torch.compile
# (reduce-overhead) DONATES backward buffers, which forbids retain_graph=True /
# create_graph=True (a second backward through an already-freed graph). The only
# retain_graph double-backward in this file was the first-batch
# _gradient_health_check; it now reads grad norms from the main loop's SINGLE
# backward instead of running its own (see _gradient_health_check + its warm-loop
# call site). No other path double-backwards one graph: warm = one backward;
# diagnostics overfit/grad-flow each backward their OWN fresh forward; GAN d_loss
# uses disc(fake.detach()) and g_loss uses a fresh disc(fake) + real_feats
# .detach() (discriminator.py:261, adv uses fake_scores only), so g never
# re-traverses the freed disc(real) graph. All single-backward → safe with
# donated buffers, so reduce-overhead + CUDA graphs run at full speed.

# torch.compile(mode='reduce-overhead') captures CUDA graphs per
# distinct input shape. Joint training has 4-9 distinct shapes
# (warm batch / QAT batch / val batch / last-partial-batch / shard
# tail / variable T from iSTFT length nudge), so by default we'd see
# "Recording too many CUDAGraphs" warnings and pay the per-shape
# capture overhead. This switch tells inductor to skip the graph
# capture for novel shapes and just run the BF16-compiled kernels in
# eager mode — same speedup on hot shapes, no overhead on cold ones.
try:
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
except (AttributeError, ImportError):
    pass   # older torch — the warning will just remain

# TF32 matmul: ~15% faster FP32 matmul, 10-bit vs 23-bit mantissa.
# Decoder trains in BF16 (unaffected). Primary benefit is SOAP preconditioner
# computation (Shampoo-style G@G.T + matrix square roots) and encoder FP32
# forward/backward during the warm phase. SOAP is robust to this precision
# level — preconditioners are approximations anyway.
torch.set_float32_matmul_precision('high')
# cuDNN algorithm search: benchmarks conv algorithms per (op, shape) pair once,
# then reuses the fastest. Fixed input sizes (21×313 encoder, 21×2500 decoder)
# means benchmarking happens ~3-5 times total, then is free. Compatible with
# gradient checkpointing (cuDNN records the algorithm choice for recomputation).
torch.backends.cudnn.benchmark = True

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'student'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'common'))  # MOVE-B: common DTOs
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'decoder'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'oracle'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant'))

from joint_codec import (
    JointCodec, build_default_joint, DEPLOYMENT_TIERS,
    TIER_DEV, TIER_MOBILE, TIER_CLINICAL, TIER_RESEARCH,
)
from checkpoint_manager import (
    CheckpointManager, GuardConfig, TrainingHaltException, make_param_groups,
)
from training_config import CONFIGS
from training_types import (
    EpochReport, RunSummary, TrainingLogger,
    alpha_stats_from_model, reduce_alpha_stats,
)
from training_dashboard import TrainingDashboard
from train_student_subband import (
    pearson_r_batch, channel_dropout, validate_epoch as _vendored_validate,
)
from augmentations import EEGAugmentor
from lamquant.common.utils import safe_torch_load as _safe_load
import math

sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'decoder'))
from discriminator import EEGDiscriminator
from seizure_head import SeizureHead


# ============================================================
# Joint validation — encode → decode end-to-end (NOT through mini-decoder)
# ============================================================

def validate_joint(model: JointCodec, val_ds, device, quantize=True,
                    batch_size: int = 64, per_band_sample: int = 512,
                    amp: bool = True, per_category: bool = False,
                    channel_agnostic: bool = False, variable_n: bool = False,
                    n_range: tuple = (8, 21)):
    """End-to-end joint validation. Returns (val_r, val_prd, per_band_prd_dict)
    or (val_r, val_prd, per_band_prd_dict, category_metrics) if per_category=True.

    - val_r:   Pearson R, mean across batches
    - val_prd: PRD %, mean across batches  (lower is better)
    - per_band_prd_dict: {band_name: prd_percent} from a CPU-side sample.
    - category_metrics (optional): {cat: {'r': float, 'prd': float, 'n': int}}

    Per-band PRD runs on the first ~per_band_sample windows (CPU + scipy
    bandpass) — too expensive on every batch. The L3 sample rate is
    ~31 Hz so only delta/theta/alpha are recoverable; beta has partial
    coverage (13-15 Hz) and gamma is above Nyquist. The LQS check at
    end-of-run uses what we have; for fullband per-band PRD run the
    codec benchmark separately on a holdout.

    assert_no_leakage(Split.VAL) is the runtime safety net.
    """
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant'))
    from data_types import Split as _Split
    from metrics import (prd_torch, per_band_prd as _per_band_prd,
                         masked_pearson_r_batch, masked_prd_torch)

    # Accept either a string ('cpu', 'cuda') or a torch.device — old
    # tests pass strings and we promised backward compat.
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    rs = []
    prds = []
    cpu_orig_chunks = []
    cpu_recon_chunks = []
    cpu_collected = 0
    val_fs = 31.3   # default to L3-domain fs; switched to 250 below if fullband
    # Per-category accumulators (only active when per_category=True)
    cat_r_acc = {}    # {cat: [r_values]}
    cat_prd_acc = {}  # {cat: [prd_values]}
    # Validation runs under inference-mode autocast when the train loop
    # used it — same dtype context for the model, ~1.8× faster eval.
    # Per-band PRD is still computed in float64 on CPU (scipy bandpass).
    val_ctx = (torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
               if amp else contextlib.nullcontext())
    with torch.no_grad(), val_ctx:
        for batch in val_ds.prefetch_typed_batches(batch_size=batch_size, device=device):
            batch.assert_no_leakage(_Split.VAL)   # safety net
            x_l3 = batch.l3_approx
            # CA: subset channels (variable-N) or pass through (N=21 parity);
            # coords/ch_mask = None on the parity path -> identical to legacy.
            x_l3, _fb_v, _coords_v, _cmask_v = _ca_inputs(
                x_l3, batch.fullband_target, channel_agnostic, variable_n, n_range)
            recon = model(x_l3, quantize=quantize, coords=_coords_v, ch_mask=_cmask_v)
            # Auto-detect domain: if the decoder emits fullband-shaped
            # output AND the dataset provides fullband targets, validate
            # against fullband. Otherwise fall back to L3.
            fb_target = _fb_v                     # subset fullband (CA) or original
            use_fullband = (fb_target is not None
                            and abs(recon.shape[-1] - fb_target.shape[-1]) <= 8)
            if use_fullband:
                val_fs = 250.0
                target = fb_target
            else:
                target = x_l3
            T = min(recon.shape[-1], target.shape[-1])
            r_crop = recon[..., :T].float()      # promote BF16→FP32 for stable
            x_crop = target[..., :T].float()     # numpy export + R/PRD math
            batch_r = masked_pearson_r_batch(r_crop, x_crop, _cmask_v)
            batch_prd = float(masked_prd_torch(x_crop, r_crop, _cmask_v))
            _bs = r_crop.shape[0]
            if os.environ.get('LAMQUANT_VAL_DEBUG') and not rs:
                print(f"[VAL_DEBUG] batch0: x_l3={tuple(x_l3.shape)} "
                      f"recon={tuple(recon.shape)} "
                      f"fb_target={None if fb_target is None else tuple(fb_target.shape)} "
                      f"use_fullband={use_fullband} batch_r={batch_r:.4f} "
                      f"batch_prd={batch_prd:.2f}", flush=True)
            rs.append((batch_r, _bs))
            prds.append((batch_prd, _bs))
            # Per-category metrics: accumulate R/PRD grouped by clinical category
            if per_category and hasattr(batch, 'clinical_categories') and batch.clinical_categories:
                # Compute per-sample R for category grouping
                for i_s, cat in enumerate(batch.clinical_categories):
                    if i_s >= r_crop.shape[0]:
                        break
                    s_r = r_crop[i_s:i_s+1]
                    s_x = x_crop[i_s:i_s+1]
                    sample_r = float(pearson_r_batch(s_r, s_x))
                    # Per-sample PRD
                    diff = (s_x - s_r)
                    sample_prd = float(torch.sqrt((diff * diff).sum() / (s_x * s_x).sum().clamp(min=1e-12)) * 100)
                    cat_r_acc.setdefault(cat, []).append(sample_r)
                    cat_prd_acc.setdefault(cat, []).append(sample_prd)
            # Per-band CPU accumulation needs a CONSISTENT channel count across
            # batches (np.concatenate). variable-N yields a different k per batch
            # → skip per-band there (a mixed-montage per-band PRD isn't meaningful
            # anyway; run the codec bench on a fixed holdout for that).
            if not variable_n and cpu_collected < per_band_sample:
                n = min(x_crop.shape[0], per_band_sample - cpu_collected)
                cpu_orig_chunks.append(x_crop[:n].detach().cpu().numpy())
                cpu_recon_chunks.append(r_crop[:n].detach().cpu().numpy())
                cpu_collected += n

    if os.environ.get('LAMQUANT_VAL_DEBUG'):
        print(f"[VAL_DEBUG] total val batches={len(rs)} "
              f"total_samples={sum(b for _, b in rs)}", flush=True)
    if not rs:
        empty = (0.0, 0.0, {}, {}) if per_category else (0.0, 0.0, {})
        return empty

    # Weighted mean by batch size (avoids bias from smaller last batch)
    _total_samples = sum(bs for _, bs in rs)
    val_r = float(sum(r * bs for r, bs in rs) / max(_total_samples, 1))
    val_prd = float(sum(p * bs for p, bs in prds) / max(_total_samples, 1))

    per_band = {}
    if cpu_orig_chunks:
        orig_cat = np.concatenate(cpu_orig_chunks, axis=0)
        recon_cat = np.concatenate(cpu_recon_chunks, axis=0)
        # When val_fs == 250 (fullband target), all five EEG bands fit
        # comfortably below Nyquist (125 Hz). When val_fs == 31.3 (L3
        # proxy), only δ/θ/α are recoverable — β/γ get clamped to 0 by
        # per_band_prd's Nyquist guard.
        per_band = _per_band_prd(orig_cat, recon_cat, fs=val_fs, order=2)

    if per_category:
        cat_metrics = {}
        for cat in set(list(cat_r_acc.keys()) + list(cat_prd_acc.keys())):
            r_vals = cat_r_acc.get(cat, [])
            prd_vals = cat_prd_acc.get(cat, [])
            cat_metrics[cat] = {
                'r': float(np.mean(r_vals)) if r_vals else 0.0,
                'prd': float(np.mean(prd_vals)) if prd_vals else 0.0,
                'n': len(r_vals),
            }
        return val_r, val_prd, per_band, cat_metrics

    return val_r, val_prd, per_band


# ============================================================
# Spectral loss (multi-resolution STFT) — same as encoder-solo
# ============================================================

def make_spectral_loss(device):
    """Multi-resolution STFT loss tuned for EEG frequency bands.

    FFT sizes chosen to resolve the clinically important EEG bands
    at 250 Hz (L3 proxy at 31.25 Hz):
      64:  captures delta (0.5-4 Hz) — 0.49 Hz resolution
      128: captures theta (4-8 Hz) — 0.24 Hz resolution
      256: captures alpha (8-13 Hz) — 0.12 Hz resolution
      512: captures beta (13-30 Hz) — 0.06 Hz resolution

    The old audio-tuned sizes [16, 32, 64] couldn't resolve any EEG
    band — 16-point FFT at 31 Hz gives 1.95 Hz bins, smearing delta
    and theta together.
    """
    from auraloss.freq import MultiResolutionSTFTLoss
    fft_sizes = [64, 128, 256, 512]
    return MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes,
        hop_sizes=[max(n // 4, 1) for n in fft_sizes],
        win_lengths=fft_sizes,
    ).to(device)


# ============================================================
# WSD Learning Rate Schedule
# ============================================================

class WSDScheduler:
    """Warmup-Stable-Decay scheduler for continual training.

    Three phases (user direction 2026-05-21, cosine-then-WSD):
      1. Warmup (cosine ramp): LR 0 → peak via half-cosine, smoother
         than linear at the start and end of the warmup window.
      2. Stable (constant): LR = peak, shippable any time.
      3. Decay (cosine): LR peak → min_lr over decay_epochs.

    Pass ``warmup_kind="linear"`` to restore the legacy linear ramp.

    Infinite mode (decay_frac=0):
      Stable phase runs forever. The model trains at peak LR
      indefinitely — every checkpoint is shippable. When you
      want to finalize, call trigger_decay(n_epochs) to start
      the cosine cooldown manually.

    For continual training: resume from any stable-phase checkpoint.
    No re-warming disruption since stable phase is at full LR.
    """

    def __init__(self, optimizer, total_epochs: int, peak_lr: float,
                 warmup_frac: float = 0.05, decay_frac: float = 0.10,
                 min_lr: float = 1e-6, warmup_kind: str = "cosine"):
        if warmup_kind not in ("cosine", "linear"):
            raise ValueError(f"warmup_kind must be 'cosine' or 'linear', got {warmup_kind!r}")
        self.warmup_kind = warmup_kind
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_epochs = max(1, int(total_epochs * warmup_frac))

        # decay_frac=0 → infinite stable phase (no automatic decay)
        if decay_frac <= 0:
            self.decay_epochs = 0
            self.decay_start = total_epochs + 1  # never reached
            self._infinite = True
        else:
            self.decay_epochs = max(1, int(total_epochs * decay_frac))
            self.decay_start = total_epochs - self.decay_epochs
            self._infinite = False

        self.stable_start = self.warmup_epochs
        self._last_lr = [peak_lr] * len(optimizer.param_groups)
        self.epoch = 0
        self._decay_triggered = False
        self._decay_trigger_epoch = None

    def trigger_decay(self, n_epochs: int = 40):
        """Manually trigger the cosine decay phase.

        Call this when you want to finalize the model. The decay
        starts at the current epoch and runs for n_epochs.
        """
        self._decay_triggered = True
        self._decay_trigger_epoch = self.epoch
        self.decay_epochs = n_epochs
        self.decay_start = self.epoch
        print(f"[WSD] Decay triggered at epoch {self.epoch}, "
              f"will decay over {n_epochs} epochs to lr={self.min_lr:.1e}")

    def step(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            # progress in [0, 1] across warmup window
            p = self.epoch / max(self.warmup_epochs, 1)
            if self.warmup_kind == "cosine":
                # Half-cosine ramp: 0 → peak via 0.5*(1 - cos(pi*p)).
                # Same start/end values as linear but smoother derivative
                # at both edges (avoids the optimizer-state shock that
                # a sharp linear corner can trigger right at peak LR).
                lr = self.peak_lr * 0.5 * (1.0 - math.cos(math.pi * p))
            else:  # "linear"
                lr = self.peak_lr * p
        elif not self._decay_triggered and self._infinite:
            # Infinite stable — runs forever at peak LR
            lr = self.peak_lr
        elif self.epoch < self.decay_start:
            lr = self.peak_lr
        else:
            # Cosine decay
            progress = (self.epoch - self.decay_start) / max(self.decay_epochs, 1)
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        self._last_lr = []
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
            self._last_lr.append(lr)

    def get_last_lr(self):
        return self._last_lr

    @property
    def phase(self) -> str:
        if self.epoch <= self.warmup_epochs:
            return 'warmup'
        elif not self._decay_triggered and self._infinite:
            return 'stable∞'
        elif self.epoch < self.decay_start:
            return 'stable'
        else:
            return 'decay'


# ============================================================
# Channel-agnostic input transform (CA-6)
# ============================================================

def _ca_inputs(x_l3, fullband, channel_agnostic, variable_n, n_range=(8, 21)):
    """Map a 21-ch batch to channel-agnostic (l3, fullband, coords, ch_mask).

    - not channel_agnostic           -> (x_l3, fullband, None, None) [legacy].
    - channel_agnostic, not variable -> (x_l3, fullband, None, None); the model
      defaults coords to canonical-21 at N=21 (warm-start parity path).
    - channel_agnostic + variable_n  -> per-batch uniform k in [n_min,n_max],
      per-SAMPLE random channel subset. l3, fullband AND coords are gathered
      with the SAME index per sample (alignment invariant), so output channel i
      of all three refers to the same electrode. Uniform k => no padding =>
      ch_mask=None (all real). Returns (l3_sub, fb_sub, coords[B,k,3], None).
    """
    if not channel_agnostic or not variable_n:
        return x_l3, fullband, None, None
    from lamquant_neural.positions import canonical_21_coords
    B, Nfull, T = x_l3.shape
    lo, hi = n_range
    k = int(torch.randint(lo, min(hi, Nfull) + 1, (1,)).item())
    # per-sample random permutation -> first k indices  [B,k]
    idx = torch.argsort(torch.rand(B, Nfull, device=x_l3.device), dim=1)[:, :k]
    x_sub = torch.gather(x_l3, 1, idx.unsqueeze(-1).expand(B, k, T))
    fb_sub = (torch.gather(fullband, 1,
                           idx.unsqueeze(-1).expand(B, k, fullband.shape[-1]))
              if fullband is not None else None)
    coords_full = torch.as_tensor(canonical_21_coords(),
                                  dtype=x_l3.dtype, device=x_l3.device)  # [Nfull,3]
    coords = coords_full[idx]                                           # [B,k,3] SAME idx
    return x_sub, fb_sub, coords, None


# ============================================================
# Main training loop
# ============================================================

def run(cfg, vocos_tier: int = 3, ckpt_dir: Optional[str] = None,
        seed: int = 0, fullband_mode: str = 'auto',
        amp: bool = True, compile_decoder: bool = True,
        asymmetric_weight: float = 0.0,
        asymmetric_kind: str = 'envelope',
        augment: str = 'moderate',
        ema: bool = True, ema_decay: float = 0.999,
        gan: bool = True, gan_weight: float = 1.0,
        feat_match_weight: float = 2.0,
        seizure_head: bool = True, seizure_weight: float = 0.1,
        encoder_init: str = None,
        clinical_sampling: bool = True,
        lr_schedule: str = 'soap',
        decay_frac: float = 0.10,
        infinite_lr: bool = False,
        int8_bridge: bool = False,
        resume: str = None,
        lma_root: Optional[str] = None,
        split_manifest: Optional[str] = None,
        detail_bands: str = 'none',
        detail_stack_mode: str = 'interp',
        max_windows_per_file: Optional[int] = None,
        soap_max_precond_dim: int = 10000,
        channel_agnostic: bool = False,
        variable_n: bool = False,
        ca_decoder_legacy: bool = False,
        n_range: tuple = (8, 21),
        diagnostics: bool = True,
        logger_backend: str = 'none'):
    """Run joint training with the given TrainingConfig.

    Speedup knobs:

      amp (default True):
        Wrap forward+backward in `torch.amp.autocast(bfloat16)`. The
        decoder is BF16-stable on Hopper/Ada (4090/H100); ConvNeXt
        blocks + iSTFT both run cleanly. The encoder's LSQ alphas stay
        FP32 by virtue of the autocast policy (alpha is a learnable
        scalar applied via mul, which autocast leaves at its declared
        dtype). Spectral loss STFTs are computed with the BF16 input
        cast back to float, then promoted internally — numerically
        equivalent to FP32 since the STFT op promotes by default.
        Result: ~1.8× faster decoder fwd+bwd, no quality compromise.

      compile_decoder (default True):
        torch.compile(decoder, mode='reduce-overhead'). One-time
        ~30-second compile, then fused kernels. ~1.3-1.5× faster
        decoder forward. The encoder is intentionally not compiled —
        ternary STE + LSQ alpha trip up the FX tracer in some PyTorch
        versions, and the encoder is tiny (435K params) so the win
        would be marginal.
    """
    _run_start_t = time.time()  # for wall-time tracking in experiment_log
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Joint training on {device}")
    print(f"[*] Preset: {cfg.name}  (epochs warm/QAT/fine = "
          f"{cfg.epochs_warmup}/{cfg.epochs_quant}/{cfg.epochs_fine})")
    # Clamp val_interval so validation fires at least ~twice per phase even on
    # short runs. Default 10 with epochs_quant=4 means QAT validation NEVER runs
    # -> no best-checkpoint tracking (best_val_r stays -inf). No-op for long runs.
    _vi = max(1, min(cfg.val_interval, max(1, cfg.epochs_warmup), max(1, cfg.epochs_quant)))
    if _vi != cfg.val_interval:
        print(f"[*] val_interval {cfg.val_interval} -> {_vi} (short-run clamp so QAT validates)")
        object.__setattr__(cfg, 'val_interval', _vi)   # cfg is a frozen dataclass
    print(f"[*] Decoder tier: {vocos_tier}  (8=mobile-200M, 2=clinical-400M, 3=research-800M)")

    ckpt_dir = Path(ckpt_dir or os.path.join(ROOT_DIR, 'lamquant', 'student'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    enc_path = ckpt_dir / f'student_encoder_joint_{cfg.name}.ckpt'
    dec_path = ckpt_dir / f'decoder_tier{vocos_tier}_joint_{cfg.name}.ckpt'

    # ---- Build model ----
    # Enable gradient checkpointing for large decoders (Tier 5+) to fit in 24 GB
    # --- input bands (l3_detail vs full experiment) -----------------------
    # Encoder input = L3 (21ch) + optional detail bands. The dataset stacks
    # them per SNN_DETAIL_BANDS; the decoder ALWAYS reconstructs the 21-ch
    # fullband target. Set the env BEFORE the dataset is built below.
    _BANDS = {'none': '', 'l3_detail': 'l3_detail',
              'all': 'l3_detail,l2_detail,l1_detail'}
    if detail_bands not in _BANDS:
        raise ValueError(f"--detail-bands must be one of {list(_BANDS)}, got {detail_bands!r}")
    if detail_stack_mode not in ('interp', 'fold'):
        raise ValueError(f"--detail-stack-mode must be 'interp' or 'fold', got {detail_stack_mode!r}")
    os.environ['SNN_DETAIL_BANDS'] = _BANDS[detail_bands]
    os.environ['SNN_DETAIL_STACK_MODE'] = detail_stack_mode
    # n_in must track _stack_detail_bands exactly: interp adds 1 block/band,
    # fold adds ceil(len/313) blocks/band (information-preserving, ADR 0031).
    from lamquant.snn.lma_dataset import detail_stack_in_channels
    _bands_list = [b for b in _BANDS[detail_bands].split(',') if b]
    n_in = detail_stack_in_channels(_bands_list, mode=detail_stack_mode)
    print(f"[*] Input bands: {detail_bands} (stack={detail_stack_mode})  -> "
          f"encoder in_channels={n_in}, decoder out_channels=21 (fullband)")

    use_grad_ckpt = vocos_tier >= 5
    encoder_kernels = tuple(int(k) for k in cfg.encoder_kernels.split(','))
    # Channel-agnostic guards (CA-6). variable-N changes the channel count, so
    # it is incompatible with the 21-ch-assuming augmentor / GAN disc / seizure
    # head; the legacy decoder head is fixed-21ch so cannot run variable-N.
    if channel_agnostic and n_in != 21:
        raise ValueError(
            f"--channel-agnostic needs raw 21-ch L3 input (n_in={n_in}); the CA "
            "front-end consumes per-electrode L3, so detail-band channel stacking "
            "is incompatible (CA detail-conditioning is a follow-on). Use --detail-bands none.")
    if ca_decoder_legacy and not channel_agnostic:
        raise ValueError("--ca-decoder-legacy requires --channel-agnostic")
    if variable_n:
        if not channel_agnostic:
            raise ValueError("--variable-n requires --channel-agnostic")
        if ca_decoder_legacy:
            raise ValueError("--variable-n needs the CA decoder head (drop --ca-decoder-legacy)")
        if augment not in (None, 'none'):
            raise ValueError("--variable-n is incompatible with augmentation (assumes 21ch); use --augment none")
        if gan:
            raise ValueError("--variable-n is incompatible with the GAN disc (assumes 21ch); use --no-gan")
        if seizure_head:
            raise ValueError("--variable-n is incompatible with the seizure head (assumes 21ch); use --no-seizure-head")
        # Validate the subset bounds here (clean argparse-time-style failure)
        # rather than letting torch.randint raise a cryptic error on the first
        # batch. channel_agnostic already pins n_in==21, so Nfull is always 21.
        _lo, _hi = n_range
        if not (1 <= _lo <= _hi <= 21):
            raise ValueError(
                f"--n-min/--n-max must satisfy 1 <= n_min <= n_max <= 21, got ({_lo},{_hi})")
    codec = build_default_joint(latent_dim=32, encoder_width=cfg.encoder_width,
                                 vocos_tier=vocos_tier, in_channels=n_in,
                                 decoder_channels=21,
                                 gradient_checkpointing=use_grad_ckpt,
                                 encoder_blocks=cfg.encoder_blocks,
                                 encoder_kernels=encoder_kernels,
                                 channel_agnostic=channel_agnostic,
                                 ca_decoder=(channel_agnostic and not ca_decoder_legacy)).to(device)
    if channel_agnostic:
        _dec_kind = 'legacy-21ch' if ca_decoder_legacy else 'position-conditioned'
        print(f"[*] channel-agnostic: encoder=CA front-end, decoder={_dec_kind}, "
              f"variable_n={variable_n} (N∈{n_range})" if variable_n else
              f"[*] channel-agnostic: encoder=CA front-end, decoder={_dec_kind}, N=21 (parity)")
    # Optionally init encoder from pretrained weights (MAE, prior run, etc.)
    if encoder_init is not None:
        enc_state = _safe_load(encoder_init, map_location=device)
        if isinstance(enc_state, dict) and 'state_dict' in enc_state:
            enc_state = enc_state['state_dict']
        codec.encoder.load_state_dict(enc_state, strict=False)
        print(f"[*] Encoder init: loaded from {encoder_init}")
    n_enc = sum(p.numel() for p in codec.encoder.parameters())
    n_dec = sum(p.numel() for p in codec.decoder.parameters())
    print(f"[*] Encoder params: {n_enc:>12,}  (ternary, MCU)")
    print(f"[*] Decoder params: {n_dec:>12,}  (fp32, base station)")

    # ---- Data loaders — typed pipeline (manifest_v3 + FileEntry) ----
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant', 'oracle'))
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant'))
    from streaming_dataset import PrecomputedL3Dataset
    from data_types import DatasetManifest, Split

    # Data directory: configurable, defaults to repo's dataset_sim/
    data_dir = cfg.data_dir if cfg.data_dir else os.path.join(
        ROOT_DIR, 'lamquant', 'dataset')
    # The legacy manifest_v3 + FileEntry path feeds ONLY the PrecomputedL3Dataset
    # branch below. Skip it entirely for LMA-direct — manifest_v3.json and the
    # Q31 NPZ corpus it indexes were deleted (q31_lml), so loading it crashes.
    train_entries = val_entries = None
    manifest = None
    manifest_path = str(split_manifest) if lma_root is not None else None
    if lma_root is None:
        manifest_path = os.path.join(data_dir, 'manifest_v3.json')
        manifest = DatasetManifest.load(manifest_path)   # validate() runs here
        print(f"[*] Loaded manifest_v3: {manifest.train_files:,} train files, "
              f"{manifest.val_files:,} val files "
              f"({manifest.val_windows:,} val windows across "
              f"{len(manifest.datasets)} datasets)")

        train_entries = manifest.get_file_entries(Split.TRAIN)
        val_entries = manifest.get_file_entries(Split.VAL)

        # Cap files when max_windows is set. Stratified by dataset so the
        # subset is representative (alphabetical order would bias toward
        # whichever dataset sorts first — CHB-MIT before TUH).
        if cfg.max_windows is not None:
            import random as _rng
            _rng.Random(cfg.seed).shuffle(train_entries)
            max_files = max(50, cfg.max_windows // 800)
            train_entries = train_entries[:max_files]
        val_entries = val_entries[:20]

        print(f"[*] Train files: {len(train_entries)}  Val files: {len(val_entries)}")

    # Tier 3+ decoders use iSTFT and emit fullband [B, 21, 2500] directly.
    # Loading the raw fullband target lets joint_loss compare against the
    # actual product metric instead of the L3 proxy, closing the 0.07-R
    # L3↔fullband gap observed on the Tier 2 baseline. Tier 1-2 stay on
    # the L3 target (their decoder output is L3-scale anyway).
    #
    # Fullband mode resolution:
    #   off                 — no fullband target (fall back to L3 loss)
    #   ram                 — load all fullband windows into RAM (~136 GB
    #                          at full scale; only viable for fast preset)
    #   memmap              — mmap a precomputed flat .dat (the production
    #                          path; no RAM cost, page cache absorbs reads)
    #   auto (default)      — ram for fast, memmap for standard/production,
    #                          off if tier < 3 (L3-scale decoder)
    use_fullband_mode = fullband_mode
    if use_fullband_mode == 'auto':
        if vocos_tier < 3:
            use_fullband_mode = 'off'
        elif cfg.name in ('fast',):
            use_fullband_mode = 'ram'
        else:
            use_fullband_mode = 'memmap'

    fb_train_ram = use_fullband_mode == 'ram'
    fb_val_ram = use_fullband_mode == 'ram'
    fb_train_mm = None
    fb_val_mm = None
    if use_fullband_mode == 'memmap':
        fb_train_mm = os.path.join(data_dir, 'fullband_train.dat')
        fb_val_mm = os.path.join(data_dir, 'fullband_val.dat')
        for p in (fb_train_mm, fb_val_mm):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"fullband memmap missing: {p}\n"
                    f"  Run: python ai_models/dataset_sim/precompute_fullband_memmap.py"
                )
    if use_fullband_mode != 'off':
        print(f"[*] Tier {vocos_tier} decoder outputs fullband — joint_loss "
              f"will optimise the product metric (mode={use_fullband_mode})")

    # LMA-direct path (BLUT canonical, ADR 0017). When both lma_root and
    # split_manifest are supplied, use the neural-side LmaTypedL3Dataset
    # adapter which decodes per-batch from per-recording .lma archives —
    # no NPZ precompute, no fullband memmap. Falls through to the
    # deprecated PrecomputedL3Dataset path otherwise so existing
    # experiments stay reproducible.
    if lma_root is not None and split_manifest is not None:
        # Neural-side typed-batch adapter (NOT the canonical codec
        # LmaL3Dataset, which is a bare map-style Dataset lacking the
        # streaming surface — calibrate_shard_budget / prefetch_typed_batches
        # / real seizure labels — that this loop requires). The adapter
        # wraps the seizure-aware lamquant.snn.lma_dataset.LmaDataset and
        # exposes that surface. See ai_models/student/lma_typed_adapter.py.
        from lma_typed_adapter import LmaTypedL3Dataset
        want_fullband = (use_fullband_mode != 'off')
        print(f"[*] LMA-direct (typed adapter): root={lma_root}, "
              f"manifest={split_manifest}, return_fullband={want_fullband}")
        _mwpf = {} if max_windows_per_file is None else {"max_windows_per_file": max_windows_per_file}
        train_ds = LmaTypedL3Dataset(
            lma_root=lma_root,
            split="train",
            split_manifest_path=split_manifest,
            windows_per_epoch=cfg.windows_per_epoch,
            return_fullband=want_fullband,
            seed=seed,
            **_mwpf,
        )
        val_ds = LmaTypedL3Dataset(
            lma_root=lma_root,
            split="val",
            split_manifest_path=split_manifest,
            windows_per_epoch=cfg.val_windows,
            return_fullband=want_fullband,
            seed=seed + 1,
            **_mwpf,
        )
    else:
        train_ds = PrecomputedL3Dataset(
            file_entries=train_entries,
            windows_per_epoch=cfg.windows_per_epoch,
            max_windows=cfg.max_windows,
            with_fullband=fb_train_ram,
            fullband_memmap_path=fb_train_mm,
            train_noise_bits=cfg.train_noise_bits,
        )
        val_ds = PrecomputedL3Dataset(
            file_entries=val_entries, windows_per_epoch=cfg.val_windows,
            with_fullband=fb_val_ram,
            fullband_memmap_path=fb_val_mm,
            train_noise_bits=cfg.train_noise_bits,
        )

    # ---- Clinical balanced sampling ----
    train_sampler = None
    if clinical_sampling and train_ds._win_clinical_category is not None:
        from clinical_sampler import ClinicalWeightedSampler
        train_sampler = ClinicalWeightedSampler(
            window_categories=train_ds._win_clinical_category,
            num_samples=cfg.windows_per_epoch,
            seed=seed,
        )
        print(f"[*] Clinical sampling: ON")
    else:
        print(f"[*] Clinical sampling: off")

    # ---- Optimizer with two parameter groups (encoder alphas separate) ----
    enc_groups = make_param_groups(
        codec.encoder, lr=cfg.lr_warmup,
        weight_decay=cfg.wd_warmup, alpha_weight_decay=1e-3,
    )
    dec_group = {'params': list(codec.decoder.parameters()),
                 'lr': cfg.lr_warmup, 'weight_decay': cfg.wd_warmup}
    # Fused AdamW collapses the per-parameter Python loop into a single
    # CUDA kernel — small but free win on a tier with hundreds of
    # parameter tensors. Falls back to non-fused on CPU automatically.
    optimizer = torch.optim.AdamW(enc_groups + [dec_group],
                                   fused=(device.type == 'cuda'))

    # ---- torch.compile the decoder ----
    # The encoder isn't compiled (tiny, plus its STE/LSQ alpha trip up
    # the FX tracer in some PyTorch versions). The decoder is the 99 %
    # of params so compiling it captures essentially all the benefit.
    # First forward pays a one-time ~30 s compile cost; thereafter the
    # fused kernels are ~1.3-1.5× faster on Tier 5+ on Ada/Hopper.
    if compile_decoder:
        try:
            codec.decoder = torch.compile(codec.decoder, mode='reduce-overhead')
            print(f"[*] Decoder compiled (mode='reduce-overhead')")
        except Exception as e:
            print(f"[!] torch.compile failed ({e}); continuing without compile")

    # ---- Augmentation pipeline ----
    augmentor = None
    if augment != 'none':
        augmentor = EEGAugmentor(mode=augment)
        print(f"[*] Augmentation: {augment} preset")
    else:
        print(f"[*] Augmentation: off")

    # ---- EMA (exponential moving average) ----
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema_model = None
    if ema:
        ema_model = AveragedModel(codec, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        print(f"[*] EMA: decay={ema_decay}")

    # ---- Multi-task seizure detection head ----
    sz_head = None
    if seizure_head:
        sz_head = SeizureHead(latent_dim=32).to(device)
        # Add seizure head params to the generator optimizer so they
        # co-train with the encoder. The head is tiny (33 params).
        optimizer.add_param_group({
            'params': list(sz_head.parameters()),
            'lr': cfg.lr_quant, 'weight_decay': 0.0,
        })
        print(f"[*] Seizure head: ON (weight={seizure_weight}, "
              f"{sum(p.numel() for p in sz_head.parameters())} params)")

    # ---- GAN discriminator (adversarial training) ----
    # MPD + MS-STFT discriminator adapted for EEG (already built in
    # ai_models/decoder/discriminator.py). Only active when --gan is
    # passed AND fullband target is available (Tier 3+ iSTFT output).
    disc = None
    disc_optimizer = None
    use_gan = gan and (use_fullband_mode != 'off')
    if use_gan:
        disc = EEGDiscriminator().to(device)
        disc_optimizer = torch.optim.AdamW(
            disc.parameters(), lr=cfg.lr_quant * 0.5,  # D lr = 0.5× G lr
            weight_decay=cfg.wd_quant, fused=(device.type == 'cuda'))
        n_disc = sum(p.numel() for p in disc.parameters())
        # GAN doubles memory (real + fake forward, R1 grad) — auto-cap batch
        # to prevent OOM on Tier 3+ at the fast preset's default batch=128.
        if cfg.batch_size_quant > 16:
            cfg = cfg.replace(batch_size_quant=16, batch_size_warmup=16)
            print(f"[*] GAN: auto-reduced batch to 16 (discriminator VRAM)")
        print(f"[*] GAN: discriminator {n_disc:,} params, "
              f"adv_weight={gan_weight}, feat_match={feat_match_weight}")
    else:
        print(f"[*] GAN: {'off (no fullband)' if gan else 'off'}")

    # ---- Loss ----
    # MSE preserves magnitude; (1-R) preserves shape; PRD makes
    # magnitude preservation explicit; spectral preserves frequency
    # content.
    #
    # IMPORTANT — target selection:
    # joint_loss prefers `fullband_target` (Tier 3+ decoder output is
    # 2500 samples = fullband EEG). This is the publishable metric:
    # the loss directly optimises what eval_fullband.py later measures
    # for LQS compliance. Without it, the loss optimised L3 (313
    # samples) and the L3↔fullband gap (~0.07 R observed on Tier 2)
    # was opaque to the optimiser.
    #
    # Falls back to L3 target when:
    #   - fullband_target is None (dataset built without with_fullband)
    #   - decoder output is L3-scale (Tier 1-2, 'direct' output mode)
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant'))
    from metrics import (prd_torch, pearson_r_torch,
                          masked_pearson_r_torch, masked_prd_torch,
                          asymmetric_eeg_loss as _asym_env,
                          band_aware_asymmetric_loss as _asym_band)
    spectral_loss = make_spectral_loss(device)
    R_W = cfg.pearson_r_weight
    SP_W = cfg.spectral_weight
    PRD_W = cfg.prd_weight
    ASYM_W = float(asymmetric_weight)
    if ASYM_W > 0:
        asym_fn = _asym_band if asymmetric_kind == 'band' else _asym_env
        print(f"[*] Asymmetric loss: {asymmetric_kind}, weight={ASYM_W}")

    def joint_loss(recon, l3_target, fullband_target=None,
                    ch_mask=None, return_parts: bool = True):
        # ch_mask [B,N] (channel-agnostic padded batches): excludes padded
        # channels from the R/PRD terms. None (the default + the variable-N
        # uniform-k path, which never pads) == the legacy unmasked behavior.
        # Decide which target the decoder output matches in length.
        # Tier 3+ → recon.shape[-1] ≈ 2500; Tier 1-2 → ≈ 313.
        if fullband_target is not None and abs(
                recon.shape[-1] - fullband_target.shape[-1]) <= 8:
            target = fullband_target
            domain = 'fullband'
        else:
            target = l3_target
            domain = 'l3'
        T = min(recon.shape[-1], target.shape[-1])
        recon_c = recon[..., :T]
        target_c = target[..., :T]
        l_mse = F.mse_loss(recon_c, target_c)
        l_r = 1.0 - masked_pearson_r_torch(recon_c, target_c, ch_mask)  # 1 − R loss (differentiable)
        # PRD/100 lands in [0, 1]ish so the weight is comparable to
        # the other terms. Don't divide inside prd_torch — keep it as
        # a percentage at the metric level.
        l_prd = masked_prd_torch(target_c, recon_c, ch_mask) / 100.0 if PRD_W > 0 else 0.0
        # Asymmetric / clinically-weighted MSE — only active when
        # asymmetric_weight > 0. Operates on the SAME (target, recon)
        # pair as MSE, just with a per-sample weight derived from the
        # original signal's amplitude envelope.
        l_asym = asym_fn(target_c, recon_c) if ASYM_W > 0 else 0.0
        # Spectral loss STFTs are nightly fragile in BF16; force FP32
        # for the spectral term specifically. Cheap (one cast).
        if SP_W > 0:
            with torch.amp.autocast(device_type=device.type, enabled=False):
                l_sp = spectral_loss(recon_c.float(), target_c.float())
        else:
            l_sp = 0.0
        total = l_mse + R_W * l_r + PRD_W * l_prd + SP_W * l_sp + ASYM_W * l_asym
        # Detach before scalar conversion — these dict entries are diagnostic
        # only, not part of the autograd graph. Without .detach() torch warns
        # about converting requires_grad tensors directly to floats and (more
        # importantly) keeps the parts dict pinning the autograd graph alive
        # until the next loss.backward(), wasting memory.
        # `return_parts=False` skips the dict construction entirely — the
        # train loop only needs scalars at val_interval boundaries, so the
        # other ~99 % of batches save a few µs per call (real on tight loops).
        if not return_parts:
            return total, None
        return total, {
            'mse': l_mse.detach().item(),
            'r_loss': l_r.detach().item(),
            'prd_loss': (l_prd.detach().item()
                         if isinstance(l_prd, torch.Tensor) else l_prd),
            'spectral': (l_sp.detach().item()
                         if isinstance(l_sp, torch.Tensor) else l_sp),
            'asym': (l_asym.detach().item()
                     if isinstance(l_asym, torch.Tensor) else l_asym),
            'loss_domain': domain,
        }

    # ---- Gradient health check (runs once on first batch) ----
    _grad_checked = [False]

    def _gradient_health_check(loss_total, loss_parts, model):
        """Verify every loss term produces non-zero gradients.

        Called once on the first training batch. Catches silent gradient
        bugs (detached tensors, broken autograd, wrong dtype) before
        wasting hours of GPU time.
        """
        if _grad_checked[0] or loss_parts is None:
            return
        _grad_checked[0] = True
        print('\n[*] Gradient health check (first batch):')
        # Check total loss has grad_fn
        if loss_total.grad_fn is None:
            print('  [FAIL] total loss has no grad_fn!')
            return
        # Check individual terms
        for name, val in loss_parts.items():
            if name == 'loss_domain':
                continue
            if isinstance(val, (int, float)):
                if val == 0.0:
                    print(f'  [skip] {name}: disabled (0.0)')
                else:
                    print(f'  [WARN] {name}: Python scalar {val} (no gradient)')
        # Grad-norm check reads the grads from the caller's SINGLE backward
        # (this is now called AFTER loss.backward(), before clip/step). NO second
        # backward here: a retain_graph=True double-backward of the same loss is
        # incompatible with torch.compile(reduce-overhead) donated buffers and was
        # crashing the run (task #255). Do NOT zero_grad — the caller's
        # clip_grad_norm_ + optimizer.step() need these grads.
        enc_gnorm = torch.sqrt(sum(p.grad.norm() ** 2
                        for p in model.encoder.parameters()
                        if p.grad is not None)).item()  # single sync
        dec_gnorm = torch.sqrt(sum(p.grad.norm() ** 2
                        for p in model.decoder.parameters()
                        if p.grad is not None)).item()  # single sync
        print(f'  encoder grad_norm: {enc_gnorm:.4f}')
        print(f'  decoder grad_norm: {dec_gnorm:.4f}')
        if enc_gnorm < 1e-8:
            print('  [FAIL] encoder receives NO gradient!')
        if dec_gnorm < 1e-8:
            print('  [FAIL] decoder receives NO gradient!')
        print()

    # ---- Provenance — embedded into every checkpoint save (refactor #72/74) ----
    # Two hashes pin the checkpoint's full identity:
    #   manifest_hash  — what DATA was used (sha256 of manifest content)
    #   training_config_hash — what RECIPE was used (sha256 of the
    #                          TrainingConfig dataclass content)
    # Together they answer "what config + what data produced these
    # weights?" with no chat-history archaeology required.
    #
    # CLI flags that override TrainingConfig defaults are reflected back
    # into the cfg object below so the hash captures the actual values
    # used (not the unmodified preset defaults).
    cfg_for_provenance = cfg.replace(
        vocos_tier=vocos_tier, seed=seed,
        precision=('bf16' if amp else 'fp32'),
        compile_decoder=bool(compile_decoder),
        fullband_mode=use_fullband_mode,
        asymmetric_weight=ASYM_W,
        asymmetric_kind=asymmetric_kind,
    )
    run_id = f'joint_{cfg.name}_t{vocos_tier}_{int(time.time())}'
    provenance = {
        'manifest_hash':         manifest.hash() if manifest is not None else f'lma-direct:{manifest_path}',
        'manifest_path':         manifest_path,
        'manifest_version':      manifest.version if manifest is not None else 'lma-direct',
        'training_config_hash':  cfg_for_provenance.hash(),
        'training_config':       cfg_for_provenance.to_dict(),
        'run_id':                run_id,
    }
    print(f"[*] Run ID:                {run_id}")
    print(f"[*] Manifest hash:         {provenance['manifest_hash']}")
    print(f"[*] Training-config hash:  {provenance['training_config_hash']}")

    # Unified logger — single source of truth for all output formats.
    # Writes per-epoch CSV + per-layer alpha CSV + terminal dashboard.
    log_dir = Path(ROOT_DIR) / 'training_logs'
    logger = TrainingLogger(run_id=run_id, log_dir=log_dir)
    print(f"[*] Epoch log:             {logger.epoch_csv}")

    # Reviewer-readable metric stream (ALWAYS on; Parquet via pyarrow, CSV
    # fallback) — a complete valid file after every epoch (read mid-run). Plus
    # optional Weights & Biases under --logger wandb (offline by default).
    from blut_core.metric_log import MetricLog
    metric_log = MetricLog(run_id=run_id, log_dir=log_dir)
    print(f"[*] Metric stream:         {metric_log.path}  (backend={metric_log._backend})")
    wandb_run = None
    if logger_backend == 'wandb':
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ.get('WANDB_PROJECT', 'lamquant'),
                name=run_id,
                config=provenance['training_config'],
                tags=['joint', 'student', f'tier{vocos_tier}', cfg.name],
                mode=os.environ.get('WANDB_MODE', 'offline'),
                dir=str(log_dir))
            print(f"[*] wandb:                 mode={os.environ.get('WANDB_MODE', 'offline')} "
                  f"project={os.environ.get('WANDB_PROJECT', 'lamquant')}")
        except Exception as e:
            print(f"[!] --logger wandb requested but wandb unavailable ({e}); continuing without it")
            wandb_run = None

    def _emit(report):
        """Log one epoch to the TrainingLogger + the reviewer metric stream +
        (optional) wandb. Each sink is independently guarded — logging must
        never crash a run, and one sink failing must not starve the others."""
        try:
            logger.log_epoch(report)
        except Exception as e:
            print(f"[!] epoch logger failed (non-fatal): {e}")
        try:
            d = report.to_dict()
            d.pop('alpha_per_layer', None)   # mirror TrainingLogger's CSV exclude
            metric_log.append(d)
            if wandb_run is not None:
                scalars = {k: v for k, v in d.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
                wandb_run.log(scalars, step=report.global_epoch)
        except Exception as e:
            print(f"[!] metric stream/wandb emit failed (non-fatal): {e}")

    dash = TrainingDashboard(
        model_name='LamQuant Joint',
        gen='7.7',
        preset=cfg.name,
        total_epochs=cfg.epochs_warmup + cfg.epochs_quant,
        device=device,
        emit_interval=50,
    )

    # ---- CheckpointManager — owns saves + safety ----
    alpha_csv = os.path.join(
        ROOT_DIR, 'training_logs',
        f'alpha_trajectory_{run_id}.csv',
    )
    cm = CheckpointManager(
        model=codec.encoder,        # alpha tracking is encoder-only
        ckpt_path=enc_path,         # CM saves encoder; we save decoder manually
        ckpt_dir=ckpt_dir / 'recovery',
        device=device,
        provenance=provenance,
        smoke_input=lambda: torch.randn(1, n_in, 313, device=device),
        alpha_log_csv=alpha_csv,
        guard=GuardConfig(
            r_plateau_patience=10**6,    # Joint is exploratory; main loop manages
            alpha_max_safe=10**6,         # stop. CM is here for the CSV + saves.
            alpha_min_safe=0,
            smoke_check_tolerance=0.5,
        ),
    )
    print(f"[*] Alpha trajectory CSV: {alpha_csv}")

    # ---- Resume from checkpoint ----
    _resume_epoch = 0
    _resume_phase = None
    if resume:
        _rec_dir = ckpt_dir / 'recovery'
        if resume == 'auto':
            # Auto-detect: prefer qat_latest > warm_latest
            for candidate in ('qat_latest.ckpt', 'warm_latest.ckpt'):
                p = _rec_dir / candidate
                if p.exists():
                    resume = str(p)
                    break
        if resume and resume != 'auto' and os.path.exists(resume):
            print(f"[*] Resuming from {resume}")
            ckpt = _safe_load(resume, map_location=device)
            # Strip _orig_mod. prefix if checkpoint was saved after torch.compile
            def _strip_compile_prefix(sd):
                return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
            enc_sd = ckpt['encoder']
            dec_sd = ckpt['decoder']
            if any(k.startswith('_orig_mod.') for k in enc_sd):
                enc_sd = _strip_compile_prefix(enc_sd)
            if any(k.startswith('_orig_mod.') for k in dec_sd):
                dec_sd = _strip_compile_prefix(dec_sd)
            codec.encoder.load_state_dict(enc_sd)
            _dec_target = getattr(codec.decoder, '_orig_mod', codec.decoder)
            _dec_target.load_state_dict(dec_sd)
            _resume_epoch = ckpt['epoch']
            _resume_phase = ckpt['phase']
            print(f"[*] Restored {_resume_phase} phase, epoch {_resume_epoch}")
        else:
            print(f"[!] Resume checkpoint not found, starting fresh")

    # ---- Phase 1: Warm (encoder+decoder FP32, no STE on encoder) ----
    # Lazy import of Split here so the script doesn't grow a top-level
    # data_types dependency at module-import time (keeps CLI --help fast).
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant'))
    from data_types import Split

    print(f"\n[*] Phase 1: WARM ({cfg.epochs_warmup} ep, encoder FP32, decoder FP32"
          + (', BF16 autocast' if amp else '') + ')')
    # AMP context — BF16 autocast on Ada/Hopper. Encoder lsq_alpha + STFT
    # losses carve back to FP32 inside the relevant ops. Disabled at zero
    # cost when amp=False.
    amp_ctx = (torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
               if amp else _nullctx())
    # Calibrate shard budget AFTER a warmup forward+backward so the VRAM
    # measurement accounts for model weights + optimizer state + activation
    # memory. Without this, the calibration measures free VRAM before the
    # model runs, then the first real shard OOMs.
    with torch.no_grad():
        _dummy = codec(torch.randn(cfg.batch_size_warmup, n_in, 313, device=device),
                        quantize=False)
        del _dummy
    torch.cuda.empty_cache()
    train_ds.calibrate_shard_budget(device)
    val_ds.calibrate_shard_budget(device)

    # ---- Pre-flight diagnostics gate — catch broken wiring (dead-grad term,
    # shape contract, NaN data, coords misrouting) BEFORE the GPU run burns
    # hours. Halts only on hard structural FAIL (data/shape/grad); never blocks
    # a run on a diagnostics-internal error. Disable with --no-diagnostics.
    if diagnostics:
        try:
            from training_diagnostics import TrainingDiagnostics, DiagReport
            _pf = next(iter(train_ds.prefetch_typed_batches(
                batch_size=min(4, cfg.batch_size_warmup), device=device,
                sampler=train_sampler)))
            _xin, _fbt, _co, _cm = _ca_inputs(
                _pf.l3_approx, _pf.fullband_target, channel_agnostic, variable_n, n_range)
            _diag = TrainingDiagnostics(
                codec,
                loss_fn=lambda r, l, fullband=None, ch_mask=None: joint_loss(
                    r, l, fullband_target=fullband, ch_mask=ch_mask, return_parts=False),
                channel_agnostic=channel_agnostic, device=str(device))
            _rep = DiagReport()
            _rep.add(_diag.check_data_sanity(_xin, _fbt))
            _rep.add(_diag.check_shape_contract(_xin, _co, _cm, fullband=_fbt))
            _rep.add(_diag.check_gradient_flow(_xin, _fbt, _co, _cm))
            _rep.add(_diag.check_masked_invariant())
            if channel_agnostic and _co is not None:
                _rep.add(_diag.check_coords_routing(_xin, _co, _cm))
            print("\n[*] PRE-FLIGHT DIAGNOSTICS\n" + _rep.summary() + "\n")
            _hard = [r for r in _rep.failed
                     if r.name.startswith(("data.", "shape.", "grad."))]
            if _hard:
                raise RuntimeError(
                    "pre-flight diagnostics FAILED (broken wiring — fix before "
                    "training): " + ", ".join(r.name for r in _hard))
            codec.zero_grad(set_to_none=True)  # clear preflight grads
        except RuntimeError:
            raise
        except Exception as _e:  # diagnostics must never block a run on its own bug
            print(f"[!] pre-flight diagnostics skipped (non-fatal): {_e}")

    best_warm_r = 0.0
    _n_batches_warm = max(cfg.windows_per_epoch // max(cfg.batch_size_warmup, 1), 1)
    _warm_start = _resume_epoch + 1 if _resume_phase == 'warm' else 1
    if _resume_phase == 'qat':
        _warm_start = cfg.epochs_warmup + 1  # skip warm entirely
    for ep in range(_warm_start, cfg.epochs_warmup + 1):
        codec.train()
        loss_acc, n = 0.0, 0
        _bi = 0
        for batch in train_ds.prefetch_typed_batches(
                batch_size=cfg.batch_size_warmup, device=device,
                sampler=train_sampler):
            batch.assert_no_leakage(Split.TRAIN)   # safety net
            x_l3 = batch.l3_approx
            # Warm phase: NO augmentation. The encoder learns clean
            # representations. Augmenting here and comparing recon to
            # original creates an impossible objective (invert augmentation).
            need_check = not _grad_checked[0]
            _xin, _fb, _coords, _cmask = _ca_inputs(
                x_l3, batch.fullband_target, channel_agnostic, variable_n, n_range)
            with amp_ctx:
                recon = codec(_xin, quantize=False, coords=_coords, ch_mask=_cmask)
                loss, parts = joint_loss(recon, _xin,
                                          fullband_target=_fb,
                                          ch_mask=_cmask,
                                          return_parts=need_check)
            optimizer.zero_grad()
            loss.backward()
            if need_check:
                # AFTER backward, BEFORE clip/step: reads grad norms from the
                # caller's single backward (no retain_graph double-backward —
                # see #255 / _gradient_health_check). Grads stay live for clip+step.
                _gradient_health_check(loss, parts, codec)
            _gnorm_warm = torch.nn.utils.clip_grad_norm_(codec.parameters(), cfg.grad_clip_warmup)
            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(codec)
            loss_acc += loss.detach(); n += 1  # accumulate on GPU, no sync
            dash.step(phase='WARM', epoch=ep, batch=_bi,
                      n_batches=_n_batches_warm,
                      loss=float(loss.detach()),
                      grad_norm=float(_gnorm_warm))
            _bi += 1

        # Always: structured EpochReport via TrainingLogger every epoch.
        avg_loss = float(loss_acc / max(n, 1))  # single GPU→CPU sync per epoch
        _elapsed = time.time() - _run_start_t
        _secs_per_ep = _elapsed / ep
        _total_eps = cfg.epochs_warmup + cfg.epochs_quant
        _eta_h = _secs_per_ep * (_total_eps - ep) / 3600
        vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        _alpha_pl = alpha_stats_from_model(codec.encoder)
        _amin, _amean, _amax = reduce_alpha_stats(_alpha_pl)

        # Run validation only at val_interval; fill R/PRD=0 on other epochs
        # (dashboard shows '-' for 0 values — clearly not a real measurement).
        val_r = val_prd = 0.0
        _saved = False
        if ep % cfg.val_interval == 0:
            val_r, val_prd, _ = validate_joint(codec, val_ds, device,
                                                 quantize=False, amp=amp,
                                                 channel_agnostic=channel_agnostic,
                                                 variable_n=variable_n, n_range=n_range)
            dash.update_val(val_r=val_r, best_r=max(best_warm_r, val_r))
            if val_r > best_warm_r:
                best_warm_r = val_r
                _saved = True
                codec.save_encoder(
                    ckpt_dir / f'student_encoder_warm_{cfg.name}.ckpt',
                    provenance={**provenance, 'phase': 'warm', 'epoch': ep})
                codec.save_decoder(
                    ckpt_dir / f'decoder_warm_{cfg.name}.ckpt',
                    provenance={**provenance, 'phase': 'warm', 'epoch': ep})

        _emit(EpochReport(
            run_id=run_id, script='train_joint', phase='warm',
            epoch=ep, global_epoch=ep, total_epochs=_total_eps,
            train_loss=avg_loss, val_r=val_r, val_prd=val_prd,
            best_val_r=best_warm_r,
            encoder_params=n_enc, decoder_params=n_dec,
            decoder_tier=str(vocos_tier),
            lr=optimizer.param_groups[0]['lr'],
            vram_gb=vram_gb,
            saved_checkpoint=_saved,
            alpha_per_layer=_alpha_pl,
            alpha_min=_amin, alpha_mean=_amean, alpha_max=_amax,
            quantize_active=False,
            secs_per_epoch=_secs_per_ep,
            eta_hours=_eta_h,
        ))

        # Rolling recovery: two files per phase — latest (every epoch) and
        # best (only when val_r improves). Crash loses ≤1 epoch; best is
        # always recoverable even if latest is corrupt on a bad shutdown.
        _rec_dir = ckpt_dir / 'recovery'
        _rec_dir.mkdir(parents=True, exist_ok=True)
        _warm_state = {'encoder': codec.encoder.state_dict(),
                       'decoder': getattr(codec.decoder, '_orig_mod', codec.decoder).state_dict(),
                       'optimizer': optimizer.state_dict(),
                       'epoch': ep, 'phase': 'warm',
                       'provenance': provenance}
        torch.save(_warm_state, _rec_dir / 'warm_latest.ckpt')
        if _saved:
            torch.save(_warm_state, _rec_dir / 'warm_best.ckpt')

    print(f"  [WARM] best ValR = {best_warm_r:.4f} (FP32, diagnostic only)")

    # Warm-only runs (epochs_quant == 0 — e.g. the E1 FP32 full-residual
    # probe) never enter the QAT loop where the joint export
    # (enc_path / dec_path) is written, so without this the run reports those
    # paths but never creates them and the BLUT stage fails "expected output
    # missing after success" (ADR 0044). The warm-best IS the final model, so
    # promote it to the joint export names. (For epochs_quant > 0 this is a
    # no-op; QAT writes enc_path/dec_path itself.)
    if cfg.epochs_quant == 0:
        import shutil
        _warm_enc = ckpt_dir / f'student_encoder_warm_{cfg.name}.ckpt'
        _warm_dec = ckpt_dir / f'decoder_warm_{cfg.name}.ckpt'
        if _warm_enc.exists():
            shutil.copy2(_warm_enc, enc_path)
        else:
            codec.save_encoder(enc_path, provenance={**provenance, 'phase': 'warm-final'})
        if _warm_dec.exists():
            shutil.copy2(_warm_dec, dec_path)
        else:
            codec.save_decoder(dec_path, provenance={**provenance, 'phase': 'warm-final'})
        print(f"  [WARM-ONLY] promoted warm best → {enc_path.name} + {dec_path.name}")

    # Seed CheckpointManager with warm-phase best so QAT doesn't
    # overwrite a good warm checkpoint with a worse QAT epoch 1.
    # QAT R values are on a different scale (ternary), but starting
    # from the warm best ensures the first QAT save must actually
    # improve over the warm-phase peak — not just beat -inf.
    # Seed unconditionally (was `if best_warm_r > 0`, which left best_val_r at
    # -inf on short/degenerate warms so QAT best-checkpoint tracking never had a
    # baseline). Even a 0.0 warm best gives QAT a real bar to beat.
    cm.best_val_r = best_warm_r
    cm.best_epoch = cfg.epochs_warmup
    print(f"  [CM] Seeded QAT tracker with warm best R={best_warm_r:.4f}")

    # ---- Phase 2: QAT (encoder ternary, decoder still FP32) ----
    print(f"\n[*] Phase 2: QAT ({cfg.epochs_quant} ep, encoder ternary STE, decoder FP32)")

    # CALIBRATE lsq_alpha at the warm->QAT boundary (the fix for the grad
    # explosion). During WARM the LSQ path is bypassed, so lsq_alpha stays at its
    # 0.1 init; with warm-converged weights (mean|W|~0.056) that sends most
    # weights to 0 at QAT onset -> latent collapse -> exploding decoder grads
    # (the measured 50k-214k / val R 0.01 / PRD 188% failure). _init_alpha sets
    # alpha = (2/3)*mean|W| per channel — the data-driven LSQ-style init that
    # every reference quantizer (LSQ/BitNet/TTQ/ParetoQ) performs and ours
    # uniquely omitted (see docs/QAT_REFERENCE_COMPARISON.md). Force it on every
    # ternary/INT8 module regardless of the per-module init flag.
    _n_cal = 0
    for _m in codec.encoder.modules():
        if hasattr(_m, '_init_alpha') and hasattr(_m, 'clamp_alpha'):
            _m._init_alpha(); _m.clamp_alpha(); _n_cal += 1
    print(f"  [QAT calib] data-driven alpha init on {_n_cal} quantized modules "
          f"(warm->QAT collapse fix)")

    # Reset optimizer with QAT learning rate.
    enc_groups = make_param_groups(
        codec.encoder, lr=cfg.lr_quant,
        weight_decay=cfg.wd_quant, alpha_weight_decay=1e-3,
    )
    dec_group = {'params': list(codec.decoder.parameters()),
                 'lr': cfg.lr_quant, 'weight_decay': cfg.wd_quant}
    # LR schedule selection:
    #   wsd (default): Warmup-Stable-Decay for continual training
    #   schedule-free: Schedule-Free AdamW (Defazio 2024)
    #   cosine: AdamW + cosine annealing
    use_schedule_free = False
    scheduler = None
    if lr_schedule == 'schedule-free':
        try:
            import schedulefree
            all_params = []
            for g in enc_groups + [dec_group]:
                all_params.extend(g['params'])
            optimizer = schedulefree.AdamWScheduleFree(
                all_params, lr=cfg.lr_quant, weight_decay=cfg.wd_quant,
                warmup_steps=100)
            use_schedule_free = True
            print(f"[*] Optimizer: Schedule-Free AdamW (lr={cfg.lr_quant})")
        except ImportError:
            print(f"[!] schedulefree not installed, falling back to WSD")
            lr_schedule = 'wsd'
    actual_decay_frac = 0.0 if infinite_lr else decay_frac
    if lr_schedule == 'wsd':
        optimizer = torch.optim.AdamW(enc_groups + [dec_group],
                                       fused=(device.type == 'cuda'))
        scheduler = WSDScheduler(
            optimizer, total_epochs=cfg.epochs_quant,
            peak_lr=cfg.lr_quant, warmup_frac=0.05,
            decay_frac=actual_decay_frac,
            min_lr=cfg.lr_quant_min)
        if scheduler._infinite:
            print(f"[*] Optimizer: AdamW + WSD∞ (warmup={scheduler.warmup_epochs}ep, "
                  f"stable=∞, decay=manual trigger)")
        else:
            print(f"[*] Optimizer: AdamW + WSD (warmup={scheduler.warmup_epochs}ep, "
                  f"stable={scheduler.decay_start - scheduler.warmup_epochs}ep, "
                  f"decay={scheduler.decay_epochs}ep)")
    elif lr_schedule == 'muon':
        from muon_optimizer import Muon, split_params_for_muon
        muon_p, adamw_p = split_params_for_muon(codec)
        optimizer = Muon([
            dict(params=muon_p, lr=0.02, momentum=0.95, weight_decay=0, use_muon=True),
            dict(params=adamw_p, lr=cfg.lr_quant, betas=(0.95, 0.95), eps=1e-8,
                 weight_decay=cfg.wd_quant, use_muon=False),
        ])
        print(f"[*] Optimizer: Muon (lr=0.02, 2D={len(muon_p)}, 1D={len(adamw_p)})")
    elif lr_schedule == 'soap':
        from soap_optimizer import SOAP
        all_params = []
        for g in enc_groups + [dec_group]:
            all_params.extend(g['params'])
        # max_precond_dim bounds the per-layer full-matrix preconditioner: any
        # tensor dim above it falls back to diagonal (Adam-like). The default
        # 10000 eigendecomposes 10000x10000 GG matrices on the big decoder
        # layers — O(d^3) compute + GBs of eigh workspace — which OOMs a 24 GB
        # card for Tier 5+ decoders at the first precondition step. Bounding to
        # a few thousand keeps full preconditioning where it helps (encoder +
        # small layers) and is strictly faster + lighter on the wide layers.
        optimizer = SOAP(all_params, lr=cfg.lr_quant, weight_decay=cfg.wd_quant,
                         precondition_frequency=10,
                         max_precond_dim=soap_max_precond_dim)
        # Wrap SOAP in WSD for warmup + optional decay
        scheduler = WSDScheduler(
            optimizer, total_epochs=cfg.epochs_quant,
            peak_lr=cfg.lr_quant, warmup_frac=0.05,
            decay_frac=actual_decay_frac,
            min_lr=cfg.lr_quant_min)
        if scheduler._infinite:
            print(f"[*] Optimizer: SOAP + WSD∞ (lr={cfg.lr_quant}, "
                  f"warmup={scheduler.warmup_epochs}ep, stable=∞)")
        else:
            print(f"[*] Optimizer: SOAP + WSD (lr={cfg.lr_quant}, "
                  f"warmup={scheduler.warmup_epochs}ep, "
                  f"decay={scheduler.decay_epochs}ep)")
    elif lr_schedule == 'cosine' and not use_schedule_free:
        optimizer = torch.optim.AdamW(enc_groups + [dec_group],
                                       fused=(device.type == 'cuda'))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs_quant, eta_min=cfg.lr_quant_min)
        print(f"[*] Optimizer: AdamW + cosine LR")

    # Re-add seizure head params to the new QAT optimizer (Bug #3: they were
    # lost when the warm-phase optimizer was replaced above).
    if sz_head is not None and not use_schedule_free:
        optimizer.add_param_group({
            'params': list(sz_head.parameters()),
            'lr': cfg.lr_quant, 'weight_decay': 0.0,
        })

    qat_no_improve = 0
    qat_patience = max(1, cfg.early_stop_patience)
    _n_batches_qat = max(cfg.windows_per_epoch // max(cfg.batch_size_quant, 1), 1)

    stable_ckpt_saved = False  # WSD stable-phase checkpoint (for continual training)

    # Track best PRD (and corresponding per-band) alongside best R so the
    # end-of-run summary can report both at the best checkpoint.
    best_val_prd_at_best_r = 100.0
    best_per_band_at_best_r = {}
    last_val_r = 0.0
    last_val_prd = 100.0
    last_per_band = {}
    last_val_epoch = 0

    # INT8 bridge / progressive quantization schedule
    _prog_sched = None
    try:
        from progressive_quant import ProgressiveQuantSchedule, set_model_bits
        has_set_bits = any(hasattr(m, 'set_bits') for m in codec.encoder.modules())
        if has_set_bits and int8_bridge:
            # INT8 bridge: FP32 warmup → INT8 QAT (40%) → ternary QAT (60%)
            _prog_sched = ProgressiveQuantSchedule(
                cfg.epochs_quant,
                schedule=[(0.40, 8), (0.60, 'ternary')]
            )
            print(f"[*] INT8 bridge: {_prog_sched}")
        elif has_set_bits:
            # Check for full progressive (ProgressiveConv1d modules)
            from progressive_quant import ProgressiveConv1d
            has_progressive = any(isinstance(m, ProgressiveConv1d) for m in codec.encoder.modules())
            if has_progressive:
                _prog_sched = ProgressiveQuantSchedule(cfg.epochs_quant)
                print(f"[*] Progressive quantization: {_prog_sched}")
    except ImportError:
        pass

    # Pre-cache modules that have lsq_alpha so the per-batch clamp loop
    # doesn't walk all modules via named_modules() every step.
    _alpha_modules = [m for _, m in codec.encoder.named_modules()
                      if hasattr(m, 'lsq_alpha')]

    if _resume_phase == 'qat' and resume and os.path.exists(resume):
        ckpt = _safe_load(resume, map_location=device)
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            if scheduler and ckpt.get('scheduler'):
                scheduler.load_state_dict(ckpt['scheduler'])
            print(f"[*] Restored QAT optimizer + scheduler state")
        except Exception as e:
            print(f"[!] Could not restore optimizer state: {e}")
        # Restore seizure head state (lost on resume without this)
        if sz_head is not None and ckpt.get('seizure_head'):
            try:
                sz_head.load_state_dict(ckpt['seizure_head'])
                print(f"[*] Restored seizure head state")
            except Exception:
                pass
        # Restore best tracking so PRD stays in sync
        if ckpt.get('best_val_r') is not None:
            cm.best_val_r = ckpt['best_val_r']
            cm.best_epoch = ckpt.get('epoch', 0)
        if ckpt.get('best_val_prd') is not None:
            best_val_prd_at_best_r = ckpt['best_val_prd']

    _qat_start = (_resume_epoch - cfg.epochs_warmup + 1) if _resume_phase == 'qat' else 1
    _qat_start = max(1, _qat_start)
    for ep in range(_qat_start, cfg.epochs_quant + 1):
        codec.train()
        if disc is not None:
            disc.train()
        # Progressive quantization: update bit width per epoch
        if _prog_sched is not None:
            bits = _prog_sched.get_bits(ep)
            set_model_bits(codec.encoder, bits)
            if ep == 1 or (ep > 1 and _prog_sched.get_bits(ep) != _prog_sched.get_bits(ep - 1)):
                b = f'{bits}b' if isinstance(bits, int) else bits
                print(f"  [PROG] Epoch {ep}: switched to {b}")
        # Schedule-Free requires explicit train/eval mode on the optimizer
        if use_schedule_free:
            optimizer.train()
        loss_acc, n = 0.0, 0
        _bi = 0
        for batch in train_ds.prefetch_typed_batches(
                batch_size=cfg.batch_size_quant, device=device,
                sampler=train_sampler):
            batch.assert_no_leakage(Split.TRAIN)   # safety net
            x_l3 = batch.l3_approx
            x_aug = augmentor(x_l3) if augmentor is not None else x_l3
            # CA: subset channels (variable-N) or pass through (N=21 parity).
            # variable_n is guarded off when augmentor is set, so x_aug==x_l3
            # there and the channel subset stays aligned with the fullband.
            x_aug, _fb, _coords, _cmask = _ca_inputs(
                x_aug, batch.fullband_target, channel_agnostic, variable_n, n_range)
            with amp_ctx:
                recon = codec(x_aug, quantize=True, coords=_coords, ch_mask=_cmask)
                # QAT: compare recon to AUGMENTED input, not original.
                # The encoder saw x_aug, so the loss should measure how
                # well it reconstructed what it saw — not how well it
                # inverted the augmentation (which caps R at ~0.93).
                g_loss, _ = joint_loss(recon, x_aug,
                                        fullband_target=_fb,
                                        ch_mask=_cmask,
                                        return_parts=False)

            # ---- Multi-task: seizure detection from latent ----
            if sz_head is not None and any(batch.has_seizure):
                # Encode with gradients — shared representation for multi-task.
                latent_for_sz = codec.encoder.encode(x_aug, quantize=True)
                sz_logits = sz_head(latent_for_sz)
                sz_labels = torch.tensor(
                    [1.0 if s else 0.0 for s in batch.has_seizure],
                    device=device)
                sz_loss = SeizureHead.loss(sz_logits, sz_labels)
                g_loss = g_loss + seizure_weight * sz_loss

            # ---- GAN: discriminator + generator adversarial step ----
            # The discriminator flattens [B, 21, T] → [B*21, 1, T],
            # creating a 21× batch multiplier. At batch=32 that's 672
            # discriminator forward samples — too much VRAM. We randomly
            # subsample 4 channels per step for the disc; the generator
            # still trains on all 21 channels via the reconstruction loss.
            if use_gan and batch.fullband_target is not None:
                fb_target = batch.fullband_target
                T_d = min(recon.shape[-1], fb_target.shape[-1])
                # Subsample channels for discriminator (4 of 21)
                n_disc_ch = min(4, recon.shape[1])
                ch_idx = torch.randperm(recon.shape[1], device=device)[:n_disc_ch]
                real = fb_target[:, ch_idx, :T_d]
                fake = recon[:, ch_idx, :T_d]

                # (a) Discriminator update — detached fake
                with amp_ctx:
                    real_scores, real_feats = disc(real)
                    fake_scores_d, _ = disc(fake.detach())
                    d_loss = disc.discriminator_loss(real_scores, fake_scores_d)
                disc_optimizer.zero_grad()
                d_loss.backward()
                disc_optimizer.step()

                # (b) Generator adversarial + feature-matching loss
                with amp_ctx:
                    fake_scores_g, fake_feats_g = disc(fake)
                    adv_loss, feat_loss = disc.generator_loss(
                        real_scores, fake_scores_g, real_feats, fake_feats_g,
                        feat_weight=feat_match_weight)
                g_loss = g_loss + gan_weight * (adv_loss + feat_loss)

            optimizer.zero_grad()
            g_loss.backward()
            # Per-COORDINATE grad clamp BEFORE the norm clip. SOAP (and any
            # Adam-family optimizer) is invariant to a global gradient rescale,
            # so clip_grad_norm_ is a no-op on the SOAP step (it cancels in
            # exp_avg/sqrt(exp_avg_sq)) — which let QAT diverge (grads -> 1e15
            # over ~50 steps even after alpha calibration fixed the onset). A
            # value clamp changes the gradient DIRECTION per coordinate, so SOAP
            # cannot cancel it; this is what actually bounds the QAT step.
            torch.nn.utils.clip_grad_value_(codec.parameters(), 1.0)
            _gnorm_qat = torch.nn.utils.clip_grad_norm_(codec.parameters(), cfg.grad_clip_quant)
            optimizer.step()
            # Hard alpha safety net AFTER optimizer step. Clamping after
            # step lets the optimizer converge smoothly — it sees the true
            # gradient and moves freely, then we project back to bounds.
            # The old placement (before step) caused oscillation: the
            # optimizer computed gradients on unclamped values but applied
            # them to clamped values, fighting the clamp every step.
            with torch.no_grad():
                for m in _alpha_modules:
                    # DATA-DRIVEN alpha clamp [0.5*std(W), 2*std(W)] per channel,
                    # not a fixed [1e-4, 20]. The fixed-20 ceiling let the learned
                    # LSQ alpha drift to ~20 while weights stayed ~0.06, so
                    # round(w/alpha)=round(0.003)=0 zeroed the entire focal_mid
                    # encoder body -> R capped at 0.34 (dissection 2026-06-03).
                    if hasattr(m, 'clamp_alpha'):
                        m.clamp_alpha()
                    else:
                        m.lsq_alpha.data.clamp_(min=1e-4, max=20.0)
            if ema_model is not None:
                ema_model.update_parameters(codec)
            loss_acc += g_loss.detach(); n += 1  # accumulate on GPU, no sync
            dash.step(phase='QAT', epoch=ep, batch=_bi,
                      n_batches=_n_batches_qat,
                      loss=float(g_loss.detach()),
                      grad_norm=float(_gnorm_qat))
            _bi += 1

        if scheduler is not None:
            scheduler.step()
        ep_total = cfg.epochs_warmup + ep

        # Always: structured EpochReport via TrainingLogger every epoch.
        avg_loss = float(loss_acc / max(n, 1))  # single GPU→CPU sync
        _elapsed = time.time() - _run_start_t
        _secs_per_ep = _elapsed / (cfg.epochs_warmup + ep)
        _total_eps = cfg.epochs_warmup + cfg.epochs_quant
        _eta_h = _secs_per_ep * (_total_eps - ep_total) / 3600
        _cur_lr = (scheduler.get_last_lr()[0] if scheduler
                   else optimizer.param_groups[0]['lr'])
        _wsd_phase = (scheduler.phase if isinstance(scheduler, WSDScheduler) else '')
        vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        _alpha_pl = alpha_stats_from_model(codec.encoder)
        _amin, _amean, _amax = reduce_alpha_stats(_alpha_pl)

        val_r = val_prd = 0.0
        per_band = {}
        _saved_qat = False
        if ep % cfg.val_interval == 0:
            if use_schedule_free:
                optimizer.eval()
            val_r, val_prd, per_band = validate_joint(
                codec, val_ds, device, quantize=True, amp=amp,
                channel_agnostic=channel_agnostic, variable_n=variable_n, n_range=n_range)
            # EMA validation: if EMA beats live model, use EMA R for
            # checkpoint selection. Free +0.003-0.01 R at no training cost.
            _ema_is_best = False
            if ema_model is not None:
                ema_r, ema_prd, _ = validate_joint(
                    ema_model, val_ds, device, quantize=True, amp=amp,
                    channel_agnostic=channel_agnostic, variable_n=variable_n, n_range=n_range)
                if ema_r > val_r:
                    print(f"           EMA R={ema_r:.4f} > live R={val_r:.4f}, using EMA")
                    val_r, val_prd = ema_r, ema_prd
                    _ema_is_best = True
            if use_schedule_free:
                optimizer.train()
            dash.update_val(val_r=val_r, best_r=cm.best_val_r)
            last_val_r, last_val_prd, last_per_band = val_r, val_prd, per_band
            last_val_epoch = ep_total
            if per_band:
                band_str = '  '.join(
                    f'{g} {per_band.get(b, 0.0):>4.1f}%'
                    for b, g in (('delta', 'δ'), ('theta', 'θ'), ('alpha', 'α'),
                                 ('beta', 'β'), ('gamma', 'γ'))
                )
                print(f"           per-band PRD: {band_str}")

        _emit(EpochReport(
            run_id=run_id, script='train_joint',
            phase=f'qat[{_wsd_phase}]' if _wsd_phase else 'qat',
            epoch=ep, global_epoch=ep_total, total_epochs=_total_eps,
            train_loss=avg_loss, val_r=val_r, val_prd=val_prd,
            best_val_r=cm.best_val_r, best_val_prd=best_val_prd_at_best_r,
            best_epoch=cm.best_epoch,
            val_prd_delta=per_band.get('delta', 0.0),
            val_prd_theta=per_band.get('theta', 0.0),
            val_prd_alpha=per_band.get('alpha', 0.0),
            val_prd_beta=per_band.get('beta', 0.0),
            val_prd_gamma=per_band.get('gamma', 0.0),
            encoder_params=n_enc, decoder_params=n_dec,
            decoder_tier=str(vocos_tier),
            lr=_cur_lr, vram_gb=vram_gb,
            saved_checkpoint=_saved_qat,
            alpha_per_layer=_alpha_pl,
            alpha_min=_amin, alpha_mean=_amean, alpha_max=_amax,
            quantize_active=True,
            secs_per_epoch=_secs_per_ep,
            eta_hours=_eta_h,
        ))

        # Rolling recovery: latest every epoch, best on improvement.
        _qat_state = {'encoder': codec.encoder.state_dict(),
                      'decoder': getattr(codec.decoder, '_orig_mod', codec.decoder).state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                      'seizure_head': sz_head.state_dict() if sz_head is not None else None,
                      'best_val_r': cm.best_val_r,
                      'best_val_prd': best_val_prd_at_best_r,
                      'epoch': ep_total, 'phase': 'qat',
                      'provenance': provenance}
        torch.save(_qat_state, _rec_dir / 'qat_latest.ckpt')
        if _saved_qat:
            torch.save(_qat_state, _rec_dir / 'qat_best.ckpt')

        if ep % cfg.val_interval == 0:
            try:
                cm_result = cm.on_validation(epoch=ep_total, val_r=val_r,
                                              val_prd=val_prd,
                                              raise_on_halt=False)
                # When CM saves encoder-best, also save the matching decoder state.
                if cm_result.get('saved_best'):
                    _saved_qat = True
                    # If EMA was better, overwrite the encoder best with EMA state
                    if _ema_is_best:
                        _ema_enc_sd = ema_model.module.encoder.state_dict()
                        torch.save({'state_dict': _ema_enc_sd,
                                    **provenance, 'phase': 'qat',
                                    'epoch': ep_total, 'source': 'ema'},
                                   enc_path)
                    codec.save_decoder(
                        dec_path,
                        provenance={**provenance, 'phase': 'qat',
                                     'epoch': ep_total,
                                     'val_r': val_r, 'val_prd': val_prd})
                    reason = cm_result.get('save_reason', 'r_improved')
                    reason_tag = ' (PRD tiebreak)' if reason == 'tie_prd' else ''
                    _ema_tag = ' [EMA]' if _ema_is_best else ''
                    print(f"    saved best to {enc_path.name} + "
                          f"{dec_path.name}{reason_tag}{_ema_tag}")
                    best_val_prd_at_best_r = val_prd
                    best_per_band_at_best_r = dict(per_band)
                    qat_no_improve = 0
                else:
                    qat_no_improve += 1
            except Exception:
                pass

            # Save stable-phase checkpoint when WSD enters decay.
            if (scheduler and isinstance(scheduler, WSDScheduler)
                    and scheduler.phase == 'decay'
                    and not stable_ckpt_saved):
                stable_enc = ckpt_dir / f'student_encoder_stable_{cfg.name}.ckpt'
                stable_dec = ckpt_dir / f'decoder_tier{vocos_tier}_stable_{cfg.name}.ckpt'
                codec.save_encoder(
                    stable_enc,
                    provenance={**provenance, 'phase': 'stable', 'epoch': ep_total})
                codec.save_decoder(
                    stable_dec,
                    provenance={**provenance, 'phase': 'stable', 'epoch': ep_total})
                print(f"    [WSD] Saved stable-phase checkpoint at epoch {ep_total} "
                      f"(resume here for continual training)")
                stable_ckpt_saved = True

            if qat_no_improve >= qat_patience:
                print(f"  [EARLY STOP] {qat_no_improve} val intervals without "
                      f"improvement (~{qat_no_improve * cfg.val_interval} epochs)")
                break

    cm.close()

    # ---- Final per-category evaluation (clinical-stratified metrics) ----
    final_cat_metrics = {}
    if use_schedule_free:
        optimizer.eval()
    print(f"\n[*] Final validation (per-category metrics)...")
    _final_r, _final_prd, _final_band, final_cat_metrics = validate_joint(
        codec, val_ds, device, quantize=True, amp=amp, per_category=True,
        channel_agnostic=channel_agnostic, variable_n=variable_n, n_range=n_range)

    # ---- EMA evaluation ----
    ema_val_r, ema_val_prd = 0.0, 100.0
    if ema_model is not None:
        print(f"[*] Evaluating EMA model (decay={ema_decay})...")
        ema_val_r, ema_val_prd, _ = validate_joint(
            ema_model, val_ds, device, quantize=True, amp=amp,
            channel_agnostic=channel_agnostic, variable_n=variable_n, n_range=n_range)
        print(f"    EMA R={ema_val_r:.4f}  PRD={ema_val_prd:.1f}%  "
              f"(vs best R={cm.best_val_r:.4f}  delta={ema_val_r - cm.best_val_r:+.4f})")

    # ---- End-of-run summary: ship/no-ship in one screen ----
    # R + PRD + per-band + LQS level + violations (= the to-do list to
    # reach the next-stricter tier). One glance tells you whether the
    # checkpoint is shippable as Clinical / Monitoring / Alerting tier.
    sys.path.insert(0, os.path.join(ROOT_DIR, 'lamquant'))
    from metrics import lqs_compliance

    lqs_level, lqs_viol = lqs_compliance(
        val_r=cm.best_val_r, val_prd=best_val_prd_at_best_r,
        per_band_prd_dict=best_per_band_at_best_r,
    )
    logger.log_summary(RunSummary(
        run_id=run_id, script='train_joint',
        config=provenance['training_config'],
        completed=True,
        best_val_r=cm.best_val_r, best_val_prd=best_val_prd_at_best_r,
        best_epoch=cm.best_epoch,
        final_val_r=last_val_r, final_val_prd=last_val_prd,
        final_epoch=last_val_epoch,
        best_prd_delta=best_per_band_at_best_r.get('delta', 0.0),
        best_prd_theta=best_per_band_at_best_r.get('theta', 0.0),
        best_prd_alpha=best_per_band_at_best_r.get('alpha', 0.0),
        best_prd_beta=best_per_band_at_best_r.get('beta', 0.0),
        best_prd_gamma=best_per_band_at_best_r.get('gamma', 0.0),
        lqs_level=lqs_level, lqs_violations=list(lqs_viol),
        start_time=_run_start_t, end_time=time.time(),
        total_seconds=time.time() - _run_start_t,
        encoder_checkpoint=str(enc_path),
        decoder_checkpoint=str(dec_path),
        alpha_csv=str(logger.alpha_csv),
        total_epochs_run=last_val_epoch,
    ))
    # Per-category + EMA are supplementary (not in RunSummary schema yet).
    if ema_model is not None:
        print(f'  EMA R: {ema_val_r:.4f}  PRD: {ema_val_prd:.1f}%  '
              f'(delta: R={ema_val_r - cm.best_val_r:+.4f})')
    if final_cat_metrics:
        print('  Clinical-stratified (validation):')
        for cat in ('seizure', 'spike_event', 'epilepsy_patient', 'sleep',
                    'pediatric', 'artifact', 'normal'):
            m = final_cat_metrics.get(cat)
            if m and m['n'] > 0:
                print(f'    {cat:20} R={m["r"]:.4f}  PRD={m["prd"]:.1f}%  '
                      f'(n={m["n"]})')

    # ---- Append to experiment log (refactor #77) ----
    # One row per training run, append-only JSONL. The (manifest_hash,
    # training_config_hash) pair uniquely identifies the recipe; the
    # rest is denormalised summary for greppability.
    try:
        from lamquant.common.experiment_log import (
            ExperimentRecord, log_experiment,
        )
        wall_seconds = float(getattr(cm, 'best_epoch', 0))   # placeholder
        try:
            wall_seconds = time.time() - _run_start_t
        except NameError:
            pass
        rec = ExperimentRecord(
            run_id=run_id,
            manifest_hash=provenance['manifest_hash'],
            training_config_hash=provenance['training_config_hash'],
            config_version=cfg_for_provenance.config_version,
            manifest_version=(manifest.version if manifest is not None
                              else 'lma-direct'),
            preset=cfg.name,
            vocos_tier=vocos_tier,
            seed=seed,
            asymmetric_weight=ASYM_W,
            asymmetric_kind=(asymmetric_kind if ASYM_W > 0 else ''),
            fullband_mode=use_fullband_mode,
            amp=bool(amp),
            compile_decoder=bool(compile_decoder),
            train_noise_bits=cfg.train_noise_bits,
            epochs_planned=cfg.epochs_warmup + cfg.epochs_quant,
            epochs_completed=last_val_epoch,
            best_val_r=cm.best_val_r,
            best_val_prd=best_val_prd_at_best_r,
            final_val_r=last_val_r,
            final_val_prd=last_val_prd,
            best_epoch=cm.best_epoch,
            per_band_prd=dict(best_per_band_at_best_r),
            lqs_level=lqs_level,
            lqs_violations=list(lqs_viol),
            wall_seconds=wall_seconds,
            completed=True,
            encoder_ckpt=str(enc_path),
            decoder_ckpt=str(dec_path),
            alpha_csv=alpha_csv,
        )
        log_path = log_experiment(rec)
        print(f'[*] Logged to {log_path}')
    except Exception as e:
        # Logging failures must NEVER take down a training run.
        print(f'[!] Failed to write experiment log entry: {e!s}')

    # Close the reviewer metric stream + finish wandb (both non-fatal).
    try:
        metric_log.close()
    except Exception:
        pass
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    return {
        'best_val_r': cm.best_val_r,
        'best_val_prd': best_val_prd_at_best_r,
        'best_epoch': cm.best_epoch,
        'best_per_band_prd': best_per_band_at_best_r,
        'final_val_r': last_val_r,
        'final_val_prd': last_val_prd,
        'lqs_level': lqs_level,
        'lqs_violations': lqs_viol,
        'ema_val_r': ema_val_r,
        'ema_val_prd': ema_val_prd,
        'encoder_path': str(enc_path),
        'decoder_path': str(dec_path),
        'alpha_csv': alpha_csv,
    }


# ============================================================
# CLI entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog='train_joint',
        description='Joint encoder+decoder training for LamQuant',
    )
    parser.add_argument('--config', choices=list(CONFIGS.keys()), default='fast',
                        help='Training preset (fast / standard / production)')
    parser.add_argument('--deployment',
                        choices=list(DEPLOYMENT_TIERS),
                        default='research',
                        help='Decoder tier by deployment use case '
                             '(dev=24K, mobile=200M, clinical=400M, research=800M). '
                             'Joint training anchor = research.')
    parser.add_argument('--tier', type=int, default=None,
                        help='Override numeric Vocos tier. Use --deployment '
                             'unless you specifically want a different config.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fullband-mode',
                        choices=['auto', 'ram', 'memmap', 'off'],
                        default='auto',
                        help='How to source the fullband target for Tier 3+ '
                             'iSTFT decoders. auto: ram for fast preset, '
                             'memmap for standard/production. ram: load all '
                             'fullband windows into RAM (~136 GB at full '
                             'scale). memmap: mmap a precomputed flat .dat '
                             '(use precompute_fullband_memmap.py). off: skip '
                             'fullband target — fall back to L3 loss.')
    parser.add_argument('--amp', dest='amp', action='store_true', default=True,
                        help='BF16 autocast for decoder fwd+bwd (~1.8× faster, '
                             'no quality compromise). Default ON.')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='Disable BF16 autocast (FP32 throughout).')
    parser.add_argument('--compile', dest='compile_decoder',
                        action='store_true', default=True,
                        help='torch.compile the decoder (~1.3-1.5× faster '
                             'after ~30s warmup). Default ON.')
    parser.add_argument('--no-compile', dest='compile_decoder',
                        action='store_false',
                        help='Disable torch.compile (use eager mode).')
    parser.add_argument('--augment', choices=['none', 'light', 'moderate', 'aggressive'],
                        default='moderate',
                        help='EEG augmentation preset (default: moderate)')
    parser.add_argument('--ema', dest='ema', action='store_true', default=True,
                        help='EMA of model weights (default ON)')
    parser.add_argument('--no-ema', dest='ema', action='store_false')
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--gan', dest='gan', action='store_true', default=True,
                        help='GAN training (MPD + MS-STFT discriminator). '
                             'Default ON. Only active with fullband target (Tier 3+).')
    parser.add_argument('--no-gan', dest='gan', action='store_false')
    parser.add_argument('--gan-weight', type=float, default=1.0,
                        help='Adversarial loss weight (default 1.0)')
    parser.add_argument('--feat-match-weight', type=float, default=2.0,
                        help='Feature matching loss weight (default 2.0)')
    parser.add_argument('--seizure-head', dest='seizure_head',
                        action='store_true', default=True,
                        help='Multi-task seizure detection head on encoder latent. '
                             'Default ON.')
    parser.add_argument('--no-seizure-head', dest='seizure_head',
                        action='store_false')
    parser.add_argument('--seizure-weight', type=float, default=0.1,
                        help='Weight on seizure detection loss (default 0.1)')
    parser.add_argument('--encoder-init', type=str, default=None,
                        help='Path to pretrained encoder weights (MAE, prior run). '
                             'Loaded before joint training starts.')
    parser.add_argument('--asymmetric-weight', type=float, default=0.0,
                        help='Coefficient on the asymmetric (envelope-weighted) '
                             'MSE term. 0 disables. Try 0.2 for the A/B test.')
    parser.add_argument('--asymmetric-kind', choices=['envelope', 'band'],
                        default='envelope',
                        help='envelope: naive Hilbert envelope weight (the '
                             'original proposal). band: per-band envelope '
                             'with positive bias on δ/θ/α and negative bias '
                             'on β/γ — addresses the EMG/spike confound.')
    parser.add_argument('--clinical-sampling', dest='clinical_sampling',
                        action='store_true', default=True,
                        help='Clinical-balanced sampling: oversample seizure/spike/rare events. '
                             'Default ON.')
    parser.add_argument('--no-clinical-sampling', dest='clinical_sampling',
                        action='store_false')
    parser.add_argument('--lr-schedule', choices=['wsd', 'cosine', 'schedule-free',
                                                  'muon', 'soap', 'adamw'],
                        default='soap',
                        help='Optimizer / LR schedule. soap: SOAP (Shampoo-Adam eigenbasis, '
                             'default, +0.0135 R over AdamW). wsd: AdamW + Warmup-Stable-Decay. '
                             'muon: Muon (DEAD with ternary QAT). cosine: AdamW + cosine. '
                             'schedule-free: Schedule-Free AdamW.')
    parser.add_argument('--infinite-lr', action='store_true', default=False,
                        help='Infinite stable phase — LR stays at peak forever. '
                             'Every checkpoint is shippable. Use for continual training.')
    parser.add_argument('--decay-frac', type=float, default=0.10,
                        help='Fraction of total epochs for cosine decay (default: 0.10). '
                             '0 = infinite stable phase (same as --infinite-lr).')
    parser.add_argument('--int8-bridge', action='store_true', default=False,
                        help='INT8 bridge: first 40%% of QAT at INT8, then ternary. '
                             'Reduces quantization shock vs direct FP32→ternary.')
    parser.add_argument('--resume', nargs='?', const='auto', default=None,
                        help='Resume from checkpoint. No arg = auto-detect from recovery dir. '
                             'Or provide explicit path.')
    # ---- LMA-direct training (BLUT canonical, ADR 0017) ----
    parser.add_argument('--lma-root', type=str, default=None,
                        help='Directory of per-recording .lma archives. When set '
                             'with --split-manifest, training reads LMA directly '
                             '(no NPZ precompute, no fullband memmap).')
    parser.add_argument('--split-manifest', type=str, default=None,
                        help='JSON split manifest (subjects + stems_by_subject). '
                             'Required when --lma-root is set.')
    parser.add_argument('--detail-bands', choices=['none', 'l3_detail', 'all'],
                        default='none',
                        help='Encoder input bands: none=L3 (21ch, MCU-deployable), '
                             'l3_detail=+15.6-31.25Hz LVFA, all=+all detail bands '
                             '(the >15Hz reconstruction basis). Channel count depends '
                             'on --detail-stack-mode. Decoder always reconstructs the '
                             '21-ch fullband target.')
    parser.add_argument('--detail-stack-mode', choices=['interp', 'fold'],
                        default='interp',
                        help='How detail bands stack onto L3 (ADR 0031). '
                             'interp (default, legacy): each band linearly resampled '
                             'to the 313 grid as 1 block (LOSSY — l1/l2 downsampled; '
                             'all=84ch). fold: zero-pad+reshape, information-preserving, '
                             'no coefficient dropped (all=168ch). Use fold for the '
                             'ADR-0031 input-limitation test so a null cannot be blamed '
                             'on the stacking.')
    parser.add_argument('--encoder-width', type=int, default=None,
                        help='Override preset encoder width (e.g. 256 research).')
    parser.add_argument('--encoder-blocks', type=int, default=None,
                        help='Override preset encoder depth / n_blocks (e.g. 12).')
    parser.add_argument('--encoder-kernels', type=str, default=None,
                        help='Override per-block kernels, comma-sep, len==blocks.')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override preset batch size (drop for big tier-7 decoder).')
    parser.add_argument('--epochs-warmup', type=int, default=None,
                        help='Override preset FP32 warmup epochs.')
    parser.add_argument('--epochs-quant', type=int, default=None,
                        help='Override preset QAT epochs.')
    parser.add_argument('--windows-per-epoch', type=int, default=None,
                        help='Override windows sampled per epoch (raise for ceiling).')
    parser.add_argument('--ckpt-dir', type=str, default=None,
                        help='Directory for the output checkpoints (enc/dec '
                             'student_*_{config}.ckpt). Default ROOT_DIR/lamquant/'
                             'student. BLUT passes a run-id-stamped dir so '
                             'concurrent same-preset runs do not clobber (ADR 0044 '
                             '#257); the stage stats exactly these files.')
    parser.add_argument('--soap-max-precond-dim', type=int, default=10000,
                        help='SOAP full-matrix preconditioner dim cap; tensors '
                        'wider than this fall back to diagonal. Default 10000 '
                        '(back-compat). Lower (e.g. 2048) to fit big Tier 5+ '
                        'decoders — the 10000-dim eigh OOMs a 24 GB card.')
    parser.add_argument('--max-windows-per-file', type=int, default=None,
                        help='Cap windows per recording in the base index (raise to '
                             'use more of long recordings; default ~5).')
    # ---- Channel-agnostic codec (CA-6) ----
    parser.add_argument('--channel-agnostic', action='store_true', default=False,
                        help='Build the channel-count-agnostic codec (position-'
                             'conditioned attention front-end + FiLM decoder head). '
                             'At N=21 with no --variable-n this is the warm-start '
                             'parity path (coords default to canonical 10-20).')
    parser.add_argument('--variable-n', action='store_true', default=False,
                        help='Random channel-subset augmentation (N∈[--n-min,--n-max]) '
                             'per batch. Requires --channel-agnostic; incompatible '
                             'with augmentation / GAN / seizure-head (all assume 21ch) '
                             '— use --augment none --no-gan --no-seizure-head.')
    parser.add_argument('--ca-decoder-legacy', action='store_true', default=False,
                        help='CA encoder + LEGACY fixed-21ch decoder head — the '
                             'warm-start isolation config (vary only the front-end). '
                             'N=21 only; incompatible with --variable-n.')
    parser.add_argument('--n-min', type=int, default=8, help='variable-N min channels')
    parser.add_argument('--n-max', type=int, default=21, help='variable-N max channels')
    parser.add_argument('--no-diagnostics', dest='diagnostics', action='store_false',
                        default=True, help='skip the pre-flight diagnostics gate '
                        '(data/shape/grad/coords sanity on the first batch).')
    parser.add_argument('--logger', choices=['none', 'wandb'], default='none',
                        help='Experiment tracker for live metrics. none (default): '
                        'the parquet/csv metric stream only. wandb: also log to '
                        'Weights & Biases (offline by default; WANDB_MODE=online to sync).')
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    import dataclasses as _dc
    _ov = {}
    if args.encoder_width is not None:   _ov['encoder_width'] = args.encoder_width
    if args.encoder_blocks is not None:  _ov['encoder_blocks'] = args.encoder_blocks
    if args.encoder_kernels is not None: _ov['encoder_kernels'] = args.encoder_kernels
    if args.batch_size is not None:
        for f in ('batch_size', 'batch_size_warmup', 'batch_size_quant', 'batch_size_fine'):
            if hasattr(cfg, f):
                _ov[f] = args.batch_size
    if args.epochs_warmup is not None: _ov['epochs_warmup'] = args.epochs_warmup
    if args.epochs_quant is not None:  _ov['epochs_quant'] = args.epochs_quant
    if args.windows_per_epoch is not None: _ov['windows_per_epoch'] = args.windows_per_epoch
    if _ov:
        cfg = _dc.replace(cfg, **_ov)
        print(f"[*] config overrides: {_ov}")
    tier = args.tier if args.tier is not None else DEPLOYMENT_TIERS[args.deployment]
    result = run(cfg, vocos_tier=tier, seed=args.seed,
                 ckpt_dir=args.ckpt_dir,
                 fullband_mode=args.fullband_mode,
                 amp=args.amp, compile_decoder=args.compile_decoder,
                 asymmetric_weight=args.asymmetric_weight,
                 asymmetric_kind=args.asymmetric_kind,
                 augment=args.augment,
                 ema=args.ema, ema_decay=args.ema_decay,
                 gan=args.gan, gan_weight=args.gan_weight,
                 feat_match_weight=args.feat_match_weight,
                 seizure_head=args.seizure_head,
                 seizure_weight=args.seizure_weight,
                 encoder_init=args.encoder_init,
                 clinical_sampling=args.clinical_sampling,
                 lr_schedule=args.lr_schedule,
                 decay_frac=args.decay_frac,
                 infinite_lr=args.infinite_lr,
                 int8_bridge=args.int8_bridge,
                 resume=args.resume,
                 lma_root=args.lma_root,
                 split_manifest=args.split_manifest,
                 detail_bands=args.detail_bands,
                 detail_stack_mode=args.detail_stack_mode,
                 max_windows_per_file=args.max_windows_per_file,
                 soap_max_precond_dim=args.soap_max_precond_dim,
                 channel_agnostic=args.channel_agnostic,
                 variable_n=args.variable_n,
                 ca_decoder_legacy=args.ca_decoder_legacy,
                 n_range=(args.n_min, args.n_max),
                 diagnostics=args.diagnostics,
                 logger_backend=args.logger)
    # Exit 0 = training RAN TO COMPLETION (ADR 0044). Quality (R/PRD/LQS)
    # is reported above and enforced by the PCCP gate stage downstream — it
    # is NOT the trainer's job to gate via the process exit code. The old
    # `0 if best_val_r > 0 else 1` made a completed probe whose R legitimately
    # starts at/near 0 (e.g. E1 early epochs, or a from-scratch warm-only run)
    # look like a subprocess FAILURE to the BLUT backend, failing the recipe
    # even though the checkpoints were written. A genuine crash raises and
    # propagates a nonzero exit on its own.
    return 0


if __name__ == '__main__':
    sys.exit(main())
