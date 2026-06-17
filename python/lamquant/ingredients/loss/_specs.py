"""Loss ingredient specs (ADR 0050/0051). Importing this registers them.

A *loss* ingredient is built into a callable closure:
``build_ingredient("loss", "<name>", cfg, **extra)`` returns ``loss_fn(...)``.
The bodies are transcribed VERBATIM from the trainer inner loops — the only
edits are (a) hoisting build-time singletons (the spectral loss module) and
(b) turning the trainer's surrounding closure variables into cfg fields / call
args, so the autograd behaviour stays byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec

# Metric primitives — same functions train_joint imports (it goes through the
# legacy ``from metrics import ...`` sys.path shim; the canonical home is
# lamquant.common.metrics, which re-exports the identical objects).
from lamquant.common.metrics import (
    asymmetric_eeg_loss as _asym_env,
    band_aware_asymmetric_loss as _asym_band,
    per_band_relative_loss as _per_band_rel,
    masked_pearson_r_torch,
    masked_prd_torch,
    prd_torch,  # noqa: F401  (kept for parity with the trainer import block)
)


# ============================================================================
# (1) joint_codec — student/train_joint.py joint_loss closure
# ============================================================================

@dataclass(frozen=True)
class JointCodecLossConfig:
    pearson_r_weight: float = 0.5
    spectral_weight: float = 0.1
    prd_weight: float = 0.1
    asymmetric_weight: float = 0.0
    band_loss_weight: float = 0.5
    asymmetric_kind: str = 'envelope'


def _make_spectral_loss(device):
    """Multi-resolution STFT loss tuned for EEG frequency bands.

    Verbatim from train_joint.make_spectral_loss (FFT sizes resolve the
    clinically important EEG bands at 250 Hz / 31.25 Hz L3 proxy).
    """
    from auraloss.freq import MultiResolutionSTFTLoss
    fft_sizes = [64, 128, 256, 512]
    return MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes,
        hop_sizes=[max(n // 4, 1) for n in fft_sizes],
        win_lengths=fft_sizes,
    ).to(device)


def _build_joint_codec_loss(cfg, *, device):
    if cfg.asymmetric_kind not in ('envelope', 'band'):
        raise ValueError(
            "asymmetric_kind must be 'envelope' or 'band', "
            f"got {cfg.asymmetric_kind!r}")
    # Build-time singleton (the trainer builds it once before the loop).
    spectral_loss = _make_spectral_loss(device)
    R_W = cfg.pearson_r_weight
    SP_W = cfg.spectral_weight
    PRD_W = cfg.prd_weight
    ASYM_W = float(cfg.asymmetric_weight)
    BAND_W = float(cfg.band_loss_weight)
    asymmetric_kind = cfg.asymmetric_kind
    if ASYM_W > 0:
        asym_fn = _asym_band if asymmetric_kind == 'band' else _asym_env

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
        # Per-band RELATIVE loss — only meaningful on the fullband target
        # (the EEG bands need fs=250 Hz; the L3 domain at ~31 Hz has no
        # beta/gamma). FP32 (FFT bandpass) outside the bf16 autocast.
        if BAND_W > 0 and domain == 'fullband':
            with torch.amp.autocast(device_type=device.type, enabled=False):
                l_band = _per_band_rel(recon_c.float(), target_c.float(), fs=250.0)
        else:
            l_band = 0.0
        total = (l_mse + R_W * l_r + PRD_W * l_prd + SP_W * l_sp
                 + ASYM_W * l_asym + BAND_W * l_band)
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
            'band': (l_band.detach().item()
                     if isinstance(l_band, torch.Tensor) else l_band),
            'loss_domain': domain,
        }

    return joint_loss


@register_ingredient
def _joint_codec_loss_spec():
    return IngredientSpec(
        name="joint_codec", kind="loss", config_cls=JointCodecLossConfig,
        # MultiResolutionSTFTLoss comes from auraloss — hard-gate it like the
        # trainer's import (the spectral term is part of the default loss).
        requires=('pkg:auraloss',),
        cache_relevant=True,
        build=_build_joint_codec_loss,
    )


# ============================================================================
# (2) four_state_objective — snn/train_4state_controller.py loss block
# ============================================================================

@dataclass(frozen=True)
class FourStateObjectiveConfig:
    use_ordinal: bool = False
    crit_floor: float = 0.88
    lambda_spike: float = 0.01  # train_4state_controller --lambda-spike default
    lambda_distill: float = 0.0


def _build_four_state_objective(cfg):
    # ``constrained_loss`` stays imported from the canonical module (the trainer
    # imports it at module top). Lazy-import so this specs module is importable
    # without exercising the ordinal-loss dependency chain until a build.
    from lamquant.snn.ordinal_loss import constrained_loss

    use_ordinal = cfg.use_ordinal
    crit_floor = cfg.crit_floor
    lambda_spike = cfg.lambda_spike
    lambda_distill = cfg.lambda_distill

    def loss_fn(class_logits, target, *, head, head_kind, class_weights,
                spike_rate, activity_logits=None, distiller=None,
                teacher_feat=None):
        # ``class_weights`` is the trainer's ``cw`` (already device-moved).
        cw = class_weights
        if head_kind == "crf":
            # CRF path is unaffected by --ordinal (the ordinal/constrained
            # objective replaces the FLAT softmax CE, not the CRF NLL).
            loss_main = head.neg_log_likelihood(class_logits, target)
        elif use_ordinal:
            # ADR-0027 #4: ordinal + constrained objective (drop-in for the
            # weighted CE). escalation/ramp args stay at their module defaults.
            loss_main = constrained_loss(class_logits, target, weight=cw,
                                         crit_floor=crit_floor)
        else:
            loss_main = F.cross_entropy(class_logits, target, weight=cw)
        loss = loss_main + lambda_spike * spike_rate

        # ADR-0027 #3: foundation-teacher feature distillation. The teacher is
        # frozen + runs under no_grad inside teacher_features; only student_proj
        # (an optimizer param group added in main) + the backbone receive grad.
        # NB: the trainer computes ``teacher_feat`` from ``l3`` inside the loop;
        # the loss ingredient receives the precomputed [B,200] tensor instead
        # (``l3`` is not a loss arg) — the distill_loss call is byte-identical.
        if distiller is not None:
            loss = loss + lambda_distill * distiller.distill_loss(
                activity_logits, teacher_feat)

        return loss

    return loss_fn


@register_ingredient
def _four_state_objective_spec():
    return IngredientSpec(
        name="four_state_objective", kind="loss",
        config_cls=FourStateObjectiveConfig,
        cache_relevant=True,
        build=_build_four_state_objective,
    )


# ============================================================================
# (3a) masked_recon_mse_time — snn/pretrain_ssl_tueg.py masked_recon_loss
#      (masked-MEAN over masked timesteps only, with the shape guards)
# ============================================================================

@dataclass(frozen=True)
class MaskedReconMseTimeConfig:
    pass


def _build_masked_recon_mse_time(cfg):
    def masked_recon_loss(recon: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """MSE between ``recon`` and ``target`` over MASKED timesteps only.

        Args:
            recon:  [B,21,T] reconstructed L3.
            target: [B,21,T] original (unmasked) L3.
            mask:   [B,T] bool, True == masked (the positions we score).

        Returns:
            scalar MSE averaged over (masked timesteps x 21 channels).
        """
        assert recon.shape == target.shape and recon.dim() == 3
        assert mask.shape == (recon.shape[0], recon.shape[2]), \
            f"mask must be [B,T], got {tuple(mask.shape)} for recon {tuple(recon.shape)}"
        m = mask.unsqueeze(1).to(recon.dtype)        # [B,1,T]
        sq = (recon - target).pow(2) * m              # zero on unmasked
        denom = m.sum() * recon.shape[1]              # masked steps x channels
        # denom > 0 is guaranteed by make_span_mask (>=1 masked step per row).
        assert denom.item() > 0, "no masked positions — empty SSL loss"
        return sq.sum() / denom

    return masked_recon_loss


@register_ingredient
def _masked_recon_mse_time_spec():
    return IngredientSpec(
        name="masked_recon_mse_time", kind="loss",
        config_cls=MaskedReconMseTimeConfig,
        cache_relevant=True,
        build=_build_masked_recon_mse_time,
    )


# ============================================================================
# (3b) masked_recon_mse_patch — student/pretrain_mae.py MAE loss
#      (all-MEAN: F.mse_loss(recon*mask, target*mask))  — NOT byte-equal to 3a
# ============================================================================

@dataclass(frozen=True)
class MaskedReconMsePatchConfig:
    pass


def _build_masked_recon_mse_patch(cfg):
    def loss_fn(recon, target, mask):
        # Verbatim from pretrain_mae.py: loss only on masked regions, but the
        # denominator is the FULL tensor numel (F.mse_loss default 'mean'),
        # NOT the masked-element count — this differs from masked_recon_mse_time.
        return F.mse_loss(recon * mask, target * mask)

    return loss_fn


@register_ingredient
def _masked_recon_mse_patch_spec():
    return IngredientSpec(
        name="masked_recon_mse_patch", kind="loss",
        config_cls=MaskedReconMsePatchConfig,
        cache_relevant=True,
        build=_build_masked_recon_mse_patch,
    )


# ============================================================================
# (4) teacher_mse — oracle/train_l3_teacher.py plain reconstruction MSE
# ============================================================================

@dataclass(frozen=True)
class TeacherMseConfig:
    pass


def _build_teacher_mse(cfg):
    def loss_fn(recon, target):
        return F.mse_loss(recon, target)

    return loss_fn


@register_ingredient
def _teacher_mse_spec():
    return IngredientSpec(
        name="teacher_mse", kind="loss", config_cls=TeacherMseConfig,
        cache_relevant=True,
        build=_build_teacher_mse,
    )
