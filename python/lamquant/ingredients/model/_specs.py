"""Model ingredient specs (ADR 0050/0051). Importing this registers them.

A *model* ingredient is the network CONSTRUCTION a trainer's run() does once
before its epoch loop: ``build_ingredient("model", "<name>", cfg, device=dev)``
returns the constructed ``nn.Module`` (or a small dict of modules) already moved
to ``device`` — byte-identical to the trainer's inline ``Model(...).to(device)``.

The heavy neural/codec wheels are imported LAZILY inside ``build`` so importing
this module to register the specs stays cheap (no torch-model import at import
time). Three specs, all ``cache_relevant=True`` (the architecture is part of the
trained artifact's identity):

  * ``l3_teacher``  — ``oracle/train_l3_teacher.py``'s ``L3Teacher(width=...)``.
  * ``mae_encoder`` — ``student/pretrain_mae.py``'s encoder + MAE prediction head
                      pair (``TernaryMobileNetV5_Subband`` + ``MAEPredictionHead``).
  * ``joint_codec`` — ``student/train_joint.py``'s ``build_default_joint(...)``
                      (the full encoder+decoder JointCodec).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec


def _maybe_to(module, device):
    """Apply ``.to(device)`` when a device is given (matches the trainer's inline
    ``Model(...).to(device)``); pass through unchanged when device is None."""
    return module if device is None else module.to(device)


# ===========================================================================
# (1) l3_teacher — oracle/train_l3_teacher.py L3Teacher(width=args.width).
# ===========================================================================

@dataclass(frozen=True)
class L3TeacherConfig:
    width: int = 512


def _build_l3_teacher(cfg, *, device=None):
    # Package-form import (NOT the trainer's bare ``from train_teacher import``,
    # which needs ``lamquant/oracle`` pre-inserted on sys.path). The module lives
    # at ``lamquant/oracle/train_teacher.py``; the fully-qualified form resolves
    # regardless of the caller's sys.path. Same class.
    from lamquant.oracle.train_teacher import L3Teacher
    return _maybe_to(L3Teacher(width=cfg.width), device)


@register_ingredient
def _l3_teacher_spec():
    return IngredientSpec(
        name="l3_teacher", kind="model", config_cls=L3TeacherConfig,
        cache_relevant=True,
        build=_build_l3_teacher,
    )


# ===========================================================================
# (2) mae_encoder — student/pretrain_mae.py encoder + MAE prediction head.
# ===========================================================================

@dataclass(frozen=True)
class MaeEncoderConfig:
    in_ch: int = 21
    latent_dim: int = 32


def _build_mae_encoder(cfg, *, device=None):
    """Build the ``{"encoder": ..., "pred_head": ...}`` pair byte-identically to
    ``pretrain_mae.run_pretraining`` (lines 155-157):

        encoder   = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32).to(device)
        pred_head = MAEPredictionHead(latent_dim=32).to(device)

    The prediction head is discarded after pretraining (only the encoder weights
    transfer); both are returned so the trainer wires the optimizer over both.
    """
    from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
    from lamquant.student.pretrain_mae import MAEPredictionHead

    encoder = _maybe_to(
        TernaryMobileNetV5_Subband(in_ch=cfg.in_ch, latent_dim=cfg.latent_dim),
        device)
    pred_head = _maybe_to(MAEPredictionHead(latent_dim=cfg.latent_dim), device)
    return {"encoder": encoder, "pred_head": pred_head}


@register_ingredient
def _mae_encoder_spec():
    return IngredientSpec(
        name="mae_encoder", kind="model", config_cls=MaeEncoderConfig,
        cache_relevant=True,
        build=_build_mae_encoder,
    )


# ===========================================================================
# (3) joint_codec — student/train_joint.py build_default_joint(...).
# ===========================================================================

@dataclass(frozen=True)
class JointCodecModelConfig:
    """Mirrors ``train_joint``'s ``build_default_joint(...)`` kwargs verbatim.

    ``encoder_kernels`` is the trainer's already-resolved value (None or a tuple)
    and is forwarded as-is. The defaults below mirror the trainer's resolved
    values for the frozen MONITOR preset (latent_dim 32, decoder_channels 21);
    a recipe overrides per-field. Construction is byte-identical to the inline
    ``build_default_joint(...).to(device)`` call (lines 517-524).
    """
    latent_dim: int = 32
    encoder_width: int = 0
    vocos_tier: int = 3
    in_channels: int = 21
    decoder_channels: int = 21
    gradient_checkpointing: bool = False
    encoder_blocks: int = 0
    encoder_kernels: Optional[tuple] = None
    channel_agnostic: bool = False
    ca_decoder: bool = False


def _build_joint_codec_model(cfg, *, device=None):
    # Package-form import (NOT the trainer's bare ``from joint_codec import``,
    # which needs ``lamquant/student`` pre-inserted on sys.path). The module
    # lives at ``lamquant/student/joint_codec.py``; the fully-qualified form
    # always resolves regardless of the caller's sys.path. Same factory.
    from lamquant.student.joint_codec import build_default_joint

    codec = build_default_joint(
        latent_dim=cfg.latent_dim, encoder_width=cfg.encoder_width,
        vocos_tier=cfg.vocos_tier, in_channels=cfg.in_channels,
        decoder_channels=cfg.decoder_channels,
        gradient_checkpointing=cfg.gradient_checkpointing,
        encoder_blocks=cfg.encoder_blocks,
        encoder_kernels=cfg.encoder_kernels,
        channel_agnostic=cfg.channel_agnostic,
        ca_decoder=cfg.ca_decoder)
    return _maybe_to(codec, device)


@register_ingredient
def _joint_codec_model_spec():
    return IngredientSpec(
        name="joint_codec", kind="model", config_cls=JointCodecModelConfig,
        cache_relevant=True,
        build=_build_joint_codec_model,
    )
