"""JointCodec — train the ternary MCU encoder and the fp32 base-station
decoder as one model, amputate at ship time.

Why
---
The earlier flow trained the encoder against its own internal mini-decoder
(`TernaryMobileNetV5_Subband.decode`). That mini-decoder:

  - is also ternary (constrained by the MCU's 64KB budget) even though
    it never ships to the MCU
  - is thrown away at firmware-export time
  - has a different architecture from the production decoder (Vocos)

Result: the encoder converged to a latent distribution the mini-decoder
liked, NOT the distribution the production Vocos decoder needed. Vocos
then had to learn the encoder's idiosyncratic representation from
scratch.

The fix is what every modern neural codec (SoundStream, EnCodec, DAC)
does: train encoder + production decoder JOINTLY, end-to-end, then
amputate.

  - Encoder: TernaryMobileNetV5_Subband.encode (ternary QAT, ships to MCU)
  - Decoder: VocosDecoder Tier 1+ (fp32, ships to base station / GPU)
  - Loss: MSE(decoder(encode(x)), x) + 0.5·R + 0.03·spectral
  - Save: encoder.ckpt and decoder.ckpt separately on each best

Memory / speed
--------------
Tier 1 Vocos (~100k params) adds ~5% to step time and <50MB GPU memory.
Tier 3 (~100M params) adds ~30% step time and ~400MB. Both fit
comfortably on a single 4090.

Usage
-----
    from joint_codec import JointCodec, build_default_joint
    codec = build_default_joint(latent_dim=32, vocos_tier=1)
    out = codec(x)              # forward pass: encode → decode
    codec.save_encoder('enc.ckpt')
    codec.save_decoder('dec.ckpt')
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lamquant' / 'student'))
sys.path.insert(0, str(ROOT / 'lamquant' / 'decoder'))

from lamquant.common.utils import safe_torch_load


class JointCodec(nn.Module):
    """A wrapper over (encoder, decoder) that exposes them as one nn.Module.

    The encoder and decoder are kept as PUBLIC attributes — `codec.encoder`
    and `codec.decoder` — so the trainer can freeze one, get distinct
    optimizer parameter groups, or save them separately at any time.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        if not hasattr(encoder, 'encode'):
            raise TypeError(
                f"encoder must have an `.encode(x, quantize=)` method; "
                f"got {type(encoder).__name__}")
        self.encoder = encoder
        self.decoder = decoder

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    def forward(self, x: torch.Tensor, *, quantize: bool = True) -> torch.Tensor:
        """End-to-end: encode → quantize → decode.

        Args:
            x:        [B, 21, 313] L3 approximation (or whatever the
                      encoder takes).
            quantize: If True, the encoder applies its ternary QAT path.
                      Set False during the warm-up phase before QAT.

        Returns:
            [B, 21, T] reconstruction (T matches decoder.target_len for
            Tier 1-2; matches raw EEG length for Tier 3+).
        """
        latent = self.encoder.encode(x, quantize=quantize)
        return self.decoder(latent)

    # ------------------------------------------------------------
    # Parameter introspection — used by the trainer to build optimizer groups.
    # ------------------------------------------------------------

    def encoder_parameters(self):
        """All trainable encoder params (including LSQ alphas)."""
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def decoder_parameters(self):
        """All trainable decoder params."""
        return [p for p in self.decoder.parameters() if p.requires_grad]

    def encoder_alpha_parameters(self):
        """Subset of encoder params that are LSQ alphas — for the
        BitNet-style separate weight-decay group."""
        return [p for n, p in self.encoder.named_parameters()
                if p.requires_grad and n.endswith('lsq_alpha')]

    def encoder_other_parameters(self):
        """Encoder params that are NOT LSQ alphas."""
        return [p for n, p in self.encoder.named_parameters()
                if p.requires_grad and not n.endswith('lsq_alpha')]

    def freeze_decoder(self):
        """Freeze decoder weights. Useful for the encoder-QAT phase
        (decoder pre-trained, encoder fine-tuning under quantization)."""
        for p in self.decoder.parameters():
            p.requires_grad = False

    def freeze_encoder(self):
        """Freeze encoder weights. Useful for an initial decoder-only
        warmup against a frozen encoder bootstrap."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------
    # Save / load — encoder and decoder go to SEPARATE files.
    # ------------------------------------------------------------

    def save_encoder(self, path, provenance: dict = None) -> Path:
        """Save encoder for firmware export.

        Schema (new):
            {'state_dict': OrderedDict, 'saved_at': str, ...provenance}
        Old saves (raw state_dict) are still readable via load_encoder().

        provenance: optional dict embedded into the saved file. Standard
        keys: manifest_hash, manifest_path, manifest_version, run_id.
        Allows post-hoc tooling to identify what manifest produced this
        encoder without inspecting filenames or filesystem timestamps.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_ckpt_payload(self.encoder, provenance), path)
        return path

    def save_decoder(self, path, provenance: dict = None) -> Path:
        """Save decoder for base-station deployment. Same schema as
        save_encoder. See its docstring."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_ckpt_payload(self.decoder, provenance), path)
        return path

    def load_encoder(self, path, strict: bool = True):
        # Contains non-tensor metadata (provenance from CheckpointManager)
        state = safe_torch_load(path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        return self.encoder.load_state_dict(state, strict=strict)

    def load_decoder(self, path, strict: bool = True):
        # Contains non-tensor metadata (provenance from CheckpointManager)
        state = safe_torch_load(path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        # torch.compile wraps the module and prefixes all state_dict keys
        # with `_orig_mod.`. Strip the prefix so the state loads into the
        # unwrapped decoder cleanly. Idempotent if prefix is absent.
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
        return self.decoder.load_state_dict(state, strict=strict)


def _ckpt_payload(model, provenance: dict = None) -> dict:
    """Build the standard checkpoint payload (state_dict + provenance)."""
    from datetime import datetime, timezone
    payload = {
        'state_dict': model.state_dict(),
        'saved_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    if provenance:
        payload.update(provenance)
    return payload


def read_ckpt_provenance(path) -> dict:
    """Inspect a checkpoint's provenance metadata without loading weights.

    Returns the non-state-dict keys (manifest_hash, manifest_path, run_id,
    saved_at, ...) as a plain dict. Returns {} for legacy raw-state-dict
    saves. Useful for `lamquant.py inspect <ckpt>`.
    """
    # Contains non-tensor metadata (provenance from CheckpointManager)
    state = safe_torch_load(path, map_location='cpu')
    if not isinstance(state, dict) or 'state_dict' not in state:
        return {}
    return {k: v for k, v in state.items() if k != 'state_dict'}


# ============================================================
# Convenience builder — production defaults
# ============================================================

def build_default_joint(latent_dim: int = 32,
                         encoder_width: int = 128,
                         vocos_tier: int = 3,
                         in_channels: int = 21,
                         target_len: int = 313,
                         gradient_checkpointing: bool = False,
                         encoder_blocks: int = 3,
                         encoder_kernels: tuple = (3, 5, 7)) -> JointCodec:
    """Build the production joint codec with sensible defaults.

    Args:
        encoder_blocks: Number of focal blocks in the encoder (default 3).
        encoder_kernels: Per-block kernel sizes (default (3, 5, 7)).
            Must have exactly encoder_blocks entries. First block gets
            stride=2 (with ZeroPadShortcut), last block gets stride=2,
            middle blocks get stride=1.

    Tier roles (deployment plan):

      Tier 3 (~800M params, default for joint training)
        Joint trained with the encoder. The encoder is permanently
        co-adapted to this decoder's representation. Ships as the
        archival / research decoder. Target R ≥ 0.94.

      Tier 8 (200M, mobile)
        Distilled from Tier 3 with the frozen encoder. FP16 → INT8 for
        phone NPU deployment. Target R ≥ 0.88 (monitoring + ambulatory
        clinical). Replaces what used to be called Tier 1 (the 100K dev
        tier remains as `tier=1` for fast iteration tests).

      Tier 2 (~400M)
        Distilled from Tier 3 with the frozen encoder. Ships as the
        clinical workstation decoder. Target R ≥ 0.92.

      Tier 1 (~100K, dev tier)
        Tiny decoder for unit tests and rapid iteration. Not for
        production. Useful when you want a forward+backward pass that
        runs in a few ms.
    """
    from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
    from lamquant_neural.models.vocos_decoder import VocosDecoder

    encoder = TernaryMobileNetV5_Subband(
        in_ch=in_channels, latent_dim=latent_dim, width=encoder_width,
        n_blocks=encoder_blocks, kernel_sizes=encoder_kernels,
    )
    decoder = VocosDecoder(
        tier=vocos_tier, latent_dim=latent_dim,
        n_channels=in_channels, target_len=target_len,
        gradient_checkpointing=gradient_checkpointing,
    )
    return JointCodec(encoder, decoder)


# ============================================================
# Named tier aliases — user-facing deployment scheme
# ============================================================
# The numeric `tier` field in vocos_decoder.TIER_CONFIGS evolved
# organically and is not monotonic in parameter count. These aliases
# map the user's deployment-intent names to the actual numeric tier:
#
#     DEV       (dev/test, ~24K)        → tier=1
#     MOBILE    (200M, phone NPU)        → tier=8
#     CLINICAL  (400M, hospital GPU)     → tier=6
#     RESEARCH  (800M, joint anchor)     → tier=7
#
# Joint training uses RESEARCH as the anchor (encoder co-adapts to the
# largest decoder, then smaller decoders distill from it).
TIER_DEV       = 1
TIER_MOBILE    = 8
TIER_CLINICAL  = 6
TIER_RESEARCH  = 7

DEPLOYMENT_TIERS = {
    'dev':       TIER_DEV,
    'mobile':    TIER_MOBILE,
    'clinical':  TIER_CLINICAL,
    'research':  TIER_RESEARCH,
}


__all__ = [
    'JointCodec', 'build_default_joint',
    'TIER_DEV', 'TIER_MOBILE', 'TIER_CLINICAL', 'TIER_RESEARCH',
    'DEPLOYMENT_TIERS',
]
