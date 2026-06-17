"""Sampler ingredient specs (ADR 0050/0051). Importing this registers them.

A *sampler* ingredient builds the per-batch MASK that selects which positions a
self-supervised reconstruction objective scores. It is built into a callable:
``build_ingredient("sampler", "<name>", cfg)`` returns ``make_mask(...)``.

The bodies are transcribed VERBATIM from the trainer module-level functions —
the only edit is moving the masking hyperparameters (mask_frac/mean_span,
mask_ratio/patch_size) onto the cfg dataclass (so the choice rides a hashed
field, never a default) while the per-call shape/RNG/device stay call args. The
output tensors are byte-identical to the inline functions.

Two specs, both ``cache_relevant=True`` (the masking regime changes what the SSL
objective learns, so it is part of the trained artifact's identity):

  * ``span_mask``  — ``snn/pretrain_ssl_tueg.py``'s ``make_span_mask``: contiguous
                     span masking over the time axis (the BERT-span regime).
  * ``patch_mask`` — ``student/pretrain_mae.py``'s ``create_mask``: contiguous
                     patch masking, expanded across channels (the MAE regime).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec


# ===========================================================================
# (1) span_mask — snn/pretrain_ssl_tueg.py make_span_mask (verbatim body).
# ===========================================================================

@dataclass(frozen=True)
class SpanMaskConfig:
    mask_frac: float = 0.5
    mean_span: int = 10


def _build_span_mask(cfg):
    mask_frac = cfg.mask_frac
    mean_span = cfg.mean_span

    def make_span_mask(B: int, T: int, generator: torch.Generator,
                       device: torch.device) -> torch.Tensor:
        """Boolean [B, T] mask, ``True`` where the timestep is MASKED (to predict).

        Contiguous spans (mean length ``mean_span``) are masked until ~``mask_frac``
        of the T timesteps are covered, independently per batch row. Span masking
        (vs i.i.d. per-timestep) forces the model to use temporal context rather
        than interpolating a single dropped frame — the regime the oracle finding
        says the event tiers (CRITICAL/INTERESTING) live in.

        Each row is guaranteed at least one masked timestep so the MSE always has
        a denominator (no silent zero-loss batch).

        Args:
            B, T: batch and time dims.
            generator: torch.Generator for reproducible masking.
            device: device for the returned mask.

        Returns:
            ``[B, T]`` bool tensor, True == masked.
        """
        assert isinstance(B, int) and B > 0, f"B must be positive int, got {B!r}"
        assert isinstance(T, int) and T > 0, f"T must be positive int, got {T!r}"
        assert 0.0 < mask_frac < 1.0, f"mask_frac must be in (0,1), got {mask_frac}"
        assert isinstance(mean_span, int) and mean_span >= 1, \
            f"mean_span must be int >= 1, got {mean_span!r}"

        target_masked = max(1, int(round(mask_frac * T)))
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            n_masked = 0
            # Cap the number of span placements so a pathological RNG draw can't
            # loop forever (each span adds >= 1 masked step, so 4*T placements is
            # a generous ceiling that the target-coverage break hits well before).
            for _ in range(4 * T):
                if n_masked >= target_masked:
                    break
                # Span length: at least 1, centred on mean_span (uniform 1..2*mean-1
                # has expectation mean_span; cheap and bounded).
                hi = 2 * mean_span - 1 if mean_span > 1 else 1
                span = int(torch.randint(1, hi + 1, (1,), generator=generator,
                                         device=device).item())
                start = int(torch.randint(0, T, (1,), generator=generator,
                                          device=device).item())
                end = min(start + span, T)
                newly = (~mask[b, start:end]).sum().item()
                mask[b, start:end] = True
                n_masked += int(newly)
            if not bool(mask[b].any()):
                # Guarantee at least one masked step per row.
                j = int(torch.randint(0, T, (1,), generator=generator,
                                       device=device).item())
                mask[b, j] = True
        assert mask.shape == (B, T) and mask.dtype == torch.bool
        assert bool(mask.any()), "span mask produced an all-False mask"
        return mask

    return make_span_mask


@register_ingredient
def _span_mask_spec():
    return IngredientSpec(
        name="span_mask", kind="sampler", config_cls=SpanMaskConfig,
        cache_relevant=True,
        build=_build_span_mask,
    )


# ===========================================================================
# (2) patch_mask — student/pretrain_mae.py create_mask (verbatim body).
# ===========================================================================

@dataclass(frozen=True)
class PatchMaskConfig:
    mask_ratio: float = 0.5
    patch_size: int = 16


def _build_patch_mask(cfg):
    mask_ratio = cfg.mask_ratio
    patch_size = cfg.patch_size

    def create_mask(batch_size: int, n_channels: int, l3_len: int,
                    device='cpu') -> torch.Tensor:
        """Create a binary mask for L3 patches. 1 = masked (to predict).

        Patches are contiguous blocks of `patch_size` timesteps across
        all channels simultaneously (same mask for all channels within
        a sample, different mask per sample in the batch).
        """
        n_patches = l3_len // patch_size
        n_masked = int(n_patches * mask_ratio)
        mask = torch.zeros(batch_size, 1, l3_len, device=device)
        for b in range(batch_size):
            # Random patch indices to mask
            masked_idx = torch.randperm(n_patches, device=device)[:n_masked]
            for idx in masked_idx:
                start = idx * patch_size
                end = min(start + patch_size, l3_len)
                mask[b, :, start:end] = 1.0
        return mask.expand(-1, n_channels, -1)

    return create_mask


@register_ingredient
def _patch_mask_spec():
    return IngredientSpec(
        name="patch_mask", kind="sampler", config_cls=PatchMaskConfig,
        cache_relevant=True,
        build=_build_patch_mask,
    )
