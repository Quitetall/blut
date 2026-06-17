"""Step ingredient specs (ADR 0050/0051). Importing this registers them.

A *step* ingredient encapsulates one inner optimizer step — the load-bearing
backward + grad-clip + optimizer.step + post-step clamp sequence — and is built
into a callable: ``build_ingredient("step", "qat_codec", cfg)`` returns
``step(loss, model, optimizer, alpha_modules) -> grad_norm``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec


@dataclass(frozen=True)
class QatCodecStepConfig:
    grad_clip_value: float = 1.0
    grad_clip_norm: float = 1.0


def _qat_codec_step(cfg):
    """train_joint's QAT generator step. The ordering is LOAD-BEARING and is
    transcribed verbatim from the inline loop:

      1. a SINGLE backward (the #255 donated-buffer / grad-health invariant —
         the gradient-health check reads grads from this one backward);
      2. a per-COORDINATE value-clip BEFORE the norm-clip. SOAP (and any
         Adam-family optimizer) is invariant to a global gradient rescale, so
         clip_grad_norm_ alone is a no-op on the SOAP step — the value-clip is
         what actually bounds the QAT step (grads otherwise -> 1e15);
      3. optimizer.step();
      4. the hard alpha-clamp AFTER the step. Clamping before the step caused
         oscillation (gradients computed on unclamped values, applied to clamped
         ones); after the step the optimizer moves freely, then we project back.
    """
    def step(loss, model, optimizer, alpha_modules):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), cfg.grad_clip_value)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.grad_clip_norm)
        optimizer.step()
        with torch.no_grad():
            for m in alpha_modules:
                if hasattr(m, 'clamp_alpha'):
                    m.clamp_alpha()
                else:
                    # FALLBACK only: the fixed [1e-4, 20] ceiling was found to
                    # let LSQ alpha drift to ~20 while weights stayed ~0.06,
                    # zeroing the focal_mid encoder body (dissection 2026-06-03).
                    # Prefer a module with clamp_alpha() (data-driven bounds).
                    m.lsq_alpha.data.clamp_(min=1e-4, max=20.0)
        return gnorm

    return step


@register_ingredient
def _qat_codec_step_spec():
    return IngredientSpec(
        name="qat_codec", kind="step", config_cls=QatCodecStepConfig,
        build=_qat_codec_step,
    )
