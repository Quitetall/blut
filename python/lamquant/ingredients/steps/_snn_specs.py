"""SNN step ingredient (ADR 0050/0051).

The nan-skip + backward + clip + optimizer.step + post-step clamp_ssm_params
sequence is byte-identical in BOTH SNN trainers (pretrain_ssl_tueg +
train_4state_controller), differing only in (clipped-params iterable, clamp-target
module) — which become call-time args. This is the strongest cross-trainer dedup
in the decomposition. clamp_ssm_params is imported lazily (inside build) so the
steps package stays importable without the neural wheel.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec


@dataclass(frozen=True)
class SnnSsmStepConfig:
    grad_clip_norm: float = 1.0


def _snn_ssm_step(cfg):
    # Lazy: only the build call (when a trainer wires the step) needs the wheel.
    from lamquant_neural.models.mamba_ssm_minimal import clamp_ssm_params

    def step(loss, optimizer, clip_params, clamp_module):
        """Returns did_step (False = nan-skipped). The caller keeps its own
        nan_skips counter + `continue`, driven by a False return."""
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            return False
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip_norm)
        optimizer.step()
        # clamp AFTER the step, inside no_grad — the B1 float32-safe SSM band.
        with torch.no_grad():
            clamp_ssm_params(clamp_module)
        return True

    return step


@register_ingredient
def _snn_ssm_step_spec():
    return IngredientSpec(
        name="snn_ssm", kind="step", config_cls=SnnSsmStepConfig,
        build=_snn_ssm_step,
    )
