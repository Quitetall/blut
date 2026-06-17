"""EMA ingredient specs (ADR 0050/0051). Importing this registers them.

``build_ingredient("ema", "avg_model", cfg, model=m)`` returns a torch
``AveragedModel`` (EMA multi-avg) wrapping ``m``, or ``None`` when disabled.
"""
from __future__ import annotations

from dataclasses import dataclass

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec


@dataclass(frozen=True)
class AvgModelConfig:
    decay: float = 0.999
    enabled: bool = True


def _build_avg_model(cfg, model):
    if not cfg.enabled:
        return None
    if not 0.0 < cfg.decay < 1.0:
        raise ValueError(f"ema decay must be in (0, 1), got {cfg.decay}")
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.decay))


@register_ingredient
def _avg_model():
    return IngredientSpec(
        name="avg_model", kind="ema", config_cls=AvgModelConfig,
        # EMA weights can become the saved best checkpoint, so enabling/disabling
        # it changes the trained artifact.
        cache_relevant=True,
        build=_build_avg_model,
    )
