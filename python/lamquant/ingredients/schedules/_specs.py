"""Scheduler ingredient specs (ADR 0050/0051). Importing this registers them.

A scheduler ingredient wraps an already-built optimizer:
``build_ingredient("scheduler", "wsd", cfg, optimizer=opt)``.
"""
from __future__ import annotations

from dataclasses import dataclass

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec
from lamquant.ingredients.schedules.wsd import WSDScheduler


@dataclass(frozen=True)
class WsdConfig:
    total_epochs: int
    peak_lr: float
    warmup_frac: float = 0.05
    decay_frac: float = 0.10
    min_lr: float = 1e-6
    warmup_kind: str = "cosine"


@register_ingredient
def _wsd():
    return IngredientSpec(
        name="wsd", kind="scheduler", config_cls=WsdConfig,
        build=lambda cfg, optimizer: WSDScheduler(
            optimizer, total_epochs=cfg.total_epochs, peak_lr=cfg.peak_lr,
            warmup_frac=cfg.warmup_frac, decay_frac=cfg.decay_frac,
            min_lr=cfg.min_lr, warmup_kind=cfg.warmup_kind),
    )
