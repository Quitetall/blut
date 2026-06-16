"""IngredientSpec — the typed contract for a training sub-stage primitive.

An *ingredient* (ADR 0051) is a registry'd Python primitive a trainer's
``run()`` is assembled from. ``IngredientSpec`` generalizes ADR 0050's
``OptimizerSpec`` across the ingredient kinds: the ``optimizer`` kind keeps
0050's contract (``build_param_groups`` owns the param routing, ``construct``
builds the optimizer); other kinds use the single ``build`` callable.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

# Canonical ingredient kinds (ADR 0051). A `cache_relevant` ingredient's
# selection MUST ride a hashed Args/extra_args field (never an env var or a
# Python-side default), so two materially different trainings never collide on
# one stage cache key.
KINDS: frozenset[str] = frozenset({
    "data", "sampler", "preprocess", "model", "forward", "loss",
    "optimizer", "scheduler", "step", "ema", "eval", "checkpoint", "logging",
})


@dataclass(frozen=True)
class IngredientSpec:
    """One registered ingredient.

    Required: ``name``, ``kind`` (in :data:`KINDS`), and ``config_cls`` — a
    dataclass whose fields mirror the primitive's real hyperparameters,
    validated fail-closed at build time.

    ``optimizer`` kind (ADR 0050): set ``build_param_groups(named_params, cfg)
    -> list[dict]`` and ``construct(param_groups, cfg) -> Optimizer``. Other
    kinds: set ``build(cfg, **extra) -> object``.

    ``cache_relevant`` flags whether selecting this ingredient changes the
    trained artifact (default True). ``requires`` are capability gates
    (``pkg:<name>`` / ``license:<name>``) enforced fail-closed at build time.
    """

    name: str
    kind: str
    config_cls: type
    build: Callable[..., Any] | None = None
    requires: tuple[str, ...] = ()
    cache_relevant: bool = True
    # ADR 0050: flags which optimizers are safe to wrap when the optimizer step
    # itself is compiled (fused/capturable + CUDA graphs). Reserved — read by
    # the training loop in a later phase, not by the registry.
    supports_compiled_step: bool = True
    # optimizer-kind contract (ADR 0050)
    build_param_groups: Callable[..., list[dict]] | None = None
    construct: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"unknown ingredient kind {self.kind!r}; "
                f"expected one of {sorted(KINDS)}")
        if not dataclasses.is_dataclass(self.config_cls):
            raise TypeError(
                f"config_cls for {self.kind}:{self.name} must be a dataclass")
        if self.kind == "optimizer":
            if self.build_param_groups is None or self.construct is None:
                raise ValueError(
                    f"optimizer ingredient {self.name!r} must set "
                    "build_param_groups + construct")
            if self.build is not None:
                raise ValueError(
                    f"optimizer ingredient {self.name!r} must not set build() "
                    "(it uses build_param_groups + construct)")
        else:
            if self.build is None:
                raise ValueError(
                    f"ingredient {self.kind}:{self.name} must set build()")
            if self.build_param_groups is not None or self.construct is not None:
                raise ValueError(
                    f"non-optimizer ingredient {self.kind}:{self.name} must not "
                    "set build_param_groups/construct")


def coerce_config(cfg: Any, config_cls: type):
    """Validate ``cfg`` against ``config_cls`` (fail-closed).

    Accepts an instance of ``config_cls`` (returned as-is), ``None``/``{}``
    (all-defaults), or a dict (constructs the dataclass — an unknown key or a
    missing required field raises ``ValueError``).
    """
    if isinstance(cfg, config_cls):
        return cfg
    if cfg is None:
        cfg = {}
    if isinstance(cfg, dict):
        try:
            return config_cls(**cfg)
        except TypeError as e:
            raise ValueError(
                f"invalid config for {config_cls.__name__}: {e}") from e
    raise TypeError(
        f"config must be a dict or {config_cls.__name__}, "
        f"got {type(cfg).__name__}")
