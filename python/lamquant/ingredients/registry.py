"""The ingredient registry: register specs, build instances fail-closed.

``build_ingredient(kind, name, cfg, **extra)`` is the single entry point that
replaces the trainers' inline ``if/elif`` optimizer chains (ADR 0050/0051).
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any

from lamquant.ingredients.spec import IngredientSpec, coerce_config

_REGISTRY: dict[tuple[str, str], IngredientSpec] = {}


def register_ingredient(factory):
    """Decorator: register the ``IngredientSpec`` returned by a zero-arg factory.

    Mirrors ADR 0050's ``@register_optimizer``. Returns the factory unchanged.
    Raises on a duplicate ``(kind, name)``.
    """
    spec = factory()
    if not isinstance(spec, IngredientSpec):
        name = getattr(factory, "__name__", repr(factory))
        raise TypeError(f"{name} must return an IngredientSpec")
    key = (spec.kind, spec.name)
    if key in _REGISTRY:
        raise ValueError(f"duplicate ingredient {spec.kind}:{spec.name}")
    _REGISTRY[key] = spec
    return factory


def list_ingredients(kind: str | None = None) -> list[str]:
    """Sorted names of registered ingredients (optionally filtered to one kind)."""
    return sorted(n for (k, n) in _REGISTRY if kind is None or k == kind)


def get_spec(kind: str, name: str) -> IngredientSpec:
    """Look up a spec, fail-closed with the available names on a miss."""
    try:
        return _REGISTRY[(kind, name)]
    except KeyError:
        raise KeyError(
            f"unknown {kind} ingredient {name!r}; "
            f"registered {kind}: {list_ingredients(kind)}") from None


def _check_requires(spec: IngredientSpec) -> None:
    """Fail-closed capability gates: ``pkg:<name>`` must import; ``license:<name>``
    needs ``LAMU_LICENSE_<NAME>`` set (preserves ADR 0050's COSMOS hard-gate)."""
    for req in spec.requires:
        gate, _, val = req.partition(":")
        if gate == "pkg":
            if importlib.util.find_spec(val) is None:
                raise RuntimeError(
                    f"ingredient {spec.kind}:{spec.name} requires package "
                    f"{val!r} (not importable)")
        elif gate == "license":
            if not os.environ.get(f"LAMU_LICENSE_{val.upper()}"):
                raise RuntimeError(
                    f"ingredient {spec.kind}:{spec.name} is license-gated "
                    f"({val}); set LAMU_LICENSE_{val.upper()} to enable")
        else:
            raise ValueError(f"unknown requires token {req!r} on {spec.name}")


def build_ingredient(kind: str, name: str, cfg: Any = None, **extra):
    """Resolve + construct an ingredient, validating ``cfg`` fail-closed.

    For ``kind == "optimizer"`` (ADR 0050): pass ``named_params`` (an iterable of
    ``(name, Parameter)``); optional ``extra_groups`` are appended after the
    spec's routed groups, and any other kwarg is rejected. Returns a
    ``torch.optim.Optimizer``.

    For every other kind, the remaining ``extra`` kwargs are forwarded verbatim
    to the spec's ``build(config, **extra)`` — they are build-specific (a data
    ingredient may take ``device=``/``split=``), so a bad kwarg surfaces as a
    ``TypeError`` from ``build`` rather than a registry-level error.
    """
    spec = get_spec(kind, name)
    _check_requires(spec)
    config = coerce_config(cfg, spec.config_cls)

    if kind == "optimizer":
        if "named_params" not in extra:
            raise TypeError(
                f"build_ingredient('optimizer', {name!r}, ...) requires "
                "named_params=<iterable of (name, Parameter)>")
        named_params = list(extra.pop("named_params"))
        extra_groups = extra.pop("extra_groups", None)
        if extra:
            raise TypeError(
                f"unexpected kwargs for optimizer build: {sorted(extra)}")
        groups = list(spec.build_param_groups(named_params, config))
        if extra_groups:
            groups = groups + list(extra_groups)
        return spec.construct(groups, config)

    return spec.build(config, **extra)
