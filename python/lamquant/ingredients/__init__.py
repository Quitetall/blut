"""LamQuant training *ingredients* — the sub-stage primitives a trainer's
``run()`` is assembled from (ADR 0051).

An *ingredient* is a typed, registry'd Python primitive (optimizer, loss,
schedule, dataset, step, …) that lives **inside** one trainer process,
below the Rust ``Stage``. ``optimizers/`` is the first ingredient kind
(ADR 0050); the ``IngredientSpec`` registry + the remaining kinds land in
later phases of the cookbook rebuild.
"""

from lamquant.ingredients.registry import (
    build_ingredient,
    get_spec,
    list_ingredients,
    register_ingredient,
)
from lamquant.ingredients.spec import KINDS, IngredientSpec

# Register the built-in ingredient specs on package import.
from lamquant.ingredients.optimizers import _specs as _optimizer_specs  # noqa: F401
from lamquant.ingredients.schedules import _specs as _schedule_specs  # noqa: F401
from lamquant.ingredients.ema import _specs as _ema_specs  # noqa: F401
from lamquant.ingredients.steps import _specs as _step_specs  # noqa: F401

__all__ = [
    "build_ingredient",
    "get_spec",
    "list_ingredients",
    "register_ingredient",
    "IngredientSpec",
    "KINDS",
]
