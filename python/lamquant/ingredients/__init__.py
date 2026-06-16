"""LamQuant training *ingredients* — the sub-stage primitives a trainer's
``run()`` is assembled from (ADR 0051).

An *ingredient* is a typed, registry'd Python primitive (optimizer, loss,
schedule, dataset, step, …) that lives **inside** one trainer process,
below the Rust ``Stage``. ``optimizers/`` is the first ingredient kind
(ADR 0050); the ``IngredientSpec`` registry + the remaining kinds land in
later phases of the cookbook rebuild.
"""
