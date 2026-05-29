"""BLUT-owned LamQuant training + preprocessing tree (MOVE-B).

After the Neural <-> BLUT boundary migration (2026-05-29), ALL training
loops and preprocessing scripts live here, under the PUBLIC ``blut/``
submodule. The neural codec MODEL DEFINITIONS (for inference) live in the
PRIVATE ``lamquant_neural`` package (LamQuant-Neural repo); the lossless
Rust codec is the PUBLIC ``lamquant_core`` PyO3 extension (LamQuant-Lossless).

Sub-packages:
  * ``lamquant.dataset``  — preprocessing, manifest + split builders, label gen
  * ``lamquant.snn``      — Mamba-SNN training + seizure-aware LMA dataset
  * ``lamquant.oracle``   — FP32 teacher / L3-teacher training
  * ``lamquant.student``  — ternary student training, harden, eval
  * ``lamquant.decoder``  — Vocos decoder training, discriminators
  * ``lamquant.common``   — shared training DTOs (data_types, metrics, utils)

PUBLIC -> PRIVATE dependency
----------------------------
The ``lamquant.*`` training scripts import the neural model definitions
from ``lamquant_neural`` (a PRIVATE wheel). BLUT is PUBLIC; it declares
``lamquant-neural`` only as an OPTIONAL extra (``pip install blut[lamquant]``).
Generic BLUT recipes (sft / dpo / distill) do NOT need it. Use
:func:`require_lamquant_neural` at the top of any module that imports
``lamquant_neural`` so an absent private wheel degrades with a clear,
actionable error instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations


def require_lamquant_neural():
    """Import + return the ``lamquant_neural`` package, or raise a clear
    error if the private wheel is not installed.

    The ``lamquant.*`` training scripts depend on the neural model
    definitions (encoder / blocks / snn / vocos_decoder / heads) that
    live in the PRIVATE ``lamquant_neural`` package. When that wheel is
    absent (a public BLUT checkout running only the generic sft/dpo/
    distill recipes), importing it fails with a bare ModuleNotFoundError
    that does not explain the optional-extra contract. This raises a
    ``RuntimeError`` naming the extra to install instead.
    """
    try:
        import lamquant_neural  # noqa: F401

        return lamquant_neural
    except ImportError as exc:  # pragma: no cover - exercised in CI both ways
        raise RuntimeError(
            "this LamQuant training/preprocess module requires the neural "
            "model definitions from the private `lamquant_neural` package, "
            "which is not installed. Install the optional extra:\n"
            "    pip install 'blut[lamquant]'\n"
            "(or `pip install -e LamQuant-Neural` from a meta-repo checkout). "
            "Generic BLUT recipes (sft/dpo/distill) do not need it."
        ) from exc
