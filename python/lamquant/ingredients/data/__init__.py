"""Data ingredients (ADR 0050/0051) — the LMA-direct dataset CONSTRUCTION each
trainer's run() materialises before its epoch loop.

Three specs, all ``kind="data"`` and ``cache_relevant=True`` (the corpus a
stage trains on is part of the trained artifact's identity):

  * ``lma_snn``      — the seizure-aware ``snn.lma_dataset.LmaDataset`` shared by
                       the 4-state controller (train+val) and the SSL pretrain
                       (single split). The byte-identical ``expand_lma_roots``
                       helper both trainers copy-pasted now lives here as the
                       single source of truth, with the shared DataLoader-worker
                       preamble ``build_lma_dataloader_kwargs``.
  * ``lma_l3``       — ``lamquant_codec.training.LmaL3Dataset`` shared by the MAE
                       pretrain + the L3 teacher.
  * ``lma_typed_l3`` — ``student/lma_typed_adapter.LmaTypedL3Dataset`` built
                       exactly as ``train_joint`` builds its train+val pair,
                       including the mandatory pre-construction decode-cache
                       priming (the epoch-2-OOM footgun).

Only the dataset construction is the ingredient — the per-epoch iteration
(prefetch / final sampler-vs-shuffle DataLoader) stays in each trainer.
"""

from lamquant.ingredients.data._specs import (  # noqa: F401
    build_lma_dataloader_kwargs,
    expand_lma_roots,
)

__all__ = [
    "expand_lma_roots",
    "build_lma_dataloader_kwargs",
]
