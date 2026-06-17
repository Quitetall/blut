"""Sampler ingredients (ADR 0050/0051) — the per-batch reconstruction MASK a
self-supervised trainer's run() draws each step.

Two specs, both ``kind="sampler"`` and ``cache_relevant=True`` (the masking
regime is part of the trained artifact's identity):

  * ``span_mask``  — contiguous time-span masking (SSL pretrain, ``make_span_mask``).
  * ``patch_mask`` — contiguous patch masking expanded across channels
                     (MAE pretrain, ``create_mask``).
"""
