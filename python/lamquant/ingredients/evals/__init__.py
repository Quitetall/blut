"""Eval ingredients (ADR 0050/0051) — held-out evaluation primitives.

Two specs, both ``kind="eval"`` and ``cache_relevant=False`` (running an eval
does not change the trained artifact):

  * ``joint_codec`` — wraps ``student.train_joint.validate_joint``, the SAME
    end-to-end codec eval the PCCP gate runs (``eval_codec_pccp`` imports it).
    WRAPPED, never relocated: one source of truth for trainer + gate.
  * ``four_state`` — bundles ``snn.train_4state_controller.four_state_metrics``
    + the lexicographic ``selection_key`` (ADR 0029 slide-killer) into one
    object so a recipe assembles per-epoch 4-state metrics + feasibility-first
    checkpoint selection from a single ingredient.
"""
