"""Model ingredients (ADR 0050/0051) — the network CONSTRUCTION each trainer's
run() does once before its epoch loop.

Three specs, all ``kind="model"`` and ``cache_relevant=True`` (the architecture
is part of the trained artifact's identity):

  * ``l3_teacher``  — ``oracle/train_l3_teacher.py``'s ``L3Teacher``.
  * ``mae_encoder`` — ``student/pretrain_mae.py``'s encoder + MAE prediction head.
  * ``joint_codec`` — ``student/train_joint.py``'s ``build_default_joint`` codec.
"""
