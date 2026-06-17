"""Checkpoint ingredients (ADR 0050/0051) — the trainers' save + safety + resume
primitives, extracted verbatim from the inline trainer code so the registry is
the single construction point.

Three specs (all ``kind="checkpoint"``, ``cache_relevant=False`` — choosing how
a run persists its weights does not change the trained artifact):

  * ``atomic_save`` — the tmp + ``os.replace`` atomic ``torch.save`` from
    ``snn/train_4state_controller.py`` (with the optional module-global
    single-worker async executor + state-dict→CPU hoist);
  * ``manager``    — wraps ``student/checkpoint_manager.CheckpointManager``;
  * ``durable_resume`` — wraps ``student/durable_resume.DurableResume`` (or
    ``None`` when no resume dir is configured, matching train_joint).
"""
