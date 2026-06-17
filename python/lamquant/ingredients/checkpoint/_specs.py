"""Checkpoint ingredient specs (ADR 0050/0051). Importing this registers them.

Three checkpoint primitives, all extracted VERBATIM from the trainers:

  build_ingredient("checkpoint", "atomic_save", cfg) -> save(payload, path)
  build_ingredient("checkpoint", "manager", cfg, model=m, ckpt_path=p, ...) -> CheckpointManager
  build_ingredient("checkpoint", "durable_resume", cfg, resume_dir=d, ...) -> DurableResume | None

All three are ``cache_relevant=False`` — how a run persists its weights does not
change the trained artifact.
"""
from __future__ import annotations

import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec


# ===========================================================================
# (1) atomic_save — tmp + os.replace atomic torch.save.
#
# Transcribed verbatim from snn/train_4state_controller.py:540-571
# (_SAVE_EXECUTOR / _SAVE_LOCK / _ensure_save_executor / _state_dict_to_cpu /
# _atomic_torch_save / _async_save). The executor is a MODULE-GLOBAL singleton
# guarded by a lock + an atexit shutdown — one per process, never one per build.
# ===========================================================================

_SAVE_EXECUTOR: Optional[ThreadPoolExecutor] = None
_SAVE_LOCK = threading.Lock()


def _ensure_save_executor() -> ThreadPoolExecutor:
    global _SAVE_EXECUTOR
    with _SAVE_LOCK:
        if _SAVE_EXECUTOR is None:
            _SAVE_EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ckpt-atomic")
            atexit.register(_SAVE_EXECUTOR.shutdown, wait=True)
    return _SAVE_EXECUTOR


def _state_dict_to_cpu(sd):
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().to("cpu", copy=True) if hasattr(v, "detach") else v
    return out


def _atomic_torch_save(payload: dict, path: str) -> None:
    """torch.save to a temp file in the same dir, then atomic rename — a
    mid-write kill leaves the prior checkpoint intact rather than a truncated
    one (MiMo a04bc8e)."""
    import torch
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _async_save(payload: dict, path: str):
    return _ensure_save_executor().submit(_atomic_torch_save, payload, path)


@dataclass(frozen=True)
class AtomicSaveConfig:
    async_: bool = False
    state_dict_to_cpu: bool = False


def _build_atomic_save(cfg: AtomicSaveConfig):
    """Return ``save(payload: dict, path: str) -> None``.

    With ``state_dict_to_cpu`` the ``"state_dict"`` entry (if present) is hoisted
    to CPU first (off the hot device before the write). With ``async_`` the write
    is handed to the module-global single-worker executor; otherwise it runs
    inline (the os.replace makes either path crash-safe)."""
    def save(payload: dict, path: str) -> None:
        if cfg.state_dict_to_cpu and isinstance(payload, dict) \
                and "state_dict" in payload:
            payload = dict(payload)
            payload["state_dict"] = _state_dict_to_cpu(payload["state_dict"])
        if cfg.async_:
            _async_save(payload, path)
        else:
            _atomic_torch_save(payload, path)

    return save


@register_ingredient
def _atomic_save_spec():
    return IngredientSpec(
        name="atomic_save", kind="checkpoint", config_cls=AtomicSaveConfig,
        cache_relevant=False,
        build=_build_atomic_save,
    )


# ===========================================================================
# (2) manager — wraps student/checkpoint_manager.CheckpointManager (UNCHANGED).
#
# GuardConfig's 6 floats are FLATTENED into the ingredient config (so
# coerce_config's config_cls(**dict) works — a nested dataclass field would
# break it); build() reconstructs GuardConfig(**those) before passing it in.
# Defaults copied EXACTLY from GuardConfig.
# ===========================================================================

@dataclass(frozen=True)
class ManagerConfig:
    # The 6 GuardConfig fields, flattened. Defaults are byte-identical to
    # GuardConfig's so an all-defaults build reproduces the production preset.
    r_plateau_patience: int = 50
    alpha_max_safe: float = 5.0
    alpha_min_safe: float = 1e-4
    smoke_check_tolerance: float = 1e-3
    improvement_eps: float = 1e-3
    prd_tiebreak_eps: float = 0.5


def _build_manager(cfg: ManagerConfig, *, model, ckpt_path, ckpt_dir=None,
                   device=None, smoke_input=None, alpha_log_csv=None,
                   provenance=None):
    from lamquant.student.checkpoint_manager import (
        CheckpointManager, GuardConfig)
    guard = GuardConfig(
        r_plateau_patience=cfg.r_plateau_patience,
        alpha_max_safe=cfg.alpha_max_safe,
        alpha_min_safe=cfg.alpha_min_safe,
        smoke_check_tolerance=cfg.smoke_check_tolerance,
        improvement_eps=cfg.improvement_eps,
        prd_tiebreak_eps=cfg.prd_tiebreak_eps,
    )
    return CheckpointManager(
        model, ckpt_path,
        ckpt_dir=ckpt_dir,
        device=device,
        smoke_input=smoke_input,
        alpha_log_csv=alpha_log_csv,
        guard=guard,
        provenance=provenance,
    )


@register_ingredient
def _manager_spec():
    return IngredientSpec(
        name="manager", kind="checkpoint", config_cls=ManagerConfig,
        cache_relevant=False,
        build=_build_manager,
    )


# ===========================================================================
# (3) durable_resume — wraps student/durable_resume.DurableResume (UNCHANGED).
#
# Returns None when resume_dir is falsy — verbatim the train_joint.py:437
# ternary: `DurableResume(resume_dir, run_id, resume_key) if resume_dir else None`.
# ===========================================================================

@dataclass(frozen=True)
class DurableResumeConfig:
    pass


def _build_durable_resume(cfg: DurableResumeConfig, *, resume_dir,
                          run_id="", resume_key=""):
    from lamquant.student.durable_resume import DurableResume
    return DurableResume(resume_dir, run_id, resume_key) if resume_dir else None


@register_ingredient
def _durable_resume_spec():
    return IngredientSpec(
        name="durable_resume", kind="checkpoint",
        config_cls=DurableResumeConfig,
        cache_relevant=False,
        build=_build_durable_resume,
    )
