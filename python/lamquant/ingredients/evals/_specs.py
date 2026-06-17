"""Eval ingredient specs (ADR 0050/0051). Importing this registers them.

Both specs WRAP existing functions (no relocation) so the trainer, the PCCP
gate, and a recipe all call byte-identical code:

  * ``build_ingredient("eval", "joint_codec", cfg)`` returns
    ``eval_fn(model, val_ds, device, *, quantize=None, per_category=None)``
    forwarding to ``student.train_joint.validate_joint``.
  * ``build_ingredient("eval", "four_state", cfg)`` returns an object with
    ``.metrics(cm) -> dict`` and ``.select_key(m, alpha) -> tuple`` wrapping
    ``snn.train_4state_controller.{four_state_metrics, selection_key}``.

Heavy deps (the neural wheel, torch) are imported lazily inside ``build`` so
importing this module to register the specs stays cheap.
"""
from __future__ import annotations

from dataclasses import dataclass

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec


# ---------------------------------------------------------------------------
# (1) joint_codec — the shared end-to-end codec eval (also the PCCP gate eval).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointEvalConfig:
    """Mirrors ``validate_joint``'s keyword args (the ones a recipe pins).

    ``quantize`` / ``per_category`` may be overridden per call (the gate runs
    both quantized and unquantized passes); the rest are fixed at build time.
    """
    quantize: bool = True
    batch_size: int = 64
    per_band_sample: int = 512
    amp: bool = True
    per_category: bool = False
    channel_agnostic: bool = False
    variable_n: bool = False
    n_range: tuple = (8, 21)


def _build_joint_codec_eval(cfg):
    def eval_fn(model, val_ds, device, *, quantize=None, per_category=None):
        # Lazy: validate_joint's module pulls in torch + the neural wheel.
        from lamquant.student.train_joint import validate_joint
        return validate_joint(
            model, val_ds, device,
            quantize=cfg.quantize if quantize is None else quantize,
            batch_size=cfg.batch_size,
            per_band_sample=cfg.per_band_sample,
            amp=cfg.amp,
            per_category=(cfg.per_category if per_category is None
                          else per_category),
            channel_agnostic=cfg.channel_agnostic,
            variable_n=cfg.variable_n,
            n_range=cfg.n_range,
        )

    return eval_fn


@register_ingredient
def _joint_codec_eval():
    return IngredientSpec(
        name="joint_codec", kind="eval", config_cls=JointEvalConfig,
        # Eval never mutates the saved artifact.
        cache_relevant=False,
        build=_build_joint_codec_eval,
    )


# ---------------------------------------------------------------------------
# (2) four_state — per-epoch 4-state metrics + feasibility-first selection.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FourStateEvalConfig:
    """No tunables: the metric/selection behaviour is fixed by ADR 0029."""


class _FourStateEval:
    """Bundles the two wrapped functions behind a stable method surface.

    ``.metrics(cm)``     -> ``four_state_metrics(cm)`` (per-epoch logged dict).
    ``.select_key(m, a)``-> ``selection_key(m, a)`` — the lexicographic 2-tuple,
                            returned UNCHANGED (ADR 0029: never a scalar / a
                            max-val_r — that is the operating-point slide the
                            tuple was designed to kill). ``alpha`` is a call-time
                            arg (the training-feasibility floor).
    """

    def metrics(self, cm) -> dict:
        from lamquant.snn.train_4state_controller import four_state_metrics
        return four_state_metrics(cm)

    def select_key(self, m: dict, alpha: float) -> tuple:
        from lamquant.snn.train_4state_controller import selection_key
        return selection_key(m, alpha)


def _build_four_state_eval(cfg):
    return _FourStateEval()


@register_ingredient
def _four_state_eval():
    return IngredientSpec(
        name="four_state", kind="eval", config_cls=FourStateEvalConfig,
        cache_relevant=False,
        build=_build_four_state_eval,
    )
