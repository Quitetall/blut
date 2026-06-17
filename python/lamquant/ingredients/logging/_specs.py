"""Logging ingredient specs (ADR 0050/0051). Importing this registers them.

A *logging* ingredient is the metric SINK a trainer's run() emits each epoch to.
It is built into a callable: ``build_ingredient("logging", "blut_metric", cfg)``
returns ``emit(metrics_dict, *, kind=..., phase=None) -> str``.

The body is transcribed VERBATIM from ``student/train_joint.py``'s ``_emit``
(the S6 / P4 runner contract): filter the dict to scalar (non-bool) fields, tag
``kind`` (+ optional ``phase``), and print one ``BLUT_METRIC <json>`` line that
the BLUT runner greps off stdout and folds into the queryable metric store
(``val_r`` is the headline; ``epoch`` is the coordinate). It is ``flush``ed so a
live tail/TUI sees it. The emitted line is byte-identical to the inline emit.

``cache_relevant=False`` — logging is an observability sink; selecting it does
not change the trained artifact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec


@dataclass(frozen=True)
class BlutMetricConfig:
    pass


def _build_blut_metric(cfg):
    def emit(metrics, *, kind="epoch", phase=None):
        """Emit one runner-parseable ``BLUT_METRIC <json>`` line.

        Args:
            metrics: a dict of per-epoch metrics (the trainer's ``report.to_dict()``
                output, with ``alpha_per_layer`` already popped). Only scalar,
                non-bool fields are forwarded.
            kind: the event tag (``'epoch'`` for the per-epoch line).
            phase: optional phase tag (e.g. WSD phase); omitted when None.

        Returns the emitted JSON string (also printed + flushed to stdout).
        """
        # S6 (P4 contract): a runner-parseable per-epoch metric line. The
        # BLUT runner greps `BLUT_METRIC ` off stdout and forwards the JSON
        # object as a StageEvent::StageStep, which folds into the queryable
        # metric store (val_r is the headline; `epoch` is the coordinate).
        # Scalars only + the phase tag; flushed so a live tail/TUI sees it.
        # (numeric fields from d, then the metadata keys kind/phase.)
        payload = {k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        payload['kind'] = kind
        if phase is not None:
            payload['phase'] = phase
        line = 'BLUT_METRIC ' + json.dumps(payload)
        print(line, flush=True)
        return line

    return emit


@register_ingredient
def _blut_metric_spec():
    return IngredientSpec(
        name="blut_metric", kind="logging", config_cls=BlutMetricConfig,
        cache_relevant=False,
        build=_build_blut_metric,
    )
