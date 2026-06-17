"""Tests for the logging ingredient registry (ADR 0050/0051).

One ``kind="logging"`` spec:

  * ``blut_metric`` — the runner-parseable ``BLUT_METRIC <json>`` stdout line
                      (``train_joint``'s ``_emit`` S6/P4 contract).

Equivalence is proven against a VERBATIM inline copy of the trainer emit block:
same input dict -> byte-identical emitted JSON line (pure stdlib, no wheels).
"""
from __future__ import annotations

import json

# Importing this module registers the logging spec.
import lamquant.ingredients.logging._specs  # noqa: F401
from lamquant.ingredients import build_ingredient, get_spec, list_ingredients


def _inline_emit(d, phase):
    """Verbatim copy of student/train_joint.py::_emit BLUT_METRIC block."""
    payload = {k: v for k, v in d.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    payload['kind'] = 'epoch'
    if phase is not None:
        payload['phase'] = phase
    return 'BLUT_METRIC ' + json.dumps(payload)


_SAMPLE = {
    "val_r": 0.4206, "epoch": 7, "loss": 0.0123,
    "is_best": True,            # bool -> excluded
    "loss_domain": "fullband",  # str  -> excluded
    "alpha_per_layer": [1, 2],  # already popped upstream; non-scalar excluded
}


def test_blut_metric_registered():
    assert "blut_metric" in list_ingredients("logging")


def test_logging_spec_not_cache_relevant():
    # Observability sink — selecting it does not change the trained artifact.
    assert get_spec("logging", "blut_metric").cache_relevant is False


def test_emit_line_equals_inline_with_phase(capsys):
    emit = build_ingredient("logging", "blut_metric", {})
    line = emit(_SAMPLE, kind="epoch", phase="warmup")
    exp = _inline_emit(_SAMPLE, "warmup")
    assert line == exp
    out = capsys.readouterr().out.strip()
    assert out == exp                      # printed line == returned line == inline
    assert out.startswith("BLUT_METRIC ")
    # The emitted JSON has the scalars + kind + phase, and drops bool/str/list.
    payload = json.loads(out.split(" ", 1)[1])
    assert payload == {"val_r": 0.4206, "epoch": 7, "loss": 0.0123,
                       "kind": "epoch", "phase": "warmup"}


def test_emit_line_equals_inline_no_phase():
    emit = build_ingredient("logging", "blut_metric", {})
    line = emit(_SAMPLE, kind="epoch", phase=None)
    assert line == _inline_emit(_SAMPLE, None)
    assert "phase" not in json.loads(line.split(" ", 1)[1])


def test_emit_excludes_bool_and_nonscalar(capsys):
    emit = build_ingredient("logging", "blut_metric", {})
    emit({"x": 1, "flag": False, "name": "y", "vec": [1, 2]})
    payload = json.loads(capsys.readouterr().out.strip().split(" ", 1)[1])
    assert "flag" not in payload and "name" not in payload and "vec" not in payload
    assert payload["x"] == 1
