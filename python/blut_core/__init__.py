"""blut_core — domain-agnostic primitives for the BLUT core cookbook.

Implement-once building blocks every training run needs, reusable by ANY
cookbook (lamquant, lamu) — no LamQuant/EEG coupling:

- ``runctx``       — run identity + filesystem anchors from the BLUT env (P10)
- ``MetricLog``    — atomic per-epoch CSV/Parquet metric writer
- ``read_metric``  — verbatim metric/log reader (no LLM in the path; ADR 0038)
- ``status``       — emit/read the StatusUpdate wire protocol (P4)
- ``RunManifest``  — run provenance, written even on crash (P8)
- ``checkpoint``   — corruption-safe save/load + resume payload contract (P7)
- ``sysgauge``     — best-effort GPU/host snapshot for the metric stream (P10)

``torch`` is lazy-imported only inside ``checkpoint``/``sysgauge`` functions, so
``import blut_core`` stays cheap and dependency-light. See ``blut/docs/metrics.md``
and decisions/0037, 0044.
"""
from . import runctx, status, sysgauge, checkpoint
from .metric_log import MetricLog
from .run_manifest import RunManifest

# read_metric is a CLI module (`python -m blut_core.read_metric`); it is NOT
# eager-imported here (that triggers a runpy double-import warning under -m).
# `from blut_core import read_metric` still works (submodule import).

__all__ = [
    "runctx", "status", "sysgauge", "checkpoint", "read_metric",
    "MetricLog", "RunManifest",
]
