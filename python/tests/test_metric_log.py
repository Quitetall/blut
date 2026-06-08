"""Tests for lamquant.common.metric_log.MetricLog — the reviewer-readable,
crash-safe, live per-epoch metric stream (BLUT L1 observability).

Run: cd blut/python && python -m pytest tests/test_metric_log.py -q
"""
from __future__ import annotations

import csv
import dataclasses
import importlib
import os

import pytest

from lamquant.common.metric_log import MetricLog

_HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


def _force_csv(m: MetricLog) -> MetricLog:
    """Pin a MetricLog to the CSV backend deterministically (env-independent)."""
    m._backend = "csv"
    m.path = m.log_dir / f"metrics_{m.run_id}.csv"
    return m


def _csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _csv_header(path):
    with open(path) as f:
        return set(next(csv.reader(f)))


def test_csv_fallback_readable_midrun(tmp_path):
    """CSV file is a COMPLETE, valid file after EVERY append (live readability),
    not just at close()."""
    m = _force_csv(MetricLog("run1", tmp_path))
    m.append({"epoch": 1, "val_r": 0.5})
    # readable after the FIRST append
    rows = _csv_rows(m.path)
    assert len(rows) == 1 and rows[0]["epoch"] == "1"
    m.append({"epoch": 2, "val_r": 0.7})
    rows = _csv_rows(m.path)
    assert len(rows) == 2
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert set(rows[0].keys()) == {"epoch", "val_r"}
    # no leftover tmp
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


@pytest.mark.skipif(not _HAS_PYARROW, reason="pyarrow not installed")
def test_parquet_roundtrip(tmp_path):
    import pyarrow.parquet as pq

    m = MetricLog("run2", tmp_path)
    assert m._backend == "parquet"
    for i in range(3):
        m.append({"epoch": i, "val_r": 0.1 * i})
    out = pq.read_table(m.path).to_pylist()
    assert len(out) == 3
    assert [r["epoch"] for r in out] == [0, 1, 2]
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())


def test_append_never_raises(tmp_path):
    """A row with an unserializable value must not crash the run."""
    m = _force_csv(MetricLog("run3", tmp_path))
    # set + arbitrary object — coerced to str, never raises
    m.append({"epoch": 1, "weird": {1, 2, 3}, "obj": object()})
    rows = _csv_rows(m.path)
    assert len(rows) == 1  # still wrote a complete file


def test_schema_matches_epochreport(tmp_path):
    """The metric stream must preserve every EpochReport field (minus
    alpha_per_layer) as a column — guards warm/qat emit-site schema drift."""
    from lamquant.common.training_types import EpochReport

    kwargs = {}
    for f in dataclasses.fields(EpochReport):
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:
            continue
        t = str(f.type)
        if "int" in t:
            kwargs[f.name] = 0
        elif "float" in t:
            kwargs[f.name] = 0.0
        elif "bool" in t:
            kwargs[f.name] = False
        elif "str" in t:
            kwargs[f.name] = ""
        else:
            kwargs[f.name] = None
    try:
        ep = EpochReport(**kwargs)
        d = ep.to_dict()
    except Exception as e:
        pytest.skip(f"EpochReport not trivially constructible for schema check: {e}")
    d.pop("alpha_per_layer", None)

    m = _force_csv(MetricLog("run4", tmp_path))
    m.append(d)
    cols = _csv_header(m.path)
    missing = set(map(str, d.keys())) - cols
    assert not missing, f"metric stream dropped EpochReport fields: {missing}"
