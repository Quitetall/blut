"""Tests for read_metric — the verbatim metric reader.

Anti-confabulation contract pinned here (futureproof style — assert against
frozen synthetic fixtures, never impl-derived values):
  1. exact values are returned byte-for-byte (strings, as the CSV stores them);
  2. a missing key yields an EXPLICIT error + zero rows — never an invented value;
  3. malformed lines are reported, not silently dropped or guessed;
  4. exit code is 2 on no-data, 0 on data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # blut/python
from lamquant.common import read_metric as rm  # noqa: E402


def _write_csv(p: Path) -> Path:
    p.write_text(
        "run_id,epoch,timestamp,val_r,train_loss\n"
        "r1,0,100.0,0.10,5.0\n"
        "r1,1,200.0,0.20,4.0\n"
        "r1,2,300.0,0.30,3.0\n"
    )
    return p


def test_csv_verbatim_key(tmp_path):
    csv = _write_csv(tmp_path / "metrics_r1.csv")
    res = rm.read_csv_metrics(csv, ["val_r"], None)
    assert res["source"] == "csv"
    assert res["available_keys"] == ["run_id", "epoch", "timestamp", "val_r", "train_loss"]
    # exact stored strings, with index columns attached
    assert [r["val_r"] for r in res["rows"]] == ["0.10", "0.20", "0.30"]
    assert [r["epoch"] for r in res["rows"]] == ["0", "1", "2"]
    assert res["errors"] == []
    assert res["n"] == 3


def test_missing_key_errors_never_fabricates(tmp_path):
    csv = _write_csv(tmp_path / "metrics_r1.csv")
    res = rm.read_csv_metrics(csv, ["val_r_FAKE"], None)
    assert res["rows"] == []           # NO invented row
    assert res["n"] == 0
    assert len(res["errors"]) == 1
    assert "val_r_FAKE" in res["errors"][0]
    assert "val_r" in res["errors"][0]  # lists the real available keys


def test_partial_missing_keeps_real_drops_fake(tmp_path):
    csv = _write_csv(tmp_path / "metrics_r1.csv")
    res = rm.read_csv_metrics(csv, ["val_r", "nope"], None)
    assert [r["val_r"] for r in res["rows"]] == ["0.10", "0.20", "0.30"]
    assert any("nope" in e for e in res["errors"])
    assert all("nope" not in r for r in res["rows"])  # fake key never appears


def test_last_n(tmp_path):
    csv = _write_csv(tmp_path / "metrics_r1.csv")
    res = rm.read_csv_metrics(csv, ["val_r"], 2)
    assert [r["val_r"] for r in res["rows"]] == ["0.20", "0.30"]


def test_no_file_errors(tmp_path):
    res = rm.read_csv_metrics(tmp_path / "metrics_absent.csv", ["val_r"], None)
    assert res["rows"] == []
    assert res["errors"] and "no file" in res["errors"][0]


def test_status_jsonl_kind_filter_and_verbatim(tmp_path):
    sj = tmp_path / "status.jsonl"
    sj.write_text(
        json.dumps({"kind": "step", "step": 1, "loss": 5.0}) + "\n"
        + json.dumps({"kind": "eval", "step": 1, "eval_loss": 4.5}) + "\n"
        + json.dumps({"kind": "step", "step": 2, "loss": 4.0}) + "\n"
    )
    res = rm.read_status_jsonl(sj, "step", None)
    assert res["n"] == 2
    assert [r["loss"] for r in res["rows"]] == [5.0, 4.0]
    assert sorted(rm.read_status_jsonl(sj, None, None)["available_keys"]) == ["eval", "step"]


def test_status_malformed_line_reported_not_guessed(tmp_path):
    sj = tmp_path / "status.jsonl"
    sj.write_text(
        json.dumps({"kind": "step", "step": 1}) + "\n"
        + "{not valid json\n"
        + json.dumps({"kind": "done", "final_loss": 0.1}) + "\n"
    )
    res = rm.read_status_jsonl(sj, None, None)
    assert res["n"] == 2  # the two valid lines
    assert len(res["errors"]) == 1 and "malformed" in res["errors"][0]


def test_manifest_verbatim(tmp_path):
    m = tmp_path / "RUN_MANIFEST.json"
    m.write_text(json.dumps({"run_id": "r1", "git_sha": "abc123", "exit_code": 0}))
    res = rm.read_manifest(m)
    assert res["rows"][0]["git_sha"] == "abc123"
    assert set(res["available_keys"]) == {"run_id", "git_sha", "exit_code"}


def test_main_exit_codes(tmp_path, capsys):
    _write_csv(tmp_path / "metrics_r1.csv")
    assert rm.main(["--run", "r1", "--log-dir", str(tmp_path), "--key", "val_r"]) == 0
    capsys.readouterr()
    assert rm.main(["--run", "absent", "--log-dir", str(tmp_path)]) == 2
