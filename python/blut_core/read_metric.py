"""read_metric.py — verbatim metric/log reader (anti-confabulation).

NO LLM IN THE PATH. Returns exact values + timestamps straight from the raw
source so "what did the run actually do" is never a model's paraphrase. This
exists because an LLM sub-agent once fabricated a val-R training trajectory;
the discipline "verify against raw source" is now a tool, not a habit.

Sources (each verbatim; a missing key/file/unit is reported as an explicit
error — a value is NEVER synthesized):

  --run RUN_ID [--log-dir DIR]   MetricLog metrics_<run_id>.csv (or .parquet).
                                 The 47-column per-epoch codec/SNN feed
                                 (val_r, val_loss, train_loss, epoch, ...).
  --csv PATH                     a metrics CSV/parquet by explicit path.
  --job JOB_ID | --status PATH   status.jsonl (StatusUpdate, "kind"-tagged:
                                 step/eval/saved/done/failed) for lamu-train jobs.
  --unit UNIT                    journald for a `blut-*.service` systemd --user
                                 unit (parses JSON StatusUpdate lines; else raw).
  --run RUN_ID --manifest        RUN_MANIFEST.json provenance.
  --wandb ENTITY/PROJECT/RUN     wandb run.history() (key in ~/.netrc).

Selectors:  --key K (repeatable; exact column/field name) · --last N · --kind K

Output: one JSON object on stdout:
  {"source","selector","available_keys","rows":[...],"n","errors":[...]}
Exit code 0 on data, 2 on no-data/not-found, 1 on usage error. The `errors`
list is populated (never an invented row) when a key/file is absent.

stdlib-only except wandb (lazy-imported, only for --wandb) and pyarrow (lazy,
only if a .parquet is encountered). Importing this module is always cheap.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import runctx

# Index columns surfaced alongside any --key selection so a value always has
# its step/epoch/time context. Mirrors metric_log.py's common row fields.
_INDEX_KEYS = ("run_id", "epoch", "global_epoch", "step", "phase", "timestamp")


def _default_log_dir() -> Path:
    # Single source of the BLUT_JOB_DIR-or-training_logs convention (ADR 0044
    # P10): the reader resolves the same anchor the writers use, via runctx, so
    # it finds metrics wherever a BLUT stage put them.
    return runctx.job_dir()


def _result(source: str, selector: str, *, keys: Optional[List[str]] = None,
            rows: Optional[List[Dict[str, Any]]] = None,
            errors: Optional[List[str]] = None) -> Dict[str, Any]:
    rows = rows or []
    return {
        "source": source,
        "selector": selector,
        "available_keys": keys if keys is not None else [],
        "rows": rows,
        "n": len(rows),
        "errors": errors or [],
    }


def _select(rows: List[Dict[str, Any]], keys: List[str],
            available: List[str]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Keep only requested keys (+ index columns). Missing keys -> errors,
    NEVER fabricated. Empty `keys` returns full rows verbatim."""
    errors: List[str] = []
    if not keys:
        return rows, errors
    missing = [k for k in keys if k not in available]
    for k in missing:
        errors.append(
            f"key {k!r} not present; available keys: {', '.join(available)}")
    keep = [k for k in keys if k in available]
    if not keep:
        return [], errors
    idx = [k for k in _INDEX_KEYS if k in available and k not in keep]
    out = [{k: r.get(k) for k in (idx + keep)} for r in rows]
    return out, errors


# --------------------------------------------------------------------------
def read_csv_metrics(path: Path, keys: List[str], last: Optional[int]) -> Dict[str, Any]:
    sel = f"csv:{path}" + (f" keys={keys}" if keys else "")
    if not path.exists():
        # try a .parquet/.csv sibling before giving up
        alt = path.with_suffix(".parquet" if path.suffix == ".csv" else ".csv")
        if alt.exists():
            path = alt
        else:
            return _result("csv", sel, errors=[f"no file: {path} (and no {alt.name} sibling)"])
    rows: List[Dict[str, Any]] = []
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # lazy
            rows = pq.read_table(path).to_pylist()
        except ImportError:
            csv_sib = path.with_suffix(".csv")
            if csv_sib.exists():
                path = csv_sib  # fall through to the CSV branch below
            else:
                return _result("parquet", sel, errors=[
                    f"{path} is parquet but pyarrow is absent and no .csv sibling exists"])
        except Exception as e:  # corrupt/unreadable parquet -> structured error, not a traceback
            return _result("parquet", sel, errors=[
                f"failed to read parquet {path}: {type(e).__name__}: {e}"])
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix != ".parquet":
        return _result("csv", sel, errors=[
            f"unsupported extension {path.suffix!r} for {path}; expected .csv or .parquet"])
    available = list(rows[0].keys()) if rows else []
    if last is not None:
        rows = rows[-last:]
    rows, errors = _select(rows, keys, available)
    return _result("csv", sel, keys=available, rows=rows, errors=errors)


def read_status_jsonl(path: Path, kind: Optional[str], last: Optional[int]) -> Dict[str, Any]:
    sel = f"status.jsonl:{path}" + (f" kind={kind}" if kind else "")
    if not path.exists():
        return _result("status.jsonl", sel, errors=[f"no file: {path}"])
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: malformed JSON ({e}); raw: {line[:120]}")
                continue
            if kind and obj.get("kind") != kind:
                continue
            rows.append(obj)
    kinds = sorted({r.get("kind") for r in rows if isinstance(r, dict)} - {None})
    if last is not None:
        rows = rows[-last:]
    return _result("status.jsonl", sel, keys=kinds, rows=rows, errors=errors)


def read_journald(unit: str, kind: Optional[str], last: Optional[int]) -> Dict[str, Any]:
    sel = f"journald:{unit}"
    if not unit.endswith(".service"):
        unit = unit + ".service"
    # Validate before handing to subprocess (journalctl -u isn't a shell, but
    # keep the arg to a known-safe charset rather than trusting CLI input).
    if not re.fullmatch(r"[A-Za-z0-9_@.\-]+", unit):
        return _result("journald", sel, errors=[f"invalid unit name: {unit!r}"])
    # Cap the fetch: a long-lived unit's journal can be huge. --last trims
    # post-fetch; --lines bounds what journalctl returns (newest N).
    lines = str(last if (last is not None and last > 0) else 10000)
    try:
        out = subprocess.run(
            ["journalctl", "--user", "-u", unit, "-o", "json", "--no-pager", "--lines", lines],
            capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return _result("journald", sel, errors=["journalctl not found on PATH"])
    except subprocess.TimeoutExpired:
        return _result("journald", sel, errors=["journalctl timed out (30s)"])
    if out.returncode != 0:
        return _result("journald", sel, errors=[
            f"journalctl exit {out.returncode}: {out.stderr.strip()[:200]}"])
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = entry.get("MESSAGE", "")
        ts_us = entry.get("__REALTIME_TIMESTAMP")
        ts = (int(ts_us) / 1e6) if ts_us and str(ts_us).isdigit() else None
        # A StatusUpdate line is JSON with a "kind" tag; else keep raw verbatim.
        rec: Dict[str, Any]
        if msg.startswith("{"):
            try:
                rec = json.loads(msg)
            except json.JSONDecodeError:
                rec = {"raw": msg}
        else:
            rec = {"raw": msg}
        if ts is not None:
            rec.setdefault("ts", ts)
        if kind and rec.get("kind") != kind:
            continue
        rows.append(rec)
    if not rows:
        errors.append(f"no journald entries for unit {unit}")
    kinds = sorted({r.get("kind") for r in rows if r.get("kind")} )
    if last is not None:
        rows = rows[-last:]
    return _result("journald", sel, keys=kinds, rows=rows, errors=errors)


def read_manifest(path: Path) -> Dict[str, Any]:
    sel = f"manifest:{path}"
    if not path.exists():
        return _result("manifest", sel, errors=[f"no file: {path}"])
    try:
        obj = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return _result("manifest", sel, errors=[f"malformed JSON: {e}"])
    keys = list(obj.keys()) if isinstance(obj, dict) else []
    return _result("manifest", sel, keys=keys, rows=[obj])


def read_wandb(run_path: str, keys: List[str], last: Optional[int]) -> Dict[str, Any]:
    sel = f"wandb:{run_path}" + (f" keys={keys}" if keys else "")
    try:
        import wandb  # lazy
    except ImportError:
        return _result("wandb", sel, errors=["wandb not installed in this interpreter"])
    try:
        api = wandb.Api()
        run = api.run(run_path)
        hist = run.history(keys=keys or None, pandas=False)
    except Exception as e:  # network/auth/not-found — surface verbatim, don't invent
        return _result("wandb", sel, errors=[f"wandb error: {type(e).__name__}: {e}"])
    rows = list(hist)
    available = sorted({k for r in rows for k in r}) if rows else []
    if last is not None:
        rows = rows[-last:]
    rows, errors = _select(rows, keys, available) if keys else (rows, [])
    return _result("wandb", sel, keys=available, rows=rows, errors=errors)


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="read_metric",
        description="Verbatim metric/log reader (no LLM). Exact values + timestamps.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="MetricLog run_id -> metrics_<run_id>.csv")
    src.add_argument("--csv", help="explicit metrics CSV/parquet path")
    src.add_argument("--job", help="lamu train-job id -> status.jsonl")
    src.add_argument("--status", help="explicit status.jsonl path")
    src.add_argument("--unit", help="journald systemd --user unit name")
    src.add_argument("--wandb", help="wandb run path entity/project/run_id")
    p.add_argument("--log-dir", default=None, help="metrics dir (default blut/python/training_logs)")
    p.add_argument("--manifest", action="store_true", help="with --run: read RUN_MANIFEST.json")
    p.add_argument("--key", action="append", default=[], help="exact column/field (repeatable)")
    p.add_argument("--kind", default=None, help="filter status/journald by StatusUpdate kind")
    p.add_argument("--last", type=int, default=None, help="only the last N rows")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log_dir = Path(args.log_dir) if args.log_dir else _default_log_dir()

    if args.run and args.manifest:
        res = read_manifest(_manifest_path(args.run, log_dir))
    elif args.run:
        res = read_csv_metrics(log_dir / f"metrics_{args.run}.csv", args.key, args.last)
    elif args.csv:
        res = read_csv_metrics(Path(args.csv), args.key, args.last)
    elif args.job:
        sj = Path.home() / ".local/share/lamu/train-jobs" / args.job / "status.jsonl"
        res = read_status_jsonl(sj, args.kind, args.last)
    elif args.status:
        res = read_status_jsonl(Path(args.status), args.kind, args.last)
    elif args.unit:
        res = read_journald(args.unit, args.kind, args.last)
    elif args.wandb:
        res = read_wandb(args.wandb, args.key, args.last)
    else:  # pragma: no cover - argparse guarantees one source
        return 1

    json.dump(res, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    # Exit 2 when no data was found (lets callers branch without parsing).
    return 0 if res["rows"] else 2


def _manifest_path(run_id: str, log_dir: Path) -> Path:
    """RUN_MANIFEST.json lives under training_logs/<run_id>/."""
    return log_dir / run_id / "RUN_MANIFEST.json"


if __name__ == "__main__":
    raise SystemExit(main())
