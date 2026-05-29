"""ai_models/experiment_log.py — append-only experiment log for the
iterate-until-saturated workflow.

Every training run records a single row into `experiment_log.jsonl`.
Each row carries the provenance hashes (manifest_hash + training_config_hash)
that uniquely identify "what data + what recipe", plus the result
metrics (R, PRD, per-band PRD, LQS level), timing, and free-form
notes.

The log is JSONL (one JSON object per line) so it's:
  - Append-only — multiple processes can write concurrently.
  - Streamable — read N rows without parsing the whole file.
  - Greppable — `grep '"asymmetric_weight": 0.2' experiment_log.jsonl`.
  - Pandas-friendly — `pd.read_json(..., lines=True)` if the user wants
    one-liner analysis.

Usage from training scripts:

    from lamquant.common.experiment_log import log_experiment, ExperimentRecord

    log_experiment(ExperimentRecord(
        run_id='joint_fast_t3_1776380533',
        manifest_hash='sha256:...',
        training_config_hash='sha256:...',
        best_val_r=0.7887,
        best_val_prd=60.7,
        per_band_prd={'delta': 51.9, 'theta': 64.7, ...},
        lqs_level='M',
        wall_seconds=1487.2,
        notes='asymmetric envelope w=0.2, baseline comparison',
    ))

Query from CLI / notebook:

    from lamquant.common.experiment_log import list_experiments, best_by

    # All experiments, newest first
    for r in list_experiments(limit=10):
        print(r.run_id, r.best_val_r, r.lqs_level)

    # Best 5 R values across all experiments
    for r in best_by('best_val_r', limit=5):
        print(r.run_id, r.best_val_r, r.training_config_hash)
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional


# Default location: outputs/experiment_log.jsonl. The directory is
# created on first write. Override via `set_log_path()` for tests.
_DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent.parent / 'outputs' / 'experiment_log.jsonl'
)
_log_path: Path = _DEFAULT_LOG_PATH
_lock = threading.Lock()


def set_log_path(path) -> None:
    """Override the log path. Useful for tests + isolated experiments."""
    global _log_path
    _log_path = Path(path)


def get_log_path() -> Path:
    return _log_path


# ============================================================
# ExperimentRecord — schema
# ============================================================

@dataclass
class ExperimentRecord:
    """One row of the experiment log.

    All fields default to sensible "missing" values so partial records
    (e.g., logged at training-start vs at training-end) round-trip
    cleanly. Required-in-practice fields:

      - run_id
      - manifest_hash + training_config_hash (provenance pin)
      - best_val_r + best_val_prd (the headline metrics)
    """
    # Identity
    run_id: str = ''
    timestamp: str = ''               # ISO 8601 UTC, set on log_experiment

    # Provenance (sha256 hashes from DatasetManifest.hash() and
    # TrainingConfig.hash()). The pair uniquely identifies "what data
    # and what recipe."
    manifest_hash: str = ''
    training_config_hash: str = ''
    config_version: str = ''
    manifest_version: str = ''

    # Recipe summary (denormalised for greppability — the hashes are
    # authoritative, but having key knobs inline makes the log readable
    # without a config-database round-trip).
    preset: str = ''                  # 'fast' / 'standard' / ...
    vocos_tier: int = 0
    seed: int = 0
    asymmetric_weight: float = 0.0
    asymmetric_kind: str = ''
    fullband_mode: str = ''           # 'auto' / 'ram' / 'memmap' / 'off'
    amp: bool = True
    compile_decoder: bool = True
    train_noise_bits: int = 0
    epochs_planned: int = 0
    epochs_completed: int = 0

    # Headline metrics
    best_val_r: float = 0.0
    best_val_prd: float = 0.0
    final_val_r: float = 0.0
    final_val_prd: float = 0.0
    best_epoch: int = 0

    # Per-band PRD (at best checkpoint)
    per_band_prd: Dict[str, float] = field(default_factory=dict)
    per_band_r: Dict[str, float] = field(default_factory=dict)

    # LQS gate result
    lqs_level: str = ''               # 'C' / 'M' / 'A' / '' (below floor)
    lqs_violations: List[str] = field(default_factory=list)

    # Performance
    wall_seconds: float = 0.0
    median_step_ms: float = 0.0       # if measured
    peak_vram_gb: float = 0.0

    # Outcomes
    completed: bool = False           # finished cleanly?
    halted: bool = False              # tripped a guard?
    halt_reason: str = ''

    # Free-form
    notes: str = ''
    tags: List[str] = field(default_factory=list)

    # Artifacts
    encoder_ckpt: str = ''
    decoder_ckpt: str = ''
    summary_json: str = ''
    alpha_csv: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentRecord':
        # Tolerant of unknown fields (forward-compat).
        from dataclasses import fields as _fields
        known = {f.name for f in _fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        return cls(**kw)


# ============================================================
# Append + read
# ============================================================

def log_experiment(record: ExperimentRecord, *, log_path=None) -> Path:
    """Append a record to the experiment log. Thread-safe.

    Sets `timestamp` to current UTC if the record didn't supply one.
    Returns the path written to.
    """
    path = Path(log_path) if log_path else _log_path
    if not record.timestamp:
        record.timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')

    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as f:
            f.write(json.dumps(record.to_dict(), separators=(',', ':')) + '\n')
    return path


def iter_records(log_path=None) -> Iterator[ExperimentRecord]:
    """Yield every record in the log, in chronological order."""
    path = Path(log_path) if log_path else _log_path
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield ExperimentRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError) as e:
                # Skip corrupt lines; don't tank the whole reader.
                print(f'[experiment_log] skipping malformed line: {e}')
                continue


def list_experiments(limit: Optional[int] = None, *,
                     newest_first: bool = True,
                     log_path=None) -> List[ExperimentRecord]:
    """All records, optionally limited. Default sort: newest first."""
    records = list(iter_records(log_path=log_path))
    if newest_first:
        records.reverse()
    if limit is not None:
        records = records[:limit]
    return records


def best_by(metric: str, limit: int = 10, *,
             ascending: bool = False,
             filter_fn: Optional[Callable[[ExperimentRecord], bool]] = None,
             log_path=None) -> List[ExperimentRecord]:
    """Top-N by `metric`. Use `ascending=True` for lower-is-better metrics
    (e.g. 'best_val_prd', 'wall_seconds').

    `filter_fn` lets you scope the search:

        # Best R among Tier 7 production runs
        best_by('best_val_r', limit=5,
                filter_fn=lambda r: r.vocos_tier == 7 and r.preset == 'production')
    """
    records = list(iter_records(log_path=log_path))
    if filter_fn is not None:
        records = [r for r in records if filter_fn(r)]
    records.sort(key=lambda r: getattr(r, metric, 0), reverse=not ascending)
    return records[:limit]


def find_by_run_id(run_id: str, log_path=None) -> Optional[ExperimentRecord]:
    """Lookup a single record by run_id. None if not found."""
    for r in iter_records(log_path=log_path):
        if r.run_id == run_id:
            return r
    return None


def compare(run_id_a: str, run_id_b: str, *,
            log_path=None) -> Dict[str, tuple]:
    """Diff two records: {field: (value_a, value_b)} for fields that differ.

    Useful for "why did X beat Y?" — returns the recipe + metric deltas
    in one call.
    """
    a = find_by_run_id(run_id_a, log_path=log_path)
    b = find_by_run_id(run_id_b, log_path=log_path)
    if a is None or b is None:
        missing = [rid for rid, rec in [(run_id_a, a), (run_id_b, b)] if rec is None]
        raise KeyError(f'run_id(s) not found in log: {missing}')
    da, db = a.to_dict(), b.to_dict()
    return {k: (da[k], db[k]) for k in da if da[k] != db[k]}


def summary_table(records: List[ExperimentRecord]) -> str:
    """Render a one-line-per-experiment summary table.

    Columns: run_id (truncated), R, PRD, LQS, tier, asym, wall, notes.
    """
    if not records:
        return '(no experiments)'
    rows = ['run_id                          R       PRD    LQS  tier  asym  wall(min)  notes']
    rows.append('-' * 100)
    for r in records:
        rows.append(
            f'{r.run_id[-30:]:30} '
            f'{r.best_val_r:6.4f}  '
            f'{r.best_val_prd:5.1f}%  '
            f'{r.lqs_level or "--":3}   '
            f'{r.vocos_tier:3d}   '
            f'{r.asymmetric_weight:4.2f}  '
            f'{r.wall_seconds / 60:7.1f}    '
            f'{r.notes[:30]}'
        )
    return '\n'.join(rows)


__all__ = [
    'ExperimentRecord', 'log_experiment', 'iter_records',
    'list_experiments', 'best_by', 'find_by_run_id', 'compare',
    'summary_table', 'set_log_path', 'get_log_path',
]
