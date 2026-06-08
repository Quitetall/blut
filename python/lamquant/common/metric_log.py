"""metric_log.py — reviewer-readable, crash-safe, live per-epoch metric stream.

A standalone metric feed for training runs, distinct from the TrainingLogger's
``epochs_<run_id>.csv``: it writes ``metrics_<run_id>.parquet`` (via pyarrow) or
falls back to ``metrics_<run_id>.csv`` when pyarrow is absent. The whole file is
rewritten atomically (``os.replace``) on every ``append`` so a reviewer can read
a COMPLETE, valid file at any instant mid-run — and a crash never leaves a
truncated/corrupt artifact (unlike a streaming ParquetWriter, whose footer is
only written at close()).

Design constraints:
- stdlib-only at import time (no torch / pyarrow / wandb at module top); pyarrow
  is lazy-imported in ``_detect`` so importing this module is always cheap+safe.
- ``append`` NEVER raises — a logging failure must not crash a training run.
- values are coerced to JSON/columnar-safe scalars so one bad row can't poison
  every subsequent flush.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

__all__ = ["MetricLog"]


def _safe_scalar(v: Any) -> Any:
    """Coerce a value to something both Parquet and CSV accept cleanly.

    Keep int/float/str/bool/None as-is; stringify everything else (sets, objects,
    nested containers) so a single odd value cannot make a whole flush raise."""
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


class MetricLog:
    """Append-only per-epoch metric stream, atomically rewritten each append."""

    def __init__(self, run_id: str, log_dir, basename: str = "metrics") -> None:
        if not run_id or not isinstance(run_id, str):
            raise ValueError(f"run_id must be a non-empty str, got {run_id!r}")
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._rows: List[Dict[str, Any]] = []
        self._backend = self._detect()  # 'parquet' if pyarrow importable else 'csv'
        ext = "parquet" if self._backend == "parquet" else "csv"
        self.path = self.log_dir / f"{basename}_{run_id}.{ext}"

    @staticmethod
    def _detect() -> str:
        """Lazy capability probe — never import pyarrow at module top."""
        try:
            import pyarrow  # noqa: F401
            return "parquet"
        except Exception:
            return "csv"

    def append(self, row: Dict[str, Any]) -> None:
        """Add one metric row and rewrite the whole file atomically.

        Wrapped end-to-end: logging must never crash a training run."""
        try:
            self._rows.append({str(k): _safe_scalar(v) for k, v in dict(row).items()})
            self._flush()
        except Exception as e:  # pragma: no cover - defensive
            print(f"[!] metric_log append failed (non-fatal): {e}")

    def close(self) -> None:
        """Final flush. Idempotent; never raises."""
        try:
            self._flush()
        except Exception as e:  # pragma: no cover - defensive
            print(f"[!] metric_log close failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    def _flush(self) -> None:
        """Whole-file atomic rewrite: write a UNIQUE tmp in the same dir (via
        mkstemp → O_EXCL, no predictable-path symlink hijack), then os.replace,
        so any concurrent reader always sees a complete valid file.

        os.replace is atomic on POSIX LOCAL filesystems; on NFS rename(2) is
        only atomic per-client — acceptable for this live-read use case, but
        note it if logs land on cluster NFS."""
        if not self._rows:
            return
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.log_dir), prefix=self.path.stem + ".",
            suffix=self.path.suffix + ".tmp")
        tmp = Path(tmp_name)
        try:
            if self._backend == "parquet":
                os.close(fd)  # pyarrow writes by path (overwrites the empty tmp)
                import pyarrow as pa
                import pyarrow.parquet as pq
                pq.write_table(pa.Table.from_pylist(self._rows), tmp)
            else:
                # fieldnames = union of all keys, stable first-seen order.
                fields: List[str] = []
                seen = set()
                for r in self._rows:
                    for k in r:
                        if k not in seen:
                            seen.add(k)
                            fields.append(k)
                with os.fdopen(fd, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    w.writeheader()
                    for r in self._rows:
                        w.writerow(r)
            os.replace(tmp, self.path)
        except Exception:
            try:
                tmp.unlink()
            except Exception:
                pass
            raise
