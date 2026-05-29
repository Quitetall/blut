#!/usr/bin/env python3
"""
Wrapper for Temple's NEDC ERDR (ResNet Real-Time Decoder) reference tool.

ERDR is an out-of-tree reference implementation — a pretrained ResNet-18
real-time seizure decoder distributed as a shell-script + Python package under
`Reference Software/nedc_eeg_resnet_decode_realtime/`. We treat it as a black
box: point it at an EDF file, let it write a `.csv_bi` into its own test
output directory, then parse that file into predictions we can score with
`SeizureMetrics`.

This module does NOT in-process import ERDR or torch. It runs ERDR as a
subprocess using ERDR's own driver script, which lets ERDR live in its own
Python environment. A `check_installation()` dry-run verifies the tool is
present and runnable without requiring torch in our venv.

Usage:
    runner = NedcErdrRunner()
    status = runner.check_installation()
    if status.ok:
        result = runner.decode('path/to/file.edf')     # runs ERDR
        events = result.events                          # list of dicts
        # Convert to per-second labels for scoring:
        y_pred, duration = runner.events_to_labels(events, epoch=1.0)
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Default install path (override via NEDC_NFC env var or ctor arg)
_REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_ERDR_ROOT = _REPO_ROOT / "Reference Software" / "nedc_eeg_resnet_decode_realtime" / "v1.0.0"


@dataclass
class InstallationStatus:
    root: Path
    ok: bool
    missing: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "OK" if self.ok else "INCOMPLETE"
        lines = [f"ERDR installation @ {self.root}: {status}"]
        if self.missing:
            lines.append("  missing:")
            lines += [f"    - {m}" for m in self.missing]
        if self.notes:
            lines.append("  notes:")
            lines += [f"    - {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class DecodeResult:
    edf_path: str
    csvbi_path: Optional[str]
    returncode: int
    stdout: str
    stderr: str
    events: List[Dict]


class NedcErdrRunner:
    """Subprocess wrapper for Temple's NEDC ERDR real-time decoder."""

    # Required files inside an ERDR v1.0.0 install
    REQUIRED_FILES = [
        "bin/nedc_eeg_resnet_realtime_decode",
        "bin/nedc_eeg_resnet_realtime_decode_env",
        "lib/params_v01.txt",
        "lib/nedc_decoder.py",
        "lib/nedc_postprocess.py",
        "src/python/util/nedc_driver/nedc_driver.py",
        "models/model.pckl",
    ]

    def __init__(self, erdr_root: Optional[os.PathLike] = None,
                 python_executable: Optional[str] = None):
        env_root = os.environ.get("NEDC_NFC")
        if erdr_root is not None:
            self.root = Path(erdr_root).resolve()
        elif env_root:
            self.root = Path(env_root).resolve()
        else:
            self.root = _DEFAULT_ERDR_ROOT.resolve()
        self.python_executable = python_executable or sys.executable

    # ------------------------------------------------------------------
    # Installation check
    # ------------------------------------------------------------------
    def check_installation(self) -> InstallationStatus:
        """Verify the ERDR install tree is intact. Does not invoke torch."""
        missing = []
        notes = []

        if not self.root.is_dir():
            return InstallationStatus(
                root=self.root,
                ok=False,
                missing=[str(self.root)],
                notes=["ERDR root directory does not exist"],
            )

        for rel in self.REQUIRED_FILES:
            if not (self.root / rel).exists():
                missing.append(rel)

        # Sanity-check the model pickle is not empty
        model_path = self.root / "models" / "model.pckl"
        if model_path.exists() and model_path.stat().st_size == 0:
            missing.append("models/model.pckl (present but empty)")

        # Heads-up about torch: ERDR loads torch when actually decoding.
        # We don't require it installed to pass check_installation().
        try:
            import importlib.util  # noqa: F401
            spec = importlib.util.find_spec("torch")
            if spec is None:
                notes.append(
                    "torch not importable in current venv; "
                    "decode() will fail unless ERDR's own env is on PYTHONPATH"
                )
        except Exception:  # torch check unavailable — note but continue
            pass

        return InstallationStatus(
            root=self.root,
            ok=(len(missing) == 0),
            missing=missing,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Run decode
    # ------------------------------------------------------------------
    def _build_driver_command(self, edf_path: str, output_dir: str,
                              basename: str) -> List[str]:
        """Construct the nedc_driver.py command ERDR's shell script would run."""
        driver = self.root / "src" / "python" / "util" / "nedc_driver" / "nedc_driver.py"
        params = self.root / "lib" / "params_v01.txt"
        return [
            self.python_executable,
            str(driver),
            "-p", str(params),
            "-o", output_dir,
            "-r", "None",
            "-b", basename,
            edf_path,
        ]

    def decode(self, edf_path: str,
               output_dir: Optional[str] = None,
               basename: str = "lamquant_erdr",
               timeout: Optional[float] = None) -> DecodeResult:
        """
        Run ERDR on a single EDF file and parse the resulting csv_bi.

        This bypasses the shell wrapper and calls nedc_driver.py directly so
        we can control the working directory and environment explicitly.
        Requires torch to be importable in `self.python_executable`.

        Args:
            edf_path: Absolute or relative path to an EDF file (50 Hz expected).
            output_dir: Where to write the csv_bi. Defaults to ERDR's test/output.
            basename: Output file basename (no extension).
            timeout: Subprocess timeout in seconds.

        Returns:
            DecodeResult with parsed events on success.
        """
        edf_abs = str(Path(edf_path).resolve())
        if not os.path.exists(edf_abs):
            raise FileNotFoundError(edf_abs)

        out_dir = Path(output_dir) if output_dir else (self.root / "test" / "output")
        out_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["NEDC_NFC"] = str(self.root)
        lib_path = str(self.root / "lib")
        src_python = str(self.root / "src" / "python")
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in [lib_path, src_python, env.get("PYTHONPATH", "")] if p
        )

        cmd = self._build_driver_command(edf_abs, str(out_dir), basename)

        proc = subprocess.run(
            cmd,
            cwd=str(self.root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # ERDR writes <basename>.csv_bi in out_dir
        csvbi_path = out_dir / f"{basename}.csv_bi"
        events: List[Dict] = []
        if csvbi_path.exists():
            events = parse_csvbi(str(csvbi_path))

        return DecodeResult(
            edf_path=edf_abs,
            csvbi_path=str(csvbi_path) if csvbi_path.exists() else None,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            events=events,
        )

    # ------------------------------------------------------------------
    # Event ↔ per-sample label conversion
    # ------------------------------------------------------------------
    @staticmethod
    def events_to_labels(events: List[Dict],
                         duration_s: Optional[float] = None,
                         epoch: float = 1.0,
                         seizure_label: str = "seiz") -> Tuple[np.ndarray, float]:
        """
        Convert parsed csv_bi events into a per-epoch binary label vector.

        Args:
            events: Output of `parse_csvbi` or `NedcEventFormatter.read_events_csv`.
            duration_s: Total duration in seconds. If None, inferred as the
                max `stop_time` across events.
            epoch: Bin width in seconds.
            seizure_label: Label string marking positive class.

        Returns:
            (y, duration) where y has shape [ceil(duration/epoch)], dtype int8,
            1 for seizure bins, 0 otherwise.
        """
        if duration_s is None:
            duration_s = max((float(e["stop_time"]) for e in events), default=0.0)
        n_bins = int(np.ceil(duration_s / epoch)) if duration_s > 0 else 0
        y = np.zeros(n_bins, dtype=np.int8)
        for e in events:
            if e.get("label", "").strip().lower() != seizure_label:
                continue
            start = max(0, int(np.floor(float(e["start_time"]) / epoch)))
            stop = min(n_bins, int(np.ceil(float(e["stop_time"]) / epoch)))
            if stop > start:
                y[start:stop] = 1
        return y, duration_s


def parse_csvbi(path: str) -> List[Dict]:
    """
    Parse a NEDC csv_bi annotation file.

    Returns a list of dicts: {channel, start_time, stop_time, label, confidence}.
    Skips '# ...' comment lines and the column header row.
    """
    events: List[Dict] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("channel,"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                events.append({
                    "channel": parts[0],
                    "start_time": float(parts[1]),
                    "stop_time": float(parts[2]),
                    "label": parts[3],
                    "confidence": float(parts[4]),
                })
            except ValueError:
                continue
    return events


if __name__ == "__main__":
    runner = NedcErdrRunner()
    status = runner.check_installation()
    print(status.summary())
    sys.exit(0 if status.ok else 1)
