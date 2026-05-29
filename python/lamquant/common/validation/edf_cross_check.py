#!/usr/bin/env python3
"""
EDF reader cross-check utility.

Compares LamQuant's in-tree binary EDF reader (`edf_to_events._read_edf_binary`)
against an independent reference implementation (pyedflib / Temple's PYED fork)
on the same file. Used as an L5 cross-implementation sanity test: if the two
readers disagree on channel values, something is wrong with either our reader
or the reference — either way, we want to know before training on that data.

Not used at training time. This is a diagnostic + test utility only.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Make the in-tree dataset_sim importable without requiring package install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET_SIM_DIR = os.path.join(_REPO_ROOT, "ai_models", "dataset_sim")
if _DATASET_SIM_DIR not in sys.path:
    sys.path.insert(0, _DATASET_SIM_DIR)


@dataclass
class ChannelDiff:
    label: str
    n_samples: int
    max_abs_diff: float
    rmse: float
    rel_rmse: float  # RMSE / RMS(reference), or inf if reference is flat


@dataclass
class CrossCheckResult:
    edf_path: str
    sfreq_ours: Optional[float]
    sfreq_pyedflib: Optional[float]
    channels_ours: List[str]
    channels_pyedflib: List[str]
    channels_compared: List[str]
    channels_only_ours: List[str]
    channels_only_pyedflib: List[str]
    per_channel: List[ChannelDiff] = field(default_factory=list)

    @property
    def max_abs_diff(self) -> float:
        return max((c.max_abs_diff for c in self.per_channel), default=0.0)

    @property
    def worst_rmse(self) -> float:
        return max((c.rmse for c in self.per_channel), default=0.0)

    @property
    def sample_rates_agree(self) -> bool:
        return (
            self.sfreq_ours is not None
            and self.sfreq_pyedflib is not None
            and abs(self.sfreq_ours - self.sfreq_pyedflib) < 1e-9
        )

    def is_bit_equivalent(self, tol: float = 1e-9) -> bool:
        """True if every compared channel matches within `tol` in physical units."""
        return (
            self.sample_rates_agree
            and len(self.per_channel) > 0
            and self.max_abs_diff <= tol
        )

    def summary(self) -> str:
        lines = [
            f"EDF: {self.edf_path}",
            f"  sfreq ours      = {self.sfreq_ours}",
            f"  sfreq pyedflib  = {self.sfreq_pyedflib}",
            f"  channels ours     = {len(self.channels_ours)}",
            f"  channels pyedflib = {len(self.channels_pyedflib)}",
            f"  compared          = {len(self.channels_compared)}",
            f"  only ours         = {self.channels_only_ours}",
            f"  only pyedflib     = {self.channels_only_pyedflib}",
            f"  max |diff|        = {self.max_abs_diff:g}",
            f"  worst RMSE        = {self.worst_rmse:g}",
        ]
        return "\n".join(lines)


def _read_with_ours(edf_path: str):
    from edf_to_events import _read_edf_binary  # noqa: E402
    signal_dict, sfreq = _read_edf_binary(edf_path)
    return signal_dict, sfreq


def _read_with_pyedflib(edf_path: str):
    """Return (signal_dict, sfreq_of_mode_channels).

    pyedflib returns physical-unit floats. We key by label exactly as stored
    in the EDF file, same as our reader, so labels can be intersected directly.
    """
    import pyedflib  # local import so the module is optional

    reader = pyedflib.EdfReader(edf_path)
    try:
        n = reader.signals_in_file
        labels = [reader.getLabel(i).strip() for i in range(n)]
        sfreqs = [float(reader.getSampleFrequency(i)) for i in range(n)]
        n_samples = reader.getNSamples()

        signal_dict: Dict[str, np.ndarray] = {}
        for i in range(n):
            if "ANNOTATION" in labels[i].upper():
                continue
            sig = reader.readSignal(i)  # float64, physical units
            signal_dict[labels[i]] = np.asarray(sig, dtype=np.float64)

        # Match our reader's mode-based multi-rate filter: keep only the most
        # common sample rate among non-annotation channels.
        non_ann = [i for i, lbl in enumerate(labels)
                   if "ANNOTATION" not in lbl.upper()]
        if not non_ann:
            return {}, None

        from collections import Counter
        mode_sfreq = Counter(sfreqs[i] for i in non_ann).most_common(1)[0][0]
        signal_dict = {
            labels[i]: signal_dict[labels[i]]
            for i in non_ann
            if sfreqs[i] == mode_sfreq and labels[i] in signal_dict
        }
        return signal_dict, mode_sfreq
    finally:
        reader.close()


def cross_check_edf(edf_path: str) -> CrossCheckResult:
    """Read an EDF with both our reader and pyedflib, return a diff report."""
    ours, sfreq_ours = _read_with_ours(edf_path)
    if ours is None:
        ours = {}

    pyed, sfreq_pyed = _read_with_pyedflib(edf_path)

    ours_labels = list(ours.keys())
    pyed_labels = list(pyed.keys())
    common = [lbl for lbl in ours_labels if lbl in pyed]
    only_ours = [lbl for lbl in ours_labels if lbl not in pyed]
    only_pyed = [lbl for lbl in pyed_labels if lbl not in ours]

    per_channel: List[ChannelDiff] = []
    for lbl in common:
        a = np.asarray(ours[lbl], dtype=np.float64)
        b = np.asarray(pyed[lbl], dtype=np.float64)
        n = min(a.shape[0], b.shape[0])
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        diff = a - b
        max_abs = float(np.max(np.abs(diff))) if n else 0.0
        rmse = float(np.sqrt(np.mean(diff * diff))) if n else 0.0
        rms_ref = float(np.sqrt(np.mean(b * b))) if n else 0.0
        rel = rmse / rms_ref if rms_ref > 0 else float("inf") if rmse > 0 else 0.0
        per_channel.append(ChannelDiff(
            label=lbl,
            n_samples=n,
            max_abs_diff=max_abs,
            rmse=rmse,
            rel_rmse=rel,
        ))

    return CrossCheckResult(
        edf_path=edf_path,
        sfreq_ours=sfreq_ours,
        sfreq_pyedflib=sfreq_pyed,
        channels_ours=ours_labels,
        channels_pyedflib=pyed_labels,
        channels_compared=common,
        channels_only_ours=only_ours,
        channels_only_pyedflib=only_pyed,
        per_channel=per_channel,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Cross-check LamQuant EDF reader against pyedflib.")
    ap.add_argument("edf", help="Path to EDF file")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="Tolerance for bit-equivalence in physical units (default 1e-9)")
    args = ap.parse_args()

    result = cross_check_edf(args.edf)
    print(result.summary())
    if result.is_bit_equivalent(tol=args.tol):
        print("BIT EQUIVALENT within tolerance.")
        sys.exit(0)
    else:
        print("DIFFERENCES FOUND.")
        sys.exit(1)
