#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""
LamQuant — New Seizure-Corpus Ingest (Helsinki Neonatal + AUB Beirut)
====================================================================
Generate per-recording SNN activity-label ``.npz`` files for a freshly
**encoded** seizure corpus, reusing the exact label primitive the rest of
the pipeline already trusts: ``chbmit_seizures_to_labels``.

Both corpora this script handles annotate **whole-brain** seizure
intervals (no spatial localization) — identical in nature to CHB-MIT and
Siena — so the per-file seizure intervals route straight through
``chbmit_seizures_to_labels`` (the ``{0=quiet, 2=seizure}`` mapping). The
8 spatial groups all receive the same recording-level seizure flag.

This module reads the recording **duration** from the already-encoded
``.lma`` archive (canonical, no MNE / no GPU): the native sample rate +
total sample count come from the LML container metadata, so the label
time-base is identical to what the decoder + ``LmaDataset`` will see at
train time.

Output ``.npz`` schema (matches the disk-staged label cache that
``lamquant.snn.lma_dataset`` / ``lamquant_codec.training.lma_dataset``
read — verified against an existing file under
``/mnt/4tb/data/Training/labels/``):

    activity_labels:  uint8 [8, T_latent]   (only key the loader reads)
    source:           str  (original EDF basename, e.g. ``helsinki_eeg7.edf``)
    annotation_file:  str  (path/marker of the annotation used)
    label_counts:     dict {quiet: N, active: N, seizure: N}

Label-filename ↔ recording-stem convention (LOAD-BEARING)
---------------------------------------------------------
The ``LmaDataset`` disk-staged label cache matches a recording stem to its
labels by the filename ``<stem>_labels.npz`` (see
``lamquant.snn.lma_dataset._label_cache_dir`` +
``build_lma_entry_index``'s ``_labels.npz`` regex). The stem is the
basename of the ``.lma`` / ``.lml`` entry minus its suffix.

To keep stems globally unique across corpora (and avoid the all-quiet
silent-fallback bug that a name miss would cause), each recording is
encoded under a corpus-prefixed name — e.g. Helsinki ``eeg7.edf`` is
symlink-farmed to ``helsinki_eeg7.edf`` *before* ``lml encode``. That
yields:

    archive  : Training/lma/helsinki_neonatal/helsinki_eeg7.lma
    entry    : helsinki_eeg7.lml          (stem = ``helsinki_eeg7``)
    labels   : Training/labels/helsinki_eeg7_labels.npz

so ``build_lma_entry_index`` keys the stem ``helsinki_eeg7`` and the
loader's ``<stem>_labels.npz`` lookup resolves to the file written here.

Usage
-----
    python -m lamquant.snn.ingest_new_seizure_corpus \
        --corpus helsinki \
        --lma-dir   /mnt/4tb/data/Training/lma/helsinki_neonatal \
        --ann-dir   /mnt/4tb/data/Downloads/helsinki_neonatal \
        --out       /mnt/4tb/data/Training/labels \
        --prefix    helsinki_

    python -m lamquant.snn.ingest_new_seizure_corpus \
        --corpus aub \
        --lma-dir   /mnt/4tb/data/Training/lma/aub_beirut \
        --ann-dir   /mnt/4tb/data/Downloads/aub_beirut \
        --out       /mnt/4tb/data/Training/labels \
        --prefix    aub_

The ``.lma`` archives must already exist (encode them first with
``lml encode <dir> -o <lma-dir> -q --cross-validate``). This script never
touches the GPU and never modifies an existing corpus or split manifest.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the SNN label primitives — do NOT reimplement chbmit_seizures_to_labels.
# This module lives next to generate_activity_labels.py under lamquant/snn/.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Repo python root, so ``lamquant_core`` / ``lamquant_codec`` import cleanly.
_BLUT_PY = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _BLUT_PY not in sys.path:
    sys.path.insert(0, _BLUT_PY)

from generate_activity_labels import (  # noqa: E402
    NUM_GROUPS,
    TARGET_FS,
    chbmit_seizures_to_labels,
)

Interval = Tuple[float, float]


# =====================================================================
# Duration / stem extraction from the encoded .lma (canonical, no MNE)
# =====================================================================

def read_lma_duration(lma_path: str, lml_entry: str) -> Optional[Tuple[float, float, int]]:
    """Read a recording's native duration from its encoded ``.lma``.

    Returns ``(duration_sec, native_fs, n_channels)`` or ``None`` on a
    read/parse failure. Uses ``lamquant_core.container_metadata`` (a
    header-only parse — no signal decode, no GPU). ``duration_sec`` is
    ``total_samples / native_fs`` so the label time-base matches what the
    decoder produces.
    """
    import lamquant_core as lc  # PyO3 extension (lazy: keeps import cheap)

    try:
        body = lc.lma_read_entry(lma_path, lml_entry)
    except Exception as exc:  # noqa: BLE001 — report and skip, never crash the run
        print(f"  [warn] lma_read_entry failed for {lma_path}::{lml_entry}: {exc}")
        return None
    try:
        meta_json, n_ch, _n_win, total_samples, _window = lc.container_metadata(body)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] container_metadata failed for {lma_path}: {exc}")
        return None
    meta = json.loads(meta_json)
    native_fs = float(meta.get("sample_rate") or 0.0)
    if native_fs <= 0.0:
        print(f"  [warn] {lma_path}: non-positive sample_rate={native_fs}; skipping")
        return None
    duration_sec = float(total_samples) / native_fs
    return duration_sec, native_fs, int(n_ch)


def index_lma_dir(lma_dir: str) -> Dict[str, Tuple[str, str]]:
    """Map each encoded stem -> ``(lma_path, lml_internal_entry)``.

    Per-recording ``lml encode`` produces one ``<stem>.lma`` per input EDF,
    each holding a single ``<stem>.lml`` (verified against the live
    Training/lma corpora). ``build_lma_entry_index`` is the canonical way
    to discover the internal entry name, so we use it here rather than
    assuming ``<stem>.lml`` — this keeps us correct if a future encoder
    layout changes the internal entry path.
    """
    from lamquant_codec.training.lma_dataset import build_lma_entry_index

    lma_paths = sorted(glob.glob(os.path.join(lma_dir, "*.lma")))
    idx = build_lma_entry_index(lma_paths)
    return {stem: (info["lma"], info["lml"]) for stem, info in idx.items()}


# =====================================================================
# Helsinki Neonatal — 3-expert consensus annotation parser
# =====================================================================
#
# The Helsinki Neonatal EEG database (Zenodo 2547147 / 4940267) ships
# three independent expert annotation CSVs:
#     annotations_2017_A_fixed.csv  (expert A, the '_fixed' revision)
#     annotations_2017_B.csv        (expert B)
#     annotations_2017_C.csv        (expert C)
# Layout: header row = EDF file numbers (1..79, the columns); each
# subsequent row = ONE SECOND of the recording; cell == 1 iff that expert
# marked seizure at that second for that file, else 0 (blanks past a
# recording's end are treated as 0).
#
# CONSENSUS RULE: a second is seizure iff >= 2 of the 3 experts marked it.
# The per-second consensus binary vector for file N is then merged into
# contiguous (start_sec, stop_sec) intervals (stop = last_sec + 1), which
# route through chbmit_seizures_to_labels exactly like CHB-MIT/Siena.
# Files with zero consensus seconds (seizure-free neonates) yield an empty
# interval list -> all-quiet labels (state 0) — correct and expected.

HELSINKI_CONSENSUS_MIN = 2  # >= 2 of 3 experts → seizure-second
_HELSINKI_ANN_FILES = (
    "annotations_2017_A_fixed.csv",
    "annotations_2017_B.csv",
    "annotations_2017_C.csv",
)


def _load_helsinki_csv(path: str) -> Tuple[np.ndarray, List[int]]:
    """Load one Helsinki expert CSV -> ``(matrix [T, n_files] uint8, file_numbers)``.

    Blank / non-numeric cells (trailing rows past a short recording's end)
    are read as 0. The header row supplies the integer EDF file numbers
    that index the columns.
    """
    rows: List[List[int]] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        file_numbers = [int(h) for h in header]
        for line in reader:
            rows.append([
                (1 if cell.strip() == "1" else 0)
                for cell in line
            ])
    mat = np.asarray(rows, dtype=np.uint8)
    return mat, file_numbers


def _binary_to_intervals(seconds: np.ndarray) -> List[Interval]:
    """Contiguous 1-runs in a per-second binary vector -> [(start, stop)].

    ``stop`` is exclusive in seconds (last seizure-second + 1), so the
    interval ``[start, stop)`` covers exactly the marked seconds when fed
    to ``chbmit_seizures_to_labels`` (which floors start*fs / end*fs).
    """
    intervals: List[Interval] = []
    in_run = False
    run_start = 0
    n = int(seconds.shape[0])
    for i in range(n):
        if seconds[i] and not in_run:
            in_run = True
            run_start = i
        elif not seconds[i] and in_run:
            in_run = False
            intervals.append((float(run_start), float(i)))
    if in_run:
        intervals.append((float(run_start), float(n)))
    return intervals


def parse_helsinki_consensus(ann_dir: str) -> Dict[int, List[Interval]]:
    """Build ``{file_number: [(start_sec, stop_sec), ...]}`` consensus intervals.

    Reads the three expert CSVs, applies the >= 2-of-3 consensus rule
    per second, then merges contiguous seizure-seconds into intervals.
    File numbers with no seizure second are present in the result with an
    empty list (caller may also infer all-quiet from a missing key).
    """
    mats: List[np.ndarray] = []
    file_numbers: Optional[List[int]] = None
    for name in _HELSINKI_ANN_FILES:
        path = os.path.join(ann_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Helsinki annotation CSV missing: {path}")
        mat, fnums = _load_helsinki_csv(path)
        if file_numbers is None:
            file_numbers = fnums
        elif fnums != file_numbers:
            raise ValueError(
                f"Helsinki CSV header mismatch in {name}: "
                f"{fnums[:5]}... vs {file_numbers[:5]}..."
            )
        mats.append(mat)

    # Align row counts (experts may differ by a trailing row).
    t_min = min(m.shape[0] for m in mats)
    stacked = np.stack([m[:t_min] for m in mats], axis=0)  # [3, T, n_files]
    votes = stacked.sum(axis=0)                            # [T, n_files] in 0..3
    consensus = (votes >= HELSINKI_CONSENSUS_MIN).astype(np.uint8)  # [T, n_files]

    out: Dict[int, List[Interval]] = {}
    assert file_numbers is not None
    for col, fnum in enumerate(file_numbers):
        out[fnum] = _binary_to_intervals(consensus[:, col])
    return out


def helsinki_intervals_for_stem(stem: str, prefix: str,
                                consensus: Dict[int, List[Interval]]) -> List[Interval]:
    """Map an encoded stem (e.g. ``helsinki_eeg7``) to its consensus intervals.

    Strips the corpus ``prefix`` then the ``eeg`` token to recover the
    file number N, and returns ``consensus[N]`` (empty list when the file
    is seizure-free or unknown).
    """
    base = stem[len(prefix):] if prefix and stem.startswith(prefix) else stem
    m = re.match(r"eeg(\d+)$", base)
    if not m:
        return []
    return consensus.get(int(m.group(1)), [])


# =====================================================================
# AUB Beirut — adult focal-epilepsy seizure annotations
# =====================================================================
#
# AUB (Mendeley 5pc2j46cbc): 6 adult focal-epilepsy patients, 21-ch
# 10-20, 500 Hz EDF, with per-seizure onset + duration annotations. The
# exact on-disk annotation format is discovered at ingest time (it may be
# .txt / .csv / a README / embedded EDF+ TALs). Whichever it is, we derive
# per-recording (start_sec, stop_sec) intervals — whole-brain, so they
# route through chbmit_seizures_to_labels exactly like Helsinki.
#
# The parser below handles the common "onset + duration" tabular form
# (CSV or whitespace-delimited): a row carries a recording identifier, a
# seizure onset (seconds into the recording), and a seizure duration (or
# an explicit offset). It tolerantly recognises column names; rows that
# don't parse are skipped with a warning. If the real corpus turns out to
# use a different layout (e.g. EDF+ Annotations), extend this function —
# the rest of the pipeline (stem mapping, label write) is format-agnostic.

# Header tokens (lowercased) we accept for each field.
_AUB_FILE_KEYS = ("file", "filename", "recording", "record", "edf", "session")
_AUB_ONSET_KEYS = ("onset", "start", "start_time", "seizure_start", "onset_sec",
                    "onset(s)", "start(s)")
_AUB_DUR_KEYS = ("duration", "dur", "length", "duration_sec", "duration(s)")
_AUB_OFFSET_KEYS = ("offset", "stop", "end", "end_time", "seizure_end",
                    "offset_sec", "end(s)", "stop(s)")


def _aub_stem_from_value(value: str) -> str:
    """Normalise an annotation recording reference to a bare stem.

    Strips directory parts and a trailing ``.edf`` so the value can be
    matched against an encoded stem's un-prefixed base.
    """
    base = os.path.basename(value.strip())
    if base.lower().endswith(".edf"):
        base = base[: -len(".edf")]
    return base


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def parse_aub_tabular(path: str) -> Dict[str, List[Interval]]:
    """Parse an AUB onset/duration annotation table (CSV or whitespace).

    Returns ``{recording_stem: [(start_sec, stop_sec), ...]}`` keyed by the
    bare recording stem (no ``.edf``, no directory). Recognises a header
    row by matching known field tokens; falls back to positional columns
    (file, onset, duration) when no recognisable header is present.
    """
    out: Dict[str, List[Interval]] = defaultdict(list)
    with open(path, newline="", errors="replace") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delim = "," if sample.count(",") >= sample.count("\t") and "," in sample else None
        reader = csv.reader(fh, delimiter=delim) if delim else (
            [ln.split() for ln in fh if ln.strip()]
        )
        rows = list(reader)

    if not rows:
        return {}

    header = [c.strip().lower() for c in rows[0]]
    has_header = any(
        any(tok in cell for tok in _AUB_ONSET_KEYS + _AUB_FILE_KEYS)
        for cell in header
    )

    def _col(keys) -> Optional[int]:
        for i, cell in enumerate(header):
            if cell in keys:
                return i
        return None

    if has_header:
        ci_file = _col(_AUB_FILE_KEYS)
        ci_onset = _col(_AUB_ONSET_KEYS)
        ci_dur = _col(_AUB_DUR_KEYS)
        ci_off = _col(_AUB_OFFSET_KEYS)
        data_rows = rows[1:]
    else:
        # Positional fallback: file, onset, duration.
        ci_file, ci_onset, ci_dur, ci_off = 0, 1, 2, None
        data_rows = rows

    for row in data_rows:
        if ci_file is None or ci_file >= len(row):
            continue
        stem = _aub_stem_from_value(row[ci_file])
        if not stem:
            continue
        onset = _to_float(row[ci_onset]) if (ci_onset is not None and ci_onset < len(row)) else None
        if onset is None:
            continue
        stop: Optional[float] = None
        if ci_off is not None and ci_off < len(row):
            stop = _to_float(row[ci_off])
        if stop is None and ci_dur is not None and ci_dur < len(row):
            dur = _to_float(row[ci_dur])
            if dur is not None:
                stop = onset + dur
        if stop is None or stop <= onset:
            continue
        out[stem].append((onset, stop))
    return dict(out)


def discover_aub_annotations(ann_dir: str) -> Dict[str, List[Interval]]:
    """Discover + parse AUB seizure annotations under ``ann_dir``.

    Scans for tabular annotation files (``*.csv`` / ``*.txt`` whose name or
    content references seizures/onsets) and merges their intervals keyed
    by bare recording stem. Returns an empty dict when nothing parseable
    is found (caller then labels every recording all-quiet and reports it).
    """
    merged: Dict[str, List[Interval]] = defaultdict(list)
    candidates = []
    for pat in ("*.csv", "*.txt", "*.tsv"):
        candidates.extend(glob.glob(os.path.join(ann_dir, "**", pat), recursive=True))
    for path in sorted(set(candidates)):
        name = os.path.basename(path).lower()
        if name.startswith(("readme", "license", "_filelist")):
            continue
        try:
            parsed = parse_aub_tabular(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] AUB annotation parse failed for {path}: {exc}")
            continue
        if parsed:
            print(f"  [aub] parsed {sum(len(v) for v in parsed.values())} seizures "
                  f"across {len(parsed)} recordings from {os.path.basename(path)}")
            for stem, ivs in parsed.items():
                merged[stem].extend(ivs)
    return dict(merged)


def aub_intervals_for_stem(stem: str, prefix: str,
                           ann: Dict[str, List[Interval]]) -> List[Interval]:
    """Map an encoded AUB stem to its seizure intervals (prefix-tolerant)."""
    base = stem[len(prefix):] if prefix and stem.startswith(prefix) else stem
    if base in ann:
        return ann[base]
    # Tolerant fallback: case-insensitive match on bare stem.
    low = base.lower()
    for k, v in ann.items():
        if k.lower() == low:
            return v
    return []


# =====================================================================
# Conformance check (montage vet) — reuse channel_resolver
# =====================================================================

def check_conformance(lma_path: str, lml_entry: str) -> Tuple[bool, List[str]]:
    """Return ``(conformant, missing)`` for one encoded recording.

    Conformant iff ``select_channels`` returns an empty missing list over
    the recording's channel names (read from LML metadata). Mirrors the
    montage-vet definition used across the corpora.
    """
    import lamquant_core as lc
    from lamquant_codec.channel_resolver import select_channels

    try:
        body = lc.lma_read_entry(lma_path, lml_entry)
        meta_json, _n_ch, _n_win, _total, _window = lc.container_metadata(body)
    except Exception as exc:  # noqa: BLE001
        return False, [f"<metadata read failed: {exc}>"]
    meta = json.loads(meta_json)
    channels = meta.get("channels", [])
    _mapping, missing = select_channels(channels)
    return (missing == []), list(missing)


# =====================================================================
# Main ingest
# =====================================================================

def build_intervals_resolver(corpus: str, ann_dir: str, prefix: str):
    """Return a ``stem -> List[Interval]`` callable for the chosen corpus."""
    if corpus == "helsinki":
        consensus = parse_helsinki_consensus(ann_dir)
        n_sz_files = sum(1 for ivs in consensus.values() if ivs)
        print(f"[*] Helsinki consensus: {n_sz_files}/{len(consensus)} file-numbers "
              f"have >= 1 consensus seizure interval")
        return lambda stem: helsinki_intervals_for_stem(stem, prefix, consensus)
    if corpus == "aub":
        ann = discover_aub_annotations(ann_dir)
        n_sz_files = sum(1 for ivs in ann.values() if ivs)
        print(f"[*] AUB annotations: {n_sz_files} recording(s) with seizures parsed")
        return lambda stem: aub_intervals_for_stem(stem, prefix, ann)
    raise ValueError(f"unknown corpus mode: {corpus!r}")


def ingest(corpus: str, lma_dir: str, ann_dir: str, out_dir: str,
           prefix: str, dry_run: bool = False) -> dict:
    """Generate label ``.npz`` files for every encoded recording in ``lma_dir``.

    Returns a stats dict with per-corpus totals (recordings, seizure-positive
    count, total seizure-seconds, conformance breakdown).
    """
    os.makedirs(out_dir, exist_ok=True)
    stem_to_lma = index_lma_dir(lma_dir)
    if not stem_to_lma:
        raise RuntimeError(f"no encoded .lma stems found under {lma_dir}")
    print(f"[*] {len(stem_to_lma)} encoded recordings under {lma_dir}")

    resolve = build_intervals_resolver(corpus, ann_dir, prefix)

    stats = {
        "corpus": corpus,
        "recordings": 0,
        "skipped": 0,
        "seizure_positive": 0,
        "total_seizure_seconds": 0.0,
        "total_seizure_timesteps": 0,
        "conformant": 0,
        "nonconformant": 0,
        "nonconformant_stems": [],
        "written": [],
    }

    for stem in sorted(stem_to_lma):
        lma_path, lml_entry = stem_to_lma[stem]
        dur = read_lma_duration(lma_path, lml_entry)
        if dur is None:
            stats["skipped"] += 1
            continue
        duration_sec, native_fs, _n_ch = dur

        conformant, missing = check_conformance(lma_path, lml_entry)
        if conformant:
            stats["conformant"] += 1
        else:
            stats["nonconformant"] += 1
            stats["nonconformant_stems"].append((stem, missing))
            print(f"  [nonconf] {stem}: missing {missing}")

        intervals = resolve(stem)
        # Clip intervals to the recording duration (defensive: annotation
        # seconds beyond the EDF tail are dropped by events_to_labels' min(T)
        # anyway, but report seizure-seconds against the clipped span).
        clipped: List[Interval] = []
        for s, e in intervals:
            s_c = max(0.0, float(s))
            e_c = min(float(duration_sec), float(e))
            if e_c > s_c:
                clipped.append((s_c, e_c))

        # Native sample rate so the label time-base matches the decoder's
        # resample to TARGET_FS happens downstream; chbmit_seizures_to_labels
        # builds labels at TARGET_FS using the second-based intervals, which
        # are sample-rate independent (they are in seconds).
        labels = chbmit_seizures_to_labels(clipped, duration_sec, fs=TARGET_FS)

        seizure_seconds = sum(e - s for s, e in clipped)
        n_seizure_ts = int(np.sum(labels == 2))
        if n_seizure_ts > 0:
            stats["seizure_positive"] += 1
        stats["total_seizure_seconds"] += seizure_seconds
        stats["total_seizure_timesteps"] += n_seizure_ts
        stats["recordings"] += 1

        label_counts = {
            "quiet": int(np.sum(labels == 0)),
            "active": int(np.sum(labels == 1)),
            "seizure": n_seizure_ts,
        }
        out_name = f"{stem}_labels.npz"
        out_path = os.path.join(out_dir, out_name)
        if not dry_run:
            np.savez_compressed(
                out_path,
                activity_labels=labels.astype(np.uint8),
                source=f"{stem}.edf",
                annotation_file=f"{corpus}_consensus" if corpus == "helsinki"
                                else f"{corpus}_annotation",
                label_counts=label_counts,
            )
        stats["written"].append(out_name)

    return stats


def _print_stats(stats: dict) -> None:
    print(f"\n{'=' * 64}")
    print(f"Ingest complete — corpus '{stats['corpus']}'")
    print(f"  Recordings labeled:    {stats['recordings']}")
    print(f"  Skipped (bad .lma):    {stats['skipped']}")
    print(f"  Seizure-positive:      {stats['seizure_positive']}")
    print(f"  Total seizure-seconds: {stats['total_seizure_seconds']:.1f} "
          f"({stats['total_seizure_seconds'] / 3600.0:.3f} h)")
    print(f"  Seizure timesteps:     {stats['total_seizure_timesteps']:,} "
          f"(latent @ {TARGET_FS}/8 Hz; {NUM_GROUPS} groups)")
    print(f"  Conformant:            {stats['conformant']}")
    print(f"  Non-conformant:        {stats['nonconformant']}")
    if stats["nonconformant_stems"]:
        for stem, missing in stats["nonconformant_stems"]:
            print(f"    - {stem}: missing {missing}")
    print(f"{'=' * 64}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SNN seizure labels for a newly-encoded corpus "
                    "(Helsinki Neonatal or AUB Beirut).")
    parser.add_argument("--corpus", required=True, choices=("helsinki", "aub"),
                        help="annotation-parsing mode")
    parser.add_argument("--lma-dir", required=True,
                        help="directory of encoded per-recording .lma archives")
    parser.add_argument("--ann-dir", required=True,
                        help="directory holding the corpus annotations "
                             "(Helsinki CSVs / AUB annotation files)")
    parser.add_argument("--out", required=True,
                        help="output dir for <stem>_labels.npz "
                             "(the disk-staged label cache, e.g. "
                             "/mnt/4tb/data/Training/labels)")
    parser.add_argument("--prefix", default="",
                        help="corpus stem prefix used at encode time "
                             "(e.g. 'helsinki_'); stripped to recover the "
                             "annotation key")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute + print stats without writing .npz files")
    args = parser.parse_args()

    stats = ingest(
        corpus=args.corpus,
        lma_dir=args.lma_dir,
        ann_dir=args.ann_dir,
        out_dir=args.out,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )
    _print_stats(stats)


if __name__ == "__main__":
    main()
