#!/usr/bin/env python3
"""On-the-fly activity labels from the clinical annotations INSIDE the LMA.

The corpus LMA already bundles the source annotations (`<stem>.csv_bi` /
`.csv` / `.tse_bi` / `.tse` / `.rec`) byte-exact alongside the LML signal. So
labels need not be precomputed to a separate NPZ + disk cache — derive them
on the fly: extract the annotation entry -> parse -> activity_labels [8, T].

Reuses the canonical TUH parsers in `generate_activity_labels` (fed the
LMA-extracted bytes), so the mapping is identical to the offline pipeline.
Annotation bytes read from the LMA are byte-identical to the on-disk source
(verified in verify_lma_annotations.py via sha256 + pyedflib).
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Optional, Tuple

import numpy as np

# annotation formats, preference order (bi = binary seiz/bckg; full = subtypes)
_ANN_EXT = (".csv_bi", ".tse_bi", ".csv", ".tse", ".rec")
_DUR_RE = re.compile(r"#\s*duration\s*=\s*([0-9.]+)")


def annotation_entry_for(lma_path: str, stem: str) -> Tuple[Optional[str], Optional[str]]:
    """Find the annotation entry for `stem` inside the LMA. Returns (entry, ext)."""
    from lamquant_codec.training.lma_dataset import list_lma_entries
    entries = list_lma_entries(lma_path)
    for ext in _ANN_EXT:
        suffix = f"{stem}{ext}"
        for e in entries:
            if e.endswith(suffix):
                return e, ext
    return None, None


def _read_entry_text(lma_path: str, entry: str) -> str:
    import lamquant_core as lc
    return lc.lma_read_entry(lma_path, entry).decode("utf-8", errors="replace")


def _duration_from_header(text: str) -> Optional[float]:
    m = _DUR_RE.search(text)
    return float(m.group(1)) if m else None


def lma_activity_labels(lma_path: str, stem: str,
                        duration_sec: Optional[float] = None) -> Optional[np.ndarray]:
    """Derive [8, T_latent] uint8 activity labels from the LMA's bundled
    annotation. Returns None if no annotation entry exists (caller treats as
    all-quiet). `duration_sec` overrides the `# duration` header when known
    (e.g. from the decoded signal length)."""
    entry, ext = annotation_entry_for(lma_path, stem)
    if entry is None:
        return None
    text = _read_entry_text(lma_path, entry)
    dur = duration_sec if duration_sec is not None else _duration_from_header(text)
    # reuse the canonical parsers (they take a file path) via a tempfile, so the
    # mapping is bit-identical to the offline generate_activity_labels pipeline.
    from lamquant.snn import generate_activity_labels as G
    fmt = "csv" if ext in (".csv", ".csv_bi") else ("rec" if ext == ".rec" else "tse")
    with tempfile.NamedTemporaryFile("w", suffix=ext, delete=False) as tf:
        tf.write(text)
        tmp = tf.name
    try:
        if fmt == "csv":
            events = G.parse_csv_annotation(tmp)
        elif fmt == "rec":
            events = G.parse_rec(tmp)
        else:
            events = G.parse_tse(tmp)
        if dur is None:
            # fall back to the last annotated stop time
            dur = max((e[1] for e in events), default=0.0) or 1.0
        return G.events_to_labels(events, dur)
    finally:
        os.unlink(tmp)


def lma_window_count(lma_path: str, lml_internal: str) -> Optional[int]:
    """Window count for a recording from the LML header (NO decode).

    `container_metadata` parses the LML container header (~7 ms incl the byte
    read) and exposes `duration_s`; each training window is 10 s @ 250 Hz, so
    n_windows = duration_s // 10. Lets background-only recordings (e.g. TUEG,
    no annotation) be indexed/all-quiet-labelled without a per-stem L3 decode
    (which is 3-40 s each -> days over the 70 K TUEG corpus).
    """
    import json
    import lamquant_core as lc
    try:
        b = lc.lma_read_entry(lma_path, lml_internal)
        m = lc.container_metadata(b)
        j = json.loads(m[0] if isinstance(m, (tuple, list)) else m)
        dur = j.get("duration_s") or j.get("duration_sec") or j.get("duration")
        return max(1, int(dur) // 10) if dur else None
    except Exception:
        return None


def seizure_intervals_ours(lma_path: str, stem: str):
    """Our parser's seizure (start,stop) intervals — for verification."""
    entry, ext = annotation_entry_for(lma_path, stem)
    if entry is None:
        return []
    text = _read_entry_text(lma_path, entry)
    from lamquant.snn import generate_activity_labels as G
    fmt = "csv" if ext in (".csv", ".csv_bi") else ("rec" if ext == ".rec" else "tse")
    with tempfile.NamedTemporaryFile("w", suffix=ext, delete=False) as tf:
        tf.write(text); tmp = tf.name
    try:
        ev = (G.parse_csv_annotation(tmp) if fmt == "csv"
              else G.parse_rec(tmp) if fmt == "rec" else G.parse_tse(tmp))
    finally:
        os.unlink(tmp)
    return sorted((s, e) for (s, e, lbl, _c) in ev if G.map_label(lbl) == 2)
