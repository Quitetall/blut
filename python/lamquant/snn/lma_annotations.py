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

import functools
import os
import re
import tempfile
from typing import List, Optional, Tuple

import numpy as np

# annotation formats, preference order (bi = binary seiz/bckg; full = subtypes)
_ANN_EXT = (".csv_bi", ".tse_bi", ".csv", ".tse", ".rec")
_DUR_RE = re.compile(r"#\s*duration\s*=\s*([0-9.]+)")

# Sentinel `label_internal`: derive labels on the fly from the LMA's bundled
# annotation instead of reading a precomputed NPZ entry. Shared by the dataset
# index/__getitem__ paths and the typed adapter's seizure-flag precompute so
# the dispatch sites cannot drift on a typo'd literal.
LMA_ANNOTATION_SENTINEL = "__lma_annotation__"


@functools.lru_cache(maxsize=16)
def _entries_cached(lma_path: str) -> Tuple[str, ...]:
    """Memoized entry list for an archive, keyed by path.

    `list_lma_entries` spawns a `lml ls` subprocess that parses the whole
    archive footer (~hundreds of ms on a 70 K-entry TUEG `.lma`). Without this
    cache, indexing the label-free codec manifest would re-list the SAME
    archive once per stem — tens of thousands of subprocess spawns, hours per
    index build. LMAs are read-only during training, so per-path memoization
    is safe.
    """
    from lamquant_codec.training.lma_dataset import list_lma_entries
    return tuple(list_lma_entries(lma_path))


@functools.lru_cache(maxsize=16)
def _annotation_index(lma_path: str) -> dict:
    """``stem -> (entry, ext)`` map for the archive's annotation entries, built
    in ONE O(entries) pass and memoized per path.

    Replaces the per-stem O(entries × exts) scan that ``annotation_entry_for``
    used to run: on a 70 K-entry TUEG ``.lma`` indexed against ~63 K train stems
    (all background-only → every lookup scanned the full list and returned None)
    that was ~13 billion ``endswith`` ops ≈ 12 min of construction. The index
    makes it O(entries + stems).

    Semantics preserved exactly: ``_ANN_EXT`` priority (a stem with both
    ``.csv_bi`` and ``.csv`` keeps ``.csv_bi`` — earliest in ``_ANN_EXT``) and
    the first-in-entry-list tiebreak for a repeated (stem, ext).
    """
    rank = {ext: i for i, ext in enumerate(_ANN_EXT)}
    idx: dict = {}
    for e in _entries_cached(lma_path):
        for ext in _ANN_EXT:
            if e.endswith(ext):
                stem = e[: -len(ext)].rsplit("/", 1)[-1]
                prev = idx.get(stem)
                if prev is None or rank[ext] < rank[prev[1]]:
                    idx[stem] = (e, ext)
                break  # an entry matches at most one ext (no _ANN_EXT is a suffix of another)
    return idx


def annotation_entry_for(lma_path: str, stem: str) -> Tuple[Optional[str], Optional[str]]:
    """Find the annotation entry for `stem` inside the LMA. Returns (entry, ext).

    O(1) via the memoized :func:`_annotation_index` (one O(entries) build per
    archive), down from a per-stem O(entries × exts) scan.
    """
    return _annotation_index(lma_path).get(stem, (None, None))


def _read_entry_text(lma_path: str, entry: str) -> str:
    import lamquant_core as lc
    return lc.lma_read_entry(lma_path, entry).decode("utf-8", errors="replace")


def _duration_from_header(text: str) -> Optional[float]:
    m = _DUR_RE.search(text)
    return float(m.group(1)) if m else None


def _parse_annotation_text(text: str, ext: str) -> List[Tuple]:
    """Parse annotation `text` (extension `ext`) into ``(start, stop, label,
    channel)`` event tuples.

    The canonical `generate_activity_labels` parsers take a file path and
    `open()` it, so write one tempfile here — keeping the mapping bit-identical
    to the offline pipeline while factoring the tempfile + format dispatch out
    of every caller.
    """
    from lamquant.snn import generate_activity_labels as G
    fmt = "csv" if ext in (".csv", ".csv_bi") else ("rec" if ext == ".rec" else "tse")
    with tempfile.NamedTemporaryFile("w", suffix=ext, delete=False) as tf:
        tmp = tf.name           # bind before write so the finally-unlink is safe
        tf.write(text)
    try:
        if fmt == "csv":
            return G.parse_csv_annotation(tmp)
        if fmt == "rec":
            return G.parse_rec(tmp)
        return G.parse_tse(tmp)
    finally:
        os.unlink(tmp)


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
    from lamquant.snn import generate_activity_labels as G
    events = _parse_annotation_text(text, ext)
    if dur is None:
        # fall back to the last annotated stop time
        dur = max((e[1] for e in events), default=0.0) or 1.0
    return G.events_to_labels(events, dur)


# Ranged-header prefix size for the fast window-count path (#229). The LML
# container header carries `duration_s` inside its metadata JSON. 64 KiB
# comfortably covers the observed metadata sizes on real corpora (TUEG max
# ~30 KB incl. the base64'd zstd EDF header + non-EEG channels) while reading
# ~100x less than the full ~6.67 MB entry. Entries whose metadata exceeds this
# (or non-raw tiers) trigger the full-read fallback below — correctness holds
# either way.
_HEADER_PREFIX_BYTES = 65536


def _window_count_from_container(meta_json: str) -> Optional[int]:
    """duration_s (s) -> training window count (10 s windows). None if absent."""
    import json
    j = json.loads(meta_json)
    dur = j.get("duration_s") or j.get("duration_sec") or j.get("duration")
    return max(1, int(dur) // 10) if dur else None


def lma_window_count(lma_path: str, lml_internal: str) -> Optional[int]:
    """Window count for a recording from the LML header (NO decode).

    `container_metadata` parses the LML container header (~7 ms incl the byte
    read) and exposes `duration_s`; each training window is 10 s @ 250 Hz, so
    n_windows = duration_s // 10. Lets background-only recordings (e.g. TUEG,
    no annotation) be indexed/all-quiet-labelled without a per-stem L3 decode
    (which is 3-40 s each -> days over the 70 K TUEG corpus).

    #229: fast path reads only a 64 KiB PREFIX of the entry via the batch
    ranged-header API (`lma_entry_headers`) and parses the container header
    from that. If the prefix is too short (metadata > 64 KiB) the parse raises
    and we FALL BACK to the full `lma_read_entry` read — we must not swallow
    that into None, or big-metadata recordings would be silently dropped.
    """
    import lamquant_core as lc
    # Fast path: 64 KiB prefix, parse-once index amortised by the batch API.
    try:
        hdrs = lc.lma_entry_headers(lma_path, [lml_internal], _HEADER_PREFIX_BYTES)
        hdr = hdrs[0] if hdrs else None
        if hdr is not None:
            m = lc.container_metadata(bytes(hdr))
            meta = m[0] if isinstance(m, (tuple, list)) else m
            return _window_count_from_container(meta)
    except Exception:
        # Prefix too short, or new API unavailable, or transient — fall
        # through to the full read. Do NOT return None here.
        pass
    # Fallback: full entry read (original behaviour).
    try:
        b = lc.lma_read_entry(lma_path, lml_internal)
        m = lc.container_metadata(b)
        meta = m[0] if isinstance(m, (tuple, list)) else m
        return _window_count_from_container(meta)
    except Exception:
        return None


def lma_window_counts(lma_path: str,
                      lml_internals: List[str]) -> List[Optional[int]]:
    """Batched window counts for many recordings in ONE LMA archive (#229).

    Parses the archive footer manifest exactly ONCE (via `lma_entry_headers`),
    then reads only a 64 KiB prefix per entry instead of the full ~6.67 MB.
    On the 70 K-entry TUEG LMA this turns ~700 GB of reads + per-call manifest
    re-parsing (~2.6 h) into one parse + N small prefix reads.

    Returns a list aligned 1:1 with `lml_internals`. Any entry whose prefix is
    too short (metadata > 64 KiB), is missing, or is a non-raw tier falls back
    to a per-entry full read so the count is never silently wrong.
    """
    import lamquant_core as lc
    n = len(lml_internals)
    out: List[Optional[int]] = [None] * n
    try:
        hdrs = lc.lma_entry_headers(lma_path, list(lml_internals),
                                    _HEADER_PREFIX_BYTES)
    except Exception:
        # API unavailable / archive open failure -> per-entry fallback.
        return [lma_window_count(lma_path, n_) for n_ in lml_internals]
    for i, hdr in enumerate(hdrs):
        if hdr is not None:
            try:
                m = lc.container_metadata(bytes(hdr))
                meta = m[0] if isinstance(m, (tuple, list)) else m
                out[i] = _window_count_from_container(meta)
                continue
            except Exception:
                pass  # prefix too short -> per-entry full-read fallback below
        # Missing entry, non-raw tier, or truncated prefix: full read.
        out[i] = lma_window_count(lma_path, lml_internals[i])
    return out


def seizure_intervals_ours(lma_path: str, stem: str) -> List[Tuple[float, float]]:
    """Our parser's seizure (start,stop) intervals — for verification."""
    entry, ext = annotation_entry_for(lma_path, stem)
    if entry is None:
        return []
    text = _read_entry_text(lma_path, entry)
    from lamquant.snn import generate_activity_labels as G
    events = _parse_annotation_text(text, ext)
    return sorted((s, e) for (s, e, lbl, _c) in events if G.map_label(lbl) == 2)
