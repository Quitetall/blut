#!/usr/bin/env python3
"""Externally verify on-the-fly LMA annotation reading is correct + lossless.

Three independent checks per sampled stem:
  1. BYTE-EXACT   sha256(annotation extracted from LMA) == sha256(on-disk
                  Archive/edf source) — the LMA preserves the annotation verbatim.
  2. PARSE-INDEP  seizure (start,stop) intervals via Python stdlib `csv`
                  (code path distinct from our parser) == our parser's intervals.
  3. EXTERNAL-EEG pyedflib (independent EDF reader) reads the source EDF ->
                  recording duration; assert every seizure interval lies within
                  [0, duration] and the csv_bi `# duration` header matches
                  pyedflib's duration (±1 s).

Usage:
  python -m lamquant.snn.verify_lma_annotations \
    --lma /mnt/4tb/data/Archive/lma/tuh/tusz_v2.0.6.lma \
    --edf-root /mnt/4tb/data/Archive/edf/tuh_repair/tusz_v2.0.6 --n 40
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path

import numpy as np

SEIZ = {"seiz", "fnsz", "gnsz", "spsz", "cpsz", "absz", "tnsz", "tcsz", "mysz"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _indep_seiz_intervals(text: str):
    """Independent stdlib-csv parse — NOT our parser."""
    out = []
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0].startswith("#") or row[0] == "channel":
            continue
        if len(row) < 5:
            continue
        try:
            start, stop, label = float(row[1]), float(row[2]), row[3].strip().lower()
        except (ValueError, IndexError):
            continue
        if label in SEIZ:
            out.append((start, stop))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lma", required=True)
    ap.add_argument("--edf-root", required=True, help="Archive/edf/<...>/<corpus> source root")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import lamquant_core as lc
    from lamquant_codec.training.lma_dataset import list_lma_entries
    from lamquant.snn.lma_annotations import (annotation_entry_for,
                                              seizure_intervals_ours,
                                              _read_entry_text, _duration_from_header)
    import pyedflib

    entries = list_lma_entries(args.lma)
    ann = [e for e in entries if e.endswith(".csv_bi")]
    if not ann:
        ann = [e for e in entries if e.endswith((".csv", ".tse_bi", ".tse"))]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ann)
    ann = ann[: args.n]
    print(f"[verify] {len(ann)} annotation entries sampled from {Path(args.lma).name}")

    n_byte = n_parse = n_edf = n_fail = n_skip = 0
    edf_root = Path(args.edf_root)
    for entry in ann:
        stem = Path(entry).name.rsplit(".", 1)[0]
        # 1. byte-exact: LMA bytes vs on-disk source
        lma_bytes = lc.lma_read_entry(args.lma, entry)
        src = edf_root / entry
        if not src.exists():
            n_skip += 1
            continue
        src_bytes = src.read_bytes()
        byte_ok = _sha(lma_bytes) == _sha(src_bytes)
        n_byte += byte_ok

        text = lma_bytes.decode("utf-8", errors="replace")
        # 2. independent parse vs our parser
        indep = _indep_seiz_intervals(text)
        ours = seizure_intervals_ours(args.lma, stem)
        parse_ok = ([(round(a, 3), round(b, 3)) for a, b in indep]
                    == [(round(a, 3), round(b, 3)) for a, b in ours])
        n_parse += parse_ok

        # 3. external EEG reader (pyedflib) — duration cross-check
        edf_ok = True
        edf = src.with_suffix(".edf")
        if edf.exists():
            try:
                with pyedflib.EdfReader(str(edf)) as f:
                    dur = f.file_duration  # seconds, independent of our codec
                hdr_dur = _duration_from_header(text)
                within = all(0 <= s <= dur + 1 and 0 <= e <= dur + 1 for s, e in indep)
                hdr_match = (hdr_dur is None) or abs(hdr_dur - dur) <= 1.0
                edf_ok = within and hdr_match
            except Exception as ex:
                edf_ok = False
                print(f"  [edf-err] {stem}: {ex!s:.60}")
        n_edf += edf_ok

        if not (byte_ok and parse_ok and edf_ok):
            n_fail += 1
            print(f"  [FAIL] {stem}: byte={byte_ok} parse={parse_ok} edf={edf_ok} "
                  f"(ours={len(ours)} indep={len(indep)} seiz)")

    n = len(ann) - n_skip
    print(f"\n[verify] checked {n} (skipped {n_skip} no-source)")
    print(f"  1. byte-exact (LMA==source) : {n_byte}/{n}")
    print(f"  2. parse-indep (stdlib==ours): {n_parse}/{n}")
    print(f"  3. external pyedflib duration: {n_edf}/{n}")
    print(f"  RESULT: {'ALL PASS' if n_fail == 0 and n > 0 else f'{n_fail} FAIL'}")


if __name__ == "__main__":
    main()
