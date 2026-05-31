#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Montage-conformance vet for LMA archives (header-only, no DSP decode).

For every recording in each archive, read ONLY the LML container metadata
(``container_metadata`` — 32-byte header + UTF-8 JSON, no sample decode, no
rayon DSP storm) to get the channel names, then run the canonical 21-channel
10-20 resolver (``lamquant_codec.channel_resolver.select_channels``). A
recording is CONFORMANT iff no *required* 10-20 channel is missing (optional
ear references are allowed absent). Nonconformant recordings are the ones that
the channel resolver would zero-pad — e.g. the cap_sleep Fpz-Cz sleep montage
that poisoned run-5.

Output: a JSON report (per-corpus + per-recording verdicts) and a printed
summary. READ-ONLY — never mutates an archive.

Usage:
    python -m lamquant.dataset.vet_montage \
        --archive /path/a.lma /path/b.lma \
        --recording-dir /path/tusz_per_recording_dir \
        --out /path/data_vet_report.json

Run single-threaded (RAYON_NUM_THREADS=1) when a training job shares the box.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _sub in ("snn", "dataset", "common"):
    _p = os.path.join(ROOT_DIR, "lamquant", _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _vet_one(lc, cr, lma_path: str, entry: str,
             lenient: bool = False) -> Tuple[bool, List[str], int]:
    """Vet one recording. Returns (conformant, missing_required, n_present).

    Strict (default): conformant iff EVERY required 10-20 channel resolves.
    Lenient (``--lenient``): conformant iff the resolver returns a usable
    mapping at all — i.e. >= ``MIN_REQUIRED_CHANNELS`` (16) channels resolve and
    the LmaDataset loader will zero-fill the few missing ones into its fixed
    21-slot input. This matches what the MODEL actually tolerates; the strict
    criterion needlessly quarantines recordings (e.g. TUSZ missing only Fz+Pz)
    the trainer can use.
    """
    lml_bytes = lc.lma_read_entry(lma_path, entry)
    meta_json, _n_ch, _n_win, _total, _ws = lc.container_metadata(lml_bytes)
    channels = json.loads(meta_json).get("channels", [])
    mapping, missing = cr.select_channels(channels)
    n_present = 0 if mapping is None else len(mapping)
    conformant = (mapping is not None) if lenient else (len(missing) == 0)
    return conformant, list(missing), n_present


def main() -> None:
    ap = argparse.ArgumentParser(description="Montage-conformance vet for LMAs")
    ap.add_argument("--archive", nargs="*", default=[], type=Path,
                    help="per-corpus .lma archives (one corpus each)")
    ap.add_argument("--recording-dir", nargs="*", default=[], type=Path,
                    help="dirs of per-recording .lma (e.g. tusz); each dir is "
                         "treated as one corpus named after the dir")
    ap.add_argument("--out", type=Path, required=True,
                    help="output JSON report path")
    ap.add_argument("--limit", type=int, default=0,
                    help="vet at most N recordings per corpus (0 = all; for a "
                         "quick probe)")
    ap.add_argument("--lenient", action="store_true",
                    help="conformant iff the resolver returns a usable mapping "
                         "(>=MIN_REQUIRED_CHANNELS resolve; loader zero-fills the "
                         "rest) — matches the model's real tolerance, recovers "
                         "recordings missing only a channel or two (e.g. TUSZ "
                         "missing Fz+Pz).")
    args = ap.parse_args()

    import lamquant_core as lc
    from lamquant_codec import channel_resolver as cr
    from lamquant_codec.training.lma_dataset import build_lma_entry_index

    print(f"[*] TARGET montage: {len(cr.TARGET_CHANNELS)} ch "
          f"{cr.TARGET_CHANNELS}")
    print(f"[*] OPTIONAL (ok-absent): {sorted(cr.OPTIONAL_CHANNELS)}")

    # Build the corpus -> {stem: entry_info} work list.
    corpora: Dict[str, Dict[str, dict]] = {}
    for a in args.archive:
        name = a.stem  # chbmit.lma -> chbmit
        try:
            corpora[name] = build_lma_entry_index([str(a)])
        except Exception as e:  # noqa: BLE001
            print(f"[!] index failed for {name}: {str(e)[:160]}; skipping")
    for d in args.recording_dir:
        name = d.name
        # A dir of per-recording .lma: glob the files (two-then-one level,
        # mirroring the trainer's union) and index them under one corpus.
        found = sorted(d.glob("*/*.lma")) or sorted(d.glob("*.lma"))
        if not found:
            print(f"[!] no .lma under {d}; skipping")
            continue
        try:
            corpora[name] = build_lma_entry_index([str(p) for p in found])
        except Exception as e:  # noqa: BLE001
            print(f"[!] index failed for {name}: {str(e)[:160]}; skipping")

    report: Dict[str, dict] = {}
    for corpus, index in corpora.items():
        stems = sorted(index.keys())
        if args.limit:
            stems = stems[: args.limit]
        conf: List[str] = []
        nonconf: List[dict] = []
        errors: List[dict] = []
        miss_hist: Dict[str, int] = {}
        for i, stem in enumerate(stems):
            info = index[stem]
            lma_path = info["lma"]
            entry = info.get("lml") or f"{stem}.lml"
            try:
                ok, missing, n_present = _vet_one(lc, cr, lma_path, entry,
                                                  lenient=args.lenient)
            except Exception as e:  # noqa: BLE001 — hostile archive entry
                errors.append({"stem": stem, "error": str(e)[:200]})
                continue
            if ok:
                conf.append(stem)
            else:
                nonconf.append({"stem": stem, "missing": missing,
                                "n_present": n_present})
                for m in missing:
                    miss_hist[m] = miss_hist.get(m, 0) + 1
            if (i + 1) % 250 == 0:
                print(f"    [{corpus}] {i+1}/{len(stems)} "
                      f"({len(conf)} conf, {len(nonconf)} nonconf)")
        report[corpus] = {
            "n_total": len(stems),
            "n_conformant": len(conf),
            "n_nonconformant": len(nonconf),
            "n_error": len(errors),
            "conformant_stems": conf,
            "nonconformant": nonconf,
            "errors": errors,
            "missing_channel_histogram": dict(
                sorted(miss_hist.items(), key=lambda kv: -kv[1])),
        }
        print(f"[=] {corpus}: {len(conf)}/{len(stems)} conformant, "
              f"{len(nonconf)} nonconformant, {len(errors)} errors")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[*] report -> {args.out}")
    print("\n=== SUMMARY ===")
    print(f"{'corpus':<22}{'total':>8}{'conf':>8}{'nonconf':>9}{'err':>6}")
    for corpus, r in report.items():
        print(f"{corpus:<22}{r['n_total']:>8}{r['n_conformant']:>8}"
              f"{r['n_nonconformant']:>9}{r['n_error']:>6}")


if __name__ == "__main__":
    main()
