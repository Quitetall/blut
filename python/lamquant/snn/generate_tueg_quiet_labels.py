#!/usr/bin/env python3
"""Generate all-quiet activity labels for TUEG so LmaDataset can index it.

TUEG (TUH EEG Corpus, 70,830 recordings / 14,987 subjects) is the largest
encoded corpus on disk but carries NO clinical annotations bundled in its
per-recording ``.lma`` files (one ``<stem>.lml`` entry each, no labels NPZ,
no meta.json). LmaDataset only indexes a stem that has a ``<stem>_labels.npz``
in the disk label cache, so unlabeled TUEG is silently skipped.

This writes a minimal **all-quiet** label NPZ per chosen TUEG stem
(``activity_labels`` = zeros [8, n_windows*312], i.e. class 0 = background
everywhere) so the stem becomes indexable. TUEG is general background EEG;
its true seizure recordings live in TUSZ/TUEP/TUEV (subsets of TUEG that we
ALREADY label), so all-quiet is the correct label, NOT label noise.

Two correctness safeguards:
  1. OVERLAP SKIP. ~10,575 TUEG stems are already labeled (they ARE the
     TUSZ/TUEP/TUEV/TUSL recordings). We never write an all-quiet label for a
     stem that already has a real one — those keep their seizure annotation.
     (build_lma_entry_index also resolves stem collisions first-LMA-wins, so
     ordering labeled roots before TUEG is a second line of defence.)
  2. SUBJECT DIVERSITY. One recording per subject by default (more subjects
     beats more sessions of the same subject for generalization), bounding the
     job to <=14,987 decodes and warming the L3 cache as a side effect.

The exact window count comes from decoding L3 (``_cached_l3_stack`` returns
[n_windows,21,313]); with L3_CACHE_DIR set this ALSO populates the per-stem
``<stem>.npy`` cache that SSL pretrain reuses, so the decode is not wasted.

Usage:
    L3_CACHE_DIR=/mnt/4tb/data/Training/l3_cache \
    PYTHONPATH=/mnt/4tb/LamQuant/blut/python \
    python -u -m lamquant.snn.generate_tueg_quiet_labels \
        --tueg-root /mnt/4tb/data/Training/lma/tueg_v2.0.2 \
        --labels    /mnt/4tb/data/Training/labels \
        --workers 12
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

LABEL_PER_WINDOW = 312          # must match lma_dataset.LABEL_PER_WINDOW
NUM_GROUPS = 8


def _stem_of(fname: str) -> str:
    """`aaaaaaaa_s001_t000.lma` -> `aaaaaaaa_s001_t000`."""
    return fname[:-4] if fname.endswith(".lma") else fname


def _subject_of(stem: str) -> str:
    """`aaaaaaaa_s001_t000` -> `aaaaaaaa` (TUH canonical subject)."""
    i = stem.find("_s")
    return stem[:i] if i > 0 else stem


def _worker(task):
    """Decode one TUEG recording's L3, write an all-quiet label NPZ.

    Returns (stem, n_windows, status). Imports happen inside the worker so
    each multiprocessing process initializes its own lazy state / LRU.
    """
    lma_path, stem, labels_dir = task
    out = Path(labels_dir) / f"{stem}_labels.npz"
    if out.exists():
        return (stem, -1, "exists")
    try:
        from lamquant.snn.lma_dataset import _cached_l3_stack
        l3 = _cached_l3_stack(str(lma_path), stem, lml_internal=f"{stem}.lml")
        if l3 is None:
            return (stem, 0, "decode_none")
        n_win = int(l3.shape[0])
        if n_win < 1:
            return (stem, 0, "empty")
        activity = np.zeros((NUM_GROUPS, n_win * LABEL_PER_WINDOW), dtype=np.uint8)
        # np.savez_compressed appends ".npz" if absent, so the temp name must
        # ALREADY end in ".npz" or the written file != the path we rename.
        tmp = out.with_name(f"{out.stem}.tmp.npz")     # <stem>_labels.tmp.npz
        np.savez_compressed(
            tmp,
            activity_labels=activity,
            source=f"{stem}.edf",
            annotation_file="tueg_all_quiet",
        )
        os.replace(tmp, out)          # atomic publish -> resumable / crash-safe
        return (stem, n_win, "ok")
    except Exception as e:            # noqa: BLE001 — log + continue, never abort the batch
        return (stem, 0, f"err:{type(e).__name__}:{str(e)[:80]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tueg-root", type=Path,
                   default=Path("/mnt/4tb/data/Training/lma/tueg_v2.0.2"))
    p.add_argument("--labels", type=Path,
                   default=Path("/mnt/4tb/data/Training/labels"),
                   help="disk label cache LmaDataset reads (<stem>_labels.npz)")
    p.add_argument("--per-subject", type=int, default=1,
                   help="recordings per subject (1 = max diversity). 0 = all")
    p.add_argument("--max-subjects", type=int, default=0,
                   help="cap number of subjects (0 = no cap)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0,
                   help="hard cap on recordings processed (0 = no cap; smoke)")
    args = p.parse_args()

    args.labels.mkdir(parents=True, exist_ok=True)

    # 1. enumerate TUEG recordings
    lmas = sorted(args.tueg_root.glob("*.lma"))
    if not lmas:
        sys.exit(f"[tueg-labels] no .lma under {args.tueg_root}")
    print(f"[tueg-labels] {len(lmas)} TUEG recordings under {args.tueg_root}",
          flush=True)

    # 2. skip stems that already have a real label (overlap with labeled corpora)
    already = {f.name[:-len("_labels.npz")]
               for f in args.labels.glob("*_labels.npz")}
    print(f"[tueg-labels] {len(already)} stems already labeled (skip overlap)",
          flush=True)

    # 3. group remaining by subject, pick up to --per-subject recordings each
    by_subject: dict[str, list[Path]] = {}
    for lp in lmas:
        stem = _stem_of(lp.name)
        if stem in already:
            continue
        by_subject.setdefault(_subject_of(stem), []).append(lp)

    subjects = sorted(by_subject)
    if args.max_subjects > 0:
        subjects = subjects[:args.max_subjects]

    tasks = []
    for subj in subjects:
        recs = sorted(by_subject[subj])
        recs = recs if args.per_subject <= 0 else recs[:args.per_subject]
        for lp in recs:
            tasks.append((str(lp), _stem_of(lp.name), str(args.labels)))
    if args.limit > 0:
        tasks = tasks[:args.limit]

    print(f"[tueg-labels] {len(subjects)} new-background subjects -> "
          f"{len(tasks)} recordings to label "
          f"(per_subject={args.per_subject})", flush=True)

    # 4. parallel decode + write
    import multiprocessing as mp
    n_ok = n_skip = n_fail = total_win = 0
    errs: dict[str, int] = {}
    shown_err = 0
    # fork (Linux default): child inherits parent state, no module re-import.
    # Safe here because the PARENT never imports torch/cuda before the pool —
    # the heavy decode imports happen inside the worker.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max(1, args.workers)) as pool:
        for i, (stem, n_win, status) in enumerate(
                pool.imap_unordered(_worker, tasks, chunksize=8), 1):
            if status == "ok":
                n_ok += 1
                total_win += n_win
            elif status == "exists":
                n_skip += 1
            else:
                n_fail += 1
                key = status.split(":")[0]
                errs[key] = errs.get(key, 0) + 1
                if shown_err < 5:        # full first failures for diagnosis
                    print(f"  [FAIL] {stem}: {status}", flush=True)
                    shown_err += 1
            if i % 500 == 0:
                print(f"  [{i}/{len(tasks)}] ok={n_ok} skip={n_skip} "
                      f"fail={n_fail} windows={total_win:,}", flush=True)

    print(f"\n[tueg-labels] DONE: ok={n_ok} skip={n_skip} fail={n_fail} "
          f"total_windows={total_win:,}", flush=True)
    if errs:
        print(f"[tueg-labels] failure breakdown: {errs}", flush=True)


if __name__ == "__main__":
    main()
