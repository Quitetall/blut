#!/usr/bin/env python3
"""build_seizure_split_manifest.py — patient-level train/val split for the
canonical LMA SNN training path.

Emits the manifest schema that ``ai_models/snn/lma_dataset.py::load_split_manifest``
reads:

    {
      "subjects":          {subject_id: "train"|"val", ...},
      "stems_by_subject":  {subject_id: [stem, stem, ...], ...},
      "meta": { ... provenance ... }
    }

Design decisions (clinical-grade):
  - SPLIT IS PATIENT-LEVEL. A subject's every recording lands in exactly one
    split — never both. Prevents the canonical seizure-leakage failure mode
    (same patient's morphology in train and val inflates val metrics).
    LmaDataset additionally asserts zero stem overlap and raises if violated.
  - SEIZURE-STRATIFIED. Subjects are bucketed into seizure-bearing vs
    non-seizure (any stem with a SEIZURE=2 label → seizure-bearing), and each
    bucket is split independently at ``--val-fraction`` so val carries a
    proportional share of seizure patients (otherwise a small val set can end
    up with zero seizure subjects — the exact bug that produced val[S=0]).
  - DETERMINISTIC, NO RNG. Assignment is ``sha1(subject_id) % 1000 < frac*1000``.
    Reproducible across machines and runs; no seed to forget; adding new
    subjects never reshuffles existing ones.

Only stems that BOTH (a) have an encoded ``.lma`` under ``--lma-root`` and
(b) have a ``<stem>_labels.npz`` under ``--labels`` are included, so the
manifest can never reference an unreadable recording.

Usage:
    python ai_models/dataset_sim/build_seizure_split_manifest.py \
        --lma-root /mnt/4tb/data/Training/lma \
        --labels   /mnt/4tb/data/Training/labels \
        --out      /mnt/4tb/data/Training/split_manifest.json \
        --val-fraction 0.10
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# TUH stem patterns. Subject grouping must be EXACT so the patient-level
# split never leaks the same patient across train/val:
#   canonical TUH:  <subject>_s<session>_t<token>   (TUSZ/TUSL/TUAR/...)
#   TUEV epileptiform corpus: <subject>_<8-digit session id>
# TUEV uses the modern <subject>_NNNNNNNN naming (no _sNNN_tNNN), so its
# recordings were previously only grouped via the coarse split("_")[0]
# fallback. Matching it explicitly documents the contract and keeps the
# subject id == alpha prefix (which IS the patient id for TUEV).
_STEM_RE = re.compile(r"^(?P<subject>[A-Za-z0-9]+)_s\d+_t\d+$")
_TUEV_STEM_RE = re.compile(r"^(?P<subject>[A-Za-z]+)_\d{8}$")


def subject_of(stem: str) -> str:
    """Extract the patient/subject id from a TUH stem.

    Handles the canonical ``_sNNN_tNNN`` layout and the TUEV
    ``<subject>_<8digit>`` layout explicitly. Falls back to the
    pre-first-underscore token for any other non-canonical stem so the
    builder never crashes on an unexpected name (it just groups coarsely).
    """
    m = _STEM_RE.match(stem)
    if m:
        return m.group("subject")
    m = _TUEV_STEM_RE.match(stem)
    if m:
        return m.group("subject")
    return stem.split("_", 1)[0]


def stem_is_seizure(label_path: str) -> bool:
    """True if the label NPZ carries any SEIZURE (==2) timestep.

    A corrupt/missing-key NPZ counts as non-seizure but is logged — silently
    miscounting a seizure recording as background would skew the stratified
    split (and is exactly the kind of quiet degradation a clinical pipeline
    must not hide).
    """
    try:
        with np.load(label_path, allow_pickle=True) as d:
            a = np.asarray(d["activity_labels"])
        return bool((a == 2).any())
    except Exception as e:
        print(f"[!] could not read seizure flag from {label_path}: {e} "
              f"(counting as non-seizure)")
        return False


def assign_split(subject_id: str, val_fraction: float) -> str:
    """Deterministic patient-level split via stable hash (no RNG, no seed)."""
    h = hashlib.sha1(subject_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 1000
    return "val" if bucket < int(round(val_fraction * 1000)) else "train"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lma-root", type=Path, required=True,
                    help="dir of <corpus>/<stem>.lma archives (globbed */*.lma then *.lma)")
    ap.add_argument("--labels", type=Path, required=True,
                    help="dir of <stem>_labels.npz files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    args = ap.parse_args()

    # 1. Encoded stems (intersection guard — only recordings we can actually read).
    lma_paths = sorted(glob.glob(str(args.lma_root / "*" / "*.lma"))) \
        or sorted(glob.glob(str(args.lma_root / "*.lma")))
    encoded_stems = {Path(p).stem for p in lma_paths}
    print(f"[*] encoded .lma stems: {len(encoded_stems)}")

    # 2. Label stems + seizure flag.
    label_files = {os.path.basename(f)[:-len("_labels.npz")]: f
                   for f in glob.glob(str(args.labels / "*_labels.npz"))}
    print(f"[*] label NPZs: {len(label_files)}")

    usable = sorted(encoded_stems & set(label_files))
    print(f"[*] usable stems (encoded ∩ labeled): {len(usable)}")
    if not usable:
        raise SystemExit("no usable stems — check --lma-root / --labels")

    # 3. Group by subject; classify subject as seizure-bearing if ANY stem is.
    stems_by_subject: dict[str, list[str]] = defaultdict(list)
    subject_has_seizure: dict[str, bool] = defaultdict(bool)
    n_seiz_stems = 0
    for stem in usable:
        subj = subject_of(stem)
        stems_by_subject[subj].append(stem)
        if stem_is_seizure(label_files[stem]):
            subject_has_seizure[subj] = True
            n_seiz_stems += 1
    print(f"[*] subjects: {len(stems_by_subject)} "
          f"({sum(subject_has_seizure.values())} seizure-bearing) | "
          f"seizure stems: {n_seiz_stems}")

    # 4. Stratified, deterministic, patient-level split.
    #    Bucket subjects into seizure-bearing vs non-seizure and split EACH
    #    bucket independently at --val-fraction, so val carries a proportional
    #    share of seizure patients regardless of cohort size. A uniform hash
    #    over all subjects (no bucketing) can — on a small seizure cohort —
    #    land zero seizure subjects in val (P≈0.35 at N=10, frac=0.10): the
    #    val[S=0] failure this builder exists to prevent. With a non-empty
    #    seizure cohort and val_fraction>0, force at least one seizure subject
    #    into each split so the guarantee holds even when the hash is unlucky.
    seiz_subjects = sorted(s for s in stems_by_subject if subject_has_seizure[s])
    nonseiz_subjects = sorted(s for s in stems_by_subject if not subject_has_seizure[s])

    subjects: dict[str, str] = {}
    for bucket in (seiz_subjects, nonseiz_subjects):
        for subj in bucket:
            subjects[subj] = assign_split(subj, args.val_fraction)

    # Guarantee: if there is a seizure cohort and we asked for a val set, make
    # sure neither split is starved of seizure subjects (deterministic repair —
    # move the lexicographically-first seizure subject of the over-full split).
    if seiz_subjects and 0.0 < args.val_fraction < 1.0:
        seiz_in_val = [s for s in seiz_subjects if subjects[s] == "val"]
        seiz_in_train = [s for s in seiz_subjects if subjects[s] == "train"]
        if not seiz_in_val and seiz_in_train:
            subjects[seiz_in_train[0]] = "val"
        elif not seiz_in_train and seiz_in_val:
            subjects[seiz_in_val[0]] = "train"

    counts = {"train": {"subj": 0, "stem": 0, "seiz_subj": 0},
              "val": {"subj": 0, "stem": 0, "seiz_subj": 0}}
    for subj in sorted(stems_by_subject):
        split = subjects[subj]
        counts[split]["subj"] += 1
        counts[split]["stem"] += len(stems_by_subject[subj])
        if subject_has_seizure[subj]:
            counts[split]["seiz_subj"] += 1

    manifest = {
        "subjects": subjects,
        "stems_by_subject": {s: sorted(v) for s, v in stems_by_subject.items()},
        "meta": {
            "builder": "build_seizure_split_manifest.py",
            "val_fraction": args.val_fraction,
            "split_method": "patient-level, seizure-stratified, sha1(subject)%1000",
            "lma_root": str(args.lma_root),
            "labels": str(args.labels),
            "n_subjects": len(subjects),
            "n_stems": len(usable),
            "counts": counts,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))

    print(f"[*] wrote {args.out}")
    for sp in ("train", "val"):
        c = counts[sp]
        print(f"    {sp:5s}: {c['subj']:5d} subjects  {c['stem']:6d} stems  "
              f"{c['seiz_subj']:5d} seizure-subjects")


if __name__ == "__main__":
    main()
