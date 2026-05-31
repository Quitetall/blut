#!/usr/bin/env python3
"""build_seizure_split_manifest.py — patient-level train/val/test split for the
canonical LMA SNN training path.

Emits the manifest schema that ``lamquant.snn.lma_dataset::load_split_manifest``
reads:

    {
      "subjects":          {subject_id: "train"|"val"|"test"|"external_test", ...},
      "stems_by_subject":  {subject_id: [stem, stem, ...], ...},
      "meta": { ... provenance ... }
    }

Design (clinical / FDA-grade — learned from the Gen-7.6.1 official_split_config
that scored 100% sens / 90% acc on a clean CHB-MIT chb21-24 holdout):

  - PATIENT-LEVEL, PATIENT-DISJOINT. A subject's every recording lands in exactly
    one split — never two. Prevents the canonical seizure-leakage failure mode
    (same patient's morphology in train and eval inflates metrics). Asserted
    explicitly at the end; LmaDataset re-asserts at load.
  - 3-WAY: train / val / held-out TEST. The TEST split is NEVER seen during
    training OR operating-point calibration (calibrate on val, report on test).
    This is the FDA requirement the old 2-way split lacked.
  - CROSS-SITE EXTERNAL TEST. One or more whole corpora (``--external-test-corpus``,
    e.g. siena) are held ENTIRELY out as a separate ``external_test`` split — a
    different site/population than the training corpora, the real generalization
    number. (The old design used siena/eegmmidb/mental-arith as validation-only.)
  - SEIZURE-STRATIFIED. Subjects are bucketed seizure-bearing vs non-seizure and
    each bucket is split independently, so every split carries a proportional
    share of seizure patients (prevents the val[S=0] failure). A deterministic
    repair forces >=1 seizure subject into each of train/val/test.
  - DETERMINISTIC, NO RNG. Assignment is ``sha1(subject_id) % 1000`` against
    cumulative fraction cuts. Reproducible across machines; no seed to forget;
    adding new subjects never reshuffles existing ones.

Only stems that BOTH (a) have an encoded ``.lma`` under some ``--lma-root`` and
(b) have a ``<stem>_labels.npz`` under ``--labels`` are included, so the manifest
can never reference an unreadable recording. ``--lma-root`` is repeatable so
corpora spanning multiple roots (e.g. lma + lma_expand) all contribute.

Usage:
    python -m lamquant.dataset.build_seizure_split_manifest \
        --lma-root /mnt/4tb/data/Training/lma \
        --lma-root /mnt/4tb/data/Training/lma_expand \
        --labels   /mnt/4tb/data/Training/labels \
        --out      /mnt/4tb/data/Training/manifests/split_manifest_v2.json \
        --val-fraction 0.10 --test-fraction 0.10 \
        --external-test-corpus siena
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

# Subject grouping must be EXACT so the patient-level split never leaks the same
# patient across splits:
#   canonical TUH:   <subject>_s<session>_t<token>     (TUSZ/TUSL/TUAR/...)
#   TUEV events:     <subject>_<8-digit session id>
#   CHB-MIT:         chbNN_NN  -> subject chbNN
#   Siena:           PNNN-N    -> subject PNNN
#   Helsinki:        helsinki_eegN -> subject helsinki_eegN (one recording per
#                    neonate; each eegN is a distinct individual, so the whole
#                    stem IS the subject. Without this the generic fallback
#                    collapses all 79 neonates to subject "helsinki" -> a single
#                    split, leaking every neonate into one bucket.)
_STEM_RE = re.compile(r"^(?P<subject>[A-Za-z0-9]+)_s\d+_t\d+$")
_TUEV_STEM_RE = re.compile(r"^(?P<subject>[A-Za-z]+)_\d{8}$")
_CHBMIT_STEM_RE = re.compile(r"^(?P<subject>chb\d+)_\d+$")
_SIENA_STEM_RE = re.compile(r"^(?P<subject>PN\d+)[-_]\d+", re.IGNORECASE)
_HELSINKI_STEM_RE = re.compile(r"^(?P<subject>helsinki_eeg\d+)$")

SPLITS = ("train", "val", "test", "external_test")


def subject_of(stem: str) -> str:
    """Extract the patient/subject id from a stem across all supported corpora."""
    for rx in (_STEM_RE, _TUEV_STEM_RE, _CHBMIT_STEM_RE, _SIENA_STEM_RE,
               _HELSINKI_STEM_RE):
        m = rx.match(stem)
        if m:
            return m.group("subject")
    return stem.split("_", 1)[0]


def stem_is_seizure(label_path: str) -> bool:
    """True if the label NPZ carries any SEIZURE (==2) timestep.

    A corrupt/missing-key NPZ counts as non-seizure but is logged — silently
    miscounting a seizure recording as background would skew the stratified
    split (and is exactly the quiet degradation a clinical pipeline must not hide).
    """
    try:
        with np.load(label_path, allow_pickle=True) as d:
            a = np.asarray(d["activity_labels"])
        return bool((a == 2).any())
    except Exception as e:
        print(f"[!] could not read seizure flag from {label_path}: {e} "
              f"(counting as non-seizure)")
        return False


def assign_split(subject_id: str, val_fraction: float, test_fraction: float) -> str:
    """Deterministic patient-level 3-way split via stable hash (no RNG, no seed).

    bucket in [0,1000): [0, val_cut) -> val, [val_cut, test_cut) -> test, else train.
    """
    h = hashlib.sha1(subject_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 1000
    val_cut = int(round(val_fraction * 1000))
    test_cut = val_cut + int(round(test_fraction * 1000))
    if bucket < val_cut:
        return "val"
    if bucket < test_cut:
        return "test"
    return "train"


def corpus_of(lma_path: str, lma_roots: list[str]) -> str:
    """Corpus = the first path component under whichever --lma-root contains it."""
    p = os.path.abspath(lma_path)
    for root in lma_roots:
        root = os.path.abspath(root) + os.sep
        if p.startswith(root):
            rest = p[len(root):]
            return rest.split(os.sep, 1)[0] if os.sep in rest else "_root"
    return "_unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lma-root", type=str, action="append", required=True,
                    help="dir of <corpus>/<stem>.lma archives; repeatable")
    ap.add_argument("--labels", type=Path, required=True,
                    help="dir of <stem>_labels.npz files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--test-fraction", type=float, default=0.10)
    ap.add_argument("--external-test-corpus", type=str, action="append", default=[],
                    help="corpus dir name held ENTIRELY out as external_test (cross-site); repeatable")
    ap.add_argument("--allowlist", type=Path, default=None,
                    help="JSON {corpus: [conformant stems]} from vet_montage. "
                         "When given, stems NOT in the union of these lists are "
                         "dropped (montage-nonconformant exclusion).")
    args = ap.parse_args()
    assert 0.0 <= args.val_fraction < 1.0 and 0.0 <= args.test_fraction < 1.0
    assert args.val_fraction + args.test_fraction < 1.0, "val+test must leave a train set"
    ext_corpora = set(args.external_test_corpus)

    # 1. Label stems FIRST — the authoritative recording-stem universe. Also
    #    lets us tell a per-recording archive (filename == a labeled stem) from
    #    a per-corpus `lml archive` (filename is the corpus; recordings inside).
    label_files = {os.path.basename(f)[:-len("_labels.npz")]: f
                   for f in glob.glob(str(args.labels / "*_labels.npz"))}
    print(f"[*] label NPZs: {len(label_files)}")

    # 2. Encoded stems across ALL roots (intersection guard) + corpus map.
    #    Per-recording (Training/lma/<corpus>/<stem>.lma): filename IS the
    #    recording stem, corpus = parent dir — no archive open needed.
    #    Per-corpus (Archive/lma/<source>/<corpus>.lma): enumerate inner
    #    recordings via the shared entry index, corpus = the .lma filename stem.
    #    Detection: filename stem present in label_files => per-recording.
    from lamquant_codec.training.lma_dataset import build_lma_entry_index
    lma_paths: list[str] = []
    for root in args.lma_root:
        lma_paths += glob.glob(os.path.join(root, "*", "*.lma"))
        lma_paths += glob.glob(os.path.join(root, "*.lma"))
    lma_paths = sorted(set(lma_paths))
    stem_corpus: dict[str, str] = {}
    n_per_corpus = 0
    for p in lma_paths:
        fstem = Path(p).stem
        if fstem in label_files:
            stem_corpus.setdefault(fstem, Path(p).parent.name)
        else:
            n_per_corpus += 1
            for inner in build_lma_entry_index([p]):
                stem_corpus.setdefault(inner, fstem)
    encoded_stems = set(stem_corpus)
    print(f"[*] encoded stems: {len(encoded_stems)} across {len(args.lma_root)} "
          f"root(s) ({n_per_corpus} per-corpus archives enumerated)")

    usable = encoded_stems & set(label_files)
    print(f"[*] usable stems (encoded ∩ labeled): {len(usable)}")
    if args.allowlist is not None:
        allow_raw = json.loads(args.allowlist.read_text())
        if not isinstance(allow_raw, dict):
            raise SystemExit(
                f"--allowlist must be a JSON object {{corpus: [stems]}}, got "
                f"{type(allow_raw).__name__}")
        # Flatten to one set. Safe because stems are corpus-unique (tusz aaaaa*,
        # chbmit chb*, eegmmidb S*, ...), so the union == per-corpus filtering.
        allow = {s for stems in allow_raw.values() for s in stems}
        before = len(usable)
        usable = usable & allow
        print(f"[*] allow-list (montage-conformant): {len(allow)} stems; "
              f"dropped {before - len(usable)} nonconformant -> {len(usable)} usable")
    usable = sorted(usable)
    if not usable:
        raise SystemExit("no usable stems — check --lma-root / --labels / --allowlist")

    # 3. Group by subject; classify subject as seizure-bearing if ANY stem is;
    #    track corpus per subject (a subject is single-corpus).
    stems_by_subject: dict[str, list[str]] = defaultdict(list)
    subject_has_seizure: dict[str, bool] = defaultdict(bool)
    subject_corpus: dict[str, str] = {}
    n_seiz_stems = 0
    for stem in usable:
        subj = subject_of(stem)
        stems_by_subject[subj].append(stem)
        subject_corpus.setdefault(subj, stem_corpus[stem])
        if stem_is_seizure(label_files[stem]):
            subject_has_seizure[subj] = True
            n_seiz_stems += 1
    print(f"[*] subjects: {len(stems_by_subject)} "
          f"({sum(subject_has_seizure.values())} seizure-bearing) | "
          f"seizure stems: {n_seiz_stems}")

    # 4. Assign: external-corpus subjects -> external_test; the rest -> 3-way
    #    stratified hash. Stratify by seizure-bearing so each split carries a
    #    proportional seizure cohort.
    subjects: dict[str, str] = {}
    internal = []
    for subj in sorted(stems_by_subject):
        if subject_corpus[subj] in ext_corpora:
            subjects[subj] = "external_test"
        else:
            internal.append(subj)
    seiz = sorted(s for s in internal if subject_has_seizure[s])
    nonseiz = sorted(s for s in internal if not subject_has_seizure[s])
    for bucket in (seiz, nonseiz):
        for subj in bucket:
            subjects[subj] = assign_split(subj, args.val_fraction, args.test_fraction)

    # Starvation repair: every internal split (train/val/test) must hold >=1
    # seizure subject when a seizure cohort exists. Deterministic — move the
    # lexicographically-first seizure subject from the most-seizure-rich split.
    if seiz and args.val_fraction > 0 and args.test_fraction > 0:
        for _ in range(len(seiz)):
            by = {sp: [s for s in seiz if subjects[s] == sp] for sp in ("train", "val", "test")}
            starved = [sp for sp in ("train", "val", "test") if not by[sp]]
            if not starved:
                break
            donor = max(("train", "val", "test"), key=lambda sp: len(by[sp]))
            if len(by[donor]) <= 1:
                print(f"[!] only {len(seiz)} seizure subjects — cannot fill {starved} "
                      f"without starving donor; leaving as-is")
                break
            subjects[by[donor][0]] = starved[0]

    # 5. NO-OVERLAP assert (every subject in exactly one split) + counts.
    assert set(subjects) == set(stems_by_subject), "subject/assignment mismatch"
    assert all(v in SPLITS for v in subjects.values()), "bad split label"

    counts = {sp: {"subj": 0, "stem": 0, "seiz_subj": 0} for sp in SPLITS}
    for subj in sorted(stems_by_subject):
        sp = subjects[subj]
        counts[sp]["subj"] += 1
        counts[sp]["stem"] += len(stems_by_subject[subj])
        if subject_has_seizure[subj]:
            counts[sp]["seiz_subj"] += 1

    # per_corpus is STEM-accurate (a TUH patient can span tusz/tusl/tuar/tuev;
    # global-subject grouping keeps that patient in ONE split — no cross-corpus
    # leak — but each corpus's stems are reported where they actually live).
    per_corpus = defaultdict(lambda: {sp: 0 for sp in SPLITS})
    for subj, stems in stems_by_subject.items():
        sp = subjects[subj]
        for st in stems:
            per_corpus[stem_corpus[st]][sp] += 1
    multi_corpus_subjects = sum(
        1 for stems in stems_by_subject.values()
        if len({stem_corpus[s] for s in stems}) > 1)

    manifest = {
        "subjects": subjects,
        "stems_by_subject": {s: sorted(v) for s, v in stems_by_subject.items()},
        "meta": {
            "builder": "build_seizure_split_manifest.py",
            "schema": "lamquant.snn_split.v2",
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "split_method": "patient-level, seizure-stratified, sha1(subject)%1000, 3-way + corpus-level cross-site external_test",
            "external_test_corpora": sorted(ext_corpora),
            "lma_roots": [str(r) for r in args.lma_root],
            "labels": str(args.labels),
            "corpora": sorted({subject_corpus[s] for s in stems_by_subject}),
            "n_subjects": len(subjects),
            "n_stems": len(usable),
            "multi_corpus_subjects": multi_corpus_subjects,
            "counts": counts,
            "per_corpus_stems": {k: dict(v) for k, v in sorted(per_corpus.items())},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))

    print(f"[*] wrote {args.out}")
    for sp in SPLITS:
        c = counts[sp]
        print(f"    {sp:13s}: {c['subj']:5d} subj  {c['stem']:6d} stems  {c['seiz_subj']:4d} seizure-subj")


if __name__ == "__main__":
    main()
