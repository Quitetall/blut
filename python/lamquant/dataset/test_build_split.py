"""Tests for build_seizure_split_manifest — the FDA-grade 3-way patient-disjoint
split builder. Verifies the invariants that make the split clinically valid:
no patient leakage, no seizure-starved split, corpus-level cross-site holdout,
determinism. Runs on a synthetic tree (fast, hermetic)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MOD = "lamquant.dataset.build_seizure_split_manifest"
import lamquant.dataset.build_seizure_split_manifest as B


# ---- unit: subject extraction across corpus naming conventions ----
@pytest.mark.parametrize("stem,subj", [
    ("aaaaaaac_s001_t000", "aaaaaaac"),   # canonical TUH
    ("aaaaaaar_00000001", "aaaaaaar"),    # TUEV 8-digit
    ("chb01_03", "chb01"),                # CHB-MIT
    ("chb21_19", "chb21"),
    ("PN00-1", "PN00"),                   # Siena
    ("PN12-1.2", "PN12"),
])
def test_subject_of(stem, subj):
    assert B.subject_of(stem) == subj


def test_assign_split_deterministic_and_3way():
    # same subject -> same split, always
    for s in ("chb01", "aaaaaaac", "PN03"):
        a = B.assign_split(s, 0.1, 0.1)
        assert a == B.assign_split(s, 0.1, 0.1)
        assert a in ("train", "val", "test")
    # fractions roughly honored over many synthetic subjects
    subs = [f"subj{i:05d}" for i in range(5000)]
    from collections import Counter
    c = Counter(B.assign_split(s, 0.1, 0.1) for s in subs)
    assert 0.07 < c["val"] / 5000 < 0.13
    assert 0.07 < c["test"] / 5000 < 0.13
    assert c["train"] / 5000 > 0.74


def test_corpus_of():
    roots = ["/data/lma", "/data/lma_expand"]
    assert B.corpus_of("/data/lma/tusz_v2.0.6/x.lma", roots) == "tusz_v2.0.6"
    assert B.corpus_of("/data/lma_expand/tuev/y.lma", roots) == "tuev"


def _mk_tree(tmp, spec):
    """spec: {corpus: [(stem, has_seizure), ...]}. Build lma + label tree."""
    lma = tmp / "lma"; labels = tmp / "labels"
    labels.mkdir(parents=True)
    for corpus, stems in spec.items():
        (lma / corpus).mkdir(parents=True)
        for stem, seiz in stems:
            (lma / corpus / f"{stem}.lma").write_bytes(b"x")
            arr = np.zeros((8, 100), dtype=np.uint8)
            if seiz:
                arr[:, 10:20] = 2
            np.savez(labels / f"{stem}_labels.npz", activity_labels=arr,
                     source=f"{stem}.edf", annotation_file="syn")
    return lma, labels


def _build(tmp, lma, labels, extra=()):
    out = tmp / "m.json"
    cmd = [sys.executable, "-m", MOD, "--lma-root", str(lma),
           "--labels", str(labels), "--out", str(out),
           "--val-fraction", "0.2", "--test-fraction", "0.2", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(out.read_text())


def test_integration_invariants(tmp_path):
    # 20 TUSZ-like subjects (12 seizure), + a Siena cross-site corpus.
    spec = {
        "tusz_v2.0.6": [(f"sub{i:03d}aaaa_s001_t000", i < 12) for i in range(20)],
        "siena": [(f"PN{i:02d}-1", True) for i in range(6)],
    }
    lma, labels = _mk_tree(tmp_path, spec)
    m = _build(tmp_path, lma, labels, extra=["--external-test-corpus", "siena"])
    subs = m["subjects"]

    # 1. patient-disjoint: every subject exactly one split (dict guarantees) +
    #    no stem shared across subjects.
    seen = set()
    for subj, stems in m["stems_by_subject"].items():
        for st in stems:
            assert st not in seen, f"stem {st} in two subjects"
            seen.add(st)

    # 2. cross-site: every Siena subject -> external_test, none elsewhere.
    siena_subs = [s for s in subs if s.startswith("PN")]
    assert siena_subs and all(subs[s] == "external_test" for s in siena_subs)
    assert all(subs[s] != "external_test" for s in subs if not s.startswith("PN"))

    # 3. no seizure-starved internal split.
    seiz_by_split = {sp: 0 for sp in ("train", "val", "test")}
    for subj, sp in subs.items():
        if sp == "external_test":
            continue
        if any((np.load(labels / f"{st}_labels.npz")["activity_labels"] == 2).any()
               for st in m["stems_by_subject"][subj]):
            seiz_by_split[sp] += 1
    assert all(n >= 1 for n in seiz_by_split.values()), seiz_by_split

    # 4. meta has 3-way + external + per-corpus
    assert m["meta"]["schema"] == "lamquant.snn_split.v2"
    assert m["meta"]["external_test_corpora"] == ["siena"]
    assert set(m["meta"]["counts"]) >= {"train", "val", "test", "external_test"}


def test_determinism(tmp_path):
    spec = {"tusz_v2.0.6": [(f"sub{i:03d}aaaa_s001_t000", i % 2 == 0) for i in range(30)]}
    lma, labels = _mk_tree(tmp_path, spec)
    a = _build(tmp_path, lma, labels)["subjects"]
    b = _build(tmp_path, lma, labels)["subjects"]
    assert a == b
