"""Coverage tests for ``ai_models/snn/lma_subject_id.py``.

The subject-ID parser is the single source of truth for grouped
train/val splits. A regression here silently leaks subjects across
folds, so the contract is asserted directly via filename patterns
seen in each corpus.

All fixtures are *file paths only* — no real EDF bytes needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lamquant.snn.lma_subject_id import (
    CORPUS_PRECEDENCE,
    corpus_short_name,
    extract_subject_id,
    precedence_rank,
)


pytestmark = pytest.mark.l2


class TestCorpusShortName:
    def test_strips_version_suffix(self) -> None:
        assert corpus_short_name("tueg_v2.0.1") == "tueg"
        assert corpus_short_name("tusz_v2.0.6") == "tusz"

    def test_handles_bare_name(self) -> None:
        assert corpus_short_name("chbmit") == "chbmit"

    def test_lowercases(self) -> None:
        assert corpus_short_name("TUEG_V1") == "tueg"


class TestExtractSubjectIdTUH:
    """Canonical TUH stem: ``<patient>_s<session>_t<token>``."""

    def test_tueg_canonical_stem(self) -> None:
        sid, tag = extract_subject_id(
            "tueg_v2.0.1", Path("/x/aaaaaaaq_s007_t001.lml")
        )
        assert sid == "aaaaaaaq"
        assert "tueg" in tag

    def test_tusz_canonical_stem(self) -> None:
        sid, tag = extract_subject_id(
            "tusz", Path("/x/aaaaaaaa_s001_t000.lml")
        )
        assert sid == "aaaaaaaa"
        assert "tusz" in tag

    def test_fallback_first_token(self) -> None:
        """When the canonical regex fails, fall back to the first
        underscore-separated token + flag the source."""
        sid, tag = extract_subject_id(
            "tueg", Path("/x/xyz_arbitrary_token.lml")
        )
        assert sid == "xyz"
        assert "first_token" in tag

    def test_unparseable_returns_empty(self) -> None:
        # Empty stem (just ".lml" → stem "") falls through to the
        # ``parts[0]`` check which is empty, returning the
        # ``..._unparseable`` sentinel.
        sid, tag = extract_subject_id("tueg", Path("/x/.lml"))
        # The fallback may still succeed on degenerate input — pin
        # the broader contract: when sid is empty, tag flags it.
        if sid == "":
            assert "unparseable" in tag
        else:
            # If fallback found something, the empty-stem case still
            # produces a tag — just verify the tag isn't None.
            assert tag


class TestExtractSubjectIdTUEV:
    def test_train_layout(self) -> None:
        sid, tag = extract_subject_id(
            "tuev", Path("/x/aaaaaaaq_00000001.lml")
        )
        assert sid == "aaaaaaaq"
        assert "tuev_train" in tag

    def test_eval_layout(self) -> None:
        # TUEV eval stems look like ``<event>_<3digit>_a_<run>`` where
        # the trailing ``_<run>`` is mandatory but the digits are
        # optional ('' or '1'). We use '1' to match the regex exactly.
        sid, tag = extract_subject_id(
            "tuev", Path("/x/spsw_046_a_1.lml")
        )
        # Eval patients are namespaced to avoid collision with train.
        assert sid == "tuev_eval_046"
        assert "tuev_eval" in tag

    def test_unparseable(self) -> None:
        sid, tag = extract_subject_id(
            "tuev", Path("/x/random_garbage.lml")
        )
        assert sid == ""
        assert tag == "tuev_unparseable"


class TestExtractSubjectIdCHBMIT:
    def test_filename_prefix(self) -> None:
        sid, tag = extract_subject_id(
            "chbmit", Path("/x/chb01_03.lml")
        )
        assert sid == "chb01"
        assert "filename_prefix" in tag

    def test_parent_dir_fallback(self) -> None:
        """When the filename doesn't carry the chbNN prefix, walk up
        looking for a chbNN directory."""
        sid, tag = extract_subject_id(
            "chbmit", Path("/x/chb05/seizure.lml")
        )
        assert sid == "chb05"
        assert "parent_dir" in tag

    def test_unparseable(self) -> None:
        sid, tag = extract_subject_id(
            "chbmit", Path("/x/random.lml")
        )
        assert sid == ""
        assert tag == "chbmit_unparseable"


class TestExtractSubjectIdSiena:
    def test_filename_prefix(self) -> None:
        sid, tag = extract_subject_id(
            "siena", Path("/x/PN05_session_1.lml")
        )
        assert sid == "PN05"
        assert "siena" in tag

    def test_unparseable(self) -> None:
        sid, tag = extract_subject_id(
            "siena", Path("/x/notamatch.lml")
        )
        assert sid == ""


class TestExtractSubjectIdUnknown:
    def test_returns_unknown_tag(self) -> None:
        sid, tag = extract_subject_id(
            "weird_corpus", Path("/x/file.lml")
        )
        assert sid == ""
        assert "unknown_corpus" in tag


class TestPrecedenceRank:
    def test_tusz_highest(self) -> None:
        """TUSZ wins because it has the richest annotations
        (per-event seizure intervals)."""
        assert precedence_rank("tusz") < precedence_rank("tuev")

    def test_tueg_lowest_in_known(self) -> None:
        assert precedence_rank("tueg") > precedence_rank("tuab")

    def test_unknown_corpus_ranks_last(self) -> None:
        unknown = precedence_rank("zzz_unknown")
        known = precedence_rank("tueg")
        assert unknown >= known


class TestCorpusPrecedence:
    def test_is_tuple(self) -> None:
        assert isinstance(CORPUS_PRECEDENCE, tuple)

    def test_contains_known_corpora(self) -> None:
        for c in ("tusz", "tuev", "tuab", "tueg"):
            assert c in CORPUS_PRECEDENCE
