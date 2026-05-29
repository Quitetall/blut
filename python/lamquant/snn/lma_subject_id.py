"""lma_subject_id.py — canonical subject_id parser for LMA bundling + training.

Single source of truth for subject_id extraction. Used by:
  - scripts/audit_subject_id_extraction.py (Phase 0.1 gate)
  - scripts/bulk_lml_to_lma.py (per-LMA meta.json stamp)
  - scripts/build_snn_train_val_split.py (subject-grouped split)
  - ai_models/snn/lma_dataset.py (defensive subject-bleed assertion)

If this parser is wrong, training+validation leak silently. Phase 0.1
audit validates it across 100 samples per corpus before any migration.
Audit pass + this module being the single import enforces consistency.

Source tags (returned alongside subject_id) record which rule produced
the ID; if a stale subject appears post-rename, the tag tells you which
corpus + rule was responsible.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final, Tuple

# Canonical TUH stem: <patient>_s<session>_t<token>
_TUH_STEM_RE = re.compile(r"^([a-z]{8})_s\d{3}_t\d{3}$")
# TUEV train layout: <patient>_<8digit_token>
_TUEV_TRAIN_RE = re.compile(r"^([a-z]{8})_\d{8}$")
# TUEV eval layout: <event>_<3digit_patient>_a_<optional_run>
# event labels: bckg, gped, pled, spsw, eyem, artf
_TUEV_EVAL_RE = re.compile(r"^(?:bckg|gped|pled|spsw|eyem|artf)_(\d{3})_a_(?:\d+)?$")

# Canonical corpus precedence: richer annotation wins on cross-corpus
# stem collision. TUSZ > TUEV > TUEP > TUSL > TUAR > TUAB > TUEG.
# Compile-time constant — never load from config.
CORPUS_PRECEDENCE: Final[Tuple[str, ...]] = (
    "tusz", "tuev", "tuep", "tusl", "tuar", "tuab", "tueg",
)


def corpus_short_name(corpus_dir_name: str) -> str:
    """Strip the version suffix from a corpus dir name.

    'tueg_v2.0.1' → 'tueg'
    'tusz_v2.0.6' → 'tusz'
    """
    return corpus_dir_name.lower().split("_")[0]


def extract_subject_id(corpus: str, lml_path: Path) -> Tuple[str, str]:
    """Return (subject_id, source_tag).

    Args:
        corpus: corpus subdir name (e.g. 'tueg_v2.0.1' or just 'tueg').
        lml_path: full path to the .lml file.

    Returns:
        (subject_id, source_tag). Empty subject_id signals an anomaly
        that the audit script should flag.
    """
    stem = lml_path.stem
    corpus_short = corpus_short_name(corpus)

    if corpus_short == "tuev":
        # TUEV mixes two layouts:
        #   train/<patient>/<patient>_<8digit>.lml — patient-indexed
        #   eval/<3digit>/<event>_<3digit>_a_[<run>].lml — event-indexed
        # Eval files use the 3-digit dir as patient. Namespace with
        # "tuev_eval_" so eval patient "001" can't collide with a train
        # patient happening to be the string "001".
        m = _TUEV_TRAIN_RE.match(stem)
        if m:
            return (m.group(1), "tuev_train_filename_regex")
        m = _TUEV_EVAL_RE.match(stem)
        if m:
            return (f"tuev_eval_{m.group(1)}", "tuev_eval_filename_event_regex")
        return ("", "tuev_unparseable")

    if corpus_short in ("tueg", "tusz", "tuab", "tuep", "tusl", "tuar"):
        m = _TUH_STEM_RE.match(stem)
        if m:
            return (m.group(1), f"{corpus_short}_filename_regex")
        parts = stem.split("_")
        if parts and parts[0]:
            return (parts[0], f"{corpus_short}_filename_first_token")
        return ("", f"{corpus_short}_unparseable")

    if corpus_short.startswith("chb"):
        m = re.match(r"^(chb\d{2})_", stem)
        if m:
            return (m.group(1), "chbmit_filename_prefix")
        for part in reversed(lml_path.parts):
            if re.match(r"^chb\d{2}$", part):
                return (part, "chbmit_parent_dir")
        return ("", "chbmit_unparseable")

    if corpus_short == "siena":
        m = re.match(r"^(PN\d{2})", stem)
        if m:
            return (m.group(1), "siena_filename_prefix")
        return ("", "siena_unparseable")

    return ("", f"unknown_corpus_{corpus_short}")


def precedence_rank(corpus: str) -> int:
    """Lower rank = higher precedence. Unknown corpora rank last.

    Used by the migration script to pick which corpus's LML wins when
    the same `<stem>.lml` exists in multiple corpora.
    """
    short = corpus_short_name(corpus)
    try:
        return CORPUS_PRECEDENCE.index(short)
    except ValueError:
        return len(CORPUS_PRECEDENCE) + 1
