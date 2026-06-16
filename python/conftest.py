"""Pytest config for the BLUT-owned LamQuant training/preprocessing tree.

Migration (2026-05-29, Phase 3): the training + preprocessing scripts
carved out of LamQuant-Neural's ``ai_models/`` (MOVE-B) live here under
``lamquant/<area>/``. Like the monorepo/Neural tree they came from, those
modules import sibling modules by bare name (e.g.
``from snn_training_config import ...``, ``from subband_preprocess import
...``) and a script's own area dir is expected to be on ``sys.path``.

Most modules self-insert their area dir at import time, but the unit
tests under ``lamquant/<area>/tests/`` import the modules cold, so we put
every area dir on ``sys.path`` here — mirroring the Neural-repo conftest.

Tests that need the neural model definitions import ``lamquant_neural``
(the private wheel); those collect/skip cleanly when it is absent.
"""

import sys
from pathlib import Path

import pytest

_PY_ROOT = Path(__file__).resolve().parent          # blut/python
_LAMQUANT = _PY_ROOT / "lamquant"
# Meta-repo root (parent of the blut/ submodule). The sibling
# LamQuant-Lossless checkout owns the real-EDF + lml-CLI test fixtures.
_META_ROOT = _PY_ROOT.parent.parent
_LOSSLESS_ROOT = _META_ROOT / "LamQuant-Lossless"

# Area dirs first (bare-name sibling imports), then the python root last
# so the ``lamquant.<area>`` package form also resolves.
for _rel in (
    "student",
    "oracle",
    "snn",
    "dataset",
    "decoder",
    "common",
):
    _p = (_LAMQUANT / _rel).resolve()
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))


# Sequestered code (ADR 0051 cookbook rebuild): lamquant/deprecated/ holds
# retired scripts + their tests (git-mv'd here, reversible). Keep them out of
# collection — they exercise now-dead modules by design. Scoped to THIS tree's
# deprecated/ subtree (not any directory merely named "deprecated") and
# independent of which rootdir pytest picks (meta vs blut), unlike a
# rootdir-relative collect_ignore path.
_DEPRECATED = (_LAMQUANT / "deprecated").resolve()


def pytest_ignore_collect(collection_path):
    p = collection_path.resolve()
    return p == _DEPRECATED or _DEPRECATED in p.parents


# ---------------------------------------------------------------------------
# Real-EDF + lml-CLI fixtures (no synthetic data — user direction 2026-05-21).
#
# The seizure-aware LMA dataset coverage test (lamquant/snn/tests/
# test_lma_dataset_coverage.py) builds a real .lma from a real EDF via the
# `lml` CLI. Those resolvers live in the sibling PUBLIC LamQuant-Lossless
# checkout (``tests.fixtures`` / ``tests.helpers.data_paths``). Re-expose
# them here so the relocated test runs when the corpus + binary are present
# and skips cleanly when they are not (CI without reference_software/).
# ---------------------------------------------------------------------------
def _lossless_fixtures():
    """Import the Lossless test-fixture helpers, or None if unavailable."""
    if not (_LOSSLESS_ROOT / "tests" / "fixtures").is_dir():
        return None
    if str(_LOSSLESS_ROOT) not in sys.path:
        sys.path.insert(0, str(_LOSSLESS_ROOT))
    try:
        from tests.fixtures import require_real_test_edf
        from tests.helpers.data_paths import lml_cli_binary as _resolve_lml
    except Exception:
        return None
    return require_real_test_edf, _resolve_lml


@pytest.fixture(scope="session")
def real_test_edf():
    """Path to a real small EDF (pyedflib test_generator). Skips if absent."""
    helpers = _lossless_fixtures()
    if helpers is None:
        pytest.skip(
            "LamQuant-Lossless test fixtures not reachable — the real-EDF "
            "corpus lives in the sibling Lossless checkout."
        )
    require_real_test_edf, _ = helpers
    return require_real_test_edf()


@pytest.fixture(scope="session")
def lml_cli_binary():
    """Path to the `lml` Rust CLI binary. Skips if not built."""
    helpers = _lossless_fixtures()
    if helpers is None:
        pytest.skip(
            "LamQuant-Lossless test fixtures not reachable — cannot resolve "
            "the lml CLI binary."
        )
    _, _resolve_lml = helpers
    p = _resolve_lml()
    if p is None:
        pytest.skip(
            "lml binary not built — run "
            "`cargo build --release --bin lml` in LamQuant-Lossless."
        )
    return p
