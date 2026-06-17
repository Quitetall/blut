"""Tests for the data ingredient registry (ADR 0050/0051).

Three ``kind="data"`` specs, each ``cache_relevant=True``:

  * ``lma_snn``      — seizure-aware ``snn.lma_dataset.LmaDataset`` (SNN trainers).
  * ``lma_l3``       — ``lamquant_codec.training.LmaL3Dataset`` (MAE + L3 teacher).
  * ``lma_typed_l3`` — ``student.lma_typed_adapter.LmaTypedL3Dataset`` (train_joint).

The heavy bodies need a REAL ``.lma`` corpus (no synthetic data — user
direction 2026-05-21), so the fixture builds a real EDF via ``pyedflib`` and
encodes it with the ``lml`` Rust CLI, exactly as
``snn/tests/test_lma_dataset_coverage.py`` builds its fixture. The whole module
skips cleanly when the codec wheel / the lml binary / pyedflib are unavailable.

Equivalence is proven against the INLINE construction the trainers do, with the
SAME kwargs: identical ``len`` + first-item / base-dataset tensor-equality.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Importing this module registers the data specs without going through the
# package __init__ (which the main agent wires up separately).
import lamquant.ingredients.data._specs  # noqa: F401
from lamquant.ingredients import build_ingredient, get_spec, list_ingredients
from lamquant.ingredients.data import (
    build_lma_dataloader_kwargs,
    expand_lma_roots,
)

# ---------------------------------------------------------------------------
# Capability gates. The dataset bodies read real LMAs via the lamquant_core
# PyO3 wheel + decode through the codec wheel; the fixture needs pyedflib + the
# lml CLI. Skip the whole module (not individual tests) when any is absent so a
# bare CI checkout collects cleanly.
# ---------------------------------------------------------------------------
try:
    import lamquant_core  # noqa: F401
    import lamquant_codec  # noqa: F401
    _HAS_WHEELS = True
except Exception:
    _HAS_WHEELS = False

try:
    import pyedflib  # noqa: F401
    _HAS_PYEDFLIB = True
except Exception:
    _HAS_PYEDFLIB = False


def _resolve_lml() -> Path | None:
    """The lml CLI binary, or None.

    Resolution order (portable first, no machine-specific hardcode):
      1. ``$LML_BIN`` env override (CI / dev points it wherever).
      2. ``<meta-repo>/target/release/lml`` relative to this file.
      3. ``lml`` on ``$PATH``.
    """
    env = os.environ.get("LML_BIN")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return Path(env)
    cand = Path(__file__).resolve().parents[6] / "target" / "release" / "lml"
    if cand.is_file() and os.access(cand, os.X_OK):
        return cand
    from shutil import which
    w = which("lml")
    return Path(w) if w else None


_LML = _resolve_lml()

pytestmark = pytest.mark.skipif(
    not (_HAS_WHEELS and _HAS_PYEDFLIB and _LML is not None),
    reason="data ingredient suite needs the lamquant_core + lamquant_codec "
           "wheels, pyedflib, and the lml CLI binary.",
)

# Canonical 21-channel 10-20 montage (CHANNEL_PRESETS[21]) — the fixture EDF's
# channel labels resolve directly through channel_resolver.
_CHANS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz", "A1", "A2",
]
# Two real recordings — one per split subject — so the subject-bleed-checked
# train/val pair both materialise (the typed adapter + lma_snn(split="train")
# build a val dataset too). Distinct subjects + distinct stems = no bleed.
_TRAIN_SUBJ, _TRAIN_STEM = "aaaaaaaa", "aaaaaaaa_s001_t000"
_VAL_SUBJ, _VAL_STEM = "bbbbbbbb", "bbbbbbbb_s001_t000"
_STEM = _TRAIN_STEM  # back-compat alias for the single-stem assertions.

_FS = 256
_SECONDS = 40  # 40 s @ 256 Hz -> 10000 samples @ 250 Hz -> 4 windows.


def _build_one_lma(work, stem, seed):
    """Build a real EDF for ``stem`` and encode it to a real ``.lma`` under
    ``work``, appending meta.json + a zero-label NPZ. Returns the ``.lma`` path
    (or skips the suite when the toolchain misbehaves)."""
    import numpy as np

    nsamp = _SECONDS * _FS
    edf = work / f"{stem}.edf"
    rng = np.random.default_rng(seed)
    headers, sigs = [], []
    for ch in _CHANS:
        sigs.append((rng.standard_normal(nsamp) * 40.0).astype(np.float64))
        headers.append(dict(
            label=ch, dimension="uV", sample_frequency=_FS,
            physical_max=500.0, physical_min=-500.0,
            digital_max=32767, digital_min=-32768,
            transducer="", prefilter=""))
    writer = pyedflib.EdfWriter(
        str(edf), len(_CHANS), file_type=pyedflib.FILETYPE_EDFPLUS)
    writer.setSignalHeaders(headers)
    writer.writeSamples(sigs)
    writer.close()

    lma = work / f"{stem}.lma"
    r = subprocess.run(
        [str(_LML), "encode", str(edf), "-o", str(lma), "-q"],
        capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not lma.is_file():
        pytest.skip(f"lml encode failed (rc={r.returncode}): {r.stderr[-300:]}")

    # meta.json — LmaL3Dataset's window scan reads n_samples_resampled (10000 =
    # 40 s resampled 256 Hz -> 250 Hz).
    meta = work / f"{stem}.meta.json"
    meta.write_text(json.dumps({"n_samples_resampled": nsamp * 250 // _FS}))
    r = subprocess.run(
        [str(_LML), "append", "-q", "--as", "meta.json", str(lma), str(meta)],
        capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        pytest.skip(f"lml append meta.json failed: {r.stderr[-300:]}")

    # labels NPZ — keeps the window count deterministic (4 all-quiet windows).
    from lamquant.snn import lma_dataset as lds
    activity = np.zeros((8, 4 * lds.LABEL_PER_WINDOW), dtype=np.uint8)
    labels_npz = work / f"{stem}_labels.npz"
    np.savez_compressed(
        labels_npz, activity_labels=activity, source=f"{stem}.edf")
    r = subprocess.run(
        [str(_LML), "append", "-q", "--as", f"labels/{stem}_labels.npz",
         str(lma), str(labels_npz)],
        capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        pytest.skip(f"lml append labels failed: {r.stderr[-300:]}")
    return lma


# ---------------------------------------------------------------------------
# Fixture: build a two-recording real LMA corpus (one train stem + one val
# stem) + a subject-grouped split manifest. Module-scoped (the encode is the
# slow part).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lma_corpus(tmp_path_factory):
    work = tmp_path_factory.mktemp("data_specs_lma")
    train_lma = _build_one_lma(work, _TRAIN_STEM, seed=0)
    _build_one_lma(work, _VAL_STEM, seed=1)

    manifest = work / "split.json"
    manifest.write_text(json.dumps({
        "schema": "lamquant.snn_split.v1",
        "seed": 42,
        "split_strategy": "subject_grouped",
        "val_fraction": 0.10,
        "subjects": {_TRAIN_SUBJ: "train", _VAL_SUBJ: "val"},
        "stems_by_subject": {
            _TRAIN_SUBJ: [_TRAIN_STEM],
            _VAL_SUBJ: [_VAL_STEM],
        },
        "promotions": [],
        "summary": {},
    }))

    return {"root": work, "lma": train_lma, "manifest": manifest,
            "stem": _TRAIN_STEM}


# ===========================================================================
# Registration + spec contract (no corpus needed).
# ===========================================================================

def test_all_three_data_specs_registered():
    names = list_ingredients("data")
    assert "lma_snn" in names
    assert "lma_l3" in names
    assert "lma_typed_l3" in names


def test_data_specs_are_cache_relevant():
    # The corpus a stage trains on is part of the artifact identity — two
    # different corpora must never collide on one stage cache key.
    for n in ("lma_snn", "lma_l3", "lma_typed_l3"):
        assert get_spec("data", n).cache_relevant is True


def test_lma_l3_seed_default_is_zero():
    # The teacher OMITS seed= (LmaL3Dataset default 0) — the cfg default must be
    # 0 so the teacher stays byte-identical when the recipe doesn't pin seed.
    assert get_spec("data", "lma_l3").config_cls().seed == 0


def test_lma_typed_l3_defaults_mirror_train_joint():
    cfg = get_spec("data", "lma_typed_l3").config_cls()
    assert cfg.windows_per_epoch == 50000
    assert cfg.val_windows == 50000
    assert cfg.return_fullband is False
    assert cfg.seed == 0
    assert cfg.max_windows_per_file is None


def test_unknown_key_fails_closed():
    with pytest.raises(ValueError):
        build_ingredient("data", "lma_l3", {"not_a_real_field": 1})


# ===========================================================================
# expand_lma_roots — identical sorted-dedup list to the inline trainer copies.
# ===========================================================================

def _inline_expand_lma_roots(roots):
    """Verbatim copy of the helper that WAS inline in both SNN trainers."""
    lma_paths = []
    for r in roots:
        r = Path(r)
        if r.is_file() and r.suffix == ".lma":
            lma_paths.append(r)
            continue
        if not r.is_dir():
            raise FileNotFoundError(f"--lma-root not found: {r}")
        found = sorted(r.glob("*/*.lma")) or sorted(r.glob("*.lma"))
        if not found:
            raise RuntimeError(f"no .lma archives under {r}")
        lma_paths.extend(found)
    seen = set()
    return [p for p in lma_paths if not (str(p) in seen or seen.add(str(p)))]


def test_expand_lma_roots_matches_inline_on_dir(lma_corpus):
    root = lma_corpus["root"]
    assert expand_lma_roots([root]) == _inline_expand_lma_roots([root])


def test_expand_lma_roots_matches_inline_on_file(lma_corpus):
    lma = lma_corpus["lma"]
    assert expand_lma_roots([lma]) == _inline_expand_lma_roots([lma])


def test_expand_lma_roots_dedups_repeated_root(lma_corpus):
    # Two copies of the same root collapse to a first-seen-wins unique list.
    lma = lma_corpus["lma"]
    out = expand_lma_roots([lma, lma])
    assert out == [Path(lma)]
    assert out == _inline_expand_lma_roots([lma, lma])


def test_expand_lma_roots_missing_root_raises():
    with pytest.raises(FileNotFoundError):
        expand_lma_roots(["/no/such/dir/at/all"])


# ===========================================================================
# build_lma_dataloader_kwargs — the L3_CACHE_DIR-conditioned worker default.
# ===========================================================================

class _FakeCudaDevice:
    type = "cuda"


class _FakeCpuDevice:
    type = "cpu"


def _inline_dl_kwargs(num_workers_arg, device):
    """Verbatim copy of the preamble that WAS inline in both SNN trainers."""
    _default_workers = 4 if os.environ.get("L3_CACHE_DIR") else 2
    num_workers = num_workers_arg if num_workers_arg is not None else \
        int(os.environ.get("LMA_NUM_WORKERS", str(_default_workers)))
    _dl_kwargs = {}
    if num_workers > 0:
        _dl_kwargs["persistent_workers"] = True
        _dl_kwargs["prefetch_factor"] = int(
            os.environ.get("LMA_PREFETCH_FACTOR", "4"))
    pin = device.type == "cuda" and num_workers > 0
    return num_workers, _dl_kwargs, pin


@pytest.mark.parametrize("nw", [None, 0, 2])
@pytest.mark.parametrize("l3cache", [False, True])
@pytest.mark.parametrize("dev", [_FakeCpuDevice(), _FakeCudaDevice()])
def test_dataloader_kwargs_matches_inline(monkeypatch, nw, l3cache, dev):
    # Isolate the env knobs the helper reads.
    monkeypatch.delenv("LMA_NUM_WORKERS", raising=False)
    monkeypatch.delenv("LMA_PREFETCH_FACTOR", raising=False)
    if l3cache:
        monkeypatch.setenv("L3_CACHE_DIR", "/tmp/l3cache_probe")
    else:
        monkeypatch.delenv("L3_CACHE_DIR", raising=False)

    got = build_lma_dataloader_kwargs(nw, dev)
    exp = _inline_dl_kwargs(nw, dev)
    assert got == exp
    # When the arg is None and an L3 cache is configured, the default is 4; when
    # not, it is 2 — the load-bearing cache-conditioned default.
    if nw is None:
        assert got[0] == (4 if l3cache else 2)


# ===========================================================================
# (1) lma_snn — equals the inline LmaDataset construction.
# ===========================================================================

def _inline_lma_snn(lma_paths, manifest, *, split, mwpf, seq):
    from lamquant.snn.lma_dataset import LmaDataset
    return LmaDataset(
        lma_paths=lma_paths, split=split, split_manifest_path=manifest,
        max_windows_per_file=mwpf, seq_windows=seq)


def test_lma_snn_train_equals_inline(lma_corpus):
    import torch
    root, manifest = lma_corpus["root"], lma_corpus["manifest"]
    out = build_ingredient(
        "data", "lma_snn",
        {"lma_root": [root], "split_manifest": manifest,
         "split": "train", "max_windows_per_file": 5, "seq_windows": 1})
    assert out["val_ds"] is not None  # split=="train" -> paired val built.

    paths = expand_lma_roots([root])
    inline = _inline_lma_snn(paths, manifest, split="train", mwpf=5, seq=1)
    got = out["train_ds"]
    assert len(got) == len(inline)
    gl3, glab = got[0]
    il3, ilab = inline[0]
    assert torch.equal(gl3, il3)
    assert torch.equal(glab, ilab)


def test_lma_snn_single_split_has_no_val(lma_corpus):
    # The SSL pretrain passes split="train" too, but the controller is the only
    # caller that wants val; a non-train split yields val_ds None.
    root, manifest = lma_corpus["root"], lma_corpus["manifest"]
    out = build_ingredient(
        "data", "lma_snn",
        {"lma_root": [root], "split_manifest": manifest, "split": "val"})
    assert out["val_ds"] is None
    assert out["train_ds"] is not None


def test_lma_snn_accepts_scalar_root(lma_corpus):
    # A recipe may pin a single path instead of the nargs='+' list.
    root, manifest = lma_corpus["root"], lma_corpus["manifest"]
    out = build_ingredient(
        "data", "lma_snn",
        {"lma_root": root, "split_manifest": manifest, "split": "train"})
    assert len(out["train_ds"]) == 4


# ===========================================================================
# (2) lma_l3 — equals the inline LmaL3Dataset construction.
# ===========================================================================

def _inline_lma_l3(root, manifest, *, wpe, mw, seed):
    from lamquant_codec.training import LmaL3Dataset, load_split_stems
    stems, _ = load_split_stems(manifest, "train")
    return LmaL3Dataset(
        lma_root=root, file_stems=stems,
        windows_per_epoch=wpe, max_windows=mw, seed=seed)


def test_lma_l3_equals_inline(lma_corpus):
    import torch
    root, manifest = lma_corpus["root"], lma_corpus["manifest"]
    got = build_ingredient(
        "data", "lma_l3",
        {"lma_root": str(root), "split_manifest": manifest,
         "windows_per_epoch": 8, "max_windows": None, "seed": 0})
    inline = _inline_lma_l3(
        str(root), manifest, wpe=8, mw=None, seed=0)
    assert len(got) == len(inline)
    assert len(got.windows) == len(inline.windows)
    # Same seed -> identical epoch index sampling -> tensor-equal first item.
    for a, b in zip(got[0], inline[0]):
        assert torch.equal(a, b)


def test_lma_l3_teacher_default_seed_is_byte_identical(lma_corpus):
    # The teacher omits seed=; the cfg default 0 must reproduce LmaL3Dataset's
    # own default-0 sampling exactly.
    import torch
    root, manifest = lma_corpus["root"], lma_corpus["manifest"]
    got = build_ingredient(
        "data", "lma_l3",
        {"lma_root": str(root), "split_manifest": manifest,
         "windows_per_epoch": 8})
    # Inline the teacher's EXACT call (no seed=).
    from lamquant_codec.training import LmaL3Dataset, load_split_stems
    stems, _ = load_split_stems(manifest, "train")
    inline = LmaL3Dataset(
        lma_root=str(root), file_stems=stems,
        windows_per_epoch=8, max_windows=None)
    for a, b in zip(got[0], inline[0]):
        assert torch.equal(a, b)


# ===========================================================================
# (3) lma_typed_l3 — equals the inline train_joint construction + the cache
#     env is primed identically BEFORE the datasets are built.
# ===========================================================================

def _prime_cache_env_inline():
    """The exact priming train_joint does inline (lines 647-654)."""
    from lamquant.common.cache_paths import apply_env
    apply_env()
    os.environ.setdefault("LMA_NUM_WORKERS", "2")


def test_lma_typed_l3_sets_cache_env(lma_corpus, monkeypatch, tmp_path):
    # apply_env() forces L3_CACHE_DIR / FB_CACHE_DIR / MEMMAP_DIR from one data
    # root; build() must have set them (and LMA_NUM_WORKERS) identically to the
    # inline priming, BEFORE constructing the datasets.
    data_root = str(tmp_path)
    monkeypatch.setenv("LAMQUANT_DATA_ROOT", data_root)
    monkeypatch.delenv("LMA_NUM_WORKERS", raising=False)
    monkeypatch.delenv("L3_CACHE_DIR", raising=False)
    monkeypatch.delenv("FB_CACHE_DIR", raising=False)
    monkeypatch.delenv("MEMMAP_DIR", raising=False)

    root, manifest = lma_corpus["root"], lma_corpus["manifest"]
    build_ingredient(
        "data", "lma_typed_l3",
        {"lma_root": root, "split_manifest": manifest,
         "windows_per_epoch": 8, "val_windows": 8, "return_fullband": False,
         "seed": 0})

    # Snapshot what build() set.
    got_env = {k: os.environ.get(k) for k in
               ("L3_CACHE_DIR", "FB_CACHE_DIR", "MEMMAP_DIR", "LMA_NUM_WORKERS")}
    # Recompute the inline priming on a fresh process-equivalent env.
    for k in ("L3_CACHE_DIR", "FB_CACHE_DIR", "MEMMAP_DIR", "LMA_NUM_WORKERS"):
        monkeypatch.delenv(k, raising=False)
    _prime_cache_env_inline()
    exp_env = {k: os.environ.get(k) for k in
               ("L3_CACHE_DIR", "FB_CACHE_DIR", "MEMMAP_DIR", "LMA_NUM_WORKERS")}
    assert got_env == exp_env
    assert got_env["LMA_NUM_WORKERS"] == "2"
    assert got_env["L3_CACHE_DIR"].startswith(data_root)


def test_lma_typed_l3_setdefault_preserves_explicit_workers(
        lma_corpus, monkeypatch, tmp_path):
    # An explicit LMA_NUM_WORKERS=0 (serial decode, debugging) must survive the
    # setdefault — build() must not clobber it to 2.
    monkeypatch.setenv("LAMQUANT_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("LMA_NUM_WORKERS", "0")
    root, manifest = lma_corpus["root"], lma_corpus["manifest"]
    build_ingredient(
        "data", "lma_typed_l3",
        {"lma_root": root, "split_manifest": manifest,
         "windows_per_epoch": 8, "val_windows": 8})
    assert os.environ["LMA_NUM_WORKERS"] == "0"


def test_lma_typed_l3_equals_inline(lma_corpus, monkeypatch, tmp_path):
    import torch
    monkeypatch.setenv("LAMQUANT_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("LMA_NUM_WORKERS", raising=False)
    root, manifest = lma_corpus["root"], lma_corpus["manifest"]

    train_ds, val_ds = build_ingredient(
        "data", "lma_typed_l3",
        {"lma_root": root, "split_manifest": manifest,
         "windows_per_epoch": 8, "val_windows": 6, "return_fullband": False,
         "seed": 0})

    # Inline the EXACT train_joint construction (with the same priming already
    # applied by build()).
    from lamquant.student.lma_typed_adapter import LmaTypedL3Dataset
    inline_train = LmaTypedL3Dataset(
        lma_root=root, split="train", split_manifest_path=manifest,
        windows_per_epoch=8, return_fullband=False, seed=0)
    inline_val = LmaTypedL3Dataset(
        lma_root=root, split="val", split_manifest_path=manifest,
        windows_per_epoch=6, return_fullband=False, seed=0 + 1)

    # windows_per_epoch is __len__; val uses val_windows; return_fullband flows.
    assert len(train_ds) == len(inline_train) == 8
    assert len(val_ds) == len(inline_val) == 6
    assert train_ds._return_fullband is inline_train._return_fullband is False
    # The wrapped seizure-aware base dataset is deterministic -> tensor-equal.
    assert len(train_ds._base) == len(inline_train._base)
    gl3, glab = train_ds._base[0]
    il3, ilab = inline_train._base[0]
    assert torch.equal(gl3, il3)
    assert torch.equal(glab, ilab)


def test_lma_typed_l3_max_windows_per_file_forwarded(
        lma_corpus, monkeypatch, tmp_path):
    # None -> the kwarg is omitted (adapter default); an int -> forwarded.
    monkeypatch.setenv("LAMQUANT_DATA_ROOT", str(tmp_path))
    root, manifest = lma_corpus["root"], lma_corpus["manifest"]
    train_ds, _ = build_ingredient(
        "data", "lma_typed_l3",
        {"lma_root": root, "split_manifest": manifest,
         "windows_per_epoch": 8, "val_windows": 8,
         "max_windows_per_file": 2})
    # Cap of 2 per file -> the wrapped base dataset holds <= 2 windows.
    assert len(train_ds._base) <= 2


def test_lma_typed_l3_builds_without_bare_area_dirs_on_path(
        lma_corpus, monkeypatch, tmp_path):
    # A recipe/framework caller of this ingredient need not have inserted
    # ``lamquant/student`` on sys.path the way train_joint does before its bare
    # ``from lma_typed_adapter import`` — the data spec must resolve the adapter
    # via its package path regardless. Strip the bare area dirs and confirm the
    # build still succeeds (regression for the package-form import fix).
    monkeypatch.setenv("LAMQUANT_DATA_ROOT", str(tmp_path))
    # Match the bare-name area dirs precisely (a path whose final two parts are
    # ``lamquant/student`` or ``lamquant/snn``), not any path that merely
    # contains the substring.
    def _is_area_dir(p):
        parts = Path(p).parts
        return len(parts) >= 2 and parts[-2] == "lamquant" and \
            parts[-1] in {"student", "snn"}
    cleaned = [p for p in sys.path if not _is_area_dir(p)]
    monkeypatch.setattr(sys, "path", cleaned)
    root, manifest = lma_corpus["root"], lma_corpus["manifest"]
    train_ds, val_ds = build_ingredient(
        "data", "lma_typed_l3",
        {"lma_root": root, "split_manifest": manifest,
         "windows_per_epoch": 8, "val_windows": 6})
    assert len(train_ds) == 8 and len(val_ds) == 6
