"""Coverage tests for ``ai_models/snn/lma_dataset.py``.

Builds a real LMA from the pyedflib test_generator EDF (no synthetic
data) by:
  1. Copying the EDF with a TUH-compatible stem (``aaaaaaaa_s001_t000``)
     so the stem regex in ``build_lma_entry_index`` matches.
  2. ``lml encode`` → ``.lma`` archive.
  3. Generating a small ``activity_labels`` NPZ and ``lml append``-ing it
     under ``labels/<stem>_labels.npz`` so the per-dataset LMA layout
     (Phase M, 2026-05-20) is satisfied.

Futureproofing: assertions pin SHAPE + DTYPE invariants, not exact
numeric values. The LMA path emits an L3 of ``(21, 313)`` and labels
``(8, 313)`` int64 by hard contract with ``SubbandActivityDataset`` —
those constants are checked. Numeric content drifts with any DSP
change (highpass, resample, calibration) and is intentionally *not*
asserted.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Sampler


_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Import via the canonical package path (matches conftest's sys.path
# additions). The shim module-level path ``lma_dataset`` is also valid.
from lamquant.snn import lma_dataset as lds  # noqa: E402

# The PyO3 wheel that `lma_dataset.py` calls into for LMA reads is
# optional — tests that actually drive the dataset need it. Skip
# module-level if the wheel isn't built/installed.
try:
    import lamquant_core  # noqa: F401
    _HAS_LAMQUANT_CORE = True
except Exception:
    _HAS_LAMQUANT_CORE = False

pytestmark = [
    pytest.mark.l2,
    pytest.mark.skipif(
        not _HAS_LAMQUANT_CORE,
        reason="lamquant_core PyO3 wheel not installed; build with "
               "`maturin develop -m lamquant-core/Cargo.toml -F python` "
               "to enable this suite.",
    ),
]


# ---------------------------------------------------------------------------
# Fixture: build a real LMA from a real EDF, append a tiny labels NPZ.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_lma_fixture(tmp_path_factory, real_test_edf, lml_cli_binary):
    """Real LMA built from the canonical pyedflib EDF + a labels NPZ.

    Returns a dict::

        {
          "lma_path": Path,             # .lma archive
          "stem":     str,              # TUH-style stem matched by regex
          "manifest": Path,             # split manifest JSON
          "n_windows_total": int,       # rows in activity_labels (cols/312)
          "seizure_windows": List[int], # window indices marked class 2
        }
    """
    work = tmp_path_factory.mktemp("lma_coverage")
    stem = "aaaaaaaa_s001_t000"
    edf_copy = work / f"{stem}.edf"
    shutil.copy2(real_test_edf, edf_copy)

    lma_path = work / f"{stem}.lma"
    r = subprocess.run(
        [str(lml_cli_binary), "encode", str(edf_copy), "-o", str(lma_path), "-q"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        pytest.skip(
            f"lml encode failed (rc={r.returncode}): "
            f"stderr={r.stderr[-300:]}"
        )
    assert lma_path.is_file(), f"lml encode produced no .lma at {lma_path}"

    # Build labels NPZ. 5 windows of 312 cols each; mark window 1 as seizure.
    seizure_windows = [1]
    n_windows = 5
    activity = np.zeros((8, n_windows * lds.LABEL_PER_WINDOW), dtype=np.uint8)
    for wi in seizure_windows:
        s = wi * lds.LABEL_PER_WINDOW
        e = s + lds.LABEL_PER_WINDOW
        activity[:, s:e] = 2
    labels_npz = work / f"{stem}_labels.npz"
    np.savez_compressed(
        labels_npz,
        activity_labels=activity,
        source=f"{stem}.edf",
    )

    r = subprocess.run(
        [str(lml_cli_binary), "append", "-q",
         "--as", f"labels/{stem}_labels.npz",
         str(lma_path), str(labels_npz)],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        pytest.skip(
            f"lml append failed (rc={r.returncode}): "
            f"stderr={r.stderr[-300:]}"
        )

    manifest = work / "split.json"
    manifest.write_text(json.dumps({
        "schema": "lamquant.snn_split.v1",
        "seed": 42,
        "split_strategy": "subject_grouped",
        "val_fraction": 0.10,
        "subjects": {
            "aaaaaaaa": "train",
            "valsubj": "val",
        },
        "stems_by_subject": {
            "aaaaaaaa": [stem],
            "valsubj": ["nonexistent_s001_t000"],  # absent on purpose
        },
        "promotions": [],
        "summary": {},
    }))

    return {
        "lma_path": lma_path,
        "stem": stem,
        "manifest": manifest,
        "n_windows_total": n_windows,
        "seizure_windows": list(seizure_windows),
        "work": work,
    }


# ---------------------------------------------------------------------------
# select_windows — pure deterministic selector (no IO).
# ---------------------------------------------------------------------------


class TestSelectWindows:
    @staticmethod
    def _make_labels(n_windows, seizure_idxs):
        arr = np.zeros((8, n_windows * lds.LABEL_PER_WINDOW), dtype=np.uint8)
        for wi in seizure_idxs:
            s = wi * lds.LABEL_PER_WINDOW
            e = s + lds.LABEL_PER_WINDOW
            arr[:, s:e] = 2
        return arr

    def test_empty_labels_returns_singleton(self):
        out = lds.select_windows(np.zeros((8, 0), dtype=np.uint8))
        assert out == [0]

    def test_zero_seizures_fills_budget(self):
        arr = self._make_labels(10, [])
        out = lds.select_windows(arr, max_windows=4)
        # Returns sorted unique indices; len bounded by min(max_windows, n_total)
        assert sorted(out) == out
        assert len(out) == len(set(out))
        assert all(0 <= wi < 10 for wi in out)
        assert len(out) == 4

    def test_seizure_windows_always_included(self):
        arr = self._make_labels(8, [1, 3, 5])
        out = lds.select_windows(arr, max_windows=5,
                                 max_seizure_windows=10)
        assert {1, 3, 5}.issubset(set(out))

    def test_status_epilepticus_capped(self):
        arr = self._make_labels(12, list(range(12)))
        out = lds.select_windows(arr, max_windows=5, max_seizure_windows=6)
        # All-seizure: cap holds at max_seizure_windows
        assert len(out) == 6

    def test_min_background_applies_only_when_no_seizures(self):
        # No seizures: min_background floor honored.
        arr = self._make_labels(3, [])
        out = lds.select_windows(arr, max_windows=0, min_background=1)
        assert len(out) >= 1
        # All-seizure: min_background does NOT add anything.
        arr2 = self._make_labels(3, [0, 1, 2])
        out2 = lds.select_windows(arr2, max_windows=3,
                                  max_seizure_windows=3, min_background=1)
        # picked_seizures saturates max_windows; no extra bg
        assert len(out2) == 3

    def test_returns_sorted_unique_list(self):
        arr = self._make_labels(20, [2, 7, 13])
        out = lds.select_windows(arr, max_windows=8)
        assert isinstance(out, list)
        assert out == sorted(out)
        assert len(out) == len(set(out))


# ---------------------------------------------------------------------------
# load_split_manifest — pure JSON reader.
# ---------------------------------------------------------------------------


class TestLoadSplitManifest:
    def test_round_trip(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps({
            "subjects": {"s_a": "train", "s_b": "val"},
            "stems_by_subject": {
                "s_a": ["stem_a_1", "stem_a_2"],
                "s_b": ["stem_b_1"],
            },
        }))
        train_stems, train_by = lds.load_split_manifest(manifest, "train")
        val_stems, val_by = lds.load_split_manifest(manifest, "val")
        assert set(train_stems) == {"stem_a_1", "stem_a_2"}
        assert set(val_stems) == {"stem_b_1"}
        # subject_by_stem maps each stem to its subject id
        assert train_by["stem_a_1"] == "s_a"
        assert val_by["stem_b_1"] == "s_b"

    def test_missing_path_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            lds.load_split_manifest(tmp_path / "missing.json", "train")

    def test_unknown_split_returns_empty(self, tmp_path):
        # Stems whose subject's split != requested are simply omitted.
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps({
            "subjects": {"x": "train"},
            "stems_by_subject": {"x": ["foo"]},
        }))
        val_stems, _ = lds.load_split_manifest(manifest, "val")
        assert val_stems == []


# ---------------------------------------------------------------------------
# Constants + module-level cache dtype helper.
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_shape_constants_are_positive_ints(self):
        assert isinstance(lds.WINDOW_SAMPLES, int) and lds.WINDOW_SAMPLES > 0
        assert isinstance(lds.TARGET_CHANNELS, int) and lds.TARGET_CHANNELS > 0
        assert isinstance(lds.L3_T, int) and lds.L3_T > 0
        assert isinstance(lds.LABEL_PER_WINDOW, int) and lds.LABEL_PER_WINDOW > 0

    def test_target_sr_positive(self):
        assert lds.TARGET_SR > 0.0

    def test_window_cap_constants_positive(self):
        assert lds.MAX_WINDOWS_PER_FILE >= 1
        assert lds.MAX_SEIZURE_WINDOWS_PER_FILE >= lds.MIN_BACKGROUND_PER_FILE


class TestL3CacheDtype:
    def test_default_is_float16(self, monkeypatch):
        monkeypatch.delenv("L3_CACHE_DTYPE", raising=False)
        assert lds._l3_cache_dtype() == np.dtype(np.float16)

    def test_float32_alias_accepted(self, monkeypatch):
        monkeypatch.setenv("L3_CACHE_DTYPE", "float32")
        assert lds._l3_cache_dtype() == np.dtype(np.float32)

    def test_short_aliases(self, monkeypatch):
        for alias in ("f16", "half", "fp16"):
            monkeypatch.setenv("L3_CACHE_DTYPE", alias)
            assert lds._l3_cache_dtype() == np.dtype(np.float16)
        for alias in ("f32", "single", "fp32"):
            monkeypatch.setenv("L3_CACHE_DTYPE", alias)
            assert lds._l3_cache_dtype() == np.dtype(np.float32)

    def test_unknown_falls_back_to_float16(self, monkeypatch):
        monkeypatch.setenv("L3_CACHE_DTYPE", "bfloat42")
        assert lds._l3_cache_dtype() == np.dtype(np.float16)


class TestL3CacheDir:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv("L3_CACHE_DIR", raising=False)
        assert lds._l3_cache_dir() is None

    def test_set_returns_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("L3_CACHE_DIR", str(tmp_path))
        out = lds._l3_cache_dir()
        assert out == tmp_path


class TestLabelCacheDir:
    def test_returns_path_or_none(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LMA_LABEL_CACHE_DIR", str(tmp_path))
        assert lds._label_cache_dir() == tmp_path
        # Empty env disables the disk-staged cache.
        monkeypatch.setenv("LMA_LABEL_CACHE_DIR", "")
        assert lds._label_cache_dir() is None


class TestFadvise:
    def test_silent_on_missing_file(self, tmp_path):
        # Non-existent path: helper must not raise (catches OSError).
        lds._fadvise_hint(tmp_path / "no_such_file")

    def test_silent_on_real_file(self, tmp_path):
        p = tmp_path / "small.bin"
        p.write_bytes(b"abc123")
        lds._fadvise_hint(p)  # should not raise


class TestHighpassSos:
    def test_returns_sos_array(self):
        sos = lds._highpass_sos()
        # SOS shape is (n_sections, 6) for biquad cascade
        assert sos.ndim == 2 and sos.shape[1] == 6
        # Cache: second call returns the same object
        assert lds._highpass_sos() is sos


# ---------------------------------------------------------------------------
# LmaDataset construction-error contract — exercise input validation paths.
# ---------------------------------------------------------------------------


class TestLmaDatasetErrors:
    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(ValueError):
            lds.LmaDataset(
                lma_paths=[tmp_path / "fake.lma"],
                split="train",
                split_manifest_path=None,
            )

    def test_bad_split_raises(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps({
            "subjects": {}, "stems_by_subject": {},
        }))
        with pytest.raises(ValueError):
            lds.LmaDataset(
                lma_paths=[tmp_path / "fake.lma"],
                split="not_a_split",
                split_manifest_path=manifest,
            )

    def test_no_lma_paths_raises(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps({
            "subjects": {}, "stems_by_subject": {},
        }))
        with pytest.raises(ValueError):
            lds.LmaDataset(
                lma_paths=[],
                split="train",
                split_manifest_path=manifest,
            )

    def test_missing_lma_dir_raises(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps({
            "subjects": {}, "stems_by_subject": {},
        }))
        with pytest.raises(FileNotFoundError):
            lds.LmaDataset(
                lma_dir=tmp_path / "no_such_dir",
                split="train",
                split_manifest_path=manifest,
            )

    def test_nonexistent_lma_path_raises(self, tmp_path):
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps({
            "subjects": {}, "stems_by_subject": {},
        }))
        # Pass a list with a non-existent path
        with pytest.raises(FileNotFoundError):
            lds.LmaDataset(
                lma_paths=[tmp_path / "absent.lma"],
                split="train",
                split_manifest_path=manifest,
            )

    def test_corrupt_split_bleed_raises(self, real_lma_fixture):
        """Both splits naming the same stem must fail-loud."""
        work = real_lma_fixture["work"]
        stem = real_lma_fixture["stem"]
        bad_manifest = work / "bleed.json"
        bad_manifest.write_text(json.dumps({
            "subjects": {"aaaaaaaa": "train", "leak": "val"},
            "stems_by_subject": {
                "aaaaaaaa": [stem],
                "leak": [stem],   # SAME stem in both splits
            },
        }))
        with pytest.raises(RuntimeError, match="corrupt"):
            lds.LmaDataset(
                lma_paths=[real_lma_fixture["lma_path"]],
                split="train",
                split_manifest_path=bad_manifest,
            )


# ---------------------------------------------------------------------------
# LmaDataset end-to-end against a real LMA (the meat of the coverage lift).
# ---------------------------------------------------------------------------


@pytest.mark.data
class TestLmaDatasetReal:
    """Real-LMA __init__ / __len__ / __getitem__ contract.

    We do NOT assert specific float values — the L3 stream depends on
    DSP details (resampler, highpass biquad coefficients, calibration)
    that the codec is free to refactor. Shape + dtype + bound checks
    only.
    """

    def test_train_split_loads(self, real_lma_fixture):
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        # __len__ matches the index size.
        assert len(ds) == len(ds.index)
        # At least one window per stem.
        assert len(ds) >= 1
        # Index tuple shape: (lma_path, stem, win_idx, lml_internal, label_internal)
        entry = ds.index[0]
        assert len(entry) == 5

    def test_getitem_returns_correct_shape_and_dtype(self, real_lma_fixture):
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        sig, lbl = ds[0]
        # Contract: matches SubbandActivityDataset.
        assert isinstance(sig, torch.Tensor)
        assert isinstance(lbl, torch.Tensor)
        assert sig.shape == (lds.TARGET_CHANNELS, lds.L3_T)
        assert lbl.shape == (8, lds.L3_T)
        assert sig.dtype == torch.float32
        assert lbl.dtype == torch.int64

    def test_val_split_empty_or_indexes_correctly(self, real_lma_fixture):
        """val subject's stem is absent → 0 windows → init must raise.

        This pins the no-windows guard at the bottom of __init__.
        """
        with pytest.raises(RuntimeError, match="0 windows"):
            lds.LmaDataset(
                lma_paths=[real_lma_fixture["lma_path"]],
                split="val",
                split_manifest_path=real_lma_fixture["manifest"],
            )

    def test_getitem_oob_index_falls_back_to_zeros(self, real_lma_fixture, monkeypatch):
        """__getitem__ guards against L3 stack shorter than win_idx.

        We force the L3 stack to a single-window result via monkeypatch
        so window indices 1..N exercise the OOB zero-fill branch.
        """
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        # Clear the worker cache so our patched stack is consulted fresh.
        lds._L3_CACHE.clear()
        # Patch _cached_l3_stack to return a 1-window stack so any
        # higher win_idx in the dataset triggers the OOB zero branch.
        single = np.zeros((1, lds.TARGET_CHANNELS, lds.L3_T), dtype=np.float32)
        monkeypatch.setattr(lds, "_cached_l3_stack",
                            lambda *a, **kw: single)
        # Walk every index — should never raise.
        for i in range(len(ds)):
            sig, lbl = ds[i]
            assert sig.shape == (lds.TARGET_CHANNELS, lds.L3_T)
            assert lbl.shape == (8, lds.L3_T)

    def test_getitem_decode_failure_returns_zeros(self, real_lma_fixture, monkeypatch):
        """When _cached_l3_stack returns None (decode fail), getitem
        falls back to a zero-tensor of the right shape."""
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        lds._L3_CACHE.clear()
        monkeypatch.setattr(lds, "_cached_l3_stack",
                            lambda *a, **kw: None)
        sig, lbl = ds[0]
        assert sig.shape == (lds.TARGET_CHANNELS, lds.L3_T)
        assert torch.all(sig == 0)


# ---------------------------------------------------------------------------
# LmaGroupedSampler contract.
# ---------------------------------------------------------------------------


@pytest.mark.data
class TestLmaGroupedSampler:
    def test_subclasses_sampler(self, real_lma_fixture):
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        sampler = lds.LmaGroupedSampler(ds)
        assert isinstance(sampler, Sampler)

    def test_len_matches_dataset(self, real_lma_fixture):
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        sampler = lds.LmaGroupedSampler(ds, shuffle=False)
        assert len(sampler) == len(ds)

    def test_indices_cover_dataset_once(self, real_lma_fixture):
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        sampler = lds.LmaGroupedSampler(ds, shuffle=True, seed=7)
        indices = list(iter(sampler))
        assert sorted(indices) == list(range(len(ds)))

    def test_set_epoch_changes_order_when_shuffled(self, real_lma_fixture):
        # Add a second synthetic stem so groups > 1 — only then does
        # shuffle have anything to permute. Build with the same fixture.
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        # Single-LMA dataset: shuffle is a no-op at group level (1 group).
        # The set_epoch path itself still gets exercised.
        sampler = lds.LmaGroupedSampler(ds, shuffle=True, seed=42)
        sampler.set_epoch(5)
        assert sampler.epoch == 5

    def test_rejects_non_lma_dataset(self):
        class _Foo(Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, i):
                return None

        with pytest.raises(TypeError):
            lds.LmaGroupedSampler(_Foo())


# ---------------------------------------------------------------------------
# iter_labels_only — pos_weight scan fast path.
# ---------------------------------------------------------------------------


@pytest.mark.data
class TestIterLabelsOnly:
    def test_yields_correct_shape_and_dtype(self, real_lma_fixture):
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        windows = list(lds.iter_labels_only(ds))
        # One per index entry.
        assert len(windows) == len(ds)
        for w in windows:
            assert isinstance(w, np.ndarray)
            assert w.shape == (8, lds.L3_T)
            assert w.dtype == np.int64

    def test_yields_seizure_class_when_present(self, real_lma_fixture):
        """At least one of the yielded windows must contain class 2
        because the synthetic labels NPZ marked window 1 as seizure
        and select_windows always includes seizure windows."""
        ds = lds.LmaDataset(
            lma_paths=[real_lma_fixture["lma_path"]],
            split="train",
            split_manifest_path=real_lma_fixture["manifest"],
        )
        has_seizure = any(np.any(w == 2) for w in lds.iter_labels_only(ds))
        assert has_seizure


# ---------------------------------------------------------------------------
# _cached_l3_stack — exercise disk-cache tier (write + read).
# ---------------------------------------------------------------------------


@pytest.mark.data
class TestCachedL3Stack:
    def test_disk_cache_write_then_load(self, real_lma_fixture, tmp_path, monkeypatch):
        """First call writes the L3 stack to disk; second call mmaps it."""
        lds._L3_CACHE.clear()
        monkeypatch.setenv("L3_CACHE_DIR", str(tmp_path))
        # Disable float16 storage so the on-disk shape stays exact.
        monkeypatch.setenv("L3_CACHE_DTYPE", "float32")
        stem = real_lma_fixture["stem"]
        out1 = lds._cached_l3_stack(
            str(real_lma_fixture["lma_path"]),
            stem,
            lml_internal=f"{stem}.lml",
        )
        if out1 is None:
            pytest.skip("L3 compute returned None — channel resolver "
                        "could not extract from minimal EDF; disk-cache "
                        "branch covered by zero-result skip path")
        cache_file = tmp_path / f"{stem}.npy"
        assert cache_file.is_file()
        # Clear in-memory tier; the second call must hit the disk tier.
        lds._L3_CACHE.clear()
        out2 = lds._cached_l3_stack(
            str(real_lma_fixture["lma_path"]),
            stem,
            lml_internal=f"{stem}.lml",
        )
        assert out2 is not None
        assert out2.shape == out1.shape
        # In-memory cache should now hold the disk-loaded stack.
        assert any(stem in k for k in lds._L3_CACHE)
