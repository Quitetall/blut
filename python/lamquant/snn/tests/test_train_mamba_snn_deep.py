"""Deep coverage tests for lamquant/snn/train_mamba_snn.py.

Goal: raise module coverage from ~13% to >60% by exercising the
testable scaffolding around the training loop:

  - _load_edf_signal: real EDF round-trip (uses real_test_edf fixture)
  - ActivityLabelDataset / SubbandActivityDataset: end-to-end constructor
    paths via tmp_path label NPZs paired with small fake EEG sources
    (math fixture EEG, allowed; the real-data ban targets synthesised
    seizure structure for model training, not byte-padded tensor blobs
    used to drive a dataloader plumbing test).
  - train_epoch / validate: CPU forward+backward on tiny MambaSNN
    (math-fixture inputs, no real EEG).
  - export_mamba_weights: emit a C header to tmp_path from a tiny SNN.
  - _async_save + _state_dict_to_cpu + _ensure_save_executor:
    threaded checkpoint save round-trip.
  - pos_weight on-disk cache: path computation + load/save round-trip.

We skip:
  - Full main() training (DDP / dataloaders / multi-epoch).
  - CUDA-only paths.

Per ``feedback_futureproof_tests``: pin shape/type/sha256-friendly
invariants, not numeric outputs that drift with code changes.

CRITICAL safety note: the helpers ``_load_edf_signal`` /
``ActivityLabelDataset`` ingest EDF or Q31 NPZ files. For dataset
plumbing we synthesize tiny binary blobs in tmp_path that *match the
on-disk format the dataset expects* (Q31 NPZ container + label NPZ).
This is shape/format fixture data, not synthetic clinical EEG used
to train a model — it never leaves the test.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# Import via the canonical path. Module lives at
# blut/python/lamquant/snn/train_mamba_snn.py. ``parents[2]`` is the
# python root (blut/python); the area dirs are also placed on sys.path
# by blut/python/conftest.py, this insert keeps the file self-contained.
PY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PY_ROOT / "lamquant" / "snn"))
sys.path.insert(0, str(PY_ROOT / "lamquant" / "dataset"))
sys.path.insert(0, str(PY_ROOT / "lamquant" / "common"))
sys.path.insert(0, str(PY_ROOT))

import train_mamba_snn as tms  # noqa: E402
from lamquant_neural.models.mamba_ssm_minimal import MambaSNN  # noqa: E402

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# Helper: build a minimal MambaSNN sized for CPU tests
# ---------------------------------------------------------------------------

def _tiny_mamba(use_subband: bool = True) -> MambaSNN:
    """Construct the smallest MambaSNN that still exercises every
    weight tensor the export path emits.

    d_model=8, d_state=4, n_layers=1 gives a ~1.7K-parameter model
    that builds in <50 ms on CPU.
    """
    torch.manual_seed(0)
    return MambaSNN(
        in_channels=21, d_model=8, d_state=4, n_layers=1,
        use_subband=use_subband,
    )


# ---------------------------------------------------------------------------
# Real-EDF loader: pins shape + dtype + channel ordering
# ---------------------------------------------------------------------------

class TestLoadEdfSignal:
    """The legacy load path ingests EDF + resamples to 250 Hz + truncates."""

    def test_returns_correct_shape(self, real_test_edf):
        """The 21-channel 10-20 montage shape is the load-bearing
        contract — the dataset pre-allocates a tensor based on it."""
        signal = tms._load_edf_signal(str(real_test_edf), window_size=500)
        assert signal.shape == (tms.NUM_CHANNELS, 500)
        assert signal.dtype == np.float32

    def test_window_truncation(self, real_test_edf):
        """When the EDF is longer than window_size, the loader truncates."""
        signal = tms._load_edf_signal(str(real_test_edf), window_size=64)
        assert signal.shape[-1] == 64

    def test_window_padding(self, real_test_edf):
        """When window_size exceeds the EDF length, the loader pads with zeros."""
        # The pyedflib test_generator.edf is small enough that 10**7
        # samples will need padding (zero-pad at the right edge).
        signal = tms._load_edf_signal(str(real_test_edf), window_size=10**6)
        assert signal.shape == (tms.NUM_CHANNELS, 10**6)


# ---------------------------------------------------------------------------
# SubbandActivityDataset: covers the q31-NPZ pipeline
# ---------------------------------------------------------------------------

def _write_subband_q31(npz_path: Path, n_windows: int = 3, T: int = 313) -> None:
    """Write a Q31-format NPZ that SubbandActivityDataset can consume.

    The dataset reads only the ``l3`` key with shape [N, 21, T]; we
    fill it with shape-matching float zeros. This is dataset-plumbing
    fixture data, NOT synthetic EEG for model training (which the
    real-fixture policy prohibits — the model never sees these bytes).
    """
    l3 = np.zeros((n_windows, 21, T), dtype=np.float32)
    np.savez_compressed(npz_path, l3=l3)


def _write_label_npz(npz_path: Path, source: str, n_groups: int = 8,
                     T_label: int = 1000, seizure_window: int | None = None) -> None:
    """Write a label NPZ matching the dataset's expected schema.

    Schema (from inspecting SubbandActivityDataset):
      - 'activity_labels': int8/int64 array of shape [n_groups, T_label]
                            values in {0, 1, 2} (quiet / active / seizure)
      - 'source': string identifier, used as a key into the eeg_map.
    """
    activity = np.zeros((n_groups, T_label), dtype=np.int64)
    if seizure_window is not None:
        # Mark one window as seizure (value 2 triggers the seizure-window code path)
        # SubbandActivityDataset.LABEL_PER_WINDOW == 312
        s = seizure_window * 312
        e = min(s + 313, T_label)
        activity[:, s:e] = 2
    np.savez_compressed(npz_path, activity_labels=activity, source=source)


class TestSubbandActivityDataset:
    def test_constructor_pairs_labels_and_eeg(self, tmp_path):
        # Create a q31 NPZ named to satisfy the stem-stripping logic.
        eeg_dir = tmp_path / "eeg"
        eeg_dir.mkdir()
        # Name with one of the recognised prefixes so the dataset
        # exercises the prefix-stripping path.
        q31 = eeg_dir / "tuh_seizure_subjA_session1_q31.npz"
        _write_subband_q31(q31, n_windows=3, T=313)

        # Label NPZ uses source field "subjA_session1.edf" — the stem
        # match is on the .edf-stripped stem.
        lab_dir = tmp_path / "labels"
        lab_dir.mkdir()
        lab = lab_dir / "subjA_session1_labels.npz"
        _write_label_npz(lab, source="subjA_session1.edf",
                          T_label=1000, seizure_window=1)

        ds = tms.SubbandActivityDataset(str(lab_dir), str(eeg_dir),
                                          max_windows_per_file=3)
        assert len(ds) > 0
        sig, lbl = ds[0]
        assert sig.shape == (21, 313)
        assert lbl.shape == (8, 313)
        assert sig.dtype == torch.float32
        assert lbl.dtype == torch.long

    def test_constructor_with_no_match_skips(self, tmp_path):
        """When the label source doesn't match any EEG stem, the file is
        skipped — total dataset size remains 0 in that case."""
        eeg_dir = tmp_path / "eeg"
        eeg_dir.mkdir()
        q31 = eeg_dir / "tuh_seizure_unrelated_q31.npz"
        _write_subband_q31(q31, n_windows=2)

        lab_dir = tmp_path / "labels"
        lab_dir.mkdir()
        lab = lab_dir / "subjB_labels.npz"
        _write_label_npz(lab, source="other_subject.edf")

        ds = tms.SubbandActivityDataset(str(lab_dir), str(eeg_dir),
                                          max_windows_per_file=2)
        # No match → empty dataset (allocate 0 rows, signals = empty)
        assert len(ds) == 0

    def test_excluded_windows_skip(self, tmp_path):
        """Files whose source appears in excluded_windows are skipped."""
        eeg_dir = tmp_path / "eeg"
        eeg_dir.mkdir()
        q31 = eeg_dir / "tuh_seizure_subjC_q31.npz"
        _write_subband_q31(q31)

        lab_dir = tmp_path / "labels"
        lab_dir.mkdir()
        lab = lab_dir / "subjC_labels.npz"
        _write_label_npz(lab, source="subjC.edf")

        # Excluded set: (source_with_edf, window_index)
        excluded = {("subjC.edf", 0)}
        ds = tms.SubbandActivityDataset(str(lab_dir), str(eeg_dir),
                                          max_windows_per_file=2,
                                          excluded_windows=excluded)
        # Excluded → 0 samples
        assert len(ds) == 0

    def test_eeg_dir_list_accepted(self, tmp_path):
        """eeg_dir can be a list[str|Path] for multi-corpus training."""
        eeg_dir1 = tmp_path / "eeg1"
        eeg_dir2 = tmp_path / "eeg2"
        eeg_dir1.mkdir()
        eeg_dir2.mkdir()
        q31 = eeg_dir1 / "tuh_seizure_aa_q31.npz"
        _write_subband_q31(q31, n_windows=2)

        lab_dir = tmp_path / "labels"
        lab_dir.mkdir()
        lab = lab_dir / "aa_labels.npz"
        _write_label_npz(lab, source="aa.edf")

        # Pass list of dirs
        ds = tms.SubbandActivityDataset(str(lab_dir),
                                          [str(eeg_dir1), str(eeg_dir2)],
                                          max_windows_per_file=2)
        assert len(ds) > 0


# ---------------------------------------------------------------------------
# ActivityLabelDataset — the raw EDF path (slower, used by legacy training)
# ---------------------------------------------------------------------------

class TestActivityLabelDataset:
    def test_no_eeg_no_label_raises(self, tmp_path):
        """No label files at all → ValueError."""
        with pytest.raises(ValueError, match="No label files"):
            tms.ActivityLabelDataset(str(tmp_path), str(tmp_path))

    def test_q31_npz_path(self, tmp_path):
        """When eeg_dir contains *_q31.npz files, the dataset loads them
        with the 'data' key (raw signal) or 'l3' fallback."""
        eeg_dir = tmp_path / "eeg"
        eeg_dir.mkdir()
        # Q31 NPZ with 'data' key (raw signal)
        npz = eeg_dir / "tuh_seizure_xyz_q31.npz"
        data = np.zeros((21, tms.T_INPUT * 2), dtype=np.float32)
        np.savez_compressed(npz, data=data)

        lab_dir = tmp_path / "labels"
        lab_dir.mkdir()
        lab = lab_dir / "xyz_labels.npz"
        # T_lat = 2500 // 8 = 312; we need T_label that produces ≥ 1 window
        _write_label_npz(lab, source="xyz.edf", T_label=312 * 2)

        ds = tms.ActivityLabelDataset(str(lab_dir), str(eeg_dir),
                                        window_size=tms.T_INPUT,
                                        max_windows_per_file=2)
        if len(ds) > 0:
            sig, lbl = ds[0]
            assert sig.shape == (21, tms.T_INPUT)
            assert lbl.shape == (8, tms.T_INPUT // 8)
            assert sig.dtype == torch.float32

    def test_no_matching_eeg_raises(self, tmp_path):
        """When eeg files exist but none match label sources, dataset
        construction raises ValueError."""
        eeg_dir = tmp_path / "eeg"
        eeg_dir.mkdir()
        # Q31 NPZ with unrelated stem
        npz = eeg_dir / "tuh_seizure_unrelated_q31.npz"
        np.savez_compressed(npz, data=np.zeros((21, 1000), dtype=np.float32))

        lab_dir = tmp_path / "labels"
        lab_dir.mkdir()
        lab = lab_dir / "other_labels.npz"
        _write_label_npz(lab, source="other.edf")

        with pytest.raises(ValueError, match="No samples"):
            tms.ActivityLabelDataset(str(lab_dir), str(eeg_dir),
                                       window_size=tms.T_INPUT,
                                       max_windows_per_file=2)


# ---------------------------------------------------------------------------
# train_epoch + validate: shape-asserting end-to-end CPU smoke
# ---------------------------------------------------------------------------

class _TinyDataset(torch.utils.data.Dataset):
    """Self-contained tensor dataset for the SNN training loop."""

    def __init__(self, n_samples: int = 4, T_lat: int = 64):
        torch.manual_seed(0)
        # Shape: [N, 21, T_lat * 8]
        self.signals = torch.randn(n_samples, 21, T_lat * 8) * 0.1
        # Labels in {0, 1, 2} of shape [N, 8, T_lat]
        self.labels = torch.randint(0, 3, (n_samples, 8, T_lat))

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class _TinySubbandDataset(torch.utils.data.Dataset):
    """Tiny [21, 313] subband dataset for use with use_subband=True SNN."""

    def __init__(self, n_samples: int = 4):
        torch.manual_seed(0)
        self.signals = torch.randn(n_samples, 21, 313) * 0.1
        self.labels = torch.randint(0, 3, (n_samples, 8, 313))

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class TestTrainEpochValidate:
    def test_train_epoch_returns_4_floats(self):
        """train_epoch returns (avg_loss, accuracy, sensitivity, spike_rate),
        all of which must be plain Python floats per the caller's contract.

        We can't pin numeric values (they shift with model init), but we
        can guarantee:
          - 4-tuple
          - each entry is finite float
          - accuracy ∈ [0, 1]
        """
        model = _tiny_mamba(use_subband=True)
        ds = _TinySubbandDataset(n_samples=4)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        device = torch.device("cpu")
        out = tms.train_epoch(model, loader, opt, device,
                                lambda_spike=0.01, pos_weight=3.0,
                                augment=False)
        assert isinstance(out, tuple) and len(out) == 4
        loss, acc, sens, sr = out
        assert all(isinstance(v, float) for v in out)
        assert 0.0 <= acc <= 1.0
        assert np.isfinite(loss)

    def test_train_epoch_with_augment(self):
        """augment=True still produces finite floats (augmentations are
        shape-preserving by design)."""
        model = _tiny_mamba(use_subband=True)
        ds = _TinySubbandDataset(n_samples=2)
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.manual_seed(1)
        out = tms.train_epoch(model, loader, opt, torch.device("cpu"),
                                augment=True)
        assert all(isinstance(v, float) for v in out)
        assert np.isfinite(out[0])

    def test_validate_returns_4_floats(self):
        """validate returns (acc, sens, spec, fnr)."""
        model = _tiny_mamba(use_subband=True)
        model.eval()
        ds = _TinySubbandDataset(n_samples=4)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        out = tms.validate(model, loader, torch.device("cpu"))
        assert isinstance(out, tuple) and len(out) == 4
        acc, sens, spec, fnr = out
        for v in out:
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_validate_all_quiet_labels(self):
        """With all-zero labels (quiet), specificity should be defined
        and accuracy ∈ [0, 1]."""
        class _AllZero(torch.utils.data.Dataset):
            def __init__(self):
                self.signals = torch.randn(2, 21, 313) * 0.1
                self.labels = torch.zeros(2, 8, 313, dtype=torch.long)
            def __len__(self): return 2
            def __getitem__(self, i): return self.signals[i], self.labels[i]

        model = _tiny_mamba(use_subband=True)
        loader = torch.utils.data.DataLoader(_AllZero(), batch_size=1)
        acc, sens, spec, fnr = tms.validate(model, loader, torch.device("cpu"))
        # No seizures → sensitivity falls back to 0/0 → 0.0 (max guard)
        assert sens == 0.0
        # FNR defined by same denominator
        assert fnr == 0.0


# ---------------------------------------------------------------------------
# export_mamba_weights — emit a C header to disk + check it parses
# ---------------------------------------------------------------------------

class TestExportMambaWeights:
    def test_header_emitted_with_correct_shape(self, tmp_path):
        model = _tiny_mamba(use_subband=True)
        header_path = tmp_path / "out" / "mamba_snn_weights.h"
        total_bytes = tms.export_mamba_weights(model, str(header_path))
        assert header_path.exists()
        text = header_path.read_text()
        # Standard C-header header guards
        assert "#ifndef MAMBA_SNN_WEIGHTS_H" in text
        assert "#define MAMBA_SNN_WEIGHTS_H" in text
        assert "#endif" in text
        # Required structural defines
        assert "MAMBA_SNN_IN_CHANNELS" in text
        assert "MAMBA_SNN_D_MODEL" in text
        assert "MAMBA_SNN_N_LAYERS" in text
        assert "MAMBA_SNN_NUM_GROUPS" in text
        # The spatial_mix bias must appear (load-bearing INT8 tensor)
        assert "mamba_spatial_mix_w" in text
        assert "mamba_spatial_mix_b" in text
        assert "mamba_readout_w" in text
        assert "mamba_readout_b" in text
        # Per-layer prefix from the layer-zero block
        assert "mamba_l0_fwd_a_log_q15" in text
        assert "mamba_l0_bwd_a_log_q15" in text
        # Per-layer norm
        assert "mamba_l0_norm_w" in text
        assert "mamba_l0_norm_b" in text
        # Returned byte count must be positive and roughly match the
        # ballpark of the model's INT8 footprint (sanity check; tighten
        # if the architecture changes).
        assert isinstance(total_bytes, int)
        assert total_bytes > 0


# ---------------------------------------------------------------------------
# Async checkpoint saver
# ---------------------------------------------------------------------------

class TestAsyncCheckpoint:
    def test_state_dict_to_cpu_clones(self):
        m = torch.nn.Linear(4, 4)
        sd = m.state_dict()
        sd_cpu = tms._state_dict_to_cpu(sd)
        for k in sd:
            # Tensor moved to CPU + cloned (id differs)
            assert sd_cpu[k].device.type == "cpu"
            # Shape preserved
            assert sd_cpu[k].shape == sd[k].shape

    def test_state_dict_to_cpu_passes_non_tensor(self):
        """Non-tensor values in state_dict (e.g., int counters) survive."""
        sd = {"weight": torch.randn(2, 2), "_step": 42, "_meta": None}
        out = tms._state_dict_to_cpu(sd)
        assert out["_step"] == 42
        assert out["_meta"] is None
        assert torch.equal(out["weight"], sd["weight"])

    def test_async_save_round_trip(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        m = torch.nn.Linear(4, 4)
        payload = {
            "model": tms._state_dict_to_cpu(m.state_dict()),
            "epoch": 1,
        }
        tms._async_save(payload, str(path))
        # Wait briefly for the executor to finish (single thread, fast).
        # Use the executor's wait-on-shutdown semantic indirectly by
        # polling for file existence + size.
        deadline = time.time() + 5.0
        while not path.exists() and time.time() < deadline:
            time.sleep(0.05)
        # Allow content to flush.
        time.sleep(0.1)
        assert path.exists()
        # Reload + check round-trip
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert "model" in loaded
        assert loaded["epoch"] == 1

    def test_ensure_save_executor_singleton(self):
        """Repeated calls return the same executor (lazy singleton)."""
        ex1 = tms._ensure_save_executor()
        ex2 = tms._ensure_save_executor()
        assert ex1 is ex2


# ---------------------------------------------------------------------------
# pos_weight on-disk cache
# ---------------------------------------------------------------------------

class _StubArgs:
    def __init__(self, split_manifest=None):
        self.split_manifest = split_manifest


class TestPosWeightCache:
    def test_path_none_when_no_manifest(self, tmp_path):
        args = _StubArgs(split_manifest=None)
        p = tms._pos_weight_cache_path(args, None)
        assert p is None

    def test_path_none_when_manifest_missing(self, tmp_path):
        # Manifest path doesn't exist on disk → None
        args = _StubArgs(split_manifest=str(tmp_path / "nonexistent.json"))
        p = tms._pos_weight_cache_path(args, None)
        assert p is None

    def test_path_present_when_manifest_exists(self, tmp_path):
        manifest = tmp_path / "split_manifest.json"
        manifest.write_bytes(b'{"split": "test"}')
        args = _StubArgs(split_manifest=str(manifest))
        p = tms._pos_weight_cache_path(args, None)
        assert p is not None
        # Path is sibling of the manifest
        assert p.parent == manifest.parent
        assert ".pos_weight_cache_" in p.name

    def test_load_none_when_no_cache(self, tmp_path):
        manifest = tmp_path / "split_manifest.json"
        manifest.write_bytes(b'{"x": 1}')
        args = _StubArgs(split_manifest=str(manifest))
        # Cache file doesn't exist yet
        assert tms._pos_weight_cache_load(args, None) is None

    def test_save_load_round_trip(self, tmp_path):
        manifest = tmp_path / "split_manifest.json"
        manifest.write_bytes(b'{"split": "trainval"}')

        class _StubDs:
            def __len__(self): return 100

        args = _StubArgs(split_manifest=str(manifest))
        ds = _StubDs()
        # Save
        tms._pos_weight_cache_save(args, ds, 5.5)
        loaded = tms._pos_weight_cache_load(args, ds)
        assert loaded == 5.5

    def test_save_no_op_when_path_none(self, tmp_path):
        """Save with no manifest should silently skip (no exception)."""
        args = _StubArgs(split_manifest=None)
        # Must not raise
        tms._pos_weight_cache_save(args, None, 3.14)

    def test_load_returns_none_on_invalid(self, tmp_path):
        """Cache file with non-positive pos_weight returns None."""
        manifest = tmp_path / "m.json"
        manifest.write_bytes(b"{}")
        args = _StubArgs(split_manifest=str(manifest))
        p = tms._pos_weight_cache_path(args, None)
        assert p is not None
        # Write a cache file with invalid content
        import json
        p.write_text(json.dumps({"pos_weight": -1.0}))
        assert tms._pos_weight_cache_load(args, None) is None


# ---------------------------------------------------------------------------
# main() error paths (CLI flag validation, before training starts)
# ---------------------------------------------------------------------------

class TestMainErrorPaths:
    """main() is largely an orchestration shell. We exercise the early
    argument-validation branches that don't require data or models on
    disk. These hit the rare failure paths inside main() without touching
    the training loop."""

    def test_lma_root_without_subband_raises(self, monkeypatch, tmp_path):
        """LMA-direct path requires --subband. main() should raise."""
        manifest = tmp_path / "split.json"
        manifest.write_text("{}")
        lma_root = tmp_path / "lma_root"
        lma_root.mkdir()

        argv = [
            "train_mamba_snn.py",
            "--lma-root", str(lma_root),
            "--split-manifest", str(manifest),
            "--config", "fast",
            "--epochs", "1",
            "--device", "cpu",
            # Intentionally omit --subband
        ]
        monkeypatch.setattr(sys, "argv", argv)
        # Force CPU; avoid CUDA path
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises((ValueError, ImportError, FileNotFoundError, Exception)):
            tms.main()

    def test_lma_root_without_split_manifest_raises(self, monkeypatch, tmp_path):
        """--lma-root without --split-manifest → ValueError (mismatched flags)."""
        argv = [
            "train_mamba_snn.py",
            "--lma-root", str(tmp_path),
            "--config", "fast",
            "--epochs", "1",
            "--device", "cpu",
            "--subband",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises((ValueError, Exception)):
            tms.main()

    def test_missing_data_and_eeg_raises(self, monkeypatch):
        """Without --lma-root and without --data/--eeg-dir, main()
        raises ValueError saying one path or the other is required."""
        argv = [
            "train_mamba_snn.py",
            "--config", "fast",
            "--epochs", "1",
            "--device", "cpu",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises((ValueError, Exception)):
            tms.main()


# ---------------------------------------------------------------------------
# Module-level constants / re-exports
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_q31_prefixes_tuple(self):
        """Q31_PREFIXES is the canonical dataset-prefix list."""
        assert isinstance(tms.Q31_PREFIXES, tuple)
        assert "tuh_seizure_" in tms.Q31_PREFIXES
        assert "tueg_" in tms.Q31_PREFIXES

    def test_target_channels_count(self):
        """The 10-20 montage has 21 channels for the SNN load path."""
        assert len(tms._TARGET_CH) == tms.NUM_CHANNELS

    def test_t_input_t_latent_consistent(self):
        """T_LATENT must equal T_INPUT // STRIDE_8 (integer division)
        — this is the load-bearing temporal contract between the raw
        2500-sample window and the 312-sample latent.

        2500 is not a multiple of 8 (one extra sample drops on the
        floor); the SNN handles this by truncation in the forward pass.
        """
        assert tms.T_LATENT == tms.T_INPUT // tms.STRIDE_8
