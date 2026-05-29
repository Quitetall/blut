"""Deep coverage for ai_models/decoder/perceptual_losses.py.

Each loss class is an ``nn.Module`` whose contract is:
  - constructs on CPU without requiring the foundation-model weights
  - ``extract_features`` returns a [B, D] tensor for [B, 21, T] input
  - ``feature_loss`` returns a finite non-negative scalar Tensor
  - identical inputs produce ~0 loss (modulo float jitter)

Per ``feedback_futureproof_tests``: we pin shape/type/boundedness/
sign invariants, not specific numeric values that would drift if the
fallback projection's internal layer config changes.

Math fixtures (``torch.randn``) are allowed — these losses are tested
in their fallback-projection paths (no foundation-model checkpoints on
CI) and the contract is "produces a finite feature vector from a real
shape", not "produces clinically meaningful features from real EEG".
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.l2


# parents[3] == blut/python (the package root for `lamquant.*`).
_PY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PY_ROOT))


@pytest.fixture(scope="module")
def perceptual():
    """Import the perceptual_losses module once per test module.

    Module-level imports happen at construction time so we want one
    instance per module not per test.
    """
    return importlib.import_module("lamquant.decoder.perceptual_losses")


def _eeg_batch(batch=2, channels=21, t=2500, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, channels, t)


# ============================================================
# DACFeatureExtractor — fallback path
# ============================================================


class TestDACFeatureExtractor:
    def test_constructs(self, perceptual):
        ext = perceptual.DACFeatureExtractor(device="cpu")
        assert isinstance(ext, torch.nn.Module)

    def test_spectral_fallback_shape(self, perceptual):
        ext = perceptual.DACFeatureExtractor(device="cpu")
        x = _eeg_batch(batch=2, channels=21, t=2500)
        feats = ext._spectral_fallback(x)
        # Multi-resolution STFT → concatenated mag features.
        assert feats.ndim == 2
        assert feats.shape[0] == 2
        assert feats.shape[1] > 0
        assert torch.isfinite(feats).all()

    def test_extract_features_shape(self, perceptual):
        ext = perceptual.DACFeatureExtractor(device="cpu")
        x = _eeg_batch(batch=2, channels=21, t=2500)
        feats = ext.extract_features(x)
        assert feats.ndim == 2
        assert feats.shape[0] == 2
        assert torch.isfinite(feats).all()

    def test_feature_loss_finite_nonnegative_scalar(self, perceptual):
        ext = perceptual.DACFeatureExtractor(device="cpu")
        pred = _eeg_batch(batch=2, channels=21, t=2500, seed=1)
        target = _eeg_batch(batch=2, channels=21, t=2500, seed=2)
        loss = ext.feature_loss(pred, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_identical_inputs_zero_loss(self, perceptual):
        ext = perceptual.DACFeatureExtractor(device="cpu")
        x = _eeg_batch(batch=1, channels=21, t=2500, seed=3)
        loss = ext.feature_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ============================================================
# LaBraMFeatureExtractor — fallback path (no checkpoint on CI)
# ============================================================


class TestLaBraMFeatureExtractor:
    def test_constructs(self, perceptual):
        ext = perceptual.LaBraMFeatureExtractor(device="cpu")
        assert isinstance(ext, torch.nn.Module)
        assert hasattr(ext, "proj")

    def test_extract_features_shape(self, perceptual):
        ext = perceptual.LaBraMFeatureExtractor(device="cpu")
        x = _eeg_batch(batch=3, channels=21, t=2500)
        feats = ext.extract_features(x)
        # proj: stride 4 → stride 4 → AdaptiveAvgPool1d(16) → 128*16=2048
        assert feats.ndim == 2
        assert feats.shape[0] == 3
        assert feats.shape[1] == 128 * 16
        assert torch.isfinite(feats).all()

    def test_feature_loss_finite_nonnegative_scalar(self, perceptual):
        ext = perceptual.LaBraMFeatureExtractor(device="cpu")
        pred = _eeg_batch(batch=2, channels=21, t=2500, seed=11)
        target = _eeg_batch(batch=2, channels=21, t=2500, seed=12)
        loss = ext.feature_loss(pred, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_identical_inputs_zero_loss(self, perceptual):
        ext = perceptual.LaBraMFeatureExtractor(device="cpu")
        x = _eeg_batch(batch=1, channels=21, t=2500, seed=13)
        loss = ext.feature_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ============================================================
# FEMBAFeatureExtractor — fallback path (CPU, no mamba_ssm)
# ============================================================


class TestFEMBAFeatureExtractor:
    @pytest.mark.parametrize("variant", ["tiny", "base", "large"])
    def test_constructs_all_variants(self, perceptual, variant):
        ext = perceptual.FEMBAFeatureExtractor(device="cpu", variant=variant)
        assert isinstance(ext, torch.nn.Module)
        # On CPU we MUST be in fallback mode (mamba_ssm requires CUDA).
        assert ext._loaded is False

    def test_extract_features_shape(self, perceptual):
        ext = perceptual.FEMBAFeatureExtractor(device="cpu", variant="tiny")
        x = _eeg_batch(batch=2, channels=21, t=2500)
        feats = ext.extract_features(x)
        # Fallback proj: same structure as LaBraM
        assert feats.ndim == 2
        assert feats.shape[0] == 2
        assert feats.shape[1] == 128 * 16
        assert torch.isfinite(feats).all()

    def test_feature_loss_finite_nonnegative_scalar(self, perceptual):
        ext = perceptual.FEMBAFeatureExtractor(device="cpu", variant="tiny")
        pred = _eeg_batch(batch=2, channels=21, t=2500, seed=21)
        target = _eeg_batch(batch=2, channels=21, t=2500, seed=22)
        loss = ext.feature_loss(pred, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_identical_inputs_zero_loss(self, perceptual):
        ext = perceptual.FEMBAFeatureExtractor(device="cpu", variant="tiny")
        x = _eeg_batch(batch=1, channels=21, t=2500, seed=23)
        loss = ext.feature_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ============================================================
# ZUNAFeatureExtractor — always fallback
# ============================================================


class TestZUNAFeatureExtractor:
    def test_constructs(self, perceptual):
        ext = perceptual.ZUNAFeatureExtractor(device="cpu")
        assert isinstance(ext, torch.nn.Module)
        assert hasattr(ext, "proj")

    def test_extract_features_shape(self, perceptual):
        ext = perceptual.ZUNAFeatureExtractor(device="cpu")
        x = _eeg_batch(batch=2, channels=21, t=2500)
        feats = ext.extract_features(x)
        # proj: Conv1d 21→128 stride4 → 128→256 stride4 → AdaptiveAvgPool(16)
        # So output is [B, 256*16].
        assert feats.ndim == 2
        assert feats.shape[0] == 2
        assert feats.shape[1] == 256 * 16
        assert torch.isfinite(feats).all()

    def test_feature_loss_finite_nonnegative_scalar(self, perceptual):
        ext = perceptual.ZUNAFeatureExtractor(device="cpu")
        pred = _eeg_batch(batch=2, channels=21, t=2500, seed=31)
        target = _eeg_batch(batch=2, channels=21, t=2500, seed=32)
        loss = ext.feature_loss(pred, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_identical_inputs_zero_loss(self, perceptual):
        ext = perceptual.ZUNAFeatureExtractor(device="cpu")
        x = _eeg_batch(batch=1, channels=21, t=2500, seed=33)
        loss = ext.feature_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ============================================================
# EEGPTFeatureExtractor — always fallback (no checkpoint loaded)
# ============================================================


class TestEEGPTFeatureExtractor:
    def test_constructs(self, perceptual):
        ext = perceptual.EEGPTFeatureExtractor(device="cpu")
        assert isinstance(ext, torch.nn.Module)
        assert hasattr(ext, "proj")

    def test_extract_features_shape(self, perceptual):
        ext = perceptual.EEGPTFeatureExtractor(device="cpu")
        x = _eeg_batch(batch=2, channels=21, t=2500)
        feats = ext.extract_features(x)
        # proj: Conv1d 21→64 stride4 → 64→128 stride4 → AdaptiveAvgPool(16)
        assert feats.ndim == 2
        assert feats.shape[0] == 2
        assert feats.shape[1] == 128 * 16
        assert torch.isfinite(feats).all()

    def test_feature_loss_finite_nonnegative_scalar(self, perceptual):
        ext = perceptual.EEGPTFeatureExtractor(device="cpu")
        pred = _eeg_batch(batch=2, channels=21, t=2500, seed=41)
        target = _eeg_batch(batch=2, channels=21, t=2500, seed=42)
        loss = ext.feature_loss(pred, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_identical_inputs_zero_loss(self, perceptual):
        ext = perceptual.EEGPTFeatureExtractor(device="cpu")
        x = _eeg_batch(batch=1, channels=21, t=2500, seed=43)
        loss = ext.feature_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ============================================================
# MultiTeacherPerceptualLoss
# ============================================================


class TestMultiTeacherPerceptualLoss:
    def test_constructs(self, perceptual):
        mt = perceptual.MultiTeacherPerceptualLoss(device="cpu")
        assert isinstance(mt, torch.nn.Module)
        assert hasattr(mt, "labram")
        assert hasattr(mt, "dac")
        assert hasattr(mt, "femba")

    def test_weights_stored(self, perceptual):
        mt = perceptual.MultiTeacherPerceptualLoss(
            device="cpu", alpha=0.5, beta=0.1, gamma=0.2, delta=0.2)
        assert mt.alpha == 0.5
        assert mt.beta == 0.1
        assert mt.gamma == 0.2
        assert mt.delta == 0.2

    def test_forward_returns_scalar_and_components(self, perceptual):
        mt = perceptual.MultiTeacherPerceptualLoss(device="cpu")
        pred = _eeg_batch(batch=1, channels=21, t=2500, seed=51)
        target = _eeg_batch(batch=1, channels=21, t=2500, seed=52)
        total, components = mt(pred, target)
        # Scalar loss, finite, non-negative.
        assert total.ndim == 0
        assert torch.isfinite(total)
        assert total.item() >= 0
        # Components dict has the four documented keys.
        assert set(components.keys()) == {"recon", "labram", "dac", "femba"}
        for k, v in components.items():
            assert isinstance(v, float)
            assert v >= 0

    def test_identical_inputs_near_zero(self, perceptual):
        mt = perceptual.MultiTeacherPerceptualLoss(device="cpu")
        x = _eeg_batch(batch=1, channels=21, t=2500, seed=53)
        total, comps = mt(x, x)
        # All components should be ~0.
        assert total.item() == pytest.approx(0.0, abs=1e-5)
        for v in comps.values():
            assert v == pytest.approx(0.0, abs=1e-5)


# ============================================================
# SoftDTWLoss
# ============================================================


class TestSoftDTWLoss:
    def test_constructs(self, perceptual):
        dtw = perceptual.SoftDTWLoss(gamma=0.1)
        assert dtw.gamma == 0.1

    def test_soft_dtw_finite_scalar(self, perceptual):
        # NOTE: Soft-DTW with logsumexp soft-min relaxation is NOT a
        # metric — it can be negative (γ * log(K) bias term from the
        # soft-min). Contract here is only "finite scalar".
        dtw = perceptual.SoftDTWLoss(gamma=0.1)
        x = torch.linspace(0, 1, 8)
        d = dtw._soft_dtw(x, x)
        assert d.ndim == 0
        assert torch.isfinite(d)

    def test_forward_finite_scalar(self, perceptual):
        dtw = perceptual.SoftDTWLoss(gamma=0.5)
        # Use small tensors — DTW is O(T^2).
        pred = torch.randn(1, 3, 64)
        target = pred + 0.01 * torch.randn(1, 3, 64)
        loss = dtw(pred, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_forward_differs_for_different_inputs(self, perceptual):
        # Different inputs should give a different soft-DTW than identical.
        dtw = perceptual.SoftDTWLoss(gamma=0.1)
        x = torch.randn(1, 2, 32)
        d_same = dtw(x, x).item()
        d_diff = dtw(x, x + 1.0).item()
        # Adding a constant shift should give a different (larger) cost,
        # because the cost matrix has nonzero squared-diff entries.
        assert d_diff > d_same


# ============================================================
# LaBraM checkpoint-load path (lines 121-136)
# ============================================================


class TestLaBraMCheckpointPath:
    def test_loads_when_checkpoint_present(self, perceptual, tmp_path,
                                            monkeypatch):
        # Plant a fake LaBraM checkpoint at the expected path.
        ref = tmp_path / "REF"
        ckpt_dir = ref / "labram" / "repo" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt_path = ckpt_dir / "labram-base.pth"
        # Use a state-dict-like dict with a 'model' wrapper to exercise
        # the unwrapping branch.
        torch.save({"model": {"layer.weight": torch.zeros(3, 3)}}, ckpt_path)
        monkeypatch.setattr(perceptual, "_REF_DIR", str(ref))

        ext = perceptual.LaBraMFeatureExtractor(device="cpu")
        assert ext._loaded is True
        assert hasattr(ext, "_encoder_state")
        # Sanity: forward still works (uses fallback proj regardless).
        x = _eeg_batch(batch=1, channels=21, t=2500)
        feats = ext.extract_features(x)
        assert feats.ndim == 2

    def test_loads_when_checkpoint_bare_state_dict(self, perceptual,
                                                     tmp_path, monkeypatch):
        # Same fake checkpoint but without the 'model' wrapper key.
        ref = tmp_path / "REF"
        ckpt_dir = ref / "labram" / "repo" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt_path = ckpt_dir / "labram-base.pth"
        torch.save({"layer.weight": torch.zeros(2, 2)}, ckpt_path)
        monkeypatch.setattr(perceptual, "_REF_DIR", str(ref))
        ext = perceptual.LaBraMFeatureExtractor(device="cpu")
        assert ext._loaded is True
        assert "layer.weight" in ext._encoder_state


# ============================================================
# EEGPT eegpt_path branch (lines 442-449)
# ============================================================


class TestEEGPTPathBranch:
    def test_eegpt_path_exists_branch_runs(self, perceptual, tmp_path,
                                            monkeypatch):
        # Plant an eegpt/repo dir to exercise the branch that prints
        # "EEGPT code found".
        ref = tmp_path / "REF"
        eegpt_repo = ref / "eegpt" / "repo"
        eegpt_repo.mkdir(parents=True)
        monkeypatch.setattr(perceptual, "_REF_DIR", str(ref))
        ext = perceptual.EEGPTFeatureExtractor(device="cpu")
        # Fallback proj should still be there regardless of branch taken.
        assert hasattr(ext, "proj")


# ============================================================
# load_wavtokenizer_weights
# ============================================================


class TestLoadWavTokenizerWeights:
    def test_no_repo_returns_zero(self, perceptual, monkeypatch):
        # Force the wavtokenizer path lookup to a non-existent dir.
        monkeypatch.setattr(perceptual, "_REF_DIR", "/nonexistent_dir_xyz_123")
        decoder = torch.nn.Conv1d(21, 32, 3)
        n = perceptual.load_wavtokenizer_weights(decoder, device="cpu")
        assert n == 0

    def test_missing_checkpoint_returns_zero(self, perceptual, tmp_path,
                                              monkeypatch):
        # Create a wavtokenizer/repo dir but no checkpoints.
        ref = tmp_path / "REF"
        (ref / "wavtokenizer" / "repo").mkdir(parents=True)
        monkeypatch.setattr(perceptual, "_REF_DIR", str(ref))
        decoder = torch.nn.Conv1d(21, 32, 3)
        n = perceptual.load_wavtokenizer_weights(decoder, device="cpu")
        # Either 0 because no checkpoint, or it falls through the try/except.
        assert n == 0

    def test_present_checkpoint_transfers_matching_layers(self, perceptual,
                                                           tmp_path,
                                                           monkeypatch):
        # Build a small decoder, plant a checkpoint whose key matches
        # one of its layers, verify transfer count == 1.
        decoder = torch.nn.Sequential()
        # Construct a decoder with a 'blocks.0.weight' layer (the name
        # the loader rewrites 'backbone.0.weight' → 'blocks.0.weight'
        # via .replace('backbone.', 'blocks.')).
        decoder.add_module("blocks", torch.nn.Sequential(
            torch.nn.Conv1d(4, 8, 3)))

        ref = tmp_path / "REF"
        wt_dir = ref / "wavtokenizer" / "repo" / "pretrained"
        wt_dir.mkdir(parents=True)
        ckpt_path = wt_dir / "wavtokenizer.pt"
        # Match the decoder's named_parameters key after the rename:
        # 'backbone.0.weight' → 'blocks.0.weight' which IS in decoder.
        state = {
            "backbone.0.weight": torch.zeros(8, 4, 3),  # matches
            "backbone.0.bias": torch.zeros(8),  # matches
            "bogus.layer.weight": torch.zeros(2, 2),  # no match
        }
        torch.save(state, ckpt_path)
        monkeypatch.setattr(perceptual, "_REF_DIR", str(ref))

        n = perceptual.load_wavtokenizer_weights(decoder, device="cpu")
        # At least one layer should match by the rename rule.
        assert n >= 1
