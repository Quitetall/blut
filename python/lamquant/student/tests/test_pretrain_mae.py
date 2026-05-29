"""Unit tests for ai_models/student/pretrain_mae.py — Phase 3.

MAEPredictionHead + create_mask covered directly; run_pretraining +
main() exercised end-to-end with stubbed DatasetManifest +
PrecomputedL3Dataset (synthetic single-batch prefetch).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

import pretrain_mae as pm

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# MAEPredictionHead
# ---------------------------------------------------------------------------
class TestMAEPredictionHead:
    def test_forward_shape(self):
        head = pm.MAEPredictionHead(latent_dim=32, n_channels=21, l3_len=313)
        lat = torch.randn(2, 32, 79)
        out = head(lat)
        assert out.shape == (2, 21, 313)

    def test_custom_dims(self):
        head = pm.MAEPredictionHead(latent_dim=16, n_channels=8, l3_len=200)
        lat = torch.randn(1, 16, 50)
        out = head(lat)
        assert out.shape[0] == 1
        assert out.shape[1] == 8
        assert out.shape[2] == 200


# ---------------------------------------------------------------------------
# create_mask
# ---------------------------------------------------------------------------
class TestCreateMask:
    def test_shape(self):
        m = pm.create_mask(batch_size=2, n_channels=4, l3_len=320,
                            mask_ratio=0.5, patch_size=16)
        assert m.shape == (2, 4, 320)

    def test_zero_ratio_unmasked(self):
        m = pm.create_mask(batch_size=1, n_channels=4, l3_len=320,
                            mask_ratio=0.0, patch_size=16)
        assert (m == 0).all()

    def test_half_ratio_masks_half_patches(self):
        torch.manual_seed(0)
        m = pm.create_mask(batch_size=1, n_channels=1, l3_len=320,
                            mask_ratio=0.5, patch_size=16)
        # 320/16 = 20 patches, 10 masked → 10 * 16 = 160 timesteps masked
        n_masked = int(m.sum().item())
        assert n_masked == 160

    def test_same_mask_across_channels(self):
        m = pm.create_mask(batch_size=1, n_channels=4, l3_len=320,
                            mask_ratio=0.5, patch_size=16)
        # All channels should have the same mask pattern
        for c in range(1, 4):
            assert torch.equal(m[0, c], m[0, 0])

    def test_different_mask_across_batch(self):
        torch.manual_seed(0)
        m = pm.create_mask(batch_size=4, n_channels=1, l3_len=320,
                            mask_ratio=0.5, patch_size=16)
        # At least one pair should differ (high probability)
        all_same = all(torch.equal(m[0], m[b]) for b in range(1, 4))
        assert not all_same


# ---------------------------------------------------------------------------
# run_pretraining + main (with stubbed heavy deps)
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_deps(monkeypatch):
    """Patch the data + model imports so run_pretraining runs end-to-end on CPU."""
    # Fake manifest
    manifest_mod = types.ModuleType("data_types")
    class _FakeManifest:
        @classmethod
        def load(cls, _p): return cls()
        def get_file_entries(self, _split): return [object()]

    manifest_mod.DatasetManifest = _FakeManifest
    manifest_mod.Split = MagicMock()
    monkeypatch.setitem(sys.modules, "data_types", manifest_mod)

    # Fake dataset that yields one batch
    streaming_mod = types.ModuleType("streaming_dataset")
    class _FakeDataset:
        def __init__(self, **kw):
            pass
        def prefetch_batches(self, batch_size, device):
            l3 = torch.randn(batch_size, 21, 313, device=device)
            yield (l3,)
    streaming_mod.PrecomputedL3Dataset = _FakeDataset
    monkeypatch.setitem(sys.modules, "streaming_dataset", streaming_mod)

    # Fake encoder package
    class _FakeEncoder(torch.nn.Module):
        def __init__(self, in_ch=21, latent_dim=32):
            super().__init__()
            self.proj = torch.nn.Conv1d(in_ch, latent_dim, kernel_size=1)

        def encode(self, x, quantize=False):
            # x [B, 21, 313] → [B, 32, 79]. Use 1x1 conv then AdaptiveAvgPool.
            x = self.proj(x)
            return torch.nn.functional.adaptive_avg_pool1d(x, 79)

    encoder_pkg = types.ModuleType("lamquant_codec")
    encoder_models = types.ModuleType("lamquant_neural.models")
    encoder_mod = types.ModuleType("lamquant_neural.models.encoder")
    encoder_mod.TernaryMobileNetV5_Subband = _FakeEncoder
    encoder_pkg.models = encoder_models
    encoder_models.encoder = encoder_mod
    monkeypatch.setitem(sys.modules, "lamquant_codec", encoder_pkg)
    monkeypatch.setitem(sys.modules, "lamquant_neural.models", encoder_models)
    monkeypatch.setitem(sys.modules, "lamquant_neural.models.encoder", encoder_mod)

    yield


class TestRunPretraining:
    def test_runs_one_epoch(self, stub_deps, tmp_path):
        out = pm.run_pretraining(
            epochs=1, mask_ratio=0.5, patch_size=16, lr=1e-3,
            batch_size=2, windows_per_epoch=2, seed=0,
            output_path=str(tmp_path / "mae.ckpt"),
        )
        assert Path(out).is_file()
        ck = torch.load(out, weights_only=False)
        assert ck["pretraining"] == "mae"
        assert ck["epochs"] == 1
        assert ck["mask_ratio"] == 0.5

    def test_runs_two_epochs(self, stub_deps, tmp_path):
        out = pm.run_pretraining(
            epochs=2, mask_ratio=0.3, patch_size=16, lr=1e-4,
            batch_size=2, windows_per_epoch=2, seed=1,
            output_path=str(tmp_path / "mae.ckpt"),
        )
        ck = torch.load(out, weights_only=False)
        assert ck["epochs"] == 2

    def test_default_output_path(self, stub_deps, monkeypatch, tmp_path):
        # Override ROOT_DIR for default save path
        monkeypatch.setattr(pm, "ROOT_DIR", str(tmp_path))
        out = pm.run_pretraining(
            epochs=1, mask_ratio=0.5, patch_size=16, lr=1e-3,
            batch_size=2, windows_per_epoch=2, seed=0,
            output_path=None,
        )
        assert "pretrained_mae.ckpt" in out


class TestMain:
    def test_main_runs(self, stub_deps, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv",
                            ["pretrain_mae", "--epochs", "1",
                             "--batch-size", "2", "--windows-per-epoch", "2",
                             "--output", str(tmp_path / "out.ckpt")])
        assert pm.main() == 0
        assert (tmp_path / "out.ckpt").is_file()
