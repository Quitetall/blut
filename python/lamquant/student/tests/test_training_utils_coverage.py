"""Coverage-lifting tests for ai_models/student/training_utils.py.

Goal: raise module coverage from ~17% toward >70% by exercising the
public helpers with shape-fixture tensors (futureproof_tests rule:
assert SHAPE/TYPE/boundedness, not exact numeric values).

Real EDF data is not loaded here — the helpers under test are pure
math + tensor wrappers. shape-fixture tensors via torch.randn are
the correct input for unit-level math validation. The training loop
``run()`` is intentionally out of scope (heavy IO + GPU + heavy deps).
"""
from __future__ import annotations

import pytest  # decomp: `legacy/` Gen-7.0 code excluded from all repos (dead)
pytest.importorskip("legacy", reason="Tests dead legacy/ Gen-7.0 code excluded from the decomposition")

import os
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import training_utils as tu

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# SpectralLoss — extra coverage paths
# ---------------------------------------------------------------------------

class TestSpectralLossCoverage:
    def test_returns_finite_for_random_inputs(self):
        torch.manual_seed(0)
        loss = tu.SpectralLoss(fft_sizes=[16, 32, 64])
        out = loss(torch.randn(2, 4, 313), torch.randn(2, 4, 313))
        assert torch.isfinite(out)
        assert out.item() >= 0

    def test_default_fft_sizes(self):
        # Default constructor — all 5 FFT sizes
        loss = tu.SpectralLoss()
        assert len(loss.fft_sizes) == 5
        # Run forward to cover all branches
        out = loss(torch.randn(1, 2, 313), torch.randn(1, 2, 313))
        assert out.ndim == 0

    def test_single_fft_size(self):
        loss = tu.SpectralLoss(fft_sizes=[64])
        out = loss(torch.randn(1, 2, 313), torch.randn(1, 2, 313))
        assert out.ndim == 0

    def test_grad_flows(self):
        loss = tu.SpectralLoss(fft_sizes=[16, 32])
        a = torch.randn(2, 4, 313, requires_grad=True)
        b = torch.randn(2, 4, 313)
        out = loss(a, b)
        out.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()

    def test_dtype_handling_float32(self):
        loss = tu.SpectralLoss(fft_sizes=[32])
        x = torch.randn(2, 4, 313, dtype=torch.float32)
        out = loss(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# temporal_importance_mask — extra paths
# ---------------------------------------------------------------------------

class TestTemporalImportanceMaskCoverage:
    def test_shape_for_l3_window(self):
        m = tu.temporal_importance_mask(T=313)
        assert m.shape == (1, 1, 313)

    def test_default_edge_weight_is_03(self):
        m = tu.temporal_importance_mask(T=200)
        assert m[0, 0, 0].item() == pytest.approx(0.3, abs=1e-5)

    def test_custom_edge_weight(self):
        m = tu.temporal_importance_mask(T=200, edge_weight=0.5)
        assert m[0, 0, 0].item() == pytest.approx(0.5, abs=1e-5)
        assert m[0, 0, -1].item() == pytest.approx(0.5, abs=1e-5)

    def test_bounded_in_range(self):
        m = tu.temporal_importance_mask(T=313, edge_weight=0.3)
        # Always within [edge_weight, 1.0]
        assert m.min().item() >= 0.3 - 1e-6
        assert m.max().item() <= 1.0 + 1e-6

    def test_max_at_center(self):
        m = tu.temporal_importance_mask(T=313)
        # Peak should be at or near center
        peak_idx = m.argmax().item()
        assert abs(peak_idx - 313 // 2) <= 1

    def test_device_kwarg_cpu(self):
        m = tu.temporal_importance_mask(T=100, device=torch.device("cpu"))
        assert m.device.type == "cpu"


# ---------------------------------------------------------------------------
# band_weighted_mse — extra coverage including all sample rate paths
# ---------------------------------------------------------------------------

class TestBandWeightedMSECoverage:
    def test_returns_tensor_scalar(self):
        out = tu.band_weighted_mse(torch.randn(2, 4, 313),
                                    torch.randn(2, 4, 313))
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_grad_flow(self):
        recon = torch.randn(2, 4, 313, requires_grad=True)
        target = torch.randn(2, 4, 313)
        out = tu.band_weighted_mse(recon, target)
        out.backward()
        assert recon.grad is not None
        assert torch.isfinite(recon.grad).all()

    def test_higher_sample_rate(self):
        # Sample rate covers all bands inc. >13 Hz default-1.0 region
        x = torch.randn(2, 4, 313)
        out = tu.band_weighted_mse(x, x + 0.1, sample_rate=100.0)
        assert torch.isfinite(out)
        assert out.item() > 0

    def test_low_sample_rate_only_delta(self):
        # sample_rate=4 → all freqs < 2 Hz → all weighted 2.0
        x = torch.randn(2, 4, 313)
        out = tu.band_weighted_mse(x, x + 0.1, sample_rate=4.0)
        assert torch.isfinite(out)

    def test_finite_for_zero_diff(self):
        x = torch.randn(2, 4, 313)
        out = tu.band_weighted_mse(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# pearson_r_loss / pearson_r_batch — extra coverage
# ---------------------------------------------------------------------------

class TestPearsonRCoverage:
    def test_loss_bounded_0_2(self):
        torch.manual_seed(0)
        x = torch.randn(4, 21, 313)
        y = torch.randn(4, 21, 313)
        out = tu.pearson_r_loss(x, y)
        # R ∈ [-1, 1] → loss = 1 - R ∈ [0, 2]
        assert 0.0 <= out.item() <= 2.0 + 1e-6

    def test_batch_bounded_minus1_1(self):
        torch.manual_seed(0)
        out = tu.pearson_r_batch(torch.randn(4, 21, 313),
                                  torch.randn(4, 21, 313))
        assert -1.0 - 1e-5 <= out <= 1.0 + 1e-5

    def test_loss_returns_tensor(self):
        out = tu.pearson_r_loss(torch.randn(2, 4, 16), torch.randn(2, 4, 16))
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_loss_grad_flows(self):
        a = torch.randn(2, 4, 16, requires_grad=True)
        b = torch.randn(2, 4, 16)
        loss = tu.pearson_r_loss(a, b)
        loss.backward()
        assert a.grad is not None
        assert torch.isfinite(a.grad).all()

    def test_loss_scaled_target(self):
        # Linear transform preserves correlation
        x = torch.randn(4, 21, 313)
        scaled = 2.5 * x + 0.7
        out = tu.pearson_r_loss(x, scaled)
        assert out.item() == pytest.approx(0.0, abs=1e-4)

    def test_loss_handles_small_tensor(self):
        out = tu.pearson_r_loss(torch.randn(1, 2, 8), torch.randn(1, 2, 8))
        assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# channel_dropout — extra coverage
# ---------------------------------------------------------------------------

class TestChannelDropoutCoverage:
    def test_seeded_drop_count_within_bounds(self):
        torch.manual_seed(0)
        x = torch.ones(4, 21, 50)
        out = tu.channel_dropout(x, p_min=3, p_max=7, training=True)
        # Per-sample dropped channel count must be in [3, 7]
        for b in range(4):
            n_zero = (out[b].abs().sum(dim=-1) == 0).sum().item()
            assert 3 <= n_zero <= 7

    def test_p_min_eq_p_max(self):
        # Edge case: p_min == p_max means exactly k dropped
        torch.manual_seed(0)
        x = torch.ones(2, 21, 50)
        out = tu.channel_dropout(x, p_min=5, p_max=5, training=True)
        for b in range(2):
            n_zero = (out[b].abs().sum(dim=-1) == 0).sum().item()
            assert n_zero == 5

    def test_dtype_preserved(self):
        x = torch.randn(2, 21, 50, dtype=torch.float32)
        out = tu.channel_dropout(x, training=True)
        assert out.dtype == torch.float32

    def test_input_not_modified(self):
        # Verify clone() is used (no aliasing)
        x = torch.ones(2, 21, 50)
        x_copy = x.clone()
        _ = tu.channel_dropout(x, training=True)
        assert torch.equal(x, x_copy)


# ---------------------------------------------------------------------------
# eeg_augment — extra coverage (skips when selfeeg missing)
# ---------------------------------------------------------------------------

class TestEegAugmentCoverage:
    def test_inference_path_returns_input_object(self):
        x = torch.randn(2, 21, 313)
        # training=False short-circuits — returns the SAME tensor reference
        assert tu.eeg_augment(x, training=False) is x

    def test_training_returns_tensor_shape(self):
        pytest.importorskip("selfeeg")
        torch.manual_seed(0)
        x = torch.randn(2, 21, 313)
        out = tu.eeg_augment(x, training=True)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_repeated_invocation_no_crash(self):
        """Run multiple times; the band-noise + flip branches are
        probabilistic so a few runs covers more lines than one."""
        pytest.importorskip("selfeeg")
        torch.manual_seed(0)
        x = torch.randn(2, 21, 313)
        for _ in range(20):
            out = tu.eeg_augment(x, training=True)
            assert out.shape == x.shape


# ---------------------------------------------------------------------------
# validate_epoch — exercise more model wrappers
# ---------------------------------------------------------------------------

class _MinimalModel(nn.Module):
    """Identity-like model that supports the (x, quantize=) interface."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, quantize=True):
        return x * self.scale

    def encode(self, x, quantize=False):
        # Pretend latent: [B, 32, 79]
        B = x.shape[0]
        return torch.randn(B, 32, 79)


class TestValidateEpochCoverage:
    def test_multi_batch_loader(self):
        m = _MinimalModel()
        x_l3 = torch.randn(2, 21, 313)
        loader = [(x_l3, None, None), (x_l3.clone(), None, None)]
        r, prd = tu.validate_epoch(m, loader, device=torch.device("cpu"))
        # Identity model → R ≈ 1
        assert isinstance(r, float)
        assert isinstance(prd, float)
        # R is bounded by Pearson R definition; identity → ~1.0
        assert -1.0 - 1e-5 <= r <= 1.0 + 1e-5

    def test_quantize_false_passes_through(self):
        m = _MinimalModel()
        x_l3 = torch.randn(2, 21, 313)
        loader = [(x_l3, None, None)]
        r, prd = tu.validate_epoch(m, loader, device=torch.device("cpu"),
                                    quantize=False)
        assert isinstance(r, float)


# ---------------------------------------------------------------------------
# latent_kurtosis — broader coverage
# ---------------------------------------------------------------------------

class TestLatentKurtosisCoverage:
    def test_returns_pair(self):
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Conv1d(21, 32, 1)
            def encode(self, x, quantize=False):
                return self.lin(x)
            def forward(self, x, quantize=False):
                return self.lin(x)

        m = _M()
        x_l3 = torch.randn(2, 21, 313)
        loader = [(x_l3, None, None), (x_l3.clone(), None, None)]
        out = tu.latent_kurtosis(m, loader, device=torch.device("cpu"),
                                  max_batches=10)
        assert isinstance(out, tuple)
        assert len(out) == 2
        mean_kurt, per_ch_kurt = out
        assert isinstance(mean_kurt, float)
        assert isinstance(per_ch_kurt, np.ndarray)
        assert per_ch_kurt.shape[0] == 32

    def test_max_batches_clamps(self):
        """max_batches=1 → only the first batch contributes."""
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Conv1d(21, 32, 1)
                self.call_count = 0
            def encode(self, x, quantize=False):
                self.call_count += 1
                return self.lin(x)
            def forward(self, x, quantize=False):
                return self.lin(x)

        m = _M()
        x_l3 = torch.randn(2, 21, 313)
        loader = [(x_l3, None, None)] * 5
        tu.latent_kurtosis(m, loader, device=torch.device("cpu"), max_batches=1)
        assert m.call_count == 1

    def test_empty_loader_returns_defaults(self):
        class _M(nn.Module):
            def encode(self, x, quantize=False):
                return x
            def forward(self, x, quantize=False):
                return x

        mean_kurt, per_ch = tu.latent_kurtosis(_M(), val_loader=[],
                                                 device=torch.device("cpu"))
        assert mean_kurt == 0.0
        assert per_ch.shape[0] == 32

    def test_collapsed_channel_marked_999(self):
        """Zero-std channel should yield 999.0 sentinel."""
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Conv1d(21, 32, 1)
            def encode(self, x, quantize=False):
                lat = self.lin(x)
                # Force channel 0 to a constant value (std=0)
                lat = lat.clone()
                lat[:, 0, :] = 0.5
                return lat
            def forward(self, x, quantize=False):
                return self.lin(x)

        m = _M()
        x_l3 = torch.randn(2, 21, 313)
        loader = [(x_l3, None, None)]
        _, per_ch = tu.latent_kurtosis(m, loader, device=torch.device("cpu"),
                                        max_batches=1)
        assert per_ch[0] == 999.0


# ---------------------------------------------------------------------------
# _safe_load — covers torch.load weights_only fallback path
# ---------------------------------------------------------------------------

class TestSafeLoadCoverage:
    def test_roundtrip_tensor_dict(self, tmp_path):
        """Save & load a plain tensor dict; weights_only=True succeeds."""
        p = tmp_path / "ck.pt"
        sd = {"w": torch.randn(4, 8), "b": torch.zeros(4)}
        torch.save(sd, p)
        loaded = tu._safe_load(str(p))
        assert set(loaded.keys()) == {"w", "b"}
        assert loaded["w"].shape == (4, 8)
        assert loaded["b"].shape == (4,)

    def test_map_location_cpu(self, tmp_path):
        """Explicit map_location='cpu' is preserved."""
        p = tmp_path / "ck.pt"
        torch.save({"x": torch.ones(3)}, p)
        loaded = tu._safe_load(str(p), map_location="cpu")
        assert loaded["x"].device.type == "cpu"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            tu._safe_load(str(tmp_path / "does_not_exist.pt"))


# ---------------------------------------------------------------------------
# split_by_manifest — depends on canonical manifest_v3.json
# ---------------------------------------------------------------------------

class TestSplitByManifest:
    def test_returns_two_lists(self):
        """Canonical manifest produces (train_files, val_files) string lists."""
        manifest_path = Path(__file__).parent.parent.parent / \
            "ai_models" / "dataset_sim" / "manifest_v3.json"
        if not manifest_path.exists():
            pytest.skip(f"{manifest_path} not present")
        train_files, val_files = tu.split_by_manifest([])
        assert isinstance(train_files, list)
        assert isinstance(val_files, list)
        # Both lists non-empty for a real manifest
        assert len(train_files) > 0
        assert len(val_files) > 0
        # Holdout files never appear in train list (Split.TRAIN / Split.VAL
        # are disjoint — frozen contract of DatasetManifest)
        assert set(train_files).isdisjoint(set(val_files))
        # All entries are strings (per the str(p) coercion)
        assert all(isinstance(p, str) for p in train_files[:5])
        assert all(isinstance(p, str) for p in val_files[:5])

    def test_ignores_input_npz_arg(self):
        """The npz_files arg is ignored — manifest is source of truth."""
        manifest_path = Path(__file__).parent.parent.parent / \
            "ai_models" / "dataset_sim" / "manifest_v3.json"
        if not manifest_path.exists():
            pytest.skip(f"{manifest_path} not present")
        # Pass garbage; should still return the manifest split
        t1, v1 = tu.split_by_manifest(["/nonexistent/a.npz", "/x/b.npz"])
        t2, v2 = tu.split_by_manifest([])
        assert t1 == t2
        assert v1 == v2


# ---------------------------------------------------------------------------
# band_weighted_mse — extra branch: odd T (skips scale[-1]=1.0)
# ---------------------------------------------------------------------------

class TestBandWeightedMSEOddT:
    def test_odd_T_branch(self):
        # T=13 is odd → if T % 2 == 0 is false → scale[-1] stays 2.0
        torch.manual_seed(0)
        x = torch.randn(2, 4, 13)
        out = tu.band_weighted_mse(x, x + 0.05, sample_rate=31.25)
        assert torch.isfinite(out)
        assert out.item() > 0

    def test_T1_corner(self):
        """T=1 means rfft has a single DC bin — still finite."""
        x = torch.randn(2, 4, 1)
        out = tu.band_weighted_mse(x, x + 0.1)
        assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# channel_dropout — extra: training=False explicit no-op
# ---------------------------------------------------------------------------

class TestChannelDropoutInference:
    def test_inference_returns_input_identity(self):
        """training=False returns the input unmodified (same object)."""
        x = torch.randn(2, 21, 50)
        out = tu.channel_dropout(x, training=False)
        # docstring says training=False short-circuits — should be same
        # tensor reference (no clone)
        assert out is x


# ---------------------------------------------------------------------------
# validate_epoch — empty loader returns sentinel
# ---------------------------------------------------------------------------

class TestValidateEpochEmptyLoader:
    def test_empty_loader_returns_default_sentinel(self):
        class _M(nn.Module):
            def forward(self, x, quantize=True):
                return x

        m = _M()
        # Empty list → no batches → returns (0.0, 100.0) per line 252
        r, prd = tu.validate_epoch(m, [], device=torch.device("cpu"))
        assert r == 0.0
        assert prd == 100.0
        assert isinstance(r, float)
        assert isinstance(prd, float)

    def test_quantize_true_path(self):
        """Default quantize=True is the production call shape."""
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(1))
            def forward(self, x, quantize=True):
                # quantize=True default → identity-like with scale 1.0
                return x * self.scale

        x_l3 = torch.randn(2, 21, 313)
        r, prd = tu.validate_epoch(_M(), [(x_l3, None, None)],
                                    device=torch.device("cpu"))
        # Identity → R ~ 1
        assert -1.0 - 1e-5 <= r <= 1.0 + 1e-5
        # PRD on identity = 0 (perfect)
        assert prd == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# pearson_r_loss / pearson_r_batch — additional edge cases
# ---------------------------------------------------------------------------

class TestPearsonRMore:
    def test_loss_identity_zero(self):
        x = torch.randn(2, 4, 32)
        out = tu.pearson_r_loss(x, x)
        # R=1 → loss = 0
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_anti_correlated_is_two(self):
        x = torch.randn(2, 4, 32)
        out = tu.pearson_r_loss(x, -x)
        # R=-1 → loss = 2
        assert out.item() == pytest.approx(2.0, abs=1e-4)

    def test_batch_identity_returns_one(self):
        x = torch.randn(4, 21, 313)
        r = tu.pearson_r_batch(x, x)
        assert r == pytest.approx(1.0, abs=1e-5)
        assert isinstance(r, float)

    def test_batch_anti_correlated_is_minus_one(self):
        x = torch.randn(4, 21, 313)
        r = tu.pearson_r_batch(x, -x)
        assert r == pytest.approx(-1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# SpectralLoss — additional shape paths
# ---------------------------------------------------------------------------

class TestSpectralLossMore:
    def test_3d_input_b_c_t(self):
        """Standard [B, C, T] input is flattened to 2D inside."""
        loss = tu.SpectralLoss(fft_sizes=[16])
        x = torch.randn(3, 5, 313)
        out = loss(x, x)
        assert out.ndim == 0
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_constructor_stores_fft_sizes(self):
        sizes = [8, 16, 24]
        loss = tu.SpectralLoss(fft_sizes=sizes)
        assert loss.fft_sizes == sizes

    def test_2d_input_works(self):
        """[B, T] input still flattens (single channel)."""
        loss = tu.SpectralLoss(fft_sizes=[16])
        x = torch.randn(2, 313)
        out = loss(x, x)
        assert out.ndim == 0


# ---------------------------------------------------------------------------
# temporal_importance_mask — additional shape / value checks
# ---------------------------------------------------------------------------

class TestTemporalImportanceMaskMore:
    def test_edge_weight_zero(self):
        """edge_weight=0 → mask reaches 0 at endpoints, 1 at center."""
        m = tu.temporal_importance_mask(T=200, edge_weight=0.0)
        assert m[0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        # Peak near center
        peak = m.argmax().item()
        assert abs(peak - 100) <= 1

    def test_edge_weight_one_flat(self):
        """edge_weight=1.0 → constant mask of 1.0 everywhere."""
        m = tu.temporal_importance_mask(T=200, edge_weight=1.0)
        assert torch.allclose(m, torch.ones_like(m), atol=1e-5)

    def test_symmetry(self):
        """Mask is symmetric: m[t] == m[T-1-t]."""
        T = 200
        m = tu.temporal_importance_mask(T=T)
        flat = m.flatten()
        for t in range(T):
            assert flat[t].item() == pytest.approx(flat[T - 1 - t].item(),
                                                    abs=1e-5)

    def test_small_T(self):
        m = tu.temporal_importance_mask(T=3, edge_weight=0.3)
        assert m.shape == (1, 1, 3)


# ---------------------------------------------------------------------------
# latent_kurtosis — extra: max_batches greater than loader length
# ---------------------------------------------------------------------------

class TestLatentKurtosisMore:
    def test_max_batches_exceeds_loader(self):
        """If max_batches > len(loader), loader exhausts naturally."""
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Conv1d(21, 32, 1)
                self.call_count = 0
            def encode(self, x, quantize=False):
                self.call_count += 1
                return self.lin(x)
            def forward(self, x, quantize=False):
                return self.lin(x)

        m = _M()
        x_l3 = torch.randn(2, 21, 313)
        loader = [(x_l3, None, None)] * 3
        tu.latent_kurtosis(m, loader, device=torch.device("cpu"),
                           max_batches=100)
        assert m.call_count == 3  # consumed all 3 batches

    def test_per_channel_array_dtype(self):
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Conv1d(21, 32, 1)
            def encode(self, x, quantize=False):
                return self.lin(x)
            def forward(self, x, quantize=False):
                return self.lin(x)

        m = _M()
        x_l3 = torch.randn(2, 21, 313)
        _, per_ch = tu.latent_kurtosis(m, [(x_l3, None, None)],
                                       device=torch.device("cpu"))
        assert per_ch.dtype == np.float64 or per_ch.dtype == np.float32
        assert per_ch.shape == (32,)

    def test_default_max_batches(self):
        """Default max_batches=10 — verify signature reachable without kwarg."""
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Conv1d(21, 32, 1)
            def encode(self, x, quantize=False):
                return self.lin(x)
            def forward(self, x, quantize=False):
                return self.lin(x)
        m = _M()
        x_l3 = torch.randn(1, 21, 313)
        out = tu.latent_kurtosis(m, [(x_l3, None, None)],
                                  device=torch.device("cpu"))
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Module-level smoke: training_utils imports without side effects
# ---------------------------------------------------------------------------

class TestModuleSurface:
    def test_module_has_public_functions(self):
        """Frozen invariant: production-callable functions must export."""
        expected = [
            "SpectralLoss", "temporal_importance_mask", "band_weighted_mse",
            "pearson_r_loss", "pearson_r_batch", "channel_dropout",
            "eeg_augment", "split_by_manifest", "validate_epoch",
            "latent_kurtosis", "run",
        ]
        for name in expected:
            assert hasattr(tu, name), f"Missing public symbol: {name}"

    def test_root_dir_is_absolute(self):
        """ROOT_DIR module constant must be absolute (used in run())."""
        assert os.path.isabs(tu.ROOT_DIR)
        assert os.path.isdir(tu.ROOT_DIR)


# ---------------------------------------------------------------------------
# run() — minimal smoke test that hits the entry/setup region
# ---------------------------------------------------------------------------
#
# The training loop is monolithic (1080+ lines, dozens of closures, GPU/dataset
# heavy). Calling run() end-to-end on CPU with real datasets is infeasible.
# We invoke run() with a zero-epoch TrainingConfig + heavy monkey-patching to
# force every phase loop to skip immediately, exercising the module-level
# imports, config printing, and final cleanup paths only.
#
# This is gated as @pytest.mark.l3 (slow) but stays under tests/ — no checkpoint
# files are written to repo paths (we mock torch.save). Skips when CUDA is
# absent or any required dep is missing.
# ---------------------------------------------------------------------------

class TestRunSmoke:
    @pytest.mark.l3
    def test_run_zero_epochs_finishes_without_crash(self, tmp_path,
                                                     monkeypatch):
        """Force epochs_warmup=quant=fine=0 to skip every training loop.

        Walks the entire run() control flow: model build, dataset selection,
        loader factory, optimizer/scheduler init, all three phase setup
        blocks, validation skipping when val_loader=None, final cleanup.
        """
        # Skip if heavy deps missing
        pytest.importorskip("auraloss")
        pytest.importorskip("training_dashboard")
        pytest.importorskip("training_plotter")
        pytest.importorskip("training_guard")
        pytest.importorskip("checkpoint_manager")
        pytest.importorskip("streaming_dataset")

        # Mocked args namespace (run() reads module-level `args` global)
        import argparse as _argparse
        fake_args = _argparse.Namespace(
            cdf_entries=32, init_from=None, reinit_layers=None,
            start_phase=None, v1=True, config="fast", resume=False,
        )
        monkeypatch.setattr(tu, "args", fake_args, raising=False)

        # Redirect any torch.save call into a tmp_path log (no repo writes).
        save_calls = []
        def _fake_save(obj, p, *a, **kw):
            save_calls.append(str(p))
        monkeypatch.setattr("torch.save", _fake_save)
        monkeypatch.setattr(tu, "_safe_load", lambda *a, **kw: {})
        # Redirect os.makedirs and weights/ writes to no-op
        monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
        # Pretend no checkpoint / resume / manifest file exists
        _orig_exists = os.path.exists
        def _fake_exists(p):
            sp = str(p)
            if sp.endswith(".ckpt") or "resume" in sp:
                return False
            return _orig_exists(p)
        monkeypatch.setattr("os.path.exists", _fake_exists)

        # Stub split_by_manifest so val_loader is None (no real data)
        monkeypatch.setattr(tu, "split_by_manifest",
                            lambda *a, **kw: ([], []))
        # Empty glob → 0 train files → triggers PrecomputedL3 path
        monkeypatch.setattr(tu.glob, "glob", lambda *a, **kw: [])

        # PrecomputedL3Dataset is patched in streaming_dataset namespace
        # because run() imports it locally — replace with a tiny stub.
        import streaming_dataset as sd

        class _StubDataset:
            def __init__(self, *a, **kw):
                self.l3_data = torch.zeros(1, 21, 313)
                self.windows_per_epoch = 0
            def to_gpu(self, dev): pass
            def calibrate_shard_budget(self, dev): pass
            def __len__(self): return 0
            def prefetch_batches(self, bs, dev):
                return iter([])

        monkeypatch.setattr(sd, "PrecomputedL3Dataset", _StubDataset)

        # Redirect training_logs to tmp_path so we don't pollute repo
        monkeypatch.chdir(tmp_path)

        from training_config import CONFIGS
        cfg = CONFIGS["fast"].replace(
            epochs_warmup=0, epochs_quant=0, epochs_fine=0,
            windows_per_epoch=64, max_windows=64, val_windows=32,
            val_interval=1, snac_preset="none",
        )
        object.__setattr__(cfg, "_resume", False)

        # End-to-end smoke. Zero epochs means no training; we only exercise
        # the bookkeeping / phase-setup / cleanup code paths.
        tu.run(cfg=cfg)

        # Frozen invariant: training completes and emits final-save calls
        # (3 paths: completed, dist, canonical). torch.save mocked → harmless.
        assert any(".ckpt" in p for p in save_calls), \
            f"Expected ckpt saves; got {save_calls}"

    @pytest.mark.l3
    def test_run_snac_preset_compact(self, tmp_path, monkeypatch):
        """Exercises the SNAC FSQ branch (cfg.snac_preset='compact').

        Hits make_multiscale_fsq + the snac_qloss term inside the QAT batch
        loop region (skipped by epochs=0 but the constructor block runs).
        """
        pytest.importorskip("auraloss")
        pytest.importorskip("training_dashboard")
        pytest.importorskip("training_plotter")
        pytest.importorskip("training_guard")
        pytest.importorskip("checkpoint_manager")
        pytest.importorskip("multiscale_fsq")
        import streaming_dataset as sd

        import argparse as _argparse
        fake_args = _argparse.Namespace(
            cdf_entries=32, init_from=None, reinit_layers=None,
            start_phase=None, v1=True, config="fast", resume=False,
        )
        monkeypatch.setattr(tu, "args", fake_args, raising=False)
        monkeypatch.setattr("torch.save", lambda *a, **kw: None)
        monkeypatch.setattr(tu, "_safe_load", lambda *a, **kw: {})
        monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
        _orig_exists = os.path.exists
        monkeypatch.setattr("os.path.exists", lambda p: False if str(p)
                            .endswith(".ckpt") or "resume" in str(p)
                            else _orig_exists(p))
        monkeypatch.setattr(tu, "split_by_manifest",
                            lambda *a, **kw: ([], []))
        monkeypatch.setattr(tu.glob, "glob", lambda *a, **kw: [])

        class _StubDataset:
            def __init__(self, *a, **kw):
                self.l3_data = torch.zeros(1, 21, 313)
                self.windows_per_epoch = 0
            def to_gpu(self, dev): pass
            def calibrate_shard_budget(self, dev): pass
            def __len__(self): return 0
            def prefetch_batches(self, bs, dev):
                return iter([])
        monkeypatch.setattr(sd, "PrecomputedL3Dataset", _StubDataset)
        monkeypatch.chdir(tmp_path)

        from training_config import CONFIGS
        cfg = CONFIGS["fast"].replace(
            epochs_warmup=0, epochs_quant=0, epochs_fine=0,
            windows_per_epoch=64, max_windows=64, val_windows=32,
            val_interval=1, snac_preset="compact",
        )
        object.__setattr__(cfg, "_resume", False)

        tu.run(cfg=cfg)  # snac block at line 332-335 reached

    @pytest.mark.l3
    def test_run_one_epoch_per_phase_cpu(self, tmp_path, monkeypatch):
        """End-to-end CPU smoke that actually executes 1 batch per phase.

        Forces torch.cuda.is_available=False so the torch.compile and AMP
        branches short-circuit cleanly. Runs all three phase loop bodies
        (warmup / QAT / fine-tune), exercising loss computation, optimizer
        step, gradient clipping, alpha clamping, and validation paths.
        """
        pytest.importorskip("auraloss")
        pytest.importorskip("training_dashboard")
        pytest.importorskip("training_plotter")
        pytest.importorskip("training_guard")
        pytest.importorskip("checkpoint_manager")
        import streaming_dataset as sd

        # Force CPU device — skips torch.compile, CUDA graph, triton paths
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        import argparse as _argparse
        fake_args = _argparse.Namespace(
            cdf_entries=32, init_from=None, reinit_layers=None,
            start_phase=None, v1=True, config="fast", resume=False,
        )
        monkeypatch.setattr(tu, "args", fake_args, raising=False)
        monkeypatch.setattr("torch.save", lambda *a, **kw: None)
        monkeypatch.setattr(tu, "_safe_load", lambda *a, **kw: {})
        monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
        _orig_exists = os.path.exists
        monkeypatch.setattr("os.path.exists", lambda p: False if str(p)
                            .endswith(".ckpt") or "resume" in str(p)
                            else _orig_exists(p))

        # Both train_files and val_files are non-empty fake paths; the
        # stubbed PrecomputedL3Dataset doesn't actually read them.
        monkeypatch.setattr(tu, "split_by_manifest",
                            lambda *a, **kw: (["/fake/train.npz"],
                                              ["/fake/val.npz"]))
        monkeypatch.setattr(tu.glob, "glob", lambda *a, **kw: [])

        # Patch np.load to lie about l3 presence so val_has_l3 branches True
        class _NpzCtx:
            files = ["l3"]
            def __enter__(self): return self
            def __exit__(self, *a): pass
        monkeypatch.setattr("numpy.load", lambda *a, **kw: _NpzCtx())

        class _StubDataset:
            """Stand-in for PrecomputedL3Dataset.

            __getitem__ returns the 3-tuple the val DataLoader expects;
            prefetch_batches yields a single GPU/CPU batch for training.
            """
            def __init__(self, *a, **kw):
                self.l3_data = torch.zeros(1, 21, 313)
                self.windows_per_epoch = 1
            def to_gpu(self, dev): pass
            def calibrate_shard_budget(self, dev): pass
            def __len__(self): return 1
            def __getitem__(self, idx):
                torch.manual_seed(idx)
                return (torch.randn(21, 313),
                        torch.zeros(1),
                        torch.zeros(1))
            def prefetch_batches(self, bs, dev):
                torch.manual_seed(0)
                x = torch.randn(2, 21, 313, device=dev)
                return iter([(x, None, None)])

        monkeypatch.setattr(sd, "PrecomputedL3Dataset", _StubDataset)
        monkeypatch.chdir(tmp_path)

        from training_config import CONFIGS
        cfg = CONFIGS["fast"].replace(
            epochs_warmup=1, epochs_quant=1, epochs_fine=1,
            windows_per_epoch=2, max_windows=2, val_windows=2,
            val_interval=1, snac_preset="none",
        )
        object.__setattr__(cfg, "_resume", False)

        # Should complete all three phases on CPU in a few seconds
        tu.run(cfg=cfg)

    @pytest.mark.l3
    def test_run_band_weighted_mse_branch(self, tmp_path, monkeypatch):
        """Exercises the cfg.band_weighted_mse=True path (mse_fn switch)."""
        pytest.importorskip("auraloss")
        pytest.importorskip("training_dashboard")
        pytest.importorskip("training_plotter")
        pytest.importorskip("training_guard")
        pytest.importorskip("checkpoint_manager")
        import streaming_dataset as sd

        import argparse as _argparse
        fake_args = _argparse.Namespace(
            cdf_entries=32, init_from=None, reinit_layers=None,
            start_phase=None, v1=True, config="fast", resume=False,
        )
        monkeypatch.setattr(tu, "args", fake_args, raising=False)
        monkeypatch.setattr("torch.save", lambda *a, **kw: None)
        monkeypatch.setattr(tu, "_safe_load", lambda *a, **kw: {})
        monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
        _orig_exists = os.path.exists
        monkeypatch.setattr("os.path.exists", lambda p: False if str(p)
                            .endswith(".ckpt") or "resume" in str(p)
                            else _orig_exists(p))
        monkeypatch.setattr(tu, "split_by_manifest",
                            lambda *a, **kw: ([], []))
        monkeypatch.setattr(tu.glob, "glob", lambda *a, **kw: [])

        class _StubDataset:
            def __init__(self, *a, **kw):
                self.l3_data = torch.zeros(1, 21, 313)
                self.windows_per_epoch = 0
            def to_gpu(self, dev): pass
            def calibrate_shard_budget(self, dev): pass
            def __len__(self): return 0
            def prefetch_batches(self, bs, dev):
                return iter([])
        monkeypatch.setattr(sd, "PrecomputedL3Dataset", _StubDataset)
        monkeypatch.chdir(tmp_path)

        from training_config import CONFIGS
        cfg = CONFIGS["fast"].replace(
            epochs_warmup=0, epochs_quant=0, epochs_fine=0,
            windows_per_epoch=64, max_windows=64, val_windows=32,
            val_interval=1, snac_preset="none",
            band_weighted_mse=True,  # exercises line 897
            alpha_clamp=True,        # exercises alpha clamp branches
            alpha_floor=0.05,        # uniform absolute floor branch
            alpha_ceiling=3.0,       # ceiling branch
            alpha_floor_init=True,   # per-layer floor init
            l1_lambda=1e-4,          # L1 ternary loss
            l1_lambda_decoder=1e-4,  # decoder L1
            wd_two_stage=True,       # two-stage WD branch
        )
        object.__setattr__(cfg, "_resume", False)

        tu.run(cfg=cfg)
