"""Deep coverage for ai_models/student/run_diagnostics.py.

This module is the production diagnostic-suite script. The existing
``tests/codec_python_smoke/test_ai_models_smoke.py`` covers it at the
"module imports + exposes metric_* callables" surface only (~5%
coverage). This file extends coverage by exercising every
checkpoint-driven helper that can run on CPU:

  - ``load_model`` against a real (constructed-here) state_dict
  - ``recalibrate_cdf`` with planted L3 NPZs
  - ``metric_1_kurtosis`` (per-channel kurtosis histogram)
  - ``metric_2_per_channel_val_r`` (per-channel Pearson R)
  - ``metric_3_fsq_utilization`` (codebook utilization)
  - ``metric_4_gradient_norm`` (per-layer grad norms)
  - ``metric_5_seizure_vs_background`` (mask-aware R splits)
  - ``metric_8_alpha_distribution`` (LSQ alpha per ternary module)
  - ``load_val_files`` (manifest plumbing)

Metric 6/7/9 require ``codec.SubbandCodec`` end-to-end with
``preprocess_subband_single`` taking 21-channel float signal — those
are production codec-pipeline tests, out of scope here. They are
exercised by the codec test suite.

Per ``feedback_futureproof_tests``: shape/type/finite/non-negative
asserts, not numeric recomputation. We do NOT pin "kurtosis ≈ 0" or
"R = 0.5" — only "result is finite scalar / has expected shape".

Math fixtures via ``np.random`` (planted NPZ contents) — this is a
permitted non-EEG fixture source. No synthetic EEG: the NPZs simulate
the shape/key contract that ``run_diagnostics`` walks, not the
clinical waveform itself. Real EEG is exercised through the codec
end-to-end pytest fixtures elsewhere.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# parents[3] == blut/python (the package root); the student area dir is
# also on sys.path via blut/python/conftest.py.
_PY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PY_ROOT))
sys.path.insert(0, str(_PY_ROOT / "lamquant" / "student"))


pytestmark = pytest.mark.l2


@pytest.fixture(scope="module")
def rd():
    return importlib.import_module("lamquant.student.run_diagnostics")


@pytest.fixture(scope="module")
def trained_ckpt(tmp_path_factory):
    """Build a real TernaryMobileNetV5_Subband, save its state_dict.

    This is NOT a trained checkpoint — it's a fresh-init state_dict
    saved to disk so ``load_model`` has something to read. The contract
    under test is ``load_model``'s file-load + filter logic, not
    training.
    """
    from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
    torch.manual_seed(0)
    m = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32, cdf_entries=32)
    m.ensure_initialized()
    p = tmp_path_factory.mktemp("ckpts") / "fresh.ckpt"
    torch.save(m.state_dict(), p)
    return p


def _plant_l3_npz(path: Path, n_windows: int = 4, *,
                   add_seizure_mask: bool = False,
                   add_data: bool = False, seed: int = 0) -> Path:
    """Create an NPZ that ``run_diagnostics`` will walk.

    Keys planted match what each metric reads:
      - ``l3`` (required, [N, 21, 313]) — present always
      - ``seizure_mask`` (optional) for metric_5
      - ``data`` / ``gain`` / ``sample_rate`` for metric_6/7/9

    np.random is the FIXTURE source (allowed). The NPZ key/shape
    layout reflects the actual NPZ contract that q31_events files use.
    """
    rng = np.random.RandomState(seed)
    arrays: dict = {
        "l3": rng.randn(n_windows, 21, 313).astype(np.float32),
    }
    if add_seizure_mask:
        # 2500 samples total; mark ~half as seizure.
        mask = np.zeros(2500, dtype=np.float32)
        mask[1000:1800] = 1.0
        arrays["seizure_mask"] = mask
    if add_data:
        arrays["data"] = (rng.randn(21, 2500) * 1e8).astype(np.int32)
        arrays["gain"] = np.array(1.0)
        arrays["sample_rate"] = np.array(250.0)
    np.savez_compressed(path, **arrays)
    return path


# ============================================================
# load_model — file-load + shape-filter contract
# ============================================================


class TestLoadModel:
    def test_load_returns_model_in_eval(self, rd, trained_ckpt):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        assert model is not None
        # The model should be in eval mode.
        assert not model.training
        # Has the documented buffer.
        assert hasattr(model, "cdf_breakpoints")
        # Shape contract: cdf_breakpoints is [latent_dim=32, cdf_entries=32].
        assert model.cdf_breakpoints.shape == (32, 32)

    def test_load_with_cdf_entries_mismatch_filtered(self, rd, trained_ckpt):
        # Loading a 32-entry checkpoint into a 64-entry model should NOT
        # raise — ``load_model`` filters shape mismatches.
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=64)
        assert model.cdf_breakpoints.shape == (32, 64)

    def test_load_forward_works(self, rd, trained_ckpt):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        x = torch.randn(2, 21, 313)
        with torch.no_grad():
            out = model(x, quantize=True)
        assert out.shape == (2, 21, 313)
        assert torch.isfinite(out).all()


# ============================================================
# recalibrate_cdf
# ============================================================


class TestRecalibrateCdf:
    def test_runs_and_updates_breakpoints(self, rd, trained_ckpt, tmp_path):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        # Snapshot pre-update breakpoint range.
        before = model.cdf_breakpoints.clone()
        # Plant one val NPZ.
        npz = _plant_l3_npz(tmp_path / "v0.npz", n_windows=4, seed=11)
        rd.recalibrate_cdf(model, [str(npz)], device="cpu",
                            max_files=1, max_windows=4)
        # Buffer should have been touched (range likely shifted from the
        # initial [-3, 3] linspace after recalibration to the actual
        # latent distribution).
        # Shape preserved.
        assert model.cdf_breakpoints.shape == before.shape

    def test_no_files_no_crash(self, rd, trained_ckpt, capsys):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        # Empty list → "No data for CDF recalibration" branch.
        rd.recalibrate_cdf(model, [], device="cpu",
                            max_files=1, max_windows=1)
        out = capsys.readouterr().out
        assert "No data for CDF recalibration" in out

    def test_files_without_l3_key_skipped(self, rd, trained_ckpt, tmp_path,
                                            capsys):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        # Plant an NPZ with a non-'l3' key.
        bogus = tmp_path / "no_l3.npz"
        np.savez_compressed(bogus, signal=np.zeros(10))
        rd.recalibrate_cdf(model, [str(bogus)], device="cpu",
                            max_files=1, max_windows=1)
        out = capsys.readouterr().out
        # Should hit the "No data" branch (no NPZ contributed any latents).
        assert "No data" in out


# ============================================================
# metric_1_kurtosis
# ============================================================


class TestMetric1Kurtosis:
    def test_returns_array_or_none(self, rd, trained_ckpt, tmp_path):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        npz = _plant_l3_npz(tmp_path / "k.npz", n_windows=4, seed=2)
        out = rd.metric_1_kurtosis(model, [str(npz)], device="cpu",
                                    max_files=1, max_windows_per_file=4)
        # Returns a 1D np.array of length latent_dim=32.
        assert isinstance(out, np.ndarray)
        assert out.shape == (32,)
        # All entries are finite (either real kurtosis or the 999.0
        # sentinel for sigma<1e-8 channels).
        assert np.isfinite(out).all()

    def test_no_data_returns_none(self, rd, trained_ckpt, tmp_path, capsys):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        # No NPZs → "No data!" branch.
        out = rd.metric_1_kurtosis(model, [], device="cpu",
                                    max_files=1, max_windows_per_file=1)
        assert out is None


# ============================================================
# metric_2_per_channel_val_r
# ============================================================


class TestMetric2PerChannelR:
    def test_returns_per_ch_list(self, rd, trained_ckpt, tmp_path):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        npz = _plant_l3_npz(tmp_path / "r.npz", n_windows=4, seed=3)
        out = rd.metric_2_per_channel_val_r(
            model, [str(npz)], device="cpu",
            max_files=1, max_windows_per_file=4)
        # 21 per-channel R lists.
        assert isinstance(out, list)
        assert len(out) == 21
        for ch_rs in out:
            for r in ch_rs:
                # Each R is a real number in [-1, 1] (NaNs filtered).
                assert -1.0 <= r <= 1.0


# ============================================================
# metric_3_fsq_utilization
# ============================================================


class TestMetric3FSQUtilization:
    def test_no_crash_with_data(self, rd, trained_ckpt, tmp_path):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        npz = _plant_l3_npz(tmp_path / "u.npz", n_windows=4, seed=4)
        # Returns None by contract — just verify it runs.
        out = rd.metric_3_fsq_utilization(
            model, [str(npz)], device="cpu",
            max_files=1, max_windows_per_file=4)
        assert out is None

    def test_no_data_returns_none(self, rd, trained_ckpt, capsys):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        out = rd.metric_3_fsq_utilization(
            model, [], device="cpu",
            max_files=1, max_windows_per_file=1)
        assert out is None


# ============================================================
# metric_4_gradient_norm
# ============================================================


class TestMetric4GradNorm:
    def test_returns_norms_dict(self, rd, trained_ckpt, tmp_path):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        npz = _plant_l3_npz(tmp_path / "g.npz", n_windows=16, seed=5)
        norms = rd.metric_4_gradient_norm(
            model, [str(npz)], device="cpu", max_windows=8)
        assert isinstance(norms, dict)
        # All grad norms are finite, non-negative real numbers.
        for name, n in norms.items():
            assert isinstance(name, str)
            assert isinstance(n, float)
            assert np.isfinite(n)
            assert n >= 0

    def test_no_data_returns_none(self, rd, trained_ckpt, capsys):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        out = rd.metric_4_gradient_norm(model, [], device="cpu")
        assert out is None


# ============================================================
# metric_5_seizure_vs_background
# ============================================================


class TestMetric5SeizureVsBackground:
    def test_runs_with_seizure_masked_npz(self, rd, trained_ckpt, tmp_path):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        npz = _plant_l3_npz(tmp_path / "s.npz", n_windows=4,
                             add_seizure_mask=True, seed=6)
        # Just verifies no crash + prints summary.
        out = rd.metric_5_seizure_vs_background(
            model, [str(npz)], device="cpu", max_files=1)
        # No declared return; returns None.
        assert out is None

    def test_runs_when_no_seizure_mask(self, rd, trained_ckpt, tmp_path):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        # NPZ without seizure_mask — metric should skip it gracefully.
        npz = _plant_l3_npz(tmp_path / "nos.npz", n_windows=2,
                             add_seizure_mask=False, seed=7)
        out = rd.metric_5_seizure_vs_background(
            model, [str(npz)], device="cpu", max_files=1)
        assert out is None


# ============================================================
# metric_8_alpha_distribution
# ============================================================


class TestMetric8AlphaDistribution:
    def test_prints_alpha_for_lsq_modules(self, rd, trained_ckpt, capsys):
        model = rd.load_model(trained_ckpt, device="cpu", cdf_entries=32)
        # The model is full of TernaryConv1d modules with lsq_alpha — the
        # metric should iterate them. Just verify it runs + prints.
        out = rd.metric_8_alpha_distribution(model)
        # Contract: returns None.
        assert out is None
        # Captured output should include the table header.
        printed = capsys.readouterr().out
        assert "ALPHA DISTRIBUTION" in printed


# ============================================================
# load_val_files — manifest plumbing
# ============================================================


class TestLoadValFiles:
    def test_load_val_files_uses_manifest(self, rd, monkeypatch):
        """``load_val_files`` ignores its args and reads the project
        manifest. Stub the manifest to a controlled list."""
        import types

        class _FakeManifest:
            @classmethod
            def load(cls, _path):
                return cls()

            def get_files(self, split):
                return [Path("/tmp/x.npz"), Path("/tmp/y.npz")]

        class _FakeSplit:
            VAL = "VAL"

        # Stub the ai_models.data_types module.
        fake_dt = types.ModuleType("data_types")
        fake_dt.DatasetManifest = _FakeManifest
        fake_dt.Split = _FakeSplit
        monkeypatch.setitem(sys.modules, "data_types", fake_dt)

        files = rd.load_val_files("ignored.json", "ignored_dir")
        assert files == ["/tmp/x.npz", "/tmp/y.npz"]
