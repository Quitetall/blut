"""Unit tests for ai_models/student/eval_fullband.py — Phase 3."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import eval_fullband as ef

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _fake_preprocess(window_uv):
    """preprocess_subband_single(window) → (l3, coeffs, subs)."""
    return np.random.randn(21, 313).astype(np.float32), None, None


def _fake_reconstruct(recon_l3, coeffs, subs):
    """reconstruct_from_subband → fullband [21, 2500]."""
    return np.random.randn(21, 2500).astype(np.float32)


class _FakeCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(313, 313)
        self.encoder = torch.nn.Linear(21, 21)
        self.decoder = torch.nn.Linear(21, 21)

    def forward(self, x, quantize=False):
        # x [B, 21, 313] → [B, 21, 313]
        return self.lin(x)


# ---------------------------------------------------------------------------
# _fullband_reconstruct
# ---------------------------------------------------------------------------
class TestFullbandReconstruct:
    def test_shape_matches_input(self):
        codec = _FakeCodec()
        window = np.random.randn(21, 2500).astype(np.float32)
        with patch.object(ef, "preprocess_subband_single",
                           side_effect=_fake_preprocess), \
             patch.object(ef, "reconstruct_from_subband",
                           side_effect=_fake_reconstruct):
            out = ef._fullband_reconstruct(window, codec, torch.device("cpu"))
        assert out.shape == (21, 2500)


# ---------------------------------------------------------------------------
# _eval_window
# ---------------------------------------------------------------------------
class TestEvalWindow:
    def test_returns_dict_with_keys(self):
        codec = _FakeCodec()
        window = np.random.randn(21, 2500).astype(np.float32)
        with patch.object(ef, "preprocess_subband_single",
                           side_effect=_fake_preprocess), \
             patch.object(ef, "reconstruct_from_subband",
                           side_effect=_fake_reconstruct):
            m = ef._eval_window(window, codec, torch.device("cpu"))
        assert set(m.keys()) == {"r", "prd", "pb_prd", "pb_r"}
        assert isinstance(m["r"], (float, np.floating))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def _make_npz(tmp_path: Path, name="f.npz", T=10000):
    p = tmp_path / name
    np.savez_compressed(
        p, data=np.random.randint(-10000, 10000, (21, T), dtype=np.int32))
    return p


class _FE:
    """File entry stub."""
    def __init__(self, path):
        self.path = path


class TestEvaluate:
    def test_no_windows_raises(self, tmp_path):
        # Empty entry list → 0 windows → raises
        with pytest.raises(RuntimeError, match="no windows"):
            ef.evaluate([], _FakeCodec(), torch.device("cpu"))

    def test_skips_short_files(self, tmp_path):
        # File too short (< 1 full window)
        p = _make_npz(tmp_path, T=500)
        with pytest.raises(RuntimeError):
            ef.evaluate([_FE(p)], _FakeCodec(), torch.device("cpu"))

    def test_skips_unloadable_files(self, tmp_path):
        # Non-existent path → np.load raises → file silently skipped
        with pytest.raises(RuntimeError):
            ef.evaluate([_FE(tmp_path / "nope.npz")], _FakeCodec(),
                        torch.device("cpu"))

    def test_returns_expected_keys(self, tmp_path):
        p = _make_npz(tmp_path)
        with patch.object(ef, "preprocess_subband_single",
                           side_effect=_fake_preprocess), \
             patch.object(ef, "reconstruct_from_subband",
                           side_effect=_fake_reconstruct):
            result = ef.evaluate([_FE(p)], _FakeCodec(), torch.device("cpu"),
                                  windows_per_file=2, max_files=1)
        expected_keys = {"n_windows", "n_files", "mean_r", "mean_prd",
                          "per_band_prd", "per_band_r",
                          "lqs_level", "lqs_violations"}
        assert expected_keys <= set(result.keys())
        assert result["n_files"] == 1
        assert result["n_windows"] >= 1

    def test_window_eval_exception_skipped(self, tmp_path, capsys):
        p = _make_npz(tmp_path)

        def _raise_on_eval(*a, **k):
            raise RuntimeError("synthetic failure")

        # Patch _eval_window to always raise → all windows skipped → raises
        with patch.object(ef, "_eval_window", side_effect=_raise_on_eval):
            with pytest.raises(RuntimeError, match="no windows"):
                ef.evaluate([_FE(p)], _FakeCodec(), torch.device("cpu"),
                            windows_per_file=2, max_files=1)
        out = capsys.readouterr().out
        assert "[skip]" in out


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------
class TestPrintReport:
    def test_prints_expected_sections(self, capsys):
        result = {
            "n_windows": 100, "n_files": 10,
            "mean_r": 0.85, "mean_prd": 22.0,
            "per_band_prd": {"delta": 5.0, "theta": 10.0, "alpha": 15.0,
                              "beta": 20.0, "gamma": 25.0},
            "per_band_r":   {"delta": 0.9, "theta": 0.85, "alpha": 0.8,
                              "beta": 0.7, "gamma": 0.5},
            "lqs_level": "M", "lqs_violations": ["alpha PRD too high"],
        }
        ef.print_report(result, "/x/enc.ckpt", "/x/dec.ckpt", duration_s=12.5)
        out = capsys.readouterr().out
        assert "FULLBAND LQS EVALUATION" in out
        assert "Monitoring" in out
        assert "alpha PRD too high" in out

    def test_no_violations_branch(self, capsys):
        result = {
            "n_windows": 1, "n_files": 1, "mean_r": 0.99, "mean_prd": 1.0,
            "per_band_prd": {b: 1.0 for b in
                              ("delta", "theta", "alpha", "beta", "gamma")},
            "per_band_r":   {b: 0.99 for b in
                              ("delta", "theta", "alpha", "beta", "gamma")},
            "lqs_level": "C", "lqs_violations": [],
        }
        ef.print_report(result, "e.ckpt", "d.ckpt", duration_s=1.0)
        out = capsys.readouterr().out
        assert "No violations" in out

    def test_many_violations_truncated(self, capsys):
        result = {
            "n_windows": 1, "n_files": 1, "mean_r": 0.5, "mean_prd": 50.0,
            "per_band_prd": {b: 30.0 for b in
                              ("delta", "theta", "alpha", "beta", "gamma")},
            "per_band_r":   {b: 0.5 for b in
                              ("delta", "theta", "alpha", "beta", "gamma")},
            "lqs_level": "", "lqs_violations": [f"v{i}" for i in range(15)],
        }
        ef.print_report(result, "e", "d", duration_s=1.0)
        out = capsys.readouterr().out
        assert "and 7 more" in out


# ---------------------------------------------------------------------------
# main — stub manifest + codec
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_main(monkeypatch, tmp_path):
    """Make main() runnable with synthetic data."""
    # Build fake DatasetManifest in module
    class _Manifest:
        @classmethod
        def load(cls, _p): return cls()
        def get_file_entries(self, split, datasets=None):
            return [_FE(_make_npz(tmp_path))]

    fake_dt = types.ModuleType("data_types")
    fake_dt.DatasetManifest = _Manifest
    fake_dt.Split = MagicMock()
    fake_dt.Dataset = lambda n: n
    monkeypatch.setitem(sys.modules, "data_types", fake_dt)

    # Patch internal imports inside ef
    monkeypatch.setattr(ef, "DatasetManifest", _Manifest)
    monkeypatch.setattr(ef, "Split", MagicMock())

    # Stub codec builder
    def _build(vocos_tier=2):
        return _FakeCodec()
    monkeypatch.setattr(ef, "build_default_joint", _build)
    monkeypatch.setattr(ef, "_safe_load",
                        lambda p, map_location=None: {"state_dict": {}})

    monkeypatch.setattr(ef, "preprocess_subband_single", _fake_preprocess)
    monkeypatch.setattr(ef, "reconstruct_from_subband", _fake_reconstruct)
    yield


class TestMain:
    def test_main_runs_basic(self, stub_main, tmp_path, monkeypatch):
        enc = tmp_path / "enc.ckpt"; enc.touch()
        dec = tmp_path / "dec.ckpt"; dec.touch()
        monkeypatch.setattr(sys, "argv",
                            ["ef", "--encoder", str(enc),
                             "--decoder", str(dec),
                             "--max-files", "1", "--windows-per-file", "2"])
        assert ef.main() == 0

    def test_main_with_json(self, stub_main, tmp_path, monkeypatch, capsys):
        enc = tmp_path / "enc.ckpt"; enc.touch()
        dec = tmp_path / "dec.ckpt"; dec.touch()
        monkeypatch.setattr(sys, "argv",
                            ["ef", "--encoder", str(enc),
                             "--decoder", str(dec),
                             "--max-files", "1", "--windows-per-file", "2",
                             "--json"])
        assert ef.main() == 0
        out = capsys.readouterr().out
        assert "__PCCP_JSON__" in out

    def test_main_with_datasets_filter(self, stub_main, tmp_path, monkeypatch):
        enc = tmp_path / "enc.ckpt"; enc.touch()
        dec = tmp_path / "dec.ckpt"; dec.touch()
        monkeypatch.setattr(sys, "argv",
                            ["ef", "--encoder", str(enc),
                             "--decoder", str(dec),
                             "--max-files", "1", "--windows-per-file", "2",
                             "--datasets", "chbmit"])
        assert ef.main() == 0

    def test_main_no_entries_returns_1(self, stub_main, tmp_path, monkeypatch):
        # Override manifest to return empty
        class _Empty:
            @classmethod
            def load(cls, _p): return cls()
            def get_file_entries(self, split, datasets=None): return []

        monkeypatch.setattr(ef, "DatasetManifest", _Empty)
        enc = tmp_path / "enc.ckpt"; enc.touch()
        dec = tmp_path / "dec.ckpt"; dec.touch()
        monkeypatch.setattr(sys, "argv",
                            ["ef", "--encoder", str(enc), "--decoder", str(dec)])
        assert ef.main() == 1
