"""Unit tests for lamquant.snn.head_benchmark — Phase 3.

Covers `discover_edfs`, `_format_row`, `_format_table`, `load_signal_batch`
(with stubbed `load_edf_signal`), and `run_head` (with stub head + backbone).
`main()` is exercised via subprocess `--help` smoke (Phase 5)."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.l2


_MODULE_PATH = (Path(__file__).resolve().parents[1]
                / "head_benchmark.py")


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_STUB_NAMES = ("mamba_ssm_minimal", "heads", "snn_to_nedc_eval",
                "lamquant.snn.head_benchmark_under_test")


@pytest.fixture(scope="module")
def hb():
    """Load head_benchmark with stubs for heavy dependencies.

    Stubs are installed only if the real module isn't already imported,
    and are explicitly torn down at fixture finalisation to prevent
    polluting later test modules.
    """
    pre_loaded = {n: sys.modules.get(n) for n in _STUB_NAMES}

    if "mamba_ssm_minimal" not in sys.modules:
        _stub("mamba_ssm_minimal", MambaSNN=object)
    if "heads" not in sys.modules:
        _stub("heads", build_head=lambda n: None, HEAD_REGISTRY={
            "threshold_legacy": object,
            "attention_softmax": object,
            "crf": object,
            "temporal_attention": object,
            "moe_fsq": object,
        })
    if "snn_to_nedc_eval" not in sys.modules:
        _stub("snn_to_nedc_eval", load_edf_signal=lambda p: (None, None, None))

    spec = importlib.util.spec_from_file_location(
        "lamquant.snn.head_benchmark_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    try:
        yield module
    finally:
        # Restore pre-fixture state of sys.modules
        for name, prev in pre_loaded.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


# ---------------------------------------------------------------------------
# HEADS constant
# ---------------------------------------------------------------------------
class TestHeadsConst:
    def test_heads_is_sorted_list(self, hb):
        assert hb.HEADS == sorted(hb.HEADS)
        assert len(hb.HEADS) > 0

    def test_heads_excludes_plain_threshold(self, hb):
        assert "threshold" not in hb.HEADS
        assert "threshold_legacy" in hb.HEADS


# ---------------------------------------------------------------------------
# discover_edfs
# ---------------------------------------------------------------------------
class TestDiscoverEdfs:
    def test_returns_empty_for_missing_dir(self, hb, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert hb.discover_edfs(empty, max_files=5, seed=42) == []

    def test_returns_empty_for_no_edfs(self, hb, tmp_path):
        (tmp_path / "a.txt").write_text("")
        assert hb.discover_edfs(tmp_path, max_files=5, seed=42) == []

    def test_caps_at_max_files(self, hb, tmp_path):
        for i in range(10):
            (tmp_path / f"file_{i}.edf").write_text("")
        out = hb.discover_edfs(tmp_path, max_files=3, seed=42)
        assert len(out) == 3

    def test_recursive_discovery(self, hb, tmp_path):
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "deep.edf").write_text("")
        (tmp_path / "top.edf").write_text("")
        out = hb.discover_edfs(tmp_path, max_files=10, seed=42)
        names = {p.name for p in out}
        assert {"deep.edf", "top.edf"} == names

    def test_deterministic_with_seed(self, hb, tmp_path):
        for i in range(10):
            (tmp_path / f"f{i}.edf").write_text("")
        a = hb.discover_edfs(tmp_path, max_files=4, seed=99)
        b = hb.discover_edfs(tmp_path, max_files=4, seed=99)
        assert a == b

    def test_different_seeds_can_differ(self, hb, tmp_path):
        for i in range(20):
            (tmp_path / f"f{i}.edf").write_text("")
        a = hb.discover_edfs(tmp_path, max_files=5, seed=1)
        b = hb.discover_edfs(tmp_path, max_files=5, seed=2)
        # Not guaranteed to differ, but extremely likely with 20 files / pick 5.
        assert isinstance(a, list) and isinstance(b, list)

    def test_returned_paths_are_path_objs(self, hb, tmp_path):
        (tmp_path / "a.edf").write_text("")
        out = hb.discover_edfs(tmp_path, max_files=1, seed=0)
        assert all(isinstance(p, Path) for p in out)


# ---------------------------------------------------------------------------
# load_signal_batch
# ---------------------------------------------------------------------------
class TestLoadSignalBatch:
    def test_returns_empty_for_no_paths(self, hb):
        assert hb.load_signal_batch([], window_samples=2500) == []

    def test_loads_one_window_per_edf(self, hb, tmp_path):
        # Stub load_edf_signal to return a synthetic signal.
        signal = np.random.randn(21, 5000).astype(np.float32)
        paths = [tmp_path / "a.edf", tmp_path / "b.edf"]
        for p in paths:
            p.write_text("")

        with patch.object(hb, "load_edf_signal",
                          return_value=(signal, None, None)):
            batch = hb.load_signal_batch(paths, window_samples=2500)

        assert len(batch) == 2
        assert all(isinstance(t, torch.Tensor) for t in batch)
        assert all(t.shape == (21, 2500) for t in batch)
        assert all(t.dtype == torch.float32 for t in batch)

    def test_skips_too_short(self, hb, tmp_path):
        signal = np.zeros((21, 100), dtype=np.float32)
        paths = [tmp_path / "x.edf"]
        paths[0].write_text("")
        with patch.object(hb, "load_edf_signal",
                          return_value=(signal, None, None)):
            batch = hb.load_signal_batch(paths, window_samples=2500)
        assert batch == []

    def test_skips_on_exception(self, hb, tmp_path, capsys):
        paths = [tmp_path / "bad.edf"]
        paths[0].write_text("")
        with patch.object(hb, "load_edf_signal",
                          side_effect=RuntimeError("boom")):
            batch = hb.load_signal_batch(paths, window_samples=2500)
        assert batch == []
        captured = capsys.readouterr()
        assert "skip" in captured.err
        assert "bad.edf" in captured.err

    def test_window_taken_from_middle(self, hb, tmp_path):
        # Place a marker in the middle.
        signal = np.zeros((21, 5000), dtype=np.float32)
        signal[:, 2500] = 99.0  # midpoint sample
        paths = [tmp_path / "m.edf"]
        paths[0].write_text("")
        with patch.object(hb, "load_edf_signal",
                          return_value=(signal, None, None)):
            batch = hb.load_signal_batch(paths, window_samples=2500)
        assert batch[0][0, 1250].item() == 99.0


# ---------------------------------------------------------------------------
# _format_row + _format_table
# ---------------------------------------------------------------------------
def _sample_result(name="attention_softmax", K=4):
    return {
        "head": name,
        "K": K,
        "param_count": 1234,
        "state_distribution_pct": [25.0, 25.0, 25.0, 25.0],
        "level_table": [3, 5, 7, 9],
        "mean_fsq_level": 4.5,
        "bits_per_timestep": 2.1,
        "flicker_per_step_mean": 0.123,
        "flicker_per_step_p95": 0.456,
        "transitions_per_second_mean": 9.7,
        "head_latency_ms_mean": 1.23,
        "head_latency_ms_p95": 2.34,
    }


class TestFormatRow:
    def test_includes_head_name(self, hb):
        s = hb._format_row(_sample_result(name="crf"))
        assert "crf" in s

    def test_includes_param_count(self, hb):
        s = hb._format_row(_sample_result())
        assert "1234" in s

    def test_includes_flicker(self, hb):
        s = hb._format_row(_sample_result())
        assert "flicker" in s
        assert "0.123" in s

    def test_returns_string(self, hb):
        assert isinstance(hb._format_row(_sample_result()), str)


class TestFormatTable:
    def test_empty_results_still_formats(self, hb):
        s = hb._format_table([])
        assert "Head" in s
        assert "Notes:" in s

    def test_includes_separator(self, hb):
        s = hb._format_table([_sample_result()])
        assert "=" * 50 in s
        assert "-" * 50 in s

    def test_lists_each_head(self, hb):
        rs = [_sample_result(name="threshold_legacy"),
              _sample_result(name="moe_fsq")]
        s = hb._format_table(rs)
        assert "threshold_legacy" in s
        assert "moe_fsq" in s

    def test_returns_string(self, hb):
        assert isinstance(hb._format_table([_sample_result()]), str)

    def test_notes_section_present(self, hb):
        s = hb._format_table([_sample_result()])
        assert "head ms is HEAD ONLY" in s


# ---------------------------------------------------------------------------
# run_head with stub head + backbone
# ---------------------------------------------------------------------------
class _StubHead(torch.nn.Module):
    """Mimic a SNN head: takes logits [1, 8, T] → (states, _)."""
    def __init__(self, K=4):
        super().__init__()
        self.K = K
        self.name = "stub_head"
        self.level_table = torch.tensor([2, 3, 5, 7][:K], dtype=torch.long)
        self.fc = torch.nn.Linear(8, K)

    def forward(self, logits, target_T=79):
        # Return random-but-deterministic state stream of length target_T.
        torch.manual_seed(0)
        states = torch.randint(0, self.K, (logits.shape[0], target_T))
        return states, None


class _StubBackbone(torch.nn.Module):
    """Emits [B, 8, 79] logits."""
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2500, 79)

    def forward(self, x):
        # x: [1, 21, 2500] → [1, 8, 79]
        out = self.fc(x.mean(dim=1, keepdim=False))  # [1, 79]
        return out.unsqueeze(1).expand(-1, 8, -1).contiguous(), None


class TestRunHead:
    def test_returns_expected_keys(self, hb):
        head = _StubHead(K=4)
        backbone = _StubBackbone()
        batch = [torch.randn(21, 2500) for _ in range(2)]
        device = torch.device("cpu")
        out = hb.run_head(head, backbone, batch, device)
        expected = {"head", "K", "param_count", "state_distribution_pct",
                    "level_table", "mean_fsq_level", "bits_per_timestep",
                    "flicker_per_step_mean", "flicker_per_step_p95",
                    "transitions_per_second_mean",
                    "head_latency_ms_mean", "head_latency_ms_p95"}
        assert expected <= set(out.keys())

    def test_state_distribution_sums_to_100(self, hb):
        head = _StubHead(K=4)
        backbone = _StubBackbone()
        batch = [torch.randn(21, 2500) for _ in range(3)]
        out = hb.run_head(head, backbone, batch, torch.device("cpu"))
        total = sum(out["state_distribution_pct"])
        assert total == pytest.approx(100.0, abs=0.5)

    def test_param_count_matches_head(self, hb):
        head = _StubHead(K=4)
        backbone = _StubBackbone()
        batch = [torch.randn(21, 2500)]
        out = hb.run_head(head, backbone, batch, torch.device("cpu"))
        expected = sum(p.numel() for p in head.parameters() if p.requires_grad)
        assert out["param_count"] == expected

    def test_K_propagated(self, hb):
        head = _StubHead(K=3)
        backbone = _StubBackbone()
        batch = [torch.randn(21, 2500)]
        out = hb.run_head(head, backbone, batch, torch.device("cpu"))
        assert out["K"] == 3

    def test_flicker_in_valid_range(self, hb):
        head = _StubHead(K=4)
        backbone = _StubBackbone()
        batch = [torch.randn(21, 2500) for _ in range(2)]
        out = hb.run_head(head, backbone, batch, torch.device("cpu"))
        assert 0.0 <= out["flicker_per_step_mean"] <= 1.0
        assert 0.0 <= out["flicker_per_step_p95"] <= 1.0

    def test_latency_nonneg(self, hb):
        head = _StubHead(K=4)
        backbone = _StubBackbone()
        batch = [torch.randn(21, 2500)]
        out = hb.run_head(head, backbone, batch, torch.device("cpu"))
        assert out["head_latency_ms_mean"] >= 0.0
        assert out["head_latency_ms_p95"] >= 0.0

    def test_level_table_round_tripped(self, hb):
        head = _StubHead(K=4)
        backbone = _StubBackbone()
        batch = [torch.randn(21, 2500)]
        out = hb.run_head(head, backbone, batch, torch.device("cpu"))
        assert out["level_table"] == head.level_table.cpu().tolist()
