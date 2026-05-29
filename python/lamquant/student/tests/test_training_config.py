"""Unit tests for ai_models/student/training_config.py — Phase 1.

Frozen TrainingConfig dataclass + CONFIGS preset dict + round-trip
helpers (from_dict, to_dict, from_yaml, to_yaml, hash, diff).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training_config import CONFIGS, TrainingConfig

pytestmark = pytest.mark.l1


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------
class TestTrainingConfig:
    def test_default_construction(self):
        c = TrainingConfig()
        assert c.name == "custom"
        assert c.epochs_warmup > 0
        assert c.epochs_quant > 0

    def test_frozen(self):
        c = TrainingConfig()
        with pytest.raises(Exception):
            c.batch_size = 999

    def test_total_epochs_sums(self):
        c = TrainingConfig(epochs_warmup=10, epochs_quant=20, epochs_fine=30)
        assert c.total_epochs == 60

    def test_replace_returns_new(self):
        a = TrainingConfig(name="a")
        b = a.replace(name="b")
        assert a.name == "a"
        assert b.name == "b"
        assert a is not b

    def test_to_dict_has_all_fields(self):
        c = TrainingConfig(name="x")
        d = c.to_dict()
        assert d["name"] == "x"
        assert "epochs_warmup" in d
        assert "batch_size" in d

    def test_str_contains_name(self):
        c = TrainingConfig(name="myrun", description="d")
        s = str(c)
        assert "myrun" in s
        assert "d" in s


# ---------------------------------------------------------------------------
# from_dict round-trip
# ---------------------------------------------------------------------------
class TestFromDict:
    def test_round_trip(self):
        a = TrainingConfig(name="r1", epochs_warmup=5)
        b = TrainingConfig.from_dict(a.to_dict())
        assert a == b

    def test_extra_keys_ignored(self, capsys):
        d = TrainingConfig().to_dict()
        d["bogus_key"] = "ignored"
        c = TrainingConfig.from_dict(d)
        assert isinstance(c, TrainingConfig)
        out = capsys.readouterr().out
        assert "ignoring unknown keys" in out

    def test_missing_keys_use_defaults(self):
        # Provide only name; everything else defaults.
        c = TrainingConfig.from_dict({"name": "partial"})
        assert c.name == "partial"
        assert c.epochs_warmup == TrainingConfig().epochs_warmup

    def test_list_spectral_fft_coerced_to_tuple(self):
        d = TrainingConfig().to_dict()
        d["spectral_fft_sizes"] = [8, 16, 32]
        c = TrainingConfig.from_dict(d)
        assert c.spectral_fft_sizes == (8, 16, 32)


# ---------------------------------------------------------------------------
# hash + diff
# ---------------------------------------------------------------------------
class TestHashDiff:
    def test_hash_deterministic(self):
        a = TrainingConfig(name="x")
        b = TrainingConfig(name="x")
        assert a.hash() == b.hash()

    def test_hash_includes_sha256_prefix(self):
        c = TrainingConfig()
        assert c.hash().startswith("sha256:")
        assert len(c.hash()) == 7 + 64

    def test_hash_differs_for_different_configs(self):
        a = TrainingConfig(name="a")
        b = TrainingConfig(name="b")
        assert a.hash() != b.hash()

    def test_diff_identical_is_empty(self):
        a = TrainingConfig(name="x")
        b = TrainingConfig(name="x")
        assert a.diff(b) == {}

    def test_diff_returns_field_tuple(self):
        a = TrainingConfig(name="a", batch_size=32)
        b = TrainingConfig(name="a", batch_size=64)
        d = a.diff(b)
        assert d == {"batch_size": (32, 64)}


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------
class TestYamlRoundTrip:
    def test_to_yaml_then_from_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        a = TrainingConfig(name="ytest", epochs_warmup=7)
        p = tmp_path / "cfg.yaml"
        a.to_yaml(str(p))
        b = TrainingConfig.from_yaml(str(p))
        assert b.name == "ytest"
        assert b.epochs_warmup == 7

    def test_yaml_preserves_fft_tuple(self, tmp_path):
        pytest.importorskip("yaml")
        a = TrainingConfig()
        p = tmp_path / "cfg2.yaml"
        a.to_yaml(str(p))
        b = TrainingConfig.from_yaml(str(p))
        assert isinstance(b.spectral_fft_sizes, tuple)


# ---------------------------------------------------------------------------
# CONFIGS presets
# ---------------------------------------------------------------------------
class TestCONFIGS:
    def test_has_presets(self):
        assert len(CONFIGS) > 0

    def test_each_is_TrainingConfig(self):
        for name, c in CONFIGS.items():
            assert isinstance(c, TrainingConfig)
            assert c.name == name or c.name  # at least non-empty

    def test_each_has_total_epochs(self):
        for c in CONFIGS.values():
            assert c.total_epochs > 0

    def test_each_hashes(self):
        # All preset hashes must be distinct.
        hashes = {c.hash() for c in CONFIGS.values()}
        assert len(hashes) == len(CONFIGS)
