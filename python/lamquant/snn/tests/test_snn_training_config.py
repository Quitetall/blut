"""Unit tests for ai_models/snn/snn_training_config.py — Phase 1 trivial.

Frozen SNNConfig dataclass + 3 SNN_CONFIGS presets (fast/standard/production).
"""
from __future__ import annotations

import pytest

from snn_training_config import SNN_CONFIGS, SNNConfig

pytestmark = pytest.mark.l1


class TestSNNConfig:
    def test_default_construction(self):
        c = SNNConfig()
        assert c.name == "custom"
        assert c.epochs == 500
        assert c.batch_size == 128
        assert c.d_model == 40

    def test_frozen(self):
        c = SNNConfig()
        with pytest.raises(Exception):
            c.epochs = 999

    def test_param_estimate_positive(self):
        c = SNNConfig()
        assert c.param_estimate > 0
        # Memory says ~57K params for production
        assert 10_000 < c.param_estimate < 200_000

    def test_param_estimate_scales_with_d_model(self):
        a = SNNConfig(d_model=20)
        b = SNNConfig(d_model=80)
        assert b.param_estimate > a.param_estimate

    def test_replace_returns_new_instance(self):
        c = SNNConfig(name="orig")
        c2 = c.replace(name="new")
        assert c.name == "orig"
        assert c2.name == "new"

    def test_to_dict_round_trip(self):
        c = SNNConfig(name="t", epochs=10)
        d = c.to_dict()
        assert d["name"] == "t"
        assert d["epochs"] == 10

    def test_str_contains_name(self):
        c = SNNConfig(name="myconfig", description="desc")
        s = str(c)
        assert "myconfig" in s
        assert "desc" in s


class TestPresets:
    def test_three_presets_exist(self):
        assert set(SNN_CONFIGS.keys()) == {"fast", "standard", "production"}

    def test_fast_smaller_epochs(self):
        assert SNN_CONFIGS["fast"].epochs < SNN_CONFIGS["production"].epochs

    def test_standard_in_middle(self):
        epochs = [SNN_CONFIGS[k].epochs for k in ("fast", "standard", "production")]
        assert epochs[0] < epochs[1] <= epochs[2]

    def test_all_have_descriptions(self):
        for c in SNN_CONFIGS.values():
            assert c.description

    def test_all_have_param_estimates_positive(self):
        for c in SNN_CONFIGS.values():
            assert c.param_estimate > 0
