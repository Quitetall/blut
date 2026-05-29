"""Unit tests for ai_models/student/progressive_quant.py — Phase 2."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from progressive_quant import (
    ProgressiveConv1d,
    ProgressiveConvTranspose1d,
    ProgressiveINT8Conv1d,
    ProgressiveQuantSchedule,
    _ProgressiveQuantFunction,
    progressive_quantize,
    set_model_bits,
)

pytestmark = pytest.mark.l2


class TestSchedule:
    def test_default_schedule(self):
        s = ProgressiveQuantSchedule(total_epochs=100)
        # Default: 15% INT8, 25% INT4, 60% ternary
        assert s.get_bits(0) == 8
        assert s.get_bits(14) == 8
        assert s.get_bits(15) == 4
        assert s.get_bits(40) == "ternary"
        assert s.get_bits(99) == "ternary"

    def test_custom_schedule(self):
        s = ProgressiveQuantSchedule(total_epochs=10,
                                       schedule=[(0.5, 8), (0.5, "ternary")])
        assert s.get_bits(0) == 8
        assert s.get_bits(4) == 8
        assert s.get_bits(5) == "ternary"

    def test_repr_contains_phases(self):
        s = ProgressiveQuantSchedule(total_epochs=100)
        r = repr(s)
        assert "ep0-" in r
        assert "ternary" in r

    def test_get_bits_past_end_returns_last(self):
        s = ProgressiveQuantSchedule(total_epochs=10)
        # Past total_epochs+1 → last phase
        assert s.get_bits(1000) == "ternary"


class TestProgressiveQuantFn:
    def test_int8_quantize(self):
        w = torch.randn(64)
        out = progressive_quantize(w, bits=8)
        # Output has discrete levels
        assert out.shape == w.shape
        assert torch.isfinite(out).all()

    def test_int4_quantize(self):
        w = torch.randn(64)
        out = progressive_quantize(w, bits=4)
        assert out.shape == w.shape

    def test_int2_quantize(self):
        w = torch.randn(64)
        out = progressive_quantize(w, bits=2)
        assert out.shape == w.shape

    def test_ternary_string(self):
        w = torch.randn(64)
        out = progressive_quantize(w, bits="ternary")
        # All values should be in {-1, 0, +1} × scale → 3 unique magnitudes
        unique_signs = set(out.sign().unique().tolist())
        assert unique_signs.issubset({-1.0, 0.0, 1.0})

    def test_bits_le_1_treated_as_ternary(self):
        w = torch.randn(32)
        out = progressive_quantize(w, bits=1)
        assert out.shape == w.shape

    def test_backward_pass(self):
        w = torch.randn(16, requires_grad=True)
        out = progressive_quantize(w, bits=4)
        out.sum().backward()
        assert w.grad is not None
        assert torch.isfinite(w.grad).all()

    def test_zero_weight_doesnt_div_by_zero(self):
        w = torch.zeros(16)
        out = progressive_quantize(w, bits=4)
        assert torch.isfinite(out).all()


class TestProgressiveConv1d:
    def test_quantize_path(self):
        c = ProgressiveConv1d(4, 8, kernel_size=3)
        x = torch.randn(2, 4, 16)
        out = c(x, quantize=True)
        assert out.shape == (2, 8, 16)

    def test_no_quantize_path(self):
        c = ProgressiveConv1d(4, 8, kernel_size=3)
        x = torch.randn(2, 4, 16)
        out = c(x, quantize=False)
        assert out.shape == (2, 8, 16)

    def test_set_bits_changes_quantization(self):
        c = ProgressiveConv1d(4, 8, kernel_size=3)
        assert c._bits == 8
        c.set_bits("ternary")
        assert c._bits == "ternary"

    def test_get_ternary_weights(self):
        c = ProgressiveConv1d(4, 8, kernel_size=3)
        w = c.get_ternary_weights()
        assert w.shape == c.weight.shape
        assert set(w.sign().unique().tolist()).issubset({-1.0, 0.0, 1.0})

    def test_grouped_convolution(self):
        c = ProgressiveConv1d(8, 8, kernel_size=3, groups=8)
        x = torch.randn(1, 8, 16)
        out = c(x, quantize=True)
        assert out.shape == (1, 8, 16)

    def test_with_bias(self):
        c = ProgressiveConv1d(4, 8, kernel_size=3, bias=True)
        assert c.bias is not None
        x = torch.randn(2, 4, 16)
        assert c(x).shape == (2, 8, 16)


class TestProgressiveConvTranspose1d:
    def test_quantize_path(self):
        c = ProgressiveConvTranspose1d(4, 8, kernel_size=3, stride=2)
        x = torch.randn(2, 4, 8)
        out = c(x, quantize=True)
        assert out.shape[0] == 2 and out.shape[1] == 8

    def test_no_quantize_path(self):
        c = ProgressiveConvTranspose1d(4, 8, kernel_size=3, stride=2)
        x = torch.randn(2, 4, 8)
        out = c(x, quantize=False)
        assert out.shape[0] == 2

    def test_get_ternary_weights(self):
        c = ProgressiveConvTranspose1d(4, 8, kernel_size=3)
        w = c.get_ternary_weights()
        assert w.shape == c.weight.shape

    def test_set_bits(self):
        c = ProgressiveConvTranspose1d(4, 8, kernel_size=3)
        c.set_bits(4)
        assert c._bits == 4


class TestProgressiveINT8Conv1d:
    def test_quantize_path(self):
        c = ProgressiveINT8Conv1d(4, 8, kernel_size=3)
        x = torch.randn(2, 4, 16)
        out = c(x, quantize=True)
        assert out.shape == (2, 8, 16)

    def test_no_quantize_path(self):
        c = ProgressiveINT8Conv1d(4, 8, kernel_size=3)
        x = torch.randn(2, 4, 16)
        out = c(x, quantize=False)
        assert out.shape == (2, 8, 16)

    def test_set_bits_clamps_to_8(self):
        c = ProgressiveINT8Conv1d(4, 8, kernel_size=3)
        c.set_bits(4)  # should stay at 8
        assert c._bits == 8
        c.set_bits(16)
        assert c._bits == 16

    def test_set_bits_string_clamps(self):
        c = ProgressiveINT8Conv1d(4, 8, kernel_size=3)
        c.set_bits("ternary")
        assert c._bits == 8


class TestSetModelBits:
    def test_propagates_to_all_progressive_modules(self):
        m = nn.Sequential(
            ProgressiveConv1d(4, 8, kernel_size=3),
            ProgressiveConv1d(8, 4, kernel_size=3),
            nn.Linear(4, 4),  # no set_bits
        )
        set_model_bits(m, "ternary")
        assert m[0]._bits == "ternary"
        assert m[1]._bits == "ternary"

    def test_skips_modules_without_set_bits(self):
        m = nn.Linear(4, 4)
        set_model_bits(m, 4)  # must not raise
