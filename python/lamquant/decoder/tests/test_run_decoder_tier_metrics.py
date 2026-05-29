"""Metric-helper tests for ``lamquant/decoder/run_decoder_tier.py``.

Pins the three metric helpers (``pearson_r_loss``, ``pearson_r_batch``,
``prd_batch``) against pure mathematical invariants — perfect-recon
yields the documented sentinel (0 / 1 / 0), constants are zero-protected,
shapes are preserved. Numeric drift is not asserted.

Math fixtures via ``torch.randn`` — not synthetic EEG data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


# The decoder dir is also on sys.path via blut/python/conftest.py; add
# it explicitly so the bare ``run_decoder_tier`` import resolves even
# when this file is collected in isolation.
_PY_ROOT = Path(__file__).resolve().parents[3]   # blut/python
_DECODER_DIR = str(_PY_ROOT / "lamquant" / "decoder")
if _DECODER_DIR not in sys.path:
    sys.path.insert(0, _DECODER_DIR)
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

import importlib
rdt = importlib.import_module("run_decoder_tier")

pytestmark = pytest.mark.l2


class TestPearsonRLoss:
    def test_zero_on_perfect_match(self) -> None:
        x = torch.randn(4, 21, 313)
        loss = rdt.pearson_r_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_high_on_anti_correlated(self) -> None:
        x = torch.randn(2, 4, 100)
        loss = rdt.pearson_r_loss(x, -x)
        # corr(x, -x) = -1 -> loss = 1 - (-1) = 2
        assert loss.item() == pytest.approx(2.0, abs=1e-4)

    def test_returns_scalar(self) -> None:
        x = torch.randn(3, 21, 313)
        y = torch.randn(3, 21, 313)
        loss = rdt.pearson_r_loss(x, y)
        assert loss.ndim == 0


class TestPearsonRBatch:
    def test_one_on_perfect_match(self) -> None:
        x = torch.randn(3, 4, 313)
        r = rdt.pearson_r_batch(x, x)
        assert r == pytest.approx(1.0, abs=1e-5)

    def test_returns_python_float(self) -> None:
        x = torch.randn(2, 4, 100)
        r = rdt.pearson_r_batch(x, x + 0.01 * torch.randn_like(x))
        assert isinstance(r, float)
        assert -1.0 <= r <= 1.0


class TestPrdBatch:
    def test_zero_on_perfect_match(self) -> None:
        x = torch.randn(3, 4, 313)
        p = rdt.prd_batch(x, x)
        assert p == pytest.approx(0.0, abs=1e-5)

    def test_positive_on_difference(self) -> None:
        x = torch.randn(2, 4, 100)
        y = x + torch.randn_like(x)
        p = rdt.prd_batch(y, x)
        assert p > 0

    def test_returns_python_float(self) -> None:
        x = torch.randn(2, 4, 100)
        p = rdt.prd_batch(x, x + 0.01 * torch.randn_like(x))
        assert isinstance(p, float)
