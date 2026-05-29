"""Unit tests for ai_models/student/seizure_head.py — Phase 1 quick win.

Tiny binary classification head over the encoder's [B, 32, 79] latent.
"""
from __future__ import annotations

import pytest
import torch

from seizure_head import SeizureHead

pytestmark = pytest.mark.l2


class TestForward:
    def test_output_shape(self):
        head = SeizureHead(latent_dim=32)
        x = torch.randn(4, 32, 79)
        out = head(x)
        assert out.shape == (4,)

    def test_custom_latent_dim(self):
        head = SeizureHead(latent_dim=16)
        x = torch.randn(2, 16, 100)
        out = head(x)
        assert out.shape == (2,)

    def test_gradient_flows(self):
        head = SeizureHead()
        x = torch.randn(2, 32, 79, requires_grad=True)
        head(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestLoss:
    def test_returns_scalar(self):
        logits = torch.tensor([0.5, -0.2, 0.8])
        labels = torch.tensor([1, 0, 1])
        out = SeizureHead.loss(logits, labels)
        assert out.ndim == 0

    def test_perfect_prediction_low_loss(self):
        logits = torch.tensor([10.0, -10.0, 10.0])
        labels = torch.tensor([1, 0, 1])
        out = SeizureHead.loss(logits, labels)
        assert out.item() < 0.01

    def test_wrong_prediction_high_loss(self):
        logits = torch.tensor([-10.0, 10.0, -10.0])
        labels = torch.tensor([1, 0, 1])
        out = SeizureHead.loss(logits, labels)
        assert out.item() > 1.0

    def test_pos_weight_applied(self):
        logits = torch.tensor([0.0, 0.0])
        labels = torch.tensor([1, 0])
        low = SeizureHead.loss(logits, labels, pos_weight=1.0)
        high = SeizureHead.loss(logits, labels, pos_weight=10.0)
        # Higher pos_weight → larger loss for missing the positive sample
        assert high.item() > low.item()


class TestAccuracy:
    def test_perfect_metrics_one(self):
        # All correct
        logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
        labels = torch.tensor([1, 0, 1, 0])
        m = SeizureHead.accuracy(logits, labels)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_inverted_predictions_zero_sens(self):
        logits = torch.tensor([-10.0, 10.0, -10.0, 10.0])
        labels = torch.tensor([1, 0, 1, 0])
        m = SeizureHead.accuracy(logits, labels)
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["sensitivity"] == pytest.approx(0.0)

    def test_returns_4_keys(self):
        logits = torch.randn(4)
        labels = torch.randint(0, 2, (4,))
        m = SeizureHead.accuracy(logits, labels)
        assert set(m.keys()) == {"accuracy", "sensitivity", "specificity", "f1"}

    def test_custom_threshold(self):
        # Low sigmoid values, high threshold → all predicted negative
        logits = torch.tensor([-1.0, -1.0])
        labels = torch.tensor([0, 0])
        m = SeizureHead.accuracy(logits, labels, threshold=0.9)
        assert m["accuracy"] == pytest.approx(1.0)
