"""Tests for the forward ingredient registry (ADR 0050/0051).

One ``kind="forward"`` spec:

  * ``mae_masked`` — ``student/pretrain_mae.py``'s masked-autoencoder forward
                     (mask-zero -> encode(quantize=False) -> predict).

Equivalence is proven against a VERBATIM inline copy of the trainer loop's
forward block, with lightweight fakes standing in for the encoder + pred_head
(the forward primitive is pure tensor ops; no neural wheel needed). Same inputs
-> tensor-equal recon.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Importing this module registers the forward spec.
import lamquant.ingredients.forward._specs  # noqa: F401
from lamquant.ingredients import build_ingredient, get_spec, list_ingredients


class _FakeEncoder(nn.Module):
    """Stands in for TernaryMobileNetV5_Subband: exposes encode(x, quantize=)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(21, 32, kernel_size=4, stride=4)

    def encode(self, x, quantize=False):
        # quantize is part of the API surface the forward primitive calls with
        # quantize=False; the fake ignores its value (records it for the test).
        self.last_quantize = quantize
        return self.proj(x)


class _FakePredHead(nn.Module):
    """Stands in for MAEPredictionHead: latent [B,32,T'] -> [B,21,T]."""

    def __init__(self, l3_len):
        super().__init__()
        self.l3_len = l3_len
        self.up = nn.ConvTranspose1d(32, 21, kernel_size=4, stride=4)

    def forward(self, latent):
        return self.up(latent)[..., :self.l3_len]


def _inline_forward(encoder, pred_head, l3, mask):
    """Verbatim copy of pretrain_mae.run_pretraining forward block."""
    l3_masked = l3 * (1.0 - mask)
    latent = encoder.encode(l3_masked, quantize=False)
    l3_pred = pred_head(latent)
    return l3_pred


def test_mae_masked_registered():
    assert "mae_masked" in list_ingredients("forward")


def test_forward_spec_not_cache_relevant():
    assert get_spec("forward", "mae_masked").cache_relevant is False


def test_mae_masked_equals_inline():
    torch.manual_seed(0)
    B, C, T = 4, 21, 312
    enc = _FakeEncoder().eval()
    head = _FakePredHead(T).eval()
    l3 = torch.randn(B, C, T)
    mask = (torch.rand(B, C, T) > 0.5).float()

    forward = build_ingredient("forward", "mae_masked", {})
    with torch.no_grad():
        got = forward(enc, head, l3, mask)
        exp = _inline_forward(enc, head, l3, mask)
    assert torch.equal(got, exp)
    assert got.shape == (B, C, T)


def test_mae_masked_calls_encode_with_quantize_false():
    B, C, T = 2, 21, 312
    enc = _FakeEncoder().eval()
    head = _FakePredHead(T).eval()
    l3 = torch.randn(B, C, T)
    mask = (torch.rand(B, C, T) > 0.5).float()
    forward = build_ingredient("forward", "mae_masked", {})
    with torch.no_grad():
        forward(enc, head, l3, mask)
    assert enc.last_quantize is False  # MAE pretrain never quantizes.


def test_mae_masked_zeros_masked_region():
    # Where mask==1, the encoder must see a zeroed input (mask-zero contract).
    B, C, T = 1, 21, 312
    seen = {}

    class _RecordEncoder(_FakeEncoder):
        def encode(self, x, quantize=False):
            seen["x"] = x.clone()
            return super().encode(x, quantize=quantize)

    enc = _RecordEncoder().eval()
    head = _FakePredHead(T).eval()
    l3 = torch.randn(B, C, T)
    mask = (torch.rand(B, C, T) > 0.5).float()
    forward = build_ingredient("forward", "mae_masked", {})
    with torch.no_grad():
        forward(enc, head, l3, mask)
    assert torch.equal(seen["x"], l3 * (1.0 - mask))
    assert torch.all(seen["x"][mask.bool()] == 0)
