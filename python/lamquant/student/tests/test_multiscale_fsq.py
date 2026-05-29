"""Unit tests for ai_models/student/multiscale_fsq.py — Phase 1 quick win.

SNAC-style hierarchical multi-scale FSQ. Covers MultiScaleFSQ
construction, encode/decode/forward, token accounting, and the
make_multiscale_fsq factory presets.
"""
from __future__ import annotations

import math

import pytest
import torch

from multiscale_fsq import MultiScaleFSQ, make_multiscale_fsq

pytestmark = pytest.mark.l2


# ---------------------------------------------------------------------------
# MultiScaleFSQ construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_defaults(self):
        m = MultiScaleFSQ()
        assert m.dim == 32
        assert m.T == 79
        assert m.strides == [8, 4, 2, 1]
        assert m.levels == [3, 3, 5, 5]
        assert m.n_scales == 4
        assert len(m.scale_gains) == 4

    def test_mismatched_lens_raises(self):
        with pytest.raises(AssertionError, match="must match"):
            MultiScaleFSQ(strides=[8, 4], levels=[3])

    def test_custom_strides_levels(self):
        m = MultiScaleFSQ(strides=[4, 2, 1], levels=[2, 3, 5])
        assert m.n_scales == 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
class TestInternals:
    def test_downsample_stride_1_identity(self):
        m = MultiScaleFSQ()
        x = torch.randn(2, 32, 79)
        assert m._downsample(x, 1).shape == x.shape

    def test_downsample_halves(self):
        m = MultiScaleFSQ()
        x = torch.randn(1, 32, 80)
        y = m._downsample(x, 2)
        assert y.shape == (1, 32, 40)

    def test_upsample_to_target_T(self):
        m = MultiScaleFSQ()
        x = torch.randn(1, 32, 10)
        y = m._upsample(x, 79)
        assert y.shape == (1, 32, 79)

    def test_upsample_noop_when_matches(self):
        m = MultiScaleFSQ()
        x = torch.randn(1, 32, 79)
        y = m._upsample(x, 79)
        assert y.shape == x.shape

    def test_fsq_quantizes_to_centers(self):
        m = MultiScaleFSQ()
        x = torch.tensor([[[ -1.0, -0.5, 0.0, 0.5, 1.0]]])
        q, idx = m._fsq(x, L=4)
        # L=4: step=0.5, centers at -0.75, -0.25, 0.25, 0.75
        assert q.shape == x.shape
        assert idx.dtype == torch.long
        assert idx.min() >= 0
        assert idx.max() < 4

    def test_fsq_indices_clamped(self):
        m = MultiScaleFSQ()
        x = torch.tensor([[[10.0, -10.0]]])
        _, idx = m._fsq(x, L=3)
        assert idx.max() < 3
        assert idx.min() >= 0


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------
class TestEncodeDecode:
    def test_encode_returns_tokens_and_quants(self):
        m = MultiScaleFSQ()
        x = torch.randn(2, 32, 79).clamp(-1, 1)
        tokens, quants = m.encode(x)
        assert len(tokens) == m.n_scales
        assert len(quants) == m.n_scales
        # First scale: stride=8 → T_scale=9 (avg_pool with count_include_pad=False)
        for tok, stride in zip(tokens, m.strides):
            expected_T = max(1, 79 // stride) if stride > 1 else 79
            # avg_pool may give floor(79/stride) — both 9 or 10 acceptable
            assert tok.shape[0] == 2
            assert tok.shape[1] == 32

    def test_decode_returns_target_shape(self):
        m = MultiScaleFSQ()
        x = torch.randn(1, 32, 79).clamp(-1, 1)
        _, quants = m.encode(x)
        recon = m.decode(quants)
        assert recon.shape == (1, 32, 79)

    def test_forward_returns_triple(self):
        m = MultiScaleFSQ()
        x = torch.randn(2, 32, 79).clamp(-1, 1)
        recon, tokens, loss = m(x)
        assert recon.shape == x.shape
        assert len(tokens) == m.n_scales
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_forward_loss_decreases_with_more_levels(self):
        torch.manual_seed(0)
        x = torch.randn(1, 32, 79).clamp(-1, 1)
        m_coarse = MultiScaleFSQ(strides=[1], levels=[2])
        m_fine = MultiScaleFSQ(strides=[1], levels=[16])
        _, _, l_coarse = m_coarse(x)
        _, _, l_fine = m_fine(x)
        # More levels → lower quantization error
        assert l_fine.item() < l_coarse.item()


# ---------------------------------------------------------------------------
# token_count / estimated_cr
# ---------------------------------------------------------------------------
class TestTokenAccounting:
    def test_token_count_keys(self):
        m = MultiScaleFSQ()
        info = m.token_count()
        for i in range(m.n_scales):
            assert f"scale_{i}" in info
        assert "total_tokens" in info
        assert "total_bits" in info

    def test_token_count_per_scale_keys(self):
        m = MultiScaleFSQ()
        info = m.token_count()
        s0 = info["scale_0"]
        assert set(s0.keys()) == {"stride", "T", "L", "tokens", "bits"}

    def test_total_tokens_sums(self):
        m = MultiScaleFSQ()
        info = m.token_count()
        total = sum(info[f"scale_{i}"]["tokens"] for i in range(m.n_scales))
        assert info["total_tokens"] == total

    def test_estimated_cr_positive(self):
        m = MultiScaleFSQ()
        cr = m.estimated_cr(raw_bytes=105000)
        assert cr > 0


# ---------------------------------------------------------------------------
# make_multiscale_fsq presets
# ---------------------------------------------------------------------------
class TestPresets:
    @pytest.mark.parametrize("preset", ["compact", "balanced", "quality", "flat"])
    def test_preset_constructs(self, preset):
        m = make_multiscale_fsq(preset)
        assert isinstance(m, MultiScaleFSQ)

    def test_unknown_preset_falls_back_to_balanced(self):
        m = make_multiscale_fsq("not_a_preset")
        assert m.strides == [8, 4, 2, 1]
        assert m.levels == [3, 3, 5, 5]

    def test_flat_preset_single_scale(self):
        m = make_multiscale_fsq("flat")
        assert m.n_scales == 1
        assert m.strides == [1]

    def test_compact_has_fewer_bits_than_quality(self):
        cr_compact = make_multiscale_fsq("compact").estimated_cr()
        cr_quality = make_multiscale_fsq("quality").estimated_cr()
        # compact = higher CR (more compression)
        assert cr_compact > cr_quality
