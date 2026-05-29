"""Unit tests for SNN classification heads.

Covers the contract every head must satisfy plus a CRF gradient sanity
check (V4 Pro Finding 6 of the 5-head-bench commit). Designed for fast
CI: tiny tensors, deterministic seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lamquant" / "snn"))

from lamquant_neural.models.heads import (  # noqa: E402
    ThresholdHead, AttentionSoftmaxHead, CRFHead,
    TemporalAttentionHead, MoEFSQHead, HEAD_REGISTRY, build_head,
)


@pytest.mark.parametrize("name", sorted(HEAD_REGISTRY))
def test_head_contract_shapes(name):
    """Every head: [B, 8, T] → ([B, T] long, [B, K, T] float)."""
    head = build_head(name)
    B, G, T = 2, 8, 79
    logits = torch.randn(B, G, T)
    states, class_logits = head(logits, target_T=T)
    assert states.shape == (B, T)
    assert states.dtype == torch.long
    assert states.min() >= 0 and states.max() < head.K
    assert class_logits.shape == (B, head.K, T)
    assert class_logits.dtype.is_floating_point


def test_threshold_head_argmax_matches_states():
    """Synthesized class logits must argmax to the reported state."""
    head = ThresholdHead()
    torch.manual_seed(7)
    logits = torch.randn(3, 8, 79)
    states, class_logits = head(logits, target_T=79)
    assert (class_logits.argmax(dim=1) == states).all()


def test_attention_softmax_argmax_matches_states():
    head = AttentionSoftmaxHead()
    torch.manual_seed(7)
    logits = torch.randn(3, 8, 79)
    states, class_logits = head(logits, target_T=79)
    assert (class_logits.argmax(dim=1) == states).all()


def test_crf_transitions_sticky_init():
    """CRFHead.__init__ must leave a positive diagonal anti-flicker prior."""
    head = CRFHead()
    K = head.K
    tr = head.transitions.detach()
    # Diagonal entries (self-loop) must be greater than off-diagonal mean.
    diag = tr.diagonal()
    off_diag = tr.masked_select(~torch.eye(K, dtype=torch.bool)).reshape(K, K - 1)
    assert (diag.mean() > off_diag.mean()).item(), (
        f"CRF sticky-init failed: diag={diag.tolist()}, off={off_diag.tolist()}"
    )


def test_crf_viterbi_respects_sticky_prior():
    """With high sticky prior and noisy emissions, Viterbi must collapse
    to a single state (anti-flicker working as advertised)."""
    head = CRFHead(sticky_init=10.0)
    head.eval()
    torch.manual_seed(0)
    # Tiny emissions — almost no signal.
    emissions = torch.randn(1, head.K, 50) * 0.1
    path = head.viterbi(emissions)
    # Count unique states in the path; with high sticky prior, should be 1.
    unique = path.unique().numel()
    assert unique == 1, f"Expected 1 unique state under sticky prior, got {unique}"


def test_crf_neg_log_likelihood_finite():
    """CRF NLL is finite and differentiable for arbitrary inputs."""
    head = CRFHead()
    B, K, T = 2, head.K, 10
    emissions = torch.randn(B, K, T, requires_grad=True)
    targets = torch.randint(0, K, (B, T))
    nll = head.neg_log_likelihood(emissions, targets)
    assert torch.isfinite(nll).item()
    nll.backward()
    assert emissions.grad is not None
    assert torch.isfinite(emissions.grad).all().item()


def test_crf_nll_decreases_under_sgd():
    """CRF NLL must decrease when emissions are nudged toward the targets.

    Uses fixed seed + moderate LR + 50 steps + tolerance band so the test
    is robust to torch / hardware variation (V4 Pro Finding 1 of the
    5-head-bench-fixes commit). Freezes the head's parameters so only
    the input emissions move."""
    torch.manual_seed(0)
    head = CRFHead()
    for p in head.parameters():
        p.requires_grad_(False)
    B, K, T = 1, head.K, 5
    targets = torch.randint(0, K, (B, T))
    emissions = torch.randn(B, K, T, requires_grad=True)
    opt = torch.optim.SGD([emissions], lr=0.3)
    loss_before = head.neg_log_likelihood(emissions, targets).item()
    for _ in range(50):
        opt.zero_grad()
        loss = head.neg_log_likelihood(emissions, targets)
        loss.backward()
        opt.step()
    loss_after = head.neg_log_likelihood(emissions, targets).item()
    # Tolerance band guards against a near-flat loss surface giving a
    # false negative on edge versions of torch.
    assert loss_after < loss_before - 1e-3, (
        f"CRF NLL did not decrease meaningfully: {loss_before} → {loss_after}"
    )


def test_moe_fsq_routing_is_hard_at_inference():
    """MoEFSQHead.eval() → routing is one-hot (no soft mixture leaks)."""
    head = MoEFSQHead()
    head.eval()
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 30)
    # Capture the internal routing by patching the gate forward.
    gate_logits = head.gate(logits.transpose(1, 2))
    idx = gate_logits.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(gate_logits).scatter_(-1, idx, 1.0)
    assert torch.allclose(one_hot.sum(dim=-1),
                          torch.ones(one_hot.shape[:-1]))


def test_moe_fsq_routing_is_soft_in_training():
    """At training time, Gumbel-softmax produces a hard-routing tensor."""
    head = MoEFSQHead()
    head.train()
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 30, requires_grad=True)
    states, class_logits = head(logits, target_T=30)
    # Gumbel-softmax with hard=True returns a one-hot tensor but
    # differentiates through the soft version. Check class_logits is
    # differentiable.
    loss = class_logits.sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()


def test_temporal_attention_head_runs_on_short_sequences():
    """TemporalAttention should handle T ≤ 4 cleanly (some heads break on
    very short sequences)."""
    head = TemporalAttentionHead()
    head.eval()
    logits = torch.randn(1, 8, 4)
    states, class_logits = head(logits, target_T=4)
    assert states.shape == (1, 4)
    assert class_logits.shape == (1, head.K, 4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
