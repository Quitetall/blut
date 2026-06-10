"""Regression test for the 2026-06-10 audit fix in geta_pruning.

The forward hook used to store `out.detach()`, which made every
`act.grad_fn is not None` check False, so importance collapsed to a uniform
ones() vector and pruning was effectively random. After the fix the gradient
reaches each captured activation (retain_grad), so importance varies across
channels.
"""

import numpy as np
import torch
import torch.nn as nn

from geta_pruning import compute_importance


class _TinyDecoder(nn.Module):
    """Minimal decoder with a 'blocks' Conv1d so the hook fires."""

    def __init__(self, ch=8):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=1)])
        self.head = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        for b in self.blocks:
            x = torch.relu(b(x))
        return self.head(x)


def _loader(ch=8, n=3):
    for _ in range(n):
        # Non-degenerate input so per-channel gradients differ.
        yield (torch.randn(2, ch, 16),)


def test_importance_is_not_uniform_after_fix():
    torch.manual_seed(0)
    dec = _TinyDecoder()
    imp = compute_importance(dec, list(_loader()), device=torch.device("cpu"),
                             n_batches=3)

    assert imp, "expected at least one scored block"
    for name, scores in imp.items():
        s = scores.detach().cpu().numpy()
        assert np.isfinite(s).all(), f"{name} importance must be finite"
        # The bug produced an all-equal (ones-derived) vector. A real
        # gradient-weighted score varies across channels.
        assert s.std() > 1e-8, f"{name} importance is uniform — grad not flowing"
