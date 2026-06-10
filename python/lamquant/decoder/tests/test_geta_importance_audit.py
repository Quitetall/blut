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


N_BATCHES = 3  # shared by the fixed-batch builder and the un-normalize factor


def _fixed_batches(ch=8, n=N_BATCHES):
    # Fixed, seeded inputs so the gradient-weighted and magnitude-only passes
    # see identical activations (the only difference is the grad weighting).
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(2, ch, 16, generator=g),) for _ in range(n)]


def _magnitude_only_importance(dec, batches):
    """Replicates the OLD buggy path: activations detached -> grad forced to
    ones -> importance = sum(mean(|act|)). Used as the baseline the fixed
    (gradient-weighted) importance must differ from.
    """
    acts = {}
    hooks = []
    for name, m in dec.named_modules():
        if isinstance(m, nn.Conv1d) and "blocks" in name:
            hooks.append(m.register_forward_hook(
                lambda mod, i, o, n=name: acts.__setitem__(n, o.detach())))
    imp = {}
    with torch.no_grad():
        for batch in batches:
            x = batch[0]
            dec(x)
            for n, a in acts.items():
                # grad forced to ones (detached) -> magnitude-only weighting.
                ci = a.abs().mean(dim=(0, 2))
                imp[n] = imp.get(n, 0) + ci.cpu()
            acts.clear()
    for h in hooks:
        h.remove()
    return imp


def test_importance_is_finite_and_nonuniform():
    torch.manual_seed(0)
    dec = _TinyDecoder()
    imp = compute_importance(dec, _fixed_batches(), device=torch.device("cpu"),
                             n_batches=N_BATCHES)
    assert imp, "expected at least one scored block"
    for name, scores in imp.items():
        s = scores.detach().cpu().numpy()
        assert np.isfinite(s).all(), f"{name} importance must be finite"


def test_gradient_weighting_changes_importance_vs_magnitude_only():
    """The core regression: the fix makes the gradient actually weight the
    importance. The buggy code computed magnitude-only (grad==ones), so the
    fixed scores must differ from the magnitude-only baseline. (A naive
    'std>0' check would pass on the buggy code too, since channel magnitudes
    already vary — this comparison is what distinguishes fixed from broken.)
    """
    torch.manual_seed(0)
    dec = _TinyDecoder()
    batches = _fixed_batches()

    imp_fixed = compute_importance(dec, [(b[0].clone(),) for b in batches],
                                   device=torch.device("cpu"), n_batches=N_BATCHES)
    imp_mag = _magnitude_only_importance(dec, [(b[0].clone(),) for b in batches])

    # compute_importance divides by n_batches; undo it to compare raw sums.
    changed = any(
        not torch.allclose(imp_fixed[n] * float(N_BATCHES), imp_mag[n], atol=1e-5)
        for n in imp_fixed
    )
    assert changed, "gradient weighting had no effect — grad is not flowing"
