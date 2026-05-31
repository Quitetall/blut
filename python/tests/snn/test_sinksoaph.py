# SPDX-License-Identifier: GPL-3.0-or-later
"""Correctness checks for the SinkSOAPH optimizer (PyTorch arm).

Run: LamQuant-Neural/.venv/bin/python -m pytest tests/snn/test_sinksoaph.py
(or invoke directly — has a __main__ runner so it works without pytest).
"""
import os
import sys

import torch

_STUDENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "lamquant", "student"))
if _STUDENT not in sys.path:
    sys.path.insert(0, _STUDENT)

from sinksoaph import (  # noqa: E402
    SinkSOAPH,
    _sinkhorn_energy_balance,
    _hyperball_apply_,
)


def test_sinkhorn_reaches_doubly_stochastic():
    """After enough iters, the balanced energy A**2 has near-uniform row/col
    marginals -> row/col energy ratios converge to 1.0."""
    torch.manual_seed(0)
    A = torch.randn(40, 17)  # non-square, like our real matrices
    A_bal = _sinkhorn_energy_balance(A, steps=20, eps=1e-8)
    E = A_bal.square()
    row_energy = E.sum(dim=1)
    col_energy = E.sum(dim=0)
    # Uniform marginals => every row carries the same energy (same for cols).
    row_ratio = (row_energy.max() / row_energy.min()).item()
    col_ratio = (col_energy.max() / col_energy.min()).item()
    assert row_ratio < 1.05, f"row energy not balanced: ratio={row_ratio:.4f}"
    assert col_ratio < 1.05, f"col energy not balanced: ratio={col_ratio:.4f}"
    print(f"[ok] sinkhorn row_ratio={row_ratio:.4f} col_ratio={col_ratio:.4f}")


def test_hyperball_applied_delta_is_lr_times_param_norm():
    """Hyperball: ||applied delta|| ~= lr * ||p|| and ||p|| is preserved."""
    torch.manual_seed(0)
    lr = 0.0125
    p = torch.randn(40, 21)
    p0 = p.clone()
    direction = torch.randn(40, 21)
    _hyperball_apply_(p, direction, lr=lr, eps=1e-8)
    delta = (p - p0).norm().item()
    ratio = delta / p0.norm().item()
    # The pre-projection step has magnitude exactly lr*||p||; the sphere
    # re-projection nudges it slightly, so allow a small tolerance.
    assert abs(ratio - lr) < 0.2 * lr, f"|delta|/|p|={ratio:.5f} vs lr={lr}"
    # Norm preserved by the sphere projection.
    assert abs(p.norm().item() - p0.norm().item()) < 1e-3
    print(f"[ok] hyperball |delta|/|p|={ratio:.5f} (lr={lr}); ||p|| preserved")


def test_sinksoaph_two_step_smoke_on_real_snn():
    """Two real steps on MambaSNN: no NaN/Inf, sinksoaph-group norms preserved,
    adamw-group params actually move."""
    from lamquant_neural.models.mamba_ssm_minimal import MambaSNN

    torch.manual_seed(1337)
    model = MambaSNN(in_channels=21, d_model=40, d_state=16, n_layers=2,
                     use_subband=True)

    suff = ('in_proj.weight', 'x_proj.weight', 'out_proj.weight',
            'spatial_mix.weight')
    sink, rest = [], []
    sink_named = []
    for nm, p in model.named_parameters():
        if p.ndim == 2 and nm.endswith(suff):
            sink.append(p)
            sink_named.append((nm, p))
        else:
            rest.append(p)
    assert len(sink) == 13, f"expected 13 linear matrices, got {len(sink)}"

    opt = SinkSOAPH(
        [{"params": sink, "method": "sinksoaph", "weight_decay": 0.0},
         {"params": rest, "method": "adamw", "weight_decay": 1e-4}],
        lr=1e-3, betas=(0.9, 0.95))

    norms0 = {nm: p.norm().item() for nm, p in sink_named}
    a_param = rest[0]
    a0 = a_param.detach().clone()

    for _ in range(2):
        x = torch.randn(4, 21, 313)            # [B, C, L3_T]-ish
        opt.zero_grad()
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = logits.float().pow(2).mean()
        loss.backward()
        opt.step()

    for nm, p in sink_named:
        assert torch.isfinite(p).all(), f"NaN/Inf in {nm}"
        drift = abs(p.norm().item() - norms0[nm]) / max(norms0[nm], 1e-12)
        assert drift < 1e-2, f"{nm} norm drifted {drift:.4f} (hyperball should preserve)"
    assert torch.isfinite(a_param).all()
    assert (a_param - a0).abs().sum().item() > 0, "adamw group did not move"
    print(f"[ok] 2-step SNN smoke: finite, 13 sink-norms preserved, adamw moved")


if __name__ == "__main__":
    test_sinkhorn_reaches_doubly_stochastic()
    test_hyperball_applied_delta_is_lr_times_param_norm()
    test_sinksoaph_two_step_smoke_on_real_snn()
    print("\nALL SINKSOAPH TESTS PASSED")
