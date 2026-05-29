"""ai_models/student/muon_optimizer.py — Muon optimizer (vendored from KellerJordan/Muon).

Muon = MomentUm Orthogonalized by Newton-schulz.
Single-device variant for LamQuant training.

Reference: https://github.com/KellerJordan/Muon
"""
from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G, steps: int = 5):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Quintic iteration with coefficients selected to maximize slope at zero.
    Produces something like US'V^T where S' ~ Uniform(0.5, 1.5), which
    empirically doesn't hurt model performance relative to exact UV^T.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    """Single-device Muon with auxiliary AdamW for non-matrix params.

    Usage:
        hidden_matrix_params = [p for p in model.parameters() if p.ndim >= 2]
        scalar_params = [p for p in model.parameters() if p.ndim < 2]
        optimizer = Muon([
            dict(params=hidden_matrix_params, lr=0.02, momentum=0.95, use_muon=True),
            dict(params=scalar_params, lr=3e-4, betas=(0.95, 0.95), eps=1e-8,
                 weight_decay=0.01, use_muon=False),
        ])
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert 'use_muon' in group
            if group['use_muon']:
                group.setdefault('lr', 0.02)
                group.setdefault('momentum', 0.95)
                group.setdefault('weight_decay', 0)
            else:
                group.setdefault('lr', 3e-4)
                group.setdefault('betas', (0.95, 0.95))
                group.setdefault('eps', 1e-8)
                group.setdefault('weight_decay', 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group['use_muon']:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    update = muon_update(p.grad, state['momentum_buffer'],
                                         beta=group['momentum'])
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update.reshape(p.shape), alpha=-group['lr'])
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['step'] = 0
                    state['step'] += 1
                    update = adam_update(p.grad, state['exp_avg'],
                                         state['exp_avg_sq'], state['step'],
                                         group['betas'], group['eps'])
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])
        return loss


def split_params_for_muon(model):
    """Split model parameters into Muon-eligible (2D+) and AdamW (1D)."""
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


__all__ = ['Muon', 'split_params_for_muon']
