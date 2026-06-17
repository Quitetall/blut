"""Optimizer ingredient specs (ADR 0050 / 0051).

Centralizes the ESOAP/SinkSOAPH suffix routing — previously copy-pasted in
``train_4state_controller`` + ``pretrain_ssl_tueg`` — and the Muon 2-D split
into one home. ``build_ingredient("optimizer", name, cfg, named_params=...)``
is the single construction path; importing this module registers all five
specs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from lamquant.ingredients.registry import register_ingredient
from lamquant.ingredients.spec import IngredientSpec
from lamquant.ingredients.optimizers.esoap import ESOAP
from lamquant.ingredients.optimizers.sinksoaph import SinkSOAPH
from lamquant.ingredients.optimizers.soap_optimizer import SOAP
from lamquant.ingredients.optimizers.muon_optimizer import Muon

# 2-D weight name-suffixes routed to the preconditioned (SOAP-family) group.
# Transcribed verbatim from the trainers' hand-rolled routing
# (train_4state_controller.py:876 / pretrain_ssl_tueg.py:460) — the single
# source of this routing now.
_LINEAR_SUFFIXES = ("in_proj.weight", "x_proj.weight",
                    "out_proj.weight", "spatial_mix.weight")


def _route_by_suffix(named_params, method, weight_decay):
    """ESOAP/SinkSOAPH routing: 2-D weights whose name ends in a linear suffix go
    to the ``method`` group; everything else (1-D, non-suffix 2-D) goes to AdamW.
    Returns ``[{routed, method}, {rest, "adamw"}]``.
    """
    routed, rest = [], []
    for nm, q in named_params:
        if not q.requires_grad:
            continue
        if q.ndim == 2 and nm.endswith(_LINEAR_SUFFIXES):
            routed.append(q)
        else:
            rest.append(q)
    return [
        {"params": routed, "method": method, "weight_decay": weight_decay},
        {"params": rest, "method": "adamw", "weight_decay": weight_decay},
    ]


# ---- ESOAP --------------------------------------------------------------
@dataclass(frozen=True)
class EsoapConfig:
    lr: float = 1e-3
    rank_frac: float = 0.5
    mu: float = 0.95
    gram_beta: float = 0.95
    betas: tuple = (0.9, 0.95)
    ns_steps: int = 5
    nesterov: bool = True
    eps: float = 1e-8
    weight_decay: float = 0.0
    cautious_wd: bool = False


@register_ingredient
def _esoap():
    return IngredientSpec(
        name="esoap", kind="optimizer", config_cls=EsoapConfig,
        build_param_groups=lambda named, cfg: _route_by_suffix(
            named, "esoap", cfg.weight_decay),
        construct=lambda groups, cfg: ESOAP(
            groups, lr=cfg.lr, rank_frac=cfg.rank_frac, mu=cfg.mu,
            gram_beta=cfg.gram_beta, betas=cfg.betas, ns_steps=cfg.ns_steps,
            nesterov=cfg.nesterov, eps=cfg.eps, weight_decay=cfg.weight_decay,
            cautious_wd=cfg.cautious_wd),
    )


# ---- SinkSOAPH (exists; A/B arm #209, not yet trainer-wired) ------------
@dataclass(frozen=True)
class SinksoaphConfig:
    lr: float = 1e-3
    mu: float = 0.95
    gram_beta: float = 0.95
    sinkhorn_steps: int = 10
    nesterov: bool = True
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    sinkhorn_eps: float = 1e-6
    weight_decay: float = 0.0


@register_ingredient
def _sinksoaph():
    return IngredientSpec(
        name="sinksoaph", kind="optimizer", config_cls=SinksoaphConfig,
        build_param_groups=lambda named, cfg: _route_by_suffix(
            named, "sinksoaph", cfg.weight_decay),
        construct=lambda groups, cfg: SinkSOAPH(
            groups, lr=cfg.lr, mu=cfg.mu, gram_beta=cfg.gram_beta,
            sinkhorn_steps=cfg.sinkhorn_steps, nesterov=cfg.nesterov,
            betas=cfg.betas, eps=cfg.eps, sinkhorn_eps=cfg.sinkhorn_eps,
            weight_decay=cfg.weight_decay),
    )


# ---- SOAP ---------------------------------------------------------------
@dataclass(frozen=True)
class SoapConfig:
    lr: float = 3e-3
    betas: tuple = (0.95, 0.95)
    shampoo_beta: float = -1.0
    eps: float = 1e-8
    weight_decay: float = 0.01
    precondition_frequency: int = 10
    max_precond_dim: int = 10000
    merge_dims: bool = False
    precondition_1d: bool = False
    correct_bias: bool = True
    cautious_wd: bool = False


def _soap_groups(named, cfg):
    # Default: one group of all trainable params. Trainer-specific groupings
    # (e.g. train_joint's encoder/decoder split) are passed as pre-built
    # ``extra_groups`` in Phase 4; the spec's default is a single group.
    return [{"params": [q for _n, q in named if q.requires_grad]}]


@register_ingredient
def _soap():
    return IngredientSpec(
        name="soap", kind="optimizer", config_cls=SoapConfig,
        build_param_groups=_soap_groups,
        construct=lambda groups, cfg: SOAP(
            groups, lr=cfg.lr, betas=cfg.betas, shampoo_beta=cfg.shampoo_beta,
            eps=cfg.eps, weight_decay=cfg.weight_decay,
            precondition_frequency=cfg.precondition_frequency,
            max_precond_dim=cfg.max_precond_dim, merge_dims=cfg.merge_dims,
            precondition_1d=cfg.precondition_1d, correct_bias=cfg.correct_bias,
            cautious_wd=cfg.cautious_wd),
    )


# ---- Muon ---------------------------------------------------------------
@dataclass(frozen=True)
class MuonConfig:
    lr: float = 0.02
    momentum: float = 0.95
    weight_decay: float = 0.0
    adamw_lr: float = 1e-3
    adamw_betas: tuple = (0.95, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.0


def _muon_groups(named, cfg):
    # Muon routes 2-D+ tensors through Muon, 1-D through AdamW
    # (mirrors muon_optimizer.split_params_for_muon + train_joint's group dicts).
    named = list(named)  # safe to iterate twice even if called with a generator
    muon_p = [q for _n, q in named if q.requires_grad and q.ndim >= 2]
    adamw_p = [q for _n, q in named if q.requires_grad and q.ndim < 2]
    return [
        dict(params=muon_p, lr=cfg.lr, momentum=cfg.momentum,
             weight_decay=cfg.weight_decay, use_muon=True),
        dict(params=adamw_p, lr=cfg.adamw_lr, betas=cfg.adamw_betas,
             eps=cfg.adamw_eps, weight_decay=cfg.adamw_weight_decay,
             use_muon=False),
    ]


@register_ingredient
def _muon():
    return IngredientSpec(
        name="muon", kind="optimizer", config_cls=MuonConfig,
        build_param_groups=_muon_groups,
        construct=lambda groups, cfg: Muon(groups),
    )


# ---- AdamW --------------------------------------------------------------
@dataclass(frozen=True)
class AdamwConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)  # torch AdamW's own default (least surprise)
    fused: bool = False


@register_ingredient
def _adamw():
    return IngredientSpec(
        name="adamw", kind="optimizer", config_cls=AdamwConfig,
        build_param_groups=lambda named, cfg: [
            {"params": [q for _n, q in named if q.requires_grad]}],
        construct=lambda groups, cfg: torch.optim.AdamW(
            groups, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas,
            fused=cfg.fused),
    )
