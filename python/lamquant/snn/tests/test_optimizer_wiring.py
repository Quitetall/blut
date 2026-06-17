"""Phase 4 equivalence: the SNN trainers' optimizer now comes from the
ingredient registry (ADR 0050/0051). These assert the registry routes a *real*
MambaSNN backbone + head exactly like the hand-rolled ESOAP suffix filter that
``pretrain_ssl_tueg`` / ``train_4state_controller`` used to inline — the dedup
proof on the production model, not a synthetic one.

Needs the private model-definition wheel; skips cleanly when it is absent.
"""
from __future__ import annotations

import pytest

pytest.importorskip("lamquant_neural")

import torch  # noqa: E402

from lamquant_neural.models.mamba_ssm_minimal import MambaSNN  # noqa: E402
from lamquant_neural.models.heads import build_head, HEAD_REGISTRY  # noqa: E402

from lamquant.ingredients import build_ingredient  # noqa: E402

pytestmark = pytest.mark.l2

# An INDEPENDENT transcription of _specs.py's _LINEAR_SUFFIXES — deliberately
# duplicated (not imported) so this test proves the registry matches the EXPECTED
# routing rather than matching itself (importing the same constant = tautology).
_SUFFIXES = ("in_proj.weight", "x_proj.weight",
             "out_proj.weight", "spatial_mix.weight")
# Nonzero weight_decay so the hyperparameter assertions exercise a real value.
_CFG = {"lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.95)}


def _tiny_model_and_head():
    model = MambaSNN(in_channels=21, d_model=32, d_state=16, n_layers=1,
                     use_subband=True)
    # Any registered head: this exercises optimizer routing, not head semantics —
    # the head's params land in the AdamW group regardless of which head it is.
    head = build_head(sorted(HEAD_REGISTRY)[0], K=4)
    return model, head


def _named(model, head):
    # Exactly the param list train_4state_controller builds.
    return list(model.named_parameters()) + \
        [(f"head.{n}", q) for n, q in head.named_parameters()]


def _handrolled(named):
    routed = {id(q) for n, q in named
              if q.requires_grad and q.ndim == 2 and n.endswith(_SUFFIXES)}
    rest = {id(q) for n, q in named
            if q.requires_grad and not (q.ndim == 2 and n.endswith(_SUFFIXES))}
    return routed, rest


def test_esoap_routing_matches_handrolled_on_real_model():
    model, head = _tiny_model_and_head()
    named = _named(model, head)
    routed, rest = _handrolled(named)
    assert routed, "real MambaSNN must expose the routed projection matrices"

    opt = build_ingredient("optimizer", "esoap", _CFG, named_params=named)
    g_routed = next(g for g in opt.param_groups if g.get("method") == "esoap")
    g_rest = next(g for g in opt.param_groups if g.get("method") == "adamw")
    assert {id(p) for p in g_routed["params"]} == routed
    assert {id(p) for p in g_rest["params"]} == rest
    # Hyperparameters match the old inline path (membership alone is not enough):
    # lr + betas come from the constructor defaults; weight_decay is set on every
    # group (the inline ESOAP path set it at both the group and constructor level).
    for g in opt.param_groups:
        assert g["lr"] == _CFG["lr"]
        assert g["betas"] == _CFG["betas"]
        assert g["weight_decay"] == _CFG["weight_decay"]


def test_extra_groups_none_is_noop():
    """distiller=None passes extra_groups=None — must construct cleanly (the
    non-distillation 4state path) with exactly the two routed groups."""
    model, head = _tiny_model_and_head()
    named = _named(model, head)
    opt = build_ingredient("optimizer", "esoap", _CFG, named_params=named,
                           extra_groups=None)
    assert len(opt.param_groups) == 2


def test_adamw_on_real_model_one_group():
    model, head = _tiny_model_and_head()
    named = _named(model, head)
    opt = build_ingredient("optimizer", "adamw", _CFG, named_params=named)
    assert isinstance(opt, torch.optim.AdamW)
    got = {id(p) for g in opt.param_groups for p in g["params"]}
    assert got == {id(q) for _n, q in named if q.requires_grad}


def test_distiller_extra_group_appended_as_adamw():
    """The 4state distiller's student_proj rides in via extra_groups (replacing
    the old post-construction add_param_group) and lands as an AdamW group."""
    model, head = _tiny_model_and_head()
    named = _named(model, head)
    student_proj = torch.nn.Linear(32, 32)  # a 2-D Linear without a routed suffix
    opt = build_ingredient(
        "optimizer", "esoap", _CFG, named_params=named,
        extra_groups=[{"params": list(student_proj.parameters())}])
    all_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(student_proj.weight) in all_ids
    assert id(student_proj.bias) in all_ids
    # student_proj.weight is 2-D but NOT a routed suffix → must be in an AdamW
    # group, never the ESOAP group (ESOAP would reject a non-routed 2-D only if
    # mislabelled; here it is correctly an adamw-method group).
    g_routed = next(g for g in opt.param_groups if g.get("method") == "esoap")
    assert id(student_proj.weight) not in {id(p) for p in g_routed["params"]}
