"""Unit + equivalence tests for the optimizer ingredient registry (ADR 0050/0051).

These run without `lamquant_neural`: they exercise `build_ingredient` on tiny
synthetic modules, so the routing logic is pinned in-env. The full
trainer-vs-registry loss-trajectory equivalence lands in Phase 4 (needs the
model-definition wheels).
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from lamquant.ingredients import build_ingredient, list_ingredients
from lamquant.ingredients.registry import _REGISTRY, register_ingredient
from lamquant.ingredients.spec import IngredientSpec

pytestmark = pytest.mark.l2


class _Routed(nn.Module):
    """Param names chosen to exercise the ESOAP/SinkSOAPH suffix routing:
    four 2-D ``*.{in,x,out}_proj.weight`` / ``spatial_mix.weight`` matrices
    (→ preconditioned group) plus a 1-D bias, a non-suffix 2-D ``fc.weight``,
    and a bare 1-D parameter (→ AdamW group).
    """

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(8, 8, bias=True)        # weight→routed, bias→adamw
        self.x_proj = nn.Linear(8, 8, bias=False)        # weight→routed
        self.out_proj = nn.Linear(8, 8, bias=False)      # weight→routed
        self.spatial_mix = nn.Linear(8, 8, bias=False)   # weight→routed
        self.fc = nn.Linear(8, 8, bias=False)            # 2-D non-suffix→adamw
        self.scale = nn.Parameter(torch.ones(8))         # 1-D→adamw


def _routed_expected(m):
    return {id(m.in_proj.weight), id(m.x_proj.weight),
            id(m.out_proj.weight), id(m.spatial_mix.weight)}


def test_five_optimizers_registered():
    assert {"adamw", "esoap", "muon", "sinksoaph", "soap"}.issubset(
        set(list_ingredients("optimizer")))


def test_esoap_routing_matches_handrolled():
    """The spec's routing must be byte-for-byte the trainers' hand-rolled
    suffix filter (the ADR 0050 dedup proof)."""
    m = _Routed()
    named = list(m.named_parameters())
    suffixes = ("in_proj.weight", "x_proj.weight",
                "out_proj.weight", "spatial_mix.weight")
    hr_routed = {id(q) for n, q in named
                 if q.requires_grad and q.ndim == 2 and n.endswith(suffixes)}
    hr_rest = {id(q) for n, q in named
               if q.requires_grad and not (q.ndim == 2 and n.endswith(suffixes))}

    opt = build_ingredient("optimizer", "esoap", {"lr": 1e-3}, named_params=named)
    routed = next(g for g in opt.param_groups if g.get("method") == "esoap")
    rest = next(g for g in opt.param_groups if g.get("method") == "adamw")
    assert {id(p) for p in routed["params"]} == hr_routed == _routed_expected(m)
    assert {id(p) for p in rest["params"]} == hr_rest


def test_sinksoaph_same_routing_other_method_label():
    m = _Routed()
    opt = build_ingredient("optimizer", "sinksoaph", {"lr": 1e-3},
                           named_params=list(m.named_parameters()))
    routed = next(g for g in opt.param_groups if g.get("method") == "sinksoaph")
    assert {id(p) for p in routed["params"]} == _routed_expected(m)


def test_adamw_one_group_all_trainable():
    m = _Routed()
    named = list(m.named_parameters())
    opt = build_ingredient("optimizer", "adamw", {"lr": 1e-3}, named_params=named)
    assert isinstance(opt, torch.optim.AdamW)
    got = {id(p) for g in opt.param_groups for p in g["params"]}
    assert got == {id(q) for _n, q in named if q.requires_grad}


def test_muon_splits_by_ndim():
    m = _Routed()
    named = list(m.named_parameters())
    opt = build_ingredient("optimizer", "muon", {}, named_params=named)
    muon_g = next(g for g in opt.param_groups if g.get("use_muon") is True)
    adamw_g = next(g for g in opt.param_groups if g.get("use_muon") is False)
    assert {id(p) for p in muon_g["params"]} == {
        id(q) for _n, q in named if q.requires_grad and q.ndim >= 2}
    assert {id(p) for p in adamw_g["params"]} == {
        id(q) for _n, q in named if q.requires_grad and q.ndim < 2}


def test_soap_single_group_all_params():
    m = _Routed()
    named = list(m.named_parameters())
    opt = build_ingredient("optimizer", "soap", {"lr": 1e-3}, named_params=named)
    got = {id(p) for g in opt.param_groups for p in g["params"]}
    assert got == {id(q) for _n, q in named if q.requires_grad}


def test_extra_groups_appended_after_routing():
    """The distiller's student_proj (4state ADR-0027 #3) rides in as extra_groups."""
    m = _Routed()
    extra = nn.Linear(4, 4)
    opt = build_ingredient(
        "optimizer", "esoap", {"lr": 1e-3},
        named_params=list(m.named_parameters()),
        extra_groups=[{"params": list(extra.parameters())}])
    all_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(extra.weight) in all_ids and id(extra.bias) in all_ids


def test_unknown_optimizer_fails_closed():
    m = nn.Linear(3, 3)
    with pytest.raises(KeyError, match="unknown optimizer"):
        build_ingredient("optimizer", "nope", {},
                         named_params=list(m.named_parameters()))


def test_bad_config_field_fails_closed():
    m = nn.Linear(3, 3)
    with pytest.raises(ValueError, match="invalid config"):
        build_ingredient("optimizer", "esoap", {"not_a_field": 1},
                         named_params=list(m.named_parameters()))


def test_missing_named_params_raises():
    with pytest.raises(TypeError, match="named_params"):
        build_ingredient("optimizer", "esoap", {})


def test_license_gate_fails_closed_then_opens(monkeypatch):
    @dataclass(frozen=True)
    class _GateCfg:
        lr: float = 1e-3

    def _factory():
        return IngredientSpec(
            name="_gated_test", kind="optimizer", config_cls=_GateCfg,
            requires=("license:testgate",),
            build_param_groups=lambda named, cfg: [
                {"params": [q for _n, q in named if q.requires_grad]}],
            construct=lambda groups, cfg: torch.optim.SGD(groups, lr=cfg.lr),
        )

    register_ingredient(_factory)
    m = nn.Linear(3, 3)
    try:
        with pytest.raises(RuntimeError, match="license-gated"):
            build_ingredient("optimizer", "_gated_test", {},
                             named_params=list(m.named_parameters()))
        # monkeypatch auto-restores the env on teardown (xdist-safe).
        monkeypatch.setenv("LAMU_LICENSE_TESTGATE", "1")
        opt = build_ingredient("optimizer", "_gated_test", {},
                               named_params=list(m.named_parameters()))
        assert isinstance(opt, torch.optim.SGD)
    finally:
        _REGISTRY.pop(("optimizer", "_gated_test"), None)


def test_bad_kind_rejected():
    @dataclass(frozen=True)
    class _C:
        x: int = 0
    with pytest.raises(ValueError, match="unknown ingredient kind"):
        IngredientSpec(name="x", kind="bogus", config_cls=_C, build=lambda c: None)
