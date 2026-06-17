"""Tests for the eval ingredient registry (ADR 0050/0051).

``joint_codec`` wraps ``validate_joint`` (the shared trainer + PCCP-gate eval);
``four_state`` bundles ``four_state_metrics`` + the lexicographic
``selection_key`` (ADR 0029 slide-killer). Both need the neural wheel, so the
heavy bodies importorskip ``lamquant_neural``.
"""
from __future__ import annotations

import pytest

# Importing this module registers the eval specs without going through the
# package __init__ (which the main agent wires up separately).
import lamquant.ingredients.evals._specs  # noqa: F401
from lamquant.ingredients import build_ingredient, get_spec, list_ingredients


# ----------------------------------------------------------------------
# Registration + spec contract (no wheel needed).
# ----------------------------------------------------------------------

def test_both_evals_registered():
    names = list_ingredients("eval")
    assert "joint_codec" in names
    assert "four_state" in names


def test_eval_specs_are_cache_irrelevant():
    # Running an eval must never change the trained artifact / stage cache key.
    assert get_spec("eval", "joint_codec").cache_relevant is False
    assert get_spec("eval", "four_state").cache_relevant is False


def test_joint_eval_config_defaults_mirror_validate_joint():
    spec = get_spec("eval", "joint_codec")
    cfg = spec.config_cls()
    assert cfg.quantize is True
    assert cfg.batch_size == 64
    assert cfg.per_band_sample == 512
    assert cfg.amp is True
    assert cfg.per_category is False
    assert cfg.channel_agnostic is False
    assert cfg.variable_n is False
    assert cfg.n_range == (8, 21)


def test_four_state_config_is_empty():
    import dataclasses
    cfg = get_spec("eval", "four_state").config_cls()
    assert dataclasses.fields(cfg) == ()


def test_unknown_joint_eval_key_fails_closed():
    with pytest.raises(ValueError):
        build_ingredient("eval", "joint_codec", {"not_a_real_field": 1})


# ----------------------------------------------------------------------
# four_state — frozen-metric tuple-equality against the wrapped functions.
# ----------------------------------------------------------------------

# Frozen metric dicts mirroring snn/tests/test_gate_selection.py: feasible,
# infeasible (the slide), and the boundary (CRIT_rec == alpha).
_ALPHA = 0.95
_FEASIBLE = {"critical_recall": 0.97, "quiet_specificity": 0.60}
_FEASIBLE_WORST = {"critical_recall": 0.95, "quiet_specificity": 0.0}
_INFEASIBLE_SLID = {"critical_recall": 0.49, "quiet_specificity": 0.85}
_INFEASIBLE_SAFE_LOWCR = {"critical_recall": 0.90, "quiet_specificity": 0.16}
_BOUNDARY = {"critical_recall": 0.95, "quiet_specificity": 0.30}


def test_four_state_select_key_matches_wrapped_selection_key():
    pytest.importorskip("lamquant_neural")
    from lamquant.snn.train_4state_controller import selection_key
    obj = build_ingredient("eval", "four_state", {})
    for m in (_FEASIBLE, _FEASIBLE_WORST, _INFEASIBLE_SLID,
              _INFEASIBLE_SAFE_LOWCR, _BOUNDARY):
        got = obj.select_key(m, _ALPHA)
        exp = selection_key(m, _ALPHA)
        # MUST be the lexicographic 2-tuple, byte-identical to the source fn.
        assert isinstance(got, tuple) and len(got) == 2
        assert got == exp


def test_four_state_select_key_is_a_tuple_never_a_scalar():
    # ADR 0029 slide-killer guard: the key must stay a (feasible, tiebreak)
    # tuple — collapsing to a scalar / max-val_r re-opens the slide.
    pytest.importorskip("lamquant_neural")
    obj = build_ingredient("eval", "four_state", {})
    k = obj.select_key(_FEASIBLE, _ALPHA)
    assert isinstance(k, tuple) and not isinstance(k, (int, float))
    assert k[0] in (0, 1)


def test_four_state_metrics_matches_wrapped_four_state_metrics():
    pytest.importorskip("lamquant_neural")
    import numpy as np
    from lamquant.snn.train_4state_controller import four_state_metrics
    obj = build_ingredient("eval", "four_state", {})
    # A non-trivial 4x4 confusion (true rows, pred cols).
    cm = np.array([
        [50, 3, 1, 0],
        [4, 40, 5, 1],
        [0, 6, 30, 4],
        [0, 1, 2, 25],
    ], dtype=np.int64)
    got = obj.metrics(cm)
    exp = four_state_metrics(cm)
    assert got == exp


# ----------------------------------------------------------------------
# joint_codec — the built eval_fn IS validate_joint; structural equality on a
# fake identity-decoder model + fake typed-batch val_ds.
# ----------------------------------------------------------------------

class _FakeBatch:
    """Minimal typed batch: L3 only, no fullband, leakage-check is a no-op."""

    def __init__(self, l3_approx):
        self.l3_approx = l3_approx
        self.fullband_target = None
        self.clinical_categories = None

    def assert_no_leakage(self, split):  # noqa: ARG002 — safety-net no-op
        return None


class _FakeValDS:
    """Yields the same 2 minimal typed batches on every iteration so two
    independent ``validate_joint`` passes produce byte-identical tuples."""

    def __init__(self, batches):
        self._batches = batches

    def prefetch_typed_batches(self, batch_size=64, device=None):  # noqa: ARG002
        import torch
        for b in self._batches:
            x = b.l3_approx
            if device is not None:
                x = x.to(device)
            yield _FakeBatch(x)


class _IdentityCodec:
    """model(x_l3, quantize=..., coords=..., ch_mask=...) -> x_l3 (R == 1.0)."""

    def eval(self):
        return self

    def __call__(self, x_l3, quantize=True, coords=None, ch_mask=None):  # noqa: ARG002
        return x_l3


def test_joint_codec_eval_equals_validate_joint():
    pytest.importorskip("lamquant_neural")
    import torch
    from lamquant.student.train_joint import validate_joint

    torch.manual_seed(0)
    b0 = _FakeBatch(torch.randn(4, 21, 64))
    b1 = _FakeBatch(torch.randn(3, 21, 64))
    model = _IdentityCodec()
    device = "cpu"

    eval_fn = build_ingredient("eval", "joint_codec", {"amp": False})
    got = eval_fn(model, _FakeValDS([b0, b1]), device)
    # The wrapped function called directly with the same defaults (amp=False to
    # keep CPU bf16 autocast out of the picture).
    exp = validate_joint(model, _FakeValDS([b0, b1]), device,
                         quantize=True, batch_size=64, per_band_sample=512,
                         amp=False, per_category=False,
                         channel_agnostic=False, variable_n=False,
                         n_range=(8, 21))

    assert got == exp
    # Identity decoder reconstructs the L3 input exactly → R == 1.0.
    val_r, val_prd, per_band = got
    assert val_r == pytest.approx(1.0, abs=1e-5)


def test_joint_codec_quantize_override_is_forwarded():
    pytest.importorskip("lamquant_neural")
    import torch
    from lamquant.student.train_joint import validate_joint

    torch.manual_seed(1)
    b0 = _FakeBatch(torch.randn(4, 21, 64))
    model = _IdentityCodec()

    # cfg pins quantize=True; the call-time override flips it to False and the
    # wrapped validate_joint must see quantize=False.
    eval_fn = build_ingredient("eval", "joint_codec",
                               {"quantize": True, "amp": False})
    got = eval_fn(model, _FakeValDS([b0]), "cpu", quantize=False)
    exp = validate_joint(model, _FakeValDS([b0]), "cpu",
                         quantize=False, batch_size=64, per_band_sample=512,
                         amp=False, per_category=False,
                         channel_agnostic=False, variable_n=False,
                         n_range=(8, 21))
    assert got == exp
