"""Tests for the model ingredient registry (ADR 0050/0051).

Three ``kind="model"`` specs, all ``cache_relevant=True``:

  * ``l3_teacher``  — ``oracle/train_l3_teacher.py``'s ``L3Teacher``.
  * ``mae_encoder`` — ``student/pretrain_mae.py``'s encoder + MAE prediction head.
  * ``joint_codec`` — ``student/train_joint.py``'s ``build_default_joint`` codec.

Equivalence is proven against the INLINE construction the trainers do, with the
SAME kwargs under a fixed seed: identical parameter count + tensor-equal params.
The construction tests need the neural/codec wheels; they skip cleanly when a
wheel is absent. Registration/contract tests are wheel-free.
"""
from __future__ import annotations

import pytest
import torch

# Importing this module registers the model specs.
import lamquant.ingredients.model._specs  # noqa: F401
from lamquant.ingredients import build_ingredient, get_spec, list_ingredients


def _n_params(m):
    return sum(p.numel() for p in m.parameters())


def _params_equal(a, b):
    pa = list(a.parameters())
    pb = list(b.parameters())
    if len(pa) != len(pb):
        return False
    return all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(pa, pb))


# ===========================================================================
# Registration + spec contract (no wheels needed).
# ===========================================================================

def test_all_three_model_specs_registered():
    names = list_ingredients("model")
    assert {"l3_teacher", "mae_encoder", "joint_codec"} <= set(names)


def test_model_specs_are_cache_relevant():
    for n in ("l3_teacher", "mae_encoder", "joint_codec"):
        assert get_spec("model", n).cache_relevant is True


def test_unknown_key_fails_closed():
    with pytest.raises(ValueError):
        build_ingredient("model", "l3_teacher", {"not_a_field": 1})


# ===========================================================================
# (1) l3_teacher — equals the inline L3Teacher(width=...) construction.
# ===========================================================================

def _has(modpath):
    import importlib
    try:
        importlib.import_module(modpath)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has("lamquant.oracle.train_teacher"),
                    reason="needs the L3Teacher module (train_teacher).")
def test_l3_teacher_equals_inline():
    from lamquant.oracle.train_teacher import L3Teacher
    torch.manual_seed(0)
    got = build_ingredient("model", "l3_teacher", {"width": 128})
    torch.manual_seed(0)
    inline = L3Teacher(width=128)
    assert _n_params(got) == _n_params(inline)
    assert _params_equal(got, inline)
    # Default width mirrors the trainer's --width default (512).
    assert get_spec("model", "l3_teacher").config_cls().width == 512


@pytest.mark.skipif(not _has("lamquant.oracle.train_teacher"),
                    reason="needs the L3Teacher module (train_teacher).")
def test_l3_teacher_device_applied():
    got = build_ingredient("model", "l3_teacher", {"width": 64},
                           device=torch.device("cpu"))
    assert next(got.parameters()).device.type == "cpu"


# ===========================================================================
# (2) mae_encoder — equals the inline encoder + pred-head construction.
# ===========================================================================

@pytest.mark.skipif(not _has("lamquant_neural.models.encoder"),
                    reason="needs the neural wheel (encoder).")
def test_mae_encoder_equals_inline():
    from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
    from lamquant.student.pretrain_mae import MAEPredictionHead
    torch.manual_seed(0)
    out = build_ingredient("model", "mae_encoder",
                           {"in_ch": 21, "latent_dim": 32})
    enc, head = out["encoder"], out["pred_head"]
    torch.manual_seed(0)
    inline_enc = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32)
    inline_head = MAEPredictionHead(latent_dim=32)
    assert _n_params(enc) == _n_params(inline_enc)
    assert _n_params(head) == _n_params(inline_head)
    assert _params_equal(enc, inline_enc)
    assert _params_equal(head, inline_head)


@pytest.mark.skipif(not _has("lamquant_neural.models.encoder"),
                    reason="needs the neural wheel (encoder).")
def test_mae_encoder_returns_dict_pair():
    out = build_ingredient("model", "mae_encoder", {})
    assert set(out) == {"encoder", "pred_head"}


# ===========================================================================
# (3) joint_codec — equals the inline build_default_joint(...) construction.
# ===========================================================================

@pytest.mark.skipif(not _has("lamquant.student.joint_codec"),
                    reason="needs the codec wheel (joint_codec).")
def test_joint_codec_equals_inline():
    from lamquant.student.joint_codec import build_default_joint
    kw = dict(latent_dim=32, encoder_width=128, vocos_tier=3, in_channels=21,
              decoder_channels=21, gradient_checkpointing=False,
              encoder_blocks=3, encoder_kernels=(3, 5, 7),
              channel_agnostic=False, ca_decoder=False)
    torch.manual_seed(0)
    got = build_ingredient("model", "joint_codec", dict(kw))
    torch.manual_seed(0)
    inline = build_default_joint(**kw)
    assert _n_params(got) == _n_params(inline)
    assert _params_equal(got, inline)
    # Encoder + decoder are both present (the JointCodec contract train_joint uses).
    assert hasattr(got, "encoder") and hasattr(got, "decoder")
