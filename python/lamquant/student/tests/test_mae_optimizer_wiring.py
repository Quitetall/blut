"""Phase 4 equivalence: pretrain_mae's optimizer now comes from the ingredient
registry (ADR 0050/0051). Asserts build_ingredient's adamw covers all trainable
encoder+head params with the same hyperparameters the inline AdamW used, on the
REAL ternary encoder. Needs the model-definition wheel; skips when absent.
"""
from __future__ import annotations

import pytest

pytest.importorskip("lamquant_neural")

import torch  # noqa: E402

from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband  # noqa: E402

from lamquant.ingredients import build_ingredient  # noqa: E402

# Lives in the trainer module (student/ is on sys.path via conftest).
from pretrain_mae import MAEPredictionHead  # noqa: E402

pytestmark = pytest.mark.l2


def test_mae_adamw_covers_all_trainable_with_same_hyperparams():
    encoder = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32)
    pred_head = MAEPredictionHead(latent_dim=32)
    named = list(encoder.named_parameters()) + list(pred_head.named_parameters())
    # The cfg pretrain_mae passes (betas torch-default, fused off on CPU).
    cfg = {"lr": 3e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999), "fused": False}

    opt = build_ingredient("optimizer", "adamw", cfg, named_params=named)
    assert isinstance(opt, torch.optim.AdamW)
    # The optimizer trains exactly the trainable encoder+head params (the inline
    # AdamW passed all of them; frozen params — if any — never update either way).
    got = {id(p) for g in opt.param_groups for p in g["params"]}
    assert got == {id(q) for _n, q in named if q.requires_grad}
    g = opt.param_groups[0]
    assert g["lr"] == 3e-4
    assert g["weight_decay"] == 1e-4
    assert g["betas"] == (0.9, 0.999)
