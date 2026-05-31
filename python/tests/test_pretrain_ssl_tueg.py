# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# test_pretrain_ssl_tueg.py — unit/smoke for the SSL backbone-pretrain module.
#
# Covers the load-bearing contracts WITHOUT the heavy LMA dataset:
#   1. make_span_mask: shape/dtype, coverage >= 1 masked step/row, mask_frac.
#   2. SSLReconstructor: forward shape [B,21,T] -> [B,21,T], masked-only MSE.
#   3. A few optimizer steps actually reduce the masked recon loss (it trains).
#   4. save_backbone writes ONLY MambaSNN.state_dict() keys (the key overlap
#      the trainer relies on for strict=False --init-backbone load), and the
#      saved dict loads back into a fresh MambaSNN with NO unexpected keys.
#
# Run: pytest -q tests/test_pretrain_ssl_tueg.py
# (needs SNN_SEIZURE_HEAD=linear for the load-compatible backbone shape).

from __future__ import annotations

import os
import sys

import torch

# Path plumbing mirrors the module under test.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYROOT = os.path.abspath(os.path.join(_HERE, ".."))   # blut/python
for _p in (_PYROOT, "/mnt/4tb/LamQuant/LamQuant-Neural"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Force the load-compatible single-Linear seizure head BEFORE importing the
# backbone (head kind is read from env in MambaSNN.__init__).
os.environ.setdefault("SNN_SEIZURE_HEAD", "linear")

from lamquant_neural.models.mamba_ssm_minimal import MambaSNN  # noqa: E402
from lamquant.snn.pretrain_ssl_tueg import (  # noqa: E402
    make_span_mask, SSLReconstructor, masked_recon_loss, save_backbone, L3_T,
    L3_CHANNELS,
)


def _backbone():
    return MambaSNN(in_channels=21, d_model=40, d_state=16, n_layers=2,
                    use_subband=True)


def test_span_mask_shape_and_coverage():
    gen = torch.Generator(device="cpu"); gen.manual_seed(0)
    B, T = 5, L3_T
    mask = make_span_mask(B, T, mask_frac=0.30, mean_span=10,
                          generator=gen, device=torch.device("cpu"))
    assert mask.shape == (B, T)
    assert mask.dtype == torch.bool
    # Every row has at least one masked step (no empty-loss batch).
    assert (mask.sum(dim=1) >= 1).all()
    # Overall coverage is in the right ballpark of mask_frac (not exact due to
    # span overlap + integer rounding, but should be substantial and < 1).
    frac = mask.float().mean().item()
    assert 0.10 < frac < 0.60, f"masked frac {frac} far from target 0.30"


def test_masked_loss_scores_masked_only():
    B, T = 2, L3_T
    recon = torch.zeros(B, L3_CHANNELS, T)
    target = torch.ones(B, L3_CHANNELS, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :10] = True            # only first 10 steps masked
    loss = masked_recon_loss(recon, target, mask)
    # (0-1)^2 == 1 everywhere, averaged over masked positions only -> 1.0.
    assert abs(loss.item() - 1.0) < 1e-6
    # Putting the error on UNMASKED positions must not be scored.
    recon2 = target.clone()
    recon2[:, :, 50:] = 999.0      # huge error, but all unmasked
    loss2 = masked_recon_loss(recon2, target, mask)
    assert loss2.item() < 1e-6


def test_forward_shape_and_trains():
    torch.manual_seed(0)
    ssl = SSLReconstructor(_backbone())
    B, T = 4, L3_T
    l3 = torch.randn(B, L3_CHANNELS, T)
    gen = torch.Generator(device="cpu"); gen.manual_seed(1)
    mask = make_span_mask(B, T, 0.30, 10, gen, torch.device("cpu"))
    x_masked = l3.clone()
    x_masked[mask.unsqueeze(1).expand(-1, L3_CHANNELS, -1)] = 0.0

    recon = ssl(x_masked)
    assert recon.shape == (B, L3_CHANNELS, T)

    opt = torch.optim.AdamW(ssl.parameters(), lr=1e-2)
    losses = []
    for _ in range(15):
        opt.zero_grad(set_to_none=True)
        loss = masked_recon_loss(ssl(x_masked), l3, mask)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # It learns: loss must drop meaningfully over a handful of steps.
    assert losses[-1] < losses[0] * 0.9, \
        f"masked recon did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_saved_keys_are_mambasnn_subset(tmp_path):
    ssl = SSLReconstructor(_backbone())
    out = str(tmp_path / "backbone_ssl.pt")
    save_backbone(ssl.backbone, out, {"epoch": 1, "masked_mse": 0.0})

    payload = torch.load(out, map_location="cpu", weights_only=False)
    saved = set(payload["backbone"].keys())

    ref = _backbone()
    ref_keys = set(ref.state_dict().keys())

    # No recon-head keys leaked; saved is a SUBSET of MambaSNN's keys.
    assert saved <= ref_keys, f"saved keys not subset: {sorted(saved - ref_keys)}"
    # seizure_head.* is intentionally DROPPED (not part of the SSL graph; its
    # shape depends on the controller's SNN_SEIZURE_HEAD env). Everything else
    # — spatial_mix, ssm_blocks, readout — is saved.
    dropped = {k for k in ref_keys if k.startswith("seizure_head.")}
    assert saved == (ref_keys - dropped), \
        f"saved != backbone-minus-seizure_head; diff: " \
        f"{sorted((ref_keys - dropped) ^ saved)}"
    assert dropped, "expected at least one seizure_head key in the reference"

    # The exact --init-backbone load the trainer runs: strict=False, expect
    # ZERO unexpected keys (the contract that makes the recipe clean for any
    # seizure-head config) and missing == the controller-only seizure head.
    missing, unexpected = ref.load_state_dict(payload["backbone"], strict=False)
    assert unexpected == [], f"unexpected keys on load: {unexpected}"
    assert set(missing) == dropped, \
        f"missing keys must be exactly the seizure head; got {sorted(missing)}"


if __name__ == "__main__":
    test_span_mask_shape_and_coverage()
    test_masked_loss_scores_masked_only()
    test_forward_shape_and_trains()
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        test_saved_keys_are_mambasnn_subset(pathlib.Path(d))
    print("ALL PASS")
