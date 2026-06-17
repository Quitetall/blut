"""Tests for the sampler ingredient registry (ADR 0050/0051).

Two ``kind="sampler"`` specs, both ``cache_relevant=True``:

  * ``span_mask``  — ``snn/pretrain_ssl_tueg.py``'s ``make_span_mask``.
  * ``patch_mask`` — ``student/pretrain_mae.py``'s ``create_mask``.

Equivalence is proven against a VERBATIM inline copy of the trainer function:
same seed -> tensor-equal mask (pure torch, no wheels — runs on bare CPU CI).
"""
from __future__ import annotations

import torch

# Importing this module registers the sampler specs.
import lamquant.ingredients.sampler._specs  # noqa: F401
from lamquant.ingredients import build_ingredient, get_spec, list_ingredients


# ===========================================================================
# Registration + spec contract.
# ===========================================================================

def test_both_sampler_specs_registered():
    names = list_ingredients("sampler")
    assert "span_mask" in names
    assert "patch_mask" in names


def test_sampler_specs_are_cache_relevant():
    for n in ("span_mask", "patch_mask"):
        assert get_spec("sampler", n).cache_relevant is True


def test_unknown_key_fails_closed():
    import pytest
    with pytest.raises(ValueError):
        build_ingredient("sampler", "span_mask", {"not_a_field": 1})


# ===========================================================================
# (1) span_mask — verbatim copy of make_span_mask from pretrain_ssl_tueg.
# ===========================================================================

def _inline_make_span_mask(B, T, mask_frac, mean_span, generator, device):
    """Verbatim copy of snn/pretrain_ssl_tueg.py::make_span_mask."""
    assert isinstance(B, int) and B > 0
    assert isinstance(T, int) and T > 0
    assert 0.0 < mask_frac < 1.0
    assert isinstance(mean_span, int) and mean_span >= 1
    target_masked = max(1, int(round(mask_frac * T)))
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        n_masked = 0
        for _ in range(4 * T):
            if n_masked >= target_masked:
                break
            hi = 2 * mean_span - 1 if mean_span > 1 else 1
            span = int(torch.randint(1, hi + 1, (1,), generator=generator,
                                     device=device).item())
            start = int(torch.randint(0, T, (1,), generator=generator,
                                      device=device).item())
            end = min(start + span, T)
            newly = (~mask[b, start:end]).sum().item()
            mask[b, start:end] = True
            n_masked += int(newly)
        if not bool(mask[b].any()):
            j = int(torch.randint(0, T, (1,), generator=generator,
                                  device=device).item())
            mask[b, j] = True
    return mask


def test_span_mask_equals_inline():
    B, T = 4, 313
    dev = torch.device("cpu")
    make = build_ingredient(
        "sampler", "span_mask", {"mask_frac": 0.5, "mean_span": 10})
    g1 = torch.Generator(device="cpu"); g1.manual_seed(1234)
    g2 = torch.Generator(device="cpu"); g2.manual_seed(1234)
    got = make(B, T, g1, dev)
    exp = _inline_make_span_mask(B, T, 0.5, 10, g2, dev)
    assert torch.equal(got, exp)
    assert got.shape == (B, T) and got.dtype == torch.bool
    assert bool(got.any())


def test_span_mask_respects_cfg_hyperparams():
    # A higher mask_frac yields strictly more masked positions (the cfg field
    # is load-bearing, not a default the trainer secretly overrides).
    dev = torch.device("cpu")
    g = torch.Generator(device="cpu"); g.manual_seed(7)
    lo = build_ingredient("sampler", "span_mask",
                          {"mask_frac": 0.2, "mean_span": 5})(8, 200, g, dev)
    g.manual_seed(7)
    hi = build_ingredient("sampler", "span_mask",
                          {"mask_frac": 0.8, "mean_span": 5})(8, 200, g, dev)
    assert int(hi.sum()) > int(lo.sum())


def test_span_mask_guarantees_one_masked_per_row():
    dev = torch.device("cpu")
    g = torch.Generator(device="cpu"); g.manual_seed(99)
    m = build_ingredient("sampler", "span_mask",
                         {"mask_frac": 0.5, "mean_span": 10})(6, 313, g, dev)
    assert bool(m.any(dim=1).all())  # every row has >=1 masked step.


# ===========================================================================
# (2) patch_mask — verbatim copy of create_mask from pretrain_mae.
# ===========================================================================

def _inline_create_mask(batch_size, n_channels, l3_len, mask_ratio, patch_size,
                        device):
    """Verbatim copy of student/pretrain_mae.py::create_mask."""
    n_patches = l3_len // patch_size
    n_masked = int(n_patches * mask_ratio)
    mask = torch.zeros(batch_size, 1, l3_len, device=device)
    for b in range(batch_size):
        masked_idx = torch.randperm(n_patches, device=device)[:n_masked]
        for idx in masked_idx:
            start = idx * patch_size
            end = min(start + patch_size, l3_len)
            mask[b, :, start:end] = 1.0
    return mask.expand(-1, n_channels, -1)


def test_patch_mask_equals_inline():
    B, C, T = 4, 21, 313
    dev = torch.device("cpu")
    make = build_ingredient(
        "sampler", "patch_mask", {"mask_ratio": 0.5, "patch_size": 16})
    torch.manual_seed(2024)
    got = make(B, C, T, dev)
    torch.manual_seed(2024)
    exp = _inline_create_mask(B, C, T, 0.5, 16, dev)
    assert torch.equal(got, exp)
    assert got.shape == (B, C, T)
    # All channels share the same mask within a sample (the .expand contract).
    assert torch.equal(got[:, 0, :], got[:, 1, :])


def test_patch_mask_respects_cfg_hyperparams():
    B, C, T = 4, 21, 320
    dev = torch.device("cpu")
    torch.manual_seed(5)
    lo = build_ingredient("sampler", "patch_mask",
                          {"mask_ratio": 0.25, "patch_size": 16})(B, C, T, dev)
    torch.manual_seed(5)
    hi = build_ingredient("sampler", "patch_mask",
                          {"mask_ratio": 0.75, "patch_size": 16})(B, C, T, dev)
    assert float(hi.sum()) > float(lo.sum())
