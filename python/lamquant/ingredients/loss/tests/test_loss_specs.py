"""Tests for the loss ingredient registry (ADR 0050/0051).

Each loss spec is pinned against a VERBATIM inline copy of the trainer code it
was extracted from (the gold-standard equivalence proof), plus the
registration + fail-closed config checks. All synthetic, CPU-only — no model
wheel needed (the metric primitives + auraloss import without it).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# Import the module directly so the specs register even if the package __init__
# has not been wired to import it yet (this is Phase 4 — wiring lands later).
import lamquant.ingredients.loss._specs as _loss_specs  # noqa: F401
from lamquant.ingredients import build_ingredient, list_ingredients

from lamquant.common.metrics import (
    masked_pearson_r_torch,
    masked_prd_torch,
    per_band_relative_loss as _per_band_rel,
)

pytestmark = pytest.mark.l2

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def test_all_loss_specs_registered():
    names = set(list_ingredients("loss"))
    assert names == {
        "joint_codec", "four_state_objective",
        "masked_recon_mse_time", "masked_recon_mse_patch", "teacher_mse",
    }


def test_joint_codec_bad_kind_fails_closed():
    with pytest.raises(ValueError):
        build_ingredient("loss", "joint_codec",
                         {"asymmetric_kind": "nope"}, device=DEVICE)


# ---------------------------------------------------------------------------
# (1) joint_codec — verbatim inline reference, both domains
# ---------------------------------------------------------------------------

def _joint_loss_inline(recon, l3_target, fullband_target, ch_mask,
                       *, device, spectral_loss, R_W, SP_W, PRD_W, ASYM_W,
                       BAND_W, asym_fn, return_parts=True):
    """Verbatim copy of train_joint.joint_loss — keep in sync with _specs.py."""
    if fullband_target is not None and abs(
            recon.shape[-1] - fullband_target.shape[-1]) <= 8:
        target = fullband_target
        domain = 'fullband'
    else:
        target = l3_target
        domain = 'l3'
    T = min(recon.shape[-1], target.shape[-1])
    recon_c = recon[..., :T]
    target_c = target[..., :T]
    l_mse = F.mse_loss(recon_c, target_c)
    l_r = 1.0 - masked_pearson_r_torch(recon_c, target_c, ch_mask)
    l_prd = masked_prd_torch(target_c, recon_c, ch_mask) / 100.0 if PRD_W > 0 else 0.0
    l_asym = asym_fn(target_c, recon_c) if ASYM_W > 0 else 0.0
    if SP_W > 0:
        with torch.amp.autocast(device_type=device.type, enabled=False):
            l_sp = spectral_loss(recon_c.float(), target_c.float())
    else:
        l_sp = 0.0
    if BAND_W > 0 and domain == 'fullband':
        with torch.amp.autocast(device_type=device.type, enabled=False):
            l_band = _per_band_rel(recon_c.float(), target_c.float(), fs=250.0)
    else:
        l_band = 0.0
    total = (l_mse + R_W * l_r + PRD_W * l_prd + SP_W * l_sp
             + ASYM_W * l_asym + BAND_W * l_band)
    if not return_parts:
        return total, None
    return total, {
        'mse': l_mse.detach().item(),
        'r_loss': l_r.detach().item(),
        'prd_loss': (l_prd.detach().item()
                     if isinstance(l_prd, torch.Tensor) else l_prd),
        'spectral': (l_sp.detach().item()
                     if isinstance(l_sp, torch.Tensor) else l_sp),
        'asym': (l_asym.detach().item()
                 if isinstance(l_asym, torch.Tensor) else l_asym),
        'band': (l_band.detach().item()
                 if isinstance(l_band, torch.Tensor) else l_band),
        'loss_domain': domain,
    }


def _spectral_ref(device):
    from auraloss.freq import MultiResolutionSTFTLoss
    fft_sizes = [64, 128, 256, 512]
    return MultiResolutionSTFTLoss(
        fft_sizes=fft_sizes,
        hop_sizes=[max(n // 4, 1) for n in fft_sizes],
        win_lengths=fft_sizes,
    ).to(device)


def _assert_joint_equiv(loss_fn, recon, l3_target, fullband_target, *,
                        device, R_W, SP_W, PRD_W, ASYM_W, BAND_W, asym_fn):
    spectral_ref = _spectral_ref(device)
    tot_ing, parts_ing = loss_fn(recon, l3_target,
                                 fullband_target=fullband_target,
                                 return_parts=True)
    tot_ref, parts_ref = _joint_loss_inline(
        recon, l3_target, fullband_target, None, device=device,
        spectral_loss=spectral_ref, R_W=R_W, SP_W=SP_W, PRD_W=PRD_W,
        ASYM_W=ASYM_W, BAND_W=BAND_W, asym_fn=asym_fn, return_parts=True)
    assert torch.allclose(tot_ing, tot_ref, atol=1e-6, rtol=1e-6)
    assert parts_ing['loss_domain'] == parts_ref['loss_domain']
    for k in ('mse', 'r_loss', 'prd_loss', 'spectral', 'asym', 'band'):
        assert abs(parts_ing[k] - parts_ref[k]) < 1e-6, (k, parts_ing[k], parts_ref[k])


def test_joint_codec_l3_domain_equiv():
    pytest.importorskip("auraloss")
    torch.manual_seed(0)
    recon = torch.randn(2, 4, 313, requires_grad=True)
    l3 = torch.randn(2, 4, 313)
    loss_fn = build_ingredient("loss", "joint_codec", {}, device=DEVICE)
    # defaults: R=0.5, SP=0.1, PRD=0.1, ASYM=0.0, BAND=0.5, kind=envelope
    _assert_joint_equiv(
        loss_fn, recon, l3, None, device=DEVICE,
        R_W=0.5, SP_W=0.1, PRD_W=0.1, ASYM_W=0.0, BAND_W=0.5, asym_fn=None)


def test_joint_codec_fullband_domain_equiv():
    pytest.importorskip("auraloss")
    torch.manual_seed(1)
    recon = torch.randn(2, 4, 2500, requires_grad=True)
    l3 = torch.randn(2, 4, 313)
    fullband = torch.randn(2, 4, 2500)
    loss_fn = build_ingredient("loss", "joint_codec", {}, device=DEVICE)
    _assert_joint_equiv(
        loss_fn, recon, l3, fullband, device=DEVICE,
        R_W=0.5, SP_W=0.1, PRD_W=0.1, ASYM_W=0.0, BAND_W=0.5, asym_fn=None)
    # band term is only non-zero in the fullband domain
    _, parts = loss_fn(recon, l3, fullband_target=fullband, return_parts=True)
    assert parts['loss_domain'] == 'fullband'
    assert parts['band'] != 0.0


def test_joint_codec_total_is_differentiable():
    pytest.importorskip("auraloss")
    recon = torch.randn(2, 4, 313, requires_grad=True)
    l3 = torch.randn(2, 4, 313)
    loss_fn = build_ingredient("loss", "joint_codec", {}, device=DEVICE)
    total, _ = loss_fn(recon, l3, return_parts=False)
    total.backward()
    assert recon.grad is not None and torch.isfinite(recon.grad).all()


# ---------------------------------------------------------------------------
# (2) four_state_objective — all three head branches + distill
# ---------------------------------------------------------------------------

class _StubCrfHead:
    def neg_log_likelihood(self, class_logits, target):
        # arbitrary differentiable scalar — distinct from the CE/ordinal paths
        return (class_logits.mean() - target.float().mean()).abs()


class _StubDistiller:
    def __init__(self):
        self.proj = torch.nn.Linear(8, 4)

    def distill_loss(self, activity_logits, teacher_feat):
        # mean over a projection — just needs to be differentiable & shaped
        return F.mse_loss(self.proj(activity_logits.mean(dim=-1)), teacher_feat)


def _four_state_inline(class_logits, target, *, head, head_kind, cw,
                       use_ordinal, crit_floor, lambda_spike, spike_rate,
                       lambda_distill, distiller, activity_logits, teacher_feat):
    from lamquant.snn.ordinal_loss import constrained_loss
    if head_kind == "crf":
        loss_main = head.neg_log_likelihood(class_logits, target)
    elif use_ordinal:
        loss_main = constrained_loss(class_logits, target, weight=cw,
                                     crit_floor=crit_floor)
    else:
        loss_main = F.cross_entropy(class_logits, target, weight=cw)
    loss = loss_main + lambda_spike * spike_rate
    if distiller is not None:
        loss = loss + lambda_distill * distiller.distill_loss(
            activity_logits, teacher_feat)
    return loss


def test_four_state_ce_branch():
    torch.manual_seed(2)
    B, C, T = 2, 4, 16
    logits = torch.randn(B, C, T, requires_grad=True)
    target = torch.randint(0, C, (B, T))
    cw = torch.ones(C)
    spike = torch.tensor(0.3)
    loss_fn = build_ingredient("loss", "four_state_objective",
                               {"use_ordinal": False, "lambda_spike": 0.01})
    out = loss_fn(logits, target, head=None, head_kind="softmax",
                  class_weights=cw, spike_rate=spike)
    ref = _four_state_inline(
        logits, target, head=None, head_kind="softmax", cw=cw,
        use_ordinal=False, crit_floor=0.88, lambda_spike=0.01,
        spike_rate=spike, lambda_distill=0.0, distiller=None,
        activity_logits=None, teacher_feat=None)
    assert torch.allclose(out, ref, atol=1e-6)
    out.backward()
    assert logits.grad is not None


def test_four_state_ordinal_branch():
    torch.manual_seed(3)
    B, C, T = 2, 4, 16
    logits = torch.randn(B, C, T, requires_grad=True)
    target = torch.randint(0, C, (B, T))
    cw = torch.ones(C)
    spike = torch.tensor(0.2)
    loss_fn = build_ingredient("loss", "four_state_objective",
                               {"use_ordinal": True, "crit_floor": 0.9,
                                "lambda_spike": 0.05})
    out = loss_fn(logits, target, head=None, head_kind="softmax",
                  class_weights=cw, spike_rate=spike)
    ref = _four_state_inline(
        logits, target, head=None, head_kind="softmax", cw=cw,
        use_ordinal=True, crit_floor=0.9, lambda_spike=0.05,
        spike_rate=spike, lambda_distill=0.0, distiller=None,
        activity_logits=None, teacher_feat=None)
    assert torch.allclose(out, ref, atol=1e-6)


def test_four_state_crf_branch():
    torch.manual_seed(4)
    B, C, T = 2, 4, 16
    logits = torch.randn(B, C, T)
    target = torch.randint(0, C, (B, T))
    cw = torch.ones(C)
    spike = torch.tensor(0.1)
    head = _StubCrfHead()
    loss_fn = build_ingredient("loss", "four_state_objective",
                               {"lambda_spike": 0.01})
    out = loss_fn(logits, target, head=head, head_kind="crf",
                  class_weights=cw, spike_rate=spike)
    ref = _four_state_inline(
        logits, target, head=head, head_kind="crf", cw=cw,
        use_ordinal=False, crit_floor=0.88, lambda_spike=0.01,
        spike_rate=spike, lambda_distill=0.0, distiller=None,
        activity_logits=None, teacher_feat=None)
    assert torch.allclose(out, ref, atol=1e-6)


def test_four_state_distill_adds_term():
    torch.manual_seed(5)
    B, C, T = 2, 4, 16
    logits = torch.randn(B, C, T, requires_grad=True)
    target = torch.randint(0, C, (B, T))
    cw = torch.ones(C)
    spike = torch.tensor(0.0)
    activity = torch.randn(B, 8, T)
    distiller = _StubDistiller()
    teacher_feat = torch.randn(B, 4)
    loss_fn = build_ingredient("loss", "four_state_objective",
                               {"lambda_spike": 0.0, "lambda_distill": 0.7})
    out = loss_fn(logits, target, head=None, head_kind="softmax",
                  class_weights=cw, spike_rate=spike, activity_logits=activity,
                  distiller=distiller, teacher_feat=teacher_feat)
    ref = _four_state_inline(
        logits, target, head=None, head_kind="softmax", cw=cw,
        use_ordinal=False, crit_floor=0.88, lambda_spike=0.0, spike_rate=spike,
        lambda_distill=0.7, distiller=distiller, activity_logits=activity,
        teacher_feat=teacher_feat)
    assert torch.allclose(out, ref, atol=1e-6)
    # the distill term actually contributes (vs no-distiller path)
    no_distill = loss_fn(logits, target, head=None, head_kind="softmax",
                         class_weights=cw, spike_rate=spike)
    assert not torch.allclose(out, no_distill)


# ---------------------------------------------------------------------------
# (3a/3b) masked recon variants — verbatim inline, NOT byte-equal to each other
# ---------------------------------------------------------------------------

def _ssl_masked_recon_inline(recon, target, mask):
    """Verbatim copy of pretrain_ssl_tueg.masked_recon_loss."""
    assert recon.shape == target.shape and recon.dim() == 3
    assert mask.shape == (recon.shape[0], recon.shape[2]), \
        f"mask must be [B,T], got {tuple(mask.shape)} for recon {tuple(recon.shape)}"
    m = mask.unsqueeze(1).to(recon.dtype)
    sq = (recon - target).pow(2) * m
    denom = m.sum() * recon.shape[1]
    assert denom.item() > 0, "no masked positions — empty SSL loss"
    return sq.sum() / denom


def test_masked_recon_mse_time_equiv():
    torch.manual_seed(6)
    recon = torch.randn(2, 21, 32, requires_grad=True)
    target = torch.randn(2, 21, 32)
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[:, 5:15] = True
    loss_fn = build_ingredient("loss", "masked_recon_mse_time", {})
    out = loss_fn(recon, target, mask)
    ref = _ssl_masked_recon_inline(recon, target, mask)
    assert torch.allclose(out, ref, atol=1e-6)
    out.backward()
    assert recon.grad is not None


def test_masked_recon_mse_time_empty_mask_asserts():
    recon = torch.randn(2, 21, 32)
    target = torch.randn(2, 21, 32)
    mask = torch.zeros(2, 32, dtype=torch.bool)  # nothing masked
    loss_fn = build_ingredient("loss", "masked_recon_mse_time", {})
    with pytest.raises(AssertionError):
        loss_fn(recon, target, mask)


def test_masked_recon_mse_patch_equiv():
    torch.manual_seed(7)
    recon = torch.randn(2, 21, 32, requires_grad=True)
    target = torch.randn(2, 21, 32)
    mask = torch.zeros(2, 1, 32)
    mask[:, :, 5:15] = 1.0
    loss_fn = build_ingredient("loss", "masked_recon_mse_patch", {})
    out = loss_fn(recon, target, mask)
    ref = F.mse_loss(recon * mask, target * mask)  # verbatim pretrain_mae
    assert torch.allclose(out, ref, atol=1e-6)
    out.backward()
    assert recon.grad is not None


def test_masked_recon_variants_not_byte_equal():
    """time (masked-mean) vs patch (all-mean) MUST differ on the same masked
    input — proves they are correctly kept as separate specs."""
    torch.manual_seed(8)
    recon = torch.randn(2, 21, 32)
    target = torch.randn(2, 21, 32)
    bool_mask = torch.zeros(2, 32, dtype=torch.bool)
    bool_mask[:, 5:15] = True
    float_mask = bool_mask.unsqueeze(1).float()  # [B,1,T] for the patch form

    time_fn = build_ingredient("loss", "masked_recon_mse_time", {})
    patch_fn = build_ingredient("loss", "masked_recon_mse_patch", {})
    t = time_fn(recon, target, bool_mask)
    p = patch_fn(recon, target, float_mask)
    # patch divides by full numel, time by masked-element count → different
    assert not torch.allclose(t, p)


# ---------------------------------------------------------------------------
# (4) teacher_mse — trivial
# ---------------------------------------------------------------------------

def test_teacher_mse_equiv():
    torch.manual_seed(9)
    recon = torch.randn(2, 21, 313, requires_grad=True)
    target = torch.randn(2, 21, 313)
    loss_fn = build_ingredient("loss", "teacher_mse", {})
    out = loss_fn(recon, target)
    assert torch.allclose(out, F.mse_loss(recon, target), atol=1e-7)
    out.backward()
    assert recon.grad is not None
