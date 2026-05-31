# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# Unit + smoke tests for distill_teacher — the LaBraM foundation-teacher
# distillation module (ADR 0027 BUILD #3).
#
# The pure-tensor tests (adapter shape, projection, loss math, channel map)
# run WITHOUT loading the 97 MB checkpoint. The full teacher load + forward is
# a single `@pytest.mark.slow` smoke that is skipped automatically if the
# checkpoint or `timm` is unavailable, so CI without the vendored weights stays
# green.

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "lamquant" / "snn"))

from lamquant.snn.distill_teacher import (  # noqa: E402
    l3_to_labram_input,
    _get_input_chans,
    _default_labram_repo,
    LAMQUANT_CH21,
    LABRAM_EMBED_DIM,
    LABRAM_PATCH_SIZE,
    STUDENT_GROUPS,
    L3_CHANNELS,
    L3_T,
    TEACHER_NAMES,
    RECOMMENDED_LAMBDA_DISTILL,
    TeacherDistiller,
)


# ----------------------------------------------------------------------
# Pure-tensor tests (no checkpoint).
# ----------------------------------------------------------------------

def test_channel_map_all_present_and_ordered():
    """Every LamQuant channel maps to a LaBraM pos_embed row; row 0 is cls."""
    ic = _get_input_chans(LAMQUANT_CH21)
    assert ic[0] == 0, "first input_chan must be the cls-token row 0"
    assert len(ic) == len(LAMQUANT_CH21) + 1, "one row per channel + cls"
    assert all(isinstance(i, int) and i > 0 for i in ic[1:]), \
        "channel rows must be positive ints"
    assert len(set(ic[1:])) == len(ic[1:]), "channel rows must be unique"


def test_channel_map_rejects_unknown_channel():
    """An electrode LaBraM never saw must raise, not silently fall back."""
    with pytest.raises(ValueError, match="not in LaBraM"):
        _get_input_chans(("FP1", "NOT_A_REAL_CHANNEL"))


@pytest.mark.parametrize("n_patches", [1, 2, 3, 8])
def test_adapter_shape(n_patches):
    """L3 [B,21,313] -> [B,21,n_patches,200], batched + unbatched."""
    l3 = torch.randn(4, L3_CHANNELS, L3_T)
    out = l3_to_labram_input(l3, n_patches=n_patches)
    assert out.shape == (4, L3_CHANNELS, n_patches, LABRAM_PATCH_SIZE)
    assert out.dtype == torch.float32

    # Unbatched [21, 313] auto-batches to B=1.
    out1 = l3_to_labram_input(torch.randn(L3_CHANNELS, L3_T), n_patches=n_patches)
    assert out1.shape == (1, L3_CHANNELS, n_patches, LABRAM_PATCH_SIZE)


def test_adapter_zscore_is_finite_and_standardised():
    """Per-channel z-score keeps the adapter output finite + ~unit-scale."""
    l3 = torch.randn(2, L3_CHANNELS, L3_T) * 1e6 + 5.0  # crazy scale + offset
    out = l3_to_labram_input(l3, n_patches=2)
    assert torch.isfinite(out).all(), "adapter must not produce NaN/Inf"
    flat = out.reshape(2, L3_CHANNELS, -1)
    # Mean ~0 per channel after z-score (interpolation is linear so the mean of
    # the resampled signal is close to the original channel mean -> ~0).
    assert flat.mean(dim=2).abs().max() < 1e-1
    # Std ~1 per channel.
    assert (flat.std(dim=2) - 1.0).abs().max() < 0.2


def test_adapter_is_autograd_safe():
    """Gradients flow through the adapter (it's on the student path option)."""
    l3 = torch.randn(2, L3_CHANNELS, L3_T, requires_grad=True)
    out = l3_to_labram_input(l3, n_patches=2)
    out.sum().backward()
    assert l3.grad is not None and torch.isfinite(l3.grad).all()


def test_adapter_rejects_bad_shape():
    with pytest.raises(AssertionError):
        l3_to_labram_input(torch.randn(2, 7, L3_T))      # wrong channel count
    with pytest.raises(AssertionError):
        l3_to_labram_input(torch.randn(2, L3_CHANNELS, L3_T), n_patches=99)


# -- projection + loss math (no teacher needed; build a stub) -----------


def _stub_distiller():
    """A TeacherDistiller-shaped object WITHOUT loading LaBraM.

    We only need `project_student` + `distill_loss`, which depend on
    `student_proj`, `embed_dim`, `loss_kind`. Build a bare nn.Module with
    those attributes and bind the two methods.
    """
    import torch.nn as nn
    obj = nn.Module()
    obj.embed_dim = LABRAM_EMBED_DIM
    obj.loss_kind = "cosine"
    obj.student_proj = nn.Linear(STUDENT_GROUPS, LABRAM_EMBED_DIM)
    obj.project_student = TeacherDistiller.project_student.__get__(obj)
    obj.distill_loss = TeacherDistiller.distill_loss.__get__(obj)
    return obj


def test_project_student_pools_time_axis():
    """[B,8,T] pools to [B,8] then projects to [B,200]; [B,8] passes through."""
    d = _stub_distiller()
    p3 = d.project_student(torch.randn(5, STUDENT_GROUPS, 313))
    assert p3.shape == (5, LABRAM_EMBED_DIM)
    p2 = d.project_student(torch.randn(5, STUDENT_GROUPS))
    assert p2.shape == (5, LABRAM_EMBED_DIM)


def test_distill_loss_cosine_zero_for_aligned():
    """Cosine distill loss is ~0 when student projects exactly onto teacher."""
    d = _stub_distiller()
    teacher = torch.randn(6, LABRAM_EMBED_DIM)
    # Feed the teacher feature in directly as already-projected student feat.
    loss = d.distill_loss(teacher.clone(), teacher)
    assert loss.item() < 1e-5, f"aligned cosine loss should be ~0, got {loss.item()}"


def test_distill_loss_cosine_positive_for_misaligned():
    d = _stub_distiller()
    teacher = torch.randn(6, LABRAM_EMBED_DIM)
    student = torch.randn(6, LABRAM_EMBED_DIM)
    loss = d.distill_loss(student, teacher)
    assert 0.0 < loss.item() < 2.0


def test_distill_loss_mse_path():
    d = _stub_distiller()
    d.loss_kind = "mse"
    teacher = torch.randn(4, LABRAM_EMBED_DIM)
    loss = d.distill_loss(torch.randn(4, STUDENT_GROUPS, 313), teacher)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_distill_loss_grad_flows_to_student_only():
    """Loss backprops into student_proj; teacher_feat stays detached."""
    d = _stub_distiller()
    teacher = torch.randn(3, LABRAM_EMBED_DIM, requires_grad=True)
    student = torch.randn(3, STUDENT_GROUPS, 313, requires_grad=True)
    loss = d.distill_loss(student, teacher)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert d.student_proj.weight.grad is not None
    # teacher_feat is .detach()'d inside distill_loss -> no grad.
    assert teacher.grad is None


def test_recommended_weight_is_small_and_in_range():
    assert 0.0 < RECOMMENDED_LAMBDA_DISTILL <= 0.2
    assert "labram" in TEACHER_NAMES


# ----------------------------------------------------------------------
# Full teacher smoke — loads LaBraM-base; skipped if weights / timm absent.
# ----------------------------------------------------------------------

def _labram_available() -> bool:
    repo = _default_labram_repo()
    ckpt = repo / "checkpoints" / "labram-base.pth"
    if not ckpt.exists():
        return False
    try:
        import timm  # noqa: F401
        import einops  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _labram_available(),
                    reason="LaBraM checkpoint or timm/einops unavailable")
def test_teacher_forward_smoke():
    """Load LaBraM, run one forward on a synthetic [2,21,313] batch."""
    d = TeacherDistiller(teacher_name="labram")
    l3 = torch.randn(2, L3_CHANNELS, L3_T)
    feat = d.teacher_features(l3)
    assert feat.shape == (2, LABRAM_EMBED_DIM)
    assert feat.requires_grad is False, "teacher feature must be detached"

    # End-to-end distill loss with a synthetic student feature.
    student = torch.randn(2, STUDENT_GROUPS, L3_T, requires_grad=True)
    loss = d.distill_loss(student, feat)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
