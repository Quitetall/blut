# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# distill_teacher.py — foundation-EEG-model feature distillation for the
# 4-state CR controller (ADR 0027, BUILD #3).
#
# ORACLE FINDING (drives this): QUIET/BASELINE are trivially separable by L3
# energy — the CR side is LOSS-limited and fixed by ordinal+constrained heads.
# CRITICAL/INTERESTING are *temporal*: per-timestep energy oracles fail them,
# the SSM beats them. RICHER TEMPORAL FEATURES (spectral, SSL, distill) push
# the event tiers. This module is the *distill* leg: it borrows the temporal /
# spectral representation a large self-supervised EEG foundation model has
# already learned from ~2500 h of EEG, and pulls the 57K-param student's
# backbone features toward it.
#
# Teacher: LaBraM-base (Large Brain Model, Jiang et al., ICLR 2024). It is the
# ONLY foundation teacher on disk with a real, loadable checkpoint AND full
# model source:
#
#     reference_software/labram/repo/checkpoints/labram-base.pth   (97 MB, zip)
#     reference_software/labram/repo/checkpoints/vqnsp.pth         (95 MB, zip)
#     reference_software/labram/repo/modeling_finetune.py          (model code)
#
# (FEMBA ships only ONE non-LFS checkpoint — TUAB/FEMBA_base.safetensors,
# 186 MB — but the other 9 weight files are git-LFS pointers, 133 B each, and
# its HF-style model code is split across the `code/` tree; EEGPT ships no
# weights on disk at all. LaBraM is the most loadable by a wide margin.)
#
# Why LaBraM is loadable here:
#   * `labram-base.pth` is a plain torch zip archive with a `model` state-dict
#     whose backbone keys are prefixed `student.` (the pretrain EMA student).
#   * Built with `use_mean_pooling=False`, the finetune `NeuralTransformer`
#     loads the backbone with ZERO missing keys (only `mask_token`,
#     `lm_head.{weight,bias}` are unexpected — pretrain-only heads we drop).
#   * Its module code only needs `timm` + `einops` (now installed in the
#     LamQuant-Neural venv). We deliberately do NOT import its `utils.py`
#     (which hard-requires `h5py`); the two things we need from it —
#     `standard_1020` and `get_input_chans` — are inlined below.
#
# -------------------------------------------------------------------------
# INPUT ADAPTER (L3 -> LaBraM), documented honestly
# -------------------------------------------------------------------------
# Our student sees L3 `[B, 21, 313]` — a level-3 DWT *latent* of a 10 s window,
# NOT raw EEG. LaBraM expects raw EEG `[B, n_ch, n_patches, 200]` at 200 Hz in
# µV, segmented into 1 s patches (patch_size=200). The adapter:
#
#   1. Treats each L3 channel's 313-length sequence as a 1-D signal and
#      linearly resamples the TIME axis 313 -> n_patches*200 (default
#      n_patches=2 -> 400), giving `[B, 21, n_patches, 200]`. 313 does not
#      divide 200, so a resample is unavoidable; linear interpolation keeps the
#      L3 temporal envelope (the thing the event tiers live in) and is cheap +
#      autograd-safe.
#   2. Per-channel z-scores the patched signal (zero mean / unit std over the
#      patch*sample axis). LaBraM was trained on µV-scaled EEG; our L3 is in
#      arbitrary Q31-derived units, so a per-channel standardisation is the
#      honest amplitude bridge (it is NOT a true µV calibration — see CAVEAT).
#   3. Maps our fixed 21-ch 10-20 montage to LaBraM's `pos_embed` rows via
#      `get_input_chans` (the SAME order the dataset emits).
#
# CAVEAT (load-bearing, not a silent fallback): this is a *domain-shifted*
# adapter. LaBraM never saw DWT-L3 latents or Q31 units in pretraining, so its
# embeddings here are an out-of-distribution read of our signal. The
# distillation target is therefore a *soft* regulariser (cosine, default
# weight 0.05), NOT a hard label — it nudges the student toward LaBraM's
# temporal feature geometry without overriding the supervised 4-state loss.
# If a future build wants an in-distribution teacher read, feed LaBraM the
# *raw* 250 Hz signal (resampled to 200 Hz) BEFORE the DWT — that signal IS
# available in the dataset's decode path (`_decode_and_preprocess`) — and swap
# `l3_to_labram_input` for a raw-EEG adapter. The loss + projection code below
# is unchanged by that swap.
#
# -------------------------------------------------------------------------
# DISTILLATION LOSS
# -------------------------------------------------------------------------
# The teacher emits one 200-d embedding per window (mean-pooled patch tokens).
# The student emits `activity_logits [B, 8, T]`; we pool it over T to a
# per-window `[B, 8]` summary, then a tiny learnable `nn.Linear(8 -> 200)`
# projects it into the teacher space. `distill_loss` matches the two with a
# cosine-distance (default) or MSE objective, to be ADDED to the supervised
# 4-state loss with `lambda_distill` (default 0.05).
#
# Programming-Bible style: SPDX header, contract assertions, no silent
# fallback, typed.

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Geometry + teacher contract constants — single source of truth.
# ----------------------------------------------------------------------

L3_CHANNELS = 21            # MambaSNN in_channels (must match spatial_mix).
L3_T = 313                  # preprocess_subband_single output time dim.
STUDENT_GROUPS = 8          # MambaSNN activity_logits group dim (NUM_GROUPS).

LABRAM_PATCH_SIZE = 200     # LaBraM patch_size (1 s @ 200 Hz).
LABRAM_EMBED_DIM = 200      # labram_base_patch200_200 embed_dim.
LABRAM_DEFAULT_PATCHES = 2  # 313 -> 2*200 = 400 resample target (<= time_embed 16).
LABRAM_MAX_PATCHES = 16     # NeuralTransformer.time_embed is [1, 16, 200].

TEACHER_NAMES = ("labram",)
DEFAULT_TEACHER = "labram"

# Default integration weight for the distillation loss when ADDED to the
# supervised 4-state loss. Small + soft on purpose (see CAVEAT in the header):
# the adapter is domain-shifted, so distillation regularises rather than
# supervises. Recommended sweep: {0.02, 0.05, 0.1}; start at 0.05.
RECOMMENDED_LAMBDA_DISTILL = 0.05

# Our fixed 21-ch 10-20 montage, in the exact channel order the dataset emits
# (matches `preprocess.CHANNEL_PRESETS` / `four_state` 21-ch convention).
LAMQUANT_CH21 = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ", "A1", "A2",
)

# LaBraM's `standard_1020` electrode list (verbatim from
# reference_software/labram/repo/utils.py). Inlined so we do NOT import
# utils.py (which hard-requires h5py). `pos_embed` row = index+1 (row 0 = cls).
# NOTE: every name in LAMQUANT_CH21 MUST appear here (asserted at load time).
_LABRAM_STANDARD_1020 = (
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9h", "TTP7h", "TPP9h", "FTT10h", "TPP8h", "TPP10h",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)


def _get_input_chans(ch_names: tuple[str, ...]) -> list[int]:
    """Map 10-20 channel names to LaBraM `pos_embed` row indices.

    Verbatim port of LaBraM `utils.get_input_chans`: row 0 is the cls token,
    each channel maps to `standard_1020.index(name) + 1`. Raises (no silent
    fallback) if a name is not in LaBraM's electrode list.
    """
    assert isinstance(ch_names, tuple) and len(ch_names) > 0, \
        f"ch_names must be a non-empty tuple, got {ch_names!r}"
    out = [0]  # cls token row
    for name in ch_names:
        if name not in _LABRAM_STANDARD_1020:
            raise ValueError(
                f"channel {name!r} not in LaBraM standard_1020 — cannot map "
                f"to a pos_embed row. Add it to the montage adapter or pick a "
                f"channel LaBraM was pretrained with."
            )
        out.append(_LABRAM_STANDARD_1020.index(name) + 1)
    return out


def _default_labram_repo() -> Path:
    """Resolve the on-disk LaBraM repo (model code + checkpoints).

    Override with `LAMQUANT_LABRAM_REPO`. Default points at the vendored
    reference_software copy relative to this file's repo root.
    """
    env = os.environ.get("LAMQUANT_LABRAM_REPO")
    if env:
        return Path(env)
    # blut/python/lamquant/snn/distill_teacher.py -> repo root is parents[5]?
    # The reference tree lives at <LamQuant>/reference_software/labram/repo.
    # We locate <LamQuant> by walking up until we find reference_software.
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "reference_software" / "labram" / "repo"
        if cand.exists():
            return cand
    # Last resort: the canonical absolute path (documented, not silent).
    return Path("/mnt/4tb/LamQuant/reference_software/labram/repo")


# ----------------------------------------------------------------------
# Input adapter — L3 [B, 21, 313] -> LaBraM [B, 21, n_patches, 200].
# ----------------------------------------------------------------------

def l3_to_labram_input(l3: torch.Tensor,
                       n_patches: int = LABRAM_DEFAULT_PATCHES) -> torch.Tensor:
    """Adapt L3 `[B, 21, 313]` to LaBraM input `[B, 21, n_patches, 200]`.

    Steps (see module header for the rationale + CAVEAT):
      1. Linear-resample the time axis 313 -> n_patches*200.
      2. Per-channel z-score (zero mean, unit std) over the time axis.
      3. Reshape the time axis into `(n_patches, 200)`.

    Args:
        l3: `[B, 21, 313]` float L3 subband signal (or `[21, 313]`, auto-batched).
        n_patches: number of 1 s LaBraM patches (1..16). Default 2 (=> 400 samples).

    Returns:
        `[B, 21, n_patches, 200]` float32, autograd-safe.
    """
    assert isinstance(l3, torch.Tensor), f"l3 must be torch.Tensor, got {type(l3).__name__}"
    if l3.dim() == 2:
        l3 = l3.unsqueeze(0)
    assert l3.dim() == 3, f"l3 must be [B,21,T] or [21,T], got shape {tuple(l3.shape)}"
    B, C, T = l3.shape
    assert C == L3_CHANNELS, f"l3 channel dim must be {L3_CHANNELS}, got {C}"
    assert T > 1, f"l3 time dim must be > 1, got {T}"
    assert isinstance(n_patches, int) and 1 <= n_patches <= LABRAM_MAX_PATCHES, \
        f"n_patches must be int in [1,{LABRAM_MAX_PATCHES}], got {n_patches!r}"

    target_len = n_patches * LABRAM_PATCH_SIZE
    x = l3.to(torch.float32)
    # Resample time axis 313 -> target_len via linear interpolation.
    # F.interpolate needs [B, C, T]; we already have that layout.
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)

    # Per-channel z-score over the (resampled) time axis. Honest amplitude
    # bridge from Q31-derived units to LaBraM's µV scale (NOT a calibration).
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)
    x = (x - mean) / (std + 1e-5)

    # [B, 21, target_len] -> [B, 21, n_patches, 200].
    x = x.reshape(B, C, n_patches, LABRAM_PATCH_SIZE)
    assert x.shape == (B, C, n_patches, LABRAM_PATCH_SIZE), \
        f"adapter output shape {tuple(x.shape)} != expected"
    return x


# ----------------------------------------------------------------------
# Teacher wrapper — loads LaBraM-base, emits a per-window embedding.
# ----------------------------------------------------------------------

class TeacherDistiller(nn.Module):
    """Frozen foundation-EEG teacher + L3 adapter, emitting per-window features.

    Public API (BUILD #3 contract):
        TeacherDistiller(teacher_name).teacher_features(l3_batch) -> [B, D]
        TeacherDistiller(...).distill_loss(student_feat, teacher_feat) -> scalar

    The teacher is loaded ONCE, frozen (`requires_grad_(False)`, `.eval()`), and
    runs under `torch.no_grad()` inside `teacher_features` — it never receives
    gradients. Only the student-side projection (`student_proj`, see
    `project_student`) is trainable.

    Args:
        teacher_name: currently only "labram".
        repo_dir: override LaBraM repo dir (default: vendored reference_software).
        ckpt_path: override checkpoint (default: <repo>/checkpoints/labram-base.pth).
        n_patches: L3 -> LaBraM patch count (default 2).
        embedding: "patch_mean" (mean-pool patch tokens, default) or "cls".
        loss_kind: "cosine" (default) or "mse" for `distill_loss`.
        device: torch device for the teacher (default: caller moves via `.to`).
        strict_load: require zero missing backbone keys (default True; the
            backbone DOES load cleanly with use_mean_pooling=False).
    """

    def __init__(self,
                 teacher_name: str = DEFAULT_TEACHER,
                 repo_dir: Optional[Path] = None,
                 ckpt_path: Optional[Path] = None,
                 n_patches: int = LABRAM_DEFAULT_PATCHES,
                 embedding: Literal["patch_mean", "cls"] = "patch_mean",
                 loss_kind: Literal["cosine", "mse"] = "cosine",
                 device: Optional[torch.device] = None,
                 strict_load: bool = True):
        super().__init__()
        if teacher_name not in TEACHER_NAMES:
            raise ValueError(
                f"unknown teacher {teacher_name!r}; supported: {TEACHER_NAMES}"
            )
        assert embedding in ("patch_mean", "cls"), \
            f"embedding must be 'patch_mean'|'cls', got {embedding!r}"
        assert loss_kind in ("cosine", "mse"), \
            f"loss_kind must be 'cosine'|'mse', got {loss_kind!r}"
        assert isinstance(n_patches, int) and 1 <= n_patches <= LABRAM_MAX_PATCHES, \
            f"n_patches must be int in [1,{LABRAM_MAX_PATCHES}], got {n_patches!r}"

        self.teacher_name = teacher_name
        self.n_patches = n_patches
        self.embedding = embedding
        self.loss_kind = loss_kind
        self.embed_dim = LABRAM_EMBED_DIM

        repo = Path(repo_dir) if repo_dir is not None else _default_labram_repo()
        ckpt = (Path(ckpt_path) if ckpt_path is not None
                else repo / "checkpoints" / "labram-base.pth")
        self.repo_dir = repo
        self.ckpt_path = ckpt

        self.teacher = self._build_labram(repo, ckpt, strict_load)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # input_chans as a buffer so `.to(device)` moves it with the module.
        ic = torch.tensor(_get_input_chans(LAMQUANT_CH21), dtype=torch.long)
        self.register_buffer("input_chans", ic, persistent=False)

        # Student-side projection: pooled [B, 8] -> [B, embed_dim]. The ONLY
        # trainable parameter this module owns. Built lazily on first
        # `project_student` call to allow callers to pick the student feature
        # layout, but with a default that matches MambaSNN activity_logits.
        self.student_proj = nn.Linear(STUDENT_GROUPS, self.embed_dim)

        if device is not None:
            self.to(device)

    # -- LaBraM construction ------------------------------------------------

    @staticmethod
    def _build_labram(repo: Path, ckpt: Path, strict_load: bool) -> nn.Module:
        """Import LaBraM's model code, build labram_base, load the checkpoint.

        Built with `use_mean_pooling=False` so the pretrained `norm` LayerNorm
        loads (the finetune `fc_norm` would otherwise be missing). With this
        config the backbone loads with ZERO missing keys.
        """
        if not repo.exists():
            raise FileNotFoundError(
                f"LaBraM repo not found at {repo} — set LAMQUANT_LABRAM_REPO."
            )
        if not ckpt.exists():
            raise FileNotFoundError(
                f"LaBraM checkpoint not found at {ckpt}. Expected the vendored "
                f"labram-base.pth (97 MB). Fetch it from the LaBraM release "
                f"(https://github.com/935963004/LaBraM, checkpoints/)."
            )

        import sys
        repo_str = str(repo)
        added = repo_str not in sys.path
        if added:
            sys.path.insert(0, repo_str)
        try:
            import importlib
            # modeling_finetune registers labram_base_patch200_200 with timm.
            importlib.import_module("modeling_finetune")
            from timm.models import create_model
        finally:
            # Leave repo on sys.path? Keep it — the timm registry holds a
            # reference to the module's factory; removing the path is fine but
            # re-imports elsewhere would need it. We pop only what we added.
            if added and repo_str in sys.path:
                sys.path.remove(repo_str)

        model = create_model(
            "labram_base_patch200_200",
            pretrained=False,
            num_classes=0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            use_mean_pooling=False,   # load pretrained `norm`, not `fc_norm`.
            init_scale=0.001,
            use_rel_pos_bias=False,
            use_abs_pos_emb=True,
            init_values=0.1,
            qkv_bias=False,
        )

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # Backbone keys are prefixed `student.` (pretrain EMA student).
        prefix = "student."
        backbone = {k[len(prefix):]: v for k, v in state.items()
                    if k.startswith(prefix)}
        if not backbone:
            # Some releases save un-prefixed; accept that too (documented).
            backbone = {k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(backbone, strict=False)
        # `fc_norm` is intentionally absent (use_mean_pooling=False). Any OTHER
        # missing key is a real load failure.
        real_missing = [m for m in missing if not m.startswith("fc_norm")]
        if real_missing and strict_load:
            raise RuntimeError(
                f"LaBraM load is missing backbone keys (not just fc_norm): "
                f"{real_missing[:8]} ... ({len(real_missing)} total). "
                f"Checkpoint/model mismatch — refusing to distill from a "
                f"partially-initialised teacher."
            )
        return model

    # -- teacher embedding --------------------------------------------------

    @torch.no_grad()
    def teacher_features(self, l3_batch: torch.Tensor) -> torch.Tensor:
        """Per-window teacher embedding for an L3 batch.

        Args:
            l3_batch: `[B, 21, 313]` float L3 (or `[21, 313]`, auto-batched).

        Returns:
            `[B, embed_dim]` (=`[B, 200]`) float32 teacher embedding, detached.
        """
        assert isinstance(l3_batch, torch.Tensor), \
            f"l3_batch must be torch.Tensor, got {type(l3_batch).__name__}"
        single = l3_batch.dim() == 2
        x = l3_to_labram_input(l3_batch, n_patches=self.n_patches)
        x = x.to(next(self.teacher.parameters()).device, dtype=torch.float32)
        ic = self.input_chans.to(x.device)

        if self.embedding == "patch_mean":
            pt = self.teacher.forward_features(
                x, input_chans=ic, return_patch_tokens=True)  # [B, 21*P, D]
            feat = pt.mean(dim=1)                              # [B, D]
        else:  # cls
            feat = self.teacher.forward_features(
                x, input_chans=ic, return_patch_tokens=False)  # [B, D]

        feat = feat.to(torch.float32)
        B = x.shape[0]
        assert feat.shape == (B, self.embed_dim), \
            f"teacher feature shape {tuple(feat.shape)} != ({B}, {self.embed_dim})"
        out = feat.detach()
        return out[0] if single else out

    # -- student projection -------------------------------------------------

    def project_student(self, student_feat: torch.Tensor) -> torch.Tensor:
        """Project a student backbone feature into the teacher embedding space.

        Accepts EITHER:
          * `[B, 8, T]` activity_logits — pooled over T to `[B, 8]` first, OR
          * `[B, 8]`     already-pooled per-window summary.

        then `student_proj: Linear(8 -> embed_dim)`.

        Args:
            student_feat: `[B, 8, T]` or `[B, 8]` (or unbatched `[8, T]`/`[8]`).

        Returns:
            `[B, embed_dim]` float32 (grad flows back into the student + proj).
        """
        assert isinstance(student_feat, torch.Tensor), \
            f"student_feat must be torch.Tensor, got {type(student_feat).__name__}"
        x = student_feat
        if x.dim() == 1:           # [8] -> [1, 8]
            x = x.unsqueeze(0)
        elif x.dim() == 2 and x.shape[0] == STUDENT_GROUPS and x.shape[1] != STUDENT_GROUPS:
            # ambiguous [8, T] unbatched: treat as single-sample [8, T].
            x = x.unsqueeze(0)     # [1, 8, T]
        if x.dim() == 3:           # [B, 8, T] -> pool over T -> [B, 8]
            assert x.shape[1] == STUDENT_GROUPS, \
                f"3-D student_feat must be [B,{STUDENT_GROUPS},T], got {tuple(x.shape)}"
            x = x.mean(dim=2)
        assert x.dim() == 2 and x.shape[1] == STUDENT_GROUPS, \
            f"pooled student_feat must be [B,{STUDENT_GROUPS}], got {tuple(x.shape)}"
        proj = self.student_proj(x.to(torch.float32))
        assert proj.shape[1] == self.embed_dim
        return proj

    # -- distillation loss --------------------------------------------------

    def distill_loss(self, student_feat: torch.Tensor,
                     teacher_feat: torch.Tensor) -> torch.Tensor:
        """Distillation loss between student + teacher per-window features.

        `student_feat` may be the RAW student backbone feature (`[B,8,T]` or
        `[B,8]`) — it is projected to the teacher space internally via
        `project_student`. If it is ALREADY `[B, embed_dim]`, it is used as-is
        (allows callers to project once and reuse).

        `teacher_feat` is `[B, embed_dim]` from `teacher_features` (detached).

        Returns a scalar:
          * cosine  -> mean(1 - cos_sim)  in [0, 2]
          * mse     -> mean L2 over normalised features

        Both features are L2-normalised before the loss so the objective is
        scale-free (the teacher embedding norm is not a learning target).
        """
        assert isinstance(student_feat, torch.Tensor) and isinstance(teacher_feat, torch.Tensor), \
            "student_feat and teacher_feat must both be torch.Tensor"
        # Project student unless it already lives in teacher space.
        if not (student_feat.dim() == 2 and student_feat.shape[1] == self.embed_dim):
            s = self.project_student(student_feat)
        else:
            s = student_feat.to(torch.float32)
        t = teacher_feat.to(torch.float32).detach()

        assert s.shape == t.shape, \
            f"projected student {tuple(s.shape)} != teacher {tuple(t.shape)}"
        assert s.dim() == 2 and s.shape[1] == self.embed_dim, \
            f"feature dim must be [B,{self.embed_dim}], got {tuple(s.shape)}"

        s_n = F.normalize(s, dim=1, eps=1e-6)
        t_n = F.normalize(t, dim=1, eps=1e-6)

        if self.loss_kind == "cosine":
            cos = (s_n * t_n).sum(dim=1)         # [B] in [-1, 1]
            loss = (1.0 - cos).mean()
        else:  # mse on normalised features
            loss = F.mse_loss(s_n, t_n)

        assert loss.dim() == 0, "distill_loss must be a scalar"
        assert torch.isfinite(loss).item(), \
            "distill_loss is non-finite — features pathological"
        return loss


__all__ = [
    "TeacherDistiller",
    "l3_to_labram_input",
    "TEACHER_NAMES",
    "DEFAULT_TEACHER",
    "RECOMMENDED_LAMBDA_DISTILL",
    "LABRAM_EMBED_DIM",
    "LAMQUANT_CH21",
]
