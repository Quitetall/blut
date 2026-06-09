#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# train_4state_controller.py — train the LamQuant SNN's ACTUAL job:
# a 4-state "needs-precision" compression-tier controller.
#
# Each EEG latent timestep is mapped to one of four SNAC compression tiers
#
#     QUIET=0       → FSQ level 2   → CR ~525:1
#     BASELINE=1    → FSQ level 3   → CR ~134:1
#     INTERESTING=2 → FSQ level 4   → CR ~82:1
#     CRITICAL=3    → FSQ level 5   → CR ~63:1
#
# This REPLACES the drifted seizure-only objective. Seizure becomes one
# trigger of CRITICAL (the seizure-safety tier), not the training target.
#
# Reuses (imports, never reimplements):
#   * MambaSNN backbone        (lamquant_neural.models.mamba_ssm_minimal)
#   * build_head K=4           (lamquant_neural.models.heads)
#   * LmaDataset / iter_labels (lamquant.snn.lma_dataset)
#   * WSDScheduler             (lamquant.student.train_joint)
#   * ESOAP                    (lamquant.student.esoap)
#   * derive_4state_target / calibrate_quiet_threshold (lamquant.snn.four_state)
#
# The backbone is loaded with the same SNN_SEIZURE_HEAD / SNN_SEIZURE_BIAS env
# the production seizure runs use (so checkpoints stay load-compatible), but
# the seizure head is NOT optimized as the objective — only activity_logits
# feed the 4-state head.
#
# Programming-Bible style: contract assertions, no silent fallback, typed.

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

LOG = logging.getLogger("lamquant.snn.train_4state_controller")
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

# ---------------------------------------------------------------------------
# Path plumbing — mirror train_mamba_snn.py so the cross-area imports resolve.
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _sub in ("snn", "student", "dataset", "common"):
    _p = os.path.join(ROOT_DIR, "lamquant", _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lamquant_neural.models.mamba_ssm_minimal import MambaSNN, clamp_ssm_params  # noqa: E402
from lamquant_neural.models.heads import build_head  # noqa: E402
from lamquant.snn.lma_dataset import LmaDataset  # noqa: E402
from lamquant.snn.four_state import (  # noqa: E402
    derive_4state_target,
    calibrate_quiet_threshold,
    NUM_STATES,
    STATE_NAMES,
    LEVEL_TABLE_4,
    CR_TABLE_4,
    TARGET_DIST_4,
)

# ---------------------------------------------------------------------------
# ADR-0027 upgrade modules. Each is imported eagerly so an --init/--distill/
# --spectral/--ordinal typo fails fast at startup, not deep in the loop. They
# are pure-function / nn.Module helpers — importing them is side-effect-free
# (no checkpoint load, no GPU) so the baseline path pays nothing.
# ---------------------------------------------------------------------------
from lamquant.snn.spectral import (  # noqa: E402
    build_augmented_input,
    AUGMENTED_IN_CHANNELS,
)
from lamquant.snn.ordinal_loss import constrained_loss  # noqa: E402
from lamquant.snn.distill_teacher import (  # noqa: E402
    TeacherDistiller,
    RECOMMENDED_LAMBDA_DISTILL,
)

# Geometry — L3 latent time dim (preprocess_subband_single output).
L3_T = 313
NUM_GROUPS = 8


# ===========================================================================
# State-balanced sampler — oversample windows with any rare state (>= 2).
# ===========================================================================

class StateBalancedSampler(Sampler[int]):
    """Oversample windows that contain any INTERESTING or CRITICAL timestep.

    Adapted from ``SeizureBalancedSampler`` but keyed on "has rare state"
    (a per-timestep 4-state target >= 2) instead of "has seizure". The
    rare-window fraction is CONSTANT (``rare_frac``) — NO down-anneal. The
    seizure-run anneal was proven to atrophy the head (it drifts into a
    background-predictor once the rare signal thins out), so this sampler
    holds the rare fraction fixed for the whole run.

    Rare windows are drawn WITH replacement when they would otherwise run
    out, so a high target fraction never truncates the epoch. Length is
    fixed to the dataset size for a stable per-epoch step count.

    The per-window "has rare state" flag is precomputed once at construction
    by deriving the 4-state target for every window's labels + L3 (cheap:
    the L3 cache is warm and only the RMS is needed).
    """

    def __init__(self, dataset: LmaDataset, quiet_rms_threshold: float,
                 rare_frac: float = 0.4, seed: int = 1337):
        if not isinstance(dataset, LmaDataset):
            raise TypeError(
                f"StateBalancedSampler requires LmaDataset, got "
                f"{type(dataset).__name__}")
        if not 0.0 <= rare_frac <= 1.0:
            raise ValueError(f"rare_frac must be in [0,1], got {rare_frac}")
        self.dataset = dataset
        self.rare_frac = float(rare_frac)
        self.seed = int(seed)
        self.epoch = 0
        self._total = len(dataset)
        assert self._total > 0, "dataset is empty"

        rare_flags = compute_rare_flags(dataset, quiet_rms_threshold)
        assert len(rare_flags) == self._total, \
            "rare_flags length must equal dataset length"
        flags = np.asarray(rare_flags, dtype=bool)
        self.rare_idx = np.nonzero(flags)[0]
        self.common_idx = np.nonzero(~flags)[0]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n = self._total
        n_rare = int(round(self.rare_frac * n))
        n_common = n - n_rare

        # Degenerate splits: if one class is empty, draw the other.
        if self.rare_idx.size == 0:
            n_rare, n_common = 0, n
        if self.common_idx.size == 0:
            n_rare, n_common = n, 0

        picks = []
        if n_rare > 0:
            replace = n_rare > self.rare_idx.size
            picks.append(rng.choice(self.rare_idx, size=n_rare, replace=replace))
        if n_common > 0:
            replace = n_common > self.common_idx.size
            picks.append(rng.choice(self.common_idx, size=n_common, replace=replace))
        order = np.concatenate(picks) if picks else np.arange(n)
        rng.shuffle(order)
        yield from (int(i) for i in order)

    def __len__(self) -> int:
        return self._total


def compute_rare_flags(dataset: LmaDataset,
                       quiet_rms_threshold: float) -> np.ndarray:
    """Per-window boolean: does the window contain any rare state (>= 2)?

    "Rare" = INTERESTING (2) or CRITICAL (3). Uses the LABELS only (a rare
    state never comes from the RMS split — RMS only separates QUIET vs
    BASELINE), so this avoids touching the L3 cache: a window has a rare
    state iff its 3-class labels contain a 1 (active) or 2 (seizure) in any
    group at any timestep. ``quiet_rms_threshold`` is accepted for interface
    symmetry but is not needed here.

    ``_iter_index_labels`` reads each label NPZ once (grouped by LMA) but
    yields the DATASET INDEX with each window, so the returned flag array is
    positionally aligned with ``__getitem__`` / the sampler.
    """
    flags = np.zeros(len(dataset), dtype=bool)
    for i, lbl in _iter_index_labels(dataset):
        # Any active (1) or seizure (2) in any group/timestep ⇒ rare.
        flags[i] = bool((lbl >= 1).any())
    return flags


def _iter_index_labels(dataset: LmaDataset):
    """Yield (index, labels[8, L3_T]) in dataset.index order, label-only.

    Groups index entries by (lma_path, label_internal) so each label NPZ is
    read once, then yields the per-window slice for every index pointing into
    it. Mirrors iter_labels_only but preserves the dataset INDEX so the
    sampler's flags align positionally with __getitem__.
    """
    import io
    from collections import defaultdict
    import lamquant_core as _lc
    from lamquant.snn.lma_dataset import (
        _label_cache_dir, LABEL_PER_WINDOW, _lazy_imports,
    )
    _lazy_imports()
    label_cache = _label_cache_dir()
    by_lma: dict = defaultdict(list)
    for idx, entry in enumerate(dataset.index):
        lma_path, stem, win_idx = entry[0], entry[1], entry[2]
        label_internal = entry[4] if len(entry) >= 5 else entry[3]
        by_lma[(str(lma_path), label_internal)].append((idx, win_idx, stem))

    n_label_load_fail = 0
    for (lma_path, label_internal), items in by_lma.items():
        stem_for_cache = items[0][2]
        cached = (label_cache / f"{stem_for_cache}_labels.npz") if label_cache else None
        try:
            if cached is not None and cached.exists():
                with np.load(cached, allow_pickle=True) as ld:
                    activity = np.asarray(ld["activity_labels"])
            else:
                lb = _lc.lma_read_entry(lma_path, label_internal)
                with np.load(io.BytesIO(lb), allow_pickle=True) as ld:
                    activity = np.asarray(ld["activity_labels"])
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            # Don't silently bias the StateBalancedSampler: a zero-label
            # fallback marks the window as having NO rare state, so a corrupt
            # NPZ would quietly down-weight real INTERESTING/CRITICAL windows.
            # Count + warn so data-pipeline breakage is visible (MiMo a04bc8e).
            n_label_load_fail += 1
            LOG.warning("4state rare-flag scan: label load failed for %s (%s): %s",
                        stem_for_cache, label_internal, e)
            activity = np.zeros((NUM_GROUPS, L3_T), dtype=np.int64)
        for idx, win_idx, _stem in items:
            lbl_start = win_idx * LABEL_PER_WINDOW
            lbl_end = min(lbl_start + L3_T, activity.shape[1])
            w = np.zeros((NUM_GROUPS, L3_T), dtype=np.int64)
            if lbl_end > lbl_start:
                n = lbl_end - lbl_start
                w[:, :n] = activity[:, lbl_start:lbl_end].astype(np.int64)
                if n < L3_T:
                    w[:, n:] = w[:, n - 1:n]
            yield idx, w
    if n_label_load_fail:
        LOG.warning("4state rare-flag scan: %d label group(s) failed to load "
                    "and were treated as all-QUIET — sampler balance may be "
                    "biased; check the data.", n_label_load_fail)


# ===========================================================================
# Target derivation on a batch (labels + l3 → 4-state target pooled to Tout).
# ===========================================================================

def _pool_states_to_T(states: np.ndarray, target_T: int) -> np.ndarray:
    """Nearest-neighbour pool a [T] integer state array to [target_T].

    Class labels can't be averaged, so we pick, for each output bin, the
    state at the bin centre's nearest input index. No-op when T == target_T.
    """
    T = states.shape[0]
    if T == target_T:
        return states
    # Map each output position to the nearest input index.
    src = np.round(np.linspace(0, T - 1, target_T)).astype(np.int64)
    return states[src]


def derive_batch_targets(labels: torch.Tensor, l3: torch.Tensor,
                         quiet_rms_threshold: float,
                         target_T: int) -> torch.Tensor:
    """Per-batch 4-state targets, pooled to ``target_T``.

    Args:
        labels: ``[B, 8, T]`` int64 3-class labels.
        l3: ``[B, 21, T_l3]`` float L3.
        quiet_rms_threshold: QUIET/BASELINE RMS cut.
        target_T: head output time resolution.

    Returns:
        ``[B, target_T]`` int64 in {0..3}, on the same device as ``labels``.
    """
    assert labels.dim() == 3 and labels.shape[1] == NUM_GROUPS, \
        f"labels must be [B,8,T], got {tuple(labels.shape)}"
    assert l3.dim() == 3 and l3.shape[1] == 21, \
        f"l3 must be [B,21,T_l3], got {tuple(l3.shape)}"
    B = labels.shape[0]
    lbl_np = labels.detach().cpu().numpy()
    l3_np = l3.detach().cpu().numpy()
    out = np.empty((B, target_T), dtype=np.int64)
    for b in range(B):
        tgt = derive_4state_target(lbl_np[b], l3_np[b], quiet_rms_threshold)
        out[b] = _pool_states_to_T(tgt, target_T)
    t = torch.from_numpy(out).to(labels.device)
    assert t.shape == (B, target_T) and t.dtype == torch.long
    return t


# ===========================================================================
# Loss — class-weighted CE (inverse-freq + hard CRITICAL floor) or CRF NLL.
# ===========================================================================

def compute_class_weights(dataset: LmaDataset, quiet_rms_threshold: float,
                          n_windows: int, critical_floor: float = 2.0,
                          seed: int = 1337) -> torch.Tensor:
    """Inverse-frequency class weights with a hard CRITICAL floor.

    Samples windows, derives 4-state targets, counts each state, and returns
    inverse-frequency weights normalised to mean 1.0 (so the absolute loss
    scale is stable). CRITICAL is then multiplied by ``critical_floor`` AFTER
    normalisation so it always carries the highest weight — missing the
    seizure-safety tier is the worst error.

    Returns: ``[4]`` float32 tensor.
    """
    n_total = len(dataset)
    assert n_total > 0, "dataset empty"
    n_sample = min(n_windows, n_total)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n_total, size=n_sample, replace=False)
    counts = np.zeros(NUM_STATES, dtype=np.float64)
    for i in idxs:
        l3_t, lab_t = dataset[int(i)]
        tgt = derive_4state_target(lab_t.numpy(), l3_t.numpy(), quiet_rms_threshold)
        counts += np.bincount(tgt, minlength=NUM_STATES)[:NUM_STATES]
    # Smooth so an empty state doesn't blow up (no silent /0).
    counts = counts + 1.0
    inv = counts.sum() / counts             # inverse frequency
    inv = inv / inv.mean()                  # normalise to mean 1.0
    inv[NUM_STATES - 1] *= float(critical_floor)  # hard CRITICAL floor
    w = torch.tensor(inv, dtype=torch.float32)
    assert w.shape == (NUM_STATES,) and torch.isfinite(w).all()
    return w


# ===========================================================================
# Per-epoch 4-state metrics.
# ===========================================================================

def _confusion(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """4x4 confusion matrix, rows = true state, cols = predicted state."""
    cm = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int64)
    np.add.at(cm, (tgt, pred), 1)
    return cm


def four_state_metrics(cm: np.ndarray) -> dict:
    """Derive accuracy / per-state P-R / CRITICAL recall / specificity / CR.

    cm: 4x4 confusion (true rows, pred cols).
    Returns a dict of the per-epoch logged numbers.
    """
    assert cm.shape == (NUM_STATES, NUM_STATES)
    total = cm.sum()
    acc = float(np.trace(cm) / max(total, 1))

    precision = np.zeros(NUM_STATES)
    recall = np.zeros(NUM_STATES)
    for k in range(NUM_STATES):
        tp = cm[k, k]
        precision[k] = tp / max(cm[:, k].sum(), 1)
        recall[k] = tp / max(cm[k, :].sum(), 1)

    # CRITICAL recall = the seizure-safety floor (most important number).
    critical_recall = float(recall[NUM_STATES - 1])

    # QUIET+BASELINE specificity: of all timesteps whose TRUE state is QUIET
    # or BASELINE, the fraction NOT escalated to a high-fidelity tier
    # (INTERESTING/CRITICAL). Over-escalating low tiers wastes bandwidth.
    low_rows = cm[:2, :]                       # true QUIET/BASELINE
    low_total = low_rows.sum()
    low_escalated = low_rows[:, 2:].sum()       # predicted INTERESTING/CRITICAL
    quiet_specificity = float((low_total - low_escalated) / max(low_total, 1))

    # Implied average CR. Predicted-state CR vs true-label CR.
    cr = np.array(CR_TABLE_4)
    pred_counts = cm.sum(axis=0).astype(np.float64)   # by predicted state
    true_counts = cm.sum(axis=1).astype(np.float64)   # by true state
    pred_cr = float((pred_counts * cr).sum() / max(pred_counts.sum(), 1))
    true_cr = float((true_counts * cr).sum() / max(true_counts.sum(), 1))

    return {
        "acc": acc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "critical_recall": critical_recall,
        "quiet_specificity": quiet_specificity,
        "pred_cr": pred_cr,
        "true_cr": true_cr,
    }


def combined_score(m: dict) -> float:
    """Selection score — maximise CRITICAL recall while keeping low-tier
    specificity high, penalising over-escalation. NOT seizure sens/spec.

        score = critical_recall + 0.5*quiet_specificity − escalation_penalty

    escalation_penalty grows when the controller compresses LESS than the
    label distribution implies (pred_cr << true_cr ⇒ over-escalation).
    """
    cr_ratio = m["pred_cr"] / max(m["true_cr"], 1e-6)
    # Penalise compressing too little (pred_cr below true_cr). Cap at 0 when
    # the controller compresses at least as hard as the labels.
    escalation_penalty = max(0.0, 1.0 - cr_ratio)
    return m["critical_recall"] + 0.5 * m["quiet_specificity"] - escalation_penalty


def selection_key(m: dict, alpha: float) -> tuple:
    """Lexicographic feasibility-first checkpoint selection (ADR 0029).

    Replaces ``combined_score`` for selection. The weighted sum made safety
    FUNGIBLE with compression and *rewarded* the operating-point slide
    (CRIT_rec 0.90->0.49 traded for QB_spec while loss fell). This key is a
    tuple compared lexicographically, so safety can never be purchased with
    compression:

      * feasible  (CRIT_rec >= alpha): key = (1, quiet_specificity)
            among SAFE checkpoints, prefer the one that compresses the boring
            (true QUIET/BASELINE) content hardest.
      * infeasible (CRIT_rec <  alpha): key = (0, critical_recall)
            none safe yet (or the data frontier cannot reach alpha): prefer
            the LEAST-slid checkpoint. This degenerates to "max CRIT_rec"
            exactly when the safe region is unreachable -- the correct
            fail-safe (keep the high-recall / low-CR checkpoint, never ship a
            slid one).

    A feasible checkpoint ALWAYS outranks an infeasible one: (1, x) > (0, y)
    for any x, y in [0, 1]. ``alpha`` is the TRAINING-FEASIBILITY floor
    (~= 1.0 - label-noise), NOT a clinical guarantee -- the clinical floor is
    the end-to-end sensitivity-degradation bound (ADR 0029 addendum).
    """
    if m["critical_recall"] >= alpha:
        return (1, m["quiet_specificity"])
    return (0, m["critical_recall"])


# ===========================================================================
# Train / validate.
# ===========================================================================

def train_epoch(model, head, loader, optimizer, device, quiet_thr,
                target_T, class_weights, head_kind, lambda_spike, grad_clip,
                use_spectral=False, use_ordinal=False, crit_floor=0.88,
                distiller=None, lambda_distill=0.0):
    model.train()
    head.train()
    total_loss = 0.0
    n_steps = 0
    nan_skips = 0
    cm = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int64)
    cw = class_weights.to(device)

    for l3, labels in loader:
        l3, labels = l3.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # --spectral: widen the backbone input with band-power features. The
        # 4-state TARGET still derives from the RAW L3 RMS (derive_batch_targets
        # below is passed `l3`, NOT `x`) — only the backbone sees the augmented
        # channels.
        if use_spectral:
            x = build_augmented_input(l3)                   # [B,105,T]
        else:
            x = l3                                          # [B,21,T]
        activity_logits, spike_rate, _seizure = model(x)    # [B,8,T], scalar, _
        states, class_logits = head(activity_logits, target_T)  # [B,Tout],[B,4,Tout]
        target = derive_batch_targets(labels, l3, quiet_thr, target_T)  # [B,Tout]

        if head_kind == "crf":
            # CRF path is unaffected by --ordinal (the ordinal/constrained
            # objective replaces the FLAT softmax CE, not the CRF NLL).
            loss_main = head.neg_log_likelihood(class_logits, target)
        elif use_ordinal:
            # ADR-0027 #4: ordinal + constrained objective (drop-in for the
            # weighted CE). escalation/ramp args stay at their module defaults.
            loss_main = constrained_loss(class_logits, target, weight=cw,
                                         crit_floor=crit_floor)
        else:
            loss_main = F.cross_entropy(class_logits, target, weight=cw)
        loss = loss_main + lambda_spike * spike_rate

        # ADR-0027 #3: foundation-teacher feature distillation. The teacher is
        # frozen + runs under no_grad inside teacher_features; only student_proj
        # (an optimizer param group added in main) + the backbone receive grad.
        if distiller is not None:
            teacher_feat = distiller.teacher_features(l3)        # [B,200] detached
            loss = loss + lambda_distill * distiller.distill_loss(
                activity_logits, teacher_feat)

        if not torch.isfinite(loss):
            nan_skips += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()), grad_clip)
        optimizer.step()
        with torch.no_grad():
            clamp_ssm_params(model)
        n_steps += 1
        total_loss += float(loss.item())

        with torch.no_grad():
            pred = class_logits.argmax(dim=1)  # [B,Tout]
            cm += _confusion(pred.cpu().numpy().ravel(),
                             target.cpu().numpy().ravel())

    avg_loss = total_loss / max(n_steps, 1)
    return avg_loss, cm, nan_skips, n_steps


@torch.no_grad()
def validate(model, head, loader, device, quiet_thr, target_T,
             use_spectral=False):
    model.eval()
    head.eval()
    cm = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int64)
    for l3, labels in loader:
        l3, labels = l3.to(device), labels.to(device)
        # Mirror train_epoch: augmented backbone input under --spectral, raw L3
        # for the target (derive_batch_targets gets the original `l3`).
        x = build_augmented_input(l3) if use_spectral else l3
        activity_logits, _, _ = model(x)
        _states, class_logits = head(activity_logits, target_T)
        target = derive_batch_targets(labels, l3, quiet_thr, target_T)
        pred = class_logits.argmax(dim=1)
        cm += _confusion(pred.cpu().numpy().ravel(),
                         target.cpu().numpy().ravel())
    return cm


# ===========================================================================
# Async checkpoint save (reuse the simple pattern from train_mamba_snn).
# ===========================================================================

import threading as _threading  # noqa: E402
import atexit as _atexit  # noqa: E402
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor  # noqa: E402

_SAVE_EXECUTOR: Optional[_ThreadPoolExecutor] = None
_SAVE_LOCK = _threading.Lock()


def _ensure_save_executor():
    global _SAVE_EXECUTOR
    with _SAVE_LOCK:
        if _SAVE_EXECUTOR is None:
            _SAVE_EXECUTOR = _ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="snn4-ckpt")
            _atexit.register(_SAVE_EXECUTOR.shutdown, wait=True)
    return _SAVE_EXECUTOR


def _state_dict_to_cpu(sd):
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().to("cpu", copy=True) if hasattr(v, "detach") else v
    return out


def _atomic_torch_save(payload: dict, path: str) -> None:
    """torch.save to a temp file in the same dir, then atomic rename — a
    mid-write kill leaves the prior checkpoint intact rather than a truncated
    one (MiMo a04bc8e)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _async_save(payload: dict, path: str):
    _ensure_save_executor().submit(_atomic_torch_save, payload, path)


# ===========================================================================
# LMA root expansion (mirror train_mamba_snn).
# ===========================================================================

def expand_lma_roots(roots) -> list[Path]:
    lma_paths: list[Path] = []
    for r in roots:
        r = Path(r)
        if r.is_file() and r.suffix == ".lma":
            lma_paths.append(r)
            continue
        if not r.is_dir():
            raise FileNotFoundError(f"--lma-root not found: {r}")
        found = sorted(r.glob("*/*.lma")) or sorted(r.glob("*.lma"))
        if not found:
            raise RuntimeError(f"no .lma archives under {r}")
        lma_paths.extend(found)
    seen: set = set()
    return [p for p in lma_paths if not (str(p) in seen or seen.add(str(p)))]


# ===========================================================================
# Main.
# ===========================================================================

def main():
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    p = argparse.ArgumentParser(
        description="Train the 4-state needs-precision CR controller.")
    p.add_argument("--lma-root", type=Path, nargs="+", required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--head", default="attention_softmax",
                   choices=["attention_softmax", "crf"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=40)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--max-windows-per-file", type=int, default=5)
    p.add_argument("--target-T", type=int, default=L3_T,
                   help="head output time resolution (default 313 = latent T)")
    p.add_argument("--seq-windows", type=int, default=1,
                   help="ADR-0027 temporal-context lever: train the SSM on K "
                        "CONSECUTIVE 10 s windows (state carries across the "
                        "boundaries -> the model sees the seizure's evolution, "
                        "not one isolated 10 s slice). K=1 = current per-window "
                        "behaviour. Deploys as streaming state-carry (O(d_state) "
                        "on the MCU, no SRAM blowup).")
    p.add_argument("--rare-frac", type=float, default=0.4,
                   help="CONSTANT fraction of rare-state windows per epoch "
                        "(no anneal)")
    p.add_argument("--critical-weight-floor", type=float, default=2.0,
                   help="extra multiplier on the CRITICAL class weight after "
                        "inverse-freq normalisation (safety tier)")
    p.add_argument("--lambda-spike", type=float, default=0.01,
                   help="small spike-rate regularizer from the backbone")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--warmup-frac", type=float, default=0.10)
    p.add_argument("--infinite-lr", action="store_true",
                   help="WSD∞: warmup then constant peak LR (continual)")
    p.add_argument("--optimizer", default="esoap",
                   choices=["adamw", "esoap"])
    p.add_argument("--early-stop-patience", type=int, default=60)
    p.add_argument("--calib-windows", type=int, default=2000,
                   help="windows sampled for threshold + class-weight calibration")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--logger", choices=["none", "wandb"], default="none",
                   help="metric sink: the MetricLog CSV is ALWAYS on; 'wandb' "
                        "adds W&B (mode via WANDB_MODE env, default offline; set "
                        "WANDB_MODE=online to stream live).")

    # ---- ADR-0027 upgrade toggles. ALL default OFF / inert so the existing
    #      run-20 baseline command is byte-for-byte unchanged in behaviour. ----
    p.add_argument("--spectral", action="store_true",
                   help="ADR-0027 #2: augment the backbone input with per-band "
                        "power features (in_channels 21 -> 105). The 4-state "
                        "TARGET still uses raw-L3 RMS.")
    p.add_argument("--ordinal", action="store_true",
                   help="ADR-0027 #4: ordinal + constrained objective "
                        "(constrained_loss) in place of weighted CE. No effect "
                        "on the CRF head.")
    p.add_argument("--crit-floor", type=float, default=0.88,
                   help="soft CRITICAL-recall floor for --ordinal "
                        "constrained_loss (default 0.88)")
    p.add_argument("--crit-alpha", type=float, default=0.95,
                   help="ADR-0029: feasibility-first CRITICAL-recall floor for "
                        "checkpoint SELECTION (training-feasibility pre-gate, NOT "
                        "a clinical guarantee). Feasible epochs (CRIT_rec>=alpha) "
                        "always outrank infeasible ones; among feasible, max "
                        "QB_spec. If the data frontier cannot reach alpha, "
                        "selection degenerates to max-CRIT_rec (the fail-safe). "
                        "Replaces the deprecated combined_score selection.")
    p.add_argument("--crit-dual-eta", type=float, default=0.5,
                   help="ADR-0029: dual-ascent step for the Lagrangian "
                        "multiplier mu on the CRITICAL class weight. mu grows "
                        "when val CRIT_rec < crit-alpha (pushes recall up), "
                        "relaxes when the floor holds (lets QB_spec/CR improve). "
                        "0 disables the dual (static class weights).")
    p.add_argument("--crit-dual-mu-max", type=float, default=8.0,
                   help="ADR-0029: cap on the dual multiplier mu, so the "
                        "CRITICAL class weight cannot explode.")
    p.add_argument("--init-backbone", type=Path, default=None,
                   help="ADR-0027 #1: SSL-pretrained backbone-init checkpoint "
                        "(.pt from pretrain_ssl_tueg.py). Loaded strict=False "
                        "for the backbone keys; head + seizure head keep their "
                        "fresh init.")
    p.add_argument("--distill", default=None,
                   choices=["labram"],
                   help="ADR-0027 #3: foundation teacher for feature "
                        "distillation (currently only 'labram').")
    p.add_argument("--lambda-distill", type=float,
                   default=RECOMMENDED_LAMBDA_DISTILL,
                   help=f"weight on the distillation loss when --distill is set "
                        f"(default {RECOMMENDED_LAMBDA_DISTILL})")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    import random as _random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[4state] device={device} head={args.head} optimizer={args.optimizer}")
    print(f"[4state] states: " + " ".join(
        f"{STATE_NAMES[k]}->L{LEVEL_TABLE_4[k]}(CR{CR_TABLE_4[k]:.0f})"
        for k in range(NUM_STATES)))

    # ---- Upgrade-flag banner (ADR-0027). Empty ⇒ this IS the run-20 baseline. ----
    upgrades = []
    if args.spectral:
        upgrades.append("spectral")
    if args.ordinal:
        upgrades.append("ordinal")
    if args.init_backbone is not None:
        upgrades.append("init-backbone")
    if args.distill is not None:
        upgrades.append(f"distill:{args.distill}")
    print(f"[4state] ADR-0027 upgrades: "
          f"{', '.join(upgrades) if upgrades else 'NONE (run-20 baseline)'}")

    # ---- Model: backbone (subband path) + 4-state head. ----
    # --spectral widens the backbone input to AUGMENTED_IN_CHANNELS (=105); the
    # only constructor change is in_channels — spatial_mix auto-widens to
    # Linear(105 -> d_model). Everything downstream is shape-identical.
    in_channels = AUGMENTED_IN_CHANNELS if args.spectral else 21
    model = MambaSNN(in_channels=in_channels, d_model=args.d_model,
                     d_state=args.d_state, n_layers=args.n_layers,
                     use_subband=True).to(device)
    head = build_head(args.head, K=NUM_STATES).to(device)

    # ---- ADR-0027 #1: SSL-pretrained backbone init (strict=False). ----
    # pretrain_ssl_tueg.py saves a backbone-only state_dict under the "backbone"
    # key (seizure_head.* dropped). strict=False so: (a) the controller's own
    # 4-state head + seizure head keep their fresh init (they're `missing`), and
    # (b) under --spectral the widened spatial_mix.{weight,bias} shape-mismatches
    # and is simply skipped (left freshly-init) — warned, not a hard error.
    if args.init_backbone is not None:
        ck = torch.load(str(args.init_backbone), map_location=device,
                        weights_only=False)
        if isinstance(ck, dict) and "backbone" in ck:
            sd = ck["backbone"]
        else:
            sd = ck
        if args.spectral:
            # Drop the 21-ch spatial_mix so load_state_dict doesn't raise on the
            # 105-vs-21 shape mismatch; the widened layer stays freshly-init.
            dropped = [k for k in list(sd.keys()) if k.startswith("spatial_mix.")]
            for k in dropped:
                sd.pop(k)
            if dropped:
                print(f"[4state] WARNING --init-backbone + --spectral: "
                      f"dropped {dropped} from the SSL init (21-ch spatial_mix "
                      f"!= 105-ch); widened spatial_mix stays freshly-init.")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        matched = len(set(sd.keys()) & set(model.state_dict().keys()))
        print(f"[4state] --init-backbone {args.init_backbone}: "
              f"matched={matched} missing={len(missing)} "
              f"unexpected={len(unexpected)}")
        if unexpected:
            print(f"[4state]   unexpected (NOT in MambaSNN): {sorted(unexpected)}")

    n_params = sum(p.numel() for p in model.parameters()) + \
        sum(p.numel() for p in head.parameters())
    print(f"[4state] params: backbone+head = {n_params:,}")

    # ---- ADR-0027 #3: foundation teacher (built ONCE, frozen). ----
    distiller = None
    if args.distill is not None:
        distiller = TeacherDistiller(args.distill).to(device)
        n_teacher = sum(p.numel() for p in distiller.teacher.parameters())
        n_proj = sum(p.numel() for p in distiller.student_proj.parameters())
        print(f"[4state] --distill {args.distill}: teacher={n_teacher:,} "
              f"(frozen) student_proj={n_proj:,} (trainable) "
              f"lambda_distill={args.lambda_distill}")

    # ---- Data. ----
    lma_paths = expand_lma_roots(args.lma_root)
    print(f"[4state] {len(lma_paths)} .lma archive(s)")
    train_ds = LmaDataset(lma_paths=lma_paths, split="train",
                          split_manifest_path=args.split_manifest,
                          max_windows_per_file=args.max_windows_per_file,
                          seq_windows=args.seq_windows)
    val_ds = LmaDataset(lma_paths=lma_paths, split="val",
                        split_manifest_path=args.split_manifest,
                        max_windows_per_file=args.max_windows_per_file,
                        seq_windows=args.seq_windows)
    print(f"[4state] train={len(train_ds)} val={len(val_ds)} "
          f"(seq_windows={args.seq_windows})")

    save_dir = os.path.join(ROOT_DIR, "weights", "snn")
    os.makedirs(save_dir, exist_ok=True)
    save_path = args.checkpoint or os.path.join(save_dir, "snn_4state_best.pt")

    # ---- Calibrate quiet threshold ONCE on train; cache next to ckpt. ----
    thr_cache = Path(os.path.splitext(save_path)[0] + "_quiet_thr.json")
    _cached = json.loads(thr_cache.read_text()) if thr_cache.exists() else None
    if _cached is not None and _cached.get("split_manifest") == str(args.split_manifest):
        quiet_thr = float(_cached["quiet_rms_threshold"])
        print(f"[4state] loaded cached quiet_rms_threshold={quiet_thr:.6g} "
              f"from {thr_cache}")
    else:
        if _cached is not None:
            print(f"[4state] stale threshold cache (manifest "
                  f"{_cached.get('split_manifest')!r} != {str(args.split_manifest)!r}) "
                  f"— recalibrating")
        print(f"[4state] calibrating quiet_rms_threshold on {args.calib_windows} "
              f"train windows ...")
        quiet_thr = calibrate_quiet_threshold(train_ds, n_windows=args.calib_windows,
                                              seed=args.seed)
        thr_cache.write_text(json.dumps(
            {"quiet_rms_threshold": quiet_thr,
             "calib_windows": args.calib_windows,
             "split_manifest": str(args.split_manifest)}, indent=2))
        print(f"[4state] quiet_rms_threshold={quiet_thr:.6g} (cached -> {thr_cache})")

    # ---- Class weights (inverse-freq + CRITICAL floor). ----
    class_weights = compute_class_weights(
        train_ds, quiet_thr, n_windows=args.calib_windows,
        critical_floor=args.critical_weight_floor, seed=args.seed)
    print("[4state] class weights: " + ", ".join(
        f"{STATE_NAMES[k]}={class_weights[k]:.3f}" for k in range(NUM_STATES)))

    # ADR-0029 dual ascent: the inverse-freq weights are the BASE; the
    # CRITICAL entry is scaled each epoch by (1 + mu), a slow Lagrangian
    # multiplier that rises while val CRIT_rec sits below --crit-alpha.
    base_class_weights = class_weights.clone()
    mu = 0.0

    # ---- Sampler: state-balanced, CONSTANT rare fraction (no anneal). ----
    print(f"[4state] building StateBalancedSampler (rare_frac={args.rare_frac}, "
          f"no anneal) ...")
    train_sampler = StateBalancedSampler(train_ds, quiet_thr,
                                         rare_frac=args.rare_frac, seed=args.seed)
    print(f"[4state]   {train_sampler.rare_idx.size} rare / "
          f"{train_sampler.common_idx.size} common windows")

    _default_workers = 4 if os.environ.get("L3_CACHE_DIR") else 2
    num_workers = args.num_workers if args.num_workers is not None else \
        int(os.environ.get("LMA_NUM_WORKERS", str(_default_workers)))
    _dl_kwargs = {}
    if num_workers > 0:
        _dl_kwargs["persistent_workers"] = True
        _dl_kwargs["prefetch_factor"] = int(os.environ.get("LMA_PREFETCH_FACTOR", "4"))
    pin = device.type == "cuda" and num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=num_workers,
                              pin_memory=pin, **_dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, **_dl_kwargs)

    # ---- Optimizer. ----
    params = list(model.named_parameters()) + \
        [(f"head.{n}", q) for n, q in head.named_parameters()]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            [q for _n, q in params if q.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:  # esoap
        from esoap import ESOAP
        _linear_suffixes = ("in_proj.weight", "x_proj.weight",
                            "out_proj.weight", "spatial_mix.weight")
        esoap_linear, adamw_rest = [], []
        for nm, q in params:
            if not q.requires_grad:
                continue
            if q.ndim == 2 and nm.endswith(_linear_suffixes):
                esoap_linear.append(q)
            else:
                adamw_rest.append(q)
        print(f"[4state] ESOAP: {len(esoap_linear)} linear matrices -> "
              f"SOAP-lead+Muon-tail; {len(adamw_rest)} -> AdamW")
        optimizer = ESOAP(
            [{"params": esoap_linear, "method": "esoap",
              "weight_decay": args.weight_decay},
             {"params": adamw_rest, "method": "adamw",
              "weight_decay": args.weight_decay}],
            lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    # ---- ADR-0027 #3: register the distiller's student_proj as a trainable
    #      param group (the teacher stays frozen + out of the optimizer). For
    #      ESOAP this lands as a plain AdamW group (a 2-D Linear without an
    #      ESOAP-routed suffix → AdamW semantics, matching the rest). ----
    if distiller is not None:
        optimizer.add_param_group(
            {"params": list(distiller.student_proj.parameters())})

    # ---- Schedule: WSD∞ (warmup→constant peak) or WSD with decay tail. ----
    from train_joint import WSDScheduler
    if args.infinite_lr:
        scheduler = WSDScheduler(optimizer, total_epochs=args.epochs,
                                 peak_lr=args.lr, warmup_frac=args.warmup_frac,
                                 decay_frac=0.0, min_lr=args.lr_min,
                                 warmup_kind="cosine")
        print(f"[4state] schedule: cosine-warmup -> WSD∞ stable "
              f"(warmup={scheduler.warmup_epochs}ep)")
    else:
        scheduler = WSDScheduler(optimizer, total_epochs=args.epochs,
                                 peak_lr=args.lr, warmup_frac=args.warmup_frac,
                                 decay_frac=0.10, min_lr=args.lr_min,
                                 warmup_kind="cosine")
        print(f"[4state] schedule: cosine-warmup -> WSD -> cosine decay "
              f"(warmup={scheduler.warmup_epochs}ep)")

    # Cross-window: the head emits K*L3_T states so the SSM scans the full
    # K-window span (state carries across the 10 s boundaries). K=1 unchanged.
    target_T = int(args.target_T) * args.seq_windows
    snap_every = int(os.environ.get("SNN_SNAPSHOT_EVERY", "0"))

    best_key = (-1, -1e9)        # ADR-0029 lexicographic feasibility-first
    best_metrics = None
    best_epoch = 0
    epochs_since_best = 0
    train_start = time.time()
    print(f"[4state] training {args.epochs} epochs x {len(train_loader)} batches "
          f"(bs={args.batch_size}, target_T={target_T})")

    # Metric sinks (mirror train_joint): the MetricLog CSV/Parquet is ALWAYS on
    # — a complete, reviewer-readable file after every epoch, read live with
    # `python -m blut_core.read_metric --run <run_id>` (verbatim, no LLM).
    # wandb is optional (--logger wandb; WANDB_MODE=online to stream).
    run_id = f"snn4state_{args.head}_{int(train_start)}"
    # ADR 0044 P10: the BLUT_JOB_DIR-or-training_logs anchor is resolved ONCE in
    # the core primitive (blut_core.runctx), not re-derived here.
    from blut_core import runctx
    from blut_core.metric_log import MetricLog
    metric_log_dir = runctx.job_dir(Path(ROOT_DIR) / "training_logs")
    metric_log = MetricLog(run_id=run_id, log_dir=metric_log_dir)
    print(f"[4state] metric stream: {metric_log.path} "
          f"(backend={metric_log._backend}) run_id={run_id}")
    wandb_run = None
    if args.logger == "wandb":
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "lamquant"),
                name=run_id,
                config=vars(args) | {"lma_root": [str(x) for x in args.lma_root],
                                     "split_manifest": str(args.split_manifest)},
                tags=["snn", "4state", args.head, args.optimizer],
                mode=os.environ.get("WANDB_MODE", "offline"),
                dir=str(metric_log_dir))
            print(f"[4state] wandb: mode={os.environ.get('WANDB_MODE', 'offline')} "
                  f"project={os.environ.get('WANDB_PROJECT', 'lamquant')}")
        except Exception as e:
            print(f"[4state] --logger wandb requested but unavailable ({e}); "
                  f"continuing without it")
            wandb_run = None

    for epoch in range(args.epochs):
        ep_start = time.time()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # ADR-0029: scale the CRITICAL class weight by the dual multiplier mu.
        epoch_weights = base_class_weights.clone()
        epoch_weights[NUM_STATES - 1] *= (1.0 + mu)

        avg_loss, tr_cm, nan_skips, n_steps = train_epoch(
            model, head, train_loader, optimizer, device, quiet_thr,
            target_T, epoch_weights, args.head, args.lambda_spike, args.grad_clip,
            use_spectral=args.spectral, use_ordinal=args.ordinal,
            crit_floor=args.crit_floor, distiller=distiller,
            lambda_distill=args.lambda_distill)
        scheduler.step()

        val_cm = validate(model, head, val_loader, device, quiet_thr, target_T,
                          use_spectral=args.spectral)
        m = four_state_metrics(val_cm)
        # ADR-0029: lexicographic feasibility-first selection. combined_score
        # is still logged for continuity but NO LONGER selects (it rewarded the
        # safety->compression slide).
        key = selection_key(m, args.crit_alpha)
        score = combined_score(m)
        feasible = m["critical_recall"] >= args.crit_alpha

        improved = ""
        if key > best_key:
            best_key = key
            best_metrics = dict(m)   # snapshot; m is rebound each epoch
            best_epoch = epoch + 1
            epochs_since_best = 0
            improved = " *BEST*"
            _async_save({
                "model": _state_dict_to_cpu(model.state_dict()),
                "head": _state_dict_to_cpu(head.state_dict()),
                "head_kind": args.head,
                "num_states": NUM_STATES,
                "in_channels": in_channels,
                "upgrades": upgrades,
                "quiet_rms_threshold": quiet_thr,
                "level_table": list(LEVEL_TABLE_4),
                "cr_table": list(CR_TABLE_4),
                # Preserve the original contract: class_weights = the BASE
                # inverse-freq weights; effective_class_weights = the
                # mu-scaled weights actually used this epoch (with mu).
                "class_weights": base_class_weights.tolist(),
                "effective_class_weights": epoch_weights.tolist(),
                "optimizer": _state_dict_to_cpu(optimizer.state_dict()),
                "epoch": epoch + 1,
                "score": score,
                "selection_key": list(key),
                "feasible": bool(feasible),
                "crit_alpha": args.crit_alpha,
                "mu": mu,
                "metrics": m,
                "config": vars(args) | {"lma_root": [str(x) for x in args.lma_root],
                                        "split_manifest": str(args.split_manifest),
                                        "checkpoint": str(save_path)},
            }, save_path)
        else:
            epochs_since_best += 1

        # ADR-0029 dual ascent (slow timescale, on the HARD-argmax val CRIT_rec
        # from the confusion matrix). mu rises while the floor is violated,
        # relaxes when it holds — the rigorous version of the static CRITICAL
        # weight floor. The step is symmetric-additive but the GAP is not:
        # violations (alpha - crit_rec large) push mu up hard, satisfaction
        # (gap ~ 0 near the floor) relaxes it slowly. If the data frontier
        # cannot reach alpha, mu pins at --crit-dual-mu-max and the model
        # over-weights CRITICAL -> high recall / low CR. That is the INTENDED
        # fail-safe (ADR 0029 addendum: infeasible -> fail-safe to max tier,
        # CR drops), bounded so the weight cannot explode.
        if args.crit_dual_eta > 0.0:
            mu = float(min(args.crit_dual_mu_max,
                           max(0.0, mu + args.crit_dual_eta
                               * (args.crit_alpha - m["critical_recall"]))))

        if snap_every > 0 and ((epoch + 1) % snap_every == 0
                               or epoch + 1 == args.epochs):
            snap = f"{os.path.splitext(save_path)[0]}_ep{epoch+1}.pt"
            _async_save({
                "model": _state_dict_to_cpu(model.state_dict()),
                "head": _state_dict_to_cpu(head.state_dict()),
                "head_kind": args.head, "num_states": NUM_STATES,
                "in_channels": in_channels, "upgrades": upgrades,
                "quiet_rms_threshold": quiet_thr, "epoch": epoch + 1,
                "metrics": m,
            }, snap)

        # Per-epoch log — CRITICAL recall + avg CR are the headline numbers.
        prec = m["precision"]; rec = m["recall"]
        ep_sec = time.time() - ep_start
        gpu_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        print(
            f"E{epoch+1:3d}/{args.epochs} L={avg_loss:.4f} "
            f"acc={m['acc']:.3f} "
            f"CRIT_rec={m['critical_recall']:.3f} "
            f"QB_spec={m['quiet_specificity']:.3f} "
            f"CR[pred={m['pred_cr']:.0f} true={m['true_cr']:.0f}] "
            f"score={score:.3f} feas={'Y' if feasible else 'n'} mu={mu:.2f} "
            f"skips={nan_skips}/{len(train_loader)} {ep_sec:.0f}s "
            f"GPU={gpu_mb:.0f}M{improved}")
        # Per-state P/R + compact confusion every epoch.
        print("        P/R: " + " ".join(
            f"{STATE_NAMES[k][:4]}[{prec[k]:.2f}/{rec[k]:.2f}]"
            for k in range(NUM_STATES)))
        print("        confusion(true rows -> pred cols): " +
              " | ".join(",".join(str(int(x)) for x in row) for row in val_cm))

        # Verbatim metric sink (always) + optional wandb. Guarded: logging
        # must never crash a run. Only flat scalars (per-state P/R vectors are
        # excluded — already printed above).
        try:
            d = {"run_id": run_id, "script": "train_4state_controller",
                 "epoch": epoch + 1, "global_epoch": epoch + 1,
                 "total_epochs": args.epochs, "timestamp": time.time(),
                 "train_loss": float(avg_loss),
                 "lr": float(optimizer.param_groups[0]["lr"]),
                 "score": float(score), "feasible": bool(feasible),
                 "mu": float(mu), "best_epoch": best_epoch,
                 "nan_skips": int(nan_skips), "secs_per_epoch": float(ep_sec),
                 "gpu_mb": float(gpu_mb),
                 **{k: float(v) for k, v in m.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}}
            metric_log.append(d)
            if wandb_run is not None:
                scalars = {k: v for k, v in d.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
                wandb_run.log(scalars, step=epoch + 1)
        except Exception as e:
            print(f"[4state] metric/wandb emit failed (non-fatal): {e}")

        if (args.early_stop_patience and args.early_stop_patience > 0
                and epochs_since_best >= args.early_stop_patience):
            print(f"[4state] early stop: no improvement for "
                  f"{args.early_stop_patience} epochs (best @ ep{best_epoch}).")
            break

    metric_log.close()
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    total_h = (time.time() - train_start) / 3600
    if best_metrics is not None:
        _bf = "feasible" if best_metrics["critical_recall"] >= args.crit_alpha else "INFEASIBLE(fail-safe)"
        print(f"\n[4state] done in {total_h:.2f}h. best @ ep{best_epoch} "
              f"[{_bf}] CRIT_rec={best_metrics['critical_recall']:.3f} "
              f"QB_spec={best_metrics['quiet_specificity']:.3f} "
              f"(alpha={args.crit_alpha}). saved: {save_path}")
    else:
        print(f"\n[4state] done in {total_h:.2f}h. no checkpoint saved. "
              f"saved: {save_path}")


if __name__ == "__main__":
    main()
