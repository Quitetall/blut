# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 LamQuant authors.
#
# ordinal_loss.py — ordinal + constrained objective for the 4-state SNN
# compression-tier controller (ADR 0027, BUILD #4).
#
# The four CR-controller states are ORDERED, not categorical:
#
#     QUIET=0  <  BASELINE=1  <  INTERESTING=2  <  CRITICAL=3
#
# i.e. monotonically increasing required fidelity (decreasing compression).
# Flat weighted cross-entropy treats every off-diagonal error as equally bad,
# which makes the controller trade CRITICAL recall away to buy QUIET/BASELINE
# specificity (run-20: CRIT_rec slid 0.85 -> 0.70). Two fixes live here:
#
#   (1) ordinal_loss — a cumulative-link / CORAL loss over the K-1 = 3
#       ordinal thresholds. An off-by-one tier error (e.g. true CRITICAL,
#       predicted INTERESTING) costs the model far less than an off-by-three
#       (true CRITICAL, predicted QUIET), because the cumulative formulation
#       penalises EACH crossed threshold once. This is the right inductive
#       bias for an ordered-tier controller.
#
#   (2) constrained_loss — a differentiable Lagrangian wrapper that
#       MAXIMISES compression (penalises escalating true-QUIET / true-BASELINE
#       timesteps up to INTERESTING / CRITICAL) SUBJECT TO a soft CRITICAL-
#       recall floor. When the in-batch CRITICAL recall drops below the floor
#       the CRITICAL ordinal term is ramped up (a smooth hinge multiplier), so
#       the optimiser is pushed back toward the seizure-safety constraint
#       before it is allowed to keep buying compression.
#
# Both are drop-in replacements for the trainer's
#     F.cross_entropy(class_logits, target, weight=cw)
# call: same tensor contract (class_logits [B, K, T] pre-softmax, target
# [B, T] int64 in {0..K-1}), same scalar-loss return.
#
# Programming-Bible style: contract assertions, no silent fallback, typed.

from __future__ import annotations

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------
# State ordering — single source of truth for this module. MUST match
# four_state.STATE_NAMES / heads.LEVEL_TABLE_4 ordering (QUIET<...<CRITICAL).
# ----------------------------------------------------------------------

NUM_STATES = 4
CRITICAL_STATE = NUM_STATES - 1     # index 3
# "low" tiers whose escalation to a "high" tier wastes bandwidth.
LOW_STATES = (0, 1)                 # QUIET, BASELINE
HIGH_STATES = (2, 3)               # INTERESTING, CRITICAL

DEFAULT_CRIT_FLOOR = 0.88
DEFAULT_ESCALATION_WEIGHT = 0.25
DEFAULT_CRIT_RAMP = 4.0            # max extra CRITICAL weight when recall -> 0


def _check_logits_target(class_logits: torch.Tensor,
                         target: torch.Tensor) -> tuple[int, int, int]:
    """Validate the shared (logits, target) contract; return (B, K, T)."""
    assert isinstance(class_logits, torch.Tensor), \
        f"class_logits must be a Tensor, got {type(class_logits).__name__}"
    assert isinstance(target, torch.Tensor), \
        f"target must be a Tensor, got {type(target).__name__}"
    assert class_logits.dim() == 3, \
        f"class_logits must be [B, K, T], got shape {tuple(class_logits.shape)}"
    B, K, T = class_logits.shape
    assert K == NUM_STATES, \
        f"class_logits K dim must be {NUM_STATES}, got {K}"
    assert target.dim() == 2, \
        f"target must be [B, T], got shape {tuple(target.shape)}"
    assert target.shape == (B, T), \
        f"target shape {tuple(target.shape)} != class_logits (B,T)=({B},{T})"
    assert target.dtype in (torch.int64, torch.long), \
        f"target dtype must be int64/long, got {target.dtype}"
    # Range check (no silent clamp).
    tmin = int(target.min())
    tmax = int(target.max())
    assert 0 <= tmin and tmax < NUM_STATES, \
        f"target out of range — expected {{0..{NUM_STATES - 1}}}, " \
        f"got min={tmin} max={tmax}"
    return B, K, T


# ----------------------------------------------------------------------
# Deliverable 1 — ordinal (cumulative-link / CORAL) loss.
# ----------------------------------------------------------------------

def _cumulative_logits(class_logits: torch.Tensor) -> torch.Tensor:
    """Reduce K-class logits [B, K, T] to K-1 cumulative threshold logits.

    For an ordered problem the natural quantity is

        P(state > k)   for k in {0 .. K-2}.

    We derive a logit for ``P(state > k)`` from the softmax tail mass:

        p_gt_k = sum_{j > k} softmax(class_logits)_j
        logit_gt_k = logit(p_gt_k)

    This keeps the existing classification head untouched (it still emits K
    logits and ``argmax`` still selects the predicted tier) while giving the
    ordinal loss a monotone, per-threshold view. The K-1 thresholds are then
    each supervised with binary cross-entropy against the cumulative target
    ``1[target > k]`` — so crossing the wrong number of thresholds is what
    costs, and an off-by-d error pays d (not 1) binary terms.

    Returns:
        ``[B, K-1, T]`` cumulative threshold logits.
    """
    probs = F.softmax(class_logits, dim=1)            # [B, K, T]
    # P(state > k) = reverse-cumsum excluding class k = tail mass above k.
    # cumsum over classes gives P(state <= k); 1 - that = P(state > k).
    cdf = torch.cumsum(probs, dim=1)                  # [B, K, T], P(state<=k)
    p_gt = 1.0 - cdf[:, :-1, :]                        # [B, K-1, T], P(state>k)
    # Numerically safe logit.
    eps = 1e-6
    p_gt = p_gt.clamp(eps, 1.0 - eps)
    return torch.log(p_gt) - torch.log1p(-p_gt)       # logit


def _cumulative_target(target: torch.Tensor, K: int) -> torch.Tensor:
    """Binary cumulative target ``1[target > k]`` for k in {0..K-2}.

    Shape ``[B, K-1, T]`` float. For target=t, exactly the first t thresholds
    are 1 (state exceeds them) and the rest 0 — the monotone CORAL encoding.
    """
    B, T = target.shape
    ks = torch.arange(K - 1, device=target.device).view(1, K - 1, 1)  # [1,K-1,1]
    return (target.unsqueeze(1) > ks).float()          # [B, K-1, T]


def ordinal_loss(class_logits: torch.Tensor,
                 target: torch.Tensor,
                 weight: torch.Tensor | None = None,
                 reduction: str = "mean") -> torch.Tensor:
    """Cumulative-link (CORAL-style) ordinal loss for ordered tiers.

    Drop-in replacement for ``F.cross_entropy(class_logits, target, weight=w)``
    when the K classes are an ORDERED scale. Penalises EACH crossed ordinal
    threshold, so an off-by-3 tier error costs ~3x an off-by-1 error.

    Args:
        class_logits: ``[B, K, T]`` pre-softmax logits (K = NUM_STATES = 4).
        target: ``[B, T]`` int64 in ``{0..K-1}``.
        weight: optional ``[K]`` per-CLASS weight (same semantics as
            ``F.cross_entropy``'s ``weight``). Each timestep's binary-threshold
            terms are scaled by ``weight[target]`` so a rare/important tier
            (e.g. CRITICAL) keeps its emphasis. ``None`` ⇒ unweighted.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``. ``"none"`` returns the
            per-timestep ``[B, T]`` loss (threshold terms summed over k).

    Returns:
        Scalar loss (``mean``/``sum``) or ``[B, T]`` (``none``).
    """
    B, K, T = _check_logits_target(class_logits, target)
    assert reduction in ("mean", "sum", "none"), \
        f"reduction must be mean|sum|none, got {reduction!r}"

    cum_logits = _cumulative_logits(class_logits)      # [B, K-1, T]
    cum_target = _cumulative_target(target, K)         # [B, K-1, T]

    # Per-threshold binary cross-entropy, summed over the K-1 thresholds →
    # per-timestep ordinal cost [B, T]. Crossing d wrong thresholds pays d.
    bce = F.binary_cross_entropy_with_logits(
        cum_logits, cum_target, reduction="none")       # [B, K-1, T]
    per_ts = bce.sum(dim=1)                              # [B, T]

    if weight is not None:
        assert isinstance(weight, torch.Tensor) and weight.shape == (K,), \
            f"weight must be [{K}] Tensor, got {getattr(weight, 'shape', None)}"
        assert torch.isfinite(weight).all(), "weight has non-finite entries"
        w_ts = weight.to(per_ts.dtype).to(per_ts.device)[target]  # [B, T]
        per_ts = per_ts * w_ts

    if reduction == "none":
        return per_ts
    if reduction == "sum":
        return per_ts.sum()
    # mean: if weighted, normalise by total weight (matches F.cross_entropy
    # weighted-mean semantics: sum(w*loss)/sum(w)); else plain mean.
    if weight is not None:
        denom = w_ts.sum().clamp_min(1e-8)
        return per_ts.sum() / denom
    return per_ts.mean()


# ----------------------------------------------------------------------
# Deliverable 2 — constrained (Lagrangian) wrapper.
# ----------------------------------------------------------------------

def _soft_critical_recall(class_logits: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    """Differentiable in-batch CRITICAL recall ∈ [0, 1].

    Recall = E[ P(pred=CRITICAL) | true=CRITICAL ]. We use the soft (softmax)
    probability of the CRITICAL class at the true-CRITICAL timesteps as a
    differentiable surrogate for the hard recall, averaged over those
    timesteps. When the batch has no CRITICAL timesteps recall is undefined;
    we return a tensor of 1.0 (constraint trivially satisfied, no ramp) so the
    floor never spuriously fires on a CRITICAL-free batch.
    """
    crit_mask = (target == CRITICAL_STATE)             # [B, T] bool
    n_crit = crit_mask.sum()
    probs = F.softmax(class_logits, dim=1)             # [B, K, T]
    p_crit = probs[:, CRITICAL_STATE, :]               # [B, T]
    if int(n_crit) == 0:
        # No CRITICAL in batch — constraint trivially satisfied. Keep it on the
        # graph (so callers can rely on a tensor) but detached from any sample.
        return p_crit.new_ones(())
    return (p_crit * crit_mask.float()).sum() / n_crit.clamp_min(1).to(p_crit.dtype)


def _escalation_penalty(class_logits: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
    """Soft penalty on escalating true-LOW timesteps to a HIGH tier.

    For timesteps whose TRUE state is QUIET or BASELINE, charge the softmax
    mass the model places on the HIGH tiers (INTERESTING, CRITICAL) — i.e. the
    probability of OVER-escalating a compressible timestep. This is the
    "maximise compression" objective: it directly pushes true-low timesteps
    away from high-fidelity (low-CR) predictions.

    Mean over the true-LOW timesteps. Zero (on-graph) when the batch has no
    LOW timesteps.
    """
    low_mask = torch.zeros_like(target, dtype=torch.bool)
    for s in LOW_STATES:
        low_mask |= (target == s)
    n_low = low_mask.sum()
    probs = F.softmax(class_logits, dim=1)             # [B, K, T]
    p_high = probs[:, HIGH_STATES[0]:, :].sum(dim=1)   # [B, T], P(high tier)
    if int(n_low) == 0:
        return probs.new_zeros(())
    return (p_high * low_mask.float()).sum() / n_low.clamp_min(1).to(probs.dtype)


def constrained_loss(class_logits: torch.Tensor,
                     target: torch.Tensor,
                     weight: torch.Tensor | None = None,
                     crit_floor: float = DEFAULT_CRIT_FLOOR,
                     escalation_weight: float = DEFAULT_ESCALATION_WEIGHT,
                     crit_ramp: float = DEFAULT_CRIT_RAMP,
                     reduction: str = "mean") -> torch.Tensor:
    """Constrained ordinal objective: max compression s.t. CRITICAL recall floor.

    A differentiable Lagrangian/penalty wrapper around :func:`ordinal_loss`:

        loss = ordinal_loss
               + crit_ramp * relu(crit_floor - soft_crit_recall)
                            * ordinal_loss_on_CRITICAL_timesteps
               + escalation_weight * escalation_penalty

    Term-by-term:

      * ``ordinal_loss`` — the base ordered-tier fit (Deliverable 1).
      * The middle term is the soft CRITICAL-recall HINGE: a multiplier that is
        ZERO while the in-batch soft CRITICAL recall is at/above ``crit_floor``
        and ramps UP (proportional to the shortfall, scaled by ``crit_ramp``)
        as recall falls below it — applied to the ordinal loss restricted to
        the true-CRITICAL timesteps. This is the Lagrangian pressure that
        re-prioritises the seizure-safety tier the moment the constraint is
        violated, instead of letting the optimiser trade it away.
      * ``escalation_weight * escalation_penalty`` — the "maximise compression"
        objective: penalise over-escalating true-QUIET/BASELINE timesteps to
        the high-fidelity (low-CR) tiers.

    Drop-in for ``F.cross_entropy(class_logits, target, weight=cw)``.

    Args:
        class_logits: ``[B, K, T]`` pre-softmax logits.
        target: ``[B, T]`` int64 in ``{0..K-1}``.
        weight: optional ``[K]`` per-class weight forwarded to the base
            ordinal loss (e.g. the trainer's inverse-freq + CRITICAL-floor
            weights). The recall hinge and escalation penalty are independent
            of it.
        crit_floor: CRITICAL-recall constraint floor (default 0.88). When the
            in-batch soft recall is below this, the CRITICAL term ramps up.
        escalation_weight: weight on the over-escalation (compression) penalty.
        crit_ramp: max additional CRITICAL emphasis as recall → 0.
        reduction: ``"mean"`` | ``"sum"`` for the base ordinal term (the two
            penalty terms are always batch-scalar means).

    Returns:
        Scalar loss tensor.
    """
    B, K, T = _check_logits_target(class_logits, target)
    assert reduction in ("mean", "sum"), \
        f"constrained reduction must be mean|sum, got {reduction!r}"
    assert 0.0 < crit_floor <= 1.0, \
        f"crit_floor must be in (0, 1], got {crit_floor!r}"
    assert escalation_weight >= 0.0, \
        f"escalation_weight must be >= 0, got {escalation_weight!r}"
    assert crit_ramp >= 0.0, f"crit_ramp must be >= 0, got {crit_ramp!r}"

    # Base ordered-tier loss over the whole batch.
    base = ordinal_loss(class_logits, target, weight=weight, reduction=reduction)

    # --- Soft CRITICAL-recall hinge -----------------------------------------
    recall = _soft_critical_recall(class_logits, target)   # scalar in [0,1]
    # ReLU shortfall: > 0 only when recall < floor.
    shortfall = torch.clamp(class_logits.new_tensor(crit_floor) - recall, min=0.0)
    ramp = crit_ramp * shortfall                            # scalar >= 0

    # Ordinal loss restricted to the true-CRITICAL timesteps (the term the
    # hinge amplifies). per-ts ordinal, then mask-mean over CRITICAL.
    crit_mask = (target == CRITICAL_STATE)
    n_crit = crit_mask.sum()
    if int(n_crit) > 0:
        per_ts = ordinal_loss(class_logits, target, weight=None,
                              reduction="none")              # [B, T]
        crit_term = (per_ts * crit_mask.float()).sum() / n_crit.to(per_ts.dtype)
    else:
        # No CRITICAL timesteps → no hinge contribution (ramp is also ~0 since
        # recall defaults to 1.0). Keep on-graph zero.
        crit_term = class_logits.new_zeros(())
    hinge = ramp * crit_term

    # --- Compression / over-escalation penalty ------------------------------
    esc = _escalation_penalty(class_logits, target)        # scalar >= 0

    total = base + hinge + escalation_weight * esc
    assert torch.isfinite(total), "constrained_loss produced a non-finite value"
    return total


# ----------------------------------------------------------------------
# Diagnostics helper (used by the trainer's logging / tests; pure-read).
# ----------------------------------------------------------------------

def constrained_loss_components(class_logits: torch.Tensor,
                                target: torch.Tensor,
                                weight: torch.Tensor | None = None,
                                crit_floor: float = DEFAULT_CRIT_FLOOR,
                                escalation_weight: float = DEFAULT_ESCALATION_WEIGHT,
                                crit_ramp: float = DEFAULT_CRIT_RAMP) -> dict:
    """Return the scalar components of :func:`constrained_loss` for logging.

    Keys: ``base``, ``soft_crit_recall``, ``hinge``, ``escalation``, ``total``.
    All are python floats (detached). Pure diagnostic — does not affect grad.
    """
    with torch.no_grad():
        base = float(ordinal_loss(class_logits, target, weight=weight))
        recall = float(_soft_critical_recall(class_logits, target))
        esc = float(_escalation_penalty(class_logits, target))
        total = float(constrained_loss(
            class_logits, target, weight=weight, crit_floor=crit_floor,
            escalation_weight=escalation_weight, crit_ramp=crit_ramp))
        shortfall = max(0.0, crit_floor - recall)
        crit_mask = (target == CRITICAL_STATE)
        if int(crit_mask.sum()) > 0:
            per_ts = ordinal_loss(class_logits, target, weight=None,
                                  reduction="none")
            crit_term = float((per_ts * crit_mask.float()).sum()
                              / crit_mask.sum().to(per_ts.dtype))
        else:
            crit_term = 0.0
        hinge = crit_ramp * shortfall * crit_term
    return {
        "base": base,
        "soft_crit_recall": recall,
        "hinge": hinge,
        "escalation": esc,
        "total": total,
    }
