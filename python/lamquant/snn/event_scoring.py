#!/usr/bin/env python3
"""Event-level seizure scoring for the LamQuant SNN (NEDC OVLP semantics).

WHY THIS EXISTS
---------------
The trainer's ``calibrate_seizure_threshold`` reports a RAW per-timestep
false-positive rate. At the L3 timestep (~0.03195 s) a single isolated
false-positive sample inflates the per-hour count into the tens of
thousands — clinically meaningless. Real seizure detectors are scored at
the EVENT level: the per-timestep probability stream is post-processed
into discrete seizure EVENTS (onset/offset pairs), and the operating
point is reported as

    sensitivity = fraction of TRUE events that any prediction overlaps
    FPR/h       = false predicted events per hour of recording

This module is pure-numpy (no torch dependency) so it can run inside the
firmware-validation harness, on the host eval path, and in CI without a
GPU. Every function is deterministic and unit-tested in
``test_event_scoring.py``.

POST-PROCESSING PIPELINE (per recording)
----------------------------------------
1. threshold the probability stream -> boolean seizure mask
2. group consecutive True samples into raw events
3. merge events separated by a gap shorter than ``merge_gap_sec``
4. drop events shorter than ``min_event_sec``
5. apply a ``refractory_sec`` dead-time after each accepted event so a
   single clinical seizure that the detector breaks into a burst of
   re-triggers is not counted as many false events.

SCORING (NEDC OVLP "any-overlap")
---------------------------------
A TRUE event is DETECTED if ANY predicted event has a strictly positive
temporal intersection with it. A predicted event is FALSE if it overlaps
NO true event. Multiple predictions overlapping one true event count that
true event once, and none of those overlapping predictions are false —
only predictions that overlap zero true events are false.

DEFAULT TIME BASE
-----------------
``sec_per_step = 10.0 / 313`` — the L3 subband timestep. One SNN L3
window is 313 timesteps (~10 s); see lamquant.snn.lma_dataset.L3_T.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# L3 timestep: one ~10 s subband window is L3_T = 313 timesteps.
# Mirrors lamquant.snn.lma_dataset.L3_T (313) and train_mamba_snn's
# L3_T_FOR_FPR. Defined locally so this module stays torch/dataset-free.
SEC_PER_STEP_L3 = 10.0 / 313

Event = Tuple[float, float]  # (start_s, end_s), start_s < end_s


# ===========================================================================
# 1. mask -> events
# ===========================================================================

def events_from_binary(
    mask: Sequence[bool] | np.ndarray,
    sec_per_step: float = SEC_PER_STEP_L3,
    min_event_sec: float = 2.0,
    merge_gap_sec: float = 0.0,
    refractory_sec: float = 0.0,
) -> List[Event]:
    """Group a thresholded per-timestep mask into seizure events.

    Args:
        mask: 1-D boolean (or 0/1) per-timestep array; True == seizure.
        sec_per_step: wall-clock seconds each timestep represents.
        min_event_sec: drop events shorter than this (post-merge) duration.
        merge_gap_sec: merge two adjacent events whose inter-event gap is
            strictly LESS than this many seconds.
        refractory_sec: after an accepted event ends, suppress the ONSET of
            any new event that starts within this many seconds of that end.

    Returns:
        List of ``(start_s, end_s)`` events in ascending start order. Each
        event spans ``[start_s, end_s)`` — a single True sample at index i
        yields ``(i * sec_per_step, (i + 1) * sec_per_step)`` so its
        duration is exactly one step, never zero.

    Semantics notes:
        * An event's time interval is half-open: a run covering sample
          indices ``[a, b]`` (inclusive) maps to ``[a, b+1) * sec_per_step``.
          The "+1" makes a 1-sample event have a non-zero one-step duration
          and makes the gap between two runs measure the number of False
          samples between them.
        * Merge happens BEFORE the min-duration drop, so two short bursts
          that individually fall under ``min_event_sec`` but together (with
          their bridging gap) exceed it survive as one merged event.
        * Refractory is applied LAST, to the merged+filtered list, walking
          left to right: the first event is always kept; a later event is
          dropped if its start is within ``refractory_sec`` of the end of
          the most recently KEPT event.
    """
    if sec_per_step <= 0.0:
        raise ValueError(f"sec_per_step must be > 0, got {sec_per_step}")
    if min_event_sec < 0.0 or merge_gap_sec < 0.0 or refractory_sec < 0.0:
        raise ValueError(
            "min_event_sec, merge_gap_sec, refractory_sec must all be >= 0"
        )

    m = np.asarray(mask).astype(bool).ravel()
    if m.size == 0:
        return []

    # --- step 2: consecutive runs of True -> raw (start_idx, end_idx_excl) ---
    # Pad with False on both ends so every rising/falling edge is captured.
    padded = np.concatenate(([False], m, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)          # indices into m where run begins
    ends = np.flatnonzero(diff == -1)           # one-past-last True index in m
    # starts[k]..ends[k] is the half-open sample span [start, end) of run k.
    runs: List[Tuple[int, int]] = list(zip(starts.tolist(), ends.tolist()))
    if not runs:
        return []

    # --- step 3: merge runs separated by a sub-threshold gap ---
    # gap (in samples) between run k and k+1 is (next_start - cur_end); convert
    # to seconds and merge when strictly less than merge_gap_sec.
    merged: List[Tuple[int, int]] = [runs[0]]
    for s, e in runs[1:]:
        prev_s, prev_e = merged[-1]
        gap_sec = (s - prev_e) * sec_per_step
        if gap_sec < merge_gap_sec:
            merged[-1] = (prev_s, e)            # extend the previous event
        else:
            merged.append((s, e))

    # --- step 4: drop events shorter than min_event_sec ---
    kept_runs: List[Tuple[int, int]] = []
    for s, e in merged:
        dur_sec = (e - s) * sec_per_step
        if dur_sec >= min_event_sec:
            kept_runs.append((s, e))

    # Convert sample spans -> second intervals [start_s, end_s).
    events: List[Event] = [
        (float(s * sec_per_step), float(e * sec_per_step)) for s, e in kept_runs
    ]

    # --- step 5: refractory dead-time after each accepted event ---
    if refractory_sec > 0.0 and events:
        suppressed: List[Event] = [events[0]]
        last_end = events[0][1]
        for start_s, end_s in events[1:]:
            if start_s - last_end < refractory_sec:
                continue                        # onset inside dead-time -> drop
            suppressed.append((start_s, end_s))
            last_end = end_s
        events = suppressed

    return events


# ===========================================================================
# 2. probs -> events
# ===========================================================================

def events_from_probs(
    probs: Sequence[float] | np.ndarray,
    threshold: float,
    sec_per_step: float = SEC_PER_STEP_L3,
    min_event_sec: float = 2.0,
    merge_gap_sec: float = 0.0,
    refractory_sec: float = 0.0,
) -> List[Event]:
    """Threshold a per-timestep probability stream then extract events.

    A timestep is seizure-positive when ``probs >= threshold`` (the same
    ``>=`` convention the trainer's threshold sweep uses), then the boolean
    mask is handed to :func:`events_from_binary`.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    mask = p >= threshold
    return events_from_binary(
        mask,
        sec_per_step=sec_per_step,
        min_event_sec=min_event_sec,
        merge_gap_sec=merge_gap_sec,
        refractory_sec=refractory_sec,
    )


# ===========================================================================
# 3. NEDC OVLP any-overlap scoring
# ===========================================================================

def _overlaps(a: Event, b: Event) -> bool:
    """True iff events ``a`` and ``b`` share a strictly positive intersection.

    Half-open intervals: overlap length is
    ``min(a_end, b_end) - max(a_start, b_start)``; we require it > 0 so two
    events that merely touch end-to-start (a_end == b_start) do NOT overlap.
    """
    return min(a[1], b[1]) - max(a[0], b[0]) > 0.0


def ovlp_score(
    pred_events: Sequence[Event],
    true_events: Sequence[Event],
) -> Dict[str, float | int]:
    """NEDC OVLP "any-overlap" scoring of predicted vs true events.

    Args:
        pred_events: predicted ``(start_s, end_s)`` events.
        true_events: ground-truth ``(start_s, end_s)`` events.

    Returns dict with:
        n_true:           number of true events.
        n_true_detected:  true events with >=1 overlapping prediction.
        n_pred:           number of predicted events.
        n_false_pred:     predictions overlapping ZERO true events.
        n_pred_hit:       predictions overlapping >=1 true event
                          (== n_pred - n_false_pred).
        sens:             n_true_detected / n_true (0.0 if n_true == 0).
        precision:        n_pred_hit / n_pred (0.0 if n_pred == 0).
        n_false_neg:      true events detected by NO prediction
                          (== n_true - n_true_detected).

    OVLP semantics (clinically standard, per NEDC):
        * A true event is detected if ANY prediction overlaps it (>0 s).
        * A prediction is false only if it overlaps NO true event.
        * Two predictions on one true event => that true counts once
          detected, and NEITHER prediction is false (both overlap a true).
    """
    preds = list(pred_events)
    trues = list(true_events)
    n_true = len(trues)
    n_pred = len(preds)

    true_detected = [False] * n_true
    pred_hit = [False] * n_pred

    for pi, pe in enumerate(preds):
        for ti, te in enumerate(trues):
            if _overlaps(pe, te):
                true_detected[ti] = True
                pred_hit[pi] = True
                # do NOT break: this prediction may overlap multiple trues,
                # and every true it touches must be marked detected.

    n_true_detected = int(sum(true_detected))
    n_pred_hit = int(sum(pred_hit))
    n_false_pred = n_pred - n_pred_hit
    n_false_neg = n_true - n_true_detected

    sens = (n_true_detected / n_true) if n_true > 0 else 0.0
    precision = (n_pred_hit / n_pred) if n_pred > 0 else 0.0

    return {
        "n_true": n_true,
        "n_true_detected": n_true_detected,
        "n_pred": n_pred,
        "n_pred_hit": n_pred_hit,
        "n_false_pred": n_false_pred,
        "n_false_neg": n_false_neg,
        "sens": float(sens),
        "precision": float(precision),
    }


# ===========================================================================
# 4. false-event FPR/h
# ===========================================================================

def event_fpr_per_hour(n_false_pred: int, total_seconds: float) -> float:
    """False predicted events per hour of recording.

    Args:
        n_false_pred: number of false predicted events (from
            :func:`ovlp_score`).
        total_seconds: total scored recording duration in seconds.

    Returns:
        ``n_false_pred * 3600 / total_seconds``. Returns 0.0 when
        ``total_seconds <= 0`` (no recording => no rate; avoids div-by-zero).
    """
    if total_seconds <= 0.0:
        return 0.0
    return float(n_false_pred) * 3600.0 / float(total_seconds)


# ===========================================================================
# 5. operating-point calibration over a small grid, per-recording
# ===========================================================================

def calibrate_event_operating_point(
    seqs: Sequence[Tuple[np.ndarray, np.ndarray]],
    sens_floor: float = 0.85,
    sec_per_step: float = SEC_PER_STEP_L3,
    threshold_grid: Sequence[float] | None = None,
    min_event_sec_grid: Sequence[float] = (2.0, 5.0, 10.0),
    merge_gap_sec_grid: Sequence[float] = (2.0, 5.0, 10.0),
    refractory_sec_grid: Sequence[float] = (0.0, 5.0, 10.0),
    select_by: str = "cost",
    fn_weight: float = 100.0,
) -> Dict[str, object]:
    """Sweep the post-processing grid for the best clinical operating point.

    ``select_by`` chooses the objective minimized over the grid:
      * ``"cost"`` (default): J = ``fn_weight``·(1−event_sens) + (1−time_spec).
        Both terms are fractions; FN is weighted ``fn_weight``× a unit of
        false-positive TIME. This is the clinically honest objective — it
        cannot be gamed by a single long "event" the way event-FPR/h can
        (flooding drives time_spec→0 → cost explodes). Each grid point's
        ``time_spec`` is computed from the post-processed predicted events.
      * ``"fpr"``: legacy — among points with event_sens ≥ ``sens_floor``,
        minimize event-FPR/h. Gameable; kept for back-compat / comparison.

    The events MUST be found within each contiguous recording — never across
    a flattened val set — because an event that "spans" the boundary between
    two unrelated recordings is an artifact. So each element of ``seqs`` is a
    single recording's ``(probs_1d, target_mask_1d)`` pair, and events are
    extracted per recording, then POOLED across recordings before scoring.

    Args:
        seqs: list of per-recording ``(probs, target_mask)`` tuples.
            ``probs`` is the 1-D seizure-head probability stream for that
            recording; ``target_mask`` is the 1-D ground-truth seizure mask
            (bool / 0-1) for the same timesteps. The two must be the same
            length within a recording.
        sens_floor: required pooled event-sensitivity (default 0.85, the
            PCCP floor).
        sec_per_step: seconds per timestep (default L3 = 10/313).
        threshold_grid: probability thresholds to sweep. Default
            ``np.linspace(0.1, 0.9, 17)``.
        min_event_sec_grid / merge_gap_sec_grid / refractory_sec_grid:
            post-processing parameter grids.

    Returns dict:
        {threshold, min_event_sec, merge_gap_sec, refractory_sec,
         event_sens, event_fpr_per_h, meets_floor}

    Selection rule:
        Among grid points whose pooled event-sensitivity >= ``sens_floor``,
        choose the one with the MINIMUM pooled event-FPR/h. Ties broken by
        higher sensitivity, then by a more permissive (lower) threshold so
        the chosen point is the most robust at equal FPR. If NO grid point
        meets the floor, return the highest-sensitivity point seen (with
        ``meets_floor=False``) so the caller still gets a usable, honest
        operating point instead of a silent default.
    """
    if threshold_grid is None:
        threshold_grid = np.linspace(0.1, 0.9, 17)

    # Pre-extract per-recording true events once (independent of threshold)
    # and accumulate total recording duration for the FPR/h denominator.
    true_events_per_rec: List[List[Event]] = []
    total_true = 0
    total_seconds = 0.0
    # Validate + normalize sequences up front (rule 30: hostile caller).
    norm_seqs: List[Tuple[np.ndarray, np.ndarray]] = []
    for ri, (probs, target) in enumerate(seqs):
        p = np.asarray(probs, dtype=np.float64).ravel()
        t = np.asarray(target).astype(bool).ravel()
        if p.shape[0] != t.shape[0]:
            raise ValueError(
                f"recording {ri}: probs len {p.shape[0]} != target len "
                f"{t.shape[0]} (probs and target must align per recording)"
            )
        norm_seqs.append((p, t))
        # True events: ground-truth runs. Ground truth is authoritative, so
        # NO post-processing (no min-duration / merge / refractory) — every
        # labeled seizure run is a true event to be detected.
        te = events_from_binary(
            t, sec_per_step=sec_per_step,
            min_event_sec=0.0, merge_gap_sec=0.0, refractory_sec=0.0,
        )
        true_events_per_rec.append(te)
        total_true += len(te)
        total_seconds += p.shape[0] * sec_per_step

    best: Dict[str, object] | None = None
    # Track the best sub-floor fallback (highest sensitivity seen).
    fallback: Dict[str, object] | None = None

    for thr in threshold_grid:
        for min_ev in min_event_sec_grid:
            for merge_gap in merge_gap_sec_grid:
                for refr in refractory_sec_grid:
                    pooled_true = 0
                    pooled_true_detected = 0
                    pooled_false = 0
                    fp_time = 0          # neg timesteps inside a predicted event
                    neg_total = 0        # total neg timesteps (time-spec denom)
                    for (p, t), true_events in zip(norm_seqs, true_events_per_rec):
                        pred_events = events_from_probs(
                            p, threshold=float(thr), sec_per_step=sec_per_step,
                            min_event_sec=float(min_ev),
                            merge_gap_sec=float(merge_gap),
                            refractory_sec=float(refr),
                        )
                        sc = ovlp_score(pred_events, true_events)
                        pooled_true += sc["n_true"]
                        pooled_true_detected += sc["n_true_detected"]
                        pooled_false += sc["n_false_pred"]
                        # Time-based FP: fraction of non-seizure TIME a predicted
                        # event covers. Robust to the long-event gaming of FPR/h.
                        # (t == 0) not ~t — bulletproof if t is int-typed (~int
                        # is bitwise NOT, not logical); norm_seqs casts to bool
                        # already, but be explicit.
                        neg = (t == 0)
                        n_neg = int(neg.sum())
                        if n_neg:
                            neg_total += n_neg
                            pmask = _predicted_positive_mask(
                                pred_events, p.shape[0], sec_per_step)
                            fp_time += int((pmask & neg).sum())

                    event_sens = (
                        pooled_true_detected / pooled_true
                        if pooled_true > 0 else 0.0
                    )
                    event_fpr_h = event_fpr_per_hour(pooled_false, total_seconds)
                    time_spec = (1.0 - fp_time / neg_total) if neg_total else 1.0
                    cost = fn_weight * (1.0 - event_sens) + (1.0 - time_spec)
                    candidate = {
                        "threshold": float(thr),
                        "min_event_sec": float(min_ev),
                        "merge_gap_sec": float(merge_gap),
                        "refractory_sec": float(refr),
                        "event_sens": float(event_sens),
                        "event_fpr_per_h": float(event_fpr_h),
                        "time_specificity": float(time_spec),
                        "cost": float(cost),
                        "meets_floor": bool(event_sens >= sens_floor),
                    }

                    if select_by == "cost":
                        # Minimize J = fn_weight*FNR + (1-time_spec). Ties ->
                        # higher time_spec, then higher sens, then higher thr
                        # (less permissive = safer against flooding).
                        if best is None or _better_cost(candidate, best):
                            best = candidate
                        continue

                    # Legacy FPR-min path (select_by == "fpr").
                    if (fallback is None
                            or candidate["event_sens"] > fallback["event_sens"]
                            or (candidate["event_sens"] == fallback["event_sens"]
                                and candidate["event_fpr_per_h"]
                                < fallback["event_fpr_per_h"])):
                        fallback = candidate
                    if not candidate["meets_floor"]:
                        continue
                    if best is None or _better(candidate, best):
                        best = candidate

    # Pick the result: floor-meeting best, else honest fallback, else empty.
    result = best if best is not None else fallback
    if result is None:
        result = {
            "threshold": float(threshold_grid[0]) if len(threshold_grid) else 0.5,
            "min_event_sec": float(min_event_sec_grid[0]),
            "merge_gap_sec": float(merge_gap_sec_grid[0]),
            "refractory_sec": float(refractory_sec_grid[0]),
            "event_sens": 0.0,
            "event_fpr_per_h": 0.0,
            "time_specificity": 0.0,
            "cost": float(fn_weight) + 1.0,
            "meets_floor": False,
        }
    # Attach specificity at the chosen operating point (clinical sens/spec pair).
    if norm_seqs:
        result.update(specificity_at_operating_point(
            norm_seqs, result, sec_per_step=sec_per_step))
    else:
        result.update({"time_specificity": 0.0, "timestep_specificity": 0.0})
    return result


def _predicted_positive_mask(
    events: Sequence[Event], n_steps: int, sec_per_step: float
) -> np.ndarray:
    """Boolean per-timestep mask of which steps fall inside any predicted event.

    An event ``[start_s, end_s)`` covers timesteps ``[floor(start/dt),
    ceil(end/dt))`` (clamped to ``[0, n_steps)``).
    """
    m = np.zeros(n_steps, dtype=bool)
    for s, e in events:
        i0 = max(0, int(np.floor(s / sec_per_step)))
        i1 = min(n_steps, int(np.ceil(e / sec_per_step)))
        if i1 > i0:
            m[i0:i1] = True
    return m


def specificity_at_operating_point(
    norm_seqs: Sequence[Tuple[np.ndarray, np.ndarray]],
    op: Dict[str, object],
    sec_per_step: float = SEC_PER_STEP_L3,
) -> Dict[str, float]:
    """Specificity of a calibrated operating point, two granularities.

    Both answer "of the truly-non-seizure portion, how much did we correctly
    leave un-flagged?" — the clinical complement to event-sensitivity.

    * ``time_specificity`` — uses the POST-PROCESSED predicted events (the same
      min-duration / merge / refractory the operating point selected). This is
      the clinically meaningful number: 1 − (non-seizure time covered by an
      accepted predicted event) / (total non-seizure time). It rewards the
      post-processing that collapses spurious bursts.
    * ``timestep_specificity`` — RAW per-timestep at the operating threshold,
      BEFORE event post-processing: TN / (TN + FP). Always ≤ time_specificity;
      reported for reference / comparison with per-epoch literature.
    """
    thr = float(op["threshold"])
    min_ev = float(op["min_event_sec"])
    merge_gap = float(op["merge_gap_sec"])
    refr = float(op["refractory_sec"])

    neg_total = 0          # timesteps with target == 0 (true negative universe)
    fp_time = 0            # neg timesteps covered by an accepted predicted event
    raw_fp = 0             # neg timesteps with prob >= thr (pre post-proc)
    for p, t in norm_seqs:
        neg = (t == 0)
        n_neg = int(neg.sum())
        if n_neg == 0:
            continue
        neg_total += n_neg
        pred_events = events_from_probs(
            p, threshold=thr, sec_per_step=sec_per_step,
            min_event_sec=min_ev, merge_gap_sec=merge_gap, refractory_sec=refr)
        pred_mask = _predicted_positive_mask(pred_events, p.shape[0], sec_per_step)
        fp_time += int((pred_mask & neg).sum())
        raw_fp += int(((p >= thr) & neg).sum())

    if neg_total == 0:
        return {"time_specificity": 1.0, "timestep_specificity": 1.0}
    return {
        "time_specificity": float(1.0 - fp_time / neg_total),
        "timestep_specificity": float(1.0 - raw_fp / neg_total),
    }


def _better_cost(cand: Dict[str, object], cur: Dict[str, object]) -> bool:
    """Ranking for select_by='cost'. Lower J wins; ties -> higher time_spec,
    then higher sens, then HIGHER threshold (less permissive = safer)."""
    if cand["cost"] < cur["cost"]:
        return True
    if cand["cost"] > cur["cost"]:
        return False
    if cand["time_specificity"] != cur["time_specificity"]:
        return cand["time_specificity"] > cur["time_specificity"]
    if cand["event_sens"] != cur["event_sens"]:
        return cand["event_sens"] > cur["event_sens"]
    return cand["threshold"] > cur["threshold"]


def _better(cand: Dict[str, object], cur: Dict[str, object]) -> bool:
    """Strict ranking among floor-meeting candidates.

    Lower FPR/h wins; ties -> higher sens; ties -> lower threshold.
    """
    if cand["event_fpr_per_h"] < cur["event_fpr_per_h"]:
        return True
    if cand["event_fpr_per_h"] > cur["event_fpr_per_h"]:
        return False
    if cand["event_sens"] > cur["event_sens"]:
        return True
    if cand["event_sens"] < cur["event_sens"]:
        return False
    return cand["threshold"] < cur["threshold"]
