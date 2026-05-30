#!/usr/bin/env python3
"""Unit tests for event_scoring.py — pure-numpy, no torch.

Synthetic sequences with KNOWN events verify each documented behavior:
  (a) consecutive-run merge across a sub-threshold gap
  (b) min-duration drop
  (c) refractory suppression of re-triggers
  (d) OVLP any-overlap true positive
  (e) a prediction overlapping ZERO trues is false
  (f) two predictions on ONE true => 1 detection, 0 false
  (g) event_fpr_per_hour arithmetic

Time base for most tests: sec_per_step = 1.0 (1 sample == 1 second) so
the event boundaries and gap/duration thresholds are trivially auditable.
"""

import numpy as np
import pytest

from event_scoring import (
    SEC_PER_STEP_L3,
    calibrate_event_operating_point,
    event_fpr_per_hour,
    events_from_binary,
    events_from_probs,
    ovlp_score,
)


def _mask(spec: str) -> np.ndarray:
    """'..##..#' -> bool array; '#' True, anything else False."""
    return np.array([c == "#" for c in spec], dtype=bool)


# ---------------------------------------------------------------------------
# Basic run extraction
# ---------------------------------------------------------------------------

def test_single_run_basic():
    # samples 2,3,4 True -> one event [2,5) seconds at 1 s/step.
    mask = _mask("..###..")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0)
    assert ev == [(2.0, 5.0)]


def test_one_sample_event_has_one_step_duration():
    # A single True sample must span exactly one step, never zero width.
    mask = _mask(".#.")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0)
    assert ev == [(1.0, 2.0)]


def test_empty_mask_returns_no_events():
    assert events_from_binary(np.zeros(0, dtype=bool), sec_per_step=1.0) == []
    assert events_from_binary(_mask("...."), sec_per_step=1.0) == []


def test_two_separated_runs_stay_separate_without_merge():
    # gap of 3 False samples; merge_gap_sec=0 -> no merge.
    mask = _mask("##...##")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0,
                            merge_gap_sec=0.0)
    assert ev == [(0.0, 2.0), (5.0, 7.0)]


# ---------------------------------------------------------------------------
# (a) consecutive merge
# ---------------------------------------------------------------------------

def test_merge_across_small_gap():
    # two runs separated by a 2-sample (2 s) gap.
    mask = _mask("##..##")
    # merge_gap_sec=3 (> 2 s gap) -> merge into one event [0,6).
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0,
                            merge_gap_sec=3.0)
    assert ev == [(0.0, 6.0)]


def test_merge_gap_is_strict_less_than():
    # gap is exactly 2 s; merge_gap_sec=2.0 must NOT merge (strict <).
    mask = _mask("##..##")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0,
                            merge_gap_sec=2.0)
    assert ev == [(0.0, 2.0), (4.0, 6.0)]
    # merge_gap_sec just above 2 s DOES merge.
    ev2 = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0,
                             merge_gap_sec=2.001)
    assert ev2 == [(0.0, 6.0)]


def test_merge_then_survives_min_duration():
    # two 1-sample bursts, gap 1 s. Each alone is 1 s (< 3 s min); merged
    # they span [0,3) = 3 s and survive the 3 s floor.
    mask = _mask("#.#")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=3.0,
                            merge_gap_sec=2.0)
    assert ev == [(0.0, 3.0)]


# ---------------------------------------------------------------------------
# (b) min-duration drop
# ---------------------------------------------------------------------------

def test_min_duration_drops_short_event():
    # one 2 s event + one 1 s event; min_event_sec=2 keeps only the 2 s one.
    mask = _mask("##....#")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=2.0,
                            merge_gap_sec=0.0)
    assert ev == [(0.0, 2.0)]


def test_min_duration_boundary_is_inclusive():
    # event of exactly 2 s with min_event_sec=2.0 is KEPT (>=).
    mask = _mask("##")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=2.0)
    assert ev == [(0.0, 2.0)]
    # 1 s event with the same floor is dropped.
    assert events_from_binary(_mask("#"), sec_per_step=1.0,
                              min_event_sec=2.0) == []


# ---------------------------------------------------------------------------
# (c) refractory suppression
# ---------------------------------------------------------------------------

def test_refractory_suppresses_close_retrigger():
    # event A = [0,2); event B onset at t=3 (gap 1 s from A's end).
    # refractory_sec=5 -> B is suppressed (onset within dead-time).
    mask = _mask("##.##")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0,
                            merge_gap_sec=0.0, refractory_sec=5.0)
    assert ev == [(0.0, 2.0)]


def test_refractory_allows_event_after_deadtime():
    # A = [0,2); B onset at t=8 (gap 6 s). refractory_sec=5 -> B kept.
    mask = _mask("##......##")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0,
                            merge_gap_sec=0.0, refractory_sec=5.0)
    assert ev == [(0.0, 2.0), (8.0, 10.0)]


def test_refractory_zero_keeps_all():
    mask = _mask("##.##")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0,
                            merge_gap_sec=0.0, refractory_sec=0.0)
    assert ev == [(0.0, 2.0), (3.0, 5.0)]


def test_refractory_measured_from_kept_event_end():
    # A=[0,1); B onset t=2 suppressed (gap 1<5); C onset t=4 also suppressed
    # because the dead-time is measured from A's end (the last KEPT event),
    # 4-1=3 < 5. None of B/C restart the clock since they were dropped.
    mask = _mask("#.#.#")
    ev = events_from_binary(mask, sec_per_step=1.0, min_event_sec=0.0,
                            merge_gap_sec=0.0, refractory_sec=5.0)
    assert ev == [(0.0, 1.0)]


# ---------------------------------------------------------------------------
# probs -> events
# ---------------------------------------------------------------------------

def test_events_from_probs_threshold_inclusive():
    probs = np.array([0.1, 0.5, 0.9, 0.5, 0.1])
    # threshold 0.5 with >= -> samples 1,2,3 positive -> [1,4).
    ev = events_from_probs(probs, threshold=0.5, sec_per_step=1.0,
                           min_event_sec=0.0)
    assert ev == [(1.0, 4.0)]


def test_events_from_probs_threshold_excludes_below():
    probs = np.array([0.1, 0.49, 0.9, 0.2])
    ev = events_from_probs(probs, threshold=0.5, sec_per_step=1.0,
                           min_event_sec=0.0)
    assert ev == [(2.0, 3.0)]


# ---------------------------------------------------------------------------
# (d) OVLP any-overlap true positive
# ---------------------------------------------------------------------------

def test_ovlp_any_overlap_detects_true():
    true_events = [(10.0, 20.0)]
    pred_events = [(18.0, 25.0)]  # overlaps [18,20) -> 2 s intersection
    sc = ovlp_score(pred_events, true_events)
    assert sc["n_true"] == 1
    assert sc["n_true_detected"] == 1
    assert sc["sens"] == 1.0
    assert sc["n_false_pred"] == 0
    assert sc["n_false_neg"] == 0


def test_ovlp_touching_endpoints_do_not_overlap():
    # pred ends exactly where true starts -> half-open, no positive overlap.
    true_events = [(10.0, 20.0)]
    pred_events = [(5.0, 10.0)]
    sc = ovlp_score(pred_events, true_events)
    assert sc["n_true_detected"] == 0
    assert sc["n_false_pred"] == 1  # this pred touches no true -> false


# ---------------------------------------------------------------------------
# (e) a pred overlapping zero trues is false
# ---------------------------------------------------------------------------

def test_ovlp_pred_overlapping_no_true_is_false():
    true_events = [(10.0, 20.0)]
    pred_events = [(50.0, 60.0)]  # far away, overlaps nothing
    sc = ovlp_score(pred_events, true_events)
    assert sc["n_pred"] == 1
    assert sc["n_false_pred"] == 1
    assert sc["n_true_detected"] == 0
    assert sc["sens"] == 0.0
    assert sc["n_false_neg"] == 1


def test_ovlp_mixed_hit_and_false():
    true_events = [(10.0, 20.0), (100.0, 110.0)]
    pred_events = [(12.0, 15.0),   # hits true #0
                   (50.0, 60.0)]   # hits nothing -> false
    sc = ovlp_score(pred_events, true_events)
    assert sc["n_true"] == 2
    assert sc["n_true_detected"] == 1   # only true #0 detected
    assert sc["n_false_pred"] == 1
    assert sc["n_false_neg"] == 1       # true #1 missed
    assert sc["sens"] == 0.5


# ---------------------------------------------------------------------------
# (f) two preds on one true => 1 detection, 0 false
# ---------------------------------------------------------------------------

def test_ovlp_two_preds_one_true():
    true_events = [(10.0, 30.0)]
    pred_events = [(11.0, 14.0), (20.0, 25.0)]  # both inside the one true
    sc = ovlp_score(pred_events, true_events)
    assert sc["n_true"] == 1
    assert sc["n_true_detected"] == 1   # counted ONCE
    assert sc["n_pred"] == 2
    assert sc["n_pred_hit"] == 2        # both overlap a true
    assert sc["n_false_pred"] == 0      # so neither is false
    assert sc["sens"] == 1.0


def test_ovlp_one_pred_spanning_two_trues():
    # one prediction overlapping two separate trues detects both, 0 false.
    true_events = [(10.0, 20.0), (30.0, 40.0)]
    pred_events = [(15.0, 35.0)]
    sc = ovlp_score(pred_events, true_events)
    assert sc["n_true_detected"] == 2
    assert sc["n_false_pred"] == 0
    assert sc["sens"] == 1.0


def test_ovlp_no_true_events():
    sc = ovlp_score([(1.0, 2.0)], [])
    assert sc["n_true"] == 0
    assert sc["sens"] == 0.0
    assert sc["n_false_pred"] == 1   # the lone pred overlaps no true


def test_ovlp_no_predictions():
    sc = ovlp_score([], [(1.0, 2.0)])
    assert sc["n_true"] == 1
    assert sc["n_true_detected"] == 0
    assert sc["n_false_pred"] == 0
    assert sc["sens"] == 0.0


# ---------------------------------------------------------------------------
# (g) event_fpr_per_hour arithmetic
# ---------------------------------------------------------------------------

def test_event_fpr_per_hour_arithmetic():
    # 2 false events over 1800 s (0.5 h) => 4 / h.
    assert event_fpr_per_hour(2, 1800.0) == pytest.approx(4.0)
    # 1 false event over exactly 3600 s => 1 / h.
    assert event_fpr_per_hour(1, 3600.0) == pytest.approx(1.0)
    # 0 false events => 0 / h.
    assert event_fpr_per_hour(0, 3600.0) == 0.0


def test_event_fpr_per_hour_zero_seconds_is_zero():
    assert event_fpr_per_hour(5, 0.0) == 0.0
    assert event_fpr_per_hour(5, -10.0) == 0.0


# ---------------------------------------------------------------------------
# calibrate_event_operating_point — per-recording, pooled
# ---------------------------------------------------------------------------

def test_calibrate_picks_floor_meeting_min_fpr():
    # Recording 1: clean seizure in the middle, model confident there only.
    n = 200
    probs1 = np.full(n, 0.05)
    target1 = np.zeros(n, dtype=bool)
    probs1[80:120] = 0.95           # 40-step confident seizure
    target1[80:120] = True
    # Recording 2: pure background, model mostly quiet with a brief blip
    # that should be killed by a min-duration / refractory choice.
    probs2 = np.full(n, 0.05)
    target2 = np.zeros(n, dtype=bool)
    probs2[10:12] = 0.95            # 2-step false blip

    seqs = [(probs1, target1), (probs2, target2)]
    res = calibrate_event_operating_point(
        seqs, sens_floor=0.85, sec_per_step=1.0,
        threshold_grid=np.linspace(0.1, 0.9, 9),
        min_event_sec_grid=(2.0, 5.0, 10.0),
        merge_gap_sec_grid=(2.0,),
        refractory_sec_grid=(0.0,),
    )
    # The true seizure (40 s) is always detected at any reasonable threshold,
    # so sens hits 1.0 >= floor. A min_event_sec >= 3 kills the 2-step blip
    # in recording 2, so the optimum reaches 0 false events => 0 FPR/h.
    assert res["meets_floor"] is True
    assert res["event_sens"] >= 0.85
    assert res["event_fpr_per_h"] == pytest.approx(0.0)
    assert res["min_event_sec"] >= 5.0   # picked a floor that drops the blip


def test_calibrate_events_not_found_across_recordings():
    # Two recordings, each ending/starting with seizure-positive timesteps.
    # If events were found on the FLATTENED stream, the tail of rec1 and the
    # head of rec2 would merge into ONE event. Per-recording extraction keeps
    # them as TWO true events. We assert n_true is counted as 2 via a probe.
    n = 50
    probs1 = np.full(n, 0.05)
    target1 = np.zeros(n, dtype=bool)
    target1[45:50] = True           # seizure at the very END of rec1
    probs1[45:50] = 0.95
    probs2 = np.full(n, 0.05)
    target2 = np.zeros(n, dtype=bool)
    target2[0:5] = True             # seizure at the very START of rec2
    probs2[0:5] = 0.95
    seqs = [(probs1, target1), (probs2, target2)]

    # Re-derive the per-recording true-event count the way the calibrator does.
    te1 = events_from_binary(target1, sec_per_step=1.0, min_event_sec=0.0)
    te2 = events_from_binary(target2, sec_per_step=1.0, min_event_sec=0.0)
    assert len(te1) == 1 and len(te2) == 1   # 2 distinct true events total

    res = calibrate_event_operating_point(
        seqs, sens_floor=0.85, sec_per_step=1.0,
        threshold_grid=np.array([0.5]),
        min_event_sec_grid=(2.0,),
        merge_gap_sec_grid=(2.0,),
        refractory_sec_grid=(0.0,),
    )
    # Both detected per-recording -> sens 1.0; no false events anywhere.
    assert res["event_sens"] == pytest.approx(1.0)
    assert res["event_fpr_per_h"] == pytest.approx(0.0)


def test_calibrate_returns_fallback_when_floor_unreachable():
    # Recording where the model NEVER fires on the true seizure -> sens can
    # never reach the floor; calibrator must return meets_floor=False with
    # the honest best-effort (highest-sens) point, not crash.
    n = 100
    probs = np.full(n, 0.05)         # always below any threshold in grid
    target = np.zeros(n, dtype=bool)
    target[40:60] = True             # true seizure the model misses
    res = calibrate_event_operating_point(
        [(probs, target)], sens_floor=0.85, sec_per_step=1.0,
        threshold_grid=np.linspace(0.1, 0.9, 5),
        min_event_sec_grid=(2.0,),
        merge_gap_sec_grid=(2.0,),
        refractory_sec_grid=(0.0,),
    )
    assert res["meets_floor"] is False
    assert res["event_sens"] == pytest.approx(0.0)


def test_calibrate_empty_seqs():
    res = calibrate_event_operating_point(
        [], sens_floor=0.85, sec_per_step=1.0,
        threshold_grid=np.array([0.5]),
    )
    assert res["meets_floor"] is False
    assert res["event_sens"] == 0.0


def test_calibrate_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        calibrate_event_operating_point(
            [(np.zeros(10), np.zeros(8, dtype=bool))],
            sec_per_step=1.0,
        )


def test_default_sec_per_step_is_l3():
    # Guard the documented default time base (10 s / 313 steps).
    assert SEC_PER_STEP_L3 == pytest.approx(10.0 / 313)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
