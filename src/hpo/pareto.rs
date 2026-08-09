// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Multi-objective HPO: Pareto dominance, the non-dominated front, and 2-D
//! hypervolume (ADR 0109).
//!
//! The rest of `hpo` is single-objective — one `f64` and a `max`/`min` mode
//! (see [`crate::hpo::results::HpoManifest`]). ADR 0109 makes objectives a
//! VECTOR, and says the deliverable of a multi-objective study is the Pareto
//! front, **not a single "best"**: with competing objectives (recon quality vs
//! compression ratio vs GPU-seconds) no total order exists, so collapsing to a
//! scalar picks a winner by an arbitrary weighting nobody declared.
//!
//! This module is deliberately PURE — no I/O, no plan, no scheduler. It is the
//! piece qEHVI acquisition and the `pareto.json` artifact are built on, and
//! keeping it free of engine state is what makes it exhaustively testable.
//!
//! **Non-finite objectives are excluded, never ranked.** A NaN comparison is
//! false in both directions, so a NaN-bearing point would be non-dominated by
//! construction and would silently land on the front as a fake optimum. A
//! trial that failed to produce a finite measurement is not a data point (the
//! fail-closed rule: unmeasured ⇒ excluded, never a free pass).

use serde::{Deserialize, Serialize};

/// Which way an objective is better.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Minimize,
    Maximize,
}

impl Direction {
    /// Parse the `max`/`min` vocabulary [`HpoManifest`](crate::hpo::results::HpoManifest)
    /// already uses, so a multi-objective study config does not invent a second
    /// spelling of the same idea.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "max" | "maximize" => Some(Self::Maximize),
            "min" | "minimize" => Some(Self::Minimize),
            _ => None,
        }
    }

    /// `true` when `a` is strictly better than `b` on this axis.
    fn better(self, a: f64, b: f64) -> bool {
        match self {
            Self::Minimize => a < b,
            Self::Maximize => a > b,
        }
    }
}

/// One evaluated trial in objective space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub trial_id: u32,
    /// One value per objective, in the study's declared objective order.
    pub objectives: Vec<f64>,
    /// Measured cost (GPU-seconds, dollars, or ε) — carried for the
    /// cost-aware acquisition wrapper. Not an objective unless declared one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

impl ParetoPoint {
    pub fn new(trial_id: u32, objectives: Vec<f64>) -> Self {
        Self {
            trial_id,
            objectives,
            cost: None,
        }
    }

    /// Every objective is finite. Points failing this are excluded from the
    /// front rather than ranked (see the module note on NaN).
    pub fn is_measured(&self) -> bool {
        self.objectives.iter().all(|v| v.is_finite())
    }
}

/// Does `a` Pareto-dominate `b`? True when `a` is at least as good on every
/// objective and strictly better on at least one.
///
/// Returns `false` if the dimensionalities disagree or either point carries a
/// non-finite value — a malformed comparison must not manufacture an ordering.
pub fn dominates(a: &[f64], b: &[f64], dirs: &[Direction]) -> bool {
    if a.len() != b.len() || a.len() != dirs.len() || a.is_empty() {
        return false;
    }
    if a.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return false;
    }
    let mut strictly_better_somewhere = false;
    for ((&x, &y), &d) in a.iter().zip(b.iter()).zip(dirs.iter()) {
        if d.better(y, x) {
            return false; // b wins this axis ⇒ a cannot dominate
        }
        if d.better(x, y) {
            strictly_better_somewhere = true;
        }
    }
    strictly_better_somewhere
}

/// Indices of the non-dominated (Pareto-optimal) points.
///
/// Points with any non-finite objective, or with the wrong dimensionality, are
/// dropped up front — they are unmeasured, not optimal. Duplicates of the same
/// objective vector are all retained: neither dominates the other, and silently
/// collapsing them would hide that two distinct configs tied.
pub fn non_dominated(points: &[ParetoPoint], dirs: &[Direction]) -> Vec<usize> {
    let eligible: Vec<usize> = points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_measured() && p.objectives.len() == dirs.len() && !dirs.is_empty())
        .map(|(i, _)| i)
        .collect();

    eligible
        .iter()
        .copied()
        .filter(|&i| {
            !eligible
                .iter()
                .any(|&j| j != i && dominates(&points[j].objectives, &points[i].objectives, dirs))
        })
        .collect()
}

/// The Pareto front itself, ordered by the first objective for stable output.
pub fn pareto_front(points: &[ParetoPoint], dirs: &[Direction]) -> Vec<ParetoPoint> {
    let mut front: Vec<ParetoPoint> = non_dominated(points, dirs)
        .into_iter()
        .map(|i| points[i].clone())
        .collect();
    front.sort_by(|a, b| {
        a.objectives[0]
            .partial_cmp(&b.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.trial_id.cmp(&b.trial_id))
    });
    front
}

/// `pareto.json` — the multi-objective study's deliverable artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParetoReport {
    /// Objective names, in the same order as every `objectives` vector.
    pub objectives: Vec<String>,
    pub directions: Vec<Direction>,
    pub front: Vec<ParetoPoint>,
    /// Trials that produced no finite measurement — reported, not silently
    /// dropped, so a study whose trials mostly failed cannot read as a clean
    /// small front.
    pub unmeasured_trials: Vec<u32>,
}

impl ParetoReport {
    pub fn build(
        objectives: Vec<String>,
        directions: Vec<Direction>,
        points: &[ParetoPoint],
    ) -> Self {
        let unmeasured_trials = points
            .iter()
            .filter(|p| !p.is_measured() || p.objectives.len() != directions.len())
            .map(|p| p.trial_id)
            .collect();
        Self {
            front: pareto_front(points, &directions),
            objectives,
            directions,
            unmeasured_trials,
        }
    }
}

/// 2-D hypervolume dominated by `front` with respect to `reference`.
///
/// This is the scalar quality measure for a 2-objective front and the base
/// qEHVI acquisition will improve against. Restricted to 2-D on purpose: the
/// exact N-D computation is a different algorithm, and a wrong general
/// implementation is worse than an honest narrow one.
///
/// Points not dominating the reference contribute nothing (never a negative
/// box). Returns `0.0` for an empty or non-2-D front.
pub fn hypervolume_2d(front: &[ParetoPoint], reference: [f64; 2], dirs: &[Direction]) -> f64 {
    if dirs.len() != 2 {
        return 0.0;
    }
    // Normalize to a pure minimization problem in the positive orthant: flip
    // maximized axes so one sweep handles every direction combination.
    let flip = |v: f64, d: Direction| if d == Direction::Maximize { -v } else { v };
    let r = [flip(reference[0], dirs[0]), flip(reference[1], dirs[1])];

    let mut pts: Vec<[f64; 2]> = front
        .iter()
        .filter(|p| p.is_measured() && p.objectives.len() == 2)
        .map(|p| {
            [
                flip(p.objectives[0], dirs[0]),
                flip(p.objectives[1], dirs[1]),
            ]
        })
        .filter(|p| p[0] < r[0] && p[1] < r[1])
        .collect();
    if pts.is_empty() {
        return 0.0;
    }
    // Sweep ascending in x; track the best y so far. Each point contributes the
    // slab (x_{i+1} − x_i) wide by (r_y − y_best) tall.
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    let mut area = 0.0;
    let mut best_y = r[1];
    let mut prev_x = pts[0][0];
    for p in &pts {
        area += (p[0] - prev_x) * (r[1] - best_y).max(0.0);
        prev_x = p[0];
        best_y = best_y.min(p[1]);
    }
    area += (r[0] - prev_x) * (r[1] - best_y).max(0.0);
    area
}

#[cfg(test)]
mod tests {
    use super::*;
    use Direction::{Maximize, Minimize};

    fn p(id: u32, objs: &[f64]) -> ParetoPoint {
        ParetoPoint::new(id, objs.to_vec())
    }

    #[test]
    fn dominance_requires_at_least_as_good_everywhere_and_better_somewhere() {
        let d = [Minimize, Minimize];
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0], &d));
        assert!(dominates(&[1.0, 2.0], &[1.0, 3.0], &d), "tie on one axis");
        // Equal points do not dominate each other.
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0], &d));
        // A trade-off is not dominance in either direction.
        assert!(!dominates(&[1.0, 5.0], &[5.0, 1.0], &d));
        assert!(!dominates(&[5.0, 1.0], &[1.0, 5.0], &d));
    }

    #[test]
    fn direction_is_respected_per_axis() {
        // Maximize accuracy, minimize latency.
        let d = [Maximize, Minimize];
        assert!(dominates(&[0.9, 10.0], &[0.8, 20.0], &d));
        assert!(!dominates(&[0.8, 10.0], &[0.9, 20.0], &d), "worse accuracy");
        // Mixed directions must not be collapsed into one sense.
        assert!(!dominates(&[0.9, 20.0], &[0.8, 10.0], &d));
    }

    #[test]
    fn non_finite_objectives_never_reach_the_front() {
        let d = [Minimize, Minimize];
        // NaN compares false both ways, so an unguarded implementation would
        // rank this as non-dominated — the exact bug this guards.
        let pts = vec![p(1, &[f64::NAN, 0.0]), p(2, &[5.0, 5.0])];
        assert_eq!(non_dominated(&pts, &d), vec![1], "only the measured trial");
        assert!(!dominates(&[f64::NAN, 0.0], &[5.0, 5.0], &d));
        assert!(!dominates(&[5.0, 5.0], &[f64::NAN, 0.0], &d));
        // Infinity is equally excluded.
        let pts = vec![p(1, &[f64::NEG_INFINITY, 0.0]), p(2, &[5.0, 5.0])];
        assert_eq!(non_dominated(&pts, &d), vec![1]);
    }

    #[test]
    fn front_keeps_every_trade_off_and_drops_the_dominated() {
        let d = [Minimize, Minimize];
        let pts = vec![
            p(1, &[1.0, 9.0]), // on front
            p(2, &[5.0, 5.0]), // on front
            p(3, &[9.0, 1.0]), // on front
            p(4, &[6.0, 6.0]), // dominated by 2
            p(5, &[2.0, 9.5]), // dominated by 1
        ];
        let ids: Vec<u32> = pareto_front(&pts, &d).iter().map(|q| q.trial_id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn duplicate_objective_vectors_both_survive() {
        let d = [Minimize, Minimize];
        let pts = vec![p(1, &[2.0, 2.0]), p(2, &[2.0, 2.0])];
        // Neither dominates the other; hiding one would conceal a real tie
        // between two distinct configurations.
        assert_eq!(non_dominated(&pts, &d).len(), 2);
    }

    #[test]
    fn mismatched_dimensionality_is_rejected_not_guessed() {
        let d = [Minimize, Minimize];
        assert!(!dominates(&[1.0], &[2.0, 2.0], &d));
        assert!(!dominates(&[1.0, 1.0], &[2.0, 2.0], &[Minimize]));
        // Index 0 declares one objective against two directions ⇒ ineligible.
        let pts = vec![p(1, &[1.0]), p(2, &[5.0, 5.0])];
        assert_eq!(non_dominated(&pts, &d), vec![1_usize]);
    }

    #[test]
    fn empty_and_degenerate_inputs_are_safe() {
        assert!(non_dominated(&[], &[Minimize]).is_empty());
        assert!(pareto_front(&[], &[Minimize]).is_empty());
        // No directions declared ⇒ nothing is optimal (fail closed).
        assert!(non_dominated(&[p(1, &[1.0])], &[]).is_empty());
        assert!(!dominates(&[], &[], &[]));
    }

    #[test]
    fn report_names_the_unmeasured_trials_instead_of_hiding_them() {
        let dirs = vec![Minimize, Maximize];
        let pts = vec![
            p(1, &[1.0, 0.9]),
            p(2, &[f64::NAN, 0.5]),
            p(3, &[2.0, 0.95]),
        ];
        let r = ParetoReport::build(vec!["loss".into(), "ratio".into()], dirs, &pts);
        assert_eq!(r.unmeasured_trials, vec![2]);
        let ids: Vec<u32> = r.front.iter().map(|q| q.trial_id).collect();
        assert_eq!(ids, vec![1, 3], "both are genuine trade-offs");
        // Round-trips as the pareto.json artifact.
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<ParetoReport>(&json).unwrap(), r);
    }

    #[test]
    fn hypervolume_2d_matches_hand_computed_areas() {
        let d = [Minimize, Minimize];
        // One point at (1,1) against reference (2,2) ⇒ a 1×1 box.
        assert!((hypervolume_2d(&[p(1, &[1.0, 1.0])], [2.0, 2.0], &d) - 1.0).abs() < 1e-12);
        // Staircase (1,3) and (3,1) vs reference (4,4):
        //   x∈[1,3) capped at y=3 ⇒ 2×1 = 2; x∈[3,4) capped at y=1 ⇒ 1×3 = 3.
        let front = vec![p(1, &[1.0, 3.0]), p(2, &[3.0, 1.0])];
        assert!((hypervolume_2d(&front, [4.0, 4.0], &d) - 5.0).abs() < 1e-12);
        // Adding a dominated point cannot change the volume.
        let mut with_dominated = front.clone();
        with_dominated.push(p(3, &[3.5, 3.5]));
        assert!(
            (hypervolume_2d(&with_dominated, [4.0, 4.0], &d)
                - hypervolume_2d(&front, [4.0, 4.0], &d))
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn hypervolume_ignores_points_worse_than_the_reference() {
        let d = [Minimize, Minimize];
        // Entirely beyond the reference ⇒ no volume, never negative.
        assert_eq!(hypervolume_2d(&[p(1, &[5.0, 5.0])], [2.0, 2.0], &d), 0.0);
        assert_eq!(hypervolume_2d(&[], [2.0, 2.0], &d), 0.0);
        // Wrong arity is 0.0 rather than a bogus area.
        assert_eq!(
            hypervolume_2d(&[p(1, &[1.0, 1.0])], [2.0, 2.0], &[Minimize]),
            0.0
        );
    }

    #[test]
    fn hypervolume_handles_maximized_axes() {
        // Maximize both: reference (0,0), point (2,3) ⇒ 6.
        let d = [Maximize, Maximize];
        assert!((hypervolume_2d(&[p(1, &[2.0, 3.0])], [0.0, 0.0], &d) - 6.0).abs() < 1e-12);
        // A better point strictly increases the volume.
        let a = hypervolume_2d(&[p(1, &[2.0, 3.0])], [0.0, 0.0], &d);
        let b = hypervolume_2d(&[p(1, &[3.0, 4.0])], [0.0, 0.0], &d);
        assert!(b > a);
    }

    #[test]
    fn direction_parses_the_existing_manifest_vocabulary() {
        assert_eq!(Direction::parse("max"), Some(Maximize));
        assert_eq!(Direction::parse("MIN"), Some(Minimize));
        assert_eq!(Direction::parse("minimize"), Some(Minimize));
        assert_eq!(Direction::parse("lowest"), None);
    }
}
