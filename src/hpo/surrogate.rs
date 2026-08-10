// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0109 surrogates: a `Grid` baseline, multivariate TPE, and the
//! cost-aware acquisition wrapper.
//!
//! These sit on the existing [`Sampler`] seam rather than beside it — the
//! landed `RandomSampler` and [`TpeSampler`](super::tpe::TpeSampler) stay the
//! baselines they already were, and everything here reuses their t-space
//! transforms so a `log_uniform` dim is treated identically by every sampler.
//!
//! **Why a multivariate TPE at all.** The landed TPE chooses each dimension
//! INDEPENDENTLY (`suggest_continuous` / `suggest_choice` run per dim), so it
//! models the marginal of each axis and is blind to interaction. When the good
//! region is diagonal — a big learning rate only works with a big batch — the
//! product of marginals puts mass on (big lr, small batch), a corner neither
//! observation supports. The multivariate variant centres its kernels on whole
//! observed POINTS and scores candidate configurations jointly, so a
//! correlation that exists in the data survives into the suggestion.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::Value;

use super::sampler::Sampler;
use super::space::{Dist, Overlay, SearchSpace, TrialResult};
use super::tpe::{from_t, gaussian, support_t, to_t};

// ───────────────────────────── Grid baseline ─────────────────────────────

/// Deterministic grid search — the honest control every model-based sampler
/// should have to beat.
///
/// Continuous dims are discretised into `levels` points **in t-space**, so a
/// `log_uniform` dim is log-spaced rather than bunched at the top of its
/// range. Categorical dims use their choices verbatim; an `int_uniform` narrower
/// than `levels` uses each integer once instead of emitting duplicates.
///
/// The cartesian product is never materialised — it is astronomically large for
/// a real space. Points are addressed by index and decoded mixed-radix, so this
/// costs O(dims) per suggestion and O(1) memory. After the product is exhausted
/// it WRAPS and repeats; a grid is a fixed finite design, and re-suggesting a
/// point already evaluated is a cache hit rather than new spend (ADR 0109).
pub struct GridSampler {
    levels: usize,
    /// Next linear index into the product.
    cursor: u64,
}

impl GridSampler {
    pub fn new(levels: usize) -> Self {
        Self {
            levels: levels.max(1),
            cursor: 0,
        }
    }

    /// This dim's ordered levels. Never empty for a valid `Dist`.
    fn levels_for(&self, dist: &Dist) -> Vec<Value> {
        if let Dist::Choice { choices } = dist {
            return choices.clone();
        }
        let Some((lo, hi)) = support_t(dist) else {
            return vec![Value::Null];
        };
        // An integer dim with fewer integers than requested levels gets one
        // level per integer — asking for 8 levels across [0, 2] must not emit
        // the same value three times and call it a grid.
        let n = match dist {
            Dist::IntUniform { low, high } => {
                let span = (high - low + 1).max(1) as usize;
                span.min(self.levels)
            }
            _ => self.levels,
        };
        if n <= 1 {
            return vec![from_t(dist, (lo + hi) / 2.0)];
        }
        // Endpoints included: a grid that never evaluates its own bounds is a
        // worse control than one that does.
        (0..n)
            .map(|i| {
                let f = i as f64 / (n - 1) as f64;
                from_t(dist, lo + f * (hi - lo))
            })
            .collect()
    }

    /// Total points in the product, saturating (a large space is capped rather
    /// than overflowing into a tiny modulus, which would silently shrink the
    /// grid to a handful of points).
    fn product_size(&self, space: &SearchSpace) -> u64 {
        space
            .dims
            .values()
            .map(|d| self.levels_for(d).len().max(1) as u64)
            .try_fold(1u64, |acc, n| acc.checked_mul(n))
            .unwrap_or(u64::MAX)
    }
}

impl Sampler for GridSampler {
    fn ask(&mut self, space: &SearchSpace, _completed: &[TrialResult]) -> Overlay {
        let total = self.product_size(space);
        let idx = if total == 0 { 0 } else { self.cursor % total };
        self.cursor = self.cursor.wrapping_add(1);

        // Mixed-radix decode over the dims in BTreeMap order (deterministic).
        let mut rem = idx;
        let mut out: Overlay = Vec::with_capacity(space.dims.len());
        for (name, dist) in &space.dims {
            let lv = self.levels_for(dist);
            let radix = lv.len().max(1) as u64;
            let pick = (rem % radix) as usize;
            rem /= radix;
            out.push((name.clone(), lv[pick.min(lv.len() - 1)].clone()));
        }
        out
    }
}

// ─────────────────────────── multivariate TPE ────────────────────────────

/// Multivariate TPE tuning. Deliberately mirrors [`TpeConfig`](super::tpe::TpeConfig)
/// so switching between the two is a sampler swap, not a re-tune.
#[derive(Clone, Debug)]
pub struct MvTpeConfig {
    /// Top quantile counted as "good".
    pub gamma: f64,
    /// Whole candidate configurations scored per `ask`.
    pub n_candidates: usize,
    /// Suggest randomly until this many trials complete.
    pub n_startup: usize,
    /// Kernel bandwidth as a fraction of each dim's t-space support.
    pub bw_factor: f64,
    /// `true` = larger objective is better.
    pub maximize: bool,
}

impl Default for MvTpeConfig {
    fn default() -> Self {
        Self {
            gamma: 0.25,
            n_candidates: 32,
            n_startup: 5,
            bw_factor: 0.15,
            maximize: true,
        }
    }
}

/// TPE whose densities are JOINT over the whole configuration.
pub struct MvTpeSampler {
    cfg: MvTpeConfig,
    rng: StdRng,
}

impl MvTpeSampler {
    pub fn new(cfg: MvTpeConfig, seed: u64) -> Self {
        Self {
            cfg,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Split completed trials into (good, bad) by the gamma quantile.
    ///
    /// Trials whose objective is non-finite are DROPPED, not ranked: a NaN
    /// sorts unpredictably and would otherwise decide which region is "good".
    fn split<'a>(
        &self,
        completed: &'a [TrialResult],
    ) -> (Vec<&'a TrialResult>, Vec<&'a TrialResult>) {
        let mut usable: Vec<&TrialResult> = completed
            .iter()
            .filter(|t| t.objective.is_finite())
            .collect();
        usable.sort_by(|a, b| {
            let (x, y) = if self.cfg.maximize {
                (b.objective, a.objective)
            } else {
                (a.objective, b.objective)
            };
            x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
        });
        // At least one good trial once anything is usable, else the "good"
        // model is empty and the ratio is meaningless.
        let n_good = ((usable.len() as f64) * self.cfg.gamma).ceil() as usize;
        let n_good = n_good.clamp(1, usable.len().max(1)).min(usable.len());
        let bad = usable.split_off(n_good);
        (usable, bad)
    }

    /// Joint Parzen density of `cand` under `obs`.
    ///
    /// This is what makes the sampler multivariate: each observation
    /// contributes ONE kernel that is the product across dims, so mass sits on
    /// the combinations actually observed rather than on the product of the
    /// per-axis marginals.
    fn joint_density(&self, space: &SearchSpace, cand: &Overlay, obs: &[&TrialResult]) -> f64 {
        let value_of = |ov: &Overlay, name: &str| -> Option<Value> {
            ov.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
        };
        // Uniform prior carrying the weight of one pseudo-observation, so a
        // region with no data never yields a zero (and a -inf log-ratio).
        let mut acc = 1.0_f64;
        for dist in space.dims.values() {
            acc *= match dist {
                Dist::Choice { choices } => 1.0 / choices.len().max(1) as f64,
                _ => {
                    let (lo, hi) = support_t(dist).unwrap_or((0.0, 1.0));
                    1.0 / (hi - lo).abs().max(1e-12)
                }
            };
        }
        let mut total = acc;

        for o in obs {
            let mut k = 1.0_f64;
            for (name, dist) in &space.dims {
                let (Some(cv), Some(ov)) = (value_of(cand, name), value_of(&o.overlay, name))
                else {
                    continue;
                };
                k *= match dist {
                    Dist::Choice { choices } => {
                        // Laplace-smoothed match/mismatch, so an unobserved
                        // category keeps non-zero mass.
                        let n = choices.len().max(1) as f64;
                        if cv == ov {
                            (1.0 + 1.0) / (1.0 + n)
                        } else {
                            1.0 / (1.0 + n)
                        }
                    }
                    _ => {
                        let (lo, hi) = support_t(dist).unwrap_or((0.0, 1.0));
                        let bw = ((hi - lo) * self.cfg.bw_factor).abs().max(1e-9);
                        let (Some(c), Some(m)) = (cv.as_f64(), ov.as_f64()) else {
                            continue;
                        };
                        gaussian(to_t(dist, c), to_t(dist, m), bw)
                    }
                };
            }
            total += k;
        }
        total / (obs.len() as f64 + 1.0)
    }

    /// A candidate: jitter a randomly chosen good observation across ALL dims
    /// at once (preserving its joint structure), or draw fresh from the prior.
    fn propose(&mut self, space: &SearchSpace, good: &[&TrialResult]) -> Overlay {
        if good.is_empty() || self.rng.gen_bool(0.25) {
            return space.sample(&mut self.rng);
        }
        let base = &good[self.rng.gen_range(0..good.len())].overlay;
        let mut out: Overlay = Vec::with_capacity(space.dims.len());
        for (name, dist) in &space.dims {
            let from_base = base.iter().find(|(k, _)| k == name).map(|(_, v)| v);
            let v = match (dist, from_base) {
                (Dist::Choice { .. }, Some(v)) => v.clone(),
                (_, Some(v)) => match (v.as_f64(), support_t(dist)) {
                    (Some(x), Some((lo, hi))) => {
                        let bw = ((hi - lo) * self.cfg.bw_factor).abs().max(1e-9);
                        let z: f64 = {
                            // Box-Muller, so the jitter is Gaussian rather than
                            // uniform — it must match the kernel it is scored by.
                            let u1: f64 = self.rng.gen_range(f64::MIN_POSITIVE..1.0);
                            let u2: f64 = self.rng.gen_range(0.0..1.0);
                            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                        };
                        from_t(dist, (to_t(dist, x) + z * bw).clamp(lo, hi))
                    }
                    _ => dist.sample(&mut self.rng),
                },
                _ => dist.sample(&mut self.rng),
            };
            out.push((name.clone(), v));
        }
        out
    }
}

impl Sampler for MvTpeSampler {
    fn ask(&mut self, space: &SearchSpace, completed: &[TrialResult]) -> Overlay {
        let usable = completed.iter().filter(|t| t.objective.is_finite()).count();
        if usable < self.cfg.n_startup {
            return space.sample(&mut self.rng);
        }
        let (good, bad) = self.split(completed);
        let mut best: Option<(f64, Overlay)> = None;
        for _ in 0..self.cfg.n_candidates.max(1) {
            let cand = self.propose(space, &good);
            let l = self.joint_density(space, &cand, &good);
            let g = self.joint_density(space, &cand, &bad);
            let score = l.ln() - g.ln();
            if score.is_finite() && best.as_ref().is_none_or(|(bs, _)| score > *bs) {
                best = Some((score, cand));
            }
        }
        best.map(|(_, o)| o)
            .unwrap_or_else(|| space.sample(&mut self.rng))
    }
}

// ────────────────────── cost-aware acquisition wrapper ───────────────────

/// One trial's measured cost (GPU-seconds, dollars, or ε) alongside its config.
#[derive(Clone, Debug)]
pub struct CostObservation {
    pub overlay: Overlay,
    pub cost: f64,
}

/// The "second cheap surrogate fit on the measured cost column" (ADR 0109).
///
/// Inverse-distance weighting in normalised t-space: near an observed config it
/// returns that config's cost; far from everything it returns the mean. That is
/// deliberately humble — the point is to stop the search preferring a
/// marginally better config that costs ten times more, and a k-NN average is
/// enough for that. A GP here would be a second thing to tune for no decision
/// it changes.
#[derive(Clone, Debug, Default)]
pub struct CostModel {
    obs: Vec<CostObservation>,
}

impl CostModel {
    /// Fit on measured costs. Non-finite or non-positive costs are DISCARDED:
    /// they are not measurements, and a zero would later divide.
    pub fn fit(observations: impl IntoIterator<Item = CostObservation>) -> Self {
        Self {
            obs: observations
                .into_iter()
                .filter(|o| o.cost.is_finite() && o.cost > 0.0)
                .collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
    }

    fn mean(&self) -> f64 {
        if self.obs.is_empty() {
            return 1.0;
        }
        self.obs.iter().map(|o| o.cost).sum::<f64>() / self.obs.len() as f64
    }

    /// Predicted cost of `cand`. Always finite and strictly positive.
    pub fn predict(&self, space: &SearchSpace, cand: &Overlay) -> f64 {
        if self.obs.is_empty() {
            return 1.0;
        }
        let dist_to = |o: &CostObservation| -> f64 {
            let mut d2 = 0.0;
            for (name, dist) in &space.dims {
                let get = |ov: &Overlay| ov.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone());
                let (Some(a), Some(b)) = (get(cand), get(&o.overlay)) else {
                    continue;
                };
                match dist {
                    Dist::Choice { .. } => {
                        if a != b {
                            d2 += 1.0;
                        }
                    }
                    _ => {
                        let (lo, hi) = support_t(dist).unwrap_or((0.0, 1.0));
                        let span = (hi - lo).abs().max(1e-12);
                        if let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) {
                            let t = (to_t(dist, x) - to_t(dist, y)) / span;
                            d2 += t * t;
                        }
                    }
                }
            }
            d2.sqrt()
        };

        let mut num = 0.0;
        let mut den = 0.0;
        for o in &self.obs {
            let d = dist_to(o);
            if d < 1e-9 {
                return o.cost; // exactly an observed point
            }
            let w = 1.0 / (d * d);
            num += w * o.cost;
            den += w;
        }
        if den <= 0.0 || !num.is_finite() || !den.is_finite() {
            return self.mean();
        }
        let p = num / den;
        if p.is_finite() && p > 0.0 {
            p
        } else {
            self.mean()
        }
    }
}

/// Cost-aware acquisition: `acq / predicted_cost` (ADR 0109's `AcqPerCost`).
///
/// Off by default in a study config — it changes what "best" means, and a study
/// that never said it cares about cost should not silently start optimising for
/// it. Returns `acq` unchanged when `cost` is unusable, rather than dividing by
/// zero and manufacturing an infinitely attractive candidate.
pub fn acq_per_cost(acq: f64, cost: f64) -> f64 {
    if !cost.is_finite() || cost <= 0.0 || !acq.is_finite() {
        return acq;
    }
    acq / cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn space_of(dims: Vec<(&str, Dist)>) -> SearchSpace {
        let mut m = BTreeMap::new();
        for (k, v) in dims {
            m.insert(k.to_string(), v);
        }
        SearchSpace { dims: m }
    }

    fn tr(pairs: Vec<(&str, Value)>, objective: f64) -> TrialResult {
        TrialResult {
            overlay: pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            objective,
        }
    }

    fn get(o: &Overlay, k: &str) -> Value {
        o.iter().find(|(n, _)| n == k).unwrap().1.clone()
    }

    // ── Grid ────────────────────────────────────────────────────────────
    #[test]
    fn grid_covers_the_product_without_repeating_early() {
        let sp = space_of(vec![
            ("a", Dist::IntUniform { low: 0, high: 1 }),
            (
                "b",
                Dist::Choice {
                    choices: vec![json!("x"), json!("y")],
                },
            ),
        ]);
        let mut g = GridSampler::new(4);
        let pts: Vec<Overlay> = (0..4).map(|_| g.ask(&sp, &[])).collect();
        let mut seen: Vec<String> = pts.iter().map(|p| format!("{p:?}")).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 4, "2x2 product must give 4 distinct points");
        // And it wraps rather than running off the end.
        let fifth = g.ask(&sp, &[]);
        assert_eq!(format!("{fifth:?}"), format!("{:?}", pts[0]));
    }

    #[test]
    fn grid_includes_both_endpoints_of_a_continuous_dim() {
        let sp = space_of(vec![(
            "lr",
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )]);
        let mut g = GridSampler::new(3);
        let vals: Vec<f64> = (0..3)
            .map(|_| get(&g.ask(&sp, &[]), "lr").as_f64().unwrap())
            .collect();
        assert!(vals.contains(&0.0) && vals.contains(&1.0), "{vals:?}");
    }

    #[test]
    fn grid_is_log_spaced_for_a_log_uniform_dim() {
        let sp = space_of(vec![(
            "lr",
            Dist::LogUniform {
                low: 1e-4,
                high: 1e-1,
            },
        )]);
        let mut g = GridSampler::new(4);
        let mut v: Vec<f64> = (0..4)
            .map(|_| get(&g.ask(&sp, &[]), "lr").as_f64().unwrap())
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Equal RATIOS, not equal differences — the whole point of log spacing.
        let r1 = v[1] / v[0];
        let r2 = v[2] / v[1];
        assert!((r1 - r2).abs() < 1e-6, "not log-spaced: {v:?}");
    }

    #[test]
    fn grid_does_not_emit_duplicates_for_a_narrow_int_dim() {
        // 8 levels requested across [0, 2] must yield 3 distinct values.
        let sp = space_of(vec![("k", Dist::IntUniform { low: 0, high: 2 })]);
        let mut g = GridSampler::new(8);
        let mut v: Vec<i64> = (0..3)
            .map(|_| get(&g.ask(&sp, &[]), "k").as_i64().unwrap())
            .collect();
        v.sort();
        assert_eq!(v, vec![0, 1, 2]);
    }

    #[test]
    fn grid_is_deterministic_and_needs_no_seed() {
        let sp = space_of(vec![(
            "a",
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )]);
        let (mut g1, mut g2) = (GridSampler::new(5), GridSampler::new(5));
        for _ in 0..7 {
            assert_eq!(g1.ask(&sp, &[]), g2.ask(&sp, &[]));
        }
    }

    // ── multivariate TPE ────────────────────────────────────────────────
    #[test]
    fn mv_tpe_draws_randomly_until_startup_is_reached() {
        let sp = space_of(vec![(
            "a",
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )]);
        let cfg = MvTpeConfig {
            n_startup: 5,
            ..Default::default()
        };
        let mut s = MvTpeSampler::new(cfg, 1);
        let hist: Vec<TrialResult> = (0..2)
            .map(|i| tr(vec![("a", json!(0.5))], i as f64))
            .collect();
        // Should not panic and should still produce a valid in-range value.
        let o = s.ask(&sp, &hist);
        let v = get(&o, "a").as_f64().unwrap();
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn mv_tpe_finds_the_diagonal_that_univariate_tpe_cannot() {
        // Good region is the DIAGONAL: (0.9,0.9) and (0.1,0.1) are good,
        // (0.9,0.1) and (0.1,0.9) are bad. Per-axis marginals of the good set
        // are bimodal and identical to the bad set's, so any sampler that
        // treats the axes independently gets no signal at all. A joint model
        // must land near a diagonal corner, never in an off-diagonal one.
        let sp = space_of(vec![
            (
                "x",
                Dist::Uniform {
                    low: 0.0,
                    high: 1.0,
                },
            ),
            (
                "y",
                Dist::Uniform {
                    low: 0.0,
                    high: 1.0,
                },
            ),
        ]);
        let mut hist = Vec::new();
        for _ in 0..6 {
            hist.push(tr(vec![("x", json!(0.9)), ("y", json!(0.9))], 1.0));
            hist.push(tr(vec![("x", json!(0.1)), ("y", json!(0.1))], 1.0));
            hist.push(tr(vec![("x", json!(0.9)), ("y", json!(0.1))], 0.0));
            hist.push(tr(vec![("x", json!(0.1)), ("y", json!(0.9))], 0.0));
        }
        let cfg = MvTpeConfig {
            n_startup: 1,
            n_candidates: 64,
            gamma: 0.5,
            ..Default::default()
        };
        let mut on_diagonal = 0;
        for seed in 0..12u64 {
            let mut s = MvTpeSampler::new(cfg.clone(), seed);
            let o = s.ask(&sp, &hist);
            let (x, y) = (
                get(&o, "x").as_f64().unwrap(),
                get(&o, "y").as_f64().unwrap(),
            );
            if (x - y).abs() < 0.5 {
                on_diagonal += 1;
            }
        }
        assert!(
            on_diagonal >= 9,
            "joint model should stay on the diagonal; got {on_diagonal}/12"
        );
    }

    #[test]
    fn mv_tpe_ignores_non_finite_objectives_when_splitting() {
        let _sp = space_of(vec![(
            "a",
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )]);
        let cfg = MvTpeConfig {
            n_startup: 1,
            ..Default::default()
        };
        let s = MvTpeSampler::new(cfg, 3);
        let hist = vec![
            tr(vec![("a", json!(0.2))], f64::NAN),
            tr(vec![("a", json!(0.8))], 1.0),
            tr(vec![("a", json!(0.4))], 0.0),
        ];
        let (good, bad) = s.split(&hist);
        let total = good.len() + bad.len();
        assert_eq!(total, 2, "the NaN trial must not enter either set");
        assert!(good.iter().all(|t| t.objective.is_finite()));
        // The best finite trial leads under maximize.
        assert_eq!(good[0].objective, 1.0);
    }

    #[test]
    fn mv_tpe_split_respects_minimize() {
        let cfg = MvTpeConfig {
            maximize: false,
            gamma: 0.5,
            ..Default::default()
        };
        let s = MvTpeSampler::new(cfg, 3);
        let hist = vec![
            tr(vec![("a", json!(0.2))], 5.0),
            tr(vec![("a", json!(0.8))], 1.0),
        ];
        let (good, _bad) = s.split(&hist);
        assert_eq!(good[0].objective, 1.0, "minimize ⇒ smallest is good");
    }

    #[test]
    fn mv_tpe_always_yields_a_value_for_every_declared_dim() {
        let sp = space_of(vec![
            (
                "a",
                Dist::Uniform {
                    low: 0.0,
                    high: 1.0,
                },
            ),
            (
                "c",
                Dist::Choice {
                    choices: vec![json!("p"), json!("q")],
                },
            ),
            ("k", Dist::IntUniform { low: 1, high: 4 }),
        ]);
        let cfg = MvTpeConfig {
            n_startup: 1,
            ..Default::default()
        };
        let mut s = MvTpeSampler::new(cfg, 11);
        let hist = vec![
            tr(
                vec![("a", json!(0.3)), ("c", json!("p")), ("k", json!(2))],
                1.0,
            ),
            tr(
                vec![("a", json!(0.7)), ("c", json!("q")), ("k", json!(3))],
                0.0,
            ),
        ];
        for _ in 0..10 {
            let o = s.ask(&sp, &hist);
            assert_eq!(o.len(), 3);
            let k = get(&o, "k").as_i64().unwrap();
            assert!((1..=4).contains(&k), "int dim out of range: {k}");
            let c = get(&o, "c");
            assert!(c == json!("p") || c == json!("q"), "bad categorical: {c}");
        }
    }

    // ── cost model / AcqPerCost ─────────────────────────────────────────
    #[test]
    fn cost_model_returns_an_observed_cost_exactly_at_that_point() {
        let sp = space_of(vec![(
            "a",
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )]);
        let m = CostModel::fit(vec![
            CostObservation {
                overlay: vec![("a".into(), json!(0.2))],
                cost: 10.0,
            },
            CostObservation {
                overlay: vec![("a".into(), json!(0.8))],
                cost: 90.0,
            },
        ]);
        assert_eq!(m.predict(&sp, &vec![("a".into(), json!(0.2))]), 10.0);
        assert_eq!(m.predict(&sp, &vec![("a".into(), json!(0.8))]), 90.0);
    }

    #[test]
    fn cost_model_interpolates_and_stays_within_the_observed_range() {
        let sp = space_of(vec![(
            "a",
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )]);
        let m = CostModel::fit(vec![
            CostObservation {
                overlay: vec![("a".into(), json!(0.0))],
                cost: 10.0,
            },
            CostObservation {
                overlay: vec![("a".into(), json!(1.0))],
                cost: 90.0,
            },
        ]);
        let mid = m.predict(&sp, &vec![("a".into(), json!(0.5))]);
        assert!(
            (10.0..=90.0).contains(&mid),
            "extrapolated outside the data: {mid}"
        );
        // Closer to the cheap point ⇒ cheaper prediction.
        let near_cheap = m.predict(&sp, &vec![("a".into(), json!(0.1))]);
        assert!(near_cheap < mid, "{near_cheap} !< {mid}");
    }

    #[test]
    fn cost_model_discards_unusable_costs_rather_than_fitting_them() {
        let sp = space_of(vec![(
            "a",
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        )]);
        let m = CostModel::fit(vec![
            CostObservation {
                overlay: vec![("a".into(), json!(0.2))],
                cost: f64::NAN,
            },
            CostObservation {
                overlay: vec![("a".into(), json!(0.3))],
                cost: 0.0,
            },
            CostObservation {
                overlay: vec![("a".into(), json!(0.4))],
                cost: -5.0,
            },
        ]);
        assert!(
            m.is_empty(),
            "a NaN/zero/negative cost is not a measurement"
        );
        // With nothing usable the prediction is a neutral 1.0, never 0 or NaN.
        let p = m.predict(&sp, &vec![("a".into(), json!(0.5))]);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn predicted_cost_is_always_finite_and_positive() {
        let sp = space_of(vec![
            (
                "a",
                Dist::LogUniform {
                    low: 1e-5,
                    high: 1e-1,
                },
            ),
            (
                "c",
                Dist::Choice {
                    choices: vec![json!("p"), json!("q")],
                },
            ),
        ]);
        let m = CostModel::fit(vec![CostObservation {
            overlay: vec![("a".into(), json!(1e-3)), ("c".into(), json!("p"))],
            cost: 42.0,
        }]);
        for v in [1e-5, 1e-4, 1e-2, 1e-1] {
            let p = m.predict(&sp, &vec![("a".into(), json!(v)), ("c".into(), json!("q"))]);
            assert!(p.is_finite() && p > 0.0, "bad prediction {p} at {v}");
        }
    }

    #[test]
    fn acq_per_cost_prefers_the_cheaper_of_two_similar_candidates() {
        // The whole point: a marginally better config that costs 10x more
        // must not win.
        let expensive = acq_per_cost(1.05, 1000.0);
        let cheap = acq_per_cost(1.00, 100.0);
        assert!(cheap > expensive, "{cheap} !> {expensive}");
    }

    #[test]
    fn acq_per_cost_never_manufactures_an_infinite_score() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let v = acq_per_cost(2.0, bad);
            assert_eq!(v, 2.0, "cost {bad} must leave acq unchanged");
            assert!(v.is_finite());
        }
        // A non-finite acquisition is passed through, not turned into a number.
        assert!(acq_per_cost(f64::NAN, 5.0).is_nan());
    }
}
