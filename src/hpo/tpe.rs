// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! TPE — Tree-structured Parzen Estimator (v0.20 Phase 8), a model-based
//! ask-tell sampler + the policy that drives it at runtime.
//!
//! TPE conditions the next suggestion on completed trials. It splits the
//! observations into a GOOD set (the top `gamma` quantile by objective) and a
//! BAD set, fits a per-dimension Parzen window (a KDE — a uniform prior plus a
//! Gaussian per observation) to each, and suggests the candidate that maximizes
//! `l(x)/g(x)` (good density over bad density) — i.e. config values that the
//! good trials favored and the bad trials avoided. Continuous dims are estimated
//! in transformed space (log for `log_uniform`); categoricals use smoothed
//! good/bad counts.
//!
//! Because TPE is ask-TELL, it can't live in the static up-front fan-out (which
//! asks N times against an EMPTY history). [`TpePolicy`] runs it at runtime: a
//! small random population starts the search, and as each trial COMPLETES the
//! policy tells TPE the result and `Spawn`s a fresh TPE-suggested trial.
//!
//! v0.20 boundary: `trial_of_topo` covers the INITIAL fan-out only, so a
//! `Spawn`'d trial's steps (topo positions beyond it) are unattributed and don't
//! feed back into the model — TPE/PBT learn from the initial population and the
//! suggestions explore from it, but spawned trials are not themselves re-told.
//! A runtime trial-registration channel (attributing new node ranges to trials)
//! is the follow-up that closes the loop.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::Value;

use crate::framework::control::{Control, ControlPolicy, SpawnDelta, StepMetrics};
use crate::framework::plan::CompiledPlan;

use super::sampler::Sampler;
use super::scheduler::{dotted_f64, dotted_u64};
use super::space::{Dist, Overlay, SearchSpace, TrialResult};

/// TPE tuning.
#[derive(Clone, Debug)]
pub struct TpeConfig {
    /// Top quantile that counts as "good" (e.g. 0.25).
    pub gamma: f64,
    /// Candidates drawn from the good model + prior, scored by `l/g`, per dim.
    pub n_candidates: usize,
    /// Suggest randomly until this many trials have completed (cold start).
    pub n_startup: usize,
    /// Parzen bandwidth as a fraction of each dim's (transformed) support.
    pub bw_factor: f64,
    /// `true` = maximize the objective.
    pub maximize: bool,
}

impl Default for TpeConfig {
    fn default() -> Self {
        Self {
            gamma: 0.25,
            n_candidates: 24,
            n_startup: 5,
            bw_factor: 0.15,
            maximize: true,
        }
    }
}

/// The model-based sampler.
pub struct TpeSampler {
    cfg: TpeConfig,
    rng: StdRng,
}

impl TpeSampler {
    pub fn new(cfg: TpeConfig, seed: u64) -> Self {
        Self {
            cfg,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

/// Standard normal pdf at `z = (x-μ)/σ`, scaled by `1/σ`.
fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    let s = sigma.max(1e-12);
    let z = (x - mu) / s;
    (-0.5 * z * z).exp() / (s * (2.0 * std::f64::consts::PI).sqrt())
}

/// A continuous dim's support in t-space (log for `log_uniform`), straight from
/// the `Dist` — independent of any data point. `None` for categoricals.
fn support_t(dist: &Dist) -> Option<(f64, f64)> {
    match dist {
        Dist::Uniform { low, high } | Dist::QUniform { low, high, .. } => Some((*low, *high)),
        Dist::LogUniform { low, high } => Some((low.ln(), high.ln())),
        Dist::IntUniform { low, high } => Some((*low as f64, *high as f64)),
        Dist::Choice { .. } => None,
    }
}

/// Forward transform of a value into t-space (log for `log_uniform`, else
/// identity).
fn to_t(dist: &Dist, v: f64) -> f64 {
    match dist {
        Dist::LogUniform { .. } => v.max(f64::MIN_POSITIVE).ln(),
        _ => v,
    }
}

/// Inverse transform: t-space value → a JSON value honoring the dim's type.
fn from_t(dist: &Dist, t: f64) -> Value {
    let num = |x: f64| {
        serde_json::Number::from_f64(x)
            .map(Value::Number)
            .unwrap_or(Value::Null)
    };
    match dist {
        Dist::Uniform { .. } => num(t),
        Dist::LogUniform { .. } => num(t.exp()),
        Dist::IntUniform { low, high } => Value::from((t.round() as i64).clamp(*low, *high)),
        Dist::QUniform { q, .. } => {
            let qq = if *q == 0.0 { 1.0 } else { *q };
            num((t / qq).round() * qq)
        }
        Dist::Choice { .. } => Value::Null,
    }
}

/// Parzen density at `c`: a uniform prior plus a Gaussian per observation,
/// normalized so the prior carries the weight of one pseudo-observation.
fn parzen(c: f64, obs: &[f64], bw: f64, lo: f64, hi: f64) -> f64 {
    let span = (hi - lo).max(1e-12);
    let prior = 1.0 / span;
    let mut acc = prior; // one pseudo-count of prior mass
    for &o in obs {
        acc += gaussian(c, o, bw);
    }
    acc / (obs.len() as f64 + 1.0)
}

impl TpeSampler {
    /// Pull this dim's numeric value out of every trial in a set.
    fn dim_values(trials: &[&TrialResult], name: &str) -> Vec<f64> {
        trials
            .iter()
            .filter_map(|t| {
                t.overlay
                    .iter()
                    .find(|(k, _)| k == name)
                    .and_then(|(_, v)| v.as_f64())
            })
            .collect()
    }

    /// This dim's categorical value out of every trial.
    fn dim_choices<'a>(trials: &'a [&TrialResult], name: &str) -> Vec<&'a Value> {
        trials
            .iter()
            .filter_map(|t| t.overlay.iter().find(|(k, _)| k == name).map(|(_, v)| v))
            .collect()
    }

    fn suggest_continuous(&mut self, dist: &Dist, good: &[f64], bad: &[f64]) -> Value {
        // Support comes from the Dist (never a data point); values transform to
        // the same t-space.
        let Some((lo, hi)) = support_t(dist) else {
            return dist.sample(&mut self.rng);
        };
        let good_t: Vec<f64> = good.iter().map(|&v| to_t(dist, v)).collect();
        let bad_t: Vec<f64> = bad.iter().map(|&v| to_t(dist, v)).collect();
        let bw = ((hi - lo) * self.cfg.bw_factor).max(1e-9);

        // Draw candidates from the good model (a random good point + Gaussian
        // jitter) plus an occasional prior draw, then keep the best l/g.
        let mut best: Option<(f64, f64)> = None; // (score, t)
        for _ in 0..self.cfg.n_candidates.max(1) {
            let cand = if good_t.is_empty() || self.rng.gen_bool(0.25) {
                self.rng.gen_range(lo..hi.max(lo + f64::EPSILON))
            } else {
                let idx = self.rng.gen_range(0..good_t.len());
                let m = good_t[idx];
                let jitter = gaussian_sample(&mut self.rng, bw);
                (m + jitter).clamp(lo, hi)
            };
            let l = parzen(cand, &good_t, bw, lo, hi);
            let g = parzen(cand, &bad_t, bw, lo, hi);
            let score = l.ln() - g.ln(); // maximize l/g
            if best.is_none_or(|(bs, _)| score > bs) {
                best = Some((score, cand));
            }
        }
        from_t(dist, best.map(|(_, t)| t).unwrap_or((lo + hi) / 2.0))
    }

    fn suggest_choice(&mut self, choices: &[Value], good: &[&Value], bad: &[&Value]) -> Value {
        if choices.is_empty() {
            return Value::Null;
        }
        let count = |set: &[&Value], c: &Value| set.iter().filter(|v| **v == c).count() as f64;
        let k = choices.len() as f64;
        let mut best: Option<(f64, usize)> = None;
        for (i, c) in choices.iter().enumerate() {
            // Laplace-smoothed good/bad likelihoods.
            let l = (count(good, c) + 1.0) / (good.len() as f64 + k);
            let g = (count(bad, c) + 1.0) / (bad.len() as f64 + k);
            let score = l / g;
            if best.is_none_or(|(bs, _)| score > bs) {
                best = Some((score, i));
            }
        }
        choices[best.map(|(_, i)| i).unwrap_or(0)].clone()
    }
}

/// Sample N(0, σ) via Box–Muller (avoids a `rand_distr` dep).
fn gaussian_sample(rng: &mut impl Rng, sigma: f64) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * sigma
}

impl super::sampler::Sampler for TpeSampler {
    fn ask(&mut self, space: &SearchSpace, completed: &[TrialResult]) -> Overlay {
        // Cold start: not enough data to model — sample at random.
        if completed.len() < self.cfg.n_startup.max(2) {
            return space.sample(&mut self.rng);
        }
        // Split good / bad by objective (higher score = better; negate to
        // minimize). n_good = ceil(gamma·n), bounded to leave ≥1 in each set.
        let score = |o: f64| if self.cfg.maximize { o } else { -o };
        let mut ranked: Vec<&TrialResult> = completed.iter().collect();
        ranked.sort_by(|a, b| {
            score(b.objective)
                .partial_cmp(&score(a.objective))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = ranked.len();
        let n_good = ((self.cfg.gamma * n as f64).ceil() as usize).clamp(1, n - 1);
        let (good, bad) = ranked.split_at(n_good);

        let mut overlay: Overlay = Vec::with_capacity(space.dims.len());
        for (name, dist) in &space.dims {
            let v = match dist {
                Dist::Choice { choices } => {
                    let g = Self::dim_choices(good, name);
                    let b = Self::dim_choices(bad, name);
                    self.suggest_choice(choices, &g, &b)
                }
                _ => {
                    let g = Self::dim_values(good, name);
                    let b = Self::dim_values(bad, name);
                    if g.is_empty() {
                        dist.sample(&mut self.rng)
                    } else {
                        self.suggest_continuous(dist, &g, &b)
                    }
                }
            };
            overlay.push((name.clone(), v));
        }
        overlay
    }
}

// ════════════════════════════════════════════════════════════════════
// TpePolicy — runtime ask-tell driver
// ════════════════════════════════════════════════════════════════════

/// Compiles a fresh TPE-suggested trial (overlay over base recipe). No resume —
/// TPE explores fresh configs (cf. PBT's warm-started clones).
pub type FreshFactory = Arc<dyn Fn(&Overlay) -> Result<CompiledPlan, String> + Send + Sync>;

/// TPE runtime config.
#[derive(Clone, Debug)]
pub struct TpePolicyConfig {
    pub metric_key: String,
    pub budget_key: String,
    /// A trial is "complete" (→ tell TPE + suggest the next) once it reaches
    /// this budget.
    pub max_budget: u64,
    /// Hard cap on TPE-suggested trials spawned.
    pub max_spawns: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TpeDecision {
    Continue,
    Spawn(Overlay),
}

struct TpeState {
    /// trial_id → that trial's best objective seen so far.
    best: HashMap<u32, f64>,
    /// trials already told to TPE (record once on completion).
    told: HashSet<u32>,
    /// completed observations fed to the sampler.
    observations: Vec<TrialResult>,
    queue: VecDeque<Overlay>,
    spawns_done: usize,
}

/// The TPE control policy: tracks trial completions, tells the sampler, and
/// `Spawn`s a fresh suggested trial.
pub struct TpePolicy {
    trial_of_topo: Vec<Option<u32>>,
    /// Per-trial overlay (the config we record when it completes).
    trial_overlays: Vec<Overlay>,
    space: SearchSpace,
    cfg: TpePolicyConfig,
    sampler: Mutex<TpeSampler>,
    factory: FreshFactory,
    state: Mutex<TpeState>,
}

impl TpePolicy {
    pub fn new(
        trial_of_topo: Vec<Option<u32>>,
        trial_overlays: Vec<Overlay>,
        space: SearchSpace,
        cfg: TpePolicyConfig,
        sampler: TpeSampler,
        factory: FreshFactory,
    ) -> Self {
        Self {
            trial_of_topo,
            trial_overlays,
            space,
            cfg,
            sampler: Mutex::new(sampler),
            factory,
            state: Mutex::new(TpeState {
                best: HashMap::new(),
                told: HashSet::new(),
                observations: Vec::new(),
                queue: VecDeque::new(),
                spawns_done: 0,
            }),
        }
    }

    /// Pure decision core (no `CompiledPlan` — unit-testable): drains a queued
    /// suggestion first, records a trial's running best, and on completion tells
    /// TPE + enqueues the next suggestion.
    pub(crate) fn decide(&self, trial: u32, obj: f64, budget: u64) -> TpeDecision {
        let mut st = self.state.lock();
        // 1. Record + tell on completion FIRST — a trial's observation must never
        //    be lost to an early Spawn return (DeepSeek review): if a trial
        //    completes on the SAME step the drain fires, recording after the
        //    drain would drop it (especially for the last trial).
        if obj.is_finite() {
            let best = {
                let entry = st.best.entry(trial).or_insert(obj);
                *entry = entry.max(obj);
                *entry
            };
            if budget >= self.cfg.max_budget && !st.told.contains(&trial) {
                st.told.insert(trial);
                if let Some(overlay) = self.trial_overlays.get(trial as usize).cloned() {
                    st.observations.push(TrialResult {
                        overlay,
                        objective: best,
                    });
                }
                // Suggest the next config if there's still spawn budget.
                if st.spawns_done + st.queue.len() < self.cfg.max_spawns {
                    let obs = st.observations.clone();
                    drop(st);
                    let suggestion = self.sampler.lock().ask(&self.space, &obs);
                    st = self.state.lock();
                    st.queue.push_back(suggestion);
                }
            }
        }
        // 2. Emit ONE queued suggestion (one per step), under the cap.
        if st.spawns_done < self.cfg.max_spawns {
            if let Some(overlay) = st.queue.pop_front() {
                st.spawns_done += 1;
                return TpeDecision::Spawn(overlay);
            }
        }
        TpeDecision::Continue
    }
}

impl ControlPolicy for TpePolicy {
    fn on_step(&self, m: &StepMetrics) -> Control {
        let Some(trial) = self
            .trial_of_topo
            .get(m.node_idx as usize)
            .copied()
            .flatten()
        else {
            return Control::Continue;
        };
        let Some(obj) = dotted_f64(m.update, &self.cfg.metric_key) else {
            return Control::Continue;
        };
        let budget = dotted_u64(m.update, &self.cfg.budget_key).unwrap_or(0);
        match self.decide(trial, obj, budget) {
            TpeDecision::Continue => Control::Continue,
            TpeDecision::Spawn(overlay) => match (self.factory)(&overlay) {
                Ok(subplan) => Control::Spawn(Box::new(SpawnDelta::new(
                    subplan,
                    Some("tpe-suggest".into()),
                ))),
                Err(e) => {
                    tracing::warn!("tpe suggest factory failed: {e}");
                    Control::Continue
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpo::sampler::Sampler;
    use serde_json::json;

    fn cont_space() -> SearchSpace {
        let mut s = SearchSpace::default();
        s.dims.insert(
            "lr".into(),
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        s
    }

    fn tr(lr: f64, obj: f64) -> TrialResult {
        TrialResult {
            overlay: vec![("lr".into(), json!(lr))],
            objective: obj,
        }
    }

    #[test]
    fn cold_start_samples_in_support() {
        let mut t = TpeSampler::new(TpeConfig::default(), 1);
        let o = t.ask(&cont_space(), &[]); // empty → random
        let lr = o
            .iter()
            .find(|(k, _)| k == "lr")
            .unwrap()
            .1
            .as_f64()
            .unwrap();
        assert!((0.0..=1.0).contains(&lr));
    }

    #[test]
    fn biases_toward_the_good_region() {
        // Good trials (high objective) cluster lr≈0.9; bad cluster lr≈0.1.
        let mut hist = vec![];
        for i in 0..8 {
            hist.push(tr(0.88 + 0.01 * i as f64, 0.9)); // good
            hist.push(tr(0.08 + 0.01 * i as f64, 0.1)); // bad
        }
        let mut t = TpeSampler::new(
            TpeConfig {
                gamma: 0.5,
                n_candidates: 64,
                n_startup: 4,
                bw_factor: 0.15,
                maximize: true,
            },
            7,
        );
        // Average several suggestions — they should lean to the good region.
        let mut sum = 0.0;
        let n = 12;
        for _ in 0..n {
            let o = t.ask(&cont_space(), &hist);
            sum += o
                .iter()
                .find(|(k, _)| k == "lr")
                .unwrap()
                .1
                .as_f64()
                .unwrap();
        }
        let avg = sum / n as f64;
        assert!(
            avg > 0.5,
            "TPE should favor the good lr≈0.9 region, got avg {avg}"
        );
    }

    #[test]
    fn minimize_flips_the_good_set() {
        // Same data, but now LOW objective is good → favor lr≈0.1.
        let mut hist = vec![];
        for i in 0..8 {
            hist.push(tr(0.88 + 0.01 * i as f64, 0.9));
            hist.push(tr(0.08 + 0.01 * i as f64, 0.1));
        }
        let mut t = TpeSampler::new(
            TpeConfig {
                gamma: 0.5,
                n_candidates: 64,
                n_startup: 4,
                bw_factor: 0.15,
                maximize: false,
            },
            7,
        );
        let mut sum = 0.0;
        let n = 12;
        for _ in 0..n {
            let o = t.ask(&cont_space(), &hist);
            sum += o
                .iter()
                .find(|(k, _)| k == "lr")
                .unwrap()
                .1
                .as_f64()
                .unwrap();
        }
        assert!(
            sum / (n as f64) < 0.5,
            "minimize should favor the low-lr region"
        );
    }

    #[test]
    fn categorical_favors_the_good_choice() {
        let mut s = SearchSpace::default();
        s.dims.insert(
            "opt".into(),
            Dist::Choice {
                choices: vec![json!("soap"), json!("adamw")],
            },
        );
        // Good trials picked "soap"; bad picked "adamw".
        let mut hist = vec![];
        for _ in 0..6 {
            hist.push(TrialResult {
                overlay: vec![("opt".into(), json!("soap"))],
                objective: 0.9,
            });
            hist.push(TrialResult {
                overlay: vec![("opt".into(), json!("adamw"))],
                objective: 0.1,
            });
        }
        let mut t = TpeSampler::new(
            TpeConfig {
                gamma: 0.5,
                n_candidates: 16,
                n_startup: 4,
                bw_factor: 0.15,
                maximize: true,
            },
            3,
        );
        let o = t.ask(&s, &hist);
        assert_eq!(o.iter().find(|(k, _)| k == "opt").unwrap().1, json!("soap"));
    }

    fn policy(max_spawns: usize) -> TpePolicy {
        let cfg = TpePolicyConfig {
            metric_key: "val_r".into(),
            budget_key: "epoch".into(),
            max_budget: 4,
            max_spawns,
        };
        let sampler = TpeSampler::new(
            TpeConfig {
                n_startup: 2,
                ..TpeConfig::default()
            },
            9,
        );
        let factory: FreshFactory = Arc::new(|_| Err("unused".into()));
        TpePolicy::new(
            vec![Some(0), Some(1)],
            vec![
                vec![("lr".into(), json!(0.5))],
                vec![("lr".into(), json!(0.6))],
            ],
            cont_space(),
            cfg,
            sampler,
            factory,
        )
    }

    #[test]
    fn completion_tells_and_enqueues_suggestion() {
        let p = policy(8);
        // A completion now RECORDS first then drains in the same step, so it
        // emits a suggestion immediately (no deferral that could lose the last
        // trial's observation). t0 completes → its observation is recorded and a
        // suggestion (random, only 1 obs < n_startup) is emitted.
        match p.decide(0, 0.3, 4) {
            TpeDecision::Spawn(o) => {
                let lr = o
                    .iter()
                    .find(|(k, _)| k == "lr")
                    .unwrap()
                    .1
                    .as_f64()
                    .unwrap();
                assert!((0.0..=1.0).contains(&lr), "suggested lr in support");
            }
            d => panic!("a completion must record + emit a suggestion, got {d:?}"),
        }
        // t1 completes → now 2 observations ≥ n_startup → a TPE-modeled
        // suggestion is emitted.
        assert!(matches!(p.decide(1, 0.7, 4), TpeDecision::Spawn(_)));
    }

    #[test]
    fn pre_completion_never_suggests() {
        let p = policy(8);
        // Budgets below max_budget → just track, never tell/suggest.
        assert_eq!(p.decide(0, 0.3, 1), TpeDecision::Continue);
        assert_eq!(p.decide(1, 0.7, 2), TpeDecision::Continue);
        assert_eq!(p.decide(0, 0.4, 3), TpeDecision::Continue);
    }

    #[test]
    fn spawn_cap_halts_suggestions() {
        let p = policy(0); // no spawns allowed
        assert_eq!(p.decide(0, 0.3, 4), TpeDecision::Continue);
        assert_eq!(p.decide(1, 0.7, 4), TpeDecision::Continue);
        assert_eq!(
            p.decide(0, 0.3, 5),
            TpeDecision::Continue,
            "cap 0 → never suggests"
        );
    }
}
