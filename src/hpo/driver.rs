// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0109's **search-driver**: the multi-objective sibling of
//! [`TpePolicy`](super::tpe::TpePolicy).
//!
//! It is the last piece of the loop this ADR describes. A trial reaches its
//! budget → its objective VECTOR is recorded → a GP is fitted per objective →
//! candidates are scored by expected hypervolume improvement against the
//! current Pareto front → the winner is emitted as `Control::Spawn`, becoming
//! an ordinary DAG node with an ordinary cache key. No new scheduler, no
//! daemon: exactly the seam ADR 0067 Slice B built and ADR 0078 fans out.
//!
//! **Two objectives, or it refuses to construct.** EHVI here is built on
//! [`hypervolume_2d`](super::pareto::hypervolume_2d), which is 2-D on purpose.
//! Silently falling back to random search for a 3-objective study would still
//! produce a plausible front while the model was never used — the same failure
//! `blut study run` refuses at the CLI. A narrow honest driver beats a general
//! wrong one.
//!
//! **The reference point is derived from the data, not configured.** A
//! hypervolume needs a bound, and a hand-set one is a hidden knob that silently
//! decides which trials count: too tight and real improvements score zero, too
//! loose and every candidate looks good. It is taken as the observed nadir
//! pushed out by a margin, recomputed as observations arrive.

use std::collections::{HashMap, HashSet, VecDeque};

use parking_lot::Mutex;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::Value;

use crate::framework::control::{Control, ControlPolicy, SpawnDelta, StepMetrics};

use super::gp::{GpConfig, GpModel, HypervolumeTarget, ehvi_mc};
use super::pareto::{Direction, ParetoPoint, pareto_front};
use super::scheduler::{dotted_f64, dotted_u64};
use super::space::{Overlay, SearchSpace, TrialResult};
use super::study::Objective;
use super::tpe::FreshFactory;

/// Runtime config for the surrogate driver.
#[derive(Clone, Debug)]
pub struct SurrogateConfig {
    /// Declared objectives, in `pareto.json` column order.
    pub objectives: Vec<Objective>,
    pub budget_key: String,
    /// A trial is complete — and therefore an observation — once it reaches
    /// this budget.
    pub max_budget: u64,
    /// Hard cap on trials this driver may spawn.
    pub max_spawns: usize,
    /// Propose randomly until this many complete observations exist. A GP
    /// fitted to one or two points is a prior with decoration.
    pub n_startup: usize,
    /// Candidate configs scored per proposal.
    pub n_candidates: usize,
    /// Posterior samples per EHVI estimate.
    pub ehvi_samples: usize,
    pub gp: GpConfig,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            objectives: Vec::new(),
            budget_key: "epoch".into(),
            max_budget: 1,
            max_spawns: 0,
            n_startup: 5,
            n_candidates: 48,
            ehvi_samples: 128,
            gp: GpConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SurrogateDecision {
    Continue,
    Spawn(Overlay),
}

#[derive(Default)]
struct SurrogateState {
    /// trial → best value per objective, in that objective's own direction.
    best: HashMap<u32, Vec<Option<f64>>>,
    /// Trials already turned into observations (record once).
    told: HashSet<u32>,
    /// Complete observations: (config, objective vector).
    observations: Vec<(Overlay, Vec<f64>)>,
    queue: VecDeque<Overlay>,
    spawns_done: usize,
}

/// The multi-objective search driver.
pub struct SurrogatePolicy {
    trial_of_topo: Vec<Option<u32>>,
    trial_overlays: Vec<Overlay>,
    space: SearchSpace,
    cfg: SurrogateConfig,
    factory: FreshFactory,
    state: Mutex<SurrogateState>,
    rng: Mutex<StdRng>,
}

impl SurrogatePolicy {
    /// Construct, or refuse.
    ///
    /// Refuses on anything that would make the search silently meaningless: a
    /// non-2 objective count (EHVI is 2-D), an empty space, or no spawn budget.
    pub fn new(
        trial_of_topo: Vec<Option<u32>>,
        trial_overlays: Vec<Overlay>,
        space: SearchSpace,
        cfg: SurrogateConfig,
        factory: FreshFactory,
        seed: u64,
    ) -> Result<Self, String> {
        if cfg.objectives.len() != 2 {
            return Err(format!(
                "the surrogate driver scores candidates with expected hypervolume \
                 improvement, which is implemented for exactly 2 objectives; this \
                 study declares {}. Refusing rather than falling back to random \
                 search, which would still produce a front while the model was \
                 never used.",
                cfg.objectives.len()
            ));
        }
        if space.dims.is_empty() {
            return Err("surrogate driver: empty search space".into());
        }
        if cfg.max_spawns == 0 {
            return Err("surrogate driver: max_spawns is 0, so it could never propose".into());
        }
        Ok(Self {
            trial_of_topo,
            trial_overlays,
            space,
            cfg,
            factory,
            state: Mutex::new(SurrogateState::default()),
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        })
    }

    /// Reference point bounding the dominated region: the observed nadir pushed
    /// out by 10% of each objective's observed range.
    ///
    /// Derived rather than configured — see the module note. A degenerate range
    /// (every trial reported the same value) falls back to an absolute margin,
    /// so the reference never lands exactly ON the front, which would make every
    /// hypervolume zero and every candidate look equally worthless.
    fn reference(&self, obs: &[(Overlay, Vec<f64>)]) -> [f64; 2] {
        let mut refp = [0.0_f64; 2];
        for (i, obj) in self.cfg.objectives.iter().enumerate() {
            let vals: Vec<f64> = obs
                .iter()
                .filter_map(|(_, v)| v.get(i).copied())
                .filter(|v| v.is_finite())
                .collect();
            if vals.is_empty() {
                refp[i] = 0.0;
                continue;
            }
            let (lo, hi) = vals
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let span = (hi - lo).abs();
            let margin = if span > 1e-12 { span * 0.1 } else { 1.0 };
            refp[i] = match obj.direction {
                // Worse than every observation, by a margin.
                Direction::Minimize => hi + margin,
                Direction::Maximize => lo - margin,
            };
        }
        refp
    }

    /// Propose the next config from the observations gathered so far.
    fn propose(&self, obs: &[(Overlay, Vec<f64>)]) -> Overlay {
        let mut rng = self.rng.lock();
        if obs.len() < self.cfg.n_startup {
            return self.space.sample(&mut *rng);
        }
        // One GP per objective. `TrialResult` is single-objective, so each model
        // sees its own column.
        let mut models = Vec::with_capacity(2);
        for i in 0..self.cfg.objectives.len() {
            let column: Vec<TrialResult> = obs
                .iter()
                .map(|(o, v)| TrialResult {
                    overlay: o.clone(),
                    objective: v[i],
                })
                .collect();
            match GpModel::fit(&self.space, &column, self.cfg.gp.clone()) {
                Some(m) => models.push(m),
                // A model that will not fit is a refusal, not a reason to guess:
                // fall back to a prior draw rather than score with half a model.
                None => return self.space.sample(&mut *rng),
            }
        }

        let dirs: Vec<Direction> = self.cfg.objectives.iter().map(|o| o.direction).collect();
        let points: Vec<ParetoPoint> = obs
            .iter()
            .enumerate()
            .map(|(i, (_, v))| ParetoPoint::new(i as u32, v.clone()))
            .collect();
        let front = pareto_front(&points, &dirs);
        let reference = self.reference(obs);
        let target = HypervolumeTarget {
            front: &front,
            dirs: &dirs,
            reference,
        };

        let mut best: Option<(f64, Overlay)> = None;
        for _ in 0..self.cfg.n_candidates.max(1) {
            let cand = self.space.sample(&mut *rng);
            let score = ehvi_mc(&models, &cand, &target, self.cfg.ehvi_samples, &mut rng);
            if score.is_finite() && best.as_ref().is_none_or(|(b, _)| score > *b) {
                best = Some((score, cand));
            }
        }
        // Every candidate scoring zero means none is expected to grow the front.
        // Returning the first is still a legitimate proposal — the DAG cache
        // makes a repeat free — but a fresh draw explores instead of stalling.
        best.map(|(s, c)| {
            if s > 0.0 {
                c
            } else {
                self.space.sample(&mut *rng)
            }
        })
        .unwrap_or_else(|| self.space.sample(&mut *rng))
    }

    pub(crate) fn decide(&self, trial: u32, update: &Value, budget: u64) -> SurrogateDecision {
        let mut st = self.state.lock();

        // Record FIRST, spawn second. If a trial completes on the same step the
        // queue drains, recording after the drain would drop its observation —
        // and for the LAST trial that observation is never recovered. The
        // single-objective driver documents the same ordering.
        let mut values: Vec<Option<f64>> = st
            .best
            .get(&trial)
            .cloned()
            .unwrap_or_else(|| vec![None; self.cfg.objectives.len()]);
        for (i, obj) in self.cfg.objectives.iter().enumerate() {
            let Some(v) = dotted_f64(update, &obj.metric) else {
                continue;
            };
            if !v.is_finite() {
                continue;
            }
            values[i] = Some(match values[i] {
                None => v,
                Some(b) => match obj.direction {
                    Direction::Maximize => b.max(v),
                    Direction::Minimize => b.min(v),
                },
            });
        }
        let complete = values.iter().all(|v| v.is_some());
        st.best.insert(trial, values.clone());

        if budget >= self.cfg.max_budget && complete && !st.told.contains(&trial) {
            st.told.insert(trial);
            if let Some(overlay) = self.trial_overlays.get(trial as usize).cloned() {
                let vector: Vec<f64> = values.iter().map(|v| v.unwrap()).collect();
                st.observations.push((overlay, vector));
            }
            if st.spawns_done + st.queue.len() < self.cfg.max_spawns {
                let obs = st.observations.clone();
                // Drop the state lock across the model fit: it is the expensive
                // part, and holding it would serialise every other trial's step.
                drop(st);
                let suggestion = self.propose(&obs);
                st = self.state.lock();
                st.queue.push_back(suggestion);
            }
        }

        if st.spawns_done < self.cfg.max_spawns
            && let Some(overlay) = st.queue.pop_front()
        {
            st.spawns_done += 1;
            return SurrogateDecision::Spawn(overlay);
        }
        SurrogateDecision::Continue
    }

    /// Observations gathered so far, for tests and reporting.
    pub fn observations(&self) -> Vec<(Overlay, Vec<f64>)> {
        self.state.lock().observations.clone()
    }
}

impl ControlPolicy for SurrogatePolicy {
    fn on_step(&self, m: &StepMetrics) -> Control {
        let Some(trial) = self
            .trial_of_topo
            .get(m.node_idx as usize)
            .copied()
            .flatten()
        else {
            return Control::Continue;
        };
        let budget = dotted_u64(m.update, &self.cfg.budget_key).unwrap_or(0);
        match self.decide(trial, m.update, budget) {
            SurrogateDecision::Continue => Control::Continue,
            SurrogateDecision::Spawn(overlay) => match (self.factory)(&overlay) {
                Ok(subplan) => Control::Spawn(Box::new(SpawnDelta::new(
                    subplan,
                    Some("surrogate-suggest".into()),
                ))),
                Err(e) => {
                    tracing::warn!("surrogate driver: sub-plan build failed: {e}");
                    Control::Continue
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpo::space::Dist;
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn objectives() -> Vec<Objective> {
        vec![
            Objective {
                name: "prd".into(),
                metric: "eval.prd".into(),
                direction: Direction::Minimize,
            },
            Objective {
                name: "ratio".into(),
                metric: "eval.ratio".into(),
                direction: Direction::Maximize,
            },
        ]
    }

    fn space() -> SearchSpace {
        let mut m = BTreeMap::new();
        m.insert(
            "x".to_string(),
            Dist::Uniform {
                low: 0.0,
                high: 1.0,
            },
        );
        SearchSpace { dims: m }
    }

    /// A factory that never actually compiles a plan — the tests exercise
    /// `decide`, which is the whole decision surface.
    fn factory() -> FreshFactory {
        Arc::new(|_o: &Overlay| Err("not compiled in tests".to_string()))
    }

    fn policy(cfg: SurrogateConfig, n_trials: usize) -> SurrogatePolicy {
        let overlays: Vec<Overlay> = (0..n_trials)
            .map(|i| vec![("x".to_string(), json!(i as f64 / n_trials as f64))])
            .collect();
        SurrogatePolicy::new(
            (0..n_trials).map(|i| Some(i as u32)).collect(),
            overlays,
            space(),
            cfg,
            factory(),
            7,
        )
        .expect("valid config")
    }

    fn cfg(max_spawns: usize) -> SurrogateConfig {
        SurrogateConfig {
            objectives: objectives(),
            budget_key: "epoch".into(),
            max_budget: 1,
            max_spawns,
            n_startup: 2,
            n_candidates: 8,
            ehvi_samples: 16,
            ..Default::default()
        }
    }

    fn upd(prd: f64, ratio: f64, epoch: u64) -> Value {
        json!({"eval":{"prd":prd,"ratio":ratio},"epoch":epoch})
    }

    // ── construction refuses what it cannot do ──────────────────────────
    #[test]
    fn a_non_two_objective_study_is_refused_not_downgraded() {
        for n in [1usize, 3] {
            let mut c = cfg(4);
            c.objectives = objectives().into_iter().cycle().take(n).collect();
            let err = SurrogatePolicy::new(vec![Some(0)], vec![vec![]], space(), c, factory(), 1)
                .err()
                .expect("must refuse");
            assert!(err.contains("exactly 2 objectives"), "{n}: {err}");
            assert!(
                err.contains("random search"),
                "must say WHY it refuses rather than falling back: {err}"
            );
        }
    }

    #[test]
    fn an_empty_space_or_zero_spawn_budget_is_refused() {
        let mut c = cfg(4);
        assert!(
            SurrogatePolicy::new(
                vec![],
                vec![],
                SearchSpace::default(),
                c.clone(),
                factory(),
                1
            )
            .err()
            .expect("must refuse")
            .contains("empty search space")
        );
        c.max_spawns = 0;
        assert!(
            SurrogatePolicy::new(vec![Some(0)], vec![vec![]], space(), c, factory(), 1)
                .err()
                .expect("must refuse")
                .contains("could never propose")
        );
    }

    // ── observation discipline ──────────────────────────────────────────
    #[test]
    fn a_trial_becomes_an_observation_only_when_every_objective_is_present() {
        let p = policy(cfg(4), 3);
        // Only one of the two objectives — must not be recorded.
        let partial = json!({"eval":{"prd":0.4},"epoch":5});
        assert_eq!(p.decide(0, &partial, 5), SurrogateDecision::Continue);
        assert!(
            p.observations().is_empty(),
            "a half-measured trial is not an observation"
        );
        // Now the second arrives.
        p.decide(0, &upd(0.4, 9.0, 5), 5);
        assert_eq!(p.observations().len(), 1);
    }

    #[test]
    fn each_objective_keeps_its_best_in_its_own_direction() {
        let p = policy(cfg(4), 3);
        // prd minimised, ratio maximised, from the same stream.
        p.decide(0, &upd(0.9, 5.0, 0), 0);
        p.decide(0, &upd(0.3, 3.0, 0), 0);
        p.decide(0, &upd(0.7, 11.0, 1), 1); // completes at budget
        let obs = p.observations();
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].1, vec![0.3, 11.0], "min(prd) and max(ratio)");
    }

    #[test]
    fn a_trial_is_told_once_however_many_steps_it_reports() {
        let p = policy(cfg(4), 3);
        for _ in 0..5 {
            p.decide(0, &upd(0.5, 7.0, 9), 9);
        }
        assert_eq!(p.observations().len(), 1, "recorded once, not per step");
    }

    // ── spawn budget ────────────────────────────────────────────────────
    #[test]
    fn spawning_stops_at_max_spawns() {
        let p = policy(cfg(2), 8);
        let mut spawned = 0;
        for t in 0..8u32 {
            if let SurrogateDecision::Spawn(_) =
                p.decide(t, &upd(0.5 - t as f64 * 0.01, 5.0 + t as f64, 3), 3)
            {
                spawned += 1;
            }
        }
        assert_eq!(spawned, 2, "the cap is hard");
    }

    #[test]
    fn the_last_trial_s_observation_is_never_lost_to_a_spawn() {
        // Record-before-spawn: a trial completing on the same step the queue
        // drains must still be recorded. With max_spawns=1 the drain fires on
        // the very first completion.
        let p = policy(cfg(1), 4);
        let d = p.decide(0, &upd(0.5, 7.0, 3), 3);
        assert!(matches!(d, SurrogateDecision::Spawn(_)), "drain fires");
        assert_eq!(
            p.observations().len(),
            1,
            "and the completing trial was still recorded"
        );
    }

    // ── the reference point ─────────────────────────────────────────────
    #[test]
    fn the_reference_is_worse_than_every_observation_on_both_axes() {
        let p = policy(cfg(4), 4);
        let obs = vec![
            (vec![], vec![0.2, 10.0]),
            (vec![], vec![0.6, 4.0]),
            (vec![], vec![0.4, 7.0]),
        ];
        let r = p.reference(&obs);
        // prd minimised ⇒ reference above the worst; ratio maximised ⇒ below.
        assert!(r[0] > 0.6, "prd reference {} must be worse than 0.6", r[0]);
        assert!(
            r[1] < 4.0,
            "ratio reference {} must be worse than 4.0",
            r[1]
        );
    }

    #[test]
    fn a_degenerate_range_still_yields_a_reference_off_the_front() {
        // Every trial reported the same values: a proportional margin would be
        // zero and put the reference ON the front, making every hypervolume 0.
        let p = policy(cfg(4), 4);
        let obs = vec![(vec![], vec![1.0, 1.0]), (vec![], vec![1.0, 1.0])];
        let r = p.reference(&obs);
        assert!(
            r[0] > 1.0 && r[1] < 1.0,
            "reference {r:?} sits on the front"
        );
    }

    #[test]
    fn proposals_stay_inside_the_declared_space() {
        let p = policy(cfg(6), 8);
        for t in 0..6u32 {
            if let SurrogateDecision::Spawn(o) =
                p.decide(t, &upd(0.5 - t as f64 * 0.05, 5.0 + t as f64, 2), 2)
            {
                let x = o
                    .iter()
                    .find(|(k, _)| k == "x")
                    .and_then(|(_, v)| v.as_f64())
                    .expect("the proposal declares x");
                assert!((0.0..=1.0).contains(&x), "proposal out of support: {x}");
            }
        }
    }
}
