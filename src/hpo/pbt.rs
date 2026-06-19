//! Population-Based Training (v0.20 Phase 7) — the one resume-on-promote
//! consumer.
//!
//! PBT runs the trials as a parallel population (the fan-out plan). At each RUNG
//! a trial below the cull quantile is EXPLOITED + EXPLORED: it is `KillBranch`'d
//! (exploit — stop the loser) and a perturbed clone of the current best survivor
//! is `Spawn`'d, warm-started from that winner's checkpoint via the cookbook's
//! factory (which bakes `--resume-from <winner_dir>` into the clone's args — the
//! executor stays oblivious to resume). Because `Control` carries ONE action per
//! step, the clone spec is QUEUED on the kill step and emitted as a `Spawn` on a
//! subsequent step (any still-running trial keeps the stream alive until the
//! population converges).
//!
//! The decision core ([`PbtPolicy::decide`]) is pure + unit-tested; `on_step`
//! only wraps it with the `CompiledPlan` factory call.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::Mutex;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::framework::control::{Control, ControlPolicy, SpawnDelta, StepMetrics};
use crate::framework::plan::CompiledPlan;

use super::median::percentile;
use super::scheduler::{dotted_f64, dotted_u64};
use super::space::{Overlay, SearchSpace};

/// Where a perturbed clone warm-starts from: the winning trial + its checkpoint
/// dir. The cookbook factory turns this into a `--resume-from` arg.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PbtResume {
    pub winner_trial: u32,
    pub resume_dir: PathBuf,
}

/// A population member's identity, supplied at construction (from the manifest +
/// the per-trial job/stage dir).
#[derive(Clone, Debug)]
pub struct PbtTrial {
    pub overlay: Overlay,
    /// The trial's checkpoint dir — a clone of THIS trial resumes from here.
    pub resume_dir: PathBuf,
}

/// Compiles a perturbed clone's sub-plan: applies `overlay` over the base recipe
/// AND wires `--resume-from resume.resume_dir`. Cookbook-provided (it owns the
/// `Registry` + recipe + the trainer's resume contract).
pub type TrialFactory =
    Arc<dyn Fn(&Overlay, &PbtResume) -> Result<CompiledPlan, String> + Send + Sync>;

/// PBT tuning.
#[derive(Clone, Debug)]
pub struct PbtConfig {
    pub metric_key: String,
    pub budget_key: String,
    pub maximize: bool,
    /// Budget milestones at which to exploit/explore.
    pub rungs: Vec<u64>,
    /// Cull a trial whose score is below this percentile of its rung peers
    /// (e.g. `25.0` = bottom quarter).
    pub bottom_quantile: f64,
    /// Need at least this many peers at a rung before acting.
    pub min_peers: usize,
    /// Hard cap on total clones spawned (a backstop; the executor caps too).
    pub max_spawns: usize,
}

/// The pure decision (no `CompiledPlan` — unit-testable).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PbtDecision {
    Continue,
    Kill,
    Spawn { overlay: Overlay, resume: PbtResume },
}

struct PbtState {
    /// trial_id → (budget → objective).
    history: HashMap<u32, BTreeMap<u64, f64>>,
    killed: HashSet<u32>,
    /// (trial, rung) pairs already decided — act once per trial per rung.
    acted: HashSet<(u32, u64)>,
    /// Perturbed-clone specs awaiting a step to ride out on.
    queue: VecDeque<(Overlay, PbtResume)>,
    spawns_done: usize,
}

/// The PBT control policy.
pub struct PbtPolicy {
    trial_of_topo: Vec<Option<u32>>,
    trials: Vec<PbtTrial>,
    space: SearchSpace,
    cfg: PbtConfig,
    factory: TrialFactory,
    rng: Mutex<StdRng>,
    state: Mutex<PbtState>,
}

impl PbtPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trial_of_topo: Vec<Option<u32>>,
        trials: Vec<PbtTrial>,
        space: SearchSpace,
        cfg: PbtConfig,
        factory: TrialFactory,
        seed: u64,
    ) -> Self {
        Self {
            trial_of_topo,
            trials,
            space,
            cfg,
            factory,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            state: Mutex::new(PbtState {
                history: HashMap::new(),
                killed: HashSet::new(),
                acted: HashSet::new(),
                queue: VecDeque::new(),
                spawns_done: 0,
            }),
        }
    }

    fn score(&self, obj: f64) -> f64 {
        if self.cfg.maximize { obj } else { -obj }
    }

    /// Perturb every dim of a winner's overlay that the search space knows; an
    /// unknown path is copied unchanged (it was a fixed override, not a search
    /// dim).
    fn perturb_overlay(&self, winner: &Overlay) -> Overlay {
        let mut rng = self.rng.lock();
        winner
            .iter()
            .map(|(path, val)| {
                let v = match self.space.dims.get(path) {
                    Some(dist) => dist.perturb(val, &mut *rng),
                    None => val.clone(),
                };
                (path.clone(), v)
            })
            .collect()
    }

    /// The pure decision core. Drains a queued clone first (one per step), then
    /// records the point and, at a rung, culls a below-quantile loser while
    /// enqueueing a perturbed clone of the best survivor.
    pub(crate) fn decide(&self, trial: u32, obj: f64, budget: u64) -> PbtDecision {
        let mut st = self.state.lock();

        // 1. Emit a queued clone first (one per step), under the spawn cap.
        if st.spawns_done < self.cfg.max_spawns {
            if let Some((overlay, resume)) = st.queue.pop_front() {
                st.spawns_done += 1;
                return PbtDecision::Spawn { overlay, resume };
            }
        }

        if st.killed.contains(&trial) {
            return PbtDecision::Continue;
        }
        // A diverged trial (non-finite objective) is killed without polluting
        // peer history or triggering an exploit (a NaN must not become a winner).
        if !obj.is_finite() {
            st.killed.insert(trial);
            return PbtDecision::Kill;
        }
        st.history.entry(trial).or_default().insert(budget, obj);

        if !self.cfg.rungs.contains(&budget) || st.acted.contains(&(trial, budget)) {
            return PbtDecision::Continue;
        }
        // Peers at THIS rung (incl. self), as scores (higher = better).
        let peers: Vec<(u32, f64)> = st
            .history
            .iter()
            .filter_map(|(t, h)| h.get(&budget).map(|o| (*t, self.score(*o))))
            .collect();
        if peers.len() < self.cfg.min_peers.max(2) {
            return PbtDecision::Continue;
        }
        st.acted.insert((trial, budget));
        let scores: Vec<f64> = peers.iter().map(|(_, s)| *s).collect();
        let thr = percentile(&scores, self.cfg.bottom_quantile);
        if self.score(obj) >= thr {
            return PbtDecision::Continue; // a survivor — keep training
        }
        // Loser: exploit the best OTHER survivor at this rung.
        let winner = peers
            .iter()
            .filter(|(t, _)| *t != trial && !st.killed.contains(t))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(t, _)| *t);
        let Some(wt) = winner else {
            return PbtDecision::Continue; // no eligible winner yet — let it run
        };
        let Some(w) = self.trials.get(wt as usize) else {
            return PbtDecision::Continue;
        };
        let resume = PbtResume {
            winner_trial: wt,
            resume_dir: w.resume_dir.clone(),
        };
        let woverlay = w.overlay.clone();
        // Drop the lock before perturbing (perturb takes the rng lock; no nested
        // state lock needed and keeps the two locks from ever ordering).
        st.killed.insert(trial);
        // Don't enqueue a clone we could never emit (already at the cap incl.
        // queued-but-unemitted) — it would just leak. The loser is still killed.
        let at_cap = st.spawns_done + st.queue.len() >= self.cfg.max_spawns;
        drop(st);
        if at_cap {
            return PbtDecision::Kill;
        }
        let overlay = self.perturb_overlay(&woverlay);
        // Re-acquire only to enqueue (on_step is single-threaded, so nothing
        // changed `spawns_done`/`queue` between the drop and here).
        self.state.lock().queue.push_back((overlay, resume));
        PbtDecision::Kill
    }
}

impl ControlPolicy for PbtPolicy {
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
            PbtDecision::Continue => Control::Continue,
            PbtDecision::Kill => Control::KillBranch,
            PbtDecision::Spawn { overlay, resume } => match (self.factory)(&overlay, &resume) {
                Ok(subplan) => Control::Spawn(SpawnDelta {
                    subplan,
                    label: Some(format!("pbt-clone<-t{}", resume.winner_trial)),
                }),
                Err(e) => {
                    tracing::warn!("pbt clone factory failed: {e}");
                    Control::Continue
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn space() -> SearchSpace {
        let mut s = SearchSpace::default();
        s.dims.insert(
            "lr".into(),
            crate::hpo::space::Dist::LogUniform {
                low: 1e-4,
                high: 1e-1,
            },
        );
        s
    }

    fn policy(rungs: Vec<u64>) -> PbtPolicy {
        // Two trials with distinct overlays + resume dirs.
        let trials = vec![
            PbtTrial {
                overlay: vec![("lr".into(), json!(0.01))],
                resume_dir: "/jobs/t0".into(),
            },
            PbtTrial {
                overlay: vec![("lr".into(), json!(0.02))],
                resume_dir: "/jobs/t1".into(),
            },
        ];
        let cfg = PbtConfig {
            metric_key: "val_r".into(),
            budget_key: "epoch".into(),
            maximize: true,
            rungs,
            bottom_quantile: 50.0,
            min_peers: 2,
            max_spawns: 16,
        };
        // Factory unused by the pure `decide` tests.
        let factory: TrialFactory = Arc::new(|_, _| Err("unused".into()));
        PbtPolicy::new(vec![Some(0), Some(1)], trials, space(), cfg, factory, 7)
    }

    #[test]
    fn loser_at_rung_is_killed_and_clone_enqueued() {
        let p = policy(vec![1]);
        // Both trials report at rung 1. t1 is the winner (0.9), t0 the loser (0.1).
        assert_eq!(
            p.decide(1, 0.9, 1),
            PbtDecision::Continue,
            "winner survives"
        );
        match p.decide(0, 0.1, 1) {
            PbtDecision::Kill => {}
            d => panic!("loser must be killed, got {d:?}"),
        }
        // The clone was enqueued; the NEXT step drains it as a Spawn warm-started
        // from t1 (the winner), with a perturbed lr within the dim's support.
        match p.decide(1, 0.95, 2) {
            PbtDecision::Spawn { overlay, resume } => {
                assert_eq!(resume.winner_trial, 1);
                assert_eq!(resume.resume_dir, PathBuf::from("/jobs/t1"));
                let lr = overlay
                    .iter()
                    .find(|(k, _)| k == "lr")
                    .and_then(|(_, v)| v.as_f64())
                    .unwrap();
                assert!((1e-4..=1e-1).contains(&lr), "perturbed lr {lr} in support");
                // It is a PERTURBATION of the winner's 0.02 (×0.8 or ×1.2).
                assert!(lr != 0.02 || (0.016..=0.024).contains(&lr));
            }
            d => panic!("queued clone must emit as Spawn, got {d:?}"),
        }
    }

    #[test]
    fn diverged_trial_killed_no_exploit() {
        let p = policy(vec![1]);
        // A non-finite objective kills the trial but must NOT enqueue a clone
        // (the next step has nothing queued → Continue, not Spawn).
        assert_eq!(p.decide(0, f64::NAN, 1), PbtDecision::Kill);
        assert_eq!(
            p.decide(1, 0.5, 2),
            PbtDecision::Continue,
            "no clone from a NaN"
        );
    }

    #[test]
    fn off_rung_never_acts() {
        let p = policy(vec![4]); // only rung 4
        assert_eq!(p.decide(0, 0.1, 1), PbtDecision::Continue);
        assert_eq!(p.decide(1, 0.9, 1), PbtDecision::Continue);
    }

    #[test]
    fn acts_once_per_trial_per_rung() {
        let p = policy(vec![1]);
        assert_eq!(p.decide(1, 0.9, 1), PbtDecision::Continue);
        assert!(matches!(p.decide(0, 0.1, 1), PbtDecision::Kill));
        // A second report from the (now killed) loser at the same rung does
        // nothing but drain the queued clone.
        assert!(matches!(p.decide(1, 0.9, 1), PbtDecision::Spawn { .. }));
        assert_eq!(
            p.decide(0, 0.1, 1),
            PbtDecision::Continue,
            "killed trial is inert"
        );
    }

    #[test]
    fn spawn_cap_halts_clones() {
        let mut pol = policy(vec![1]);
        pol.cfg.max_spawns = 0; // no clones allowed
        assert_eq!(pol.decide(1, 0.9, 1), PbtDecision::Continue);
        assert!(matches!(pol.decide(0, 0.1, 1), PbtDecision::Kill));
        // Cap is 0 → the queued clone never emits.
        assert_eq!(
            pol.decide(1, 0.95, 2),
            PbtDecision::Continue,
            "cap blocks the clone"
        );
    }
}
