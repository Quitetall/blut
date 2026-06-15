//! HPO scheduler (v0.20) — a [`ControlPolicy`] that early-stops underperforming
//! trials via the existing `KillBranch`.
//!
//! Each trial is a sub-graph of the merged fan-out plan. The executor stamps
//! every `StageStep` with the emitting node's TOPO position (`node_idx`); the
//! scheduler maps that → trial via [`build_trial_of_topo`], reads the objective +
//! budget from the step payload, tracks per-trial history, and asks an
//! [`EarlyStop`] strategy whether to kill the trial at the current budget.
//! `on_step` is `&self` (the trait), so cross-step state lives behind a
//! `parking_lot::Mutex` — the sole caller is the single coordinator thread, so
//! it is uncontended.

use std::collections::{BTreeMap, HashMap, HashSet};

use parking_lot::Mutex;
use serde_json::Value;

use crate::framework::control::{Control, ControlPolicy, StepMetrics};
use crate::framework::plan::NodeId;

/// An early-stop rule. Works on a "score" where **higher is better** — the
/// scheduler negates the raw objective for `minimize`, so the strategy never
/// branches on direction.
pub trait EarlyStop: Send + Sync {
    /// Should the trial reporting `score` at `budget` be stopped, given the
    /// `peer_scores` (all trials with a value at this budget, incl. self)?
    fn should_stop(&self, budget: u64, score: f64, peer_scores: &[f64]) -> bool;
    /// Name (for logs / `--algo`).
    fn name(&self) -> &'static str;
}

struct SchedState {
    /// trial_id → (budget → objective).
    history: HashMap<u32, BTreeMap<u64, f64>>,
    /// trials already KillBranch'd (idempotent — don't re-kill).
    killed: HashSet<u32>,
}

/// The HPO control policy.
pub struct HpoScheduler {
    /// topo position → trial_id (None for a node not owned by any trial — never
    /// happens in a pure fan-out, but tolerated).
    trial_of_topo: Vec<Option<u32>>,
    /// Dotted key into the step payload for the objective + the budget coord.
    metric_key: String,
    budget_key: String,
    /// `true` = maximize the objective (negated to a score otherwise).
    maximize: bool,
    /// No trial may be stopped before reaching this budget (avoids killing on
    /// noisy early values).
    grace: u64,
    strategy: Box<dyn EarlyStop>,
    state: Mutex<SchedState>,
}

impl HpoScheduler {
    pub fn new(
        trial_of_topo: Vec<Option<u32>>,
        metric_key: impl Into<String>,
        budget_key: impl Into<String>,
        maximize: bool,
        grace: u64,
        strategy: Box<dyn EarlyStop>,
    ) -> Self {
        Self {
            trial_of_topo,
            metric_key: metric_key.into(),
            budget_key: budget_key.into(),
            maximize,
            grace,
            strategy,
            state: Mutex::new(SchedState {
                history: HashMap::new(),
                killed: HashSet::new(),
            }),
        }
    }

    fn score(&self, obj: f64) -> f64 {
        if self.maximize { obj } else { -obj }
    }
}

impl ControlPolicy for HpoScheduler {
    fn on_step(&self, m: &StepMetrics) -> Control {
        let Some(trial) = self
            .trial_of_topo
            .get(m.node_idx as usize)
            .copied()
            .flatten()
        else {
            return Control::Continue;
        };
        // Objective is required to make any decision; budget defaults to 0
        // (pre-grace) when absent.
        let Some(obj) = dotted_f64(m.update, &self.metric_key) else {
            return Control::Continue;
        };
        let budget = dotted_u64(m.update, &self.budget_key).unwrap_or(0);

        let mut st = self.state.lock();
        if st.killed.contains(&trial) {
            return Control::Continue;
        }
        // A non-finite objective (a "nan"/"inf" stringy metric → parsed) means
        // the trial diverged — kill it immediately (don't pollute peer history).
        if !obj.is_finite() {
            st.killed.insert(trial);
            return Control::KillBranch;
        }
        // Record this point regardless — peers need the history even pre-grace.
        st.history.entry(trial).or_default().insert(budget, obj);
        if budget < self.grace {
            return Control::Continue;
        }
        // Peer scores at THIS budget (incl. self), higher = better.
        let peers: Vec<f64> = st
            .history
            .values()
            .filter_map(|h| h.get(&budget).copied())
            .map(|o| self.score(o))
            .collect();
        if self.strategy.should_stop(budget, self.score(obj), &peers) {
            st.killed.insert(trial);
            return Control::KillBranch;
        }
        Control::Continue
    }
}

/// Map each topo position → the trial that owns it. `topo_order[p]` is the
/// `NodeId` at topo position `p`; a node belongs to trial `i` iff its id is in
/// `[offsets[i], offsets[i+1])` (the last trial runs to `n_nodes`).
pub fn build_trial_of_topo(
    topo_order: &[NodeId],
    offsets: &[NodeId],
    n_nodes: NodeId,
) -> Vec<Option<u32>> {
    let trial_of_node = |id: NodeId| -> Option<u32> {
        for i in 0..offsets.len() {
            let lo = offsets[i];
            let hi = offsets.get(i + 1).copied().unwrap_or(n_nodes);
            if id >= lo && id < hi {
                return Some(i as u32);
            }
        }
        None
    };
    topo_order.iter().map(|&id| trial_of_node(id)).collect()
}

/// Read a dotted path (`a.b.c`) from a JSON object as an f64 (tolerating a
/// numeric value or a numeric string — some trainers stringify metrics).
pub fn dotted_f64(v: &Value, path: &str) -> Option<f64> {
    let node = dotted(v, path)?;
    node.as_f64().or_else(|| node.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
}

/// Read a dotted path as a u64 (number or numeric string; a float is floored).
pub fn dotted_u64(v: &Value, path: &str) -> Option<u64> {
    let node = dotted(v, path)?;
    node.as_u64()
        .or_else(|| node.as_f64().map(|x| x.max(0.0) as u64))
        .or_else(|| node.as_str().and_then(|s| s.trim().parse::<f64>().ok()).map(|x| x.max(0.0) as u64))
}

fn dotted<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = v;
    for part in path.split('.') {
        cur = cur.get(part)?;
    }
    Some(cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpo::median::MedianStop;
    use serde_json::json;

    #[test]
    fn dotted_reads_nested_and_stringy() {
        let v = json!({ "a": { "b": 0.5 }, "epoch": 3, "s": "0.7", "e2": "4" });
        assert_eq!(dotted_f64(&v, "a.b"), Some(0.5));
        assert_eq!(dotted_f64(&v, "s"), Some(0.7));
        assert_eq!(dotted_u64(&v, "epoch"), Some(3));
        assert_eq!(dotted_u64(&v, "e2"), Some(4));
        assert_eq!(dotted_f64(&v, "missing"), None);
    }

    #[test]
    fn trial_of_topo_maps_ranges() {
        // 2 trials, 2 nodes each: offsets [0,2], n=4. topo order interleaved.
        let topo = vec![0u32, 2, 1, 3];
        let map = build_trial_of_topo(&topo, &[0, 2], 4);
        assert_eq!(map, vec![Some(0), Some(1), Some(0), Some(1)]);
    }

    fn sched(maximize: bool, grace: u64) -> HpoScheduler {
        // 3 trials, 1 node each (topo == trial).
        let map = vec![Some(0), Some(1), Some(2)];
        HpoScheduler::new(
            map,
            "val_r",
            "epoch",
            maximize,
            grace,
            Box::new(MedianStop { percentile: 50.0, min_peers: 2 }),
        )
    }

    fn step<'a>(node_idx: u32, u: &'a Value) -> StepMetrics<'a> {
        StepMetrics { node_idx, stage_name: "t", update: u }
    }

    #[test]
    fn median_kills_below_median_after_grace() {
        let s = sched(true, 1);
        // Maximize. trial0=0.5 reports alone (<2 peers → continue). trial1=0.9 is
        // at/above the cohort median → survives. trial2=0.1 is below the cohort
        // median {0.5,0.9,0.1}→0.5 → killed.
        assert_eq!(s.on_step(&step(0, &json!({"val_r":0.5,"epoch":1}))), Control::Continue);
        assert_eq!(s.on_step(&step(1, &json!({"val_r":0.9,"epoch":1}))), Control::Continue);
        assert_eq!(s.on_step(&step(2, &json!({"val_r":0.1,"epoch":1}))), Control::KillBranch);
        // a killed trial is not re-killed.
        assert_eq!(s.on_step(&step(2, &json!({"val_r":0.05,"epoch":2}))), Control::Continue);
    }

    #[test]
    fn diverged_trial_killed_regardless_of_grace() {
        let s = sched(true, 100); // huge grace
        // a "nan" objective (stringy) → non-finite → killed immediately even
        // below grace, and even with no peers.
        assert_eq!(s.on_step(&step(0, &json!({"val_r":"nan","epoch":1}))), Control::KillBranch);
    }

    #[test]
    fn grace_protects_early_budgets() {
        let s = sched(true, 5);
        // budget 1 < grace 5: never killed even if worst.
        assert_eq!(s.on_step(&step(0, &json!({"val_r":0.9,"epoch":1}))), Control::Continue);
        assert_eq!(s.on_step(&step(1, &json!({"val_r":0.5,"epoch":1}))), Control::Continue);
        assert_eq!(s.on_step(&step(2, &json!({"val_r":0.1,"epoch":1}))), Control::Continue);
    }

    #[test]
    fn minimize_kills_above_median() {
        let s = sched(false, 1); // minimize: lower obj is better
        // trial0=0.5 alone (continue). trial1=0.1 is BEST when minimizing →
        // survives. trial2=0.9 is the worst (above cohort median) → killed.
        assert_eq!(s.on_step(&step(0, &json!({"val_r":0.5,"epoch":1}))), Control::Continue);
        assert_eq!(s.on_step(&step(1, &json!({"val_r":0.1,"epoch":1}))), Control::Continue);
        assert_eq!(s.on_step(&step(2, &json!({"val_r":0.9,"epoch":1}))), Control::KillBranch);
    }
}
