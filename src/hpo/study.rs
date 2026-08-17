// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0109 multi-objective **studies**: the spec a study is declared in, the
//! ledger that runs the ask/admit/record loop, and the `pareto.json` it emits.
//!
//! The single-objective [`HpoManifest`](super::results::HpoManifest) reads one
//! metric and ranks it. A study reads a VECTOR and reports the Pareto front,
//! because with competing objectives no total order exists — collapsing them to
//! one score picks a winner by a weighting nobody declared.
//!
//! **Two behaviours the ADR calls out explicitly, both implemented here rather
//! than left to the executor:**
//!
//! *A re-proposed config is a cache hit, not new spend.* Model-based samplers
//! re-suggest points near good ones and will land on an evaluated config exactly.
//! The ledger keys trials by canonical config identity and returns the recorded
//! outcome instead of scheduling anything.
//!
//! *An over-budget proposal is REFUSED and recorded, never forced onto the box.*
//! Admission is consulted BEFORE a trial is spawned, so a refusal costs nothing
//! and still becomes an observation — infeasibility is signal, and a sampler
//! that never learns which region is unaffordable proposes it forever. This is
//! also why the refusal is not recovered by parsing `status.jsonl`: a refusal
//! reason is free-form prose, and classifying on substrings is the same trap
//! that makes an OOM kill look like a scheduler cancel.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::pareto::{Direction, ParetoPoint, ParetoReport};
use super::space::{Overlay, SearchSpace};

/// One declared objective: where to read it, and which way is better.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Objective {
    /// Display name, used as the `pareto.json` column label.
    pub name: String,
    /// Dotted key read out of each trial's `StageStep` update.
    pub metric: String,
    pub direction: Direction,
}

/// A study, as declared in TOML.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudySpec {
    /// Recipe each trial runs (the fixed part; search dims overlay its args).
    pub recipe: String,
    /// At least two objectives — a one-objective "study" is an HPO run, and
    /// `blut hpo run` already does that better.
    pub objectives: Vec<Objective>,
    #[serde(default)]
    pub space: SearchSpace,
    /// random | grid | tpe | mvtpe | gp
    #[serde(default = "default_sampler")]
    pub sampler: String,
    #[serde(default)]
    pub seed: u64,
    /// Per-trial RAM estimate in GiB, used for the admission check. Declared
    /// rather than measured because admission must happen BEFORE the spawn.
    #[serde(default)]
    pub trial_ram_gib: f64,
}

fn default_sampler() -> String {
    "mvtpe".to_string()
}

impl StudySpec {
    /// Parse a study TOML, rejecting a spec that cannot produce a front.
    pub fn from_toml(text: &str) -> Result<StudySpec, String> {
        let spec: StudySpec =
            toml::from_str(text).map_err(|e| format!("study spec parse error: {e}"))?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.recipe.trim().is_empty() {
            return Err("study declares no recipe".into());
        }
        if self.objectives.len() < 2 {
            return Err(format!(
                "a study needs at least 2 objectives (got {}); one objective is \
                 an HPO run — use `blut hpo run`",
                self.objectives.len()
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for o in &self.objectives {
            if o.metric.trim().is_empty() {
                return Err(format!("objective '{}' declares no metric", o.name));
            }
            if !seen.insert(&o.name) {
                return Err(format!("duplicate objective name '{}'", o.name));
            }
        }
        if self.space.dims.is_empty() {
            return Err("study declares an empty search space".into());
        }
        self.space.validate()?;
        Ok(())
    }

    pub fn directions(&self) -> Vec<Direction> {
        self.objectives.iter().map(|o| o.direction).collect()
    }

    pub fn objective_names(&self) -> Vec<String> {
        self.objectives.iter().map(|o| o.name.clone()).collect()
    }
}

/// How a proposed trial resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    /// Ran and reported every declared objective.
    Done,
    /// Identical config already evaluated — returned from the ledger, nothing
    /// scheduled. Free.
    CacheHit,
    /// Broker refused admission before the spawn. Costs nothing, and is still
    /// an observation.
    Infeasible,
    /// Ran but did not report every objective (or reported a non-finite one).
    Unmeasured,
    /// Genuine crash.
    Failed,
    /// Stopped early by the scheduler (ASHA/median) or a plan cancel. Its
    /// reported values ARE real measurements — early-stop happens AFTER a trial
    /// reports — so it still contributes a point; the status exists so the
    /// operator can see the point came from a trial that never ran to budget.
    Killed,
}

impl TrialStatus {
    /// Whether this trial contributes a point to the Pareto front. Only a
    /// complete, finite measurement does.
    pub fn is_measured(self) -> bool {
        matches!(
            self,
            TrialStatus::Done | TrialStatus::CacheHit | TrialStatus::Killed
        )
    }
}

/// One trial's record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudyTrial {
    pub trial_id: u32,
    pub overlay: Overlay,
    /// One entry per declared objective, in declaration order. `None` = the
    /// trial never reported that objective.
    pub objectives: Vec<Option<f64>>,
    pub status: TrialStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// Present for `Infeasible`: why admission refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl StudyTrial {
    /// The point this trial contributes, or `None` if it is not a complete
    /// finite measurement.
    pub fn point(&self) -> Option<ParetoPoint> {
        if !self.status.is_measured() {
            return None;
        }
        let mut objs = Vec::with_capacity(self.objectives.len());
        for o in &self.objectives {
            match o {
                Some(v) if v.is_finite() => objs.push(*v),
                _ => return None,
            }
        }
        Some(ParetoPoint {
            trial_id: self.trial_id,
            objectives: objs,
            cost: self.cost,
        })
    }
}

/// Canonical identity of a proposed config — the cache key.
///
/// Sorted by dimension name so two overlays that differ only in ordering are
/// the SAME config; a sampler is free to emit dims in any order and must not
/// thereby buy a second evaluation of a point already paid for. Values are
/// compared through their JSON form, so `2` and `2.0` are distinct — which is
/// correct: they reach the recipe as different argument text.
pub fn config_key(overlay: &Overlay) -> String {
    let sorted: BTreeMap<&str, String> = overlay
        .iter()
        .map(|(k, v)| (k.as_str(), v.to_string()))
        .collect();
    sorted
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("\u{1f}")
}

/// The running record of a study.
#[derive(Debug, Default)]
pub struct StudyLedger {
    trials: Vec<StudyTrial>,
    /// config key → index into `trials`.
    seen: BTreeMap<String, usize>,
    next_id: u32,
}

/// What the ledger did with a proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proposal {
    /// Not seen before and admitted — the caller must evaluate it.
    Evaluate(u32),
    /// Already evaluated; nothing was scheduled.
    CacheHit(u32),
    /// Admission refused; nothing was scheduled.
    Refused(u32),
}

impl StudyLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn trials(&self) -> &[StudyTrial] {
        &self.trials
    }

    pub fn len(&self) -> usize {
        self.trials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trials.is_empty()
    }

    /// How many trials the caller actually has to run (i.e. excluding cache
    /// hits and refusals). This is the number a budget should be spent against.
    pub fn evaluated(&self) -> usize {
        self.trials
            .iter()
            .filter(|t| {
                matches!(
                    t.status,
                    TrialStatus::Done | TrialStatus::Unmeasured | TrialStatus::Failed
                )
            })
            .count()
    }

    /// Offer a proposed config.
    ///
    /// Order is deliberate: the CACHE is checked before admission. A config
    /// already evaluated costs nothing to return, so asking the broker whether
    /// we may afford it would be asking permission to spend nothing — and a
    /// tightened budget would then start "refusing" points already paid for,
    /// silently shrinking the front.
    pub fn propose(
        &mut self,
        overlay: Overlay,
        n_objectives: usize,
        admit: impl FnOnce(&Overlay) -> Result<(), String>,
    ) -> Proposal {
        let key = config_key(&overlay);
        if let Some(&idx) = self.seen.get(&key) {
            return Proposal::CacheHit(self.trials[idx].trial_id);
        }
        let id = self.next_id;
        self.next_id += 1;
        match admit(&overlay) {
            Err(refusal) => {
                self.trials.push(StudyTrial {
                    trial_id: id,
                    overlay,
                    objectives: vec![None; n_objectives],
                    status: TrialStatus::Infeasible,
                    cost: None,
                    refusal: Some(refusal),
                });
                // Refusals are NOT cached by config key: admission depends on
                // live box state, so the same config may be affordable later.
                // Caching it would make one transient refusal permanent.
                Proposal::Refused(id)
            }
            Ok(()) => {
                self.trials.push(StudyTrial {
                    trial_id: id,
                    overlay,
                    objectives: vec![None; n_objectives],
                    status: TrialStatus::Unmeasured,
                    cost: None,
                    refusal: None,
                });
                self.seen.insert(key, self.trials.len() - 1);
                Proposal::Evaluate(id)
            }
        }
    }

    /// Record an evaluated trial's objectives.
    ///
    /// A trial reporting fewer values than declared, or any non-finite one,
    /// stays `Unmeasured` — it is not a point on the front. Silently treating a
    /// partial report as complete would put a fabricated optimum on the front.
    pub fn record(&mut self, trial_id: u32, objectives: Vec<Option<f64>>, cost: Option<f64>) {
        let Some(t) = self.trials.iter_mut().find(|t| t.trial_id == trial_id) else {
            return;
        };
        let complete = objectives.len() == t.objectives.len()
            && objectives.iter().all(|o| o.is_some_and(|v| v.is_finite()));
        t.objectives = objectives;
        t.cost = cost;
        t.status = if complete {
            TrialStatus::Done
        } else {
            TrialStatus::Unmeasured
        };
    }

    /// Mark a trial as a genuine crash.
    pub fn record_failure(&mut self, trial_id: u32) {
        if let Some(t) = self.trials.iter_mut().find(|t| t.trial_id == trial_id) {
            t.status = TrialStatus::Failed;
        }
    }

    /// The study's deliverable.
    pub fn report(&self, spec: &StudySpec) -> ParetoReport {
        let points: Vec<ParetoPoint> = self.trials.iter().filter_map(|t| t.point()).collect();
        let mut report = ParetoReport::build(spec.objective_names(), spec.directions(), &points);
        // `ParetoReport::build` only knows about points it was given, so trials
        // that produced no point at all must be added here — otherwise a study
        // whose trials mostly failed reports a clean small front and looks
        // healthy.
        let mut unmeasured: Vec<u32> = self
            .trials
            .iter()
            .filter(|t| t.point().is_none())
            .map(|t| t.trial_id)
            .collect();
        unmeasured.extend(report.unmeasured_trials.iter().copied());
        unmeasured.sort_unstable();
        unmeasured.dedup();
        report.unmeasured_trials = unmeasured;
        report
    }

    /// Trials the sampler should learn from, as `(overlay, objective_index)`
    /// observations for a single objective. Infeasible trials are excluded from
    /// the objective view — they have no measurement — but the caller can still
    /// see them via [`Self::trials`].
    pub fn observations(&self, objective_index: usize) -> Vec<(Overlay, f64)> {
        self.trials
            .iter()
            .filter(|t| t.status.is_measured())
            .filter_map(|t| {
                t.objectives
                    .get(objective_index)
                    .and_then(|o| *o)
                    .filter(|v| v.is_finite())
                    .map(|v| (t.overlay.clone(), v))
            })
            .collect()
    }
}

/// Rebuild a study's ledger from the manifest plus the raw `status.jsonl`
/// lines, reading EVERY declared objective per trial.
///
/// This is the multi-objective sibling of
/// [`results::reconstruct`](super::results::reconstruct), and it differs in the
/// one way that matters: that function keeps the best value of ONE metric, so a
/// trial that reported it is done. Here a trial is only `Done` when it reported
/// **all** of them — a config that optimised error but never emitted a ratio is
/// not a point on the front, and treating it as one would put a half-measured
/// config where a real trade-off belongs.
///
/// Per objective the best value in that objective's OWN direction is kept, for
/// the same reason single-objective HPO keeps a best rather than a last: a
/// training curve reports many steps, and the trial's result is its best, not
/// wherever it happened to stop.
///
/// `stage_skipped` is the DAG cache hit ADR 0109 relies on ("re-proposing an
/// already-evaluated config is a cache hit, i.e. free, no special-casing") —
/// the node was not re-run, so it is recorded as [`TrialStatus::CacheHit`]
/// rather than as work performed.
pub fn reconstruct_study(
    spec: &StudySpec,
    manifest: &super::results::HpoManifest,
    status_lines: &[String],
) -> StudyLedger {
    use super::scheduler::dotted_f64;
    use serde_json::Value;

    let n_obj = spec.objectives.len();
    let n = manifest.trials.len();
    let mut best: Vec<Vec<Option<f64>>> = vec![vec![None; n_obj]; n];
    let mut skipped = vec![false; n];
    let mut failed = vec![false; n];
    let mut killed = vec![false; n];

    let trial_of = |topo: u32| -> Option<usize> {
        manifest
            .trial_of_topo
            .get(topo as usize)
            .copied()
            .flatten()
            .map(|t| t as usize)
    };

    for line in status_lines {
        let Ok(ev) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let kind = ev.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        let Some(topo) = ev.get("node_idx").and_then(|v| v.as_u64()) else {
            continue;
        };
        let Some(t) = trial_of(topo as u32) else {
            continue;
        };
        match kind {
            "stage_step" => {
                let Some(update) = ev.get("update") else {
                    continue;
                };
                for (i, obj) in spec.objectives.iter().enumerate() {
                    let Some(v) = dotted_f64(update, &obj.metric) else {
                        continue;
                    };
                    if !v.is_finite() {
                        continue;
                    }
                    best[t][i] = Some(match best[t][i] {
                        None => v,
                        Some(b) => match obj.direction {
                            Direction::Maximize => b.max(v),
                            Direction::Minimize => b.min(v),
                        },
                    });
                }
            }
            // The cache hit the ADR relies on: the node was NOT re-run.
            "stage_skipped" => skipped[t] = true,
            "stage_failed" => {
                // Same discrimination the single-objective path documents: a
                // cancel is a scheduler kill, anything else is a real crash.
                // Match only "cancel" — never "kill", which would misread the
                // Linux OOM killer's "Killed process …" as a scheduler action.
                let err = ev.get("error").and_then(|e| e.as_str()).unwrap_or("");
                if err.to_ascii_lowercase().contains("cancel") {
                    killed[t] = true;
                } else {
                    // No error string is a crash, not a cancel: every cancel
                    // path stamps a "cancelled…" message.
                    failed[t] = true;
                }
            }
            _ => {}
        }
    }

    let mut ledger = StudyLedger::new();
    for (t, rec) in manifest.trials.iter().enumerate() {
        let objectives = best[t].clone();
        let complete = objectives.iter().all(|o| o.is_some_and(|v| v.is_finite()));
        let status = if failed[t] {
            TrialStatus::Failed
        } else if !complete {
            // Incomplete beats every other label: a trial that never reported
            // all its objectives is not a point, however it ended.
            TrialStatus::Unmeasured
        } else if killed[t] {
            TrialStatus::Killed
        } else if skipped[t] {
            TrialStatus::CacheHit
        } else {
            TrialStatus::Done
        };
        ledger.trials.push(StudyTrial {
            trial_id: rec.trial_id,
            overlay: rec.overlay.clone(),
            objectives,
            status,
            cost: None,
            refusal: None,
        });
    }
    ledger.next_id = ledger.trials.len() as u32;
    ledger
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpo::space::Dist;
    use serde_json::json;

    fn spec_toml() -> &'static str {
        r#"
recipe = "demo"
sampler = "mvtpe"
seed = 7
trial_ram_gib = 4.0

[[objectives]]
name = "prd"
metric = "eval.prd"
direction = "minimize"

[[objectives]]
name = "ratio"
metric = "eval.compression_ratio"
direction = "maximize"

[space.dims.lr]
dist = "log_uniform"
low = 1e-5
high = 1e-1

[space.dims.batch]
dist = "int_uniform"
low = 8
high = 64
"#
    }

    fn ov(lr: f64, batch: i64) -> Overlay {
        vec![("lr".into(), json!(lr)), ("batch".into(), json!(batch))]
    }

    fn admit_all(_: &Overlay) -> Result<(), String> {
        Ok(())
    }

    // ── spec ────────────────────────────────────────────────────────────
    #[test]
    fn a_study_spec_round_trips_from_toml() {
        let s = StudySpec::from_toml(spec_toml()).expect("valid spec");
        assert_eq!(s.recipe, "demo");
        assert_eq!(s.objective_names(), vec!["prd", "ratio"]);
        assert_eq!(
            s.directions(),
            vec![Direction::Minimize, Direction::Maximize]
        );
        assert_eq!(s.sampler, "mvtpe");
        assert_eq!(s.space.dims.len(), 2);
        assert!(matches!(
            s.space.dims.get("lr"),
            Some(Dist::LogUniform { .. })
        ));
    }

    #[test]
    fn a_one_objective_study_is_rejected_with_a_pointer_to_hpo_run() {
        let text = r#"
recipe = "demo"
[[objectives]]
name = "prd"
metric = "eval.prd"
direction = "minimize"
[space.dims.lr]
dist = "uniform"
low = 0.0
high = 1.0
"#;
        let err = StudySpec::from_toml(text).unwrap_err();
        assert!(err.contains("at least 2 objectives"), "{err}");
        assert!(
            err.contains("hpo run"),
            "should point at the right tool: {err}"
        );
    }

    #[test]
    fn a_spec_with_no_space_or_a_duplicate_objective_is_rejected() {
        let no_space = r#"
recipe = "demo"
[[objectives]]
name = "a"
metric = "m.a"
direction = "minimize"
[[objectives]]
name = "b"
metric = "m.b"
direction = "minimize"
"#;
        assert!(
            StudySpec::from_toml(no_space)
                .unwrap_err()
                .contains("empty search space")
        );

        let dupe = r#"
recipe = "demo"
[[objectives]]
name = "a"
metric = "m.a"
direction = "minimize"
[[objectives]]
name = "a"
metric = "m.b"
direction = "maximize"
[space.dims.x]
dist = "uniform"
low = 0.0
high = 1.0
"#;
        assert!(
            StudySpec::from_toml(dupe)
                .unwrap_err()
                .contains("duplicate objective name")
        );
    }

    /// The gate names `tests/fixtures/study_multiobj.toml` by path. Parsing it
    /// here means a spec-format change breaks the build rather than the gate,
    /// and the fixture cannot quietly rot into something the parser rejects.
    #[test]
    fn the_gate_fixture_parses_and_declares_competing_objectives() {
        let text = include_str!("../../tests/fixtures/study_multiobj.toml");
        let spec = StudySpec::from_toml(text).expect("the ADR 0109 gate fixture must parse");
        assert_eq!(spec.objective_names(), vec!["prd", "compression_ratio"]);
        assert_eq!(
            spec.directions(),
            vec![Direction::Minimize, Direction::Maximize],
            "the two objectives must pull in OPPOSITE directions, or the front \
             collapses to one point and a >=2-point gate passes by accident"
        );
        assert_eq!(spec.space.dims.len(), 3);
        assert!(
            spec.trial_ram_gib > 0.0,
            "admission needs a declared footprint"
        );
    }

    // ── cache identity ──────────────────────────────────────────────────
    #[test]
    fn config_key_ignores_dim_ordering_but_not_values() {
        let a = vec![("lr".to_string(), json!(0.1)), ("b".to_string(), json!(8))];
        let b = vec![("b".to_string(), json!(8)), ("lr".to_string(), json!(0.1))];
        assert_eq!(config_key(&a), config_key(&b), "ordering is not identity");
        let c = vec![("lr".to_string(), json!(0.2)), ("b".to_string(), json!(8))];
        assert_ne!(config_key(&a), config_key(&c));
    }

    #[test]
    fn a_reproposed_config_is_a_cache_hit_and_schedules_nothing() {
        let mut led = StudyLedger::new();
        let p1 = led.propose(ov(0.01, 32), 2, admit_all);
        let Proposal::Evaluate(id) = p1 else {
            panic!("first proposal must be evaluated, got {p1:?}")
        };
        led.record(id, vec![Some(0.3), Some(12.0)], Some(100.0));

        // A model-based sampler re-suggesting the same point must be free.
        let mut spawned = 0;
        let p2 = led.propose(ov(0.01, 32), 2, |_| {
            spawned += 1;
            Ok(())
        });
        assert_eq!(p2, Proposal::CacheHit(id));
        assert_eq!(spawned, 0, "a cache hit must not even consult admission");
        assert_eq!(led.len(), 1, "no second trial record");
        assert_eq!(led.evaluated(), 1, "and no second unit of spend");
    }

    #[test]
    fn dim_order_does_not_buy_a_second_evaluation() {
        let mut led = StudyLedger::new();
        let Proposal::Evaluate(id) = led.propose(ov(0.05, 16), 2, admit_all) else {
            panic!()
        };
        led.record(id, vec![Some(1.0), Some(2.0)], None);
        let reordered = vec![
            ("batch".to_string(), json!(16)),
            ("lr".to_string(), json!(0.05)),
        ];
        assert_eq!(led.propose(reordered, 2, admit_all), Proposal::CacheHit(id));
    }

    // ── admission / infeasibility ───────────────────────────────────────
    #[test]
    fn a_refused_proposal_is_recorded_infeasible_and_never_spawned() {
        let mut led = StudyLedger::new();
        let p = led.propose(ov(0.9, 64), 2, |_| {
            Err("RAM footprint 96.0G exceeds box capacity".into())
        });
        let Proposal::Refused(id) = p else {
            panic!("expected a refusal, got {p:?}")
        };
        let t = &led.trials()[0];
        assert_eq!(t.trial_id, id);
        assert_eq!(t.status, TrialStatus::Infeasible);
        assert!(
            t.refusal
                .as_deref()
                .unwrap()
                .contains("exceeds box capacity")
        );
        assert_eq!(led.evaluated(), 0, "a refusal costs no spend");
        assert!(t.point().is_none(), "and contributes no point");
    }

    #[test]
    fn a_refusal_is_not_cached_because_admission_depends_on_live_state() {
        // The box was full; later it is not. The same config must be allowed to
        // run — caching the refusal would make one transient state permanent.
        let mut led = StudyLedger::new();
        assert!(matches!(
            led.propose(ov(0.5, 32), 2, |_| Err("no room right now".into())),
            Proposal::Refused(_)
        ));
        assert!(
            matches!(
                led.propose(ov(0.5, 32), 2, admit_all),
                Proposal::Evaluate(_)
            ),
            "a later proposal of the same config must be evaluable"
        );
    }

    // ── measurement discipline ──────────────────────────────────────────
    #[test]
    fn a_partial_or_non_finite_report_stays_unmeasured() {
        let mut led = StudyLedger::new();
        let Proposal::Evaluate(a) = led.propose(ov(0.1, 8), 2, admit_all) else {
            panic!()
        };
        led.record(a, vec![Some(1.0)], None); // one value for two objectives
        assert_eq!(led.trials()[0].status, TrialStatus::Unmeasured);
        assert!(led.trials()[0].point().is_none());

        let Proposal::Evaluate(b) = led.propose(ov(0.2, 8), 2, admit_all) else {
            panic!()
        };
        led.record(b, vec![Some(1.0), Some(f64::NAN)], None);
        assert_eq!(led.trials()[1].status, TrialStatus::Unmeasured);
        assert!(
            led.trials()[1].point().is_none(),
            "NaN is not a measurement"
        );
    }

    #[test]
    fn observations_exclude_everything_that_is_not_a_measurement() {
        let mut led = StudyLedger::new();
        let Proposal::Evaluate(a) = led.propose(ov(0.1, 8), 2, admit_all) else {
            panic!()
        };
        led.record(a, vec![Some(0.4), Some(9.0)], None);
        let _ = led.propose(ov(0.9, 64), 2, |_| Err("refused".into()));
        let Proposal::Evaluate(c) = led.propose(ov(0.3, 8), 2, admit_all) else {
            panic!()
        };
        led.record_failure(c);

        let obs = led.observations(0);
        assert_eq!(obs.len(), 1, "only the measured trial is an observation");
        assert_eq!(obs[0].1, 0.4);
        assert_eq!(led.observations(1)[0].1, 9.0);
    }

    // ── the deliverable ─────────────────────────────────────────────────
    #[test]
    fn the_report_is_a_front_and_names_every_trial_that_produced_no_point() {
        let spec = StudySpec::from_toml(spec_toml()).unwrap();
        let mut led = StudyLedger::new();
        // minimize prd, maximize ratio
        for (lr, b, prd, ratio) in [
            (0.01, 32, 0.27, 9.8),  // on front
            (0.02, 32, 0.31, 12.4), // on front
            (0.03, 32, 0.52, 11.0), // dominated by (0.31, 12.4)
        ] {
            let Proposal::Evaluate(id) = led.propose(ov(lr, b), 2, admit_all) else {
                panic!()
            };
            led.record(id, vec![Some(prd), Some(ratio)], None);
        }
        // One refused, one crashed, one partial — none may reach the front.
        let _ = led.propose(ov(0.9, 64), 2, |_| Err("too big".into()));
        let Proposal::Evaluate(f) = led.propose(ov(0.04, 32), 2, admit_all) else {
            panic!()
        };
        led.record_failure(f);
        let Proposal::Evaluate(p) = led.propose(ov(0.05, 32), 2, admit_all) else {
            panic!()
        };
        led.record(p, vec![Some(0.1), None], None);

        let report = led.report(&spec);
        assert_eq!(report.objectives, vec!["prd", "ratio"]);
        let ids: Vec<u32> = report.front.iter().map(|q| q.trial_id).collect();
        assert_eq!(ids, vec![0, 1], "the dominated trial is off the front");
        assert!(report.front.len() >= 2, "ADR 0109 requires >= 2 points");
        // The refused (3), crashed (4) and partial (5) trials are all named.
        // Trial 2 is NOT among them: it is a complete, finite measurement that
        // simply lost. "Dominated" and "unmeasured" are different facts, and
        // conflating them would either hide a real evaluation or imply a failed
        // one was merely outcompeted.
        assert_eq!(report.unmeasured_trials, vec![3, 4, 5]);
        assert_eq!(led.trials()[2].status, TrialStatus::Done);
        assert!(
            led.trials()[2].point().is_some(),
            "the dominated trial still produced a point; it just is not on the front"
        );

        // And it serialises as the pareto.json the validator reads.
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"objectives\":[\"prd\",\"ratio\"]"));
        assert!(json.contains("\"directions\":[\"minimize\",\"maximize\"]"));
    }

    // ── read-back from status.jsonl ─────────────────────────────────────
    fn manifest_for(n: usize) -> crate::hpo::results::HpoManifest {
        crate::hpo::results::HpoManifest {
            recipe: "demo".into(),
            algo: "mvtpe".into(),
            metric: "eval.prd".into(),
            mode: "min".into(),
            budget_key: "epoch".into(),
            trials: (0..n)
                .map(|i| crate::hpo::results::TrialRec {
                    trial_id: i as u32,
                    overlay: ov(0.01 * (i + 1) as f64, 32),
                    n_nodes: 1,
                })
                .collect(),
            // node_idx i belongs to trial i
            trial_of_topo: (0..n).map(|i| Some(i as u32)).collect(),
        }
    }

    fn step(node: usize, prd: f64, ratio: f64) -> String {
        json!({"kind":"stage_step","node_idx":node,
               "update":{"eval":{"prd":prd,"compression_ratio":ratio}}})
        .to_string()
    }

    #[test]
    fn read_back_keeps_each_objective_s_best_in_its_own_direction() {
        let spec = StudySpec::from_toml(spec_toml()).unwrap();
        let m = manifest_for(1);
        // prd is minimised, ratio maximised — from the SAME stream the trial's
        // result must be min(prd) and max(ratio), not the last pair seen.
        let lines = vec![step(0, 0.9, 5.0), step(0, 0.4, 11.0), step(0, 0.7, 8.0)];
        let led = reconstruct_study(&spec, &m, &lines);
        assert_eq!(led.trials()[0].status, TrialStatus::Done);
        assert_eq!(led.trials()[0].objectives, vec![Some(0.4), Some(11.0)]);
    }

    #[test]
    fn a_trial_reporting_only_one_objective_is_unmeasured_not_done() {
        let spec = StudySpec::from_toml(spec_toml()).unwrap();
        let m = manifest_for(1);
        // Reports prd only. Single-objective HPO would call this finished;
        // a study must not, or a half-measured config lands on the front.
        let lines = vec![
            json!({"kind":"stage_step","node_idx":0,"update":{"eval":{"prd":0.3}}}).to_string(),
        ];
        let led = reconstruct_study(&spec, &m, &lines);
        assert_eq!(led.trials()[0].status, TrialStatus::Unmeasured);
        assert!(led.trials()[0].point().is_none());
    }

    #[test]
    fn a_skipped_stage_is_recorded_as_the_cache_hit_it_is() {
        let spec = StudySpec::from_toml(spec_toml()).unwrap();
        let m = manifest_for(1);
        let lines = vec![
            step(0, 0.2, 14.0),
            json!({"kind":"stage_skipped","node_idx":0}).to_string(),
        ];
        let led = reconstruct_study(&spec, &m, &lines);
        assert_eq!(
            led.trials()[0].status,
            TrialStatus::CacheHit,
            "a re-proposed config resolved from cache is free, not work performed"
        );
        assert!(led.trials()[0].point().is_some(), "and still a real point");
    }

    #[test]
    fn an_early_stopped_trial_is_killed_but_still_a_point() {
        let spec = StudySpec::from_toml(spec_toml()).unwrap();
        let m = manifest_for(1);
        // ASHA stops underperformers AFTER they report, so the values are real
        // measurements — but the operator must be able to see that this point
        // came from a trial that never ran to budget.
        let lines = vec![
            step(0, 0.45, 6.0),
            json!({"kind":"stage_failed","node_idx":0,"error":"cancelled by scheduler (asha rung 1)"})
                .to_string(),
        ];
        let led = reconstruct_study(&spec, &m, &lines);
        assert_eq!(led.trials()[0].status, TrialStatus::Killed);
        assert!(
            led.trials()[0].point().is_some(),
            "an early-stopped trial's reported values are still measurements"
        );
    }

    #[test]
    fn an_incomplete_trial_is_unmeasured_however_it_ended() {
        let spec = StudySpec::from_toml(spec_toml()).unwrap();
        let m = manifest_for(1);
        // Killed before it reported the second objective: not a point.
        let lines = vec![
            json!({"kind":"stage_step","node_idx":0,"update":{"eval":{"prd":0.5}}}).to_string(),
            json!({"kind":"stage_failed","node_idx":0,"error":"cancelled"}).to_string(),
        ];
        let led = reconstruct_study(&spec, &m, &lines);
        assert_eq!(
            led.trials()[0].status,
            TrialStatus::Unmeasured,
            "incomplete must beat Killed — a partial measurement is not a point"
        );
        assert!(led.trials()[0].point().is_none());
    }

    #[test]
    fn a_crash_is_failed_but_a_cancel_is_not() {
        let spec = StudySpec::from_toml(spec_toml()).unwrap();
        let m = manifest_for(2);
        let lines = vec![
            step(0, 0.3, 9.0),
            json!({"kind":"stage_failed","node_idx":0,"error":"Killed process 123 (OOM)"})
                .to_string(),
            step(1, 0.4, 8.0),
            json!({"kind":"stage_failed","node_idx":1,"error":"stage cancelled by scheduler"})
                .to_string(),
        ];
        let led = reconstruct_study(&spec, &m, &lines);
        assert_eq!(
            led.trials()[0].status,
            TrialStatus::Failed,
            "an OOM is a crash"
        );
        assert_ne!(
            led.trials()[1].status,
            TrialStatus::Failed,
            "a scheduler cancel is not a crash"
        );
    }

    #[test]
    fn read_back_ignores_events_for_nodes_owned_by_no_trial() {
        let spec = StudySpec::from_toml(spec_toml()).unwrap();
        let mut m = manifest_for(1);
        m.trial_of_topo.push(None); // node 1 belongs to no trial
        let lines = vec![step(0, 0.5, 7.0), step(1, 0.01, 99.0)];
        let led = reconstruct_study(&spec, &m, &lines);
        assert_eq!(
            led.trials()[0].objectives,
            vec![Some(0.5), Some(7.0)],
            "a non-trial node must not contribute a spectacular fake result"
        );
        assert_eq!(led.len(), 1);
    }

    #[test]
    fn a_study_where_everything_failed_reports_an_empty_front_not_a_clean_one() {
        let spec = StudySpec::from_toml(spec_toml()).unwrap();
        let mut led = StudyLedger::new();
        for i in 0..3 {
            let Proposal::Evaluate(id) = led.propose(ov(0.1 * (i + 1) as f64, 8), 2, admit_all)
            else {
                panic!()
            };
            led.record_failure(id);
        }
        let report = led.report(&spec);
        assert!(report.front.is_empty());
        assert_eq!(
            report.unmeasured_trials.len(),
            3,
            "every failure is named, so this cannot read as a healthy small front"
        );
    }
}
