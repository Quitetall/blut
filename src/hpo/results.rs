// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! HPO trial tracking (v0.20 Phase 5).
//!
//! Source of truth = the filesystem (the BLUT philosophy): at launch the HPO run
//! writes `<job_dir>/hpo.json` (the trials + their overlays + the topo→trial
//! map + the objective metric/direction), and the per-trial outcome is
//! RECONSTRUCTED from the durable `status.jsonl` event stream — so `blut hpo
//! show`/`best` work during AND after a run, for every algorithm, with no DB to
//! keep in sync.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::scheduler::dotted_f64;
use super::space::Overlay;

/// One trial's identity, written into the manifest at launch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrialRec {
    pub trial_id: u32,
    pub overlay: Overlay,
    /// Number of plan nodes in this trial's sub-graph (for done-detection).
    pub n_nodes: u32,
}

/// `<job_dir>/hpo.json` — the HPO run's manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HpoManifest {
    pub recipe: String,
    pub algo: String,
    pub metric: String,
    /// "max" | "min".
    pub mode: String,
    pub budget_key: String,
    pub trials: Vec<TrialRec>,
    /// Topo position → trial_id (None = a node owned by no trial). The
    /// authoritative map for attributing a StageStep's `node_idx` to a trial.
    pub trial_of_topo: Vec<Option<u32>>,
}

impl HpoManifest {
    pub fn write_to(&self, job_dir: &std::path::Path) -> std::io::Result<()> {
        let body = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = job_dir.join("hpo.json.tmp");
        std::fs::write(&tmp, body)?;
        std::fs::rename(tmp, job_dir.join("hpo.json"))
    }

    pub fn read_from(job_dir: &std::path::Path) -> Option<HpoManifest> {
        let path = job_dir.join("hpo.json");
        let body = std::fs::read_to_string(&path).ok()?;
        // The file EXISTS (read succeeded) — a parse error means it is corrupt,
        // not "absent". Warn so a corrupt manifest isn't silently mistaken for a
        // non-HPO job (e.g. the `resolve_hpo_job` most-recent fallback).
        match serde_json::from_str(&body) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("warning: corrupt hpo manifest {}: {e}", path.display());
                None
            }
        }
    }

    /// Maximize unless the direction is "min" (case-insensitive — the CLI only
    /// ever writes lowercase, but a hand-edited manifest must not silently
    /// invert the leaderboard).
    pub fn maximize(&self) -> bool {
        !self.mode.eq_ignore_ascii_case("min")
    }
}

/// A trial's reconstructed outcome.
#[derive(Clone, Debug, Serialize)]
pub struct TrialOutcome {
    pub trial_id: u32,
    pub overlay: Overlay,
    /// Best objective seen (per the manifest's direction); None if the trial
    /// never reported the metric.
    pub objective: Option<f64>,
    /// pending | running | done | stopped | cancelled | failed. "stopped" =
    /// early-stopped by the HPO scheduler (a branch prune); "cancelled" = the
    /// plan was cancelled while the trial ran; "failed" = a genuine crash.
    pub status: &'static str,
}

/// Reconstruct per-trial outcomes from the manifest + the status.jsonl stream.
/// `status_lines` are the raw JSON lines (`jobs::read_status_lines`).
pub fn reconstruct(manifest: &HpoManifest, status_lines: &[String]) -> Vec<TrialOutcome> {
    let maximize = manifest.maximize();
    let n = manifest.trials.len();
    let mut best: Vec<Option<f64>> = vec![None; n];
    let mut finished: Vec<u32> = vec![0; n]; // successful output/cache/control terminals
    let mut began: Vec<bool> = vec![false; n];
    let mut stopped: Vec<bool> = vec![false; n]; // scheduler early stop (branch prune)
    let mut cancelled: Vec<bool> = vec![false; n]; // plan cancel
    let mut failed: Vec<bool> = vec![false; n]; // genuine crash (OOM/code/timeout)

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
                began[t] = true;
                if let Some(update) = ev.get("update")
                    && let Some(obj) = dotted_f64(update, &manifest.metric)
                    && obj.is_finite()
                {
                    best[t] = Some(match best[t] {
                        None => obj,
                        Some(b) if maximize => b.max(obj),
                        Some(b) => b.min(obj),
                    });
                }
            }
            "stage_begin" => began[t] = true,
            // A control-pruned stage is an expected terminal outcome for the
            // selected execution path. It contributes no objective and is not
            // a scheduler kill or a failure.
            "stage_end" | "stage_skipped" | "stage_pruned" => finished[t] += 1,
            "stage_failed" => {
                // A retry emits `stage_retrying`, NOT `stage_failed` (verified in
                // executor.rs), so a `stage_failed` is always terminal. EVERY
                // control-kill / plan-cancel path stamps a "cancelled…" string
                // (verified: the literal cancel messages + `StageError::Cancelled`
                // ⇒ "cancelled"); a genuine crash carries the real error. Match
                // ONLY "cancel" — NOT "kill", which would misclassify the Linux
                // OOM killer's "Killed process …" / "Out of memory: Killed" (a
                // real failure) as a scheduler kill.
                let err = ev.get("error").and_then(|e| e.as_str()).unwrap_or("");
                match classify_stage_failure(err) {
                    FailureKind::Stopped => stopped[t] = true,
                    FailureKind::Cancelled => cancelled[t] = true,
                    FailureKind::Failed => failed[t] = true,
                }
            }
            _ => {}
        }
    }

    manifest
        .trials
        .iter()
        .enumerate()
        .map(|(t, rec)| {
            // Cascade ORDER is load-bearing: a killed trial's pruned descendants
            // emit `stage_skipped` (counted in `finished`), so a killed trial can
            // satisfy `finished >= n_nodes` — checking `killed`/`failed` FIRST
            // keeps it labelled correctly rather than "done".
            let status = if stopped[t] {
                "stopped"
            } else if cancelled[t] {
                "cancelled"
            } else if failed[t] {
                "failed"
            } else if finished[t] >= rec.n_nodes {
                "done"
            } else if began[t] {
                "running"
            } else {
                "pending"
            };
            TrialOutcome {
                trial_id: rec.trial_id,
                overlay: rec.overlay.clone(),
                objective: best[t],
                status,
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureKind {
    Stopped,
    Cancelled,
    Failed,
}

/// Classify a terminal `stage_failed` error string. The executor writes
/// "cancelled by the control policy …" for a scheduler prune, a "cancelled…"
/// string for a plan cancel, and the stage's own error otherwise.
fn classify_stage_failure(err: &str) -> FailureKind {
    let lower = err.to_ascii_lowercase();
    if lower.contains("by the control policy") {
        FailureKind::Stopped
    } else if lower.contains("cancel") {
        FailureKind::Cancelled
    } else {
        FailureKind::Failed
    }
}

/// The error of every trial that genuinely failed, by trial position in
/// `manifest.trials` (`None` for a trial that did not fail). The first
/// failing node of a trial wins.
pub(crate) fn trial_errors(manifest: &HpoManifest, status_lines: &[String]) -> Vec<Option<String>> {
    let mut errors: Vec<Option<String>> = vec![None; manifest.trials.len()];
    for line in status_lines {
        let Ok(ev) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if ev.get("kind").and_then(|k| k.as_str()) != Some("stage_failed") {
            continue;
        }
        let Some(topo) = ev.get("node_idx").and_then(|v| v.as_u64()) else {
            continue;
        };
        let Some(t) = manifest
            .trial_of_topo
            .get(topo as usize)
            .copied()
            .flatten()
            .map(|t| t as usize)
        else {
            continue;
        };
        let err = ev.get("error").and_then(|e| e.as_str()).unwrap_or("");
        if classify_stage_failure(err) == FailureKind::Failed
            && let Some(slot) = errors.get_mut(t)
            && slot.is_none()
        {
            *slot = Some(err.to_string());
        }
    }
    errors
}

/// The trial an HPO job's headline numbers should come from, and the topo
/// positions of its nodes: the best COMPLETED trial, else (nothing completed
/// yet) the best trial that reported the objective at all. `None` when no
/// trial reported it. A job-level "final value" over every node would mix
/// trials — the max of a loss across trials, the last step of whichever trial
/// finished last — so per-run comparisons read this trial instead.
pub(crate) fn headline_trial(
    manifest: &HpoManifest,
    status_lines: &[String],
) -> Option<(u32, Vec<u32>)> {
    let board = leaderboard(manifest, status_lines);
    let best = board
        .iter()
        .find(|o| o.objective.is_some() && o.status == "done")
        .or_else(|| board.iter().find(|o| o.objective.is_some()))?;
    let position = manifest
        .trials
        .iter()
        .position(|t| t.trial_id == best.trial_id)? as u32;
    let nodes = manifest
        .trial_of_topo
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == Some(position))
        .map(|(idx, _)| idx as u32)
        .collect();
    Some((best.trial_id, nodes))
}

/// [`headline_trial`] for the HPO job `job_id`, read from its job directory.
/// `None` for a job that is not an HPO search.
pub(crate) fn job_headline_trial(job_id: &str) -> Option<(u32, Vec<u32>)> {
    let dir = crate::paths::jobs_dir().ok()?.join(job_id);
    if !dir.join("hpo.json").is_file() {
        return None;
    }
    let manifest = HpoManifest::read_from(&dir)?;
    let lines = crate::jobs::read_status_lines(job_id).ok()?;
    headline_trial(&manifest, &lines)
}

/// Leaderboard: outcomes sorted best-objective-first (per direction); trials
/// with no objective sink to the bottom.
pub fn leaderboard(manifest: &HpoManifest, status_lines: &[String]) -> Vec<TrialOutcome> {
    let maximize = manifest.maximize();
    let mut out = reconstruct(manifest, status_lines);
    out.sort_by(|a, b| match (a.objective, b.objective) {
        (Some(x), Some(y)) => {
            if maximize {
                y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest() -> HpoManifest {
        HpoManifest {
            recipe: "r".into(),
            algo: "asha".into(),
            metric: "val_r".into(),
            mode: "max".into(),
            budget_key: "epoch".into(),
            trials: vec![
                TrialRec {
                    trial_id: 0,
                    overlay: vec![("lr".into(), json!(0.1))],
                    n_nodes: 1,
                },
                TrialRec {
                    trial_id: 1,
                    overlay: vec![("lr".into(), json!(0.2))],
                    n_nodes: 1,
                },
            ],
            trial_of_topo: vec![Some(0), Some(1)],
        }
    }

    #[test]
    fn reconstruct_best_status_and_leaderboard() {
        let m = manifest();
        let lines: Vec<String> = [
            json!({"kind":"stage_begin","node_idx":0,"stage_name":"t","input_hash":"x"}),
            json!({"kind":"stage_step","node_idx":0,"stage_name":"t","update":{"val_r":0.3,"epoch":1}}),
            json!({"kind":"stage_step","node_idx":0,"stage_name":"t","update":{"val_r":0.7,"epoch":2}}),
            json!({"kind":"stage_end","node_idx":0,"stage_name":"t","output_hash":"h","elapsed":1}),
            json!({"kind":"stage_begin","node_idx":1,"stage_name":"t","input_hash":"x"}),
            json!({"kind":"stage_step","node_idx":1,"stage_name":"t","update":{"val_r":0.2,"epoch":1}}),
            json!({"kind":"stage_failed","node_idx":1,"stage_name":"t","error":crate::framework::executor::CONTROL_PRUNE_MESSAGE}),
        ]
        .iter()
        .map(|v| v.to_string())
        .collect();

        let board = leaderboard(&m, &lines);
        // trial0 best=0.7 done; trial1 best=0.2 stopped. Leaderboard: trial0 first.
        assert_eq!(board[0].trial_id, 0);
        assert_eq!(board[0].objective, Some(0.7));
        assert_eq!(board[0].status, "done");
        assert_eq!(board[1].trial_id, 1);
        assert_eq!(board[1].objective, Some(0.2));
        assert_eq!(board[1].status, "stopped");
    }

    #[test]
    fn cancel_is_cancelled_genuine_error_is_failed() {
        let m = manifest();
        let lines: Vec<String> = [
            // trial0: a plan cancel → cancelled.
            json!({"kind":"stage_begin","node_idx":0,"stage_name":"t","input_hash":"x"}),
            json!({"kind":"stage_failed","node_idx":0,"stage_name":"t","error":"cancelled during stage"}),
            // trial1: a genuine crash whose message contains "kill" (the Linux
            // OOM killer) → failed, NOT killed (the "kill" word must not be
            // treated as a scheduler kill).
            json!({"kind":"stage_begin","node_idx":1,"stage_name":"t","input_hash":"x"}),
            json!({"kind":"stage_failed","node_idx":1,"stage_name":"t","error":"Out of memory: Killed process 12345 (python)"}),
        ]
        .iter()
        .map(|v| v.to_string())
        .collect();
        let out = reconstruct(&m, &lines);
        assert_eq!(out[0].status, "cancelled");
        assert_eq!(
            out[1].status, "failed",
            "OOM-killer 'Killed process' is a crash, not a scheduler kill"
        );
    }

    #[test]
    fn control_pruned_nodes_complete_a_trial_without_marking_failure() {
        let mut m = manifest();
        m.trials.truncate(1);
        m.trials[0].n_nodes = 2;
        m.trial_of_topo = vec![Some(0), Some(0)];
        let lines = [
            json!({"kind":"stage_end","node_idx":0,"stage_name":"decision","output_hash":"h","elapsed":1}),
            json!({"kind":"stage_pruned","node_idx":1,"stage_name":"guarded","reason":"condition false"}),
        ]
        .iter()
        .map(|event| event.to_string())
        .collect::<Vec<_>>();

        let out = reconstruct(&m, &lines);
        assert_eq!(out[0].status, "done");
        assert_eq!(out[0].objective, None);
    }

    #[test]
    fn failure_and_cancel_override_terminal_control_accounting() {
        let m = manifest();
        let lines = [
            json!({"kind":"stage_pruned","node_idx":0,"stage_name":"a","reason":"condition false"}),
            json!({"kind":"stage_failed","node_idx":0,"stage_name":"a","error":crate::framework::executor::CONTROL_PRUNE_MESSAGE}),
            json!({"kind":"stage_pruned","node_idx":1,"stage_name":"b","reason":"condition false"}),
            json!({"kind":"stage_failed","node_idx":1,"stage_name":"b","error":"boom"}),
        ]
        .iter()
        .map(|event| event.to_string())
        .collect::<Vec<_>>();

        let out = reconstruct(&m, &lines);
        assert_eq!(out[0].status, "stopped");
        assert_eq!(out[1].status, "failed");
        assert_eq!(
            trial_errors(&m, &lines),
            vec![None, Some("boom".to_string())],
            "a prune is not an error; the crash keeps its message"
        );
    }

    #[test]
    fn min_mode_sorts_ascending_and_keeps_min() {
        let mut m = manifest();
        m.mode = "min".into();
        let lines: Vec<String> = [
            json!({"kind":"stage_step","node_idx":0,"stage_name":"t","update":{"val_r":0.9,"epoch":1}}),
            json!({"kind":"stage_step","node_idx":0,"stage_name":"t","update":{"val_r":0.4,"epoch":2}}),
            json!({"kind":"stage_step","node_idx":1,"stage_name":"t","update":{"val_r":0.8,"epoch":1}}),
        ]
        .iter()
        .map(|v| v.to_string())
        .collect();
        let board = leaderboard(&m, &lines);
        assert_eq!(board[0].objective, Some(0.4), "min mode keeps the minimum");
        assert_eq!(board[0].trial_id, 0);
    }

    #[test]
    fn the_headline_trial_is_the_best_completed_one_with_its_nodes() {
        let mut m = manifest();
        m.trial_of_topo = vec![Some(0), Some(1)];
        let lines: Vec<String> = [
            // trial 0 completes at 0.5; trial 1 reads 0.9 but is stopped early.
            json!({"kind":"stage_step","node_idx":0,"stage_name":"t","update":{"val_r":0.5}}),
            json!({"kind":"stage_end","node_idx":0,"stage_name":"t","output_hash":"h","elapsed":1}),
            json!({"kind":"stage_step","node_idx":1,"stage_name":"t","update":{"val_r":0.9}}),
            json!({"kind":"stage_failed","node_idx":1,"stage_name":"t","error":crate::framework::executor::CONTROL_PRUNE_MESSAGE}),
        ]
        .iter()
        .map(|v| v.to_string())
        .collect();
        assert_eq!(headline_trial(&m, &lines), Some((0, vec![0])));
        assert_eq!(headline_trial(&m, &[]), None);
    }
}
