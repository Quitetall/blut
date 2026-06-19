// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! DAG-graph backend (v0.20).
//!
//! A queryable snapshot of a job's plan graph — the structure (persisted at
//! launch as `<job_dir>/plan.json`) joined with the live node status folded from
//! the durable `status.jsonl` event stream. No daemon, no HTTP: `blut dag <job>
//! [--json]` reads it on demand, and a future TUI polls the same builder.
//!
//! Node indices are TOPO POSITIONS — identical to the `node_idx` the executor
//! stamps on every `StageEvent`, so the status fold attributes events to plan
//! nodes directly. For an HPO run the (optional) `hpo.json` manifest layers each
//! node's owning trial + its overlay on top.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The persisted plan STRUCTURE (`<job_dir>/plan.json`): nodes in topo order
/// (so `idx` == the executor's `node_idx`) + edges in those same topo indices.
/// Written once at launch; immutable for the life of the job.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanGraph {
    pub name: String,
    pub nodes: Vec<PlanGraphNode>,
    pub edges: Vec<PlanGraphEdge>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanGraphNode {
    /// Topo position == the executor's `node_idx`.
    pub idx: usize,
    pub stage_name: String,
    /// A compact one-line digest of the node's args (top-level scalars).
    pub args_summary: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanGraphEdge {
    pub from: usize,
    pub to: usize,
}

impl PlanGraph {
    pub fn write_to(&self, job_dir: &std::path::Path) -> std::io::Result<()> {
        let body = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = job_dir.join("plan.json.tmp");
        std::fs::write(&tmp, body)?;
        std::fs::rename(tmp, job_dir.join("plan.json"))
    }

    pub fn read_from(job_dir: &std::path::Path) -> Option<PlanGraph> {
        let path = job_dir.join("plan.json");
        let body = std::fs::read_to_string(&path).ok()?;
        match serde_json::from_str(&body) {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("warning: corrupt plan graph {}: {e}", path.display());
                None
            }
        }
    }
}

/// A compact one-line digest of a node's args: top-level scalar key=value pairs
/// (nested objects/arrays elided), capped so a row stays readable.
pub fn summarize_args(args: &Value) -> String {
    let Some(obj) = args.as_object() else {
        return String::new();
    };
    let mut parts: Vec<String> = Vec::new();
    for (k, v) in obj {
        let rendered = match v {
            Value::Object(_) | Value::Array(_) => continue,
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        parts.push(format!("{k}={rendered}"));
        if parts.len() >= 8 {
            parts.push("…".into());
            break;
        }
    }
    let s = parts.join(" ");
    if s.len() > 120 {
        format!(
            "{}…",
            &s[..s.char_indices().nth(120).map(|(i, _)| i).unwrap_or(s.len())]
        )
    } else {
        s
    }
}

/// A node's runtime status, derived from the status stream + graph topology.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NodeStatus {
    /// Has an upstream dependency that has not finished.
    Pending,
    /// All upstream deps finished; eligible to run but not started.
    Ready,
    /// Entered its run body (begin/step/retry observed).
    Running,
    /// Waiting on a resource semaphore (e.g. the GPU).
    Blocked,
    /// Produced output.
    Done,
    /// Terminal crash (OOM/code/timeout).
    Failed,
    /// Cache hit — output reused without running.
    Skipped,
    /// Stopped by the scheduler (HPO early-stop) or a plan cancel.
    Killed,
    /// Never ran because an upstream node was killed/failed.
    Pruned,
}

impl NodeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            NodeStatus::Pending => "pending",
            NodeStatus::Ready => "ready",
            NodeStatus::Running => "running",
            NodeStatus::Blocked => "blocked",
            NodeStatus::Done => "done",
            NodeStatus::Failed => "failed",
            NodeStatus::Skipped => "skipped",
            NodeStatus::Killed => "killed",
            NodeStatus::Pruned => "pruned",
        }
    }
    fn is_terminal_good(self) -> bool {
        matches!(self, NodeStatus::Done | NodeStatus::Skipped)
    }
    fn is_terminal_bad(self) -> bool {
        matches!(
            self,
            NodeStatus::Killed | NodeStatus::Failed | NodeStatus::Pruned
        )
    }
}

/// The HPO attribution for a node (only present for HPO runs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HpoNodeInfo {
    pub trial_id: u32,
    /// The trial's sampled overlay rendered as `key=value` pairs.
    pub overlay: Vec<(String, Value)>,
}

/// One node in a built snapshot: structure + derived status + observed metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphNode {
    pub idx: usize,
    pub stage_name: String,
    pub status: NodeStatus,
    pub args_summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_secs: Option<f64>,
    pub cache_hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hpo: Option<HpoNodeInfo>,
}

/// The full job graph snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub job: String,
    pub name: String,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<PlanGraphEdge>,
}

/// Per-node accumulator while folding the status stream.
#[derive(Default, Clone)]
struct Obs {
    status: Option<NodeStatus>,
    input_hash: Option<String>,
    output_hash: Option<String>,
    elapsed_secs: Option<f64>,
    cache_hit: bool,
}

fn dur_secs(v: &Value) -> Option<f64> {
    // serde's Duration → {"secs":N,"nanos":N}; tolerate a bare number too.
    if let Some(n) = v.as_f64() {
        return Some(n);
    }
    let secs = v.get("secs")?.as_f64()?;
    let nanos = v.get("nanos").and_then(|n| n.as_f64()).unwrap_or(0.0);
    Some(secs + nanos / 1e9)
}

/// Fold the raw status.jsonl lines into a per-node observation table. Each line
/// is a `StageEvent` JSON object keyed by topo `node_idx`. Last-write-wins for
/// the live status; terminal states are never downgraded by a later event.
fn fold_status(lines: &[String], n_nodes: usize) -> Vec<Obs> {
    let mut obs = vec![Obs::default(); n_nodes];
    for line in lines {
        let Ok(ev) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let kind = ev.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        let Some(idx) = ev.get("node_idx").and_then(|v| v.as_u64()) else {
            continue;
        };
        let idx = idx as usize;
        let Some(o) = obs.get_mut(idx) else {
            continue;
        };
        // A terminal state is final — a stray later event can't revive it.
        let terminal = o
            .status
            .map(|s| s.is_terminal_good() || s.is_terminal_bad())
            .unwrap_or(false);
        match kind {
            "stage_begin" => {
                if let Some(h) = ev.get("input_hash").and_then(|v| v.as_str()) {
                    o.input_hash = Some(h.to_string());
                }
                if !terminal {
                    o.status = Some(NodeStatus::Running);
                }
            }
            "stage_step" | "stage_retrying" => {
                if !terminal {
                    o.status = Some(NodeStatus::Running);
                }
            }
            "stage_blocked" => {
                if !terminal {
                    o.status = Some(NodeStatus::Blocked);
                }
            }
            // The terminal arms are ALSO `!terminal`-guarded → FIRST terminal
            // wins; a duplicate/late terminal event (log replay, rotation) can't
            // downgrade a node (e.g. a stray stage_skipped flipping Done→Skipped).
            "stage_end" if !terminal => {
                o.status = Some(NodeStatus::Done);
                if let Some(h) = ev.get("output_hash").and_then(|v| v.as_str()) {
                    o.output_hash = Some(h.to_string());
                }
                if let Some(d) = ev.get("elapsed").and_then(dur_secs) {
                    o.elapsed_secs = Some(d);
                }
            }
            "stage_skipped" if !terminal => {
                o.status = Some(NodeStatus::Skipped);
                o.cache_hit = true;
            }
            "stage_failed" if !terminal => {
                // Same split as the HPO leaderboard: a control-kill / plan-cancel
                // carries a "cancelled…" string; anything else is a real crash.
                let err = ev.get("error").and_then(|e| e.as_str()).unwrap_or("");
                o.status = Some(if err.to_ascii_lowercase().contains("cancel") {
                    NodeStatus::Killed
                } else {
                    NodeStatus::Failed
                });
            }
            _ => {}
        }
    }
    obs
}

/// Build the graph snapshot for `job_id`: read `plan.json` (structure),
/// `status.jsonl` (status), and `hpo.json` (optional trial attribution).
pub fn graph_snapshot(job_id: &str) -> Result<GraphSnapshot, String> {
    let job_dir = crate::paths::job_dir(job_id).map_err(|e| format!("job dir: {e}"))?;
    let plan = PlanGraph::read_from(&job_dir)
        .ok_or_else(|| format!("job '{job_id}' has no plan.json (pre-v0.20 run?)"))?;
    let n = plan.nodes.len();
    let lines = crate::jobs::read_status_lines(job_id).unwrap_or_default();
    let obs = fold_status(&lines, n);

    // Optional HPO attribution: trial-of-topo → overlay per node. Index the
    // trials by id ONCE (O(trials)) so the per-node lookup is O(1) — a linear
    // scan per node would be O(nodes × trials), a cliff for large sweeps.
    let hpo = crate::hpo::HpoManifest::read_from(&job_dir);
    let trial_index: std::collections::HashMap<u32, &crate::hpo::TrialRec> = hpo
        .as_ref()
        .map(|m| m.trials.iter().map(|r| (r.trial_id, r)).collect())
        .unwrap_or_default();
    let hpo_for = |idx: usize| -> Option<HpoNodeInfo> {
        let m = hpo.as_ref()?;
        let t = (*m.trial_of_topo.get(idx)?)?;
        let rec = trial_index.get(&t)?;
        Some(HpoNodeInfo {
            trial_id: t,
            overlay: rec.overlay.clone(),
        })
    };

    // Predecessor adjacency in topo indices (idx == topo position).
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for e in &plan.edges {
        if e.to < n && e.from < n {
            preds[e.to].push(e.from);
        }
    }

    // Derive a concrete status for every node. plan.nodes are in topo order
    // (ascending idx), so a node's predecessors are resolved before it — the
    // Ready/Pending/Pruned derivation can read already-decided predecessors.
    let mut status: Vec<NodeStatus> = vec![NodeStatus::Pending; n];
    for node in &plan.nodes {
        let i = node.idx;
        if i >= n {
            continue;
        }
        status[i] = match obs[i].status {
            Some(s) => s,
            None => {
                // Unobserved: a bad predecessor prunes it; all-good preds make it
                // Ready; otherwise it is still waiting (Pending). A root with no
                // preds and no events is Ready (it simply hasn't started).
                if preds[i].iter().any(|&p| status[p].is_terminal_bad()) {
                    NodeStatus::Pruned
                } else if preds[i].iter().all(|&p| status[p].is_terminal_good()) {
                    NodeStatus::Ready
                } else {
                    NodeStatus::Pending
                }
            }
        };
    }

    let nodes = plan
        .nodes
        .iter()
        .map(|pn| {
            let i = pn.idx;
            let o = obs.get(i).cloned().unwrap_or_default();
            GraphNode {
                idx: i,
                stage_name: pn.stage_name.clone(),
                status: status[i],
                args_summary: pn.args_summary.clone(),
                input_hash: o.input_hash,
                output_hash: o.output_hash,
                elapsed_secs: o.elapsed_secs,
                cache_hit: o.cache_hit,
                hpo: hpo_for(i),
            }
        })
        .collect();

    Ok(GraphSnapshot {
        job: job_id.to_string(),
        name: plan.name,
        nodes,
        edges: plan.edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn lines(evs: &[Value]) -> Vec<String> {
        evs.iter().map(|v| v.to_string()).collect()
    }

    #[test]
    fn fold_assigns_terminal_states() {
        let l = lines(&[
            json!({"kind":"stage_begin","node_idx":0,"stage_name":"a","input_hash":"aa"}),
            json!({"kind":"stage_end","node_idx":0,"stage_name":"a","output_hash":"bb","elapsed":{"secs":3,"nanos":500000000}}),
            json!({"kind":"stage_skipped","node_idx":1,"stage_name":"b","cache_key":"cc"}),
            json!({"kind":"stage_failed","node_idx":2,"stage_name":"c","error":"cancelled during stage"}),
            json!({"kind":"stage_failed","node_idx":3,"stage_name":"d","error":"Out of memory: Killed process 9"}),
        ]);
        let o = fold_status(&l, 4);
        assert_eq!(o[0].status, Some(NodeStatus::Done));
        assert_eq!(o[0].output_hash.as_deref(), Some("bb"));
        assert_eq!(o[0].elapsed_secs, Some(3.5));
        assert_eq!(o[1].status, Some(NodeStatus::Skipped));
        assert!(o[1].cache_hit);
        assert_eq!(o[2].status, Some(NodeStatus::Killed), "cancel → killed");
        assert_eq!(o[3].status, Some(NodeStatus::Failed), "OOM-killer → failed");
    }

    #[test]
    fn terminal_state_not_revived() {
        // A stray stage_step after stage_end must not flip Done back to Running.
        let l = lines(&[
            json!({"kind":"stage_end","node_idx":0,"stage_name":"a","output_hash":"x","elapsed":{"secs":1,"nanos":0}}),
            json!({"kind":"stage_step","node_idx":0,"stage_name":"a","update":{"loss":0.1}}),
        ]);
        let o = fold_status(&l, 1);
        assert_eq!(o[0].status, Some(NodeStatus::Done));
    }

    #[test]
    fn first_terminal_wins_over_later_terminal() {
        // A duplicate/late terminal event (log replay/rotation) must not
        // downgrade the node: FIRST terminal wins, metadata preserved.
        let l = lines(&[
            json!({"kind":"stage_end","node_idx":0,"stage_name":"a","output_hash":"good","elapsed":{"secs":2,"nanos":0}}),
            json!({"kind":"stage_skipped","node_idx":0,"stage_name":"a","cache_key":"zz"}),
            json!({"kind":"stage_failed","node_idx":0,"stage_name":"a","error":"boom"}),
        ]);
        let o = fold_status(&l, 1);
        assert_eq!(
            o[0].status,
            Some(NodeStatus::Done),
            "first terminal (Done) wins"
        );
        assert_eq!(o[0].output_hash.as_deref(), Some("good"));
        assert_eq!(o[0].elapsed_secs, Some(2.0));
        assert!(
            !o[0].cache_hit,
            "late stage_skipped must not flip cache_hit"
        );
    }

    #[test]
    fn summarize_args_compacts_scalars_only() {
        let s = summarize_args(&json!({"lr":0.1,"epochs":10,"nested":{"x":1},"tag":"run"}));
        assert!(s.contains("lr=0.1"));
        assert!(s.contains("epochs=10"));
        assert!(s.contains("tag=run"));
        assert!(!s.contains("nested"), "nested objects elided");
    }
}
