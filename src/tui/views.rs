// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Engine-data gatherers for the cockpit's analytic + provenance views.
//!
//! Every function here reads BLUT's **own** state — the jobs store, the
//! embedded lineage DB, the content-addressed cache, the plan graph, and the
//! recipe registry — and returns owned, render-ready rows. There is no
//! domain knowledge and no parallel data model: a view shows what the engine
//! actually recorded for a run, so it works for any cookbook.
//!
//! Best-effort, never a panic: a missing lineage DB, an unknown job, an
//! absent metric, or a job with no plan graph all degrade to an empty result
//! (the drawer then shows an empty-state line).
//!
//! View → source map:
//!   * History     → [`run_history`]   (jobs store ⋈ lineage `get_run`)
//!   * Leaderboard → [`leaderboard`]   (lineage `top_runs_by_metric`)
//!   * Compare     → [`compare`]       (lineage `final_metrics` + GPU sat)
//!   * Metrics     → [`metrics_for`]   (lineage `final_metrics`, one run)
//!   * Artifacts   → [`artifacts_for`] (cache sidecars via `scan_artifacts`)
//!   * Lineage     → [`lineage_for`]   (`job_lineage` + cache stats + freshness)
//!   * DAG         → [`dag_for`]       (`graph_snapshot`)
//!   * Reset       → [`reset`]         (generic cache/job/footprint maintenance)

use crate::framework::lineage;
use crate::framework::{GraphSnapshot, graph_snapshot};
use crate::jobs;
use crate::lineage_db::LineageDb;

/// The metric the leaderboard ranks by in 1.0. Every trainer records a
/// `loss`; lower is better (so the leaderboard minimizes). Bespoke metrics
/// are surfaced per-run in Compare / Metrics and via the `hpo show` CLI.
pub const DEFAULT_METRIC: &str = "loss";

/// Render a UNIX-seconds timestamp as `YYYY-MM-DD HH:MM` without pulling in
/// chrono (reuses the civil-from-days helper `jobs` uses for ids).
fn fmt_unix(secs: u64) -> String {
    let (y, m, d, h, mi, _s) = jobs::unix_to_ymdhms_pub(secs);
    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}")
}

/// A run's wall-clock label: its recorded start time if the lineage DB has
/// one, else the timestamp embedded in the job id (`YYYYMMDD-HHMMSS-…`).
fn when_of(job_id: &str, started_unix: Option<i64>) -> String {
    if let Some(s) = started_unix {
        if s > 0 {
            return fmt_unix(s as u64);
        }
    }
    fmt_job_id_date(job_id)
}

/// Parse the date+time a job id encodes: `YYYYMMDD-HHMMSS-NNNNNNNNN`.
/// Returns the id verbatim if it isn't in that shape.
fn fmt_job_id_date(id: &str) -> String {
    let mut parts = id.split('-');
    let (Some(d), Some(t)) = (parts.next(), parts.next()) else {
        return id.to_string();
    };
    // ASCII digit byte-slicing is char-boundary safe; the length checks gate it.
    if d.len() == 8 && t.len() == 6 && d.bytes().chain(t.bytes()).all(|b| b.is_ascii_digit()) {
        format!(
            "{}-{}-{} {}:{}",
            &d[0..4],
            &d[4..6],
            &d[6..8],
            &t[0..2],
            &t[2..4]
        )
    } else {
        id.to_string()
    }
}

/// First `n` chars of a hash, or `—` when absent. Keeps the analytic tables
/// narrow without losing the disambiguating prefix.
fn short_hash(h: &Option<String>, n: usize) -> String {
    h.as_deref()
        .map(|s| s.chars().take(n).collect())
        .unwrap_or_else(|| "—".into())
}

// ── History + Leaderboard ────────────────────────────────────────────────

/// One run row (History + Leaderboard).
#[derive(Clone, Debug)]
pub struct RunRow {
    pub job_id: String,
    pub recipe: String,
    /// Recorded outcome (lineage DB) or the job state, whichever is known.
    pub outcome: String,
    /// The active metric's final value for this run (`None` if unrecorded).
    pub metric: Option<f64>,
    pub when: String,
}

/// History: every job, newest first, joined with its lineage-DB run record
/// and the active metric's final value. Degrades to the bare jobs store when
/// the lineage DB is absent.
pub fn run_history(metric: &str) -> Vec<RunRow> {
    let db = LineageDb::open().ok();
    let mut rows: Vec<RunRow> = jobs::list_jobs()
        .unwrap_or_default()
        .into_iter()
        .map(|j| {
            let rec = db.as_ref().and_then(|d| d.get_run(&j.id).ok().flatten());
            let recipe = rec
                .as_ref()
                .map(|r| r.recipe.clone())
                .or_else(|| j.output_name.clone())
                .unwrap_or_else(|| "?".into());
            let outcome = rec
                .as_ref()
                .and_then(|r| r.outcome.clone())
                .unwrap_or_else(|| j.state.as_str().to_string());
            let metric_v = db
                .as_ref()
                .and_then(|d| d.final_metric(&j.id, metric).ok().flatten());
            let when = when_of(&j.id, rec.as_ref().and_then(|r| r.started_unix));
            RunRow {
                job_id: j.id,
                recipe,
                outcome,
                metric: metric_v,
                when,
            }
        })
        .collect();
    rows.sort_by(|a, b| b.job_id.cmp(&a.job_id)); // newest first
    rows
}

/// Leaderboard: runs ranked by the active metric via the lineage DB's
/// `top_runs_by_metric` (the final, `step=-1` value), each filled with its
/// run-record context. Empty when the DB has no rows for that metric.
pub fn leaderboard(metric: &str, maximize: bool) -> Vec<RunRow> {
    let Ok(db) = LineageDb::open() else {
        return Vec::new();
    };
    // Cap at 20 — the leaderboard is a top-N ranking, and 20 matches the
    // view's render limit so the list cursor can't run past the visible rows.
    db.top_runs_by_metric(metric, maximize, 20)
        .unwrap_or_default()
        .into_iter()
        .map(|(job_id, _node_idx, value)| {
            let rec = db.get_run(&job_id).ok().flatten();
            let recipe = rec
                .as_ref()
                .map(|r| r.recipe.clone())
                .unwrap_or_else(|| "?".into());
            let outcome = rec
                .as_ref()
                .and_then(|r| r.outcome.clone())
                .unwrap_or_default();
            let when = when_of(&job_id, rec.as_ref().and_then(|r| r.started_unix));
            RunRow {
                job_id,
                recipe,
                outcome,
                metric: Some(value),
                when,
            }
        })
        .collect()
}

// ── Compare + Metrics ────────────────────────────────────────────────────

/// One column of the Compare view: a run and all its final metrics.
#[derive(Clone, Debug)]
pub struct CompareCol {
    pub job_id: String,
    pub recipe: String,
    /// All final (`step=-1`) metrics for the run, `name → value`.
    pub metrics: Vec<(String, f64)>,
    /// Mean GPU saturation (%) if the run recorded gauge samples.
    pub gpu: Option<f64>,
}

/// Compare: gather each marked run's final metric set + GPU saturation.
pub fn compare(job_ids: &[String]) -> Vec<CompareCol> {
    let db = LineageDb::open().ok();
    job_ids
        .iter()
        .map(|id| {
            let metrics = db
                .as_ref()
                .and_then(|d| d.final_metrics(id).ok())
                .unwrap_or_default();
            let recipe = db
                .as_ref()
                .and_then(|d| d.get_run(id).ok().flatten())
                .map(|r| r.recipe)
                .unwrap_or_else(|| "?".into());
            let gpu = db
                .as_ref()
                .and_then(|d| d.gpu_saturation(id, 50.0).ok().flatten())
                .map(|g| g.saturation);
            CompareCol {
                job_id: id.clone(),
                recipe,
                metrics,
                gpu,
            }
        })
        .collect()
}

/// Metrics: one run's final metric set (`name → value`), sorted by name.
pub fn metrics_for(job_id: &str) -> Vec<(String, f64)> {
    let mut m = LineageDb::open()
        .ok()
        .and_then(|d| d.final_metrics(job_id).ok())
        .unwrap_or_default();
    m.sort_by(|a, b| a.0.cmp(&b.0));
    m
}

// ── Artifacts ────────────────────────────────────────────────────────────

/// One materialized artifact (a stage output) of a run.
#[derive(Clone, Debug)]
pub struct ArtifactRow {
    pub kind: String,
    pub stage: String,
    pub hash: String,
    pub when: String,
}

/// Artifacts: the content-addressed outputs a run materialized, read from
/// its `stages/*/output.metadata.json` sidecars (newest first).
pub fn artifacts_for(job_id: &str) -> Vec<ArtifactRow> {
    let mut rows: Vec<(u64, ArtifactRow)> = lineage::scan_artifacts(job_id)
        .unwrap_or_default()
        .into_iter()
        .map(|a| {
            let ts = a.meta.produced_at_unix_secs;
            (
                ts,
                ArtifactRow {
                    kind: a.meta.kind,
                    stage: a.meta.produced_by_stage.unwrap_or_else(|| "—".into()),
                    hash: a.meta.content_hash.to_hex().chars().take(12).collect(),
                    when: fmt_unix(ts),
                },
            )
        })
        .collect();
    rows.sort_by_key(|(ts, _)| std::cmp::Reverse(*ts));
    rows.into_iter().map(|(_, r)| r).collect()
}

// ── Lineage ──────────────────────────────────────────────────────────────

/// One stage's lineage hop (input → output content hashes + cache + timing).
#[derive(Clone, Debug)]
pub struct LineageRow {
    pub node_idx: u32,
    pub stage: String,
    pub input: String,
    pub output: String,
    pub cached: bool,
    pub elapsed: String,
}

/// A run's full provenance: per-stage lineage, cache hit/miss totals, and the
/// code-freshness verdict (did the code that built it drift from HEAD?).
#[derive(Clone, Debug, Default)]
pub struct LineageView {
    pub rows: Vec<LineageRow>,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub freshness: String,
}

/// Lineage: replay a run's stage events into ordered lineage rows, tally its
/// cache hits/misses, and read its code-freshness verdict.
pub fn lineage_for(job_id: &str) -> LineageView {
    let rows = lineage::job_lineage(job_id)
        .unwrap_or_default()
        .into_iter()
        .map(|n| LineageRow {
            node_idx: n.node_idx,
            stage: n.stage,
            input: short_hash(&n.input_hash, 10),
            output: short_hash(&n.output_hash, 10),
            cached: n.cached,
            elapsed: n
                .elapsed
                .map(|d| format!("{:.1}s", d.as_secs_f64()))
                .unwrap_or_else(|| "—".into()),
        })
        .collect();
    let (cache_hits, cache_misses) = lineage::cache_stats(job_id)
        .map(|c| c.totals())
        .unwrap_or((0, 0));
    let freshness = LineageDb::open()
        .ok()
        .and_then(|d| d.code_freshness(job_id).ok())
        .map(|f| f.tag().to_string())
        .unwrap_or_else(|| "UNKNOWN".into());
    LineageView {
        rows,
        cache_hits,
        cache_misses,
        freshness,
    }
}

// ── DAG ──────────────────────────────────────────────────────────────────

/// DAG: the plan/job graph snapshot (nodes with derived status + edges).
/// `None` when the run wrote no `plan.json` (a legacy bare-spawn job).
pub fn dag_for(job_id: &str) -> Option<GraphSnapshot> {
    graph_snapshot(job_id).ok()
}

// ── Reset (generic maintenance) ──────────────────────────────────────────

/// One destructive maintenance action. All are domain-agnostic and operate
/// on the engine's own state (the global cache, a job dir, the footprint
/// calibration store) — no assumptions about session names or repo layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResetAction {
    /// LRU-prune the content-addressed cache to its cap.
    PruneCache,
    /// Remove the currently-selected job's directory (spec/status/stages).
    ClearJob,
    /// Forget all footprint calibrations (they re-measure on next run).
    ForgetFootprints,
}

impl ResetAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::PruneCache => "Prune content-addressed cache (LRU, to cap)",
            Self::ClearJob => "Delete the selected job's directory",
            Self::ForgetFootprints => "Forget all footprint calibrations",
        }
    }
}

/// Cache cap in bytes: `$LAMU_CACHE_MAX_GB` GiB, else 50 GiB (matches the
/// `blut cache prune` default).
fn cache_cap_bytes() -> u64 {
    let gib = std::env::var("LAMU_CACHE_MAX_GB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(50);
    gib.saturating_mul(1024 * 1024 * 1024)
}

/// Run a confirmed reset action. `current_job` is the job id the cockpit has
/// selected (required for [`ResetAction::ClearJob`]). Returns a result line.
pub fn reset(action: ResetAction, current_job: Option<&str>) -> String {
    match action {
        ResetAction::PruneCache => match crate::framework::CacheHandle::default_global_path() {
            Some(root) => match crate::framework::cache::lru_prune(&root, cache_cap_bytes()) {
                Ok(freed) => format!(
                    "pruned cache to cap; freed {:.2} GiB",
                    freed as f64 / (1024.0 * 1024.0 * 1024.0)
                ),
                Err(e) => format!("cache prune failed: {e}"),
            },
            None => "no global cache configured; nothing to prune".into(),
        },
        ResetAction::ClearJob => {
            let Some(id) = current_job else {
                return "no job selected; nothing to clear".into();
            };
            match jobs::job_dir_path(id) {
                Ok(dir) if dir.exists() => match std::fs::remove_dir_all(&dir) {
                    Ok(()) => format!("deleted job dir {id}"),
                    Err(e) => format!("failed to delete job {id}: {e}"),
                },
                Ok(_) => format!("job dir for {id} not found; nothing to clear"),
                Err(e) => format!("could not resolve job {id}: {e}"),
            }
        }
        ResetAction::ForgetFootprints => {
            let mut store = crate::broker::FootprintStore::load();
            let keys: Vec<String> = store
                .entries_snapshot()
                .into_iter()
                .map(|(k, _)| k)
                .collect();
            let mut forgotten = 0usize;
            for k in &keys {
                if store.forget(k).unwrap_or(false) {
                    forgotten += 1;
                }
            }
            format!("forgot {forgotten} footprint calibration(s)")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_job_id_date_parses_canonical_id() {
        assert_eq!(fmt_job_id_date("20260618-073012-000000001"), "2026-06-18 07:30");
    }

    #[test]
    fn fmt_job_id_date_passes_through_non_canonical() {
        assert_eq!(fmt_job_id_date("not-an-id"), "not-an-id");
        assert_eq!(fmt_job_id_date("weird"), "weird");
    }

    #[test]
    fn short_hash_truncates_and_handles_none() {
        assert_eq!(short_hash(&Some("abcdef0123456789".into()), 10), "abcdef0123");
        assert_eq!(short_hash(&None, 10), "—");
    }

    #[test]
    fn when_of_prefers_recorded_start() {
        // started_unix wins over the id-derived time when present + positive.
        let s = when_of("20260101-000000-000000000", Some(1_700_000_000));
        assert!(s.starts_with("2023-"), "got {s}");
        // falls back to the id when no start recorded.
        let s2 = when_of("20260618-073012-000000001", None);
        assert_eq!(s2, "2026-06-18 07:30");
    }

    #[test]
    fn cache_cap_defaults_to_50_gib() {
        // No env override → 50 GiB. (Env-independent: we don't set the var.)
        if std::env::var("LAMU_CACHE_MAX_GB").is_err() {
            assert_eq!(cache_cap_bytes(), 50 * 1024 * 1024 * 1024);
        }
    }

    #[test]
    fn reset_clear_job_without_selection_is_noop() {
        let msg = reset(ResetAction::ClearJob, None);
        assert!(msg.contains("no job selected"), "msg={msg}");
    }
}
