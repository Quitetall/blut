// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Lineage + cache observability (E3).
//!
//! The executor already records everything needed to answer "what ran,
//! what did it produce, and what was served from cache" — per-stage
//! `output.metadata.json` sidecars (kind, schema, content_hash,
//! producing stage) and the `status.jsonl` `StageEvent` stream
//! (Begin/End/Skipped with input/output hashes + elapsed). This module
//! surfaces it; the CLI (`blut lineage` / `blut cache stats` /
//! `blut artifact`) renders it. Read-only, no new on-disk state.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::error::Result;
use crate::framework::artifact::ArtifactMetadata;
use crate::framework::status::StageEvent;
use crate::jobs;

/// One stage's lineage row, folded from its Begin/End/Skipped events.
#[derive(Clone, Debug, serde::Serialize)]
pub struct LineageNode {
    pub node_idx: u32,
    pub stage: String,
    pub input_hash: Option<String>,
    pub output_hash: Option<String>,
    /// `true` if the stage was served from cache (a `StageSkipped`).
    pub cached: bool,
    pub elapsed: Option<Duration>,
}

/// Replay a job's `status.jsonl` `StageEvent`s into a per-node lineage
/// (ordered by `node_idx`). Tolerant: non-`StageEvent` lines (a legacy
/// bare-spawn job's `StatusUpdate` stream) are skipped.
pub fn job_lineage(job_id: &str) -> Result<Vec<LineageNode>> {
    let id = jobs::resolve_job_id(job_id)?;
    let mut by_idx: BTreeMap<u32, LineageNode> = BTreeMap::new();
    for line in jobs::read_status_lines(&id)? {
        let Ok(ev) = serde_json::from_str::<StageEvent>(&line) else {
            continue;
        };
        match ev {
            StageEvent::StageBegin {
                node_idx,
                stage_name,
                input_hash,
            } => {
                ensure_node(&mut by_idx, node_idx, &stage_name).input_hash =
                    Some(input_hash.to_hex());
            }
            StageEvent::StageEnd {
                node_idx,
                stage_name,
                output_hash,
                elapsed,
            } => {
                let n = ensure_node(&mut by_idx, node_idx, &stage_name);
                n.output_hash = Some(output_hash.to_hex());
                n.elapsed = Some(elapsed);
            }
            StageEvent::StageSkipped {
                node_idx,
                stage_name,
                cache_key,
            } => {
                let n = ensure_node(&mut by_idx, node_idx, &stage_name);
                n.cached = true;
                n.output_hash.get_or_insert_with(|| cache_key.to_hex());
            }
            _ => {}
        }
    }
    Ok(by_idx.into_values().collect())
}

/// One job's TERMINAL failure — replayed from `status.jsonl`'s
/// `StageFailed` events, the same source [`job_lineage`] replays. A stage
/// that fails and still has retries left emits `StageRetrying`, not
/// `StageFailed` (executor.rs); only an exhausted-retries or non-retried
/// failure reaches this stream, and it halts the plan — so the LAST
/// `StageFailed` line in the file is the terminal one. `blut errors show`
/// (ADR 0072 A4) is the sole consumer.
#[derive(Clone, Debug, serde::Serialize)]
pub struct JobFailure {
    pub node_idx: u32,
    pub stage: String,
    /// The `Display` form of the `StageError` (always present).
    pub error: String,
    /// Structured origin/course/recipe/ingredient breakdown, when the
    /// failing stage's error chain carried a `StageFailure`. `None` for a
    /// legacy status.jsonl predating ADR 0072, or a `StageError` variant
    /// that never wraps one — the raw `error` string is still shown.
    pub failure: Option<crate::framework::error_domain::FailureSummary>,
}

/// `None` when the job has no `StageFailed` event at all — it succeeded,
/// is still running, or hasn't started. Never assumes a failure exists.
pub fn job_failure(job_id: &str) -> Result<Option<JobFailure>> {
    let id = jobs::resolve_job_id(job_id)?;
    let mut last: Option<JobFailure> = None;
    for line in jobs::read_status_lines(&id)? {
        let Ok(ev) = serde_json::from_str::<StageEvent>(&line) else {
            continue;
        };
        if let StageEvent::StageFailed {
            node_idx,
            stage_name,
            error,
            failure,
        } = ev
        {
            last = Some(JobFailure {
                node_idx,
                stage: stage_name,
                error,
                failure,
            });
        }
    }
    Ok(last)
}

/// Fold a job's `status.jsonl` `StageStep` events into metric rows for the
/// queryable metric store (E1) — the sibling of [`job_lineage`], NO new writer.
/// Each finite numeric leaf of a step's `update` payload (`val_r`, `train_loss`,
/// `lr`, `grad_norm`, …) becomes a [`MetricRow`] at the step's coordinate
/// (`step`/`epoch`, else a per-node counter); the coordinate keys themselves are
/// not recorded as metrics. The LATEST value per (node, metric) is also emitted
/// at `step = -1` (the run's headline, what `final_metric` reads).
pub fn fold_metrics(job_id: &str) -> Result<Vec<crate::lineage_db::MetricRow>> {
    use crate::lineage_db::MetricRow;
    let id = jobs::resolve_job_id(job_id)?;
    let mut rows: Vec<MetricRow> = Vec::new();
    let mut last: BTreeMap<(u32, String), f64> = BTreeMap::new();
    let mut counter: BTreeMap<u32, i64> = BTreeMap::new();
    for line in jobs::read_status_lines(&id)? {
        let Ok(StageEvent::StageStep {
            node_idx, update, ..
        }) = serde_json::from_str::<StageEvent>(&line)
        else {
            continue;
        };
        let Some(obj) = update.as_object() else {
            continue;
        };
        // E2: gauge / sentinel events ride the SAME StageStep channel but
        // are NOT training metrics — route them out so their numeric
        // fields (gpu_util, …) don't pollute the metrics table. They are
        // folded separately by `fold_gauges`.
        if let Some(kind) = obj.get("kind").and_then(|k| k.as_str()) {
            if kind == "gpu_gauge" || kind == "gpu_starved" {
                continue;
            }
        }
        let step = obj
            .get("step")
            .and_then(|v| v.as_i64())
            .or_else(|| obj.get("epoch").and_then(|v| v.as_i64()))
            .unwrap_or_else(|| {
                let c = counter.entry(node_idx).or_insert(0);
                *c += 1;
                *c
            });
        for (k, v) in obj {
            if k == "step" || k == "epoch" {
                continue; // a coordinate, not a metric
            }
            if let Some(x) = v.as_f64() {
                if x.is_finite() {
                    rows.push(MetricRow {
                        job_id: id.clone(),
                        node_idx: node_idx as i64,
                        step,
                        metric: k.clone(),
                        value: x,
                        wall_unix: None,
                    });
                    last.insert((node_idx, k.clone()), x);
                }
            }
        }
    }
    for ((node_idx, metric), value) in last {
        rows.push(MetricRow {
            job_id: id.clone(),
            node_idx: node_idx as i64,
            step: -1,
            metric,
            value,
            wall_unix: None,
        });
    }
    Ok(rows)
}

/// Fold a job's `gpu_gauge` StageStep samples (E2) out of `status.jsonl`
/// into [`GaugeRow`](crate::lineage_db::GaugeRow)s for the `gauges`
/// table. Sibling of [`fold_metrics`]: the live GPU sampler
/// (`gpu_sampler`) emits one `StageStep{kind:"gpu_gauge", wall_unix,
/// gpu_util, gpu_mem_mib, gpu_temp_c, gpu_power_w}` per sample; this
/// reconstructs them so `gpu_saturation` / `gpu_wasted` can be derived
/// from the queryable store (the lineage DB stays rebuildable from
/// status.jsonl, never a second source of truth).
///
/// `gpu_starved` sentinels are skipped here — they carry no time-series
/// gauge value (they are a Log-pane signal, surfaced from status.jsonl
/// directly). A sample missing `wall_unix` is dropped (the table keys on
/// it).
pub fn fold_gauges(job_id: &str) -> Result<Vec<crate::lineage_db::GaugeRow>> {
    use crate::lineage_db::GaugeRow;
    let id = jobs::resolve_job_id(job_id)?;
    let mut rows: Vec<GaugeRow> = Vec::new();
    for line in jobs::read_status_lines(&id)? {
        let Ok(StageEvent::StageStep {
            node_idx, update, ..
        }) = serde_json::from_str::<StageEvent>(&line)
        else {
            continue;
        };
        let Some(obj) = update.as_object() else {
            continue;
        };
        if obj.get("kind").and_then(|k| k.as_str()) != Some("gpu_gauge") {
            continue;
        }
        let Some(wall_unix) = obj.get("wall_unix").and_then(|v| v.as_i64()) else {
            continue;
        };
        let f = |k: &str| obj.get(k).and_then(|v| v.as_f64());
        rows.push(GaugeRow {
            job_id: id.clone(),
            node_idx: node_idx as i64,
            wall_unix,
            gpu_util: f("gpu_util"),
            gpu_mem_mib: f("gpu_mem_mib"),
            gpu_temp_c: f("gpu_temp_c"),
            gpu_power_w: f("gpu_power_w"),
            host_ram_mib: f("host_ram_mib"),
            host_disk_free_mib: f("host_disk_free_mib"),
        });
    }
    Ok(rows)
}

fn ensure_node<'a>(
    m: &'a mut BTreeMap<u32, LineageNode>,
    idx: u32,
    name: &str,
) -> &'a mut LineageNode {
    m.entry(idx).or_insert_with(|| LineageNode {
        node_idx: idx,
        stage: name.to_string(),
        input_hash: None,
        output_hash: None,
        cached: false,
        elapsed: None,
    })
}

/// Per-stage cache hit/miss tallies for a job (a `StageSkipped` is a
/// hit; a `StageEnd` is a miss-that-ran).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct CacheStats {
    /// stage name → (hits, misses).
    pub per_stage: BTreeMap<String, (u64, u64)>,
}

impl CacheStats {
    pub fn totals(&self) -> (u64, u64) {
        self.per_stage
            .values()
            .fold((0, 0), |(h, m), (sh, sm)| (h + sh, m + sm))
    }
}

pub fn cache_stats(job_id: &str) -> Result<CacheStats> {
    let id = jobs::resolve_job_id(job_id)?;
    let mut stats = CacheStats::default();
    for line in jobs::read_status_lines(&id)? {
        let Ok(ev) = serde_json::from_str::<StageEvent>(&line) else {
            continue;
        };
        match ev {
            StageEvent::StageSkipped { stage_name, .. } => {
                stats.per_stage.entry(stage_name).or_default().0 += 1;
            }
            StageEvent::StageEnd { stage_name, .. } => {
                stats.per_stage.entry(stage_name).or_default().1 += 1;
            }
            _ => {}
        }
    }
    Ok(stats)
}

/// A materialized artifact located on disk via its sidecar.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ArtifactRecord {
    pub meta: ArtifactMetadata,
    pub sidecar_path: PathBuf,
    pub job_id: String,
}

/// Scan one job's `stages/*/output.metadata.json` sidecars.
pub fn scan_artifacts(job_id: &str) -> Result<Vec<ArtifactRecord>> {
    let id = jobs::resolve_job_id(job_id)?;
    let stages = jobs::job_dir_path(&id)?.join("stages");
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&stages) {
        for e in rd.flatten() {
            let sidecar = e.path().join("output.metadata.json");
            if let Ok(meta) = ArtifactMetadata::read_from(&sidecar) {
                out.push(ArtifactRecord {
                    meta,
                    sidecar_path: sidecar,
                    job_id: id.clone(),
                });
            }
        }
    }
    // Stable order by stage dir name (idx-name).
    out.sort_by(|a, b| a.sidecar_path.cmp(&b.sidecar_path));
    Ok(out)
}

/// Find artifacts whose content hash starts with `prefix` across ALL
/// jobs (single-box scale — a directory scan is cheap).
pub fn find_by_hash_prefix(prefix: &str) -> Result<Vec<ArtifactRecord>> {
    // `to_hex()` is lowercase; normalize the query so an uppercase prefix
    // still matches.
    let prefix = prefix.to_lowercase();
    let jobs_root = crate::paths::jobs_dir()?;
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&jobs_root) {
        for e in rd.flatten() {
            let Some(job_id) = e.file_name().to_str().map(String::from) else {
                continue;
            };
            for rec in scan_artifacts(&job_id).unwrap_or_default() {
                if rec.meta.content_hash.to_hex().starts_with(&prefix) {
                    out.push(rec);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_stats_tallies_hits_and_misses() {
        let mut s = CacheStats::default();
        s.per_stage.insert("a".into(), (2, 1));
        s.per_stage.insert("b".into(), (0, 3));
        assert_eq!(s.totals(), (2, 4));
    }

    /// E2 round-trip: a `status.jsonl` mixing a real training metric, two
    /// `gpu_gauge` samples, and a `gpu_starved` sentinel must fold so the
    /// gauges land in `fold_gauges` (sentinel skipped) and NONE of the
    /// gauge/sentinel fields leak into `fold_metrics` (the guard), while
    /// the genuine training metric still does.
    #[test]
    fn fold_gauges_routes_gpu_samples_and_metrics_guard_excludes_them() {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
        unsafe {
            std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path());
        }

        let job = "20260616-000000-e2gauge";
        let jdir = td.path().join(job);
        std::fs::create_dir_all(&jdir).unwrap();
        // Outer `"kind":"stage_step"` is the StageEvent tag; the INNER
        // `update.kind` ("gpu_gauge"/"gpu_starved") is the E2 routing key.
        let lines = [
            r#"{"kind":"stage_step","node_idx":1,"stage_name":"train","update":{"step":10,"val_r":0.42}}"#,
            r#"{"kind":"stage_step","node_idx":1,"stage_name":"train","update":{"kind":"gpu_gauge","wall_unix":1000,"gpu_util":95.0,"gpu_mem_mib":18000.0,"gpu_temp_c":70.0,"gpu_power_w":300.0}}"#,
            r#"{"kind":"stage_step","node_idx":1,"stage_name":"train","update":{"kind":"gpu_gauge","wall_unix":1002,"gpu_util":12.0,"gpu_mem_mib":17000.0,"gpu_temp_c":65.0,"gpu_power_w":120.0}}"#,
            r#"{"kind":"stage_step","node_idx":1,"stage_name":"train","update":{"kind":"gpu_starved","wall_unix":1010,"gpu_util":5.0,"samples_below":5,"floor_pct":25.0}}"#,
        ];
        std::fs::write(jdir.join("status.jsonl"), lines.join("\n") + "\n").unwrap();

        let gauges = fold_gauges(job).unwrap();
        assert_eq!(
            gauges.len(),
            2,
            "two gpu_gauge samples (starved sentinel skipped)"
        );
        assert_eq!(gauges[0].wall_unix, 1000);
        assert_eq!(gauges[0].gpu_util, Some(95.0));
        assert_eq!(gauges[0].gpu_mem_mib, Some(18000.0));
        assert_eq!(gauges[1].gpu_util, Some(12.0));

        let metrics = fold_metrics(job).unwrap();
        assert!(
            metrics.iter().all(|m| m.metric != "gpu_util"
                && m.metric != "floor_pct"
                && m.metric != "samples_below"),
            "gauge/sentinel fields must NOT pollute the metrics table"
        );
        assert!(
            metrics.iter().any(|m| m.metric == "val_r"),
            "the real training metric still folds"
        );

        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
    }

    /// `blut errors show` (ADR 0072 A4) reads exactly this: a terminal
    /// `StageFailed` carrying a full `FailureSummary` must surface all 5
    /// breakdown fields (origin/course/recipe/stage/ingredient), and a
    /// preceding `StageRetrying` on the same node must NOT be mistaken
    /// for the terminal failure.
    #[test]
    fn job_failure_surfaces_full_structured_breakdown() {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
        unsafe {
            std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path());
        }

        let job = "20260702-000000-errshow";
        let jdir = td.path().join(job);
        std::fs::create_dir_all(&jdir).unwrap();
        let lines = [
            r#"{"kind":"stage_retrying","node_idx":2,"stage_name":"train_joint","attempt":1,"max_attempts":3,"error":"transient OOM","backoff_ms":500}"#,
            r#"{"kind":"stage_failed","node_idx":2,"stage_name":"train_joint","error":"stage failed: OOM killed at epoch 3","failure":{"code":"E_OOM","domain":"lamquant","stage":"train_joint","severity":"critical","origin":"external","course":"train","recipe":"train_joint","ingredient":"trainer","context":[["ram_gib","64"]],"message":"OOM killed at epoch 3"}}"#,
        ];
        std::fs::write(jdir.join("status.jsonl"), lines.join("\n") + "\n").unwrap();

        let jf = job_failure(job).unwrap().expect("a StageFailed event exists");
        assert_eq!(jf.node_idx, 2);
        assert_eq!(jf.stage, "train_joint");
        assert!(jf.error.contains("OOM killed"));
        let f = jf.failure.expect("a structured FailureSummary was attached");
        assert_eq!(f.origin, crate::framework::error_domain::FaultOrigin::External);
        assert_eq!(f.course.as_deref(), Some("train"));
        assert_eq!(f.recipe.as_deref(), Some("train_joint"));
        assert_eq!(f.stage.as_deref(), Some("train_joint"));
        assert_eq!(f.ingredient.as_deref(), Some("trainer"));
        assert_eq!(f.code, "E_OOM");
        assert_eq!(f.domain, "lamquant");
        assert_eq!(
            f.severity,
            crate::framework::error_domain::Severity::Critical
        );

        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
    }

    /// A job that succeeded (or hasn't run yet) has no `StageFailed`
    /// event at all — `job_failure` must return `None`, never panic or
    /// synthesize a failure. `blut errors show` reads this as "no
    /// failure recorded".
    #[test]
    fn job_failure_none_when_no_failed_event_recorded() {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
        unsafe {
            std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path());
        }

        let job = "20260702-000001-errshowok";
        let jdir = td.path().join(job);
        std::fs::create_dir_all(&jdir).unwrap();
        let zeros = "0".repeat(64);
        let ones = "1".repeat(64);
        let lines = [
            format!(
                r#"{{"kind":"stage_begin","node_idx":0,"stage_name":"prep","input_hash":"{zeros}"}}"#
            ),
            format!(
                r#"{{"kind":"stage_end","node_idx":0,"stage_name":"prep","output_hash":"{ones}","elapsed":{{"secs":1,"nanos":0}}}}"#
            ),
        ];
        std::fs::write(jdir.join("status.jsonl"), lines.join("\n") + "\n").unwrap();

        assert!(job_failure(job).unwrap().is_none());

        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
    }
}
