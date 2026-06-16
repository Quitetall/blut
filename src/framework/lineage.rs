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
        let Ok(StageEvent::StageStep { node_idx, update, .. }) =
            serde_json::from_str::<StageEvent>(&line)
        else {
            continue;
        };
        let Some(obj) = update.as_object() else {
            continue;
        };
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
}
