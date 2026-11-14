// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! A machine-readable summary of one `recipe run` / `recipe declare --run`,
//! written to the file named by `BLUT_RUN_SUMMARY` when the run ends.
//!
//! The consumer this exists for is a supervisor that cannot read the job
//! directory: the Kubernetes operator points the variable at the container's
//! termination-message file, and the kubelet hands the summary back in the
//! pod status. The summary is one JSON object, at most
//! [`RUN_SUMMARY_MAX_BYTES`] bytes (the kubelet's own limit), and it is written
//! for failures that happen before any stage starts as well as after.
//!
//! ```json
//! {"version":1,"state":"succeeded","job":"20261023-…","stagesTotal":3,
//!  "cacheHits":0,"cacheMisses":3,"stages":{"split_train_eval":"ok"}}
//! ```
//!
//! Stage states: `running`, `ok`, `cached`, `failed`, `pruned`, `blocked`,
//! `retrying`. Without the variable nothing is written and nothing changes.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};

/// Environment variable naming the summary file.
pub(super) const RUN_SUMMARY_ENV: &str = "BLUT_RUN_SUMMARY";

/// Upper bound on the summary, in bytes. The kubelet truncates a termination
/// message beyond 4096 bytes; a truncated JSON object would not parse.
pub(super) const RUN_SUMMARY_MAX_BYTES: usize = 4096;

/// Set once a summary is written, so the top-level failure path does not
/// overwrite the more specific one `execute` already wrote.
static WRITTEN: AtomicBool = AtomicBool::new(false);

/// The outcome of a run, as the summary reports it.
pub(super) enum RunOutcome<'a> {
    Succeeded {
        stages_total: usize,
        cache_hits: usize,
        cache_misses: usize,
        warnings: usize,
    },
    Failed {
        error: &'a anyhow::Error,
    },
}

/// Write the summary for a run whose job directory is known.
pub(super) fn write_run_summary(job_id: &str, job_dir: &Path, outcome: RunOutcome<'_>) {
    let Some(target) = summary_target() else {
        return;
    };
    let stages = fold_stage_states(&job_dir.join("status.jsonl"));
    let summary = render_summary(Some(job_id), &stages, &outcome);
    persist(&target, &summary);
}

/// Write a failure summary unless one was already written for this process.
/// Covers errors that happen before a job directory exists (a recipe that does
/// not compile, a remote cache that cannot be reached).
pub(super) fn write_failed_run_summary_once(error: &anyhow::Error) {
    if WRITTEN.load(Ordering::SeqCst) {
        return;
    }
    let Some(target) = summary_target() else {
        return;
    };
    let summary = render_summary(None, &BTreeMap::new(), &RunOutcome::Failed { error });
    persist(&target, &summary);
}

fn summary_target() -> Option<std::path::PathBuf> {
    let raw = std::env::var_os(RUN_SUMMARY_ENV)?;
    if raw.is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(raw))
}

fn persist(target: &Path, summary: &str) {
    match std::fs::write(target, summary.as_bytes()) {
        Ok(()) => WRITTEN.store(true, Ordering::SeqCst),
        Err(error) => tracing::warn!(
            "{RUN_SUMMARY_ENV}: could not write {}: {error}",
            target.display()
        ),
    }
}

/// Last state of every stage in a `status.jsonl`, keyed by stage name, or by
/// `name#node` when one stage name appears at more than one node. Unreadable
/// files and unknown event kinds contribute nothing.
pub(super) fn fold_stage_states(status_jsonl: &Path) -> BTreeMap<String, String> {
    let Ok(text) = std::fs::read_to_string(status_jsonl) else {
        return BTreeMap::new();
    };
    let mut by_node: BTreeMap<u64, (String, &'static str)> = BTreeMap::new();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let state = match value.get("kind").and_then(Value::as_str) {
            Some("stage_begin") => "running",
            Some("stage_end") => "ok",
            Some("stage_skipped") => "cached",
            Some("stage_failed") => "failed",
            Some("stage_pruned") => "pruned",
            Some("stage_blocked") => "blocked",
            Some("stage_retrying") => "retrying",
            _ => continue,
        };
        let (Some(node), Some(name)) = (
            value.get("node_idx").and_then(Value::as_u64),
            value.get("stage_name").and_then(Value::as_str),
        ) else {
            continue;
        };
        by_node.insert(node, (name.to_string(), state));
    }
    let mut name_count: BTreeMap<&str, usize> = BTreeMap::new();
    for (name, _) in by_node.values() {
        *name_count.entry(name.as_str()).or_default() += 1;
    }
    by_node
        .iter()
        .map(|(node, (name, state))| {
            let key = if name_count[name.as_str()] > 1 {
                format!("{name}#{node}")
            } else {
                name.clone()
            };
            (key, (*state).to_string())
        })
        .collect()
}

/// Render the summary, shrinking it until it fits [`RUN_SUMMARY_MAX_BYTES`]:
/// first the error text is shortened, then the stage map is dropped (the
/// counts stay). The result always parses as JSON.
pub(super) fn render_summary(
    job_id: Option<&str>,
    stages: &BTreeMap<String, String>,
    outcome: &RunOutcome<'_>,
) -> String {
    let mut object = json!({ "version": 1 });
    if let Some(job_id) = job_id {
        object["job"] = json!(job_id);
    }
    let mut error_text = None;
    match outcome {
        RunOutcome::Succeeded {
            stages_total,
            cache_hits,
            cache_misses,
            warnings,
        } => {
            object["state"] = json!("succeeded");
            object["stagesTotal"] = json!(stages_total);
            object["cacheHits"] = json!(cache_hits);
            object["cacheMisses"] = json!(cache_misses);
            if *warnings > 0 {
                object["warnings"] = json!(warnings);
            }
        }
        RunOutcome::Failed { error } => {
            object["state"] = json!("failed");
            error_text = Some(format!("{error:#}"));
        }
    }
    if !stages.is_empty() {
        object["stages"] = json!(stages);
    }
    for error_budget in [2048usize, 512, 128] {
        if let Some(text) = &error_text {
            object["error"] = json!(truncate_chars(text, error_budget));
        }
        let rendered = object.to_string();
        if rendered.len() <= RUN_SUMMARY_MAX_BYTES {
            return rendered;
        }
    }
    if let Some(map) = object.as_object_mut() {
        map.remove("stages");
        map.insert("stagesOmitted".into(), json!(stages.len()));
    }
    object.to_string()
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(lines: &[&str]) -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        std::fs::write(td.path().join("status.jsonl"), lines.join("\n")).unwrap();
        td
    }

    #[test]
    fn the_last_event_of_each_node_wins() {
        let td = status(&[
            r#"{"kind":"stage_begin","node_idx":0,"stage_name":"load","input_hash":"00"}"#,
            r#"{"kind":"stage_end","node_idx":0,"stage_name":"load","elapsed":{"secs":0,"nanos":1}}"#,
            r#"{"kind":"stage_skipped","node_idx":1,"stage_name":"split","invocation_key":"00"}"#,
            r#"{"kind":"stage_begin","node_idx":2,"stage_name":"take","input_hash":"00"}"#,
            r#"{"kind":"stage_failed","node_idx":2,"stage_name":"take","error":"boom"}"#,
            r#"{"kind":"telemetry","telemetry":"counter"}"#,
            "not json",
        ]);
        let states = fold_stage_states(&td.path().join("status.jsonl"));
        assert_eq!(states["load"], "ok");
        assert_eq!(states["split"], "cached");
        assert_eq!(states["take"], "failed");
        assert_eq!(states.len(), 3);
    }

    #[test]
    fn a_repeated_stage_name_is_keyed_by_node() {
        let td = status(&[
            r#"{"kind":"stage_end","node_idx":3,"stage_name":"take","elapsed":{"secs":0,"nanos":1}}"#,
            r#"{"kind":"stage_skipped","node_idx":4,"stage_name":"take","invocation_key":"00"}"#,
        ]);
        let states = fold_stage_states(&td.path().join("status.jsonl"));
        assert_eq!(states["take#3"], "ok");
        assert_eq!(states["take#4"], "cached");
    }

    #[test]
    fn a_missing_status_file_yields_no_stages() {
        let td = tempfile::tempdir().unwrap();
        assert!(fold_stage_states(&td.path().join("status.jsonl")).is_empty());
    }

    #[test]
    fn a_success_summary_carries_the_counts() {
        let stages = BTreeMap::from([("load".to_string(), "ok".to_string())]);
        let out = render_summary(
            Some("job-1"),
            &stages,
            &RunOutcome::Succeeded {
                stages_total: 1,
                cache_hits: 0,
                cache_misses: 1,
                warnings: 0,
            },
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["state"], "succeeded");
        assert_eq!(v["cacheMisses"], 1);
        assert_eq!(v["stages"]["load"], "ok");
        assert_eq!(v["job"], "job-1");
        assert!(v.get("warnings").is_none());
    }

    #[test]
    fn an_oversized_summary_still_parses_and_fits() {
        let stages: BTreeMap<String, String> = (0..400)
            .map(|i| (format!("stage_with_a_long_name_{i}"), "ok".to_string()))
            .collect();
        let error = anyhow::anyhow!("{}", "x".repeat(10_000));
        let out = render_summary(Some("job"), &stages, &RunOutcome::Failed { error: &error });
        assert!(out.len() <= RUN_SUMMARY_MAX_BYTES, "{} bytes", out.len());
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["state"], "failed");
        assert_eq!(v["stagesOmitted"], 400);
        assert!(v["error"].as_str().unwrap().ends_with('…'));
    }
}
