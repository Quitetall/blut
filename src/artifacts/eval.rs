// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Eval reports.
//!
//! Paradigm-agnostic metric bag emitted by `eval_loss`, `eval_lm_harness`,
//! `eval_judge`, and any future eval stages. Combined by
//! `merge_reports` into a single artifact for the `eval_suite` recipe's
//! terminal node.
//!
//! Metric values are JSON to keep the schema open (numbers,
//! per-task subreports, optional CIs). Consumers (the report
//! renderer, downstream `continual_finetune` recipes) parse the
//! values they care about and ignore the rest.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::framework::artifact::{Artifact, ContentHash};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalReport {
    /// Where the JSON form of this report lives on disk (so it
    /// can be inspected without going through the cache).
    pub path: PathBuf,
    /// Logical name of the evaluator that produced this report
    /// (`"eval_loss"`, `"eval_lm_harness"`, `"eval_judge"`,
    /// `"merge_reports"` for the combined output).
    pub evaluator: String,
    /// Free-form metric bag. Convention: top-level numeric metrics
    /// at the root, per-task breakdowns nested under
    /// `tasks.<task_name>`. Consumers may ignore unknown keys.
    pub metrics: serde_json::Value,
    pub content_hash: ContentHash,
}

impl Artifact for EvalReport {
    const KIND: &'static str = "eval.report";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_report_round_trips_via_serde() {
        let r = EvalReport {
            path: PathBuf::from("/tmp/eval.json"),
            evaluator: "eval_loss".into(),
            metrics: serde_json::json!({"loss": 0.42, "ppl": 1.52}),
            content_hash: ContentHash::of_bytes(b"x"),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: EvalReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.evaluator, "eval_loss");
        assert_eq!(back.metrics["loss"], serde_json::json!(0.42));
    }

    #[test]
    fn artifact_kind_is_stable() {
        assert_eq!(EvalReport::KIND, "eval.report");
        assert_eq!(EvalReport::SCHEMA, 1);
    }
}
