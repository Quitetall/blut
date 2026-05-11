//! Merge stage — `merge_reports`.
//!
//! Combines a 3-way fork of EvalReports (the standard
//! `eval_suite` shape: `eval_loss`, `eval_lm_harness`,
//! `eval_judge`) into a single terminal `EvalReport`. Result's
//! `metrics` is `{by_evaluator: {<name>: <child metrics>}}` plus
//! a flat top-level `summary` with the headline numbers each
//! child surfaced (loss, mean_score, mean task accuracy).
//!
//! Input is `(EvalReport, EvalReport, EvalReport)`; the typed
//! Plan builder's `merge3` enforces arity at compile time.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::EvalReport;
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct MergeReports;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {}

#[async_trait]
impl Stage for MergeReports {
    const NAME: &'static str = "merge_reports";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = (EvalReport, EvalReport, EvalReport);
    type Output = EvalReport;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        _args: &Args,
    ) -> Result<EvalReport, StageError> {
        let (a, b, c) = input;
        // R23: empty evaluator labels would produce malformed
        // by_evaluator JSON (duplicate-key or empty-key collisions).
        // Promoted from debug_assert per V4 Pro retrofit-C review —
        // release builds need the guard too.
        for (i, label) in [a.evaluator.as_str(), b.evaluator.as_str(), c.evaluator.as_str()]
            .iter()
            .enumerate()
        {
            if label.is_empty() {
                return Err(StageError::BadInput(format!(
                    "EvalReport[{i}].evaluator must be non-empty"
                )));
            }
        }
        let by_evaluator = serde_json::json!({
            &a.evaluator: a.metrics.clone(),
            &b.evaluator: b.metrics.clone(),
            &c.evaluator: c.metrics.clone(),
        });
        let summary = build_summary(&[&a, &b, &c]);
        let metrics = serde_json::json!({
            "by_evaluator": by_evaluator,
            "summary": summary,
        });
        let path = ctx.stage_dir.join("eval_summary.json");
        super::util::write_report(&path, &metrics)?;
        let content_hash = ContentHash::hash_file(&path).map_err(|source| StageError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(EvalReport {
            path,
            evaluator: "merge_reports".into(),
            metrics,
            content_hash,
        })
    }
}

fn build_summary(reports: &[&EvalReport]) -> serde_json::Value {
    let mut summary = serde_json::Map::new();
    for r in reports {
        if let Some(v) = r.metrics.get("loss") {
            summary.insert(format!("{}.loss", r.evaluator), v.clone());
        }
        if let Some(v) = r.metrics.get("mean_score") {
            summary.insert(format!("{}.mean_score", r.evaluator), v.clone());
        }
        if let Some(tasks) = r.metrics.get("tasks").and_then(|t| t.as_object()) {
            let accs: Vec<f64> = tasks
                .values()
                .filter_map(|t| t.get("acc").and_then(|a| a.as_f64()))
                .collect();
            if !accs.is_empty() {
                let mean = accs.iter().sum::<f64>() / accs.len() as f64;
                summary.insert(format!("{}.mean_acc", r.evaluator), serde_json::json!(mean));
            }
        }
    }
    serde_json::Value::Object(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn rep(evaluator: &str, metrics: serde_json::Value) -> EvalReport {
        EvalReport {
            path: PathBuf::from(format!("/tmp/{evaluator}.json")),
            evaluator: evaluator.into(),
            metrics,
            content_hash: ContentHash::of_bytes(evaluator.as_bytes()),
        }
    }
    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn combines_three_reports() {
        let td = tempfile::tempdir().unwrap();
        let a = rep("eval_loss", serde_json::json!({"loss": 0.42}));
        let b = rep(
            "eval_lm_harness",
            serde_json::json!({"tasks": {"hellaswag": {"acc": 0.7}, "arc_easy": {"acc": 0.6}}}),
        );
        let c = rep("eval_judge", serde_json::json!({"mean_score": 0.55}));
        let r = MergeReports
            .run(&ctx(td.path()), (a, b, c), &Args::default())
            .await
            .unwrap();
        assert_eq!(r.evaluator, "merge_reports");
        assert_eq!(r.metrics["summary"]["eval_loss.loss"], serde_json::json!(0.42));
        assert_eq!(r.metrics["summary"]["eval_judge.mean_score"], serde_json::json!(0.55));
        let mean_acc = r.metrics["summary"]["eval_lm_harness.mean_acc"].as_f64().unwrap();
        assert!((mean_acc - 0.65).abs() < 1e-9);
    }
}
