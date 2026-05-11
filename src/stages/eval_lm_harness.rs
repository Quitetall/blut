//! Stage 14 — `eval_lm_harness`.
//!
//! Wraps EleutherAI lm-eval-harness as a subprocess. Input is a
//! checkpoint, output is an `EvalReport` keyed by task name.
//! Resource declaration is `[Gpu, Network]` because the harness
//! downloads benchmark datasets on first use.
//!
//! Like `eval_loss`, this commit ships a synthetic fallback that
//! computes a deterministic-per-checkpoint score per task. Real
//! subprocess invocation lands in a follow-up that resolves
//! `lm_eval` via `paths::resolve_python()` and parses the harness
//! JSON output.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{DatasetJsonl, EvalReport, HfCheckpoint};
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct EvalLmHarness;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub tasks: Vec<String>,
    #[serde(default = "default_num_fewshot")]
    pub num_fewshot: u32,
}
fn default_num_fewshot() -> u32 {
    0
}

#[async_trait]
impl Stage for EvalLmHarness {
    const NAME: &'static str = "eval_lm_harness";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    // Tuple input: shares the (HfCheckpoint, DatasetJsonl) shape
    // with `eval_loss` + `eval_judge` so the `eval_suite` recipe
    // can fork3 from a single materialization. The dataset half
    // is currently unused by lm-eval-harness itself, but recipes
    // that want to feed a custom eval split (e.g. via the harness's
    // `--include_path` future flag) will have it available.
    type Input = (HfCheckpoint, DatasetJsonl);
    type Output = EvalReport;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Args,
    ) -> Result<EvalReport, StageError> {
        let (ckpt, _dataset) = input;
        // R21 + R23.
        debug_assert!(!ckpt.base_model.is_empty(), "ckpt base_model required");
        if args.tasks.is_empty() {
            return Err(StageError::BadInput(
                "eval_lm_harness: tasks list must be non-empty".into(),
            ));
        }
        let mut tasks_map = serde_json::Map::new();
        let seed = ckpt.content_hash.0[1] as f32 / 255.0;
        for (i, t) in args.tasks.iter().enumerate() {
            // Synthetic deterministic score; real path parses harness output.
            let score = 0.3 + (seed + i as f32 * 0.05) % 0.6;
            tasks_map.insert(
                t.clone(),
                serde_json::json!({
                    "acc": score,
                    "n_fewshot": args.num_fewshot,
                }),
            );
        }
        let metrics = serde_json::json!({
            "tasks": serde_json::Value::Object(tasks_map),
            "synthetic": true,
        });
        let path = ctx.stage_dir.join("eval_lm_harness.json");
        super::util::write_report(&path, &metrics)?;
        let content_hash = ContentHash::hash_file(&path).map_err(|source| StageError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(EvalReport {
            path,
            evaluator: "eval_lm_harness".into(),
            metrics,
            content_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ckpt() -> HfCheckpoint {
        HfCheckpoint {
            path: PathBuf::from("/tmp/ckpt"),
            base_model: "Qwen/Qwen3-7B".into(),
            method_tag: "qlora_merged".into(),
            content_hash: ContentHash::of_bytes(b"ckpt"),
            final_loss: 0.4,
        }
    }
    fn ds() -> DatasetJsonl {
        DatasetJsonl {
            path: PathBuf::from("/tmp/x.jsonl"),
            content_hash: ContentHash::of_bytes(b""),
            n_examples: 10,
        }
    }
    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn produces_per_task_scores() {
        let td = tempfile::tempdir().unwrap();
        let r = EvalLmHarness
            .run(
                &ctx(td.path()),
                (ckpt(), ds()),
                &Args {
                    tasks: vec!["hellaswag".into(), "arc_easy".into()],
                    num_fewshot: 5,
                },
            )
            .await
            .unwrap();
        assert!(r.metrics["tasks"]["hellaswag"]["acc"].is_number());
        assert!(r.metrics["tasks"]["arc_easy"]["acc"].is_number());
    }

    #[tokio::test]
    async fn rejects_empty_tasks() {
        let td = tempfile::tempdir().unwrap();
        let r = EvalLmHarness
            .run(
                &ctx(td.path()),
                (ckpt(), ds()),
                &Args {
                    tasks: vec![],
                    num_fewshot: 0,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
