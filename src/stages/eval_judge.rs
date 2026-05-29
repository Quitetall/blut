//! Stage 15 — `eval_judge`.
//!
//! Pair-of-prompts evaluation by a judge model. Generates a
//! response from `input.0` (the candidate checkpoint) on each
//! prompt in `args.prompts`, sends candidate + prompt to a judge
//! model via HTTP (not via the MCP server to keep this crate's
//! dependency tree shallow), and aggregates judge scores.
//!
//! Input is `(HfCheckpoint, DatasetJsonl)`. The dataset's first
//! `args.n_samples` lines are read as eval prompts when
//! `args.prompts` is empty — keeps recipes that already have an
//! eval split from needing to duplicate prompt strings.
//!
//! Stub-body shape matches the other eval stages: synthetic
//! per-checkpoint scoring so the recipe DAG round-trips end-to-end
//! before the real HTTP path lands.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{DatasetJsonl, EvalReport, HfCheckpoint};
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct EvalJudge;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub judge_model: String,
    #[serde(default)]
    pub prompts: Vec<String>,
    #[serde(default = "default_n_samples")]
    pub n_samples: u32,
}
fn default_n_samples() -> u32 {
    20
}

#[async_trait]
impl Stage for EvalJudge {
    const NAME: &'static str = "eval_judge";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    type Input = (HfCheckpoint, DatasetJsonl);
    type Output = EvalReport;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Args,
    ) -> Result<EvalReport, StageError> {
        let (ckpt, ds) = input;
        // R23: hard guards (release-mode too). Negative n_examples
        // would wrap on the `as u32` cast below; large n_samples
        // could overflow into negative via `as i64`. Reject both
        // up front rather than silently producing garbage `n`.
        if ds.n_examples < 0 {
            return Err(StageError::BadInput(format!(
                "eval_judge: dataset n_examples {} is negative",
                ds.n_examples
            )));
        }
        if args.judge_model.is_empty() {
            return Err(StageError::BadInput(
                "eval_judge: judge_model is empty".into(),
            ));
        }
        if args.n_samples == 0 && args.prompts.is_empty() {
            return Err(StageError::BadInput(
                "eval_judge: either n_samples > 0 or prompts must be non-empty".into(),
            ));
        }
        // After the ds.n_examples >= 0 + n_samples: u32 checks,
        // `args.n_samples as i64` is lossless and the min() result
        // is bounded by u32::MAX, so the `as u32` cast below is
        // well-defined.
        let seed = ckpt.content_hash.0[2] as f32 / 255.0;
        let n = if args.prompts.is_empty() {
            (args.n_samples as i64).min(ds.n_examples).max(0) as u32
        } else {
            args.prompts.len() as u32
        };
        let score = 0.4 + seed * 0.5;
        let metrics = serde_json::json!({
            "judge_model": args.judge_model,
            "n_samples": n,
            "mean_score": score,
            "synthetic": true,
        });
        let path = ctx.stage_dir.join("eval_judge.json");
        super::util::write_report(&path, &metrics)?;
        let content_hash = ContentHash::hash_file(&path).map_err(|source| StageError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(EvalReport {
            path,
            evaluator: "eval_judge".into(),
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
    fn ds(n: i64) -> DatasetJsonl {
        DatasetJsonl {
            path: PathBuf::from("/tmp/x.jsonl"),
            content_hash: ContentHash::of_bytes(b""),
            n_examples: n,
        }
    }
    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn produces_mean_score() {
        let td = tempfile::tempdir().unwrap();
        let r = EvalJudge
            .run(
                &ctx(td.path()),
                (ckpt(), ds(100)),
                &Args {
                    judge_model: "claude-opus-4-7".into(),
                    prompts: vec![],
                    n_samples: 10,
                },
            )
            .await
            .unwrap();
        assert!(r.metrics["mean_score"].is_number());
        assert_eq!(r.metrics["n_samples"], serde_json::json!(10));
    }

    #[tokio::test]
    async fn rejects_empty_judge_model() {
        let td = tempfile::tempdir().unwrap();
        let r = EvalJudge
            .run(
                &ctx(td.path()),
                (ckpt(), ds(10)),
                &Args {
                    judge_model: "".into(),
                    prompts: vec![],
                    n_samples: 1,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
