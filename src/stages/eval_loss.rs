//! Stage 13 — `eval_loss`.
//!
//! Cross-entropy + perplexity over an eval split, evaluated against
//! a checkpoint. Input is `(HfCheckpoint, DatasetJsonl)` — the
//! checkpoint + the eval split. Output is an `EvalReport` with
//! `{loss, perplexity, n_examples}` at the metric root.
//!
//! Implementation note: this commit ships the framework wiring +
//! a synthetic-result fallback path so the recipe DAG can be
//! exercised end-to-end without the Python evaluator. Real
//! evaluation lands in a follow-up that shells out to
//! `python/eval_loss.py` (the trainer's existing `--eval-only`
//! mode is the obvious shim). The synthetic path computes
//! `loss = 0.5 + (input_hash[0] as f32 / 512.0)` so test runs
//! get a deterministic non-trivial number per checkpoint.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{DatasetJsonl, EvalReport, HfCheckpoint};
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};

pub struct EvalLoss;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_max_seq")]
    pub max_seq: u32,
}
fn default_batch_size() -> u32 {
    1
}
fn default_max_seq() -> u32 {
    4096
}

#[async_trait]
impl Stage for EvalLoss {
    const NAME: &'static str = "eval_loss";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
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
        // Synthetic-result fallback (see module doc).
        let seed = ckpt.content_hash.0[0] as f32;
        let loss = 0.5 + seed / 512.0;
        let metrics = serde_json::json!({
            "loss": loss,
            "perplexity": loss.exp(),
            "n_examples": ds.n_examples,
            "batch_size": args.batch_size,
            "max_seq": args.max_seq,
            "synthetic": true,
        });
        let path = ctx.stage_dir.join("eval_loss.json");
        write_report(&path, &metrics)?;
        let content_hash = ContentHash::hash_file(&path).map_err(|source| StageError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(EvalReport {
            path,
            evaluator: "eval_loss".into(),
            metrics,
            content_hash,
        })
    }
}

pub(crate) fn write_report(
    path: &std::path::Path,
    metrics: &serde_json::Value,
) -> Result<(), StageError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let body = serde_json::to_vec_pretty(metrics).map_err(|e| {
        StageError::Backend(anyhow::anyhow!("serialize eval report: {e}"))
    })?;
    std::fs::write(path, body).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ckpt(byte: u8) -> HfCheckpoint {
        HfCheckpoint {
            path: PathBuf::from("/tmp/ckpt"),
            base_model: "Qwen/Qwen3-7B".into(),
            method_tag: "qlora_merged".into(),
            content_hash: ContentHash([byte; 32]),
            final_loss: 0.4,
        }
    }
    fn ds() -> DatasetJsonl {
        DatasetJsonl {
            path: PathBuf::from("/tmp/eval.jsonl"),
            content_hash: ContentHash::of_bytes(b"eval"),
            n_examples: 50,
        }
    }
    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn produces_a_report() {
        let td = tempfile::tempdir().unwrap();
        let r = EvalLoss
            .run(
                &ctx(td.path()),
                (ckpt(0), ds()),
                &Args {
                    batch_size: 1,
                    max_seq: 4096,
                },
            )
            .await
            .unwrap();
        assert_eq!(r.evaluator, "eval_loss");
        assert_eq!(r.metrics["n_examples"], serde_json::json!(50));
        assert!(r.metrics["loss"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn loss_varies_with_checkpoint() {
        let td = tempfile::tempdir().unwrap();
        let r1 = EvalLoss
            .run(
                &ctx(td.path()),
                (ckpt(0), ds()),
                &Args::default(),
            )
            .await
            .unwrap();
        let r2 = EvalLoss
            .run(
                &ctx(td.path()),
                (ckpt(64), ds()),
                &Args::default(),
            )
            .await
            .unwrap();
        assert_ne!(r1.metrics["loss"], r2.metrics["loss"]);
    }
}

impl Default for Args {
    fn default() -> Self {
        Args {
            batch_size: default_batch_size(),
            max_seq: default_max_seq(),
        }
    }
}
