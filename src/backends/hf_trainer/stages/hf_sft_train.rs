//! Stage — `hf_sft_train`. SFT via `transformers.Trainer`.
//! Backend-coupled to `HfTrainerBackend`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{DatasetJsonl, HfCheckpoint};
use crate::backends::hf_trainer::{HfTrainerJob, HfTrainerRunner, PeftConfig, StatusLine};
use crate::backends::HfTrainerBackend;
use crate::framework::artifact::ContentHash;
use crate::framework::compat::Compatible;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::framework::status::StageEvent;

pub struct HfSftTrain;
impl Compatible<HfTrainerBackend> for HfSftTrain {}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub base_model: String,
    /// "lora" | "qlora" | "full". Drives PEFT config.
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default = "default_rank")]
    pub rank: u32,
    #[serde(default = "default_alpha")]
    pub alpha: u32,
    pub lr: f32,
    pub epochs: u32,
    #[serde(default = "default_batch")]
    pub batch_size: u32,
    #[serde(default = "default_grad_accum")]
    pub grad_accum: u32,
    #[serde(default = "default_seq_len")]
    pub seq_len: u32,
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Optional eval dataset path (string for clean JSON args).
    #[serde(default)]
    pub eval_dataset_path: String,
    /// Free-form `TrainingArguments` overrides (optim, scheduler,
    /// warmup_ratio, etc.) folded into the runner's `extra`.
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn default_method() -> String {
    "qlora".into()
}
fn default_rank() -> u32 {
    16
}
fn default_alpha() -> u32 {
    32
}
fn default_batch() -> u32 {
    1
}
fn default_grad_accum() -> u32 {
    8
}
fn default_seq_len() -> u32 {
    4096
}
fn default_seed() -> u64 {
    42
}

#[async_trait]
impl Stage for HfSftTrain {
    const NAME: &'static str = "hf_sft_train";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = DatasetJsonl;
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: DatasetJsonl,
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        // R23: empty dataset is a user error, not an internal
        // invariant — promoted from debug_assert.
        if input.n_examples <= 0 {
            return Err(StageError::BadInput(format!(
                "dataset has no examples (n_examples = {})",
                input.n_examples
            )));
        }
        if args.seq_len == 0 {
            return Err(StageError::BadInput("seq_len must be > 0".into()));
        }
        if !matches!(args.method.as_str(), "qlora" | "lora" | "full") {
            return Err(StageError::BadInput(format!(
                "method '{}' must be qlora|lora|full",
                args.method
            )));
        }
        if !(args.lr > 0.0 && args.lr.is_finite()) {
            return Err(StageError::BadInput(format!(
                "lr must be positive finite; got {}",
                args.lr
            )));
        }
        if args.epochs == 0 || args.batch_size == 0 || args.grad_accum == 0 {
            return Err(StageError::BadInput(
                "epochs/batch_size/grad_accum must be > 0".into(),
            ));
        }
        if args.base_model.is_empty() {
            return Err(StageError::BadInput("base_model must be non-empty".into()));
        }

        let output_dir = ctx.stage_dir.join("checkpoint");
        std::fs::create_dir_all(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;

        let peft = match args.method.as_str() {
            "full" => None,
            _ => Some(PeftConfig {
                method: args.method.clone(),
                rank: args.rank,
                alpha: args.alpha,
            }),
        };

        let eval_dataset_path = if args.eval_dataset_path.is_empty() {
            None
        } else {
            Some(std::path::PathBuf::from(&args.eval_dataset_path))
        };

        let job = HfTrainerJob {
            task: "sft".into(),
            base_model: args.base_model.clone(),
            train_dataset_path: input.path.clone(),
            eval_dataset_path,
            output_dir: output_dir.clone(),
            lr: args.lr,
            epochs: args.epochs,
            batch_size: args.batch_size,
            grad_accum: args.grad_accum,
            seq_len: args.seq_len,
            seed: args.seed,
            extra: args.extra.clone(),
            peft,
            dpo: None,
        };

        // Fan tqdm-style step events into the executor's status
        // broadcast. The runner's `StatusLine::Step` carries
        // optional loss + lr; pass them through as `StageStep`.
        let status_tx = ctx.status_tx.clone();
        let node_idx = ctx.node_idx;
        let cb = Box::new(move |s: StatusLine| match s {
            StatusLine::Step { step, total, loss, lr } => {
                let _ = status_tx.send(StageEvent::StageStep {
                    node_idx,
                    stage_name: HfSftTrain::NAME.to_string(),
                    update: serde_json::json!({
                        "kind": "hf_step",
                        "step": step,
                        "total": total,
                        "loss": loss,
                        "lr": lr,
                    }),
                });
            }
            StatusLine::Saved { path } => {
                let _ = status_tx.send(StageEvent::StageStep {
                    node_idx,
                    stage_name: HfSftTrain::NAME.to_string(),
                    update: serde_json::json!({
                        "kind": "hf_saved",
                        "path": path,
                    }),
                });
            }
            StatusLine::Done { .. } | StatusLine::Failed { .. } => {
                // Terminal events — handled via the run's return value.
            }
        })
            as Box<dyn Fn(StatusLine) + Send + Sync>;

        let mut runner = HfTrainerRunner::new();
        let result = runner
            .run(job, cb)
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let content_hash = ContentHash::hash_dir(&result.checkpoint_dir).map_err(|source| {
            StageError::Io {
                path: result.checkpoint_dir.clone(),
                source,
            }
        })?;

        Ok(HfCheckpoint {
            path: result.checkpoint_dir,
            base_model: args.base_model.clone(),
            method_tag: args.method.clone(),
            content_hash,
            final_loss: result.final_loss.unwrap_or(0.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn ds(td: &std::path::Path) -> DatasetJsonl {
        let p = td.join("train.jsonl");
        std::fs::write(&p, r#"{"text":"hello"}"#).unwrap();
        DatasetJsonl {
            path: p,
            content_hash: ContentHash::of_bytes(b"x"),
            n_examples: 1,
        }
    }

    #[tokio::test]
    async fn rejects_invalid_method() {
        let td = tempfile::tempdir().unwrap();
        let r = HfSftTrain
            .run(
                &ctx(td.path()),
                ds(td.path()),
                &Args {
                    base_model: "Qwen/Qwen3-7B".into(),
                    method: "nonsense".into(),
                    rank: 16,
                    alpha: 32,
                    lr: 2e-4,
                    epochs: 1,
                    batch_size: 1,
                    grad_accum: 1,
                    seq_len: 1024,
                    seed: 42,
                    eval_dataset_path: String::new(),
                    extra: serde_json::Map::new(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_zero_lr() {
        let td = tempfile::tempdir().unwrap();
        let r = HfSftTrain
            .run(
                &ctx(td.path()),
                ds(td.path()),
                &Args {
                    base_model: "Qwen/Qwen3-7B".into(),
                    method: "qlora".into(),
                    rank: 16,
                    alpha: 32,
                    lr: 0.0,
                    epochs: 1,
                    batch_size: 1,
                    grad_accum: 1,
                    seq_len: 1024,
                    seed: 42,
                    eval_dataset_path: String::new(),
                    extra: serde_json::Map::new(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[test]
    fn deterministic_false() {
        assert!(!<HfSftTrain as Stage>::DETERMINISTIC);
    }
}
