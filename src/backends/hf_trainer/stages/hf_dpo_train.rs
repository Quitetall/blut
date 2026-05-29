//! Stage — `hf_dpo_train`. DPO via `trl.DPOTrainer`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{HfCheckpoint, PreferenceJsonl};
use crate::backends::HfTrainerBackend;
use crate::backends::hf_trainer::{DpoConfig, HfTrainerJob, HfTrainerRunner, StatusLine};
use crate::framework::artifact::ContentHash;
use crate::framework::compat::Compatible;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::framework::status::StageEvent;

pub struct HfDpoTrain;
impl Compatible<HfTrainerBackend> for HfDpoTrain {}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub base_model: String,
    pub beta: f32,
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
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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
impl Stage for HfDpoTrain {
    const NAME: &'static str = "hf_dpo_train";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = PreferenceJsonl;
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: PreferenceJsonl,
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        if input.n_pairs <= 0 {
            return Err(StageError::BadInput(format!(
                "preferences must be non-empty (n_pairs = {})",
                input.n_pairs
            )));
        }
        if args.seq_len == 0 {
            return Err(StageError::BadInput("seq_len must be > 0".into()));
        }
        if !(args.beta > 0.0 && args.beta.is_finite()) {
            return Err(StageError::BadInput(format!(
                "beta must be positive finite; got {}",
                args.beta
            )));
        }
        if !(args.lr > 0.0 && args.lr.is_finite()) {
            return Err(StageError::BadInput(format!(
                "lr must be positive finite; got {}",
                args.lr
            )));
        }
        if args.epochs == 0 {
            return Err(StageError::BadInput("epochs must be > 0".into()));
        }
        if args.base_model.is_empty() {
            return Err(StageError::BadInput("base_model must be non-empty".into()));
        }

        let output_dir = ctx.stage_dir.join("checkpoint");
        std::fs::create_dir_all(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;

        let job = HfTrainerJob {
            task: "dpo".into(),
            base_model: args.base_model.clone(),
            train_dataset_path: input.path.clone(),
            eval_dataset_path: None,
            output_dir: output_dir.clone(),
            lr: args.lr,
            epochs: args.epochs,
            batch_size: args.batch_size,
            grad_accum: args.grad_accum,
            seq_len: args.seq_len,
            seed: args.seed,
            extra: args.extra.clone(),
            peft: None,
            dpo: Some(DpoConfig {
                beta: args.beta,
                preferences_path: None,
            }),
        };

        let status_tx = ctx.status_tx.clone();
        let node_idx = ctx.node_idx;
        let cb = Box::new(move |s: StatusLine| match s {
            StatusLine::Step {
                step,
                total,
                loss,
                lr,
            } => {
                let _ = status_tx.send(StageEvent::StageStep {
                    node_idx,
                    stage_name: HfDpoTrain::NAME.to_string(),
                    update: serde_json::json!({
                        "kind": "hf_step", "step": step, "total": total,
                        "loss": loss, "lr": lr,
                    }),
                });
            }
            StatusLine::Saved { path } => {
                let _ = status_tx.send(StageEvent::StageStep {
                    node_idx,
                    stage_name: HfDpoTrain::NAME.to_string(),
                    update: serde_json::json!({"kind": "hf_saved", "path": path}),
                });
            }
            _ => {}
        }) as Box<dyn Fn(StatusLine) + Send + Sync>;

        let mut runner = HfTrainerRunner::new();
        let result = runner
            .run(job, cb)
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let content_hash =
            ContentHash::hash_dir(&result.checkpoint_dir).map_err(|source| StageError::Io {
                path: result.checkpoint_dir.clone(),
                source,
            })?;

        Ok(HfCheckpoint {
            path: result.checkpoint_dir,
            base_model: args.base_model.clone(),
            method_tag: "dpo".into(),
            content_hash,
            final_loss: result.final_loss.unwrap_or(0.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn prefs(td: &std::path::Path) -> PreferenceJsonl {
        let p = td.join("prefs.jsonl");
        std::fs::write(&p, r#"{"prompt":"p","chosen":"c","rejected":"r"}"#).unwrap();
        PreferenceJsonl {
            path: p,
            content_hash: ContentHash::of_bytes(b""),
            n_pairs: 1,
        }
    }

    #[test]
    fn deterministic_false() {
        const { assert!(!<HfDpoTrain as Stage>::DETERMINISTIC) };
    }

    #[tokio::test]
    async fn rejects_nonpositive_beta() {
        let td = tempfile::tempdir().unwrap();
        let r = HfDpoTrain
            .run(
                &ctx(td.path()),
                prefs(td.path()),
                &Args {
                    base_model: "Qwen/Qwen3-7B".into(),
                    beta: -0.1,
                    lr: 5e-6,
                    epochs: 1,
                    batch_size: 1,
                    grad_accum: 1,
                    seq_len: 1024,
                    seed: 42,
                    extra: serde_json::Map::new(),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
