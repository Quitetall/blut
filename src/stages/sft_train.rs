//! Stage 7 — `sft_train`.
//!
//! Wraps `python_backend::PythonTrainBackend` (the existing SFT
//! trainer.py runner). Converts a typed `DatasetJsonl` input + Args
//! into a legacy `TrainSpec`, runs through the python subprocess,
//! returns a typed `HfCheckpoint`.
//!
//! Resource: Gpu + Network. The cross-process scheduler_lock from
//! lamu-core is acquired by recipes (or the executor's commit-6
//! Resource::Gpu enforcement); this stage only concerns itself
//! with running the trainer once it has the GPU.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{DatasetJsonl, HfCheckpoint};
use crate::backend::TrainBackend;
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::paths;
use crate::python_backend::PythonTrainBackend;
use crate::spec::{DatasetSource, Method, Optim, TrainSpec};

pub struct SftTrain;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub base_model: String,
    pub output_name: String,
    /// `qlora`, `lora`, or `full`.
    pub method: String,
    pub rank: u32,
    pub alpha: u32,
    pub optimizer: String,
    pub lr: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub seq_len: u32,
    pub seed: u64,
}

#[async_trait]
impl Stage for SftTrain {
    const NAME: &'static str = "sft_train";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    type Input = DatasetJsonl;
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: DatasetJsonl,
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        let output_dir = ctx.stage_dir.join("checkpoint");
        std::fs::create_dir_all(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;

        let method = match args.method.as_str() {
            "qlora" => Method::QLora { rank: args.rank, alpha: args.alpha },
            "lora" => Method::Lora { rank: args.rank, alpha: args.alpha },
            "full" => Method::Full,
            other => {
                return Err(StageError::BadInput(format!(
                    "unknown method '{other}', expected qlora|lora|full"
                )));
            }
        };
        let optimizer = match args.optimizer.as_str() {
            "adam_w" | "adamw" => Optim::AdamW,
            "adam_w8bit" | "adamw8bit" => Optim::AdamW8bit,
            "apollo_mini" => Optim::ApolloMini,
            "apollo_rank4" | "apollo" => Optim::ApolloRank4,
            other => {
                return Err(StageError::BadInput(format!(
                    "unknown optimizer '{other}'"
                )));
            }
        };

        let spec = TrainSpec {
            base_model: args.base_model.clone(),
            output_name: args.output_name.clone(),
            output_dir: output_dir.clone(),
            method,
            dataset: DatasetSource::JsonlPath { path: input.path.clone() },
            optimizer,
            lr: args.lr,
            epochs: args.epochs,
            batch_size: args.batch_size,
            grad_accum: args.grad_accum,
            seq_len: args.seq_len,
            seed: args.seed,
            quant: "Q4_K_M".into(), // unused by the trainer; convert_gguf owns quant
            skip_convert: true,
        };
        spec.validate()
            .map_err(|e| StageError::BadInput(format!("{e}")))?;

        let python = paths::resolve_python()
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
        let trainer_script = paths::resolve_trainer_script()
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
        let mut backend = PythonTrainBackend::new(python, trainer_script);

        // Forward StageStep events from trainer.py's per-step
        // status updates so the unified status.jsonl carries them.
        let stage_name_owned = SftTrain::NAME.to_string();
        let status_tx = ctx.status_tx.clone();
        let on_status: crate::backend::StatusFn = Box::new(move |update| {
            if let Ok(value) = serde_json::to_value(&update) {
                let _ = status_tx.send(crate::framework::status::StageEvent::StageStep {
                    node_idx: 0, // recipe layer doesn't surface idx here; logger reconstructs
                    stage_name: stage_name_owned.clone(),
                    update: value,
                });
            }
        });

        let artifact = backend
            .run(spec.clone(), on_status)
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let hash = ContentHash::hash_dir(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;

        Ok(HfCheckpoint {
            path: output_dir,
            base_model: args.base_model.clone(),
            method_tag: args.method.clone(),
            content_hash: hash,
            final_loss: artifact.final_loss,
        })
    }
}

#[allow(dead_code)]
fn _path_marker(_p: &PathBuf) {}
