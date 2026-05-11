//! Stage 8 — `dpo_train`. Wraps trainer_dpo.py.
//!
//! Full DPO implementation pending in a follow-up. Rust-side
//! typed contract is complete; the Python side currently emits
//! Failed for non-self-check invocations. Recipe + executor wiring
//! works end-to-end as a smoke target.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::artifacts::{HfCheckpoint, PreferenceJsonl};
use crate::backend::TrainBackend;
use crate::framework::artifact::ContentHash;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::paths;
use crate::python_backend::PythonTrainBackend;
use crate::spec::{DatasetSource, Method, Optim, TrainSpec};

pub struct DpoTrain;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub base_model: String,
    pub output_name: String,
    /// DPO temperature. Smaller = stronger preference signal.
    pub beta: f32,
    pub lr: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub seq_len: u32,
    pub seed: u64,
}

#[async_trait]
impl Stage for DpoTrain {
    const NAME: &'static str = "dpo_train";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    type Input = PreferenceJsonl;
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: PreferenceJsonl,
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        let output_dir = ctx.stage_dir.join("checkpoint");
        std::fs::create_dir_all(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;

        let spec = TrainSpec {
            base_model: args.base_model.clone(),
            output_name: args.output_name.clone(),
            output_dir: output_dir.clone(),
            method: Method::QLora { rank: 16, alpha: 32 },
            dataset: DatasetSource::JsonlPath { path: input.path.clone() },
            optimizer: Optim::AdamW8bit,
            lr: args.lr,
            epochs: args.epochs,
            batch_size: args.batch_size,
            grad_accum: args.grad_accum,
            seq_len: args.seq_len,
            seed: args.seed,
            quant: "Q4_K_M".into(),
            skip_convert: true,
        };
        spec.validate().map_err(|e| StageError::BadInput(format!("{e}")))?;

        let python =
            paths::resolve_python().map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
        let trainer_script = paths::resolve_trainer_script_named("trainer_dpo.py")
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
        let mut backend = PythonTrainBackend::new(python, trainer_script);
        let on_status: crate::backend::StatusFn = Box::new(|_u| {});

        // Pass beta via spec extension (not yet first-class on
        // TrainSpec). For now beta is propagated through
        // RUST_LOG-discoverable env or future TrainSpec field.
        let _ = args.beta;

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
            method_tag: "dpo".into(),
            content_hash: hash,
            final_loss: artifact.final_loss,
        })
    }
}
