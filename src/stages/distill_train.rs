//! Stage 9 — `distill_train`. Wraps trainer_distill.py.
//!
//! Two-input stage: takes (HfCheckpoint teacher, DatasetJsonl) and
//! produces an HfCheckpoint student. Teacher's outputs are sampled
//! into the dataset path before training (the Python side handles
//! that). Currently uses trainer_distill.py stub.

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

pub struct DistillTrain;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub student_base: String,
    pub output_name: String,
    pub kl_weight: f32,
    pub lr: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub seq_len: u32,
    pub seed: u64,
}

#[async_trait]
impl Stage for DistillTrain {
    const NAME: &'static str = "distill_train";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    type Input = (HfCheckpoint, DatasetJsonl);
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: (HfCheckpoint, DatasetJsonl),
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        let (teacher, dataset) = input;
        let output_dir = ctx.stage_dir.join("checkpoint");
        std::fs::create_dir_all(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;

        let spec = TrainSpec {
            base_model: args.student_base.clone(),
            output_name: args.output_name.clone(),
            output_dir: output_dir.clone(),
            method: Method::QLora { rank: 16, alpha: 32 },
            dataset: DatasetSource::JsonlPath { path: dataset.path.clone() },
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
        let trainer_script = paths::resolve_trainer_script_named("trainer_distill.py")
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
        let mut backend = PythonTrainBackend::new(python, trainer_script)
            .with_env("LAMU_TEACHER_PATH", teacher.path.to_string_lossy().into_owned())
            .with_env("LAMU_KL_WEIGHT", format!("{}", args.kl_weight));
        let on_status: crate::backend::StatusFn = Box::new(|_u| {});

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
            base_model: args.student_base.clone(),
            method_tag: "distill".into(),
            content_hash: hash,
            final_loss: artifact.final_loss,
        })
    }
}
