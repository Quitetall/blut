//! Recipe — `distill_from_teacher`.
//!
//! Compiles a knowledge-distillation pipeline:
//!
//!   materialize_for_eval (teacher + dataset)
//!     → distill_train     (student checkpoint)
//!     → merge_lora        (clean ckpt)
//!     → convert_gguf      (Q4_K_M / whatever quant)
//!     → register_model    (registry entry)
//!
//! `materialize_for_eval` is reused as the input materializer
//! because it already produces `(HfCheckpoint, DatasetJsonl)` — the
//! exact shape `distill_train` consumes. The stage's lock-tag is
//! immaterial here ("loaded" for the teacher); the recipe pre-trains
//! step uses the model as a fixed reference and never modifies it.

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};
use crate::stages::{
    convert_gguf::{Args as ConvertArgs, ConvertGguf},
    distill_train::{Args as DistillArgs, DistillTrain},
    materialize_for_eval::{Args as MatArgs, MaterializeForEval},
    merge_lora::{Args as MergeArgs, MergeLora},
    register_model::{Args as RegArgs, RegisterModel},
};

pub struct DistillFromTeacher;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub output_name: String,
    /// Path to the teacher checkpoint directory.
    pub teacher_path: std::path::PathBuf,
    #[serde(default)]
    pub teacher_base_model: String,
    /// Distillation dataset (jsonl path). Typically the teacher's
    /// own outputs sampled to disk by an upstream pipeline.
    pub dataset_path: std::path::PathBuf,
    /// Student base. Smaller than teacher in the common case.
    pub student_base: String,
    #[serde(default = "default_kl_weight")]
    pub kl_weight: f32,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default = "default_batch")]
    pub batch_size: u32,
    #[serde(default = "default_grad_accum")]
    pub grad_accum: u32,
    #[serde(default = "default_seq_len")]
    pub seq_len: u32,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_quant")]
    pub quant: String,
    #[serde(default)]
    pub notes: String,
}

fn default_kl_weight() -> f32 {
    0.5
}
fn default_lr() -> f32 {
    1e-4
}
fn default_epochs() -> u32 {
    3
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
fn default_quant() -> String {
    "Q4_K_M".into()
}

impl Recipe for DistillFromTeacher {
    const NAME: &'static str = "distill_from_teacher";
    const DESCRIPTION: &'static str =
        "Knowledge-distill a smaller student from a teacher checkpoint over a \
         supplied dataset. KL-divergence-weighted loss; same merge/convert/register \
         tail as the SFT recipe.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<()>, RecipeError> {
        if !(args.kl_weight >= 0.0 && args.kl_weight <= 1.0) {
            return Err(RecipeError::InvalidArgs(format!(
                "kl_weight must be in [0, 1]; got {}",
                args.kl_weight
            )));
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                MaterializeForEval,
                MatArgs {
                    model_path: args.teacher_path.clone(),
                    base_model: args.teacher_base_model.clone(),
                    dataset_path: args.dataset_path.clone(),
                },
            )
            .then(
                DistillTrain,
                DistillArgs {
                    student_base: args.student_base.clone(),
                    output_name: args.output_name.clone(),
                    kl_weight: args.kl_weight,
                    lr: args.lr,
                    epochs: args.epochs,
                    batch_size: args.batch_size,
                    grad_accum: args.grad_accum,
                    seq_len: args.seq_len,
                    seed: args.seed,
                },
            )
            .then(MergeLora, MergeArgs::default())
            .then(
                ConvertGguf,
                ConvertArgs {
                    quant: args.quant.clone(),
                    name: args.output_name.clone(),
                },
            )
            .then(
                RegisterModel,
                RegArgs {
                    name: args.output_name.clone(),
                    notes: args.notes.clone(),
                    arch: "distilled".into(),
                },
            )
            .finish();
        Ok(plan)
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: DistillFromTeacher::NAME,
    description: DistillFromTeacher::DESCRIPTION,
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).unwrap_or(serde_json::Value::Null)
    },
    compile_fn: |raw| {
        let args: Args = serde_json::from_value(raw)
            .map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        DistillFromTeacher.compile(args)
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            output_name: "student-v1".into(),
            teacher_path: "/tmp/teacher".into(),
            teacher_base_model: "Qwen/Qwen3-7B".into(),
            dataset_path: "/tmp/distill.jsonl".into(),
            student_base: "Qwen/Qwen3-1.5B".into(),
            kl_weight: default_kl_weight(),
            lr: default_lr(),
            epochs: default_epochs(),
            batch_size: default_batch(),
            grad_accum: default_grad_accum(),
            seq_len: default_seq_len(),
            seed: default_seed(),
            quant: default_quant(),
            notes: String::new(),
        }
    }

    #[test]
    fn compiles_to_5_node_plan() {
        let plan = DistillFromTeacher.compile(args()).unwrap();
        assert_eq!(plan.n_nodes(), 5);
        assert_eq!(plan.n_edges(), 4);
        let order = plan.topo_order().unwrap();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn rejects_out_of_range_kl_weight() {
        let mut a = args();
        a.kl_weight = 1.5;
        let r = DistillFromTeacher.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn defaults_apply_when_fields_omitted() {
        let args = serde_json::json!({
            "output_name": "s",
            "teacher_path": "/t",
            "dataset_path": "/d",
            "student_base": "Qwen/Qwen3-1.5B",
        });
        let plan = (DEF.compile_fn)(args).unwrap();
        assert_eq!(plan.n_nodes(), 5);
    }
}
