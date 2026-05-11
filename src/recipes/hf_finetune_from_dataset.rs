//! Recipe — `hf_finetune_from_dataset`.
//!
//! HF Trainer parallel to `finetune_from_dataset` (lamu). Plan
//! shape identical; differs only in the train step (HfSftTrain
//! vs SftTrain) and the typed backend.
//!
//!   materialize_dataset_path → split_train_eval → take_train →
//!   hf_sft_train → merge_lora → convert_gguf → register_model
//!
//! Backend: HfTrainerBackend (auto-managed venv + transformers.Trainer).

use serde::{Deserialize, Serialize};

use crate::backends::hf_trainer::HfSftTrain;
use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};
use crate::stages::{
    convert_gguf::{Args as ConvertArgs, ConvertGguf},
    materialize_dataset_path::{Args as MatArgs, MaterializeDatasetPath},
    merge_lora::{Args as MergeArgs, MergeLora},
    register_model::{Args as RegArgs, RegisterModel},
    split_train_eval::{Args as SplitArgs, SplitTrainEval},
    TakeTrain,
};

pub struct HfFinetuneFromDataset;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub output_name: String,
    #[serde(default)]
    pub dataset_path: Option<std::path::PathBuf>,
    #[serde(default)]
    pub registered_dataset: Option<String>,
    #[serde(default = "default_base")]
    pub base_model: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default = "default_quant")]
    pub quant: String,
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
    #[serde(default = "default_rank")]
    pub rank: u32,
    #[serde(default = "default_alpha")]
    pub alpha: u32,
    #[serde(default = "default_eval_ratio")]
    pub eval_ratio: f32,
    /// `TrainingArguments` overrides forwarded to `transformers.Trainer`.
    #[serde(default)]
    pub hf_extra: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub notes: String,
}

fn default_base() -> String {
    "Qwen/Qwen3-7B".into()
}
fn default_method() -> String {
    "qlora".into()
}
fn default_quant() -> String {
    "Q4_K_M".into()
}
fn default_lr() -> f32 {
    2e-4
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
fn default_rank() -> u32 {
    16
}
fn default_alpha() -> u32 {
    32
}
fn default_eval_ratio() -> f32 {
    0.1
}

impl Recipe for HfFinetuneFromDataset {
    type Backend = crate::backends::HfTrainerBackend;
    const NAME: &'static str = "hf_finetune_from_dataset";
    const DESCRIPTION: &'static str =
        "HuggingFace Trainer SFT from an existing dataset (JSONL path or registered name). \
         Auto-managed venv via transformers.Trainer + PEFT LoRA/QLoRA + bitsandbytes.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        match (&args.dataset_path, &args.registered_dataset) {
            (Some(_), Some(_)) => {
                return Err(RecipeError::InvalidArgs(
                    "pass exactly one of dataset_path / registered_dataset, not both".into(),
                ));
            }
            (None, None) => {
                return Err(RecipeError::InvalidArgs(
                    "one of dataset_path / registered_dataset is required".into(),
                ));
            }
            _ => {}
        }
        if !matches!(args.method.as_str(), "qlora" | "lora" | "full") {
            return Err(RecipeError::InvalidArgs(format!(
                "method '{}' must be qlora|lora|full",
                args.method
            )));
        }
        if args.output_name.is_empty() {
            return Err(RecipeError::InvalidArgs("output_name is empty".into()));
        }
        if !(args.eval_ratio > 0.0 && args.eval_ratio < 1.0) {
            return Err(RecipeError::InvalidArgs(format!(
                "eval_ratio must be in (0, 1); got {}",
                args.eval_ratio
            )));
        }
        if args.epochs == 0 || args.batch_size == 0 || args.grad_accum == 0 || args.seq_len == 0 {
            return Err(RecipeError::InvalidArgs(
                "epochs, batch_size, grad_accum, seq_len must all be > 0".into(),
            ));
        }
        if !(args.lr > 0.0 && args.lr.is_finite()) {
            return Err(RecipeError::InvalidArgs(format!(
                "lr must be positive finite; got {}",
                args.lr
            )));
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let plan = Plan::<(), Self::Backend>::new(Self::NAME, recipe_args_json)
            .start(
                MaterializeDatasetPath,
                MatArgs {
                    path: args.dataset_path.clone(),
                    registered_name: args.registered_dataset.clone(),
                },
            )
            .then(
                SplitTrainEval,
                SplitArgs {
                    eval_ratio: args.eval_ratio,
                    seed: args.seed,
                },
            )
            .then(TakeTrain, crate::stages::take_train::Args::default())
            .then(
                HfSftTrain,
                crate::backends::hf_trainer::stages::hf_sft_train::Args {
                    base_model: args.base_model.clone(),
                    method: args.method.clone(),
                    rank: args.rank,
                    alpha: args.alpha,
                    lr: args.lr,
                    epochs: args.epochs,
                    batch_size: args.batch_size,
                    grad_accum: args.grad_accum,
                    seq_len: args.seq_len,
                    seed: args.seed,
                    eval_dataset_path: String::new(),
                    extra: args.hf_extra.clone(),
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
                    arch: "trained-hf".into(),
                },
            )
            .finish();
        Ok(plan)
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: HfFinetuneFromDataset::NAME,
    description: HfFinetuneFromDataset::DESCRIPTION,
    backend_id: <crate::backends::HfTrainerBackend as crate::backends::TrainingBackend>::ID,
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).expect("schemars-derived JsonSchema must serialize cleanly")
    },
    compile_fn: |raw| {
        let args: Args = serde_json::from_value(raw)
            .map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        HfFinetuneFromDataset.compile(args).map(|p| p.into_compiled())
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            output_name: "demo".into(),
            dataset_path: Some("/tmp/x.jsonl".into()),
            registered_dataset: None,
            base_model: default_base(),
            method: default_method(),
            quant: default_quant(),
            lr: default_lr(),
            epochs: default_epochs(),
            batch_size: default_batch(),
            grad_accum: default_grad_accum(),
            seq_len: default_seq_len(),
            seed: default_seed(),
            rank: default_rank(),
            alpha: default_alpha(),
            eval_ratio: default_eval_ratio(),
            hf_extra: serde_json::Map::new(),
            notes: String::new(),
        }
    }

    #[test]
    fn compiles_to_7_node_plan() {
        let plan = HfFinetuneFromDataset.compile(args()).unwrap().into_compiled();
        assert_eq!(plan.n_nodes(), 7);
        assert_eq!(plan.n_edges(), 6);
    }

    #[test]
    fn rejects_invalid_method() {
        let mut a = args();
        a.method = "rlhf".into();
        let r = HfFinetuneFromDataset.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_both_dataset_specifiers() {
        let mut a = args();
        a.registered_dataset = Some("foo".into());
        let r = HfFinetuneFromDataset.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }
}
