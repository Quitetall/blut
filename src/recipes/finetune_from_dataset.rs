//! Recipe — `finetune_from_dataset`.
//!
//! SFT from an existing JSONL dataset (path on disk or a name
//! registered in `datasets_db`). Mirrors recipe 1's tail half;
//! differs only at the materializer:
//!
//!   materialize_dataset_path
//!     → split_train_eval
//!     → take_train
//!     → sft_train
//!     → merge_lora
//!     → convert_gguf
//!     → register_model
//!
//! No `filter_dataset` / `register_dataset` stage: the dataset is
//! pre-existing (either a registered curated set or a one-off path
//! the user takes responsibility for). Recipe 1 owns the
//! conversations-→-curated pipeline.

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};
use crate::stages::{
    convert_gguf::{Args as ConvertArgs, ConvertGguf},
    materialize_dataset_path::{Args as MatArgs, MaterializeDatasetPath},
    merge_lora::{Args as MergeArgs, MergeLora},
    register_model::{Args as RegArgs, RegisterModel},
    sft_train::{Args as SftArgs, SftTrain},
    split_train_eval::{Args as SplitArgs, SplitTrainEval},
};
use crate::stages::take_train::TakeTrain;

pub struct FinetuneFromDataset;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub output_name: String,
    /// Path to a JSONL dataset. Exclusive with `registered_dataset`.
    #[serde(default)]
    pub dataset_path: Option<std::path::PathBuf>,
    /// Registered dataset name. Exclusive with `dataset_path`.
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
    #[serde(default = "default_optim")]
    pub optimizer: String,
    #[serde(default = "default_eval_ratio")]
    pub eval_ratio: f32,
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
fn default_optim() -> String {
    "apollo_mini".into()
}
fn default_eval_ratio() -> f32 {
    0.1
}

impl Recipe for FinetuneFromDataset {
    const NAME: &'static str = "finetune_from_dataset";
    const DESCRIPTION: &'static str =
        "SFT from an existing dataset — JSONL path on disk or a registered name. \
         Tail half identical to finetune_from_conversations; differs only at the \
         source materializer.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<()>, RecipeError> {
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
        // R23: numeric arg ranges.
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

        let plan = Plan::new(Self::NAME, recipe_args_json)
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
                SftTrain,
                SftArgs {
                    base_model: args.base_model.clone(),
                    output_name: args.output_name.clone(),
                    method: args.method.clone(),
                    rank: args.rank,
                    alpha: args.alpha,
                    optimizer: args.optimizer.clone(),
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
                    arch: "trained".into(),
                },
            )
            .finish();
        Ok(plan)
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: FinetuneFromDataset::NAME,
    description: FinetuneFromDataset::DESCRIPTION,
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).expect("schemars-derived JsonSchema must serialize cleanly")
    },
    compile_fn: |raw| {
        let args: Args = serde_json::from_value(raw)
            .map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        FinetuneFromDataset.compile(args)
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
            optimizer: default_optim(),
            eval_ratio: default_eval_ratio(),
            notes: String::new(),
        }
    }

    #[test]
    fn compiles_to_7_node_plan() {
        let plan = FinetuneFromDataset.compile(args()).unwrap();
        assert_eq!(plan.n_nodes(), 7);
        assert_eq!(plan.n_edges(), 6);
    }

    #[test]
    fn rejects_both_dataset_specifiers() {
        let mut a = args();
        a.registered_dataset = Some("foo".into());
        let r = FinetuneFromDataset.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_neither_dataset_specifier() {
        let mut a = args();
        a.dataset_path = None;
        a.registered_dataset = None;
        let r = FinetuneFromDataset.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_invalid_method() {
        let mut a = args();
        a.method = "rlhf".into();
        let r = FinetuneFromDataset.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }
}
