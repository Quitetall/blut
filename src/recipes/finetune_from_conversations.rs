//! Recipe: `finetune_from_conversations`.
//!
//! Full v2-commit-4 pipeline as a typed Plan:
//!
//!   materialize_conversations
//!     → filter_dataset
//!     → register_dataset
//!     → split_train_eval        (DatasetSplit)
//!     → take_train (passthrough to DatasetJsonl)
//!     → sft_train
//!     → merge_lora
//!     → convert_gguf
//!     → register_model
//!
//! `split_train_eval` produces `DatasetSplit`, but `sft_train` takes
//! `DatasetJsonl`. To keep the typed Plan strict + linear, we use a
//! tiny `take_train` adapter stage (defined in
//! `stages/take_train.rs`) that projects the split's train half.
//! When the executor grows real branch support (commit 6), this
//! becomes a `fork` + selective merge instead — but for the
//! sequential executor the adapter is the cheapest way to keep the
//! type lattice clean.

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};
use crate::stages::{
    convert_gguf::{Args as ConvertArgs, ConvertGguf},
    filter_dataset::{Args as FilterArgs, FilterDataset},
    materialize_conversations::{Args as MatArgs, MaterializeConversations},
    merge_lora::{Args as MergeArgs, MergeLora},
    register_dataset::{Args as RegDsArgs, RegisterDataset},
    register_model::{Args as RegArgs, RegisterModel},
    sft_train::{Args as SftArgs, SftTrain},
    split_train_eval::{Args as SplitArgs, SplitTrainEval},
};
use crate::stages::take_train::TakeTrain;

pub struct FinetuneFromConversations;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub output_name: String,
    /// Humantime duration like "30d", "12h". Parsed by `compile`
    /// into seconds for `MaterializeConversations`.
    pub since: String,
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
    #[serde(default)]
    pub notes: String,
    /// Eval fraction. Defaults to 0.1 (10%).
    #[serde(default = "default_eval_ratio")]
    pub eval_ratio: f32,
    /// Min messages per example. 0 disables.
    #[serde(default = "default_min_turns")]
    pub min_turns: u32,
    /// Max bytes per message. 0 disables.
    #[serde(default = "default_max_msg_bytes")]
    pub max_msg_bytes: u32,
    #[serde(default = "default_drop_errors")]
    pub drop_errors: bool,
    /// Optional dataset registry name. None = skip register_dataset
    /// stage entirely (still part of the pipeline; recipes can
    /// configure it to no-op).
    #[serde(default = "default_register_dataset_name")]
    pub dataset_registry_name: String,
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
fn default_min_turns() -> u32 {
    2
}
fn default_max_msg_bytes() -> u32 {
    65_536
}
fn default_drop_errors() -> bool {
    true
}
fn default_register_dataset_name() -> String {
    String::new()
}

impl Recipe for FinetuneFromConversations {
    const NAME: &'static str = "finetune_from_conversations";
    const DESCRIPTION: &'static str =
        "Fine-tune a base model on the user's recent LAMU conversation history. \
         Pulls turns from conversations.db, runs SFT via the python trainer, \
         converts to GGUF, registers the result.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<()>, RecipeError> {
        // R23 (validate both ends): tighten arg range checks at the
        // recipe boundary before any plumbing runs. Caller bugs
        // surface here rather than as cryptic Python errors three
        // hours into training.
        if args.output_name.is_empty() {
            return Err(RecipeError::InvalidArgs("output_name is empty".into()));
        }
        let since_secs = humantime::parse_duration(&args.since)
            .map_err(|e| RecipeError::InvalidArgs(format!("since '{}': {e}", args.since)))?
            .as_secs();
        if since_secs == 0 {
            return Err(RecipeError::InvalidArgs(format!(
                "since '{}' resolves to zero seconds",
                args.since
            )));
        }
        if !matches!(args.method.as_str(), "qlora" | "lora" | "full") {
            return Err(RecipeError::InvalidArgs(format!(
                "method '{}' must be qlora|lora|full",
                args.method
            )));
        }
        if !(args.eval_ratio > 0.0 && args.eval_ratio < 1.0) {
            return Err(RecipeError::InvalidArgs(format!(
                "eval_ratio must be in (0, 1); got {}",
                args.eval_ratio
            )));
        }
        if args.epochs == 0 {
            return Err(RecipeError::InvalidArgs("epochs must be > 0".into()));
        }
        if args.batch_size == 0 || args.grad_accum == 0 {
            return Err(RecipeError::InvalidArgs(
                "batch_size and grad_accum must be > 0".into(),
            ));
        }
        if !(args.lr > 0.0 && args.lr.is_finite()) {
            return Err(RecipeError::InvalidArgs(format!(
                "lr must be a positive finite number; got {}",
                args.lr
            )));
        }
        if args.seq_len == 0 {
            return Err(RecipeError::InvalidArgs("seq_len must be > 0".into()));
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let registry_name = if args.dataset_registry_name.is_empty() {
            // Default to "<output_name>-conversations" so the recipe
            // self-tags without the user having to invent a name.
            format!("{}-conversations", args.output_name)
        } else {
            args.dataset_registry_name.clone()
        };

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(MaterializeConversations, MatArgs { since_seconds: since_secs })
            .then(
                FilterDataset,
                FilterArgs {
                    min_turns: args.min_turns,
                    max_msg_bytes: args.max_msg_bytes,
                    drop_errors: args.drop_errors,
                },
            )
            .then(
                RegisterDataset,
                RegDsArgs {
                    name: registry_name,
                    kind: "sft".into(),
                    metadata: None,
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

/// Erased catalog entry. The `RECIPES` slice in `recipes::recipe`
/// references this.
pub static DEF: RecipeDef = RecipeDef {
    name: FinetuneFromConversations::NAME,
    description: FinetuneFromConversations::DESCRIPTION,
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).expect("schemars-derived JsonSchema must serialize cleanly")
    },
    compile_fn: |raw| {
        let args: Args = serde_json::from_value(raw)
            .map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        FinetuneFromConversations.compile(args)
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_to_9_node_plan() {
        // materialize → filter → register_dataset → split → take_train
        // → sft_train → merge_lora → convert_gguf → register_model
        let args = Args {
            output_name: "demo".into(),
            since: "30d".into(),
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
            notes: String::new(),
            eval_ratio: default_eval_ratio(),
            min_turns: default_min_turns(),
            max_msg_bytes: default_max_msg_bytes(),
            drop_errors: default_drop_errors(),
            dataset_registry_name: default_register_dataset_name(),
        };
        let plan = FinetuneFromConversations.compile(args).unwrap();
        assert_eq!(plan.n_nodes(), 9);
        assert_eq!(plan.n_edges(), 8);
        let order = plan.topo_order().unwrap();
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn rejects_unsupported_since() {
        let mut args = json_args();
        args["since"] = serde_json::json!("nonsense");
        let r = (DEF.compile_fn)(args);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn rejects_invalid_method() {
        let mut args = json_args();
        args["method"] = serde_json::json!("rlhf");
        let r = (DEF.compile_fn)(args);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }

    #[test]
    fn defaults_apply_when_fields_omitted() {
        // Only required fields supplied; defaults fill the rest.
        let args = serde_json::json!({
            "output_name": "demo",
            "since": "7d",
        });
        let plan = (DEF.compile_fn)(args).unwrap();
        assert_eq!(plan.n_nodes(), 9);
    }

    fn json_args() -> serde_json::Value {
        serde_json::json!({
            "output_name": "demo",
            "since": "30d",
        })
    }
}
