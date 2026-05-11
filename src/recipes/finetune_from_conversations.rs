//! Recipe: `finetune_from_conversations`.
//!
//! Replicates the legacy `--from-conversations` flow as a typed
//! Plan: materialize → sft_train → convert_gguf → register_model.
//! Filter / split / merge_lora are deliberately omitted from this
//! commit; they ship as separate stages in later commits and the
//! recipe will gain `.then(filter_dataset)` etc. when they land.

use serde::{Deserialize, Serialize};

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;
use crate::recipes::recipe::{Recipe, RecipeDef};
use crate::stages::{
    convert_gguf::{Args as ConvertArgs, ConvertGguf},
    materialize_conversations::{Args as MatArgs, MaterializeConversations},
    register_model::{Args as RegArgs, RegisterModel},
    sft_train::{Args as SftArgs, SftTrain},
};

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

impl Recipe for FinetuneFromConversations {
    const NAME: &'static str = "finetune_from_conversations";
    const DESCRIPTION: &'static str =
        "Fine-tune a base model on the user's recent LAMU conversation history. \
         Pulls turns from conversations.db, runs SFT via the python trainer, \
         converts to GGUF, registers the result.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<()>, RecipeError> {
        let since_secs = humantime::parse_duration(&args.since)
            .map_err(|e| RecipeError::InvalidArgs(format!("since '{}': {e}", args.since)))?
            .as_secs();
        if !matches!(args.method.as_str(), "qlora" | "lora" | "full") {
            return Err(RecipeError::InvalidArgs(format!(
                "method '{}' must be qlora|lora|full",
                args.method
            )));
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(MaterializeConversations, MatArgs { since_seconds: since_secs })
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
        serde_json::to_value(s).unwrap_or(serde_json::Value::Null)
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
    fn compiles_to_4_node_plan() {
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
        };
        let plan = FinetuneFromConversations.compile(args).unwrap();
        assert_eq!(plan.n_nodes(), 4);
        assert_eq!(plan.n_edges(), 3);
        let order = plan.topo_order().unwrap();
        assert_eq!(order, vec![0, 1, 2, 3]);
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
        assert_eq!(plan.n_nodes(), 4);
    }

    fn json_args() -> serde_json::Value {
        serde_json::json!({
            "output_name": "demo",
            "since": "30d",
        })
    }
}
