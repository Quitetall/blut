//! Recipe: `dpo_from_preferences`. preference JSONL → DPO training
//! → GGUF → registry.

use serde::{Deserialize, Serialize};

use crate::artifacts::PreferenceJsonl;
use crate::framework::artifact::ContentHash;
use crate::framework::error::{RecipeError, StageError};
use crate::framework::plan::Plan;
use crate::framework::resource::Resource;
use crate::framework::stage::{Stage, StageContext};
use crate::recipes::recipe::{Recipe, RecipeDef};
use crate::stages::{
    convert_gguf::{Args as ConvertArgs, ConvertGguf},
    dpo_train::{Args as DpoArgs, DpoTrain},
    register_model::{Args as RegArgs, RegisterModel},
};

/// Trivial materializer: validates a preferences JSONL exists and
/// hashes it into a `PreferenceJsonl` artifact.
struct MaterializePreferences;

impl crate::framework::Compatible<crate::backends::LamuTrainerBackend> for MaterializePreferences {}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct MatPrefArgs {
    pub path: std::path::PathBuf,
}

#[async_trait::async_trait]
impl Stage for MaterializePreferences {
    const NAME: &'static str = "materialize_preferences";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = ();
    type Output = PreferenceJsonl;
    type Args = MatPrefArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _: (),
        args: &Self::Args,
    ) -> Result<PreferenceJsonl, StageError> {
        if !args.path.exists() {
            return Err(StageError::BadInput(format!(
                "preferences file not found: {}",
                args.path.display()
            )));
        }
        let hash = ContentHash::hash_file(&args.path).map_err(|source| StageError::Io {
            path: args.path.clone(),
            source,
        })?;
        // Count examples by line.
        let body = std::fs::read_to_string(&args.path).map_err(|source| StageError::Io {
            path: args.path.clone(),
            source,
        })?;
        let n_pairs = body.lines().filter(|l| !l.trim().is_empty()).count() as i64;
        Ok(PreferenceJsonl {
            path: args.path.clone(),
            content_hash: hash,
            n_pairs,
        })
    }
}

pub struct DpoFromPreferences;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub output_name: String,
    pub preferences_path: std::path::PathBuf,
    #[serde(default = "default_base")]
    pub base_model: String,
    #[serde(default = "default_beta")]
    pub beta: f32,
    #[serde(default = "default_lr")]
    pub lr: f32,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default = "default_quant")]
    pub quant: String,
}

fn default_base() -> String {
    "Qwen/Qwen3-7B".into()
}
fn default_beta() -> f32 {
    0.1
}
fn default_lr() -> f32 {
    5e-6
}
fn default_epochs() -> u32 {
    3
}
fn default_quant() -> String {
    "Q4_K_M".into()
}

impl Recipe for DpoFromPreferences {
    type Backend = crate::backends::LamuTrainerBackend;
    const NAME: &'static str = "dpo_from_preferences";
    const DESCRIPTION: &'static str = "Direct preference optimization from a chosen/rejected JSONL. Stub trainer; full DPO impl pending.";
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        // R23: arg validation at the recipe boundary.
        if args.output_name.is_empty() {
            return Err(RecipeError::InvalidArgs("output_name is empty".into()));
        }
        if !(args.beta > 0.0 && args.beta.is_finite()) {
            return Err(RecipeError::InvalidArgs(format!(
                "beta must be positive finite; got {}",
                args.beta
            )));
        }
        if !(args.lr > 0.0 && args.lr.is_finite()) {
            return Err(RecipeError::InvalidArgs(format!(
                "lr must be positive finite; got {}",
                args.lr
            )));
        }
        if args.epochs == 0 {
            return Err(RecipeError::InvalidArgs("epochs must be > 0".into()));
        }
        let recipe_args =
            serde_json::to_value(&args).map_err(|e| RecipeError::CompileFailed(format!("{e}")))?;
        let plan = Plan::new(Self::NAME, recipe_args)
            .start(
                MaterializePreferences,
                MatPrefArgs {
                    path: args.preferences_path.clone(),
                },
            )
            .then(
                DpoTrain,
                DpoArgs {
                    base_model: args.base_model.clone(),
                    output_name: args.output_name.clone(),
                    beta: args.beta,
                    lr: args.lr,
                    epochs: args.epochs,
                    batch_size: 1,
                    grad_accum: 8,
                    seq_len: 4096,
                    seed: 42,
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
                    notes: format!("DPO from {}", args.preferences_path.display()),
                    arch: "trained-dpo".into(),
                },
            )
            .finish();
        Ok(plan)
    }
}

pub static DEF: RecipeDef = RecipeDef {
    name: DpoFromPreferences::NAME,
    description: DpoFromPreferences::DESCRIPTION,
    backend_id: <crate::backends::LamuTrainerBackend as crate::backends::TrainingBackend>::ID,
    category: crate::recipes::recipe::RecipeCategory::Train,
    input_kinds: &["dataset.preferences"],
    output_kind: "model.gguf",
    args_schema_fn: || {
        let mut g = schemars::r#gen::SchemaGenerator::default();
        let s = g.subschema_for::<Args>();
        serde_json::to_value(s).expect("schemars-derived JsonSchema must serialize cleanly")
    },
    compile_fn: |raw| {
        let args: Args =
            serde_json::from_value(raw).map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
        DpoFromPreferences.compile(args).map(|p| p.into_compiled())
    },
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_to_4_node_plan() {
        let plan = DpoFromPreferences
            .compile(Args {
                output_name: "demo".into(),
                preferences_path: "/tmp/p.jsonl".into(),
                base_model: default_base(),
                beta: default_beta(),
                lr: default_lr(),
                epochs: default_epochs(),
                quant: default_quant(),
            })
            .unwrap()
            .into_compiled();
        assert_eq!(plan.n_nodes(), 4);
    }
}
