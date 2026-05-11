//! Recipe trait + erased catalog (`RECIPES`).
//!
//! Each recipe declares its target backend via `type Backend`,
//! holds a typed `Args` struct (serde + JsonSchema), and a
//! `compile` method producing a `Plan<(), Self::Backend>`. The
//! static `RECIPES` slice stores erased entries so the CLI / MCP
//! layer can list, schema, and run by name regardless of which
//! backend each recipe targets — erasure happens at the
//! `Plan<(), B>::into_compiled() → CompiledPlan` boundary.

use crate::backends::TrainingBackend;
use crate::framework::error::RecipeError;
use crate::framework::plan::{CompiledPlan, Plan};

pub trait Recipe: Send + Sync + 'static {
    const NAME: &'static str;
    const DESCRIPTION: &'static str;
    type Backend: TrainingBackend;
    type Args: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync + 'static;
    fn compile(&self, args: Self::Args)
        -> Result<Plan<(), Self::Backend>, RecipeError>;
}

/// Erased registry entry. Stored in the static `RECIPES` slice.
pub struct RecipeDef {
    pub name: &'static str,
    pub description: &'static str,
    /// Backend identity (e.g. "lamu", "hf_trainer", "lamquant").
    /// Set from `<Recipe>::Backend::ID` in each recipe's DEF.
    pub backend_id: &'static str,
    /// Returns the recipe's args JSON schema.
    pub args_schema_fn: fn() -> serde_json::Value,
    /// Parse JSON args + compile to a backend-erased CompiledPlan.
    /// Recipe DEFs wrap their `Plan<(), B>` via `.into_compiled()`.
    pub compile_fn: fn(serde_json::Value) -> Result<CompiledPlan, RecipeError>,
}

/// Slice of `&RecipeDef` (not `RecipeDef`): RecipeDef contains
/// fn-pointers that can't be Copy-moved into an array initializer.
/// Each entry is a reference to the `pub static DEF` defined
/// inside its recipe module.
pub static RECIPES: &[&RecipeDef] = &[
    &crate::recipes::finetune_from_conversations::DEF,
    &crate::recipes::finetune_from_dataset::DEF,
    &crate::recipes::dpo_from_preferences::DEF,
    &crate::recipes::eval_suite::DEF,
    &crate::recipes::distill_from_teacher::DEF,
    &crate::recipes::lamquant_snn::DEF,
    &crate::recipes::lamquant_encoder::DEF,
    &crate::recipes::lamquant_oracle::DEF,
];

pub fn find(name: &str) -> Option<&'static RecipeDef> {
    RECIPES.iter().copied().find(|r| r.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finetune_from_conversations_in_catalog() {
        let r = find("finetune_from_conversations").expect("missing recipe");
        assert!(!r.description.is_empty());
        let schema = (r.args_schema_fn)();
        assert!(schema != serde_json::Value::Null);
    }

    #[test]
    fn missing_recipe_returns_none() {
        assert!(find("definitely-not-here").is_none());
    }
}
