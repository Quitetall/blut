//! Recipe trait + erased catalog (`RECIPES`).
//!
//! Each recipe has a typed `Args` struct (serde + JsonSchema) and
//! a `compile` method that turns those args into a `Plan<()>`.
//! The static `RECIPES` slice carries an erased entry per recipe
//! so the CLI / MCP layer can list, schema, and run by name.

use crate::framework::error::RecipeError;
use crate::framework::plan::Plan;

pub trait Recipe: Send + Sync + 'static {
    const NAME: &'static str;
    const DESCRIPTION: &'static str;
    type Args: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync + 'static;
    fn compile(&self, args: Self::Args) -> Result<Plan<()>, RecipeError>;
}

/// Erased registry entry. Stored in the static `RECIPES` slice.
pub struct RecipeDef {
    pub name: &'static str,
    pub description: &'static str,
    /// Returns the recipe's args JSON schema.
    pub args_schema_fn: fn() -> serde_json::Value,
    /// Parse JSON args + compile to a Plan.
    pub compile_fn: fn(serde_json::Value) -> Result<Plan<()>, RecipeError>,
}

/// Slice of `&RecipeDef` (not `RecipeDef`): RecipeDef contains
/// fn-pointers that can't be Copy-moved into an array initializer.
/// Each entry is a reference to the `pub static DEF` defined
/// inside its recipe module.
pub static RECIPES: &[&RecipeDef] = &[
    &crate::recipes::finetune_from_conversations::DEF,
    &crate::recipes::dpo_from_preferences::DEF,
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
