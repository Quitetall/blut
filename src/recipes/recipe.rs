//! Recipe trait + erased catalog entry (`RecipeDef`).
//!
//! Each recipe declares its target backend via `type Backend`,
//! holds a typed `Args` struct (serde + JsonSchema), and a
//! `compile` method producing a `Plan<(), Self::Backend>`. The erased
//! [`RecipeDef`] lets the CLI / MCP layer list, schema, and run a recipe
//! by name regardless of which backend it targets — erasure happens at
//! the `Plan<(), B>::into_compiled() → CompiledPlan` boundary.
//!
//! blut-core ships ZERO concrete recipes: the static catalog + lookup
//! helpers moved to the cookbook crates (`cookbook-lamu` at C2b,
//! `cookbook-lamquant` at C2a). Recipes are composed at runtime via
//! [`crate::framework::Registry`] (find / by_category / all over the
//! union of registered cookbooks).

use crate::backends::TrainingBackend;
use crate::framework::error::RecipeError;
use crate::framework::plan::{CompiledPlan, Plan};

pub trait Recipe: Send + Sync + 'static {
    const NAME: &'static str;
    const DESCRIPTION: &'static str;
    type Backend: TrainingBackend;
    type Args: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync + 'static;
    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError>;
}

/// Category bucket the BLUT Training Cockpit menu groups by.
///
/// Each recipe declares its category statically; the TUI lists
/// recipes under the matching section ("DATA PREPARATION",
/// "TRAINING", "EVALUATION", "EXPORT", "PIPELINE", "USER").
/// Categories are also used by `blut recipe list --category <…>`
/// for filtered CLI browsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeCategory {
    /// Recipes that prepare / convert / index raw input → typed
    /// artifacts (e.g. a data-prep recipe packs a corpus into a typed
    /// artifact).
    DataPrep,
    /// Recipes that consume artifacts + produce checkpoints
    /// (`SnnCkpt`, `JointCkpt`, `HfCheckpoint`, etc.).
    Train,
    /// Recipes that consume checkpoints + produce `EvalReport`.
    Eval,
    /// Recipes that take a checkpoint + materialize a deployable
    /// artifact (`HardenedCkpt`, `FirmwareBundle`, `GgufModel`).
    Export,
    /// Recipes that chain multiple categories end-to-end (e.g. a
    /// pipeline recipe = data prep → train → gate).
    Pipeline,
    /// User-authored recipes from `blut/src/recipes/user/`.
    User,
}

impl RecipeCategory {
    /// Human-readable section header used by `blut tui` + `blut
    /// recipe list`.
    pub fn label(self) -> &'static str {
        match self {
            Self::DataPrep => "DATA PREPARATION",
            Self::Train => "TRAINING",
            Self::Eval => "EVALUATION",
            Self::Export => "EXPORT",
            Self::Pipeline => "PIPELINE",
            Self::User => "USER",
        }
    }
}

/// Erased registry entry. Stored in the static `RECIPES` slice.
pub struct RecipeDef {
    pub name: &'static str,
    pub description: &'static str,
    /// Backend identity (e.g. "lamu", "hf_trainer", "lamquant").
    /// Set from `<Recipe>::Backend::ID` in each recipe's DEF.
    pub backend_id: &'static str,
    /// Where this recipe lives in the cockpit menu hierarchy.
    pub category: RecipeCategory,
    /// Artifact `Kind` IDs the recipe consumes (the first stage's
    /// `Input::KIND`, or its tuple flattening for multi-input
    /// stages). Empty when the recipe's first stage is graph-input
    /// (`Input = ()`). Used by `blut tui` to filter the dataset
    /// picker to compatible kinds.
    pub input_kinds: &'static [&'static str],
    /// Artifact `Kind` ID the recipe ultimately produces (the last
    /// stage's `Output::KIND`). Used by `blut tui` to surface
    /// swap-candidate recipes (those with matching I/O kinds).
    pub output_kind: &'static str,
    /// Returns the recipe's args JSON schema.
    pub args_schema_fn: fn() -> serde_json::Value,
    /// Parse JSON args + compile to a backend-erased CompiledPlan.
    /// Recipe DEFs wrap their `Plan<(), B>` via `.into_compiled()`.
    pub compile_fn: fn(serde_json::Value) -> Result<CompiledPlan, RecipeError>,
}

// blut-core ships NO static recipe catalog and NO by-name lookup over
// one. Concrete recipes live in cookbook crates; the
// [`crate::framework::Registry`] composes the live catalog (find /
// by_category / all) over the union of registered cookbooks. The old
// free-fn lookups + the static catalog slices moved out with the recipes
// (C2a/C2b).

#[cfg(test)]
mod tests {
    use super::*;

    /// A neutral, domain-free `RecipeDef` fixture so the erased-catalog
    /// invariants (non-empty metadata, schema serializes) can be checked
    /// without any concrete cookbook in scope. blut-core owns the
    /// machinery, not the recipes.
    static FIXTURE: RecipeDef = RecipeDef {
        name: "fixture_recipe",
        description: "test-only RecipeDef for engine invariants",
        backend_id: "fixture",
        category: RecipeCategory::Train,
        input_kinds: &["dataset.jsonl"],
        output_kind: "checkpoint.hf",
        args_schema_fn: || serde_json::json!({"type": "object", "properties": {}}),
        compile_fn: |_raw| Err(RecipeError::CompileFailed("fixture not runnable".into())),
    };

    #[test]
    fn recipe_def_metadata_is_well_formed() {
        // U1 invariant: a RecipeDef must declare non-empty name /
        // description / output_kind and a schema that serializes.
        assert!(!FIXTURE.name.is_empty(), "recipe missing name");
        assert!(!FIXTURE.description.is_empty(), "empty description");
        assert!(!FIXTURE.output_kind.is_empty(), "empty output_kind");
        let schema = (FIXTURE.args_schema_fn)();
        assert!(schema != serde_json::Value::Null);
        assert!(schema.is_object());
        let _ = FIXTURE.category;
    }

    #[test]
    fn category_label_is_stable() {
        assert_eq!(RecipeCategory::Train.label(), "TRAINING");
        assert_eq!(RecipeCategory::DataPrep.label(), "DATA PREPARATION");
    }
}
