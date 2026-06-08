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

/// Return all recipes whose category matches.
pub fn by_category(cat: RecipeCategory) -> impl Iterator<Item = &'static RecipeDef> {
    RECIPES.iter().copied().filter(move |r| r.category == cat)
}

/// Return all recipes whose (input_kinds, output_kind) tuple matches
/// the given recipe's — i.e. drop-in swap candidates. Excludes the
/// recipe itself. Empty iterator when no swap-candidates exist.
pub fn swap_candidates(of: &'static RecipeDef) -> impl Iterator<Item = &'static RecipeDef> {
    let want_in = of.input_kinds;
    let want_out = of.output_kind;
    let name = of.name;
    RECIPES
        .iter()
        .copied()
        .filter(move |r| r.name != name && r.input_kinds == want_in && r.output_kind == want_out)
}

/// Slice of `&RecipeDef` (not `RecipeDef`): RecipeDef contains
/// fn-pointers that can't be Copy-moved into an array initializer.
/// Each entry is a reference to the `pub static DEF` defined
/// inside its recipe module.
/// blut-core's built-in catalog: the lamu recipes (see `LAMU_RECIPES`).
/// This is the transitional in-crate source for blut-core's own
/// cookbook; domain cookbooks (e.g. `cookbook-lamquant`, moved out at
/// C2a) register their recipes separately and the catalog is composed
/// at runtime by the [`crate::framework::Registry`]. Several tests still
/// index this slice, so it stays until the lamu recipes move (C2b).
pub static RECIPES: &[&RecipeDef] = &[
    // ── blut-lamu cookbook (generic LLM: lamu + hf_trainer backends);
    //    canonical members mirror LAMU_RECIPES. The lamquant cookbook's
    //    recipes moved to the `cookbook-lamquant` crate at C2a; blut-core
    //    now ships only the lamu recipes. hf_finetune stays LAST. ──
    &crate::recipes::finetune_from_conversations::DEF,
    &crate::recipes::finetune_from_dataset::DEF,
    &crate::recipes::dpo_from_preferences::DEF,
    &crate::recipes::eval_suite::DEF,
    &crate::recipes::distill_from_teacher::DEF,
    &crate::recipes::hf_finetune_from_dataset::DEF,
];

/// Recipes owned by the **blut-lamu** cookbook (generic LLM: lamu +
/// hf_trainer backends). Transitional in-crate home — moves to the
/// `blut-lamu` cookbook crate/repo (C2b). See
/// `[[project_blut_cookbook_split]]`.
pub static LAMU_RECIPES: &[&RecipeDef] = &[
    &crate::recipes::finetune_from_conversations::DEF,
    &crate::recipes::finetune_from_dataset::DEF,
    &crate::recipes::dpo_from_preferences::DEF,
    &crate::recipes::eval_suite::DEF,
    &crate::recipes::distill_from_teacher::DEF,
    &crate::recipes::hf_finetune_from_dataset::DEF,
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

    #[test]
    fn recipe_metadata_complete() {
        // U1 gate: every catalog entry must declare non-empty
        // category + output_kind so the cockpit can group / swap.
        for r in RECIPES {
            assert!(!r.name.is_empty(), "recipe missing name");
            assert!(!r.description.is_empty(), "{}: empty description", r.name);
            assert!(
                !r.output_kind.is_empty(),
                "{}: empty output_kind — set to the Plan's final stage output Kind",
                r.name
            );
            // category enum has no Default, so its presence in the
            // RecipeDef literal is enforced by the type system at
            // compile time; this test is the runtime backstop in
            // case someone introduces an Option<RecipeCategory>
            // wrapper later.
            let _ = r.category;
        }
    }

    #[test]
    fn swap_candidates_excludes_self() {
        // No recipe should ever appear in its own swap-candidate list.
        for r in RECIPES {
            assert!(
                !swap_candidates(r).any(|c| c.name == r.name),
                "{} appears in its own swap_candidates",
                r.name
            );
        }
    }
}
