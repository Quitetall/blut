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
    /// Cockpit menu bucket (`blut recipe list --category`, TUI sections).
    /// Defaulted so the trait extension stays additive; every shipped
    /// recipe overrides it via [`register_recipe!`].
    const CATEGORY: RecipeCategory = RecipeCategory::User;
    /// Artifact `Kind` IDs the first stage consumes. Default none
    /// (graph-input recipe, `Input = ()`).
    const INPUT_KINDS: &'static [&'static str] = &[];
    /// Artifact `Kind` ID the last stage produces. Default `""`; real
    /// recipes override it (the macro copies it into `RecipeDef`).
    const OUTPUT_KIND: &'static str = "";
    /// Optional default `systemd` `OnCalendar` for `blut schedule install`
    /// when no `--calendar` is given. `None` = no built-in schedule.
    const SCHEDULE: Option<&'static str> = None;
    type Backend: TrainingBackend;
    type Args: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync + 'static;
    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError>;
}

/// Build the args JSON schema for a recipe's `Args` type. This is the body
/// every recipe's `RecipeDef.args_schema_fn` used to hand-inline; the macro
/// references it so the schemars-gen call lives in exactly one place.
///
/// `subschema_for` emits `{"$ref":"#/definitions/Args"}` and parks the actual
/// definition(s) in the generator. The hand-DEFs forgot to attach them, so the
/// schema was a dangling `$ref` and every `definitions`-walking consumer
/// (`blut tui` template prefill, E2's `args_template`) silently fell back to
/// `{}`. Centralizing here lets us attach `take_definitions()` once, fixing
/// that for every recipe.
pub fn schema_of<A: schemars::JsonSchema>() -> serde_json::Value {
    let mut g = schemars::r#gen::SchemaGenerator::default();
    let root = g.subschema_for::<A>();
    let mut v =
        serde_json::to_value(root).expect("schemars-derived JsonSchema must serialize cleanly");
    let defs = g.take_definitions();
    if let (serde_json::Value::Object(map), Ok(defs_val)) =
        (&mut v, serde_json::to_value(&defs))
    {
        if !defs.is_empty() {
            map.insert("definitions".to_string(), defs_val);
        }
    }
    v
}

/// Parse JSON args and compile a recipe to a backend-erased [`CompiledPlan`].
/// This is the body every recipe's `RecipeDef.compile_fn` used to hand-inline.
/// Requires `Default` to construct the (unit-struct) recipe instance — the
/// macro adds `#[derive(Default)]` expectations to migrated recipes.
pub fn compile_erased<R: Recipe + Default>(
    raw: serde_json::Value,
) -> Result<CompiledPlan, RecipeError> {
    let args: R::Args =
        serde_json::from_value(raw).map_err(|e| RecipeError::InvalidArgs(format!("{e}")))?;
    R::default().compile(args).map(|p| p.into_compiled())
}

/// Emit a recipe's `pub static DEF: RecipeDef` from its [`Recipe`] impl.
///
/// Every field is derived from trait consts + associated types, so
/// `backend_id`/`schema`/`compile`/`category` can never drift from the impl.
/// Invoke at module level in a recipe file, after the `impl Recipe`:
///
/// ```ignore
/// #[derive(Default)]
/// pub struct MyRecipe;
/// impl Recipe for MyRecipe { /* … CATEGORY, OUTPUT_KIND, … */ }
/// blut::register_recipe!(MyRecipe);   // → pub static DEF
/// ```
///
/// `$crate` keeps every path resolved against `blut` regardless of the
/// invoking cookbook crate.
#[macro_export]
macro_rules! register_recipe {
    ($ty:ty) => {
        pub static DEF: $crate::recipes::recipe::RecipeDef =
            $crate::recipes::recipe::RecipeDef {
                name: <$ty as $crate::recipes::recipe::Recipe>::NAME,
                description: <$ty as $crate::recipes::recipe::Recipe>::DESCRIPTION,
                backend_id: <<$ty as $crate::recipes::recipe::Recipe>::Backend
                    as $crate::backends::TrainingBackend>::ID,
                category: <$ty as $crate::recipes::recipe::Recipe>::CATEGORY,
                input_kinds: <$ty as $crate::recipes::recipe::Recipe>::INPUT_KINDS,
                output_kind: <$ty as $crate::recipes::recipe::Recipe>::OUTPUT_KIND,
                schedule: <$ty as $crate::recipes::recipe::Recipe>::SCHEDULE,
                args_schema_fn: || {
                    $crate::recipes::recipe::schema_of::<
                        <$ty as $crate::recipes::recipe::Recipe>::Args,
                    >()
                },
                compile_fn: |raw| $crate::recipes::recipe::compile_erased::<$ty>(raw),
            };
    };
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
    /// Optional default `systemd` `OnCalendar` expression. When set,
    /// `blut schedule install <recipe>` uses it if no `--calendar` is
    /// given. `None` = the recipe ships no built-in schedule (E1↔E5).
    pub schedule: Option<&'static str>,
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
        schedule: None,
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
    fn schema_of_builds_an_object_schema() {
        // `schema_of` is the body every `RecipeDef.args_schema_fn` now
        // references through `register_recipe!`. The full macro expansion +
        // `compile_erased` are exercised end-to-end by each cookbook's recipe
        // tests (they build real Plans); here we pin the standalone helper.
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Args {
            lr: f64,
            epochs: u32,
        }
        let schema = schema_of::<Args>();
        assert!(schema.is_object(), "args schema must be a JSON object");
        // The fix: definitions are now attached (was a dangling $ref), so the
        // TUI/registry can resolve `#/definitions/Args` → properties.
        let props = schema
            .get("definitions")
            .and_then(|d| d.get("Args"))
            .and_then(|a| a.get("properties"))
            .and_then(|p| p.as_object())
            .expect("schema must carry definitions/Args/properties");
        assert!(props.contains_key("lr") && props.contains_key("epochs"));
    }

    #[test]
    fn category_label_is_stable() {
        assert_eq!(RecipeCategory::Train.label(), "TRAINING");
        assert_eq!(RecipeCategory::DataPrep.label(), "DATA PREPARATION");
    }
}
