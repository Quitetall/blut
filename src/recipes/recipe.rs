// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
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
    /// Defaulted so the trait extension stays additive; every shipped recipe
    /// overrides it in its `impl Recipe` ([`register_recipe!`] just copies it
    /// into the `RecipeDef`).
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
    let mut v = serde_json::to_value(root).unwrap_or_else(|e| {
        panic!(
            "schema for {} must serialize: {e}",
            std::any::type_name::<A>()
        )
    });
    // A `#[derive(JsonSchema)]` struct/enum is *referenceable*, so
    // `subschema_for::<A>()` always parks A's own schema in the generator under
    // its name and returns `{"$ref":"#/definitions/<A>"}`. Attach the collected
    // definitions so that `$ref` resolves (the bug the hand-DEFs left dangling).
    // `expect`, not swallow — dropping defs here would re-create the dangling ref.
    let defs = g.take_definitions();
    if !defs.is_empty() {
        let defs_val = serde_json::to_value(&defs).unwrap_or_else(|e| {
            panic!(
                "definitions for {} must serialize: {e}",
                std::any::type_name::<A>()
            )
        });
        if let serde_json::Value::Object(map) = &mut v {
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
    // Schema PREFLIGHT (B / P5): validate `raw` against the recipe's own
    // schema BEFORE serde deserializes it — so a missing required arg or a
    // wrong-typed field fails with a FIELD-NAMED message (`arg 'lr' expected
    // number, got string`) instead of serde's positional `expected f64 at
    // line 1 column 17`. serde's `from_value` still backstops deeper structure.
    let schema = schema_of::<R::Args>();
    validate_args_against_schema(R::NAME, &schema, &raw)?;
    // Prefix the recipe name so a bad-args error names the culprit. This was
    // hand-done in only 2 of the migrated recipes; centralizing here gives the
    // prefix to every recipe uniformly.
    //
    // ZERO-ARG RECIPES: a recipe that takes no args declares a unit-struct
    // `Args` (`struct Foo;`), which serde deserializes from `null` — but the CLI
    // / TUI default missing args to an empty object `{}`. Coerce `{}` → `null` on
    // failure so `recipe run <name>` works with no `--args` for such recipes.
    let args: R::Args = serde_json::from_value(raw.clone())
        .or_else(|e| {
            if raw.as_object().is_some_and(|o| o.is_empty()) {
                serde_json::from_value(serde_json::Value::Null)
            } else {
                Err(e)
            }
        })
        .map_err(|e| RecipeError::InvalidArgs {
            field: None,
            message: format!("{}: {e}", R::NAME),
        })?;
    R::default().compile(args).map(|p| p.into_compiled())
}

/// Whether a JSON value satisfies a schema `type` keyword. `integer` accepts
/// only whole numbers; `number` accepts any. A `null` is tolerated (an
/// `Option<T>` field schemars-types as `["string","null"]`, which arrives here
/// as a non-string `type` and is skipped — see the caller).
fn json_type_matches(v: &serde_json::Value, expected: &str) -> bool {
    use serde_json::Value;
    match expected {
        "string" => v.is_string(),
        "boolean" => v.is_boolean(),
        // Accept a whole-valued float (`8.0`) too — serde happily takes it for a
        // u32, so the preflight must not be STRICTER than serde (a false reject).
        "integer" => matches!(v, Value::Number(n)
            if n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0)),
        "number" => v.is_number(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        "null" => v.is_null(),
        _ => true, // unknown keyword → don't reject (serde backstops)
    }
}

/// Field-named preflight: every `required` field is present, and every supplied
/// field whose schema declares a scalar `type` matches it. Deliberately shallow
/// (recipe `Args` are flat structs) — it catches the common operator mistakes
/// (typo'd / missing / wrong-typed arg) with a precise message; serde validates
/// the rest. A `null` value or a field schemars typed as a `["T","null"]` union
/// (Option) is not type-checked here (it's nullable by construction).
///
/// `pub(crate)` so the TUI Editor (F2/U3) runs the SAME preflight before it
/// launches a recipe — a non-AI human gets the precise message in-TUI instead
/// of a detached job that fails deep in the executor.
pub(crate) fn validate_args_against_schema(
    recipe: &str,
    schema: &serde_json::Value,
    raw: &serde_json::Value,
) -> Result<(), RecipeError> {
    let Some(root) = schema_root(schema) else {
        return Ok(()); // no resolvable schema → let serde handle it
    };
    let bad = |m: String| {
        Err(RecipeError::InvalidArgs {
            field: None,
            message: format!("{recipe}: {m}"),
        })
    };
    let Some(obj) = raw.as_object() else {
        // A unit/`()` Args serializes as null; only object-args reach here.
        return if raw.is_null() {
            Ok(())
        } else {
            bad("args must be a JSON object".into())
        };
    };
    if let Some(req) = root.get("required").and_then(|r| r.as_array()) {
        for field in req.iter().filter_map(|v| v.as_str()) {
            if !obj.contains_key(field) {
                return bad(format!("missing required arg '{field}'"));
            }
        }
    }
    if let Some(props) = root.get("properties").and_then(|p| p.as_object()) {
        for (k, v) in obj {
            // Only check a declared field with a SCALAR `type` string; a union
            // type (Option) has `type: [..]` (not a str) → skipped.
            if let Some(expected) = props
                .get(k)
                .and_then(|d| d.get("type"))
                .and_then(|t| t.as_str())
            {
                if !v.is_null() && !json_type_matches(v, expected) {
                    return bad(format!(
                        "arg '{k}' expected {expected}, got {}",
                        json_kind(v)
                    ));
                }
            }
        }
    }
    Ok(())
}

fn json_kind(v: &serde_json::Value) -> &'static str {
    use serde_json::Value;
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Resolve the root args object of a `schema_of`-shaped schema
/// (`{"$ref":"#/definitions/<Name>","definitions":{...}}`) to the `<Name>`
/// definition object. Falls back to the conventional `"Args"` key.
fn schema_root(schema: &serde_json::Value) -> Option<&serde_json::Map<String, serde_json::Value>> {
    let defs = schema.get("definitions")?.as_object()?;
    let name = schema
        .get("$ref")
        .and_then(|r| r.as_str())
        .and_then(|r| r.rsplit('/').next())
        .unwrap_or("Args");
    defs.get(name)
        .or_else(|| defs.get("Args"))
        .and_then(|d| d.as_object())
}

/// Type-aware placeholder for a required field that declares no default.
fn placeholder_for(prop: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    let ty = prop.get("type").and_then(|t| t.as_str());
    match ty {
        Some("number") | Some("integer") => Value::Number(0.into()),
        Some("boolean") => Value::Bool(false),
        Some("array") => Value::Array(vec![]),
        Some("object") => Value::Object(serde_json::Map::new()),
        // string + unknown (incl. Option-typed `["string","null"]`) → TODO.
        _ => Value::String("<TODO>".into()),
    }
}

/// Build a starting-point args object for a recipe straight from its schema
/// (E2): every field with a non-null `default` (the `#[serde(default=…)]`
/// source-of-truth schemars embeds) gets that default; every REQUIRED field
/// without a default gets a type-aware `<TODO>` placeholder. This is the
/// single defaults source — cookbooks no longer hand-duplicate them; a
/// cookbook's `default_args` overlay only adds domain paths + curated
/// non-default starts on top (see `Registry::prefill_args`).
pub fn args_template(def: &RecipeDef) -> serde_json::Value {
    args_template_from_schema(&(def.args_schema_fn)())
}

/// The same starting-args template as [`args_template`], but from a raw JSON
/// schema `Value` rather than a `RecipeDef` — so the console DAG builder can
/// prefill a STAGE's args (via `StageDyn::args_schema`) exactly the way the
/// recipe editor prefills a recipe's.
pub fn args_template_from_schema(schema: &serde_json::Value) -> serde_json::Value {
    use serde_json::{Map, Value};
    let Some(root) = schema_root(schema) else {
        return Value::Object(Map::new());
    };
    let required: std::collections::HashSet<&str> = root
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let mut out = Map::new();
    if let Some(props) = root.get("properties").and_then(|p| p.as_object()) {
        for (k, v) in props {
            match v.get("default") {
                Some(d) if !d.is_null() => {
                    out.insert(k.clone(), d.clone());
                }
                // A required field never carries a serde default (they're
                // mutually exclusive), so this only fires for genuinely
                // user-supplied fields (the domain paths).
                _ if required.contains(k.as_str()) => {
                    out.insert(k.clone(), placeholder_for(v));
                }
                _ => {}
            }
        }
    }
    Value::Object(out)
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
///
/// One recipe per module: the macro emits an unnamespaced `pub static DEF`, so
/// a second invocation in the same module is a duplicate-symbol error. This
/// matches the cookbook layout (one recipe file = one `DEF`).
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
/// "PRETRAINING", "TRAINING", "EVALUATION", "GATE", "EXPORT",
/// "PIPELINE", "USER").
/// Courses are also used by `blut recipe list --category <…>`
/// for filtered CLI browsing.
///
/// "Course" is the culinary cookbook-taxonomy grouping layer
/// (ADR 0051: BLUT → Cookbook → Course → Recipe → Ingredient) — a
/// static tag, not a registered object. `RecipeCategory` remains as a
/// back-compat alias below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Course {
    /// Recipes that prepare / convert / index raw input → typed
    /// artifacts (e.g. a data-prep recipe packs a corpus into a typed
    /// artifact).
    DataPrep,
    /// Self-supervised / unlabelled pretraining that produces a
    /// backbone consumed by a later `Train` course (e.g. a masked-
    /// autoencoder or SSL pretrain stage).
    Pretrain,
    /// Recipes that consume artifacts + produce checkpoints
    /// (`Checkpoint`, `HfCheckpoint`, etc.).
    Train,
    /// Recipes that consume checkpoints + produce `EvalReport`.
    Eval,
    /// Acceptance-gate course — the fail-closed accept/reject
    /// asset-check run after evaluation (ADR 0037 asset-check ≡ gate).
    Gate,
    /// Recipes that take a checkpoint + materialize a deployable
    /// artifact (`HardenedCkpt`, `FirmwareBundle`, `GgufModel`).
    Export,
    /// Recipes that chain multiple courses end-to-end (e.g. a
    /// pipeline recipe = data prep → train → gate).
    Pipeline,
    /// User-authored recipes from `blut/src/recipes/user/`.
    User,
}

impl Course {
    /// Menu position in pipeline (lifecycle-phase) order, used to group +
    /// sort recipes in `blut tui` and `blut recipe list`. The match is
    /// exhaustive, so a newly-added course cannot compile without being
    /// given a position here — there is no silent "sorts last" fallback.
    pub fn order(self) -> u8 {
        match self {
            Self::DataPrep => 0,
            Self::Pretrain => 1,
            Self::Train => 2,
            Self::Eval => 3,
            Self::Gate => 4,
            Self::Export => 5,
            Self::Pipeline => 6,
            Self::User => 7,
        }
    }

    /// Human-readable section header used by `blut tui` + `blut
    /// recipe list`.
    pub fn label(self) -> &'static str {
        match self {
            Self::DataPrep => "DATA PREPARATION",
            Self::Pretrain => "PRETRAINING",
            Self::Train => "TRAINING",
            Self::Eval => "EVALUATION",
            Self::Gate => "GATE",
            Self::Export => "EXPORT",
            Self::Pipeline => "PIPELINE",
            Self::User => "USER",
        }
    }
}

/// Back-compat alias. The canonical name is [`Course`] (ADR 0051 — the
/// culinary cookbook taxonomy BLUT → Cookbook → Course → Recipe →
/// Ingredient). Existing `RecipeCategory` references keep compiling; new
/// code should prefer `Course`.
pub type RecipeCategory = Course;

/// Erased registry entry. Stored in the static `RECIPES` slice.
pub struct RecipeDef {
    pub name: &'static str,
    pub description: &'static str,
    /// Backend identity (e.g. "hf_trainer", "my_backend").
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
    fn validate_args_catches_missing_required_and_bad_types() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Args {
            lr: f64,
            epochs: u32,
        }
        let schema = schema_of::<Args>();
        // Valid → Ok.
        assert!(
            validate_args_against_schema(
                "r",
                &schema,
                &serde_json::json!({"lr": 0.1, "epochs": 8})
            )
            .is_ok()
        );
        // Missing a required field → field-named error.
        let e = validate_args_against_schema("r", &schema, &serde_json::json!({"lr": 0.1}))
            .unwrap_err()
            .to_string();
        assert!(e.contains("epochs"), "names the missing field: {e}");
        // Wrong type → field-named error mentioning the field + expectation.
        let e = validate_args_against_schema(
            "r",
            &schema,
            &serde_json::json!({"lr": "fast", "epochs": 8}),
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("lr") && e.contains("number"),
            "names the field + type: {e}"
        );
        // A unit/`()`-args recipe serializes its args as null → tolerated.
        let empty = serde_json::json!({"type": "object", "properties": {}});
        assert!(validate_args_against_schema("r", &empty, &serde_json::Value::Null).is_ok());
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
    fn schema_of_attaches_definitions_even_for_primitive_only_struct() {
        // Refutes the "flat struct → no definitions → dangling $ref" concern:
        // a `#[derive(JsonSchema)]` struct is itself referenceable, so it lands
        // in definitions regardless of whether its FIELDS are primitives.
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Flat {
            name: String,
            enabled: bool,
            count: i32,
        }
        let schema = schema_of::<Flat>();
        let defs = schema
            .get("definitions")
            .and_then(|d| d.as_object())
            .expect("definitions must be attached for a referenceable struct");
        let flat = defs
            .get("Flat")
            .and_then(|a| a.get("properties"))
            .and_then(|p| p.as_object())
            .expect("definitions/Flat/properties must resolve the $ref");
        assert!(flat.contains_key("name") && flat.contains_key("enabled"));
    }

    #[test]
    fn args_template_harvests_defaults_and_placeholders() {
        fn default_preset() -> String {
            "production".into()
        }
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Args {
            #[serde(default = "default_preset")]
            preset: String,
            #[serde(default)]
            epochs: Option<u32>, // null default → omitted (noise)
            lma_root: String, // required, no default → placeholder
        }
        static D: RecipeDef = RecipeDef {
            name: "t",
            description: "d",
            backend_id: "b",
            category: RecipeCategory::Train,
            input_kinds: &[],
            output_kind: "k",
            schedule: None,
            args_schema_fn: || schema_of::<Args>(),
            compile_fn: |_| Err(RecipeError::CompileFailed("x".into())),
        };
        let t = args_template(&D);
        assert_eq!(t["preset"], serde_json::json!("production"));
        assert_eq!(t["lma_root"], serde_json::json!("<TODO>"));
        assert!(
            t.get("epochs").is_none(),
            "null defaults are omitted as noise"
        );
    }

    #[test]
    fn category_label_is_stable() {
        assert_eq!(Course::Train.label(), "TRAINING");
        assert_eq!(Course::DataPrep.label(), "DATA PREPARATION");
        assert_eq!(Course::Pretrain.label(), "PRETRAINING");
        assert_eq!(Course::Gate.label(), "GATE");
    }

    /// `order()` and `label()` are both exhaustive matches, so the compiler
    /// forces every course to have a menu position + header — a new variant
    /// cannot drop silently. This only checks the values are sane: orders
    /// are distinct + contiguous (0..8) and labels are unique.
    #[test]
    fn course_order_and_label_are_well_formed() {
        let courses = [
            Course::DataPrep,
            Course::Pretrain,
            Course::Train,
            Course::Eval,
            Course::Gate,
            Course::Export,
            Course::Pipeline,
            Course::User,
        ];
        let mut orders: Vec<u8> = courses.iter().map(|c| c.order()).collect();
        orders.sort_unstable();
        assert_eq!(
            orders,
            (0..8).collect::<Vec<u8>>(),
            "course orders must be distinct + contiguous 0..8"
        );
        let mut labels: Vec<&str> = courses.iter().map(|c| c.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), 8, "every course must have a unique label");
    }
}
