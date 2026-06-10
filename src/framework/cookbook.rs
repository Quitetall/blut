//! Cookbook seam (the engine's recipe-registry mechanism).
//!
//! A "cookbook" is the unit a domain hands BLUT (ADR 0034 / ADR 0037):
//! a bundle of recipes plus the stages / artifacts they reference.
//! blut-core ships ZERO concrete cookbooks and ZERO recipes — it owns
//! only the [`Cookbook`] trait + the [`Registry`] that composes the live
//! catalog. The cookbook crates (`cookbook-lamu`, moved out at C2b;
//! `cookbook-lamquant`, moved out at C2a) define concrete `Cookbook`
//! impls and register their recipes at runtime; the binary in each
//! cookbook crate builds the registry and hands it to [`crate::cli::run`].
//!
//! Remaining lanes (`[[project_blut_cookbook_split]]`): split each
//! cookbook crate to its own repo (C2c); populate [`Cookbook::stages`] /
//! [`Cookbook::artifacts`] (today DESCRIPTORS only — name / kind /
//! schema, not executable handles).

use crate::recipes::recipe::{RecipeCategory, RecipeDef};

/// Lightweight descriptor for a stage a cookbook declares. Name /
/// kind / schema only — NOT an executable handle (extraction
/// deferred).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageDescriptor {
    pub name: &'static str,
    pub input_kind: &'static str,
    pub output_kind: &'static str,
    pub schema: u32,
}

/// Descriptor for an artifact KIND a cookbook's stages produce /
/// consume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactDescriptor {
    pub kind: &'static str,
    pub schema: u32,
}

/// A cookbook is a self-contained bundle of recipes (plus the stages /
/// artifacts they reference). The seam for ADR-0034 "BLUT owns
/// recipes / artifacts". blut-core defines the trait; concrete impls
/// live in cookbook crates.
pub trait Cookbook: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn recipes(&self) -> &'static [&'static RecipeDef];
    /// Descriptors only (deferred: executable dispatch). Default empty
    /// so a cookbook can declare recipes without yet enumerating its
    /// stages.
    fn stages(&self) -> &'static [StageDescriptor] {
        &[]
    }
    /// Descriptors only (deferred). Default empty.
    fn artifacts(&self) -> &'static [ArtifactDescriptor] {
        &[]
    }
    /// Pre-baked args JSON for one of this cookbook's recipes (domain
    /// data — e.g. default corpus paths), used to prefill the TUI args
    /// editor. Default `None` so non-domain cookbooks need no impl; the
    /// TUI falls back to the schemars template. Keeps domain paths OUT of
    /// blut-core (they live with the cookbook).
    fn default_args(&self, _recipe: &str) -> Option<String> {
        None
    }
}

/// Runtime registry that ingests cookbooks: holds boxed cookbooks and
/// exposes `find` / `by_category` / `all` over their union. This is the
/// live catalog source for the CLI (the cookbook binary builds one and
/// passes it to [`crate::cli::run`]) and the TUI (which indexes the
/// composed catalog, not any static slice).
pub struct Registry {
    cookbooks: Vec<Box<dyn Cookbook>>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            cookbooks: Vec::new(),
        }
    }

    pub fn register(&mut self, c: Box<dyn Cookbook>) {
        self.cookbooks.push(c);
    }

    pub fn all(&self) -> impl Iterator<Item = &'static RecipeDef> + '_ {
        self.cookbooks
            .iter()
            .flat_map(|c| c.recipes().iter().copied())
    }

    pub fn find(&self, name: &str) -> Option<&'static RecipeDef> {
        self.all().find(|r| r.name == name)
    }

    pub fn by_category(
        &self,
        cat: RecipeCategory,
    ) -> impl Iterator<Item = &'static RecipeDef> + '_ {
        self.all().filter(move |r| r.category == cat)
    }

    /// Pre-baked args JSON for a recipe, from whichever registered
    /// cookbook owns it (first match wins). `None` if no cookbook
    /// supplies defaults — the caller falls back to the schema template.
    pub fn default_args(&self, recipe: &str) -> Option<String> {
        self.cookbooks.iter().find_map(|c| c.default_args(recipe))
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test-only `RecipeDef` fixture (domain-free) so the registry
    /// composition logic can be exercised without any concrete cookbook.
    static FIXTURE_A: RecipeDef = RecipeDef {
        name: "alpha_train",
        description: "fixture train recipe",
        backend_id: "fixture",
        category: RecipeCategory::Train,
        input_kinds: &["dataset.jsonl"],
        output_kind: "checkpoint.hf",
        schedule: None,
        args_schema_fn: || serde_json::json!({"type": "object", "properties": {}}),
        compile_fn: |_| {
            Err(crate::framework::error::RecipeError::CompileFailed(
                "fixture".into(),
            ))
        },
    };
    static FIXTURE_B: RecipeDef = RecipeDef {
        name: "beta_eval",
        description: "fixture eval recipe",
        backend_id: "fixture",
        category: RecipeCategory::Eval,
        input_kinds: &["checkpoint.hf"],
        output_kind: "eval.report",
        schedule: None,
        args_schema_fn: || serde_json::json!({"type": "object", "properties": {}}),
        compile_fn: |_| {
            Err(crate::framework::error::RecipeError::CompileFailed(
                "fixture".into(),
            ))
        },
    };
    static FIXTURES: &[&RecipeDef] = &[&FIXTURE_A, &FIXTURE_B];

    /// A test-only cookbook wrapping the fixture recipes. blut-core ships
    /// no concrete cookbook, so the registry tests bring their own.
    struct FixtureCookbook;
    impl Cookbook for FixtureCookbook {
        fn name(&self) -> &'static str {
            "fixture"
        }
        fn recipes(&self) -> &'static [&'static RecipeDef] {
            FIXTURES
        }
    }

    fn registry() -> Registry {
        let mut r = Registry::new();
        r.register(Box::new(FixtureCookbook));
        r
    }

    #[test]
    fn empty_registry_has_no_recipes() {
        let r = Registry::new();
        assert_eq!(r.all().count(), 0, "blut-core ships no recipes");
        assert!(r.find("anything").is_none());
    }

    #[test]
    fn registry_finds_registered_recipe() {
        let r = registry();
        assert!(r.find("alpha_train").is_some());
        assert!(r.find("nope").is_none());
    }

    #[test]
    fn registry_by_category_filters() {
        let r = registry();
        assert!(
            r.by_category(RecipeCategory::Train)
                .any(|r| r.name == "alpha_train")
        );
        assert!(
            r.by_category(RecipeCategory::Eval)
                .any(|r| r.name == "beta_eval")
        );
        assert!(
            !r.by_category(RecipeCategory::Eval)
                .any(|r| r.name == "alpha_train")
        );
    }

    #[test]
    fn registry_all_composes_union() {
        use std::collections::BTreeSet;
        let r = registry();
        let composed: BTreeSet<&str> = r.all().map(|r| r.name).collect();
        let expected: BTreeSet<&str> = FIXTURES.iter().map(|r| r.name).collect();
        assert_eq!(composed, expected);
    }

    #[test]
    fn descriptors_default_empty() {
        // Documents the deferred stage/artifact extraction.
        assert!(FixtureCookbook.stages().is_empty());
        assert!(FixtureCookbook.artifacts().is_empty());
    }

    #[test]
    fn default_args_defaults_to_none() {
        // A cookbook with no override returns None (TUI falls back to the
        // schemars template).
        let r = registry();
        assert!(r.default_args("alpha_train").is_none());
    }
}
