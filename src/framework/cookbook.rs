//! Cookbook seam (SKELETON).
//!
//! The runtime-registry boundary that the static `RECIPES` catalog
//! (`recipes/recipe.rs`) will eventually populate FROM. Today: one
//! [`BuiltinCookbook`] wrapping the existing static slice, plus a
//! [`Registry`] that ingests cookbooks. Stage / artifact extraction
//! is DEFERRED — this module designs the seam, not the move.
//!
//! Per ADR 0034 (BLUT owns recipes / artifacts), a "cookbook" is the
//! unit a domain hands BLUT: a bundle of recipes plus the stages /
//! artifacts they reference. The [`Cookbook::stages`] /
//! [`Cookbook::artifacts`] methods return DESCRIPTORS only (name /
//! kind / schema) — they are not yet executable handles. Making them
//! dispatchable, and swapping the CLI from `recipe::find` to
//! `Registry::find`, are follow-up lanes.

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
/// recipes / artifacts".
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
}

/// The built-in cookbook: wraps the existing static `RECIPES`
/// verbatim. Zero behaviour change — `recipe::find` / `by_category`
/// still read the static slice today; this is the SEAM they CAN
/// delegate to later.
pub struct BuiltinCookbook;

impl Cookbook for BuiltinCookbook {
    fn name(&self) -> &'static str {
        "builtin"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        crate::recipes::recipe::RECIPES
    }
}

/// Runtime registry that ingests cookbooks. SKELETON: holds boxed
/// cookbooks and exposes `find` / `by_category` / `all` over their
/// union. NOT yet wired into the CLI (`main.rs` still calls
/// `recipe::find`) — that swap is a follow-up lane.
pub struct Registry {
    cookbooks: Vec<Box<dyn Cookbook>>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            cookbooks: Vec::new(),
        }
    }

    /// Seed with the built-in cookbook.
    pub fn with_builtin() -> Self {
        let mut r = Self::new();
        r.register(Box::new(BuiltinCookbook));
        r
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

    pub fn by_category(&self, cat: RecipeCategory) -> impl Iterator<Item = &'static RecipeDef> + '_ {
        self.all().filter(move |r| r.category == cat)
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::with_builtin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::recipe::{self, RECIPES};

    #[test]
    fn builtin_lists_all_static_recipes() {
        assert_eq!(BuiltinCookbook.recipes().len(), RECIPES.len());
    }

    #[test]
    fn registry_with_builtin_finds_known() {
        let r = Registry::with_builtin();
        assert!(r.find("lamquant_encoder").is_some());
        assert!(r.find("nope").is_none());
    }

    #[test]
    fn registry_by_category_filters() {
        // Train category must include the standalone joint-codec recipe.
        let r = Registry::with_builtin();
        assert!(r
            .by_category(RecipeCategory::Train)
            .any(|r| r.name == "lamquant_joint_codec"));
    }

    #[test]
    fn registry_find_matches_static_find() {
        // Non-lossy seam parity: for every catalog entry, the registry
        // resolves the same RecipeDef that `recipe::find` does.
        let reg = Registry::with_builtin();
        for r in RECIPES {
            let via_reg = reg.find(r.name).expect("registry must find catalog entry");
            let via_static = recipe::find(r.name).expect("recipe::find must find catalog entry");
            assert_eq!(via_reg.name, r.name);
            assert_eq!(via_reg.name, via_static.name);
        }
    }

    #[test]
    fn descriptors_default_empty() {
        // Documents the deferred stage/artifact extraction.
        assert!(BuiltinCookbook.stages().is_empty());
        assert!(BuiltinCookbook.artifacts().is_empty());
    }
}
