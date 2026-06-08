//! Cookbook seam (C1 — registry is the live catalog source).
//!
//! A "cookbook" is the unit a domain hands BLUT (ADR 0034 / ADR 0037):
//! a bundle of recipes plus the stages / artifacts they reference. The
//! CLI now reads its recipe catalog from [`default_registry`] (the
//! union of the registered cookbooks), NOT the static `RECIPES` slice
//! directly. Two real cookbooks are registered — [`LamuCookbook`] and
//! [`LamquantCookbook`] — both still living in-crate (TRANSITIONAL).
//! [`BuiltinCookbook`] (wrapping the full static slice) is retained for
//! the TUI + back-compat parity tests until the cookbooks move out.
//!
//! Remaining lanes (`[[project_blut_cookbook_split]]`): move each
//! cookbook (recipes + stages + artifacts + backend + python) to its
//! own crate/repo (`blut-lamquant`, `blut-lamu`) so blut-core has ZERO
//! domain symbols; populate [`Cookbook::stages`] / [`Cookbook::artifacts`]
//! (today DESCRIPTORS only — name / kind / schema, not executable
//! handles); rewire the TUI off the static `RECIPES` indices.

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

/// The LamQuant cookbook (neural EEG codec): the 7 `lamquant_*` recipes.
/// TRANSITIONAL in-crate home — the real cookbook (recipes + stages +
/// artifacts + backend + `python/lamquant/`) moves to the dedicated
/// `blut-lamquant` crate/repo (C2a), at which point this struct and its
/// recipe references leave blut-core entirely. See
/// `[[project_blut_cookbook_split]]`.
pub struct LamquantCookbook;

impl Cookbook for LamquantCookbook {
    fn name(&self) -> &'static str {
        "lamquant"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        crate::recipes::recipe::LAMQUANT_RECIPES
    }
}

/// The lamu cookbook (generic LLM: SFT / DPO / distill / eval over the
/// lamu + hf_trainer backends). TRANSITIONAL in-crate home — moves to
/// the `blut-lamu` crate/repo (C2b).
pub struct LamuCookbook;

impl Cookbook for LamuCookbook {
    fn name(&self) -> &'static str {
        "lamu"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        crate::recipes::recipe::LAMU_RECIPES
    }
}

/// Runtime registry that ingests cookbooks: holds boxed cookbooks and
/// exposes `find` / `by_category` / `all` over their union. This is the
/// live catalog source for the CLI (`main.rs` builds one via
/// [`default_registry`] per command); the TUI still indexes the static
/// `RECIPES` slice (rewire is a follow-up lane).
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

    pub fn by_category(
        &self,
        cat: RecipeCategory,
    ) -> impl Iterator<Item = &'static RecipeDef> + '_ {
        self.all().filter(move |r| r.category == cat)
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::with_builtin()
    }
}

/// The cookbooks this binary ships with — the CLI reads the recipe
/// catalog from here, not the static `RECIPES` slice. TODAY both
/// cookbooks live in-crate (transitional); once they move to their own
/// crates (C2a/C2b) the binary composes this by registering each
/// compiled-in cookbook, and a bare blut-core binary ships an EMPTY
/// registry (the pure engine). Registration order = catalog order
/// before the CLI's own sort.
pub fn default_registry() -> Registry {
    let mut r = Registry::new();
    r.register(Box::new(LamuCookbook));
    r.register(Box::new(LamquantCookbook));
    r
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
        assert!(
            r.by_category(RecipeCategory::Train)
                .any(|r| r.name == "lamquant_joint_codec")
        );
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

    #[test]
    fn default_registry_composes_full_catalog() {
        use std::collections::BTreeSet;
        // The two real cookbooks (lamu + lamquant) must compose to
        // exactly the full static catalog — same names, no overlap, no
        // gap. This is the C1 invariant: the catalog = union of
        // registered cookbooks.
        let reg = default_registry();
        let composed: BTreeSet<&str> = reg.all().map(|r| r.name).collect();
        let union: BTreeSet<&str> = RECIPES.iter().map(|r| r.name).collect();
        // The set comparison is the DISJOINTNESS guard: a name duplicated
        // across both sub-slices collapses in `composed` but not in the
        // length sum, so set-equality would fail.
        assert_eq!(composed, union, "cookbooks must cover RECIPES exactly");
        // The length sum is the COMPLETENESS guard (no recipe in a
        // sub-slice that's missing from RECIPES, and vice-versa).
        assert_eq!(
            recipe::LAMU_RECIPES.len() + recipe::LAMQUANT_RECIPES.len(),
            RECIPES.len(),
            "LAMU + LAMQUANT must partition RECIPES (disjoint + complete)"
        );
        // Cookbook identities resolve.
        assert_eq!(LamquantCookbook.name(), "lamquant");
        assert_eq!(LamuCookbook.name(), "lamu");
    }
}
