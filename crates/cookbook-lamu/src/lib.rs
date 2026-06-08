//! **blut-lamu** — the lamu cookbook for BLUT (generic LLM training:
//! SFT / DPO / distillation / eval over the llama.cpp + HuggingFace
//! Trainer backends).
//!
//! Owns the 6 generic-LLM recipes (`finetune_from_*`, `dpo_from_*`,
//! `eval_suite`, `distill_from_teacher`, `hf_finetune_from_dataset`).
//! Depends on the `blut` framework crate for the domain-agnostic engine
//! (Stage / Plan / Recipe / Registry / Cookbook) AND for the generic
//! stages / backends / artifacts the recipes wire (`blut::stages`,
//! `blut::backends`, `blut::artifacts`) — those are engine primitives
//! the BLUT CLI also drives, so they STAY in blut-core.
//!
//! TRANSITIONAL in-tree workspace member: splits to a standalone
//! `blut-lamu` repo at C2c. The atomic move that pulled this source out
//! of `blut` core (so `blut` ships zero recipes + zero concrete
//! `Cookbook` impls) was C2b. See `[[project_blut_cookbook_split]]`.

pub mod recipes;

pub use blut::framework::{Cookbook, Registry};

use blut::recipes::recipe::RecipeDef;

/// Recipes owned by the **blut-lamu** cookbook (generic LLM: lamu +
/// hf_trainer backends). `hf_finetune_from_dataset` stays LAST (mirrors
/// the historical catalog order). Moved out of blut-core's
/// `recipes::recipe::LAMU_RECIPES` at C2b.
pub static LAMU_RECIPES: &[&RecipeDef] = &[
    &recipes::finetune_from_conversations::DEF,
    &recipes::finetune_from_dataset::DEF,
    &recipes::dpo_from_preferences::DEF,
    &recipes::eval_suite::DEF,
    &recipes::distill_from_teacher::DEF,
    &recipes::hf_finetune_from_dataset::DEF,
];

/// The lamu cookbook (generic LLM: SFT / DPO / distill / eval over the
/// lamu + hf_trainer backends): the 6 recipes plus the (blut-core)
/// stages / artifacts / backends they reference. The unit this crate
/// hands BLUT — `registry()` registers it and the BLUT CLI / TUI compose
/// its recipes at runtime via [`Registry`]. (Moved out of blut-core's
/// `framework::cookbook` at C2b.)
pub struct LamuCookbook;

impl Cookbook for LamuCookbook {
    fn name(&self) -> &'static str {
        "lamu"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        LAMU_RECIPES
    }
}

/// A [`Registry`] containing exactly the lamu cookbook — what the
/// `blut-lamu` binary registers before handing off to the BLUT CLI.
pub fn registry() -> Registry {
    let mut r = Registry::new();
    r.register(Box::new(LamuCookbook));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lamu_cookbook_registers_and_lists_recipes() {
        let r = registry();
        assert!(r.find("finetune_from_dataset").is_some());
        assert!(r.all().count() >= 6, "expected the 6 lamu recipes");
    }

    #[test]
    fn cookbook_name_is_lamu() {
        assert_eq!(LamuCookbook.name(), "lamu");
    }

    #[test]
    fn hf_finetune_is_last() {
        // Mirrors the historical catalog order (positional contract from
        // when the TUI indexed the static slice).
        assert_eq!(
            LAMU_RECIPES.last().map(|r| r.name),
            Some("hf_finetune_from_dataset")
        );
    }
}
