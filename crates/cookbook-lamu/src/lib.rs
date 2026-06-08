//! **blut-lamu** — the lamu cookbook for BLUT (generic LLM training:
//! SFT / DPO / distillation / eval over the llama.cpp + HuggingFace
//! Trainer backends).
//!
//! TRANSITIONAL in-tree skeleton (see the sibling cookbook-lamquant).
//! Re-exports the lamu cookbook whose recipes / stages / artifacts /
//! backends still live inside `blut` (C1); the atomic move HERE is C2b,
//! the split to a standalone `blut-lamu` repo is C2c. See
//! `[[project_blut_cookbook_split]]`.

pub use blut::framework::{Cookbook, LamuCookbook, Registry};

/// A [`Registry`] containing exactly the lamu cookbook — what the
/// `blut-lamu` binary will register before handing off to the BLUT CLI.
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
}
