//! **blut-lamquant** — the LamQuant cookbook for BLUT (neural EEG codec
//! training: encoder / oracle / SNN / decoder / joint → PCCP gate).
//!
//! TRANSITIONAL in-tree skeleton. Depends on the `blut` framework crate
//! and, for now, re-exports the LamQuant cookbook whose recipes / stages
//! / artifacts / backend still live inside `blut` (C1). The atomic move
//! that pulls that source HERE (so `blut` core holds zero LamQuant
//! symbols) is C2a; the split to a standalone `blut-lamquant` repo is
//! C2c. See `[[project_blut_cookbook_split]]`.

pub use blut::framework::{Cookbook, LamquantCookbook, Registry};

/// A [`Registry`] containing exactly the LamQuant cookbook — what the
/// `blut-lamquant` binary will register before handing off to the BLUT
/// CLI. (The binary relocation is part of C2a/C2c.)
pub fn registry() -> Registry {
    let mut r = Registry::new();
    r.register(Box::new(LamquantCookbook));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lamquant_cookbook_registers_and_lists_recipes() {
        let r = registry();
        // The cookbook must expose the LamQuant recipes (joint codec is
        // the canonical training entry).
        assert!(r.find("lamquant_joint_codec").is_some());
        assert!(r.all().count() >= 7, "expected the 7 lamquant recipes");
    }
}
