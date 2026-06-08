//! **blut-lamquant** — the LamQuant cookbook for BLUT (neural EEG codec
//! training: encoder / oracle / SNN / decoder / joint → PCCP gate).
//!
//! Owns the LamQuant recipes / stages / artifacts / backend + the
//! multi-root path map (`paths::LamquantRoots`) + the python training
//! payload (`python/lamquant/`). Depends on the `blut` framework crate
//! for the domain-agnostic engine (Stage / Plan / Recipe / Registry /
//! Cookbook + the generic path primitives `meta_repo_root` /
//! `validate_holds` / `join_existing`).
//!
//! TRANSITIONAL in-tree workspace member: splits to a standalone
//! `blut-lamquant` repo at C2c. The atomic move that pulled this source
//! out of `blut` core (so `blut` holds zero LamQuant symbols) was C2a.
//! See `[[project_blut_cookbook_split]]`.

pub mod artifacts;
pub mod backends;
pub mod paths;
pub mod recipes;
pub mod stages;

pub use blut::framework::{Cookbook, LamuCookbook, Registry};
pub use recipes::LAMQUANT_RECIPES;

use blut::recipes::recipe::RecipeDef;

/// Process-wide lock for cookbook tests that mutate environment
/// variables (the `BLUT_*` root overrides). This crate's test binary is
/// a separate process from blut's, so a crate-local lock is correct —
/// blut's `TEST_ENV_LOCK` is `pub(crate)` and invisible cross-crate.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The LamQuant cookbook (neural EEG codec): the 7 `lamquant_*` recipes
/// plus the stages / artifacts / backend they reference. The unit this
/// crate hands BLUT — `registry()` registers it and the BLUT CLI / TUI
/// compose its recipes at runtime via [`Registry`]. (Moved out of
/// blut-core's `framework::cookbook` at C2a.)
pub struct LamquantCookbook;

impl Cookbook for LamquantCookbook {
    fn name(&self) -> &'static str {
        "lamquant"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        LAMQUANT_RECIPES
    }
    /// Pre-baked args JSON for the LamQuant training recipes, pointing at
    /// the corpus paths the rest of the repo uses by default. (Lives with
    /// the cookbook so blut-core holds no domain paths.) Overridable via
    /// the `R` custom-recipe overlay.
    fn default_args(&self, recipe: &str) -> Option<String> {
        let lma = "/mnt/4tb/data/lma";
        let split = "/mnt/4tb/LamQuant/data/manifests/snn_train_val_split.json";
        let labels = "/mnt/4tb/LamQuant/ai_models/snn/labels";
        let eeg = "/mnt/4tb/data/lml/edf.lml";
        Some(match recipe {
            "lamquant_data_prep" => format!(
                r#"{{
  "lml_root": "{eeg}",
  "output_dir": "{lma}"
}}"#
            ),
            "lamquant_snn" => format!(
                r#"{{
  "labels_dir": "{labels}",
  "eeg_dir": "{eeg}",
  "preset": "production",
  "subband": true,
  "epochs": 5,
  "lma_output_dir": "{lma}",
  "convert_limit": 1,
  "split_manifest": "{split}"
}}"#
            ),
            "lamquant_encoder" | "lamquant_combined_decoder" | "lamquant_oracle" => format!(
                r#"{{
  "lma_output_dir": "{lma}",
  "split_manifest": "{split}"
}}"#
            ),
            _ => return None,
        })
    }
}

/// A [`Registry`] containing exactly the LamQuant cookbook — what the
/// `blut` binary registers (alongside the lamu cookbook) before handing
/// off to the BLUT CLI.
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

    #[test]
    fn cookbook_name_is_lamquant() {
        assert_eq!(LamquantCookbook.name(), "lamquant");
    }

    #[test]
    fn default_args_for_data_prep_is_some() {
        // Domain defaults live with the cookbook (not blut-core).
        assert!(
            LamquantCookbook
                .default_args("lamquant_data_prep")
                .is_some()
        );
        assert!(LamquantCookbook.default_args("not_a_recipe").is_none());
    }
}
