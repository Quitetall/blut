//! Recipe machinery — the engine seam domain cookbooks build against.
//!
//! blut-core ships ZERO recipe DEFs. The `Recipe` trait, `RecipeDef`
//! (erased catalog entry), and `RecipeCategory` live here; concrete
//! recipes live in cookbook crates (`cookbook-lamu`, `cookbook-lamquant`)
//! that depend on this crate and register their recipes at runtime via
//! [`crate::framework::Registry`]. The lamu recipes moved to
//! `cookbook-lamu` at C2b, the lamquant recipes to `cookbook-lamquant`
//! at C2a. See `[[project_blut_cookbook_split]]`.

pub mod declarative;
pub mod recipe;

pub use recipe::{Recipe, RecipeCategory, RecipeDef};
