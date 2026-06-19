// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Recipe machinery — the engine seam domain cookbooks build against.
//!
//! blut-core ships ZERO recipe DEFs. The `Recipe` trait, `RecipeDef`
//! (erased catalog entry), and `Course` (the grouping tag, ADR 0051;
//! `RecipeCategory` is its back-compat alias) live here; concrete
//! recipes live in cookbook crates (`cookbook-lamu`, `cookbook-lamquant`)
//! that depend on this crate and register their recipes at runtime via
//! [`crate::framework::Registry`]. The lamu recipes moved to
//! `cookbook-lamu` at C2b, the lamquant recipes to `cookbook-lamquant`
//! at C2a. See `[[project_blut_cookbook_split]]`.

pub mod declarative;
pub mod recipe;

pub use recipe::{Course, Recipe, RecipeCategory, RecipeDef};
