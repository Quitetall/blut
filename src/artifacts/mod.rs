// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Concrete typed artifacts used by stages + recipes.
//!
//! Each artifact is a Rust struct that references on-disk bytes
//! and implements `framework::Artifact`. Together with the stages
//! that produce / consume them, they define BLUT's typed lattice.

pub mod checkpoint;
pub mod dataset;
pub mod eval;
pub mod preferences;

pub use checkpoint::{GgufModel, HfCheckpoint};
pub use dataset::{DatasetJsonl, DatasetSplit};
pub use eval::EvalReport;
pub use preferences::PreferenceJsonl;
