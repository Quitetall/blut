// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Concrete typed artifacts used by stages + recipes.
//!
//! Each artifact is a Rust struct that references on-disk bytes
//! and implements `framework::Artifact`. Together with the stages
//! that produce / consume them, they define BLUT's typed lattice.
//!
//! The built-in catalog is external-capable because source materializers,
//! model registries, and trainer backends may legitimately return user-owned
//! paths outside the executor stage root. Those instances receive a
//! non-addressable identity; instances fully owned by a stage remain portable.

pub mod checkpoint;
pub mod dataset;
pub mod eval;
pub mod preferences;

pub use checkpoint::{GgufModel, HfCheckpoint};
pub use dataset::{DatasetJsonl, DatasetSplit};
pub use eval::EvalReport;
pub use preferences::PreferenceJsonl;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::Artifact;

    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn builtin_catalog_declares_external_path_capability() {
        assert!(DatasetJsonl::ALLOW_EXTERNAL_PATHS);
        assert!(DatasetSplit::ALLOW_EXTERNAL_PATHS);
        assert!(PreferenceJsonl::ALLOW_EXTERNAL_PATHS);
        assert!(HfCheckpoint::ALLOW_EXTERNAL_PATHS);
        assert!(GgufModel::ALLOW_EXTERNAL_PATHS);
        assert!(EvalReport::ALLOW_EXTERNAL_PATHS);
    }
}
