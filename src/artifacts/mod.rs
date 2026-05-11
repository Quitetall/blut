//! Concrete typed artifacts used by stages + recipes.
//!
//! Each artifact is a Rust struct that references on-disk bytes
//! and implements `framework::Artifact`. Together with the stages
//! that produce / consume them, they define BLUT's typed lattice.

pub mod checkpoint;
pub mod dataset;
pub mod preferences;

pub use checkpoint::{GgufModel, HfCheckpoint};
pub use dataset::DatasetJsonl;
pub use preferences::PreferenceJsonl;
