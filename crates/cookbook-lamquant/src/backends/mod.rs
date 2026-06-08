//! LamQuant backend (cookbook-local).
//!
//! Owns the bespoke argparse-shaped Python subprocess runner for the
//! LamQuant training kernels. The runner struct `LamquantBackend` also
//! carries the `blut::backends::TrainingBackend` identity (the marker +
//! runner were folded into one type at C2a — see `lamquant::runner`).
//!
//! Re-exported at `crate::backends::LamquantBackend` so the recipe DEFs'
//! `type Backend = crate::backends::LamquantBackend` and every
//! `impl blut::framework::Compatible<crate::backends::LamquantBackend>`
//! resolve cookbook-locally.

pub mod lamquant;

pub use lamquant::LamquantBackend;
