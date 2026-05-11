//! Backend-compatibility marker trait.
//!
//! Bridges the `framework::Stage` trait (backend-unaware) with
//! `backends::TrainingBackend`. Plans are parameterized over a
//! backend `B`; `Plan<Out, B>::then` requires its stage to be
//! `Compatible<B>` so wrong-backend wiring fails at `cargo build`.
//!
//! Two flavors:
//!
//!   - **Backend-coupled**: a stage that shells out to one
//!     specific backend (e.g. `LamquantTrainMambaSnn` only makes
//!     sense via `LamquantBackend`). Provide a single
//!     `impl Compatible<LamquantBackend> for LamquantTrainMambaSnn`.
//!
//!   - **Backend-agnostic**: pure-data stages (filter, split,
//!     projection) work in any plan. Provide a blanket
//!     `impl<B: TrainingBackend> Compatible<B> for FilterDataset`.
//!
//! No methods — the trait is a pure marker. Compile time is the
//! only cost, runtime is zero.
//!
//! Why a separate trait instead of an associated type on `Stage`?
//! Because Rust associated types are 1:1; a truly agnostic stage
//! would have to pick ONE backend or duplicate impls. A marker
//! trait composes freely with `where S: Compatible<B>` bounds and
//! blanket impls; that's the cleanest expression of "this stage
//! works with these backends and only these".

use crate::backends::TrainingBackend;

/// Backend-compatibility marker. Implement for each
/// `(Stage, Backend)` pair you want to allow.
pub trait Compatible<B: TrainingBackend> {}
