//! BLUT — Brian Lam's Universal Trainer.
//!
//! A Rust framework for orchestrating local ML training workloads
//! (SFT, DPO, distillation, evaluation) via typed stages, plans,
//! and recipes. Compile-time DAG enforcement via PhantomData on
//! `Plan<Out>` — wrong stage wiring = `cargo build` error, not
//! runtime panic.
//!
//! Public surface:
//!
//!   - `framework::*` — Artifact / Stage / Plan / Executor + cache.
//!   - `stages::*` — concrete stage impls (materialize, train,
//!     convert, register, eval).
//!   - `recipes::*` — saved plans for common workflows.
//!   - `blut` binary — CLI driver (`blut recipe run <name>`,
//!     `blut jobs`, `blut log <id>`, etc.).
//!
//! Originally extracted from the LAMU monorepo; runs as a
//! standalone binary + library now with no upward Rust dep on
//! LAMU. Integration with LAMU is via the `blut` binary's stdio
//! contract (lamu-mcp's `train_from_conversations` tool shells
//! out to it).

// Production code is unsafe-free EXCEPT one narrow place: mmap
// in framework::artifact::hash_file_mmap. Tests also use unsafe
// std::env::set_var for hermetic env injection. The combination:
// deny by default (so audit grep stays easy), allow per-call-site
// with #[allow(unsafe_code)] + SAFETY comment, plus tests get
// blanket allow.
#![cfg_attr(not(test), deny(unsafe_code))]

pub mod artifacts;
pub mod backend;
pub mod config;
pub mod conversations;
pub mod convert;
pub mod datasets_db;
pub mod error;
pub mod framework;
pub mod jobs;
pub mod paths;
pub mod policy;
pub mod protocol;
pub mod python_backend;
pub mod recipes;
pub mod registry;
pub mod scheduler_lock;
pub mod spec;
pub mod stages;

/// Process-wide lock for tests that mutate environment variables.
/// Multiple test modules touch `LAMU_TRAIN_*` env vars; without a
/// shared mutex parallel test execution races on the global env.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub use backend::{TrainArtifact, TrainBackend};
pub use error::TrainError;
pub use protocol::StatusUpdate;
pub use python_backend::PythonTrainBackend;
pub use spec::{DatasetSource, Method, Optim, TrainSpec};
