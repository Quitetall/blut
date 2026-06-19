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
pub mod backends;
pub mod broker;
pub mod cli;
pub mod config;
pub mod datasets_db;
pub mod error;
pub mod framework;
pub mod hpo;
pub mod jobs;
pub mod lineage_db;
pub mod paths;
pub mod policy;
pub mod protocol;
pub mod python_kill;
pub mod recipes;
pub mod registry;
pub mod runs;
pub mod schedule;
pub mod scheduler_lock;
pub mod sensor;
pub mod spec;
/// The interactive ratatui cockpit. Behind the off-by-default `tui` feature:
/// 1.0 ships CLI-only; the TUI returns in 1.1. The CLI (`cli::run`) degrades
/// to printing help for the bare `blut` command when this is disabled.
#[cfg(feature = "tui")]
pub mod tui;

// ENGINE CARVE (v1.0): the generic-LLM cookbook — concrete `stages`,
// the `backend` trait + concrete backends (`backends::{lamu,hf_trainer}`),
// `convert`, and `conversations` — moved to the `blut-backends` crate.
// `backends` here keeps ONLY the abstract `TrainingBackend` trait (the
// public 1.0 backend-identity seam). The engine RETAINS `spec` /
// `protocol` / `python_kill` because the framework's job-persistence
// layer (`jobs.rs`) reads the on-disk `TrainSpec` / `StatusUpdate`
// schema and uses the subprocess-group lifecycle primitives.

/// Process-wide lock for tests that mutate environment variables.
/// Multiple test modules touch `LAMU_TRAIN_*` env vars; without a
/// shared mutex parallel test execution races on the global env.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub use error::TrainError;
