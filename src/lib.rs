// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT — Brian Lam's Universal Trainer.
//!
//! A standalone, domain-agnostic Rust framework for orchestrating
//! local ML training workloads (SFT, DPO, distillation, evaluation)
//! via typed ingredients, plans, and recipes. Compile-time DAG enforcement
//! via PhantomData on `Plan<Out>` — wrong ingredient wiring = `cargo build`
//! error, not runtime panic.
//!
//! Public surface:
//!
//!   - `framework::*` — Artifact / Stage / Plan / Executor + cache.
//!   - concrete ingredient impls (materialize, train, convert, register,
//!     eval) and saved plans for common workflows.
//!
//! The library has no upward dependency on any host application;
//! integration with an outer tool is via the CLI driver's stdio
//! contract.

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
pub mod containment;
pub mod datasets_db;
/// Starlark front-end for authoring PlanSpecs from `.star` scripts (ADR
/// 0078). Gated behind the `dsl` feature (off by default — keeps the lean
/// CLI build free of the starlark dependency).
#[cfg(feature = "dsl")]
pub mod dsl;
pub mod error;
pub mod framework;
pub mod hpo;
pub mod jobs;
pub mod lineage_db;
/// P2P distributed compute — trust model, encryption, task dispatch.
/// Gated behind the `p2p` feature.
#[cfg(feature = "p2p")]
pub mod p2p;
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
/// The interactive ratatui cockpit. Part of the default-on `tui` feature, so
/// it ships in 1.0; a `--no-default-features` build drops it and the CLI
/// (`cli::run`) degrades to printing help for the bare `blut` command.
#[cfg(feature = "tui")]
pub mod tui;

// ENGINE CARVE (v1.0): the generic-LLM cookbook — concrete `ingredients`,
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
