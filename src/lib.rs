// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT — Basically Less Unsound Training (affectionately, Brian Lam's
//! Universal Trainer).
//!
//! A standalone, domain-agnostic Rust framework for orchestrating ML workflows
//! via typed stages, plans, and recipes. Compile-time DAG enforcement
//! via PhantomData on `Plan<Out>` — wrong ingredient wiring = `cargo build`
//! error, not runtime panic.
//!
//! Public surface:
//!
//!   - `framework::*` — Artifact / Stage / Plan / Executor + cache.
//!   - built-in policy/check/connector stages shared by downstream cookbooks.
//!   - CLI/TUI orchestration seams used by cookbook-owned binaries.
//!
//! Domain training stages (materialize, train, convert, register, evaluate)
//! and their recipes live in downstream cookbook crates.
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
pub mod auto_tune;
pub mod backends;
pub mod broker;
pub mod catalog;
pub mod checks;
pub mod cli;
/// Cloud compute queue (ADR 0082 / 0067 T3.1) — the P2P data plane over an
/// object store. Gated behind the `cloud` feature (which implies `p2p`).
#[cfg(feature = "cloud")]
pub mod cloud;
pub mod config;
pub mod connectors;
pub mod containment;
pub mod cost;
pub mod dataset_registry;
pub mod datasets_db;
pub mod error;
pub mod experiment_registry;
pub mod framework;
pub mod hpo;
pub mod jobs;
pub mod lineage_db;
pub mod lineage_report;
pub mod model_registry;
/// P2P distributed compute — trust model, encryption, task dispatch.
/// Gated behind the `p2p` feature.
#[cfg(feature = "p2p")]
pub mod p2p;

pub mod paths;
pub mod policy;
pub mod protocol;
pub mod python_kill;
/// Highly-available scheduler state via embedded Raft (ADR 0106) — leader
/// election plus replicated dispatch decisions, so losing the coordinator
/// mid-DAG cannot lose the in-flight table or re-dispatch running work.
/// Gated behind the `raft` feature (which implies `p2p`).
#[cfg(feature = "raft")]
pub mod raft;
pub mod rbac;
pub mod recipes;
pub mod registry;
pub mod registry_args;
pub mod registry_db;
pub mod run_ledger;
pub mod runs;
pub mod schedule;
pub mod scheduler_lock;
pub mod secrets;
pub mod sensor;
pub mod sensord;
pub mod sla;
pub mod spec;
pub mod tenant;
pub mod trigger;
pub mod trust;
// The interactive ratatui cockpit moved to the `crates/blut-tui` SIDECAR
// (ADR 0083 M2). The engine keeps only the seams: `cli::TuiHook` (attach an
// in-process cockpit) and `framework::CookbookTui` (a cookbook's bespoke TUI).

// ENGINE CARVE (v1.0): the generic-LLM cookbook — concrete `ingredients`,
// the `backend` trait + concrete backends (`backends::{lamu,hf_trainer}`),
// `convert`, and `conversations` — moved to the `blut-backends` crate.
// `backends` here keeps ONLY the abstract `TrainingBackend` trait (the
// intended public backend-identity seam). The engine RETAINS `spec` /
// `protocol` / `python_kill` because the framework's job-persistence
// layer (`jobs.rs`) reads the on-disk `TrainSpec` / `StatusUpdate`
// schema and uses the subprocess-group lifecycle primitives.

/// Process-wide lock for tests that mutate environment variables.
/// Multiple test modules touch `LAMU_TRAIN_*` env vars; without a
/// shared mutex parallel test execution races on the global env.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub use error::TrainError;
