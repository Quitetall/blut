// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! blut-types (ADR 0083) — the wasm32-safe KEYSTONE: pure-serde wire types the
//! engine and its (Rust-end-to-end) web/notify sidecars share. NO tokio,
//! rusqlite, filesystem, or engine dep — a required `cargo check --target
//! wasm32-unknown-unknown -p blut-types` CI job keeps that invariant green.
//!
//! The engine RE-EXPORTS these at their old paths, so moving a type here is
//! zero-churn downstream (ADR 0083 §2).

pub mod report;
