// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Read-side lineage reports (ADR 0099) — the provenance graph + model card +
//! run diff types.
//!
//! ADR 0083: these are wasm32-safe pure-serde WIRE types, so they live in the
//! `blut-types` keystone crate (shared with the web/notify sidecars). This
//! module RE-EXPORTS them at their historical `crate::lineage_report` path — no
//! downstream churn; `lineage_db` + the CLI keep referring to
//! `crate::lineage_report::{ProvenanceGraph, ModelCard, RunDiff, …}`.

// Explicit re-export (not a glob): a new `pub` item added to `blut_types::report`
// must be named here to enter the engine's `crate::lineage_report` surface — so
// the keystone can't silently widen the engine's public API.
pub use blut_types::report::{
    ArgDelta, CardContent, GraphNode, ModelCard, ProvenanceGraph, RunDiff,
};
