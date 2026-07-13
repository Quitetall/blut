// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Tenant identity is owned by the WASM-safe `blut-types` keystone so the
//! engine and sidecars enforce one parser and one Restricted classification.

pub use blut_types::tenant::{DEFAULT_PROJECT, Tenant};
