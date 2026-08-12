// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Compatibility view of the canonical framework artifact store.
//!
//! P2P transports manifests and payload packs, but it does not own their format,
//! validation, or rehydration rules. Those contracts live in
//! [`crate::framework::artifact_store`].

pub use crate::framework::artifact_store::*;
