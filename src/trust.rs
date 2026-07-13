// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Transport-independent custody policy shared by default-feature gates,
//! P2P/cloud adapters, and sidecars.

pub use blut_types::trust::{DataClass, DispatchMatrix, TrustLevel, custody_allows_off_box};
