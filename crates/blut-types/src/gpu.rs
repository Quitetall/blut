// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! GPU resource-envelope wire types (ADR 0087).
//!
//! ADR 0083: `GpuRequest` is a stage's typed GPU ask — a pure-serde value that
//! rides the PlanSpec and (soon) crosses to the distributed launchers and the
//! `blut-web` sidecar. So it lives in the wasm32-safe `blut-types` keystone; the
//! engine re-exports it at `crate::broker::gpu::GpuRequest` (zero churn) and the
//! `GpuScheduler` / `GpuInventory` runtime machinery stays engine-side.

use serde::{Deserialize, Serialize};

/// A stage's GPU ask, part of its typed resource envelope. `Default` is the
/// whole-device exclusive request — an un-annotated stage behaves exactly as
/// under the legacy single semaphore.
///
/// `deny_unknown_fields` makes this a strict wire type: it fails loud on an
/// unknown key rather than silently dropping it. Adding a field is therefore a
/// forward-compatible change only for readers that predate it if the field is
/// `#[serde(default)]`-covered by the container `default` — a NEW required field
/// would reject old payloads, so extend with defaulted fields (as here).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct GpuRequest {
    /// Devices to hold for the stage's lifetime.
    pub count: u32,
    /// Minimum free VRAM (MiB) each granted device must have. `0` = any device.
    pub min_vram_mib: u64,
    /// Whole-device exclusivity. v1 is exclusive-only (no fractional sharing —
    /// see ADR 0087 Alternatives); the field is reserved for a future MIG path.
    pub exclusive: bool,
}

impl Default for GpuRequest {
    fn default() -> Self {
        Self {
            count: 1,
            min_vram_mib: 0,
            exclusive: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_whole_device_exclusive() {
        let r = GpuRequest::default();
        assert_eq!(r.count, 1);
        assert_eq!(r.min_vram_mib, 0);
        assert!(r.exclusive);
    }

    #[test]
    fn roundtrips_and_rejects_unknown_fields() {
        let r = GpuRequest {
            count: 2,
            min_vram_mib: 24576,
            exclusive: true,
        };
        let js = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<GpuRequest>(&js).unwrap(), r);
        // Unset fields fall back to Default (serde `default`), so `{}` is the
        // legacy whole-device request — the byte-compatible un-annotated stage.
        assert_eq!(
            serde_json::from_str::<GpuRequest>("{}").unwrap(),
            GpuRequest::default()
        );
        // A typo'd field is a hard error (deny_unknown_fields), not silently
        // dropped — a wire type must fail loud on drift.
        assert!(serde_json::from_str::<GpuRequest>(r#"{"conut":2}"#).is_err());
    }
}
