// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Declarative resource envelope (ADR 0133) — the ONE struct a stage returns
//! from the ONE trait method (`Stage::resource_envelope`). All future resource
//! capability (calibration dimensions, tunables, io-profile terms) lands as
//! FIELDS here with serde defaults — wire-type versioning — so the `Stage`
//! trait itself never grows another resource method.

use serde::{Deserialize, Serialize};

use crate::gpu::GpuRequest;

/// A stage's declared pre-launch resource envelope. `Default` (all-zero) means
/// UNDECLARED — the engine bills a small compatibility estimate and (per the
/// ADR 0133 floor policy) says so loudly rather than silently.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourceEnvelope {
    /// Conservative peak host RAM for this invocation, in BYTES (byte-granular
    /// so a cookbook formula's estimate survives round-trip exactly — the
    /// parity gate compares at byte level). `0` = undeclared.
    pub ram_bytes: u64,
    /// The GPU ask (device count + hard VRAM floor).
    pub gpu: GpuRequest,
    /// ORDERED calibration dimensions: the footprint-relevant values the
    /// measured-peak store keys on. The ENGINE composes the store key from the
    /// stage/recipe identity plus these values in declaration order — a
    /// cookbook cannot collide keys across stages by accident. Empty = no
    /// calibration (estimate-only admission).
    ///
    /// Dimensions that only the runtime CONTEXT knows (e.g. a warmed cache)
    /// are appended by the engine at key-composition time in a later
    /// increment; a stage declares only what its typed args determine.
    pub calibration_dimensions: Vec<(String, String)>,
    /// Audited opt-in for stages that genuinely share footprint physics
    /// (e.g. train vs its resume twin): replaces the stage identity in the
    /// composed key so their measured peaks pool. Deliberate sharing is
    /// visible in one greppable place; accidental sharing is impossible.
    pub shared_calibration_group: Option<String>,
}

impl ResourceEnvelope {
    /// Compose from the legacy per-method declarations (`memory_gib_for` +
    /// `gpu_request`) — the default `Stage::resource_envelope` path, so every
    /// existing stage compiles and behaves identically.
    pub fn from_parts(memory_gib: u32, gpu: GpuRequest) -> Self {
        Self {
            ram_bytes: u64::from(memory_gib).saturating_mul(1024 * 1024 * 1024),
            gpu,
            calibration_dimensions: Vec::new(),
            shared_calibration_group: None,
        }
    }

    /// Did the stage declare anything at all? `false` ⇒ the engine's
    /// compatibility floor applies (and is reported loudly per ADR 0133).
    pub fn is_declared(&self) -> bool {
        self.ram_bytes > 0 || self.gpu.min_vram_mib > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_undeclared_and_from_parts_scales_gib() {
        assert!(!ResourceEnvelope::default().is_declared());
        let e = ResourceEnvelope::from_parts(3, GpuRequest::default());
        assert_eq!(e.ram_bytes, 3 * 1024 * 1024 * 1024);
        assert!(e.is_declared());
    }

    #[test]
    fn serde_roundtrip_and_forward_compat_defaults() {
        let e = ResourceEnvelope {
            ram_bytes: 42,
            gpu: GpuRequest::default(),
            calibration_dimensions: vec![("tier".into(), "3".into())],
            shared_calibration_group: Some("train".into()),
        };
        let js = serde_json::to_string(&e).unwrap();
        assert_eq!(serde_json::from_str::<ResourceEnvelope>(&js).unwrap(), e);
        // An empty object deserializes to the undeclared default — new fields
        // added later with defaults stay wire-compatible the same way.
        assert_eq!(
            serde_json::from_str::<ResourceEnvelope>("{}").unwrap(),
            ResourceEnvelope::default()
        );
    }
}
