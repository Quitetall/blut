// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Trust model — data classification × peer trust tiers → dispatch matrix.
//!
//! The coordinator enforces the matrix before every task dispatch. A peer
//! cannot request data above its trust level.

use serde::{Deserialize, Serialize};

/// Sensitivity classification of the data a stage processes.
/// Set per-corpus by the user; rides the task manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataClass {
    /// Open datasets (TUH, Physionet, synthetic). Any peer can compute.
    Public,
    /// Proprietary but non-clinical (lab recordings, dev data).
    /// Registered + Trusted peers only.
    Internal,
    /// Clinical / PHI (hospital EEG, patient records). Node-local through M5;
    /// no peer trust tier may receive it.
    Restricted,
}

/// How much the coordinator trusts a peer. Set by the coordinator
/// when a peer registers; persisted in the peer registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Self-generated key, no identity verification.
    Anonymous,
    /// Identity verified (email, NDA record).
    Registered,
    /// Lab collaborator, whitelisted by the coordinator.
    Trusted,
}

impl TrustLevel {
    /// Numeric ordering for comparison (Anonymous < Registered < Trusted).
    pub fn level(self) -> u8 {
        match self {
            Self::Anonymous => 0,
            Self::Registered => 1,
            Self::Trusted => 2,
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Anonymous => "anonymous",
            Self::Registered => "registered",
            Self::Trusted => "trusted",
        }
    }
}

/// The dispatch matrix: which (DataClass, TrustLevel) combinations are
/// allowed. The coordinator checks `matrix.can_dispatch(data_class,
/// peer_trust)` before every task dispatch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DispatchMatrix {
    /// `allowed[class][trust]` = true if dispatch is permitted.
    allowed: [[bool; 3]; 3],
}

impl DispatchMatrix {
    /// Can a peer with `trust` level compute a task with `data` class?
    pub fn can_dispatch(&self, data: DataClass, trust: TrustLevel) -> bool {
        // ADR 0096 M2.1: Restricted data may run locally inside its tenant but
        // cannot leave the owning node through M5. This hard check dominates
        // even a deserialized/custom matrix with its Restricted cell set true.
        if data == DataClass::Restricted {
            return false;
        }
        let ci = match data {
            DataClass::Public => 0,
            DataClass::Internal => 1,
            DataClass::Restricted => 2,
        };
        let ti = trust.level() as usize;
        self.allowed[ci][ti]
    }

    /// Set a specific cell in the matrix.
    pub fn set(&mut self, data: DataClass, trust: TrustLevel, allowed: bool) {
        let ci = match data {
            DataClass::Public => 0,
            DataClass::Internal => 1,
            DataClass::Restricted => 2,
        };
        let ti = trust.level() as usize;
        self.allowed[ci][ti] = allowed;
    }
}

impl Default for DispatchMatrix {
    /// The default matrix from ADR 0061:
    ///
    /// ```text
    ///                 Anonymous  Registered  Trusted
    /// Public            ✅         ✅          ✅
    /// Internal          ❌         ✅          ✅
    /// Restricted        ❌         ❌          ❌  (node-local through M5)
    /// ```
    fn default() -> Self {
        Self {
            allowed: [
                [true, true, true],    // Public → all
                [false, true, true],   // Internal → Registered + Trusted
                [false, false, false], // Restricted → node-local only
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matrix_public_any() {
        let m = DispatchMatrix::default();
        assert!(m.can_dispatch(DataClass::Public, TrustLevel::Anonymous));
        assert!(m.can_dispatch(DataClass::Public, TrustLevel::Registered));
        assert!(m.can_dispatch(DataClass::Public, TrustLevel::Trusted));
    }

    #[test]
    fn default_matrix_internal_no_anonymous() {
        let m = DispatchMatrix::default();
        assert!(!m.can_dispatch(DataClass::Internal, TrustLevel::Anonymous));
        assert!(m.can_dispatch(DataClass::Internal, TrustLevel::Registered));
        assert!(m.can_dispatch(DataClass::Internal, TrustLevel::Trusted));
    }

    #[test]
    fn restricted_is_node_local_even_for_trusted_or_custom_matrix() {
        let m = DispatchMatrix::default();
        assert!(!m.can_dispatch(DataClass::Restricted, TrustLevel::Anonymous));
        assert!(!m.can_dispatch(DataClass::Restricted, TrustLevel::Registered));
        assert!(!m.can_dispatch(DataClass::Restricted, TrustLevel::Trusted));

        let mut custom = DispatchMatrix::default();
        custom.set(DataClass::Restricted, TrustLevel::Trusted, true);
        assert!(
            !custom.can_dispatch(DataClass::Restricted, TrustLevel::Trusted),
            "through M5, no custom trust matrix may override Restricted node-local custody"
        );
    }

    #[test]
    fn custom_matrix_override() {
        let mut m = DispatchMatrix::default();
        // Allow anonymous to compute internal data (e.g. for a public project).
        m.set(DataClass::Internal, TrustLevel::Anonymous, true);
        assert!(m.can_dispatch(DataClass::Internal, TrustLevel::Anonymous));
    }

    #[test]
    fn trust_level_ordering() {
        assert!((TrustLevel::Anonymous.level()) < TrustLevel::Registered.level());
        assert!((TrustLevel::Registered.level()) < TrustLevel::Trusted.level());
    }

    #[test]
    fn trust_level_labels() {
        assert_eq!(TrustLevel::Anonymous.label(), "anonymous");
        assert_eq!(TrustLevel::Registered.label(), "registered");
        assert_eq!(TrustLevel::Trusted.label(), "trusted");
    }
}
