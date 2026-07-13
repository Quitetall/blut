// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! WASM-safe custody and peer-trust policy (ADRs 0061 and 0096).

use serde::{Deserialize, Serialize};

use crate::tenant::Tenant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataClass {
    Public,
    Internal,
    Restricted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrustLevel {
    Anonymous,
    Registered,
    Trusted,
}

/// The shared custody rule for every transport and sidecar boundary.
pub fn custody_allows_off_box(tenant: &Tenant, data: DataClass) -> bool {
    !tenant.is_restricted() && data != DataClass::Restricted
}

impl TrustLevel {
    pub fn level(self) -> u8 {
        match self {
            Self::Anonymous => 0,
            Self::Registered => 1,
            Self::Trusted => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Anonymous => "anonymous",
            Self::Registered => "registered",
            Self::Trusted => "trusted",
        }
    }
}

/// Transport-independent dispatch policy. Restricted always returns false,
/// even if a deserialized/custom matrix attempts to enable it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DispatchMatrix {
    allowed: [[bool; 3]; 3],
}

impl DispatchMatrix {
    pub fn can_dispatch(&self, data: DataClass, trust: TrustLevel) -> bool {
        if data == DataClass::Restricted {
            return false;
        }
        self.allowed[class_index(data)][trust.level() as usize]
    }

    /// Update the serialized policy cell. Restricted-row values are retained
    /// for wire compatibility, but can never override [`Self::can_dispatch`]'s
    /// fail-closed custody rule.
    pub fn set(&mut self, data: DataClass, trust: TrustLevel, allowed: bool) {
        self.allowed[class_index(data)][trust.level() as usize] = allowed;
    }
}

fn class_index(data: DataClass) -> usize {
    match data {
        DataClass::Public => 0,
        DataClass::Internal => 1,
        DataClass::Restricted => 2,
    }
}

impl Default for DispatchMatrix {
    fn default() -> Self {
        Self {
            allowed: [
                [true, true, true],
                [false, true, true],
                [false, false, false],
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matrix_and_custom_restricted_override() {
        let mut matrix = DispatchMatrix::default();
        assert!(matrix.can_dispatch(DataClass::Public, TrustLevel::Anonymous));
        assert!(!matrix.can_dispatch(DataClass::Internal, TrustLevel::Anonymous));
        assert!(matrix.can_dispatch(DataClass::Internal, TrustLevel::Registered));
        matrix.set(DataClass::Restricted, TrustLevel::Trusted, true);
        assert!(!matrix.can_dispatch(DataClass::Restricted, TrustLevel::Trusted));
    }

    #[test]
    fn trust_order_and_labels_are_stable() {
        assert!(TrustLevel::Anonymous.level() < TrustLevel::Registered.level());
        assert!(TrustLevel::Registered.level() < TrustLevel::Trusted.level());
        assert_eq!(TrustLevel::Anonymous.label(), "anonymous");
        assert_eq!(TrustLevel::Registered.label(), "registered");
        assert_eq!(TrustLevel::Trusted.label(), "trusted");
    }

    #[test]
    fn legacy_serde_variant_spelling_is_unchanged() {
        assert_eq!(
            serde_json::to_string(&DataClass::Restricted).unwrap(),
            r#""Restricted""#
        );
        assert_eq!(
            serde_json::to_string(&TrustLevel::Registered).unwrap(),
            r#""Registered""#
        );
        let mut matrix = DispatchMatrix::default();
        matrix.set(DataClass::Internal, TrustLevel::Anonymous, true);
        let wire = serde_json::to_string(&matrix).unwrap();
        let decoded: DispatchMatrix = serde_json::from_str(&wire).unwrap();
        assert!(decoded.can_dispatch(DataClass::Internal, TrustLevel::Anonymous));
    }

    #[test]
    fn custody_combines_tenant_and_payload_classification() {
        assert!(!custody_allows_off_box(
            &Tenant::parse("clinical/prod").unwrap(),
            DataClass::Public
        ));
        assert!(!custody_allows_off_box(
            &Tenant::parse("research/dev").unwrap(),
            DataClass::Restricted
        ));
        assert!(custody_allows_off_box(
            &Tenant::parse("research/dev").unwrap(),
            DataClass::Internal
        ));
    }
}
