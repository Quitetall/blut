// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Back-compatible P2P path for the keystone-owned trust policy.

pub use crate::trust::{DataClass, DispatchMatrix, TrustLevel, custody_allows_off_box};

impl From<crate::framework::execution::DataClassification> for DataClass {
    fn from(value: crate::framework::execution::DataClassification) -> Self {
        match value {
            crate::framework::execution::DataClassification::Public => Self::Public,
            crate::framework::execution::DataClassification::Internal => Self::Internal,
            crate::framework::execution::DataClassification::Restricted => Self::Restricted,
        }
    }
}

impl From<DataClass> for crate::framework::execution::DataClassification {
    fn from(value: DataClass) -> Self {
        match value {
            DataClass::Public => Self::Public,
            DataClass::Internal => Self::Internal,
            DataClass::Restricted => Self::Restricted,
        }
    }
}
