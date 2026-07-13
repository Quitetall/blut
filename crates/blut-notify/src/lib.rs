// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Notification delivery boundary for the `blut-notify` sidecar.
//!
//! This is the first vertical slice of ADRs 0083/0094: sink implementations
//! remain additive, but every sink already passes through one custody check.
//! Restricted tenants or payload classifications may be handled locally, never
//! emitted off the owning box.

use blut_types::tenant::Tenant;
use blut_types::trust::{DataClass, custody_allows_off_box};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SinkBoundary {
    Local,
    OffBox,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotificationEnvelope {
    pub tenant: Tenant,
    pub data_class: DataClass,
    /// PHI-bearing text is deliberately never included in a refusal error.
    pub summary: String,
}

impl std::fmt::Debug for NotificationEnvelope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NotificationEnvelope")
            .field("tenant", &self.tenant)
            .field("data_class", &self.data_class)
            .field("summary", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotifyError {
    RestrictedOutbound {
        tenant: String,
        data_class: DataClass,
    },
    Sink(String),
}

impl std::fmt::Display for NotifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RestrictedOutbound { tenant, data_class } => write!(
                f,
                "notification refused: tenant '{tenant}' / {data_class:?} data is node-local"
            ),
            Self::Sink(message) => write!(f, "notification sink failed: {message}"),
        }
    }
}

impl std::error::Error for NotifyError {}

pub trait NotifySink {
    fn boundary(&self) -> SinkBoundary;
    /// Deliver an already-authorized envelope. Error strings may reach logs or
    /// stderr and therefore must never contain envelope fields or payload data.
    fn send(&mut self, envelope: &NotificationEnvelope) -> Result<(), String>;
}

/// Deliver through exactly one custody chokepoint. Policy runs before the sink
/// sees the envelope, so a refused payload cannot leak through sink logging,
/// retries, or serialization.
pub fn deliver(
    sink: &mut impl NotifySink,
    envelope: &NotificationEnvelope,
) -> Result<(), NotifyError> {
    if sink.boundary() == SinkBoundary::OffBox
        && !custody_allows_off_box(&envelope.tenant, envelope.data_class)
    {
        return Err(NotifyError::RestrictedOutbound {
            tenant: envelope.tenant.to_string(),
            data_class: envelope.data_class,
        });
    }
    sink.send(envelope).map_err(NotifyError::Sink)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingSink {
        boundary: SinkBoundary,
        calls: usize,
    }

    impl NotifySink for CountingSink {
        fn boundary(&self) -> SinkBoundary {
            self.boundary
        }

        fn send(&mut self, _envelope: &NotificationEnvelope) -> Result<(), String> {
            self.calls += 1;
            Ok(())
        }
    }

    fn envelope(tenant: &str, data_class: DataClass) -> NotificationEnvelope {
        NotificationEnvelope {
            tenant: Tenant::parse(tenant).unwrap(),
            data_class,
            summary: "patient-name-must-not-leak".into(),
        }
    }

    #[test]
    fn restricted_tenant_or_class_never_reaches_off_box_sink() {
        for envelope in [
            envelope("clinical/prod", DataClass::Public),
            envelope("research/dev", DataClass::Restricted),
        ] {
            let mut sink = CountingSink {
                boundary: SinkBoundary::OffBox,
                calls: 0,
            };
            let error = deliver(&mut sink, &envelope).unwrap_err();
            assert_eq!(sink.calls, 0);
            assert!(!error.to_string().contains(&envelope.summary));
        }
    }

    #[test]
    fn local_delivery_and_nonrestricted_outbound_remain_available() {
        for (tenant, data_class, boundary) in [
            ("clinical/prod", DataClass::Restricted, SinkBoundary::Local),
            ("research/dev", DataClass::Internal, SinkBoundary::OffBox),
        ] {
            let mut sink = CountingSink { boundary, calls: 0 };
            deliver(&mut sink, &envelope(tenant, data_class)).unwrap();
            assert_eq!(sink.calls, 1);
        }
    }
}
