// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Notification delivery boundary for the `blut-notify` sidecar.
//!
//! This is the first vertical slice of ADRs 0083/0094: sink implementations
//! remain additive, but every sink already passes through one custody check.
//! Restricted tenants or payload classifications may be handled locally, never
//! emitted off the owning box.

use blut_types::sla::SlaBreach;
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

// ── SLA breaches → notifications (ADR 0094) ─────────────────────────

/// Build a notification envelope from an SLA breach. The breach `summary` is
/// already PHI-free (redacted at the producer, `blut sla check`), so this is a
/// pure field lift — a `restricted` breach carries no patient content into the
/// envelope, only job id + seconds. An unparseable tenant string falls back to
/// the default tenant (never a panic on operator data).
pub fn breach_to_envelope(breach: &SlaBreach) -> NotificationEnvelope {
    let tenant = Tenant::parse(&breach.tenant)
        .unwrap_or_else(|| Tenant::parse("default").expect("default tenant parses"));
    NotificationEnvelope {
        tenant,
        data_class: breach.data_class,
        summary: breach.summary.clone(),
    }
}

/// Read `sla.jsonl` breach rows (absent file ⇒ no breaches). Malformed lines are
/// skipped, never fatal — a partially-written tail can't stall the notifier.
pub fn read_breaches(path: &std::path::Path) -> std::io::Result<Vec<SlaBreach>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text
            .lines()
            .filter_map(|l| SlaBreach::from_line(l).ok())
            .collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// Route each breach to `sink` through the custody chokepoint. Returns the count
/// DELIVERED (a `RestrictedOutbound` refusal is not a delivery — a restricted
/// breach to an off-box sink is dropped, fail-closed, and counted as 0). Errors
/// other than a custody refusal (a sink failure) propagate.
pub fn notify_breaches(
    sink: &mut impl NotifySink,
    breaches: &[SlaBreach],
) -> Result<usize, NotifyError> {
    let mut delivered = 0;
    for breach in breaches {
        let envelope = breach_to_envelope(breach);
        match deliver(sink, &envelope) {
            Ok(()) => delivered += 1,
            Err(NotifyError::RestrictedOutbound { .. }) => {} // fail-closed drop
            Err(other) => return Err(other),
        }
    }
    Ok(delivered)
}

/// A sink that pipes the (already custody-checked) envelope JSON to a command's
/// stdin — the `exec` escape hatch (ADR 0094). PHI never touches argv: the
/// payload is on stdin only. The boundary is declared at construction (an exec
/// target that reaches off-box, e.g. a curl-to-Slack wrapper, is `OffBox`; a
/// local logger is `Local`).
pub struct ExecSink {
    program: std::ffi::OsString,
    args: Vec<std::ffi::OsString>,
    boundary: SinkBoundary,
}

impl ExecSink {
    pub fn new(
        program: impl Into<std::ffi::OsString>,
        args: Vec<std::ffi::OsString>,
        boundary: SinkBoundary,
    ) -> Self {
        Self {
            program: program.into(),
            args,
            boundary,
        }
    }
}

impl NotifySink for ExecSink {
    fn boundary(&self) -> SinkBoundary {
        self.boundary
    }

    fn send(&mut self, envelope: &NotificationEnvelope) -> Result<(), String> {
        use std::io::Write as _;
        let body = serde_json::to_vec(envelope).map_err(|e| e.to_string())?;
        let mut child = std::process::Command::new(&self.program)
            .args(&self.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {:?}: {e}", self.program))?;
        child
            .stdin
            .take()
            .ok_or_else(|| "no stdin pipe".to_string())?
            .write_all(&body)
            .map_err(|e| e.to_string())?;
        let status = child.wait().map_err(|e| e.to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("exec sink exited with {status}"))
        }
    }
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

    #[test]
    fn breach_lifts_to_a_phi_free_envelope() {
        let breach = SlaBreach {
            rule: "cap".into(),
            kind: blut_types::sla::SlaKind::MaxRuntime,
            job_id: "job-9".into(),
            tenant: "clinical/prod".into(),
            data_class: DataClass::Restricted,
            observed_secs: 4200,
            limit_secs: 3600,
            summary: "max_runtime breach on job job-9: 4200s vs limit 3600s".into(),
            detected_unix: 1,
        };
        let env = breach_to_envelope(&breach);
        assert_eq!(env.data_class, DataClass::Restricted);
        assert!(env.tenant.is_restricted());
        assert_eq!(env.summary, breach.summary);
        assert!(!env.summary.contains("patient"));
    }

    #[test]
    fn exec_sink_delivers_on_success_and_errors_on_failure() {
        // `true` exits 0 (delivered); `false` exits 1 (sink error). Both read
        // the payload from stdin, never argv.
        let mut ok = ExecSink::new("true", vec![], SinkBoundary::Local);
        deliver(&mut ok, &envelope("shared", DataClass::Internal)).unwrap();
        let mut bad = ExecSink::new("false", vec![], SinkBoundary::Local);
        assert!(deliver(&mut bad, &envelope("shared", DataClass::Internal)).is_err());
    }
}
