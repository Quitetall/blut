// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! SLA breach wire type (ADR 0094) — the record the engine's `blut sla check`
//! writes to `sla.jsonl` and the `blut-notify` sidecar reads to route alerts.
//! Lives in the keystone so producer (engine, lineage-DB reader) and consumer
//! (notify sidecar) share ONE definition; a schema change breaks both compiles.
//!
//! Custody: `summary` is the ONLY human-facing text and MUST be PHI-free — the
//! producer writes a redacted one-liner (e.g. "run late by 42s"), never patient
//! content, because a `restricted` breach can still notify a LOCAL sink (ADR
//! 0061/0096). The `tenant`/`data_class` fields let the notifier apply the
//! off-box custody rule ([`crate::trust::custody_allows_off_box`]).

use serde::{Deserialize, Serialize};

use crate::trust::DataClass;

/// Which SLA the run violated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlaKind {
    /// Ran longer than `max_runtime` allowed.
    MaxRuntime,
    /// Missed a wall-clock `deadline`.
    Deadline,
    /// Consumed/produced data older than the `freshness` window.
    Freshness,
}

impl SlaKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SlaKind::MaxRuntime => "max_runtime",
            SlaKind::Deadline => "deadline",
            SlaKind::Freshness => "freshness",
        }
    }
}

/// One SLA violation — a line of `sla.jsonl`. A breach is itself an event that
/// may notify or re-trigger; `summary` is redacted at the producer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlaBreach {
    /// The rule name that fired (from the SLA rule file).
    pub rule: String,
    pub kind: SlaKind,
    pub job_id: String,
    /// Owning tenant (`project[/domain]`); drives the off-box custody rule.
    pub tenant: String,
    /// Data classification of the breached run.
    pub data_class: DataClass,
    /// Observed value (seconds for runtime/deadline/freshness).
    pub observed_secs: i64,
    /// The limit that was exceeded (seconds).
    pub limit_secs: i64,
    /// PHI-FREE one-liner (redacted at the producer — never patient content).
    pub summary: String,
    /// Unix seconds the breach was detected.
    pub detected_unix: i64,
}

impl SlaBreach {
    /// Serialize to one `sla.jsonl` line (no trailing newline).
    pub fn to_line(&self) -> String {
        // Infallible: every field is a plain serde scalar/enum.
        serde_json::to_string(self).expect("SlaBreach serializes")
    }

    /// Parse one `sla.jsonl` line.
    pub fn from_line(line: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breach_line_roundtrips() {
        let b = SlaBreach {
            rule: "nightly-deadline".into(),
            kind: SlaKind::MaxRuntime,
            job_id: "job-123".into(),
            tenant: "clinical/prod".into(),
            data_class: DataClass::Restricted,
            observed_secs: 4200,
            limit_secs: 3600,
            summary: "run late by 600s".into(),
            detected_unix: 1_700_000_000,
        };
        let line = b.to_line();
        assert_eq!(SlaBreach::from_line(&line).unwrap(), b);
        // The redacted summary carries no patient content by construction.
        assert!(!line.contains("patient"));
    }
}
