// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Telemetry wire schema (ADR 0097) — the records the ENGINE writes and the
//! `blut-metrics` SIDECAR exports as Prometheus/OpenMetrics and OTLP spans.
//!
//! This lives in the wasm-safe keystone so producer and consumer share ONE
//! definition: a schema change breaks both compiles at once instead of drifting
//! (ADR 0083). Nothing here opens a socket or knows what Prometheus is — these
//! are plain serde records appended to `status.jsonl`. All network I/O belongs
//! to the sidecar (ADR 0034).
//!
//! Identifier shapes follow W3C Trace Context so an OTLP forwarder can pass
//! them through unmodified: a 128-bit trace id as 32 lowercase hex characters,
//! a 64-bit span id as 16. Metric and label names are validated against the
//! Prometheus grammar AT CONSTRUCTION, so an invalid name can never reach a
//! scrape body and corrupt an entire exposition response.

use serde::{Deserialize, Serialize};

/// A 128-bit trace id — 32 lowercase hex characters (W3C Trace Context).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TraceId(String);

/// A 64-bit span id — 16 lowercase hex characters (W3C Trace Context).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpanId(String);

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl TraceId {
    /// Parse a 32-char lowercase-hex trace id. An all-zero id is invalid per
    /// W3C, and accepting it would silently join unrelated runs into one trace.
    pub fn parse(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        if is_lower_hex(&value, 32) && value.bytes().any(|b| b != b'0') {
            Some(Self(value))
        } else {
            None
        }
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl SpanId {
    /// Parse a 16-char lowercase-hex span id (all-zero rejected, as above).
    pub fn parse(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        if is_lower_hex(&value, 16) && value.bytes().any(|b| b != b'0') {
            Some(Self(value))
        } else {
            None
        }
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One span's position in a trace. `parent_span_id` is `None` for a root span;
/// carrying it across a mesh hop is what keeps a host-hopping stage ONE trace
/// instead of N disconnected fragments (ADR 0097's differentiator).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanContext {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<SpanId>,
}

impl SpanContext {
    pub fn root(trace_id: TraceId, span_id: SpanId) -> Self {
        Self {
            trace_id,
            span_id,
            parent_span_id: None,
        }
    }

    /// A child span in the SAME trace, parented to this span. This is the only
    /// blessed way to descend, so a child can never be minted into a different
    /// trace by accident.
    pub fn child(&self, span_id: SpanId) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id,
            parent_span_id: Some(self.span_id.clone()),
        }
    }
}

/// Prometheus metric-name grammar: `[a-zA-Z_:][a-zA-Z0-9_:]*`.
fn is_metric_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    match bytes.next() {
        Some(b) if b.is_ascii_alphabetic() || b == b'_' || b == b':' => {}
        _ => return false,
    }
    bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b':')
}

/// Prometheus label-name grammar: `[a-zA-Z_][a-zA-Z0-9_]*` (no colons).
fn is_label_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    match bytes.next() {
        Some(b) if b.is_ascii_alphabetic() || b == b'_' => {}
        _ => return false,
    }
    bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// A validated `name="value"` pair. Label VALUES are unrestricted UTF-8 (the
/// exporter escapes them); only the NAME is grammar-checked.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Label {
    pub name: String,
    pub value: String,
}

impl Label {
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Option<Self> {
        let name = name.into();
        is_label_name(&name).then(|| Self {
            name,
            value: value.into(),
        })
    }
}

/// A validated metric name.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MetricName(String);

impl MetricName {
    pub fn parse(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        is_metric_name(&value).then_some(Self(value))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// How a span finished — mapped onto OTLP status codes by the sidecar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanStatus {
    #[default]
    Ok,
    Error,
}

/// One telemetry record — a `status.jsonl` line the sidecar turns into a
/// scrape sample or an OTLP span.
///
/// `#[non_exhaustive]` because the sidecar must tolerate records emitted by a
/// NEWER engine: an unknown variant is skipped, never a parse failure that
/// stalls the whole tail.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "telemetry", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TelemetryRecord {
    /// Monotonically increasing count (Prometheus `counter`).
    Counter {
        name: MetricName,
        value: u64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<Label>,
    },
    /// Instantaneous sample (Prometheus `gauge`) — broker headroom, cache
    /// hit-rate, the ε privacy budget.
    Gauge {
        name: MetricName,
        value: f64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<Label>,
    },
    /// An observation fed into a duration histogram (Prometheus `histogram`);
    /// the sidecar owns bucketing so the engine stays allocation-cheap.
    Duration {
        name: MetricName,
        seconds: f64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<Label>,
    },
    /// A span opened.
    SpanStart {
        #[serde(flatten)]
        ctx: SpanContext,
        name: String,
        start_unix_nanos: u64,
    },
    /// A span closed. Carries its own duration so a consumer never has to
    /// correlate against a start it may not have seen (a tail can begin
    /// mid-stream).
    SpanEnd {
        #[serde(flatten)]
        ctx: SpanContext,
        name: String,
        end_unix_nanos: u64,
        duration_nanos: u64,
        #[serde(default)]
        status: SpanStatus,
    },
}

impl TelemetryRecord {
    /// The span context, when this record is a span event.
    pub fn span_context(&self) -> Option<&SpanContext> {
        match self {
            TelemetryRecord::SpanStart { ctx, .. } | TelemetryRecord::SpanEnd { ctx, .. } => {
                Some(ctx)
            }
            _ => None,
        }
    }

    /// Serialize to one `status.jsonl` line (no trailing newline).
    pub fn to_line(&self) -> String {
        serde_json::to_string(self).expect("TelemetryRecord serializes")
    }

    /// Parse one line. `None` for a line that is not a telemetry record (the
    /// stream is shared with stage events) or is from a newer schema.
    pub fn from_line(line: &str) -> Option<Self> {
        serde_json::from_str(line).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_and_span_ids_enforce_w3c_shape() {
        assert!(TraceId::parse("4bf92f3577b34da6a3ce929d0e0e4736").is_some());
        assert!(
            TraceId::parse("4BF92F3577B34DA6A3CE929D0E0E4736").is_none(),
            "uppercase"
        );
        assert!(TraceId::parse("4bf92f3577b34da6").is_none(), "too short");
        assert!(
            TraceId::parse("00000000000000000000000000000000").is_none(),
            "all-zero would fuse unrelated runs into one trace"
        );
        assert!(SpanId::parse("00f067aa0ba902b7").is_some());
        assert!(SpanId::parse("0000000000000000").is_none());
        assert!(SpanId::parse("00f067aa0ba902b7ff").is_none(), "too long");
    }

    #[test]
    fn child_stays_in_the_same_trace_and_parents_correctly() {
        let root = SpanContext::root(
            TraceId::parse("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::parse("00f067aa0ba902b7").unwrap(),
        );
        assert!(root.parent_span_id.is_none());
        let child = root.child(SpanId::parse("1122334455667788").unwrap());
        assert_eq!(child.trace_id, root.trace_id, "one trace across the hop");
        assert_eq!(child.parent_span_id.as_ref(), Some(&root.span_id));
    }

    #[test]
    fn names_are_validated_against_the_prometheus_grammar() {
        assert!(MetricName::parse("blut_stage_duration_seconds").is_some());
        assert!(MetricName::parse("blut:ratio").is_some(), "colons allowed");
        assert!(MetricName::parse("1_leading_digit").is_none());
        assert!(MetricName::parse("has-dash").is_none());
        assert!(MetricName::parse("").is_none());
        assert!(Label::new("stage", "train").is_some());
        assert!(
            Label::new("has:colon", "x").is_none(),
            "labels forbid colons"
        );
        assert!(Label::new("has-dash", "x").is_none());
    }

    #[test]
    fn records_round_trip_through_a_status_line() {
        let ctx = SpanContext::root(
            TraceId::parse("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::parse("00f067aa0ba902b7").unwrap(),
        );
        for record in [
            TelemetryRecord::Counter {
                name: MetricName::parse("blut_cache_hits_total").unwrap(),
                value: 7,
                labels: vec![Label::new("tenant", "shared").unwrap()],
            },
            TelemetryRecord::Gauge {
                name: MetricName::parse("blut_privacy_epsilon").unwrap(),
                value: 0.5,
                labels: vec![],
            },
            TelemetryRecord::Duration {
                name: MetricName::parse("blut_stage_duration_seconds").unwrap(),
                seconds: 1.25,
                labels: vec![],
            },
            TelemetryRecord::SpanEnd {
                ctx: ctx.clone(),
                name: "stage:train".into(),
                end_unix_nanos: 1_700_000_000_000_000_000,
                duration_nanos: 5_000_000,
                status: SpanStatus::Error,
            },
        ] {
            let line = record.to_line();
            assert_eq!(TelemetryRecord::from_line(&line), Some(record));
        }
    }

    #[test]
    fn a_non_telemetry_line_is_skipped_not_an_error() {
        // status.jsonl interleaves stage events; the tail must ignore them.
        assert!(TelemetryRecord::from_line(r#"{"kind":"stage_begin","node_idx":0}"#).is_none());
        assert!(TelemetryRecord::from_line("not json").is_none());
    }
}
