// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! blut-metrics (ADR 0097) — the telemetry sidecar's pure core.
//!
//! Ingests the engine's `status.jsonl` telemetry records ([`blut_types::telemetry`]),
//! aggregates them, and renders an OpenMetrics scrape body; separately it
//! reconstructs spans for an OTLP forwarder. Everything here is I/O-free and
//! synchronous so it is testable without a socket — the binary owns the
//! listener and the collector connection, which is the whole reason this code
//! lives outside the engine (ADR 0034/0083).

use std::collections::BTreeMap;

use blut_types::telemetry::{Label, SpanContext, SpanStatus, TelemetryRecord};

/// Histogram bucket upper bounds in seconds. Fixed rather than configurable:
/// a scrape endpoint whose buckets change between restarts produces
/// un-comparable time series, which is worse than imperfect resolution.
pub const DURATION_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 1800.0,
];

/// Identity of one time series: metric name plus its sorted label set.
/// Labels are sorted so two records with the same labels in a different order
/// land on the SAME series instead of silently forking one metric into two.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SeriesKey {
    name: String,
    labels: Vec<(String, String)>,
}

impl SeriesKey {
    fn new(name: &str, labels: &[Label]) -> Self {
        let mut labels: Vec<(String, String)> = labels
            .iter()
            .map(|l| (l.name.clone(), l.value.clone()))
            .collect();
        labels.sort();
        Self {
            name: name.to_string(),
            labels,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Histogram {
    /// Cumulative counts per `DURATION_BUCKETS` index.
    buckets: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn observe(&mut self, seconds: f64) {
        if self.buckets.is_empty() {
            self.buckets = vec![0; DURATION_BUCKETS.len()];
        }
        for (idx, bound) in DURATION_BUCKETS.iter().enumerate() {
            if seconds <= *bound {
                self.buckets[idx] += 1;
            }
        }
        self.sum += seconds;
        self.count += 1;
    }
}

/// A span reconstructed from a `SpanEnd` record.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub ctx: SpanContext,
    pub name: String,
    pub end_unix_nanos: u64,
    pub duration_nanos: u64,
    pub status: SpanStatus,
}

/// Aggregated telemetry state. Feed it lines; render a scrape body.
#[derive(Debug, Default)]
pub struct MetricStore {
    counters: BTreeMap<SeriesKey, u64>,
    gauges: BTreeMap<SeriesKey, f64>,
    histograms: BTreeMap<SeriesKey, Histogram>,
    spans: Vec<Span>,
    /// Lines that parsed as JSON but were not telemetry — the stream is shared
    /// with stage events, so this is expected traffic, not an error rate.
    skipped: u64,
}

impl MetricStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one `status.jsonl` line. Non-telemetry lines are counted and
    /// skipped: the sidecar tails a stream it shares with stage events, and a
    /// newer engine may emit variants this build does not know.
    pub fn ingest_line(&mut self, line: &str) {
        match TelemetryRecord::from_line(line) {
            Some(record) => self.ingest(record),
            None => self.skipped += 1,
        }
    }

    pub fn ingest(&mut self, record: TelemetryRecord) {
        match record {
            TelemetryRecord::Counter {
                name,
                value,
                labels,
            } => {
                let key = SeriesKey::new(name.as_str(), &labels);
                // Counters are CUMULATIVE: take the max so an out-of-order or
                // replayed line can never walk a counter backwards, which
                // Prometheus would read as a counter reset.
                let slot = self.counters.entry(key).or_insert(0);
                *slot = (*slot).max(value);
            }
            TelemetryRecord::Gauge {
                name,
                value,
                labels,
            } => {
                self.gauges
                    .insert(SeriesKey::new(name.as_str(), &labels), value);
            }
            TelemetryRecord::Duration {
                name,
                seconds,
                labels,
            } => {
                self.histograms
                    .entry(SeriesKey::new(name.as_str(), &labels))
                    .or_default()
                    .observe(seconds);
            }
            TelemetryRecord::SpanStart { .. } => {
                // A start alone carries no duration; SpanEnd is self-contained
                // (ADR 0097) so a tail beginning mid-stream still exports.
            }
            TelemetryRecord::SpanEnd {
                ctx,
                name,
                end_unix_nanos,
                duration_nanos,
                status,
            } => self.spans.push(Span {
                ctx,
                name,
                end_unix_nanos,
                duration_nanos,
                status,
            }),
            _ => self.skipped += 1,
        }
    }

    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// Take the accumulated spans, leaving the store empty — the forwarder
    /// drains so a span is exported at most once.
    pub fn drain_spans(&mut self) -> Vec<Span> {
        std::mem::take(&mut self.spans)
    }

    /// Render an OpenMetrics exposition body.
    ///
    /// OpenMetrics requires the `# EOF` terminator and names a COUNTER family
    /// without its `_total` sample suffix, so `blut_x_total` is exposed as
    /// family `blut_x` with sample `blut_x_total`. Getting that wrong makes an
    /// otherwise-valid body rejected wholesale by a strict parser.
    pub fn render_openmetrics(&self) -> String {
        let mut out = String::new();
        for (key, value) in &self.counters {
            let family = key.name.strip_suffix("_total").unwrap_or(&key.name);
            out.push_str(&format!("# TYPE {family} counter\n"));
            out.push_str(&format!(
                "{}_total{} {}\n",
                family,
                render_labels(&key.labels),
                value
            ));
        }
        for (key, value) in &self.gauges {
            out.push_str(&format!("# TYPE {} gauge\n", key.name));
            out.push_str(&format!(
                "{}{} {}\n",
                key.name,
                render_labels(&key.labels),
                render_float(*value)
            ));
        }
        for (key, hist) in &self.histograms {
            out.push_str(&format!("# TYPE {} histogram\n", key.name));
            for (idx, bound) in DURATION_BUCKETS.iter().enumerate() {
                let count = hist.buckets.get(idx).copied().unwrap_or(0);
                out.push_str(&format!(
                    "{}_bucket{} {}\n",
                    key.name,
                    render_labels_with(&key.labels, Some(("le", &render_float(*bound)))),
                    count
                ));
            }
            // The +Inf bucket is MANDATORY and must equal the total count.
            out.push_str(&format!(
                "{}_bucket{} {}\n",
                key.name,
                render_labels_with(&key.labels, Some(("le", "+Inf"))),
                hist.count
            ));
            out.push_str(&format!(
                "{}_sum{} {}\n",
                key.name,
                render_labels(&key.labels),
                render_float(hist.sum)
            ));
            out.push_str(&format!(
                "{}_count{} {}\n",
                key.name,
                render_labels(&key.labels),
                hist.count
            ));
        }
        out.push_str("# EOF\n");
        out
    }
}

fn render_labels(labels: &[(String, String)]) -> String {
    render_labels_with(labels, None)
}

fn render_labels_with(labels: &[(String, String)], extra: Option<(&str, &str)>) -> String {
    if labels.is_empty() && extra.is_none() {
        return String::new();
    }
    let mut parts: Vec<String> = labels
        .iter()
        .map(|(n, v)| format!("{n}=\"{}\"", escape_label_value(v)))
        .collect();
    if let Some((n, v)) = extra {
        parts.push(format!("{n}=\"{}\"", escape_label_value(v)));
    }
    format!("{{{}}}", parts.join(","))
}

/// Escape per the exposition format: backslash, double-quote, and newline.
/// Unescaped, a label value containing `"` would terminate the value early and
/// corrupt every following sample in the body.
fn escape_label_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out
}

/// Format a float the way the exposition format expects, including the special
/// values (`+Inf` / `-Inf` / `NaN`) that a plain `{}` would render wrongly.
fn render_float(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "+Inf".to_string()
        } else {
            "-Inf".to_string()
        }
    } else {
        format!("{value}")
    }
}

/// The scrape router. Lives in the lib so the acceptance gate exercises the
/// REAL endpoint over a real socket — a gate that scrapes a stand-in proves
/// nothing about what an operator's Prometheus will receive.
pub fn router(store: std::sync::Arc<std::sync::Mutex<MetricStore>>) -> axum::Router {
    use axum::routing::get;
    axum::Router::new()
        .route("/metrics", get(scrape))
        .route("/healthz", get(|| async { "ok" }))
        .with_state(store)
}

/// OpenMetrics content type, exactly as a scraper negotiates it.
pub const OPENMETRICS_CONTENT_TYPE: &str =
    "application/openmetrics-text; version=1.0.0; charset=utf-8";

async fn scrape(
    axum::extract::State(store): axum::extract::State<
        std::sync::Arc<std::sync::Mutex<MetricStore>>,
    >,
) -> ([(&'static str, &'static str); 1], String) {
    let body = store
        .lock()
        .map(|s| s.render_openmetrics())
        .unwrap_or_else(|_| "# EOF\n".to_string());
    ([("content-type", OPENMETRICS_CONTENT_TYPE)], body)
}

/// Where reconstructed spans go. A trait so the acceptance gate can capture
/// them without a collector, and so a collector outage can never reach back
/// into the engine's hot path.
pub trait SpanExporter {
    fn export(&mut self, spans: &[Span]) -> Result<usize, String>;
}

/// Test/inspection exporter that keeps everything handed to it.
#[derive(Debug, Default)]
pub struct CapturingExporter {
    pub exported: Vec<Span>,
}

impl SpanExporter for CapturingExporter {
    fn export(&mut self, spans: &[Span]) -> Result<usize, String> {
        self.exported.extend_from_slice(spans);
        Ok(spans.len())
    }
}

/// Render spans as an OTLP/JSON `ExportTraceServiceRequest` body. Kept
/// data-only (no HTTP here) so the payload shape is unit-testable and the
/// binary decides when and where to POST it.
pub fn otlp_json(spans: &[Span], service_name: &str) -> serde_json::Value {
    let otlp_spans: Vec<serde_json::Value> = spans
        .iter()
        .map(|s| {
            let mut span = serde_json::json!({
                "traceId": s.ctx.trace_id.as_str(),
                "spanId": s.ctx.span_id.as_str(),
                "name": s.name,
                "startTimeUnixNano": (s.end_unix_nanos.saturating_sub(s.duration_nanos)).to_string(),
                "endTimeUnixNano": s.end_unix_nanos.to_string(),
                // OTLP status codes: 0 unset, 1 ok, 2 error.
                "status": { "code": if s.status == SpanStatus::Error { 2 } else { 1 } },
            });
            if let Some(parent) = &s.ctx.parent_span_id {
                span["parentSpanId"] = serde_json::Value::String(parent.as_str().to_string());
            }
            span
        })
        .collect();
    serde_json::json!({
        "resourceSpans": [{
            "resource": { "attributes": [{
                "key": "service.name",
                "value": { "stringValue": service_name }
            }]},
            "scopeSpans": [{ "spans": otlp_spans }]
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut_types::telemetry::{MetricName, SpanId, TraceId};

    fn gauge(name: &str, value: f64) -> TelemetryRecord {
        TelemetryRecord::Gauge {
            name: MetricName::parse(name).unwrap(),
            value,
            labels: vec![],
        }
    }

    #[test]
    fn counters_never_walk_backwards() {
        let mut store = MetricStore::new();
        for value in [5_u64, 9, 7] {
            store.ingest(TelemetryRecord::Counter {
                name: MetricName::parse("blut_cache_hits_total").unwrap(),
                value,
                labels: vec![],
            });
        }
        // A replayed/out-of-order line must not look like a counter RESET.
        assert!(
            store
                .render_openmetrics()
                .contains("blut_cache_hits_total 9")
        );
    }

    #[test]
    fn label_order_does_not_fork_a_series() {
        let mut a = MetricStore::new();
        a.ingest(TelemetryRecord::Gauge {
            name: MetricName::parse("g").unwrap(),
            value: 1.0,
            labels: vec![Label::new("b", "2").unwrap(), Label::new("a", "1").unwrap()],
        });
        a.ingest(TelemetryRecord::Gauge {
            name: MetricName::parse("g").unwrap(),
            value: 2.0,
            labels: vec![Label::new("a", "1").unwrap(), Label::new("b", "2").unwrap()],
        });
        let body = a.render_openmetrics();
        assert_eq!(
            body.matches("# TYPE g gauge").count(),
            1,
            "one series: {body}"
        );
        assert!(body.contains(r#"g{a="1",b="2"} 2"#), "{body}");
    }

    #[test]
    fn counter_family_drops_the_total_suffix_for_openmetrics() {
        let mut store = MetricStore::new();
        store.ingest(TelemetryRecord::Counter {
            name: MetricName::parse("blut_runs_total").unwrap(),
            value: 3,
            labels: vec![],
        });
        let body = store.render_openmetrics();
        assert!(body.contains("# TYPE blut_runs counter"), "{body}");
        assert!(body.contains("blut_runs_total 3"), "{body}");
    }

    #[test]
    fn histogram_has_cumulative_buckets_and_a_mandatory_inf() {
        let mut store = MetricStore::new();
        for seconds in [0.002_f64, 0.2, 7.0] {
            store.ingest(TelemetryRecord::Duration {
                name: MetricName::parse("blut_stage_duration_seconds").unwrap(),
                seconds,
                labels: vec![],
            });
        }
        let body = store.render_openmetrics();
        assert!(
            body.contains(r#"blut_stage_duration_seconds_bucket{le="0.005"} 1"#),
            "{body}"
        );
        assert!(
            body.contains(r#"blut_stage_duration_seconds_bucket{le="0.5"} 2"#),
            "{body}"
        );
        assert!(
            body.contains(r#"blut_stage_duration_seconds_bucket{le="+Inf"} 3"#),
            "{body}"
        );
        assert!(
            body.contains("blut_stage_duration_seconds_count 3"),
            "{body}"
        );
    }

    #[test]
    fn label_values_are_escaped_so_one_value_cannot_corrupt_the_body() {
        let mut store = MetricStore::new();
        store.ingest(TelemetryRecord::Gauge {
            name: MetricName::parse("g").unwrap(),
            value: 1.0,
            labels: vec![Label::new("stage", "a\"b\\c\nd").unwrap()],
        });
        let body = store.render_openmetrics();
        assert!(body.contains(r#"g{stage="a\"b\\c\nd"} 1"#), "{body}");
        assert_eq!(body.lines().filter(|l| l.starts_with("g{")).count(), 1);
    }

    #[test]
    fn non_telemetry_lines_are_skipped_not_fatal() {
        let mut store = MetricStore::new();
        store.ingest_line(r#"{"kind":"stage_begin","node_idx":0,"stage_name":"t"}"#);
        store.ingest_line("garbage");
        store.ingest_line(&gauge("blut_ok", 1.0).to_line());
        assert_eq!(store.skipped(), 2);
        assert!(store.render_openmetrics().contains("blut_ok 1"));
    }

    #[test]
    fn otlp_payload_carries_the_parent_chain() {
        let root = SpanContext::root(
            TraceId::parse("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::parse("00f067aa0ba902b7").unwrap(),
        );
        let child = root.child(SpanId::parse("1122334455667788").unwrap());
        let spans = vec![
            Span {
                ctx: root.clone(),
                name: "root".into(),
                end_unix_nanos: 1_000_000_000,
                duration_nanos: 500_000_000,
                status: SpanStatus::Ok,
            },
            Span {
                ctx: child,
                name: "child".into(),
                end_unix_nanos: 1_000_000_000,
                duration_nanos: 250_000_000,
                status: SpanStatus::Error,
            },
        ];
        let payload = otlp_json(&spans, "blut");
        let out = &payload["resourceSpans"][0]["scopeSpans"][0]["spans"];
        assert!(out[0].get("parentSpanId").is_none(), "root has no parent");
        assert_eq!(out[1]["parentSpanId"], "00f067aa0ba902b7");
        assert_eq!(out[1]["traceId"], out[0]["traceId"], "one trace");
        assert_eq!(out[1]["status"]["code"], 2, "error maps to OTLP 2");
        assert_eq!(out[0]["startTimeUnixNano"], "500000000");
    }

    #[test]
    fn drain_exports_each_span_at_most_once() {
        let mut store = MetricStore::new();
        store.ingest(TelemetryRecord::SpanEnd {
            ctx: SpanContext::root(
                TraceId::parse("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
                SpanId::parse("00f067aa0ba902b7").unwrap(),
            ),
            name: "s".into(),
            end_unix_nanos: 1,
            duration_nanos: 1,
            status: SpanStatus::Ok,
        });
        let mut exporter = CapturingExporter::default();
        exporter.export(&store.drain_spans()).unwrap();
        exporter.export(&store.drain_spans()).unwrap();
        assert_eq!(exporter.exported.len(), 1, "no double export");
    }
}
