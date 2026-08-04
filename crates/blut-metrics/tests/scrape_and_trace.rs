// SPDX-License-Identifier: AGPL-3.0-or-later
// The ADR 0097 acceptance gate. A 2-HOST MESH fixture is ingested, the REAL
// scrape endpoint is served over a REAL socket and scraped over TCP, and we
// assert:
//   1. broker / stage / mesh / cache / ε gauges are present with NON-DEFAULT
//      values, plus a stage-duration histogram;
//   2. the body is OpenMetrics-conformant (promtool is not installed here, so
//      the grammar check is implemented in-test rather than silently skipped —
//      a gate that degrades quietly is worse than one that fails loudly);
//   3. the spans for a HOST-HOPPING stage share ONE trace_id with a correct
//      parent chain — the property that makes a mesh run one waterfall instead
//      of N per-host fragments.

use std::sync::{Arc, Mutex};

use blut_metrics::{CapturingExporter, MetricStore, SpanExporter, otlp_json};
use blut_types::telemetry::{
    Label, MetricName, SpanContext, SpanId, SpanStatus, TelemetryRecord, TraceId,
};

/// Build the status.jsonl a 2-host mesh run would leave behind: host A admits
/// and dispatches, host B executes the hop, and both emit telemetry into the
/// initiator's stream (the ADR 0083 D5 forwarding shape).
fn two_host_mesh_fixture() -> (String, SpanContext, SpanContext) {
    let trace = TraceId::parse("4bf92f3577b34da6a3ce929d0e0e4736").unwrap();
    let root = SpanContext::root(trace, SpanId::parse("00f067aa0ba902b7").unwrap());
    // The remote stage is a CHILD of the dispatching span — same trace.
    let hop = root.child(SpanId::parse("1122334455667788").unwrap());

    let host_a = Label::new("host", "hosta").unwrap();
    let host_b = Label::new("host", "hostb").unwrap();
    let mut lines: Vec<String> = Vec::new();

    // Interleave real stage events: the sidecar shares this stream and must
    // skip what isn't telemetry.
    lines.push(
        r#"{"kind":"stage_begin","node_idx":0,"stage_name":"train","input_hash":"ab"}"#.into(),
    );

    for (name, value, label) in [
        (
            "blut_broker_headroom_bytes",
            12.0_f64 * 1024.0 * 1024.0 * 1024.0,
            &host_a,
        ),
        ("blut_mesh_dispatch_rtt_seconds", 0.042, &host_a),
        ("blut_cache_hit_ratio", 0.75, &host_a),
        ("blut_privacy_epsilon", 0.5, &host_a),
        ("blut_stage_active", 1.0, &host_b),
    ] {
        lines.push(
            TelemetryRecord::Gauge {
                name: MetricName::parse(name).unwrap(),
                value,
                labels: vec![label.clone()],
            }
            .to_line(),
        );
    }
    lines.push(
        TelemetryRecord::Counter {
            name: MetricName::parse("blut_cache_hits_total").unwrap(),
            value: 12,
            labels: vec![host_a.clone()],
        }
        .to_line(),
    );
    for seconds in [0.02_f64, 1.5, 42.0] {
        lines.push(
            TelemetryRecord::Duration {
                name: MetricName::parse("blut_stage_duration_seconds").unwrap(),
                seconds,
                labels: vec![host_b.clone()],
            }
            .to_line(),
        );
    }
    // The host-hopping stage: dispatch span on A, execution span on B.
    lines.push(
        TelemetryRecord::SpanEnd {
            ctx: hop.clone(),
            name: "stage:train@hostb".into(),
            end_unix_nanos: 2_000_000_000,
            duration_nanos: 900_000_000,
            status: SpanStatus::Ok,
        }
        .to_line(),
    );
    lines.push(
        TelemetryRecord::SpanEnd {
            ctx: root.clone(),
            name: "run:dispatch@hosta".into(),
            end_unix_nanos: 2_500_000_000,
            duration_nanos: 2_000_000_000,
            status: SpanStatus::Ok,
        }
        .to_line(),
    );
    (lines.join("\n"), root, hop)
}

/// Minimal OpenMetrics conformance check (stand-in for promtool, which is not
/// installed in this environment). Verifies the properties a strict parser
/// rejects a whole body over.
fn assert_openmetrics_conformant(body: &str) {
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(
        lines.last(),
        Some(&"# EOF"),
        "OpenMetrics REQUIRES a trailing # EOF"
    );
    let mut declared: Vec<(String, String)> = Vec::new();
    for line in &lines {
        if *line == "# EOF" {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let mut parts = rest.split_whitespace();
            let family = parts.next().expect("# TYPE needs a family").to_string();
            let kind = parts.next().expect("# TYPE needs a kind").to_string();
            assert!(
                ["counter", "gauge", "histogram", "summary", "info"].contains(&kind.as_str()),
                "unknown metric type {kind}"
            );
            declared.push((family, kind));
            continue;
        }
        assert!(!line.starts_with('#'), "unexpected comment: {line}");
        // sample: name[{labels}] value
        let (name_part, value_part) = line.rsplit_once(' ').expect("sample needs a value");
        let name = name_part.split('{').next().unwrap();
        let mut chars = name.chars();
        let first = chars.next().expect("empty metric name");
        assert!(
            first.is_ascii_alphabetic() || first == '_' || first == ':',
            "bad metric name start in {line}"
        );
        assert!(
            chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':'),
            "bad metric name in {line}"
        );
        assert!(
            value_part.parse::<f64>().is_ok() || ["+Inf", "-Inf", "NaN"].contains(&value_part),
            "bad sample value in {line}"
        );
        // Every sample must belong to a declared family.
        assert!(
            declared.iter().any(|(family, _)| name == family
                || name == format!("{family}_total")
                || name == format!("{family}_bucket")
                || name == format!("{family}_sum")
                || name == format!("{family}_count")),
            "sample {name} has no preceding # TYPE"
        );
        // Balanced label braces.
        assert_eq!(
            name_part.matches('{').count(),
            name_part.matches('}').count(),
            "unbalanced labels in {line}"
        );
    }
    // Every histogram family must expose the mandatory +Inf bucket.
    for (family, kind) in &declared {
        if kind == "histogram" {
            assert!(
                body.contains(&format!("{family}_bucket{{")) && body.contains(r#"le="+Inf""#),
                "histogram {family} is missing its mandatory +Inf bucket"
            );
        }
    }
}

#[tokio::test]
async fn scrape_and_trace() {
    let (fixture, root, hop) = two_host_mesh_fixture();

    // ── ingest the mesh fixture ────────────────────────────────────────
    let mut store = MetricStore::new();
    for line in fixture.lines() {
        store.ingest_line(line);
    }
    assert!(
        store.skipped() >= 1,
        "the interleaved stage event must be skipped, not fatal"
    );

    // ── spans: one trace, correct parent chain (the mesh property) ─────
    let spans = store.drain_spans();
    assert_eq!(spans.len(), 2, "one span per host");
    let trace_ids: std::collections::BTreeSet<_> =
        spans.iter().map(|s| s.ctx.trace_id.as_str()).collect();
    assert_eq!(
        trace_ids.len(),
        1,
        "a host-hopping stage must stay ONE trace, got {trace_ids:?}"
    );
    let remote = spans
        .iter()
        .find(|s| s.name.contains("hostb"))
        .expect("the remote span");
    assert_eq!(
        remote.ctx.parent_span_id.as_ref().map(|s| s.as_str()),
        Some(root.span_id.as_str()),
        "the remote span must parent onto the dispatching span"
    );
    assert_eq!(remote.ctx.span_id.as_str(), hop.span_id.as_str());
    let initiator = spans
        .iter()
        .find(|s| s.name.contains("hosta"))
        .expect("the initiator span");
    assert!(
        initiator.ctx.parent_span_id.is_none(),
        "the initiating span is the trace root"
    );

    // The OTLP payload preserves that chain for a collector.
    let mut exporter = CapturingExporter::default();
    assert_eq!(exporter.export(&spans).unwrap(), 2);
    let payload = otlp_json(&exporter.exported, "blut");
    let otlp = &payload["resourceSpans"][0]["scopeSpans"][0]["spans"];
    assert_eq!(otlp[0]["traceId"], otlp[1]["traceId"], "one trace in OTLP");

    // ── serve the REAL router on a REAL socket and scrape it ───────────
    let mut store = MetricStore::new();
    for line in fixture.lines() {
        store.ingest_line(line);
    }
    let shared = Arc::new(Mutex::new(store));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = blut_metrics::router(shared);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let (status, content_type, body) = scrape(addr).await;
    assert_eq!(status, 200, "scrape must succeed");
    assert!(
        content_type.contains("openmetrics-text"),
        "must negotiate OpenMetrics, got {content_type}"
    );

    // ── the gauges/histogram the ADR names, with NON-DEFAULT values ────
    for (needle, why) in [
        ("blut_broker_headroom_bytes", "broker"),
        ("blut_stage_active", "stage"),
        ("blut_mesh_dispatch_rtt_seconds", "mesh"),
        ("blut_cache_hit_ratio", "cache"),
        ("blut_privacy_epsilon", "ε privacy budget"),
        (
            "blut_stage_duration_seconds_bucket",
            "stage-duration histogram",
        ),
    ] {
        assert!(
            body.contains(needle),
            "{why} gauge/histogram missing:\n{body}"
        );
    }
    // Non-default: a zeroed series would pass a presence check while telling
    // an operator nothing.
    assert!(
        body.contains("blut_privacy_epsilon{host=\"hosta\"} 0.5"),
        "{body}"
    );
    assert!(
        body.contains("blut_cache_hit_ratio{host=\"hosta\"} 0.75"),
        "{body}"
    );
    assert!(
        body.contains("blut_cache_hits_total{host=\"hosta\"} 12"),
        "{body}"
    );
    assert!(
        body.contains("blut_stage_duration_seconds_count{host=\"hostb\"} 3"),
        "histogram must have observed all three durations:\n{body}"
    );

    // ── format conformance (promtool stand-in) ─────────────────────────
    assert_openmetrics_conformant(&body);
}

/// Raw HTTP/1.1 GET so the gate depends on no HTTP client crate.
async fn scrape(addr: std::net::SocketAddr) -> (u16, String, String) {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    sock.write_all(
        b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nAccept: application/openmetrics-text\r\nConnection: close\r\n\r\n",
    )
    .await
    .unwrap();
    let mut raw = Vec::new();
    sock.read_to_end(&mut raw).await.unwrap();
    let text = String::from_utf8_lossy(&raw).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").expect("HTTP response");
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    let content_type = head
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-type:"))
        .unwrap_or("")
        .to_string();
    (status, content_type, body.to_string())
}

/// The conformance checker must REJECT malformed bodies — otherwise the gate
/// above would pass on anything and prove nothing.
#[test]
fn the_conformance_check_is_not_vacuous() {
    let bad_bodies = [
        // missing the mandatory # EOF terminator
        "# TYPE g gauge\ng 1\n",
        // sample with no preceding # TYPE
        "orphan_metric 1\n# EOF",
        // illegal metric name (leading digit)
        "# TYPE 1bad gauge\n1bad 1\n# EOF",
        // non-numeric sample value
        "# TYPE g gauge\ng notanumber\n# EOF",
        // unknown metric type
        "# TYPE g mystery\ng 1\n# EOF",
    ];
    for body in bad_bodies {
        let caught = std::panic::catch_unwind(|| assert_openmetrics_conformant(body));
        assert!(
            caught.is_err(),
            "conformance check wrongly ACCEPTED a malformed body:\n{body}"
        );
    }
    // ...and accepts a well-formed one.
    assert_openmetrics_conformant("# TYPE g gauge\ng{a=\"1\"} 1.5\n# EOF");
}
