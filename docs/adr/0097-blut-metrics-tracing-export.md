<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0097 — Export Prometheus/OpenMetrics and OpenTelemetry telemetry from a sidecar crate so the engine stays HTTP-free

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-08-03`

## Context

Every competing orchestrator ships first-class metrics and distributed tracing:
Dagster, Flyte, Prefect, and Temporal all expose a Prometheus `/metrics`
endpoint and emit OpenTelemetry spans that stitch a run into Grafana/Tempo/Jaeger
dashboards out of the box. BLUT exposes **none** of this. Today an operator
reconstructs run health by tailing `status.jsonl` ([ADR 0067](0067-remaining-work-toward-vision.md)) and reading the
lineage SQLite (src/lineage_db.rs) by hand — there is no time-series of broker
admission latency, mesh-dispatch RTT, cache hit-rate, per-stage duration, or the
ε privacy-budget gauge, and no way to trace a single stage across the hosts it
hopped through on the P2P mesh ([ADR 0061](0061-blut-p2p-compute.md)). This is genuine white-space, not a
re-skin of an existing surface. The hard constraint is [ADR 0034](0034-blut-scope-charter.md): the engine
source tree (`src`) must not host an HTTP server or a scrape endpoint, and
per [ADR 0083](0083-blut-interface-stack.md) all server/network concerns live in opt-in sidecar crates.

## Decision

Ship telemetry from a new sidecar crate `crates/blut-metrics` (dispatched as
`blut metrics` per the [ADR 0083](0083-blut-interface-stack.md) external-subcommand convention), never from the
engine. The engine's only new responsibility is to **write** structured
telemetry records — counters, gauge samples, and span start/end events with a
`trace_id`/`span_id`/`parent_span_id` triple — into the existing
`status.jsonl` stream and lineage tables it already writes; these are typed in
`blut-types` (wasm-safe, serde-only) so both engine and sidecar share one wire
schema. `blut-metrics` tails those records and (a) serves a
Prometheus/OpenMetrics `/metrics` scrape endpoint exposing run/stage/broker/
mesh/cache/ε gauges and duration histograms, and (b) forwards reconstructed
spans to an OTLP collector, propagating trace context across mesh dispatch by
threading the `trace_id`/`parent_span_id` through the `HostedEvent` envelope so a
stage that hops hosts stays one connected trace. The engine never opens a socket;
the sidecar owns every byte of network I/O.

## Rationale

- **White-space parity at zero charter cost.** The engine already writes
  `status.jsonl` and lineage rows on the hot path; adding a monotonic
  `trace_id`/`span_id` triple and periodic gauge samples to records it already
  emits is a data change, not a server. All socket-facing work is quarantined in
  the sidecar, exactly where [ADR 0034](0034-blut-scope-charter.md)/0083 says it belongs.
- **Trace-context propagation is the differentiator.** A mesh run today is
  observable only per-host; threading the parent span id through `HostedEvent`
  turns N per-host fragments into one Jaeger waterfall — the thing operators
  actually reach for when a remote stage stalls.
- **One source of truth.** The sidecar derives metrics from the same
  `status.jsonl`/lineage artifacts the TUI and `blut sla check` read, so
  Grafana, the TUI, and CI cannot disagree about what happened.
- **Broker/ε visibility closes a governance gap.** Surfacing broker admission
  latency and the ε privacy-budget gauge as time-series makes the fail-closed
  guarantees ([ADR 0046](0046-blut-resource-broker.md)/0047/0061) auditable over time, not just at the instant of
  refusal.

## Alternatives Considered

- **Embed a `metrics`/`axum` exporter directly in the engine (the Dagster/Flyte
  shape).** Rejected outright: it puts a listening socket inside `src`,
  which the scope guard ([ADR 0083](0083-blut-interface-stack.md)) fails CI on. The whole point of the sidecar
  boundary is that this is not negotiable.
- **Push metrics from the engine straight to an OTLP collector over gRPC.** Still
  an outbound network client compiled into the engine, still a charter
  violation, and it couples the engine to the collector's availability on the hot
  path — a slow collector would backpressure stage execution. Writing to
  `status.jsonl` and letting the sidecar forward decouples the two.
- **StatsD/UDP fire-and-forget from the engine.** Lossy by design (drops under
  load exactly when you need the data), no histograms, and still an in-engine
  network dependency. Rejected.

## Consequences

- Upside: BLUT reaches Prometheus + OTel parity with the incumbents; mesh runs
  become traceable end-to-end; broker/cache/ε metrics become dashboardable and
  alertable via `blut-notify` ([ADR 0083](0083-blut-interface-stack.md)) SLA rules.
- Upside: `blut-types` becomes the single telemetry schema; the web dashboard
  ([ADR 0093](0093-blut-web-control-plane.md)) can read the same gauges without a second data path.
- Downside: the engine gains a small always-on cost — emitting gauge samples and
  span ids on the hot path (bounded to O(1) per stage-event; sampling rate is a
  config knob so it can be turned down).
- Downside: a new lockstep constraint — the telemetry record schema in
  `blut-types` is now shared by engine + sidecar and changes require a
  Progress-Log entry here.
- Load-bearing: `HostedEvent` gains `trace_id`/`parent_span_id` fields; anything
  that constructs a `HostedEvent` must carry them forward or the trace breaks.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** `crates/blut-metrics` sidecar binary serving an
  OpenMetrics-conformant `/metrics` endpoint and an OTLP span forwarder, driven
  entirely by engine-written telemetry records in `status.jsonl` + lineage_db;
  telemetry record types added to `blut-types`; `trace_id`/`parent_span_id`
  threaded through `HostedEvent` across mesh dispatch.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `scripts/scope_guard.sh && cargo test -p blut-metrics --test scrape_and_trace`
  - `pass when:` `exit==0` — scope guard confirms zero listening sockets in
    `src`; the test runs a 2-host mesh fixture, scrapes `/metrics` and
    asserts the broker/stage/mesh/cache/ε gauges + stage-duration histogram are
    present with non-default values, and asserts the exported spans for a
    host-hopping stage share one `trace_id` with a correct parent chain
    (promtool/OpenMetrics-format check on the scrape body).
- **Rollback:** delete `crates/blut-metrics` and revert the `blut-types`
  telemetry records + `HostedEvent` field additions; the engine falls back to
  raw `status.jsonl` tailing with no behavioral change (the fields are additive).

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-08-03 — BUILT + FROZEN in three increments (engine `9f840b6` →
  `3de67b9` → `60d149c`)._
  1. **Wire schema (`9f840b6`)** — `blut_types::telemetry`: W3C-shaped
     `TraceId`/`SpanId` (all-zero rejected, since accepting it would fuse
     unrelated runs into one trace), `SpanContext::child()` as the only way to
     descend, `MetricName`/`Label` validated at CONSTRUCTION against the
     Prometheus grammar, and `TelemetryRecord`. In the keystone so producer and
     consumer cannot drift; the wasm32 gate still passes.
  2. **Engine emission (`3de67b9`)** — `StageEvent::Telemetry` carries the
     record FLATTENED, so ONE status.jsonl line is readable both as a stage
     event and as a `TelemetryRecord`: one line, two readers, no second stream
     that could disagree. Telemetry rides the LOSSLESS lane (a dropped sample
     is a silent hole in a time series). `HostedEvent` gained
     `trace_id`/`parent_span_id`, omitted when absent so an untraced line is
     byte-identical for pre-0097 readers.
  3. **Sidecar (`60d149c`)** — `crates/blut-metrics` links `blut-types` ONLY,
     never the engine, so the charter holds by construction. Counters
     max-merge (no false counter resets), label sets sort into the series key
     (no silently forked metrics), foreign lines are skipped as expected
     traffic. OpenMetrics rendering handles the whole-body-rejection cases:
     `_total` family naming, the mandatory `+Inf` bucket, label escaping,
     Inf/NaN forms.
  Deviation recorded: promtool is not installed in this environment, so the
  ADR's format check was implemented as an in-test OpenMetrics grammar check
  rather than skipped — and proven non-vacuous (it rejects five distinct
  malformed bodies). A gate that degrades silently is worse than one that
  fails loudly.

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** passed
- **gate_cmd:** `bash scripts/scope_guard.sh && (cd crates/blut-metrics &&
  cargo test --test scrape_and_trace)` — run from the crate directory as with
  every other standalone-workspace sidecar
- **result:** scope guard PASS and the engine contains ZERO
  axum/hyper/reqwest references (no scrape or OTLP socket in `src`). The
  gate ingests a 2-host mesh fixture, serves the REAL router on a REAL socket,
  scrapes it over TCP, and asserts the broker/stage/mesh/cache/ε gauges and the
  stage-duration histogram are present with NON-DEFAULT values (a zeroed series
  would pass a presence check while telling an operator nothing); the
  host-hopping stage's spans share ONE `trace_id` with the remote span parented
  onto the dispatching span, and that chain survives into the OTLP payload.
  Sidecar 11 tests / 4 binaries, clippy 0, fmt clean; engine 18/18 suites (909
  tests), clippy 0. CI + release gates added.
- **git_sha:** engine `60d149c` (schema `9f840b6`, emission `3de67b9`, sidecar
  `60d149c`)

## Related Decisions

[ADR 0083](0083-blut-interface-stack.md) (interface stack / sidecar boundary), [ADR 0093](0093-blut-web-control-plane.md) (web dashboard consuming
the same gauges), [ADR 0067](0067-remaining-work-toward-vision.md) (status.jsonl observability substrate this reads),
[ADR 0034](0034-blut-scope-charter.md) (no-HTTP charter), [ADR 0046](0046-blut-resource-broker.md)/0047 (broker metrics source), [ADR 0061](0061-blut-p2p-compute.md)
(mesh dispatch / ε budget).

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: the engine ever opens a scrape/OTLP socket (charter breach caught by
the scope guard); mesh traces fragment because a `HostedEvent` construction site
drops the parent span id; the sidecar's gauges drift from what the TUI reports
off the same artifacts; or hot-path gauge emission shows up as a measurable
stage-latency regression that can't be tuned away via the sampling knob.
