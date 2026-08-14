// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Plan-execution status events.
//!
//! Every stage emits `StageEvent`s over a `tokio::sync::broadcast`
//! channel inside `StageContext`. Subscribers: the persisted
//! `status.jsonl` writer (lands commit 3), the live CLI renderer,
//! eventually a TUI dashboard / web UI.
//!
//! This commit ships the enum + a no-op channel constructor; the
//! persistent writer + render helpers land in commit 3 alongside
//! the executor.
//!
//! Wire format: every variant serializes with `kind` discriminator
//! at the front so the same parser handles live broadcast streams
//! and post-hoc `status.jsonl` reads.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};

use crate::framework::artifact::{ArtifactContentId, ContentHash, InvocationKey};
use crate::framework::error_domain::FailureSummary;
use crate::framework::resource::Resource;

/// Default broadcast channel capacity. 4096 sized for chatty
/// trainers (per-step loss + per-grad-accum logs) so the writer
/// task can fall behind a few seconds without RecvError::Lagged
/// dropping events. Still bounded — a stuck consumer can't OOM
/// the producer indefinitely. Originally 256; bumped after
/// observing realistic per-step emission rates from trainer.py.
///
/// Note: lifecycle events (everything but `StageStep`) DO NOT ride this
/// lossy channel to the writer — they go through the [`StatusHub`]'s
/// separate lossless mpsc, so a writer that falls behind on Step spam
/// can never drop a `StageBegin`/`StageEnd`/`StageFailed`/`StagePruned` from the
/// audit trail. The broadcast is for the live UI only.
pub const DEFAULT_BROADCAST_CAPACITY: usize = 4096;

/// A09 is a deliberate writer-side clean break: current records name
/// `content_id` and `invocation_key` explicitly. Current readers retain aliases
/// for pre-A09 fields. Legacy invocation keys remain in the same typed domain;
/// legacy `StageEnd.output_hash` values do not and therefore deserialize into a
/// separate field instead of manufacturing a portable [`ArtifactContentId`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StageEvent {
    /// The complete execution-only async-I/O profile admitted for this node.
    /// Emitted on the lossless lifecycle lane before `StageBegin`. Stages with
    /// no declared candidates emit no record and keep legacy behavior.
    StageIoConfigured {
        node_idx: u32,
        stage_name: String,
        profile: crate::framework::async_io::TrainingIoProfile,
    },
    /// Stage entered its `run` body. `node_idx` is its position in
    /// the plan's topological order (0-indexed).
    StageBegin {
        node_idx: u32,
        stage_name: String,
        /// Legacy logical-input digest retained for invocation/cache diagnostics.
        input_hash: ContentHash,
        /// Portable identities of predecessor artifacts, in dependency order.
        /// Empty for graph roots and legacy status records.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        input_content_ids: Vec<ArtifactContentId>,
    },
    /// Stage successfully produced output.
    StageEnd {
        node_idx: u32,
        stage_name: String,
        /// Portable output identity. Absent only on a pre-A09 status record.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_id: Option<ArtifactContentId>,
        /// Pre-A09 logical output hash, preserved only for display/audit.
        #[serde(
            default,
            rename = "output_hash",
            skip_serializing_if = "Option::is_none"
        )]
        legacy_output_hash: Option<ContentHash>,
        elapsed: Duration,
    },
    /// Stage was skipped because the cache hit on
    /// `(stage_name, input_hash, args_hash)`.
    StageSkipped {
        node_idx: u32,
        stage_name: String,
        /// Serialized as the legacy `cache_key` field during the 7.8 bridge.
        #[serde(rename = "invocation_key", alias = "cache_key")]
        invocation_key: InvocationKey,
        /// Absent only when replaying a pre-A09 status record.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_id: Option<ArtifactContentId>,
    },
    /// Stage failed. `error` is the `Display` form of the
    /// `StageError`. When the error chain contains a [`crate::framework::error_domain::StageFailure`],
    /// `failure` carries the structured summary (code, severity,
    /// context) for lineage storage and machine parsing.
    StageFailed {
        node_idx: u32,
        stage_name: String,
        error: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        failure: Option<FailureSummary>,
    },
    /// The scheduler proved this stage is unreachable for the selected
    /// conditional path. Unlike `StageFailed`, this is a successful control
    /// decision and the stage never entered its run body.
    StagePruned {
        node_idx: u32,
        stage_name: String,
        reason: String,
    },
    /// Stage is blocked on a resource semaphore. Useful for the TUI
    /// to show "waiting on GPU" instead of "running".
    StageBlocked {
        node_idx: u32,
        stage_name: String,
        resource: Resource,
    },
    /// Step-level progress from inside a stage (e.g. trainer.py
    /// emitting per-step loss). Pre-existing `StatusUpdate` from
    /// `protocol.rs` rides through here at framework level so the
    /// status.jsonl format is uniform.
    StageStep {
        node_idx: u32,
        stage_name: String,
        update: serde_json::Value,
    },
    /// The lossy broadcast lagged: `dropped` `StageStep` events were
    /// lost before the writer drained them. Recorded so a status.jsonl
    /// reader sees the gap instead of a silently-short stream. Normally
    /// generated by the writer; private speculative streams may replay an
    /// explicit marker through [`StatusHub::emit`] when their own receiver
    /// lagged before selection.
    StepGap { dropped: u64 },
    /// An ADR 0097 telemetry record (counter / gauge / duration / span). The
    /// record is FLATTENED, so the emitted line carries both `kind:
    /// "telemetry"` for stage-event readers and the record's own `telemetry`
    /// tag for `blut_types::telemetry::TelemetryRecord::from_line` — one line,
    /// two readers, no second stream to keep in sync.
    Telemetry {
        #[serde(flatten)]
        record: blut_types::telemetry::TelemetryRecord,
    },
    /// A stage attempt failed with a retryable error and will be retried
    /// (D1). `attempt` is the one that just failed (1-based).
    StageRetrying {
        node_idx: u32,
        stage_name: String,
        attempt: u32,
        max_attempts: u32,
        error: String,
        backoff_ms: u64,
    },
}

impl StageEvent {
    /// Lifecycle events — the structurally important audit trail
    /// (begin/end/skipped/failed/pruned/blocked). These ride the LOSSLESS
    /// channel; `StageStep` (and the writer-generated `StepGap`) are
    /// the lossy, high-volume class.
    pub fn is_lifecycle(&self) -> bool {
        !matches!(
            self,
            StageEvent::StageStep { .. } | StageEvent::StepGap { .. }
        )
    }

    /// The telemetry record this event carries, if any (ADR 0097).
    pub fn telemetry(&self) -> Option<&blut_types::telemetry::TelemetryRecord> {
        match self {
            StageEvent::Telemetry { record } => Some(record),
            _ => None,
        }
    }
}

/// A [`StageEvent`] tagged with the host that produced it (D5, cross-host
/// observability). This is the status.jsonl LINE format: the event's fields
/// are flattened in, plus an optional `host` (a peer's short-hex id). `host`
/// is omitted when `None` (local events), so old single-host readers are
/// unaffected — the field is purely additive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HostedEvent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    /// ADR 0097 trace context. Threading these across a mesh hop is what keeps
    /// a host-hopping stage ONE trace instead of N fragments; both are omitted
    /// when absent, so a pre-0097 reader sees a byte-identical line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<blut_types::telemetry::TraceId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<blut_types::telemetry::SpanId>,
    #[serde(flatten)]
    pub event: StageEvent,
}

impl HostedEvent {
    /// Tag `event` with `host` (clone of the local node id, or a forwarded
    /// peer's id). `None` host ⇒ a local, untagged line.
    pub fn wrap(host: &Option<String>, event: StageEvent) -> Self {
        Self {
            host: host.clone(),
            trace_id: None,
            parent_span_id: None,
            event,
        }
    }

    /// Tag with host AND the ADR 0097 trace context. `ctx` supplies the trace
    /// this line belongs to and the span that is its parent, so a consumer can
    /// stitch events across hosts into one waterfall.
    pub fn wrap_traced(
        host: &Option<String>,
        event: StageEvent,
        ctx: Option<&blut_types::telemetry::SpanContext>,
    ) -> Self {
        Self {
            host: host.clone(),
            trace_id: ctx.map(|c| c.trace_id.clone()),
            parent_span_id: ctx.map(|c| c.span_id.clone()),
            event,
        }
    }
}

/// Fan-out hub for stage status. A single [`emit`](StatusHub::emit)
/// choke point stamps a process-wide sequence number and routes each
/// event:
///   * lifecycle events → BOTH the lossless mpsc (the writer, so the
///     audit trail never has gaps) AND the lossy broadcast (live UI);
///   * `StageStep` → the lossy broadcast only (batched by the writer);
///   * an explicit `StepGap` → the lossless audit path plus the lossy live
///     broadcast. The writer ignores the broadcast copy so persistence is
///     exact and non-duplicated.
///
/// The lifecycle mpsc preserves emit order (FIFO), so status.jsonl reads
/// lifecycle events in the order they were emitted across concurrent
/// stages.
pub struct StatusHub {
    broadcast: broadcast::Sender<StageEvent>,
    lifecycle_tx: mpsc::UnboundedSender<StageEvent>,
    /// This node's short-hex id, stamped onto locally-emitted lines (D5).
    /// `None` (default) ⇒ single-host run, `host` omitted from the JSON.
    host: Option<String>,
    /// Already-host-tagged events forwarded FROM other nodes (D5). Drained by
    /// the same writer so one status.jsonl on the initiator tells the whole
    /// multi-host story. The receiver is taken once by [`spawn_status_writer`].
    /// BOUNDED (a display audit trail) so a flooding/malicious worker can't OOM
    /// the initiator — on a full channel the event is dropped + logged.
    remote_tx: mpsc::Sender<HostedEvent>,
    remote_rx: std::sync::Mutex<Option<mpsc::Receiver<HostedEvent>>>,
}

impl StatusHub {
    /// Build a hub + the lifecycle receiver the writer drains. The
    /// caller hands the receiver to [`spawn_status_writer`].
    pub fn new() -> (Arc<StatusHub>, mpsc::UnboundedReceiver<StageEvent>) {
        let (broadcast, _rx) = broadcast::channel(DEFAULT_BROADCAST_CAPACITY);
        let (lifecycle_tx, lifecycle_rx) = mpsc::unbounded_channel();
        let (remote_tx, remote_rx) = mpsc::channel(DEFAULT_BROADCAST_CAPACITY);
        let hub = Arc::new(StatusHub {
            broadcast,
            lifecycle_tx,
            host: None,
            remote_tx,
            remote_rx: std::sync::Mutex::new(Some(remote_rx)),
        });
        (hub, lifecycle_rx)
    }

    /// Set this node's short id, stamped onto locally-emitted status lines
    /// (D5). Call IMMEDIATELY after [`new`](Self::new), before the `Arc` is
    /// cloned/shared or the writer is spawned — it consumes + rebuilds the
    /// `Arc` and PANICS if any other strong reference exists.
    pub fn with_host(self: Arc<Self>, host: impl Into<String>) -> Arc<Self> {
        let mut inner = Arc::try_unwrap(self)
            .unwrap_or_else(|_| panic!("with_host must be called before the hub is shared"));
        inner.host = Some(host.into());
        Arc::new(inner)
    }

    /// This node's configured host id, if any.
    pub fn host(&self) -> Option<&str> {
        self.host.as_deref()
    }

    /// Ingest a lifecycle event FORWARDED from another node, tagged with that
    /// node's `host` (D5). Written to this node's status.jsonl by the writer.
    /// Non-lifecycle events are ignored (only the audit trail crosses hosts).
    pub fn ingest_remote(&self, host: impl Into<String>, event: StageEvent) {
        if !event.is_lifecycle() {
            // Only the lossless audit trail crosses hosts; Steps stay local.
            tracing::trace!("status: dropping forwarded non-lifecycle event");
            return;
        }
        // try_send (not await): sync caller + bounded channel. A full channel
        // means the initiator's writer is far behind (or a flood) — drop the
        // event rather than block/OOM; it's a display trail.
        if self
            .remote_tx
            .try_send(HostedEvent {
                host: Some(host.into()),
                trace_id: None,
                parent_span_id: None,
                event,
            })
            .is_err()
        {
            tracing::warn!("status: remote event channel full/closed; dropped a forwarded event");
        }
    }

    /// Emit one ADR 0097 telemetry record. It rides the LOSSLESS lane: a
    /// dropped metric is a silent hole in a time series, which is worse than
    /// back-pressure on what is otherwise a display channel.
    pub fn emit_telemetry(&self, record: blut_types::telemetry::TelemetryRecord) {
        self.emit(StageEvent::Telemetry { record });
    }

    /// Emit one event. Lifecycle events go to the lossless writer
    /// channel as well as the broadcast; steps go to the broadcast only.
    pub fn emit(&self, ev: StageEvent) {
        if ev.is_lifecycle() || matches!(ev, StageEvent::StepGap { .. }) {
            let _ = self.lifecycle_tx.send(ev.clone());
        }
        let _ = self.broadcast.send(ev);
    }

    /// A broadcast sender clone — for `StageContext`, which only emits
    /// `StageStep` (lossy is fine).
    pub fn broadcast_sender(&self) -> broadcast::Sender<StageEvent> {
        self.broadcast.clone()
    }

    /// Subscribe a live receiver (TUI / CLI renderer).
    pub fn subscribe(&self) -> broadcast::Receiver<StageEvent> {
        self.broadcast.subscribe()
    }
}

/// Make a fresh broadcast channel sized at
/// `DEFAULT_BROADCAST_CAPACITY`. Returned receiver is dropped — the
/// caller is expected to subscribe their own consumers via
/// `Sender::subscribe`. The returned sender is what `StageContext`
/// holds.
pub fn make_broadcast() -> tokio::sync::broadcast::Sender<StageEvent> {
    let (tx, _rx) = tokio::sync::broadcast::channel(DEFAULT_BROADCAST_CAPACITY);
    tx
}

/// How long to batch buffered StageStep events before forcing a
/// flush. Lifecycle events (Begin/End/Failed/Skipped/Blocked) flush
/// immediately regardless.
const STEP_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

/// Spawn the background task that appends every `StageEvent` as one
/// JSON line to `<job_dir>/status.jsonl`. It drains TWO sources:
///
///   * `lifecycle_rx` (LOSSLESS mpsc): begin/end/skipped/failed/blocked
///     /retrying — written and flushed IMMEDIATELY, in `seq` order, so
///     a `kill -9` mid-run preserves the structurally important audit
///     trail and it can never be dropped under Step backpressure.
///   * the hub's broadcast (LOSSY): `StageStep` spam, batched and
///     flushed every `STEP_FLUSH_INTERVAL`. On `Lagged(n)` the writer
///     records a `StepGap { dropped: n }` line so the gap is visible. An
///     explicitly emitted `StepGap` persists through the lossless input and is
///     ignored on this writer-owned broadcast copy.
///
/// The task ends once the lifecycle channel closes (the last
/// `StatusHub` dropped) — it then drains any remaining broadcast Steps
/// and returns.
/// Back-compatible status writer API. Persistence failures are logged, matching
/// the historical `JoinHandle<()>` contract. Executors that must fail a
/// successful plan when authoritative lifecycle persistence fails use
/// [`spawn_status_writer_checked`] instead.
pub fn spawn_status_writer(
    hub: &Arc<StatusHub>,
    lifecycle_rx: mpsc::UnboundedReceiver<StageEvent>,
    job_dir: &std::path::Path,
) -> std::io::Result<tokio::task::JoinHandle<()>> {
    let checked = spawn_status_writer_checked(hub, lifecycle_rx, job_dir)?;
    // The historical handle owned the writer task directly: aborting it stopped
    // persistence. Keep that behavior even though the compatibility wrapper now
    // translates the checked writer's result into a warning.
    let mut checked_abort = AbortTaskOnDrop::new(checked.abort_handle());
    Ok(tokio::spawn(async move {
        let result = checked.await;
        checked_abort.disarm();
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::warn!("status lifecycle persistence failed: {error}"),
            Err(error) => tracing::warn!("status lifecycle writer task failed: {error}"),
        }
    }))
}

struct AbortTaskOnDrop(Option<tokio::task::AbortHandle>);

impl AbortTaskOnDrop {
    fn new(handle: tokio::task::AbortHandle) -> Self {
        Self(Some(handle))
    }

    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for AbortTaskOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

/// Checked status writer used by the executor's authoritative lifecycle path.
/// The nested `io::Result` carries serialization/write/flush/reopen failures.
pub fn spawn_status_writer_checked(
    hub: &Arc<StatusHub>,
    mut lifecycle_rx: mpsc::UnboundedReceiver<StageEvent>,
    job_dir: &std::path::Path,
) -> std::io::Result<tokio::task::JoinHandle<std::io::Result<()>>> {
    use std::io::Write;
    std::fs::create_dir_all(job_dir)?;
    let path = job_dir.join("status.jsonl");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let mut brx = hub.subscribe();
    // D5: the local host tag (stamped on local lines) + the remote channel of
    // already-tagged events forwarded from other nodes (drained by this same
    // writer so one status.jsonl carries the whole multi-host story).
    let local_host = hub.host.clone();
    let mut remote_rx = hub.remote_rx.lock().ok().and_then(|mut g| g.take());
    Ok(tokio::spawn(async move {
        let mut writer = std::io::BufWriter::with_capacity(64 * 1024, file);
        let reopen = |p: &std::path::Path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .map(|f| std::io::BufWriter::with_capacity(64 * 1024, f))
        };
        // `flush = true` is the lifecycle durability boundary. Propagate every
        // serialization/write/flush failure through the JoinHandle so a plan
        // cannot report success after losing its authoritative profile record.
        macro_rules! write_event {
            ($ev:expr, $flush:expr) => {{
                let line = serde_json::to_string(&$ev)
                    .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
                writeln!(writer, "{line}")?;
                if $flush {
                    writer.flush()?;
                }
            }};
        }
        loop {
            let timeout = tokio::time::sleep(STEP_FLUSH_INTERVAL);
            tokio::pin!(timeout);
            tokio::select! {
                // Lossless lifecycle — immediate flush, seq-ordered. Local
                // events are tagged with this node's host (D5).
                got = lifecycle_rx.recv() => match got {
                    Some(event) => {
                        write_event!(HostedEvent::wrap(&local_host, event), true);
                    }
                    None => {
                        // Last StatusHub dropped → run is finishing. Drain any
                        // remaining REMOTE lifecycle events (D5) + broadcast
                        // Steps before exiting, so a worker event that arrived
                        // just before shutdown isn't lost.
                        writer.flush()?;
                        if let Some(rx) = remote_rx.as_mut() {
                            while let Ok(hosted) = rx.try_recv() {
                                write_event!(hosted, false);
                            }
                        }
                        loop {
                            match brx.try_recv() {
                                Ok(event) => {
                                    if matches!(event, StageEvent::StageStep { .. }) {
                                        write_event!(HostedEvent::wrap(&local_host, event), false);
                                    }
                                }
                                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                                    tracing::warn!(
                                        "status writer: broadcast lagged by {n} steps during shutdown"
                                    );
                                    write_event!(
                                        HostedEvent::wrap(
                                            &local_host,
                                            StageEvent::StepGap { dropped: n },
                                        ),
                                        false
                                    );
                                }
                                Err(
                                    broadcast::error::TryRecvError::Empty
                                    | broadcast::error::TryRecvError::Closed,
                                ) => break,
                            }
                        }
                        writer.flush()?;
                        return Ok(());
                    }
                },
                // Lossless REMOTE lifecycle (D5): events forwarded from other
                // nodes, already host-tagged. A never-future when no remote
                // channel is attached so the select arm is inert.
                remote = async {
                    match &mut remote_rx {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending().await,
                    }
                } => if let Some(hosted) = remote {
                    write_event!(hosted, true);
                },
                // Lossy Step spam — batched.
                got = brx.recv() => match got {
                    Ok(event) => {
                        // Lifecycle events also arrive here (broadcast), but
                        // the lossless path already wrote them — skip to avoid
                        // duplicates. Explicit StepGap markers already used the
                        // lossless path; only Steps are writer-owned here.
                        if matches!(event, StageEvent::StageStep { .. }) {
                            write_event!(HostedEvent::wrap(&local_host, event), false);
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        // Broadcast closed but lifecycle may still be open;
                        // keep looping on lifecycle.
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("status writer: broadcast lagged by {n} steps");
                        write_event!(
                            HostedEvent::wrap(&local_host, StageEvent::StepGap { dropped: n }),
                            false
                        );
                    }
                },
                _ = &mut timeout => {
                    writer.flush()?;
                    if crate::jobs::rotate_status_if_needed(&path) || !path.exists() {
                        match reopen(&path) {
                            Ok(w) => writer = w,
                            Err(error) => return Err(error),
                        }
                    }
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn stage_event_serialize_round_trip() {
        let e = StageEvent::StageBegin {
            node_idx: 0,
            stage_name: "materialize_conversations".into(),
            input_hash: ContentHash::of_bytes(b""),
            input_content_ids: Vec::new(),
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"kind\":\"stage_begin\""));
        let _back: StageEvent = serde_json::from_str(&s).unwrap();
    }

    #[test]
    fn stage_end_uses_explicit_content_id_and_preserves_legacy_hash_as_unknown() {
        let content = ArtifactContentId::from_digest(ContentHash::of_bytes(b"content"));
        let event = StageEvent::StageEnd {
            node_idx: 1,
            stage_name: "producer".into(),
            content_id: Some(content),
            legacy_output_hash: None,
            elapsed: Duration::from_millis(12),
        };

        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["content_id"], content.to_hex());
        assert!(value.get("output_hash").is_none());
        let round_trip: StageEvent = serde_json::from_value(value).unwrap();
        assert!(matches!(
            round_trip,
            StageEvent::StageEnd { content_id: Some(content_id), legacy_output_hash: None, .. }
                if content_id == content
        ));

        let legacy: StageEvent = serde_json::from_value(serde_json::json!({
            "kind": "stage_end",
            "node_idx": 1,
            "stage_name": "producer",
            "output_hash": ContentHash::of_bytes(b"legacy logical").to_hex(),
            "elapsed": {"secs": 0, "nanos": 0}
        }))
        .unwrap();
        assert!(matches!(
            legacy,
            StageEvent::StageEnd {
                content_id: None,
                legacy_output_hash: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn skipped_event_serializes_invocation_and_content_identity() {
        let invocation = InvocationKey::from_digest(ContentHash::of_bytes(b"invocation"));
        let content = ArtifactContentId::from_digest(ContentHash::of_bytes(b"content"));
        let event = StageEvent::StageSkipped {
            node_idx: 3,
            stage_name: "cached".into(),
            invocation_key: invocation,
            content_id: Some(content),
        };

        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["invocation_key"], invocation.to_hex());
        assert_eq!(value["content_id"], content.to_hex());
        let round_trip: StageEvent = serde_json::from_value(value).unwrap();
        assert!(matches!(
            round_trip,
            StageEvent::StageSkipped {
                invocation_key,
                content_id: Some(content_id),
                ..
            } if invocation_key == invocation && content_id == content
        ));

        let legacy = serde_json::json!({
            "kind": "stage_skipped",
            "node_idx": 3,
            "stage_name": "cached",
            "cache_key": invocation.to_hex(),
        });
        let legacy: StageEvent = serde_json::from_value(legacy).unwrap();
        assert!(matches!(
            legacy,
            StageEvent::StageSkipped {
                invocation_key,
                content_id: None,
                ..
            } if invocation_key == invocation
        ));
    }

    #[test]
    fn hosted_event_flattens_host_and_omits_when_none() {
        let event = StageEvent::StageEnd {
            node_idx: 2,
            stage_name: "train".into(),
            content_id: Some(ArtifactContentId::from_digest(ContentHash::of_bytes(
                b"out",
            ))),
            legacy_output_hash: None,
            elapsed: Duration::from_secs(1),
        };
        // No host → the field is omitted (old single-host readers unaffected).
        let local = HostedEvent::wrap(&None, event.clone());
        let s = serde_json::to_string(&local).unwrap();
        assert!(
            s.contains("\"kind\":\"stage_end\""),
            "event fields flattened in"
        );
        assert!(!s.contains("\"host\""), "host omitted when None: {s}");

        // With a host → inline `host` alongside the flattened event.
        let hosted = HostedEvent::wrap(&Some("ab12cd".into()), event);
        let s = serde_json::to_string(&hosted).unwrap();
        assert!(s.contains("\"host\":\"ab12cd\""), "host tagged inline: {s}");
        assert!(s.contains("\"node_idx\":2"), "event still flattened");
        // Round-trips back to the tagged wrapper.
        let back: HostedEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back.host.as_deref(), Some("ab12cd"));
        assert!(matches!(
            back.event,
            StageEvent::StageEnd { node_idx: 2, .. }
        ));
    }

    #[tokio::test]
    async fn writer_aggregates_local_and_remote_host_tagged_events() {
        let (hub, lifecycle_rx) = StatusHub::new();
        let hub = hub.with_host("initiatorX");
        let td = tempfile::tempdir().unwrap();
        let handle = spawn_status_writer(&hub, lifecycle_rx, td.path()).unwrap();

        // A local lifecycle event (tagged with the initiator's host).
        hub.emit(StageEvent::StageBegin {
            node_idx: 0,
            stage_name: "local".into(),
            input_hash: ContentHash::of_bytes(b"i"),
            input_content_ids: Vec::new(),
        });
        // A lifecycle event forwarded FROM a worker, tagged with the worker id.
        hub.ingest_remote(
            "workerC",
            StageEvent::StageEnd {
                node_idx: 5,
                stage_name: "remote".into(),
                content_id: Some(ArtifactContentId::from_digest(ContentHash::of_bytes(b"o"))),
                legacy_output_hash: None,
                elapsed: Duration::from_millis(3),
            },
        );

        // Drop the hub so the writer drains + exits, then read status.jsonl.
        drop(hub);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), handle).await;
        let body = std::fs::read_to_string(td.path().join("status.jsonl")).unwrap();

        assert!(
            body.contains("\"host\":\"initiatorX\"") && body.contains("\"stage_name\":\"local\""),
            "local event host-tagged: {body}"
        );
        assert!(
            body.contains("\"host\":\"workerC\"") && body.contains("\"stage_name\":\"remote\""),
            "worker event forwarded + host-tagged into the initiator's status.jsonl: {body}"
        );
    }

    #[tokio::test]
    async fn aborting_legacy_writer_handle_stops_the_backing_writer() {
        let (hub, lifecycle_rx) = StatusHub::new();
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("status.jsonl");
        let writer = spawn_status_writer(&hub, lifecycle_rx, td.path()).unwrap();

        hub.emit(StageEvent::StageBegin {
            node_idx: 0,
            stage_name: "before-abort".into(),
            input_hash: ContentHash::of_bytes(b"in"),
            input_content_ids: Vec::new(),
        });
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if std::fs::read_to_string(&path).is_ok_and(|body| body.contains("before-abort")) {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("writer persisted the pre-abort event");

        writer.abort();
        assert!(writer.await.unwrap_err().is_cancelled());
        hub.emit(StageEvent::StageEnd {
            node_idx: 0,
            stage_name: "after-abort".into(),
            content_id: Some(ArtifactContentId::from_digest(ContentHash::of_bytes(
                b"out",
            ))),
            legacy_output_hash: None,
            elapsed: Duration::from_millis(1),
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        let body = std::fs::read_to_string(path).unwrap();
        assert!(body.contains("before-abort"));
        assert!(!body.contains("after-abort"));
    }

    #[test]
    fn stage_event_blocked_carries_resource() {
        let e = StageEvent::StageBlocked {
            node_idx: 7,
            stage_name: "sft_train".into(),
            resource: Resource::Gpu,
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"resource\":\"gpu\""));
    }

    /// Forward-compat regression pin at the exact level the finding
    /// describes: a raw `status.jsonl` line for `StageFailed` carrying a
    /// `failure.origin`/`failure.severity` token this binary version
    /// doesn't recognize (e.g. written by a future binary with a new
    /// `FaultOrigin` variant) must still deserialize as a `StageEvent` --
    /// not be silently dropped by callers like `lineage.rs`'s
    /// `let Ok(ev) = serde_json::from_str::<StageEvent>(&line) else {
    /// continue }` pattern, which previously dropped the WHOLE event (not
    /// just the one unrecognized field) on any unrecognized token.
    #[test]
    fn stage_failed_survives_unrecognized_origin_and_severity_tokens() {
        let line = r#"{"kind":"stage_failed","node_idx":3,"stage_name":"train_joint","error":"stage failed: e2e recon mismatch","failure":{"code":"E_FUTURE","domain":"lamquant","stage":"train_joint","severity":"apocalyptic","origin":"quantum","course":"train","recipe":"train_joint","ingredient":"trainer","context":[["ram_gib","64"]],"message":"e2e recon mismatch"}}"#;
        let ev: StageEvent = serde_json::from_str(line)
            .expect("unrecognized origin/severity tokens must not fail the whole StageEvent");
        match ev {
            StageEvent::StageFailed {
                node_idx,
                stage_name,
                error,
                failure,
            } => {
                assert_eq!(node_idx, 3);
                assert_eq!(stage_name, "train_joint");
                assert!(error.contains("e2e recon mismatch"));
                let f = failure.expect("the structured FailureSummary must still attach");
                assert_eq!(
                    f.origin,
                    crate::framework::error_domain::FaultOrigin::Unknown
                );
                assert_eq!(
                    f.severity,
                    crate::framework::error_domain::Severity::Unknown
                );
                // The rest of the record survives too, not just the two
                // fallback fields.
                assert_eq!(f.code, "E_FUTURE");
                assert_eq!(f.domain, "lamquant");
                assert_eq!(f.stage.as_deref(), Some("train_joint"));
                assert_eq!(f.course.as_deref(), Some("train"));
                assert_eq!(f.recipe.as_deref(), Some("train_joint"));
                assert_eq!(f.ingredient.as_deref(), Some("trainer"));
                assert_eq!(f.message, "e2e recon mismatch");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[tokio::test]
    async fn make_broadcast_subscribes_round_trip() {
        let tx = make_broadcast();
        let mut rx = tx.subscribe();
        let h = ContentHash::of_bytes(b"x");
        tx.send(StageEvent::StageEnd {
            node_idx: 1,
            stage_name: "filter_dataset".into(),
            content_id: Some(ArtifactContentId::from_digest(h)),
            legacy_output_hash: None,
            elapsed: Duration::from_millis(42),
        })
        .unwrap();
        let got = rx.recv().await.unwrap();
        match got {
            StageEvent::StageEnd {
                node_idx,
                stage_name,
                ..
            } => {
                assert_eq!(node_idx, 1);
                assert_eq!(stage_name, "filter_dataset");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn lifecycle_classification() {
        assert!(
            StageEvent::StageBegin {
                node_idx: 0,
                stage_name: "s".into(),
                input_hash: ContentHash::of_bytes(b""),
                input_content_ids: Vec::new(),
            }
            .is_lifecycle()
        );
        assert!(
            !StageEvent::StageStep {
                node_idx: 0,
                stage_name: "s".into(),
                update: serde_json::json!({}),
            }
            .is_lifecycle()
        );
        assert!(!StageEvent::StepGap { dropped: 3 }.is_lifecycle());
        assert!(
            StageEvent::StagePruned {
                node_idx: 1,
                stage_name: "guarded".into(),
                reason: "condition false".into(),
            }
            .is_lifecycle()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lifecycle_events_are_lossless_under_step_flood() {
        // The lossless guarantee: lifecycle events (begin/end) reach
        // status.jsonl even buried in a flood of Step spam far exceeding
        // the broadcast capacity. Steps may be dropped (lossy, fine);
        // lifecycle never is.
        let td = tempfile::tempdir().unwrap();
        let (hub, lifecycle_rx) = StatusHub::new();
        let writer = spawn_status_writer(&hub, lifecycle_rx, td.path()).unwrap();

        hub.emit(StageEvent::StageBegin {
            node_idx: 0,
            stage_name: "flooded".into(),
            input_hash: ContentHash::of_bytes(b"in"),
            input_content_ids: Vec::new(),
        });
        // Flood far more Steps than the broadcast can hold (4096).
        for i in 0..20_000u32 {
            hub.emit(StageEvent::StageStep {
                node_idx: 0,
                stage_name: "flooded".into(),
                update: serde_json::json!({ "step": i }),
            });
        }
        hub.emit(StageEvent::StageEnd {
            node_idx: 0,
            stage_name: "flooded".into(),
            content_id: Some(ArtifactContentId::from_digest(ContentHash::of_bytes(
                b"out",
            ))),
            legacy_output_hash: None,
            elapsed: Duration::from_millis(1),
        });

        // Drop the hub → lifecycle channel closes → writer drains + exits.
        drop(hub);
        let _ = writer.await;

        let body = std::fs::read_to_string(td.path().join("status.jsonl")).unwrap();
        assert_eq!(
            body.matches("\"kind\":\"stage_begin\"").count(),
            1,
            "lifecycle StageBegin must survive the flood"
        );
        assert_eq!(
            body.matches("\"kind\":\"stage_end\"").count(),
            1,
            "lifecycle StageEnd must survive the flood"
        );
        assert!(
            body.contains("\"kind\":\"step_gap\""),
            "shutdown drain must record broadcast lag instead of silently truncating steps"
        );
    }

    // PathBuf retained for older call sites that build paths in tests.
    #[allow(dead_code)]
    fn _path_marker() -> PathBuf {
        PathBuf::new()
    }
}

#[cfg(test)]
mod telemetry_tests {
    use super::*;
    use blut_types::telemetry::{Label, MetricName, SpanContext, SpanId, TelemetryRecord, TraceId};

    fn ctx() -> SpanContext {
        SpanContext::root(
            TraceId::parse("4bf92f3577b34da6a3ce929d0e0e4736").unwrap(),
            SpanId::parse("00f067aa0ba902b7").unwrap(),
        )
    }

    /// The load-bearing claim of the design: ONE status.jsonl line is readable
    /// both as a stage event (engine/TUI) and as a telemetry record (the
    /// blut-metrics sidecar). If this breaks, the two would need separate
    /// streams that could silently disagree.
    #[test]
    fn one_line_parses_as_both_a_stage_event_and_a_telemetry_record() {
        let record = TelemetryRecord::Gauge {
            name: MetricName::parse("blut_privacy_epsilon").unwrap(),
            value: 0.25,
            labels: vec![Label::new("tenant", "shared").unwrap()],
        };
        let line = serde_json::to_string(&HostedEvent::wrap(
            &Some("ab12cd".into()),
            StageEvent::Telemetry {
                record: record.clone(),
            },
        ))
        .unwrap();

        // Reader A — the engine's own line format.
        let back: HostedEvent = serde_json::from_str(&line).unwrap();
        assert_eq!(back.host.as_deref(), Some("ab12cd"));
        assert_eq!(back.event.telemetry(), Some(&record));
        // Reader B — the sidecar, which knows nothing of StageEvent.
        assert_eq!(TelemetryRecord::from_line(&line), Some(record));
    }

    /// Telemetry must ride the LOSSLESS lane; a dropped sample is a silent gap
    /// in a time series.
    #[test]
    fn telemetry_is_lifecycle_so_it_is_never_dropped() {
        let ev = StageEvent::Telemetry {
            record: TelemetryRecord::Counter {
                name: MetricName::parse("blut_cache_hits_total").unwrap(),
                value: 1,
                labels: vec![],
            },
        };
        assert!(ev.is_lifecycle(), "telemetry must not ride the lossy lane");
    }

    /// The trace fields are ADDITIVE: an untraced line is byte-identical to
    /// what a pre-0097 engine wrote, so old readers are unaffected.
    #[test]
    fn untraced_lines_stay_byte_identical_for_old_readers() {
        let ev = StageEvent::StepGap { dropped: 3 };
        let line = serde_json::to_string(&HostedEvent::wrap(&None, ev)).unwrap();
        assert!(
            !line.contains("trace_id"),
            "absent trace must not serialize"
        );
        assert!(!line.contains("parent_span_id"));
        assert!(!line.contains("host"), "absent host still omitted: {line}");
    }

    /// A hop carries the trace: the forwarded line names the same trace and
    /// parents onto the originating span, which is what stitches N per-host
    /// fragments into one waterfall.
    #[test]
    fn a_hop_keeps_one_trace_and_parents_onto_the_origin_span() {
        let origin = ctx();
        let hosted = HostedEvent::wrap_traced(
            &Some("peer99".into()),
            StageEvent::StepGap { dropped: 0 },
            Some(&origin),
        );
        let line = serde_json::to_string(&hosted).unwrap();
        let back: HostedEvent = serde_json::from_str(&line).unwrap();
        assert_eq!(back.trace_id.as_ref(), Some(&origin.trace_id));
        assert_eq!(
            back.parent_span_id.as_ref(),
            Some(&origin.span_id),
            "the remote span must parent onto the span that dispatched it"
        );
        // A child minted from that context stays in the same trace.
        let child = origin.child(SpanId::parse("1122334455667788").unwrap());
        assert_eq!(child.trace_id, origin.trace_id);
    }

    /// The hub's emit_telemetry reaches the lossless writer lane.
    #[tokio::test]
    async fn emit_telemetry_reaches_the_writer_lane() {
        let (hub, mut rx) = StatusHub::new();
        hub.emit_telemetry(TelemetryRecord::Duration {
            name: MetricName::parse("blut_stage_duration_seconds").unwrap(),
            seconds: 2.5,
            labels: vec![],
        });
        let got = rx.recv().await.expect("telemetry reached the writer");
        assert!(matches!(
            got.telemetry(),
            Some(TelemetryRecord::Duration { .. })
        ));
    }
}
