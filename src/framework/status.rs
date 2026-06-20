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

use crate::framework::artifact::ContentHash;
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
/// can never drop a `StageBegin`/`StageEnd`/`StageFailed` from the
/// audit trail. The broadcast is for the live UI only.
pub const DEFAULT_BROADCAST_CAPACITY: usize = 4096;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageEvent {
    /// Stage entered its `run` body. `node_idx` is its position in
    /// the plan's topological order (0-indexed).
    StageBegin {
        node_idx: u32,
        stage_name: String,
        input_hash: ContentHash,
    },
    /// Stage successfully produced output.
    StageEnd {
        node_idx: u32,
        stage_name: String,
        output_hash: ContentHash,
        elapsed: Duration,
    },
    /// Stage was skipped because the cache hit on
    /// `(stage_name, input_hash, args_hash)`.
    StageSkipped {
        node_idx: u32,
        stage_name: String,
        cache_key: ContentHash,
    },
    /// Stage failed. `error` is the `Display` form of the
    /// `StageError`. When the error chain contains a [`StageFailure`],
    /// `failure` carries the structured summary (code, severity,
    /// context) for lineage storage and machine parsing.
    StageFailed {
        node_idx: u32,
        stage_name: String,
        error: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        failure: Option<FailureSummary>,
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
    /// reader sees the gap instead of a silently-short stream. Generated
    /// by the writer, never emitted via [`StatusHub::emit`].
    StepGap { dropped: u64 },
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
    /// (begin/end/skipped/failed/blocked). These ride the LOSSLESS
    /// channel; `StageStep` (and the writer-generated `StepGap`) are
    /// the lossy, high-volume class.
    pub fn is_lifecycle(&self) -> bool {
        !matches!(
            self,
            StageEvent::StageStep { .. } | StageEvent::StepGap { .. }
        )
    }
}

/// Fan-out hub for stage status. A single [`emit`](StatusHub::emit)
/// choke point stamps a process-wide sequence number and routes each
/// event:
///   * lifecycle events → BOTH the lossless mpsc (the writer, so the
///     audit trail never has gaps) AND the lossy broadcast (live UI);
///   * `StageStep` → the lossy broadcast only (batched by the writer).
///
/// The lifecycle mpsc preserves emit order (FIFO), so status.jsonl reads
/// lifecycle events in the order they were emitted across concurrent
/// stages.
pub struct StatusHub {
    broadcast: broadcast::Sender<StageEvent>,
    lifecycle_tx: mpsc::UnboundedSender<StageEvent>,
}

impl StatusHub {
    /// Build a hub + the lifecycle receiver the writer drains. The
    /// caller hands the receiver to [`spawn_status_writer`].
    pub fn new() -> (Arc<StatusHub>, mpsc::UnboundedReceiver<StageEvent>) {
        let (broadcast, _rx) = broadcast::channel(DEFAULT_BROADCAST_CAPACITY);
        let (lifecycle_tx, lifecycle_rx) = mpsc::unbounded_channel();
        let hub = Arc::new(StatusHub {
            broadcast,
            lifecycle_tx,
        });
        (hub, lifecycle_rx)
    }

    /// Emit one event. Lifecycle events go to the lossless writer
    /// channel as well as the broadcast; steps go to the broadcast only.
    pub fn emit(&self, ev: StageEvent) {
        if ev.is_lifecycle() {
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
///     records a `StepGap { dropped: n }` line so the gap is visible.
///
/// The task ends once the lifecycle channel closes (the last
/// `StatusHub` dropped) — it then drains any remaining broadcast Steps
/// and returns.
pub fn spawn_status_writer(
    hub: &Arc<StatusHub>,
    mut lifecycle_rx: mpsc::UnboundedReceiver<StageEvent>,
    job_dir: &std::path::Path,
) -> std::io::Result<tokio::task::JoinHandle<()>> {
    use std::io::Write;
    std::fs::create_dir_all(job_dir)?;
    let path = job_dir.join("status.jsonl");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let mut brx = hub.subscribe();
    Ok(tokio::spawn(async move {
        let mut writer = std::io::BufWriter::with_capacity(64 * 1024, file);
        let reopen = |p: &std::path::Path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .map(|f| std::io::BufWriter::with_capacity(64 * 1024, f))
        };
        // Write one event line; returns false on an unrecoverable I/O
        // error (caller exits). `flush` forces the line to disk now.
        macro_rules! write_event {
            ($ev:expr, $flush:expr) => {{
                let mut ok = true;
                match serde_json::to_string(&$ev) {
                    Ok(line) => {
                        if writeln!(writer, "{line}").is_err() {
                            tracing::warn!("status writer: write failed, exiting");
                            ok = false;
                        }
                    }
                    Err(e) => tracing::error!("status writer: serialize event failed: {e}"),
                }
                if ok && $flush && writer.flush().is_err() {
                    tracing::warn!("status writer: flush failed, exiting");
                    ok = false;
                }
                ok
            }};
        }
        loop {
            let timeout = tokio::time::sleep(STEP_FLUSH_INTERVAL);
            tokio::pin!(timeout);
            tokio::select! {
                // Lossless lifecycle — immediate flush, seq-ordered.
                got = lifecycle_rx.recv() => match got {
                    Some(event) => {
                        if !write_event!(event, true) { return; }
                    }
                    None => {
                        // Last StatusHub dropped → run is finishing. Drain
                        // any remaining broadcast Steps, then exit.
                        let _ = writer.flush();
                        while let Ok(event) = brx.try_recv() {
                            if matches!(event, StageEvent::StageStep { .. }) {
                                let _ = write_event!(event, false);
                            }
                        }
                        let _ = writer.flush();
                        return;
                    }
                },
                // Lossy Step spam — batched.
                got = brx.recv() => match got {
                    Ok(event) => {
                        // Lifecycle events also arrive here (broadcast), but
                        // the lossless path already wrote them — skip to avoid
                        // duplicates; only Steps are writer-owned on this path.
                        if matches!(event, StageEvent::StageStep { .. })
                            && !write_event!(event, false)
                        {
                            return;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        // Broadcast closed but lifecycle may still be open;
                        // keep looping on lifecycle.
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("status writer: broadcast lagged by {n} steps");
                        let _ = write_event!(StageEvent::StepGap { dropped: n }, false);
                    }
                },
                _ = &mut timeout => {
                    let _ = writer.flush();
                    if crate::jobs::rotate_status_if_needed(&path) || !path.exists() {
                        match reopen(&path) {
                            Ok(w) => writer = w,
                            Err(e) => {
                                tracing::warn!("status writer: reopen after rotate failed: {e}");
                                return;
                            }
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
        };
        let s = serde_json::to_string(&e).unwrap();
        assert!(s.contains("\"kind\":\"stage_begin\""));
        let _back: StageEvent = serde_json::from_str(&s).unwrap();
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

    #[tokio::test]
    async fn make_broadcast_subscribes_round_trip() {
        let tx = make_broadcast();
        let mut rx = tx.subscribe();
        let h = ContentHash::of_bytes(b"x");
        tx.send(StageEvent::StageEnd {
            node_idx: 1,
            stage_name: "filter_dataset".into(),
            output_hash: h,
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
            other => panic!("wrong variant: {:?}", other),
        }
    }

    #[test]
    fn lifecycle_classification() {
        assert!(
            StageEvent::StageBegin {
                node_idx: 0,
                stage_name: "s".into(),
                input_hash: ContentHash::of_bytes(b""),
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
    }

    #[tokio::test]
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
            output_hash: ContentHash::of_bytes(b"out"),
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
    }

    // PathBuf retained for older call sites that build paths in tests.
    #[allow(dead_code)]
    fn _path_marker() -> PathBuf {
        PathBuf::new()
    }
}
