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

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::framework::artifact::ContentHash;
use crate::framework::resource::Resource;

/// Default broadcast channel capacity. 256 is generous for a single
/// pipeline (most plans have <50 events total) and still bounded so
/// a stuck consumer can't OOM the producer.
pub const DEFAULT_BROADCAST_CAPACITY: usize = 256;

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
    /// `StageError`; structured fields land in a follow-up if the
    /// CLI ever needs to format errors specially.
    StageFailed {
        node_idx: u32,
        stage_name: String,
        error: String,
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

/// Spawn a background task that subscribes to `tx` and appends each
/// received `StageEvent` as one JSON line to `<job_dir>/status.jsonl`.
///
/// Performance: writes are buffered with a 64 KiB BufWriter.
/// Lifecycle events (StageBegin/End/Failed/Skipped/Blocked) trigger
/// an immediate flush so a `kill -9` mid-run preserves the
/// structurally important audit trail. High-volume `StageStep`
/// events (per-training-step loss) are batched — flushed on the
/// next lifecycle event OR every `STEP_FLUSH_INTERVAL`, whichever
/// first. This drops syscall count by ~100× on a long training
/// without sacrificing crash recovery for the events that matter.
///
/// The task ends when the broadcast sender is dropped (RecvError::
/// Closed) or when an unrecoverable I/O error occurs on the file.
/// Lagged receivers are tolerated; the gap is logged.
pub fn spawn_status_writer(
    tx: &tokio::sync::broadcast::Sender<StageEvent>,
    job_dir: &std::path::Path,
) -> std::io::Result<tokio::task::JoinHandle<()>> {
    use std::io::Write;
    std::fs::create_dir_all(job_dir)?;
    let path = job_dir.join("status.jsonl");
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let mut rx = tx.subscribe();
    Ok(tokio::spawn(async move {
        let mut writer = std::io::BufWriter::with_capacity(64 * 1024, file);
        loop {
            let timeout =
                tokio::time::sleep(STEP_FLUSH_INTERVAL);
            tokio::pin!(timeout);
            tokio::select! {
                got = rx.recv() => match got {
                    Ok(event) => {
                        let immediate = !matches!(event, StageEvent::StageStep { .. });
                        if let Ok(line) = serde_json::to_string(&event) {
                            if writeln!(writer, "{line}").is_err() {
                                tracing::warn!("status writer: write failed, exiting");
                                return;
                            }
                        }
                        if immediate {
                            if writer.flush().is_err() {
                                tracing::warn!("status writer: flush failed, exiting");
                                return;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        let _ = writer.flush();
                        return;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("status writer: lagged by {n} events");
                    }
                },
                _ = &mut timeout => {
                    // Periodic flush of buffered StageStep events.
                    let _ = writer.flush();
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
            StageEvent::StageEnd { node_idx, stage_name, .. } => {
                assert_eq!(node_idx, 1);
                assert_eq!(stage_name, "filter_dataset");
            }
            other => panic!("wrong variant: {:?}", other),
        }
    }

    // PathBuf is used in commit-3 status writer; quiet the unused
    // import if the writer isn't here yet.
    #[allow(dead_code)]
    fn _path_marker() -> PathBuf {
        PathBuf::new()
    }
}
