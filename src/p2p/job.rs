// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! P2P job handle — implements the [`RemoteJob`] trait for tasks
//! dispatched to peer GPUs.
//!
//! A `P2pJob` tracks the lifecycle of a single task: pending → running
//! → done/failed/cancelled. The coordinator creates one per dispatched
//! task and hands it to the executor for progress tracking.

use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::config::launcher::{JobState, RemoteJob};
use crate::error::TrainError;
use crate::p2p::peer::PeerId;
use crate::p2p::task::TaskResult;

/// Internal state of a P2P job.
#[derive(Debug)]
pub enum P2pJobState {
    /// Task submitted, waiting for a peer to pick it up.
    Pending,
    /// Task is being executed by a peer.
    Running { peer_id: PeerId },
    /// Task completed successfully.
    Done(TaskResult),
    /// Task failed (bad hash, peer error, timeout).
    Failed(String),
    /// Task was cancelled by the coordinator.
    Cancelled,
}

/// A handle to a P2P task, implementing the `RemoteJob` trait.
pub struct P2pJob {
    task_id: String,
    peer_id: PeerId,
    state: Arc<Mutex<P2pJobState>>,
    result_rx: Mutex<Option<oneshot::Receiver<Result<TaskResult, String>>>>,
}

impl P2pJob {
    /// Create a new P2P job handle.
    pub fn new(
        task_id: String,
        peer_id: PeerId,
        result_rx: oneshot::Receiver<Result<TaskResult, String>>,
    ) -> Self {
        Self {
            task_id,
            peer_id,
            state: Arc::new(Mutex::new(P2pJobState::Running {
                peer_id: PeerId([0u8; 32]), // will be set when peer accepts
            })),
            result_rx: Mutex::new(Some(result_rx)),
        }
    }

    /// Get the current state (for testing).
    pub fn state(&self) -> P2pJobState {
        // We can't clone P2pJobState, so we just check via poll semantics.
        // This is a testing-only method.
        let rx = self.result_rx.lock();
        if rx.is_none() {
            // Result already consumed.
            return P2pJobState::Done(TaskResult {
                task_id: self.task_id.clone(),
                peer_id: self.peer_id.clone(),
                output_hash: crate::framework::artifact::ContentHash::of_bytes(&[0u8; 32]),
                encrypted_output: None,
                wall_time_ms: 0,
                signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
            });
        }
        P2pJobState::Pending
    }
}

impl RemoteJob for P2pJob {
    fn id(&self) -> &str {
        &self.task_id
    }

    fn poll(&self) -> Result<JobState, TrainError> {
        let mut rx_guard = self.result_rx.lock();
        if let Some(rx) = rx_guard.as_mut() {
            match rx.try_recv() {
                Ok(Ok(result)) => {
                    *rx_guard = None;
                    *self.state.lock() = P2pJobState::Done(result);
                    Ok(JobState::Succeeded)
                }
                Ok(Err(reason)) => {
                    *rx_guard = None;
                    *self.state.lock() = P2pJobState::Failed(reason.clone());
                    Ok(JobState::Failed(reason))
                }
                Err(oneshot::error::TryRecvError::Empty) => {
                    Ok(JobState::Running)
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    *rx_guard = None;
                    *self.state.lock() = P2pJobState::Failed("channel closed".into());
                    Ok(JobState::Failed("channel closed".into()))
                }
            }
        } else {
            // Already consumed.
            match &*self.state.lock() {
                P2pJobState::Done(_) => Ok(JobState::Succeeded),
                P2pJobState::Failed(r) => Ok(JobState::Failed(r.clone())),
                P2pJobState::Cancelled => Ok(JobState::Cancelled),
                _ => Ok(JobState::Running),
            }
        }
    }

    fn stream(&self, sink: &dyn Fn(&str)) -> Result<(), TrainError> {
        // P2P jobs don't stream intermediate output — the result comes
        // as a single TaskResult. Progress is reported via poll().
        // Wait for the result, then report it.
        let mut rx_guard = self.result_rx.lock();
        if let Some(rx) = rx_guard.as_mut() {
            match rx.try_recv() {
                Ok(Ok(result)) => {
                    sink(&format!("P2P task {} completed in {}ms", self.task_id, result.wall_time_ms));
                    *rx_guard = None;
                    *self.state.lock() = P2pJobState::Done(result);
                    Ok(())
                }
                Ok(Err(reason)) => {
                    sink(&format!("P2P task {} failed: {reason}", self.task_id));
                    *rx_guard = None;
                    *self.state.lock() = P2pJobState::Failed(reason.clone());
                    Err(TrainError::other(reason))
                }
                Err(oneshot::error::TryRecvError::Empty) => {
                    // Still running — nothing to report yet.
                    Ok(())
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    *rx_guard = None;
                    *self.state.lock() = P2pJobState::Failed("channel closed".into());
                    Err(TrainError::other("P2P job channel closed"))
                }
            }
        } else {
            Ok(())
        }
    }

    fn cancel(&self) -> Result<(), TrainError> {
        *self.state.lock() = P2pJobState::Cancelled;
        // Drop the receiver so the coordinator's send fails.
        *self.result_rx.lock() = None;
        tracing::info!("P2P task {} cancelled", self.task_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2p::peer::PeerId;

    #[test]
    fn job_poll_pending() {
        let (_tx, rx) = oneshot::channel();
        let job = P2pJob::new("test-1".into(), PeerId([0u8; 32]), rx);
        assert!(matches!(job.poll().unwrap(), JobState::Running));
    }

    #[test]
    fn job_poll_success() {
        let (tx, rx) = oneshot::channel();
        let job = P2pJob::new("test-1".into(), PeerId([0u8; 32]), rx);

        let result = TaskResult {
            task_id: "test-1".into(),
            peer_id: PeerId([0u8; 32]),
            output_hash: crate::framework::artifact::ContentHash::of_bytes(&[42u8; 32]),
            encrypted_output: None,
            wall_time_ms: 1000,
            signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
        };
        tx.send(Ok(result)).unwrap();

        assert!(matches!(job.poll().unwrap(), JobState::Succeeded));
        // Subsequent polls return the same state.
        assert!(matches!(job.poll().unwrap(), JobState::Succeeded));
    }

    #[test]
    fn job_poll_failure() {
        let (tx, rx) = oneshot::channel();
        let job = P2pJob::new("test-1".into(), PeerId([0u8; 32]), rx);

        tx.send(Err("bad hash".into())).unwrap();
        assert!(matches!(job.poll().unwrap(), JobState::Failed(_)));
    }

    #[test]
    fn job_cancel() {
        let (_tx, rx) = oneshot::channel();
        let job = P2pJob::new("test-1".into(), PeerId([0u8; 32]), rx);
        job.cancel().unwrap();
        assert!(matches!(job.poll().unwrap(), JobState::Cancelled));
    }

    #[test]
    fn job_id() {
        let (_tx, rx) = oneshot::channel();
        let job = P2pJob::new("test-42".into(), PeerId([0u8; 32]), rx);
        assert_eq!(job.id(), "test-42");
    }
}
