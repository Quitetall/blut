// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! P2P coordinator — manages task dispatch, peer connections, and
//! result verification.
//!
//! The coordinator runs a QUIC server, accepts peer connections, dispatches
//! tasks from the DAG executor to peers, and verifies returned results.
//! It is the single point of control for the P2P compute network.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use tokio::sync::{Mutex, RwLock};
use tokio_util::sync::CancellationToken;

use crate::error::TrainError;
use crate::framework::artifact::ContentId;
use crate::framework::execution::{
    Assignment, ExecutionAdapter, ExecutionArtifact, ExecutionDeadline, ExecutionFailure,
    ExecutionFailureKind, ExecutionHandle, ExecutionLifecycle, ExecutionMode, ExecutionPhase,
    ExecutionRequest, ExecutionSnapshot, ExecutionTerminal,
};
use crate::p2p::crypto::KeyPair;
use crate::p2p::dispatch::{DispatchPolicy, DispatchVerdict};
use crate::p2p::peer::PeerId;
use crate::p2p::peer_exec::{StoredDispatchOutcome, dispatch_stored_to_peer};
use crate::p2p::registry::PeerRegistry;
use crate::p2p::task::TaskResult;
use crate::p2p::transport::P2pServer;

/// One live QUIC peer. Blob receive consumes every uni stream, so this gate is
/// load-bearing: one task owns a connection until its terminal reply arrives.
#[derive(Clone)]
pub(crate) struct PeerSession {
    conn: quinn::Connection,
    gate: Arc<Mutex<()>>,
}

/// The P2P coordinator. Manages peers, dispatches tasks, verifies results.
pub struct Coordinator {
    server: Arc<P2pServer>,
    dispatch: Arc<dyn DispatchPolicy>,
    /// The coordinator's keypair for signing task manifests.
    keypair: Arc<KeyPair>,
    /// Active peer connections, keyed by PeerId.
    pub(crate) connections: Arc<RwLock<HashMap<PeerId, PeerSession>>>,
    /// Strictly increasing fence for every assignment, including re-use of one
    /// peer after a prior lease or connection is abandoned.
    generation: Arc<AtomicU64>,
}

impl Coordinator {
    /// Start the coordinator, binding the QUIC server to `addr`.
    pub async fn start(
        addr: SocketAddr,
        keypair: Arc<KeyPair>,
        dispatch: Arc<dyn DispatchPolicy>,
        peers: PeerRegistry,
    ) -> Result<Self, TrainError> {
        let server = Arc::new(P2pServer::bind(addr, keypair.clone(), peers).await?);
        let connections = Arc::new(RwLock::new(HashMap::new()));

        // Spawn the accept loop.
        let server_c = server.clone();
        let connections_c = connections.clone();
        tokio::spawn(async move {
            Self::accept_loop(server_c, connections_c).await;
        });

        Ok(Self {
            server,
            dispatch,
            keypair,
            connections,
            generation: Arc::new(AtomicU64::new(0)),
        })
    }

    /// The coordinator's local address.
    pub fn local_addr(&self) -> Result<SocketAddr, TrainError> {
        self.server.local_addr()
    }

    /// Get the peer registry.
    pub fn peers(&self) -> Arc<RwLock<PeerRegistry>> {
        self.server.peers.clone()
    }

    /// Verify a task result. Returns the dispatch verdict.
    pub async fn verify_result(
        &self,
        result: &TaskResult,
        expected: Option<ContentId>,
    ) -> DispatchVerdict {
        let peers = self.server.peers.read().await;
        if let Some(peer) = peers.get(&result.peer_id) {
            self.dispatch.verify_result(result, expected, &peer.pubkey)
        } else {
            DispatchVerdict::Reject(format!("unknown peer: {}", result.peer_id))
        }
    }

    /// The accept loop only owns live connection registration. Per-attempt
    /// adapters own task replies and blob streams; a background result reader
    /// would race `recv_blob`, which consumes all incoming uni streams.
    async fn accept_loop(
        server: Arc<P2pServer>,
        connections: Arc<RwLock<HashMap<PeerId, PeerSession>>>,
    ) {
        loop {
            match server.accept_peer().await {
                Ok((peer_id, conn)) => {
                    let stable_id = conn.stable_id();
                    let replaced = {
                        let mut conns = connections.write().await;
                        conns.insert(
                            peer_id.clone(),
                            PeerSession {
                                conn: conn.clone(),
                                gate: Arc::new(Mutex::new(())),
                            },
                        )
                    };
                    if let Some(previous) = replaced {
                        previous
                            .conn
                            .close(0u32.into(), b"peer reconnected with a new session");
                    }
                    let connections_c = connections.clone();
                    tokio::spawn(async move {
                        let _ = conn.closed().await;
                        let mut conns = connections_c.write().await;
                        if conns
                            .get(&peer_id)
                            .is_some_and(|session| session.conn.stable_id() == stable_id)
                        {
                            conns.remove(&peer_id);
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!("P2P accept error: {e}");
                }
            }
        }
    }

    /// Shut down the coordinator.
    pub fn shutdown(&self) {
        if let Ok(connections) = self.connections.try_read() {
            for session in connections.values() {
                session.conn.close(0u32.into(), b"coordinator shutdown");
            }
        } else {
            tracing::warn!("P2P coordinator shutdown could not immediately lock peer sessions");
        }
        self.server.shutdown();
    }

    /// Apply a reputation update in memory and try to persist the peer
    /// registry to disk, logging (rather than silently discarding) any
    /// persistence failure.
    ///
    /// The in-memory update via `update_reputation` always applies
    /// regardless of whether `save()` succeeds — a failed persist must not
    /// be allowed to crash the peer loop, but it also must not vanish with
    /// zero log trail: if `peers.json`'s directory hits ENOSPC or the
    /// process is killed/restarted before a later successful save,
    /// `PeerRegistry::load()` will read the stale on-disk file on restart.
    ///
    /// `success = false` (a rejection-driven demotion, e.g. a peer just
    /// returned a tampered/failing result) logs at `error!` rather than
    /// `warn!`: losing that persist is security-relevant — a known-bad peer
    /// can silently regain its pre-demotion trust level after a restart,
    /// not just an operational hiccup.
    async fn persist_reputation_update(
        peers: &RwLock<PeerRegistry>,
        peer_id: &PeerId,
        success: bool,
        context: &str,
    ) {
        let mut peers = peers.write().await;
        peers.update_reputation(peer_id, success);
        if let Err(e) = peers.save() {
            if success {
                tracing::warn!(
                    "p2p: failed to persist peer registry after reputation update for {peer_id} ({context}): {e}"
                );
            } else {
                tracing::error!(
                    "p2p: failed to persist peer registry after reputation demotion for {peer_id} ({context}): {e}"
                );
            }
        }
    }
}

struct CoordinatorExecutionHandle {
    lifecycle: ExecutionLifecycle,
    cancel: CancellationToken,
    connection: Arc<Mutex<Option<quinn::Connection>>>,
}

#[async_trait]
impl ExecutionHandle for CoordinatorExecutionHandle {
    async fn snapshot(&self) -> Result<ExecutionSnapshot, ExecutionFailure> {
        Ok(self.lifecycle.snapshot())
    }

    async fn cancel(&self) -> Result<(), ExecutionFailure> {
        self.cancel.cancel();
        if let Some(conn) = self.connection.lock().await.take() {
            // `recv_blob` owns every incoming uni stream, so a per-task Cancel
            // frame would corrupt an in-flight transfer. The session gate gives
            // this attempt exclusive connection ownership; close is safe and
            // peer execution observes it through `conn.closed()`.
            conn.close(0u32.into(), b"execution cancelled");
        }
        Ok(())
    }
}

/// Last-resort lifecycle latch for a panic or newly added early return in the
/// detached adapter task. First-terminal-wins keeps explicit outcomes intact.
struct P2pAttemptGuard(ExecutionLifecycle);

impl Drop for P2pAttemptGuard {
    fn drop(&mut self) {
        if self.0.snapshot().terminal.is_none() {
            let _ = self.0.finish(
                None,
                ExecutionTerminal::Failed {
                    failure: ExecutionFailure::new(
                        ExecutionFailureKind::Unknown,
                        "EXECUTION_ABANDONED",
                        "P2P adapter exited before recording a terminal outcome",
                    ),
                },
            );
        }
    }
}

async fn sleep_optional(duration: Option<std::time::Duration>) {
    match duration {
        Some(duration) => tokio::time::sleep(duration).await,
        None => std::future::pending::<()>().await,
    }
}

fn finish_cancel(
    lifecycle: &ExecutionLifecycle,
    assignment: Option<&Assignment>,
    reason: impl Into<String>,
) {
    let _ = lifecycle.finish(
        assignment,
        ExecutionTerminal::Cancelled {
            reason: reason.into(),
        },
    );
}

fn finish_timeout(
    lifecycle: &ExecutionLifecycle,
    assignment: Option<&Assignment>,
    deadline_unix_ms: u64,
) {
    let _ = lifecycle.finish(
        assignment,
        ExecutionTerminal::TimedOut {
            phase: lifecycle.snapshot().phase,
            deadline_unix_ms,
        },
    );
}

fn finish_cancel_or_timeout(
    lifecycle: &ExecutionLifecycle,
    assignment: Option<&Assignment>,
    deadline: ExecutionDeadline,
    reason: &str,
) {
    if let Some(soft) = deadline.soft_remaining()
        && soft.is_zero()
    {
        finish_timeout(
            lifecycle,
            assignment,
            deadline.soft_unix_ms.expect("soft duration exists"),
        );
    } else if deadline.hard_remaining().is_zero() {
        finish_timeout(lifecycle, assignment, deadline.hard_unix_ms);
    } else {
        finish_cancel(lifecycle, assignment, reason);
    }
}

async fn select_peer_session(
    dispatch: &dyn DispatchPolicy,
    peers: &RwLock<PeerRegistry>,
    connections: &RwLock<HashMap<PeerId, PeerSession>>,
    request: &ExecutionRequest,
) -> Option<(crate::p2p::peer::PeerInfo, PeerSession)> {
    let connected_ids: Vec<PeerId> = connections.read().await.keys().cloned().collect();
    let peer = {
        let registry = peers.read().await;
        let candidates: Vec<crate::p2p::peer::PeerInfo> = connected_ids
            .iter()
            .filter_map(|id| registry.get(id).cloned())
            .collect();
        let resources = crate::p2p::task::ResourceRequest {
            cpu_cores: request.resources.cpu_cores,
            memory_gib: request.resources.memory_gib,
            gpu: request.resources.gpu,
            gpu_vram_gib: request.resources.gpu_vram_gib,
        };
        dispatch
            .select_peer(
                &request.stage_name,
                &resources,
                request.data_class.into(),
                &candidates,
            )
            .and_then(|id| registry.get(&id).cloned())
    }?;
    let session = connections.read().await.get(&peer.id).cloned()?;
    Some((peer, session))
}

struct P2pAttempt {
    dispatch: Arc<dyn DispatchPolicy>,
    keypair: Arc<KeyPair>,
    peers: Arc<RwLock<PeerRegistry>>,
    connections: Arc<RwLock<HashMap<PeerId, PeerSession>>>,
    generation: Arc<AtomicU64>,
    lifecycle: ExecutionLifecycle,
    cancel: CancellationToken,
    active_connection: Arc<Mutex<Option<quinn::Connection>>>,
    request: ExecutionRequest,
}

impl P2pAttempt {
    async fn run(self) {
        let Self {
            dispatch,
            keypair,
            peers,
            connections,
            generation,
            lifecycle,
            cancel,
            active_connection,
            request,
        } = self;
        let _guard = P2pAttemptGuard(lifecycle.clone());
        if cancel.is_cancelled() {
            finish_cancel_or_timeout(
                &lifecycle,
                None,
                request.deadline,
                "cancelled before queueing",
            );
            return;
        }
        if lifecycle.transition(ExecutionPhase::Queued, None).is_err() {
            return;
        }

        let Some((peer, session)) = select_peer_session(
            dispatch.as_ref(),
            peers.as_ref(),
            connections.as_ref(),
            &request,
        )
        .await
        else {
            let _ = lifecycle.finish(
                None,
                ExecutionTerminal::Failed {
                    failure: ExecutionFailure::unavailable(
                        "no connected peer cleared the P2P dispatch policy",
                    ),
                },
            );
            return;
        };

        let gate = tokio::select! {
            guard = session.gate.lock() => guard,
            _ = cancel.cancelled() => {
                finish_cancel_or_timeout(&lifecycle, None, request.deadline, "cancelled while queued for peer");
                return;
            }
            _ = sleep_optional(request.deadline.soft_remaining()) => {
                finish_timeout(&lifecycle, None, request.deadline.soft_unix_ms.unwrap_or(request.deadline.hard_unix_ms));
                return;
            }
            _ = tokio::time::sleep(request.deadline.hard_remaining()) => {
                finish_timeout(&lifecycle, None, request.deadline.hard_unix_ms);
                return;
            }
        };
        if session.conn.close_reason().is_some() {
            let _ = lifecycle.finish(
                None,
                ExecutionTerminal::Failed {
                    failure: ExecutionFailure::disconnected(
                        "selected peer disconnected before assignment",
                    ),
                },
            );
            return;
        }

        let assignment = Assignment::new(
            peer.id.to_string(),
            generation.fetch_add(1, Ordering::Relaxed).saturating_add(1),
        );
        if lifecycle
            .transition(ExecutionPhase::Assigned, Some(assignment.clone()))
            .is_err()
        {
            return;
        }
        *active_connection.lock().await = Some(session.conn.clone());
        if cancel.is_cancelled() {
            session.conn.close(0u32.into(), b"execution cancelled");
            let _ = active_connection.lock().await.take();
            finish_cancel_or_timeout(
                &lifecycle,
                Some(&assignment),
                request.deadline,
                "cancelled before P2P transfer",
            );
            return;
        }

        let transfer = dispatch_stored_to_peer(
            &session.conn,
            keypair.as_ref(),
            &peer,
            &request,
            Some(dispatch.as_ref()),
            Some((&lifecycle, &assignment)),
        );
        tokio::pin!(transfer);
        let outcome = tokio::select! {
            outcome = &mut transfer => outcome,
            _ = cancel.cancelled() => {
                session.conn.close(0u32.into(), b"execution cancelled");
                let _ = active_connection.lock().await.take();
                finish_cancel_or_timeout(&lifecycle, Some(&assignment), request.deadline, "cancelled during P2P transfer");
                return;
            }
            _ = sleep_optional(request.deadline.soft_remaining()) => {
                session.conn.close(0u32.into(), b"execution soft deadline");
                let _ = active_connection.lock().await.take();
                finish_timeout(&lifecycle, Some(&assignment), request.deadline.soft_unix_ms.unwrap_or(request.deadline.hard_unix_ms));
                return;
            }
            _ = tokio::time::sleep(request.deadline.hard_remaining()) => {
                session.conn.close(0u32.into(), b"execution hard deadline");
                let _ = active_connection.lock().await.take();
                finish_timeout(&lifecycle, Some(&assignment), request.deadline.hard_unix_ms);
                return;
            }
        };
        drop(gate);
        let _ = active_connection.lock().await.take();

        match outcome {
            Ok(StoredDispatchOutcome::Succeeded(output)) => {
                let output = *output;
                Coordinator::persist_reputation_update(
                    peers.as_ref(),
                    &peer.id,
                    true,
                    &format!("task {} accepted", request.execution_id),
                )
                .await;
                let _ = lifecycle.finish(
                    Some(&assignment),
                    ExecutionTerminal::Succeeded {
                        artifact: ExecutionArtifact {
                            content_id: output.content_id,
                            stored: Some(output.stored),
                        },
                        wall_time_ms: output.wall_time_ms,
                    },
                );
            }
            Ok(StoredDispatchOutcome::Cancelled { reason }) => {
                finish_cancel(&lifecycle, Some(&assignment), reason);
            }
            Err(failure) => {
                Coordinator::persist_reputation_update(
                    peers.as_ref(),
                    &peer.id,
                    false,
                    &format!("task {} failed: {}", request.execution_id, failure.code),
                )
                .await;
                match failure.code.as_str() {
                    "EXECUTION_SOFT_DEADLINE" => finish_timeout(
                        &lifecycle,
                        Some(&assignment),
                        request
                            .deadline
                            .soft_unix_ms
                            .unwrap_or(request.deadline.hard_unix_ms),
                    ),
                    "EXECUTION_HARD_DEADLINE" => {
                        finish_timeout(&lifecycle, Some(&assignment), request.deadline.hard_unix_ms)
                    }
                    _ => {
                        let _ = lifecycle
                            .finish(Some(&assignment), ExecutionTerminal::Failed { failure });
                    }
                }
            }
        }
    }
}

#[async_trait]
impl ExecutionAdapter for Coordinator {
    fn mode(&self) -> ExecutionMode {
        ExecutionMode::P2p
    }

    async fn submit(
        &self,
        request: ExecutionRequest,
    ) -> Result<Box<dyn ExecutionHandle>, ExecutionFailure> {
        if request.protocol_version != crate::framework::execution::EXECUTION_PROTOCOL_VERSION {
            return Err(ExecutionFailure::protocol(format!(
                "execution protocol v{} unsupported (want v{})",
                request.protocol_version,
                crate::framework::execution::EXECUTION_PROTOCOL_VERSION,
            )));
        }
        if !crate::p2p::trust::custody_allows_off_box(&request.tenant, request.data_class.into()) {
            return Err(ExecutionFailure::protocol(format!(
                "P2P execution denied by custody policy for tenant '{}' and {:?} data",
                request.tenant, request.data_class
            )));
        }
        let lifecycle = ExecutionLifecycle::new(ExecutionMode::P2p);
        let cancel = CancellationToken::new();
        let connection = Arc::new(Mutex::new(None));
        tokio::spawn(
            P2pAttempt {
                dispatch: self.dispatch.clone(),
                keypair: self.keypair.clone(),
                peers: self.server.peers.clone(),
                connections: self.connections.clone(),
                generation: self.generation.clone(),
                lifecycle: lifecycle.clone(),
                cancel: cancel.clone(),
                active_connection: connection.clone(),
                request,
            }
            .run(),
        );
        Ok(Box::new(CoordinatorExecutionHandle {
            lifecycle,
            cancel,
            connection,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2p::crypto::KeyPair;
    use crate::p2p::peer::PeerCapabilities;
    use crate::p2p::trust::TrustLevel;

    fn make_peer(trust: TrustLevel) -> crate::p2p::peer::PeerInfo {
        let kp = KeyPair::generate();
        crate::p2p::peer::PeerInfo::new(
            kp.verifying,
            kp.x25519_public,
            trust,
            PeerCapabilities::default(),
        )
    }

    /// Regression test for the silent `let _ = peers.save();` bug: a
    /// reputation demotion must still apply in memory even when the
    /// subsequent persist to disk fails (e.g. ENOSPC, read-only
    /// filesystem). Before the fix, callers had no way to distinguish "save
    /// failed" from "save succeeded" — both looked identical from the
    /// in-memory registry's perspective, which made the bug invisible to a
    /// black-box test. What we CAN verify without a tracing-capture harness
    /// (not a dev-dependency of this crate) is the behavioral contract that
    /// actually matters operationally: `persist_reputation_update` must not
    /// panic when `save()` fails, and the in-memory reputation update must
    /// still have applied (so the running coordinator's *current* dispatch
    /// decisions are correct even if the on-disk copy is stale).
    #[cfg_attr(not(unix), ignore = "relies on unix directory permission bits")]
    #[tokio::test]
    async fn persist_reputation_update_applies_in_memory_when_save_fails() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("peers.json");

            let mut reg = PeerRegistry::load(&path).unwrap();
            let peer = make_peer(TrustLevel::Registered);
            let id = peer.id.clone();
            reg.add(peer).unwrap();
            // One successful save so the file exists before we break writes.
            reg.save().unwrap();

            let peers = RwLock::new(reg);

            // Make the directory read-only so PeerRegistry::save()'s
            // `std::fs::write(&tmp, ..)` fails with a permission error —
            // stands in for ENOSPC / a read-only filesystem without
            // actually needing to exhaust disk space.
            let mut perms = std::fs::metadata(dir.path()).unwrap().permissions();
            perms.set_mode(0o500);
            std::fs::set_permissions(dir.path(), perms.clone()).unwrap();

            Coordinator::persist_reputation_update(&peers, &id, false, "test-reject").await;

            // Restore write perms so the tempdir can clean itself up.
            perms.set_mode(0o700);
            std::fs::set_permissions(dir.path(), perms).unwrap();

            let guard = peers.read().await;
            let p = guard.get(&id).unwrap();
            assert_eq!(
                p.tasks_failed, 1,
                "reputation demotion must apply in-memory even when persistence fails"
            );
        }
    }

    /// Sanity companion: when `save()` succeeds, behavior is unchanged from
    /// before this refactor (in-memory update + on-disk update agree).
    #[tokio::test]
    async fn persist_reputation_update_persists_on_success() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("peers.json");

        let mut reg = PeerRegistry::load(&path).unwrap();
        let peer = make_peer(TrustLevel::Registered);
        let id = peer.id.clone();
        reg.add(peer).unwrap();
        reg.save().unwrap();

        let peers = RwLock::new(reg);
        Coordinator::persist_reputation_update(&peers, &id, true, "test-accept").await;

        let guard = peers.read().await;
        assert_eq!(guard.get(&id).unwrap().tasks_completed, 1);
        drop(guard);

        // Reload from disk to confirm the save actually happened.
        let reloaded = PeerRegistry::load(&path).unwrap();
        assert_eq!(reloaded.get(&id).unwrap().tasks_completed, 1);
    }
}
