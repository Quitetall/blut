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

use tokio::sync::{oneshot, RwLock};

use crate::error::TrainError;
use crate::framework::artifact::ContentHash;
use crate::p2p::crypto::KeyPair;
use crate::p2p::dispatch::{DispatchPolicy, DispatchVerdict};
use crate::p2p::job::P2pJob;
use crate::p2p::peer::PeerId;
use crate::p2p::registry::PeerRegistry;
use crate::p2p::task::{TaskManifest, TaskResult};
use crate::p2p::transport::P2pServer;
use crate::config::launcher::JobState;

/// Handle to a task submitted to the coordinator, used to track its
/// lifecycle and deliver the result.
pub(crate) struct PendingTask {
    result_tx: oneshot::Sender<Result<TaskResult, String>>,
    expected_output_hash: ContentHash,
}

/// The P2P coordinator. Manages peers, dispatches tasks, verifies results.
pub struct Coordinator {
    server: Arc<P2pServer>,
    dispatch: Arc<dyn DispatchPolicy>,
    /// The coordinator's keypair for signing task manifests.
    keypair: Arc<KeyPair>,
    /// The coordinator's peer ID (derived from keypair).
    coordinator_id: PeerId,
    /// Pending tasks. Uses parking_lot so the sync DispatchSubmitter::submit
    /// can lock without async.
    pub(crate) pending: Arc<parking_lot::RwLock<HashMap<String, PendingTask>>>,
    /// Active peer connections, keyed by PeerId.
    pub(crate) connections: Arc<RwLock<HashMap<PeerId, quinn::Connection>>>,
}

impl Coordinator {
    /// Start the coordinator, binding the QUIC server to `addr`.
    pub async fn start(
        addr: SocketAddr,
        keypair: Arc<KeyPair>,
        dispatch: Arc<dyn DispatchPolicy>,
        peers: PeerRegistry,
    ) -> Result<Self, TrainError> {
        let coordinator_id = PeerId::from_pubkey(&keypair.verifying);
        let server = Arc::new(P2pServer::bind(addr, keypair.clone(), peers).await?);
        let pending = Arc::new(parking_lot::RwLock::new(HashMap::new()));
        let connections = Arc::new(RwLock::new(HashMap::new()));

        // Spawn the accept loop.
        let server_c = server.clone();
        let pending_c = pending.clone();
        let dispatch_c = dispatch.clone();
        let connections_c = connections.clone();
        tokio::spawn(async move {
            Self::accept_loop(server_c, pending_c, dispatch_c, connections_c).await;
        });

        Ok(Self { server, dispatch, keypair, coordinator_id, pending, connections })
    }

    /// The coordinator's local address.
    pub fn local_addr(&self) -> Result<SocketAddr, TrainError> {
        self.server.local_addr()
    }

    /// Get the peer registry.
    pub fn peers(&self) -> Arc<RwLock<PeerRegistry>> {
        self.server.peers.clone()
    }

    /// Submit a task to the coordinator. The coordinator will dispatch it
    /// to the best available peer. Returns a `P2pJob` handle for tracking.
    pub async fn submit_task(&self, manifest: TaskManifest) -> Result<P2pJob, TrainError> {
        let task_id = manifest.task_id.clone();
        let expected_output_hash = manifest.expected_output_hash;

        let (result_tx, result_rx) = oneshot::channel();

        // Select a peer.
        let peers = self.server.peers.read().await;
        let peer_list: Vec<_> = peers.list().into_iter().cloned().collect();
        drop(peers);

        let data_class = self.dispatch.classify_stage(&manifest.stage_name, &manifest.args);
        let peer_id = self.dispatch.select_peer(
            &manifest.stage_name,
            &manifest.resources,
            data_class,
            &peer_list,
        ).ok_or_else(|| TrainError::other("no suitable peer available"))?;

        // Register as pending.
        {
            let mut pending = self.pending.write();
            pending.insert(task_id.clone(), PendingTask {
                result_tx,
                expected_output_hash,
            });
        }

        // Send the task to the peer. Clone the connection out, drop the
        // lock, then send — avoids holding the read lock during I/O.
        let conn = {
            let connections = self.connections.read().await;
            connections.get(&peer_id).cloned()
        };
        if let Some(conn) = conn {
            P2pServer::send_task(&conn, &manifest).await?;
            tracing::info!("Dispatched task {} to peer {}", task_id, peer_id);
        } else {
            // Peer not connected — remove from pending and fail.
            let mut pending = self.pending.write();
            pending.remove(&task_id);
            return Err(TrainError::other(format!(
                "peer {} not connected (task {})", peer_id, task_id
            )));
        }

        let job = P2pJob::new(task_id, peer_id, result_rx);
        Ok(job)
    }

    /// Verify a task result. Returns the dispatch verdict.
    pub async fn verify_result(&self, result: &TaskResult, expected: &ContentHash) -> DispatchVerdict {
        let peers = self.server.peers.read().await;
        if let Some(peer) = peers.get(&result.peer_id) {
            self.dispatch.verify_result(result, expected, &peer.pubkey)
        } else {
            DispatchVerdict::Reject(format!("unknown peer: {}", result.peer_id))
        }
    }

    /// The accept loop: processes incoming peer connections and dispatches
    /// results to pending tasks.
    async fn accept_loop(
        server: Arc<P2pServer>,
        pending: Arc<parking_lot::RwLock<HashMap<String, PendingTask>>>,
        dispatch: Arc<dyn DispatchPolicy>,
        connections: Arc<RwLock<HashMap<PeerId, quinn::Connection>>>,
    ) {
        loop {
            match server.accept_peer().await {
                Ok((peer_id, conn)) => {
                    // Store the connection.
                    {
                        let mut conns = connections.write().await;
                        conns.insert(peer_id.clone(), conn.clone());
                    }

                    let server_c = server.clone();
                    let pending_c = pending.clone();
                    let dispatch_c = dispatch.clone();
                    let connections_c = connections.clone();
                    tokio::spawn(async move {
                        Self::handle_peer(peer_id.clone(), conn, server_c, pending_c, dispatch_c).await;
                        // Remove connection when peer disconnects.
                        connections_c.write().await.remove(&peer_id);
                    });
                }
                Err(e) => {
                    tracing::warn!("P2P accept error: {e}");
                }
            }
        }
    }

    /// Handle a single peer connection: receive results and deliver them
    /// to pending tasks.
    async fn handle_peer(
        peer_id: PeerId,
        conn: quinn::Connection,
        server: Arc<P2pServer>,
        pending: Arc<parking_lot::RwLock<HashMap<String, PendingTask>>>,
        dispatch: Arc<dyn DispatchPolicy>,
    ) {
        tracing::debug!("Handling peer {}", peer_id);

        loop {
            match P2pServer::recv_result(&conn).await {
                Ok(result) => {
                    let task_id = result.task_id.clone();

                    // Verify the result.
                    let verdict = {
                        let peers = server.peers.read().await;
                        if let Some(peer_info) = peers.get(&result.peer_id) {
                            let pending_map = pending.read();
                            if let Some(pt) = pending_map.get(&task_id) {
                                dispatch.verify_result(&result, &pt.expected_output_hash, &peer_info.pubkey)
                            } else {
                                DispatchVerdict::Reject(format!("no pending task: {task_id}"))
                            }
                        } else {
                            DispatchVerdict::Reject(format!("unknown peer: {}", result.peer_id))
                        }
                    };

                    match verdict {
                        DispatchVerdict::Accept => {
                            tracing::info!("Task {} completed by peer {}", task_id, peer_id);
                            {
                                let mut peers = server.peers.write().await;
                                peers.update_reputation(&result.peer_id, true);
                                let _ = peers.save();
                            }
                            let mut pending_map = pending.write();
                            if let Some(pt) = pending_map.remove(&task_id) {
                                let _ = pt.result_tx.send(Ok(result));
                            }
                        }
                        DispatchVerdict::Reject(reason) => {
                            tracing::warn!("Task {} rejected: {reason}", task_id);
                            {
                                let mut peers = server.peers.write().await;
                                peers.update_reputation(&result.peer_id, false);
                                let _ = peers.save();
                            }
                            let mut pending_map = pending.write();
                            if let Some(pt) = pending_map.remove(&task_id) {
                                let _ = pt.result_tx.send(Err(reason));
                            }
                        }
                        DispatchVerdict::RetryOnDifferentPeer => {
                            tracing::info!("Task {} needs retry on different peer", task_id);
                            let mut pending_map = pending.write();
                            if let Some(pt) = pending_map.remove(&task_id) {
                                let _ = pt.result_tx.send(Err("retry on different peer".into()));
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::debug!("Peer {} disconnected: {e}", peer_id);
                    break;
                }
            }
        }
    }

    /// Shut down the coordinator.
    pub fn shutdown(&self) {
        self.server.shutdown();
    }
}

/// A handle to a dispatched P2P task, polling via the oneshot receiver.
struct CoordinatorDispatchHandle {
    task_id: String,
    result_rx: parking_lot::Mutex<Option<tokio::sync::oneshot::Receiver<Result<TaskResult, String>>>>,
    pending: Arc<parking_lot::RwLock<HashMap<String, PendingTask>>>,
}

impl crate::framework::executor::DispatchHandle for CoordinatorDispatchHandle {
    fn poll(&self) -> Result<Option<crate::config::launcher::JobState>, crate::error::TrainError> {
        let mut rx_guard = self.result_rx.lock();
        if let Some(rx) = rx_guard.as_mut() {
            match rx.try_recv() {
                Ok(Ok(_result)) => {
                    *rx_guard = None;
                    // Clean up pending entry.
                    self.pending.write().remove(&self.task_id);
                    Ok(Some(JobState::Succeeded))
                }
                Ok(Err(reason)) => {
                    *rx_guard = None;
                    self.pending.write().remove(&self.task_id);
                    Ok(Some(JobState::Failed(reason)))
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => Ok(None),
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    *rx_guard = None;
                    self.pending.write().remove(&self.task_id);
                    Ok(Some(JobState::Failed("channel closed".into())))
                }
            }
        } else {
            Ok(Some(JobState::Cancelled))
        }
    }

    fn cancel(&self) -> Result<(), crate::error::TrainError> {
        // Drop the receiver so poll() returns Cancelled.
        *self.result_rx.lock() = None;
        // Remove from pending map.
        self.pending.write().remove(&self.task_id);
        tracing::info!("P2P dispatch task {} cancelled", self.task_id);
        Ok(())
    }
}

impl crate::framework::executor::DispatchSubmitter for Coordinator {
    fn submit(
        &self,
        req: crate::framework::executor::DispatchRequest<'_>,
    ) -> Result<Box<dyn crate::framework::executor::DispatchHandle>, crate::error::TrainError> {
        let task_id = format!("p2p-{}", uuid::Uuid::new_v4());
        let data_class = match req.data_class {
            0 => crate::p2p::trust::DataClass::Public,
            1 => crate::p2p::trust::DataClass::Internal,
            2 => crate::p2p::trust::DataClass::Restricted,
            other => return Err(TrainError::other(format!(
                "unknown data_class: {other} (expected 0=Public, 1=Internal, 2=Restricted)"
            ))),
        };
        let resources = crate::p2p::task::ResourceRequest {
            cpu_cores: req.resource_request.cpu_cores,
            memory_gib: req.resource_request.memory_gib,
            gpu: req.resource_request.gpu,
            gpu_vram_gib: req.resource_request.gpu_vram_gib,
        };
        // Captured for peer selection inside the async dispatch (the manifest
        // moves `resources`/`data_class`, so clone what `select_peer` needs).
        let select_stage = req.stage_name.to_string();
        let select_resources = resources; // ResourceRequest is Copy
        let select_data_class = data_class;

        let expected_output_hash = req.expected_output_hash;
        let mut manifest = TaskManifest {
            task_id: task_id.clone(),
            coordinator_id: self.coordinator_id.clone(),
            stage_name: req.stage_name.to_string(),
            stage_schema: req.stage_schema,
            input_hash: req.input_hash,
            args_hash: req.args_hash,
            expected_output_hash,
            args: req.args.clone(),
            resources,
            data_class,
            timeout_secs: 3600,
            encrypted_input: None,
            signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
        };
        manifest.signature = self.keypair.sign(&manifest.sign_payload());

        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        // Register as pending.
        {
            let mut pending = self.pending.write();
            pending.insert(task_id.clone(), PendingTask {
                result_tx,
                expected_output_hash,
            });
        }

        // Spawn the async dispatch.
        let connections = self.connections.clone();
        let pending_c = self.pending.clone();
        let task_id_c = task_id.clone();
        let dispatch = self.dispatch.clone();
        let registry = self.server.peers.clone();
        tokio::spawn(async move {
            let conn = {
                let conns = connections.read().await;
                // Route through the dispatch policy: gather the PeerInfo for the
                // peers we actually hold a live connection to, ask the policy to
                // pick one (trust matrix + capability + reputation), and dispatch
                // to THAT peer. `None` ⇒ no suitable peer (we fail the task back
                // rather than silently sending to an arbitrary connection).
                let selected = {
                    let reg = registry.read().await;
                    let candidates: Vec<crate::p2p::peer::PeerInfo> = conns
                        .keys()
                        .filter_map(|id| reg.get(id).cloned())
                        .collect();
                    dispatch.select_peer(
                        &select_stage,
                        &select_resources,
                        select_data_class,
                        &candidates,
                    )
                };
                selected.and_then(|peer_id| conns.get(&peer_id).cloned())
            };
            if let Some(conn) = conn {
                if let Err(e) = P2pServer::send_task(&conn, &manifest).await {
                    tracing::warn!("P2P dispatch failed for {}: {e}", task_id_c);
                    let mut pending = pending_c.write();
                    if let Some(pt) = pending.remove(&task_id_c) {
                        let _ = pt.result_tx.send(Err(format!("dispatch failed: {e}")));
                    }
                }
            } else {
                let mut pending = pending_c.write();
                if let Some(pt) = pending.remove(&task_id_c) {
                    let _ = pt.result_tx.send(Err(
                        "no suitable peer (none connected, or none cleared the dispatch policy)"
                            .into(),
                    ));
                }
            }
        });

        Ok(Box::new(CoordinatorDispatchHandle {
            task_id,
            result_rx: parking_lot::Mutex::new(Some(result_rx)),
            pending: self.pending.clone(),
        }))
    }
}
