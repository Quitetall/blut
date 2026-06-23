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

/// Handle to a task submitted to the coordinator, used to track its
/// lifecycle and deliver the result.
struct PendingTask {
    result_tx: oneshot::Sender<Result<TaskResult, String>>,
    expected_output_hash: ContentHash,
}

/// The P2P coordinator. Manages peers, dispatches tasks, verifies results.
pub struct Coordinator {
    server: Arc<P2pServer>,
    dispatch: Arc<dyn DispatchPolicy>,
    pending: Arc<RwLock<HashMap<String, PendingTask>>>,
    /// Active peer connections, keyed by PeerId.
    connections: Arc<RwLock<HashMap<PeerId, quinn::Connection>>>,
}

impl Coordinator {
    /// Start the coordinator, binding the QUIC server to `addr`.
    pub async fn start(
        addr: SocketAddr,
        keypair: Arc<KeyPair>,
        dispatch: Arc<dyn DispatchPolicy>,
        peers: PeerRegistry,
    ) -> Result<Self, TrainError> {
        let server = Arc::new(P2pServer::bind(addr, keypair, peers).await?);
        let pending = Arc::new(RwLock::new(HashMap::new()));
        let connections = Arc::new(RwLock::new(HashMap::new()));

        // Spawn the accept loop.
        let server_c = server.clone();
        let pending_c = pending.clone();
        let dispatch_c = dispatch.clone();
        let connections_c = connections.clone();
        tokio::spawn(async move {
            Self::accept_loop(server_c, pending_c, dispatch_c, connections_c).await;
        });

        Ok(Self { server, dispatch, pending, connections })
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
            let mut pending = self.pending.write().await;
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
            let mut pending = self.pending.write().await;
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
        pending: Arc<RwLock<HashMap<String, PendingTask>>>,
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
        pending: Arc<RwLock<HashMap<String, PendingTask>>>,
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
                            let pending_map = pending.read().await;
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
                            let mut pending_map = pending.write().await;
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
                            let mut pending_map = pending.write().await;
                            if let Some(pt) = pending_map.remove(&task_id) {
                                let _ = pt.result_tx.send(Err(reason));
                            }
                        }
                        DispatchVerdict::RetryOnDifferentPeer => {
                            tracing::info!("Task {} needs retry on different peer", task_id);
                            let mut pending_map = pending.write().await;
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
