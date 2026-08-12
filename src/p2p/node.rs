// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `MeshNode` — one process that is a QUIC server + worker + optional scheduler
//! (ADR 0079 A3).
//!
//! The legacy split — `blut p2p serve` (coordinator) vs `blut p2p connect`
//! (worker) — hard-codes the role at the CLI. A symmetric mesh makes the NODE
//! the unit: every node listens, can accept + run work (the `worker`
//! capability), and can dispatch work (the `scheduler` capability). "Coordinator"
//! becomes the scheduler capability; the existing dispatch machinery is reused,
//! only connection ownership moves into the node.
//!
//! A node accepts inbound connections (mutual-TLS-authenticated, A1), then
//! serves the symmetric bi-stream protocol (A2) on each: `Ping`→`Pong`,
//! `Hello`→`HelloAck`, and — if it has the `worker` capability — `Task`→run it
//! and reply `Result`. It dispatches by opening a bi-stream to a peer and
//! sending a `Task` (the `scheduler` capability).
//!
//! Task EXECUTION is pluggable via [`MeshTaskRunner`] so the node core is
//! testable without the full stage-execution stack; the production runner
//! (wrapping `peer_exec`'s materialize-run-seal path over mesh frames) is wired
//! separately.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::TrainError;
use crate::p2p::crypto::KeyPair;
use crate::p2p::mesh_wire::{MeshFrame, accept_request, mesh_request, write_frame};
use crate::p2p::peer::PeerId;
use crate::p2p::registry::PeerRegistry;
use crate::p2p::task::{TaskManifest, TaskResult};
use crate::p2p::transport::{P2pClient, P2pServer};
use crate::p2p::trust::TrustLevel;

/// What a node is willing to do. Both default on: a plain `blut p2p node` is a
/// full peer. `--no-worker` / `--no-scheduler` carve capabilities off (e.g. a
/// k8s worker pool runs `--no-scheduler` nodes).
#[derive(Clone, Copy, Debug)]
pub struct NodeCapabilities {
    /// Accept + execute dispatched tasks.
    pub worker: bool,
    /// Dispatch tasks to peers.
    pub scheduler: bool,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            worker: true,
            scheduler: true,
        }
    }
}

/// Executes a dispatched task and returns a signed result. Pluggable so the
/// node core is independent of the stage-execution stack.
///
/// RESOURCE OBLIGATION: the node serves `Task` frames without a built-in
/// concurrency cap or timeout — a production `MeshTaskRunner` MUST enforce
/// `task.timeout_secs` (e.g. `tokio::time::timeout`) and bound its own
/// concurrency (the broker admission gate / a semaphore), so a peer cannot
/// flood a worker. The node-level dispatch trust gate lands in A4.
#[async_trait]
pub trait MeshTaskRunner: Send + Sync {
    async fn run(&self, task: TaskManifest) -> Result<TaskResult, TrainError>;
}

/// Max distinct peers the gossip address book will hold — a memory bound
/// against a flood of distinct keys gossiped over many exchanges (A5).
const MAX_ADDR_BOOK: usize = 4096;

/// A symmetric mesh node.
pub struct MeshNode {
    server: Arc<P2pServer>,
    keypair: Arc<KeyPair>,
    node_id: PeerId,
    caps: NodeCapabilities,
    /// Present iff `caps.worker` — how the node runs a dispatched task.
    runner: Option<Arc<dyn MeshTaskRunner>>,
    /// Live inbound connections, keyed by the peer that dialed us.
    connections: Arc<RwLock<HashMap<PeerId, quinn::Connection>>>,
    /// Addresses learned via gossip (A5): peer id → last-advertised address.
    /// Gossip teaches only WHERE a peer is; its identity + keys are confirmed
    /// at connect time by mutual TLS (A1), never asserted in the gossip itself.
    addr_book: Arc<RwLock<HashMap<PeerId, SocketAddr>>>,
    /// Optional local status hub (D5). On an INITIATOR node, forwarded worker
    /// `Status` frames are ingested here so one status.jsonl tells the whole
    /// multi-host story. `None` on a plain worker.
    status_hub: std::sync::Mutex<Option<Arc<crate::framework::status::StatusHub>>>,
}

impl MeshNode {
    /// Bind the node's QUIC server and start its accept loop. A `worker` node
    /// MUST supply a `runner`; a scheduler-only node passes `None`. `allow_legacy`
    /// accepts pre-mutual-auth peers (the A1 escape hatch).
    pub async fn bind(
        addr: SocketAddr,
        keypair: Arc<KeyPair>,
        peers: PeerRegistry,
        caps: NodeCapabilities,
        runner: Option<Arc<dyn MeshTaskRunner>>,
        allow_legacy: bool,
    ) -> Result<Arc<Self>, TrainError> {
        if caps.worker && runner.is_none() {
            return Err(TrainError::other(
                "a worker-capable MeshNode requires a MeshTaskRunner",
            ));
        }
        if !caps.worker && !caps.scheduler {
            return Err(TrainError::other(
                "a MeshNode must have at least one of {worker, scheduler}",
            ));
        }
        let node_id = PeerId::from_pubkey(&keypair.verifying);
        let server = Arc::new(
            P2pServer::bind_with_options(addr, keypair.clone(), peers, allow_legacy).await?,
        );
        let node = Arc::new(Self {
            server,
            keypair,
            node_id,
            caps,
            runner,
            connections: Arc::new(RwLock::new(HashMap::new())),
            addr_book: Arc::new(RwLock::new(HashMap::new())),
            status_hub: std::sync::Mutex::new(None),
        });
        let accept = node.clone();
        tokio::spawn(async move { accept.accept_loop().await });
        Ok(node)
    }

    /// The node's own identity.
    pub fn node_id(&self) -> &PeerId {
        &self.node_id
    }

    /// The address the node is listening on.
    pub fn local_addr(&self) -> Result<SocketAddr, TrainError> {
        self.server.local_addr()
    }

    /// The node's capabilities.
    pub fn capabilities(&self) -> NodeCapabilities {
        self.caps
    }

    /// This node's LOCAL reputation for a peer (0.0–1.0), or `None` if unknown.
    /// Local-only (A6) — never shared over the wire.
    pub async fn peer_reputation(&self, id: &PeerId) -> Option<f64> {
        self.server.peers.read().await.get(id).map(|p| p.reputation)
    }

    /// The address learned for a peer via gossip (A5), if any.
    pub async fn learned_addr(&self, id: &PeerId) -> Option<SocketAddr> {
        self.addr_book.read().await.get(id).copied()
    }

    /// Attach a status hub (D5): forwarded worker `Status` frames are ingested
    /// into it, so an initiator's status.jsonl carries every host's lifecycle.
    pub fn set_status_hub(&self, hub: Arc<crate::framework::status::StatusHub>) {
        *self.status_hub.lock().unwrap() = Some(hub);
    }

    /// Forward one lifecycle status event to a connected initiator (D5), tagged
    /// with this node's short id. The initiator ingests it into its status hub.
    pub async fn forward_status(
        &self,
        conn: &quinn::Connection,
        event: crate::framework::status::StageEvent,
    ) -> Result<(), TrainError> {
        let frame = MeshFrame::Status {
            host: self.node_id.short(),
            event: Box::new(event),
        };
        match mesh_request(conn, &frame).await? {
            MeshFrame::Ack => Ok(()),
            MeshFrame::Error { message } => Err(TrainError::other(message)),
            other => Err(TrainError::other(format!(
                "unexpected status response: {}",
                other.kind()
            ))),
        }
    }

    /// Send a signed peer exchange (A5) to a connected peer. The responder
    /// verifies + learns the advertised addresses and replies `Ack`.
    pub async fn send_gossip(
        &self,
        conn: &quinn::Connection,
        exchange: crate::p2p::gossip::PeerExchange,
    ) -> Result<(), TrainError> {
        match mesh_request(conn, &MeshFrame::PeerExchange(Box::new(exchange))).await? {
            MeshFrame::Ack => Ok(()),
            MeshFrame::Error { message } => Err(TrainError::other(message)),
            other => Err(TrainError::other(format!(
                "unexpected gossip response: {}",
                other.kind()
            ))),
        }
    }

    /// Dial a seed/peer node and return the connection to dispatch over. The
    /// mutual-TLS handshake (A1) binds the peer's identity; `server_pubkey`
    /// pins the peer we intend to reach.
    pub async fn connect_to(
        &self,
        addr: SocketAddr,
        server_pubkey: [u8; 32],
    ) -> Result<quinn::Connection, TrainError> {
        let client = P2pClient::with_coordinator_pin(self.keypair.clone(), server_pubkey);
        let (conn, _peer_id) = client.connect(addr).await?;
        Ok(conn)
    }

    /// Dispatch a task to a connected peer and await its result (the `scheduler`
    /// capability). Refused if this node isn't a scheduler.
    pub async fn dispatch(
        &self,
        conn: &quinn::Connection,
        task: TaskManifest,
    ) -> Result<TaskResult, TrainError> {
        if !self.caps.scheduler {
            return Err(TrainError::other(
                "node lacks the scheduler capability (dispatch refused)",
            ));
        }
        match mesh_request(conn, &MeshFrame::Task(Box::new(task))).await? {
            MeshFrame::Result(r) => Ok(*r),
            MeshFrame::Error { message } => Err(TrainError::other(message)),
            other => Err(TrainError::other(format!(
                "unexpected dispatch response: {}",
                other.kind()
            ))),
        }
    }

    /// Liveness probe to a connected peer.
    pub async fn ping(&self, conn: &quinn::Connection, nonce: u64) -> Result<(), TrainError> {
        match mesh_request(conn, &MeshFrame::Ping { nonce }).await? {
            MeshFrame::Pong { nonce: got } if got == nonce => Ok(()),
            MeshFrame::Pong { nonce: got } => Err(TrainError::other(format!(
                "pong nonce mismatch: sent {nonce}, got {got}"
            ))),
            other => Err(TrainError::other(format!(
                "unexpected ping response: {}",
                other.kind()
            ))),
        }
    }

    async fn accept_loop(self: Arc<Self>) {
        loop {
            match self.server.accept_peer().await {
                Ok((peer_id, conn)) => {
                    self.connections
                        .write()
                        .await
                        .insert(peer_id.clone(), conn.clone());
                    let node = self.clone();
                    tokio::spawn(async move { node.serve_connection(peer_id, conn).await });
                }
                // A per-connection failure (bad TLS handshake, a peer that
                // hung up mid-handshake) must NOT kill the whole node — log and
                // keep accepting. Only a closed/shut-down endpoint is terminal.
                Err(e) if self.server.is_endpoint_closed() => {
                    tracing::debug!("mesh node accept loop stopped (endpoint closed): {e}");
                    break;
                }
                Err(e) => {
                    tracing::warn!("mesh node: dropped a bad inbound connection: {e}");
                    // Guard against a hot spin if the endpoint is closed by
                    // something other than our own `shutdown` (errors would
                    // then return immediately and forever).
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            }
        }
    }

    /// Serve the mesh protocol on one connection until it closes: read a
    /// request frame per bi-stream, handle it, write the response.
    async fn serve_connection(self: Arc<Self>, peer: PeerId, conn: quinn::Connection) {
        // Loops until `accept_request` errors (connection closed / reset).
        while let Ok((frame, mut send)) = accept_request(&conn).await {
            let (response, score) = self.handle_frame(frame).await;
            // Mutual reputation (A6): a worker also scores the SCHEDULER it
            // served, via the same local EMA the scheduler uses on the worker —
            // so "whom to accept work from" uses the same signal as "whom to
            // dispatch to". `score` is None when the outcome isn't the
            // scheduler's doing (e.g. our own missing worker capability).
            // Local-only: no reputation ever crosses the wire (ADR 0079).
            if let Some(ok) = score {
                let mut peers = self.server.peers.write().await;
                peers.update_reputation(&peer, ok);
                if let Err(e) = peers.save() {
                    tracing::warn!("mesh node: persist reputation for {peer} failed: {e}");
                }
            }
            if let Err(e) = write_frame(&mut send, &response).await {
                tracing::debug!("mesh node: reply to {peer} failed: {e}");
            }
        }
        // Only evict the map entry if it is STILL this connection — a peer that
        // reconnected already overwrote it with a live connection, and this
        // (now-dead) task must not remove that.
        let mut conns = self.connections.write().await;
        if conns
            .get(&peer)
            .is_some_and(|c| c.stable_id() == conn.stable_id())
        {
            conns.remove(&peer);
        }
    }

    /// Produce the response frame for one request, plus an optional
    /// scheduler-reputation signal (A6): `Some(true)` = the peer sent us valid
    /// work we served, `Some(false)` = the peer misbehaved, `None` = the
    /// outcome isn't attributable to the peer (our own limitation / a
    /// non-task request). The worker capability gates task execution; a `Task`
    /// to a scheduler-only node is a clean `Error`.
    async fn handle_frame(&self, frame: MeshFrame) -> (MeshFrame, Option<bool>) {
        match frame {
            MeshFrame::Ping { nonce } => (MeshFrame::Pong { nonce }, None),
            MeshFrame::Hello(_) => (
                MeshFrame::HelloAck {
                    peer_id: self.node_id.clone(),
                    // TODO(A4): fold in the registry-derived trust for the
                    // calling peer. Until then the ack reports Anonymous; the
                    // dispatch matrix (fail-closed) is the real gate, so a peer
                    // must NOT treat this as a trust grant.
                    trust: TrustLevel::Anonymous,
                },
                None,
            ),
            MeshFrame::Task(task) => match &self.runner {
                Some(runner) => match runner.run(*task).await {
                    // Served the scheduler's work → a positive signal for it.
                    Ok(result) => (MeshFrame::Result(Box::new(result)), Some(true)),
                    // A runner error is OUR side (stage crash / resource) — not
                    // the scheduler's fault, so don't score it.
                    Err(e) => (
                        MeshFrame::Error {
                            message: format!("task execution failed: {e}"),
                        },
                        None,
                    ),
                },
                // Refused for our OWN missing capability — not the scheduler's
                // fault. (Abuse penalties for unauthorized tasks attach at the
                // A4 trust gate.)
                None => (
                    MeshFrame::Error {
                        message: "node lacks the worker capability (task refused)".into(),
                    },
                    None,
                ),
            },
            // The Ack IS the response (sent by serve_connection's write_frame);
            // None here is the reputation signal, not "no reply".
            MeshFrame::PeerExchange(ex) => (self.handle_peer_exchange(*ex).await, None),
            // D5: ingest a forwarded worker lifecycle event into the local hub
            // (if this node is an initiator with one attached).
            MeshFrame::Status { host, event } => {
                if let Some(hub) = self.status_hub.lock().unwrap().as_ref() {
                    hub.ingest_remote(host, *event);
                }
                (MeshFrame::Ack, None)
            }
            other => (
                MeshFrame::Error {
                    message: format!("unsupported request: {}", other.kind()),
                },
                None,
            ),
        }
    }

    /// A5: verify a gossiped peer exchange against the SENDER's registry key,
    /// then learn each advertised address. Gossip populates only the address
    /// book — identities are confirmed by mutual TLS when the peer is actually
    /// dialed, so a gossip can widen reachability but never confer trust.
    async fn handle_peer_exchange(&self, ex: crate::p2p::gossip::PeerExchange) -> MeshFrame {
        // Bound the message before any work: a single (authenticated but only
        // Anonymous) peer must not flood us in one exchange.
        if !ex.within_size_limit() {
            return MeshFrame::Error {
                message: "peer exchange exceeds the record limit (rejected)".into(),
            };
        }
        // Drop stale replays (issued_at is a finite-window guard).
        if !ex.is_fresh(chrono::Utc::now().timestamp()) {
            return MeshFrame::Error {
                message: "peer exchange is stale or future-dated (rejected)".into(),
            };
        }
        // We can only verify gossip from a peer we already know (its pubkey is
        // in our registry). An unknown sender ⇒ can't verify ⇒ reject.
        let sender_pubkey = {
            let peers = self.server.peers.read().await;
            peers.get(&ex.from).map(|p| p.pubkey)
        };
        let Some(pubkey) = sender_pubkey else {
            return MeshFrame::Error {
                message: "peer exchange from an unknown sender (rejected)".into(),
            };
        };
        if !ex.verify(&pubkey) {
            return MeshFrame::Error {
                message: "peer exchange signature failed to verify (rejected)".into(),
            };
        }
        let mut book = self.addr_book.write().await;
        for record in &ex.records {
            let id = record.peer_id();
            // Never learn our own address back from a peer.
            if id == self.node_id {
                continue;
            }
            // Cap total growth: refresh a known peer's address, but stop taking
            // NEW ids once the book is full (bounds memory against a flood of
            // distinct keys across many exchanges).
            if book.contains_key(&id) || book.len() < MAX_ADDR_BOOK {
                book.insert(id, record.addr);
            }
        }
        MeshFrame::Ack
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;
    use crate::p2p::task::ResourceRequest;
    use crate::p2p::trust::DataClass;

    /// A trivial worker: echoes the task's expected output hash back in a
    /// signed result. Enough to prove the node topology + dispatch path.
    struct EchoRunner {
        keypair: Arc<KeyPair>,
    }

    #[async_trait]
    impl MeshTaskRunner for EchoRunner {
        async fn run(&self, task: TaskManifest) -> Result<TaskResult, TrainError> {
            let content_id = task.expected_content_id.ok_or_else(|| {
                TrainError::other("echo runner requires expected content identity")
            })?;
            let mut result = TaskResult {
                protocol_version: crate::p2p::task::TASK_PROTOCOL_VERSION,
                task_id: task.task_id,
                peer_id: PeerId::from_pubkey(&self.keypair.verifying),
                content_id,
                encrypted_output: None,
                wall_time_ms: 1,
                signature: self.keypair.sign(b"placeholder"),
            };
            result.signature = self.keypair.sign(&result.sign_payload());
            Ok(result)
        }
    }

    async fn spawn_node(
        caps: NodeCapabilities,
        runner: Option<Arc<dyn MeshTaskRunner>>,
    ) -> (Arc<MeshNode>, Arc<KeyPair>) {
        let keypair = Arc::new(KeyPair::generate());
        let dir = tempfile::tempdir().unwrap();
        let peers = PeerRegistry::load(&dir.path().join("peers.json")).unwrap();
        std::mem::forget(dir); // keep registry path alive for the test process
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let node = MeshNode::bind(addr, keypair.clone(), peers, caps, runner, false)
            .await
            .unwrap();
        (node, keypair)
    }

    fn echo_task(from: &PeerId, signer: &KeyPair) -> TaskManifest {
        let mut task = TaskManifest {
            protocol_version: crate::p2p::task::TASK_PROTOCOL_VERSION,
            task_id: "echo-1".into(),
            coordinator_id: from.clone(),
            stage_name: "p2p-echo".into(),
            stage_schema: 1,
            input_content_id: crate::framework::ContentId::from_digest(ContentHash::of_bytes(
                b"in",
            )),
            invocation_key: crate::framework::InvocationKey::from_digest(ContentHash::of_bytes(
                b"echo-invocation",
            )),
            args_hash: ContentHash::of_bytes(b"args"),
            expected_content_id: Some(crate::framework::ContentId::from_digest(
                ContentHash::of_bytes(b"out"),
            )),
            args: serde_json::json!({}),
            resources: ResourceRequest::default(),
            data_class: DataClass::Public,
            timeout_secs: 30,
            encrypted_input: None,
            signature: signer.sign(b"placeholder"),
        };
        task.signature = signer.sign(&task.sign_payload());
        task
    }

    /// The T4.4 gate (loopback slice): a WORKER-ONLY node runs a task
    /// dispatched by a SCHEDULER-ONLY node — any-to-any over the mesh.
    #[tokio::test]
    async fn scheduler_dispatches_to_worker_only_node() {
        // Worker-only node C.
        let c_keypair = Arc::new(KeyPair::generate());
        let (c, c_id) = {
            let runner: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
                keypair: c_keypair.clone(),
            });
            // Rebind with the shared keypair so the runner's signature matches
            // the node identity.
            let dir = tempfile::tempdir().unwrap();
            let peers = PeerRegistry::load(&dir.path().join("peers.json")).unwrap();
            std::mem::forget(dir);
            let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
            let node = MeshNode::bind(
                addr,
                c_keypair.clone(),
                peers,
                NodeCapabilities {
                    worker: true,
                    scheduler: false,
                },
                Some(runner),
                false,
            )
            .await
            .unwrap();
            let id = node.node_id().clone();
            (node, id)
        };

        // Scheduler-only node A.
        let (a, _a_kp) = spawn_node(
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
        )
        .await;

        // A dials C and dispatches the echo task.
        let conn = a
            .connect_to(c.local_addr().unwrap(), c_keypair.verifying.to_bytes())
            .await
            .expect("A connects to C");
        let task = echo_task(a.node_id(), &_a_kp);
        let expected_out = task.expected_content_id.unwrap();
        let result =
            tokio::time::timeout(std::time::Duration::from_secs(10), a.dispatch(&conn, task))
                .await
                .expect("dispatch must not hang")
                .expect("worker executes the dispatched task");

        assert_eq!(result.content_id, expected_out, "worker echoed the output");
        assert_eq!(result.peer_id, c_id, "result came from worker C");
        // The result is signed by C — verify it.
        assert!(
            crate::p2p::crypto::verify(
                &c_keypair.verifying,
                &result.sign_payload(),
                &result.signature
            ),
            "worker's result signature verifies"
        );
    }

    /// A scheduler-only node refuses a Task (no worker capability) with a clean
    /// error, not a panic or hang.
    #[tokio::test]
    async fn worker_capability_gates_task_execution() {
        let (server, server_kp) = spawn_node(
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
        )
        .await;
        let (client, client_kp) = spawn_node(NodeCapabilities::default(), {
            let r: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
                keypair: Arc::new(KeyPair::generate()),
            });
            Some(r)
        })
        .await;

        let conn = client
            .connect_to(server.local_addr().unwrap(), server_kp.verifying.to_bytes())
            .await
            .unwrap();
        let task = echo_task(client.node_id(), &client_kp);
        let err = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            client.dispatch(&conn, task),
        )
        .await
        .expect("must not hang")
        .expect_err("scheduler-only server has no worker capability");
        assert!(
            format!("{err}").contains("worker capability"),
            "clean capability error, got: {err}"
        );
    }

    /// A6: after serving a scheduler's task, the WORKER scores that scheduler
    /// (mutual, local-only). C starts A at the 0.5 prior; a served task nudges
    /// it up.
    #[tokio::test]
    async fn worker_scores_scheduler_after_serving() {
        let c_keypair = Arc::new(KeyPair::generate());
        let dir = tempfile::tempdir().unwrap();
        let peers = PeerRegistry::load(&dir.path().join("peers.json")).unwrap();
        std::mem::forget(dir);
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let runner: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
            keypair: c_keypair.clone(),
        });
        let c = MeshNode::bind(
            addr,
            c_keypair.clone(),
            peers,
            NodeCapabilities {
                worker: true,
                scheduler: false,
            },
            Some(runner),
            false,
        )
        .await
        .unwrap();

        let (a, a_kp) = spawn_node(
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
        )
        .await;
        let a_id = a.node_id().clone();

        let conn = a
            .connect_to(c.local_addr().unwrap(), c_keypair.verifying.to_bytes())
            .await
            .unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            a.dispatch(&conn, echo_task(&a_id, &a_kp)),
        )
        .await
        .expect("no hang")
        .expect("served");

        // Give C's serve loop a moment to record the outcome after replying.
        let mut rep = None;
        for _ in 0..50 {
            rep = c.peer_reputation(&a_id).await;
            if rep.is_some_and(|r| r > 0.5) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            rep.is_some_and(|r| r > 0.5),
            "worker C should have scored scheduler A up from the 0.5 prior, got {rep:?}"
        );
    }

    /// A5: a node verifies a gossiped peer exchange from a KNOWN sender and
    /// learns the advertised addresses; a forged/unknown-sender exchange is
    /// rejected and teaches nothing.
    #[tokio::test]
    async fn gossip_learns_addresses_from_known_sender_only() {
        use crate::p2p::gossip::{PeerExchange, PeerRecord};

        let (server, server_kp) = spawn_node(NodeCapabilities::default(), {
            let r: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
                keypair: Arc::new(KeyPair::generate()),
            });
            Some(r)
        })
        .await;
        // Client A dials the server, so the server registers A (mutual TLS) and
        // can therefore verify A's gossip.
        let (client, client_kp) = spawn_node(
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
        )
        .await;
        let conn = client
            .connect_to(server.local_addr().unwrap(), server_kp.verifying.to_bytes())
            .await
            .unwrap();

        // A advertises a (fabricated) peer's address.
        let advertised_kp = KeyPair::generate();
        let advertised_id = PeerId::from_pubkey(&advertised_kp.verifying);
        let record = PeerRecord {
            pubkey: advertised_kp.verifying,
            addr: "203.0.113.7:9100".parse().unwrap(),
        };
        let now = chrono::Utc::now().timestamp();
        let ex = PeerExchange::create(&client_kp, now, vec![record]);
        client
            .send_gossip(&conn, ex)
            .await
            .expect("valid gossip accepted");
        assert_eq!(
            server.learned_addr(&advertised_id).await,
            Some("203.0.113.7:9100".parse().unwrap()),
            "server learned the gossiped address"
        );

        // A forged exchange (signed by A but CLAIMING to be from a stranger)
        // is rejected — the server verifies `from` against its registry.
        let stranger = KeyPair::generate();
        let mut forged = PeerExchange::create(&client_kp, now, vec![]);
        forged.from = PeerId::from_pubkey(&stranger.verifying); // lie about origin
        let err = client.send_gossip(&conn, forged).await.unwrap_err();
        assert!(
            format!("{err}").contains("unknown sender") || format!("{err}").contains("verify"),
            "forged-origin gossip rejected, got: {err}"
        );
    }

    /// D5: a worker forwards a lifecycle event to an initiator over the mesh;
    /// the initiator ingests it (host-tagged) into its status hub → its
    /// status.jsonl carries the worker's event.
    #[tokio::test]
    async fn worker_forwards_status_to_initiator() {
        use crate::framework::artifact::ContentHash;
        use crate::framework::status::{StageEvent, StatusHub, spawn_status_writer};

        // Initiator node with a status hub + writer.
        let (initiator, initiator_kp) = spawn_node(NodeCapabilities::default(), {
            let r: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
                keypair: Arc::new(KeyPair::generate()),
            });
            Some(r)
        })
        .await;
        let (hub, lifecycle_rx) = StatusHub::new();
        let hub = hub.with_host("initiator");
        let td = tempfile::tempdir().unwrap();
        let writer = spawn_status_writer(&hub, lifecycle_rx, td.path()).unwrap();
        initiator.set_status_hub(hub.clone());

        // Worker dials the initiator and forwards a lifecycle event.
        let (worker, _wkp) = spawn_node(
            NodeCapabilities {
                worker: true,
                scheduler: false,
            },
            {
                let r: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
                    keypair: Arc::new(KeyPair::generate()),
                });
                Some(r)
            },
        )
        .await;
        let worker_host = worker.node_id().short();
        let conn = worker
            .connect_to(
                initiator.local_addr().unwrap(),
                initiator_kp.verifying.to_bytes(),
            )
            .await
            .unwrap();
        worker
            .forward_status(
                &conn,
                StageEvent::StageEnd {
                    node_idx: 3,
                    stage_name: "worker-stage".into(),
                    content_id: Some(crate::framework::ContentId::from_digest(
                        ContentHash::of_bytes(b"o"),
                    )),
                    legacy_output_hash: None,
                    elapsed: std::time::Duration::from_millis(2),
                },
            )
            .await
            .expect("forward_status");

        // Flush the writer (drop the hub) and read the initiator's status.jsonl.
        drop(hub);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), writer).await;
        let body = std::fs::read_to_string(td.path().join("status.jsonl")).unwrap();
        assert!(
            body.contains(&format!("\"host\":\"{worker_host}\"")) && body.contains("worker-stage"),
            "initiator status.jsonl carries the worker's host-tagged event: {body}"
        );
    }

    #[tokio::test]
    async fn ping_pong_over_node() {
        let (server, server_kp) = spawn_node(NodeCapabilities::default(), {
            let r: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
                keypair: Arc::new(KeyPair::generate()),
            });
            Some(r)
        })
        .await;
        let (client, _kp) = spawn_node(
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
        )
        .await;
        let conn = client
            .connect_to(server.local_addr().unwrap(), server_kp.verifying.to_bytes())
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(10), client.ping(&conn, 7))
            .await
            .expect("ping must not hang")
            .expect("pong");
    }

    #[tokio::test]
    async fn worker_without_runner_is_rejected() {
        let keypair = Arc::new(KeyPair::generate());
        let dir = tempfile::tempdir().unwrap();
        let peers = PeerRegistry::load(&dir.path().join("peers.json")).unwrap();
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        match MeshNode::bind(
            addr,
            keypair,
            peers,
            NodeCapabilities::default(),
            None,
            false,
        )
        .await
        {
            Err(e) => assert!(
                format!("{e}").contains("requires a MeshTaskRunner"),
                "wrong error: {e}"
            ),
            Ok(_) => panic!("worker node without a runner must be refused"),
        }
    }
}
