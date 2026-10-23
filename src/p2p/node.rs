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
//! A node serves the symmetric bi-stream protocol (A2) on EVERY connection it
//! holds, inbound (mutual-TLS-authenticated, A1) or dialed by it: `Ping`→`Pong`,
//! `Hello`→`HelloAck`, `PeerExchange`→`Ack` (A5), and — with the `worker`
//! capability — `Task`→run it and reply `Result`. Streams that start with
//! [`OBJECT_STREAM_TAG`] are the object
//! plane ([`crate::p2p::objects`]): a peer fetching stage data it was granted.
//! Each stream is served in its own task, so a long task does not block a
//! gossip exchange or an object fetch on the same connection.
//!
//! **Gossip (A5).** With [`MeshNode::start_gossip`], the node pushes a signed
//! [`PeerExchange`] to every connected peer on connect and every
//! [`GossipConfig::interval`], learns the peers its neighbours advertise, and
//! dials them (bounded, with backoff). Gossip never changes trust: learned
//! peers are not written to the peer registry, and every dial pins the
//! advertised key, so an address learned from gossip can only ever reach the
//! identity it was advertised with.
//!
//! Task EXECUTION is pluggable via [`MeshTaskRunner`] so the node core is
//! testable without the full stage-execution stack; the production runner is
//! [`SharedCacheRunner`](crate::p2p::mesh_runner::SharedCacheRunner).

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use tokio::sync::RwLock;

use crate::error::TrainError;
use crate::p2p::crypto::KeyPair;
use crate::p2p::gossip::{MAX_RECORDS_PER_EXCHANGE, PeerExchange, PeerRecord};
use crate::p2p::mesh_wire::{MeshFrame, mesh_request, read_frame_after_version, write_frame};
use crate::p2p::objects::{MeshObjects, OBJECT_STREAM_TAG};
use crate::p2p::peer::PeerId;
use crate::p2p::registry::PeerRegistry;
use crate::p2p::task::{TaskManifest, TaskResult};
use crate::p2p::transport::{P2pClient, P2pServer, tls_authenticated_pubkey};
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

/// Where a dispatched task came from: the authenticated peer and the
/// connection it arrived on. A runner that lacks the task's input fetches it
/// from this peer over this connection ([`crate::p2p::objects`]).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TaskOrigin {
    /// The TLS-authenticated peer that sent the task.
    pub peer: PeerId,
    /// The connection the task arrived on.
    pub conn: quinn::Connection,
}

impl TaskOrigin {
    pub fn new(peer: PeerId, conn: quinn::Connection) -> Self {
        Self { peer, conn }
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

    /// Run a task knowing where it came from. The node calls this; the default
    /// ignores the origin and calls [`run`](Self::run). A runner that can fetch
    /// missing inputs from the dispatching peer overrides it.
    async fn run_from(
        &self,
        task: TaskManifest,
        origin: &TaskOrigin,
    ) -> Result<TaskResult, TrainError> {
        let _ = origin;
        self.run(task).await
    }
}

/// Max distinct peers the gossip address book will hold — a memory bound
/// against a flood of distinct keys gossiped over many exchanges (A5).
const MAX_ADDR_BOOK: usize = 4096;

/// Concurrent bi-streams one connection may have in service at once. QUIC's
/// own stream limit bounds what is open; this bounds what is being worked on.
const MAX_STREAMS_IN_SERVICE: usize = 16;

/// How long a peer may take to send the first byte of a request once it has
/// opened a stream.
const REQUEST_HEADER_TIMEOUT: Duration = Duration::from_secs(30);

/// How long a peer may take to send the rest of a request frame.
const REQUEST_BODY_TIMEOUT: Duration = Duration::from_secs(300);

/// Inbound handshakes completed concurrently.
const MAX_HANDSHAKES_IN_FLIGHT: usize = 64;

/// Bound on one gossip push (open stream, send, read the ack).
const GOSSIP_PUSH_TIMEOUT: Duration = Duration::from_secs(15);

/// Token bucket for inbound exchanges per peer: burst, then one per second.
const GOSSIP_BURST: f64 = 8.0;

/// How a node runs its gossip (A5). Defaults suit a WAN mesh; tests shorten
/// the interval.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct GossipConfig {
    /// How often to push the node's view to every connected peer (it is also
    /// pushed to each new connection immediately).
    pub interval: Duration,
    /// Stop dialing learned peers once this many are connected (in either
    /// direction).
    pub max_peers: usize,
    /// Bound on one dial (QUIC + mutual TLS + handshake).
    pub dial_timeout: Duration,
    /// Delay before redialing a peer after its first failed dial; doubles per
    /// consecutive failure up to `backoff_max`.
    pub backoff_base: Duration,
    /// Ceiling on the redial delay.
    pub backoff_max: Duration,
    /// Dials in flight at once.
    pub max_concurrent_dials: usize,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            max_peers: 32,
            dial_timeout: Duration::from_secs(10),
            backoff_base: Duration::from_secs(5),
            backoff_max: Duration::from_secs(600),
            max_concurrent_dials: 8,
        }
    }
}

impl GossipConfig {
    /// Set [`interval`](Self::interval).
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Set [`max_peers`](Self::max_peers).
    pub fn with_max_peers(mut self, max_peers: usize) -> Self {
        self.max_peers = max_peers;
        self
    }

    /// Set [`dial_timeout`](Self::dial_timeout).
    pub fn with_dial_timeout(mut self, dial_timeout: Duration) -> Self {
        self.dial_timeout = dial_timeout;
        self
    }

    /// Set the redial backoff: `base` after the first failure, doubling, at
    /// most `max`.
    pub fn with_backoff(mut self, base: Duration, max: Duration) -> Self {
        self.backoff_base = base;
        self.backoff_max = max;
        self
    }

    /// The redial delay after `failures` consecutive failed dials.
    pub fn backoff(&self, failures: u32) -> Duration {
        if failures == 0 {
            return Duration::ZERO;
        }
        let factor = 1u32
            .checked_shl(failures.saturating_sub(1))
            .unwrap_or(u32::MAX);
        self.backoff_base
            .checked_mul(factor)
            .unwrap_or(self.backoff_max)
            .min(self.backoff_max)
    }
}

/// How the node came to know a peer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PeerSource {
    /// Given on the command line (`--seed`).
    Seed,
    /// Dialed directly by this node's caller.
    Dialed,
    /// Learned from a neighbour's signed peer exchange.
    Gossip,
    /// Connected to us; known only from that connection.
    Inbound,
}

impl PeerSource {
    pub fn label(self) -> &'static str {
        match self {
            PeerSource::Seed => "seed",
            PeerSource::Dialed => "dialed",
            PeerSource::Gossip => "gossip",
            PeerSource::Inbound => "inbound",
        }
    }
}

/// One row of [`MeshNode::mesh_view`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MeshPeerView {
    pub id: PeerId,
    /// The address the node would dial, if it knows one.
    pub addr: Option<SocketAddr>,
    pub source: PeerSource,
    /// Whether a live connection exists (either direction).
    pub connected: bool,
    /// Consecutive failed dials.
    pub failures: u32,
}

/// One address-book entry: who (the key every dial pins) and where.
#[derive(Clone, Debug)]
struct BookEntry {
    pubkey: VerifyingKey,
    addr: SocketAddr,
    source: PeerSource,
    /// This node has connected to `addr` and the pinned key answered.
    confirmed: bool,
    /// The address came from the peer's OWN signed record (not a third party).
    self_advertised: bool,
    failures: u32,
    next_dial: Option<Instant>,
}

/// Per-sender gossip receive state: replay order + rate limit.
#[derive(Clone, Debug)]
struct GossipRx {
    last_issued_at: i64,
    tokens: f64,
    refilled: Instant,
}

/// Whether `addr` is worth dialing: a concrete unicast address and port.
fn dialable(addr: &SocketAddr) -> bool {
    let ip = addr.ip();
    let broadcast = matches!(ip, std::net::IpAddr::V4(v4) if v4.is_broadcast());
    addr.port() != 0 && !ip.is_unspecified() && !ip.is_multicast() && !broadcast
}

/// A symmetric mesh node.
pub struct MeshNode {
    server: Arc<P2pServer>,
    keypair: Arc<KeyPair>,
    node_id: PeerId,
    caps: NodeCapabilities,
    /// Present iff `caps.worker` — how the node runs a dispatched task.
    runner: Option<Arc<dyn MeshTaskRunner>>,
    /// Live connections in either direction, keyed by the peer's
    /// TLS-authenticated id.
    connections: Arc<RwLock<HashMap<PeerId, quinn::Connection>>>,
    /// Peers this node can dial: id → pinned key + address (A5). Gossip
    /// teaches only WHERE a peer is; the key in each entry is what every dial
    /// pins, so a learned address can only reach the advertised identity.
    addr_book: Arc<RwLock<HashMap<PeerId, BookEntry>>>,
    /// Optional local status hub (D5). On an INITIATOR node, forwarded worker
    /// `Status` frames are ingested here so one status.jsonl tells the whole
    /// multi-host story. `None` on a plain worker.
    status_hub: std::sync::Mutex<Option<Arc<crate::framework::status::StatusHub>>>,
    /// Object plane: what this node serves to (and fetches from) peers.
    objects: std::sync::Mutex<Option<Arc<MeshObjects>>>,
    /// The address advertised in this node's own gossip record.
    advertise: std::sync::Mutex<Option<SocketAddr>>,
    /// Some once gossip runs.
    gossip: std::sync::Mutex<Option<GossipConfig>>,
    gossip_rx: std::sync::Mutex<HashMap<PeerId, GossipRx>>,
    dialing: std::sync::Mutex<HashSet<PeerId>>,
    wake: tokio::sync::Notify,
    me: Weak<MeshNode>,
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
        let node = Arc::new_cyclic(|me| Self {
            server,
            keypair,
            node_id,
            caps,
            runner,
            connections: Arc::new(RwLock::new(HashMap::new())),
            addr_book: Arc::new(RwLock::new(HashMap::new())),
            status_hub: std::sync::Mutex::new(None),
            objects: std::sync::Mutex::new(None),
            advertise: std::sync::Mutex::new(None),
            gossip: std::sync::Mutex::new(None),
            gossip_rx: std::sync::Mutex::new(HashMap::new()),
            dialing: std::sync::Mutex::new(HashSet::new()),
            wake: tokio::sync::Notify::new(),
            me: me.clone(),
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

    /// This node's registry trust for a peer, or `None` if it is not in the
    /// registry. Gossip never writes the registry, so learning a peer by
    /// gossip leaves this `None`.
    pub async fn peer_trust(&self, id: &PeerId) -> Option<TrustLevel> {
        self.server.peers.read().await.get(id).map(|p| p.trust)
    }

    /// The address known for a peer (from a seed, a dial, or gossip, A5).
    pub async fn learned_addr(&self, id: &PeerId) -> Option<SocketAddr> {
        self.addr_book.read().await.get(id).map(|e| e.addr)
    }

    /// The Ed25519 key this node pins for a peer: from its address book, or
    /// the TLS identity of a live connection.
    pub async fn peer_pubkey(&self, id: &PeerId) -> Option<VerifyingKey> {
        if let Some(e) = self.addr_book.read().await.get(id) {
            return Some(e.pubkey);
        }
        let conn = self.connections.read().await.get(id).cloned()?;
        let key = tls_authenticated_pubkey(&conn)?;
        VerifyingKey::from_bytes(&key).ok()
    }

    /// A live connection to `id`, if there is one (either direction).
    pub async fn connection_to(&self, id: &PeerId) -> Option<quinn::Connection> {
        self.connections
            .read()
            .await
            .get(id)
            .filter(|c| c.close_reason().is_none())
            .cloned()
    }

    /// Ids of every peer with a live connection.
    pub async fn connected_peers(&self) -> Vec<PeerId> {
        self.connections
            .read()
            .await
            .iter()
            .filter(|(_, c)| c.close_reason().is_none())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Every peer this node knows: address-book entries plus live connections
    /// it has no address for. Sorted by id.
    pub async fn mesh_view(&self) -> Vec<MeshPeerView> {
        let connected: HashSet<PeerId> = self.connected_peers().await.into_iter().collect();
        let book = self.addr_book.read().await;
        let mut out: Vec<MeshPeerView> = book
            .iter()
            .map(|(id, e)| MeshPeerView {
                id: id.clone(),
                addr: Some(e.addr),
                source: e.source,
                connected: connected.contains(id),
                failures: e.failures,
            })
            .collect();
        for id in &connected {
            if !book.contains_key(id) {
                out.push(MeshPeerView {
                    id: id.clone(),
                    addr: None,
                    source: PeerSource::Inbound,
                    connected: true,
                    failures: 0,
                });
            }
        }
        out.sort_by(|a, b| a.id.0.cmp(&b.id.0));
        out
    }

    /// Attach a status hub (D5): forwarded worker `Status` frames are ingested
    /// into it, so an initiator's status.jsonl carries every host's lifecycle.
    pub fn set_status_hub(&self, hub: Arc<crate::framework::status::StatusHub>) {
        *self.status_hub.lock().unwrap() = Some(hub);
    }

    /// Attach the object plane: the node then serves granted objects to peers
    /// and grants each task's output to the peer that dispatched it.
    pub fn set_objects(&self, objects: Arc<MeshObjects>) {
        *self.objects.lock().unwrap_or_else(|p| p.into_inner()) = Some(objects);
    }

    /// The attached object plane, if any.
    pub fn objects(&self) -> Option<Arc<MeshObjects>> {
        self.objects
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Set the address this node advertises for itself in gossip. Without it
    /// the node advertises its listen address; a wildcard listen address
    /// (`0.0.0.0` / `::`) is sent as-is and each receiver substitutes the IP it
    /// observes on the connection. Behind NAT or a port forward whose public
    /// port differs from the listen port, set this to the public address.
    pub fn set_advertise_addr(&self, addr: SocketAddr) {
        *self.advertise.lock().unwrap_or_else(|p| p.into_inner()) = Some(addr);
    }

    /// Remember a seed peer: it is dialed (and redialed after a drop) by the
    /// gossip loop, and advertised once connected.
    pub async fn add_seed(&self, addr: SocketAddr, pubkey: [u8; 32]) -> Result<PeerId, TrainError> {
        let key = VerifyingKey::from_bytes(&pubkey)
            .map_err(|e| TrainError::other(format!("seed pubkey: {e}")))?;
        let id = PeerId::from_pubkey(&key);
        if id == self.node_id {
            return Err(TrainError::other("a node cannot seed itself"));
        }
        let mut book = self.addr_book.write().await;
        let entry = book.entry(id.clone()).or_insert(BookEntry {
            pubkey: key,
            addr,
            source: PeerSource::Seed,
            confirmed: false,
            self_advertised: false,
            failures: 0,
            next_dial: None,
        });
        entry.source = PeerSource::Seed;
        entry.addr = addr;
        Ok(id)
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
        exchange: PeerExchange,
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

    /// Build this node's signed peer exchange: its own record, then every
    /// connected peer whose identity TLS verified and whose address is either
    /// confirmed by a dial or the peer's own signed claim. Addresses one
    /// neighbour told us about a third peer are not re-advertised until this
    /// node has reached that peer itself.
    pub async fn build_exchange(&self) -> PeerExchange {
        let mut records = Vec::new();
        if let Some(own) = self.own_record() {
            records.push(own);
        }
        let connected = self.connected_peers().await;
        let book = self.addr_book.read().await;
        for id in connected {
            if records.len() >= MAX_RECORDS_PER_EXCHANGE {
                break;
            }
            if let Some(e) = book.get(&id)
                && (e.confirmed || e.self_advertised)
                && dialable(&e.addr)
            {
                records.push(PeerRecord {
                    pubkey: e.pubkey,
                    addr: e.addr,
                });
            }
        }
        PeerExchange::create(&self.keypair, chrono::Utc::now().timestamp(), records)
    }

    fn own_record(&self) -> Option<PeerRecord> {
        let addr = (*self.advertise.lock().unwrap_or_else(|p| p.into_inner()))
            .or_else(|| self.local_addr().ok())?;
        (addr.port() != 0).then(|| PeerRecord {
            pubkey: self.keypair.verifying,
            addr,
        })
    }

    /// Dial a seed/peer node and return the connection to dispatch over. The
    /// mutual-TLS handshake (A1) binds the peer's identity; `server_pubkey`
    /// pins the peer we intend to reach. The node then serves the mesh
    /// protocol on the new connection too (so the peer can fetch objects and
    /// gossip back over it), and, if gossip runs, pushes its view.
    pub async fn connect_to(
        &self,
        addr: SocketAddr,
        server_pubkey: [u8; 32],
    ) -> Result<quinn::Connection, TrainError> {
        let key = VerifyingKey::from_bytes(&server_pubkey)
            .map_err(|e| TrainError::other(format!("peer pubkey: {e}")))?;
        let id = PeerId::from_pubkey(&key);
        if id == self.node_id {
            return Err(TrainError::other("refusing to dial this node's own key"));
        }
        // Mark the dial in flight so the gossip loop does not dial the same
        // peer in parallel (it skips ids in `dialing`). Only the call that
        // inserted the mark removes it.
        let marked = self
            .dialing
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id.clone());
        let dialed = self.dial(addr, id.clone(), key, server_pubkey).await;
        if marked {
            self.dialing
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remove(&id);
        }
        dialed
    }

    async fn dial(
        &self,
        addr: SocketAddr,
        id: PeerId,
        key: VerifyingKey,
        server_pubkey: [u8; 32],
    ) -> Result<quinn::Connection, TrainError> {
        let client = P2pClient::with_coordinator_pin(self.keypair.clone(), server_pubkey);
        let (conn, _peer_id) = client.connect(addr).await?;
        {
            let mut book = self.addr_book.write().await;
            let entry = book.entry(id.clone()).or_insert(BookEntry {
                pubkey: key,
                addr,
                source: PeerSource::Dialed,
                confirmed: false,
                self_advertised: false,
                failures: 0,
                next_dial: None,
            });
            entry.addr = addr;
            entry.confirmed = true;
            entry.failures = 0;
            entry.next_dial = None;
        }
        self.adopt_connection(id, conn.clone()).await;
        Ok(conn)
    }

    /// Track a live connection, serve the mesh protocol on it, and push gossip
    /// to it if gossip runs.
    async fn adopt_connection(&self, peer: PeerId, conn: quinn::Connection) {
        self.connections
            .write()
            .await
            .insert(peer.clone(), conn.clone());
        let Some(node) = self.me.upgrade() else {
            return;
        };
        let serve = node.clone();
        let (p, c) = (peer.clone(), conn.clone());
        tokio::spawn(async move { serve.serve_connection(p, c).await });
        if self.gossip_running() {
            tokio::spawn(async move { node.push_gossip(&peer, &conn).await });
        }
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

    // ── gossip (A5) ──────────────────────────────────────────────────────────

    fn gossip_running(&self) -> bool {
        self.gossip
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some()
    }

    /// Start gossip: push this node's view to each new connection and every
    /// `cfg.interval`, and dial learned peers within `cfg`'s bounds. Returns
    /// false (and changes nothing) if gossip already runs.
    pub fn start_gossip(&self, cfg: GossipConfig) -> bool {
        {
            let mut g = self.gossip.lock().unwrap_or_else(|p| p.into_inner());
            if g.is_some() {
                return false;
            }
            *g = Some(cfg.clone());
        }
        let weak = self.me.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(cfg.interval.max(Duration::from_millis(50)));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                let wake = match weak.upgrade() {
                    Some(node) => node,
                    None => break,
                };
                let periodic = tokio::select! {
                    _ = tick.tick() => true,
                    _ = wake.wake.notified() => false,
                };
                if wake.server.is_endpoint_closed() {
                    break;
                }
                wake.dial_candidates(&cfg).await;
                if periodic {
                    wake.push_gossip_all().await;
                }
            }
        });
        // Push to connections that predate gossip.
        if let Some(node) = self.me.upgrade() {
            tokio::spawn(async move { node.push_gossip_all().await });
        }
        true
    }

    async fn push_gossip_all(&self) {
        let conns: Vec<(PeerId, quinn::Connection)> = self
            .connections
            .read()
            .await
            .iter()
            .map(|(id, c)| (id.clone(), c.clone()))
            .collect();
        let pushes = conns.iter().map(|(id, conn)| self.push_gossip(id, conn));
        futures::future::join_all(pushes).await;
    }

    async fn push_gossip(&self, peer: &PeerId, conn: &quinn::Connection) {
        let ex = self.build_exchange().await;
        match tokio::time::timeout(GOSSIP_PUSH_TIMEOUT, self.send_gossip(conn, ex)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::debug!("gossip to {peer}: {e}"),
            Err(_) => tracing::debug!("gossip to {peer}: no ack within {GOSSIP_PUSH_TIMEOUT:?}"),
        }
    }

    /// Dial address-book peers that are not connected, not backing off, and
    /// not already being dialed — up to `cfg.max_peers` connections and
    /// `cfg.max_concurrent_dials` dials in flight. Seeds go first.
    async fn dial_candidates(&self, cfg: &GossipConfig) {
        let connected: HashSet<PeerId> = self.connected_peers().await.into_iter().collect();
        if connected.len() >= cfg.max_peers {
            return;
        }
        let now = Instant::now();
        let picks: Vec<(PeerId, SocketAddr, [u8; 32])> = {
            let book = self.addr_book.read().await;
            let mut dialing = self.dialing.lock().unwrap_or_else(|p| p.into_inner());
            let room = cfg
                .max_peers
                .saturating_sub(connected.len() + dialing.len())
                .min(cfg.max_concurrent_dials.saturating_sub(dialing.len()));
            let mut cands: Vec<(&PeerId, &BookEntry)> = book
                .iter()
                .filter(|(id, e)| {
                    **id != self.node_id
                        && !connected.contains(*id)
                        && !dialing.contains(*id)
                        && dialable(&e.addr)
                        && e.next_dial.is_none_or(|t| t <= now)
                })
                .collect();
            cands.sort_by_key(|(_, e)| (e.source != PeerSource::Seed, e.failures));
            let picks: Vec<_> = cands
                .into_iter()
                .take(room)
                .map(|(id, e)| (id.clone(), e.addr, e.pubkey.to_bytes()))
                .collect();
            for (id, _, _) in &picks {
                dialing.insert(id.clone());
            }
            picks
        };
        for (id, addr, pubkey) in picks {
            let Some(node) = self.me.upgrade() else {
                return;
            };
            let cfg = cfg.clone();
            tokio::spawn(async move {
                let result = match VerifyingKey::from_bytes(&pubkey) {
                    Ok(key) => {
                        tokio::time::timeout(
                            cfg.dial_timeout,
                            node.dial(addr, id.clone(), key, pubkey),
                        )
                        .await
                    }
                    Err(e) => Ok(Err(TrainError::other(format!("peer pubkey: {e}")))),
                };
                let ok = matches!(result, Ok(Ok(_)));
                match result {
                    Ok(Ok(_)) => tracing::info!("mesh: connected to {id} at {addr}"),
                    Ok(Err(e)) => tracing::debug!("mesh: dial {id} at {addr} failed: {e}"),
                    Err(_) => tracing::debug!("mesh: dial {id} at {addr} timed out"),
                }
                if !ok {
                    let mut book = node.addr_book.write().await;
                    if let Some(e) = book.get_mut(&id) {
                        e.failures = e.failures.saturating_add(1);
                        e.next_dial = Some(Instant::now() + cfg.backoff(e.failures));
                    }
                }
                node.dialing
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .remove(&id);
            });
        }
    }

    // ── serving ──────────────────────────────────────────────────────────────

    /// Accept connection attempts and complete each handshake in its own task
    /// (at most [`MAX_HANDSHAKES_IN_FLIGHT`] at once), so a slow or stalled
    /// handshake never holds up the next peer.
    async fn accept_loop(self: Arc<Self>) {
        let in_flight = Arc::new(tokio::sync::Semaphore::new(MAX_HANDSHAKES_IN_FLIGHT));
        loop {
            let incoming = match self.server.accept_incoming().await {
                Ok(incoming) => incoming,
                Err(e) => {
                    tracing::debug!("mesh node accept loop stopped (endpoint closed): {e}");
                    break;
                }
            };
            let Ok(permit) = in_flight.clone().acquire_owned().await else {
                break;
            };
            let node = self.clone();
            tokio::spawn(async move {
                let _permit = permit;
                match node.server.complete_incoming(incoming).await {
                    Ok((peer_id, conn)) => node.adopt_connection(peer_id, conn).await,
                    // A per-connection failure (bad TLS handshake, a peer that
                    // hung up mid-handshake) must NOT affect the node — log it.
                    Err(e) => tracing::warn!("mesh node: dropped a bad inbound connection: {e}"),
                }
            });
        }
    }

    /// Serve the mesh protocol on one connection until it closes: each
    /// bi-stream the peer opens is served in its own task, at most
    /// [`MAX_STREAMS_IN_SERVICE`] at a time.
    async fn serve_connection(self: Arc<Self>, peer: PeerId, conn: quinn::Connection) {
        let in_service = Arc::new(tokio::sync::Semaphore::new(MAX_STREAMS_IN_SERVICE));
        while let Ok((send, recv)) = conn.accept_bi().await {
            let Ok(permit) = in_service.clone().acquire_owned().await else {
                break;
            };
            let node = self.clone();
            let (p, c) = (peer.clone(), conn.clone());
            tokio::spawn(async move {
                let _permit = permit;
                node.serve_stream(p, c, send, recv).await;
            });
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

    /// Serve one request stream: route on its first byte (object plane vs a
    /// mesh frame), handle it, reply.
    async fn serve_stream(
        &self,
        peer: PeerId,
        conn: quinn::Connection,
        mut send: quinn::SendStream,
        mut recv: quinn::RecvStream,
    ) {
        let mut first = [0u8; 1];
        match tokio::time::timeout(REQUEST_HEADER_TIMEOUT, recv.read_exact(&mut first)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                tracing::debug!("mesh node: stream from {peer} ended before a request: {e}");
                return;
            }
            Err(_) => {
                tracing::debug!("mesh node: stream from {peer} sent nothing; dropped");
                return;
            }
        }
        if first[0] == OBJECT_STREAM_TAG {
            let result = match self.objects() {
                Some(objects) => objects.serve_stream(&peer, send, recv).await,
                None => crate::p2p::objects::refuse(&mut send, "this node serves no objects").await,
            };
            if let Err(e) = result {
                tracing::debug!("mesh node: object stream for {peer}: {e}");
            }
            return;
        }
        let frame = match tokio::time::timeout(
            REQUEST_BODY_TIMEOUT,
            read_frame_after_version(first[0], &mut recv),
        )
        .await
        {
            Ok(Ok(frame)) => frame,
            // A malformed or too-new frame fails this stream, not the whole
            // connection: the peer gets an error and may keep using it.
            Ok(Err(e)) => {
                let reply = MeshFrame::Error {
                    message: format!("bad request: {e}"),
                };
                let _ = write_frame(&mut send, &reply).await;
                return;
            }
            Err(_) => {
                tracing::debug!("mesh node: request from {peer} did not arrive in time");
                return;
            }
        };
        let (response, score) = self.handle_frame(&peer, &conn, frame).await;
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

    /// Produce the response frame for one request, plus an optional
    /// scheduler-reputation signal (A6): `Some(true)` = the peer sent us valid
    /// work we served, `Some(false)` = the peer misbehaved, `None` = the
    /// outcome isn't attributable to the peer (our own limitation / a
    /// non-task request). The worker capability gates task execution; a `Task`
    /// to a scheduler-only node is a clean `Error`.
    async fn handle_frame(
        &self,
        peer: &PeerId,
        conn: &quinn::Connection,
        frame: MeshFrame,
    ) -> (MeshFrame, Option<bool>) {
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
                Some(runner) => {
                    let origin = TaskOrigin::new(peer.clone(), conn.clone());
                    match runner.run_from(*task, &origin).await {
                        // Served the scheduler's work → a positive signal for it.
                        Ok(result) => {
                            // The dispatcher may fetch the output from us.
                            if let Some(objects) = self.objects() {
                                objects.grant(peer, result.content_id);
                            }
                            (MeshFrame::Result(Box::new(result)), Some(true))
                        }
                        // A runner error is OUR side (stage crash / resource) — not
                        // the scheduler's fault, so don't score it.
                        Err(e) => (
                            MeshFrame::Error {
                                message: format!("task execution failed: {e}"),
                            },
                            None,
                        ),
                    }
                }
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
            // The Ack IS the response (sent by serve_stream's write_frame);
            // None here is the reputation signal, not "no reply".
            MeshFrame::PeerExchange(ex) => (self.handle_peer_exchange(peer, conn, *ex).await, None),
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

    /// A5: verify a gossiped peer exchange and learn what it advertises.
    ///
    /// The exchange must come from the peer at the other end of THIS
    /// connection (its `from` equals the TLS-authenticated id) and verify
    /// against that connection's TLS key, so a peer cannot relay or forge
    /// another peer's view. Then: bounded size, fresh, not older than the last
    /// one from the same sender, and within the per-sender rate limit. Gossip
    /// writes only the address book — never the peer registry, never trust.
    async fn handle_peer_exchange(
        &self,
        peer: &PeerId,
        conn: &quinn::Connection,
        ex: PeerExchange,
    ) -> MeshFrame {
        let reject = |why: &str| MeshFrame::Error {
            message: format!("peer exchange {why} (rejected)"),
        };
        // Bound the message before any work: a single (authenticated but only
        // Anonymous) peer must not flood us in one exchange.
        if !ex.within_size_limit() {
            return reject("exceeds the record limit");
        }
        // Drop stale replays (issued_at is a finite-window guard).
        let now = chrono::Utc::now().timestamp();
        if !ex.is_fresh(now) {
            return reject("is stale or future-dated");
        }
        if ex.from != *peer {
            return reject(&format!(
                "claims to be from {}, but this connection is authenticated as {peer}",
                ex.from
            ));
        }
        // The key TLS authenticated on this connection; a legacy peer (no
        // client cert) falls back to its registry entry.
        let sender_key = match tls_authenticated_pubkey(conn) {
            Some(k) => VerifyingKey::from_bytes(&k).ok(),
            None => self.server.peers.read().await.get(peer).map(|p| p.pubkey),
        };
        let Some(sender_key) = sender_key else {
            return reject("from a sender whose key is unknown");
        };
        if !ex.verify(&sender_key) {
            return reject("signature failed to verify");
        }
        {
            let mut rx = self.gossip_rx.lock().unwrap_or_else(|p| p.into_inner());
            let state = rx.entry(peer.clone()).or_insert(GossipRx {
                last_issued_at: i64::MIN,
                tokens: GOSSIP_BURST,
                refilled: Instant::now(),
            });
            let elapsed = state.refilled.elapsed().as_secs_f64();
            state.tokens = (state.tokens + elapsed).min(GOSSIP_BURST);
            state.refilled = Instant::now();
            if ex.issued_at < state.last_issued_at {
                return reject("is older than one already accepted from this peer");
            }
            if state.tokens < 1.0 {
                return reject("arrived too soon after the last one (rate limited)");
            }
            state.tokens -= 1.0;
            state.last_issued_at = ex.issued_at;
        }
        let observed_ip = conn.remote_address().ip();
        let mut changed = false;
        let mut book = self.addr_book.write().await;
        for record in &ex.records {
            let id = record.peer_id();
            // Never learn our own address back from a peer.
            if id == self.node_id {
                continue;
            }
            let own = id == ex.from;
            let mut addr = record.addr;
            if addr.ip().is_unspecified() {
                // A wildcard is only meaningful in the sender's OWN record: it
                // means "my listen port, at whatever IP you see me on".
                if !own {
                    continue;
                }
                addr.set_ip(observed_ip);
            }
            if !dialable(&addr) {
                continue;
            }
            let room = book.len() < MAX_ADDR_BOOK;
            match book.get_mut(&id) {
                Some(e) => {
                    // The peer's own signed claim about where it listens wins;
                    // a third party's claim never overrides an address this
                    // node has confirmed or was given as a seed.
                    let may_update =
                        own || (!e.confirmed && !e.self_advertised && e.source != PeerSource::Seed);
                    if may_update && e.addr != addr {
                        e.addr = addr;
                        e.confirmed = false;
                        e.failures = 0;
                        e.next_dial = None;
                        changed = true;
                    }
                    if own {
                        e.self_advertised = true;
                    }
                }
                // Cap total growth: stop taking NEW ids once the book is full
                // (bounds memory against a flood of distinct keys across many
                // exchanges).
                None if room => {
                    book.insert(
                        id,
                        BookEntry {
                            pubkey: record.pubkey,
                            addr,
                            source: if own {
                                PeerSource::Inbound
                            } else {
                                PeerSource::Gossip
                            },
                            confirmed: false,
                            self_advertised: own,
                            failures: 0,
                            next_dial: None,
                        },
                    );
                    changed = true;
                }
                None => {}
            }
        }
        drop(book);
        if changed {
            self.wake.notify_one();
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
            input_content_id: crate::framework::ArtifactContentId::from_digest(
                ContentHash::of_bytes(b"in"),
            ),
            invocation_key: crate::framework::InvocationKey::from_digest(ContentHash::of_bytes(
                b"echo-invocation",
            )),
            args_hash: ContentHash::of_bytes(b"args"),
            expected_content_id: Some(crate::framework::ArtifactContentId::from_digest(
                ContentHash::of_bytes(b"out"),
            )),
            args: serde_json::json!({}),
            resources: ResourceRequest::default(),
            data_class: DataClass::Public,
            timeout_secs: 30,
            deadline: crate::framework::execution::ExecutionDeadline::from_now(
                None,
                std::time::Duration::from_secs(30),
            ),
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
        let result = tokio::time::timeout(crate::TEST_HANG_GUARD, a.dispatch(&conn, task))
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
        let err = tokio::time::timeout(crate::TEST_HANG_GUARD, client.dispatch(&conn, task))
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
            crate::TEST_HANG_GUARD,
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
            format!("{err}").contains("claims to be from"),
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
                    content_id: Some(crate::framework::ArtifactContentId::from_digest(
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
        let _ = tokio::time::timeout(crate::TEST_HANG_GUARD, writer).await;
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
        tokio::time::timeout(crate::TEST_HANG_GUARD, client.ping(&conn, 7))
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

    // ── gossip (A5) ──────────────────────────────────────────────────────────

    use crate::p2p::gossip::{MAX_RECORDS_PER_EXCHANGE, PeerExchange, PeerRecord};

    /// A full node whose echo runner signs with the node's own key, with gossip
    /// running on a short interval.
    async fn gossip_node(max_peers: usize) -> (Arc<MeshNode>, Arc<KeyPair>) {
        let kp = Arc::new(KeyPair::generate());
        let dir = tempfile::tempdir().unwrap();
        let peers = PeerRegistry::load(&dir.path().join("peers.json")).unwrap();
        std::mem::forget(dir);
        let runner: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
            keypair: kp.clone(),
        });
        let node = MeshNode::bind(
            "127.0.0.1:0".parse().unwrap(),
            kp.clone(),
            peers,
            NodeCapabilities::default(),
            Some(runner),
            false,
        )
        .await
        .unwrap();
        let cfg = GossipConfig::default()
            .with_interval(Duration::from_millis(300))
            .with_max_peers(max_peers)
            .with_dial_timeout(Duration::from_secs(3))
            .with_backoff(Duration::from_millis(200), Duration::from_secs(2));
        assert!(node.start_gossip(cfg.clone()));
        assert!(!node.start_gossip(cfg), "second start is a no-op");
        (node, kp)
    }

    async fn wait_until<F, Fut>(what: &str, mut cond: F)
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if cond().await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("timed out waiting for: {what}");
    }

    fn rec(kp: &KeyPair, addr: &str) -> PeerRecord {
        PeerRecord {
            pubkey: kp.verifying,
            addr: addr.parse().unwrap(),
        }
    }

    fn now() -> i64 {
        chrono::Utc::now().timestamp()
    }

    /// Plain (no-gossip) receiver S and a dialer A connected to it.
    async fn receiver_and_sender() -> (
        Arc<MeshNode>,
        Arc<KeyPair>,
        Arc<MeshNode>,
        Arc<KeyPair>,
        quinn::Connection,
    ) {
        let (s, s_kp) = spawn_node(NodeCapabilities::default(), {
            let r: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
                keypair: Arc::new(KeyPair::generate()),
            });
            Some(r)
        })
        .await;
        let (a, a_kp) = spawn_node(
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
        )
        .await;
        let conn = a
            .connect_to(s.local_addr().unwrap(), s_kp.verifying.to_bytes())
            .await
            .unwrap();
        (s, s_kp, a, a_kp, conn)
    }

    /// The headline: D seeds only A, learns B and C from A's signed exchange,
    /// dials them (pinning the advertised keys), and can dispatch to C.
    #[tokio::test]
    async fn gossip_lets_a_node_seeded_with_one_peer_reach_the_rest() {
        let (a, a_kp) = gossip_node(32).await;
        let (b, _) = gossip_node(32).await;
        let (c, c_kp) = gossip_node(32).await;
        let (d, d_kp) = gossip_node(32).await;
        let a_addr = a.local_addr().unwrap();
        let a_key = a_kp.verifying.to_bytes();
        for n in [&b, &c, &d] {
            n.add_seed(a_addr, a_key).await.unwrap();
            n.connect_to(a_addr, a_key).await.unwrap();
        }
        let (b_id, c_id) = (b.node_id().clone(), c.node_id().clone());
        wait_until("D connects to B and C", || {
            let d = d.clone();
            let (b_id, c_id) = (b_id.clone(), c_id.clone());
            async move {
                let up = d.connected_peers().await;
                up.contains(&b_id) && up.contains(&c_id)
            }
        })
        .await;
        let view = d.mesh_view().await;
        let c_row = view.iter().find(|p| p.id == c_id).unwrap();
        assert!(c_row.connected);
        assert_eq!(c_row.addr, Some(c.local_addr().unwrap()));
        // Gossip never wrote D's registry: learned peers carry no trust.
        assert_eq!(d.peer_trust(&b_id).await, None);
        assert_eq!(d.peer_trust(&c_id).await, None);
        // And D can dispatch to C over the learned connection.
        let conn = d.connection_to(&c_id).await.unwrap();
        let result = tokio::time::timeout(
            crate::TEST_HANG_GUARD,
            d.dispatch(&conn, echo_task(d.node_id(), &d_kp)),
        )
        .await
        .expect("no hang")
        .expect("C runs D's task");
        assert_eq!(result.peer_id, c_id);
        assert!(crate::p2p::crypto::verify(
            &c_kp.verifying,
            &result.sign_payload(),
            &result.signature
        ));
    }

    /// `max_peers` bounds what a node DIALS. B and C don't gossip (so they
    /// never dial D themselves); A reaches them, so A advertises them; D learns
    /// both but, at max_peers = 1, dials neither.
    #[tokio::test]
    async fn gossip_respects_max_peers() {
        let (a, a_kp) = gossip_node(32).await;
        let quiet = || async {
            spawn_node(NodeCapabilities::default(), {
                let r: Arc<dyn MeshTaskRunner> = Arc::new(EchoRunner {
                    keypair: Arc::new(KeyPair::generate()),
                });
                Some(r)
            })
            .await
        };
        let (b, b_kp) = quiet().await;
        let (c, c_kp) = quiet().await;
        a.connect_to(b.local_addr().unwrap(), b_kp.verifying.to_bytes())
            .await
            .unwrap();
        a.connect_to(c.local_addr().unwrap(), c_kp.verifying.to_bytes())
            .await
            .unwrap();
        let (d, _) = gossip_node(1).await;
        let a_addr = a.local_addr().unwrap();
        let a_key = a_kp.verifying.to_bytes();
        d.add_seed(a_addr, a_key).await.unwrap();
        d.connect_to(a_addr, a_key).await.unwrap();
        let (b_id, c_id) = (b.node_id().clone(), c.node_id().clone());
        wait_until("D learns B and C", || {
            let d = d.clone();
            let (b_id, c_id) = (b_id.clone(), c_id.clone());
            async move {
                d.learned_addr(&b_id).await.is_some() && d.learned_addr(&c_id).await.is_some()
            }
        })
        .await;
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let up = d.connected_peers().await;
        assert_eq!(up.len(), 1, "D dialed beyond max_peers: {up:?}");
    }

    /// An exchange signed by another (real) peer and relayed over A's
    /// connection is refused: `from` must be the connection's TLS identity.
    #[tokio::test]
    async fn gossip_claiming_another_identity_is_rejected() {
        let (s, _s_kp, a, _a_kp, conn) = receiver_and_sender().await;
        let b_kp = KeyPair::generate();
        let x = KeyPair::generate();
        // Validly signed by B, but sent by A.
        let relayed = PeerExchange::create(&b_kp, now(), vec![rec(&x, "127.0.0.1:4001")]);
        let err = a.send_gossip(&conn, relayed).await.unwrap_err();
        assert!(format!("{err}").contains("claims to be from"), "{err}");
        assert_eq!(
            s.learned_addr(&PeerId::from_pubkey(&x.verifying)).await,
            None
        );
    }

    #[tokio::test]
    async fn gossip_with_a_bad_or_missing_signature_is_rejected() {
        let (s, _s_kp, a, a_kp, conn) = receiver_and_sender().await;
        let x = KeyPair::generate();
        let x_id = PeerId::from_pubkey(&x.verifying);
        // Tampered after signing.
        let mut tampered = PeerExchange::create(&a_kp, now(), vec![rec(&x, "127.0.0.1:4001")]);
        tampered.records[0].addr = "127.0.0.1:4002".parse().unwrap();
        let err = a.send_gossip(&conn, tampered).await.unwrap_err();
        assert!(format!("{err}").contains("signature"), "{err}");
        // "Unsigned": an all-zero signature.
        let mut unsigned = PeerExchange::create(&a_kp, now(), vec![rec(&x, "127.0.0.1:4001")]);
        unsigned.signature = ed25519_dalek::Signature::from_bytes(&[0u8; 64]);
        let err = a.send_gossip(&conn, unsigned).await.unwrap_err();
        assert!(format!("{err}").contains("signature"), "{err}");
        assert_eq!(s.learned_addr(&x_id).await, None);
    }

    #[tokio::test]
    async fn oversized_stale_and_replayed_exchanges_are_rejected() {
        let (s, _s_kp, a, a_kp, conn) = receiver_and_sender().await;
        let big: Vec<_> = (0..MAX_RECORDS_PER_EXCHANGE + 1)
            .map(|_| rec(&KeyPair::generate(), "127.0.0.1:4001"))
            .collect();
        let err = a
            .send_gossip(&conn, PeerExchange::create(&a_kp, now(), big))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("record limit"), "{err}");

        let stale = PeerExchange::create(&a_kp, now() - 2 * 24 * 3600, vec![]);
        let err = a.send_gossip(&conn, stale).await.unwrap_err();
        assert!(format!("{err}").contains("stale"), "{err}");

        // A newer exchange is accepted, then an older (captured) one is not.
        let older = PeerExchange::create(&a_kp, now() - 60, vec![]);
        let newer = PeerExchange::create(&a_kp, now(), vec![]);
        a.send_gossip(&conn, newer).await.expect("fresh exchange");
        let err = a.send_gossip(&conn, older).await.unwrap_err();
        assert!(format!("{err}").contains("older"), "{err}");
        assert!(
            s.mesh_view().await.iter().all(|p| p.addr.is_none()),
            "nothing was learned"
        );
    }

    #[tokio::test]
    async fn a_flood_of_exchanges_is_rate_limited() {
        let (_s, _s_kp, a, a_kp, conn) = receiver_and_sender().await;
        let mut limited = 0;
        for _ in 0..20 {
            if let Err(e) = a
                .send_gossip(&conn, PeerExchange::create(&a_kp, now(), vec![]))
                .await
            {
                assert!(format!("{e}").contains("rate limited"), "{e}");
                limited += 1;
            }
        }
        assert!(limited >= 10, "only {limited} of 20 were rate limited");
    }

    /// A wildcard address means "my port, at the IP you see me on" — and only
    /// in the sender's own record. A record naming the receiver itself is
    /// ignored (no self-dial).
    #[tokio::test]
    async fn wildcard_only_resolves_in_the_senders_own_record() {
        let (s, s_kp, a, a_kp, conn) = receiver_and_sender().await;
        let x = KeyPair::generate();
        let ex = PeerExchange::create(
            &a_kp,
            now(),
            vec![
                rec(&a_kp, "0.0.0.0:4500"),
                rec(&x, "0.0.0.0:4600"),
                rec(&s_kp, "127.0.0.1:1"),
            ],
        );
        a.send_gossip(&conn, ex).await.unwrap();
        assert_eq!(
            s.learned_addr(a.node_id()).await,
            Some("127.0.0.1:4500".parse().unwrap()),
            "the sender's wildcard took the observed IP"
        );
        assert_eq!(
            s.learned_addr(&PeerId::from_pubkey(&x.verifying)).await,
            None,
            "a third party's wildcard is dropped"
        );
        assert_eq!(s.learned_addr(s.node_id()).await, None, "no self entry");
    }

    /// A third party cannot redirect a peer this node has reached itself, and
    /// gossip does not touch the trust of a registered peer.
    #[tokio::test]
    async fn gossip_cannot_move_a_confirmed_peer_or_change_trust() {
        let (s, _s_kp, a, a_kp, conn) = receiver_and_sender().await;
        // S dials Y itself → confirmed.
        let (y, y_kp) = spawn_node(
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
        )
        .await;
        let y_addr = y.local_addr().unwrap();
        s.connect_to(y_addr, y_kp.verifying.to_bytes())
            .await
            .unwrap();
        let trust_before = s.peer_trust(a.node_id()).await;
        assert_eq!(trust_before, Some(TrustLevel::Anonymous));
        let ex = PeerExchange::create(&a_kp, now(), vec![rec(&y_kp, "127.0.0.1:9")]);
        a.send_gossip(&conn, ex).await.unwrap();
        assert_eq!(s.learned_addr(y.node_id()).await, Some(y_addr));
        assert_eq!(s.peer_trust(a.node_id()).await, trust_before);
        assert_eq!(s.peer_trust(y.node_id()).await, None);
    }

    /// A record pairing a key with the address of a node that holds a
    /// DIFFERENT key: the dial pins the advertised key, fails, and backs off.
    #[tokio::test]
    async fn a_learned_peer_whose_key_does_not_answer_is_not_connected() {
        let (s, _) = gossip_node(32).await;
        let (a, a_kp) = gossip_node(32).await;
        let (y, _) = gossip_node(32).await; // listens, but under its own key
        let impostor = KeyPair::generate();
        let impostor_id = PeerId::from_pubkey(&impostor.verifying);
        let conn = a
            .connect_to(s.local_addr().unwrap(), s.keypair.verifying.to_bytes())
            .await
            .unwrap();
        let ex = PeerExchange::create(
            &a_kp,
            now(),
            vec![rec(&impostor, &y.local_addr().unwrap().to_string())],
        );
        a.send_gossip(&conn, ex).await.unwrap();
        wait_until("S tries the impostor and fails", || {
            let s = s.clone();
            let id = impostor_id.clone();
            async move {
                s.mesh_view()
                    .await
                    .iter()
                    .any(|p| p.id == id && p.failures > 0)
            }
        })
        .await;
        assert!(!s.connected_peers().await.contains(&impostor_id));
        assert!(!s.connected_peers().await.contains(y.node_id()));
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let cfg =
            GossipConfig::default().with_backoff(Duration::from_secs(5), Duration::from_secs(60));
        assert_eq!(cfg.backoff(0), Duration::ZERO);
        assert_eq!(cfg.backoff(1), Duration::from_secs(5));
        assert_eq!(cfg.backoff(2), Duration::from_secs(10));
        assert_eq!(cfg.backoff(4), Duration::from_secs(40));
        assert_eq!(cfg.backoff(5), Duration::from_secs(60));
        assert_eq!(cfg.backoff(200), Duration::from_secs(60));
    }

    #[tokio::test]
    async fn a_node_refuses_to_seed_or_dial_itself() {
        let (n, kp) = gossip_node(32).await;
        let addr = n.local_addr().unwrap();
        assert!(n.add_seed(addr, kp.verifying.to_bytes()).await.is_err());
        assert!(n.connect_to(addr, kp.verifying.to_bytes()).await.is_err());
    }

    /// A malformed frame fails its stream, not the connection: the peer can
    /// keep using the connection afterwards.
    #[tokio::test]
    async fn a_bad_frame_does_not_end_the_connection() {
        let (_s, _s_kp, a, _a_kp, conn) = receiver_and_sender().await;
        let (mut send, mut recv) = conn.open_bi().await.unwrap();
        send.write_all(&[
            crate::p2p::mesh_wire::PROTOCOL_VERSION,
            3,
            0,
            0,
            0,
            b'x',
            b'y',
            b'z',
        ])
        .await
        .unwrap();
        send.finish().unwrap();
        let reply = crate::p2p::mesh_wire::read_frame(&mut recv).await.unwrap();
        assert_eq!(reply.kind(), "Error");
        tokio::time::timeout(crate::TEST_HANG_GUARD, a.ping(&conn, 11))
            .await
            .unwrap()
            .expect("connection still served");
    }
}
