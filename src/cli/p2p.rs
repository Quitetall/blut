// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut p2p` — key management, peer registry, serve/connect, smoke probe.
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

use super::*;

/// `blut p2p` subcommands. Behind the `p2p` feature.
#[cfg(feature = "p2p")]
#[derive(Subcommand, Debug)]
pub(super) enum P2pCommand {
    /// Identity key management (Ed25519 + X25519).
    Keys {
        #[command(subcommand)]
        cmd: P2pKeysCommand,
    },
    /// Run as a COORDINATOR: bind a QUIC server, accept peers, hold the
    /// registry. Peers connect to this address. Runs until Ctrl-C.
    Serve {
        /// Listen address (host:port). 0.0.0.0:9320 by default.
        #[arg(long, default_value = "0.0.0.0:9320")]
        addr: String,
        /// Identity key file (defaults to the standard p2p key path).
        #[arg(long)]
        key: Option<std::path::PathBuf>,
        /// Turnkey smoke test (ADR 0067 · T2.3): instead of the long-running
        /// coordinator, accept ONE worker and dispatch this built-in stage to it
        /// over the full data plane (bundle → blob → remote run → verify), print
        /// the verified output, then exit. Use `p2p-echo` for connectivity.
        #[arg(long)]
        smoke_stage: Option<String>,
        /// Text payload for `--smoke-stage` (default: "hello p2p world").
        #[arg(long)]
        smoke_input: Option<String>,
    },
    /// Run as a worker PEER: connect to a coordinator and execute dispatched
    /// stages until the connection closes.
    Connect {
        /// Coordinator address (host:port).
        coordinator: String,
        /// The coordinator's public key (hex, 64 bytes = Ed25519 ‖ X25519) —
        /// required to verify dispatched tasks + seal results. Get it from
        /// `blut p2p keys show` on the coordinator.
        #[arg(long)]
        coordinator_pubkey: String,
        /// Identity key file (defaults to the standard p2p key path).
        #[arg(long)]
        key: Option<std::path::PathBuf>,
    },
    /// Run a symmetric MESH NODE (ADR 0079): one process that is a QUIC server,
    /// a worker, and an optional scheduler at once. Supersedes the role-locked
    /// `serve` (coordinator) / `connect` (worker) split. Runs until Ctrl-C.
    Node {
        /// Listen address (host:port). 0.0.0.0:9320 by default.
        #[arg(long, default_value = "0.0.0.0:9320")]
        listen: String,
        /// Seed peers to dial on start, as `addr@pubkey-hex` (repeatable).
        #[arg(long = "seed")]
        seeds: Vec<String>,
        /// Don't accept + run dispatched tasks (a scheduler-only node).
        #[arg(long, default_value_t = false)]
        no_worker: bool,
        /// Don't dispatch tasks to peers (a worker-only node, e.g. a k8s pool).
        #[arg(long, default_value_t = false)]
        no_scheduler: bool,
        /// Accept legacy peers that present no client cert (the A1 escape hatch,
        /// one transition release).
        #[arg(long, default_value_t = false)]
        allow_legacy_peers: bool,
        /// Identity key file (defaults to the standard p2p key path).
        #[arg(long)]
        key: Option<std::path::PathBuf>,
    },
    /// Peer registry: list / set-trust / remove.
    Peers {
        #[command(subcommand)]
        cmd: P2pPeersCommand,
    },
}

#[cfg(feature = "p2p")]
#[derive(Subcommand, Debug)]
pub(super) enum P2pKeysCommand {
    /// Generate a fresh identity keypair (refuses to overwrite an existing one).
    Generate {
        #[arg(long)]
        key: Option<std::path::PathBuf>,
        /// Overwrite an existing key file.
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Print this node's public key (hex) + PeerId. Share the pubkey with peers.
    Show {
        #[arg(long)]
        key: Option<std::path::PathBuf>,
    },
}

#[cfg(feature = "p2p")]
#[derive(Subcommand, Debug)]
pub(super) enum P2pPeersCommand {
    /// List known peers (id, trust, reputation, capabilities).
    List {
        #[arg(long)]
        json: bool,
    },
    /// Set a peer's trust level (anonymous | registered | trusted).
    Trust {
        /// Peer id (hex, or unique prefix).
        id: String,
        /// New trust level.
        level: String,
    },
    /// Remove a peer from the registry.
    Remove {
        /// Peer id (hex, or unique prefix).
        id: String,
    },
}

// ── P2P CLI (behind the `p2p` feature) ───────────────────────────────────────

#[cfg(feature = "p2p")]
mod p2p_cli {
    use super::*;
    use crate::p2p::crypto::KeyPair;
    use crate::p2p::peer::PeerId;
    use crate::p2p::registry::PeerRegistry;
    use crate::p2p::trust::TrustLevel;

    /// Default identity key path: `<data_dir>/p2p/identity.key`.
    pub fn default_key_path() -> Result<std::path::PathBuf> {
        Ok(paths::data_dir()?.join("p2p").join("identity.key"))
    }

    /// Default peer-registry path: `<data_dir>/p2p/peers.json`.
    pub fn default_registry_path() -> Result<std::path::PathBuf> {
        Ok(paths::data_dir()?.join("p2p").join("peers.json"))
    }

    /// Load the identity keypair from `path` (64-byte file), or error if absent.
    pub fn load_keypair(path: &std::path::Path) -> Result<KeyPair> {
        let bytes = std::fs::read(path).with_context(|| {
            format!(
                "read identity key {} (run `blut p2p keys generate`)",
                path.display()
            )
        })?;
        let arr: [u8; 64] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("identity key {} is not 64 bytes", path.display()))?;
        Ok(KeyPair::from_bytes(&arr))
    }

    /// Parse a trust level from a CLI string.
    pub fn parse_trust(s: &str) -> Result<TrustLevel> {
        match s.to_lowercase().as_str() {
            "anonymous" => Ok(TrustLevel::Anonymous),
            "registered" => Ok(TrustLevel::Registered),
            "trusted" => Ok(TrustLevel::Trusted),
            other => Err(anyhow!(
                "unknown trust level '{other}' (anonymous|registered|trusted)"
            )),
        }
    }

    /// Resolve a peer id from a hex string or unique prefix in the registry.
    /// Case-insensitive (PeerId displays lowercase hex; the user may paste any
    /// case).
    pub fn resolve_peer_id(reg: &PeerRegistry, prefix: &str) -> Result<PeerId> {
        let needle = prefix.to_lowercase();
        let matches: Vec<_> = reg
            .list()
            .into_iter()
            .filter(|p| p.id.to_string().to_lowercase().starts_with(&needle))
            .collect();
        match matches.as_slice() {
            [one] => Ok(one.id.clone()),
            [] => Err(anyhow!("no peer matches '{prefix}'")),
            _ => Err(anyhow!(
                "'{prefix}' is ambiguous ({} peers match)",
                matches.len()
            )),
        }
    }
}

#[cfg(feature = "p2p")]
pub(super) async fn run_p2p_cmd(
    mut reg: crate::framework::Registry,
    cmd: P2pCommand,
) -> Result<()> {
    use crate::p2p::crypto::KeyPair;
    use crate::p2p::peer::PeerId;
    use crate::p2p::registry::PeerRegistry;
    use p2p_cli::*;

    // Ship the built-in connectivity probe (`p2p-echo`) so both the dispatching
    // coordinator (`serve --smoke-stage`) and the executing worker (`connect`)
    // can resolve it via find_erased_stage.
    crate::p2p::smoke::register(&mut reg);

    match cmd {
        P2pCommand::Keys { cmd } => match cmd {
            P2pKeysCommand::Generate { key, force } => {
                let path = key.map(Ok).unwrap_or_else(default_key_path)?;
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let kp = KeyPair::generate();
                // Write the SECRET key atomically with 0600 set AT CREATION (so
                // it is never momentarily world-readable). `create_new` is the
                // atomic refuse-to-overwrite — no exists()/write TOCTOU.
                use std::io::Write as _;
                let mut opts = std::fs::OpenOptions::new();
                opts.write(true);
                if force {
                    opts.create(true).truncate(true);
                } else {
                    opts.create_new(true);
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    opts.mode(0o600);
                }
                let mut f = opts.open(&path).map_err(|e| {
                    if e.kind() == std::io::ErrorKind::AlreadyExists {
                        anyhow!(
                            "identity key already exists at {} (pass --force to overwrite)",
                            path.display()
                        )
                    } else {
                        anyhow!("open identity key {}: {e}", path.display())
                    }
                })?;
                f.write_all(&kp.to_bytes())
                    .with_context(|| format!("write identity key {}", path.display()))?;
                // Durable before we print success — the key is the only copy.
                f.sync_all()
                    .with_context(|| format!("fsync identity key {}", path.display()))?;
                let pid = PeerId::from_pubkey(&kp.verifying);
                println!("identity written to {}", path.display());
                println!("peer id   {pid}");
                println!("pubkey    {}", p2p_pubkey_hex(&kp));
                Ok(())
            }
            P2pKeysCommand::Show { key } => {
                let path = key.map(Ok).unwrap_or_else(default_key_path)?;
                let kp = load_keypair(&path)?;
                let pid = PeerId::from_pubkey(&kp.verifying);
                println!("peer id   {pid}");
                println!("pubkey    {}", p2p_pubkey_hex(&kp));
                println!(
                    "(share the pubkey with peers: `blut p2p connect <coord> --coordinator-pubkey <hex>`)"
                );
                Ok(())
            }
        },
        P2pCommand::Serve {
            addr,
            key,
            smoke_stage,
            smoke_input,
        } => {
            let path = key.map(Ok).unwrap_or_else(default_key_path)?;
            let kp = std::sync::Arc::new(load_keypair(&path)?);
            match smoke_stage {
                Some(stage) => {
                    let input = smoke_input.unwrap_or_else(|| "hello p2p world".to_string());
                    run_p2p_smoke_serve(addr, kp, &reg, stage, input).await
                }
                None => run_p2p_serve(addr, kp).await,
            }
        }
        P2pCommand::Connect {
            coordinator,
            coordinator_pubkey,
            key,
        } => {
            let path = key.map(Ok).unwrap_or_else(default_key_path)?;
            let kp = load_keypair(&path)?;
            run_p2p_connect(&reg, coordinator, coordinator_pubkey, kp).await
        }
        P2pCommand::Node {
            listen,
            seeds,
            no_worker,
            no_scheduler,
            allow_legacy_peers,
            key,
        } => {
            let path = key.map(Ok).unwrap_or_else(default_key_path)?;
            let kp = std::sync::Arc::new(load_keypair(&path)?);
            run_p2p_node(
                listen,
                seeds,
                !no_worker,
                !no_scheduler,
                allow_legacy_peers,
                kp,
            )
            .await
        }
        P2pCommand::Peers { cmd } => {
            let reg_path = default_registry_path()?;
            let mut registry =
                PeerRegistry::load(&reg_path).map_err(|e| anyhow!("load peer registry: {e}"))?;
            match cmd {
                P2pPeersCommand::List { json } => {
                    if json {
                        let arr: Vec<_> = registry
                            .list()
                            .into_iter()
                            .map(|p| {
                                serde_json::json!({
                                    "id": p.id.to_string(),
                                    "trust": p.trust.label(),
                                    "reputation": p.reputation,
                                    "tasks_completed": p.tasks_completed,
                                    "tasks_failed": p.tasks_failed,
                                })
                            })
                            .collect();
                        emit_json(&arr)?;
                    } else {
                        let peers = registry.list();
                        if peers.is_empty() {
                            println!("(no peers registered)");
                        }
                        for p in peers {
                            println!(
                                "{}  trust={:<10} rep={:.2}  ok={} fail={}",
                                p.id.short(),
                                p.trust.label(),
                                p.reputation,
                                p.tasks_completed,
                                p.tasks_failed
                            );
                        }
                    }
                    Ok(())
                }
                P2pPeersCommand::Trust { id, level } => {
                    let pid = resolve_peer_id(&registry, &id)?;
                    let lvl = parse_trust(&level)?;
                    if registry.set_trust(&pid, lvl) {
                        registry.save().map_err(|e| anyhow!("save registry: {e}"))?;
                        println!("set {} trust = {}", pid.short(), lvl.label());
                        Ok(())
                    } else {
                        Err(anyhow!("peer {} not found", pid.short()))
                    }
                }
                P2pPeersCommand::Remove { id } => {
                    let pid = resolve_peer_id(&registry, &id)?;
                    if registry.remove(&pid) {
                        registry.save().map_err(|e| anyhow!("save registry: {e}"))?;
                        println!("removed {}", pid.short());
                        Ok(())
                    } else {
                        Err(anyhow!("peer {} not found", pid.short()))
                    }
                }
            }
        }
    }
}

/// Hex-encode a keypair's public bytes (64 = Ed25519 ‖ X25519) for sharing.
#[cfg(feature = "p2p")]
pub(super) fn p2p_pubkey_hex(kp: &crate::p2p::crypto::KeyPair) -> String {
    let mut pubbytes = [0u8; 64];
    pubbytes[..32].copy_from_slice(kp.verifying.as_bytes());
    pubbytes[32..].copy_from_slice(kp.x25519_public.as_bytes());
    faster_hex::hex_string(&pubbytes)
}

/// Run as a coordinator: bind the QUIC server, accept peers, hold the registry.
#[cfg(feature = "p2p")]
pub(super) async fn run_p2p_serve(
    addr: String,
    keypair: std::sync::Arc<crate::p2p::crypto::KeyPair>,
) -> Result<()> {
    use crate::p2p::Coordinator;
    use crate::p2p::dispatch::{DefaultDispatchPolicy, DispatchPolicy};
    use crate::p2p::registry::PeerRegistry;
    use crate::p2p::trust::DispatchMatrix;

    let sockaddr: std::net::SocketAddr = addr
        .parse()
        .with_context(|| format!("parse listen addr '{addr}'"))?;
    let reg_path = p2p_cli::default_registry_path()?;
    let registry = PeerRegistry::load(&reg_path).map_err(|e| anyhow!("load registry: {e}"))?;
    let dispatch: std::sync::Arc<dyn DispatchPolicy> =
        std::sync::Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default()));

    let coordinator = Coordinator::start(sockaddr, keypair.clone(), dispatch, registry)
        .await
        .map_err(|e| anyhow!("start coordinator: {e}"))?;
    let bound = coordinator.local_addr().map_err(|e| anyhow!("{e}"))?;
    let pid = crate::p2p::peer::PeerId::from_pubkey(&keypair.verifying);
    eprintln!("coordinator listening on {bound}");
    eprintln!("peer id   {pid}");
    eprintln!("pubkey    {}", p2p_pubkey_hex(&keypair));
    eprintln!("(peers: `blut p2p connect {bound} --coordinator-pubkey <pubkey>`)");
    eprintln!("Ctrl-C to stop.");

    // Persist the registry on shutdown so newly-handshaked peers survive a restart.
    tokio::signal::ctrl_c().await.ok();
    eprintln!("\nshutting down…");
    {
        let peers = coordinator.peers();
        let g = peers.read().await;
        if let Err(e) = g.save() {
            eprintln!("warning: failed to persist peer registry: {e}");
        }
    }
    coordinator.shutdown();
    Ok(())
}

/// Turnkey smoke serve (ADR 0067 · T2.3): bind a one-shot QUIC server, accept ONE
/// worker, dispatch `stage` to it over the full data plane, print the
/// content-verified output, then exit. Drives a raw `P2pServer` (not the full
/// `Coordinator`) so `dispatch_to_peer`'s `recv_result` doesn't race the
/// `Coordinator`'s background `handle_peer` reader on the same connection.
#[cfg(feature = "p2p")]
pub(super) async fn run_p2p_smoke_serve(
    addr: String,
    keypair: std::sync::Arc<crate::p2p::crypto::KeyPair>,
    reg: &crate::framework::Registry,
    stage: String,
    input: String,
) -> Result<()> {
    use crate::p2p::registry::PeerRegistry;
    use crate::p2p::transport::P2pServer;

    // Fail fast on a bad stage name BEFORE binding / waiting for a worker — a typo
    // would otherwise hang on accept_peer and then fail cryptically on the worker.
    // (smoke_dispatch_once ships a text SmokeText input, so the built-in `p2p-echo`
    // is the dispatchable target here; a real-workload dispatch CLI is future work.)
    if reg.find_erased_stage(&stage).is_none() {
        return Err(anyhow!(
            "unknown stage '{stage}': not registered. Use `--smoke-stage p2p-echo` \
             for the built-in connectivity probe."
        ));
    }

    let sockaddr: std::net::SocketAddr = addr
        .parse()
        .with_context(|| format!("parse listen addr '{addr}'"))?;
    let reg_path = p2p_cli::default_registry_path()?;
    let peers = PeerRegistry::load(&reg_path).map_err(|e| anyhow!("load registry: {e}"))?;
    let server = std::sync::Arc::new(
        P2pServer::bind(sockaddr, keypair.clone(), peers)
            .await
            .map_err(|e| anyhow!("bind smoke server: {e}"))?,
    );
    let bound = server.local_addr().map_err(|e| anyhow!("{e}"))?;
    eprintln!("smoke coordinator listening on {bound}");
    eprintln!("pubkey    {}", p2p_pubkey_hex(&keypair));
    eprintln!("(on the worker: `blut p2p connect {bound} --coordinator-pubkey <pubkey>`)");
    eprintln!("waiting for ONE worker to connect, then dispatching stage '{stage}'…");

    let (peer_id, conn) = server
        .accept_peer()
        .await
        .map_err(|e| anyhow!("accept worker: {e}"))?;
    eprintln!("worker {peer_id} connected — dispatching…");

    let out = crate::p2p::smoke::smoke_dispatch_once(
        &server, &conn, &keypair, reg, &peer_id, &stage, &input, 60,
    )
    .await
    .map_err(|e| anyhow!("smoke dispatch: {e}"))?;

    println!("✔ stage '{stage}' ran on peer {peer_id}; verified output:");
    println!("{out}");
    server.shutdown();
    Ok(())
}

/// Run as a worker peer: connect to the coordinator and execute dispatched tasks.
#[cfg(feature = "p2p")]
pub(super) async fn run_p2p_connect(
    reg: &crate::framework::Registry,
    coordinator: String,
    coordinator_pubkey_hex: String,
    keypair: crate::p2p::crypto::KeyPair,
) -> Result<()> {
    use crate::p2p::dispatch::DefaultDispatchPolicy;
    use crate::p2p::peer_exec::{CoordinatorKeys, run_peer_loop};
    use crate::p2p::transport::P2pClient;
    use crate::p2p::trust::DispatchMatrix;

    let sockaddr: std::net::SocketAddr = coordinator
        .parse()
        .with_context(|| format!("parse coordinator addr '{coordinator}'"))?;

    // Decode the coordinator's 64-byte public key (Ed25519 ‖ X25519).
    let mut pub64 = [0u8; 64];
    faster_hex::hex_decode(coordinator_pubkey_hex.as_bytes(), &mut pub64)
        .map_err(|e| anyhow!("invalid --coordinator-pubkey hex: {e}"))?;
    let verifying = ed25519_dalek::VerifyingKey::from_bytes(&pub64[..32].try_into().unwrap())
        .map_err(|e| anyhow!("invalid coordinator Ed25519 key: {e}"))?;
    let x_arr: [u8; 32] = pub64[32..].try_into().unwrap();
    let x25519_pub = x25519_dalek::PublicKey::from(x_arr);
    let coord_keys = CoordinatorKeys {
        verifying,
        x25519_pub,
    };

    let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
    let work_root = crate::p2p::peer_exec::default_work_root();
    std::fs::create_dir_all(&work_root)
        .with_context(|| format!("create peer work root {}", work_root.display()))?;

    let client = P2pClient::with_coordinator_pin(std::sync::Arc::new(clone_keypair(&keypair)), {
        let mut k = [0u8; 32];
        k.copy_from_slice(verifying.as_bytes());
        k
    });
    let (conn, my_id) = client
        .connect(sockaddr)
        .await
        .map_err(|e| anyhow!("connect to coordinator: {e}"))?;
    eprintln!("connected to coordinator {coordinator} as peer {my_id}");
    eprintln!("waiting for dispatched tasks (Ctrl-C to stop)…");

    run_peer_loop(&conn, &keypair, &coord_keys, reg, &policy, &work_root)
        .await
        .map_err(|e| anyhow!("peer loop: {e}"))?;
    eprintln!("coordinator connection closed.");
    Ok(())
}

/// `KeyPair` doesn't derive Clone (it holds secrets); reconstruct from bytes
/// when we need a second owner (the client pin path takes an Arc).
#[cfg(feature = "p2p")]
pub(super) fn clone_keypair(kp: &crate::p2p::crypto::KeyPair) -> crate::p2p::crypto::KeyPair {
    crate::p2p::crypto::KeyPair::from_bytes(&kp.to_bytes())
}

/// Parse a `--seed` spec `addr@ed25519-pubkey-hex` into its parts (the pubkey
/// pins the peer's TLS identity when dialing, A1).
#[cfg(feature = "p2p")]
fn parse_seed(spec: &str) -> Result<(std::net::SocketAddr, [u8; 32])> {
    let (addr, hex) = spec
        .split_once('@')
        .ok_or_else(|| anyhow!("seed must be 'addr@pubkey-hex', got '{spec}'"))?;
    let addr: std::net::SocketAddr = addr
        .parse()
        .with_context(|| format!("seed address '{addr}'"))?;
    let mut pubkey = [0u8; 32];
    faster_hex::hex_decode(hex.as_bytes(), &mut pubkey)
        .map_err(|e| anyhow!("seed pubkey hex: {e}"))?;
    Ok((addr, pubkey))
}

/// A connectivity/smoke worker runner (ADR 0079 A3): returns a signed result
/// echoing the task's expected output hash. Proves the mesh dispatch→execute→
/// result round-trip end-to-end. Production stage execution over the mesh
/// (materialize input → run cookbook stage → seal output) rides the chunked
/// blob transfer (D1.1) + the cookbook registry, wired separately.
#[cfg(feature = "p2p")]
struct SmokeRunner {
    keypair: std::sync::Arc<crate::p2p::crypto::KeyPair>,
}

#[cfg(feature = "p2p")]
#[async_trait::async_trait]
impl crate::p2p::node::MeshTaskRunner for SmokeRunner {
    async fn run(
        &self,
        task: crate::p2p::task::TaskManifest,
    ) -> std::result::Result<crate::p2p::task::TaskResult, crate::error::TrainError> {
        // Build with a zero signature, then sign (sign_payload never reads the
        // signature field — no wasted signing pass).
        let mut result = crate::p2p::task::TaskResult {
            task_id: task.task_id,
            peer_id: crate::p2p::peer::PeerId::from_pubkey(&self.keypair.verifying),
            output_hash: task.expected_output_hash,
            encrypted_output: None,
            wall_time_ms: 0,
            signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
        };
        result.signature = self.keypair.sign(&result.sign_payload());
        Ok(result)
    }
}

/// Run a symmetric mesh node until Ctrl-C (ADR 0079 A3).
#[cfg(feature = "p2p")]
pub(super) async fn run_p2p_node(
    listen: String,
    seeds: Vec<String>,
    worker: bool,
    scheduler: bool,
    allow_legacy: bool,
    keypair: std::sync::Arc<crate::p2p::crypto::KeyPair>,
) -> Result<()> {
    use crate::p2p::node::{MeshNode, MeshTaskRunner, NodeCapabilities};
    use crate::p2p::registry::PeerRegistry;

    let sockaddr: std::net::SocketAddr = listen
        .parse()
        .with_context(|| format!("parse listen addr '{listen}'"))?;
    let reg_path = p2p_cli::default_registry_path()?;
    let peers = PeerRegistry::load(&reg_path).map_err(|e| anyhow!("load peer registry: {e}"))?;

    let caps = NodeCapabilities { worker, scheduler };
    let runner: Option<std::sync::Arc<dyn MeshTaskRunner>> = if worker {
        Some(std::sync::Arc::new(SmokeRunner {
            keypair: keypair.clone(),
        }))
    } else {
        None
    };

    let node = MeshNode::bind(sockaddr, keypair.clone(), peers, caps, runner, allow_legacy)
        .await
        .map_err(|e| anyhow!("bind mesh node: {e}"))?;
    let bound = node.local_addr().map_err(|e| anyhow!("{e}"))?;
    eprintln!("mesh node {} listening on {bound}", node.node_id());
    eprintln!("pubkey       {}", p2p_pubkey_hex(&keypair));
    eprintln!("capabilities worker={worker} scheduler={scheduler} allow_legacy={allow_legacy}");

    // Dial seeds (best-effort — a down/hung seed must not block the others, so
    // each dial is time-bounded).
    for spec in &seeds {
        match parse_seed(spec) {
            Ok((addr, pubkey)) => {
                let dial = node.connect_to(addr, pubkey);
                match tokio::time::timeout(std::time::Duration::from_secs(10), dial).await {
                    Ok(Ok(_conn)) => eprintln!("connected to seed {addr}"),
                    Ok(Err(e)) => eprintln!("seed {addr}: {e}"),
                    Err(_) => eprintln!("seed {addr}: dial timed out"),
                }
            }
            Err(e) => eprintln!("bad --seed '{spec}': {e}"),
        }
    }

    eprintln!("node running — Ctrl-C to stop");
    if let Err(e) = tokio::signal::ctrl_c().await {
        eprintln!("signal wait failed: {e}");
    }
    eprintln!("shutting down mesh node");
    Ok(())
}

#[cfg(all(test, feature = "p2p"))]
mod p2p_cli_tests {
    use super::{Cli, Command, P2pCommand, P2pKeysCommand, P2pPeersCommand};
    use clap::Parser;

    fn p2p_of(argv: &[&str]) -> P2pCommand {
        match Cli::try_parse_from(argv).expect("parse").command {
            Some(Command::P2p { cmd }) => cmd,
            other => panic!("expected p2p, got {other:?}"),
        }
    }

    #[test]
    fn keys_generate_parses() {
        match p2p_of(&["blut", "p2p", "keys", "generate", "--force"]) {
            P2pCommand::Keys {
                cmd: P2pKeysCommand::Generate { force, .. },
            } => assert!(force),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn serve_defaults_addr() {
        match p2p_of(&["blut", "p2p", "serve"]) {
            P2pCommand::Serve { addr, .. } => assert_eq!(addr, "0.0.0.0:9320"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn node_parses_capabilities_and_seeds() {
        match p2p_of(&[
            "blut",
            "p2p",
            "node",
            "--listen",
            "0.0.0.0:9999",
            "--no-scheduler",
            "--seed",
            "1.2.3.4:9320@abcd",
            "--seed",
            "5.6.7.8:9320@ef01",
        ]) {
            P2pCommand::Node {
                listen,
                seeds,
                no_worker,
                no_scheduler,
                allow_legacy_peers,
                ..
            } => {
                assert_eq!(listen, "0.0.0.0:9999");
                assert_eq!(seeds.len(), 2);
                assert!(!no_worker);
                assert!(no_scheduler, "--no-scheduler set");
                assert!(!allow_legacy_peers);
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn node_defaults_are_full_peer() {
        match p2p_of(&["blut", "p2p", "node"]) {
            P2pCommand::Node {
                listen,
                no_worker,
                no_scheduler,
                ..
            } => {
                assert_eq!(listen, "0.0.0.0:9320");
                assert!(!no_worker && !no_scheduler, "default = worker + scheduler");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn connect_requires_pubkey() {
        // Missing --coordinator-pubkey must fail to parse.
        assert!(Cli::try_parse_from(["blut", "p2p", "connect", "1.2.3.4:9320"]).is_err());
        // With it, parses.
        match p2p_of(&[
            "blut",
            "p2p",
            "connect",
            "1.2.3.4:9320",
            "--coordinator-pubkey",
            "deadbeef",
        ]) {
            P2pCommand::Connect {
                coordinator,
                coordinator_pubkey,
                ..
            } => {
                assert_eq!(coordinator, "1.2.3.4:9320");
                assert_eq!(coordinator_pubkey, "deadbeef");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn peers_subcommands_parse() {
        match p2p_of(&["blut", "p2p", "peers", "list", "--json"]) {
            P2pCommand::Peers {
                cmd: P2pPeersCommand::List { json },
            } => assert!(json),
            other => panic!("got {other:?}"),
        }
        match p2p_of(&["blut", "p2p", "peers", "trust", "abc123", "trusted"]) {
            P2pCommand::Peers {
                cmd: P2pPeersCommand::Trust { id, level },
            } => {
                assert_eq!(id, "abc123");
                assert_eq!(level, "trusted");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn trust_level_parses() {
        use super::p2p_cli::parse_trust;
        use crate::p2p::trust::TrustLevel;
        assert_eq!(parse_trust("trusted").unwrap(), TrustLevel::Trusted);
        assert_eq!(parse_trust("ANONYMOUS").unwrap(), TrustLevel::Anonymous);
        assert!(parse_trust("bogus").is_err());
    }

    #[test]
    fn keypair_file_roundtrips_and_is_0600() {
        // generate (atomic create_new + 0600) → load → same identity. Then a
        // second generate without --force must refuse.
        use super::p2p_cli::load_keypair;
        use crate::p2p::crypto::KeyPair;
        use crate::p2p::peer::PeerId;
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("p2p/identity.key");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let kp = KeyPair::generate();
        // Mirror the CLI's atomic 0600 write.
        use std::io::Write as _;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        opts.open(&path).unwrap().write_all(&kp.to_bytes()).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "secret key must be owner-only");
        }

        let loaded = load_keypair(&path).unwrap();
        assert_eq!(
            PeerId::from_pubkey(&loaded.verifying),
            PeerId::from_pubkey(&kp.verifying),
            "loaded identity matches generated"
        );

        // create_new on an existing path is the atomic refuse-overwrite.
        let mut opts2 = std::fs::OpenOptions::new();
        opts2.write(true).create_new(true);
        assert_eq!(
            opts2.open(&path).unwrap_err().kind(),
            std::io::ErrorKind::AlreadyExists
        );
    }
}
