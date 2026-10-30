// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `--launcher p2p` for `recipe run` and `recipe declare --run`: join the
//! `p2p node` mesh and place eligible stages on peers
//! ([`crate::p2p::mesh_exec`]).
//!
//! The submitting command starts a scheduler-only mesh node with this
//! machine's mesh identity and peer registry, dials the seeds, lets gossip
//! find the rest, and installs the mesh execution adapter on every plan it
//! launches. The node lives for the command: a sweep's combos share it, and it
//! closes its connections (stopping any stage still running on a peer) when
//! the command returns.
//!
//! Configuration comes from the flags below, or the matching environment
//! variables, and the files `blut p2p` already uses: the identity key
//! (`<data_dir>/p2p/identity.key`) and the peer registry
//! (`<data_dir>/p2p/peers.json`, which holds the trust an operator gave each
//! peer with `blut p2p peers trust`).

use super::*;

/// Seeds for `--launcher p2p`, `addr@pubkey`, separated by commas or spaces.
pub(super) const SEEDS_ENV: &str = "BLUT_P2P_SEEDS";
/// Identity key for `--launcher p2p`.
pub(super) const KEY_ENV: &str = "BLUT_P2P_KEY";
/// Listen address of the submitting node.
pub(super) const LISTEN_ENV: &str = "BLUT_P2P_LISTEN";
/// Address the submitting node advertises in gossip.
pub(super) const ADVERTISE_ENV: &str = "BLUT_P2P_ADVERTISE";
/// Seconds an eligible stage waits for a capable peer.
pub(super) const PEER_WAIT_ENV: &str = "BLUT_P2P_PEER_WAIT";

/// The mesh a `--launcher p2p` run joins. Only meaningful with `--launcher
/// p2p`; any of these with another launcher is refused rather than ignored.
#[derive(clap::Args, Debug, Clone, Default)]
pub(super) struct MeshArgs {
    /// `--launcher p2p`: a mesh peer to join through, as `addr@pubkey`, where
    /// `pubkey` is the line `blut p2p keys show` prints on that peer
    /// (repeatable; also `BLUT_P2P_SEEDS`, comma-separated). Gossip finds the
    /// rest of the mesh.
    #[arg(long = "p2p-seed", value_name = "ADDR@PUBKEY")]
    pub(super) seeds: Vec<String>,
    /// `--launcher p2p`: this machine's mesh identity (default
    /// `<data_dir>/p2p/identity.key`; also `BLUT_P2P_KEY`). A worker on the
    /// same machine needs an identity of its own.
    #[arg(long = "p2p-key", value_name = "FILE")]
    pub(super) key: Option<std::path::PathBuf>,
    /// `--launcher p2p`: address the submitting node listens on (default
    /// `0.0.0.0:0`, any free port: workers fetch inputs over the connection
    /// the stage arrived on, so nothing needs to reach this machine; also
    /// `BLUT_P2P_LISTEN`).
    #[arg(long = "p2p-listen", value_name = "HOST:PORT")]
    pub(super) listen: Option<String>,
    /// `--launcher p2p`: address to advertise in gossip, behind NAT or a port
    /// forward (also `BLUT_P2P_ADVERTISE`).
    #[arg(long = "p2p-advertise", value_name = "HOST:PORT")]
    pub(super) advertise: Option<String>,
    /// `--launcher p2p`: seconds a stage that may leave this machine waits for
    /// a peer that can take it before the run fails (default 60; also
    /// `BLUT_P2P_PEER_WAIT`). The run never runs such a stage here instead.
    #[arg(long = "p2p-peer-wait", value_name = "SECS")]
    pub(super) peer_wait: Option<u64>,
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

impl MeshArgs {
    fn flags_given(&self) -> Vec<&'static str> {
        let mut given = Vec::new();
        if !self.seeds.is_empty() {
            given.push("--p2p-seed");
        }
        if self.key.is_some() {
            given.push("--p2p-key");
        }
        if self.listen.is_some() {
            given.push("--p2p-listen");
        }
        if self.advertise.is_some() {
            given.push("--p2p-advertise");
        }
        if self.peer_wait.is_some() {
            given.push("--p2p-peer-wait");
        }
        given
    }

    /// Seeds from the flags, else from `BLUT_P2P_SEEDS`.
    fn seed_specs(&self) -> Vec<String> {
        if !self.seeds.is_empty() {
            return self.seeds.clone();
        }
        env_value(SEEDS_ENV)
            .map(|v| {
                v.split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn peer_wait(&self) -> Result<std::time::Duration> {
        let secs = match (self.peer_wait, env_value(PEER_WAIT_ENV)) {
            (Some(secs), _) => secs,
            (None, Some(raw)) => raw
                .parse::<u64>()
                .map_err(|_| anyhow!("{PEER_WAIT_ENV}={raw:?} is not a whole number of seconds"))?,
            (None, None) => crate_default_peer_wait_secs(),
        };
        if secs == 0 {
            return Err(anyhow!(
                "--p2p-peer-wait must be at least 1 second: a stage needs time to find a peer"
            ));
        }
        Ok(std::time::Duration::from_secs(secs))
    }
}

#[cfg(feature = "p2p")]
fn crate_default_peer_wait_secs() -> u64 {
    crate::p2p::mesh_exec::DEFAULT_PEER_WAIT.as_secs()
}

#[cfg(not(feature = "p2p"))]
fn crate_default_peer_wait_secs() -> u64 {
    60
}

/// Refuse `--p2p-*` flags given with a launcher that would ignore them.
pub(super) fn refuse_stray(
    args: &MeshArgs,
    launch_target: crate::config::launcher::LaunchTarget,
) -> Result<()> {
    let given = args.flags_given();
    if launch_target != crate::config::launcher::LaunchTarget::P2P && !given.is_empty() {
        return Err(anyhow!(
            "{} only apply to --launcher p2p, and this run uses --launcher {}; drop them \
             or use --launcher p2p",
            given.join(", "),
            launch_target.as_str()
        ));
    }
    Ok(())
}

/// Validate a `--launcher p2p` configuration without touching the network:
/// what `--dry-run` checks, and what a real run checks before anything is
/// written. `surface` names the command, as in `check_launcher_placeable`.
#[cfg(not(feature = "p2p"))]
pub(super) fn check(_args: &MeshArgs, surface: &str) -> Result<()> {
    crate::config::launcher::check_launcher_placeable(
        crate::config::launcher::LaunchTarget::P2P,
        surface,
    )
    .map_err(|e| anyhow!("{e} (this binary was built without the `p2p` feature)"))
}

/// The mesh a `--launcher p2p` command joined. Close it when the command is
/// done: it is not shared beyond the command.
pub(super) struct MeshGuard {
    #[cfg(feature = "p2p")]
    session: std::sync::Arc<MeshSession>,
}

#[cfg(not(feature = "p2p"))]
impl MeshGuard {
    pub(super) async fn close(self) {}
}

#[cfg(not(feature = "p2p"))]
pub(super) async fn join(args: &MeshArgs, surface: &str) -> Result<MeshGuard> {
    check(args, surface)?;
    Ok(MeshGuard {})
}

#[cfg(feature = "p2p")]
pub(super) use with_p2p::*;

#[cfg(feature = "p2p")]
mod with_p2p {
    use super::*;
    use crate::p2p::crypto::KeyPair;
    use crate::p2p::mesh_exec::MeshExecutionAdapter;
    use crate::p2p::node::{GossipConfig, MeshNode, NodeCapabilities};
    use crate::p2p::objects::MeshObjects;
    use std::sync::Arc;

    /// A joined mesh: the node, its object plane and the adapter plans use.
    pub(in crate::cli) struct MeshSession {
        node: Arc<MeshNode>,
        adapter: MeshExecutionAdapter,
    }

    /// The session `recipe run` / `recipe declare --run` joined, for the
    /// launch path (`prepare_compiled_plan_launch`) to install. One per
    /// process; the command that joined it removes it when it closes.
    static ACTIVE: std::sync::Mutex<Option<Arc<MeshSession>>> = std::sync::Mutex::new(None);

    pub(in crate::cli) fn active() -> Option<Arc<MeshSession>> {
        ACTIVE.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    struct Config {
        keypair: KeyPair,
        key_path: std::path::PathBuf,
        seeds: Vec<(String, std::net::SocketAddr, [u8; 32])>,
        listen: std::net::SocketAddr,
        advertise: Option<String>,
        peer_wait: std::time::Duration,
    }

    fn config(args: &MeshArgs) -> Result<Config> {
        let key_path = match args
            .key
            .clone()
            .or_else(|| env_value(KEY_ENV).map(Into::into))
        {
            Some(path) => path,
            None => crate::cli::p2p::p2p_cli::default_key_path()?,
        };
        if !key_path.is_file() {
            return Err(anyhow!(
                "--launcher p2p needs this machine's mesh identity, and {} does not exist: \
                 run `blut p2p keys generate` (or pass --p2p-key <file>). Nothing ran",
                key_path.display()
            ));
        }
        let keypair = crate::cli::p2p::p2p_cli::load_keypair(&key_path)?;
        let specs = args.seed_specs();
        if specs.is_empty() {
            return Err(anyhow!(
                "--launcher p2p needs a mesh to join: give at least one --p2p-seed \
                 <addr@pubkey> (or {SEEDS_ENV}), where pubkey is the line `blut p2p keys \
                 show` prints on that peer. Nothing ran"
            ));
        }
        let mut seeds = Vec::new();
        for spec in specs {
            let (addr, pubkey) = crate::cli::p2p::parse_seed(&spec)
                .map_err(|e| anyhow!("--p2p-seed '{spec}': {e}"))?;
            seeds.push((spec, addr, pubkey));
        }
        let listen_spec = args
            .listen
            .clone()
            .or_else(|| env_value(LISTEN_ENV))
            .unwrap_or_else(|| "0.0.0.0:0".into());
        let listen = listen_spec
            .parse()
            .map_err(|e| anyhow!("--p2p-listen '{listen_spec}': {e}"))?;
        Ok(Config {
            keypair,
            key_path,
            seeds,
            listen,
            advertise: args.advertise.clone().or_else(|| env_value(ADVERTISE_ENV)),
            peer_wait: args.peer_wait()?,
        })
    }

    pub(in crate::cli) fn check(args: &MeshArgs, _surface: &str) -> Result<()> {
        config(args).map(|_| ())
    }

    /// Join the mesh: bind a scheduler-only node, dial the seeds, let gossip
    /// settle, and make the session the one launches install. Refuses before
    /// anything runs when no seed answers.
    pub(in crate::cli) async fn join(args: &MeshArgs, _surface: &str) -> Result<MeshGuard> {
        let cfg = config(args)?;
        let keypair = Arc::new(cfg.keypair);
        let peers = crate::p2p::registry::PeerRegistry::load(
            &crate::cli::p2p::p2p_cli::default_registry_path()?,
        )
        .map_err(|e| anyhow!("load peer registry: {e}"))?;
        let store = crate::cli::p2p::open_node_store(None)?;
        let objects = Arc::new(MeshObjects::new(
            store,
            paths::data_dir()?.join("p2p").join("chunks"),
        ));
        let node = MeshNode::bind(
            cfg.listen,
            keypair.clone(),
            peers,
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
            false,
        )
        .await
        .map_err(|e| anyhow!("--launcher p2p: start the mesh node: {e}"))?;
        node.set_objects(objects.clone());
        if let Some(spec) = &cfg.advertise {
            node.set_advertise_addr(crate::cli::p2p::resolve_advertise(spec).await?);
        }
        node.start_gossip(GossipConfig::default());
        eprintln!(
            "mesh   {} (identity {})",
            node.node_id().short(),
            cfg.key_path.display()
        );
        let mut reached = 0usize;
        let mut failures = Vec::new();
        for (spec, addr, pubkey) in &cfg.seeds {
            if let Err(e) = node.add_seed(*addr, *pubkey).await {
                failures.push(format!("{spec}: {e}"));
                continue;
            }
            match tokio::time::timeout(
                std::time::Duration::from_secs(10),
                node.connect_to(*addr, *pubkey),
            )
            .await
            {
                Ok(Ok(_)) => reached += 1,
                Ok(Err(e)) => failures.push(format!("{addr}: {e}")),
                Err(_) => failures.push(format!("{addr}: dial timed out after 10s")),
            }
        }
        if reached == 0 {
            node.shutdown(crate::p2p::transport::CLOSE_DRAIN_TIMEOUT)
                .await;
            return Err(anyhow!(
                "--launcher p2p could not reach any seed ({}). Nothing ran",
                failures.join("; ")
            ));
        }
        for failure in &failures {
            eprintln!("warning: seed {failure}");
        }
        crate::cli::p2p::discover(&node, std::time::Duration::from_secs(3)).await;
        eprintln!(
            "{}",
            crate::cli::p2p::mesh_view_line(&node.mesh_view().await)
        );
        let policy: Arc<dyn crate::p2p::dispatch::DispatchPolicy> =
            Arc::new(crate::p2p::dispatch::DefaultDispatchPolicy::new(
                crate::p2p::trust::DispatchMatrix::default(),
            ));
        let adapter = MeshExecutionAdapter::new(node.clone(), objects, keypair, policy)
            .with_peer_wait(cfg.peer_wait);
        let session = Arc::new(MeshSession { node, adapter });
        *ACTIVE.lock().unwrap_or_else(|p| p.into_inner()) = Some(session.clone());
        Ok(MeshGuard { session })
    }

    impl MeshGuard {
        /// Leave the mesh: peers see an orderly close, and a worker still
        /// running one of this command's stages stops it.
        pub(in crate::cli) async fn close(self) {
            {
                let mut active = ACTIVE.lock().unwrap_or_else(|p| p.into_inner());
                if active
                    .as_ref()
                    .is_some_and(|s| Arc::ptr_eq(s, &self.session))
                {
                    *active = None;
                }
            }
            self.session
                .node
                .shutdown(crate::p2p::transport::CLOSE_DRAIN_TIMEOUT)
                .await;
        }
    }

    /// Install the joined mesh on a launch's executor context.
    pub(in crate::cli) fn attach(
        ctx: crate::framework::ExecCtx,
    ) -> Result<crate::framework::ExecCtx> {
        let session = active().ok_or_else(|| {
            anyhow!(
                "--launcher p2p reached the launch path without a joined mesh; only `recipe \
                 run` and `recipe declare --run` join one. Nothing ran"
            )
        })?;
        let policy: Arc<dyn crate::p2p::dispatch::DispatchPolicy> =
            Arc::new(crate::p2p::dispatch::DefaultDispatchPolicy::new(
                crate::p2p::trust::DispatchMatrix::default(),
            ));
        Ok(ctx.with_execution_adapter(policy, Arc::new(session.adapter.clone())))
    }
}
