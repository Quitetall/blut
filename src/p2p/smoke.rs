//! Built-in P2P connectivity probe: the `p2p-echo` smoke stage + a one-shot
//! coordinator-side smoke dispatch.
//!
//! `p2p-echo` is a tiny, deterministic, dependency-free stage — it reads a text
//! file artifact and writes its uppercased contents. It exists so that
//! `blut p2p serve --smoke-stage` can prove the **full data plane** end-to-end
//! between two real machines (bundle → QUIC blob → remote execute → bundle back
//! → content-hash verify) without a heavyweight cookbook stage or real data.
//! It is always-dispatchable (added to `DefaultDispatchPolicy`'s default set).
//!
//! The dispatch half ([`smoke_dispatch_once`]) mirrors the proven loopback path
//! in `tests/p2p_integration.rs::peer_runs_real_stage_end_to_end`, but with the
//! bind/dial roles matching the CLI: the **coordinator binds** (`serve`) and the
//! **worker dials** (`connect`). `dispatch_to_peer` / `run_peer_loop` open and
//! accept uni-streams symmetrically, so they pair regardless of who dialed. We
//! drive a **raw `P2pServer`** here (not `Coordinator`) on purpose: the full
//! `Coordinator` spawns a background `handle_peer` reader that would race
//! `dispatch_to_peer`'s own `recv_result` on the same connection.

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::TrainError;
use crate::framework::artifact::{Artifact, ContentHash};
use crate::framework::cookbook::{Cookbook, Registry};
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::stage::{ErasedArtifact, ErasedStageCtor, Stage, StageContext};
use crate::recipes::recipe::RecipeDef;

/// The stage name of the built-in connectivity probe.
pub const SMOKE_STAGE: &str = "p2p-echo";

/// A file-backed text artifact: one UTF-8 file on disk (a file-backed handle,
/// like every BLUT artifact — the bytes ride the bundle blob, not this struct).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmokeText {
    pub path: PathBuf,
    pub content_hash: ContentHash,
}

impl Artifact for SmokeText {
    const KIND: &'static str = "p2p.smoke.text";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SmokeArgs {}

/// `p2p-echo`: read the input text file, uppercase it, write the output. The
/// canonical connectivity probe — deterministic, Public, CPU-only.
pub struct SmokeEcho;

#[async_trait]
impl Stage for SmokeEcho {
    const NAME: &'static str = SMOKE_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = SmokeText;
    type Output = SmokeText;
    type Args = SmokeArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        _args: &Self::Args,
    ) -> Result<Self::Output, StageError> {
        let body = std::fs::read_to_string(&input.path)
            .map_err(|e| StageError::Backend(anyhow::anyhow!("read smoke input: {e}")))?;
        let out = ctx.stage_dir.join("p2p-echo-out.txt");
        std::fs::write(&out, body.to_uppercase().as_bytes())
            .map_err(|e| StageError::Backend(anyhow::anyhow!("write smoke output: {e}")))?;
        let content_hash = ContentHash::hash_file(&out)
            .map_err(|e| StageError::Backend(anyhow::anyhow!("hash smoke output: {e}")))?;
        Ok(SmokeText {
            content_hash,
            path: out,
        })
    }
}

/// Cookbook exposing only the `p2p-echo` smoke stage.
pub struct SmokeCookbook;

impl Cookbook for SmokeCookbook {
    fn name(&self) -> &'static str {
        "p2p-smoke"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static S: &[(&str, ErasedStageCtor)] = &[(SMOKE_STAGE, || std::sync::Arc::new(SmokeEcho))];
        S
    }
}

/// Register the built-in smoke cookbook into `reg`. Call once per process before
/// any `blut p2p` action so both the dispatching coordinator and the executing
/// worker can resolve `p2p-echo` via `find_erased_stage`.
pub fn register(reg: &mut Registry) {
    reg.register(Box::new(SmokeCookbook));
}

/// The expected output hash of `p2p-echo` for `input_text`: the hash of the
/// uppercased bytes. The coordinator needs this up-front — `dispatch_to_peer`
/// verifies the returned bundle against it (content-addressed, fail-closed).
///
/// Uses the same `str::to_uppercase` the stage uses, so it matches byte-for-byte
/// regardless of the input's Unicode (the stage and this share one transform).
pub fn expected_echo_hash(input_text: &str) -> ContentHash {
    ContentHash::of_bytes(input_text.to_uppercase().as_bytes())
}

/// Drive ONE smoke dispatch over an already-accepted peer connection: bundle a
/// tiny text artifact, ship it to the worker running `p2p-echo`, receive +
/// content-verify the uppercased result, and return it as a string.
///
/// `conn` is the server-end connection from `P2pServer::accept_peer`; `peer_id`
/// is the worker that produced it. `coord_kp` signs the task; `reg` must have the
/// smoke stage registered (see [`register`]). Fails closed on every gate the
/// bundle path enforces.
#[allow(clippy::too_many_arguments)]
pub async fn smoke_dispatch_once(
    server: &crate::p2p::transport::P2pServer,
    conn: &quinn::Connection,
    coord_kp: &crate::p2p::crypto::KeyPair,
    reg: &Registry,
    peer_id: &crate::p2p::PeerId,
    stage_name: &str,
    input_text: &str,
    timeout_secs: u64,
) -> Result<String, TrainError> {
    // The peer registers during the handshake; poll briefly for its PeerInfo.
    let mut peer_info = None;
    for _ in 0..40 {
        if let Some(info) = server.peers.read().await.get(peer_id).cloned() {
            peer_info = Some(info);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let peer_info =
        peer_info.ok_or_else(|| TrainError::other("peer did not register within 2s"))?;

    // Produce the input artifact on the coordinator's disk (its src_root).
    let src_root =
        tempfile::tempdir().map_err(|e| TrainError::other(format!("smoke src_root: {e}")))?;
    let in_path = src_root.path().join("p2p-echo-in.txt");
    std::fs::write(&in_path, input_text.as_bytes())
        .map_err(|e| TrainError::other(format!("write smoke input: {e}")))?;
    let input_hash = ContentHash::hash_file(&in_path)
        .map_err(|e| TrainError::other(format!("hash smoke input: {e}")))?;
    let input = SmokeText {
        content_hash: input_hash,
        path: in_path,
    };
    let input_erased = ErasedArtifact::from_typed(&input)
        .map_err(|e| TrainError::other(format!("erase smoke input: {e}")))?;
    let expected = expected_echo_hash(input_text);

    let out_dir =
        tempfile::tempdir().map_err(|e| TrainError::other(format!("smoke out_dir: {e}")))?;
    let dispatched = crate::p2p::peer_exec::dispatch_to_peer(
        conn,
        coord_kp,
        &peer_info,
        reg,
        "p2p-smoke-1",
        stage_name,
        input_erased,
        src_root.path(),
        serde_json::json!({}),
        input_hash,
        expected,
        crate::p2p::trust::DataClass::Public,
        timeout_secs,
        out_dir.path(),
    )
    .await?;

    // The output is a local handle; read its file (already hash-verified).
    let out: SmokeText = dispatched
        .output
        .into_typed()
        .map_err(|e| TrainError::other(format!("decode smoke output: {e}")))?;
    let body = std::fs::read_to_string(&out.path)
        .map_err(|e| TrainError::other(format!("read smoke output: {e}")))?;
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_echo_hash_matches_uppercase() {
        // The stage hashes the file it writes; expected_echo_hash hashes the
        // uppercased bytes directly. They must agree (proven end-to-end by the
        // integration test; this pins the pure transform).
        let h = expected_echo_hash("hello p2p world");
        assert_eq!(h, ContentHash::of_bytes(b"HELLO P2P WORLD"));
    }

    #[test]
    fn smoke_cookbook_exposes_echo() {
        let mut reg = Registry::new();
        register(&mut reg);
        assert!(
            reg.find_erased_stage(SMOKE_STAGE).is_some(),
            "p2p-echo must resolve after register()"
        );
    }
}
