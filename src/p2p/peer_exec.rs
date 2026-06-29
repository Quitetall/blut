// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Peer-side task execution loop — the worker half of the P2P data plane.
//!
//! A peer connects to a coordinator (via [`crate::p2p::transport::P2pClient`]),
//! then runs this loop: receive a signed [`TaskManifest`], verify it, pull the
//! input artifact bundle, materialize + verify it locally, run the dispatched
//! stage, bundle the output, and ship it back. Everything the loop needs from
//! the lower layers already exists:
//!
//! - [`crate::p2p::bundle`] — `BundleManifest` (un)packing + the four
//!   fail-closed verification gates.
//! - [`crate::p2p::transport`] — `recv_task` / `send_result` / the blob
//!   side-stream (`recv_blob` / `send_blob`).
//! - [`crate::framework::cookbook::Registry::find_erased_stage`] — resolve a
//!   stage constructor by name; the peer hosts the cookbook.
//! - [`crate::framework::stage::StageContext::for_peer`] — an isolated context
//!   for running ONE stage outside an executor.
//!
//! The peer must be told the coordinator's keys (Ed25519 verifying key to check
//! the manifest signature, X25519 public key to seal the result) — the CLI
//! supplies them from `blut p2p connect <addr> --coordinator-pubkey ...`.

use std::path::PathBuf;
use std::sync::Arc;

use quinn::Connection as QuinnConnection;

use crate::error::TrainError;
use crate::framework::artifact::ContentHash;
use crate::framework::cookbook::Registry;
use crate::framework::stage::{ErasedArtifact, StageContext};
use crate::framework::cache::CacheHandle;
use crate::p2p::bundle::{self, BlobDir};
use crate::p2p::crypto::{self, KeyPair};
use crate::p2p::dispatch::DispatchPolicy;
use crate::p2p::task::{TaskManifest, TaskResult};
use crate::p2p::transport::{self, P2pClient, MAX_BLOB_SIZE};
use crate::p2p::PeerId;

/// The coordinator's public identity a peer needs to trust a dispatch and reply.
pub struct CoordinatorKeys {
    /// Ed25519 — verify the manifest signature.
    pub verifying: ed25519_dalek::VerifyingKey,
    /// X25519 — seal the result bundle back to the coordinator.
    pub x25519_pub: x25519_dalek::PublicKey,
}

/// Run the peer execution loop until the connection closes. Each iteration
/// handles one dispatched task; an error on a single task is reported back to
/// the coordinator (`send_error`) and the loop continues to the next.
///
/// `work_root` is where the peer materializes inputs + runs stages (one subdir
/// per task). `keypair` is the peer's own identity (to decrypt inputs sealed to
/// it). `policy` gates which stages this peer will run.
pub async fn run_peer_loop(
    conn: &QuinnConnection,
    keypair: &KeyPair,
    coordinator: &CoordinatorKeys,
    registry: &Registry,
    policy: &dyn DispatchPolicy,
    work_root: &std::path::Path,
) -> Result<(), TrainError> {
    loop {
        // recv_task surfaces a closed connection as an Err — that ends the loop.
        let task = match P2pClient::recv_task(conn).await {
            Ok(t) => t,
            Err(e) => {
                tracing::debug!("peer loop: recv_task ended ({e})");
                return Ok(());
            }
        };
        let task_id = task.task_id.clone();
        match execute_one(conn, keypair, coordinator, registry, policy, work_root, task).await {
            Ok(()) => {}
            Err(e) => {
                tracing::warn!("peer task {task_id} failed: {e}");
                // Best-effort: tell the coordinator so it can re-dispatch.
                let _ = P2pClient::send_error(conn, &format!("{e}")).await;
            }
        }
    }
}

/// Handle a single dispatched task end-to-end.
async fn execute_one(
    conn: &QuinnConnection,
    keypair: &KeyPair,
    coordinator: &CoordinatorKeys,
    registry: &Registry,
    policy: &dyn DispatchPolicy,
    work_root: &std::path::Path,
    task: TaskManifest,
) -> Result<(), TrainError> {
    // 1. Verify the coordinator's Ed25519 signature over the manifest.
    if !crypto::verify(&coordinator.verifying, &task.sign_payload(), &task.signature) {
        return Err(TrainError::other("task manifest signature invalid"));
    }
    // The signer must be the coordinator we connected to.
    if task.coordinator_id != PeerId::from_pubkey(&coordinator.verifying) {
        return Err(TrainError::other("task coordinator_id != connected coordinator"));
    }

    // task_id is network-controlled and becomes a path component below — reject
    // anything that isn't a flat, safe slug so a malicious coordinator can't
    // escape `work_root` via `../` or an absolute path.
    if !is_safe_task_id(&task.task_id) {
        return Err(TrainError::other(format!(
            "unsafe task_id '{}': expected [A-Za-z0-9._-]+ (no '..')",
            task.task_id
        )));
    }

    // 2. Policy gate: refuse a non-dispatchable stage (training never leaves home).
    if !policy.is_dispatchable(&task.stage_name) {
        return Err(TrainError::other(format!(
            "stage '{}' is not dispatchable",
            task.stage_name
        )));
    }

    // 3. Resolve the stage constructor (the peer hosts the cookbook).
    let ctor = registry.find_erased_stage(&task.stage_name).ok_or_else(|| {
        TrainError::other(format!("unknown stage '{}'", task.stage_name))
    })?;
    let stage = ctor();

    // 4. Decode the input BundleManifest from the encrypted_input slot. Public
    //    data may ride in the clear (bincode in the ciphertext field with an
    //    empty seal); non-Public is sealed to this peer's X25519 key.
    let input_manifest = match &task.encrypted_input {
        Some(payload) => {
            let bytes = keypair.decrypt(payload)?;
            bincode::deserialize::<bundle::BundleManifest>(&bytes)
                .map_err(|e| TrainError::other(format!("decode input BundleManifest: {e}")))?
        }
        None => {
            return Err(TrainError::other(
                "task has no encrypted_input — shared-FS dispatch not supported on this peer",
            ));
        }
    };

    // 5. Per-task work dir + a job-local cache. (task_id validated safe above.)
    let stage_dir = work_root.join(&task.task_id);
    std::fs::create_dir_all(&stage_dir)
        .map_err(|e| TrainError::other(format!("create peer stage_dir: {e}")))?;
    let cache = Arc::new(CacheHandle::job_local(stage_dir.join(".cache")));
    // Isolated: job_dir == stage_dir for a single dispatched stage. cache_key =
    // the input hash (unique per task) so concurrent peer tasks don't collide if
    // a stage does a content-addressed cache lookup.
    let ctx = StageContext::for_peer(stage_dir.clone(), stage_dir.clone(), cache, task.input_hash);

    // 6. Receive the input blob side-stream and unbundle into stage_dir. The
    //    bundle layer runs the four fail-closed gates against task.input_hash.
    let pack = transport::recv_blob(conn, &task.task_id, BlobDir::Input, MAX_BLOB_SIZE).await?;
    let input: ErasedArtifact = bundle::unbundle(
        &*stage,
        &input_manifest,
        &pack,
        &stage_dir,
        &task.input_hash,
        BlobDir::Input,
    )
    .map_err(|e| TrainError::other(format!("unbundle input: {e}")))?;

    // 7. Run the stage under the coordinator's wall-clock deadline. Its run()
    //    opens primary_path(), which now exists locally. A runaway stage can't
    //    block the peer loop forever — timeout_secs caps it.
    let started = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(task.timeout_secs.max(1));
    let output = tokio::time::timeout(timeout, stage.run_erased(&ctx, input, task.args.clone()))
        .await
        .map_err(|_| TrainError::other(format!(
            "stage '{}' exceeded timeout_secs {}", task.stage_name, task.timeout_secs
        )))?
        .map_err(|e| TrainError::other(format!("stage run failed: {e}")))?;
    let wall_time_ms = started.elapsed().as_millis() as u64;

    // 8. Bundle the output (rooted at the peer's stage_dir) + ship it back. The
    //    coordinator re-verifies against task.expected_output_hash.
    let (out_manifest, out_pack) = bundle::bundle(
        &*stage,
        output,
        &stage_dir,
        BlobDir::Output,
        &task.expected_output_hash,
    )
    .map_err(|e| TrainError::other(format!("bundle output: {e}")))?;

    // Seal the output manifest to the coordinator's X25519 key.
    let manifest_bytes = bincode::serialize(&out_manifest)
        .map_err(|e| TrainError::other(format!("serialize output manifest: {e}")))?;
    let encrypted_output = crypto::encrypt(&manifest_bytes, &coordinator.x25519_pub);

    let mut result = TaskResult {
        task_id: task.task_id.clone(),
        peer_id: PeerId::from_pubkey(&keypair.verifying),
        output_hash: out_manifest.content_hash,
        encrypted_output: Some(encrypted_output),
        wall_time_ms,
        signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
    };
    result.signature = keypair.sign(&result.sign_payload());

    P2pClient::send_result(conn, &result).await?;
    transport::send_blob(conn, &task.task_id, BlobDir::Output, &out_pack).await?;
    Ok(())
}

/// A `task_id` is safe to use as a single path component: non-empty, only
/// `[A-Za-z0-9._-]`, and not `.`/`..`. Network-controlled, so this gates the
/// peer's `work_root.join(task_id)` against directory traversal.
fn is_safe_task_id(id: &str) -> bool {
    !id.is_empty()
        && id != "."
        && id != ".."
        && id.len() <= 128
        && id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// The verified output of a dispatched task: the rebased local artifact handle
/// plus the peer that produced it.
pub struct DispatchedOutput {
    pub output: ErasedArtifact,
    pub peer_id: PeerId,
}

/// Coordinator side of one dispatch: bundle `input` (rooted at its producing
/// `src_root`), seal + sign a `TaskManifest`, send it + the input blob to the
/// peer over `conn`, then receive the result + output blob, unbundle it into
/// `out_stage_dir`, and verify it against `expected_output_hash`.
///
/// This is the symmetric coordinator half of [`run_peer_loop`]; the CLI
/// `blut p2p dispatch` drives it. (The executor's `DispatchSubmitter` seam can
/// adopt this once it threads each node's producing stage_dir; until then the
/// CLI path is the live-validation route.)
#[allow(clippy::too_many_arguments)]
pub async fn dispatch_to_peer(
    conn: &QuinnConnection,
    coordinator_kp: &KeyPair,
    peer: &crate::p2p::PeerInfo,
    registry: &Registry,
    task_id: &str,
    stage_name: &str,
    input: ErasedArtifact,
    src_root: &std::path::Path,
    args: serde_json::Value,
    input_hash: ContentHash,
    expected_output_hash: ContentHash,
    data_class: crate::p2p::DataClass,
    timeout_secs: u64,
    out_stage_dir: &std::path::Path,
) -> Result<DispatchedOutput, TrainError> {
    if !is_safe_task_id(task_id) {
        return Err(TrainError::other(format!("unsafe task_id '{task_id}'")));
    }
    let stage = registry
        .find_erased_stage(stage_name)
        .ok_or_else(|| TrainError::other(format!("unknown stage '{stage_name}'")))?(
    );

    // 1. Bundle the input rooted at its producing stage_dir.
    let (in_manifest, in_pack) =
        bundle::bundle(&*stage, input, src_root, BlobDir::Input, &input_hash)
            .map_err(|e| TrainError::other(format!("bundle input: {e}")))?;

    // 2. Seal the bundle manifest to the PEER's X25519 key + build the task.
    let manifest_bytes = bincode::serialize(&in_manifest)
        .map_err(|e| TrainError::other(format!("serialize input manifest: {e}")))?;
    let encrypted_input = crypto::encrypt(&manifest_bytes, &peer.x25519_pub);
    let args_hash = ContentHash::of_bytes(
        &serde_json::to_vec(&args).map_err(|e| TrainError::other(format!("args hash: {e}")))?,
    );
    let mut task = TaskManifest {
        task_id: task_id.to_string(),
        coordinator_id: PeerId::from_pubkey(&coordinator_kp.verifying),
        stage_name: stage_name.to_string(),
        stage_schema: stage.schema(),
        input_hash,
        args_hash,
        expected_output_hash,
        args,
        resources: crate::p2p::task::ResourceRequest::default(),
        data_class,
        timeout_secs,
        encrypted_input: Some(encrypted_input),
        signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
    };
    task.signature = coordinator_kp.sign(&task.sign_payload());

    // 3. Send task + input blob.
    transport::P2pServer::send_task(conn, &task).await?;
    transport::send_blob(conn, task_id, BlobDir::Input, &in_pack).await?;

    // 4. Receive the result, then the output blob, and verify. Bound by the
    //    peer's deadline + slack so a hung/stalled peer can't block the
    //    coordinator forever (symmetric with the peer-side timeout enforcement).
    let deadline = std::time::Duration::from_secs(timeout_secs.max(1) + 30);
    let result = tokio::time::timeout(deadline, transport::P2pServer::recv_result(conn))
        .await
        .map_err(|_| TrainError::other("timed out waiting for peer result"))??;
    if result.task_id != task_id {
        return Err(TrainError::other("result task_id mismatch"));
    }
    if !crypto::verify(&peer.pubkey, &result.sign_payload(), &result.signature) {
        return Err(TrainError::other("result signature invalid"));
    }
    let out_payload = result
        .encrypted_output
        .as_ref()
        .ok_or_else(|| TrainError::other("result has no encrypted_output"))?;
    let out_bytes = coordinator_kp.decrypt(out_payload)?;
    let out_manifest = bincode::deserialize::<bundle::BundleManifest>(&out_bytes)
        .map_err(|e| TrainError::other(format!("decode output manifest: {e}")))?;
    let out_pack =
        transport::recv_blob(conn, task_id, BlobDir::Output, MAX_BLOB_SIZE).await?;
    let output = bundle::unbundle(
        &*stage,
        &out_manifest,
        &out_pack,
        out_stage_dir,
        &expected_output_hash,
        BlobDir::Output,
    )
    .map_err(|e| TrainError::other(format!("unbundle output: {e}")))?;

    Ok(DispatchedOutput {
        output,
        peer_id: result.peer_id,
    })
}

/// Default per-task work root for a peer: `<data_dir>/blut-p2p-peer/` (or
/// `$TMPDIR/blut-p2p-peer/` if the data dir isn't configured). The peer
/// materializes each task's input + output under a `<task_id>` subdir here.
pub fn default_work_root() -> PathBuf {
    crate::paths::data_dir()
        .map(|d| d.join("blut-p2p-peer"))
        .unwrap_or_else(|_| std::env::temp_dir().join("blut-p2p-peer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_task_id_rejects_traversal() {
        // Network-controlled task_id must not escape work_root.
        assert!(!is_safe_task_id("../../etc/evil"));
        assert!(!is_safe_task_id("/abs/path"));
        assert!(!is_safe_task_id(".."));
        assert!(!is_safe_task_id("."));
        assert!(!is_safe_task_id(""));
        assert!(!is_safe_task_id("has/slash"));
        assert!(!is_safe_task_id("has\0nul"));
        assert!(!is_safe_task_id(&"x".repeat(129)));
        // Legitimate ids pass.
        assert!(is_safe_task_id("blut-job7-node3"));
        assert!(is_safe_task_id("task-e2e-1"));
        assert!(is_safe_task_id("a.b_c-1"));
    }
}
