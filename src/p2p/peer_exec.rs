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
//! - [`mod@crate::p2p::bundle`] — `BundleManifest` (un)packing + the four
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
use crate::framework::artifact::{ContentHash, ContentId, InvocationKey};
use crate::framework::artifact_store::StoredArtifact;
use crate::framework::cache::CacheHandle;
use crate::framework::cookbook::Registry;
use crate::framework::execution::{
    Assignment, DataClassification, ExecutionDeadline, ExecutionFailure, ExecutionFailureKind,
    ExecutionLifecycle, ExecutionPhase, ExecutionRequest,
};
use crate::framework::stage::{ErasedArtifact, StageContext};
use crate::p2p::PeerId;
use crate::p2p::bundle::{self, BlobDir};
use crate::p2p::crypto::{self, KeyPair};
use crate::p2p::dispatch::{DispatchPolicy, DispatchVerdict};
use crate::p2p::task::{TaskManifest, TaskResult};
use crate::p2p::transport::{self, MAX_BLOB_SIZE, P2pClient};
use crate::p2p::trust::DataClass;

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
        match execute_one(
            conn,
            keypair,
            coordinator,
            registry,
            policy,
            work_root,
            task,
        )
        .await
        {
            Ok(()) => {}
            Err(failure) if failure.code == "EXECUTION_CANCELLED" => {
                tracing::info!("peer task {task_id} cancelled: {failure}");
                let _ = P2pClient::send_execution_cancelled(conn, &task_id, &failure.message).await;
            }
            Err(failure) => {
                tracing::warn!("peer task {task_id} failed: {failure}");
                // Best-effort: tell the coordinator so it can re-dispatch.
                let _ = P2pClient::send_execution_failure(conn, &task_id, &failure).await;
            }
        }
    }
}

/// RAII guard: removes the peer's per-task work directory (`stage_dir`) when
/// dropped — i.e. on every exit path of [`execute_one`], success or error.
/// Without this, a coordinator dispatching repeated tasks accumulates one
/// leaked directory (containing full artifact bytes: imported input, stage
/// outputs, job-local cache) per task, unboundedly. Best-effort: logs on
/// failure rather than propagating — cleanup must never turn an otherwise-
/// successful task into a failure.
struct StageDirGuard(PathBuf);

impl Drop for StageDirGuard {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                "peer stage_dir cleanup failed for {}: {e}",
                self.0.display()
            );
        }
    }
}

/// Seal a blob's plaintext bundle pack for `recipient`, then frame the result
/// as opaque bytes for [`transport::send_blob`] (whose own doc comment says
/// encryption, if any, is the caller's job — that layer only frames/chunks).
/// Uses the SAME AES-256-GCM hybrid primitive as the small `BundleManifest`
/// (`crypto::encrypt`, already used for `encrypted_input`/`encrypted_output`);
/// bulk artifact bytes previously rode `send_blob` in plaintext even for
/// `DataClass::Restricted` (real clinical EEG/PHI corpora) — bundle.rs's doc
/// comment claims the transport layer owns per-chunk encryption, which it does
/// not (see transport.rs's own doc comment on `send_blob`/`recv_blob`).
///
/// One `crypto::encrypt` call per blob is safe to repeat across many blob
/// sends: each call mints a FRESH random AES-256 key and a fresh nonce (see
/// `crypto::encrypt`'s implementation), so there is no nonce reuse across
/// calls even when sealing to the same recipient repeatedly.
fn seal_blob(pack: &[u8], recipient: &x25519_dalek::PublicKey) -> Result<Vec<u8>, TrainError> {
    let sealed = crypto::encrypt(pack, recipient);
    bincode::serialize(&sealed)
        .map_err(|e| TrainError::other(format!("serialize encrypted blob: {e}")))
}

/// Inverse of [`seal_blob`]: decode the framed [`crypto::EncryptedPayload`]
/// returned by [`transport::recv_blob`] and decrypt it with `kp`'s own X25519
/// secret.
fn open_blob(bytes: &[u8], kp: &KeyPair) -> Result<Vec<u8>, TrainError> {
    let sealed: crypto::EncryptedPayload = bincode::deserialize(bytes)
        .map_err(|e| TrainError::other(format!("decode encrypted blob: {e}")))?;
    kp.decrypt(&sealed)
}

/// Handle a single dispatched task end-to-end.
// The peer emits this rich, serializable failure over the wire; boxing here
// would add an allocation without reducing any caller-facing interface cost.
#[allow(clippy::result_large_err)]
async fn execute_one(
    conn: &QuinnConnection,
    keypair: &KeyPair,
    coordinator: &CoordinatorKeys,
    registry: &Registry,
    policy: &dyn DispatchPolicy,
    work_root: &std::path::Path,
    task: TaskManifest,
) -> Result<(), ExecutionFailure> {
    if task.protocol_version != crate::p2p::task::TASK_PROTOCOL_VERSION {
        return Err(ExecutionFailure::protocol(format!(
            "task protocol v{} unsupported (want v{})",
            task.protocol_version,
            crate::p2p::task::TASK_PROTOCOL_VERSION
        )));
    }
    // 1. Verify the coordinator's Ed25519 signature over the manifest.
    if !crypto::verify(
        &coordinator.verifying,
        &task.sign_payload(),
        &task.signature,
    ) {
        return Err(ExecutionFailure::protocol(
            "task manifest signature invalid",
        ));
    }
    // The signer must be the coordinator we connected to.
    if task.coordinator_id != PeerId::from_pubkey(&coordinator.verifying) {
        return Err(ExecutionFailure::protocol(
            "task coordinator_id != connected coordinator",
        ));
    }
    // Defense in depth at the receiving mesh boundary. The coordinator's
    // DispatchMatrix already refuses Restricted tasks, but a malicious or stale
    // coordinator could send a correctly signed manifest directly. Through M5,
    // Restricted data is node-local regardless of peer trust or encryption.
    if task.data_class == DataClass::Restricted {
        return Err(ExecutionFailure::protocol(
            "remote task DENIED: Restricted data is node-local through M5 (ADR 0096)",
        ));
    }
    // Belt-and-suspenders: sign_payload() now covers `args` directly (a prior
    // version only signed args_hash and never checked it against the received
    // args), but reconcile args_hash explicitly too in case that ever regresses.
    task.verify_args(&task.args)
        .map_err(|e| ExecutionFailure::protocol(e.to_string()))?;

    // task_id is network-controlled and becomes a path component below — reject
    // anything that isn't a flat, safe slug so a malicious coordinator can't
    // escape `work_root` via `../` or an absolute path.
    if !is_safe_task_id(&task.task_id) {
        return Err(ExecutionFailure::protocol(format!(
            "unsafe task_id '{}': expected [A-Za-z0-9._-]+ (no '..')",
            task.task_id
        )));
    }

    // 2. Policy gate: refuse a non-dispatchable stage (training never leaves home).
    if !policy.is_dispatchable(&task.stage_name) {
        return Err(ExecutionFailure::protocol(format!(
            "stage '{}' is not dispatchable",
            task.stage_name
        )));
    }

    // 3. Resolve the stage constructor (the peer hosts the cookbook).
    let ctor = registry
        .find_erased_stage(&task.stage_name)
        .ok_or_else(|| {
            ExecutionFailure::protocol(format!("unknown stage '{}'", task.stage_name))
        })?;
    let stage = ctor();

    // 4. Decode the input BundleManifest from the encrypted_input slot. Public
    //    data may ride in the clear (bincode in the ciphertext field with an
    //    empty seal); non-Public is sealed to this peer's X25519 key.
    let input_manifest = match &task.encrypted_input {
        Some(payload) => {
            let bytes = keypair
                .decrypt(payload)
                .map_err(|e| ExecutionFailure::artifact(format!("decrypt input manifest: {e}")))?;
            bincode::deserialize::<bundle::BundleManifest>(&bytes).map_err(|e| {
                ExecutionFailure::artifact(format!("decode input BundleManifest: {e}"))
            })?
        }
        None => {
            return Err(ExecutionFailure::protocol(
                "task has no encrypted_input — shared-FS dispatch not supported on this peer",
            ));
        }
    };

    // 4b. Fail-fast hash-binding check, BEFORE `recv_blob` buffers the (up to
    //     MAX_BLOB_SIZE = 16 GiB) blob bytes. `bundle::unbundle` re-checks this
    //     exact comparison later (defense in depth, on the rebased path); doing
    //     it here too means a mismatched/malicious blob from a low-trust peer
    //     is never fully received into memory in the first place.
    if input_manifest.content_id != task.input_content_id {
        return Err(ExecutionFailure::artifact(format!(
            "identity binding: artifact {} != signed input identity {} — rejecting before blob receive",
            input_manifest.content_id.to_hex(),
            task.input_content_id.to_hex(),
        )));
    }

    // 5. Per-task work dir + a job-local cache. (task_id validated safe above.)
    let stage_dir = work_root.join(&task.task_id);
    std::fs::create_dir_all(&stage_dir).map_err(|e| {
        ExecutionFailure::new(
            ExecutionFailureKind::Storage,
            "EXECUTION_STAGE_DIR",
            format!("create peer stage_dir: {e}"),
        )
    })?;
    // RAII cleanup: `stage_dir` holds the imported input, the job-local cache,
    // and the stage's own outputs. Nothing past this function needs any of it
    // on disk — `run_peer_loop` only reads `task_id` (a String it already
    // cloned) for logging, and by the time this function returns, the output
    // bytes it produced are already fully buffered in memory and sent over the
    // wire (step 8 below). Removed on EVERY exit path: the many early `?`
    // returns below and the success path alike.
    let _stage_dir_guard = StageDirGuard(stage_dir.clone());
    let cache = Arc::new(CacheHandle::job_local(stage_dir.join(".cache")));
    // Isolated: job_dir == stage_dir for a single dispatched stage. cache_key =
    // the input hash (unique per task) so concurrent peer tasks don't collide if
    // a stage does a content-addressed cache lookup.
    let ctx = StageContext::for_peer(
        stage_dir.clone(),
        stage_dir.clone(),
        cache,
        task.invocation_key,
    );

    // 6. Receive the input blob side-stream (sealed to this peer's X25519 key —
    //    see `seal_blob`) and unbundle into stage_dir. The bundle layer runs
    //    the four fail-closed gates against task.input_hash (content_hash was
    //    already pre-checked in 4b, before this buffered the blob).
    let sealed_pack = transport::recv_blob(conn, &task.task_id, BlobDir::Input, MAX_BLOB_SIZE)
        .await
        .map_err(|e| ExecutionFailure::transport(format!("receive input blob: {e}")))?;
    let pack = open_blob(&sealed_pack, keypair)
        .map_err(|e| ExecutionFailure::artifact(format!("open input blob: {e}")))?;
    let input: ErasedArtifact = bundle::unbundle(
        &*stage,
        &input_manifest,
        &pack,
        &stage_dir,
        Some(task.input_content_id),
        BlobDir::Input,
    )
    .map_err(|e| ExecutionFailure::artifact(format!("unbundle input: {e}")))?;

    // 7. Run under the coordinator's absolute deadline. It includes queueing
    //    and transfer time, so a late peer never rebuilds a fresh hour-long
    //    relative timeout after receiving an already-expired task.
    let started = std::time::Instant::now();
    let ctx = ctx;
    let stage_cancel = ctx.cancel.clone();
    let run = stage.run_erased(&ctx, input, task.args.clone());
    tokio::pin!(run);
    let output = tokio::select! {
        result = &mut run => result
            .map_err(|e| ExecutionFailure::from_stage_error(&task.stage_name, &e))?,
        _ = ctx.cancel.cancelled() => {
            return Err(ExecutionFailure::new(
                ExecutionFailureKind::Stage,
                "EXECUTION_CANCELLED",
                "peer stage cancelled",
            ));
        }
        _ = conn.closed() => {
            stage_cancel.cancel();
            return Err(ExecutionFailure::disconnected("coordinator connection closed while stage ran"));
        }
        _ = sleep_until_deadline(task.deadline.soft_remaining()) => {
            stage_cancel.cancel();
            return Err(ExecutionFailure::new(
                ExecutionFailureKind::Stage,
                "EXECUTION_SOFT_DEADLINE",
                format!("stage '{}' exceeded soft execution deadline", task.stage_name),
            ));
        }
        _ = tokio::time::sleep(task.deadline.hard_remaining()) => {
            stage_cancel.cancel();
            return Err(ExecutionFailure::new(
                ExecutionFailureKind::Stage,
                "EXECUTION_HARD_DEADLINE",
                format!("stage '{}' exceeded hard execution deadline", task.stage_name),
            ));
        }
    };
    let wall_time_ms = started.elapsed().as_millis() as u64;

    // 8. Bundle the output (rooted at the peer's stage_dir) + ship it back. An
    //    analytical expectation is checked when present; otherwise capture
    //    derives the identity from the typed output.
    let (out_manifest, out_pack) = bundle::bundle(
        &*stage,
        output,
        &stage_dir,
        BlobDir::Output,
        task.expected_content_id,
    )
    .map_err(|e| ExecutionFailure::artifact(format!("bundle output: {e}")))?;

    // Seal the output manifest to the coordinator's X25519 key.
    let manifest_bytes = bincode::serialize(&out_manifest)
        .map_err(|e| ExecutionFailure::artifact(format!("serialize output manifest: {e}")))?;
    let encrypted_output = crypto::encrypt(&manifest_bytes, &coordinator.x25519_pub);

    let mut result = TaskResult {
        protocol_version: crate::p2p::task::TASK_PROTOCOL_VERSION,
        task_id: task.task_id.clone(),
        peer_id: PeerId::from_pubkey(&keypair.verifying),
        content_id: out_manifest.content_id,
        encrypted_output: Some(encrypted_output),
        wall_time_ms,
        signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
    };
    result.signature = keypair.sign(&result.sign_payload());

    // Seal the bulk output blob to the coordinator too (same reasoning as the
    // input leg in step 6 — see `seal_blob`); previously this shipped the
    // plaintext `out_pack` straight to `send_blob`.
    let sealed_out_pack = seal_blob(&out_pack, &coordinator.x25519_pub)
        .map_err(|e| ExecutionFailure::artifact(format!("seal output blob: {e}")))?;

    P2pClient::send_result(conn, &result)
        .await
        .map_err(|e| ExecutionFailure::transport(format!("send task result: {e}")))?;
    transport::send_blob(conn, &task.task_id, BlobDir::Output, &sealed_out_pack)
        .await
        .map_err(|e| ExecutionFailure::transport(format!("send output blob: {e}")))?;
    Ok(())
}

async fn sleep_until_deadline(duration: Option<std::time::Duration>) {
    match duration {
        Some(duration) => tokio::time::sleep(duration).await,
        None => std::future::pending::<()>().await,
    }
}

/// A `task_id` is safe to use as a single path component: non-empty, only
/// `[A-Za-z0-9._-]`, and not `.`/`..`. Network-controlled, so this gates the
/// peer's `work_root.join(task_id)` against directory traversal.
pub(crate) fn is_safe_task_id(id: &str) -> bool {
    !id.is_empty()
        && id != "."
        && id != ".."
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// The verified output of a dispatched task: the rebased local artifact handle
/// plus the peer that produced it.
pub struct DispatchedOutput {
    pub output: ErasedArtifact,
    pub content_id: ContentId,
    pub peer_id: PeerId,
}

/// Portable output returned by the P2P data plane before the executor restores
/// it into its own attempt directory.
pub struct DispatchedStoredOutput {
    pub stored: StoredArtifact,
    pub content_id: ContentId,
    pub peer_id: PeerId,
    pub wall_time_ms: u64,
}

/// Cancellation is a terminal disposition, not a stringly transport error.
pub enum StoredDispatchOutcome {
    Succeeded(Box<DispatchedStoredOutput>),
    Cancelled { reason: String },
}

fn lifecycle_phase(
    lifecycle: Option<(&ExecutionLifecycle, &Assignment)>,
    phase: ExecutionPhase,
) -> Result<(), crate::framework::execution::LifecycleError> {
    if let Some((lifecycle, _assignment)) = lifecycle {
        lifecycle.transition(phase, None)?;
    }
    Ok(())
}

/// Coordinator side of the portable P2P data plane. This module owns task
/// signing, encrypted transport, peer-result verification, and conversion back
/// to A09's [`StoredArtifact`]. The executor sees only `ExecutionAdapter`.
pub async fn dispatch_stored_to_peer(
    conn: &QuinnConnection,
    coordinator_kp: &KeyPair,
    peer: &crate::p2p::PeerInfo,
    request: &ExecutionRequest,
    verification: Option<&dyn DispatchPolicy>,
    lifecycle: Option<(&ExecutionLifecycle, &Assignment)>,
) -> Result<StoredDispatchOutcome, ExecutionFailure> {
    if request.protocol_version != crate::framework::execution::EXECUTION_PROTOCOL_VERSION {
        return Err(ExecutionFailure::protocol(format!(
            "execution protocol v{} unsupported (want v{})",
            request.protocol_version,
            crate::framework::execution::EXECUTION_PROTOCOL_VERSION,
        )));
    }
    if !is_safe_task_id(&request.execution_id) {
        return Err(ExecutionFailure::protocol(format!(
            "unsafe execution_id '{}'",
            request.execution_id
        )));
    }
    let manifest_bytes = bincode::serialize(&request.input.manifest)
        .map_err(|e| ExecutionFailure::artifact(format!("serialize input manifest: {e}")))?;
    let encrypted_input = crypto::encrypt(&manifest_bytes, &peer.x25519_pub);
    let sealed_input = seal_blob(&request.input.pack, &peer.x25519_pub)
        .map_err(|e| ExecutionFailure::artifact(format!("seal input blob: {e}")))?;
    let mut task = TaskManifest {
        protocol_version: crate::p2p::task::TASK_PROTOCOL_VERSION,
        task_id: request.execution_id.clone(),
        coordinator_id: PeerId::from_pubkey(&coordinator_kp.verifying),
        stage_name: request.stage_name.clone(),
        stage_schema: request.stage_schema,
        input_content_id: request.input.manifest.content_id,
        invocation_key: request.invocation_key,
        args_hash: request.args_hash,
        expected_content_id: request.expected_content_id,
        args: request.args.clone(),
        resources: crate::p2p::task::ResourceRequest {
            cpu_cores: request.resources.cpu_cores,
            memory_gib: request.resources.memory_gib,
            gpu: request.resources.gpu,
            gpu_vram_gib: request.resources.gpu_vram_gib,
        },
        data_class: request.data_class.into(),
        timeout_secs: request.deadline.hard_remaining().as_secs().max(1),
        deadline: request.deadline,
        encrypted_input: Some(encrypted_input),
        signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
    };
    task.signature = coordinator_kp.sign(&task.sign_payload());

    lifecycle_phase(lifecycle, ExecutionPhase::UploadingInput)
        .map_err(|error| ExecutionFailure::protocol(format!("P2P upload transition: {error}")))?;
    transport::P2pServer::send_task(conn, &task)
        .await
        .map_err(|e| ExecutionFailure::transport(format!("send task: {e}")))?;
    transport::send_blob(conn, &task.task_id, BlobDir::Input, &sealed_input)
        .await
        .map_err(|e| ExecutionFailure::transport(format!("send input blob: {e}")))?;
    lifecycle_phase(lifecycle, ExecutionPhase::Running)
        .map_err(|error| ExecutionFailure::protocol(format!("P2P running transition: {error}")))?;

    let reply = transport::P2pServer::recv_task_reply(conn)
        .await
        .map_err(|e| ExecutionFailure::disconnected(format!("receive task reply: {e}")))?;
    let result = match reply {
        transport::TaskReply::Succeeded(result) => result,
        transport::TaskReply::Failed { task_id, failure } => {
            if task_id != task.task_id {
                return Err(ExecutionFailure::protocol(format!(
                    "failure task_id {task_id} != dispatched task {}",
                    task.task_id
                )));
            }
            return Err(failure);
        }
        transport::TaskReply::Cancelled { task_id, reason } => {
            if task_id != task.task_id {
                return Err(ExecutionFailure::protocol(format!(
                    "cancel task_id {task_id} != dispatched task {}",
                    task.task_id
                )));
            }
            return Ok(StoredDispatchOutcome::Cancelled { reason });
        }
    };
    if result.task_id != task.task_id {
        return Err(ExecutionFailure::protocol("result task_id mismatch"));
    }
    if result.protocol_version != crate::p2p::task::TASK_PROTOCOL_VERSION {
        return Err(ExecutionFailure::protocol(format!(
            "result protocol v{} unsupported (want v{})",
            result.protocol_version,
            crate::p2p::task::TASK_PROTOCOL_VERSION,
        )));
    }
    if result.peer_id != peer.id {
        return Err(ExecutionFailure::protocol(format!(
            "result peer {} != selected peer {}",
            result.peer_id, peer.id
        )));
    }
    if !crypto::verify(&peer.pubkey, &result.sign_payload(), &result.signature) {
        return Err(ExecutionFailure::protocol("result signature invalid"));
    }
    if let Some(policy) = verification {
        match policy.verify_result(&result, request.expected_content_id, &peer.pubkey) {
            DispatchVerdict::Accept => {}
            DispatchVerdict::Reject(reason) => {
                return Err(ExecutionFailure::protocol(format!(
                    "dispatch policy rejected peer result: {reason}"
                )));
            }
            DispatchVerdict::RetryOnDifferentPeer => {
                return Err(ExecutionFailure::unavailable(
                    "dispatch policy requires retry on a different peer",
                ));
            }
        }
    }
    let encrypted_output = result
        .encrypted_output
        .as_ref()
        .ok_or_else(|| ExecutionFailure::artifact("result has no encrypted_output"))?;
    let output_manifest_bytes = coordinator_kp
        .decrypt(encrypted_output)
        .map_err(|e| ExecutionFailure::artifact(format!("decrypt output manifest: {e}")))?;
    let manifest = bincode::deserialize::<bundle::BundleManifest>(&output_manifest_bytes)
        .map_err(|e| ExecutionFailure::artifact(format!("decode output manifest: {e}")))?;
    if manifest.content_id != result.content_id {
        return Err(ExecutionFailure::artifact(format!(
            "identity binding: artifact {} != signed result identity {}",
            manifest.content_id, result.content_id
        )));
    }
    if let Some(expected) = request.expected_content_id
        && expected != result.content_id
    {
        return Err(ExecutionFailure::artifact(format!(
            "output identity {} != analytically expected {}",
            result.content_id, expected
        )));
    }

    lifecycle_phase(lifecycle, ExecutionPhase::UploadingOutput).map_err(|error| {
        ExecutionFailure::protocol(format!("P2P output-upload transition: {error}"))
    })?;
    lifecycle_phase(lifecycle, ExecutionPhase::DownloadingOutput).map_err(|error| {
        ExecutionFailure::protocol(format!("P2P output-download transition: {error}"))
    })?;
    let sealed_output = transport::recv_blob(conn, &task.task_id, BlobDir::Output, MAX_BLOB_SIZE)
        .await
        .map_err(|e| ExecutionFailure::disconnected(format!("receive output blob: {e}")))?;
    let pack = open_blob(&sealed_output, coordinator_kp)
        .map_err(|e| ExecutionFailure::artifact(format!("open output blob: {e}")))?;
    if manifest.blob_len != u64::try_from(pack.len()).unwrap_or(u64::MAX)
        || manifest.blob_sha256 != ContentHash::of_bytes(&pack)
    {
        return Err(ExecutionFailure::artifact(
            "output pack does not match its signed manifest",
        ));
    }

    Ok(StoredDispatchOutcome::Succeeded(Box::new(
        DispatchedStoredOutput {
            stored: StoredArtifact { manifest, pack },
            content_id: result.content_id,
            peer_id: result.peer_id,
            wall_time_ms: result.wall_time_ms,
        },
    )))
}

/// Coordinator side of one dispatch: bundle `input` (rooted at its producing
/// `src_root`), seal + sign a `TaskManifest`, send it + the input blob to the
/// peer over `conn`, then receive the result + output blob, unbundle it into
/// `out_stage_dir`, and verify it against `expected_output_hash`.
///
/// This is the standalone CLI wrapper over [`dispatch_stored_to_peer`]'s
/// portable transport contract. The executor uses that canonical path through
/// `ExecutionAdapter` and restores its output independently.
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
    invocation_key: InvocationKey,
    expected_content_id: Option<ContentId>,
    data_class: crate::p2p::DataClass,
    timeout_secs: u64,
    out_stage_dir: &std::path::Path,
) -> Result<DispatchedOutput, TrainError> {
    if !is_safe_task_id(task_id) {
        return Err(TrainError::other(format!("unsafe task_id '{task_id}'")));
    }
    let ctor = registry
        .find_erased_stage(stage_name)
        .ok_or_else(|| TrainError::other(format!("unknown stage '{stage_name}'")))?;
    let stage = ctor();
    let (manifest, pack) = bundle::bundle(&*stage, input, src_root, BlobDir::Input, None)
        .map_err(|e| TrainError::other(format!("bundle input: {e}")))?;
    let data_class = match data_class {
        crate::p2p::DataClass::Public => DataClassification::Public,
        crate::p2p::DataClass::Internal => DataClassification::Internal,
        crate::p2p::DataClass::Restricted => DataClassification::Restricted,
    };
    let request = ExecutionRequest {
        protocol_version: crate::framework::execution::EXECUTION_PROTOCOL_VERSION,
        execution_id: task_id.to_string(),
        tenant: crate::tenant::Tenant::default(),
        stage_name: stage_name.to_string(),
        stage_schema: stage.schema(),
        invocation_key,
        args_hash: ContentHash::of_bytes(
            &serde_json::to_vec(&args).map_err(|e| TrainError::other(format!("args hash: {e}")))?,
        ),
        args,
        input: StoredArtifact { manifest, pack },
        expected_content_id,
        resources: crate::framework::execution::ExecutionResources::default(),
        data_class,
        deadline: ExecutionDeadline::from_now(
            None,
            std::time::Duration::from_secs(timeout_secs.max(1)),
        ),
    };
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs.max(1).saturating_add(30)),
        dispatch_stored_to_peer(conn, coordinator_kp, peer, &request, None, None),
    )
    .await
    .map_err(|_| TrainError::other("timed out waiting for peer result"))?
    .map_err(|failure| TrainError::other(failure.to_string()))?;
    let remote = match outcome {
        StoredDispatchOutcome::Succeeded(output) => *output,
        StoredDispatchOutcome::Cancelled { reason } => {
            return Err(TrainError::other(format!("peer task cancelled: {reason}")));
        }
    };
    let output = bundle::unbundle(
        &*stage,
        &remote.stored.manifest,
        &remote.stored.pack,
        out_stage_dir,
        Some(remote.content_id),
        BlobDir::Output,
    )
    .map_err(|e| TrainError::other(format!("unbundle output: {e}")))?;
    Ok(DispatchedOutput {
        output,
        content_id: remote.content_id,
        peer_id: remote.peer_id,
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

    // ── Finding 1: stage_dir leak ────────────────────────────────────────
    //
    // `execute_one` itself needs a live QuinnConnection to unit-test
    // end-to-end (see tests/p2p_integration.rs's `e2e` module for the real
    // dispatch path), so these tests pin the RAII mechanism directly: a
    // `StageDirGuard` removes its directory when dropped, on any path
    // (Rust drops locals on every fn exit — early `?` return, explicit
    // `return Err`, or falling off the end — so proving the Drop impl fires
    // once is sufficient to cover every exit path of `execute_one`).

    #[test]
    fn stage_dir_guard_removes_dir_on_drop() {
        let root = tempfile::tempdir().unwrap();
        let stage_dir = root.path().join("task-123");
        std::fs::create_dir_all(&stage_dir).unwrap();
        std::fs::write(stage_dir.join("secret.bin"), b"leaked artifact bytes").unwrap();
        assert!(stage_dir.exists());
        {
            let _guard = StageDirGuard(stage_dir.clone());
            // still present while the guard is alive
            assert!(stage_dir.exists());
        }
        assert!(
            !stage_dir.exists(),
            "stage_dir must be removed when the guard drops"
        );
    }

    #[test]
    fn stage_dir_guard_early_return_still_cleans_up() {
        // Simulates the shape of `execute_one`: a guard is created, then a
        // later `?`-style early return happens — the guard must still fire.
        fn run(stage_dir: &std::path::Path) -> Result<(), &'static str> {
            let _guard = StageDirGuard(stage_dir.to_path_buf());
            Err("simulated mid-function failure")?;
            Ok(())
        }
        let root = tempfile::tempdir().unwrap();
        let stage_dir = root.path().join("task-456");
        std::fs::create_dir_all(&stage_dir).unwrap();
        let _ = run(&stage_dir);
        assert!(
            !stage_dir.exists(),
            "early-return path must still clean up stage_dir"
        );
    }

    #[test]
    fn stage_dir_guard_missing_dir_is_noop() {
        // Best-effort cleanup: removing an already-absent dir must not panic.
        let root = tempfile::tempdir().unwrap();
        let stage_dir = root.path().join("never-created");
        let guard = StageDirGuard(stage_dir);
        drop(guard);
    }

    // ── Finding 3: plaintext bulk artifact transfer ──────────────────────

    #[test]
    fn seal_open_blob_roundtrip() {
        let kp = KeyPair::generate();
        let plaintext = b"bulk artifact bytes: real clinical EEG corpus".to_vec();
        let sealed = seal_blob(&plaintext, &kp.x25519_public).unwrap();
        // The wire bytes must not contain the plaintext verbatim — proves
        // actual encryption happened, not a pass-through/no-op.
        assert_ne!(sealed, plaintext);
        assert!(
            !sealed
                .windows(plaintext.len())
                .any(|w| w == plaintext.as_slice()),
            "sealed blob must not contain the plaintext as a contiguous substring"
        );
        let opened = open_blob(&sealed, &kp).unwrap();
        assert_eq!(opened, plaintext);
    }

    #[test]
    fn seal_blob_wrong_recipient_cannot_open() {
        let kp1 = KeyPair::generate();
        let kp2 = KeyPair::generate();
        let plaintext = b"restricted data class corpus".to_vec();
        let sealed = seal_blob(&plaintext, &kp1.x25519_public).unwrap();
        assert!(
            open_blob(&sealed, &kp2).is_err(),
            "wrong recipient must not decrypt"
        );
    }

    #[test]
    fn seal_blob_fresh_nonce_and_key_each_call() {
        // crypto::encrypt mints a fresh AES-256 key + nonce per call (a full
        // ephemeral-ECDH hybrid seal), so calling it once per blob send — as
        // `seal_blob` does — is safe even when sealing the same plaintext to
        // the same recipient repeatedly: no static key/nonce reuse across
        // calls (which would be a real AES-GCM vulnerability).
        let kp = KeyPair::generate();
        let plaintext = b"same bytes sent twice".to_vec();
        let a = seal_blob(&plaintext, &kp.x25519_public).unwrap();
        let b = seal_blob(&plaintext, &kp.x25519_public).unwrap();
        assert_ne!(
            a, b,
            "two seals of identical plaintext must produce different wire bytes"
        );
        assert_eq!(open_blob(&a, &kp).unwrap(), plaintext);
        assert_eq!(open_blob(&b, &kp).unwrap(), plaintext);
    }

    #[test]
    fn seal_open_blob_roundtrip_large() {
        // Exercise a multi-MiB buffer (the realistic shape of a bundle
        // pack), not just a short string.
        let kp = KeyPair::generate();
        let plaintext: Vec<u8> = (0..(4 * 1024 * 1024)).map(|i| (i % 251) as u8).collect();
        let sealed = seal_blob(&plaintext, &kp.x25519_public).unwrap();
        let opened = open_blob(&sealed, &kp).unwrap();
        assert_eq!(opened, plaintext);
    }
}
