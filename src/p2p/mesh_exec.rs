// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Recipe-level dispatch over the `p2p node` mesh: the execution adapter that
//! `recipe run --launcher p2p` installs, and the worker that runs what it sends.
//!
//! **Which stages leave.** The executor asks
//! [`DispatchPolicy::remote_class`](crate::p2p::dispatch::DispatchPolicy::remote_class),
//! whose default is the stage's own declaration
//! ([`Stage::placement`](crate::framework::Stage::placement)). A stage that
//! does not declare it may leave runs on the submitting machine as part of the
//! same plan, contained like any local stage; each such decision is logged.
//!
//! **Where they go.** [`MeshExecutionAdapter`] picks, for each eligible stage,
//! a connected peer that (a) answers on the stage plane, (b) runs the same
//! [code identity](crate::framework::code_identity) as the submitter, (c)
//! accepts work from it, (d) holds the trust the stage's data class needs
//! ([`DispatchMatrix`](crate::p2p::trust::DispatchMatrix)), and (e) has the
//! declared cores and memory. Among those it prefers the one with the fewest
//! of this submitter's stages in flight, then the one it has used least, then
//! the best local reputation, so a plan's stages spread over the mesh.
//!
//! **No peer is a failure.** When no such peer appears within the configured
//! wait ([`MeshExecutionAdapter::with_peer_wait`]) the attempt fails with
//! [`ExecutionFailureKind::Unavailable`], naming every peer it saw and why each
//! was passed over. Under `--launcher p2p` the executor fails the stage on it;
//! it never runs the stage here instead. Peers that are capable but busy are
//! waited for up to the stage's own deadline.
//!
//! **Data.** The input moves by the mesh's pull-based object plane
//! ([`MeshObjects`]): the submitter publishes it into its store and grants it
//! to the chosen worker, which fetches it by content id over the connection
//! the stage arrived on, verifying every chunk and the whole artifact. The
//! output comes back the same way, granted to the submitter, and the executor
//! restores and re-verifies it before anything downstream sees it.
//!
//! **Worker.** [`MeshStageWorker`] serves the stage plane on a `p2p node`. It
//! refuses a stage from a submitter it was not told to accept, a stage whose
//! submitter runs different code (the code identity digest must be equal: the
//! same engine build and the same cookbook builds, or the worker would compute
//! something else), an unknown stage, a stage its own build declares
//! local-only, and work beyond its slots or memory admission. It runs the stage
//! with a [`StageContext`] placed `Local`, so a stage that starts processes
//! starts them contained exactly as on the submitting machine.
//!
//! **Cancel and loss.** The submitter keeps the stage's stream open while the
//! stage runs. Cancelling (`blut cancel <job>`, Ctrl-C) writes a cancel
//! message; a submitter that exits closes its connection; one that dies without
//! a word is detected by the QUIC idle timeout. Each ends the worker's watch on
//! the stream, which cancels the stage.
//!
//! **Wire.** One stage (or one hello) per QUIC bi-stream, opened by the
//! submitter: `[STAGE_STREAM_TAG][STAGE_PLANE_VERSION]`, then length-prefixed
//! JSON messages in both directions (`[len:u32-le][json]`). The tag byte
//! cannot collide with a mesh frame (which starts with its protocol version)
//! or the object plane ([`OBJECT_STREAM_TAG`](crate::p2p::objects::OBJECT_STREAM_TAG)).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::error::TrainError;
use crate::framework::artifact::{ArtifactContentId, ContentHash};
use crate::framework::artifact_store::{ArtifactRole, StoredArtifact, bundle, unbundle};
use crate::framework::cache::CacheHandle;
use crate::framework::code_identity::CodeIdentity;
use crate::framework::cookbook::Registry;
use crate::framework::execution::{
    Assignment, EXECUTION_PROTOCOL_VERSION, ExecutionAdapter, ExecutionArtifact, ExecutionFailure,
    ExecutionFailureKind, ExecutionHandle, ExecutionLifecycle, ExecutionMode, ExecutionPhase,
    ExecutionRequest, ExecutionSnapshot, ExecutionTerminal,
};
use crate::framework::object_store::ObjectKey;
use crate::framework::stage::{StageContext, StageDyn};
use crate::p2p::crypto::{KeyPair, verify};
use crate::p2p::dispatch::{DefaultDispatchPolicy, DispatchPolicy};
use crate::p2p::node::MeshNode;
use crate::p2p::objects::MeshObjects;
use crate::p2p::peer::{PeerCapabilities, PeerId, PeerInfo};
use crate::p2p::task::{ResourceRequest, TASK_PROTOCOL_VERSION, TaskManifest, TaskResult};
use crate::p2p::transport::tls_authenticated_pubkey;
use crate::p2p::trust::{DataClass, DispatchMatrix, TrustLevel};

/// First byte of a stage-plane stream.
pub const STAGE_STREAM_TAG: u8 = 0xB8;

/// Version of the stage-plane message layout.
pub const STAGE_PLANE_VERSION: u8 = 1;

/// Default time an eligible stage waits for a capable peer before the attempt
/// fails as unavailable.
pub const DEFAULT_PEER_WAIT: Duration = Duration::from_secs(60);

/// Largest message on the stage plane (manifests and identities are small;
/// stage data moves on the object plane).
const MAX_STAGE_MESSAGE: usize = 8 * 1024 * 1024;

/// Bound on a hello exchange, and on any single control read.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(15);

/// How long a peer's hello is trusted before it is asked again.
const HELLO_TTL: Duration = Duration::from_secs(30);

/// How long a peer that answered "busy" is left alone.
const BUSY_BACKOFF: Duration = Duration::from_secs(1);

/// How often a waiting stage looks at the mesh again.
const WAIT_POLL: Duration = Duration::from_millis(250);

/// How long a cancelled stage waits for the worker's acknowledgement.
const CANCEL_GRACE: Duration = Duration::from_secs(2);

/// How long a cancelled stage on a worker gets to stop on its own before it
/// is dropped.
const STAGE_STOP_GRACE: Duration = Duration::from_secs(10);

/// Domain tag of the dispatch signature.
const DISPATCH_DOMAIN: &[u8] = b"blut.mesh.stage-dispatch.v1";

// ── wire ──────────────────────────────────────────────────────────────────

/// Submitter → worker.
#[derive(Clone, Debug, Serialize, Deserialize)]
enum StageRequest {
    /// Who are you, and will you run my stages?
    Hello,
    /// Run this stage.
    Run(Box<StageDispatch>),
    /// Stop the stage this stream carries.
    Cancel { reason: String },
}

/// A signed stage dispatch: the task manifest plus the submitter's code
/// identity, both covered by one signature from the submitter's key.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct StageDispatch {
    manifest: TaskManifest,
    code_identity: CodeIdentity,
    /// Hex Ed25519 signature over [`dispatch_payload`].
    signature: String,
}

fn dispatch_payload(manifest: &TaskManifest, identity_digest: &[u8; 32]) -> Vec<u8> {
    let mut payload = DISPATCH_DOMAIN.to_vec();
    payload.extend_from_slice(identity_digest);
    payload.extend_from_slice(&manifest.sign_payload());
    payload
}

fn signature_hex(signature: &Signature) -> String {
    faster_hex::hex_string(&signature.to_bytes())
}

fn signature_from_hex(hex: &str) -> Option<Signature> {
    let mut bytes = [0u8; 64];
    faster_hex::hex_decode(hex.as_bytes(), &mut bytes).ok()?;
    Some(Signature::from_bytes(&bytes))
}

/// Worker → submitter.
#[derive(Clone, Debug, Serialize, Deserialize)]
enum StageReply {
    Hello(WorkerHello),
    /// Every check passed; the worker owns the stage now.
    Accepted,
    /// The input is in place and the stage is running.
    Running,
    /// The stage finished; its output is granted to the submitter.
    Done(Box<TaskResult>),
    /// The worker will not run it (nothing was started).
    Refused(StageRefusal),
    /// The stage ran and failed.
    Failed(ExecutionFailure),
    /// The stage was stopped.
    Cancelled {
        reason: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WorkerHello {
    code_identity: CodeIdentity,
    x25519_pub: [u8; 32],
    /// Whether this worker runs stages for the peer that asked.
    accepts_from_you: bool,
    cpu_cores: u32,
    memory_gib: u32,
    max_tasks: u32,
    running: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RefusalCode {
    CodeIdentityMismatch,
    UnknownStage,
    StaysLocal,
    NotAccepted,
    /// All slots in use: the stage waits for the worker.
    Busy,
    /// Memory admission refused it: not waited for past the peer wait, since
    /// a worker too small for the stage would otherwise hold it forever.
    NoMemory,
    BadRequest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StageRefusal {
    code: RefusalCode,
    message: String,
}

impl StageRefusal {
    fn new(code: RefusalCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

async fn write_msg<T: Serialize>(
    send: &mut quinn::SendStream,
    message: &T,
) -> Result<(), TrainError> {
    let json = serde_json::to_vec(message)
        .map_err(|e| TrainError::other(format!("encode stage message: {e}")))?;
    if json.len() > MAX_STAGE_MESSAGE {
        return Err(TrainError::other(format!(
            "stage message too large: {} bytes",
            json.len()
        )));
    }
    let mut out = Vec::with_capacity(4 + json.len());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    send.write_all(&out)
        .await
        .map_err(|e| TrainError::other(format!("write stage message: {e}")))
}

/// Read one message; `Ok(None)` when the other side finished its half cleanly.
async fn read_msg<T: DeserializeOwned>(
    recv: &mut quinn::RecvStream,
) -> Result<Option<T>, TrainError> {
    let mut len = [0u8; 4];
    match recv.read_exact(&mut len).await {
        Ok(()) => {}
        Err(quinn::ReadExactError::FinishedEarly(0)) => return Ok(None),
        Err(e) => return Err(TrainError::other(format!("read stage message: {e}"))),
    }
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_STAGE_MESSAGE {
        return Err(TrainError::other(format!(
            "stage message too large: {len} bytes"
        )));
    }
    let mut body = vec![0u8; len];
    recv.read_exact(&mut body)
        .await
        .map_err(|e| TrainError::other(format!("read stage message body: {e}")))?;
    serde_json::from_slice(&body)
        .map(Some)
        .map_err(|e| TrainError::other(format!("decode stage message: {e}")))
}

async fn open_stage_stream(
    conn: &quinn::Connection,
) -> Result<(quinn::SendStream, quinn::RecvStream), TrainError> {
    let (mut send, recv) = conn
        .open_bi()
        .await
        .map_err(|e| TrainError::other(format!("open stage stream: {e}")))?;
    send.write_all(&[STAGE_STREAM_TAG, STAGE_PLANE_VERSION])
        .await
        .map_err(|e| TrainError::other(format!("write stage stream header: {e}")))?;
    Ok((send, recv))
}

// ── worker ────────────────────────────────────────────────────────────────

/// Runs stages dispatched over the stage plane by the submitters it accepts.
/// Attach it to a node with
/// [`MeshNode::set_stage_worker`](crate::p2p::node::MeshNode::set_stage_worker).
pub struct MeshStageWorker {
    registry: Arc<Registry>,
    objects: Arc<MeshObjects>,
    keypair: Arc<KeyPair>,
    work_root: PathBuf,
    accept: HashSet<[u8; 32]>,
    identity: CodeIdentity,
    identity_digest: [u8; 32],
    policy: Arc<dyn DispatchPolicy>,
    slots: Arc<Semaphore>,
    max_tasks: u32,
    running: AtomicU32,
    memory_gib: u32,
}

impl MeshStageWorker {
    /// A worker that runs `registry`'s stages, moving data through `objects`
    /// and working under `work_root`. It accepts no submitter until
    /// [`accept_from`](Self::accept_from) names one, and it runs with this
    /// process's installed code identity.
    pub fn new(
        registry: Arc<Registry>,
        objects: Arc<MeshObjects>,
        keypair: Arc<KeyPair>,
        work_root: PathBuf,
    ) -> Self {
        let identity = crate::framework::code_identity::current().clone();
        let max_tasks = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 16);
        Self {
            registry,
            objects,
            keypair,
            work_root,
            accept: HashSet::new(),
            identity_digest: identity.digest(),
            identity,
            policy: Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default())),
            slots: Arc::new(Semaphore::new(max_tasks)),
            max_tasks: max_tasks as u32,
            running: AtomicU32::new(0),
            memory_gib: crate::broker::tenant_quota::TenantAdmission::prepare(
                crate::tenant::Tenant::default(),
            )
            .map(|admission| admission.snapshot().mem_total_gb.max(0.0) as u32)
            .unwrap_or(0),
        }
    }

    /// Run stages for the submitter holding `key`.
    pub fn accept_from(mut self, key: VerifyingKey) -> Self {
        self.accept.insert(key.to_bytes());
        self
    }

    /// Run at most `n` stages at once (default: the CPU count, at most 16).
    pub fn with_max_tasks(mut self, n: usize) -> Self {
        let n = n.clamp(1, 1024);
        self.slots = Arc::new(Semaphore::new(n));
        self.max_tasks = n as u32;
        self
    }

    /// Compare submitters against `identity` instead of this process's
    /// installed one (tests, and tools that host several builds).
    pub fn with_code_identity(mut self, identity: CodeIdentity) -> Self {
        self.identity_digest = identity.digest();
        self.identity = identity;
        self
    }

    /// Decide eligibility with `policy` instead of the default (which honours
    /// each stage's own declaration).
    pub fn with_policy(mut self, policy: Arc<dyn DispatchPolicy>) -> Self {
        self.policy = policy;
        self
    }

    fn hello(&self, submitter: Option<[u8; 32]>) -> WorkerHello {
        WorkerHello {
            code_identity: self.identity.clone(),
            x25519_pub: self.keypair.x25519_public.to_bytes(),
            accepts_from_you: submitter.is_some_and(|key| self.accept.contains(&key)),
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1),
            memory_gib: self.memory_gib,
            max_tasks: self.max_tasks,
            running: self.running.load(Ordering::Relaxed),
        }
    }

    /// Serve one stage-plane stream whose tag byte the node already read.
    /// Returns the reputation signal for the submitter: `Some(true)` when it
    /// sent valid work this worker served, `None` otherwise.
    pub(crate) async fn serve_stream(
        &self,
        peer: &PeerId,
        conn: &quinn::Connection,
        mut send: quinn::SendStream,
        mut recv: quinn::RecvStream,
    ) -> Option<bool> {
        let mut version = [0u8; 1];
        match tokio::time::timeout(CONTROL_TIMEOUT, recv.read_exact(&mut version)).await {
            Ok(Ok(())) => {}
            _ => return None,
        }
        if version[0] != STAGE_PLANE_VERSION {
            let refusal = StageRefusal::new(
                RefusalCode::BadRequest,
                format!(
                    "stage plane version {} unsupported (this worker speaks {STAGE_PLANE_VERSION})",
                    version[0]
                ),
            );
            let _ = write_msg(&mut send, &StageReply::Refused(refusal)).await;
            let _ = send.finish();
            return None;
        }
        let request = match tokio::time::timeout(
            CONTROL_TIMEOUT,
            read_msg::<StageRequest>(&mut recv),
        )
        .await
        {
            Ok(Ok(Some(request))) => request,
            Ok(Ok(None)) | Err(_) => return None,
            Ok(Err(e)) => {
                let refusal = StageRefusal::new(RefusalCode::BadRequest, e.to_string());
                let _ = write_msg(&mut send, &StageReply::Refused(refusal)).await;
                let _ = send.finish();
                return None;
            }
        };
        let submitter_key = tls_authenticated_pubkey(conn);
        match request {
            StageRequest::Hello => {
                let _ = write_msg(&mut send, &StageReply::Hello(self.hello(submitter_key))).await;
                let _ = send.finish();
                None
            }
            StageRequest::Cancel { .. } => None,
            StageRequest::Run(dispatch) => {
                self.serve_run(peer, conn, submitter_key, *dispatch, send, recv)
                    .await
            }
        }
    }

    /// Every check a dispatch must pass before anything is fetched or run.
    fn admit(
        &self,
        peer: &PeerId,
        submitter_key: Option<[u8; 32]>,
        dispatch: &StageDispatch,
    ) -> Result<(Arc<dyn StageDyn>, VerifyingKey), StageRefusal> {
        let bad = |message: String| StageRefusal::new(RefusalCode::BadRequest, message);
        let Some(key_bytes) = submitter_key else {
            return Err(StageRefusal::new(
                RefusalCode::NotAccepted,
                "the connection carries no client identity (mutual TLS is required)",
            ));
        };
        if !self.accept.contains(&key_bytes) {
            return Err(StageRefusal::new(
                RefusalCode::NotAccepted,
                format!(
                    "this worker runs stages only for the dispatchers named by its \
                     --coordinator-pubkey; {} is not one of them",
                    peer.short()
                ),
            ));
        }
        let key = VerifyingKey::from_bytes(&key_bytes).map_err(|e| bad(format!("key: {e}")))?;
        let manifest = &dispatch.manifest;
        let identity_digest = dispatch.code_identity.digest();
        let signature = signature_from_hex(&dispatch.signature)
            .ok_or_else(|| bad("dispatch signature is not 64 hex bytes".into()))?;
        if !verify(
            &key,
            &dispatch_payload(manifest, &identity_digest),
            &signature,
        ) {
            return Err(bad(
                "dispatch signature does not verify against the submitter's key".into(),
            ));
        }
        // The code identity: a worker that runs different code would compute
        // something else under the same cache key.
        if identity_digest != self.identity_digest {
            return Err(StageRefusal::new(
                RefusalCode::CodeIdentityMismatch,
                format!(
                    "this worker runs {}, the submitter runs {}; a stage runs only on a \
                     worker with the code that submitted it",
                    self.identity, dispatch.code_identity
                ),
            ));
        }
        if manifest.protocol_version != TASK_PROTOCOL_VERSION {
            return Err(bad(format!(
                "task protocol v{} unsupported (want v{TASK_PROTOCOL_VERSION})",
                manifest.protocol_version
            )));
        }
        if !verify(&key, &manifest.sign_payload(), &manifest.signature) {
            return Err(bad("task manifest signature invalid".into()));
        }
        manifest
            .verify_args(&manifest.args)
            .map_err(|e| bad(e.to_string()))?;
        if manifest.coordinator_id != *peer {
            return Err(bad(format!(
                "the manifest names coordinator {}, but the connection is {}",
                manifest.coordinator_id.short(),
                peer.short()
            )));
        }
        if !crate::p2p::peer_exec::is_safe_task_id(&manifest.task_id) {
            return Err(bad(format!("unsafe task id '{}'", manifest.task_id)));
        }
        if manifest.deadline.hard_remaining().is_zero() {
            return Err(bad("the stage's deadline has already passed".into()));
        }
        let ctor = self
            .registry
            .find_erased_stage(&manifest.stage_name)
            .ok_or_else(|| {
                StageRefusal::new(
                    RefusalCode::UnknownStage,
                    format!(
                        "no stage named '{}' is registered on this worker",
                        manifest.stage_name
                    ),
                )
            })?;
        let stage = ctor();
        if stage.schema() != manifest.stage_schema {
            return Err(StageRefusal::new(
                RefusalCode::CodeIdentityMismatch,
                format!(
                    "stage '{}' is schema {} here, {} on the submitter",
                    manifest.stage_name,
                    stage.schema(),
                    manifest.stage_schema
                ),
            ));
        }
        let Some(class) = self.policy.remote_class(stage.as_ref(), &manifest.args) else {
            return Err(StageRefusal::new(
                RefusalCode::StaysLocal,
                format!(
                    "stage '{}' does not declare that it may run on another machine",
                    manifest.stage_name
                ),
            ));
        };
        if class != manifest.data_class {
            return Err(bad(format!(
                "stage '{}' declares {class:?} data here, {:?} in the manifest",
                manifest.stage_name, manifest.data_class
            )));
        }
        Ok((stage, key))
    }

    async fn serve_run(
        &self,
        peer: &PeerId,
        conn: &quinn::Connection,
        submitter_key: Option<[u8; 32]>,
        dispatch: StageDispatch,
        mut send: quinn::SendStream,
        recv: quinn::RecvStream,
    ) -> Option<bool> {
        let stage_name = dispatch.manifest.stage_name.clone();
        let refuse = |refusal: StageRefusal| {
            tracing::info!(
                "mesh worker: refused stage '{stage_name}' from {}: {}",
                peer.short(),
                refusal.message
            );
            refusal
        };
        let (stage, _key) = match self.admit(peer, submitter_key, &dispatch) {
            Ok(admitted) => admitted,
            Err(refusal) => {
                let refusal = refuse(refusal);
                let _ = write_msg(&mut send, &StageReply::Refused(refusal)).await;
                let _ = send.finish();
                return None;
            }
        };
        let manifest = dispatch.manifest;
        let Ok(slot) = self.slots.clone().try_acquire_owned() else {
            let refusal = refuse(StageRefusal::new(
                RefusalCode::Busy,
                format!("all {} stage slots are in use", self.max_tasks),
            ));
            let _ = write_msg(&mut send, &StageReply::Refused(refusal)).await;
            let _ = send.finish();
            return None;
        };
        // Memory admission, as for a local run of the same stage.
        let envelope = stage.resource_envelope(&manifest.args);
        let footprint = crate::broker::Footprint {
            ram_bytes: envelope.ram_bytes.max(crate::broker::GIB),
            vram_mib: 0,
        };
        // The resource probe reads the host and may start a GPU tool, so it
        // stays off the task that serves this connection's other streams.
        let admitted = tokio::task::spawn_blocking(move || {
            crate::broker::tenant_quota::TenantAdmission::prepare(crate::tenant::Tenant::default())
                .and_then(|admission| {
                    admission.reserve(&footprint, crate::broker::admission::DEFAULT_FLOOR_GIB)
                })
        })
        .await;
        let admitted = admitted
            .map_err(|e| e.to_string())
            .and_then(|reserved| reserved.map_err(|e| e.to_string()));
        let reservation = match admitted {
            Ok(reservation) => reservation,
            Err(why) => {
                let refusal = refuse(StageRefusal::new(
                    RefusalCode::NoMemory,
                    format!("memory admission refused: {why}"),
                ));
                let _ = write_msg(&mut send, &StageReply::Refused(refusal)).await;
                let _ = send.finish();
                return None;
            }
        };
        if write_msg(&mut send, &StageReply::Accepted).await.is_err() {
            return None;
        }
        self.running.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            "mesh worker: running stage '{}' ({}) for {}",
            manifest.stage_name,
            manifest.task_id,
            peer.short()
        );

        // Anything that ends the submitter's side of the stream (a cancel
        // message, a clean finish before the result, a reset, the connection
        // closing or timing out) stops the stage.
        let cancel = CancellationToken::new();
        let watcher = {
            let cancel = cancel.clone();
            let mut recv = recv;
            tokio::spawn(async move {
                let reason = loop {
                    match read_msg::<StageRequest>(&mut recv).await {
                        Ok(Some(StageRequest::Cancel { reason })) => {
                            break format!("the submitter cancelled it: {reason}");
                        }
                        Ok(Some(_)) => continue,
                        Ok(None) => break "the submitter closed the stream".to_string(),
                        Err(e) => break format!("lost the submitter: {e}"),
                    }
                };
                cancel.cancel();
                reason
            })
        };

        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::channel::<()>(1);
        let run = self.execute(stage, &manifest, peer, conn, cancel.clone(), progress_tx);
        tokio::pin!(run);
        let outcome = loop {
            tokio::select! {
                outcome = &mut run => break outcome,
                Some(()) = progress_rx.recv() => {
                    let _ = write_msg(&mut send, &StageReply::Running).await;
                }
            }
        };
        drop(reservation);
        drop(slot);
        self.running.fetch_sub(1, Ordering::Relaxed);
        let cancelled_because = if cancel.is_cancelled() {
            watcher.await.ok()
        } else {
            watcher.abort();
            None
        };
        let (reply, signal) = match outcome {
            Ok(result) => {
                self.objects.grant(peer, result.content_id);
                tracing::info!(
                    "mesh worker: stage '{}' ({}) done in {} ms, output {}",
                    manifest.stage_name,
                    manifest.task_id,
                    result.wall_time_ms,
                    result.content_id.to_hex()
                );
                (StageReply::Done(Box::new(result)), Some(true))
            }
            Err(_) if cancelled_because.is_some() => {
                let reason = cancelled_because.unwrap_or_default();
                tracing::info!(
                    "mesh worker: stopped stage '{}' ({}): {reason}",
                    manifest.stage_name,
                    manifest.task_id
                );
                (StageReply::Cancelled { reason }, None)
            }
            Err(failure) => {
                tracing::warn!(
                    "mesh worker: stage '{}' ({}) failed: {failure}",
                    manifest.stage_name,
                    manifest.task_id
                );
                (StageReply::Failed(failure), None)
            }
        };
        let _ = write_msg(&mut send, &reply).await;
        let _ = send.finish();
        signal
    }

    /// Fetch the input, run the stage, publish the output.
    async fn execute(
        &self,
        stage: Arc<dyn StageDyn>,
        manifest: &TaskManifest,
        peer: &PeerId,
        conn: &quinn::Connection,
        cancel: CancellationToken,
        progress: tokio::sync::mpsc::Sender<()>,
    ) -> Result<TaskResult, ExecutionFailure> {
        let stage_dir = self.work_root.join(&manifest.task_id);
        let _cleanup = RemoveOnDrop(stage_dir.clone());
        std::fs::create_dir_all(&stage_dir).map_err(|e| {
            ExecutionFailure::new(
                ExecutionFailureKind::Storage,
                "EXECUTION_STORAGE",
                format!("create {}: {e}", stage_dir.display()),
            )
        })?;
        let budget = manifest.deadline.hard_remaining();
        let input_id = manifest.input_content_id;

        // Input: from this node's store, or pulled from the submitter.
        let local = read_stored(&self.objects, input_id).await;
        let (stored_in, fetched) = match local {
            Some(stored) => (stored, false),
            None => {
                let fetch = self.objects.fetch_artifact(conn, input_id, budget);
                tokio::select! {
                    fetched = fetch => (fetched.map_err(|e| ExecutionFailure::transport(format!(
                        "fetch input {} from {}: {e}", input_id.to_hex(), peer.short()
                    )))?, true),
                    _ = cancel.cancelled() => {
                        return Err(ExecutionFailure::new(
                            ExecutionFailureKind::Unknown, "EXECUTION_CANCELLED", "cancelled"));
                    }
                }
            }
        };
        let restore_stage = stage.clone();
        let restore_dir = stage_dir.clone();
        let (input, stored_in) = tokio::task::spawn_blocking(move || {
            unbundle(
                restore_stage.as_ref(),
                &stored_in.manifest,
                &stored_in.pack,
                &restore_dir,
                Some(input_id),
                ArtifactRole::Input,
            )
            .map(|input| (input, stored_in))
        })
        .await
        .map_err(|e| ExecutionFailure::artifact(format!("restore input task: {e}")))?
        .map_err(|e| ExecutionFailure::artifact(format!("restore input: {e}")))?;
        if fetched {
            // Verified by `unbundle`; keep it so a re-dispatch is not fetched again.
            match bincode::serialize(&stored_in) {
                Ok(bytes) => {
                    if let Err(e) = self
                        .objects
                        .store()
                        .put(ObjectKey::Artifact(input_id), bytes)
                        .await
                    {
                        tracing::warn!("mesh worker: keep input {}: {e}", input_id.to_hex());
                    }
                }
                Err(e) => tracing::warn!("mesh worker: encode input {}: {e}", input_id.to_hex()),
            }
        }
        drop(stored_in);
        let _ = progress.try_send(());

        let mut ctx = StageContext::for_peer(
            stage_dir.clone(),
            stage_dir.clone(),
            Arc::new(CacheHandle::job_local(stage_dir.join(".cache"))),
            manifest.invocation_key,
        );
        ctx.cancel = cancel.child_token();
        let started = Instant::now();
        let run = stage.run_erased(&ctx, input, manifest.args.clone());
        tokio::pin!(run);
        let cancelled = || {
            ExecutionFailure::new(
                ExecutionFailureKind::Unknown,
                "EXECUTION_CANCELLED",
                "cancelled",
            )
        };
        let output = tokio::select! {
            // The stage first: one that saw the cancel through its context
            // finishes its own teardown before anything is dropped.
            biased;
            output = &mut run => output.map_err(|e| ExecutionFailure::from_stage_error(&manifest.stage_name, &e))?,
            _ = cancel.cancelled() => {
                // `ctx.cancel` fired with it: give the stage time to stop its
                // processes, then drop it (which kills a contained child).
                let _ = tokio::time::timeout(STAGE_STOP_GRACE, &mut run).await;
                return Err(cancelled());
            }
            _ = tokio::time::sleep(budget) => {
                return Err(ExecutionFailure::new(
                    ExecutionFailureKind::Stage,
                    "EXECUTION_TIMEOUT",
                    format!("stage '{}' passed its deadline on the worker", manifest.stage_name),
                ));
            }
        };
        let wall_time_ms = started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);

        let pack_stage = stage.clone();
        let pack_dir = stage_dir.clone();
        let (out_manifest, out_pack) = tokio::task::spawn_blocking(move || {
            bundle(
                pack_stage.as_ref(),
                output,
                &pack_dir,
                ArtifactRole::Output,
                None,
            )
        })
        .await
        .map_err(|e| ExecutionFailure::artifact(format!("pack output task: {e}")))?
        .map_err(|e| ExecutionFailure::artifact(format!("pack output: {e}")))?;
        let output_id = out_manifest.content_id;
        let bytes = bincode::serialize(&StoredArtifact {
            manifest: out_manifest,
            pack: out_pack,
        })
        .map_err(|e| ExecutionFailure::artifact(format!("encode output: {e}")))?;
        self.objects
            .store()
            .put(ObjectKey::Artifact(output_id), bytes)
            .await
            .map_err(|e| {
                ExecutionFailure::new(
                    ExecutionFailureKind::Storage,
                    "EXECUTION_STORAGE",
                    format!("store output: {e}"),
                )
            })?;
        let mut result = TaskResult {
            protocol_version: TASK_PROTOCOL_VERSION,
            task_id: manifest.task_id.clone(),
            peer_id: PeerId::from_pubkey(&self.keypair.verifying),
            content_id: output_id,
            encrypted_output: None,
            wall_time_ms,
            signature: Signature::from_bytes(&[0u8; 64]),
        };
        result.signature = self.keypair.sign(&result.sign_payload());
        Ok(result)
    }
}

struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn read_stored(objects: &MeshObjects, id: ArtifactContentId) -> Option<StoredArtifact> {
    let bytes = objects.store().get(ObjectKey::Artifact(id)).await.ok()??;
    bincode::deserialize(&bytes).ok()
}

// ── submitter ─────────────────────────────────────────────────────────────

/// The execution adapter that places eligible stages on mesh peers
/// ([module docs](self)). Mode [`ExecutionMode::P2p`].
#[derive(Clone)]
pub struct MeshExecutionAdapter {
    inner: Arc<AdapterInner>,
}

struct AdapterInner {
    node: Arc<MeshNode>,
    objects: Arc<MeshObjects>,
    keypair: Arc<KeyPair>,
    policy: Arc<dyn DispatchPolicy>,
    identity: CodeIdentity,
    identity_digest: [u8; 32],
    peer_wait: Duration,
    generation: AtomicU64,
    peers: parking_lot::Mutex<HashMap<PeerId, PeerState>>,
}

#[derive(Default)]
struct PeerState {
    /// The last hello (or why there was none), when, and on which connection.
    hello: Option<(Result<WorkerHello, String>, Instant, usize)>,
    in_flight: u32,
    dispatched: u64,
    busy_until: Option<Instant>,
}

/// One connected peer as the selection sees it.
struct Candidate {
    id: PeerId,
    conn: quinn::Connection,
    key: VerifyingKey,
    hello: WorkerHello,
    rank: (u32, u64, i64),
}

impl MeshExecutionAdapter {
    /// An adapter dispatching from `node` (which must have the scheduler
    /// capability), publishing inputs through `objects`, signing with
    /// `keypair`. Eligibility and trust come from `policy`; the code identity
    /// is this process's installed one.
    pub fn new(
        node: Arc<MeshNode>,
        objects: Arc<MeshObjects>,
        keypair: Arc<KeyPair>,
        policy: Arc<dyn DispatchPolicy>,
    ) -> Self {
        let identity = crate::framework::code_identity::current().clone();
        Self {
            inner: Arc::new(AdapterInner {
                node,
                objects,
                keypair,
                policy,
                identity_digest: identity.digest(),
                identity,
                peer_wait: DEFAULT_PEER_WAIT,
                generation: AtomicU64::new(0),
                peers: parking_lot::Mutex::new(HashMap::new()),
            }),
        }
    }

    fn inner_mut(&mut self) -> &mut AdapterInner {
        Arc::get_mut(&mut self.inner).expect("configure the adapter before sharing it")
    }

    /// How long an eligible stage waits for a capable peer before its attempt
    /// fails as unavailable (default [`DEFAULT_PEER_WAIT`]). Configure before
    /// the adapter is shared.
    pub fn with_peer_wait(mut self, wait: Duration) -> Self {
        self.inner_mut().peer_wait = wait;
        self
    }

    /// Dispatch under `identity` instead of this process's installed one
    /// (tests, and tools that host several builds). Configure before the
    /// adapter is shared.
    pub fn with_code_identity(mut self, identity: CodeIdentity) -> Self {
        let inner = self.inner_mut();
        inner.identity_digest = identity.digest();
        inner.identity = identity;
        self
    }
}

#[async_trait]
impl ExecutionAdapter for MeshExecutionAdapter {
    fn mode(&self) -> ExecutionMode {
        ExecutionMode::P2p
    }

    async fn submit(
        &self,
        request: ExecutionRequest,
    ) -> Result<Box<dyn ExecutionHandle>, ExecutionFailure> {
        if request.protocol_version != EXECUTION_PROTOCOL_VERSION {
            return Err(ExecutionFailure::protocol(format!(
                "execution protocol {} unsupported (want {EXECUTION_PROTOCOL_VERSION})",
                request.protocol_version
            )));
        }
        if request.input.is_none() {
            return Err(ExecutionFailure::protocol(
                "a mesh dispatch needs the portable input",
            ));
        }
        if request.tenant.is_restricted()
            || request.data_class == crate::framework::execution::DataClassification::Restricted
        {
            return Err(ExecutionFailure::protocol(
                "restricted data never leaves this machine",
            ));
        }
        let lifecycle = ExecutionLifecycle::new(ExecutionMode::P2p);
        let cancel = CancellationToken::new();
        let task = tokio::spawn(run_dispatch(
            self.inner.clone(),
            request,
            lifecycle.clone(),
            cancel.clone(),
        ));
        Ok(Box::new(MeshHandle {
            lifecycle,
            cancel,
            task,
        }))
    }
}

struct MeshHandle {
    lifecycle: ExecutionLifecycle,
    cancel: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for MeshHandle {
    fn drop(&mut self) {
        // Dropping the dispatch task drops its stream, which the worker sees
        // as the end of the submitter and stops the stage.
        self.task.abort();
    }
}

#[async_trait]
impl ExecutionHandle for MeshHandle {
    async fn snapshot(&self) -> Result<ExecutionSnapshot, ExecutionFailure> {
        Ok(self.lifecycle.snapshot())
    }

    async fn cancel(&self) -> Result<(), ExecutionFailure> {
        self.cancel.cancel();
        // Give the dispatch task the chance to put the cancel on the wire.
        let deadline = Instant::now() + CANCEL_GRACE;
        while self.lifecycle.snapshot().terminal.is_none() && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let _ = self.lifecycle.finish(
            None,
            ExecutionTerminal::Cancelled {
                reason: "cancelled by the submitter".into(),
            },
        );
        Ok(())
    }
}

fn unavailable(message: String) -> ExecutionFailure {
    let mut failure = ExecutionFailure::unavailable(message);
    // Waiting again would wait the same way: the wait is configured.
    failure.retryable = false;
    failure
}

fn trust_hint(id: &PeerId, class: DataClass) -> String {
    match class {
        DataClass::Internal => format!(
            "internal data needs a registered or trusted peer: `blut p2p peers trust \
             {} registered` on this machine",
            id.short()
        ),
        _ => format!("{class:?} data may not go to it under the trust matrix"),
    }
}

impl AdapterInner {
    async fn hello(&self, id: &PeerId, conn: &quinn::Connection) -> Result<WorkerHello, String> {
        let stable = conn.stable_id();
        if let Some(state) = self.peers.lock().get(id)
            && let Some((hello, at, conn_id)) = &state.hello
            && *conn_id == stable
            && at.elapsed() < HELLO_TTL
        {
            return hello.clone();
        }
        let exchange = async {
            let (mut send, mut recv) = open_stage_stream(conn).await?;
            write_msg(&mut send, &StageRequest::Hello).await?;
            let _ = send.finish();
            match read_msg::<StageReply>(&mut recv).await? {
                Some(StageReply::Hello(hello)) => Ok(hello),
                Some(StageReply::Refused(refusal)) => Err(TrainError::other(refusal.message)),
                Some(_) => Err(TrainError::other("unexpected reply to hello")),
                None => Err(TrainError::other("no reply to hello")),
            }
        };
        let hello = match tokio::time::timeout(CONTROL_TIMEOUT, exchange).await {
            Ok(Ok(hello)) => Ok(hello),
            Ok(Err(e)) => Err(e.to_string()),
            Err(_) => Err("no answer on the stage plane".to_string()),
        };
        self.peers.lock().entry(id.clone()).or_default().hello =
            Some((hello.clone(), Instant::now(), stable));
        hello
    }

    /// Every connected peer that may take this stage now, and why each other
    /// connected peer may not. `busy` counts capable peers that are full.
    async fn candidates(
        &self,
        request: &ExecutionRequest,
        excluded: &HashMap<PeerId, String>,
    ) -> (Vec<Candidate>, Vec<String>, usize) {
        let class = DataClass::from(request.data_class);
        let resources = ResourceRequest {
            cpu_cores: request.resources.cpu_cores,
            memory_gib: request.resources.memory_gib,
            gpu: request.resources.gpu,
            gpu_vram_gib: request.resources.gpu_vram_gib,
        };
        let mut out = Vec::new();
        let mut passed_over = Vec::new();
        let mut busy = 0usize;
        let mut reachable = Vec::new();
        for id in self.node.connected_peers().await {
            if let Some(reason) = excluded.get(&id) {
                passed_over.push(format!("{}: {reason}", id.short()));
                continue;
            }
            if let (Some(conn), Some(key)) = (
                self.node.connection_to(&id).await,
                self.node.peer_pubkey(&id).await,
            ) {
                reachable.push((id, conn, key));
            }
        }
        let hellos =
            futures::future::join_all(reachable.iter().map(|(id, conn, _)| self.hello(id, conn)))
                .await;
        for ((id, conn, key), hello) in reachable.into_iter().zip(hellos) {
            let name = id.short();
            let hello = match hello {
                Ok(hello) => hello,
                Err(e) => {
                    passed_over.push(format!(
                        "{name}: does not run stages ({e}); a `p2p node` worker of this \
                         release answers on the stage plane"
                    ));
                    continue;
                }
            };
            if hello.code_identity.digest() != self.identity_digest {
                passed_over.push(format!(
                    "{name}: runs {}, not this submitter's {}",
                    hello.code_identity, self.identity
                ));
                continue;
            }
            if !hello.accepts_from_you {
                passed_over.push(format!(
                    "{name}: does not accept stages from this machine; start it with \
                     --coordinator-pubkey {}",
                    crate::p2p::peer::PeerId::from_pubkey(&self.keypair.verifying).short()
                ));
                continue;
            }
            let (trust, reputation) = self
                .node
                .peer_trust_and_reputation(&id)
                .await
                .unwrap_or((TrustLevel::Anonymous, 0.5));
            let mut info = PeerInfo::new(
                key,
                x25519_dalek::PublicKey::from(hello.x25519_pub),
                trust,
                PeerCapabilities {
                    cpu_cores: hello.cpu_cores,
                    memory_gib: hello.memory_gib,
                    gpu_model: None,
                    gpu_vram_gib: None,
                },
            );
            info.reputation = reputation;
            if self
                .policy
                .select_peer(&request.stage_name, &resources, class, &[info])
                .is_none()
            {
                let why = if resources.gpu {
                    "the stage needs a GPU, and a mesh worker does not place GPU stages".to_string()
                } else if DispatchMatrix::default().can_dispatch(class, trust) {
                    format!(
                        "has {} cores and {} GiB, the stage asks for {} and {}",
                        hello.cpu_cores,
                        hello.memory_gib,
                        resources.cpu_cores,
                        resources.memory_gib
                    )
                } else {
                    format!("holds {} trust; {}", trust.label(), trust_hint(&id, class))
                };
                passed_over.push(format!("{name}: {why}"));
                continue;
            }
            let state = {
                let peers = self.peers.lock();
                peers
                    .get(&id)
                    .map(|s| (s.in_flight, s.dispatched, s.busy_until))
                    .unwrap_or((0, 0, None))
            };
            let (in_flight, dispatched, busy_until) = state;
            if in_flight >= hello.max_tasks || busy_until.is_some_and(|t| t > Instant::now()) {
                busy += 1;
                passed_over.push(format!("{name}: busy"));
                continue;
            }
            let rank = (in_flight, dispatched, -((reputation * 1_000_000.0) as i64));
            out.push(Candidate {
                id,
                conn,
                key,
                hello,
                rank,
            });
        }
        out.sort_by(|a, b| a.rank.cmp(&b.rank).then_with(|| a.id.0.cmp(&b.id.0)));
        (out, passed_over, busy)
    }

    fn sign(&self, manifest: TaskManifest) -> StageDispatch {
        sign_dispatch(&self.keypair, &self.identity, manifest)
    }
}

/// Sign `manifest`, and the dispatch of it under `identity`, with `keypair`.
fn sign_dispatch(
    keypair: &KeyPair,
    identity: &CodeIdentity,
    mut manifest: TaskManifest,
) -> StageDispatch {
    manifest.signature = keypair.sign(&manifest.sign_payload());
    let signature = keypair.sign(&dispatch_payload(&manifest, &identity.digest()));
    StageDispatch {
        manifest,
        code_identity: identity.clone(),
        signature: signature_hex(&signature),
    }
}

/// What came of offering the stage to one peer.
enum Offer {
    Accepted(quinn::SendStream, quinn::RecvStream),
    Refused(StageRefusal),
    Lost(String),
}

async fn offer(
    candidate: &Candidate,
    dispatch: &StageDispatch,
    cancel: &CancellationToken,
) -> Offer {
    let attempt = async {
        let (mut send, mut recv) = open_stage_stream(&candidate.conn).await?;
        write_msg(&mut send, &StageRequest::Run(Box::new(dispatch.clone()))).await?;
        match read_msg::<StageReply>(&mut recv).await? {
            Some(StageReply::Accepted) => Ok::<_, TrainError>(Offer::Accepted(send, recv)),
            Some(StageReply::Refused(refusal)) => Ok(Offer::Refused(refusal)),
            Some(_) => Ok(Offer::Lost("unexpected reply to a dispatch".into())),
            None => Ok(Offer::Lost("the worker closed the stream".into())),
        }
    };
    tokio::select! {
        result = tokio::time::timeout(CONTROL_TIMEOUT * 2, attempt) => match result {
            Ok(Ok(offer)) => offer,
            Ok(Err(e)) => Offer::Lost(e.to_string()),
            Err(_) => Offer::Lost("no answer to the dispatch".into()),
        },
        _ = cancel.cancelled() => Offer::Lost("cancelled".into()),
    }
}

async fn run_dispatch(
    inner: Arc<AdapterInner>,
    mut request: ExecutionRequest,
    lifecycle: ExecutionLifecycle,
    cancel: CancellationToken,
) {
    let finish_failed = |failure: ExecutionFailure| {
        let _ = lifecycle.finish(None, ExecutionTerminal::Failed { failure });
    };
    let stage_name = request.stage_name.clone();
    let input = request.input.take().expect("checked in submit");
    let input_id = input.manifest.content_id;

    // Publish the input where the chosen worker can pull it.
    let _ = lifecycle.transition(ExecutionPhase::UploadingInput, None);
    let bytes = match bincode::serialize(&input) {
        Ok(bytes) => bytes,
        Err(e) => return finish_failed(ExecutionFailure::artifact(format!("encode input: {e}"))),
    };
    drop(input);
    if let Err(e) = inner
        .objects
        .store()
        .put(ObjectKey::Artifact(input_id), bytes)
        .await
    {
        return finish_failed(ExecutionFailure::new(
            ExecutionFailureKind::Storage,
            "EXECUTION_STORAGE",
            format!("publish input of '{stage_name}': {e}"),
        ));
    }
    let _ = lifecycle.transition(ExecutionPhase::Queued, None);

    let args_bytes = serde_json::to_vec(&request.args).unwrap_or_default();
    let manifest = TaskManifest {
        protocol_version: TASK_PROTOCOL_VERSION,
        task_id: request.execution_id.clone(),
        coordinator_id: PeerId::from_pubkey(&inner.keypair.verifying),
        stage_name: stage_name.clone(),
        stage_schema: request.stage_schema,
        input_content_id: input_id,
        invocation_key: request.invocation_key,
        args_hash: ContentHash::of_bytes(&args_bytes),
        expected_content_id: request.expected_content_id,
        args: request.args.clone(),
        resources: ResourceRequest {
            cpu_cores: request.resources.cpu_cores,
            memory_gib: request.resources.memory_gib,
            gpu: request.resources.gpu,
            gpu_vram_gib: request.resources.gpu_vram_gib,
        },
        data_class: request.data_class.into(),
        timeout_secs: request.deadline.hard_remaining().as_secs().max(1),
        deadline: request.deadline,
        encrypted_input: None,
        signature: Signature::from_bytes(&[0u8; 64]),
    };
    let dispatch = inner.sign(manifest);

    // Find a peer that takes it.
    let waited = Instant::now();
    let mut excluded: HashMap<PeerId, String> = HashMap::new();
    let mut announced = false;
    let (chosen, mut send, mut recv) = loop {
        if cancel.is_cancelled() {
            let _ = lifecycle.finish(
                None,
                ExecutionTerminal::Cancelled {
                    reason: "cancelled before a peer took it".into(),
                },
            );
            return;
        }
        let (mut candidates, passed_over, busy) = inner.candidates(&request, &excluded).await;
        let mut accepted = None;
        // Choose and reserve under one lock, from the counts as they are now:
        // stages placed concurrently each see the others' reservations, so
        // they spread instead of all picking the peer that looked idlest.
        while let Some(candidate) = inner.reserve_best(&mut candidates) {
            inner.objects.grant(&candidate.id, input_id);
            match offer(&candidate, &dispatch, &cancel).await {
                Offer::Accepted(send, recv) => {
                    accepted = Some((candidate, send, recv));
                    break;
                }
                Offer::Refused(refusal) if refusal.code == RefusalCode::Busy => {
                    inner.release(&candidate.id, true);
                    inner
                        .peers
                        .lock()
                        .entry(candidate.id.clone())
                        .or_default()
                        .busy_until = Some(Instant::now() + BUSY_BACKOFF);
                }
                Offer::Refused(refusal) => {
                    inner.release(&candidate.id, true);
                    tracing::info!(
                        "stage {stage_name}: {} refused it: {}",
                        candidate.id.short(),
                        refusal.message
                    );
                    excluded.insert(candidate.id.clone(), refusal.message);
                }
                Offer::Lost(reason) => {
                    inner.release(&candidate.id, true);
                    excluded.insert(candidate.id.clone(), reason);
                }
            }
        }
        if let Some(found) = accepted {
            break found;
        }
        // Candidates `reserve_best` left behind were full when it looked.
        let busy = busy + candidates.len();
        let waiting_for_busy = busy > 0 && !request.deadline.hard_remaining().is_zero();
        if !waiting_for_busy && waited.elapsed() >= inner.peer_wait {
            let seen = if passed_over.is_empty() {
                "no peer is connected".to_string()
            } else {
                passed_over.join("; ")
            };
            return finish_failed(unavailable(format!(
                "no mesh peer could take stage '{stage_name}' within {}s, so it did not \
                 run (--launcher p2p never runs an eligible stage on the submitting \
                 machine instead): {seen}",
                inner.peer_wait.as_secs()
            )));
        }
        if !announced {
            tracing::info!("stage {stage_name}: waiting for a mesh peer that can take it");
            announced = true;
        }
        tokio::select! {
            _ = tokio::time::sleep(WAIT_POLL) => {}
            _ = cancel.cancelled() => {}
        }
    };

    let generation = inner.generation.fetch_add(1, Ordering::Relaxed) + 1;
    let assignment = Assignment::new(chosen.id.short(), generation);
    // `reserve_best` counted it; the guard gives the slot back when the
    // stage ends, however it ends.
    let _guard = InFlight {
        inner: inner.clone(),
        id: chosen.id.clone(),
    };
    let _ = lifecycle.transition(ExecutionPhase::Assigned, Some(assignment.clone()));
    tracing::info!(
        "stage {stage_name} → peer {} at {} (same code identity)",
        chosen.id.short(),
        chosen.conn.remote_address()
    );
    let started = Instant::now();

    // Follow the stage on the worker.
    let terminal = loop {
        tokio::select! {
            reply = read_msg::<StageReply>(&mut recv) => match reply {
                Ok(Some(StageReply::Running)) => {
                    let _ = lifecycle.transition(ExecutionPhase::Running, None);
                }
                Ok(Some(StageReply::Done(result))) => break Ok(*result),
                Ok(Some(StageReply::Failed(failure))) => break Err(Some(failure)),
                Ok(Some(StageReply::Cancelled { reason })) => {
                    break Err(Some(ExecutionFailure::disconnected(format!(
                        "peer {} stopped stage '{stage_name}': {reason}",
                        chosen.id.short()
                    ))));
                }
                Ok(Some(_)) => continue,
                Ok(None) | Err(_) => {
                    let why = match reply {
                        Err(e) => e.to_string(),
                        _ => "it closed the stream".into(),
                    };
                    break Err(Some(ExecutionFailure::disconnected(format!(
                        "lost peer {} while it ran stage '{stage_name}': {why}",
                        chosen.id.short()
                    ))));
                }
            },
            _ = cancel.cancelled() => {
                let _ = write_msg(&mut send, &StageRequest::Cancel {
                    reason: "the run was cancelled".into(),
                }).await;
                let _ = send.finish();
                // Wait briefly for the worker to confirm.
                let _ = tokio::time::timeout(CANCEL_GRACE, async {
                    while let Ok(Some(reply)) = read_msg::<StageReply>(&mut recv).await {
                        if matches!(reply, StageReply::Cancelled { .. } | StageReply::Done(_) | StageReply::Failed(_)) {
                            break;
                        }
                    }
                }).await;
                break Err(None);
            }
        }
    };
    let result = match terminal {
        Ok(result) => result,
        Err(None) => {
            let _ = lifecycle.finish(
                None,
                ExecutionTerminal::Cancelled {
                    reason: "cancelled".into(),
                },
            );
            return;
        }
        Err(Some(failure)) => {
            tracing::warn!(
                "stage {stage_name} failed on peer {}: {failure}",
                chosen.id.short()
            );
            inner.record(&chosen, false).await;
            let _ = lifecycle.finish(Some(&assignment), ExecutionTerminal::Failed { failure });
            return;
        }
    };
    // The result must come from the worker it was sent to, for this task.
    let worker_id = PeerId::from_pubkey(&chosen.key);
    let bad_result = if result.peer_id != worker_id {
        Some(format!(
            "the result names peer {}, not {}",
            result.peer_id.short(),
            worker_id.short()
        ))
    } else if result.task_id != dispatch.manifest.task_id {
        Some("the result is for a different task".to_string())
    } else if !verify(&chosen.key, &result.sign_payload(), &result.signature) {
        Some("the result signature does not verify against the worker's key".to_string())
    } else {
        None
    };
    if let Some(why) = bad_result {
        inner.record(&chosen, false).await;
        let _ = lifecycle.finish(
            Some(&assignment),
            ExecutionTerminal::Failed {
                failure: ExecutionFailure::artifact(format!(
                    "stage '{stage_name}' on peer {}: {why}",
                    chosen.id.short()
                )),
            },
        );
        return;
    }
    let _ = lifecycle.transition(ExecutionPhase::Running, None);
    let _ = lifecycle.transition(ExecutionPhase::DownloadingOutput, None);
    let fetch_started = Instant::now();
    let fetched = match read_stored(&inner.objects, result.content_id).await {
        Some(stored) => Ok(stored),
        None => {
            inner
                .objects
                .fetch_artifact(
                    &chosen.conn,
                    result.content_id,
                    request.deadline.hard_remaining(),
                )
                .await
        }
    };
    let stored = match fetched {
        Ok(stored) => stored,
        Err(e) => {
            inner.record(&chosen, false).await;
            let _ = lifecycle.finish(
                Some(&assignment),
                ExecutionTerminal::Failed {
                    failure: ExecutionFailure::transport(format!(
                        "fetch the output of '{stage_name}' from peer {}: {e}",
                        chosen.id.short()
                    )),
                },
            );
            return;
        }
    };
    tracing::info!(
        "stage {stage_name} ran on peer {} in {} ms ({} ms there); output {} bytes back in {} ms",
        chosen.id.short(),
        started.elapsed().as_millis(),
        result.wall_time_ms,
        stored.pack.len(),
        fetch_started.elapsed().as_millis()
    );
    inner.record(&chosen, true).await;
    let _ = lifecycle.finish(
        Some(&assignment),
        ExecutionTerminal::Succeeded {
            artifact: ExecutionArtifact {
                content_id: result.content_id,
                stored: Some(stored),
            },
            wall_time_ms: result.wall_time_ms,
        },
    );
    let _ = send.finish();
}

impl AdapterInner {
    /// Take the best of `candidates` by the current counts (fewest in flight,
    /// then least used, then reputation), skipping full or busy peers, and
    /// count it as in flight and used before the lock is released.
    fn reserve_best(&self, candidates: &mut Vec<Candidate>) -> Option<Candidate> {
        let mut peers = self.peers.lock();
        let now = Instant::now();
        let best = candidates
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let state = peers.get(&c.id);
                let in_flight = state.map_or(0, |s| s.in_flight);
                let busy = state
                    .and_then(|s| s.busy_until)
                    .is_some_and(|until| until > now);
                (in_flight < c.hello.max_tasks && !busy).then(|| {
                    let dispatched = state.map_or(0, |s| s.dispatched);
                    ((in_flight, dispatched, c.rank.2, c.id.0), i)
                })
            })
            .min()
            .map(|(_, i)| i)?;
        let candidate = candidates.swap_remove(best);
        let state = peers.entry(candidate.id.clone()).or_default();
        state.in_flight += 1;
        state.dispatched += 1;
        Some(candidate)
    }

    /// Undo [`reserve_best`](Self::reserve_best) for a peer that did not
    /// take the stage.
    fn release(&self, id: &PeerId, unused: bool) {
        if let Some(state) = self.peers.lock().get_mut(id) {
            state.in_flight = state.in_flight.saturating_sub(1);
            if unused {
                state.dispatched = state.dispatched.saturating_sub(1);
            }
        }
    }

    async fn record(&self, candidate: &Candidate, success: bool) {
        self.node
            .record_peer_outcome(&candidate.key, Some(candidate.hello.x25519_pub), success)
            .await;
    }
}

struct InFlight {
    inner: Arc<AdapterInner>,
    id: PeerId,
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.inner.release(&self.id, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::{Artifact, InvocationKey};
    use crate::framework::artifact_store::capture;
    use crate::framework::code_identity::BuildIdentity;
    use crate::framework::cookbook::Cookbook;
    use crate::framework::error::StageError;
    use crate::framework::execution::{
        DataClassification, ExecutionDeadline, ExecutionResources, ExecutionResult, drive_execution,
    };
    use crate::framework::object_store::ObjectStore;
    use crate::framework::stage::{ErasedArtifact, ErasedStageCtor, Stage, StagePlacement};
    use crate::p2p::node::{MeshTaskRunner, NodeCapabilities};
    use crate::p2p::registry::PeerRegistry;
    use crate::p2p::smoke::{SMOKE_STAGE, SmokeEcho, SmokeText};
    use crate::recipes::recipe::RecipeDef;
    use std::net::SocketAddr;

    /// Holds until cancelled, leaving marker files where its args say: what a
    /// long remote stage looks like to cancel and to a lost submitter.
    struct Hold;

    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct HoldArgs {
        markers: String,
    }

    #[async_trait]
    impl Stage for Hold {
        const NAME: &'static str = "mesh_exec_test.hold";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [crate::framework::Resource] = &[];
        type Input = SmokeText;
        type Output = SmokeText;
        type Args = HoldArgs;
        async fn run(
            &self,
            ctx: &StageContext,
            input: SmokeText,
            args: &HoldArgs,
        ) -> Result<SmokeText, StageError> {
            let markers = std::path::Path::new(&args.markers);
            std::fs::write(markers.join("started"), b"").unwrap();
            tokio::select! {
                _ = ctx.cancel.cancelled() => {
                    std::fs::write(markers.join("cancelled"), b"").unwrap();
                    Err(StageError::Cancelled)
                }
                _ = tokio::time::sleep(Duration::from_secs(120)) => Ok(input),
            }
        }
        fn placement(&self, _: &HoldArgs) -> StagePlacement {
            StagePlacement::remote(DataClassification::Public)
        }
    }

    /// Declares nothing: it must stay where its plan runs.
    struct StaysHome;

    #[async_trait]
    impl Stage for StaysHome {
        const NAME: &'static str = "mesh_exec_test.stays_home";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [crate::framework::Resource] = &[];
        type Input = SmokeText;
        type Output = SmokeText;
        type Args = serde_json::Value;
        async fn run(
            &self,
            _: &StageContext,
            input: SmokeText,
            _: &serde_json::Value,
        ) -> Result<SmokeText, StageError> {
            Ok(input)
        }
    }

    struct TestCookbook;

    impl Cookbook for TestCookbook {
        fn name(&self) -> &'static str {
            "mesh_exec_test"
        }
        fn recipes(&self) -> &'static [&'static RecipeDef] {
            &[]
        }
        fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
            static S: &[(&str, ErasedStageCtor)] = &[
                (Hold::NAME, || Arc::new(Hold)),
                (StaysHome::NAME, || Arc::new(StaysHome)),
            ];
            S
        }
    }

    struct NoTasks;

    #[async_trait]
    impl MeshTaskRunner for NoTasks {
        async fn run(&self, _: TaskManifest) -> Result<TaskResult, TrainError> {
            Err(TrainError::other("not used"))
        }
    }

    fn identity(version: &str) -> CodeIdentity {
        CodeIdentity::with_engine(BuildIdentity::engine())
            .with_cookbook(BuildIdentity::new("mesh-exec-test-cookbook", version))
    }

    fn objects(dir: &std::path::Path) -> Arc<MeshObjects> {
        Arc::new(MeshObjects::new(
            ObjectStore::filesystem(dir.join("store")),
            dir.join("chunks"),
        ))
    }

    struct Worker {
        node: Arc<MeshNode>,
        key: Arc<KeyPair>,
        addr: SocketAddr,
        _dir: tempfile::TempDir,
    }

    async fn worker(code: &CodeIdentity, accept: &KeyPair) -> Worker {
        let dir = tempfile::tempdir().unwrap();
        let key = Arc::new(KeyPair::generate());
        let mut registry = Registry::new();
        crate::p2p::smoke::register(&mut registry);
        registry.register(Box::new(TestCookbook));
        let objects = objects(dir.path());
        let node = MeshNode::bind(
            "127.0.0.1:0".parse().unwrap(),
            key.clone(),
            PeerRegistry::load(&dir.path().join("peers.json")).unwrap(),
            NodeCapabilities {
                worker: true,
                scheduler: false,
            },
            Some(Arc::new(NoTasks)),
            false,
        )
        .await
        .unwrap();
        node.set_objects(objects.clone());
        node.set_stage_worker(Arc::new(
            MeshStageWorker::new(
                Arc::new(registry),
                objects,
                key.clone(),
                dir.path().join("work"),
            )
            .accept_from(accept.verifying)
            .with_code_identity(code.clone()),
        ));
        let addr = node.local_addr().unwrap();
        Worker {
            node,
            key,
            addr,
            _dir: dir,
        }
    }

    struct Submitter {
        node: Arc<MeshNode>,
        key: Arc<KeyPair>,
        objects: Arc<MeshObjects>,
        dir: tempfile::TempDir,
    }

    async fn submitter(key: Arc<KeyPair>, workers: &[&Worker]) -> Submitter {
        let dir = tempfile::tempdir().unwrap();
        let objects = objects(dir.path());
        let node = MeshNode::bind(
            "127.0.0.1:0".parse().unwrap(),
            key.clone(),
            PeerRegistry::load(&dir.path().join("peers.json")).unwrap(),
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
            false,
        )
        .await
        .unwrap();
        node.set_objects(objects.clone());
        for w in workers {
            tokio::time::timeout(
                crate::TEST_HANG_GUARD,
                node.connect_to(w.addr, w.key.verifying.to_bytes()),
            )
            .await
            .unwrap()
            .unwrap();
        }
        Submitter {
            node,
            key,
            objects,
            dir,
        }
    }

    impl Submitter {
        fn adapter(&self, code: &CodeIdentity, wait: Duration) -> MeshExecutionAdapter {
            MeshExecutionAdapter::new(
                self.node.clone(),
                self.objects.clone(),
                self.key.clone(),
                Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default())),
            )
            .with_code_identity(code.clone())
            .with_peer_wait(wait)
        }

        /// A request for `stage` on a text input.
        fn request(
            &self,
            stage: &dyn StageDyn,
            text: &str,
            args: serde_json::Value,
            class: DataClassification,
        ) -> ExecutionRequest {
            let src = self
                .dir
                .path()
                .join(format!("src-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&src).unwrap();
            let path = src.join("in.txt");
            std::fs::write(&path, text).unwrap();
            let input = SmokeText {
                content_hash: ContentHash::hash_file(&path).unwrap(),
                path,
            };
            let stored = capture(
                stage,
                ErasedArtifact::from_typed(&input).unwrap(),
                &src,
                ArtifactRole::Input,
                None,
            )
            .unwrap();
            ExecutionRequest {
                protocol_version: EXECUTION_PROTOCOL_VERSION,
                execution_id: format!("exec-test-{}", uuid::Uuid::new_v4()),
                tenant: crate::tenant::Tenant::default(),
                stage_name: stage.name().to_string(),
                stage_schema: stage.schema(),
                invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(text.as_bytes())),
                args_hash: ContentHash::of_bytes(&serde_json::to_vec(&args).unwrap()),
                args,
                input: Some(stored),
                expected_content_id: None,
                resources: ExecutionResources {
                    cpu_cores: 1,
                    memory_gib: 0,
                    gpu: false,
                    gpu_vram_gib: None,
                },
                data_class: class,
                deadline: ExecutionDeadline::from_now(None, Duration::from_secs(120)),
            }
        }
    }

    async fn wait_for(path: &std::path::Path, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if path.exists() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        path.exists()
    }

    /// The whole path: an eligible stage goes to the worker with the same code,
    /// runs there, and its output comes back and restores, verified. The
    /// submitter's registry then counts the worker's completed task (it used
    /// to keep every peer at reputation 0.50 with nothing completed).
    #[tokio::test]
    async fn an_eligible_stage_runs_on_a_worker_and_its_output_comes_back_verified() {
        let code = identity("1");
        let sub_key = Arc::new(KeyPair::generate());
        let w = worker(&code, &sub_key).await;
        let s = submitter(sub_key, &[&w]).await;
        let adapter = s.adapter(&code, Duration::from_secs(10));
        let stage: Arc<dyn StageDyn> = Arc::new(SmokeEcho);
        let request = s.request(
            stage.as_ref(),
            "hello mesh dispatch",
            serde_json::json!({}),
            DataClassification::Public,
        );
        let out = tempfile::tempdir().unwrap();
        let result = tokio::time::timeout(
            crate::TEST_HANG_GUARD,
            drive_execution(
                &adapter,
                request,
                &CancellationToken::new(),
                stage,
                &out.path().join("attempt"),
                Duration::from_millis(20),
            ),
        )
        .await
        .unwrap();
        let ExecutionResult::Succeeded { artifact, .. } = result else {
            panic!("the stage must succeed on the worker");
        };
        let text: SmokeText = artifact.into_typed().unwrap();
        assert_eq!(
            std::fs::read_to_string(&text.path).unwrap(),
            "HELLO MESH DISPATCH"
        );
        assert!(
            text.path.starts_with(out.path()),
            "restored under the attempt"
        );
        assert_eq!(text.content_hash, text.content_hash(), "{text:?}");
        let worker_id = PeerId::from_pubkey(&w.key.verifying);
        let deadline = Instant::now() + Duration::from_secs(5);
        let reputation = loop {
            let rep = s.node.peer_trust_and_reputation(&worker_id).await;
            if rep.is_some() || Instant::now() > deadline {
                break rep;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        let (trust, reputation) = reputation.expect("the worker is in the registry now");
        assert_eq!(trust, TrustLevel::Anonymous);
        assert!(
            reputation > 0.5,
            "a completed dispatch raises it: {reputation}"
        );
    }

    /// Stages placed at the same moment go to different workers: the choice
    /// and the reservation are one step, so the second sees the first.
    #[tokio::test]
    async fn concurrent_stages_spread_over_the_workers() {
        let code = identity("1");
        let sub_key = Arc::new(KeyPair::generate());
        let w1 = worker(&code, &sub_key).await;
        let w2 = worker(&code, &sub_key).await;
        let s = submitter(sub_key, &[&w1, &w2]).await;
        let adapter = s.adapter(&code, Duration::from_secs(10));
        let stage: Arc<dyn StageDyn> = Arc::new(SmokeEcho);
        let out = tempfile::tempdir().unwrap();
        let runs = ["one", "two"].map(|text| {
            let request = s.request(
                stage.as_ref(),
                text,
                serde_json::json!({}),
                DataClassification::Public,
            );
            let (adapter, stage) = (adapter.clone(), stage.clone());
            let dir = out.path().join(text);
            tokio::spawn(async move {
                drive_execution(
                    &adapter,
                    request,
                    &CancellationToken::new(),
                    stage,
                    &dir,
                    Duration::from_millis(20),
                )
                .await
            })
        });
        for run in runs {
            let result = tokio::time::timeout(crate::TEST_HANG_GUARD, run)
                .await
                .unwrap()
                .unwrap();
            assert!(matches!(result, ExecutionResult::Succeeded { .. }));
        }
        for w in [&w1, &w2] {
            let id = PeerId::from_pubkey(&w.key.verifying);
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if let Some((_, rep)) = s.node.peer_trust_and_reputation(&id).await
                    && rep > 0.5
                {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "worker {} ran nothing",
                    id.short()
                );
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }

    async fn raw_run(s: &Submitter, w: &Worker, dispatch: &StageDispatch) -> StageReply {
        let conn = s
            .node
            .connection_to(&PeerId::from_pubkey(&w.key.verifying))
            .await
            .unwrap();
        let (mut send, mut recv) = open_stage_stream(&conn).await.unwrap();
        write_msg(&mut send, &StageRequest::Run(Box::new(dispatch.clone())))
            .await
            .unwrap();
        tokio::time::timeout(crate::TEST_HANG_GUARD, read_msg::<StageReply>(&mut recv))
            .await
            .unwrap()
            .unwrap()
            .unwrap()
    }

    fn manifest(s: &Submitter, stage: &str, args: serde_json::Value) -> TaskManifest {
        TaskManifest {
            protocol_version: TASK_PROTOCOL_VERSION,
            task_id: format!("raw-{}", uuid::Uuid::new_v4()),
            coordinator_id: PeerId::from_pubkey(&s.key.verifying),
            stage_name: stage.into(),
            stage_schema: 1,
            input_content_id: ArtifactContentId::from_digest(ContentHash::of_bytes(b"in")),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"raw")),
            args_hash: ContentHash::of_bytes(&serde_json::to_vec(&args).unwrap()),
            expected_content_id: None,
            args,
            resources: ResourceRequest::default(),
            data_class: DataClass::Public,
            timeout_secs: 60,
            deadline: ExecutionDeadline::from_now(None, Duration::from_secs(60)),
            encrypted_input: None,
            signature: Signature::from_bytes(&[0u8; 64]),
        }
    }

    fn refusal(reply: StageReply) -> StageRefusal {
        match reply {
            StageReply::Refused(refusal) => refusal,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// The worker refuses a stage from a submitter running other code, even
    /// when the submitter skips its own check: computing it with different
    /// code under the same cache key would be computing something else.
    #[tokio::test]
    async fn a_worker_refuses_a_submitter_whose_code_identity_differs() {
        let sub_key = Arc::new(KeyPair::generate());
        let w = worker(&identity("1"), &sub_key).await;
        let s = submitter(sub_key.clone(), &[&w]).await;
        let other = identity("2");
        let dispatch = sign_dispatch(
            &sub_key,
            &other,
            manifest(&s, SMOKE_STAGE, serde_json::json!({})),
        );
        let refused = refusal(raw_run(&s, &w, &dispatch).await);
        assert_eq!(
            refused.code,
            RefusalCode::CodeIdentityMismatch,
            "{refused:?}"
        );
        assert!(
            refused.message.contains("mesh-exec-test-cookbook 2"),
            "{}",
            refused.message
        );
        // The same request under the worker's own identity gets past that
        // check (and fails later, on the input it does not have).
        let same = sign_dispatch(
            &sub_key,
            &identity("1"),
            manifest(&s, SMOKE_STAGE, serde_json::json!({})),
        );
        assert!(!matches!(
            raw_run(&s, &w, &same).await,
            StageReply::Refused(StageRefusal {
                code: RefusalCode::CodeIdentityMismatch,
                ..
            })
        ));
    }

    /// Unknown stages, stages that declare they stay home, submitters the
    /// worker was not told to accept, and forged dispatches are refused
    /// before anything is fetched or run.
    #[tokio::test]
    async fn a_worker_refuses_unknown_local_only_unaccepted_and_forged_work() {
        let code = identity("1");
        let sub_key = Arc::new(KeyPair::generate());
        let w = worker(&code, &sub_key).await;
        let s = submitter(sub_key.clone(), &[&w]).await;
        let unknown = sign_dispatch(
            &sub_key,
            &code,
            manifest(&s, "no_such_stage", serde_json::json!({})),
        );
        assert_eq!(
            refusal(raw_run(&s, &w, &unknown).await).code,
            RefusalCode::UnknownStage
        );
        let home = sign_dispatch(
            &sub_key,
            &code,
            manifest(&s, StaysHome::NAME, serde_json::json!({})),
        );
        assert_eq!(
            refusal(raw_run(&s, &w, &home).await).code,
            RefusalCode::StaysLocal
        );
        let mut forged = sign_dispatch(
            &sub_key,
            &code,
            manifest(&s, SMOKE_STAGE, serde_json::json!({})),
        );
        forged.manifest.args = serde_json::json!({ "swapped": true });
        assert_eq!(
            refusal(raw_run(&s, &w, &forged).await).code,
            RefusalCode::BadRequest
        );

        let stranger_key = Arc::new(KeyPair::generate());
        let stranger = submitter(stranger_key.clone(), &[&w]).await;
        let theirs = sign_dispatch(
            &stranger_key,
            &code,
            manifest(&stranger, SMOKE_STAGE, serde_json::json!({})),
        );
        assert_eq!(
            refusal(raw_run(&stranger, &w, &theirs).await).code,
            RefusalCode::NotAccepted
        );
    }

    /// With no peer able to take an eligible stage, the attempt fails as
    /// unavailable within the configured wait, not retryable, naming why
    /// each peer was passed over.
    #[tokio::test]
    async fn no_capable_peer_fails_the_attempt_and_says_why() {
        let sub_key = Arc::new(KeyPair::generate());
        let w = worker(&identity("1"), &sub_key).await;
        let s = submitter(sub_key, &[&w]).await;
        let stage: Arc<dyn StageDyn> = Arc::new(SmokeEcho);
        let cases = [
            (identity("2"), DataClassification::Public, "runs"),
            (identity("1"), DataClassification::Internal, "registered"),
        ];
        for (code, class, why) in cases {
            let adapter = s.adapter(&code, Duration::from_secs(1));
            let request = s.request(stage.as_ref(), "x", serde_json::json!({}), class);
            let started = Instant::now();
            let out = tempfile::tempdir().unwrap();
            let result = drive_execution(
                &adapter,
                request,
                &CancellationToken::new(),
                stage.clone(),
                &out.path().join("a"),
                Duration::from_millis(20),
            )
            .await;
            let ExecutionResult::Failed(failure) = result else {
                panic!("{why}: must fail");
            };
            assert_eq!(failure.kind, ExecutionFailureKind::Unavailable, "{failure}");
            assert!(!failure.retryable, "{failure}");
            assert!(failure.message.contains(why), "{why}: {failure}");
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "bounded by the wait"
            );
        }
    }

    async fn start_hold(
        s: &Submitter,
        code: &CodeIdentity,
        markers: &std::path::Path,
        cancel: CancellationToken,
    ) -> tokio::task::JoinHandle<ExecutionResult> {
        let adapter = s.adapter(code, Duration::from_secs(10));
        let stage: Arc<dyn StageDyn> = Arc::new(Hold);
        let request = s.request(
            stage.as_ref(),
            "hold",
            serde_json::json!({ "markers": markers }),
            DataClassification::Public,
        );
        let out = s.dir.path().join(format!("out-{}", uuid::Uuid::new_v4()));
        tokio::spawn(async move {
            drive_execution(
                &adapter,
                request,
                &cancel,
                stage,
                &out,
                Duration::from_millis(20),
            )
            .await
        })
    }

    /// Cancelling the run reaches the stage running on the worker.
    #[tokio::test]
    async fn cancelling_the_run_stops_the_remote_stage() {
        let code = identity("1");
        let sub_key = Arc::new(KeyPair::generate());
        let w = worker(&code, &sub_key).await;
        let s = submitter(sub_key, &[&w]).await;
        let markers = tempfile::tempdir().unwrap();
        let cancel = CancellationToken::new();
        let run = start_hold(&s, &code, markers.path(), cancel.clone()).await;
        assert!(wait_for(&markers.path().join("started"), Duration::from_secs(30)).await);
        cancel.cancel();
        let result = tokio::time::timeout(crate::TEST_HANG_GUARD, run)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(result, ExecutionResult::Cancelled));
        assert!(
            wait_for(&markers.path().join("cancelled"), Duration::from_secs(10)).await,
            "the worker's stage saw the cancel"
        );
    }

    /// A worker lost mid-stage fails the stage as a lost peer, retryable by
    /// the stage's own policy. Nothing falls back to this machine.
    #[tokio::test]
    async fn losing_the_worker_fails_the_stage() {
        let code = identity("1");
        let sub_key = Arc::new(KeyPair::generate());
        let w = worker(&code, &sub_key).await;
        let s = submitter(sub_key, &[&w]).await;
        let markers = tempfile::tempdir().unwrap();
        let run = start_hold(&s, &code, markers.path(), CancellationToken::new()).await;
        assert!(wait_for(&markers.path().join("started"), Duration::from_secs(30)).await);
        w.node.shutdown(Duration::from_secs(2)).await;
        let result = tokio::time::timeout(crate::TEST_HANG_GUARD, run)
            .await
            .unwrap()
            .unwrap();
        let ExecutionResult::Failed(failure) = result else {
            panic!("a lost worker must fail the stage");
        };
        assert_eq!(
            failure.kind,
            ExecutionFailureKind::Disconnected,
            "{failure}"
        );
        assert!(failure.message.contains("lost peer"), "{failure}");
    }

    /// A worker whose submitter goes away stops the stage it was running.
    #[tokio::test]
    async fn a_worker_that_loses_its_submitter_stops_the_stage() {
        let code = identity("1");
        let sub_key = Arc::new(KeyPair::generate());
        let w = worker(&code, &sub_key).await;
        let s = submitter(sub_key, &[&w]).await;
        let markers = tempfile::tempdir().unwrap();
        let run = start_hold(&s, &code, markers.path(), CancellationToken::new()).await;
        assert!(wait_for(&markers.path().join("started"), Duration::from_secs(30)).await);
        s.node.shutdown(Duration::from_secs(2)).await;
        assert!(
            wait_for(&markers.path().join("cancelled"), Duration::from_secs(10)).await,
            "the worker's stage stopped when its submitter closed"
        );
        run.abort();
    }
}
