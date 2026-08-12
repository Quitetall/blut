//! Coordinator side of cloud dispatch (ADR 0067 · T3.1c) — the object-store analog
//! of `p2p::peer_exec::dispatch_to_peer`, split into submit + poll.
//!
//! `submit` bundles the input artifact (reusing `p2p::bundle::bundle`), uploads the
//! pack to the object store keyed by its hash, and enqueues a `CloudJob`. The
//! returned [`CloudJobHandle`] polls the queue; once the worker reports success it
//! downloads + `unbundle`s the output, which runs the four fail-closed gates
//! (including the bind to `expected_output_hash`) — so a wrong/garbled result is
//! rejected exactly as in the P2P path.
//!
//! v1 ships the bundle manifest in the job record + the pack in the store in the
//! clear: the object store is operator-controlled and clinical data is hard-blocked
//! from cloud (see [`crate::cloud`]). At-rest encryption (reusing `p2p::crypto`) and
//! the verified-novel-work model are T3.2 (untrusted multi-tenant providers).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::CloudError;
use super::job::{CloudJob, JobOutcome};
use super::queue::{CloudQueue, JobStatus};
use crate::framework::Registry;
use crate::framework::artifact::{ContentHash, ContentId, InvocationKey};
use crate::framework::artifact_store::StoredArtifact;
use crate::framework::cache::CacheHandle;
use crate::framework::execution::{
    Assignment, ExecutionAdapter, ExecutionArtifact, ExecutionDeadline, ExecutionFailure,
    ExecutionFailureKind, ExecutionHandle, ExecutionLifecycle, ExecutionMode, ExecutionPhase,
    ExecutionRequest, ExecutionSnapshot, ExecutionTerminal, LifecycleError,
};
use crate::framework::object_store::{MAX_OBJECT_SIZE, ObjectKey, ObjectStore};
use crate::framework::stage::ErasedArtifact;
use crate::p2p::bundle::{BlobDir, bundle, unbundle};
use crate::p2p::task::ResourceRequest;
use crate::p2p::trust::DataClass;

/// Everything `submit` needs to dispatch one stage. Mirrors `dispatch_to_peer`'s
/// argument list; `expected_output_hash` is caller-supplied (analytic for a
/// deterministic stage, or a cache-derived address for verified re-execution).
pub struct CloudSubmitSpec {
    pub job_id: String,
    pub tenant: crate::tenant::Tenant,
    pub stage_name: String,
    pub invocation_key: InvocationKey,
    pub input: ErasedArtifact,
    pub src_root: PathBuf,
    pub args: Value,
    pub expected_content_id: Option<ContentId>,
    pub data_class: DataClass,
    pub resources: ResourceRequest,
    pub priority: i32,
    pub timeout_secs: u64,
}

/// Submits cloud jobs over an [`ObjectStore`] + [`CloudQueue`], resolving stages from
/// a `Registry` (needed to bundle/unbundle the typed artifact).
#[derive(Clone)]
pub struct CloudSubmitter {
    store: ObjectStore,
    queue: Arc<dyn CloudQueue>,
    registry: Arc<Registry>,
}

/// What [`CloudJobHandle::poll`] reports.
pub enum CloudPoll {
    /// Enqueued or executing — keep polling.
    Pending,
    /// Done + the output downloaded, unbundled, and content-verified.
    Succeeded(ErasedArtifact),
    /// The worker failed the job.
    Failed(String),
    /// The job was cancelled.
    Cancelled,
    /// The end-to-end execution deadline elapsed.
    TimedOut,
    /// No such job (evicted / never enqueued).
    Unknown,
}

impl CloudSubmitter {
    pub fn new(store: ObjectStore, queue: Arc<dyn CloudQueue>, registry: Arc<Registry>) -> Self {
        Self {
            store,
            queue,
            registry,
        }
    }

    /// Bundle `spec.input`, upload it, and enqueue the job. Returns a handle to poll.
    pub async fn submit(&self, spec: CloudSubmitSpec) -> Result<CloudJobHandle, CloudError> {
        if !crate::p2p::trust::custody_allows_off_box(&spec.tenant, spec.data_class) {
            return Err(CloudError::Dispatch(format!(
                "cloud execution denied by custody policy for tenant '{}' and {:?} data",
                spec.tenant, spec.data_class
            )));
        }
        let factory = self
            .registry
            .find_erased_stage(&spec.stage_name)
            .ok_or_else(|| CloudError::Dispatch(format!("unknown stage '{}'", spec.stage_name)))?;
        let stage = factory();

        // Bundle the input rooted at its producing dir (reuses the p2p data plane).
        let (manifest, pack) = bundle(&*stage, spec.input, &spec.src_root, BlobDir::Input, None)
            .map_err(|e| CloudError::Artifact(format!("bundle input: {e}")))?;

        // Upload the pack keyed by its own hash; the small manifest rides the job.
        let blob_key = ContentHash::of_bytes(&pack);
        self.store
            .put(ObjectKey::DispatchBundle(blob_key), pack)
            .await?;

        let args_hash = ContentHash::of_bytes(&CacheHandle::canonical_json_bytes(&spec.args));
        let job = CloudJob {
            protocol_version: super::job::CLOUD_JOB_PROTOCOL_VERSION,
            id: spec.job_id.clone(),
            tenant: spec.tenant,
            stage_name: spec.stage_name.clone(),
            stage_schema: stage.schema(),
            invocation_key: spec.invocation_key,
            args_hash,
            args: spec.args,
            input_blob_key: blob_key,
            input_manifest: manifest,
            expected_content_id: spec.expected_content_id,
            resources: spec.resources,
            data_class: spec.data_class,
            priority: spec.priority,
            timeout_secs: spec.timeout_secs,
            deadline: ExecutionDeadline::from_now(
                None,
                std::time::Duration::from_secs(spec.timeout_secs.max(1)),
            ),
        };
        self.queue.enqueue(job).await?;

        Ok(CloudJobHandle {
            store: self.store.clone(),
            queue: self.queue.clone(),
            registry: self.registry.clone(),
            job_id: spec.job_id,
            stage_name: spec.stage_name,
            expected_content_id: spec.expected_content_id,
        })
    }
}

struct CloudExecutionHandle {
    store: ObjectStore,
    queue: Arc<dyn CloudQueue>,
    job_id: String,
    lifecycle: ExecutionLifecycle,
    cancel: CancellationToken,
    enqueued: Arc<AtomicBool>,
    /// Serializes queue refresh and terminal download across concurrent snapshots.
    refresh: Mutex<()>,
}

impl Drop for CloudExecutionHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
        if self.enqueued.load(Ordering::Acquire)
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            let queue = self.queue.clone();
            let job_id = self.job_id.clone();
            runtime.spawn(async move {
                let cancel = queue.cancel(&job_id, "execution handle dropped");
                let _ = tokio::time::timeout(std::time::Duration::from_millis(250), cancel).await;
            });
        }
    }
}

#[async_trait]
impl ExecutionHandle for CloudExecutionHandle {
    async fn snapshot(&self) -> Result<ExecutionSnapshot, ExecutionFailure> {
        let _refresh = self.refresh.lock().await;
        if self.lifecycle.snapshot().terminal.is_some() || !self.enqueued.load(Ordering::Acquire) {
            return Ok(self.lifecycle.snapshot());
        }

        match self
            .queue
            .status(&self.job_id)
            .await
            .map_err(cloud_failure)?
        {
            JobStatus::Queued => {
                let current = self.lifecycle.snapshot();
                if current.phase == ExecutionPhase::Running {
                    transition(&self.lifecycle, ExecutionPhase::Queued, None)?;
                }
            }
            JobStatus::Running(assignment) => {
                observe_assignment(&self.lifecycle, &assignment)?;
            }
            JobStatus::Done(result) => {
                observe_terminal(&self.store, &self.lifecycle, *result).await?;
            }
            JobStatus::Unknown => {
                finish(
                    &self.lifecycle,
                    None,
                    ExecutionTerminal::Failed {
                        failure: ExecutionFailure::unavailable(format!(
                            "cloud job '{}' is no longer present",
                            self.job_id
                        )),
                    },
                )?;
            }
        }
        Ok(self.lifecycle.snapshot())
    }

    async fn cancel(&self) -> Result<(), ExecutionFailure> {
        self.cancel.cancel();
        let was_enqueued = self.enqueued.load(Ordering::Acquire);
        let queue_cancel = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            self.queue
                .cancel(&self.job_id, "execution caller cancelled"),
        )
        .await;
        let cancellation_won = !was_enqueued || matches!(&queue_cancel, Ok(Ok(true)));
        if cancellation_won && self.lifecycle.snapshot().terminal.is_none() {
            finish(
                &self.lifecycle,
                None,
                ExecutionTerminal::Cancelled {
                    reason: "execution caller cancelled".into(),
                },
            )?;
        }
        match queue_cancel {
            Ok(Ok(_)) | Err(_) => Ok(()),
            Ok(Err(error)) => Err(cloud_failure(error)),
        }
    }
}

struct CloudAttempt {
    store: ObjectStore,
    queue: Arc<dyn CloudQueue>,
    lifecycle: ExecutionLifecycle,
    cancel: CancellationToken,
    enqueued: Arc<AtomicBool>,
    request: ExecutionRequest,
}

impl CloudAttempt {
    async fn run(self) {
        let _guard = CloudAttemptGuard {
            lifecycle: self.lifecycle.clone(),
            enqueued: self.enqueued.clone(),
        };
        if let Err(failure) = self.run_inner().await {
            let _ = finish(&self.lifecycle, None, ExecutionTerminal::Failed { failure });
        }
    }

    async fn run_inner(&self) -> Result<(), ExecutionFailure> {
        transition(&self.lifecycle, ExecutionPhase::UploadingInput, None)?;
        let input_key = ContentHash::of_bytes(&self.request.input.pack);
        if input_key != self.request.input.manifest.blob_sha256
            || self.request.input.pack.len() as u64 != self.request.input.manifest.blob_len
        {
            return Err(ExecutionFailure::artifact(
                "canonical input pack does not match its manifest",
            ));
        }

        let upload = self.store.put(
            ObjectKey::DispatchBundle(input_key),
            self.request.input.pack.clone(),
        );
        tokio::pin!(upload);
        tokio::select! {
            result = &mut upload => {
                result.map_err(CloudError::from).map_err(cloud_failure)?;
            }
            _ = self.cancel.cancelled() => {
                finish_cancel_or_timeout(&self.lifecycle, self.request.deadline, "cancelled while uploading cloud input");
                return Ok(());
            }
            _ = sleep_optional(self.request.deadline.soft_remaining()) => {
                finish_timeout(&self.lifecycle, self.request.deadline.soft_unix_ms.unwrap_or(self.request.deadline.hard_unix_ms));
                return Ok(());
            }
            _ = tokio::time::sleep(self.request.deadline.hard_remaining()) => {
                finish_timeout(&self.lifecycle, self.request.deadline.hard_unix_ms);
                return Ok(());
            }
        }

        if self.cancel.is_cancelled() {
            finish_cancel_or_timeout(
                &self.lifecycle,
                self.request.deadline,
                "cancelled before cloud enqueue",
            );
            return Ok(());
        }

        let job = CloudJob {
            protocol_version: super::job::CLOUD_JOB_PROTOCOL_VERSION,
            id: self.request.execution_id.clone(),
            tenant: self.request.tenant.clone(),
            stage_name: self.request.stage_name.clone(),
            stage_schema: self.request.stage_schema,
            invocation_key: self.request.invocation_key,
            args_hash: self.request.args_hash,
            args: self.request.args.clone(),
            input_blob_key: input_key,
            input_manifest: self.request.input.manifest.clone(),
            expected_content_id: self.request.expected_content_id,
            resources: ResourceRequest {
                cpu_cores: self.request.resources.cpu_cores,
                memory_gib: self.request.resources.memory_gib,
                gpu: self.request.resources.gpu,
                gpu_vram_gib: self.request.resources.gpu_vram_gib,
            },
            data_class: self.request.data_class.into(),
            priority: 0,
            timeout_secs: self.request.deadline.hard_remaining().as_secs().max(1),
            deadline: self.request.deadline,
        };
        let enqueue = self.queue.enqueue(job);
        tokio::pin!(enqueue);
        tokio::select! {
            result = &mut enqueue => result.map_err(cloud_failure)?,
            _ = self.cancel.cancelled() => {
                finish_cancel_or_timeout(&self.lifecycle, self.request.deadline, "cancelled while enqueueing cloud work");
                return Ok(());
            }
            _ = sleep_optional(self.request.deadline.soft_remaining()) => {
                finish_timeout(&self.lifecycle, self.request.deadline.soft_unix_ms.unwrap_or(self.request.deadline.hard_unix_ms));
                return Ok(());
            }
            _ = tokio::time::sleep(self.request.deadline.hard_remaining()) => {
                finish_timeout(&self.lifecycle, self.request.deadline.hard_unix_ms);
                return Ok(());
            }
        }
        self.enqueued.store(true, Ordering::Release);
        transition(&self.lifecycle, ExecutionPhase::Queued, None)?;

        // Keep a cancellation owner alive after enqueue. This closes the submit
        // future race: dropping the returned handle still revokes durable work.
        let reason = tokio::select! {
            _ = self.cancel.cancelled() => "execution caller cancelled",
            _ = sleep_optional(self.request.deadline.soft_remaining()) => "cloud execution soft deadline",
            _ = tokio::time::sleep(self.request.deadline.hard_remaining()) => "cloud execution hard deadline",
        };
        let cancel = self.queue.cancel(&self.request.execution_id, reason);
        if matches!(
            tokio::time::timeout(std::time::Duration::from_millis(250), cancel).await,
            Ok(Ok(true))
        ) {
            finish_cancel_or_timeout(&self.lifecycle, self.request.deadline, reason);
        }
        Ok(())
    }
}

struct CloudAttemptGuard {
    lifecycle: ExecutionLifecycle,
    enqueued: Arc<AtomicBool>,
}

impl Drop for CloudAttemptGuard {
    fn drop(&mut self) {
        let snapshot = self.lifecycle.snapshot();
        if !self.enqueued.load(Ordering::Acquire) && snapshot.terminal.is_none() {
            let _ = self.lifecycle.finish(
                None,
                ExecutionTerminal::Failed {
                    failure: ExecutionFailure::new(
                        ExecutionFailureKind::Unknown,
                        "EXECUTION_ABANDONED",
                        "cloud adapter exited before enqueue or terminal outcome",
                    ),
                },
            );
        }
    }
}

#[async_trait]
impl ExecutionAdapter for CloudSubmitter {
    fn mode(&self) -> ExecutionMode {
        ExecutionMode::Cloud
    }

    async fn submit(
        &self,
        request: ExecutionRequest,
    ) -> Result<Box<dyn ExecutionHandle>, ExecutionFailure> {
        if request.protocol_version != crate::framework::execution::EXECUTION_PROTOCOL_VERSION {
            return Err(ExecutionFailure::protocol(format!(
                "execution protocol v{} unsupported (want v{})",
                request.protocol_version,
                crate::framework::execution::EXECUTION_PROTOCOL_VERSION,
            )));
        }
        if !crate::p2p::trust::custody_allows_off_box(&request.tenant, request.data_class.into()) {
            return Err(ExecutionFailure::protocol(format!(
                "cloud execution denied by custody policy for tenant '{}' and {:?} data",
                request.tenant, request.data_class
            )));
        }
        let stage = self
            .registry
            .find_erased_stage(&request.stage_name)
            .ok_or_else(|| {
                ExecutionFailure::protocol(format!("unknown stage '{}'", request.stage_name))
            })?();
        if stage.schema() != request.stage_schema {
            return Err(ExecutionFailure::protocol(format!(
                "stage '{}' schema {} does not match request {}",
                request.stage_name,
                stage.schema(),
                request.stage_schema
            )));
        }

        let lifecycle = ExecutionLifecycle::new(ExecutionMode::Cloud);
        let cancel = CancellationToken::new();
        let enqueued = Arc::new(AtomicBool::new(false));
        tokio::spawn(
            CloudAttempt {
                store: self.store.clone(),
                queue: self.queue.clone(),
                lifecycle: lifecycle.clone(),
                cancel: cancel.clone(),
                enqueued: enqueued.clone(),
                request: request.clone(),
            }
            .run(),
        );
        Ok(Box::new(CloudExecutionHandle {
            store: self.store.clone(),
            queue: self.queue.clone(),
            job_id: request.execution_id,
            lifecycle,
            cancel,
            enqueued,
            refresh: Mutex::new(()),
        }))
    }
}

#[allow(clippy::result_large_err)] // Canonical adapter trait fixes ExecutionFailure by value.
fn observe_assignment(
    lifecycle: &ExecutionLifecycle,
    assignment: &Assignment,
) -> Result<(), ExecutionFailure> {
    let current = lifecycle.snapshot();
    if current.terminal.is_some() {
        return Ok(());
    }
    if current.assignment.as_ref() != Some(assignment) {
        if current.phase == ExecutionPhase::Running {
            transition(lifecycle, ExecutionPhase::Queued, None)?;
        }
        transition(
            lifecycle,
            ExecutionPhase::Assigned,
            Some(assignment.clone()),
        )?;
    }
    transition(lifecycle, ExecutionPhase::Running, Some(assignment.clone()))
}

async fn observe_terminal(
    store: &ObjectStore,
    lifecycle: &ExecutionLifecycle,
    result: super::job::CloudResult,
) -> Result<(), ExecutionFailure> {
    if result.protocol_version != super::job::CLOUD_JOB_PROTOCOL_VERSION {
        return Err(ExecutionFailure::protocol(format!(
            "cloud result protocol v{} unsupported (want v{})",
            result.protocol_version,
            super::job::CLOUD_JOB_PROTOCOL_VERSION
        )));
    }
    if let Some(assignment) = &result.assignment {
        observe_assignment(lifecycle, assignment)?;
    }
    if lifecycle.snapshot().terminal.is_some() {
        return Ok(());
    }

    match result.outcome {
        JobOutcome::Failed => finish(
            lifecycle,
            result.assignment.as_ref(),
            ExecutionTerminal::Failed {
                failure: result.failure.unwrap_or_else(|| {
                    ExecutionFailure::new(
                        ExecutionFailureKind::Unknown,
                        "EXECUTION_UNKNOWN",
                        "cloud worker returned failure without typed details",
                    )
                }),
            },
        ),
        JobOutcome::Cancelled => finish(
            lifecycle,
            result.assignment.as_ref(),
            ExecutionTerminal::Cancelled {
                reason: result
                    .failure
                    .map(|failure| failure.message)
                    .unwrap_or_else(|| "cloud job cancelled".into()),
            },
        ),
        JobOutcome::TimedOut => finish(
            lifecycle,
            result.assignment.as_ref(),
            ExecutionTerminal::TimedOut {
                phase: result.timeout_phase.unwrap_or(ExecutionPhase::Running),
                deadline_unix_ms: result.deadline_unix_ms.unwrap_or_default(),
            },
        ),
        JobOutcome::Succeeded => {
            let assignment = result.assignment.as_ref().ok_or_else(|| {
                ExecutionFailure::protocol("successful cloud result has no assignment")
            })?;
            transition(
                lifecycle,
                ExecutionPhase::DownloadingOutput,
                Some(assignment.clone()),
            )?;
            let blob_key = result.output_blob_key.ok_or_else(|| {
                ExecutionFailure::artifact("successful cloud result has no output blob key")
            })?;
            let manifest = result.output_manifest.ok_or_else(|| {
                ExecutionFailure::artifact("successful cloud result has no output manifest")
            })?;
            let content_id = result.content_id.ok_or_else(|| {
                ExecutionFailure::artifact("successful cloud result has no content identity")
            })?;
            if manifest.content_id != content_id {
                return Err(ExecutionFailure::artifact(format!(
                    "cloud result identity {content_id} != manifest {}",
                    manifest.content_id
                )));
            }
            if manifest.blob_len > MAX_OBJECT_SIZE {
                return Err(ExecutionFailure::artifact(format!(
                    "cloud output bundle declares {} bytes; maximum is {MAX_OBJECT_SIZE}",
                    manifest.blob_len
                )));
            }
            let pack = store
                .get(ObjectKey::DispatchBundle(blob_key))
                .await
                .map_err(CloudError::from)
                .map_err(cloud_failure)?
                .ok_or_else(|| {
                    ExecutionFailure::artifact(format!(
                        "cloud output bundle {} is missing",
                        blob_key.to_hex()
                    ))
                })?;
            if ContentHash::of_bytes(&pack) != blob_key {
                return Err(ExecutionFailure::artifact(
                    "cloud output object key does not match downloaded bytes",
                ));
            }
            finish(
                lifecycle,
                Some(assignment),
                ExecutionTerminal::Succeeded {
                    artifact: ExecutionArtifact {
                        content_id,
                        stored: Some(StoredArtifact { manifest, pack }),
                    },
                    wall_time_ms: result.wall_time_ms,
                },
            )
        }
    }
}

#[allow(clippy::result_large_err)] // Canonical adapter trait fixes ExecutionFailure by value.
fn transition(
    lifecycle: &ExecutionLifecycle,
    phase: ExecutionPhase,
    assignment: Option<Assignment>,
) -> Result<(), ExecutionFailure> {
    match lifecycle.transition(phase, assignment) {
        Ok(_) | Err(LifecycleError::AlreadyTerminal) => Ok(()),
        Err(error) => Err(ExecutionFailure::protocol(format!(
            "cloud lifecycle transition: {error}"
        ))),
    }
}

#[allow(clippy::result_large_err)] // Canonical adapter trait fixes ExecutionFailure by value.
fn finish(
    lifecycle: &ExecutionLifecycle,
    assignment: Option<&Assignment>,
    terminal: ExecutionTerminal,
) -> Result<(), ExecutionFailure> {
    match lifecycle.finish(assignment, terminal) {
        Ok(_) | Err(LifecycleError::AlreadyTerminal) => Ok(()),
        Err(error) => Err(ExecutionFailure::protocol(format!(
            "cloud lifecycle completion: {error}"
        ))),
    }
}

fn finish_timeout(lifecycle: &ExecutionLifecycle, deadline_unix_ms: u64) {
    let _ = lifecycle.finish(
        None,
        ExecutionTerminal::TimedOut {
            phase: lifecycle.snapshot().phase,
            deadline_unix_ms,
        },
    );
}

fn finish_cancel_or_timeout(
    lifecycle: &ExecutionLifecycle,
    deadline: ExecutionDeadline,
    reason: &str,
) {
    if let Some(soft) = deadline.soft_remaining()
        && soft.is_zero()
    {
        finish_timeout(
            lifecycle,
            deadline.soft_unix_ms.expect("soft deadline exists"),
        );
    } else if deadline.hard_remaining().is_zero() {
        finish_timeout(lifecycle, deadline.hard_unix_ms);
    } else {
        let _ = lifecycle.finish(
            None,
            ExecutionTerminal::Cancelled {
                reason: reason.into(),
            },
        );
    }
}

async fn sleep_optional(duration: Option<std::time::Duration>) {
    match duration {
        Some(duration) => tokio::time::sleep(duration).await,
        None => std::future::pending::<()>().await,
    }
}

fn cloud_failure(error: CloudError) -> ExecutionFailure {
    match error {
        CloudError::Store(error) => ExecutionFailure::new(
            ExecutionFailureKind::Storage,
            "EXECUTION_STORAGE",
            error.to_string(),
        ),
        CloudError::Artifact(message) => ExecutionFailure::artifact(message),
        CloudError::Dispatch(message) => ExecutionFailure::protocol(message),
        CloudError::BadJobId(id) => {
            ExecutionFailure::protocol(format!("unsafe cloud job id '{id}'"))
        }
        CloudError::DuplicateJob(id) => {
            ExecutionFailure::protocol(format!("duplicate cloud job id '{id}'"))
        }
        CloudError::LeaseLost(id) => {
            ExecutionFailure::unavailable(format!("cloud job '{id}' lost its lease"))
        }
    }
}

/// Handle to a submitted cloud job: poll status, and on success download + verify
/// the output bundle. The CLI wrapper keeps this explicit while the A08 cloud
/// adapter consumes the same queue state through `ExecutionHandle`.
pub struct CloudJobHandle {
    store: ObjectStore,
    queue: Arc<dyn CloudQueue>,
    registry: Arc<Registry>,
    job_id: String,
    stage_name: String,
    expected_content_id: Option<ContentId>,
}

impl CloudJobHandle {
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    /// Poll the queue. On terminal success, download the output bundle into
    /// `out_dir`, `unbundle` it (running the four fail-closed gates, including the
    /// bind to `expected_output_hash`), and return the rehydrated artifact.
    pub async fn poll(&self, out_dir: &Path) -> Result<CloudPoll, CloudError> {
        match self.queue.status(&self.job_id).await? {
            JobStatus::Queued | JobStatus::Running(_) => Ok(CloudPoll::Pending),
            JobStatus::Unknown => Ok(CloudPoll::Unknown),
            JobStatus::Done(result) => {
                if result.protocol_version != super::job::CLOUD_JOB_PROTOCOL_VERSION {
                    return Err(CloudError::Artifact(format!(
                        "cloud result protocol v{} unsupported (want v{})",
                        result.protocol_version,
                        super::job::CLOUD_JOB_PROTOCOL_VERSION
                    )));
                }
                match result.outcome {
                    JobOutcome::Failed => Ok(CloudPoll::Failed(
                        result
                            .failure
                            .map(|failure| failure.to_string())
                            .unwrap_or_else(|| "unknown error".into()),
                    )),
                    JobOutcome::Cancelled => Ok(CloudPoll::Cancelled),
                    JobOutcome::TimedOut => Ok(CloudPoll::TimedOut),
                    JobOutcome::Succeeded => {
                        let blob_key = result.output_blob_key.ok_or_else(|| {
                            CloudError::Artifact("succeeded result missing output_blob_key".into())
                        })?;
                        let manifest = result.output_manifest.ok_or_else(|| {
                            CloudError::Artifact("succeeded result missing output_manifest".into())
                        })?;
                        let content_id = result.content_id.ok_or_else(|| {
                            CloudError::Artifact("succeeded result missing content_id".into())
                        })?;
                        if let Some(expected) = self.expected_content_id
                            && content_id != expected
                        {
                            return Err(CloudError::Artifact(format!(
                                "output identity {content_id} != analytically expected {expected}"
                            )));
                        }
                        let pack = self
                            .store
                            .get(ObjectKey::DispatchBundle(blob_key))
                            .await?
                            .ok_or_else(|| {
                                CloudError::Artifact(format!(
                                    "output bundle {} is missing",
                                    blob_key.to_hex()
                                ))
                            })?;
                        let factory = self
                            .registry
                            .find_erased_stage(&self.stage_name)
                            .ok_or_else(|| {
                                CloudError::Dispatch(format!("unknown stage '{}'", self.stage_name))
                            })?;
                        let stage = factory();
                        let output = unbundle(
                            &*stage,
                            &manifest,
                            &pack,
                            out_dir,
                            Some(content_id),
                            BlobDir::Output,
                        )
                        .map_err(|e| CloudError::Artifact(format!("unbundle output: {e}")))?;
                        Ok(CloudPoll::Succeeded(output))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assignment_observation_cannot_displace_terminal_state() {
        let lifecycle = ExecutionLifecycle::new(ExecutionMode::Cloud);
        lifecycle
            .finish(
                None,
                ExecutionTerminal::Cancelled {
                    reason: "test".into(),
                },
            )
            .unwrap();

        observe_assignment(&lifecycle, &Assignment::new("late-worker", 1)).unwrap();
        let snapshot = lifecycle.snapshot();
        assert!(snapshot.assignment.is_none());
        assert!(matches!(
            snapshot.terminal,
            Some(ExecutionTerminal::Cancelled { .. })
        ));
    }
}
