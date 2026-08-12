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

use serde_json::Value;

use super::CloudError;
use super::job::{CloudJob, JobOutcome};
use super::queue::{CloudQueue, JobStatus};
use crate::framework::Registry;
use crate::framework::artifact::{ContentHash, ContentId, InvocationKey};
use crate::framework::object_store::{ObjectKey, ObjectStore};
use crate::framework::stage::ErasedArtifact;
use crate::p2p::bundle::{BlobDir, bundle, unbundle};
use crate::p2p::task::ResourceRequest;
use crate::p2p::trust::DataClass;

/// Everything `submit` needs to dispatch one stage. Mirrors `dispatch_to_peer`'s
/// argument list; `expected_output_hash` is caller-supplied (analytic for a
/// deterministic stage, or a cache-derived address for verified re-execution).
pub struct CloudSubmitSpec {
    pub job_id: String,
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

        let job = CloudJob {
            protocol_version: super::job::CLOUD_JOB_PROTOCOL_VERSION,
            id: spec.job_id.clone(),
            stage_name: spec.stage_name.clone(),
            stage_schema: stage.schema(),
            invocation_key: spec.invocation_key,
            args: spec.args,
            input_blob_key: blob_key,
            input_manifest: manifest,
            expected_content_id: spec.expected_content_id,
            resources: spec.resources,
            data_class: spec.data_class,
            priority: spec.priority,
            timeout_secs: spec.timeout_secs,
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
            JobStatus::Queued | JobStatus::Running => Ok(CloudPoll::Pending),
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
                        result.error.unwrap_or_else(|| "unknown error".into()),
                    )),
                    JobOutcome::Cancelled => Ok(CloudPoll::Cancelled),
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
