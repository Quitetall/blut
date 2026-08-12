//! Worker side of cloud dispatch (ADR 0067 · T3.1d) — the object-store analog of
//! `p2p::peer_exec::execute_one`.
//!
//! [`run_one`] claims one job, downloads + `unbundle`s its input (four fail-closed
//! gates), runs the stage under the job's wall-clock deadline, bundles the output,
//! uploads it, and `complete`s the job (success OR failure — a claimed job is never
//! left stranded on its lease). Loop `run_one` to drain a queue.
//!
//! Two gates run before any work, mirroring the P2P peer:
//!   1. `policy.is_dispatchable` — training stages never leave home.
//!   2. `matrix.can_dispatch(data_class, worker_trust)` — the clinical hard-block.
//!      A v1 cloud worker is `Registered`, so the default matrix refuses
//!      `Restricted` (PHI EEG) here even if a job slips into the queue.
//!
//! It also re-validates `job.id` as traversal-safe — defense in depth, so the worker
//! never trusts a foreign queue implementation before joining it into a path.

use std::path::Path;
use std::sync::Arc;

use super::CloudError;
use super::job::{CloudJob, CloudResult, JobOutcome};
use super::queue::{CloudQueue, is_safe_job_id};
use crate::framework::Registry;
use crate::framework::artifact::ContentHash;
use crate::framework::cache::CacheHandle;
use crate::framework::execution::{
    Assignment, ExecutionDeadline, ExecutionFailure, ExecutionFailureKind, ExecutionPhase,
};
use crate::framework::object_store::{ObjectKey, ObjectStore};
use crate::framework::stage::StageContext;
use crate::p2p::bundle::{BlobDir, BundleManifest, bundle, unbundle};
use crate::p2p::dispatch::DispatchPolicy;
use crate::p2p::trust::{DispatchMatrix, TrustLevel};
use tokio_util::sync::CancellationToken;

enum WorkerFailure {
    Failed(Box<ExecutionFailure>),
    Cancelled,
    TimedOut {
        phase: ExecutionPhase,
        deadline_unix_ms: u64,
    },
}

impl From<ExecutionFailure> for WorkerFailure {
    fn from(failure: ExecutionFailure) -> Self {
        Self::Failed(Box::new(failure))
    }
}

struct WorkerExecution<'a> {
    store: &'a ObjectStore,
    registry: &'a Registry,
    policy: &'a dyn DispatchPolicy,
    matrix: &'a DispatchMatrix,
    worker_trust: TrustLevel,
    work_root: &'a Path,
}

/// Claim and run at most one job. Returns `Ok(Some(job_id))` if a job was claimed
/// (whether it succeeded or failed — the result is recorded on the queue), or
/// `Ok(None)` if the queue was idle.
#[allow(clippy::too_many_arguments)]
pub async fn run_one(
    store: &ObjectStore,
    queue: &dyn CloudQueue,
    registry: &Registry,
    policy: &dyn DispatchPolicy,
    matrix: &DispatchMatrix,
    worker_id: &str,
    worker_trust: TrustLevel,
    lease_secs: u64,
    work_root: &Path,
    ledger: Option<&super::cost::CostLedger>,
) -> Result<Option<String>, CloudError> {
    let Some(claimed) = queue.claim(worker_id, lease_secs).await? else {
        return Ok(None);
    };
    let assignment = claimed.assignment;
    let job = claimed.job;
    let job_id = job.id.clone();
    let started = std::time::Instant::now();

    let execution = WorkerExecution {
        store,
        registry,
        policy,
        matrix,
        worker_trust,
        work_root,
    };
    let result = match execute_with_lease(queue, &execution, &job, &assignment).await? {
        Ok((out_blob_key, out_manifest, compute_ms)) => CloudResult {
            protocol_version: super::job::CLOUD_JOB_PROTOCOL_VERSION,
            job_id: job_id.clone(),
            assignment: None,
            outcome: JobOutcome::Succeeded,
            content_id: Some(out_manifest.content_id),
            output_blob_key: Some(out_blob_key),
            output_manifest: Some(out_manifest),
            // Billing signal: COMPUTE wall-clock only (the stage run), not the I/O
            // around it — matches p2p execute_one.
            wall_time_ms: compute_ms,
            failure: None,
            timeout_phase: None,
            deadline_unix_ms: None,
        },
        // A failed job produced no compute; report total occupied time (for
        // observability) but it is NOT billed (see below).
        Err(WorkerFailure::Failed(failure)) => CloudResult::failed(
            job_id.clone(),
            started.elapsed().as_millis() as u64,
            *failure,
        ),
        Err(WorkerFailure::TimedOut {
            phase,
            deadline_unix_ms,
        }) => CloudResult::timed_out(
            job_id.clone(),
            started.elapsed().as_millis() as u64,
            phase,
            deadline_unix_ms,
        ),
        // Cancellation or lease loss is already terminal in the queue. A stale
        // worker must not overwrite that terminal with a late completion.
        Err(WorkerFailure::Cancelled) => return Ok(Some(job_id)),
    };

    // Bill compute-only, and ONLY a successful job — the operator absorbs failed
    // work in v1. Capture the figures BEFORE `result` moves into `complete`, and
    // record AFTER `complete` durably commits, so a failed-then-retried completion
    // can't double-bill.
    let billable = matches!(result.outcome, JobOutcome::Succeeded).then_some(result.wall_time_ms);
    let resources = job.resources;
    if !complete_if_current(queue, &assignment, result).await? {
        return Ok(Some(job_id));
    }
    if let (Some(l), Some(compute_ms)) = (ledger, billable) {
        l.record(&job_id, &resources, compute_ms);
    }
    Ok(Some(job_id))
}

async fn complete_if_current(
    queue: &dyn CloudQueue,
    assignment: &Assignment,
    result: CloudResult,
) -> Result<bool, CloudError> {
    match queue.complete(assignment, result).await {
        Ok(()) => Ok(true),
        // Cancellation or lease expiry may win after execution finishes but before
        // the result is committed. The queue's current owner/terminal state wins.
        Err(CloudError::LeaseLost(_)) => Ok(false),
        Err(error) => Err(error),
    }
}

async fn execute_with_lease(
    queue: &dyn CloudQueue,
    execution: &WorkerExecution<'_>,
    job: &CloudJob,
    assignment: &Assignment,
) -> Result<Result<(ContentHash, BundleManifest, u64), WorkerFailure>, CloudError> {
    if !crate::p2p::trust::custody_allows_off_box(&job.tenant, job.data_class) {
        return Ok(Err(ExecutionFailure::protocol(format!(
            "cloud execution denied by custody policy for tenant '{}' and {:?} data",
            job.tenant, job.data_class
        ))
        .into()));
    }
    let cancel = CancellationToken::new();
    let claimed = execute_claimed(execution, job, &cancel);
    tokio::pin!(claimed);
    loop {
        tokio::select! {
            result = &mut claimed => return Ok(result),
            status = poll_assignment(queue, &job.id) => {
                match status {
                    Ok(super::queue::JobStatus::Running(current)) if &current == assignment => {}
                    Ok(_) => cancel.cancel(),
                    // A transient queue read failure does not revoke a lease. The
                    // fenced completion remains authoritative.
                    Err(()) => {}
                }
            }
        }
    }
}

async fn poll_assignment(
    queue: &dyn CloudQueue,
    job_id: &str,
) -> Result<super::queue::JobStatus, ()> {
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);
    const POLL_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);

    tokio::time::sleep(POLL_INTERVAL).await;
    tokio::time::timeout(POLL_TIMEOUT, queue.status(job_id))
        .await
        .map_err(|_| ())?
        .map_err(|_| ())
}

/// Gate the job, create its work dir, run it, then clean the dir up. Returns the
/// output blob key + manifest + the COMPUTE-only wall time (ms).
async fn execute_claimed(
    execution: &WorkerExecution<'_>,
    job: &CloudJob,
    cancel: &CancellationToken,
) -> Result<(ContentHash, BundleManifest, u64), WorkerFailure> {
    if job.protocol_version != super::job::CLOUD_JOB_PROTOCOL_VERSION {
        return Err(ExecutionFailure::protocol(format!(
            "cloud job protocol v{} unsupported (want v{})",
            job.protocol_version,
            super::job::CLOUD_JOB_PROTOCOL_VERSION
        ))
        .into());
    }
    // Defense in depth: the worker re-validates the id before joining it into a
    // path, rather than trust the queue's enqueue-time check.
    if !is_safe_job_id(&job.id) {
        return Err(ExecutionFailure::protocol(format!("unsafe job id '{}'", job.id)).into());
    }
    let actual_args_hash = ContentHash::of_bytes(&CacheHandle::canonical_json_bytes(&job.args));
    if actual_args_hash != job.args_hash {
        return Err(ExecutionFailure::protocol(format!(
            "cloud job args hash {} does not match received args {}",
            job.args_hash.to_hex(),
            actual_args_hash.to_hex()
        ))
        .into());
    }
    // Gate 1: dispatchable stage (training never leaves home).
    if !execution.policy.is_dispatchable(&job.stage_name) {
        return Err(ExecutionFailure::protocol(format!(
            "stage '{}' is not dispatchable",
            job.stage_name
        ))
        .into());
    }
    // Gate 2: data-class vs worker trust — the clinical hard-block.
    if !execution
        .matrix
        .can_dispatch(job.data_class, execution.worker_trust)
    {
        return Err(ExecutionFailure::protocol(format!(
            "data_class {:?} not permitted on a {:?} cloud worker",
            job.data_class, execution.worker_trust
        ))
        .into());
    }

    let factory = execution
        .registry
        .find_erased_stage(&job.stage_name)
        .ok_or_else(|| {
            WorkerFailure::from(ExecutionFailure::protocol(format!(
                "unknown stage '{}'",
                job.stage_name
            )))
        })?;
    let stage = factory();
    if stage.schema() != job.stage_schema {
        return Err(ExecutionFailure::protocol(format!(
            "stage '{}' schema {} does not match cloud job {}",
            job.stage_name,
            stage.schema(),
            job.stage_schema
        ))
        .into());
    }

    let stage_dir = execution.work_root.join(&job.id);
    std::fs::create_dir_all(&stage_dir).map_err(|e| {
        WorkerFailure::from(ExecutionFailure::new(
            ExecutionFailureKind::Storage,
            "EXECUTION_STAGE_DIR",
            format!("create worker stage_dir: {e}"),
        ))
    })?;

    // Do the fs-touching work, then clean the dir up regardless of outcome — the
    // output bytes now live in the object store, so the local dir is disposable.
    let out = run_in_dir(execution.store, &*stage, &stage_dir, job, cancel).await;
    let _ = std::fs::remove_dir_all(&stage_dir);
    out
}

/// The fs choreography inside an already-created `stage_dir`: download → unbundle →
/// run (timed) → bundle → upload.
async fn bounded<T, F>(
    future: F,
    cancel: &CancellationToken,
    deadline: ExecutionDeadline,
    phase: ExecutionPhase,
) -> Result<T, WorkerFailure>
where
    F: std::future::Future<Output = Result<T, ExecutionFailure>>,
{
    tokio::pin!(future);
    tokio::select! {
        result = &mut future => result.map_err(WorkerFailure::from),
        _ = cancel.cancelled() => Err(WorkerFailure::Cancelled),
        _ = sleep_optional(deadline.soft_remaining()) => Err(WorkerFailure::TimedOut {
            phase,
            deadline_unix_ms: deadline.soft_unix_ms.unwrap_or(deadline.hard_unix_ms),
        }),
        _ = tokio::time::sleep(deadline.hard_remaining()) => Err(WorkerFailure::TimedOut {
            phase,
            deadline_unix_ms: deadline.hard_unix_ms,
        }),
    }
}

async fn sleep_optional(duration: Option<std::time::Duration>) {
    match duration {
        Some(duration) => tokio::time::sleep(duration).await,
        None => std::future::pending::<()>().await,
    }
}

async fn run_in_dir(
    store: &ObjectStore,
    stage: &dyn crate::framework::stage::StageDyn,
    stage_dir: &Path,
    job: &CloudJob,
    cancel: &CancellationToken,
) -> Result<(ContentHash, BundleManifest, u64), WorkerFailure> {
    let cache = Arc::new(CacheHandle::job_local(stage_dir.join(".cache")));
    let input_content_id = job.input_manifest.content_id;
    let mut ctx = StageContext::for_peer(
        stage_dir.to_path_buf(),
        stage_dir.to_path_buf(),
        cache,
        job.invocation_key,
    );
    ctx.cancel = cancel.clone();

    // Download the input pack + unbundle (the four fail-closed gates run here).
    let pack = bounded(
        async {
            store
                .get(ObjectKey::DispatchBundle(job.input_blob_key))
                .await
                .map_err(|error| {
                    ExecutionFailure::new(
                        ExecutionFailureKind::Storage,
                        "EXECUTION_STORAGE",
                        format!("download input bundle: {error}"),
                    )
                })?
                .ok_or_else(|| {
                    ExecutionFailure::artifact(format!(
                        "input bundle {} is missing",
                        job.input_blob_key.to_hex()
                    ))
                })
        },
        cancel,
        job.deadline,
        ExecutionPhase::Running,
    )
    .await?;
    let input = unbundle(
        stage,
        &job.input_manifest,
        &pack,
        stage_dir,
        Some(input_content_id),
        BlobDir::Input,
    )
    .map_err(|e| WorkerFailure::from(ExecutionFailure::artifact(format!("unbundle input: {e}"))))?;

    // Run under the original end-to-end deadline; time ONLY the compute.
    let started = std::time::Instant::now();
    let output = bounded(
        async {
            stage
                .run_erased(&ctx, input, job.args.clone())
                .await
                .map_err(|error| ExecutionFailure::from_stage_error(&job.stage_name, &error))
        },
        cancel,
        job.deadline,
        ExecutionPhase::Running,
    )
    .await?;
    let compute_ms = started.elapsed().as_millis() as u64;

    // Bundle the output bound to the job's expected_output_hash, then upload it.
    let (out_manifest, out_pack) = bundle(
        stage,
        output,
        stage_dir,
        BlobDir::Output,
        job.expected_content_id,
    )
    .map_err(|e| WorkerFailure::from(ExecutionFailure::artifact(format!("bundle output: {e}"))))?;
    let out_blob_key = ContentHash::of_bytes(&out_pack);
    bounded(
        async {
            store
                .put(ObjectKey::DispatchBundle(out_blob_key), out_pack)
                .await
                .map_err(|error| {
                    ExecutionFailure::new(
                        ExecutionFailureKind::Storage,
                        "EXECUTION_STORAGE",
                        format!("upload output bundle: {error}"),
                    )
                })
        },
        cancel,
        job.deadline,
        ExecutionPhase::UploadingOutput,
    )
    .await?;

    Ok((out_blob_key, out_manifest, compute_ms))
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::cloud::queue::{ClaimedJob, JobStatus};

    struct FailingQueue;

    #[async_trait]
    impl CloudQueue for FailingQueue {
        async fn enqueue(&self, _job: CloudJob) -> Result<(), CloudError> {
            unreachable!()
        }

        async fn claim(
            &self,
            _worker_id: &str,
            _lease_secs: u64,
        ) -> Result<Option<ClaimedJob>, CloudError> {
            unreachable!()
        }

        async fn complete(
            &self,
            _assignment: &Assignment,
            _result: CloudResult,
        ) -> Result<(), CloudError> {
            Err(CloudError::LeaseLost("job".into()))
        }

        async fn cancel(&self, _job_id: &str, _reason: &str) -> Result<bool, CloudError> {
            unreachable!()
        }

        async fn status(&self, _job_id: &str) -> Result<JobStatus, CloudError> {
            Err(CloudError::Dispatch("transient queue read".into()))
        }

        async fn reclaim_expired(&self) -> Result<usize, CloudError> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn transient_status_failure_does_not_revoke_assignment() {
        assert!(poll_assignment(&FailingQueue, "job").await.is_err());
    }

    #[tokio::test]
    async fn stale_completion_is_an_expected_fenced_race() {
        let assignment = Assignment::new("worker", 1);
        let result = CloudResult::cancelled("job", "test");
        assert!(
            !complete_if_current(&FailingQueue, &assignment, result)
                .await
                .unwrap()
        );
    }
}
