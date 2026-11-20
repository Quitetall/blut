//! Worker side of cloud dispatch (ADR 0067 · T3.1d) — the object-store analog of
//! `p2p::peer_exec::execute_one`.
//!
//! [`run_one`] claims one job, downloads + `unbundle`s its input (four fail-closed
//! gates), runs the stage under the job's wall-clock deadline, bundles the output,
//! uploads it, and `complete`s the job (success OR failure — a claimed job is never
//! left stranded on its lease). [`serve`] is the long-running form `blut cloud
//! worker` uses: it adds lease renewal, cancellation by re-reading the job,
//! release on shutdown and backoff.
//!
//! Three gates run before any work, mirroring the P2P peer. Stated in the order
//! they actually fire, because the first one shadows the others for PHI:
//!   0. `custody_allows_off_box(tenant, data_class)` — the clinical hard-block.
//!      This is what refuses `Restricted` (PHI EEG) at the worker, and it is
//!      UNCONDITIONAL: no trust level and no operator-set policy reaches it.
//!      The submitter runs the same check, so this is the second, independent
//!      line — it holds even when a job reaches the queue by some path that
//!      skipped the submitter (an older client, a bug, a bypass).
//!   1. `policy.is_dispatchable` — training stages never leave home.
//!   2. `matrix.can_dispatch(data_class, worker_trust)` — trust-vs-class policy.
//!      Its `Restricted` branch is unreachable here (gate 0 already returned)
//!      and `DispatchMatrix::can_dispatch` independently returns false for
//!      `Restricted` before indexing the trust row. Both are deliberate
//!      redundancy: do not "simplify" either away on the grounds that the other
//!      covers it. `restricted_job_is_refused_by_a_registered_cloud_worker`
//!      (tests/cloud_integration.rs) pins the behaviour.
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
        // A stage that watches `ctx.cancel` ends with an error of its own
        // when it is stopped. That is the cancellation, not a stage failure:
        // recording it as one would fail a stage a shutting-down worker is
        // handing back.
        result = &mut future => result.map_err(|failure| {
            if cancel.is_cancelled() {
                WorkerFailure::Cancelled
            } else {
                WorkerFailure::from(failure)
            }
        }),
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
    // The stage runs under the job's custody namespace, as it would locally.
    ctx.tenant = job.tenant.clone();

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
        // Contained like a local stage: a process the stage places in a
        // systemd unit (`LocalSystemd::spawn_contained`) is recorded under
        // this stage dir before it starts, exactly as the executor records it
        // under a job dir.
        crate::jobs::in_job(stage_dir.to_path_buf(), async {
            stage
                .run_erased(&ctx, input, job.args.clone())
                .await
                .map_err(|error| ExecutionFailure::from_stage_error(&job.stage_name, &error))
        }),
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

/// How a [`serve`] loop behaves.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ServeOptions {
    /// Lease owner name. Unique per worker process (default `<host>-<pid>`).
    pub worker_id: String,
    /// Trust level the dispatch matrix checks jobs against.
    pub trust: TrustLevel,
    /// Lease length. The holder renews at a third of it, so a worker that
    /// dies is replaced about this long after its last renewal.
    pub lease_secs: u64,
    /// Idle polling backs off from `poll_min` to `poll_max` while the queue
    /// is empty, and drops back to `poll_min` after each job.
    pub poll_min: std::time::Duration,
    pub poll_max: std::time::Duration,
    /// How often a running job's state is re-read to notice a cancellation.
    pub status_interval: std::time::Duration,
    /// Stop after this many claimed jobs.
    pub max_jobs: Option<u64>,
    /// Stop after the queue has been empty this long.
    pub idle_exit: Option<std::time::Duration>,
    /// Scratch root; each job gets `<work_root>/<job id>`, removed afterwards.
    pub work_root: std::path::PathBuf,
}

impl ServeOptions {
    pub fn new(worker_id: impl Into<String>, work_root: impl Into<std::path::PathBuf>) -> Self {
        Self {
            worker_id: worker_id.into(),
            trust: TrustLevel::Registered,
            lease_secs: 60,
            poll_min: std::time::Duration::from_millis(500),
            poll_max: std::time::Duration::from_secs(5),
            status_interval: std::time::Duration::from_secs(2),
            max_jobs: None,
            idle_exit: None,
            work_root: work_root.into(),
        }
    }

    pub fn with_lease_secs(mut self, lease_secs: u64) -> Self {
        self.lease_secs = lease_secs.max(1);
        self
    }

    pub fn with_poll(mut self, min: std::time::Duration, max: std::time::Duration) -> Self {
        self.poll_min = min;
        self.poll_max = max.max(min);
        self
    }

    pub fn with_status_interval(mut self, interval: std::time::Duration) -> Self {
        self.status_interval = interval;
        self
    }

    pub fn with_max_jobs(mut self, max_jobs: Option<u64>) -> Self {
        self.max_jobs = max_jobs;
        self
    }

    pub fn with_idle_exit(mut self, idle_exit: Option<std::time::Duration>) -> Self {
        self.idle_exit = idle_exit;
        self
    }

    pub fn with_trust(mut self, trust: TrustLevel) -> Self {
        self.trust = trust;
        self
    }
}

/// What a [`serve`] loop did, reported as it happens.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ServeEvent {
    /// A job was claimed and is starting.
    Claimed {
        job_id: String,
        stage_name: String,
        assignment: Assignment,
    },
    /// The job's result was recorded on the queue.
    Finished {
        job_id: String,
        outcome: JobOutcome,
        wall_time_ms: u64,
        detail: Option<String>,
    },
    /// The job stopped because the queue no longer assigns it to this worker
    /// (cancelled, or the lease was lost); nothing was recorded.
    Abandoned { job_id: String },
    /// Shutdown: the job was handed back unfinished for another worker.
    Released { job_id: String },
    /// A queue call failed; the loop backs off and carries on.
    QueueError { message: String },
}

/// Totals for a [`serve`] loop.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ServeReport {
    pub claimed: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub abandoned: u64,
    pub released: u64,
}

/// Claim and run jobs until `shutdown` fires, `max_jobs` is reached, or the
/// queue has been idle for `idle_exit`.
///
/// While a job runs, its lease is renewed every third of `lease_secs` and its
/// state re-read every `status_interval`: when the job is cancelled or the
/// lease goes to another worker, the stage is cancelled and nothing is
/// recorded. On `shutdown` the running stage is cancelled and the job is
/// released, so another worker picks it up at once. A failing queue call is
/// reported and retried after a backoff; it does not end the loop.
#[allow(clippy::too_many_arguments)]
pub async fn serve(
    store: &ObjectStore,
    queue: &dyn CloudQueue,
    registry: &Registry,
    policy: &dyn DispatchPolicy,
    matrix: &DispatchMatrix,
    options: &ServeOptions,
    shutdown: &CancellationToken,
    ledger: Option<&super::cost::CostLedger>,
    on_event: &(dyn Fn(&ServeEvent) + Send + Sync),
) -> ServeReport {
    let mut report = ServeReport::default();
    let mut backoff = options.poll_min;
    let mut idle_since = std::time::Instant::now();
    let execution = WorkerExecution {
        store,
        registry,
        policy,
        matrix,
        worker_trust: options.trust,
        work_root: &options.work_root,
    };
    loop {
        if shutdown.is_cancelled() || options.max_jobs.is_some_and(|max| report.claimed >= max) {
            return report;
        }
        let claimed = tokio::select! {
            claimed = queue.claim(&options.worker_id, options.lease_secs) => claimed,
            _ = shutdown.cancelled() => return report,
        };
        let claimed = match claimed {
            Ok(Some(claimed)) => claimed,
            Ok(None) => {
                if options
                    .idle_exit
                    .is_some_and(|idle| idle_since.elapsed() >= idle)
                {
                    return report;
                }
                tokio::select! {
                    _ = tokio::time::sleep(backoff) => {}
                    _ = shutdown.cancelled() => return report,
                }
                backoff = (backoff * 2).min(options.poll_max);
                continue;
            }
            Err(error) => {
                on_event(&ServeEvent::QueueError {
                    message: error.to_string(),
                });
                tokio::select! {
                    _ = tokio::time::sleep(options.poll_max) => {}
                    _ = shutdown.cancelled() => return report,
                }
                continue;
            }
        };
        report.claimed += 1;
        backoff = options.poll_min;
        let job = claimed.job;
        let assignment = claimed.assignment;
        on_event(&ServeEvent::Claimed {
            job_id: job.id.clone(),
            stage_name: job.stage_name.clone(),
            assignment: assignment.clone(),
        });
        let started = std::time::Instant::now();
        let stage_cancel = CancellationToken::new();
        let mut released_for_shutdown = false;
        let outcome = {
            let custody = crate::p2p::trust::custody_allows_off_box(&job.tenant, job.data_class);
            let run = async {
                if !custody {
                    return Err(WorkerFailure::from(ExecutionFailure::protocol(format!(
                        "cloud execution denied by custody policy for tenant '{}' and {:?} data",
                        job.tenant, job.data_class
                    ))));
                }
                execute_claimed(&execution, &job, &stage_cancel).await
            };
            tokio::pin!(run);
            let period = std::time::Duration::from_millis(
                (options.lease_secs.saturating_mul(1000) / 3).max(200),
            );
            let mut renew = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
            let mut status = tokio::time::interval_at(
                tokio::time::Instant::now() + options.status_interval,
                options
                    .status_interval
                    .max(std::time::Duration::from_millis(10)),
            );
            loop {
                tokio::select! {
                    result = &mut run => break result,
                    _ = renew.tick() => {
                        match queue.renew(&job.id, &assignment, options.lease_secs).await {
                            Ok(true) => {}
                            Ok(false) => stage_cancel.cancel(),
                            Err(error) => on_event(&ServeEvent::QueueError {
                                message: format!("renew lease of {}: {error}", job.id),
                            }),
                        }
                    }
                    _ = status.tick() => {
                        match tokio::time::timeout(options.status_interval.max(std::time::Duration::from_secs(5)), queue.status(&job.id)).await {
                            Ok(Ok(super::queue::JobStatus::Running(current))) if current == assignment => {}
                            // An expired lease reads as queued; it is still
                            // ours until someone else claims it, and renewal
                            // decides that. Anything terminal or re-leased is not.
                            Ok(Ok(super::queue::JobStatus::Queued)) => {}
                            Ok(Ok(_)) => stage_cancel.cancel(),
                            // A failed read does not revoke a lease; the
                            // fenced completion stays authoritative.
                            Ok(Err(_)) | Err(_) => {}
                        }
                    }
                    _ = shutdown.cancelled(), if !released_for_shutdown => {
                        released_for_shutdown = true;
                        stage_cancel.cancel();
                    }
                }
            }
        };
        let result = match outcome {
            Ok((out_blob_key, out_manifest, compute_ms)) => Some(CloudResult {
                protocol_version: super::job::CLOUD_JOB_PROTOCOL_VERSION,
                job_id: job.id.clone(),
                assignment: None,
                outcome: JobOutcome::Succeeded,
                content_id: Some(out_manifest.content_id),
                output_blob_key: Some(out_blob_key),
                output_manifest: Some(out_manifest),
                wall_time_ms: compute_ms,
                failure: None,
                timeout_phase: None,
                deadline_unix_ms: None,
            }),
            Err(WorkerFailure::Failed(failure)) => Some(CloudResult::failed(
                job.id.clone(),
                started.elapsed().as_millis() as u64,
                *failure,
            )),
            Err(WorkerFailure::TimedOut {
                phase,
                deadline_unix_ms,
            }) => Some(CloudResult::timed_out(
                job.id.clone(),
                started.elapsed().as_millis() as u64,
                phase,
                deadline_unix_ms,
            )),
            Err(WorkerFailure::Cancelled) => None,
        };
        match result {
            Some(result) => {
                let finished = ServeEvent::Finished {
                    job_id: job.id.clone(),
                    outcome: result.outcome,
                    wall_time_ms: result.wall_time_ms,
                    detail: result.failure.as_ref().map(ToString::to_string),
                };
                let billable =
                    matches!(result.outcome, JobOutcome::Succeeded).then_some(result.wall_time_ms);
                match complete_if_current(queue, &assignment, result).await {
                    Ok(true) => {
                        if billable.is_some() {
                            report.succeeded += 1;
                        } else {
                            report.failed += 1;
                        }
                        if let (Some(ledger), Some(compute_ms)) = (ledger, billable) {
                            ledger.record(&job.id, &job.resources, compute_ms);
                        }
                        on_event(&finished);
                    }
                    Ok(false) => {
                        report.abandoned += 1;
                        on_event(&ServeEvent::Abandoned {
                            job_id: job.id.clone(),
                        });
                    }
                    Err(error) => {
                        report.abandoned += 1;
                        on_event(&ServeEvent::QueueError {
                            message: format!("record the result of {}: {error}", job.id),
                        });
                    }
                }
            }
            None if released_for_shutdown => {
                match queue
                    .release(&job.id, &assignment, "the worker was shut down")
                    .await
                {
                    Ok(true) => {
                        report.released += 1;
                        on_event(&ServeEvent::Released {
                            job_id: job.id.clone(),
                        });
                    }
                    Ok(false) => {
                        report.abandoned += 1;
                        on_event(&ServeEvent::Abandoned {
                            job_id: job.id.clone(),
                        });
                    }
                    Err(error) => {
                        report.abandoned += 1;
                        on_event(&ServeEvent::QueueError {
                            message: format!(
                                "release {}: {error}; it is requeued when its lease runs out",
                                job.id
                            ),
                        });
                    }
                }
            }
            None => {
                report.abandoned += 1;
                on_event(&ServeEvent::Abandoned {
                    job_id: job.id.clone(),
                });
            }
        }
        if released_for_shutdown {
            return report;
        }
        idle_since = std::time::Instant::now();
    }
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
