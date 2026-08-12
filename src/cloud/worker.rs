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

use bytes::Bytes;

use super::CloudError;
use super::job::{CloudJob, CloudResult, JobOutcome};
use super::queue::{CloudQueue, is_safe_job_id};
use super::store::BlobStore;
use crate::framework::Registry;
use crate::framework::artifact::ContentHash;
use crate::framework::cache::CacheHandle;
use crate::framework::stage::StageContext;
use crate::p2p::bundle::{BlobDir, BundleManifest, bundle, unbundle};
use crate::p2p::dispatch::DispatchPolicy;
use crate::p2p::trust::{DispatchMatrix, TrustLevel};

/// Claim and run at most one job. Returns `Ok(Some(job_id))` if a job was claimed
/// (whether it succeeded or failed — the result is recorded on the queue), or
/// `Ok(None)` if the queue was idle.
#[allow(clippy::too_many_arguments)]
pub async fn run_one(
    store: &dyn BlobStore,
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
    let Some(job) = queue.claim(worker_id, lease_secs).await? else {
        return Ok(None);
    };
    let job_id = job.id.clone();
    let started = std::time::Instant::now();

    let result = match execute_claimed(
        store,
        registry,
        policy,
        matrix,
        worker_trust,
        work_root,
        &job,
    )
    .await
    {
        Ok((out_blob_key, out_manifest, compute_ms)) => CloudResult {
            protocol_version: super::job::CLOUD_JOB_PROTOCOL_VERSION,
            job_id: job_id.clone(),
            outcome: JobOutcome::Succeeded,
            content_id: Some(out_manifest.content_id),
            output_blob_key: Some(out_blob_key),
            output_manifest: Some(out_manifest),
            // Billing signal: COMPUTE wall-clock only (the stage run), not the I/O
            // around it — matches p2p execute_one.
            wall_time_ms: compute_ms,
            error: None,
        },
        // A failed job produced no compute; report total occupied time (for
        // observability) but it is NOT billed (see below).
        Err(e) => CloudResult::failed(
            job_id.clone(),
            started.elapsed().as_millis() as u64,
            e.to_string(),
        ),
    };

    // Bill compute-only, and ONLY a successful job — the operator absorbs failed
    // work in v1. Capture the figures BEFORE `result` moves into `complete`, and
    // record AFTER `complete` durably commits, so a failed-then-retried completion
    // can't double-bill.
    let billable = matches!(result.outcome, JobOutcome::Succeeded).then_some(result.wall_time_ms);
    let resources = job.resources;
    queue.complete(worker_id, result).await?;
    if let (Some(l), Some(compute_ms)) = (ledger, billable) {
        l.record(&job_id, &resources, compute_ms);
    }
    Ok(Some(job_id))
}

/// Gate the job, create its work dir, run it, then clean the dir up. Returns the
/// output blob key + manifest + the COMPUTE-only wall time (ms).
async fn execute_claimed(
    store: &dyn BlobStore,
    registry: &Registry,
    policy: &dyn DispatchPolicy,
    matrix: &DispatchMatrix,
    worker_trust: TrustLevel,
    work_root: &Path,
    job: &CloudJob,
) -> Result<(ContentHash, BundleManifest, u64), CloudError> {
    if job.protocol_version != super::job::CLOUD_JOB_PROTOCOL_VERSION {
        return Err(CloudError::Dispatch(format!(
            "cloud job protocol v{} unsupported (want v{})",
            job.protocol_version,
            super::job::CLOUD_JOB_PROTOCOL_VERSION
        )));
    }
    // Defense in depth: the worker re-validates the id before joining it into a
    // path, rather than trust the queue's enqueue-time check.
    if !is_safe_job_id(&job.id) {
        return Err(CloudError::Dispatch(format!("unsafe job id '{}'", job.id)));
    }
    // Gate 1: dispatchable stage (training never leaves home).
    if !policy.is_dispatchable(&job.stage_name) {
        return Err(CloudError::Dispatch(format!(
            "stage '{}' is not dispatchable",
            job.stage_name
        )));
    }
    // Gate 2: data-class vs worker trust — the clinical hard-block.
    if !matrix.can_dispatch(job.data_class, worker_trust) {
        return Err(CloudError::Dispatch(format!(
            "data_class {:?} not permitted on a {:?} cloud worker",
            job.data_class, worker_trust
        )));
    }

    let factory = registry
        .find_erased_stage(&job.stage_name)
        .ok_or_else(|| CloudError::Dispatch(format!("unknown stage '{}'", job.stage_name)))?;
    let stage = factory();

    let stage_dir = work_root.join(&job.id);
    std::fs::create_dir_all(&stage_dir)
        .map_err(|e| CloudError::Store(format!("create worker stage_dir: {e}")))?;

    // Do the fs-touching work, then clean the dir up regardless of outcome — the
    // output bytes now live in the object store, so the local dir is disposable.
    let out = run_in_dir(store, &*stage, &stage_dir, job).await;
    let _ = std::fs::remove_dir_all(&stage_dir);
    out
}

/// The fs choreography inside an already-created `stage_dir`: download → unbundle →
/// run (timed) → bundle → upload.
async fn run_in_dir(
    store: &dyn BlobStore,
    stage: &dyn crate::framework::stage::StageDyn,
    stage_dir: &Path,
    job: &CloudJob,
) -> Result<(ContentHash, BundleManifest, u64), CloudError> {
    let cache = Arc::new(CacheHandle::job_local(stage_dir.join(".cache")));
    let input_content_id = job.input_manifest.content_id;
    let ctx = StageContext::for_peer(
        stage_dir.to_path_buf(),
        stage_dir.to_path_buf(),
        cache,
        job.invocation_key,
    );

    // Download the input pack + unbundle (the four fail-closed gates run here).
    let pack = store.get_blob(&job.input_blob_key).await?;
    let input = unbundle(
        stage,
        &job.input_manifest,
        &pack,
        stage_dir,
        Some(input_content_id),
        BlobDir::Input,
    )
    .map_err(|e| CloudError::Dispatch(format!("unbundle input: {e}")))?;

    // Run under the job's wall-clock deadline; time ONLY the compute.
    let started = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(job.timeout_secs.max(1));
    let output = tokio::time::timeout(timeout, stage.run_erased(&ctx, input, job.args.clone()))
        .await
        .map_err(|_| {
            CloudError::Dispatch(format!(
                "stage '{}' exceeded timeout_secs {}",
                job.stage_name, job.timeout_secs
            ))
        })?
        .map_err(|e| CloudError::Dispatch(format!("stage run failed: {e}")))?;
    let compute_ms = started.elapsed().as_millis() as u64;

    // Bundle the output bound to the job's expected_output_hash, then upload it.
    let (out_manifest, out_pack) = bundle(
        stage,
        output,
        stage_dir,
        BlobDir::Output,
        job.expected_content_id,
    )
    .map_err(|e| CloudError::Dispatch(format!("bundle output: {e}")))?;
    let out_blob_key = ContentHash::of_bytes(&out_pack);
    store.put_blob(&out_blob_key, Bytes::from(out_pack)).await?;

    Ok((out_blob_key, out_manifest, compute_ms))
}
