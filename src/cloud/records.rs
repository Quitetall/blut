// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! What a `--launcher cloud` run leaves in its job directory, so the work it
//! submitted can be found again after the submitting process is gone.
//!
//! * `cloud_queue`: the queue location ([`CloudLocation::spec`]): an
//!   absolute directory or a URL, never a credential.
//! * `cloud_jobs`: one cloud job id per line, each written before that job is
//!   enqueued ([`CloudSubmitter::with_job_record`](super::submitter::CloudSubmitter::with_job_record)).
//!
//! `blut cancel` cancels every recorded job that is still queued or running,
//! and `plan resume` refuses while one is, as they do for a job's Slurm jobs
//! and systemd units. Reaching an `s3://` queue needs the same `AWS_*`
//! variables the run had.

use std::path::Path;

use super::CloudError;
use super::location::CloudLocation;
use super::object_queue::ObjectQueueOptions;
use super::queue::{CloudQueue, JobStatus};

/// File in a job directory naming the cloud queue its stages were sent to.
pub const QUEUE_RECORD_FILE: &str = "cloud_queue";
/// File in a job directory listing its cloud job ids, one per line.
pub const JOBS_RECORD_FILE: &str = "cloud_jobs";

/// Record the queue location for the job in `job_dir`.
pub fn write_queue_record(job_dir: &Path, location: &CloudLocation) -> std::io::Result<()> {
    std::fs::write(
        job_dir.join(QUEUE_RECORD_FILE),
        format!("{}\n", location.spec()),
    )
}

/// The queue location recorded for the job in `job_dir`, if any.
pub fn recorded_queue(job_dir: &Path) -> Option<String> {
    std::fs::read_to_string(job_dir.join(QUEUE_RECORD_FILE))
        .ok()
        .map(|body| body.trim().to_string())
        .filter(|spec| !spec.is_empty())
}

/// The cloud job ids recorded for the job in `job_dir`.
pub fn recorded_jobs(job_dir: &Path) -> Vec<String> {
    std::fs::read_to_string(job_dir.join(JOBS_RECORD_FILE))
        .map(|body| {
            body.lines()
                .map(str::trim)
                .filter(|line| super::queue::is_safe_job_id(line))
                .map(String::from)
                .collect()
        })
        .unwrap_or_default()
}

/// Open the recorded queue, if the job has one and recorded any job.
fn open_recorded(job_dir: &Path) -> Result<Option<(CloudLocation, Vec<String>)>, CloudError> {
    let jobs = recorded_jobs(job_dir);
    if jobs.is_empty() {
        return Ok(None);
    }
    let Some(spec) = recorded_queue(job_dir) else {
        return Err(CloudError::Dispatch(format!(
            "{} lists cloud jobs but {} names no queue",
            job_dir.join(JOBS_RECORD_FILE).display(),
            QUEUE_RECORD_FILE
        )));
    };
    let options = ObjectQueueOptions::default().with_status_refresh(std::time::Duration::ZERO);
    Ok(Some((CloudLocation::open(&spec, options)?, jobs)))
}

/// Recorded cloud jobs of `job_dir` that are still queued or running, as
/// `(id, "queued" | "running on <worker>")`.
pub async fn live_recorded_jobs(job_dir: &Path) -> Result<Vec<(String, String)>, CloudError> {
    let Some((location, jobs)) = open_recorded(job_dir)? else {
        return Ok(Vec::new());
    };
    let mut live = Vec::new();
    for id in jobs {
        match location.queue().status(&id).await? {
            JobStatus::Queued => live.push((id, "queued".to_string())),
            JobStatus::Running(assignment) => {
                live.push((id, format!("running on {}", assignment.owner)));
            }
            JobStatus::Done(_) | JobStatus::Unknown => {}
        }
    }
    Ok(live)
}

/// Cancel every recorded cloud job of `job_dir` that is still queued or
/// running. Returns the ids cancelled. A worker running one notices at its
/// next status read and stops the stage.
pub async fn cancel_recorded_jobs(job_dir: &Path, reason: &str) -> Result<Vec<String>, CloudError> {
    let Some((location, jobs)) = open_recorded(job_dir)? else {
        return Ok(Vec::new());
    };
    let mut cancelled = Vec::new();
    for id in jobs {
        if location.queue().cancel(&id, reason).await? {
            cancelled.push(id);
        }
    }
    Ok(cancelled)
}

/// Why `plan resume` must not start the job in `job_dir` yet: a cloud job it
/// submitted is still queued or running, or the queue cannot be asked.
pub async fn resume_blocker(job_id: &str, job_dir: &Path) -> Option<String> {
    match live_recorded_jobs(job_dir).await {
        Ok(live) if live.is_empty() => None,
        Ok(live) => Some(format!(
            "job {job_id} still has cloud job(s) that its stages submitted: {}. Stop them \
             with `blut cancel {job_id}`, or wait for them to finish, then resume",
            live.iter()
                .map(|(id, state)| format!("{id} ({state})"))
                .collect::<Vec<_>>()
                .join(", ")
        )),
        Err(error) => Some(format!(
            "job {job_id} submitted stages to a cloud queue, and whether they still run \
             cannot be checked: {error}. Resume with the queue reachable (for an s3:// \
             queue, the AWS_* variables the run used)"
        )),
    }
}
