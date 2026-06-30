//! Cloud job queue (ADR 0067 · T3.1b) — a leased, priority queue.
//!
//! The lease (visibility timeout + requeue) is the durability property
//! `blut-worker`'s file queue lacks: a worker that claims a job and crashes does
//! not strand it — `reclaim_expired` returns the job to the pending set once its
//! lease elapses, so another worker picks it up.
//!
//! [`MemQueue`] is the in-process implementation used by the loopback proof and
//! single-box runs. A real cross-process deployment (submitter local, worker on a
//! cloud GPU box) swaps in an object-store / REST-backed `CloudQueue` with the same
//! trait — the dispatch path above it is unchanged.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex;

use super::job::{CloudJob, CloudResult};
use super::CloudError;

/// What the submitter's handle sees when it polls a job.
#[derive(Clone, Debug)]
pub enum JobStatus {
    /// Enqueued, not yet claimed.
    Queued,
    /// Leased to a worker and (presumably) executing.
    Running,
    /// Finished — carries the worker's result (success or failure).
    Done(CloudResult),
    /// No such job id (never enqueued, or evicted).
    Unknown,
}

/// A leased, priority job queue. All methods are `async` for parity with a
/// network-backed implementation; the in-process one never actually awaits.
#[async_trait]
pub trait CloudQueue: Send + Sync {
    /// Enqueue a job (pending, claimable). Rejects an unsafe id (`BadJobId`) or a
    /// duplicate of one already pending/leased/done (`DuplicateJob`).
    async fn enqueue(&self, job: CloudJob) -> Result<(), CloudError>;
    /// Claim the highest-priority pending job for `worker_id`, leasing it for
    /// `lease_secs`. Returns `None` if nothing is pending. Reclaims expired leases
    /// first, so a crashed worker's job is re-offered here.
    async fn claim(&self, worker_id: &str, lease_secs: u64) -> Result<Option<CloudJob>, CloudError>;
    /// Mark a leased job complete with its result (moves it to `Done`). FENCED on
    /// the lease: only the worker that currently holds the lease may complete it.
    /// A late `complete` from a worker whose lease expired (and was reclaimed +
    /// re-leased) is rejected (`LeaseLost`) — it cannot clobber the new worker's
    /// lease or write a stale result.
    async fn complete(&self, worker_id: &str, result: CloudResult) -> Result<(), CloudError>;
    /// The job's current status (for the submitter's `DispatchHandle::poll`).
    async fn status(&self, job_id: &str) -> Result<JobStatus, CloudError>;
    /// Return every job whose lease has expired to the pending set; returns how
    /// many were requeued.
    async fn reclaim_expired(&self) -> Result<usize, CloudError>;
}

struct Leased {
    job: CloudJob,
    until: Instant,
    #[allow(dead_code)] // recorded for observability / future fencing
    worker: String,
}

#[derive(Default)]
struct Inner {
    pending: Vec<CloudJob>,
    leased: HashMap<String, Leased>,
    done: HashMap<String, CloudResult>,
}

/// Whether `id` is safe to use as an object-store key (the cross-process backend
/// turns job ids into keys). Mirrors the p2p `is_safe_task_id` guard.
fn is_safe_job_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && !id.contains("..")
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
}

impl Inner {
    /// Move every lease that has elapsed back to the pending set. Returns the count.
    /// Reclaimed jobs go to the BACK of `pending`, so among equal-priority jobs a
    /// reclaimed one loses its original insertion position (priority still wins).
    fn reclaim(&mut self, now: Instant) -> usize {
        let expired: Vec<String> = self
            .leased
            .iter()
            .filter(|(_, l)| l.until <= now)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            if let Some(l) = self.leased.remove(id) {
                self.pending.push(l.job);
            }
        }
        expired.len()
    }
}

/// In-process [`CloudQueue`] — loopback tests + single-box use.
#[derive(Default)]
pub struct MemQueue {
    inner: Mutex<Inner>,
}

impl MemQueue {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl CloudQueue for MemQueue {
    async fn enqueue(&self, job: CloudJob) -> Result<(), CloudError> {
        if !is_safe_job_id(&job.id) {
            return Err(CloudError::BadJobId(job.id));
        }
        let mut g = self.inner.lock();
        if g.leased.contains_key(&job.id)
            || g.done.contains_key(&job.id)
            || g.pending.iter().any(|j| j.id == job.id)
        {
            return Err(CloudError::DuplicateJob(job.id));
        }
        g.pending.push(job);
        Ok(())
    }

    async fn claim(&self, worker_id: &str, lease_secs: u64) -> Result<Option<CloudJob>, CloudError> {
        let now = Instant::now();
        let mut g = self.inner.lock();
        g.reclaim(now); // re-offer crashed workers' jobs before picking
        if g.pending.is_empty() {
            return Ok(None);
        }
        // Highest priority first; ties broken by insertion order (stable: max_by_key
        // keeps the LAST max, so scan front-to-back picking the first max index).
        let idx = g
            .pending
            .iter()
            .enumerate()
            .max_by_key(|(i, j)| (j.priority, std::cmp::Reverse(*i)))
            .map(|(i, _)| i)
            .expect("pending non-empty");
        let job = g.pending.remove(idx);
        g.leased.insert(
            job.id.clone(),
            Leased {
                job: job.clone(),
                until: now + Duration::from_secs(lease_secs),
                worker: worker_id.to_string(),
            },
        );
        Ok(Some(job))
    }

    async fn complete(&self, worker_id: &str, result: CloudResult) -> Result<(), CloudError> {
        let mut g = self.inner.lock();
        // Fence: only the current lease holder may complete. If the lease expired
        // and the job was reclaimed (now pending, or re-leased to another worker),
        // reject — don't evict the new lease or write a stale result.
        match g.leased.get(&result.job_id) {
            Some(l) if l.worker == worker_id => {}
            _ => return Err(CloudError::LeaseLost(result.job_id)),
        }
        g.leased.remove(&result.job_id);
        g.done.insert(result.job_id.clone(), result);
        Ok(())
    }

    async fn status(&self, job_id: &str) -> Result<JobStatus, CloudError> {
        let g = self.inner.lock();
        if let Some(r) = g.done.get(job_id) {
            return Ok(JobStatus::Done(r.clone()));
        }
        if g.leased.contains_key(job_id) {
            return Ok(JobStatus::Running);
        }
        if g.pending.iter().any(|j| j.id == job_id) {
            return Ok(JobStatus::Queued);
        }
        Ok(JobStatus::Unknown)
    }

    async fn reclaim_expired(&self) -> Result<usize, CloudError> {
        Ok(self.inner.lock().reclaim(Instant::now()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud::job::JobOutcome;
    use crate::framework::artifact::ContentHash;
    use crate::framework::stage::ErasedArtifact;
    use crate::p2p::bundle::BundleManifest;
    use crate::p2p::task::ResourceRequest;
    use crate::p2p::trust::DataClass;

    // The queue never inspects the manifest, so a minimal one suffices here.
    fn dummy_manifest() -> BundleManifest {
        BundleManifest {
            bundle_version: 1,
            erased: ErasedArtifact { kind: "test".into(), schema: 1, payload: vec![] },
            kind: "test".into(),
            schema: 1,
            content_hash: ContentHash::of_bytes(b""),
            src_root: std::path::PathBuf::from("/tmp"),
            files: vec![],
            blob_len: 0,
            blob_sha256: ContentHash::of_bytes(b""),
        }
    }

    fn job(id: &str, priority: i32) -> CloudJob {
        CloudJob {
            id: id.to_string(),
            stage_name: "p2p-echo".to_string(),
            stage_schema: 1,
            args: serde_json::json!({}),
            input_blob_key: ContentHash::of_bytes(id.as_bytes()),
            input_manifest: dummy_manifest(),
            expected_output_hash: ContentHash::of_bytes(b"out"),
            resources: ResourceRequest::default(),
            data_class: DataClass::Public,
            priority,
            timeout_secs: 30,
        }
    }

    fn done(id: &str) -> CloudResult {
        CloudResult {
            job_id: id.to_string(),
            outcome: JobOutcome::Succeeded,
            output_blob_key: Some(ContentHash::of_bytes(b"ok")),
            output_manifest: None,
            output_hash: Some(ContentHash::of_bytes(b"ok")),
            wall_time_ms: 5,
            error: None,
        }
    }

    #[tokio::test]
    async fn enqueue_claim_complete_lifecycle() {
        let q = MemQueue::new();
        q.enqueue(job("a", 0)).await.unwrap();
        assert!(matches!(q.status("a").await.unwrap(), JobStatus::Queued));

        let claimed = q.claim("w1", 30).await.unwrap().unwrap();
        assert_eq!(claimed.id, "a");
        assert!(matches!(q.status("a").await.unwrap(), JobStatus::Running));

        q.complete("w1", done("a")).await.unwrap();
        assert!(matches!(q.status("a").await.unwrap(), JobStatus::Done(_)));
        // Nothing left to claim.
        assert!(q.claim("w1", 30).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn enqueue_rejects_dup_and_unsafe_id() {
        let q = MemQueue::new();
        q.enqueue(job("a", 0)).await.unwrap();
        assert!(matches!(
            q.enqueue(job("a", 0)).await,
            Err(CloudError::DuplicateJob(_))
        ));
        assert!(matches!(
            q.enqueue(job("../escape", 0)).await,
            Err(CloudError::BadJobId(_))
        ));
    }

    #[tokio::test]
    async fn complete_is_lease_fenced() {
        let q = MemQueue::new();
        q.enqueue(job("a", 0)).await.unwrap();
        // w1 claims with a 0s lease then "crashes"; the job is reclaimed + re-claimed by w2.
        let _ = q.claim("w1", 0).await.unwrap().unwrap();
        assert_eq!(q.reclaim_expired().await.unwrap(), 1);
        assert_eq!(q.claim("w2", 30).await.unwrap().unwrap().id, "a");
        // w1's late completion is rejected — it no longer holds the lease.
        assert!(matches!(
            q.complete("w1", done("a")).await,
            Err(CloudError::LeaseLost(_))
        ));
        assert!(matches!(q.status("a").await.unwrap(), JobStatus::Running));
        // w2 (the current holder) completes successfully.
        q.complete("w2", done("a")).await.unwrap();
        assert!(matches!(q.status("a").await.unwrap(), JobStatus::Done(_)));
    }

    #[tokio::test]
    async fn claim_is_priority_descending() {
        let q = MemQueue::new();
        q.enqueue(job("low", 1)).await.unwrap();
        q.enqueue(job("high", 9)).await.unwrap();
        q.enqueue(job("mid", 5)).await.unwrap();
        assert_eq!(q.claim("w", 30).await.unwrap().unwrap().id, "high");
        assert_eq!(q.claim("w", 30).await.unwrap().unwrap().id, "mid");
        assert_eq!(q.claim("w", 30).await.unwrap().unwrap().id, "low");
    }

    #[tokio::test]
    async fn expired_lease_is_reclaimed() {
        let q = MemQueue::new();
        q.enqueue(job("a", 0)).await.unwrap();
        // Lease 0s → expires immediately; the worker "crashes" (never completes).
        let _ = q.claim("w1", 0).await.unwrap().unwrap();
        assert!(matches!(q.status("a").await.unwrap(), JobStatus::Running));
        assert_eq!(q.reclaim_expired().await.unwrap(), 1, "expired lease requeued");
        assert!(matches!(q.status("a").await.unwrap(), JobStatus::Queued));
        // A second worker can now pick it up.
        assert_eq!(q.claim("w2", 30).await.unwrap().unwrap().id, "a");
    }

    #[tokio::test]
    async fn unknown_job_status() {
        let q = MemQueue::new();
        assert!(matches!(q.status("ghost").await.unwrap(), JobStatus::Unknown));
    }
}
