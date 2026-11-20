// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Durable cloud job queue on object storage.
//!
//! [`ObjectQueue`] is the cross-process [`CloudQueue`]: the submitter enqueues
//! on one machine, workers on others claim, and every decision is an object in
//! a bucket (S3, Cloudflare R2, MinIO) or a directory. It needs nothing from the
//! provider beyond create-only writes (`If-None-Match: *`), reads, listing and
//! deletes, which is what [`crate::framework::object_store`] already requires
//! of a provider; Backblaze B2, which has no conditional writes, is refused.
//!
//! # Layout
//!
//! Under `<prefix>/queue/v1/`:
//!
//! | key | written | meaning |
//! |---|---|---|
//! | `jobs/<id>/log/<seq>` | create-only | the job's append-only log (20-digit `seq`) |
//! | `jobs/<id>/lease/<seq>` | overwritten by the lease holder | renewal of the lease written at `seq` |
//! | `jobs/<id>/refusals/<worker>` | create-only | why a worker left the job alone |
//! | `active/<rank>-<ms>-<id>` | create-only, deleted when terminal | discovery index |
//!
//! # Protocol
//!
//! A job's state is the last entry of its log. Entry 0 is the job record
//! (`enqueued`): the [`CloudJob`] and the submitter's [`CodeIdentity`]. Every
//! later entry is written at `last + 1` with a create-only write, so a log slot
//! can be filled exactly once and every transition is decided by whoever fills
//! the next slot first:
//!
//! * **claim** writes `leased` (owner, lease end). Its `seq` is the assignment
//!   generation, so a re-claim always has a larger generation than the claim
//!   it replaces, even by the same worker.
//! * **renew** overwrites `lease/<seq>` with a later lease end. Only the holder
//!   writes it, and it decides nothing: a lease that ran out can be taken by
//!   whoever writes the next slot, whatever its heartbeat says afterwards.
//! * **requeue** needs no writer. A `leased` entry whose lease end (the later of
//!   the entry and its heartbeat) has passed is claimable, so the next claim
//!   writes the next slot. A worker that is shut down writes `released`, which
//!   makes the job claimable at once.
//! * **complete** writes `finished` with the worker's result at `generation + 1`.
//!   That slot is the fence: if a newer claim or a cancellation got there first,
//!   the completion is [`CloudError::LeaseLost`] and changes nothing. Repeating
//!   a completion that already landed returns `Ok`, so a retry after a lost
//!   response is harmless.
//! * **cancel** writes `finished` with a cancelled result at `last + 1`, from any
//!   process: the worker running the job sees the job is no longer its lease
//!   and stops. A job that already finished is not cancelled.
//!
//! Lease ends are wall-clock milliseconds written by the holder and read by
//! other machines, so the hosts' clocks must agree to well within a lease
//! (NTP does). A clock error only moves when a dead worker's job is requeued;
//! it cannot let two completions both land, because the log slot decides that.
//!
//! # Code identity
//!
//! A job carries the [`CodeIdentity`] of the process that enqueued it: the
//! same identity every cache key folds. A queue only claims jobs whose identity
//! equals its own. A job built by different code is left for a worker that runs
//! that code, and the refusing worker records why under `refusals/`, which the
//! submitter reports if no worker ever picks the job up.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::TryStreamExt;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStoreExt, PutMode, PutOptions, PutPayload};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::CloudError;
use super::job::{CloudJob, CloudResult};
use super::queue::{ClaimedJob, CloudQueue, JobStatus, is_safe_job_id};
use crate::framework::code_identity::CodeIdentity;
use crate::framework::execution::Assignment;

/// Version of the queue's object layout. Part of every key (`queue/v1/…`).
pub const OBJECT_QUEUE_LAYOUT_VERSION: u16 = 1;

/// Largest queue object read back. A job record carries the input bundle's
/// file table; anything beyond this is not a record this engine wrote.
const MAX_QUEUE_OBJECT: u64 = 64 * 1024 * 1024;

/// How often one operation retries after another process filled the slot it
/// wanted. Each retry re-reads the log, so this only bounds a pathological
/// storm of writers on one job.
const MAX_APPEND_RACES: usize = 32;

/// Tunables for an [`ObjectQueue`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ObjectQueueOptions {
    /// Shortest interval between two provider reads of one job's state through
    /// [`CloudQueue::status`]. The executor polls status every 100 ms; against
    /// a network bucket that would be dozens of requests a second per stage.
    /// A queue's own writes refresh the cached state immediately.
    pub status_refresh: Duration,
    /// Claim jobs that request a GPU. A worker without one leaves them (and
    /// records why) instead of running a GPU stage on a CPU host.
    pub accept_gpu: bool,
}

impl Default for ObjectQueueOptions {
    fn default() -> Self {
        Self {
            status_refresh: Duration::from_secs(1),
            accept_gpu: false,
        }
    }
}

impl ObjectQueueOptions {
    pub fn with_status_refresh(mut self, refresh: Duration) -> Self {
        self.status_refresh = refresh;
        self
    }

    pub fn with_accept_gpu(mut self, accept: bool) -> Self {
        self.accept_gpu = accept;
        self
    }
}

/// Entry 0 of a job's log: what was submitted, and by which code.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct QueuedJobRecord {
    /// [`OBJECT_QUEUE_LAYOUT_VERSION`] of the queue that wrote it.
    pub layout_version: u16,
    pub job: CloudJob,
    /// The code identity of the enqueuing process (engine + cookbooks).
    pub code_identity: CodeIdentity,
    pub enqueued_unix_ms: u64,
    /// `host:pid` of the enqueuing process, for diagnostics.
    pub enqueued_by: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "entry", rename_all = "snake_case")]
enum LogEntry {
    Enqueued {
        record: Box<QueuedJobRecord>,
    },
    Leased {
        owner: String,
        /// Identifies this write, so a claim whose create-only PUT succeeded
        /// but whose response was lost recognises its own entry on read-back.
        nonce: String,
        claimed_unix_ms: u64,
        lease_until_unix_ms: u64,
    },
    Released {
        owner: String,
        generation: u64,
        reason: String,
        at_unix_ms: u64,
    },
    Finished {
        result: Box<CloudResult>,
        at_unix_ms: u64,
    },
}

/// Heartbeat body under `lease/<seq>`.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LeaseRenewal {
    owner: String,
    lease_until_unix_ms: u64,
}

/// A worker's note on why it left a job unclaimed.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Refusal {
    worker: String,
    reason: String,
    at_unix_ms: u64,
}

/// What the log says about a job right now.
#[derive(Clone, Debug)]
enum Derived {
    /// Enqueued or released: claimable at `next`.
    Pending {
        next: u64,
    },
    /// A lease that has not run out.
    Leased {
        assignment: Assignment,
        next: u64,
    },
    /// A lease that ran out: claimable at `next`, and still completable by
    /// its holder until someone else fills `next`.
    Expired {
        assignment: Assignment,
        next: u64,
    },
    Done(Box<CloudResult>),
}

impl Derived {
    fn as_status(&self) -> JobStatus {
        match self {
            Derived::Pending { .. } | Derived::Expired { .. } => JobStatus::Queued,
            Derived::Leased { assignment, .. } => JobStatus::Running(assignment.clone()),
            Derived::Done(result) => JobStatus::Done(result.clone()),
        }
    }
}

/// A job as listed by [`ObjectQueue::list_active`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ActiveJob {
    pub job_id: String,
    pub stage_name: Option<String>,
    pub priority: Option<i32>,
    pub status: JobStatus,
}

/// Durable [`CloudQueue`] over an object store. See the [module docs](self).
pub struct ObjectQueue {
    provider: Arc<dyn object_store::ObjectStore>,
    root: String,
    options: ObjectQueueOptions,
    identity: CodeIdentity,
    identity_digest: [u8; 32],
    host: String,
    records: Mutex<HashMap<String, Arc<QueuedJobRecord>>>,
    status_cache: Mutex<HashMap<String, (Instant, JobStatus)>>,
    refused: Mutex<HashSet<String>>,
}

impl std::fmt::Debug for ObjectQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectQueue")
            .field("root", &self.root)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl ObjectQueue {
    /// A queue under `<prefix>/queue/v1/` of `provider`. Enqueues stamp, and
    /// claims require, this process's installed code identity
    /// ([`crate::framework::code_identity::current`]).
    pub fn new(
        provider: Arc<dyn object_store::ObjectStore>,
        prefix: impl AsRef<str>,
        options: ObjectQueueOptions,
    ) -> Self {
        let prefix = prefix.as_ref().trim_matches('/');
        let root = if prefix.is_empty() {
            format!("queue/v{OBJECT_QUEUE_LAYOUT_VERSION}")
        } else {
            format!("{prefix}/queue/v{OBJECT_QUEUE_LAYOUT_VERSION}")
        };
        let identity = crate::framework::code_identity::current().clone();
        let identity_digest = identity.digest();
        Self {
            provider,
            root,
            options,
            identity,
            identity_digest,
            host: format!("{}:{}", hostname(), std::process::id()),
            records: Mutex::new(HashMap::new()),
            status_cache: Mutex::new(HashMap::new()),
            refused: Mutex::new(HashSet::new()),
        }
    }

    /// A queue in a local directory (created if absent), for one machine or a
    /// shared filesystem. Create-only writes are atomic there too.
    pub fn local(
        dir: impl AsRef<std::path::Path>,
        prefix: impl AsRef<str>,
        options: ObjectQueueOptions,
    ) -> Result<Self, CloudError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|error| {
            CloudError::Dispatch(format!(
                "create cloud queue directory {}: {error}",
                dir.display()
            ))
        })?;
        let provider =
            object_store::local::LocalFileSystem::new_with_prefix(dir).map_err(|error| {
                CloudError::Dispatch(format!("open cloud queue directory: {error}"))
            })?;
        Ok(Self::new(Arc::new(provider), prefix, options))
    }

    /// Stamp and require `identity` instead of the process's own. For tools
    /// that act for another build, and for tests that model two builds in one
    /// process.
    pub fn with_code_identity(mut self, identity: CodeIdentity) -> Self {
        self.identity_digest = identity.digest();
        self.identity = identity;
        self
    }

    /// The code identity this queue stamps on enqueue and requires on claim.
    pub fn code_identity(&self) -> &CodeIdentity {
        &self.identity
    }

    /// The job record (entry 0 of its log), if the job exists.
    pub async fn record(&self, job_id: &str) -> Result<Option<Arc<QueuedJobRecord>>, CloudError> {
        if !is_safe_job_id(job_id) {
            return Err(CloudError::BadJobId(job_id.to_string()));
        }
        if let Some(record) = self.records.lock().get(job_id) {
            return Ok(Some(record.clone()));
        }
        let Some(bytes) = self.read(&self.log_key(job_id, 0)).await? else {
            return Ok(None);
        };
        match decode::<LogEntry>(&bytes, "job record")? {
            LogEntry::Enqueued { record } => {
                let record: Arc<QueuedJobRecord> = Arc::from(record);
                self.records
                    .lock()
                    .insert(job_id.to_string(), record.clone());
                Ok(Some(record))
            }
            _ => Err(CloudError::Dispatch(format!(
                "cloud job '{job_id}': log entry 0 is not a job record"
            ))),
        }
    }

    /// One listing request, to prove the location is reachable and the
    /// credentials are accepted before anything is submitted.
    pub async fn probe(&self) -> Result<(), CloudError> {
        let prefix = self.active_prefix();
        let mut listing = self.provider.list(Some(&prefix));
        match listing.try_next().await {
            Ok(_) => Ok(()),
            Err(error) => Err(io_error("list", &prefix, error)),
        }
    }

    /// Every job still in the discovery index, highest priority first, with
    /// its state. Finished jobs whose index entry outlived them are cleaned up
    /// on the way.
    pub async fn list_active(&self) -> Result<Vec<ActiveJob>, CloudError> {
        let mut out = Vec::new();
        for (key, job_id) in self.active_keys().await? {
            let Some(derived) = self.derive(&job_id).await? else {
                continue;
            };
            if let Derived::Done(_) = derived {
                self.delete_quietly(&key).await;
                continue;
            }
            let record = self.record(&job_id).await.ok().flatten();
            out.push(ActiveJob {
                job_id,
                stage_name: record.as_ref().map(|r| r.job.stage_name.clone()),
                priority: record.as_ref().map(|r| r.job.priority),
                status: derived.as_status(),
            });
        }
        Ok(out)
    }

    fn key(&self, rest: &str) -> ObjectPath {
        ObjectPath::from(format!("{}/{rest}", self.root))
    }

    fn log_key(&self, job_id: &str, seq: u64) -> ObjectPath {
        self.key(&format!("jobs/{job_id}/log/{seq:020}"))
    }

    fn log_prefix(&self, job_id: &str) -> ObjectPath {
        self.key(&format!("jobs/{job_id}/log"))
    }

    fn lease_key(&self, job_id: &str, seq: u64) -> ObjectPath {
        self.key(&format!("jobs/{job_id}/lease/{seq:020}"))
    }

    fn refusal_prefix(&self, job_id: &str) -> ObjectPath {
        self.key(&format!("jobs/{job_id}/refusals"))
    }

    fn active_prefix(&self) -> ObjectPath {
        self.key("active")
    }

    /// `active/<rank>-<ms>-<id>`: lexical order is priority descending, then
    /// enqueue order, then id.
    fn active_key(&self, record: &QueuedJobRecord) -> ObjectPath {
        let rank = i64::from(i32::MAX) - i64::from(record.job.priority);
        self.key(&format!(
            "active/{rank:010}-{:013}-{}",
            record.enqueued_unix_ms, record.job.id
        ))
    }

    async fn read(&self, key: &ObjectPath) -> Result<Option<Vec<u8>>, CloudError> {
        let result = match self.provider.get(key).await {
            Ok(result) => result,
            Err(object_store::Error::NotFound { .. }) => return Ok(None),
            Err(error) => return Err(io_error("read", key, error)),
        };
        if result.meta.size > MAX_QUEUE_OBJECT {
            return Err(CloudError::Dispatch(format!(
                "cloud queue object {key} is {} bytes; the most a queue object may be is {MAX_QUEUE_OBJECT}",
                result.meta.size
            )));
        }
        let bytes = result
            .bytes()
            .await
            .map_err(|error| io_error("read", key, error))?;
        if bytes.len() as u64 > MAX_QUEUE_OBJECT {
            return Err(CloudError::Dispatch(format!(
                "cloud queue object {key} exceeds {MAX_QUEUE_OBJECT} bytes"
            )));
        }
        Ok(Some(bytes.to_vec()))
    }

    /// Create `key` only if absent. `Ok(false)`: another writer got there first.
    async fn create(&self, key: &ObjectPath, body: Vec<u8>) -> Result<bool, CloudError> {
        let options = PutOptions::from(PutMode::Create);
        match self
            .provider
            .put_opts(key, PutPayload::from(body), options)
            .await
        {
            Ok(_) => Ok(true),
            Err(object_store::Error::AlreadyExists { .. }) => Ok(false),
            Err(error) if crate::framework::object_store::rejects_conditional_put(&error) => {
                Err(CloudError::Dispatch(
                    crate::framework::object_store::ConditionalPutUnsupported(error.to_string())
                        .to_string(),
                ))
            }
            Err(error) => Err(io_error("create", key, error)),
        }
    }

    async fn overwrite(&self, key: &ObjectPath, body: Vec<u8>) -> Result<(), CloudError> {
        self.provider
            .put_opts(key, PutPayload::from(body), PutOptions::default())
            .await
            .map(|_| ())
            .map_err(|error| io_error("write", key, error))
    }

    async fn delete_quietly(&self, key: &ObjectPath) {
        match self.provider.delete(key).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => {}
            Err(error) => tracing::warn!("cloud queue: delete {key}: {error}"),
        }
    }

    async fn list(&self, prefix: &ObjectPath) -> Result<Vec<String>, CloudError> {
        let metas: Vec<object_store::ObjectMeta> = self
            .provider
            .list(Some(prefix))
            .try_collect()
            .await
            .map_err(|error| io_error("list", prefix, error))?;
        let mut names: Vec<String> = metas
            .into_iter()
            .filter_map(|meta| meta.location.filename().map(str::to_string))
            .collect();
        names.sort();
        Ok(names)
    }

    async fn active_keys(&self) -> Result<Vec<(ObjectPath, String)>, CloudError> {
        let prefix = self.active_prefix();
        let mut out = Vec::new();
        for name in self.list(&prefix).await? {
            let mut parts = name.splitn(3, '-');
            let (Some(_rank), Some(_ms), Some(job_id)) = (parts.next(), parts.next(), parts.next())
            else {
                continue;
            };
            if is_safe_job_id(job_id) {
                out.push((self.key(&format!("active/{name}")), job_id.to_string()));
            }
        }
        Ok(out)
    }

    async fn log_seqs(&self, job_id: &str) -> Result<Vec<u64>, CloudError> {
        let mut seqs: Vec<u64> = self
            .list(&self.log_prefix(job_id))
            .await?
            .iter()
            .filter_map(|name| name.parse::<u64>().ok())
            .collect();
        seqs.sort_unstable();
        Ok(seqs)
    }

    async fn entry(&self, job_id: &str, seq: u64) -> Result<Option<LogEntry>, CloudError> {
        match self.read(&self.log_key(job_id, seq)).await? {
            Some(bytes) => decode(&bytes, "cloud queue log entry").map(Some),
            None => Ok(None),
        }
    }

    /// Read the log and say what state the job is in. `None`: no such job.
    async fn derive(&self, job_id: &str) -> Result<Option<Derived>, CloudError> {
        let seqs = self.log_seqs(job_id).await?;
        let Some(&last) = seqs.last() else {
            return Ok(None);
        };
        let Some(entry) = self.entry(job_id, last).await? else {
            return Err(CloudError::Dispatch(format!(
                "cloud job '{job_id}': log entry {last} was listed but cannot be read"
            )));
        };
        let next = last + 1;
        let derived = match entry {
            LogEntry::Enqueued { record } => {
                let record: Arc<QueuedJobRecord> = Arc::from(record);
                self.records.lock().insert(job_id.to_string(), record);
                Derived::Pending { next }
            }
            LogEntry::Released { .. } => Derived::Pending { next },
            LogEntry::Finished { result, .. } => Derived::Done(result),
            LogEntry::Leased {
                owner,
                lease_until_unix_ms,
                ..
            } => {
                let renewed = match self.read(&self.lease_key(job_id, last)).await? {
                    Some(bytes) => decode::<LeaseRenewal>(&bytes, "lease renewal")
                        .ok()
                        .filter(|renewal| renewal.owner == owner)
                        .map(|renewal| renewal.lease_until_unix_ms),
                    None => None,
                };
                let until = renewed.map_or(lease_until_unix_ms, |r| r.max(lease_until_unix_ms));
                let assignment = Assignment::new(owner, last);
                if unix_ms() >= until {
                    Derived::Expired { assignment, next }
                } else {
                    Derived::Leased { assignment, next }
                }
            }
        };
        self.status_cache
            .lock()
            .insert(job_id.to_string(), (Instant::now(), derived.as_status()));
        Ok(Some(derived))
    }

    fn forget_status(&self, job_id: &str) {
        self.status_cache.lock().remove(job_id);
    }

    async fn drop_from_index(&self, job_id: &str) {
        if let Ok(Some(record)) = self.record(job_id).await {
            let key = self.active_key(&record);
            self.delete_quietly(&key).await;
        }
    }

    /// Whether this queue may claim `record`, and why not.
    fn refusal_for(&self, record: &QueuedJobRecord) -> Option<String> {
        if record.code_identity.digest() != self.identity_digest {
            return Some(format!(
                "code identity differs: the job was built by [{}], this worker runs [{}]",
                record.code_identity, self.identity
            ));
        }
        if record.job.resources.gpu && !self.options.accept_gpu {
            return Some(
                "the stage requests a GPU and this worker was not started with --gpu".into(),
            );
        }
        None
    }

    async fn note_refusal(&self, job_id: &str, worker_id: &str, reason: &str) {
        let marker = format!("{job_id}\u{0}{worker_id}");
        if !self.refused.lock().insert(marker) {
            return;
        }
        tracing::warn!("cloud worker {worker_id} leaves job {job_id}: {reason}");
        let body = match serde_json::to_vec(&Refusal {
            worker: worker_id.to_string(),
            reason: reason.to_string(),
            at_unix_ms: unix_ms(),
        }) {
            Ok(body) => body,
            Err(_) => return,
        };
        let key = self.key(&format!("jobs/{job_id}/refusals/{}", safe_name(worker_id)));
        if let Err(error) = self.create(&key, body).await {
            tracing::warn!("cloud queue: record refusal of {job_id}: {error}");
        }
    }

    /// Write `entry` at `seq` of `job_id`'s log. `Ok(false)`: the slot was
    /// already filled by someone else.
    async fn append(&self, job_id: &str, seq: u64, entry: &LogEntry) -> Result<bool, CloudError> {
        let body = serde_json::to_vec(entry)
            .map_err(|error| CloudError::Dispatch(format!("encode queue entry: {error}")))?;
        let created = self.create(&self.log_key(job_id, seq), body).await?;
        self.forget_status(job_id);
        Ok(created)
    }

    async fn try_claim(
        &self,
        job_id: &str,
        next: u64,
        worker_id: &str,
        lease_secs: u64,
    ) -> Result<Option<ClaimedJob>, CloudError> {
        let Some(record) = self.record(job_id).await? else {
            return Ok(None);
        };
        if let Some(reason) = self.refusal_for(&record) {
            self.note_refusal(job_id, worker_id, &reason).await;
            return Ok(None);
        }
        let now = unix_ms();
        let nonce = uuid::Uuid::new_v4().to_string();
        let entry = LogEntry::Leased {
            owner: worker_id.to_string(),
            nonce: nonce.clone(),
            claimed_unix_ms: now,
            lease_until_unix_ms: now.saturating_add(lease_secs.saturating_mul(1000)),
        };
        let won = if self.append(job_id, next, &entry).await? {
            true
        } else {
            // A create whose response was lost is retried by the client and
            // then reports "already exists" for our own write.
            matches!(
                self.entry(job_id, next).await?,
                Some(LogEntry::Leased { nonce: written, .. }) if written == nonce
            )
        };
        Ok(won.then(|| ClaimedJob {
            job: record.job.clone(),
            assignment: Assignment::new(worker_id, next),
        }))
    }
}

#[async_trait]
impl CloudQueue for ObjectQueue {
    async fn enqueue(&self, job: CloudJob) -> Result<(), CloudError> {
        if !is_safe_job_id(&job.id) {
            return Err(CloudError::BadJobId(job.id));
        }
        let job_id = job.id.clone();
        let record = QueuedJobRecord {
            layout_version: OBJECT_QUEUE_LAYOUT_VERSION,
            job,
            code_identity: self.identity.clone(),
            enqueued_unix_ms: unix_ms(),
            enqueued_by: self.host.clone(),
        };
        let active = self.active_key(&record);
        let entry = LogEntry::Enqueued {
            record: Box::new(record),
        };
        if !self.append(&job_id, 0, &entry).await? {
            return Err(CloudError::DuplicateJob(job_id));
        }
        // The log is written first: an index entry never points at nothing.
        // A crash between the two leaves an undiscoverable job, which the
        // submitter's pickup window then cancels.
        self.create(&active, br#"{}"#.to_vec()).await?;
        Ok(())
    }

    async fn claim(
        &self,
        worker_id: &str,
        lease_secs: u64,
    ) -> Result<Option<ClaimedJob>, CloudError> {
        for (key, job_id) in self.active_keys().await? {
            let next = match self.derive(&job_id).await? {
                None | Some(Derived::Leased { .. }) => continue,
                Some(Derived::Done(_)) => {
                    self.delete_quietly(&key).await;
                    continue;
                }
                Some(Derived::Pending { next } | Derived::Expired { next, .. }) => next,
            };
            if let Some(claimed) = self.try_claim(&job_id, next, worker_id, lease_secs).await? {
                return Ok(Some(claimed));
            }
        }
        Ok(None)
    }

    async fn complete(
        &self,
        assignment: &Assignment,
        mut result: CloudResult,
    ) -> Result<(), CloudError> {
        let job_id = result.job_id.clone();
        if !is_safe_job_id(&job_id) {
            return Err(CloudError::BadJobId(job_id));
        }
        result.assignment = Some(assignment.clone());
        match self.derive(&job_id).await? {
            Some(Derived::Leased {
                assignment: current,
                next,
            })
            | Some(Derived::Expired {
                assignment: current,
                next,
            }) if &current == assignment => {
                let entry = LogEntry::Finished {
                    result: Box::new(result),
                    at_unix_ms: unix_ms(),
                };
                if self.append(&job_id, next, &entry).await? {
                    self.drop_from_index(&job_id).await;
                    return Ok(());
                }
                // Our own completion, written by an earlier attempt whose
                // response was lost, is a success; anything else won the slot.
                match self.entry(&job_id, next).await? {
                    Some(LogEntry::Finished { result, .. })
                        if result.assignment.as_ref() == Some(assignment) =>
                    {
                        Ok(())
                    }
                    _ => Err(CloudError::LeaseLost(job_id)),
                }
            }
            Some(Derived::Done(done)) if done.assignment.as_ref() == Some(assignment) => Ok(()),
            _ => Err(CloudError::LeaseLost(job_id)),
        }
    }

    async fn cancel(&self, job_id: &str, reason: &str) -> Result<bool, CloudError> {
        if !is_safe_job_id(job_id) {
            return Err(CloudError::BadJobId(job_id.to_string()));
        }
        for _ in 0..MAX_APPEND_RACES {
            let next = match self.derive(job_id).await? {
                None | Some(Derived::Done(_)) => return Ok(false),
                Some(
                    Derived::Pending { next }
                    | Derived::Leased { next, .. }
                    | Derived::Expired { next, .. },
                ) => next,
            };
            let entry = LogEntry::Finished {
                result: Box::new(CloudResult::cancelled(job_id, reason)),
                at_unix_ms: unix_ms(),
            };
            if self.append(job_id, next, &entry).await? {
                self.drop_from_index(job_id).await;
                return Ok(true);
            }
        }
        Err(CloudError::Dispatch(format!(
            "cancel of cloud job '{job_id}' lost {MAX_APPEND_RACES} races in a row"
        )))
    }

    async fn status(&self, job_id: &str) -> Result<JobStatus, CloudError> {
        if !is_safe_job_id(job_id) {
            return Err(CloudError::BadJobId(job_id.to_string()));
        }
        if let Some((at, status)) = self.status_cache.lock().get(job_id)
            && at.elapsed() < self.options.status_refresh
        {
            return Ok(status.clone());
        }
        Ok(match self.derive(job_id).await? {
            Some(derived) => derived.as_status(),
            None => JobStatus::Unknown,
        })
    }

    async fn reclaim_expired(&self) -> Result<usize, CloudError> {
        // Nothing to write: an expired lease is claimable as it stands. Count
        // them, which is what callers of this method observe.
        let mut expired = 0;
        for (_, job_id) in self.active_keys().await? {
            if matches!(self.derive(&job_id).await?, Some(Derived::Expired { .. })) {
                expired += 1;
            }
        }
        Ok(expired)
    }

    async fn renew(
        &self,
        job_id: &str,
        assignment: &Assignment,
        lease_secs: u64,
    ) -> Result<bool, CloudError> {
        if !is_safe_job_id(job_id) {
            return Err(CloudError::BadJobId(job_id.to_string()));
        }
        match self.derive(job_id).await? {
            Some(Derived::Leased {
                assignment: current,
                ..
            })
            | Some(Derived::Expired {
                assignment: current,
                ..
            }) if &current == assignment => {
                let body = serde_json::to_vec(&LeaseRenewal {
                    owner: assignment.owner.clone(),
                    lease_until_unix_ms: unix_ms().saturating_add(lease_secs.saturating_mul(1000)),
                })
                .map_err(|error| CloudError::Dispatch(format!("encode lease: {error}")))?;
                self.overwrite(&self.lease_key(job_id, assignment.generation), body)
                    .await?;
                self.forget_status(job_id);
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    async fn release(
        &self,
        job_id: &str,
        assignment: &Assignment,
        reason: &str,
    ) -> Result<bool, CloudError> {
        if !is_safe_job_id(job_id) {
            return Err(CloudError::BadJobId(job_id.to_string()));
        }
        match self.derive(job_id).await? {
            Some(Derived::Leased {
                assignment: current,
                next,
            })
            | Some(Derived::Expired {
                assignment: current,
                next,
            }) if &current == assignment => {
                let entry = LogEntry::Released {
                    owner: assignment.owner.clone(),
                    generation: assignment.generation,
                    reason: reason.to_string(),
                    at_unix_ms: unix_ms(),
                };
                self.append(job_id, next, &entry).await
            }
            _ => Ok(false),
        }
    }

    async fn refusals(&self, job_id: &str) -> Result<Vec<String>, CloudError> {
        if !is_safe_job_id(job_id) {
            return Err(CloudError::BadJobId(job_id.to_string()));
        }
        let prefix = self.refusal_prefix(job_id);
        let mut out = Vec::new();
        for name in self.list(&prefix).await? {
            let key = self.key(&format!("jobs/{job_id}/refusals/{name}"));
            if let Some(bytes) = self.read(&key).await?
                && let Ok(refusal) = decode::<Refusal>(&bytes, "refusal")
            {
                out.push(format!("worker {}: {}", refusal.worker, refusal.reason));
            }
        }
        Ok(out)
    }
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8], what: &str) -> Result<T, CloudError> {
    serde_json::from_slice(bytes)
        .map_err(|error| CloudError::Dispatch(format!("decode {what}: {error}")))
}

fn io_error(op: &str, key: &ObjectPath, error: object_store::Error) -> CloudError {
    CloudError::Dispatch(format!("cloud queue {op} {key}: {error}"))
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn hostname() -> String {
    crate::framework::stage_checkpoint::hostname().unwrap_or_else(|| "unknown-host".into())
}

/// A worker id as one key segment: `[A-Za-z0-9._-]`, anything else `_`.
fn safe_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .take(128)
        .collect();
    let cleaned = cleaned.trim_start_matches('.').to_string();
    if cleaned.is_empty() {
        "worker".into()
    } else {
        cleaned
    }
}

#[cfg(test)]
#[path = "object_queue_tests.rs"]
mod tests;
