// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Mid-stage checkpoint slots: where a long stage keeps its progress so an
//! interrupted attempt resumes from its latest durable checkpoint instead of
//! step 0 (trainer contract v1.1, CONTRACT.md §8).
//!
//! A stage opts in with [`crate::framework::stage::Stage::checkpoint_slot`].
//! For every local attempt of such a stage the executor:
//!
//! 1. resolves a **slot**: a directory outside the job directory, keyed by the
//!    tenant, the stage name and the stage's cache key ([`slot_dir`]). The
//!    cache key covers the stage's arguments, schema, code fingerprint and the
//!    content of its inputs, so a changed config or input lands in a different
//!    slot and can never resume a stale checkpoint. The slot survives a retry,
//!    a `plan resume`, and a Slurm requeue that re-runs the recipe as a new job;
//! 2. takes an exclusive, non-blocking lock on the slot ([`SlotLease`]), so two
//!    BLUT processes never drive one slot at once;
//! 3. reads the trainer's marker ([`SlotMarker`], `state.json`) and the owner
//!    record of the attempt that last used the slot ([`SlotOwner`],
//!    `owner.json`) and decides ([`decide_slot`]) whether this attempt starts
//!    fresh, resumes, or must refuse because a trainer from an earlier attempt
//!    may still be alive;
//! 4. hands the decision to the stage as a [`CheckpointSlot`] in
//!    `StageContext::checkpoint`. The stage passes it to its trainer process
//!    as environment variables ([`CheckpointSlot::apply_to`]);
//! 5. after the attempt, records whether the trainer actually resumed and from
//!    which step (`StageEvent::StageCheckpoint`), and after the stage's output
//!    is committed to the cache, deletes the slot.
//!
//! The engine never reads or writes checkpoint files. It reads two small JSON
//! records, and it clears the slot when it decides an attempt starts fresh.
//! A trainer that ignores the environment variables writes no marker, so every
//! attempt is `Fresh` and the stage behaves exactly as it did before slots.

use std::fs::File;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::framework::artifact::InvocationKey;

/// Environment variable naming the slot directory the trainer writes its
/// checkpoints and its `state.json` marker into.
pub const ENV_CHECKPOINT_DIR: &str = "BLUT_CHECKPOINT_DIR";
/// `1` when the trainer must resume from the newest valid checkpoint in the
/// slot, `0` when it must start from step 0.
pub const ENV_RESUME: &str = "BLUT_RESUME";
/// The job (run) id of the attempt, copied into the marker's `run_id`.
pub const ENV_RUN_ID: &str = "BLUT_RUN_ID";
/// An identity unique to this attempt, copied into the marker's `attempt_id`.
pub const ENV_ATTEMPT_ID: &str = "BLUT_ATTEMPT_ID";
/// Overrides the directory slots are created under ([`checkpoints_root`]).
/// It must be on a filesystem every node of a multi-node stage can reach
/// when ranks other than 0 read checkpoints.
pub const ENV_CHECKPOINTS_ROOT: &str = "BLUT_CHECKPOINTS_DIR";

/// The trainer-written marker inside a slot.
pub const MARKER_FILE: &str = "state.json";
/// The engine-written record of the attempt that last took the slot.
pub const OWNER_FILE: &str = "owner.json";
/// The lock file the engine holds for the duration of an attempt.
pub const LOCK_FILE: &str = ".lock";

/// What a stage hands its trainer: where to checkpoint and whether to resume.
///
/// Built by the executor; a stage reads it from `StageContext::checkpoint`
/// and forwards it with [`apply_to`](Self::apply_to) or
/// [`env_vars`](Self::env_vars).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CheckpointSlot {
    /// The slot directory. Stable for the stage's cache key.
    pub dir: PathBuf,
    /// `true` when the trainer must resume from the newest valid checkpoint.
    pub resume: bool,
    /// The step the marker says was last made durable. Advisory: the trainer
    /// may fall back to an older checkpoint if the newest one is corrupt.
    pub offered_step: Option<u64>,
    /// The job id of this attempt.
    pub run_id: String,
    /// The identity of this attempt (`<run_id>/<invocation>/<attempt>`).
    pub attempt_id: String,
}

impl CheckpointSlot {
    /// A slot handed to a stage. Public so cookbook tests can build one.
    pub fn new(
        dir: impl Into<PathBuf>,
        resume: bool,
        run_id: impl Into<String>,
        attempt_id: impl Into<String>,
    ) -> Self {
        Self {
            dir: dir.into(),
            resume,
            offered_step: None,
            run_id: run_id.into(),
            attempt_id: attempt_id.into(),
        }
    }

    /// The environment variables a trainer reads (CONTRACT.md §8).
    pub fn env_vars(&self) -> Vec<(&'static str, String)> {
        vec![
            (ENV_CHECKPOINT_DIR, self.dir.to_string_lossy().into_owned()),
            (ENV_RESUME, if self.resume { "1" } else { "0" }.to_string()),
            (ENV_RUN_ID, self.run_id.clone()),
            (ENV_ATTEMPT_ID, self.attempt_id.clone()),
        ]
    }

    /// Set [`env_vars`](Self::env_vars) on a trainer command. A wrapping
    /// launcher (`srun`, `torchrun`) passes the environment on to every rank.
    pub fn apply_to(&self, cmd: &mut std::process::Command) {
        for (k, v) in self.env_vars() {
            cmd.env(k, v);
        }
    }
}

/// The trainer's `state.json` as the engine reads it. The same file the
/// durable-resume marker ([`crate::framework::resume::ResumeState`]) describes,
/// with the fields slot decisions need. Unknown fields are ignored and every
/// field defaults, so a marker written by an older trainer still parses.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SlotMarker {
    /// `"running"` while a trainer owns the slot; `"finished"` after a run
    /// its trainer considers complete and not worth resuming.
    #[serde(default)]
    pub status: String,
    /// The `BLUT_RUN_ID` of the writing attempt.
    #[serde(default)]
    pub run_id: String,
    /// PID of the writing process (rank 0 under DDP).
    #[serde(default)]
    pub pid: u32,
    /// The host the writing process runs on (its `gethostname`).
    #[serde(default)]
    pub host: Option<String>,
    /// Last heartbeat, Unix seconds.
    #[serde(default)]
    pub heartbeat_unix: u64,
    /// The global step of the newest durable checkpoint, if any.
    #[serde(default)]
    pub step: Option<u64>,
    /// The epoch of that checkpoint.
    #[serde(default)]
    pub epoch: Option<u64>,
    /// The `BLUT_ATTEMPT_ID` of the writing attempt.
    #[serde(default)]
    pub attempt_id: Option<String>,
    /// The step this attempt actually resumed from, written once the
    /// checkpoint is loaded. `None` when the attempt started at step 0.
    #[serde(default)]
    pub resumed_from_step: Option<u64>,
}

impl SlotMarker {
    /// Read `<dir>/state.json`. `None` when absent or unparseable; a corrupt
    /// marker is logged, and treated as no marker (a fresh start).
    pub fn read(dir: &Path) -> Option<SlotMarker> {
        read_json(&dir.join(MARKER_FILE))
    }
}

/// The engine's record of the attempt that took a slot (`owner.json`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SlotOwner {
    /// The job id.
    #[serde(default)]
    pub run_id: String,
    /// `<run_id>/<invocation>/<attempt>`.
    #[serde(default)]
    pub attempt_id: String,
    /// Unique per BLUT process execution of a plan; in-process retries share
    /// it, a `plan resume` or a requeued job does not.
    #[serde(default)]
    pub invocation: String,
    /// The engine's host.
    #[serde(default)]
    pub host: Option<String>,
    /// The engine's PID.
    #[serde(default)]
    pub pid: u32,
    /// `SLURM_JOB_ID` of the engine's environment, when it runs in an
    /// allocation (for example a requeueable `sbatch` job).
    #[serde(default)]
    pub slurm_job_id: Option<String>,
    /// `SLURM_RESTART_COUNT`: how many times Slurm has requeued that job.
    /// Absent in the first run of a job; read as 0.
    #[serde(default)]
    pub slurm_restart_count: Option<u32>,
    /// When the attempt took the slot, Unix seconds.
    #[serde(default)]
    pub started_unix: u64,
}

impl SlotOwner {
    /// The owner record for an attempt of this process, reading the host and
    /// the Slurm job identity from the environment.
    pub fn current(run_id: &str, invocation: &str, attempt: u32) -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let slurm_job_id = env("SLURM_JOB_ID");
        let slurm_restart_count = slurm_job_id.as_ref().map(|_| {
            env("SLURM_RESTART_COUNT")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0)
        });
        Self {
            run_id: run_id.to_string(),
            attempt_id: format!("{run_id}/{invocation}/{attempt}"),
            invocation: invocation.to_string(),
            host: hostname(),
            pid: std::process::id(),
            slurm_job_id,
            slurm_restart_count,
            started_unix: now_unix(),
        }
    }

    /// Read `<dir>/owner.json`; `None` when absent or unparseable.
    pub fn read(dir: &Path) -> Option<SlotOwner> {
        read_json(&dir.join(OWNER_FILE))
    }

    /// Write `<dir>/owner.json` atomically (temp file, fsync, rename).
    pub fn write(&self, dir: &Path) -> std::io::Result<()> {
        let body = serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?;
        write_atomic(&dir.join(OWNER_FILE), &body)
    }
}

/// A slot event recorded in the job's status stream
/// (`StageEvent::StageCheckpoint`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CheckpointAction {
    /// The attempt starts from step 0 (the slot was cleared), or was offered
    /// a checkpoint and its trainer found none it could load.
    Fresh,
    /// The engine told the trainer to resume; `step` is the marker's step.
    ResumeOffered,
    /// The trainer resumed; `step` is the step it loaded.
    Resumed,
    /// The stage's output was committed and the slot deleted.
    Cleaned,
    /// The stage finished but the slot was kept (`detail` says why).
    Kept,
}

/// What an attempt does with its slot.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SlotDecision {
    /// Start from step 0. The slot is cleared first.
    Fresh,
    /// Resume from the newest valid checkpoint. `step` is the marker's
    /// last durable step.
    Resume {
        /// The marker's last durable step.
        step: u64,
    },
    /// A trainer from an earlier attempt may still be running against this
    /// slot. Starting now would put two trainers on one checkpoint directory.
    Busy {
        /// Who holds the slot and why it counts as alive.
        reason: String,
    },
}

/// Decide what an attempt does with its slot. Pure; the liveness probe is
/// injected so tests need no real processes.
///
/// * `marker` — the trainer's `state.json`, if any.
/// * `previous` — the `owner.json` of the attempt that last took the slot.
/// * `me` — this attempt's owner record.
/// * `alive(pid)` — `Some(false)` only when the PID is known to be gone on
///   this host; `Some(true)` when it exists; `None` when it cannot be told.
///
/// A `running` marker counts as abandoned, and its checkpoint as resumable,
/// when any of these holds:
///
/// * the previous attempt belongs to this same BLUT invocation (an in-process
///   retry: the executor only starts an attempt after the last one returned);
/// * the previous attempt ran in the same Slurm job with a lower
///   `SLURM_RESTART_COUNT` (Slurm requeued the job, which ends every process
///   of the earlier run);
/// * the marker was written on this host by a PID that no longer exists;
/// * the heartbeat is at least `stale_after_secs` old.
///
/// Otherwise the trainer may be alive and the answer is
/// [`SlotDecision::Busy`]. An abandoned marker with a checkpoint step resumes;
/// one without (the trainer died before its first checkpoint) starts fresh.
/// No marker, an unparseable one, or `status: "finished"` starts fresh.
pub fn decide_slot(
    marker: Option<&SlotMarker>,
    previous: Option<&SlotOwner>,
    me: &SlotOwner,
    now_unix: u64,
    stale_after_secs: u64,
    alive: &dyn Fn(u32) -> Option<bool>,
) -> SlotDecision {
    let Some(m) = marker else {
        return SlotDecision::Fresh;
    };
    if m.status == "finished" {
        return SlotDecision::Fresh;
    }
    let same_invocation = previous.is_some_and(|p| {
        !p.invocation.is_empty()
            && p.invocation == me.invocation
            // The marker, when it names its attempt, must belong to that
            // previous attempt. A marker from some other writer is judged on
            // liveness below.
            && m.attempt_id.as_ref().is_none_or(|a| *a == p.attempt_id)
    });
    let requeued = previous.is_some_and(|p| {
        p.slurm_job_id.is_some()
            && p.slurm_job_id == me.slurm_job_id
            && p.slurm_restart_count.unwrap_or(0) < me.slurm_restart_count.unwrap_or(0)
    });
    let age = now_unix.saturating_sub(m.heartbeat_unix);
    let gone = !writer_may_be_alive(m, me.host.as_deref(), now_unix, stale_after_secs, alive);
    if !(same_invocation || requeued || gone) {
        return SlotDecision::Busy {
            reason: format!(
                "trainer pid {} on {} (run {}) wrote a heartbeat {age}s ago; it may still be \
                 running. Stop it, or wait {}s for the heartbeat to go stale, then retry",
                m.pid,
                m.host.as_deref().unwrap_or("an unknown host"),
                if m.run_id.is_empty() {
                    "unknown"
                } else {
                    m.run_id.as_str()
                },
                stale_after_secs.saturating_sub(age),
            ),
        };
    }
    match m.step {
        Some(step) => SlotDecision::Resume { step },
        None => SlotDecision::Fresh,
    }
}

/// The directory slots are created under.
///
/// `$BLUT_CHECKPOINTS_DIR` when set; otherwise a sibling of the jobs directory
/// named `<jobs-dir-name>-checkpoints` (for the default jobs directory,
/// `…/lamu/train-jobs-checkpoints`). It sits outside every job directory so a
/// requeued run, which is a new job, finds the slot its predecessor used.
pub fn checkpoints_root(job_dir: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os(ENV_CHECKPOINTS_ROOT).filter(|v| !v.is_empty()) {
        return PathBuf::from(p);
    }
    match job_dir.parent() {
        Some(jobs) if jobs.file_name().is_some() => {
            let name = jobs.file_name().unwrap_or_default().to_string_lossy();
            jobs.with_file_name(format!("{name}-checkpoints"))
        }
        _ => job_dir.join(".checkpoints"),
    }
}

/// [`checkpoints_root`] for jobs kept in `jobs_dir` (what `cache prune`
/// collects).
pub fn checkpoints_root_for_jobs(jobs_dir: &Path) -> PathBuf {
    checkpoints_root(&jobs_dir.join("job"))
}

/// What [`collect_slots`] found and did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SlotCollection {
    /// Slots deleted, and the bytes they held.
    pub removed: Vec<PathBuf>,
    /// Bytes freed by the deletions.
    pub bytes_freed: u64,
    /// Locked by a running BLUT process.
    pub kept_locked: Vec<PathBuf>,
    /// Their trainer may still be running (fresh heartbeat, or its PID is
    /// alive on this host).
    pub kept_live: Vec<PathBuf>,
    /// Used more recently than the age bound.
    pub kept_recent: Vec<PathBuf>,
}

/// Delete checkpoint slots under `root` that nothing has used for `max_age`.
///
/// A slot is left alone while another process holds its lock (a BLUT process
/// is running that stage), while its trainer may still be alive by the same
/// rules [`decide_slot`] uses, and while its newest file or heartbeat is
/// younger than `max_age`. Slots of stages that finished are deleted when
/// their output is committed, so what this collects is mostly the slots of
/// stages that failed and were never resumed.
pub fn collect_slots(root: &Path, max_age: std::time::Duration) -> std::io::Result<SlotCollection> {
    collect_slots_with(
        root,
        max_age.as_secs(),
        now_unix(),
        crate::framework::resume::DEFAULT_STALE_AFTER_SECS,
        hostname().as_deref(),
        &pid_alive,
    )
}

pub(crate) fn collect_slots_with(
    root: &Path,
    max_age_secs: u64,
    now_unix: u64,
    stale_after_secs: u64,
    this_host: Option<&str>,
    alive: &dyn Fn(u32) -> Option<bool>,
) -> std::io::Result<SlotCollection> {
    let mut out = SlotCollection::default();
    let mut slots = Vec::new();
    find_slots(root, 0, &mut slots)?;
    slots.sort();
    for dir in slots {
        let Some(lease) = SlotLease::try_acquire(&dir)? else {
            out.kept_locked.push(dir);
            continue;
        };
        if let Some(m) = SlotMarker::read(&dir)
            && m.status != "finished"
            && writer_may_be_alive(&m, this_host, now_unix, stale_after_secs, alive)
        {
            out.kept_live.push(dir);
            continue;
        }
        let last = last_activity(&dir)?;
        if now_unix.saturating_sub(last) < max_age_secs {
            out.kept_recent.push(dir);
            continue;
        }
        let bytes = dir_bytes(&dir);
        lease.remove()?;
        out.bytes_freed += bytes;
        out.removed.push(dir);
    }
    Ok(out)
}

/// Whether the process that wrote `m` may still be running: its heartbeat is
/// fresh, unless it ran on this host under a PID that no longer exists.
fn writer_may_be_alive(
    m: &SlotMarker,
    this_host: Option<&str>,
    now_unix: u64,
    stale_after_secs: u64,
    alive: &dyn Fn(u32) -> Option<bool>,
) -> bool {
    let same_host = m.host.is_some() && m.host.as_deref() == this_host;
    if same_host && m.pid != 0 && alive(m.pid) == Some(false) {
        return false;
    }
    now_unix.saturating_sub(m.heartbeat_unix) < stale_after_secs
}

/// A slot is a directory holding the engine's lock or owner record or a
/// trainer marker. Tenants nest at most two levels (`project/domain`).
fn find_slots(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path();
        if [LOCK_FILE, OWNER_FILE, MARKER_FILE]
            .iter()
            .any(|f| path.join(f).exists())
        {
            out.push(path);
        } else if depth < 2 {
            find_slots(&path, depth + 1, out)?;
        }
    }
    Ok(())
}

/// The newest modification time among the slot's files, Unix seconds.
fn last_activity(dir: &Path) -> std::io::Result<u64> {
    let mut newest = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let modified = entry?.metadata()?.modified()?;
        let secs = modified
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        newest = newest.max(secs);
    }
    Ok(newest)
}

fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// The slot for one stage invocation: `<root>/<tenant>/<stage>-<cache key>`.
pub fn slot_dir(
    root: &Path,
    tenant: &crate::tenant::Tenant,
    stage_name: &str,
    key: &InvocationKey,
) -> PathBuf {
    root.join(tenant.as_path())
        .join(format!("{}-{}", slug(stage_name), key.to_hex()))
}

/// An exclusive lock on a slot, held for the duration of an attempt. The lock
/// is released when the lease drops or the process dies, so a crashed BLUT
/// process never wedges its slot.
#[derive(Debug)]
pub struct SlotLease {
    dir: PathBuf,
    // Held only for its lock; never read.
    _lock: LockGuard,
}

impl SlotLease {
    /// Lock `dir` (created if needed). `Ok(None)` when another process holds
    /// it.
    pub fn try_acquire(dir: &Path) -> std::io::Result<Option<SlotLease>> {
        std::fs::create_dir_all(dir)?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK_FILE))?;
        Ok(try_lock(file)?.map(|guard| SlotLease {
            dir: dir.to_path_buf(),
            _lock: guard,
        }))
    }

    /// The slot directory this lease holds.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Delete everything in the slot except its lock file: the start of a
    /// fresh attempt.
    pub fn clear(&self) -> std::io::Result<()> {
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry.file_name() == LOCK_FILE {
                continue;
            }
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                std::fs::remove_dir_all(&path)?;
            } else {
                std::fs::remove_file(&path)?;
            }
        }
        Ok(())
    }

    /// Delete the slot: its checkpoints are no longer needed once the
    /// stage's output is committed. The lock is released after the directory
    /// is gone.
    pub fn remove(self) -> std::io::Result<()> {
        std::fs::remove_dir_all(&self.dir)
    }
}

#[cfg(unix)]
type LockGuard = nix::fcntl::Flock<File>;

#[cfg(unix)]
fn try_lock(file: File) -> std::io::Result<Option<LockGuard>> {
    use nix::fcntl::{Flock, FlockArg};
    match Flock::lock(file, FlockArg::LockExclusiveNonblock) {
        Ok(guard) => Ok(Some(guard)),
        Err((_, nix::errno::Errno::EWOULDBLOCK)) => Ok(None),
        Err((_, errno)) => Err(std::io::Error::from(errno)),
    }
}

#[cfg(windows)]
#[derive(Debug)]
// Never read: holding the file holds the lock, and dropping it unlocks.
struct LockGuard(#[allow(dead_code)] File);

#[cfg(windows)]
fn try_lock(file: File) -> std::io::Result<Option<LockGuard>> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::ERROR_LOCK_VIOLATION;
    use windows_sys::Win32::Storage::FileSystem::{
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;
    let mut overlapped = OVERLAPPED::default();
    // SAFETY: the handle is live for the call (owned by `file`) and
    // `overlapped` outlives this synchronous call.
    #[allow(unsafe_code)]
    let ok = unsafe {
        LockFileEx(
            file.as_raw_handle() as _,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            &mut overlapped as *mut OVERLAPPED,
        )
    } != 0;
    if ok {
        return Ok(Some(LockGuard(file)));
    }
    let err = std::io::Error::last_os_error();
    if err.raw_os_error() == Some(ERROR_LOCK_VIOLATION as i32) {
        Ok(None)
    } else {
        Err(err)
    }
}

// Closing the handle releases a Windows byte-range lock, so dropping the
// guard's `File` is the unlock.

#[cfg(not(any(unix, windows)))]
#[derive(Debug)]
struct LockGuard(#[allow(dead_code)] File);

#[cfg(not(any(unix, windows)))]
fn try_lock(file: File) -> std::io::Result<Option<LockGuard>> {
    // No advisory lock on this platform: the marker check still refuses a
    // live trainer, but two BLUT processes are not kept apart.
    Ok(Some(LockGuard(file)))
}

/// Whether `pid` exists on this host. `None` when it cannot be told.
pub(crate) fn pid_alive(pid: u32) -> Option<bool> {
    #[cfg(unix)]
    {
        use crate::python_kill::PidStatus;
        Some(!matches!(
            crate::python_kill::pid_alive(pid),
            PidStatus::Gone
        ))
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        None
    }
}

/// This host's name as `gethostname` reports it. A trainer writes the same
/// value into its marker's `host`, which is how the engine knows it may probe
/// the marker's PID.
pub fn hostname() -> Option<String> {
    #[cfg(unix)]
    {
        if let Ok(h) = nix::unistd::gethostname() {
            let h = h.to_string_lossy().trim().to_string();
            if !h.is_empty() {
                return Some(h);
            }
        }
    }
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .filter(|h| !h.trim().is_empty())
}

pub(crate) fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let body = std::fs::read(path).ok()?;
    match serde_json::from_slice(&body) {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!(
                "checkpoint slot: {} is unparseable ({e}); treating it as absent",
                path.display()
            );
            None
        }
    }
}

fn write_atomic(path: &Path, body: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(body)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Anything outside `[A-Za-z0-9_-]` becomes `_`, so a stage name cannot add a
/// path separator or `..`.
fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;

    fn owner(invocation: &str, attempt: u32) -> SlotOwner {
        SlotOwner {
            run_id: "job-1".into(),
            attempt_id: format!("job-1/{invocation}/{attempt}"),
            invocation: invocation.into(),
            host: Some("h0".into()),
            pid: 10,
            slurm_job_id: None,
            slurm_restart_count: None,
            started_unix: 1000,
        }
    }

    fn marker(step: Option<u64>, hb: u64, attempt_id: &str) -> SlotMarker {
        SlotMarker {
            status: "running".into(),
            run_id: "job-1".into(),
            pid: 4242,
            host: Some("h0".into()),
            heartbeat_unix: hb,
            step,
            epoch: Some(1),
            attempt_id: Some(attempt_id.into()),
            resumed_from_step: None,
        }
    }

    const STALE: u64 = 180;
    fn alive(_: u32) -> Option<bool> {
        Some(true)
    }
    fn dead(_: u32) -> Option<bool> {
        Some(false)
    }

    #[test]
    fn no_marker_or_finished_starts_fresh() {
        let me = owner("inv-a", 1);
        assert_eq!(
            decide_slot(None, None, &me, 2000, STALE, &alive),
            SlotDecision::Fresh
        );
        let mut m = marker(Some(40), 1999, "x");
        m.status = "finished".into();
        assert_eq!(
            decide_slot(Some(&m), None, &me, 2000, STALE, &alive),
            SlotDecision::Fresh
        );
    }

    #[test]
    fn in_process_retry_resumes_even_with_a_fresh_heartbeat() {
        let prev = owner("inv-a", 1);
        let me = owner("inv-a", 2);
        let m = marker(Some(40), 1999, &prev.attempt_id);
        assert_eq!(
            decide_slot(Some(&m), Some(&prev), &me, 2000, STALE, &alive),
            SlotDecision::Resume { step: 40 }
        );
    }

    #[test]
    fn a_live_trainer_from_another_invocation_is_busy() {
        // `plan resume` after the BLUT process was killed while its trainer
        // kept running: same host, PID alive, heartbeat fresh.
        let prev = owner("inv-a", 1);
        let me = owner("inv-b", 1);
        let m = marker(Some(40), 1990, &prev.attempt_id);
        let d = decide_slot(Some(&m), Some(&prev), &me, 2000, STALE, &alive);
        assert!(
            matches!(&d, SlotDecision::Busy { reason } if reason.contains("pid 4242")),
            "{d:?}"
        );
    }

    #[test]
    fn a_dead_trainer_on_this_host_resumes_without_waiting() {
        let prev = owner("inv-a", 1);
        let me = owner("inv-b", 1);
        let m = marker(Some(40), 1990, &prev.attempt_id);
        assert_eq!(
            decide_slot(Some(&m), Some(&prev), &me, 2000, STALE, &dead),
            SlotDecision::Resume { step: 40 }
        );
        // A PID on another host cannot be probed from here.
        let mut other = m.clone();
        other.host = Some("h1".into());
        assert!(matches!(
            decide_slot(Some(&other), Some(&prev), &me, 2000, STALE, &dead),
            SlotDecision::Busy { .. }
        ));
    }

    #[test]
    fn a_stale_heartbeat_resumes() {
        let prev = owner("inv-a", 1);
        let me = owner("inv-b", 1);
        let mut m = marker(Some(40), 1000, &prev.attempt_id);
        m.host = Some("elsewhere".into());
        assert_eq!(
            decide_slot(Some(&m), Some(&prev), &me, 1000 + STALE, STALE, &alive),
            SlotDecision::Resume { step: 40 }
        );
        assert!(matches!(
            decide_slot(Some(&m), Some(&prev), &me, 1000 + STALE - 1, STALE, &alive),
            SlotDecision::Busy { .. }
        ));
    }

    #[test]
    fn a_slurm_requeue_resumes_even_with_a_fresh_heartbeat_on_another_host() {
        let mut prev = owner("inv-a", 1);
        prev.slurm_job_id = Some("17".into());
        prev.slurm_restart_count = Some(0);
        let mut me = owner("inv-b", 1);
        me.run_id = "job-2".into();
        me.slurm_job_id = Some("17".into());
        me.slurm_restart_count = Some(1);
        let mut m = marker(Some(120), 1995, &prev.attempt_id);
        m.host = Some("n1".into());
        assert_eq!(
            decide_slot(Some(&m), Some(&prev), &me, 2000, STALE, &alive),
            SlotDecision::Resume { step: 120 }
        );
        // A different Slurm job is not a requeue of this one.
        me.slurm_job_id = Some("18".into());
        assert!(matches!(
            decide_slot(Some(&m), Some(&prev), &me, 2000, STALE, &alive),
            SlotDecision::Busy { .. }
        ));
    }

    #[test]
    fn an_abandoned_marker_without_a_checkpoint_starts_fresh() {
        let prev = owner("inv-a", 1);
        let me = owner("inv-a", 2);
        let m = marker(None, 1999, &prev.attempt_id);
        assert_eq!(
            decide_slot(Some(&m), Some(&prev), &me, 2000, STALE, &alive),
            SlotDecision::Fresh
        );
    }

    #[test]
    fn a_marker_from_a_foreign_attempt_is_not_trusted_as_ours() {
        // Same invocation in owner.json, but the marker names another attempt
        // (an orphan writing into the slot): judged on liveness, so busy.
        let prev = owner("inv-a", 1);
        let me = owner("inv-a", 2);
        let m = marker(Some(40), 1999, "job-0/inv-z/1");
        assert!(matches!(
            decide_slot(Some(&m), Some(&prev), &me, 2000, STALE, &alive),
            SlotDecision::Busy { .. }
        ));
    }

    #[test]
    fn a_marker_from_an_older_trainer_still_parses() {
        // The durable-resume marker shape (no host, no attempt_id).
        let old =
            r#"{"status":"running","run_id":"r","pid":7,"heartbeat_unix":5,"step":3,"extra":1}"#;
        let m: SlotMarker = serde_json::from_str(old).unwrap();
        assert_eq!(m.step, Some(3));
        assert_eq!(m.host, None);
    }

    #[test]
    fn slot_key_changes_with_the_cache_key_and_the_tenant() {
        let root = Path::new("/r");
        let t = crate::tenant::Tenant::default();
        let k1 = InvocationKey::from_digest(ContentHash::of_bytes(b"args-1"));
        let k2 = InvocationKey::from_digest(ContentHash::of_bytes(b"args-2"));
        let a = slot_dir(root, &t, "train_model", &k1);
        assert_eq!(a, slot_dir(root, &t, "train_model", &k1));
        assert_ne!(a, slot_dir(root, &t, "train_model", &k2));
        let other = crate::tenant::Tenant::parse("lab/vision").unwrap();
        assert_ne!(a, slot_dir(root, &other, "train_model", &k1));
        assert!(a.starts_with(root));
        let hostile = slot_dir(root, &t, "../../etc", &k1);
        assert!(hostile.starts_with(root.join(t.as_path())));
        assert!(!hostile.to_string_lossy().contains(".."));
    }

    #[test]
    fn root_is_a_sibling_of_the_jobs_dir() {
        // Read with the override unset in this process.
        if std::env::var_os(ENV_CHECKPOINTS_ROOT).is_some() {
            return;
        }
        assert_eq!(
            checkpoints_root(Path::new("/data/train-jobs/20261023-1")),
            PathBuf::from("/data/train-jobs-checkpoints")
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_held_slot_is_refused_and_released_on_drop() {
        let td = tempfile::tempdir().unwrap();
        let dir = td.path().join("slot");
        let lease = SlotLease::try_acquire(&dir).unwrap().expect("first lease");
        assert!(SlotLease::try_acquire(&dir).unwrap().is_none(), "second");
        drop(lease);
        assert!(
            SlotLease::try_acquire(&dir).unwrap().is_some(),
            "after drop"
        );
    }

    #[test]
    fn clear_keeps_the_lock_and_remove_deletes_the_slot() {
        let td = tempfile::tempdir().unwrap();
        let dir = td.path().join("slot");
        let lease = SlotLease::try_acquire(&dir).unwrap().unwrap();
        std::fs::write(dir.join("step-00000010.pt"), b"x").unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        lease.clear().unwrap();
        let left: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left, vec![std::ffi::OsString::from(LOCK_FILE)]);
        lease.remove().unwrap();
        assert!(!dir.exists());
    }

    #[test]
    fn owner_round_trips_atomically() {
        let td = tempfile::tempdir().unwrap();
        let o = owner("inv-a", 3);
        o.write(td.path()).unwrap();
        assert_eq!(SlotOwner::read(td.path()), Some(o));
        let names: Vec<_> = std::fs::read_dir(td.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from(OWNER_FILE)]);
    }

    fn slot_with(root: &Path, name: &str, marker: Option<&SlotMarker>) -> PathBuf {
        let dir = root.join("default").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("step-0000000010.pt"), vec![0u8; 100]).unwrap();
        if let Some(m) = marker {
            std::fs::write(dir.join(MARKER_FILE), serde_json::to_vec(m).unwrap()).unwrap();
        }
        dir
    }

    fn collect(
        root: &Path,
        max_age: u64,
        now: u64,
        alive: &dyn Fn(u32) -> Option<bool>,
    ) -> SlotCollection {
        collect_slots_with(root, max_age, now, STALE, Some("h0"), alive).unwrap()
    }

    fn file_age_now() -> u64 {
        now_unix()
    }

    #[test]
    fn an_old_abandoned_slot_is_collected_and_a_recent_one_kept() {
        let td = tempfile::tempdir().unwrap();
        // A crashed trainer: dead pid, old heartbeat.
        let mut m = marker(Some(10), 1000, "a");
        m.pid = 4242;
        let old = slot_with(td.path(), "train-old", Some(&m));
        let now = file_age_now();
        // The files were just written, so with a 1-day bound nothing is old yet.
        let kept = collect(td.path(), 86_400, now, &dead);
        assert_eq!(kept.kept_recent, vec![old.clone()]);
        assert!(old.exists());
        // A day later, it goes.
        let gone = collect(td.path(), 86_400, now + 86_400 + 1, &dead);
        assert_eq!(gone.removed, vec![old.clone()]);
        assert!(gone.bytes_freed >= 100);
        assert!(!old.exists());
    }

    #[test]
    fn a_slot_whose_trainer_may_be_alive_is_never_collected() {
        let td = tempfile::tempdir().unwrap();
        let now = file_age_now() + 10 * 86_400;
        // Fresh heartbeat on another host.
        let mut fresh = marker(Some(10), now - 5, "a");
        fresh.host = Some("elsewhere".into());
        let live = slot_with(td.path(), "train-live", Some(&fresh));
        // Same host, PID alive, heartbeat fresh.
        let local = slot_with(
            td.path(),
            "train-local",
            Some(&marker(Some(10), now - 5, "b")),
        );
        let r = collect(td.path(), 0, now, &alive);
        assert_eq!(r.kept_live, vec![live.clone(), local.clone()]);
        assert!(r.removed.is_empty());
        // The same host with the PID gone: collectable even with a fresh heartbeat.
        let r = collect(td.path(), 0, now, &dead);
        assert_eq!(r.removed, vec![local]);
        assert_eq!(r.kept_live, vec![live]);
    }

    #[cfg(unix)]
    #[test]
    fn a_locked_slot_is_never_collected() {
        let td = tempfile::tempdir().unwrap();
        let dir = slot_with(td.path(), "train-locked", None);
        let lease = SlotLease::try_acquire(&dir).unwrap().unwrap();
        let r = collect(td.path(), 0, file_age_now() + 10 * 86_400, &dead);
        assert_eq!(r.kept_locked, vec![dir.clone()]);
        assert!(dir.join("step-0000000010.pt").exists());
        drop(lease);
        let r = collect(td.path(), 0, file_age_now() + 10 * 86_400, &dead);
        assert_eq!(r.removed, vec![dir]);
    }

    #[test]
    fn collection_finds_slots_under_nested_tenants_and_ignores_other_dirs() {
        let td = tempfile::tempdir().unwrap();
        let nested = td.path().join("lab").join("vision").join("train-x");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join(OWNER_FILE), b"{}").unwrap();
        let unrelated = td.path().join("default").join("not-a-slot");
        std::fs::create_dir_all(&unrelated).unwrap();
        let r = collect(td.path(), 0, file_age_now() + 86_400, &dead);
        assert_eq!(r.removed, vec![nested]);
        assert!(unrelated.exists());
        // A missing root is an empty collection, not an error.
        assert_eq!(
            collect(&td.path().join("absent"), 0, 0, &dead),
            SlotCollection::default()
        );
    }

    #[test]
    fn env_vars_name_the_contract() {
        let mut s = CheckpointSlot::new("/c/slot", true, "job-1", "job-1/i/2");
        s.offered_step = Some(9);
        let env: std::collections::BTreeMap<_, _> = s.env_vars().into_iter().collect();
        assert_eq!(env[ENV_CHECKPOINT_DIR], "/c/slot");
        assert_eq!(env[ENV_RESUME], "1");
        assert_eq!(env[ENV_RUN_ID], "job-1");
        assert_eq!(env[ENV_ATTEMPT_ID], "job-1/i/2");
    }
}
