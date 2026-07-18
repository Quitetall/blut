// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Cross-process advisory lock for the GPU.
//!
//! Why this exists: lamu-cli, lamu-mcp, lamu-api, and lamu-train are
//! separate binaries that all want to allocate VRAM on the same
//! card. The in-process `VramScheduler` arbitrates within one
//! binary; this lockfile arbitrates across binaries.
//!
//! Design:
//!
//!   - One file at `~/.local/state/lamu/scheduler.lock`. Holding it
//!     means "I have exclusive access to the GPU; everyone else
//!     should refuse new allocations or wait."
//!   - Created with `OpenOptions::create_new(true)` (`O_EXCL` on
//!     POSIX). Race-free.
//!   - Body is a JSON `LockInfo` with holder name, kind, pid,
//!     timestamp. Future tooling can render `lamu-train jobs` from
//!     this without grep.
//!   - Stale lock detection: before erroring, read the lock and
//!     check whether the recorded pid is still alive. If not,
//!     remove and retry once. Crash recovery without manual cleanup.
//!   - RAII: `ExclusiveLock`'s `Drop` removes the file. Panics or
//!     early returns release automatically. Process kill -9 leaves
//!     a stale lock that the next caller cleans up.
//!
//! What this is NOT:
//!
//!   - Not a kernel-level lock (`flock`). Two processes that both
//!     ignore the file can both load models — this is advisory.
//!     Inference paths are expected to call `check_unlocked()`
//!     before allocation; that's the social contract.
//!   - Not a fairness queue. `await_unlock` polls and races; if
//!     two waiters wake at the same time, whichever calls
//!     `acquire_exclusive` first wins.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError as Error};

const LOCK_FILENAME: &str = "scheduler.lock";

/// What the lock holder is doing. Carried in the lock body so error
/// messages can be specific (`"GPU held by training job ..."` is
/// more actionable than just `"GPU busy"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LockKind {
    /// `lamu-train` has the card for a training run.
    Training,
    /// A long-running inference operation that doesn't tolerate
    /// preemption (rare; reserved for future eviction-sensitive
    /// callers).
    ExclusiveInference,
}

/// On-disk lock body. Versioned via the type itself — schema changes
/// require a `V2` enum branch and back-compat parse.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LockInfo {
    pub holder: String,
    pub kind: LockKind,
    pub pid: u32,
    /// UNIX seconds; SystemTime is renderable in two lines on the
    /// reading side and avoids a chrono dep.
    pub since_unix: u64,
}

impl LockInfo {
    fn current(holder: impl Into<String>, kind: LockKind) -> Self {
        Self {
            holder: holder.into(),
            kind,
            pid: std::process::id(),
            since_unix: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }
}

/// RAII handle to the lock. Drop removes the lockfile. If the file
/// has been removed externally between acquire and drop, the unlink
/// silently fails — by design, since we don't want a panic in Drop.
#[derive(Debug)]
pub struct ExclusiveLock {
    path: PathBuf,
    info: LockInfo,
}

impl ExclusiveLock {
    pub fn info(&self) -> &LockInfo {
        &self.info
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ExclusiveLock {
    fn drop(&mut self) {
        // Defensive: only remove if the file we wrote is still ours.
        // A stale lock cleaner from another process could in theory
        // remove our file and another holder could create one with
        // the same path; unlinking that would be a bug.
        match std::fs::read_to_string(&self.path) {
            Ok(content) => {
                if let Ok(disk) = serde_json::from_str::<LockInfo>(&content) {
                    if disk.pid == self.info.pid && disk.since_unix == self.info.since_unix {
                        let _ = std::fs::remove_file(&self.path);
                    } else {
                        tracing::warn!(
                            "scheduler_lock Drop: file no longer ours (now pid {}); leaving in place",
                            disk.pid
                        );
                    }
                }
            }
            Err(_) => {
                // File already gone — nothing to clean up.
            }
        }
    }
}

/// Resolve the canonical lockfile path. Creates parent directories
/// on first use so callers don't need a setup step.
pub fn lock_path() -> Result<PathBuf> {
    let dir = dirs::data_local_dir()
        .ok_or_else(|| {
            Error::Other("data_local_dir() unavailable; cannot place scheduler lock".into())
        })?
        .join("lamu");
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join(LOCK_FILENAME))
}

/// Try to claim the lock. Errors with `Error::Config` if held by a
/// live process; cleans up + retries once if held by a dead one.
pub fn acquire_exclusive(holder: impl Into<String>, kind: LockKind) -> Result<ExclusiveLock> {
    acquire_exclusive_at(&lock_path()?, holder, kind)
}

/// The PER-DEVICE lock path (Phase G scheduler) — `scheduler-gpu-<idx>.lock`
/// alongside the box-wide `scheduler.lock`, so device `i` and device `j`
/// arbitrate INDEPENDENTLY (a multi-GPU box runs one job PER device).
pub fn per_device_lock_path(device_idx: usize) -> Result<PathBuf> {
    let dir = lock_path()?
        .parent()
        .ok_or_else(|| Error::Other("scheduler lock dir has no parent".into()))?
        .to_path_buf();
    Ok(dir.join(format!("scheduler-gpu-{device_idx}.lock")))
}

/// Claim the lock for ONE device (Phase G scheduler). Independent of the
/// box-wide lock and of other devices' locks — so `capacity` cells can run
/// concurrently, one per device. Same stale-cleanup + RAII-release as
/// [`acquire_exclusive`].
pub fn acquire_exclusive_device(
    device_idx: usize,
    holder: impl Into<String>,
    kind: LockKind,
) -> Result<ExclusiveLock> {
    acquire_exclusive_at(&per_device_lock_path(device_idx)?, holder, kind)
}

/// Path-injectable variant. The default `acquire_exclusive` calls
/// this with the canonical path; tests use it directly with a
/// tempdir-scoped path so parallel test execution doesn't collide
/// on the real lockfile.
pub fn acquire_exclusive_at(
    path: &Path,
    holder: impl Into<String>,
    kind: LockKind,
) -> Result<ExclusiveLock> {
    let holder = holder.into();
    if let Some(existing) = read_lock(path) {
        if pid_alive(existing.pid) {
            return Err(Error::Other(format!(
                "GPU held by '{}' (pid {}, kind {:?}, since unix {}). \
                 Wait or pass --allow-evict.",
                existing.holder, existing.pid, existing.kind, existing.since_unix
            )));
        }
        tracing::warn!(
            "removing stale scheduler lock from pid {} (holder '{}', kind {:?})",
            existing.pid,
            existing.holder,
            existing.kind
        );
        let _ = std::fs::remove_file(path);
    }
    write_lock(path, &holder, kind)
}

/// Non-blocking read-only check. Used by inference paths before
/// VRAM allocation. Returns `Ok(())` when no lock or stale lock,
/// `Err(Error::Config)` when held by a live process.
pub fn check_unlocked() -> Result<()> {
    check_unlocked_at(&lock_path()?)
}

pub fn check_unlocked_at(path: &Path) -> Result<()> {
    if let Some(existing) = read_lock(path)
        && pid_alive(existing.pid)
    {
        return Err(Error::Other(format!(
            "GPU held by '{}' (pid {}, kind {:?}). \
                 Pass --allow-evict to wait.",
            existing.holder, existing.pid, existing.kind
        )));
    }
    // Stale lock; readers don't clean up to avoid racing the
    // legitimate holder of a freshly-created file. Just report
    // unlocked and leave cleanup to the next acquire.
    Ok(())
}

/// Block until the lock is free or `timeout` elapses. Polls every
/// 500 ms — coarse enough to avoid CPU burn, fine enough that a
/// release feels responsive.
pub async fn await_unlock(timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    let path = lock_path()?;
    loop {
        if check_unlocked_at(&path).is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(Error::Other(format!(
                "timed out after {timeout:?} waiting for GPU lock release"
            )));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

fn read_lock(path: &Path) -> Option<LockInfo> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn write_lock(path: &Path, holder: &str, kind: LockKind) -> Result<ExclusiveLock> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let info = LockInfo::current(holder, kind);
    let body = serde_json::to_vec_pretty(&info)
        .map_err(|e| Error::Other(format!("serialize LockInfo: {e}")))?;
    let mut f = match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // Someone raced us between the stale-cleanup and the
            // create. Re-read; if alive, surface the conflict.
            if let Some(existing) = read_lock(path)
                && pid_alive(existing.pid)
            {
                return Err(Error::Other(format!(
                    "GPU lock raced; now held by '{}' (pid {}, kind {:?})",
                    existing.holder, existing.pid, existing.kind
                )));
            }
            return Err(Error::Other(format!(
                "scheduler lock create_new race at {}: {e}",
                path.display()
            )));
        }
        Err(e) => return Err(e.into()),
    };
    f.write_all(&body)?;
    let _ = f.sync_all();
    Ok(ExclusiveLock {
        path: path.to_path_buf(),
        info,
    })
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> bool {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;
    // signal 0 = "is the process alive". Ok = alive + signalable;
    // EPERM = alive but different user (still alive); ESRCH = gone.
    matches!(
        kill(Pid::from_raw(pid as i32), None),
        Ok(()) | Err(Errno::EPERM)
    )
}

#[cfg(not(unix))]
fn pid_alive(_pid: u32) -> bool {
    // No portable cheap check on non-Unix; assume alive so callers
    // err on the side of refusing to clobber an unknown lock.
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_lock() -> (tempfile::TempDir, PathBuf) {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("scheduler.lock");
        (td, p)
    }

    #[test]
    fn acquire_creates_lockfile() {
        let (_td, path) = temp_lock();
        let lock = acquire_exclusive_at(&path, "test", LockKind::Training).expect("acquire");
        assert!(path.exists());
        let info = lock.info();
        assert_eq!(info.holder, "test");
        assert_eq!(info.kind, LockKind::Training);
        assert_eq!(info.pid, std::process::id());
    }

    #[test]
    fn drop_removes_lockfile() {
        let (_td, path) = temp_lock();
        {
            let _lock = acquire_exclusive_at(&path, "x", LockKind::Training).unwrap();
            assert!(path.exists());
        }
        assert!(!path.exists(), "Drop must remove lockfile");
    }

    #[test]
    fn second_acquire_errors_while_first_held() {
        let (_td, path) = temp_lock();
        let _first = acquire_exclusive_at(&path, "first", LockKind::Training).unwrap();
        let err = acquire_exclusive_at(&path, "second", LockKind::Training)
            .expect_err("second acquire must fail");
        let msg = format!("{err}");
        assert!(msg.contains("first"), "msg should name the holder: {msg}");
        assert!(
            msg.contains("--allow-evict"),
            "msg should hint --allow-evict: {msg}"
        );
    }

    #[test]
    fn second_acquire_succeeds_after_first_drops() {
        let (_td, path) = temp_lock();
        {
            let _first = acquire_exclusive_at(&path, "first", LockKind::Training).unwrap();
        }
        let second = acquire_exclusive_at(&path, "second", LockKind::Training)
            .expect("second acquire after drop");
        assert_eq!(second.info().holder, "second");
    }

    #[test]
    fn stale_lock_from_dead_pid_is_cleaned_up() {
        let (_td, path) = temp_lock();
        // Write a lock body referencing a pid that almost certainly
        // doesn't exist. Pid 0 is the scheduler/idle thread on
        // Linux; signal(0, 0) returns ESRCH.
        let stale = LockInfo {
            holder: "ghost".into(),
            kind: LockKind::Training,
            pid: 1, // init — exists; use a synthetic high pid instead
            since_unix: 0,
        };
        // Use pid=0xDEAD_BEEF which is well above any real pid_max.
        let stale = LockInfo {
            pid: 0xDEAD_BEEF,
            ..stale
        };
        std::fs::write(&path, serde_json::to_vec(&stale).unwrap()).unwrap();
        let lock = acquire_exclusive_at(&path, "fresh", LockKind::Training)
            .expect("stale lock must be replaceable");
        assert_eq!(lock.info().holder, "fresh");
    }

    #[test]
    fn check_unlocked_passes_when_no_file() {
        let (_td, path) = temp_lock();
        check_unlocked_at(&path).expect("no lockfile = unlocked");
    }

    #[test]
    fn check_unlocked_passes_when_lock_is_stale() {
        let (_td, path) = temp_lock();
        let stale = LockInfo {
            holder: "ghost".into(),
            kind: LockKind::Training,
            pid: 0xDEAD_BEEF,
            since_unix: 0,
        };
        std::fs::write(&path, serde_json::to_vec(&stale).unwrap()).unwrap();
        check_unlocked_at(&path).expect("stale lock should read as unlocked");
    }

    #[test]
    fn check_unlocked_errors_when_held() {
        let (_td, path) = temp_lock();
        let _held = acquire_exclusive_at(&path, "trainer", LockKind::Training).unwrap();
        let err = check_unlocked_at(&path).expect_err("must error while held");
        assert!(format!("{err}").contains("trainer"));
    }

    #[test]
    fn drop_does_not_remove_other_holders_lock() {
        // Hand-craft a scenario where Drop sees a different lock
        // body on disk (e.g. stale-cleanup race). Drop must NOT
        // remove a lock that isn't ours.
        let (_td, path) = temp_lock();
        let lock = acquire_exclusive_at(&path, "us", LockKind::Training).unwrap();
        // Stomp the file with a different body (different pid +
        // since_unix) — simulates a misbehaving cleaner. Use a pid
        // that's unambiguously not us and unambiguously not a real
        // process (well above pid_max on Linux); wrapping_add could
        // theoretically collide with a real pid in a container with
        // remapped pid namespaces.
        let imposter = LockInfo {
            holder: "imposter".into(),
            kind: LockKind::Training,
            pid: 0xDEAD_BEEF,
            since_unix: lock.info().since_unix.wrapping_add(99),
        };
        std::fs::write(&path, serde_json::to_vec(&imposter).unwrap()).unwrap();
        drop(lock);
        assert!(path.exists(), "Drop must not remove an imposter lock body");
    }

    #[test]
    fn per_device_locks_are_independent() {
        // Phase G scheduler: device 0 and device 1 arbitrate independently
        // (distinct paths), so `capacity` cells run one-per-device; a second
        // acquire on the SAME device still blocks. (Path-injected — no touch
        // of the real lock dir.)
        let td = tempfile::tempdir().unwrap();
        let p0 = td.path().join("scheduler-gpu-0.lock");
        let p1 = td.path().join("scheduler-gpu-1.lock");
        let l0 = acquire_exclusive_at(&p0, "dev0", LockKind::Training).unwrap();
        // Device 1 is free even while device 0 is held.
        let l1 = acquire_exclusive_at(&p1, "dev1", LockKind::Training).unwrap();
        // A second acquire on device 0 (same path) fails.
        assert!(acquire_exclusive_at(&p0, "dev0b", LockKind::Training).is_err());
        drop(l0);
        // Device 0 frees independently of device 1.
        let l0b = acquire_exclusive_at(&p0, "dev0c", LockKind::Training)
            .expect("device 0 free after its holder drops");
        assert_eq!(l0b.info().holder, "dev0c");
        assert_eq!(l1.info().holder, "dev1", "device 1 unaffected throughout");
    }
}
