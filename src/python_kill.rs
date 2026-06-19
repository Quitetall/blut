//! Shared subprocess-kill primitive + process-group plumbing.
//!
//! Two responsibilities live here:
//!
//!   1. **Process-group spawn + kill (KILL-1, KILL-4).** Training
//!      subprocesses background their own grandchildren (PyTorch
//!      DataLoader workers, `torchrun` ranks, the NCCL watchdog).
//!      Signalling only the direct child orphans those grandchildren
//!      — they keep the GPU busy and leak CUDA/IPC semaphores. The
//!      fix: every spawn calls [`pre_exec_setsid`] so the child
//!      becomes a *session + process-group leader* (its `pgid` ==
//!      its own `pid`). Cancel then signals the whole **group** via
//!      `killpg`, escalating SIGTERM→SIGKILL, and `waitpid`-reaps the
//!      direct child so it never lingers as a zombie. Before
//!      signalling we re-validate the target's identity (start-time)
//!      to close the pid-reuse window.
//!
//!   2. **Active-child registry (KILL-2).** The recipe / TUI launch
//!      path runs the executor *in-process*: the Python child is
//!      spawned deep inside a backend stage, but `blut cancel <id>`
//!      runs in a *separate* process and can only reach that child
//!      through the job's `pid` file. Backends therefore publish the
//!      child pid+pgid via [`register_child`] / [`unregister_child`];
//!      when a job id has been bound for the current process
//!      ([`bind_current_job`]) those calls mirror the pgid into the
//!      job's `pid` file so cross-process cancel can find it.
//!
//! Behavioral guarantees of [`graceful_kill_pid`]:
//!   - Returns immediately if the group/process is already gone (ESRCH).
//!   - Returns immediately if unsignalable (EPERM), logging an error.
//!     Don't burn the grace period — SIGKILL would fail the same way.
//!   - Polls every 200 ms during grace, then escalates to SIGKILL.
//!   - `waitpid`-reaps the direct child (when it is our own child)
//!     so no `<defunct>` survives.
//!   - On non-Unix targets this is a best-effort no-op.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A spawned training child's identity. `pid` is the direct child,
/// `pgid` its process group (== pid after `setsid`). `start_time` is
/// the kernel's monotonic start tick from `/proc/<pid>/stat` field 22
/// — together with the pid it forms a reuse-proof identity (KILL-4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChildIdentity {
    pub pid: u32,
    pub pgid: u32,
    /// `starttime` (clock ticks since boot) from `/proc/<pid>/stat`.
    /// `None` if /proc was unreadable at capture time (non-Linux,
    /// or the process exited between spawn and stat).
    pub start_time: Option<u64>,
}

// ── Active-child registry (KILL-2) ──────────────────────────────────

/// The currently-bound job id for *this* blut process. Set by the
/// in-process recipe/TUI launch path before `execute()` so backend
/// spawns can mirror their child pgid into the right job pid file.
static CURRENT_JOB: Mutex<Option<String>> = Mutex::new(None);

/// Live training children of *this* process, keyed by pid. A REGISTRY
/// (not a single slot): the ParallelExecutor can run two subprocess
/// stages at once, and the in-process cancel handler (SIGTERM/ctrl-c)
/// must reach EVERY live group, not just the last one registered.
static ACTIVE_CHILDREN: Mutex<Option<HashMap<u32, ChildIdentity>>> = Mutex::new(None);

fn with_children<R>(f: impl FnOnce(&mut HashMap<u32, ChildIdentity>) -> R) -> R {
    let mut g = ACTIVE_CHILDREN.lock().expect("ACTIVE_CHILDREN poisoned");
    f(g.get_or_insert_with(HashMap::new))
}

/// Bind a job id to this process so subsequent [`register_child`]
/// calls mirror the child pgid into that job's `pid` file. The recipe
/// run path calls this right after creating the job dir.
pub fn bind_current_job(job_id: impl Into<String>) {
    *CURRENT_JOB.lock().expect("CURRENT_JOB poisoned") = Some(job_id.into());
}

/// Clear the bound job id (called when the run completes).
pub fn unbind_current_job() {
    *CURRENT_JOB.lock().expect("CURRENT_JOB poisoned") = None;
}

/// Record a live training child. Backends call this immediately after a
/// successful spawn. When a job is bound, the child's **pgid** (not
/// blut's own pid — KILL-3) is written to the job pid file so a separate
/// `blut cancel <id>` process can `killpg` the whole tree.
pub fn register_child(id: ChildIdentity) {
    with_children(|m| {
        m.insert(id.pid, id);
    });
    if let Some(job_id) = CURRENT_JOB.lock().expect("CURRENT_JOB poisoned").clone() {
        // Mirror the GROUP id, so cross-process cancel kills the
        // whole tree, not just the leader.
        if let Err(e) = crate::jobs::write_pid(&job_id, id.pgid) {
            tracing::warn!("failed to record child pgid {} for {job_id}: {e}", id.pgid);
        }
    }
}

/// Remove a live training child by pid (called on that child's exit).
/// When the LAST child of a bound job exits, also clear that job's pid
/// file so a later `blut cancel <id>` can't `killpg` a now-stale (and
/// possibly reused) pgid. (KILL-4 start-time validation is the deeper
/// guard against reuse; this just keeps the file honest.)
pub fn unregister_child(pid: u32) {
    let now_empty = with_children(|m| {
        m.remove(&pid);
        m.is_empty()
    });
    if now_empty {
        if let Some(job_id) = CURRENT_JOB.lock().expect("CURRENT_JOB poisoned").clone() {
            if let Err(e) = crate::jobs::clear_pid(&job_id) {
                tracing::debug!("clear pid file for {job_id} after last child exit: {e}");
            }
        }
    }
}

/// Snapshot every currently-registered live child. Used by the
/// in-process signal handler (to kill ALL groups) and by tests.
pub fn active_children() -> Vec<ChildIdentity> {
    with_children(|m| m.values().copied().collect())
}

// ── Spawn helper (KILL-1) ───────────────────────────────────────────

/// Capture a child's identity for the active-child registry + the
/// reuse guard. After [`pre_exec_setsid`] the child is its own group
/// leader, so `pgid == pid`.
pub fn capture_identity(pid: u32) -> ChildIdentity {
    ChildIdentity {
        pid,
        pgid: pid,
        start_time: read_start_time(pid),
    }
}

/// `pre_exec` hook that makes the child a session + process-group
/// leader via `setsid(2)`. Call from a `tokio::process::Command` (or
/// `std::process::Command`) builder:
///
/// ```ignore
/// use std::os::unix::process::CommandExt;
/// // SAFETY: setsid is async-signal-safe; no allocation in the closure.
/// unsafe { cmd.pre_exec(blut::python_kill::pre_exec_setsid); }
/// ```
///
/// `setsid` fails with EPERM only if the caller is already a group
/// leader — never true for a freshly-forked child — so this closure
/// cannot realistically fail. We still propagate the errno so a
/// pathological case surfaces as a spawn error rather than a silent
/// non-leader child.
#[cfg(unix)]
pub fn pre_exec_setsid() -> std::io::Result<()> {
    match nix::unistd::setsid() {
        Ok(_) => Ok(()),
        Err(errno) => Err(std::io::Error::from_raw_os_error(errno as i32)),
    }
}

/// Read `starttime` (field 22) from `/proc/<pid>/stat`. Robust to a
/// process name containing spaces/parentheses: the comm field is
/// wrapped in the *last* `)`, so we split there first.
#[cfg(target_os = "linux")]
fn read_start_time(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_comm = stat.rsplit_once(')')?.1;
    // Fields after comm, space-separated: state(0) ppid(1) ... starttime(19).
    // (field 22 overall = index 19 here, since pid+comm are fields 1-2.)
    after_comm.split_whitespace().nth(19)?.parse().ok()
}

#[cfg(not(target_os = "linux"))]
fn read_start_time(_pid: u32) -> Option<u64> {
    None
}

// ── Kill path (KILL-1, KILL-4) ──────────────────────────────────────

/// Kill the process **group** led by `pgid` (== the child's pid after
/// `setsid`): SIGTERM the group, wait up to `grace`, escalate to
/// SIGKILL. Reaps the direct child if it is our own. Marks the job
/// cancelled at the caller. Idempotent (ESRCH → no-op).
///
/// The legacy name `graceful_kill_pid` is kept because every backend +
/// `jobs::cancel_job` call it; the argument is now treated as a group
/// leader. With `pre_exec_setsid` on the spawn path, pid == pgid, so
/// existing callers signal the full tree for free.
#[cfg(unix)]
pub async fn graceful_kill_pid(pgid: u32, grace: Duration) {
    graceful_kill_group(pgid, None, grace).await
}

#[cfg(not(unix))]
pub async fn graceful_kill_pid(_pgid: u32, _grace: Duration) {
    // No-op on non-Unix; cancel is best-effort.
}

/// Group-aware kill with an optional identity guard. If
/// `expected_identity` is provided, the call is a no-op when the
/// live process's start-time no longer matches — defends against the
/// pid-reuse window (KILL-4). Public so the in-process signal handler
/// can pass the captured [`ChildIdentity`].
#[cfg(unix)]
pub async fn graceful_kill_group(
    pgid: u32,
    expected_identity: Option<ChildIdentity>,
    grace: Duration,
) {
    use nix::errno::Errno;
    use nix::sys::signal::{Signal, killpg};
    use nix::unistd::Pid;

    // KILL-4: refuse to signal a recycled pid. If the leader's pid is
    // gone or its start-time drifted, the original tree is already
    // dead; signalling would hit an unrelated process that reused the
    // pid.
    if let Some(expected) = expected_identity {
        match identity_matches(&expected) {
            IdentityCheck::Match => {}
            IdentityCheck::Gone => {
                tracing::debug!("child pid {} already gone; nothing to kill", expected.pid);
                return;
            }
            IdentityCheck::Reused => {
                tracing::warn!(
                    "pid {} was recycled (start-time drift); refusing to signal a \
                     stranger's process group",
                    expected.pid
                );
                return;
            }
        }
    }

    let group = Pid::from_raw(pgid as i32);
    match killpg(group, Signal::SIGTERM) {
        Ok(()) => {}
        Err(Errno::ESRCH) => {
            reap_if_child(expected_identity, pgid);
            return;
        }
        Err(Errno::EPERM) => {
            tracing::error!(
                "process group {} cannot be signalled (EPERM); cancel is a no-op",
                pgid
            );
            return;
        }
        Err(e) => {
            tracing::error!(
                "SIGTERM group {} returned unexpected errno: {}; skipping grace",
                pgid,
                e
            );
            return;
        }
    }

    let deadline = Instant::now() + grace;
    while Instant::now() < deadline {
        match group_status(pgid) {
            GroupStatus::Gone => {
                tracing::debug!("process group {} exited cleanly after SIGTERM", pgid);
                reap_if_child(expected_identity, pgid);
                return;
            }
            GroupStatus::Unsignalable => {
                tracing::error!("process group {} unreachable mid-wait (EPERM)", pgid);
                return;
            }
            GroupStatus::Alive => {}
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    tracing::warn!(
        "process group {} ignored SIGTERM for {:?}, escalating to SIGKILL",
        pgid,
        grace
    );
    let _ = killpg(group, Signal::SIGKILL);
    // Brief settle so the reap below finds the direct child exited.
    tokio::time::sleep(Duration::from_millis(50)).await;
    reap_if_child(expected_identity, pgid);
}

/// Reap the direct child as a zombie if it is *our* child. `waitpid`
/// only works on direct children; for a pid we merely tracked (the
/// cross-process cancel path) the kernel reaps it via the original
/// parent, so a failed waitpid here is expected and ignored.
#[cfg(unix)]
fn reap_if_child(expected: Option<ChildIdentity>, pgid: u32) {
    use nix::sys::wait::{WaitPidFlag, waitpid};
    use nix::unistd::Pid;
    let pid = expected.map(|e| e.pid).unwrap_or(pgid);
    // WNOHANG: never block. If it's not our child, ECHILD → ignore.
    let _ = waitpid(Pid::from_raw(pid as i32), Some(WaitPidFlag::WNOHANG));
}

#[cfg(unix)]
enum IdentityCheck {
    Match,
    Gone,
    Reused,
}

#[cfg(unix)]
fn identity_matches(expected: &ChildIdentity) -> IdentityCheck {
    match pid_alive(expected.pid) {
        PidStatus::Gone => return IdentityCheck::Gone,
        PidStatus::Alive | PidStatus::Unsignalable => {}
    }
    // Process exists. If we captured a start-time, confirm it still
    // matches — a mismatch means the pid was recycled.
    match (expected.start_time, read_start_time(expected.pid)) {
        (Some(want), Some(now)) if want != now => IdentityCheck::Reused,
        _ => IdentityCheck::Match,
    }
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupStatus {
    Alive,
    Gone,
    Unsignalable,
}

/// Probe whether *any* member of the group is still alive via
/// `killpg(pgid, 0)`.
#[cfg(unix)]
fn group_status(pgid: u32) -> GroupStatus {
    use nix::errno::Errno;
    use nix::sys::signal::killpg;
    use nix::unistd::Pid;
    match killpg(Pid::from_raw(pgid as i32), None) {
        Ok(()) => GroupStatus::Alive,
        Err(Errno::ESRCH) => GroupStatus::Gone,
        Err(Errno::EPERM) => GroupStatus::Unsignalable,
        Err(_) => GroupStatus::Alive,
    }
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PidStatus {
    Alive,
    Gone,
    Unsignalable,
}

#[cfg(unix)]
pub(crate) fn pid_alive(pid: u32) -> PidStatus {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;
    match kill(Pid::from_raw(pid as i32), None) {
        Ok(()) => PidStatus::Alive,
        Err(Errno::ESRCH) => PidStatus::Gone,
        Err(Errno::EPERM) => PidStatus::Unsignalable,
        Err(_) => PidStatus::Alive,
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    /// Poll `kill(pid, 0)` until the pid is gone or `timeout` elapses.
    /// Returns true if the pid died within the window.
    fn wait_dead(pid: u32, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if pid_alive(pid) == PidStatus::Gone {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        pid_alive(pid) == PidStatus::Gone
    }

    /// Spawn a real parent→grandchild process tree as a new session
    /// leader. The child backgrounds a grandchild (`sleep 300 &`),
    /// prints the grandchild pid, then **closes its stdout** (so the
    /// reader hits EOF immediately) and `exec`s into its own
    /// `sleep 300`. Both processes live in the same process group
    /// (`pgid == child pid`, courtesy of `setsid`). Returns
    /// `(child_identity, grandchild_pid)`.
    ///
    /// Both the grandchild AND the child must drop the stdout pipe
    /// before the final long sleep, or `read_to_string` blocks for
    /// the whole sleep (the grandchild inherits fd 1; the exec'd
    /// child inherits fd 1). So: grandchild stdout → /dev/null, echo
    /// the pid, `exec 1>&-` to close the child's copy, then exec into
    /// `sleep`. EOF then arrives on the reader immediately.
    fn spawn_tree() -> (ChildIdentity, u32) {
        let mut cmd = Command::new("sh");
        cmd.arg("-c")
            .arg("sleep 300 >/dev/null 2>&1 & echo $! ; exec 1>&- ; exec sleep 300")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        // SAFETY: setsid is async-signal-safe and the closure does no
        // allocation — sound to run between fork and exec.
        #[allow(unsafe_code)]
        unsafe {
            cmd.pre_exec(pre_exec_setsid);
        }
        let mut child = cmd.spawn().expect("spawn sh tree");
        let pid = child.id();
        // Read the single echoed grandchild pid line; EOF arrives as
        // soon as the shell closes fd 1.
        use std::io::Read;
        let mut buf = String::new();
        child
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut buf)
            .expect("read grandchild pid");
        let grandchild: u32 = buf.trim().parse().expect("parse grandchild pid");
        // Detach: we waitpid-reap via graceful_kill, not via child.wait.
        std::mem::forget(child);
        let id = capture_identity(pid);
        // Give the backgrounded grandchild a beat to actually be
        // exec'd into `sleep` before the test probes it.
        std::thread::sleep(Duration::from_millis(50));
        (id, grandchild)
    }

    #[tokio::test]
    async fn killpg_kills_child_and_grandchild() {
        let (id, grandchild) = spawn_tree();
        // Both alive at the start.
        assert_eq!(pid_alive(id.pid), PidStatus::Alive, "child not alive");
        assert_eq!(
            pid_alive(grandchild),
            PidStatus::Alive,
            "grandchild not alive"
        );
        assert_eq!(id.pgid, id.pid, "setsid should make pid == pgid");

        // Kill the GROUP. The old direct-child-only path left the
        // backgrounded grandchild alive — this is the leaked-semaphore
        // failure. killpg reaps the whole tree.
        graceful_kill_group(id.pgid, Some(id), Duration::from_secs(3)).await;

        assert!(
            wait_dead(id.pid, Duration::from_secs(5)),
            "child pid {} survived group kill",
            id.pid
        );
        assert!(
            wait_dead(grandchild, Duration::from_secs(5)),
            "grandchild pid {} survived group kill (orphaned — the bug)",
            grandchild
        );
    }

    #[tokio::test]
    async fn graceful_kill_reaps_no_zombie() {
        let (id, grandchild) = spawn_tree();
        graceful_kill_group(id.pgid, Some(id), Duration::from_secs(3)).await;
        assert!(wait_dead(id.pid, Duration::from_secs(5)));
        // After reap, /proc/<pid>/stat is gone entirely (not "Z").
        // A surviving zombie would still have a readable stat with
        // state 'Z'. Confirm no zombie for the direct child.
        let stat = std::fs::read_to_string(format!("/proc/{}/stat", id.pid));
        match stat {
            Err(_) => {} // reaped + gone — correct.
            Ok(s) => {
                let state = s.rsplit_once(')').map(|(_, r)| r.trim_start());
                assert!(
                    !matches!(state.and_then(|r| r.chars().next()), Some('Z')),
                    "direct child {} left as zombie: {s}",
                    id.pid
                );
            }
        }
        let _ = grandchild;
    }

    #[tokio::test]
    async fn identity_guard_refuses_recycled_pid() {
        // Build a fake identity for a pid that is alive but whose
        // recorded start-time is deliberately wrong (simulating reuse).
        let (id, _grandchild) = spawn_tree();
        let mut forged = id;
        forged.start_time = Some(id.start_time.unwrap_or(0).wrapping_add(999_999));
        // The guard must classify this as Reused and NOT signal.
        assert!(
            matches!(identity_matches(&forged), IdentityCheck::Reused),
            "start-time drift should be flagged as pid reuse"
        );
        // Calling graceful_kill_group with the forged identity must be
        // a no-op: the real tree stays alive.
        graceful_kill_group(forged.pgid, Some(forged), Duration::from_millis(300)).await;
        assert_eq!(
            pid_alive(id.pid),
            PidStatus::Alive,
            "identity guard failed: killed a process despite start-time drift"
        );
        // Clean up the real tree with the correct identity.
        graceful_kill_group(id.pgid, Some(id), Duration::from_secs(3)).await;
        assert!(wait_dead(id.pid, Duration::from_secs(5)));
    }

    #[tokio::test]
    async fn graceful_kill_gone_pid_is_noop() {
        // A pid that never existed (or is long dead): ESRCH → return.
        // Use a very high pid unlikely to be live; if it happens to be
        // live, skip rather than flake.
        let probe = 4_000_000_000u32;
        if pid_alive(probe) != PidStatus::Gone {
            return;
        }
        graceful_kill_group(probe, None, Duration::from_millis(100)).await;
    }

    #[test]
    fn registry_tracks_multiple_children() {
        // KILL-2 registry: two concurrent subprocess stages register
        // distinct pids; the signal handler must see BOTH (the old
        // single-slot would have lost the first). Uses synthetic,
        // unique pids so it doesn't touch real processes; these test
        // pids are far above any other test's so there's no collision.
        let a = ChildIdentity {
            pid: 3_900_000_001,
            pgid: 3_900_000_001,
            start_time: Some(1),
        };
        let b = ChildIdentity {
            pid: 3_900_000_002,
            pgid: 3_900_000_002,
            start_time: Some(2),
        };
        register_child(a);
        register_child(b);
        let live = active_children();
        assert!(
            live.contains(&a) && live.contains(&b),
            "both children registered: {live:?}"
        );

        unregister_child(a.pid);
        let live = active_children();
        assert!(!live.contains(&a), "a removed");
        assert!(live.contains(&b), "b still live");

        unregister_child(b.pid);
        let live = active_children();
        assert!(!live.contains(&a) && !live.contains(&b), "both cleared");
    }
}
