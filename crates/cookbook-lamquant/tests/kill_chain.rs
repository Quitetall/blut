//! Cancel/kill reliability chain — real subprocess death (KILL-1..4).
//!
//! These tests spawn REAL process trees and assert the WHOLE tree
//! dies (or, for the pid-file contract, that the recorded pid is the
//! child group leader and not blut's own pid). They are the
//! regression gate for the leaked-semaphore failure: the old
//! direct-child-only kill orphaned the backgrounded grandchild, and
//! the old recipe/TUI path recorded blut's own pid (so `blut cancel`
//! SIGTERM'd blut, never the trainer).
//!
//! All tests run only on Unix (the kill path is a no-op elsewhere)
//! and are guarded against CI flakiness with bounded polling +
//! best-effort cleanup on every exit path. The death assertions are
//! never weakened.

#![cfg(unix)]
// intentional: every test holds the process-wide `GLOBAL_LOCK` across
// `graceful_kill_group(...).await` to serialize the process-global job
// binding + `$LAMU_TRAIN_JOBS_DIR` env state across concurrently-scheduled
// tokio tests. The std guard across an await is the deliberate
// serialization mechanism, not a bug.
#![allow(clippy::await_holding_lock)]

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use blut::python_kill::{
    active_child, bind_current_job, capture_identity, clear_active_child, graceful_kill_group,
    set_active_child, unbind_current_job,
};

/// Serialize tests that touch the process-global job binding +
/// `$LAMU_TRAIN_JOBS_DIR`.
static GLOBAL_LOCK: Mutex<()> = Mutex::new(());

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_blut"))
}

/// `kill(pid, 0)` → ESRCH means gone.
fn pid_gone(pid: u32) -> bool {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;
    matches!(kill(Pid::from_raw(pid as i32), None), Err(Errno::ESRCH))
}

fn wait_dead(pid: u32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if pid_gone(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    pid_gone(pid)
}

/// Spawn a real parent→grandchild tree as a new process-group leader
/// via the same `pre_exec(setsid)` the production spawn path uses.
/// Returns `(child_pid, child_pgid, grandchild_pid)`. Both processes
/// are alive in the same group; reader sees EOF immediately because
/// both drop the stdout pipe before sleeping (see python_kill tests).
fn spawn_tree() -> (u32, u32, u32) {
    use std::os::unix::process::CommandExt;
    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg("sleep 300 >/dev/null 2>&1 & echo $! ; exec 1>&- ; exec sleep 300")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    // SAFETY: setsid is async-signal-safe; the hook allocates nothing.
    unsafe {
        cmd.pre_exec(blut::python_kill::pre_exec_setsid);
    }
    let mut child = cmd.spawn().expect("spawn sh tree");
    let child_pid = child.id();
    use std::io::Read;
    let mut buf = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut buf)
        .expect("read grandchild pid");
    let grandchild: u32 = buf.trim().parse().expect("parse grandchild pid");
    // Detached: reaped via graceful_kill_group, not child.wait().
    std::mem::forget(child);
    let id = capture_identity(child_pid);
    std::thread::sleep(Duration::from_millis(50));
    (child_pid, id.pgid, grandchild)
}

/// Best-effort SIGKILL of a whole group so a failed assertion never
/// leaks a 300s sleep tree onto the CI box.
fn cleanup_group(pgid: u32) {
    use nix::sys::signal::{Signal, killpg};
    use nix::unistd::Pid;
    let _ = killpg(Pid::from_raw(pgid as i32), Signal::SIGKILL);
}

/// KILL-1: cancel must kill the FULL process tree, not just the
/// direct child. Spawns parent→grandchild in one group, kills the
/// group, asserts BOTH pids reach ESRCH. Pre-fix (single-pid
/// SIGTERM) the backgrounded grandchild survives → this FAILS.
#[tokio::test]
async fn cancel_kills_process_tree() {
    let _g = GLOBAL_LOCK.lock().unwrap();
    let (child, pgid, grandchild) = spawn_tree();
    assert!(!pid_gone(child), "child should be alive at start");
    assert!(!pid_gone(grandchild), "grandchild should be alive at start");

    let id = capture_identity(child);
    graceful_kill_group(pgid, Some(id), Duration::from_secs(3)).await;

    let child_dead = wait_dead(child, Duration::from_secs(5));
    let gc_dead = wait_dead(grandchild, Duration::from_secs(5));
    if !child_dead || !gc_dead {
        cleanup_group(pgid);
    }
    assert!(child_dead, "child pid {child} survived group cancel");
    assert!(
        gc_dead,
        "grandchild pid {grandchild} survived group cancel (orphaned — the leaked-semaphore bug)"
    );
}

/// KILL-2 / KILL-3: the pid recorded at a recipe/TUI launch must be
/// the spawned child's PROCESS GROUP id, NOT blut's own pid. We bind
/// a job (as the recipe Run path does), spawn a real child, publish
/// its identity (as the backend spawn does), and assert the job's
/// `pid` file holds the child pgid — and emphatically NOT
/// `std::process::id()`.
#[tokio::test]
async fn pid_file_records_child_not_blut() {
    let _g = GLOBAL_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
    // SAFETY: serialized by GLOBAL_LOCK; restored below.
    unsafe {
        std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path());
    }

    let job_id = "kill2-pidfile-test";
    let blut_pid = std::process::id();
    let (child, pgid, grandchild) = spawn_tree();

    // The production sequence: recipe Run binds the job, backend
    // spawn publishes the child identity → pgid mirrored to pid file.
    bind_current_job(job_id);
    set_active_child(capture_identity(child));

    let written = blut::jobs::read_pid(job_id).expect("read pid file");

    // Restore env before assertions so a panic doesn't leak it.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
            None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
        }
    }
    clear_active_child();
    unbind_current_job();
    cleanup_group(pgid);
    let _ = (child, grandchild);

    assert_eq!(
        written,
        Some(pgid),
        "pid file must record the child PROCESS GROUP id"
    );
    assert_ne!(
        written,
        Some(blut_pid),
        "pid file must NOT record blut's own pid (KILL-3 regression)"
    );
}

/// KILL-2 end-to-end via the real CLI: `blut cancel <id>` runs in a
/// SEPARATE process and must kill the live trainer group it finds in
/// the job pid file. We stage a job dir with the child pgid in `pid`,
/// invoke the compiled `blut cancel` binary, and assert the whole
/// tree dies + the binary exits 0.
#[tokio::test]
async fn blut_cancel_cli_kills_recorded_group() {
    let _g = GLOBAL_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let job_id = "kill2-cli-cancel";

    let (child, pgid, grandchild) = spawn_tree();

    // Stage the job dir exactly as the recipe path leaves it: a
    // `state` file (so resolve_job_id finds it) + `pid` holding the
    // child group leader.
    let job_dir = td.path().join(job_id);
    std::fs::create_dir_all(&job_dir).unwrap();
    std::fs::write(job_dir.join("state"), "running").unwrap();
    std::fs::write(job_dir.join("pid"), pgid.to_string()).unwrap();

    let out = Command::new(binary())
        .env("LAMU_TRAIN_JOBS_DIR", td.path())
        .arg("cancel")
        .arg(job_id)
        .arg("--grace")
        .arg("3s")
        .stdin(Stdio::null())
        .output()
        .expect("spawn blut cancel");

    // `blut cancel` runs in a SEPARATE process, so its killpg signals
    // our tree but cannot waitpid-reap our child (not its child). We
    // (the real parent) reap the now-killed child so `pid_gone`
    // reports ESRCH rather than seeing a lingering zombie. The
    // grandchild is reparented to init and reaped there.
    let reap = || {
        use nix::sys::wait::{WaitPidFlag, waitpid};
        use nix::unistd::Pid;
        let _ = waitpid(Pid::from_raw(child as i32), Some(WaitPidFlag::WNOHANG));
    };
    let child_dead = {
        let deadline = Instant::now() + Duration::from_secs(6);
        loop {
            reap();
            if pid_gone(child) {
                break true;
            }
            if Instant::now() >= deadline {
                break pid_gone(child);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    let gc_dead = wait_dead(grandchild, Duration::from_secs(6));
    if !child_dead || !gc_dead {
        cleanup_group(pgid);
    }

    assert!(
        out.status.success(),
        "blut cancel must exit 0; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(child_dead, "blut cancel left child {child} alive");
    assert!(
        gc_dead,
        "blut cancel left grandchild {grandchild} alive (no process-group kill)"
    );
}

/// `cancel_reaps_no_zombie`: after the group kill, the direct child
/// must be reaped (no `<defunct>`). We confirm via `/proc/<pid>/stat`
/// state never sticks at 'Z'.
#[tokio::test]
async fn cancel_reaps_no_zombie() {
    let _g = GLOBAL_LOCK.lock().unwrap();
    let (child, pgid, grandchild) = spawn_tree();
    let id = capture_identity(child);
    graceful_kill_group(pgid, Some(id), Duration::from_secs(3)).await;
    let dead = wait_dead(child, Duration::from_secs(5));
    if !dead {
        cleanup_group(pgid);
    }
    assert!(dead, "child {child} not dead");

    // Reaped → /proc entry gone, or at worst not stuck as zombie.
    match std::fs::read_to_string(format!("/proc/{child}/stat")) {
        Err(_) => {} // gone — reaped.
        Ok(s) => {
            let state = s
                .rsplit_once(')')
                .map(|(_, r)| r.trim_start())
                .and_then(|r| r.chars().next());
            assert_ne!(state, Some('Z'), "direct child {child} left as zombie: {s}");
        }
    }
    cleanup_group(pgid);
    let _ = grandchild;
}

/// Smoke-check the active-child registry plumbing in isolation: set →
/// observe → clear. Confirms the in-process signal handler can find
/// the live child to killpg (KILL-3).
#[test]
fn active_child_registry_roundtrip() {
    let _g = GLOBAL_LOCK.lock().unwrap();
    clear_active_child();
    assert!(active_child().is_none());
    let id = capture_identity(std::process::id());
    set_active_child(id);
    assert_eq!(active_child(), Some(id));
    clear_active_child();
    assert!(active_child().is_none());
}
