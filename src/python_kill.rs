//! Shared subprocess-kill primitive.
//!
//! `graceful_kill_pid(pid, grace)`: SIGTERM, wait up to `grace`,
//! escalate to SIGKILL. Extracted from `python_backend` so the
//! pattern is reusable across every backend that owns a Python
//! subprocess — `PythonTrainBackend` (trainer.py), `LamquantBackend`
//! (LamQuant kernels), and any future remote backend.
//!
//! Behavioral guarantees:
//!   - Returns immediately if `pid` is already gone (ESRCH).
//!   - Returns immediately if the process is unsignalable (EPERM),
//!     logs an error. Don't burn the grace period — SIGKILL would
//!     fail the same way.
//!   - Polls every 200 ms during grace.
//!   - On non-Unix targets this is a no-op (best-effort cancel).

use std::time::{Duration, Instant};

#[cfg(unix)]
pub async fn graceful_kill_pid(pid: u32, grace: Duration) {
    graceful_kill_inner(pid, grace).await
}

#[cfg(not(unix))]
pub async fn graceful_kill_pid(_pid: u32, _grace: Duration) {
    // No-op on non-Unix; cancel is best-effort.
}

#[cfg(unix)]
async fn graceful_kill_inner(pid: u32, grace: Duration) {
    use nix::errno::Errno;
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;
    let raw = Pid::from_raw(pid as i32);
    match kill(raw, Signal::SIGTERM) {
        Ok(()) => {}
        Err(Errno::ESRCH) => return,
        Err(Errno::EPERM) => {
            tracing::error!(
                "subprocess pid {} cannot be signalled (EPERM); cancel is a no-op",
                pid
            );
            return;
        }
        Err(e) => {
            tracing::error!(
                "SIGTERM pid {} returned unexpected errno: {}; skipping grace period",
                pid,
                e
            );
            return;
        }
    }
    let deadline = Instant::now() + grace;
    while Instant::now() < deadline {
        match pid_alive(pid) {
            PidStatus::Gone => {
                tracing::debug!("subprocess pid {} exited cleanly after SIGTERM", pid);
                return;
            }
            PidStatus::Unsignalable => {
                tracing::error!("subprocess pid {} unreachable mid-wait (EPERM)", pid);
                return;
            }
            PidStatus::Alive => {}
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    tracing::warn!(
        "subprocess pid {} ignored SIGTERM for {:?}, escalating to SIGKILL",
        pid,
        grace
    );
    let _ = kill(raw, Signal::SIGKILL);
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PidStatus {
    Alive,
    Gone,
    Unsignalable,
}

#[cfg(unix)]
fn pid_alive(pid: u32) -> PidStatus {
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
