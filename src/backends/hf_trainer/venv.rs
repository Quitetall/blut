//! Auto-managed HF Trainer venv.
//!
//! BLUT's first-class HF Trainer backend ships with its own
//! Python environment so users don't have to chase
//! `transformers` / `trl` / `peft` / `bitsandbytes` version drift.
//! The venv lives at `~/.local/share/blut/hf-venv/` (override with
//! `BLUT_HF_VENV`) and is provisioned on first use via
//! `python -m venv` + `pip install`.
//!
//! Reentrancy: `ensure_venv()` is safe to call concurrently — if
//! the marker file exists with a matching version stamp, all
//! callers short-circuit. The first caller takes a sentinel lock
//! file; concurrent callers wait via filesystem polling.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// On-disk version stamp. Bump when the pinned dep set changes;
/// `ensure_venv()` will rebuild rather than reuse an older stamp.
pub const VENV_VERSION: &str = "1";

/// Pinned dep specs. Bumping requires VENV_VERSION bump so stale
/// venvs are rebuilt. The pins are conservative — known-good as
/// of the BB-4 ship date.
pub const REQUIRED_PKGS: &[&str] = &[
    "transformers>=4.45,<5",
    "datasets>=2.20,<3",
    "accelerate>=0.34,<2",
    "peft>=0.13,<1",
    "bitsandbytes>=0.43,<1",
    "trl>=0.11,<1",
    "torch>=2.4,<3",
    "safetensors>=0.4,<1",
];

/// Resolve the venv root. `$BLUT_HF_VENV` env wins; default
/// `~/.local/share/blut/hf-venv/`.
pub fn venv_root() -> PathBuf {
    if let Ok(p) = std::env::var("BLUT_HF_VENV") {
        return PathBuf::from(p);
    }
    let home = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    home.join("blut").join("hf-venv")
}

/// Path to the venv's Python interpreter. Caller-side; resolved
/// AFTER `ensure_venv()` returns.
pub fn python_path(venv_root: &Path) -> PathBuf {
    venv_root.join("bin").join("python")
}

/// Marker file storing `VENV_VERSION` after a successful build.
fn marker_path(venv_root: &Path) -> PathBuf {
    venv_root.join(".blut_hf_venv_version")
}

/// Mutex file held during provisioning. Concurrent callers see
/// this and poll the marker instead of racing pip.
fn lock_path(venv_root: &Path) -> PathBuf {
    venv_root.join(".blut_hf_venv_lock")
}

#[derive(Debug, thiserror::Error)]
pub enum VenvError {
    #[error("python3 not on PATH; can't bootstrap venv at {venv_root}")]
    PythonMissing { venv_root: PathBuf },
    #[error("`python -m venv` failed at {venv_root}: {status}")]
    VenvCreateFailed { venv_root: PathBuf, status: String },
    #[error("pip install failed: {status}")]
    PipFailed { status: String },
    #[error("io at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("timed out waiting for concurrent venv provisioning ({path})")]
    LockTimeout { path: PathBuf },
}

/// Ensure the HF venv exists with the current pinned dep set.
/// Returns the absolute path to the venv's Python interpreter.
///
/// Behavior:
///   1. Marker exists + matches `VENV_VERSION` → return python_path.
///   2. Lock file exists → poll for marker (up to 30 min for
///      first-time install — pip download of torch is slow).
///   3. Neither → take lock, run `python -m venv`, `pip install`,
///      write marker, release lock.
pub fn ensure_venv() -> Result<PathBuf, VenvError> {
    let root = venv_root();
    let marker = marker_path(&root);
    let py = python_path(&root);

    // Fast path: marker says we're good.
    if marker.exists() && py.exists() {
        if let Ok(s) = std::fs::read_to_string(&marker) {
            if s.trim() == VENV_VERSION {
                return Ok(py);
            }
        }
    }

    // Take the lock atomically. `create_new(true)` returns
    // AlreadyExists if another process beat us to it — at which
    // point we wait for the marker (with stale-lock recovery on
    // top of plain timeout). `create_new` is symlink-safe: the
    // syscall fails rather than following a pre-existing symlink
    // that would otherwise let `std::fs::write` overwrite an
    // attacker-targeted file.
    let lock = lock_path(&root);
    if let Some(parent) = lock.parent() {
        std::fs::create_dir_all(parent).map_err(|source| VenvError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let acquired_lock = match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&lock)
    {
        Ok(mut f) => {
            use std::io::Write;
            let _ = write!(f, "{}", std::process::id());
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => false,
        Err(source) => {
            return Err(VenvError::Io {
                path: lock.clone(),
                source,
            });
        }
    };
    if !acquired_lock {
        // Someone else is provisioning. If their PID is dead,
        // break the lock (recover from a crashed prior caller)
        // and retry once. Otherwise poll for the marker.
        if pid_in_lock_is_dead(&lock) {
            tracing::warn!(
                target: "blut::hf_venv",
                "stale lock at {} (holder PID dead); breaking and retrying",
                lock.display()
            );
            let _ = std::fs::remove_file(&lock);
            return ensure_venv();
        }
        wait_for_marker(&marker, &py, Duration::from_secs(30 * 60))?;
        return Ok(py);
    }

    let result = (|| -> Result<(), VenvError> {
        provision(&root)?;
        std::fs::write(&marker, VENV_VERSION).map_err(|source| VenvError::Io {
            path: marker.clone(),
            source,
        })?;
        Ok(())
    })();

    // Release lock regardless of outcome.
    let _ = std::fs::remove_file(&lock);
    result?;
    Ok(py)
}

/// Read the PID stored inside the lock file and check whether
/// that process is still alive. Returns true if the lock can be
/// reasonably reclaimed (file missing, PID unreadable, or process
/// gone). On non-Unix targets always returns false — we don't
/// have a portable cross-platform pid-alive check, so we wait
/// the full timeout instead of breaking locks blind.
#[cfg(unix)]
fn pid_in_lock_is_dead(lock: &Path) -> bool {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;
    let body = match std::fs::read_to_string(lock) {
        Ok(b) => b,
        Err(_) => return true,
    };
    let pid: i32 = match body.trim().parse() {
        Ok(v) => v,
        Err(_) => return true,
    };
    // Signal 0: probe without delivering. Ok = alive,
    // ESRCH = gone, EPERM = alive-but-not-ours.
    match kill(Pid::from_raw(pid), None) {
        Ok(()) => false,
        Err(Errno::ESRCH) => true,
        Err(Errno::EPERM) => false,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn pid_in_lock_is_dead(_lock: &Path) -> bool {
    false
}

fn wait_for_marker(
    marker: &Path,
    py: &Path,
    timeout: Duration,
) -> Result<(), VenvError> {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if marker.exists() && py.exists() {
            if let Ok(s) = std::fs::read_to_string(marker) {
                if s.trim() == VENV_VERSION {
                    return Ok(());
                }
            }
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    Err(VenvError::LockTimeout {
        path: marker.to_path_buf(),
    })
}

fn provision(root: &Path) -> Result<(), VenvError> {
    tracing::info!(target: "blut::hf_venv", "provisioning HF venv at {}", root.display());

    let python3 = which::which("python3").map_err(|_| VenvError::PythonMissing {
        venv_root: root.to_path_buf(),
    })?;

    let venv_status = Command::new(&python3)
        .arg("-m")
        .arg("venv")
        .arg(root)
        .status()
        .map_err(|source| VenvError::Io {
            path: root.to_path_buf(),
            source,
        })?;
    if !venv_status.success() {
        return Err(VenvError::VenvCreateFailed {
            venv_root: root.to_path_buf(),
            status: format!("{venv_status}"),
        });
    }

    let pip = root.join("bin").join("pip");
    let mut cmd = Command::new(&pip);
    cmd.arg("install").arg("--upgrade").arg("pip");
    let status = cmd.status().map_err(|source| VenvError::Io {
        path: pip.clone(),
        source,
    })?;
    if !status.success() {
        return Err(VenvError::PipFailed {
            status: format!("{status}"),
        });
    }

    let mut cmd = Command::new(&pip);
    cmd.arg("install");
    for pkg in REQUIRED_PKGS {
        cmd.arg(pkg);
    }
    let status = cmd.status().map_err(|source| VenvError::Io {
        path: pip.clone(),
        source,
    })?;
    if !status.success() {
        return Err(VenvError::PipFailed {
            status: format!("{status}"),
        });
    }

    tracing::info!(target: "blut::hf_venv", "HF venv ready at {}", root.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn venv_root_honors_env_override() {
        // SAFETY: tests are serialized via TEST_ENV_LOCK at the
        // crate root for env-mutating tests. This single-shot test
        // grabs + releases without checking other state.
        let prev = std::env::var("BLUT_HF_VENV").ok();
        // SAFETY: see #[allow] in lib.rs for env-test serialization.
        unsafe {
            std::env::set_var("BLUT_HF_VENV", "/tmp/blut-hf-venv-test");
        }
        assert_eq!(venv_root(), PathBuf::from("/tmp/blut-hf-venv-test"));
        match prev {
            Some(v) => unsafe { std::env::set_var("BLUT_HF_VENV", v) },
            None => unsafe { std::env::remove_var("BLUT_HF_VENV") },
        }
    }

    #[test]
    fn default_path_is_under_data_local_dir() {
        // Make sure we're not pointing at /tmp by default.
        let _prev = std::env::var("BLUT_HF_VENV").ok();
        unsafe {
            std::env::remove_var("BLUT_HF_VENV");
        }
        let root = venv_root();
        assert!(root.ends_with("blut/hf-venv") || root.ends_with("blut\\hf-venv"));
    }

    #[test]
    fn python_path_layout_matches_unix_venv() {
        let root = PathBuf::from("/tmp/x");
        assert_eq!(python_path(&root), PathBuf::from("/tmp/x/bin/python"));
    }
}
