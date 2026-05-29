//! `LamquantBackend` — generic argparse-shaped Python subprocess
//! runner. Companion to `PythonTrainBackend` (which expects the
//! `trainer.py` wire format: TrainSpec JSON in, StatusUpdate lines
//! out). LamQuant kernels (`train_joint.py`, `train_mamba_snn.py`,
//! `train_teacher.py`, etc.) are argparse-driven, use tqdm + wandb
//! + `RunManifest` for their own observability, and write
//! checkpoints to known paths.
//!
//! Behavior contract:
//!
//!   1. Spawn `python <script> <args...>` with the supplied `cwd`
//!      (LamQuant repo root, so `RunManifest` resolves the git_sha
//!      + dataset_manifests correctly).
//!   2. Stream stdout → `tracing::info!` AND parse tqdm progress
//!      lines into `StageEvent::StageStep` via the optional
//!      `progress_tx` channel.
//!   3. Stream stderr → `tracing::warn!`.
//!   4. Wait for exit. Nonzero status = `BackendError::Failed`.
//!   5. Verify each path in `expected_outputs` exists on disk.
//!   6. Return a `RunArtifact { run_manifest_path, expected_outputs,
//!      elapsed }`. Stage caller hashes each output path itself
//!      (Artifact::HASH_CONTENTS decides bytes-vs-stat).
//!
//! Cancellation: stage sets the executor `CancellationToken`; the
//! backend's `cancel()` issues `graceful_kill_pid` (SIGTERM, 10s,
//! SIGKILL) from `python_kill`.
//!
//! Not a `TrainBackend` impl — `TrainBackend::run` takes a typed
//! `TrainSpec` shaped for the LLM trainer. LamquantBackend's
//! invocation is bespoke; impl-as-struct is cleaner than forcing
//! a one-size-fits-all trait.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::python_kill::graceful_kill_pid;

/// Caller-supplied invocation spec. Owned by value so the backend
/// can `cmd.arg(...)` without lifetime gymnastics.
#[derive(Clone, Debug)]
pub struct LamquantInvocation {
    pub python: PathBuf,
    pub script: PathBuf,
    pub cwd: PathBuf,
    pub args: Vec<String>,
    /// Extra env vars layered onto the inherited environment.
    /// `BLUT_JOB_ID`, `BLUT_STAGE_NAME`, etc. land here for the
    /// `RunManifest` subprocess pre-hook to read.
    pub env: Vec<(String, String)>,
    /// Paths the stage promises will exist post-run. Verified by
    /// the backend; missing = `BackendError::MissingOutput`.
    pub expected_outputs: Vec<PathBuf>,
    /// Path the trainer is expected to write its `RunManifest` JSON
    /// to. Verified to exist; embedded by the stage into its
    /// sidecar metadata. None when the kernel doesn't write one
    /// (e.g. a precompute step).
    pub run_manifest_path: Option<PathBuf>,
}

/// Returned to the calling stage on success.
#[derive(Debug)]
pub struct LamquantRunArtifact {
    pub expected_outputs: Vec<PathBuf>,
    pub run_manifest_path: Option<PathBuf>,
    pub elapsed: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("spawn {python}: {source}")]
    Spawn {
        python: String,
        #[source]
        source: std::io::Error,
    },
    #[error("subprocess wait failed: {0}")]
    Wait(#[source] std::io::Error),
    #[error("subprocess exited with status {status} (nonzero)")]
    Failed { status: String },
    #[error("expected output missing after success: {path}")]
    MissingOutput { path: PathBuf },
}

pub struct LamquantBackend {
    child_pid: Arc<Mutex<Option<u32>>>,
}

impl LamquantBackend {
    pub fn new() -> Self {
        Self {
            child_pid: Arc::new(Mutex::new(None)),
        }
    }

    /// Spawn the subprocess, stream pipes, wait for exit, verify
    /// outputs. Errors mid-flight cancel the subprocess via the
    /// graceful_kill pattern.
    ///
    /// `progress_tx` is an optional unbounded callback fired once
    /// per parsed tqdm progress line. The caller (stage) decides
    /// whether to forward as `StageEvent::StageStep` or to ignore.
    pub async fn run(
        &mut self,
        inv: LamquantInvocation,
        progress_tx: Option<Box<dyn Fn(Progress) + Send + Sync>>,
    ) -> Result<LamquantRunArtifact, BackendError> {
        let started = Instant::now();

        let mut cmd = Command::new(&inv.python);
        cmd.arg(&inv.script);
        for a in &inv.args {
            cmd.arg(a);
        }
        cmd.current_dir(&inv.cwd);
        for (k, v) in &inv.env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // KILL-1: new session/process group — see python_kill::pre_exec_setsid.
        #[cfg(unix)]
        {
            // tokio's Command exposes `pre_exec` inherently.
            // SAFETY: setsid is async-signal-safe and allocates nothing.
            #[allow(unsafe_code)]
            unsafe {
                cmd.pre_exec(crate::python_kill::pre_exec_setsid);
            }
        }

        let mut child = cmd.spawn().map_err(|source| BackendError::Spawn {
            python: inv.python.display().to_string(),
            source,
        })?;
        if let Some(pid) = child.id() {
            *self.child_pid.lock() = Some(pid);
            // KILL-2: publish for in-process + cross-process cancel.
            crate::python_kill::set_active_child(crate::python_kill::capture_identity(pid));
        }

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let progress_tx_arc = progress_tx.map(Arc::new);

        let stdout_handle = if let Some(stdout) = stdout {
            let progress_tx = progress_tx_arc.clone();
            Some(tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if let Some(p) = parse_tqdm_progress(&line) {
                        if let Some(tx) = &progress_tx {
                            tx(p);
                        }
                    }
                    tracing::info!(target: "blut::lamquant_stdout", "{}", line);
                }
            }))
        } else {
            None
        };

        let stderr_handle = if let Some(stderr) = stderr {
            let progress_tx = progress_tx_arc.clone();
            Some(tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    // tqdm writes progress on stderr by default;
                    // parse the same regex there too.
                    if let Some(p) = parse_tqdm_progress(&line) {
                        if let Some(tx) = &progress_tx {
                            tx(p);
                        }
                    }
                    tracing::warn!(target: "blut::lamquant_stderr", "{}", line);
                }
            }))
        } else {
            None
        };

        let exit_status = child.wait().await.map_err(BackendError::Wait)?;
        if let Some(h) = stdout_handle {
            let _ = h.await;
        }
        if let Some(h) = stderr_handle {
            let _ = h.await;
        }
        *self.child_pid.lock() = None;
        crate::python_kill::clear_active_child();

        if !exit_status.success() {
            return Err(BackendError::Failed {
                status: format!("{exit_status}"),
            });
        }

        for p in &inv.expected_outputs {
            if !p.exists() {
                return Err(BackendError::MissingOutput { path: p.clone() });
            }
        }

        Ok(LamquantRunArtifact {
            expected_outputs: inv.expected_outputs,
            run_manifest_path: inv.run_manifest_path,
            elapsed: started.elapsed(),
        })
    }

    /// SIGTERM-then-SIGKILL the running subprocess if any. Safe to
    /// call concurrently; atomic take() ensures the kill sequence
    /// fires once.
    pub async fn cancel(&mut self) {
        let pid = match self.child_pid.lock().take() {
            Some(p) => p,
            None => return,
        };
        graceful_kill_pid(pid, Duration::from_secs(10)).await;
    }
}

impl Default for LamquantBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Progress events parsed off the tqdm stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    pub current: u64,
    pub total: u64,
}

/// Parse a tqdm progress line. tqdm's default format is roughly
/// `42%|████      | 42/100 [00:05<00:07,  8.40it/s]`. The regex
/// is intentionally loose: it just matches `<int>/<int>` followed
/// by whitespace + `[` because the rest of the format varies with
/// bar character + speed unit + tqdm version.
///
/// Returns `None` for non-progress lines so callers can use it as
/// a cheap filter.
fn parse_tqdm_progress(line: &str) -> Option<Progress> {
    // Find a `<int>/<int>` separated by `/` followed by ` [`. Avoid
    // regex dep for one pattern; manual scan.
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Locate a run of ASCII digits.
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == start || i >= bytes.len() || bytes[i] != b'/' {
            i = i.saturating_add(1);
            continue;
        }
        let cur_str = &line[start..i];
        let slash = i + 1;
        let mut j = slash;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j == slash {
            i = slash;
            continue;
        }
        let tot_str = &line[slash..j];
        // Look for whitespace + '[' to confirm it's tqdm-shaped.
        let tail = &line[j..];
        if !tail
            .trim_start_matches(|c: char| c.is_whitespace())
            .starts_with('[')
        {
            i = j;
            continue;
        }
        let current = cur_str.parse::<u64>().ok()?;
        let total = tot_str.parse::<u64>().ok()?;
        return Some(Progress { current, total });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_tqdm_line() {
        let line = "42%|████      | 42/100 [00:05<00:07,  8.40it/s]";
        let p = parse_tqdm_progress(line).unwrap();
        assert_eq!(p, Progress { current: 42, total: 100 });
    }

    #[test]
    fn parses_compact_tqdm_line() {
        let line = "epoch 3/10 [00:01<00:00, 1.5it/s]";
        let p = parse_tqdm_progress(line).unwrap();
        assert_eq!(p, Progress { current: 3, total: 10 });
    }

    #[test]
    fn parses_with_leading_text() {
        let line = "[ 2026-05-11 ] training 7/9 [steps 70/100, loss=0.42]";
        let p = parse_tqdm_progress(line).unwrap();
        // Picks the first <int>/<int> followed by whitespace + '['.
        assert_eq!(p, Progress { current: 7, total: 9 });
    }

    #[test]
    fn returns_none_for_non_progress() {
        assert!(parse_tqdm_progress("loaded 100 examples").is_none());
        assert!(parse_tqdm_progress("loss: 0.42").is_none());
        assert!(parse_tqdm_progress("").is_none());
    }

    #[test]
    fn returns_none_when_no_bracket() {
        // `42/100` alone isn't tqdm-shaped without ` [`.
        assert!(parse_tqdm_progress("42/100").is_none());
        assert!(parse_tqdm_progress("42/100 done").is_none());
    }

    #[tokio::test]
    async fn spawns_and_captures_exit_status() {
        let mut be = LamquantBackend::new();
        let inv = LamquantInvocation {
            python: PathBuf::from("/bin/sh"),
            script: PathBuf::from("-c"),
            cwd: std::env::temp_dir(),
            args: vec!["exit 0".into()],
            env: vec![],
            expected_outputs: vec![],
            run_manifest_path: None,
        };
        let r = be.run(inv, None).await.expect("run");
        assert!(r.expected_outputs.is_empty());
        assert!(r.run_manifest_path.is_none());
    }

    #[tokio::test]
    async fn nonzero_exit_returns_failed() {
        let mut be = LamquantBackend::new();
        let inv = LamquantInvocation {
            python: PathBuf::from("/bin/sh"),
            script: PathBuf::from("-c"),
            cwd: std::env::temp_dir(),
            args: vec!["exit 7".into()],
            env: vec![],
            expected_outputs: vec![],
            run_manifest_path: None,
        };
        let r = be.run(inv, None).await;
        assert!(matches!(r, Err(BackendError::Failed { .. })));
    }

    #[tokio::test]
    async fn missing_output_returns_error() {
        let mut be = LamquantBackend::new();
        let td = tempfile::tempdir().unwrap();
        let inv = LamquantInvocation {
            python: PathBuf::from("/bin/sh"),
            script: PathBuf::from("-c"),
            cwd: std::env::temp_dir(),
            args: vec!["true".into()],
            env: vec![],
            expected_outputs: vec![td.path().join("never-written")],
            run_manifest_path: None,
        };
        let r = be.run(inv, None).await;
        assert!(matches!(r, Err(BackendError::MissingOutput { .. })));
    }

    #[tokio::test]
    async fn forwards_progress_when_subscribed() {
        let mut be = LamquantBackend::new();
        let progress_collected = Arc::new(Mutex::new(Vec::new()));
        let pc = progress_collected.clone();
        let tx = Box::new(move |p: Progress| {
            pc.lock().push(p);
        }) as Box<dyn Fn(Progress) + Send + Sync>;
        let inv = LamquantInvocation {
            python: PathBuf::from("/bin/sh"),
            script: PathBuf::from("-c"),
            cwd: std::env::temp_dir(),
            // Emit a tqdm-shaped line and exit.
            args: vec![
                r#"printf '50%%|####    | 50/100 [00:01<00:01, 50it/s]\n'"#.into(),
            ],
            env: vec![],
            expected_outputs: vec![],
            run_manifest_path: None,
        };
        let _ = be.run(inv, Some(tx)).await.unwrap();
        let got = progress_collected.lock().clone();
        assert_eq!(got, vec![Progress { current: 50, total: 100 }]);
    }
}

/// Resolve `$LAMQUANT_HOME` with a default. Stage Args pass an
/// explicit path; this helper exists for tests + as the env-fallback
/// hint when a stage's `lamquant_home` field is left empty.
pub fn default_lamquant_home() -> PathBuf {
    if let Ok(p) = std::env::var("LAMQUANT_HOME") {
        return PathBuf::from(p);
    }
    let mut p = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    p.push("Desktop");
    p.push("LamQuant");
    p
}

/// Resolve `$LAMQUANT_PYTHON` with a fallback: an `LAMQUANT_PYTHON`
/// env var, then `<lamquant_home>/.venv/bin/python`, then system
/// `python3`. Match the pattern in `paths::resolve_python`.
pub fn resolve_lamquant_python(lamquant_home: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("LAMQUANT_PYTHON") {
        return PathBuf::from(p);
    }
    let venv = lamquant_home.join(".venv").join("bin").join("python");
    if venv.exists() {
        return venv;
    }
    PathBuf::from("python3")
}
