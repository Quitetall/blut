//! `LamquantBackend` — generic argparse-shaped Python subprocess
//! runner. Companion to `PythonTrainBackend` (which expects the
//! `trainer.py` wire format: TrainSpec JSON in, StatusUpdate lines
//! out). LamQuant kernels (`train_joint.py`, `train_mamba_snn.py`,
//! `train_teacher.py`, etc.) are argparse-driven, use tqdm + wandb +
//! `RunManifest` for their own observability, and write checkpoints
//! to known paths.
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
    /// Transient systemd `--user` unit name when the run is contained
    /// (env `BLUT_CONTAINED=1`). `None` for the default bare-spawn
    /// path. Set alongside `child_pid` so `cancel()` can `systemctl
    /// --user stop` the unit mid-run.
    contained_unit: Arc<Mutex<Option<String>>>,
}

impl LamquantBackend {
    pub fn new() -> Self {
        Self {
            child_pid: Arc::new(Mutex::new(None)),
            contained_unit: Arc::new(Mutex::new(None)),
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

        // BLUT_CONTAINED=1 wraps the kernel in a transient,
        // memory-capped systemd `--user` unit (OOM containment +
        // session-SIGTERM survival). Default-off: when unset, the
        // bare-spawn path below is byte-identical to the legacy
        // behaviour. Idiom matches `LAMU_TRAIN_USE_LEGACY`.
        let contained = matches!(std::env::var("BLUT_CONTAINED").as_deref(), Ok("1"));
        let unit = if contained {
            contained_unit_name(&inv.env)
        } else {
            None
        };

        let mut cmd = match &unit {
            // Contained path: cwd + env cross the unit boundary via
            // `--working-directory=` / `--setenv=`, so they must NOT
            // be applied to the systemd-run client process here.
            Some(u) => build_contained_command(&inv, u),
            // Bare-spawn path: identical to the pre-seam behaviour.
            None => {
                let mut c = Command::new(&inv.python);
                c.arg(&inv.script);
                for a in &inv.args {
                    c.arg(a);
                }
                c.current_dir(&inv.cwd);
                for (k, v) in &inv.env {
                    c.env(k, v);
                }
                c
            }
        };

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
        // Publish the unit name (if any) BEFORE awaiting exit so a
        // concurrent cancel() can tear the unit down mid-run.
        *self.contained_unit.lock() = unit;
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
        *self.contained_unit.lock() = None;
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
    ///
    /// When the run is contained (`BLUT_CONTAINED=1`), the captured
    /// `child_pid` is the systemd-run CLIENT pid, not the reparented
    /// python process inside the unit cgroup — so `systemctl --user
    /// stop <unit>` is the real teardown (verified: status=15/TERM,
    /// client unblocks). We stop the unit FIRST, then still fall
    /// through to graceful_kill_pid as belt-and-suspenders.
    pub async fn cancel(&mut self) {
        // Take the unit name out FIRST, dropping the (non-async) guard
        // before any await — never hold a parking_lot lock across .await.
        let unit = self.contained_unit.lock().take();
        if let Some(unit) = unit {
            let _ = Command::new("systemctl")
                .arg("--user")
                .arg("stop")
                .arg(format!("{unit}.service"))
                .status()
                .await;
        }
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

/// Sanitize one string to the systemd unit-name charset
/// (`[A-Za-z0-9_-]`). Any other byte becomes `_`. Empty input maps to
/// `_` so the derived unit name is always non-empty.
fn sanitize_unit_part(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if out.is_empty() {
        "_".into()
    } else {
        out
    }
}

/// Derive the transient unit name from the invocation env.
///
/// Reads `BLUT_JOB_DIR` + `BLUT_STAGE_NAME` (set by `blut_env`). When
/// both are present, returns `Some("blut-<job>-<stage>")` where
/// `<job>` is the sanitized last path component of the job dir —
/// deterministic so `cancel()` can target the same unit. Returns
/// `None` when either key is absent (tests / direct callers), which
/// forces the bare-spawn fallback even if `BLUT_CONTAINED=1`.
fn contained_unit_name(env: &[(String, String)]) -> Option<String> {
    let mut job_dir: Option<&str> = None;
    let mut stage: Option<&str> = None;
    for (k, v) in env {
        match k.as_str() {
            "BLUT_JOB_DIR" => job_dir = Some(v),
            "BLUT_STAGE_NAME" => stage = Some(v),
            _ => {}
        }
    }
    let (job_dir, stage) = (job_dir?, stage?);
    let job_leaf = Path::new(job_dir)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(job_dir);
    Some(format!(
        "blut-{}-{}",
        sanitize_unit_part(job_leaf),
        sanitize_unit_part(stage)
    ))
}

/// Build the `systemd-run --user --pipe --wait` command that runs the
/// kernel inside a transient, memory-capped unit. Mirrors the policy
/// in `tools/run_contained.sh` (the source of truth for the memory
/// knobs) but stays SYNCHRONOUS (`--pipe --wait`) so the runner's
/// stream → await exit → map-nonzero contract is preserved — unlike
/// the script, which is detached (`--collect`, no `--wait`).
///
/// cwd + env cross the unit boundary via `--working-directory=` /
/// `--setenv=` (verified on this box). `PATH` + `HOME` are propagated
/// from the caller because `--user` units otherwise run in the user
/// manager's minimal env; `PYTHONPATH` (in `inv.env`) MUST cross or
/// `lamquant.*` imports fail.
fn build_contained_command(inv: &LamquantInvocation, unit: &str) -> Command {
    // Memory knobs: same env vars + defaults as run_contained.sh.
    let memmax = std::env::var("MEMMAX").unwrap_or_else(|_| "44G".into());
    let memhigh = std::env::var("MEMHIGH").unwrap_or_else(|_| "40G".into());
    let swapmax = std::env::var("SWAPMAX").unwrap_or_else(|_| "12G".into());

    let mut c = Command::new("systemd-run");
    c.arg("--user")
        .arg("--pipe")
        .arg("--wait")
        .arg("--collect")
        .arg(format!("--unit={unit}"))
        .arg("-p")
        .arg("MemoryAccounting=yes")
        .arg("-p")
        .arg(format!("MemoryMax={memmax}"))
        .arg("-p")
        .arg(format!("MemoryHigh={memhigh}"))
        .arg("-p")
        .arg(format!("MemorySwapMax={swapmax}"))
        .arg(format!("--working-directory={}", inv.cwd.display()));
    // Propagate PATH + HOME (minimal --user env otherwise).
    if let Ok(path) = std::env::var("PATH") {
        c.arg(format!("--setenv=PATH={path}"));
    }
    if let Ok(home) = std::env::var("HOME") {
        c.arg(format!("--setenv=HOME={home}"));
    }
    // Cross the invocation env (incl. PYTHONPATH + BLUT_* identity).
    for (k, v) in &inv.env {
        c.arg(format!("--setenv={k}={v}"));
    }
    // Terminator, then the actual kernel command.
    c.arg("--").arg(&inv.python).arg(&inv.script);
    for a in &inv.args {
        c.arg(a);
    }
    c
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_tqdm_line() {
        let line = "42%|████      | 42/100 [00:05<00:07,  8.40it/s]";
        let p = parse_tqdm_progress(line).unwrap();
        assert_eq!(
            p,
            Progress {
                current: 42,
                total: 100
            }
        );
    }

    #[test]
    fn parses_compact_tqdm_line() {
        let line = "epoch 3/10 [00:01<00:00, 1.5it/s]";
        let p = parse_tqdm_progress(line).unwrap();
        assert_eq!(
            p,
            Progress {
                current: 3,
                total: 10
            }
        );
    }

    #[test]
    fn parses_with_leading_text() {
        let line = "[ 2026-05-11 ] training 7/9 [steps 70/100, loss=0.42]";
        let p = parse_tqdm_progress(line).unwrap();
        // Picks the first <int>/<int> followed by whitespace + '['.
        assert_eq!(
            p,
            Progress {
                current: 7,
                total: 9
            }
        );
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
            args: vec![r#"printf '50%%|####    | 50/100 [00:01<00:01, 50it/s]\n'"#.into()],
            env: vec![],
            expected_outputs: vec![],
            run_manifest_path: None,
        };
        let _ = be.run(inv, Some(tx)).await.unwrap();
        let got = progress_collected.lock().clone();
        assert_eq!(
            got,
            vec![Progress {
                current: 50,
                total: 100
            }]
        );
    }

    // ── BLUT_CONTAINED systemd seam (pure-function tests) ─────
    //
    // These assert the DERIVED unit name + the BUILT argv without
    // touching the process-global `BLUT_CONTAINED` env var (which the
    // 9 tests above don't lock) — `contained_unit_name` /
    // `build_contained_command` are pure of that flag; only `run()`
    // reads it. The off-path stays covered by `spawns_and_captures_*`.

    #[test]
    fn contained_unit_name_derives_from_env() {
        let env = vec![
            ("BLUT_JOB_DIR".to_string(), "/var/blut/jobs/job42".to_string()),
            ("BLUT_STAGE_NAME".to_string(), "train_joint".to_string()),
        ];
        assert_eq!(
            contained_unit_name(&env).as_deref(),
            Some("blut-job42-train_joint")
        );
        // Missing either key → None (forces bare-spawn fallback).
        let only_job = vec![("BLUT_JOB_DIR".to_string(), "/x/y".to_string())];
        assert!(contained_unit_name(&only_job).is_none());
        let only_stage = vec![("BLUT_STAGE_NAME".to_string(), "s".to_string())];
        assert!(contained_unit_name(&only_stage).is_none());
        assert!(contained_unit_name(&[]).is_none());
    }

    #[test]
    fn contained_unit_name_sanitizes() {
        let env = vec![
            (
                "BLUT_JOB_DIR".to_string(),
                "/jobs/run #3 (x)".to_string(),
            ),
            ("BLUT_STAGE_NAME".to_string(), "weird/stage:name".to_string()),
        ];
        let unit = contained_unit_name(&env).unwrap();
        // Every char must be in the systemd unit charset + the prefix.
        assert!(
            unit.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
            "unit '{unit}' has non-charset chars"
        );
        assert!(unit.starts_with("blut-"));
    }

    #[test]
    fn build_contained_command_includes_setenv_pythonpath() {
        let inv = LamquantInvocation {
            python: PathBuf::from("/venv/bin/python"),
            script: PathBuf::from("train.py"),
            cwd: PathBuf::from("/work/dir"),
            args: vec!["--config".into(), "fast".into()],
            env: vec![("PYTHONPATH".to_string(), "/x".to_string())],
            expected_outputs: vec![],
            run_manifest_path: None,
        };
        let cmd = build_contained_command(&inv, "blut-job-stage");
        let std_cmd = cmd.as_std();
        assert_eq!(std_cmd.get_program(), "systemd-run");
        let argv: Vec<String> = std_cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        // Synchronous streaming + containment flags.
        assert!(argv.iter().any(|a| a == "--pipe"), "missing --pipe: {argv:?}");
        assert!(argv.iter().any(|a| a == "--wait"), "missing --wait: {argv:?}");
        assert!(
            argv.iter().any(|a| a == "--working-directory=/work/dir"),
            "missing --working-directory: {argv:?}"
        );
        // PYTHONPATH must cross the unit boundary.
        assert!(
            argv.iter().any(|a| a == "--setenv=PYTHONPATH=/x"),
            "missing --setenv=PYTHONPATH=/x: {argv:?}"
        );
        // Trailing `-- <python> <script> <args...>`.
        let dash = argv.iter().position(|a| a == "--").expect("missing --");
        assert_eq!(&argv[dash + 1..], &[
            "/venv/bin/python".to_string(),
            "train.py".to_string(),
            "--config".to_string(),
            "fast".to_string(),
        ]);
    }
}
