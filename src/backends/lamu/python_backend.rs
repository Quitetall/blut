//! `PythonTrainBackend` — runs `trainer.py` as a subprocess.
//!
//! Wire format: stdin is unused; one `StatusUpdate` JSON line per
//! line of stdout. The reader is a dedicated tokio task that
//! streams stdout into the `on_status` callback so the trainer
//! never blocks on a Rust-side queue.
//!
//! Cancellation: SIGTERM, 10s grace, SIGKILL. The grace period
//! lets the trainer flush partial checkpoints + close any open file
//! handles. Modeled on `lamu_core::backends::graceful_kill`; kept
//! local so this crate has no upward dep on lamu-core.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::oneshot;

use crate::backend::{StatusFn, TrainArtifact, TrainBackend};
use crate::error::{Result, TrainError};
use crate::protocol::StatusUpdate;
use crate::spec::TrainSpec;

/// Where to find the python interpreter and the trainer script.
///
/// Resolution is the caller's responsibility — `PythonTrainBackend`
/// takes both as explicit paths so tests can point at a stdlib-only
/// python and the bundled `trainer.py --self-check` mode without
/// any environment magic. The CLI binary (step 5) wires the
/// production resolver: `$LAMU_TRAIN_PYTHON` env > `~/local-llm/.venv/bin/python`
/// > `~/.local/share/lamu/train-venv/bin/python` > system `python3`.
#[derive(Clone, Debug)]
pub struct PythonTrainBackend {
    pub python: PathBuf,
    pub trainer_script: PathBuf,
    /// Extra env passed to the trainer subprocess (PYTHONPATH, etc.).
    pub env: Vec<(String, String)>,
    child_pid: Arc<Mutex<Option<u32>>>,
}

impl PythonTrainBackend {
    pub fn new(python: PathBuf, trainer_script: PathBuf) -> Self {
        Self {
            python,
            trainer_script,
            env: Vec::new(),
            child_pid: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }
}

#[async_trait]
impl TrainBackend for PythonTrainBackend {
    async fn run(&mut self, spec: TrainSpec, on_status: StatusFn) -> Result<TrainArtifact> {
        spec.validate()?;
        let spec_json = serde_json::to_string(&spec)
            .map_err(|e| TrainError::other(format!("serialize TrainSpec for trainer.py: {}", e)))?;

        let mut cmd = Command::new(&self.python);
        cmd.arg(&self.trainer_script).arg(&spec_json);
        for (k, v) in &self.env {
            cmd.env(k, v);
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // KILL-1: new session/process group so cancel can killpg the
        // whole tree (DataLoader workers, torchrun ranks) — not just
        // this direct child.
        #[cfg(unix)]
        {
            // tokio::process::Command exposes `pre_exec` inherently
            // (no std CommandExt import needed).
            // SAFETY: setsid is async-signal-safe; pre_exec_setsid
            // allocates nothing. Sound to run between fork and exec.
            #[allow(unsafe_code)]
            unsafe {
                cmd.pre_exec(crate::python_kill::pre_exec_setsid);
            }
        }

        let mut child = cmd.spawn().map_err(|e| {
            TrainError::Trainer(format!(
                "spawn {} {}: {}",
                self.python.display(),
                self.trainer_script.display(),
                e
            ))
        })?;
        if let Some(pid) = child.id() {
            *self.child_pid.lock() = Some(pid);
            // KILL-2: publish the child's identity so the in-process
            // cancel handler + a cross-process `blut cancel` (via the
            // job pid file) can reach the whole group.
            crate::python_kill::set_active_child(crate::python_kill::capture_identity(pid));
        }

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| TrainError::Trainer("trainer subprocess stdout pipe missing".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| TrainError::Trainer("trainer subprocess stderr pipe missing".into()))?;

        let (artifact_tx, artifact_rx) = oneshot::channel();
        let started = Instant::now();
        let on_status: Arc<StatusFn> = Arc::new(on_status);

        // Stdout reader: forwards every parsed StatusUpdate to the
        // caller's callback. Captures the terminal Done/Failed and
        // ships an artifact down the oneshot channel.
        let on_status_for_reader = Arc::clone(&on_status);
        let stdout_reader = tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            let mut last_done: Option<(f32, PathBuf)> = None;
            let mut last_failed: Option<String> = None;
            while let Ok(Some(line)) = reader.next_line().await {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<StatusUpdate>(line) {
                    Ok(u) => {
                        if let StatusUpdate::Done {
                            final_loss,
                            checkpoint_dir,
                        } = &u
                        {
                            last_done = Some((*final_loss, checkpoint_dir.clone()));
                        }
                        if let StatusUpdate::Failed { error } = &u {
                            last_failed = Some(error.clone());
                        }
                        (on_status_for_reader)(u);
                    }
                    Err(e) => {
                        // Malformed lines never stall the run — log + continue.
                        tracing::warn!(
                            "trainer.py emitted unparseable status: {} ({} bytes): {}",
                            e,
                            line.len(),
                            line.chars().take(200).collect::<String>()
                        );
                    }
                }
            }
            let _ = artifact_tx.send((last_done, last_failed));
        });

        // Stderr drain. Forward to tracing so a buggy trainer's
        // python traceback doesn't disappear into a closed pipe.
        let stderr_reader = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!(target: "blut::trainer_stderr", "{}", line);
            }
        });

        let exit_status = child
            .wait()
            .await
            .map_err(|e| TrainError::Trainer(format!("wait for trainer.py: {}", e)))?;

        // Reader tasks finish once their pipes hit EOF (they always
        // do once the child exits). Awaiting here serializes the
        // final on_status callback before we return.
        let _ = stdout_reader.await;
        let _ = stderr_reader.await;

        *self.child_pid.lock() = None;
        crate::python_kill::clear_active_child();
        let elapsed = started.elapsed();

        let (last_done, last_failed) = artifact_rx
            .await
            .map_err(|_| TrainError::Trainer("status reader dropped before report".into()))?;

        if let Some(error) = last_failed {
            return Err(TrainError::Trainer(error));
        }
        if !exit_status.success() {
            return Err(TrainError::Trainer(format!(
                "trainer.py exited with {} and emitted no Failed status",
                exit_status
            )));
        }
        let (final_loss, checkpoint_dir) = last_done.ok_or_else(|| {
            TrainError::Trainer("trainer.py exited successfully but emitted no Done status".into())
        })?;

        Ok(TrainArtifact {
            checkpoint_dir,
            gguf_path: None,
            final_loss,
            elapsed,
        })
    }

    async fn cancel(&mut self) -> Result<()> {
        // Atomic take() rather than read-then-clear so two concurrent
        // cancel() calls don't both fire the kill sequence.
        let pid = match self.child_pid.lock().take() {
            Some(p) => p,
            None => return Ok(()),
        };
        graceful_kill(pid).await;
        Ok(())
    }
}

/// Re-export for `jobs::cancel_job` callers. The implementation
/// now lives in `crate::python_kill::graceful_kill_pid` and is
/// shared with the cookbook subprocess runners + any future
/// subprocess backend.
pub(crate) use crate::python_kill::graceful_kill_pid;

async fn graceful_kill(pid: u32) {
    crate::python_kill::graceful_kill_pid(pid, Duration::from_secs(10)).await
}
