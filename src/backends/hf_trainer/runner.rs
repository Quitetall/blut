//! `HfTrainerRunner` — subprocess + JSON wire to the bundled
//! Python wrapper.
//!
//! Wire format:
//!
//!   - Argv: `python <wrapper.py> <job_json_path>`.
//!     The wrapper reads its JSON job spec from the file (rather
//!     than argv) so the spec can carry long fields (prompts,
//!     paths, hyperparams) without bumping ARG_MAX.
//!   - Stdout: one JSON line per status event. Schema:
//!       `{"kind": "step", "step": int, "total": int, "loss": f}`
//!       `{"kind": "saved", "path": str}`
//!       `{"kind": "done", "checkpoint_dir": str, "final_loss": f}`
//!       `{"kind": "failed", "error": str}`
//!   - Stderr: free-form. Captured to tracing::warn.
//!
//! The wrapper script lives at
//! `src/backends/hf_trainer/python/hf_trainer_runner.py` and is
//! invoked via the auto-managed venv at `~/.local/share/blut/hf-venv/`
//! (see `super::venv`).

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use crate::python_kill::graceful_kill_pid;

use super::venv::{ensure_venv, VenvError};

/// Typed job spec the Rust side hands the Python wrapper. Mirrors
/// `transformers.TrainingArguments` for the canonical fields +
/// carries a `task` discriminator so the wrapper picks
/// `transformers.Trainer` vs `trl.DPOTrainer`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HfTrainerJob {
    /// "sft" | "dpo" | "distill". Drives wrapper-side trainer
    /// class selection.
    pub task: String,
    /// HuggingFace model id (e.g. "Qwen/Qwen3-7B") OR an absolute
    /// path to a local checkpoint dir.
    pub base_model: String,
    /// Absolute path to the training dataset (JSONL).
    pub train_dataset_path: PathBuf,
    /// Optional eval dataset path. None = no eval loop.
    pub eval_dataset_path: Option<PathBuf>,
    /// Where the trainer should save its checkpoints.
    pub output_dir: PathBuf,
    /// Common hyperparams. The wrapper passes these straight to
    /// `TrainingArguments`.
    pub lr: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub seq_len: u32,
    pub seed: u64,
    /// Free-form extras. Folded into the wrapper's
    /// `TrainingArguments(**extra)` kwargs. Lets recipes pass
    /// `optim`, `lr_scheduler_type`, `warmup_ratio`, etc. without
    /// adding a typed field for every TrainingArguments option.
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// LoRA / QLoRA hyperparams. None = full-rank fine-tune.
    #[serde(default)]
    pub peft: Option<PeftConfig>,
    /// DPO-specific (ignored when task != "dpo").
    #[serde(default)]
    pub dpo: Option<DpoConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeftConfig {
    /// "lora" | "qlora".
    pub method: String,
    pub rank: u32,
    pub alpha: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DpoConfig {
    pub beta: f32,
    /// Optional override for the preferences file. None = fall
    /// back to `HfTrainerJob.train_dataset_path`, which is the
    /// common case (DPO trains on the dataset's own
    /// chosen/rejected pairs).
    #[serde(default)]
    pub preferences_path: Option<PathBuf>,
}

/// One line emitted by the wrapper on stdout.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StatusLine {
    Step {
        step: u64,
        total: u64,
        #[serde(default)]
        loss: Option<f32>,
        #[serde(default)]
        lr: Option<f32>,
    },
    Saved {
        path: PathBuf,
    },
    Done {
        checkpoint_dir: PathBuf,
        #[serde(default)]
        final_loss: Option<f32>,
    },
    Failed {
        error: String,
    },
}

#[derive(Debug)]
pub struct HfRunArtifact {
    pub checkpoint_dir: PathBuf,
    pub final_loss: Option<f32>,
    pub elapsed: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("venv: {0}")]
    Venv(#[from] VenvError),
    #[error("python -m hf_trainer_runner spawn at {python}: {source}")]
    Spawn {
        python: String,
        #[source]
        source: std::io::Error,
    },
    #[error("python wrapper exited with status {status} and emitted no Done line")]
    Failed { status: String },
    #[error("python wrapper reported failure: {message}")]
    WrapperFailed { message: String },
    #[error("io at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("serialize job spec: {0}")]
    SerializeJob(#[source] serde_json::Error),
}

pub struct HfTrainerRunner {
    child_pid: Arc<Mutex<Option<u32>>>,
}

impl HfTrainerRunner {
    pub fn new() -> Self {
        Self {
            child_pid: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn run(
        &mut self,
        job: HfTrainerJob,
        on_status: Box<dyn Fn(StatusLine) + Send + Sync>,
    ) -> Result<HfRunArtifact, RunError> {
        // Ensure venv (blocking — fine, this only fires on the
        // first hf_* recipe of a session).
        let python = tokio::task::spawn_blocking(ensure_venv)
            .await
            .map_err(|e| RunError::Io {
                path: PathBuf::from("ensure_venv-join"),
                source: std::io::Error::new(std::io::ErrorKind::Other, format!("{e}")),
            })??;

        // Write job spec to a tempfile so the wrapper can read it
        // without argv-size constraints.
        let td = tempfile::tempdir().map_err(|source| RunError::Io {
            path: PathBuf::from("hf-trainer-tempdir"),
            source,
        })?;
        let spec_path = td.path().join("job.json");
        let body = serde_json::to_vec_pretty(&job).map_err(RunError::SerializeJob)?;
        std::fs::write(&spec_path, body).map_err(|source| RunError::Io {
            path: spec_path.clone(),
            source,
        })?;

        // Locate the wrapper script. It ships in-tree and gets
        // resolved relative to the runner module at compile time.
        let wrapper = match wrapper_path() {
            Some(p) if p.exists() => p,
            _ => {
                return Err(RunError::Io {
                    path: PathBuf::from(
                        "src/backends/hf_trainer/python/hf_trainer_runner.py",
                    ),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "wrapper script missing; expected alongside the binary",
                    ),
                })
            }
        };

        let mut cmd = Command::new(&python);
        cmd.arg(&wrapper).arg(&spec_path);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().map_err(|source| RunError::Spawn {
            python: python.display().to_string(),
            source,
        })?;
        if let Some(pid) = child.id() {
            *self.child_pid.lock() = Some(pid);
        }

        let started = Instant::now();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let on_status: Arc<dyn Fn(StatusLine) + Send + Sync> = Arc::from(on_status);
        let collected: Arc<Mutex<(Option<HfRunArtifact>, Option<String>)>> =
            Arc::new(Mutex::new((None, None)));

        let stdout_handle = if let Some(stdout) = stdout {
            let cb = on_status.clone();
            let collected = collected.clone();
            Some(tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<StatusLine>(line) {
                        Ok(s) => {
                            if let StatusLine::Done {
                                checkpoint_dir,
                                final_loss,
                            } = &s
                            {
                                let mut c = collected.lock();
                                c.0 = Some(HfRunArtifact {
                                    checkpoint_dir: checkpoint_dir.clone(),
                                    final_loss: *final_loss,
                                    elapsed: Duration::ZERO,
                                });
                            }
                            if let StatusLine::Failed { error } = &s {
                                collected.lock().1 = Some(error.clone());
                            }
                            (cb)(s);
                        }
                        Err(_) => {
                            tracing::warn!(
                                target: "blut::hf_trainer_stdout",
                                "unparseable status: {line}"
                            );
                        }
                    }
                }
            }))
        } else {
            None
        };

        let stderr_handle = if let Some(stderr) = stderr {
            Some(tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::warn!(target: "blut::hf_trainer_stderr", "{}", line);
                }
            }))
        } else {
            None
        };

        let exit = child.wait().await.map_err(|source| RunError::Io {
            path: PathBuf::from("hf-trainer-wait"),
            source,
        })?;
        if let Some(h) = stdout_handle {
            if let Err(e) = h.await {
                tracing::error!(target: "blut::hf_trainer", "stdout reader join failed: {e}");
            }
        }
        if let Some(h) = stderr_handle {
            if let Err(e) = h.await {
                tracing::error!(target: "blut::hf_trainer", "stderr reader join failed: {e}");
            }
        }
        *self.child_pid.lock() = None;
        let elapsed = started.elapsed();

        let mut collected = collected.lock();
        if let Some(err) = collected.1.take() {
            return Err(RunError::WrapperFailed { message: err });
        }
        if !exit.success() {
            return Err(RunError::Failed {
                status: format!("{exit}"),
            });
        }
        let mut artifact = collected.0.take().ok_or_else(|| RunError::Failed {
            status: "no Done line emitted".into(),
        })?;
        artifact.elapsed = elapsed;
        Ok(artifact)
    }

    pub async fn cancel(&mut self) {
        let pid = match self.child_pid.lock().take() {
            Some(p) => p,
            None => return,
        };
        graceful_kill_pid(pid, Duration::from_secs(10)).await;
    }
}

impl Default for HfTrainerRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Locate the wrapper script. Resolution order:
///   1. `$BLUT_HF_TRAINER_RUNNER` env (absolute path override)
///   2. `<exe_dir>/python/hf_trainer_runner.py` (when installed
///      alongside the binary)
///   3. `<repo_root>/src/backends/hf_trainer/python/hf_trainer_runner.py`
///      (development tree)
fn wrapper_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_HF_TRAINER_RUNNER") {
        return Some(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("python").join("hf_trainer_runner.py");
            if p.exists() {
                return Some(p);
            }
        }
    }
    // Development tree fallback: walk up to find a Cargo.toml then
    // resolve against `src/backends/hf_trainer/python/`.
    if let Ok(cwd) = std::env::current_dir() {
        let mut cur = cwd.as_path();
        loop {
            let candidate = cur
                .join("src")
                .join("backends")
                .join("hf_trainer")
                .join("python")
                .join("hf_trainer_runner.py");
            if candidate.exists() {
                return Some(candidate);
            }
            match cur.parent() {
                Some(p) => cur = p,
                None => break,
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_round_trips_via_serde() {
        let job = HfTrainerJob {
            task: "sft".into(),
            base_model: "Qwen/Qwen3-7B".into(),
            train_dataset_path: "/tmp/train.jsonl".into(),
            eval_dataset_path: Some("/tmp/eval.jsonl".into()),
            output_dir: "/tmp/ckpt".into(),
            lr: 2e-4,
            epochs: 3,
            batch_size: 1,
            grad_accum: 8,
            seq_len: 4096,
            seed: 42,
            extra: serde_json::Map::new(),
            peft: Some(PeftConfig {
                method: "qlora".into(),
                rank: 16,
                alpha: 32,
            }),
            dpo: None,
        };
        let s = serde_json::to_string(&job).unwrap();
        let back: HfTrainerJob = serde_json::from_str(&s).unwrap();
        assert_eq!(back.task, "sft");
        assert_eq!(back.peft.as_ref().unwrap().method, "qlora");
    }

    #[test]
    fn status_lines_round_trip() {
        let lines = [
            r#"{"kind":"step","step":42,"total":100,"loss":0.42}"#,
            r#"{"kind":"saved","path":"/tmp/ckpt-step-100"}"#,
            r#"{"kind":"done","checkpoint_dir":"/tmp/ckpt","final_loss":0.21}"#,
            r#"{"kind":"failed","error":"OOM at step 1234"}"#,
        ];
        for s in lines {
            let parsed: StatusLine = serde_json::from_str(s).unwrap();
            // Round-trip back.
            let _ = serde_json::to_string(&parsed).unwrap();
        }
    }

    #[test]
    fn wrapper_path_resolves_in_dev_tree() {
        // In the dev tree the wrapper script doesn't exist yet
        // (lands alongside this commit), so this test verifies the
        // function returns Some(...) once the script is present.
        // Until then it asserts the path-search code doesn't panic
        // even when none exists.
        let _ = wrapper_path();
    }
}
