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
    /// D3 liveness watchdog: if `Some`, a trainer that emits NO status
    /// line (Step/Eval/Saved/Heartbeat) AND NO stderr line for this long
    /// is presumed hung and SIGTERM'd. `None` (default) = no watchdog,
    /// so a chatty-or-heartbeating trainer is never falsely killed and
    /// old trainers (no heartbeats) are unaffected.
    pub liveness_timeout: Option<Duration>,
    /// Where to place the trainer (#3). `Local` (default) spawns it directly
    /// here (byte-identical to the original behavior); `Slurm`/`Ray` wrap the
    /// `python script spec` invocation via the matching launcher so the SAME
    /// streaming/cancel path runs against a cluster-placed job.
    launch_target: crate::config::launcher::LaunchTarget,
    child_pid: Arc<Mutex<Option<u32>>>,
}

impl PythonTrainBackend {
    pub fn new(python: PathBuf, trainer_script: PathBuf) -> Self {
        Self {
            python,
            trainer_script,
            env: Vec::new(),
            liveness_timeout: None,
            launch_target: crate::config::launcher::LaunchTarget::Local,
            child_pid: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Enable the D3 liveness watchdog with the given idle timeout.
    pub fn with_liveness_timeout(mut self, timeout: Duration) -> Self {
        self.liveness_timeout = Some(timeout);
        self
    }

    /// Place the trainer on `target` (#3). `Local` = direct spawn here.
    pub fn with_launch_target(mut self, target: crate::config::launcher::LaunchTarget) -> Self {
        self.launch_target = target;
        self
    }
}

#[async_trait]
impl TrainBackend for PythonTrainBackend {
    async fn run(&mut self, spec: TrainSpec, on_status: StatusFn) -> Result<TrainArtifact> {
        spec.validate()?;
        let spec_json = serde_json::to_string(&spec)
            .map_err(|e| TrainError::other(format!("serialize TrainSpec for trainer.py: {}", e)))?;

        // Local spawns the trainer directly (byte-identical to the original);
        // Slurm/Ray wrap `python script spec` via the launcher (program/argv/env
        // are launcher data) but keep the SAME streaming + cancel path below.
        use crate::config::launcher::{launcher_for, LaunchTarget};
        let mut cmd = match self.launch_target {
            LaunchTarget::Local => {
                let mut c = Command::new(&self.python);
                c.arg(&self.trainer_script).arg(&spec_json);
                c
            }
            target => {
                let inner = vec![
                    self.python.display().to_string(),
                    self.trainer_script.display().to_string(),
                    spec_json.clone(),
                ];
                let w = launcher_for(target)
                    .wrap("blut-train", &inner)
                    .map_err(|e| TrainError::other(format!("launcher wrap: {e}")))?;
                let mut c = Command::new(&w.program);
                c.args(&w.args);
                for (k, v) in &w.env {
                    c.env(k, v);
                }
                c
            }
        };
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
            crate::python_kill::register_child(crate::python_kill::capture_identity(pid));
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

        // D3 liveness: every status line AND every stderr line refreshes
        // `last_activity`; the watchdog kills a trainer that goes silent
        // on BOTH for `liveness_timeout`.
        let last_activity = Arc::new(Mutex::new(Instant::now()));
        let watchdog_reason: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        // Stdout reader: forwards every parsed StatusUpdate to the
        // caller's callback. Captures the terminal Done/Failed and
        // ships an artifact down the oneshot channel. Heartbeats (D4)
        // refresh liveness but are SWALLOWED (no status.jsonl noise).
        let on_status_for_reader = Arc::clone(&on_status);
        let activity_stdout = Arc::clone(&last_activity);
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
                        *activity_stdout.lock() = Instant::now();
                        if matches!(u, StatusUpdate::Heartbeat { .. }) {
                            continue; // liveness only — don't forward
                        }
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
        // python traceback doesn't disappear into a closed pipe. Stderr
        // chatter (e.g. tqdm) is also a liveness signal.
        let activity_stderr = Arc::clone(&last_activity);
        let stderr_reader = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                *activity_stderr.lock() = Instant::now();
                tracing::info!(target: "blut::trainer_stderr", "{}", line);
            }
        });

        // D3 watchdog task (only when configured). Ticks at min(timeout/4,
        // 30s); on idle > timeout it SIGTERMs the child group and records
        // a reason. Aborted below once the child exits.
        let watchdog = self.liveness_timeout.map(|timeout| {
            let activity = Arc::clone(&last_activity);
            let pid_handle = Arc::clone(&self.child_pid);
            let reason = Arc::clone(&watchdog_reason);
            tokio::spawn(async move {
                let tick = std::cmp::min(timeout / 4, Duration::from_secs(30))
                    .max(Duration::from_millis(100));
                loop {
                    tokio::time::sleep(tick).await;
                    let idle = activity.lock().elapsed();
                    if idle <= timeout {
                        continue;
                    }
                    // Idle past the limit — record the reason (always) and
                    // kill the child group if we have its pid, then stop.
                    *reason.lock() = Some(format!(
                        "liveness watchdog: no output for {idle:?} (limit {timeout:?})"
                    ));
                    // Bind the Copy pid (drop the guard) before the await.
                    let pid = *pid_handle.lock();
                    match pid {
                        Some(pid) => {
                            tracing::warn!(
                                "liveness watchdog: no trainer output for {idle:?} (> {timeout:?}); \
                                 killing pid {pid}"
                            );
                            crate::python_kill::graceful_kill_pid(pid, Duration::from_secs(10)).await;
                        }
                        None => tracing::warn!(
                            "liveness watchdog: idle {idle:?} but no child pid to kill"
                        ),
                    }
                    return;
                }
            })
        });

        let exit_status = child
            .wait()
            .await
            .map_err(|e| TrainError::Trainer(format!("wait for trainer.py: {}", e)))?;

        // Child exited — stop the watchdog if it's still ticking.
        if let Some(w) = &watchdog {
            w.abort();
        }

        // Reader tasks finish once their pipes hit EOF (they always
        // do once the child exits). Awaiting here serializes the
        // final on_status callback before we return.
        let _ = stdout_reader.await;
        let _ = stderr_reader.await;

        if let Some(pid) = self.child_pid.lock().take() {
            crate::python_kill::unregister_child(pid);
        }
        let elapsed = started.elapsed();

        // If the watchdog killed the trainer, surface THAT (a hang), not
        // the generic "exited with no Done" — it's the actionable cause.
        if let Some(reason) = watchdog_reason.lock().take() {
            return Err(TrainError::Trainer(reason));
        }

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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::backend::StatusFn;
    use crate::spec::{DatasetSource, Method, Optim, TrainSpec};

    fn python3() -> Option<PathBuf> {
        which::which("python3").ok()
    }

    fn spec() -> TrainSpec {
        TrainSpec {
            base_model: "org/m".into(),
            output_name: "wd-test".into(),
            output_dir: PathBuf::from("/tmp/blut-wd-test"),
            method: Method::QLora { rank: 16, alpha: 32 },
            dataset: DatasetSource::JsonlPath {
                path: PathBuf::from("/tmp/x.jsonl"),
            },
            optimizer: Optim::AdamW8bit,
            lr: 2e-4,
            epochs: 1,
            batch_size: 1,
            grad_accum: 1,
            seq_len: 512,
            seed: 42,
            quant: "Q4_K_M".into(),
            skip_convert: true,
            dpo_beta: None,
        }
    }

    /// Write a python script that prints one Step line, then runs the
    /// given tail (`silent` = sleep forever; `heartbeat` = ping then Done).
    fn trainer_script(td: &std::path::Path, tail: &str) -> PathBuf {
        let body = format!(
            "import json,sys,time\n\
             print(json.dumps({{'kind':'step','step':1,'total':10,'loss':1.0,'lr':0.0,'vram_mb':0}}),flush=True)\n\
             {tail}\n"
        );
        let p = td.join("fake_trainer.py");
        std::fs::write(&p, body).unwrap();
        p
    }

    #[tokio::test]
    async fn liveness_watchdog_kills_a_silent_trainer() {
        let Some(py) = python3() else {
            eprintln!("skip: python3 not found");
            return;
        };
        let td = tempfile::tempdir().unwrap();
        // Prints one Step, then goes silent for 60s — the watchdog must
        // kill it well before that.
        let script = trainer_script(td.path(), "time.sleep(60)");
        let mut backend = PythonTrainBackend::new(py, script)
            .with_liveness_timeout(Duration::from_millis(500));
        let on_status: StatusFn = Box::new(|_u| {});
        let fut = backend.run(spec(), on_status);
        let r = tokio::time::timeout(Duration::from_secs(10), fut)
            .await
            .expect("watchdog must fire long before the 10s guard");
        match r {
            Err(TrainError::Trainer(msg)) => {
                assert!(msg.contains("liveness watchdog"), "got: {msg}");
            }
            other => panic!("expected a liveness-watchdog error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn heartbeating_trainer_survives_the_watchdog() {
        let Some(py) = python3() else {
            eprintln!("skip: python3 not found");
            return;
        };
        let td = tempfile::tempdir().unwrap();
        let ckpt = td.path().join("ckpt");
        std::fs::create_dir_all(&ckpt).unwrap();
        // Heartbeats every 100ms for ~1.5s (keeping liveness fresh under
        // a 500ms timeout), then Done.
        let tail = format!(
            "for _ in range(15):\n\
             \x20 print(json.dumps({{'kind':'heartbeat','phase':'load'}}),flush=True)\n\
             \x20 time.sleep(0.1)\n\
             print(json.dumps({{'kind':'done','final_loss':0.5,'checkpoint_dir':'{}'}}),flush=True)",
            ckpt.display()
        );
        let script = trainer_script(td.path(), &tail);
        let mut backend = PythonTrainBackend::new(py, script)
            .with_liveness_timeout(Duration::from_millis(500));
        let on_status: StatusFn = Box::new(|_u| {});
        let r = tokio::time::timeout(Duration::from_secs(10), backend.run(spec(), on_status))
            .await
            .expect("must complete");
        assert!(r.is_ok(), "heartbeats must keep the trainer alive: {r:?}");
    }
}
