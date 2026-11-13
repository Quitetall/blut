// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! A child process spawned inside its containment and owned together with it.
//!
//! [`Containment`] is construction-only: a caller wraps a command, spawns it,
//! and must then keep the backend and the [`TeardownHandle`] alive for the
//! whole run, call [`Containment::after_spawn`], read the peak after exit, and
//! call [`Containment::cleanup`]. Each step that a caller forgets fails
//! differently per platform; a Windows Job Object, for instance, kills its
//! processes as soon as the last teardown handle drops. [`ContainedChild`]
//! does all of it in one value: it spawns through [`super::spawn_attached`],
//! holds the containment until the child has exited, and exposes the peak and
//! a whole-tree kill.
//!
//! # A contained run does not outlive its `ContainedChild`
//!
//! The direct child is not always the workload. Under the systemd backend it
//! is the `systemd-run` client, and killing that client leaves the transient
//! service running. So the backend's own teardown (`systemctl --user stop`,
//! `cgroup.kill`, `TerminateJobObject`) runs whenever the direct child exits,
//! however it exits, and again, best-effort, when the value is dropped first
//! (a hard stage timeout drops the stage future). For a backend with no
//! teardown of its own (`bare`, `rlimit`) a drop kills the child's process
//! group, which reaches the whole tree when the caller spawned the child as a
//! group leader (`python_kill::pre_exec_setsid`).

use std::process::ExitStatus;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::sync::CancellationToken;

use super::{Containment, PeakSource, TeardownHandle, WrappedRun, spawn_attached};
use crate::error::{Result, TrainError};

/// How long a backend's out-of-band teardown command (`systemctl --user stop`)
/// may run before the child is killed directly regardless. `systemctl stop`
/// returns once the unit has stopped, which for a process ignoring SIGTERM is
/// the systemd backend's 10 s stop timeout plus the SIGKILL; this leaves room
/// for both.
const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(15);

/// How long to wait, after the child exits, for its stderr pump to drain. A
/// grandchild that keeps the pipe open would otherwise hold the wait forever.
const PUMP_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// A spawned child plus the containment it runs in. See the module docs for
/// what dropping it does.
pub struct ContainedChild {
    child: tokio::process::Child,
    backend: Arc<dyn Containment>,
    teardown: TeardownHandle,
    unit: String,
    peak_source: PeakSource,
    peak: Option<u64>,
    /// Peak scraped from stderr by [`Self::pump_stderr`], for
    /// [`PeakSource::StderrLine`] backends.
    stderr_peak: Arc<Mutex<Option<u64>>>,
    oom_kills: Option<u64>,
    /// Whether stderr reported an OOM kill, scraped by [`Self::pump_stderr`]
    /// for [`PeakSource::StderrLine`] backends.
    stderr_oom: Arc<Mutex<Option<bool>>>,
    stderr_pump: Option<tokio::task::JoinHandle<()>>,
    finished: bool,
}

impl std::fmt::Debug for ContainedChild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainedChild")
            .field("unit", &self.unit)
            .field("backend", &self.backend.kind())
            .field("pid", &self.child.id())
            .field("peak_source", &self.peak_source)
            .field("peak", &self.peak)
            .field("finished", &self.finished)
            .finish()
    }
}

impl ContainedChild {
    /// Spawn `run` (from `backend`'s [`Containment::wrap_command`]) and finish
    /// its containment before the child runs.
    ///
    /// `configure` adjusts the command before the spawn: pipes,
    /// `kill_on_drop`, a Unix `pre_exec` hook. It runs before
    /// [`Containment::before_spawn`], so a backend's own spawn flags win. On
    /// any failure the backend's per-run resource is cleaned up and no child
    /// is left running. The error is [`spawn_attached`]'s, unchanged: a
    /// failed `spawn()` stays a [`TrainError::Io`] with its OS error kind.
    pub async fn spawn(
        backend: Box<dyn Containment>,
        run: WrappedRun,
        unit: impl Into<String>,
        configure: impl FnOnce(&mut tokio::process::Command),
    ) -> Result<Self> {
        let backend: Arc<dyn Containment> = Arc::from(backend);
        let WrappedRun {
            mut command,
            peak_source,
            teardown,
        } = run;
        let unit = unit.into();
        configure(&mut command);
        let child = match spawn_attached(backend.as_ref(), &mut command, &teardown).await {
            Ok(child) => child,
            Err(e) => {
                tracing::warn!(
                    "containment: {unit} failed to start under {}: {e}",
                    backend.kind()
                );
                backend.cleanup(&teardown);
                return Err(e);
            }
        };
        tracing::info!(
            "containment: {unit} attached to {} (pid {})",
            backend.kind(),
            child
                .id()
                .map_or_else(|| "unknown".to_string(), |p| p.to_string()),
        );
        Ok(Self {
            child,
            backend,
            teardown,
            unit,
            peak_source,
            peak: None,
            stderr_peak: Arc::new(Mutex::new(None)),
            oom_kills: None,
            stderr_oom: Arc::new(Mutex::new(None)),
            stderr_pump: None,
            finished: false,
        })
    }

    /// The child, to take its pipes or read its pid.
    pub fn child_mut(&mut self) -> &mut tokio::process::Child {
        &mut self.child
    }

    /// The direct child's pid, `None` once it has been reaped. Under the
    /// systemd backend this is the `systemd-run` client, not the workload; a
    /// workload that needs its own pid known (a trainer's checkpoint marker)
    /// reports it itself.
    pub fn id(&self) -> Option<u32> {
        self.child.id()
    }

    /// The containment backend's id (`windows-job-object`, `systemd`, ...).
    pub fn kind(&self) -> &'static str {
        self.backend.kind()
    }

    /// Whether the backend can kill the whole tree by itself (systemd,
    /// cgroup v2, Job Object). `false` for `bare` and `rlimit`, whose callers
    /// reach the tree through its process group.
    pub fn tears_down_tree(&self) -> bool {
        matches!(
            self.backend.kind(),
            "systemd" | "cgroup2" | "windows-job-object"
        )
    }

    /// The unit name the run was wrapped under.
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Where this run's peak comes from.
    pub fn peak_source(&self) -> PeakSource {
        self.peak_source
    }

    /// Parse a peak from one of the child's stderr lines, for
    /// [`PeakSource::StderrLine`] backends.
    pub fn parse_peak(&self, line: &str) -> Option<u64> {
        self.backend.parse_peak(line)
    }

    /// Read the child's piped stderr on a task, handing every line to `sink`
    /// (without its line ending, invalid UTF-8 replaced). For a
    /// [`PeakSource::StderrLine`] backend the line carrying the tree's peak
    /// (systemd's `Memory peak:`) is recorded too, so [`Self::peak_bytes`]
    /// reports it; the line is still passed to `sink`. Returns `false`, doing
    /// nothing, when stderr was not piped or was already taken.
    pub fn pump_stderr(&mut self, mut sink: impl FnMut(String) + Send + 'static) -> bool {
        let Some(stderr) = self.child.stderr.take() else {
            return false;
        };
        let scrape = self.peak_source == PeakSource::StderrLine;
        let backend = Arc::clone(&self.backend);
        let slot = Arc::clone(&self.stderr_peak);
        let oom_slot = Arc::clone(&self.stderr_oom);
        self.stderr_pump = Some(tokio::spawn(async move {
            let mut reader = BufReader::new(stderr);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                let line = String::from_utf8_lossy(&buf)
                    .trim_end_matches(['\n', '\r'])
                    .to_string();
                if scrape && let Some(bytes) = backend.parse_peak(&line) {
                    *slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(bytes);
                }
                if scrape && let Some(oom) = backend.parse_oom_kill(&line) {
                    *oom_slot.lock().unwrap_or_else(|p| p.into_inner()) = Some(oom);
                }
                sink(line);
            }
        }));
        true
    }

    /// The peak memory of the contained tree, in bytes, once the child has
    /// exited. `None` before exit and for backends that measure no peak; for
    /// a [`PeakSource::StderrLine`] backend, also when stderr was not pumped
    /// through [`Self::pump_stderr`].
    pub fn peak_bytes(&self) -> Option<u64> {
        self.peak
    }

    /// How many OOM kills the kernel reported in the contained tree, once the
    /// child has exited. `Some(0)` means the backend looked and the kernel
    /// reported none, so the exit was not an OOM whatever its status or peak.
    /// `None` before exit and when the backend cannot tell: the rlimit, bare
    /// and Windows backends, and a [`PeakSource::StderrLine`] backend whose
    /// stderr was not pumped through [`Self::pump_stderr`] or was cut short.
    ///
    /// The cgroup-v2 backend reports the kernel's `oom_kill` counter. The
    /// systemd backend reports `1` when the unit's result was `oom-kill` and
    /// `0` for any other result; it does not count processes.
    pub fn oom_kills(&self) -> Option<u64> {
        self.oom_kills
    }

    /// Wait for the child to exit, then tear down what is left of the tree,
    /// read the peak and release the backend's per-run resource.
    pub async fn wait(&mut self) -> Result<ExitStatus> {
        let status = self.child.wait().await.map_err(|e| {
            TrainError::other(format!("wait for {} under {}: {e}", self.unit, self.kind()))
        })?;
        self.finish().await;
        Ok(status)
    }

    /// Wait for the child to exit, or kill the whole contained tree when
    /// `cancel` fires first. `Ok(None)` means it was cancelled and killed.
    pub async fn wait_or_cancel(
        &mut self,
        cancel: &CancellationToken,
    ) -> Result<Option<ExitStatus>> {
        let exited = tokio::select! {
            status = self.child.wait() => Some(status),
            () = cancel.cancelled() => None,
        };
        match exited {
            Some(status) => {
                let status = status.map_err(|e| {
                    TrainError::other(format!("wait for {} under {}: {e}", self.unit, self.kind()))
                })?;
                self.finish().await;
                Ok(Some(status))
            }
            None => {
                tracing::warn!(
                    "containment: {} cancelled; killing its process tree ({})",
                    self.unit,
                    self.kind()
                );
                self.kill_tree().await?;
                Ok(None)
            }
        }
    }

    /// Kill every process in the contained tree, then reap the child.
    ///
    /// Uses the backend's own teardown (`TerminateJobObject`, `cgroup.kill`,
    /// `systemctl --user stop`) and then kills the direct child as well. A
    /// backend with no teardown of its own (`bare`, `rlimit`) reaches only
    /// the direct child here; its caller signals the process group first.
    pub async fn kill_tree(&mut self) -> Result<ExitStatus> {
        self.backend_teardown().await;
        let _ = self.child.start_kill();
        self.wait().await
    }

    /// Run the backend's teardown, bounded by [`TEARDOWN_TIMEOUT`].
    async fn backend_teardown(&self) {
        let Some(mut stop) = self.backend.cancel(&self.unit, &self.teardown) else {
            return;
        };
        stop.stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        match stop.spawn() {
            Ok(mut c) => {
                if tokio::time::timeout(TEARDOWN_TIMEOUT, c.wait())
                    .await
                    .is_err()
                {
                    tracing::warn!(
                        "containment: teardown of {} did not finish in {TEARDOWN_TIMEOUT:?}",
                        self.unit
                    );
                    let _ = c.start_kill();
                }
            }
            Err(e) => tracing::warn!("containment: teardown of {}: {e}", self.unit),
        }
    }

    /// After the direct child has exited: tear down anything still running in
    /// the contained tree (the direct child exiting does not mean the
    /// workload did: see the module docs), drain the stderr pump, read the
    /// peak, log it, and release the per-run resource. Runs once.
    async fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.backend_teardown().await;
        if let Some(pump) = self.stderr_pump.take()
            && tokio::time::timeout(PUMP_DRAIN_TIMEOUT, pump)
                .await
                .is_err()
        {
            tracing::warn!(
                "containment: stderr of {} still open {PUMP_DRAIN_TIMEOUT:?} after it exited",
                self.unit
            );
        }
        self.peak = match self.peak_source {
            PeakSource::CgroupFile => self.backend.read_peak(&self.teardown),
            PeakSource::StderrLine => *self.stderr_peak.lock().unwrap_or_else(|p| p.into_inner()),
            PeakSource::None => None,
        };
        self.oom_kills = match self.peak_source {
            PeakSource::CgroupFile => self.backend.read_oom_kills(&self.teardown),
            PeakSource::StderrLine => self
                .stderr_oom
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .map(u64::from),
            PeakSource::None => None,
        };
        if let Some(kills @ 1..) = self.oom_kills {
            tracing::warn!(
                "containment: the kernel OOM-killed {} ({kills} reported, {})",
                self.unit,
                self.kind()
            );
        }
        match self.peak {
            Some(peak) => tracing::info!(
                "containment: {} exited; peak memory {peak} bytes ({})",
                self.unit,
                self.kind()
            ),
            None => tracing::info!(
                "containment: {} exited; no peak measured ({})",
                self.unit,
                self.kind()
            ),
        }
        self.backend.cleanup(&self.teardown);
    }
}

impl Drop for ContainedChild {
    /// Dropped before [`ContainedChild::wait`] finished: kill the tree,
    /// best-effort and without blocking. The backend's teardown command, if
    /// it has one, is started as a plain process (a drop can happen outside
    /// a runtime) and reaped on a detached thread.
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if let Some(stop) = self.backend.cancel(&self.unit, &self.teardown) {
            let std_stop = stop.as_std();
            let mut c = std::process::Command::new(std_stop.get_program());
            c.args(std_stop.get_args())
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            match c.spawn() {
                Ok(mut child) => {
                    let _ = std::thread::Builder::new()
                        .name("blut-containment-teardown".into())
                        .spawn(move || {
                            let _ = child.wait();
                        });
                }
                Err(e) => tracing::warn!("containment: teardown of {} on drop: {e}", self.unit),
            }
        }
        #[cfg(unix)]
        if !self.tears_down_tree()
            && let (Ok(None), Some(pid)) = (self.child.try_wait(), self.child.id())
        {
            // Still running and not reaped, so its pid cannot have been reused.
            crate::python_kill::kill_group_now(pid);
        }
        self.backend.cleanup(&self.teardown);
    }
}
