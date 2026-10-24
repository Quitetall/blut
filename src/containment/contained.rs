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

use std::process::ExitStatus;
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::{Containment, PeakSource, TeardownHandle, WrappedRun, spawn_attached};
use crate::error::{Result, TrainError};

/// How long a backend's out-of-band teardown command (`systemctl --user stop`)
/// may run before the child is killed directly regardless.
const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(10);

/// A spawned child plus the containment it runs in.
///
/// Dropping it before the child exits releases the containment: on Windows
/// that closes the Job Object, which kills every process still in it. On Unix
/// the child keeps running unless its command was built with `kill_on_drop`.
pub struct ContainedChild {
    child: tokio::process::Child,
    backend: Box<dyn Containment>,
    teardown: TeardownHandle,
    unit: String,
    peak_source: PeakSource,
    peak: Option<u64>,
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
            finished: false,
        })
    }

    /// The child, to take its pipes or read its pid.
    pub fn child_mut(&mut self) -> &mut tokio::process::Child {
        &mut self.child
    }

    /// The child's pid, `None` once it has been reaped.
    pub fn id(&self) -> Option<u32> {
        self.child.id()
    }

    /// The containment backend's id (`windows-job-object`, `systemd`, ...).
    pub fn kind(&self) -> &'static str {
        self.backend.kind()
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
    /// [`PeakSource::StderrLine`] backends; a stderr pump calls this and
    /// reports the value it returns.
    pub fn parse_peak(&self, line: &str) -> Option<u64> {
        self.backend.parse_peak(line)
    }

    /// The peak memory of the contained tree, in bytes, once the child has
    /// exited. `None` before exit and for backends that measure no peak.
    pub fn peak_bytes(&self) -> Option<u64> {
        self.peak
    }

    /// Wait for the child to exit, then read the peak and release the
    /// backend's per-run resource.
    pub async fn wait(&mut self) -> Result<ExitStatus> {
        let status = self.child.wait().await.map_err(|e| {
            TrainError::other(format!("wait for {} under {}: {e}", self.unit, self.kind()))
        })?;
        self.finish();
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
                self.finish();
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
    /// the direct child; its caller signals the process group as before.
    pub async fn kill_tree(&mut self) -> Result<ExitStatus> {
        if !self.finished
            && let Some(mut stop) = self.backend.cancel(&self.unit, &self.teardown)
        {
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
        let _ = self.child.start_kill();
        self.wait().await
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if self.peak_source == PeakSource::CgroupFile {
            self.peak = self.backend.read_peak(&self.teardown);
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
