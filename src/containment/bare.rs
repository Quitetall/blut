// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Bare containment: NO memory cap. The broker's admission gate (free-RAM
//! refusal) is the only floor, and the kernel OOM killer is the only hard
//! backstop. Used on macOS, under `BLUT_NO_CONTAIN=1`, and as the last-resort
//! fallback when no real cap mechanism is available. Available on all
//! platforms.

use std::path::Path;

use super::{Availability, CapSpec, Containment, PeakSource, TeardownHandle, WrappedRun};
use crate::error::{Result, TrainError};

#[derive(Debug, Default)]
pub struct Bare;

impl Containment for Bare {
    fn kind(&self) -> &'static str {
        "bare"
    }

    fn available(&self) -> Availability {
        Availability::Present
    }

    fn wrap_command(
        &self,
        program: &Path,
        script: &Path,
        args: &[String],
        cwd: &Path,
        env: &[(String, String)],
        _unit: &str,
        _caps: &CapSpec,
    ) -> Result<WrappedRun> {
        if program.as_os_str().is_empty() {
            return Err(TrainError::other("bare wrap: empty program"));
        }
        let mut c = tokio::process::Command::new(program);
        c.arg(script);
        for a in args {
            c.arg(a);
        }
        c.current_dir(cwd);
        for (k, v) in env {
            c.env(k, v);
        }
        Ok(WrappedRun {
            command: c,
            peak_source: PeakSource::None,
            teardown: TeardownHandle::default(),
        })
    }
}
