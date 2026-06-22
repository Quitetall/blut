// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Windows Job Object containment — STUB. Reports `Unavailable` until
//! implemented, so the factory falls through to `Bare` on Windows (the crate
//! is otherwise Unix-first; Windows is not yet a supported target). The trait's
//! `read_peak` + `TeardownHandle` already generalize to the Job Object handle,
//! so implementing this later needs no trait change.
//!
//! Mechanism, for the future implementation:
//!   - `CreateJobObject` → a job handle.
//!   - `SetInformationJobObject(JobObjectExtendedLimitInformation,
//!      JOBOBJECT_EXTENDED_LIMIT_INFORMATION { BasicLimitInformation.LimitFlags
//!      |= JOB_OBJECT_LIMIT_JOB_MEMORY, JobMemoryLimit = caps.mem_max })` — the
//!      hard tree-wide memory cap (the Job Object analogue of cgroup
//!      `memory.max` with `oom.group=1`).
//!   - spawn the child SUSPENDED → `AssignProcessToJobObject(job, child)` →
//!      resume. All descendants inherit the job's cap.
//!   - cancel = `TerminateJobObject` (the analogue of `cgroup.kill`).
//!   - peak via `QueryInformationJobObject(...).PeakJobMemoryUsed` →
//!      `PeakSource::CgroupFile`-style post-wait read in `read_peak`.

use std::path::Path;

use super::{Availability, CapSpec, Containment, WrappedRun};
use crate::error::{Result, TrainError};

#[derive(Debug, Default)]
pub struct WindowsJobObject;

impl Containment for WindowsJobObject {
    fn kind(&self) -> &'static str {
        "windows-job-object"
    }

    fn available(&self) -> Availability {
        // Not implemented yet → the factory falls to Bare.
        Availability::Unavailable
    }

    fn wrap_command(
        &self,
        _program: &Path,
        _script: &Path,
        _args: &[String],
        _cwd: &Path,
        _env: &[(String, String)],
        _unit: &str,
        _caps: &CapSpec,
    ) -> Result<WrappedRun> {
        Err(TrainError::other(
            "WindowsJobObject containment is not implemented",
        ))
    }
}
