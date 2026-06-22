// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `RLIMIT_AS` containment: a portable per-process address-space cap installed
//! via a `pre_exec` hook (`setrlimit(2)`). No root, no cgroup, no systemd —
//! works on locked-down cloud containers (the common rented-GPU case, where
//! `/sys/fs/cgroup` is read-only and there's no user bus) and on macOS.
//!
//! WEAKER than cgroup/systemd: `RLIMIT_AS` caps the VIRTUAL address space of
//! EACH process (not the RSS of the whole fork tree), so a multi-worker
//! DataLoader gets per-worker caps, not one tree-wide cap, and a process that
//! mmaps-but-doesn't-touch can hit the limit early. But it is a REAL cap that
//! makes an oversized allocation fail with `MemoryError`/`ENOMEM` instead of
//! driving the box into global OOM — verified on a Thunder k8s box:
//! `prlimit --as=512M python -c "bytearray(100GiB)"` → MemoryError, box intact.
//!
//! No trustworthy peak (RLIMIT gives no accounting) → `PeakSource::None`.
//! Teardown is the runner's `killpg` fallback (no out-of-band mechanism).

use std::path::Path;

use super::{Availability, CapSpec, Containment, PeakSource, TeardownHandle, WrappedRun};
use crate::error::{Result, TrainError};

#[derive(Debug, Default)]
pub struct RlimitAddressSpace;

impl Containment for RlimitAddressSpace {
    fn kind(&self) -> &'static str {
        "rlimit"
    }

    fn available(&self) -> Availability {
        // setrlimit(RLIMIT_AS) is part of POSIX; always usable on unix. The
        // factory only reaches this backend after systemd/cgroup2 declined.
        Availability::Present
    }

    fn wrap_command(
        &self,
        program: &Path,
        args: &[String],
        cwd: &Path,
        env: &[(String, String)],
        _unit: &str,
        caps: &CapSpec,
    ) -> Result<WrappedRun> {
        if program.as_os_str().is_empty() {
            return Err(TrainError::other("rlimit wrap: empty program"));
        }
        let mut c = tokio::process::Command::new(program);
        // args[0] is the script (or `-m` for a torchrun DDP launch); the runner's
        // kernel_argv builds the full argv after the program.
        for a in args {
            c.arg(a);
        }
        c.current_dir(cwd);
        for (k, v) in env {
            c.env(k, v);
        }

        // Install the RLIMIT_AS cap in the forked child, before exec. ONLY
        // when an explicit hard cap is requested (`mem_max`) — NEVER a default.
        // RLIMIT_AS caps VIRTUAL address space, and a CUDA process reserves a
        // huge VA region up front (often tens of GiB of unbacked mappings); a
        // defaulted cap would abort it spuriously. So a `None` cap leaves the
        // process unlimited, and the caller must opt in to an AS cap sized with
        // CUDA's VA reservation in mind (typically generous). For RSS-precise
        // capping, prefer cgroup2/systemd; rlimit is the last-resort portable
        // floor for locked-down boxes.
        if let Some(max) = caps.mem_max.filter(|&b| b > 0) {
            // SAFETY: the closure runs in the forked child between fork and
            // exec. `setrlimit` is async-signal-safe and the closure does NO
            // heap allocation (the rlimit value is a stack copy). Mirrors the
            // discipline of `python_kill::pre_exec_setsid`.
            #[allow(unsafe_code)]
            unsafe {
                c.pre_exec(move || {
                    use nix::sys::resource::{getrlimit, setrlimit, Resource};
                    // Don't exceed the inherited HARD limit — a process can
                    // only LOWER its hard limit, so requesting `max` above it
                    // fails with EPERM (an opaque spawn error). Clamp to the
                    // current hard ceiling (unless it's unlimited / 0).
                    let want = match getrlimit(Resource::RLIMIT_AS) {
                        Ok((_soft, hard)) if hard != 0 => max.min(hard),
                        _ => max,
                    };
                    // Cap both soft and hard to `want`; the child can't raise
                    // its own hard limit, so this is a firm ceiling.
                    setrlimit(Resource::RLIMIT_AS, want, want)
                        .map_err(|e| std::io::Error::from_raw_os_error(e as i32))
                });
            }
        }

        Ok(WrappedRun {
            command: c,
            peak_source: PeakSource::None,
            teardown: TeardownHandle::default(),
        })
    }
}
