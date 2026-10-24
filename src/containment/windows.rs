// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Windows Job Object containment: one kernel Job Object per run holds the
//! child and every process it spawns, caps their combined committed memory,
//! reports the tree's peak, and kills the whole tree in one call.
//!
//! Mechanism:
//!   - [`Containment::wrap_command`] calls `CreateJobObjectW` and sets
//!     `JOBOBJECT_EXTENDED_LIMIT_INFORMATION` with
//!     `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (closing the last job handle kills
//!     every process still in the job),
//!     `JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION` (a crash exits instead of
//!     parking on a Windows Error Reporting dialog), and, when a cap resolves,
//!     `JOB_OBJECT_LIMIT_JOB_MEMORY` with `JobMemoryLimit`. The command is
//!     built with `CREATE_SUSPENDED`.
//!   - [`Containment::after_spawn`] runs `AssignProcessToJobObject` on the
//!     still-suspended child, then resumes its initial thread. The child
//!     executes no instruction outside the job, so nothing it spawns can
//!     escape the job. The caller MUST call `after_spawn` (or use
//!     [`super::spawn_attached`]); a child that is never attached stays
//!     suspended, so a missed call hangs rather than running uncapped.
//!   - [`Containment::cancel`] = `TerminateJobObject`, the analogue of
//!     cgroup-v2 `cgroup.kill`: it kills every process in the job at once.
//!   - [`Containment::read_peak`] = `QueryInformationJobObject(...)
//!     .PeakJobMemoryUsed`, read after `wait()` ([`PeakSource::CgroupFile`]).
//!
//! # What the cap measures
//!
//! `JobMemoryLimit` bounds the COMMITTED private memory of all processes in
//! the job together (RAM plus the pagefile share), not resident set size. It
//! is tree-wide like cgroup `memory.max`, but on overrun Windows FAILS the
//! allocation (`VirtualAlloc` returns an error, Python raises `MemoryError`)
//! rather than killing the tree; the process dies only if it does not handle
//! the failure. `PeakJobMemoryUsed` is the same quantity's high-water mark.
//!
//! # `CapSpec` mapping
//!
//! | `CapSpec` field | Windows | Notes |
//! |---|---|---|
//! | `mem_max` | ENFORCED as `JobMemoryLimit` | typed value, else `$MEMMAX`; no built-in default, because a commit cap is not an RSS cap (see `rlimit` for the same reasoning) |
//! | `mem_high` | NOT MAPPED | Job Objects have no soft reclaim throttle |
//! | `swap_max` | NOT MAPPED | the commit cap already counts RAM and pagefile together; there is no separate pagefile knob |
//!
//! `CapSpec` carries no CPU field, so the job sets no
//! `JOBOBJECT_CPU_RATE_CONTROL_INFORMATION`. CPU rate control would be added
//! here when `CapSpec` grows a CPU cap.
//!
//! # Caller obligations
//!
//! The job lives as long as the [`TeardownHandle`] (and its clones). Dropping
//! the last one closes the job handle and, by `KILL_ON_JOB_CLOSE`, kills any
//! process still in the job, so keep the handle until the run is over;
//! [`super::ContainedChild`] holds it for exactly that long. A caller must
//! spawn through [`super::spawn_attached`] (which `ContainedChild` and
//! `LocalSystemd::spawn_contained` use): it starts the child suspended
//! ([`Containment::before_spawn`]), assigns it to the job, then lets it run,
//! so nothing the child starts can escape. A caller that calls
//! `spawn()` itself gets a running child that is NOT in the job (uncapped,
//! and its processes outlive cancellation); when that run's teardown handle is
//! dropped, a warning says so and names `spawn_attached`. That is deliberate:
//! the earlier design left such a child suspended forever, a silent hang.

use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION,
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    QueryInformationJobObject, SetInformationJobObject, TerminateJobObject,
};
use windows_sys::Win32::System::Threading::{
    CREATE_SUSPENDED, OpenThread, ResumeThread, THREAD_SUSPEND_RESUME, TerminateProcess,
};

use super::memsize::parse_memsize;
use super::{Availability, CapSpec, Containment, PeakSource, TeardownHandle, WrappedRun};
use crate::error::{Result, TrainError};

/// Exit code given to processes killed by [`Containment::cancel`] or by a
/// failed attach. Distinct from 0 and 1 so a log shows where a death came from.
const KILLED_EXIT_CODE: u32 = 0xB1_07;

#[derive(Debug, Default)]
pub struct WindowsJobObject;

/// A Job Object handle, closed on drop. Shared through [`TeardownHandle`] by
/// `Arc`, so the job closes (and `KILL_ON_JOB_CLOSE` fires) when the last
/// clone of the handle is dropped.
#[derive(Debug)]
pub(crate) struct JobHandle {
    handle: OwnedHandle,
    /// Set once a child has been assigned to the job by `after_spawn`.
    attached: AtomicBool,
}

impl JobHandle {
    fn raw(&self) -> HANDLE {
        self.handle.as_raw_handle() as HANDLE
    }
}

impl Drop for JobHandle {
    fn drop(&mut self) {
        if !self.attached.load(Ordering::Relaxed) {
            tracing::warn!(
                "windows job object: no child was ever assigned to this run's job, so it \
                 ran without memory containment and outside cancellation; spawn the \
                 wrapped command with blut::containment::spawn_attached"
            );
        }
    }
}

fn last_os_error(what: &str) -> TrainError {
    TrainError::other(format!(
        "windows job object: {what}: {}",
        std::io::Error::last_os_error()
    ))
}

/// Create an anonymous, non-inheritable Job Object.
fn create_job_object() -> Result<OwnedHandle> {
    // SAFETY: null security attributes (non-inheritable handle) and a null
    // name (anonymous job) are both documented inputs. A non-null return is a
    // fresh handle we own; ownership moves into `OwnedHandle`.
    #[allow(unsafe_code)]
    unsafe {
        let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if h.is_null() {
            return Err(last_os_error("CreateJobObjectW"));
        }
        Ok(OwnedHandle::from_raw_handle(h as RawHandle))
    }
}

/// A Job Object for one run. Dropping it unattached warns (see `Drop`).
fn create_job() -> Result<JobHandle> {
    Ok(JobHandle {
        handle: create_job_object()?,
        attached: AtomicBool::new(false),
    })
}

fn query_extended(job: &JobHandle) -> Option<JOBOBJECT_EXTENDED_LIMIT_INFORMATION> {
    // SAFETY: the struct is plain old data, so all-zero is a valid value; the
    // buffer is exactly the size passed, and the job handle is live.
    #[allow(unsafe_code)]
    unsafe {
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        let ok = QueryInformationJobObject(
            job.raw(),
            JobObjectExtendedLimitInformation,
            (&raw mut info).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            std::ptr::null_mut(),
        );
        (ok != 0).then_some(info)
    }
}

/// The hard cap: typed `mem_max` wins, then `$MEMMAX`; otherwise none. A zero
/// is never honored (it would refuse every allocation).
fn resolve_job_memory_limit(caps: &CapSpec) -> Option<u64> {
    caps.mem_max.filter(|&b| b > 0).or_else(|| {
        std::env::var("MEMMAX")
            .ok()
            .and_then(|s| parse_memsize(&s))
            .filter(|&b| b > 0)
    })
}

fn set_limits(job: &JobHandle, job_memory_limit: Option<u64>) -> Result<()> {
    // SAFETY: plain-old-data struct, zero-initialized, then the fields we use
    // are set; the size passed matches the struct; the job handle is live.
    #[allow(unsafe_code)]
    unsafe {
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
        if let Some(limit) = job_memory_limit {
            let limit = usize::try_from(limit).map_err(|_| {
                TrainError::other(format!(
                    "windows job object: memory cap {limit} bytes exceeds this platform's address size"
                ))
            })?;
            info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
            info.JobMemoryLimit = limit;
        }
        let ok = SetInformationJobObject(
            job.raw(),
            JobObjectExtendedLimitInformation,
            (&raw const info).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok == 0 {
            return Err(last_os_error("SetInformationJobObject"));
        }
    }
    Ok(())
}

/// Resume every thread of `pid`. Called only on a child created with
/// `CREATE_SUSPENDED` and not yet resumed, which has exactly one thread (its
/// initial thread); std does not expose that thread's handle, so it is found
/// through a ToolHelp thread snapshot. Returns how many threads were resumed.
fn resume_process_threads(pid: u32) -> Result<usize> {
    // SAFETY: every handle opened here is checked and closed on every path;
    // THREADENTRY32 is plain old data with `dwSize` set as the API requires.
    #[allow(unsafe_code)]
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return Err(last_os_error("CreateToolhelp32Snapshot"));
        }
        let mut entry: THREADENTRY32 = std::mem::zeroed();
        entry.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut resumed = 0usize;
        let mut failure = None;
        let mut more = Thread32First(snap, &mut entry) != 0;
        while more {
            if entry.th32OwnerProcessID == pid {
                let th = OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID);
                if th.is_null() {
                    failure = Some(last_os_error("OpenThread"));
                } else {
                    if ResumeThread(th) == u32::MAX {
                        failure = Some(last_os_error("ResumeThread"));
                    } else {
                        resumed += 1;
                    }
                    CloseHandle(th);
                }
            }
            more = Thread32Next(snap, &mut entry) != 0;
        }
        CloseHandle(snap);
        match (resumed, failure) {
            (0, Some(e)) => Err(e),
            (0, None) => Err(TrainError::other(format!(
                "windows job object: no threads found for suspended child pid {pid}"
            ))),
            _ => Ok(resumed),
        }
    }
}

fn probe() -> Availability {
    static AVAIL: OnceLock<Availability> = OnceLock::new();
    // Creating (and closing) a job is the whole capability check: nested
    // jobs are supported since Windows 8, so a process that is itself inside
    // a job (a CI runner, a service) can still create and assign its own. A
    // bare handle, not a `JobHandle`: the probe's job is never meant to hold a
    // child, so it must not raise the "never assigned" warning a run's job
    // raises (it did, once per process, ahead of every first contained spawn).
    *AVAIL.get_or_init(|| match create_job_object() {
        Ok(_) => Availability::Present,
        Err(e) => {
            tracing::warn!("containment: Job Objects unavailable: {e}");
            Availability::Unavailable
        }
    })
}

impl Containment for WindowsJobObject {
    fn kind(&self) -> &'static str {
        "windows-job-object"
    }

    fn available(&self) -> Availability {
        probe()
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
            return Err(TrainError::other("windows job object wrap: empty program"));
        }
        let job = create_job()?;
        set_limits(&job, resolve_job_memory_limit(caps))?;

        let mut c = tokio::process::Command::new(program);
        for a in args {
            c.arg(a);
        }
        c.current_dir(cwd);
        for (k, v) in env {
            c.env(k, v);
        }
        // Not suspended here: `before_spawn` (called by `spawn_attached`)
        // adds CREATE_SUSPENDED, so a caller that spawns directly gets a
        // running child rather than one frozen forever.

        Ok(WrappedRun {
            command: c,
            peak_source: PeakSource::CgroupFile,
            teardown: TeardownHandle {
                job: Some(Arc::new(job)),
                ..TeardownHandle::default()
            },
        })
    }

    fn before_spawn(&self, command: &mut tokio::process::Command) {
        // Assign before the child runs a single instruction.
        command.creation_flags(CREATE_SUSPENDED);
    }

    fn after_spawn(&self, child: &tokio::process::Child, teardown: &TeardownHandle) -> Result<()> {
        let job = teardown
            .job
            .as_ref()
            .ok_or_else(|| TrainError::other("windows job object: teardown carries no job"))?;
        let (Some(process), Some(pid)) = (child.raw_handle(), child.id()) else {
            return Err(TrainError::other(
                "windows job object: child already reaped before it could be attached",
            ));
        };
        let process = process as HANDLE;
        // SAFETY: `process` is the live child's handle, owned by `child` and
        // valid for this call; the job handle is live.
        #[allow(unsafe_code)]
        let assigned = unsafe { AssignProcessToJobObject(job.raw(), process) } != 0;
        let attached = if assigned {
            job.attached.store(true, Ordering::Relaxed);
            resume_process_threads(pid).map(|_| ())
        } else {
            Err(last_os_error("AssignProcessToJobObject"))
        };
        if attached.is_err() {
            // Never leave a suspended child behind, and never resume one that
            // is outside the job: it has run nothing, so kill it here.
            // SAFETY: same live child handle as above.
            #[allow(unsafe_code)]
            unsafe {
                TerminateProcess(process, KILLED_EXIT_CODE);
            }
        }
        attached
    }

    fn read_peak(&self, teardown: &TeardownHandle) -> Option<u64> {
        let info = query_extended(teardown.job.as_ref()?)?;
        let peak = info.PeakJobMemoryUsed as u64;
        (peak > 0).then_some(peak)
    }

    fn cancel(&self, _unit: &str, teardown: &TeardownHandle) -> Option<tokio::process::Command> {
        if let Some(job) = &teardown.job {
            // Kills every process in the job at once, done inline: there is
            // no command for the runner to spawn. Best-effort, like cgroup2's
            // `cgroup.kill`; the job closing later is the backstop.
            // SAFETY: the job handle is live for the borrow of `teardown`.
            #[allow(unsafe_code)]
            let ok = unsafe { TerminateJobObject(job.raw(), KILLED_EXIT_CODE) } != 0;
            if !ok {
                tracing::warn!(
                    "containment: TerminateJobObject failed: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
        None
    }

    // `cleanup` stays the default no-op: the job is released when the last
    // `TeardownHandle` clone drops, and `KILL_ON_JOB_CLOSE` then reaps any
    // straggler, which is the Windows analogue of removing the cgroup leaf.
}
