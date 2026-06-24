// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Pluggable process-containment — the "enforce the memory cap" half of the
//! never-OOM guarantee (the broker is the "suggest the cap" half).
//!
//! BLUT's original containment was hardcoded to `systemd-run --user` + cgroup
//! v2. That works on a developer's systemd Linux box but FAILS on cloud /
//! container hosts (k8s pods, CI runners) where the `systemd-run` binary is on
//! PATH but the `--user` bus is offline — the old `containment_available()`
//! probed the BINARY, not the BUS, so it reported "available" and then died at
//! runtime. This module fixes that and makes containment a TRAIT with several
//! backends, chosen by a probing factory:
//!
//!   - [`systemd::SystemdCgroup`] — the original path (systemd `--user` +
//!     cgroup-v2 `MemoryMax`). Best on a developer Linux box.
//!   - [`cgroup2::CgroupV2Direct`] — write a delegated cgroup-v2 subtree
//!     directly, no systemd. Works on cloud boxes that delegate the subtree.
//!   - [`rlimit::RlimitAddressSpace`] — a portable `setrlimit(RLIMIT_AS)` cap
//!     via a `pre_exec` hook. No root, no cgroup, no systemd; works on
//!     locked-down containers and macOS. Weaker (caps address space, not RSS
//!     of the whole tree) but a REAL cap.
//!   - [`bare::Bare`] — no cap; the admission gate is the only floor. Last
//!     resort, loud warning.
//!   - [`windows::WindowsJobObject`] — stub for a future Windows Job Object
//!     cap (`#[cfg(windows)]`).
//!
//! The `Launcher` trait (in `config::launcher`) is ORTHOGONAL: it answers
//! "which MACHINE runs this" (local / Slurm / Ray); `Containment` answers "how
//! is THIS-box memory capped". A local run is placed by `LocalSystemd` and
//! capped by whichever `Containment` the factory selects.

#[cfg(target_os = "linux")]
pub mod cgroup2;
pub mod bare;
pub mod memsize;
#[cfg(unix)]
pub mod rlimit;
#[cfg(target_os = "linux")]
pub mod systemd;
#[cfg(windows)]
pub mod windows;

use std::path::{Path, PathBuf};

use crate::error::Result;

/// Probe result for a containment backend. The three-state shape is the BUG
/// FIX: the old `bool` conflated "mechanism present and usable" with "mechanism
/// present but unusable right now", so a box with the `systemd-run` binary but
/// no user bus reported `true` and then failed at runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Availability {
    /// Fully usable (systemd bus up / cgroup subtree writable / etc.).
    Present,
    /// Mechanism present but unusable NOW: `systemd-run` on PATH but the
    /// `--user` bus is offline; cgroup-v2 mounted but the subtree is read-only.
    /// The factory treats this as "not this backend — try the next", and a
    /// genuinely-contained run with a derivable unit REFUSES (never silently
    /// runs uncapped) — see the runner's fail-closed integration.
    BusOffline,
    /// Mechanism absent: no `systemd-run`, no cgroup-v2 mount, wrong OS.
    Unavailable,
}

/// The memory knobs, in BYTES. Each backend formats / applies them in its
/// native units (systemd memsize string, cgroup decimal bytes, `RLIMIT_AS`
/// rlim_t). `None` = leave that knob at the system default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapSpec {
    /// Hard cap → the process/tree is OOM-killed if it exceeds this.
    pub mem_max: Option<u64>,
    /// Soft throttle (reclaim pressure). Not all backends honor it.
    pub mem_high: Option<u64>,
    /// Swap ceiling (small = fail-fast instead of thrash). cgroup/systemd only.
    pub swap_max: Option<u64>,
}

/// Where this backend's peak-RAM measurement comes from, so the runner knows
/// whether to scrape a stderr line, read a file post-wait, or record nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeakSource {
    /// systemd-run prints `Memory peak: <N>` on stderr at unit stop → the
    /// runner's stderr pump scrapes it via [`Containment::parse_peak`].
    StderrLine,
    /// Read a file (cgroup `memory.peak` / Windows `PeakJobMemoryUsed`) AFTER
    /// wait(), via [`Containment::read_peak`].
    CgroupFile,
    /// No trustworthy peak (Bare / RLIMIT_AS). The runner records Uncontained.
    None,
}

/// Backend-owned teardown state, threaded back to `cancel`/`read_peak`/
/// `cleanup`. systemd keys off the unit name (the runner already has it);
/// cgroup2 needs the created leaf dir; rlimit/bare need nothing.
#[derive(Clone, Debug, Default)]
pub struct TeardownHandle {
    /// The cgroup-v2 leaf directory created for this run (cgroup2 backend).
    pub cgroup_path: Option<PathBuf>,
}

/// The constructed child command + the metadata the runner needs to manage it.
/// The runner sets pipes / `pre_exec_setsid` / `kill_on_drop` on `command`
/// exactly as before — the backend only constructs it (and may install an
/// ADDITIONAL `pre_exec` hook, e.g. cgroup-join / setrlimit).
pub struct WrappedRun {
    pub command: tokio::process::Command,
    pub peak_source: PeakSource,
    pub teardown: TeardownHandle,
}

/// A pluggable process-containment mechanism. Object-safe (no `async_trait`):
/// `cancel` returns the teardown COMMAND for the runner to spawn, rather than
/// being `async` itself. Construction-only — the runner owns the process.
pub trait Containment: Send + Sync {
    /// Stable id for logging / the `BLUT_CONTAINMENT` echo.
    fn kind(&self) -> &'static str;

    /// Probe usability. Should be cheap + internally cached where it shells
    /// out (the user-manager / cgroup-delegation state doesn't change
    /// mid-process).
    fn available(&self) -> Availability;

    /// Build the child command + teardown metadata. `program` is the executable
    /// (e.g. `python3`); `args` is the full argv after it (e.g.
    /// `[script, ...]` or `[-m, torch.distributed.run, ..., script, ...]` for a
    /// DDP run). `unit` is the runner-derived identity (`blut-<job>-<stage>`),
    /// already sanitized to `[A-Za-z0-9_-]`. `env` carries `PYTHONPATH` +
    /// `BLUT_*` identity that must reach the child. Errors if the backend can't
    /// construct (e.g. cgroup mkdir/delegation refused).
    fn wrap_command(
        &self,
        program: &Path,
        args: &[String],
        cwd: &Path,
        env: &[(String, String)],
        unit: &str,
        caps: &CapSpec,
    ) -> Result<WrappedRun>;

    /// Parse a peak from a stderr line (StderrLine backends). Default `None`.
    fn parse_peak(&self, _line: &str) -> Option<u64> {
        None
    }

    /// Read the peak post-wait for CgroupFile backends (reads `memory.peak`).
    /// Default `None`.
    fn read_peak(&self, _teardown: &TeardownHandle) -> Option<u64> {
        None
    }

    /// PRIMARY teardown: the command that kills the contained tree
    /// (`systemctl --user stop <unit>` / a `cgroup.kill` write). `None` when
    /// the backend has no out-of-band teardown (rlimit/bare) — the runner's
    /// `killpg` fallback handles those. Spawned by the runner; best-effort.
    fn cancel(&self, _unit: &str, _teardown: &TeardownHandle) -> Option<tokio::process::Command> {
        None
    }

    /// Remove the per-run resource after wait() (rmdir the empty cgroup leaf).
    /// No-op for systemd (`--collect` reaps), rlimit, and bare. Best-effort.
    fn cleanup(&self, _teardown: &TeardownHandle) {}
}

/// Select a containment backend. Probing order (the `auto` default):
///   `BLUT_NO_CONTAIN=1`            → [`bare::Bare`]
///   `BLUT_CONTAINMENT=<kind>`      → that backend, NO fallback (an explicit
///                                    name that's unusable surfaces as the
///                                    runner's fail-closed refusal — loud).
///   else AUTO: systemd-bus-Present → cgroup2-writable → rlimit → bare.
///
/// On a developer systemd box this is `SystemdCgroup`; on a cloud box with a
/// delegated cgroup it's `CgroupV2Direct`; on a locked-down container (this is
/// the common rented-GPU case) it's `RlimitAddressSpace` — a REAL cap, not
/// Bare; only a box with none of the above falls to `Bare` with a warning.
pub fn containment_for() -> Box<dyn Containment> {
    if std::env::var_os("BLUT_NO_CONTAIN").is_some() {
        return Box::new(bare::Bare);
    }

    match std::env::var("BLUT_CONTAINMENT").as_deref() {
        Ok("bare") => return Box::new(bare::Bare),
        #[cfg(target_os = "linux")]
        Ok("systemd") => return Box::new(systemd::SystemdCgroup),
        #[cfg(target_os = "linux")]
        Ok("cgroup2") => return Box::new(cgroup2::CgroupV2Direct),
        #[cfg(unix)]
        Ok("rlimit") => return Box::new(rlimit::RlimitAddressSpace),
        Ok("auto") | Err(_) => {}
        Ok(other) => {
            tracing::warn!("unknown BLUT_CONTAINMENT='{other}', using auto");
        }
    }

    // AUTO probe order. `Present` selects; `BusOffline`/`Unavailable` fall on.
    #[cfg(target_os = "linux")]
    {
        let sd = systemd::SystemdCgroup;
        if sd.available() == Availability::Present {
            return Box::new(sd);
        }
        let cg = cgroup2::CgroupV2Direct;
        if cg.available() == Availability::Present {
            tracing::info!(
                "containment: systemd --user bus unavailable; using CgroupV2Direct"
            );
            return Box::new(cg);
        }
    }
    #[cfg(unix)]
    {
        let rl = rlimit::RlimitAddressSpace;
        if rl.available() == Availability::Present {
            tracing::info!(
                "containment: no systemd bus / writable cgroup; using RLIMIT_AS \
                 (address-space cap)"
            );
            return Box::new(rl);
        }
    }

    tracing::warn!(
        "containment: no systemd --user bus, no writable cgroup-v2 subtree, and \
         no RLIMIT_AS — running BARE (admission-gated but NOT memory-capped; an \
         oversized run can OOM the box). Set BLUT_CONTAINMENT or fix delegation."
    );
    Box::new(bare::Bare)
}
