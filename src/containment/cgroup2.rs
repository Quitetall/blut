// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Direct cgroup-v2 containment — no systemd. Creates a leaf cgroup under a
//! writable, delegated subtree, writes the memory caps, joins the child into
//! the leaf via a `pre_exec` hook (so the child + ALL its fork workers are in
//! one cgroup), reads the true peak from `memory.peak`, and tears the tree
//! down atomically via `cgroup.kill`. This is the strongest cap on a cloud box
//! that delegates a cgroup subtree to the container but has no systemd `--user`
//! bus.
//!
//! Requires: cgroup-v2 unified mount, the `memory` controller, and a WRITABLE
//! delegated subtree (the pod/container's own cgroup, or `$BLUT_CGROUP_PARENT`).
//! When the subtree is read-only (e.g. an unprivileged k8s pod with no
//! delegation — verified on a Thunder box), `available()` returns `BusOffline`
//! and the factory falls through to RLIMIT_AS.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::memsize::resolve_mem_bytes;
use super::{Availability, CapSpec, Containment, PeakSource, TeardownHandle, WrappedRun};
use crate::error::{Result, TrainError};

const CG_ROOT: &str = "/sys/fs/cgroup";

#[derive(Debug, Default)]
pub struct CgroupV2Direct;

/// The writable base cgroup under which we create per-run leaves. Resolution:
///   1. `$BLUT_CGROUP_PARENT` — an operator-prepared delegated dir.
///   2. the process's own cgroup dir (from `/proc/self/cgroup`), if writable.
/// Returns `None` when neither is writable (→ `BusOffline`).
fn writable_base() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_CGROUP_PARENT") {
        let pb = PathBuf::from(p);
        if dir_writable(&pb) {
            return Some(pb);
        }
    }
    // /proc/self/cgroup is a single `0::<path>` line for v2.
    let self_cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let rel = self_cg
        .lines()
        .find_map(|l| l.strip_prefix("0::"))?
        .trim();
    let base = PathBuf::from(CG_ROOT).join(rel.trim_start_matches('/'));
    if dir_writable(&base) {
        Some(base)
    } else {
        None
    }
}

/// Probe writability by attempting a probe mkdir + rmdir. The cgroupfs is
/// often mounted read-only in unprivileged containers, so a stat is not
/// enough — only an actual mkdir tells the truth.
fn dir_writable(base: &Path) -> bool {
    let probe = base.join("blut-probe-wcheck");
    match std::fs::create_dir(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_dir(&probe);
            true
        }
        Err(_) => false,
    }
}

fn probe() -> Availability {
    static AVAIL: OnceLock<Availability> = OnceLock::new();
    *AVAIL.get_or_init(|| {
        // cgroup-v2 + memory controller present?
        let controllers = match std::fs::read_to_string(format!("{CG_ROOT}/cgroup.controllers")) {
            Ok(s) => s,
            Err(_) => return Availability::Unavailable, // not cgroup-v2
        };
        if !controllers.split_whitespace().any(|c| c == "memory") {
            return Availability::Unavailable;
        }
        // Mounted, but is a subtree writable (delegated)?
        if writable_base().is_some() {
            Availability::Present
        } else {
            Availability::BusOffline
        }
    })
}

/// Ensure the `memory` controller is enabled in the base's `subtree_control`
/// so child leaves can set `memory.max`. Best-effort: if it's already enabled
/// (or we lack permission), proceed — the cap write will surface a real error.
fn enable_memory_subtree(base: &Path) {
    let f = base.join("cgroup.subtree_control");
    if let Ok(cur) = std::fs::read_to_string(&f) {
        if cur.split_whitespace().any(|c| c == "memory") {
            return;
        }
    }
    let _ = std::fs::write(&f, b"+memory");
}

impl Containment for CgroupV2Direct {
    fn kind(&self) -> &'static str {
        "cgroup2"
    }

    fn available(&self) -> Availability {
        probe()
    }

    fn wrap_command(
        &self,
        program: &Path,
        script: &Path,
        args: &[String],
        cwd: &Path,
        env: &[(String, String)],
        unit: &str,
        caps: &CapSpec,
    ) -> Result<WrappedRun> {
        if program.as_os_str().is_empty() {
            return Err(TrainError::other("cgroup2 wrap: empty program"));
        }
        let base = writable_base()
            .ok_or_else(|| TrainError::other("cgroup2: no writable delegated subtree"))?;
        enable_memory_subtree(&base);

        // `unit` is already sanitized to [A-Za-z0-9_-] by the runner → safe dir.
        let cgdir = base.join(unit);
        std::fs::create_dir_all(&cgdir)
            .map_err(|e| TrainError::other(format!("cgroup2 mkdir {}: {e}", cgdir.display())))?;

        // Write the caps. cgroup-v2 memory files take a decimal byte count or
        // the literal `max`. Resolve TYPED → env → default (RSS-based, so a
        // default is safe — unlike RLIMIT_AS). Reject 0 (a 0-byte hard cap =
        // instant kill) — leave the knob at `max` instead.
        if let Some(b) = resolve_mem_bytes(caps.mem_max, "MEMMAX", "44G") {
            write_knob(&cgdir, "memory.max", b)?;
        }
        if let Some(b) = resolve_mem_bytes(caps.mem_high, "MEMHIGH", "40G") {
            // soft throttle — best-effort, don't fail the run if absent.
            let _ = std::fs::write(cgdir.join("memory.high"), b.to_string());
        }
        if let Some(b) = resolve_mem_bytes(caps.swap_max, "SWAPMAX", "2G") {
            let _ = std::fs::write(cgdir.join("memory.swap.max"), b.to_string());
        }
        // Kill the whole leaf as a unit on OOM (matches systemd MemoryMax
        // semantics where the unit dies, not just one worker). Best-effort.
        let _ = std::fs::write(cgdir.join("memory.oom.group"), b"1");

        let mut c = tokio::process::Command::new(program);
        c.arg(script);
        for a in args {
            c.arg(a);
        }
        c.current_dir(cwd);
        for (k, v) in env {
            c.env(k, v);
        }

        // Join the child into the leaf cgroup BEFORE exec, so the python
        // process AND every fork worker it spawns are inside the cap.
        let procs_path = cgdir.join("cgroup.procs");
        // SAFETY: runs in the forked child between fork and exec. Writes the
        // child's own pid (formatted into a stack buffer, no heap) to
        // cgroup.procs via raw libc open/write — async-signal-safe. Composes
        // after pre_exec_setsid (the runner installs setsid first).
        #[allow(unsafe_code)]
        unsafe {
            c.pre_exec(move || join_cgroup(&procs_path));
        }

        Ok(WrappedRun {
            command: c,
            peak_source: PeakSource::CgroupFile,
            teardown: TeardownHandle {
                cgroup_path: Some(cgdir),
            },
        })
    }

    fn read_peak(&self, teardown: &TeardownHandle) -> Option<u64> {
        let dir = teardown.cgroup_path.as_ref()?;
        // memory.peak (kernel ≥5.19) is the high-water RSS of the leaf.
        let s = std::fs::read_to_string(dir.join("memory.peak")).ok()?;
        s.trim().parse::<u64>().ok()
    }

    fn cancel(&self, _unit: &str, teardown: &TeardownHandle) -> Option<tokio::process::Command> {
        let dir = teardown.cgroup_path.as_ref()?;
        // Write `1` to `cgroup.kill` DIRECTLY (no `sh -c` — avoids any shell
        // injection from the path, defense-in-depth even though `unit` is
        // sanitized upstream). This SIGKILLs every process in the subtree
        // atomically (kernel ≥5.14) — the cgroup analogue of `systemctl stop`,
        // immune to the pid-reuse race that killpg guards against. Best-effort;
        // the runner's killpg is the fallback. Returns None — the kill is done
        // inline, there is no command for the runner to spawn.
        let _ = std::fs::write(dir.join("cgroup.kill"), b"1");
        None
    }

    fn cleanup(&self, teardown: &TeardownHandle) {
        if let Some(dir) = &teardown.cgroup_path {
            // rmdir only succeeds when the cgroup is empty (all procs gone);
            // best-effort — a leaked empty leaf is harmless.
            let _ = std::fs::remove_dir(dir);
        }
    }
}

fn write_knob(cgdir: &Path, file: &str, bytes: u64) -> Result<()> {
    std::fs::write(cgdir.join(file), bytes.to_string())
        .map_err(|e| TrainError::other(format!("cgroup2 write {file}: {e}")))
}

/// Async-signal-safe cgroup-join: open `cgroup.procs` and write our own pid.
/// No heap allocation — the pid is formatted into a fixed stack buffer.
#[cfg(target_os = "linux")]
fn join_cgroup(procs_path: &Path) -> std::io::Result<()> {
    use std::io::Write;
    // Runs in the forked child between fork and exec (single-threaded there, so
    // the std open+write are safe in this context — no other thread can observe
    // a half-state). Write our own pid to cgroup.procs to join the leaf.
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .open(procs_path)
        .map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("cgroup2 join {}: {e}", procs_path.display()),
            )
        })?;
    // getpid is async-signal-safe; itoa via a stack buffer (no alloc).
    let pid = nix::unistd::getpid().as_raw();
    let mut buf = [0u8; 24];
    let s = fmt_i32(pid, &mut buf);
    f.write_all(s)?;
    Ok(())
}

/// Format an i32 into a stack buffer, returning the written slice. No heap.
#[cfg(target_os = "linux")]
fn fmt_i32(n: i32, buf: &mut [u8; 24]) -> &[u8] {
    // pids are non-negative, but be defensive.
    let neg = n < 0;
    let mut i = buf.len();
    if n == 0 {
        i -= 1;
        buf[i] = b'0';
    } else {
        let mut v = (n as i64).unsigned_abs();
        while v > 0 {
            i -= 1;
            buf[i] = b'0' + (v % 10) as u8;
            v /= 10;
        }
    }
    if neg {
        i -= 1;
        buf[i] = b'-';
    }
    &buf[i..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_i32_basic() {
        let mut b = [0u8; 24];
        assert_eq!(fmt_i32(0, &mut b), b"0");
        let mut b = [0u8; 24];
        assert_eq!(fmt_i32(12345, &mut b), b"12345");
        let mut b = [0u8; 24];
        assert_eq!(fmt_i32(2147483647, &mut b), b"2147483647");
    }

    #[test]
    fn available_is_three_state() {
        // On CI/dev this is Present, BusOffline, or Unavailable — never panics.
        let a = CgroupV2Direct.available();
        assert!(matches!(
            a,
            Availability::Present | Availability::BusOffline | Availability::Unavailable
        ));
    }
}
