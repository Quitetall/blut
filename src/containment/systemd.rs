// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! systemd `--user` + cgroup-v2 containment — the original BLUT path, moved
//! here from the lamquant runner so all backends share one implementation.
//! Runs the kernel inside a transient, memory-capped `systemd-run --user
//! --pipe --wait` unit. Best on a developer systemd Linux box.
//!
//! The availability probe (`systemctl --user show-environment`) is the BUS
//! probe — it is the fix for the old binary-only `which systemd-run` check
//! that reported "available" on cloud boxes with no user bus and then failed.

use std::path::Path;
use std::sync::OnceLock;

use super::memsize::{parse_memory_peak_line, resolve_mem_knob};
use super::{Availability, CapSpec, Containment, PeakSource, TeardownHandle, WrappedRun};
use crate::error::{Result, TrainError};

#[derive(Debug, Default)]
pub struct SystemdCgroup;

/// Probe the systemd `--user` manager once per process (cached). Availability
/// does not change mid-process, so a high-throughput scheduler probes exactly
/// once. Distinguishes `Unavailable` (no `systemd-run` binary) from
/// `BusOffline` (binary present, `--user` bus unreachable — the k8s/CI case).
fn probe() -> Availability {
    static AVAIL: OnceLock<Availability> = OnceLock::new();
    *AVAIL.get_or_init(|| {
        // No binary at all → Unavailable.
        if which::which("systemd-run").is_err() {
            return Availability::Unavailable;
        }
        // Binary present; is the --user bus up? This is the load-bearing
        // probe (the old code checked only the binary).
        let bus_ok = std::process::Command::new("systemctl")
            .args(["--user", "show-environment"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if bus_ok {
            Availability::Present
        } else {
            Availability::BusOffline
        }
    })
}

impl Containment for SystemdCgroup {
    fn kind(&self) -> &'static str {
        "systemd"
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
        unit: &str,
        caps: &CapSpec,
    ) -> Result<WrappedRun> {
        if program.as_os_str().is_empty() {
            return Err(TrainError::other("systemd wrap: empty program"));
        }
        // Resolve each cgroup knob: TYPED bytes (wins) → env → default. The
        // typed field is the load-bearing transport — it MUST NOT travel via
        // the child env (`--setenv`, dead for the `-p MemoryMax=` param).
        let memmax = resolve_mem_knob(caps.mem_max, "MEMMAX", "44G");
        let memhigh = resolve_mem_knob(caps.mem_high, "MEMHIGH", "40G");
        // A SMALL swap ceiling (default 2G) so a residual overshoot OOM-kills
        // the UNIT fast instead of thrashing host swap.
        let swapmax = resolve_mem_knob(caps.swap_max, "SWAPMAX", "2G");

        let mut c = tokio::process::Command::new("systemd-run");
        c.arg("--user")
            .arg("--pipe")
            .arg("--wait")
            .arg("--collect")
            .arg(format!("--unit={unit}"))
            .arg("-p")
            .arg("MemoryAccounting=yes")
            .arg("-p")
            .arg(format!("MemoryMax={memmax}"))
            .arg("-p")
            .arg(format!("MemoryHigh={memhigh}"))
            .arg("-p")
            .arg(format!("MemorySwapMax={swapmax}"))
            .arg(format!("--working-directory={}", cwd.display()));
        // Propagate PATH + HOME (minimal --user env otherwise).
        if let Ok(path) = std::env::var("PATH") {
            c.arg(format!("--setenv=PATH={path}"));
        }
        if let Ok(home) = std::env::var("HOME") {
            c.arg(format!("--setenv=HOME={home}"));
        }
        // Cross the invocation env (incl. PYTHONPATH + BLUT_* identity).
        for (k, v) in env {
            c.arg(format!("--setenv={k}={v}"));
        }
        // Terminator, then the actual kernel command (program + full argv).
        c.arg("--").arg(program);
        for a in args {
            c.arg(a);
        }
        Ok(WrappedRun {
            command: c,
            peak_source: PeakSource::StderrLine,
            teardown: TeardownHandle::default(),
        })
    }

    fn parse_peak(&self, line: &str) -> Option<u64> {
        parse_memory_peak_line(line)
    }

    fn cancel(&self, unit: &str, _teardown: &TeardownHandle) -> Option<tokio::process::Command> {
        // Stop the transient unit; the cgroup reaps the whole tree.
        let mut c = tokio::process::Command::new("systemctl");
        c.arg("--user")
            .arg("stop")
            .arg(format!("{unit}.service"));
        Some(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn wrap_shapes_systemd_run_argv() {
        let be = SystemdCgroup;
        let caps = CapSpec {
            mem_max: Some(8 * 1024 * 1024 * 1024),
            mem_high: None,
            swap_max: None,
        };
        let wr = be
            .wrap_command(
                &PathBuf::from("python3"),
                &["train.py".into(), "--epochs".into(), "5".into()],
                &PathBuf::from("/tmp/job"),
                &[("PYTHONPATH".into(), "/x".into())],
                "blut-job-stage",
                &caps,
            )
            .unwrap();
        let std = wr.command.as_std();
        assert_eq!(std.get_program(), "systemd-run");
        let argv: Vec<String> = std
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(argv.contains(&"--user".to_string()));
        assert!(argv.contains(&"--unit=blut-job-stage".to_string()));
        assert!(argv.contains(&"MemoryMax=8G".to_string()));
        assert!(argv.contains(&"--setenv=PYTHONPATH=/x".to_string()));
        // kernel command comes after `--`.
        let dd = argv.iter().position(|a| a == "--").unwrap();
        assert_eq!(argv[dd + 1], "python3");
        assert_eq!(argv[dd + 2], "train.py");
        assert_eq!(wr.peak_source, PeakSource::StderrLine);
    }

    #[test]
    fn parse_peak_delegates() {
        let be = SystemdCgroup;
        assert_eq!(
            be.parse_peak("Memory peak: 4G"),
            Some(4 * 1024 * 1024 * 1024)
        );
        assert_eq!(be.parse_peak("nope"), None);
    }
}
