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
//!
//! A stopped unit's processes get SIGTERM and, 10 s later, SIGKILL
//! (`TimeoutStopSec=10s`). The unit survives a SIGKILL of the process that
//! started it; `LocalSystemd::spawn_contained` records it with the BLUT job so
//! `blut cancel` can still stop it (`jobs::cancel_systemd_units`).

use std::path::Path;
use std::sync::OnceLock;

use super::memsize::{parse_memory_peak_line, parse_unit_result_oom, resolve_mem_knob};
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

/// The first systemd release whose `systemd-run --setenv=NAME` (no `=VALUE`)
/// imports the value from systemd-run's own environment.
pub(crate) const SETENV_BY_NAME_SINCE: u32 = 250;

/// The installed `systemd-run`'s version (`systemd 262 (262.1-1-arch)` →
/// 262), probed once. `None` when it cannot be run or parsed.
pub(crate) fn systemd_run_version() -> Option<u32> {
    static VERSION: OnceLock<Option<u32>> = OnceLock::new();
    *VERSION.get_or_init(|| {
        let out = std::process::Command::new("systemd-run")
            .arg("--version")
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .ok()?;
        parse_systemd_version(&String::from_utf8_lossy(&out.stdout))
    })
}

fn parse_systemd_version(text: &str) -> Option<u32> {
    let mut words = text.lines().next()?.split_whitespace();
    (words.next()? == "systemd").then_some(())?;
    words.next()?.parse().ok()
}

/// Whether `systemd-run --setenv=NAME` imports values by name here.
pub(crate) fn imports_env_by_name() -> bool {
    systemd_run_version().is_some_and(|v| v >= SETENV_BY_NAME_SINCE)
}

/// The first systemd release with `systemd-run --expand-environment=`.
pub(crate) const EXPAND_ENVIRONMENT_SINCE: u32 = 254;

/// Whether this systemd-run can pass a service's arguments unexpanded.
fn expansion_can_be_disabled() -> bool {
    systemd_run_version().is_some_and(|v| v >= EXPAND_ENVIRONMENT_SINCE)
}

/// An argument that survives `ExecStart=` expansion unchanged: every `$`
/// doubled, which the expansion turns back into one.
fn escape_dollars(arg: &str) -> String {
    arg.replace('$', "$$")
}

/// Variables systemd sets for a service itself, or that describe the
/// caller's own service: never forwarded.
const NOT_FORWARDED: &[&str] = &[
    "INVOCATION_ID",
    "JOURNAL_STREAM",
    "LISTEN_FDNAMES",
    "LISTEN_FDS",
    "LISTEN_PID",
    "MAINPID",
    "MANAGERPID",
    "MEMORY_PRESSURE_WATCH",
    "MEMORY_PRESSURE_WRITE",
    "NOTIFY_SOCKET",
    "SYSTEMD_EXEC_PID",
    "WATCHDOG_PID",
    "WATCHDOG_USEC",
];

/// This process's environment as a list a transient service can be given,
/// with `extra` added last (an `extra` name replaces the inherited value).
///
/// A `systemd-run --user` service starts from the user manager's minimal
/// environment, not the caller's; every other backend's child inherits the
/// caller's. Passing the caller's environment explicitly makes a
/// systemd-contained stage see what a directly spawned one would. Skipped:
/// variables that are not valid UTF-8, whose name is not a portable shell
/// identifier (exported shell functions such as `BASH_FUNC_x%%`), whose
/// value holds a control character other than tab or newline (systemd refuses
/// the assignment), and the ones systemd manages ([`NOT_FORWARDED`]).
pub(crate) fn inherited_env(extra: &[(String, String)]) -> Vec<(String, String)> {
    let valid_name = |k: &str| {
        let mut chars = k.chars();
        chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    let valid_value = |v: &str| !v.chars().any(|c| c.is_control() && c != '\t' && c != '\n');
    let mut out: Vec<(String, String)> = std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
        .filter(|(k, v)| {
            valid_name(k)
                && valid_value(v)
                && !NOT_FORWARDED.contains(&k.as_str())
                && !extra.iter().any(|(e, _)| e == k)
        })
        .collect();
    out.sort();
    out.extend(extra.iter().cloned());
    out
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
        self.wrap_described(program, args, cwd, env, unit, caps, None)
    }

    fn parse_peak(&self, line: &str) -> Option<u64> {
        parse_memory_peak_line(line)
    }

    fn parse_oom_kill(&self, line: &str) -> Option<bool> {
        parse_unit_result_oom(line)
    }

    fn cancel(&self, unit: &str, _teardown: &TeardownHandle) -> Option<tokio::process::Command> {
        // Stop the transient unit; the cgroup reaps the whole tree. Processes
        // that ignore SIGTERM are SIGKILLed after STOP_TIMEOUT.
        let mut c = tokio::process::Command::new("systemctl");
        c.arg("--user").arg("stop").arg(format!("{unit}.service"));
        Some(c)
    }
}

/// How long a stopped unit's processes get after SIGTERM before systemd
/// SIGKILLs them (`TimeoutStopSec=`): the 10 s that the `rlimit`/`bare` path
/// and `blut cancel` give a process group. Without it a unit inherits the user
/// manager's `DefaultTimeoutStopSec=` (90 s unless configured otherwise), so
/// a trainer that ignores SIGTERM lingered that long after a cancel. A
/// transient service has accepted `TimeoutStopSec=` long before systemd 250,
/// this backend's floor.
pub(crate) const STOP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

impl SystemdCgroup {
    /// [`Containment::wrap_command`], with the unit's `Description=` set to
    /// `description` when given (`systemd-run` otherwise describes a unit by
    /// its command line). `LocalSystemd::spawn_contained` names the BLUT job
    /// there, which is how `blut cancel` tells the job's unit from a later
    /// one that got the same name.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn wrap_described(
        &self,
        program: &Path,
        args: &[String],
        cwd: &Path,
        env: &[(String, String)],
        unit: &str,
        caps: &CapSpec,
        description: Option<&str>,
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
            .arg("-p")
            .arg(format!("TimeoutStopSec={}s", STOP_TIMEOUT.as_secs()))
            .arg(format!("--working-directory={}", cwd.display()));
        if let Some(description) = description {
            c.arg(format!("--description={description}"));
        }
        // Propagate PATH + HOME (minimal --user env otherwise), then the
        // invocation env (incl. PYTHONPATH + BLUT_* identity). Where
        // systemd-run can import a variable by name, the value is set on the
        // client's own environment and only the name goes on its command line:
        // a command line is readable by every user on the host, an
        // environment only by its owner.
        let by_name = imports_env_by_name();
        for key in ["PATH", "HOME"] {
            if let Ok(value) = std::env::var(key) {
                if by_name {
                    c.arg(format!("--setenv={key}"));
                } else {
                    c.arg(format!("--setenv={key}={value}"));
                }
            }
        }
        for (k, v) in env {
            if by_name {
                c.env(k, v);
                c.arg(format!("--setenv={k}"));
            } else {
                c.arg(format!("--setenv={k}={v}"));
            }
        }
        // A service's command line is expanded like `ExecStart=`: `$VAR`,
        // `${VAR}` and `$$` are rewritten, so an argument holding a `$` (a
        // JSON spec, a password, a regex) would reach the stage changed.
        // Turn expansion off where systemd-run can; escape `$` elsewhere.
        let verbatim = expansion_can_be_disabled();
        if verbatim {
            c.arg("--expand-environment=no");
        }
        // Terminator, then the actual kernel command (program + full argv).
        c.arg("--");
        match program.to_str() {
            Some(p) if !verbatim => c.arg(escape_dollars(p)),
            _ => c.arg(program),
        };
        for a in args {
            if verbatim {
                c.arg(a);
            } else {
                c.arg(escape_dollars(a));
            }
        }
        Ok(WrappedRun {
            command: c,
            peak_source: PeakSource::StderrLine,
            teardown: TeardownHandle::default(),
        })
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
        // A cancel escalates to SIGKILL after 10 s, not the manager's default.
        let timeout = argv
            .iter()
            .position(|a| a == "TimeoutStopSec=10s")
            .expect("TimeoutStopSec set");
        assert_eq!(argv[timeout - 1], "-p");
        // systemd-run's own description (the command line) unless given one.
        assert!(!argv.iter().any(|a| a.starts_with("--description")));
        if imports_env_by_name() {
            // Name only on the command line; the value rides the client's env.
            assert!(argv.contains(&"--setenv=PYTHONPATH".to_string()));
            assert!(
                !argv.iter().any(|a| a.contains("/x")),
                "value leaked: {argv:?}"
            );
            let env: Vec<_> = std
                .get_envs()
                .map(|(k, v)| (k.to_os_string(), v.map(|v| v.to_os_string())))
                .collect();
            assert!(env.contains(&("PYTHONPATH".into(), Some("/x".into()))));
        } else {
            assert!(argv.contains(&"--setenv=PYTHONPATH=/x".to_string()));
        }
        assert_eq!(
            argv.contains(&"--expand-environment=no".to_string()),
            expansion_can_be_disabled()
        );
        // kernel command comes after `--`.
        let dd = argv.iter().position(|a| a == "--").unwrap();
        assert_eq!(argv[dd + 1], "python3");
        assert_eq!(argv[dd + 2], "train.py");
        assert_eq!(wr.peak_source, PeakSource::StderrLine);
    }

    #[test]
    fn a_description_goes_before_the_command() {
        let wr = SystemdCgroup
            .wrap_described(
                &PathBuf::from("sleep"),
                &["--description=not-an-option".into()],
                &PathBuf::from("/tmp/job"),
                &[],
                "blut-job-stage",
                &CapSpec::default(),
                Some("blut job 20261030-000000-000000001"),
            )
            .unwrap();
        let argv: Vec<String> = wr
            .command
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let dd = argv.iter().position(|a| a == "--").unwrap();
        let described = argv
            .iter()
            .position(|a| a == "--description=blut job 20261030-000000-000000001")
            .expect("description set");
        assert!(described < dd, "{argv:?}");
        assert_eq!(argv[dd + 2], "--description=not-an-option");
    }

    #[test]
    fn dollars_are_escaped_for_execstart_expansion() {
        assert_eq!(
            escape_dollars(r#"{"pw":"a$b","x":"$$"}"#),
            r#"{"pw":"a$$b","x":"$$$$"}"#
        );
        assert_eq!(escape_dollars("plain"), "plain");
    }

    #[test]
    fn systemd_versions_parse() {
        assert_eq!(
            parse_systemd_version("systemd 262 (262.1-1-arch)\n+PAM"),
            Some(262)
        );
        assert_eq!(
            parse_systemd_version("systemd 249 (249.11-0ubuntu3)"),
            Some(249)
        );
        assert_eq!(parse_systemd_version("not systemd"), None);
        assert_eq!(parse_systemd_version(""), None);
    }

    #[test]
    fn inherited_env_keeps_the_callers_and_lets_extra_win() {
        // Every test process has PATH; HOME is set on any CI runner.
        let env = inherited_env(&[
            ("PATH".into(), "/override".into()),
            ("BLUT_X".into(), "1".into()),
        ]);
        let paths: Vec<_> = env.iter().filter(|(k, _)| k == "PATH").collect();
        assert_eq!(paths, [&("PATH".to_string(), "/override".to_string())]);
        assert!(env.iter().any(|(k, v)| k == "BLUT_X" && v == "1"));
        assert!(
            env.iter()
                .all(|(k, _)| !NOT_FORWARDED.contains(&k.as_str()))
        );
        assert!(env.iter().all(|(k, _)| !k.contains('%') && !k.is_empty()));
    }

    #[test]
    fn parse_peak_reads_the_line_systemd_262_prints() {
        let be = SystemdCgroup;
        assert_eq!(
            be.parse_peak("                   Memory peak: 2.4M (swap: 0B)"),
            Some((2.4 * 1024.0 * 1024.0) as u64)
        );
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
