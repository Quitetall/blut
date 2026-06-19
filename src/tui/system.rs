// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! System-probe panel data for `blut tui`.
//!
//! Cheap once-per-tick reads: `/proc/meminfo`, `/proc/loadavg`,
//! `nvidia-smi --query-gpu`, and `df` of the blut state filesystem. All
//! best-effort — a missing tool / unreadable file degrades to a single
//! "n/a" line rather than failing the TUI loop.

use std::process::Command;

#[derive(Clone, Debug, Default)]
pub(super) struct SystemSnapshot {
    pub gpu_mem_used_mib: Option<u64>,
    pub gpu_mem_total_mib: Option<u64>,
    pub gpu_util_pct: Option<u32>,
    pub gpu_name: Option<String>,
    pub mem_total_gb: f64,
    pub mem_used_gb: f64,
    pub mem_avail_gb: f64,
    /// The path whose filesystem the disk panel reports (for the panel label).
    pub disk_path: String,
    pub disk_free_human: String,
    pub disk_used_pct: u32,
    pub load1: f64,
}

impl SystemSnapshot {
    pub(super) fn probe() -> Self {
        let disk_path = disk_probe_path();
        Self {
            gpu_mem_used_mib: probe_gpu_field("memory.used").and_then(|s| s.parse().ok()),
            gpu_mem_total_mib: probe_gpu_field("memory.total").and_then(|s| s.parse().ok()),
            gpu_util_pct: probe_gpu_field("utilization.gpu").and_then(|s| s.parse().ok()),
            gpu_name: probe_gpu_field("name"),
            mem_total_gb: probe_meminfo_kib("MemTotal:") / 1024.0 / 1024.0,
            mem_used_gb: {
                let total = probe_meminfo_kib("MemTotal:");
                let avail = probe_meminfo_kib("MemAvailable:");
                ((total - avail).max(0.0)) / 1024.0 / 1024.0
            },
            mem_avail_gb: probe_meminfo_kib("MemAvailable:") / 1024.0 / 1024.0,
            disk_free_human: probe_disk_free_human(&disk_path),
            disk_used_pct: probe_disk_used_pct(&disk_path),
            disk_path,
            load1: probe_loadavg(),
        }
    }

    pub(super) fn gpu_summary(&self) -> String {
        match (
            self.gpu_mem_used_mib,
            self.gpu_mem_total_mib,
            self.gpu_util_pct,
        ) {
            (Some(u), Some(t), Some(util)) => format!(
                "{} {} / {} MiB  util {}%",
                self.gpu_name.clone().unwrap_or_else(|| "GPU".into()),
                u,
                t,
                util
            ),
            _ => "n/a (nvidia-smi missing?)".into(),
        }
    }
}

/// The path whose filesystem the disk panel reports. Never a hardcoded machine
/// path, so a stranger sees their own layout. Resolution order:
///   1. `$BLUT_DISK_PATH` — explicit override.
///   2. `$BLUT_HOME` — the engine's home, if set.
///   3. the job-state dir (`paths::jobs_dir`) — the filesystem blut writes to.
///   4. `.` (current dir) — last resort.
fn disk_probe_path() -> String {
    if let Ok(p) = std::env::var("BLUT_DISK_PATH") {
        return p;
    }
    if let Ok(p) = std::env::var("BLUT_HOME") {
        return p;
    }
    if let Ok(p) = crate::paths::jobs_dir() {
        return p.to_string_lossy().into_owned();
    }
    ".".to_string()
}

fn probe_gpu_field(field: &str) -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={field}"),
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines()
        .next()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
}

fn probe_meminfo_kib(prefix: &str) -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(prefix))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<f64>().ok())
        })
        .unwrap_or(0.0)
}

fn probe_disk_free_human(path: &str) -> String {
    let out = Command::new("df")
        .args(["-h", "--output=avail", path])
        .output()
        .ok();
    let Some(out) = out else {
        return "n/a".into();
    };
    if !out.status.success() {
        return "n/a".into();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .nth(1)
        .map(|l| l.trim().to_string())
        .unwrap_or_else(|| "n/a".into())
}

fn probe_disk_used_pct(path: &str) -> u32 {
    let out = Command::new("df")
        .args(["--output=pcent", path])
        .output()
        .ok();
    let Some(out) = out else { return 0 };
    if !out.status.success() {
        return 0;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .nth(1)
        .and_then(|l| l.trim().trim_end_matches('%').parse().ok())
        .unwrap_or(0)
}

fn probe_loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().map(|x| x.to_string()))
        .and_then(|n| n.parse().ok())
        .unwrap_or(0.0)
}
