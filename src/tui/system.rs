//! System-probe panel data for `blut tui`.
//!
//! Cheap once-per-tick reads: `/proc/meminfo`, `/proc/loadavg`,
//! `nvidia-smi --query-gpu`, and `statvfs(/mnt/4tb)`. All best-effort
//! — a missing tool / unreadable file degrades to a single "n/a"
//! line rather than failing the TUI loop.

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
    pub disk_free_human: String,
    pub disk_used_pct: u32,
    pub load1: f64,
}

impl SystemSnapshot {
    pub(super) fn probe() -> Self {
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
            disk_free_human: probe_disk_free_human("/mnt/4tb"),
            disk_used_pct: probe_disk_used_pct("/mnt/4tb"),
            load1: probe_loadavg(),
        }
    }

    pub(super) fn gpu_summary(&self) -> String {
        match (self.gpu_mem_used_mib, self.gpu_mem_total_mib, self.gpu_util_pct) {
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
    s.lines().next().map(|l| l.trim().to_string()).filter(|l| !l.is_empty())
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
