//! Resource probe for the admission gate (ADR 0046, slice-1).
//!
//! Cheap, best-effort reads of free RAM (`/proc/meminfo` MemAvailable)
//! and free VRAM (`nvidia-smi`). Independent of the TUI's
//! `SystemSnapshot` so the broker carries no coupling to the
//! presentation layer; the two probes share the same OS sources but
//! the broker only needs the admission levers (MemAvailable + free
//! VRAM), not the disk / utilization fields the TUI panel renders.
//!
//! `MemAvailable` already nets out RAM held by running units, so the
//! admission RAM test does not need a separate "sum of running blut
//! units" term (that cross-check is a display-only nicety the
//! `blut_admit.sh` operator message keeps; not load-bearing here).

use std::process::Command;

/// A best-effort snapshot of the admission-relevant OS resources.
/// Field absence (`None` / `0.0`) means "couldn't read it" — callers
/// treat a missing VRAM read as "no GPU constraint known" and a
/// missing RAM read as 0 free (fail-safe toward refusing).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResourceSnapshot {
    /// `/proc/meminfo` MemTotal, GiB.
    pub mem_total_gb: f64,
    /// `/proc/meminfo` MemAvailable, GiB. The RAM admission lever.
    pub mem_avail_gb: f64,
    /// `nvidia-smi` total VRAM, MiB (None = no nvidia-smi).
    pub vram_total_mib: Option<u64>,
    /// `nvidia-smi` free VRAM, MiB (None = no nvidia-smi).
    pub vram_free_mib: Option<u64>,
}

impl ResourceSnapshot {
    /// Probe live OS state. Never fails — missing tools degrade to
    /// `None` / `0.0` fields (fail-safe toward refusal on RAM).
    pub fn probe() -> Self {
        let mem_total_gb = probe_meminfo_kib("MemTotal:") / 1024.0 / 1024.0;
        let mem_avail_gb = probe_meminfo_kib("MemAvailable:") / 1024.0 / 1024.0;
        let vram_total_mib = probe_gpu_field("memory.total");
        let vram_used_mib = probe_gpu_field("memory.used");
        let vram_free_mib = match (vram_total_mib, vram_used_mib) {
            (Some(t), Some(u)) => Some(t.saturating_sub(u)),
            _ => None,
        };
        Self {
            mem_total_gb,
            mem_avail_gb,
            vram_total_mib,
            vram_free_mib,
        }
    }
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

/// Single-GPU read (`.lines().next()`); `--id=<i>` multi-card
/// enumeration is the deferred slice-3 extension.
fn probe_gpu_field(field: &str) -> Option<u64> {
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
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .and_then(|l| l.parse::<u64>().ok())
}
