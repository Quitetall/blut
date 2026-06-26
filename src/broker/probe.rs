// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
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

/// Per-GPU resource information.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuInfo {
    /// GPU index.
    pub index: u32,
    /// GPU model name (e.g. "NVIDIA RTX 4090").
    pub model: String,
    /// Total VRAM, MiB.
    pub vram_total_mib: u64,
    /// Free VRAM, MiB.
    pub vram_free_mib: u64,
}

/// A best-effort snapshot of the admission-relevant OS resources.
/// Field absence (`None` / `0.0`) means "couldn't read it" — callers
/// treat a missing VRAM read as "no GPU constraint known" and a
/// missing RAM read as 0 free (fail-safe toward refusing).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResourceSnapshot {
    /// `/proc/meminfo` MemTotal, GiB.
    pub mem_total_gb: f64,
    /// `/proc/meminfo` MemAvailable, GiB. The RAM admission lever.
    pub mem_avail_gb: f64,
    /// `nvidia-smi` total VRAM, MiB (None = no nvidia-smi). Sum across all GPUs.
    pub vram_total_mib: Option<u64>,
    /// `nvidia-smi` free VRAM, MiB (None = no nvidia-smi). Sum across all GPUs.
    pub vram_free_mib: Option<u64>,
    /// Per-GPU information (empty if no nvidia-smi).
    pub gpus: Vec<GpuInfo>,
    /// Number of GPUs detected.
    pub gpu_count: u32,
}

impl ResourceSnapshot {
    /// Probe live OS state. Never fails — missing tools degrade to
    /// `None` / `0.0` fields (fail-safe toward refusal on RAM).
    pub fn probe() -> Self {
        let mem_total_gb = probe_meminfo_kib("MemTotal:") / 1024.0 / 1024.0;
        let mem_avail_gb = probe_meminfo_kib("MemAvailable:") / 1024.0 / 1024.0;
        let gpus = probe_all_gpus();
        let gpu_count = gpus.len() as u32;
        let vram_total_mib = if gpus.is_empty() {
            None
        } else {
            Some(gpus.iter().map(|g| g.vram_total_mib).sum())
        };
        let vram_free_mib = if gpus.is_empty() {
            None
        } else {
            Some(gpus.iter().map(|g| g.vram_free_mib).sum())
        };
        Self {
            mem_total_gb,
            mem_avail_gb,
            vram_total_mib,
            vram_free_mib,
            gpus,
            gpu_count,
        }
    }

    /// Get the number of GPUs available.
    pub fn gpu_count(&self) -> u32 {
        self.gpu_count
    }

    /// Get per-GPU VRAM info for device scheduling.
    pub fn gpu_vram_mib(&self) -> Vec<(u32, u64, u64)> {
        self.gpus.iter().map(|g| (g.index, g.vram_total_mib, g.vram_free_mib)).collect()
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

/// Probe all GPUs via nvidia-smi. Returns per-GPU info.
/// Falls back to probing AMD GPUs via rocm-smi if nvidia-smi fails.
fn probe_all_gpus() -> Vec<GpuInfo> {
    // Try nvidia-smi first
    if let Some(gpus) = probe_nvidia_gpus() {
        return gpus;
    }
    // Fall back to rocm-smi for AMD GPUs
    if let Some(gpus) = probe_rocm_gpus() {
        return gpus;
    }
    Vec::new()
}

/// Probe NVIDIA GPUs via nvidia-smi.
fn probe_nvidia_gpus() -> Option<Vec<GpuInfo>> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut gpus = Vec::new();
    for line in s.lines() {
        let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
        if parts.len() >= 4 {
            let index = parts[0].parse::<u32>().ok()?;
            let model = parts[1].to_string();
            let total = parts[2].parse::<u64>().ok()?;
            let used = parts[3].parse::<u64>().ok()?;
            gpus.push(GpuInfo {
                index,
                model,
                vram_total_mib: total,
                vram_free_mib: total.saturating_sub(used),
            });
        }
    }
    if gpus.is_empty() { None } else { Some(gpus) }
}

/// Probe AMD GPUs via rocm-smi.
fn probe_rocm_gpus() -> Option<Vec<GpuInfo>> {
    let out = Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram", "--csv"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut gpus = Vec::new();
    // rocm-smi CSV format varies; try a simple parse
    for (i, line) in s.lines().enumerate() {
        if i == 0 && line.contains("GPU") {
            continue; // skip header
        }
        let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
        if parts.len() >= 2 {
            let model = parts[0].to_string();
            let total = parts.last().and_then(|p| p.parse::<u64>().ok()).unwrap_or(0);
            gpus.push(GpuInfo {
                index: i as u32,
                model,
                vram_total_mib: total,
                vram_free_mib: total, // rocm-smi doesn't easily give free VRAM in CSV
            });
        }
    }
    if gpus.is_empty() { None } else { Some(gpus) }
}
