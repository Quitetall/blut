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
}

impl ResourceSnapshot {
    /// Probe live OS state. Never fails — missing tools degrade to
    /// `None` / `0.0` fields (fail-safe toward refusal on RAM).
    pub fn probe() -> Self {
        let mem_total_gb = probe_meminfo_kib("MemTotal:") / 1024.0 / 1024.0;
        let mem_avail_gb = probe_meminfo_kib("MemAvailable:") / 1024.0 / 1024.0;
        let gpus = probe_all_gpus();
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
        }
    }

    /// Get the number of GPUs available.
    pub fn gpu_count(&self) -> u32 {
        self.gpus.len() as u32
    }

    /// Get per-GPU info for device scheduling.
    pub fn gpus(&self) -> &[GpuInfo] {
        &self.gpus
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

/// Per-GPU info for admission's VRAM totals — sourced from the single
/// consolidated one-shot probe [`crate::broker::gpu::GpuInventory`] (ADR 0087),
/// mapping its richer `GpuDevice` down to the fields admission needs. All
/// `nvidia-smi` / `rocm-smi` shell-out now lives in `broker/gpu.rs`.
fn probe_all_gpus() -> Vec<GpuInfo> {
    crate::broker::gpu::GpuInventory::probe()
        .devices
        .into_iter()
        .map(|d| GpuInfo {
            index: d.index as u32,
            model: d.model,
            vram_total_mib: d.vram_total_mib,
            vram_free_mib: d.vram_free_mib,
        })
        .collect()
}
