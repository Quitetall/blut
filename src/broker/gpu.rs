// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! GPU-aware scheduling (ADR 0087) — the never-OOM broker still gates RAM.
//!
//! This module centralizes ALL GPU discovery behind [`GpuInventory`] (a
//! one-shot probe — charter-safe, no monitoring loop, ADR 0034) and replaces the
//! executor's single `Semaphore::new(1)` with a [`GpuScheduler`] holding one
//! semaphore PER physical device. A stage declares a [`GpuRequest`]; the
//! scheduler grants a device set whose free-VRAM estimate satisfies
//! `min_vram_mib`, holds those exclusive per-device permits for the stage's
//! lifetime, and exposes the granted indices for `CUDA_VISIBLE_DEVICES`.
//!
//! The GPU grant is a SECOND key in series with the RAM broker (ADR 0046/0047),
//! never a bypass: a stage launches only when the GPU scheduler grants its
//! devices AND the RAM broker admits its host-memory envelope.
//!
//! At inventory size 1 with the default request (`count=1, exclusive=true`) the
//! behaviour is byte-identical to the legacy single-semaphore path.

use std::process::Command;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// One physical accelerator, as seen by the one-shot inventory probe. `index`
/// is the driver's device index (what `CUDA_VISIBLE_DEVICES` names); `uuid`
/// anchors topology adjacency; VRAM is a MiB snapshot, not a monitored value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuDevice {
    pub index: usize,
    pub uuid: String,
    pub vram_total_mib: u64,
    pub vram_free_mib: u64,
    pub model: String,
}

/// A stage's GPU ask, part of its typed resource envelope. `Default` is the
/// whole-device exclusive request — an un-annotated stage behaves exactly as
/// under the legacy single semaphore.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct GpuRequest {
    /// Devices to hold for the stage's lifetime.
    pub count: u32,
    /// Minimum free VRAM (MiB) each granted device must have. `0` = any device.
    pub min_vram_mib: u64,
    /// Whole-device exclusivity. v1 is exclusive-only (no fractional sharing —
    /// see ADR 0087 Alternatives); the field is reserved for a future MIG path.
    pub exclusive: bool,
}

impl Default for GpuRequest {
    fn default() -> Self {
        Self {
            count: 1,
            min_vram_mib: 0,
            exclusive: true,
        }
    }
}

/// A one-shot snapshot of the box's GPUs. Built once per executor run — never
/// polled. Source order: `BLUT_GPU_INVENTORY` override → `nvidia-smi` →
/// `rocm-smi` → empty. Honors `CUDA_VISIBLE_DEVICES` as a scheduling filter.
#[derive(Clone, Debug, Default)]
pub struct GpuInventory {
    pub devices: Vec<GpuDevice>,
}

impl GpuInventory {
    /// The canonical one-shot probe (replaces the scattered `nvidia-smi` calls
    /// in `probe.rs` + `launcher.rs`). Never fails — a missing tool degrades to
    /// an empty inventory (fail-safe: no GPU ⇒ a scheduler of one cell).
    pub fn probe() -> Self {
        if let Some(inv) = Self::from_env() {
            return inv;
        }
        let devices = probe_nvidia().or_else(probe_rocm).unwrap_or_default();
        Self {
            devices: filter_visible(devices),
        }
    }

    /// `BLUT_GPU_INVENTORY` override for CI / no-CUDA hosts + deterministic
    /// tests. Two forms: a bare count (`"4"` ⇒ 4 devices, ample VRAM) or a
    /// per-device spec `"0:24576,1:24576"` (`index:free_vram_mib`).
    pub fn from_env() -> Option<Self> {
        let spec = std::env::var("BLUT_GPU_INVENTORY").ok()?;
        let spec = spec.trim();
        if spec.is_empty() {
            return Some(Self::default());
        }
        // Bare count.
        if let Ok(n) = spec.parse::<usize>() {
            let devices = (0..n)
                .map(|i| GpuDevice {
                    index: i,
                    uuid: format!("FAKE-{i}"),
                    vram_total_mib: 40960,
                    vram_free_mib: 40960,
                    model: "fake".into(),
                })
                .collect();
            return Some(Self { devices });
        }
        // Per-device `index:vram` spec.
        let mut devices = Vec::new();
        for part in spec.split(',') {
            let (i, v) = part.split_once(':')?;
            let index = i.trim().parse::<usize>().ok()?;
            let vram = v.trim().parse::<u64>().ok()?;
            devices.push(GpuDevice {
                index,
                uuid: format!("FAKE-{index}"),
                vram_total_mib: vram,
                vram_free_mib: vram,
                model: "fake".into(),
            });
        }
        Some(Self { devices })
    }

    /// Construct from an explicit device list — the test / fake seam (mirrors
    /// `admission.rs`'s `snap()` fixture for RAM).
    pub fn from_devices(devices: Vec<GpuDevice>) -> Self {
        Self { devices }
    }

    /// `n` identical devices with `vram_mib` free each — for sizing the scheduler
    /// from a bare count (tests, `with_resource_limit(Gpu, n)`, remote launchers
    /// whose device count is known but whose VRAM the local box can't probe).
    pub fn homogeneous(n: usize, vram_mib: u64) -> Self {
        let devices = (0..n)
            .map(|i| GpuDevice {
                index: i,
                uuid: format!("H-{i}"),
                vram_total_mib: vram_mib,
                vram_free_mib: vram_mib,
                model: "homogeneous".into(),
            })
            .collect();
        Self { devices }
    }

    /// Device count (≥1 for scheduler sizing — a no-GPU box still runs one cell).
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }
}

/// Restrict a probed device list to `CUDA_VISIBLE_DEVICES` when set, preserving
/// the driver indices (what the child's fresh `CUDA_VISIBLE_DEVICES` will name).
fn filter_visible(devices: Vec<GpuDevice>) -> Vec<GpuDevice> {
    let Ok(v) = std::env::var("CUDA_VISIBLE_DEVICES") else {
        return devices;
    };
    let v = v.trim();
    // CUDA convention: the var SET but empty means "no GPUs visible".
    if v.is_empty() {
        return Vec::new();
    }
    let visible: Vec<usize> = v
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if visible.is_empty() {
        return devices; // unparseable (not empty) → don't hide real GPUs
    }
    devices
        .into_iter()
        .filter(|d| visible.contains(&d.index))
        .collect()
}

/// `nvidia-smi` one-shot probe (adds `uuid` vs the old `probe.rs` version for
/// topology). A single malformed line is skipped, not fatal.
fn probe_nvidia() -> Option<Vec<GpuDevice>> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.total,memory.used,uuid,name",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut devices = Vec::new();
    for line in s.lines() {
        let p: Vec<&str> = line.split(',').map(|x| x.trim()).collect();
        if p.len() < 5 {
            continue;
        }
        let (Ok(index), Ok(total), Ok(used)) = (
            p[0].parse::<usize>(),
            p[1].parse::<u64>(),
            p[2].parse::<u64>(),
        ) else {
            continue;
        };
        devices.push(GpuDevice {
            index,
            vram_total_mib: total,
            vram_free_mib: total.saturating_sub(used),
            uuid: p[3].to_string(),
            model: p[4].to_string(),
        });
    }
    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

/// `rocm-smi` fallback (no per-device free-VRAM signal ⇒ conservative 0).
fn probe_rocm() -> Option<Vec<GpuDevice>> {
    let out = Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram", "--csv"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut devices = Vec::new();
    for (i, line) in s.lines().enumerate() {
        if i == 0 && line.contains("GPU") {
            continue;
        }
        let p: Vec<&str> = line.split(',').map(|x| x.trim()).collect();
        if p.len() < 2 {
            continue;
        }
        let total = p.last().and_then(|x| x.parse::<u64>().ok()).unwrap_or(0);
        let index = devices.len();
        devices.push(GpuDevice {
            index,
            uuid: format!("ROCM-{index}"),
            vram_total_mib: total,
            vram_free_mib: 0,
            model: p[0].to_string(),
        });
    }
    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

/// Why a GPU grant could not be satisfied at all (distinct from *waiting*).
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error(
        "no device set satisfies the request: need {need} device(s) with ≥{min_vram_mib} MiB free, \
         only {have} qualify"
    )]
    Unsatisfiable {
        need: usize,
        have: usize,
        min_vram_mib: u64,
    },
    #[error("gpu semaphore closed")]
    Closed,
}

/// One semaphore per physical device + a counting `avail` gate. The `avail`
/// permit total == device count: a grant first admits through `avail` (blocks
/// until `need` devices are free — deadlock-free, no fixed device ordering),
/// THEN grabs `need` free devices non-blockingly (so it never waits while
/// holding a device — no circular wait). Sizing 1 reproduces the legacy
/// single-semaphore path exactly.
pub struct GpuScheduler {
    devices: Vec<GpuDevice>,
    sems: Vec<Arc<Semaphore>>,
    avail: Arc<Semaphore>,
}

impl GpuScheduler {
    /// Build from an inventory. A no-GPU box yields one cell so a CPU-only plan
    /// still runs (the request's `count=1` acquires the single cell).
    pub fn new(inv: GpuInventory) -> Self {
        let devices = if inv.devices.is_empty() {
            vec![GpuDevice {
                index: 0,
                uuid: "none".into(),
                vram_total_mib: 0,
                vram_free_mib: 0,
                model: "none".into(),
            }]
        } else {
            inv.devices
        };
        let sems = devices
            .iter()
            .map(|_| Arc::new(Semaphore::new(1)))
            .collect();
        let avail = Arc::new(Semaphore::new(devices.len()));
        Self {
            devices,
            sems,
            avail,
        }
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// The count actually schedulable: clamped to the device pool. Warns (as the
    /// pre-ADR path did) when a DDP request exceeds the pool and runs degraded, so
    /// a demotion from full width isn't silent.
    fn effective_need(&self, count: u32) -> usize {
        let dc = self.devices.len();
        let want = (count.max(1) as usize).min(dc);
        if count as usize > dc {
            tracing::warn!(
                "GPU request for {count} devices exceeds the {dc}-device pool — \
                 running degraded on {dc} (DDP width reduced)"
            );
        }
        want
    }

    /// Positions whose free-VRAM snapshot meets the floor, ascending (adjacency-
    /// preferred). A `0` floor qualifies every device (incl. the no-GPU cell).
    fn candidates(&self, min_vram_mib: u64) -> Vec<usize> {
        (0..self.devices.len())
            .filter(|&i| self.devices[i].vram_free_mib >= min_vram_mib)
            .collect()
    }

    /// Grab `need` FREE devices, VRAM-qualifying first (soft — under contention a
    /// non-qualifying free device is a best-effort fallback; the ADR treats VRAM
    /// as an estimate). Caller holds `need` `avail` permits, so `need` devices
    /// ARE free ⇒ this never blocks and always fills.
    fn grab(&self, need: usize, min_vram_mib: u64) -> (Vec<OwnedSemaphorePermit>, Vec<usize>) {
        let mut permits = Vec::with_capacity(need);
        let mut got = Vec::with_capacity(need);
        for &pos in &self.candidates(min_vram_mib) {
            if got.len() == need {
                break;
            }
            if let Ok(p) = self.sems[pos].clone().try_acquire_owned() {
                permits.push(p);
                got.push(pos);
            }
        }
        for pos in 0..self.devices.len() {
            if got.len() == need {
                break;
            }
            if !got.contains(&pos) {
                if let Ok(p) = self.sems[pos].clone().try_acquire_owned() {
                    permits.push(p);
                    got.push(pos);
                }
            }
        }
        (permits, got)
    }

    fn grant(
        &self,
        avail: OwnedSemaphorePermit,
        permits: Vec<OwnedSemaphorePermit>,
        mut got: Vec<usize>,
    ) -> GpuGrant {
        got.sort_unstable();
        let devices = got.iter().map(|&p| self.devices[p].index).collect();
        GpuGrant {
            devices,
            _permits: permits,
            _avail: avail,
        }
    }

    /// Non-blocking grant (the fast path). `None` on contention OR when
    /// unsatisfiable — the caller emits `StageBlocked` then falls to
    /// [`acquire`](Self::acquire), which distinguishes the two (waits vs errors).
    pub fn try_acquire(&self, req: GpuRequest) -> Option<GpuGrant> {
        let need = self.effective_need(req.count);
        if self.candidates(req.min_vram_mib).len() < need {
            return None;
        }
        let avail = self
            .avail
            .clone()
            .try_acquire_many_owned(need as u32)
            .ok()?;
        let (permits, got) = self.grab(need, req.min_vram_mib);
        if got.len() < need {
            return None; // permits + avail drop here
        }
        Some(self.grant(avail, permits, got))
    }

    /// Grant a device set, blocking until `need` devices free. Admission via the
    /// `avail` counting semaphore is deadlock-free (no per-device ordering); the
    /// device grabs afterward never block. Fails fast (never hangs) when no set
    /// can EVER satisfy `min_vram_mib`.
    pub async fn acquire(&self, req: GpuRequest) -> Result<GpuGrant, GpuError> {
        // effective_need warns once here on the slow (blocking) path; the fast
        // try_acquire path already warned if it ran first, but run_node calls one
        // or the other per attempt, so a clamped request warns at most twice.
        let need = self.effective_need(req.count);
        let have = self.candidates(req.min_vram_mib).len();
        if have < need {
            return Err(GpuError::Unsatisfiable {
                need,
                have,
                min_vram_mib: req.min_vram_mib,
            });
        }
        let avail = match self.avail.clone().acquire_many_owned(need as u32).await {
            Ok(a) => a,
            Err(_) => return Err(GpuError::Closed),
        };
        let (permits, got) = self.grab(need, req.min_vram_mib);
        debug_assert_eq!(
            got.len(),
            need,
            "avail admission guarantees `need` free devices"
        );
        Ok(self.grant(avail, permits, got))
    }
}

/// A live grant: the assigned driver device indices + the held per-device +
/// `avail` permits. Dropping it releases the devices; the exclusive per-device
/// semaphores are the hard no-collision guarantee independent of the child.
#[derive(Debug)]
pub struct GpuGrant {
    pub devices: Vec<usize>,
    _permits: Vec<OwnedSemaphorePermit>,
    _avail: OwnedSemaphorePermit,
}

impl GpuGrant {
    /// The `CUDA_VISIBLE_DEVICES` value for the granted set (driver indices, csv).
    pub fn cuda_visible_devices(&self) -> String {
        self.devices
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Named `gpu_sched` so ADR 0087's acceptance gate `cargo test -p blut gpu_sched`
/// selects exactly this module (count adjacency · VRAM wait · CVD masking ·
/// size-1 legacy parity).
#[cfg(test)]
mod gpu_sched {
    use super::*;

    fn dev(index: usize, free: u64) -> GpuDevice {
        GpuDevice {
            index,
            uuid: format!("U{index}"),
            vram_total_mib: 40960,
            vram_free_mib: free,
            model: "t".into(),
        }
    }

    fn sched4() -> GpuScheduler {
        GpuScheduler::new(GpuInventory::from_devices(vec![
            dev(0, 40960),
            dev(1, 40960),
            dev(2, 40960),
            dev(3, 40960),
        ]))
    }

    #[tokio::test]
    async fn count2_colocates_on_adjacent_indices() {
        let s = sched4();
        let g = s
            .acquire(GpuRequest {
                count: 2,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(g.devices, vec![0, 1], "count=2 picks the low adjacent run");
        assert_eq!(g.cuda_visible_devices(), "0,1");
    }

    #[tokio::test]
    async fn vram_floor_over_budget_is_unsatisfiable_not_a_hang() {
        let s = GpuScheduler::new(GpuInventory::from_devices(vec![dev(0, 8000), dev(1, 8000)]));
        let err = s
            .acquire(GpuRequest {
                count: 1,
                min_vram_mib: 24000,
                exclusive: true,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, GpuError::Unsatisfiable { .. }));
    }

    #[tokio::test]
    async fn a_busy_device_makes_the_next_request_wait_until_the_grant_drops() {
        let s = Arc::new(GpuScheduler::new(GpuInventory::from_devices(vec![dev(
            0, 40960,
        )])));
        let g = s.acquire(GpuRequest::default()).await.unwrap();
        // Second request for the only device must block while `g` is held.
        let s2 = s.clone();
        let waiter = tokio::spawn(async move { s2.acquire(GpuRequest::default()).await });
        // Give it a moment; it must NOT have completed.
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert!(
            !waiter.is_finished(),
            "the second grant must wait on the busy device"
        );
        drop(g); // release device 0
        let g2 = waiter.await.unwrap().unwrap();
        assert_eq!(g2.devices, vec![0]);
    }

    #[tokio::test]
    async fn size1_inventory_serializes_like_the_legacy_single_semaphore() {
        // One device + default request == the old Semaphore::new(1): a second
        // acquire can't proceed until the first drops. CVD masks to "0".
        let s = Arc::new(GpuScheduler::new(GpuInventory::from_devices(vec![dev(
            0, 40960,
        )])));
        assert_eq!(s.device_count(), 1);
        let g = s.acquire(GpuRequest::default()).await.unwrap();
        assert_eq!(g.cuda_visible_devices(), "0");
        assert!(s.sems[0].try_acquire().is_err(), "device held ⇒ serialized");
        drop(g);
        assert!(s.sems[0].try_acquire().is_ok(), "released after drop");
    }

    #[test]
    fn from_env_bare_count_and_spec() {
        // Serialize env mutation against every other env-touching test (probe()
        // reads BLUT_GPU_INVENTORY on other threads).
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // SAFETY: serialized by TEST_ENV_LOCK; removed below.
        unsafe { std::env::set_var("BLUT_GPU_INVENTORY", "4") };
        assert_eq!(GpuInventory::from_env().unwrap().len(), 4);
        unsafe { std::env::set_var("BLUT_GPU_INVENTORY", "0:24576,1:12288") };
        let inv = GpuInventory::from_env().unwrap();
        assert_eq!(inv.devices.len(), 2);
        assert_eq!(inv.devices[1].vram_free_mib, 12288);
        unsafe { std::env::remove_var("BLUT_GPU_INVENTORY") };
    }

    #[test]
    fn default_request_is_whole_device_exclusive() {
        let r = GpuRequest::default();
        assert_eq!(r.count, 1);
        assert_eq!(r.min_vram_mib, 0);
        assert!(r.exclusive);
    }
}
