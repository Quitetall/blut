// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Live GPU-saturation sampler (E2 — owner directive: "GPU must run
//! close to maximum, never wasted — and this must be MEASURED, not
//! assumed").
//!
//! While a GPU stage runs, a background task polls `nvidia-smi` every
//! [`SAMPLE_INTERVAL`] and emits one `StageStep{kind:"gpu_gauge"}` event
//! per sample through the [`StatusHub`] broadcast. The status writer
//! lands those in `status.jsonl`; at run-end
//! [`fold_gauges`](crate::framework::lineage::fold_gauges) ingests them
//! into the `gauges` table, where `gpu_saturation` / `gpu_wasted` are
//! derived (see [`LineageDb::gpu_saturation`](crate::lineage_db::LineageDb::gpu_saturation)).
//!
//! When utilization stays below [`STARVED_UTIL_PCT`] for
//! [`STARVED_CONSECUTIVE`] consecutive samples the sampler emits a
//! one-shot `kind:"gpu_starved"` sentinel, so the known decode-bound,
//! GPU-starved training failure mode is LOUD and immediate instead of a
//! silent waste.
//!
//! Best-effort and side-effect-free on the run: a missing / failing
//! `nvidia-smi` simply yields no samples, and the poll cadence is
//! independent of the trainer's own epoch events.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use crate::framework::status::{StageEvent, StatusHub};

/// Poll cadence. 2 s keeps overhead negligible (one short-lived
/// subprocess every 2 s) while still resolving multi-second starvation
/// windows — finer than that buys nothing for a GPU that holds util for
/// whole epochs.
pub const SAMPLE_INTERVAL: Duration = Duration::from_secs(2);

/// Below this GPU-utilization %, a sample counts as "wasted". This is
/// also the conventional floor `blut`'s GPU-saturation surfaces pass to
/// [`gpu_saturation`](crate::lineage_db::LineageDb::gpu_saturation) for
/// the `gpu_wasted` fraction.
pub const STARVED_UTIL_PCT: f64 = 25.0;

/// Consecutive sub-floor samples before the one-shot starvation sentinel
/// fires (≈ `STARVED_CONSECUTIVE × SAMPLE_INTERVAL` of sustained low
/// util). Re-arms once util recovers above the floor.
pub const STARVED_CONSECUTIVE: u32 = 5;

// Static invariants (checked at compile time): the "wasted" floor is a
// real sub-saturation threshold, so a fully-utilized GPU is never
// flagged starved, and the sentinel needs at least one sub-floor sample.
const _: () = assert!(STARVED_UTIL_PCT > 0.0 && STARVED_UTIL_PCT < 100.0);
const _: () = assert!(STARVED_CONSECUTIVE >= 1);

/// One `nvidia-smi` reading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSample {
    /// SM utilization, percent.
    pub util: f64,
    /// Device memory in use, MiB.
    pub mem_mib: f64,
    /// Core temperature, °C.
    pub temp_c: f64,
    /// Board power draw, W (0.0 if the device reports `[N/A]`).
    pub power_w: f64,
}

/// One `nvidia-smi` call for all four fields at once (one subprocess per
/// tick, not one per field). `None` when nvidia-smi is absent, exits
/// non-zero, or the row doesn't parse — the caller treats that as "no
/// sample this tick" and keeps polling.
pub fn sample_gpu() -> Option<GpuSample> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    parse_sample(s.lines().next()?)
}

/// Parse one CSV row (`util, mem.used, temp, power.draw`). Split out so
/// the field handling — including the `[N/A]` power reading some GPUs
/// emit — is unit-testable without an nvidia-smi on the box.
fn parse_sample(line: &str) -> Option<GpuSample> {
    let mut it = line.split(',').map(|f| f.trim());
    let util = it.next()?.parse().ok()?;
    let mem_mib = it.next()?.parse().ok()?;
    let temp_c = it.next()?.parse().ok()?;
    // `power.draw` reads "[N/A]" on some GPUs / VMs — tolerate it as 0.0
    // rather than dropping the whole (otherwise-useful) sample.
    let power_w = it.next().and_then(|p| p.parse().ok()).unwrap_or(0.0);
    Some(GpuSample {
        util,
        mem_mib,
        temp_c,
        power_w,
    })
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Handle to a running sampler. Call [`stop`](GpuSamplerHandle::stop) at
/// the end of the run window. NOTE: a bare `drop` only *detaches* the
/// task (tokio semantics) — it keeps sampling until process exit — so
/// always `stop()` it. `run_node` does this on every exit path.
pub struct GpuSamplerHandle {
    task: tokio::task::JoinHandle<()>,
}

impl GpuSamplerHandle {
    /// Stop the sampler and await its teardown (idempotent). `abort()`
    /// cancels the task at its next await point; because the blocking
    /// `nvidia-smi` probe runs on a `spawn_blocking` thread (awaited),
    /// an abort during an in-flight probe DETACHES that probe (it
    /// finishes on the blocking pool, unobserved) and returns promptly —
    /// a hung nvidia-smi can never wedge `stop()` and, through it, the
    /// GPU permit. Aborting mid-loop cannot corrupt `status.jsonl`: every
    /// emitted event is a fully-serialized line.
    pub async fn stop(self) {
        self.task.abort();
        let _ = self.task.await;
    }
}

/// Spawn the periodic GPU sampler for one node. Emits `gpu_gauge`
/// samples (and at most one `gpu_starved` sentinel per starvation
/// episode) via `status` until [`stop`](GpuSamplerHandle::stop)ped.
///
/// The first sample is taken AFTER one interval, so a stage that
/// finishes in under [`SAMPLE_INTERVAL`] records nothing — and needs
/// nothing, since there was no GPU window long enough to waste.
pub fn spawn_gpu_sampler(
    status: Arc<StatusHub>,
    node_idx: u32,
    stage_name: String,
) -> GpuSamplerHandle {
    let task = tokio::spawn(async move {
        let mut low_streak: u32 = 0;
        let mut starved_fired = false;
        loop {
            tokio::time::sleep(SAMPLE_INTERVAL).await;
            // Run the blocking nvidia-smi off the async worker pool. A
            // hung probe then strands a blocking thread, NOT a runtime
            // worker — and an `abort()` during the probe detaches it so
            // `stop()` returns promptly (no GPU-permit deadlock).
            let sampled = match tokio::task::spawn_blocking(sample_gpu).await {
                Ok(s) => s,
                Err(_) => continue, // probe task cancelled/panicked → skip tick
            };
            let Some(s) = sampled else {
                continue;
            };
            let wall = now_unix();
            status.emit(StageEvent::StageStep {
                node_idx,
                stage_name: stage_name.clone(),
                update: json!({
                    "kind": "gpu_gauge",
                    "wall_unix": wall,
                    "gpu_util": s.util,
                    "gpu_mem_mib": s.mem_mib,
                    "gpu_temp_c": s.temp_c,
                    "gpu_power_w": s.power_w,
                }),
            });
            // Starvation sentinel: one loud event per sustained sub-floor
            // episode (not per-sample spam), so a decode-bound run that
            // starves the GPU surfaces immediately in the Log pane /
            // status.jsonl. Re-arm only after util recovers.
            if s.util < STARVED_UTIL_PCT {
                low_streak += 1;
                if low_streak >= STARVED_CONSECUTIVE && !starved_fired {
                    starved_fired = true;
                    status.emit(StageEvent::StageStep {
                        node_idx,
                        stage_name: stage_name.clone(),
                        update: json!({
                            "kind": "gpu_starved",
                            "wall_unix": wall,
                            "gpu_util": s.util,
                            "samples_below": low_streak,
                            "floor_pct": STARVED_UTIL_PCT,
                        }),
                    });
                }
            } else {
                low_streak = 0;
                starved_fired = false;
            }
        }
    });
    GpuSamplerHandle { task }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_well_formed_row() {
        let s = parse_sample("97, 18432, 71, 312.50").unwrap();
        assert_eq!(s.util, 97.0);
        assert_eq!(s.mem_mib, 18432.0);
        assert_eq!(s.temp_c, 71.0);
        assert_eq!(s.power_w, 312.50);
    }

    #[test]
    fn tolerates_na_power_draw() {
        // Some GPUs / VMs report power.draw as "[N/A]" — the sample is
        // still useful for util/mem/temp, so we keep it with power 0.0.
        let s = parse_sample("3, 512, 40, [N/A]").unwrap();
        assert_eq!(s.util, 3.0);
        assert_eq!(s.power_w, 0.0);
    }

    #[test]
    fn rejects_short_or_garbage_rows() {
        assert!(parse_sample("").is_none());
        assert!(parse_sample("not,a,number").is_none());
        // Missing temp+power → too short (util+mem only).
        assert!(parse_sample("50, 1024").is_none());
    }
}
