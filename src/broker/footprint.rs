//! Footprint estimation — the SCALING RAM model (ADR 0046, hole #3/#4).
//!
//! The blueprint's "constant 40-50G hint" is rejected by the review:
//! it is not an upper bound across tier / batch / latent-dim, so a
//! larger config can admit then OOM. The real driver of the RAM OOMs
//! on this box is the dataloader: `LMA_NUM_WORKERS × per-worker LMA
//! prefetch ≈ 23 GiB`. So the estimate is dominated by
//! `workers × prefetch_per_worker`, with smaller additive terms for
//! the model (scales with tier + latent_dim) and the live batch.
//!
//! The number is deliberately **conservative-HIGH**: a wrong (too
//! large) estimate only costs us a refused / smaller-capped job; a
//! wrong (too small) estimate is a live OOM. Containment (the cgroup
//! cap) is the backstop for any residual error here — this estimate
//! sets the cap and gates admission, but the kernel is the hard floor.

/// One gibibyte in bytes.
pub const GIB: u64 = 1024 * 1024 * 1024;

/// Conservative default batch billed when a recipe leaves `batch_size`
/// to the kernel default. SHARED between the cli admission gate
/// (`recipe_footprint`) and the train stage (`train_containment`) so
/// the admission `need` and the cgroup `cap = need + 2G` are computed
/// from the SAME inputs — otherwise admission could pass a job on a
/// smaller estimate than the cap it then runs under, breaching the OS
/// floor admission promised. Conservative-high so the cap is never
/// under-sized for an unspecified batch.
pub const DEFAULT_BATCH: u32 = 32;

/// Per-DataLoader-worker LMA prefetch RAM, conservative-high.
///
/// Measured envelope on the 62 GiB box: ~23 GiB total dataloader RSS
/// at the historical default worker count. We bill each worker a fat
/// ~3.5 GiB so the sum stays an UPPER bound (over-reserve, never
/// under). Tunable later by the calibration store (deferred slice 2).
const PREFETCH_PER_WORKER_BYTES: u64 = 7 * GIB / 2; // 3.5 GiB

/// Base RSS floor: python + torch + CUDA context + framework overhead,
/// independent of workers/batch. Conservative-high.
const BASE_RSS_BYTES: u64 = 6 * GIB;

/// Model + optimizer-state RAM per decoder tier (SOAP keeps
/// preconditioners; bill generously). Multiplied by `tier`.
const PER_TIER_BYTES: u64 = 2 * GIB;

/// RAM per 256 units of encoder latent width (the latent-dim knob,
/// tasks #270/#271). Conservative; mostly host-side staging buffers.
const PER_LATENT256_BYTES: u64 = GIB;

/// Host-side RAM that scales with the live mini-batch (pinned buffers,
/// collation staging), per unit of batch. Small vs the worker term.
const PER_BATCH_BYTES: u64 = GIB / 4; // 256 MiB / batch unit

/// A resolved footprint estimate. Slice-1 tracks RAM only as a hard
/// number; VRAM is carried for the (deferred) VRAM courtesy pre-check
/// but never gates the box-survival guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// Peak resident RAM the job is expected to need, in bytes.
    pub ram_bytes: u64,
    /// Peak VRAM in MiB (0 = unknown / not estimated). Courtesy only.
    pub vram_mib: u64,
}

impl Footprint {
    /// Headroom added on top of the estimate to derive the cgroup
    /// `MemoryMax` cap. Mirrors `blut_admit.sh`'s `need + 2`.
    pub const MEMMAX_HEADROOM_BYTES: u64 = 2 * GIB;

    /// The cgroup `MemoryMax` cap to apply: estimate + headroom.
    pub fn memmax_bytes(&self) -> u64 {
        self.ram_bytes.saturating_add(Self::MEMMAX_HEADROOM_BYTES)
    }
}

/// Estimate peak RAM (bytes) for a train-shaped job. **Monotone
/// non-decreasing** in every argument — the property the unit tests
/// pin and the property that makes "conservative-high" meaningful.
///
/// - `workers`: DataLoader workers (the dominant term via prefetch).
/// - `batch`: live mini-batch size.
/// - `tier`: decoder tier (1..=4); larger tier ⇒ more model/opt RAM.
/// - `latent_dim`: encoder latent width (0 ⇒ default, billed as 256).
pub fn estimate_ram_bytes(workers: u32, batch: u32, tier: u32, latent_dim: u32) -> u64 {
    let workers = workers as u64;
    let batch = batch as u64;
    let tier = tier.max(1) as u64; // tier 0 is nonsensical; floor at 1
    // Treat an unspecified latent (0) as the default 256-wide encoder
    // so the model term is never under-counted.
    let latent = if latent_dim == 0 { 256 } else { latent_dim } as u64;

    let workers_term = workers.saturating_mul(PREFETCH_PER_WORKER_BYTES);
    let tier_term = tier.saturating_mul(PER_TIER_BYTES);
    // ceil-div by 256 so any latent > 0 bills at least one unit.
    let latent_units = latent.div_ceil(256);
    let latent_term = latent_units.saturating_mul(PER_LATENT256_BYTES);
    let batch_term = batch.saturating_mul(PER_BATCH_BYTES);

    BASE_RSS_BYTES
        .saturating_add(workers_term)
        .saturating_add(tier_term)
        .saturating_add(latent_term)
        .saturating_add(batch_term)
}

/// Convenience: build a [`Footprint`] from the cost drivers (RAM
/// scaled, VRAM left unknown for slice-1).
pub fn estimate(workers: u32, batch: u32, tier: u32, latent_dim: u32) -> Footprint {
    Footprint {
        ram_bytes: estimate_ram_bytes(workers, batch, tier, latent_dim),
        vram_mib: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_floor_with_zero_drivers() {
        // Even all-zero drivers bill the base RSS + a tier-1 + default
        // latent floor — never zero, so admission can't be fooled.
        let r = estimate_ram_bytes(0, 0, 0, 0);
        assert!(r >= BASE_RSS_BYTES, "got {r}");
        // floors: base + tier1 + latent256 = 6 + 2 + 1 = 9 GiB
        assert_eq!(r, 9 * GIB);
    }

    #[test]
    fn monotone_in_workers() {
        let lo = estimate_ram_bytes(2, 16, 3, 256);
        let hi = estimate_ram_bytes(8, 16, 3, 256);
        assert!(hi > lo, "workers must increase RAM: {lo} !< {hi}");
        // workers dominate: +6 workers × 3.5 GiB = +21 GiB
        assert_eq!(hi - lo, 6 * PREFETCH_PER_WORKER_BYTES);
    }

    #[test]
    fn monotone_in_batch() {
        let lo = estimate_ram_bytes(4, 8, 3, 256);
        let hi = estimate_ram_bytes(4, 32, 3, 256);
        assert!(hi > lo, "batch must increase RAM: {lo} !< {hi}");
    }

    #[test]
    fn monotone_in_tier() {
        let lo = estimate_ram_bytes(4, 16, 1, 256);
        let hi = estimate_ram_bytes(4, 16, 4, 256);
        assert!(hi > lo, "tier must increase RAM: {lo} !< {hi}");
    }

    #[test]
    fn monotone_in_latent() {
        let lo = estimate_ram_bytes(4, 16, 3, 256);
        let hi = estimate_ram_bytes(4, 16, 3, 512);
        assert!(hi > lo, "latent_dim must increase RAM: {lo} !< {hi}");
    }

    #[test]
    fn workers_term_dominates() {
        // The whole point (hole #4): RAM is workers-driven, not
        // batch-driven. Doubling workers must move RAM more than
        // doubling batch from the same baseline.
        let base = estimate_ram_bytes(4, 16, 3, 256);
        let more_workers = estimate_ram_bytes(8, 16, 3, 256);
        let more_batch = estimate_ram_bytes(4, 32, 3, 256);
        assert!(
            more_workers - base > more_batch - base,
            "workers must dominate batch: dW={} dB={}",
            more_workers - base,
            more_batch - base
        );
    }

    #[test]
    fn memmax_adds_headroom() {
        let fp = estimate(4, 16, 3, 256);
        assert_eq!(fp.memmax_bytes(), fp.ram_bytes + 2 * GIB);
    }

    #[test]
    fn conservative_default_admits_one_train_on_62g_box() {
        // The load-bearing slice-1 property: the uncalibrated default
        // (capped workers ≤ 4) is conservative-high but still fits ONE
        // train on the 62 GiB box with the 6 GiB floor.
        let fp = estimate(4, 16, 3, 256);
        // 6 + 4×3.5 + 3×2 + 1 + 16×0.25 = 6+14+6+1+4 = 31 GiB
        assert_eq!(fp.ram_bytes, 31 * GIB);
        assert!(fp.ram_bytes < (62 - 6) * GIB, "must fit one train on 62G box");
    }
}
