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

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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

/// Provenance of a stored footprint, in MONOTONE-override order
/// (`OomCorrected` > `Measured` > `Default`). A run that OOM'd records
/// the cgroup limit it HIT as a known lower bound on true need, so it
/// must never be overwritten by a (smaller, lucky) measured peak from a
/// later run; `Measured` in turn always beats the conservative
/// `Default` estimate.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FootprintSource {
    /// The conservative pre-launch estimate (never persisted by
    /// `record`; the resolve fallback when no entry exists).
    Default,
    /// A cgroup-attributed peak observed at a clean (non-OOM) exit.
    Measured,
    /// The cgroup cap that was HIT on an OOM — a known lower bound on
    /// the true need (ADR 0046 deferred OOM slice will write this).
    OomCorrected,
}

impl FootprintSource {
    /// Override rank: higher wins on merge. `OomCorrected` (2) beats
    /// `Measured` (1) beats `Default` (0).
    fn rank(self) -> u8 {
        match self {
            FootprintSource::Default => 0,
            FootprintSource::Measured => 1,
            FootprintSource::OomCorrected => 2,
        }
    }
}

/// Stable composite key for the calibration store. `recipe` is the
/// RECIPE name (e.g. `lamquant_joint_codec`), NOT the stage name — the
/// cli admission gate resolves from the recipe it was asked to run, and
/// the stage records under the SAME recipe name (threaded via
/// `StageContext.recipe_name`) so the RESOLVE and RECORD keys match.
/// `tier`/`batch`/`workers` are the cost drivers the footprint scales
/// on (latent is folded into the estimate, not the key — it rarely
/// varies and would fragment the calibration).
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct FootprintKey {
    pub recipe: String,
    pub tier: u32,
    pub batch: u32,
    pub workers: u32,
}

impl FootprintKey {
    /// Flat string form used as the JSON map key. Pipe-delimited so it
    /// round-trips unambiguously (recipe names never contain `|`).
    pub fn flat(&self) -> String {
        // `|` is the delimiter — a recipe name containing it would collide two
        // distinct keys (e.g. `a|b`+tier1 vs `a`+tier`b|1`). Recipe names are
        // internal identifiers (never user free-text), so a debug_assert catches
        // a violation at test time without a release-path cost.
        debug_assert!(!self.recipe.contains('|'), "recipe name must not contain '|'");
        format!("{}|{}|{}|{}", self.recipe, self.tier, self.batch, self.workers)
    }
}

/// Build the calibration key from a recipe name + the cost drivers.
/// THE single shared constructor: both the cli admission gate (RESOLVE)
/// and the train stage (RECORD) call this so the keys are byte-identical
/// — if they diverged the calibration would never be hit and the broker
/// would over-refuse forever. `workers`/`batch`/`tier` MUST be the same
/// values fed to [`estimate`].
pub fn footprint_key(recipe: &str, workers: u32, batch: u32, tier: u32) -> FootprintKey {
    FootprintKey {
        recipe: recipe.to_string(),
        tier,
        batch,
        workers,
    }
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

/// One persisted calibration entry. RAM is MAX-merged (monotone-up: a
/// measured cgroup peak is the true need and, being cgroup-isolated,
/// can't be poisoned by external contention — see ADR 0046 anti-poison
/// note), VRAM is carried but DEFERRED (the conservative estimate still
/// gates VRAM; RAM is the over-refuse constraint this slice fixes).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FootprintEntry {
    /// Peak resident RAM observed (cgroup-attributed), bytes. MAX-merged.
    pub ram_bytes: u64,
    /// Peak VRAM, MiB. DEFERRED — recorded but the admission gate keeps
    /// the conservative estimate for VRAM (see resolve()).
    pub vram_mib: u64,
    /// How many runs have fed this key (monotone-increasing).
    pub n_samples: u32,
    /// Provenance / override rank.
    pub source: FootprintSource,
    /// Wall-clock of the last update (Unix seconds). Display/audit only.
    pub updated_unix: u64,
}

/// The footprint calibration store: a JSON map `flat_key → entry`,
/// loaded from [`crate::config::footprint_store_path`] and atomically
/// rewritten (tmp + rename, the `registry.rs::write_registry` pattern)
/// on each `record`.
///
/// MERGE SEMANTICS (`record`):
///   * RAM is MAX-merged — never shrinks below an observed peak.
///   * `source` follows the override rank (OomCorrected > Measured >
///     Default); a higher-rank source always wins, an equal-rank source
///     keeps the larger RAM.
///   * `n_samples` increments on every record.
///
/// RESOLVE: a calibrated (Measured/OomCorrected) entry replaces the
/// conservative `hint`'s RAM; absent → the `hint` is returned verbatim.
#[derive(Clone, Debug, Default)]
pub struct FootprintStore {
    path: PathBuf,
    entries: HashMap<String, FootprintEntry>,
}

impl FootprintStore {
    /// Load from the default store path
    /// ([`crate::config::footprint_store_path`]). A missing or corrupt
    /// file degrades to an empty store (a calibration miss only falls
    /// back to the conservative hint — never a hard failure).
    pub fn load() -> Self {
        Self::load_from(crate::config::footprint_store_path())
    }

    /// Load from an explicit path (tests pin a tempdir here).
    pub fn load_from(path: PathBuf) -> Self {
        let entries = std::fs::read_to_string(&path)
            .ok()
            .and_then(|body| serde_json::from_str::<HashMap<String, FootprintEntry>>(&body).ok())
            .unwrap_or_default();
        Self { path, entries }
    }

    /// In-memory entry count (test/inspection helper).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolve a footprint for `key`: the calibrated RAM (Measured /
    /// OomCorrected) if present, else the conservative `hint`. VRAM is
    /// DEFERRED — we keep the hint's VRAM regardless (RAM is the
    /// over-refuse constraint; recording VRAM peaks is a later slice).
    pub fn resolve(&self, key: &FootprintKey, hint: Footprint) -> Footprint {
        match self.entries.get(&key.flat()) {
            Some(e) if e.source != FootprintSource::Default => Footprint {
                ram_bytes: e.ram_bytes,
                // DEFER VRAM: keep the conservative estimate.
                vram_mib: hint.vram_mib,
            },
            _ => hint,
        }
    }

    /// Record a MEASURED peak for `key`, MAX-merging RAM and applying
    /// the source override rank, then atomically persist. `measured`
    /// carries `ram_bytes` (the cgroup-attributed peak) and a `source`
    /// (Measured for a clean exit, OomCorrected for an OOM lower bound).
    pub fn record(
        &mut self,
        key: &FootprintKey,
        ram_bytes: u64,
        vram_mib: u64,
        source: FootprintSource,
    ) -> std::io::Result<()> {
        let now = now_unix();
        let flat = key.flat();
        let merged = match self.entries.get(&flat).copied() {
            Some(prev) => {
                // Higher-rank source wins outright; equal/lower rank keeps
                // the larger RAM (monotone-up). The stored source is the
                // MAX rank ever seen so an OomCorrected lower bound is
                // never demoted by a later lucky Measured run.
                let winning_source = if source.rank() >= prev.source.rank() {
                    source
                } else {
                    prev.source
                };
                FootprintEntry {
                    ram_bytes: ram_bytes.max(prev.ram_bytes),
                    vram_mib: vram_mib.max(prev.vram_mib),
                    n_samples: prev.n_samples.saturating_add(1),
                    source: winning_source,
                    updated_unix: now,
                }
            }
            None => FootprintEntry {
                ram_bytes,
                vram_mib,
                n_samples: 1,
                source,
                updated_unix: now,
            },
        };
        self.entries.insert(flat, merged);
        self.save()
    }

    /// Atomic tmp+rename write (mirrors `registry.rs::write_registry`):
    /// a crash mid-write leaves the old store intact, never a partial.
    fn save(&self) -> std::io::Result<()> {
        use std::io::Write;
        let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;
        let body = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = self.path.with_extension("json.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(body.as_bytes())?;
            let _ = f.sync_all();
        }
        if let Err(e) = std::fs::rename(&tmp, &self.path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(())
    }
}

/// Current Unix time in seconds; 0 if the clock is before the epoch
/// (impossible in practice). Read from `SystemTime` in non-test code —
/// callers that need determinism pass an explicit `updated_unix` via the
/// stored entry, not through `record`.
fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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

    // ── calibration store (ADR 0046 slice-2) ──────────────────────────

    fn key() -> FootprintKey {
        footprint_key("lamquant_joint_codec", 4, 16, 3)
    }

    #[test]
    fn footprint_key_flat_is_stable_and_pipe_delimited() {
        assert_eq!(key().flat(), "lamquant_joint_codec|3|16|4");
    }

    #[test]
    fn store_roundtrip_write_then_read() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path.clone());
        s.record(&key(), 20 * GIB, 18_000, FootprintSource::Measured)
            .unwrap();
        // Re-load from disk: the entry persisted.
        let s2 = FootprintStore::load_from(path);
        assert_eq!(s2.len(), 1);
        let e = s2.entries.get(&key().flat()).unwrap();
        assert_eq!(e.ram_bytes, 20 * GIB);
        assert_eq!(e.n_samples, 1);
        assert_eq!(e.source, FootprintSource::Measured);
    }

    #[test]
    fn record_max_merges_ram_keeps_larger() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        s.record(&key(), 25 * GIB, 0, FootprintSource::Measured)
            .unwrap();
        // A SMALLER later peak must NOT shrink the stored cap.
        s.record(&key(), 20 * GIB, 0, FootprintSource::Measured)
            .unwrap();
        let e = s.entries.get(&key().flat()).unwrap();
        assert_eq!(e.ram_bytes, 25 * GIB, "max-merge must keep the larger peak");
        assert_eq!(e.n_samples, 2, "n_samples bumps on every record");
    }

    #[test]
    fn oom_corrected_overrides_measured_source() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        s.record(&key(), 30 * GIB, 0, FootprintSource::Measured)
            .unwrap();
        // An OOM at a SMALLER cap is still a known lower bound: the
        // SOURCE must promote to OomCorrected even though RAM didn't grow.
        s.record(&key(), 22 * GIB, 0, FootprintSource::OomCorrected)
            .unwrap();
        let e = s.entries.get(&key().flat()).unwrap();
        assert_eq!(e.source, FootprintSource::OomCorrected, "OomCorrected > Measured");
        assert_eq!(e.ram_bytes, 30 * GIB, "still max-merged");
        // And a later Measured run must NOT demote the source back.
        s.record(&key(), 31 * GIB, 0, FootprintSource::Measured)
            .unwrap();
        let e = s.entries.get(&key().flat()).unwrap();
        assert_eq!(e.source, FootprintSource::OomCorrected, "Measured must not demote");
        assert_eq!(e.ram_bytes, 31 * GIB);
    }

    #[test]
    fn resolve_returns_measured_when_present_else_hint() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        let hint = estimate(4, 16, 3, 256); // 31 GiB conservative
        // Absent → hint verbatim.
        assert_eq!(s.resolve(&key(), hint), hint);
        // Present (measured ~20G) → measured RAM, hint VRAM (deferred).
        s.record(&key(), 20 * GIB, 18_000, FootprintSource::Measured)
            .unwrap();
        let r = s.resolve(&key(), hint);
        assert_eq!(r.ram_bytes, 20 * GIB, "calibrated RAM admits at the measured ~20G");
        assert_eq!(r.vram_mib, hint.vram_mib, "VRAM stays the conservative estimate");
    }

    #[test]
    fn atomic_write_leaves_no_partial_tmp() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path.clone());
        s.record(&key(), 20 * GIB, 0, FootprintSource::Measured)
            .unwrap();
        // The tmp file is renamed away — no `.json.tmp` left behind.
        let tmp = path.with_extension("json.tmp");
        assert!(!tmp.exists(), "atomic rename must leave no partial tmp");
        assert!(path.exists(), "final store file must exist");
    }

    #[test]
    fn corrupt_store_degrades_to_empty() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        std::fs::write(&path, b"{ this is not json").unwrap();
        let s = FootprintStore::load_from(path);
        assert!(s.is_empty(), "corrupt store must degrade to empty, not panic");
    }
}
