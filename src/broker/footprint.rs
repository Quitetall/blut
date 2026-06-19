// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Footprint estimation — the SCALING RAM model (ADR 0046, hole #3/#4).
//!
//! The blueprint's "constant 40-50G hint" is rejected by the review:
//! it is not an upper bound across tier / batch / latent-dim, so a
//! larger config can admit then OOM. The real driver of the RAM OOMs
//! on this box is the dataloader: `num_workers × per-worker prefetch
//! ≈ 23 GiB`. So the estimate is dominated by
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

/// Conservative DataLoader worker cap for a train-shaped stage (ADR 0046
/// slice-1 item 5). Each fork-worker is a CoW copy of the ~6 GiB parent
/// plus decode buffers + a per-worker sample LRU, so RAM scales ~linearly
/// with workers. MEASURED (2026-06-10): workers=4 peaked ~23 GiB RSS +
/// ~9 GiB swap under a 25 GiB cap and OOM-killed under added pressure.
/// Cap at **2**: ~20 GiB real demand, under the cap with headroom.
///
/// THE cross-crate contract: the cli admission gate (RESOLVE) clamps its
/// `workers` driver to `1..=UNCALIBRATED_WORKER_CAP` and the cookbook
/// train stage (RECORD) launches exactly this many — if the two ever
/// disagreed, the calibration key would never hit and the broker would
/// over-refuse forever. Lives here (the shared crate) so neither side
/// can drift from it (the prior copy lived in the cookbook and the cli
/// hard-coded a different `1..=4` clamp — a live parity bug).
pub const UNCALIBRATED_WORKER_CAP: u32 = 2;

/// The footprint cost drivers for a train-shaped recipe/stage, plus THE
/// single extraction from a recipe's args JSON. Both the cli admission
/// gate (RESOLVE) and the cookbook train stage (RECORD) build their
/// calibration key from this so the keys are byte-identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Drivers {
    /// DataLoader workers (the dominant RAM term), clamped
    /// `1..=UNCALIBRATED_WORKER_CAP`.
    pub workers: u32,
    /// Live mini-batch size (resolved default applied).
    pub batch: u32,
    /// Model tier (1..=8); larger ⇒ more model/optimizer RAM.
    pub tier: u32,
    /// Model latent width (0 ⇒ billed as the 256-wide default).
    /// Folded into the estimate, NOT the calibration key.
    pub latent: u32,
    /// Never-OOM Phase 3: the per-sample disk cache is warmed upstream
    /// (`warm_fb_cache` recipe arg). Lowers the per-worker term (the warm worker
    /// holds no whole-input decode) AND is part of the calibration key, so a
    /// warm `Measured` peak can never resolve a cold run (and vice versa).
    pub warm: bool,
    /// Model INPUT channels (`detail_bands`): 21 = the narrow baseline, 168 =
    /// the full input width (the wide default `--detail-bands all`). The 8×
    /// wider full-width front-end (wider model layers + SOAP preconditioners +
    /// the stacked dataloader input) costs materially more RAM than the narrow
    /// baseline — without this term a full-width launch billed identically to the
    /// baseline and was admitted then cgroup-killed. Folded into the ESTIMATE,
    /// not the key: the store's MAX-merge keeps the largest (full-width) peak per
    /// key, so a low baseline peak can never under-size a full-width run. SET VIA
    /// [`in_ch_from_args`] (or [`L3_ONLY_IN_CH`]/[`DEFAULT_IN_CH`]); the estimate
    /// rounds a non-multiple of 21 UP, so an arbitrary value is billed
    /// conservatively, never under.
    pub in_ch: u32,
}

/// Default model input channels when no `--detail-bands`/`--n` override is
/// present: the kernel's own default is `detail_bands='all'` (the full input
/// width → 168 ch), so a bare run IS full-width. Defaulting here to 168 (not 21)
/// is the load-bearing fix — the implicit full-width default must not be
/// under-billed.
pub const DEFAULT_IN_CH: u32 = 168;
/// Narrow-baseline model input (`--detail-bands none` / `--n none`).
pub const L3_ONLY_IN_CH: u32 = 21;

/// Map a `--detail-bands` / `--n` mode token to the conservative model in_ch.
/// `none` → narrow baseline (21). ANY other mode (`all` / `l3_detail` / …) → the
/// full input width (168), billed conservatively so a partial run is never
/// UNDER-sized (over-billing a partial width only over-provisions; the store
/// self-heals).
pub fn in_ch_from_detail_bands(mode: &str) -> u32 {
    if mode.trim().eq_ignore_ascii_case("none") {
        L3_ONLY_IN_CH
    } else {
        DEFAULT_IN_CH
    }
}

/// Resolve the model in_ch from a train invocation's passthrough args.
///
/// Precedence: `--detail-bands <m>` (or its `--n <m>` alias) in `extra_args`
/// wins; else `SNN_DETAIL_BANDS=<bands>` in `extra_env` (empty ⇒ none ⇒ 21);
/// else the kernel default `detail_bands='all'` ⇒ [`DEFAULT_IN_CH`] (168).
///
/// THE single shared derivation: the cli RESOLVE side (`from_args_json`) and the
/// cookbook RECORD side (the train stage) both call this so the billed in_ch —
/// hence the estimate — matches.
pub fn in_ch_from_args(extra_args: &[&str], extra_env: &[&str]) -> u32 {
    for (i, &tok) in extra_args.iter().enumerate() {
        for flag in ["--detail-bands", "--n"] {
            // Equals form `--detail-bands=<m>` / `--n=<m>`. (`--n` can't false-
            // match `--no-gan`: stripping `--n` leaves `o-gan`, no leading `=`.)
            if let Some(m) = tok.strip_prefix(flag).and_then(|r| r.strip_prefix('=')) {
                return in_ch_from_detail_bands(m);
            }
            // Space form `--detail-bands <m>` / `--n <m>`.
            if tok == flag {
                if let Some(&m) = extra_args.get(i + 1) {
                    return in_ch_from_detail_bands(m);
                }
            }
        }
    }
    for kv in extra_env {
        if let Some(val) = kv.strip_prefix("SNN_DETAIL_BANDS=") {
            // The trainer sets this to `''` for `detail_bands='none'`, so
            // empty ⇒ narrow baseline; tolerate a literal `none` too. Any band
            // list ⇒ full width (conservative).
            let v = val.trim();
            return if v.is_empty() || v.eq_ignore_ascii_case("none") {
                L3_ONLY_IN_CH
            } else {
                DEFAULT_IN_CH
            };
        }
    }
    DEFAULT_IN_CH
}

/// `in_ch` from a recipe args JSON (the RESOLVE side) — extracts the
/// `extra_args`/`extra_env` string arrays and defers to [`in_ch_from_args`].
fn in_ch_from_args_json(raw: &serde_json::Value) -> u32 {
    let strs = |key: &str| -> Vec<&str> {
        raw.get(key)
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default()
    };
    in_ch_from_args(&strs("extra_args"), &strs("extra_env"))
}

impl Drivers {
    /// Extract the cost drivers from a recipe's args JSON. Workers
    /// defaults to and is clamped by [`UNCALIBRATED_WORKER_CAP`] (the
    /// value the train stage actually launches), `batch` to
    /// [`DEFAULT_BATCH`], `tier` to 3 (the train-recipe default), and
    /// `latent` is parsed from a `--encoder-width N` token in
    /// `extra_args` (0 = unspecified).
    pub fn from_args_json(raw: &serde_json::Value) -> Self {
        let u32_or = |key: &str, default: u32| -> u32 {
            raw.get(key)
                .and_then(|v| v.as_u64())
                // saturate, never wrap-to-0 (would under-bill)
                .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
                .unwrap_or(default)
        };
        let workers = u32_or("workers", UNCALIBRATED_WORKER_CAP).clamp(1, UNCALIBRATED_WORKER_CAP);
        let batch = u32_or("batch_size", DEFAULT_BATCH);
        let tier = u32_or("tier", 3);
        let latent = raw
            .get("extra_args")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter()
                    .position(|x| x.as_str() == Some("--encoder-width"))
                    .and_then(|i| arr.get(i + 1))
                    .and_then(|x| x.as_str())
                    .and_then(|s| s.parse::<u32>().ok())
            })
            .unwrap_or(0);
        // `warm_fb_cache` (Phase 2/3): present + true on the warm-by-default
        // train recipe; ABSENT ⇒ false (the conservative cold term) so a recipe
        // that doesn't warm is never under-billed. The cli RESOLVE side reads
        // it here; the cookbook RECORD side reads the same flag off the train
        // stage's args — they must agree or the calibration key never hits.
        let warm = raw
            .get("warm_fb_cache")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Model in_ch from the detail-band mode (default 168 = the kernel's
        // implicit `detail_bands='all'`); the under-bill fix for full width.
        let in_ch = in_ch_from_args_json(raw);
        Self {
            workers,
            batch,
            tier,
            latent,
            warm,
            in_ch,
        }
    }

    /// Build directly from the resolved drivers a train stage launches
    /// with (RECORD side). Clamps `workers` to the cap so a stage that
    /// passes a raw count still keys identically to the cli.
    pub fn new(workers: u32, batch: u32, tier: u32, latent: u32, warm: bool, in_ch: u32) -> Self {
        Self {
            workers: workers.clamp(1, UNCALIBRATED_WORKER_CAP),
            batch,
            tier,
            latent,
            warm,
            in_ch,
        }
    }

    /// The conservative-high RAM/VRAM estimate for these drivers.
    pub fn estimate(&self) -> Footprint {
        estimate(
            self.workers,
            self.batch,
            self.tier,
            self.latent,
            self.warm,
            self.in_ch,
        )
    }

    /// The calibration key for these drivers under `recipe`. Latent is
    /// folded into the estimate, not the key (it rarely varies and would
    /// fragment the calibration). `warm` IS part of the key (a warm peak
    /// must never resolve a cold run).
    pub fn key(&self, recipe: &str) -> FootprintKey {
        footprint_key(recipe, self.workers, self.batch, self.tier, self.warm)
    }
}

/// Per-DataLoader-worker prefetch RAM (CoW fork + decode buffers +
/// per-worker sample LRU).
///
/// MEASURED (2026-06-10): a tier-3 warm run at workers=4 peaks ~23 GiB
/// RESIDENT *plus ~9 GiB swap* under a 25 GiB cgroup cap — i.e. its true
/// working set is ~32 GiB, the cap forced the overflow to swap and it
/// OOM-killed under any added pressure. So ~3.7-6 GiB/worker is the real
/// envelope; the earlier 2.0 GiB UNDER-sized it (the cap then sat at the
/// peak with no headroom → OOM-on-pressure). 4.0 GiB/worker is the honest
/// upper-mid, so a workers=2 run (the new default cap) bills ~23 GiB /
/// caps ~25 GiB over a ~20 GiB real demand — real headroom, no swap. The
/// calibration store refines per key; a cgroup cap (estimate + headroom)
/// hard-bounds any under-shoot to a unit kill, never a box OOM.
const PREFETCH_PER_WORKER_BYTES: u64 = 4 * GIB;

/// Per-DataLoader-worker RAM when the per-sample disk cache is WARM
/// (never-OOM Phase 2: the warm-cache stage ran upstream). With every used
/// sample already on disk, the adapter's disk tier hits FIRST and the in-proc
/// whole-input LRU stays EMPTY, so the per-worker resident set collapses to
/// CoW-fork + a reclaimable mmap page + the small per-sample LRU — NOT a whole
/// large input. The cold [`PREFETCH_PER_WORKER_BYTES`] (4 GiB) was sized for the
/// PRE-Phase-1/2 worker that held + re-decoded a whole input every epoch (the
/// OOM driver); the warm worker's true set is ~1.5-2.5 GiB.
///
/// Set conservative-HIGH at 3 GiB (a 25% cut, not the full ~40%) because no
/// post-warm clean run has been MEASURED yet — the store auto-tightens DOWN
/// from the first warm `Measured` peak (resolve returns it verbatim), and the
/// 90%-rode-cap → `OomCorrected` → escalate self-heal bounds any under-shoot to
/// a unit kill (never a box OOM). So this is the cold-START hint only; the
/// calibration store does the rest. A run that DIDN'T warm bills the higher
/// cold term (the `warm` flag is part of the calibration key, so warm + cold
/// runs of the same recipe never share — nor poison — an entry).
const PREFETCH_PER_WORKER_BYTES_WARM: u64 = 3 * GIB;

/// Base RSS floor: python + torch + CUDA context + framework overhead,
/// independent of workers/batch. Conservative-high.
const BASE_RSS_BYTES: u64 = 6 * GIB;

/// Model + optimizer-state RAM per model tier (SOAP keeps
/// preconditioners; bill generously). Multiplied by `tier`.
const PER_TIER_BYTES: u64 = 2 * GIB;

/// RAM per 256 units of model latent width (the latent-dim knob,
/// tasks #270/#271). Conservative; mostly host-side staging buffers.
const PER_LATENT256_BYTES: u64 = GIB;

/// Host-side RAM that scales with the live mini-batch (pinned buffers,
/// collation staging), per unit of batch. Small vs the worker term —
/// the actual batch tensors live on the GPU; only the CPU collation /
/// pinned-staging buffers for a handful of samples are host RAM, so
/// 64 MiB/unit (batch 32 ⇒ 2 GiB) is realistic. The prior 256 MiB/unit
/// double-counted the dataloader's own batch staging (already in the
/// worker term) and inflated batch-32 to a spurious 8 GiB.
const PER_BATCH_BYTES: u64 = GIB / 16; // 64 MiB / batch unit

/// RAM per extra 21-channel group of model input beyond the narrow baseline
/// (`in_ch` > 21). The full input width (`detail_bands='all'` → 168 ch = 8
/// groups) drives an 8× wider model front-end (wider conv/linear layers + their
/// SOAP preconditioners) plus the stacked `[in_ch, 313]` dataloader input — none
/// of which the 21-ch baseline carries. So the full width bills `(8-1) × 1 GiB =
/// +7 GiB` over the baseline. Conservative-high (a full-width tier-3 truly needs
/// ~30 GiB vs the baseline-shaped ~23 GiB estimate that was admitted then
/// cgroup-killed); the store self-heals DOWN from the first full-width
/// `Measured` peak.
const PER_INCH_GROUP_BYTES: u64 = GIB;

// ── OOM-correction growth (R2 / ADR 0046 slice-3) ────────────────────────
// An `OomCorrected` entry stores the cgroup cap that was HIT on an OOM — a
// known LOWER bound on the true need, not the true need itself. `resolve`
// must therefore return a cap STRICTLY ABOVE the stored bound, so a retry
// never sits back down on the same OOMing cap (the perpetual-OOM bug).
// Growth is multiplicative (scales with job size) with an additive floor
// (guarantees a meaningful bump for a small bound); `resolve` takes the
// MAX of the two so small jobs use the step and large jobs use the factor.
// NUM/DEN MUST stay 5/4 (=1.25) — change BOTH or neither.
const OOM_GROWTH_NUM: u64 = 5; // ×5/4 = +25% per observed OOM
const OOM_GROWTH_DEN: u64 = 4;
/// Additive growth floor: 8 GiB so a single OOM cycle converges for the
/// common mid-range under-estimate (e.g. a 25G cap that truly needs ~32G:
/// 25+8=33G clears it in ONE retry, vs +4G→31G which would OOM again and
/// take two cycles — each cycle is a wasted, expensive training run, so a
/// medical-grade self-heal converges fast). The `.max(hint)` floor and the
/// box-fit admission gate keep this from over-refusing small jobs.
const OOM_GROWTH_STEP_BYTES: u64 = 8 * GIB;
/// Secondary runaway guard on the escalated cap. The AUTHORITATIVE box-fit
/// refusal lives in `admission.rs` (it refuses when the footprint would
/// leave < the free-RAM floor); this clamp only stops a pathological
/// repeated-OOM key from walking the cap to an absurd value before
/// admission gets to refuse. 64 GiB (not the plan's ~44G) so it covers a
/// future larger box; on a 62G box the clamp returns 64G and admission
/// refuses (the intended fail-closed exit). At/above the ceiling the cap
/// stops growing — admission's box-fit refusal is then the only exit.
const OOM_RESOLVE_CEILING_BYTES: u64 = 64 * GIB;

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
/// RECIPE name (e.g. `train_model`), NOT the stage name — the
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
    /// Never-OOM Phase 3: whether the run warmed the per-sample disk cache. A
    /// warm run's per-worker footprint is much lower, so warm + cold runs MUST
    /// key separately — else a warm `Measured` peak resolves a cold run and
    /// under-sizes it (and an OomCorrected cold bound over-refuses a warm run).
    pub warm: bool,
}

impl FootprintKey {
    /// Flat string form used as the JSON map key. Pipe-delimited so it
    /// round-trips unambiguously (recipe names never contain `|`).
    pub fn flat(&self) -> String {
        // `|` is the delimiter — a recipe name containing it would collide two
        // distinct keys (e.g. `a|b`+tier1 vs `a`+tier`b|1`). Recipe names are
        // internal identifiers (never user free-text), so a debug_assert catches
        // a violation at test time without a release-path cost.
        debug_assert!(
            !self.recipe.contains('|'),
            "recipe name must not contain '|'"
        );
        // `warm` is the trailing segment (`w`/`c`) so the key partitions warm vs
        // cold calibration. NOTE: this changes the flat format — pre-Phase-3
        // entries (4-segment, no warm/cold suffix) become unreachable, a
        // deliberate one-time reset (their cold-regime peaks are invalid for the
        // re-modeled warm worker; `load_from` debug-logs the count).
        format!(
            "{}|{}|{}|{}|{}",
            self.recipe,
            self.tier,
            self.batch,
            self.workers,
            if self.warm { "w" } else { "c" }
        )
    }
}

/// Build the calibration key from a recipe name + the cost drivers.
/// THE single shared constructor: both the cli admission gate (RESOLVE)
/// and the train stage (RECORD) call this so the keys are byte-identical
/// — if they diverged the calibration would never be hit and the broker
/// would over-refuse forever. `workers`/`batch`/`tier`/`warm` MUST be the
/// same values fed to [`estimate`].
pub fn footprint_key(
    recipe: &str,
    workers: u32,
    batch: u32,
    tier: u32,
    warm: bool,
) -> FootprintKey {
    FootprintKey {
        recipe: recipe.to_string(),
        tier,
        batch,
        workers,
        warm,
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
/// - `tier`: model tier (1..=4); larger tier ⇒ more model/opt RAM.
/// - `latent_dim`: model latent width (0 ⇒ default, billed as 256).
/// - `warm`: the per-sample disk cache was warmed upstream (Phase 2) ⇒ the
///   per-worker term drops to [`PREFETCH_PER_WORKER_BYTES_WARM`] (no
///   whole-input decode held).
/// - `in_ch`: model input channels (21 = narrow baseline, 168 = full input
///   width) ⇒ a `(in_ch/21 − 1) × `[`PER_INCH_GROUP_BYTES`] full-width term, so a
///   168-ch run is no longer billed like a 21-ch run.
pub fn estimate_ram_bytes(
    workers: u32,
    batch: u32,
    tier: u32,
    latent_dim: u32,
    warm: bool,
    in_ch: u32,
) -> u64 {
    let workers = workers as u64;
    let batch = batch as u64;
    let tier = tier.max(1) as u64; // tier 0 is nonsensical; floor at 1
    // Treat an unspecified latent (0) as the default 256-wide model
    // so the model term is never under-counted.
    let latent = if latent_dim == 0 { 256 } else { latent_dim } as u64;

    let per_worker = if warm {
        PREFETCH_PER_WORKER_BYTES_WARM
    } else {
        PREFETCH_PER_WORKER_BYTES
    };
    let workers_term = workers.saturating_mul(per_worker);
    let tier_term = tier.saturating_mul(PER_TIER_BYTES);
    // ceil-div by 256 so any latent > 0 bills at least one unit.
    let latent_units = latent.div_ceil(256);
    let latent_term = latent_units.saturating_mul(PER_LATENT256_BYTES);
    let batch_term = batch.saturating_mul(PER_BATCH_BYTES);
    // Full-width front-end: extra 21-ch groups beyond the narrow baseline. 168
    // ch ⇒ (168/21 − 1) = 7 groups ⇒ +7 GiB; 21 ch ⇒ 0. `div_ceil` rounds a
    // non-multiple UP (a 30-ch model bills 1 group, never 0 — conservative,
    // never under-bills); `max(21)` floors so a sub-baseline value can't wrap.
    let in_ch_groups = (in_ch.max(L3_ONLY_IN_CH).div_ceil(L3_ONLY_IN_CH)).saturating_sub(1) as u64;
    let inch_term = in_ch_groups.saturating_mul(PER_INCH_GROUP_BYTES);

    BASE_RSS_BYTES
        .saturating_add(workers_term)
        .saturating_add(tier_term)
        .saturating_add(latent_term)
        .saturating_add(batch_term)
        .saturating_add(inch_term)
}

/// Convenience: build a [`Footprint`] from the cost drivers (RAM
/// scaled, VRAM left unknown for slice-1).
pub fn estimate(
    workers: u32,
    batch: u32,
    tier: u32,
    latent_dim: u32,
    warm: bool,
    in_ch: u32,
) -> Footprint {
    Footprint {
        ram_bytes: estimate_ram_bytes(workers, batch, tier, latent_dim, warm, in_ch),
        vram_mib: 0,
    }
}

// ── Warm-stage (parallel cache precompute) RAM model ──────────────────────
//
// DISTINCT from the train scaling formula above. The warm-cache stage forks N
// copy-on-write workers, each driving the SAME dataset adapter over a
// contiguous slice of the sample list. CPython refcount writes defeat CoW on
// the fork-inherited sample index, and each worker holds ~one decoded input
// + its fp16 cast buffer — so each worker's RSS climbs toward a near-full
// private copy. This is the hole that OOM'd the BOX (the prior flat
// `MEMORY_GIB = 8` reservation under ~6 workers × ~6 GiB ≈ 36 GiB real, with NO
// cgroup cap to catch the overshoot). The warm stage now bills + caps from this
// model, exactly as the train stage does from `estimate`.

/// Never-OOM cap on warm fork workers (the warm-side analogue of
/// [`UNCALIBRATED_WORKER_CAP`]). The warm is a one-time precompute, so
/// box-survival dominates throughput: 4 workers is near the validated ~5.5×
/// speedup knee, and [`warm_workers_for_budget`] drops it further on a box that
/// can't hold the cap's footprint.
pub const WARM_WORKER_CAP: u32 = 4;

/// Parent-process RSS floor of the warm driver: python + the sample index +
/// the decode of the first input + framework overhead,
/// independent of worker count. Conservative-high (mirrors [`BASE_RSS_BYTES`]).
const WARM_BASE_RSS_BYTES: u64 = 6 * GIB;

/// Per-fork-worker RSS: a CoW-defeated near-full copy of the inherited sample
/// index plus the worker's own one-input decode + fp16 cast buffer. Raised
/// 6→8 GiB to match the MEASURED warm peak: a 4-worker warm rode ~23 GiB RSS +
/// ~9 GiB swap = ~32 GiB true working set (≈8 GiB/worker), so the prior 6 GiB
/// under-sized it → the 32 GiB cgroup cap was ridden → OOM-kill → partial cache.
/// 8 GiB makes `warm_estimate` bill the real peak so `warm_workers_for_budget`
/// reduces workers BEFORE the OOM (raise-to-measured is always the safe
/// direction; cf. the never-lower-an-unproven-cap rule).
const PER_WARM_WORKER_BYTES: u64 = 8 * GIB;

/// Conservative-high peak RSS (bytes) of the warm stage at `workers` fork
/// workers: `base + workers × per_worker`. Monotone in `workers`; floors at 1
/// worker (a serial warm still pays the base + one worker's set).
pub fn warm_ram_bytes(workers: u32) -> u64 {
    let w = workers.max(1) as u64;
    WARM_BASE_RSS_BYTES.saturating_add(w.saturating_mul(PER_WARM_WORKER_BYTES))
}

/// A [`Footprint`] for the warm stage at `workers` (RAM scaled, VRAM 0 — the
/// warm is CPU + disk only). `memmax_bytes()` adds the standard 2 GiB headroom.
pub fn warm_estimate(workers: u32) -> Footprint {
    Footprint {
        ram_bytes: warm_ram_bytes(workers),
        vram_mib: 0,
    }
}

/// Pick the warm worker count that stays box-safe: the largest
/// `w ∈ 1..=min(requested, WARM_WORKER_CAP)` whose cgroup cap
/// (`warm_estimate(w).memmax_bytes()`) fits `budget_bytes` (the box-fit RAM
/// ceiling, `MemTotal − floor`). `budget_bytes == 0` (probe unavailable) skips
/// the box-fit reduction and returns `min(requested, WARM_WORKER_CAP)`. Always
/// ≥ 1 — a single worker is the floor even on a box too small for its cap (the
/// cgroup then kills the unit rather than the box; the warm fails closed and
/// the contained trainer re-decodes + self-heals).
pub fn warm_workers_for_budget(requested: u32, budget_bytes: u64) -> u32 {
    let ceil = requested.clamp(1, WARM_WORKER_CAP);
    if budget_bytes == 0 {
        return ceil;
    }
    let mut w = ceil;
    while w > 1 && warm_estimate(w).memmax_bytes() > budget_bytes {
        w -= 1;
    }
    w
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
        // Phase 3: a pre-warm-key entry has 4 pipe-delimited segments (no
        // trailing `w`/`c`); the warm-aware key has 5. Such entries no longer
        // resolve, so their calibration is IGNORED until a new run re-measures
        // under the warm/cold key. This is SAFE (the fallback is the
        // conservative estimate, which the cgroup cap + OOM self-heal backstop)
        // — debug-log it so an operator wondering why calibration "reset" can
        // see it, without spamming the warn channel on every load.
        let stale = entries
            .keys()
            .filter(|k| k.matches('|').count() == 3)
            .count();
        if stale > 0 {
            tracing::debug!(
                "footprint store: {stale} pre-Phase-3 entries (no warm/cold key suffix) \
                 are ignored — re-calibration needed for those configs"
            );
        }
        Self { path, entries }
    }

    /// In-memory entry count (test/inspection helper).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolve a footprint for `key`:
    ///   * `Measured` (clean-exit cgroup peak) → the calibrated RAM verbatim
    ///     (it is the true need, monotone-up via record's max-merge).
    ///   * `OomCorrected` (a cap that was HIT on OOM) → a cap ESCALATED
    ///     strictly above the stored lower bound (R2), never below the
    ///     conservative `hint`, clamped by the runaway guard. This is the
    ///     self-heal: an OOM grows the next cap instead of re-sitting on it.
    ///   * `Default` / absent → the conservative `hint` verbatim.
    ///
    /// VRAM is DEFERRED — we keep the hint's VRAM regardless (RAM is the
    /// over-refuse constraint; recording VRAM peaks is a later slice).
    pub fn resolve(&self, key: &FootprintKey, hint: Footprint) -> Footprint {
        match self.entries.get(&key.flat()) {
            Some(e) if e.source == FootprintSource::OomCorrected => {
                // Grow strictly above the OOMing lower bound; never below
                // the conservative hint; clamped by OOM_RESOLVE_CEILING_BYTES
                // (a runaway guard) — admission.rs is the authoritative
                // box-fit refusal.
                let grown = (e.ram_bytes.saturating_mul(OOM_GROWTH_NUM) / OOM_GROWTH_DEN)
                    .max(e.ram_bytes.saturating_add(OOM_GROWTH_STEP_BYTES));
                Footprint {
                    ram_bytes: grown.max(hint.ram_bytes).min(OOM_RESOLVE_CEILING_BYTES),
                    vram_mib: hint.vram_mib,
                }
            }
            Some(e) if e.source == FootprintSource::Measured => Footprint {
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

    /// Sorted snapshot of `(flat_key, entry)` pairs for `blut footprint list`.
    pub fn entries_snapshot(&self) -> Vec<(String, FootprintEntry)> {
        let mut v: Vec<(String, FootprintEntry)> =
            self.entries.iter().map(|(k, e)| (k.clone(), *e)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// Forget ONE calibration entry by its flat key, then persist. Returns
    /// whether an entry existed.
    ///
    /// The SANCTIONED, audited way to clear a stale `OomCorrected` bound that no
    /// longer reflects reality (e.g. after a data-pipeline memory fix dropped the
    /// true peak below the recorded OOM cap, which `record`'s monotone rank can
    /// never demote). Never-OOM is preserved: after a forget, `resolve` falls
    /// back to the conservative `Default` hint and the cgroup cap still
    /// hard-bounds the run, so the next clean exit records a fresh `Measured`.
    pub fn forget(&mut self, key_flat: &str) -> std::io::Result<bool> {
        let removed = self.entries.remove(key_flat).is_some();
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    /// Forget EVERY entry for a recipe (all `"<recipe>|*"` keys), then persist.
    /// Returns the count removed. For `blut footprint forget --recipe <name>`
    /// after a change that invalidates the whole recipe's calibration.
    pub fn forget_recipe(&mut self, recipe: &str) -> std::io::Result<usize> {
        if recipe.is_empty() {
            // Guard the full-store-wipe footgun: an empty prefix matches EVERY
            // key. A blanket reset must be an explicit, separate action.
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "forget_recipe: empty recipe name would match every key",
            ));
        }
        let prefix = format!("{recipe}|");
        let before = self.entries.len();
        self.entries.retain(|k, _| !k.starts_with(&prefix));
        let removed = before - self.entries.len();
        if removed > 0 {
            self.save()?;
        }
        Ok(removed)
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
        // Write+sync+rename in one fallible step; clean up the tmp on ANY
        // failure (now that sync_all propagates, a sync error must not leave
        // an orphaned .json.tmp behind, same as a rename error).
        let result = (|| -> std::io::Result<()> {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(body.as_bytes())?;
            f.sync_all()?;
            std::fs::rename(&tmp, &self.path)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
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
        let r = estimate_ram_bytes(0, 0, 0, 0, false, 21);
        assert!(r >= BASE_RSS_BYTES, "got {r}");
        // floors: base + tier1 + latent256 = 6 + 2 + 1 = 9 GiB
        assert_eq!(r, 9 * GIB);
    }

    #[test]
    fn monotone_in_workers() {
        let lo = estimate_ram_bytes(2, 16, 3, 256, false, 21);
        let hi = estimate_ram_bytes(8, 16, 3, 256, false, 21);
        assert!(hi > lo, "workers must increase RAM: {lo} !< {hi}");
        // workers dominate: +6 workers × 4 GiB = +24 GiB
        assert_eq!(hi - lo, 6 * PREFETCH_PER_WORKER_BYTES);
    }

    #[test]
    fn monotone_in_batch() {
        let lo = estimate_ram_bytes(4, 8, 3, 256, false, 21);
        let hi = estimate_ram_bytes(4, 32, 3, 256, false, 21);
        assert!(hi > lo, "batch must increase RAM: {lo} !< {hi}");
    }

    #[test]
    fn monotone_in_tier() {
        let lo = estimate_ram_bytes(4, 16, 1, 256, false, 21);
        let hi = estimate_ram_bytes(4, 16, 4, 256, false, 21);
        assert!(hi > lo, "tier must increase RAM: {lo} !< {hi}");
    }

    #[test]
    fn monotone_in_latent() {
        let lo = estimate_ram_bytes(4, 16, 3, 256, false, 21);
        let hi = estimate_ram_bytes(4, 16, 3, 512, false, 21);
        assert!(hi > lo, "latent_dim must increase RAM: {lo} !< {hi}");
    }

    #[test]
    fn workers_term_dominates() {
        // The whole point (hole #4): RAM is workers-driven, not
        // batch-driven. Doubling workers must move RAM more than
        // doubling batch from the same baseline.
        let base = estimate_ram_bytes(4, 16, 3, 256, false, 21);
        let more_workers = estimate_ram_bytes(8, 16, 3, 256, false, 21);
        let more_batch = estimate_ram_bytes(4, 32, 3, 256, false, 21);
        assert!(
            more_workers - base > more_batch - base,
            "workers must dominate batch: dW={} dB={}",
            more_workers - base,
            more_batch - base
        );
    }

    #[test]
    fn memmax_adds_headroom() {
        let fp = estimate(4, 16, 3, 256, false, 21);
        assert_eq!(fp.memmax_bytes(), fp.ram_bytes + 2 * GIB);
    }

    #[test]
    fn conservative_default_admits_one_train_on_62g_box() {
        // The load-bearing slice-1 property: the uncalibrated default
        // (capped workers ≤ 4) is conservative-high but still fits ONE
        // train on the 62 GiB box with the 6 GiB floor.
        let fp = estimate(4, 16, 3, 256, false, 21);
        // 6 + 4×4 + 3×2 + 1 + 16×64MiB = 6+16+6+1+1 = 30 GiB
        assert_eq!(fp.ram_bytes, 30 * GIB);
        assert!(
            fp.ram_bytes < (62 - 6) * GIB,
            "must fit one train on 62G box"
        );
    }

    #[test]
    fn cold_tier3_cap_exceeds_measured_workers2_demand() {
        // R3 regression pin: at the capped worker count (UNCALIBRATED_WORKER_CAP
        // = 2 in blut-lamquant), the COLD tier-3 cap must exceed the MEASURED
        // workers=2 true working set (~16-20 GiB, DEV_LOG 2026-06-10 db39698),
        // so a cold run never OOMs at the cap. A future constant tweak that
        // re-under-sizes the hint (the 51bcc43 bug) trips this test.
        let cold_cap = estimate(2, 32, 3, 256, false, 21).memmax_bytes();
        // 6 + 2×4 + 3×2 + 1 + 32×64MiB = 23 GiB estimate, +2 GiB headroom = 25 GiB.
        // Pin the ACTUAL cap (24G threshold = the 25G cap with 1G slack), not
        // a loose ">demand" floor — a constant tweak that drops the cold cap
        // below the measured ~20G workers-2 demand (the 51bcc43 bug) trips this.
        assert!(
            cold_cap >= 24 * GIB,
            "cold tier-3 workers-2 cap {cold_cap} must hold the ~25G right-sized value \
             (>> the ~20G measured demand)"
        );
    }

    // ── Phase 3: warm-aware footprint ─────────────────────────────────

    #[test]
    fn warm_lowers_per_worker_term_only() {
        // The warm flag drops ONLY the per-worker term (no whole-recording
        // decode held); base/tier/latent/batch are unchanged.
        let cold = estimate_ram_bytes(2, 32, 3, 256, false, 21);
        let warm = estimate_ram_bytes(2, 32, 3, 256, true, 21);
        assert!(
            warm < cold,
            "warm must be tighter than cold: {warm} !< {cold}"
        );
        // Δ = workers × (cold_per_worker − warm_per_worker) = 2 × (4−3) GiB.
        assert_eq!(
            cold - warm,
            2 * (PREFETCH_PER_WORKER_BYTES - PREFETCH_PER_WORKER_BYTES_WARM)
        );
    }

    #[test]
    fn warm_key_differs_from_cold() {
        // A warm run and a cold run of the same drivers MUST key separately so
        // a warm Measured peak can never resolve a cold run (and vice versa).
        let w = footprint_key("lamquant_joint_codec", 2, 32, 3, true);
        let c = footprint_key("lamquant_joint_codec", 2, 32, 3, false);
        assert_ne!(w.flat(), c.flat());
        assert!(w.flat().ends_with("|w"));
        assert!(c.flat().ends_with("|c"));
    }

    #[test]
    fn warm_tier3_cap_still_holds_its_demand() {
        // The re-modeled warm cap (workers-2 tier-3) must still exceed the warm
        // true working set. 6 + 2×3 + 3×2 + 1 + 32×64MiB = 21 GiB est, +2 = 23.
        // Pin it ABOVE the conservative warm demand (~20 GiB) yet BELOW the cold
        // 25 GiB cap — the tightening Phase 3 delivers, without re-OOMing.
        let warm_cap = estimate(2, 32, 3, 256, true, 21).memmax_bytes();
        let cold_cap = estimate(2, 32, 3, 256, false, 21).memmax_bytes();
        assert!(
            warm_cap < cold_cap,
            "warm cap must be tighter: {warm_cap} !< {cold_cap}"
        );
        // Pin the EXACT cap so a future constant drift is caught concretely:
        // 6 + 2×3 + 3×2 + 1 + 32×64MiB = 21 GiB estimate, +2 GiB headroom = 23.
        assert_eq!(
            warm_cap,
            23 * GIB,
            "warm tier-3 workers-2 cap must be exactly 23G"
        );
        assert!(
            warm_cap >= 22 * GIB,
            "warm cap {warm_cap} must still hold the ~20G warm demand with headroom"
        );
    }

    #[test]
    fn drivers_from_args_reads_warm_flag() {
        // RESOLVE side: the warm flag comes off the recipe args JSON. Absent ⇒
        // cold (conservative). Present+true ⇒ warm.
        let cold = Drivers::from_args_json(&serde_json::json!({"tier": 3}));
        assert!(!cold.warm, "absent warm_fb_cache ⇒ cold");
        let warm = Drivers::from_args_json(&serde_json::json!({"warm_fb_cache": true, "tier": 3}));
        assert!(warm.warm, "warm_fb_cache=true ⇒ warm");
        // And the warm estimate is tighter than the cold one for the same args.
        assert!(warm.estimate().ram_bytes < cold.estimate().ram_bytes);
    }

    // ── in_ch fullband term (the under-bill fix) ──────────────────────

    #[test]
    fn fullband_in_ch_adds_term_over_l3() {
        // 168-ch fullband bills (168/21 − 1) = 7 GiB OVER the 21-ch L3 baseline —
        // previously they were identical (the admit-then-cgroup-kill bug).
        let l3 = estimate_ram_bytes(2, 32, 3, 256, false, 21);
        let fb = estimate_ram_bytes(2, 32, 3, 256, false, 168);
        assert!(fb > l3, "fullband must bill more than L3: {fb} !> {l3}");
        assert_eq!(fb - l3, 7 * PER_INCH_GROUP_BYTES, "168ch ⇒ +7 groups");
        // L3 baseline (21) adds nothing; a sub-baseline in_ch never wraps negative.
        assert_eq!(
            estimate_ram_bytes(2, 32, 3, 256, false, 0),
            estimate_ram_bytes(2, 32, 3, 256, false, 21),
            "in_ch < baseline floors at 21 (no wrap)"
        );
        // A non-multiple rounds UP (conservative): 30 ch ⇒ 1 group, not 0.
        assert_eq!(
            estimate_ram_bytes(2, 32, 3, 256, false, 30) - l3,
            PER_INCH_GROUP_BYTES,
            "30ch rounds up to 1 group (never under-bills)"
        );
    }

    #[test]
    fn in_ch_from_detail_bands_mapping() {
        assert_eq!(in_ch_from_detail_bands("none"), L3_ONLY_IN_CH);
        assert_eq!(in_ch_from_detail_bands("NONE"), L3_ONLY_IN_CH);
        assert_eq!(in_ch_from_detail_bands("all"), DEFAULT_IN_CH);
        assert_eq!(in_ch_from_detail_bands("l3_detail"), DEFAULT_IN_CH);
    }

    #[test]
    fn in_ch_from_args_precedence() {
        // --detail-bands wins.
        assert_eq!(in_ch_from_args(&["--detail-bands", "none"], &[]), 21);
        assert_eq!(in_ch_from_args(&["--detail-bands", "all"], &[]), 168);
        // --n alias.
        assert_eq!(in_ch_from_args(&["--n", "none"], &[]), 21);
        // Equals form (shell convention) — must parse too.
        assert_eq!(in_ch_from_args(&["--detail-bands=none"], &[]), 21);
        assert_eq!(in_ch_from_args(&["--n=all"], &[]), 168);
        // `--n` must NOT false-match `--no-gan` etc.
        assert_eq!(in_ch_from_args(&["--no-gan"], &[]), 168);
        // SNN_DETAIL_BANDS env: empty (the kernel's `none`) or literal `none`
        // ⇒ L3; a band list ⇒ fullband.
        assert_eq!(in_ch_from_args(&[], &["SNN_DETAIL_BANDS="]), 21);
        assert_eq!(in_ch_from_args(&[], &["SNN_DETAIL_BANDS=none"]), 21);
        assert_eq!(in_ch_from_args(&[], &["SNN_DETAIL_BANDS=l3_detail"]), 168);
        // Nothing ⇒ the kernel default detail_bands='all' ⇒ fullband (the fix).
        assert_eq!(in_ch_from_args(&[], &[]), 168);
    }

    #[test]
    fn drivers_default_is_fullband_then_overridable() {
        // RESOLVE side: a bare joint run defaults to fullband (168) — the implicit
        // 'all' default that was being under-billed.
        let bare = Drivers::from_args_json(&serde_json::json!({"tier": 3}));
        assert_eq!(bare.in_ch, 168, "bare joint run is fullband by default");
        // Explicit L3-only drops the fullband term.
        let l3 = Drivers::from_args_json(
            &serde_json::json!({"tier": 3, "extra_args": ["--detail-bands", "none"]}),
        );
        assert_eq!(l3.in_ch, 21);
        assert!(
            l3.estimate().ram_bytes < bare.estimate().ram_bytes,
            "L3 bills less than fullband"
        );
    }

    // ── warm-stage footprint model (never-OOM hole: uncontained warm) ──

    #[test]
    fn warm_ram_is_base_plus_per_worker_monotone() {
        // The warm bills base + workers × per-worker, monotone-up in workers —
        // the replacement for the flat 8 GiB that under-billed the fork pool.
        let w1 = warm_ram_bytes(1);
        let w4 = warm_ram_bytes(4);
        assert_eq!(w1, WARM_BASE_RSS_BYTES + PER_WARM_WORKER_BYTES);
        assert_eq!(w4, WARM_BASE_RSS_BYTES + 4 * PER_WARM_WORKER_BYTES);
        assert!(w4 > w1, "more workers ⇒ more RAM");
        // Floors at 1 worker: 0 bills the same as 1 (a serial warm still pays).
        assert_eq!(warm_ram_bytes(0), warm_ram_bytes(1));
        // The 4-worker cap blows the prior flat 8 GiB reservation out of the
        // water — that mismatch (30 GiB real vs 8 GiB billed) is the box-OOM.
        assert!(w4 > 8 * GIB, "warm cap must dwarf the old flat 8 GiB lie");
    }

    #[test]
    fn warm_estimate_adds_headroom() {
        let fp = warm_estimate(2);
        assert_eq!(fp.vram_mib, 0, "warm is CPU + disk only");
        assert_eq!(fp.memmax_bytes(), warm_ram_bytes(2) + 2 * GIB);
    }

    #[test]
    fn warm_workers_for_budget_reduces_to_fit_box() {
        // The cap (4) costs base+4×per = 6+32 = 38 GiB est, +2 = 40 GiB cap
        // (PER_WARM_WORKER_BYTES raised 6→8 GiB to match the measured ~8 GiB/
        // worker warm peak — see the const's doc).
        let cap4 = warm_estimate(4).memmax_bytes();
        assert_eq!(cap4, 40 * GIB);
        // A box that can hold the cap keeps all 4.
        assert_eq!(warm_workers_for_budget(4, 56 * GIB), 4);
        // A tighter box steps workers DOWN until the cap fits. memmax(w)=8+8w:
        // w2=24G, w3=32G, w4=40G.
        assert_eq!(warm_estimate(2).memmax_bytes(), 24 * GIB);
        assert_eq!(
            warm_workers_for_budget(4, 24 * GIB),
            2,
            "2-worker cap (24G) fits a 24G box"
        );
        assert_eq!(
            warm_workers_for_budget(4, 23 * GIB),
            1,
            "only 1 worker (16G) fits 23G"
        );
        // Never below 1 even on an impossibly small box (the unit-kill floor).
        assert_eq!(warm_workers_for_budget(4, GIB), 1);
        // requested clamps to the cap; budget 0 (no probe) skips the reduction.
        assert_eq!(warm_workers_for_budget(99, 0), WARM_WORKER_CAP);
        assert_eq!(warm_workers_for_budget(2, 0), 2);
        assert_eq!(warm_workers_for_budget(0, 0), 1, "requested 0 floors at 1");
    }

    // ── calibration store (ADR 0046 slice-2) ──────────────────────────

    fn key() -> FootprintKey {
        footprint_key("lamquant_joint_codec", 4, 16, 3, false)
    }

    #[test]
    fn footprint_key_flat_is_stable_and_pipe_delimited() {
        assert_eq!(key().flat(), "lamquant_joint_codec|3|16|4|c");
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
        assert_eq!(
            e.source,
            FootprintSource::OomCorrected,
            "OomCorrected > Measured"
        );
        assert_eq!(e.ram_bytes, 30 * GIB, "still max-merged");
        // And a later Measured run must NOT demote the source back.
        s.record(&key(), 31 * GIB, 0, FootprintSource::Measured)
            .unwrap();
        let e = s.entries.get(&key().flat()).unwrap();
        assert_eq!(
            e.source,
            FootprintSource::OomCorrected,
            "Measured must not demote"
        );
        assert_eq!(e.ram_bytes, 31 * GIB);
    }

    #[test]
    fn resolve_returns_measured_when_present_else_hint() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        let hint = estimate(4, 16, 3, 256, false, 21); // 31 GiB conservative
        // Absent → hint verbatim.
        assert_eq!(s.resolve(&key(), hint), hint);
        // Present (measured ~20G) → measured RAM, hint VRAM (deferred).
        s.record(&key(), 20 * GIB, 18_000, FootprintSource::Measured)
            .unwrap();
        let r = s.resolve(&key(), hint);
        assert_eq!(
            r.ram_bytes,
            20 * GIB,
            "calibrated RAM admits at the measured ~20G"
        );
        assert_eq!(
            r.vram_mib, hint.vram_mib,
            "VRAM stays the conservative estimate"
        );
    }

    #[test]
    fn resolve_escalates_oom_corrected_above_bound() {
        // R2: an OOM at 24G is a LOWER bound; resolve must return a cap
        // STRICTLY ABOVE it so a retry doesn't re-sit on the OOMing cap.
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        let hint = estimate(2, 16, 3, 256, false, 21); // conservative cold hint
        s.record(&key(), 24 * GIB, 0, FootprintSource::OomCorrected)
            .unwrap();
        let r = s.resolve(&key(), hint);
        // max(24×5/4=30, 24+8=32) = 32 GiB, above the 24G bound.
        assert_eq!(r.ram_bytes, 32 * GIB, "OOM bound must grow, not re-sit");
        assert!(
            r.ram_bytes > 24 * GIB,
            "escalated cap must exceed the OOMing cap"
        );
    }

    #[test]
    fn resolve_oom_corrected_never_below_hint() {
        // A small OOM bound must never resolve below the conservative hint.
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        let hint = estimate(2, 16, 3, 256, false, 21);
        s.record(&key(), 8 * GIB, 0, FootprintSource::OomCorrected)
            .unwrap();
        let r = s.resolve(&key(), hint);
        assert!(r.ram_bytes >= hint.ram_bytes, "never below the cold hint");
    }

    #[test]
    fn resolve_oom_corrected_clamped_to_ceiling() {
        // A pathological large OOM bound clamps at the runaway guard so the
        // number stays sane; admission.rs makes the box-fit refusal.
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        let hint = estimate(2, 16, 3, 256, false, 21);
        s.record(&key(), 60 * GIB, 0, FootprintSource::OomCorrected)
            .unwrap();
        let r = s.resolve(&key(), hint);
        assert_eq!(
            r.ram_bytes, OOM_RESOLVE_CEILING_BYTES,
            "clamped at the guard"
        );
    }

    #[test]
    fn resolve_monotone_after_repeated_oom() {
        // Each OOM at a higher cap raises the stored bound (max-merge) and
        // resolve always grows above it — strictly non-decreasing, no loop.
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        let hint = estimate(2, 16, 3, 256, false, 21);
        s.record(&key(), 24 * GIB, 0, FootprintSource::OomCorrected)
            .unwrap();
        let r1 = s.resolve(&key(), hint).ram_bytes; // max(30, 32)=32G
        // A retry OOMs at the escalated 32G cap → record it.
        s.record(&key(), r1, 0, FootprintSource::OomCorrected)
            .unwrap();
        let r2 = s.resolve(&key(), hint).ram_bytes; // max(32×5/4=40, 32+8=40)=40G
        assert!(r2 > r1, "repeated OOM must keep escalating: {r1} !< {r2}");
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
        assert!(
            s.is_empty(),
            "corrupt store must degrade to empty, not panic"
        );
    }
}
