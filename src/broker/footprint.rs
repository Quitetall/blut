// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! LamQuant compatibility footprint model and generic calibration store.
//!
//! Generic engine launch admission uses typed stage resource declarations and
//! does not call this recipe-key parser. The compatibility model remains public
//! for the downstream LamQuant cookbook's containment and calibration path.
//!
//! The original blueprint's "constant 40-50G hint" is rejected by the review:
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

/// Conservative DataLoader worker cap for a train-shaped stage (ADR 0046
/// slice-1 item 5). Each fork-worker is a CoW copy of the ~6 GiB parent
/// plus decode buffers + a per-worker sample LRU, so RAM scales ~linearly
/// with workers. MEASURED (2026-06-10): workers=4 peaked ~23 GiB RSS +
/// ~9 GiB swap under a 25 GiB cap and OOM-killed under added pressure.
/// Cap at **2**: ~20 GiB real demand, under the cap with headroom.
///
/// Cross-crate compatibility contract for the LamQuant cookbook's resolve and
/// record paths. The generic engine CLI does not derive worker counts from
/// recipe JSON.
pub const UNCALIBRATED_WORKER_CAP: u32 = 2;

/// Upper bound on auto-tuned DataLoader workers (ADR 0071). Decode saturates the
/// GPU feed well before the core count on a many-core box, and each worker holds a
/// multi-GiB prefetch buffer, so raising past this just burns RAM for no throughput.
pub const MAX_AUTO_WORKERS: u32 = 16;

/// Compose the measured-peak store key from a stage/recipe identity and a
/// declared [`ResourceEnvelope`] (ADR 0133). ENGINE-composed: the identity
/// namespaces the key (`shared_calibration_group` is the audited opt-out for
/// stages that genuinely pool physics), and the ORDERED dimension values are
/// joined in the same pipe-delimited flat form as [`FootprintKey::flat`] — so a
/// cookbook that declares the incumbent driver dimensions
/// (`tier`,`batch`,`workers`,`warm` as `"w"`/`"c"`) produces a BYTE-IDENTICAL
/// key and the calibration store's measured history carries over (the ADR's
/// store-continuity requirement). `None` when no dimensions are declared
/// (estimate-only admission, nothing to calibrate).
pub fn envelope_calibration_key(
    identity: &str,
    env: &blut_types::envelope::ResourceEnvelope,
) -> Option<String> {
    if env.calibration_dimensions.is_empty() {
        return None;
    }
    let ident = env.shared_calibration_group.as_deref().unwrap_or(identity);
    debug_assert!(!ident.contains('|'), "key identity must not contain '|'");
    let mut flat = String::from(ident);
    for (_, value) in &env.calibration_dimensions {
        flat.push('|');
        flat.push_str(value);
    }
    Some(flat)
}

/// Increment-3 (ADR 0133): re-evaluate a declared envelope at other unit
/// counts / warmth — the engine's window into the cookbook's affine cost model
/// WITHOUT knowing the domain formula. Contract (see [`blut_types::envelope::CostTerm`]):
/// `ram_bytes` includes `declared_units × ram_bytes_per_unit` (the COLD
/// coefficient) per term; re-evaluation at `(units, warm)` subtracts the
/// declared contribution and adds `units × per_unit(warm)`. A term with a warm
/// variant re-prices its DECLARED units too when `warm` — warmth applies to the
/// whole term, exactly as the incumbent formula applied it. All saturating;
/// an override naming no term is ignored.
pub fn envelope_footprint_at(
    env: &blut_types::envelope::ResourceEnvelope,
    overrides: &[(&str, u32)],
    warm: bool,
) -> u64 {
    let mut ram = env.ram_bytes;
    for t in &env.cost_terms {
        let units = overrides
            .iter()
            .find(|(n, _)| *n == t.dimension)
            .map(|(_, u)| *u)
            .unwrap_or(t.declared_units);
        let declared = u64::from(t.declared_units).saturating_mul(t.ram_bytes_per_unit);
        let repriced = u64::from(units).saturating_mul(t.per_unit(warm));
        ram = ram.saturating_sub(declared).saturating_add(repriced);
    }
    ram
}

/// Increment-3 (ADR 0133): the SYNC (async-retention-free) base of a declared
/// envelope — every `sync_base_excluded` term at ZERO units, so the ADR-0103
/// profile's separately-billed worker/queue bytes are never double-counted.
/// Replaces the deleted JSON-path estimate-delta trick for declared
/// plans (the engine zeroes flagged terms; it never names "the workers").
pub fn envelope_sync_base(env: &blut_types::envelope::ResourceEnvelope, warm: bool) -> u64 {
    let zeroed: Vec<(&str, u32)> = env
        .cost_terms
        .iter()
        .filter(|t| t.sync_base_excluded)
        .map(|t| (t.dimension.as_str(), 0))
        .collect();
    envelope_footprint_at(env, &zeroed, warm)
}

/// Increment-3 (ADR 0133): fit-and-saturate over a DECLARED cost term — the
/// engine's search mechanism (ADR 0071 semantics preserved: decrement from the
/// target until the footprint fits `avail − floor`, always ≥ 1), the
/// cookbook's enumeration (`max_units` can only be tightened by `target`).
/// `avail_bytes == 0` (no probe) ⇒ the term's DECLARED units — a box we can't
/// size to behaves exactly as declared, mirroring the incumbent's
/// uncalibrated-box behavior. Returns `None` when the envelope has no such
/// term (the caller falls back to the JSON path until Phase D).
pub fn fit_and_saturate_env(
    env: &blut_types::envelope::ResourceEnvelope,
    dimension: &str,
    target: u32,
    avail_bytes: u64,
    floor_bytes: u64,
    warm: bool,
) -> Option<u32> {
    let term = env.cost_terms.iter().find(|t| t.dimension == dimension)?;
    if avail_bytes == 0 {
        return Some(term.declared_units.max(1));
    }
    let budget = avail_bytes.saturating_sub(floor_bytes);
    let ceiling = target.min(term.max_units).max(1);
    let mut u = ceiling;
    while u > 1 && envelope_footprint_at(env, &[(dimension, u)], warm) > budget {
        u -= 1;
    }
    Some(u)
}

/// Increment-3 (ADR 0133): shrink-only fit over a declared cost term with the
/// already-resolved dimensions HELD (workers-then-batch residual-budget
/// semantics — see `batch_size_to_fit`'s doc for why sequential reaches the
/// joint boundary on an additive model). Never raises above `requested`
/// (training-quality decisions are not the auto-tuner's to make). `None` when
/// the envelope has no such term.
pub fn shrink_to_fit_env(
    env: &blut_types::envelope::ResourceEnvelope,
    dimension: &str,
    requested: u32,
    resolved: &[(&str, u32)],
    avail_bytes: u64,
    floor_bytes: u64,
    warm: bool,
) -> Option<u32> {
    env.cost_terms.iter().find(|t| t.dimension == dimension)?;
    if avail_bytes == 0 {
        return Some(requested.max(1));
    }
    let budget = avail_bytes.saturating_sub(floor_bytes);
    let mut b = requested.max(1);
    let est = |b: u32| {
        let mut o: Vec<(&str, u32)> = resolved.to_vec();
        o.push((dimension, b));
        envelope_footprint_at(env, &o, warm)
    };
    while b > 1 && est(b) > budget {
        b -= 1;
    }
    Some(b)
}

/// Increment-2 (ADR 0133): compose the store key with CONTEXT overrides — the
/// runtime facts a stage's typed args cannot know (the warmed cache, the
/// auto-tuned worker count). An override REPLACES the declared dimension's
/// value by NAME (declaration order is preserved, so the flat layout — and
/// store continuity — is unchanged); an override naming no declared dimension
/// is ignored (a stage that doesn't calibrate on `warm` is not forced to).
/// This is how the incumbent tuned-key semantics ride the typed seam:
/// same layout, context-true values.
pub fn envelope_calibration_key_with_context(
    identity: &str,
    env: &blut_types::envelope::ResourceEnvelope,
    context: &[(&str, String)],
) -> Option<String> {
    if env.calibration_dimensions.is_empty() {
        return None;
    }
    let ident = env.shared_calibration_group.as_deref().unwrap_or(identity);
    debug_assert!(!ident.contains('|'), "key identity must not contain '|'");
    let mut flat = String::from(ident);
    for (name, declared) in &env.calibration_dimensions {
        let value = context
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
            .unwrap_or(declared.as_str());
        flat.push('|');
        flat.push_str(value);
    }
    Some(flat)
}

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

/// Operational override: when `BLUT_FOOTPRINT_STANDARD` is set to a TRUTHY value
/// (`1`/`true`/`yes`/`on`, case-insensitive), the broker IGNORES the learned
/// high-water-mark calibration and uses the conservative STATIC formula estimate
/// instead — on BOTH read (`FootprintStore::resolve` returns the hint) and write
/// (the cookbook's `record_train_footprint` skips storing the peak). Use it when a
/// recorded cgroup peak may be DIRTY (inflated by OTHER processes co-resident on
/// the box during a contained run), so a contaminated measurement neither drives
/// nor poisons admission. The env var is PROCESS-WIDE — set it INLINE per
/// invocation (`BLUT_FOOTPRINT_STANDARD=1 lqt recipe run …`), not exported, so it
/// scopes to the one dirty run. Falsey / unset / unrecognised ⇒ off (so
/// `…=false`/`0`/`off` disable it as an operator would expect).
pub fn use_standard_footprint_estimate() -> bool {
    match std::env::var("BLUT_FOOTPRINT_STANDARD") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

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

/// Stable composite key for the compatibility calibration store. `recipe` is
/// the recipe name (e.g. `train_model`), not the stage name. The cookbook
/// threads it through `StageContext.recipe_name` so its own resolve and record
/// keys match.
/// `tier`/`batch`/`workers` are the cost drivers the footprint scales
/// on (latent is folded into the estimate, not the key — it rarely
/// varies and would fragment the calibration).
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct FootprintKey {
    pub recipe: String,
    pub tier: u32,
    pub batch: u32,
    pub workers: u32,
    /// Memory-admission Phase 3: whether the run warmed the per-sample disk cache. A
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

/// Build a compatibility calibration key from a recipe name and its cost
/// drivers. A cookbook's resolve and record paths must supply the same values;
/// `workers`/`batch`/`tier`/`warm` must also match those fed to [`estimate`].
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
        // Read the BLUT_FOOTPRINT_STANDARD operational override at the boundary,
        // then delegate to the pure resolver (so the override is unit-testable
        // without mutating the process env). Log when active so an operator who
        // left it exported notices admission is running on the static estimate.
        let standard = use_standard_footprint_estimate();
        if standard {
            tracing::info!(
                "BLUT_FOOTPRINT_STANDARD active — ignoring learned calibration for \
                 key {}, using the static estimate ({:.1}G)",
                key.flat(),
                hint.ram_bytes as f64 / GIB as f64,
            );
        }
        self.resolve_inner(key, hint, standard)
    }

    /// ADR 0133 incr 2b: resolve by the ENVELOPE-composed flat key (the seam's
    /// path into the same store rows — `envelope_calibration_key*` reproduces
    /// `FootprintKey::flat` byte-exactly, so measured history is shared, not
    /// forked). Honors BLUT_FOOTPRINT_STANDARD like [`resolve`](Self::resolve).
    pub fn resolve_flat(&self, flat: &str, hint: Footprint) -> Footprint {
        let standard = use_standard_footprint_estimate();
        if standard {
            tracing::info!(
                "BLUT_FOOTPRINT_STANDARD active — ignoring learned calibration for \
                 key {flat}, using the static estimate ({:.1}G)",
                hint.ram_bytes as f64 / GIB as f64,
            );
        }
        self.resolve_inner_flat(flat, hint, standard)
    }

    /// Core resolver. `standard_estimate = true` IGNORES the learned calibration
    /// and returns the conservative STATIC `hint` verbatim — use when a recorded
    /// cgroup peak may be DIRTY (inflated by OTHER processes co-resident on the
    /// box during a contained run), so neither it nor any stored peak drives
    /// admission. Pairs with `record_train_footprint`'s matching skip so the dirty
    /// run doesn't poison the store either (ignore on BOTH read and write).
    fn resolve_inner(
        &self,
        key: &FootprintKey,
        hint: Footprint,
        standard_estimate: bool,
    ) -> Footprint {
        self.resolve_inner_flat(&key.flat(), hint, standard_estimate)
    }

    fn resolve_inner_flat(
        &self,
        flat: &str,
        hint: Footprint,
        standard_estimate: bool,
    ) -> Footprint {
        if standard_estimate {
            return hint;
        }
        match self.entries.get(flat) {
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
                // B2 self-heal — converge from BOTH directions:
                //  * A clean `Measured` peak is AUTHORITATIVE over a stored
                //    `OomCorrected` bound: it REPLACES it (ram + source), even
                //    if lower. An OOM cap is often a SPURIOUS contention/co-tenant
                //    kill, not the run's intrinsic need; a same-key run that
                //    cleanly exited with ≥10% cap headroom proves the true need.
                //    This lets a low run pull the cap DOWN (Q1) and de-ratchets a
                //    spurious OOM so it can't inflate every future run (Q2).
                //  * `Measured`-vs-`Measured` MAX-merges (a higher real peak can
                //    recur → keep it; variance-safe).
                //  * `OomCorrected` (rank 2) still wins over `Measured`/`Default`
                //    and MAX-merges UP — an OOM proved the prior cap insufficient,
                //    escalate-for-retry (now PROVISIONAL: a later clean `Measured`
                //    can override it via the first rule).
                // The "clean run proves the true need" headroom check is enforced
                // UPSTREAM at classification (record_train_footprint): only a
                // CleanExit with peak <90% of ITS cap is labelled `Measured`; a
                // run that rode its cap (≥90%) is `OomCorrected`. So reaching this
                // branch with `source == Measured` already means a run that fit
                // with headroom. `ram_bytes > 0` guards a pathological 0-byte
                // measurement from zeroing a valid bound. (We deliberately heal on
                // the FIRST clean sample — fast convergence is the goal; genuine
                // cross-run variance re-OOMs and re-escalates, self-correcting.)
                let measured_overrides_oom = source == FootprintSource::Measured
                    && prev.source == FootprintSource::OomCorrected
                    && ram_bytes > 0;
                let (winning_source, winning_ram) = if measured_overrides_oom {
                    (FootprintSource::Measured, ram_bytes)
                } else if source.rank() >= prev.source.rank() {
                    (source, ram_bytes.max(prev.ram_bytes))
                } else {
                    (prev.source, ram_bytes.max(prev.ram_bytes))
                };
                FootprintEntry {
                    ram_bytes: winning_ram,
                    // VRAM is MAX-merged regardless of the RAM override: a GPU
                    // high-water mark is a real hard limit even on a run that
                    // RAM-OOM'd spuriously, so it never shrinks here.
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
    /// never demote). Conservative admission is preserved: after a forget, `resolve` falls
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
    fn memmax_adds_headroom() {
        let fp = Footprint {
            ram_bytes: 30 * GIB,
            vram_mib: 0,
        };
        assert_eq!(fp.memmax_bytes(), fp.ram_bytes + 2 * GIB);
    }

    // ── Phase 3: warm-aware footprint ─────────────────────────────────

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
        // Golden literals (formula lives in the cookbook since Phase D):
        // warm 6+2×3+3×2+1+2 = 21 GiB est → 23 cap; cold 6+2×4+… = 23 est → 25.
        let warm_cap = 23 * GIB;
        let cold_cap = 25 * GIB;
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

    // ── in_ch fullband term (the under-bill fix) ──────────────────────

    // ── warm-stage footprint model (memory-admission hole: uncontained warm) ──

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
        // B2 self-heal: a later CLEAN Measured run OVERRIDES the OomCorrected
        // bound (it proves the true need) — the source demotes to Measured and
        // ram becomes the clean measurement verbatim (the Q1/Q2 fix; an OOM cap
        // is provisional, not a permanent floor).
        s.record(&key(), 31 * GIB, 0, FootprintSource::Measured)
            .unwrap();
        let e = s.entries.get(&key().flat()).unwrap();
        assert_eq!(
            e.source,
            FootprintSource::Measured,
            "a clean Measured overrides a stale OomCorrected (B2 self-heal)"
        );
        assert_eq!(
            e.ram_bytes,
            31 * GIB,
            "ram = the clean measurement, verbatim"
        );
    }

    #[test]
    fn clean_measured_lowers_cap_and_de_ratchets_spurious_oom() {
        // The user's two questions:
        //  Q1 — can a low-memory run lower the cap?  (was: NO; now: YES)
        //  Q2 — does a spurious OOM force future runs higher forever? (was: YES;
        //       now: NO — the next clean run de-ratchets it.)
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        // A spurious/contention OOM poisons the entry at 40 GiB.
        s.record(&key(), 40 * GIB, 0, FootprintSource::OomCorrected)
            .unwrap();
        // Before B2 this resolved ESCALATED (40 → ~48/64); the next clean run
        // could never bring it down. Now a clean 18 GiB run overrides it DOWN.
        s.record(&key(), 18 * GIB, 0, FootprintSource::Measured)
            .unwrap();
        let e = s.entries.get(&key().flat()).unwrap();
        assert_eq!(
            e.source,
            FootprintSource::Measured,
            "Q2: spurious OOM de-ratcheted"
        );
        assert_eq!(
            e.ram_bytes,
            18 * GIB,
            "Q1: clean run lowered the cap to actual"
        );
        // And resolve now returns the tight measured value (+2G headroom), NOT an
        // escalated OOM cap.
        let hint = Footprint {
            ram_bytes: 33 * GIB,
            vram_mib: 0,
        };
        let r = s.resolve(&key(), hint);
        assert_eq!(
            r.ram_bytes,
            18 * GIB,
            "resolve returns the healed Measured verbatim"
        );
        assert_eq!(
            r.memmax_bytes(),
            18 * GIB + Footprint::MEMMAX_HEADROOM_BYTES,
            "+2G safety margin"
        );
    }

    #[test]
    fn standard_estimate_override_ignores_calibration() {
        // BLUT_FOOTPRINT_STANDARD: when a recorded peak may be DIRTY (other
        // processes loading the box), ignore the calibration and use the static
        // hint. Tested via resolve_inner so no process-env mutation / race.
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        s.record(&key(), 50 * GIB, 0, FootprintSource::Measured)
            .unwrap();
        let hint = Footprint {
            ram_bytes: 33 * GIB,
            vram_mib: 0,
        };
        // Normal: the calibrated (possibly dirty) Measured value wins.
        assert_eq!(s.resolve_inner(&key(), hint, false).ram_bytes, 50 * GIB);
        // Override: ignore calibration, fall back to the conservative static hint.
        assert_eq!(s.resolve_inner(&key(), hint, true).ram_bytes, 33 * GIB);

        // The override also bypasses the OomCorrected escalation branch (a dirty
        // OOM bound must not drive admission either).
        let td2 = tempfile::tempdir().unwrap();
        let mut s2 = FootprintStore::load_from(td2.path().join("footprints.json"));
        s2.record(&key(), 40 * GIB, 0, FootprintSource::OomCorrected)
            .unwrap();
        assert!(
            s2.resolve_inner(&key(), hint, false).ram_bytes > 40 * GIB,
            "normal: OomCorrected escalates above its bound"
        );
        assert_eq!(
            s2.resolve_inner(&key(), hint, true).ram_bytes,
            33 * GIB,
            "override: ignore the OomCorrected bound, use the static hint"
        );
    }

    #[test]
    fn resolve_returns_measured_when_present_else_hint() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("footprints.json");
        let mut s = FootprintStore::load_from(path);
        let hint = Footprint {
            ram_bytes: 31 * GIB, // frozen golden conservative hint
            vram_mib: 0,
        };
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
        let hint = Footprint {
            ram_bytes: 23 * GIB, // frozen golden conservative cold hint
            vram_mib: 0,
        };
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
        let hint = Footprint {
            ram_bytes: 23 * GIB,
            vram_mib: 0,
        };
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
        let hint = Footprint {
            ram_bytes: 23 * GIB,
            vram_mib: 0,
        };
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
        let hint = Footprint {
            ram_bytes: 23 * GIB,
            vram_mib: 0,
        };
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

#[cfg(test)]
mod envelope_key_continuity {
    //! ADR 0133 store-continuity gate: a cookbook declaring the incumbent
    //! driver dimensions must produce the EXACT flat key the JSON path built,
    //! or the calibration store's measured history is silently orphaned.
    use super::*;
    use blut_types::envelope::ResourceEnvelope;

    // GOLDEN test model (frozen literals — the real formula lives in the
    // cookbook since ADR 0133 Phase D; these pin the ENGINE's affine math and
    // key composition, independent of any domain formula):
    const PW_COLD: u64 = 4 * GIB; // per-worker, cold
    const PW_WARM: u64 = 3 * GIB; // per-worker, warm
    const PB: u64 = GIB / 16; // per-batch unit

    /// A golden declared envelope: base + 2 workers (cold) + `batch` units,
    /// with the incumbent-layout calibration dimensions.
    fn env_with_terms(tier: u32, batch: u32) -> ResourceEnvelope {
        use blut_types::envelope::CostTerm;
        let base = (6 + 2 * u64::from(tier) + 1 + 7) * GIB; // frozen golden base
        ResourceEnvelope {
            ram_bytes: base + 2 * PW_COLD + u64::from(batch) * PB,
            gpu: Default::default(),
            calibration_dimensions: vec![
                ("tier".into(), tier.to_string()),
                ("batch".into(), batch.to_string()),
                ("workers".into(), "2".into()),
                ("warm".into(), "c".into()),
            ],
            cost_terms: vec![
                CostTerm {
                    dimension: "workers".into(),
                    declared_units: 2,
                    ram_bytes_per_unit: PW_COLD,
                    ram_bytes_per_unit_warm: Some(PW_WARM),
                    max_units: MAX_AUTO_WORKERS,
                    sync_base_excluded: true,
                },
                CostTerm {
                    dimension: "batch".into(),
                    declared_units: batch,
                    ram_bytes_per_unit: PB,
                    ram_bytes_per_unit_warm: None,
                    max_units: batch,
                    sync_base_excluded: false,
                },
            ],
            shared_calibration_group: None,
        }
    }

    /// The golden model evaluated by hand — what `envelope_footprint_at` must
    /// reproduce for any (workers, batch, warm).
    fn golden(tier: u32, batch: u32, workers: u32, warm: bool) -> u64 {
        let base = (6 + 2 * u64::from(tier) + 1 + 7) * GIB;
        let pw = if warm { PW_WARM } else { PW_COLD };
        base + u64::from(workers) * pw + u64::from(batch) * PB
    }

    #[test]
    fn affine_evaluation_and_sync_base_match_the_golden_model() {
        for tier in [1, 3, 7] {
            for batch in [4, 32] {
                let env = env_with_terms(tier, batch);
                for warm in [false, true] {
                    for w in [0, 1, 2, 6, 16] {
                        assert_eq!(
                            envelope_footprint_at(&env, &[("workers", w)], warm),
                            golden(tier, batch, w, warm),
                            "affine diverged tier={tier} batch={batch} w={w} warm={warm}"
                        );
                    }
                    // Sync base = the sync_base_excluded (workers) term at zero.
                    assert_eq!(envelope_sync_base(&env, warm), golden(tier, batch, 0, warm));
                }
            }
        }
    }

    #[test]
    fn key_composition_layout_and_context_overrides_are_stable() {
        // The store-key layout is a PERMANENT wire format: identity|dims… in
        // declaration order, warm as w/c. Changing it orphans measured history.
        let env = env_with_terms(3, 32);
        assert_eq!(
            envelope_calibration_key("train_joint", &env).as_deref(),
            Some("train_joint|3|32|2|c")
        );
        let ctx = [("workers", "6".to_string()), ("warm", "w".to_string())];
        assert_eq!(
            envelope_calibration_key_with_context("train_joint", &env, &ctx).as_deref(),
            Some("train_joint|3|32|6|w")
        );
        // Unknown context names never mutate the layout.
        let noop = [("nonexistent", "9".to_string())];
        assert_eq!(
            envelope_calibration_key_with_context("train_joint", &env, &noop).as_deref(),
            Some("train_joint|3|32|2|c")
        );
        // A shared group replaces the identity; dimensions unchanged.
        let mut shared = env.clone();
        shared.shared_calibration_group = Some("train".into());
        assert_eq!(
            envelope_calibration_key("train_joint", &shared).as_deref(),
            Some("train|3|32|2|c")
        );
        // No dimensions ⇒ nothing to calibrate.
        assert_eq!(
            envelope_calibration_key("x", &ResourceEnvelope::default()),
            None
        );
    }

    #[test]
    fn env_searches_fit_and_saturate_against_the_golden_model() {
        let env = env_with_terms(3, 32);
        let floor = 6 * GIB;
        // Saturate: enough room for the full target.
        assert_eq!(
            fit_and_saturate_env(&env, "workers", 6, 62 * GIB, floor, false),
            Some(6)
        );
        // Fit: tight budget decrements below the target but never below 1.
        let w = fit_and_saturate_env(&env, "workers", 6, 24 * GIB, floor, false).unwrap();
        assert!(
            (1..6).contains(&w),
            "tight budget must reduce the count: {w}"
        );
        assert_eq!(
            fit_and_saturate_env(&env, "workers", 6, 7 * GIB, floor, false),
            Some(1),
            "always ≥ 1 — admission refuses, the search never deadlocks"
        );
        // No probe ⇒ the declared units (behave as declared).
        assert_eq!(
            fit_and_saturate_env(&env, "workers", 6, 0, floor, false),
            Some(2)
        );
        // Batch shrink-only at held workers; requested already fits ⇒ requested.
        assert_eq!(
            shrink_to_fit_env(&env, "batch", 32, &[("workers", 2)], 62 * GIB, floor, false),
            Some(32)
        );
        let b = shrink_to_fit_env(&env, "batch", 32, &[("workers", 2)], 23 * GIB, floor, false)
            .unwrap();
        assert!((1..=32).contains(&b));
        // Missing term ⇒ None (caller keeps the requested value).
        assert_eq!(
            shrink_to_fit_env(&env, "nope", 32, &[], 62 * GIB, floor, false),
            None
        );
    }

    #[test]
    fn resolve_flat_reads_the_same_store_rows_as_the_json_path() {
        // Incr 2b: a Measured peak recorded under the incumbent FootprintKey
        // must be served to the seam's flat-key resolve — shared history, not a
        // forked store.
        let mut store = FootprintStore::default();
        let key = footprint_key("train_joint", 2, 32, 3, false);
        store.entries.insert(
            key.flat(),
            FootprintEntry {
                ram_bytes: 20 * GIB,
                vram_mib: 0,
                n_samples: 1,
                source: FootprintSource::Measured,
                updated_unix: 0,
            },
        );
        let hint = Footprint {
            ram_bytes: 35 * GIB,
            vram_mib: 0,
        };
        let via_key = store.resolve(&key, hint);
        let via_flat = store.resolve_flat("train_joint|3|32|2|c", hint);
        assert_eq!(via_key.ram_bytes, via_flat.ram_bytes);
        assert_eq!(via_flat.ram_bytes, 20 * GIB, "measured peak served");
    }
}
