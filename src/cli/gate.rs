// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Broker admission math used by launches: recipe footprints, admitted
//! workers/batch, box-fit budget, and the gated cell-run wrapper.
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

/// ADR 0133 increment 2b: the UNTUNED launch footprint through the TYPED seam.
///
/// Applies when a plan node declares an envelope with calibration dimensions.
/// The envelope's args-only estimate is the hint; the ENGINE-composed key (the
/// RECIPE name as identity, so the SAME store rows the JSON path calibrates)
/// resolves against the measured-peak store. Returns `None` for undeclared
/// plans, where the JSON fallback applies unchanged. STRANGLER SHADOW: the
/// JSON path is still computed and any divergence is warned loudly (the parity
/// gate keeps this green in CI; the double-compute dies at Phase D). The TUNED
/// path, whose worker/batch overrides change the ESTIMATE itself, stays on the
/// JSON formula until the envelope's structured cost terms land (increment 3).
pub(super) fn plan_footprint_declared(
    plan: &crate::framework::CompiledPlan,
    recipe: &str,
    raw: &serde_json::Value,
) -> Option<crate::broker::Footprint> {
    let (stage_name, env) = plan.max_declared_envelope()?;
    let flat = crate::broker::footprint::envelope_calibration_key(recipe, &env)?;
    let hint = crate::broker::Footprint {
        ram_bytes: env.ram_bytes,
        vram_mib: 0,
    };
    let resolved = crate::broker::FootprintStore::load().resolve_flat(&flat, hint);
    // Shadow referee (runtime defense on top of the CI parity gate).
    let incumbent = recipe_footprint(recipe, raw);
    if incumbent.ram_bytes != resolved.ram_bytes {
        tracing::warn!(
            "ADR 0133 shadow divergence for {recipe} (stage {stage_name}, key {flat}): \
             typed {}G vs JSON {}G — investigate before Phase D",
            resolved.ram_bytes / (1024 * 1024 * 1024),
            incumbent.ram_bytes / (1024 * 1024 * 1024),
        );
    }
    Some(resolved)
}

/// Estimate a job's RAM footprint from the recipe's raw args JSON
/// (ADR 0046 slice-1). Best-effort + recipe-agnostic: blut can't see
/// the cookbook's typed Args, so it reads the well-known cost-driver
/// keys directly off the JSON, defaulting CONSERVATIVELY when absent so
/// a recipe that omits them still gates oversubscription rather than
/// admitting blind. The dominant term is `workers` (dataloader
/// prefetch); since the train stage caps uncalibrated workers at 4 and
/// blut can't read that cap here, we mirror the cap as the default.
///
/// The scaling formula itself lives in `broker::footprint` so the cli
/// admission gate and the cookbook's train stage share ONE source of
/// truth.
/// Extract the footprint cost drivers `(workers, batch, tier, latent)`
/// from a recipe's raw args JSON, applying the SAME conservative
/// defaults the train stage's `train_containment` uses. PURE + testable:
/// this is the RESOLVE-side half of the calibration key parity (the
/// RECORD side is the cookbook's `train_containment`). If the two
/// diverged the calibration would never be hit and the broker would
/// over-refuse forever — `footprint_key_parity` pins them equal.
///
///   * `workers` — env-only in the cookbook (`LMA_NUM_WORKERS`), so the
///     JSON rarely carries it; default to the 4-worker uncalibrated cap
///     (matches `UNCALIBRATED_WORKER_CAP`). Clamped 1..=4.
///   * `batch` — `batch_size` JSON field or broker `DEFAULT_BATCH`.
///   * `tier` — `tier` JSON field or 3 (matches the joint recipe default).
///   * `latent` — `--encoder-width N` in `extra_args` (folded into the
///     estimate, NOT the key).
pub(super) fn recipe_footprint(name: &str, raw: &serde_json::Value) -> crate::broker::Footprint {
    // A recipe invoked with NO args (null or an empty object) is billed a LIGHT
    // base footprint, not the conservative trainer estimate. A heavy data-trainer
    // always declares required args (data roots, a manifest), so an arg-less
    // recipe is a lightweight in-process workflow; without this a trivial zero-arg
    // recipe is billed the full trainer ~30G and refused on a loaded box. Safe for
    // the trainer recipes (they always carry args → the estimate path below). A
    // general per-recipe DECLARED footprint is a tracked post-1.0 addition (API.md).
    if raw.is_null() || raw.as_object().is_some_and(|o| o.is_empty()) {
        return crate::broker::Footprint {
            ram_bytes: 2 * 1024 * 1024 * 1024,
            vram_mib: 0,
        };
    }
    // THE single shared extraction (RESOLVE side). `Drivers::from_args_json`
    // clamps workers to `UNCALIBRATED_WORKER_CAP` and resolves batch/tier the
    // same way the train stage's `train_containment` does (RECORD side), so
    // the calibration key built below is byte-identical to the one the stage
    // records under — the prior copy here clamped `1..=4` while the stage
    // capped at 2, so an explicit `workers:4` config never calibrated.
    let drivers = crate::broker::Drivers::from_args_json(raw);
    // The conservative-high estimate (over-refuses) — the fallback when
    // no calibration exists for this key.
    let hint = drivers.estimate();
    // ADR 0046 slice-2: if a MEASURED peak exists for this exact
    // (recipe,tier,batch,workers) key, resolve admits at the real
    // footprint (~20G) instead of the conservative hint (~35G). A miss
    // is benign: `resolve` returns the hint, so admission stays safe.
    let key = drivers.key(name);
    crate::broker::FootprintStore::load().resolve(&key, hint)
}

/// Like [`recipe_footprint`] but with the decode worker count OVERRIDDEN to the
/// auto-tuned `workers` (ADR 0071 A2), and OPTIONALLY the batch size too (E2,
/// extends the same auto-tune to a second knob) — so the gate's footprint +
/// calibration key reflect what the stage will actually launch (threaded via
/// `ExecCtx::with_admitted_workers`/`with_admitted_batch_size`). The light
/// arg-less path is unchanged.
pub(super) fn recipe_footprint_tuned(
    name: &str,
    raw: &serde_json::Value,
    workers: u32,
    batch: Option<u32>,
) -> crate::broker::Footprint {
    if raw.is_null() || raw.as_object().is_some_and(|o| o.is_empty()) {
        return crate::broker::Footprint {
            ram_bytes: 2 * 1024 * 1024 * 1024,
            vram_mib: 0,
        };
    }
    let mut drivers = crate::broker::Drivers::from_args_json(raw);
    drivers.workers = workers; // the fit-and-saturate count (overrides the cap)
    if let Some(b) = batch {
        drivers.batch = b; // the fit-and-saturate batch (E2, overrides the request)
    }
    let hint = drivers.estimate();
    let key = drivers.key(name);
    crate::broker::FootprintStore::load().resolve(&key, hint)
}

/// Synchronous/base half of a train-shaped footprint for ADR 0103 profiles.
/// The selected profile separately bills worker process RSS and every retained
/// async queue payload, so this intentionally evaluates the existing model at
/// zero workers to avoid counting those bytes twice. Batch/model/tier/input
/// terms remain in the base because they exist on the inline path too.
pub(super) fn recipe_footprint_sync_base(
    raw: &serde_json::Value,
    batch: Option<u32>,
    resolved_workers: u32,
    resolved: crate::broker::Footprint,
) -> crate::broker::Footprint {
    if raw.is_null() || raw.as_object().is_some_and(|object| object.is_empty()) {
        return crate::broker::Footprint {
            ram_bytes: resolved.ram_bytes.max(2 * 1024 * 1024 * 1024),
            vram_mib: resolved.vram_mib,
        };
    }
    let mut drivers = crate::broker::Drivers::from_args_json(raw);
    if let Some(batch) = batch {
        drivers.batch = batch;
    }
    let formula_sync = crate::broker::footprint::estimate(
        0,
        drivers.batch,
        drivers.tier,
        drivers.latent,
        drivers.warm,
        drivers.in_ch,
    );
    let formula_with_workers = crate::broker::footprint::estimate(
        resolved_workers,
        drivers.batch,
        drivers.tier,
        drivers.latent,
        drivers.warm,
        drivers.in_ch,
    );
    let known_worker_term = formula_with_workers
        .ram_bytes
        .saturating_sub(formula_sync.ram_bytes);
    // Preserve calibration/OOM-correction conservatively. Removing only the
    // known formula worker term leaves every unexplained measured excess in
    // the synchronous base; the profile then adds its explicit worker/queue
    // terms without discarding a previously raised safety bound.
    let resolved_minus_worker = resolved.ram_bytes.saturating_sub(known_worker_term);
    crate::broker::Footprint {
        ram_bytes: formula_sync.ram_bytes.max(resolved_minus_worker),
        vram_mib: formula_sync.vram_mib.max(resolved.vram_mib),
    }
}

pub(super) fn training_io_live_budget_bytes(
    snapshot: &crate::broker::ResourceSnapshot,
    launch_target: crate::config::launcher::LaunchTarget,
) -> Option<u64> {
    let snapshot_available = snapshot.mem_total_gb.is_finite()
        && snapshot.mem_total_gb > 0.0
        && snapshot.mem_avail_gb.is_finite()
        && snapshot.mem_avail_gb > 0.0;
    if !snapshot_available || launch_target != crate::config::launcher::LaunchTarget::Local {
        return None;
    }
    let gib = crate::broker::footprint::GIB as f64;
    let available_bytes = (snapshot.mem_avail_gb * gib) as u64;
    let floor_bytes = (crate::broker::admission::DEFAULT_FLOOR_GIB * gib) as u64;
    Some(available_bytes.saturating_sub(floor_bytes))
}

/// Resolve ADR 0103 profile admission from the same immutable snapshot and
/// calibrated footprint used by the tenant gate. This is shared by fresh runs
/// and resume so their downgrade/refusal behavior cannot drift.
#[allow(clippy::too_many_arguments)]
pub(super) fn configure_training_io_admission(
    label: &str,
    plan: crate::framework::plan::CompiledPlan,
    ctx: &mut crate::framework::ExecCtx,
    raw: &serde_json::Value,
    admitted_workers: Option<u32>,
    admitted_batch_size: Option<u32>,
    resolved_footprint: crate::broker::Footprint,
    snapshot: &crate::broker::ResourceSnapshot,
    launch_target: crate::config::launcher::LaunchTarget,
) -> anyhow::Result<(
    crate::framework::plan::CompiledPlan,
    crate::broker::Footprint,
)> {
    use crate::framework::TrainingIoDowngradeReason;

    // Fresh launch and resume must derive cache warmth from the same defaulted
    // recipe args before candidates/base bytes are evaluated. Keeping this in
    // their shared helper prevents resume from silently reverting to the cold
    // profile for an otherwise identical recipe.
    ctx.fb_warm = crate::broker::Drivers::from_args_json(raw).warm;
    let live_budget_bytes = training_io_live_budget_bytes(snapshot, launch_target);
    ctx.training_io_selection_budget_bytes = live_budget_bytes;
    let downgrade_reason = if ctx.sync_io {
        Some(TrainingIoDowngradeReason::UserForced)
    } else if launch_target != crate::config::launcher::LaunchTarget::Local {
        Some(TrainingIoDowngradeReason::UnsupportedLauncher)
    } else if live_budget_bytes.is_none() {
        Some(TrainingIoDowngradeReason::SnapshotUnavailable)
    } else {
        None
    };
    ctx.set_training_io_downgrade_reason(downgrade_reason);

    let resolved_workers =
        admitted_workers.unwrap_or_else(|| crate::broker::Drivers::from_args_json(raw).workers);
    let sync_footprint = recipe_footprint_sync_base(
        raw,
        admitted_batch_size,
        resolved_workers,
        resolved_footprint,
    );
    // The production launch path deliberately supports one declaring stage,
    // so its fastest-fit decision must use the WHOLE calibrated/OOM-corrected
    // synchronous base that tenant admission and containment will enforce.
    ctx.training_io_whole_job_base_bytes = Some(sync_footprint.ram_bytes);
    let plan = crate::framework::executor::prepare_plan_for_execution(plan, ctx)
        .map_err(|error| anyhow::anyhow!("{label} execution preparation: {error}"))?;
    if ctx.training_io_profiles.len() > 1 {
        return Err(anyhow::anyhow!(
            "{label} declares async-I/O profiles on {} nodes; whole-job admission currently supports exactly one declaring training node",
            ctx.training_io_profiles.len()
        ));
    }

    let Some(profile) = ctx.training_io_profiles.values().next() else {
        // The base override is unused when no stage opts into the profile seam;
        // preserve the exact legacy footprint.
        return Ok((plan, resolved_footprint));
    };
    if profile.sync_base_bytes < sync_footprint.ram_bytes {
        return Err(anyhow::anyhow!(
            "{label} selected profile base {} bytes is below calibrated floor {} bytes",
            profile.sync_base_bytes,
            sync_footprint.ram_bytes
        ));
    }
    let mut admitted = sync_footprint;
    admitted.ram_bytes = profile
        .sync_base_bytes
        .checked_add(profile.billed_overhead_bytes)
        .ok_or_else(|| anyhow::anyhow!("{label} async-I/O whole-job footprint overflow"))?;
    Ok((plan, admitted))
}

/// ADR 0071 A2: auto-tune the decode worker count to FIT-AND-SATURATE from a SINGLE
/// memory snapshot. Returns `Some(W)` for a train-shaped recipe (so RESOLVE +
/// RECORD share the cached count), or `None` for a light/arg-less recipe or when
/// the mem probe is unavailable (keep the conservative cap = unchanged behaviour).
/// Prints the `workers N→W` admission note when it changes the count.
pub(super) fn admitted_workers_for(
    name: &str,
    raw: &serde_json::Value,
    snap: &crate::broker::ResourceSnapshot,
) -> Option<u32> {
    if raw.is_null() || raw.as_object().is_some_and(|o| o.is_empty()) {
        return None; // light recipe — no decode workers to tune
    }
    if snap.mem_total_gb <= 0.0 {
        return None; // no probe → leave the conservative cap
    }
    let gib = crate::broker::footprint::GIB as f64;
    let cpu = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4);
    let avail = (snap.mem_avail_gb * gib) as u64;
    let floor = (crate::broker::admission::DEFAULT_FLOOR_GIB * gib) as u64;
    // Critically-low RAM (less than the floor free): don't tune — fall back to the
    // conservative cap and let the existing gate refuse on the cap footprint.
    if avail <= floor {
        return None;
    }
    // This snapshot is serialized BLUT-vs-BLUT by the scheduler lock and nets
    // out other processes via MemAvailable. It reduces over-admission risk but
    // cannot guarantee against drift or uncontained processes.
    let base = crate::broker::Drivers::from_args_json(raw);
    let w = crate::broker::footprint::workers_to_fit_and_saturate(cpu, avail, floor, &base);
    if w != crate::broker::footprint::UNCALIBRATED_WORKER_CAP {
        eprintln!(
            "admission: recipe '{name}' decode workers {} → {w} to fit {:.0}G available + {} cores (auto-tuned estimate)",
            crate::broker::footprint::UNCALIBRATED_WORKER_CAP,
            snap.mem_avail_gb,
            cpu
        );
    }
    Some(w)
}

/// E2: auto-tune the batch size to FIT the RAM budget, extending ADR 0071 A2's
/// fit-and-saturate auto-tune to a second knob. Runs ONLY when `resolved_workers`
/// came from [`admitted_workers_for`] on this SAME admission call (batch is
/// searched against the residual budget after workers is already fixed — see
/// [`crate::broker::footprint::batch_size_to_fit`]'s doc comment for why this
/// reaches the same feasibility boundary a joint search would). Returns
/// `Some(B)` only when it actually LOWERS the recipe's requested batch (unlike
/// workers, there is no "saturate up" direction — see that function's doc);
/// `None` means "the requested batch already fits, launch it unchanged."
pub(super) fn admitted_batch_size_for(
    name: &str,
    raw: &serde_json::Value,
    resolved_workers: u32,
    snap: &crate::broker::ResourceSnapshot,
) -> Option<u32> {
    if raw.is_null() || raw.as_object().is_some_and(|o| o.is_empty()) {
        return None; // light recipe — no batch to tune
    }
    if snap.mem_total_gb <= 0.0 {
        return None; // no probe → leave the requested batch
    }
    let gib = crate::broker::footprint::GIB as f64;
    let avail = (snap.mem_avail_gb * gib) as u64;
    let floor = (crate::broker::admission::DEFAULT_FLOOR_GIB * gib) as u64;
    if avail <= floor {
        return None;
    }
    let base = crate::broker::Drivers::from_args_json(raw);
    let requested = base.batch;
    let b = crate::broker::footprint::batch_size_to_fit(resolved_workers, avail, floor, &base);
    if b == requested {
        return None; // already fits — no override needed
    }
    eprintln!(
        "admission: recipe '{name}' batch size {requested} → {b} to fit {:.0}G available at {resolved_workers} workers (auto-tuned estimate)",
        snap.mem_avail_gb
    );
    Some(b)
}

/// Run one scheduled cell under a shared cross-cell RAM semaphore so the SUM of
/// concurrently-running cells cannot exceed the declared budget (memory admission for
/// the parallel partition backfill). MIRRORS the `ParallelExecutor`'s per-node
/// memory admission (`executor::run_node`): acquire `footprint_gib` permits
/// (GiB units, matching `NodeEnv::memory`), CLAMPED to the budget so a single
/// cell larger than the whole box runs ALONE instead of deadlocking, hold the
/// permit for the cell's ENTIRE run, and release it on drop AFTER `run`
/// completes so the next queued cell can proceed.
///
/// Defense-in-depth: this bounds the scheduled-cell SUM; each prepared launch
/// still reserves its exact footprint through the same immutable
/// [`crate::broker::tenant_quota::TenantAdmission`] — both stay in force.
///
/// `budget_gib == 0` is treated as "no budget known" (the probe failed): run
/// ungated, exactly as before this slice. A non-zero budget always admits at
/// least 1 permit (`.max(1)`), so a `footprint_gib == 0` cell can't slip a
/// 0-permit no-op past the gate.
pub(super) async fn gated_cell_run<F, T>(
    mem_sem: std::sync::Arc<tokio::sync::Semaphore>,
    footprint_gib: u32,
    budget_gib: u32,
    run: F,
) -> T
where
    F: std::future::Future<Output = T>,
{
    if budget_gib == 0 {
        // No box-fit budget known ⇒ the semaphore is a no-op; the per-cell
        // broker admission inside `launch_compiled_plan` is the sole guard.
        return run.await;
    }
    // Clamp to the budget (the executor's `.min(budget)` trick): a cell whose
    // footprint exceeds the whole box still acquires ALL permits and runs
    // alone, never `> budget` permits (which `acquire_many_owned` could never
    // grant ⇒ permanent hang).
    let want = footprint_gib.min(budget_gib).max(1);
    // Held for the whole `run`, dropped after it returns. `acquire_many_owned`
    // on a never-closed semaphore only errors on closure; the scheduler never
    // closes it, so map the (unreachable) error to running ungated rather than
    // dropping the cell.
    let _permit = mem_sem.acquire_many_owned(want).await.ok();
    run.await
}

/// HPO entry point (v0.20). Samples trials from a search space, runs them as
/// parallel nodes in ONE plan (the fan-out), and — once schedulers land —
/// adaptively early-stops via the control policy. Phase 2 ships `--algo random`
/// (a parallel random search, control=None); other algos error until their
/// slice lands. Mirrors `run_one_recipe`'s job/admission/lock setup so HPO runs
/// are memory-admission-gated + scheduler-arbitrated exactly like a normal recipe run.
/// Layer the broad `KillOnNaN` safety net UNDER an HPO policy. `with_control`
/// REPLACES the executor's default `KillOnNaN`, so wiring an HPO policy raw
/// would drop the payload-wide non-finite kill (HPO policies only watch their
/// objective key; TPE doesn't kill on divergence at all). `[KillOnNaN, hpo]`
/// keeps the safety net active — order is load-bearing (KillOnNaN first
/// short-circuits, so a doomed step never consumes the HPO policy's spawn slot).
pub(super) fn with_nan_safety(
    hpo: std::sync::Arc<dyn crate::framework::control::ControlPolicy>,
) -> std::sync::Arc<dyn crate::framework::control::ControlPolicy> {
    std::sync::Arc::new(crate::framework::control::CompositePolicy::new(vec![
        std::sync::Arc::new(crate::framework::control::KillOnNaN),
        hpo,
    ]))
}

#[cfg(test)]
mod footprint_resolve_tests {
    use super::training_io_live_budget_bytes;

    #[test]
    fn async_profiles_require_local_launcher_and_available_snapshot() {
        let available = crate::broker::ResourceSnapshot {
            mem_total_gb: 64.0,
            mem_avail_gb: 32.0,
            ..Default::default()
        };
        assert!(
            training_io_live_budget_bytes(&available, crate::config::launcher::LaunchTarget::Local)
                .is_some()
        );
        assert!(
            training_io_live_budget_bytes(&available, crate::config::launcher::LaunchTarget::Slurm)
                .is_none(),
            "remote retained-memory/cgroup contract is not proven"
        );
        assert!(
            training_io_live_budget_bytes(
                &crate::broker::ResourceSnapshot::default(),
                crate::config::launcher::LaunchTarget::Local
            )
            .is_none()
        );
        let unknown = crate::broker::ResourceSnapshot {
            mem_total_gb: f64::NAN,
            mem_avail_gb: f64::NAN,
            ..Default::default()
        };
        assert!(
            training_io_live_budget_bytes(&unknown, crate::config::launcher::LaunchTarget::Local)
                .is_none()
        );
    }

    /// RESOLVE-side cost-driver extraction for `lamquant_joint_codec`
    /// DEFAULTS (`tier`/`batch_size` absent) — the over-refuse target the
    /// slice fixes. The tuple here MUST equal the RECORD-side
    /// `train_containment(None, 3, 0)` tuple in the cookbook
    /// (`footprint_key_parity` there anchors on the same literal) or the
    /// calibration never gets hit.
    #[test]
    fn joint_codec_default_drivers() {
        let raw = serde_json::json!({});
        let d = crate::broker::Drivers::from_args_json(&raw);
        assert_eq!(d.workers, 2, "uncalibrated worker cap (robustness default)");
        assert_eq!(d.batch, crate::broker::footprint::DEFAULT_BATCH);
        assert_eq!(d.tier, 3, "joint recipe default tier");
        assert_eq!(d.latent, 0, "no --encoder-width ⇒ default latent");
        assert!(
            !d.warm,
            "raw {{}} has no warm_fb_cache ⇒ cold (defaults applied via the plan, not here)"
        );
        // The exact key the cli RESOLVES under for a RAW (undefaulted) joint
        // run. Production bills the plan's DEFAULTED args (warm_fb_cache=true ⇒
        // `|w`); from_args_json on raw args is the conservative cold `|c`.
        assert_eq!(
            d.key("lamquant_joint_codec").flat(),
            "lamquant_joint_codec|3|32|2|c"
        );
    }

    /// Explicit tier/batch flow through to the key (so a tier-6 fullband
    /// run keys separately from a tier-3 run).
    #[test]
    fn explicit_tier_batch_flow_to_key() {
        let raw = serde_json::json!({ "tier": 6, "batch_size": 16 });
        let d = crate::broker::Drivers::from_args_json(&raw);
        assert_eq!((d.workers, d.batch, d.tier), (2, 16, 6));
        assert_eq!(
            d.key("lamquant_joint_codec").flat(),
            "lamquant_joint_codec|6|16|2|c"
        );
    }

    /// THE parity-bug regression: an explicit `workers:4` must clamp to the
    /// cap (2) on the RESOLVE side, so it keys identically to the RECORD
    /// side (which always launches `UNCALIBRATED_WORKER_CAP`). Before the
    /// fix the cli clamped `1..=4` → keyed under workers=4, a permanent miss.
    #[test]
    fn explicit_workers_clamps_to_cap_for_key_parity() {
        let raw = serde_json::json!({ "workers": 4, "tier": 3, "batch_size": 32 });
        let d = crate::broker::Drivers::from_args_json(&raw);
        assert_eq!(d.workers, crate::broker::UNCALIBRATED_WORKER_CAP);
        assert_eq!(
            d.key("lamquant_joint_codec").flat(),
            "lamquant_joint_codec|3|32|2|c"
        );
    }

    /// Phase 3: the warm flag (off the recipe's DEFAULTED args) flows into the
    /// estimate AND the key — a warm run bills the tighter per-worker term and
    /// keys `|w` so it can't share calibration with a cold `|c` run.
    #[test]
    fn warm_flag_flows_to_estimate_and_key() {
        let warm = crate::broker::Drivers::from_args_json(
            &serde_json::json!({ "warm_fb_cache": true, "tier": 3, "batch_size": 32 }),
        );
        let cold = crate::broker::Drivers::from_args_json(
            &serde_json::json!({ "warm_fb_cache": false, "tier": 3, "batch_size": 32 }),
        );
        assert!(warm.warm && !cold.warm);
        assert!(warm.estimate().ram_bytes < cold.estimate().ram_bytes);
        assert_eq!(
            warm.key("lamquant_joint_codec").flat(),
            "lamquant_joint_codec|3|32|2|w"
        );
        assert_eq!(
            cold.key("lamquant_joint_codec").flat(),
            "lamquant_joint_codec|3|32|2|c"
        );
    }
}

#[cfg(test)]
mod gated_cell_run_tests {
    //! Cross-cell RAM admission: the shared box-fit semaphore must SERIALIZE
    //! concurrent cells whose footprints SUM over the budget, run cells whose
    //! footprints SUM within the budget CONCURRENTLY, and CLAMP a single
    //! over-budget cell to the budget (run alone, never deadlock). Mirrors the
    //! executor's `memory_budget_serializes_when_sum_exceeds_box_fit` style.
    use super::gated_cell_run;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::sync::Semaphore;

    /// A cell body that bumps a shared `live` counter (tracking `peak`
    /// concurrency), holds for a beat, then drops — so the test can assert
    /// whether two gated cells overlapped or serialized.
    async fn busy_cell(peak: Arc<AtomicU32>, live: Arc<AtomicU32>) {
        let now = live.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        live.fetch_sub(1, Ordering::SeqCst);
    }

    /// SUM over budget ⇒ the two cells must NOT both hold permits at once.
    /// Budget 4, each cell wants 3 (sum 6 > 4) ⇒ peak concurrency 1.
    #[tokio::test]
    async fn over_budget_pair_serializes() {
        let sem = Arc::new(Semaphore::new(4));
        let peak = Arc::new(AtomicU32::new(0));
        let live = Arc::new(AtomicU32::new(0));
        let c1 = gated_cell_run(sem.clone(), 3, 4, busy_cell(peak.clone(), live.clone()));
        let c2 = gated_cell_run(sem.clone(), 3, 4, busy_cell(peak.clone(), live.clone()));
        tokio::join!(c1, c2);
        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "two cells wanting 3 GiB each (sum 6 > budget 4) must serialize"
        );
    }

    /// SUM within budget ⇒ the two cells run CONCURRENTLY. Budget 8, each
    /// wants 3 (sum 6 <= 8) ⇒ peak concurrency 2.
    #[tokio::test]
    async fn within_budget_pair_runs_concurrently() {
        let sem = Arc::new(Semaphore::new(8));
        let peak = Arc::new(AtomicU32::new(0));
        let live = Arc::new(AtomicU32::new(0));
        let c1 = gated_cell_run(sem.clone(), 3, 8, busy_cell(peak.clone(), live.clone()));
        let c2 = gated_cell_run(sem.clone(), 3, 8, busy_cell(peak.clone(), live.clone()));
        tokio::join!(c1, c2);
        assert_eq!(
            peak.load(Ordering::SeqCst),
            2,
            "two cells wanting 3 GiB each (sum 6 <= budget 8) must run concurrently"
        );
    }

    /// THE clamp: a cell whose footprint EXCEEDS the whole budget acquires
    /// `budget` permits (runs alone), NOT `> budget` (which `acquire_many_owned`
    /// could never grant ⇒ permanent hang). The cell must still complete, and a
    /// second cell must wait for it (peak concurrency 1).
    #[tokio::test]
    async fn over_box_cell_clamps_and_runs_alone() {
        let sem = Arc::new(Semaphore::new(4));
        let peak = Arc::new(AtomicU32::new(0));
        let live = Arc::new(AtomicU32::new(0));
        // footprint 100 GiB >> budget 4 ⇒ clamps to 4 ⇒ acquires all permits.
        let big = gated_cell_run(sem.clone(), 100, 4, busy_cell(peak.clone(), live.clone()));
        let other = gated_cell_run(sem.clone(), 1, 4, busy_cell(peak.clone(), live.clone()));
        // tokio::join completing at all proves the clamped cell did NOT hang.
        tokio::join!(big, other);
        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "an over-budget cell clamps to the whole budget and runs alone"
        );
    }

    /// A zero footprint still acquires at least 1 permit (`.max(1)`), so a
    /// 0-GiB cell can't slip a no-op past the gate. With budget 1, two 0-GiB
    /// cells therefore serialize (each takes the single permit).
    #[tokio::test]
    async fn zero_footprint_takes_one_permit() {
        let sem = Arc::new(Semaphore::new(1));
        let peak = Arc::new(AtomicU32::new(0));
        let live = Arc::new(AtomicU32::new(0));
        let c1 = gated_cell_run(sem.clone(), 0, 1, busy_cell(peak.clone(), live.clone()));
        let c2 = gated_cell_run(sem.clone(), 0, 1, busy_cell(peak.clone(), live.clone()));
        tokio::join!(c1, c2);
        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "a 0-GiB footprint still takes 1 permit (budget 1 ⇒ serialize)"
        );
    }

    /// `budget == 0` (probe failed) ⇒ ungated: cells run with no admission, so
    /// two overlap freely (the per-cell broker gate is the sole guard). A 0-cap
    /// semaphore would block forever if the budget path acquired from it — this
    /// pins the early-return that skips acquisition entirely.
    #[tokio::test]
    async fn zero_budget_runs_ungated() {
        let sem = Arc::new(Semaphore::new(1)); // tiny; must NOT be acquired
        let peak = Arc::new(AtomicU32::new(0));
        let live = Arc::new(AtomicU32::new(0));
        let c1 = gated_cell_run(sem.clone(), 9, 0, busy_cell(peak.clone(), live.clone()));
        let c2 = gated_cell_run(sem.clone(), 9, 0, busy_cell(peak.clone(), live.clone()));
        tokio::join!(c1, c2);
        assert_eq!(
            peak.load(Ordering::SeqCst),
            2,
            "budget 0 ⇒ semaphore is a no-op, cells run concurrently (per-cell gate aside)"
        );
    }
}
