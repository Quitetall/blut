// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Broker admission math used by launches: recipe footprints, admitted
//! workers/batch, box-fit budget, and the gated cell-run wrapper.
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

/// ADR 0133 floor policy (fail-LOUD): an entirely undeclared plan bills the
/// 2 GiB compatibility floor and SAYS SO on every launch — implement
/// `Stage::resource_envelope` for real admission. `BLUT_STRICT_DECLARATIONS=1`
/// is the operator strict mode: refuse instead of floor.
pub(super) fn footprint_or_floor(
    declared_fp: Option<crate::broker::Footprint>,
    name: &str,
) -> anyhow::Result<crate::broker::Footprint> {
    match declared_fp {
        Some(fp) => Ok(fp),
        None => {
            if std::env::var("BLUT_STRICT_DECLARATIONS").is_ok_and(|v| v == "1") {
                anyhow::bail!(
                    "'{name}' declares no resource envelope and BLUT_STRICT_DECLARATIONS=1 \
                     — implement Stage::resource_envelope (ADR 0133)"
                );
            }
            eprintln!(
                "warning: no stage in '{name}' declares a resource envelope — billing the \
                 2G compatibility floor. Implement Stage::resource_envelope (ADR 0133) for \
                 real admission."
            );
            Ok(crate::broker::Footprint {
                ram_bytes: 2 * 1024 * 1024 * 1024,
                vram_mib: 0,
            })
        }
    }
}

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
    declared: Option<&(String, blut_types::envelope::ResourceEnvelope)>,
    recipe: &str,
    tuned: Option<(u32, Option<u32>)>,
    warm: bool,
) -> Option<crate::broker::Footprint> {
    let (stage_name, env) = declared?;
    // Tuned units re-evaluate the affine cost model (increment 3); the same
    // overrides + the warm context compose the store key, so the calibration
    // row is exactly the one the JSON path reads/writes for this launch shape.
    let mut unit_overrides: Vec<(&str, u32)> = Vec::new();
    let mut key_ctx: Vec<(&str, String)> = Vec::new();
    if let Some((workers, batch)) = tuned {
        // A tuned launch needs the re-evaluable model — without a workers cost
        // term the declared estimate can't reflect the override; bill the
        // DECLARED (conservative-high) estimate instead of guessing.
        if env.cost_terms.iter().all(|t| t.dimension != "workers") {
            return Some(crate::broker::Footprint {
                ram_bytes: env.ram_bytes,
                vram_mib: 0,
            });
        }
        unit_overrides.push(("workers", workers));
        key_ctx.push(("workers", workers.to_string()));
        if let Some(b) = batch {
            unit_overrides.push(("batch", b));
            key_ctx.push(("batch", b.to_string()));
        }
    }
    key_ctx.push(("warm", if warm { "w" } else { "c" }.to_string()));
    // No calibration dimensions ⇒ estimate-only admission (nothing to key the
    // measured-peak store on) — the declared envelope IS the footprint.
    let Some(flat) =
        crate::broker::footprint::envelope_calibration_key_with_context(recipe, env, &key_ctx)
    else {
        return Some(crate::broker::Footprint {
            ram_bytes: crate::broker::footprint::envelope_footprint_at(env, &unit_overrides, warm),
            vram_mib: 0,
        });
    };
    let hint = crate::broker::Footprint {
        ram_bytes: crate::broker::footprint::envelope_footprint_at(env, &unit_overrides, warm),
        vram_mib: 0,
    };
    let resolved = crate::broker::FootprintStore::load().resolve_flat(&flat, hint);
    let _ = stage_name; // identity kept for future diagnostics
    Some(resolved)
}

/// The warm-cache CONTEXT fact (`warm_fb_cache` recipe arg) — a runtime launch
/// condition the CLI threads into `StageContext.fb_warm`, the warm coefficient
/// selection, and the calibration key. This is NOT footprint interpretation of
/// cookbook keys (the coefficients live in the cookbook's declared cost terms);
/// it is the one blessed context read that survives Phase D (ADR 0133).
pub(super) fn warm_context(raw: &serde_json::Value) -> bool {
    raw.get("warm_fb_cache")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// Synchronous/base half of a train-shaped footprint for ADR 0103 profiles.
/// The selected profile separately bills worker process RSS and every retained
/// async queue payload, so this intentionally evaluates the existing model at
/// zero workers to avoid counting those bytes twice. Batch/model/tier/input
/// terms remain in the base because they exist on the inline path too.
pub(super) fn sync_base_footprint(
    raw: &serde_json::Value,
    declared: Option<&(String, blut_types::envelope::ResourceEnvelope)>,
    batch: Option<u32>,
    resolved_workers: u32,
    resolved: crate::broker::Footprint,
) -> crate::broker::Footprint {
    // ADR 0133 incr 3: declared plans compute the sync base from the envelope —
    // sync_base_excluded terms at zero (the engine never names "the workers"),
    // the calibrated `resolved` floor preserved via the same max() shape.
    if let Some((_, env)) = declared
        && !env.cost_terms.is_empty()
    {
        let warm = warm_context(raw);
        let mut overrides: Vec<(&str, u32)> = vec![("workers", resolved_workers)];
        if let Some(b) = batch {
            overrides.push(("batch", b));
        }
        let sync = crate::broker::footprint::envelope_sync_base(env, warm);
        let with_workers = crate::broker::footprint::envelope_footprint_at(env, &overrides, warm);
        let known_worker_term = with_workers.saturating_sub(sync);
        let resolved_minus_worker = resolved.ram_bytes.saturating_sub(known_worker_term);
        return crate::broker::Footprint {
            ram_bytes: sync.max(resolved_minus_worker),
            vram_mib: resolved.vram_mib,
        };
    }
    // No declared cost terms: nothing separates the worker bytes from the
    // whole-footprint estimate, so the sync base IS the resolved footprint
    // (floored at the light base for an arg-less recipe). An async-I/O
    // profile on such a stage would double-bill its worker term — but a
    // profile-declaring stage necessarily declares cost terms (train_joint
    // does), so this branch only serves profile-less stages, where the value
    // is unused beyond containment sizing. (ADR 0133 Phase D: the recipe-JSON
    // formula path is gone.)
    crate::broker::Footprint {
        ram_bytes: resolved.ram_bytes.max(2 * 1024 * 1024 * 1024),
        vram_mib: resolved.vram_mib,
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
    ctx.fb_warm = warm_context(raw);
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

    // Untuned: the declared workers-term units (what the stage launches), else
    // the conservative cap constant — the recipe JSON is never consulted.
    let declared = plan.max_declared_envelope();
    let resolved_workers = admitted_workers.unwrap_or_else(|| {
        declared
            .as_ref()
            .and_then(|(_, e)| {
                e.cost_terms
                    .iter()
                    .find(|t| t.dimension == "workers")
                    .map(|t| t.declared_units)
            })
            .unwrap_or(crate::broker::footprint::UNCALIBRATED_WORKER_CAP)
    });
    let sync_footprint = sync_base_footprint(
        raw,
        declared.as_ref(),
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
    declared: Option<&(String, blut_types::envelope::ResourceEnvelope)>,
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
    // ADR 0133 incr 3: env-first — a declared cost model searches through the
    // typed seam (cookbook enumerates the term + ceiling, engine owns the
    // search). Shadow-compares against the JSON search until Phase D.
    // This snapshot is serialized BLUT-vs-BLUT by the scheduler lock and nets
    // out other processes via MemAvailable. It reduces over-admission risk but
    // cannot guarantee against drift or uncontained processes. The search runs
    // over the DECLARED workers cost term (ADR 0133 Phase D: the recipe-JSON
    // formula is gone) — no term declared ⇒ nothing to tune, keep the launch
    // count as declared (`None`, exactly the pre-auto-tune behavior).
    let (_, env) = declared?;
    let target = cpu
        .saturating_sub(2)
        .clamp(1, crate::broker::footprint::MAX_AUTO_WORKERS);
    let w = crate::broker::footprint::fit_and_saturate_env(
        env,
        "workers",
        target,
        avail,
        floor,
        warm_context(raw),
    )?;
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
    declared: Option<&(String, blut_types::envelope::ResourceEnvelope)>,
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
    // Batch shrink over the DECLARED batch cost term at the held workers
    // (residual budget; ADR 0133 Phase D — the recipe-JSON formula is gone).
    // No term declared ⇒ nothing to tune (`None`, requested batch unchanged).
    let (_, env) = declared?;
    let term = env.cost_terms.iter().find(|t| t.dimension == "batch")?;
    let requested = term.declared_units.max(1);
    let b = crate::broker::footprint::shrink_to_fit_env(
        env,
        "batch",
        requested,
        &[("workers", resolved_workers)],
        avail,
        floor,
        warm_context(raw),
    )?;
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
