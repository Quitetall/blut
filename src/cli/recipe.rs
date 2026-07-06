// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut recipe` — catalog list/show, declare, run, sweep + plan launch.
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

use super::*;

#[derive(Subcommand, Debug)]
pub(super) enum RecipeCommand {
    /// List the recipe catalog.
    List {
        /// Emit the catalog as JSON (for scripts/agents).
        #[arg(long)]
        json: bool,
    },
    /// Print one recipe's args JSON schema.
    Show {
        /// Recipe name (as listed by `recipe list`).
        name: String,
    },
    /// DECLARATIVE recipes (G/C3): compile a `.toml` recipe (a named chain
    /// of stages-by-name + args) into a runtime-kind-checked plan and render
    /// its DAG. With NO file, lists the declarative recipes discovered under
    /// `~/.config/blut/recipes/*.toml` ($BLUT_USER_RECIPES_DIR). Resolves
    /// stages from the registered cookbooks' `stages_erased()` registries.
    Declare {
        /// Path to a `.toml` recipe (omit to list discovered recipes).
        file: Option<std::path::PathBuf>,
        /// LAUNCH the `.toml` recipe (C3): after it compiles + kind-checks,
        /// execute it end-to-end through the SAME admission-gated, cgroup-
        /// contained, cache-honouring path as `recipe run`. Without `--run`
        /// (default) the DAG is only rendered — nothing executes. Requires a
        /// `<file>`.
        #[arg(long, default_value_t = false)]
        run: bool,
        /// Promote this run's outputs to the global cache (only with `--run`).
        #[arg(long, default_value_t = false)]
        shared_cache: bool,
        /// Force-recompute on launch: bypass the stage cache READ so every stage
        /// runs even with a warm entry (only with `--run`). Alias: `--force`.
        #[arg(long = "no-cache", alias = "force", default_value_t = false)]
        no_cache: bool,
    },
    /// Execute a recipe, or a config-driven sweep over it.
    Run {
        /// Recipe name.
        name: String,
        /// Args as inline JSON. Ignored in config mode (--config-dir).
        #[arg(long, default_value = "{}")]
        args: String,
        /// Promote this run's outputs to the global cache for
        /// future re-use. Default: per-job cache only.
        #[arg(long, default_value_t = false)]
        shared_cache: bool,
        /// Force-recompute (S4): BYPASS the stage cache READ so every stage runs
        /// even when a warm cached entry exists. The fresh result is STILL
        /// written to the cache, so later runs hit again — this is the "force
        /// recompute" A/B semantic, NOT a cache wipe. Alias: `--force`.
        #[arg(long = "no-cache", alias = "force", default_value_t = false)]
        no_cache: bool,
        /// Hydra-style config dir (enables config mode). The composed config's
        /// top-level keys must match the recipe's flat Args fields.
        #[arg(long)]
        config_dir: Option<String>,
        /// Config name within --config-dir (required in config mode).
        #[arg(long)]
        config_name: Option<String>,
        /// Top-level config key whose subtree is the recipe's Args (default:
        /// the recipe name). Nest Args under this key so `--set`/`--sweep` can
        /// target them with dotted paths, e.g. `<key>.epochs=2`.
        #[arg(long)]
        config_key: Option<String>,
        /// Base override(s) applied to the composed config, e.g.
        /// `--set lr=1e-3 --set epochs=5` (repeatable).
        #[arg(long = "set", value_name = "KEY=VAL")]
        set: Vec<String>,
        /// Sweep axis/axes, e.g. `--sweep "lr=1e-3,1e-4" --sweep "bs=8,16"`
        /// → cartesian product (repeatable). Requires --config-dir/--config-name.
        #[arg(long, value_name = "KEY=V1,V2")]
        sweep: Vec<String>,
        /// Print the expanded combos (fingerprint + skip status) without
        /// running anything.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        /// #3 distributed placement: `local` (default) runs broker-gated +
        /// cgroup-contained on THIS box (the never-OOM path); `slurm`/`ray`
        /// submit each train stage to a cluster via the configured launcher
        /// (`BLUT_SLURM_*` / `RAY_ADDRESS` env). SHARED-FS CONTRACT: the
        /// content-addressed cache + job dirs must be reachable from the
        /// compute node (NFS/Lustre); local admission still gates (conservative).
        #[arg(long, default_value = "local")]
        launcher: String,
    },
}

pub(super) async fn run_recipe(reg: &crate::framework::Registry, cmd: RecipeCommand) -> Result<()> {
    // The recipe catalog comes from the caller-supplied cookbook registry.
    let find_recipe = |name: &str| reg.find(name);
    match cmd {
        RecipeCommand::List { json } => {
            // Sort by (category label, name) so the catalog reads
            // top-down like the BLUT Training Cockpit menu (DATA →
            // TRAINING → EVAL → EXPORT → PIPELINE → USER).
            let mut sorted: Vec<&'static crate::recipes::recipe::RecipeDef> = reg.all().collect();
            sorted.sort_by(|a, b| {
                a.category
                    .label()
                    .cmp(b.category.label())
                    .then_with(|| a.name.cmp(b.name))
            });
            if json {
                let arr: Vec<_> = sorted
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "name": r.name,
                            "category": r.category.label(),
                            "backend": r.backend_id,
                            "input_kinds": r.input_kinds,
                            "output_kind": r.output_kind,
                            "description": r.description,
                        })
                    })
                    .collect();
                emit_json(&arr)?;
                return Ok(());
            }
            println!(
                "{:<32} {:<14} {:<12} {:<24} → output",
                "name", "category", "backend", "inputs"
            );
            for r in sorted {
                let inputs = if r.input_kinds.is_empty() {
                    "(graph-input)".to_string()
                } else {
                    r.input_kinds.join(",")
                };
                println!(
                    "{:<32} {:<14} {:<12} {:<24} → {}",
                    r.name,
                    r.category.label(),
                    r.backend_id,
                    inputs,
                    r.output_kind,
                );
            }
        }
        RecipeCommand::Show { name } => {
            let r = find_recipe(&name).ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
            println!("name        : {}", r.name);
            println!("category    : {}", r.category.label());
            println!("backend     : {}", r.backend_id);
            println!(
                "input kinds : {}",
                if r.input_kinds.is_empty() {
                    "(graph-input)".into()
                } else {
                    r.input_kinds.join(", ")
                }
            );
            println!("output kind : {}", r.output_kind);
            println!("description : {}", r.description);
            let schema = (r.args_schema_fn)();
            println!(
                "args schema :\n{}",
                serde_json::to_string_pretty(&schema)
                    .unwrap_or_else(|e| format!("(serialize error: {e})"))
            );
        }
        RecipeCommand::Declare {
            file,
            run,
            shared_cache,
            no_cache,
        } => {
            use crate::recipes::declarative::{
                DeclarativeRecipe, scan_user_recipes, user_recipes_dir,
            };
            match file {
                None => {
                    if run {
                        return Err(anyhow!(
                            "--run requires a <file> (a .toml recipe to launch)"
                        ));
                    }
                    // F4 discovery: list ~/.config/blut/recipes/*.toml.
                    let found = scan_user_recipes();
                    let dir = user_recipes_dir()
                        .map(|d| d.display().to_string())
                        .unwrap_or_else(|| "(no config dir)".into());
                    if found.is_empty() {
                        println!("no declarative recipes under {dir}");
                    } else {
                        println!("declarative recipes under {dir} ({}):", found.len());
                        for (rname, path) in found {
                            println!("  {rname:<24} {}", path.display());
                        }
                    }
                }
                Some(path) => {
                    // Compile + kind-check the .toml against the cookbook's
                    // stages_erased registry.
                    let recipe = DeclarativeRecipe::load(&path).map_err(|e| anyhow!("{e}"))?;
                    let n = recipe.stages.len();
                    let plan = recipe.compile(reg).map_err(|e| anyhow!("{e}"))?;
                    if run {
                        // C3 LAUNCH: execute the compiled plan through the same
                        // admission-gated / cgroup-contained / cache-honouring
                        // core as `recipe run`. No RecipeMarker (declarative
                        // recipes don't resume by registry name); `Local`
                        // placement (clusters target registry recipes only).
                        println!(
                            "✓ '{}' compiles + kind-checks ({n} ingredient(s)); launching…",
                            recipe.name
                        );
                        launch_compiled_plan(
                            &recipe.name,
                            plan,
                            None,
                            None,
                            shared_cache,
                            crate::config::launcher::LaunchTarget::Local,
                            None,
                            no_cache,
                        )
                        .await?;
                    } else {
                        // Render-only (default): print the runnable DAG, no exec.
                        print!("{}", plan.render_ascii().map_err(|e| anyhow!("{e}"))?);
                        println!(
                            "✓ '{}' compiles + kind-checks ({n} ingredient(s)).",
                            recipe.name
                        );
                    }
                }
            }
        }
        RecipeCommand::Run {
            name,
            args,
            shared_cache,
            no_cache,
            config_dir,
            config_name,
            config_key,
            set,
            sweep,
            dry_run,
            launcher,
        } => {
            // #3 distributed: parse placement up front so a typo fails the run
            // BEFORE any job dir / state is written (vs deep in the executor).
            let launch_target: crate::config::launcher::LaunchTarget = launcher
                .parse()
                .map_err(|e| anyhow!("invalid --launcher {launcher:?}: {e}"))?;
            // Any of these put us in config mode — so a stray --set / --config-key
            // can't be silently dropped (run_recipe_sweep then errors cleanly if
            // --config-dir/--config-name are missing).
            let config_mode = config_dir.is_some()
                || config_name.is_some()
                || config_key.is_some()
                || !set.is_empty()
                || !sweep.is_empty();
            if config_mode {
                if args != "{}" {
                    eprintln!(
                        "warning: --args is ignored in config mode (args come from the config)"
                    );
                }
                run_recipe_sweep(
                    reg,
                    &name,
                    config_dir,
                    config_name,
                    config_key,
                    &set,
                    &sweep,
                    dry_run,
                    shared_cache,
                    launch_target,
                    no_cache,
                )
                .await?;
            } else {
                let raw: serde_json::Value = serde_json::from_str(&args)
                    .with_context(|| format!("parse --args as JSON: {args}"))?;
                if dry_run {
                    // `--dry-run` is documented as "without running anything". The
                    // config-mode sweep path honors that (run_recipe_sweep), but the
                    // single-invocation `--args` path previously fell straight into
                    // run_one_recipe — which compiled the plan AND executed every
                    // stage (spawning the warm systemd-run unit + acquiring the
                    // exclusive GPU lock) before any value was produced. Short-circuit
                    // here: VALIDATE the args + confirm the plan COMPILES (so a
                    // dry-run can't report "OK" on invalid args — B/P5), then report
                    // the resolved RAM footprint and return WITHOUT executing or
                    // touching any resource (compile builds the plan; it never runs).
                    let def = reg
                        .find(&name)
                        .ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
                    if let Err(e) = (def.compile_fn)(raw.clone()) {
                        return Err(anyhow!("{e}")); // RecipeError already names the cause
                    }
                    let fp = recipe_footprint(&name, &raw);
                    let gib = fp.ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                    println!(
                        "[dry-run] recipe={name} resolved RAM footprint ≈ {gib:.1}G \
                         (admission would gate this against free RAM + the 6G floor). \
                         No ingredients executed; no GPU/cgroup acquired."
                    );
                    return Ok(());
                }
                run_one_recipe(
                    reg,
                    &name,
                    raw,
                    None,
                    shared_cache,
                    launch_target,
                    None,
                    no_cache,
                )
                .await?;
            }
        }
    }
    Ok(())
}

/// Run ONE recipe invocation end-to-end: compile → job dir → admission gate →
/// scheduler lock → execute → Done/Failed. Extracted from the `recipe run`
/// handler so the sweep runner can call it per combo. `sweep_fp` ties a combo
/// to the sweep-completion index: on success it records the final output so a
/// re-run can skip this combo (best-effort — recording never fails the run).
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_one_recipe(
    reg: &crate::framework::Registry,
    name: &str,
    args: serde_json::Value,
    sweep_fp: Option<crate::framework::ContentHash>,
    shared_cache: bool,
    launch_target: crate::config::launcher::LaunchTarget,
    // Phase-G scheduler: pin this run to a GPU device. `Some(i)` takes the
    // PER-DEVICE scheduler lock (so cells on distinct GPUs run concurrently)
    // and exports CUDA_VISIBLE_DEVICES; `None` = box default + box-wide lock.
    device_index: Option<usize>,
    // INC D (S4): force-recompute. `true` bypasses the stage cache READ so every
    // stage runs even with a warm entry (the fresh result is still cached).
    no_cache: bool,
) -> Result<String> {
    let r = reg
        .find(name)
        .ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
    let plan = (r.compile_fn)(args.clone()).map_err(|e| anyhow!("recipe compile failed: {e}"))?;
    // A registry recipe CAN resume by name+args (the RecipeMarker is the resume
    // oracle for `blut plan resume`). Declarative `.toml` launches pass `None`
    // (no registry recipe to re-compile from) — see `launch_compiled_plan`.
    launch_compiled_plan(
        name,
        plan,
        Some(RecipeMarker {
            name: name.to_string(),
            args,
        }),
        sweep_fp,
        shared_cache,
        launch_target,
        device_index,
        no_cache,
    )
    .await
}

/// Launch an ALREADY-COMPILED plan end-to-end: footprint → job dir → ExecCtx →
/// admission gate → scheduler lock → execute → Done/Failed → lineage index.
/// The shared launch core behind both `run_one_recipe` (a registry recipe,
/// compiled via its `compile_fn`) and the declarative `.toml` launch path
/// (`recipe declare --run`, compiled via `DeclarativeRecipe::compile`). Both
/// paths get IDENTICAL admission/containment/cache treatment — the only
/// difference is `marker`: `Some` for a registry recipe (resumable by
/// name+args), `None` for a declarative launch (no registry recipe to resume
/// from, so no marker is written).
#[allow(clippy::too_many_arguments)]
pub(super) async fn launch_compiled_plan(
    name: &str,
    plan: crate::framework::plan::CompiledPlan,
    marker: Option<RecipeMarker>,
    sweep_fp: Option<crate::framework::ContentHash>,
    shared_cache: bool,
    launch_target: crate::config::launcher::LaunchTarget,
    device_index: Option<usize>,
    no_cache: bool,
) -> Result<String> {
    use crate::framework::ExecCtx;

    // ADR 0046 slice-1: resolve the RAM footprint from the recipe's DEFAULTED
    // args (the plan re-serialized them with serde defaults applied) — NOT raw
    // user args — so a defaulted driver like `warm_fb_cache` (Phase 3) and
    // tier/batch are read IDENTICALLY to what the train stage records under
    // (RECORD side), keeping the RESOLVE/RECORD calibration key in parity even
    // when the user omitted the field.
    // ADR 0071 A2: auto-tune decode workers to fit-AND-saturate (one knob fixes
    // both the over-refuse and the GPU-starvation). Compute W from a SINGLE mem
    // snapshot; the gate's footprint uses W, and W is cached on the ExecCtx below
    // so the cookbook train stage (RECORD) launches exactly this count.
    // E2: auto-tune batch size against the SAME snapshot, against the residual
    // budget after W is fixed (extends the same knob to a second driver).
    let admitted_workers = admitted_workers_for(name, plan.exec_view().recipe_args);
    let admitted_batch_size = admitted_workers
        .and_then(|w| admitted_batch_size_for(name, plan.exec_view().recipe_args, w));
    let footprint = match admitted_workers {
        Some(w) => {
            recipe_footprint_tuned(name, plan.exec_view().recipe_args, w, admitted_batch_size)
        }
        None => recipe_footprint(name, plan.exec_view().recipe_args),
    };

    let job_id = crate::jobs::new_job_id();
    let job_dir = crate::paths::job_dir(&job_id)?;
    let mut ctx = ExecCtx::new(job_dir.clone());
    // Phase 5: size the executor's memory admission to box-fit (MemTotal −
    // floor) so the parallel executor can't stack concurrent stages past the
    // box. Sequential runs one stage at a time, so this is a no-op there.
    {
        let snap = crate::broker::ResourceSnapshot::probe();
        if snap.mem_total_gb > 0.0 {
            let box_fit =
                (snap.mem_total_gb - crate::broker::admission::DEFAULT_FLOOR_GIB).max(1.0) as u32;
            ctx = ctx.with_memory_budget(box_fit);
        }
    }
    // #3 distributed: thread placement into the ExecCtx → every StageContext
    // built by the executor carries it → a lamquant train stage routes to the
    // cluster. `Local` (default) is a no-op vs the pre-launcher behaviour.
    ctx = ctx.with_launch_target(launch_target);
    // Phase-G scheduler: pin this run to a GPU device so the cookbook backend
    // exports CUDA_VISIBLE_DEVICES for its trainer.
    ctx = ctx.with_device_index(device_index);
    // Single-job multi-GPU: size the GPU semaphore pool to the box's device
    // count so a DDP stage can acquire `nproc` permits (and a single-GPU cell
    // can't co-schedule onto a device the DDP job owns). On a 1-GPU box this is
    // 1 → byte-identical to before. `capacity()` probes CUDA_VISIBLE_DEVICES /
    // nvidia-smi for the local launcher; Slurm reports its --gpus allocation.
    let gpu_pool = crate::config::launcher::launcher_for(launch_target).capacity();
    ctx = ctx.with_resource_limit(crate::framework::Resource::Gpu, gpu_pool.max(1));
    // Phase 3: thread the warm flag from the recipe's DEFAULTED args (the SAME
    // source `recipe_footprint` reads above) into every StageContext, so a
    // train stage's footprint RECORD keys identically to the admission RESOLVE.
    // Carried on the context (not a stage Arg) so warm never enters the
    // checkpoint cache key — a warm and a cold run share the trained output.
    let fb_warm = crate::broker::Drivers::from_args_json(plan.exec_view().recipe_args).warm;
    ctx = ctx.with_fb_warm(fb_warm);
    // A2: cache the auto-tuned worker count on the ctx so the cookbook train stage
    // launches exactly what admission sized (RESOLVE↔RECORD parity, never-OOM).
    if let Some(w) = admitted_workers {
        ctx = ctx.with_admitted_workers(w);
    }
    // E2: cache the auto-tuned batch size on the ctx so the cookbook train stage
    // launches exactly what admission sized (RESOLVE↔RECORD parity, never-OOM).
    if let Some(b) = admitted_batch_size {
        ctx = ctx.with_admitted_batch_size(b);
    }
    // INC D (S4): `--no-cache`/`--force` bypasses the stage cache READ so every
    // stage recomputes; the fresh result is still written to the cache.
    ctx = ctx.with_bypass_cache(no_cache);
    if shared_cache {
        if let Some(global) = crate::framework::CacheHandle::default_global_path() {
            std::fs::create_dir_all(&global)
                .with_context(|| format!("create global cache dir {}", global.display()))?;
            let cache_handle = (*ctx.cache).clone().with_global(global);
            ctx.cache = std::sync::Arc::new(cache_handle);
        }
    }
    // Mark recipe for plan resume — only for a registry recipe (a declarative
    // `.toml` launch passes `None`: there is no registry recipe to re-compile
    // from on resume, so writing a marker would be a dangling resume oracle).
    if let Some(m) = marker {
        m.write_to(&job_dir)?;
    }

    crate::jobs::write_state(&job_id, JobState::Running)
        .with_context(|| format!("write Running state for {job_id}"))?;

    // KILL-2/KILL-3: bind this job so backend spawns mirror the python child's
    // PROCESS GROUP id into the job pid file (not blut's own pid). A separate
    // `blut cancel <id>` reads that pgid and killpg's the whole tree.
    crate::python_kill::bind_current_job(job_id.clone());

    // KILL-3: trap SIGTERM/ctrl-c. On signal, cancel the executor token AND
    // killpg the live child group, then let the function return so `lock`
    // Drops (RAII unlocks the scheduler — fixes the stale-lock-on-SIGTERM case).
    install_cancel_handler(ctx.cancel.clone());

    // ADR 0046 slice-1 (item 4): RAM-refuse admission gate, BEFORE the lock.
    // The lock already serializes blut-vs-blut GPU jobs (fail-fast), so this is
    // a pure single-job over-subscription guard — if the conservative-high
    // footprint can't fit free RAM, refuse CLEANLY: no launch, no transient
    // unit, no OOM. Best-effort: args with no cost drivers fall back to the
    // conservative default footprint, which still gates oversubscription.
    if let Err(reason) = crate::broker::gate(&format!("recipe '{name}'"), &footprint) {
        crate::python_kill::unbind_current_job();
        if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
            tracing::warn!("write Failed state for {job_id}: {se}");
        }
        return Err(anyhow!("{reason}"));
    }

    // Cross-process GPU arbitration — recipes that don't hit GPU still pay the
    // (cheap) lock cost. Phase-G: a device-pinned run takes its PER-DEVICE
    // lock, so cells on distinct GPUs run concurrently; an unpinned run keeps
    // the box-wide lock (one GPU job at a time).
    let holder = format!("blut-recipe:{job_id}");
    let lock = match device_index {
        Some(dev) => scheduler_lock::acquire_exclusive_device(dev, holder, LockKind::Training),
        None => scheduler_lock::acquire_exclusive(holder, LockKind::Training),
    };
    let lock = match lock {
        Ok(l) => l,
        Err(e) => {
            crate::python_kill::unbind_current_job();
            if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
                tracing::warn!("write Failed state for {job_id}: {se}");
            }
            return Err(anyhow!("acquire_exclusive: {e}"));
        }
    };

    eprintln!("recipe {name}");
    eprintln!("job    {job_id}");
    eprintln!("dir    {}", job_dir.display());
    eprintln!("lock   {}", lock.path().display());

    persist_plan_graph(&plan, &job_dir);
    let result = crate::framework::execute_plan(plan, ctx).await;
    drop(lock);
    crate::python_kill::unbind_current_job();
    match result {
        Ok(r) => {
            crate::jobs::write_state(&job_id, JobState::Done)
                .with_context(|| format!("write Done state for {job_id}"))?;
            eprintln!(
                "done — {} ingredients, {} cache hits, {} misses, elapsed {:?}",
                r.n_stages, r.n_cache_hits, r.n_cache_misses, r.elapsed
            );
            // ADR 0071: advisory stages (e.g. a dry-run/verdict gate) failing do
            // NOT fail the run — surface them as warnings so a good experiment is
            // never mis-read as a failure, and point at the preserved output.
            if !r.warnings.is_empty() {
                eprintln!(
                    "⚠ training OK — completed with {} advisory warning(s) (the run did NOT fail):",
                    r.warnings.len()
                );
                for w in &r.warnings {
                    eprintln!("    · {} (advisory, skipped): {}", w.stage, w.reason);
                }
                eprintln!(
                    "  the trained output + metrics are preserved — `blut results {job_id}` / `blut lineage show {job_id}`."
                );
            }
            if let Some(fp) = sweep_fp {
                record_sweep_completion(fp, &job_id);
            }
            // LineageDB index (fail-soft — the sidecars/status.jsonl are
            // canonical, the DB is a rebuildable index; a failure must not fail
            // a successful run).
            if let Err(e) = crate::lineage_db::ingest_job(&job_id, name, "done") {
                tracing::warn!("lineage index {job_id}: {e}");
            }
            Ok(job_id)
        }
        Err(e) => {
            if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
                tracing::warn!("write Failed state for {job_id}: {se}");
            }
            // Index the failure too (OOM/cache history) — best-effort.
            if let Err(ie) = crate::lineage_db::ingest_job(&job_id, name, "failed") {
                tracing::debug!("lineage index (failed) {job_id}: {ie}");
            }
            // See the `plan execution failed` site in `run_plan_cmd` — same
            // chain-preservation rationale (ADR 0072 A2).
            Err(anyhow::Error::from(e).context("plan execution failed"))
        }
    }
}

/// Best-effort: record a finished sweep combo into the global sweep-completion
/// index (fingerprint → final-stage output hash + sidecar), so a later sweep
/// re-run skips it. The final stage is the last `output.metadata.json` sidecar
/// (lineage scans stage dirs in order). A failure here must NOT fail the run —
/// the index is a skip optimization, never a correctness gate.
pub(super) fn record_sweep_completion(fp: crate::framework::ContentHash, job_id: &str) {
    let recs = match crate::framework::lineage::scan_artifacts(job_id) {
        Ok(recs) => recs,
        Err(e) => {
            tracing::warn!("sweep completion {job_id}: scan artifacts: {e}");
            return;
        }
    };
    // Pick the TERMINAL stage by numeric node-idx. scan_artifacts sorts
    // sidecar paths LEXICALLY, so `.last()` would pick stage "9" over "10" for
    // a ≥10-stage plan — anchor liveness on the real final stage instead.
    let Some(rec) = recs.iter().max_by_key(|r| stage_idx_of(&r.sidecar_path)) else {
        tracing::warn!("sweep completion {job_id}: no artifacts to anchor liveness");
        return;
    };
    if let Err(e) = crate::config::sweep_index::record_completion(
        fp,
        job_id,
        rec.meta.content_hash,
        rec.sidecar_path.clone(),
    ) {
        tracing::warn!("sweep completion {job_id}: record: {e}");
    }
}

/// Numeric stage index from a sidecar path whose parent dir is
/// `<idx>-<stage_name>`. Returns 0 if unparseable (so a malformed dir never
/// wins the terminal-stage `max_by_key`).
pub(super) fn stage_idx_of(sidecar: &std::path::Path) -> u32 {
    sidecar
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .and_then(|n| n.split('-').next())
        .and_then(|d| d.parse::<u32>().ok())
        .unwrap_or(0)
}

/// Config-driven recipe run: compose a base config from `--config-dir` /
/// `--config-name` + `--set` overrides, cartesian-expand `--sweep` axes into
/// combos, then run each through [`run_one_recipe`] (admission-gated, scheduler-
/// lock serialized). Combos already complete in the sweep-index are skipped;
/// a failed combo is tallied and reported, never aborting the rest.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_recipe_sweep(
    reg: &crate::framework::Registry,
    name: &str,
    config_dir: Option<String>,
    config_name: Option<String>,
    config_key: Option<String>,
    set: &[String],
    sweep: &[String],
    dry_run: bool,
    shared_cache: bool,
    launch_target: crate::config::launcher::LaunchTarget,
    // INC D (S4): force-recompute — threaded into every combo's run_one_recipe.
    no_cache: bool,
) -> Result<()> {
    // Fail on a bad recipe name before composing anything.
    if reg.find(name).is_none() {
        return Err(anyhow!("recipe '{name}' not in catalog"));
    }
    let dir = config_dir.ok_or_else(|| anyhow!("--config-dir is required in config/sweep mode"))?;
    let cfg_name =
        config_name.ok_or_else(|| anyhow!("--config-name is required in config/sweep mode"))?;
    // Args subtree key (default = recipe name). Overrides/sweeps must be dotted
    // paths INTO this subtree; dot-less keys are consumed by the compose layer
    // as defaults-list group selections and silently never reach a config value.
    let key = config_key.unwrap_or_else(|| name.to_string());
    warn_dotless_overrides(set, "--set", &key);
    warn_dotless_overrides(sweep, "--sweep", &key);

    let entries = crate::config::expand_and_fingerprint(&dir, &cfg_name, set, sweep)
        .map_err(|e| anyhow!("config compose/expand: {e}"))?;
    if entries.is_empty() {
        return Err(anyhow!("sweep expanded to 0 combos"));
    }
    let total = entries.len();
    eprintln!("sweep: {total} combo(s) for recipe '{name}'");

    if dry_run {
        eprintln!("args subtree key: '{key}'");
        for (i, e) in entries.iter().enumerate() {
            eprintln!(
                "[{i}] fp={} skip={} overrides={:?}",
                e.fingerprint.to_hex(),
                e.cache_skip,
                e.overrides,
            );
        }
        return Ok(());
    }

    let (mut ran, mut skipped, mut failed) = (0usize, 0usize, 0usize);
    for (i, entry) in entries.into_iter().enumerate() {
        let fp = entry.fingerprint;
        if entry.cache_skip {
            eprintln!(
                "[{}/{total}] skip — already complete (fp={})",
                i + 1,
                fp.to_hex()
            );
            skipped += 1;
            continue;
        }
        eprintln!("[{}/{total}] run (fp={})", i + 1, fp.to_hex());
        let args = project_args(entry.config.json, &key);
        match run_one_recipe(
            reg,
            name,
            args,
            Some(fp),
            shared_cache,
            launch_target,
            None,
            no_cache,
        )
        .await
        {
            Ok(_job_id) => ran += 1,
            Err(e) => {
                eprintln!("[{}/{total}] FAILED: {e}", i + 1);
                failed += 1;
            }
        }
    }
    eprintln!("sweep done — ran {ran}, skipped {skipped}, failed {failed}");
    if failed > 0 {
        return Err(anyhow!("{failed}/{total} sweep combo(s) failed"));
    }
    Ok(())
}

/// Project the recipe's flat Args out of a composed config: when the config
/// nests them under `key` (the recipe name by default) as an object, return
/// that subtree — so `--set`/`--sweep` dotted paths `<key>.field=v` reach the
/// Args. Otherwise (a flat config with no such subtree) return the whole config
/// as-is (it feeds the Args directly, but top-level overrides can't apply — a
/// compose-grammar limitation; `warn_dotless_overrides` surfaces it).
pub(super) fn project_args(mut config: serde_json::Value, key: &str) -> serde_json::Value {
    if let serde_json::Value::Object(map) = &mut config {
        if let Some(sub) = map.get_mut(key) {
            if sub.is_object() {
                return sub.take();
            }
        }
    }
    config
}

/// Warn about `key=val` overrides whose key has no `.` — the compose layer
/// treats those as defaults-list group selections, NOT config-value overrides,
/// so they silently don't change a value (and the sweep would collapse to
/// identical fingerprints). `subtree_key` is the Args subtree to target.
pub(super) fn warn_dotless_overrides(items: &[String], flag: &str, subtree_key: &str) {
    for it in items {
        let key = it.split_once('=').map_or(it.as_str(), |(k, _)| k);
        if !key.contains('.') {
            eprintln!(
                "warning: {flag} '{it}' key is dot-less — it is treated as a \
                 defaults-list group selection, not a value override; nest Args under \
                 '{subtree_key}:' and use a dotted path (e.g. '{subtree_key}.{key}=…')."
            );
        }
    }
}

#[cfg(test)]
mod sweep_projection_tests {
    use super::project_args;
    use serde_json::json;

    #[test]
    fn projects_named_subtree() {
        // Args nested under the recipe name → that subtree is the Args.
        let cfg = json!({"lamquant_snn": {"epochs": 2, "preset": "fast"}, "other": 9});
        let args = project_args(cfg, "lamquant_snn");
        assert_eq!(args, json!({"epochs": 2, "preset": "fast"}));
    }

    #[test]
    fn flat_config_passes_through() {
        // No subtree under the key → whole config feeds Args verbatim.
        let cfg = json!({"epochs": 1, "labels_dir": "/x"});
        let args = project_args(cfg.clone(), "lamquant_snn");
        assert_eq!(args, cfg);
    }

    #[test]
    fn non_object_subtree_is_not_projected() {
        // A scalar under the key is not a subtree → fall back to whole config.
        let cfg = json!({"lamquant_snn": 5, "epochs": 1});
        let args = project_args(cfg.clone(), "lamquant_snn");
        assert_eq!(args, cfg);
    }
}

#[cfg(test)]
mod recipe_run_flag_tests {
    //! INC D (S4): the `--no-cache` / `--force` flag on `recipe run` parses,
    //! defaults to false (so the no-flag path is byte-identical to before), and
    //! `--force` is an accepted alias.
    use super::{Cli, Command, RecipeCommand};
    use clap::Parser;

    fn no_cache_of(argv: &[&str]) -> bool {
        match Cli::try_parse_from(argv).expect("parse").command {
            Some(Command::Recipe {
                cmd: RecipeCommand::Run { no_cache, .. },
            }) => no_cache,
            other => panic!("expected recipe run, got {other:?}"),
        }
    }

    #[test]
    fn no_cache_defaults_false() {
        assert!(!no_cache_of(&["blut", "recipe", "run", "demo"]));
    }

    #[test]
    fn no_cache_flag_sets_true() {
        assert!(no_cache_of(&[
            "blut",
            "recipe",
            "run",
            "demo",
            "--no-cache"
        ]));
    }

    #[test]
    fn force_alias_sets_true() {
        assert!(no_cache_of(&["blut", "recipe", "run", "demo", "--force"]));
    }
}

#[cfg(test)]
mod recipe_declare_flag_tests {
    //! INC G (C3): `recipe declare --run` parses (the launch flag), defaults to
    //! render-only (`run=false`), and carries `--shared-cache` / `--no-cache`.
    use super::{Cli, Command, RecipeCommand};
    use clap::Parser;

    fn declare_of(argv: &[&str]) -> (Option<std::path::PathBuf>, bool, bool, bool) {
        match Cli::try_parse_from(argv).expect("parse").command {
            Some(Command::Recipe {
                cmd:
                    RecipeCommand::Declare {
                        file,
                        run,
                        shared_cache,
                        no_cache,
                    },
            }) => (file, run, shared_cache, no_cache),
            other => panic!("expected recipe declare, got {other:?}"),
        }
    }

    #[test]
    fn declare_defaults_to_render_only() {
        let (file, run, sc, nc) = declare_of(&["blut", "recipe", "declare", "r.toml"]);
        assert!(file.is_some());
        assert!(!run, "no --run ⇒ render only (no execution)");
        assert!(!sc);
        assert!(!nc);
    }

    #[test]
    fn declare_run_flag_launches() {
        let (_f, run, _sc, _nc) = declare_of(&["blut", "recipe", "declare", "r.toml", "--run"]);
        assert!(run);
    }

    #[test]
    fn declare_run_carries_cache_flags() {
        let (_f, run, sc, nc) = declare_of(&[
            "blut",
            "recipe",
            "declare",
            "r.toml",
            "--run",
            "--shared-cache",
            "--force",
        ]);
        assert!(run && sc && nc, "--force aliases --no-cache");
    }
}
