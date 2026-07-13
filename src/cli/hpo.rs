// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut hpo` — run/show/best over the HPO manifest.
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

use super::*;

#[derive(Subcommand, Debug)]
// `Run` carries the full launch config (many flags) while `Show`/`Best` are
// tiny — a one-shot parse, so the size spread is harmless (boxing would only
// fight clap's derive).
#[allow(clippy::large_enum_variant)]
pub(super) enum HpoCommand {
    /// Run hyperparameter optimization over a recipe: sample trials from a
    /// search space, run them as parallel nodes in one plan, adaptively
    /// early-stop the underperformers (v0.20).
    Run {
        /// Recipe name (the trial's base; the search space overlays its args).
        name: String,
        /// Base args as inline JSON (the fixed part; search dims overlay it).
        #[arg(long, default_value = "{}")]
        args: String,
        /// Search-space YAML file (`dims:` map of dotted-arg-path → distribution).
        #[arg(long)]
        space: Option<String>,
        /// Inline search dim(s): `--param 'lr=loguniform(1e-5,1e-2)'` (repeatable;
        /// merged over --space, later wins). At least one dim total is required.
        #[arg(long = "param", value_name = "NAME=FN(...)")]
        param: Vec<String>,
        /// Search algorithm: asha (default) | random | median | percentile |
        /// pbt (population-based, resume-on-promote) | tpe (Parzen ask-tell).
        #[arg(long, default_value = "asha")]
        algo: String,
        /// Objective metric — a dotted key read from each trial's StageStep
        /// payload (e.g. `val_r`).
        #[arg(long, default_value = "val_r")]
        metric: String,
        /// Optimization direction.
        #[arg(long, default_value = "max", value_parser = ["max", "min"])]
        mode: String,
        /// Number of trials to sample.
        #[arg(long, default_value_t = 8)]
        max_trials: u32,
        /// RNG seed (reproducible sampling).
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// The StageStep key carrying the trial's BUDGET coordinate (epoch/step)
        /// — rung milestones + median comparisons key on equal budget.
        #[arg(long, default_value = "epoch")]
        metric_budget_key: String,
        /// ASHA reduction factor (keep top 1/eta at each rung).
        #[arg(long, default_value_t = 3)]
        eta: u32,
        /// ASHA min / max budget (in `metric_budget_key` units) + grace before
        /// any trial may be stopped.
        #[arg(long, default_value_t = 1)]
        min_budget: u32,
        #[arg(long, default_value_t = 0)]
        max_budget: u32,
        #[arg(long, default_value_t = 1)]
        grace: u32,
        /// median/percentile: stop a trial below this percentile of peers.
        #[arg(long, default_value_t = 50)]
        percentile: u32,
        /// Promote outputs to the global cache (shared trial-cache reuse).
        #[arg(long, default_value_t = false)]
        shared_cache: bool,
        /// Placement: local (default) | slurm | ray (per-trial; see `recipe run`).
        #[arg(long, default_value = "local")]
        launcher: String,
        /// Tenant (`project[/domain]`, ADR 0096) that owns this HPO job, cache,
        /// lineage row, and RAM sub-envelope.
        #[arg(long, default_value = "default")]
        tenant: String,
        /// Experiment/campaign name for lineage grouping. Defaults to the recipe.
        #[arg(long)]
        experiment: Option<String>,
    },
    /// Leaderboard for an HPO job: per-trial best objective + status, sorted
    /// best-first. Reconstructed from `<job_dir>/hpo.json` + the durable
    /// status.jsonl stream, so it works during AND after a run.
    Show {
        /// Job id (the `blut hpo run` output, or `blut jobs`). Defaults to the
        /// most recent HPO job.
        job: Option<String>,
        /// Emit JSON instead of the text table.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Print the winning trial's overlay (the best hyperparameters) for an HPO
    /// job — ready to paste into `recipe run --args`.
    Best {
        /// Job id. Defaults to the most recent HPO job.
        job: Option<String>,
        /// Emit JSON (the overlay as an object) instead of the text summary.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

pub(super) async fn run_hpo(reg: &crate::framework::Registry, cmd: HpoCommand) -> Result<()> {
    use crate::framework::ExecCtx;
    use crate::hpo::{RandomSampler, Sampler, SearchSpace};

    // The read-only subcommands need no executor — dispatch (borrowing `cmd`) and
    // return before the launch machinery; only `Run` falls through.
    match &cmd {
        HpoCommand::Show { job, json } => return run_hpo_show(job.clone(), *json),
        HpoCommand::Best { job, json } => return run_hpo_best(job.clone(), *json),
        HpoCommand::Run { .. } => {}
    }
    let HpoCommand::Run {
        name,
        args,
        space,
        param,
        algo,
        metric,
        mode,
        max_trials,
        seed,
        metric_budget_key,
        eta,
        min_budget,
        max_budget,
        grace,
        percentile,
        shared_cache,
        launcher,
        tenant,
        experiment,
    } = cmd
    else {
        unreachable!("non-Run HpoCommand variants dispatched above")
    };

    // Base args (the fixed part; search dims overlay each trial).
    let base_args: serde_json::Value =
        serde_json::from_str(&args).map_err(|e| anyhow!("--args is not valid JSON: {e}"))?;
    let source_args = base_args.clone();

    // Search space: YAML file (if any) then inline --param (later wins), validate.
    let mut sp = match &space {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read search-space file {path}"))?;
            SearchSpace::from_yaml(&text).map_err(|e| anyhow!("{e}"))?
        }
        None => SearchSpace::default(),
    };
    for p in &param {
        let (dim, dist) = SearchSpace::parse_param(p).map_err(|e| anyhow!("{e}"))?;
        sp.dims.insert(dim, dist);
    }
    sp.validate()
        .map_err(|e| anyhow!("invalid search space: {e}"))?;

    // Sampler — random/median/percentile/asha/pbt all sample the INITIAL
    // population randomly (they differ in the control policy below: early-stop
    // for median/asha, exploit/explore clones for pbt). TPE lands in a later
    // slice (model-based sampler).
    let mut sampler: Box<dyn Sampler> = match algo.as_str() {
        // TPE's initial population is also random (the model-based ask conditions
        // on completed trials, which arrive only at runtime via the policy).
        "random" | "median" | "percentile" | "asha" | "pbt" | "tpe" => {
            Box::new(RandomSampler::new(seed))
        }
        other => {
            return Err(anyhow!(
                "--algo '{other}' is not recognized — v0.20 ships \
                 random/median/percentile/asha/pbt/tpe"
            ));
        }
    };

    let launch_target: crate::config::launcher::LaunchTarget =
        launcher.parse().map_err(|e| anyhow!("{e}"))?;
    let tenant = crate::tenant::Tenant::parse(&tenant)
        .ok_or_else(|| anyhow!("invalid --tenant '{tenant}'"))?;
    let base_args = crate::registry_args::resolve_recipe_args(base_args, &tenant, launch_target)
        .map_err(|e| anyhow!("registry arg resolution: {e}"))?;
    for (dimension, distribution) in &mut sp.dims {
        if let crate::hpo::Dist::Choice { choices } = distribution {
            for choice in choices {
                *choice = crate::registry_args::resolve_recipe_args(
                    std::mem::take(choice),
                    &tenant,
                    launch_target,
                )
                .map_err(|e| {
                    anyhow!("registry arg resolution for HPO dimension '{dimension}': {e}")
                })?;
            }
        }
    }
    let tenant_admission = crate::broker::tenant_quota::TenantAdmission::prepare(tenant.clone())
        .map_err(|e| anyhow!("tenant admission: {e}"))?;

    // Fan-out: N sampled trials → one merged plan.
    let (plan, trials) = crate::hpo::plan_build::build_hpo_plan(
        reg,
        &name,
        &base_args,
        &sp,
        sampler.as_mut(),
        max_trials,
    )
    .map_err(|e| anyhow!("{e}"))?;
    eprintln!(
        "hpo {name}: {} trials, {} nodes (algo={algo}, metric={metric}, mode={mode})",
        trials.len(),
        plan.n_nodes()
    );

    // Job + ExecCtx — mirror run_one_recipe (control=None for random search).
    // Gate on the WORST-CASE trial footprint (max over the sampled overlays):
    // if the search space tunes a memory driver (batch/tier), a trial's overlaid
    // footprint can exceed the base, and admission must reflect that. (The
    // executor's per-stage memory admission is the authoritative never-OOM gate
    // across concurrent trials; this pre-run gate is the courtesy early-refuse.)
    let footprint = trials
        .iter()
        .fold(recipe_footprint(&name, &base_args), |acc, t| {
            let mut a = base_args.clone();
            crate::hpo::apply_overlay(&mut a, &t.overlay);
            let f = recipe_footprint(&name, &a);
            if f.ram_bytes > acc.ram_bytes { f } else { acc }
        });
    let job_id = crate::jobs::new_job_id();
    let job_dir = crate::paths::job_dir(&job_id)?;
    crate::jobs::write_tenant(&job_id, &tenant)?;
    crate::jobs::write_experiment(&job_id, experiment.as_deref().unwrap_or(&name))?;
    let mut ctx = ExecCtx::new(job_dir.clone());
    ctx = ctx.with_tenant(tenant.clone());

    // Trial→topo map, computed ONCE: the executor emits a StageStep's topo
    // `node_idx`, and both the early-stop scheduler (below) and `blut hpo
    // show/best` (post-hoc, from status.jsonl) attribute it to a trial via this
    // map. Written into `<job_dir>/hpo.json` for EVERY algo (random included),
    // so the leaderboard reconstructs without a DB.
    let n_nodes = plan.n_nodes() as u32;
    let offsets: Vec<crate::framework::plan::NodeId> =
        trials.iter().map(|t| t.node_offset).collect();
    let topo = plan
        .topo_order()
        .map_err(|e| anyhow!("hpo plan topo order: {e}"))?;
    let trial_of_topo = crate::hpo::build_trial_of_topo(&topo, &offsets, n_nodes);
    {
        use crate::hpo::{HpoManifest, TrialRec};
        let recs: Vec<TrialRec> = trials
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let lo = offsets[i];
                let hi = offsets.get(i + 1).copied().unwrap_or(n_nodes);
                TrialRec {
                    trial_id: t.trial_id,
                    overlay: t.overlay.clone(),
                    n_nodes: hi - lo,
                }
            })
            .collect();
        let manifest = HpoManifest {
            recipe: name.to_string(),
            algo: algo.clone(),
            metric: metric.clone(),
            mode: mode.clone(),
            budget_key: metric_budget_key.clone(),
            trials: recs,
            trial_of_topo: trial_of_topo.clone(),
        };
        manifest
            .write_to(&job_dir)
            .with_context(|| format!("write hpo manifest for {job_id}"))?;
    }
    if let Some(budget) = tenant_admission
        .executor_budget_gib(crate::broker::admission::DEFAULT_FLOOR_GIB)
        .map_err(|e| anyhow!("tenant admission: {e}"))?
    {
        ctx = ctx.with_memory_budget(budget);
    }
    ctx = ctx.with_launch_target(launch_target);
    ctx = ctx.with_fb_warm(crate::broker::Drivers::from_args_json(&base_args).warm);
    if shared_cache {
        if let Some(global) = crate::framework::CacheHandle::default_global_path() {
            std::fs::create_dir_all(&global)
                .with_context(|| format!("create global cache dir {}", global.display()))?;
            let cache_handle = (*ctx.cache)
                .clone()
                .with_global(global)
                .with_tenant(&tenant);
            ctx.cache = std::sync::Arc::new(cache_handle);
        }
    }

    // Early-stop scheduler. The scheduler maps each StageStep's topo node_idx ->
    // trial, reads the objective + budget, and KillBranch-es underperformers:
    // median/percentile cut at the p-th percentile of peers at the same budget;
    // ASHA culls to the top 1/eta only at rung milestones. Random has
    // control=None (no early stop).
    if matches!(algo.as_str(), "median" | "percentile" | "asha") {
        use crate::hpo::{AshaStop, EarlyStop, HpoScheduler, MedianStop};
        let strategy: Box<dyn EarlyStop> = match algo.as_str() {
            "asha" => {
                if min_budget == 0 {
                    return Err(anyhow!("asha needs --min-budget >= 1 (the first rung)"));
                }
                if max_budget <= min_budget {
                    return Err(anyhow!(
                        "asha needs --max-budget ({max_budget}) > --min-budget ({min_budget})"
                    ));
                }
                let asha = AshaStop::from_budgets(min_budget as u64, max_budget as u64, eta);
                eprintln!("hpo: asha rungs={:?} eta={eta}", asha.rungs);
                Box::new(asha)
            }
            "median" => Box::new(MedianStop {
                percentile: 50.0,
                min_peers: 2,
            }),
            _ => Box::new(MedianStop {
                percentile: percentile as f64,
                min_peers: 2,
            }),
        };
        let sched = HpoScheduler::new(
            trial_of_topo,
            metric.clone(),
            metric_budget_key.clone(),
            mode == "max",
            grace as u64,
            strategy,
        );
        ctx = ctx.with_control(with_nan_safety(std::sync::Arc::new(sched)));
        eprintln!(
            "hpo: {algo} early-stop (metric={metric} {mode}, budget-key={metric_budget_key}, grace={grace})"
        );
    } else if algo == "pbt" {
        // Population-Based Training: at each rung a below-quantile trial is
        // KillBranch'd and a perturbed clone of the best survivor is Spawn'd,
        // warm-started from the winner's checkpoint dir (`--resume-from`, baked
        // into the clone's args by the factory below).
        use crate::hpo::{PbtConfig, PbtPolicy, PbtTrial};
        if min_budget == 0 || max_budget <= min_budget {
            return Err(anyhow!(
                "pbt needs --min-budget >= 1 and --max-budget > --min-budget (the rungs)"
            ));
        }
        let rungs = crate::hpo::AshaStop::rung_ladder(min_budget as u64, max_budget as u64, eta);
        // Each trial's checkpoint dir = its TERMINAL node's stage dir
        // (`<job_dir>/stages/<topo_idx>-<stage>`); a clone resumes from the
        // winner's. The terminal node is the last topo position the trial owns.
        let pg = plan
            .graph_structure()
            .map_err(|e| anyhow!("pbt: plan graph: {e}"))?;
        let mut terminal_topo: Vec<Option<usize>> = vec![None; trials.len()];
        for (p, t) in trial_of_topo.iter().enumerate() {
            if let Some(t) = t {
                terminal_topo[*t as usize] = Some(p); // topo ascending → last wins
            }
        }
        let pbt_trials: Vec<PbtTrial> = trials
            .iter()
            .enumerate()
            .map(|(i, tp)| {
                let resume_dir = terminal_topo[i]
                    .and_then(|p| pg.nodes.get(p))
                    .map(|n| {
                        job_dir
                            .join("stages")
                            .join(format!("{}-{}", n.idx, n.stage_name))
                    })
                    .unwrap_or_else(|| job_dir.clone());
                PbtTrial {
                    overlay: tp.overlay.clone(),
                    resume_dir,
                }
            })
            .collect();
        // The clone factory: perturbed overlay + `resume_from` arg → recompile.
        // `compile_fn` is a plain fn pointer (`'static`), so it captures cleanly.
        let def = reg
            .find(&name)
            .ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
        let cfn = def.compile_fn;
        let base_for_factory = base_args.clone();
        let factory: crate::hpo::TrialFactory = std::sync::Arc::new(move |overlay, resume| {
            let mut a = base_for_factory.clone();
            crate::hpo::apply_overlay(&mut a, overlay);
            if let Some(obj) = a.as_object_mut() {
                obj.insert(
                    "resume_from".into(),
                    serde_json::json!(resume.resume_dir.to_string_lossy()),
                );
            }
            cfn(a).map_err(|e| format!("{e}"))
        });
        let cfg = PbtConfig {
            metric_key: metric.clone(),
            budget_key: metric_budget_key.clone(),
            maximize: mode == "max",
            rungs: rungs.clone(),
            bottom_quantile: percentile as f64,
            min_peers: 2,
            max_spawns: (max_trials as usize).saturating_mul(8).max(1),
        };
        let sched = PbtPolicy::new(trial_of_topo, pbt_trials, sp.clone(), cfg, factory, seed);
        ctx = ctx.with_control(with_nan_safety(std::sync::Arc::new(sched)));
        eprintln!(
            "hpo: pbt rungs={rungs:?} (metric={metric} {mode}, cull<p{percentile}, resume-on-promote)"
        );
    } else if algo == "tpe" {
        // TPE: the fan-out is the random initial population; as each trial
        // completes (reaches --max-budget) the policy tells the Parzen model and
        // Spawns a fresh suggested trial (no resume — TPE explores fresh).
        use crate::hpo::{TpeConfig, TpePolicy, TpePolicyConfig, TpeSampler};
        if max_budget == 0 {
            return Err(anyhow!(
                "tpe needs --max-budget >= 1 (the per-trial completion budget)"
            ));
        }
        let trial_overlays: Vec<crate::hpo::Overlay> =
            trials.iter().map(|t| t.overlay.clone()).collect();
        let def = reg
            .find(&name)
            .ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
        let cfn = def.compile_fn;
        let base_for_factory = base_args.clone();
        let factory: crate::hpo::FreshFactory = std::sync::Arc::new(move |overlay| {
            let mut a = base_for_factory.clone();
            crate::hpo::apply_overlay(&mut a, overlay);
            cfn(a).map_err(|e| format!("{e}"))
        });
        let sampler = TpeSampler::new(
            TpeConfig {
                maximize: mode == "max",
                ..TpeConfig::default()
            },
            seed,
        );
        let cfg = TpePolicyConfig {
            metric_key: metric.clone(),
            budget_key: metric_budget_key.clone(),
            max_budget: max_budget as u64,
            max_spawns: (max_trials as usize).max(1),
        };
        let sched = TpePolicy::new(
            trial_of_topo,
            trial_overlays,
            sp.clone(),
            cfg,
            sampler,
            factory,
        );
        ctx = ctx.with_control(with_nan_safety(std::sync::Arc::new(sched)));
        eprintln!(
            "hpo: tpe (metric={metric} {mode}, complete@{max_budget}, ≤{max_trials} suggested)"
        );
    }

    RecipeMarker {
        name: name.to_string(),
        args: base_args.clone(),
        source_args: Some(source_args),
    }
    .write_to(&job_dir)?;
    crate::jobs::write_state(&job_id, JobState::Running)
        .with_context(|| format!("write Running state for {job_id}"))?;
    crate::python_kill::bind_current_job(job_id.clone());
    install_cancel_handler(ctx.cancel.clone());

    // Admission gate on a SINGLE trial's footprint — the executor's per-stage
    // memory admission gates concurrency ACROSS trials, so the box can't OOM
    // even with the full fan-out in flight (never-OOM-the-box, unchanged).
    let _tenant_reservation =
        match tenant_admission.reserve(&footprint, crate::broker::admission::DEFAULT_FLOOR_GIB) {
            Ok(reservation) => reservation,
            Err(reason) => {
                crate::python_kill::unbind_current_job();
                let _ = crate::jobs::write_state(&job_id, JobState::Failed);
                return Err(anyhow!("hpo '{name}' admission refused: {reason}"));
            }
        };
    let lock =
        match scheduler_lock::acquire_exclusive(format!("blut-hpo:{job_id}"), LockKind::Training) {
            Ok(l) => l,
            Err(e) => {
                crate::python_kill::unbind_current_job();
                let _ = crate::jobs::write_state(&job_id, JobState::Failed);
                return Err(anyhow!("acquire_exclusive: {e}"));
            }
        };
    eprintln!("job    {job_id}");
    eprintln!("dir    {}", job_dir.display());
    eprintln!("lock   {}", lock.path().display());

    persist_plan_graph(&plan, &job_dir);
    let result = crate::framework::execute_plan(plan, ctx).await;
    drop(lock);
    crate::python_kill::unbind_current_job();
    match result {
        Ok(_) => {
            crate::jobs::write_state(&job_id, JobState::Done)
                .with_context(|| format!("write Done state for {job_id}"))?;
            if let Err(e) = crate::lineage_db::ingest_job(&job_id, &name, "done") {
                tracing::warn!("lineage index {job_id}: {e}");
            }
            eprintln!(
                "hpo done: {} trials ran (job {job_id}). Leaderboard: `blut hpo show {job_id}`; \
                 winning config: `blut hpo best {job_id}`.",
                trials.len()
            );
            Ok(())
        }
        Err(e) => {
            let _ = crate::jobs::write_state(&job_id, JobState::Failed);
            if let Err(ie) = crate::lineage_db::ingest_job(&job_id, &name, "failed") {
                tracing::warn!("lineage index {job_id}: {ie}");
            }
            // See the `plan execution failed` site in `run_plan_cmd` — same
            // chain-preservation rationale (ADR 0072 A2).
            Err(anyhow::Error::from(e).context("hpo plan execution failed"))
        }
    }
}

/// Resolve the HPO job to inspect: an explicit id (via `jobs::resolve_job_id`,
/// so a prefix works) or — when omitted — the most recent job carrying an
/// `hpo.json` manifest (job ids are timestamp-monotonic, sorted ascending).
pub(super) fn resolve_hpo_job(job: Option<String>) -> Result<(String, crate::hpo::HpoManifest)> {
    let load = |id: &str| -> Option<crate::hpo::HpoManifest> {
        let dir = crate::paths::job_dir(id).ok()?;
        crate::hpo::HpoManifest::read_from(&dir)
    };
    let id = match job {
        Some(q) => crate::jobs::resolve_job_id(&q).map_err(|e| anyhow!("{e}"))?,
        None => crate::jobs::list_jobs()
            .map_err(|e| anyhow!("list jobs: {e}"))?
            .into_iter()
            .rev()
            .map(|s| s.id)
            .find(|id| load(id).is_some())
            .ok_or_else(|| anyhow!("no HPO jobs found (run `blut hpo run ...` first)"))?,
    };
    let manifest =
        load(&id).ok_or_else(|| anyhow!("job '{id}' has no hpo.json (not an HPO run?)"))?;
    Ok((id, manifest))
}

/// `blut hpo show [job] [--json]` — the trial leaderboard.
pub(super) fn run_hpo_show(job: Option<String>, json: bool) -> Result<()> {
    let (id, manifest) = resolve_hpo_job(job)?;
    let lines = crate::jobs::read_status_lines(&id).map_err(|e| anyhow!("read status: {e}"))?;
    let board = crate::hpo::leaderboard(&manifest, &lines);
    if json {
        let arr: Vec<_> = board
            .iter()
            .map(|o| {
                serde_json::json!({
                    "trial_id": o.trial_id,
                    "objective": o.objective,
                    "status": o.status,
                    "overlay": serde_json::Map::from_iter(
                        o.overlay.iter().map(|(k, v)| (k.clone(), v.clone())),
                    ),
                })
            })
            .collect();
        emit_json(&serde_json::json!({
            "job": id,
            "recipe": manifest.recipe,
            "algo": manifest.algo,
            "metric": manifest.metric,
            "mode": manifest.mode,
            "trials": arr,
        }))?;
        return Ok(());
    }
    println!(
        "hpo {} (job {id}) — {} {} ({} trials)",
        manifest.recipe,
        manifest.metric,
        manifest.mode,
        board.len()
    );
    println!(
        "{:<6} {:<10} {:<8} overlay",
        "trial", manifest.metric, "status"
    );
    for o in &board {
        let obj = match o.objective {
            Some(x) => format!("{x:.4}"),
            None => "—".to_string(),
        };
        let overlay = o
            .overlay
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("{:<6} {:<10} {:<8} {}", o.trial_id, obj, o.status, overlay);
    }
    Ok(())
}

/// `blut hpo best [job] [--json]` — the winning trial's overlay.
pub(super) fn run_hpo_best(job: Option<String>, json: bool) -> Result<()> {
    let (id, manifest) = resolve_hpo_job(job)?;
    let lines = crate::jobs::read_status_lines(&id).map_err(|e| anyhow!("read status: {e}"))?;
    let board = crate::hpo::leaderboard(&manifest, &lines);
    let best = board
        .iter()
        .find(|o| o.objective.is_some())
        .ok_or_else(|| anyhow!("no trial reported metric '{}' yet", manifest.metric))?;
    let overlay_obj =
        serde_json::Map::from_iter(best.overlay.iter().map(|(k, v)| (k.clone(), v.clone())));
    if json {
        emit_json(&serde_json::Value::Object(overlay_obj))?;
        return Ok(());
    }
    let best_obj = best
        .objective
        .expect("find() above guarantees objective.is_some()");
    println!(
        "best trial {} — {}={best_obj:.4} (job {id})",
        best.trial_id, manifest.metric,
    );
    for (k, v) in &best.overlay {
        println!("  {k} = {v}");
    }
    println!(
        "\nreproduce: blut recipe run {} --args '{}'",
        manifest.recipe,
        serde_json::to_string(&serde_json::Value::Object(overlay_obj.clone()))
            .unwrap_or_else(|_| "{}".into())
    );
    Ok(())
}
