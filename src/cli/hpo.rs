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
        /// Force every declaring training stage onto its explicit Inline I/O
        /// profile. This is execution-only and does not change trial/cache identity.
        #[arg(long, default_value_t = false)]
        sync_io: bool,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HpoTrialAdmission {
    resolved_footprint: crate::broker::Footprint,
    sync_footprint: crate::broker::Footprint,
    node: crate::framework::executor::TrainingIoNodeAdmission,
}

/// Resolve one trial's launch facts from the HPO job's immutable tenant
/// snapshot. This is also the callback body retained for runtime PBT/TPE
/// children, so fresh and injected trials cannot drift or re-probe.
fn hpo_trial_admission(
    recipe: &str,
    recipe_args: &serde_json::Value,
    declared: Option<&(String, blut_types::envelope::ResourceEnvelope)>,
    snapshot: &crate::broker::ResourceSnapshot,
) -> HpoTrialAdmission {
    let admitted_workers = admitted_workers_for(recipe, recipe_args, declared, snapshot);
    let admitted_batch_size = admitted_workers.and_then(|workers| {
        admitted_batch_size_for(recipe, recipe_args, declared, workers, snapshot)
    });
    let resolved_footprint = match admitted_workers {
        Some(workers) => recipe_footprint_tuned(recipe, recipe_args, workers, admitted_batch_size),
        None => recipe_footprint(recipe, recipe_args),
    };
    let resolved_workers = admitted_workers
        .unwrap_or_else(|| crate::broker::Drivers::from_args_json(recipe_args).workers);
    let sync_footprint = recipe_footprint_sync_base(
        recipe_args,
        declared,
        admitted_batch_size,
        resolved_workers,
        resolved_footprint,
    );
    HpoTrialAdmission {
        resolved_footprint,
        sync_footprint,
        node: crate::framework::executor::TrainingIoNodeAdmission {
            admitted_decode_workers: admitted_workers,
            admitted_batch_size,
            cache_warm: crate::broker::Drivers::from_args_json(recipe_args).warm,
            calibrated_base_floor_bytes: Some(sync_footprint.ram_bytes),
            selection_budget_bytes: None,
        },
    }
}

fn hpo_trial_recipe_args(
    plan: &crate::framework::plan::CompiledPlan,
    trials: &[crate::hpo::TrialPlan],
) -> anyhow::Result<Vec<std::sync::Arc<serde_json::Value>>> {
    let nodes = plan.exec_view().nodes;
    trials
        .iter()
        .map(|trial| {
            nodes
                .get(trial.node_offset as usize)
                .and_then(|node| node.admission_scope_args.clone())
                .ok_or_else(|| {
                    anyhow!(
                        "HPO trial {} has no component admission provenance",
                        trial.trial_id
                    )
                })
        })
        .collect()
}

fn hpo_trial_index(
    node: &crate::framework::plan::PlanNode,
    trial_recipe_args: &[std::sync::Arc<serde_json::Value>],
) -> Option<usize> {
    let admission_args = node.admission_scope_args.as_ref()?;
    trial_recipe_args
        .iter()
        .position(|trial_args| std::sync::Arc::ptr_eq(trial_args, admission_args))
}

fn hpo_trial_of_topo(
    plan: &crate::framework::plan::CompiledPlan,
    trial_recipe_args: &[std::sync::Arc<serde_json::Value>],
) -> anyhow::Result<Vec<Option<u32>>> {
    let nodes = plan.exec_view().nodes;
    plan.topo_order()
        .map_err(|error| anyhow!("hpo plan topo order: {error}"))?
        .into_iter()
        .map(|node_id| {
            let node = &nodes[node_id as usize];
            let trial = hpo_trial_index(node, trial_recipe_args).ok_or_else(|| {
                anyhow!(
                    "optimized HPO node {} ({}) lost trial admission provenance",
                    node.id,
                    node.stage.name()
                )
            })?;
            u32::try_from(trial)
                .map(Some)
                .map_err(|_| anyhow!("HPO trial index {trial} exceeds u32"))
        })
        .collect()
}

/// Reserve the worst exact selected trial envelope. The executor gates
/// concurrency across trials, so HPO holds one whole-job tenant reservation;
/// its size is the component-wise maximum of the independently selected trial
/// envelopes. A trial with no declaring stage retains its exact legacy bill.
fn hpo_selected_footprint(
    plan: &crate::framework::plan::CompiledPlan,
    trial_recipe_args: &[std::sync::Arc<serde_json::Value>],
    trials: &[HpoTrialAdmission],
    profiles: &std::collections::HashMap<
        crate::framework::plan::NodeId,
        crate::framework::TrainingIoProfile,
    >,
) -> anyhow::Result<crate::broker::Footprint> {
    if trial_recipe_args.len() != trials.len() {
        return Err(anyhow!(
            "HPO admission bookkeeping mismatch: {} trial args, {} trial envelopes",
            trial_recipe_args.len(),
            trials.len()
        ));
    }
    let nodes = plan.exec_view().nodes;
    let mut selected_ram = vec![None::<u64>; trials.len()];
    let mut declaring_nodes = vec![0usize; trials.len()];
    for (&node_id, profile) in profiles {
        let node = nodes
            .get(node_id as usize)
            .ok_or_else(|| anyhow!("selected HPO profile refers to missing node {node_id}"))?;
        let trial = hpo_trial_index(node, trial_recipe_args).ok_or_else(|| {
            anyhow!(
                "selected HPO node {} ({}) has no trial admission provenance",
                node.id,
                node.stage.name()
            )
        })?;
        declaring_nodes[trial] += 1;
        if declaring_nodes[trial] > 1 {
            return Err(anyhow!(
                "HPO trial {trial} declares async-I/O profiles on more than one node; exact whole-trial admission currently supports one declaring training node"
            ));
        }
        if profile.sync_base_bytes < trials[trial].sync_footprint.ram_bytes {
            return Err(anyhow!(
                "HPO trial {trial} selected profile base {} bytes is below calibrated floor {} bytes",
                profile.sync_base_bytes,
                trials[trial].sync_footprint.ram_bytes
            ));
        }
        let exact = profile
            .sync_base_bytes
            .checked_add(profile.billed_overhead_bytes)
            .ok_or_else(|| anyhow!("HPO trial {trial} async-I/O footprint overflow"))?;
        selected_ram[trial] = Some(exact);
    }

    let mut worst = crate::broker::Footprint {
        ram_bytes: 0,
        vram_mib: 0,
    };
    for (trial, selected) in trials.iter().zip(selected_ram) {
        worst.ram_bytes = worst
            .ram_bytes
            .max(selected.unwrap_or(trial.resolved_footprint.ram_bytes));
        worst.vram_mib = worst.vram_mib.max(trial.resolved_footprint.vram_mib);
    }
    Ok(worst)
}

fn hpo_node_admission_resolver(
    recipe: String,
    snapshot: crate::broker::ResourceSnapshot,
    initial: impl IntoIterator<Item = (std::sync::Arc<serde_json::Value>, HpoTrialAdmission)>,
    dynamic_limit: std::sync::Arc<std::sync::OnceLock<crate::broker::Footprint>>,
) -> anyhow::Result<crate::framework::executor::TrainingIoNodeAdmissionResolver> {
    let mut initial_cache = std::collections::HashMap::new();
    for (recipe_args, admission) in initial {
        let key = crate::framework::CacheHandle::canonical_json_bytes(recipe_args.as_ref());
        if let Some(previous) = initial_cache.insert(key, admission)
            && previous != admission
        {
            return Err(anyhow!(
                "identical defaulted HPO recipe args resolved to conflicting admission facts"
            ));
        }
    }
    let cache = std::sync::Arc::new(std::sync::Mutex::new(initial_cache));
    Ok(std::sync::Arc::new(
        move |_stage_name, _node_args, recipe_args| {
            let key = crate::framework::CacheHandle::canonical_json_bytes(recipe_args);
            let admission = if let Some(admission) = cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&key)
                .copied()
            {
                admission
            } else {
                // Runtime-injected trial: no compiled sub-plan is threaded here; the
                // JSON path bills it (the declared-envelope spawn guard already bounds
                // any oversized suggestion). Per-trial envelopes cover initial trials.
                let admission = hpo_trial_admission(&recipe, recipe_args, None, &snapshot);
                let mut cached = cache
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *cached.entry(key).or_insert(admission)
            };
            constrain_dynamic_hpo_admission(admission, dynamic_limit.get()).map(Some)
        },
    ))
}

/// A runtime PBT/TPE child must fit inside the one immutable reservation held
/// for this HPO job. Its synchronous/legacy envelope and known VRAM cannot
/// exceed the initial worst trial, while profile selection receives the same
/// RAM ceiling so retained async bytes downgrade to Inline or refuse.
fn constrain_dynamic_hpo_admission(
    admission: HpoTrialAdmission,
    limit: Option<&crate::broker::Footprint>,
) -> Result<crate::framework::executor::TrainingIoNodeAdmission, String> {
    let mut node = admission.node;
    let Some(limit) = limit else {
        return Ok(node);
    };
    let required_ram = admission
        .resolved_footprint
        .ram_bytes
        .max(admission.sync_footprint.ram_bytes);
    if required_ram > limit.ram_bytes {
        return Err(format!(
            "runtime HPO child needs {required_ram} RAM bytes, above the held {}-byte tenant reservation",
            limit.ram_bytes
        ));
    }
    let required_vram = admission
        .resolved_footprint
        .vram_mib
        .max(admission.sync_footprint.vram_mib);
    if limit.vram_mib != 0 && required_vram > limit.vram_mib {
        return Err(format!(
            "runtime HPO child needs {required_vram} MiB VRAM, above the held {}-MiB reservation",
            limit.vram_mib
        ));
    }
    node.selection_budget_bytes = Some(limit.ram_bytes);
    Ok(node)
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
        sync_io,
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

    // `from_components` keeps each defaulted trial recipe args behind a private,
    // admission-only Arc. The Arc identity survives optimization/DCE and remains
    // distinct even when a sampler repeats an identical overlay.
    let trial_recipe_args = hpo_trial_recipe_args(&plan, &trials)?;
    // ADR 0133: bill each trial its OWN declared envelope (node-range scoped) —
    // a small trial is never billed the biggest trial's footprint.
    let trial_envelopes: Vec<_> = trials
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let end = trials
                .get(i + 1)
                .map(|n| n.node_offset)
                .unwrap_or(plan.n_nodes() as u32);
            plan.max_declared_envelope_in(t.node_offset..end)
        })
        .collect();
    let trial_admissions: Vec<_> = trial_recipe_args
        .iter()
        .zip(&trial_envelopes)
        .map(|(args, env)| {
            hpo_trial_admission(&name, args, env.as_ref(), tenant_admission.snapshot())
        })
        .collect();
    let dynamic_admission_limit = std::sync::Arc::new(std::sync::OnceLock::new());
    let node_admission = hpo_node_admission_resolver(
        name.clone(),
        tenant_admission.snapshot().clone(),
        trial_recipe_args
            .iter()
            .cloned()
            .zip(trial_admissions.iter().copied()),
        std::sync::Arc::clone(&dynamic_admission_limit),
    )?;

    // Job + ExecCtx — mirror run_one_recipe while retaining one immutable HPO
    // admission witness for both initial and runtime-injected trials.
    let job_id = crate::jobs::new_job_id();
    let job_dir = crate::paths::jobs_dir()?.join(&job_id);
    let mut ctx = ExecCtx::new(job_dir.clone());
    ctx = ctx.with_tenant(tenant.clone());
    ctx = ctx.with_sync_io(sync_io);

    if let Some(budget) = tenant_admission
        .executor_budget_gib(crate::broker::admission::DEFAULT_FLOOR_GIB)
        .map_err(|e| anyhow!("tenant admission: {e}"))?
    {
        ctx = ctx.with_memory_budget(budget);
    }
    ctx = ctx.with_launch_target(launch_target);
    ctx = ctx.with_fb_warm(crate::broker::Drivers::from_args_json(&base_args).warm);
    if let Some(budget_bytes) =
        training_io_live_budget_bytes(tenant_admission.snapshot(), launch_target)
    {
        ctx = ctx.with_training_io_selection_budget_bytes(budget_bytes);
    }
    ctx = ctx.with_training_io_node_admission_resolver(node_admission);
    if shared_cache && let Some(global) = crate::framework::CacheHandle::default_global_path() {
        std::fs::create_dir_all(&global)
            .with_context(|| format!("create global cache dir {}", global.display()))?;
        let cache_handle = (*ctx.cache)
            .clone()
            .with_global(global)
            .with_tenant(&tenant);
        ctx.cache = std::sync::Arc::new(cache_handle);
    }

    // Force the HPO fan-out through the parallel optimizer, then select every
    // initial declaring node exactly once in the post-DCE id space. The normal
    // execute entry point consumes this private witness and cannot prepare a
    // second time.
    let plan = crate::framework::executor::prepare_plan_for_parallel_execution(plan, &mut ctx)
        .map_err(|error| anyhow!("hpo '{name}' execution preparation: {error}"))?;
    let footprint = hpo_selected_footprint(
        &plan,
        &trial_recipe_args,
        &trial_admissions,
        &ctx.training_io_profiles,
    )?;
    dynamic_admission_limit
        .set(footprint)
        .map_err(|_| anyhow!("HPO dynamic admission limit was already initialized"))?;

    // Trial→topo is derived only after optimization because lifecycle node_idx
    // is a post-DCE topo position. Arc provenance keeps repeated overlays
    // distinct without putting trial identity into cache keys.
    let trial_of_topo = hpo_trial_of_topo(&plan, &trial_recipe_args)?;
    let hpo_manifest = {
        use crate::hpo::{HpoManifest, TrialRec};
        let mut nodes_per_trial = vec![0u32; trials.len()];
        for trial in trial_of_topo.iter().flatten() {
            let count = nodes_per_trial
                .get_mut(*trial as usize)
                .ok_or_else(|| anyhow!("optimized HPO topo refers to unknown trial {trial}"))?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| anyhow!("HPO trial {trial} node count overflow"))?;
        }
        let recs: Vec<TrialRec> = trials
            .iter()
            .enumerate()
            .map(|(i, trial)| TrialRec {
                trial_id: trial.trial_id,
                overlay: trial.overlay.clone(),
                n_nodes: nodes_per_trial[i],
            })
            .collect();
        HpoManifest {
            recipe: name.to_string(),
            algo: algo.clone(),
            metric: metric.clone(),
            mode: mode.clone(),
            budget_key: metric_budget_key.clone(),
            trials: recs,
            trial_of_topo: trial_of_topo.clone(),
        }
    };

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

    // Preparation and algorithm validation above are side-effect free with
    // respect to the job registry. Materialize only when this launch is ready
    // to enter the same reservation/lock/execution path as a prepared recipe.
    let materialized_job_dir = crate::paths::job_dir(&job_id)?;
    debug_assert_eq!(materialized_job_dir, job_dir);
    crate::jobs::write_tenant(&job_id, &tenant)
        .with_context(|| format!("persist tenant for {job_id}"))?;
    crate::jobs::write_experiment(&job_id, experiment.as_deref().unwrap_or(&name))
        .with_context(|| format!("persist experiment for {job_id}"))?;
    hpo_manifest
        .write_to(&job_dir)
        .with_context(|| format!("write hpo manifest for {job_id}"))?;

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
    // even with the full fan-out in flight (memory-admission, unchanged).
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

#[cfg(test)]
mod hpo_run_flag_tests {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use async_trait::async_trait;
    use clap::Parser;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::{
        Cli, Command, HpoCommand, HpoTrialAdmission, constrain_dynamic_hpo_admission,
        hpo_node_admission_resolver, hpo_selected_footprint, hpo_trial_of_topo,
        hpo_trial_recipe_args,
    };
    use crate::broker::Footprint;
    use crate::framework::artifact::{Artifact, ContentHash};
    use crate::framework::async_io::{
        IoMode, TrainingIoCandidate, TrainingIoDowngradeReason, TrainingIoHints,
    };
    use crate::framework::error::StageError;
    use crate::framework::executor::{
        ExecCtx, TrainingIoNodeAdmission, prepare_plan_for_parallel_execution,
    };
    use crate::framework::plan::CompiledPlan;
    use crate::framework::resource::Resource;
    use crate::framework::stage::{Stage, StageContext, StageDyn};
    use crate::hpo::TrialPlan;

    const GIB: u64 = crate::broker::footprint::GIB;
    const MIB: u64 = 1024 * 1024;

    #[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
    struct AdmissionArgs {}

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct AdmissionValue;

    impl Artifact for AdmissionValue {
        const KIND: &'static str = "hpo-admission-value";
        const SCHEMA: u32 = 1;

        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(Self::KIND.as_bytes())
        }

        fn primary_path(&self) -> &Path {
            Path::new("")
        }
    }

    struct AdmissionStage;

    #[async_trait]
    impl Stage for AdmissionStage {
        const NAME: &'static str = "hpo_admission_stage";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = AdmissionValue;
        type Args = AdmissionArgs;

        fn training_io_sync_base_bytes(&self, _args: &Self::Args, hints: TrainingIoHints) -> u64 {
            u64::from(hints.admitted_batch_size.unwrap_or(1)) * GIB
        }

        fn training_io_candidates(
            &self,
            _args: &Self::Args,
            hints: TrainingIoHints,
        ) -> Vec<TrainingIoCandidate> {
            let batch_bytes = match hints.admitted_batch_size {
                Some(2) => 512 * MIB,
                _ => 256 * MIB,
            };
            vec![
                TrainingIoCandidate {
                    data_replicas: 1,
                    decode_workers: 1,
                    prefetch_per_worker: 1,
                    cuda_staging_slots: 0,
                    pipeline: IoMode::Inline,
                    metrics: IoMode::Inline,
                    checkpoints: IoMode::Inline,
                    batch_bytes: Some(batch_bytes),
                    checkpoint_snapshot_bytes: Some(0),
                    fixed_overhead_bytes: Some(0),
                },
                TrainingIoCandidate {
                    data_replicas: 1,
                    decode_workers: 0,
                    prefetch_per_worker: 0,
                    cuda_staging_slots: 0,
                    pipeline: IoMode::Inline,
                    metrics: IoMode::Inline,
                    checkpoints: IoMode::Inline,
                    batch_bytes: Some(0),
                    checkpoint_snapshot_bytes: Some(0),
                    fixed_overhead_bytes: Some(0),
                },
            ]
        }

        async fn run(
            &self,
            _ctx: &StageContext,
            _input: (),
            _args: &AdmissionArgs,
        ) -> Result<AdmissionValue, StageError> {
            Ok(AdmissionValue)
        }
    }

    fn component(recipe_args: serde_json::Value) -> CompiledPlan {
        let stage: Arc<dyn StageDyn> = Arc::new(AdmissionStage);
        CompiledPlan::from_erased_chain(
            "hpo-admission-component",
            recipe_args,
            vec![(stage, serde_json::json!({}))],
        )
        .expect("compile HPO admission component")
    }

    fn merged_plan(recipe_args: [serde_json::Value; 2]) -> (CompiledPlan, Vec<TrialPlan>) {
        let components = recipe_args.into_iter().map(component).collect();
        let (plan, offsets) = CompiledPlan::from_components(
            "hpo-admission".into(),
            serde_json::json!({}),
            components,
        );
        let trials = offsets
            .into_iter()
            .enumerate()
            .map(|(trial, node_offset)| TrialPlan {
                trial_id: trial as u32,
                overlay: Vec::new(),
                node_offset,
            })
            .collect();
        (plan, trials)
    }

    fn nested_component(recipe_args: serde_json::Value, leaf: u32) -> CompiledPlan {
        CompiledPlan::from_components(
            "nested-hpo-component".into(),
            recipe_args,
            vec![
                component(serde_json::json!({"leaf": leaf * 10})),
                component(serde_json::json!({"leaf": leaf * 10 + 1})),
            ],
        )
        .0
    }

    fn admission(batch: u32, vram_mib: u64) -> HpoTrialAdmission {
        let sync_ram = u64::from(batch) * GIB;
        HpoTrialAdmission {
            resolved_footprint: Footprint {
                ram_bytes: sync_ram,
                vram_mib,
            },
            sync_footprint: Footprint {
                ram_bytes: sync_ram,
                vram_mib,
            },
            node: TrainingIoNodeAdmission {
                admitted_decode_workers: Some(1),
                admitted_batch_size: Some(batch),
                cache_warm: false,
                calibrated_base_floor_bytes: Some(sync_ram),
                selection_budget_bytes: None,
            },
        }
    }

    fn sync_io_of(argv: &[&str]) -> bool {
        match Cli::try_parse_from(argv).expect("parse").command {
            Some(Command::Hpo {
                cmd: HpoCommand::Run { sync_io, .. },
            }) => sync_io,
            other => panic!("expected hpo run, got {other:?}"),
        }
    }

    #[test]
    fn sync_io_is_explicit_and_defaults_false() {
        assert!(!sync_io_of(&[
            "blut",
            "hpo",
            "run",
            "demo",
            "--param",
            "lr=choice(0.1,0.2)",
        ]));
        assert!(sync_io_of(&[
            "blut",
            "hpo",
            "run",
            "demo",
            "--param",
            "lr=choice(0.1,0.2)",
            "--sync-io",
        ]));
    }

    #[test]
    fn repeated_trial_args_keep_distinct_admission_provenance() {
        let repeated = serde_json::json!({"batch": 1});
        let (plan, trials) = merged_plan([repeated.clone(), repeated]);
        let trial_args = hpo_trial_recipe_args(&plan, &trials).expect("trial provenance");

        assert_eq!(trial_args[0].as_ref(), trial_args[1].as_ref());
        assert!(
            !Arc::ptr_eq(&trial_args[0], &trial_args[1]),
            "identical sampled overlays still identify distinct trials"
        );
        assert_eq!(
            hpo_trial_of_topo(&plan, &trial_args).expect("trial topo"),
            vec![Some(0), Some(1)]
        );
    }

    #[test]
    fn nested_trial_components_keep_the_outer_hpo_attribution() {
        let outer_args = [
            serde_json::json!({"trial": 1}),
            serde_json::json!({"trial": 2}),
        ];
        let (plan, offsets) = CompiledPlan::from_components(
            "nested-hpo".into(),
            serde_json::json!({}),
            vec![
                nested_component(outer_args[0].clone(), 1),
                nested_component(outer_args[1].clone(), 2),
            ],
        );
        let trials: Vec<_> = offsets
            .into_iter()
            .enumerate()
            .map(|(trial, node_offset)| TrialPlan {
                trial_id: trial as u32,
                overlay: Vec::new(),
                node_offset,
            })
            .collect();
        let trial_args = hpo_trial_recipe_args(&plan, &trials).expect("outer trial provenance");

        assert_eq!(trial_args[0].as_ref(), &outer_args[0]);
        assert_eq!(trial_args[1].as_ref(), &outer_args[1]);
        assert_eq!(
            hpo_trial_of_topo(&plan, &trial_args).expect("nested trial topo"),
            vec![Some(0), Some(0), Some(1), Some(1)]
        );
        assert_eq!(
            plan.exec_view().nodes[1]
                .admission_recipe_args
                .as_deref()
                .expect("specific leaf provenance remains available"),
            &serde_json::json!({"leaf": 11})
        );
    }

    #[test]
    fn hpo_profiles_bill_the_worst_exact_selected_trial() {
        let (plan, trials) = merged_plan([
            serde_json::json!({"batch": 1}),
            serde_json::json!({"batch": 2}),
        ]);
        let trial_args = hpo_trial_recipe_args(&plan, &trials).expect("trial provenance");
        let admissions = [admission(1, 1024), admission(2, 2048)];
        let resolver = hpo_node_admission_resolver(
            "hpo-admission".into(),
            crate::broker::ResourceSnapshot::default(),
            trial_args.iter().cloned().zip(admissions),
            Arc::new(std::sync::OnceLock::new()),
        )
        .expect("node resolver");
        let temp = tempfile::tempdir().expect("tempdir");
        let mut ctx = ExecCtx::new(PathBuf::from(temp.path()))
            .with_memory_budget(4)
            .with_training_io_selection_budget_bytes(4 * GIB)
            .with_training_io_node_admission_resolver(resolver);

        let plan = prepare_plan_for_parallel_execution(plan, &mut ctx).expect("prepare HPO plan");
        assert_eq!(ctx.training_io_profiles.len(), 2);
        assert_eq!(
            hpo_trial_of_topo(&plan, &trial_args).expect("optimized trial topo"),
            vec![Some(0), Some(1)]
        );
        let footprint =
            hpo_selected_footprint(&plan, &trial_args, &admissions, &ctx.training_io_profiles)
                .expect("exact selected footprint");
        assert_eq!(footprint.ram_bytes, 2 * GIB + 512 * MIB);
        assert_eq!(footprint.vram_mib, 2048);
    }

    #[test]
    fn hpo_sync_io_selects_inline_and_bills_only_the_exact_base() {
        let (plan, trials) = merged_plan([
            serde_json::json!({"batch": 1}),
            serde_json::json!({"batch": 2}),
        ]);
        let trial_args = hpo_trial_recipe_args(&plan, &trials).expect("trial provenance");
        let admissions = [admission(1, 1024), admission(2, 2048)];
        let resolver = hpo_node_admission_resolver(
            "hpo-admission".into(),
            crate::broker::ResourceSnapshot::default(),
            trial_args.iter().cloned().zip(admissions),
            Arc::new(std::sync::OnceLock::new()),
        )
        .expect("node resolver");
        let temp = tempfile::tempdir().expect("tempdir");
        let mut ctx = ExecCtx::new(PathBuf::from(temp.path()))
            .with_memory_budget(4)
            .with_training_io_selection_budget_bytes(4 * GIB)
            .with_sync_io(true)
            .with_training_io_node_admission_resolver(resolver);

        let plan = prepare_plan_for_parallel_execution(plan, &mut ctx).expect("prepare HPO plan");
        assert_eq!(ctx.training_io_profiles.len(), 2);
        assert!(ctx.training_io_profiles.values().all(|profile| {
            profile.is_inline()
                && profile.downgrade_reason == Some(TrainingIoDowngradeReason::UserForced)
        }));
        let footprint =
            hpo_selected_footprint(&plan, &trial_args, &admissions, &ctx.training_io_profiles)
                .expect("inline selected footprint");
        assert_eq!(footprint.ram_bytes, 2 * GIB);
        assert_eq!(footprint.vram_mib, 2048);
    }

    #[test]
    fn dynamic_hpo_child_cannot_outgrow_the_held_reservation() {
        let limit = Footprint {
            ram_bytes: GIB + 128 * MIB,
            vram_mib: 2048,
        };
        let constrained = constrain_dynamic_hpo_admission(admission(1, 1024), Some(&limit))
            .expect("base fits held reservation");
        assert_eq!(constrained.selection_budget_bytes, Some(limit.ram_bytes));
        assert!(
            constrain_dynamic_hpo_admission(admission(2, 1024), Some(&limit))
                .expect_err("larger RAM child must fail closed")
                .contains("above the held")
        );
        assert!(
            constrain_dynamic_hpo_admission(admission(1, 4096), Some(&limit))
                .expect_err("larger VRAM child must fail closed")
                .contains("VRAM")
        );

        let resolver: crate::framework::executor::TrainingIoNodeAdmissionResolver =
            Arc::new(move |_stage, _node_args, _recipe_args| {
                constrain_dynamic_hpo_admission(admission(1, 1024), Some(&limit)).map(Some)
            });
        let temp = tempfile::tempdir().expect("tempdir");
        let mut ctx = ExecCtx::new(PathBuf::from(temp.path()))
            .with_memory_budget(4)
            .with_training_io_selection_budget_bytes(4 * GIB)
            .with_training_io_node_admission_resolver(resolver);
        let _plan = prepare_plan_for_parallel_execution(
            component(serde_json::json!({"batch": 1})),
            &mut ctx,
        )
        .expect("bounded child downgrades inside held reservation");
        let profile = ctx
            .training_io_profiles
            .values()
            .next()
            .expect("declaring child profile");
        assert!(profile.is_inline());
        assert_eq!(profile.sync_base_bytes, GIB);
        assert_eq!(
            profile.downgrade_reason,
            Some(TrainingIoDowngradeReason::BudgetPressure)
        );
    }
}
