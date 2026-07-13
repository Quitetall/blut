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
        /// Path to a recipe file — `.toml` (linear chain), `.star` (Starlark
        /// script, evaluated by the `blut-dsl` helper binary), or `.json` (a
        /// PlanSpec). Omit to list discovered recipes.
        file: Option<std::path::PathBuf>,
        /// Args JSON passed to a `.star` script's `build(args)` (`.json`/`.toml`
        /// recipes carry their own args; `--args` is rejected for those).
        #[arg(long, default_value = "{}")]
        args: String,
        /// LAUNCH the recipe: after it compiles + kind-checks, execute it
        /// end-to-end through the SAME admission-gated, cgroup-contained,
        /// cache-honouring path as `recipe run`. Without `--run` (default) the
        /// DAG is only rendered — nothing executes. Requires a `<file>`.
        #[arg(long, default_value_t = false)]
        run: bool,
        /// Promote this run's outputs to the global cache (only with `--run`).
        #[arg(long, default_value_t = false)]
        shared_cache: bool,
        /// Force-recompute on launch: bypass the stage cache READ so every stage
        /// runs even with a warm entry (only with `--run`). Alias: `--force`.
        #[arg(long = "no-cache", alias = "force", default_value_t = false)]
        no_cache: bool,
        /// Tenant to resolve a `registry://plan@<name>` deploy URI against and,
        /// with `--run`, to own the launched job (ADR 0085/0096).
        #[arg(long, default_value = "default")]
        tenant: String,
        /// Experiment/campaign name for lineage grouping. Defaults to the recipe.
        #[arg(long)]
        experiment: Option<String>,
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
        /// Tenant (`project[/domain]`, ADR 0096) whose namespace this run's
        /// shared cache lives under (default `default` = the flat store). A
        /// `clinical/*` or `restricted` tenant is a sealed clinical namespace.
        #[arg(long, default_value = "default")]
        tenant: String,
        /// Experiment/campaign name for lineage grouping. Defaults to the recipe.
        #[arg(long)]
        experiment: Option<String>,
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
            args,
            run,
            shared_cache,
            no_cache,
            tenant,
            experiment,
        } => {
            use crate::recipes::declarative::{scan_user_recipes, user_recipes_dir};
            match file {
                None => {
                    if run {
                        return Err(anyhow!("--run requires a <file> (a recipe to launch)"));
                    }
                    // F4 discovery: list ~/.config/blut/recipes/{*.toml,*.star,*.json}.
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
                    let tenant = crate::tenant::Tenant::parse(&tenant)
                        .ok_or_else(|| anyhow!("invalid --tenant '{tenant}'"))?;
                    let launch_target = crate::config::launcher::LaunchTarget::Local;
                    // A `registry://plan@<name>` deploy URI (ADR 0085) resolves
                    // to a FROZEN PlanSpec from the local registry and dispatches
                    // that exact graph; otherwise `path` is a recipe file
                    // (dispatch by extension: .toml chain / .star script / .json
                    // PlanSpec) into a runnable plan, then render or launch it.
                    let (name, plan, n) = if let Some(ptr) =
                        crate::registry_db::parse_pointer_uri(&path.to_string_lossy())
                    {
                        let conn = crate::registry_db::open().map_err(|e| anyhow!("{e}"))?;
                        let mut spec =
                            crate::registry_db::resolve_spec(&conn, &tenant.to_string(), ptr)
                                .map_err(|e| anyhow!("{e}"))?;
                        resolve_plan_spec_registry_args(&mut spec, &tenant, launch_target)?;
                        let plan = spec.compile(reg).map_err(|e| anyhow!("{e}"))?;
                        let n = plan.n_nodes();
                        (format!("registry://plan@{ptr}"), plan, n)
                    } else {
                        compile_declared_recipe(reg, &path, &args, &tenant, launch_target)?
                    };
                    if run {
                        // LAUNCH: execute through the same admission-gated /
                        // cgroup-contained / cache-honouring core as `recipe
                        // run`. No RecipeMarker (declared recipes don't resume
                        // by registry name); `Local` placement (clusters target
                        // registry recipes only).
                        println!(
                            "✓ '{name}' compiles + kind-checks ({n} ingredient(s)); launching…"
                        );
                        launch_compiled_plan(
                            &name,
                            plan,
                            None,
                            None,
                            shared_cache,
                            crate::config::launcher::LaunchTarget::Local,
                            None,
                            no_cache,
                            tenant,
                            experiment,
                        )
                        .await?;
                    } else {
                        // Render-only (default): print the runnable DAG, no exec.
                        print!("{}", plan.render_ascii().map_err(|e| anyhow!("{e}"))?);
                        println!("✓ '{name}' compiles + kind-checks ({n} ingredient(s)).");
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
            tenant,
            experiment,
        } => {
            // #3 distributed: parse placement up front so a typo fails the run
            // BEFORE any job dir / state is written (vs deep in the executor).
            let launch_target: crate::config::launcher::LaunchTarget = launcher
                .parse()
                .map_err(|e| anyhow!("invalid --launcher {launcher:?}: {e}"))?;
            // ADR 0096: parse the tenant up front (same fail-fast discipline).
            let tenant = crate::tenant::Tenant::parse(&tenant)
                .ok_or_else(|| anyhow!("invalid --tenant '{tenant}'"))?;
            // M2.1: fail before compilation or job-dir creation when the tenant
            // is unknown to the active quota policy. With no config, only the
            // flat `default` tenant exists and owns 100% of usable RAM.
            crate::config::tenants::TenantQuotaPolicy::load()?
                .fraction_for(&tenant)
                .map_err(|e| anyhow!("{e}"))?;
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
                    tenant,
                    experiment,
                )
                .await?;
            } else {
                let raw: serde_json::Value = serde_json::from_str(&args)
                    .with_context(|| format!("parse --args as JSON: {args}"))?;
                if dry_run {
                    let raw =
                        crate::registry_args::resolve_recipe_args(raw, &tenant, launch_target)
                            .map_err(|e| anyhow!("registry arg resolution: {e}"))?;
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
                    tenant,
                    experiment,
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
    // ADR 0096: the tenant whose namespace this run's shared cache lives under.
    tenant: crate::tenant::Tenant,
    // ADR 0090: explicit experiment/campaign key; recipe name when absent.
    experiment: Option<String>,
) -> Result<String> {
    let source_args = args.clone();
    let args = crate::registry_args::resolve_recipe_args(args, &tenant, launch_target)
        .map_err(|e| anyhow!("registry arg resolution: {e}"))?;
    run_one_recipe_resolved(
        reg,
        name,
        source_args,
        args,
        sweep_fp,
        shared_cache,
        launch_target,
        device_index,
        no_cache,
        tenant,
        experiment,
        None,
    )
    .await
    .map(|(job_id, _, _)| job_id)
}

/// Partition entry point: resolve registry handles exactly once, bind the
/// typed partition key to every compiled node, and return the exact resolved
/// args that the executor persisted for lineage indexing.
#[allow(clippy::too_many_arguments)]
pub(super) async fn run_one_partitioned_recipe(
    reg: &crate::framework::Registry,
    name: &str,
    source_args: serde_json::Value,
    shared_cache: bool,
    launch_target: crate::config::launcher::LaunchTarget,
    device_index: Option<usize>,
    no_cache: bool,
    tenant: crate::tenant::Tenant,
    partition: blut_types::partition::PartitionKey,
) -> Result<(String, serde_json::Value, String)> {
    let resolved_args =
        crate::registry_args::resolve_recipe_args(source_args.clone(), &tenant, launch_target)
            .map_err(|e| anyhow!("registry arg resolution: {e}"))?;
    run_one_recipe_resolved(
        reg,
        name,
        source_args,
        resolved_args.clone(),
        None,
        shared_cache,
        launch_target,
        device_index,
        no_cache,
        tenant,
        None,
        Some(partition),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_one_recipe_resolved(
    reg: &crate::framework::Registry,
    name: &str,
    source_args: serde_json::Value,
    args: serde_json::Value,
    sweep_fp: Option<crate::framework::ContentHash>,
    shared_cache: bool,
    launch_target: crate::config::launcher::LaunchTarget,
    device_index: Option<usize>,
    no_cache: bool,
    tenant: crate::tenant::Tenant,
    experiment: Option<String>,
    partition: Option<blut_types::partition::PartitionKey>,
) -> Result<(String, serde_json::Value, String)> {
    let r = reg
        .find(name)
        .ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
    let mut plan =
        (r.compile_fn)(args.clone()).map_err(|e| anyhow!("recipe compile failed: {e}"))?;
    if let Some(partition) = partition {
        plan = plan.with_partition(partition);
    }
    let persisted_args = plan.recipe_args().clone();
    let input_fingerprint = crate::config::partition::partition_input_fingerprint_with_execution(
        &source_args,
        &persisted_args,
        &plan.execution_fingerprint(),
    );
    // A registry recipe CAN resume by name+args (the RecipeMarker is the resume
    // oracle for `blut plan resume`). Declarative `.toml` launches pass `None`
    // (no registry recipe to re-compile from) — see `launch_compiled_plan`.
    launch_compiled_plan(
        name,
        plan,
        Some(RecipeMarker {
            name: name.to_string(),
            args,
            source_args: Some(source_args),
        }),
        sweep_fp,
        shared_cache,
        launch_target,
        device_index,
        no_cache,
        tenant,
        experiment,
    )
    .await
    .map(|job_id| (job_id, persisted_args, input_fingerprint))
}

/// Compile a declared recipe FILE into a runnable plan, dispatching on its
/// extension (ADR 0078): `.toml` → the linear declarative chain; `.star` → a
/// Starlark script (needs the `dsl` feature) evaluated with `args_json` then
/// compiled through the same `PlanSpec` path; `.json` → a `PlanSpec` read
/// verbatim. All three resolve stage NAMES against the registry (no dynamic
/// code loading) and are fully kind-checked before returning. Returns
/// `(plan_name, plan, n_nodes)`.
fn compile_declared_recipe(
    reg: &crate::framework::Registry,
    path: &std::path::Path,
    args_json: &str,
    tenant: &crate::tenant::Tenant,
    launch_target: crate::config::launcher::LaunchTarget,
) -> Result<(String, crate::framework::plan::CompiledPlan, usize)> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    // "given" = the user passed real args, semantically (not a string compare
    // against "{}", so `--args "{ }"` / `{}\n` aren't false positives). An
    // unparseable value counts as given so the error surfaces on the arm that
    // actually uses it.
    let args_given = match serde_json::from_str::<serde_json::Value>(args_json) {
        Ok(serde_json::Value::Null) => false,
        Ok(serde_json::Value::Object(o)) => !o.is_empty(),
        Ok(_) => true,
        Err(_) => true,
    };
    match ext.as_str() {
        "toml" => {
            if args_given {
                return Err(anyhow!(
                    "--args is only for .star recipes; a .toml recipe carries its own per-stage args"
                ));
            }
            let mut recipe = crate::recipes::declarative::DeclarativeRecipe::load(path)
                .map_err(|e| anyhow!("{e}"))?;
            resolve_declarative_registry_args(&mut recipe, tenant, launch_target)?;
            let n = recipe.stages.len();
            let plan = recipe.compile(reg).map_err(|e| anyhow!("{e}"))?;
            Ok((recipe.name, plan, n))
        }
        "json" => {
            if args_given {
                return Err(anyhow!(
                    "--args is only for .star recipes; a .json PlanSpec is already fully specified"
                ));
            }
            let body = std::fs::read_to_string(path)
                .with_context(|| format!("read PlanSpec {}", path.display()))?;
            let mut spec: crate::framework::plan_spec::PlanSpec = serde_json::from_str(&body)
                .with_context(|| format!("parse PlanSpec {}", path.display()))?;
            resolve_plan_spec_registry_args(&mut spec, tenant, launch_target)?;
            let n = spec.nodes.len();
            let plan = spec.compile(reg).map_err(|e| anyhow!("{e}"))?;
            Ok((spec.name, plan, n))
        }
        "star" => compile_star_recipe(reg, path, args_json, tenant, launch_target),
        other => Err(anyhow!(
            "unsupported recipe extension '.{other}' — expected .toml, .star, or .json"
        )),
    }
}

/// The `.star` arm of [`compile_declared_recipe`] (ADR 0078). Starlark runs
/// OUT OF PROCESS: the engine shells out to the `blut-dsl` binary (which
/// links `starlark`, and hence its `serde_json/arbitrary_precision` feature,
/// away from the engine), captures the emitted `PlanSpec` JSON, then compiles
/// it against the registry through the same path a `.json` recipe uses. The
/// engine binary itself never links Starlark.
fn compile_star_recipe(
    reg: &crate::framework::Registry,
    path: &std::path::Path,
    args_json: &str,
    tenant: &crate::tenant::Tenant,
    launch_target: crate::config::launcher::LaunchTarget,
) -> Result<(String, crate::framework::plan::CompiledPlan, usize)> {
    let source_args: serde_json::Value =
        serde_json::from_str(args_json).with_context(|| "parse --args as JSON")?;
    let args =
        crate::registry_args::resolve_recipe_args(source_args.clone(), tenant, launch_target)
            .map_err(|e| anyhow!("registry arg resolution for Starlark build args: {e}"))?;
    let resolved_args_json =
        serde_json::to_string(&args).context("serialize registry-resolved Starlark build args")?;
    let src =
        std::fs::read_to_string(path).with_context(|| format!("read script {}", path.display()))?;
    let label = path.display().to_string();

    // Run `blut-dsl <script> --args <json>` and capture its PlanSpec JSON.
    let bin = blut_dsl_binary();
    let out = std::process::Command::new(&bin)
        .arg(path)
        .arg("--args")
        .arg(&resolved_args_json)
        .output()
        .with_context(|| {
            format!(
                "run the Starlark evaluator `{}` for {label} — is `blut-dsl` installed? \
                 (set $BLUT_DSL_BIN to its path, or author the recipe as `.json`)",
                bin.display()
            )
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow!("blut-dsl failed on {label}: {}", stderr.trim()));
    }
    let mut spec: crate::framework::plan_spec::PlanSpec = serde_json::from_slice(&out.stdout)
        .with_context(|| format!("parse the PlanSpec blut-dsl emitted for {label}"))?;
    resolve_plan_spec_registry_args(&mut spec, tenant, launch_target)?;

    let n = spec.nodes.len();
    let fingerprint = spec.provenance_fingerprint(&src, &args);
    let plan = spec.compile(reg).map_err(|e| anyhow!("{e}"))?;
    // Stamp richer provenance than PlanSpec::compile's default blob: the
    // script path + content-hash fingerprint + the args it was built with, so
    // lineage records exactly what produced this plan.
    let plan = plan.override_recipe_args(serde_json::json!({
        "starlark": true,
        "script_path": label,
        "plan_fingerprint": fingerprint.to_hex(),
        "version": crate::framework::plan_spec::PLAN_SPEC_VERSION,
        "args": args,
        "source_args": source_args,
    }));
    Ok((spec.name, plan, n))
}

/// Resolve every stage argument in a PlanSpec, including nested `map_output`
/// templates, before any registered stage receives typed args.
fn resolve_plan_spec_registry_args(
    spec: &mut crate::framework::plan_spec::PlanSpec,
    tenant: &crate::tenant::Tenant,
    launch_target: crate::config::launcher::LaunchTarget,
) -> Result<()> {
    let mut resolve = |args| {
        crate::registry_args::resolve_recipe_args(args, tenant, launch_target)
            .map_err(|e| anyhow!("{e}"))
    };
    resolve_plan_spec_args_with(spec, &mut resolve)
}

fn resolve_plan_spec_args_with(
    spec: &mut crate::framework::plan_spec::PlanSpec,
    resolve: &mut impl FnMut(serde_json::Value) -> Result<serde_json::Value>,
) -> Result<()> {
    for node in &mut spec.nodes {
        node.args = resolve(std::mem::take(&mut node.args))
            .map_err(|e| anyhow!("registry arg resolution for stage '{}': {e}", node.stage))?;
    }
    for expansion in &mut spec.expansions {
        resolve_plan_spec_args_with(&mut expansion.template, resolve)?;
    }
    Ok(())
}

fn resolve_declarative_registry_args(
    recipe: &mut crate::recipes::declarative::DeclarativeRecipe,
    tenant: &crate::tenant::Tenant,
    launch_target: crate::config::launcher::LaunchTarget,
) -> Result<()> {
    let mut resolve = |args| {
        crate::registry_args::resolve_recipe_args(args, tenant, launch_target)
            .map_err(|e| anyhow!("{e}"))
    };
    resolve_declarative_args_with(recipe, &mut resolve)
}

fn resolve_declarative_args_with(
    recipe: &mut crate::recipes::declarative::DeclarativeRecipe,
    resolve: &mut impl FnMut(serde_json::Value) -> Result<serde_json::Value>,
) -> Result<()> {
    for stage in &mut recipe.stages {
        stage.args = resolve(std::mem::take(&mut stage.args))
            .map_err(|e| anyhow!("registry arg resolution for stage '{}': {e}", stage.stage))?;
    }
    Ok(())
}

/// Locate the `blut-dsl` evaluator binary: `$BLUT_DSL_BIN` if set, else a
/// sibling of the current executable (the usual install layout), else bare
/// `blut-dsl` resolved on `$PATH`.
fn blut_dsl_binary() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("BLUT_DSL_BIN") {
        return std::path::PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let sibling = dir.join("blut-dsl");
        if sibling.exists() {
            return sibling;
        }
    }
    std::path::PathBuf::from("blut-dsl")
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
    tenant: crate::tenant::Tenant,
    experiment: Option<String>,
) -> Result<String> {
    use crate::framework::ExecCtx;

    // Validate quota configuration and resolve the tenant BEFORE creating any
    // job state. This function is also called by declarative and sweep paths,
    // so it is the authoritative enforcement seam even when a caller bypasses
    // the `recipe run` command arm's earlier UX-oriented check.
    let tenant_admission = crate::broker::tenant_quota::TenantAdmission::prepare(tenant.clone())
        .map_err(|e| anyhow!("tenant admission: {e}"))?;

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
    crate::jobs::write_tenant(&job_id, &tenant)
        .with_context(|| format!("persist tenant for {job_id}"))?;
    crate::jobs::write_experiment(&job_id, experiment.as_deref().unwrap_or(name))
        .with_context(|| format!("persist experiment for {job_id}"))?;
    let mut ctx = ExecCtx::new(job_dir.clone());
    ctx = ctx.with_tenant(tenant.clone());
    // Phase 5: size the executor's memory admission to box-fit (MemTotal −
    // floor) so the parallel executor can't stack concurrent stages past the
    // box. Sequential runs one stage at a time, so this is a no-op there.
    if let Some(budget) = tenant_admission
        .executor_budget_gib(crate::broker::admission::DEFAULT_FLOOR_GIB)
        .map_err(|e| anyhow!("tenant admission: {e}"))?
    {
        // The executor semaphore is the tenant sub-envelope too. This is
        // load-bearing for HPO/wide DAGs: one job may run many concurrent
        // stages, but their summed declared RAM cannot exceed its share.
        ctx = ctx.with_memory_budget(budget);
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
    // ADR 0087: size the GPU-aware scheduler. Prefer the live local inventory
    // (VRAM-aware placement) when it covers the pool; otherwise (e.g. a Slurm
    // submit node whose local GPU count is below the allocated `--gpus`) fall
    // back to `gpu_pool` homogeneous cells so remote concurrency is preserved.
    let gpu_inv = crate::broker::gpu::GpuInventory::probe();
    let gpu_inv = if gpu_inv.len() >= gpu_pool.max(1) {
        gpu_inv
    } else {
        crate::broker::gpu::GpuInventory::homogeneous(gpu_pool.max(1), 0)
    };
    ctx = ctx.with_gpu_scheduler(crate::broker::gpu::GpuScheduler::new(gpu_inv));
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
            // ADR 0096: namespace the shared cache by tenant (disjoint roots per
            // tenant; the `default` tenant is the flat store, byte-identical).
            let cache_handle = (*ctx.cache)
                .clone()
                .with_global(global)
                .with_tenant(&tenant);
            if let Some(g) = &cache_handle.global {
                std::fs::create_dir_all(g)
                    .with_context(|| format!("create global cache dir {}", g.display()))?;
            }
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
    let _tenant_reservation =
        match tenant_admission.reserve(&footprint, crate::broker::admission::DEFAULT_FLOOR_GIB) {
            Ok(reservation) => reservation,
            Err(reason) => {
                crate::python_kill::unbind_current_job();
                if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
                    tracing::warn!("write Failed state for {job_id}: {se}");
                }
                return Err(anyhow!("recipe '{name}' admission refused: {reason}"));
            }
        };

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
    // ADR 0096: the tenant namespace for every combo's shared cache.
    tenant: crate::tenant::Tenant,
    // ADR 0090: one campaign key shared by every sweep combo.
    experiment: Option<String>,
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
            tenant.clone(),
            experiment.clone(),
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
                        args: _,
                        run,
                        shared_cache,
                        no_cache,
                        tenant: _,
                        experiment: _,
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

    #[test]
    fn declare_args_flag_parses() {
        match Cli::try_parse_from([
            "blut",
            "recipe",
            "declare",
            "r.star",
            "--args",
            r#"{"n":1}"#,
        ])
        .expect("parse")
        .command
        {
            Some(Command::Recipe {
                cmd: RecipeCommand::Declare { args, .. },
            }) => assert_eq!(args, r#"{"n":1}"#),
            other => panic!("expected declare, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod declare_dispatch_tests {
    //! `compile_declared_recipe` routes by extension and validates inputs
    //! (ADR 0078). These need no real stages — they exercise routing + the
    //! `--args` guards against an EMPTY registry (so a resolvable graph fails
    //! at stage lookup, which is the expected error).
    use super::{
        compile_declared_recipe, resolve_declarative_args_with, resolve_plan_spec_args_with,
    };
    use crate::framework::Registry;

    fn compile(
        reg: &Registry,
        path: &std::path::Path,
        args: &str,
    ) -> super::Result<(String, crate::framework::plan::CompiledPlan, usize)> {
        compile_declared_recipe(
            reg,
            path,
            args,
            &crate::tenant::Tenant::default(),
            crate::config::launcher::LaunchTarget::Local,
        )
    }

    fn write(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    // The Ok type holds a CompiledPlan (no Debug), so `unwrap_err` can't be
    // used — pull the error out by hand.
    fn expect_err(
        r: super::Result<(String, crate::framework::plan::CompiledPlan, usize)>,
    ) -> anyhow::Error {
        match r {
            Ok(_) => panic!("expected an Err, got Ok"),
            Err(e) => e,
        }
    }

    #[test]
    fn json_planspec_routes_and_resolves_against_registry() {
        let td = tempfile::tempdir().unwrap();
        let p = write(
            td.path(),
            "s.json",
            r#"{"name":"js","nodes":[{"stage":"nope"}],"edges":[]}"#,
        );
        // Routes to the .json arm, parses, then fails at unknown stage.
        let err = expect_err(compile(&Registry::new(), &p, "{}"));
        assert!(
            err.to_string().contains("nope"),
            "names the unknown stage: {err}"
        );
    }

    #[test]
    fn args_rejected_for_toml_and_json() {
        let td = tempfile::tempdir().unwrap();
        let toml = write(td.path(), "r.toml", "name=\"x\"\n[[stages]]\nstage=\"a\"\n");
        let json = write(td.path(), "r.json", r#"{"name":"x","nodes":[],"edges":[]}"#);
        for p in [toml, json] {
            let err = expect_err(compile(&Registry::new(), &p, r#"{"n":1}"#));
            assert!(
                err.to_string().contains("--args is only for .star"),
                "{err}"
            );
        }
    }

    #[test]
    fn unsupported_extension_is_rejected() {
        let td = tempfile::tempdir().unwrap();
        let p = write(td.path(), "r.yaml", "nope");
        let err = expect_err(compile(&Registry::new(), &p, "{}"));
        assert!(
            err.to_string().contains("unsupported recipe extension"),
            "{err}"
        );
    }

    #[test]
    fn declarative_and_nested_planspec_args_are_resolved_before_compile() {
        let mut recipe = crate::recipes::declarative::DeclarativeRecipe::parse(
            "name='x'\n[[stages]]\nstage='a'\nargs={data='dataset://train@v1'}\n",
            "x.toml",
        )
        .unwrap();
        let mut resolve = |value: serde_json::Value| {
            Ok(replace_test_handle(
                value,
                "dataset://train@v1",
                "/verified/train.jsonl",
            ))
        };
        resolve_declarative_args_with(&mut recipe, &mut resolve).unwrap();
        assert_eq!(recipe.stages[0].args["data"], "/verified/train.jsonl");

        let mut spec: crate::framework::plan_spec::PlanSpec =
            serde_json::from_value(serde_json::json!({
                "name": "root",
                "nodes": [{"stage":"a", "args":{"model":"model://enc@staging"}}],
                "edges": [],
                "expansions": [{
                    "parent": 0,
                    "template": {
                        "name":"nested",
                        "nodes":[{"stage":"b", "args":{"data":"dataset://train@v1"}}],
                        "edges": []
                    }
                }]
            }))
            .unwrap();
        let mut resolve = |value: serde_json::Value| {
            let value = replace_test_handle(value, "model://enc@staging", "model-hash");
            Ok(replace_test_handle(
                value,
                "dataset://train@v1",
                "/verified/train.jsonl",
            ))
        };
        resolve_plan_spec_args_with(&mut spec, &mut resolve).unwrap();
        assert_eq!(spec.nodes[0].args["model"], "model-hash");
        assert_eq!(
            spec.expansions[0].template.nodes[0].args["data"],
            "/verified/train.jsonl"
        );
    }

    fn replace_test_handle(value: serde_json::Value, from: &str, to: &str) -> serde_json::Value {
        match value {
            serde_json::Value::String(value) if value == from => to.into(),
            serde_json::Value::Array(values) => values
                .into_iter()
                .map(|value| replace_test_handle(value, from, to))
                .collect::<Vec<_>>()
                .into(),
            serde_json::Value::Object(values) => values
                .into_iter()
                .map(|(key, value)| (key, replace_test_handle(value, from, to)))
                .collect::<serde_json::Map<_, _>>()
                .into(),
            value => value,
        }
    }

    #[test]
    fn empty_args_variants_do_not_trip_the_toml_guard() {
        // `{ }`, `{}\n`, and `null` are all "no args" — they must NOT be
        // mistaken for real args on a .toml recipe (semantic, not string, cmp).
        let td = tempfile::tempdir().unwrap();
        let toml = write(td.path(), "r.toml", "name=\"x\"\n[[stages]]\nstage=\"a\"\n");
        for a in ["{ }", "{}\n", "null"] {
            let err = expect_err(compile(&Registry::new(), &toml, a));
            // Reaches stage resolution (unknown stage 'a'), NOT the --args guard.
            assert!(
                !err.to_string().contains("--args is only for"),
                "empty args `{a}` wrongly tripped the guard: {err}"
            );
        }
    }

    #[test]
    fn star_route_reports_missing_evaluator_clearly() {
        // `.star` is evaluated OUT OF PROCESS by `blut-dsl` (ADR 0078). When
        // that binary can't be found, the error must be actionable (name the
        // binary + the .json escape hatch), not a raw ENOENT. Point
        // BLUT_DSL_BIN at a path that certainly doesn't exist. (This engine
        // test binary no longer links starlark, so there is no pre-main thread
        // and this scoped env set is sound; TEST_ENV_LOCK serializes it.)
        let _g = crate::TEST_ENV_LOCK.lock().unwrap();
        let td = tempfile::tempdir().unwrap();
        let p = write(td.path(), "r.star", "def build(args):\n    add(\"x\")\n");
        let prev = std::env::var("BLUT_DSL_BIN").ok();
        unsafe {
            std::env::set_var("BLUT_DSL_BIN", td.path().join("no-such-blut-dsl"));
        }
        let err = expect_err(compile(&Registry::new(), &p, "{}"));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BLUT_DSL_BIN", v),
                None => std::env::remove_var("BLUT_DSL_BIN"),
            }
        }
        let msg = err.to_string();
        assert!(
            msg.contains("blut-dsl") && msg.contains(".json"),
            "actionable evaluator-missing error: {msg}"
        );
    }
}
