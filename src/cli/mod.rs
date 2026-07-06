// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! The `blut` CLI — clap surface (`Cli` / `Command` + subcommand enums),
//! the `run()` dispatcher, and the small shared helpers (`emit_json`,
//! `truncate_for_col`, tracing/cancel setup). Subcommand implementations
//! live in the per-subsystem submodules declared at the bottom of this
//! file (recipe, hpo, lineage, runs, errors, partition, p2p, stale, gate).
//!
//! Training runs acquire the cross-process scheduler lock for their
//! duration; inference paths (lamu-mcp, lamu-api) refuse during that
//! window.

use std::path::PathBuf;
use std::time::Duration;

use crate::scheduler_lock::{self, LockKind};
use crate::{
    jobs::{self, JobState},
    paths,
};
use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};

/// Top-level `about` line. The default build is TUI-on, so bare `blut` opens the
/// cockpit; a `--no-default-features` build is CLI-only (so the banner reflects
/// which build this is).
#[cfg(feature = "tui")]
const CLI_ABOUT: &str = "BLUT — typed-DAG orchestrator for local ML training. Bare `blut` opens the \
     interactive cockpit; subcommands: recipe, jobs, log, cancel, plan, cache, \
     footprint, partition, schedule, sensor, policy, tui.";
#[cfg(not(feature = "tui"))]
const CLI_ABOUT: &str = "BLUT — typed-DAG orchestrator for local ML training. Run a subcommand: \
     recipe, jobs, log, cancel, plan, cache, footprint, partition, schedule, \
     sensor, policy.";

#[derive(Parser, Debug)]
#[command(
    name = "blut",
    // Stamp the build-time commit into `--version` (e.g. `0.1.0+a1b2c3d4e5f6`,
    // or `…-dirty` for an uncommitted build) so the running binary's provenance
    // is visible at a glance; build.rs composes BLUT_VERSION. Complements the
    // runtime `warn_if_stale_binary` check.
    version = env!("BLUT_VERSION"),
    about = CLI_ABOUT
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List training jobs (running + completed).
    Jobs {
        /// Emit the job list as a JSON array (for scripts/agents).
        #[arg(long)]
        json: bool,
    },
    /// SIGTERM a running training job.
    Cancel {
        /// Job id (or unique prefix).
        id: String,
        /// Grace period before escalating to SIGKILL.
        #[arg(long, default_value = "10s", value_parser = parse_duration)]
        grace: Duration,
    },
    /// Print rendered training log for a job.
    Log {
        /// Job id (or unique prefix).
        id: String,
        /// How many lines to tail. 0 = all.
        #[arg(long, default_value_t = 0)]
        tail: usize,
        /// Emit the raw status stream as JSON lines (for scripts/agents).
        #[arg(long)]
        json: bool,
    },
    /// Compare two runs: spec/args provenance diff + one-line outcomes.
    Runs {
        #[command(subcommand)]
        cmd: RunsCommand,
    },
    /// Query the lineage index: per-job view, reproducibility trace, reindex.
    Lineage {
        #[command(subcommand)]
        cmd: LineageCommand,
    },
    /// Hyperparameter optimization: adaptive search over a recipe (v0.20).
    Hpo {
        #[command(subcommand)]
        cmd: HpoCommand,
    },
    /// Render a job's DAG: per-node status + edges (the graph backend, v0.20).
    Dag {
        /// Job id (defaults to the most recent job).
        job: Option<String>,
        /// Emit JSON (the full GraphSnapshot) instead of the text table.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Compare two runs: provenance + a final-metric panel + GPU saturation
    /// (reads the queryable metric store). `blut compare <jobA> <jobB>`.
    Compare {
        /// First job id (prefix ok).
        a: String,
        /// Second job id.
        b: String,
    },
    /// One run's results from the metric store (ADR 0071 A3): best val_r +
    /// trajectory + per-band PRD + ckpt path. `blut results <job> [--json]` —
    /// replaces grepping BLUT_METRIC out of raw logs + `ls -t`-hunting a CSV.
    Results {
        /// Job id (prefix ok).
        job: String,
        /// Emit machine-readable JSON instead of a human summary.
        #[arg(long)]
        json: bool,
        /// The headline metric to report best/trajectory for (default `val_r`).
        #[arg(long, default_value = "val_r")]
        metric: String,
        /// Force "best = max" (override the name heuristic for a non-standard
        /// metric). Mutually exclusive with --minimize.
        #[arg(long, conflicts_with = "minimize")]
        maximize: bool,
        /// Force "best = min" (override the name heuristic — e.g. a custom loss
        /// not matching the prd/loss/err naming convention).
        #[arg(long)]
        minimize: bool,
    },
    /// Error-domain catalog + per-job failure breakdown (ADR 0072 A4).
    Errors {
        #[command(subcommand)]
        cmd: ErrorsCommand,
    },
    /// Declared, persistent partition key-space over a recipe + per-cell
    /// backfill (Dagster-class partitions, v0.20 Phase G).
    Partition {
        #[command(subcommand)]
        cmd: PartitionCommand,
    },
    /// Inspect materialized artifacts via their sidecars.
    Artifact {
        #[command(subcommand)]
        cmd: ArtifactCommand,
    },
    /// Manage recipe schedules (systemd --user timers; no daemon).
    Schedule {
        #[command(subcommand)]
        cmd: ScheduleCommand,
    },
    /// Manage the datasets registry.
    Data {
        #[command(subcommand)]
        cmd: DataCommand,
    },
    /// Inspect or modify the auto-trigger policy.
    Policy {
        #[command(subcommand)]
        cmd: PolicyCommand,
    },
    /// Recipe catalog. List / show / run named recipes.
    Recipe {
        #[command(subcommand)]
        cmd: RecipeCommand,
    },
    /// Plan-level operations: resume a partially-run job.
    Plan {
        #[command(subcommand)]
        cmd: PlanCommand,
    },
    /// Inspect / prune the BLUT cache.
    Cache {
        #[command(subcommand)]
        cmd: CacheCommand,
    },
    /// Inspect / heal the resource-footprint calibration store (ADR 0046).
    Footprint {
        #[command(subcommand)]
        cmd: FootprintCommand,
    },
    /// Named sensors (G2): observe external state (GPU lock, auto-train
    /// policy gates) and report a typed READY/SKIP/WAIT outcome.
    Sensor {
        #[command(subcommand)]
        cmd: SensorCommand,
    },
    /// P2P distributed compute: manage identity keys + peers, run as a
    /// coordinator or a worker peer, and dispatch a dispatchable stage to a
    /// peer over QUIC (behind the off-by-default `p2p` feature).
    #[cfg(feature = "p2p")]
    P2p {
        #[command(subcommand)]
        cmd: P2pCommand,
    },
    /// Open the canonical interactive training cockpit (ratatui). The
    /// single, complete cockpit: recipe launcher + live jobs/log/system
    /// panels + run history / leaderboard / compare / checkpoints /
    /// presets / live-metrics / reset views. Keys: ↑↓ select, Enter log,
    /// c cancel, R recipe picker, J/L/Y/H/B/K/P/M/X switch views, q quit.
    ///
    /// Behind the `tui` feature — DEFAULT-ON since 1.0 (bare `blut` opens
    /// the cockpit); `--no-default-features` builds a lean CLI-only binary.
    #[cfg(feature = "tui")]
    Tui {
        /// Headless self-check: build the cockpit + render every view to a test
        /// backend, exit 0 if all draw non-blank (no raw mode). For CI / smoke.
        #[arg(long, default_value_t = false)]
        check: bool,
    },
}

#[derive(Subcommand, Debug)]
enum PlanCommand {
    /// Re-run a job's plan. Cached stages (under the same job-local
    /// cache dir) hit immediately, so this picks up from the last
    /// failed/killed stage with no manual surgery.
    Resume {
        /// Job id (or unique prefix).
        id: String,
        /// Also consult the global cache (--shared-cache semantics).
        #[arg(long, default_value_t = false)]
        shared_cache: bool,
    },
    /// Compile a recipe + its args into a Plan and print the ASCII
    /// DAG render. Does NOT execute. Useful for previewing a
    /// recipe's shape before committing to a run.
    Inspect {
        /// Recipe name (as listed by `recipe list`).
        name: String,
        /// Recipe args as inline JSON.
        #[arg(long)]
        args: String,
    },
}

#[derive(Subcommand, Debug)]
enum CacheCommand {
    /// Show the global cache path + current size.
    Show,
    /// LRU-prune the global cache to fit under `max_gb` (or the
    /// `LAMU_CACHE_MAX_GB` env var, or 50 GiB by default).
    Prune {
        /// Cap in GiB. Overrides `LAMU_CACHE_MAX_GB`.
        #[arg(long)]
        max_gb: Option<f64>,
    },
    /// Per-ingredient cache hit/miss tally for a job (from its status.jsonl).
    Stats {
        /// Job id (or unique prefix).
        id: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum SensorCommand {
    /// List the named sensors + their current outcome.
    List {
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Evaluate ONE named sensor and print its outcome (exit 0 = READY,
    /// 1 = SKIP/WAIT) — usable as a cron/daemon gate.
    Eval {
        /// Sensor name (see `blut sensor list`).
        name: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum FootprintCommand {
    /// List the calibration store: per-key RAM, source (Default/Measured/
    /// OomCorrected), and sample count.
    List {
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Forget a stale calibration entry so admission stops resolving it (the
    /// SANCTIONED, audited reset for an `OomCorrected` bound that no longer
    /// reflects reality — e.g. after a memory fix dropped the true peak below
    /// the recorded OOM cap, which the monotone rank can never demote). Pass an
    /// exact `<key>` (e.g. `train_model|3|16|2|w`) OR `--recipe <name>`
    /// to clear every key for a recipe. Never-OOM holds: admission then uses the
    /// conservative Default + the cgroup cap still bounds the next run.
    Forget {
        /// Exact flat key to remove (omit when using --recipe).
        key: Option<String>,
        /// Remove ALL keys for this recipe prefix instead of one exact key.
        #[arg(long)]
        recipe: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum ScheduleCommand {
    /// List installed blut recipe timers.
    List,
    /// Install (or replace) a timer that runs a recipe on a schedule.
    Install {
        /// Recipe name (as listed by `recipe list`).
        recipe: String,
        /// systemd `OnCalendar` expression, e.g. "daily", "Mon *-*-* 02:00:00".
        /// Omit to use the recipe's built-in `SCHEDULE` (if it declares one).
        #[arg(long)]
        calendar: Option<String>,
        /// Recipe args as inline JSON. Defaults to `{}`.
        #[arg(long, default_value = "{}")]
        args: String,
    },
    /// Remove a recipe's timer + service.
    Uninstall {
        /// Recipe name.
        recipe: String,
    },
}

#[derive(Subcommand, Debug)]
enum ArtifactCommand {
    /// List a job's output artifacts (kind, hash, stage).
    Ls {
        /// Job id (or unique prefix).
        id: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Print the full sidecar of an artifact by content-hash prefix
    /// (searches all jobs).
    Inspect {
        /// Content-hash prefix (>= 6 hex chars recommended).
        hash: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum DataCommand {
    /// List registered datasets, newest first.
    List,
    /// Register a JSONL file under a name.
    Add {
        /// Registry name (must match [A-Za-z0-9_.-]+).
        name: String,
        /// Path to a JSONL file.
        path: PathBuf,
        /// Free-form kind tag stored in the record.
        #[arg(long, default_value = "jsonl")]
        kind: String,
    },
    /// Remove a registered dataset (deletes the registry row only,
    /// not the JSONL file on disk).
    Rm { name: String },
    /// Print metadata for one dataset as JSON.
    Show { name: String },
}

#[derive(Subcommand, Debug)]
enum PolicyCommand {
    /// Print the current policy (TOML).
    Show,
    /// Set enabled=true and write the policy file. Prints a
    /// suggested cron line on success.
    Enable,
    /// Set enabled=false. The policy file is preserved so
    /// thresholds + cooldowns aren't lost on re-enable.
    Disable,
}

fn parse_duration(s: &str) -> Result<Duration, String> {
    humantime::parse_duration(s).map_err(|e| format!("{e}"))
}

/// The uniform `--json` output tail every subcommand shares: pretty-print
/// a serializable value to stdout. Serializing these in-memory values can
/// only fail on pathological data (e.g. non-string map keys) — surface
/// that as a CLI error rather than panicking or silently printing nothing.
pub(super) fn emit_json<T: serde::Serialize>(value: &T) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(value).map_err(|e| anyhow!("serialize --json output: {e}"))?
    );
    Ok(())
}

/// BLUT CLI entrypoint. The recipe catalog is supplied by the caller as
/// a composed [`Registry`] (the binary — in a cookbook crate — registers
/// the cookbooks it ships and passes them here). This is the lib seam
/// that lets blut-core stay domain-agnostic: a bare blut engine binary
/// would pass an empty registry; the cookbook binaries pass theirs.
pub async fn run(reg: crate::framework::Registry) -> Result<()> {
    init_tracing();
    warn_if_stale_binary();
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Command::Jobs { json }) => run_jobs(json),
        Some(Command::Cancel { id, grace }) => run_cancel(&id, grace).await,
        Some(Command::Log { id, tail, json }) => run_log(&id, tail, json),
        Some(Command::Runs { cmd }) => run_runs_cmd(cmd),
        Some(Command::Lineage { cmd }) => run_lineage_cmd(cmd),
        Some(Command::Hpo { cmd }) => run_hpo(&reg, cmd).await,
        Some(Command::Dag { job, json }) => run_dag(job, json),
        Some(Command::Compare { a, b }) => run_compare(&a, &b),
        Some(Command::Results {
            job,
            json,
            metric,
            maximize,
            minimize,
        }) => {
            // Explicit flags override the name heuristic; clap's conflicts_with
            // guarantees at most one is set.
            let force = if maximize {
                Some(true)
            } else if minimize {
                Some(false)
            } else {
                None
            };
            run_results(&job, json, &metric, force)
        }
        Some(Command::Errors { cmd }) => run_errors(&reg, cmd),
        Some(Command::Partition { cmd }) => run_partition(&reg, cmd).await,
        Some(Command::Artifact { cmd }) => run_artifact_cmd(cmd),
        Some(Command::Schedule { cmd }) => run_schedule_cmd(&reg, cmd),
        Some(Command::Data { cmd }) => run_data(cmd),
        Some(Command::Policy { cmd }) => run_policy(cmd),
        Some(Command::Recipe { cmd }) => run_recipe(&reg, cmd).await,
        Some(Command::Plan { cmd }) => run_plan_cmd(&reg, cmd).await,
        Some(Command::Cache { cmd }) => run_cache_cmd(cmd),
        Some(Command::Footprint { cmd }) => run_footprint_cmd(cmd),
        Some(Command::Sensor { cmd }) => run_sensor_cmd(cmd),
        #[cfg(feature = "p2p")]
        Some(Command::P2p { cmd }) => run_p2p_cmd(reg, cmd).await,
        #[cfg(feature = "tui")]
        Some(Command::Tui { check }) => {
            if check {
                crate::tui::check(reg)
            } else {
                crate::tui::run(reg).await
            }
        }
        // Bare `blut`: the default (TUI-on) build opens the interactive cockpit.
        // A `--no-default-features` (CLI-only) build has no interactive mode —
        // print help so the user sees the subcommands.
        #[cfg(feature = "tui")]
        None => crate::tui::run(reg).await,
        #[cfg(not(feature = "tui"))]
        None => {
            use clap::CommandFactory;
            Cli::command().print_help().ok();
            println!(
                "\n(this is a CLI-only build — run a subcommand above. The interactive \
                 cockpit ships in the default build; rebuild without \
                 `--no-default-features` to get it.)"
            );
            Ok(())
        }
    };
    // ADR 0072 A2: the command dispatch above is the CLI's single top-level
    // error boundary — every subcommand's Result funnels through here before
    // the caller's own `main()` formats it for the user. A `StageFailure`
    // (ADR 0072) buried in the chain carries fields (`origin`/`course`/
    // `recipe`/`ingredient`) that its own `Display` impl does NOT print (only
    // severity/code/stage/context/message do) — surface them here so they're
    // never silently lost. No-op (and silent) when the chain carries no
    // `StageFailure`, e.g. a plain arg-parse error.
    if let Err(ref e) = result {
        print_stage_failure_detail(e);
    }
    result
}

/// See the call in [`run`]. Mirrors the terseness of the ADR-0071
/// advisory-warning print (`⚠ training OK — completed with N advisory
/// warning(s)...` below `run_recipe`): a short header line, then one
/// `    · field: value` bullet per populated field. Fields left `None`
/// (e.g. a `StageFailure` with no `ingredient` set) are simply omitted, not
/// printed as empty.
fn print_stage_failure_detail(err: &anyhow::Error) {
    let Some(sf) = err
        .chain()
        .find_map(|cause| cause.downcast_ref::<crate::framework::StageFailure>())
    else {
        return;
    };
    eprintln!("  [{}] {}:", sf.severity, sf.code);
    eprintln!("    · origin: {:?}", sf.origin);
    if let Some(course) = &sf.course {
        eprintln!("    · course: {course}");
    }
    if let Some(recipe) = &sf.recipe {
        eprintln!("    · recipe: {recipe}");
    }
    if let Some(stage) = &sf.stage {
        eprintln!("    · stage: {stage}");
    }
    if let Some(ingredient) = &sf.ingredient {
        eprintln!("    · ingredient: {ingredient}");
    }
}

#[cfg(test)]
mod chain_preservation_tests {
    //! ADR 0072 A2 regression pin. The three `run_plan_cmd`/`run_hpo`/
    //! `launch_compiled_plan` sites used to do
    //! `Err(anyhow!("plan execution failed: {e}"))` — Display-interpolating
    //! `e` into a FRESH `anyhow!()` erases `e` as the
    //! new error's `source()`, so nothing downstream can downcast back to the
    //! `StageFailure` a cookbook stage attached. The fix wraps with
    //! `anyhow::Error::from(e).context(...)` instead, which preserves `e` as
    //! the source. Reconstruct the exact chain the real code produces —
    //! `StageFailure::into_error` → `StageError::Backend` →
    //! `PlanError::StageFailed` → the cli.rs `.context()` wrap — and assert
    //! the `StageFailure`, with every field, survives to the top.
    use crate::framework::error::{PlanError, StageError};
    use crate::framework::{FaultOrigin, Severity, StageFailure};

    fn sample_stage_failure() -> StageFailure {
        StageFailure {
            course: Some("train".to_string()),
            recipe: Some("train_joint".to_string()),
            ingredient: Some("encoder".to_string()),
            ..StageFailure::new("E_ROUNDTRIP", "eagle")
                .severity(Severity::Critical)
                .origin(FaultOrigin::External)
                .stage("eagle_decode")
                .context("ch", "4")
        }
    }

    /// Build the exact chain `run_plan_cmd`/`run_recipe` produce on a stage
    /// failure, ending with the (now-fixed) cli.rs wrap.
    fn wrapped_chain(sf: StageFailure) -> anyhow::Error {
        let backend_err = sf.into_error("decode(encode(x)) != x: first diff at sample 1847");
        let stage_err = StageError::Backend(backend_err);
        let plan_err = PlanError::StageFailed {
            idx: 2,
            stage: "eagle_decode".to_string(),
            source: stage_err,
        };
        // The cli.rs fix (was: `anyhow!("plan execution failed: {e}")`).
        anyhow::Error::from(plan_err).context("plan execution failed")
    }

    #[test]
    fn stage_failure_survives_the_cli_context_wrap() {
        let original = sample_stage_failure();
        let final_err = wrapped_chain(original.clone());

        // Top-level message still reads as before (context wrap didn't
        // regress the user-facing text).
        assert_eq!(final_err.to_string(), "plan execution failed");

        // The regression pin: `e` must still be downcastable out of the
        // chain — this is exactly what `print_stage_failure_detail` and
        // any future `blut errors show`-style tooling rely on.
        let found = final_err
            .chain()
            .find_map(|cause| cause.downcast_ref::<StageFailure>())
            .expect("StageFailure must survive the .context() wrap — chain-preservation regressed");

        assert_eq!(found.code, original.code);
        assert_eq!(found.domain, original.domain);
        assert_eq!(found.severity, original.severity);
        assert_eq!(found.origin, original.origin);
        assert_eq!(found.course, original.course);
        assert_eq!(found.recipe, original.recipe);
        assert_eq!(found.ingredient, original.ingredient);
        assert_eq!(found.stage, original.stage);
        assert_eq!(found.context, original.context);
    }

    #[test]
    fn print_stage_failure_detail_finds_it_via_the_same_chain_walk() {
        // Exercises the actual production helper (not a reimplementation) —
        // it must not panic, and its internal `.chain().find_map(...)` must
        // resolve to `Some` for this chain (verified indirectly: calling it
        // is safe and it's a pure eprintln sink with no other observable
        // side effect to assert on here).
        let final_err = wrapped_chain(sample_stage_failure());
        super::print_stage_failure_detail(&final_err);
    }
}

/// Marker file written next to `args.json` so `plan resume` can
/// look up which recipe to re-compile. Kept distinct from
/// `args.json` (which holds the recipe-args payload only) so the
/// shape stays simple: one record per concern.
#[derive(serde::Serialize, serde::Deserialize)]
struct RecipeMarker {
    name: String,
    args: serde_json::Value,
}

impl RecipeMarker {
    fn write_to(&self, job_dir: &std::path::Path) -> Result<()> {
        let path = job_dir.join("recipe.json");
        let body = serde_json::to_vec_pretty(self).context("serialize recipe marker")?;
        std::fs::write(&path, body)
            .with_context(|| format!("write recipe marker {}", path.display()))?;
        Ok(())
    }
    fn read_from(job_dir: &std::path::Path) -> Result<Self> {
        let path = job_dir.join("recipe.json");
        let body = std::fs::read(&path)
            .with_context(|| format!("read recipe marker {}", path.display()))?;
        serde_json::from_slice(&body).context("parse recipe marker")
    }
}

async fn run_plan_cmd(reg: &crate::framework::Registry, cmd: PlanCommand) -> Result<()> {
    use crate::framework::{CacheHandle, ExecCtx};
    // Recipes resolve via the caller-supplied cookbook registry.
    match cmd {
        PlanCommand::Resume { id, shared_cache } => {
            let job_id = crate::jobs::resolve_job_id(&id).with_context(|| {
                format!("resolve job id '{id}' (ambiguous prefix or missing job)")
            })?;
            let job_dir =
                paths::job_dir(&job_id).with_context(|| format!("resolve job dir for {job_id}"))?;
            let marker = RecipeMarker::read_from(&job_dir)?;
            let r = reg
                .find(&marker.name)
                .ok_or_else(|| anyhow!("recipe '{}' not in catalog", marker.name))?;
            let plan =
                (r.compile_fn)(marker.args.clone()).map_err(|e| anyhow!("recipe compile: {e}"))?;

            let mut ctx = ExecCtx::new(job_dir.clone());
            if shared_cache {
                match CacheHandle::default_global_path() {
                    Some(global) => {
                        std::fs::create_dir_all(&global).with_context(|| {
                            format!("create global cache dir {}", global.display())
                        })?;
                        let cache_handle = (*ctx.cache).clone().with_global(global);
                        ctx.cache = std::sync::Arc::new(cache_handle);
                    }
                    None => eprintln!(
                        "warning: --shared-cache requested but no global cache \
                         path; falling back to job-local."
                    ),
                }
            }

            crate::jobs::write_state(&job_id, JobState::Running)
                .with_context(|| format!("write Running state for {job_id}"))?;

            // GPU lock — same arbitration as initial runs. Without
            // this, two resumes (or a resume + a fresh recipe run)
            // on the same machine could both touch the GPU.
            let lock = match scheduler_lock::acquire_exclusive(
                format!("blut-resume:{job_id}"),
                LockKind::Training,
            ) {
                Ok(l) => l,
                Err(e) => {
                    if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
                        tracing::warn!("write Failed state for {job_id}: {se}");
                    }
                    return Err(anyhow!("acquire_exclusive: {e}"));
                }
            };

            eprintln!("resuming {} ({})", marker.name, job_id);
            eprintln!("dir      {}", job_dir.display());
            eprintln!("lock     {}", lock.path().display());
            persist_plan_graph(&plan, &job_dir);
            let result = crate::framework::execute_plan(plan, ctx).await;
            drop(lock);
            match result {
                Ok(r) => {
                    crate::jobs::write_state(&job_id, JobState::Done)
                        .with_context(|| format!("write Done state for {job_id}"))?;
                    eprintln!(
                        "done — {} ingredients, {} cache hits, {} misses, elapsed {:?}",
                        r.n_stages, r.n_cache_hits, r.n_cache_misses, r.elapsed
                    );
                    Ok(())
                }
                Err(e) => {
                    if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
                        tracing::warn!("write Failed state for {job_id}: {se}");
                    }
                    // `.context()` (not `anyhow!("...: {e}")`) — preserves `e` as the
                    // source() of the new error, so a StageFailure buried in the
                    // PlanError→StageError chain survives to the top-level printer
                    // in `run()` (ADR 0072 A2).
                    Err(anyhow::Error::from(e).context("plan execution failed"))
                }
            }
        }
        PlanCommand::Inspect { name, args } => {
            let r = reg
                .find(&name)
                .ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
            let raw: serde_json::Value = serde_json::from_str(&args)
                .with_context(|| format!("parse --args as JSON: {args}"))?;
            let plan = (r.compile_fn)(raw).map_err(|e| anyhow!("recipe compile: {e}"))?;
            let rendered = plan
                .render_ascii()
                .map_err(|e| anyhow!("render plan: {e}"))?;
            print!("{rendered}");
            Ok(())
        }
    }
}

fn run_sensor_cmd(cmd: SensorCommand) -> Result<()> {
    use crate::sensor;
    match cmd {
        SensorCommand::List { json } => {
            let sensors = sensor::registry();
            if json {
                let arr: Vec<serde_json::Value> = sensors
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "name": s.name(),
                            "description": s.description(),
                            "outcome": s.evaluate(),
                        })
                    })
                    .collect();
                emit_json(&arr)?;
                return Ok(());
            }
            let (hname, hstatus, hdetail) = ("sensor", "status", "detail");
            println!("{hname:<20} {hstatus:<6} {hdetail}");
            for s in &sensors {
                let o = s.evaluate();
                let detail = if o.reason().is_empty() {
                    s.description()
                } else {
                    o.reason()
                };
                let (name, tag) = (s.name(), o.tag());
                println!("{name:<20} {tag:<6} {detail}");
            }
            Ok(())
        }
        SensorCommand::Eval { name, json } => {
            let s = sensor::find(&name)
                .ok_or_else(|| anyhow!("unknown sensor '{name}' — see `blut sensor list`"))?;
            let o = s.evaluate();
            if json {
                emit_json(&serde_json::json!({ "name": s.name(), "outcome": o }))?;
            } else {
                println!("{}: {} {}", s.name(), o.tag(), o.reason());
            }
            // Exit code is the contract: 0 = READY, 1 = SKIP/WAIT — so a
            // cron/daemon can gate a launch on `blut sensor eval <name>`.
            // We already printed the outcome, so a bare non-zero exit (not
            // an Err — which would print an ugly "Error:" line) is right.
            if matches!(o, crate::sensor::SensorOutcome::Ready) {
                Ok(())
            } else {
                std::process::exit(1);
            }
        }
    }
}

fn run_footprint_cmd(cmd: FootprintCommand) -> Result<()> {
    let store = crate::broker::FootprintStore::load();
    let path = crate::config::footprint_store_path();
    match cmd {
        FootprintCommand::List { json } => {
            let entries = store.entries_snapshot();
            if json {
                // {key: {ram_gb, source, n_samples}} — greppable for tooling.
                let obj: serde_json::Map<String, serde_json::Value> = entries
                    .into_iter()
                    .map(|(k, e)| {
                        (
                            k,
                            serde_json::json!({
                                "ram_gb": e.ram_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                                "source": format!("{:?}", e.source),
                                "n_samples": e.n_samples,
                            }),
                        )
                    })
                    .collect();
                emit_json(&obj)?;
                return Ok(());
            }
            println!("footprint store: {}", path.display());
            if entries.is_empty() {
                println!("(empty)");
                return Ok(());
            }
            println!("{:<40} {:>8}  {:<13} n", "key", "ram", "source");
            for (k, e) in entries {
                println!(
                    "{:<40} {:>6.1}G  {:<13} {}",
                    k,
                    e.ram_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
                    format!("{:?}", e.source),
                    e.n_samples
                );
            }
            Ok(())
        }
        FootprintCommand::Forget { key, recipe } => {
            let mut store = store;
            match (key, recipe) {
                (Some(_), Some(_)) => Err(anyhow!("pass EITHER a <key> OR --recipe, not both")),
                (Some(k), None) => {
                    if store.forget(&k)? {
                        println!("forgot calibration entry: {k}");
                    } else {
                        println!("no entry for key: {k} (nothing to forget)");
                    }
                    Ok(())
                }
                (None, Some(r)) => {
                    let n = store.forget_recipe(&r)?;
                    println!(
                        "forgot {n} calibration entr{} for recipe '{r}'",
                        if n == 1 { "y" } else { "ies" }
                    );
                    Ok(())
                }
                (None, None) => Err(anyhow!("specify a <key> or --recipe <name> to forget")),
            }
        }
    }
}

fn run_cache_cmd(cmd: CacheCommand) -> Result<()> {
    use crate::framework::CacheHandle;
    let global = CacheHandle::default_global_path()
        .ok_or_else(|| anyhow!("could not determine global cache path"))?;
    match cmd {
        CacheCommand::Show => {
            println!("global cache: {}", global.display());
            if !global.exists() {
                println!("(not created yet)");
                return Ok(());
            }
            let size = dir_size_bytes(&global)?;
            println!("size: {:.2} GiB", size as f64 / 1024.0 / 1024.0 / 1024.0);
            Ok(())
        }
        CacheCommand::Prune { max_gb } => {
            // Resolution order: --max-gb flag → $LAMU_CACHE_MAX_GB →
            // 50 GiB default. The default matches the plan's spec.
            let cap_gb = max_gb
                .or_else(|| {
                    std::env::var("LAMU_CACHE_MAX_GB")
                        .ok()
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or(50.0);
            let cap_bytes = (cap_gb * 1024.0 * 1024.0 * 1024.0) as u64;
            let freed = crate::framework::cache::lru_prune(&global, cap_bytes)
                .with_context(|| format!("lru_prune {}", global.display()))?;
            println!(
                "pruned {:.2} GiB from {} (cap {:.2} GiB)",
                freed as f64 / 1024.0 / 1024.0 / 1024.0,
                global.display(),
                cap_gb
            );
            Ok(())
        }
        CacheCommand::Stats { id, json } => {
            let stats = crate::framework::lineage::cache_stats(&id).map_err(|e| anyhow!("{e}"))?;
            if json {
                emit_json(&stats)?;
                return Ok(());
            }
            let (th, tm) = stats.totals();
            println!(
                "{:<32} {:>6} {:>6} {:>7}",
                "ingredient", "hits", "miss", "hit%"
            );
            for (stage, (h, m)) in &stats.per_stage {
                let pct = if h + m == 0 {
                    0.0
                } else {
                    *h as f64 * 100.0 / (*h + *m) as f64
                };
                println!("{stage:<32} {h:>6} {m:>6} {pct:>6.1}%");
            }
            let tpct = if th + tm == 0 {
                0.0
            } else {
                th as f64 * 100.0 / (th + tm) as f64
            };
            println!("{:<32} {th:>6} {tm:>6} {tpct:>6.1}%", "TOTAL");
            Ok(())
        }
    }
}

fn run_artifact_cmd(cmd: ArtifactCommand) -> Result<()> {
    use crate::framework::lineage;
    match cmd {
        ArtifactCommand::Ls { id, json } => {
            let recs = lineage::scan_artifacts(&id).map_err(|e| anyhow!("{e}"))?;
            if json {
                emit_json(&recs)?;
                return Ok(());
            }
            if recs.is_empty() {
                println!("no artifacts (job has no materialized ingredient outputs).");
                return Ok(());
            }
            println!("{:<24} {:<14} {:<10} ingredient", "kind", "hash", "schema");
            for r in &recs {
                let hash = r
                    .meta
                    .content_hash
                    .to_hex()
                    .chars()
                    .take(12)
                    .collect::<String>();
                let stage = r.meta.produced_by_stage.as_deref().unwrap_or("-");
                println!(
                    "{:<24} {:<14} v{:<9} {stage}",
                    r.meta.kind, hash, r.meta.schema
                );
            }
            Ok(())
        }
        ArtifactCommand::Inspect { hash, json } => {
            if hash.len() < 4 {
                return Err(anyhow!("hash prefix too short — give at least 4 hex chars"));
            }
            let recs = lineage::find_by_hash_prefix(&hash).map_err(|e| anyhow!("{e}"))?;
            match recs.as_slice() {
                [] => Err(anyhow!("no artifact with content hash prefix '{hash}'")),
                [r] => {
                    if json {
                        emit_json(&r)?;
                        return Ok(());
                    }
                    println!("sidecar: {}", r.sidecar_path.display());
                    println!("job:     {}", r.job_id);
                    emit_json(&r.meta)?;
                    Ok(())
                }
                many => {
                    eprintln!("ambiguous prefix '{hash}' — {} matches:", many.len());
                    for r in many {
                        eprintln!("  {} ({})", r.meta.content_hash.to_hex(), r.job_id);
                    }
                    Err(anyhow!("give a longer prefix"))
                }
            }
        }
    }
}

fn run_schedule_cmd(reg: &crate::framework::Registry, cmd: ScheduleCommand) -> Result<()> {
    use crate::schedule;
    match cmd {
        ScheduleCommand::List => {
            let recipes = schedule::list().map_err(|e| anyhow!("{e}"))?;
            if recipes.is_empty() {
                println!("no blut schedules installed.");
            } else {
                println!("installed recipe timers:");
                for r in recipes {
                    println!("  {r}");
                }
            }
            Ok(())
        }
        ScheduleCommand::Install {
            recipe,
            calendar,
            args,
        } => {
            // Validate the recipe exists + the args parse before touching
            // systemd, so a typo doesn't leave a broken unit behind.
            let def = reg
                .find(&recipe)
                .ok_or_else(|| anyhow!("recipe '{recipe}' not in catalog"))?;
            // `--calendar` wins; otherwise fall back to the recipe's built-in
            // SCHEDULE (E1↔E5). Neither present → hard error (no silent default).
            let calendar = calendar
                .or_else(|| def.schedule.map(str::to_string))
                .ok_or_else(|| {
                    anyhow!(
                        "recipe '{recipe}' has no built-in SCHEDULE; pass --calendar <OnCalendar>"
                    )
                })?;
            let _: serde_json::Value = serde_json::from_str(&args)
                .with_context(|| format!("parse --args as JSON: {args}"))?;
            schedule::install(&recipe, &calendar, &args).map_err(|e| anyhow!("{e}"))?;
            eprintln!("installed timer blut-{recipe}.timer (OnCalendar={calendar})");
            eprintln!("inspect: systemctl --user list-timers | grep {recipe}");
            Ok(())
        }
        ScheduleCommand::Uninstall { recipe } => {
            schedule::uninstall(&recipe).map_err(|e| anyhow!("{e}"))?;
            eprintln!("removed timer for {recipe}");
            Ok(())
        }
    }
}

/// Iterative dir-size walk with a bounded depth limit. Replaces
/// the prior recursive impl per Brian's Programming Bible Rule 19
/// (bound or eliminate recursion). The cache root has shallow
/// structure (`<cache_root>/<hex>/output.bin`); depth 16 is two
/// orders of magnitude beyond what any sane cache layout produces.
/// Anything deeper is corruption or a symlink loop and gets
/// surfaced as an error rather than stack-overflowed.
fn dir_size_bytes(path: &std::path::Path) -> Result<u64> {
    const MAX_DEPTH: u32 = 16;
    let mut total: u64 = 0;
    let mut stack: Vec<(std::path::PathBuf, u32)> = vec![(path.to_path_buf(), 0)];
    while let Some((dir, depth)) = stack.pop() {
        debug_assert!(depth <= MAX_DEPTH, "dir_size_bytes depth invariant");
        if depth > MAX_DEPTH {
            return Err(anyhow!(
                "dir_size_bytes: depth {depth} exceeds MAX_DEPTH {MAX_DEPTH} at {}",
                dir.display()
            ));
        }
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("read_dir {}", dir.display()))?
        {
            let entry = entry?;
            let m = entry.metadata()?;
            if m.is_dir() {
                stack.push((entry.path(), depth + 1));
            } else {
                total = total.saturating_add(m.len());
            }
        }
    }
    Ok(total)
}

/// Persist the plan STRUCTURE for the DAG backend (`blut dag`). Best-effort: a
/// snapshot write must NEVER fail or delay the actual run, so errors only warn.
/// Call with `&plan` BEFORE `execute_plan` moves it.
fn persist_plan_graph(plan: &crate::framework::CompiledPlan, job_dir: &std::path::Path) {
    match plan.graph_structure() {
        Ok(g) => {
            if let Err(e) = g.write_to(job_dir) {
                eprintln!("warning: could not write plan.json (blut dag unavailable): {e}");
            }
        }
        Err(e) => eprintln!("warning: plan graph unavailable, not persisted: {e}"),
    }
}

fn run_policy(cmd: PolicyCommand) -> Result<()> {
    use crate::policy;
    match cmd {
        PolicyCommand::Show => {
            let p = policy::load().context("load policy")?;
            print!(
                "{}",
                toml::to_string_pretty(&p).map_err(|e| anyhow!("serialize policy: {e}"))?
            );
            let path = policy::policy_path()?;
            eprintln!("# loaded from: {}", path.display());
        }
        PolicyCommand::Enable => {
            let mut p = policy::load().context("load policy")?;
            p.enabled = true;
            policy::validate(&p).context("validate policy")?;
            policy::save(&p).context("save policy")?;
            let path = policy::policy_path()?;
            println!("auto-trigger enabled (policy: {})", path.display());
            println!();
            println!("# Add to crontab so the heuristic runs every 30 min:");
            let exe = std::env::current_exe()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "lamu-train".into());
            // cron doesn't run through a shell that expands ~ , so
            // resolve the log path to an absolute string before
            // printing. Falls back to /tmp if XDG resolution fails.
            let log_path = dirs::data_local_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
                .join("lamu")
                .join("train-auto.log");
            println!("*/30 * * * * {exe} auto >> {} 2>&1", log_path.display());
        }
        PolicyCommand::Disable => {
            let mut p = policy::load().context("load policy")?;
            p.enabled = false;
            policy::save(&p).context("save policy")?;
            println!("auto-trigger disabled");
        }
    }
    Ok(())
}

fn run_data(cmd: DataCommand) -> Result<()> {
    use crate::datasets_db;
    let conn = datasets_db::open()?;
    match cmd {
        DataCommand::List => {
            let rows = datasets_db::list(&conn)?;
            if rows.is_empty() {
                println!("no datasets registered.");
                return Ok(());
            }
            println!(
                "{:<24} {:<12} {:>10} {:<16} path",
                "name", "kind", "examples", "sha256[:8]"
            );
            for r in rows {
                println!(
                    "{:<24} {:<12} {:>10} {:<16} {}",
                    truncate_for_col(&r.name, 24),
                    truncate_for_col(&r.kind, 12),
                    r.n_examples,
                    truncate_for_col(&r.sha256, 16),
                    r.source_path.display()
                );
            }
        }
        DataCommand::Add { name, path, kind } => {
            let rec = datasets_db::record_from_jsonl(&name, &path, &kind, None)?;
            datasets_db::add(&conn, &rec)?;
            println!(
                "registered '{name}' ({} examples, sha256={})",
                rec.n_examples, rec.sha256
            );
        }
        DataCommand::Rm { name } => {
            let removed = datasets_db::remove(&conn, &name)?;
            if removed {
                println!("removed '{name}'");
            } else {
                return Err(anyhow!("no dataset named '{name}'"));
            }
        }
        DataCommand::Show { name } => match datasets_db::get_by_name(&conn, &name)? {
            Some(rec) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rec)
                        .unwrap_or_else(|e| format!("serialize error: {e}"))
                );
            }
            None => return Err(anyhow!("no dataset named '{name}'")),
        },
    }
    Ok(())
}

fn truncate_for_col(s: &str, max: usize) -> String {
    // `max` is a CHAR budget — compare char count, not byte length, so a
    // multibyte string that fits the column isn't truncated early.
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,blut=info,hyper=warn,reqwest=warn")),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}

/// Install a one-shot SIGTERM + ctrl-c handler for an in-process
/// training run (KILL-3). On either signal it:
///   1. cancels the executor's `CancellationToken` so the running
///      stage's `tokio::select!` sees the cancel and returns,
///   2. `killpg`s the live python child group (the trainer tree),
///   3. drops the guard so RAII unlocks the scheduler lock cleanly.
///
/// Spawned as a detached task; it observes the *first* signal, kills,
/// and exits. The main task continues — `execute()` returns
/// `Cancelled`, the lock Drops, the process exits via the normal
/// `Err` path. We deliberately do NOT `process::exit()` so Drop glue
/// (lock unlink) runs.
fn install_cancel_handler(cancel: tokio_util::sync::CancellationToken) {
    tokio::spawn(async move {
        let term = async {
            #[cfg(unix)]
            {
                use tokio::signal::unix::{SignalKind, signal};
                if let Ok(mut s) = signal(SignalKind::terminate()) {
                    s.recv().await;
                }
            }
            #[cfg(not(unix))]
            {
                std::future::pending::<()>().await;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term => {}
        }
        eprintln!("\nsignal received — cancelling job + killing trainer group(s)...");
        cancel.cancel();
        // Kill EVERY registered child group — the parallel executor may
        // have more than one subprocess stage live at once.
        for id in crate::python_kill::active_children() {
            crate::python_kill::graceful_kill_group(id.pgid, Some(id), Duration::from_secs(10))
                .await;
        }
    });
}

// Subcommand implementations, split per subsystem (formerly inline in
// this file). Each is glob-imported so the Command enum and run()
// dispatcher reference their enums/handlers unchanged.
mod errors;
use errors::*;

mod gate;
use gate::*;

mod hpo;
use hpo::*;

mod lineage;
use lineage::*;

#[cfg(feature = "p2p")]
mod p2p;
#[cfg(feature = "p2p")]
use p2p::*;

mod partition;
use partition::*;

mod recipe;
use recipe::*;

mod runs;
use runs::*;

mod stale;
use stale::*;
