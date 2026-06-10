//! `lamu-train` — local fine-tuning subcommand binary.
//!
//! Top-level subcommands:
//!
//!   train (default)        Run a single fine-tune to completion (or
//!                          background if --background).
//!   jobs                   List jobs (running + completed).
//!   cancel <id>            SIGTERM the trainer subprocess.
//!   log <id>               Print rendered status.jsonl tail.
//!
//! Acquires the cross-process scheduler lock for the duration of a
//! training run; inference paths (lamu-mcp, lamu-api) refuse during
//! that window. Pass `--allow-evict` to wait if the lock is already
//! held by an inference exclusive instead of erroring.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::scheduler_lock::{self, LockKind};
use crate::{
    backend::{StatusFn, TrainBackend},
    convert,
    jobs::{self, JobState},
    paths,
    protocol::StatusUpdate,
    python_backend::PythonTrainBackend,
    spec::{DatasetSource, Method, Optim, TrainSpec},
};
use anyhow::{Context, Result, anyhow};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "blut",
    version,
    about = "BLUT — interactive training cockpit (bare `blut` opens the TUI). Subcommands: train, jobs, log, cancel, recipe, plan, cache, stage, data, auto, policy, tui."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    train_args: TrainArgs,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a fine-tune (also the default when invoked without a subcommand).
    Train(TrainArgs),
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
    /// Show a job's stage lineage (input→output hashes, cache hits).
    Lineage {
        /// Job id (or unique prefix).
        id: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Inspect materialized artifacts via their sidecars.
    Artifact {
        #[command(subcommand)]
        cmd: ArtifactCommand,
    },
    /// Manage the datasets registry.
    Data {
        #[command(subcommand)]
        cmd: DataCommand,
    },
    /// Cron entry point: read train-policy.toml, decide whether
    /// to spawn a training run, exit. Prints the decision reason
    /// on stdout regardless of outcome (for cron log readers).
    Auto,
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
    /// Run a single stage standalone — Unix-style. Reads erased
    /// input bytes from stdin (or skipped for graph-input stages),
    /// writes the produced erased artifact bytes to stdout.
    /// Pipeable; recipes are just compositions of these.
    Stage {
        #[command(subcommand)]
        cmd: StageCommand,
    },
    /// Open the canonical interactive training cockpit (ratatui). The
    /// single, complete cockpit: recipe launcher + live jobs/log/system
    /// panels + run history / leaderboard / compare / checkpoints /
    /// presets / live-metrics / reset views (superset of the retired
    /// hub + Python cockpits). Keys: ↑↓ select, Enter log, c cancel,
    /// R recipe picker, J/L/Y/H/B/K/P/M/X switch views, q quit.
    Tui,
}

#[derive(Subcommand, Debug)]
enum StageCommand {
    /// List the stage catalog.
    List,
    /// Execute one stage.
    Run {
        /// Stage name (e.g. filter_dataset).
        name: String,
        /// Stage args as inline JSON.
        #[arg(long)]
        args: String,
        /// Input source: `-` for stdin (bincode-encoded
        /// ErasedArtifact), or `:unit` for graph-input stages whose
        /// input is `()`. Defaults to `:unit`.
        #[arg(long, default_value = ":unit")]
        input: String,
        /// Output sink: `-` for stdout (bincode-encoded
        /// ErasedArtifact). Defaults to `-`.
        #[arg(long, default_value = "-")]
        output: String,
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
    /// Per-stage cache hit/miss tally for a job (from its status.jsonl).
    Stats {
        /// Job id (or unique prefix).
        id: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
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
enum RunsCommand {
    /// Diff two jobs' recipe/args provenance + show each outcome.
    Diff {
        /// First job id (or unique prefix).
        id1: String,
        /// Second job id (or unique prefix).
        id2: String,
        /// Show identical keys too (default elides them).
        #[arg(long)]
        all: bool,
        /// Emit the diff as JSON (for scripts/agents).
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum RecipeCommand {
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
    /// Execute a recipe with given args.
    Run {
        /// Recipe name.
        name: String,
        /// Args as inline JSON.
        #[arg(long)]
        args: String,
        /// Promote this run's outputs to the global cache for
        /// future re-use. Default: per-job cache only.
        #[arg(long, default_value_t = false)]
        shared_cache: bool,
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

#[derive(Args, Debug)]
struct TrainArgs {
    /// Registry name for the trained model. Required for actual runs;
    /// missing → help text.
    output_name: Option<String>,

    /// HuggingFace base model (org/name).
    #[arg(long, default_value = "Qwen/Qwen3-7B")]
    base: String,

    /// JSONL chat dataset path.
    #[arg(long)]
    dataset: Option<PathBuf>,

    /// Pull conversations from lamu-mcp memory (overrides --dataset).
    /// Materialization happens in step 7; flag accepted now for parity.
    #[arg(long, default_value_t = false)]
    from_conversations: bool,

    /// Window for --from-conversations.
    #[arg(long, default_value = "30d", value_parser = parse_duration)]
    since: Duration,

    /// Fine-tuning method.
    #[arg(long, value_enum, default_value_t = MethodArg::Qlora)]
    method: MethodArg,

    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    rank: u32,

    /// LoRA alpha.
    #[arg(long, default_value_t = 32)]
    alpha: u32,

    /// Optimizer.
    #[arg(long, value_enum)]
    optim: Option<OptimArg>,

    #[arg(long, default_value_t = 2e-4)]
    lr: f32,

    #[arg(long, default_value_t = 3)]
    epochs: u32,

    #[arg(long, default_value_t = 1)]
    batch_size: u32,

    #[arg(long, default_value_t = 8)]
    grad_accum: u32,

    #[arg(long, default_value_t = 4096)]
    seq_len: u32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Final GGUF quant.
    #[arg(long, default_value = "Q4_K_M")]
    quant: String,

    /// Skip GGUF convert + registry register (HF checkpoint only).
    #[arg(long, default_value_t = false)]
    no_convert: bool,

    /// Detach: write to ~/.local/share/lamu/train-jobs/<id>/, return
    /// the job id immediately. Use `lamu-train jobs` + `log <id>`.
    /// (Implementation: spawns a child of itself with --foreground.
    ///  v1 placeholder — wires to real detach in step 10 hardening.)
    #[arg(long, default_value_t = false)]
    background: bool,

    /// Wait for the GPU lock to release instead of erroring on hold.
    /// Polling interval is 500 ms; default timeout 1 h.
    #[arg(long, default_value_t = false)]
    allow_evict: bool,

    /// Promote this run's stage outputs to the global cache so
    /// future jobs can hit them. Only honoured by the v2 recipe
    /// path (--from-conversations without LAMU_TRAIN_USE_LEGACY=1).
    #[arg(long, default_value_t = false)]
    shared_cache: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MethodArg {
    Qlora,
    Lora,
    Full,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OptimArg {
    Adamw,
    Adamw8bit,
    Apollo,
    ApolloMini,
}

fn parse_duration(s: &str) -> Result<Duration, String> {
    humantime::parse_duration(s).map_err(|e| format!("{e}"))
}

/// BLUT CLI entrypoint. The recipe catalog is supplied by the caller as
/// a composed [`Registry`] (the binary — in a cookbook crate — registers
/// the cookbooks it ships and passes them here). This is the lib seam
/// that lets blut-core stay domain-agnostic: a bare blut engine binary
/// would pass an empty registry; the cookbook binaries pass theirs.
pub async fn run(reg: crate::framework::Registry) -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Train(args)) => run_train(&reg, args).await,
        Some(Command::Jobs { json }) => run_jobs(json),
        Some(Command::Cancel { id, grace }) => run_cancel(&id, grace).await,
        Some(Command::Log { id, tail, json }) => run_log(&id, tail, json),
        Some(Command::Runs { cmd }) => run_runs_cmd(cmd),
        Some(Command::Lineage { id, json }) => run_lineage(&id, json),
        Some(Command::Artifact { cmd }) => run_artifact_cmd(cmd),
        Some(Command::Data { cmd }) => run_data(cmd),
        Some(Command::Auto) => run_auto().await,
        Some(Command::Policy { cmd }) => run_policy(cmd),
        Some(Command::Recipe { cmd }) => run_recipe(&reg, cmd).await,
        Some(Command::Plan { cmd }) => run_plan_cmd(&reg, cmd).await,
        Some(Command::Cache { cmd }) => run_cache_cmd(cmd),
        Some(Command::Stage { cmd }) => run_stage_cmd(cmd).await,
        Some(Command::Tui) => crate::tui::run(reg).await,
        // Bare `blut` opens the interactive cockpit (T-track). Use
        // `blut train …` for explicit CLI training.
        None => crate::tui::run(reg).await,
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
            let result = crate::framework::execute_plan(plan, ctx).await;
            drop(lock);
            match result {
                Ok(r) => {
                    crate::jobs::write_state(&job_id, JobState::Done)
                        .with_context(|| format!("write Done state for {job_id}"))?;
                    eprintln!(
                        "done — {} stages, {} cache hits, {} misses, elapsed {:?}",
                        r.n_stages, r.n_cache_hits, r.n_cache_misses, r.elapsed
                    );
                    Ok(())
                }
                Err(e) => {
                    if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
                        tracing::warn!("write Failed state for {job_id}: {se}");
                    }
                    Err(anyhow!("plan execution failed: {e}"))
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&stats).map_err(|e| anyhow!("{e}"))?
                );
                return Ok(());
            }
            let (th, tm) = stats.totals();
            println!("{:<32} {:>6} {:>6} {:>7}", "stage", "hits", "miss", "hit%");
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

fn run_lineage(id_query: &str, json: bool) -> Result<()> {
    let nodes = crate::framework::lineage::job_lineage(id_query).map_err(|e| anyhow!("{e}"))?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&nodes).map_err(|e| anyhow!("{e}"))?
        );
        return Ok(());
    }
    if nodes.is_empty() {
        println!("no stage lineage (job has no framework status events).");
        return Ok(());
    }
    for n in &nodes {
        let inp = n.input_hash.as_deref().unwrap_or("-");
        let out = n.output_hash.as_deref().unwrap_or("-");
        let short = |h: &str| h.chars().take(12).collect::<String>();
        if n.cached {
            println!("  {:>2} {:<28} [CACHE HIT {}]", n.node_idx, n.stage, short(out));
        } else {
            let took = n
                .elapsed
                .map(|e| format!("{e:?}"))
                .unwrap_or_else(|| "-".into());
            println!(
                "  {:>2} {:<28} in={} → out={}  {}",
                n.node_idx,
                n.stage,
                short(inp),
                short(out),
                took
            );
        }
    }
    Ok(())
}

fn run_artifact_cmd(cmd: ArtifactCommand) -> Result<()> {
    use crate::framework::lineage;
    match cmd {
        ArtifactCommand::Ls { id, json } => {
            let recs = lineage::scan_artifacts(&id).map_err(|e| anyhow!("{e}"))?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&recs).map_err(|e| anyhow!("{e}"))?
                );
                return Ok(());
            }
            if recs.is_empty() {
                println!("no artifacts (job has no materialized stage outputs).");
                return Ok(());
            }
            println!("{:<24} {:<14} {:<10} stage", "kind", "hash", "schema");
            for r in &recs {
                let hash = r.meta.content_hash.to_hex().chars().take(12).collect::<String>();
                let stage = r.meta.produced_by_stage.as_deref().unwrap_or("-");
                println!("{:<24} {:<14} v{:<9} {stage}", r.meta.kind, hash, r.meta.schema);
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
                        println!(
                            "{}",
                            serde_json::to_string_pretty(&r).map_err(|e| anyhow!("{e}"))?
                        );
                        return Ok(());
                    }
                    println!("sidecar: {}", r.sidecar_path.display());
                    println!("job:     {}", r.job_id);
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&r.meta).map_err(|e| anyhow!("{e}"))?
                    );
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

async fn run_stage_cmd(cmd: StageCommand) -> Result<()> {
    use crate::framework::artifact::Artifact;
    use crate::framework::cache::CacheHandle;
    use crate::framework::stage::{ErasedArtifact, StageContext};
    use crate::stages::catalog;
    use std::io::{Read, Write};
    use std::sync::Arc;

    match cmd {
        StageCommand::List => {
            println!(
                "{:<32} {:<20} {:<20} resources",
                "name", "input_kind", "output_kind"
            );
            for n in catalog::names() {
                let s = catalog::make_stage(n).expect("listed → constructs");
                println!(
                    "{:<32} {:<20} {:<20} {:?}",
                    s.name(),
                    s.input_kind(),
                    s.output_kind(),
                    s.resources(),
                );
            }
            Ok(())
        }
        StageCommand::Run {
            name,
            args,
            input,
            output,
        } => {
            let stage = catalog::make_stage(&name)
                .ok_or_else(|| anyhow!("stage '{name}' not in catalog"))?;
            let args_val: serde_json::Value = serde_json::from_str(&args)
                .with_context(|| format!("parse --args as JSON: {args}"))?;

            // Read input. `:unit` produces a synthetic () artifact;
            // `-` reads bincode-encoded ErasedArtifact from stdin;
            // a path reads from disk.
            let erased_input: ErasedArtifact = match input.as_str() {
                ":unit" => ErasedArtifact {
                    kind: <() as Artifact>::KIND.into(),
                    schema: <() as Artifact>::SCHEMA,
                    payload: bincode::serialize(&()).map_err(|e| anyhow!("encode unit: {e}"))?,
                },
                "-" => {
                    let mut buf = Vec::new();
                    std::io::stdin()
                        .read_to_end(&mut buf)
                        .context("read stdin for --input -")?;
                    bincode::deserialize(&buf)
                        .map_err(|e| anyhow!("decode stdin ErasedArtifact: {e}"))?
                }
                path => {
                    let buf =
                        std::fs::read(path).with_context(|| format!("read input from {path}"))?;
                    bincode::deserialize(&buf).map_err(|e| anyhow!("decode {path}: {e}"))?
                }
            };

            // Each `blut stage` invocation gets its own scratch
            // tempdir, including a private cache dir. The cache
            // lives only for this invocation — recipes are the
            // entry point for cross-stage cache hits.
            let td = tempfile::tempdir().context("create stage tempdir")?;
            let stage_dir = td.path().join("stage");
            std::fs::create_dir_all(&stage_dir)?;
            let ctx = StageContext {
                job_dir: td.path().to_path_buf(),
                stage_dir,
                node_idx: 0,
                status_tx: crate::framework::status::make_broadcast(),
                cancel: tokio_util::sync::CancellationToken::new(),
                cache: Arc::new(CacheHandle::job_local(td.path().join("_cache"))),
                recipe_name: String::new(),
            };

            let result = stage
                .run_erased(&ctx, erased_input, args_val)
                .await
                .map_err(|e| anyhow!("stage '{name}' failed: {e}"))?;

            // Write output, then flush so a downstream pipe sees
            // the bytes immediately rather than waiting for process
            // exit + OS buffer drain.
            let body = bincode::serialize(&result).map_err(|e| anyhow!("encode output: {e}"))?;
            match output.as_str() {
                "-" => {
                    let mut out = std::io::stdout().lock();
                    out.write_all(&body).context("write stdout")?;
                    out.flush().context("flush stdout")?;
                }
                path => {
                    std::fs::write(path, &body)
                        .with_context(|| format!("write output to {path}"))?;
                }
            }
            // Tempdir is dropped at function exit; we don't call
            // td.close() because subprocesses (e.g. trainer) that
            // outlive run_erased can leave open handles in the dir,
            // and a noisy "cleanup failed" warning isn't actionable.
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
fn recipe_footprint(name: &str, raw: &serde_json::Value) -> crate::broker::Footprint {
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

async fn run_recipe(reg: &crate::framework::Registry, cmd: RecipeCommand) -> Result<()> {
    use crate::framework::ExecCtx;
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&arr)
                        .map_err(|e| anyhow!("serialize recipes: {e}"))?
                );
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
        RecipeCommand::Run {
            name,
            args,
            shared_cache,
        } => {
            let r = find_recipe(&name).ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
            let raw: serde_json::Value = serde_json::from_str(&args)
                .with_context(|| format!("parse --args as JSON: {args}"))?;
            let plan =
                (r.compile_fn)(raw.clone()).map_err(|e| anyhow!("recipe compile failed: {e}"))?;
            // ADR 0046 slice-1: resolve the RAM footprint from the args
            // BEFORE `raw` is consumed by the RecipeMarker below; the
            // admission gate (after the job state is written) reuses it.
            let footprint = recipe_footprint(&name, &raw);

            let job_id = crate::jobs::new_job_id();
            let job_dir = crate::paths::job_dir(&job_id)?;
            let mut ctx = ExecCtx::new(job_dir.clone());
            if shared_cache {
                if let Some(global) = crate::framework::CacheHandle::default_global_path() {
                    std::fs::create_dir_all(&global)
                        .with_context(|| format!("create global cache dir {}", global.display()))?;
                    let cache_handle = (*ctx.cache).clone().with_global(global);
                    ctx.cache = std::sync::Arc::new(cache_handle);
                }
            }
            // Mark recipe for plan resume. Reuse the parsed
            // `raw` rather than re-parsing `args` — re-parse +
            // unwrap_or would silently swallow malformed JSON
            // that already failed above.
            RecipeMarker {
                name: name.clone(),
                args: raw,
            }
            .write_to(&job_dir)?;

            crate::jobs::write_state(&job_id, JobState::Running)
                .with_context(|| format!("write Running state for {job_id}"))?;

            // KILL-2/KILL-3: bind this job so backend spawns mirror
            // the python child's PROCESS GROUP id into the job pid
            // file (not blut's own pid). A separate `blut cancel <id>`
            // reads that pgid and killpg's the whole tree.
            crate::python_kill::bind_current_job(job_id.clone());

            // KILL-3: trap SIGTERM/ctrl-c. On signal, cancel the
            // executor token AND killpg the live child group, then
            // let the function return so `lock` Drops (RAII unlocks
            // the scheduler — fixes the stale-lock-on-SIGTERM case).
            install_cancel_handler(ctx.cancel.clone());

            // ADR 0046 slice-1 (item 4): RAM-refuse admission gate,
            // placed BEFORE the scheduler lock. The review verified the
            // lock already serializes blut-vs-blut GPU jobs (fail-fast),
            // so this is a pure single-job over-subscription guard — if
            // the conservative-high footprint can't fit free RAM (or the
            // box at all), refuse CLEANLY here: no launch, no transient
            // unit, no OOM. Cross-job queuing stays the lock's job; we do
            // NOT build a poll-queue. Best-effort: a recipe whose args
            // carry no cost drivers falls back to the conservative
            // default footprint, which still gates oversubscription.
            if let Err(reason) = crate::broker::gate(&format!("recipe '{name}'"), &footprint) {
                crate::python_kill::unbind_current_job();
                if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
                    tracing::warn!("write Failed state for {job_id}: {se}");
                }
                return Err(anyhow!("{reason}"));
            }

            // Cross-process GPU arbitration — same lock acquisition
            // pattern as the legacy train path. Recipes that don't
            // hit GPU still pay the lock cost, which is cheap.
            let lock = match scheduler_lock::acquire_exclusive(
                format!("blut-recipe:{job_id}"),
                LockKind::Training,
            ) {
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

            let result = crate::framework::execute_plan(plan, ctx).await;
            drop(lock);
            crate::python_kill::unbind_current_job();
            match result {
                Ok(r) => {
                    crate::jobs::write_state(&job_id, JobState::Done)
                        .with_context(|| format!("write Done state for {job_id}"))?;
                    eprintln!(
                        "done — {} stages, {} cache hits, {} misses, elapsed {:?}",
                        r.n_stages, r.n_cache_hits, r.n_cache_misses, r.elapsed
                    );
                }
                Err(e) => {
                    if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
                        tracing::warn!("write Failed state for {job_id}: {se}");
                    }
                    return Err(anyhow!("plan execution failed: {e}"));
                }
            }
        }
    }
    Ok(())
}

async fn run_auto() -> Result<()> {
    use crate::{conversations, policy};

    let pol = policy::load().context("load policy")?;
    let (now_unix, now_local_min) = policy::current_clock();
    let lock_held = crate::scheduler_lock::check_unlocked().is_err();
    let new_turns = match conversations::count_turns_since(pol.last_train_ts) {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!(
                "count_turns_since failed ({e}); treating as 0. Auto will skip \
                 on the threshold check rather than spawning blindly."
            );
            0
        }
    };

    let decision = policy::decide(&pol, now_unix, now_local_min, new_turns, lock_held);
    match decision {
        policy::Decision::Skip(reason) => {
            println!("auto: {reason}");
            return Ok(());
        }
        policy::Decision::Run {
            base,
            method,
            since,
        } => {
            println!(
                "auto: triggering training (new_turns={new_turns}, threshold={})",
                pol.threshold_new_turns
            );
            let bin = std::env::current_exe().context("locate own binary for auto-spawn")?;
            let auto_name = format!("auto-{}", crate::jobs::new_job_id());
            let mut cmd = tokio::process::Command::new(&bin);
            cmd.arg(&auto_name)
                .arg("--from-conversations")
                .arg("--since")
                .arg(&since)
                .arg("--base")
                .arg(&base)
                .arg("--method")
                .arg(&method)
                .arg("--background")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .kill_on_drop(false);
            match cmd.spawn() {
                Ok(mut child) => {
                    let pid = child.id().unwrap_or(0);
                    println!("auto: spawned lamu-train pid={pid} as '{auto_name}'");
                    // Stamp the ATTEMPT (not last_train_ts) at spawn — keeps
                    // the cron from double-spawning a live run, while leaving
                    // `last_train_ts` (the cooldown anchor) to advance ONLY on
                    // a real completion. The child run (output_name `auto-*`)
                    // records the outcome via `record_auto_outcome`: success
                    // advances last_train_ts + clears the failure counter,
                    // failure increments it (driving the backoff).
                    let mut updated = pol.clone();
                    updated.last_attempt_ts = now_unix;
                    updated.last_train_n_turns = new_turns;
                    if let Err(e) = policy::save(&updated) {
                        tracing::warn!("failed to update last_attempt_ts: {e}");
                    }
                    // Reap the zombie when training finishes; the
                    // cron-driven `auto` exits while the child runs.
                    // Log non-zero exit so train-auto.log shows the
                    // outcome instead of just the spawn line.
                    tokio::spawn(async move {
                        match child.wait().await {
                            Ok(status) if !status.success() => {
                                tracing::warn!("auto-train (pid={pid}) exited with {status}");
                            }
                            Ok(status) => {
                                tracing::info!("auto-train (pid={pid}) exited cleanly: {status}");
                            }
                            Err(e) => {
                                tracing::warn!("auto-train (pid={pid}) wait failed: {e}");
                            }
                        }
                    });
                }
                Err(e) => return Err(anyhow!("spawn lamu-train: {e}")),
            }
        }
    }
    Ok(())
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

/// Register a JSONL dataset in the datasets registry. Best-effort:
/// callers handle failure by logging + continuing. Used by
/// auto-registration after `--from-conversations` materialization.
fn register_dataset(name: &str, path: &Path, kind: &str, metadata: Option<String>) -> Result<()> {
    let conn = crate::datasets_db::open()?;
    let rec = crate::datasets_db::record_from_jsonl(name, path, kind, metadata)?;
    crate::datasets_db::add(&conn, &rec)?;
    Ok(())
}

fn truncate_for_col(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let cut = s
            .char_indices()
            .nth(max.saturating_sub(1))
            .map(|(i, _)| i)
            .unwrap_or(s.len().min(max));
        format!("{}…", &s[..cut])
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

async fn run_train(reg: &crate::framework::Registry, args: TrainArgs) -> Result<()> {
    let output_name = args
        .output_name
        .clone()
        .ok_or_else(|| anyhow!("output-name is required (positional). See `lamu-train --help`."))?;

    // v2 commit 8: `--from-conversations` now unconditionally
    // delegates to the typed-Plan recipe pipeline. The
    // `LAMU_TRAIN_USE_LEGACY=1` kill-switch shipped in commit 4b
    // is gone — the recipe path has been the default through a
    // release window and the legacy linear flow only remains for
    // `--dataset <path>` runs (no recipe equivalent yet).
    if args.from_conversations {
        return run_train_via_recipe(reg, &output_name, &args).await;
    }

    let dataset_src = build_dataset(&args)?;
    let optimizer = pick_optimizer(args.optim, args.method);
    let method = build_method(args.method, args.rank, args.alpha);

    let job_id = jobs::new_job_id();
    let job_dir =
        paths::job_dir(&job_id).with_context(|| format!("create job dir for {job_id}"))?;
    let output_dir = job_dir.join("checkpoint");
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("create checkpoint dir {}", output_dir.display()))?;

    // trainer.py only accepts JsonlPath at runtime. Materialize
    // Conversations sources to a JSONL file under paths::data_dir
    // before spec construction so the file path lands in the
    // committed spec.json on disk for audit.
    let dataset = match dataset_src {
        DatasetSource::Conversations { .. } => {
            let data_dir = paths::data_dir().context("resolve train-data dir")?;
            std::fs::create_dir_all(&data_dir)
                .with_context(|| format!("create {}", data_dir.display()))?;
            let out_path = data_dir.join(format!("{job_id}.jsonl"));
            let stats = crate::conversations::dump_to_jsonl(args.since, &out_path)
                .context("dump conversations to JSONL")?;
            eprintln!(
                "dataset materialized: {} conversations, {} turns → {}",
                stats.n_conversations,
                stats.n_turns,
                stats.path.display()
            );
            if stats.n_conversations == 0 {
                return Err(anyhow!(
                    "no usable conversations in window (--since {:?}). \
                     {} short raw, {} gutted by filters, \
                     {} error messages, {} oversize messages.",
                    args.since,
                    stats.n_dropped_short,
                    stats.n_dropped_filtered_below_min,
                    stats.n_dropped_errors,
                    stats.n_dropped_oversize
                ));
            }
            // Lineage: register the materialized dataset under
            // 'conversations-<since>-<jobid>' so the trained model
            // can be traced back to its source. Failure to register
            // is a warning, not a hard error — the training itself
            // doesn't depend on the registry, only its audit trail.
            let dataset_name = format!(
                "conversations-{}-{job_id}",
                humantime::format_duration(args.since)
            );
            let metadata = serde_json::json!({
                "source": "conversations",
                "since": humantime::format_duration(args.since).to_string(),
                "n_dropped_short": stats.n_dropped_short,
                "n_dropped_filtered_below_min": stats.n_dropped_filtered_below_min,
                "n_dropped_errors": stats.n_dropped_errors,
                "n_dropped_oversize": stats.n_dropped_oversize,
                "job_id": job_id,
            })
            .to_string();
            match register_dataset(&dataset_name, &out_path, "conversations", Some(metadata)) {
                Ok(()) => eprintln!("dataset registered as '{dataset_name}'"),
                Err(e) => tracing::warn!(
                    "failed to register dataset '{dataset_name}': {e}; \
                     training will continue without lineage record"
                ),
            }
            DatasetSource::JsonlPath { path: out_path }
        }
        other => other,
    };

    let spec = TrainSpec {
        base_model: args.base.clone(),
        output_name: output_name.clone(),
        output_dir: output_dir.clone(),
        method,
        dataset,
        optimizer,
        lr: args.lr,
        epochs: args.epochs,
        batch_size: args.batch_size,
        grad_accum: args.grad_accum,
        seq_len: args.seq_len,
        seed: args.seed,
        quant: args.quant.clone(),
        skip_convert: args.no_convert,
        dpo_beta: None,
    };
    spec.validate().context("TrainSpec validation")?;
    jobs::write_spec(&job_id, &spec)?;
    jobs::write_state(&job_id, JobState::Running)?;

    eprintln!("job  {job_id}");
    eprintln!("dir  {}", job_dir.display());

    if args.background {
        // Background scaffold — spawn ourselves with the same args
        // minus --background. v1 implementation: print the job id +
        // a hint; the actual detach is wired in a follow-up commit
        // since clean nohup-style detach + log redirection deserves
        // its own review pass.
        eprintln!(
            "background mode is recognised but real detach lands in a follow-up.\n\
             For now, run without --background and use `lamu-train cancel {job_id}`\n\
             from another terminal if you need to stop early."
        );
        return Ok(());
    }

    // Resolve subprocess paths BEFORE acquiring the GPU lock so a
    // path-resolution failure doesn't hold the lock. Cheap (a few
    // env reads + stat calls); failure here means the user's setup
    // is wrong and they need a clear error, not a held lock.
    let python = paths::resolve_python().context("resolve python")?;
    let trainer_script = paths::resolve_trainer_script().context("resolve trainer.py")?;
    eprintln!("python {}", python.display());
    eprintln!("trainer {}", trainer_script.display());

    // Admission gate BEFORE the lock (ADR 0046) — the legacy bare-spawn
    // path was previously un-gated and could OOM the box. Bill a
    // conservative legacy footprint (cap workers, the spec's batch, the
    // smallest model tier) so admission's `need` never under-counts. A
    // probe miss admits; a refusal fails the job cleanly with no lock.
    // (Stopgap: this path is slated for replacement by `systemd-run`.)
    {
        let drivers = crate::broker::Drivers::new(
            crate::broker::UNCALIBRATED_WORKER_CAP,
            spec.batch_size,
            1,
            0,
        );
        if let Err(reason) = crate::broker::gate(&format!("train:{job_id}"), &drivers.estimate()) {
            if let Err(se) = jobs::write_state(&job_id, JobState::Failed) {
                tracing::warn!("write Failed state for {job_id}: {se}");
            }
            return Err(anyhow!("{reason}"));
        }
    }

    // Acquire the GPU lock. --allow-evict waits for an existing
    // inference exclusive to release; otherwise hard error.
    let lock = if args.allow_evict {
        eprintln!("lock waiting for GPU release (--allow-evict, up to 1h)...");
        scheduler_lock::await_unlock(Duration::from_secs(3600))
            .await
            .context("await_unlock")?;
        scheduler_lock::acquire_exclusive(format!("lamu-train:{job_id}"), LockKind::Training)
            .context("acquire_exclusive after wait")?
    } else {
        scheduler_lock::acquire_exclusive(format!("lamu-train:{job_id}"), LockKind::Training)
            .context("acquire_exclusive (use --allow-evict to wait)")?
    };
    eprintln!("lock acquired ({})", lock.path().display());

    let mut backend = PythonTrainBackend::new(python, trainer_script);

    let job_id_for_cb = job_id.clone();
    let on_status: StatusFn = Box::new(move |u: StatusUpdate| {
        // Persist to status.jsonl + render to stderr so the user
        // sees progress live in foreground mode. A persist failure
        // (full disk, permissions, etc.) is logged but doesn't stop
        // the run — losing status history is bad but losing the
        // training job mid-flight is worse.
        if let Err(e) = jobs::append_status(&job_id_for_cb, &u) {
            tracing::warn!("failed to persist status to {}: {}", job_id_for_cb, e);
        }
        match &u {
            StatusUpdate::Step {
                step,
                total,
                loss,
                lr,
                vram_mb,
            } => eprintln!("step {step}/{total}  loss={loss:.4}  lr={lr:.2e}  vram={vram_mb}MB"),
            StatusUpdate::Eval { step, eval_loss } => {
                eprintln!("eval @{step}  loss={eval_loss:.4}")
            }
            StatusUpdate::Saved { path } => eprintln!("saved {}", path.display()),
            StatusUpdate::Done {
                final_loss,
                checkpoint_dir,
            } => eprintln!(
                "done  final_loss={final_loss:.4}  ckpt={}",
                checkpoint_dir.display()
            ),
            StatusUpdate::Failed { error } => eprintln!("FAILED: {error}"),
            StatusUpdate::Heartbeat { phase, .. } => {
                if let Some(p) = phase {
                    eprintln!("… {p}");
                }
            }
        }
    });

    let result = backend.run(spec.clone(), on_status).await;

    drop(lock); // release GPU before convert + register; convert is
    // CPU-bound and llama.cpp tools don't need the card.

    match result {
        Ok(artifact) => {
            jobs::write_state(&job_id, JobState::Done)?;
            // Advance the auto cooldown + clear the failure backoff only on
            // a real completion (no-op for non-`auto-*` runs).
            crate::policy::record_auto_outcome(&output_name, true);
            eprintln!(
                "trained in {:?}, final_loss={:.4}, ckpt={}",
                artifact.elapsed,
                artifact.final_loss,
                artifact.checkpoint_dir.display()
            );

            if !args.no_convert {
                eprintln!("converting to GGUF ({})...", args.quant);
                let gguf =
                    convert::convert_to_gguf(&artifact.checkpoint_dir, &output_name, &args.quant)
                        .await
                        .context("convert_to_gguf")?;
                eprintln!("gguf  {}", gguf.display());
                register_in_registry(&output_name, &gguf, &spec)?;
                eprintln!(
                    "registry updated; `mcp__local-llm__query model={output_name}` should work."
                );
            } else {
                eprintln!(
                    "--no-convert: HF checkpoint left at {}",
                    artifact.checkpoint_dir.display()
                );
            }
        }
        Err(e) => {
            jobs::write_state(&job_id, JobState::Failed)?;
            // Grow the auto failure backoff (no-op for non-`auto-*` runs).
            crate::policy::record_auto_outcome(&output_name, false);
            return Err(anyhow!(e));
        }
    }
    Ok(())
}

fn run_jobs(json: bool) -> Result<()> {
    let jobs = jobs::list_jobs()?;
    if json {
        // JobSummary derives Serialize — emit the array verbatim so a
        // script/agent gets the same data the table renders.
        let out = serde_json::to_string_pretty(&jobs)
            .map_err(|e| anyhow!("serialize jobs: {e}"))?;
        println!("{out}");
        return Ok(());
    }
    if jobs.is_empty() {
        println!("no jobs.");
        return Ok(());
    }
    println!(
        "{:<24} {:<10} {:<6} {:<24} last",
        "id", "state", "pid", "output"
    );
    for j in jobs {
        let last = match (j.last_step, j.last_loss, j.final_loss) {
            (_, _, Some(fl)) => format!("final_loss={fl:.4}"),
            (Some(step), Some(loss), _) => format!("step={step} loss={loss:.4}"),
            _ => "-".into(),
        };
        let pid = j.pid.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
        let output = j.output_name.unwrap_or_else(|| "-".into());
        println!(
            "{:<24} {:<10} {:<6} {:<24} {}",
            j.id,
            j.state.as_str(),
            pid,
            output,
            last
        );
    }
    Ok(())
}

fn run_runs_cmd(cmd: RunsCommand) -> Result<()> {
    match cmd {
        RunsCommand::Diff {
            id1,
            id2,
            all,
            json,
        } => crate::runs::diff(&id1, &id2, all, json).map_err(|e| anyhow!("{e}")),
    }
}

async fn run_cancel(id_query: &str, grace: Duration) -> Result<()> {
    let id = jobs::resolve_job_id(id_query)?;
    eprintln!("cancelling {id} (grace {grace:?})...");
    jobs::cancel_job(&id, grace).await?;
    eprintln!("cancelled.");
    Ok(())
}

fn run_log(id_query: &str, tail: usize, json: bool) -> Result<()> {
    let id = jobs::resolve_job_id(id_query)?;
    let updates = jobs::read_status(&id)?;
    if json {
        // Raw status stream as JSON lines (one StatusUpdate per line),
        // tail-trimmed like the rendered view.
        let start = if tail == 0 {
            0
        } else {
            updates.len().saturating_sub(tail)
        };
        for u in &updates[start..] {
            println!(
                "{}",
                serde_json::to_string(u).map_err(|e| anyhow!("serialize status: {e}"))?
            );
        }
        return Ok(());
    }
    let rendered = jobs::render_log(&updates);
    if tail == 0 {
        print!("{rendered}");
    } else {
        let lines: Vec<&str> = rendered.lines().collect();
        let start = lines.len().saturating_sub(tail);
        for l in &lines[start..] {
            println!("{l}");
        }
    }
    Ok(())
}

/// Dispatch `--from-conversations` through the v2 typed Plan
/// pipeline. Builds the recipe Args from the legacy CLI flags so
/// users don't have to learn a new invocation, then runs the
/// compiled 9-stage Plan through `SequentialExecutor`.
async fn run_train_via_recipe(
    reg: &crate::framework::Registry,
    output_name: &str,
    args: &TrainArgs,
) -> Result<()> {
    use crate::framework::{CacheHandle, ExecCtx};

    // The recipe now lives in the lamu cookbook crate (C2b); blut-core
    // can't name its typed `Args`, so we build the args JSON directly and
    // resolve the erased `RecipeDef` by name through the caller-supplied
    // cookbook registry. Field names MUST match the recipe's serde
    // contract (validated at runtime inside `compile_fn`). Fields whose
    // value equals the recipe's serde default are omitted (notes,
    // eval_ratio, min_turns, max_msg_bytes, drop_errors,
    // dataset_registry_name).
    let method: &str = match args.method {
        MethodArg::Qlora => "qlora",
        MethodArg::Lora => "lora",
        MethodArg::Full => "full",
    };
    let optimizer: &str = match pick_optimizer(args.optim, args.method) {
        crate::spec::Optim::AdamW => "adamw",
        crate::spec::Optim::AdamW8bit => "adamw8bit",
        crate::spec::Optim::ApolloRank4 => "apollo",
        crate::spec::Optim::ApolloMini => "apollo_mini",
    };
    let recipe_args = serde_json::json!({
        "output_name": output_name,
        "since": humantime::format_duration(args.since).to_string(),
        "base_model": args.base,
        "method": method,
        "quant": args.quant,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "rank": args.rank,
        "alpha": args.alpha,
        "optimizer": optimizer,
    });

    let def = reg.find("finetune_from_conversations").ok_or_else(|| {
        anyhow!(
            "recipe `finetune_from_conversations` is not registered in this binary \
             (it belongs to the lamu cookbook — run via the `blut-lamu` binary)"
        )
    })?;
    let plan =
        (def.compile_fn)(recipe_args.clone()).map_err(|e| anyhow!("recipe compile failed: {e}"))?;

    let job_id = jobs::new_job_id();
    let job_dir =
        paths::job_dir(&job_id).with_context(|| format!("create job dir for {job_id}"))?;

    // Match the legacy path's lifecycle so `lamu-train jobs` shows
    // this run and `lamu-train cancel` can find its pid.
    jobs::write_state(&job_id, JobState::Running)
        .with_context(|| format!("write initial job state for {job_id}"))?;
    // KILL-3: DO NOT write blut's own pid here. The pid file must
    // hold the python child's PROCESS GROUP id so `blut cancel <id>`
    // SIGTERMs the trainer tree, not blut (which has no handler and
    // would just die, orphaning the GPU child). The child pgid is
    // mirrored into the pid file by the backend spawn once we bind
    // the job below; until then the job has no pid (cancel no-ops
    // safely rather than killing the wrong process).
    crate::python_kill::bind_current_job(job_id.clone());

    let mut ctx = ExecCtx::new(job_dir.clone());
    if args.shared_cache {
        match CacheHandle::default_global_path() {
            Some(global) => {
                std::fs::create_dir_all(&global)
                    .with_context(|| format!("create global cache dir {}", global.display()))?;
                let cache_handle = (*ctx.cache).clone().with_global(global);
                ctx.cache = std::sync::Arc::new(cache_handle);
            }
            None => {
                eprintln!(
                    "warning: --shared-cache requested but global cache \
                     path could not be determined (set $LAMU_TRAIN_CACHE_DIR \
                     or fix $XDG_DATA_HOME); falling back to job-local cache."
                );
            }
        }
    }

    // Mark recipe for plan resume. recipe_args is already the args JSON
    // value built above (reused verbatim so the marker matches what was
    // compiled).
    RecipeMarker {
        name: "finetune_from_conversations".into(),
        args: recipe_args,
    }
    .write_to(&job_dir)?;

    eprintln!("recipe finetune_from_conversations (v2)");
    eprintln!("job    {job_id}");
    eprintln!("dir    {}", job_dir.display());

    if args.background {
        eprintln!(
            "background mode is recognised but real detach lands in a \
             follow-up. For now, run without --background and use \
             `lamu-train cancel {job_id}` from another terminal."
        );
        return Ok(());
    }

    // Acquire the GPU lock — required for cross-process arbitration
    // before any training subprocess runs. Released on Drop after
    // execute() returns. `--allow-evict` waits up to 1h for an
    // existing exclusive to release, matching the legacy path.
    //
    // Lock acquisition errors must transition the job out of
    // `Running` so `lamu-train jobs` doesn't show a permanently-
    // stuck row after a lock timeout / permission failure.
    let lock = {
        let acq = async {
            if args.allow_evict {
                eprintln!("lock waiting for GPU release (--allow-evict, up to 1h)...");
                scheduler_lock::await_unlock(Duration::from_secs(3600))
                    .await
                    .context("await_unlock")?;
                scheduler_lock::acquire_exclusive(
                    format!("lamu-train:{job_id}"),
                    LockKind::Training,
                )
                .context("acquire_exclusive after wait")
            } else {
                scheduler_lock::acquire_exclusive(
                    format!("lamu-train:{job_id}"),
                    LockKind::Training,
                )
                .context("acquire_exclusive (use --allow-evict to wait)")
            }
        };
        match acq.await {
            Ok(l) => l,
            Err(e) => {
                crate::python_kill::unbind_current_job();
                if let Err(state_err) = jobs::write_state(&job_id, JobState::Failed) {
                    tracing::warn!(
                        "failed to record Failed state for {job_id} after lock error: {state_err}"
                    );
                }
                return Err(e);
            }
        }
    };
    eprintln!("lock acquired ({})", lock.path().display());

    // KILL-3: trap SIGTERM/ctrl-c → cancel token + killpg the child
    // group, then return so `lock` Drops (RAII unlocks the scheduler).
    install_cancel_handler(ctx.cancel.clone());

    let result = crate::framework::execute_plan(plan, ctx).await;
    drop(lock);
    crate::python_kill::unbind_current_job();

    match result {
        Ok(r) => {
            jobs::write_state(&job_id, JobState::Done)
                .with_context(|| format!("write Done state for {job_id}"))?;
            eprintln!(
                "done — {} stages, {} cache hits, {} misses, elapsed {:?}",
                r.n_stages, r.n_cache_hits, r.n_cache_misses, r.elapsed
            );
            Ok(())
        }
        Err(e) => {
            if let Err(state_err) = jobs::write_state(&job_id, JobState::Failed) {
                tracing::warn!(
                    "failed to record Failed state for {job_id} after plan error: {state_err}"
                );
            }
            Err(anyhow!("plan execution failed: {e}"))
        }
    }
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

fn build_dataset(args: &TrainArgs) -> Result<DatasetSource> {
    if args.from_conversations {
        // Step 7 materializes this to a JsonlPath; for v1 the CLI
        // accepts the flag and constructs the variant so the spec
        // round-trips cleanly. Use checked_sub so a since-window
        // larger than time-since-epoch (~55 years) saturates at 0
        // instead of panicking on SystemTime underflow.
        let cutoff = std::time::SystemTime::now()
            .checked_sub(args.since)
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Ok(DatasetSource::Conversations { since_ts: cutoff })
    } else {
        let path = args
            .dataset
            .clone()
            .ok_or_else(|| anyhow!("--dataset is required unless --from-conversations is set"))?;
        Ok(DatasetSource::JsonlPath { path })
    }
}

fn build_method(method: MethodArg, rank: u32, alpha: u32) -> Method {
    match method {
        MethodArg::Qlora => Method::QLora { rank, alpha },
        MethodArg::Lora => Method::Lora { rank, alpha },
        MethodArg::Full => Method::Full,
    }
}

fn pick_optimizer(opt: Option<OptimArg>, method: MethodArg) -> Optim {
    if let Some(o) = opt {
        return match o {
            OptimArg::Adamw => Optim::AdamW,
            OptimArg::Adamw8bit => Optim::AdamW8bit,
            OptimArg::Apollo => Optim::ApolloRank4,
            OptimArg::ApolloMini => Optim::ApolloMini,
        };
    }
    // Defaults pegged to memory profile of each method.
    match method {
        MethodArg::Qlora => Optim::ApolloMini,
        MethodArg::Lora => Optim::AdamW8bit,
        MethodArg::Full => Optim::AdamW,
    }
}

fn register_in_registry(name: &str, gguf_path: &Path, spec: &TrainSpec) -> Result<()> {
    use crate::registry;
    use crate::registry::{BackendType, Capability, ModelEntry, ModelFormat, ModelStatus};
    let registry_path = crate::config::registry_path();
    let entry = ModelEntry {
        name: name.into(),
        path: gguf_path.to_path_buf(),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp,
        arch: "trained".into(), // refined post-conversion in a future step
        params_b: 0.0,          // unknown until we parse GGUF
        quant: spec.quant.clone(),
        vram_mb: 0,
        context_max: spec.seq_len,
        capabilities: vec![Capability::Chat],
        notes: format!("trained from {} via blut", spec.base_model),
        status: ModelStatus::default(),
    };
    registry::add_entry(entry, &registry_path, true)
        .map_err(|e| anyhow!("registry update failed: {e}"))
}

#[cfg(test)]
mod footprint_resolve_tests {
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
        // The exact key the cli RESOLVES under for the joint default run.
        assert_eq!(d.key("lamquant_joint_codec").flat(), "lamquant_joint_codec|3|32|2");
    }

    /// Explicit tier/batch flow through to the key (so a tier-6 fullband
    /// run keys separately from a tier-3 run).
    #[test]
    fn explicit_tier_batch_flow_to_key() {
        let raw = serde_json::json!({ "tier": 6, "batch_size": 16 });
        let d = crate::broker::Drivers::from_args_json(&raw);
        assert_eq!((d.workers, d.batch, d.tier), (2, 16, 6));
        assert_eq!(d.key("lamquant_joint_codec").flat(), "lamquant_joint_codec|6|16|2");
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
        assert_eq!(d.key("lamquant_joint_codec").flat(), "lamquant_joint_codec|3|32|2");
    }
}
