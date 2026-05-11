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

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use blut::scheduler_lock::{self, LockKind};
use blut::{
    backend::{StatusFn, TrainBackend},
    convert,
    jobs::{self, JobState},
    paths,
    protocol::StatusUpdate,
    python_backend::PythonTrainBackend,
    spec::{DatasetSource, Method, Optim, TrainSpec},
};

#[derive(Parser, Debug)]
#[command(name = "lamu-train", version, about = "Local fine-tuning")]
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
    Jobs,
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
}

#[derive(Subcommand, Debug)]
enum RecipeCommand {
    /// List the recipe catalog.
    List,
    /// Print one recipe's args JSON schema.
    Show {
        /// Recipe name (e.g. finetune_from_conversations).
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
    Rm {
        name: String,
    },
    /// Print metadata for one dataset as JSON.
    Show {
        name: String,
    },
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

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Train(args)) => run_train(args).await,
        Some(Command::Jobs) => run_jobs(),
        Some(Command::Cancel { id, grace }) => run_cancel(&id, grace).await,
        Some(Command::Log { id, tail }) => run_log(&id, tail),
        Some(Command::Data { cmd }) => run_data(cmd),
        Some(Command::Auto) => run_auto().await,
        Some(Command::Policy { cmd }) => run_policy(cmd),
        Some(Command::Recipe { cmd }) => run_recipe(cmd).await,
        None => run_train(cli.train_args).await,
    }
}

async fn run_recipe(cmd: RecipeCommand) -> Result<()> {
    use blut::framework::{ExecCtx, SequentialExecutor};
    use blut::recipes::recipe::{find as find_recipe, RECIPES};
    match cmd {
        RecipeCommand::List => {
            println!("{:<32} {}", "name", "description");
            for r in RECIPES {
                println!("{:<32} {}", r.name, r.description);
            }
        }
        RecipeCommand::Show { name } => {
            let r = find_recipe(&name)
                .ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
            println!("name        : {}", r.name);
            println!("description : {}", r.description);
            let schema = (r.args_schema_fn)();
            println!(
                "args schema :\n{}",
                serde_json::to_string_pretty(&schema)
                    .unwrap_or_else(|e| format!("(serialize error: {e})"))
            );
        }
        RecipeCommand::Run { name, args, shared_cache } => {
            let r = find_recipe(&name)
                .ok_or_else(|| anyhow!("recipe '{name}' not in catalog"))?;
            let raw: serde_json::Value = serde_json::from_str(&args)
                .with_context(|| format!("parse --args as JSON: {args}"))?;
            let plan = (r.compile_fn)(raw)
                .map_err(|e| anyhow!("recipe compile failed: {e}"))?;

            let job_id = blut::jobs::new_job_id();
            let job_dir = blut::paths::job_dir(&job_id)?;
            let mut ctx = ExecCtx::new(job_dir.clone());
            if shared_cache {
                if let Some(global) = blut::framework::CacheHandle::default_global_path() {
                    std::fs::create_dir_all(&global).with_context(|| {
                        format!("create global cache dir {}", global.display())
                    })?;
                    let cache_handle = (*ctx.cache).clone().with_global(global);
                    ctx.cache = std::sync::Arc::new(cache_handle);
                }
            }
            eprintln!("recipe {name}");
            eprintln!("job    {job_id}");
            eprintln!("dir    {}", job_dir.display());
            let result = SequentialExecutor::execute(plan, ctx)
                .await
                .map_err(|e| anyhow!("plan execution failed: {e}"))?;
            eprintln!(
                "done — {} stages, {} cache hits, {} misses, elapsed {:?}",
                result.n_stages, result.n_cache_hits, result.n_cache_misses, result.elapsed
            );
        }
    }
    Ok(())
}

async fn run_auto() -> Result<()> {
    use blut::{conversations, policy};

    let pol = policy::load().context("load policy")?;
    let (now_unix, now_local_min) = policy::current_clock();
    let lock_held = blut::scheduler_lock::check_unlocked().is_err();
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
        policy::Decision::Run { base, method, since } => {
            println!(
                "auto: triggering training (new_turns={new_turns}, threshold={})",
                pol.threshold_new_turns
            );
            let bin = std::env::current_exe()
                .context("locate own binary for auto-spawn")?;
            let auto_name = format!("auto-{}", blut::jobs::new_job_id());
            let mut cmd = tokio::process::Command::new(&bin);
            cmd.arg(&auto_name)
                .arg("--from-conversations")
                .arg("--since").arg(&since)
                .arg("--base").arg(&base)
                .arg("--method").arg(&method)
                .arg("--background")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .kill_on_drop(false);
            match cmd.spawn() {
                Ok(mut child) => {
                    let pid = child.id().unwrap_or(0);
                    println!(
                        "auto: spawned lamu-train pid={pid} as '{auto_name}'"
                    );
                    // Update last_train_ts at spawn time. Failed
                    // runs still count toward cooldown — better
                    // than retrying immediately on every cron tick
                    // when something's broken.
                    let mut updated = pol.clone();
                    updated.last_train_ts = now_unix;
                    updated.last_train_n_turns = new_turns;
                    if let Err(e) = policy::save(&updated) {
                        tracing::warn!("failed to update last_train_ts: {e}");
                    }
                    // Reap the zombie when training finishes; the
                    // cron-driven `auto` exits while the child runs.
                    // Log non-zero exit so train-auto.log shows the
                    // outcome instead of just the spawn line.
                    tokio::spawn(async move {
                        match child.wait().await {
                            Ok(status) if !status.success() => {
                                tracing::warn!(
                                    "auto-train (pid={pid}) exited with {status}"
                                );
                            }
                            Ok(status) => {
                                tracing::info!(
                                    "auto-train (pid={pid}) exited cleanly: {status}"
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "auto-train (pid={pid}) wait failed: {e}"
                                );
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
    use blut::policy;
    match cmd {
        PolicyCommand::Show => {
            let p = policy::load().context("load policy")?;
            print!(
                "{}",
                toml::to_string_pretty(&p)
                    .map_err(|e| anyhow!("serialize policy: {e}"))?
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
            println!(
                "*/30 * * * * {exe} auto >> {} 2>&1",
                log_path.display()
            );
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
    use blut::datasets_db;
    let conn = datasets_db::open()?;
    match cmd {
        DataCommand::List => {
            let rows = datasets_db::list(&conn)?;
            if rows.is_empty() {
                println!("no datasets registered.");
                return Ok(());
            }
            println!(
                "{:<24} {:<12} {:>10} {:<16} {}",
                "name", "kind", "examples", "sha256[:8]", "path"
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
            let rec =
                datasets_db::record_from_jsonl(&name, &path, &kind, None)?;
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
        DataCommand::Show { name } => {
            match datasets_db::get_by_name(&conn, &name)? {
                Some(rec) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&rec)
                            .unwrap_or_else(|e| format!("serialize error: {e}"))
                    );
                }
                None => return Err(anyhow!("no dataset named '{name}'")),
            }
        }
    }
    Ok(())
}

/// Register a JSONL dataset in the datasets registry. Best-effort:
/// callers handle failure by logging + continuing. Used by
/// auto-registration after `--from-conversations` materialization.
fn register_dataset(
    name: &str,
    path: &PathBuf,
    kind: &str,
    metadata: Option<String>,
) -> Result<()> {
    let conn = blut::datasets_db::open()?;
    let rec = blut::datasets_db::record_from_jsonl(name, path, kind, metadata)?;
    blut::datasets_db::add(&conn, &rec)?;
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
    use tracing_subscriber::{fmt, EnvFilter};
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,blut=info,hyper=warn,reqwest=warn")),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}

async fn run_train(args: TrainArgs) -> Result<()> {
    let output_name = args
        .output_name
        .clone()
        .ok_or_else(|| anyhow!("output-name is required (positional). See `lamu-train --help`."))?;

    let dataset_src = build_dataset(&args)?;
    let optimizer = pick_optimizer(args.optim, args.method);
    let method = build_method(args.method, args.rank, args.alpha);

    let job_id = jobs::new_job_id();
    let job_dir = paths::job_dir(&job_id)
        .with_context(|| format!("create job dir for {job_id}"))?;
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
            let stats = blut::conversations::dump_to_jsonl(
                args.since,
                &out_path,
            )
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
            tracing::warn!(
                "failed to persist status to {}: {}",
                job_id_for_cb,
                e
            );
        }
        match &u {
            StatusUpdate::Step {
                step, total, loss, lr, vram_mb,
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
        }
    });

    let result = backend.run(spec.clone(), on_status).await;

    drop(lock); // release GPU before convert + register; convert is
                // CPU-bound and llama.cpp tools don't need the card.

    match result {
        Ok(artifact) => {
            jobs::write_state(&job_id, JobState::Done)?;
            eprintln!(
                "trained in {:?}, final_loss={:.4}, ckpt={}",
                artifact.elapsed,
                artifact.final_loss,
                artifact.checkpoint_dir.display()
            );

            if !args.no_convert {
                eprintln!("converting to GGUF ({})...", args.quant);
                let gguf = convert::convert_to_gguf(
                    &artifact.checkpoint_dir,
                    &output_name,
                    &args.quant,
                )
                .await
                .context("convert_to_gguf")?;
                eprintln!("gguf  {}", gguf.display());
                register_in_registry(&output_name, &gguf, &spec)?;
                eprintln!("registry updated; `mcp__local-llm__query model={output_name}` should work.");
            } else {
                eprintln!("--no-convert: HF checkpoint left at {}", artifact.checkpoint_dir.display());
            }
        }
        Err(e) => {
            jobs::write_state(&job_id, JobState::Failed)?;
            return Err(anyhow!(e));
        }
    }
    Ok(())
}

fn run_jobs() -> Result<()> {
    let jobs = jobs::list_jobs()?;
    if jobs.is_empty() {
        println!("no jobs.");
        return Ok(());
    }
    println!("{:<24} {:<10} {:<6} {:<24} {}", "id", "state", "pid", "output", "last");
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

async fn run_cancel(id_query: &str, grace: Duration) -> Result<()> {
    let id = jobs::resolve_job_id(id_query)?;
    eprintln!("cancelling {id} (grace {grace:?})...");
    jobs::cancel_job(&id, grace).await?;
    eprintln!("cancelled.");
    Ok(())
}

fn run_log(id_query: &str, tail: usize) -> Result<()> {
    let id = jobs::resolve_job_id(id_query)?;
    let updates = jobs::read_status(&id)?;
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
        let path = args.dataset.clone().ok_or_else(|| {
            anyhow!("--dataset is required unless --from-conversations is set")
        })?;
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

fn register_in_registry(
    name: &str,
    gguf_path: &PathBuf,
    spec: &TrainSpec,
) -> Result<()> {
    use blut::registry;
    use blut::registry::{
        BackendType, Capability, ModelEntry, ModelFormat, ModelStatus,
    };
    let registry_path = blut::config::registry_path();
    let entry = ModelEntry {
        name: name.into(),
        path: gguf_path.clone(),
        format: ModelFormat::Gguf,
        backend: BackendType::LlamaCpp,
        arch: "trained".into(), // refined post-conversion in a future step
        params_b: 0.0,           // unknown until we parse GGUF
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

