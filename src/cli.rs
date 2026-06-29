// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
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
    /// Behind the off-by-default `tui` feature (1.0 is CLI-only; the cockpit
    /// returns in 1.1). Build with `--features tui` to enable.
    #[cfg(feature = "tui")]
    Tui {
        /// Headless self-check: build the cockpit + render every view to a test
        /// backend, exit 0 if all draw non-blank (no raw mode). For CI / smoke.
        #[arg(long, default_value_t = false)]
        check: bool,
    },
}

/// `blut p2p` subcommands. Behind the `p2p` feature.
#[cfg(feature = "p2p")]
#[derive(Subcommand, Debug)]
enum P2pCommand {
    /// Identity key management (Ed25519 + X25519).
    Keys {
        #[command(subcommand)]
        cmd: P2pKeysCommand,
    },
    /// Run as a COORDINATOR: bind a QUIC server, accept peers, hold the
    /// registry. Peers connect to this address. Runs until Ctrl-C.
    Serve {
        /// Listen address (host:port). 0.0.0.0:9320 by default.
        #[arg(long, default_value = "0.0.0.0:9320")]
        addr: String,
        /// Identity key file (defaults to the standard p2p key path).
        #[arg(long)]
        key: Option<std::path::PathBuf>,
    },
    /// Run as a worker PEER: connect to a coordinator and execute dispatched
    /// stages until the connection closes.
    Connect {
        /// Coordinator address (host:port).
        coordinator: String,
        /// The coordinator's public key (hex, 64 bytes = Ed25519 ‖ X25519) —
        /// required to verify dispatched tasks + seal results. Get it from
        /// `blut p2p keys show` on the coordinator.
        #[arg(long)]
        coordinator_pubkey: String,
        /// Identity key file (defaults to the standard p2p key path).
        #[arg(long)]
        key: Option<std::path::PathBuf>,
    },
    /// Peer registry: list / set-trust / remove.
    Peers {
        #[command(subcommand)]
        cmd: P2pPeersCommand,
    },
}

#[cfg(feature = "p2p")]
#[derive(Subcommand, Debug)]
enum P2pKeysCommand {
    /// Generate a fresh identity keypair (refuses to overwrite an existing one).
    Generate {
        #[arg(long)]
        key: Option<std::path::PathBuf>,
        /// Overwrite an existing key file.
        #[arg(long, default_value_t = false)]
        force: bool,
    },
    /// Print this node's public key (hex) + PeerId. Share the pubkey with peers.
    Show {
        #[arg(long)]
        key: Option<std::path::PathBuf>,
    },
}

#[cfg(feature = "p2p")]
#[derive(Subcommand, Debug)]
enum P2pPeersCommand {
    /// List known peers (id, trust, reputation, capabilities).
    List {
        #[arg(long)]
        json: bool,
    },
    /// Set a peer's trust level (anonymous | registered | trusted).
    Trust {
        /// Peer id (hex, or unique prefix).
        id: String,
        /// New trust level.
        level: String,
    },
    /// Remove a peer from the registry.
    Remove {
        /// Peer id (hex, or unique prefix).
        id: String,
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
enum LineageCommand {
    /// Show a job's ingredient lineage (input→output hashes, cache hits).
    Show {
        /// Job id (or unique prefix).
        id: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Trace a checkpoint's full UPSTREAM provenance by content-hash prefix:
    /// the git SHA, hardware, and input-hash chain that produced it.
    Trace {
        /// Output content-hash prefix (≥ 6 hex chars recommended).
        hash: String,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Rebuild the lineage index from the job dirs (the DB is a derived index —
    /// safe to delete + reindex).
    Reindex,
    /// FRESHNESS (G): is a job's output stale? Reports code-drift (the run's
    /// git SHA vs current HEAD — a STALE result means a re-run would
    /// re-execute, since the cache key includes code_sha). With
    /// `--data-version <v>`, also flags any produced artifact whose recorded
    /// `data_version` (in its metadata `extra`) differs from `<v>`.
    Freshness {
        /// Job id (or unique prefix).
        id: String,
        /// Current data/manifest version to check artifact `extra.data_version`
        /// against (optional — code-drift is always reported).
        #[arg(long)]
        data_version: Option<String>,
        /// Emit as JSON.
        #[arg(long)]
        json: bool,
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

#[derive(Subcommand, Debug)]
// `Run` carries the full launch config (many flags) while `Show`/`Best` are
// tiny — a one-shot parse, so the size spread is harmless (boxing would only
// fight clap's derive).
#[allow(clippy::large_enum_variant)]
enum HpoCommand {
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

/// BLUT CLI entrypoint. The recipe catalog is supplied by the caller as
/// a composed [`Registry`] (the binary — in a cookbook crate — registers
/// the cookbooks it ships and passes them here). This is the lib seam
/// that lets blut-core stay domain-agnostic: a bare blut engine binary
/// would pass an empty registry; the cookbook binaries pass theirs.
pub async fn run(reg: crate::framework::Registry) -> Result<()> {
    init_tracing();
    warn_if_stale_binary();
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Jobs { json }) => run_jobs(json),
        Some(Command::Cancel { id, grace }) => run_cancel(&id, grace).await,
        Some(Command::Log { id, tail, json }) => run_log(&id, tail, json),
        Some(Command::Runs { cmd }) => run_runs_cmd(cmd),
        Some(Command::Lineage { cmd }) => run_lineage_cmd(cmd),
        Some(Command::Hpo { cmd }) => run_hpo(&reg, cmd).await,
        Some(Command::Dag { job, json }) => run_dag(job, json),
        Some(Command::Compare { a, b }) => run_compare(&a, &b),
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
        Some(Command::P2p { cmd }) => run_p2p_cmd(&reg, cmd).await,
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
    }
}

/// Warn (once, at startup) if the running binary was built from a DIFFERENT
/// commit than its source tree's CURRENT HEAD — the "git pull, forgot to
/// rebuild/reinstall, silently ran the stale binary" trap. The in_ch /
/// warm-containment never-OOM fixes only go live after a rebuild; a human who
/// `git pull`s and runs the old `~/.cargo/bin/blut` would otherwise get the
/// stale admission/footprint/containment logic with no signal.
///
/// build.rs stamps the build-time hash (`BLUT_GIT_HASH`) + the source dir
/// (`BLUT_SRC_DIR`); this re-resolves that dir's live HEAD at RUNTIME and warns
/// on mismatch. SILENT when up to date, when the source tree is gone (binary
/// copied off the build box), when git is unavailable, or when the build was
/// not stamped (`unknown`) — a missing signal must never become noise or a
/// false alarm.
fn warn_if_stale_binary() {
    // `--version` / `--help` should be fast and clean: skip the git probe AND
    // the warning when the user only wants version/help (clap exits during
    // parse, so the stale notice would just be stderr noise atop the output).
    if std::env::args().any(|a| matches!(a.as_str(), "--version" | "-V" | "--help" | "-h")) {
        return;
    }
    let built = env!("BLUT_GIT_HASH");
    let src = env!("BLUT_SRC_DIR");
    if built == "unknown" || src.is_empty() {
        return;
    }
    let live = std::process::Command::new("git")
        .args(["-C", src, "rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    // Source tree gone / not a repo / no git → no trustworthy comparison; stay
    // silent rather than cry wolf.
    let Some(live) = live else { return };
    if live != built {
        // Deliberately NOT a `--path` hint: the `blut` binary is built from the
        // cookbook crate (blut-lamquant), not this engine crate (BLUT_SRC_DIR),
        // so a specific `--path` would point at the wrong directory. Keep it
        // generic — the operator knows how they installed.
        tracing::warn!(
            "blut binary is STALE: built from {built} but its source tree ({src}) is now \
             at {live} — this run uses OLD code (admission / footprint / containment logic \
             may predate the source). Rebuild + reinstall (`cargo install --force`, or \
             `cargo build` for a local checkout)."
        );
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
                println!("{}", serde_json::to_string_pretty(&arr)?);
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &serde_json::json!({ "name": s.name(), "outcome": o })
                    )?
                );
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

// ── P2P CLI (behind the `p2p` feature) ───────────────────────────────────────

#[cfg(feature = "p2p")]
mod p2p_cli {
    use super::*;
    use crate::p2p::crypto::KeyPair;
    use crate::p2p::peer::PeerId;
    use crate::p2p::registry::PeerRegistry;
    use crate::p2p::trust::TrustLevel;

    /// Default identity key path: `<data_dir>/p2p/identity.key`.
    pub fn default_key_path() -> Result<std::path::PathBuf> {
        Ok(paths::data_dir()?.join("p2p").join("identity.key"))
    }

    /// Default peer-registry path: `<data_dir>/p2p/peers.json`.
    pub fn default_registry_path() -> Result<std::path::PathBuf> {
        Ok(paths::data_dir()?.join("p2p").join("peers.json"))
    }

    /// Load the identity keypair from `path` (64-byte file), or error if absent.
    pub fn load_keypair(path: &std::path::Path) -> Result<KeyPair> {
        let bytes = std::fs::read(path).with_context(|| {
            format!("read identity key {} (run `blut p2p keys generate`)", path.display())
        })?;
        let arr: [u8; 64] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("identity key {} is not 64 bytes", path.display()))?;
        Ok(KeyPair::from_bytes(&arr))
    }

    /// Parse a trust level from a CLI string.
    pub fn parse_trust(s: &str) -> Result<TrustLevel> {
        match s.to_lowercase().as_str() {
            "anonymous" => Ok(TrustLevel::Anonymous),
            "registered" => Ok(TrustLevel::Registered),
            "trusted" => Ok(TrustLevel::Trusted),
            other => Err(anyhow!("unknown trust level '{other}' (anonymous|registered|trusted)")),
        }
    }

    /// Resolve a peer id from a hex string or unique prefix in the registry.
    /// Case-insensitive (PeerId displays lowercase hex; the user may paste any
    /// case).
    pub fn resolve_peer_id(reg: &PeerRegistry, prefix: &str) -> Result<PeerId> {
        let needle = prefix.to_lowercase();
        let matches: Vec<_> = reg
            .list()
            .into_iter()
            .filter(|p| p.id.to_string().to_lowercase().starts_with(&needle))
            .collect();
        match matches.as_slice() {
            [one] => Ok(one.id.clone()),
            [] => Err(anyhow!("no peer matches '{prefix}'")),
            _ => Err(anyhow!("'{prefix}' is ambiguous ({} peers match)", matches.len())),
        }
    }
}

#[cfg(feature = "p2p")]
async fn run_p2p_cmd(reg: &crate::framework::Registry, cmd: P2pCommand) -> Result<()> {
    use crate::p2p::crypto::KeyPair;
    use crate::p2p::peer::PeerId;
    use crate::p2p::registry::PeerRegistry;
    use p2p_cli::*;

    match cmd {
        P2pCommand::Keys { cmd } => match cmd {
            P2pKeysCommand::Generate { key, force } => {
                let path = key.map(Ok).unwrap_or_else(default_key_path)?;
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let kp = KeyPair::generate();
                // Write the SECRET key atomically with 0600 set AT CREATION (so
                // it is never momentarily world-readable). `create_new` is the
                // atomic refuse-to-overwrite — no exists()/write TOCTOU.
                use std::io::Write as _;
                let mut opts = std::fs::OpenOptions::new();
                opts.write(true);
                if force {
                    opts.create(true).truncate(true);
                } else {
                    opts.create_new(true);
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    opts.mode(0o600);
                }
                let mut f = opts.open(&path).map_err(|e| {
                    if e.kind() == std::io::ErrorKind::AlreadyExists {
                        anyhow!(
                            "identity key already exists at {} (pass --force to overwrite)",
                            path.display()
                        )
                    } else {
                        anyhow!("open identity key {}: {e}", path.display())
                    }
                })?;
                f.write_all(&kp.to_bytes())
                    .with_context(|| format!("write identity key {}", path.display()))?;
                // Durable before we print success — the key is the only copy.
                f.sync_all()
                    .with_context(|| format!("fsync identity key {}", path.display()))?;
                let pid = PeerId::from_pubkey(&kp.verifying);
                println!("identity written to {}", path.display());
                println!("peer id   {pid}");
                println!("pubkey    {}", p2p_pubkey_hex(&kp));
                Ok(())
            }
            P2pKeysCommand::Show { key } => {
                let path = key.map(Ok).unwrap_or_else(default_key_path)?;
                let kp = load_keypair(&path)?;
                let pid = PeerId::from_pubkey(&kp.verifying);
                println!("peer id   {pid}");
                println!("pubkey    {}", p2p_pubkey_hex(&kp));
                println!("(share the pubkey with peers: `blut p2p connect <coord> --coordinator-pubkey <hex>`)");
                Ok(())
            }
        },
        P2pCommand::Serve { addr, key } => {
            let path = key.map(Ok).unwrap_or_else(default_key_path)?;
            let kp = std::sync::Arc::new(load_keypair(&path)?);
            run_p2p_serve(addr, kp).await
        }
        P2pCommand::Connect { coordinator, coordinator_pubkey, key } => {
            let path = key.map(Ok).unwrap_or_else(default_key_path)?;
            let kp = load_keypair(&path)?;
            run_p2p_connect(reg, coordinator, coordinator_pubkey, kp).await
        }
        P2pCommand::Peers { cmd } => {
            let reg_path = default_registry_path()?;
            let mut registry = PeerRegistry::load(&reg_path)
                .map_err(|e| anyhow!("load peer registry: {e}"))?;
            match cmd {
                P2pPeersCommand::List { json } => {
                    if json {
                        let arr: Vec<_> = registry
                            .list()
                            .into_iter()
                            .map(|p| {
                                serde_json::json!({
                                    "id": p.id.to_string(),
                                    "trust": p.trust.label(),
                                    "reputation": p.reputation,
                                    "tasks_completed": p.tasks_completed,
                                    "tasks_failed": p.tasks_failed,
                                })
                            })
                            .collect();
                        println!("{}", serde_json::to_string_pretty(&arr)?);
                    } else {
                        let peers = registry.list();
                        if peers.is_empty() {
                            println!("(no peers registered)");
                        }
                        for p in peers {
                            println!(
                                "{}  trust={:<10} rep={:.2}  ok={} fail={}",
                                p.id.short(),
                                p.trust.label(),
                                p.reputation,
                                p.tasks_completed,
                                p.tasks_failed
                            );
                        }
                    }
                    Ok(())
                }
                P2pPeersCommand::Trust { id, level } => {
                    let pid = resolve_peer_id(&registry, &id)?;
                    let lvl = parse_trust(&level)?;
                    if registry.set_trust(&pid, lvl) {
                        registry.save().map_err(|e| anyhow!("save registry: {e}"))?;
                        println!("set {} trust = {}", pid.short(), lvl.label());
                        Ok(())
                    } else {
                        Err(anyhow!("peer {} not found", pid.short()))
                    }
                }
                P2pPeersCommand::Remove { id } => {
                    let pid = resolve_peer_id(&registry, &id)?;
                    if registry.remove(&pid) {
                        registry.save().map_err(|e| anyhow!("save registry: {e}"))?;
                        println!("removed {}", pid.short());
                        Ok(())
                    } else {
                        Err(anyhow!("peer {} not found", pid.short()))
                    }
                }
            }
        }
    }
}

/// Hex-encode a keypair's public bytes (64 = Ed25519 ‖ X25519) for sharing.
#[cfg(feature = "p2p")]
fn p2p_pubkey_hex(kp: &crate::p2p::crypto::KeyPair) -> String {
    let mut pubbytes = [0u8; 64];
    pubbytes[..32].copy_from_slice(kp.verifying.as_bytes());
    pubbytes[32..].copy_from_slice(kp.x25519_public.as_bytes());
    faster_hex::hex_string(&pubbytes)
}

/// Run as a coordinator: bind the QUIC server, accept peers, hold the registry.
#[cfg(feature = "p2p")]
async fn run_p2p_serve(
    addr: String,
    keypair: std::sync::Arc<crate::p2p::crypto::KeyPair>,
) -> Result<()> {
    use crate::p2p::dispatch::{DefaultDispatchPolicy, DispatchPolicy};
    use crate::p2p::registry::PeerRegistry;
    use crate::p2p::trust::DispatchMatrix;
    use crate::p2p::Coordinator;

    let sockaddr: std::net::SocketAddr = addr
        .parse()
        .with_context(|| format!("parse listen addr '{addr}'"))?;
    let reg_path = p2p_cli::default_registry_path()?;
    let registry = PeerRegistry::load(&reg_path).map_err(|e| anyhow!("load registry: {e}"))?;
    let dispatch: std::sync::Arc<dyn DispatchPolicy> =
        std::sync::Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default()));

    let coordinator = Coordinator::start(sockaddr, keypair.clone(), dispatch, registry)
        .await
        .map_err(|e| anyhow!("start coordinator: {e}"))?;
    let bound = coordinator.local_addr().map_err(|e| anyhow!("{e}"))?;
    let pid = crate::p2p::peer::PeerId::from_pubkey(&keypair.verifying);
    eprintln!("coordinator listening on {bound}");
    eprintln!("peer id   {pid}");
    eprintln!("pubkey    {}", p2p_pubkey_hex(&keypair));
    eprintln!("(peers: `blut p2p connect {bound} --coordinator-pubkey <pubkey>`)");
    eprintln!("Ctrl-C to stop.");

    // Persist the registry on shutdown so newly-handshaked peers survive a restart.
    tokio::signal::ctrl_c().await.ok();
    eprintln!("\nshutting down…");
    {
        let peers = coordinator.peers();
        let g = peers.read().await;
        if let Err(e) = g.save() {
            eprintln!("warning: failed to persist peer registry: {e}");
        }
    }
    coordinator.shutdown();
    Ok(())
}

/// Run as a worker peer: connect to the coordinator and execute dispatched tasks.
#[cfg(feature = "p2p")]
async fn run_p2p_connect(
    reg: &crate::framework::Registry,
    coordinator: String,
    coordinator_pubkey_hex: String,
    keypair: crate::p2p::crypto::KeyPair,
) -> Result<()> {
    use crate::p2p::dispatch::{DefaultDispatchPolicy};
    use crate::p2p::peer_exec::{run_peer_loop, CoordinatorKeys};
    use crate::p2p::transport::P2pClient;
    use crate::p2p::trust::DispatchMatrix;

    let sockaddr: std::net::SocketAddr = coordinator
        .parse()
        .with_context(|| format!("parse coordinator addr '{coordinator}'"))?;

    // Decode the coordinator's 64-byte public key (Ed25519 ‖ X25519).
    let mut pub64 = [0u8; 64];
    faster_hex::hex_decode(coordinator_pubkey_hex.as_bytes(), &mut pub64)
        .map_err(|e| anyhow!("invalid --coordinator-pubkey hex: {e}"))?;
    let verifying = ed25519_dalek::VerifyingKey::from_bytes(
        &pub64[..32].try_into().unwrap(),
    )
    .map_err(|e| anyhow!("invalid coordinator Ed25519 key: {e}"))?;
    let x_arr: [u8; 32] = pub64[32..].try_into().unwrap();
    let x25519_pub = x25519_dalek::PublicKey::from(x_arr);
    let coord_keys = CoordinatorKeys { verifying, x25519_pub };

    let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
    let work_root = crate::p2p::peer_exec::default_work_root();
    std::fs::create_dir_all(&work_root)
        .with_context(|| format!("create peer work root {}", work_root.display()))?;

    let client = P2pClient::with_coordinator_pin(
        std::sync::Arc::new(clone_keypair(&keypair)),
        {
            let mut k = [0u8; 32];
            k.copy_from_slice(verifying.as_bytes());
            k
        },
    );
    let (conn, my_id) = client
        .connect(sockaddr)
        .await
        .map_err(|e| anyhow!("connect to coordinator: {e}"))?;
    eprintln!("connected to coordinator {coordinator} as peer {my_id}");
    eprintln!("waiting for dispatched tasks (Ctrl-C to stop)…");

    run_peer_loop(&conn, &keypair, &coord_keys, reg, &policy, &work_root)
        .await
        .map_err(|e| anyhow!("peer loop: {e}"))?;
    eprintln!("coordinator connection closed.");
    Ok(())
}

/// `KeyPair` doesn't derive Clone (it holds secrets); reconstruct from bytes
/// when we need a second owner (the client pin path takes an Arc).
#[cfg(feature = "p2p")]
fn clone_keypair(kp: &crate::p2p::crypto::KeyPair) -> crate::p2p::crypto::KeyPair {
    crate::p2p::crypto::KeyPair::from_bytes(&kp.to_bytes())
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
                println!("{}", serde_json::to_string_pretty(&obj)?);
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
                println!(
                    "{}",
                    serde_json::to_string_pretty(&stats).map_err(|e| anyhow!("{e}"))?
                );
                return Ok(());
            }
            let (th, tm) = stats.totals();
            println!("{:<32} {:>6} {:>6} {:>7}", "ingredient", "hits", "miss", "hit%");
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
        println!("no ingredient lineage (job has no framework status events).");
        return Ok(());
    }
    for n in &nodes {
        let inp = n.input_hash.as_deref().unwrap_or("-");
        let out = n.output_hash.as_deref().unwrap_or("-");
        let short = |h: &str| h.chars().take(12).collect::<String>();
        if n.cached {
            println!(
                "  {:>2} {:<28} [CACHE HIT {}]",
                n.node_idx,
                n.stage,
                short(out)
            );
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

fn run_lineage_cmd(cmd: LineageCommand) -> Result<()> {
    match cmd {
        LineageCommand::Show { id, json } => run_lineage(&id, json),
        LineageCommand::Trace { hash, json } => run_lineage_trace(&hash, json),
        LineageCommand::Reindex => run_lineage_reindex(),
        LineageCommand::Freshness {
            id,
            data_version,
            json,
        } => run_lineage_freshness(&id, data_version, json),
    }
}

/// FRESHNESS (G): report whether a job's output is stale. Code-drift is
/// always reported (the run's git SHA vs HEAD — STALE ⇒ a re-run
/// re-executes because the cache key includes code_sha). With
/// `--data-version`, also flag artifacts whose recorded `extra.data_version`
/// differs from the supplied current value.
fn run_lineage_freshness(id_query: &str, data_version: Option<String>, json: bool) -> Result<()> {
    let job_id = crate::jobs::resolve_job_id(id_query).map_err(|e| anyhow!("{e}"))?;
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("{e}"))?;
    let code = db.code_freshness(&job_id).map_err(|e| anyhow!("{e}"))?;
    let short = |h: &str| h.get(..12).unwrap_or(h).to_string();

    // Optional data-version freshness: compare each produced artifact's
    // recorded `extra.data_version` against the supplied current value.
    let mut data_stale: Vec<(String, String)> = Vec::new(); // (stage, recorded)
    let mut data_checked = 0usize;
    if let Some(current) = data_version.as_deref() {
        for rec in crate::framework::lineage::scan_artifacts(&job_id).map_err(|e| anyhow!("{e}"))? {
            if let Some(v) = rec.meta.extra.get("data_version").and_then(|v| v.as_str()) {
                data_checked += 1;
                if v != current {
                    data_stale.push((
                        rec.meta.produced_by_stage.clone().unwrap_or_default(),
                        v.to_string(),
                    ));
                }
            }
        }
    }

    if json {
        let v = serde_json::json!({
            "job_id": job_id,
            "code": code,
            "data_version_current": data_version,
            "data_artifacts_checked": data_checked,
            "data_stale": data_stale.iter()
                .map(|(s, r)| serde_json::json!({ "stage": s, "recorded": r }))
                .collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }

    match &code {
        crate::lineage_db::CodeFreshness::Fresh { git_sha } => {
            println!(
                "job {job_id}: code FRESH (built at HEAD {})",
                short(git_sha)
            )
        }
        crate::lineage_db::CodeFreshness::Stale { built_sha, head } => println!(
            "job {job_id}: code STALE (built at {}, HEAD is {} — a re-run re-executes)",
            short(built_sha),
            short(head)
        ),
        crate::lineage_db::CodeFreshness::Unknown => {
            println!("job {job_id}: code UNKNOWN (no recorded git SHA to compare)")
        }
    }
    if let Some(current) = &data_version {
        if data_stale.is_empty() {
            println!("data: FRESH ({data_checked} artifact(s) at data_version {current})");
        } else {
            for (stage, recorded) in &data_stale {
                println!(
                    "data: STALE — {stage} built from data_version {recorded}, current is {current}"
                );
            }
        }
    }
    Ok(())
}

/// Reproducibility query: the full upstream provenance chain that produced a
/// checkpoint, by content-hash prefix, from the LineageDB.
fn run_lineage_trace(hash: &str, json: bool) -> Result<()> {
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("{e}"))?;
    let matches = db.find_artifacts(hash).map_err(|e| anyhow!("{e}"))?;
    let target = match matches.first() {
        None => {
            return Err(anyhow!(
                "no indexed artifact with content-hash prefix '{hash}' \
                 (older runs predate the index — `blut lineage reindex`)"
            ));
        }
        Some(a) => {
            if matches.len() > 1 {
                eprintln!(
                    "note: {} artifacts match '{hash}'; tracing the most recent",
                    matches.len()
                );
            }
            a.content_hash.clone()
        }
    };
    let chain = db.trace(&target).map_err(|e| anyhow!("{e}"))?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&chain).map_err(|e| anyhow!("serialize trace: {e}"))?
        );
        return Ok(());
    }
    let short = |h: &str| h.get(..16).unwrap_or(h).to_string();
    println!(
        "provenance trace for {} — {} hop(s), upstream:",
        short(&target),
        chain.len()
    );
    for (i, step) in chain.iter().enumerate() {
        let a = &step.artifact;
        println!(
            "  [{i}] {} :: {} = {}",
            a.stage_name,
            a.kind,
            short(&a.content_hash)
        );
        if let Some(run) = &step.run {
            // `?` for unrecorded hardware — NOT `0` (which reads as "zero RAM").
            let ram = run
                .ram_gib
                .map(|g| format!("{g}G"))
                .unwrap_or_else(|| "?".into());
            let vram = run
                .vram_mib
                .map(|m| format!("{m}M"))
                .unwrap_or_else(|| "?".into());
            println!(
                "      job={} recipe={} git={} ram={ram} vram={vram} outcome={}",
                run.job_id,
                run.recipe,
                run.git_sha.as_deref().unwrap_or("?"),
                run.outcome.as_deref().unwrap_or("?"),
            );
        }
    }
    Ok(())
}

/// Rebuild the lineage index from the job dirs. The DB is a DERIVED index, so
/// this is always safe (idempotent ingest); use it after deleting `lineage.db`
/// or to backfill runs that predate the index.
fn run_lineage_reindex() -> Result<()> {
    let jobs_root = crate::paths::jobs_dir()?;
    let (mut indexed, mut skipped) = (0u32, 0u32);
    let rd = std::fs::read_dir(&jobs_root)
        .with_context(|| format!("read jobs dir {}", jobs_root.display()))?;
    {
        for e in rd.flatten() {
            let Some(job_id) = e.file_name().to_str().map(String::from) else {
                continue;
            };
            // Recipe identity comes from the run marker; legacy/bare-spawn jobs
            // have none → skip (nothing to attribute the run to).
            let Ok(marker) = RecipeMarker::read_from(&e.path()) else {
                skipped += 1;
                continue;
            };
            let outcome = crate::jobs::read_state(&job_id)
                .map(|s| format!("{s:?}").to_lowercase())
                .unwrap_or_else(|_| "unknown".into());
            match crate::lineage_db::ingest_job(&job_id, &marker.name, &outcome) {
                Ok(()) => indexed += 1,
                Err(err) => {
                    tracing::warn!("reindex {job_id}: {err}");
                    skipped += 1;
                }
            }
        }
    }
    eprintln!("reindexed {indexed} run(s), skipped {skipped}");
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

/// The box-fit RAM budget (GiB) for a scheduler / executor that runs cells
/// concurrently. MIRRORS the executor's Phase-5 sizing (cli.rs `run_hpo` /
/// `launch_compiled_plan`): `MemTotal − floor`, clamped `>= 1`. Box-fit TOTAL
/// (minus the standard reserve), NOT live-free — the per-cell broker admission
/// already nets out transient other-consumers via `MemAvailable`; this budget
/// bounds the SUM of concurrently SCHEDULED cells to the box. A `0` total
/// (non-Linux / sandbox where `/proc/meminfo` is unreadable) ⇒ `None`: caller
/// degrades to the per-cell gate alone (the old behaviour), never a bogus cap.
fn scheduler_box_fit_budget_gib() -> Option<u32> {
    let snap = crate::broker::ResourceSnapshot::probe();
    if snap.mem_total_gb > 0.0 {
        Some((snap.mem_total_gb - crate::broker::admission::DEFAULT_FLOOR_GIB).max(1.0) as u32)
    } else {
        None
    }
}

/// Run one scheduled cell under a shared cross-cell RAM semaphore so the SUM of
/// concurrently-running cells can't overcommit the box (never-OOM-the-BOX for
/// the parallel partition backfill). MIRRORS the `ParallelExecutor`'s per-node
/// memory admission (`executor::run_node`): acquire `footprint_gib` permits
/// (GiB units, matching `NodeEnv::memory`), CLAMPED to the budget so a single
/// cell larger than the whole box runs ALONE instead of deadlocking, hold the
/// permit for the cell's ENTIRE run, and release it on drop AFTER `run`
/// completes so the next queued cell can proceed.
///
/// Defense-in-depth: this bounds the scheduled-cell SUM; the per-cell broker
/// admission inside `launch_compiled_plan` still gates on LIVE free RAM
/// (incl. non-scheduler consumers) — both stay in force.
///
/// `budget_gib == 0` is treated as "no budget known" (the probe failed): run
/// ungated, exactly as before this slice. A non-zero budget always admits at
/// least 1 permit (`.max(1)`), so a `footprint_gib == 0` cell can't slip a
/// 0-permit no-op past the gate.
async fn gated_cell_run<F, T>(
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
/// are never-OOM-gated + scheduler-arbitrated exactly like a normal recipe run.
/// Layer the broad `KillOnNaN` safety net UNDER an HPO policy. `with_control`
/// REPLACES the executor's default `KillOnNaN`, so wiring an HPO policy raw
/// would drop the payload-wide non-finite kill (HPO policies only watch their
/// objective key; TPE doesn't kill on divergence at all). `[KillOnNaN, hpo]`
/// keeps the safety net active — order is load-bearing (KillOnNaN first
/// short-circuits, so a doomed step never consumes the HPO policy's spawn slot).
fn with_nan_safety(
    hpo: std::sync::Arc<dyn crate::framework::control::ControlPolicy>,
) -> std::sync::Arc<dyn crate::framework::control::ControlPolicy> {
    std::sync::Arc::new(crate::framework::control::CompositePolicy::new(vec![
        std::sync::Arc::new(crate::framework::control::KillOnNaN),
        hpo,
    ]))
}

async fn run_hpo(reg: &crate::framework::Registry, cmd: HpoCommand) -> Result<()> {
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
    } = cmd
    else {
        unreachable!("non-Run HpoCommand variants dispatched above")
    };

    // Base args (the fixed part; search dims overlay each trial).
    let base_args: serde_json::Value =
        serde_json::from_str(&args).map_err(|e| anyhow!("--args is not valid JSON: {e}"))?;

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
    let mut ctx = ExecCtx::new(job_dir.clone());

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
    {
        let snap = crate::broker::ResourceSnapshot::probe();
        if snap.mem_total_gb > 0.0 {
            let box_fit =
                (snap.mem_total_gb - crate::broker::admission::DEFAULT_FLOOR_GIB).max(1.0) as u32;
            ctx = ctx.with_memory_budget(box_fit);
        }
    }
    ctx = ctx.with_launch_target(launch_target);
    ctx = ctx.with_fb_warm(crate::broker::Drivers::from_args_json(&base_args).warm);
    if shared_cache {
        if let Some(global) = crate::framework::CacheHandle::default_global_path() {
            std::fs::create_dir_all(&global)
                .with_context(|| format!("create global cache dir {}", global.display()))?;
            let cache_handle = (*ctx.cache).clone().with_global(global);
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
    }
    .write_to(&job_dir)?;
    crate::jobs::write_state(&job_id, JobState::Running)
        .with_context(|| format!("write Running state for {job_id}"))?;
    crate::python_kill::bind_current_job(job_id.clone());
    install_cancel_handler(ctx.cancel.clone());

    // Admission gate on a SINGLE trial's footprint — the executor's per-stage
    // memory admission gates concurrency ACROSS trials, so the box can't OOM
    // even with the full fan-out in flight (never-OOM-the-box, unchanged).
    if let Err(reason) = crate::broker::gate(&format!("hpo '{name}'"), &footprint) {
        crate::python_kill::unbind_current_job();
        let _ = crate::jobs::write_state(&job_id, JobState::Failed);
        return Err(anyhow!("{reason}"));
    }
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
            eprintln!(
                "hpo done: {} trials ran (job {job_id}). Leaderboard: `blut hpo show {job_id}`; \
                 winning config: `blut hpo best {job_id}`.",
                trials.len()
            );
            Ok(())
        }
        Err(e) => {
            let _ = crate::jobs::write_state(&job_id, JobState::Failed);
            Err(anyhow!("hpo plan execution failed: {e}"))
        }
    }
}

/// Resolve the HPO job to inspect: an explicit id (via `jobs::resolve_job_id`,
/// so a prefix works) or — when omitted — the most recent job carrying an
/// `hpo.json` manifest (job ids are timestamp-monotonic, sorted ascending).
fn resolve_hpo_job(job: Option<String>) -> Result<(String, crate::hpo::HpoManifest)> {
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
fn run_hpo_show(job: Option<String>, json: bool) -> Result<()> {
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
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "job": id,
                "recipe": manifest.recipe,
                "algo": manifest.algo,
                "metric": manifest.metric,
                "mode": manifest.mode,
                "trials": arr,
            }))
            .map_err(|e| anyhow!("serialize leaderboard: {e}"))?
        );
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
fn run_hpo_best(job: Option<String>, json: bool) -> Result<()> {
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
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::Value::Object(overlay_obj))
                .map_err(|e| anyhow!("serialize overlay: {e}"))?
        );
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

#[derive(Subcommand, Debug)]
enum PartitionCommand {
    /// Declare + persist a partition set: `define <recipe> <name> --dim
    /// corpus=dataset_a,corpus_x --dim fold=0,1,2`.
    Define {
        recipe: String,
        name: String,
        /// One axis per flag: `--dim axis=v1,v2,v3` (repeatable).
        #[arg(long = "dim", value_name = "AXIS=v1,v2")]
        dim: Vec<String>,
    },
    /// List all defined partition sets.
    List,
    /// Per-cell materialization status (done / pending) for a set.
    Status { recipe: String, name: String },
    /// Run the recipe for every NOT-yet-materialized cell (`--force` = all),
    /// recording per-cell status. Cells run sequentially.
    Backfill {
        recipe: String,
        name: String,
        #[arg(long, default_value_t = false)]
        force: bool,
        /// Base recipe args (the fixed part; each cell overlays its axis values).
        #[arg(long, default_value = "{}")]
        args: String,
        #[arg(long, default_value = "local")]
        launcher: String,
    },
}

/// Apply a partition cell's `axis=value` overrides onto base recipe args:
/// `args[axis] = <scalar>` (int / float / bool, else the verbatim string).
fn apply_cell_overrides(base: &serde_json::Value, overrides: &[String]) -> serde_json::Value {
    let mut obj = base.as_object().cloned().unwrap_or_default();
    for ov in overrides {
        if let Some((k, v)) = ov.split_once('=') {
            let val = if let Ok(i) = v.parse::<i64>() {
                serde_json::json!(i)
            } else if let Ok(f) = v.parse::<f64>() {
                serde_json::json!(f)
            } else if let Ok(b) = v.parse::<bool>() {
                serde_json::json!(b)
            } else {
                serde_json::json!(v)
            };
            obj.insert(k.to_string(), val);
        }
    }
    serde_json::Value::Object(obj)
}

async fn run_partition(reg: &crate::framework::Registry, cmd: PartitionCommand) -> Result<()> {
    use crate::config::partition::{PartitionDim, PartitionSet, PartitionStatus};
    match cmd {
        PartitionCommand::Define { recipe, name, dim } => {
            if reg.find(&recipe).is_none() {
                return Err(anyhow!("recipe '{recipe}' not in catalog"));
            }
            let dims: Vec<PartitionDim> = dim
                .iter()
                .map(|d| {
                    let (axis, vals) = d
                        .split_once('=')
                        .ok_or_else(|| anyhow!("--dim '{d}' must be axis=v1,v2"))?;
                    let values: Vec<String> = vals
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .collect();
                    Ok::<_, anyhow::Error>(PartitionDim {
                        axis: axis.to_string(),
                        values,
                    })
                })
                .collect::<Result<_>>()?;
            let set = PartitionSet { name, recipe, dims };
            let cells = set.validate().map_err(|e| anyhow!("{e}"))?;
            let path = set.save().map_err(|e| anyhow!("{e}"))?;
            println!(
                "defined partition '{}/{}' — {cells} cells → {}",
                set.recipe,
                set.name,
                path.display()
            );
        }
        PartitionCommand::List => {
            let sets = PartitionSet::list().map_err(|e| anyhow!("{e}"))?;
            if sets.is_empty() {
                println!("(no partition sets defined)");
            }
            for (recipe, name) in sets {
                println!("{recipe}/{name}");
            }
        }
        PartitionCommand::Status { recipe, name } => {
            let set = PartitionSet::load(&recipe, &name).map_err(|e| anyhow!("{e}"))?;
            let done = set.statuses().map_err(|e| anyhow!("{e}"))?;
            let cells = set.cells();
            let n_done = cells
                .iter()
                .filter(|c| {
                    done.get(&c.key)
                        .is_some_and(PartitionStatus::is_materialized)
                })
                .count();
            println!("{recipe}/{name} — {n_done}/{} materialized", cells.len());
            for c in &cells {
                let st = match done.get(&c.key) {
                    Some(s) if s.is_materialized() => format!("done (job {})", s.job_id),
                    Some(s) => format!("{} (job {})", s.outcome, s.job_id),
                    None => "pending".to_string(),
                };
                println!("  {:<32} {st}", c.key);
            }
        }
        PartitionCommand::Backfill {
            recipe,
            name,
            force,
            args,
            launcher,
        } => {
            let set = PartitionSet::load(&recipe, &name).map_err(|e| anyhow!("{e}"))?;
            let base: serde_json::Value = serde_json::from_str(&args)
                .map_err(|e| anyhow!("--args is not valid JSON: {e}"))?;
            let launch_target: crate::config::launcher::LaunchTarget =
                launcher.parse().map_err(|e| anyhow!("{e}"))?;
            let targets = set.backfill_targets(force).map_err(|e| anyhow!("{e}"))?;
            if targets.is_empty() {
                println!("{recipe}/{name}: nothing to backfill (all cells materialized)");
                return Ok(());
            }
            // Phase-G PARALLEL scheduler: round-robin cells across the
            // launcher's device set, then run each device's cells SEQUENTIALLY
            // (its per-device lock would REJECT a second concurrent cell — so
            // never start two cells on one GPU), with the devices running
            // CONCURRENTLY. Result: at most one cell per GPU at a time, up to
            // `n_dev` cells in flight. capacity=1 → one chain → sequential
            // (byte-identical to the old loop). record_status appends atomically,
            // so concurrent writes from different device-chains are safe.
            //
            // NB the broker RAM-admission gate runs PER cell against LIVE free
            // RAM; two cells launching ~simultaneously could both pass it on the
            // SAME snapshot before either's usage registers. Cross-cell RAM
            // coordination is now DONE (not a future slice): a SHARED box-fit
            // RAM semaphore (`mem_sem` below) bounds the SUM of concurrently
            // SCHEDULED cells to the box, mirroring the `ParallelExecutor`'s
            // per-node memory budget. The per-cell gate stays in force
            // (defense-in-depth: it handles live free RAM incl. non-scheduler
            // consumers). On 1 GPU there's no concurrency, so the semaphore is
            // acquired/released serially — byte-identical to the old loop.
            // Concurrency is bounded by the launcher's CAPACITY: `device_set()`
            // is `0..Launcher::capacity()` by default (see
            // `config::launcher::Launcher::{capacity,device_set}`), with
            // `$BLUT_SCHED_DEVICES` layered on top ONLY as a SUBSET override.
            // So `n_dev == launcher.capacity()` in the common (no-override)
            // case, and a subset otherwise — never more than capacity.
            let launcher = crate::config::launcher::launcher_for(launch_target);
            let devices = launcher.device_set();
            if devices.is_empty() {
                return Err(anyhow!(
                    "launcher for {launch_target:?} reports no devices — cannot backfill"
                ));
            }
            // At most `launcher.capacity()` cells run concurrently; with a
            // `$BLUT_SCHED_DEVICES` subset, fewer. `device_set().len()` IS that
            // bound (it can only narrow `0..capacity()`, never widen it).
            debug_assert!(devices.len() <= launcher.capacity().max(1));
            let n_dev = devices.len();
            eprintln!(
                "backfill {recipe}/{name}: {} cell(s) across {} device(s) {devices:?}",
                targets.len(),
                n_dev
            );
            // Bucket cells per device (round-robin), preserving cell order.
            let mut per_device: Vec<Vec<_>> = (0..n_dev).map(|_| Vec::new()).collect();
            for (i, cell) in targets.into_iter().enumerate() {
                per_device[i % n_dev].push(cell);
            }
            // Cross-cell RAM admission (never-OOM): size the box-fit budget ONCE
            // (MemTotal − floor, same source the executor uses) and share one
            // semaphore across all device-chains. Each cell acquires its
            // (clamped) footprint in GiB permits for its whole run, so the SUM
            // of concurrent cells can't exceed the box. `0` ⇒ probe failed ⇒
            // ungated (per-cell broker gate alone), the pre-slice behaviour.
            let budget_gib = scheduler_box_fit_budget_gib().unwrap_or(0);
            let mem_sem =
                std::sync::Arc::new(tokio::sync::Semaphore::new(budget_gib.max(1) as usize));
            let (set, recipe, base) = (&set, &recipe, &base);
            let chains = per_device.into_iter().enumerate().map(|(d, cells)| {
                let dev = devices[d];
                let mem_sem = mem_sem.clone();
                async move {
                    let (mut ok, mut failed) = (0usize, 0usize);
                    for cell in cells {
                        let cell_args = apply_cell_overrides(base, &cell.overrides);
                        eprintln!("[gpu {dev}] cell {}", cell.key);
                        // Per-cell footprint (RAM GiB) for the cross-cell gate:
                        // the SAME `recipe_footprint` the per-cell broker
                        // admission resolves on, rounded UP to whole GiB (never
                        // under-bill). Clamp+acquire happens in `gated_cell_run`.
                        let footprint = recipe_footprint(recipe, &cell_args);
                        let footprint_gib =
                            footprint.ram_bytes.div_ceil(crate::broker::footprint::GIB) as u32;
                        let cell_run = run_one_recipe(
                            reg,
                            recipe,
                            cell_args,
                            None,
                            false,
                            launch_target,
                            Some(dev),
                            false,
                        );
                        let (outcome, job_id) = match gated_cell_run(
                            mem_sem.clone(),
                            footprint_gib,
                            budget_gib,
                            cell_run,
                        )
                        .await
                        {
                            Ok(jid) => {
                                ok += 1;
                                ("done", jid)
                            }
                            Err(e) => {
                                eprintln!("[gpu {dev}] cell {} FAILED: {e}", cell.key);
                                failed += 1;
                                ("failed", String::new())
                            }
                        };
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs() as i64)
                            .unwrap_or(0);
                        if let Err(e) = set.record_status(&PartitionStatus {
                            key: cell.key.clone(),
                            job_id,
                            outcome: outcome.to_string(),
                            recorded_at: now,
                        }) {
                            // A lost status write would silently re-run a
                            // completed cell next backfill — warn, don't swallow.
                            eprintln!(
                                "warning: could not record status for cell {}: {e}",
                                cell.key
                            );
                        }
                    }
                    (ok, failed)
                }
            });
            let totals = futures::future::join_all(chains).await;
            let ok: usize = totals.iter().map(|(o, _)| o).sum();
            let failed: usize = totals.iter().map(|(_, f)| f).sum();
            eprintln!("backfill done — {ok} materialized, {failed} failed");
            if failed > 0 {
                return Err(anyhow!(
                    "{failed} cell(s) failed (re-run `partition backfill` to retry only those)"
                ));
            }
        }
    }
    Ok(())
}

/// `blut compare <A> <B>` (E4): provenance + a side-by-side FINAL-metric panel
/// (with Δ) + the GPU-saturation summary, all from the queryable metric store.
fn run_compare(a: &str, b: &str) -> Result<()> {
    use std::collections::BTreeMap;
    let ja = crate::jobs::resolve_job_id(a).map_err(|e| anyhow!("{e}"))?;
    let jb = crate::jobs::resolve_job_id(b).map_err(|e| anyhow!("{e}"))?;
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("open lineage.db: {e}"))?;

    let show_run = |id: &str| -> String {
        match db.get_run(id) {
            Ok(Some(r)) => format!(
                "{id}  recipe={} outcome={} git={}",
                r.recipe,
                r.outcome.as_deref().unwrap_or("?"),
                r.git_sha
                    .as_deref()
                    .map(|s| &s[..s.len().min(8)])
                    .unwrap_or("?"),
            ),
            _ => format!("{id}  (no provenance row)"),
        }
    };
    println!("A  {}", show_run(&ja));
    println!("B  {}", show_run(&jb));

    let ma: BTreeMap<String, f64> = db
        .final_metrics(&ja)
        .map_err(|e| anyhow!("{e}"))?
        .into_iter()
        .collect();
    let mb: BTreeMap<String, f64> = db
        .final_metrics(&jb)
        .map_err(|e| anyhow!("{e}"))?
        .into_iter()
        .collect();
    let keys: std::collections::BTreeSet<&String> = ma.keys().chain(mb.keys()).collect();
    if keys.is_empty() {
        println!("\n(no metrics recorded for either run)");
    } else {
        println!(
            "\n{:<18} {:>12} {:>12} {:>12}",
            "metric", "A", "B", "Δ(B−A)"
        );
        for k in keys {
            let fmt = |v: Option<&f64>| v.map(|x| format!("{x:.4}")).unwrap_or_else(|| "—".into());
            let delta = match (ma.get(k), mb.get(k)) {
                (Some(x), Some(y)) => format!("{:+.4}", y - x),
                _ => "—".into(),
            };
            println!(
                "{:<18} {:>12} {:>12} {:>12}",
                k,
                fmt(ma.get(k)),
                fmt(mb.get(k)),
                delta
            );
        }
    }

    // GPU saturation (the owner's first-class metric).
    let sat = |id: &str| db.gpu_saturation(id, 50.0).ok().flatten();
    if let (Some(sa), Some(sb)) = (sat(&ja), sat(&jb)) {
        println!(
            "\nGPU saturation   A {:.1}% (wasted {:.0}%)   B {:.1}% (wasted {:.0}%)",
            sa.saturation,
            sa.wasted * 100.0,
            sb.saturation,
            sb.wasted * 100.0
        );
    }
    Ok(())
}

/// `blut dag <job> [--json]` — render a job's DAG: per-node status + edges,
/// built from the persisted `plan.json` + the live `status.jsonl` (+ HPO trial
/// attribution when present). No daemon; re-run to refresh.
fn run_dag(job: Option<String>, json: bool) -> Result<()> {
    let job_id = match job {
        Some(q) => crate::jobs::resolve_job_id(&q).map_err(|e| anyhow!("{e}"))?,
        // `list_jobs` sorts ascending by timestamp-monotonic id; prefer the most
        // recent job that ACTUALLY has a `plan.json` (a pre-v0.20 run has none,
        // so blindly taking the last job would error on a stale job).
        None => crate::jobs::list_jobs()
            .map_err(|e| anyhow!("list jobs: {e}"))?
            .into_iter()
            .rev()
            .map(|s| s.id)
            .find(|id| {
                crate::paths::job_dir(id)
                    .ok()
                    .and_then(|d| crate::framework::graph::PlanGraph::read_from(&d))
                    .is_some()
            })
            .ok_or_else(|| anyhow!("no jobs with a plan.json (run a recipe first)"))?,
    };
    let snap = crate::framework::graph_snapshot(&job_id).map_err(|e| anyhow!("{e}"))?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&snap).map_err(|e| anyhow!("serialize snapshot: {e}"))?
        );
        return Ok(());
    }
    // Tally per-status for a one-line header.
    let mut counts: std::collections::BTreeMap<&'static str, u32> =
        std::collections::BTreeMap::new();
    for n in &snap.nodes {
        *counts.entry(n.status.as_str()).or_default() += 1;
    }
    let tally = counts
        .iter()
        .map(|(s, c)| format!("{c} {s}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "dag {} (job {}) — {} nodes, {} edges [{tally}]",
        snap.name,
        snap.job,
        snap.nodes.len(),
        snap.edges.len()
    );
    println!(
        "{:<4} {:<22} {:<8} {:<8} {:<6} preds  detail",
        "idx", "ingredient", "status", "elapsed", "trial"
    );
    for n in &snap.nodes {
        let preds = snap
            .edges
            .iter()
            .filter(|e| e.to == n.idx)
            .map(|e| e.from.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let preds = if preds.is_empty() {
            "─".to_string()
        } else {
            preds
        };
        let elapsed = n
            .elapsed_secs
            .map(|s| format!("{s:.1}s"))
            .unwrap_or_else(|| "─".into());
        let trial = n
            .hpo
            .as_ref()
            .map(|h| format!("t{}", h.trial_id))
            .unwrap_or_else(|| "─".into());
        // For an HPO node the overlay (the diff that defines the trial) is the
        // useful detail; otherwise fall back to the args summary.
        let detail = match &n.hpo {
            Some(h) => h
                .overlay
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" "),
            None => n.args_summary.clone(),
        };
        println!(
            "{:<4} {:<22} {:<8} {:<8} {:<6} {:<6} {}",
            n.idx,
            n.stage_name,
            n.status.as_str(),
            elapsed,
            trial,
            preds,
            detail
        );
    }
    Ok(())
}

async fn run_recipe(reg: &crate::framework::Registry, cmd: RecipeCommand) -> Result<()> {
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
                        println!("✓ '{}' compiles + kind-checks ({n} ingredient(s)).", recipe.name);
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
async fn run_one_recipe(
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
async fn launch_compiled_plan(
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
    let footprint = recipe_footprint(name, plan.exec_view().recipe_args);

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
            Err(anyhow!("plan execution failed: {e}"))
        }
    }
}

/// Best-effort: record a finished sweep combo into the global sweep-completion
/// index (fingerprint → final-stage output hash + sidecar), so a later sweep
/// re-run skips it. The final stage is the last `output.metadata.json` sidecar
/// (lineage scans stage dirs in order). A failure here must NOT fail the run —
/// the index is a skip optimization, never a correctness gate.
fn record_sweep_completion(fp: crate::framework::ContentHash, job_id: &str) {
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
fn stage_idx_of(sidecar: &std::path::Path) -> u32 {
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
async fn run_recipe_sweep(
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
fn project_args(mut config: serde_json::Value, key: &str) -> serde_json::Value {
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
fn warn_dotless_overrides(items: &[String], flag: &str, subtree_key: &str) {
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

fn run_jobs(json: bool) -> Result<()> {
    let jobs = jobs::list_jobs()?;
    if json {
        // JobSummary derives Serialize — emit the array verbatim so a
        // script/agent gets the same data the table renders.
        let out =
            serde_json::to_string_pretty(&jobs).map_err(|e| anyhow!("serialize jobs: {e}"))?;
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

#[cfg(all(test, feature = "p2p"))]
mod p2p_cli_tests {
    use super::{
        Cli, Command, P2pCommand, P2pKeysCommand, P2pPeersCommand,
    };
    use clap::Parser;

    fn p2p_of(argv: &[&str]) -> P2pCommand {
        match Cli::try_parse_from(argv).expect("parse").command {
            Some(Command::P2p { cmd }) => cmd,
            other => panic!("expected p2p, got {other:?}"),
        }
    }

    #[test]
    fn keys_generate_parses() {
        match p2p_of(&["blut", "p2p", "keys", "generate", "--force"]) {
            P2pCommand::Keys { cmd: P2pKeysCommand::Generate { force, .. } } => assert!(force),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn serve_defaults_addr() {
        match p2p_of(&["blut", "p2p", "serve"]) {
            P2pCommand::Serve { addr, .. } => assert_eq!(addr, "0.0.0.0:9320"),
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn connect_requires_pubkey() {
        // Missing --coordinator-pubkey must fail to parse.
        assert!(Cli::try_parse_from(["blut", "p2p", "connect", "1.2.3.4:9320"]).is_err());
        // With it, parses.
        match p2p_of(&[
            "blut", "p2p", "connect", "1.2.3.4:9320", "--coordinator-pubkey", "deadbeef",
        ]) {
            P2pCommand::Connect { coordinator, coordinator_pubkey, .. } => {
                assert_eq!(coordinator, "1.2.3.4:9320");
                assert_eq!(coordinator_pubkey, "deadbeef");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn peers_subcommands_parse() {
        match p2p_of(&["blut", "p2p", "peers", "list", "--json"]) {
            P2pCommand::Peers { cmd: P2pPeersCommand::List { json } } => assert!(json),
            other => panic!("got {other:?}"),
        }
        match p2p_of(&["blut", "p2p", "peers", "trust", "abc123", "trusted"]) {
            P2pCommand::Peers { cmd: P2pPeersCommand::Trust { id, level } } => {
                assert_eq!(id, "abc123");
                assert_eq!(level, "trusted");
            }
            other => panic!("got {other:?}"),
        }
    }

    #[test]
    fn trust_level_parses() {
        use super::p2p_cli::parse_trust;
        use crate::p2p::trust::TrustLevel;
        assert_eq!(parse_trust("trusted").unwrap(), TrustLevel::Trusted);
        assert_eq!(parse_trust("ANONYMOUS").unwrap(), TrustLevel::Anonymous);
        assert!(parse_trust("bogus").is_err());
    }

    #[test]
    fn keypair_file_roundtrips_and_is_0600() {
        // generate (atomic create_new + 0600) → load → same identity. Then a
        // second generate without --force must refuse.
        use super::p2p_cli::load_keypair;
        use crate::p2p::crypto::KeyPair;
        use crate::p2p::peer::PeerId;
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("p2p/identity.key");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        let kp = KeyPair::generate();
        // Mirror the CLI's atomic 0600 write.
        use std::io::Write as _;
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        opts.open(&path).unwrap().write_all(&kp.to_bytes()).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "secret key must be owner-only");
        }

        let loaded = load_keypair(&path).unwrap();
        assert_eq!(
            PeerId::from_pubkey(&loaded.verifying),
            PeerId::from_pubkey(&kp.verifying),
            "loaded identity matches generated"
        );

        // create_new on an existing path is the atomic refuse-overwrite.
        let mut opts2 = std::fs::OpenOptions::new();
        opts2.write(true).create_new(true);
        assert_eq!(
            opts2.open(&path).unwrap_err().kind(),
            std::io::ErrorKind::AlreadyExists
        );
    }
}
