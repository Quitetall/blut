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

/// Top-level `about` line. The cockpit lives in the `blut-tui` sidecar crate
/// (ADR 0083 M2): a cookbook binary that links it opens it on bare invocation;
/// the bare engine reaches it via `blut tui` (external dispatch).
const CLI_ABOUT: &str = "BLUT — typed-DAG orchestrator for local ML training. Subcommands: recipe, \
     jobs, log, cancel, plan, cache, footprint, partition, schedule, sensor, \
     policy, tui.";

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
    /// Data-quality checks (ADR 0091): report a run's guardrail breaches.
    Checks {
        #[command(subcommand)]
        cmd: ChecksCommand,
    },
    /// Dataset catalog (ADR 0100): search / show / tag / rebuild a read-only
    /// projection over the datasets registry + lineage.
    Catalog {
        #[command(subcommand)]
        cmd: CatalogCommand,
    },
    /// Ecosystem connectors (ADR 0112): list the registered connector stages +
    /// their typed I/O kinds, and validate the registry.
    Connectors {
        #[command(subcommand)]
        cmd: ConnectorsCommand,
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
    /// Immutable, tenant-scoped `dataset://<name>@<version>` bindings (ADR 0090).
    Dataset {
        #[command(subcommand)]
        cmd: DatasetCommand,
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
    /// Model registry (ADR 0090): bind a name + alias to a checkpoint hash and
    /// move that binding under audit — `model://<name>@<alias>`. Same verbs as
    /// `plan`.
    Model {
        #[command(subcommand)]
        cmd: ModelCommand,
    },
    /// Experiment registry views over lineage (`experiment://<recipe>/<run>`).
    #[command(name = "exp", visible_alias = "experiment")]
    Experiment {
        #[command(subcommand)]
        cmd: ExperimentCommand,
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
    /// Cloud compute queue (ADR 0082 / 0067 T3.1) — submit a job, get the
    /// result back, billed on compute. Behind the off-by-default `cloud`
    /// feature (implies `p2p`).
    #[cfg(feature = "cloud")]
    Cloud {
        #[command(subcommand)]
        cmd: CloudCommand,
    },
    /// Open the canonical interactive training cockpit (ratatui). The
    /// single, complete cockpit: recipe launcher + live jobs/log/system
    /// panels + run history / leaderboard / compare / checkpoints /
    /// presets / live-metrics / reset views. Keys: ↑↓ select, Enter log,
    /// c cancel, R recipe picker, J/L/Y/H/B/K/P/M/X switch views, q quit.
    ///
    /// The cockpit lives in the `blut-tui` SIDECAR crate (ADR 0083 M2): a
    /// cookbook binary that links it opens it in-process over the live
    /// registry; the bare engine execs the `blut-tui` binary from PATH
    /// (engine-generic views, no cookbook recipes).
    Tui {
        /// Headless self-check: build the cockpit + render every view to a test
        /// backend, exit 0 if all draw non-blank (no raw mode). For CI / smoke.
        #[arg(long, default_value_t = false)]
        check: bool,
    },
    /// A cargo-style EXTERNAL subcommand (ADR 0083): `blut <cmd> …` with no
    /// built-in match execs `blut-<cmd>` from PATH with the remaining args — the
    /// seam by which `blut web`/`blut notify` (and eventually `blut tui`) route
    /// to their SIDECAR binaries the engine never links.
    #[command(external_subcommand)]
    External(Vec<String>),
}

#[derive(Subcommand, Debug)]
enum ConnectorsCommand {
    /// List the registered connectors (name + typed input→output kinds).
    List {
        /// Validate the registry: exit non-zero if any connector declares a
        /// non-connector I/O kind.
        #[arg(long)]
        check: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum CatalogCommand {
    /// Re-project the catalog index from the datasets registry + lineage
    /// (verifies each entry's schema against its source); prints the count.
    Rebuild,
    /// Filter the catalog: `modality: fs: kind: tag: hash:` terms (AND).
    Search {
        /// e.g. `"modality:eeg fs:256 tag:sleep"`.
        query: String,
        /// A cloud-surfaced view — exclude clinical/PHI entries (ADR 0061).
        #[arg(long)]
        cloud: bool,
        #[arg(long)]
        json: bool,
    },
    /// Show one entry's schema + lineage neighborhood (producing stage +
    /// downstream consumers) + tags. Fails if the entry's schema has drifted
    /// from its source manifest.
    Show {
        /// Dataset `name` or `name@version`.
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Append a tag to a dataset (persists across a rebuild).
    Tag { name: String, tag: String },
}

#[derive(Subcommand, Debug)]
enum ChecksCommand {
    /// Report the data-quality breaches (blocked + advisory) recorded in a
    /// run's lineage. `blut checks report <job> | grep DataQuality`.
    Report {
        /// Job id (or unique prefix).
        job: String,
        /// Emit the breaches as a JSON array (for scripts/agents).
        #[arg(long)]
        json: bool,
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
        /// Force all declared async-I/O lanes onto the synchronous Inline
        /// fallback for this resumed execution.
        #[arg(long, default_value_t = false)]
        sync_io: bool,
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
    /// Publish a `.json` PlanSpec to the deployment registry (ADR 0085):
    /// typecheck fail-closed, then store an immutable fingerprint-keyed row.
    Publish {
        /// Path to a `.json` PlanSpec.
        spec: std::path::PathBuf,
        /// Owning tenant (default `shared`; `restricted` = clinical/PHI).
        #[arg(long, default_value = "shared")]
        tenant: String,
    },
    /// Promote a published fingerprint onto a named pointer
    /// (`registry://plan@<name>`), recording the move in the audit trail.
    Promote {
        /// The deployment fingerprint to promote.
        fingerprint: String,
        /// Target pointer, e.g. `registry://plan@prod`.
        pointer: String,
        /// Acting tenant — must match the deployment's tenant.
        #[arg(long, default_value = "shared")]
        tenant: String,
    },
    /// Roll a pointer back to its previous target atomically.
    Rollback {
        /// Pointer, e.g. `registry://plan@prod`.
        pointer: String,
        #[arg(long, default_value = "shared")]
        tenant: String,
    },
    /// Print a pointer's deployment audit trail (oldest first).
    History {
        /// Pointer, e.g. `registry://plan@prod`.
        pointer: String,
        #[arg(long, default_value = "shared")]
        tenant: String,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommand {
    /// Register a checkpoint hash under a model name (immutable candidate row).
    Register {
        /// The checkpoint sha256 (lowercase 64-hex — as the cache/lineage emits).
        hash: String,
        /// Model name, e.g. `encoder-v1`.
        name: String,
        /// Owning tenant (default `default`; `restricted` = clinical/PHI).
        #[arg(long, default_value = "default")]
        tenant: String,
        /// Freeform provenance note (recipe, run id, …).
        #[arg(long)]
        source: Option<String>,
    },
    /// Promote a registered hash onto `model://<name>@<alias>`, recording the
    /// move in the audit trail. Promotion to a GOVERNED alias (default `prod`,
    /// override `$BLUT_MODEL_GOVERNED_ALIASES`) fail-closes on a governance gate
    /// (`--gate-cmd` / `$BLUT_MODEL_GATE_CMD`) + `--change-id` (ADR 0090) — so
    /// `@prod` can never point at an unvetted checkpoint.
    Promote {
        /// The checkpoint hash to promote.
        hash: String,
        /// Target pointer, e.g. `model://encoder-v1@prod`.
        pointer: String,
        /// Acting tenant — must match the model's tenant.
        #[arg(long, default_value = "default")]
        tenant: String,
        /// Change-request id for a governed-alias promotion (fed to the gate).
        #[arg(long)]
        change_id: Option<String>,
        /// Governance gate argv (e.g. `"python …/pccp_gate.py --candidate {hash}
        /// --model {name} --change-id {change_id}"`). Falls back to
        /// `$BLUT_MODEL_GATE_CMD`. Required to promote onto a governed alias.
        #[arg(long)]
        gate_cmd: Option<String>,
        /// Hard wall-clock bound for the governance process.
        #[arg(long, default_value = "5m", value_parser = parse_duration)]
        gate_timeout: Duration,
    },
    /// Roll an alias back to its previous target atomically.
    Rollback {
        /// Pointer, e.g. `model://encoder-v1@prod`.
        pointer: String,
        #[arg(long, default_value = "default")]
        tenant: String,
    },
    /// Resolve `model://<name>@<alias>` to the checkpoint hash it points at.
    Resolve {
        /// Pointer, e.g. `model://encoder-v1@prod`.
        pointer: String,
        #[arg(long, default_value = "default")]
        tenant: String,
    },
    /// Print an alias's audit trail (oldest first).
    History {
        /// Pointer, e.g. `model://encoder-v1@prod`.
        pointer: String,
        #[arg(long, default_value = "default")]
        tenant: String,
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
    /// to clear every key for a recipe. Admission remains conservative: it then uses the
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
enum DatasetCommand {
    /// Pin an existing `blut data add` source to an immutable version URI.
    Pin {
        /// Existing raw dataset-registry name.
        source: String,
        /// Immutable target, e.g. `dataset://tuh@v3`.
        uri: String,
        #[arg(long, default_value = "default")]
        tenant: String,
    },
    /// Resolve a version to its hash-verified local path.
    Resolve {
        uri: String,
        #[arg(long, default_value = "default")]
        tenant: String,
        /// Intended placement. Restricted datasets resolve only for `local`.
        #[arg(long, default_value = "local")]
        launcher: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
enum ExperimentCommand {
    /// Compare the two newest runs of a recipe in one tenant.
    Compare {
        name: String,
        #[arg(long, default_value = "default")]
        tenant: String,
        #[arg(long)]
        json: bool,
    },
    /// Resolve `experiment://<recipe>/<run>` to its tenant-scoped lineage row.
    Resolve {
        uri: String,
        #[arg(long, default_value = "default")]
        tenant: String,
        #[arg(long)]
        json: bool,
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

fn parse_duration(s: &str) -> Result<Duration, String> {
    humantime::parse_duration(s).map_err(|e| format!("{e}"))
}

/// `blut connectors list [--check]` (ADR 0112) — enumerate connector stages +
/// their typed I/O kinds; `--check` validates the registry (non-zero on a
/// non-connector kind leaking into an integration graph).
/// Cargo-style external-subcommand dispatch (ADR 0083): `blut <cmd> <args…>`
/// with no built-in match execs `blut-<cmd>` from `PATH`, forwarding the
/// remaining args and propagating its exit status. This is the seam by which
/// the SIDECAR binaries the engine deliberately never links — `blut-web`
/// (Leptos+WASM dashboard), `blut-notify`, and eventually `blut-tui` — are
/// reachable as first-class `blut` subcommands without the engine growing an
/// in-process server or a dylib-plugin loader (ADR 0034 charter).
fn run_external(argv: Vec<String>) -> Result<()> {
    let (name, rest) = argv
        .split_first()
        .ok_or_else(|| anyhow!("empty external subcommand"))?;
    // ALLOWLIST the name (cargo's own convention for `cargo-<cmd>`): only
    // `[A-Za-z0-9_-]`. Strictly safer than blocklisting separators — no path
    // component, escape, or platform-specific separator can survive, so the child
    // is always a plain `blut-<name>` resolved on PATH, never a path.
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(anyhow!("unknown subcommand `{name}`"));
    }
    let bin = format!("blut-{name}");
    // Inherits the parent's stdio (no capture) — a sidecar's own output reaches
    // the terminal directly; diagnostics that must survive a pipe go to stderr.
    let status = std::process::Command::new(&bin)
        .args(rest)
        .status()
        .map_err(|e| {
            anyhow!(
                "`{bin}` could not be executed — `blut {name}` dispatches to the \
                 `{bin}` sidecar binary (ADR 0083); install it or check PATH: {e}"
            )
        })?;
    if !status.success() {
        return Err(anyhow!(
            "`{bin}` exited with {}",
            exit_status_reason(&status)
        ));
    }
    Ok(())
}

/// Human-readable failure reason for a child `ExitStatus` — the code when it
/// exited normally, or the terminating signal (Unix) so an OOM-killed sidecar
/// reports e.g. `signal 9` rather than an opaque "signal".
fn exit_status_reason(status: &std::process::ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("status {code}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return format!("signal {sig}");
        }
    }
    "an unknown status".to_string()
}

fn run_connectors_cmd(cmd: ConnectorsCommand) -> Result<()> {
    match cmd {
        ConnectorsCommand::List { check, json } => {
            let list = crate::connectors::list();
            if json {
                emit_json(&list)?;
            } else {
                for d in &list {
                    println!("{:<20} {:<24} → {}", d.name, d.input_kind, d.output_kind);
                }
            }
            if check {
                let problems = crate::connectors::check();
                if !problems.is_empty() {
                    for p in &problems {
                        eprintln!("connector check FAILED: {p}");
                    }
                    return Err(anyhow!(
                        "{} connector(s) declare a non-connector kind",
                        problems.len()
                    ));
                }
                eprintln!("connectors: registry healthy ({} connectors)", list.len());
            }
            Ok(())
        }
    }
}

/// `blut catalog {rebuild,search,show,tag}` (ADR 0100) — a read-only projection
/// over the datasets registry + lineage, with a persisted tags table.
fn run_catalog_cmd(cmd: CatalogCommand) -> Result<()> {
    use crate::catalog;
    let datasets = crate::datasets_db::open().map_err(|e| anyhow!("{e}"))?;
    let lineage = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("{e}"))?;
    let tags = catalog::open_tags(&catalog::catalog_db_path().map_err(|e| anyhow!("{e}"))?)
        .map_err(|e| anyhow!("{e}"))?;

    match cmd {
        CatalogCommand::Rebuild => {
            // Re-project + verify each entry against its source manifest.
            let records = crate::datasets_db::list(&datasets).map_err(|e| anyhow!("{e}"))?;
            let mut n = 0usize;
            for r in &records {
                let entry = catalog::project(r, &lineage).map_err(|e| anyhow!("{e}"))?;
                catalog::verify_consistency(&entry, r).map_err(|e| anyhow!("{e}"))?;
                n += 1;
            }
            println!(
                "catalog: {n} entr{} projected + verified",
                if n == 1 { "y" } else { "ies" }
            );
            Ok(())
        }
        CatalogCommand::Search { query, cloud, json } => {
            let q = catalog::CatalogQuery::parse(&query).map_err(|e| anyhow!("{e}"))?;
            let entries = catalog::build_index(&datasets, &lineage).map_err(|e| anyhow!("{e}"))?;
            let tags_of = |n: &str| catalog::tags_for(&tags, n).unwrap_or_default();
            let hits = catalog::search(&entries, tags_of, &q, cloud);
            if json {
                emit_json(&hits)?;
            } else if hits.is_empty() {
                println!("no catalog entries match {query:?}");
            } else {
                for e in hits {
                    let fs = e
                        .schema
                        .fs
                        .map(|f| f.to_string())
                        .unwrap_or_else(|| "-".into());
                    println!(
                        "{:<24} {:<8} modality={} fs={} kind={}",
                        e.version
                            .as_ref()
                            .map(|v| format!("{}@{v}", e.name))
                            .unwrap_or_else(|| e.name.clone()),
                        &e.hash.get(..8).unwrap_or(&e.hash),
                        e.schema.modality.as_deref().unwrap_or("-"),
                        fs,
                        e.kind,
                    );
                }
            }
            Ok(())
        }
        CatalogCommand::Show { name, json } => {
            let bare = name.split_once('@').map(|(n, _)| n).unwrap_or(&name);
            let record = crate::datasets_db::get_by_name(&datasets, &name)
                .map_err(|e| anyhow!("{e}"))?
                .or_else(|| {
                    crate::datasets_db::get_by_name(&datasets, bare)
                        .ok()
                        .flatten()
                })
                .ok_or_else(|| anyhow!("no dataset '{name}' in the registry"))?;
            let entry = catalog::project(&record, &lineage).map_err(|e| anyhow!("{e}"))?;
            // Fail-closed consistency: a drifted schema is an error, not a serve.
            catalog::verify_consistency(&entry, &record).map_err(|e| anyhow!("{e}"))?;
            let entry_tags = catalog::tags_for(&tags, &entry.name).map_err(|e| anyhow!("{e}"))?;
            if json {
                emit_json(&serde_json::json!({ "entry": entry, "tags": entry_tags }))?;
            } else {
                println!(
                    "name       : {}{}",
                    entry.name,
                    entry
                        .version
                        .as_ref()
                        .map(|v| format!("@{v}"))
                        .unwrap_or_default()
                );
                println!("kind       : {}", entry.kind);
                println!("hash       : {}", entry.hash);
                println!(
                    "modality   : {}",
                    entry.schema.modality.as_deref().unwrap_or("-")
                );
                println!(
                    "fs         : {}",
                    entry
                        .schema
                        .fs
                        .map(|f| f.to_string())
                        .unwrap_or_else(|| "-".into())
                );
                println!(
                    "channels   : {}",
                    entry
                        .schema
                        .channels
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "-".into())
                );
                println!("clinical   : {}", entry.clinical);
                println!(
                    "produced_by: {}",
                    entry.produced_by.as_deref().unwrap_or("-")
                );
                println!("consumers  : {}", entry.consumers.len());
                println!("tags       : {}", entry_tags.join(", "));
            }
            Ok(())
        }
        CatalogCommand::Tag { name, tag } => {
            let bare = name
                .split_once('@')
                .map(|(n, _)| n.to_string())
                .unwrap_or(name);
            catalog::add_tag(&tags, &bare, &tag).map_err(|e| anyhow!("{e}"))?;
            println!("tagged {bare} += {tag}");
            Ok(())
        }
    }
}

/// `blut checks report <job>` — replay a run's `status.jsonl` for its
/// data-quality breaches (blocked + advisory) and print a grep-friendly report
/// (ADR 0091). `--json` emits the breach array.
fn run_checks_cmd(cmd: ChecksCommand) -> Result<()> {
    match cmd {
        ChecksCommand::Report { job, json } => {
            let breaches = crate::checks::report_job(&job)?;
            if json {
                emit_json(&breaches)?;
            } else {
                print!("{}", crate::checks::render_report(&breaches));
            }
            Ok(())
        }
    }
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

/// The cockpit seam (ADR 0083 M2): the engine carries NO terminal-UI code, so
/// a binary that wants the in-process cockpit (a cookbook binary linking the
/// `blut-tui` sidecar crate) hands its entry points here —
/// `blut::cli::run_with_tui(reg, Some(blut_tui::hook()))`. Plain `fn` pointers
/// keep this a data seam, not a widget API: the trait-shaped cookbook-TUI
/// contract stays [`crate::framework::CookbookTui`] (owner-locked 2026-07-12).
pub struct TuiHook {
    /// Run the interactive console to completion (owns the terminal).
    pub console: ConsoleFn,
    /// Headless self-check: render every view to a test backend (CI / smoke).
    pub check: fn(crate::framework::Registry) -> Result<()>,
}

/// The console entry a [`TuiHook`] carries: registry in, boxed future out
/// (a plain `fn` pointer — the sidecar's `run_console_loop` wrapped in a pin).
pub type ConsoleFn = fn(
    std::sync::Arc<crate::framework::Registry>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>>>>;

/// BLUT CLI entrypoint. The recipe catalog is supplied by the caller as
/// a composed [`crate::framework::cookbook::Registry`] (the binary — in a cookbook crate — registers
/// the cookbooks it ships and passes them here). This is the library seam that
/// keeps the engine domain-agnostic; the engine crate itself has no binary.
/// No cockpit attached — `blut tui` execs the `blut-tui` sidecar from PATH.
pub async fn run(reg: crate::framework::Registry) -> Result<()> {
    run_with_tui(reg, None).await
}

/// [`run`] with an optional in-process cockpit (see [`TuiHook`]).
pub async fn run_with_tui(reg: crate::framework::Registry, tui: Option<TuiHook>) -> Result<()> {
    init_tracing();
    warn_if_stale_binary();
    // The engine owns the built-in `checks` cookbook (ADR 0091): augment the
    // caller's registry so `check_jsonl`/`assert` + the `checks` error domain are
    // available to every binary (like `p2p-smoke` for the p2p path).
    let mut reg = reg;
    crate::checks::register(&mut reg);
    // ADR 0112: the built-in `connectors` cookbook so an integration graph
    // (`blut recipe declare s3_roundtrip.json`) resolves + kind-checks.
    crate::connectors::register(&mut reg);
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
        Some(Command::Checks { cmd }) => run_checks_cmd(cmd),
        Some(Command::Catalog { cmd }) => run_catalog_cmd(cmd),
        Some(Command::Connectors { cmd }) => run_connectors_cmd(cmd),
        Some(Command::Partition { cmd }) => run_partition(&reg, cmd).await,
        Some(Command::Artifact { cmd }) => run_artifact_cmd(cmd),
        Some(Command::Schedule { cmd }) => run_schedule_cmd(&reg, cmd),
        Some(Command::Data { cmd }) => run_data(cmd),
        Some(Command::Dataset { cmd }) => run_dataset_cmd(cmd),
        Some(Command::Policy { cmd }) => run_policy(cmd),
        Some(Command::Recipe { cmd }) => run_recipe(&reg, cmd).await,
        Some(Command::Plan { cmd }) => run_plan_cmd(&reg, cmd).await,
        Some(Command::Model { cmd }) => run_model_cmd(cmd).await,
        Some(Command::Experiment { cmd }) => run_experiment_cmd(cmd),
        Some(Command::Cache { cmd }) => run_cache_cmd(cmd),
        Some(Command::Footprint { cmd }) => run_footprint_cmd(cmd),
        Some(Command::Sensor { cmd }) => run_sensor_cmd(cmd),
        #[cfg(feature = "p2p")]
        Some(Command::P2p { cmd }) => run_p2p_cmd(reg, cmd).await,
        #[cfg(feature = "cloud")]
        Some(Command::Cloud { cmd }) => run_cloud_cmd(reg, cmd).await,
        Some(Command::Tui { check }) => match &tui {
            Some(hook) => {
                if check {
                    (hook.check)(reg)
                } else {
                    (hook.console)(std::sync::Arc::new(reg)).await
                }
            }
            // No in-process cockpit: exec the `blut-tui` sidecar from PATH
            // (the ADR-0083 dispatch — same seam as any external subcommand).
            None => {
                let mut argv = vec!["tui".to_string()];
                if check {
                    argv.push("--check".to_string());
                }
                run_external(argv)
            }
        },
        Some(Command::External(argv)) => run_external(argv),
        // Bare invocation: a binary with an attached cockpit opens it (the
        // cookbook binaries); the bare engine prints help — interactive mode
        // is one `blut tui` away, not a silent exec of another binary.
        None => match &tui {
            Some(hook) => (hook.console)(std::sync::Arc::new(reg)).await,
            None => {
                use clap::CommandFactory;
                Cli::command().print_help().ok();
                println!(
                    "\n(the interactive cockpit lives in the `blut-tui` sidecar — run \
                     `blut tui`, or use a cookbook binary, which opens it directly.)"
                );
                Ok(())
            }
        },
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
mod external_subcommand_tests {
    //! ADR 0083 — the cargo-style external-subcommand seam. `blut <cmd>` with no
    //! built-in match execs `blut-<cmd>` from PATH. These pin the two
    //! fail-closed guards without needing a real sidecar on PATH.
    use super::run_external;

    #[test]
    fn rejects_path_traversal_names() {
        for bad in ["../evil", "a/b", "..", "sub\\dir"] {
            let err = run_external(vec![bad.to_string()])
                .expect_err("a name with a path separator or `..` must be rejected");
            assert!(
                err.to_string().contains("unknown subcommand"),
                "expected traversal rejection for {bad:?}, got: {err}"
            );
        }
    }

    #[test]
    fn empty_argv_is_an_error() {
        let err = run_external(vec![]).expect_err("empty external subcommand must error");
        assert!(err.to_string().contains("empty external subcommand"));
    }

    #[test]
    fn missing_sidecar_reports_the_binary_name() {
        // A name that resolves to no `blut-<name>` binary on PATH: the error must
        // name the sidecar and cite the dispatch, not silently succeed.
        let err = run_external(vec!["nonexistent-sidecar-xyz".to_string()])
            .expect_err("a missing sidecar must be a hard error, never a silent no-op");
        let msg = err.to_string();
        assert!(
            msg.contains("blut-nonexistent-sidecar-xyz") && msg.contains("could not be executed"),
            "error must name the sidecar binary + the dispatch: {msg}"
        );
    }
}

#[cfg(test)]
mod registry_completion_cli_tests {
    use super::{
        Cli, Command, DatasetCommand, ExperimentCommand, ModelCommand, PlanCommand, RecipeCommand,
        RecipeMarker, ensure_resume_registry_snapshot, governed_aliases,
    };
    use clap::Parser;
    use std::time::Duration;

    #[test]
    fn dataset_pin_and_exp_compare_parse_as_builtins() {
        let dataset = Cli::try_parse_from([
            "blut",
            "dataset",
            "pin",
            "raw",
            "dataset://tuh@v3",
            "--tenant",
            "research/dev",
        ])
        .unwrap();
        assert!(matches!(
            dataset.command,
            Some(Command::Dataset {
                cmd: DatasetCommand::Pin { source, uri, tenant }
            }) if source == "raw" && uri == "dataset://tuh@v3" && tenant == "research/dev"
        ));

        let experiment = Cli::try_parse_from([
            "blut",
            "exp",
            "compare",
            "codec-train",
            "--tenant",
            "research/dev",
        ])
        .unwrap();
        assert!(matches!(
            experiment.command,
            Some(Command::Experiment {
                cmd: ExperimentCommand::Compare { name, tenant, json: false }
            }) if name == "codec-train" && tenant == "research/dev"
        ));

        let recipe = Cli::try_parse_from([
            "blut",
            "recipe",
            "run",
            "codec-train",
            "--experiment",
            "campaign-a",
        ])
        .unwrap();
        assert!(matches!(
            recipe.command,
            Some(Command::Recipe {
                cmd: RecipeCommand::Run {
                    experiment: Some(name),
                    ..
                }
            }) if name == "campaign-a"
        ));
    }

    #[test]
    fn governed_model_promotion_has_a_finite_default_timeout() {
        let cli = Cli::try_parse_from([
            "blut",
            "model",
            "promote",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "model://encoder@prod",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Model {
                cmd: ModelCommand::Promote { gate_timeout, tenant, .. }
            }) if gate_timeout == Duration::from_secs(300) && tenant == "default"
        ));
    }

    #[test]
    fn plan_resume_sync_io_is_explicit_and_defaults_false() {
        let default = Cli::try_parse_from(["blut", "plan", "resume", "job-1"]).unwrap();
        assert!(matches!(
            default.command,
            Some(Command::Plan {
                cmd: PlanCommand::Resume { sync_io: false, .. }
            })
        ));
        let forced = Cli::try_parse_from(["blut", "plan", "resume", "job-1", "--sync-io"]).unwrap();
        assert!(matches!(
            forced.command,
            Some(Command::Plan {
                cmd: PlanCommand::Resume { sync_io: true, .. }
            })
        ));
    }

    #[test]
    fn governed_alias_override_can_only_widen_prod_boundary() {
        assert_eq!(governed_aliases(None), vec!["prod"]);
        assert_eq!(governed_aliases(Some("")), vec!["prod"]);
        assert_eq!(governed_aliases(Some("staging")), vec!["prod", "staging"]);
        assert_eq!(
            governed_aliases(Some("staging,prod,canary,staging")),
            vec!["prod", "staging", "canary"]
        );
    }

    #[test]
    fn recipe_marker_preserves_source_handles_and_reads_legacy_markers() {
        let marker = RecipeMarker {
            name: "train".into(),
            args: serde_json::json!({"data":"/verified/data.jsonl"}),
            source_args: Some(serde_json::json!({"data":"dataset://train@v1"})),
        };
        let encoded = serde_json::to_value(&marker).unwrap();
        assert_eq!(encoded["source_args"]["data"], "dataset://train@v1");
        assert!(ensure_resume_registry_snapshot(&marker, &marker.args).is_ok());
        assert!(
            ensure_resume_registry_snapshot(
                &marker,
                &serde_json::json!({"data":"/verified/other.jsonl"}),
            )
            .is_err(),
            "a moved mutable registry pointer must not change a resumed run"
        );

        let legacy: RecipeMarker = serde_json::from_value(serde_json::json!({
            "name":"train",
            "args":{"data":"/legacy/data.jsonl"}
        }))
        .unwrap();
        assert!(legacy.source_args.is_none());
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
    /// Resolved args used to compile this exact run. Kept for backward
    /// compatibility and for an auditable launch snapshot.
    args: serde_json::Value,
    /// Original user args, including immutable registry handles. New markers
    /// always carry this so resume can re-resolve/revalidate live dataset bytes
    /// before recompiling. Older markers deserialize with `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_args: Option<serde_json::Value>,
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
        PlanCommand::Resume {
            id,
            shared_cache,
            sync_io,
        } => {
            let job_id = crate::jobs::resolve_job_id(&id).with_context(|| {
                format!("resolve job id '{id}' (ambiguous prefix or missing job)")
            })?;
            let job_dir =
                paths::job_dir(&job_id).with_context(|| format!("resolve job dir for {job_id}"))?;
            let marker = RecipeMarker::read_from(&job_dir)?;
            let tenant = crate::jobs::read_tenant(&job_id)
                .with_context(|| format!("read tenant for {job_id}"))?;
            let source_args = marker.source_args.as_ref().unwrap_or(&marker.args).clone();
            let args = crate::registry_args::resolve_recipe_args(
                source_args,
                &tenant,
                crate::config::launcher::LaunchTarget::Local,
            )
            .map_err(|e| anyhow!("registry arg revalidation: {e}"))?;
            ensure_resume_registry_snapshot(&marker, &args)?;
            let r = reg
                .find(&marker.name)
                .ok_or_else(|| anyhow!("recipe '{}' not in catalog", marker.name))?;
            let plan = (r.compile_fn)(args.clone()).map_err(|e| anyhow!("recipe compile: {e}"))?;
            let tenant_admission =
                crate::broker::tenant_quota::TenantAdmission::prepare(tenant.clone())
                    .map_err(|e| anyhow!("tenant admission: {e}"))?;
            let snapshot = tenant_admission.snapshot();
            let declared = plan.max_declared_envelope();
            let admitted_workers =
                admitted_workers_for(&marker.name, &args, declared.as_ref(), snapshot);
            let admitted_batch_size = admitted_workers.and_then(|workers| {
                admitted_batch_size_for(&marker.name, &args, declared.as_ref(), workers, snapshot)
            });
            let warm = warm_context(&args);
            let tuned = admitted_workers.map(|w| (w, admitted_batch_size));
            let mut footprint = footprint_or_floor(
                plan_footprint_declared(declared.as_ref(), &marker.name, tuned, warm),
                &marker.name,
            )?;

            let mut ctx = ExecCtx::new(job_dir.clone());
            ctx = ctx.with_tenant(tenant.clone()).with_sync_io(sync_io);
            if let Some(budget) = tenant_admission
                .executor_budget_gib(crate::broker::admission::DEFAULT_FLOOR_GIB)
                .map_err(|e| anyhow!("tenant admission: {e}"))?
            {
                ctx = ctx.with_memory_budget(budget);
            }
            if let Some(workers) = admitted_workers {
                ctx = ctx.with_admitted_workers(workers);
            }
            if let Some(batch) = admitted_batch_size {
                ctx = ctx.with_admitted_batch_size(batch);
            }
            let (plan, prepared_footprint) = configure_training_io_admission(
                &format!("resume '{}'", marker.name),
                plan,
                &mut ctx,
                &args,
                admitted_workers,
                admitted_batch_size,
                footprint,
                snapshot,
                crate::config::launcher::LaunchTarget::Local,
            )?;
            footprint = prepared_footprint;
            if shared_cache {
                match CacheHandle::default_global_path() {
                    Some(global) => {
                        std::fs::create_dir_all(&global).with_context(|| {
                            format!("create global cache dir {}", global.display())
                        })?;
                        let cache_handle = (*ctx.cache)
                            .clone()
                            .with_global(global)
                            .with_tenant(&tenant);
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

            let _tenant_reservation = match tenant_admission
                .reserve(&footprint, crate::broker::admission::DEFAULT_FLOOR_GIB)
            {
                Ok(reservation) => reservation,
                Err(reason) => {
                    let _ = crate::jobs::write_state(&job_id, JobState::Failed);
                    return Err(anyhow!(
                        "resume '{}' admission refused: {reason}",
                        marker.name
                    ));
                }
            };

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
                    if let Err(e) = crate::lineage_db::ingest_job(&job_id, &marker.name, "done") {
                        tracing::warn!("lineage index {job_id}: {e}");
                    }
                    Ok(())
                }
                Err(e) => {
                    if let Err(se) = crate::jobs::write_state(&job_id, JobState::Failed) {
                        tracing::warn!("write Failed state for {job_id}: {se}");
                    }
                    if let Err(ie) = crate::lineage_db::ingest_job(&job_id, &marker.name, "failed")
                    {
                        tracing::warn!("lineage index {job_id}: {ie}");
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
        PlanCommand::Publish { spec, tenant } => {
            let text = std::fs::read_to_string(&spec)
                .with_context(|| format!("read PlanSpec {}", spec.display()))?;
            let plan_spec: crate::framework::plan_spec::PlanSpec =
                serde_json::from_str(&text).with_context(|| "parse PlanSpec JSON")?;
            let conn = crate::registry_db::open().map_err(|e| anyhow!("{e}"))?;
            let fp = crate::registry_db::publish(
                &conn,
                reg,
                &plan_spec,
                &acting_user(),
                &tenant,
                spec.to_str(),
                now_unix(),
            )
            .map_err(|e| anyhow!("{e}"))?;
            println!("published {fp}  (tenant={tenant})");
            Ok(())
        }
        PlanCommand::Promote {
            fingerprint,
            pointer,
            tenant,
        } => {
            let name = crate::registry_db::parse_pointer_uri(&pointer)
                .ok_or_else(|| anyhow!("not a `registry://plan@<name>` URI: {pointer}"))?;
            let mut conn = crate::registry_db::open().map_err(|e| anyhow!("{e}"))?;
            crate::registry_db::promote(&mut conn, &fingerprint, &tenant, name, now_unix())
                .map_err(|e| anyhow!("{e}"))?;
            println!("promoted {fingerprint} → registry://plan@{name}");
            Ok(())
        }
        PlanCommand::Rollback { pointer, tenant } => {
            let name = crate::registry_db::parse_pointer_uri(&pointer)
                .ok_or_else(|| anyhow!("not a `registry://plan@<name>` URI: {pointer}"))?;
            let mut conn = crate::registry_db::open().map_err(|e| anyhow!("{e}"))?;
            let prev = crate::registry_db::rollback(&mut conn, &tenant, name, now_unix())
                .map_err(|e| anyhow!("{e}"))?;
            println!("rolled back registry://plan@{name} → {prev}");
            Ok(())
        }
        PlanCommand::History { pointer, tenant } => {
            let name = crate::registry_db::parse_pointer_uri(&pointer)
                .ok_or_else(|| anyhow!("not a `registry://plan@<name>` URI: {pointer}"))?;
            let conn = crate::registry_db::open().map_err(|e| anyhow!("{e}"))?;
            let hist =
                crate::registry_db::history(&conn, &tenant, name).map_err(|e| anyhow!("{e}"))?;
            if hist.is_empty() {
                println!("no history for registry://plan@{name}");
            }
            for h in &hist {
                println!("{}  {}", h.moved_at, h.plan_fingerprint);
            }
            Ok(())
        }
    }
}

fn ensure_resume_registry_snapshot(
    marker: &RecipeMarker,
    revalidated_args: &serde_json::Value,
) -> Result<()> {
    if marker.source_args.is_some() && revalidated_args != &marker.args {
        return Err(anyhow!(
            "registry arg revalidation refused: one or more handles resolve to a different immutable identity than the original launch"
        ));
    }
    Ok(())
}

/// `blut model` — the ADR-0090 model registry, mirroring `blut plan`'s verbs.
/// Sync (no recipe registry needed — a model is an opaque checkpoint hash, not a
/// typechecked PlanSpec). A pointer is `model://<name>@<alias>`.
async fn run_model_cmd(cmd: ModelCommand) -> Result<()> {
    use crate::model_registry as mr;
    let parse = |pointer: &str| -> Result<(String, String)> {
        mr::parse_model_uri(pointer)
            .map(|(n, a)| (n.to_string(), a.to_string()))
            .ok_or_else(|| anyhow!("not a `model://<name>@<alias>` URI: {pointer}"))
    };
    match cmd {
        ModelCommand::Register {
            hash,
            name,
            tenant,
            source,
        } => {
            let conn = mr::open().map_err(|e| anyhow!("{e}"))?;
            mr::register(&conn, &hash, &name, &tenant, source.as_deref(), now_unix())
                .map_err(|e| anyhow!("{e}"))?;
            println!("registered {hash} as model '{name}'  (tenant={tenant})");
            Ok(())
        }
        ModelCommand::Promote {
            hash,
            pointer,
            tenant,
            change_id,
            gate_cmd,
            gate_timeout,
        } => {
            let (name, alias) = parse(&pointer)?;
            // Governed-alias set: `$BLUT_MODEL_GOVERNED_ALIASES` can only add
            // aliases. The built-in `prod` boundary is never removable.
            let gov_env = std::env::var("BLUT_MODEL_GOVERNED_ALIASES").ok();
            let governed = governed_aliases(gov_env.as_deref());
            // For a governed alias, RUN the caller-supplied gate (flag or env) and
            // compute a verdict; otherwise the verdict is unused. Fail-closed: a
            // governed alias with no gate/change-id yields NotConfigured → refuse.
            let verdict = if mr::is_governed(&alias, &governed) {
                let gate_spec = gate_cmd.or_else(|| std::env::var("BLUT_MODEL_GATE_CMD").ok());
                match (
                    gate_spec.as_deref().and_then(mr::GateCmd::parse),
                    &change_id,
                ) {
                    (Some(gate), Some(cid)) => {
                        eprintln!("running governance gate for model://{name}@{alias} …");
                        Some(
                            mr::run_gate_async(&gate, &hash, &name, &alias, cid, gate_timeout)
                                .await,
                        )
                    }
                    _ => Some(mr::GateVerdict::NotConfigured),
                }
            } else {
                None
            };
            let mut conn = mr::open().map_err(|e| anyhow!("{e}"))?;
            mr::promote_governed(
                &mut conn,
                &hash,
                &tenant,
                &name,
                &alias,
                &governed,
                verdict.as_ref(),
                now_unix(),
            )
            .map_err(|e| anyhow!("{e}"))?;
            println!("promoted {hash} → model://{name}@{alias}");
            Ok(())
        }
        ModelCommand::Rollback { pointer, tenant } => {
            let (name, alias) = parse(&pointer)?;
            let mut conn = mr::open().map_err(|e| anyhow!("{e}"))?;
            let prev = mr::rollback(&mut conn, &tenant, &name, &alias, now_unix())
                .map_err(|e| anyhow!("{e}"))?;
            println!("rolled back model://{name}@{alias} → {prev}");
            Ok(())
        }
        ModelCommand::Resolve { pointer, tenant } => {
            let (name, alias) = parse(&pointer)?;
            let conn = mr::open().map_err(|e| anyhow!("{e}"))?;
            match mr::resolve_pointer(&conn, &tenant, &name, &alias).map_err(|e| anyhow!("{e}"))? {
                Some(hash) => {
                    println!("{hash}");
                    Ok(())
                }
                None => Err(anyhow!(
                    "no pointer model://{name}@{alias} (tenant={tenant})"
                )),
            }
        }
        ModelCommand::History { pointer, tenant } => {
            let (name, alias) = parse(&pointer)?;
            let conn = mr::open().map_err(|e| anyhow!("{e}"))?;
            let hist = mr::history(&conn, &tenant, &name, &alias).map_err(|e| anyhow!("{e}"))?;
            if hist.is_empty() {
                println!("no history for model://{name}@{alias}");
            }
            for h in &hist {
                println!("{}  {}", h.moved_at, h.model_hash);
            }
            Ok(())
        }
    }
}

fn run_dataset_cmd(cmd: DatasetCommand) -> Result<()> {
    let conn = crate::datasets_db::open().map_err(|e| anyhow!("{e}"))?;
    match cmd {
        DatasetCommand::Pin {
            source,
            uri,
            tenant,
        } => {
            let tenant = crate::tenant::Tenant::parse(&tenant)
                .ok_or_else(|| anyhow!("invalid --tenant '{tenant}'"))?;
            let binding = crate::dataset_registry::pin(&conn, &source, &uri, &tenant, now_unix())
                .map_err(|e| anyhow!("{e}"))?;
            println!(
                "pinned dataset://{}@{} -> {} ({})",
                binding.name,
                binding.version,
                binding.manifest_sha256,
                binding.source_path.display()
            );
            Ok(())
        }
        DatasetCommand::Resolve {
            uri,
            tenant,
            launcher,
            json,
        } => {
            let tenant = crate::tenant::Tenant::parse(&tenant)
                .ok_or_else(|| anyhow!("invalid --tenant '{tenant}'"))?;
            let target: crate::config::launcher::LaunchTarget = launcher
                .parse()
                .map_err(|e| anyhow!("invalid --launcher {launcher:?}: {e}"))?;
            let binding = crate::dataset_registry::resolve_uri(&conn, &uri, &tenant, target)
                .map_err(|e| anyhow!("{e}"))?;
            if json {
                emit_json(&binding)?;
            } else {
                println!("{}", binding.source_path.display());
            }
            Ok(())
        }
    }
}

fn run_experiment_cmd(cmd: ExperimentCommand) -> Result<()> {
    let db = crate::lineage_db::LineageDb::open().map_err(|e| anyhow!("{e}"))?;
    match cmd {
        ExperimentCommand::Compare { name, tenant, json } => {
            let tenant = crate::tenant::Tenant::parse(&tenant)
                .ok_or_else(|| anyhow!("invalid --tenant '{tenant}'"))?;
            let comparison = crate::experiment_registry::compare_latest(&db, &name, &tenant)
                .map_err(|e| anyhow!("{e}"))?;
            if json {
                emit_json(&comparison)?;
            } else {
                println!(
                    "experiment://{}  tenant={}",
                    comparison.experiment, comparison.tenant
                );
                println!("baseline  : {}", comparison.run_a);
                println!("candidate : {}", comparison.run_b);
                println!(
                    "{}",
                    serde_json::to_string_pretty(&comparison.diff)
                        .unwrap_or_else(|e| format!("serialize error: {e}"))
                );
            }
            Ok(())
        }
        ExperimentCommand::Resolve { uri, tenant, json } => {
            let tenant = crate::tenant::Tenant::parse(&tenant)
                .ok_or_else(|| anyhow!("invalid --tenant '{tenant}'"))?;
            let run = crate::experiment_registry::resolve_uri(&db, &uri, &tenant)
                .map_err(|e| anyhow!("{e}"))?;
            if json {
                emit_json(&run)?;
            } else {
                println!("{}", run.job_id);
            }
            Ok(())
        }
    }
}

/// The acting user recorded as a deployment's publisher.
fn acting_user() -> String {
    std::env::var("USER").unwrap_or_else(|_| "unknown".to_string())
}

/// UNIX seconds now (CLI-side; the registry fns take the timestamp explicitly so
/// they stay pure/deterministic for tests).
fn governed_aliases(raw: Option<&str>) -> Vec<&str> {
    let mut governed = crate::model_registry::DEFAULT_GOVERNED_ALIASES.to_vec();
    if let Some(raw) = raw {
        for alias in raw
            .split(',')
            .map(str::trim)
            .filter(|alias| !alias.is_empty())
        {
            if !governed.contains(&alias) {
                governed.push(alias);
            }
        }
    }
    governed
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
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

#[cfg(feature = "cloud")]
mod cloud;
#[cfg(feature = "cloud")]
use cloud::*;

mod partition;
use partition::*;

mod recipe;
use recipe::*;

mod runs;
use runs::*;

mod stale;
use stale::*;
