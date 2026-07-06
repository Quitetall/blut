// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT cloud worker agent.
//!
//! Pulls jobs from a queue, executes BLUT DAGs, and pushes results.
//! The queue is currently file-based (JSON files in a directory);
//! this will be replaced with a real queue (Redis, SQS, etc.) later.
//!
//! Usage:
//!     blut-worker --queue-dir /tmp/blut-queue --work-dir /tmp/blut-work
//!
//! Job format (JSON file in queue dir):
//!     {
//!         "id": "job-abc",
//!         "recipe": "train_from_dataset",
//!         "args": { ... },
//!         "resources": { "gpu": true, "memory_gib": 16 }
//!     }
//!
//! Result format (JSON file written to results dir):
//!     {
//!         "job_id": "job-abc",
//!         "status": "succeeded",
//!         "artifacts": [...],
//!         "metrics": [...],
//!         "compute_time_secs": 1847
//!     }

mod api;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::fs;

/// BLUT cloud worker agent — pulls jobs from a queue, executes DAGs.
#[derive(Parser, Debug)]
#[command(name = "blut-worker", version)]
struct Cli {
    /// Directory to watch for incoming jobs (JSON files).
    #[arg(long, default_value = "/tmp/blut-queue")]
    queue_dir: PathBuf,

    /// Directory for job working directories.
    #[arg(long, default_value = "/tmp/blut-work")]
    work_dir: PathBuf,

    /// Directory for completed job results.
    #[arg(long, default_value = "/tmp/blut-results")]
    results_dir: PathBuf,

    /// Poll interval in seconds.
    #[arg(long, default_value = "5")]
    poll_interval: u64,

    /// Worker ID (auto-generated if not provided).
    #[arg(long)]
    worker_id: Option<String>,

    /// Enable REST API server on this port.
    #[arg(long)]
    api_port: Option<u16>,

    /// Bind address for the REST API. Defaults to loopback. Binding a
    /// non-loopback address REQUIRES a bearer token in BLUT_WORKER_TOKEN
    /// (the API executes recipes — arbitrary code); startup fails closed
    /// otherwise. The token is env-only, never a CLI flag, so it can't
    /// leak through /proc/<pid>/cmdline or shell history.
    #[arg(long, default_value = "127.0.0.1")]
    api_bind: std::net::IpAddr,
}

/// A job pulled from the queue.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Job {
    /// Unique job ID.
    pub id: String,
    /// Recipe name to execute.
    pub recipe: String,
    /// Recipe args (passed to the recipe's compile function).
    pub args: serde_json::Value,
    /// Resource requirements (informational for now).
    #[serde(default)]
    pub resources: ResourceRequest,
}

/// Resource requirements for a job.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ResourceRequest {
    #[serde(default)]
    pub gpu: bool,
    #[serde(default)]
    pub memory_gib: u32,
    #[serde(default)]
    pub cpu_cores: u32,
}

/// Result of a completed job.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct JobResult {
    pub job_id: String,
    pub status: JobStatus,
    pub artifacts: Vec<String>,
    pub metrics: Vec<serde_json::Value>,
    pub compute_time_secs: u64,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JobStatus {
    Succeeded,
    Failed,
    Cancelled,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("blut_worker=info".parse()?),
        )
        .init();

    let cli = Cli::parse();
    let worker_id = cli
        .worker_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    tracing::info!("BLUT worker {worker_id} starting");
    tracing::info!("  queue:   {}", cli.queue_dir.display());
    tracing::info!("  work:    {}", cli.work_dir.display());
    tracing::info!("  results: {}", cli.results_dir.display());

    // Ensure directories exist
    fs::create_dir_all(&cli.queue_dir).await?;
    fs::create_dir_all(&cli.work_dir).await?;
    fs::create_dir_all(&cli.results_dir).await?;

    // Start API server if configured. Fail-closed: POST /jobs executes
    // recipes (arbitrary code), so a non-loopback bind without a bearer
    // token is refused at startup rather than warned about.
    if let Some(port) = cli.api_port {
        let token = std::env::var("BLUT_WORKER_TOKEN")
            .ok()
            .filter(|t| !t.is_empty());
        if !cli.api_bind.is_loopback() && token.is_none() {
            anyhow::bail!(
                "refusing to serve the REST API on non-loopback {} without a bearer token: \
                 POST /jobs executes recipes (arbitrary code). Set BLUT_WORKER_TOKEN, or keep \
                 the default --api-bind 127.0.0.1.",
                cli.api_bind
            );
        } else if token.is_none() {
            tracing::warn!(
                "REST API is running WITHOUT auth (BLUT_WORKER_TOKEN unset) — \
                 loopback-only mode; any local process can submit jobs"
            );
        }
        let api_state = api::ApiState {
            queue_dir: cli.queue_dir.clone(),
            results_dir: cli.results_dir.clone(),
            jobs: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            worker_id: worker_id.clone(),
            token: token.map(Arc::from),
        };
        let app = api::router(api_state);
        let addr = std::net::SocketAddr::new(cli.api_bind, port);
        // Bind BEFORE spawning: a bind failure (port taken, no permission)
        // is a startup error the operator must see, not a background panic.
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .with_context(|| format!("bind REST API on {addr}"))?;
        tracing::info!("API server listening on {addr}");
        tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!("REST API server exited: {e:#}");
            }
        });
    }

    let poll_interval = Duration::from_secs(cli.poll_interval);

    tracing::info!(
        "worker {worker_id} entering poll loop (interval: {}s)",
        cli.poll_interval
    );

    loop {
        match poll_and_run(&cli, &worker_id).await {
            Ok(had_work) => {
                if !had_work {
                    tokio::time::sleep(poll_interval).await;
                }
            }
            Err(e) => {
                tracing::error!("worker error: {e:#}");
                tokio::time::sleep(poll_interval).await;
            }
        }
    }
}

/// Poll the queue for a job, run it if found. Returns true if work was done.
async fn poll_and_run(cli: &Cli, worker_id: &str) -> Result<bool> {
    // List job files in the queue directory
    let mut entries = fs::read_dir(&cli.queue_dir).await?;
    let mut job_files: Vec<PathBuf> = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            job_files.push(path);
        }
    }

    if job_files.is_empty() {
        return Ok(false);
    }

    // Take the first job (FIFO)
    let job_file = &job_files[0];
    let job: Job = match read_job(job_file).await {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!("failed to read {}: {e:#}", job_file.display());
            // Move malformed file aside
            let bad = job_file.with_extension("json.bad");
            fs::rename(job_file, bad).await.ok();
            return Ok(true);
        }
    };

    tracing::info!("picked up job {} ({})", job.id, job.recipe);

    // Remove from queue (atomic: rename to .processing)
    let processing = job_file.with_extension("json.processing");
    fs::rename(job_file, &processing).await?;

    // Run the job
    let started = Instant::now();
    let result = run_job(&job, cli, worker_id).await;
    let elapsed = started.elapsed().as_secs();

    // Write result
    let job_result = match result {
        Ok(artifacts) => JobResult {
            job_id: job.id.clone(),
            status: JobStatus::Succeeded,
            artifacts,
            metrics: Vec::new(),
            compute_time_secs: elapsed,
            error: None,
        },
        Err(e) => JobResult {
            job_id: job.id.clone(),
            status: JobStatus::Failed,
            artifacts: Vec::new(),
            metrics: Vec::new(),
            compute_time_secs: elapsed,
            error: Some(format!("{e:#}")),
        },
    };

    write_result(&cli.results_dir, &job_result).await?;

    // Clean up processing file
    fs::remove_file(&processing).await.ok();

    tracing::info!(
        "job {} completed: {:?} in {}s",
        job.id,
        job_result.status,
        elapsed
    );

    Ok(true)
}

/// Read and parse a job file.
async fn read_job(path: &Path) -> Result<Job> {
    let content = fs::read_to_string(path).await.context("read job file")?;
    serde_json::from_str(&content).context("parse job JSON")
}

/// Execute a job by compiling its recipe and running the DAG.
async fn run_job(job: &Job, cli: &Cli, _worker_id: &str) -> Result<Vec<String>> {
    let job_dir = cli.work_dir.join(&job.id);
    fs::create_dir_all(&job_dir).await?;

    // Build the execution context
    let ctx = blut::framework::ExecCtx::new(job_dir.clone());

    // Look up the recipe in the registry
    let reg = blut::framework::Registry::new();
    let recipe = reg
        .find(&job.recipe)
        .ok_or_else(|| anyhow::anyhow!("unknown recipe: {}", job.recipe))?;

    // Compile the recipe
    let compiled = (recipe.compile_fn)(job.args.clone())
        .map_err(|e| anyhow::anyhow!("recipe compile failed: {e}"))?;

    tracing::info!(
        "executing plan: {} ({} nodes)",
        compiled.name(),
        compiled.n_nodes()
    );

    // Execute the DAG
    let result = blut::framework::ParallelExecutor::execute(compiled, ctx)
        .await
        .map_err(|e| anyhow::anyhow!("execution failed: {e}"))?;

    tracing::info!("plan completed in {:.1}s", result.elapsed.as_secs_f64());

    // Collect artifacts
    let mut artifacts = Vec::new();
    if let Some(final_output) = &result.final_output {
        artifacts.push(format!("{:?}", final_output));
    }

    Ok(artifacts)
}

/// Write a job result to the results directory.
async fn write_result(results_dir: &Path, result: &JobResult) -> Result<()> {
    let path = results_dir.join(format!("{}.json", result.job_id));
    let content = serde_json::to_string_pretty(result)?;
    fs::write(&path, content).await.context("write result file")
}
