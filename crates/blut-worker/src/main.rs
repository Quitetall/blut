// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT cloud worker agent.
//!
//! Pulls jobs from an internal file queue, executes BLUT DAGs, and writes
//! results. Jobs are accepted only through the loopback REST API, which owns
//! durable ID reservation and atomic publication. Directly dropping `.json`
//! files into the queue is unsupported and rejected.
//!
//! Usage:
//!     blut-worker --queue-dir /tmp/blut-queue --work-dir /tmp/blut-work --api-port 8080
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

    /// Bind address for the REST API. This deprecated prototype has no TLS and
    /// therefore permits loopback only. Use an authenticated local client;
    /// non-loopback startup fails closed even when a bearer token is present.
    #[arg(long, default_value = "127.0.0.1")]
    api_bind: std::net::IpAddr,
}

/// A job pulled from the queue.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Job {
    /// Unique job ID.
    pub id: api::JobId,
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
    pub job_id: api::JobId,
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
    // recipes (arbitrary code), and this deprecated prototype has no TLS.
    // Bearer auth over plaintext is not adequate for any non-loopback hop.
    if let Some(port) = cli.api_port {
        validate_api_bind(cli.api_bind)?;
        let token = std::env::var("BLUT_WORKER_TOKEN")
            .ok()
            .filter(|t| !t.is_empty());
        if token.is_none() {
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

fn validate_api_bind(bind: std::net::IpAddr) -> Result<()> {
    if !bind.is_loopback() {
        anyhow::bail!(
            "refusing non-loopback REST API bind {bind}: blut-worker is a deprecated \
             plaintext prototype and bearer credentials would be exposed in transit"
        );
    }
    Ok(())
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

    // Take the first job (FIFO). REST submission creates a durable reservation
    // before publishing final `.json`; direct file producers are unsupported
    // because they can expose partial JSON and reuse IDs.
    let job_file = &job_files[0];
    let Some(stem) = job_file.file_stem().and_then(|s| s.to_str()) else {
        let bad = job_file.with_extension("json.unreserved");
        fs::rename(job_file, bad).await.ok();
        return Ok(true);
    };
    let reservation = api::JobId::parse(stem)
        .ok()
        .map(|id| cli.queue_dir.join(".job-ids").join(id.as_str()));
    if !reservation.as_ref().is_some_and(|path| path.is_file()) {
        tracing::warn!(
            "refusing unreserved queue file {}: submit through the loopback REST API",
            job_file.display()
        );
        let bad = job_file.with_extension("json.unreserved");
        fs::rename(job_file, bad).await.ok();
        return Ok(true);
    }
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
    let job_dir = cli.work_dir.join(job.id.as_str());
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
        artifacts.push(format!("{final_output:?}"));
    }

    Ok(artifacts)
}

/// Write a job result to the results directory.
async fn write_result(results_dir: &Path, result: &JobResult) -> Result<()> {
    let path = results_dir.join(format!("{}.json", result.job_id));
    let content = serde_json::to_string_pretty(result)?;
    fs::write(&path, content).await.context("write result file")
}

#[cfg(test)]
mod security_tests {
    use super::*;

    #[test]
    fn api_bind_is_loopback_only_even_when_auth_may_be_configured() {
        assert!(validate_api_bind("127.0.0.1".parse().unwrap()).is_ok());
        assert!(validate_api_bind("::1".parse().unwrap()).is_ok());
        assert!(validate_api_bind("0.0.0.0".parse().unwrap()).is_err());
        assert!(validate_api_bind("192.0.2.1".parse().unwrap()).is_err());
    }

    #[tokio::test]
    async fn poller_quarantines_direct_unreserved_queue_files() {
        let td = tempfile::tempdir().unwrap();
        let queue_dir = td.path().join("queue");
        let work_dir = td.path().join("work");
        let results_dir = td.path().join("results");
        fs::create_dir_all(&queue_dir).await.unwrap();
        fs::create_dir_all(&work_dir).await.unwrap();
        fs::create_dir_all(&results_dir).await.unwrap();
        fs::write(
            queue_dir.join("direct.json"),
            r#"{"id":"direct","recipe":"noop","args":{},"resources":{}}"#,
        )
        .await
        .unwrap();
        let cli = Cli {
            queue_dir: queue_dir.clone(),
            work_dir,
            results_dir,
            poll_interval: 1,
            worker_id: None,
            api_port: None,
            api_bind: "127.0.0.1".parse().unwrap(),
        };

        assert!(poll_and_run(&cli, "test-worker").await.unwrap());
        assert!(!queue_dir.join("direct.json").exists());
        assert!(queue_dir.join("direct.json.unreserved").exists());
    }
}
