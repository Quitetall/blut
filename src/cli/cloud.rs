// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut cloud` — the cloud compute queue (ADR 0082 / 0067 T3.1).
//!
//! Mounted as a child of the `cli` module and glob-imported back (the
//! `cli/p2p.rs` pattern), so `super::*` resolves the shared imports.

use super::*;

/// `blut cloud` subcommands. Behind the `cloud` feature.
#[cfg(feature = "cloud")]
#[derive(Subcommand, Debug)]
pub(super) enum CloudCommand {
    /// Turnkey end-to-end smoke: submit the built-in `p2p-echo` stage through the
    /// cloud queue over an object store, run one worker, and print the
    /// content-verified output. Proves the whole data plane with NO cloud account.
    /// Point `--store` at a durable local or network-mounted directory.
    Smoke {
        /// Object store: a local directory (created if absent), or a URL.
        ///
        /// `s3://bucket/prefix` reaches AWS S3 or an S3-compatible endpoint
        /// when built with `--features s3`. The endpoint must support
        /// conditional writes (`If-None-Match: *`): Cloudflare R2 and MinIO do
        /// (both checked), Backblaze B2 does not.
        /// Credentials and endpoint come from the environment -- the same
        /// variables the AWS CLI reads -- never from a flag, because an
        /// argument lands in shell history, the process table, and any log that
        /// records argv.
        #[arg(long, default_value = "/tmp/blut-cloud-smoke")]
        store: String,
        /// Text payload — `p2p-echo` returns it uppercased.
        #[arg(long, default_value = "hello cloud")]
        input: String,
    },
    /// Run a cloud worker: claim the stages that `recipe run --launcher cloud`
    /// submitted to the queue, run each here (contained like a local stage),
    /// and record the result. The lease is renewed while a stage runs; a stage
    /// that is cancelled (`blut cancel`) is stopped. Ctrl-C or SIGTERM stops
    /// the running stage and hands it back for another worker.
    ///
    /// A worker only claims stages submitted by the same code: the engine and
    /// every cookbook at the same build as this binary (the identity cache keys
    /// carry). It leaves the others and records why, which the submitter
    /// reports if nobody picks them up.
    Worker {
        /// The queue: `s3://bucket[/prefix]` (a build with `s3`; credentials and
        /// endpoint from the `AWS_*` environment only), `file:///dir`, or a
        /// directory. Default: `$BLUT_CLOUD_QUEUE`.
        #[arg(long)]
        queue: Option<String>,
        /// Lease owner name, unique per worker (default `<host>-<pid>`).
        #[arg(long)]
        id: Option<String>,
        /// Lease length in seconds, renewed every third of it. A worker that
        /// dies is replaced about this long after its last renewal.
        #[arg(long, default_value_t = 60)]
        lease_secs: u64,
        /// Longest wait between two polls of an empty queue, in milliseconds
        /// (polling backs off to it from half a second).
        #[arg(long, default_value_t = 5000)]
        poll_ms: u64,
        /// Stop after this many stages.
        #[arg(long)]
        max_jobs: Option<u64>,
        /// Stop once the queue has been empty for this many seconds.
        #[arg(long)]
        idle_exit_secs: Option<u64>,
        /// Scratch directory for stage runs (default: a temporary directory,
        /// removed on exit).
        #[arg(long)]
        work_dir: Option<std::path::PathBuf>,
        /// Also claim stages that request a GPU.
        #[arg(long, default_value_t = false)]
        gpu: bool,
    },
    /// List the stages a cloud queue holds that are queued or running.
    Queue {
        /// The queue (default `$BLUT_CLOUD_QUEUE`).
        #[arg(long)]
        queue: Option<String>,
    },
}

/// `--queue`, else `$BLUT_CLOUD_QUEUE`.
#[cfg(feature = "cloud")]
fn queue_spec(flag: Option<String>) -> Result<String> {
    use crate::config::launcher::CLOUD_QUEUE_ENV;
    flag.filter(|q| !q.trim().is_empty())
        .or_else(|| std::env::var(CLOUD_QUEUE_ENV).ok())
        .filter(|q| !q.trim().is_empty())
        .ok_or_else(|| {
            anyhow!(
                "name the queue with --queue or {CLOUD_QUEUE_ENV} (s3://bucket[/prefix] or a \
                 directory)"
            )
        })
}

#[cfg(feature = "cloud")]
fn default_worker_id() -> String {
    let host = nix::unistd::gethostname()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "worker".into());
    format!("{host}-{}", std::process::id())
}

/// `blut cloud worker`.
#[cfg(feature = "cloud")]
#[allow(clippy::too_many_arguments)]
async fn run_cloud_worker(
    reg: &crate::framework::Registry,
    queue: Option<String>,
    id: Option<String>,
    lease_secs: u64,
    poll_ms: u64,
    max_jobs: Option<u64>,
    idle_exit_secs: Option<u64>,
    work_dir: Option<std::path::PathBuf>,
    gpu: bool,
) -> Result<()> {
    use crate::cloud::location::CloudLocation;
    use crate::cloud::object_queue::ObjectQueueOptions;
    use crate::cloud::worker::{ServeEvent, ServeOptions, serve};
    use std::time::Duration;

    let spec = queue_spec(queue)?;
    let poll_max = Duration::from_millis(poll_ms.max(10));
    let poll_min = Duration::from_millis(500).min(poll_max);
    let options = ObjectQueueOptions::default()
        .with_accept_gpu(gpu)
        .with_status_refresh(poll_min);
    let location = CloudLocation::open(&spec, options).map_err(|e| flat_error("open queue", e))?;
    location
        .probe()
        .await
        .map_err(|e| flat_error(&format!("cloud queue {}", location.display()), e))?;
    let worker_id = id
        .filter(|i| !i.trim().is_empty())
        .unwrap_or_else(default_worker_id);
    let scratch;
    let work_root = match work_dir {
        Some(dir) => {
            std::fs::create_dir_all(&dir)
                .with_context(|| format!("create work dir {}", dir.display()))?;
            dir
        }
        None => {
            scratch = tempfile::Builder::new()
                .prefix("blut-cloud-worker-")
                .tempdir()?;
            scratch.path().to_path_buf()
        }
    };
    eprintln!(
        "cloud worker {worker_id} on {} (pid {})",
        location.display(),
        std::process::id()
    );
    eprintln!("code   {}", location.queue().code_identity());
    eprintln!(
        "lease  {lease_secs}s; gpu stages: {}",
        if gpu { "yes" } else { "no" }
    );

    let shutdown = tokio_util::sync::CancellationToken::new();
    {
        let shutdown = shutdown.clone();
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
                std::future::pending::<()>().await;
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term => {}
            }
            eprintln!("signal received: stopping the running stage and handing it back");
            shutdown.cancel();
        });
    }

    let serve_options = ServeOptions::new(worker_id.clone(), work_root)
        .with_lease_secs(lease_secs)
        .with_poll(poll_min, poll_max)
        .with_status_interval(poll_min.max(Duration::from_secs(1)))
        .with_max_jobs(max_jobs)
        .with_idle_exit(idle_exit_secs.map(Duration::from_secs));
    let matrix = crate::p2p::trust::DispatchMatrix::default();
    let policy = crate::cloud::policy::CloudDispatchPolicy::new(matrix.clone());
    let ledger = crate::cloud::cost::CostLedger::default();
    let print = |event: &ServeEvent| match event {
        ServeEvent::Claimed {
            job_id,
            stage_name,
            assignment,
        } => eprintln!(
            "claimed  {job_id} ({stage_name}), generation {}",
            assignment.generation
        ),
        ServeEvent::Finished {
            job_id,
            outcome,
            wall_time_ms,
            detail,
        } => match detail {
            Some(detail) => {
                eprintln!("finished {job_id}: {outcome:?} after {wall_time_ms} ms: {detail}")
            }
            None => eprintln!("finished {job_id}: {outcome:?} in {wall_time_ms} ms"),
        },
        ServeEvent::Abandoned { job_id } => eprintln!(
            "stopped  {job_id}: it was cancelled or given to another worker; nothing recorded"
        ),
        ServeEvent::Released { job_id } => {
            eprintln!("released {job_id} for another worker")
        }
        ServeEvent::QueueError { message } => {
            eprintln!("queue error (retrying): {message}")
        }
    };
    let report = serve(
        location.store(),
        location.queue().as_ref(),
        reg,
        &policy,
        &matrix,
        &serve_options,
        &shutdown,
        Some(&ledger),
        &print,
    )
    .await;
    eprintln!(
        "cloud worker {worker_id} stopped: {} claimed, {} succeeded, {} failed, {} released, \
         {} stopped; {:.3} compute units",
        report.claimed,
        report.succeeded,
        report.failed,
        report.released,
        report.abandoned,
        ledger.total()
    );
    Ok(())
}

/// What a `--launcher cloud` run keeps beside its `ExecCtx`: the queue it
/// submits to, to probe it before the job starts, record it with the job, and
/// withdraw any work still queued or running when the run ends.
#[cfg(feature = "cloud")]
pub(super) struct CloudRun {
    location: crate::cloud::location::CloudLocation,
}

/// Install the cloud adapter on `ctx`: a `CloudSubmitter` over the queue in
/// `$BLUT_CLOUD_QUEUE`, with the pickup window and a record of every job it
/// enqueues in `job_dir`. Opens nothing remote yet ([`CloudRun::probe`] does).
#[cfg(feature = "cloud")]
pub(super) fn install_cloud_launch(
    reg: &crate::framework::Registry,
    ctx: crate::framework::ExecCtx,
    job_dir: &std::path::Path,
    tenant: &crate::tenant::Tenant,
    surface: &str,
) -> Result<(crate::framework::ExecCtx, CloudRun)> {
    use crate::cloud::location::CloudLocation;
    use crate::cloud::object_queue::ObjectQueueOptions;
    use crate::config::launcher::{CLOUD_QUEUE_ENV, check_cloud_launch, cloud_pickup_window};
    use std::sync::Arc;

    check_cloud_launch(surface).map_err(|e| anyhow!("{e}"))?;
    if tenant.is_restricted() {
        return Err(anyhow!(
            "--launcher cloud cannot place {surface}: tenant '{tenant}' is restricted, and \
             restricted data never leaves this machine. Nothing ran"
        ));
    }
    let spec = std::env::var(CLOUD_QUEUE_ENV).unwrap_or_default();
    let pickup = cloud_pickup_window().map_err(|e| anyhow!("{e}"))?;
    let location = CloudLocation::open(&spec, ObjectQueueOptions::default())
        .map_err(|e| flat_error(CLOUD_QUEUE_ENV, e))?;
    let queue: Arc<dyn crate::cloud::queue::CloudQueue> = location.queue().clone();
    let submitter = crate::cloud::submitter::CloudSubmitter::new(
        location.store().clone(),
        queue,
        Arc::new(reg.stages_only()),
    )
    .with_pickup_timeout(pickup)
    .with_job_record(job_dir.join(crate::cloud::records::JOBS_RECORD_FILE));
    let ctx = ctx.with_execution_adapter(
        Arc::new(crate::cloud::policy::CloudDispatchPolicy::default()),
        Arc::new(submitter),
    );
    Ok((ctx, CloudRun { location }))
}

#[cfg(feature = "cloud")]
impl CloudRun {
    /// Reach the queue once, so a wrong endpoint or rejected credentials fail
    /// the launch before anything is written for the job.
    pub(super) async fn probe(&self) -> Result<()> {
        use crate::config::launcher::CLOUD_QUEUE_ENV;
        self.location.probe().await.map_err(|e| {
            flat_error(
                &format!(
                    "{CLOUD_QUEUE_ENV}: the cloud queue {} is not reachable",
                    self.location.display()
                ),
                e,
            )
        })
    }

    /// Refuse to resume `job_id` on a different queue than it ran on: its
    /// recorded cloud jobs would then be out of `blut cancel`'s reach.
    pub(super) fn check_same_queue(&self, job_id: &str, job_dir: &std::path::Path) -> Result<()> {
        use crate::config::launcher::CLOUD_QUEUE_ENV;
        match crate::cloud::records::recorded_queue(job_dir) {
            Some(recorded) if recorded != self.location.spec() => Err(anyhow!(
                "job {job_id} ran on the cloud queue {}, and {CLOUD_QUEUE_ENV} names {}; \
                 resume it on the queue it ran on",
                crate::cloud::location::redact(&recorded),
                self.location.display()
            )),
            _ => Ok(()),
        }
    }

    /// Record the queue with the job, for `blut cancel` and `plan resume`.
    pub(super) fn record(&self, job_dir: &std::path::Path) -> Result<()> {
        crate::cloud::records::write_queue_record(job_dir, &self.location)
            .with_context(|| format!("record the cloud queue in {}", job_dir.display()))?;
        eprintln!(
            "cloud  {} (workers: `blut cloud worker --queue <the same location>`)",
            self.location.display()
        );
        Ok(())
    }

    /// Withdraw any of this job's cloud work that is still queued or running.
    /// Normally there is none: every stage waited for its result. After a
    /// cancellation or a failure elsewhere in the plan, stages still in the
    /// queue would otherwise run for nobody.
    pub(super) async fn sweep(&self, job_dir: &std::path::Path) {
        match crate::cloud::records::cancel_recorded_jobs(job_dir, "the submitting run ended").await
        {
            Ok(cancelled) if cancelled.is_empty() => {}
            Ok(cancelled) => eprintln!(
                "withdrew {} cloud job(s) still queued or running: {}",
                cancelled.len(),
                cancelled.join(", ")
            ),
            Err(error) => eprintln!(
                "warning: could not withdraw this run's leftover cloud jobs: {error}; \
                 `blut cancel` on the job retries"
            ),
        }
    }
}

/// `blut cloud` — the cloud compute queue (ADR 0082 / 0067 T3.1).
#[cfg(feature = "cloud")]
pub(super) async fn run_cloud_cmd(
    mut reg: crate::framework::Registry,
    cmd: CloudCommand,
) -> Result<()> {
    use crate::cloud::queue::MemQueue;
    use crate::cloud::submitter::{CloudPoll, CloudSubmitSpec, CloudSubmitter};
    use crate::cloud::worker::run_one;
    use crate::framework::artifact::ContentHash;
    use crate::framework::stage::ErasedArtifact;
    use crate::p2p::dispatch::DefaultDispatchPolicy;
    use crate::p2p::smoke::{self, SMOKE_STAGE, SmokeText};
    use crate::p2p::task::ResourceRequest;
    use crate::p2p::trust::{DataClass, DispatchMatrix, TrustLevel};
    use std::sync::Arc;

    // Ship the built-in p2p-echo smoke stage so both submit + worker resolve it.
    smoke::register(&mut reg);
    let reg = Arc::new(reg);

    match cmd {
        CloudCommand::Worker {
            queue,
            id,
            lease_secs,
            poll_ms,
            max_jobs,
            idle_exit_secs,
            work_dir,
            gpu,
        } => {
            run_cloud_worker(
                &reg,
                queue,
                id,
                lease_secs,
                poll_ms,
                max_jobs,
                idle_exit_secs,
                work_dir,
                gpu,
            )
            .await
        }
        CloudCommand::Queue { queue } => {
            use crate::cloud::location::CloudLocation;
            use crate::cloud::object_queue::ObjectQueueOptions;
            use crate::cloud::queue::JobStatus;
            let spec = queue_spec(queue)?;
            let location = CloudLocation::open(&spec, ObjectQueueOptions::default())
                .map_err(|e| flat_error("open queue", e))?;
            let jobs = location
                .queue()
                .list_active()
                .await
                .map_err(|e| flat_error("list the queue", e))?;
            println!("{} job(s) on {}", jobs.len(), location.display());
            for job in jobs {
                let status = match &job.status {
                    JobStatus::Queued => "queued".to_string(),
                    JobStatus::Running(a) => {
                        format!("running on {} (generation {})", a.owner, a.generation)
                    }
                    JobStatus::Done(_) => "finished".to_string(),
                    JobStatus::Unknown => "unknown".to_string(),
                };
                println!(
                    "{:<48} {:<24} {}",
                    job.job_id,
                    job.stage_name.as_deref().unwrap_or("?"),
                    status
                );
            }
            Ok(())
        }
        CloudCommand::Smoke { store, input } => {
            let blob_store = open_smoke_store(&store)?;
            let queue = Arc::new(MemQueue::new());

            // Produce the input artifact on disk (the submitter's src_root).
            let src_root = tempfile::tempdir()?;
            let in_path = src_root.path().join("in.txt");
            std::fs::write(&in_path, input.as_bytes())?;
            let art = SmokeText {
                content_hash: ContentHash::hash_file(&in_path).context("hash input")?,
                path: in_path,
            };
            let erased = ErasedArtifact::from_typed(&art).context("erase input")?;

            let submitter = CloudSubmitter::new(blob_store.clone(), queue.clone(), reg.clone());
            eprintln!("submitting p2p-echo to the cloud queue (store: {store})…");
            let handle = submitter
                .submit(CloudSubmitSpec {
                    // Fixed id: re-running against the same --store is idempotent
                    // (blobs are content-addressed; the MemQueue is fresh each run).
                    job_id: "cloud-smoke-1".into(),
                    tenant: crate::tenant::Tenant::default(),
                    stage_name: SMOKE_STAGE.into(),
                    invocation_key: crate::framework::InvocationKey::from_digest(
                        ContentHash::of_bytes(b"cloud-smoke-1"),
                    ),
                    input: erased,
                    src_root: src_root.path().to_path_buf(),
                    args: serde_json::json!({}),
                    expected_content_id: None,
                    data_class: DataClass::Public,
                    resources: ResourceRequest::default(),
                    priority: 0,
                    timeout_secs: 30,
                })
                .await
                .map_err(|e| flat_error("submit cloud job", e))?;

            // Run one worker pass (Registered trust — the clinical hard-block posture).
            // One matrix instance, shared by the policy and the worker, so a future
            // customization can't let them diverge.
            let matrix = DispatchMatrix::default();
            let policy = DefaultDispatchPolicy::new(matrix.clone());
            let work_root = tempfile::tempdir()?;
            eprintln!("running a cloud worker…");
            run_one(
                &blob_store,
                queue.as_ref(),
                reg.as_ref(),
                &policy,
                &matrix,
                "cloud-smoke-worker",
                TrustLevel::Registered,
                30,
                work_root.path(),
                None,
            )
            .await
            .map_err(|e| flat_error("cloud worker", e))?;

            let out_dir = tempfile::tempdir()?;
            match handle
                .poll(out_dir.path())
                .await
                .map_err(|e| flat_error("poll cloud job", e))?
            {
                CloudPoll::Succeeded(output) => {
                    let out: SmokeText = output.into_typed().context("decode output")?;
                    let body = std::fs::read_to_string(&out.path)?;
                    println!("✔ cloud round-trip verified over {store}; output:");
                    println!("{body}");
                    Ok(())
                }
                CloudPoll::Failed(m) => Err(anyhow!("cloud job failed: {m}")),
                CloudPoll::Pending => Err(anyhow!("job still pending after the worker ran")),
                CloudPoll::Cancelled => Err(anyhow!("job cancelled")),
                CloudPoll::TimedOut => Err(anyhow!("job timed out")),
                CloudPoll::Unknown => Err(anyhow!("job unknown")),
            }
        }
    }
}

#[cfg(all(test, feature = "cloud"))]
mod cloud_cli_tests {
    use super::{Cli, CloudCommand, Command, run_cloud_cmd};
    use clap::Parser;

    #[test]
    fn cloud_smoke_parses_with_defaults() {
        match Cli::try_parse_from(["blut", "cloud", "smoke"])
            .expect("parse")
            .command
        {
            Some(Command::Cloud {
                cmd: CloudCommand::Smoke { input, .. },
            }) => {
                assert_eq!(input, "hello cloud");
            }
            other => panic!("expected cloud smoke, got {other:?}"),
        }
    }

    #[test]
    fn cloud_smoke_parses_overrides() {
        match Cli::try_parse_from([
            "blut", "cloud", "smoke", "--input", "hi", "--store", "/tmp/x",
        ])
        .expect("parse")
        .command
        {
            Some(Command::Cloud {
                cmd: CloudCommand::Smoke { input, store },
            }) => {
                assert_eq!(input, "hi");
                assert_eq!(store, "/tmp/x");
            }
            other => panic!("got {other:?}"),
        }
    }

    // Drives the REAL `blut cloud smoke` handler end-to-end (submit → worker →
    // poll → content-verify) over a local-fs object store — the CLI proof.
    #[tokio::test]
    async fn cloud_smoke_handler_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let reg = crate::framework::Registry::new();
        run_cloud_cmd(
            reg,
            CloudCommand::Smoke {
                store: dir.path().display().to_string(),
                input: "hello cli".to_string(),
            },
        )
        .await
        .expect("cloud smoke handler must round-trip");
    }
}

/// One line per failure. The cloud and object-store errors each embed their
/// source's message in their own, so printing the error chain repeated the
/// provider's response (an S3 XML body appeared five times).
#[cfg(feature = "cloud")]
fn flat_error(what: &str, error: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{what}: {error}")
}

/// Open the smoke store from `--store`: a URL when it has a scheme, otherwise a
/// local directory (created if absent, so the default path keeps working).
///
/// A bare Windows-style path like `C:\\store` is NOT a URL; the scheme check
/// requires more than one character before the colon so a drive letter cannot
/// be mistaken for one.
#[cfg(feature = "cloud")]
fn open_smoke_store(store: &str) -> Result<crate::framework::object_store::ObjectStore> {
    use crate::framework::object_store::ObjectStore;
    if looks_like_url(store) {
        #[cfg(feature = "s3")]
        {
            // `--store` promises a directory "created if absent"; a `file://`
            // URL to a missing directory failed with "Unable to canonicalize
            // filesystem root" while the same path given bare worked.
            if let Some(dir) = store.strip_prefix("file://").filter(|d| d.starts_with('/')) {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("create object-store dir {dir}"))?;
            }
            return ObjectStore::from_url(store, "cloud")
                .with_context(|| format!("open object store {store}"));
        }
        #[cfg(not(feature = "s3"))]
        {
            return Err(anyhow!(
                "{store} is a URL, but this binary was built without the `s3` feature; \
                 rebuild with --features s3, or pass a local directory"
            ));
        }
    }
    let path = std::path::Path::new(store);
    std::fs::create_dir_all(path)
        .with_context(|| format!("create object-store dir {}", path.display()))?;
    ObjectStore::local_provider(path, "cloud").context("open object store")
}

/// Does this look like a URL rather than a path? Requires a scheme of at least
/// two characters, so `C:\\store` stays a path.
#[cfg(feature = "cloud")]
fn looks_like_url(s: &str) -> bool {
    match s.split_once("://") {
        Some((scheme, _)) => {
            scheme.len() > 1
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
        }
        None => false,
    }
}

#[cfg(all(test, feature = "cloud"))]
mod cloud_store_arg_tests {
    use super::looks_like_url;

    #[test]
    fn urls_are_urls_and_paths_are_paths() {
        assert!(looks_like_url("s3://bucket/prefix"));
        assert!(looks_like_url("file:///tmp/store"));
        assert!(!looks_like_url("/tmp/blut-cloud-smoke"));
        assert!(!looks_like_url("relative/dir"));
    }

    #[test]
    fn a_windows_drive_letter_is_not_a_scheme() {
        // One-character "scheme" is a drive letter, not a URL.
        assert!(!looks_like_url("C://store"));
    }
}

#[cfg(all(test, feature = "s3"))]
mod cloud_file_url_tests {
    #[test]
    fn a_file_url_store_is_created_like_a_bare_path() {
        // E2E regression: `blut cloud smoke --store file:///new/dir` failed
        // with "Unable to canonicalize filesystem root".
        let td = tempfile::tempdir().unwrap();
        let dir = td.path().join("fresh");
        let url = format!("file://{}", dir.display());
        super::open_smoke_store(&url).expect("file:// store opens");
        assert!(dir.is_dir());
    }
}

#[cfg(all(test, feature = "cloud"))]
mod flat_error_tests {
    #[test]
    fn a_wrapped_error_prints_its_message_once() {
        #[derive(Debug)]
        struct Leaf;
        impl std::fmt::Display for Leaf {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("<Error>NoSuchBucket</Error>")
            }
        }
        impl std::error::Error for Leaf {}
        #[derive(Debug)]
        struct Wrapper(Leaf);
        impl std::fmt::Display for Wrapper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "put failed: {}", self.0)
            }
        }
        impl std::error::Error for Wrapper {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        let shown = format!("{:?}", super::flat_error("submit cloud job", Wrapper(Leaf)));
        assert_eq!(shown.matches("NoSuchBucket").count(), 1, "{shown}");
    }
}
