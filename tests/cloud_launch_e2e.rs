// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `recipe run --launcher cloud` end to end, across processes.
//!
//! Every process here is this test binary re-executed with `ARGS_ENV` set: the
//! child runs `blut::cli::run_from` with those arguments and this file's
//! cookbook, so the submitter (`recipe run`, `plan resume`, `cancel`) and each
//! worker (`cloud worker`) are separate processes sharing nothing but a queue
//! directory. Each stage writes the pid of the process that ran it into its
//! output, which is how a test tells a worker's run from the submitter's.
#![cfg(feature = "cloud")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use blut::backends::TrainingBackend;
use blut::framework::artifact::ContentHash;
use blut::framework::code_identity::{BuildIdentity, CodeIdentity};
use blut::framework::compat::Compatible;
use blut::framework::cookbook::{Cookbook, Registry};
use blut::framework::plan::{CompiledPlan, Plan};
use blut::framework::resource::Resource;
use blut::framework::stage::{ErasedStageCtor, Stage, StageContext};
use blut::framework::{RecipeError, StageError};
use blut::p2p::smoke::SmokeText;
use blut::recipes::recipe::{RecipeCategory, RecipeDef};
use serde::{Deserialize, Serialize};

const ARGS_ENV: &str = "BLUT_CLOUD_E2E_ARGS";
const IDENTITY_ENV: &str = "BLUT_CLOUD_E2E_IDENTITY";

// ── the cookbook every process registers ────────────────────────────────

struct E2eBackend;
impl TrainingBackend for E2eBackend {
    const ID: &'static str = "cloud-e2e";
    const DESCRIPTION: &'static str = "cloud launcher end-to-end tests";
}

fn write_text(ctx: &StageContext, name: &str, body: &str) -> Result<SmokeText, StageError> {
    let path = ctx.stage_dir.join(name);
    std::fs::write(&path, body).map_err(|source| StageError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(SmokeText {
        content_hash: ContentHash::hash_file(&path).map_err(|source| StageError::Io {
            path: path.clone(),
            source,
        })?,
        path,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct SeedArgs {
    text: String,
}

/// `()` → a text file: the seed text and the pid of the process that ran it.
struct Seed;
#[async_trait]
impl Stage for Seed {
    const NAME: &'static str = "e2e-seed";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = SmokeText;
    type Args = SeedArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &SeedArgs,
    ) -> Result<SmokeText, StageError> {
        write_text(
            ctx,
            "seed.txt",
            &format!("{}\nseed pid={}\n", args.text, std::process::id()),
        )
    }
}
impl Compatible<E2eBackend> for Seed {}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct AppendArgs {
    label: String,
}

/// Copies its input and appends `<label> pid=<pid>`.
struct Append;
#[async_trait]
impl Stage for Append {
    const NAME: &'static str = "e2e-append";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = SmokeText;
    type Output = SmokeText;
    type Args = AppendArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: SmokeText,
        args: &AppendArgs,
    ) -> Result<SmokeText, StageError> {
        let body = std::fs::read_to_string(&input.path).map_err(|source| StageError::Io {
            path: input.path.clone(),
            source,
        })?;
        write_text(
            ctx,
            &format!("{}.txt", args.label),
            &format!("{body}{} pid={}\n", args.label, std::process::id()),
        )
    }
}
impl Compatible<E2eBackend> for Append {}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct HoldArgs {
    secs: u64,
    marker_dir: String,
}

/// Writes `started-<pid>` into `marker_dir`, waits `secs`, and passes its
/// input on. If it is stopped before then, `stopped-<pid>` appears instead of
/// `finished-<pid>`: the stage future is dropped, and this guard sees it.
struct Hold;
struct StopMarker(Option<PathBuf>);
impl Drop for StopMarker {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::write(path, b"stopped");
        }
    }
}
#[async_trait]
impl Stage for Hold {
    const NAME: &'static str = "e2e-hold";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = SmokeText;
    type Output = SmokeText;
    type Args = HoldArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: SmokeText,
        args: &HoldArgs,
    ) -> Result<SmokeText, StageError> {
        let dir = Path::new(&args.marker_dir);
        let pid = std::process::id();
        std::fs::write(dir.join(format!("started-{pid}")), b"").map_err(|source| {
            StageError::Io {
                path: dir.to_path_buf(),
                source,
            }
        })?;
        let mut guard = StopMarker(Some(dir.join(format!("stopped-{pid}"))));
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(args.secs)) => {}
            _ = ctx.cancel.cancelled() => return Err(StageError::Cancelled),
        }
        guard.0 = None;
        let _ = std::fs::write(dir.join(format!("finished-{pid}")), b"");
        let body = std::fs::read_to_string(&input.path).map_err(|source| StageError::Io {
            path: input.path.clone(),
            source,
        })?;
        write_text(ctx, "held.txt", &format!("{body}hold pid={pid}\n"))
    }
}
impl Compatible<E2eBackend> for Hold {}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct ChainArgs {
    text: String,
}

fn compile_chain(raw: serde_json::Value) -> Result<CompiledPlan, RecipeError> {
    let args: ChainArgs =
        serde_json::from_value(raw.clone()).map_err(|e| RecipeError::InvalidArgs {
            field: None,
            message: e.to_string(),
        })?;
    Ok(Plan::<(), E2eBackend>::new("cloud_e2e_chain", raw)
        .start(Seed, SeedArgs { text: args.text })
        .then(Append, AppendArgs { label: "a".into() })
        .then(Append, AppendArgs { label: "b".into() })
        .finish()
        .into_compiled())
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct HoldChainArgs {
    secs: u64,
    marker_dir: String,
}

fn compile_hold(raw: serde_json::Value) -> Result<CompiledPlan, RecipeError> {
    let args: HoldChainArgs =
        serde_json::from_value(raw.clone()).map_err(|e| RecipeError::InvalidArgs {
            field: None,
            message: e.to_string(),
        })?;
    Ok(Plan::<(), E2eBackend>::new("cloud_e2e_hold", raw)
        .start(
            Seed,
            SeedArgs {
                text: "hold".into(),
            },
        )
        .then(
            Hold,
            HoldArgs {
                secs: args.secs,
                marker_dir: args.marker_dir,
            },
        )
        .finish()
        .into_compiled())
}

static CHAIN: RecipeDef = RecipeDef {
    name: "cloud_e2e_chain",
    description: "seed, then two appends",
    backend_id: "cloud-e2e",
    category: RecipeCategory::User,
    input_kinds: &[],
    output_kind: "p2p.smoke.text",
    schedule: None,
    args_schema_fn: || blut::recipes::recipe::schema_of::<ChainArgs>(),
    compile_fn: compile_chain,
};

static HOLD: RecipeDef = RecipeDef {
    name: "cloud_e2e_hold",
    description: "seed, then a stage that waits",
    backend_id: "cloud-e2e",
    category: RecipeCategory::User,
    input_kinds: &[],
    output_kind: "p2p.smoke.text",
    schedule: None,
    args_schema_fn: || blut::recipes::recipe::schema_of::<HoldChainArgs>(),
    compile_fn: compile_hold,
};

struct E2eCookbook;
impl Cookbook for E2eCookbook {
    fn name(&self) -> &'static str {
        "cloud-e2e"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        static R: &[&RecipeDef] = &[&CHAIN, &HOLD];
        R
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static S: &[(&str, ErasedStageCtor)] = &[
            ("e2e-seed", || std::sync::Arc::new(Seed)),
            ("e2e-append", || std::sync::Arc::new(Append)),
            ("e2e-hold", || std::sync::Arc::new(Hold)),
        ];
        S
    }
    fn identity(&self) -> Option<BuildIdentity> {
        Some(BuildIdentity::new("cloud-e2e-cookbook", "1.0.0"))
    }
}

fn registry() -> Registry {
    let mut r = Registry::new();
    r.register(Box::new(E2eCookbook));
    r
}

/// The child process: run the CLI with the arguments in `ARGS_ENV`. With
/// `IDENTITY_ENV=other`, it first fixes a code identity no other process here
/// has, as a build of different code would.
#[test]
#[ignore = "child-process entry for the cloud launcher e2e tests"]
fn cli_child() {
    let Ok(args) = std::env::var(ARGS_ENV) else {
        return;
    };
    let args: Vec<String> = serde_json::from_str(&args).expect("child args");
    if std::env::var(IDENTITY_ENV).as_deref() == Ok("other") {
        blut::framework::code_identity::install(
            CodeIdentity::engine_only()
                .with_cookbook(BuildIdentity::new("cloud-e2e-cookbook", "2.0.0-other")),
        )
        .expect("install the other identity first");
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    if let Err(error) = rt.block_on(blut::cli::run_from(registry(), args)) {
        eprintln!("child error: {error:#}");
        std::process::exit(1);
    }
}

// ── harness ─────────────────────────────────────────────────────────────

/// A machine of its own: state directories, plus a log per process.
struct Host {
    root: tempfile::TempDir,
}

impl Host {
    fn new() -> Self {
        Self {
            root: tempfile::tempdir().unwrap(),
        }
    }

    fn path(&self) -> &Path {
        self.root.path()
    }

    fn jobs_dir(&self) -> PathBuf {
        self.path().join("jobs")
    }

    fn command(&self, args: &[&str], log: &str) -> Command {
        let mut argv = vec!["blut".to_string()];
        argv.extend(args.iter().map(|a| a.to_string()));
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "cli_child",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ARGS_ENV, serde_json::to_string(&argv).unwrap())
        .env("LAMU_TRAIN_JOBS_DIR", self.jobs_dir())
        .env("LAMU_TRAIN_DATA_DIR", self.path().join("data"))
        .env("XDG_DATA_HOME", self.path().join("xdg-data"))
        .env("XDG_CACHE_HOME", self.path().join("xdg-cache"))
        .env("XDG_CONFIG_HOME", self.path().join("xdg-config"))
        .env("BLUT_PARTITIONS_DIR", self.path().join("partitions"))
        .env("BLUT_SCHED_DEVICES", "0")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("BLUT_NO_CONTAIN", "1")
        .env("RUST_LOG", "warn")
        .env_remove("BLUT_CLOUD_QUEUE")
        .env_remove("BLUT_CLOUD_PICKUP_SECS")
        .env_remove(IDENTITY_ENV)
        .stdin(Stdio::null());
        let log = std::fs::File::create(self.path().join(log)).unwrap();
        cmd.stdout(log.try_clone().unwrap()).stderr(log);
        cmd
    }

    fn log(&self, name: &str) -> String {
        std::fs::read_to_string(self.path().join(name)).unwrap_or_default()
    }

    /// The one job this host has run.
    fn only_job(&self) -> String {
        let jobs: Vec<String> = std::fs::read_dir(self.jobs_dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(jobs.len(), 1, "{jobs:?}");
        jobs.into_iter().next().unwrap()
    }
}

/// A child process that is killed if a test ends without waiting for it.
struct Proc {
    child: Option<Child>,
}

impl Proc {
    fn spawn(cmd: &mut Command) -> Self {
        Self {
            child: Some(cmd.spawn().expect("spawn child")),
        }
    }

    fn pid(&self) -> u32 {
        self.child.as_ref().unwrap().id()
    }

    fn wait(&mut self, within: Duration) -> std::process::ExitStatus {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.as_mut().unwrap().try_wait().unwrap() {
                self.child = None;
                return status;
            }
            assert!(
                started.elapsed() < within,
                "child still running after {within:?}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn signal(&self, signal: nix::sys::signal::Signal) {
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(self.pid() as i32), signal).unwrap();
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Generous: the host running these tests may be heavily loaded.
const PATIENCE: Duration = Duration::from_secs(240);

fn worker(host: &Host, queue: &Path, id: &str, log: &str) -> Proc {
    worker_with(host, queue, id, log, "10", false)
}

fn worker_with(
    host: &Host,
    queue: &Path,
    id: &str,
    log: &str,
    lease_secs: &str,
    other_code: bool,
) -> Proc {
    let q = queue.display().to_string();
    let work = host.path().join(format!("work-{id}")).display().to_string();
    let mut cmd = host.command(
        &[
            "cloud",
            "worker",
            "--queue",
            &q,
            "--id",
            id,
            "--lease-secs",
            lease_secs,
            "--poll-ms",
            "200",
            "--work-dir",
            &work,
        ],
        log,
    );
    if other_code {
        cmd.env(IDENTITY_ENV, "other");
    }
    Proc::spawn(&mut cmd)
}

fn wait_for(what: &str, within: Duration, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < within, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(root).ok()?.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|n| n == name) {
            return Some(path);
        }
    }
    None
}

fn pids_in(body: &str) -> Vec<u32> {
    body.lines()
        .filter_map(|line| {
            line.split_once("pid=")
                .map(|(_, pid)| pid.trim().parse().unwrap())
        })
        .collect()
}

/// Where this test's queue lives: a directory of `host`, or, with
/// `BLUT_TEST_E2E_ON_S3=1`, a fresh prefix of the bucket in
/// `BLUT_TEST_S3_QUEUE` (with the `AWS_*` variables set). Children inherit
/// those, and a URL passes through a `Path` unchanged.
fn queue_for(host: &Host) -> PathBuf {
    match (
        std::env::var("BLUT_TEST_E2E_ON_S3").as_deref(),
        std::env::var("BLUT_TEST_S3_QUEUE"),
    ) {
        (Ok("1"), Ok(base)) => PathBuf::from(format!(
            "{}/e2e-{}",
            base.trim_end_matches('/'),
            uuid::Uuid::new_v4()
        )),
        _ => host.path().join("queue"),
    }
}

fn queue_jobs(queue: &Path) -> Vec<blut::cloud::object_queue::ActiveJob> {
    let location = blut::cloud::location::CloudLocation::open(
        &queue.display().to_string(),
        blut::cloud::object_queue::ObjectQueueOptions::default(),
    )
    .unwrap();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(location.queue().list_active())
        .unwrap()
}

// ── the tests ───────────────────────────────────────────────────────────

/// Three stages, two workers, one submitter: every stage runs in a worker
/// process, the result comes back to the submitter's job, and the job records
/// the queue and its cloud jobs. `queue` is a directory or an `s3://` URL.
fn every_stage_runs_on_the_workers(submit_host: &Host, queue: &Path) {
    let worker_host = Host::new();
    let w1 = worker(&worker_host, queue, "w1", "w1.log");
    let w2 = worker(&worker_host, queue, "w2", "w2.log");
    let workers = [w1.pid(), w2.pid()];

    let mut run = Proc::spawn(
        submit_host
            .command(
                &[
                    "recipe",
                    "run",
                    "cloud_e2e_chain",
                    "--args",
                    r#"{"text":"over the queue"}"#,
                    "--launcher",
                    "cloud",
                ],
                "submit.log",
            )
            .env("BLUT_CLOUD_QUEUE", queue)
            .env("BLUT_CLOUD_PICKUP_SECS", "120"),
    );
    let submitter = run.pid();
    let status = run.wait(PATIENCE);
    assert!(
        status.success(),
        "submitter failed:\n{}\n--- w1:\n{}\n--- w2:\n{}",
        submit_host.log("submit.log"),
        worker_host.log("w1.log"),
        worker_host.log("w2.log")
    );

    let job = submit_host.only_job();
    let job_dir = submit_host.jobs_dir().join(&job);
    let out = find_file(&job_dir, "b.txt").expect("the last stage's output is in the job");
    let body = std::fs::read_to_string(&out).unwrap();
    assert!(body.starts_with("over the queue\n"), "{body}");
    let pids = pids_in(&body);
    assert_eq!(pids.len(), 3, "three stages ran: {body}");
    for pid in &pids {
        assert_ne!(*pid, submitter, "a stage ran in the submitter: {body}");
        assert!(
            workers.contains(pid),
            "a stage ran outside the workers: {body}"
        );
    }

    // The job records where its stages went, for `blut cancel`/`plan resume`.
    let recorded = std::fs::read_to_string(job_dir.join("cloud_jobs")).unwrap();
    assert_eq!(recorded.lines().count(), 3, "{recorded}");
    assert_eq!(
        std::fs::read_to_string(job_dir.join("cloud_queue"))
            .unwrap()
            .trim(),
        queue.display().to_string()
    );
    let claimed = worker_host.log("w1.log").matches("claimed ").count()
        + worker_host.log("w2.log").matches("claimed ").count();
    assert_eq!(claimed, 3, "each stage was claimed once");

    // SIGTERM: an idle worker stops cleanly.
    for (proc, log) in [(w1, "w1.log"), (w2, "w2.log")] {
        let mut proc = proc;
        proc.signal(nix::sys::signal::Signal::SIGTERM);
        assert!(proc.wait(PATIENCE).success(), "{}", worker_host.log(log));
        assert!(
            worker_host.log(log).contains("stopped:"),
            "{}",
            worker_host.log(log)
        );
    }

    // `blut cloud queue` sees nothing left.
    let mut list = Proc::spawn(&mut submit_host.command(
        &["cloud", "queue", "--queue", &queue.display().to_string()],
        "queue.log",
    ));
    assert!(list.wait(PATIENCE).success());
    assert!(
        submit_host.log("queue.log").contains("0 job(s) on"),
        "{}",
        submit_host.log("queue.log")
    );
}

#[test]
fn every_stage_runs_in_a_worker_process_not_the_submitter() {
    let host = Host::new();
    let queue = queue_for(&host);
    every_stage_runs_on_the_workers(&host, &queue);
    assert!(queue_jobs(&queue).is_empty(), "the queue is drained");
}

/// The same run over a bucket: `BLUT_TEST_S3_QUEUE=s3://bucket/prefix` and the
/// `AWS_*` variables (run against MinIO; skipped when unset).
#[cfg(feature = "s3")]
#[test]
fn every_stage_runs_on_the_workers_over_an_s3_queue() {
    let Ok(base) = std::env::var("BLUT_TEST_S3_QUEUE") else {
        eprintln!("skipped: BLUT_TEST_S3_QUEUE is not set");
        return;
    };
    let host = Host::new();
    let queue = format!(
        "{}/e2e-{}",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4()
    );
    every_stage_runs_on_the_workers(&host, Path::new(&queue));
}

/// No worker: the stage is withdrawn after the pickup window and the run
/// fails saying so. Nothing runs on the submitting machine instead.
#[test]
fn with_no_worker_the_run_fails_after_the_pickup_window() {
    let host = Host::new();
    let queue = queue_for(&host);
    let started = Instant::now();
    let mut run = Proc::spawn(
        host.command(
            &[
                "recipe",
                "run",
                "cloud_e2e_chain",
                "--args",
                r#"{"text":"nobody home"}"#,
                "--launcher",
                "cloud",
            ],
            "submit.log",
        )
        .env("BLUT_CLOUD_QUEUE", &queue)
        .env("BLUT_CLOUD_PICKUP_SECS", "3"),
    );
    let status = run.wait(PATIENCE);
    let log = host.log("submit.log");
    assert!(!status.success(), "{log}");
    assert!(
        log.contains("no cloud worker claimed stage 'e2e-seed'"),
        "{log}"
    );
    assert!(started.elapsed() >= Duration::from_secs(3));
    let job_dir = host.jobs_dir().join(host.only_job());
    assert!(
        find_file(&job_dir, "seed.txt").is_none(),
        "the stage ran here instead"
    );
    assert!(queue_jobs(&queue).is_empty(), "the job was withdrawn");
}

/// A worker built from other code leaves the job and says why; the
/// submitter's failure carries that reason.
#[test]
fn a_worker_running_other_code_does_not_claim_the_stage() {
    let submit_host = Host::new();
    let worker_host = Host::new();
    let queue = queue_for(&submit_host);
    let _other = worker_with(&worker_host, &queue, "stale", "stale.log", "10", true);
    let mut run = Proc::spawn(
        submit_host
            .command(
                &[
                    "recipe",
                    "run",
                    "cloud_e2e_chain",
                    "--args",
                    r#"{"text":"wrong build"}"#,
                    "--launcher",
                    "cloud",
                ],
                "submit.log",
            )
            .env("BLUT_CLOUD_QUEUE", &queue)
            .env("BLUT_CLOUD_PICKUP_SECS", "8"),
    );
    let status = run.wait(PATIENCE);
    let log = submit_host.log("submit.log");
    assert!(!status.success(), "{log}");
    assert!(log.contains("no cloud worker claimed"), "{log}");
    assert!(
        log.contains("worker stale: code identity differs"),
        "the refusal reaches the submitter: {log}"
    );
    let worker_log = worker_host.log("stale.log");
    assert!(!worker_log.contains("claimed "), "{worker_log}");
}

/// The submitter is SIGKILLed while a worker runs its stage. `plan resume`
/// refuses while that stage still runs; `blut cancel` stops it on the
/// worker; afterwards nothing blocks the resume.
#[test]
fn blut_cancel_stops_a_running_cloud_stage_after_the_submitter_is_gone() {
    let submit_host = Host::new();
    let worker_host = Host::new();
    let queue = queue_for(&submit_host);
    let markers = submit_host.path().join("markers");
    std::fs::create_dir_all(&markers).unwrap();
    let w1 = worker(&worker_host, &queue, "w1", "w1.log");
    let args = serde_json::json!({ "secs": 600, "marker_dir": markers }).to_string();
    let run = Proc::spawn(
        submit_host
            .command(
                &[
                    "recipe",
                    "run",
                    "cloud_e2e_hold",
                    "--args",
                    &args,
                    "--launcher",
                    "cloud",
                ],
                "submit.log",
            )
            .env("BLUT_CLOUD_QUEUE", &queue),
    );
    let started = markers.join(format!("started-{}", w1.pid()));
    wait_for("the worker to start the held stage", PATIENCE, || {
        started.exists()
    });
    drop(run); // SIGKILL: the submitter is gone, its stage still runs.
    let job = submit_host.only_job();

    let mut resume = Proc::spawn(
        submit_host
            .command(&["plan", "resume", &job], "resume.log")
            .env("BLUT_CLOUD_QUEUE", &queue),
    );
    assert!(!resume.wait(PATIENCE).success());
    let log = submit_host.log("resume.log");
    assert!(
        log.contains("still has cloud job(s)") && log.contains(&format!("blut cancel {job}")),
        "{log}"
    );

    // `blut cancel` reopens the queue from the job's record.
    let mut cancel = Proc::spawn(&mut submit_host.command(&["cancel", &job], "cancel.log"));
    assert!(
        cancel.wait(PATIENCE).success(),
        "{}",
        submit_host.log("cancel.log")
    );
    assert!(
        submit_host
            .log("cancel.log")
            .contains("cancelled cloud job(s)"),
        "{}",
        submit_host.log("cancel.log")
    );
    let stopped = markers.join(format!("stopped-{}", w1.pid()));
    wait_for("the worker to stop the cancelled stage", PATIENCE, || {
        stopped.exists()
    });
    assert!(!markers.join(format!("finished-{}", w1.pid())).exists());
    wait_for("the worker to report it", PATIENCE, || {
        worker_host.log("w1.log").contains("stopped  ")
    });
    assert!(queue_jobs(&queue).is_empty());

    // Nothing is left to block a resume. (Resuming would hold for another
    // 600 s, so only the blocker is checked here.)
    let job_dir = submit_host.jobs_dir().join(&job);
    let rt = tokio::runtime::Runtime::new().unwrap();
    assert_eq!(
        rt.block_on(blut::cloud::records::resume_blocker(&job, &job_dir)),
        None
    );
}

/// The submitter dies after the first stage; the second finishes on the
/// worker with nobody waiting. `plan resume` refuses until it has finished,
/// then completes the job: the first stage from the job's cache, the second
/// submitted again.
#[test]
fn plan_resume_finishes_a_cloud_job_whose_submitter_died() {
    let submit_host = Host::new();
    let worker_host = Host::new();
    let queue = queue_for(&submit_host);
    let markers = submit_host.path().join("markers");
    std::fs::create_dir_all(&markers).unwrap();
    let w1 = worker(&worker_host, &queue, "w1", "w1.log");
    let args = serde_json::json!({ "secs": 4, "marker_dir": markers }).to_string();
    let run = Proc::spawn(
        submit_host
            .command(
                &[
                    "recipe",
                    "run",
                    "cloud_e2e_hold",
                    "--args",
                    &args,
                    "--launcher",
                    "cloud",
                ],
                "submit.log",
            )
            .env("BLUT_CLOUD_QUEUE", &queue),
    );
    let started = markers.join(format!("started-{}", w1.pid()));
    wait_for("the worker to start the held stage", PATIENCE, || {
        started.exists()
    });
    drop(run);
    let job = submit_host.only_job();
    let finished = markers.join(format!("finished-{}", w1.pid()));
    wait_for(
        "the orphaned stage to finish on the worker",
        PATIENCE,
        || finished.exists(),
    );
    // Its result is recorded on the queue, so the resume may go ahead.
    let job_dir = submit_host.jobs_dir().join(&job);
    let rt = tokio::runtime::Runtime::new().unwrap();
    wait_for("the queue to record the result", PATIENCE, || {
        rt.block_on(blut::cloud::records::resume_blocker(&job, &job_dir))
            .is_none()
    });
    std::fs::remove_file(&finished).unwrap();

    let mut resume = Proc::spawn(
        submit_host
            .command(&["plan", "resume", &job], "resume.log")
            .env("BLUT_CLOUD_QUEUE", &queue),
    );
    let resumer = resume.pid();
    assert!(
        resume.wait(PATIENCE).success(),
        "{}\n--- worker:\n{}",
        submit_host.log("resume.log"),
        worker_host.log("w1.log")
    );
    let log = submit_host.log("resume.log");
    assert!(log.contains("1 cache hits, 1 misses"), "{log}");
    assert!(finished.exists(), "the held stage ran again, on the worker");
    let out = find_file(&job_dir, "held.txt").expect("the resumed job has its output");
    let pids = pids_in(&std::fs::read_to_string(&out).unwrap());
    assert_eq!(pids, vec![w1.pid(), w1.pid()]);
    assert!(!pids.contains(&resumer));
    assert_eq!(
        std::fs::read_to_string(job_dir.join("cloud_jobs"))
            .unwrap()
            .lines()
            .count(),
        3,
        "seed and hold from the first run, hold again from the resume"
    );
}

/// A hold-recipe submitter that waits on the queue in `queue`.
fn submit_hold(host: &Host, queue: &Path, secs: u64, markers: &Path) -> Proc {
    let args = serde_json::json!({ "secs": secs, "marker_dir": markers }).to_string();
    Proc::spawn(
        host.command(
            &[
                "recipe",
                "run",
                "cloud_e2e_hold",
                "--args",
                &args,
                "--launcher",
                "cloud",
            ],
            "submit.log",
        )
        .env("BLUT_CLOUD_QUEUE", queue),
    )
}

fn markers_dir(host: &Host) -> PathBuf {
    let markers = host.path().join("markers");
    std::fs::create_dir_all(&markers).unwrap();
    markers
}

/// SIGTERM stops the running stage and hands it back at once: a second
/// worker finishes it, long before the first worker's lease could run out.
#[test]
fn a_worker_given_sigterm_releases_its_stage_to_another_worker() {
    let submit_host = Host::new();
    let worker_host = Host::new();
    let queue = queue_for(&submit_host);
    let markers = markers_dir(&submit_host);
    let mut first = worker_with(&worker_host, &queue, "first", "first.log", "600", false);
    let first_pid = first.pid();
    let mut run = submit_hold(&submit_host, &queue, 3, &markers);
    let started = markers.join(format!("started-{first_pid}"));
    wait_for("the first worker to start the stage", PATIENCE, || {
        started.exists()
    });

    first.signal(nix::sys::signal::Signal::SIGTERM);
    assert!(
        first.wait(PATIENCE).success(),
        "{}",
        worker_host.log("first.log")
    );
    let log = worker_host.log("first.log");
    assert!(log.contains("released "), "{log}");
    assert!(log.contains("1 released"), "{log}");
    assert!(markers.join(format!("stopped-{first_pid}")).exists());
    assert!(!markers.join(format!("finished-{first_pid}")).exists());

    let second = worker(&worker_host, &queue, "second", "second.log");
    assert!(
        run.wait(PATIENCE).success(),
        "{}\n--- second:\n{}",
        submit_host.log("submit.log"),
        worker_host.log("second.log")
    );
    assert!(markers.join(format!("finished-{}", second.pid())).exists());
    let job_dir = submit_host.jobs_dir().join(submit_host.only_job());
    let body = std::fs::read_to_string(find_file(&job_dir, "held.txt").unwrap()).unwrap();
    assert_eq!(pids_in(&body).last(), Some(&second.pid()), "{body}");
}

/// A worker killed outright releases nothing. Its lease runs out, another
/// worker claims the stage under a newer generation, and the run succeeds.
#[test]
fn a_killed_workers_stage_is_requeued_when_its_lease_runs_out() {
    let submit_host = Host::new();
    let worker_host = Host::new();
    let queue = queue_for(&submit_host);
    let markers = markers_dir(&submit_host);
    let mut doomed = worker_with(&worker_host, &queue, "doomed", "doomed.log", "2", false);
    let doomed_pid = doomed.pid();
    let mut run = submit_hold(&submit_host, &queue, 3, &markers);
    let started = markers.join(format!("started-{doomed_pid}"));
    wait_for("the worker to start the stage", PATIENCE, || {
        started.exists()
    });

    doomed.signal(nix::sys::signal::Signal::SIGKILL);
    doomed.wait(PATIENCE);
    assert!(!markers.join(format!("finished-{doomed_pid}")).exists());

    let heir = worker_with(&worker_host, &queue, "heir", "heir.log", "10", false);
    assert!(
        run.wait(PATIENCE).success(),
        "{}\n--- heir:\n{}",
        submit_host.log("submit.log"),
        worker_host.log("heir.log")
    );
    assert!(markers.join(format!("finished-{}", heir.pid())).exists());
    let log = worker_host.log("heir.log");
    assert!(log.contains("(e2e-hold), generation 2"), "{log}");
}

/// With the submitter alive and no worker, `blut cancel` ends the run and the
/// queued stage is withdrawn: a worker that turns up later finds nothing.
#[test]
fn blut_cancel_withdraws_a_queued_stage_and_ends_the_run() {
    let submit_host = Host::new();
    let queue = queue_for(&submit_host);
    let markers = markers_dir(&submit_host);
    let mut run = submit_hold(&submit_host, &queue, 600, &markers);
    wait_for("the seed stage to be queued", PATIENCE, || {
        queue_jobs(&queue).len() == 1
    });
    let job = submit_host.only_job();

    let mut cancel = Proc::spawn(&mut submit_host.command(&["cancel", &job], "cancel.log"));
    assert!(
        cancel.wait(PATIENCE).success(),
        "{}",
        submit_host.log("cancel.log")
    );
    run.wait(PATIENCE);
    assert!(queue_jobs(&queue).is_empty(), "the stage was withdrawn");
    assert_eq!(
        std::fs::read_to_string(submit_host.jobs_dir().join(&job).join("state"))
            .unwrap_or_default()
            .trim(),
        "cancelled",
        "{}",
        submit_host.log("submit.log")
    );
    let latecomer_host = Host::new();
    let mut latecomer = Proc::spawn(&mut latecomer_host.command(
        &[
            "cloud",
            "worker",
            "--queue",
            &queue.display().to_string(),
            "--idle-exit-secs",
            "1",
            "--poll-ms",
            "100",
        ],
        "late.log",
    ));
    assert!(latecomer.wait(PATIENCE).success());
    assert!(
        latecomer_host.log("late.log").contains("0 claimed"),
        "{}",
        latecomer_host.log("late.log")
    );
}

/// A cloud run uses nothing of the submitting machine's training slot: a
/// second run starts while the first is waiting on its worker.
#[test]
fn two_cloud_runs_share_one_submitting_machine() {
    let submit_host = Host::new();
    let worker_host = Host::new();
    let queue = queue_for(&submit_host);
    let markers = markers_dir(&submit_host);
    let holder = worker(&worker_host, &queue, "holder", "holder.log");
    let mut first = submit_hold(&submit_host, &queue, 5, &markers);
    let started = markers.join(format!("started-{}", holder.pid()));
    wait_for("the first run's held stage", PATIENCE, || started.exists());

    let _chain_worker = worker(&worker_host, &queue, "chain", "chain.log");
    let mut second = Proc::spawn(
        submit_host
            .command(
                &[
                    "recipe",
                    "run",
                    "cloud_e2e_chain",
                    "--args",
                    r#"{"text":"beside the first"}"#,
                    "--launcher",
                    "cloud",
                ],
                "second.log",
            )
            .env("BLUT_CLOUD_QUEUE", &queue),
    );
    assert!(
        second.wait(PATIENCE).success(),
        "{}",
        submit_host.log("second.log")
    );
    assert!(
        first.wait(PATIENCE).success(),
        "{}",
        submit_host.log("submit.log")
    );
}
