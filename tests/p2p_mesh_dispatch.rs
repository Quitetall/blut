// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `recipe run --launcher p2p` end to end, with every node its own process on
//! 127.0.0.1: workers are `blut p2p node`, the submitter is `blut recipe run
//! --launcher p2p`, and cancellation is `blut cancel`. This test binary plays
//! each part by re-executing itself with `CHILD_ENV` set and handing the
//! command line to `blut::cli::run_from`, with a test cookbook registered.
//!
//! The recipe is `seed → transform(1) → transform(2) → collect`. `seed` and
//! `collect` do not declare that they may leave their machine; `transform`
//! declares public data. Each stage appends `<stage> <round> <pid> <event>` to
//! a trace file and records its process id in the artifact it returns, so the
//! test can tell which process ran what, including stages that failed.

#![cfg(feature = "p2p")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use blut::backends::TrainingBackend;
use blut::framework::artifact::{Artifact, ContentHash};
use blut::framework::code_identity::BuildIdentity;
use blut::framework::compat::Compatible;
use blut::framework::cookbook::{Cookbook, Registry};
use blut::framework::execution::DataClassification;
use blut::framework::plan::Plan;
use blut::framework::resource::Resource;
use blut::framework::stage::{ErasedStageCtor, Stage, StageContext, StagePlacement};
use blut::framework::{RecipeError, StageError};
use blut::p2p::crypto::KeyPair;
use blut::p2p::peer::PeerId;
use blut::recipes::recipe::{Recipe, RecipeDef};
use serde::{Deserialize, Serialize};

const CHILD_ENV: &str = "BLUT_MESH_E2E_CHILD";

// ── the test cookbook ──────────────────────────────────────────────────────

struct E2eBackend;
impl TrainingBackend for E2eBackend {
    const ID: &'static str = "mesh-e2e";
    const DESCRIPTION: &'static str = "p2p mesh dispatch end-to-end test";
}

/// File-backed bytes plus the processes that produced them, in order.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Blob {
    path: PathBuf,
    content_hash: ContentHash,
    ran_on: Vec<u32>,
}

impl Artifact for Blob {
    const KIND: &'static str = "test.mesh_e2e.blob";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

fn trace(path: &str, line: String) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    writeln!(f, "{line}").unwrap();
}

fn seed_bytes(size: usize, salt: u64) -> Vec<u8> {
    let mut x = salt.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..size)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u8
        })
        .collect()
}

fn transform(bytes: &[u8], round: u32) -> Vec<u8> {
    bytes.iter().map(|b| b ^ (round as u8 + 1)).collect()
}

fn write_blob(ctx: &StageContext, name: &str, bytes: &[u8], ran_on: Vec<u32>) -> Blob {
    let path = ctx.stage_dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    Blob {
        content_hash: ContentHash::hash_file(&path).unwrap(),
        path,
        ran_on,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct SeedArgs {
    size: usize,
    salt: u64,
    trace: String,
}

struct Seed;
#[async_trait]
impl Stage for Seed {
    const NAME: &'static str = "mesh_e2e.seed";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Blob;
    type Args = SeedArgs;
    async fn run(&self, ctx: &StageContext, _: (), a: &SeedArgs) -> Result<Blob, StageError> {
        let pid = std::process::id();
        trace(&a.trace, format!("seed 0 {pid} done"));
        Ok(write_blob(
            ctx,
            "seed.bin",
            &seed_bytes(a.size, a.salt),
            vec![pid],
        ))
    }
}
impl Compatible<E2eBackend> for Seed {}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct TransformArgs {
    round: u32,
    trace: String,
    /// Hold this long (until cancelled) before transforming.
    hold_secs: u64,
}

struct Transform;
#[async_trait]
impl Stage for Transform {
    const NAME: &'static str = "mesh_e2e.transform";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Blob;
    type Output = Blob;
    type Args = TransformArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: Blob,
        a: &TransformArgs,
    ) -> Result<Blob, StageError> {
        let pid = std::process::id();
        trace(&a.trace, format!("transform {} {pid} started", a.round));
        if a.hold_secs > 0 {
            tokio::select! {
                _ = ctx.cancel.cancelled() => {
                    trace(&a.trace, format!("transform {} {pid} cancelled", a.round));
                    return Err(StageError::Cancelled);
                }
                _ = tokio::time::sleep(Duration::from_secs(a.hold_secs)) => {}
            }
        }
        let bytes = std::fs::read(&input.path).unwrap();
        let mut ran_on = input.ran_on.clone();
        ran_on.push(pid);
        let out = write_blob(ctx, "out.bin", &transform(&bytes, a.round), ran_on);
        trace(&a.trace, format!("transform {} {pid} done", a.round));
        Ok(out)
    }
    fn placement(&self, _: &TransformArgs) -> StagePlacement {
        StagePlacement::remote(DataClassification::Public)
    }
}
impl Compatible<E2eBackend> for Transform {}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct CollectArgs {
    trace: String,
    report: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Report {
    ran_on: Vec<u32>,
    hash: String,
    len: usize,
}

struct Collect;
#[async_trait]
impl Stage for Collect {
    const NAME: &'static str = "mesh_e2e.collect";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Blob;
    type Output = Blob;
    type Args = CollectArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: Blob,
        a: &CollectArgs,
    ) -> Result<Blob, StageError> {
        let pid = std::process::id();
        let bytes = std::fs::read(&input.path).unwrap();
        let mut ran_on = input.ran_on.clone();
        ran_on.push(pid);
        let report = Report {
            ran_on: ran_on.clone(),
            hash: ContentHash::of_bytes(&bytes).to_hex(),
            len: bytes.len(),
        };
        std::fs::write(&a.report, serde_json::to_vec(&report).unwrap()).unwrap();
        trace(&a.trace, format!("collect 0 {pid} done"));
        Ok(write_blob(ctx, "final.bin", &bytes, ran_on))
    }
}
impl Compatible<E2eBackend> for Collect {}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct E2eArgs {
    size: usize,
    salt: u64,
    trace: String,
    report: String,
    #[serde(default)]
    hold_round1_secs: u64,
}

#[derive(Default)]
struct MeshE2e;
impl Recipe for MeshE2e {
    const NAME: &'static str = "mesh_e2e";
    const DESCRIPTION: &'static str = "seed → transform → transform → collect";
    type Backend = E2eBackend;
    type Args = E2eArgs;
    fn compile(&self, a: E2eArgs) -> Result<Plan<(), E2eBackend>, RecipeError> {
        let args = serde_json::to_value(&a).unwrap();
        Ok(Plan::<(), E2eBackend>::new("mesh_e2e", args)
            .start(
                Seed,
                SeedArgs {
                    size: a.size,
                    salt: a.salt,
                    trace: a.trace.clone(),
                },
            )
            .then(
                Transform,
                TransformArgs {
                    round: 1,
                    trace: a.trace.clone(),
                    hold_secs: a.hold_round1_secs,
                },
            )
            .then(
                Transform,
                TransformArgs {
                    round: 2,
                    trace: a.trace.clone(),
                    hold_secs: 0,
                },
            )
            .then(
                Collect,
                CollectArgs {
                    trace: a.trace.clone(),
                    report: a.report,
                },
            )
            .finish())
    }
}

mod recipe_def {
    blut::register_recipe!(super::MeshE2e);
}

struct E2eCookbook {
    version: &'static str,
}

impl Cookbook for E2eCookbook {
    fn identity(&self) -> Option<BuildIdentity> {
        Some(BuildIdentity::new("mesh-e2e-cookbook", self.version))
    }
    fn name(&self) -> &'static str {
        "mesh_e2e"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        static R: &[&RecipeDef] = &[&recipe_def::DEF];
        R
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static S: &[(&str, ErasedStageCtor)] = &[
            (Seed::NAME, || std::sync::Arc::new(Seed)),
            (Transform::NAME, || std::sync::Arc::new(Transform)),
            (Collect::NAME, || std::sync::Arc::new(Collect)),
        ];
        S
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ChildSpec {
    /// The cookbook build this process claims (its code identity).
    version: String,
    argv: Vec<String>,
}

/// The child half: a no-op unless this process was started as a node.
#[test]
fn child_entry() {
    let Ok(spec) = std::env::var(CHILD_ENV) else {
        return;
    };
    let spec: ChildSpec = serde_json::from_str(&spec).unwrap();
    let mut registry = Registry::new();
    registry.register(Box::new(E2eCookbook {
        version: Box::leak(spec.version.into_boxed_str()),
    }));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let result = runtime.block_on(blut::cli::run_from(registry, spec.argv));
    match result {
        Ok(()) => {
            eprintln!("CHILD OK");
            std::process::exit(0)
        }
        Err(e) => {
            eprintln!("CHILD FAILED: {e:#}");
            std::process::exit(1)
        }
    }
}

// ── the parent half ────────────────────────────────────────────────────────

/// One process's private world: data dir (identity, store, registry), jobs.
struct Home {
    dir: PathBuf,
    key: KeyPair,
}

impl Home {
    fn new(root: &Path, name: &str) -> Self {
        let dir = root.join(name);
        let key = KeyPair::generate();
        let p2p = dir.join("data").join("p2p");
        std::fs::create_dir_all(&p2p).unwrap();
        std::fs::write(p2p.join("identity.key"), key.to_bytes()).unwrap();
        Self { dir, key }
    }

    fn pubkey_line(&self) -> String {
        let mut bytes = [0u8; 64];
        bytes[..32].copy_from_slice(self.key.verifying.as_bytes());
        bytes[32..].copy_from_slice(self.key.x25519_public.as_bytes());
        faster_hex::hex_string(&bytes)
    }

    fn peer_id(&self) -> String {
        PeerId::from_pubkey(&self.key.verifying).to_string()
    }

    fn jobs(&self) -> PathBuf {
        self.dir.join("jobs")
    }

    fn spawn(&self, version: &str, argv: &[&str], log: &str) -> Node {
        let spec = ChildSpec {
            version: version.into(),
            argv: std::iter::once("blut")
                .chain(argv.iter().copied())
                .map(String::from)
                .collect(),
        };
        let log_path = self.dir.join(log);
        let out = std::fs::File::create(&log_path).unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, serde_json::to_string(&spec).unwrap())
            .env("LAMU_TRAIN_DATA_DIR", self.dir.join("data"))
            .env("LAMU_TRAIN_JOBS_DIR", self.jobs())
            .env("XDG_DATA_HOME", self.dir.join("xdg-data"))
            .env("XDG_CACHE_HOME", self.dir.join("xdg-cache"))
            .env("XDG_CONFIG_HOME", self.dir.join("xdg-config"))
            .env("BLUT_PARTITIONS_DIR", self.dir.join("partitions"))
            .env("BLUT_SCHED_DEVICES", "0")
            .env("CUDA_VISIBLE_DEVICES", "")
            .env("BLUT_NO_CONTAIN", "1")
            .env("RUST_LOG", "info")
            .env_remove("BLUT_EXECUTOR")
            .stdin(Stdio::null())
            .stdout(out.try_clone().unwrap())
            .stderr(out)
            .spawn()
            .unwrap();
        Node {
            child: Some(child),
            log: log_path,
        }
    }
}

struct Node {
    child: Option<std::process::Child>,
    log: PathBuf,
}

impl Node {
    fn pid(&self) -> u32 {
        self.child.as_ref().unwrap().id()
    }

    fn log(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }

    /// Wait for the process to exit; `true` if it exited 0.
    fn wait(&mut self, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        let child = self.child.as_mut().unwrap();
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                return status.success();
            }
            assert!(
                Instant::now() < deadline,
                "process did not exit within {within:?}; log:\n{}",
                self.log()
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        self.kill();
    }
}

fn wait_until(within: Duration, what: &str, mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + within;
    while !check() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// A worker that runs stages for `submitter`, or for nobody.
struct Worker {
    home: Home,
    node: Node,
    addr: String,
}

impl Worker {
    fn start(root: &Path, name: &str, version: &str, submitter: Option<&Home>) -> Self {
        let home = Home::new(root, name);
        let accept = submitter.map(|s| faster_hex::hex_string(s.key.verifying.as_bytes()));
        let mut argv = vec!["p2p", "node", "--listen", "127.0.0.1:0", "--no-gossip"];
        if let Some(accept) = &accept {
            argv.extend(["--coordinator-pubkey", accept.as_str()]);
        }
        let node = home.spawn(version, &argv, "node.log");
        let mut addr = String::new();
        wait_until(Duration::from_secs(60), "a worker to listen", || {
            if let Some(line) = node.log().lines().find(|l| l.contains("listening on ")) {
                addr = line
                    .rsplit("listening on ")
                    .next()
                    .unwrap()
                    .trim()
                    .to_string();
            }
            !addr.is_empty() && node.log().contains("node running")
        });
        Self { home, node, addr }
    }

    fn seed(&self) -> String {
        format!("{}@{}", self.addr, self.home.pubkey_line())
    }
}

struct Run {
    root: tempfile::TempDir,
    submitter: Home,
}

impl Run {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let submitter = Home::new(root.path(), "submitter");
        Self { root, submitter }
    }

    fn trace(&self) -> PathBuf {
        self.root.path().join("trace.txt")
    }

    fn report(&self) -> PathBuf {
        self.root.path().join("report.json")
    }

    fn trace_lines(&self) -> Vec<String> {
        std::fs::read_to_string(self.trace())
            .unwrap_or_default()
            .lines()
            .map(String::from)
            .collect()
    }

    fn recipe_run(&self, workers: &[&Worker], hold: u64, wait_secs: &str) -> Node {
        let args = serde_json::json!({
            "size": 300_000,
            "salt": 7,
            "trace": self.trace(),
            "report": self.report(),
            "hold_round1_secs": hold,
        })
        .to_string();
        let seeds: Vec<String> = workers.iter().map(|w| w.seed()).collect();
        let mut argv = vec![
            "recipe",
            "run",
            "mesh_e2e",
            "--args",
            args.as_str(),
            "--launcher",
            "p2p",
            "--p2p-peer-wait",
            wait_secs,
        ];
        for seed in &seeds {
            argv.extend(["--p2p-seed", seed.as_str()]);
        }
        self.submitter.spawn("1", &argv, "submitter.log")
    }

    fn job_ids(&self) -> Vec<String> {
        std::fs::read_dir(self.submitter.jobs())
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `blut jobs --json` as the submitter sees it.
    fn job_state(&self, id: &str) -> String {
        let mut cli = self.submitter.spawn("1", &["jobs", "--json"], "jobs.log");
        assert!(cli.wait(Duration::from_secs(60)), "{}", cli.log());
        let log = cli.log();
        let start = log.find('[').expect("a JSON array");
        let end = log.rfind(']').expect("a JSON array");
        let jobs: serde_json::Value = serde_json::from_str(&log[start..=end]).unwrap();
        jobs.as_array()
            .unwrap()
            .iter()
            .find(|j| j["id"] == id || j["job_id"] == id)
            .map(|j| j["state"].as_str().unwrap_or_default().to_string())
            .unwrap_or_else(|| panic!("no job {id} in {log}"))
    }
}

fn pid_of(line: &str) -> u32 {
    line.split_whitespace().nth(2).unwrap().parse().unwrap()
}

/// Eligible stages run on the workers (one each), the stages that must stay
/// run in the submitting process, the output comes back verified, and the
/// submitter's peer registry counts each worker's completed task.
#[test]
fn eligible_stages_run_on_mesh_workers_and_the_rest_stays_home() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let run = Run::new();
    let w1 = Worker::start(run.root.path(), "w1", "1", Some(&run.submitter));
    let w2 = Worker::start(run.root.path(), "w2", "1", Some(&run.submitter));
    let mut submitter = run.recipe_run(&[&w1, &w2], 0, "30");
    let ok = submitter.wait(Duration::from_secs(240));
    assert!(ok, "the run failed:\n{}", submitter.log());

    let report: Report = serde_json::from_slice(&std::fs::read(run.report()).unwrap()).unwrap();
    let here = submitter.pid();
    assert_eq!(report.ran_on.len(), 4, "{report:?}");
    assert_eq!(report.ran_on[0], here, "seed stays home: {report:?}");
    assert_eq!(report.ran_on[3], here, "collect stays home: {report:?}");
    let remote = [report.ran_on[1], report.ran_on[2]];
    assert!(
        !remote.contains(&here),
        "a transform ran in the submitter: {report:?}"
    );
    let mut workers = vec![w1.node.pid(), w2.node.pid()];
    let mut used = remote.to_vec();
    workers.sort();
    used.sort();
    assert_eq!(used, workers, "one transform on each worker: {report:?}");

    let expected = transform(&transform(&seed_bytes(300_000, 7), 1), 2);
    assert_eq!(report.len, expected.len());
    assert_eq!(
        report.hash,
        ContentHash::of_bytes(&expected).to_hex(),
        "bytes came back intact"
    );
    for line in run
        .trace_lines()
        .iter()
        .filter(|l| l.starts_with("transform"))
    {
        assert_ne!(pid_of(line), here, "{line}");
    }
    let log = submitter.log();
    assert!(log.contains("runs on this machine"), "{log}");
    assert!(log.contains("→ peer"), "{log}");

    // The submitter's registry now holds each worker with one completed task.
    let mut peers = run
        .submitter
        .spawn("1", &["p2p", "peers", "list", "--json"], "peers.log");
    assert!(peers.wait(Duration::from_secs(60)), "{}", peers.log());
    let log = peers.log();
    let start = log.find('[').expect("a JSON array");
    let end = log.rfind(']').expect("a JSON array");
    let listed: serde_json::Value = serde_json::from_str(&log[start..=end]).unwrap();
    for w in [&w1, &w2] {
        let entry = listed
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == w.home.peer_id())
            .unwrap_or_else(|| panic!("worker {} missing from {log}", w.home.peer_id()));
        assert_eq!(entry["tasks_completed"], 1, "{entry}");
        assert!(entry["reputation"].as_f64().unwrap() > 0.5, "{entry}");
    }
}

/// No peer can take the eligible stage (the only one does not accept this
/// submitter, or runs other code): the run fails with a reason, and the stage
/// ran nowhere, least of all in the submitting process.
#[test]
fn no_capable_peer_fails_the_run_and_nothing_runs_here() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    for (case, version, accept, expect) in [
        (
            "not accepted",
            "1",
            false,
            "does not accept stages from this machine",
        ),
        ("other code", "2", true, "mesh-e2e-cookbook 2"),
    ] {
        let run = Run::new();
        let w = Worker::start(
            run.root.path(),
            "w",
            version,
            accept.then_some(&run.submitter),
        );
        let mut submitter = run.recipe_run(&[&w], 0, "3");
        let ok = submitter.wait(Duration::from_secs(180));
        let log = submitter.log();
        assert!(!ok, "{case}: the run must fail:\n{log}");
        assert!(
            log.contains("no mesh peer could take stage 'mesh_e2e.transform'"),
            "{case}: {log}"
        );
        assert!(log.contains(expect), "{case}: {log}");
        let trace = run.trace_lines();
        assert!(
            trace.iter().all(|l| !l.starts_with("transform")),
            "{case}: a transform ran somewhere: {trace:?}"
        );
        assert!(
            trace.iter().any(|l| l.starts_with("seed")),
            "{case}: {trace:?}"
        );
        let jobs = run.job_ids();
        assert_eq!(jobs.len(), 1, "{case}: {jobs:?}");
        assert_eq!(run.job_state(&jobs[0]), "failed", "{case}");
    }
}

/// `blut cancel <job>` stops the stage running on the worker.
#[test]
fn blut_cancel_stops_the_remote_stage() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let run = Run::new();
    let w = Worker::start(run.root.path(), "w", "1", Some(&run.submitter));
    let mut submitter = run.recipe_run(&[&w], 600, "30");
    wait_until(
        Duration::from_secs(120),
        "the remote stage to start",
        || {
            run.trace_lines()
                .iter()
                .any(|l| l.starts_with("transform 1") && l.ends_with("started"))
        },
    );
    let started = run
        .trace_lines()
        .into_iter()
        .find(|l| l.starts_with("transform 1"))
        .unwrap();
    assert_eq!(pid_of(&started), w.node.pid(), "it runs on the worker");
    let jobs = run.job_ids();
    assert_eq!(jobs.len(), 1, "{jobs:?}");
    let mut cancel = run
        .submitter
        .spawn("1", &["cancel", jobs[0].as_str()], "cancel.log");
    assert!(cancel.wait(Duration::from_secs(60)), "{}", cancel.log());
    wait_until(
        Duration::from_secs(30),
        "the worker to stop the stage",
        || {
            run.trace_lines()
                .iter()
                .any(|l| l.starts_with("transform 1") && l.ends_with("cancelled"))
        },
    );
    assert!(
        !submitter.wait(Duration::from_secs(60)),
        "a cancelled run fails"
    );
    assert_eq!(run.job_state(&jobs[0]), "cancelled");
    assert!(
        run.trace_lines()
            .iter()
            .all(|l| !l.starts_with("transform 2")),
        "nothing after the cancelled stage ran"
    );
}

/// A worker whose submitter dies without a word (SIGKILL) stops the stage
/// once the connection times out.
#[test]
fn a_worker_stops_the_stage_when_its_submitter_dies() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let run = Run::new();
    let w = Worker::start(run.root.path(), "w", "1", Some(&run.submitter));
    let mut submitter = run.recipe_run(&[&w], 600, "30");
    wait_until(
        Duration::from_secs(120),
        "the remote stage to start",
        || {
            run.trace_lines()
                .iter()
                .any(|l| l.starts_with("transform 1") && l.ends_with("started"))
        },
    );
    submitter.kill();
    // The QUIC idle timeout is 30 s.
    wait_until(
        Duration::from_secs(60),
        "the worker to stop the stage",
        || {
            run.trace_lines()
                .iter()
                .any(|l| l.starts_with("transform 1") && l.ends_with("cancelled"))
        },
    );
    let log = w.node.log();
    assert!(log.contains("stopped stage 'mesh_e2e.transform'"), "{log}");
}
