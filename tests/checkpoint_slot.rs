// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Mid-stage checkpoint slots (CONTRACT.md §8) through the public executor API.
//!
//! The stage under test stands in for a trainer: on a crashing attempt it
//! writes a step-5 "checkpoint" and the `state.json` marker the contract
//! describes, then fails. What it sees in `StageContext::checkpoint` on the
//! next attempt is the engine's decision.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use blut::backends::TrainingBackend;
use blut::framework::artifact::{Artifact, ContentHash};
use blut::framework::compat::Compatible;
use blut::framework::executor::{ExecCtx, ParallelExecutor};
use blut::framework::plan::Plan;
use blut::framework::resource::Resource;
use blut::framework::retry::{Backoff, RetryOn, RetryPolicy};
use blut::framework::stage::{Stage, StageContext};
use blut::framework::stage_checkpoint::{self as sc, CheckpointAction};
use blut::framework::status::{HostedEvent, StageEvent};
use blut::framework::{CompiledPlan, StageError};
use serde::{Deserialize, Serialize};

/// Tests share the statics below; run them one at a time.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static RUNS: AtomicU32 = AtomicU32::new(0);
static SEEN: Mutex<Vec<Option<sc::CheckpointSlot>>> = Mutex::new(Vec::new());
static STARTED: AtomicBool = AtomicBool::new(false);
static RELEASE: AtomicBool = AtomicBool::new(false);
/// Attempts numbered below this crash after writing a checkpoint. Kept out of
/// the stage arguments, which are part of the cache key.
static CRASH_BEFORE: AtomicU32 = AtomicU32::new(0);

fn reset(crash_before: u32) {
    CRASH_BEFORE.store(crash_before, Ordering::SeqCst);
    RUNS.store(0, Ordering::SeqCst);
    SEEN.lock().unwrap().clear();
    STARTED.store(false, Ordering::SeqCst);
    RELEASE.store(false, Ordering::SeqCst);
}

fn seen() -> Vec<Option<sc::CheckpointSlot>> {
    SEEN.lock().unwrap().clone()
}

struct TestBackend;
impl TrainingBackend for TestBackend {
    const ID: &'static str = "checkpoint-slot-test";
    const DESCRIPTION: &'static str = "checkpoint slot integration tests";
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Number {
    n: u32,
}
impl Artifact for Number {
    const KIND: &'static str = "test.number";
    const SCHEMA: u32 = 1;
    const INLINE: bool = true;
    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&self.n.to_le_bytes())
    }
    fn primary_path(&self) -> &Path {
        Path::new(".")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct SeedArgs {
    n: u32,
}

struct Seed;
#[async_trait]
impl Stage for Seed {
    const NAME: &'static str = "seed";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Number;
    type Args = SeedArgs;
    async fn run(&self, _: &StageContext, _: (), a: &SeedArgs) -> Result<Number, StageError> {
        Ok(Number { n: a.n })
    }
}
impl Compatible<TestBackend> for Seed {}

/// Which PID the crashing attempt writes into its marker.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
enum MarkerPid {
    /// This test process: alive, on this host.
    Own,
    /// A child that has already exited.
    Dead,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct TrainArgs {
    tag: u32,
    /// Whether the "trainer" follows the contract at all.
    cooperate: bool,
    marker_pid: MarkerPid,
    /// Park until `RELEASE` (concurrency test).
    block: bool,
}

impl TrainArgs {
    fn new(marker_pid: MarkerPid) -> Self {
        Self {
            tag: 1,
            cooperate: true,
            marker_pid,
            block: false,
        }
    }
}

fn dead_pid() -> u32 {
    let mut child = std::process::Command::new("true").spawn().unwrap();
    let pid = child.id();
    child.wait().unwrap();
    pid
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn write_marker(slot: &sc::CheckpointSlot, pid: u32, step: Option<u64>, resumed: Option<u64>) {
    let marker = serde_json::json!({
        "status": "running",
        "run_id": slot.run_id,
        "pid": pid,
        "host": sc::hostname(),
        "heartbeat_unix": now(),
        "step": step,
        "epoch": 1,
        "attempt_id": slot.attempt_id,
        "resumed_from_step": resumed,
    });
    std::fs::write(slot.dir.join("state.json"), marker.to_string()).unwrap();
}

struct SlotTrainer;
#[async_trait]
impl Stage for SlotTrainer {
    const NAME: &'static str = "slot_trainer";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const DETERMINISTIC: bool = false;
    type Input = Number;
    type Output = Number;
    type Args = TrainArgs;

    fn checkpoint_slot(&self, _: &TrainArgs) -> bool {
        true
    }

    async fn run(
        &self,
        ctx: &StageContext,
        input: Number,
        a: &TrainArgs,
    ) -> Result<Number, StageError> {
        let run = RUNS.fetch_add(1, Ordering::SeqCst);
        SEEN.lock().unwrap().push(ctx.checkpoint.clone());
        if a.block {
            STARTED.store(true, Ordering::SeqCst);
            while !RELEASE.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        let slot = ctx
            .checkpoint
            .as_ref()
            .expect("an opted-in stage gets a slot");
        if a.cooperate {
            if slot.resume {
                // A trainer loads the newest checkpoint and records the step.
                assert!(slot.dir.join("step-0000000005.pt").exists());
                write_marker(slot, std::process::id(), Some(5), Some(5));
            } else {
                assert!(
                    !slot.dir.join("step-0000000005.pt").exists(),
                    "a fresh attempt must not see an old checkpoint"
                );
            }
        }
        if run < CRASH_BEFORE.load(Ordering::SeqCst) {
            if a.cooperate {
                std::fs::write(slot.dir.join("step-0000000005.pt"), b"weights at step 5").unwrap();
                let pid = match a.marker_pid {
                    MarkerPid::Own => std::process::id(),
                    MarkerPid::Dead => dead_pid(),
                };
                write_marker(slot, pid, Some(5), None);
            }
            return Err(StageError::Backend(anyhow::anyhow!("node lost at step 5")));
        }
        Ok(Number { n: input.n + 100 })
    }
}
impl Compatible<TestBackend> for SlotTrainer {}

struct NoSlot;
#[async_trait]
impl Stage for NoSlot {
    const NAME: &'static str = "no_slot";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Number;
    type Output = Number;
    type Args = SeedArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: Number,
        _: &SeedArgs,
    ) -> Result<Number, StageError> {
        SEEN.lock().unwrap().push(ctx.checkpoint.clone());
        Ok(input)
    }
}
impl Compatible<TestBackend> for NoSlot {}

fn plan(seed: u32, args: TrainArgs, retries: u32) -> CompiledPlan {
    Plan::<(), TestBackend>::new("checkpoint-slot", serde_json::json!({}))
        .start(Seed, SeedArgs { n: seed })
        .then(SlotTrainer, args)
        .with_retry(RetryPolicy {
            max_attempts: retries,
            backoff: Backoff::None,
            retry_on: RetryOn::Transient,
        })
        .finish()
        .into_compiled()
}

struct Workspace {
    _td: tempfile::TempDir,
    root: PathBuf,
    jobs: PathBuf,
}

impl Workspace {
    fn new() -> Self {
        let td = tempfile::tempdir().unwrap();
        let root = td.path().join("checkpoints");
        let jobs = td.path().join("jobs");
        Self {
            _td: td,
            root,
            jobs,
        }
    }

    fn ctx(&self, job: &str) -> ExecCtx {
        let dir = self.jobs.join(job);
        std::fs::create_dir_all(&dir).unwrap();
        ExecCtx::new(dir).with_checkpoint_root(self.root.clone())
    }

    fn events(&self, job: &str) -> Vec<(CheckpointAction, Option<u64>)> {
        std::fs::read_to_string(self.jobs.join(job).join("status.jsonl"))
            .unwrap()
            .lines()
            .filter_map(|l| serde_json::from_str::<HostedEvent>(l).ok())
            .filter_map(|h| match h.event {
                StageEvent::StageCheckpoint { action, step, .. } => Some((action, step)),
                _ => None,
            })
            .collect()
    }

    fn slots(&self) -> Vec<PathBuf> {
        let tenant = self.root.join("default");
        let Ok(rd) = std::fs::read_dir(&tenant) else {
            return Vec::new();
        };
        let mut out: Vec<_> = rd.map(|e| e.unwrap().path()).collect();
        out.sort();
        out
    }
}

fn slot_of(s: &Option<sc::CheckpointSlot>) -> &sc::CheckpointSlot {
    s.as_ref().expect("slot")
}

async fn run(plan: CompiledPlan, ctx: ExecCtx) -> Result<blut::framework::PlanResult, String> {
    tokio::time::timeout(
        Duration::from_secs(60),
        ParallelExecutor::execute(plan, ctx),
    )
    .await
    .expect("plan hung")
    .map_err(|e| format!("{e:#}"))
}

fn has_file(dir: &Path, name: &str) -> bool {
    dir.join(name).exists()
}

#[tokio::test]
async fn a_retry_resumes_from_the_checkpoint_and_the_slot_is_removed_after_commit() {
    let _g = SERIAL.lock().await;
    reset(1);
    let ws = Workspace::new();
    run(plan(1, TrainArgs::new(MarkerPid::Own), 2), ws.ctx("job-a"))
        .await
        .expect("the retry succeeds");
    let seen = seen();
    assert_eq!(seen.len(), 2);
    assert!(!slot_of(&seen[0]).resume, "first attempt starts fresh");
    assert!(slot_of(&seen[1]).resume, "the retry resumes");
    assert_eq!(slot_of(&seen[1]).offered_step, Some(5));
    assert_eq!(
        slot_of(&seen[0]).dir,
        slot_of(&seen[1]).dir,
        "one slot per stage"
    );
    assert!(!slot_of(&seen[1]).dir.exists(), "slot deleted after commit");
    assert_eq!(
        ws.events("job-a"),
        vec![
            (CheckpointAction::Fresh, None),
            (CheckpointAction::ResumeOffered, Some(5)),
            (CheckpointAction::Resumed, Some(5)),
            (CheckpointAction::Cleaned, None),
        ]
    );
}

#[tokio::test]
async fn plan_resume_after_the_process_died_resumes_from_the_checkpoint() {
    let _g = SERIAL.lock().await;
    reset(1);
    let ws = Workspace::new();
    // No retries: the first execution fails and keeps its slot.
    let err = run(plan(1, TrainArgs::new(MarkerPid::Dead), 1), ws.ctx("job-b"))
        .await
        .unwrap_err();
    assert!(err.contains("node lost"), "{err}");
    let dir = slot_of(&seen()[0]).dir.clone();
    assert!(
        has_file(&dir, "step-0000000005.pt"),
        "a failed stage keeps its checkpoints"
    );

    // A new execution of the same job (what `plan resume` runs), whose
    // trainer from before is gone.
    run(plan(1, TrainArgs::new(MarkerPid::Dead), 1), ws.ctx("job-b"))
        .await
        .expect("the resumed execution succeeds");
    let seen = seen();
    assert_eq!(seen.len(), 2);
    assert!(slot_of(&seen[1]).resume);
    assert_eq!(slot_of(&seen[1]).offered_step, Some(5));
    assert!(!dir.exists());
}

#[tokio::test]
async fn a_live_trainer_from_an_earlier_execution_is_not_raced() {
    let _g = SERIAL.lock().await;
    reset(1);
    let ws = Workspace::new();
    // The marker names this test process: alive, on this host, fresh
    // heartbeat. That is what an orphaned trainer looks like after its BLUT
    // process was killed.
    run(plan(1, TrainArgs::new(MarkerPid::Own), 1), ws.ctx("job-c"))
        .await
        .unwrap_err();
    let dir = slot_of(&seen()[0]).dir.clone();
    let err = run(plan(1, TrainArgs::new(MarkerPid::Own), 1), ws.ctx("job-c"))
        .await
        .unwrap_err();
    assert!(err.contains("busy"), "{err}");
    assert_eq!(
        RUNS.load(Ordering::SeqCst),
        1,
        "the stage must not have run again"
    );
    assert!(
        has_file(&dir, "step-0000000005.pt"),
        "the checkpoint is untouched"
    );
}

#[tokio::test]
async fn a_concurrent_run_of_the_same_stage_is_refused_and_does_not_clobber() {
    let _g = SERIAL.lock().await;
    reset(0);
    let ws = Workspace::new();
    let mut blocking = TrainArgs::new(MarkerPid::Own);
    blocking.block = true;
    let first = tokio::spawn(run(plan(1, blocking.clone(), 1), ws.ctx("job-d1")));
    while !STARTED.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let dir = slot_of(&seen()[0]).dir.clone();
    std::fs::write(dir.join("step-0000000003.pt"), b"first run's checkpoint").unwrap();
    // Same stage, args and input from another job: same slot.
    let err = run(plan(1, blocking, 1), ws.ctx("job-d2"))
        .await
        .unwrap_err();
    assert!(err.contains("locked by another BLUT process"), "{err}");
    assert_eq!(RUNS.load(Ordering::SeqCst), 1);
    assert!(
        has_file(&dir, "step-0000000003.pt"),
        "the refused run cleared nothing"
    );
    RELEASE.store(true, Ordering::SeqCst);
    first.await.unwrap().expect("the first run finishes");
    assert!(!dir.exists());
}

#[tokio::test]
async fn a_trainer_that_ignores_the_slot_runs_fresh_every_attempt() {
    let _g = SERIAL.lock().await;
    reset(1);
    let ws = Workspace::new();
    let mut args = TrainArgs::new(MarkerPid::Own);
    args.cooperate = false;
    let result = run(plan(1, args, 2), ws.ctx("job-e"))
        .await
        .expect("succeeds");
    let out: Number = result.final_output.unwrap().into_typed().unwrap();
    assert_eq!(out.n, 101);
    let seen = seen();
    assert_eq!(seen.len(), 2);
    assert!(
        seen.iter().all(|s| !slot_of(s).resume),
        "no marker, nothing to resume"
    );
    assert!(ws.slots().is_empty(), "the unused slot is removed too");
}

#[tokio::test]
async fn a_changed_config_or_input_never_resumes_an_old_checkpoint() {
    let _g = SERIAL.lock().await;
    reset(1);
    let ws = Workspace::new();
    // Leave a resumable checkpoint behind for (seed 1, tag 1).
    run(plan(1, TrainArgs::new(MarkerPid::Dead), 1), ws.ctx("job-f"))
        .await
        .unwrap_err();
    let original = slot_of(&seen()[0]).dir.clone();

    let mut other_config = TrainArgs::new(MarkerPid::Dead);
    other_config.tag = 2;
    run(plan(1, other_config, 1), ws.ctx("job-f"))
        .await
        .unwrap();
    run(plan(2, TrainArgs::new(MarkerPid::Dead), 1), ws.ctx("job-f"))
        .await
        .unwrap();
    let seen_now = seen();
    for (i, why) in [(1, "changed config"), (2, "changed input")] {
        let s = slot_of(&seen_now[i]);
        assert!(!s.resume, "{why} must start fresh");
        assert_ne!(s.dir, original, "{why} must use another slot");
    }
    assert!(
        has_file(&original, "step-0000000005.pt"),
        "the old slot is untouched"
    );

    // The original key still resumes.
    run(plan(1, TrainArgs::new(MarkerPid::Dead), 1), ws.ctx("job-f"))
        .await
        .unwrap();
    let last = seen();
    let s = slot_of(last.last().unwrap());
    assert!(s.resume && s.dir == original);
}

#[tokio::test]
async fn a_stage_that_does_not_opt_in_gets_no_slot() {
    let _g = SERIAL.lock().await;
    reset(0);
    let ws = Workspace::new();
    let plan = Plan::<(), TestBackend>::new("no-slot", serde_json::json!({}))
        .start(Seed, SeedArgs { n: 1 })
        .then(NoSlot, SeedArgs { n: 0 })
        .finish()
        .into_compiled();
    run(plan, ws.ctx("job-g")).await.unwrap();
    assert_eq!(seen(), vec![None]);
    assert!(!ws.root.exists(), "no slot directory is created");
    assert!(ws.events("job-g").is_empty());
}
