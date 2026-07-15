// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0102 executable condition-gate acceptance tests.
//!
//! These tests stay on the public cookbook/PlanSpec/executor seams. A condition
//! gate is control metadata: it may delay or prune a data-ready node, but must
//! not become a typed predecessor or change that node's cache identity.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use blut::framework::artifact::{Artifact, BranchDecision, ContentHash};
use blut::framework::async_io::{IoMode, TrainingIoCandidate, TrainingIoHints};
use blut::framework::cache::{CacheHandle, CacheProof};
use blut::framework::cookbook::{Cookbook, Registry};
use blut::framework::dag_opt::DagOptimizer;
use blut::framework::error::{PlanError, StageError};
use blut::framework::executor::{ExecCtx, execute_plan};
use blut::framework::object_store::BlobStore;
use blut::framework::plan::CompiledPlan;
use blut::framework::plan_spec::{ConditionGateSpec, PLAN_SPEC_VERSION, PlanSpec, SpecNode};
use blut::framework::resource::Resource;
use blut::framework::stage::{ErasedStageCtor, Stage, StageContext};
use blut::framework::status::{DEFAULT_BROADCAST_CAPACITY, HostedEvent, StageEvent};
use blut::recipes::recipe::RecipeDef;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const DECISION_STAGE: &str = "conditional_test_decision";
const SOURCE_STAGE: &str = "conditional_test_source";
const GUARDED_STAGE: &str = "conditional_test_guarded";
const HELD_DECISION_STAGE: &str = "conditional_test_held_decision";
const PROBE_SOURCE_STAGE: &str = "conditional_test_probe_source";
const ALT_PROBE_SOURCE_STAGE: &str = "conditional_test_alt_probe_source";
const PROBE_GUARDED_STAGE: &str = "conditional_test_probe_guarded";
const HELD_IDENTITY_STAGE: &str = "conditional_test_held_identity";
const HELD_FAILURE_STAGE: &str = "conditional_test_held_failure";
const PANIC_PUBLISH_STAGE: &str = "conditional_test_panic_publish";

static SPECULATION_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static PROBE_SIGNALS: Mutex<Option<Arc<ProbeSignals>>> = Mutex::new(None);

struct ProbeSignals {
    decision_started: tokio::sync::Semaphore,
    decision_release: tokio::sync::Semaphore,
    source_finished: tokio::sync::Semaphore,
    identity_started: tokio::sync::Semaphore,
    identity_release: tokio::sync::Semaphore,
    target_started: tokio::sync::Semaphore,
    target_release: tokio::sync::Semaphore,
    target_finished: tokio::sync::Semaphore,
    target_runs: AtomicUsize,
    hold_target: AtomicBool,
    panic_first_target: AtomicBool,
    seal_target_scratch: AtomicBool,
    profile_target: AtomicBool,
    target_steps: AtomicUsize,
}

impl ProbeSignals {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            decision_started: tokio::sync::Semaphore::new(0),
            decision_release: tokio::sync::Semaphore::new(0),
            source_finished: tokio::sync::Semaphore::new(0),
            identity_started: tokio::sync::Semaphore::new(0),
            identity_release: tokio::sync::Semaphore::new(0),
            target_started: tokio::sync::Semaphore::new(0),
            target_release: tokio::sync::Semaphore::new(0),
            target_finished: tokio::sync::Semaphore::new(0),
            target_runs: AtomicUsize::new(0),
            hold_target: AtomicBool::new(false),
            panic_first_target: AtomicBool::new(false),
            seal_target_scratch: AtomicBool::new(false),
            profile_target: AtomicBool::new(false),
            target_steps: AtomicUsize::new(1),
        })
    }
}

fn install_probe_signals(signals: Arc<ProbeSignals>) {
    *PROBE_SIGNALS.lock().expect("probe signals mutex") = Some(signals);
}

fn probe_signals() -> Arc<ProbeSignals> {
    PROBE_SIGNALS
        .lock()
        .expect("probe signals mutex")
        .as_ref()
        .expect("probe signals installed")
        .clone()
}

async fn take_signal(signal: &tokio::sync::Semaphore) {
    signal.acquire().await.expect("probe signal open").forget();
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct TestValue {
    value: u32,
}

impl Artifact for TestValue {
    const KIND: &'static str = "conditional-test.value";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&self.value.to_le_bytes())
    }

    fn primary_path(&self) -> &Path {
        Path::new("")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PanicPublishValue {
    value: u32,
    path: std::path::PathBuf,
}

impl Artifact for PanicPublishValue {
    const KIND: &'static str = "conditional-test.panic-publish";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        if !self.path.to_string_lossy().contains(".speculation") {
            #[cfg(unix)]
            if probe_signals().seal_target_scratch.load(Ordering::SeqCst) {
                use std::os::unix::fs::PermissionsExt;
                let parent = self.path.parent().expect("panic fixture has stage parent");
                std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o555))
                    .expect("seal renamed canonical stage before publication panic");
            }
            panic!("canonical publication content-hash panic fixture");
        }
        ContentHash::of_bytes(&self.value.to_le_bytes())
    }

    fn primary_path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct DecisionArgs {
    value: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct SourceArgs {
    value: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct GuardArgs {
    add: u32,
}

struct DecisionStage;

#[async_trait]
impl Stage for DecisionStage {
    const NAME: &'static str = DECISION_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = BranchDecision;
    type Args = DecisionArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &DecisionArgs,
    ) -> Result<BranchDecision, StageError> {
        Ok(BranchDecision { value: args.value })
    }
}

struct SourceStage;

#[async_trait]
impl Stage for SourceStage {
    const NAME: &'static str = SOURCE_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = TestValue;
    type Args = SourceArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &SourceArgs,
    ) -> Result<TestValue, StageError> {
        Ok(TestValue { value: args.value })
    }
}

struct GuardedStage;

#[async_trait]
impl Stage for GuardedStage {
    const NAME: &'static str = GUARDED_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const SPECULATION_SAFE: bool = true;
    type Input = TestValue;
    type Output = TestValue;
    type Args = GuardArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: TestValue,
        args: &GuardArgs,
    ) -> Result<TestValue, StageError> {
        let marker = ctx.stage_dir.join("guard-ran");
        std::fs::write(&marker, b"ran").map_err(|source| StageError::Io {
            path: marker,
            source,
        })?;
        Ok(TestValue {
            value: input.value + args.add,
        })
    }
}

struct HeldDecisionStage;

#[async_trait]
impl Stage for HeldDecisionStage {
    const NAME: &'static str = HELD_DECISION_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[];
    type Input = ();
    type Output = BranchDecision;
    type Args = DecisionArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &DecisionArgs,
    ) -> Result<BranchDecision, StageError> {
        let signals = probe_signals();
        signals.decision_started.add_permits(1);
        tokio::select! {
            permit = signals.decision_release.acquire() => {
                permit.expect("decision release signal open").forget();
            }
            _ = ctx.cancel.cancelled() => return Err(StageError::Cancelled),
        }
        Ok(BranchDecision { value: args.value })
    }
}

struct ProbeSourceStage;

#[async_trait]
impl Stage for ProbeSourceStage {
    const NAME: &'static str = PROBE_SOURCE_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[];
    type Input = ();
    type Output = TestValue;
    type Args = SourceArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &SourceArgs,
    ) -> Result<TestValue, StageError> {
        probe_signals().source_finished.add_permits(1);
        Ok(TestValue { value: args.value })
    }
}

struct AltProbeSourceStage;

#[async_trait]
impl Stage for AltProbeSourceStage {
    const NAME: &'static str = ALT_PROBE_SOURCE_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[];
    type Input = ();
    type Output = TestValue;
    type Args = SourceArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &SourceArgs,
    ) -> Result<TestValue, StageError> {
        probe_signals().source_finished.add_permits(1);
        Ok(TestValue { value: args.value })
    }
}

struct ProbeGuardedStage;

#[async_trait]
impl Stage for ProbeGuardedStage {
    const NAME: &'static str = PROBE_GUARDED_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Network];
    const SPECULATION_SAFE: bool = true;
    type Input = TestValue;
    type Output = TestValue;
    type Args = GuardArgs;

    fn training_io_sync_base_bytes(&self, _args: &Self::Args, _hints: TrainingIoHints) -> u64 {
        1
    }

    fn training_io_candidates(
        &self,
        _args: &Self::Args,
        _hints: TrainingIoHints,
    ) -> Vec<TrainingIoCandidate> {
        if !probe_signals().profile_target.load(Ordering::SeqCst) {
            return Vec::new();
        }
        vec![
            TrainingIoCandidate {
                data_replicas: 1,
                decode_workers: 1,
                prefetch_per_worker: 1,
                cuda_staging_slots: 1,
                pipeline: IoMode::Inline,
                metrics: IoMode::Bounded {
                    capacity: 1,
                    max_item_bytes: 1,
                },
                checkpoints: IoMode::Inline,
                batch_bytes: Some(1),
                checkpoint_snapshot_bytes: Some(0),
                fixed_overhead_bytes: Some(1),
            },
            TrainingIoCandidate {
                data_replicas: 1,
                decode_workers: 0,
                prefetch_per_worker: 0,
                cuda_staging_slots: 0,
                pipeline: IoMode::Inline,
                metrics: IoMode::Inline,
                checkpoints: IoMode::Inline,
                batch_bytes: Some(0),
                checkpoint_snapshot_bytes: Some(0),
                fixed_overhead_bytes: Some(0),
            },
        ]
    }

    async fn run(
        &self,
        ctx: &StageContext,
        input: TestValue,
        args: &GuardArgs,
    ) -> Result<TestValue, StageError> {
        let signals = probe_signals();
        let run_number = signals.target_runs.fetch_add(1, Ordering::SeqCst) + 1;
        signals.target_started.add_permits(1);
        for step in 0..signals.target_steps.load(Ordering::SeqCst) {
            let _ = ctx.status_tx.send(StageEvent::StageStep {
                node_idx: ctx.node_idx,
                stage_name: Self::NAME.to_string(),
                update: serde_json::json!({ "speculation_probe": true, "step": step }),
            });
        }
        let marker = ctx.stage_dir.join("guard-ran");
        std::fs::write(&marker, b"ran").map_err(|source| StageError::Io {
            path: marker,
            source,
        })?;
        #[cfg(unix)]
        if signals.seal_target_scratch.load(Ordering::SeqCst) {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&ctx.stage_dir, std::fs::Permissions::from_mode(0o555))
                .map_err(|source| StageError::Io {
                    path: ctx.stage_dir.clone(),
                    source,
                })?;
        }
        if signals.hold_target.load(Ordering::SeqCst) && run_number == 1 {
            tokio::select! {
                permit = signals.target_release.acquire() => {
                    permit.expect("target release signal open").forget();
                }
                _ = ctx.cancel.cancelled() => return Err(StageError::Cancelled),
            }
        }
        signals.target_finished.add_permits(1);
        if signals.panic_first_target.swap(false, Ordering::SeqCst) {
            panic!("private speculation panic fixture");
        }
        Ok(TestValue {
            value: input.value + args.add,
        })
    }
}

struct HeldIdentityStage;

#[async_trait]
impl Stage for HeldIdentityStage {
    const NAME: &'static str = HELD_IDENTITY_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[];
    type Input = ();
    type Output = TestValue;
    type Args = SourceArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &SourceArgs,
    ) -> Result<TestValue, StageError> {
        let signals = probe_signals();
        signals.identity_started.add_permits(1);
        tokio::select! {
            permit = signals.identity_release.acquire() => {
                permit.expect("identity release signal open").forget();
            }
            _ = ctx.cancel.cancelled() => return Err(StageError::Cancelled),
        }
        Ok(TestValue { value: args.value })
    }
}

struct PanicPublishStage;

struct HeldFailureStage;

#[async_trait]
impl Stage for HeldFailureStage {
    const NAME: &'static str = HELD_FAILURE_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[];
    type Input = ();
    type Output = TestValue;
    type Args = ();

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        _args: &(),
    ) -> Result<TestValue, StageError> {
        let signals = probe_signals();
        signals.identity_started.add_permits(1);
        tokio::select! {
            permit = signals.identity_release.acquire() => {
                permit.expect("failure release signal open").forget();
            }
            _ = ctx.cancel.cancelled() => return Err(StageError::Cancelled),
        }
        Err(StageError::BadInput(
            "held unrelated failure fixture".into(),
        ))
    }
}

#[async_trait]
impl Stage for PanicPublishStage {
    const NAME: &'static str = PANIC_PUBLISH_STAGE;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Network];
    const SPECULATION_SAFE: bool = true;
    type Input = TestValue;
    type Output = PanicPublishValue;
    type Args = GuardArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: TestValue,
        args: &GuardArgs,
    ) -> Result<PanicPublishValue, StageError> {
        let signals = probe_signals();
        signals.target_runs.fetch_add(1, Ordering::SeqCst);
        signals.target_started.add_permits(1);
        let path = ctx.stage_dir.join("panic-publish.bin");
        std::fs::write(&path, b"private").map_err(|source| StageError::Io {
            path: path.clone(),
            source,
        })?;
        signals.target_finished.add_permits(1);
        Ok(PanicPublishValue {
            value: input.value + args.add,
            path,
        })
    }
}

#[derive(Debug)]
struct PanicOnTargetPutStore {
    target: ContentHash,
}

impl BlobStore for PanicOnTargetPutStore {
    fn get(&self, _key: ContentHash) -> std::io::Result<Option<Vec<u8>>> {
        Ok(None)
    }

    fn put(&self, key: ContentHash, _bytes: &[u8]) -> std::io::Result<()> {
        if key == self.target {
            panic!("remote target put panic fixture");
        }
        Ok(())
    }

    fn head(&self, _key: ContentHash) -> std::io::Result<bool> {
        Ok(false)
    }
}

struct ConditionalTestCookbook;

impl Cookbook for ConditionalTestCookbook {
    fn name(&self) -> &'static str {
        "conditional-execution-test"
    }

    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }

    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static STAGES: &[(&str, ErasedStageCtor)] = &[
            (DECISION_STAGE, || Arc::new(DecisionStage)),
            (SOURCE_STAGE, || Arc::new(SourceStage)),
            (GUARDED_STAGE, || Arc::new(GuardedStage)),
            (HELD_DECISION_STAGE, || Arc::new(HeldDecisionStage)),
            (PROBE_SOURCE_STAGE, || Arc::new(ProbeSourceStage)),
            (ALT_PROBE_SOURCE_STAGE, || Arc::new(AltProbeSourceStage)),
            (PROBE_GUARDED_STAGE, || Arc::new(ProbeGuardedStage)),
            (HELD_IDENTITY_STAGE, || Arc::new(HeldIdentityStage)),
            (HELD_FAILURE_STAGE, || Arc::new(HeldFailureStage)),
            (PANIC_PUBLISH_STAGE, || Arc::new(PanicPublishStage)),
        ];
        STAGES
    }
}

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.register(Box::new(ConditionalTestCookbook));
    registry
}

fn node(stage: &str, args: serde_json::Value) -> SpecNode {
    SpecNode {
        stage: stage.into(),
        args,
        retry: None,
        timeout: None,
        priority: None,
        pure: false,
    }
}

fn conditional_spec(decision: bool) -> PlanSpec {
    PlanSpec {
        name: format!("conditional-{decision}"),
        nodes: vec![
            node(DECISION_STAGE, serde_json::json!({ "value": decision })),
            node(SOURCE_STAGE, serde_json::json!({ "value": 41 })),
            node(GUARDED_STAGE, serde_json::json!({ "add": 1 })),
        ],
        edges: vec![(1, 2)],
        expansions: Vec::new(),
        condition_gates: vec![ConditionGateSpec {
            condition: 0,
            target: 2,
            when: true,
        }],
        version: PLAN_SPEC_VERSION,
    }
}

fn speculation_spec(decision: bool) -> PlanSpec {
    let mut guarded = node(PROBE_GUARDED_STAGE, serde_json::json!({ "add": 1 }));
    guarded.pure = true;
    PlanSpec {
        name: format!("speculation-{decision}"),
        nodes: vec![
            node(
                HELD_DECISION_STAGE,
                serde_json::json!({ "value": decision }),
            ),
            node(PROBE_SOURCE_STAGE, serde_json::json!({ "value": 41 })),
            guarded,
        ],
        edges: vec![(1, 2)],
        expansions: Vec::new(),
        condition_gates: vec![ConditionGateSpec {
            condition: 0,
            target: 2,
            when: true,
        }],
        version: PLAN_SPEC_VERSION,
    }
}

fn speculation_ungated_spec() -> PlanSpec {
    PlanSpec {
        name: "speculation-ungated".into(),
        nodes: vec![
            node(PROBE_SOURCE_STAGE, serde_json::json!({ "value": 41 })),
            node(PROBE_GUARDED_STAGE, serde_json::json!({ "add": 1 })),
        ],
        edges: vec![(0, 1)],
        expansions: Vec::new(),
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
}

fn speculation_same_key_supersede_plan(registry: &Registry) -> CompiledPlan {
    let conditional = speculation_spec(true)
        .compile(registry)
        .expect("compile conditional same-key component");
    let ordinary = PlanSpec {
        name: "ordinary-same-key-component".into(),
        nodes: vec![
            node(HELD_IDENTITY_STAGE, serde_json::json!({ "value": 41 })),
            node(PROBE_GUARDED_STAGE, serde_json::json!({ "add": 1 })),
        ],
        edges: vec![(0, 1)],
        expansions: Vec::new(),
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
    .compile(registry)
    .expect("compile ordinary same-key component");
    CompiledPlan::from_components(
        "speculation-same-key-supersede".into(),
        serde_json::Value::Null,
        vec![conditional, ordinary],
    )
    .0
}

fn speculation_with_unrelated_failure_plan(registry: &Registry) -> CompiledPlan {
    let conditional = speculation_spec(true)
        .compile(registry)
        .expect("compile conditional cleanup component");
    let failure = PlanSpec {
        name: "unrelated-failure-component".into(),
        nodes: vec![
            node(HELD_FAILURE_STAGE, serde_json::Value::Null),
            node(GUARDED_STAGE, serde_json::json!({ "add": 1 })),
        ],
        edges: vec![(0, 1)],
        expansions: Vec::new(),
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
    .compile(registry)
    .expect("compile unrelated failure component");
    CompiledPlan::from_components(
        "speculation-unrelated-failure".into(),
        serde_json::Value::Null,
        vec![conditional, failure],
    )
    .0
}

fn speculation_publish_panic_spec() -> PlanSpec {
    let mut guarded = node(PANIC_PUBLISH_STAGE, serde_json::json!({ "add": 1 }));
    guarded.pure = true;
    PlanSpec {
        name: "speculation-publish-panic".into(),
        nodes: vec![
            node(HELD_DECISION_STAGE, serde_json::json!({ "value": true })),
            node(PROBE_SOURCE_STAGE, serde_json::json!({ "value": 41 })),
            guarded,
        ],
        edges: vec![(1, 2)],
        expansions: Vec::new(),
        condition_gates: vec![ConditionGateSpec {
            condition: 0,
            target: 2,
            when: true,
        }],
        version: PLAN_SPEC_VERSION,
    }
}

fn speculation_ctx(job_dir: std::path::PathBuf, enabled: bool) -> ExecCtx {
    let mut optimizer = DagOptimizer::new();
    optimizer.speculative_execution = enabled;
    let mut ctx = ExecCtx::new(job_dir)
        .with_max_in_flight(2)
        .with_resource_limit(Resource::Network, 1);
    ctx.dag_optimizer = Some(optimizer);
    ctx
}

fn ungated_spec() -> PlanSpec {
    PlanSpec {
        name: "ungated-equivalent".into(),
        nodes: vec![
            node(SOURCE_STAGE, serde_json::json!({ "value": 41 })),
            node(GUARDED_STAGE, serde_json::json!({ "add": 1 })),
        ],
        edges: vec![(0, 1)],
        expansions: Vec::new(),
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
}

fn conditional_root_spec(decision: bool) -> PlanSpec {
    PlanSpec {
        name: format!("conditional-root-{decision}"),
        nodes: vec![
            node(DECISION_STAGE, serde_json::json!({ "value": decision })),
            node(SOURCE_STAGE, serde_json::json!({ "value": 41 })),
        ],
        edges: Vec::new(),
        expansions: Vec::new(),
        condition_gates: vec![ConditionGateSpec {
            condition: 0,
            target: 1,
            when: true,
        }],
        version: PLAN_SPEC_VERSION,
    }
}

fn status_events(job_dir: &Path) -> Vec<StageEvent> {
    let body = std::fs::read_to_string(job_dir.join("status.jsonl"))
        .expect("conditional execution writes status.jsonl");
    body.lines()
        .map(|line| {
            serde_json::from_str::<HostedEvent>(line)
                .expect("status line is a HostedEvent")
                .event
        })
        .collect()
}

fn materializes_stage(event: &StageEvent, expected: &str) -> bool {
    match event {
        StageEvent::StageBegin { stage_name, .. }
        | StageEvent::StageEnd { stage_name, .. }
        | StageEvent::StageSkipped { stage_name, .. }
        | StageEvent::StageFailed { stage_name, .. }
        | StageEvent::StageBlocked { stage_name, .. }
        | StageEvent::StageStep { stage_name, .. }
        | StageEvent::StageRetrying { stage_name, .. } => stage_name == expected,
        StageEvent::StageIoConfigured { .. }
        | StageEvent::StagePruned { .. }
        | StageEvent::StepGap { .. } => false,
        _ => false,
    }
}

fn is_pruned_stage(event: &StageEvent, expected: &str) -> bool {
    matches!(
        event,
        StageEvent::StagePruned { stage_name, .. } if stage_name == expected
    )
}

fn cache_entry_count(job_dir: &Path) -> usize {
    std::fs::read_dir(job_dir.join("_cache"))
        .expect("job-local cache exists")
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .count()
}

fn event_count(events: &[StageEvent], stage: &str, kind: &str) -> usize {
    events
        .iter()
        .filter(|event| match (*event, kind) {
            (StageEvent::StageBegin { stage_name, .. }, "begin") => stage_name == stage,
            (StageEvent::StageEnd { stage_name, .. }, "end") => stage_name == stage,
            _ => false,
        })
        .count()
}

fn guarded_residue(root: &Path) -> Vec<std::path::PathBuf> {
    fn visit(path: &Path, found: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.contains(PROBE_GUARDED_STAGE) || name == "guard-ran" {
                found.push(path.clone());
            }
            if path.is_dir() {
                visit(&path, found);
            }
        }
    }
    let mut found = Vec::new();
    visit(root, &mut found);
    found
}

#[test]
fn optimizer_owns_cache_neutral_speculation_witness() {
    let mut spec = conditional_spec(true);
    spec.nodes[2].pure = true;
    let plan = spec
        .compile(&registry())
        .expect("compile pure conditional target");
    let fingerprint = plan.execution_fingerprint();

    let optimizer = DagOptimizer {
        speculative_execution: true,
        ..DagOptimizer::new()
    };
    let (enabled, _) = optimizer.optimize(plan);
    assert_eq!(
        enabled.speculation_candidates().collect::<Vec<_>>(),
        vec![2]
    );
    assert_eq!(enabled.execution_fingerprint(), fingerprint);

    let (disabled_again, _) = DagOptimizer::new().optimize(enabled);
    assert!(disabled_again.speculation_candidates().next().is_none());
    assert_eq!(disabled_again.execution_fingerprint(), fingerprint);
}

#[tokio::test(flavor = "current_thread")]
async fn speculation_default_off_waits_for_condition_before_starting_target() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("default-off speculation tempdir");
    let job_dir = temp.path().join("job");
    let plan = speculation_spec(true)
        .compile(&registry())
        .expect("compile default-off speculation plan");
    let handle = tokio::spawn(execute_plan(plan, speculation_ctx(job_dir.clone(), false)));

    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            take_signal(&signals.target_started)
        )
        .await
        .is_err(),
        "default-off optimizer must not run a pure/safe target early"
    );
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 0);

    signals.decision_release.add_permits(1);
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    let result = handle
        .await
        .expect("default-off executor task")
        .expect("default-off selected plan");
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 1);
    assert_eq!(result.n_cache_misses, 3);
    assert!(
        job_dir
            .join(format!("stages/2-{PROBE_GUARDED_STAGE}/guard-ran"))
            .is_file()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn selected_speculation_runs_early_and_publishes_once_with_ordinary_identity() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let registry = registry();

    let baseline_signals = ProbeSignals::new();
    install_probe_signals(baseline_signals);
    let baseline_temp = tempfile::tempdir().expect("ordinary identity tempdir");
    let baseline_job = baseline_temp.path().join("job");
    let baseline = execute_plan(
        speculation_ungated_spec()
            .compile(&registry)
            .expect("compile ordinary identity plan"),
        ExecCtx::new(baseline_job.clone()),
    )
    .await
    .expect("ordinary identity run");
    let baseline_output = baseline
        .final_output
        .expect("ordinary guarded output")
        .into_typed::<TestValue>()
        .expect("decode ordinary guarded output");
    let baseline_proof = CacheProof::read_from(
        &baseline_job.join(format!("stages/1-{PROBE_GUARDED_STAGE}/cache-proof.json")),
    )
    .expect("ordinary guarded proof");

    let signals = ProbeSignals::new();
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("selected speculation tempdir");
    let job_dir = temp.path().join("job");
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry)
            .expect("compile selected speculation plan"),
        speculation_ctx(job_dir.clone(), true),
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        take_signal(&signals.target_started),
    )
    .await
    .expect("speculative target starts before selector release");
    take_signal(&signals.target_finished).await;
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 1);
    signals.decision_release.add_permits(1);

    let result = handle
        .await
        .expect("selected speculation executor task")
        .expect("selected speculation run");
    let output = result
        .final_output
        .expect("selected speculative output")
        .into_typed::<TestValue>()
        .expect("decode selected speculative output");
    assert_eq!(output, baseline_output);
    assert_eq!(output, TestValue { value: 42 });
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 1);
    assert_eq!(result.n_cache_misses, 3);
    assert_eq!(cache_entry_count(&job_dir), 3);
    let proof = CacheProof::read_from(
        &job_dir.join(format!("stages/2-{PROBE_GUARDED_STAGE}/cache-proof.json")),
    )
    .expect("selected speculative proof");
    assert_eq!(proof.key, baseline_proof.key);
    assert!(proof.entry_path.is_file());
    let events = status_events(&job_dir);
    assert_eq!(event_count(&events, PROBE_GUARDED_STAGE, "begin"), 1);
    assert_eq!(event_count(&events, PROBE_GUARDED_STAGE, "end"), 1);
    assert!(!job_dir.join(".speculation").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn unselected_speculation_discards_lifecycle_cache_and_scratch() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let registry = registry();
    let signals = ProbeSignals::new();
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("discarded speculation tempdir");
    let job_dir = temp.path().join("job");
    let handle = tokio::spawn(execute_plan(
        speculation_spec(false)
            .compile(&registry)
            .expect("compile discarded speculation plan"),
        speculation_ctx(job_dir.clone(), true),
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        take_signal(&signals.target_started),
    )
    .await
    .expect("discarded target actually ran speculatively");
    take_signal(&signals.target_finished).await;
    signals.decision_release.add_permits(1);

    let result = handle
        .await
        .expect("discard speculation executor task")
        .expect("false speculative branch remains successful");
    assert!(result.final_output.is_none());
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 1);
    assert_eq!(result.n_cache_misses, 2);
    assert_eq!(cache_entry_count(&job_dir), 2);
    assert!(guarded_residue(&job_dir).is_empty());
    assert!(!job_dir.join(".speculation").exists());
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, PROBE_GUARDED_STAGE))
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| is_pruned_stage(event, PROBE_GUARDED_STAGE))
            .count(),
        1
    );

    let ungated = execute_plan(
        speculation_ungated_spec()
            .compile(&registry)
            .expect("compile ungated cache probe"),
        ExecCtx::new(job_dir.clone()),
    )
    .await
    .expect("ungated run after discarded speculation");
    assert_eq!((ungated.n_cache_hits, ungated.n_cache_misses), (1, 1));
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 2);
}

#[cfg(unix)]
fn make_tree_owner_writable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return;
    };
    if metadata.is_dir() {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .expect("restore scratch directory permissions");
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                make_tree_owner_writable(&entry.path());
            }
        }
    }
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn scratch_cleanup_failure_is_fatal_instead_of_false_success() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.seal_target_scratch.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("cleanup failure tempdir");
    let job_dir = temp.path().join("job");
    let handle = tokio::spawn(execute_plan(
        speculation_spec(false)
            .compile(&registry())
            .expect("compile cleanup failure plan"),
        speculation_ctx(job_dir.clone(), true),
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    signals.decision_release.add_permits(1);

    let error = handle
        .await
        .expect("cleanup failure executor task")
        .expect_err("observable private residue must fail the plan closed");
    assert!(
        error
            .to_string()
            .contains("failed to remove private speculation scratch"),
        "cleanup failure must retain its exact classification: {error}"
    );
    let scratch = job_dir.join(".speculation");
    assert!(scratch.exists(), "fixture must prove cleanup really failed");

    make_tree_owner_writable(&scratch);
    std::fs::remove_dir_all(&scratch).expect("remove restored cleanup-failure fixture");
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn private_panic_with_cleanup_failure_is_fatal_instead_of_hidden() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.seal_target_scratch.store(true, Ordering::SeqCst);
    signals.panic_first_target.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("panic cleanup failure tempdir");
    let job_dir = temp.path().join("job");
    let handle = tokio::spawn(execute_plan(
        speculation_spec(false)
            .compile(&registry())
            .expect("compile panic cleanup failure plan"),
        speculation_ctx(job_dir.clone(), true),
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    signals.decision_release.add_permits(1);

    let error = handle
        .await
        .expect("panic cleanup failure executor task")
        .expect_err("a panic must not hide undeletable private residue");
    assert!(
        error
            .to_string()
            .contains("failed to remove private speculation scratch"),
        "checked cleanup must dominate the disposable private panic: {error}"
    );
    let scratch = job_dir.join(".speculation");
    assert!(scratch.exists(), "fixture must prove cleanup really failed");

    make_tree_owner_writable(&scratch);
    std::fs::remove_dir_all(&scratch).expect("remove restored panic-cleanup fixture");
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn unrelated_failure_explicitly_drains_prepared_private_scratch() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.seal_target_scratch.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let registry = registry();
    let temp = tempfile::tempdir().expect("unrelated failure cleanup tempdir");
    let job_dir = temp.path().join("job");
    let ctx = speculation_ctx(job_dir.clone(), true).with_max_in_flight(3);
    let handle = tokio::spawn(execute_plan(
        speculation_with_unrelated_failure_plan(&registry),
        ctx,
    ));

    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.identity_started).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    // No other task can complete while the selector and failure fixture are
    // held, so yielding lets the coordinator retain the prepared result.
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;
    signals.identity_release.add_permits(1);

    let error = handle
        .await
        .expect("unrelated failure executor task")
        .expect_err("terminal failure must explicitly drain prepared speculation");
    let message = error.to_string();
    assert!(
        message.contains("held unrelated failure fixture"),
        "{message}"
    );
    assert!(
        message.contains("failed to remove private speculation scratch"),
        "cleanup integrity failure must be aggregated with the first error: {message}"
    );
    let scratch = job_dir.join(".speculation");
    assert!(scratch.exists(), "fixture must prove cleanup really failed");

    make_tree_owner_writable(&scratch);
    std::fs::remove_dir_all(&scratch).expect("remove restored unrelated-failure fixture");
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn unrelated_failure_aggregates_running_private_cleanup_failure() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.seal_target_scratch.store(true, Ordering::SeqCst);
    signals.hold_target.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let registry = registry();
    let temp = tempfile::tempdir().expect("running failure cleanup tempdir");
    let job_dir = temp.path().join("job");
    let ctx = speculation_ctx(job_dir.clone(), true).with_max_in_flight(3);
    let handle = tokio::spawn(execute_plan(
        speculation_with_unrelated_failure_plan(&registry),
        ctx,
    ));

    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.identity_started).await;
    take_signal(&signals.target_started).await;
    // The target signals before its synchronous write/seal sequence, then
    // yields only when it reaches the held cancellation point.
    tokio::task::yield_now().await;
    signals.identity_release.add_permits(1);

    let error = handle
        .await
        .expect("running failure executor task")
        .expect_err("late private cleanup failure must join the primary error");
    let message = error.to_string();
    assert!(
        message.contains("held unrelated failure fixture"),
        "{message}"
    );
    assert!(
        message.contains("failed to remove private speculation scratch"),
        "running cleanup integrity failure must not lose to first-error policy: {message}"
    );
    let scratch = job_dir.join(".speculation");
    assert!(scratch.exists(), "fixture must prove cleanup really failed");

    make_tree_owner_writable(&scratch);
    std::fs::remove_dir_all(&scratch).expect("remove restored running-failure fixture");
}

#[tokio::test(flavor = "current_thread")]
async fn speculation_declines_silently_when_admission_is_not_spare() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("speculation admission tempdir");
    let job_dir = temp.path().join("job");
    let ctx = speculation_ctx(job_dir, true);
    let network = ctx.resources[&Resource::Network].clone();
    let held = network
        .acquire_owned()
        .await
        .expect("hold speculative Network envelope");
    let mut status_rx = ctx.status.subscribe();
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry())
            .expect("compile admission-decline plan"),
        ctx,
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            take_signal(&signals.target_started)
        )
        .await
        .is_err()
    );
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 0);
    while let Ok(event) = status_rx.try_recv() {
        assert!(
            !materializes_stage(&event, PROBE_GUARDED_STAGE),
            "try-only speculative admission must emit no guarded event: {event:?}"
        );
    }

    signals.decision_release.add_permits(1);
    drop(held);
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    let result = handle
        .await
        .expect("admission-decline executor task")
        .expect("selected fallback runs ordinarily");
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 1);
    assert_eq!(result.n_cache_misses, 3);
}

#[tokio::test(flavor = "current_thread")]
async fn selected_while_speculation_is_running_does_not_spawn_a_duplicate() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.hold_target.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("running selection tempdir");
    let job_dir = temp.path().join("job");
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry())
            .expect("compile running selection plan"),
        speculation_ctx(job_dir.clone(), true),
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    signals.decision_release.add_permits(1);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(
        signals.target_runs.load(Ordering::SeqCst),
        1,
        "selection must retain the running private attempt instead of scheduling ordinary work"
    );
    signals.target_release.add_permits(2);
    take_signal(&signals.target_finished).await;
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
        .await
        .expect("running selection completes")
        .expect("running selection executor task")
        .expect("running selected speculation succeeds");
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 1);
    assert_eq!(result.n_cache_misses, 3);
    let events = status_events(&job_dir);
    assert_eq!(event_count(&events, PROBE_GUARDED_STAGE, "begin"), 1);
    assert_eq!(event_count(&events, PROBE_GUARDED_STAGE, "end"), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn rejecting_running_speculation_cancels_and_releases_its_envelope() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.hold_target.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("running rejection tempdir");
    let job_dir = temp.path().join("job");
    let ctx = speculation_ctx(job_dir.clone(), true);
    let network = ctx.resources[&Resource::Network].clone();
    let handle = tokio::spawn(execute_plan(
        speculation_spec(false)
            .compile(&registry())
            .expect("compile running rejection plan"),
        ctx,
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    signals.decision_release.add_permits(1);
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
        .await
        .expect("running rejection drains")
        .expect("running rejection executor task")
        .expect("running rejection is a successful prune");
    assert!(result.final_output.is_none());
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 1);
    assert_eq!(network.available_permits(), 1);
    assert!(!job_dir.join(".speculation").exists());
    assert!(guarded_residue(&job_dir).is_empty());
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, PROBE_GUARDED_STAGE))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn warm_canonical_target_suppresses_speculative_recomputation() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let registry = registry();
    let temp = tempfile::tempdir().expect("warm speculation tempdir");
    let job_dir = temp.path().join("job");
    install_probe_signals(ProbeSignals::new());
    execute_plan(
        speculation_ungated_spec()
            .compile(&registry)
            .expect("compile warm seed plan"),
        ExecCtx::new(job_dir.clone()),
    )
    .await
    .expect("seed warm target cache");

    let signals = ProbeSignals::new();
    install_probe_signals(signals.clone());
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry)
            .expect("compile warm conditional plan"),
        speculation_ctx(job_dir, true),
    ));
    take_signal(&signals.decision_started).await;
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            take_signal(&signals.target_started)
        )
        .await
        .is_err(),
        "presence probe must suppress optional recomputation"
    );
    signals.decision_release.add_permits(1);
    let result = handle
        .await
        .expect("warm conditional executor task")
        .expect("warm conditional run");
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 0);
    assert_eq!((result.n_cache_hits, result.n_cache_misses), (2, 1));
}

#[tokio::test(flavor = "current_thread")]
async fn private_panic_is_hidden_and_selected_branch_falls_back_ordinarily() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.panic_first_target.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("private panic tempdir");
    let job_dir = temp.path().join("job");
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry())
            .expect("compile private panic plan"),
        speculation_ctx(job_dir.clone(), true),
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    signals.decision_release.add_permits(1);
    let result = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
        .await
        .expect("panic fallback completes")
        .expect("panic fallback executor task")
        .expect("selected branch retries ordinarily after private panic");
    let output = result
        .final_output
        .expect("ordinary fallback output")
        .into_typed::<TestValue>()
        .expect("decode ordinary fallback output");
    assert_eq!(output, TestValue { value: 42 });
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 2);
    assert_eq!(result.n_cache_misses, 3);
    assert!(!job_dir.join(".speculation").exists());
    let events = status_events(&job_dir);
    assert_eq!(event_count(&events, PROBE_GUARDED_STAGE, "begin"), 1);
    assert_eq!(event_count(&events, PROBE_GUARDED_STAGE, "end"), 1);
    assert!(events.iter().all(|event| !matches!(
        event,
        StageEvent::StageFailed { stage_name, .. } if stage_name == PROBE_GUARDED_STAGE
    )));
}

#[tokio::test(flavor = "current_thread")]
async fn prepared_speculation_does_not_publish_after_plan_deadline() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("prepared deadline tempdir");
    let job_dir = temp.path().join("job");
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut ctx = speculation_ctx(job_dir.clone(), true);
    ctx.deadline = Some(deadline);
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry())
            .expect("compile prepared deadline plan"),
        ctx,
    ));

    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    tokio::time::sleep_until(tokio::time::Instant::from_std(
        deadline + std::time::Duration::from_millis(25),
    ))
    .await;
    signals.decision_release.add_permits(1);

    let error = handle
        .await
        .expect("prepared deadline executor task")
        .expect_err("expired plan must refuse prepared publication");
    assert!(matches!(error, PlanError::DeadlineExceeded { .. }));
    assert!(
        !job_dir
            .join(format!("stages/2-{PROBE_GUARDED_STAGE}"))
            .exists()
    );
    assert!(guarded_residue(&job_dir).is_empty());
    assert!(!job_dir.join(".speculation").exists());
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, PROBE_GUARDED_STAGE))
    );
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn prepared_deadline_surfaces_private_scratch_cleanup_failure() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.seal_target_scratch.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("sealed deadline tempdir");
    let job_dir = temp.path().join("job");
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut ctx = speculation_ctx(job_dir.clone(), true);
    ctx.deadline = Some(deadline);
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry())
            .expect("compile sealed deadline plan"),
        ctx,
    ));

    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    tokio::time::sleep_until(tokio::time::Instant::from_std(
        deadline + std::time::Duration::from_millis(25),
    ))
    .await;
    signals.decision_release.add_permits(1);

    let error = handle
        .await
        .expect("sealed deadline executor task")
        .expect_err("deadline must not hide undeletable private residue");
    assert!(
        error
            .to_string()
            .contains("failed to remove private speculation scratch"),
        "cleanup failure must dominate terminal deadline teardown: {error}"
    );
    let scratch = job_dir.join(".speculation");
    assert!(scratch.exists(), "fixture must prove cleanup really failed");
    assert!(
        !job_dir
            .join(format!("stages/2-{PROBE_GUARDED_STAGE}"))
            .exists(),
        "failed cleanup must never expose the private stage canonically"
    );

    make_tree_owner_writable(&scratch);
    std::fs::remove_dir_all(&scratch).expect("remove restored sealed-deadline fixture");
}

#[tokio::test(flavor = "current_thread")]
async fn selected_running_speculation_does_not_publish_after_plan_deadline() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.hold_target.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("running deadline tempdir");
    let job_dir = temp.path().join("job");
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
    let mut ctx = speculation_ctx(job_dir.clone(), true);
    ctx.deadline = Some(deadline);
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry())
            .expect("compile running deadline plan"),
        ctx,
    ));

    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    signals.decision_release.add_permits(1);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    tokio::time::sleep_until(tokio::time::Instant::from_std(
        deadline + std::time::Duration::from_millis(25),
    ))
    .await;
    signals.target_release.add_permits(1);

    let error = handle
        .await
        .expect("running deadline executor task")
        .expect_err("expired plan must refuse selected running publication");
    assert!(matches!(error, PlanError::DeadlineExceeded { .. }));
    assert!(
        !job_dir
            .join(format!("stages/2-{PROBE_GUARDED_STAGE}"))
            .exists()
    );
    assert!(guarded_residue(&job_dir).is_empty());
    assert!(!job_dir.join(".speculation").exists());
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, PROBE_GUARDED_STAGE))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn selected_speculation_replays_step_gap_after_private_overflow() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.profile_target.store(true, Ordering::SeqCst);
    signals.target_steps.store(5_000, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("speculative step overflow tempdir");
    let job_dir = temp.path().join("job");
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry())
            .expect("compile speculative step overflow plan"),
        speculation_ctx(job_dir.clone(), true).with_training_io_selection_budget_bytes(1024 * 1024),
    ));

    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    signals.decision_release.add_permits(1);
    handle
        .await
        .expect("step overflow executor task")
        .expect("selected overflow plan succeeds");

    let events = status_events(&job_dir);
    let configured: Vec<_> = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match event {
            StageEvent::StageIoConfigured {
                stage_name,
                profile,
                ..
            } if stage_name == PROBE_GUARDED_STAGE => Some((index, profile)),
            _ => None,
        })
        .collect();
    assert_eq!(
        configured.len(),
        1,
        "selected speculation replays one profile"
    );
    assert!(matches!(configured[0].1.metrics, IoMode::Bounded { .. }));
    let begin_index = events
        .iter()
        .position(|event| {
            matches!(
                event,
                StageEvent::StageBegin { stage_name, .. } if stage_name == PROBE_GUARDED_STAGE
            )
        })
        .expect("selected speculation replays StageBegin");
    assert!(configured[0].0 < begin_index);

    let expected_retained = DEFAULT_BROADCAST_CAPACITY - 4;
    let expected_dropped = 5_000 - expected_retained;
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StageEvent::StepGap { dropped } if *dropped as usize == expected_dropped)),
        "private overflow must preserve the exact dropped-step count"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event,
                StageEvent::StageStep { stage_name, .. } if stage_name == PROBE_GUARDED_STAGE
            ))
            .count(),
        expected_retained,
        "retained private steps must fill only the canonical replay budget"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn ordinary_same_key_supersession_keeps_selected_target_accounted() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.hold_target.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("same-key supersession tempdir");
    let job_dir = temp.path().join("job");
    let ctx = speculation_ctx(job_dir.clone(), true).with_max_in_flight(3);
    let handle = tokio::spawn(execute_plan(
        speculation_same_key_supersede_plan(&registry()),
        ctx,
    ));

    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.identity_started).await;
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        take_signal(&signals.target_started),
    )
    .await
    .expect("spare slot starts speculation after ordinary work is offered");
    signals.decision_release.add_permits(1);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(signals.target_runs.load(Ordering::SeqCst), 1);
    signals.identity_release.add_permits(1);

    let result = tokio::time::timeout(std::time::Duration::from_secs(3), handle)
        .await
        .expect("same-key supersession plan terminates")
        .expect("same-key supersession executor task")
        .expect("same-key supersession remains successful");
    assert_eq!(
        signals.target_runs.load(Ordering::SeqCst),
        2,
        "one private attempt and one ordinary same-key owner must suffice"
    );
    assert!(result.n_cache_hits >= 1);
    assert!(result.final_output.is_some());
    assert!(!job_dir.join(".speculation").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn merged_same_key_candidates_start_only_one_private_attempt() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let registry = registry();
    let first = speculation_spec(true)
        .compile(&registry)
        .expect("compile first speculative component");
    let mut second_spec = speculation_spec(true);
    second_spec.nodes[1].stage = ALT_PROBE_SOURCE_STAGE.into();
    let second = second_spec
        .compile(&registry)
        .expect("compile second speculative component");
    let (plan, _) = CompiledPlan::from_components(
        "same-key-speculative-components".into(),
        serde_json::Value::Null,
        vec![first, second],
    );
    let optimizer = DagOptimizer {
        speculative_execution: true,
        ..DagOptimizer::new()
    };
    let (plan, _) = optimizer.optimize(plan);
    assert_eq!(
        plan.speculation_candidates().collect::<Vec<_>>(),
        vec![2, 5]
    );
    let signals = ProbeSignals::new();
    signals.hold_target.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("same-key candidate tempdir");
    let job_dir = temp.path().join("job");
    let mut ctx = speculation_ctx(job_dir, true)
        .with_max_in_flight(6)
        .with_resource_limit(Resource::Network, 2);
    ctx.dag_optimizer = None;
    let cancel = ctx.cancel.clone();
    let handle = tokio::spawn(execute_plan(plan, ctx));

    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    assert_eq!(
        signals.target_runs.load(Ordering::SeqCst),
        1,
        "a cache key may have only one private speculative owner"
    );
    cancel.cancel();
    let error = handle
        .await
        .expect("same-key candidate executor task")
        .expect_err("test cancellation stops the merged plan");
    assert!(matches!(error, PlanError::Cancelled));
}

#[tokio::test(flavor = "current_thread")]
async fn selected_publication_contains_remote_cache_plugin_panic() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let registry = registry();
    install_probe_signals(ProbeSignals::new());
    let baseline_temp = tempfile::tempdir().expect("panic target baseline tempdir");
    let baseline_job = baseline_temp.path().join("job");
    execute_plan(
        speculation_ungated_spec()
            .compile(&registry)
            .expect("compile panic target baseline"),
        ExecCtx::new(baseline_job.clone()),
    )
    .await
    .expect("panic target baseline run");
    let target_key = CacheProof::read_from(
        &baseline_job.join(format!("stages/1-{PROBE_GUARDED_STAGE}/cache-proof.json")),
    )
    .expect("baseline target proof")
    .key;

    let signals = ProbeSignals::new();
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("panic target publication tempdir");
    let job_dir = temp.path().join("job");
    let mut ctx = speculation_ctx(job_dir.clone(), true).with_bypass_cache(true);
    ctx.cache = Arc::new(
        CacheHandle::job_local(job_dir.join("_cache"))
            .with_remote(Arc::new(PanicOnTargetPutStore { target: target_key })),
    );
    let handle = tokio::spawn(execute_plan(
        speculation_spec(true)
            .compile(&registry)
            .expect("compile panic target publication plan"),
        ctx,
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    signals.decision_release.add_permits(1);

    let result = handle
        .await
        .expect("remote panic must not unwind the coordinator")
        .expect("selected result remains locally publishable");
    assert!(result.final_output.is_some());
    let events = status_events(&job_dir);
    assert_eq!(event_count(&events, PROBE_GUARDED_STAGE, "begin"), 1);
    assert_eq!(event_count(&events, PROBE_GUARDED_STAGE, "end"), 1);
    assert!(
        job_dir
            .join(format!("stages/2-{PROBE_GUARDED_STAGE}/cache-proof.json"))
            .is_file()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn selected_publication_panic_rolls_back_canonical_state() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("publication rollback tempdir");
    let job_dir = temp.path().join("job");
    let handle = tokio::spawn(execute_plan(
        speculation_publish_panic_spec()
            .compile(&registry())
            .expect("compile publication rollback plan"),
        speculation_ctx(job_dir.clone(), true),
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    signals.decision_release.add_permits(1);

    let error = handle
        .await
        .expect("publication panic must not unwind the coordinator")
        .expect_err("publication panic fails the selected plan closed");
    assert!(
        matches!(error, PlanError::Other(message) if message.contains("speculative publication panicked"))
    );
    assert!(
        !job_dir
            .join(format!("stages/2-{PANIC_PUBLISH_STAGE}"))
            .exists(),
        "transaction guard must remove the renamed canonical stage"
    );
    assert!(!job_dir.join(".speculation").exists());
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, PANIC_PUBLISH_STAGE)),
        "publication hooks run before canonical lifecycle becomes visible"
    );
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn selected_publication_panic_surfaces_canonical_rollback_failure() {
    let _guard = SPECULATION_TEST_LOCK.lock().await;
    let signals = ProbeSignals::new();
    signals.seal_target_scratch.store(true, Ordering::SeqCst);
    install_probe_signals(signals.clone());
    let temp = tempfile::tempdir().expect("publication rollback failure tempdir");
    let job_dir = temp.path().join("job");
    let handle = tokio::spawn(execute_plan(
        speculation_publish_panic_spec()
            .compile(&registry())
            .expect("compile publication rollback failure plan"),
        speculation_ctx(job_dir.clone(), true),
    ));
    take_signal(&signals.decision_started).await;
    take_signal(&signals.source_finished).await;
    take_signal(&signals.target_started).await;
    take_signal(&signals.target_finished).await;
    signals.decision_release.add_permits(1);

    let error = handle
        .await
        .expect("rollback failure must not unwind the coordinator")
        .expect_err("incomplete canonical publication must fail closed");
    let message = error.to_string();
    assert!(
        message.contains("speculative publication panicked"),
        "{message}"
    );
    assert!(
        message.contains("failed to roll back speculative publication"),
        "rollback failure must survive panic capture: {message}"
    );
    let canonical = job_dir.join(format!("stages/2-{PANIC_PUBLISH_STAGE}"));
    assert!(
        canonical.exists(),
        "fixture must prove the incomplete canonical directory resisted rollback"
    );
    assert!(!job_dir.join(".speculation").exists());
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, PANIC_PUBLISH_STAGE)),
        "failed publication must remain absent from canonical lifecycle"
    );

    make_tree_owner_writable(&canonical);
    std::fs::remove_dir_all(&canonical).expect("remove restored rollback-failure fixture");
}

#[tokio::test(flavor = "current_thread")]
async fn true_decision_runs_guarded_branch_and_returns_its_output() {
    let temp = tempfile::tempdir().expect("conditional true tempdir");
    let job_dir = temp.path().join("job");
    let plan = conditional_spec(true)
        .compile(&registry())
        .expect("compile true condition");

    let result = execute_plan(plan, ExecCtx::new(job_dir.clone()))
        .await
        .expect("execute selected branch");

    let output = result
        .final_output
        .expect("selected guarded terminal produces final output")
        .into_typed::<TestValue>()
        .expect("decode guarded output");
    assert_eq!(output, TestValue { value: 42 });
    assert!(
        job_dir
            .join(format!("stages/2-{GUARDED_STAGE}/guard-ran"))
            .is_file(),
        "the selected guarded stage must run and promote its private output"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn false_decision_prunes_without_lifecycle_or_cache_materialization() {
    let temp = tempfile::tempdir().expect("conditional false tempdir");
    let job_dir = temp.path().join("job");
    let plan = conditional_spec(false)
        .compile(&registry())
        .expect("compile false condition");

    let result = execute_plan(plan, ExecCtx::new(job_dir.clone()))
        .await
        .expect("pruned branch is a successful plan outcome");

    assert!(
        result.final_output.is_none(),
        "pruned terminal has no output"
    );
    assert_eq!(result.n_cache_misses, 2, "only selector and source execute");
    assert_eq!(cache_entry_count(&job_dir), 2);
    assert!(
        !job_dir.join(format!("stages/2-{GUARDED_STAGE}")).exists(),
        "an unselected branch must not acquire a real stage directory"
    );
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, GUARDED_STAGE)),
        "an unselected branch must emit no run/cache lifecycle record"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| is_pruned_stage(event, GUARDED_STAGE))
            .count(),
        1,
        "the audit trail distinguishes an intentional prune from a missing stage"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn false_decision_does_not_return_a_guarded_roots_seed_input() {
    let temp = tempfile::tempdir().expect("conditional root tempdir");
    let job_dir = temp.path().join("job");

    let result = execute_plan(
        conditional_root_spec(false)
            .compile(&registry())
            .expect("compile guarded graph-input root"),
        ExecCtx::new(job_dir.clone()),
    )
    .await
    .expect("prune guarded graph-input root");

    assert!(
        result.final_output.is_none(),
        "the root's pre-seeded unit input is not a produced final artifact"
    );
    assert_eq!((result.n_cache_hits, result.n_cache_misses), (0, 1));
    let events = status_events(&job_dir);
    assert!(
        events
            .iter()
            .all(|event| !materializes_stage(event, SOURCE_STAGE))
    );
    assert!(
        events
            .iter()
            .any(|event| is_pruned_stage(event, SOURCE_STAGE))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn condition_relation_is_cache_neutral_for_the_selected_data_path() {
    let temp = tempfile::tempdir().expect("cache-neutral condition tempdir");
    let job_dir = temp.path().join("same-job");
    let registry = registry();

    let selected = execute_plan(
        conditional_spec(true)
            .compile(&registry)
            .expect("compile selected conditional plan"),
        ExecCtx::new(job_dir.clone()),
    )
    .await
    .expect("run selected conditional plan");
    assert_eq!((selected.n_cache_hits, selected.n_cache_misses), (0, 3));
    let selected_proof =
        CacheProof::read_from(&job_dir.join(format!("stages/2-{GUARDED_STAGE}/cache-proof.json")))
            .expect("selected guarded cache proof");

    let ungated = execute_plan(
        ungated_spec()
            .compile(&registry)
            .expect("compile equivalent ungated plan"),
        ExecCtx::new(job_dir.clone()),
    )
    .await
    .expect("run equivalent ungated plan in the same job/cache");
    assert_eq!(
        (ungated.n_cache_hits, ungated.n_cache_misses),
        (2, 0),
        "source and guarded node must reuse the selected plan's exact entries"
    );
    let ungated_proof =
        CacheProof::read_from(&job_dir.join(format!("stages/1-{GUARDED_STAGE}/cache-proof.json")))
            .expect("ungated guarded cache proof");
    assert_eq!(
        selected_proof.key, ungated_proof.key,
        "a condition relation must not enter the guarded node's cache key"
    );
}

#[test]
fn execute_plan_forces_parallel_with_no_executor_environment_override() {
    let current_exe = std::env::current_exe().expect("locate this integration-test binary");
    let output = Command::new(current_exe)
        .arg("--ignored")
        .arg("--exact")
        .arg("conditional_parallel_subprocess_entry")
        .arg("--nocapture")
        .env_remove("BLUT_EXECUTOR")
        .env_remove("BLUT_KILL_ON_NAN")
        .output()
        .expect("launch isolated conditional executor check");
    assert!(
        output.status.success(),
        "isolated conditional execution failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "subprocess entry used by execute_plan_forces_parallel_with_no_executor_environment_override"]
fn conditional_parallel_subprocess_entry() {
    assert!(std::env::var_os("BLUT_EXECUTOR").is_none());
    assert!(std::env::var_os("BLUT_KILL_ON_NAN").is_none());

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build subprocess runtime");
    runtime.block_on(async {
        let temp = tempfile::tempdir().expect("parallel dispatch tempdir");
        let job_dir = temp.path().join("job");
        let result = execute_plan(
            conditional_spec(false)
                .compile(&registry())
                .expect("compile subprocess conditional plan"),
            ExecCtx::new(job_dir.clone()),
        )
        .await
        .expect("condition gate forces the parallel executor");

        assert!(result.final_output.is_none());
        let events = status_events(&job_dir);
        assert!(
            events
                .iter()
                .all(|event| !materializes_stage(event, GUARDED_STAGE)),
            "false condition must still prune when no executor env override exists"
        );
        assert!(
            events
                .iter()
                .any(|event| is_pruned_stage(event, GUARDED_STAGE)),
            "the forced-parallel path must record the control prune"
        );
    });
}
