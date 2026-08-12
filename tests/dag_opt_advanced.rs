// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Named progress gate for ADR 0102's landed advanced-optimizer slice.
//!
//! User-priority scheduling, live cache-warm ready ordering, conservative
//! linear coalescing, private conditional speculation, and manifest-certified
//! pipeline parallelism each have an independently named gate.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use blut::framework::artifact::{
    Artifact, ArtifactMetadata, BranchDecision, ContentHash, InvocationKey, ListOf,
};
use blut::framework::async_io::{IoMode, TrainingIoCandidate, TrainingIoHints};
use blut::framework::cache::{CacheHandle, CacheProof};
use blut::framework::cookbook::{Cookbook, Registry};
use blut::framework::dag_opt::DagOptimizer;
use blut::framework::executor::{ExecCtx, ParallelExecutor};
use blut::framework::object_store::{ObjectKey, ObjectStore, ObjectStoreAdapter, StoreError};
use blut::framework::plan::CompiledPlan;
use blut::framework::plan_spec::{
    ConditionGateSpec, MapSpec, PLAN_SPEC_VERSION, PlanSpec, SpecNode,
};
use blut::framework::resource::Resource;
use blut::framework::stage::{
    ErasedStageCtor, PipelineManifest, Stage, StageContext, StageExecutionBoundary,
};
use blut::framework::status::StageEvent;
use blut::framework::{PlanError, StageError};
use blut::recipes::recipe::RecipeDef;
use futures::FutureExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

static EXECUTION_ORDER: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
static FUSION_TASK_IDS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
static FUSION_ROOT_STARTED: tokio::sync::Notify = tokio::sync::Notify::const_new();
static FUSION_RELEASE_ROOT: tokio::sync::Notify = tokio::sync::Notify::const_new();
static FUSION_WAITER_STARTED: tokio::sync::Notify = tokio::sync::Notify::const_new();
static FUSION_RELEASE_WAITER: tokio::sync::Notify = tokio::sync::Notify::const_new();
static FUSION_WAITER_ACQUIRED: AtomicBool = AtomicBool::new(false);
static DIRECT_ARTIFACT_BINARY_DESERIALIZES: AtomicUsize = AtomicUsize::new(0);
static FUSION_DUPLICATE_RUNS: AtomicUsize = AtomicUsize::new(0);
static FUSION_BOUNDARY_DEADLINE: std::sync::Mutex<Option<std::time::Instant>> =
    std::sync::Mutex::new(None);
static SPEC_GATE_DECISION_STARTED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static SPEC_GATE_DECISION_RELEASE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static SPEC_GATE_TARGET_FINISHED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static SPEC_GATE_TARGET_RUNS: AtomicUsize = AtomicUsize::new(0);
static PIPE_PARENT_READY: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_PARENT_RELEASE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_CHILD_STARTED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_CHILD_RELEASE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_SECOND_SUBMIT_STARTED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_SECOND_EMIT_RETURNED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_CAP_CHILD_ZERO_STARTED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_CAP_CHILD_ZERO_RELEASE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_CAP_CHILD_ONE_STARTED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_CAP_CHILD_ONE_RELEASE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_PRIVATE_CHILD_FINISHED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_SIBLING_STARTED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_SIBLING_RELEASE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_STUCK_CHILD_STARTED: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_STUCK_CHILD_RELEASE: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(0);
static PIPE_CHILD_RUNS: AtomicUsize = AtomicUsize::new(0);
static PIPE_CHILD_PROFILE_DECLARATIONS: AtomicUsize = AtomicUsize::new(0);
static PIPE_MISMATCH_ACCEPTED: AtomicBool = AtomicBool::new(false);
static PIPE_FAILURE_ACCEPTED: AtomicBool = AtomicBool::new(false);
static PIPE_DEFAULT_OFF_PROFILE_INLINE: AtomicBool = AtomicBool::new(false);
static PIPE_PUBLICATION_HASH_CALLS: AtomicUsize = AtomicUsize::new(0);
static PIPE_PUBLICATION_CANCEL: std::sync::Mutex<Option<CancellationToken>> =
    std::sync::Mutex::new(None);

fn record_fusion_task(label: &str) {
    if label.starts_with("fuse-") {
        let task = tokio::task::try_id()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "outside-tokio-task".into());
        FUSION_TASK_IDS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(task);
    }
}

#[derive(Debug)]
struct CountingMissStore {
    gets: Arc<AtomicUsize>,
    delay: std::time::Duration,
}

#[derive(Debug, Default)]
struct RacingHitState {
    first_key: Option<ObjectKey>,
    gets_by_key: std::collections::HashMap<ObjectKey, usize>,
    objects: std::collections::HashMap<ObjectKey, Vec<u8>>,
}

#[derive(Debug)]
struct RacingHitStore {
    state: std::sync::Mutex<RacingHitState>,
}

impl RacingHitStore {
    fn arm(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.first_key = None;
        state.gets_by_key.clear();
    }
}

#[async_trait]
impl ObjectStoreAdapter for RacingHitStore {
    async fn read_raw(&self, key: ObjectKey) -> Result<Option<Vec<u8>>, StoreError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let first_key = *state.first_key.get_or_insert(key);
        let gets = state.gets_by_key.entry(key).or_default();
        *gets += 1;
        if key == first_key && *gets == 1 {
            return Ok(None);
        }
        Ok(state.objects.get(&key).cloned())
    }

    async fn create_raw(&self, key: ObjectKey, stored: Vec<u8>) -> Result<bool, StoreError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match state.objects.entry(key) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(stored);
                Ok(true)
            }
            std::collections::hash_map::Entry::Occupied(_) => Ok(false),
        }
    }

    async fn contains_raw(&self, key: ObjectKey) -> Result<bool, StoreError> {
        Ok(self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .objects
            .contains_key(&key))
    }
}

#[async_trait]
impl ObjectStoreAdapter for CountingMissStore {
    async fn read_raw(&self, _key: ObjectKey) -> Result<Option<Vec<u8>>, StoreError> {
        self.gets.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(self.delay);
        Ok(None)
    }

    async fn create_raw(&self, _key: ObjectKey, _stored: Vec<u8>) -> Result<bool, StoreError> {
        Ok(true)
    }

    async fn contains_raw(&self, _key: ObjectKey) -> Result<bool, StoreError> {
        Ok(false)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct OrderArgs {
    label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OrderArtifact {
    path: std::path::PathBuf,
    content_hash: ContentHash,
}

impl Artifact for OrderArtifact {
    const KIND: &'static str = "test.order-artifact";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }

    fn primary_path(&self) -> &std::path::Path {
        &self.path
    }
}

#[derive(Clone, Debug, Serialize)]
struct DirectArtifact {
    value: u32,
    content_hash: ContentHash,
}

#[derive(Deserialize)]
struct DirectArtifactWire {
    value: u32,
    content_hash: ContentHash,
}

impl<'de> Deserialize<'de> for DirectArtifact {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let is_binary = !deserializer.is_human_readable();
        let wire = DirectArtifactWire::deserialize(deserializer)?;
        if is_binary {
            DIRECT_ARTIFACT_BINARY_DESERIALIZES.fetch_add(1, Ordering::SeqCst);
        }
        Ok(Self {
            value: wire.value,
            content_hash: wire.content_hash,
        })
    }
}

impl Artifact for DirectArtifact {
    const KIND: &'static str = "test.direct-artifact";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }

    fn primary_path(&self) -> &std::path::Path {
        std::path::Path::new(".")
    }
}

struct DirectRoot;

#[async_trait]
impl Stage for DirectRoot {
    const NAME: &'static str = "direct_root";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = ();
    type Output = DirectArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &OrderArgs,
    ) -> Result<DirectArtifact, StageError> {
        Ok(DirectArtifact {
            value: 1,
            content_hash: ContentHash::of_bytes(&1u32.to_le_bytes()),
        })
    }
}

struct DirectAfter;

#[async_trait]
impl Stage for DirectAfter {
    const NAME: &'static str = "direct_after";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = DirectArtifact;
    type Output = DirectArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: DirectArtifact,
        _args: &OrderArgs,
    ) -> Result<DirectArtifact, StageError> {
        let value = input.value + 1;
        Ok(DirectArtifact {
            value,
            content_hash: ContentHash::of_bytes(&value.to_le_bytes()),
        })
    }
}

struct DirectIdentity;

#[async_trait]
impl Stage for DirectIdentity {
    const NAME: &'static str = "direct_identity";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = DirectArtifact;
    type Output = DirectArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: DirectArtifact,
        args: &OrderArgs,
    ) -> Result<DirectArtifact, StageError> {
        record_fusion_task(&args.label);
        Ok(input)
    }
}

struct DirectCountedSlow;

#[async_trait]
impl Stage for DirectCountedSlow {
    const NAME: &'static str = "direct_counted_slow";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = DirectArtifact;
    type Output = DirectArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: DirectArtifact,
        args: &OrderArgs,
    ) -> Result<DirectArtifact, StageError> {
        record_fusion_task(&args.label);
        FUSION_DUPLICATE_RUNS.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let value = input.value + 1;
        Ok(DirectArtifact {
            value,
            content_hash: ContentHash::of_bytes(&value.to_le_bytes()),
        })
    }
}

struct RecordOrder;

#[async_trait]
impl Stage for RecordOrder {
    const NAME: &'static str = "record_order";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = ();
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        record_fusion_task(&args.label);
        if args.label == "fuse-admission-root" {
            FUSION_ROOT_STARTED.notify_one();
            FUSION_RELEASE_ROOT.notified().await;
        }
        EXECUTION_ORDER
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(args.label.clone());
        std::fs::create_dir_all(&ctx.stage_dir)
            .map_err(|error| StageError::Backend(error.into()))?;
        let path = ctx.stage_dir.join(format!("{}.txt", args.label));
        std::fs::write(&path, args.label.as_bytes())
            .map_err(|error| StageError::Backend(error.into()))?;
        Ok(OrderArtifact {
            content_hash: ContentHash::of_bytes(args.label.as_bytes()),
            path,
        })
    }
}

struct RecordAfter;

#[async_trait]
impl Stage for RecordAfter {
    const NAME: &'static str = "record_after";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = OrderArtifact;
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: OrderArtifact,
        args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        record_fusion_task(&args.label);
        EXECUTION_ORDER
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(args.label.clone());
        std::fs::create_dir_all(&ctx.stage_dir)
            .map_err(|error| StageError::Backend(error.into()))?;
        let path = ctx.stage_dir.join(format!("{}.txt", args.label));
        std::fs::write(&path, args.label.as_bytes())
            .map_err(|error| StageError::Backend(error.into()))?;
        Ok(OrderArtifact {
            content_hash: ContentHash::of_bytes(args.label.as_bytes()),
            path,
        })
    }
}

/// A deterministic typed stage that intentionally leaves its execution
/// boundary at the framework default. The optimizer must preserve that opaque
/// boundary instead of inferring in-process safety from the blanket StageDyn
/// implementation.
struct RecordOpaqueBoundary;

#[async_trait]
impl Stage for RecordOpaqueBoundary {
    const NAME: &'static str = "record_opaque_boundary";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = OrderArtifact;
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: OrderArtifact,
        args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        RecordAfter.run(ctx, input, args).await
    }
}

/// Known child-process ownership remains a hard optimizer boundary even when
/// the stage is deterministic and implements the typed handoff mechanism.
struct RecordSubprocessBoundary;

#[async_trait]
impl Stage for RecordSubprocessBoundary {
    const NAME: &'static str = "record_subprocess_boundary";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::Subprocess;
    type Input = OrderArtifact;
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: OrderArtifact,
        args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        RecordAfter.run(ctx, input, args).await
    }
}

struct RecordNondeterministic;

#[async_trait]
impl Stage for RecordNondeterministic {
    const NAME: &'static str = "record_nondeterministic";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const DETERMINISTIC: bool = false;
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = OrderArtifact;
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: OrderArtifact,
        args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        RecordAfter.run(ctx, input, args).await
    }
}

struct RecordSlowAfter;

#[async_trait]
impl Stage for RecordSlowAfter {
    const NAME: &'static str = "record_slow_after";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = OrderArtifact;
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: OrderArtifact,
        args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        let deadline = *FUSION_BOUNDARY_DEADLINE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(deadline) = deadline {
            tokio::time::sleep_until(tokio::time::Instant::from_std(
                deadline + std::time::Duration::from_millis(25),
            ))
            .await;
        }
        RecordAfter.run(ctx, input, args).await
    }
}

struct RecordPanicking;

#[async_trait]
impl Stage for RecordPanicking {
    const NAME: &'static str = "record_panicking";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = OrderArtifact;
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: OrderArtifact,
        _args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        panic!("simulated fused-stage panic");
    }
}

struct RecordFailing;

#[async_trait]
impl Stage for RecordFailing {
    const NAME: &'static str = "record_failing";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = OrderArtifact;
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        _input: OrderArtifact,
        _args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        Err(StageError::BadInput("simulated fused-stage failure".into()))
    }
}

struct RecordNetworkAfter;

#[async_trait]
impl Stage for RecordNetworkAfter {
    const NAME: &'static str = "record_network_after";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Network];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = OrderArtifact;
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: OrderArtifact,
        args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        RecordAfter.run(ctx, input, args).await
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct GateDecisionArgs {
    value: bool,
}

struct HeldGateDecision;

#[async_trait]
impl Stage for HeldGateDecision {
    const NAME: &'static str = "held_gate_decision";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[];
    type Input = ();
    type Output = BranchDecision;
    type Args = GateDecisionArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &GateDecisionArgs,
    ) -> Result<BranchDecision, StageError> {
        SPEC_GATE_DECISION_STARTED.add_permits(1);
        tokio::select! {
            permit = SPEC_GATE_DECISION_RELEASE.acquire() => {
                permit.expect("speculation gate decision release open").forget();
            }
            _ = ctx.cancel.cancelled() => return Err(StageError::Cancelled),
        }
        Ok(BranchDecision { value: args.value })
    }
}

struct RecordSpeculativeAfter;

#[async_trait]
impl Stage for RecordSpeculativeAfter {
    const NAME: &'static str = "record_speculative_after";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Network];
    const SPECULATION_SAFE: bool = true;
    type Input = OrderArtifact;
    type Output = OrderArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: OrderArtifact,
        args: &OrderArgs,
    ) -> Result<OrderArtifact, StageError> {
        SPEC_GATE_TARGET_RUNS.fetch_add(1, Ordering::SeqCst);
        std::fs::create_dir_all(&ctx.stage_dir)
            .map_err(|error| StageError::Backend(error.into()))?;
        let path = ctx.stage_dir.join(format!("{}.txt", args.label));
        std::fs::write(&path, input.content_hash.to_hex())
            .map_err(|error| StageError::Backend(error.into()))?;
        let output = OrderArtifact {
            content_hash: ContentHash::of_bytes(args.label.as_bytes()),
            path,
        };
        SPEC_GATE_TARGET_FINISHED.add_permits(1);
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum PipelineCase {
    Overlap,
    Capacity,
    ManifestMismatch,
    ManifestTooShort,
    ManifestTooLong,
    FailAfterEmission,
    FailWhileChildBlocked,
    CancelDuringPublication,
    CorruptSpill,
    CorruptInputSpill,
    OversizePrivateResult,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct PipelineParentArgs {
    case: PipelineCase,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PipelineItem {
    case: PipelineCase,
    index: u32,
    content_hash: ContentHash,
}

impl Artifact for PipelineItem {
    const KIND: &'static str = "test.pipeline-item";
    const SCHEMA: u32 = 1;
    const PIPELINE_STANDARD_ENCODING: bool = true;

    fn pipeline_storage_is_stable(&self, _producer_stage_dir: &std::path::Path) -> bool {
        true
    }

    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }

    fn primary_path(&self) -> &std::path::Path {
        std::path::Path::new(".")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PipelineChildArtifact {
    source_index: u32,
    content_hash: ContentHash,
    padding: Vec<u8>,
}

impl Artifact for PipelineChildArtifact {
    const KIND: &'static str = "test.pipeline-child";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        let cancel = PIPE_PUBLICATION_CANCEL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        if let Some(cancel) = cancel
            && PIPE_PUBLICATION_HASH_CALLS.fetch_add(1, Ordering::SeqCst) == 1
        {
            // The first call computes the private run's output identity. The
            // second runs after its scratch directory has been renamed for
            // selected publication, exercising the rollback/stop boundary.
            cancel.cancel();
        }
        self.content_hash
    }

    fn primary_path(&self) -> &std::path::Path {
        std::path::Path::new(".")
    }
}

fn pipeline_items(case: PipelineCase) -> Vec<PipelineItem> {
    let width = usize::from(matches!(
        case,
        PipelineCase::Capacity | PipelineCase::ManifestTooShort
    )) + 1;
    (0..width)
        .map(|index| {
            let identity = format!("pipeline-item:{case:?}:{index}");
            PipelineItem {
                case,
                index: index as u32,
                content_hash: ContentHash::of_bytes(identity.as_bytes()),
            }
        })
        .collect()
}

async fn take_pipeline_signal(
    ctx: &StageContext,
    semaphore: &'static tokio::sync::Semaphore,
) -> Result<(), StageError> {
    tokio::select! {
        permit = semaphore.acquire() => {
            permit.expect("pipeline test semaphore remains open").forget();
            Ok(())
        }
        _ = ctx.cancel.cancelled() => Err(StageError::Cancelled),
    }
}

struct PipelineParent;

#[async_trait]
impl Stage for PipelineParent {
    const NAME: &'static str = "pipeline_parent";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    const PIPELINE_OUTPUT_SAFE: bool = true;
    type Input = ();
    type Output = ListOf<PipelineItem>;
    type Args = PipelineParentArgs;

    fn training_io_candidates(
        &self,
        _args: &Self::Args,
        _hints: TrainingIoHints,
    ) -> Vec<TrainingIoCandidate> {
        vec![
            TrainingIoCandidate {
                data_replicas: 1,
                decode_workers: 0,
                prefetch_per_worker: 0,
                cuda_staging_slots: 0,
                pipeline: IoMode::Bounded {
                    capacity: 1,
                    max_item_bytes: 4096,
                },
                metrics: IoMode::Inline,
                checkpoints: IoMode::Inline,
                batch_bytes: None,
                checkpoint_snapshot_bytes: None,
                fixed_overhead_bytes: Some(0),
            },
            TrainingIoCandidate {
                data_replicas: 1,
                decode_workers: 0,
                prefetch_per_worker: 0,
                cuda_staging_slots: 0,
                pipeline: IoMode::Inline,
                metrics: IoMode::Inline,
                checkpoints: IoMode::Inline,
                batch_bytes: None,
                checkpoint_snapshot_bytes: None,
                fixed_overhead_bytes: None,
            },
        ]
    }

    fn pipeline_manifest(
        &self,
        _input: &Self::Input,
        args: &Self::Args,
    ) -> Option<PipelineManifest> {
        let mut element_hashes = pipeline_items(args.case)
            .into_iter()
            .map(|item| item.content_hash())
            .collect::<Vec<_>>();
        if args.case == PipelineCase::ManifestMismatch {
            element_hashes[0] = ContentHash::of_bytes(b"deliberately-wrong-pipeline-manifest");
        } else if args.case == PipelineCase::ManifestTooShort {
            element_hashes.truncate(1);
        } else if args.case == PipelineCase::ManifestTooLong {
            element_hashes.push(ContentHash::of_bytes(b"nonexistent-pipeline-element"));
        }
        Some(PipelineManifest::new(element_hashes))
    }

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &Self::Args,
    ) -> Result<ListOf<PipelineItem>, StageError> {
        let items = pipeline_items(args.case);
        match args.case {
            PipelineCase::Overlap => {
                if ctx.pipeline_enabled() {
                    let _accepted = ctx.emit_pipeline_item(0, &items[0]).await?;
                } else {
                    PIPE_DEFAULT_OFF_PROFILE_INLINE.store(
                        ctx.training_io_profile
                            .as_ref()
                            .is_some_and(|profile| profile.pipeline == IoMode::Inline),
                        Ordering::SeqCst,
                    );
                }
                PIPE_PARENT_READY.add_permits(1);
                take_pipeline_signal(ctx, &PIPE_PARENT_RELEASE).await?;
            }
            PipelineCase::Capacity => {
                assert!(
                    ctx.pipeline_enabled(),
                    "capacity fixture requires pipeline mode"
                );
                let first_accepted = ctx.emit_pipeline_item(0, &items[0]).await?;
                assert!(first_accepted, "first certified item must enter the lane");
                PIPE_SECOND_SUBMIT_STARTED.add_permits(1);
                let second_accepted = ctx.emit_pipeline_item(1, &items[1]).await?;
                assert!(second_accepted, "second certified item must enter the lane");
                PIPE_SECOND_EMIT_RETURNED.add_permits(1);
            }
            PipelineCase::ManifestMismatch => {
                if ctx.pipeline_enabled() {
                    let accepted = ctx.emit_pipeline_item(0, &items[0]).await?;
                    PIPE_MISMATCH_ACCEPTED.store(accepted, Ordering::SeqCst);
                }
            }
            PipelineCase::ManifestTooShort | PipelineCase::ManifestTooLong => {
                if ctx.pipeline_enabled() {
                    for (index, item) in items.iter().enumerate() {
                        let _ = ctx.emit_pipeline_item(index, item).await?;
                    }
                }
            }
            PipelineCase::FailAfterEmission => {
                if ctx.pipeline_enabled() {
                    let accepted = ctx.emit_pipeline_item(0, &items[0]).await?;
                    PIPE_FAILURE_ACCEPTED.store(accepted, Ordering::SeqCst);
                    if accepted {
                        take_pipeline_signal(ctx, &PIPE_PRIVATE_CHILD_FINISHED).await?;
                    }
                }
                return Err(StageError::BadInput(
                    "simulated parent failure after pipeline emission".into(),
                ));
            }
            PipelineCase::FailWhileChildBlocked => {
                assert!(
                    ctx.pipeline_enabled(),
                    "blocked-child fixture needs pipeline"
                );
                assert!(ctx.emit_pipeline_item(0, &items[0]).await?);
                take_pipeline_signal(ctx, &PIPE_STUCK_CHILD_STARTED).await?;
                return Err(StageError::BadInput(
                    "simulated parent failure while private child is blocked".into(),
                ));
            }
            PipelineCase::CancelDuringPublication => {
                assert!(ctx.pipeline_enabled());
                assert!(ctx.emit_pipeline_item(0, &items[0]).await?);
                take_pipeline_signal(ctx, &PIPE_PRIVATE_CHILD_FINISHED).await?;
            }
            PipelineCase::CorruptSpill | PipelineCase::CorruptInputSpill => {
                assert!(ctx.pipeline_enabled());
                assert!(ctx.emit_pipeline_item(0, &items[0]).await?);
                take_pipeline_signal(ctx, &PIPE_PRIVATE_CHILD_FINISHED).await?;
                let target = if args.case == PipelineCase::CorruptSpill {
                    "pipeline-prepared.bin"
                } else {
                    "pipeline-input.bin"
                };
                let mut spill = None;
                for _ in 0..100 {
                    spill = find_named_file(&ctx.job_dir.join(".pipeline"), target);
                    if spill.is_some() {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                let Some(spill) = spill else {
                    return Err(StageError::BadInput(
                        "pipeline corruption fixture did not observe the private spill".into(),
                    ));
                };
                std::fs::write(spill, b"deliberately-corrupt-private-spill")
                    .map_err(|error| StageError::Backend(error.into()))?;
            }
            PipelineCase::OversizePrivateResult => {
                assert!(ctx.pipeline_enabled());
                assert!(ctx.emit_pipeline_item(0, &items[0]).await?);
            }
        }
        Ok(ListOf(items))
    }
}

struct PipelineChild;

#[async_trait]
impl Stage for PipelineChild {
    const NAME: &'static str = "pipeline_child";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    const PIPELINE_INPUT_SAFE: bool = true;
    type Input = PipelineItem;
    type Output = PipelineChildArtifact;
    type Args = OrderArgs;

    fn training_io_candidates(
        &self,
        _args: &Self::Args,
        _hints: TrainingIoHints,
    ) -> Vec<TrainingIoCandidate> {
        PIPE_CHILD_PROFILE_DECLARATIONS.fetch_add(1, Ordering::SeqCst);
        vec![TrainingIoCandidate::default()]
    }

    async fn run(
        &self,
        ctx: &StageContext,
        input: PipelineItem,
        _args: &OrderArgs,
    ) -> Result<PipelineChildArtifact, StageError> {
        PIPE_CHILD_RUNS.fetch_add(1, Ordering::SeqCst);
        match input.case {
            PipelineCase::Overlap => {
                PIPE_CHILD_STARTED.add_permits(1);
                take_pipeline_signal(ctx, &PIPE_CHILD_RELEASE).await?;
            }
            PipelineCase::Capacity if input.index == 0 => {
                PIPE_CAP_CHILD_ZERO_STARTED.add_permits(1);
                take_pipeline_signal(ctx, &PIPE_CAP_CHILD_ZERO_RELEASE).await?;
            }
            PipelineCase::Capacity => {
                PIPE_CAP_CHILD_ONE_STARTED.add_permits(1);
                take_pipeline_signal(ctx, &PIPE_CAP_CHILD_ONE_RELEASE).await?;
            }
            PipelineCase::ManifestMismatch
            | PipelineCase::ManifestTooShort
            | PipelineCase::ManifestTooLong
            | PipelineCase::FailAfterEmission
            | PipelineCase::CancelDuringPublication
            | PipelineCase::CorruptSpill
            | PipelineCase::CorruptInputSpill
            | PipelineCase::OversizePrivateResult => {
                PIPE_PRIVATE_CHILD_FINISHED.add_permits(1);
            }
            PipelineCase::FailWhileChildBlocked => {
                PIPE_STUCK_CHILD_STARTED.add_permits(1);
                PIPE_STUCK_CHILD_RELEASE
                    .acquire()
                    .await
                    .expect("blocked-child test semaphore remains open")
                    .forget();
            }
        }
        let identity = format!("pipeline-child:{}", input.content_hash.to_hex());
        Ok(PipelineChildArtifact {
            source_index: input.index,
            content_hash: ContentHash::of_bytes(identity.as_bytes()),
            padding: if input.case == PipelineCase::OversizePrivateResult {
                vec![7; 8192]
            } else {
                Vec::new()
            },
        })
    }
}

struct PipelineSibling;

#[async_trait]
impl Stage for PipelineSibling {
    const NAME: &'static str = "pipeline_sibling";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = ();
    type Output = PipelineChildArtifact;
    type Args = OrderArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        _args: &OrderArgs,
    ) -> Result<PipelineChildArtifact, StageError> {
        PIPE_SIBLING_STARTED.add_permits(1);
        take_pipeline_signal(ctx, &PIPE_SIBLING_RELEASE).await?;
        Ok(PipelineChildArtifact {
            source_index: u32::MAX,
            content_hash: ContentHash::of_bytes(b"pipeline-sibling"),
            padding: Vec::new(),
        })
    }
}

struct GateCookbook;

impl Cookbook for GateCookbook {
    fn name(&self) -> &'static str {
        "dag-opt-gate"
    }

    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }

    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static STAGES: &[(&str, ErasedStageCtor)] = &[
            ("record_order", || Arc::new(RecordOrder)),
            ("record_after", || Arc::new(RecordAfter)),
            ("record_opaque_boundary", || Arc::new(RecordOpaqueBoundary)),
            ("record_subprocess_boundary", || {
                Arc::new(RecordSubprocessBoundary)
            }),
            ("record_nondeterministic", || {
                Arc::new(RecordNondeterministic)
            }),
            ("record_slow_after", || Arc::new(RecordSlowAfter)),
            ("record_panicking", || Arc::new(RecordPanicking)),
            ("record_failing", || Arc::new(RecordFailing)),
            ("record_network_after", || Arc::new(RecordNetworkAfter)),
            ("held_gate_decision", || Arc::new(HeldGateDecision)),
            ("record_speculative_after", || {
                Arc::new(RecordSpeculativeAfter)
            }),
            ("pipeline_parent", || Arc::new(PipelineParent)),
            ("pipeline_child", || Arc::new(PipelineChild)),
            ("pipeline_sibling", || Arc::new(PipelineSibling)),
            ("direct_root", || Arc::new(DirectRoot)),
            ("direct_after", || Arc::new(DirectAfter)),
            ("direct_identity", || Arc::new(DirectIdentity)),
            ("direct_counted_slow", || Arc::new(DirectCountedSlow)),
        ];
        STAGES
    }
}

fn compiled(priority: Option<i32>) -> CompiledPlan {
    compiled_nodes(&[("only", priority)])
}

fn compiled_nodes(nodes: &[(&str, Option<i32>)]) -> CompiledPlan {
    let graph_nodes: Vec<(&str, &str, Option<i32>)> = nodes
        .iter()
        .map(|(label, priority)| ("record_order", *label, *priority))
        .collect();
    compiled_graph(&graph_nodes, &[])
}

fn compiled_graph(nodes: &[(&str, &str, Option<i32>)], edges: &[(u32, u32)]) -> CompiledPlan {
    let mut registry = Registry::new();
    registry.register(Box::new(GateCookbook));
    PlanSpec {
        name: "priority-gate".into(),
        nodes: nodes
            .iter()
            .map(|(stage, label, priority)| SpecNode {
                stage: (*stage).into(),
                args: serde_json::json!({ "label": label }),
                retry: None,
                timeout: None,
                priority: *priority,
                pure: false,
            })
            .collect(),
        edges: edges.to_vec(),
        expansions: Vec::new(),
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
    .compile(&registry)
    .expect("compile gate plan")
}

fn speculation_discard_plan() -> CompiledPlan {
    let mut registry = Registry::new();
    registry.register(Box::new(GateCookbook));
    PlanSpec {
        name: "speculation-discard-gate".into(),
        nodes: vec![
            SpecNode {
                stage: "held_gate_decision".into(),
                args: serde_json::json!({ "value": false }),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            },
            SpecNode {
                stage: "record_order".into(),
                args: serde_json::json!({ "label": "spec-source" }),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            },
            SpecNode {
                stage: "record_speculative_after".into(),
                args: serde_json::json!({ "label": "must-discard" }),
                retry: None,
                timeout: None,
                priority: None,
                pure: true,
            },
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
    .compile(&registry)
    .expect("compile speculation discard gate")
}

fn pipeline_plan(case: PipelineCase) -> CompiledPlan {
    let mut registry = Registry::new();
    registry.register(Box::new(GateCookbook));
    PlanSpec {
        name: format!("pipeline-{case:?}"),
        nodes: vec![SpecNode {
            stage: "pipeline_parent".into(),
            args: serde_json::json!({ "case": case }),
            retry: None,
            timeout: None,
            priority: None,
            pure: false,
        }],
        edges: Vec::new(),
        expansions: vec![MapSpec {
            parent: 0,
            template: PlanSpec {
                name: "pipeline-child-template".into(),
                nodes: vec![SpecNode {
                    stage: "pipeline_child".into(),
                    args: serde_json::json!({ "label": "pipeline-child" }),
                    retry: None,
                    timeout: None,
                    priority: None,
                    // PlanSpec v1 rejects pure map templates. The child stage's
                    // PIPELINE_INPUT_SAFE const is the independent certificate.
                    pure: false,
                }],
                edges: Vec::new(),
                expansions: Vec::new(),
                condition_gates: Vec::new(),
                version: PLAN_SPEC_VERSION,
            },
            label: Some("pipeline-item".into()),
        }],
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
    .compile(&registry)
    .expect("compile manifest-certified pipeline map")
}

fn pipeline_with_running_sibling_plan() -> CompiledPlan {
    let mut registry = Registry::new();
    registry.register(Box::new(GateCookbook));
    PlanSpec {
        name: "pipeline-running-sibling".into(),
        nodes: vec![
            SpecNode {
                stage: "pipeline_sibling".into(),
                args: serde_json::json!({ "label": "pipeline-sibling" }),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            },
            SpecNode {
                stage: "pipeline_parent".into(),
                args: serde_json::json!({ "case": PipelineCase::Overlap }),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            },
        ],
        edges: Vec::new(),
        expansions: vec![MapSpec {
            parent: 1,
            template: PlanSpec {
                name: "pipeline-child-template".into(),
                nodes: vec![SpecNode {
                    stage: "pipeline_child".into(),
                    args: serde_json::json!({ "label": "pipeline-child" }),
                    retry: None,
                    timeout: None,
                    priority: None,
                    pure: false,
                }],
                edges: Vec::new(),
                expansions: Vec::new(),
                condition_gates: Vec::new(),
                version: PLAN_SPEC_VERSION,
            },
            label: Some("pipeline-item".into()),
        }],
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
    .compile(&registry)
    .expect("compile pipeline plan with running sibling")
}

fn priority_only(enabled: bool) -> DagOptimizer {
    DagOptimizer {
        eliminate_dead_code: false,
        critical_path: false,
        cache_aware: false,
        memory_aware: false,
        priority_aware: enabled,
        stage_fusion: false,
        speculative_execution: false,
        pipeline_parallelism: false,
    }
}

fn cache_only(enabled: bool) -> DagOptimizer {
    DagOptimizer {
        eliminate_dead_code: false,
        critical_path: false,
        cache_aware: enabled,
        memory_aware: false,
        priority_aware: false,
        stage_fusion: false,
        speculative_execution: false,
        pipeline_parallelism: false,
    }
}

fn fusion_only(enabled: bool) -> DagOptimizer {
    DagOptimizer {
        eliminate_dead_code: false,
        critical_path: false,
        cache_aware: false,
        memory_aware: false,
        priority_aware: false,
        stage_fusion: enabled,
        speculative_execution: false,
        pipeline_parallelism: false,
    }
}

fn pipeline_only(enabled: bool) -> DagOptimizer {
    DagOptimizer {
        eliminate_dead_code: false,
        critical_path: false,
        cache_aware: false,
        memory_aware: false,
        priority_aware: false,
        stage_fusion: false,
        speculative_execution: false,
        pipeline_parallelism: enabled,
    }
}

async fn execute_observed(
    plan: CompiledPlan,
    ctx: ExecCtx,
    expected_events: usize,
) -> (
    blut::framework::executor::PlanResult,
    Vec<(&'static str, u32)>,
) {
    let mut events = ctx.status.subscribe();
    let execute = ParallelExecutor::execute(plan, ctx);
    let observe = async move {
        tokio::time::timeout(std::time::Duration::from_secs(5), async move {
            let mut order = Vec::new();
            while order.len() < expected_events {
                match events.recv().await.expect("status stream stays open") {
                    StageEvent::StageBegin { node_idx, .. } => order.push(("begin", node_idx)),
                    StageEvent::StageSkipped { node_idx, .. } => order.push(("skip", node_idx)),
                    _ => {}
                }
            }
            order
        })
        .await
        .expect("expected scheduling events arrive")
    };
    let (result, order) = tokio::join!(execute, observe);
    (result.expect("execute cache-order fixture"), order)
}

async fn run_cache_order_fixture(
    cache_aware: bool,
    bypass_cache: bool,
    priorities: (Option<i32>, Option<i32>),
    priority_aware: bool,
) -> (Vec<(&'static str, u32)>, usize, usize, ContentHash) {
    let temp = tempfile::tempdir().expect("cache-order tempdir");
    let cache_root = temp.path().join("shared-cache");

    let make_ctx = |job: &str| {
        let job_dir = temp.path().join(job);
        let mut ctx = ExecCtx::new(job_dir.clone()).with_max_in_flight(1);
        ctx.cache = Arc::new(
            CacheHandle::job_local(job_dir.join("_cache")).with_global(cache_root.clone()),
        );
        ctx
    };

    // Materialize only node 1's exact root-stage key through the real executor.
    ParallelExecutor::execute(compiled_nodes(&[("warm", None)]), make_ctx("prewarm"))
        .await
        .expect("prewarm cache fixture");
    EXECUTION_ORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();

    let mut ctx = make_ctx("measured").with_bypass_cache(bypass_cache);
    let mut optimizer = cache_only(cache_aware);
    optimizer.priority_aware = priority_aware;
    ctx.dag_optimizer = Some(optimizer);
    let (result, order) = execute_observed(
        compiled_nodes(&[("cold", priorities.0), ("warm", priorities.1)]),
        ctx,
        2,
    )
    .await;
    let output: OrderArtifact = result
        .final_output
        .expect("fixture has final output")
        .into_typed()
        .expect("decode final output");
    (
        order,
        result.n_cache_hits,
        result.n_cache_misses,
        output.content_hash,
    )
}

fn materialized_stage_dirs(job_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut stage_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(job_dir.join("stages"))
        .expect("read materialized stage dirs")
        .map(|entry| entry.expect("read stage-dir entry").path())
        .filter(|path| path.is_dir())
        .collect();
    stage_dirs.sort();
    stage_dirs
}

fn materialized_hashes(job_dir: &std::path::Path) -> Vec<ContentHash> {
    materialized_stage_dirs(job_dir)
        .into_iter()
        .map(|stage_dir| {
            let body = std::fs::read(stage_dir.join("output.metadata.json"))
                .expect("read materialized output metadata");
            serde_json::from_slice::<ArtifactMetadata>(&body)
                .expect("decode materialized output metadata")
                .content_id()
                .expect("A09+ materialization carries ContentId")
                .digest()
        })
        .collect()
}

fn materialized_logical_hashes(job_dir: &std::path::Path) -> Vec<ContentHash> {
    materialized_stage_dirs(job_dir)
        .into_iter()
        .map(|stage_dir| {
            let body = std::fs::read(stage_dir.join("output.metadata.json"))
                .expect("read materialized output metadata");
            serde_json::from_slice::<ArtifactMetadata>(&body)
                .expect("decode materialized output metadata")
                .logical_hash
                .expect("executor sidecar records logical identity")
        })
        .collect()
}

fn materialized_cache_keys(job_dir: &std::path::Path) -> Vec<InvocationKey> {
    materialized_stage_dirs(job_dir)
        .into_iter()
        .map(|stage_dir| {
            CacheProof::read_from(&stage_dir.join("cache-proof.json"))
                .expect("read materialized cache proof")
                .key
        })
        .collect()
}

fn drain_pipeline_signal(semaphore: &'static tokio::sync::Semaphore) {
    while let Ok(permit) = semaphore.try_acquire() {
        permit.forget();
    }
}

fn find_named_file(root: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        let entries = std::fs::read_dir(path).ok()?;
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().is_some_and(|candidate| candidate == name) {
                return Some(path);
            }
        }
    }
    None
}

fn reset_pipeline_fixture() {
    for semaphore in [
        &PIPE_PARENT_READY,
        &PIPE_PARENT_RELEASE,
        &PIPE_CHILD_STARTED,
        &PIPE_CHILD_RELEASE,
        &PIPE_SECOND_SUBMIT_STARTED,
        &PIPE_SECOND_EMIT_RETURNED,
        &PIPE_CAP_CHILD_ZERO_STARTED,
        &PIPE_CAP_CHILD_ZERO_RELEASE,
        &PIPE_CAP_CHILD_ONE_STARTED,
        &PIPE_CAP_CHILD_ONE_RELEASE,
        &PIPE_PRIVATE_CHILD_FINISHED,
        &PIPE_SIBLING_STARTED,
        &PIPE_SIBLING_RELEASE,
        &PIPE_STUCK_CHILD_STARTED,
        &PIPE_STUCK_CHILD_RELEASE,
    ] {
        drain_pipeline_signal(semaphore);
    }
    PIPE_CHILD_RUNS.store(0, Ordering::SeqCst);
    PIPE_CHILD_PROFILE_DECLARATIONS.store(0, Ordering::SeqCst);
    PIPE_MISMATCH_ACCEPTED.store(false, Ordering::SeqCst);
    PIPE_FAILURE_ACCEPTED.store(false, Ordering::SeqCst);
    PIPE_DEFAULT_OFF_PROFILE_INLINE.store(false, Ordering::SeqCst);
    PIPE_PUBLICATION_HASH_CALLS.store(0, Ordering::SeqCst);
    *PIPE_PUBLICATION_CANCEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

async fn take_test_pipeline_signal(semaphore: &'static tokio::sync::Semaphore) {
    semaphore
        .acquire()
        .await
        .expect("pipeline test semaphore remains open")
        .forget();
}

fn pipeline_ctx(job_dir: std::path::PathBuf, enabled: bool) -> ExecCtx {
    let mut ctx = ExecCtx::new(job_dir)
        .with_max_in_flight(4)
        .with_resource_limit(Resource::Cpu, 2)
        .with_memory_budget(4)
        .with_training_io_selection_budget_bytes(4 * blut::broker::footprint::GIB);
    ctx.dag_optimizer = Some(pipeline_only(enabled));
    ctx
}

fn canonical_pipeline_child_dirs(job_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let stages = job_dir.join("stages");
    let Ok(entries) = std::fs::read_dir(stages) else {
        return Vec::new();
    };
    let mut children = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.contains("pipeline_child"))
        })
        .collect::<Vec<_>>();
    children.sort();
    children
}

fn completed_local_cache_entries(job_dir: &std::path::Path) -> usize {
    std::fs::read_dir(job_dir.join("_cache/v1/cache-invocations"))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .count()
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_default_off_and_enabled_overlap_preserve_canonical_identity()
 {
    let _guard = TEST_LOCK.lock().await;
    assert!(
        !DagOptimizer::new().pipeline_parallelism,
        "pipeline parallelism must remain opt-in"
    );
    let temp = tempfile::tempdir().expect("pipeline overlap tempdir");

    reset_pipeline_fixture();
    let baseline_dir = temp.path().join("default-off");
    let baseline_task = tokio::spawn(ParallelExecutor::execute(
        pipeline_plan(PipelineCase::Overlap),
        pipeline_ctx(baseline_dir.clone(), false),
    ));
    take_test_pipeline_signal(&PIPE_PARENT_READY).await;
    assert!(
        PIPE_CHILD_STARTED.try_acquire().is_err(),
        "default-off map child cannot begin before its parent returns"
    );
    PIPE_PARENT_RELEASE.add_permits(1);
    take_test_pipeline_signal(&PIPE_CHILD_STARTED).await;
    PIPE_CHILD_RELEASE.add_permits(1);
    let baseline = baseline_task
        .await
        .expect("default-off task joins")
        .expect("default-off pipeline fixture succeeds");
    assert!(
        PIPE_DEFAULT_OFF_PROFILE_INLINE.load(Ordering::SeqCst),
        "optimizer-off execution must select an inline pipeline profile instead of billing an unusable lane"
    );
    let baseline_hashes = materialized_hashes(&baseline_dir);
    let baseline_keys = materialized_cache_keys(&baseline_dir);

    reset_pipeline_fixture();
    let enabled_dir = temp.path().join("enabled");
    let enabled_task = tokio::spawn(ParallelExecutor::execute(
        pipeline_plan(PipelineCase::Overlap),
        pipeline_ctx(enabled_dir.clone(), true),
    ));
    take_test_pipeline_signal(&PIPE_PARENT_READY).await;
    take_test_pipeline_signal(&PIPE_CHILD_STARTED).await;
    assert!(
        !enabled_task.is_finished(),
        "enabled child starts while the parent is still held"
    );
    PIPE_CHILD_RELEASE.add_permits(1);
    PIPE_PARENT_RELEASE.add_permits(1);
    let enabled = enabled_task
        .await
        .expect("enabled task joins")
        .expect("enabled pipeline fixture succeeds");

    assert_eq!((baseline.n_stages, enabled.n_stages), (2, 2));
    assert_eq!(materialized_hashes(&enabled_dir), baseline_hashes);
    assert_eq!(
        materialized_cache_keys(&enabled_dir),
        baseline_keys,
        "private overlap must preserve every ordinary node cache key"
    );
    assert_eq!(canonical_pipeline_child_dirs(&enabled_dir).len(), 1);
    assert!(!enabled_dir.join(".pipeline").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_warm_parent_never_opens_lane() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("warm parent tempdir");
    let job_dir = temp.path().join("warm-parent");
    let first = tokio::spawn(ParallelExecutor::execute(
        pipeline_plan(PipelineCase::Overlap),
        pipeline_ctx(job_dir.clone(), false),
    ));
    take_test_pipeline_signal(&PIPE_PARENT_READY).await;
    PIPE_PARENT_RELEASE.add_permits(1);
    take_test_pipeline_signal(&PIPE_CHILD_STARTED).await;
    PIPE_CHILD_RELEASE.add_permits(1);
    first
        .await
        .expect("cache priming task joins")
        .expect("cache priming succeeds");

    reset_pipeline_fixture();
    let warm = ParallelExecutor::execute(
        pipeline_plan(PipelineCase::Overlap),
        pipeline_ctx(job_dir, true),
    )
    .await
    .expect("warm pipeline plan succeeds");

    assert_eq!((warm.n_cache_hits, warm.n_cache_misses), (2, 0));
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 0);
    assert!(PIPE_PARENT_READY.try_acquire().is_err());
    assert!(PIPE_CHILD_STARTED.try_acquire().is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_warm_child_falls_back_to_one_cache_hit() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("warm child tempdir");
    let job_dir = temp.path().join("warm-child");
    let first = tokio::spawn(ParallelExecutor::execute(
        pipeline_plan(PipelineCase::Overlap),
        pipeline_ctx(job_dir.clone(), false),
    ));
    take_test_pipeline_signal(&PIPE_PARENT_READY).await;
    PIPE_PARENT_RELEASE.add_permits(1);
    take_test_pipeline_signal(&PIPE_CHILD_STARTED).await;
    PIPE_CHILD_RELEASE.add_permits(1);
    first
        .await
        .expect("cache priming task joins")
        .expect("cache priming succeeds");
    let keys = materialized_cache_keys(&job_dir);
    assert_eq!(keys.len(), 2);
    let parent_proof =
        CacheProof::read_from(&job_dir.join("stages/0-pipeline_parent/cache-proof.json"))
            .expect("read parent cache proof");
    assert_eq!(parent_proof.key, keys[0]);
    std::fs::remove_file(parent_proof.entry_path)
        .expect("remove only the parent invocation record");

    reset_pipeline_fixture();
    let second = tokio::spawn(ParallelExecutor::execute(
        pipeline_plan(PipelineCase::Overlap),
        pipeline_ctx(job_dir, true),
    ));
    take_test_pipeline_signal(&PIPE_PARENT_READY).await;
    assert!(
        PIPE_CHILD_STARTED.try_acquire().is_err(),
        "a warm child suppresses private overlap"
    );
    PIPE_PARENT_RELEASE.add_permits(1);
    let warm_child = second
        .await
        .expect("warm-child task joins")
        .expect("warm-child fallback succeeds");

    assert_eq!((warm_child.n_cache_hits, warm_child.n_cache_misses), (1, 1));
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_shared_cache_suppresses_overlap() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("shared cache tempdir");
    let global = temp.path().join("global-cache");
    let first_dir = temp.path().join("producer-job");
    let mut first_ctx = pipeline_ctx(first_dir.clone(), false);
    first_ctx.cache =
        Arc::new(CacheHandle::job_local(first_dir.join("_cache")).with_global(global.clone()));
    let first = tokio::spawn(ParallelExecutor::execute(
        pipeline_plan(PipelineCase::Overlap),
        first_ctx,
    ));
    take_test_pipeline_signal(&PIPE_PARENT_READY).await;
    PIPE_PARENT_RELEASE.add_permits(1);
    take_test_pipeline_signal(&PIPE_CHILD_STARTED).await;
    PIPE_CHILD_RELEASE.add_permits(1);
    first
        .await
        .expect("shared-cache priming task joins")
        .expect("shared-cache priming succeeds");

    reset_pipeline_fixture();
    let second_dir = temp.path().join("consumer-job");
    let mut second_ctx = pipeline_ctx(second_dir.clone(), true);
    second_ctx.cache =
        Arc::new(CacheHandle::job_local(second_dir.join("_cache")).with_global(global));
    let shared = ParallelExecutor::execute(pipeline_plan(PipelineCase::Overlap), second_ctx)
        .await
        .expect("shared-cache pipeline plan succeeds");

    assert_eq!((shared.n_cache_hits, shared.n_cache_misses), (2, 0));
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 0);
    assert!(PIPE_PARENT_READY.try_acquire().is_err());
    assert!(PIPE_CHILD_STARTED.try_acquire().is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_capacity_one_counts_running_and_queued_work() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline capacity tempdir");
    let job_dir = temp.path().join("capacity-one");
    let task = tokio::spawn(ParallelExecutor::execute(
        pipeline_plan(PipelineCase::Capacity),
        pipeline_ctx(job_dir.clone(), true),
    ));

    take_test_pipeline_signal(&PIPE_CAP_CHILD_ZERO_STARTED).await;
    take_test_pipeline_signal(&PIPE_SECOND_SUBMIT_STARTED).await;
    assert!(
        PIPE_SECOND_EMIT_RETURNED.try_acquire().is_err(),
        "capacity one must stay occupied while item zero is running"
    );
    PIPE_CAP_CHILD_ZERO_RELEASE.add_permits(1);
    take_test_pipeline_signal(&PIPE_SECOND_EMIT_RETURNED).await;
    take_test_pipeline_signal(&PIPE_CAP_CHILD_ONE_STARTED).await;
    PIPE_CAP_CHILD_ONE_RELEASE.add_permits(1);

    let result = task
        .await
        .expect("capacity task joins")
        .expect("capacity fixture succeeds");
    assert_eq!(result.n_stages, 3);
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 2);
    assert_eq!(canonical_pipeline_child_dirs(&job_dir).len(), 2);
    assert!(!job_dir.join(".pipeline").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_respects_max_in_flight_with_running_sibling() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline in-flight cap tempdir");
    let job_dir = temp.path().join("running-sibling");
    let mut ctx = pipeline_ctx(job_dir, true)
        .with_max_in_flight(2)
        .with_resource_limit(Resource::Cpu, 3);
    ctx.dag_optimizer = Some(pipeline_only(true));
    let task = tokio::spawn(ParallelExecutor::execute(
        pipeline_with_running_sibling_plan(),
        ctx,
    ));

    take_test_pipeline_signal(&PIPE_SIBLING_STARTED).await;
    take_test_pipeline_signal(&PIPE_PARENT_READY).await;
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            PIPE_CHILD_STARTED.acquire(),
        )
        .await
        .is_err(),
        "a parent plus private child must not exceed max_in_flight while a sibling runs"
    );

    PIPE_PARENT_RELEASE.add_permits(1);
    take_test_pipeline_signal(&PIPE_CHILD_STARTED).await;
    PIPE_CHILD_RELEASE.add_permits(1);
    PIPE_SIBLING_RELEASE.add_permits(1);
    let result = task
        .await
        .expect("running-sibling task joins")
        .expect("running-sibling fixture succeeds by ordinary fallback");
    assert_eq!(result.n_stages, 3);
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_late_admission_decline_reuses_resolved_child_profile() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline late-decline tempdir");
    let job_dir = temp.path().join("late-admission-decline");
    // The parent and child each need one CPU. A single permit admits either
    // ordinary stage sequentially but refuses the combined pipeline envelope
    // only after the child template's profile has been resolved.
    let ctx = pipeline_ctx(job_dir.clone(), true).with_resource_limit(Resource::Cpu, 1);
    let task = tokio::spawn(ParallelExecutor::execute(
        pipeline_plan(PipelineCase::Overlap),
        ctx,
    ));

    take_test_pipeline_signal(&PIPE_PARENT_READY).await;
    assert!(
        PIPE_CHILD_STARTED.try_acquire().is_err(),
        "combined admission decline must retain ordinary post-parent fan-out"
    );
    PIPE_PARENT_RELEASE.add_permits(1);
    take_test_pipeline_signal(&PIPE_CHILD_STARTED).await;
    PIPE_CHILD_RELEASE.add_permits(1);

    let result = task
        .await
        .expect("late-decline task joins")
        .expect("late admission decline falls back ordinarily");
    assert_eq!(result.n_stages, 2);
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 1);
    assert_eq!(canonical_pipeline_child_dirs(&job_dir).len(), 1);
    assert_eq!(
        PIPE_CHILD_PROFILE_DECLARATIONS.load(Ordering::SeqCst),
        1,
        "ordinary fallback must reuse the immutable profile resolved by the pipeline probe"
    );
    assert!(!job_dir.join(".pipeline").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_corrupt_spill_fallback_reuses_resolved_child_profile() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline corrupt-spill tempdir");
    let job_dir = temp.path().join("corrupt-spill");

    let result = ParallelExecutor::execute(
        pipeline_plan(PipelineCase::CorruptSpill),
        pipeline_ctx(job_dir.clone(), true),
    )
    .await
    .expect("corrupt optional spill falls back to ordinary fan-out");

    assert_eq!(result.n_stages, 2);
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 2);
    assert_eq!(canonical_pipeline_child_dirs(&job_dir).len(), 1);
    assert_eq!(completed_local_cache_entries(&job_dir), 2);
    assert_eq!(
        PIPE_CHILD_PROFILE_DECLARATIONS.load(Ordering::SeqCst),
        1,
        "spill corruption fallback must reuse the immutable profile resolved by the pipeline probe"
    );
    assert!(!job_dir.join(".pipeline").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_corrupt_input_spill_falls_back_without_unbounded_decode() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline corrupt-input tempdir");
    let job_dir = temp.path().join("corrupt-input");

    let result = ParallelExecutor::execute(
        pipeline_plan(PipelineCase::CorruptInputSpill),
        pipeline_ctx(job_dir.clone(), true),
    )
    .await
    .expect("corrupt optional input spill falls back to ordinary fan-out");

    assert_eq!(result.n_stages, 2);
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 2);
    assert_eq!(PIPE_CHILD_PROFILE_DECLARATIONS.load(Ordering::SeqCst), 1);
    assert_eq!(canonical_pipeline_child_dirs(&job_dir).len(), 1);
    assert_eq!(completed_local_cache_entries(&job_dir), 2);
    assert!(!job_dir.join(".pipeline").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_oversize_private_result_falls_back_within_admission() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline oversize-result tempdir");
    let job_dir = temp.path().join("oversize-private-result");

    let result = ParallelExecutor::execute(
        pipeline_plan(PipelineCase::OversizePrivateResult),
        pipeline_ctx(job_dir.clone(), true),
    )
    .await
    .expect("oversize optional private result falls back ordinarily");

    assert_eq!(result.n_stages, 2);
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 2);
    assert_eq!(PIPE_CHILD_PROFILE_DECLARATIONS.load(Ordering::SeqCst), 1);
    assert_eq!(canonical_pipeline_child_dirs(&job_dir).len(), 1);
    assert_eq!(completed_local_cache_entries(&job_dir), 2);
    assert!(!job_dir.join(".pipeline").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_manifest_item_mismatch_falls_back_without_duplicate_publication()
 {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline mismatch tempdir");
    let baseline_dir = temp.path().join("mismatch-default-off");
    let baseline = ParallelExecutor::execute(
        pipeline_plan(PipelineCase::ManifestMismatch),
        pipeline_ctx(baseline_dir.clone(), false),
    )
    .await
    .expect("default-off mismatch fixture succeeds");
    let baseline_hashes = materialized_hashes(&baseline_dir);
    let baseline_keys = materialized_cache_keys(&baseline_dir);

    reset_pipeline_fixture();
    let job_dir = temp.path().join("mismatch");
    let result = ParallelExecutor::execute(
        pipeline_plan(PipelineCase::ManifestMismatch),
        pipeline_ctx(job_dir.clone(), true),
    )
    .await
    .expect("manifest mismatch falls back to ordinary fan-out");

    assert!(
        PIPE_MISMATCH_ACCEPTED.load(Ordering::SeqCst),
        "the mismatched item must reach the certified lane before validation rejects it"
    );
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 1);
    assert_eq!((baseline.n_stages, result.n_stages), (2, 2));
    assert_eq!(canonical_pipeline_child_dirs(&job_dir).len(), 1);
    assert_eq!(materialized_hashes(&job_dir), baseline_hashes);
    assert_eq!(
        materialized_cache_keys(&job_dir),
        baseline_keys,
        "ordinary fallback must recompute child identity from the authoritative parent hash"
    );
    assert_eq!(completed_local_cache_entries(&job_dir), 2);
    assert!(!job_dir.join(".pipeline").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_manifest_cardinality_mismatch_uses_authoritative_fanout() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("pipeline cardinality tempdir");

    for case in [
        PipelineCase::ManifestTooShort,
        PipelineCase::ManifestTooLong,
    ] {
        reset_pipeline_fixture();
        let baseline_dir = temp.path().join(format!("{case:?}-default-off"));
        let baseline = ParallelExecutor::execute(
            pipeline_plan(case),
            pipeline_ctx(baseline_dir.clone(), false),
        )
        .await
        .expect("default-off cardinality fixture succeeds");
        let baseline_hashes = materialized_hashes(&baseline_dir);
        let baseline_keys = materialized_cache_keys(&baseline_dir);
        let baseline_declarations = PIPE_CHILD_PROFILE_DECLARATIONS.load(Ordering::SeqCst);

        reset_pipeline_fixture();
        let enabled_dir = temp.path().join(format!("{case:?}-enabled"));
        let enabled =
            ParallelExecutor::execute(pipeline_plan(case), pipeline_ctx(enabled_dir.clone(), true))
                .await
                .expect("invalid manifest cardinality falls back ordinarily");

        assert_eq!(enabled.n_stages, baseline.n_stages, "case {case:?}");
        assert_eq!(
            materialized_hashes(&enabled_dir),
            baseline_hashes,
            "case {case:?} artifact identity"
        );
        assert_eq!(
            materialized_cache_keys(&enabled_dir),
            baseline_keys,
            "case {case:?} cache identity"
        );
        assert_eq!(
            baseline_declarations,
            pipeline_items(case).len(),
            "ordinary fan-out resolves each actual child once for {case:?}"
        );
        assert_eq!(
            PIPE_CHILD_PROFILE_DECLARATIONS.load(Ordering::SeqCst),
            1,
            "pipeline resolves the immutable one-node template once and reuses it across manifest reconciliation for {case:?}"
        );
        assert!(!enabled_dir.join(".pipeline").exists());
    }
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_parent_failure_after_emission_leaves_no_canonical_child_state()
 {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline parent failure tempdir");
    let job_dir = temp.path().join("parent-failure");
    let error = ParallelExecutor::execute(
        pipeline_plan(PipelineCase::FailAfterEmission),
        pipeline_ctx(job_dir.clone(), true),
    )
    .await
    .expect_err("parent fails after its accepted private emission");

    assert!(
        error
            .to_string()
            .contains("simulated parent failure after pipeline emission")
    );
    assert!(PIPE_FAILURE_ACCEPTED.load(Ordering::SeqCst));
    assert_eq!(PIPE_CHILD_RUNS.load(Ordering::SeqCst), 1);
    assert!(canonical_pipeline_child_dirs(&job_dir).is_empty());
    assert_eq!(completed_local_cache_entries(&job_dir), 0);
    assert!(!job_dir.join(".pipeline").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_parent_failure_aborts_noncooperative_private_child() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline blocked-child tempdir");
    let job_dir = temp.path().join("blocked-child");
    let result = tokio::time::timeout(
        std::time::Duration::from_millis(500),
        ParallelExecutor::execute(
            pipeline_plan(PipelineCase::FailWhileChildBlocked),
            pipeline_ctx(job_dir.clone(), true),
        ),
    )
    .await
    .expect("parent failure must not await a noncooperative private child forever");
    let error = result.expect_err("blocked-child parent must fail");

    assert!(
        error
            .to_string()
            .contains("simulated parent failure while private child is blocked")
    );
    assert!(canonical_pipeline_child_dirs(&job_dir).is_empty());
    assert_eq!(completed_local_cache_entries(&job_dir), 0);
    assert!(!job_dir.join(".pipeline").exists());
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_pipeline_cancel_during_publication_rolls_back_private_child() {
    let _guard = TEST_LOCK.lock().await;
    reset_pipeline_fixture();
    let temp = tempfile::tempdir().expect("pipeline publication-cancel tempdir");
    let job_dir = temp.path().join("publication-cancel");
    let ctx = pipeline_ctx(job_dir.clone(), true);
    let mut events = ctx.status.subscribe();
    *PIPE_PUBLICATION_CANCEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(ctx.cancel.clone());

    let error =
        ParallelExecutor::execute(pipeline_plan(PipelineCase::CancelDuringPublication), ctx)
            .await
            .expect_err("publication-time cancellation must fail the plan");

    assert!(matches!(error, PlanError::Cancelled));
    assert_eq!(
        PIPE_PUBLICATION_HASH_CALLS.load(Ordering::SeqCst),
        2,
        "fixture must cancel at the selected-publication content-hash boundary"
    );
    assert!(canonical_pipeline_child_dirs(&job_dir).is_empty());
    assert_eq!(
        completed_local_cache_entries(&job_dir),
        0,
        "publication cancellation must not leave a selected private cache entry"
    );
    let mut child_begin = false;
    while let Ok(event) = events.try_recv() {
        child_begin |= matches!(
            event,
            StageEvent::StageBegin { stage_name, .. } if stage_name == "pipeline_child"
        );
    }
    assert!(
        !child_begin,
        "private child lifecycle must remain unpublished"
    );
    assert!(!job_dir.join(".pipeline").exists());
    *PIPE_PUBLICATION_CANCEL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_discards_private_speculation_without_identity_drift() {
    let _guard = TEST_LOCK.lock().await;
    SPEC_GATE_TARGET_RUNS.store(0, Ordering::SeqCst);

    let optimizer = DagOptimizer {
        speculative_execution: true,
        pipeline_parallelism: false,
        ..DagOptimizer::new()
    };
    let (default_off, _) = DagOptimizer::new().optimize(speculation_discard_plan());
    assert!(
        default_off.speculation_candidates().next().is_none(),
        "the named gate must prove speculation remains default-off"
    );
    let (witnessed, _) = optimizer.optimize(speculation_discard_plan());
    assert_eq!(
        witnessed.speculation_candidates().collect::<Vec<_>>(),
        vec![2],
        "only the optimizer may authorize the certified conditional target"
    );

    let temp = tempfile::tempdir().expect("speculation discard gate tempdir");
    let baseline_dir = temp.path().join("baseline");
    let baseline = tokio::spawn(ParallelExecutor::execute(
        speculation_discard_plan(),
        ExecCtx::new(baseline_dir.clone()).with_max_in_flight(3),
    ));
    SPEC_GATE_DECISION_STARTED
        .acquire()
        .await
        .expect("baseline decision started")
        .forget();
    SPEC_GATE_DECISION_RELEASE.add_permits(1);
    let baseline = baseline
        .await
        .expect("baseline discard task joins")
        .expect("baseline discard plan succeeds");
    assert_eq!(SPEC_GATE_TARGET_RUNS.load(Ordering::SeqCst), 0);

    let speculative_dir = temp.path().join("speculative");
    let mut speculative_ctx = ExecCtx::new(speculative_dir.clone()).with_max_in_flight(3);
    speculative_ctx.dag_optimizer = Some(optimizer);
    let speculative = tokio::spawn(ParallelExecutor::execute(
        speculation_discard_plan(),
        speculative_ctx,
    ));
    SPEC_GATE_DECISION_STARTED
        .acquire()
        .await
        .expect("speculative decision started")
        .forget();
    tokio::time::timeout(
        std::time::Duration::from_secs(2),
        SPEC_GATE_TARGET_FINISHED.acquire(),
    )
    .await
    .expect("private target must finish before the false selector is released")
    .expect("private target signal open")
    .forget();
    SPEC_GATE_DECISION_RELEASE.add_permits(1);
    let speculative = speculative
        .await
        .expect("speculative discard task joins")
        .expect("speculative discard plan succeeds");

    assert_eq!(SPEC_GATE_TARGET_RUNS.load(Ordering::SeqCst), 1);
    assert!(baseline.final_output.is_none());
    assert!(speculative.final_output.is_none());
    assert_eq!(
        materialized_hashes(&baseline_dir),
        materialized_hashes(&speculative_dir),
        "discarded optional work must not change any canonical output hash"
    );
    assert_eq!(
        materialized_cache_keys(&baseline_dir),
        materialized_cache_keys(&speculative_dir),
        "discarded optional work must not change any canonical cache identity"
    );
    assert!(
        !speculative_dir
            .join("stages/2-record_speculative_after")
            .exists(),
        "discarded target must not materialize a canonical stage directory"
    );
    assert!(
        !speculative_dir.join(".speculation").exists(),
        "discarded target scratch must be deleted"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_fuses_linear_chain_without_changing_artifact_identity() {
    let _guard = TEST_LOCK.lock().await;
    assert!(
        !DagOptimizer::new().stage_fusion,
        "stage fusion must remain opt-in"
    );
    let temp = tempfile::tempdir().expect("fusion tempdir");
    let plan = || {
        compiled_graph(
            &[
                ("record_order", "fuse-root", None),
                ("record_after", "fuse-middle", None),
                ("record_after", "fuse-terminal", None),
            ],
            &[(0, 1), (1, 2)],
        )
    };

    let run = |name: &str, enabled: bool| {
        let job_dir = temp.path().join(name);
        let mut ctx = ExecCtx::new(job_dir.clone()).with_max_in_flight(1);
        ctx.dag_optimizer = Some(fusion_only(enabled));
        async move {
            FUSION_TASK_IDS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
            let result = ParallelExecutor::execute(plan(), ctx)
                .await
                .expect("execute fusion equivalence fixture");
            let task_ids = FUSION_TASK_IDS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            (
                result,
                materialized_hashes(&job_dir),
                materialized_cache_keys(&job_dir),
                task_ids,
            )
        }
    };

    let (unfused, unfused_hashes, unfused_keys, unfused_tasks) = run("unfused", false).await;
    let (fused, fused_hashes, fused_keys, fused_tasks) = run("fused", true).await;
    let unfused_output: OrderArtifact = unfused
        .final_output
        .expect("unfused terminal output")
        .into_typed()
        .expect("decode unfused output");
    let fused_output: OrderArtifact = fused
        .final_output
        .expect("fused terminal output")
        .into_typed()
        .expect("decode fused output");

    assert_eq!(fused.n_stages, 3);
    assert_eq!((fused.n_cache_hits, fused.n_cache_misses), (0, 3));
    assert_eq!(
        fused_hashes, unfused_hashes,
        "fusion must preserve every stage's ordinary artifact identity"
    );
    assert_eq!(fused_output.content_hash, unfused_output.content_hash);
    assert_eq!(
        fused_keys, unfused_keys,
        "fusion must reuse every ordinary node key, including the terminal key"
    );
    assert_eq!(
        fused_tasks
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        1,
        "the eligible chain must execute inside one executor task"
    );
    assert_eq!(
        unfused_tasks
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3,
        "the default-off path must retain one spawned task per stage"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_fuses_internal_linear_subchain_in_branched_plan() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("internal fusion tempdir");
    let plan = || {
        compiled_graph(
            &[
                ("record_order", "fuse-subchain-root", None),
                ("record_after", "fuse-subchain-left-a", None),
                ("record_after", "fuse-subchain-left-b", None),
                ("record_after", "fuse-subchain-right", None),
            ],
            &[(0, 1), (1, 2), (0, 3)],
        )
    };

    let run = |name: &str, enabled: bool| {
        let job_dir = temp.path().join(name);
        let mut ctx = ExecCtx::new(job_dir.clone()).with_max_in_flight(1);
        ctx.dag_optimizer = Some(fusion_only(enabled));
        async move {
            FUSION_TASK_IDS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
            let result = ParallelExecutor::execute(plan(), ctx)
                .await
                .expect("execute internal fusion fixture");
            let task_ids = FUSION_TASK_IDS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            (
                result,
                materialized_hashes(&job_dir),
                materialized_cache_keys(&job_dir),
                task_ids,
            )
        }
    };

    let (unfused, unfused_hashes, unfused_keys, unfused_tasks) =
        run("unfused-subchain", false).await;
    let (fused, fused_hashes, fused_keys, fused_tasks) = run("fused-subchain", true).await;

    assert_eq!(fused.n_stages, 4);
    assert_eq!((fused.n_cache_hits, fused.n_cache_misses), (0, 4));
    assert_eq!(fused_hashes, unfused_hashes);
    assert_eq!(fused_keys, unfused_keys);
    let fused_output: OrderArtifact = fused
        .final_output
        .expect("fused terminal output")
        .into_typed()
        .expect("decode fused terminal output");
    let unfused_output: OrderArtifact = unfused
        .final_output
        .expect("unfused terminal output")
        .into_typed()
        .expect("decode unfused terminal output");
    assert_eq!(fused_output.content_hash, unfused_output.content_hash);
    assert_eq!(unfused_tasks.len(), 4);
    assert_eq!(fused_tasks.len(), 4);
    assert_ne!(unfused_tasks[1], unfused_tasks[2]);
    assert_eq!(
        fused_tasks[1], fused_tasks[2],
        "eligible internal chain must run inside one executor task"
    );
    assert_ne!(fused_tasks[0], fused_tasks[1]);
    assert_ne!(fused_tasks[2], fused_tasks[3]);
}

#[test]
fn dag_opt_advanced_gate_emits_internal_fusion_plan_witness() {
    let plan = || {
        compiled_graph(
            &[
                ("record_order", "fuse-witness-root", None),
                ("record_after", "fuse-witness-left-a", None),
                ("record_after", "fuse-witness-left-b", None),
                ("record_after", "fuse-witness-right", None),
            ],
            &[(0, 1), (1, 2), (0, 3)],
        )
    };

    let (optimized, _) = fusion_only(true).optimize(plan());
    assert_eq!(
        optimized.fused_subchains().collect::<Vec<_>>(),
        vec![&[1, 2][..]],
        "the stage-fusion pass must represent the internal chain on the optimized plan"
    );

    let (default_off, _) = fusion_only(false).optimize(plan());
    assert!(
        default_off.fused_subchains().next().is_none(),
        "the default-off plan must carry no fused execution groups"
    );
}

#[test]
fn dag_opt_advanced_gate_preserves_default_opaque_stage_boundary() {
    let (optimized, _) = fusion_only(true).optimize(compiled_graph(
        &[
            ("record_order", "opaque-root", None),
            ("record_after", "opaque-left", None),
            ("record_opaque_boundary", "opaque-middle", None),
            ("record_after", "opaque-right-a", None),
            ("record_after", "opaque-right-b", None),
        ],
        &[(0, 1), (1, 2), (2, 3), (3, 4)],
    ));

    assert_eq!(
        optimized.fused_subchains().collect::<Vec<_>>(),
        vec![&[0, 1][..], &[3, 4][..]],
        "an unclassified typed stage must split otherwise eligible fusion witnesses"
    );
}

#[test]
fn dag_opt_advanced_gate_preserves_declared_subprocess_boundary() {
    let (optimized, _) = fusion_only(true).optimize(compiled_graph(
        &[
            ("record_order", "subprocess-root", None),
            ("record_after", "subprocess-left", None),
            ("record_subprocess_boundary", "subprocess-middle", None),
            ("record_after", "subprocess-right-a", None),
            ("record_after", "subprocess-right-b", None),
        ],
        &[(0, 1), (1, 2), (2, 3), (3, 4)],
    ));

    assert_eq!(
        optimized.fused_subchains().collect::<Vec<_>>(),
        vec![&[0, 1][..], &[3, 4][..]],
        "a declared subprocess stage must split otherwise eligible fusion witnesses"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_dce_dense_renumbers_middle_hole_before_fusion() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("DCE dense-id tempdir");
    let mut optimizer = fusion_only(true);
    optimizer.eliminate_dead_code = true;
    let mut ctx = ExecCtx::new(temp.path().join("job")).with_max_in_flight(1);
    ctx.dag_optimizer = Some(optimizer);

    let result = ParallelExecutor::execute(
        compiled_graph(
            &[
                ("record_order", "fuse-dce-root", None),
                ("record_order", "fuse-dce-disconnected", None),
                ("record_after", "fuse-dce-terminal", None),
            ],
            &[(0, 2)],
        ),
        ctx,
    )
    .await
    .expect("post-DCE dense plan must execute");

    assert_eq!(result.n_stages, 2);
    let output: OrderArtifact = result
        .final_output
        .expect("DCE terminal output")
        .into_typed()
        .expect("decode DCE terminal output");
    assert_eq!(
        output.content_hash,
        ContentHash::of_bytes(b"fuse-dce-terminal")
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_stops_at_internal_fusion_deadline_boundary() {
    let _guard = TEST_LOCK.lock().await;
    EXECUTION_ORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    let temp = tempfile::tempdir().expect("internal fusion deadline tempdir");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    *FUSION_BOUNDARY_DEADLINE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(deadline);

    let mut ctx = ExecCtx::new(temp.path().join("job")).with_max_in_flight(1);
    ctx.deadline = Some(deadline);
    ctx.dag_optimizer = Some(fusion_only(true));
    let result = ParallelExecutor::execute(
        compiled_graph(
            &[
                ("record_order", "fuse-deadline-root", None),
                ("record_slow_after", "fuse-deadline-slow", None),
                ("record_after", "fuse-deadline-tail", None),
                ("record_after", "fuse-deadline-sibling", None),
            ],
            &[(0, 1), (1, 2), (0, 3)],
        ),
        ctx,
    )
    .await;
    *FUSION_BOUNDARY_DEADLINE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;

    assert!(matches!(result, Err(PlanError::DeadlineExceeded { .. })));
    let order = EXECUTION_ORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    assert!(
        order.iter().any(|label| label == "fuse-deadline-slow"),
        "the deadline fixture must expire while the first internal fused stage is running: {order:?}"
    );
    assert!(
        !order.iter().any(|label| label == "fuse-deadline-tail"),
        "a fused group must re-check the plan deadline before its next stage: {order:?}"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_internal_fusion_preserves_cache_single_flight() {
    let _guard = TEST_LOCK.lock().await;
    FUSION_DUPLICATE_RUNS.store(0, Ordering::SeqCst);
    let temp = tempfile::tempdir().expect("internal fusion single-flight tempdir");
    let mut ctx = ExecCtx::new(temp.path().join("job")).with_max_in_flight(2);
    ctx.dag_optimizer = Some(fusion_only(true));

    let result = ParallelExecutor::execute(
        compiled_graph(
            &[
                ("direct_root", "fuse-single-flight-root", None),
                ("direct_identity", "fuse-single-flight-identity", None),
                ("direct_counted_slow", "fuse-single-flight-shared", None),
                ("direct_counted_slow", "fuse-single-flight-shared", None),
            ],
            &[(0, 1), (1, 2), (0, 3)],
        ),
        ctx,
    )
    .await
    .expect("execute internal fusion single-flight fixture");

    assert_eq!(result.n_stages, 4);
    assert_eq!(
        FUSION_DUPLICATE_RUNS.load(Ordering::SeqCst),
        1,
        "a fused tail and ordinary sibling with the same exact cache key must not both execute"
    );
    assert_eq!((result.n_cache_hits, result.n_cache_misses), (1, 3));
}

#[tokio::test]
async fn dag_opt_advanced_gate_fused_chain_uses_direct_typed_handoffs() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("direct fusion tempdir");
    let plan = || {
        compiled_graph(
            &[
                ("direct_root", "direct-root", None),
                ("direct_after", "direct-middle", None),
                ("direct_after", "direct-terminal", None),
            ],
            &[(0, 1), (1, 2)],
        )
    };
    let run = |name: &str, enabled: bool| {
        let mut ctx = ExecCtx::new(temp.path().join(name)).with_max_in_flight(1);
        ctx.dag_optimizer = Some(fusion_only(enabled));
        async move {
            DIRECT_ARTIFACT_BINARY_DESERIALIZES.store(0, Ordering::SeqCst);
            let result = ParallelExecutor::execute(plan(), ctx)
                .await
                .expect("execute direct fusion fixture");
            let decodes = DIRECT_ARTIFACT_BINARY_DESERIALIZES.load(Ordering::SeqCst);
            (
                result.final_output.expect("terminal direct artifact"),
                decodes,
            )
        }
    };

    let (unfused, unfused_decodes) = run("unfused", false).await;
    let (fused, fused_decodes) = run("fused", true).await;
    assert!(
        unfused_decodes > 0,
        "the ordinary StageDyn boundary must exercise the bincode-decode witness"
    );
    assert!(
        fused_decodes < unfused_decodes,
        "fused miss chain must remove inter-stage bincode decodes even though portable-store validation still decodes typed metadata: fused={fused_decodes}, unfused={unfused_decodes}"
    );
    assert_eq!(fused.kind, unfused.kind);
    assert_eq!(fused.schema, unfused.schema);
    assert_eq!(fused.payload, unfused.payload);
}

#[tokio::test]
async fn dag_opt_advanced_gate_fused_chain_holds_one_admission() {
    let _guard = TEST_LOCK.lock().await;
    EXECUTION_ORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    FUSION_WAITER_ACQUIRED.store(false, Ordering::SeqCst);

    let temp = tempfile::tempdir().expect("fusion admission tempdir");
    let mut ctx = ExecCtx::new(temp.path().join("job"))
        .with_resource_limit(Resource::Cpu, 1)
        .with_max_in_flight(1);
    let cpu = ctx.resources[&Resource::Cpu].clone();
    ctx.dag_optimizer = Some(fusion_only(true));
    let execution = tokio::spawn(ParallelExecutor::execute(
        compiled_graph(
            &[
                ("record_order", "fuse-admission-root", None),
                ("record_after", "fuse-admission-middle", None),
                ("record_after", "fuse-admission-terminal", None),
            ],
            &[(0, 1), (1, 2)],
        ),
        ctx,
    ));

    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        FUSION_ROOT_STARTED.notified(),
    )
    .await
    .expect("root starts while holding CPU admission");
    let waiter = tokio::spawn(async move {
        FUSION_WAITER_STARTED.notify_one();
        let permit = cpu.acquire_owned().await.expect("CPU semaphore stays open");
        FUSION_WAITER_ACQUIRED.store(true, Ordering::SeqCst);
        FUSION_RELEASE_WAITER.notified().await;
        drop(permit);
    });
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        FUSION_WAITER_STARTED.notified(),
    )
    .await
    .expect("external waiter reaches the CPU acquire");
    tokio::task::yield_now().await;
    FUSION_RELEASE_ROOT.notify_one();

    let held_continuously = tokio::time::timeout(std::time::Duration::from_secs(1), async {
        loop {
            if EXECUTION_ORDER
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .any(|label| label == "fuse-admission-middle")
            {
                break true;
            }
            if FUSION_WAITER_ACQUIRED.load(Ordering::SeqCst) {
                break false;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("either the fused middle or the external waiter makes progress");

    FUSION_RELEASE_WAITER.notify_one();
    execution
        .await
        .expect("fusion execution task joins")
        .expect("fusion admission fixture executes");
    waiter.await.expect("external CPU waiter joins");
    assert!(
        held_continuously,
        "a fused chain must not release and reacquire its broker admission between stages"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_splits_heterogeneous_envelopes_into_equal_groups() {
    let _guard = TEST_LOCK.lock().await;
    FUSION_TASK_IDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    let temp = tempfile::tempdir().expect("heterogeneous fusion tempdir");
    let mut ctx = ExecCtx::new(temp.path().join("job")).with_max_in_flight(1);
    ctx.dag_optimizer = Some(fusion_only(true));

    ParallelExecutor::execute(
        compiled_graph(
            &[
                ("record_order", "fuse-hetero-cpu-root", None),
                ("record_after", "fuse-hetero-cpu-a", None),
                ("record_network_after", "fuse-hetero-net-a", None),
                ("record_network_after", "fuse-hetero-net-b", None),
                ("record_after", "fuse-hetero-cpu-tail", None),
            ],
            &[(0, 1), (1, 2), (2, 3), (3, 4)],
        ),
        ctx,
    )
    .await
    .expect("execute heterogeneous fusion fixture");

    let tasks = FUSION_TASK_IDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    assert_eq!(tasks.len(), 5);
    assert_eq!(tasks[0], tasks[1], "equal CPU envelopes should fuse");
    assert_eq!(tasks[2], tasks[3], "equal network envelopes should fuse");
    assert_ne!(tasks[1], tasks[2], "an envelope change must split groups");
    assert_ne!(
        tasks[3], tasks[4],
        "a trailing one-node envelope segment must retain its ordinary boundary"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_fused_cache_hits_need_no_admission() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("fusion cache tempdir");
    let cache_root = temp.path().join("shared-cache");
    let plan = || {
        compiled_graph(
            &[
                ("record_order", "fuse-cache-root", None),
                ("record_after", "fuse-cache-middle", None),
                ("record_after", "fuse-cache-terminal", None),
            ],
            &[(0, 1), (1, 2)],
        )
    };
    let make_cache = |job_dir: &std::path::Path| {
        Arc::new(CacheHandle::job_local(job_dir.join("_cache")).with_global(cache_root.clone()))
    };

    let prewarm_dir = temp.path().join("prewarm");
    let mut prewarm = ExecCtx::new(prewarm_dir.clone());
    prewarm.cache = make_cache(&prewarm_dir);
    ParallelExecutor::execute(plan(), prewarm)
        .await
        .expect("prewarm fused cache fixture");

    let measured_dir = temp.path().join("measured");
    let mut measured = ExecCtx::new(measured_dir.clone())
        .with_resource_limit(Resource::Cpu, 0)
        .with_max_in_flight(1);
    measured.cache = make_cache(&measured_dir);
    measured.dag_optimizer = Some(fusion_only(true));
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        ParallelExecutor::execute(plan(), measured),
    )
    .await
    .expect("a fully warm fused chain must not wait for an unavailable CPU permit")
    .expect("execute fully warm fused chain");
    assert_eq!((result.n_cache_hits, result.n_cache_misses), (3, 0));
}

#[tokio::test]
async fn dag_opt_advanced_gate_fusion_falls_back_for_branches_and_nondeterministic_nodes() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("fusion fallback tempdir");

    let run = |name: &str, plan: CompiledPlan, optimizer: DagOptimizer| {
        let job_dir = temp.path().join(name);
        let mut ctx = ExecCtx::new(job_dir).with_max_in_flight(1);
        ctx.dag_optimizer = Some(optimizer);
        async move {
            FUSION_TASK_IDS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
            let result = ParallelExecutor::execute(plan, ctx)
                .await
                .expect("execute fusion fallback fixture");
            let unique_tasks = FUSION_TASK_IDS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .len();
            (result, unique_tasks)
        }
    };

    let (branched, branched_tasks) = run(
        "branched",
        compiled_graph(
            &[
                ("record_order", "fuse-branch-root", None),
                ("record_after", "fuse-branch-left", None),
                ("record_after", "fuse-branch-right", None),
            ],
            &[(0, 1), (0, 2)],
        ),
        fusion_only(true),
    )
    .await;
    assert_eq!(branched.n_stages, 3);
    assert_eq!(
        branched_tasks, 3,
        "an externally consumed fork must not fuse"
    );

    let (nondeterministic, nondeterministic_tasks) = run(
        "nondeterministic",
        compiled_graph(
            &[
                ("record_order", "fuse-nondeterministic-root", None),
                (
                    "record_nondeterministic",
                    "fuse-nondeterministic-middle",
                    None,
                ),
                ("record_after", "fuse-nondeterministic-terminal", None),
            ],
            &[(0, 1), (1, 2)],
        ),
        fusion_only(true),
    )
    .await;
    assert_eq!(nondeterministic.n_stages, 3);
    assert_eq!(
        nondeterministic_tasks, 3,
        "a non-deterministic node must keep ordinary executor boundaries"
    );

    let (mixed_admission, mixed_admission_tasks) = run(
        "mixed-admission",
        compiled_graph(
            &[
                ("record_order", "fuse-mixed-root", None),
                ("record_network_after", "fuse-mixed-middle", None),
                ("record_after", "fuse-mixed-terminal", None),
            ],
            &[(0, 1), (1, 2)],
        ),
        fusion_only(true),
    )
    .await;
    assert_eq!(mixed_admission.n_stages, 3);
    assert_eq!(
        mixed_admission_tasks, 3,
        "a mixed resource envelope must fall back instead of reserving a later stage's resources early"
    );

    let mut cache_aware_fusion = fusion_only(true);
    cache_aware_fusion.cache_aware = true;
    let (cache_aware, cache_aware_tasks) = run(
        "cache-aware",
        compiled_graph(
            &[
                ("record_order", "fuse-cache-aware-root", None),
                ("record_after", "fuse-cache-aware-middle", None),
                ("record_after", "fuse-cache-aware-terminal", None),
            ],
            &[(0, 1), (1, 2)],
        ),
        cache_aware_fusion,
    )
    .await;
    assert_eq!(cache_aware.n_stages, 3);
    assert_eq!(
        cache_aware_tasks, 3,
        "cache-aware mode must retain its deadline-aware probe path"
    );

    let mut priority_fusion = fusion_only(true);
    priority_fusion.priority_aware = true;
    let (priority_aware, priority_aware_tasks) = run(
        "priority-aware",
        compiled_graph(
            &[
                ("record_order", "fuse-priority-root", Some(0)),
                ("record_after", "fuse-priority-middle", Some(-10)),
                ("record_after", "fuse-priority-terminal", Some(-20)),
            ],
            &[(0, 1), (1, 2)],
        ),
        priority_fusion,
    )
    .await;
    assert_eq!(priority_aware.n_stages, 3);
    assert_eq!(
        priority_aware_tasks, 3,
        "explicit per-node priority must retain coordinator scheduling boundaries"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_fused_panic_returns_plan_error() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("fusion panic tempdir");
    let mut ctx = ExecCtx::new(temp.path().join("job")).with_max_in_flight(1);
    ctx.dag_optimizer = Some(fusion_only(true));
    let caught = std::panic::AssertUnwindSafe(ParallelExecutor::execute(
        compiled_graph(
            &[
                ("record_order", "fuse-panic-root", None),
                ("record_panicking", "fuse-panic-middle", None),
                ("record_after", "fuse-panic-terminal", None),
            ],
            &[(0, 1), (1, 2)],
        ),
        ctx,
    ))
    .catch_unwind()
    .await;
    let result = caught.expect("a fused stage panic must not unwind the executor caller");
    match result {
        Err(PlanError::Other(message)) => assert!(
            message.contains("panicked"),
            "panic must retain the established plan-level classification: {message}"
        ),
        other => panic!("expected PlanError::Other for fused panic, got {other:?}"),
    }
}

#[tokio::test]
async fn dag_opt_advanced_gate_fused_failure_matches_cancellation_semantics() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("fusion failure tempdir");
    let run = |name: &str, enabled: bool| {
        let mut ctx = ExecCtx::new(temp.path().join(name)).with_max_in_flight(1);
        ctx.dag_optimizer = Some(fusion_only(enabled));
        let cancel = ctx.cancel.clone();
        async move {
            let result = ParallelExecutor::execute(
                compiled_graph(
                    &[
                        ("record_order", "fuse-failure-root", None),
                        ("record_failing", "fuse-failure-middle", None),
                        ("record_after", "fuse-failure-terminal", None),
                    ],
                    &[(0, 1), (1, 2)],
                ),
                ctx,
            )
            .await;
            (result, cancel.is_cancelled())
        }
    };

    let (unfused, unfused_cancelled) = run("unfused", false).await;
    let (fused, fused_cancelled) = run("fused", true).await;
    assert!(
        matches!(unfused, Err(PlanError::StageFailed { ref stage, .. }) if stage == "record_failing")
    );
    assert!(
        matches!(fused, Err(PlanError::StageFailed { ref stage, .. }) if stage == "record_failing")
    );
    assert!(unfused_cancelled, "ordinary failure cancels the plan token");
    assert!(
        fused_cancelled,
        "fused failure must cancel the same caller-visible plan token"
    );
}

async fn run_downstream_cache_fixture(
    cache_aware: bool,
) -> (
    Vec<(&'static str, u32)>,
    usize,
    usize,
    Vec<ContentHash>,
    Vec<ContentHash>,
) {
    let temp = tempfile::tempdir().expect("downstream-cache tempdir");
    let cache_root = temp.path().join("shared-cache");
    let make_ctx = |job: &str| {
        let job_dir = temp.path().join(job);
        let mut ctx = ExecCtx::new(job_dir.clone()).with_max_in_flight(1);
        ctx.cache = Arc::new(
            CacheHandle::job_local(job_dir.join("_cache")).with_global(cache_root.clone()),
        );
        ctx
    };

    ParallelExecutor::execute(
        compiled_graph(
            &[
                ("record_order", "warm-seed", None),
                ("record_after", "warm-downstream", None),
            ],
            &[(0, 1)],
        ),
        make_ctx("prewarm"),
    )
    .await
    .expect("prewarm input-dependent downstream key");

    let measured_dir = temp.path().join("measured");
    let mut ctx = make_ctx("measured");
    ctx.dag_optimizer = Some(cache_only(cache_aware));
    let (result, order) = execute_observed(
        compiled_graph(
            &[
                ("record_order", "cold", None),
                ("record_order", "warm-seed", None),
                ("record_after", "warm-downstream", None),
            ],
            &[(1, 2)],
        ),
        ctx,
        3,
    )
    .await;
    (
        order,
        result.n_cache_hits,
        result.n_cache_misses,
        materialized_hashes(&measured_dir),
        materialized_logical_hashes(&measured_dir),
    )
}

#[test]
fn dag_opt_advanced_gate_covers_priority_pass_and_default_off_parity() {
    let (_, enabled) = priority_only(true).optimize(compiled(Some(42)));
    assert_eq!(enabled[&0].user_priority, 42);

    let (_, disabled) = priority_only(false).optimize(compiled(Some(42)));
    assert!(
        disabled.values().all(|hint| hint.user_priority == 0),
        "default-off priority pass must not change scheduling hints"
    );

    let (_, unset) = priority_only(true).optimize(compiled(None));
    assert!(
        unset.values().all(|hint| hint.user_priority == 0),
        "an omitted priority remains neutral"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_executes_higher_priority_ready_node_first() {
    let _guard = TEST_LOCK.lock().await;
    EXECUTION_ORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    let plan = compiled_graph(
        &[
            ("record_order", "bulk", Some(1)),
            ("record_order", "urgent", Some(50)),
            ("record_after", "bulk-tail", None),
        ],
        &[(0, 2)],
    );
    let temp = tempfile::tempdir().expect("executor tempdir");
    let mut ctx = ExecCtx::new(temp.path().join("job")).with_max_in_flight(1);
    let mut optimizer = priority_only(true);
    optimizer.critical_path = true;
    ctx.dag_optimizer = Some(optimizer);

    ParallelExecutor::execute(plan, ctx)
        .await
        .expect("execute priority fixture");
    assert_eq!(
        *EXECUTION_ORDER
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
        ["urgent", "bulk", "bulk-tail"],
        "user priority must outrank the cold sibling's longer critical path"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_schedules_real_cache_hit_before_cold_sibling() {
    let _guard = TEST_LOCK.lock().await;
    let (order, hits, misses, output_hash) =
        run_cache_order_fixture(true, false, (None, None), false).await;
    assert_eq!(
        order,
        [("skip", 1), ("begin", 0)],
        "enabled cache-aware pass must choose the live cache hit first"
    );
    assert_eq!((hits, misses), (1, 1));
    assert_eq!(output_hash, ContentHash::of_bytes(b"warm"));
}

#[tokio::test]
async fn dag_opt_advanced_gate_cache_ordering_is_default_off_and_hash_neutral() {
    let _guard = TEST_LOCK.lock().await;
    assert!(
        !DagOptimizer::new().cache_aware,
        "advanced cache ordering must default off"
    );
    let (order, hits, misses, output_hash) =
        run_cache_order_fixture(false, false, (None, None), false).await;
    assert_eq!(
        order,
        [("begin", 0), ("skip", 1)],
        "disabled pass must retain ascending-NodeId ready order"
    );
    assert_eq!((hits, misses), (1, 1));
    assert_eq!(
        output_hash,
        ContentHash::of_bytes(b"warm"),
        "ready ordering must not change the terminal artifact identity"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_uses_downstream_key_and_preserves_every_node_hash() {
    let _guard = TEST_LOCK.lock().await;
    let (enabled_order, enabled_hits, enabled_misses, enabled_hashes, enabled_logical_hashes) =
        run_downstream_cache_fixture(true).await;
    assert_eq!(
        enabled_order,
        [("skip", 1), ("skip", 2), ("begin", 0)],
        "root and downstream hits must drain before the cold lower-NodeId sibling"
    );
    assert_eq!((enabled_hits, enabled_misses), (2, 1));

    let (disabled_order, disabled_hits, disabled_misses, disabled_hashes, disabled_logical_hashes) =
        run_downstream_cache_fixture(false).await;
    assert_eq!(
        disabled_order,
        [("begin", 0), ("skip", 1), ("skip", 2)],
        "default-off scheduling must retain ascending ready-node order"
    );
    assert_eq!((disabled_hits, disabled_misses), (2, 1));
    assert_eq!(
        enabled_hashes, disabled_hashes,
        "cache-aware reordering must preserve every portable content identity"
    );
    assert_eq!(
        enabled_logical_hashes,
        [
            ContentHash::of_bytes(b"cold"),
            ContentHash::of_bytes(b"warm-seed"),
            ContentHash::of_bytes(b"warm-downstream"),
        ],
        "logical invocation identities remain the stage-declared hashes"
    );
    assert_eq!(enabled_logical_hashes, disabled_logical_hashes);
}

#[tokio::test]
async fn dag_opt_advanced_gate_cache_hit_dominates_cold_user_priority() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("precedence tempdir");
    let cache_root = temp.path().join("shared-cache");
    let make_ctx = |job: &str| {
        let job_dir = temp.path().join(job);
        let mut ctx = ExecCtx::new(job_dir.clone()).with_max_in_flight(1);
        ctx.cache = Arc::new(
            CacheHandle::job_local(job_dir.join("_cache")).with_global(cache_root.clone()),
        );
        ctx
    };
    ParallelExecutor::execute(compiled_nodes(&[("warm", None)]), make_ctx("prewarm"))
        .await
        .expect("prewarm precedence fixture");

    let mut optimizer = cache_only(true);
    optimizer.priority_aware = true;
    optimizer.critical_path = true;
    let mut ctx = make_ctx("measured");
    ctx.dag_optimizer = Some(optimizer);
    let (result, order) = execute_observed(
        compiled_graph(
            &[
                ("record_order", "cold", Some(100)),
                ("record_order", "warm", Some(1)),
                ("record_after", "cold-tail", None),
            ],
            &[(0, 2)],
        ),
        ctx,
        3,
    )
    .await;
    assert_eq!(
        order,
        [("skip", 1), ("begin", 0), ("begin", 2)],
        "live cache warmth must dominate both user priority and critical path"
    );
    assert_eq!((result.n_cache_hits, result.n_cache_misses), (1, 2));
}

#[tokio::test]
async fn dag_opt_advanced_gate_force_recompute_makes_every_ready_node_cold() {
    let _guard = TEST_LOCK.lock().await;
    let (order, hits, misses, output_hash) =
        run_cache_order_fixture(true, true, (None, None), false).await;
    assert_eq!(
        order,
        [("begin", 0), ("begin", 1)],
        "bypass-cache must neutralize cache warmth and preserve NodeId order"
    );
    assert_eq!((hits, misses), (0, 2));
    assert_eq!(output_hash, ContentHash::of_bytes(b"warm"));
}

#[tokio::test]
async fn dag_opt_advanced_gate_probes_each_cold_key_at_most_twice() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("probe-count tempdir");
    let gets = Arc::new(AtomicUsize::new(0));
    let remote = Arc::new(CountingMissStore {
        gets: gets.clone(),
        delay: std::time::Duration::ZERO,
    });
    let remote = ObjectStore::adapter(remote).blocking();
    let job_dir = temp.path().join("measured");
    let mut ctx = ExecCtx::new(job_dir.clone()).with_max_in_flight(1);
    ctx.cache = Arc::new(CacheHandle::job_local(job_dir.join("_cache")).with_remote(remote));
    ctx.dag_optimizer = Some(cache_only(true));
    let labels: Vec<String> = (0..4).map(|i| format!("cold-{i}")).collect();
    let nodes: Vec<(&str, Option<i32>)> =
        labels.iter().map(|label| (label.as_str(), None)).collect();

    ParallelExecutor::execute(compiled_nodes(&nodes), ctx)
        .await
        .expect("execute cold probe-count fixture");
    assert_eq!(
        gets.load(Ordering::SeqCst),
        2 * nodes.len(),
        "each exact cold key gets one scheduling probe plus run_node's race-safe lookup"
    );
}

#[tokio::test]
async fn dag_opt_advanced_gate_reprobes_key_after_miss_to_hit_race() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("probe-race tempdir");
    let remote_adapter = Arc::new(RacingHitStore {
        state: std::sync::Mutex::new(RacingHitState::default()),
    });
    let remote = ObjectStore::adapter(remote_adapter.clone()).blocking();
    let prewarm_dir = temp.path().join("prewarm");
    let mut prewarm = ExecCtx::new(prewarm_dir.clone()).with_max_in_flight(1);
    prewarm.cache =
        Arc::new(CacheHandle::job_local(prewarm_dir.join("_cache")).with_remote(remote.clone()));
    ParallelExecutor::execute(compiled_nodes(&[("race-warm", None)]), prewarm)
        .await
        .expect("seed raced portable cache artifact");
    remote_adapter.arm();
    let job_dir = temp.path().join("measured");
    let mut ctx = ExecCtx::new(job_dir.clone()).with_max_in_flight(1);
    ctx.cache = Arc::new(CacheHandle::job_local(job_dir.join("_cache")).with_remote(remote));
    ctx.dag_optimizer = Some(cache_only(true));

    let (result, order) = execute_observed(
        compiled_graph(
            &[
                ("record_order", "race-warm", None),
                ("record_after", "cold", None),
                ("record_order", "race-warm", None),
            ],
            &[(0, 1)],
        ),
        ctx,
        3,
    )
    .await;
    assert_eq!(
        order,
        [("skip", 0), ("skip", 1), ("begin", 2)],
        "a run_node race hit must invalidate the earlier miss before scheduling a same-key sibling"
    );
    assert_eq!((result.n_cache_hits, result.n_cache_misses), (2, 1));
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_slow_remote_probe_does_not_block_runtime() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("slow-probe tempdir");
    let gets = Arc::new(AtomicUsize::new(0));
    let remote = Arc::new(CountingMissStore {
        gets: gets.clone(),
        delay: std::time::Duration::from_millis(250),
    });
    let remote = ObjectStore::adapter(remote).blocking();
    let job_dir = temp.path().join("measured");
    let mut ctx = ExecCtx::new(job_dir.clone()).with_max_in_flight(1);
    ctx.cache = Arc::new(CacheHandle::job_local(job_dir.join("_cache")).with_remote(remote));
    ctx.dag_optimizer = Some(cache_only(true));

    let started = std::time::Instant::now();
    let execution = tokio::spawn(ParallelExecutor::execute(
        compiled_nodes(&[("slow-cold", None)]),
        ctx,
    ));
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert!(
        started.elapsed() < std::time::Duration::from_millis(150),
        "cache I/O blocked the single-threaded async runtime"
    );
    execution
        .await
        .expect("executor task joins")
        .expect("slow remote miss falls back to execution");
    assert_eq!(gets.load(Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "current_thread")]
async fn dag_opt_advanced_gate_slow_probe_honors_plan_deadline_before_stage_begin() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("deadline-probe tempdir");
    let gets = Arc::new(AtomicUsize::new(0));
    let remote = Arc::new(CountingMissStore {
        gets: gets.clone(),
        delay: std::time::Duration::from_millis(250),
    });
    let remote = ObjectStore::adapter(remote).blocking();
    let job_dir = temp.path().join("measured");
    let mut ctx = ExecCtx::new(job_dir.clone())
        .with_max_in_flight(1)
        .with_deadline(std::time::Duration::from_millis(25));
    ctx.cache = Arc::new(CacheHandle::job_local(job_dir.join("_cache")).with_remote(remote));
    ctx.dag_optimizer = Some(cache_only(true));
    let mut events = ctx.status.subscribe();

    let started = std::time::Instant::now();
    let result = ParallelExecutor::execute(compiled_nodes(&[("slow-cold", None)]), ctx).await;
    assert!(
        matches!(result, Err(PlanError::DeadlineExceeded { .. })),
        "a probe that outlives the plan deadline must fail as DeadlineExceeded"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_millis(150),
        "the coordinator waited for the slow cache probe after the plan deadline"
    );
    assert!(
        std::iter::from_fn(|| events.try_recv().ok())
            .all(|event| !matches!(event, StageEvent::StageBegin { .. })),
        "no stage may launch after its deadline expires during cache probing"
    );
    assert_eq!(gets.load(Ordering::SeqCst), 1);
}
