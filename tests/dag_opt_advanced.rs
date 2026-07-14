// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Named progress gate for ADR 0102's landed advanced-optimizer slice.
//!
//! User-priority scheduling, live cache-warm ready ordering, and the first
//! conservative whole-plan linear coalescing slice have landed. General
//! internal-subchain fusion, speculation, and pipeline parallelism remain
//! later, independently gated increments.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use blut::framework::artifact::{Artifact, ArtifactMetadata, ContentHash};
use blut::framework::cache::{CacheHandle, CacheProof};
use blut::framework::cookbook::{Cookbook, Registry};
use blut::framework::dag_opt::DagOptimizer;
use blut::framework::executor::{ExecCtx, ParallelExecutor};
use blut::framework::object_store::BlobStore;
use blut::framework::plan::CompiledPlan;
use blut::framework::plan_spec::{PLAN_SPEC_VERSION, PlanSpec, SpecNode};
use blut::framework::resource::Resource;
use blut::framework::stage::{ErasedStageCtor, Stage, StageContext};
use blut::framework::status::StageEvent;
use blut::framework::{PlanError, StageError};
use blut::recipes::recipe::RecipeDef;
use futures::FutureExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
    first_key: Option<ContentHash>,
    gets_by_key: std::collections::HashMap<ContentHash, usize>,
}

#[derive(Debug)]
struct RacingHitStore {
    state: std::sync::Mutex<RacingHitState>,
    body: Vec<u8>,
}

impl BlobStore for RacingHitStore {
    fn get(&self, key: ContentHash) -> std::io::Result<Option<Vec<u8>>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let first_key = *state.first_key.get_or_insert(key);
        let gets = state.gets_by_key.entry(key).or_default();
        *gets += 1;
        Ok((key == first_key && *gets == 2).then(|| self.body.clone()))
    }

    fn put(&self, _key: ContentHash, _bytes: &[u8]) -> std::io::Result<()> {
        Ok(())
    }

    fn head(&self, _key: ContentHash) -> std::io::Result<bool> {
        Ok(false)
    }
}

impl BlobStore for CountingMissStore {
    fn get(&self, _key: ContentHash) -> std::io::Result<Option<Vec<u8>>> {
        self.gets.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(self.delay);
        Ok(None)
    }

    fn put(&self, _key: ContentHash, _bytes: &[u8]) -> std::io::Result<()> {
        Ok(())
    }

    fn head(&self, _key: ContentHash) -> std::io::Result<bool> {
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

struct RecordNondeterministic;

#[async_trait]
impl Stage for RecordNondeterministic {
    const NAME: &'static str = "record_nondeterministic";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const DETERMINISTIC: bool = false;
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
            ("record_nondeterministic", || {
                Arc::new(RecordNondeterministic)
            }),
            ("record_slow_after", || Arc::new(RecordSlowAfter)),
            ("record_panicking", || Arc::new(RecordPanicking)),
            ("record_failing", || Arc::new(RecordFailing)),
            ("record_network_after", || Arc::new(RecordNetworkAfter)),
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
            })
            .collect(),
        edges: edges.to_vec(),
        expansions: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
    .compile(&registry)
    .expect("compile gate plan")
}

fn priority_only(enabled: bool) -> DagOptimizer {
    DagOptimizer {
        eliminate_dead_code: false,
        critical_path: false,
        cache_aware: false,
        memory_aware: false,
        priority_aware: enabled,
        stage_fusion: false,
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
                .content_hash
        })
        .collect()
}

fn materialized_cache_keys(job_dir: &std::path::Path) -> Vec<ContentHash> {
    materialized_stage_dirs(job_dir)
        .into_iter()
        .map(|stage_dir| {
            CacheProof::read_from(&stage_dir.join("cache-proof.json"))
                .expect("read materialized cache proof")
                .key
        })
        .collect()
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
    assert_eq!(
        fused_decodes, 0,
        "a fused miss chain must hand typed artifacts directly between stages without bincode-decoding the handoff"
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
) -> (Vec<(&'static str, u32)>, usize, usize, Vec<ContentHash>) {
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
    let (enabled_order, enabled_hits, enabled_misses, enabled_hashes) =
        run_downstream_cache_fixture(true).await;
    assert_eq!(
        enabled_order,
        [("skip", 1), ("skip", 2), ("begin", 0)],
        "root and downstream hits must drain before the cold lower-NodeId sibling"
    );
    assert_eq!((enabled_hits, enabled_misses), (2, 1));

    let (disabled_order, disabled_hits, disabled_misses, disabled_hashes) =
        run_downstream_cache_fixture(false).await;
    assert_eq!(
        disabled_order,
        [("begin", 0), ("skip", 1), ("skip", 2)],
        "default-off scheduling must retain ascending ready-node order"
    );
    assert_eq!((disabled_hits, disabled_misses), (2, 1));
    assert_eq!(
        enabled_hashes, disabled_hashes,
        "cache-aware reordering must preserve every materialized node hash"
    );
    assert_eq!(
        enabled_hashes,
        [
            ContentHash::of_bytes(b"cold"),
            ContentHash::of_bytes(b"warm-seed"),
            ContentHash::of_bytes(b"warm-downstream"),
        ]
    );
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
    let cached = OrderArtifact {
        path: temp.path().join("race-warm.txt"),
        content_hash: ContentHash::of_bytes(b"race-warm"),
    };
    let erased = blut::framework::stage::ErasedArtifact::from_typed(&cached)
        .expect("erase raced cache artifact");
    let remote = Arc::new(RacingHitStore {
        state: std::sync::Mutex::new(RacingHitState::default()),
        body: bincode::serialize(&erased).expect("encode raced cache artifact"),
    });
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
