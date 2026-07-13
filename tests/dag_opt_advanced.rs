// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Named progress gate for ADR 0102's landed advanced-optimizer slice.
//!
//! User-priority scheduling and live cache-warm ready-queue ordering have
//! landed. Fusion, speculation, and pipeline parallelism remain later,
//! independently gated increments.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use blut::framework::artifact::{Artifact, ArtifactMetadata, ContentHash};
use blut::framework::cache::CacheHandle;
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
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

static EXECUTION_ORDER: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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
    }
}

fn cache_only(enabled: bool) -> DagOptimizer {
    DagOptimizer {
        eliminate_dead_code: false,
        critical_path: false,
        cache_aware: enabled,
        memory_aware: false,
        priority_aware: false,
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

fn materialized_hashes(job_dir: &std::path::Path) -> Vec<ContentHash> {
    let mut stage_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(job_dir.join("stages"))
        .expect("read materialized stage dirs")
        .map(|entry| entry.expect("read stage-dir entry").path())
        .filter(|path| path.is_dir())
        .collect();
    stage_dirs.sort();
    stage_dirs
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
