//! Executor test suite — fixture stages + end-to-end plan/cache/retry
//! integration tests. Split out of `executor.rs` purely for navigability
//! (the suite had grown larger than the executor itself); mounted back as
//! `executor::tests` via `#[path]`, so `use super::*` still resolves to the
//! executor module's private items exactly as before the split.

// intentional (BLD-C2): the executor tests serialize on a process-wide
// `TEST_LOCK` std Mutex held across `.await` to stop concurrent tests from
// racing on the shared content cache / job dirs. A std guard across an await
// is exactly the pattern clippy flags, but here it is the deliberate
// serialization mechanism — the guard is never contended by real async work,
// only by the test harness, so it cannot deadlock the runtime.
#![allow(clippy::await_holding_lock)]

use super::*;
use crate::backends::LamuTrainerBackend;
use crate::framework::artifact::Artifact;
use crate::framework::compat::Compatible;
use crate::framework::plan::Plan;
use crate::framework::stage::Stage;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

// Toy artifacts.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Counter {
    n: u32,
}
impl Artifact for Counter {
    const KIND: &'static str = "test.counter";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&self.n.to_le_bytes())
    }
    fn primary_path(&self) -> &Path {
        Path::new(".")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct EmptyArgs;

static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

static MAKE_RUN_COUNT: AtomicU32 = AtomicU32::new(0);

// --- next_ready: critical-path-first ready-node selection (B1) ---------

/// Build a `ScheduleHint` carrying only a critical-path length (the only
/// field `next_ready` reads).
fn cp_hint(critical_path_len: u32) -> crate::framework::dag_opt::ScheduleHint {
    crate::framework::dag_opt::ScheduleHint {
        critical_path_len,
        ..Default::default()
    }
}

#[test]
fn next_ready_prefers_longer_critical_path() {
    // The higher-priority node (2) has the LARGER id, so the old
    // smallest-id rule would have wrongly returned 1.
    let ready: BTreeSet<NodeId> = [1, 2].into_iter().collect();
    let hints: HashMap<NodeId, _> = [(1, cp_hint(1)), (2, cp_hint(3))].into_iter().collect();
    assert_eq!(next_ready(&ready, &hints), Some(2));
}

#[test]
fn next_ready_tie_breaks_ascending_id() {
    // Equal critical paths → smallest NodeId, matching the historical
    // `ready.iter().next()` behaviour (byte-equal output).
    let ready: BTreeSet<NodeId> = [1, 4].into_iter().collect();
    let hints: HashMap<NodeId, _> = [(1, cp_hint(1)), (4, cp_hint(1))].into_iter().collect();
    assert_eq!(next_ready(&ready, &hints), Some(1));
}

#[test]
fn next_ready_empty_hints_is_smallest_id() {
    // No optimizer configured (empty map) → smallest id, identical to today.
    let ready: BTreeSet<NodeId> = [5, 6].into_iter().collect();
    let hints: HashMap<NodeId, crate::framework::dag_opt::ScheduleHint> = HashMap::new();
    assert_eq!(next_ready(&ready, &hints), Some(5));
}

#[test]
fn next_ready_injected_node_defaults_to_zero() {
    // A runtime-injected node (id 100, absent from hints) defaults to
    // cp=0, so the critical-path original (id 2, cp=3) is picked first.
    let ready: BTreeSet<NodeId> = [2, 100].into_iter().collect();
    let hints: HashMap<NodeId, _> = [(2, cp_hint(3))].into_iter().collect();
    assert_eq!(next_ready(&ready, &hints), Some(2));
}

#[test]
fn next_ready_empty_set_is_none() {
    // Preserves the `else break` termination in the spawn loop.
    let ready: BTreeSet<NodeId> = BTreeSet::new();
    let hints: HashMap<NodeId, crate::framework::dag_opt::ScheduleHint> = HashMap::new();
    assert_eq!(next_ready(&ready, &hints), None);
}

struct MakeOne;
#[async_trait]
impl Stage for MakeOne {
    const NAME: &'static str = "make_one";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        MAKE_RUN_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(Counter { n: 1 })
    }
}

static INC_RUN_COUNT: AtomicU32 = AtomicU32::new(0);

struct Increment;
#[async_trait]
impl Stage for Increment {
    const NAME: &'static str = "increment";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        INC_RUN_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(Counter { n: input.n + 1 })
    }
}

impl Compatible<LamuTrainerBackend> for MakeOne {}
impl Compatible<LamuTrainerBackend> for Increment {}

struct AlwaysFail;
#[async_trait]
impl Stage for AlwaysFail {
    const NAME: &'static str = "always_fail";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        Err(StageError::BadInput("forced failure".into()))
    }
}
impl Compatible<LamuTrainerBackend> for AlwaysFail {}

// An ADVISORY gate that always fails — models a dry-run/verdict gate after
// training (ADR 0071). Its failure must NOT fail the plan.
struct AdvisoryGate;
#[async_trait]
impl Stage for AdvisoryGate {
    const NAME: &'static str = "advisory_gate";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const ADVISORY: bool = true;
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        Err(StageError::BadInput(
            "advisory verdict: would-not-promote".into(),
        ))
    }
}
impl Compatible<LamuTrainerBackend> for AdvisoryGate {}

fn fresh_ctx() -> (tempfile::TempDir, ExecCtx) {
    let td = tempfile::tempdir().unwrap();
    let ctx = ExecCtx::new(td.path().to_path_buf());
    (td, ctx)
}

#[tokio::test]
async fn linear_plan_executes_in_order() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
    INC_RUN_COUNT.store(0, Ordering::SeqCst);
    let (_td, ctx) = fresh_ctx();
    let plan = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(Increment, EmptyArgs)
        .then(Increment, EmptyArgs)
        .finish()
        .into_compiled();
    let result = SequentialExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(result.n_stages, 3);
    assert_eq!(result.n_cache_misses, 3);
    assert_eq!(result.n_cache_hits, 0);
    let out = result.final_output.unwrap();
    let counter: Counter = out.into_typed().unwrap();
    assert_eq!(counter.n, 3);
}

#[tokio::test]
async fn advisory_gate_failure_is_non_fatal_and_preserves_output() {
    // ADR 0071: train (MakeOne) → an advisory gate that FAILS. The plan must
    // NOT fail; the warning is recorded and the train output is surfaced.
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
    INC_RUN_COUNT.store(0, Ordering::SeqCst);
    let (_td, ctx) = fresh_ctx();
    let plan = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(AdvisoryGate, EmptyArgs)
        .finish()
        .into_compiled();
    let result = SequentialExecutor::execute(plan, ctx)
        .await
        .expect("an advisory gate's failure must NOT fail the plan");
    assert_eq!(result.warnings.len(), 1, "the advisory failure is recorded");
    assert_eq!(result.warnings[0].stage, "advisory_gate");
    // ADR 0072 A5: the coordinator/sequential paths don't downcast the
    // failure to a `StageFailure` before building the warning today, so
    // there's no origin to thread through yet — this pins that current
    // behaviour rather than silently starting to assume a value.
    assert_eq!(result.warnings[0].origin, None);
    // The upstream train output is surfaced even though the terminal gate tripped.
    let counter: Counter = result
        .final_output
        .expect("train output preserved past the advisory gate")
        .into_typed()
        .unwrap();
    assert_eq!(counter.n, 1);
}

/// ADR 0072 A5: `StageWarning::origin` can carry fault-attribution data
/// when it's available. `StageWarning` isn't `Serialize` and its
/// `warnings` Vec is only ever inspected in-memory (printed by `cli.rs`,
/// asserted on directly in tests) — no serde round-trip exists to pin,
/// so this is a plain construction + field-access regression test.
#[test]
fn stage_warning_carries_optional_fault_origin() {
    let w = StageWarning {
        idx: 0,
        stage: "advisory_gate".to_string(),
        reason: "upstream service timed out".to_string(),
        origin: Some(crate::framework::error_domain::FaultOrigin::Engine),
    };
    assert_eq!(
        w.origin,
        Some(crate::framework::error_domain::FaultOrigin::Engine)
    );
    // Clone must preserve it too (StageWarning derives Clone).
    let cloned = w.clone();
    assert_eq!(cloned.origin, w.origin);
}

#[tokio::test]
async fn cache_hit_skips_run_on_repeat_execution() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
    INC_RUN_COUNT.store(0, Ordering::SeqCst);
    let (_td, ctx) = fresh_ctx();
    let cache = ctx.cache.clone();

    let plan = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(Increment, EmptyArgs)
        .finish()
        .into_compiled();
    let r1 = SequentialExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(r1.n_cache_misses, 2);
    assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 1);

    let td_keepalive_for_second_run = tempfile::tempdir().unwrap();
    let job_dir2 = td_keepalive_for_second_run.path().to_path_buf();
    let ctx2 = ExecCtx::new(job_dir2);
    let ctx2 = ExecCtx { cache, ..ctx2 };
    let plan2 = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(Increment, EmptyArgs)
        .finish()
        .into_compiled();
    let r2 = SequentialExecutor::execute(plan2, ctx2).await.unwrap();
    assert_eq!(
        r2.n_cache_hits, 2,
        "second run should hit cache for both stages"
    );
    assert_eq!(r2.n_cache_misses, 0);
    assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 1);
}

/// INC D (S4): `with_bypass_cache(true)` forces a recompute — a stage with a
/// WARM cache entry still EXECUTES (run counts climb, all misses), and the
/// fresh result is STILL cached, so a subsequent NON-bypass run hits again.
#[tokio::test]
async fn bypass_cache_forces_rerun_but_still_caches() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
    INC_RUN_COUNT.store(0, Ordering::SeqCst);

    // Run 1 (cold): both stages execute + populate the cache.
    let (_td, ctx) = fresh_ctx();
    let cache = ctx.cache.clone();
    let plan = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(Increment, EmptyArgs)
        .finish()
        .into_compiled();
    let r1 = SequentialExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(r1.n_cache_misses, 2);
    assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 1);
    assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 1);

    // Run 2 (BYPASS, shared cache): warm entries exist, but bypass forces
    // every stage to run again — all misses, run counts climb to 2.
    let td2 = tempfile::tempdir().unwrap();
    let ctx2 = ExecCtx::new(td2.path().to_path_buf());
    let ctx2 = ExecCtx {
        cache: cache.clone(),
        ..ctx2
    }
    .with_bypass_cache(true);
    let plan2 = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(Increment, EmptyArgs)
        .finish()
        .into_compiled();
    let r2 = SequentialExecutor::execute(plan2, ctx2).await.unwrap();
    assert_eq!(r2.n_cache_hits, 0, "bypass: warm entries are NOT read");
    assert_eq!(r2.n_cache_misses, 2, "bypass: every stage executes");
    assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 2, "MakeOne re-ran");
    assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 2, "Increment re-ran");

    // Run 3 (NO bypass, shared cache): the fresh result Run 2 wrote is still
    // cached → both stages hit, run counts stay at 2.
    let td3 = tempfile::tempdir().unwrap();
    let ctx3 = ExecCtx::new(td3.path().to_path_buf());
    let ctx3 = ExecCtx { cache, ..ctx3 };
    let plan3 = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(Increment, EmptyArgs)
        .finish()
        .into_compiled();
    let r3 = SequentialExecutor::execute(plan3, ctx3).await.unwrap();
    assert_eq!(
        r3.n_cache_hits, 2,
        "bypass still wrote fresh entries → later run hits"
    );
    assert_eq!(r3.n_cache_misses, 0);
    assert_eq!(
        MAKE_RUN_COUNT.load(Ordering::SeqCst),
        2,
        "Run 3 did not execute"
    );
    assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 2);
}

/// INC G (C3): a declarative `.toml` recipe COMPILES (via
/// `DeclarativeRecipe::compile`) AND EXECUTES to completion through the
/// same executor `launch_compiled_plan` drives. This is the load-bearing
/// half of the CLI `recipe declare --run` launch path — the CLI half only
/// adds job-dir/admission/lock plumbing on top of this compile→execute
/// chain. A small `make_one → increment` chain runs end-to-end and produces
/// its `Counter` output, proving the `.toml` → CompiledPlan → execute bridge.
#[tokio::test]
async fn declarative_toml_compiles_and_executes_to_completion() {
    use crate::framework::cookbook::{Cookbook, Registry};
    use crate::framework::stage::ErasedStageCtor;
    use crate::recipes::declarative::DeclarativeRecipe;
    use crate::recipes::recipe::RecipeDef;

    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
    INC_RUN_COUNT.store(0, Ordering::SeqCst);

    // A cookbook exposing the two toy stages as erased ctors — exactly what
    // `Registry::find_erased_stage` resolves a `.toml` stage NAME against.
    static ERASED: &[(&str, ErasedStageCtor)] = &[
        ("make_one", || std::sync::Arc::new(MakeOne)),
        ("increment", || std::sync::Arc::new(Increment)),
    ];
    static NO_RECIPES: &[&RecipeDef] = &[];
    struct ToyCookbook;
    impl Cookbook for ToyCookbook {
        fn name(&self) -> &'static str {
            "toy"
        }
        fn recipes(&self) -> &'static [&'static RecipeDef] {
            NO_RECIPES
        }
        fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
            ERASED
        }
    }
    let mut reg = Registry::new();
    reg.register(Box::new(ToyCookbook));

    // The declarative recipe: () → make_one → increment.
    let toml = r#"
        name = "toy_chain"
        backend = "lamu"
        [[stages]]
        stage = "make_one"
        [[stages]]
        stage = "increment"
    "#;
    let recipe = DeclarativeRecipe::parse(toml, "toy_chain.toml").unwrap();
    // Compile through the SAME path the CLI launch uses.
    let plan = recipe
        .compile(&reg)
        .expect("declarative recipe compiles + kind-checks");
    assert_eq!(plan.n_nodes(), 2);
    assert_eq!(plan.name(), "toy_chain");

    // Execute to completion (the half `launch_compiled_plan` runs internally).
    let (_td, ctx) = fresh_ctx();
    let result = SequentialExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(result.n_stages, 2);
    assert_eq!(result.n_cache_misses, 2, "both stages executed");
    assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 1, "make_one ran");
    assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 1, "increment ran");
    let out: Counter = result.final_output.unwrap().into_typed().unwrap();
    assert_eq!(out.n, 2, "make_one(1) → increment(+1) = 2");
}

#[tokio::test]
async fn stage_failure_propagates_as_plan_error() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, ctx) = fresh_ctx();
    let plan = Plan::<(), LamuTrainerBackend>::new("failing", serde_json::json!({}))
        .start(AlwaysFail, EmptyArgs)
        .finish()
        .into_compiled();
    let r = SequentialExecutor::execute(plan, ctx).await;
    match r {
        Err(PlanError::StageFailed { idx, stage, source }) => {
            assert_eq!(idx, 0);
            assert_eq!(stage, "always_fail");
            assert!(matches!(source, StageError::BadInput(_)));
        }
        other => panic!("unexpected: {:?}", other),
    }
}

#[tokio::test]
async fn cancel_before_first_stage_returns_cancelled() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, ctx) = fresh_ctx();
    ctx.cancel.cancel();
    let plan = Plan::<(), LamuTrainerBackend>::new("c", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .finish()
        .into_compiled();
    let r = SequentialExecutor::execute(plan, ctx).await;
    assert!(matches!(r, Err(PlanError::Cancelled)));
}

#[tokio::test]
async fn status_jsonl_persists_to_disk() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (td, ctx) = fresh_ctx();
    let plan = Plan::<(), LamuTrainerBackend>::new("p", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(Increment, EmptyArgs)
        .finish()
        .into_compiled();
    let _ = SequentialExecutor::execute(plan, ctx).await.unwrap();
    let path = td.path().join("status.jsonl");
    assert!(path.exists());
    let body = std::fs::read_to_string(&path).unwrap();
    let n_begin = body.matches("\"kind\":\"stage_begin\"").count();
    let n_end = body.matches("\"kind\":\"stage_end\"").count();
    assert_eq!(n_begin, 2);
    assert_eq!(n_end, 2);
}

#[tokio::test]
async fn args_json_persisted_at_job_root() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (td, ctx) = fresh_ctx();
    let recipe_args = serde_json::json!({"output_name": "test", "since": "30d"});
    let plan = Plan::<(), LamuTrainerBackend>::new("p", recipe_args.clone())
        .start(MakeOne, EmptyArgs)
        .finish()
        .into_compiled();
    let _ = SequentialExecutor::execute(plan, ctx).await.unwrap();
    let body = std::fs::read_to_string(td.path().join("args.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed, recipe_args);
}

#[tokio::test]
async fn sidecar_metadata_written_per_stage() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (td, ctx) = fresh_ctx();
    let plan = Plan::<(), LamuTrainerBackend>::new("p", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .finish()
        .into_compiled();
    let _ = SequentialExecutor::execute(plan, ctx).await.unwrap();
    let sidecar = td.path().join("stages/0-make_one/output.metadata.json");
    assert!(
        sidecar.exists(),
        "expected sidecar at {}",
        sidecar.display()
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&sidecar).unwrap()).unwrap();
    assert_eq!(parsed["kind"], "test.counter");
    assert_eq!(parsed["produced_by_stage"], "make_one");
}

// ── FW-1 cross-machine-stable cache key ──────────────────────────
use crate::framework::artifact::ContentHash as CH;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PathArt {
    content: u8,
    path: PathBuf,
}
impl Artifact for PathArt {
    const KIND: &'static str = "test.path_art";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> CH {
        CH::of_bytes(&[self.content])
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct PathArtArgs {
    abs_path: String,
    content: u8,
}

struct MakePathArt;
#[async_trait]
impl Stage for MakePathArt {
    const NAME: &'static str = "make_path_art";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = PathArt;
    type Args = PathArtArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        args: &PathArtArgs,
    ) -> Result<PathArt, StageError> {
        Ok(PathArt {
            content: args.content,
            path: PathBuf::from(&args.abs_path),
        })
    }
}
impl Compatible<LamuTrainerBackend> for MakePathArt {}

struct ConsumePathArt;
#[async_trait]
impl Stage for ConsumePathArt {
    const NAME: &'static str = "consume_path_art";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = PathArt;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        input: PathArt,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        Ok(Counter {
            n: input.content as u32,
        })
    }
}
impl Compatible<LamuTrainerBackend> for ConsumePathArt {}

async fn downstream_input_hash(abs_path: &str, content: u8) -> CH {
    let td = tempfile::tempdir().unwrap();
    let ctx = ExecCtx::new(td.path().to_path_buf());
    let mut rx = ctx.status.subscribe();
    let plan = Plan::<(), LamuTrainerBackend>::new("fw1", serde_json::json!({}))
        .start(
            MakePathArt,
            PathArtArgs {
                abs_path: abs_path.to_string(),
                content,
            },
        )
        .then(ConsumePathArt, EmptyArgs)
        .finish()
        .into_compiled();
    SequentialExecutor::execute(plan, ctx).await.unwrap();
    let mut found = None;
    while let Ok(evt) = rx.try_recv() {
        if let StageEvent::StageBegin {
            node_idx: 1,
            input_hash,
            ..
        } = evt
        {
            found = Some(input_hash);
        }
    }
    found.expect("downstream StageBegin must carry an input_hash")
}

#[tokio::test]
async fn cache_key_stable_across_abs_path() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let h1 = downstream_input_hash("/machine-a/jobs/run1/stages/0-make_path_art/out", 7).await;
    let h2 = downstream_input_hash("/totally/different/machine-b/xyz/out", 7).await;
    assert_eq!(h1, h2, "FW-1: same content at different paths → same key");
    let h3 = downstream_input_hash("/machine-a/jobs/run1/stages/0-make_path_art/out", 8).await;
    assert_ne!(h1, h3, "different content must still change the key");
}

#[tokio::test]
async fn content_hash_invoked_for_deterministic_stage() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let abs = "/some/abs/path/out";
    let content = 42u8;
    let observed = downstream_input_hash(abs, content).await;
    let art = PathArt {
        content,
        path: PathBuf::from(abs),
    };
    let want = art.content_hash();
    assert_eq!(observed, want, "FW-1: logical hash = content_hash()");
    let erased = ErasedArtifact::from_typed(&art).unwrap();
    let handle_hash = content_hash_from_erased(&erased);
    assert_ne!(
        observed, handle_hash,
        "content hash must differ from handle hash here"
    );
}

/// Capture the EMITTED `StageEnd.output_hash` for node 0 of a single-stage
/// plan (the recorded provenance hash my B.1(a) fix changed).
async fn recorded_output_hash(abs_path: &str, content: u8) -> CH {
    let td = tempfile::tempdir().unwrap();
    let ctx = ExecCtx::new(td.path().to_path_buf());
    let mut rx = ctx.status.subscribe();
    let plan = Plan::<(), LamuTrainerBackend>::new("fw1b", serde_json::json!({}))
        .start(
            MakePathArt,
            PathArtArgs {
                abs_path: abs_path.to_string(),
                content,
            },
        )
        .finish()
        .into_compiled();
    SequentialExecutor::execute(plan, ctx).await.unwrap();
    let mut found = None;
    while let Ok(evt) = rx.try_recv() {
        if let StageEvent::StageEnd {
            node_idx: 0,
            output_hash,
            ..
        } = evt
        {
            found = Some(output_hash);
        }
    }
    found.expect("StageEnd must carry an output_hash")
}

#[tokio::test]
async fn recorded_output_hash_is_content_based_and_path_stable() {
    // B.1(a): the EMITTED StageEnd.output_hash (→ output.metadata.json sidecar
    // + lineage_db) must be the artifact CONTENT hash, not the bincode-handle
    // hash (which embeds absolute paths) — so recorded provenance is
    // cross-machine stable + consistent with the downstream cache key.
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let h1 = recorded_output_hash("/machine-a/jobs/r1/stages/0-make/out", 7).await;
    let h2 = recorded_output_hash("/machine-b/elsewhere/out", 7).await;
    assert_eq!(
        h1, h2,
        "B.1(a): recorded output_hash stable across abs paths"
    );
    let art = PathArt {
        content: 7,
        path: PathBuf::from("/machine-a/jobs/r1/stages/0-make/out"),
    };
    assert_eq!(h1, art.content_hash(), "recorded hash = content_hash()");
    let erased = ErasedArtifact::from_typed(&art).unwrap();
    assert_ne!(
        h1,
        content_hash_from_erased(&erased),
        "must be the content hash, not the path-embedding handle hash"
    );
    let h3 = recorded_output_hash("/machine-a/jobs/r1/stages/0-make/out", 8).await;
    assert_ne!(h1, h3, "different content must change the recorded hash");
}

// ── FW-2 atomic stage outputs ────────────────────────────────────
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct WriteThenArgs {
    mode: String,
}

struct WriteThen;
#[async_trait]
impl Stage for WriteThen {
    const NAME: &'static str = "write_then";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Counter;
    type Args = WriteThenArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &WriteThenArgs,
    ) -> Result<Counter, StageError> {
        let marker = ctx.stage_dir.join("partial.txt");
        std::fs::write(&marker, b"half-written").map_err(|source| StageError::Io {
            path: marker,
            source,
        })?;
        match args.mode.as_str() {
            "ok" => Ok(Counter { n: 1 }),
            "cancel" => {
                ctx.cancel.cancel();
                Ok(Counter { n: 1 })
            }
            _ => Err(StageError::BadInput("forced mid-stage failure".into())),
        }
    }
}
impl Compatible<LamuTrainerBackend> for WriteThen {}

fn write_then_final_dir(job_dir: &Path) -> PathBuf {
    job_dir.join("stages").join("0-write_then")
}

fn leftover_tmp_dirs(job_dir: &Path) -> Vec<PathBuf> {
    let stages = job_dir.join("stages");
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&stages) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with(".tmp-") {
                out.push(e.path());
            }
        }
    }
    out
}

fn cache_entry_count(job_dir: &Path) -> usize {
    let cache_root = job_dir.join("_cache");
    let mut n = 0;
    if let Ok(rd) = std::fs::read_dir(&cache_root) {
        for e in rd.flatten() {
            if e.path().join("output.bin").exists() {
                n += 1;
            }
        }
    }
    n
}

#[tokio::test]
async fn partial_output_cleaned_on_stage_error() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let td = tempfile::tempdir().unwrap();
    let job_dir = td.path().to_path_buf();
    let ctx = ExecCtx::new(job_dir.clone());
    let plan = Plan::<(), LamuTrainerBackend>::new("fw2-err", serde_json::json!({}))
        .start(WriteThen, WriteThenArgs { mode: "err".into() })
        .finish()
        .into_compiled();
    let r = SequentialExecutor::execute(plan, ctx).await;
    assert!(
        matches!(r, Err(PlanError::StageFailed { .. })),
        "stage must fail"
    );
    let final_dir = write_then_final_dir(&job_dir);
    assert!(!final_dir.exists(), "FW-2: no final dir after error");
    assert!(
        leftover_tmp_dirs(&job_dir).is_empty(),
        "FW-2: tmp removed on error"
    );
    assert_eq!(
        cache_entry_count(&job_dir),
        0,
        "FW-2: failed stage not cached"
    );
}

#[tokio::test]
async fn partial_output_cleaned_on_cancel() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let td = tempfile::tempdir().unwrap();
    let job_dir = td.path().to_path_buf();
    let ctx = ExecCtx::new(job_dir.clone());
    let plan = Plan::<(), LamuTrainerBackend>::new("fw2-cancel", serde_json::json!({}))
        .start(
            WriteThen,
            WriteThenArgs {
                mode: "cancel".into(),
            },
        )
        .finish()
        .into_compiled();
    let r = SequentialExecutor::execute(plan, ctx).await;
    assert!(
        matches!(r, Err(PlanError::Cancelled)),
        "mid-stage cancel → Cancelled, got {r:?}"
    );
    let final_dir = write_then_final_dir(&job_dir);
    assert!(
        !final_dir.exists(),
        "FW-2: cancelled stage leaves no partial"
    );
    assert!(
        leftover_tmp_dirs(&job_dir).is_empty(),
        "FW-2: tmp removed on cancel"
    );
    assert_eq!(
        cache_entry_count(&job_dir),
        0,
        "FW-2: cancelled stage not cached"
    );
}

#[tokio::test]
async fn successful_stage_output_promoted_atomically() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let td = tempfile::tempdir().unwrap();
    let job_dir = td.path().to_path_buf();
    let ctx = ExecCtx::new(job_dir.clone());
    let cache = ctx.cache.clone();
    let plan = Plan::<(), LamuTrainerBackend>::new("fw2-ok", serde_json::json!({}))
        .start(WriteThen, WriteThenArgs { mode: "ok".into() })
        .finish()
        .into_compiled();
    let res = SequentialExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(res.n_cache_misses, 1);
    let final_dir = write_then_final_dir(&job_dir);
    assert!(
        final_dir.join("partial.txt").exists(),
        "promoted output present"
    );
    assert!(
        final_dir.join("output.metadata.json").exists(),
        "sidecar in promoted dir"
    );
    assert!(
        leftover_tmp_dirs(&job_dir).is_empty(),
        "no tmp survives promote"
    );
    assert_eq!(
        cache_entry_count(&job_dir),
        1,
        "FW-2: successful stage cached"
    );
    let ctx2 = ExecCtx::new(td.path().join("job2"));
    let ctx2 = ExecCtx { cache, ..ctx2 };
    let plan2 = Plan::<(), LamuTrainerBackend>::new("fw2-ok", serde_json::json!({}))
        .start(WriteThen, WriteThenArgs { mode: "ok".into() })
        .finish()
        .into_compiled();
    let res2 = SequentialExecutor::execute(plan2, ctx2).await.unwrap();
    assert_eq!(res2.n_cache_hits, 1, "second run hits promoted cache");
    assert_eq!(res2.n_cache_misses, 0);
}

#[tokio::test]
async fn stale_partial_does_not_contaminate_rerun() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let td = tempfile::tempdir().unwrap();
    let job_dir = td.path().to_path_buf();
    let final_dir = write_then_final_dir(&job_dir);
    std::fs::create_dir_all(&final_dir).unwrap();
    std::fs::write(final_dir.join("orphan.txt"), b"stale junk").unwrap();
    let ctx = ExecCtx::new(job_dir.clone());
    let plan = Plan::<(), LamuTrainerBackend>::new("fw2-stale", serde_json::json!({}))
        .start(WriteThen, WriteThenArgs { mode: "ok".into() })
        .finish()
        .into_compiled();
    SequentialExecutor::execute(plan, ctx).await.unwrap();
    assert!(
        final_dir.join("partial.txt").exists(),
        "fresh output present"
    );
    assert!(
        !final_dir.join("orphan.txt").exists(),
        "FW-2: stale orphan gone"
    );
}

// ════════════════════════════════════════════════════════════════
// ParallelExecutor — equivalence, concurrency, fail-fast.
// ════════════════════════════════════════════════════════════════

/// Sums a fork's two `Counter` outputs. Used to build a diamond
/// (fork → two branches → merge).
struct SumTwo;
#[async_trait]
impl Stage for SumTwo {
    const NAME: &'static str = "sum_two";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = (Counter, Counter);
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        input: (Counter, Counter),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        Ok(Counter {
            n: input.0.n + input.1.n,
        })
    }
}
impl Compatible<LamuTrainerBackend> for SumTwo {}

/// Two concurrent `Cpu` stages that each rendezvous on a shared
/// barrier released only when BOTH have begun — proves genuine
/// overlap (would deadlock under sequential execution; guarded by a
/// test timeout). `peak` records the max concurrent live count.
///
/// `BarrierArgs.id` differs per instance so the two stages get
/// DISTINCT cache keys — otherwise the executor's single-flight
/// dedup (correctly) collapses two identical stage+args+input nodes
/// to one, and a barrier-of-2 would deadlock.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct BarrierArgs {
    id: u32,
}
struct Barrier {
    gate: Arc<tokio::sync::Barrier>,
    peak: Arc<std::sync::atomic::AtomicU32>,
    live: Arc<std::sync::atomic::AtomicU32>,
}
#[async_trait]
impl Stage for Barrier {
    const NAME: &'static str = "barrier";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Counter;
    type Output = Counter;
    type Args = BarrierArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        input: Counter,
        _args: &BarrierArgs,
    ) -> Result<Counter, StageError> {
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        self.gate.wait().await;
        self.live.fetch_sub(1, Ordering::SeqCst);
        Ok(Counter { n: input.n })
    }
}
impl Compatible<LamuTrainerBackend> for Barrier {}

#[tokio::test]
async fn parallel_linear_matches_sequential() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, ctx) = fresh_ctx();
    let plan = Plan::<(), LamuTrainerBackend>::new("pl", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(Increment, EmptyArgs)
        .then(Increment, EmptyArgs)
        .finish()
        .into_compiled();
    let result = ParallelExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(result.n_stages, 3);
    assert_eq!(result.n_cache_misses, 3);
    let counter: Counter = result.final_output.unwrap().into_typed().unwrap();
    assert_eq!(counter.n, 3);
}

#[tokio::test]
async fn parallel_diamond_fork_merge() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, ctx) = fresh_ctx();
    // MakeOne(1) → fork(Inc, Inc) → both produce 2 → merge SumTwo = 4.
    let plan = Plan::<(), LamuTrainerBackend>::new("diamond", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .fork(Increment, EmptyArgs, Increment, EmptyArgs)
        .merge(SumTwo, EmptyArgs)
        .finish()
        .into_compiled();
    let result = ParallelExecutor::execute(plan, ctx).await.unwrap();
    let counter: Counter = result.final_output.unwrap().into_typed().unwrap();
    assert_eq!(counter.n, 4, "fork both branches (1→2, 1→2) then sum = 4");
}

#[tokio::test]
async fn parallel_cpu_stages_genuinely_overlap() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let td = tempfile::tempdir().unwrap();
    // Give CPU enough permits for 2 concurrent, but only 1 GPU.
    let ctx = ExecCtx::new(td.path().to_path_buf()).with_resource_limit(Resource::Cpu, 4);
    let peak = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let live = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let gate = Arc::new(tokio::sync::Barrier::new(2));
    let b1 = Barrier {
        gate: gate.clone(),
        peak: peak.clone(),
        live: live.clone(),
    };
    let b2 = Barrier {
        gate: gate.clone(),
        peak: peak.clone(),
        live: live.clone(),
    };
    // MakeOne → fork(b1, b2). Distinct args (id 0/1) → distinct keys
    // → both spawn → both rendezvous on the barrier → must overlap.
    let plan = Plan::<(), LamuTrainerBackend>::new("overlap", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .fork(b1, BarrierArgs { id: 0 }, b2, BarrierArgs { id: 1 })
        .merge(SumTwo, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("parallel CPU stages must overlap (barrier would deadlock if serialized)")
        .unwrap();
    assert_eq!(
        peak.load(Ordering::SeqCst),
        2,
        "two CPU stages must run concurrently"
    );
    let _ = result;
}

/// Records peak concurrency; declares `MEMORY_GIB = 4` so the memory
/// admission can serialize two of these under a tight budget even though
/// CPU permits would allow overlap. No hard barrier (that would deadlock if
/// serialized) — a short sleep makes any overlap observable.
struct MemHog {
    peak: Arc<std::sync::atomic::AtomicU32>,
    live: Arc<std::sync::atomic::AtomicU32>,
}
#[async_trait]
impl Stage for MemHog {
    const NAME: &'static str = "mem_hog";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const MEMORY_GIB: u32 = 4;
    type Input = Counter;
    type Output = Counter;
    type Args = BarrierArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        input: Counter,
        _args: &BarrierArgs,
    ) -> Result<Counter, StageError> {
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        self.live.fetch_sub(1, Ordering::SeqCst);
        Ok(Counter { n: input.n })
    }
}
impl Compatible<LamuTrainerBackend> for MemHog {}

#[tokio::test]
async fn memory_budget_serializes_when_sum_exceeds_box_fit() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let td = tempfile::tempdir().unwrap();
    // CPU permits allow 2 concurrent; the memory budget (4) fits only ONE
    // MEMORY_GIB=4 stage → the two must serialize despite being a fork.
    let peak = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let live = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let ctx = ExecCtx::new(td.path().to_path_buf())
        .with_resource_limit(Resource::Cpu, 4)
        .with_memory_budget(4);
    let h1 = MemHog {
        peak: peak.clone(),
        live: live.clone(),
    };
    let h2 = MemHog {
        peak: peak.clone(),
        live: live.clone(),
    };
    let plan = Plan::<(), LamuTrainerBackend>::new("memgate", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .fork(h1, BarrierArgs { id: 0 }, h2, BarrierArgs { id: 1 })
        .merge(SumTwo, EmptyArgs)
        .finish()
        .into_compiled();
    let result = ParallelExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(
        peak.load(Ordering::SeqCst),
        1,
        "memory budget (4) must serialize two MEMORY_GIB=4 stages (sum 8 > budget)"
    );
    let _ = result;
}

/// A GPU stage that holds `gpu_permits = args.id + 1` permits (so id=1
/// requests 2 GPUs = a DDP job owning the whole 2-GPU pool, id=0 requests
/// 1). Records peak concurrency; a short sleep makes overlap observable.
struct GpuHog {
    peak: Arc<std::sync::atomic::AtomicU32>,
    live: Arc<std::sync::atomic::AtomicU32>,
}
#[async_trait]
impl Stage for GpuHog {
    const NAME: &'static str = "gpu_hog";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    type Input = Counter;
    type Output = Counter;
    type Args = BarrierArgs;
    fn gpu_permits(&self, args: &BarrierArgs) -> u32 {
        // id>=2 → a 2-GPU DDP job; id 0/1 → a 1-GPU cell (distinct ids keep
        // distinct cache keys so the executor doesn't dedup the fork).
        if args.id >= 2 { 2 } else { 1 }
    }
    async fn run(
        &self,
        _ctx: &StageContext,
        input: Counter,
        _args: &BarrierArgs,
    ) -> Result<Counter, StageError> {
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        self.live.fetch_sub(1, Ordering::SeqCst);
        Ok(Counter { n: input.n })
    }
}
impl Compatible<LamuTrainerBackend> for GpuHog {}

#[tokio::test]
async fn ddp_stage_holds_whole_gpu_pool_blocking_single_gpu_cell() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let td = tempfile::tempdir().unwrap();
    // A 2-GPU box. One branch is a DDP job (id=1 → 2 GPU permits = the whole
    // pool); the other is a single-GPU cell (id=0 → 1 permit). The DDP job
    // holding both permits MUST block the single-GPU cell — they serialize.
    let peak = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let live = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let ctx = ExecCtx::new(td.path().to_path_buf())
        .with_resource_limit(Resource::Gpu, 2)
        .with_resource_limit(Resource::Cpu, 4);
    let ddp = GpuHog {
        peak: peak.clone(),
        live: live.clone(),
    };
    let cell = GpuHog {
        peak: peak.clone(),
        live: live.clone(),
    };
    let plan = Plan::<(), LamuTrainerBackend>::new("gpugate", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .fork(ddp, BarrierArgs { id: 2 }, cell, BarrierArgs { id: 0 })
        .merge(SumTwo, EmptyArgs)
        .finish()
        .into_compiled();
    let result = ParallelExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(
        peak.load(Ordering::SeqCst),
        1,
        "a DDP job holding all GPU permits must block the single-GPU cell"
    );
    let _ = result;
}

#[tokio::test]
async fn two_single_gpu_cells_overlap_on_two_gpu_pool() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let td = tempfile::tempdir().unwrap();
    // A 2-GPU box, two single-GPU cells (1 permit each) → they overlap.
    let peak = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let live = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let ctx = ExecCtx::new(td.path().to_path_buf())
        .with_resource_limit(Resource::Gpu, 2)
        .with_resource_limit(Resource::Cpu, 4);
    let a = GpuHog {
        peak: peak.clone(),
        live: live.clone(),
    };
    let b = GpuHog {
        peak: peak.clone(),
        live: live.clone(),
    };
    let plan = Plan::<(), LamuTrainerBackend>::new("gpupair", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .fork(a, BarrierArgs { id: 0 }, b, BarrierArgs { id: 1 })
        .merge(SumTwo, EmptyArgs)
        .finish()
        .into_compiled();
    let result = ParallelExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(
        peak.load(Ordering::SeqCst),
        2,
        "two 1-GPU cells must overlap on a 2-GPU pool"
    );
    let _ = result;
}

#[tokio::test]
async fn parallel_fail_fast_reports_first_error() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, ctx) = fresh_ctx();
    // Two independent branches off MakeOne: one fails, one is a no-op
    // increment. Fail-fast must surface the failure.
    let plan = Plan::<(), LamuTrainerBackend>::new("ff", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .fork(FailMid, EmptyArgs, Increment, EmptyArgs)
        .merge(SumTwo, EmptyArgs)
        .finish()
        .into_compiled();
    let r = ParallelExecutor::execute(plan, ctx).await;
    assert!(
        matches!(r, Err(PlanError::StageFailed { ref stage, .. }) if stage == "fail_mid"),
        "fail-fast must report the failing stage, got {r:?}"
    );
}

/// A stage that consumes a Counter and always fails (for a fork arm).
struct FailMid;
#[async_trait]
impl Stage for FailMid {
    const NAME: &'static str = "fail_mid";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        Err(StageError::BadInput("fork-arm failure".into()))
    }
}
impl Compatible<LamuTrainerBackend> for FailMid {}

#[tokio::test]
async fn executor_equivalence_final_output_and_cache() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // Run the SAME diamond under both executors with a shared cache:
    // the parallel run must hit the cache the sequential run populated
    // (proves identical cache keys), and produce the same output.
    let td = tempfile::tempdir().unwrap();
    let ctx_seq = ExecCtx::new(td.path().join("seq"));
    let cache = ctx_seq.cache.clone();
    let mk = || {
        Plan::<(), LamuTrainerBackend>::new("eq", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .fork(Increment, EmptyArgs, Increment, EmptyArgs)
            .merge(SumTwo, EmptyArgs)
            .finish()
            .into_compiled()
    };
    let seq = SequentialExecutor::execute(mk(), ctx_seq).await.unwrap();
    let seq_out: Counter = seq.final_output.unwrap().into_typed().unwrap();

    let ctx_par = ExecCtx::new(td.path().join("par"));
    let ctx_par = ExecCtx { cache, ..ctx_par };
    let par = ParallelExecutor::execute(mk(), ctx_par).await.unwrap();
    let par_out: Counter = par.final_output.unwrap().into_typed().unwrap();

    assert_eq!(
        seq_out.n, par_out.n,
        "both executors produce the same output"
    );
    assert_eq!(
        par.n_cache_hits, par.n_stages,
        "parallel run must hit the sequential run's cache for every stage (identical keys)"
    );
    assert_eq!(par.n_cache_misses, 0);
}

// ════════════════════════════════════════════════════════════════
// D1 retry + D2 timeout.
// ════════════════════════════════════════════════════════════════

use crate::framework::retry::{Backoff, RetryOn, RetryPolicy, StageTimeout};

static FLAKY_ATTEMPTS: AtomicU32 = AtomicU32::new(0);
static FLAKY_FAILS: AtomicU32 = AtomicU32::new(0);

/// Fails its first `FLAKY_FAILS` attempts (transient Backend error),
/// then succeeds.
struct Flaky;
#[async_trait]
impl Stage for Flaky {
    const NAME: &'static str = "flaky";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 4,
        backoff: Backoff::None,
        retry_on: RetryOn::Transient,
    };
    type Input = ();
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        let n = FLAKY_ATTEMPTS.fetch_add(1, Ordering::SeqCst) + 1;
        if n <= FLAKY_FAILS.load(Ordering::SeqCst) {
            Err(StageError::Backend(anyhow::anyhow!("transient blip #{n}")))
        } else {
            Ok(Counter { n })
        }
    }
}
impl Compatible<LamuTrainerBackend> for Flaky {}

#[tokio::test]
async fn retry_succeeds_after_transient_failures() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    FLAKY_ATTEMPTS.store(0, Ordering::SeqCst);
    FLAKY_FAILS.store(2, Ordering::SeqCst); // fail twice, succeed on #3
    let td = tempfile::tempdir().unwrap();
    let job_dir = td.path().to_path_buf();
    let ctx = ExecCtx::new(job_dir.clone());
    let mut rx = ctx.status.subscribe();
    let plan = Plan::<(), LamuTrainerBackend>::new("retry", serde_json::json!({}))
        .start(Flaky, EmptyArgs)
        .finish()
        .into_compiled();
    let r = SequentialExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(FLAKY_ATTEMPTS.load(Ordering::SeqCst), 3, "ran 3 attempts");
    assert_eq!(r.n_cache_misses, 1);
    assert_eq!(
        cache_entry_count(&job_dir),
        1,
        "only the successful attempt is cached"
    );
    let mut retrying = 0;
    while let Ok(evt) = rx.try_recv() {
        if matches!(evt, StageEvent::StageRetrying { .. }) {
            retrying += 1;
        }
    }
    assert_eq!(retrying, 2, "two retry events for two transient failures");
}

// ── S3 auto-resume: a retry injects --resume from the checkpoint ──
static RESUMABLE_ATTEMPTS: AtomicU32 = AtomicU32::new(0);
static RESUMABLE_SAW_RESUME: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fails transiently on attempt 1 (after writing a `running` resume marker),
/// succeeds on attempt 2 — and asserts the Executor injected `resume_from`
/// (the auto-resume wiring) by then.
struct ResumableFlaky;
impl ResumableFlaky {
    fn resume_dir(ctx: &StageContext) -> std::path::PathBuf {
        ctx.job_dir.join("resume_dir")
    }
}
#[async_trait]
impl Stage for ResumableFlaky {
    const NAME: &'static str = "resumable_flaky";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 2,
        backoff: Backoff::None,
        retry_on: RetryOn::Transient,
    };
    type Input = ();
    type Output = Counter;
    type Args = EmptyArgs;
    fn resume_handle(
        &self,
        ctx: &StageContext,
        _args: &EmptyArgs,
    ) -> Option<crate::framework::resume::ResumeToken> {
        Some(crate::framework::resume::ResumeToken {
            resume_dir: Self::resume_dir(ctx),
            required_keys: &[],
        })
    }
    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        let n = RESUMABLE_ATTEMPTS.fetch_add(1, Ordering::SeqCst) + 1;
        if ctx.attempt == 1 {
            // Write a `running` marker keyed on THIS run's id so the retry
            // (same run_id) resolves to Resume.
            let dir = Self::resume_dir(ctx);
            std::fs::create_dir_all(&dir).unwrap();
            let run_id = ctx
                .job_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let state = crate::framework::resume::ResumeState {
                status: "running".into(),
                run_id,
                pid: 0,
                heartbeat_unix: now,
            };
            std::fs::write(
                dir.join("state.json"),
                serde_json::to_string(&state).unwrap(),
            )
            .unwrap();
            return Err(StageError::OutOfMemory {
                detail: "transient #1".into(),
            });
        }
        // Attempt 2: the executor must have injected the resume dir.
        if ctx.resume_from.as_deref() == Some(Self::resume_dir(ctx).as_path()) {
            RESUMABLE_SAW_RESUME.store(true, Ordering::SeqCst);
        }
        Ok(Counter { n })
    }
}
impl Compatible<LamuTrainerBackend> for ResumableFlaky {}

#[tokio::test]
async fn retry_auto_injects_resume_from_checkpoint() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    RESUMABLE_ATTEMPTS.store(0, Ordering::SeqCst);
    RESUMABLE_SAW_RESUME.store(false, Ordering::SeqCst);
    let td = tempfile::tempdir().unwrap();
    let ctx = ExecCtx::new(td.path().to_path_buf());
    let plan = Plan::<(), LamuTrainerBackend>::new("resume", serde_json::json!({}))
        .start(ResumableFlaky, EmptyArgs)
        .finish()
        .into_compiled();
    let r = SequentialExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(
        RESUMABLE_ATTEMPTS.load(Ordering::SeqCst),
        2,
        "ran twice (fail then resume)"
    );
    assert!(
        RESUMABLE_SAW_RESUME.load(Ordering::SeqCst),
        "attempt 2 must see resume_from = the checkpoint dir (auto-resume wired)"
    );
    let _ = r;
}

static DET_ATTEMPTS: AtomicU32 = AtomicU32::new(0);

/// Always fails with a DETERMINISTIC error (BadInput) — must NOT be
/// retried even under `AllErrors` with `max_attempts=5`.
struct DeterministicFail;
#[async_trait]
impl Stage for DeterministicFail {
    const NAME: &'static str = "deterministic_fail";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 5,
        backoff: Backoff::None,
        retry_on: RetryOn::AllErrors,
    };
    type Input = ();
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        DET_ATTEMPTS.fetch_add(1, Ordering::SeqCst);
        Err(StageError::BadInput("nope".into()))
    }
}
impl Compatible<LamuTrainerBackend> for DeterministicFail {}

#[tokio::test]
async fn deterministic_error_not_retried() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    DET_ATTEMPTS.store(0, Ordering::SeqCst);
    let (_td, ctx) = fresh_ctx();
    let plan = Plan::<(), LamuTrainerBackend>::new("det", serde_json::json!({}))
        .start(DeterministicFail, EmptyArgs)
        .finish()
        .into_compiled();
    let r = SequentialExecutor::execute(plan, ctx).await;
    assert!(matches!(r, Err(PlanError::StageFailed { .. })));
    assert_eq!(
        DET_ATTEMPTS.load(Ordering::SeqCst),
        1,
        "a deterministic BadInput must run exactly once despite max_attempts=5"
    );
}

/// Sleeps far longer than its hard timeout — tests the HARD timeout.
struct SleepForever;
#[async_trait]
impl Stage for SleepForever {
    const NAME: &'static str = "sleep_forever";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const TIMEOUT: StageTimeout = StageTimeout {
        soft: None,
        hard: Some(std::time::Duration::from_millis(100)),
    };
    type Input = ();
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        Ok(Counter { n: 1 })
    }
}
impl Compatible<LamuTrainerBackend> for SleepForever {}

#[tokio::test]
async fn hard_timeout_fails_a_hung_stage() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, ctx) = fresh_ctx();
    let plan = Plan::<(), LamuTrainerBackend>::new("to", serde_json::json!({}))
        .start(SleepForever, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = SequentialExecutor::execute(plan, ctx);
    let r = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("hard timeout must fire well before the test's 5s guard");
    match r {
        Err(PlanError::StageFailed { source, .. }) => {
            assert!(
                matches!(source, StageError::Timeout { .. }),
                "got {source:?}"
            );
        }
        other => panic!("expected StageFailed(Timeout), got {other:?}"),
    }
}

// ════════════════════════════════════════════════════════════════
// #4 dynamic runtime DAG mutation — KILL-on-NaN.
// ════════════════════════════════════════════════════════════════

static DIVERGER_RAN: AtomicU32 = AtomicU32::new(0);
static NAN_OK_RAN: AtomicU32 = AtomicU32::new(0);

/// Emits a NON-FINITE step metric, then PARKS on its own cancel token
/// (which the control watcher fires on a KILL-on-NaN). Declares `Gpu` so a
/// freed-permit assertion after the kill is meaningful — the node holds the
/// single GPU permit the entire time it is parked.
struct Diverger;
#[async_trait]
impl Stage for Diverger {
    const NAME: &'static str = "diverger";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        _input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        DIVERGER_RAN.fetch_add(1, Ordering::SeqCst);
        // A diverged step — KillOnNaN must fire on this.
        let _ = ctx.status_tx.send(StageEvent::StageStep {
            node_idx: ctx.node_idx,
            stage_name: Self::NAME.to_string(),
            update: serde_json::json!({ "loss": "nan", "step": 1 }),
        });
        // Park until the coordinator kills THIS node (its stage cancel
        // fires). Guarded by the test's outer timeout if the kill never
        // arrives (which would itself be the failure).
        ctx.cancel.cancelled().await;
        Err(StageError::Cancelled)
    }
}
impl Compatible<LamuTrainerBackend> for Diverger {}

/// Emits a non-finite step then returns Ok immediately. With NO control
/// policy the metric is inert data — the stage completes normally, proving
/// the watcher is truly off by default (byte-identical to pre-#4).
struct NanThenOk;
#[async_trait]
impl Stage for NanThenOk {
    const NAME: &'static str = "nan_then_ok";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        NAN_OK_RAN.fetch_add(1, Ordering::SeqCst);
        let _ = ctx.status_tx.send(StageEvent::StageStep {
            node_idx: ctx.node_idx,
            stage_name: Self::NAME.to_string(),
            update: serde_json::json!({ "loss": "nan" }),
        });
        Ok(Counter { n: input.n })
    }
}
impl Compatible<LamuTrainerBackend> for NanThenOk {}

#[tokio::test]
async fn divergence_kill_single_attempt_fails_and_frees_gpu() {
    // S1: `Diverger` has the DEFAULT retry (max_attempts=1), so its single
    // attempt diverges and EXHAUSTS retries immediately → a REAL surfaced
    // `Diverged` failure (the deliberate S1 change from the old silent
    // prune). Increment downstream never runs; the GPU permit is freed; the
    // diverged node leaves no promoted stage dir.
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    DIVERGER_RAN.store(0, Ordering::SeqCst);
    INC_RUN_COUNT.store(0, Ordering::SeqCst);
    let td = tempfile::tempdir().unwrap();
    let job_dir = td.path().to_path_buf();
    let ctx = ExecCtx::new(job_dir.clone())
        .with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));
    // Clone the GPU semaphore Arc so we can assert the permit is returned
    // after the diverged node drops it.
    let gpu = ctx.resources[&Resource::Gpu].clone();

    // MakeOne(Cpu) → Diverger(Gpu, diverges) → Increment(Cpu, downstream).
    let plan = Plan::<(), LamuTrainerBackend>::new("kill", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(Diverger, EmptyArgs)
        .then(Increment, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("divergence kill must fire — a parked Diverger would otherwise hang");

    assert_eq!(
        DIVERGER_RAN.load(Ordering::SeqCst),
        1,
        "diverger ran once (max_attempts=1)"
    );
    assert!(
        matches!(
            result,
            Err(PlanError::StageFailed { ref stage, source: StageError::Diverged { .. }, .. })
                if stage == "diverger"
        ),
        "an exhausted divergence is a REAL failure (Diverged), not a silent prune: {result:?}"
    );
    assert_eq!(
        INC_RUN_COUNT.load(Ordering::SeqCst),
        0,
        "downstream Increment must never run (the diverged parent never produced its output)"
    );
    assert_eq!(
        gpu.available_permits(),
        1,
        "the diverged node's GPU permit must be freed"
    );
    // FW-2: a diverged node is NOT promoted → no `<idx>-diverger` stage dir
    // and no leftover tmp.
    let stages = job_dir.join("stages");
    if stages.is_dir() {
        for entry in std::fs::read_dir(&stages).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.contains("diverger"),
                "diverged node left a stage dir: {name}"
            );
        }
    }
}

#[tokio::test]
async fn divergence_kill_lets_concurrent_sibling_finish_before_failfast() {
    // S1: `Diverger` (default retry, max_attempts=1) diverges → exhausts →
    // fails the plan. The CONCURRENT Increment sibling must still run to
    // completion before the plan fails-fast (the kill targets only the
    // diverging node, never the sibling).
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    DIVERGER_RAN.store(0, Ordering::SeqCst);
    INC_RUN_COUNT.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    let ctx = base.with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));

    // MakeOne → fork(Diverger[Gpu], Increment[Cpu]) → merge(SumTwo).
    let plan = Plan::<(), LamuTrainerBackend>::new("kill_fork", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .fork(Diverger, EmptyArgs, Increment, EmptyArgs)
        .merge(SumTwo, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("sibling must finish + divergence kill must fire");

    assert!(
        matches!(
            result,
            Err(PlanError::StageFailed { ref stage, .. }) if stage == "diverger"
        ),
        "the exhausted divergence surfaces a StageFailed: {result:?}"
    );
    assert_eq!(
        INC_RUN_COUNT.load(Ordering::SeqCst),
        1,
        "the sibling Increment branch must run to completion despite the divergence kill"
    );
}

#[tokio::test]
async fn no_control_policy_is_byte_identical_nan_ignored() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    NAN_OK_RAN.store(0, Ordering::SeqCst);
    let (_td, ctx) = fresh_ctx(); // no control policy

    // With no watcher, a non-finite step metric is inert: the stage
    // completes and the plan produces its final output exactly as before.
    let plan = Plan::<(), LamuTrainerBackend>::new("noop", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(NanThenOk, EmptyArgs)
        .finish()
        .into_compiled();
    let result = ParallelExecutor::execute(plan, ctx).await.unwrap();
    assert_eq!(NAN_OK_RAN.load(Ordering::SeqCst), 1);
    let counter: Counter = result.final_output.unwrap().into_typed().unwrap();
    assert_eq!(counter.n, 1, "plan completes normally; NaN metric ignored");
    assert_eq!(result.n_cache_misses, 2);
}

// ════════════════════════════════════════════════════════════════
// S1 / ADR 0044 P7 — divergence kill → RETRYABLE Diverged + auto-resume,
// and the new `Stage::divergence_check` invocation.
// ════════════════════════════════════════════════════════════════

static DIV_RESUME_ATTEMPTS: AtomicU32 = AtomicU32::new(0);
static DIV_RESUME_SAW_RESUME: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// A trainer that DIVERGES once (emits a non-finite step, parks on its
/// cancel token, bails) then SUCCEEDS on the auto-resumed retry. Writes a
/// `running` resume marker on attempt 1 so the executor's S3 `decide_resume`
/// returns Resume on attempt 2 and injects `resume_from`. Mirrors
/// `ResumableFlaky` but the failure mechanism is a DIVERGENCE KILL (the
/// coordinator cancels its token after the NaN step), not a returned error.
struct DivergeThenResume;
impl DivergeThenResume {
    fn resume_dir(ctx: &StageContext) -> std::path::PathBuf {
        ctx.job_dir.join("diverge_resume_dir")
    }
}
#[async_trait]
impl Stage for DivergeThenResume {
    const NAME: &'static str = "diverge_then_resume";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 2,
        backoff: Backoff::None,
        retry_on: RetryOn::Transient,
    };
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    fn resume_handle(
        &self,
        ctx: &StageContext,
        _args: &EmptyArgs,
    ) -> Option<crate::framework::resume::ResumeToken> {
        Some(crate::framework::resume::ResumeToken {
            resume_dir: Self::resume_dir(ctx),
            required_keys: &[],
        })
    }
    async fn run(
        &self,
        ctx: &StageContext,
        input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        DIV_RESUME_ATTEMPTS.fetch_add(1, Ordering::SeqCst);
        if ctx.attempt == 1 {
            // Write a `running` marker keyed on THIS run's id so the retry
            // (same run_id) resolves to Resume.
            let dir = Self::resume_dir(ctx);
            std::fs::create_dir_all(&dir).unwrap();
            let run_id = ctx
                .job_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let state = crate::framework::resume::ResumeState {
                status: "running".into(),
                run_id,
                pid: 0,
                heartbeat_unix: now,
            };
            std::fs::write(
                dir.join("state.json"),
                serde_json::to_string(&state).unwrap(),
            )
            .unwrap();
            // Diverge: emit a non-finite step. The coordinator's KillOnNaN
            // watcher records the divergence + cancels THIS node's token.
            let _ = ctx.status_tx.send(StageEvent::StageStep {
                node_idx: ctx.node_idx,
                stage_name: Self::NAME.to_string(),
                update: serde_json::json!({ "loss": "nan", "step": 1 }),
            });
            // Park until the kill fires, then bail like a SIGTERM'd trainer.
            ctx.cancel.cancelled().await;
            return Err(StageError::Cancelled);
        }
        // Attempt 2: the executor must have injected the resume dir (auto-resume).
        if ctx.resume_from.as_deref() == Some(Self::resume_dir(ctx).as_path()) {
            DIV_RESUME_SAW_RESUME.store(true, Ordering::SeqCst);
        }
        Ok(Counter { n: input.n + 1 })
    }
}
impl Compatible<LamuTrainerBackend> for DivergeThenResume {}

#[tokio::test]
async fn divergence_kill_retries_resumes_and_completes() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    DIV_RESUME_ATTEMPTS.store(0, Ordering::SeqCst);
    DIV_RESUME_SAW_RESUME.store(false, Ordering::SeqCst);
    let td = tempfile::tempdir().unwrap();
    let ctx = ExecCtx::new(td.path().to_path_buf())
        .with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));

    // MakeOne → DivergeThenResume (diverges on attempt 1, resumes on 2).
    let plan = Plan::<(), LamuTrainerBackend>::new("div_resume", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(DivergeThenResume, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("divergence kill must fire + retry — a parked stage would hang")
        .expect("a diverged-then-resumed run RECOVERS → execute returns Ok");

    assert_eq!(
        DIV_RESUME_ATTEMPTS.load(Ordering::SeqCst),
        2,
        "ran twice: diverge (killed) then resumed"
    );
    assert!(
        DIV_RESUME_SAW_RESUME.load(Ordering::SeqCst),
        "attempt 2 must see resume_from = the checkpoint dir (S3 auto-resume reached)"
    );
    let counter: Counter = result.final_output.unwrap().into_typed().unwrap();
    assert_eq!(
        counter.n, 2,
        "the recovered run produces its real output (1 → 2)"
    );
}

static ALWAYS_DIV_ATTEMPTS: AtomicU32 = AtomicU32::new(0);

/// ALWAYS diverges: every attempt emits a non-finite step + parks + bails.
/// Must exhaust its retries then surface a REAL `Diverged` failure — NOT a
/// silent prune.
struct AlwaysDiverge;
#[async_trait]
impl Stage for AlwaysDiverge {
    const NAME: &'static str = "always_diverge";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 3,
        backoff: Backoff::None,
        retry_on: RetryOn::Transient,
    };
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        _input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        ALWAYS_DIV_ATTEMPTS.fetch_add(1, Ordering::SeqCst);
        let _ = ctx.status_tx.send(StageEvent::StageStep {
            node_idx: ctx.node_idx,
            stage_name: Self::NAME.to_string(),
            update: serde_json::json!({ "loss": "nan" }),
        });
        ctx.cancel.cancelled().await;
        Err(StageError::Cancelled)
    }
}
impl Compatible<LamuTrainerBackend> for AlwaysDiverge {}

#[tokio::test]
async fn always_diverges_exhausts_retries_then_fails_not_pruned() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    ALWAYS_DIV_ATTEMPTS.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    let ctx = base.with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));

    let plan = Plan::<(), LamuTrainerBackend>::new("always_div", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(AlwaysDiverge, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), fut)
        .await
        .expect("repeated divergence must keep firing the kill + bounded retry");

    // Terminal: a REAL surfaced failure (NOT a silent Ok-with-pruned-branch).
    match result {
        Err(PlanError::StageFailed { stage, source, .. }) => {
            assert_eq!(stage, "always_diverge");
            assert!(
                matches!(source, StageError::Diverged { .. }),
                "exhausted divergence must surface StageError::Diverged, got {source:?}"
            );
        }
        other => panic!("expected StageFailed(Diverged), got {other:?}"),
    }
    assert_eq!(
        ALWAYS_DIV_ATTEMPTS.load(Ordering::SeqCst),
        3,
        "diverged node retries up to max_attempts (3) before the terminal failure"
    );
}

// ════════════════════════════════════════════════════════════════
// S1 RACE — stale buffered divergence steps must NOT re-kill the retry.
//
// A REAL diverging trainer emits MANY `{"loss":"nan"}` StageSteps before its
// killpg teardown reaps the subprocess. Those steps buffer in the lossy
// broadcast and arrive at the coordinator AFTER `run_node` promoted the kill
// to `Diverged`, re-armed the slot with a FRESH token, and started attempt 2.
// WITHOUT the `kill_flagged` latch a stale step finds the fresh (un-cancelled)
// token, slips past the idempotency guard, and cancels it — spuriously killing
// attempt 2. These two tests emit ≥3 NaN steps per attempt to exercise that
// window (the prior single-NaN-step tests never could).
// ════════════════════════════════════════════════════════════════

static MULTI_DIV_RESUME_ATTEMPTS: AtomicU32 = AtomicU32::new(0);
static MULTI_DIV_RESUME_SAW_RESUME: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Like `DivergeThenResume`, but models a REAL diverging trainer faithfully:
/// it emits a BURST of non-finite steps (all flushed to the broadcast BEFORE
/// the stage future returns — exactly as the lamu backend's `run` joins its
/// stdout reader via `stdout_reader.await` before returning, so every
/// `StageStep` precedes the worker's `StageRetrying`), then parks + bails. The
/// burst is long (≥3 is the spec; the extra steps just make the coordinator's
/// drain lag the worker's re-arm reliably, so the race is hit on every run).
///
/// The race (needs the multi-thread runtime, hence the test's `multi_thread`
/// flavor): NaN#1 fires the kill (latch set); while the coordinator still has
/// NaN#2/#3 BUFFERED and undrained, the worker — on the other thread — bails,
/// re-arms the slot with a FRESH token, and emits `StageRetrying`. WITHOUT the
/// `kill_flagged` latch the coordinator then drains NaN#2 against that fresh
/// (un-cancelled) token, slips past the `is_cancelled()` guard, and re-fires
/// the kill → attempt 2 is spuriously killed → exhausted → the plan FAILS.
/// WITH the latch, NaN#2/#3 short-circuit at the `kill_flagged` guard until
/// the node's `StageRetrying` clears it (and the single broadcast guarantees
/// all of attempt-1's NaN steps precede that `StageRetrying`).
///
/// Attempt 2 sleeps a beat (watching its own cancel token) so a pre-fix
/// re-kill lands mid-run: the re-kill cancels it → it bails → a 3rd attempt
/// runs → the attempt-count assertion (== 2) trips.
struct MultiStepDivergeThenResume;
impl MultiStepDivergeThenResume {
    fn resume_dir(ctx: &StageContext) -> std::path::PathBuf {
        ctx.job_dir.join("multi_diverge_resume_dir")
    }
}
#[async_trait]
impl Stage for MultiStepDivergeThenResume {
    const NAME: &'static str = "multi_step_diverge_then_resume";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 2,
        backoff: Backoff::None,
        retry_on: RetryOn::Transient,
    };
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    fn resume_handle(
        &self,
        ctx: &StageContext,
        _args: &EmptyArgs,
    ) -> Option<crate::framework::resume::ResumeToken> {
        Some(crate::framework::resume::ResumeToken {
            resume_dir: Self::resume_dir(ctx),
            required_keys: &[],
        })
    }
    async fn run(
        &self,
        ctx: &StageContext,
        input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        MULTI_DIV_RESUME_ATTEMPTS.fetch_add(1, Ordering::SeqCst);
        if ctx.attempt == 1 {
            // Write a `running` marker so the retry auto-resumes (S3).
            let dir = Self::resume_dir(ctx);
            std::fs::create_dir_all(&dir).unwrap();
            let run_id = ctx
                .job_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let state = crate::framework::resume::ResumeState {
                status: "running".into(),
                run_id,
                pid: 0,
                heartbeat_unix: now,
            };
            std::fs::write(
                dir.join("state.json"),
                serde_json::to_string(&state).unwrap(),
            )
            .unwrap();
            // Burst of NaN steps, all emitted BEFORE this future returns
            // (faithful to the joined stdout reader). They land in the
            // broadcast in order; the steps after the first are the stale
            // buffered steps the coordinator may not drain until after the
            // re-arm. A long burst widens the window: while the coordinator
            // works through the queue, the worker (other thread) bails +
            // re-arms, so a later step is drained against the FRESH token. ≥3
            // satisfies the spec; the extra steps just make the race reliable.
            for step in 1..=24u32 {
                let _ = ctx.status_tx.send(StageEvent::StageStep {
                    node_idx: ctx.node_idx,
                    stage_name: Self::NAME.to_string(),
                    update: serde_json::json!({ "loss": "nan", "step": step }),
                });
            }
            // Park until the kill fires (NaN#1), then bail like a SIGTERM'd
            // trainer. The return → re-arm → StageRetrying happens while the
            // coordinator may still have NaN#2/#3 buffered.
            ctx.cancel.cancelled().await;
            return Err(StageError::Cancelled);
        }
        // Attempt 2: must have been auto-resumed AND not spuriously re-killed.
        if ctx.resume_from.as_deref() == Some(Self::resume_dir(ctx).as_path()) {
            MULTI_DIV_RESUME_SAW_RESUME.store(true, Ordering::SeqCst);
        }
        // Stay alive a beat, watching our own cancel token, so a pre-fix
        // re-kill (from a stale buffered NaN step) lands mid-run → bail → a
        // 3rd attempt → the attempt-count assertion (== 2) trips. With the
        // fix nothing cancels us and we return Ok.
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_millis(200)) => {}
            _ = ctx.cancel.cancelled() => {
                return Err(StageError::Cancelled);
            }
        }
        Ok(Counter { n: input.n + 1 })
    }
}
impl Compatible<LamuTrainerBackend> for MultiStepDivergeThenResume {}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_step_divergence_burst_does_not_re_kill_the_retry() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    MULTI_DIV_RESUME_ATTEMPTS.store(0, Ordering::SeqCst);
    MULTI_DIV_RESUME_SAW_RESUME.store(false, Ordering::SeqCst);
    let td = tempfile::tempdir().unwrap();
    let ctx = ExecCtx::new(td.path().to_path_buf())
        .with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));

    // MakeOne → MultiStepDivergeThenResume (3 NaN steps on attempt 1, resumes).
    let plan = Plan::<(), LamuTrainerBackend>::new("multi_div_resume", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(MultiStepDivergeThenResume, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(8), fut)
        .await
        .expect("divergence kill must fire + retry — a parked stage would hang")
        // THE BUG: without the latch, stale step #2/#3 cancel attempt 2's
        // fresh token → attempt 2 is killed → exhausted → `Err(Diverged)`.
        // With the latch, attempt 2 runs clean and the plan RECOVERS.
        .expect("a multi-NaN-step diverge must still recover (attempt 2 not re-killed)");

    assert_eq!(
        MULTI_DIV_RESUME_ATTEMPTS.load(Ordering::SeqCst),
        2,
        "ran exactly twice: attempt 1 diverged (burst of NaN), attempt 2 succeeded — \
         the stale buffered steps must NOT inflate the attempt count by re-killing"
    );
    assert!(
        MULTI_DIV_RESUME_SAW_RESUME.load(Ordering::SeqCst),
        "attempt 2 must see resume_from (auto-resume reached) and run to completion"
    );
    let counter: Counter = result.final_output.unwrap().into_typed().unwrap();
    assert_eq!(
        counter.n, 2,
        "the recovered run produces its real output (1 → 2)"
    );
}

static MULTI_ALWAYS_DIV_ATTEMPTS: AtomicU32 = AtomicU32::new(0);

/// ALWAYS diverges, emitting a long BURST of NaN steps every attempt (≥3 is
/// the spec; 24 makes the coordinator's drain lag the worker's re-arm reliably
/// — same rationale as `MultiStepDivergeThenResume`). The extra stale steps
/// must neither INFLATE the count (re-kill a fresh attempt with a straggler so
/// the run burns attempts faster than it should) nor DEFLATE it — it must
/// terminate as `Diverged` after EXACTLY `max_attempts`.
struct MultiStepAlwaysDiverge;
#[async_trait]
impl Stage for MultiStepAlwaysDiverge {
    const NAME: &'static str = "multi_step_always_diverge";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 3,
        backoff: Backoff::None,
        retry_on: RetryOn::Transient,
    };
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        _input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        MULTI_ALWAYS_DIV_ATTEMPTS.fetch_add(1, Ordering::SeqCst);
        for step in 1..=24u32 {
            let _ = ctx.status_tx.send(StageEvent::StageStep {
                node_idx: ctx.node_idx,
                stage_name: Self::NAME.to_string(),
                update: serde_json::json!({ "loss": "nan", "step": step }),
            });
        }
        ctx.cancel.cancelled().await;
        Err(StageError::Cancelled)
    }
}
impl Compatible<LamuTrainerBackend> for MultiStepAlwaysDiverge {}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_step_always_diverge_exhausts_exactly_max_attempts() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    MULTI_ALWAYS_DIV_ATTEMPTS.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    let ctx = base.with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));

    let plan = Plan::<(), LamuTrainerBackend>::new("multi_always_div", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(MultiStepAlwaysDiverge, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), fut)
        .await
        .expect("repeated multi-step divergence must keep firing the kill + bounded retry");

    match result {
        Err(PlanError::StageFailed { stage, source, .. }) => {
            assert_eq!(stage, "multi_step_always_diverge");
            assert!(
                matches!(source, StageError::Diverged { .. }),
                "exhausted divergence must surface StageError::Diverged, got {source:?}"
            );
        }
        other => panic!("expected StageFailed(Diverged), got {other:?}"),
    }
    assert_eq!(
        MULTI_ALWAYS_DIV_ATTEMPTS.load(Ordering::SeqCst),
        3,
        "the burst of stale NaN steps must NOT inflate or deflate the count — \
         exactly max_attempts (3) attempts, then terminal Diverged"
    );
}

static TARGETED_KILL_RAN: AtomicU32 = AtomicU32::new(0);

/// Under S1 every `KillBranch` is a DIVERGENCE kill (it populates the
/// registry), so the only PLAIN (non-divergence) cancel reachable through the
/// public API is a plan cancel. This node parks WITHOUT emitting a divergence
/// step, so a plan cancel leaves the registry empty → the prior fail-fast
/// `Cancelled` (no retry) path, byte-identical to before S1.
struct SlowOnce;
#[async_trait]
impl Stage for SlowOnce {
    const NAME: &'static str = "slow_once";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 3,
        backoff: Backoff::None,
        retry_on: RetryOn::AllErrors,
    };
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        _input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        TARGETED_KILL_RAN.fetch_add(1, Ordering::SeqCst);
        // Park on the (plan-rooted) cancel token; bail when it fires. No
        // divergence step emitted → no registry entry → a PLAIN cancel.
        ctx.cancel.cancelled().await;
        Err(StageError::Cancelled)
    }
}
impl Compatible<LamuTrainerBackend> for SlowOnce {}

#[tokio::test]
async fn plan_cancel_of_non_diverging_node_does_not_retry() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    TARGETED_KILL_RAN.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    // Control policy ON (so the registry exists), but the node never diverges.
    let ctx = base.with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));
    let cancel = ctx.cancel.clone();

    let plan = Plan::<(), LamuTrainerBackend>::new("plain_cancel", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(SlowOnce, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    // Fire a plan cancel shortly after launch — SlowOnce is parked, not diverged.
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        cancel.cancel();
    });
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("plan cancel must unwind promptly");
    assert!(
        matches!(result, Err(PlanError::Cancelled)),
        "a plan cancel of a non-diverged node is Cancelled (fail-fast), not retried/Diverged"
    );
    assert_eq!(
        TARGETED_KILL_RAN.load(Ordering::SeqCst),
        1,
        "a plain (non-divergence) cancel must NOT retry — runs exactly once"
    );
}

static DCHECK_RAN: AtomicU32 = AtomicU32::new(0);

/// Emits a BENIGN-looking step (finite loss) but overrides
/// `divergence_check` to return true on it — proving the coordinator now
/// INVOKES `divergence_check` (previously dead code). Diverges once then
/// succeeds on the (plain, non-resumable) retry.
struct DivergenceCheckStage;
#[async_trait]
impl Stage for DivergenceCheckStage {
    const NAME: &'static str = "divergence_check_stage";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 2,
        backoff: Backoff::None,
        retry_on: RetryOn::Transient,
    };
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    fn divergence_check(&self, step: &serde_json::Value) -> bool {
        // Fire on a FINITE metric KillOnNaN would let pass → only the
        // divergence_check invocation can produce this kill.
        step.get("loss")
            .and_then(|v| v.as_f64())
            .is_some_and(|l| l > 100.0)
    }
    async fn run(
        &self,
        ctx: &StageContext,
        input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        let attempt = DCHECK_RAN.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt == 1 {
            // A BENIGN-looking (finite) but huge loss — KillOnNaN ignores it;
            // only divergence_check fires.
            let _ = ctx.status_tx.send(StageEvent::StageStep {
                node_idx: ctx.node_idx,
                stage_name: Self::NAME.to_string(),
                update: serde_json::json!({ "loss": 9999.0, "step": 1 }),
            });
            ctx.cancel.cancelled().await;
            return Err(StageError::Cancelled);
        }
        Ok(Counter { n: input.n + 5 })
    }
}
impl Compatible<LamuTrainerBackend> for DivergenceCheckStage {}

#[tokio::test]
async fn divergence_check_override_fires_kill_and_retry() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    DCHECK_RAN.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    let ctx = base.with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));

    let plan = Plan::<(), LamuTrainerBackend>::new("dcheck", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(DivergenceCheckStage, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("divergence_check kill must fire + retry")
        .expect("diverged-then-recovered → Ok");

    assert_eq!(
        DCHECK_RAN.load(Ordering::SeqCst),
        2,
        "divergence_check fired the kill on attempt 1 (finite metric KillOnNaN ignores), retried"
    );
    let counter: Counter = result.final_output.unwrap().into_typed().unwrap();
    assert_eq!(counter.n, 6, "recovered run produces its output (1 → 6)");
}

static SIB_DIV_RAN: AtomicU32 = AtomicU32::new(0);
static SIB_OK_RAN: AtomicU32 = AtomicU32::new(0);

/// A sibling that runs normally (no divergence) — must complete even when a
/// concurrent sibling is divergence-killed.
struct QuietSibling;
#[async_trait]
impl Stage for QuietSibling {
    const NAME: &'static str = "quiet_sibling";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        SIB_OK_RAN.fetch_add(1, Ordering::SeqCst);
        Ok(Counter { n: input.n + 10 })
    }
}
impl Compatible<LamuTrainerBackend> for QuietSibling {}

/// ALWAYS diverges on a `Gpu` so it is concurrent with the `Cpu` sibling and
/// (after exhausting retries) fails the plan — proving the kill targets ONLY
/// the diverging node, never its sibling.
struct AlwaysDivergeGpu;
#[async_trait]
impl Stage for AlwaysDivergeGpu {
    const NAME: &'static str = "always_diverge_gpu";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const RETRY: RetryPolicy = RetryPolicy {
        max_attempts: 2,
        backoff: Backoff::None,
        retry_on: RetryOn::Transient,
    };
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        _input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        SIB_DIV_RAN.fetch_add(1, Ordering::SeqCst);
        let _ = ctx.status_tx.send(StageEvent::StageStep {
            node_idx: ctx.node_idx,
            stage_name: Self::NAME.to_string(),
            update: serde_json::json!({ "loss": "nan" }),
        });
        ctx.cancel.cancelled().await;
        Err(StageError::Cancelled)
    }
}
impl Compatible<LamuTrainerBackend> for AlwaysDivergeGpu {}

#[tokio::test]
async fn divergence_kill_does_not_affect_concurrent_sibling() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    SIB_DIV_RAN.store(0, Ordering::SeqCst);
    SIB_OK_RAN.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    let ctx = base.with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));

    // MakeOne → fork(AlwaysDivergeGpu[Gpu], QuietSibling[Cpu]) → merge(SumTwo).
    // The Gpu branch diverges (killed, retried, then fails the plan); the Cpu
    // sibling runs CONCURRENTLY and must complete before the plan fails-fast.
    let plan = Plan::<(), LamuTrainerBackend>::new("sib_div", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .fork(AlwaysDivergeGpu, EmptyArgs, QuietSibling, EmptyArgs)
        .merge(SumTwo, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), fut)
        .await
        .expect("sibling must finish; diverging branch must exhaust + fail");

    // The plan fails on the exhausted divergence (a REAL failure now).
    assert!(
        matches!(
            result,
            Err(PlanError::StageFailed { ref stage, .. }) if stage == "always_diverge_gpu"
        ),
        "the exhausted-divergence branch surfaces a StageFailed, got {result:?}"
    );
    assert_eq!(
        SIB_OK_RAN.load(Ordering::SeqCst),
        1,
        "the concurrent QUIET sibling must run to completion despite the divergence kill"
    );
    assert_eq!(
        SIB_DIV_RAN.load(Ordering::SeqCst),
        2,
        "the diverging node ran its max_attempts (2), never touching the sibling"
    );
}

// ── #4 runtime Spawn (PBT/TPE) ──────────────────────────────────────
static SPAWN_MARKER_RAN: AtomicU32 = AtomicU32::new(0);
static SPAWN_CHILD_RAN: AtomicU32 = AtomicU32::new(0);

/// A spawned sub-plan ROOT (Input = ()): increments a counter so a test can
/// prove an injected node actually executed.
struct SpawnMarker;
#[async_trait]
impl Stage for SpawnMarker {
    const NAME: &'static str = "spawn_marker";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        SPAWN_MARKER_RAN.fetch_add(1, Ordering::SeqCst);
        Ok(Counter { n: 7 })
    }
}
impl Compatible<LamuTrainerBackend> for SpawnMarker {}

/// A spawned sub-plan CHILD (depends on the root) — proves intra-delta edges
/// + the successor-decrement path work for injected nodes.
struct SpawnChild;
#[async_trait]
impl Stage for SpawnChild {
    const NAME: &'static str = "spawn_child";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        SPAWN_CHILD_RAN.fetch_add(1, Ordering::SeqCst);
        Ok(Counter { n: input.n })
    }
}
impl Compatible<LamuTrainerBackend> for SpawnChild {}

/// Emits ONE benign step then sleeps briefly before completing — stays
/// in-flight long enough for the coordinator's watcher to read the step
/// (and the policy to queue its Spawn) before this node joins.
struct StepThenSleep;
#[async_trait]
impl Stage for StepThenSleep {
    const NAME: &'static str = "step_then_sleep";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        let _ = ctx.status_tx.send(StageEvent::StageStep {
            node_idx: ctx.node_idx,
            stage_name: Self::NAME.to_string(),
            update: serde_json::json!({ "loss": 0.5, "step": 1 }),
        });
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        Ok(Counter { n: input.n })
    }
}
impl Compatible<LamuTrainerBackend> for StepThenSleep {}

/// Emits N benign steps (each followed by a short sleep) then completes —
/// drives a spawn-every-step policy for the bounded-termination test.
struct MultiStepEmitter {
    steps: u32,
}
#[async_trait]
impl Stage for MultiStepEmitter {
    const NAME: &'static str = "multi_step_emitter";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Counter;
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        ctx: &StageContext,
        input: Counter,
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        for s in 0..self.steps {
            let _ = ctx.status_tx.send(StageEvent::StageStep {
                node_idx: ctx.node_idx,
                stage_name: Self::NAME.to_string(),
                update: serde_json::json!({ "loss": 0.5, "step": s }),
            });
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        }
        Ok(Counter { n: input.n })
    }
}
impl Compatible<LamuTrainerBackend> for MultiStepEmitter {}

/// Builds a fresh single- or two-node sub-plan to inject. Kept here so both
/// policies share one compile path.
fn spawn_subplan(two_node: bool) -> CompiledPlan {
    let p = Plan::<(), LamuTrainerBackend>::new("spawned", serde_json::json!({}))
        .start(SpawnMarker, EmptyArgs);
    if two_node {
        p.then(SpawnChild, EmptyArgs).finish().into_compiled()
    } else {
        p.finish().into_compiled()
    }
}

/// Spawns a two-node sub-plan on the FIRST step, then never again.
struct SpawnOnce {
    fired: std::sync::atomic::AtomicBool,
}
impl crate::framework::control::ControlPolicy for SpawnOnce {
    fn on_step(&self, _m: &StepMetrics) -> Control {
        if self.fired.swap(true, Ordering::SeqCst) {
            return Control::Continue;
        }
        Control::Spawn(Box::new(crate::framework::control::SpawnDelta::new(
            spawn_subplan(true),
            Some("child".into()),
        )))
    }
}

/// Spawns a single-node sub-plan on EVERY step — drives the bounded
/// termination test (spawned nodes emit no steps, so it converges).
struct SpawnEveryStep;
impl crate::framework::control::ControlPolicy for SpawnEveryStep {
    fn on_step(&self, _m: &StepMetrics) -> Control {
        Control::Spawn(Box::new(crate::framework::control::SpawnDelta::new(
            spawn_subplan(false),
            None,
        )))
    }
}

#[tokio::test]
async fn spawn_injects_subplan_and_runs_to_completion() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
    SPAWN_MARKER_RAN.store(0, Ordering::SeqCst);
    SPAWN_CHILD_RAN.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    let ctx = base.with_control(std::sync::Arc::new(SpawnOnce {
        fired: std::sync::atomic::AtomicBool::new(false),
    }));

    // MakeOne → StepThenSleep (emits a step → policy spawns SpawnMarker →
    // SpawnChild). The injected 2-node sub-plan must run to completion.
    let plan = Plan::<(), LamuTrainerBackend>::new("spawn_main", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(StepThenSleep, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("spawn run must terminate")
        .expect("a spawn is not a failure → Ok");

    assert_eq!(
        SPAWN_MARKER_RAN.load(Ordering::SeqCst),
        1,
        "injected root ran"
    );
    assert_eq!(
        SPAWN_CHILD_RAN.load(Ordering::SeqCst),
        1,
        "injected child ran"
    );
    // 2 base nodes + 2 spawned = 4 accounted (the in-test debug_assert in
    // execute() would have panicked on an accounting imbalance).
    assert_eq!(
        result.n_stages, 4,
        "order grew to include the spawned nodes"
    );
}

#[tokio::test]
async fn repeated_spawns_stay_bounded_and_terminate() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    SPAWN_MARKER_RAN.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    let ctx = base.with_control(std::sync::Arc::new(SpawnEveryStep));

    // The emitter fires 4 steps; the policy spawns on each. Spawned nodes
    // emit NO steps, so the graph converges — the run MUST terminate, and
    // the spawn count is bounded by the steps actually observed.
    let plan = Plan::<(), LamuTrainerBackend>::new("spawn_many", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(MultiStepEmitter { steps: 4 }, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), fut)
        .await
        .expect("repeated spawns must still terminate (no infinite loop)")
        .expect("spawns are not failures → Ok");

    // At least one spawn ran. (The injected single-node sub-plans share a
    // cache key, so duplicates cache-HIT rather than re-run — exactly once
    // executes its body; the rest are skipped. The point of THIS test is
    // termination + boundedness, not distinct execution — that is covered by
    // `spawn_injects_subplan_and_runs_to_completion`.)
    assert!(
        SPAWN_MARKER_RAN.load(Ordering::SeqCst) >= 1,
        "at least one spawn ran"
    );
    // Bounded: 2 base nodes + at most one injected per observed step (≤ 4).
    // The in-test debug_assert in execute() already proved completed+pruned
    // balanced the (grown) order, so no node leaked.
    assert!(
        (3..=6).contains(&result.n_stages),
        "spawn count bounded, no runaway: n_stages={}",
        result.n_stages
    );
}

// --- CompositePolicy integration (B2): safety layered under a spawner ---

#[tokio::test]
async fn composite_kills_nan_under_hpo_spawner() {
    // [KillOnNaN, SpawnEveryStep] — the diverging node emits NaN; KillOnNaN
    // must fire FIRST and short-circuit, so the spawner is never consulted
    // on the kill step (no SpawnMarker injected) and the branch is killed.
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    ALWAYS_DIV_ATTEMPTS.store(0, Ordering::SeqCst);
    SPAWN_MARKER_RAN.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    let comp = crate::framework::control::CompositePolicy::new(vec![
        std::sync::Arc::new(crate::framework::control::KillOnNaN),
        std::sync::Arc::new(SpawnEveryStep),
    ]);
    let ctx = base.with_control(std::sync::Arc::new(comp));

    let plan = Plan::<(), LamuTrainerBackend>::new("comp_nan", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(AlwaysDiverge, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), fut)
        .await
        .expect("composite kill must keep firing + bounded retry");

    // KillOnNaN wins: the diverging node surfaces a terminal failure.
    match result {
        Err(PlanError::StageFailed { stage, source, .. }) => {
            assert_eq!(stage, "always_diverge");
            assert!(
                matches!(source, StageError::Diverged { .. }),
                "composite must surface StageError::Diverged, got {source:?}"
            );
        }
        other => panic!("expected StageFailed(Diverged), got {other:?}"),
    }
    // The spawner was short-circuited on every NaN step — no sub-plan ran.
    assert_eq!(
        SPAWN_MARKER_RAN.load(Ordering::SeqCst),
        0,
        "KillOnNaN short-circuited the spawner; no spawn leaked"
    );
}

#[tokio::test]
async fn composite_spawns_when_finite() {
    // [KillOnNaN, SpawnOnce] — finite steps, so KillOnNaN continues and the
    // inner spawner's injection runs to completion (composition does not
    // suppress legitimate spawns).
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
    SPAWN_MARKER_RAN.store(0, Ordering::SeqCst);
    SPAWN_CHILD_RAN.store(0, Ordering::SeqCst);
    let (_td, base) = fresh_ctx();
    let comp = crate::framework::control::CompositePolicy::new(vec![
        std::sync::Arc::new(crate::framework::control::KillOnNaN),
        std::sync::Arc::new(SpawnOnce {
            fired: std::sync::atomic::AtomicBool::new(false),
        }),
    ]);
    let ctx = base.with_control(std::sync::Arc::new(comp));

    let plan = Plan::<(), LamuTrainerBackend>::new("comp_spawn", serde_json::json!({}))
        .start(MakeOne, EmptyArgs)
        .then(StepThenSleep, EmptyArgs)
        .finish()
        .into_compiled();
    let fut = ParallelExecutor::execute(plan, ctx);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("composite spawn run must terminate")
        .expect("a spawn is not a failure → Ok");

    assert_eq!(
        SPAWN_MARKER_RAN.load(Ordering::SeqCst),
        1,
        "injected root ran"
    );
    assert_eq!(
        SPAWN_CHILD_RAN.load(Ordering::SeqCst),
        1,
        "injected child ran"
    );
    assert_eq!(
        result.n_stages, 4,
        "order grew to include the spawned nodes"
    );
}

// ── P2P dispatch (audit findings 1 & 2) ─────────────────────────────
//
// Pre-fix, a P2P-dispatched node's completion poll loop was a detached
// `tokio::spawn` that bumped `in_flight` but was never a member of the
// `JoinSet` the coordinator actually awaits via `join.join_next()`. Once
// every ready node was dispatched, the JoinSet went empty and
// `join_next()` returned `None` immediately — ending the coordinator
// loop while the remote work was still running (finding 1), and a
// remote `JobState::Failed` never touched `first_error`/`env.cancel`, so
// an explicit remote failure could not fail the plan (finding 2). The
// fix makes the poll loop itself a `join.spawn`-ed task that produces a
// real `Result<NodeOutcome, NodeFailure>`, so both properties are
// enforced by the SAME machinery a local node uses.
//
// These mocks stand in for `p2p::coordinator::Coordinator` (the real
// `DispatchSubmitter`/`DispatchHandle` impls, in `src/p2p/coordinator.rs`,
// outside this file's scope) without needing a live peer connection.
#[cfg(feature = "p2p")]
struct MockDispatchPolicy {
    dispatchable: &'static str,
}
#[cfg(feature = "p2p")]
impl crate::p2p::dispatch::DispatchPolicy for MockDispatchPolicy {
    fn is_dispatchable(&self, stage_name: &str) -> bool {
        stage_name == self.dispatchable
    }
    fn classify_stage(
        &self,
        _stage_name: &str,
        _args: &serde_json::Value,
    ) -> crate::p2p::trust::DataClass {
        crate::p2p::trust::DataClass::Public
    }
    fn select_peer(
        &self,
        _stage_name: &str,
        _resources: &crate::p2p::task::ResourceRequest,
        _data_class: crate::p2p::trust::DataClass,
        _peers: &[crate::p2p::peer::PeerInfo],
    ) -> Option<crate::p2p::peer::PeerId> {
        // Peer selection is the coordinator's async dispatch loop (p2p/
        // coordinator.rs), never called on the executor's `submit` path
        // these tests exercise.
        unimplemented!("not exercised by the executor dispatch path")
    }
    fn verify_result(
        &self,
        _result: &crate::p2p::task::TaskResult,
        _expected: &ContentHash,
        _peer_pubkey: &ed25519_dalek::VerifyingKey,
    ) -> crate::p2p::dispatch::DispatchVerdict {
        unimplemented!("not exercised by the executor dispatch path")
    }
}

/// Terminal state a [`MockDispatchHandle`] settles into after
/// `polls_before_terminal` `Ok(None)` ("still running") answers.
#[cfg(feature = "p2p")]
#[derive(Clone)]
enum MockTerminal {
    Succeeded,
    Failed(String),
}

#[cfg(feature = "p2p")]
struct MockDispatchHandle {
    polls_remaining: std::sync::atomic::AtomicU32,
    terminal: MockTerminal,
    poll_count: Arc<AtomicU32>,
}
#[cfg(feature = "p2p")]
impl DispatchHandle for MockDispatchHandle {
    fn poll(&self) -> Result<Option<JobState>, crate::error::TrainError> {
        self.poll_count.fetch_add(1, Ordering::SeqCst);
        let still_running = self
            .polls_remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                if n == 0 { None } else { Some(n - 1) }
            })
            .is_ok();
        if still_running {
            return Ok(None);
        }
        Ok(Some(match &self.terminal {
            MockTerminal::Succeeded => JobState::Succeeded,
            MockTerminal::Failed(reason) => JobState::Failed(reason.clone()),
        }))
    }
    fn cancel(&self) -> Result<(), crate::error::TrainError> {
        Ok(())
    }
}

/// Submits every dispatchable node to a [`MockDispatchHandle`]. On a
/// `Succeeded` terminal it ALSO pre-populates `cache` under the
/// request's `expected_output_hash` — standing in for the P2P data
/// plane having already landed the peer's output bytes by the time the
/// job goes terminal, which is what the fixed dispatch-success arm now
/// relies on (`cache.lookup(key)` in the executor's P2P dispatch block).
#[cfg(feature = "p2p")]
struct MockDispatchSubmitter {
    cache: Arc<CacheHandle>,
    polls_before_terminal: u32,
    terminal: MockTerminal,
    succeed_with: Counter,
    poll_count: Arc<AtomicU32>,
    submit_count: Arc<AtomicU32>,
}
#[cfg(feature = "p2p")]
impl DispatchSubmitter for MockDispatchSubmitter {
    fn submit(
        &self,
        request: DispatchRequest<'_>,
    ) -> Result<Box<dyn DispatchHandle>, crate::error::TrainError> {
        self.submit_count.fetch_add(1, Ordering::SeqCst);
        if matches!(self.terminal, MockTerminal::Succeeded) {
            let art = ErasedArtifact::from_typed(&self.succeed_with).unwrap();
            self.cache
                .insert(request.expected_output_hash, &art)
                .expect("mock cache insert");
        }
        Ok(Box::new(MockDispatchHandle {
            polls_remaining: std::sync::atomic::AtomicU32::new(self.polls_before_terminal),
            terminal: self.terminal.clone(),
            poll_count: self.poll_count.clone(),
        }))
    }
}

#[cfg(feature = "p2p")]
struct DispatchableStage;
#[cfg(feature = "p2p")]
#[async_trait]
impl Stage for DispatchableStage {
    const NAME: &'static str = "dispatchable_thing";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        // Distinct from `succeed_with` below: if the executor ever fell
        // through to running this LOCALLY instead of honouring the
        // dispatch, this value makes that wiring bug obvious.
        Ok(Counter { n: 999 })
    }
}
#[cfg(feature = "p2p")]
impl Compatible<LamuTrainerBackend> for DispatchableStage {}

#[cfg(feature = "p2p")]
#[tokio::test]
async fn p2p_dispatch_success_is_awaited_before_plan_completes() {
    // Finding 1 repro shape: a dispatch policy that dispatches the
    // SINGLE (and therefore last/only ready) node in the plan. Pre-fix,
    // the coordinator's very next `join.join_next().await` hit an EMPTY
    // JoinSet (the detached poll task was never added to it) and
    // returned `None` immediately, so the loop broke — either tripping
    // the `completed + pruned == order.len()` debug assertion or (in a
    // release build) returning an incomplete `PlanResult` — well before
    // the mock had gone terminal. The fix makes the poll loop a real
    // JoinSet member, so the coordinator must actually wait through the
    // mock's `Ok(None)` backoff cycles.
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, base) = fresh_ctx();
    let cache = base.cache.clone();
    let poll_count = Arc::new(AtomicU32::new(0));
    let submitter = Arc::new(MockDispatchSubmitter {
        cache,
        polls_before_terminal: 2,
        terminal: MockTerminal::Succeeded,
        succeed_with: Counter { n: 42 },
        poll_count: poll_count.clone(),
        submit_count: Arc::new(AtomicU32::new(0)),
    });
    let policy = Arc::new(MockDispatchPolicy {
        dispatchable: "dispatchable_thing",
    });
    let ctx = base.with_dispatch(policy, submitter);

    let plan = Plan::<(), LamuTrainerBackend>::new("p2p_success", serde_json::json!({}))
        .start(DispatchableStage, EmptyArgs)
        .finish()
        .into_compiled();

    let start = std::time::Instant::now();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        ParallelExecutor::execute(plan, ctx),
    )
    .await
    .expect("dispatched plan must terminate")
    .expect("a Succeeded remote node must not fail the plan");
    let elapsed = start.elapsed();

    // The mock forces 2 "still running" poll cycles (500ms backoff each,
    // per the executor's poll loop) before going terminal. A coordinator
    // that raced ahead of the real completion (the pre-fix bug) would
    // return in a few milliseconds instead.
    assert!(
        elapsed >= std::time::Duration::from_millis(900),
        "coordinator returned in {elapsed:?}, before the dispatched node's \
         mock backoff cycles could have completed — it did not genuinely \
         await the dispatched node's result"
    );
    assert!(
        poll_count.load(Ordering::SeqCst) >= 3,
        "expected at least 3 polls (2×still-running + 1 terminal), got {}",
        poll_count.load(Ordering::SeqCst)
    );
    let out: Counter = result
        .final_output
        .expect("the dispatched node's real output must be in the plan result")
        .into_typed()
        .unwrap();
    assert_eq!(
        out.n, 42,
        "final output must be the artifact delivered by the mock P2P peer \
         (via cache), not a local re-run (999) or a missing/stale output"
    );
}

#[cfg(feature = "p2p")]
#[tokio::test]
async fn p2p_dispatch_failure_fails_the_plan() {
    // Finding 2 repro: pre-fix, a remote `JobState::Failed` only emitted
    // a `StageFailed` status event on the detached side-channel —
    // `first_error`/`env.cancel` were never touched, so `execute()`
    // could still return `Ok` past an explicitly failed dispatched node.
    // The fix routes the Failed outcome through the SAME
    // `Err(NodeFailure::Stage)` path a local stage failure uses.
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, base) = fresh_ctx();
    let cache = base.cache.clone();
    let submitter = Arc::new(MockDispatchSubmitter {
        cache,
        polls_before_terminal: 1,
        terminal: MockTerminal::Failed("remote OOM".to_string()),
        succeed_with: Counter { n: 0 },
        poll_count: Arc::new(AtomicU32::new(0)),
        submit_count: Arc::new(AtomicU32::new(0)),
    });
    let policy = Arc::new(MockDispatchPolicy {
        dispatchable: "dispatchable_thing",
    });
    let ctx = base.with_dispatch(policy, submitter);

    let plan = Plan::<(), LamuTrainerBackend>::new("p2p_failure", serde_json::json!({}))
        .start(DispatchableStage, EmptyArgs)
        .finish()
        .into_compiled();

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        ParallelExecutor::execute(plan, ctx),
    )
    .await
    .expect("dispatched plan must terminate");

    match result {
        Err(PlanError::StageFailed { stage, source, .. }) => {
            assert_eq!(stage, "dispatchable_thing");
            let msg = source.to_string();
            assert!(
                msg.contains("remote OOM"),
                "expected the remote failure reason surfaced in the error, got: {msg}"
            );
        }
        other => panic!(
            "an explicit remote-stage failure must fail the plan \
             (StageFailed), got: {other:?}"
        ),
    }
}

// ── GPU sampler leaked on panic (audit finding 4) ───────────────────
//
// `GpuSamplerHandle` (gpu_sampler.rs) has no `Drop` impl — a bare drop
// only DETACHES its background nvidia-smi poller (it keeps sampling
// until process exit) — so `run_node` must always reach `h.stop().await`
// to tear it down cleanly. Pre-fix, that call sat strictly after the
// stage's run future was awaited, so a panic inside the stage unwound
// straight past it, leaking the sampler task. The fix wraps the
// run-with-timeout future in `catch_unwind`, runs the same `.stop()`
// teardown on a caught panic, then `resume_unwind`s.
//
// `GpuSamplerHandle`'s inner `JoinHandle` is private to gpu_sampler.rs,
// so the leak itself isn't observable from here; this test instead
// pins the two properties that ARE observable at this layer: a
// panicking GPU-resource stage still reports as a plan-level panic
// (not swallowed, not silently downgraded to a normal `StageFailed`),
// and the catch_unwind wrapping doesn't hang the plan.
struct PanickingGpuStage;
#[async_trait]
impl Stage for PanickingGpuStage {
    const NAME: &'static str = "panicking_gpu_stage";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    type Input = ();
    type Output = Counter;
    type Args = EmptyArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &EmptyArgs,
    ) -> Result<Counter, StageError> {
        panic!("simulated stage panic — GpuSamplerHandle must still be stopped");
    }
}
impl Compatible<LamuTrainerBackend> for PanickingGpuStage {}

#[tokio::test]
async fn panicking_gpu_stage_is_reported_and_does_not_hang() {
    let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, ctx) = fresh_ctx();
    let plan = Plan::<(), LamuTrainerBackend>::new("gpu_panic", serde_json::json!({}))
        .start(PanickingGpuStage, EmptyArgs)
        .finish()
        .into_compiled();
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        ParallelExecutor::execute(plan, ctx),
    )
    .await
    .expect("a panicking GPU-resource stage must not hang the plan");
    match result {
        Err(PlanError::Other(msg)) => {
            assert!(
                msg.contains("panicked"),
                "expected a 'node task panicked' PlanError::Other, got: {msg}"
            );
        }
        other => panic!("expected PlanError::Other(\"node task panicked...\"), got {other:?}"),
    }
}
