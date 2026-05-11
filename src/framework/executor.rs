//! Plan executor.
//!
//! Walks the topo-sorted plan one stage at a time. For each:
//!
//!   1. Build the input artifact (initial map for graph-input
//!      nodes; otherwise the predecessor's cached output).
//!   2. Compute the cache key from `(stage_name, schema,
//!      input_hash, args)`.
//!   3. Cache hit → emit `StageSkipped`, advance with the cached
//!      output.
//!   4. Cache miss → emit `StageBegin`, build `StageContext`, call
//!      `StageDyn::run_erased`, persist the output to cache, write
//!      sidecar metadata next to the artifact's primary path,
//!      emit `StageEnd`.
//!
//! v2 commit 3: sequential only. Branches/parallel land commit 6
//! when the typed `fork`/`merge` builder API + per-Resource
//! semaphores arrive together.
//!
//! Cancellation: the executor honours a `CancellationToken`
//! threaded through every `StageContext`. Cancelling between
//! stages aborts cleanly with `PlanError::Cancelled`. Cancelling
//! mid-stage is the stage's responsibility (it must observe the
//! token; `python_backend` already does).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::framework::artifact::{ArtifactMetadata, ContentHash};
use crate::framework::cache::CacheHandle;
use crate::framework::error::{PlanError, StageError};
use crate::framework::plan::{NodeId, Plan};
use crate::framework::stage::{ErasedArtifact, StageContext};
use crate::framework::status::{spawn_status_writer, StageEvent};

/// Caller-supplied execution context. Threaded through every
/// `StageContext`. Lives for the duration of one `execute` call.
pub struct ExecCtx {
    pub job_dir: PathBuf,
    pub cache: Arc<CacheHandle>,
    pub status_tx: broadcast::Sender<StageEvent>,
    pub cancel: CancellationToken,
    /// Per-resource semaphores. Stages acquire all permits in
    /// their `RESOURCES` slice before `run` is called. Default
    /// limits: Gpu=1 (single-card), Cpu=num_cpus, Network=4,
    /// Disk=2. Override via ExecCtx::with_resource_limit.
    pub resources: std::collections::HashMap<crate::framework::resource::Resource, Arc<tokio::sync::Semaphore>>,
}

impl ExecCtx {
    /// Construct an `ExecCtx` rooted at `job_dir`. The caller is
    /// responsible for creating `job_dir` if it doesn't exist.
    pub fn new(job_dir: PathBuf) -> Self {
        let cache = Arc::new(CacheHandle::job_local(job_dir.join("_cache")));
        let status_tx = crate::framework::status::make_broadcast();
        let cancel = CancellationToken::new();
        let mut resources = std::collections::HashMap::new();
        use crate::framework::resource::Resource;
        let cpu_n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        resources.insert(Resource::Gpu, Arc::new(tokio::sync::Semaphore::new(1)));
        resources.insert(Resource::Cpu, Arc::new(tokio::sync::Semaphore::new(cpu_n)));
        resources.insert(Resource::Network, Arc::new(tokio::sync::Semaphore::new(4)));
        resources.insert(Resource::Disk, Arc::new(tokio::sync::Semaphore::new(2)));
        Self {
            job_dir,
            cache,
            status_tx,
            cancel,
            resources,
        }
    }

    pub fn with_resource_limit(
        mut self,
        resource: crate::framework::resource::Resource,
        permits: usize,
    ) -> Self {
        self.resources
            .insert(resource, Arc::new(tokio::sync::Semaphore::new(permits)));
        self
    }
}

/// What the executor returns on success. Carries the final node's
/// output (when the plan has one) plus diagnostics about how the
/// run went.
#[derive(Debug)]
pub struct PlanResult {
    pub final_output: Option<ErasedArtifact>,
    pub n_stages: usize,
    pub n_cache_hits: usize,
    pub n_cache_misses: usize,
    pub elapsed: std::time::Duration,
}

/// Sequential executor. Walks the plan in topo order; every stage
/// runs in the calling task (no `tokio::spawn`) since linear
/// chains have no concurrency to exploit. Commit 6 introduces
/// `ParallelExecutor` for plans with branches.
pub struct SequentialExecutor;

impl SequentialExecutor {
    /// Execute the plan to completion.
    pub async fn execute(plan: Plan<()>, ctx: ExecCtx) -> Result<PlanResult, PlanError> {
        let started = Instant::now();
        let order = plan.topo_order()?;
        let view = plan.exec_view();

        // Spawn the persistent status writer. Its task ends when
        // ctx.status_tx is dropped (at end of execute).
        std::fs::create_dir_all(&ctx.job_dir)?;
        let writer_handle = spawn_status_writer(&ctx.status_tx, &ctx.job_dir)?;

        // Persist plan + recipe args at job-dir root for audit.
        let args_path = ctx.job_dir.join("args.json");
        let args_body = serde_json::to_vec_pretty(view.recipe_args)
            .map_err(|e| PlanError::Other(format!("serialize args: {e}")))?;
        std::fs::write(&args_path, args_body)?;

        // Outputs by node id. Populated as the executor walks the
        // topo order; each subsequent node looks up its
        // predecessor here.
        let mut outputs: HashMap<NodeId, ErasedArtifact> = HashMap::new();
        // Pre-seed initial inputs (graph inputs).
        for (id, art) in view.initial {
            outputs.insert(*id, art.clone());
        }

        let mut n_hits = 0usize;
        let mut n_misses = 0usize;

        for (idx, node_id) in order.iter().enumerate() {
            if ctx.cancel.is_cancelled() {
                let _ = ctx.status_tx.send(StageEvent::StageFailed {
                    node_idx: idx as u32,
                    stage_name: "<cancelled>".into(),
                    error: "plan cancelled before stage".into(),
                });
                return Err(PlanError::Cancelled);
            }

            let node = &view.nodes[*node_id as usize];
            let stage_name = node.stage.name();

            // Find input. Linear plans have at most one
            // predecessor; multi-input merge nodes (commit 6)
            // gather all predecessors into a tuple.
            let preds: Vec<NodeId> = view
                .edges
                .iter()
                .filter(|e| e.to == *node_id)
                .map(|e| e.from)
                .collect();
            let input: ErasedArtifact = match preds.as_slice() {
                [] => outputs.get(node_id).cloned().ok_or_else(|| {
                    PlanError::Other(format!(
                        "node {} has no predecessors and no initial input",
                        node_id
                    ))
                })?,
                [single] => outputs.get(single).cloned().ok_or_else(|| {
                    PlanError::Other(format!(
                        "node {} predecessor {} produced no output",
                        node_id, single
                    ))
                })?,
                multi => {
                    // Gather predecessors' outputs as a tuple
                    // ErasedArtifact. Commit-3 commit message
                    // promised this lands here. Tuple kind is
                    // "tuple<N>" where N is the arity; payload is
                    // [child0, child1, ...] preserving fork order
                    // (the order the predecessors appear in the
                    // edges Vec, which corresponds to the order
                    // the recipe author called fork/fork3).
                    let mut payloads = Vec::with_capacity(multi.len());
                    for &pid in multi {
                        let art = outputs.get(&pid).cloned().ok_or_else(|| {
                            PlanError::Other(format!(
                                "node {} predecessor {} produced no output",
                                node_id, pid
                            ))
                        })?;
                        payloads.push(art.payload);
                    }
                    let tuple_kind = format!("tuple<{}>", multi.len());
                    ErasedArtifact {
                        kind: tuple_kind,
                        schema: 1,
                        payload: serde_json::Value::Array(payloads),
                    }
                }
            };

            // Cache key. Hash of the canonical-JSON form of the
            // erased input artifact (kind + schema + sorted-keys
            // payload). Cheap; the cache lookup is on the hot
            // path so this needs to be fast.
            let input_hash = input_hash_from_erased(&input);

            let key = CacheHandle::key_for(stage_name, node.stage.schema(), input_hash, &node.args);

            // Lookup.
            if let Some(hit) = ctx.cache.lookup(key) {
                let _ = ctx.status_tx.send(StageEvent::StageSkipped {
                    node_idx: idx as u32,
                    stage_name: stage_name.to_string(),
                    cache_key: key,
                });
                outputs.insert(*node_id, hit.artifact);
                n_hits += 1;
                continue;
            }

            // Miss → run.
            let _ = ctx.status_tx.send(StageEvent::StageBegin {
                node_idx: idx as u32,
                stage_name: stage_name.to_string(),
                input_hash,
            });

            let stage_dir = ctx.job_dir.join("stages").join(format!("{idx}-{stage_name}"));
            std::fs::create_dir_all(&stage_dir)?;
            let stage_ctx = StageContext {
                job_dir: ctx.job_dir.clone(),
                stage_dir: stage_dir.clone(),
                status_tx: ctx.status_tx.clone(),
                cancel: ctx.cancel.clone(),
                cache: ctx.cache.clone(),
            };

            // Acquire resource permits in declared order. We keep
            // the OwnedSemaphorePermits in a Vec dropped at end of
            // the loop iteration so the next stage can acquire.
            let mut permits = Vec::new();
            for resource in node.stage.resources() {
                if let Some(sem) = ctx.resources.get(resource) {
                    let _ = ctx.status_tx.send(StageEvent::StageBlocked {
                        node_idx: idx as u32,
                        stage_name: stage_name.to_string(),
                        resource: *resource,
                    });
                    match sem.clone().acquire_owned().await {
                        Ok(p) => permits.push(p),
                        Err(_) => {
                            // Semaphore closed (shouldn't happen);
                            // fail the stage cleanly rather than
                            // panicking.
                            return Err(PlanError::Other(format!(
                                "resource '{}' semaphore closed",
                                resource
                            )));
                        }
                    }
                }
            }

            let stage_started = Instant::now();
            let run_result = node
                .stage
                .run_erased(&stage_ctx, input, node.args.clone())
                .await;
            // Permits drop here, releasing the resource for the
            // next stage.
            drop(permits);
            // Drop the stage_ctx (and its status_tx clone)
            // BEFORE awaiting the writer handle on the failure
            // path — otherwise the writer's broadcast channel
            // never closes (clones are still alive) and the await
            // hangs forever.
            drop(stage_ctx);
            let output = match run_result {
                Ok(o) => o,
                Err(e) => {
                    let _ = ctx.status_tx.send(StageEvent::StageFailed {
                        node_idx: idx as u32,
                        stage_name: stage_name.to_string(),
                        error: format!("{e}"),
                    });
                    drop(ctx.status_tx);
                    let _ = writer_handle.await;
                    return Err(PlanError::StageFailed {
                        idx: idx as u32,
                        stage: stage_name.to_string(),
                        source: e,
                    });
                }
            };

            // Persist sidecar metadata next to the artifact's
            // primary path. We don't have access to the typed
            // artifact's `primary_path()` here (it's erased), but
            // the producing stage has already written the on-disk
            // payload to its `stage_dir`. Write metadata at
            // `<stage_dir>/output.metadata.json`.
            let output_hash = output_hash_from_erased(&output);
            let metadata = ArtifactMetadata::new(
                output.kind.clone(),
                output.schema,
                output_hash,
            )
            .with_stage(stage_name.to_string());
            let _ = metadata.write_to(&stage_dir.join("output.metadata.json"));

            // Insert into cache for resumes.
            if let Err(e) = ctx.cache.insert(key, &output) {
                tracing::warn!(
                    "executor: cache insert for stage '{}' failed: {e}; continuing",
                    stage_name
                );
            }

            let _ = ctx.status_tx.send(StageEvent::StageEnd {
                node_idx: idx as u32,
                stage_name: stage_name.to_string(),
                output_hash,
                elapsed: stage_started.elapsed(),
            });

            outputs.insert(*node_id, output);
            n_misses += 1;
        }

        // Pick the final output (the last node in topo order; for
        // linear plans this is unambiguous).
        let final_output = order.last().and_then(|id| outputs.remove(id));

        // Drop the broadcast sender so the writer task exits, then
        // await it to flush any tail events.
        drop(ctx.status_tx);
        let _ = writer_handle.await;

        Ok(PlanResult {
            final_output,
            n_stages: order.len(),
            n_cache_hits: n_hits,
            n_cache_misses: n_misses,
            elapsed: started.elapsed(),
        })
    }
}

/// Compute the `ContentHash` of an erased artifact's payload by
/// SHA-256-ing the canonical JSON. Cheap; the cache lookup is the
/// hot path so this needs to be fast.
///
/// Why not call the typed artifact's `content_hash`? Because at
/// the executor level we've already erased the type. The trade-off:
/// erased hash is over the JSON metadata blob (handle), not the
/// on-disk payload. Two artifacts whose JSON differs but whose
/// on-disk content is identical will get different cache keys —
/// not a correctness bug (we never claim "same key ⇒ same disk
/// content", only "same key ⇒ same prior result"), but it's a
/// missed cache opportunity. Concrete artifact impls of
/// `content_hash` already account for this by normalizing the
/// JSON itself before serializing.
fn input_hash_from_erased(art: &ErasedArtifact) -> ContentHash {
    // Reuse the canonical-JSON path so reordered fields don't
    // change the input hash (mirrors cache::canonical_value).
    let canon = canonical_value(&art.payload);
    let body = serde_json::to_vec(&canon).unwrap_or_default();
    let mut h = sha2::Sha256::new();
    use sha2::Digest;
    h.update(art.kind.as_bytes());
    h.update(art.schema.to_le_bytes());
    h.update(&body);
    let arr: [u8; 32] = h.finalize().into();
    ContentHash(arr)
}

fn output_hash_from_erased(art: &ErasedArtifact) -> ContentHash {
    input_hash_from_erased(art)
}

fn canonical_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted: std::collections::BTreeMap<String, serde_json::Value> =
                std::collections::BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonical_value(v));
            }
            let mut out = serde_json::Map::new();
            for (k, v) in sorted {
                out.insert(k, v);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.iter().map(canonical_value).collect())
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::Artifact;
    use crate::framework::resource::Resource;
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

    /// Counter that always emits Counter { n: 1 } and counts how
    /// many times its `run` was invoked across the test process.
    /// Tests in this module touch process-global counters
    /// (MAKE_RUN_COUNT, INC_RUN_COUNT) that the parallel test
    /// runner would race on. Serialize via a module-wide mutex.
    /// Production code path uses no static state.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    static MAKE_RUN_COUNT: AtomicU32 = AtomicU32::new(0);

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

    /// Stage that always returns a backend error. Tests the
    /// failure-propagation path.
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
        let plan = Plan::new("test", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .then(Increment, EmptyArgs)
            .then(Increment, EmptyArgs)
            .finish();
        let result = SequentialExecutor::execute(plan, ctx).await.unwrap();
        assert_eq!(result.n_stages, 3);
        assert_eq!(result.n_cache_misses, 3);
        assert_eq!(result.n_cache_hits, 0);
        // Final output is Counter { n: 3 } (1 → 2 → 3).
        let out = result.final_output.unwrap();
        let counter: Counter = out.into_typed().unwrap();
        assert_eq!(counter.n, 3);
    }

    #[tokio::test]
    async fn cache_hit_skips_run_on_repeat_execution() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
        INC_RUN_COUNT.store(0, Ordering::SeqCst);
        let (_td, ctx) = fresh_ctx();
        let cache = ctx.cache.clone();

        // First run: every stage is a miss.
        let plan = Plan::new("test", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .then(Increment, EmptyArgs)
            .finish();
        let r1 = SequentialExecutor::execute(plan, ctx).await.unwrap();
        assert_eq!(r1.n_cache_misses, 2);
        assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 1);

        // Second run with a fresh ExecCtx but the SAME cache dir.
        let td_keepalive_for_second_run = tempfile::tempdir().unwrap();
        let job_dir2 = td_keepalive_for_second_run.path().to_path_buf();
        let ctx2 = ExecCtx::new(job_dir2);
        let ctx2 = ExecCtx { cache, ..ctx2 };
        let plan2 = Plan::new("test", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .then(Increment, EmptyArgs)
            .finish();
        let r2 = SequentialExecutor::execute(plan2, ctx2).await.unwrap();
        assert_eq!(r2.n_cache_hits, 2, "second run should hit cache for both stages");
        assert_eq!(r2.n_cache_misses, 0);
        // Run counters didn't increment.
        assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stage_failure_propagates_as_plan_error() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (_td, ctx) = fresh_ctx();
        let plan = Plan::new("failing", serde_json::json!({}))
            .start(AlwaysFail, EmptyArgs)
            .finish();
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
        let plan = Plan::new("c", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .finish();
        let r = SequentialExecutor::execute(plan, ctx).await;
        assert!(matches!(r, Err(PlanError::Cancelled)));
    }

    #[tokio::test]
    async fn status_jsonl_persists_to_disk() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (td, ctx) = fresh_ctx();
        let plan = Plan::new("p", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .then(Increment, EmptyArgs)
            .finish();
        let _ = SequentialExecutor::execute(plan, ctx).await.unwrap();
        let path = td.path().join("status.jsonl");
        assert!(path.exists());
        let body = std::fs::read_to_string(&path).unwrap();
        // 2 stage_begin + 2 stage_end (one per stage).
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
        let plan = Plan::new("p", recipe_args.clone())
            .start(MakeOne, EmptyArgs)
            .finish();
        let _ = SequentialExecutor::execute(plan, ctx).await.unwrap();
        let body = std::fs::read_to_string(td.path().join("args.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed, recipe_args);
    }

    #[tokio::test]
    async fn sidecar_metadata_written_per_stage() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (td, ctx) = fresh_ctx();
        let plan = Plan::new("p", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .finish();
        let _ = SequentialExecutor::execute(plan, ctx).await.unwrap();
        let sidecar = td
            .path()
            .join("stages/0-make_one/output.metadata.json");
        assert!(sidecar.exists(), "expected sidecar at {}", sidecar.display());
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&sidecar).unwrap()).unwrap();
        assert_eq!(parsed["kind"], "test.counter");
        assert_eq!(parsed["produced_by_stage"], "make_one");
    }
}
