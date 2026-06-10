//! Plan executor.
//!
//! Two executors share one per-stage core, [`run_node`]:
//!
//!   1. Build the input artifact (initial map for graph-input
//!      nodes; the predecessor's output for a linear edge; a
//!      `tuple<N>` for a merge node).
//!   2. Compute the cache key from `(stage_name, schema,
//!      input_hash, args)`.
//!   3. Cache hit → emit `StageSkipped`, advance with the cached
//!      output.
//!   4. Cache miss → emit `StageBegin`, acquire resource permits,
//!      run the stage in a private `.tmp-<key>` dir, atomically
//!      promote it, write sidecar metadata, insert the cache entry,
//!      emit `StageEnd`.
//!
//! [`SequentialExecutor`] walks the topo order one node at a time
//! (the debugging-friendly default). [`ParallelExecutor`] drives the
//! DAG with real concurrency: a single coordinator owns all scheduler
//! state and spawns ready nodes onto a `JoinSet`, bounded by
//! `max_in_flight` and gated by the per-`Resource` semaphores. Both
//! drive the SAME `run_node`, so the FW-2 atomicity + cache-key
//! invariants hold identically and the two executors are observably
//! equivalent on a given plan.
//!
//! Cancellation: the executor honours a `CancellationToken` threaded
//! through every `StageContext`. Cancelling between stages aborts
//! cleanly with `PlanError::Cancelled`. Cancelling mid-stage is the
//! stage's responsibility (it must observe the token; `python_backend`
//! already does). The parallel executor fails FAST: the first stage
//! error cancels in-flight siblings, drains them (so their FW-2 tmp
//! cleanup runs), then reports the first error.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::framework::artifact::{ArtifactMetadata, ContentHash};
use crate::framework::cache::CacheHandle;
use crate::framework::error::{PlanError, StageError};
use crate::framework::plan::{CompiledPlan, NodeId};
use crate::framework::resource::Resource;
use crate::framework::stage::{ErasedArtifact, StageContext, StageDyn};
use crate::framework::status::{StageEvent, StatusHub, spawn_status_writer};

/// Default bound on concurrently-spawned node tasks in the parallel
/// executor. The real throttle is the per-`Resource` semaphores; this
/// only caps task/memory overhead so a very wide DAG can't spawn
/// thousands of futures at once.
pub const DEFAULT_MAX_IN_FLIGHT: usize = 8;

/// Caller-supplied execution context. Threaded through every
/// `StageContext`. Lives for the duration of one `execute` call.
pub struct ExecCtx {
    pub job_dir: PathBuf,
    pub cache: Arc<CacheHandle>,
    /// Status fan-out hub. Subscribe a live receiver via
    /// `ctx.status.subscribe()`; the executor emits through it.
    pub status: Arc<StatusHub>,
    /// The lossless lifecycle receiver, handed to the status writer by
    /// the executor's prelude. `None` once taken (after one execute).
    lifecycle_rx: Option<mpsc::UnboundedReceiver<StageEvent>>,
    pub cancel: CancellationToken,
    /// Per-resource semaphores. Stages acquire all permits in
    /// their `RESOURCES` slice before `run` is called. Default
    /// limits: Gpu=1 (single-card), Cpu=num_cpus, Network=4,
    /// Disk=2. Override via ExecCtx::with_resource_limit.
    pub resources: std::collections::HashMap<Resource, Arc<tokio::sync::Semaphore>>,
    /// Max concurrently-spawned node tasks (parallel executor only).
    pub max_in_flight: usize,
}

impl ExecCtx {
    /// Construct an `ExecCtx` rooted at `job_dir`. The caller is
    /// responsible for creating `job_dir` if it doesn't exist.
    pub fn new(job_dir: PathBuf) -> Self {
        let cache = Arc::new(CacheHandle::job_local(job_dir.join("_cache")));
        let (status, lifecycle_rx) = StatusHub::new();
        let cancel = CancellationToken::new();
        let mut resources = std::collections::HashMap::new();
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
            status,
            lifecycle_rx: Some(lifecycle_rx),
            cancel,
            resources,
            max_in_flight: DEFAULT_MAX_IN_FLIGHT,
        }
    }

    pub fn with_resource_limit(mut self, resource: Resource, permits: usize) -> Self {
        self.resources
            .insert(resource, Arc::new(tokio::sync::Semaphore::new(permits)));
        self
    }

    pub fn with_max_in_flight(mut self, n: usize) -> Self {
        self.max_in_flight = n.max(1);
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

// ════════════════════════════════════════════════════════════════════
// Shared per-stage core (run_node) + its env/task/outcome types. BOTH
// executors drive this, so FW-2 atomicity lives in exactly one place.
// ════════════════════════════════════════════════════════════════════

/// Immutable per-run environment shared by every node task. Cheaply
/// cloneable handles only — a worker task borrows nothing from the plan.
struct NodeEnv {
    job_dir: PathBuf,
    cache: Arc<CacheHandle>,
    status: Arc<StatusHub>,
    cancel: CancellationToken,
    resources: HashMap<Resource, Arc<tokio::sync::Semaphore>>,
    recipe_name: String,
}

/// Everything one node needs to run, snapshotted by the coordinator
/// (which alone reads the `outputs`/`logical_outputs` maps). Owned /
/// Arc fields so the task can move to another tokio worker thread.
struct NodeTask {
    node_id: NodeId,
    /// Position in topo order — STABLE across executors and runs, so
    /// status consumers key on it (never on event arrival order).
    node_idx: u32,
    stage: Arc<dyn StageDyn>,
    args: serde_json::Value,
    canon_args: Vec<u8>,
    input: ErasedArtifact,
    input_hash: ContentHash,
    key: ContentHash,
}

/// A node's result, fed back to the coordinator to advance scheduling.
struct NodeOutcome {
    node_id: NodeId,
    output: ErasedArtifact,
    logical: ContentHash,
    cache_hit: bool,
}

/// How a node run failed. The coordinator maps this to a `PlanError`;
/// the `StageFailed`/cancel status event is already emitted by
/// `run_node` before it returns.
enum NodeFailure {
    /// The plan token fired (before or during the stage). Output, if
    /// any, was discarded; nothing was cached.
    Cancelled,
    /// The stage (or its promote) failed.
    Stage {
        idx: u32,
        stage: String,
        source: StageError,
    },
    /// An executor-internal failure (e.g. a closed semaphore).
    Other(String),
}

/// Run ONE node: cache lookup → tmp dir → resource permits → run →
/// (cancel check) → atomic promote → rebase → sidecar → cache insert →
/// events. The single home of the FW-2 atomicity contract. Emits
/// lifecycle events via `env.status`; never manages the status
/// writer's lifecycle (the coordinator owns that).
async fn run_node(task: NodeTask, env: Arc<NodeEnv>) -> Result<NodeOutcome, NodeFailure> {
    let idx = task.node_idx;
    let stage_name = task.stage.name().to_string();

    // ── Cache lookup ────────────────────────────────────────────────
    if let Some(hit) = env.cache.lookup(task.key) {
        env.status.emit(StageEvent::StageSkipped {
            node_idx: idx,
            stage_name: stage_name.clone(),
            cache_key: task.key,
        });
        let logical = compute_logical_output_hash(
            task.stage.as_ref(),
            &hit.artifact,
            task.stage.deterministic(),
            &stage_name,
            task.stage.schema(),
            task.input_hash,
            &task.canon_args,
        );
        return Ok(NodeOutcome {
            node_id: task.node_id,
            output: hit.artifact,
            logical,
            cache_hit: true,
        });
    }

    // ── Miss → run ──────────────────────────────────────────────────
    env.status.emit(StageEvent::StageBegin {
        node_idx: idx,
        stage_name: stage_name.clone(),
        input_hash: task.input_hash,
    });

    // FW-2: the stage runs against a private `.tmp-<key>` dir; on Ok we
    // atomically rename it to the final name and ONLY THEN insert the
    // cache entry (the sole resume oracle). No promote ⇒ no cache ⇒
    // re-run. Tmp name is key-scoped so two positions of the same stage
    // (or a re-run with different args) never collide.
    let stages_root = env.job_dir.join("stages");
    let final_stage_dir = stages_root.join(format!("{idx}-{stage_name}"));
    let tmp_stage_dir = stages_root.join(format!(".tmp-{idx}-{stage_name}-{}", task.key.to_hex()));
    let _ = std::fs::remove_dir_all(&tmp_stage_dir);
    if let Err(e) = std::fs::create_dir_all(&tmp_stage_dir) {
        return Err(NodeFailure::Stage {
            idx,
            stage: stage_name,
            source: StageError::Io {
                path: tmp_stage_dir,
                source: e,
            },
        });
    }

    let stage_ctx = StageContext {
        job_dir: env.job_dir.clone(),
        stage_dir: tmp_stage_dir.clone(),
        node_idx: idx,
        status_tx: env.status.broadcast_sender(),
        cancel: env.cancel.clone(),
        cache: env.cache.clone(),
        recipe_name: env.recipe_name.clone(),
    };

    // ── Resource permits ────────────────────────────────────────────
    // Acquire in canonical (sorted) order so two concurrent stages can
    // never deadlock on the same pair in opposite orders. `try_acquire`
    // first; only emit `StageBlocked` on ACTUAL contention.
    let mut sorted_resources: Vec<Resource> = task.stage.resources().to_vec();
    sorted_resources.sort();
    let mut permits = Vec::new();
    for resource in sorted_resources {
        let Some(sem) = env.resources.get(&resource) else {
            continue;
        };
        let permit = match sem.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                env.status.emit(StageEvent::StageBlocked {
                    node_idx: idx,
                    stage_name: stage_name.clone(),
                    resource,
                });
                match sem.clone().acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => {
                        let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                        return Err(NodeFailure::Other(format!(
                            "resource '{resource}' semaphore closed"
                        )));
                    }
                }
            }
        };
        permits.push(permit);
    }

    let stage_started = Instant::now();
    let run_result = task
        .stage
        .run_erased(&stage_ctx, task.input, task.args.clone())
        .await;
    // Permits drop here, releasing the resource for queued stages.
    drop(permits);
    // Drop the stage_ctx (its status_tx clone) before any await so it
    // can't keep the broadcast channel open.
    drop(stage_ctx);

    let output = match run_result {
        Ok(o) => {
            debug_assert_eq!(
                o.kind,
                task.stage.output_kind(),
                "stage '{stage_name}' produced kind '{}' but declares output_kind '{}'",
                o.kind,
                task.stage.output_kind()
            );
            // A cancel observed during the run must NOT be promoted /
            // cached — discard the tmp output, report Cancelled.
            if env.cancel.is_cancelled() {
                let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                env.status.emit(StageEvent::StageFailed {
                    node_idx: idx,
                    stage_name,
                    error: "plan cancelled during stage".into(),
                });
                return Err(NodeFailure::Cancelled);
            }
            o
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp_stage_dir);
            env.status.emit(StageEvent::StageFailed {
                node_idx: idx,
                stage_name: stage_name.clone(),
                error: format!("{e}"),
            });
            return Err(NodeFailure::Stage {
                idx,
                stage: stage_name,
                source: e,
            });
        }
    };

    // FW-2 promote: atomic rename of the completed tmp dir to the final
    // name. Remove any stale final dir first (a prior crash that left a
    // partial but never reached cache.insert).
    let _ = std::fs::remove_dir_all(&final_stage_dir);
    if let Err(e) = std::fs::rename(&tmp_stage_dir, &final_stage_dir) {
        let _ = std::fs::remove_dir_all(&tmp_stage_dir);
        env.status.emit(StageEvent::StageFailed {
            node_idx: idx,
            stage_name: stage_name.clone(),
            error: format!("promote stage output: {e}"),
        });
        return Err(NodeFailure::Stage {
            idx,
            stage: stage_name,
            source: StageError::Io {
                path: final_stage_dir,
                source: e,
            },
        });
    }

    // Re-point tmp-rooted absolute paths in the output handle at the
    // promoted final dir so downstream (and a later cache hit) read the
    // files where they now live.
    let output = task
        .stage
        .rebase_output_paths(output, &tmp_stage_dir, &final_stage_dir);

    // Sidecar metadata next to the promoted payload.
    let output_hash = content_hash_from_erased(&output);
    let metadata = ArtifactMetadata::new(output.kind.clone(), output.schema, output_hash)
        .with_stage(stage_name.clone());
    let _ = metadata.write_to(&final_stage_dir.join("output.metadata.json"));

    // Cache insert — STRICTLY after the atomic promote (the load-bearing
    // FW-2 ordering: the resume oracle appears only once the output is
    // fully in place).
    if let Err(e) = env.cache.insert(task.key, &output) {
        tracing::warn!("executor: cache insert for stage '{stage_name}' failed: {e}; continuing");
    }

    env.status.emit(StageEvent::StageEnd {
        node_idx: idx,
        stage_name: stage_name.clone(),
        output_hash,
        elapsed: stage_started.elapsed(),
    });

    let logical = compute_logical_output_hash(
        task.stage.as_ref(),
        &output,
        task.stage.deterministic(),
        &stage_name,
        task.stage.schema(),
        task.input_hash,
        &task.canon_args,
    );
    Ok(NodeOutcome {
        node_id: task.node_id,
        output,
        logical,
        cache_hit: false,
    })
}

/// Predecessor node ids of `node_id`, in edge order (which preserves
/// the order a recipe author called `fork`/`merge`).
fn predecessors(edges: &[crate::framework::plan::PlanEdge], node_id: NodeId) -> Vec<NodeId> {
    edges
        .iter()
        .filter(|e| e.to == node_id)
        .map(|e| e.from)
        .collect()
}

/// Build a node's input artifact from its predecessors' outputs
/// (or, for a graph-input node, from the pre-seeded `initial` output).
fn gather_input(
    node_id: NodeId,
    preds: &[NodeId],
    outputs: &HashMap<NodeId, ErasedArtifact>,
) -> Result<ErasedArtifact, PlanError> {
    match preds {
        [] => outputs.get(&node_id).cloned().ok_or_else(|| {
            PlanError::Other(format!(
                "node {node_id} has no predecessors and no initial input"
            ))
        }),
        [single] => outputs.get(single).cloned().ok_or_else(|| {
            PlanError::Other(format!(
                "node {node_id} predecessor {single} produced no output"
            ))
        }),
        multi => {
            // Merge: a `tuple<N>` envelope (B4). Payload is a
            // length-prefixed `bincode(Vec<ErasedArtifact>)` of the
            // children in edge (fork-call) order — each child keeps its
            // own kind+schema, so the tuple-consuming stage's
            // `decode_erased` validates every member recursively (a
            // wrong child kind names the ACTUAL kind, not an opaque
            // concat-decode error).
            // Collect child references (no clone) — serde serializes
            // `Vec<&ErasedArtifact>` byte-identically to `Vec<ErasedArtifact>`,
            // which the consumer's `decode_erased` reads as owned.
            let mut children: Vec<&ErasedArtifact> = Vec::with_capacity(multi.len());
            for &pid in multi {
                let art = outputs.get(&pid).ok_or_else(|| {
                    PlanError::Other(format!(
                        "node {node_id} predecessor {pid} produced no output"
                    ))
                })?;
                children.push(art);
            }
            let payload = bincode::serialize(&children).map_err(|e| {
                PlanError::Other(format!("encode tuple<{}> input: {e}", multi.len()))
            })?;
            Ok(ErasedArtifact {
                kind: format!("tuple<{}>", multi.len()),
                schema: crate::framework::stage::TUPLE_ENVELOPE_SCHEMA,
                payload,
            })
        }
    }
}

/// The LOGICAL input hash a node folds into its cache key — built from
/// predecessors' logical hashes (stable across stochastic re-runs), not
/// from real content bytes.
fn gather_input_hash(
    node_id: NodeId,
    preds: &[NodeId],
    logical_outputs: &HashMap<NodeId, ContentHash>,
) -> Result<ContentHash, PlanError> {
    match preds {
        [] => logical_outputs
            .get(&node_id)
            .copied()
            .ok_or_else(|| PlanError::Other(format!("node {node_id} has no logical input hash"))),
        [single] => logical_outputs.get(single).copied().ok_or_else(|| {
            PlanError::Other(format!(
                "node {node_id} predecessor {single} missing logical hash"
            ))
        }),
        multi => {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(b"tuple");
            h.update([multi.len() as u8]);
            for &pid in multi {
                let lh = logical_outputs.get(&pid).ok_or_else(|| {
                    PlanError::Other(format!(
                        "node {node_id} predecessor {pid} missing logical hash"
                    ))
                })?;
                h.update(lh.0);
            }
            Ok(ContentHash(h.finalize().into()))
        }
    }
}

/// Build the `NodeTask` for `node_id`, reading its input + input_hash
/// from the coordinator's maps and computing the cache key.
fn build_task(
    node: &crate::framework::plan::PlanNode,
    node_idx: u32,
    edges: &[crate::framework::plan::PlanEdge],
    outputs: &HashMap<NodeId, ErasedArtifact>,
    logical_outputs: &HashMap<NodeId, ContentHash>,
) -> Result<NodeTask, PlanError> {
    let preds = predecessors(edges, node.id);
    let input = gather_input(node.id, &preds, outputs)?;
    let input_hash = gather_input_hash(node.id, &preds, logical_outputs)?;
    let key = CacheHandle::key_for_canon_bytes(
        node.stage.name(),
        node.stage.schema(),
        input_hash,
        &node.canon_args,
    );
    Ok(NodeTask {
        node_id: node.id,
        node_idx,
        stage: node.stage.clone(),
        args: node.args.clone(),
        canon_args: node.canon_args.clone(),
        input,
        input_hash,
        key,
    })
}

/// Map a `NodeFailure` to a `PlanError`.
fn plan_error_of(f: NodeFailure) -> PlanError {
    match f {
        NodeFailure::Cancelled => PlanError::Cancelled,
        NodeFailure::Stage { idx, stage, source } => PlanError::StageFailed { idx, stage, source },
        NodeFailure::Other(s) => PlanError::Other(s),
    }
}

/// Shared coordinator setup: validate, spawn the status writer, persist
/// args.json, and seed the initial outputs. CONSUMES `ctx`, MOVING the
/// `StatusHub` into the one `NodeEnv` — when the last `Arc<NodeEnv>`
/// drops, the hub (and its lifecycle Sender) drop, the writer's
/// lifecycle channel closes, and the writer exits. The lossless
/// lifecycle receiver is handed to the writer here.
struct Prelude {
    writer_handle: tokio::task::JoinHandle<()>,
    env: Arc<NodeEnv>,
    outputs: HashMap<NodeId, ErasedArtifact>,
    logical_outputs: HashMap<NodeId, ContentHash>,
}

fn prelude(mut ctx: ExecCtx, plan: &CompiledPlan) -> Result<Prelude, PlanError> {
    debug_assert!(
        !ctx.resources.is_empty(),
        "ExecCtx must declare resource semaphores"
    );
    std::fs::create_dir_all(&ctx.job_dir)?;
    // Hand the writer the lossless lifecycle receiver + a broadcast
    // subscription (taken inside spawn_status_writer).
    let lifecycle_rx = ctx
        .lifecycle_rx
        .take()
        .ok_or_else(|| PlanError::Other("ExecCtx.lifecycle_rx already consumed".into()))?;
    let writer_handle = spawn_status_writer(&ctx.status, lifecycle_rx, &ctx.job_dir)?;

    let view = plan.exec_view();
    let args_path = ctx.job_dir.join("args.json");
    let args_body = serde_json::to_vec_pretty(view.recipe_args)
        .map_err(|e| PlanError::Other(format!("serialize args: {e}")))?;
    std::fs::write(&args_path, args_body)?;

    let mut outputs: HashMap<NodeId, ErasedArtifact> = HashMap::new();
    let mut logical_outputs: HashMap<NodeId, ContentHash> = HashMap::new();
    for (id, art) in view.initial {
        let lh = content_hash_from_erased(art);
        outputs.insert(*id, art.clone());
        logical_outputs.insert(*id, lh);
    }

    // MOVE ctx's fields into env — the hub Arc lives only here now.
    let env = Arc::new(NodeEnv {
        job_dir: ctx.job_dir,
        cache: ctx.cache,
        status: ctx.status,
        cancel: ctx.cancel,
        resources: ctx.resources,
        recipe_name: plan.name().to_string(),
    });

    Ok(Prelude {
        writer_handle,
        env,
        outputs,
        logical_outputs,
    })
}

/// Drop the (sole) `NodeEnv` Arc so the status channel closes, then
/// await the writer to flush the tail events. The caller MUST have
/// dropped every other `Arc<NodeEnv>` first (the parallel JoinSet must
/// be fully drained), or this hangs.
async fn finish_writer(env: Arc<NodeEnv>, writer_handle: tokio::task::JoinHandle<()>) {
    drop(env);
    let _ = writer_handle.await;
}

/// Dispatch a plan to the configured executor. Default is
/// [`SequentialExecutor`] (the debugging-friendly, burn-in-stable
/// path); set `BLUT_EXECUTOR=parallel` to opt into [`ParallelExecutor`].
/// One seam so the CLI/TUI launch sites don't each branch on the env.
pub async fn execute_plan(plan: CompiledPlan, ctx: ExecCtx) -> Result<PlanResult, PlanError> {
    let parallel = std::env::var("BLUT_EXECUTOR")
        .map(|v| v.eq_ignore_ascii_case("parallel"))
        .unwrap_or(false);
    if parallel {
        ParallelExecutor::execute(plan, ctx).await
    } else {
        SequentialExecutor::execute(plan, ctx).await
    }
}

// ════════════════════════════════════════════════════════════════════
// Sequential executor — one node at a time, topo order.
// ════════════════════════════════════════════════════════════════════

pub struct SequentialExecutor;

impl SequentialExecutor {
    /// Execute the plan to completion, one stage at a time.
    pub async fn execute(plan: CompiledPlan, ctx: ExecCtx) -> Result<PlanResult, PlanError> {
        let started = Instant::now();
        let order = plan.topo_order()?;
        let view = plan.exec_view();
        debug_assert_eq!(
            order.len(),
            view.nodes.len(),
            "topo_order must cover all nodes"
        );

        let Prelude {
            writer_handle,
            env,
            mut outputs,
            mut logical_outputs,
        } = prelude(ctx, &plan)?;

        let mut n_hits = 0usize;
        let mut n_misses = 0usize;

        for (idx, node_id) in order.iter().enumerate() {
            if env.cancel.is_cancelled() {
                env.status.emit(StageEvent::StageFailed {
                    node_idx: idx as u32,
                    stage_name: "<cancelled>".into(),
                    error: "plan cancelled before stage".into(),
                });
                finish_writer(env, writer_handle).await;
                return Err(PlanError::Cancelled);
            }

            let node = &view.nodes[*node_id as usize];
            let task = match build_task(node, idx as u32, view.edges, &outputs, &logical_outputs) {
                Ok(t) => t,
                Err(e) => {
                    finish_writer(env, writer_handle).await;
                    return Err(e);
                }
            };

            match run_node(task, env.clone()).await {
                Ok(outcome) => {
                    if outcome.cache_hit {
                        n_hits += 1;
                    } else {
                        n_misses += 1;
                    }
                    outputs.insert(outcome.node_id, outcome.output);
                    logical_outputs.insert(outcome.node_id, outcome.logical);
                }
                Err(f) => {
                    finish_writer(env, writer_handle).await;
                    return Err(plan_error_of(f));
                }
            }
        }

        let final_output = order.last().and_then(|id| outputs.remove(id));
        finish_writer(env, writer_handle).await;

        Ok(PlanResult {
            final_output,
            n_stages: order.len(),
            n_cache_hits: n_hits,
            n_cache_misses: n_misses,
            elapsed: started.elapsed(),
        })
    }
}

// ════════════════════════════════════════════════════════════════════
// Parallel executor — ready-set scheduling, JoinSet workers.
// ════════════════════════════════════════════════════════════════════

pub struct ParallelExecutor;

impl ParallelExecutor {
    /// Execute the plan with real concurrency. A single coordinator owns
    /// every piece of mutable scheduler state (`outputs`,
    /// `logical_outputs`, in-degrees, the ready set); mutations only
    /// happen between `join_next().await`s, so the FW-2 / cache
    /// reasoning is identical to the sequential path. Ready nodes are
    /// spawned onto a `JoinSet`, bounded by `ctx.max_in_flight` and
    /// gated by the per-`Resource` semaphores inside `run_node`.
    ///
    /// **Failure semantics: fail-fast.** The cache is the resume oracle,
    /// so a completed branch is already durable; nothing is salvaged by
    /// running siblings of a doomed plan. On the first error the
    /// coordinator cancels the plan token, drains the in-flight tasks
    /// (so their FW-2 tmp cleanup runs), and returns the FIRST error —
    /// sibling `Cancelled` results never mask it. This keeps the two
    /// executors observably equivalent.
    pub async fn execute(plan: CompiledPlan, ctx: ExecCtx) -> Result<PlanResult, PlanError> {
        let started = Instant::now();
        let order = plan.topo_order()?; // also the cycle check
        let view = plan.exec_view();
        debug_assert_eq!(
            order.len(),
            view.nodes.len(),
            "topo_order must cover all nodes"
        );

        // node_idx = topo position (stable status key).
        let mut node_idx_of: HashMap<NodeId, u32> = HashMap::new();
        for (i, id) in order.iter().enumerate() {
            node_idx_of.insert(*id, i as u32);
        }

        // Adjacency + in-degrees.
        let mut indeg: HashMap<NodeId, usize> = HashMap::new();
        let mut succs: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in view.nodes {
            indeg.entry(node.id).or_insert(0);
            succs.entry(node.id).or_default();
        }
        for e in view.edges {
            *indeg.entry(e.to).or_insert(0) += 1;
            succs.entry(e.from).or_default().push(e.to);
        }

        let max_in_flight = ctx.max_in_flight;
        let Prelude {
            writer_handle,
            env,
            mut outputs,
            mut logical_outputs,
        } = prelude(ctx, &plan)?;

        // Ready set = in-degree-0 nodes, ascending NodeId for
        // deterministic spawn order.
        let mut ready: BTreeSet<NodeId> = indeg
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();

        // Single-flight: defer a ready node whose cache key matches one
        // already in flight, so duplicate-key fork nodes don't both burn
        // GPU — the deferred node hits the cache when the first finishes.
        // `inflight_keys` is the set of keys currently running (one node
        // each, by construction); `node_key_of` is the O(1) reverse lookup
        // for "which key did this finished node run under".
        let mut inflight_keys: HashSet<ContentHash> = HashSet::new();
        let mut node_key_of: HashMap<NodeId, ContentHash> = HashMap::new();
        let mut deferred: HashMap<ContentHash, Vec<NodeId>> = HashMap::new();

        let mut join: tokio::task::JoinSet<Result<NodeOutcome, NodeFailure>> =
            tokio::task::JoinSet::new();
        let mut in_flight = 0usize;
        let mut n_hits = 0usize;
        let mut n_misses = 0usize;
        let mut first_error: Option<PlanError> = None;
        let mut completed = 0usize;

        // Pre-cancel: honour a token already fired before the first spawn.
        if env.cancel.is_cancelled() {
            env.status.emit(StageEvent::StageFailed {
                node_idx: 0,
                stage_name: "<cancelled>".into(),
                error: "plan cancelled before stage".into(),
            });
            finish_writer(env, writer_handle).await;
            return Err(PlanError::Cancelled);
        }

        loop {
            // Spawn ready nodes up to the in-flight cap (unless we're
            // already failing — then stop spawning and just drain).
            if first_error.is_none() {
                while in_flight < max_in_flight {
                    let Some(&node_id) = ready.iter().next() else {
                        break;
                    };
                    ready.remove(&node_id);
                    let node = &view.nodes[node_id as usize];
                    let node_idx = node_idx_of[&node_id];
                    let task =
                        match build_task(node, node_idx, view.edges, &outputs, &logical_outputs) {
                            Ok(t) => t,
                            Err(e) => {
                                // Surface the failure on the status channel
                                // (in-flight siblings keep emitting, so a
                                // silent build error would be conspicuous).
                                env.status.emit(StageEvent::StageFailed {
                                    node_idx,
                                    stage_name: node.stage.name().to_string(),
                                    error: format!("{e}"),
                                });
                                first_error.get_or_insert(e);
                                env.cancel.cancel();
                                break;
                            }
                        };
                    // Single-flight: if this exact key is already running,
                    // defer until it completes (then it cache-hits).
                    if inflight_keys.contains(&task.key) {
                        deferred.entry(task.key).or_default().push(node_id);
                        continue;
                    }
                    inflight_keys.insert(task.key);
                    node_key_of.insert(node_id, task.key);
                    let env_c = env.clone();
                    join.spawn(async move { run_node(task, env_c).await });
                    in_flight += 1;
                }
            }

            if in_flight == 0 {
                break; // nothing running and nothing spawnable → done
            }

            // Await the next completed node.
            let joined = join.join_next().await;
            in_flight -= 1;
            let res = match joined {
                Some(Ok(r)) => r,
                Some(Err(join_err)) => {
                    // Task panicked. Record as the first error, cancel.
                    first_error.get_or_insert(PlanError::Other(format!(
                        "node task panicked: {join_err}"
                    )));
                    env.cancel.cancel();
                    continue;
                }
                None => break,
            };

            match res {
                Ok(outcome) => {
                    completed += 1;
                    if outcome.cache_hit {
                        n_hits += 1;
                    } else {
                        n_misses += 1;
                    }
                    // O(1) reverse lookup of the key this node ran under.
                    let key = node_key_of.remove(&outcome.node_id);
                    outputs.insert(outcome.node_id, outcome.output);
                    logical_outputs.insert(outcome.node_id, outcome.logical);

                    // Release any nodes deferred behind this key — they
                    // can now cache-hit. Re-add them to the ready set.
                    if let Some(k) = key {
                        inflight_keys.remove(&k);
                        if let Some(waiters) = deferred.remove(&k) {
                            for w in waiters {
                                ready.insert(w);
                            }
                        }
                    }

                    // Decrement successors' in-degrees; newly-zero → ready.
                    if first_error.is_none() {
                        if let Some(ss) = succs.get(&outcome.node_id) {
                            for &s in ss {
                                if let Some(d) = indeg.get_mut(&s) {
                                    *d -= 1;
                                    if *d == 0 {
                                        ready.insert(s);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(NodeFailure::Cancelled) => {
                    // A sibling cancelled (or this node observed the token
                    // after a peer failed). Never the PRIMARY error — only
                    // record Cancelled if nothing else failed.
                    if first_error.is_none() {
                        first_error = Some(PlanError::Cancelled);
                        env.cancel.cancel();
                    }
                }
                Err(f) => {
                    if first_error.is_none() {
                        first_error = Some(plan_error_of(f));
                        env.cancel.cancel(); // fail-fast: cancel siblings
                    }
                    // else: a later error after we've already started
                    // failing — drop it; the first error wins.
                }
            }
        }

        // All tasks drained.
        if let Some(err) = first_error {
            finish_writer(env, writer_handle).await;
            return Err(err);
        }

        debug_assert_eq!(
            completed,
            order.len(),
            "parallel executor must complete every node on success"
        );
        let final_output = order.last().and_then(|id| outputs.remove(id));
        finish_writer(env, writer_handle).await;

        Ok(PlanResult {
            final_output,
            n_stages: order.len(),
            n_cache_hits: n_hits,
            n_cache_misses: n_misses,
            elapsed: started.elapsed(),
        })
    }
}

/// Logical output hash for a stage — the value downstream stages fold
/// into THEIR cache key. Deterministic stages report the real content
/// address (so byte-identical content at different paths / machines
/// yields the same downstream key — FW-1). Nondet stages report a
/// synthesized fingerprint = hash(stage_name ‖ schema ‖ input_hash ‖
/// args), byte-stable across stochastic re-runs even when ckpt bytes
/// differ.
fn compute_logical_output_hash(
    stage: &dyn StageDyn,
    output: &ErasedArtifact,
    deterministic: bool,
    stage_name: &str,
    schema: u32,
    input_hash: ContentHash,
    canon_args: &[u8],
) -> ContentHash {
    if deterministic {
        return stage
            .output_content_hash(output)
            .unwrap_or_else(|| content_hash_from_erased(output));
    }
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"blut.nondet.v1");
    h.update([0u8]);
    h.update(stage_name.as_bytes());
    h.update([0u8]);
    h.update(schema.to_le_bytes());
    h.update(input_hash.0);
    h.update(canon_args);
    ContentHash(h.finalize().into())
}

fn content_hash_from_erased(art: &ErasedArtifact) -> ContentHash {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(art.kind.as_bytes());
    h.update(art.schema.to_le_bytes());
    h.update(&art.payload);
    ContentHash(h.finalize().into())
}

// intentional (BLD-C2): the executor tests serialize on a process-wide
// `TEST_LOCK` std Mutex held across `.await` to stop concurrent tests from
// racing on the shared content cache / job dirs. A std guard across an await
// is exactly the pattern clippy flags, but here it is the deliberate
// serialization mechanism — the guard is never contended by real async work,
// only by the test harness, so it cannot deadlock the runtime.
#[allow(clippy::await_holding_lock)]
#[cfg(test)]
mod tests {
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
        assert_eq!(r2.n_cache_hits, 2, "second run should hit cache for both stages");
        assert_eq!(r2.n_cache_misses, 0);
        assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 1);
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
        assert!(sidecar.exists(), "expected sidecar at {}", sidecar.display());
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
        assert_ne!(observed, handle_hash, "content hash must differ from handle hash here");
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
        assert!(matches!(r, Err(PlanError::StageFailed { .. })), "stage must fail");
        let final_dir = write_then_final_dir(&job_dir);
        assert!(!final_dir.exists(), "FW-2: no final dir after error");
        assert!(leftover_tmp_dirs(&job_dir).is_empty(), "FW-2: tmp removed on error");
        assert_eq!(cache_entry_count(&job_dir), 0, "FW-2: failed stage not cached");
    }

    #[tokio::test]
    async fn partial_output_cleaned_on_cancel() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let job_dir = td.path().to_path_buf();
        let ctx = ExecCtx::new(job_dir.clone());
        let plan = Plan::<(), LamuTrainerBackend>::new("fw2-cancel", serde_json::json!({}))
            .start(WriteThen, WriteThenArgs { mode: "cancel".into() })
            .finish()
            .into_compiled();
        let r = SequentialExecutor::execute(plan, ctx).await;
        assert!(matches!(r, Err(PlanError::Cancelled)), "mid-stage cancel → Cancelled, got {r:?}");
        let final_dir = write_then_final_dir(&job_dir);
        assert!(!final_dir.exists(), "FW-2: cancelled stage leaves no partial");
        assert!(leftover_tmp_dirs(&job_dir).is_empty(), "FW-2: tmp removed on cancel");
        assert_eq!(cache_entry_count(&job_dir), 0, "FW-2: cancelled stage not cached");
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
        assert!(final_dir.join("partial.txt").exists(), "promoted output present");
        assert!(final_dir.join("output.metadata.json").exists(), "sidecar in promoted dir");
        assert!(leftover_tmp_dirs(&job_dir).is_empty(), "no tmp survives promote");
        assert_eq!(cache_entry_count(&job_dir), 1, "FW-2: successful stage cached");
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
        assert!(final_dir.join("partial.txt").exists(), "fresh output present");
        assert!(!final_dir.join("orphan.txt").exists(), "FW-2: stale orphan gone");
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
        assert_eq!(peak.load(Ordering::SeqCst), 2, "two CPU stages must run concurrently");
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

        assert_eq!(seq_out.n, par_out.n, "both executors produce the same output");
        assert_eq!(
            par.n_cache_hits, par.n_stages,
            "parallel run must hit the sequential run's cache for every stage (identical keys)"
        );
        assert_eq!(par.n_cache_misses, 0);
    }
}
