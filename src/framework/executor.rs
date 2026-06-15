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

use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::framework::artifact::{ArtifactMetadata, ContentHash};
use crate::framework::cache::CacheHandle;
use crate::framework::control::{Control, ControlPolicy, StepMetrics};
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

/// Default memory-admission budget (GiB): effectively unlimited, so a stage's
/// `MEMORY_GIB` reservation never blocks until the CLI sizes the budget to
/// box-fit. Picked large enough to never gate, small enough to stay a valid
/// `tokio::Semaphore` permit count.
pub const UNLIMITED_MEM_GIB: u32 = 1_000_000;

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
    /// Optional plan-level deadline (D2). When `Instant::now()` reaches
    /// it the executor cancels and returns `PlanError::DeadlineExceeded`.
    pub deadline: Option<Instant>,
    /// Optional retry hook (D1). Invoked when a stage attempt fails with
    /// a retryable error, BEFORE the backoff. The cookbook wires the
    /// broker's OOM-escalation here; the framework stays broker-agnostic.
    pub on_retry: Option<crate::framework::retry::RetryHook>,
    /// Capacity-aware memory admission (Phase 5). A stage holds
    /// `Stage::MEMORY_GIB` permits from this for its whole run; the budget is
    /// the box-fit GiB (`MemTotal − floor`). Concurrent stages can't acquire
    /// more than the budget in total → never-OOM-the-BOX under the parallel
    /// executor. Default budget is effectively unlimited (no gating); the CLI
    /// sizes it to box-fit via `with_memory_budget`.
    pub memory: Arc<tokio::sync::Semaphore>,
    pub memory_budget_gib: u32,
    /// Where stages place their work (#3). `Local` (default) = this box; a
    /// launcher-aware backend reads this from `StageContext` to submit to
    /// Slurm/Ray instead. Set by the CLI `--launcher` flag.
    pub launch_target: crate::config::launcher::LaunchTarget,
    /// Runtime DAG control policy (#4). `None` (default) = the static plan
    /// runs unchanged — the executor never watches the metric stream, so the
    /// behaviour is byte-identical to the pre-control path. `Some(policy)`
    /// (parallel executor only) consults the policy against each live
    /// `StageStep` and may `KillBranch` a diverged node. Set the env
    /// `BLUT_KILL_ON_NAN=1` (or call `with_control`) to wire the built-in
    /// `KillOnNaN`.
    pub control: Option<Arc<dyn ControlPolicy>>,
    /// Never-OOM Phase 3: was the fullband disk cache warmed upstream? Threaded
    /// into every `StageContext` so a train stage bills the warm (lower)
    /// per-worker footprint + the `|w` calibration key. Set by the CLI from the
    /// recipe's `warm_fb_cache` arg (the SAME source the admission gate reads),
    /// so RECORD and RESOLVE never disagree. Default false (cold). NOT a stage
    /// Arg — warm doesn't change the trained output, so it stays out of the
    /// checkpoint cache key.
    pub fb_warm: bool,
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
            deadline: None,
            on_retry: None,
            // Effectively unlimited until the CLI sizes it to box-fit; a stage
            // requesting MEMORY_GIB ≪ this never blocks, so default = no gating.
            memory: Arc::new(tokio::sync::Semaphore::new(UNLIMITED_MEM_GIB as usize)),
            memory_budget_gib: UNLIMITED_MEM_GIB,
            launch_target: crate::config::launcher::LaunchTarget::Local,
            control: None,
            fb_warm: false,
        }
    }

    /// Place stages on `target` (#3). Default `Local`.
    pub fn with_launch_target(mut self, target: crate::config::launcher::LaunchTarget) -> Self {
        self.launch_target = target;
        self
    }

    /// Mark the fullband cache as warmed upstream (Phase 3). Threaded into every
    /// `StageContext.fb_warm` so a train stage bills the warm footprint.
    pub fn with_fb_warm(mut self, warm: bool) -> Self {
        self.fb_warm = warm;
        self
    }

    /// Wire a runtime DAG control policy (#4). With a policy set, the
    /// PARALLEL executor watches the live step stream and can prune a
    /// diverged branch. `None` (default) = static plan, no watching.
    pub fn with_control(mut self, policy: Arc<dyn ControlPolicy>) -> Self {
        self.control = Some(policy);
        self
    }

    pub fn with_resource_limit(mut self, resource: Resource, permits: usize) -> Self {
        self.resources
            .insert(resource, Arc::new(tokio::sync::Semaphore::new(permits)));
        self
    }

    /// Size the memory admission budget to `gib` (box-fit = `MemTotal − floor`).
    /// A stage's `MEMORY_GIB` is clamped to this, so a stage needing the whole
    /// box runs alone rather than deadlocking.
    pub fn with_memory_budget(mut self, gib: u32) -> Self {
        let gib = gib.max(1);
        self.memory_budget_gib = gib;
        self.memory = Arc::new(tokio::sync::Semaphore::new(gib as usize));
        self
    }

    pub fn with_max_in_flight(mut self, n: usize) -> Self {
        self.max_in_flight = n.max(1);
        self
    }

    pub fn with_deadline(mut self, after: std::time::Duration) -> Self {
        self.deadline = Some(Instant::now() + after);
        self
    }

    pub fn with_retry_hook(
        mut self,
        hook: crate::framework::retry::RetryHook,
    ) -> Self {
        self.on_retry = Some(hook);
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
    memory: Arc<tokio::sync::Semaphore>,
    memory_budget_gib: u32,
    launch_target: crate::config::launcher::LaunchTarget,
    fb_warm: bool,
    recipe_name: String,
    on_retry: Option<crate::framework::retry::RetryHook>,
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
    /// Resolved retry policy + timeout (node override, else stage const).
    retry: crate::framework::retry::RetryPolicy,
    timeout: crate::framework::retry::StageTimeout,
    /// Per-node cancellation token (#4). A CHILD of the plan token, so a
    /// plan-wide cancel still propagates here, but the coordinator can ALSO
    /// fire it alone to kill THIS node's branch (KILL-on-NaN) without
    /// touching siblings. With no control policy it only ever fires via the
    /// parent → behaviour is identical to the pre-#4 single-token path.
    node_cancel: CancellationToken,
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
    /// This node's OWN token fired while the plan token did NOT (#4): a
    /// targeted KILL-on-NaN, not a plan-wide cancel. The coordinator prunes
    /// this node's descendants and continues other branches — it is NOT a
    /// plan failure. Carries the node id so the coordinator knows which
    /// subtree to prune. Output discarded, nothing cached (FW-2 cleanup).
    Killed { node_id: NodeId },
    /// The stage (or its promote) failed.
    Stage {
        idx: u32,
        stage: String,
        source: StageError,
    },
    /// An executor-internal failure (e.g. a closed semaphore).
    Other(String),
}

/// Classify a cancel observed inside `run_node`: a targeted KILL (this
/// node's token fired but the plan token did NOT) vs a plan-wide cancel.
/// The distinction is what lets the coordinator prune one branch on a kill
/// instead of failing the whole plan.
fn cancel_failure(
    node_id: NodeId,
    node_cancel: &CancellationToken,
    plan_cancel: &CancellationToken,
) -> NodeFailure {
    if node_cancel.is_cancelled() && !plan_cancel.is_cancelled() {
        NodeFailure::Killed { node_id }
    } else {
        NodeFailure::Cancelled
    }
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
    // (or a re-run with different args) never collide. The attempt loop
    // (D1) recreates the tmp dir + reacquires permits per attempt, so
    // FW-2 holds for EACH attempt; the cache insert is still strictly
    // post-promote (below the loop, on success).
    let stages_root = env.job_dir.join("stages");
    let final_stage_dir = stages_root.join(format!("{idx}-{stage_name}"));
    let tmp_stage_dir = stages_root.join(format!(".tmp-{idx}-{stage_name}-{}", task.key.to_hex()));

    let mut attempt = 0u32;
    let (output, run_elapsed) = loop {
        attempt += 1;
        // Backoff before a re-attempt — cancellable (a backing-off stage
        // must drop the GPU/permits, which it already has by here).
        if attempt > 1 {
            let backoff = task.retry.backoff_before(attempt);
            if !backoff.is_zero() {
                // Wake on EITHER a plan cancel or a targeted kill (the node
                // token is a child of the plan token, so it fires on both).
                tokio::select! {
                    _ = tokio::time::sleep(backoff) => {}
                    _ = task.node_cancel.cancelled() => {
                        env.status.emit(StageEvent::StageFailed {
                            node_idx: idx,
                            stage_name: stage_name.clone(),
                            error: "cancelled during retry backoff".into(),
                        });
                        return Err(cancel_failure(task.node_id, &task.node_cancel, &env.cancel));
                    }
                }
            }
        }
        if task.node_cancel.is_cancelled() {
            env.status.emit(StageEvent::StageFailed {
                node_idx: idx,
                stage_name: stage_name.clone(),
                error: "cancelled before stage attempt".into(),
            });
            return Err(cancel_failure(task.node_id, &task.node_cancel, &env.cancel));
        }

        // Clean tmp per attempt — each attempt starts from an empty dir.
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

        // A child cancel token so a SOFT timeout (D2) can wind THIS stage
        // down cooperatively without touching the plan token / siblings;
        // a plan cancel still propagates (child tokens fire on parent).
        // Rooted at the PER-NODE token (#4), so a targeted KILL-on-NaN fires
        // it via that parent exactly as a plan cancel would — the stage's own
        // cancel handling is unchanged.
        let stage_cancel = task.node_cancel.child_token();
        let mut stage_ctx = StageContext {
            job_dir: env.job_dir.clone(),
            stage_dir: tmp_stage_dir.clone(),
            node_idx: idx,
            status_tx: env.status.broadcast_sender(),
            cancel: stage_cancel.clone(),
            cache: env.cache.clone(),
            recipe_name: env.recipe_name.clone(),
            launch_target: env.launch_target,
            fb_warm: env.fb_warm,
            // Durable resume (Phase D): the stage's cache key is its stable
            // per-config fingerprint — a resume train stage keys its recovery
            // dir on it so a re-run with identical args finds the checkpoint.
            cache_key: task.key,
            attempt,
            resume_from: None,
        };

        // ── Auto-resume on retry (S3 / P7) ──────────────────────────
        // On a re-attempt, ask the stage WHERE its checkpoint lives
        // (`resume_handle`, default None = not resumable → no-op), then run the
        // EXISTING crash-gated `decide_resume` against the marker there. A
        // same-run_id retry deterministically `Resume`s; a live foreign run
        // `RefuseConcurrent`s (we must not race two trainers on one dir). The
        // Executor owns the resume axis: it sets `resume_from`, the stage reads
        // it and appends `--resume`. Non-resumable stages re-run unchanged.
        if attempt > 1 {
            // A job dir with no basename (e.g. a filesystem root) can't yield a
            // run identity — never auto-resume in that case (re-run fresh rather
            // than risk matching a foreign empty run_id). job_dir is always
            // `<jobs>/<job_id>` in practice, so this guard is belt-and-suspenders.
            let run_id = env
                .job_dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .filter(|s| !s.is_empty());
            if let (Some(run_id), Some(token)) =
                (run_id, task.stage.resume_handle_erased(&stage_ctx, &task.args))
            {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let state = crate::framework::resume::ResumeState::read(&token.resume_dir);
                match crate::framework::resume::decide_resume(
                    state.as_ref(),
                    &run_id,
                    now,
                    crate::framework::resume::DEFAULT_STALE_AFTER_SECS,
                ) {
                    crate::framework::resume::ResumeDecision::Resume => {
                        tracing::info!(
                            "auto-resume node {idx} ({stage_name}) attempt {attempt} from {}",
                            token.resume_dir.display()
                        );
                        stage_ctx.resume_from = Some(token.resume_dir);
                    }
                    crate::framework::resume::ResumeDecision::Fresh => {}
                    crate::framework::resume::ResumeDecision::RefuseConcurrent => {
                        let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                        let msg = format!(
                            "resume checkpoint for '{stage_name}' owned by a live run: {}",
                            token.resume_dir.display()
                        );
                        env.status.emit(StageEvent::StageFailed {
                            node_idx: idx,
                            stage_name: stage_name.clone(),
                            error: msg.clone(),
                        });
                        // TRANSIENT: back off + retry (each attempt re-checks the
                        // marker) — never resume concurrently, but the blocker
                        // finishes. After max_attempts, this is the terminal error.
                        return Err(NodeFailure::Stage {
                            idx,
                            stage: stage_name,
                            source: StageError::CheckpointBusy { detail: msg },
                        });
                    }
                }
            }
        }

        // ── Resource permits ────────────────────────────────────────
        // Acquire in canonical (sorted) order so two concurrent stages
        // can never deadlock on the same pair in opposite orders.
        // `try_acquire` first; only emit `StageBlocked` on contention.
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

        // ── Memory admission (Phase 5) ──────────────────────────────
        // Hold MEMORY_GIB permits from the box-fit budget for the whole run, so
        // the SUM of concurrent stages can't exceed the box (never-OOM-the-BOX
        // under the parallel executor). Clamp to the budget so a stage needing
        // the whole box runs alone instead of deadlocking. `0` = no reservation.
        let mem_want = task.stage.memory_gib().min(env.memory_budget_gib);
        let _mem_permit = if mem_want > 0 {
            match env.memory.clone().acquire_many_owned(mem_want).await {
                Ok(p) => Some(p),
                Err(_) => {
                    let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                    return Err(NodeFailure::Other("memory semaphore closed".into()));
                }
            }
        } else {
            None
        };

        let stage_started = Instant::now();
        let run_fut =
            task.stage
                .run_erased(&stage_ctx, task.input.clone(), task.args.clone());
        let run_result = run_with_timeout(
            run_fut,
            &stage_cancel,
            task.timeout.soft,
            task.timeout.hard,
            stage_started,
        )
        .await;
        // Permits drop here, releasing the resource for queued stages
        // (including during a backoff before the next attempt).
        drop(permits);
        drop(stage_ctx);

        match run_result {
            Ok(o) => {
                debug_assert_eq!(
                    o.kind,
                    task.stage.output_kind(),
                    "stage '{stage_name}' produced kind '{}' but declares output_kind '{}'",
                    o.kind,
                    task.stage.output_kind()
                );
                // A cancel observed during the run must NOT be promoted /
                // cached — discard, report Cancelled. Check the STAGE
                // token: it fires both on a plan cancel (child inherits
                // the parent) AND on a stage's own cooperative cancel.
                if stage_cancel.is_cancelled() {
                    let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                    env.status.emit(StageEvent::StageFailed {
                        node_idx: idx,
                        stage_name,
                        error: "cancelled during stage".into(),
                    });
                    return Err(cancel_failure(task.node_id, &task.node_cancel, &env.cancel));
                }
                // StageEnd reports the SUCCESSFUL attempt's wall time;
                // failed attempts + backoff are visible as StageRetrying
                // events, not folded into this duration.
                break (o, stage_started.elapsed());
            }
            Err(e) => {
                let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                // The node token fired on EITHER a plan cancel or a targeted
                // kill — a killed node must never retry (it would re-run the
                // doomed work), so gate retry on the node token, not just the
                // plan token (the node token is a superset).
                let token_fired = task.node_cancel.is_cancelled();
                let retry = attempt < task.retry.max_attempts
                    && !token_fired
                    && crate::framework::retry::is_retryable(&e, task.retry.retry_on);
                if retry {
                    // OOM-escalation / observability hook (broker wiring).
                    if let Some(hook) = &env.on_retry {
                        hook(&crate::framework::retry::RetryEvent {
                            stage_name: stage_name.clone(),
                            recipe_name: env.recipe_name.clone(),
                            attempt,
                            max_attempts: task.retry.max_attempts,
                            was_oom: matches!(e, StageError::OutOfMemory { .. }),
                            error: format!("{e}"),
                        });
                    }
                    let next_backoff = task.retry.backoff_before(attempt + 1);
                    env.status.emit(StageEvent::StageRetrying {
                        node_idx: idx,
                        stage_name: stage_name.clone(),
                        attempt,
                        max_attempts: task.retry.max_attempts,
                        error: format!("{e}"),
                        backoff_ms: next_backoff.as_millis() as u64,
                    });
                    continue;
                }
                // Terminal failure.
                let cancelled = token_fired || matches!(e, StageError::Cancelled);
                env.status.emit(StageEvent::StageFailed {
                    node_idx: idx,
                    stage_name: stage_name.clone(),
                    error: format!("{e}"),
                });
                if cancelled {
                    // Targeted kill → Killed (prune branch); plan cancel or a
                    // bare Cancelled error → Cancelled (fail-fast).
                    return Err(cancel_failure(task.node_id, &task.node_cancel, &env.cancel));
                }
                return Err(NodeFailure::Stage {
                    idx,
                    stage: stage_name,
                    source: e,
                });
            }
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

    // Sidecar metadata next to the promoted payload. Record the CONTENT hash
    // (the FW-1 fix path, same as the downstream logical hash uses) so the
    // StageEnd event + the output.metadata.json sidecar + lineage are
    // cross-machine-stable — `content_hash_from_erased` hashes the bincode
    // handle, which embeds the producer's absolute paths. Falls back to the
    // handle hash for non-deterministic / tuple outputs (the same well-tested
    // fallback `compute_logical_output_hash` uses).
    let output_hash = task
        .stage
        .output_content_hash(&output)
        .unwrap_or_else(|| content_hash_from_erased(&output));
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
        elapsed: run_elapsed,
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

/// Bound one stage attempt by an optional soft/hard timeout (D2). With
/// neither set, awaits the run directly. The SOFT deadline fires
/// `stage_cancel` (cooperative wind-down — a Python trainer SIGTERMs its
/// child); a run that then returns is FAILED with `Timeout` (its output
/// exceeded budget and must not be promoted). The HARD deadline drops
/// the run future (its `kill_on_drop` subprocess is reaped) and returns
/// `Timeout`.
async fn run_with_timeout(
    run_fut: impl std::future::Future<Output = Result<ErasedArtifact, StageError>>,
    stage_cancel: &CancellationToken,
    soft: Option<std::time::Duration>,
    hard: Option<std::time::Duration>,
    started: Instant,
) -> Result<ErasedArtifact, StageError> {
    if soft.is_none() && hard.is_none() {
        return run_fut.await;
    }
    tokio::pin!(run_fut);
    let soft_at = soft.map(|d| started + d);
    let hard_at = hard.map(|d| started + d);
    let mut soft_fired = false;
    loop {
        tokio::select! {
            res = &mut run_fut => {
                return if soft_fired {
                    Err(StageError::Timeout { limit: soft.unwrap(), elapsed: started.elapsed() })
                } else {
                    res
                };
            }
            _ = sleep_until_opt(soft_at), if soft_at.is_some() && !soft_fired => {
                soft_fired = true;
                stage_cancel.cancel();
            }
            _ = sleep_until_opt(hard_at), if hard_at.is_some() => {
                return Err(StageError::Timeout {
                    limit: hard.unwrap(),
                    elapsed: started.elapsed(),
                });
            }
        }
    }
}

/// Sleep until `at`, or never (pending) when `None` — lets a
/// `tokio::select!` branch be a no-op for an unset deadline.
async fn sleep_until_opt(at: Option<Instant>) {
    match at {
        Some(t) => tokio::time::sleep_until(tokio::time::Instant::from_std(t)).await,
        None => std::future::pending::<()>().await,
    }
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
    node_cancel: CancellationToken,
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
    // Resolve retry/timeout: a per-node override wins over the stage const.
    let retry = node.retry.unwrap_or_else(|| node.stage.retry());
    let timeout = node.timeout.unwrap_or_else(|| node.stage.timeout());
    Ok(NodeTask {
        node_id: node.id,
        node_idx,
        stage: node.stage.clone(),
        args: node.args.clone(),
        canon_args: node.canon_args.clone(),
        input,
        input_hash,
        key,
        retry,
        timeout,
        node_cancel,
    })
}

/// Resolve a node id to its `PlanNode` across the ORIGINAL plan (borrowed
/// immutably for the run) and the runtime-`Spawn`-`appended` side-vec. Spawned
/// ids are `orig_n + appended_index`, so the split is a single bound check. Kept
/// a free fn (not a closure) so it never holds a borrow across an `appended`
/// push — the two happen at different points in the coordinator loop.
fn node_at<'a>(
    view: &'a crate::framework::plan::ExecView<'a>,
    appended: &'a [crate::framework::plan::PlanNode],
    orig_n: usize,
    id: NodeId,
) -> &'a crate::framework::plan::PlanNode {
    let i = id as usize;
    if i < orig_n {
        &view.nodes[i]
    } else {
        &appended[i - orig_n]
    }
}

/// Hard backstop on runtime spawns — a runaway policy must not append forever.
/// Real PBT/TPE runs spawn O(trials), far below this; it only guards a bug.
const MAX_RUNTIME_SPAWNS: usize = 4096;

/// Inject a `Spawn` delta into the running parallel schedule (v0.20). The
/// sub-plan's local node ids `0..k` are relabelled to globals `base + l`
/// (`base = orig_n + appended.len()`), its nodes moved into `appended`, its
/// edges/in-degrees/successors/topo-order extended, its graph-inputs seeded as
/// root outputs, and its roots inserted into `ready`. Returns the count
/// injected. Errors only if the sub-plan is cyclic/empty (caller logs + skips).
#[allow(clippy::too_many_arguments)]
fn inject_spawn(
    delta: crate::framework::control::SpawnDelta,
    orig_n: usize,
    appended: &mut Vec<crate::framework::plan::PlanNode>,
    all_edges: &mut Vec<crate::framework::plan::PlanEdge>,
    order: &mut Vec<NodeId>,
    node_idx_of: &mut HashMap<NodeId, u32>,
    indeg: &mut HashMap<NodeId, usize>,
    succs: &mut HashMap<NodeId, Vec<NodeId>>,
    ready: &mut BTreeSet<NodeId>,
    outputs: &mut HashMap<NodeId, ErasedArtifact>,
    logical_outputs: &mut HashMap<NodeId, ContentHash>,
) -> Result<usize, PlanError> {
    use crate::framework::plan::PlanEdge;
    let subplan = delta.subplan;
    // Local topo order (also the cycle/empty check) BEFORE we mutate anything.
    let local_order = subplan.topo_order()?;
    let base = (orig_n + appended.len()) as NodeId;
    let (nodes, edges, initial) = subplan.into_parts();
    let k = nodes.len();

    // Move nodes in LOCAL-ID ORDER so `appended[base - orig_n + l].id == base + l`
    // — the invariant `node_at` relies on. (Compiled sub-plans have node.id == its
    // index; assert it so a future builder change can't silently break the map.)
    for (local_id, mut node) in nodes.into_iter().enumerate() {
        debug_assert_eq!(
            node.id as usize, local_id,
            "spawned sub-plan node ids must be dense 0..k in index order"
        );
        let gid = base + local_id as NodeId;
        node.id = gid;
        appended.push(node);
        indeg.entry(gid).or_insert(0);
        succs.entry(gid).or_default();
    }
    // Relabel + wire edges.
    for e in &edges {
        let g = PlanEdge { from: base + e.from, to: base + e.to };
        all_edges.push(g);
        *indeg.entry(g.to).or_insert(0) += 1;
        succs.entry(g.from).or_default().push(g.to);
    }
    // Seed the sub-plan's graph-inputs as root outputs (mirrors `prelude`).
    for (local_id, art) in initial {
        let gid = base + local_id;
        let lh = content_hash_from_erased(&art);
        outputs.insert(gid, art);
        logical_outputs.insert(gid, lh);
    }
    // Extend topo order (node_idx == position) in the sub-plan's topo order.
    for &lid in &local_order {
        let gid = base + lid;
        node_idx_of.insert(gid, order.len() as u32);
        order.push(gid);
    }
    // Roots (global in-degree 0) become runnable now.
    for l in 0..k as NodeId {
        let gid = base + l;
        if indeg.get(&gid).copied().unwrap_or(0) == 0 {
            ready.insert(gid);
        }
    }
    Ok(k)
}

/// Map a `NodeFailure` to a `PlanError`.
fn plan_error_of(f: NodeFailure) -> PlanError {
    match f {
        NodeFailure::Cancelled => PlanError::Cancelled,
        // A `Killed` reaching here means a control policy fired on a path
        // that doesn't special-case it (the sequential executor, which wires
        // no policy, so this is unreachable there). Map to Cancelled — a
        // pruned branch is a caller-requested stop, never a stage error.
        NodeFailure::Killed { .. } => PlanError::Cancelled,
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
        memory: ctx.memory,
        memory_budget_gib: ctx.memory_budget_gib,
        launch_target: ctx.launch_target,
        fb_warm: ctx.fb_warm,
        recipe_name: plan.name().to_string(),
        on_retry: ctx.on_retry,
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
pub async fn execute_plan(plan: CompiledPlan, mut ctx: ExecCtx) -> Result<PlanResult, PlanError> {
    // #4: `BLUT_KILL_ON_NAN=1` wires the built-in KillOnNaN policy (unless a
    // caller already set one). Runtime control needs the live step watcher,
    // which only the PARALLEL executor runs — so a policy forces parallel.
    if ctx.control.is_none()
        && std::env::var("BLUT_KILL_ON_NAN")
            .map(|v| v == "1")
            .unwrap_or(false)
    {
        ctx = ctx.with_control(Arc::new(crate::framework::control::KillOnNaN));
    }
    let parallel = ctx.control.is_some()
        || std::env::var("BLUT_EXECUTOR")
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

        let deadline = ctx.deadline;
        let Prelude {
            writer_handle,
            env,
            mut outputs,
            mut logical_outputs,
        } = prelude(ctx, &plan)?;

        let mut n_hits = 0usize;
        let mut n_misses = 0usize;

        for (idx, node_id) in order.iter().enumerate() {
            // Plan-level deadline (D2): coarse between-stage check; a
            // stage mid-run is bounded by its own hard timeout instead.
            if let Some(dl) = deadline {
                if Instant::now() >= dl {
                    env.cancel.cancel();
                    finish_writer(env, writer_handle).await;
                    return Err(PlanError::DeadlineExceeded {
                        elapsed: started.elapsed(),
                    });
                }
            }
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
            // Per-node child token (#4). Sequential wires no control policy, so
            // it only ever fires via the plan token → identical to the prior
            // single-token behaviour.
            let node_cancel = env.cancel.child_token();
            let task = match build_task(
                node,
                idx as u32,
                view.edges,
                &outputs,
                &logical_outputs,
                node_cancel,
            ) {
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
        // `mut`: runtime `Spawn` (PBT/TPE) extends the topo order at runtime.
        let mut order = plan.topo_order()?; // also the cycle check
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
        let deadline = ctx.deadline;
        // #4: pull the control policy out BEFORE prelude consumes ctx. `None`
        // → the watcher is never subscribed and the loop is byte-identical to
        // the pre-control path.
        let control = ctx.control.clone();
        let Prelude {
            writer_handle,
            env,
            mut outputs,
            mut logical_outputs,
        } = prelude(ctx, &plan)?;

        // #4 runtime control state. `control_rx` is the live step stream the
        // coordinator watches between joins; `node_tokens` maps an in-flight
        // node to its kill token; `pruned` is the set of nodes a kill removed
        // from the schedule (the killed node + its descendants); they plus
        // `completed` must cover every node at the end.
        let mut control_rx: Option<broadcast::Receiver<StageEvent>> =
            control.as_ref().map(|_| env.status.subscribe());
        let mut node_tokens: HashMap<NodeId, CancellationToken> = HashMap::new();

        // #4 runtime SPAWN (PBT/TPE) state. The original plan's nodes stay
        // borrowed through `view` (immutable for the whole run); nodes appended
        // at runtime live in `appended` (ids `orig_n + idx`), reached via
        // `node_at`. `all_edges` starts as the plan's edges and grows with each
        // injected sub-plan. `pending_spawns` is filled by the watcher inside
        // `select!` and DRAINED at the top of the loop (never mid-`select!`), so
        // schedule mutation only happens on the single-threaded coordinator seam.
        let orig_n = view.nodes.len();
        let mut appended: Vec<crate::framework::plan::PlanNode> = Vec::new();
        let mut all_edges: Vec<crate::framework::plan::PlanEdge> = view.edges.to_vec();
        let mut pending_spawns: Vec<crate::framework::control::SpawnDelta> = Vec::new();
        let mut spawns_total = 0usize;
        // The killed nodes + their pruned descendants. `pruned.len()` (not a
        // parallel counter) is the accounting source of truth — it can't drift
        // out of sync with the set the spawn loop consults.
        let mut pruned: HashSet<NodeId> = HashSet::new();

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
            // Plan-level deadline (D2): once past it, stop spawning new
            // nodes, cancel + drain the in-flight ones, report
            // DeadlineExceeded (a stage mid-run is bounded by its own
            // hard timeout).
            if first_error.is_none() {
                if let Some(dl) = deadline {
                    if Instant::now() >= dl {
                        first_error = Some(PlanError::DeadlineExceeded {
                            elapsed: started.elapsed(),
                        });
                        env.cancel.cancel();
                    }
                }
            }
            // #4 SPAWN: drain runtime-injected sub-plans on the coordinator seam
            // BEFORE the spawn-ready loop, so newly-ready roots are scheduled
            // this iteration and the `in_flight == 0` termination check below
            // sees them. Stop injecting once failing (drop pending deltas).
            if first_error.is_none() && !pending_spawns.is_empty() {
                for delta in pending_spawns.drain(..) {
                    if spawns_total >= MAX_RUNTIME_SPAWNS {
                        tracing::warn!(
                            "runtime spawn cap {MAX_RUNTIME_SPAWNS} reached; dropping further spawns"
                        );
                        break;
                    }
                    match inject_spawn(
                        delta,
                        orig_n,
                        &mut appended,
                        &mut all_edges,
                        &mut order,
                        &mut node_idx_of,
                        &mut indeg,
                        &mut succs,
                        &mut ready,
                        &mut outputs,
                        &mut logical_outputs,
                    ) {
                        Ok(k) => spawns_total += k,
                        Err(e) => tracing::warn!("ignored malformed spawn delta: {e}"),
                    }
                }
            }
            // Spawn ready nodes up to the in-flight cap (unless we're
            // already failing — then stop spawning and just drain).
            if first_error.is_none() {
                while in_flight < max_in_flight {
                    let Some(&node_id) = ready.iter().next() else {
                        break;
                    };
                    ready.remove(&node_id);
                    // #4: a node pruned by a KILL-on-NaN upstream must never be
                    // scheduled — its input can't materialize. (A pruned node
                    // can land in `ready` if it was already there when the kill
                    // happened; skip it here.)
                    if pruned.contains(&node_id) {
                        continue;
                    }
                    let node = node_at(&view, &appended, orig_n, node_id);
                    let node_idx = node_idx_of[&node_id];
                    // Per-node kill token: a child of the plan token, retained
                    // in `node_tokens` so the control watcher can fire it alone.
                    let node_cancel = env.cancel.child_token();
                    let task = match build_task(
                        node,
                        node_idx,
                        &all_edges,
                        &outputs,
                        &logical_outputs,
                        node_cancel.clone(),
                    ) {
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
                    // Retain the kill token only for an actually-spawned node
                    // (a deferred node `continue`s above; its token is dropped
                    // and a fresh one is built when it re-enters `ready`).
                    node_tokens.insert(node_id, node_cancel);
                    let env_c = env.clone();
                    join.spawn(async move { run_node(task, env_c).await });
                    in_flight += 1;
                }
            }

            if in_flight == 0 {
                break; // nothing running and nothing spawnable → done
            }

            // Await the next completed node. With a control policy set (#4),
            // concurrently watch the live StageStep stream and apply runtime
            // graph control (KILL-on-NaN) on the SAME coordinator thread, so a
            // kill decision can never race the FW-2 promote / scheduler state.
            // With no policy, this is a plain `join_next` → byte-identical.
            let joined = match control_rx.as_mut() {
                None => join.join_next().await,
                Some(rx) => {
                    loop {
                        tokio::select! {
                            // `biased`: always make scheduling progress first —
                            // a completed node takes priority over a step event.
                            biased;
                            j = join.join_next() => break j,
                            ev = rx.recv() => {
                                match ev {
                                    Ok(StageEvent::StageStep { node_idx, stage_name, update }) => {
                                        if let Some(policy) = control.as_ref() {
                                            let m = StepMetrics {
                                                node_idx,
                                                stage_name: &stage_name,
                                                update: &update,
                                            };
                                            match policy.on_step(&m) {
                                                Control::Continue => {}
                                                Control::KillBranch => {
                                                    // Target the EMITTING node:
                                                    // topo idx → node id → its token.
                                                    if let Some(&nid) = order.get(node_idx as usize) {
                                                        if let Some(tok) = node_tokens.get(&nid) {
                                                            if !tok.is_cancelled() {
                                                                tracing::warn!(
                                                                    "control policy KILL on node {node_idx} \
                                                                     ({stage_name}): non-finite step metric"
                                                                );
                                                                tok.cancel();
                                                            }
                                                        }
                                                    }
                                                }
                                                // Queue the delta; it is injected at
                                                // the top of the loop (never mid-select!).
                                                // The QUEUE is capped so a runaway policy
                                                // can't grow it unbounded between joins.
                                                Control::Spawn(delta) if first_error.is_none() => {
                                                    if spawns_total + pending_spawns.len()
                                                        < MAX_RUNTIME_SPAWNS
                                                    {
                                                        pending_spawns.push(delta);
                                                    } else {
                                                        tracing::warn!(
                                                            "runtime spawn cap reached; dropping a Spawn from node {node_idx}"
                                                        );
                                                    }
                                                }
                                                // Already failing → drain to exit; don't
                                                // queue a delta the drain would just drop.
                                                Control::Spawn(_) => {}
                                            }
                                        }
                                    }
                                    // Lifecycle echoes + step-gap markers: ignored
                                    // by the watcher (the writer owns those).
                                    Ok(_) => {}
                                    // Dropped step spam under load is fine: divergence
                                    // PERSISTS (a NaN loss stays NaN), so a kill signal
                                    // dropped on lag re-arrives on the very next step —
                                    // it is not a single-shot edge (see KillOnNaN docs).
                                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                                    // The hub Sender lives in `env` for the whole
                                    // run, so Closed cannot occur before the drain
                                    // below; treat defensively as "stop watching".
                                    Err(broadcast::error::RecvError::Closed) => {
                                        break join.join_next().await;
                                    }
                                }
                                // Loop back to keep awaiting a completed node.
                            }
                        }
                    }
                }
            };
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
                    // Token no longer needed once the node is done (#4).
                    node_tokens.remove(&outcome.node_id);
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
                Err(NodeFailure::Killed { node_id }) => {
                    // #4 KILL-on-NaN: an INTENTIONAL branch prune, NOT a plan
                    // failure — other branches keep running. The killed node's
                    // GPU/memory permits already dropped when run_node returned.
                    node_tokens.remove(&node_id);
                    // Free its single-flight key. Same-key deferred waiters are
                    // duplicate-COMPUTATION siblings (not descendants — a
                    // descendant folds this output into a DIFFERENT key); the
                    // kill left no cacheable output, so release them to run on
                    // their own (the policy will kill them too if they diverge).
                    if let Some(k) = node_key_of.remove(&node_id) {
                        inflight_keys.remove(&k);
                        if let Some(waiters) = deferred.remove(&k) {
                            for w in waiters {
                                if !pruned.contains(&w) {
                                    ready.insert(w);
                                }
                            }
                        }
                    }
                    // Prune the killed node + every node reachable from it
                    // (their input can never materialize). DFS via a Vec stack;
                    // visitation order is irrelevant for a reachability prune.
                    // A pruned node already in `ready` is removed here; one
                    // re-added later by a completing OTHER parent is skipped at
                    // spawn (the `pruned` check).
                    let mut stack = vec![node_id];
                    while let Some(d) = stack.pop() {
                        if pruned.insert(d) {
                            ready.remove(&d);
                            if let Some(ss) = succs.get(&d) {
                                stack.extend(ss.iter().copied());
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

        // On the success path every node is either completed OR pruned by a
        // KILL-on-NaN (#4); a real failure returns above via `first_error`, so
        // a Cancelled node can't reach here uncounted. (`completed` counts Ok
        // outcomes; `pruned` holds killed nodes + descendants — disjoint sets.)
        debug_assert_eq!(
            completed + pruned.len(),
            order.len(),
            "parallel executor must account for every node (completed + pruned) on success"
        );
        // Release builds don't run the debug_assert — surface an accounting
        // mismatch (a real bug) in the log rather than silently returning an
        // incomplete result. Not fatal: a `None` final output from a pruned
        // terminal node is a LEGITIMATE outcome, so we never panic here.
        if completed + pruned.len() != order.len() {
            tracing::error!(
                "executor node accounting mismatch: completed={} pruned={} total={} \
                 (final output may be incomplete)",
                completed,
                pruned.len(),
                order.len()
            );
        }
        // If the terminal node was pruned, there is no final output — a killed
        // branch legitimately changed the graph (the caller sees the missing
        // output + the StageFailed events in status.jsonl).
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
        assert_eq!(h1, h2, "B.1(a): recorded output_hash stable across abs paths");
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
        let h1 = MemHog { peak: peak.clone(), live: live.clone() };
        let h2 = MemHog { peak: peak.clone(), live: live.clone() };
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
        assert_eq!(cache_entry_count(&job_dir), 1, "only the successful attempt is cached");
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
                std::fs::write(dir.join("state.json"), serde_json::to_string(&state).unwrap())
                    .unwrap();
                return Err(StageError::OutOfMemory { detail: "transient #1".into() });
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
        assert_eq!(RESUMABLE_ATTEMPTS.load(Ordering::SeqCst), 2, "ran twice (fail then resume)");
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
                assert!(matches!(source, StageError::Timeout { .. }), "got {source:?}");
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
    async fn kill_on_nan_prunes_branch_and_frees_gpu() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        DIVERGER_RAN.store(0, Ordering::SeqCst);
        INC_RUN_COUNT.store(0, Ordering::SeqCst);
        let td = tempfile::tempdir().unwrap();
        let job_dir = td.path().to_path_buf();
        let ctx = ExecCtx::new(job_dir.clone())
            .with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));
        // Clone the GPU semaphore Arc so we can assert the permit is returned
        // after the killed node drops it.
        let gpu = ctx.resources[&Resource::Gpu].clone();

        // MakeOne(Cpu) → Diverger(Gpu, diverges) → Increment(Cpu, downstream).
        // The kill must prune Increment (its input never materializes).
        let plan = Plan::<(), LamuTrainerBackend>::new("kill", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .then(Diverger, EmptyArgs)
            .then(Increment, EmptyArgs)
            .finish()
            .into_compiled();
        let fut = ParallelExecutor::execute(plan, ctx);
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
            .await
            .expect("KILL-on-NaN must fire — a parked Diverger would otherwise hang")
            .expect("a killed branch is NOT a plan failure → execute returns Ok");

        assert_eq!(DIVERGER_RAN.load(Ordering::SeqCst), 1, "diverger ran once");
        assert_eq!(
            INC_RUN_COUNT.load(Ordering::SeqCst),
            0,
            "downstream Increment must be PRUNED (never scheduled)"
        );
        assert!(
            result.final_output.is_none(),
            "terminal node was pruned → no final output"
        );
        assert_eq!(
            gpu.available_permits(),
            1,
            "the killed node's GPU permit must be freed"
        );
        // FW-2: a killed node is NOT promoted → no `<idx>-diverger` stage dir
        // and no leftover tmp.
        let stages = job_dir.join("stages");
        if stages.is_dir() {
            for entry in std::fs::read_dir(&stages).unwrap() {
                let name = entry.unwrap().file_name().to_string_lossy().into_owned();
                assert!(
                    !name.contains("diverger"),
                    "killed node left a stage dir: {name}"
                );
            }
        }
    }

    #[tokio::test]
    async fn kill_on_nan_lets_sibling_branch_finish() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        DIVERGER_RAN.store(0, Ordering::SeqCst);
        INC_RUN_COUNT.store(0, Ordering::SeqCst);
        let (_td, base) = fresh_ctx();
        let ctx = base.with_control(std::sync::Arc::new(crate::framework::control::KillOnNaN));

        // MakeOne → fork(Diverger[Gpu], Increment[Cpu]) → merge(SumTwo).
        // Diverger is killed; the Increment SIBLING must still complete; the
        // merge (a descendant of Diverger) is pruned.
        let plan = Plan::<(), LamuTrainerBackend>::new("kill_fork", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .fork(Diverger, EmptyArgs, Increment, EmptyArgs)
            .merge(SumTwo, EmptyArgs)
            .finish()
            .into_compiled();
        let fut = ParallelExecutor::execute(plan, ctx);
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
            .await
            .expect("sibling must finish + kill must fire")
            .expect("a killed branch is NOT a plan failure → Ok");

        assert_eq!(
            INC_RUN_COUNT.load(Ordering::SeqCst),
            1,
            "the sibling Increment branch must run to completion despite the kill"
        );
        assert!(
            result.final_output.is_none(),
            "the merge (descendant of the killed node) is pruned → no final output"
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
            Control::Spawn(crate::framework::control::SpawnDelta {
                subplan: spawn_subplan(true),
                label: Some("child".into()),
            })
        }
    }

    /// Spawns a single-node sub-plan on EVERY step — drives the bounded
    /// termination test (spawned nodes emit no steps, so it converges).
    struct SpawnEveryStep;
    impl crate::framework::control::ControlPolicy for SpawnEveryStep {
        fn on_step(&self, _m: &StepMetrics) -> Control {
            Control::Spawn(crate::framework::control::SpawnDelta {
                subplan: spawn_subplan(false),
                label: None,
            })
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

        assert_eq!(SPAWN_MARKER_RAN.load(Ordering::SeqCst), 1, "injected root ran");
        assert_eq!(SPAWN_CHILD_RAN.load(Ordering::SeqCst), 1, "injected child ran");
        // 2 base nodes + 2 spawned = 4 accounted (the in-test debug_assert in
        // execute() would have panicked on an accounting imbalance).
        assert_eq!(result.n_stages, 4, "order grew to include the spawned nodes");
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
        assert!(SPAWN_MARKER_RAN.load(Ordering::SeqCst) >= 1, "at least one spawn ran");
        // Bounded: 2 base nodes + at most one injected per observed step (≤ 4).
        // The in-test debug_assert in execute() already proved completed+pruned
        // balanced the (grown) order, so no node leaked.
        assert!(
            (3..=6).contains(&result.n_stages),
            "spawn count bounded, no runaway: n_stages={}",
            result.n_stages
        );
    }
}
