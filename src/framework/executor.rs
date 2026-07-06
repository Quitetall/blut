// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
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

use futures::FutureExt;
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::config::launcher::JobState;
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

/// Handle to a dispatched remote task. The executor polls this to
/// determine when the task completes.
pub trait DispatchHandle: Send + Sync {
    /// Poll the remote task. Returns `Some(JobState)` when terminal,
    /// `None` if still running.
    fn poll(&self) -> Result<Option<JobState>, crate::error::TrainError>;
    /// Cancel the remote task.
    fn cancel(&self) -> Result<(), crate::error::TrainError>;
}

/// Trait for submitting tasks to a remote compute network. The P2P
/// coordinator implements this; the executor calls it when a stage
/// is dispatchable.
/// Parameters for dispatching a stage to a remote peer.
pub struct DispatchRequest<'a> {
    pub stage_name: &'a str,
    pub stage_schema: u32,
    pub input_hash: ContentHash,
    pub args_hash: ContentHash,
    pub args: &'a serde_json::Value,
    pub expected_output_hash: ContentHash,
    pub resource_request: ResourceRequest,
    pub data_class: u8, // 0=Public, 1=Internal, 2=Restricted
}

pub trait DispatchSubmitter: Send + Sync {
    /// Submit a stage for remote execution. Returns a handle for
    /// tracking the task's lifecycle.
    fn submit(
        &self,
        request: DispatchRequest<'_>,
    ) -> Result<Box<dyn DispatchHandle>, crate::error::TrainError>;
}

/// Resource requirements for a dispatched task. Mirrors
/// `p2p::task::ResourceRequest` without the p2p dependency.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResourceRequest {
    pub cpu_cores: u32,
    pub memory_gib: u32,
    pub gpu: bool,
    pub gpu_vram_gib: Option<u32>,
}

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
    /// Never-OOM Phase 3: was the per-sample disk cache warmed upstream? Threaded
    /// into every `StageContext` so a train stage bills the warm (lower)
    /// per-worker footprint + the `|w` calibration key. Set by the CLI from the
    /// recipe's `warm_fb_cache` arg (the SAME source the admission gate reads),
    /// so RECORD and RESOLVE never disagree. Default false (cold). NOT a stage
    /// Arg — warm doesn't change the trained output, so it stays out of the
    /// checkpoint cache key.
    pub fb_warm: bool,
    /// Auto-tuned decode worker count (ADR 0071 A2), cached at admission so the
    /// cookbook train stage (RECORD) launches the SAME count the cli sized (RESOLVE)
    /// — parity + never-OOM. `None` ⇒ the conservative cap (unchanged behaviour).
    pub admitted_workers: Option<u32>,
    /// Auto-tuned batch size, extending ADR 0071's fit-and-saturate auto-tune
    /// to a second knob (E2). Resolved from the SAME admission snapshot as
    /// `admitted_workers` (batch is searched against the residual budget
    /// AFTER workers is fixed — see `batch_size_to_fit`'s doc comment for why
    /// this reaches the same result a joint search would). `None` ⇒ the
    /// recipe's requested batch, unchanged (no auto-tune ran).
    pub admitted_batch_size: Option<u32>,
    /// Phase-G scheduler: the GPU DEVICE index this whole job is pinned to,
    /// or `None` for the box default. Threaded into every `StageContext` so a
    /// launcher-aware backend exports `CUDA_VISIBLE_DEVICES=<idx>` for the
    /// trainer subprocess — so `capacity` partition cells run one-per-device
    /// concurrently. Set by the parallel-backfill scheduler.
    pub device_index: Option<usize>,
    /// Force-recompute (INC D / S4). `false` (default) = the executor honours
    /// the stage cache: a warm entry skips the run. `true` = the cache READ is
    /// bypassed, so every stage EXECUTES even when a cached entry exists; the
    /// fresh result is STILL written to the cache (later runs hit again). This
    /// is the A/B "force recompute" semantic — set by the CLI `--no-cache` /
    /// `--force` flag. Default false ⇒ byte-identical to the pre-INC-D path.
    pub bypass_cache: bool,
    /// P2P dispatch: policy that decides which stages are dispatchable.
    /// When set alongside `dispatcher`, the parallel executor offloads
    /// dispatchable DAG nodes to remote peers.
    #[cfg(feature = "p2p")]
    pub dispatch_policy: Option<Arc<dyn crate::p2p::dispatch::DispatchPolicy>>,
    /// P2P dispatch: submits tasks to the remote compute network.
    #[cfg(feature = "p2p")]
    pub dispatcher: Option<Arc<dyn DispatchSubmitter>>,
    /// DAG optimizer. When set, the executor runs the optimizer pass
    /// on the plan before execution (dead code elimination, critical
    /// path scheduling, cache-aware ordering). Default: enabled.
    pub dag_optimizer: Option<crate::framework::dag_opt::DagOptimizer>,
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
            admitted_workers: None,
            admitted_batch_size: None,
            device_index: None,
            bypass_cache: false,
            #[cfg(feature = "p2p")]
            dispatch_policy: None,
            #[cfg(feature = "p2p")]
            dispatcher: None,
            dag_optimizer: Some(crate::framework::dag_opt::DagOptimizer::new()),
        }
    }

    /// Place stages on `target` (#3). Default `Local`.
    pub fn with_launch_target(mut self, target: crate::config::launcher::LaunchTarget) -> Self {
        self.launch_target = target;
        self
    }

    /// Pin the job to GPU `device_index` (Phase-G scheduler). Default `None`
    /// (box default device).
    pub fn with_device_index(mut self, device_index: Option<usize>) -> Self {
        self.device_index = device_index;
        self
    }

    /// Mark the per-sample cache as warmed upstream (Phase 3). Threaded into every
    /// `StageContext.fb_warm` so a train stage bills the warm footprint.
    pub fn with_fb_warm(mut self, warm: bool) -> Self {
        self.fb_warm = warm;
        self
    }
    /// Set the auto-tuned decode worker count (ADR 0071 A2). Threaded into every
    /// `StageContext.admitted_workers` so the cookbook train stage launches it.
    pub fn with_admitted_workers(mut self, workers: u32) -> Self {
        self.admitted_workers = Some(workers);
        self
    }

    /// Set the auto-tuned batch size (E2, extends ADR 0071's fit-and-saturate
    /// to a second knob). Threaded into every `StageContext.admitted_batch_size`
    /// so the cookbook train stage launches it.
    pub fn with_admitted_batch_size(mut self, batch_size: u32) -> Self {
        self.admitted_batch_size = Some(batch_size);
        self
    }

    /// Set the P2P dispatch policy and submitter. When both are set,
    /// the parallel executor offloads dispatchable stages to peers.
    #[cfg(feature = "p2p")]
    pub fn with_dispatch(
        mut self,
        policy: Arc<dyn crate::p2p::dispatch::DispatchPolicy>,
        submitter: Arc<dyn DispatchSubmitter>,
    ) -> Self {
        self.dispatch_policy = Some(policy);
        self.dispatcher = Some(submitter);
        self
    }

    /// Force-recompute (INC D / S4): bypass the stage cache READ so every stage
    /// runs even with a warm entry, while STILL writing the fresh result to the
    /// cache. `false` (default) = honour the cache (skip on hit). Wired from the
    /// CLI `--no-cache` / `--force` flag.
    pub fn with_bypass_cache(mut self, bypass: bool) -> Self {
        self.bypass_cache = bypass;
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

    pub fn with_retry_hook(mut self, hook: crate::framework::retry::RetryHook) -> Self {
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
    /// Advisory stages that FAILED (ADR 0071): each warns + prunes its
    /// descendants but does not fail the plan. NON-EMPTY ⇒ the run completed
    /// "with warnings" — the machine-parseable signal that distinguishes this
    /// from a clean success (both exit 0). Tooling that acts on `final_output`
    /// (e.g. a promoter) MUST check this is empty first.
    pub warnings: Vec<StageWarning>,
}

/// One advisory stage that tripped (ADR 0071).
#[derive(Debug, Clone)]
pub struct StageWarning {
    /// Topo index of the advisory stage.
    pub idx: u32,
    pub stage: String,
    /// The failure the advisory stage produced (downgraded from fatal).
    pub reason: String,
    /// Who's at fault for the advisory failure — engine, cookbook glue, or
    /// external (ADR 0072). `None` when the underlying `StageError` doesn't
    /// carry a `StageFailure` to extract it from (the common case today;
    /// cookbooks don't yet attach structured failures to advisory stages).
    pub origin: Option<crate::framework::error_domain::FaultOrigin>,
}

/// Whether advisory stages are forced FATAL for this run (ADR 0071 strict mode),
/// via `BLUT_STRICT_ADVISORY=1` — for CI that wants the old fail-hard behaviour.
fn strict_advisory() -> bool {
    std::env::var("BLUT_STRICT_ADVISORY")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
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
    /// Total `Resource::Gpu` permits (== the box's GPU pool / device count). A
    /// DDP stage's `gpu_permits` is clamped to this so it never asks for more
    /// GPUs than exist.
    gpu_pool: usize,
    memory: Arc<tokio::sync::Semaphore>,
    memory_budget_gib: u32,
    launch_target: crate::config::launcher::LaunchTarget,
    device_index: Option<usize>,
    fb_warm: bool,
    admitted_workers: Option<u32>,
    admitted_batch_size: Option<u32>,
    /// Force-recompute (INC D / S4). When true, `run_node` skips the cache READ
    /// so the stage always runs; the fresh result is still cached.
    bypass_cache: bool,
    recipe_name: String,
    on_retry: Option<crate::framework::retry::RetryHook>,
    /// Divergence registry (S1 / ADR 0044 P7). Node id → the offending step's
    /// divergence detail, populated by the coordinator at the KILL site (a
    /// `Control::KillBranch` or a `Stage::divergence_check` true) BEFORE it
    /// cancels the node's token. `run_node` reads it under `task.node_id` to
    /// skip the stage on a targeted KILL (vs plan-wide cancel).
    /// tell a DIVERGENCE kill (→ retryable `StageError::Diverged`, auto-resume)
    /// apart from a plain targeted kill / plan cancel (→ unchanged Killed /
    /// Cancelled). The `Mutex` is held only for a tiny insert/get/remove — NEVER
    /// across an `.await` — so it can't deadlock the coordinator seam. Empty when
    /// no control policy is set, so the non-control path never touches it.
    diverged: Arc<std::sync::Mutex<HashMap<NodeId, String>>>,
    /// P2P dispatch policy + submitter. When set, dispatchable stages
    /// are offloaded to peers instead of running locally.
    #[cfg(feature = "p2p")]
    dispatch_policy: Option<Arc<dyn crate::p2p::dispatch::DispatchPolicy>>,
    #[cfg(feature = "p2p")]
    dispatcher: Option<Arc<dyn DispatchSubmitter>>,
}

/// A REPLACEABLE per-node kill token (#4 / S1). Shared between the coordinator
/// (which fires it on a divergence/targeted kill) and `run_node` (which derives
/// each attempt's `stage_cancel` from it). A plain `CancellationToken` can never
/// be un-cancelled, so a DIVERGENCE kill — which must let the node RETRY — would
/// otherwise poison every later attempt. The slot lets `run_node` swap in a
/// fresh token (a child of the plan token) for the next attempt after a
/// divergence kill, while the coordinator still targets the CURRENT token for a
/// second divergence or a plan cancel. The `Mutex` is held only for a token
/// clone/swap — never across an `.await`. Without a divergence retry the slot
/// holds one token for the node's whole life → identical to the prior single
/// `CancellationToken`.
#[derive(Clone)]
struct KillSlot(Arc<std::sync::Mutex<CancellationToken>>);

impl KillSlot {
    fn new(tok: CancellationToken) -> Self {
        KillSlot(Arc::new(std::sync::Mutex::new(tok)))
    }
    /// The CURRENT live token (clone). Cheap; lock held only for the clone.
    fn current(&self) -> CancellationToken {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
    /// Did the current token fire?
    fn is_cancelled(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_cancelled()
    }
    /// Cancel the current token (coordinator's kill).
    fn cancel(&self) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).cancel();
    }
    /// Swap in a fresh token (run_node re-arm before a divergence retry), so the
    /// next attempt is not poisoned by the prior kill but is STILL killable by
    /// the coordinator. Rooted at the plan token so a plan cancel still reaches it.
    fn rearm(&self, plan: &CancellationToken) {
        *self.0.lock().unwrap_or_else(|p| p.into_inner()) = plan.child_token();
    }
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
    /// Per-node kill SLOT (#4 / S1). Holds a CHILD of the plan token, so a
    /// plan-wide cancel still propagates here, but the coordinator can ALSO fire
    /// it alone to kill THIS node's branch (KILL-on-NaN) without touching
    /// siblings. A divergence kill re-arms it with a fresh token so the node can
    /// retry. With no control policy it holds one token for the node's whole life
    /// → behaviour is identical to the pre-#4 single-token path.
    node_cancel: KillSlot,
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
    node_cancel: &KillSlot,
    plan_cancel: &CancellationToken,
) -> NodeFailure {
    if node_cancel.is_cancelled() && !plan_cancel.is_cancelled() {
        NodeFailure::Killed { node_id }
    } else {
        NodeFailure::Cancelled
    }
}

/// Coordinator-side DIVERGENCE kill (S1 / ADR 0044 P7). Records the offending
/// step's detail in the divergence registry under the emitting node's id
/// (`env.diverged`) BEFORE cancelling its token, so `run_node` — unwinding on
/// the cancel — reads the entry and routes the kill to a retryable
/// `StageError::Diverged` (auto-resume) instead of a permanent branch prune.
///
/// PER-ATTEMPT KILL LATCH (S1 race fix). A REAL diverging trainer emits MANY
/// `{"loss":"nan"}` `StageStep` lines before its `killpg` teardown reaps the
/// subprocess — those stale steps buffer in the coordinator's lossy `broadcast`
/// receiver and arrive AFTER `run_node` has already promoted the kill to
/// `Diverged`, re-armed the slot with a FRESH un-cancelled token, and started
/// attempt 2. A stale step would then find `tok.is_cancelled() == false` (the
/// fresh token), slip past the idempotency guard below, and CANCEL the fresh
/// token — spuriously killing attempt 2 before it even diverged on its own. The
/// `kill_flagged` latch closes that window: once this coordinator has issued a
/// divergence kill for `nid`, it refuses to re-issue one until the node's
/// `StageEvent::StageRetrying` (the new-attempt boundary) clears the latch.
/// Because the single broadcast preserves emission order and `run_node` joins
/// the stage's stdout reader BEFORE emitting `StageRetrying`, ALL of attempt-1's
/// stale NaN steps precede that `StageRetrying` in the stream — so they are all
/// consumed-and-skipped here before the latch clears.
///
/// Idempotent: a no-op if `nid` is already kill-flagged this attempt, or if the
/// token already fired (the `is_cancelled` guard is kept as defense-in-depth for
/// the single-step-per-attempt path). Holds the registry `Mutex` only for a tiny
/// insert — never across an `.await`.
fn record_divergence_and_kill(
    nid: Option<NodeId>,
    node_idx: u32,
    stage_name: &str,
    update: &serde_json::Value,
    node_tokens: &HashMap<NodeId, KillSlot>,
    kill_flagged: &mut HashSet<NodeId>,
    env: &NodeEnv,
) {
    let Some(nid) = nid else { return };
    // Already kill-flagged this attempt: a stale buffered step from the SAME
    // (still-in-flight) attempt must not re-fire the kill — and crucially must
    // not cancel a fresh token a divergence retry just re-armed. Cleared at the
    // node's `StageRetrying` (attempt boundary) or on node removal.
    if kill_flagged.contains(&nid) {
        return;
    }
    let Some(tok) = node_tokens.get(&nid) else {
        return;
    };
    if tok.is_cancelled() {
        return;
    }
    // Short, human-readable detail: stage name + the offending step payload (so
    // the surfaced `Diverged` error / status line names WHAT diverged). A step
    // payload can be large, and this string flows into the `Diverged` error +
    // status events + logs, so cap it (a truncated tail is still diagnostic).
    const MAX_DETAIL: usize = 240;
    let mut step = update.to_string();
    if step.len() > MAX_DETAIL {
        // Truncate on a char boundary so the `String` stays valid UTF-8.
        let mut cut = MAX_DETAIL;
        while !step.is_char_boundary(cut) {
            cut -= 1;
        }
        step.truncate(cut);
        step.push('…');
    }
    let detail = format!("{stage_name}: divergence on step {step}");
    {
        let mut reg = env.diverged.lock().unwrap_or_else(|p| p.into_inner());
        reg.insert(nid, detail);
    }
    tracing::warn!(
        "divergence KILL on node {node_idx} ({stage_name}): retryable (auto-resume on retry)"
    );
    // Latch BEFORE cancel: any stale buffered NaN step for this node that the
    // coordinator processes before the node's `StageRetrying` boundary now
    // short-circuits at the `kill_flagged` guard above (so it can't re-fire on
    // the fresh, re-armed token of the next attempt).
    kill_flagged.insert(nid);
    tok.cancel();
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
    // INC D (S4): `bypass_cache` forces a recompute — skip the READ so the
    // stage always runs even with a warm entry. The fresh result is still
    // inserted into the cache below the run path, so later runs hit again.
    if !env.bypass_cache
        && let Some(hit) = env.cache.lookup(task.key)
    {
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
                // Wake on EITHER a plan cancel or a targeted kill (the slot's
                // current token is a child of the plan token, so it fires on
                // both). Bind the token so it outlives the `.cancelled()` future.
                let cur = task.node_cancel.current();
                tokio::select! {
                    _ = tokio::time::sleep(backoff) => {}
                    _ = cur.cancelled() => {
                        env.status.emit(StageEvent::StageFailed {
                            node_idx: idx,
                            stage_name: stage_name.clone(),
                            error: "cancelled during retry backoff".into(),
                            failure: None,
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
                failure: None,
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
        // Rooted at the kill slot's CURRENT token (#4 / S1), so a targeted /
        // divergence KILL fires it via that parent exactly as a plan cancel
        // would — the stage's own cancel handling is unchanged. After a
        // divergence retry the slot holds a FRESH (un-poisoned) token, so this
        // attempt starts clean yet stays killable by the coordinator.
        let stage_cancel = task.node_cancel.current().child_token();
        let mut stage_ctx = StageContext {
            job_dir: env.job_dir.clone(),
            stage_dir: tmp_stage_dir.clone(),
            node_idx: idx,
            status_tx: env.status.broadcast_sender(),
            cancel: stage_cancel.clone(),
            cache: env.cache.clone(),
            recipe_name: env.recipe_name.clone(),
            launch_target: env.launch_target,
            device_index: env.device_index,
            fb_warm: env.fb_warm,
            admitted_workers: env.admitted_workers,
            admitted_batch_size: env.admitted_batch_size,
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
            if let (Some(run_id), Some(token)) = (
                run_id,
                task.stage.resume_handle_erased(&stage_ctx, &task.args),
            ) {
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
                            failure: None,
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
        // A DDP stage holds `nproc` GPU permits (sized to the box's GPU pool);
        // every other resource holds 1. Clamp to the pool so a request for more
        // permits than exist can't park forever (the GPU pool == device count,
        // set by the CLI via `with_resource_limit`).
        let mut permits = Vec::new();
        for resource in sorted_resources {
            let Some(sem) = env.resources.get(&resource) else {
                continue;
            };
            let want = if resource == Resource::Gpu {
                let n = task.stage.gpu_permits(&task.args).max(1) as usize;
                // The pool size is the semaphore's total permits; clamp so a
                // DDP job asking for more GPUs than the box has runs on all of
                // them rather than deadlocking.
                let clamped = n.min(env.gpu_pool.max(1));
                if clamped < n {
                    // A DDP stage requested more GPUs than the box has — it will
                    // run DEGRADED (on `clamped` GPUs). Warn loudly so a user
                    // who thinks they're at full width isn't silently demoted.
                    tracing::warn!(
                        "stage '{}' requested {n} GPU permits but the pool has \
                         only {} — running on {clamped} (DDP width degraded)",
                        stage_name,
                        env.gpu_pool
                    );
                }
                clamped
            } else {
                1
            };
            let permit = match sem.clone().try_acquire_many_owned(want as u32) {
                Ok(p) => p,
                Err(_) => {
                    env.status.emit(StageEvent::StageBlocked {
                        node_idx: idx,
                        stage_name: stage_name.clone(),
                        resource,
                    });
                    match sem.clone().acquire_many_owned(want as u32).await {
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
        let mem_want = task
            .stage
            .memory_gib_for(&task.args)
            .min(env.memory_budget_gib);
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

        // ── GPU-saturation sampler (E2) ─────────────────────────────
        // While a GPU stage holds the device, sample nvidia-smi so the
        // run's utilization is MEASURED, not assumed (owner directive:
        // "GPU must run close to maximum, never wasted"). Gated on the
        // Gpu resource — a CPU-only stage (manifest build, …) has no GPU
        // window to sample. The samples land in status.jsonl and fold
        // into the gauges table at run-end; a sustained sub-floor streak
        // raises a `gpu_starved` sentinel. Best-effort: no nvidia-smi ⇒
        // no samples, run unaffected.
        let gpu_sampler = task.stage.resources().contains(&Resource::Gpu).then(|| {
            crate::framework::gpu_sampler::spawn_gpu_sampler(
                env.status.clone(),
                idx,
                stage_name.clone(),
            )
        });

        let run_fut = task
            .stage
            .run_erased(&stage_ctx, task.input.clone(), task.args.clone());
        let timed_fut = run_with_timeout(
            run_fut,
            &stage_cancel,
            task.timeout.soft,
            task.timeout.hard,
            stage_started,
        );
        // Panic-safe stage run. `GpuSamplerHandle` has no `Drop` impl (a bare
        // drop only DETACHES its background nvidia-smi poller — see the NOTE
        // on `GpuSamplerHandle` in gpu_sampler.rs — it keeps sampling until
        // process exit), so a panic unwinding straight through this scope
        // used to skip the `h.stop().await` below entirely and leak the
        // sampler task forever. `catch_unwind` runs the SAME teardown on a
        // caught panic, then `resume_unwind`s unchanged — the coordinator's
        // panic handling (`JoinError` → `PlanError::Other("node task
        // panicked...")`, see the `join.join_next()` match) is untouched;
        // only the sampler cleanup is now unwind-safe. `AssertUnwindSafe` is
        // sound here: `timed_fut` is dropped either way immediately after
        // this point, so no unwind-unsafe state is ever observed again.
        let run_result = match std::panic::AssertUnwindSafe(timed_fut).catch_unwind().await {
            Ok(r) => r,
            Err(panic_payload) => {
                if let Some(h) = gpu_sampler {
                    h.stop().await;
                }
                // Match every other error exit from this attempt (see the
                // sibling `let _ = std::fs::remove_dir_all(&tmp_stage_dir)`
                // calls above/below): a caught panic must not skip cleanup
                // of this attempt's tmp dir either, or it lingers on disk
                // until process exit.
                let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                drop(permits);
                drop(stage_ctx);
                std::panic::resume_unwind(panic_payload);
            }
        };

        // The run window is over — stop sampling before releasing the GPU
        // permit (any later device activity isn't this stage's). Runs on
        // every exit path from this attempt (the match above already
        // stopped it on the panic exit; this is the normal-return path).
        if let Some(h) = gpu_sampler {
            h.stop().await;
        }
        // Permits drop here, releasing the resource for queued stages
        // (including during a backoff before the next attempt).
        drop(permits);
        drop(stage_ctx);

        // ── Divergence (S1 / P7) ────────────────────────────────────
        // Was THIS node's token fired by a DIVERGENCE kill (a control
        // `KillBranch` or a `Stage::divergence_check` true), as opposed to a
        // plain targeted kill / plan cancel? The coordinator recorded the
        // offending step's detail in the registry under our node id BEFORE it
        // cancelled the token, so the entry is visible by the time the cancel
        // unwinds the run here. CONSUME it (take) so a later attempt's own
        // token state starts clean — a repeated divergence re-populates the
        // registry on the next kill. Lock held only for a tiny remove.
        let diverged_detail: Option<String> = {
            let mut reg = env.diverged.lock().unwrap_or_else(|p| p.into_inner());
            reg.remove(&task.node_id)
        };

        // Resolve this attempt to either `break` (success) or a single `err` to
        // classify. A DIVERGENCE cancel can surface two ways: the stage returns
        // Ok but its cancel token fired mid-run (it observed the kill and bailed
        // cleanly), OR the stage returns an error. In BOTH cases a registry hit
        // promotes it to the typed `StageError::Diverged` so it routes through
        // the retryable + auto-resume path below; otherwise the prior behaviour
        // is byte-identical (Ok-cancel → cancel_failure, Err → classify `e`).
        let err: StageError = match run_result {
            Ok(o) => {
                debug_assert_eq!(
                    o.kind,
                    task.stage.output_kind(),
                    "stage '{stage_name}' produced kind '{}' but declares output_kind '{}'",
                    o.kind,
                    task.stage.output_kind()
                );
                // A cancel observed during the run must NOT be promoted /
                // cached — discard. Check the STAGE token: it fires both on a
                // plan cancel (child inherits the parent) AND on a stage's own
                // cooperative cancel.
                if stage_cancel.is_cancelled() {
                    let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                    match diverged_detail {
                        // A DIVERGENCE cancel: synthesize a retryable `Diverged`
                        // and fall into the retry/terminal block (auto-resume on
                        // retry) instead of a silent Killed prune.
                        Some(detail) => StageError::Diverged { detail },
                        // A plain plan cancel / targeted kill: unchanged.
                        None => {
                            env.status.emit(StageEvent::StageFailed {
                                node_idx: idx,
                                stage_name,
                                error: "cancelled during stage".into(),
                                failure: None,
                            });
                            return Err(cancel_failure(
                                task.node_id,
                                &task.node_cancel,
                                &env.cancel,
                            ));
                        }
                    }
                } else {
                    // StageEnd reports the SUCCESSFUL attempt's wall time;
                    // failed attempts + backoff are visible as StageRetrying
                    // events, not folded into this duration.
                    break (o, stage_started.elapsed());
                }
            }
            Err(e) => {
                let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                // A registry hit promotes the error to `Diverged` regardless of
                // whether the node token has fired yet: in the narrow window
                // where the coordinator inserted the registry entry + cancelled
                // but the stage errored *just* before observing the cancel,
                // `diverged_detail` is `Some` while `token_fired` is still false.
                // Promoting anyway is CORRECT — a divergence should retry — and
                // the gate below admits it via the `diverged` disjunct.
                match diverged_detail {
                    Some(detail) => StageError::Diverged { detail },
                    None => e,
                }
            }
        };

        // `diverged` = this attempt's kill was a DIVERGENCE kill (the
        // synthesized `Diverged`). The node token fired on EITHER a plan cancel
        // or a targeted kill — a plain killed node must never retry (it would
        // re-run the doomed work), so retry gates on the node token; BUT a
        // divergence kill IS retryable (S1): the next attempt auto-resumes from
        // the last good checkpoint (the `attempt > 1` block above) and may
        // recover with a fresh RNG / lower effective LR.
        let diverged = matches!(err, StageError::Diverged { .. });
        let token_fired = task.node_cancel.is_cancelled();
        let retry = attempt < task.retry.max_attempts
            && (!token_fired || diverged)
            && crate::framework::retry::is_retryable(&err, task.retry.retry_on);
        if retry {
            // OOM-escalation / observability hook (broker wiring).
            if let Some(hook) = &env.on_retry {
                hook(&crate::framework::retry::RetryEvent {
                    stage_name: stage_name.clone(),
                    recipe_name: env.recipe_name.clone(),
                    attempt,
                    max_attempts: task.retry.max_attempts,
                    was_oom: matches!(err, StageError::OutOfMemory { .. }),
                    error: format!("{err}"),
                });
            }
            let next_backoff = task.retry.backoff_before(attempt + 1);
            // S1 ORDERING INVARIANT (load-bearing for the coordinator's
            // `kill_flagged` latch). This `StageRetrying` is the divergence latch's
            // CLEAR signal: the coordinator re-enables divergence kills for this
            // node only when it consumes this event. That is sound ONLY because
            // every `StageStep` this attempt emitted has ALREADY been flushed into
            // the broadcast (in order) by the time we reach here — so all of this
            // attempt's stale NaN steps precede this `StageRetrying` in the single
            // broadcast stream and are consumed-and-skipped (latch set) before the
            // latch clears. That holds because a backend emits its steps INLINE in
            // its `run_erased` future (the lamu backend JOINS its spawned stdout
            // reader via `stdout_reader.await` before `run` returns), so no step
            // can arrive after this point. A future backend that emits steps from a
            // DETACHED task outliving its run future would break this invariant —
            // it must instead join that task before returning (or the coordinator
            // latch must move to attempt-stamping).
            env.status.emit(StageEvent::StageRetrying {
                node_idx: idx,
                stage_name: stage_name.clone(),
                attempt,
                max_attempts: task.retry.max_attempts,
                error: format!("{err}"),
                backoff_ms: next_backoff.as_millis() as u64,
            });
            // S1: a divergence retry's kill already FIRED the slot's token; swap
            // in a fresh (un-poisoned) token rooted at the plan so the next
            // attempt starts clean yet the coordinator can still kill it on a
            // repeated divergence (or a plan cancel). A non-divergence retry left
            // the token un-fired → no re-arm needed.
            // ORDERING: `rearm` MUST precede `continue` — the next iteration
            // derives `stage_cancel` from `task.node_cancel.current()`, so it has
            // to see the fresh token. Do not reorder these two statements.
            if diverged {
                task.node_cancel.rearm(&env.cancel);
            }
            continue;
        }
        // Terminal failure.
        env.status.emit(StageEvent::StageFailed {
            node_idx: idx,
            stage_name: stage_name.clone(),
            error: format!("{err}"),
            failure: crate::framework::error_domain::extract_failure_summary(&err),
        });
        // A diverged node that exhausted its retries is a REAL surfaced failure
        // (the run diverged and could not recover) — NOT a silent Killed prune.
        // This is the deliberate S1 behavior change: a divergence is now
        // diverged → bounded retry → then `StageError::Diverged` fail.
        // ORDERING: this `diverged` check MUST come BEFORE the `cancelled` check
        // below — a diverged node's token IS fired (`token_fired` is true), so
        // reordering would misclassify an exhausted divergence as a `Killed`
        // prune instead of the intended surfaced `Diverged` failure.
        if diverged {
            return Err(NodeFailure::Stage {
                idx,
                stage: stage_name,
                source: err,
            });
        }
        // A token-fired, NON-diverged node keeps the EXACT prior behaviour:
        // targeted kill → Killed (prune); plan cancel or a bare Cancelled error
        // → Cancelled (fail-fast).
        let cancelled = token_fired || matches!(err, StageError::Cancelled);
        if cancelled {
            return Err(cancel_failure(task.node_id, &task.node_cancel, &env.cancel));
        }
        return Err(NodeFailure::Stage {
            idx,
            stage: stage_name,
            source: err,
        });
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
            failure: None,
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
    if let Err(e) = metadata.write_to(&final_stage_dir.join("output.metadata.json")) {
        tracing::warn!(
            "executor: sidecar write for stage '{stage_name}' failed: {e}; lineage tooling will not see this artifact"
        );
    }

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
/// The code identity that keys the cache for a node (S4 / P9): the build-time
/// git hash of THIS binary (catches a committed engine/cookbook change) folded
/// with the stage's own `code_fingerprint` (the script content hash — catches an
/// UNCOMMITTED kernel edit the git hash misses). A pure stage with no external
/// code returns `None` and is keyed on the git hash alone.
fn node_code_sha(stage: &dyn StageDyn) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(env!("BLUT_GIT_HASH").as_bytes());
    if let Some(fp) = stage.code_fingerprint() {
        h.update([0u8]);
        h.update(&fp);
    }
    h.finalize().to_vec()
}

fn build_task(
    node: &crate::framework::plan::PlanNode,
    node_idx: u32,
    edges: &[crate::framework::plan::PlanEdge],
    outputs: &HashMap<NodeId, ErasedArtifact>,
    logical_outputs: &HashMap<NodeId, ContentHash>,
    node_cancel: KillSlot,
) -> Result<NodeTask, PlanError> {
    let preds = predecessors(edges, node.id);
    let input = gather_input(node.id, &preds, outputs)?;
    let input_hash = gather_input_hash(node.id, &preds, logical_outputs)?;
    let code_sha = node_code_sha(node.stage.as_ref());
    let key = CacheHandle::key_for_canon_bytes(
        node.stage.name(),
        node.stage.schema(),
        input_hash,
        &node.canon_args,
        &code_sha,
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

/// Pick the ready node with the LONGEST critical path to the terminal, so the
/// critical path is never starved behind a cheap side branch. Ties are broken
/// by ascending `NodeId` — which makes the no-optimizer / all-equal-priority
/// case byte-identical to the historical `ready.iter().next()` (smallest id),
/// since `ready` is a `BTreeSet`. Nodes absent from `hints` (runtime-injected
/// sub-plans, or no optimizer configured) default to `critical_path_len = 0`,
/// so they schedule after any positive-priority original node but are never
/// dropped — `cap` keeps draining them.
///
/// Determinism: iteration is driven by the sorted `ready` set and only does
/// point `get`s into the `HashMap` `hints`; the composite key
/// `(critical_path_len, Reverse(id))` is unique per node, so the argmax never
/// depends on `max_by_key`'s tie rule.
///
/// Cost: O(|ready|) per call (a linear argmax), vs the old `ready.iter().next()`
/// at O(log n). The `max_in_flight` cap bounds calls per loop pass and ML
/// training DAGs keep the ready set in the single-to-low-hundreds range, so the
/// linear scan is negligible; revisit only if a workload makes `ready` very wide.
fn next_ready(
    ready: &BTreeSet<NodeId>,
    hints: &HashMap<NodeId, crate::framework::dag_opt::ScheduleHint>,
) -> Option<NodeId> {
    ready.iter().copied().max_by_key(|id| {
        let cp = hints.get(id).map(|h| h.critical_path_len).unwrap_or(0);
        (cp, std::cmp::Reverse(*id))
    })
}

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
        let g = PlanEdge {
            from: base + e.from,
            to: base + e.to,
        };
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
    // Snapshot the GPU pool size (total permits) BEFORE any node acquires, so a
    // DDP stage's gpu_permits clamps to the real device count. ORDERING: this
    // MUST stay before the `resources: ctx.resources` move below AND before any
    // node spawns — `available_permits()` reads the CURRENT free count, which
    // equals the total only while nothing is held (true here in prelude).
    let gpu_pool = ctx
        .resources
        .get(&Resource::Gpu)
        .map(|s| s.available_permits())
        .unwrap_or(1);
    let env = Arc::new(NodeEnv {
        job_dir: ctx.job_dir,
        cache: ctx.cache,
        status: ctx.status,
        cancel: ctx.cancel,
        resources: ctx.resources,
        gpu_pool,
        memory: ctx.memory,
        memory_budget_gib: ctx.memory_budget_gib,
        launch_target: ctx.launch_target,
        device_index: ctx.device_index,
        fb_warm: ctx.fb_warm,
        admitted_workers: ctx.admitted_workers,
        admitted_batch_size: ctx.admitted_batch_size,
        bypass_cache: ctx.bypass_cache,
        recipe_name: plan.name().to_string(),
        on_retry: ctx.on_retry,
        diverged: Arc::new(std::sync::Mutex::new(HashMap::new())),
        #[cfg(feature = "p2p")]
        dispatch_policy: ctx.dispatch_policy,
        #[cfg(feature = "p2p")]
        dispatcher: ctx.dispatcher,
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
        let mut warnings: Vec<StageWarning> = Vec::new();

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
                    failure: None,
                });
                finish_writer(env, writer_handle).await;
                return Err(PlanError::Cancelled);
            }

            let node = &view.nodes[*node_id as usize];
            // Per-node kill slot (#4). Sequential wires no control policy, so it
            // only ever fires via the plan token → identical to the prior
            // single-token behaviour (never re-armed).
            let node_cancel = KillSlot::new(env.cancel.child_token());
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
                    // ADR 0071: an advisory stage's failure is a non-fatal warning,
                    // not a plan failure — record it and STOP (the remaining topo
                    // nodes are its descendants and can't run). `strict_advisory()`
                    // forces the old fail-hard behaviour for CI.
                    if let NodeFailure::Stage { idx, stage, source } = &f {
                        if node.stage.is_advisory() && !strict_advisory() {
                            tracing::warn!("advisory stage '{stage}' failed (non-fatal): {source}");
                            warnings.push(StageWarning {
                                idx: *idx,
                                stage: stage.clone(),
                                reason: source.to_string(),
                                // No StageFailure downcast happens on this path today
                                // (source is only stringified above) — nothing to
                                // thread through yet (ADR 0072 B-series wires real
                                // origins into cookbook StageFailures).
                                origin: None,
                            });
                            break;
                        }
                    }
                    finish_writer(env, writer_handle).await;
                    return Err(plan_error_of(f));
                }
            }
        }

        // If an advisory stage pruned the terminal node, surface the last COMPLETED
        // node in topo order (the upstream train ckpt in the canonical linear
        // train→gate plan) rather than None (ADR 0071).
        let final_output = if warnings.is_empty() {
            order.last().and_then(|id| outputs.remove(id))
        } else {
            order.iter().rev().find_map(|id| outputs.remove(id))
        };
        finish_writer(env, writer_handle).await;

        Ok(PlanResult {
            final_output,
            n_stages: order.len(),
            n_cache_hits: n_hits,
            n_cache_misses: n_misses,
            elapsed: started.elapsed(),
            warnings,
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

        // DAG optimization pass: dead code elimination, critical path
        // scheduling, cache-aware ordering. Runs before topo_sort. The
        // per-node `schedule_hints` drive ready-node priority in the spawn
        // loop (longest critical path first); with no optimizer the map is
        // empty and `next_ready` falls back to smallest-NodeId order, which
        // is byte-identical to the historical `ready.iter().next()`.
        let (plan, schedule_hints) = if let Some(ref optimizer) = ctx.dag_optimizer {
            optimizer.optimize(plan)
        } else {
            (plan, std::collections::HashMap::new())
        };

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
        let mut node_tokens: HashMap<NodeId, KillSlot> = HashMap::new();
        // S1 / P7: the coordinator does not keep the stage after spawn, but it
        // must consult the EMITTING node's `Stage::divergence_check` on each live
        // step. Hold an `Arc<dyn StageDyn>` clone per in-flight node here (the
        // stage is a zero-sized marker, so the clone is just a refcount bump),
        // keyed the same as `node_tokens`, and drop it alongside the token on
        // completion/kill so it never leaks across nodes. Empty without a policy.
        let mut node_stages: HashMap<NodeId, Arc<dyn StageDyn>> = HashMap::new();
        // S1 race fix: per-node kill latch. A node is inserted here when the
        // coordinator issues a divergence kill for it (in `record_divergence_and_kill`)
        // and removed at its `StageEvent::StageRetrying` boundary (the new attempt
        // started) or on node removal. While flagged, the coordinator refuses to
        // re-issue a divergence kill — so the burst of stale buffered NaN steps a
        // diverging trainer emits before its subprocess is reaped cannot re-fire the
        // kill on a fresh, re-armed token of the next attempt. Plain coordinator-local
        // state (no lock): the coordinator seam is single-threaded. Empty without a
        // control policy, so the non-control path never touches it.
        let mut kill_flagged: HashSet<NodeId> = HashSet::new();

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
        let in_flight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut n_hits = 0usize;
        let mut n_misses = 0usize;
        let mut first_error: Option<PlanError> = None;
        let mut completed = 0usize;
        let mut warnings: Vec<StageWarning> = Vec::new();

        // Pre-cancel: honour a token already fired before the first spawn.
        if env.cancel.is_cancelled() {
            env.status.emit(StageEvent::StageFailed {
                node_idx: 0,
                stage_name: "<cancelled>".into(),
                error: "plan cancelled before stage".into(),
                failure: None,
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
                while in_flight.load(std::sync::atomic::Ordering::Relaxed) < max_in_flight {
                    // Critical-path-first among ready nodes (longest path to the
                    // terminal wins); ties → smallest NodeId, byte-identical to the
                    // historical `ready.iter().next()` when priorities are equal.
                    let Some(node_id) = next_ready(&ready, &schedule_hints) else {
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
                    // Per-node kill slot: holds a child of the plan token,
                    // retained in `node_tokens` so the control watcher can fire
                    // it alone (and a divergence retry can re-arm it).
                    let node_cancel = KillSlot::new(env.cancel.child_token());
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
                                failure: None,
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
                    // S1: retain a stage handle for the live `divergence_check`
                    // (only when a control policy watches — otherwise the watcher
                    // never runs, so the clone would be dead weight). COUPLING:
                    // the sole `divergence_check` invocation lives inside the
                    // `control_rx` select arm, which only exists when
                    // `control.is_some()` — so this guard and that arm must stay
                    // in lockstep (a future non-control divergence path would
                    // need to populate `node_stages` unconditionally).
                    if control.is_some() {
                        node_stages.insert(node_id, task.stage.clone());
                    }

                    // P2P dispatch: if the stage is dispatchable and a
                    // dispatcher is available, offload to a peer instead
                    // of running locally.
                    #[cfg(feature = "p2p")]
                    if let (Some(policy), Some(dispatcher)) =
                        (env.dispatch_policy.as_ref(), env.dispatcher.as_ref())
                    {
                        if policy.is_dispatchable(task.stage.name()) {
                            let args_hash = ContentHash::of_bytes(&task.canon_args);
                            let stage_resources = task.stage.resources();
                            let has_gpu = stage_resources.contains(&Resource::Gpu);
                            let resource_request = ResourceRequest {
                                cpu_cores: 1, // TODO: derive from RESOURCES
                                memory_gib: task.stage.memory_gib(),
                                gpu: has_gpu,
                                gpu_vram_gib: None,
                            };
                            let data_class =
                                policy.classify_stage(task.stage.name(), &task.args) as u8;
                            let request = DispatchRequest {
                                stage_name: task.stage.name(),
                                stage_schema: task.stage.schema(),
                                input_hash: task.input_hash,
                                args_hash,
                                args: &task.args,
                                expected_output_hash: task.key,
                                resource_request,
                                data_class,
                            };
                            match dispatcher.submit(request) {
                                Ok(handle) => {
                                    tracing::info!(
                                        "Dispatched node {} ({}) to P2P peer",
                                        node_idx,
                                        task.stage.name()
                                    );
                                    let status = env.status.clone();
                                    let cache = env.cache.clone();
                                    let stage_name = task.stage.name().to_string();
                                    let stage = task.stage.clone();
                                    let deterministic = task.stage.deterministic();
                                    let schema = task.stage.schema();
                                    let key = task.key;
                                    let node_id = task.node_id;
                                    let input_hash = task.input_hash;
                                    let canon_args = task.canon_args.clone();
                                    // Route this dispatch's completion through the SAME
                                    // JoinSet the coordinator awaits below (`join.join_next()`)
                                    // instead of a detached `tokio::spawn` side-channel. A
                                    // detached task bumps `in_flight` but is invisible to
                                    // `join_next()`, so once every ready node is P2P-dispatched
                                    // the JoinSet goes empty and `join_next()` returns `None`
                                    // immediately — ending the coordinator loop while the
                                    // remote work is still running, and tripping the
                                    // `completed + pruned == order.len()` accounting check
                                    // below. Being a JoinSet member also means a
                                    // `JobState::Failed` now produces a real
                                    // `NodeFailure::Stage` that flows through the SAME
                                    // `first_error` / `env.cancel.cancel()` handling as a local
                                    // stage failure (the `Err(f)` arm a few hundred lines down) —
                                    // previously it only emitted a status event on a detached
                                    // side-channel and the plan could return `Ok` past an
                                    // explicitly failed remote stage.
                                    join.spawn(async move {
                                        let start = std::time::Instant::now();
                                        loop {
                                            match handle.poll() {
                                                Ok(Some(JobState::Succeeded)) => {
                                                    // `DispatchHandle::poll` carries no artifact
                                                    // payload (`JobState::Succeeded` is a unit
                                                    // variant), so the only route back to a real
                                                    // `ErasedArtifact` without widening that trait
                                                    // is the content-addressed cache: the P2P data
                                                    // plane is expected to have landed the peer's
                                                    // output bytes there under
                                                    // `expected_output_hash` (== `key`) by the time
                                                    // the job goes terminal. A miss here means the
                                                    // peer claimed success but never delivered the
                                                    // artifact — fail closed instead of returning a
                                                    // phantom `Ok` with no real output.
                                                    return match cache.lookup(key) {
                                                        Some(hit) => {
                                                            let logical = compute_logical_output_hash(
                                                                stage.as_ref(),
                                                                &hit.artifact,
                                                                deterministic,
                                                                &stage_name,
                                                                schema,
                                                                input_hash,
                                                                &canon_args,
                                                            );
                                                            // Match the local run_node path: `output_hash`
                                                            // must be a content hash of the ARTIFACT, not
                                                            // `key` (a hash of the job's inputs). Lineage
                                                            // tooling reads this field expecting content
                                                            // addressability regardless of whether the node
                                                            // ran locally or was P2P-dispatched.
                                                            let output_hash = stage
                                                                .output_content_hash(&hit.artifact)
                                                                .unwrap_or_else(|| {
                                                                    content_hash_from_erased(&hit.artifact)
                                                                });
                                                            status.emit(StageEvent::StageEnd {
                                                                node_idx,
                                                                stage_name: stage_name.clone(),
                                                                output_hash,
                                                                elapsed: start.elapsed(),
                                                            });
                                                            Ok(NodeOutcome {
                                                                node_id,
                                                                output: hit.artifact,
                                                                logical,
                                                                cache_hit: false,
                                                            })
                                                        }
                                                        None => {
                                                            let msg = format!(
                                                                "P2P dispatch reported success for node {node_idx} ({stage_name}) but no artifact was found in the cache for key {}",
                                                                key.to_hex()
                                                            );
                                                            status.emit(StageEvent::StageFailed {
                                                                node_idx,
                                                                stage_name: stage_name.clone(),
                                                                error: msg.clone(),
                                                                failure: None,
                                                            });
                                                            Err(NodeFailure::Stage {
                                                                idx: node_idx,
                                                                stage: stage_name,
                                                                source: StageError::Backend(anyhow::anyhow!(msg)),
                                                            })
                                                        }
                                                    };
                                                }
                                                Ok(Some(JobState::Failed(reason))) => {
                                                    status.emit(StageEvent::StageFailed {
                                                        node_idx,
                                                        stage_name: stage_name.clone(),
                                                        error: reason.clone(),
                                                        failure: None,
                                                    });
                                                    // Was: "Don't cancel the whole plan — just
                                                    // report the failure", with first_error/cancel
                                                    // never touched. Now: return a real Err so the
                                                    // coordinator's normal Err(f) handling (which
                                                    // sets first_error + cancels siblings) applies —
                                                    // an explicit remote-stage failure fails the plan.
                                                    return Err(NodeFailure::Stage {
                                                        idx: node_idx,
                                                        stage: stage_name,
                                                        source: StageError::Backend(anyhow::anyhow!(reason)),
                                                    });
                                                }
                                                Ok(Some(JobState::Cancelled)) => {
                                                    return Err(NodeFailure::Cancelled);
                                                }
                                                Ok(Some(JobState::Unknown(reason))) => {
                                                    let msg = format!(
                                                        "P2P dispatch for node {node_idx} ({stage_name}) ended in an unknown state: {reason}"
                                                    );
                                                    status.emit(StageEvent::StageFailed {
                                                        node_idx,
                                                        stage_name: stage_name.clone(),
                                                        error: msg.clone(),
                                                        failure: None,
                                                    });
                                                    return Err(NodeFailure::Stage {
                                                        idx: node_idx,
                                                        stage: stage_name,
                                                        source: StageError::Backend(anyhow::anyhow!(msg)),
                                                    });
                                                }
                                                Ok(Some(JobState::Running)) | Ok(None) => {
                                                    tokio::time::sleep(
                                                        std::time::Duration::from_millis(500),
                                                    ).await;
                                                }
                                                Err(e) => {
                                                    let msg = format!("{e}");
                                                    status.emit(StageEvent::StageFailed {
                                                        node_idx,
                                                        stage_name: stage_name.clone(),
                                                        error: msg.clone(),
                                                        failure: None,
                                                    });
                                                    return Err(NodeFailure::Stage {
                                                        idx: node_idx,
                                                        stage: stage_name,
                                                        source: StageError::Backend(anyhow::anyhow!(msg)),
                                                    });
                                                }
                                            }
                                        }
                                    });
                                    in_flight.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    continue; // skip local spawn
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        "P2P dispatch failed for node {} ({}), running locally: {e}",
                                        node_idx,
                                        task.stage.name()
                                    );
                                    // Fall through to local spawn.
                                }
                            }
                        }
                    }

                    let env_c = env.clone();
                    join.spawn(async move { run_node(task, env_c).await });
                    in_flight.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }

            if in_flight.load(std::sync::atomic::Ordering::Relaxed) == 0 {
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
                                            // Resolve the emitting node id once: topo idx → node id.
                                            let nid = order.get(node_idx as usize).copied();
                                            // S1 / P7: consult the EMITTING node's
                                            // `Stage::divergence_check` IN ADDITION to the policy.
                                            // A domain-aware threshold (loss > k·EMA, grad spike)
                                            // is a divergence too — same kill + retry path as a
                                            // `KillBranch`. This is the ONLY invocation site of
                                            // `divergence_check` (it was previously dead code).
                                            let stage_diverged = nid
                                                .and_then(|n| node_stages.get(&n))
                                                .map(|s| s.divergence_check(&update))
                                                .unwrap_or(false);
                                            match policy.on_step(&m) {
                                                // A stage-side divergence still kills even if the
                                                // policy said Continue (e.g. the default null policy
                                                // is never wired, but KillOnNaN's Continue + a
                                                // threshold override must still fire).
                                                Control::Continue if stage_diverged => {
                                                    record_divergence_and_kill(
                                                        nid, node_idx, &stage_name, &update,
                                                        &node_tokens, &mut kill_flagged, &env,
                                                    );
                                                }
                                                Control::Continue => {}
                                                // A `KillBranch` IS a divergence kill — KillOnNaN
                                                // is its only emitter (non-finite step metric); the
                                                // separate `Spawn` arm below is NOT a kill. Record
                                                // the divergence + cancel the token so `run_node`
                                                // retries it (auto-resume) instead of pruning.
                                                Control::KillBranch => {
                                                    record_divergence_and_kill(
                                                        nid, node_idx, &stage_name, &update,
                                                        &node_tokens, &mut kill_flagged, &env,
                                                    );
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
                                    // S1 race fix: a node's `StageRetrying` is the
                                    // ATTEMPT BOUNDARY — `run_node` emits it right
                                    // before it re-arms the kill slot and `continue`s
                                    // into the next attempt. Clear the kill latch so
                                    // the node is killable again ONLY once its NEW
                                    // attempt has actually begun. The single broadcast
                                    // preserves emission order and `run_node` joins the
                                    // stage's stdout reader before emitting this event,
                                    // so ALL of the prior attempt's stale NaN StageSteps
                                    // precede this in the stream and are already
                                    // consumed-and-skipped (the latch was set) — no
                                    // straggler step can clear-then-refire the kill.
                                    Ok(StageEvent::StageRetrying { node_idx, .. }) => {
                                        if let Some(&nid) = order.get(node_idx as usize) {
                                            kill_flagged.remove(&nid);
                                        }
                                    }
                                    // Lifecycle echoes + step-gap markers: ignored
                                    // by the watcher (the writer owns those).
                                    Ok(_) => {}
                                    // Dropped step spam under load is fine: divergence
                                    // PERSISTS (a NaN loss stays NaN), so a kill signal
                                    // dropped on lag re-arrives on the very next step —
                                    // it is not a single-shot edge (see KillOnNaN docs).
                                    //
                                    // But the divergence latch's CLEAR signal
                                    // (`StageRetrying`) also rides this lossy broadcast.
                                    // Under sustained backpressure that overran the ring, a
                                    // node's `StageRetrying` boundary could be evicted
                                    // before we reach it — wedging `kill_flagged` SET, which
                                    // would silently disable the divergence kill for that
                                    // node's NEXT attempt. A `Lagged` reliably co-occurs
                                    // with exactly that overrun, so clear the latch
                                    // defensively. This cannot resurrect the stale-refire
                                    // race: the dropped events were the OLDEST (the prior
                                    // attempt's stale NaN steps), so no straggler survives
                                    // the lag to spuriously re-kill a healthy attempt —
                                    // re-killing now requires a SUBSEQUENTLY-OBSERVED
                                    // non-finite step, which is by definition the current
                                    // attempt genuinely diverging.
                                    Err(broadcast::error::RecvError::Lagged(_)) => {
                                        kill_flagged.clear();
                                    }
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
            in_flight.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            let res = match joined {
                Some(Ok(r)) => r,
                Some(Err(join_err)) => {
                    // Task panicked. Record as the first error, cancel.
                    first_error
                        .get_or_insert(PlanError::Other(format!("node task panicked: {join_err}")));
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
                    // S1: drop the stage handle alongside (no leak across nodes).
                    node_stages.remove(&outcome.node_id);
                    // S1 race fix: clear any leftover kill latch. A
                    // divergence-then-recovered node was flagged on each diverging
                    // attempt and cleared at the following `StageRetrying`; the
                    // successful attempt set no flag, so this is usually a no-op —
                    // but the explicit remove keeps the set from leaking across
                    // nodes regardless of the per-attempt history.
                    kill_flagged.remove(&outcome.node_id);
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
                    node_stages.remove(&node_id);
                    // S1 race fix: clear the kill latch for the pruned node.
                    kill_flagged.remove(&node_id);
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
                    // record Cancelled if nothing else failed. (No node id to
                    // remove — the `Cancelled` variant carries none; the maps are
                    // coordinator-local and dropped when `execute` returns on the
                    // ensuing fail-fast drain, so no cross-node leak.)
                    if first_error.is_none() {
                        first_error = Some(PlanError::Cancelled);
                        env.cancel.cancel();
                    }
                }
                Err(f) => {
                    // S1 race fix: a `Stage` failure carries the topo idx, so map
                    // it back to the node id and clear its in-flight maps + kill
                    // latch. Capture whether the failing stage is ADVISORY (ADR
                    // 0071) BEFORE removing it from `node_stages`.
                    let mut advisory: Option<(u32, String, String, NodeId)> = None;
                    if let NodeFailure::Stage { idx, stage, source } = &f {
                        if let Some(&nid) = order.get(*idx as usize) {
                            let is_adv = node_stages
                                .get(&nid)
                                .map(|s| s.is_advisory())
                                .unwrap_or(false)
                                && !strict_advisory();
                            node_tokens.remove(&nid);
                            node_stages.remove(&nid);
                            kill_flagged.remove(&nid);
                            if is_adv {
                                advisory = Some((*idx, stage.clone(), source.to_string(), nid));
                            }
                        }
                    }
                    if let Some((idx, stage, reason, nid)) = advisory {
                        // Advisory failure: WARN, prune descendants exactly like a
                        // KILL (their input can't materialize), but do NOT fail the
                        // plan — other branches keep running.
                        tracing::warn!("advisory stage '{stage}' failed (non-fatal): {reason}");
                        // No StageFailure downcast happens on this path today
                        // (reason is only stringified above) — nothing to thread
                        // through yet (ADR 0072 B-series wires real origins into
                        // cookbook StageFailures).
                        warnings.push(StageWarning {
                            idx,
                            stage,
                            reason,
                            origin: None,
                        });
                        if let Some(k) = node_key_of.remove(&nid) {
                            inflight_keys.remove(&k);
                            if let Some(waiters) = deferred.remove(&k) {
                                for w in waiters {
                                    if !pruned.contains(&w) {
                                        ready.insert(w);
                                    }
                                }
                            }
                        }
                        let mut stack = vec![nid];
                        while let Some(d) = stack.pop() {
                            if pruned.insert(d) {
                                ready.remove(&d);
                                if let Some(ss) = succs.get(&d) {
                                    stack.extend(ss.iter().copied());
                                }
                            }
                        }
                    } else if first_error.is_none() {
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
        // output + the StageFailed events in status.jsonl). EXCEPT when an
        // advisory stage did the pruning (ADR 0071): surface the last completed
        // node in topo order (the upstream train ckpt for a linear train→gate plan)
        // so the operator gets it live.
        let final_output = if warnings.is_empty() {
            order.last().and_then(|id| outputs.remove(id))
        } else {
            order.iter().rev().find_map(|id| outputs.remove(id))
        };
        finish_writer(env, writer_handle).await;

        Ok(PlanResult {
            final_output,
            n_stages: order.len(),
            n_cache_hits: n_hits,
            n_cache_misses: n_misses,
            elapsed: started.elapsed(),
            warnings,
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
    // S4: fold `code_sha` into the NONDET synthesized fingerprint too — else a
    // kernel edit reuses the stale checkpoint through the DOWNSTREAM path (a
    // nondet stage's output hash is synthesized from this, not its bytes), even
    // after the direct cache key changes. Bumped v1→v2 to match the cache bust.
    use sha2::{Digest, Sha256};
    let code_sha = node_code_sha(stage);
    let mut h = Sha256::new();
    h.update(b"blut.nondet.v2");
    h.update([0u8]);
    h.update(stage_name.as_bytes());
    h.update([0u8]);
    h.update(schema.to_le_bytes());
    h.update(input_hash.0);
    h.update((code_sha.len() as u64).to_le_bytes());
    h.update(&code_sha);
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

// The test suite (~3100 lines of fixture stages + integration tests) lives in
// executor_tests.rs, mounted here as a child module so it keeps access to the
// executor's private items. Rationale for the file-level clippy allow is
// documented at the top of that file.
#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
