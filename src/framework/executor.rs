// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Plan executor.
//!
//! Two executors share one private per-stage core, `run_node`:
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

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use bincode::Options;
use futures::{FutureExt, StreamExt};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::config::launcher::JobState;
use crate::framework::artifact::{
    ArtifactMetadata, BranchDecision, ContentHash, ContentId, InvocationKey,
};
use crate::framework::artifact_store::{
    ArtifactRole, ArtifactStoreError, capture, unpersisted_content_id,
};
use crate::framework::cache::{CacheHandle, CacheHit};
use crate::framework::control::{Control, ControlPolicy, StepMetrics};
use crate::framework::error::{PlanError, StageError};
use crate::framework::plan::{CompiledPlan, NodeId};
use crate::framework::resource::Resource;
use crate::framework::stage::{ErasedArtifact, InProcessArtifact, StageContext, StageDyn};
use crate::framework::status::{
    DEFAULT_BROADCAST_CAPACITY, StageEvent, StatusHub, spawn_status_writer_checked,
};

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

static SPECULATION_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreparedExecutorMode {
    Sequential,
    Parallel,
}

/// Private proof that optimization and execution-only profile selection have
/// already run against this plan's final node-id space.
struct PreparedExecution {
    mode: PreparedExecutorMode,
    schedule_hints: HashMap<NodeId, crate::framework::dag_opt::ScheduleHint>,
}

/// Per-node launch facts selected by the trusted admission boundary. These are
/// execution-only and are never exposed as cookbook args or cache inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TrainingIoNodeAdmission {
    pub(crate) admitted_decode_workers: Option<u32>,
    pub(crate) admitted_batch_size: Option<u32>,
    pub(crate) cache_warm: bool,
    pub(crate) calibrated_base_floor_bytes: Option<u64>,
    /// Optional node-local ceiling captured by a launcher that holds a fixed
    /// whole-job reservation. Runtime HPO children use this to downgrade or
    /// refuse instead of selecting a profile larger than the initial tenant
    /// reservation. The global executor/live-snapshot ceiling still applies;
    /// selection uses the smaller value.
    pub(crate) selection_budget_bytes: Option<u64>,
}

/// Optional child-aware admission resolver. It is captured in the same
/// immutable select-once witness as the resource budget, then applied to both
/// post-DCE static nodes and runtime-injected PBT/TPE/map nodes. `None` is an
/// explicit choice to use the launch-wide defaults for an irrelevant stage;
/// a required node that cannot be calibrated must return `Err` and fail closed.
pub(crate) type TrainingIoNodeAdmissionResolver = Arc<
    dyn Fn(
            &str,
            &serde_json::Value,
            &serde_json::Value,
        ) -> Result<Option<TrainingIoNodeAdmission>, String>
        + Send
        + Sync
        + 'static,
>;

#[derive(Clone)]
struct TrainingIoResolver {
    default_hints: crate::framework::async_io::TrainingIoHints,
    budget_bytes: u64,
    force_inline_reason: Option<crate::framework::async_io::TrainingIoDowngradeReason>,
    whole_job_base_bytes: Option<u64>,
    node_admission: Option<TrainingIoNodeAdmissionResolver>,
}

#[derive(Clone)]
struct ResolvedTrainingIoNode {
    profile: crate::framework::async_io::TrainingIoProfile,
    hints: crate::framework::async_io::TrainingIoHints,
}

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
    pub invocation_key: InvocationKey,
    pub input_content_id: ContentId,
    pub args_hash: ContentHash,
    pub args: &'a serde_json::Value,
    /// Present only when the stage can derive its output identity without
    /// executing. `None` delegates identity production to the worker; success is
    /// accepted only after the receiver restores and recomputes the artifact.
    pub expected_content_id: Option<ContentId>,
    pub resource_request: ResourceRequest,
    pub data_class: u8, // 0=Public, 1=Internal, 2=Restricted
    /// Owning tenant. Dispatchers must refuse restricted tenants even if a
    /// cookbook accidentally classifies the individual stage as Public.
    pub tenant: &'a crate::tenant::Tenant,
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
    /// Owning tenant, threaded into every StageContext and remote-dispatch
    /// decision. Restricted tenants are node-local through M5.
    pub tenant: crate::tenant::Tenant,
    /// Status fan-out hub. Subscribe a live receiver via
    /// `ctx.status.subscribe()`; the executor emits through it.
    pub status: Arc<StatusHub>,
    /// The lossless lifecycle receiver, handed to the status writer by
    /// the executor's prelude. `None` once taken (after one execute).
    lifecycle_rx: Option<mpsc::UnboundedReceiver<StageEvent>>,
    pub cancel: CancellationToken,
    /// Per-resource semaphores. Stages acquire all permits in
    /// their `RESOURCES` slice before `run` is called. Default
    /// limits: Cpu=num_cpus, Network=4, Disk=2. GPU is NOT here — it is
    /// scheduled by [`gpu`](Self::gpu) (ADR 0087). Override via
    /// ExecCtx::with_resource_limit.
    pub resources: std::collections::HashMap<Resource, Arc<tokio::sync::Semaphore>>,
    /// GPU-aware scheduler (ADR 0087): per-device exclusive permits + VRAM-aware
    /// placement, admitted in series with the RAM broker. Default is a single
    /// device (byte-identical to the legacy `Semaphore::new(1)`); the CLI sizes
    /// it from the live inventory via `with_gpu_scheduler`.
    pub gpu: Arc<crate::broker::gpu::GpuScheduler>,
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
    /// `Stage::MEMORY_GIB` permits from this for its whole run. The CLI caps the
    /// budget to the whole-job RAM reservation (and never above the tenant
    /// ceiling), so concurrent stages cannot spend more declared RAM than the
    /// job acquired. This is admission accounting, not an OOM guarantee:
    /// declarations can be wrong and the default budget is effectively
    /// unlimited for direct library callers.
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
    /// Transitional LamQuant compatibility signal for a warmed per-sample
    /// cache. The generic CLI leaves it false and never interprets recipe keys.
    pub fb_warm: bool,
    /// Transitional LamQuant compatibility override for decode workers. The
    /// generic CLI leaves it `None`.
    pub admitted_workers: Option<u32>,
    /// Transitional LamQuant compatibility override for batch size. The
    /// generic CLI leaves it `None`.
    pub admitted_batch_size: Option<u32>,
    /// ADR 0103 node-keyed effective profiles. Selection is resolved once from
    /// immutable launch hints before scheduling; each StageContext sees only
    /// its own node's value. Empty preserves legacy behavior.
    pub(crate) training_io_profiles: HashMap<NodeId, crate::framework::async_io::TrainingIoProfile>,
    training_io_node_hints: HashMap<NodeId, crate::framework::async_io::TrainingIoHints>,
    /// One private immutable resolver shared by static preparation and runtime
    /// graph injection. Populated only by `prepare_plan_with_mode`.
    training_io_resolver: Option<TrainingIoResolver>,
    /// Optional launcher/HPO calibration callback captured into the resolver.
    /// It receives actual post-DCE or injected node args and must fail closed
    /// when a required child-specific floor cannot be produced.
    training_io_node_admission_resolver: Option<TrainingIoNodeAdmissionResolver>,
    /// Force every declaring stage onto its explicit Inline candidate.
    pub sync_io: bool,
    /// Optional live-availability ceiling used only while selecting a profile.
    /// Executor permits still use `memory_budget_gib`; this tighter byte value
    /// prevents choosing a fast profile that whole-job admission would
    /// immediately refuse under the same resource snapshot.
    pub training_io_selection_budget_bytes: Option<u64>,
    /// Calibrated/OOM-corrected whole-job synchronous-base FLOOR used for
    /// profile selection when the production launcher has one declaring node.
    /// Stage-owned structure (for example DDP rank replication) may raise it.
    /// `None` lets each stage derive its exact base from args + immutable hints.
    pub training_io_whole_job_base_bytes: Option<u64>,
    /// Explicit conservative downgrade selected by the launch boundary. The
    /// framework derives the same reason for direct callers when this is absent.
    training_io_downgrade_reason: Option<crate::framework::async_io::TrainingIoDowngradeReason>,
    /// Select-once witness consumed by the chosen executor implementation.
    prepared_execution: Option<PreparedExecution>,
    #[cfg(test)]
    status_writer_failure: Option<std::io::ErrorKind>,
    #[cfg(test)]
    status_writer_started: Option<Arc<std::sync::atomic::AtomicBool>>,
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
    /// DAG optimizer. When set, the executor runs its enabled passes before
    /// execution. The established DCE/critical-path/memory passes default on;
    /// advanced cache-aware and user-priority ordering default off.
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
        resources.insert(Resource::Cpu, Arc::new(tokio::sync::Semaphore::new(cpu_n)));
        resources.insert(Resource::Network, Arc::new(tokio::sync::Semaphore::new(4)));
        resources.insert(Resource::Disk, Arc::new(tokio::sync::Semaphore::new(2)));
        Self {
            job_dir,
            cache,
            tenant: crate::tenant::Tenant::default(),
            status,
            lifecycle_rx: Some(lifecycle_rx),
            cancel,
            resources,
            // Default 1 device ⇒ byte-identical to the legacy Semaphore::new(1);
            // the CLI sizes it from the live inventory via with_gpu_scheduler.
            gpu: Arc::new(crate::broker::gpu::GpuScheduler::new(
                crate::broker::gpu::GpuInventory::default(),
            )),
            max_in_flight: DEFAULT_MAX_IN_FLIGHT,
            deadline: None,
            on_retry: None,
            // Effectively unlimited until the CLI binds it to the job reservation; a stage
            // requesting MEMORY_GIB ≪ this never blocks, so default = no gating.
            memory: Arc::new(tokio::sync::Semaphore::new(UNLIMITED_MEM_GIB as usize)),
            memory_budget_gib: UNLIMITED_MEM_GIB,
            launch_target: crate::config::launcher::LaunchTarget::Local,
            control: None,
            fb_warm: false,
            admitted_workers: None,
            admitted_batch_size: None,
            training_io_profiles: HashMap::new(),
            training_io_node_hints: HashMap::new(),
            training_io_resolver: None,
            training_io_node_admission_resolver: None,
            sync_io: false,
            training_io_selection_budget_bytes: None,
            training_io_whole_job_base_bytes: None,
            training_io_downgrade_reason: None,
            prepared_execution: None,
            #[cfg(test)]
            status_writer_failure: None,
            #[cfg(test)]
            status_writer_started: None,
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

    /// Bind this execution to a tenant (ADR 0096).
    pub fn with_tenant(mut self, tenant: crate::tenant::Tenant) -> Self {
        self.tenant = tenant;
        self
    }

    /// Pin the job to GPU `device_index` (Phase-G scheduler). Default `None`
    /// (box default device).
    pub fn with_device_index(mut self, device_index: Option<usize>) -> Self {
        self.device_index = device_index;
        self
    }

    /// Set the transitional LamQuant warm-cache compatibility signal.
    pub fn with_fb_warm(mut self, warm: bool) -> Self {
        self.fb_warm = warm;
        self
    }
    /// Set the transitional LamQuant decode-worker compatibility override.
    pub fn with_admitted_workers(mut self, workers: u32) -> Self {
        self.admitted_workers = Some(workers);
        self
    }

    /// Set the transitional LamQuant batch-size compatibility override.
    pub fn with_admitted_batch_size(mut self, batch_size: u32) -> Self {
        self.admitted_batch_size = Some(batch_size);
        self
    }

    /// Force every stage with async-I/O candidates onto its explicit fully
    /// inline tail. This is execution-only and cache-neutral.
    pub fn with_sync_io(mut self, sync_io: bool) -> Self {
        self.sync_io = sync_io;
        self
    }

    pub fn with_training_io_selection_budget_bytes(mut self, budget_bytes: u64) -> Self {
        self.training_io_selection_budget_bytes = Some(budget_bytes);
        self
    }

    /// Select a single declaring stage against at least this calibrated
    /// whole-job base. Resolution rejects ambiguity if more than one node
    /// declares profiles.
    pub fn with_training_io_whole_job_base_bytes(mut self, base_bytes: u64) -> Self {
        self.training_io_whole_job_base_bytes = Some(base_bytes);
        self
    }

    /// Install a launch-owned child-aware hints/calibration resolver.
    /// Crate-private by design: cookbook authors declare candidates and stage
    /// structure, while the trusted launch admission path owns calibration.
    pub(crate) fn with_training_io_node_admission_resolver(
        mut self,
        resolver: TrainingIoNodeAdmissionResolver,
    ) -> Self {
        self.training_io_node_admission_resolver = Some(resolver);
        self
    }

    pub(crate) fn set_training_io_downgrade_reason(
        &mut self,
        reason: Option<crate::framework::async_io::TrainingIoDowngradeReason>,
    ) {
        self.training_io_downgrade_reason = reason;
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
        // GPU is no longer a plain semaphore (ADR 0087): sizing the "Gpu limit"
        // sizes the per-device scheduler to `permits` homogeneous devices, so
        // existing callers/tests keep the same concurrency semantics.
        if resource == Resource::Gpu {
            self.gpu = Arc::new(crate::broker::gpu::GpuScheduler::new(
                crate::broker::gpu::GpuInventory::homogeneous(permits.max(1), 40960),
            ));
            return self;
        }
        self.resources
            .insert(resource, Arc::new(tokio::sync::Semaphore::new(permits)));
        self
    }

    /// Install a GPU scheduler built from the live inventory (ADR 0087) — the
    /// CLI's production path (VRAM-aware placement). Replaces `with_resource_limit
    /// (Gpu, …)`'s homogeneous sizing.
    pub fn with_gpu_scheduler(mut self, sched: crate::broker::gpu::GpuScheduler) -> Self {
        self.gpu = Arc::new(sched);
        self
    }

    /// Size the concurrent-stage memory budget to `gib`.
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
    tenant: crate::tenant::Tenant,
    status: Arc<StatusHub>,
    cancel: CancellationToken,
    resources: HashMap<Resource, Arc<tokio::sync::Semaphore>>,
    /// GPU-aware scheduler (ADR 0087): per-device exclusive permits + VRAM-aware
    /// placement, in series with the RAM `memory` broker below. Replaces the old
    /// single `Resource::Gpu` semaphore; sized 1 == the legacy behaviour.
    gpu: Arc<crate::broker::gpu::GpuScheduler>,
    memory: Arc<tokio::sync::Semaphore>,
    memory_budget_gib: u32,
    launch_target: crate::config::launcher::LaunchTarget,
    device_index: Option<usize>,
    fb_warm: bool,
    admitted_workers: Option<u32>,
    admitted_batch_size: Option<u32>,
    /// Insert-once node profiles. Runtime Spawn extends this registry on the
    /// single-threaded coordinator seam before a new node becomes ready; task
    /// threads can only clone an already-resolved immutable profile.
    training_io_profiles: std::sync::RwLock<HashMap<NodeId, ResolvedTrainingIoNode>>,
    training_io_resolver: TrainingIoResolver,
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

impl NodeEnv {
    fn training_io_node(&self, node_id: NodeId) -> Option<ResolvedTrainingIoNode> {
        self.training_io_profiles
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&node_id)
            .cloned()
    }

    /// Install a fully-resolved batch for fresh global node ids. No caller can
    /// update an existing entry: one node receives one immutable profile for
    /// its whole cache/admission/execution lifecycle.
    fn install_training_io_profiles(
        &self,
        profiles: impl IntoIterator<Item = (NodeId, ResolvedTrainingIoNode)>,
    ) -> Result<(), PlanError> {
        let profiles: Vec<_> = profiles.into_iter().collect();
        let mut selected = self
            .training_io_profiles
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((node_id, _)) = profiles
            .iter()
            .find(|(node_id, _)| selected.contains_key(node_id))
        {
            return Err(PlanError::Other(format!(
                "training I/O profile for runtime node {node_id} was already resolved"
            )));
        }
        selected.extend(profiles);
        Ok(())
    }

    fn training_io_snapshot(&self) -> HashMap<NodeId, ResolvedTrainingIoNode> {
        self.training_io_profiles
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
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
    /// Portable identities of predecessor outputs. Kept distinct from
    /// `input_hash`, which is the folded logical digest used by invocation keys.
    input_content_ids: Vec<ContentId>,
    /// Logical predecessor identity used only for deterministic invocation keys.
    input_hash: ContentHash,
    key: InvocationKey,
    /// Cache artifact already decoded by cache-aware ready-queue probing. The
    /// parallel executor carries it into `run_node` so a warm node is read once
    /// and can short-circuit before optional remote dispatch.
    prepared_cache_hit: Option<Arc<CacheHit>>,
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
    /// Present only for an optimizer-witnessed, runtime-admitted map parent.
    /// The stage must explicitly emit manifest-indexed items through this
    /// semantic lane; ordinary/default-off tasks retain `None`.
    pipeline_emitter: Option<crate::framework::stage::PipelineEmitter>,
}

/// A node's result, fed back to the coordinator to advance scheduling.
struct NodeOutcome {
    node_id: NodeId,
    output: ErasedArtifact,
    /// Present only on the fused fast path. The next stage consumes this box
    /// directly instead of decoding `output` through bincode again.
    in_process_output: Option<InProcessArtifact>,
    content_id: ContentId,
    logical: ContentHash,
    cache_hit: bool,
}

struct StageRunOutput {
    erased: ErasedArtifact,
    in_process: Option<InProcessArtifact>,
}

/// A private speculative result. Until this value is consumed by
/// `publish_speculative`, every byte and status event remains under the scratch
/// root and is invisible to the canonical job/cache/lineage surfaces.
struct SpeculativePrepared {
    _scratch: SpeculationScratch,
    scratch_stage_dir: PathBuf,
    node_id: NodeId,
    node_idx: u32,
    stage: Arc<dyn StageDyn>,
    stage_name: String,
    input_hash: ContentHash,
    input_content_ids: Vec<ContentId>,
    canon_args: Vec<u8>,
    key: InvocationKey,
    output: ErasedArtifact,
    elapsed: std::time::Duration,
    training_io_profile: Option<crate::framework::async_io::TrainingIoProfile>,
    buffered_steps: Vec<StageEvent>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PipelinePreparedPayload {
    output: ErasedArtifact,
    buffered_steps: Vec<StageEvent>,
}

struct BoundedPipelineSpillWriter<W> {
    inner: W,
    remaining: u64,
}

impl<W: std::io::Write> std::io::Write for BoundedPipelineSpillWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let byte_count = u64::try_from(bytes.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "pipeline spill write length cannot be represented",
            )
        })?;
        if byte_count > self.remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "pipeline spill exceeds the admitted item bound",
            ));
        }
        let written = self.inner.write(bytes)?;
        self.remaining = self
            .remaining
            .checked_sub(written as u64)
            .expect("writer cannot report more bytes than it received");
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn write_pipeline_spill<T: serde::Serialize>(
    path: &std::path::Path,
    value: &T,
    max_bytes: u64,
) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut writer = BoundedPipelineSpillWriter {
        inner: std::io::BufWriter::new(file),
        remaining: max_bytes,
    };
    bincode::serialize_into(&mut writer, value).map_err(std::io::Error::other)?;
    std::io::Write::flush(&mut writer)
}

fn read_pipeline_spill<T: serde::de::DeserializeOwned>(
    path: &std::path::Path,
    max_bytes: u64,
) -> std::io::Result<T> {
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    if file_len > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "pipeline spill is {file_len} bytes, above its {max_bytes}-byte admitted bound"
            ),
        ));
    }
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(max_bytes)
        .deserialize_from(std::io::BufReader::new(file))
        .map_err(std::io::Error::other)
}

/// Disk-backed private child result. A producer may emit thousands of map
/// items before it returns its authoritative `ListOf`; retaining every child
/// output/status buffer in RAM until that validation point would make the
/// pipeline queue bounded in name only. Each completed child is therefore
/// spilled under its already-private scratch root before the lane slot is
/// returned to the producer.
struct SpilledPipelinePrepared {
    _scratch: SpeculationScratch,
    scratch_stage_dir: PathBuf,
    input_path: PathBuf,
    payload_path: PathBuf,
    node_id: NodeId,
    node_idx: u32,
    stage: Arc<dyn StageDyn>,
    stage_name: String,
    input_hash: ContentHash,
    input_content_ids: Vec<ContentId>,
    canon_args: Vec<u8>,
    key: InvocationKey,
    elapsed: std::time::Duration,
    training_io_profile: Option<crate::framework::async_io::TrainingIoProfile>,
    max_spill_bytes: u64,
}

impl SpilledPipelinePrepared {
    fn spill(
        prepared: SpeculativePrepared,
        input: &ErasedArtifact,
        max_spill_bytes: u64,
    ) -> Result<Self, NodeFailure> {
        let SpeculativePrepared {
            _scratch,
            scratch_stage_dir,
            node_id,
            node_idx,
            stage,
            stage_name,
            input_hash,
            input_content_ids,
            canon_args,
            key,
            output,
            elapsed,
            training_io_profile,
            buffered_steps,
        } = prepared;
        let input_path = _scratch.0.join("pipeline-input.bin");
        let payload_path = _scratch.0.join("pipeline-prepared.bin");
        let write = || -> std::io::Result<()> {
            write_pipeline_spill(&input_path, input, max_spill_bytes)?;
            write_pipeline_spill(
                &payload_path,
                &PipelinePreparedPayload {
                    output,
                    buffered_steps,
                },
                max_spill_bytes,
            )
        };
        if let Err(source) = write() {
            let path = _scratch.0.clone();
            if let Err(cleanup_source) = _scratch.cleanup() {
                return Err(NodeFailure::SpeculationCleanup {
                    path,
                    source: cleanup_source,
                });
            }
            return Err(NodeFailure::Other(format!(
                "pipeline private-result spill at {} failed: {source}",
                path.display()
            )));
        }
        Ok(Self {
            _scratch,
            scratch_stage_dir,
            input_path,
            payload_path,
            node_id,
            node_idx,
            stage,
            stage_name,
            input_hash,
            input_content_ids,
            canon_args,
            key,
            elapsed,
            training_io_profile,
            max_spill_bytes,
        })
    }

    fn input_matches(
        &self,
        expected: &ErasedArtifact,
        producer: &dyn StageDyn,
        producer_tmp_stage_dir: &std::path::Path,
        producer_final_stage_dir: &std::path::Path,
    ) -> Result<bool, NodeFailure> {
        let actual: ErasedArtifact = read_pipeline_spill(&self.input_path, self.max_spill_bytes)
            .map_err(|source| {
                NodeFailure::Other(format!(
                    "pipeline input spill {} could not be decoded: {source}",
                    self.input_path.display()
                ))
            })?;
        let actual =
            producer.rebase_output_paths(actual, producer_tmp_stage_dir, producer_final_stage_dir);
        Ok(actual.kind == expected.kind
            && actual.schema == expected.schema
            && actual.payload == expected.payload)
    }

    fn validate_payload(&self) -> Result<(), NodeFailure> {
        read_pipeline_spill::<PipelinePreparedPayload>(&self.payload_path, self.max_spill_bytes)
            .map(|_| ())
            .map_err(|source| {
                NodeFailure::Other(format!(
                    "pipeline private-result spill {} could not be decoded: {source}",
                    self.payload_path.display()
                ))
            })
    }

    fn into_resident(self) -> Result<SpeculativePrepared, NodeFailure> {
        let payload: PipelinePreparedPayload =
            match read_pipeline_spill(&self.payload_path, self.max_spill_bytes) {
                Ok(payload) => payload,
                Err(source) => return self.load_failure(source),
            };
        Ok(SpeculativePrepared {
            _scratch: self._scratch,
            scratch_stage_dir: self.scratch_stage_dir,
            node_id: self.node_id,
            node_idx: self.node_idx,
            stage: self.stage,
            stage_name: self.stage_name,
            input_hash: self.input_hash,
            input_content_ids: self.input_content_ids,
            canon_args: self.canon_args,
            key: self.key,
            output: payload.output,
            elapsed: self.elapsed,
            training_io_profile: self.training_io_profile,
            buffered_steps: payload.buffered_steps,
        })
    }

    fn load_failure(self, source: std::io::Error) -> Result<SpeculativePrepared, NodeFailure> {
        let path = self._scratch.0.clone();
        let detail = format!(
            "pipeline private-result spill {} could not be loaded: {source}",
            self.payload_path.display()
        );
        if let Err(cleanup_source) = self._scratch.cleanup() {
            return Err(NodeFailure::SpeculationCleanup {
                path,
                source: cleanup_source,
            });
        }
        Err(NodeFailure::Other(detail))
    }

    fn discard(self) -> Result<(), NodeFailure> {
        let path = self._scratch.0.clone();
        self._scratch
            .cleanup()
            .map_err(|source| NodeFailure::SpeculationCleanup { path, source })
    }
}

impl SpeculativePrepared {
    fn discard(self) -> Result<(), NodeFailure> {
        let path = self._scratch.0.clone();
        self._scratch
            .cleanup()
            .map_err(|source| NodeFailure::SpeculationCleanup { path, source })
    }
}

/// Removes all private work on rejection, failure, cancellation, or plan exit.
struct SpeculationScratch(PathBuf);

impl SpeculationScratch {
    fn reset(&self) -> std::io::Result<()> {
        match std::fs::remove_dir_all(&self.0) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        std::fs::create_dir_all(&self.0)
    }

    fn cleanup(&self) -> std::io::Result<()> {
        Self::cleanup_path(&self.0)
    }

    fn cleanup_path(path: &std::path::Path) -> std::io::Result<()> {
        #[cfg(test)]
        if path
            .file_name()
            .is_some_and(|name| name == std::ffi::OsStr::new(".blut-test-pipeline-cleanup-failure"))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "injected pipeline cleanup failure",
            ));
        }
        match std::fs::remove_dir_all(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        if let Some(parent) = path.parent() {
            match std::fs::remove_dir(parent) {
                Ok(()) => {}
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty
                    ) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

impl Drop for SpeculationScratch {
    fn drop(&mut self) {
        if let Err(error) = self.cleanup() {
            tracing::error!(
                "executor: failed to remove private speculation scratch {}: {error}",
                self.0.display()
            );
        }
    }
}

/// Removes a renamed stage directory if selected publication unwinds before
/// its canonical cache/status commit is complete.
struct SpeculativePublishGuard {
    final_stage_dir: PathBuf,
    committed: bool,
    rollback_failure: Arc<std::sync::Mutex<Option<std::io::Error>>>,
}

/// Removes a pipeline-selected cache entry when cancellation, deadline, panic,
/// or another publication failure wins before the lifecycle commit. Pipeline
/// launch is disabled for force-recompute and shared cache tiers, and its exact
/// key was probed cold, so this guard never removes a pre-existing entry.
struct PipelineCachePublishGuard {
    entry_path: PathBuf,
    committed: bool,
    rollback_failure: Arc<std::sync::Mutex<Option<std::io::Error>>>,
}

impl PipelineCachePublishGuard {
    fn new(
        entry_path: PathBuf,
        rollback_failure: Arc<std::sync::Mutex<Option<std::io::Error>>>,
    ) -> Self {
        Self {
            entry_path,
            committed: false,
            rollback_failure,
        }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for PipelineCachePublishGuard {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let removal = match std::fs::remove_file(&self.entry_path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
        .and_then(|()| {
            let Some(parent) = self.entry_path.parent() else {
                return Ok(());
            };
            match std::fs::remove_dir(parent) {
                Ok(()) => Ok(()),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::DirectoryNotEmpty
                    ) =>
                {
                    Ok(())
                }
                Err(error) => Err(error),
            }
        });
        if let Err(error) = removal {
            *self
                .rollback_failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
        }
    }
}

impl SpeculativePublishGuard {
    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for SpeculativePublishGuard {
    fn drop(&mut self) {
        if !self.committed
            && let Err(error) = std::fs::remove_dir_all(&self.final_stage_dir)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::error!(
                "executor: failed to roll back speculative publication {}: {error}",
                self.final_stage_dir.display()
            );
            let mut failure = self
                .rollback_failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if failure.is_none() {
                *failure = Some(error);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpeculationDisposition {
    Pending,
    Selected,
    Rejected,
    Superseded,
}

enum SpeculationState {
    Eligible,
    Running {
        cancel: CancellationToken,
        disposition: SpeculationDisposition,
    },
    Prepared(Box<SpeculativePrepared>),
    Declined,
    Rejected,
}

enum SchedulerTaskResult {
    Ordinary(Result<Vec<NodeOutcome>, NodeFailure>),
    Pipeline(Result<PipelineRunResult, NodeFailure>),
    Speculative {
        target: NodeId,
        key: InvocationKey,
        result: Result<Box<SpeculativePrepared>, NodeFailure>,
    },
}

/// Pipeline work occupies two scheduler slots (producer + one private
/// consumer) even though it returns through one `JoinSet` task. The extra
/// weighted slot is released on success, error, panic, cancellation, or task
/// abort, preventing the coordinator from oversubscribing `max_in_flight`.
struct PipelineInFlightExtra(Arc<std::sync::atomic::AtomicUsize>);

impl PipelineInFlightExtra {
    fn reserve(counter: Arc<std::sync::atomic::AtomicUsize>) -> Self {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(counter)
    }
}

impl Drop for PipelineInFlightExtra {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[derive(Clone)]
struct PipelineChildSpec {
    node: crate::framework::plan::PlanNode,
    node_idx: u32,
    input_logical: ContentHash,
    key: InvocationKey,
}

struct PipelineRunResult {
    /// Keep the exact combined parent-plus-child envelope held through spill
    /// validation and selected publication, not only while stage futures run.
    _admission: Arc<AdmissionLease>,
    parent: NodeOutcome,
    elements: Vec<ErasedArtifact>,
    children: Vec<PipelineChildSpec>,
    profiles: Vec<(NodeId, ResolvedTrainingIoNode)>,
    prepared: Option<Vec<SpilledPipelinePrepared>>,
    spawn_capacity: usize,
}

struct PipelineLaunch {
    receiver: tokio::sync::mpsc::UnboundedReceiver<crate::framework::stage::PipelineEmission>,
    admission: Arc<AdmissionLease>,
    manifest: crate::framework::stage::PipelineManifest,
    predicted_parent_logical: ContentHash,
    children: Vec<PipelineChildSpec>,
    profiles: Vec<(NodeId, ResolvedTrainingIoNode)>,
    producer_stage: Arc<dyn StageDyn>,
    producer_tmp_stage_dir: PathBuf,
    producer_final_stage_dir: PathBuf,
    spawn_capacity: usize,
    max_item_bytes: u64,
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
    /// A plan-level stop or coordinator invariant detected inside a grouped
    /// execution boundary. Preserve its exact public classification when the
    /// grouped task rejoins the coordinator.
    Plan(PlanError),
    /// An executor-internal failure (e.g. a closed semaphore).
    Other(String),
    /// Optional work could not restore its private scratch boundary. This is
    /// fatal even when the computation itself was disposable: returning plan
    /// success with observable residue would violate discard equivalence.
    SpeculationCleanup {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A selected result was renamed into its canonical stage path, then a
    /// publication hook failed and that incomplete path could not be removed.
    SpeculationRollback {
        path: PathBuf,
        source: std::io::Error,
        cause: String,
    },
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
async fn cache_lookup_off_thread(
    cache: Arc<CacheHandle>,
    key: InvocationKey,
    stage: Arc<dyn StageDyn>,
    into_stage_dir: PathBuf,
) -> Result<Option<CacheHit>, tokio::task::JoinError> {
    tokio::task::spawn_blocking(move || cache.lookup(key, stage.as_ref(), &into_stage_dir)).await
}

async fn tagged_cache_lookup_off_thread(
    cache: Arc<CacheHandle>,
    key: InvocationKey,
    stage: Arc<dyn StageDyn>,
    into_stage_dir: PathBuf,
) -> (
    InvocationKey,
    Result<Option<CacheHit>, tokio::task::JoinError>,
) {
    (
        key,
        cache_lookup_off_thread(cache, key, stage, into_stage_dir).await,
    )
}

async fn cache_presence_probe_off_thread(
    cache: Arc<CacheHandle>,
    key: InvocationKey,
) -> Result<std::io::Result<bool>, tokio::task::JoinError> {
    tokio::task::spawn_blocking(move || cache.probe_presence(key)).await
}

fn install_builtin_control(ctx: &mut ExecCtx) {
    if ctx.control.is_none()
        && std::env::var("BLUT_KILL_ON_NAN")
            .map(|value| value == "1")
            .unwrap_or(false)
    {
        ctx.control = Some(Arc::new(crate::framework::control::KillOnNaN));
    }
}

fn plan_requires_parallel(plan: &CompiledPlan, ctx: &ExecCtx) -> bool {
    ctx.control.is_some() || plan.has_condition_gates() || !plan.expansions().is_empty()
}

fn auto_executor_mode(plan: &CompiledPlan, ctx: &ExecCtx) -> PreparedExecutorMode {
    if plan_requires_parallel(plan, ctx)
        || std::env::var("BLUT_EXECUTOR")
            .map(|value| value.eq_ignore_ascii_case("parallel"))
            .unwrap_or(false)
    {
        PreparedExecutorMode::Parallel
    } else {
        PreparedExecutorMode::Sequential
    }
}

fn requested_executor_mode(
    requested: PreparedExecutorMode,
    plan: &CompiledPlan,
    ctx: &ExecCtx,
) -> PreparedExecutorMode {
    if requested == PreparedExecutorMode::Parallel || plan_requires_parallel(plan, ctx) {
        PreparedExecutorMode::Parallel
    } else {
        PreparedExecutorMode::Sequential
    }
}

/// Optimize (when parallel), select every training-I/O profile exactly once,
/// and retain the executor/scheduling witness privately on `ctx`.
pub(crate) fn prepare_plan_for_execution(
    plan: CompiledPlan,
    ctx: &mut ExecCtx,
) -> Result<CompiledPlan, PlanError> {
    install_builtin_control(ctx);
    let mode = auto_executor_mode(&plan, ctx);
    prepare_plan_with_mode(plan, ctx, mode)
}

/// Crate-private prepared-plan seam for launchers whose scheduling contract is
/// intrinsically parallel even when the runtime control policy is absent (for
/// example a static HPO trial batch). It still executes the exact same
/// optimize/select-once path and records one private mode witness for
/// `execute_plan`; this is not a second profile-selection mechanism.
pub(crate) fn prepare_plan_for_parallel_execution(
    plan: CompiledPlan,
    ctx: &mut ExecCtx,
) -> Result<CompiledPlan, PlanError> {
    prepare_plan_for_requested_executor(plan, ctx, PreparedExecutorMode::Parallel)
}

fn prepare_plan_for_requested_executor(
    plan: CompiledPlan,
    ctx: &mut ExecCtx,
    requested: PreparedExecutorMode,
) -> Result<CompiledPlan, PlanError> {
    install_builtin_control(ctx);
    let mode = requested_executor_mode(requested, &plan, ctx);
    prepare_plan_with_mode(plan, ctx, mode)
}

fn prepare_plan_with_mode(
    plan: CompiledPlan,
    ctx: &mut ExecCtx,
    mode: PreparedExecutorMode,
) -> Result<CompiledPlan, PlanError> {
    if ctx.prepared_execution.is_some() {
        return Err(PlanError::Other(
            "execution context already owns a prepared plan witness".into(),
        ));
    }
    if !ctx.training_io_profiles.is_empty() {
        return Err(PlanError::Other(
            "training I/O profiles are executor-selected and cannot be preloaded".into(),
        ));
    }
    if ctx.training_io_resolver.is_some() {
        return Err(PlanError::Other(
            "training I/O resolver is executor-owned and cannot be preloaded".into(),
        ));
    }

    // The optimizer is the sole owner of post-DCE node ids. Candidate methods
    // therefore run only after this transform and never need a second pass.
    let (plan, schedule_hints) = match mode {
        PreparedExecutorMode::Sequential => (plan, HashMap::new()),
        PreparedExecutorMode::Parallel => match ctx.dag_optimizer.as_ref() {
            Some(optimizer) => optimizer.optimize(plan),
            None => (plan, HashMap::new()),
        },
    };
    resolve_training_io_profiles_once(&plan, ctx)?;
    ctx.prepared_execution = Some(PreparedExecution {
        mode,
        schedule_hints,
    });
    Ok(plan)
}

fn take_prepared_execution(
    ctx: &mut ExecCtx,
    expected: PreparedExecutorMode,
) -> Result<PreparedExecution, PlanError> {
    let prepared = ctx.prepared_execution.take().ok_or_else(|| {
        PlanError::Other("executor received a plan without a preparation witness".into())
    })?;
    if prepared.mode != expected {
        return Err(PlanError::Other(format!(
            "prepared executor mode mismatch: expected {expected:?}, found {:?}",
            prepared.mode
        )));
    }
    Ok(prepared)
}

fn resolve_training_io_profiles_once(
    plan: &CompiledPlan,
    ctx: &mut ExecCtx,
) -> Result<(), PlanError> {
    let resolver = TrainingIoResolver::from_ctx(ctx)?;
    let selected_nodes = resolver.resolve_static_plan(plan)?;
    ctx.training_io_profiles = selected_nodes
        .iter()
        .map(|(&node_id, selected)| (node_id, selected.profile.clone()))
        .collect();
    ctx.training_io_node_hints = selected_nodes
        .into_iter()
        .map(|(node_id, selected)| (node_id, selected.hints))
        .collect();
    ctx.training_io_resolver = Some(resolver);
    Ok(())
}

impl TrainingIoResolver {
    fn from_ctx(ctx: &ExecCtx) -> Result<Self, PlanError> {
        use crate::framework::async_io::{TrainingIoDowngradeReason, TrainingIoHints};

        let hints = TrainingIoHints {
            admitted_decode_workers: ctx.admitted_workers,
            admitted_batch_size: ctx.admitted_batch_size,
            cache_warm: ctx.fb_warm,
        };
        let executor_budget_bytes = u64::from(ctx.memory_budget_gib)
            .checked_mul(crate::broker::footprint::GIB)
            .ok_or_else(|| PlanError::Other("training I/O memory budget overflow".into()))?;
        let budget_bytes = ctx
            .training_io_selection_budget_bytes
            .map_or(executor_budget_bytes, |live| {
                live.min(executor_budget_bytes)
            });

        let force_inline_reason = if ctx.sync_io {
            Some(TrainingIoDowngradeReason::UserForced)
        } else if ctx.launch_target != crate::config::launcher::LaunchTarget::Local {
            Some(TrainingIoDowngradeReason::UnsupportedLauncher)
        } else if ctx.training_io_selection_budget_bytes.is_none() {
            Some(TrainingIoDowngradeReason::SnapshotUnavailable)
        } else {
            ctx.training_io_downgrade_reason.clone()
        };

        Ok(Self {
            default_hints: hints,
            budget_bytes,
            force_inline_reason,
            whole_job_base_bytes: ctx.training_io_whole_job_base_bytes,
            node_admission: ctx.training_io_node_admission_resolver.clone(),
        })
    }

    /// Resolve a complete node batch from the already-captured launch facts.
    /// Candidate declaration and optional calibration each run exactly once per
    /// declaring node. This method performs no resource/snapshot probe.
    fn resolve_static_plan(
        &self,
        plan: &CompiledPlan,
    ) -> Result<HashMap<NodeId, ResolvedTrainingIoNode>, PlanError> {
        let pipeline_parents = plan
            .expansions
            .iter()
            .filter(|expansion| expansion.pipeline)
            .map(|expansion| expansion.parent)
            .collect();
        self.resolve_plan_with_global_floor(plan, self.whole_job_base_bytes, &pipeline_parents)
    }

    fn resolve_injected_plan(
        &self,
        plan: &CompiledPlan,
    ) -> Result<HashMap<NodeId, ResolvedTrainingIoNode>, PlanError> {
        // A launcher's one whole-job floor describes the prepared parent job,
        // not an arbitrary later PBT/TPE/map child. Injected nodes must use
        // their own stage-derived base plus an optional child-specific floor
        // from the captured callback.
        self.resolve_plan_with_global_floor(plan, None, &HashSet::new())
    }

    fn resolve_plan_with_global_floor(
        &self,
        plan: &CompiledPlan,
        whole_job_base_bytes: Option<u64>,
        pipeline_parents: &HashSet<NodeId>,
    ) -> Result<HashMap<NodeId, ResolvedTrainingIoNode>, PlanError> {
        use crate::framework::async_io::{TrainingIoHints, select_training_io_profile_with_reason};

        let mut declarations = Vec::with_capacity(plan.nodes.len());
        for node in &plan.nodes {
            let admission_recipe_args = node
                .admission_scope_args
                .as_deref()
                .or(node.admission_recipe_args.as_deref())
                .unwrap_or(plan.recipe_args());
            let decision = self
                .node_admission
                .as_ref()
                .map(|resolve| resolve(node.stage.name(), &node.args, admission_recipe_args))
                .transpose()
                .map_err(|error| {
                    PlanError::Other(format!(
                        "node {} ({}) training I/O calibration refused: {error}",
                        node.id,
                        node.stage.name()
                    ))
                })?
                .flatten();
            let hints = decision.map_or(self.default_hints, |decision| TrainingIoHints {
                admitted_decode_workers: decision.admitted_decode_workers,
                admitted_batch_size: decision.admitted_batch_size,
                cache_warm: decision.cache_warm,
            });
            let calibrated_floor =
                decision.and_then(|decision| decision.calibrated_base_floor_bytes);
            let selection_budget = decision.and_then(|decision| decision.selection_budget_bytes);
            let mut candidates = node.stage.training_io_candidates(&node.args, hints);
            let pipeline_filtered = !pipeline_parents.contains(&node.id)
                && candidates
                    .iter()
                    .any(|candidate| !candidate.pipeline.is_inline());
            if pipeline_filtered {
                candidates.retain(|candidate| candidate.pipeline.is_inline());
            }
            declarations.push((
                node,
                hints,
                calibrated_floor,
                selection_budget,
                candidates,
                pipeline_filtered,
            ));
        }
        let declaring_count = declarations
            .iter()
            .filter(|(_, _, _, _, candidates, _)| !candidates.is_empty())
            .count();
        if whole_job_base_bytes.is_some() && declaring_count > 1 {
            return Err(PlanError::Other(format!(
                "calibrated whole-job async-I/O admission supports exactly one declaring node, found {declaring_count}"
            )));
        }

        let mut selected_profiles = HashMap::new();
        for (node, hints, calibrated_floor, selection_budget, candidates, pipeline_filtered) in
            declarations
        {
            if candidates.is_empty() {
                continue;
            }

            let stage_base_bytes = node.stage.training_io_sync_base_bytes(&node.args, hints);
            // Calibration is a conservative floor, not a replacement for
            // stage-owned structure such as DDP rank replication. The same
            // rule applies to a global launcher floor and a child-aware HPO
            // floor captured in this resolver.
            let base_bytes = [
                Some(stage_base_bytes),
                whole_job_base_bytes,
                calibrated_floor,
            ]
            .into_iter()
            .flatten()
            .max()
            .expect("stage base is always present");

            let node_budget = selection_budget
                .map(|limit| limit.min(self.budget_bytes))
                .unwrap_or(self.budget_bytes);
            let mut selected = select_training_io_profile_with_reason(
                base_bytes,
                node_budget,
                &candidates,
                self.force_inline_reason.clone(),
            )
            .map_err(|error| {
                PlanError::Other(format!(
                    "node {} ({}) training I/O admission refused: {error}",
                    node.id,
                    node.stage.name()
                ))
            })?;
            if pipeline_filtered && selected.downgrade_reason.is_none() {
                selected.downgrade_reason = Some(
                    crate::framework::async_io::TrainingIoDowngradeReason::PipelineUnavailable,
                );
            }
            selected_profiles.insert(
                node.id,
                ResolvedTrainingIoNode {
                    profile: selected,
                    hints,
                },
            );
        }

        Ok(selected_profiles)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AdmissionRequest {
    resources: Vec<Resource>,
    gpu: Option<crate::broker::gpu::GpuRequest>,
    memory_gib: u32,
}

impl AdmissionRequest {
    fn for_stage(
        stage: &dyn StageDyn,
        args: &serde_json::Value,
        memory_budget_gib: u32,
        profile: Option<&crate::framework::async_io::TrainingIoProfile>,
    ) -> Result<Self, String> {
        let mut resources: Vec<Resource> = stage
            .resources()
            .iter()
            .copied()
            .filter(|resource| *resource != Resource::Gpu)
            .collect();
        resources.sort();
        if profile.is_none() {
            // A stage that declares no profile keeps the exact historical
            // admission behavior. ADR 0103's fail-closed byte envelope applies
            // only after a cookbook opts into the new profile contract.
            return Ok(Self {
                resources,
                gpu: stage
                    .resources()
                    .contains(&Resource::Gpu)
                    .then(|| stage.gpu_request(args)),
                memory_gib: stage.memory_gib_for(args).min(memory_budget_gib),
            });
        }
        let base_bytes = profile.map_or_else(
            || u64::from(stage.memory_gib_for(args)) * crate::broker::footprint::GIB,
            |profile| profile.sync_base_bytes,
        );
        let overhead_bytes = profile.map_or(0, |profile| profile.billed_overhead_bytes);
        let total_bytes = base_bytes
            .checked_add(overhead_bytes)
            .ok_or_else(|| format!("stage '{}' memory envelope overflow", stage.name()))?;
        let memory_gib = total_bytes
            .checked_add(crate::broker::footprint::GIB - 1)
            .ok_or_else(|| format!("stage '{}' rounded memory envelope overflow", stage.name()))?
            / crate::broker::footprint::GIB;
        let memory_gib = u32::try_from(memory_gib)
            .map_err(|_| format!("stage '{}' memory envelope exceeds u32 GiB", stage.name()))?;
        if memory_gib > memory_budget_gib {
            return Err(format!(
                "stage '{}' memory envelope {memory_gib} GiB exceeds memory budget {memory_budget_gib} GiB",
                stage.name()
            ));
        }
        Ok(Self {
            resources,
            gpu: stage
                .resources()
                .contains(&Resource::Gpu)
                .then(|| stage.gpu_request(args)),
            memory_gib,
        })
    }

    fn for_task(task: &NodeTask, env: &NodeEnv) -> Result<Self, String> {
        let selected = env.training_io_node(task.node_id);
        Self::for_stage(
            task.stage.as_ref(),
            &task.args,
            env.memory_budget_gib,
            selected.as_ref().map(|selected| &selected.profile),
        )
    }

    fn for_chain(
        nodes: &[crate::framework::plan::PlanNode],
        memory_budget_gib: u32,
    ) -> Option<Self> {
        let request_for = |node: &crate::framework::plan::PlanNode| {
            Self::for_stage(node.stage.as_ref(), &node.args, memory_budget_gib, None).ok()
        };
        let mut requests = nodes.iter().map(request_for);
        let first = requests.next()??;
        // A chain-wide union would reserve resources for a later node before
        // its input-dependent cache lookup is possible. If that node is warm,
        // fusion could block or fail on a GPU/network/memory envelope it never
        // uses. The first conservative slice therefore fuses only identical
        // envelopes; one shared grant is then exactly what every miss would
        // have requested, never a synthetic or premature superset.
        requests
            .all(|request| request.as_ref() == Some(&first))
            .then_some(first)
    }

    /// Exact simultaneous parent + one-consumer envelope for ADR 0102's
    /// conservative pipeline lane. V1 declines GPU stages because one combined
    /// heterogeneous device grant cannot be divided into truthful per-stage
    /// `StageContext` assignments. Repeated CPU/network/disk resources are
    /// retained as repeated semaphore permits, and memory is summed checked.
    fn combine_pipeline(mut parent: Self, child: Self, memory_budget_gib: u32) -> Option<Self> {
        if parent.gpu.is_some() || child.gpu.is_some() {
            return None;
        }
        parent.resources.extend(child.resources);
        parent.resources.sort();
        parent.memory_gib = parent.memory_gib.checked_add(child.memory_gib)?;
        (parent.memory_gib <= memory_budget_gib).then_some(parent)
    }
}

struct AdmissionLease {
    resources: Vec<tokio::sync::OwnedSemaphorePermit>,
    gpu: Option<crate::broker::gpu::GpuGrant>,
    _memory: Option<tokio::sync::OwnedSemaphorePermit>,
}

impl AdmissionLease {
    fn release_non_gpu_resources(&mut self) {
        self.resources.clear();
    }
}

/// Try to acquire a complete speculative envelope without waiting or emitting
/// normal lifecycle events. Any failed component drops all earlier permits
/// before returning `None`.
fn try_acquire_admission_from(
    request: &AdmissionRequest,
    resources: &HashMap<Resource, Arc<tokio::sync::Semaphore>>,
    gpu_scheduler: &crate::broker::gpu::GpuScheduler,
    memory: &Arc<tokio::sync::Semaphore>,
) -> Option<AdmissionLease> {
    let mut permits = Vec::with_capacity(request.resources.len());
    for resource in &request.resources {
        let semaphore = resources.get(resource)?;
        permits.push(semaphore.clone().try_acquire_owned().ok()?);
    }

    let gpu = match request.gpu {
        Some(request) => Some(gpu_scheduler.try_acquire(request)?),
        None => None,
    };
    let memory = match request.memory_gib {
        0 => None,
        count => Some(memory.clone().try_acquire_many_owned(count).ok()?),
    };

    Some(AdmissionLease {
        resources: permits,
        gpu,
        _memory: memory,
    })
}

/// Whether a speculative request can fit after reserving every ordinary
/// in-flight request against the permits visible right now.
///
/// Ordinary tasks acquire after their authoritative cache lookup, so merely
/// spawning one does not immediately move the underlying semaphores. Counting
/// its full envelope here prevents optional work from taking that unclaimed
/// capacity first. A task that already owns permits is conservatively counted
/// twice (once in `available_permits`, once here); that may decline speculation
/// but can never delay ordinary work.
fn admission_is_spare_after_ordinary(
    candidate: &AdmissionRequest,
    ordinary: &HashMap<NodeId, AdmissionRequest>,
    resources: &HashMap<Resource, Arc<tokio::sync::Semaphore>>,
    gpu: &crate::broker::gpu::GpuScheduler,
    memory: &tokio::sync::Semaphore,
) -> bool {
    let candidate_resources: HashSet<Resource> = candidate.resources.iter().copied().collect();
    for resource in candidate_resources {
        let Some(semaphore) = resources.get(&resource) else {
            return false;
        };
        let candidate_need = candidate
            .resources
            .iter()
            .filter(|&&item| item == resource)
            .count();
        let ordinary_need: usize = ordinary
            .values()
            .map(|request| {
                request
                    .resources
                    .iter()
                    .filter(|&&item| item == resource)
                    .count()
            })
            .sum();
        if semaphore.available_permits() < ordinary_need.saturating_add(candidate_need) {
            return false;
        }
    }

    if let Some(request) = candidate.gpu {
        // A count-only shadow cannot reserve the right devices on a
        // heterogeneous box: a permissive speculative request could otherwise
        // take the sole high-VRAM device before an ordinary request reaches
        // authoritative admission. Optional work may always decline, so keep
        // the ordinary path dominant until a residual-device reservation can be
        // acquired atomically with the speculative grant.
        if ordinary.values().any(|request| request.gpu.is_some()) {
            return false;
        }
        let capacity = gpu.device_count();
        let effective_need = |count: u32| (count.max(1) as usize).min(capacity);
        let candidate_need = effective_need(request.count);
        if gpu.available_device_count() < candidate_need {
            return false;
        }
    }

    if candidate.memory_gib > 0 {
        let ordinary_need: usize = ordinary
            .values()
            .map(|request| request.memory_gib as usize)
            .sum();
        if memory.available_permits() < ordinary_need.saturating_add(candidate.memory_gib as usize)
        {
            return false;
        }
    }

    true
}

struct FusionAdmission {
    request: AdmissionRequest,
    lease: Option<AdmissionLease>,
}

impl FusionAdmission {
    fn new(request: AdmissionRequest) -> Self {
        Self {
            request,
            lease: None,
        }
    }

    async fn ensure_acquired(
        &mut self,
        env: &NodeEnv,
        idx: u32,
        stage_name: &str,
    ) -> Result<&AdmissionLease, NodeFailure> {
        if self.lease.is_none() {
            self.lease = Some(acquire_admission(&self.request, env, idx, stage_name).await?);
        }
        Ok(self.lease.as_ref().expect("fusion admission initialized"))
    }
}

async fn acquire_admission(
    request: &AdmissionRequest,
    env: &NodeEnv,
    idx: u32,
    stage_name: &str,
) -> Result<AdmissionLease, NodeFailure> {
    let mut permits = Vec::with_capacity(request.resources.len());
    for &resource in &request.resources {
        let Some(sem) = env.resources.get(&resource) else {
            continue;
        };
        let permit = match sem.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                env.status.emit(StageEvent::StageBlocked {
                    node_idx: idx,
                    stage_name: stage_name.to_string(),
                    resource,
                });
                sem.clone().acquire_owned().await.map_err(|_| {
                    NodeFailure::Other(format!("resource '{resource}' semaphore closed"))
                })?
            }
        };
        permits.push(permit);
    }

    let gpu = if let Some(request) = request.gpu {
        let grant = match env.gpu.try_acquire(request) {
            Some(grant) => grant,
            None => {
                env.status.emit(StageEvent::StageBlocked {
                    node_idx: idx,
                    stage_name: stage_name.to_string(),
                    resource: Resource::Gpu,
                });
                env.gpu
                    .acquire(request)
                    .await
                    .map_err(|error| NodeFailure::Other(format!("GPU admission: {error}")))?
            }
        };
        Some(grant)
    } else {
        None
    };

    let memory = if request.memory_gib > 0 {
        Some(
            env.memory
                .clone()
                .acquire_many_owned(request.memory_gib)
                .await
                .map_err(|_| NodeFailure::Other("memory semaphore closed".into()))?,
        )
    } else {
        None
    };

    Ok(AdmissionLease {
        resources: permits,
        gpu,
        _memory: memory,
    })
}

async fn run_node(task: NodeTask, env: Arc<NodeEnv>) -> Result<NodeOutcome, NodeFailure> {
    run_node_with_admission(task, env, None, None, false, None).await
}

async fn run_node_with_admission(
    mut task: NodeTask,
    env: Arc<NodeEnv>,
    mut shared_admission: Option<&mut FusionAdmission>,
    pre_acquired_admission: Option<Arc<AdmissionLease>>,
    allow_in_process_handoff: bool,
    mut in_process_input: Option<InProcessArtifact>,
) -> Result<NodeOutcome, NodeFailure> {
    let idx = task.node_idx;
    let stage_name = task.stage.name().to_string();
    let stages_root = env.job_dir.join("stages");
    let final_stage_dir = stages_root.join(format!("{idx}-{stage_name}"));
    let tmp_stage_dir = stages_root.join(format!(".tmp-{idx}-{stage_name}-{}", task.key.to_hex()));

    // ── Cache lookup ────────────────────────────────────────────────
    // INC D (S4): `bypass_cache` forces a recompute — skip the READ so the
    // stage always runs even with a warm entry. The fresh result is still
    // inserted into the cache below the run path, so later runs hit again.
    let cache_hit: Option<Arc<CacheHit>> = if env.bypass_cache {
        None
    } else if let Some(hit) = task.prepared_cache_hit.take() {
        Some(hit)
    } else {
        cache_lookup_off_thread(
            env.cache.clone(),
            task.key,
            task.stage.clone(),
            final_stage_dir.clone(),
        )
        .await
        .map_err(|error| {
            NodeFailure::Other(format!(
                "cache lookup worker failed for {stage_name}: {error}"
            ))
        })?
        .map(Arc::new)
    };
    if let Some(hit) = cache_hit {
        let hit = Arc::try_unwrap(hit).unwrap_or_else(|shared| (*shared).clone());
        env.status.emit(StageEvent::StageSkipped {
            node_idx: idx,
            stage_name: stage_name.clone(),
            invocation_key: task.key,
            content_id: Some(hit.content_id),
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
        // A cache hit is still a materialization of THIS job. Mirror a small
        // sidecar + cache proof into its stage directory so lineage ingestion
        // and partition status do not produce artifact-less "done" jobs.
        let stage_dir = env
            .job_dir
            .join("stages")
            .join(format!("{idx}-{stage_name}"));
        if let Err(e) = std::fs::create_dir_all(&stage_dir) {
            tracing::warn!("cache-hit stage dir {}: {e}", stage_dir.display());
        } else {
            let metadata = ArtifactMetadata::new(
                hit.artifact.kind.clone(),
                hit.artifact.schema,
                hit.content_id.digest(),
            )
            .with_logical_hash(logical)
            .with_extra("persisted", serde_json::Value::Bool(true))
            .with_stage(stage_name.clone());
            if let Err(e) = metadata.write_to(&stage_dir.join("output.metadata.json")) {
                tracing::warn!("cache-hit sidecar {}: {e}", stage_dir.display());
            }
            let proof = crate::framework::cache::CacheProof {
                key: task.key,
                entry_path: env.cache.entry_path_for_write(task.key),
            };
            if let Err(e) = proof.write_to(&stage_dir.join("cache-proof.json")) {
                tracing::warn!("cache-hit proof {}: {e}", stage_dir.display());
            }
        }
        return Ok(NodeOutcome {
            node_id: task.node_id,
            output: hit.artifact,
            in_process_output: None,
            content_id: hit.content_id,
            logical,
            cache_hit: true,
        });
    }

    // ── Miss → run ──────────────────────────────────────────────────
    let training_io = env.training_io_node(task.node_id);
    if let Some(profile) = training_io.as_ref().map(|selected| &selected.profile) {
        env.status.emit(StageEvent::StageIoConfigured {
            node_idx: idx,
            stage_name: stage_name.clone(),
            profile: profile.clone(),
        });
    }
    env.status.emit(StageEvent::StageBegin {
        node_idx: idx,
        stage_name: stage_name.clone(),
        input_hash: task.input_hash,
        input_content_ids: task.input_content_ids.clone(),
    });

    // FW-2: the stage runs against a private `.tmp-<key>` dir; on Ok we
    // atomically rename it to the final name and ONLY THEN insert the
    // cache entry (the sole resume oracle). No promote ⇒ no cache ⇒
    // re-run. Tmp name is key-scoped so two positions of the same stage
    // (or a re-run with different args) never collide. The attempt loop
    // (D1) recreates the tmp dir + reacquires permits per attempt, so
    // FW-2 holds for EACH attempt; the cache insert is still strictly
    // post-promote (below the loop, on success).
    let fused_handoff = allow_in_process_handoff && task.stage.supports_in_process_handoff();
    let mut attempt = 0u32;
    let (stage_output, run_elapsed) = loop {
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
            tenant: env.tenant.clone(),
            recipe_name: env.recipe_name.clone(),
            launch_target: env.launch_target,
            device_index: env.device_index,
            gpu_devices: Vec::new(), // set from the GpuScheduler grant below
            fb_warm: training_io
                .as_ref()
                .map_or(env.fb_warm, |selected| selected.hints.cache_warm),
            admitted_workers: training_io
                .as_ref()
                .map_or(env.admitted_workers, |selected| {
                    selected.hints.admitted_decode_workers
                }),
            admitted_batch_size: training_io
                .as_ref()
                .map_or(env.admitted_batch_size, |selected| {
                    selected.hints.admitted_batch_size
                }),
            training_io_profile: training_io
                .as_ref()
                .map(|selected| selected.profile.clone()),
            pipeline_emitter: task.pipeline_emitter.clone(),
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

        // ── Resource/GPU/memory admission ───────────────────────────
        // Ordinary nodes acquire their own canonical envelope per attempt. A
        // fused linear chain supplies one pre-acquired identical lease instead,
        // so its stages cannot release/reacquire between boundaries. Both paths
        // use this same helper: fusion changes lease lifetime, never admission
        // policy or the StageContext device assignment.
        let mut owned_admission = None;
        let admission = if let Some(pre_acquired) = pre_acquired_admission.as_deref() {
            pre_acquired
        } else if let Some(shared) = shared_admission.as_deref_mut() {
            shared
                .ensure_acquired(&env, idx, &stage_name)
                .await
                .inspect_err(|_| {
                    let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                })?
        } else {
            owned_admission = Some(
                acquire_admission(
                    &AdmissionRequest::for_task(&task, &env).map_err(NodeFailure::Other)?,
                    &env,
                    idx,
                    &stage_name,
                )
                .await
                .inspect_err(|_| {
                    let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                })?,
            );
            owned_admission
                .as_ref()
                .expect("ordinary admission initialized")
        };
        if task.stage.resources().contains(&Resource::Gpu)
            && let Some(grant) = &admission.gpu
        {
            // Mirror `GpuScheduler::effective_need`: a degenerate GPU stage
            // declaring count=0 is still admitted as one device, exactly like
            // the pre-refactor path that exposed the scheduler's whole grant.
            let requested = task.stage.gpu_request(&task.args).count.max(1) as usize;
            stage_ctx.gpu_devices = grant.devices.iter().copied().take(requested).collect();
            stage_ctx.device_index = stage_ctx
                .gpu_devices
                .first()
                .copied()
                .or(stage_ctx.device_index);
        }

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

        let run_fut = async {
            if fused_handoff {
                // The typed predecessor value is single-owner and is consumed
                // by the first attempt. A retry intentionally decodes the
                // canonical erased input: the prior attempt may have consumed
                // or mutated its typed value before failing.
                task.stage
                    .run_in_process(
                        &stage_ctx,
                        task.input.clone(),
                        in_process_input.take(),
                        task.args.clone(),
                    )
                    .await
                    .map(|(erased, in_process)| StageRunOutput {
                        erased,
                        in_process: Some(in_process),
                    })
            } else {
                task.stage
                    .run_erased(&stage_ctx, task.input.clone(), task.args.clone())
                    .await
                    .map(|erased| StageRunOutput {
                        erased,
                        in_process: None,
                    })
            }
        };
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
                if let Some(owned) = owned_admission.as_mut() {
                    owned.release_non_gpu_resources();
                }
                // Match every other error exit from this attempt (see the
                // sibling `let _ = std::fs::remove_dir_all(&tmp_stage_dir)`
                // calls above/below): a caught panic must not skip cleanup
                // of this attempt's tmp dir either, or it lingers on disk
                // until process exit.
                let _ = std::fs::remove_dir_all(&tmp_stage_dir);
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
        // Preserve the historical ordinary-node boundary: CPU/network/disk
        // permits are available to other work before divergence classification
        // and the synchronous retry hook. GPU and memory stay scoped to this
        // attempt as before. A shared fusion lease is only borrowed here and
        // remains held by the fused-plan driver across every stage in the chain.
        if let Some(owned) = owned_admission.as_mut() {
            owned.release_non_gpu_resources();
        }
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
                    o.erased.kind,
                    task.stage.output_kind(),
                    "stage '{stage_name}' produced kind '{}' but declares output_kind '{}'",
                    o.erased.kind,
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

    // Re-point tmp-rooted absolute paths in the output handle at the promoted
    // final dir. A fused stage finishes this from the just-produced typed value
    // (no bincode decode at the inter-stage boundary); if that internal seam
    // declines, fall back to the established erased path.
    let StageRunOutput { erased, in_process } = stage_output;
    let (output, in_process_output, known_output_hash) = match in_process.and_then(|typed| {
        task.stage
            .promote_in_process_output(typed, &tmp_stage_dir, &final_stage_dir)
    }) {
        Some((output, typed, output_hash)) => (output, Some(typed), Some(output_hash)),
        None => (
            task.stage
                .rebase_output_paths(erased, &tmp_stage_dir, &final_stage_dir),
            None,
            None,
        ),
    };

    // Preserve the existing logical hash for downstream invocation keys, but
    // derive lineage/cache identity from the canonical persisted bytes. Capture
    // happens before the sidecar write so metadata cannot hash itself.
    let logical_hash = known_output_hash.unwrap_or_else(|| {
        task.stage
            .output_content_hash(&output)
            .unwrap_or_else(|| content_hash_from_erased(&output))
    });
    let (content_id, stored) = match capture(
        task.stage.as_ref(),
        output.clone(),
        &final_stage_dir,
        ArtifactRole::Output,
        None,
    ) {
        Ok(stored) => (stored.manifest.content_id, Some(stored)),
        // A declared external reference is an expected ownership policy outcome,
        // not a failed store write. `capture` has only read the promoted output.
        Err(error @ ArtifactStoreError::NonPortable(_)) => {
            tracing::warn!(
                "executor: output for stage '{stage_name}' is not portable and will not be cached: {error}"
            );
            (
                unpersisted_content_id(
                    task.stage.as_ref(),
                    &output,
                    ArtifactRole::Output,
                    logical_hash,
                ),
                None,
            )
        }
        Err(error) => {
            let source = std::io::Error::new(std::io::ErrorKind::InvalidData, error);
            let _ = std::fs::remove_dir_all(&final_stage_dir);
            env.status.emit(StageEvent::StageFailed {
                node_idx: idx,
                stage_name: stage_name.clone(),
                error: format!("capture canonical output: {source}"),
                failure: None,
            });
            return Err(NodeFailure::Stage {
                idx,
                stage: stage_name,
                source: StageError::Io {
                    path: final_stage_dir,
                    source,
                },
            });
        }
    };
    if let Some(stored) = &stored {
        debug_assert_eq!(stored.manifest.logical_hash, logical_hash);
    }
    let metadata = ArtifactMetadata::new(output.kind.clone(), output.schema, content_id.digest())
        .with_logical_hash(logical_hash)
        .with_extra("persisted", serde_json::Value::Bool(stored.is_some()))
        .with_stage(stage_name.clone());
    if let Err(e) = metadata.write_to(&final_stage_dir.join("output.metadata.json")) {
        tracing::warn!(
            "executor: sidecar write for stage '{stage_name}' failed: {e}; lineage tooling will not see this artifact"
        );
    }

    // Cache insert — STRICTLY after the atomic promote (the load-bearing
    // FW-2 ordering: the resume oracle appears only once the output is
    // fully in place).
    if let Some(stored) = &stored {
        match env.cache.insert_stored(task.key, stored) {
            Ok(stored_id) => {
                debug_assert_eq!(stored_id, content_id);
                let proof = crate::framework::cache::CacheProof {
                    key: task.key,
                    entry_path: env.cache.entry_path_for_write(task.key),
                };
                if let Err(e) = proof.write_to(&final_stage_dir.join("cache-proof.json")) {
                    tracing::warn!("executor: cache proof for stage '{stage_name}' failed: {e}");
                }
            }
            Err(e) => {
                tracing::warn!(
                    "executor: cache insert for stage '{stage_name}' failed: {e}; continuing"
                );
            }
        }
    }

    env.status.emit(StageEvent::StageEnd {
        node_idx: idx,
        stage_name: stage_name.clone(),
        content_id,
        elapsed: run_elapsed,
    });

    let logical = if task.stage.deterministic() {
        logical_hash
    } else {
        compute_logical_output_hash(
            task.stage.as_ref(),
            &output,
            task.stage.deterministic(),
            &stage_name,
            task.stage.schema(),
            task.input_hash,
            &task.canon_args,
        )
    };
    Ok(NodeOutcome {
        node_id: task.node_id,
        output,
        in_process_output,
        content_id,
        logical,
        cache_hit: false,
    })
}

/// Execute one optimizer-authorized target against a job-private scratch
/// environment. The canonical status hub, cache, stage directory, retry hook,
/// and lineage surfaces are intentionally absent. The caller owns the already
/// acquired all-or-none admission lease and decides whether to publish or drop
/// the returned value after its condition resolves.
async fn prepare_speculative(
    task: NodeTask,
    canonical_env: Arc<NodeEnv>,
    cancel: CancellationToken,
    lease: AdmissionLease,
    scratch_root: PathBuf,
) -> Result<SpeculativePrepared, NodeFailure> {
    prepare_private(
        task,
        canonical_env,
        cancel,
        Arc::new(lease),
        scratch_root,
        None,
    )
    .await
}

async fn prepare_private(
    mut task: NodeTask,
    canonical_env: Arc<NodeEnv>,
    cancel: CancellationToken,
    pre_acquired_admission: Arc<AdmissionLease>,
    scratch_root: PathBuf,
    training_io_override: Option<ResolvedTrainingIoNode>,
) -> Result<SpeculativePrepared, NodeFailure> {
    let node_id = task.node_id;
    let node_idx = task.node_idx;
    let stage = task.stage.clone();
    let stage_name = stage.name().to_string();
    let input_hash = task.input_hash;
    let input_content_ids = task.input_content_ids.clone();
    let canon_args = task.canon_args.clone();
    let key = task.key;
    let training_io = training_io_override.or_else(|| canonical_env.training_io_node(node_id));
    let training_io_profile = training_io
        .as_ref()
        .map(|selected| selected.profile.clone());
    let scratch_stage_dir = scratch_root
        .join("stages")
        .join(format!("{node_idx}-{stage_name}"));

    // Arm cleanup before the first filesystem mutation so a partial mkdir also
    // has an owner if setup fails.
    let scratch = SpeculationScratch(scratch_root.clone());
    if let Err(source) = scratch.reset() {
        if let Err(cleanup_source) = scratch.cleanup() {
            return Err(NodeFailure::SpeculationCleanup {
                path: scratch_root,
                source: cleanup_source,
            });
        }
        return Err(NodeFailure::Stage {
            idx: node_idx,
            stage: stage_name.clone(),
            source: StageError::Io {
                path: scratch_root,
                source,
            },
        });
    }

    // One private attempt only. If it fails, normal selection owns the stage's
    // configured retries and retry hook.
    task.retry = crate::framework::retry::RetryPolicy::NONE;
    task.prepared_cache_hit = None;

    let (status, mut lifecycle_rx) = StatusHub::new();
    let mut live_rx = status.subscribe();
    let mut training_io_profiles = canonical_env.training_io_snapshot();
    if let Some(selected) = training_io.clone() {
        training_io_profiles.insert(node_id, selected);
    }
    let private_env = Arc::new(NodeEnv {
        job_dir: scratch_root.clone(),
        cache: Arc::new(CacheHandle::job_local(scratch_root.join("_cache"))),
        tenant: canonical_env.tenant.clone(),
        status,
        cancel,
        resources: canonical_env.resources.clone(),
        gpu: canonical_env.gpu.clone(),
        memory: canonical_env.memory.clone(),
        memory_budget_gib: canonical_env.memory_budget_gib,
        launch_target: crate::config::launcher::LaunchTarget::Local,
        device_index: canonical_env.device_index,
        fb_warm: canonical_env.fb_warm,
        admitted_workers: canonical_env.admitted_workers,
        admitted_batch_size: canonical_env.admitted_batch_size,
        training_io_profiles: std::sync::RwLock::new(training_io_profiles),
        training_io_resolver: canonical_env.training_io_resolver.clone(),
        bypass_cache: true,
        recipe_name: canonical_env.recipe_name.clone(),
        on_retry: None,
        diverged: Arc::new(std::sync::Mutex::new(HashMap::new())),
        #[cfg(feature = "p2p")]
        dispatch_policy: None,
        #[cfg(feature = "p2p")]
        dispatcher: None,
    });
    let started = Instant::now();
    // Keep the explicit scratch owner outside the unwind boundary. If plugin
    // code panics, cleanup must still be checked while that owner is alive;
    // relying on Drop here would turn an undeletable private tree into a log
    // while the ordinary plan continued successfully.
    let run_result = std::panic::AssertUnwindSafe(run_node_with_admission(
        task,
        private_env,
        None,
        Some(pre_acquired_admission),
        false,
        None,
    ))
    .catch_unwind()
    .await;
    let outcome = match run_result {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(failure)) => {
            if let Err(source) = scratch.cleanup() {
                return Err(NodeFailure::SpeculationCleanup {
                    path: scratch_root,
                    source,
                });
            }
            return Err(failure);
        }
        Err(_) => {
            if let Err(source) = scratch.cleanup() {
                return Err(NodeFailure::SpeculationCleanup {
                    path: scratch_root,
                    source,
                });
            }
            return Err(NodeFailure::Other(format!(
                "speculative node task panicked for target {node_id}"
            )));
        }
    };

    let mut elapsed = started.elapsed();
    while let Ok(event) = lifecycle_rx.try_recv() {
        if let StageEvent::StageEnd {
            elapsed: stage_elapsed,
            ..
        } = event
        {
            elapsed = stage_elapsed;
        }
    }
    let pre_step_lifecycle_count = 1 + usize::from(training_io_profile.is_some());
    let buffered_steps = drain_speculative_steps(&mut live_rx, pre_step_lifecycle_count);

    Ok(SpeculativePrepared {
        _scratch: scratch,
        scratch_stage_dir,
        node_id,
        node_idx,
        stage,
        stage_name,
        input_hash,
        input_content_ids,
        canon_args,
        key,
        output: outcome.output,
        elapsed,
        training_io_profile,
        buffered_steps,
    })
}

fn discard_pipeline_prepared(
    prepared: &mut Vec<SpilledPipelinePrepared>,
) -> Result<(), NodeFailure> {
    let mut first_failure = None;
    for child in prepared.drain(..) {
        if let Err(failure) = child.discard()
            && first_failure.is_none()
        {
            first_failure = Some(failure);
        }
    }
    first_failure.map_or(Ok(()), Err)
}

/// Only corruption/read failures may conservatively fall back to ordinary
/// execution. An undeletable private subtree is an integrity failure and must
/// remain visible even though the pipeline optimization itself is optional.
fn allow_pipeline_load_fallback(failure: NodeFailure) -> Result<(), NodeFailure> {
    match failure {
        failure @ NodeFailure::SpeculationCleanup { .. } => Err(failure),
        _ => Ok(()),
    }
}

/// Run one manifest-certified map producer while consuming its semantic item
/// lane with one private worker. The combined parent + one-child admission is
/// already held, so a capacity-one channel cannot deadlock behind a resource
/// the producer itself owns. Nothing from a child becomes canonical until the
/// producer's final `ListOf` validates the entire manifest and emitted order.
async fn run_pipeline_parent(
    task: NodeTask,
    env: Arc<NodeEnv>,
    mut launch: PipelineLaunch,
    deadline: Option<Instant>,
    plan_started: Instant,
) -> Result<PipelineRunResult, NodeFailure> {
    let parent_id = task.node_id;
    let parent_cancel = task.node_cancel.current();
    let admission = launch.admission.clone();
    let env_for_parent = env.clone();
    let mut parent_future = Box::pin(run_node_with_admission(
        task,
        env_for_parent,
        None,
        Some(admission),
        false,
        None,
    ));
    let mut parent_result: Option<Result<NodeOutcome, NodeFailure>> = None;
    let mut prepared = Vec::with_capacity(launch.children.len());
    let mut emitted_count = 0usize;
    let mut valid = true;
    let mut fatal_failure = None;

    loop {
        let emission = if parent_result.is_some() {
            launch.receiver.recv().await
        } else {
            tokio::select! {
                biased;
                emission = launch.receiver.recv() => emission,
                result = &mut parent_future => {
                    parent_result = Some(result);
                    continue;
                }
            }
        };
        let Some(emission) = emission else {
            break;
        };
        let index = emitted_count;
        emitted_count += 1;
        if fatal_failure.is_some() || !valid || parent_result.as_ref().is_some_and(Result::is_err) {
            drop(emission);
            continue;
        }
        let expected_hash = launch.manifest.element_hashes.get(index).copied();
        if emission.index != index
            || expected_hash != Some(emission.content_hash)
            || index >= launch.children.len()
        {
            valid = false;
            if let Err(failure) = discard_pipeline_prepared(&mut prepared) {
                fatal_failure = Some(failure);
                env.cancel.cancel();
                parent_cancel.cancel();
            }
            drop(emission);
            continue;
        }
        let child = &launch.children[index];
        if let Some(error) = plan_stop_error(deadline, plan_started, &env.cancel) {
            discard_pipeline_prepared(&mut prepared)?;
            return Err(NodeFailure::Plan(error));
        }
        let cache_cold = env.bypass_cache
            || matches!(
                cache_presence_probe_off_thread(env.cache.clone(), child.key).await,
                Ok(Ok(false))
            );
        if let Some(error) = plan_stop_error(deadline, plan_started, &env.cancel) {
            discard_pipeline_prepared(&mut prepared)?;
            return Err(NodeFailure::Plan(error));
        }
        if !cache_cold {
            valid = false;
            if let Err(failure) = discard_pipeline_prepared(&mut prepared) {
                fatal_failure = Some(failure);
                env.cancel.cancel();
                parent_cancel.cancel();
            }
            drop(emission);
            continue;
        }
        let child_cancel = env.cancel.child_token();
        let retry = child.node.retry.unwrap_or_else(|| child.node.stage.retry());
        let timeout = child
            .node
            .timeout
            .unwrap_or_else(|| child.node.stage.timeout());
        let input_content_id = unpersisted_content_id(
            child.node.stage.as_ref(),
            &emission.artifact,
            ArtifactRole::Input,
            child.input_logical,
        );
        let child_task = NodeTask {
            node_id: child.node.id,
            node_idx: child.node_idx,
            stage: child.node.stage.clone(),
            args: child.node.args.clone(),
            canon_args: child.node.canon_args.clone(),
            input: emission.artifact.clone(),
            input_content_ids: vec![input_content_id],
            input_hash: child.input_logical,
            key: child.key,
            prepared_cache_hit: None,
            retry,
            timeout,
            node_cancel: KillSlot::new(child_cancel.clone()),
            pipeline_emitter: None,
        };
        let scratch_root = env.job_dir.join(".pipeline").join(format!(
            "{parent_id}-{index}-{}",
            SPECULATION_NONCE.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        let scratch_cleanup_root = scratch_root.clone();
        let child_training_io = launch
            .profiles
            .iter()
            .find(|(node_id, _)| *node_id == child.node.id)
            .map(|(_, selected)| selected.clone());
        let mut child_future = Box::pin(prepare_private(
            child_task,
            env.clone(),
            child_cancel.clone(),
            launch.admission.clone(),
            scratch_root,
            child_training_io,
        ));
        let child_result = loop {
            if parent_result.as_ref().is_some_and(Result::is_err) {
                child_cancel.cancel();
                drop(child_future);
                break match SpeculationScratch::cleanup_path(&scratch_cleanup_root) {
                    Ok(()) => Err(NodeFailure::Cancelled),
                    Err(source) => Err(NodeFailure::SpeculationCleanup {
                        path: scratch_cleanup_root,
                        source,
                    }),
                };
            }
            if parent_result.is_some() {
                break child_future.await;
            }
            tokio::select! {
                result = &mut child_future => break result,
                result = &mut parent_future => {
                    if result.is_err() {
                        child_cancel.cancel();
                    }
                    parent_result = Some(result);
                }
                _ = env.cancel.cancelled() => {
                    child_cancel.cancel();
                    drop(child_future);
                    SpeculationScratch::cleanup_path(&scratch_cleanup_root).map_err(|source| {
                        NodeFailure::SpeculationCleanup {
                            path: scratch_cleanup_root.clone(),
                            source,
                        }
                    })?;
                    discard_pipeline_prepared(&mut prepared)?;
                    return Err(NodeFailure::Plan(
                        plan_stop_error(deadline, plan_started, &env.cancel)
                            .unwrap_or(PlanError::Cancelled),
                    ));
                }
            }
        };
        match child_result {
            Ok(child_prepared) => {
                match SpilledPipelinePrepared::spill(
                    child_prepared,
                    &emission.artifact,
                    launch.max_item_bytes,
                ) {
                    Ok(spilled) => prepared.push(spilled),
                    Err(failure @ NodeFailure::SpeculationCleanup { .. }) => {
                        fatal_failure = Some(failure);
                        env.cancel.cancel();
                        parent_cancel.cancel();
                    }
                    Err(_) => {
                        valid = false;
                        if let Err(failure) = discard_pipeline_prepared(&mut prepared) {
                            fatal_failure = Some(failure);
                            env.cancel.cancel();
                            parent_cancel.cancel();
                        }
                    }
                }
            }
            Err(failure @ NodeFailure::SpeculationCleanup { .. }) => {
                fatal_failure = Some(failure);
                env.cancel.cancel();
                parent_cancel.cancel();
            }
            Err(_) => {
                valid = false;
                if let Err(failure) = discard_pipeline_prepared(&mut prepared) {
                    fatal_failure = Some(failure);
                    env.cancel.cancel();
                    parent_cancel.cancel();
                }
            }
        }
        drop(emission);
    }

    let parent_result = match parent_result {
        Some(result) => result,
        None => parent_future.await,
    };
    if let Some(failure) = fatal_failure {
        discard_pipeline_prepared(&mut prepared)?;
        return Err(failure);
    }
    let parent = match parent_result {
        Ok(parent) => parent,
        Err(failure) => {
            discard_pipeline_prepared(&mut prepared)?;
            return Err(failure);
        }
    };
    let elements = match crate::framework::artifact::decode_list_children(parent.output.clone()) {
        Ok(elements) => elements,
        Err(error) => {
            discard_pipeline_prepared(&mut prepared)?;
            return Err(NodeFailure::Other(format!(
                "map over node {parent_id}: parent output is not a valid list: {error}"
            )));
        }
    };
    valid &= !parent.cache_hit
        && parent.logical == launch.predicted_parent_logical
        && emitted_count == launch.manifest.element_hashes.len()
        && elements.len() == launch.manifest.element_hashes.len()
        && prepared.len() == elements.len();
    if valid {
        for (child, expected) in prepared.iter().zip(&elements) {
            match child.input_matches(
                expected,
                launch.producer_stage.as_ref(),
                &launch.producer_tmp_stage_dir,
                &launch.producer_final_stage_dir,
            ) {
                Ok(true) => {}
                Ok(false) | Err(NodeFailure::Other(_)) => {
                    valid = false;
                    break;
                }
                Err(failure) => {
                    discard_pipeline_prepared(&mut prepared)?;
                    return Err(failure);
                }
            }
        }
    }
    if !valid {
        discard_pipeline_prepared(&mut prepared)?;
    }
    Ok(PipelineRunResult {
        _admission: launch.admission,
        parent,
        elements,
        children: launch.children,
        profiles: launch.profiles,
        prepared: valid.then_some(prepared),
        spawn_capacity: launch.spawn_capacity,
    })
}

/// Drain the private lossy status stream without turning a lag notification
/// into silent truncation. The replay reserves canonical broadcast slots for
/// Begin, one exact Gap marker, and End so this synchronous burst cannot itself
/// overwrite retained steps before the status writer gets polled.
fn drain_speculative_steps(
    rx: &mut broadcast::Receiver<StageEvent>,
    pre_step_lifecycle_count: usize,
) -> Vec<StageEvent> {
    let mut steps = VecDeque::new();
    let mut dropped_steps = 0u64;
    let mut remaining_pre_step_lifecycle = pre_step_lifecycle_count as u64;
    loop {
        match rx.try_recv() {
            Ok(event @ StageEvent::StageStep { .. }) => steps.push_back(event),
            Ok(_) => {
                remaining_pre_step_lifecycle = remaining_pre_step_lifecycle.saturating_sub(1);
            }
            Err(broadcast::error::TryRecvError::Lagged(dropped)) => {
                let lifecycle = dropped.min(remaining_pre_step_lifecycle);
                remaining_pre_step_lifecycle -= lifecycle;
                dropped_steps = dropped_steps.saturating_add(dropped - lifecycle);
            }
            Err(broadcast::error::TryRecvError::Empty | broadcast::error::TryRecvError::Closed) => {
                break;
            }
        }
    }

    // Canonical replay emits every pre-step lifecycle record plus StageEnd.
    let lifecycle_replay_count = pre_step_lifecycle_count.saturating_add(1);
    let max_without_gap = DEFAULT_BROADCAST_CAPACITY.saturating_sub(lifecycle_replay_count);
    let needs_gap = dropped_steps > 0 || steps.len() > max_without_gap;
    let max_steps = if needs_gap {
        DEFAULT_BROADCAST_CAPACITY.saturating_sub(lifecycle_replay_count.saturating_add(1))
    } else {
        max_without_gap
    };
    while steps.len() > max_steps {
        steps.pop_front();
        dropped_steps = dropped_steps.saturating_add(1);
    }

    let mut buffered = Vec::with_capacity(steps.len() + usize::from(dropped_steps > 0));
    if dropped_steps > 0 {
        buffered.push(StageEvent::StepGap {
            dropped: dropped_steps,
        });
    }
    buffered.extend(steps);
    buffered
}

fn discard_speculation_result(
    result: Result<Box<SpeculativePrepared>, NodeFailure>,
) -> Result<(), NodeFailure> {
    match result {
        Ok(prepared) => prepared.discard(),
        Err(failure @ NodeFailure::SpeculationCleanup { .. }) => Err(failure),
        Err(_) => Ok(()),
    }
}

/// Explicitly close every retained private result before any terminal return.
/// Drop remains a last-resort diagnostic guard, but a plan outcome may not hide
/// undeletable speculative residue behind best-effort destruction.
fn discard_retained_speculation(
    speculation: &mut HashMap<NodeId, SpeculationState>,
    speculative_keys: &mut HashMap<InvocationKey, NodeId>,
) -> Result<(), NodeFailure> {
    let mut first_failure = None;
    for (_, state) in std::mem::take(speculation) {
        match state {
            SpeculationState::Prepared(prepared) => {
                if let Err(failure) = prepared.discard()
                    && first_failure.is_none()
                {
                    first_failure = Some(failure);
                }
            }
            SpeculationState::Running { cancel, .. } => cancel.cancel(),
            SpeculationState::Eligible
            | SpeculationState::Declined
            | SpeculationState::Rejected => {}
        }
    }
    speculative_keys.clear();
    first_failure.map_or(Ok(()), Err)
}

/// Commit a selected private result through the same canonical FW-2 ordering as
/// an ordinary miss: begin, same-filesystem rename, path rebase, sidecar, cache
/// proof, buffered steps, end, then scheduler visibility.
fn publish_speculative(
    prepared: SpeculativePrepared,
    env: &NodeEnv,
    deadline: Option<Instant>,
    plan_started: Instant,
    rollback_cache_on_stop: bool,
) -> Result<NodeOutcome, NodeFailure> {
    if let Some(error) = plan_stop_error(deadline, plan_started, &env.cancel) {
        return Err(NodeFailure::Plan(error));
    }
    let idx = prepared.node_idx;
    let stage_name = prepared.stage_name.clone();
    let final_stage_dir = env
        .job_dir
        .join("stages")
        .join(format!("{idx}-{stage_name}"));
    let rollback_failure = Arc::new(std::sync::Mutex::new(None));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        publish_speculative_inner(
            prepared,
            env,
            deadline,
            plan_started,
            rollback_cache_on_stop,
            rollback_failure.clone(),
        )
    }));
    let rollback_failure = rollback_failure
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    if let Some(source) = rollback_failure {
        let cause = match &caught {
            Err(_) => format!("speculative publication panicked for node {idx} ({stage_name})"),
            Ok(Err(_)) => {
                format!("speculative publication failed for node {idx} ({stage_name})")
            }
            Ok(Ok(_)) => {
                format!("speculative publication rollback fired for node {idx} ({stage_name})")
            }
        };
        return Err(NodeFailure::SpeculationRollback {
            path: final_stage_dir,
            source,
            cause,
        });
    }
    match caught {
        Ok(result) => result,
        Err(_) => Err(NodeFailure::Other(format!(
            "speculative publication panicked for node {idx} ({stage_name})"
        ))),
    }
}

fn publish_speculative_inner(
    prepared: SpeculativePrepared,
    env: &NodeEnv,
    deadline: Option<Instant>,
    plan_started: Instant,
    rollback_cache_on_stop: bool,
    rollback_failure: Arc<std::sync::Mutex<Option<std::io::Error>>>,
) -> Result<NodeOutcome, NodeFailure> {
    let idx = prepared.node_idx;
    let stage_name = prepared.stage_name.clone();
    let stages_root = env.job_dir.join("stages");
    let final_stage_dir = stages_root.join(format!("{idx}-{stage_name}"));
    if let Err(source) = std::fs::create_dir_all(&stages_root) {
        return Err(NodeFailure::Stage {
            idx,
            stage: stage_name,
            source: StageError::Io {
                path: stages_root,
                source,
            },
        });
    }
    let _ = std::fs::remove_dir_all(&final_stage_dir);
    if let Err(source) = std::fs::rename(&prepared.scratch_stage_dir, &final_stage_dir) {
        return Err(NodeFailure::Stage {
            idx,
            stage: stage_name,
            source: StageError::Io {
                path: final_stage_dir,
                source,
            },
        });
    }
    let publish_guard = SpeculativePublishGuard {
        final_stage_dir: final_stage_dir.clone(),
        committed: false,
        rollback_failure: rollback_failure.clone(),
    };
    // The private run wrote a proof for its disposable scratch cache. Never let
    // that path survive selection if the canonical insert below degrades.
    let _ = std::fs::remove_file(final_stage_dir.join("cache-proof.json"));

    let output = prepared.stage.rebase_output_paths(
        prepared.output.clone(),
        &prepared.scratch_stage_dir,
        &final_stage_dir,
    );
    let output_hash = prepared
        .stage
        .output_content_hash(&output)
        .unwrap_or_else(|| content_hash_from_erased(&output));
    let logical = compute_logical_output_hash(
        prepared.stage.as_ref(),
        &output,
        prepared.stage.deterministic(),
        &prepared.stage_name,
        prepared.stage.schema(),
        prepared.input_hash,
        &prepared.canon_args,
    );
    let (content_id, stored) = match capture(
        prepared.stage.as_ref(),
        output.clone(),
        &final_stage_dir,
        ArtifactRole::Output,
        None,
    ) {
        Ok(stored) => (stored.manifest.content_id, Some(stored)),
        Err(error @ ArtifactStoreError::NonPortable(_)) => {
            tracing::warn!(
                "executor: speculative output for stage '{stage_name}' is not portable and will not be cached: {error}"
            );
            (
                unpersisted_content_id(
                    prepared.stage.as_ref(),
                    &output,
                    ArtifactRole::Output,
                    output_hash,
                ),
                None,
            )
        }
        Err(error) => {
            return Err(NodeFailure::Stage {
                idx,
                stage: stage_name.clone(),
                source: StageError::Io {
                    path: final_stage_dir.clone(),
                    source: std::io::Error::new(std::io::ErrorKind::InvalidData, error),
                },
            });
        }
    };
    if let Some(stored) = &stored {
        debug_assert_eq!(stored.manifest.logical_hash, output_hash);
    }
    if let Err(source) = prepared._scratch.cleanup() {
        return Err(NodeFailure::SpeculationCleanup {
            path: prepared._scratch.0.clone(),
            source,
        });
    }
    // Path rebasing and content hashing are cookbook/plugin code and can be
    // arbitrarily expensive. If a deadline or external cancellation arrived
    // while they ran, the still-armed publication guard removes the renamed
    // directory before any cache entry or lifecycle event becomes visible.
    if let Some(error) = plan_stop_error(deadline, plan_started, &env.cancel) {
        return Err(NodeFailure::Plan(error));
    }
    let metadata = ArtifactMetadata::new(output.kind.clone(), output.schema, content_id.digest())
        .with_logical_hash(output_hash)
        .with_extra("persisted", serde_json::Value::Bool(stored.is_some()))
        .with_stage(prepared.stage_name.clone());
    if let Err(error) = metadata.write_to(&final_stage_dir.join("output.metadata.json")) {
        tracing::warn!(
            "executor: speculative sidecar write for stage '{}' failed: {error}",
            prepared.stage_name
        );
    }
    // Metadata remains inside the guarded stage directory. Recheck immediately
    // before the first separately-visible cache write, then keep a second guard
    // for the pipeline lane until lifecycle publication commits.
    if let Some(error) = plan_stop_error(deadline, plan_started, &env.cancel) {
        return Err(NodeFailure::Plan(error));
    }
    let mut pipeline_cache_guard = None;
    let mut optional_cache_body = None;
    if let Some(stored) = &stored {
        match env.cache.insert_optional_local(prepared.key, stored) {
            Ok(body) => {
                optional_cache_body = Some(body);
                if rollback_cache_on_stop {
                    pipeline_cache_guard = Some(PipelineCachePublishGuard::new(
                        env.cache.entry_path_for_write(prepared.key),
                        rollback_failure,
                    ));
                }
                if rollback_cache_on_stop
                    && let Some(error) = plan_stop_error(deadline, plan_started, &env.cancel)
                {
                    return Err(NodeFailure::Plan(error));
                }
                let proof = crate::framework::cache::CacheProof {
                    key: prepared.key,
                    entry_path: env.cache.entry_path_for_write(prepared.key),
                };
                if let Err(error) = proof.write_to(&final_stage_dir.join("cache-proof.json")) {
                    tracing::warn!(
                        "executor: speculative cache proof for stage '{}' failed: {error}",
                        prepared.stage_name
                    );
                }
            }
            Err(error) => tracing::warn!(
                "executor: speculative cache insert for stage '{}' failed: {error}; continuing",
                prepared.stage_name
            ),
        }
    }
    // Pipeline keys are cold, private reservations, so their cache write stays
    // rollback-capable until lifecycle publication. Ordinary speculation can
    // race another process or shared writer for the same cache path; once its
    // local insert succeeds, that insert is its publication linearization point
    // and cancellation loses rather than leaving a cache entry whose guarded
    // stage directory is removed. If the insert failed, no separate cache state
    // exists and the ordinary lane may still stop safely here.
    if (rollback_cache_on_stop || optional_cache_body.is_none())
        && let Some(error) = plan_stop_error(deadline, plan_started, &env.cancel)
    {
        return Err(NodeFailure::Plan(error));
    }
    // Publish lifecycle only after every stage/plugin hook and canonical
    // filesystem/cache operation that can unwind has completed. The rollback
    // guard removes the renamed directory on any earlier panic/error, so a
    // failed publication cannot strand an orphan StageBegin.
    if let Some(profile) = prepared.training_io_profile.clone() {
        env.status.emit(StageEvent::StageIoConfigured {
            node_idx: idx,
            stage_name: stage_name.clone(),
            profile,
        });
    }
    env.status.emit(StageEvent::StageBegin {
        node_idx: idx,
        stage_name: stage_name.clone(),
        input_hash: prepared.input_hash,
        input_content_ids: prepared.input_content_ids.clone(),
    });
    for event in prepared.buffered_steps.iter().cloned() {
        env.status.emit(event);
    }
    env.status.emit(StageEvent::StageEnd {
        node_idx: idx,
        stage_name: prepared.stage_name.clone(),
        content_id,
        elapsed: prepared.elapsed,
    });
    if let Some(cache_guard) = pipeline_cache_guard {
        cache_guard.commit();
    }
    publish_guard.commit();
    // Remote object-store replication is optional acceleration, not part of
    // the local canonical commit. Run it only after the stage directory, cache
    // entry, and lifecycle batch are committed; plugin panics stay contained.
    if let Some(write) = optional_cache_body {
        env.cache.replicate_optional(&write);
    }
    Ok(NodeOutcome {
        node_id: prepared.node_id,
        output,
        in_process_output: None,
        content_id,
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
async fn run_with_timeout<T>(
    run_fut: impl std::future::Future<Output = Result<T, StageError>>,
    stage_cancel: &CancellationToken,
    soft: Option<std::time::Duration>,
    hard: Option<std::time::Duration>,
    started: Instant,
) -> Result<T, StageError> {
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

/// Return the plan-level stop reason that forbids another node launch.
/// Deadline dominates cancellation when both become observable together,
/// matching the coordinator's outer-loop and cache-probe `select!` ordering.
fn plan_stop_error(
    deadline: Option<Instant>,
    started: Instant,
    cancel: &CancellationToken,
) -> Option<PlanError> {
    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        Some(PlanError::DeadlineExceeded {
            elapsed: started.elapsed(),
        })
    } else if cancel.is_cancelled() {
        Some(PlanError::Cancelled)
    } else {
        None
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
    content_outputs: &HashMap<NodeId, ContentId>,
    node_cancel: KillSlot,
) -> Result<NodeTask, PlanError> {
    let preds = predecessors(edges, node.id);
    let input = gather_input(node.id, &preds, outputs)?;
    let input_hash = gather_input_hash(node.id, &preds, logical_outputs)?;
    let input_content_ids = preds
        .iter()
        .filter_map(|predecessor| content_outputs.get(predecessor).copied())
        .collect();
    let key = node_cache_key(node, input_hash);
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
        input_content_ids,
        input_hash,
        key,
        prepared_cache_hit: None,
        retry,
        timeout,
        node_cancel,
        pipeline_emitter: None,
    })
}

/// Exact ADR-0078/0101 cache key for one node after predecessor hashes resolve.
/// Scheduling probes and `build_task` share this function, so cache-aware order
/// cannot drift from the key `run_node` later reads and writes.
fn node_cache_key(
    node: &crate::framework::plan::PlanNode,
    input_hash: ContentHash,
) -> InvocationKey {
    let code_sha = node_code_sha(node.stage.as_ref());
    CacheHandle::key_for_canon_bytes_partitioned(
        node.stage.name(),
        node.stage.schema(),
        input_hash,
        &node.canon_args,
        &code_sha,
        node.partition.as_ref(),
    )
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

/// Resolve real cache warmth for every currently-ready node. Ready means every
/// predecessor logical hash exists, so the exact input-dependent key is known.
/// [`CacheHandle::lookup`] requires a live, decodable entry and matches the read
/// path `run_node` uses. Exact-key results are memoized for the run and a miss is
/// invalidated only when that key completes, avoiding quadratic
/// remote/filesystem probes on wide DAGs. Blocking I/O runs off the coordinator.
/// Force-recompute makes every node cold without touching the cache.
const MAX_CACHE_PROBE_CONCURRENCY: usize = 8;

#[allow(clippy::too_many_arguments)]
async fn refresh_cache_warm_hints(
    ready: &BTreeSet<NodeId>,
    pruned: &HashSet<NodeId>,
    view: &crate::framework::plan::ExecView<'_>,
    appended: &[crate::framework::plan::PlanNode],
    orig_n: usize,
    edges: &[crate::framework::plan::PlanEdge],
    logical_outputs: &HashMap<NodeId, ContentHash>,
    node_idx_of: &HashMap<NodeId, u32>,
    job_dir: &std::path::Path,
    cache: Arc<CacheHandle>,
    bypass_cache: bool,
    cancel: &CancellationToken,
    deadline: Option<Instant>,
    started: Instant,
    hints: &mut HashMap<NodeId, crate::framework::dag_opt::ScheduleHint>,
    probes: &mut HashMap<InvocationKey, Option<Arc<CacheHit>>>,
    prepared_hits: &mut HashMap<NodeId, Arc<CacheHit>>,
) -> Result<(), PlanError> {
    prepared_hits.clear();
    if bypass_cache {
        for &node_id in ready {
            if !pruned.contains(&node_id) {
                hints.entry(node_id).or_default().cache_warm = false;
            }
        }
        return Ok(());
    }

    let mut node_keys = Vec::with_capacity(ready.len());
    let mut unseen_keys = Vec::new();
    let mut unseen = HashSet::new();
    for &node_id in ready {
        // A control-policy-pruned descendant can be re-added to `ready` when a
        // different parent completes. Its killed input intentionally has no
        // logical hash; the spawn loop will discard it, so do not probe it.
        if pruned.contains(&node_id) {
            continue;
        }
        let node = node_at(view, appended, orig_n, node_id);
        let preds = predecessors(edges, node_id);
        let input_hash = gather_input_hash(node_id, &preds, logical_outputs)?;
        let key = node_cache_key(node, input_hash);
        node_keys.push((node_id, key));
        if !probes.contains_key(&key) && unseen.insert(key) {
            let node_idx = node_idx_of[&node_id];
            let into_stage_dir = job_dir
                .join("stages")
                .join(format!("{node_idx}-{}", node.stage.name()));
            unseen_keys.push((key, node.stage.clone(), into_stage_dir));
        }
    }

    let mut pending = Vec::with_capacity(unseen_keys.len());
    for (key, stage, path) in unseen_keys {
        pending.push(tagged_cache_lookup_off_thread(
            cache.clone(),
            key,
            stage,
            path,
        ));
    }
    let completed = futures::stream::iter(pending)
        .buffer_unordered(MAX_CACHE_PROBE_CONCURRENCY)
        .collect::<Vec<_>>();
    tokio::pin!(completed);
    let completed = tokio::select! {
        // If deadline and cancellation become ready together, preserve the
        // deadline error the loop's pre-probe check would have reported.
        biased;
        _ = sleep_until_opt(deadline), if deadline.is_some() => {
            return Err(PlanError::DeadlineExceeded {
                elapsed: started.elapsed(),
            });
        }
        _ = cancel.cancelled() => return Err(PlanError::Cancelled),
        completed = &mut completed => completed,
    };
    for (key, result) in completed {
        let hit = result.map_err(|error| {
            PlanError::Other(format!(
                "cache-aware probe worker failed for {}: {error}",
                key.to_hex()
            ))
        })?;
        probes.insert(key, hit.map(Arc::new));
    }

    for (node_id, key) in node_keys {
        let hit = probes.get(&key).and_then(Option::as_ref);
        hints.entry(node_id).or_default().cache_warm = hit.is_some();
        if let Some(hit) = hit {
            prepared_hits.insert(node_id, hit.clone());
        }
    }
    // `completed` may win immediately before expiry/cancellation, and the
    // synchronous result + hint loops above can still take time on a wide DAG.
    // Recheck after that work so a probe cannot return launchable hints after
    // the plan has stopped accepting new stages.
    if let Some(error) = plan_stop_error(deadline, started, cancel) {
        return Err(error);
    }
    Ok(())
}

/// Pick a real cache hit first when ADR 0102's cache-aware pass is enabled,
/// then highest user priority, then longest critical path. Ties use ascending
/// `NodeId`, keeping no-optimizer/all-neutral order byte-identical to historical
/// `ready.iter().next()`. Nodes absent from hints remain fully neutral.
///
/// Determinism: iteration is driven by the sorted `ready` set and only does
/// point `get`s into the `HashMap` `hints`; the composite key
/// `(cache_warm, user_priority, critical_path_len, Reverse(id))` is unique per
/// node (the
/// `Reverse(id)` tail is a total order), so the argmax never depends on
/// `max_by_key`'s tie rule. `user_priority` (ADR 0102) is 0 unless the
/// `priority_aware` pass ran, and `cache_warm` is false unless the cache-aware
/// pass ran, so default selection remains historical.
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
        let h = hints.get(id);
        // ADR 0102: a real cache hit is cheapest progress and dominates user
        // priority. Both fields stay neutral unless their default-off passes ran.
        let warm = h.map(|h| h.cache_warm).unwrap_or(false);
        let prio = h.map(|h| h.user_priority).unwrap_or(0);
        let cp = h.map(|h| h.critical_path_len).unwrap_or(0);
        (warm, prio, cp, std::cmp::Reverse(*id))
    })
}

/// Release a cache-key reservation that will never produce a completion.
///
/// Pipeline prediction is the only path that can abandon an in-flight key.
/// Ordinary completion performs the same cache-probe invalidation and waiter
/// wake-up inline. Keeping this operation explicit prevents a waiter deferred
/// behind a rejected manifest from disappearing permanently.
#[allow(clippy::too_many_arguments)]
fn abandon_inflight_key(
    node_id: NodeId,
    key: InvocationKey,
    inflight_keys: &mut HashSet<InvocationKey>,
    node_key_of: &mut HashMap<NodeId, InvocationKey>,
    deferred: &mut HashMap<InvocationKey, Vec<NodeId>>,
    cache_probes: &mut HashMap<InvocationKey, Option<Arc<CacheHit>>>,
    prepared_cache_hits: &mut HashMap<NodeId, Arc<CacheHit>>,
    ready: &mut BTreeSet<NodeId>,
    pruned: &HashSet<NodeId>,
) {
    node_key_of.remove(&node_id);
    cache_probes.remove(&key);
    inflight_keys.remove(&key);
    if let Some(waiters) = deferred.remove(&key) {
        for waiter in waiters {
            prepared_cache_hits.remove(&waiter);
            if !pruned.contains(&waiter) {
                ready.insert(waiter);
            }
        }
    }
}

/// Distinguishes a best-effort malformed HPO delta from a fail-closed
/// training-I/O admission refusal.
enum SpawnInjectionError {
    Structural(PlanError),
    TrainingIo(PlanError),
}

/// Reserve the exact canonical node ids/status indices for one manifest-sized
/// single-node map fan-out before its producer starts. The nodes stay dormant:
/// no root artifact, ready-set entry, cache write, directory, or lifecycle
/// event exists until the producer succeeds. Reserving up front gives private
/// early consumers the same `node_idx` and cache key the ordinary fan-out will
/// use, while the single-expansion optimizer rule keeps allocation order equal
/// to the default path.
struct PreparedPipelineChildren {
    children: Vec<PipelineChildSpec>,
    profiles: Vec<(NodeId, ResolvedTrainingIoNode)>,
}

fn local_pipeline_profiles(
    profiles: &[(NodeId, ResolvedTrainingIoNode)],
) -> HashMap<NodeId, ResolvedTrainingIoNode> {
    profiles
        .first()
        .map(|(_, selected)| HashMap::from([(0, selected.clone())]))
        .unwrap_or_default()
}

/// Result of the optional pipeline launch probe. `ResolvedFallback` is
/// distinct from `Declined`: once the child template's cookbook declaration
/// and calibration callback has run, ordinary fan-out must reuse that immutable
/// answer rather than invoking user code a second time.
enum PipelineLaunchDecision {
    Declined,
    ResolvedFallback(HashMap<NodeId, ResolvedTrainingIoNode>),
    Launch(PipelineLaunch),
}

struct PendingSpawn {
    delta: crate::framework::control::SpawnDelta,
    /// `Some(empty)` is meaningful: the template was resolved and declared no
    /// training-I/O profile. `None` means this is an ordinary control/map spawn
    /// whose declaration has not run yet.
    resolved_training_io: Option<HashMap<NodeId, ResolvedTrainingIoNode>>,
}

struct PipelineReservationContext<'a> {
    inherited_partition: Option<blut_types::partition::PartitionKey>,
    env: &'a NodeEnv,
    orig_n: usize,
    appended_len: usize,
    order_len: usize,
}

struct PipelineLaunchContext<'a> {
    inherited_partition: Option<blut_types::partition::PartitionKey>,
    env: &'a NodeEnv,
    orig_n: usize,
    appended_len: usize,
    order_len: usize,
    inflight_keys: &'a HashSet<InvocationKey>,
    spawn_capacity: usize,
}

fn prepare_pipeline_children(
    expansion: &crate::framework::plan::MapExpansion,
    element_hashes: &[ContentHash],
    parent_logical: ContentHash,
    context: PipelineReservationContext<'_>,
) -> Result<PreparedPipelineChildren, SpawnInjectionError> {
    if expansion.template.nodes.len() != 1 || !expansion.template.edges.is_empty() {
        return Err(SpawnInjectionError::Structural(PlanError::Other(
            "pipeline witness requires a one-node map template".into(),
        )));
    }
    let base = (context.orig_n + context.appended_len) as NodeId;
    let base_label = expansion.label.clone().unwrap_or_else(|| "map".into());
    let mut reserved = Vec::with_capacity(element_hashes.len());
    let mut selected_profiles = Vec::new();

    // Every instance has the same one-node template, arguments, partition,
    // captured hints, and launch snapshot. Resolve that declaration exactly
    // once, then clone the concrete immutable profile onto each canonical
    // instance id. This also lets an invalid manifest cardinality be reconciled
    // without invoking cookbook calibration a second time.
    let mut representative = expansion
        .template
        .instantiate(format!("{base_label}[pipeline-template]"));
    if let Some(partition) = context.inherited_partition.clone() {
        representative = representative.with_partition(partition);
    }
    let local_profiles = context
        .env
        .training_io_resolver
        .resolve_injected_plan(&representative)
        .map_err(SpawnInjectionError::TrainingIo)?;
    let (mut nodes, edges, initial) = representative
        .into_parts()
        .map_err(SpawnInjectionError::Structural)?;
    if nodes.len() != 1 || !edges.is_empty() || !initial.is_empty() {
        return Err(SpawnInjectionError::Structural(PlanError::Other(
            "pipeline map template changed after optimizer certification".into(),
        )));
    }
    if local_profiles.keys().any(|local_id| *local_id != 0) {
        return Err(SpawnInjectionError::TrainingIo(PlanError::Other(
            "pipeline template resolved a non-root training I/O profile".into(),
        )));
    }
    let representative_profile = local_profiles.get(&0).cloned();
    let template_node = nodes.pop().expect("one-node pipeline template");

    // Build the full dormant batch before mutating coordinator topology. Any
    // structural arithmetic refusal therefore leaves no partial reservation.
    for (index, _element_hash) in element_hashes.iter().enumerate() {
        let gid = base + index as NodeId;
        let mut node = template_node.clone();
        node.id = gid;
        let node_idx = context.order_len as u32 + index as u32;
        let input_logical = map_element_logical(&parent_logical, index);
        let key = node_cache_key(&node, input_logical);
        if let Some(selected) = representative_profile.clone() {
            selected_profiles.push((gid, selected));
        }
        reserved.push(PipelineChildSpec {
            node,
            node_idx,
            input_logical,
            key,
        });
    }
    Ok(PreparedPipelineChildren {
        children: reserved,
        profiles: selected_profiles,
    })
}

#[allow(clippy::too_many_arguments)]
fn commit_pipeline_children(
    prepared: PreparedPipelineChildren,
    env: &NodeEnv,
    appended: &mut Vec<crate::framework::plan::PlanNode>,
    order: &mut Vec<NodeId>,
    node_idx_of: &mut HashMap<NodeId, u32>,
    indeg: &mut HashMap<NodeId, usize>,
    succs: &mut HashMap<NodeId, Vec<NodeId>>,
) -> Result<Vec<PipelineChildSpec>, SpawnInjectionError> {
    env.install_training_io_profiles(prepared.profiles)
        .map_err(SpawnInjectionError::TrainingIo)?;
    let reserved = prepared.children;
    for child in &reserved {
        let gid = child.node.id;
        appended.push(child.node.clone());
        indeg.insert(gid, 0);
        succs.insert(gid, Vec::new());
        node_idx_of.insert(gid, child.node_idx);
        order.push(gid);
    }
    Ok(reserved)
}

fn try_prepare_pipeline_launch(
    task: &mut NodeTask,
    expansion: &crate::framework::plan::MapExpansion,
    context: PipelineLaunchContext<'_>,
) -> Result<PipelineLaunchDecision, SpawnInjectionError> {
    use crate::framework::async_io::IoMode;
    let PipelineLaunchContext {
        inherited_partition,
        env,
        orig_n,
        appended_len,
        order_len,
        inflight_keys,
        spawn_capacity,
    } = context;

    if !expansion.pipeline
        || task.retry != crate::framework::retry::RetryPolicy::NONE
        || env.launch_target != crate::config::launcher::LaunchTarget::Local
    {
        return Ok(PipelineLaunchDecision::Declined);
    }
    let Some(selected) = env.training_io_node(task.node_id) else {
        return Ok(PipelineLaunchDecision::Declined);
    };
    let (capacity, max_item_bytes) = match selected.profile.pipeline {
        IoMode::Inline => return Ok(PipelineLaunchDecision::Declined),
        IoMode::Bounded {
            capacity,
            max_item_bytes,
        } if capacity > 0 && max_item_bytes > 0 => (capacity, max_item_bytes),
        IoMode::Bounded { .. } => return Ok(PipelineLaunchDecision::Declined),
    };
    let Some(element_kind) = task.stage.output_element_kind() else {
        return Ok(PipelineLaunchDecision::Declined);
    };
    let manifest_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        task.stage.pipeline_manifest_erased(&task.input, &task.args)
    }));
    let manifest = match manifest_result {
        Err(_) => {
            tracing::warn!(
                "pipeline manifest for stage '{}' panicked; declining optional overlap",
                task.stage.name()
            );
            return Ok(PipelineLaunchDecision::Declined);
        }
        Ok(result) => match result {
            Ok(Some(manifest)) if !manifest.element_hashes.is_empty() => manifest,
            Ok(_) => return Ok(PipelineLaunchDecision::Declined),
            Err(error) => {
                tracing::warn!(
                    "pipeline manifest for stage '{}' declined: {error}",
                    task.stage.name()
                );
                return Ok(PipelineLaunchDecision::Declined);
            }
        },
    };
    if manifest.element_hashes.len() > spawn_capacity {
        return Ok(PipelineLaunchDecision::Declined);
    }
    let predicted_parent_logical =
        crate::framework::artifact::list_content_hash_from_element_hashes(
            manifest.element_hashes.iter().copied(),
        );
    let prepared = prepare_pipeline_children(
        expansion,
        &manifest.element_hashes,
        predicted_parent_logical,
        PipelineReservationContext {
            inherited_partition,
            env,
            orig_n,
            appended_len,
            order_len,
        },
    )?;
    let resolved_fallback = || local_pipeline_profiles(&prepared.profiles);
    let mut predicted_keys = HashSet::with_capacity(prepared.children.len());
    for child in &prepared.children {
        if child.key == task.key
            || inflight_keys.contains(&child.key)
            || !predicted_keys.insert(child.key)
            || child.node.retry.unwrap_or_else(|| child.node.stage.retry())
                != crate::framework::retry::RetryPolicy::NONE
        {
            return Ok(PipelineLaunchDecision::ResolvedFallback(resolved_fallback()));
        }
    }
    let profiles: HashMap<NodeId, &ResolvedTrainingIoNode> = prepared
        .profiles
        .iter()
        .map(|(node_id, selected)| (*node_id, selected))
        .collect();
    let mut child_request = None;
    for child in &prepared.children {
        let request = AdmissionRequest::for_stage(
            child.node.stage.as_ref(),
            &child.node.args,
            env.memory_budget_gib,
            profiles
                .get(&child.node.id)
                .map(|selected| &selected.profile),
        )
        .map_err(|error| SpawnInjectionError::Structural(PlanError::Other(error)))?;
        if child_request
            .as_ref()
            .is_some_and(|selected: &AdmissionRequest| selected != &request)
        {
            return Ok(PipelineLaunchDecision::ResolvedFallback(resolved_fallback()));
        }
        child_request.get_or_insert(request);
    }
    let Some(child_request) = child_request else {
        return Ok(PipelineLaunchDecision::ResolvedFallback(resolved_fallback()));
    };
    let parent_request = AdmissionRequest::for_task(task, env)
        .map_err(|error| SpawnInjectionError::Structural(PlanError::Other(error)))?;
    let Some(combined) =
        AdmissionRequest::combine_pipeline(parent_request, child_request, env.memory_budget_gib)
    else {
        return Ok(PipelineLaunchDecision::ResolvedFallback(resolved_fallback()));
    };
    let Some(lease) = try_acquire_admission_from(&combined, &env.resources, &env.gpu, &env.memory)
    else {
        return Ok(PipelineLaunchDecision::ResolvedFallback(resolved_fallback()));
    };
    let (emitter, receiver) = crate::framework::stage::pipeline_channel(
        capacity,
        max_item_bytes,
        element_kind,
        task.stage.name(),
    );
    task.pipeline_emitter = Some(emitter);
    let producer_stage = task.stage.clone();
    let stages_root = env.job_dir.join("stages");
    let producer_final_stage_dir =
        stages_root.join(format!("{}-{}", task.node_idx, task.stage.name()));
    let producer_tmp_stage_dir = stages_root.join(format!(
        ".tmp-{}-{}-{}",
        task.node_idx,
        task.stage.name(),
        task.key.to_hex()
    ));
    Ok(PipelineLaunchDecision::Launch(PipelineLaunch {
        receiver,
        admission: Arc::new(lease),
        manifest,
        predicted_parent_logical,
        children: prepared.children,
        profiles: prepared.profiles,
        producer_stage,
        producer_tmp_stage_dir,
        producer_final_stage_dir,
        spawn_capacity,
        max_item_bytes,
    }))
}

impl std::fmt::Display for SpawnInjectionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Structural(error) | Self::TrainingIo(error) => error.fmt(formatter),
        }
    }
}

/// Inject a `Spawn` delta into the running parallel schedule (v0.20). The
/// sub-plan's local node ids `0..k` are relabelled to globals `base + l`
/// (`base = orig_n + appended.len()`), its nodes moved into `appended`, its
/// edges/in-degrees/successors/topo-order extended, its graph-inputs seeded as
/// root outputs, and its roots inserted into `ready`. Returns the count
/// injected. Rejects cyclic/empty sub-plans and any declaration larger than the
/// RAM envelope reserved before this execution began. Structural HPO errors
/// remain best-effort; training-I/O admission errors are distinguished so the
/// caller can fail closed before a child runs.
#[allow(clippy::too_many_arguments)]
fn inject_spawn(
    delta: crate::framework::control::SpawnDelta,
    resolved_training_io: Option<HashMap<NodeId, ResolvedTrainingIoNode>>,
    inherited_partition: Option<blut_types::partition::PartitionKey>,
    env: &NodeEnv,
    memory_budget_gib: u32,
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
) -> Result<usize, SpawnInjectionError> {
    use crate::framework::plan::PlanEdge;
    let crate::framework::control::SpawnDelta {
        mut subplan,
        root_seeds,
        ..
    } = delta;
    // A runtime-created node is still part of the partitioned execution that
    // created it. Bind the whole sub-plan before extracting its nodes so PBT,
    // TPE, and map shards cannot reuse another cell's cache entries.
    if let Some(partition) = inherited_partition {
        subplan = subplan.with_partition(partition);
    }
    // Reject a runtime sub-plan whose largest declared stage exceeds the
    // whole-job RAM reservation. Silently clamping that stage to the smaller
    // semaphore would not reduce its real memory use and would let TPE/PBT
    // suggestions escape the envelope acquired before execution began.
    let spawn_gib = subplan
        .declared_footprint()
        .ram_bytes
        .div_ceil(crate::broker::footprint::GIB)
        .clamp(1, u64::from(u32::MAX)) as u32;
    if spawn_gib > memory_budget_gib {
        // STRUCTURAL, not TrainingIo: an oversized best-effort HPO suggestion is
        // dropped with a warning (the search continues), while the same error on
        // a REQUIRED map shard fails the plan — the map_spawn arm downstream
        // distinguishes the two. TrainingIo would make every oversized
        // suggestion fatal.
        return Err(SpawnInjectionError::Structural(PlanError::Other(format!(
            "runtime spawn requires {spawn_gib} GiB but this job reserved only {memory_budget_gib} GiB"
        ))));
    }

    // Local topo order (also the cycle/empty check) BEFORE we mutate anything.
    let local_order = subplan
        .topo_order()
        .map_err(SpawnInjectionError::Structural)?;
    let base = (orig_n + appended.len()) as NodeId;
    // Resolve every declaring node from the SAME immutable launch witness as
    // the static plan. This happens before any structural mutation or ready-set
    // insertion, so a missing/oversized child profile fails closed rather than
    // running without its retained-byte bill.
    let local_profiles = match resolved_training_io {
        Some(profiles) => profiles,
        None => env
            .training_io_resolver
            .resolve_injected_plan(&subplan)
            .map_err(SpawnInjectionError::TrainingIo)?,
    };
    let (nodes, edges, initial) = subplan
        .into_parts()
        .map_err(SpawnInjectionError::Structural)?;
    let k = nodes.len();
    env.install_training_io_profiles(
        local_profiles
            .into_iter()
            .map(|(local_id, selected)| (base + local_id, selected)),
    )
    .map_err(SpawnInjectionError::TrainingIo)?;

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
    // Seed explicit root inputs (ADR 0078 `map_output`: the list element).
    // The seed carries its OWN logical hash (derived from the parent's logical
    // hash + element index) so children of a nondeterministic parent keep
    // stable cache keys across reruns.
    for (local_id, art, logical) in root_seeds {
        let gid = base + local_id;
        outputs.insert(gid, art);
        logical_outputs.insert(gid, logical);
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
        NodeFailure::Plan(error) => error,
        NodeFailure::Other(s) => PlanError::Other(s),
        NodeFailure::SpeculationCleanup { path, source } => PlanError::Other(format!(
            "failed to remove private speculation scratch {}: {source}",
            path.display()
        )),
        NodeFailure::SpeculationRollback {
            path,
            source,
            cause,
        } => PlanError::Other(format!(
            "{cause}; failed to roll back speculative publication {}: {source}",
            path.display()
        )),
    }
}

/// Shared coordinator setup: validate and persist every fallible static input,
/// then spawn the status writer and seed the initial outputs. CONSUMES `ctx`, MOVING the
/// `StatusHub` into the one `NodeEnv` — when the last `Arc<NodeEnv>`
/// drops, the hub (and its lifecycle Sender) drop, the writer's
/// lifecycle channel closes, and the writer exits. The lossless
/// lifecycle receiver is handed to the writer here.
struct Prelude {
    writer_handle: tokio::task::JoinHandle<std::io::Result<()>>,
    env: Arc<NodeEnv>,
    outputs: HashMap<NodeId, ErasedArtifact>,
    logical_outputs: HashMap<NodeId, ContentHash>,
    content_outputs: HashMap<NodeId, ContentId>,
}

fn prelude(mut ctx: ExecCtx, plan: &CompiledPlan) -> Result<Prelude, PlanError> {
    debug_assert!(
        !ctx.resources.is_empty(),
        "ExecCtx must declare resource semaphores"
    );
    std::fs::create_dir_all(&ctx.job_dir)?;
    let view = plan.exec_view();
    let args_path = ctx.job_dir.join("args.json");
    let args_body = serde_json::to_vec_pretty(view.recipe_args)
        .map_err(|e| PlanError::Other(format!("serialize args: {e}")))?;
    std::fs::write(&args_path, args_body)?;

    let mut outputs: HashMap<NodeId, ErasedArtifact> = HashMap::new();
    let mut logical_outputs: HashMap<NodeId, ContentHash> = HashMap::new();
    let content_outputs: HashMap<NodeId, ContentId> = HashMap::new();
    for (id, art) in view.initial {
        let lh = content_hash_from_erased(art);
        outputs.insert(*id, art.clone());
        logical_outputs.insert(*id, lh);
    }

    let training_io_resolver = ctx.training_io_resolver.take().ok_or_else(|| {
        PlanError::Other("prepared execution is missing its training I/O resolver witness".into())
    })?;
    let mut training_io_node_hints = std::mem::take(&mut ctx.training_io_node_hints);
    let mut training_io_profiles = HashMap::with_capacity(ctx.training_io_profiles.len());
    for (node_id, profile) in std::mem::take(&mut ctx.training_io_profiles) {
        let hints = training_io_node_hints.remove(&node_id).ok_or_else(|| {
            PlanError::Other(format!(
                "training I/O profile for node {node_id} is missing its resolved launch hints"
            ))
        })?;
        training_io_profiles.insert(node_id, ResolvedTrainingIoNode { profile, hints });
    }
    if let Some(node_id) = training_io_node_hints.keys().next() {
        return Err(PlanError::Other(format!(
            "training I/O launch hints for node {node_id} have no resolved profile"
        )));
    }

    // Only after every fallible setup step above succeeds do we hand the
    // lifecycle receiver to a spawned writer. An args/provenance failure can
    // therefore never detach a writer whose eventual error no caller awaits.
    let lifecycle_rx = ctx
        .lifecycle_rx
        .take()
        .ok_or_else(|| PlanError::Other("ExecCtx.lifecycle_rx already consumed".into()))?;
    #[cfg(test)]
    let writer_handle = if let Some(kind) = ctx.status_writer_failure.take() {
        if let Some(started) = ctx.status_writer_started.take() {
            started.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        tokio::spawn(async move {
            let mut lifecycle_rx = lifecycle_rx;
            while lifecycle_rx.recv().await.is_some() {}
            Err(std::io::Error::new(
                kind,
                "injected lifecycle persistence failure",
            ))
        })
    } else {
        spawn_status_writer_checked(&ctx.status, lifecycle_rx, &ctx.job_dir)?
    };
    #[cfg(not(test))]
    let writer_handle = spawn_status_writer_checked(&ctx.status, lifecycle_rx, &ctx.job_dir)?;

    // MOVE ctx's fields into env — the hub Arc lives only here now.
    let env = Arc::new(NodeEnv {
        job_dir: ctx.job_dir,
        cache: ctx.cache,
        tenant: ctx.tenant,
        status: ctx.status,
        cancel: ctx.cancel,
        resources: ctx.resources,
        gpu: ctx.gpu,
        memory: ctx.memory,
        memory_budget_gib: ctx.memory_budget_gib,
        launch_target: ctx.launch_target,
        device_index: ctx.device_index,
        fb_warm: ctx.fb_warm,
        admitted_workers: ctx.admitted_workers,
        admitted_batch_size: ctx.admitted_batch_size,
        training_io_profiles: std::sync::RwLock::new(training_io_profiles),
        training_io_resolver,
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
        content_outputs,
    })
}

/// Drop the (sole) `NodeEnv` Arc so the status channel closes, then
/// await the writer to flush the tail events. The caller MUST have
/// dropped every other `Arc<NodeEnv>` first (the parallel JoinSet must
/// be fully drained), or this hangs.
async fn finish_writer(
    env: Arc<NodeEnv>,
    writer_handle: tokio::task::JoinHandle<std::io::Result<()>>,
) -> Result<(), PlanError> {
    drop(env);
    await_status_writer(writer_handle).await
}

async fn finish_writer_after_error(
    env: Arc<NodeEnv>,
    writer_handle: tokio::task::JoinHandle<std::io::Result<()>>,
    primary: PlanError,
) -> PlanError {
    match finish_writer(env, writer_handle).await {
        Ok(()) => primary,
        Err(status_error) => PlanError::Other(format!("{primary}; additionally, {status_error}")),
    }
}

async fn await_status_writer(
    writer_handle: tokio::task::JoinHandle<std::io::Result<()>>,
) -> Result<(), PlanError> {
    match writer_handle.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(PlanError::Other(format!(
            "status lifecycle persistence failed: {error}"
        ))),
        Err(error) => Err(PlanError::Other(format!(
            "status lifecycle writer task failed: {error}"
        ))),
    }
}

/// Dispatch a plan to the configured executor. Default is
/// [`SequentialExecutor`] (the debugging-friendly, burn-in-stable
/// path); set `BLUT_EXECUTOR=parallel` to opt into [`ParallelExecutor`].
/// One seam so the CLI/TUI launch sites don't each branch on the env.
pub async fn execute_plan(plan: CompiledPlan, mut ctx: ExecCtx) -> Result<PlanResult, PlanError> {
    let plan = if ctx.prepared_execution.is_some() {
        plan
    } else {
        prepare_plan_for_execution(plan, &mut ctx)?
    };
    execute_prepared_plan(plan, ctx).await
}

async fn execute_prepared_plan(plan: CompiledPlan, ctx: ExecCtx) -> Result<PlanResult, PlanError> {
    let mode = ctx
        .prepared_execution
        .as_ref()
        .map(|prepared| prepared.mode)
        .ok_or_else(|| {
            PlanError::Other("executor received a plan without a preparation witness".into())
        })?;
    match mode {
        PreparedExecutorMode::Sequential => SequentialExecutor::execute_prepared(plan, ctx).await,
        PreparedExecutorMode::Parallel => ParallelExecutor::execute_prepared(plan, ctx).await,
    }
}

// ════════════════════════════════════════════════════════════════════
// Sequential executor — one node at a time, topo order.
// ════════════════════════════════════════════════════════════════════

pub struct SequentialExecutor;

impl SequentialExecutor {
    /// Execute the plan to completion, one stage at a time.
    pub async fn execute(plan: CompiledPlan, mut ctx: ExecCtx) -> Result<PlanResult, PlanError> {
        let plan = if ctx.prepared_execution.is_some() {
            plan
        } else {
            prepare_plan_for_requested_executor(plan, &mut ctx, PreparedExecutorMode::Sequential)?
        };
        execute_prepared_plan(plan, ctx).await
    }

    async fn execute_prepared(
        plan: CompiledPlan,
        mut ctx: ExecCtx,
    ) -> Result<PlanResult, PlanError> {
        let prepared = take_prepared_execution(&mut ctx, PreparedExecutorMode::Sequential)?;
        debug_assert!(prepared.schedule_hints.is_empty());
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
            mut content_outputs,
        } = prelude(ctx, &plan)?;

        let mut n_hits = 0usize;
        let mut n_misses = 0usize;
        let mut warnings: Vec<StageWarning> = Vec::new();

        for (idx, node_id) in order.iter().enumerate() {
            // Plan-level deadline (D2): coarse between-stage check; a
            // stage mid-run is bounded by its own hard timeout instead.
            if let Some(dl) = deadline
                && Instant::now() >= dl
            {
                env.cancel.cancel();
                let error = PlanError::DeadlineExceeded {
                    elapsed: started.elapsed(),
                };
                return Err(finish_writer_after_error(env, writer_handle, error).await);
            }
            if env.cancel.is_cancelled() {
                env.status.emit(StageEvent::StageFailed {
                    node_idx: idx as u32,
                    stage_name: "<cancelled>".into(),
                    error: "plan cancelled before stage".into(),
                    failure: None,
                });
                return Err(
                    finish_writer_after_error(env, writer_handle, PlanError::Cancelled).await,
                );
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
                &content_outputs,
                node_cancel,
            ) {
                Ok(t) => t,
                Err(e) => {
                    return Err(finish_writer_after_error(env, writer_handle, e).await);
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
                    content_outputs.insert(outcome.node_id, outcome.content_id);
                }
                Err(f) => {
                    // ADR 0071: an advisory stage's failure is a non-fatal warning,
                    // not a plan failure — record it and STOP (the remaining topo
                    // nodes are its descendants and can't run). `strict_advisory()`
                    // forces the old fail-hard behaviour for CI.
                    if let NodeFailure::Stage { idx, stage, source } = &f
                        && node.stage.is_advisory()
                        && !strict_advisory()
                    {
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
                    let error = plan_error_of(f);
                    return Err(finish_writer_after_error(env, writer_handle, error).await);
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
        finish_writer(env, writer_handle).await?;

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

/// Resolve the optimizer's explicit Plan -> Plan witness into the special
/// whole-plan fast path. Runtime policy may refine a witness, never invent one.
fn full_linear_fusion_order(plan: &CompiledPlan) -> Option<Vec<NodeId>> {
    let mut groups = plan.fused_subchains();
    let members = groups.next()?;
    if groups.next().is_some()
        || members.len() != plan.nodes.len()
        || plan
            .initial
            .keys()
            .any(|node_id| Some(node_id) != members.first())
    {
        return None;
    }
    Some(members.to_vec())
}

#[derive(Clone)]
struct FusionGroup {
    node_ids: Vec<NodeId>,
    admission: AdmissionRequest,
}

/// Refine optimizer-recorded structural candidates into maximal equal-admission
/// runtime groups. A fixed dummy input produces a digest of every static cache
/// key component; tails whose identity also exists outside their group retain
/// ordinary coordinator boundaries so duplicate-key single-flight cannot race.
fn internal_linear_fusion_groups(
    plan: &CompiledPlan,
    memory_budget_gib: u32,
) -> HashMap<NodeId, FusionGroup> {
    let request_for = |id: NodeId| {
        let node = &plan.nodes[id as usize];
        AdmissionRequest::for_stage(node.stage.as_ref(), &node.args, memory_budget_gib, None).ok()
    };
    let static_ids: Vec<InvocationKey> = plan
        .nodes
        .iter()
        .map(|node| node_cache_key(node, ContentHash([0; 32])))
        .collect();
    let mut global_counts = HashMap::<InvocationKey, usize>::new();
    for identity in &static_ids {
        *global_counts.entry(*identity).or_default() += 1;
    }
    let mut groups = HashMap::new();

    for candidate in plan.fused_subchains() {
        let mut start = 0;
        while start < candidate.len() {
            let admission = request_for(candidate[start]);
            let mut end = start + 1;
            while end < candidate.len() && request_for(candidate[end]) == admission {
                end += 1;
            }
            let node_ids = candidate[start..end].to_vec();
            if node_ids.len() >= 2 {
                let mut local_counts = HashMap::<InvocationKey, usize>::new();
                for node_id in &node_ids {
                    *local_counts
                        .entry(static_ids[*node_id as usize])
                        .or_default() += 1;
                }
                let tail_can_collide = node_ids.iter().skip(1).any(|node_id| {
                    let identity = static_ids[*node_id as usize];
                    global_counts[&identity] > local_counts[&identity]
                });
                if !tail_can_collide && let Some(admission) = admission {
                    groups.insert(
                        node_ids[0],
                        FusionGroup {
                            node_ids,
                            admission,
                        },
                    );
                }
            }
            start = end;
        }
    }
    groups
}

#[cfg(feature = "p2p")]
fn fusion_runtime_is_local(ctx: &ExecCtx) -> bool {
    ctx.dispatch_policy.is_none() && ctx.dispatcher.is_none()
}

#[cfg(not(feature = "p2p"))]
fn fusion_runtime_is_local(_ctx: &ExecCtx) -> bool {
    true
}

/// Execute one eligible linear plan without returning to the ready-queue
/// coordinator between nodes. Every node still goes through `build_task` and
/// `run_node`, so its normal cache key, FW-2 promotion, sidecars, events, and
/// output artifact remain the source of truth. One lazily-acquired shared
/// admission lease spans every actual stage run in the chain; all-hit chains
/// acquire nothing.
async fn execute_fused_linear_plan(
    plan: CompiledPlan,
    ctx: ExecCtx,
    order: Vec<NodeId>,
    admission_request: AdmissionRequest,
    started: Instant,
) -> Result<PlanResult, PlanError> {
    let view = plan.exec_view();
    let deadline = ctx.deadline;
    let Prelude {
        writer_handle,
        env,
        mut outputs,
        mut logical_outputs,
        mut content_outputs,
    } = prelude(ctx, &plan)?;
    let mut n_hits = 0usize;
    let mut n_misses = 0usize;
    let mut admission = FusionAdmission::new(admission_request);
    let mut in_process_input = None;

    if env.cancel.is_cancelled() {
        env.status.emit(StageEvent::StageFailed {
            node_idx: 0,
            stage_name: "<cancelled>".into(),
            error: "plan cancelled before stage".into(),
            failure: None,
        });
        return Err(finish_writer_after_error(env, writer_handle, PlanError::Cancelled).await);
    }

    for (idx, node_id) in order.iter().enumerate() {
        if let Some(error) = plan_stop_error(deadline, started, &env.cancel) {
            env.cancel.cancel();
            drop(admission);
            return Err(finish_writer_after_error(env, writer_handle, error).await);
        }
        let node = &view.nodes[*node_id as usize];
        let task = match build_task(
            node,
            idx as u32,
            view.edges,
            &outputs,
            &logical_outputs,
            &content_outputs,
            KillSlot::new(env.cancel.child_token()),
        ) {
            Ok(task) => task,
            Err(error) => {
                env.cancel.cancel();
                drop(admission);
                return Err(finish_writer_after_error(env, writer_handle, error).await);
            }
        };
        let fused_stage_name = task.stage.name();
        let run_result = std::panic::AssertUnwindSafe(run_node_with_admission(
            task,
            env.clone(),
            Some(&mut admission),
            None,
            true,
            in_process_input.take(),
        ))
        .catch_unwind()
        .await;
        match run_result {
            Ok(Ok(mut outcome)) => {
                if outcome.cache_hit {
                    n_hits += 1;
                } else {
                    n_misses += 1;
                }
                in_process_input = outcome.in_process_output.take();
                outputs.insert(outcome.node_id, outcome.output);
                logical_outputs.insert(outcome.node_id, outcome.logical);
                content_outputs.insert(outcome.node_id, outcome.content_id);
            }
            Ok(Err(failure)) => {
                env.cancel.cancel();
                drop(admission);
                let error = plan_error_of(failure);
                return Err(finish_writer_after_error(env, writer_handle, error).await);
            }
            Err(_) => {
                env.cancel.cancel();
                drop(admission);
                let error = PlanError::Other(format!(
                    "node task panicked in fused node {idx} ({fused_stage_name})"
                ));
                return Err(finish_writer_after_error(env, writer_handle, error).await);
            }
        }
    }

    let final_output = order.last().and_then(|id| outputs.remove(id));
    drop(admission);
    finish_writer(env, writer_handle).await?;
    Ok(PlanResult {
        final_output,
        n_stages: order.len(),
        n_cache_hits: n_hits,
        n_cache_misses: n_misses,
        elapsed: started.elapsed(),
        warnings: Vec::new(),
    })
}

/// Execute one coordinator-selected internal fusion group. The first task is
/// built by the coordinator from external predecessor state; later tasks are
/// built here from the preceding fused outcomes. All outcomes return together
/// so scheduler mutation remains coordinator-owned.
async fn run_fused_group(
    first_task: NodeTask,
    remaining: Vec<(crate::framework::plan::PlanNode, u32)>,
    env: Arc<NodeEnv>,
    admission_request: AdmissionRequest,
    deadline: Option<Instant>,
    started: Instant,
) -> Result<Vec<NodeOutcome>, NodeFailure> {
    let mut admission = FusionAdmission::new(admission_request);
    let mut outputs = HashMap::new();
    let mut logical_outputs = HashMap::new();
    let mut content_outputs = HashMap::new();
    let mut outcomes = Vec::with_capacity(remaining.len() + 1);
    let mut in_process_input = None;
    let mut next_task = Some(first_task);
    let mut remaining = remaining.into_iter();

    loop {
        if let Some(error) = plan_stop_error(deadline, started, &env.cancel) {
            return Err(NodeFailure::Plan(error));
        }
        let task = next_task.take().expect("fusion group task initialized");
        let node_idx = task.node_idx;
        let stage_name = task.stage.name().to_string();
        let run_result = std::panic::AssertUnwindSafe(run_node_with_admission(
            task,
            env.clone(),
            Some(&mut admission),
            None,
            true,
            in_process_input.take(),
        ))
        .catch_unwind()
        .await;
        let mut outcome = match run_result {
            Ok(result) => result?,
            Err(_) => {
                return Err(NodeFailure::Other(format!(
                    "node task panicked in fused node {node_idx} ({stage_name})"
                )));
            }
        };
        in_process_input = outcome.in_process_output.take();

        let Some((node, next_idx)) = remaining.next() else {
            outcomes.push(outcome);
            break;
        };
        // The optimizer proves a linear subchain, so the next fused member has
        // exactly this outcome as its sole predecessor. Keep scratch maps only
        // for task construction, then clear them: `outcomes` is the one retained
        // copy returned to the coordinator, matching ordinary-path residency.
        let predecessor = outcome.node_id;
        outputs.insert(predecessor, outcome.output.clone());
        logical_outputs.insert(predecessor, outcome.logical);
        content_outputs.insert(predecessor, outcome.content_id);
        let edge = [crate::framework::plan::PlanEdge {
            from: predecessor,
            to: node.id,
        }];
        next_task = Some(
            build_task(
                &node,
                next_idx,
                &edge,
                &outputs,
                &logical_outputs,
                &content_outputs,
                KillSlot::new(env.cancel.child_token()),
            )
            .map_err(NodeFailure::Plan)?,
        );
        outputs.clear();
        logical_outputs.clear();
        content_outputs.clear();
        outcomes.push(outcome);
    }

    Ok(outcomes)
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
    pub async fn execute(plan: CompiledPlan, mut ctx: ExecCtx) -> Result<PlanResult, PlanError> {
        let plan = if ctx.prepared_execution.is_some() {
            plan
        } else {
            prepare_plan_for_requested_executor(plan, &mut ctx, PreparedExecutorMode::Parallel)?
        };
        execute_prepared_plan(plan, ctx).await
    }

    async fn execute_prepared(
        plan: CompiledPlan,
        mut ctx: ExecCtx,
    ) -> Result<PlanResult, PlanError> {
        let prepared = take_prepared_execution(&mut ctx, PreparedExecutorMode::Parallel)?;
        let mut schedule_hints = prepared.schedule_hints;
        let started = Instant::now();

        // Static optimization and async-I/O profile selection already ran in
        // the select-once preparation pass. The retained post-DCE hints drive
        // ready-node priority; an empty map keeps historical NodeId ordering.
        // Cache warmth needs resolved predecessor hashes plus this run's
        // tenant-scoped CacheHandle, neither of which exists at static optimizer
        // time. Preserve the default-off flag for the ready-queue seam below.
        let cache_aware = ctx
            .dag_optimizer
            .as_ref()
            .is_some_and(|optimizer| optimizer.cache_aware);
        let stage_fusion = ctx.dag_optimizer.as_ref().is_some_and(|optimizer| {
            optimizer.stage_fusion && !optimizer.cache_aware && !optimizer.priority_aware
        });
        let mut speculation_candidates: Vec<NodeId> = plan.speculation_candidates().collect();
        speculation_candidates.sort_unstable();
        let speculation_runtime_enabled = !speculation_candidates.is_empty()
            && ctx.control.is_none()
            && ctx.launch_target == crate::config::launcher::LaunchTarget::Local
            && fusion_runtime_is_local(&ctx);
        // Conditional control is deliberately separate from typed data edges:
        // it orders selector before target without contributing an input or a
        // cache-key component. The PlanSpec compiler currently admits one
        // non-reconvergent gate, but keep the scheduler state keyed generically
        // so widening the validated IR does not require changing this seam.
        let has_condition_gates = plan.has_condition_gates();

        // Cache-aware mode retains its deadline/cancellation-aware ready probe
        // path. A linear chain has no ready-order choice, so composing it with
        // fusion has no scheduling upside; fall back rather than weakening the
        // increment-2 slow-store deadline guarantee.
        if stage_fusion
            && ctx.training_io_profiles.is_empty()
            && !has_condition_gates
            && !cache_aware
            && ctx.control.is_none()
            && fusion_runtime_is_local(&ctx)
            && let Some(order) = full_linear_fusion_order(&plan)
            && let Some(admission_request) =
                AdmissionRequest::for_chain(&plan.nodes, ctx.memory_budget_gib)
        {
            return execute_fused_linear_plan(plan, ctx, order, admission_request, started).await;
        }

        let internal_fusion_groups = if stage_fusion
            && ctx.training_io_profiles.is_empty()
            && !has_condition_gates
            && !cache_aware
            && ctx.control.is_none()
            && fusion_runtime_is_local(&ctx)
        {
            internal_linear_fusion_groups(&plan, ctx.memory_budget_gib)
        } else {
            HashMap::new()
        };
        let fused_internal_nodes: HashSet<NodeId> = internal_fusion_groups
            .values()
            .flat_map(|group| group.node_ids.iter().skip(1).copied())
            .collect();

        // `mut`: runtime `Spawn` (PBT/TPE) extends the topo order at runtime.
        let mut order = plan.topo_order()?; // also the cycle check
        let view = plan.exec_view();
        let condition_gates: Vec<_> = view
            .condition_gates
            .iter()
            .map(|gate| (gate.condition, gate.target, gate.when))
            .collect();
        debug_assert_eq!(
            order.len(),
            view.nodes.len(),
            "topo_order must cover all nodes"
        );
        // PlanSpec validation gives a conditional plan one unambiguous data
        // terminal. Capture it before runtime `Spawn` appends unrelated nodes
        // to `order`; spawned work must never become the conditional result.
        let conditional_terminal = has_condition_gates.then(|| {
            *order
                .last()
                .expect("a compiled conditional plan is never empty")
        });
        let conditional_output_candidates = conditional_terminal.map(|terminal| {
            let mut ancestors = HashSet::new();
            let mut stack = vec![terminal];
            while let Some(node_id) = stack.pop() {
                if ancestors.insert(node_id) {
                    stack.extend(
                        view.edges
                            .iter()
                            .filter(|edge| edge.to == node_id)
                            .map(|edge| edge.from),
                    );
                }
            }
            ancestors
        });

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
        let condition_by_target: HashMap<NodeId, (NodeId, bool)> = condition_gates
            .iter()
            .map(|&(condition, target, when)| (target, (condition, when)))
            .collect();
        let mut condition_targets: HashMap<NodeId, Vec<(NodeId, bool)>> = HashMap::new();
        for &(condition, target, when) in &condition_gates {
            condition_targets
                .entry(condition)
                .or_default()
                .push((target, when));
        }
        // A target is absent until its selector has completed with the matching
        // decision. Losing targets are placed in `pruned` with all data
        // descendants and therefore never enter this set.
        let mut enabled_condition_targets: HashSet<NodeId> = HashSet::new();
        let mut speculation: HashMap<NodeId, SpeculationState> = if speculation_runtime_enabled {
            speculation_candidates
                .iter()
                .copied()
                .filter(|target| condition_by_target.contains_key(target))
                .map(|target| (target, SpeculationState::Eligible))
                .collect()
        } else {
            HashMap::new()
        };
        let mut speculative_keys: HashMap<InvocationKey, NodeId> = HashMap::new();
        // Shadow every ordinary in-flight envelope even before its task reaches
        // post-cache admission. Optional speculation must leave this demand
        // untouched; entries disappear only when ordinary work completes or is
        // intentionally pruned.
        let mut ordinary_admission_demands: HashMap<NodeId, AdmissionRequest> = HashMap::new();

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
            mut content_outputs,
        } = prelude(ctx, &plan)?;

        // #4 runtime control state. `control_rx` is the live step stream the
        // coordinator watches between joins; `node_tokens` maps an in-flight
        // node to its kill token; `pruned` is the set of nodes a kill removed
        // from the schedule (the killed node + its descendants); they plus
        // `completed` must cover every node at the end.
        let mut control_rx: Option<broadcast::Receiver<StageEvent>> =
            control.as_ref().map(|_| env.status.subscribe());
        let mut node_tokens: HashMap<NodeId, KillSlot> = HashMap::new();
        // The coordinator does not otherwise keep the stage after spawn. Hold
        // one cheap `Arc<dyn StageDyn>` clone per in-flight node for both live
        // divergence checks and advisory-failure classification, keyed like
        // `node_tokens`, then drop it on completion/kill.
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
        let mut pending_spawns: Vec<PendingSpawn> = Vec::new();
        let mut spawns_total = 0usize;
        // Parents whose manifest-sized map children were allocated before the
        // producer started. Their ordinary completion seam must not inject a
        // second copy; the Pipeline result either publishes private children
        // or seeds these exact dormant nodes for ordinary fallback.
        let mut pipeline_reserved_parents: HashSet<NodeId> = HashSet::new();
        // A runtime pipeline probe may resolve the one-node child template and
        // then decline at a later collision/admission check. Retain that exact
        // immutable answer until ordinary fan-out is queued so cookbook
        // declaration/calibration callbacks still run once.
        let mut pipeline_fallback_profiles: HashMap<
            NodeId,
            HashMap<NodeId, ResolvedTrainingIoNode>,
        > = HashMap::new();
        // The killed nodes + their pruned descendants. `pruned.len()` (not a
        // parallel counter) is the accounting source of truth — it can't drift
        // out of sync with the set the spawn loop consults.
        let mut pruned: HashSet<NodeId> = HashSet::new();

        // Ready set = in-degree-0 nodes, ascending NodeId for
        // deterministic spawn order.
        let mut ready: BTreeSet<NodeId> = indeg
            .iter()
            .filter(|(id, d)| **d == 0 && !condition_by_target.contains_key(id))
            .map(|(id, _)| *id)
            .collect();

        // Single-flight: defer a ready node whose cache key matches one
        // already in flight, so duplicate-key fork nodes don't both burn
        // GPU — the deferred node hits the cache when the first finishes.
        // `inflight_keys` is the set of keys currently running (one node
        // each, by construction); `node_key_of` is the O(1) reverse lookup
        // for "which key did this finished node run under".
        let mut inflight_keys: HashSet<InvocationKey> = HashSet::new();
        let mut node_key_of: HashMap<NodeId, InvocationKey> = HashMap::new();
        let mut deferred: HashMap<InvocationKey, Vec<NodeId>> = HashMap::new();
        // ADR 0102 cache-aware scheduling state. `None` means a probed miss;
        // absence means unprobed. Prepared hits are cheap Arc clones rebuilt
        // for the current ready set and consumed by selected tasks.
        let mut cache_probes: HashMap<InvocationKey, Option<Arc<CacheHit>>> = HashMap::new();
        let mut prepared_cache_hits: HashMap<NodeId, Arc<CacheHit>> = HashMap::new();

        let mut join: tokio::task::JoinSet<SchedulerTaskResult> = tokio::task::JoinSet::new();
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
            return Err(finish_writer_after_error(env, writer_handle, PlanError::Cancelled).await);
        }

        loop {
            // Plan-level deadline (D2): once past it, stop spawning new
            // nodes, cancel + drain the in-flight ones, report
            // DeadlineExceeded (a stage mid-run is bounded by its own
            // hard timeout).
            if first_error.is_none()
                && let Some(dl) = deadline
                && Instant::now() >= dl
            {
                first_error = Some(PlanError::DeadlineExceeded {
                    elapsed: started.elapsed(),
                });
                env.cancel.cancel();
            }
            // #4 SPAWN: drain runtime-injected sub-plans on the coordinator seam
            // BEFORE the spawn-ready loop, so newly-ready roots are scheduled
            // this iteration and the `in_flight == 0` termination check below
            // sees them. Stop injecting once failing (drop pending deltas).
            if first_error.is_none() && !pending_spawns.is_empty() {
                // Drain map_output shards (provenance_parent set) BEFORE
                // best-effort HPO spawns, so a soft HPO spawn can't claim the
                // last cap slot and force a wrong-answer shard to fail. Stable:
                // relative order within each group is preserved.
                pending_spawns.sort_by_key(|pending| pending.delta.provenance_parent.is_none());
                for pending in pending_spawns.drain(..) {
                    let PendingSpawn {
                        delta,
                        resolved_training_io,
                    } = pending;
                    let map_spawn = delta.provenance_parent.is_some();
                    if spawns_total >= MAX_RUNTIME_SPAWNS {
                        // A map_output shard hitting the cap is a WRONG ANSWER
                        // (a dropped element), so fail the plan loudly — unlike
                        // an HPO trial, which is best-effort and may be dropped.
                        if delta.provenance_parent.is_some() {
                            first_error = Some(PlanError::Other(format!(
                                "map fan-out exceeded the runtime spawn cap \
                                 ({MAX_RUNTIME_SPAWNS} nodes); a dropped shard would be a wrong answer"
                            )));
                            break;
                        }
                        tracing::warn!(
                            "runtime spawn cap {MAX_RUNTIME_SPAWNS} reached; dropping further HPO spawns"
                        );
                        break;
                    }
                    let inherited_partition = delta
                        .provenance_parent
                        .and_then(|parent| {
                            node_at(&view, &appended, orig_n, parent).partition.clone()
                        })
                        .or_else(|| {
                            view.nodes
                                .iter()
                                .chain(appended.iter())
                                .find_map(|node| node.partition.clone())
                        });
                    match inject_spawn(
                        delta,
                        resolved_training_io,
                        inherited_partition,
                        &env,
                        env.memory_budget_gib,
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
                        Err(SpawnInjectionError::TrainingIo(error)) => {
                            // A child profile/calibration refusal is an
                            // admission failure, not a best-effort HPO shape
                            // error. Succeeding after silently dropping that
                            // trial would make the admitted search differ from
                            // the executed search, so fail the plan before any
                            // child node reaches the ready set.
                            first_error = Some(error);
                            env.cancel.cancel();
                            break;
                        }
                        Err(SpawnInjectionError::Structural(error)) if map_spawn => {
                            // A malformed map shard is a wrong answer (one list
                            // element would disappear), so it is fatal just like
                            // the map spawn-cap path above.
                            first_error = Some(error);
                            env.cancel.cancel();
                            break;
                        }
                        Err(SpawnInjectionError::Structural(error)) => {
                            tracing::warn!("ignored malformed spawn delta: {error}");
                        }
                    }
                }
            }
            // Spawn ready nodes up to the in-flight cap (unless we're
            // already failing — then stop spawning and just drain).
            if first_error.is_none() {
                if cache_aware
                    && let Err(error) = refresh_cache_warm_hints(
                        &ready,
                        &pruned,
                        &view,
                        &appended,
                        orig_n,
                        &all_edges,
                        &logical_outputs,
                        &node_idx_of,
                        &env.job_dir,
                        env.cache.clone(),
                        env.bypass_cache,
                        &env.cancel,
                        deadline,
                        started,
                        &mut schedule_hints,
                        &mut cache_probes,
                        &mut prepared_cache_hits,
                    )
                    .await
                {
                    first_error.get_or_insert(error);
                    env.cancel.cancel();
                }
                while first_error.is_none()
                    && in_flight.load(std::sync::atomic::Ordering::Relaxed) < max_in_flight
                {
                    // A wide ready set can keep this loop busy across the plan
                    // deadline, and external cancellation can race the probe's
                    // final check. Check before readiness work, then again at
                    // the concrete dispatch/spawn boundaries below.
                    if let Some(error) = plan_stop_error(deadline, started, &env.cancel) {
                        first_error = Some(error);
                        env.cancel.cancel();
                        break;
                    }
                    // Cache-hit first when enabled, then user priority and
                    // critical path; ties retain historical smallest NodeId.
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
                    let node_partition = node.partition.clone();
                    let node_idx = node_idx_of[&node_id];
                    // Per-node kill slot: holds a child of the plan token,
                    // retained in `node_tokens` so the control watcher can fire
                    // it alone (and a divergence retry can re-arm it).
                    let node_cancel = KillSlot::new(env.cancel.child_token());
                    let mut task = match build_task(
                        node,
                        node_idx,
                        &all_edges,
                        &outputs,
                        &logical_outputs,
                        &content_outputs,
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
                    // Ordinary work always wins. If an unrelated ready node has
                    // the same cache identity as private speculative work,
                    // cancel/drop the private copy before normal single-flight
                    // admission. A selected target will later reuse the ordinary
                    // cache transition instead of duplicating the computation.
                    let mut superseded_selected = None;
                    let mut supersession_cleanup_failure = None;
                    if let Some(spec_target) = speculative_keys.remove(&task.key)
                        && let Some(state) = speculation.get_mut(&spec_target)
                    {
                        match state {
                            SpeculationState::Running {
                                cancel,
                                disposition,
                            } => {
                                let was_selected = *disposition == SpeculationDisposition::Selected;
                                cancel.cancel();
                                *disposition = SpeculationDisposition::Superseded;
                                if was_selected {
                                    superseded_selected = Some(spec_target);
                                }
                            }
                            SpeculationState::Prepared(_) => {
                                let prepared =
                                    match std::mem::replace(state, SpeculationState::Declined) {
                                        SpeculationState::Prepared(prepared) => prepared,
                                        _ => unreachable!("matched prepared speculation state"),
                                    };
                                if let Err(failure) = prepared.discard() {
                                    supersession_cleanup_failure = Some(failure);
                                }
                            }
                            SpeculationState::Eligible
                            | SpeculationState::Declined
                            | SpeculationState::Rejected => {}
                        }
                    }
                    if let Some(failure) = supersession_cleanup_failure {
                        first_error = Some(plan_error_of(failure));
                        env.cancel.cancel();
                        break;
                    }
                    // A selected target was deliberately withheld from `ready`
                    // while its private attempt ran. If ordinary work takes
                    // ownership of the same key, put that target back into the
                    // normal single-flight path so it later consumes the
                    // ordinary cache transition instead of vanishing.
                    if let Some(target) = superseded_selected
                        && enabled_condition_targets.contains(&target)
                        && indeg.get(&target).copied() == Some(0)
                        && !pruned.contains(&target)
                    {
                        ready.insert(target);
                    }
                    // Single-flight: if this exact key is already running,
                    // defer until it completes (then it cache-hits).
                    if inflight_keys.contains(&task.key) {
                        deferred.entry(task.key).or_default().push(node_id);
                        continue;
                    }
                    task.prepared_cache_hit = prepared_cache_hits.remove(&node_id);
                    let mut pipeline_launch = None;
                    if control.is_none()
                        // Force-recompute may intentionally overwrite a warm
                        // cache entry. Keep that path ordinary so cancellation
                        // rollback can never delete prior canonical state.
                        && !env.bypass_cache
                        // The coordinator counts the parent only after this
                        // branch. Pipeline overlap also runs one private child,
                        // so reserve two logical task slots in addition to
                        // every sibling already in flight.
                        && in_flight
                            .load(std::sync::atomic::Ordering::Relaxed)
                            .saturating_add(2)
                            <= max_in_flight
                        && ready.is_empty()
                        && task.prepared_cache_hit.is_none()
                        && let Some(expansion) = plan
                            .expansions()
                            .iter()
                            .find(|expansion| expansion.parent == node_id && expansion.pipeline)
                            .cloned()
                    {
                        let cache_cold = env.bypass_cache
                            || matches!(
                                cache_presence_probe_off_thread(env.cache.clone(), task.key).await,
                                Ok(Ok(false))
                            );
                        if cache_cold {
                            match try_prepare_pipeline_launch(
                                &mut task,
                                &expansion,
                                PipelineLaunchContext {
                                    inherited_partition: node_partition,
                                    env: &env,
                                    orig_n,
                                    appended_len: appended.len(),
                                    order_len: order.len(),
                                    inflight_keys: &inflight_keys,
                                    spawn_capacity: MAX_RUNTIME_SPAWNS.saturating_sub(spawns_total),
                                },
                            ) {
                                Ok(PipelineLaunchDecision::Launch(launch)) => {
                                    for child in &launch.children {
                                        inflight_keys.insert(child.key);
                                        node_key_of.insert(child.node.id, child.key);
                                    }
                                    pipeline_reserved_parents.insert(node_id);
                                    pipeline_launch = Some(launch);
                                }
                                Ok(PipelineLaunchDecision::ResolvedFallback(profiles)) => {
                                    debug_assert!(
                                        pipeline_fallback_profiles
                                            .insert(node_id, profiles)
                                            .is_none(),
                                        "one parent may have only one resolved pipeline fallback"
                                    );
                                }
                                Ok(PipelineLaunchDecision::Declined) => {}
                                Err(
                                    SpawnInjectionError::TrainingIo(error)
                                    | SpawnInjectionError::Structural(error),
                                ) => {
                                    first_error = Some(error);
                                    env.cancel.cancel();
                                    break;
                                }
                            }
                        }
                    }
                    inflight_keys.insert(task.key);
                    node_key_of.insert(node_id, task.key);
                    // Retain the kill token only for an actually-spawned node
                    // (a deferred node `continue`s above; its token is dropped
                    // and a fresh one is built when it re-enters `ready`).
                    node_tokens.insert(node_id, node_cancel);
                    // Retain unconditionally: the control watcher uses it for
                    // divergence checks, and the completion path uses it to
                    // preserve advisory-stage semantics even when conditional
                    // control forced the parallel executor without a policy.
                    node_stages.insert(node_id, task.stage.clone());

                    // P2P dispatch: a prepared cache hit is already complete
                    // locally and must reach `run_node`'s skip path. Only a
                    // genuine miss may be offloaded to a peer. A declaring
                    // stage also stays local: DispatchRequest has no checked
                    // TrainingIoProfile wire, so sending it would lose both the
                    // selected bounded policy and its exact retained-byte bill.
                    // The local path emits the profile losslessly before Begin.
                    #[cfg(feature = "p2p")]
                    if task.prepared_cache_hit.is_none()
                        && env.training_io_node(task.node_id).is_none()
                        && let (Some(policy), Some(dispatcher)) =
                            (env.dispatch_policy.as_ref(), env.dispatcher.as_ref())
                    {
                        // Restricted tenant custody dominates a cookbook's data
                        // classification. Even a buggy/custom policy that labels
                        // a clinical stage Public cannot move it off-node.
                        if !env.tenant.is_restricted() && policy.is_dispatchable(task.stage.name())
                        {
                            let args_hash = ContentHash::of_bytes(&task.canon_args);
                            let stage_resources = task.stage.resources();
                            let has_gpu = stage_resources.contains(&Resource::Gpu);
                            let resource_request = ResourceRequest {
                                cpu_cores: task.stage.cpu_cores(),
                                memory_gib: task.stage.memory_gib(),
                                gpu: has_gpu,
                                gpu_vram_gib: None,
                            };
                            let data_class =
                                policy.classify_stage(task.stage.name(), &task.args) as u8;
                            let input_content_id = match capture(
                                task.stage.as_ref(),
                                task.input.clone(),
                                &env.job_dir,
                                ArtifactRole::Input,
                                None,
                            ) {
                                Ok(stored) => Some(stored.manifest.content_id),
                                Err(error) => {
                                    tracing::warn!(
                                        "P2P input capture failed for node {} ({}), running locally: {error}",
                                        node_idx,
                                        task.stage.name()
                                    );
                                    None
                                }
                            };
                            if let Some(input_content_id) = input_content_id {
                                let request = DispatchRequest {
                                    stage_name: task.stage.name(),
                                    stage_schema: task.stage.schema(),
                                    invocation_key: task.key,
                                    input_content_id,
                                    args_hash,
                                    args: &task.args,
                                    expected_content_id: None,
                                    resource_request,
                                    data_class,
                                    tenant: &env.tenant,
                                };
                                if let Some(error) = plan_stop_error(deadline, started, &env.cancel)
                                {
                                    first_error = Some(error);
                                    env.cancel.cancel();
                                    break;
                                }
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
                                        let into_stage_dir = env
                                            .job_dir
                                            .join("stages")
                                            .join(format!("{node_idx}-{stage_name}"));
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
                                                    return match cache_lookup_off_thread(
                                                        cache.clone(),
                                                        key,
                                                        stage.clone(),
                                                        into_stage_dir.clone(),
                                                    )
                                                    .await
                                                    {
                                                        Ok(Some(hit)) => {
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
                                                            status.emit(StageEvent::StageEnd {
                                                                node_idx,
                                                                stage_name: stage_name.clone(),
                                                                content_id: hit.content_id,
                                                                elapsed: start.elapsed(),
                                                            });
                                                            Ok(vec![NodeOutcome {
                                                                node_id,
                                                                output: hit.artifact,
                                                                in_process_output: None,
                                                                content_id: hit.content_id,
                                                                logical,
                                                                cache_hit: false,
                                                            }])
                                                        }
                                                        Ok(None) => {
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
                                                        Err(error) => {
                                                            let msg = format!(
                                                                "P2P cache lookup worker failed for node {node_idx} ({stage_name}): {error}"
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
                                    }
                                    .map(SchedulerTaskResult::Ordinary));
                                        in_flight
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
                    }

                    let env_c = env.clone();
                    if let Some(error) = plan_stop_error(deadline, started, &env.cancel) {
                        first_error = Some(error);
                        env.cancel.cancel();
                        break;
                    }
                    if speculation_runtime_enabled {
                        match AdmissionRequest::for_task(&task, &env) {
                            Ok(request) => {
                                ordinary_admission_demands.insert(node_id, request);
                            }
                            Err(error) => {
                                first_error = Some(PlanError::Other(error));
                                env.cancel.cancel();
                                break;
                            }
                        }
                    }
                    if let Some(launch) = pipeline_launch {
                        let extra = PipelineInFlightExtra::reserve(in_flight.clone());
                        join.spawn(async move {
                            let _extra = extra;
                            SchedulerTaskResult::Pipeline(
                                run_pipeline_parent(task, env_c, launch, deadline, started).await,
                            )
                        });
                    } else if let Some(group) = internal_fusion_groups.get(&node_id).cloned() {
                        let remaining = group
                            .node_ids
                            .iter()
                            .skip(1)
                            .map(|id| (view.nodes[*id as usize].clone(), node_idx_of[id]))
                            .collect();
                        join.spawn(
                            run_fused_group(
                                task,
                                remaining,
                                env_c,
                                group.admission,
                                deadline,
                                started,
                            )
                            .map(SchedulerTaskResult::Ordinary),
                        );
                    } else {
                        join.spawn(
                            async move { run_node(task, env_c).await.map(|outcome| vec![outcome]) }
                                .map(SchedulerTaskResult::Ordinary),
                        );
                    }
                    in_flight.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }

            // Speculation is strictly opportunistic: ordinary ready work is
            // offered first, and only a task slot still spare after exhausting
            // that ready set may host a data-ready private candidate. The
            // optimizer witness is necessary but runtime admission/cache state
            // may still decline without changing plan semantics.
            if first_error.is_none()
                && in_flight.load(std::sync::atomic::Ordering::Relaxed) < max_in_flight
            {
                let candidate = speculation_candidates.iter().copied().find(|target| {
                    matches!(speculation.get(target), Some(SpeculationState::Eligible))
                        && indeg.get(target).copied() == Some(0)
                        && !pruned.contains(target)
                        && !enabled_condition_targets.contains(target)
                });
                if let Some(target) = candidate {
                    let node = node_at(&view, &appended, orig_n, target);
                    let node_idx = node_idx_of[&target];
                    let cancel = env.cancel.child_token();
                    let task = build_task(
                        node,
                        node_idx,
                        &all_edges,
                        &outputs,
                        &logical_outputs,
                        &content_outputs,
                        KillSlot::new(cancel.clone()),
                    );
                    match task {
                        Err(_) => {
                            speculation.insert(target, SpeculationState::Declined);
                        }
                        Ok(task)
                            if inflight_keys.contains(&task.key)
                                || speculative_keys.contains_key(&task.key) =>
                        {
                            speculation.insert(target, SpeculationState::Declined);
                        }
                        Ok(task) => {
                            let key = task.key;
                            let Ok(request) = AdmissionRequest::for_task(&task, &env) else {
                                speculation.insert(target, SpeculationState::Declined);
                                continue;
                            };
                            if !admission_is_spare_after_ordinary(
                                &request,
                                &ordinary_admission_demands,
                                &env.resources,
                                env.gpu.as_ref(),
                                env.memory.as_ref(),
                            ) {
                                speculation.insert(target, SpeculationState::Declined);
                                continue;
                            }
                            let cache_cold = env.bypass_cache
                                || matches!(
                                    cache_presence_probe_off_thread(env.cache.clone(), key).await,
                                    Ok(Ok(false))
                                );
                            if let Some(error) = plan_stop_error(deadline, started, &env.cancel) {
                                speculation.insert(target, SpeculationState::Declined);
                                first_error = Some(error);
                                env.cancel.cancel();
                                continue;
                            }
                            if !cache_cold {
                                speculation.insert(target, SpeculationState::Declined);
                            } else if let Some(lease) = try_acquire_admission_from(
                                &request,
                                &env.resources,
                                env.gpu.as_ref(),
                                &env.memory,
                            ) {
                                let scratch_root = env.job_dir.join(".speculation").join(format!(
                                    "{target}-{}-{}",
                                    key.to_hex(),
                                    SPECULATION_NONCE.fetch_add(1, AtomicOrdering::Relaxed)
                                ));
                                speculation.insert(
                                    target,
                                    SpeculationState::Running {
                                        cancel: cancel.clone(),
                                        disposition: SpeculationDisposition::Pending,
                                    },
                                );
                                speculative_keys.insert(key, target);
                                let env_c = env.clone();
                                join.spawn(async move {
                                    let result =
                                        match std::panic::AssertUnwindSafe(prepare_speculative(
                                            task,
                                            env_c,
                                            cancel,
                                            lease,
                                            scratch_root,
                                        ))
                                        .catch_unwind()
                                        .await
                                        {
                                            Ok(result) => result.map(Box::new),
                                            Err(_) => Err(NodeFailure::Other(format!(
                                                "speculative node task panicked for target {target}"
                                            ))),
                                        };
                                    SchedulerTaskResult::Speculative {
                                        target,
                                        key,
                                        result,
                                    }
                                });
                                in_flight.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            } else {
                                speculation.insert(target, SpeculationState::Declined);
                            }
                        }
                    }
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
                                                        pending_spawns.push(PendingSpawn {
                                                            delta: *delta,
                                                            resolved_training_io: None,
                                                        });
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
            let completion = match joined {
                Some(Ok(completion)) => completion,
                Some(Err(join_err)) => {
                    // Task panicked. Record as the first error, cancel.
                    first_error
                        .get_or_insert(PlanError::Other(format!("node task panicked: {join_err}")));
                    env.cancel.cancel();
                    continue;
                }
                None => break,
            };
            let res = match completion {
                SchedulerTaskResult::Ordinary(result) => result,
                SchedulerTaskResult::Pipeline(result) => match result {
                    Err(failure) => Err(failure),
                    Ok(mut pipeline) => (|| -> Result<Vec<NodeOutcome>, NodeFailure> {
                        let parent_id = pipeline.parent.node_id;
                        // The producer task may finish at the same instant as a
                        // plan deadline/external cancellation. Private spill
                        // state is still disposable here; reject it before any
                        // dormant child topology or canonical output becomes
                        // visible.
                        if let Some(error) = plan_stop_error(deadline, started, &env.cancel) {
                            let cleanup =
                                pipeline.prepared.take().map_or(Ok(()), |mut prepared| {
                                    discard_pipeline_prepared(&mut prepared)
                                });
                            return match cleanup {
                                Ok(()) => Err(NodeFailure::Plan(error)),
                                Err(failure) => Err(failure),
                            };
                        }
                        let Some(spilled) = pipeline.prepared.take() else {
                            // No private batch survived validation. Abandon
                            // every predicted key through the full
                            // single-flight wake-up seam before recomputing
                            // canonical identities from the authoritative list.
                            for child in &pipeline.children {
                                abandon_inflight_key(
                                    child.node.id,
                                    child.key,
                                    &mut inflight_keys,
                                    &mut node_key_of,
                                    &mut deferred,
                                    &mut cache_probes,
                                    &mut prepared_cache_hits,
                                    &mut ready,
                                    &pruned,
                                );
                            }

                            let actual_len = pipeline.elements.len();
                            if actual_len > pipeline.spawn_capacity {
                                pipeline_reserved_parents.remove(&parent_id);
                                return Err(NodeFailure::Plan(PlanError::Other(format!(
                                    "map fan-out exceeded the runtime spawn cap \
                                     ({MAX_RUNTIME_SPAWNS} nodes); a dropped shard would be a wrong answer"
                                ))));
                            }
                            let prototype =
                                pipeline.children.first().cloned().ok_or_else(|| {
                                    NodeFailure::Other(
                                        "non-empty pipeline manifest reserved no child template"
                                            .into(),
                                    )
                                })?;
                            let base_id = prototype.node.id;
                            let base_idx = prototype.node_idx;
                            let profile_prototype = pipeline
                                .profiles
                                .first()
                                .map(|(_, selected)| selected.clone());
                            pipeline.children = (0..actual_len)
                                .map(|index| {
                                    let offset = index as NodeId;
                                    let mut child = prototype.clone();
                                    child.node.id = base_id + offset;
                                    child.node_idx = base_idx + offset;
                                    child.input_logical =
                                        map_element_logical(&pipeline.parent.logical, index);
                                    child.key = node_cache_key(&child.node, child.input_logical);
                                    child
                                })
                                .collect();
                            pipeline.profiles =
                                profile_prototype.map_or_else(Vec::new, |selected| {
                                    pipeline
                                        .children
                                        .iter()
                                        .map(|child| (child.node.id, selected.clone()))
                                        .collect()
                                });
                            let children = commit_pipeline_children(
                                PreparedPipelineChildren {
                                    children: pipeline.children,
                                    profiles: pipeline.profiles,
                                },
                                &env,
                                &mut appended,
                                &mut order,
                                &mut node_idx_of,
                                &mut indeg,
                                &mut succs,
                            )
                            .map_err(|error| match error {
                                SpawnInjectionError::Structural(error)
                                | SpawnInjectionError::TrainingIo(error) => {
                                    NodeFailure::Plan(error)
                                }
                            })?;
                            spawns_total += children.len();
                            for (child, element) in children.iter().zip(pipeline.elements) {
                                outputs.insert(child.node.id, element);
                                logical_outputs.insert(child.node.id, child.input_logical);
                                ready.insert(child.node.id);
                            }
                            return Ok(vec![pipeline.parent]);
                        };
                        let mut spilled = VecDeque::from(spilled);

                        // `parent.output` retains the authoritative ListOf just
                        // as ordinary fan-out does. The separately decoded
                        // element vector was needed only for manifest/input
                        // validation in `run_pipeline_parent`; release it before
                        // child outputs begin accumulating so publication
                        // replaces input residency instead of retaining both
                        // complete batches.
                        drop(std::mem::take(&mut pipeline.elements));

                        // Validate every private payload before committing the
                        // dynamic nodes, but drop each decoded record before
                        // opening the next. This preserves ordinary fallback
                        // for pre-publication corruption without retaining an
                        // O(manifest cardinality) duplicate batch outside the
                        // canonical output map.
                        let load_failure = spilled
                            .iter()
                            .find_map(|child| child.validate_payload().err());
                        if let Some(load_failure) = load_failure {
                            for child in &pipeline.children {
                                abandon_inflight_key(
                                    child.node.id,
                                    child.key,
                                    &mut inflight_keys,
                                    &mut node_key_of,
                                    &mut deferred,
                                    &mut cache_probes,
                                    &mut prepared_cache_hits,
                                    &mut ready,
                                    &pruned,
                                );
                            }
                            pipeline_reserved_parents.remove(&parent_id);
                            let mut remaining = Vec::from(spilled);
                            discard_pipeline_prepared(&mut remaining)?;
                            allow_pipeline_load_fallback(load_failure)?;
                            pipeline_fallback_profiles
                                .insert(parent_id, local_pipeline_profiles(&pipeline.profiles));
                            Ok(vec![pipeline.parent])
                        } else {
                            // Validation is synchronous and may be substantial.
                            // Recheck immediately before topology commit;
                            // cancellation during validation must not publish a
                            // canonical child.
                            if let Some(error) = plan_stop_error(deadline, started, &env.cancel) {
                                for child in &pipeline.children {
                                    abandon_inflight_key(
                                        child.node.id,
                                        child.key,
                                        &mut inflight_keys,
                                        &mut node_key_of,
                                        &mut deferred,
                                        &mut cache_probes,
                                        &mut prepared_cache_hits,
                                        &mut ready,
                                        &pruned,
                                    );
                                }
                                pipeline_reserved_parents.remove(&parent_id);
                                let mut remaining = Vec::from(spilled);
                                return match discard_pipeline_prepared(&mut remaining) {
                                    Ok(()) => Err(NodeFailure::Plan(error)),
                                    Err(failure) => Err(failure),
                                };
                            }
                            let child_count = pipeline.children.len();
                            let committed = commit_pipeline_children(
                                PreparedPipelineChildren {
                                    children: pipeline.children,
                                    profiles: pipeline.profiles,
                                },
                                &env,
                                &mut appended,
                                &mut order,
                                &mut node_idx_of,
                                &mut indeg,
                                &mut succs,
                            );
                            if let Err(error) = committed {
                                let mut remaining = Vec::from(spilled);
                                discard_pipeline_prepared(&mut remaining)?;
                                let error = match error {
                                    SpawnInjectionError::Structural(error)
                                    | SpawnInjectionError::TrainingIo(error) => error,
                                };
                                return Err(NodeFailure::Plan(error));
                            }
                            spawns_total += child_count;
                            let mut outcomes = Vec::with_capacity(child_count + 1);
                            outcomes.push(pipeline.parent);
                            while !spilled.is_empty() {
                                if let Some(error) = plan_stop_error(deadline, started, &env.cancel)
                                {
                                    let mut remaining = Vec::from(spilled);
                                    return match discard_pipeline_prepared(&mut remaining) {
                                        Ok(()) => Err(NodeFailure::Plan(error)),
                                        Err(failure) => Err(failure),
                                    };
                                }
                                let child = spilled
                                    .pop_front()
                                    .expect("pipeline spill queue was non-empty");
                                // The prevalidation pass makes corruption here
                                // an integrity failure after topology commit,
                                // not an optional fallback. Rehydrate only this
                                // child so resident private payload stays O(1).
                                let prepared = match child.into_resident() {
                                    Ok(prepared) => prepared,
                                    Err(failure) => {
                                        let mut remaining = Vec::from(spilled);
                                        discard_pipeline_prepared(&mut remaining)?;
                                        return Err(failure);
                                    }
                                };
                                let child_id = prepared.node_id;
                                match publish_speculative(prepared, &env, deadline, started, true) {
                                    Ok(outcome) if outcome.node_id == child_id => {
                                        outcomes.push(outcome);
                                    }
                                    Ok(outcome) => {
                                        let mut remaining = Vec::from(spilled);
                                        discard_pipeline_prepared(&mut remaining)?;
                                        return Err(NodeFailure::Other(format!(
                                            "pipeline child publication changed node id {child_id} to {}",
                                            outcome.node_id
                                        )));
                                    }
                                    Err(failure) => {
                                        let mut remaining = Vec::from(spilled);
                                        discard_pipeline_prepared(&mut remaining)?;
                                        return Err(failure);
                                    }
                                }
                            }
                            Ok(outcomes)
                        }
                    })(),
                },
                SchedulerTaskResult::Speculative {
                    target,
                    key,
                    result,
                } => {
                    let state = speculation
                        .remove(&target)
                        .unwrap_or(SpeculationState::Declined);
                    match state {
                        SpeculationState::Running {
                            disposition: SpeculationDisposition::Pending,
                            ..
                        } => match result {
                            Ok(prepared) => {
                                speculation.insert(target, SpeculationState::Prepared(prepared));
                                continue;
                            }
                            Err(failure @ NodeFailure::SpeculationCleanup { .. }) => {
                                if speculative_keys.get(&key).copied() == Some(target) {
                                    speculative_keys.remove(&key);
                                }
                                Err(failure)
                            }
                            Err(_) => {
                                if speculative_keys.get(&key).copied() == Some(target) {
                                    speculative_keys.remove(&key);
                                }
                                speculation.insert(target, SpeculationState::Declined);
                                continue;
                            }
                        },
                        SpeculationState::Running {
                            disposition: SpeculationDisposition::Selected,
                            ..
                        } => {
                            if speculative_keys.get(&key).copied() == Some(target) {
                                speculative_keys.remove(&key);
                            }
                            match result {
                                Ok(prepared) => {
                                    if let Some(error) =
                                        plan_stop_error(deadline, started, &env.cancel)
                                    {
                                        match prepared.discard() {
                                            Ok(()) => Err(NodeFailure::Plan(error)),
                                            Err(failure) => Err(failure),
                                        }
                                    } else {
                                        match publish_speculative(
                                            *prepared, &env, deadline, started, false,
                                        ) {
                                            Ok(outcome) => {
                                                node_key_of.insert(target, key);
                                                Ok(vec![outcome])
                                            }
                                            Err(error) => Err(error),
                                        }
                                    }
                                }
                                Err(failure @ NodeFailure::SpeculationCleanup { .. }) => {
                                    Err(failure)
                                }
                                Err(_) => {
                                    speculation.insert(target, SpeculationState::Declined);
                                    if indeg.get(&target).copied() == Some(0)
                                        && !pruned.contains(&target)
                                    {
                                        ready.insert(target);
                                    }
                                    continue;
                                }
                            }
                        }
                        SpeculationState::Running {
                            disposition: SpeculationDisposition::Rejected,
                            ..
                        }
                        | SpeculationState::Rejected => {
                            if speculative_keys.get(&key).copied() == Some(target) {
                                speculative_keys.remove(&key);
                            }
                            speculation.insert(target, SpeculationState::Rejected);
                            match discard_speculation_result(result) {
                                Ok(()) => continue,
                                Err(failure) => Err(failure),
                            }
                        }
                        SpeculationState::Running {
                            disposition: SpeculationDisposition::Superseded,
                            ..
                        } => {
                            if speculative_keys.get(&key).copied() == Some(target) {
                                speculative_keys.remove(&key);
                            }
                            speculation.insert(target, SpeculationState::Declined);
                            match discard_speculation_result(result) {
                                Ok(()) => continue,
                                Err(failure) => Err(failure),
                            }
                        }
                        SpeculationState::Prepared(prepared) => {
                            if speculative_keys.get(&key).copied() == Some(target) {
                                speculative_keys.remove(&key);
                            }
                            let retained_cleanup = prepared.discard();
                            let result_cleanup = discard_speculation_result(result);
                            match retained_cleanup.and(result_cleanup) {
                                Ok(()) => continue,
                                Err(failure) => Err(failure),
                            }
                        }
                        SpeculationState::Eligible | SpeculationState::Declined => {
                            if speculative_keys.get(&key).copied() == Some(target) {
                                speculative_keys.remove(&key);
                            }
                            match discard_speculation_result(result) {
                                Ok(()) => continue,
                                Err(failure) => Err(failure),
                            }
                        }
                    }
                }
            };

            match res {
                Ok(outcomes) => {
                    let mut outcomes = VecDeque::from(outcomes);
                    while let Some(outcome) = outcomes.pop_front() {
                        ordinary_admission_demands.remove(&outcome.node_id);
                        completed += 1;
                        let was_cache_hit = outcome.cache_hit;
                        if was_cache_hit {
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
                        content_outputs.insert(outcome.node_id, outcome.content_id);

                        // Release any nodes deferred behind this key — they
                        // can now cache-hit. Re-add them to the ready set.
                        if let Some(k) = key {
                            // Completion is the only relevant cache transition for
                            // this exact key. Always invalidate it: even a cache hit
                            // may have raced a prior scheduling miss, and retaining
                            // that stale `None` could misorder or remotely dispatch
                            // a same-key waiter. Unrelated keys remain memoized.
                            cache_probes.remove(&k);
                            inflight_keys.remove(&k);
                            if let Some(waiters) = deferred.remove(&k) {
                                for w in waiters {
                                    ready.insert(w);
                                }
                            }
                        }

                        // Decrement successors' in-degrees; newly-zero → ready.
                        if first_error.is_none()
                            && let Some(ss) = succs.get(&outcome.node_id)
                        {
                            for &s in ss {
                                if let Some(d) = indeg.get_mut(&s) {
                                    *d -= 1;
                                    let condition_allows = !condition_by_target.contains_key(&s)
                                        || enabled_condition_targets.contains(&s);
                                    if *d == 0
                                        && condition_allows
                                        && !fused_internal_nodes.contains(&s)
                                    {
                                        ready.insert(s);
                                    }
                                }
                            }
                        }

                        // Resolve boolean control only after the selector's
                        // ordinary artifact has been published. A matching
                        // target becomes schedulable once its data inputs are
                        // ready; a losing target and every data descendant are
                        // accounted as pruned without running or emitting a
                        // lifecycle event. The control relation never enters
                        // `all_edges`, `build_task`, or the node cache key.
                        if first_error.is_none()
                            && let Some(targets) = condition_targets.get(&outcome.node_id)
                        {
                            let decision = outputs
                                .get(&outcome.node_id)
                                .cloned()
                                .expect("completed selector output was just inserted")
                                .into_typed::<BranchDecision>();
                            match decision {
                                Ok(decision) => {
                                    for &(target, when) in targets {
                                        if decision.value == when {
                                            enabled_condition_targets.insert(target);
                                            let mut schedule_ordinary = true;
                                            if let Some(state) = speculation.remove(&target) {
                                                match state {
                                                    SpeculationState::Prepared(prepared) => {
                                                        let key = prepared.key;
                                                        if speculative_keys.get(&key).copied()
                                                            == Some(target)
                                                        {
                                                            speculative_keys.remove(&key);
                                                        }
                                                        if let Some(error) = plan_stop_error(
                                                            deadline,
                                                            started,
                                                            &env.cancel,
                                                        ) {
                                                            first_error =
                                                                Some(match prepared.discard() {
                                                                    Ok(()) => error,
                                                                    Err(failure) => {
                                                                        plan_error_of(failure)
                                                                    }
                                                                });
                                                            env.cancel.cancel();
                                                            break;
                                                        }
                                                        match publish_speculative(
                                                            *prepared, &env, deadline, started,
                                                            false,
                                                        ) {
                                                            Ok(outcome) => {
                                                                node_key_of.insert(target, key);
                                                                outcomes.push_back(outcome);
                                                                schedule_ordinary = false;
                                                            }
                                                            Err(error) => {
                                                                first_error =
                                                                    Some(plan_error_of(error));
                                                                env.cancel.cancel();
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    SpeculationState::Running {
                                                        cancel,
                                                        disposition: SpeculationDisposition::Pending,
                                                    } => {
                                                        speculation.insert(
                                                            target,
                                                            SpeculationState::Running {
                                                                cancel,
                                                                disposition:
                                                                    SpeculationDisposition::Selected,
                                                            },
                                                        );
                                                        schedule_ordinary = false;
                                                    }
                                                    SpeculationState::Running {
                                                        cancel,
                                                        disposition,
                                                    } => {
                                                        speculation.insert(
                                                            target,
                                                            SpeculationState::Running {
                                                                cancel,
                                                                disposition,
                                                            },
                                                        );
                                                        schedule_ordinary = disposition
                                                            == SpeculationDisposition::Superseded;
                                                    }
                                                    SpeculationState::Eligible
                                                    | SpeculationState::Declined
                                                    | SpeculationState::Rejected => {
                                                        speculation.insert(
                                                            target,
                                                            SpeculationState::Declined,
                                                        );
                                                    }
                                                }
                                            }
                                            if schedule_ordinary
                                                && indeg.get(&target).copied() == Some(0)
                                                && !pruned.contains(&target)
                                            {
                                                ready.insert(target);
                                            }
                                        } else {
                                            if let Some(state) = speculation.remove(&target) {
                                                match state {
                                                    SpeculationState::Running {
                                                        cancel, ..
                                                    } => {
                                                        cancel.cancel();
                                                        speculation.insert(
                                                            target,
                                                            SpeculationState::Running {
                                                                cancel,
                                                                disposition:
                                                                    SpeculationDisposition::Rejected,
                                                            },
                                                        );
                                                    }
                                                    SpeculationState::Prepared(prepared) => {
                                                        let key = prepared.key;
                                                        if speculative_keys.get(&key).copied()
                                                            == Some(target)
                                                        {
                                                            speculative_keys.remove(&key);
                                                        }
                                                        if let Err(failure) = prepared.discard() {
                                                            first_error =
                                                                Some(plan_error_of(failure));
                                                            env.cancel.cancel();
                                                            break;
                                                        }
                                                        speculation.insert(
                                                            target,
                                                            SpeculationState::Rejected,
                                                        );
                                                    }
                                                    SpeculationState::Eligible
                                                    | SpeculationState::Declined
                                                    | SpeculationState::Rejected => {
                                                        speculation.insert(
                                                            target,
                                                            SpeculationState::Rejected,
                                                        );
                                                    }
                                                }
                                            }
                                            let reason = format!(
                                                "condition node {} resolved to {}, expected {}",
                                                outcome.node_id, decision.value, when
                                            );
                                            let mut stack = vec![target];
                                            while let Some(node_id) = stack.pop() {
                                                if pruned.insert(node_id) {
                                                    ready.remove(&node_id);
                                                    // Graph-input nodes are pre-seeded with their
                                                    // INPUT envelope under the same id. Remove it
                                                    // when the branch is rejected so final-output
                                                    // selection cannot mistake an unexecuted root's
                                                    // unit input for a produced artifact.
                                                    outputs.remove(&node_id);
                                                    logical_outputs.remove(&node_id);
                                                    if let Some(&node_idx) =
                                                        node_idx_of.get(&node_id)
                                                    {
                                                        let node = node_at(
                                                            &view, &appended, orig_n, node_id,
                                                        );
                                                        env.status.emit(StageEvent::StagePruned {
                                                            node_idx,
                                                            stage_name: node
                                                                .stage
                                                                .name()
                                                                .to_string(),
                                                            reason: reason.clone(),
                                                        });
                                                    }
                                                    if let Some(children) = succs.get(&node_id) {
                                                        stack.extend(children.iter().copied());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(error) => {
                                    first_error = Some(PlanError::Other(format!(
                                        "condition node {} produced an invalid BranchDecision: {error}",
                                        outcome.node_id
                                    )));
                                    env.cancel.cancel();
                                }
                            }
                        }

                        // ADR 0078 `map_output`: if this node drives a fan-out,
                        // queue one template instance per list element. This runs
                        // on the LOSSLESS completion seam (the output is promoted
                        // into `outputs` above) — never a dropped step event — so a
                        // fan-out can't be missed. The deltas are injected at the
                        // top of the next loop iteration (the single-threaded
                        // schedule-mutation seam) via `pending_spawns`.
                        if first_error.is_none()
                            && !pipeline_reserved_parents.contains(&outcome.node_id)
                        {
                            let mut resolved_pipeline_fallback =
                                pipeline_fallback_profiles.remove(&outcome.node_id);
                            for exp in plan.expansions() {
                                if exp.parent != outcome.node_id {
                                    continue;
                                }
                                // The parent's output + logical hash were promoted
                                // into the maps immediately above; a miss is an
                                // internal invariant break, not a silent skip (a
                                // fallback would collide unrelated fan-outs' cache
                                // keys).
                                let (Some(list_env), Some(parent_logical)) = (
                                    outputs.get(&outcome.node_id).cloned(),
                                    logical_outputs.get(&outcome.node_id).copied(),
                                ) else {
                                    first_error = Some(PlanError::Other(format!(
                                        "map over node {}: parent output/logical-hash missing after \
                                     its completion (internal invariant)",
                                        outcome.node_id
                                    )));
                                    break;
                                };
                                // Decode the parent's `list` output into its element
                                // artifacts (erased — the executor doesn't know the
                                // concrete element type).
                                let elements =
                                    match crate::framework::artifact::decode_list_children(list_env)
                                    {
                                        Ok(v) => v,
                                        Err(e) => {
                                            first_error = Some(PlanError::Other(format!(
                                                "map over node {}: parent output is not a valid list: {e}",
                                                outcome.node_id
                                            )));
                                            break;
                                        }
                                    };
                                let base_label = exp.label.clone().unwrap_or_else(|| "map".into());
                                let resolved_training_io = if exp.pipeline {
                                    resolved_pipeline_fallback.take()
                                } else {
                                    None
                                };
                                for (i, elem) in elements.into_iter().enumerate() {
                                    // Invariant: each element's kind is the element
                                    // kind the template was compiled against (the
                                    // parent produced `ListOf<Item>` where
                                    // `Item::KIND == elem_kind`). Cheap guard
                                    // against a producer/template kind drift.
                                    debug_assert_eq!(
                                        elem.kind, exp.template.elem_kind,
                                        "map element kind must match the template's element kind"
                                    );
                                    let label = format!("{base_label}[{i}]");
                                    let subplan = exp.template.instantiate(label.clone());
                                    let elem_logical = map_element_logical(&parent_logical, i);
                                    pending_spawns.push(PendingSpawn {
                                        delta: crate::framework::control::SpawnDelta {
                                            subplan,
                                            label: Some(label),
                                            root_seeds: vec![(
                                                exp.template.root,
                                                elem,
                                                elem_logical,
                                            )],
                                            provenance_parent: Some(outcome.node_id),
                                        },
                                        resolved_training_io: resolved_training_io.clone(),
                                    });
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
                    ordinary_admission_demands.remove(&node_id);
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
                            // Graph-input roots carry a prelude seed under their
                            // own id. Once control pruning reaches one, that seed
                            // is not a produced result and must not survive into
                            // final-output selection.
                            outputs.remove(&d);
                            logical_outputs.remove(&d);
                            if let Some(ss) = succs.get(&d) {
                                stack.extend(ss.iter().copied());
                            }
                            if let Some(targets) = condition_targets.get(&d) {
                                stack.extend(targets.iter().map(|(target, _)| *target));
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
                    if let NodeFailure::Stage { idx, stage, source } = &f
                        && let Some(&nid) = order.get(*idx as usize)
                    {
                        let is_adv = node_stages
                            .get(&nid)
                            .map(|s| s.is_advisory())
                            .unwrap_or(false)
                            && !strict_advisory();
                        node_tokens.remove(&nid);
                        node_stages.remove(&nid);
                        ordinary_admission_demands.remove(&nid);
                        kill_flagged.remove(&nid);
                        if is_adv {
                            advisory = Some((*idx, stage.clone(), source.to_string(), nid));
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
                                outputs.remove(&d);
                                logical_outputs.remove(&d);
                                if let Some(ss) = succs.get(&d) {
                                    stack.extend(ss.iter().copied());
                                }
                                if let Some(targets) = condition_targets.get(&d) {
                                    stack.extend(targets.iter().map(|(target, _)| *target));
                                }
                            }
                        }
                    } else if matches!(
                        &f,
                        NodeFailure::SpeculationCleanup { .. }
                            | NodeFailure::SpeculationRollback { .. }
                    ) {
                        let cleanup_error = plan_error_of(f);
                        first_error = Some(match first_error.take() {
                            Some(error) => PlanError::Other(format!(
                                "{error}; additionally, speculative cleanup failed: {cleanup_error}"
                            )),
                            None => cleanup_error,
                        });
                        env.cancel.cancel();
                    } else if first_error.is_none() {
                        first_error = Some(plan_error_of(f));
                        env.cancel.cancel(); // fail-fast: cancel siblings
                    }
                    // else: a later error after we've already started
                    // failing — drop it; the first error wins.
                }
            }
        }

        // All tasks drained. Prepared optional work has no JoinSet entry, so it
        // must be closed explicitly on both success and failure paths. If an
        // earlier error exists, preserve it in the aggregate while making the
        // cleanup integrity failure visible to the caller.
        if let Err(failure) = discard_retained_speculation(&mut speculation, &mut speculative_keys)
        {
            let cleanup_error = plan_error_of(failure);
            first_error = Some(match first_error.take() {
                Some(error) => PlanError::Other(format!(
                    "{error}; additionally, speculative cleanup failed: {cleanup_error}"
                )),
                None => cleanup_error,
            });
        }

        if let Some(err) = first_error {
            return Err(finish_writer_after_error(env, writer_handle, err).await);
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
            conditional_terminal
                .or_else(|| order.last().copied())
                .and_then(|id| outputs.remove(&id))
        } else if let Some(candidates) = conditional_output_candidates.as_ref() {
            order
                .iter()
                .rev()
                .filter(|id| candidates.contains(id))
                .find_map(|id| outputs.remove(id))
        } else {
            order.iter().rev().find_map(|id| outputs.remove(id))
        };
        finish_writer(env, writer_handle).await?;

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

/// Logical hash for map element `i` (ADR 0078): derived from the parent's
/// LOGICAL hash + the index, domain-separated. Using the parent's logical
/// hash (which is the synthesized-stable fingerprint for a nondeterministic
/// parent) rather than the element's raw content keeps a shard's downstream
/// cache key stable across reruns even when the sharder isn't deterministic —
/// matching the executor's existing DETERMINISTIC=false discipline.
fn map_element_logical(parent_logical: &ContentHash, i: usize) -> ContentHash {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(b"map-elem");
    h.update(parent_logical.0);
    h.update((i as u64).to_le_bytes());
    ContentHash(h.finalize().into())
}

// The test suite (~3100 lines of fixture stages + integration tests) lives in
// executor_tests.rs, mounted here as a child module so it keeps access to the
// executor's private items. Rationale for the file-level clippy allow is
// documented at the top of that file.
#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
