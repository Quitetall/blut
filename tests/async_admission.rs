// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0103 bounded async-I/O admission gate.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use blut::framework::artifact::{Artifact, ContentHash};
use blut::framework::async_io::{
    IoMode, TrainingIoAdmissionError, TrainingIoCandidate, TrainingIoDowngradeReason,
    TrainingIoHints, select_training_io_profile,
};
use blut::framework::cache::CacheHandle;
use blut::framework::control::{Control, ControlPolicy, SpawnDelta, StepMetrics};
use blut::framework::cookbook::{Cookbook, Registry};
use blut::framework::error::StageError;
#[cfg(feature = "p2p")]
use blut::framework::executor::{DispatchRequest, DispatchSubmitter};
use blut::framework::executor::{ExecCtx, ParallelExecutor, execute_plan};
use blut::framework::plan_spec::{PLAN_SPEC_VERSION, PlanSpec, SpecNode};
use blut::framework::resource::Resource;
use blut::framework::stage::{ErasedStageCtor, Stage, StageContext};
use blut::framework::status::StageEvent;
#[cfg(feature = "p2p")]
use blut::p2p::dispatch::{DispatchPolicy, DispatchVerdict};
#[cfg(feature = "p2p")]
use blut::p2p::peer::{PeerId, PeerInfo};
#[cfg(feature = "p2p")]
use blut::p2p::task::{ResourceRequest as P2pResourceRequest, TaskResult};
#[cfg(feature = "p2p")]
use blut::p2p::trust::DataClass;
use blut::recipes::recipe::RecipeDef;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const MIB: u64 = 1024 * 1024;
static PROFILE_STAGE_RAN: AtomicBool = AtomicBool::new(false);
static DDP_STAGE_RAN: AtomicBool = AtomicBool::new(false);
static LEGACY_STAGE_RAN: AtomicBool = AtomicBool::new(false);
static PROFILE_CANDIDATE_CALLS: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn inline_candidate() -> TrainingIoCandidate {
    TrainingIoCandidate {
        data_replicas: 1,
        decode_workers: 0,
        prefetch_per_worker: 0,
        cuda_staging_slots: 0,
        metrics: IoMode::Inline,
        checkpoints: IoMode::Inline,
        batch_bytes: Some(0),
        checkpoint_snapshot_bytes: Some(0),
        fixed_overhead_bytes: Some(0),
    }
}

fn bounded_candidate(batch_bytes: Option<u64>) -> TrainingIoCandidate {
    TrainingIoCandidate {
        data_replicas: 1,
        decode_workers: 2,
        prefetch_per_worker: 2,
        cuda_staging_slots: 1,
        metrics: IoMode::Bounded {
            capacity: 2,
            max_item_bytes: 5 * MIB,
        },
        checkpoints: IoMode::Bounded {
            capacity: 1,
            max_item_bytes: 20 * MIB,
        },
        batch_bytes,
        checkpoint_snapshot_bytes: Some(20 * MIB),
        fixed_overhead_bytes: Some(0),
    }
}

#[test]
fn selects_first_complete_profile_that_fits_checked_byte_envelope() {
    let fastest = bounded_candidate(Some(10 * MIB)); // 80 MiB retained.
    let reduced = TrainingIoCandidate {
        data_replicas: 1,
        decode_workers: 1,
        prefetch_per_worker: 1,
        cuda_staging_slots: 0,
        metrics: IoMode::Inline,
        checkpoints: IoMode::Inline,
        batch_bytes: Some(10 * MIB),
        checkpoint_snapshot_bytes: Some(20 * MIB),
        fixed_overhead_bytes: Some(0),
    }; // 10 MiB retained.
    let candidates = [fastest, reduced, inline_candidate()];

    let selected = select_training_io_profile(100 * MIB, 190 * MIB, &candidates, false)
        .expect("fast profile fits");
    assert_eq!(selected.billed_overhead_bytes, 80 * MIB);
    assert_eq!(selected.decode_workers, 2);
    assert_eq!(selected.downgrade_reason, None);

    let selected = select_training_io_profile(100 * MIB, 120 * MIB, &candidates, false)
        .expect("reduced profile fits");
    assert_eq!(selected.billed_overhead_bytes, 10 * MIB);
    assert_eq!(selected.decode_workers, 1);
    assert_eq!(
        selected.downgrade_reason,
        Some(TrainingIoDowngradeReason::BudgetPressure)
    );
}

#[test]
fn unknown_overflow_and_force_inline_choose_explicit_inline_tail() {
    let mut overflow = bounded_candidate(Some(u64::MAX));
    overflow.decode_workers = u32::MAX;
    let unknown = bounded_candidate(None);
    let candidates = [unknown, overflow.clone(), inline_candidate()];

    let unknown_selected =
        select_training_io_profile(1, u64::MAX, &candidates, false).expect("inline fallback");
    assert!(unknown_selected.is_inline());
    assert_eq!(
        unknown_selected.downgrade_reason,
        Some(TrainingIoDowngradeReason::UnknownSize)
    );

    let zero_sized = bounded_candidate(Some(0));
    let zero_selected =
        select_training_io_profile(1, u64::MAX, &[zero_sized, inline_candidate()], false)
            .expect("zero retained-batch measurement falls back");
    assert!(zero_selected.is_inline());
    assert_eq!(
        zero_selected.downgrade_reason,
        Some(TrainingIoDowngradeReason::UnknownSize)
    );

    let overflow_candidates = [overflow, inline_candidate()];
    let overflow_selected = select_training_io_profile(1, u64::MAX, &overflow_candidates, false)
        .expect("overflow falls back");
    assert!(overflow_selected.is_inline());
    assert_eq!(
        overflow_selected.downgrade_reason,
        Some(TrainingIoDowngradeReason::ArithmeticOverflow)
    );

    let forced = select_training_io_profile(
        100 * MIB,
        500 * MIB,
        &[bounded_candidate(Some(10 * MIB)), inline_candidate()],
        true,
    )
    .expect("forced inline");
    assert!(forced.is_inline());
    assert_eq!(
        forced.downgrade_reason,
        Some(TrainingIoDowngradeReason::UserForced)
    );

    let mut dishonest_inline = inline_candidate();
    dishonest_inline.fixed_overhead_bytes = Some(1);
    assert_eq!(
        select_training_io_profile(0, MIB, &[dishonest_inline], false),
        Err(TrainingIoAdmissionError::InvalidInlineFallback)
    );
}

#[test]
fn refuses_when_synchronous_base_does_not_fit() {
    let error = select_training_io_profile(101, 100, &[inline_candidate()], false)
        .expect_err("base envelope must not be clamped");
    assert!(error.to_string().contains("synchronous base"));
}

#[test]
fn bills_data_pipeline_replicas_and_fixed_worker_rss_without_changing_worker_env() {
    let mut ddp = bounded_candidate(Some(10 * MIB));
    ddp.data_replicas = 2;
    ddp.fixed_overhead_bytes = Some(5 * MIB);
    let selected =
        select_training_io_profile(100 * MIB, 300 * MIB, &[ddp, inline_candidate()], false)
            .expect("replicated profile fits");
    // data queues: 2 replicas * (4 prefetched + 1 CUDA) * 10 MiB =
    // 100 MiB; metrics 10 MiB + checkpoint 20 MiB + fixed 5 MiB/rank.
    assert_eq!(selected.billed_overhead_bytes, 140 * MIB);
    assert_eq!(selected.decode_workers, 2, "worker env stays per rank");
    assert_eq!(selected.data_replicas, 2);
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct Args;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Value(u32);

impl Artifact for Value {
    const KIND: &'static str = "async-admission.value";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&self.0.to_le_bytes())
    }

    fn primary_path(&self) -> &Path {
        Path::new("")
    }
}

struct ProfileStage;

#[async_trait]
impl Stage for ProfileStage {
    const NAME: &'static str = "async_profile_stage";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const MEMORY_GIB: u32 = 1;
    type Input = ();
    type Output = Value;
    type Args = Args;

    fn training_io_sync_base_bytes(&self, _args: &Self::Args, hints: TrainingIoHints) -> u64 {
        if hints.admitted_batch_size == Some(1) {
            900 * MIB
        } else {
            1024 * MIB
        }
    }

    fn training_io_candidates(
        &self,
        _args: &Self::Args,
        hints: TrainingIoHints,
    ) -> Vec<TrainingIoCandidate> {
        PROFILE_CANDIDATE_CALLS.fetch_add(1, Ordering::SeqCst);
        let mut bounded = bounded_candidate(Some(10 * MIB));
        bounded.decode_workers = hints.admitted_decode_workers.unwrap_or(2);
        vec![bounded, inline_candidate()]
    }

    async fn run(&self, ctx: &StageContext, _input: (), _args: &Args) -> Result<Value, StageError> {
        let profile = ctx
            .training_io_profile
            .as_ref()
            .expect("declaring stage receives one concrete profile");
        if ctx.admitted_batch_size == Some(1) {
            assert!(profile.is_inline());
            assert_eq!(profile.sync_base_bytes, 1100 * MIB);
            PROFILE_STAGE_RAN.store(true, Ordering::SeqCst);
            return Ok(Value(7));
        }
        if profile.is_inline() {
            assert!(matches!(
                profile.downgrade_reason,
                Some(TrainingIoDowngradeReason::UserForced)
                    | Some(TrainingIoDowngradeReason::SnapshotUnavailable)
                    | Some(TrainingIoDowngradeReason::UnsupportedLauncher)
            ));
            PROFILE_STAGE_RAN.store(true, Ordering::SeqCst);
            return Ok(Value(7));
        }
        assert_eq!(profile.decode_workers, 3);
        assert_eq!(ctx.admitted_workers, Some(3));
        assert_eq!(ctx.admitted_batch_size, None);
        assert!(!ctx.fb_warm);
        assert_eq!(
            profile.sync_base_bytes,
            1024 * MIB,
            "an injected trial uses its own stage base, not its parent's whole-job floor"
        );
        assert_eq!(profile.billed_overhead_bytes, 100 * MIB);
        let env = profile.env_pairs();
        assert_eq!(
            env.get("BLUT_IO_METRICS_MODE").map(String::as_str),
            Some("bounded")
        );
        assert_eq!(
            env.get("BLUT_IO_BILLED_OVERHEAD_BYTES").map(String::as_str),
            Some("104857600")
        );
        PROFILE_STAGE_RAN.store(true, Ordering::SeqCst);
        Ok(Value(7))
    }
}

struct LegacyStage;

#[async_trait]
impl Stage for LegacyStage {
    const NAME: &'static str = "async_legacy_stage";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Value;
    type Args = Args;

    async fn run(&self, ctx: &StageContext, _input: (), _args: &Args) -> Result<Value, StageError> {
        assert!(ctx.training_io_profile.is_none());
        LEGACY_STAGE_RAN.store(true, Ordering::SeqCst);
        Ok(Value(9))
    }
}

struct DdpProfileStage;

#[async_trait]
impl Stage for DdpProfileStage {
    const NAME: &'static str = "async_ddp_profile_stage";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const MEMORY_GIB: u32 = 1;
    type Input = ();
    type Output = Value;
    type Args = Args;

    fn training_io_sync_base_bytes(&self, _args: &Self::Args, _hints: TrainingIoHints) -> u64 {
        2 * 1024 * MIB
    }

    fn training_io_candidates(
        &self,
        _args: &Self::Args,
        _hints: TrainingIoHints,
    ) -> Vec<TrainingIoCandidate> {
        let mut bounded = bounded_candidate(Some(10 * MIB));
        bounded.data_replicas = 2;
        bounded.fixed_overhead_bytes = Some(5 * MIB);
        vec![bounded, inline_candidate()]
    }

    async fn run(&self, ctx: &StageContext, _input: (), _args: &Args) -> Result<Value, StageError> {
        let profile = ctx.training_io_profile.as_ref().expect("DDP profile");
        assert_eq!(profile.data_replicas, 2);
        assert_eq!(profile.sync_base_bytes, 2 * 1024 * MIB);
        assert_eq!(profile.billed_overhead_bytes, 140 * MIB);
        assert_eq!(
            profile
                .sync_base_bytes
                .checked_add(profile.billed_overhead_bytes),
            Some(2188 * MIB)
        );
        DDP_STAGE_RAN.store(true, Ordering::SeqCst);
        Ok(Value(13))
    }
}

struct SinkStage;

#[async_trait]
impl Stage for SinkStage {
    const NAME: &'static str = "async_sink_stage";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Value;
    type Output = Value;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Value,
        _args: &Args,
    ) -> Result<Value, StageError> {
        assert!(ctx.training_io_profile.is_none());
        Ok(input)
    }
}

/// A public control-plane seam for runtime Spawn/PBT tests. It emits exactly
/// one benign step and remains alive long enough for the parallel coordinator
/// to observe the lossily broadcast step before the node completes.
struct DynamicSpawnTrigger;

#[async_trait]
impl Stage for DynamicSpawnTrigger {
    const NAME: &'static str = "async_dynamic_spawn_trigger";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Value;
    type Args = Args;

    async fn run(&self, ctx: &StageContext, _input: (), _args: &Args) -> Result<Value, StageError> {
        let _ = ctx.status_tx.send(StageEvent::StageStep {
            node_idx: ctx.node_idx,
            stage_name: Self::NAME.to_string(),
            update: serde_json::json!({"step": 1, "loss": 0.5}),
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(Value(1))
    }
}

struct TestCookbook;

impl Cookbook for TestCookbook {
    fn name(&self) -> &'static str {
        "async-admission-test"
    }

    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }

    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static STAGES: &[(&str, ErasedStageCtor)] = &[
            ("async_profile_stage", || Arc::new(ProfileStage)),
            ("async_ddp_profile_stage", || Arc::new(DdpProfileStage)),
            ("async_legacy_stage", || Arc::new(LegacyStage)),
            ("async_sink_stage", || Arc::new(SinkStage)),
            ("async_dynamic_spawn_trigger", || {
                Arc::new(DynamicSpawnTrigger)
            }),
        ];
        STAGES
    }
}

struct SpawnProfileOnce {
    fired: AtomicBool,
}

impl ControlPolicy for SpawnProfileOnce {
    fn on_step(&self, _metrics: &StepMetrics<'_>) -> Control {
        if self.fired.swap(true, Ordering::SeqCst) {
            Control::Continue
        } else {
            Control::Spawn(Box::new(SpawnDelta::new(
                plan("async_profile_stage"),
                Some("profile-trial".into()),
            )))
        }
    }
}

#[cfg(feature = "p2p")]
struct ProfileDispatchPolicy;

#[cfg(feature = "p2p")]
impl DispatchPolicy for ProfileDispatchPolicy {
    fn is_dispatchable(&self, stage_name: &str) -> bool {
        stage_name == ProfileStage::NAME
    }

    fn classify_stage(&self, _stage_name: &str, _args: &serde_json::Value) -> DataClass {
        DataClass::Public
    }

    fn select_peer(
        &self,
        _stage_name: &str,
        _resources: &P2pResourceRequest,
        _data_class: DataClass,
        _peers: &[PeerInfo],
    ) -> Option<PeerId> {
        None
    }

    fn verify_result(
        &self,
        _result: &TaskResult,
        _expected: &ContentHash,
        _peer_pubkey: &ed25519_dalek::VerifyingKey,
    ) -> DispatchVerdict {
        DispatchVerdict::Reject("not exercised".into())
    }
}

#[cfg(feature = "p2p")]
struct RefuseProfileSubmitter {
    submit_calls: Arc<AtomicUsize>,
}

#[cfg(feature = "p2p")]
impl DispatchSubmitter for RefuseProfileSubmitter {
    fn submit(
        &self,
        _request: DispatchRequest<'_>,
    ) -> Result<Box<dyn blut::framework::executor::DispatchHandle>, blut::error::TrainError> {
        self.submit_calls.fetch_add(1, Ordering::SeqCst);
        Err(blut::error::TrainError::other(
            "profile-declaring stages must stay local until the remote profile wire exists",
        ))
    }
}

fn registry() -> Registry {
    let mut registry = Registry::new();
    registry.register(Box::new(TestCookbook));
    registry
}

fn plan(stage: &str) -> blut::framework::plan::CompiledPlan {
    PlanSpec {
        name: format!("{stage}-plan"),
        nodes: vec![SpecNode {
            stage: stage.into(),
            args: serde_json::Value::Null,
            retry: None,
            timeout: None,
            priority: None,
            pure: false,
        }],
        edges: Vec::new(),
        expansions: Vec::new(),
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
    .compile(&registry())
    .expect("compile test plan")
}

fn dce_before_profile_plan() -> blut::framework::plan::CompiledPlan {
    PlanSpec {
        name: "dce-before-profile".into(),
        nodes: vec![
            SpecNode {
                stage: "async_profile_stage".into(),
                args: serde_json::Value::Null,
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            },
            SpecNode {
                stage: "async_profile_stage".into(),
                args: serde_json::Value::Null,
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            },
            SpecNode {
                stage: "async_sink_stage".into(),
                args: serde_json::Value::Null,
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            },
        ],
        edges: vec![(1, 2)],
        expansions: Vec::new(),
        condition_gates: Vec::new(),
        version: PLAN_SPEC_VERSION,
    }
    .compile(&registry())
    .expect("compile DCE profile plan")
}

#[tokio::test]
async fn executor_threads_profile_losslessly_without_changing_cache_identity() {
    let _guard = TEST_LOCK.lock().await;
    PROFILE_STAGE_RAN.store(false, Ordering::SeqCst);
    let temp = tempfile::tempdir().expect("tempdir");
    let shared_cache = Arc::new(CacheHandle::job_local(temp.path().join("shared-cache")));

    let first_job = temp.path().join("first");
    let mut first_ctx = ExecCtx::new(first_job.clone())
        .with_memory_budget(2)
        .with_admitted_workers(3)
        .with_training_io_selection_budget_bytes(2 * 1024 * MIB);
    first_ctx.cache = shared_cache.clone();
    let first = execute_plan(plan("async_profile_stage"), first_ctx)
        .await
        .expect("profile run");
    assert_eq!(first.n_cache_misses, 1);
    assert!(PROFILE_STAGE_RAN.load(Ordering::SeqCst));

    let body = std::fs::read_to_string(first_job.join("status.jsonl"))
        .expect("status persisted before successful completion");
    let configured = body
        .find("\"kind\":\"stage_io_configured\"")
        .expect("structured profile lifecycle event");
    let began = body
        .find("\"kind\":\"stage_begin\"")
        .expect("stage begin lifecycle event");
    assert!(configured < began, "profile is persisted before StageBegin");
    assert!(body.contains("\"billed_overhead_bytes\":104857600"));

    // The execution-only profile and force-inline policy must not enter the
    // stage cache key. The second run asks for a different execution profile
    // but still consumes the first run's cached artifact.
    PROFILE_STAGE_RAN.store(false, Ordering::SeqCst);
    let second_job = temp.path().join("second");
    let mut second_ctx = ExecCtx::new(second_job)
        .with_memory_budget(2)
        .with_admitted_workers(3)
        .with_sync_io(true);
    second_ctx.cache = shared_cache;
    let second = execute_plan(plan("async_profile_stage"), second_ctx)
        .await
        .expect("cache-neutral forced-inline run");
    assert_eq!(second.n_cache_hits, 1);
    assert!(!PROFILE_STAGE_RAN.load(Ordering::SeqCst));
}

#[tokio::test]
async fn no_candidate_stage_keeps_legacy_context_and_status() {
    let _guard = TEST_LOCK.lock().await;
    LEGACY_STAGE_RAN.store(false, Ordering::SeqCst);
    let temp = tempfile::tempdir().expect("tempdir");
    let job = temp.path().join("legacy");
    execute_plan(plan("async_legacy_stage"), ExecCtx::new(job.clone()))
        .await
        .expect("legacy stage");
    assert!(LEGACY_STAGE_RAN.load(Ordering::SeqCst));
    let body = std::fs::read_to_string(job.join("status.jsonl")).expect("status");
    assert!(!body.contains("stage_io_configured"));
}

#[tokio::test]
async fn declaring_stage_refuses_oversized_base_instead_of_clamping_it() {
    let _guard = TEST_LOCK.lock().await;
    PROFILE_STAGE_RAN.store(false, Ordering::SeqCst);
    let temp = tempfile::tempdir().expect("tempdir");
    let error = execute_plan(
        plan("async_profile_stage"),
        ExecCtx::new(temp.path().join("oversized"))
            .with_memory_budget(2)
            .with_training_io_whole_job_base_bytes(3 * 1024 * MIB),
    )
    .await
    .expect_err("oversized base must refuse");
    assert!(error.to_string().contains("synchronous base envelope"));
    assert!(!PROFILE_STAGE_RAN.load(Ordering::SeqCst));
}

#[tokio::test]
async fn calibrated_whole_job_base_drives_fastest_fit_before_reservation() {
    let _guard = TEST_LOCK.lock().await;
    PROFILE_STAGE_RAN.store(false, Ordering::SeqCst);
    let temp = tempfile::tempdir().expect("tempdir");
    let job = temp.path().join("calibrated-base");
    execute_plan(
        plan("async_profile_stage"),
        ExecCtx::new(job.clone())
            .with_memory_budget(2)
            .with_admitted_batch_size(1)
            .with_training_io_selection_budget_bytes(1150 * MIB)
            .with_training_io_whole_job_base_bytes(1100 * MIB),
    )
    .await
    .expect("inline whole-job base fits after bounded profile is downgraded");
    assert!(PROFILE_STAGE_RAN.load(Ordering::SeqCst));
    let body = std::fs::read_to_string(job.join("status.jsonl")).expect("status");
    assert!(body.contains("\"sync_base_bytes\":1153433600"));
    assert!(body.contains("\"downgrade_reason\":\"budget_pressure\""));
}

#[tokio::test]
async fn dce_removes_extra_declaring_node_before_selecting_once() {
    let _guard = TEST_LOCK.lock().await;
    PROFILE_STAGE_RAN.store(false, Ordering::SeqCst);
    PROFILE_CANDIDATE_CALLS.store(0, Ordering::SeqCst);
    let temp = tempfile::tempdir().expect("tempdir");
    let job = temp.path().join("dce-profile");
    let result = ParallelExecutor::execute(
        dce_before_profile_plan(),
        ExecCtx::new(job.clone())
            .with_memory_budget(2)
            .with_admitted_workers(3)
            .with_training_io_selection_budget_bytes(2 * 1024 * MIB)
            .with_training_io_whole_job_base_bytes(1024 * MIB),
    )
    .await
    .expect("DCE removes the extra declaring node before selection");
    assert_eq!(result.n_stages, 2, "disconnected leading node was DCE'd");
    assert!(PROFILE_STAGE_RAN.load(Ordering::SeqCst));
    assert_eq!(
        PROFILE_CANDIDATE_CALLS.load(Ordering::SeqCst),
        1,
        "the surviving stage declares candidates exactly once"
    );
    let body = std::fs::read_to_string(job.join("status.jsonl")).expect("status");
    assert!(body.contains("\"kind\":\"stage_io_configured\",\"node_idx\":0"));
    assert!(body.contains("\"stage_name\":\"async_profile_stage\""));
}

#[tokio::test]
async fn direct_calls_force_inline_for_missing_snapshot_and_unsupported_launcher() {
    let _guard = TEST_LOCK.lock().await;
    let temp = tempfile::tempdir().expect("tempdir");

    let snapshot_job = temp.path().join("no-snapshot");
    execute_plan(
        plan("async_profile_stage"),
        ExecCtx::new(snapshot_job.clone())
            .with_memory_budget(2)
            .with_admitted_workers(3),
    )
    .await
    .expect("legacy direct call falls back to Inline without a snapshot witness");
    let snapshot_status =
        std::fs::read_to_string(snapshot_job.join("status.jsonl")).expect("snapshot status");
    assert!(snapshot_status.contains("\"downgrade_reason\":\"snapshot_unavailable\""));

    let slurm_job = temp.path().join("slurm");
    execute_plan(
        plan("async_profile_stage"),
        ExecCtx::new(slurm_job.clone())
            .with_memory_budget(2)
            .with_admitted_workers(3)
            .with_training_io_selection_budget_bytes(2 * 1024 * MIB)
            .with_launch_target(blut::config::launcher::LaunchTarget::Slurm),
    )
    .await
    .expect("unsupported remote retained-memory contract falls back to Inline");
    let slurm_status = std::fs::read_to_string(slurm_job.join("status.jsonl"))
        .expect("unsupported launcher status");
    assert!(slurm_status.contains("\"downgrade_reason\":\"unsupported_launcher\""));
}

#[tokio::test]
async fn stage_owned_ddp_base_raises_calibrated_floor_without_double_counting() {
    let _guard = TEST_LOCK.lock().await;
    DDP_STAGE_RAN.store(false, Ordering::SeqCst);
    let temp = tempfile::tempdir().expect("tempdir");
    ParallelExecutor::execute(
        plan("async_ddp_profile_stage"),
        ExecCtx::new(temp.path().join("ddp-base"))
            .with_memory_budget(3)
            .with_training_io_selection_budget_bytes(3 * 1024 * MIB)
            .with_training_io_whole_job_base_bytes(1500 * MIB),
    )
    .await
    .expect("two-rank stage base plus replicated retained overhead fits exactly once");
    assert!(DDP_STAGE_RAN.load(Ordering::SeqCst));
}

#[tokio::test]
async fn runtime_spawn_resolves_one_profile_from_the_launch_snapshot_before_execution() {
    let _guard = TEST_LOCK.lock().await;
    PROFILE_STAGE_RAN.store(false, Ordering::SeqCst);
    PROFILE_CANDIDATE_CALLS.store(0, Ordering::SeqCst);
    let temp = tempfile::tempdir().expect("tempdir");
    let job = temp.path().join("dynamic-profile");

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        ParallelExecutor::execute(
            plan("async_dynamic_spawn_trigger"),
            ExecCtx::new(job.clone())
                .with_memory_budget(4)
                .with_admitted_workers(3)
                .with_training_io_selection_budget_bytes(4 * 1024 * MIB)
                .with_training_io_whole_job_base_bytes(3 * 1024 * MIB)
                .with_control(Arc::new(SpawnProfileOnce {
                    fired: AtomicBool::new(false),
                })),
        ),
    )
    .await
    .expect("runtime spawn terminates")
    .expect("dynamic profile stage is admitted");

    assert_eq!(result.n_stages, 2, "one runtime node joined the plan");
    assert!(
        PROFILE_STAGE_RAN.load(Ordering::SeqCst),
        "dynamic stage received and consumed its profile"
    );
    assert_eq!(
        PROFILE_CANDIDATE_CALLS.load(Ordering::SeqCst),
        1,
        "the injected node declares candidates exactly once"
    );
    let status = std::fs::read_to_string(job.join("status.jsonl")).expect("status");
    assert!(status.contains("\"kind\":\"stage_io_configured\",\"node_idx\":1"));
    assert!(status.contains("\"stage_name\":\"async_profile_stage\""));
    assert!(status.contains("\"billed_overhead_bytes\":104857600"));
}

#[cfg(feature = "p2p")]
#[tokio::test]
async fn p2p_never_dispatches_a_locally_selected_bounded_profile() {
    let _guard = TEST_LOCK.lock().await;
    PROFILE_STAGE_RAN.store(false, Ordering::SeqCst);
    PROFILE_CANDIDATE_CALLS.store(0, Ordering::SeqCst);
    let temp = tempfile::tempdir().expect("tempdir");
    let job = temp.path().join("bounded-profile-stays-local");
    let submit_calls = Arc::new(AtomicUsize::new(0));

    ParallelExecutor::execute(
        plan("async_profile_stage"),
        ExecCtx::new(job.clone())
            .with_memory_budget(2)
            .with_admitted_workers(3)
            .with_training_io_selection_budget_bytes(2 * 1024 * MIB)
            .with_dispatch(
                Arc::new(ProfileDispatchPolicy),
                Arc::new(RefuseProfileSubmitter {
                    submit_calls: submit_calls.clone(),
                }),
            ),
    )
    .await
    .expect("profile-declaring stage stays on the local admitted path");

    assert_eq!(
        submit_calls.load(Ordering::SeqCst),
        0,
        "remote dispatch has no profile wire and must not see this node"
    );
    assert!(PROFILE_STAGE_RAN.load(Ordering::SeqCst));
    assert_eq!(PROFILE_CANDIDATE_CALLS.load(Ordering::SeqCst), 1);
    let status = std::fs::read_to_string(job.join("status.jsonl")).expect("status");
    assert!(status.contains("\"kind\":\"stage_io_configured\""));
    assert!(status.contains("\"billed_overhead_bytes\":104857600"));
    assert!(!status.contains("\"downgrade_reason\""));
}
