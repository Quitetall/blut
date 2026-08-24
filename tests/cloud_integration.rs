//! Cloud compute queue — end-to-end loopback proof (ADR 0067 · T3.1).
//!
//! The cloud analog of `p2p_integration::peer_runs_real_stage_end_to_end`: a real
//! stage (`p2p-echo`) is dispatched through `CloudSubmitter` over a LOCAL-FILESYSTEM
//! object store + an in-process `MemQueue`, executed by `cloud::worker::run_one`,
//! and the output is downloaded + content-hash-verified — the whole data plane, with
//! no cloud account. Network provider adapters are outside the public preview.
#![cfg(feature = "cloud")]

use std::sync::Arc;

use blut::cloud::queue::{CloudQueue, JobStatus, MemQueue};
use blut::cloud::submitter::{CloudPoll, CloudSubmitSpec, CloudSubmitter};
use blut::cloud::worker::run_one;
use blut::framework::artifact::{ContentHash, InvocationKey};
use blut::framework::artifact_store::{ArtifactRole, capture};
use blut::framework::cache::CacheHandle;
use blut::framework::cookbook::Registry;
use blut::framework::execution::{
    DataClassification, ExecutionAdapter, ExecutionDeadline, ExecutionMode, ExecutionRequest,
    ExecutionResources, ExecutionResult, ExecutionTerminal, drive_execution,
};
use blut::framework::object_store::ObjectStore;
use blut::framework::stage::ErasedArtifact;
use blut::p2p::dispatch::DefaultDispatchPolicy;
use blut::p2p::smoke::{self, SMOKE_STAGE, SmokeText};
use blut::p2p::task::ResourceRequest;
use blut::p2p::trust::{DataClass, DispatchMatrix, TrustLevel};
use tokio_util::sync::CancellationToken;

/// Build a registry with the built-in `p2p-echo` smoke stage registered.
fn smoke_registry() -> Arc<Registry> {
    let mut reg = Registry::new();
    smoke::register(&mut reg);
    Arc::new(reg)
}

/// Produce a `SmokeText` input artifact on disk and return (erased, src_root, input_hash).
fn make_input(text: &str) -> (ErasedArtifact, tempfile::TempDir) {
    let src_root = tempfile::tempdir().unwrap();
    let in_path = src_root.path().join("in.txt");
    std::fs::write(&in_path, text.as_bytes()).unwrap();
    let input = SmokeText {
        content_hash: ContentHash::hash_file(&in_path).unwrap(),
        path: in_path,
    };
    let erased = ErasedArtifact::from_typed(&input).unwrap();
    (erased, src_root)
}

fn canonical_request(
    id: &str,
    text: &str,
    registry: &Registry,
) -> (ExecutionRequest, tempfile::TempDir) {
    let (input, source) = make_input(text);
    let stage = registry.find_erased_stage(SMOKE_STAGE).unwrap()();
    let stored = capture(
        stage.as_ref(),
        input,
        source.path(),
        ArtifactRole::Input,
        None,
    )
    .unwrap();
    (
        ExecutionRequest {
            protocol_version: blut::framework::execution::EXECUTION_PROTOCOL_VERSION,
            execution_id: id.into(),
            tenant: blut::tenant::Tenant::default(),
            stage_name: SMOKE_STAGE.into(),
            stage_schema: stage.schema(),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(id.as_bytes())),
            args_hash: ContentHash::of_bytes(&CacheHandle::canonical_json_bytes(
                &serde_json::json!({}),
            )),
            args: serde_json::json!({}),
            input: Some(stored),
            expected_content_id: None,
            resources: ExecutionResources::default(),
            data_class: DataClassification::Public,
            deadline: ExecutionDeadline::from_now(None, std::time::Duration::from_secs(30)),
        },
        source,
    )
}

async fn wait_until_queued(queue: &MemQueue, job_id: &str) {
    for _ in 0..100 {
        if matches!(queue.status(job_id).await.unwrap(), JobStatus::Queued) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    panic!("cloud job '{job_id}' was not queued");
}

#[tokio::test]
async fn cloud_dispatch_round_trips_over_local_object_store() {
    let store_root = tempfile::tempdir().unwrap();
    let store = ObjectStore::local_provider(store_root.path(), "cloud-test").unwrap();
    let queue = Arc::new(MemQueue::new());
    let reg = smoke_registry();

    let (erased, _src) = make_input("hello cloud");

    let submitter = CloudSubmitter::new(store.clone(), queue.clone(), reg.clone());
    let handle = submitter
        .submit(CloudSubmitSpec {
            job_id: "job-1".into(),
            tenant: blut::tenant::Tenant::default(),
            stage_name: SMOKE_STAGE.into(),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"job-1")),
            input: erased,
            src_root: _src.path().to_path_buf(),
            args: serde_json::json!({}),
            expected_content_id: None,
            data_class: DataClass::Public,
            resources: ResourceRequest::default(),
            priority: 0,
            timeout_secs: 30,
        })
        .await
        .expect("submit");

    // Before any worker runs, the job is pending.
    let out_dir = tempfile::tempdir().unwrap();
    assert!(matches!(
        handle.poll(out_dir.path()).await.unwrap(),
        CloudPoll::Pending
    ));

    // A Registered cloud worker drains one job: p2p-echo on "hello cloud".
    let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
    let matrix = DispatchMatrix::default();
    let work_root = tempfile::tempdir().unwrap();
    let ledger = blut::cloud::cost::CostLedger::new(Default::default());
    let ran = run_one(
        &store,
        queue.as_ref(),
        reg.as_ref(),
        &policy,
        &matrix,
        "cloud-worker-1",
        TrustLevel::Registered,
        30,
        work_root.path(),
        Some(&ledger),
    )
    .await
    .expect("worker run");
    assert_eq!(
        ran.as_deref(),
        Some("job-1"),
        "worker claimed + ran the job"
    );
    // The job was billed (one ledger entry — the "billed on compute" plumbing fired;
    // p2p-echo is sub-ms so the unit total may round to ~0, hence assert the entry).
    assert_eq!(
        ledger.entries().len(),
        1,
        "completed job recorded a cost entry"
    );

    // The coordinator downloads + verifies the output (the four bundle gates run,
    // including the bind to expected_output_hash).
    match handle.poll(out_dir.path()).await.unwrap() {
        CloudPoll::Succeeded(output) => {
            let out: SmokeText = output.into_typed().unwrap();
            let body = std::fs::read_to_string(&out.path).unwrap();
            assert_eq!(
                body, "HELLO CLOUD",
                "worker ran the real stage on shipped data"
            );
            assert!(
                out.path.starts_with(out_dir.path()),
                "output materialized locally"
            );
        }
        other => panic!("expected Succeeded, got {:?}", poll_label(&other)),
    }
}

#[tokio::test]
async fn restricted_job_is_refused_before_cloud_enqueue() {
    // Custody fails before bytes reach object storage or queue.
    let store_root = tempfile::tempdir().unwrap();
    let store = ObjectStore::local_provider(store_root.path(), "cloud-test").unwrap();
    let queue = Arc::new(MemQueue::new());
    let reg = smoke_registry();

    let (erased, _src) = make_input("phi data");

    let submitter = CloudSubmitter::new(store.clone(), queue.clone(), reg.clone());
    let refusal = match submitter
        .submit(CloudSubmitSpec {
            job_id: "job-phi".into(),
            tenant: blut::tenant::Tenant::default(),
            stage_name: SMOKE_STAGE.into(),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"job-phi")),
            input: erased,
            src_root: _src.path().to_path_buf(),
            args: serde_json::json!({}),
            expected_content_id: None,
            data_class: DataClass::Restricted, // clinical
            resources: ResourceRequest::default(),
            priority: 0,
            timeout_secs: 30,
        })
        .await
    {
        Ok(_) => panic!("restricted data reached cloud enqueue"),
        Err(error) => error,
    };
    assert!(refusal.to_string().contains("custody policy"));
    assert!(matches!(
        queue.status("job-phi").await.unwrap(),
        JobStatus::Unknown
    ));

    let (clinical_input, clinical_source) = make_input("clinical tenant");
    let refusal = match submitter
        .submit(CloudSubmitSpec {
            job_id: "job-clinical".into(),
            tenant: blut::tenant::Tenant::parse("clinical/prod").unwrap(),
            stage_name: SMOKE_STAGE.into(),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"job-clinical")),
            input: clinical_input,
            src_root: clinical_source.path().to_path_buf(),
            args: serde_json::json!({}),
            expected_content_id: None,
            data_class: DataClass::Public,
            resources: ResourceRequest::default(),
            priority: 0,
            timeout_secs: 30,
        })
        .await
    {
        Ok(_) => panic!("clinical tenant reached cloud enqueue"),
        Err(error) => error,
    };
    assert!(refusal.to_string().contains("custody policy"));
    assert!(matches!(
        queue.status("job-clinical").await.unwrap(),
        JobStatus::Unknown
    ));
}

#[tokio::test]
async fn canonical_adapter_round_trips_through_execution_driver() {
    let store_root = tempfile::tempdir().unwrap();
    let store = ObjectStore::local_provider(store_root.path(), "cloud-adapter").unwrap();
    let queue = Arc::new(MemQueue::new());
    let registry = smoke_registry();
    let submitter = CloudSubmitter::new(store.clone(), queue.clone(), registry.clone());
    let (request, _source) = canonical_request("canonical-roundtrip", "adapter", &registry);
    let stage = registry.find_erased_stage(SMOKE_STAGE).unwrap()();
    let output_root = Arc::new(tempfile::tempdir().unwrap());
    let driver_output = output_root.clone();
    let driver_submitter = submitter.clone();
    let driver = tokio::spawn(async move {
        drive_execution(
            &driver_submitter,
            request,
            &CancellationToken::new(),
            stage,
            driver_output.path(),
            std::time::Duration::from_millis(5),
        )
        .await
    });

    wait_until_queued(&queue, "canonical-roundtrip").await;
    let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
    let work_root = tempfile::tempdir().unwrap();
    run_one(
        &store,
        queue.as_ref(),
        registry.as_ref(),
        &policy,
        &DispatchMatrix::default(),
        "cloud-worker-1",
        TrustLevel::Registered,
        30,
        work_root.path(),
        None,
    )
    .await
    .unwrap();

    match driver.await.unwrap() {
        ExecutionResult::Succeeded { artifact, .. } => {
            let output: SmokeText = artifact.into_typed().unwrap();
            assert_eq!(std::fs::read_to_string(output.path).unwrap(), "ADAPTER");
        }
        other => panic!(
            "canonical cloud execution failed: {}",
            execution_label(&other)
        ),
    }
}

#[tokio::test]
async fn completed_job_reconstructs_assignment_on_first_snapshot() {
    let store_root = tempfile::tempdir().unwrap();
    let store = ObjectStore::local_provider(store_root.path(), "cloud-fast").unwrap();
    let queue = Arc::new(MemQueue::new());
    let registry = smoke_registry();
    let submitter = CloudSubmitter::new(store.clone(), queue.clone(), registry.clone());
    let (request, _source) = canonical_request("fast-job", "fast", &registry);
    let handle = ExecutionAdapter::submit(&submitter, request).await.unwrap();
    wait_until_queued(&queue, "fast-job").await;

    let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
    let work_root = tempfile::tempdir().unwrap();
    run_one(
        &store,
        queue.as_ref(),
        registry.as_ref(),
        &policy,
        &DispatchMatrix::default(),
        "same-worker",
        TrustLevel::Registered,
        30,
        work_root.path(),
        None,
    )
    .await
    .unwrap();

    let snapshot = handle.snapshot().await.unwrap();
    assert_eq!(snapshot.mode, ExecutionMode::Cloud);
    assert_eq!(snapshot.assignment.as_ref().unwrap().owner, "same-worker");
    assert!(
        matches!(snapshot.terminal, Some(ExecutionTerminal::Succeeded { .. })),
        "late cancellation displaced success: {snapshot:?}"
    );
}

#[tokio::test]
async fn dropping_adapter_handle_cancels_queued_work() {
    let store_root = tempfile::tempdir().unwrap();
    let store = ObjectStore::local_provider(store_root.path(), "cloud-drop").unwrap();
    let queue = Arc::new(MemQueue::new());
    let registry = smoke_registry();
    let submitter = CloudSubmitter::new(store, queue.clone(), registry.clone());
    let (request, _source) = canonical_request("dropped-job", "drop", &registry);
    let handle = ExecutionAdapter::submit(&submitter, request).await.unwrap();
    wait_until_queued(&queue, "dropped-job").await;

    drop(handle);
    for _ in 0..100 {
        if matches!(
            queue.status("dropped-job").await.unwrap(),
            JobStatus::Done(result) if result.outcome == blut::cloud::job::JobOutcome::Cancelled
        ) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    panic!("dropping the adapter handle left durable work queued");
}

#[tokio::test]
async fn committed_success_wins_over_late_cancellation() {
    let store_root = tempfile::tempdir().unwrap();
    let store = ObjectStore::local_provider(store_root.path(), "cloud-late-cancel").unwrap();
    let queue = Arc::new(MemQueue::new());
    let registry = smoke_registry();
    let submitter = CloudSubmitter::new(store.clone(), queue.clone(), registry.clone());
    let (request, _source) = canonical_request("late-cancel", "winner", &registry);
    let handle = ExecutionAdapter::submit(&submitter, request).await.unwrap();
    wait_until_queued(&queue, "late-cancel").await;

    let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
    let work_root = tempfile::tempdir().unwrap();
    run_one(
        &store,
        queue.as_ref(),
        registry.as_ref(),
        &policy,
        &DispatchMatrix::default(),
        "winner-worker",
        TrustLevel::Registered,
        30,
        work_root.path(),
        None,
    )
    .await
    .unwrap();

    handle.cancel().await.unwrap();
    let snapshot = handle.snapshot().await.unwrap();
    assert!(
        matches!(snapshot.terminal, Some(ExecutionTerminal::Succeeded { .. })),
        "late cancellation displaced success: {snapshot:?}"
    );
}

fn execution_label(result: &ExecutionResult) -> &'static str {
    match result {
        ExecutionResult::Succeeded { .. } => "Succeeded",
        ExecutionResult::Failed(_) => "Failed",
        ExecutionResult::Cancelled => "Cancelled",
        ExecutionResult::TimedOut { .. } => "TimedOut",
    }
}

fn poll_label(p: &CloudPoll) -> &'static str {
    match p {
        CloudPoll::Pending => "Pending",
        CloudPoll::Succeeded(_) => "Succeeded",
        CloudPoll::Failed(_) => "Failed",
        CloudPoll::Cancelled => "Cancelled",
        CloudPoll::TimedOut => "TimedOut",
        CloudPoll::Unknown => "Unknown",
    }
}

/// ADR 0082's second named acceptance test: the clinical hard-block at the
/// WORKER, not just at submit.
///
/// `restricted_job_is_refused_before_cloud_enqueue` proves the first gate — the
/// submitter's custody check refuses `Restricted` before any byte reaches object
/// storage. This proves the SECOND, independent gate: if a `Restricted` job
/// reaches the queue *anyway* — an older client, a bug, a compromised or
/// bypassed submitter — a cloud worker must still refuse to execute it. Only a
/// test that puts such a job in the queue can demonstrate that, so this one
/// enqueues directly and deliberately skips `CloudSubmitter`.
///
/// The refusal is UNCONDITIONAL, which is stronger than "a v1 worker is only
/// `Registered`": `DispatchMatrix::can_dispatch` returns false for
/// `DataClass::Restricted` before it ever indexes the trust row
/// (`blut-types/src/trust.rs`), so no trust level and no operator-set policy
/// cell can turn it on. The control below is therefore a PUBLIC job on the same
/// worker — not a higher-trust worker, which would prove nothing.
#[tokio::test]
async fn restricted_job_is_refused_by_a_registered_cloud_worker() {
    let store_root = tempfile::tempdir().unwrap();
    let store = ObjectStore::local_provider(store_root.path(), "cloud-test").unwrap();
    let queue = Arc::new(MemQueue::new());
    let reg = smoke_registry();
    let submitter = CloudSubmitter::new(store.clone(), queue.clone(), reg.clone());

    // Submit a legitimate Public job purely to get a REAL uploaded bundle: the
    // point is to test the classification gate, so the smuggled job must be
    // well-formed in every other respect. A malformed one would be rejected by
    // an earlier check and the test would pass for the wrong reason.
    let (erased, _src) = make_input("phi payload");
    submitter
        .submit(CloudSubmitSpec {
            job_id: "job-seed".into(),
            tenant: blut::tenant::Tenant::default(),
            stage_name: SMOKE_STAGE.into(),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"job-seed")),
            input: erased,
            src_root: _src.path().to_path_buf(),
            args: serde_json::json!({}),
            expected_content_id: None,
            data_class: DataClass::Public,
            resources: ResourceRequest::default(),
            priority: 0,
            timeout_secs: 30,
        })
        .await
        .expect("seed submit");
    let seed = queue
        .claim("scratch-worker", 30)
        .await
        .unwrap()
        .expect("seed job claimable")
        .job;

    // The smuggled job: byte-identical to the seed except its classification.
    let mut smuggled = seed.clone();
    smuggled.id = "job-phi-slipped".into();
    smuggled.data_class = DataClass::Restricted;
    queue
        .enqueue(smuggled)
        .await
        .expect("enqueue bypasses the submitter on purpose");

    let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
    let matrix = DispatchMatrix::default();
    let work_root = tempfile::tempdir().unwrap();
    let ledger = blut::cloud::cost::CostLedger::new(Default::default());
    let ran = run_one(
        &store,
        queue.as_ref(),
        reg.as_ref(),
        &policy,
        &matrix,
        "cloud-worker-1",
        TrustLevel::Registered,
        30,
        work_root.path(),
        Some(&ledger),
    )
    .await
    .expect("worker drains the queue without erroring out");
    assert_eq!(
        ran.as_deref(),
        Some("job-phi-slipped"),
        "the worker must CLAIM the job and then refuse it — silently leaving it \
         pending would let another worker pick it up"
    );

    match queue.status("job-phi-slipped").await.unwrap() {
        JobStatus::Done(result) => {
            assert!(
                matches!(result.outcome, blut::cloud::job::JobOutcome::Failed),
                "restricted job must terminate as Failed, got {:?}",
                result.outcome
            );
            let failure = result.failure.expect("a refusal carries a typed failure");
            let text = failure.to_string();
            // The worker refuses at its own CUSTODY check
            // (`custody_allows_off_box`, worker.rs), which runs before the
            // trust-matrix gate. That ordering means `can_dispatch`'s
            // Restricted branch is a redundant second line here rather than the
            // one that fires — worth stating, because a reader looking only at
            // the matrix would conclude this path is what stops PHI, and would
            // then be free to "simplify" the custody check away.
            assert!(
                text.contains("Restricted")
                    && (text.contains("custody policy") || text.contains("not permitted")),
                "the failure must name the classification refusal, got: {text}"
            );
        }
        other => panic!("expected a terminal refusal, got {other:?}"),
    }

    // The stage never ran, so there is nothing to bill. A cost entry here would
    // mean the worker executed before checking, i.e. the block came too late.
    assert!(
        ledger.entries().is_empty(),
        "refused work must not be billed — a ledger entry implies it executed"
    );
    // And it left no output bundle behind in the store.
    assert!(
        !work_root.path().join("job-phi-slipped").exists(),
        "a refused job must not materialize a stage directory"
    );

    // CONTROL: the identical job, classified Public, IS executed by the SAME
    // worker at the SAME trust level. Without this the assertions above would
    // also pass if the job were simply broken.
    let mut allowed = seed.clone();
    allowed.id = "job-public-control".into();
    allowed.data_class = DataClass::Public;
    queue.enqueue(allowed).await.expect("enqueue control");
    let ran = run_one(
        &store,
        queue.as_ref(),
        reg.as_ref(),
        &policy,
        &matrix,
        "cloud-worker-1",
        TrustLevel::Registered,
        30,
        work_root.path(),
        Some(&ledger),
    )
    .await
    .expect("worker run");
    assert_eq!(ran.as_deref(), Some("job-public-control"));
    match queue.status("job-public-control").await.unwrap() {
        JobStatus::Done(result) => assert!(
            matches!(result.outcome, blut::cloud::job::JobOutcome::Succeeded),
            "the control job proves only the CLASSIFICATION was refused, not the job; \
             got {:?} ({:?})",
            result.outcome,
            result.failure
        ),
        other => panic!("expected the control job to succeed, got {other:?}"),
    }
}
