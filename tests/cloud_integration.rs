//! Cloud compute queue — end-to-end loopback proof (ADR 0067 · T3.1).
//!
//! The cloud analog of `p2p_integration::peer_runs_real_stage_end_to_end`: a real
//! stage (`p2p-echo`) is dispatched through `CloudSubmitter` over a LOCAL-FILESYSTEM
//! object store + an in-process `MemQueue`, executed by `cloud::worker::run_one`,
//! and the output is downloaded + content-hash-verified — the whole data plane, with
//! no cloud account. Network provider adapters are outside the public preview.
#![cfg(feature = "cloud")]

use std::sync::Arc;

use blut::cloud::queue::MemQueue;
use blut::cloud::submitter::{CloudPoll, CloudSubmitSpec, CloudSubmitter};
use blut::cloud::worker::run_one;
use blut::framework::artifact::{ContentHash, InvocationKey};
use blut::framework::cookbook::Registry;
use blut::framework::object_store::ObjectStore;
use blut::framework::stage::ErasedArtifact;
use blut::p2p::dispatch::DefaultDispatchPolicy;
use blut::p2p::smoke::{self, SMOKE_STAGE, SmokeText};
use blut::p2p::task::ResourceRequest;
use blut::p2p::trust::{DataClass, DispatchMatrix, TrustLevel};

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
async fn restricted_job_is_refused_by_a_registered_cloud_worker() {
    // The clinical hard-block: PHI EEG (DataClass::Restricted) must never run on a
    // cloud worker capped at Registered trust.
    let store_root = tempfile::tempdir().unwrap();
    let store = ObjectStore::local_provider(store_root.path(), "cloud-test").unwrap();
    let queue = Arc::new(MemQueue::new());
    let reg = smoke_registry();

    let (erased, _src) = make_input("phi data");

    let submitter = CloudSubmitter::new(store.clone(), queue.clone(), reg.clone());
    let handle = submitter
        .submit(CloudSubmitSpec {
            job_id: "job-phi".into(),
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
        .expect("submit");

    let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
    let matrix = DispatchMatrix::default();
    let work_root = tempfile::tempdir().unwrap();
    run_one(
        &store,
        queue.as_ref(),
        reg.as_ref(),
        &policy,
        &matrix,
        "cloud-worker-1",
        TrustLevel::Registered,
        30,
        work_root.path(),
        None,
    )
    .await
    .expect("worker run");

    let out_dir = tempfile::tempdir().unwrap();
    match handle.poll(out_dir.path()).await.unwrap() {
        CloudPoll::Failed(msg) => assert!(
            msg.contains("not permitted"),
            "refused for the data-class reason: {msg}"
        ),
        other => panic!("Restricted must be refused, got {:?}", poll_label(&other)),
    }
}

fn poll_label(p: &CloudPoll) -> &'static str {
    match p {
        CloudPoll::Pending => "Pending",
        CloudPoll::Succeeded(_) => "Succeeded",
        CloudPoll::Failed(_) => "Failed",
        CloudPoll::Cancelled => "Cancelled",
        CloudPoll::Unknown => "Unknown",
    }
}
