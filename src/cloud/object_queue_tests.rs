// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Queue semantics of [`ObjectQueue`], run against a local directory and,
//! when `BLUT_TEST_S3_QUEUE=s3://bucket/prefix` and the `AWS_*` variables are
//! set (a build with the `s3` feature), against that bucket too.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::cloud::job::{CLOUD_JOB_PROTOCOL_VERSION, JobOutcome};
use crate::framework::artifact::{ArtifactContentId, ContentHash, InvocationKey};
use crate::framework::code_identity::{BuildIdentity, CodeIdentity};
use crate::framework::stage::ErasedArtifact;
use crate::p2p::bundle::BundleManifest;
use crate::p2p::task::ResourceRequest;
use crate::p2p::trust::DataClass;

/// One place to put queues for a test: a temp directory, or a unique prefix
/// in the bucket named by `BLUT_TEST_S3_QUEUE`.
struct Site {
    provider: Arc<dyn object_store::ObjectStore>,
    prefix: String,
    _dir: Option<tempfile::TempDir>,
}

impl Site {
    fn local() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let provider = object_store::local::LocalFileSystem::new_with_prefix(dir.path()).unwrap();
        Self {
            provider: Arc::new(provider),
            prefix: "t".into(),
            _dir: Some(dir),
        }
    }

    #[cfg(feature = "s3")]
    fn s3() -> Option<Self> {
        let url = std::env::var("BLUT_TEST_S3_QUEUE").ok()?;
        let (provider, url_prefix) =
            crate::framework::object_store::provider_from_url(&url).expect("BLUT_TEST_S3_QUEUE");
        let prefix = crate::framework::object_store::join_prefix(
            &url_prefix,
            &format!("queue-test-{}", uuid::Uuid::new_v4()),
        );
        Some(Self {
            provider,
            prefix,
            _dir: None,
        })
    }

    fn queue(&self) -> ObjectQueue {
        self.queue_with(ObjectQueueOptions::default().with_status_refresh(Duration::ZERO))
    }

    fn queue_with(&self, options: ObjectQueueOptions) -> ObjectQueue {
        ObjectQueue::new(self.provider.clone(), &self.prefix, options)
            .with_code_identity(identity("a"))
    }
}

/// Every site the environment offers: always a directory, and a bucket when
/// one is configured.
fn sites() -> Vec<(&'static str, Site)> {
    #[allow(unused_mut)]
    let mut sites = vec![("local directory", Site::local())];
    #[cfg(feature = "s3")]
    if let Some(site) = Site::s3() {
        sites.push(("s3 bucket", site));
    }
    sites
}

fn identity(cookbook_commit: &str) -> CodeIdentity {
    CodeIdentity::with_engine(BuildIdentity::new("blut", "0.0.0-test")).with_cookbook(
        BuildIdentity::from_build_env("cookbook", "1.0.0", Some(cookbook_commit), None),
    )
}

fn manifest() -> BundleManifest {
    BundleManifest {
        format_version: crate::framework::artifact_store::ARTIFACT_FORMAT_VERSION,
        erased: ErasedArtifact {
            kind: "test".into(),
            schema: 1,
            payload: vec![],
        },
        kind: "test".into(),
        schema: 1,
        content_id: ArtifactContentId::from_digest(ContentHash::of_bytes(b"")),
        logical_hash: ContentHash::of_bytes(b""),
        handle_root: std::path::PathBuf::from("__test_artifact_root__"),
        files: vec![],
        blob_len: 0,
        blob_sha256: ContentHash::of_bytes(b""),
    }
}

fn job(id: &str, priority: i32) -> CloudJob {
    CloudJob {
        protocol_version: CLOUD_JOB_PROTOCOL_VERSION,
        id: id.to_string(),
        tenant: crate::tenant::Tenant::default(),
        stage_name: "p2p-echo".to_string(),
        stage_schema: 1,
        invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(id.as_bytes())),
        args_hash: ContentHash::of_bytes(b"{}"),
        args: serde_json::json!({}),
        input_blob_key: ContentHash::of_bytes(id.as_bytes()),
        input_manifest: manifest(),
        expected_content_id: None,
        resources: ResourceRequest::default(),
        data_class: DataClass::Public,
        priority,
        timeout_secs: 30,
        deadline: crate::framework::execution::ExecutionDeadline::from_now(
            None,
            Duration::from_secs(30),
        ),
    }
}

fn done(id: &str) -> CloudResult {
    CloudResult {
        protocol_version: CLOUD_JOB_PROTOCOL_VERSION,
        job_id: id.to_string(),
        assignment: None,
        outcome: JobOutcome::Succeeded,
        output_blob_key: Some(ContentHash::of_bytes(b"ok")),
        output_manifest: None,
        content_id: Some(ArtifactContentId::from_digest(ContentHash::of_bytes(b"ok"))),
        wall_time_ms: 5,
        failure: None,
        timeout_phase: None,
        deadline_unix_ms: None,
    }
}

#[tokio::test]
async fn enqueue_claim_complete_lifecycle() {
    for (site_name, site) in sites() {
        let submitter = site.queue();
        let worker = site.queue();
        submitter.enqueue(job("a", 0)).await.unwrap();
        assert!(
            matches!(submitter.status("a").await.unwrap(), JobStatus::Queued),
            "{site_name}"
        );
        assert_eq!(
            submitter.list_active().await.unwrap().len(),
            1,
            "{site_name}"
        );

        let claimed = worker.claim("w1", 30).await.unwrap().expect(site_name);
        assert_eq!(claimed.job.id, "a");
        assert_eq!(claimed.assignment, Assignment::new("w1", 1), "{site_name}");
        assert!(
            matches!(submitter.status("a").await.unwrap(), JobStatus::Running(a) if a == claimed.assignment),
            "{site_name}"
        );

        worker
            .complete(&claimed.assignment, done("a"))
            .await
            .unwrap();
        match submitter.status("a").await.unwrap() {
            JobStatus::Done(result) => {
                assert_eq!(result.outcome, JobOutcome::Succeeded, "{site_name}");
                assert_eq!(result.assignment.as_ref(), Some(&claimed.assignment));
            }
            other => panic!("{site_name}: {other:?}"),
        }
        assert!(
            worker.claim("w1", 30).await.unwrap().is_none(),
            "{site_name}"
        );
        assert!(
            submitter.list_active().await.unwrap().is_empty(),
            "{site_name}: a finished job leaves the index"
        );
        assert!(
            matches!(submitter.status("ghost").await.unwrap(), JobStatus::Unknown),
            "{site_name}"
        );
    }
}

#[tokio::test]
async fn enqueue_refuses_duplicates_and_unsafe_ids() {
    for (site_name, site) in sites() {
        let q = site.queue();
        q.enqueue(job("a", 0)).await.unwrap();
        assert!(
            matches!(
                q.enqueue(job("a", 0)).await,
                Err(CloudError::DuplicateJob(_))
            ),
            "{site_name}"
        );
        assert!(
            matches!(
                q.enqueue(job("../x", 0)).await,
                Err(CloudError::BadJobId(_))
            ),
            "{site_name}"
        );
        assert!(
            matches!(q.status("../x").await, Err(CloudError::BadJobId(_))),
            "{site_name}"
        );
    }
}

/// Eight workers, each with its own queue handle (as separate processes
/// would have), race for one job at a time and then drain a batch. Every job
/// is claimed exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn claims_are_exclusive_under_concurrent_workers() {
    for (site_name, site) in sites() {
        let site = Arc::new(site);
        for round in 0..5 {
            let id = format!("race-{round}");
            site.queue().enqueue(job(&id, 0)).await.unwrap();
            let mut tasks = Vec::new();
            for w in 0..8 {
                let site = site.clone();
                tasks.push(tokio::spawn(async move {
                    site.queue()
                        .claim(&format!("w{w}"), 30)
                        .await
                        .unwrap()
                        .map(|c| c.assignment)
                }));
            }
            let mut winners = Vec::new();
            for task in tasks {
                if let Some(assignment) = task.await.unwrap() {
                    winners.push(assignment);
                }
            }
            assert_eq!(
                winners.len(),
                1,
                "{site_name}: round {round}: exactly one worker claims a job, got {winners:?}"
            );
        }

        let ids: Vec<String> = (0..12).map(|i| format!("batch-{i}")).collect();
        for id in &ids {
            site.queue().enqueue(job(id, 0)).await.unwrap();
        }
        let mut tasks = Vec::new();
        for w in 0..8 {
            let site = site.clone();
            tasks.push(tokio::spawn(async move {
                let q = site.queue();
                let mut mine = Vec::new();
                while let Some(claimed) = q.claim(&format!("drain-{w}"), 30).await.unwrap() {
                    mine.push(claimed.job.id);
                }
                mine
            }));
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for task in tasks {
            for id in task.await.unwrap() {
                *counts.entry(id).or_default() += 1;
            }
        }
        for id in &ids {
            assert_eq!(
                counts.get(id),
                Some(&1),
                "{site_name}: {id} claimed {:?} times",
                counts.get(id)
            );
        }
    }
}

#[tokio::test]
async fn an_expired_lease_is_requeued_and_its_holder_fenced() {
    for (site_name, site) in sites() {
        let q = site.queue();
        q.enqueue(job("a", 0)).await.unwrap();
        // w1 claims and then dies (never renews). The lease outlasts a claim
        // attempt over a slow network, or the next assertion would race it.
        let first = q.claim("w1", 4).await.unwrap().expect(site_name);
        assert!(
            q.claim("w2", 30).await.unwrap().is_none(),
            "{site_name}: a live lease is not claimable"
        );
        tokio::time::sleep(Duration::from_millis(4300)).await;
        assert!(
            matches!(q.status("a").await.unwrap(), JobStatus::Queued),
            "{site_name}: an expired lease reads as queued"
        );
        assert_eq!(q.reclaim_expired().await.unwrap(), 1, "{site_name}");
        let second = q.claim("w2", 30).await.unwrap().expect(site_name);
        assert_eq!(second.job.id, "a");
        assert!(second.assignment.generation > first.assignment.generation);

        // The dead worker comes back: it can neither finish, renew nor release.
        assert!(
            matches!(
                q.complete(&first.assignment, done("a")).await,
                Err(CloudError::LeaseLost(_))
            ),
            "{site_name}"
        );
        assert!(!q.renew("a", &first.assignment, 30).await.unwrap());
        assert!(!q.release("a", &first.assignment, "late").await.unwrap());
        assert!(
            matches!(q.status("a").await.unwrap(), JobStatus::Running(a) if a == second.assignment),
            "{site_name}"
        );
        q.complete(&second.assignment, done("a")).await.unwrap();
        assert!(matches!(q.status("a").await.unwrap(), JobStatus::Done(_)));
    }
}

#[tokio::test]
async fn renewal_keeps_a_lease_alive() {
    for (site_name, site) in sites() {
        let q = site.queue();
        q.enqueue(job("a", 0)).await.unwrap();
        let claimed = q.claim("w1", 1).await.unwrap().expect(site_name);
        assert!(q.renew("a", &claimed.assignment, 30).await.unwrap());
        tokio::time::sleep(Duration::from_millis(1300)).await;
        assert!(
            matches!(q.status("a").await.unwrap(), JobStatus::Running(_)),
            "{site_name}: a renewed lease outlives its first term"
        );
        assert!(q.claim("w2", 30).await.unwrap().is_none(), "{site_name}");
    }
}

#[tokio::test]
async fn a_released_job_is_claimable_at_once() {
    for (site_name, site) in sites() {
        let q = site.queue();
        q.enqueue(job("a", 0)).await.unwrap();
        let first = q.claim("w1", 300).await.unwrap().expect(site_name);
        assert!(
            q.release("a", &first.assignment, "shutting down")
                .await
                .unwrap()
        );
        assert!(matches!(q.status("a").await.unwrap(), JobStatus::Queued));
        let second = q.claim("w2", 30).await.unwrap().expect(site_name);
        assert!(second.assignment.generation > first.assignment.generation);
    }
}

#[tokio::test]
async fn cancellation_is_terminal_for_queued_and_running_jobs() {
    for (site_name, site) in sites() {
        let submitter = site.queue();
        let worker = site.queue();
        submitter.enqueue(job("queued", 0)).await.unwrap();
        assert!(
            submitter
                .cancel("queued", "caller cancelled")
                .await
                .unwrap()
        );
        assert!(
            matches!(
                submitter.status("queued").await.unwrap(),
                JobStatus::Done(r) if r.outcome == JobOutcome::Cancelled
            ),
            "{site_name}"
        );
        assert!(
            worker.claim("w1", 30).await.unwrap().is_none(),
            "{site_name}: a cancelled job is never claimed"
        );

        submitter.enqueue(job("running", 0)).await.unwrap();
        let claimed = worker.claim("w1", 30).await.unwrap().expect(site_name);
        // A different process cancels: the holder sees it on its next renewal
        // and its late result is refused.
        site.queue()
            .cancel("running", "blut cancel")
            .await
            .map(|cancelled| assert!(cancelled, "{site_name}"))
            .unwrap();
        assert!(
            !worker
                .renew("running", &claimed.assignment, 30)
                .await
                .unwrap()
        );
        assert!(
            matches!(
                worker.complete(&claimed.assignment, done("running")).await,
                Err(CloudError::LeaseLost(_))
            ),
            "{site_name}"
        );
        assert!(
            matches!(
                submitter.status("running").await.unwrap(),
                JobStatus::Done(r) if r.outcome == JobOutcome::Cancelled
            ),
            "{site_name}"
        );
        assert!(!submitter.cancel("running", "again").await.unwrap());
        assert!(!submitter.cancel("ghost", "never existed").await.unwrap());
    }
}

#[tokio::test]
async fn completion_is_idempotent_and_final() {
    for (site_name, site) in sites() {
        let q = site.queue();
        q.enqueue(job("a", 0)).await.unwrap();
        let claimed = q.claim("w1", 30).await.unwrap().expect(site_name);
        q.complete(&claimed.assignment, done("a")).await.unwrap();
        // A retry of the same completion (say, after a lost response) is fine.
        q.complete(&claimed.assignment, done("a")).await.unwrap();
        // Nobody else can complete it, and it cannot be cancelled any more.
        assert!(
            matches!(
                q.complete(&Assignment::new("w2", 1), done("a")).await,
                Err(CloudError::LeaseLost(_))
            ),
            "{site_name}"
        );
        assert!(!q.cancel("a", "too late").await.unwrap(), "{site_name}");
        match q.status("a").await.unwrap() {
            JobStatus::Done(result) => assert_eq!(result.outcome, JobOutcome::Succeeded),
            other => panic!("{site_name}: {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_job_from_other_code_is_left_unclaimed_and_the_refusal_recorded() {
    for (site_name, site) in sites() {
        let submitter = site.queue();
        let other = ObjectQueue::new(
            site.provider.clone(),
            &site.prefix,
            ObjectQueueOptions::default().with_status_refresh(Duration::ZERO),
        )
        .with_code_identity(identity("b"));
        submitter.enqueue(job("a", 0)).await.unwrap();

        assert!(
            other.claim("stale-worker", 30).await.unwrap().is_none(),
            "{site_name}: a worker running other code must not claim the job"
        );
        assert!(matches!(
            submitter.status("a").await.unwrap(),
            JobStatus::Queued
        ));
        let refusals = submitter.refusals("a").await.unwrap();
        assert_eq!(refusals.len(), 1, "{site_name}: {refusals:?}");
        assert!(
            refusals[0].contains("stale-worker") && refusals[0].contains("code identity differs"),
            "{site_name}: {refusals:?}"
        );

        // A worker running the submitter's code takes it.
        assert!(
            site.queue()
                .claim("good-worker", 30)
                .await
                .unwrap()
                .is_some()
        );
    }
}

#[tokio::test]
async fn a_gpu_stage_waits_for_a_gpu_worker() {
    for (site_name, site) in sites() {
        let mut gpu_job = job("gpu", 0);
        gpu_job.resources.gpu = true;
        site.queue().enqueue(gpu_job).await.unwrap();
        assert!(
            site.queue()
                .claim("cpu-worker", 30)
                .await
                .unwrap()
                .is_none(),
            "{site_name}"
        );
        let refusals = site.queue().refusals("gpu").await.unwrap();
        assert!(refusals[0].contains("GPU"), "{site_name}: {refusals:?}");
        let gpu_worker = site.queue_with(
            ObjectQueueOptions::default()
                .with_status_refresh(Duration::ZERO)
                .with_accept_gpu(true),
        );
        assert!(gpu_worker.claim("gpu-worker", 30).await.unwrap().is_some());
    }
}

#[tokio::test]
async fn claims_follow_priority_then_enqueue_order() {
    for (site_name, site) in sites() {
        let q = site.queue();
        q.enqueue(job("low", -5)).await.unwrap();
        q.enqueue(job("high", 9)).await.unwrap();
        q.enqueue(job("mid-1", 5)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        q.enqueue(job("mid-2", 5)).await.unwrap();
        let mut order = Vec::new();
        while let Some(claimed) = q.claim("w", 30).await.unwrap() {
            order.push(claimed.job.id);
        }
        assert_eq!(order, ["high", "mid-1", "mid-2", "low"], "{site_name}");
    }
}

#[tokio::test]
async fn status_reads_are_rate_limited_but_own_writes_are_seen() {
    let site = Site::local();
    let q =
        site.queue_with(ObjectQueueOptions::default().with_status_refresh(Duration::from_secs(60)));
    let other = site.queue();
    q.enqueue(job("a", 0)).await.unwrap();
    assert!(matches!(q.status("a").await.unwrap(), JobStatus::Queued));
    // Another process claims; within the refresh window this handle still
    // answers from its cache instead of asking the store again.
    let claimed = other.claim("w1", 30).await.unwrap().unwrap();
    assert!(matches!(q.status("a").await.unwrap(), JobStatus::Queued));
    // This handle's own write drops its cached state.
    assert!(q.cancel("a", "stop").await.unwrap());
    assert!(matches!(q.status("a").await.unwrap(), JobStatus::Done(_)));
    assert!(!other.renew("a", &claimed.assignment, 30).await.unwrap());
}
