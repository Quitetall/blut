// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
#![cfg(feature = "openwarrant-receipts")]
use async_trait::async_trait;
use blut::framework::openwarrant_receipt::{
    DispatchBinding, Outcome, execute_with_receipt, verify,
};
use blut::framework::stage::ErasedStageCtor;
use blut::framework::{
    Artifact, ContentHash, Cookbook, ExecCtx, PlanSpec, Registry, Resource, Stage, StageContext,
    StageDyn, StageError,
};
use blut::recipes::recipe::RecipeDef;
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Proof {
    path: PathBuf,
}
impl Artifact for Proof {
    const KIND: &'static str = "receipt.test.proof";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(b"native execution happened\n")
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}
struct Produce;
#[async_trait]
impl Stage for Produce {
    const NAME: &'static str = "receipt_test_produce";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Proof;
    type Args = Option<bool>;
    async fn run(
        &self,
        ctx: &StageContext,
        _: (),
        args: &Option<bool>,
    ) -> Result<Proof, StageError> {
        if *args == Some(true) {
            let ambiguous = ctx.stage_dir.join("a\\b.txt");
            std::fs::write(&ambiguous, b"ambiguous native path").map_err(|source| {
                StageError::Io {
                    path: ambiguous,
                    source,
                }
            })?;
        }
        let path = ctx.stage_dir.join("proof.txt");
        std::fs::write(&path, b"native execution happened\n").map_err(|source| StageError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(Proof { path })
    }
}
struct Book;
impl Cookbook for Book {
    fn name(&self) -> &'static str {
        "receipt_test"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static STAGES: &[(&str, ErasedStageCtor)] = &[("receipt_test_produce", || {
            Arc::new(Produce) as Arc<dyn StageDyn>
        })];
        STAGES
    }
}
fn registry() -> Registry {
    let mut r = Registry::new();
    r.register(Box::new(Book));
    r
}
fn spec() -> PlanSpec {
    serde_json::from_str(r#"{"name":"receipt-positive","nodes":[{"stage":"receipt_test_produce","args":null}],"edges":[]}"#).unwrap()
}
fn binding() -> DispatchBinding {
    DispatchBinding {
        warrant_ref: "war://test-prototype".into(),
        contract_digest: "test-contract".into(),
        dispatch_digest: "test-dispatch".into(),
        stage_id: "S1".into(),
        attempt_id: "attempt-1".into(),
    }
}

#[tokio::test]
async fn actual_cpu_execution_yields_a_verifiable_dispatch_bound_receipt() {
    let td = tempfile::tempdir().unwrap();
    let job = td.path().join("job-1");
    // Synthetic in-memory test key only; never written to disk or used for authority.
    let signer = SigningKey::from_bytes(&[7; 32]);
    let reg = registry();
    let plan = spec();
    let expected = binding();
    let run = execute_with_receipt(
        &plan,
        &reg,
        ExecCtx::new(job.clone()),
        expected.clone(),
        &signer,
    )
    .await
    .unwrap();
    let result = run.result.unwrap();
    assert_eq!(result.n_stages, 1);
    assert_eq!(result.n_cache_hits, 0);
    let verified = verify(
        &run.envelope,
        &signer.verifying_key(),
        &expected,
        &plan,
        &reg,
        &job,
        &std::env::current_exe().unwrap(),
    )
    .unwrap();
    assert_eq!(verified.binding, expected);
    assert_eq!(verified.outcome, Outcome::Completed);
    assert!(std::fs::read_dir(&job).unwrap().next().is_some());
}

#[tokio::test]
async fn wrong_basis_key_payload_and_native_bytes_cannot_validate_completion() {
    use blut::framework::openwarrant_receipt::ReceiptError;
    let td = tempfile::tempdir().unwrap();
    let job = td.path().join("refusals");
    let signer = SigningKey::from_bytes(&[7; 32]);
    let key = signer.verifying_key();
    let reg = registry();
    let plan = spec();
    let expected = binding();
    let exe = std::env::current_exe().unwrap();
    let run = execute_with_receipt(
        &plan,
        &reg,
        ExecCtx::new(job.clone()),
        expected.clone(),
        &signer,
    )
    .await
    .unwrap();
    let proof = run
        .result
        .unwrap()
        .final_output
        .unwrap()
        .into_typed::<Proof>()
        .unwrap();
    assert_eq!(
        std::fs::read(&proof.path).unwrap(),
        b"native execution happened\n"
    );
    let wrong = SigningKey::from_bytes(&[8; 32]).verifying_key();
    assert!(matches!(
        verify(&run.envelope, &wrong, &expected, &plan, &reg, &job, &exe),
        Err(ReceiptError::Rejected(_))
    ));
    let mut other = expected.clone();
    other.attempt_id = "another-attempt".into();
    assert!(matches!(
        verify(&run.envelope, &key, &other, &plan, &reg, &job, &exe),
        Err(ReceiptError::Rejected(_))
    ));
    let mut other_plan = spec();
    other_plan.name = "changed-plan".into();
    assert!(matches!(
        verify(
            &run.envelope,
            &key,
            &expected,
            &other_plan,
            &reg,
            &job,
            &exe
        ),
        Err(ReceiptError::Rejected(_))
    ));
    let wrong_binary = td.path().join("wrong-binary");
    std::fs::write(&wrong_binary, b"not the producer executable").unwrap();
    assert!(matches!(
        verify(
            &run.envelope,
            &key,
            &expected,
            &plan,
            &reg,
            &job,
            &wrong_binary
        ),
        Err(ReceiptError::Rejected(_))
    ));
    let mut altered = run.envelope.clone();
    altered.payload[0] ^= 1;
    assert!(matches!(
        verify(&altered, &key, &expected, &plan, &reg, &job, &exe),
        Err(ReceiptError::Rejected(_))
    ));
    std::fs::write(&proof.path, b"altered output\n").unwrap();
    assert!(matches!(
        verify(&run.envelope, &key, &expected, &plan, &reg, &job, &exe),
        Err(ReceiptError::Rejected(_))
    ));
    assert!(matches!(
        verify(
            &run.envelope,
            &key,
            &expected,
            &plan,
            &reg,
            &td.path().join("missing-job"),
            &exe
        ),
        Err(ReceiptError::Unavailable(_))
    ));
}

#[tokio::test]
async fn actual_cancelled_execution_remains_cancelled() {
    let td = tempfile::tempdir().unwrap();
    let job = td.path().join("cancelled");
    let signer = SigningKey::from_bytes(&[7; 32]);
    let reg = registry();
    let plan = spec();
    let expected = binding();
    let ctx = ExecCtx::new(job.clone());
    ctx.cancel.cancel();
    let run = execute_with_receipt(&plan, &reg, ctx, expected.clone(), &signer)
        .await
        .unwrap();
    assert!(matches!(
        run.result,
        Err(blut::framework::PlanError::Cancelled)
    ));
    assert_eq!(
        verify(
            &run.envelope,
            &signer.verifying_key(),
            &expected,
            &plan,
            &reg,
            &job,
            &std::env::current_exe().unwrap()
        )
        .unwrap()
        .outcome,
        Outcome::Cancelled
    );
}

#[tokio::test]
async fn existing_job_is_preserved_and_never_executed_again() {
    let td = tempfile::tempdir().unwrap();
    let job = td.path().join("existing");
    std::fs::create_dir(&job).unwrap();
    let marker = job.join("original");
    std::fs::write(&marker, b"preserve").unwrap();
    let signer = SigningKey::from_bytes(&[7; 32]);
    assert!(matches!(
        execute_with_receipt(
            &spec(),
            &registry(),
            ExecCtx::new(job.clone()),
            binding(),
            &signer
        )
        .await,
        Err(blut::framework::openwarrant_receipt::ReceiptError::Rejected(_))
    ));
    assert_eq!(std::fs::read(marker).unwrap(), b"preserve");
    assert_eq!(std::fs::read_dir(job).unwrap().count(), 1);
}

#[cfg(unix)]
#[tokio::test]
async fn ambiguous_native_path_is_not_sealed_as_completed() {
    use blut::framework::openwarrant_receipt::ReceiptError;
    let td = tempfile::tempdir().unwrap();
    let plan: PlanSpec = serde_json::from_str(r#"{"name":"ambiguous-path","nodes":[{"stage":"receipt_test_produce","args":true}],"edges":[]}"#).unwrap();
    let signer = SigningKey::from_bytes(&[7; 32]);
    let result = execute_with_receipt(
        &plan,
        &registry(),
        ExecCtx::new(td.path().join("job")),
        binding(),
        &signer,
    )
    .await;
    assert!(
        matches!(result, Err(ReceiptError::Rejected(ref reason)) if reason.contains("ambiguous native path"))
    );
}
