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
struct Fail;
#[async_trait]
impl Stage for Fail {
    const NAME: &'static str = "receipt_test_fail";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Proof;
    type Args = ();
    async fn run(&self, _: &StageContext, _: (), _: &()) -> Result<Proof, StageError> {
        Err(StageError::BadInput("deliberate fixture failure".into()))
    }
}
struct Warn;
#[async_trait]
impl Stage for Warn {
    const NAME: &'static str = "receipt_test_warn";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const ADVISORY: bool = true;
    type Input = ();
    type Output = Proof;
    type Args = ();
    async fn run(&self, _: &StageContext, _: (), _: &()) -> Result<Proof, StageError> {
        Err(StageError::BadInput(
            "deliberate advisory fixture failure".into(),
        ))
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
        static STAGES: &[(&str, ErasedStageCtor)] = &[
            ("receipt_test_produce", || {
                Arc::new(Produce) as Arc<dyn StageDyn>
            }),
            ("receipt_test_fail", || Arc::new(Fail) as Arc<dyn StageDyn>),
            ("receipt_test_warn", || Arc::new(Warn) as Arc<dyn StageDyn>),
        ];
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
async fn file_transport_checks_native_receipt_without_executing_again() {
    use blut::framework::openwarrant_receipt::{VerificationFiles, registry_digest, verify_files};
    let td = tempfile::tempdir().unwrap();
    let job = td.path().join("transport-job");
    let signer = SigningKey::from_bytes(&[7; 32]);
    let reg = registry();
    // Resolve the expected native catalog from a trusted registry before the
    // receipt exists. Do not use a schema pin or borrow identity from payload.
    let expected_catalog = registry_digest(&reg).unwrap();
    assert_ne!(expected_catalog, registry_digest(&Registry::new()).unwrap());
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
    run.result.unwrap();
    let receipt = td.path().join("receipt.json");
    let binding_file = td.path().join("binding.json");
    let plan_file = td.path().join("plan.json");
    std::fs::write(&receipt, serde_json::to_vec(&run.envelope).unwrap()).unwrap();
    std::fs::write(&binding_file, serde_json::to_vec(&expected).unwrap()).unwrap();
    std::fs::write(&plan_file, serde_json::to_vec(&plan).unwrap()).unwrap();
    let request = VerificationFiles {
        receipt,
        binding: binding_file,
        plan: plan_file,
        job,
        producer_executable: std::env::current_exe().unwrap(),
    };
    let verified = verify_files(&request, &signer.verifying_key(), &reg).unwrap();
    assert_eq!(verified.outcome, Outcome::Completed);
    assert_eq!(verified.binding, expected);
    assert_eq!(verified.registry_digest, expected_catalog);
    assert_eq!(verified.lineage_reference.unwrap().path, "status.jsonl");
}

#[test]
fn file_transport_refuses_oversized_malformed_and_missing_inputs() {
    use blut::framework::openwarrant_receipt::{
        ReceiptError, VerificationFiles, read_verification_key, verify_files,
    };
    let td = tempfile::tempdir().unwrap();
    let receipt = td.path().join("receipt.json");
    let key_file = td.path().join("public-key.bin");
    let key = SigningKey::from_bytes(&[7; 32]).verifying_key();
    std::fs::write(&key_file, key.as_bytes()).unwrap();
    assert_eq!(read_verification_key(&key_file).unwrap(), key);
    std::fs::write(&key_file, [7; 33]).unwrap();
    assert!(matches!(
        read_verification_key(&key_file),
        Err(ReceiptError::Rejected(_))
    ));
    let request = VerificationFiles {
        receipt: receipt.clone(),
        binding: td.path().join("binding.json"),
        plan: td.path().join("plan.json"),
        job: td.path().join("job"),
        producer_executable: std::env::current_exe().unwrap(),
    };
    let reg = registry();
    assert!(matches!(
        verify_files(&request, &key, &reg),
        Err(ReceiptError::Unavailable(_))
    ));
    std::fs::write(&receipt, b"{broken}").unwrap();
    assert!(matches!(
        verify_files(&request, &key, &reg),
        Err(ReceiptError::Rejected(_))
    ));
    let oversized = std::fs::File::create(&receipt).unwrap();
    oversized.set_len(8 * 1024 * 1024 + 1025).unwrap();
    assert!(matches!(
        verify_files(&request, &key, &reg),
        Err(ReceiptError::Rejected(_))
    ));
    #[cfg(unix)]
    {
        let link = td.path().join("link.json");
        std::os::unix::fs::symlink(&receipt, &link).unwrap();
        assert!(matches!(
            verify_files(
                &VerificationFiles {
                    receipt: link,
                    ..request
                },
                &key,
                &reg
            ),
            Err(ReceiptError::Rejected(_))
        ));
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
    let proof = result.final_output.unwrap().into_typed::<Proof>().unwrap();
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
    let proof_ref = proof.path.strip_prefix(&job).unwrap().to_str().unwrap();
    let artifact = verified
        .native_files
        .iter()
        .find(|f| f.path == proof_ref)
        .unwrap();
    assert_eq!(artifact.bytes, 26);
    assert_eq!(
        std::fs::read(job.join(&artifact.path)).unwrap(),
        b"native execution happened\n"
    );
    let lineage = verified.lineage_reference.as_ref().unwrap();
    assert_eq!(lineage.path, "status.jsonl");
    assert!(lineage.bytes > 0);
    assert!(verified.native_files.contains(lineage));
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
    let mut bad_signature = run.envelope.clone();
    bad_signature.signature[0] ^= 1;
    assert!(matches!(
        verify(&bad_signature, &key, &expected, &plan, &reg, &job, &exe),
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
    let mut otherwise_valid: serde_json::Value = serde_json::from_slice(&altered.payload).unwrap();
    otherwise_valid["ended_unix_ms"] =
        serde_json::json!(otherwise_valid["ended_unix_ms"].as_u64().unwrap() + 1);
    altered.payload = serde_json::to_vec(&otherwise_valid).unwrap();
    assert!(matches!(
        verify(&altered, &key, &expected, &plan, &reg, &job, &exe),
        Err(ReceiptError::Rejected(_))
    ));
    std::fs::write(&proof.path, b"Native execution happened\n").unwrap();
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

#[tokio::test]
async fn authenticated_malformed_or_contradictory_payload_is_refused() {
    use blut::framework::openwarrant_receipt::{Envelope, ReceiptError};
    use ed25519_dalek::Signer;
    let td = tempfile::tempdir().unwrap();
    let job = td.path().join("typed-refusals");
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
    assert!(run.result.is_ok());
    let original = std::str::from_utf8(&run.envelope.payload).unwrap();
    let mut unsupported: serde_json::Value = serde_json::from_slice(&run.envelope.payload).unwrap();
    unsupported["version"] = serde_json::json!(2);
    let mut contradictory: serde_json::Value =
        serde_json::from_slice(&run.envelope.payload).unwrap();
    contradictory["outcome"] = serde_json::json!("completed_with_warnings");
    for payload in [
        format!("{},\"version\":1}}", original.strip_suffix('}').unwrap()).into_bytes(),
        format!(
            "{},\"unrecognized\":true}}",
            original.strip_suffix('}').unwrap()
        )
        .into_bytes(),
        serde_json::to_vec(&unsupported).unwrap(),
        serde_json::to_vec(&contradictory).unwrap(),
    ] {
        // Sign exact bytes using the proposed public wire protocol. A valid seal
        // must not make duplicate fields, unknown fields or contradictions valid.
        let mut message = b"blut-openwarrant-receipt-v1".to_vec();
        message.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        message.extend_from_slice(&payload);
        let envelope = Envelope {
            signature: signer.sign(&message).to_bytes().to_vec(),
            payload,
        };
        assert!(matches!(
            verify(
                &envelope,
                &signer.verifying_key(),
                &expected,
                &plan,
                &reg,
                &job,
                &std::env::current_exe().unwrap()
            ),
            Err(ReceiptError::Rejected(_))
        ));
    }
}

#[tokio::test]
async fn actual_deadline_expiry_is_not_completed_or_cancelled() {
    let td = tempfile::tempdir().unwrap();
    let job = td.path().join("deadline");
    let signer = SigningKey::from_bytes(&[7; 32]);
    let reg = registry();
    let plan = spec();
    let expected = binding();
    let mut ctx = ExecCtx::new(job.clone());
    ctx.deadline = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    let run = execute_with_receipt(&plan, &reg, ctx, expected.clone(), &signer)
        .await
        .unwrap();
    assert!(matches!(
        run.result,
        Err(blut::framework::PlanError::DeadlineExceeded { .. })
    ));
    let result = verify(
        &run.envelope,
        &signer.verifying_key(),
        &expected,
        &plan,
        &reg,
        &job,
        &std::env::current_exe().unwrap(),
    )
    .unwrap();
    assert_eq!(result.outcome, Outcome::DeadlineExceeded);
}

#[tokio::test]
async fn actual_stage_failure_cannot_be_reported_as_completed() {
    let td = tempfile::tempdir().unwrap();
    let job = td.path().join("failed");
    let signer = SigningKey::from_bytes(&[7; 32]);
    let reg = registry();
    let plan: PlanSpec = serde_json::from_str(r#"{"name":"receipt-failure","nodes":[{"stage":"receipt_test_fail","args":null}],"edges":[]}"#).unwrap();
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
    assert!(run.result.is_err());
    let result = verify(
        &run.envelope,
        &signer.verifying_key(),
        &expected,
        &plan,
        &reg,
        &job,
        &std::env::current_exe().unwrap(),
    )
    .unwrap();
    assert_eq!(result.outcome, Outcome::Failed);
}

#[tokio::test]
async fn actual_advisory_failure_remains_completion_with_warnings() {
    let td = tempfile::tempdir().unwrap();
    let job = td.path().join("warnings");
    let signer = SigningKey::from_bytes(&[7; 32]);
    let reg = registry();
    let plan: PlanSpec = serde_json::from_str(r#"{"name":"receipt-warning","nodes":[{"stage":"receipt_test_warn","args":null}],"edges":[]}"#).unwrap();
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
    assert_eq!(run.result.unwrap().warnings.len(), 1);
    let result = verify(
        &run.envelope,
        &signer.verifying_key(),
        &expected,
        &plan,
        &reg,
        &job,
        &std::env::current_exe().unwrap(),
    )
    .unwrap();
    assert_eq!(result.outcome, Outcome::CompletedWithWarnings);
}
