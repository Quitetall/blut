// SPDX-License-Identifier: AGPL-3.0-or-later
//! ADR 0092 A08 execution-lifecycle contract.

use std::future::pending;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use blut::framework::artifact::{Artifact, ContentHash, ContentId, InvocationKey};
use blut::framework::artifact_store::{ARTIFACT_FORMAT_VERSION, ArtifactManifest, StoredArtifact};
use blut::framework::error::StageError;
use blut::framework::error_domain::StageFailure;
use blut::framework::execution::{
    Assignment, DataClassification, EXECUTION_PROTOCOL_VERSION, ExecutionAdapter,
    ExecutionArtifact, ExecutionDeadline, ExecutionFailure, ExecutionHandle, ExecutionLifecycle,
    ExecutionMode, ExecutionPhase, ExecutionRequest, ExecutionResources, ExecutionResult,
    ExecutionSnapshot, ExecutionTerminal, LifecycleError, drive_execution,
};
use blut::framework::stage::ErasedArtifact;
use blut::framework::{Resource, Stage, StageContext};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ContractFile {
    path: PathBuf,
    content_hash: ContentHash,
}

impl Artifact for ContractFile {
    const KIND: &'static str = "contract";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }

    fn primary_path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
struct ContractArgs {}

struct ContractStage;

#[async_trait]
impl Stage for ContractStage {
    const NAME: &'static str = "contract";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ContractFile;
    type Output = ContractFile;
    type Args = ContractArgs;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: Self::Input,
        _args: &Self::Args,
    ) -> Result<Self::Output, StageError> {
        Ok(input)
    }
}

fn content_id(label: &[u8]) -> ContentId {
    ContentId::from_digest(ContentHash::of_bytes(label))
}

fn stored(label: &[u8]) -> StoredArtifact {
    let id = content_id(label);
    StoredArtifact {
        manifest: ArtifactManifest {
            format_version: ARTIFACT_FORMAT_VERSION,
            erased: ErasedArtifact {
                kind: "contract".into(),
                schema: 1,
                payload: Vec::new(),
            },
            kind: "contract".into(),
            schema: 1,
            content_id: id,
            logical_hash: ContentHash::of_bytes(label),
            handle_root: "__blut_artifact_root_v2__".into(),
            files: Vec::new(),
            blob_len: 0,
            blob_sha256: ContentHash::of_bytes(&[]),
        },
        pack: Vec::new(),
    }
}

fn request(deadline: ExecutionDeadline) -> ExecutionRequest {
    let input = stored(b"input");
    ExecutionRequest {
        protocol_version: EXECUTION_PROTOCOL_VERSION,
        execution_id: "contract-execution".into(),
        tenant: blut::tenant::Tenant::default(),
        stage_name: "contract".into(),
        stage_schema: 1,
        invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"invocation")),
        args_hash: ContentHash::of_bytes(b"{}"),
        args: serde_json::json!({}),
        input,
        expected_content_id: None,
        resources: ExecutionResources::default(),
        data_class: DataClassification::Public,
        deadline,
    }
}

fn success(label: &[u8]) -> ExecutionTerminal {
    let stored = stored(label);
    ExecutionTerminal::Succeeded {
        artifact: ExecutionArtifact {
            content_id: stored.manifest.content_id,
            stored: Some(stored),
        },
        wall_time_ms: 7,
    }
}

#[test]
fn every_mode_obeys_the_same_transition_and_terminal_contract() {
    for mode in [
        ExecutionMode::Local,
        ExecutionMode::P2p,
        ExecutionMode::Cloud,
    ] {
        let lifecycle = ExecutionLifecycle::new(mode);
        let assignment = Assignment::new(format!("{mode:?}-worker"), 1);
        lifecycle
            .transition(ExecutionPhase::Assigned, Some(assignment.clone()))
            .unwrap();
        lifecycle.transition(ExecutionPhase::Running, None).unwrap();
        let terminal = lifecycle
            .finish(Some(&assignment), success(b"output"))
            .unwrap();
        assert!(matches!(
            terminal.terminal,
            Some(ExecutionTerminal::Succeeded { .. })
        ));
        assert!(
            matches!(
                lifecycle.finish(
                    Some(&assignment),
                    ExecutionTerminal::Cancelled {
                        reason: "late cancel".into()
                    }
                ),
                Err(LifecycleError::AlreadyTerminal)
            ),
            "{mode:?} must keep exactly one terminal outcome"
        );
    }
}

#[test]
fn assignment_generation_fences_late_completion_even_for_same_owner() {
    let lifecycle = ExecutionLifecycle::new(ExecutionMode::Cloud);
    let first = Assignment::new("worker", 1);
    let second = Assignment::new("worker", 2);
    lifecycle.transition(ExecutionPhase::Queued, None).unwrap();
    lifecycle
        .transition(ExecutionPhase::Assigned, Some(first.clone()))
        .unwrap();
    lifecycle.transition(ExecutionPhase::Running, None).unwrap();
    assert!(matches!(
        lifecycle.transition(ExecutionPhase::Queued, Some(first.clone())),
        Err(LifecycleError::UnexpectedAssignment(ExecutionPhase::Queued))
    ));
    lifecycle.transition(ExecutionPhase::Queued, None).unwrap();
    assert!(matches!(
        lifecycle.transition(ExecutionPhase::Assigned, Some(first.clone())),
        Err(LifecycleError::StaleAssignment)
    ));
    lifecycle
        .transition(ExecutionPhase::Assigned, Some(second.clone()))
        .unwrap();
    lifecycle.transition(ExecutionPhase::Running, None).unwrap();

    assert!(matches!(
        lifecycle.finish(Some(&first), success(b"stale")),
        Err(LifecycleError::StaleAssignment)
    ));
    assert!(lifecycle.finish(Some(&second), success(b"fresh")).is_ok());
}

#[test]
fn typed_stage_failure_survives_serialization() {
    let typed = StageFailure::new("E_CONTRACT", "contract")
        .stage("contract")
        .message("typed failure");
    let failure = ExecutionFailure {
        kind: blut::framework::execution::ExecutionFailureKind::Stage,
        code: typed.code.clone(),
        message: typed.message.clone(),
        retryable: false,
        stage_failure: Some(typed),
    };

    let bytes = serde_json::to_vec(&failure).unwrap();
    let decoded: ExecutionFailure = serde_json::from_slice(&bytes).unwrap();
    let stage_failure = decoded.stage_failure.expect("typed identity preserved");
    assert_eq!(stage_failure.code, "E_CONTRACT");
    assert_eq!(stage_failure.domain, "contract");
    assert_eq!(stage_failure.stage.as_deref(), Some("contract"));
}

#[test]
fn execution_failure_retryability_survives_stage_error_conversion() {
    for retryable in [false, true] {
        let mut failure = ExecutionFailure::transport("contract transport");
        failure.retryable = retryable;
        let error = failure.into_stage_error();
        assert!(matches!(error, StageError::Execution(_)));
        assert_eq!(
            blut::framework::retry::is_retryable(
                &error,
                blut::framework::retry::RetryOn::Transient
            ),
            retryable
        );
    }
}

enum AdapterBehavior {
    HangSubmit,
    Handle(Arc<ScriptHandle>),
}

struct ScriptAdapter {
    mode: ExecutionMode,
    behavior: AdapterBehavior,
}

#[async_trait]
impl ExecutionAdapter for ScriptAdapter {
    fn mode(&self) -> ExecutionMode {
        self.mode
    }

    async fn submit(
        &self,
        _request: ExecutionRequest,
    ) -> Result<Box<dyn ExecutionHandle>, ExecutionFailure> {
        match &self.behavior {
            AdapterBehavior::HangSubmit => pending().await,
            AdapterBehavior::Handle(handle) => Ok(Box::new(ScriptHandleRef(handle.clone()))),
        }
    }
}

struct ScriptHandle {
    lifecycle: ExecutionLifecycle,
    hang_poll: bool,
    cancel_calls: AtomicUsize,
}

impl ScriptHandle {
    fn hanging(mode: ExecutionMode) -> Arc<Self> {
        Arc::new(Self {
            lifecycle: ExecutionLifecycle::new(mode),
            hang_poll: true,
            cancel_calls: AtomicUsize::new(0),
        })
    }

    fn completed(mode: ExecutionMode, terminal: ExecutionTerminal) -> Arc<Self> {
        let lifecycle = ExecutionLifecycle::new(mode);
        let assignment = Assignment::new(format!("{mode:?}-worker"), 1);
        lifecycle
            .transition(ExecutionPhase::Assigned, Some(assignment.clone()))
            .unwrap();
        lifecycle.transition(ExecutionPhase::Running, None).unwrap();
        lifecycle.finish(Some(&assignment), terminal).unwrap();
        Arc::new(Self {
            lifecycle,
            hang_poll: false,
            cancel_calls: AtomicUsize::new(0),
        })
    }
}

struct ScriptHandleRef(Arc<ScriptHandle>);

#[async_trait]
impl ExecutionHandle for ScriptHandleRef {
    async fn snapshot(&self) -> Result<ExecutionSnapshot, ExecutionFailure> {
        if self.0.hang_poll {
            pending().await
        } else {
            Ok(self.0.lifecycle.snapshot())
        }
    }

    async fn cancel(&self) -> Result<(), ExecutionFailure> {
        self.0.cancel_calls.fetch_add(1, Ordering::SeqCst);
        let _ = self.0.lifecycle.finish(
            None,
            ExecutionTerminal::Cancelled {
                reason: "contract cancel".into(),
            },
        );
        Ok(())
    }
}

#[tokio::test]
async fn cancellation_interrupts_a_hanging_submit() {
    let adapter = ScriptAdapter {
        mode: ExecutionMode::Cloud,
        behavior: AdapterBehavior::HangSubmit,
    };
    let cancellation = tokio_util::sync::CancellationToken::new();
    cancellation.cancel();

    let submit = adapter.submit(request(ExecutionDeadline::from_now(
        None,
        Duration::from_secs(30),
    )));
    tokio::pin!(submit);
    tokio::select! {
        _ = cancellation.cancelled() => {}
        _ = &mut submit => panic!("hanging submit unexpectedly completed"),
    }
}

#[tokio::test]
async fn adapter_returns_the_shared_lifecycle_handle() {
    let scripted = ScriptHandle::hanging(ExecutionMode::Local);
    let adapter = ScriptAdapter {
        mode: ExecutionMode::Local,
        behavior: AdapterBehavior::Handle(scripted.clone()),
    };
    let handle = adapter
        .submit(request(ExecutionDeadline::from_now(
            None,
            Duration::from_secs(30),
        )))
        .await
        .unwrap();

    handle.cancel().await.unwrap();
    assert_eq!(scripted.cancel_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancellation_interrupts_a_hanging_poll_and_signals_the_handle() {
    let handle = ScriptHandle::hanging(ExecutionMode::P2p);
    let cancellation = tokio_util::sync::CancellationToken::new();
    cancellation.cancel();
    let handle_ref = ScriptHandleRef(handle.clone());

    tokio::select! {
        _ = cancellation.cancelled() => handle_ref.cancel().await.unwrap(),
        result = handle_ref.snapshot() => panic!("hanging poll completed: {result:?}"),
    }
    assert_eq!(handle.cancel_calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        handle.lifecycle.snapshot().terminal,
        Some(ExecutionTerminal::Cancelled { .. })
    ));
}

#[tokio::test]
async fn driver_restores_a_validated_success_artifact() {
    use blut::framework::artifact_store::{ArtifactRole, capture};

    let source = tempfile::tempdir().unwrap();
    let source_path = source.path().join("output.txt");
    std::fs::write(&source_path, b"RESTORED").unwrap();
    let typed = ContractFile {
        path: source_path.clone(),
        content_hash: ContentHash::hash_file(&source_path).unwrap(),
    };
    let stage: Arc<dyn blut::framework::stage::StageDyn> = Arc::new(ContractStage);
    let stored = capture(
        stage.as_ref(),
        ErasedArtifact::from_typed(&typed).unwrap(),
        source.path(),
        ArtifactRole::Output,
        None,
    )
    .unwrap();
    let content_id = stored.manifest.content_id;
    let handle = ScriptHandle::completed(
        ExecutionMode::Local,
        ExecutionTerminal::Succeeded {
            artifact: ExecutionArtifact {
                content_id,
                stored: Some(stored),
            },
            wall_time_ms: 9,
        },
    );
    let adapter = ScriptAdapter {
        mode: ExecutionMode::Local,
        behavior: AdapterBehavior::Handle(handle),
    };
    let mut execution_request = request(ExecutionDeadline::from_now(None, Duration::from_secs(5)));
    execution_request.expected_content_id = Some(content_id);
    let destination = tempfile::tempdir().unwrap();

    let result = drive_execution(
        &adapter,
        execution_request,
        &tokio_util::sync::CancellationToken::new(),
        stage,
        destination.path(),
        Duration::from_millis(1),
    )
    .await;
    match result {
        ExecutionResult::Succeeded {
            artifact,
            content_id: actual,
            wall_time_ms,
        } => {
            let restored = artifact.into_typed::<ContractFile>().unwrap();
            assert_eq!(actual, content_id);
            assert_eq!(wall_time_ms, 9);
            assert_eq!(std::fs::read(restored.primary_path()).unwrap(), b"RESTORED");
            assert!(restored.primary_path().starts_with(destination.path()));
        }
        _ => panic!("validated success did not complete"),
    }
}

#[tokio::test]
async fn driver_hard_deadline_interrupts_hanging_poll_and_cancels_handle() {
    let handle = ScriptHandle::hanging(ExecutionMode::Cloud);
    let adapter = ScriptAdapter {
        mode: ExecutionMode::Cloud,
        behavior: AdapterBehavior::Handle(handle.clone()),
    };
    let destination = tempfile::tempdir().unwrap();
    let result = drive_execution(
        &adapter,
        request(ExecutionDeadline::from_now(None, Duration::from_millis(20))),
        &tokio_util::sync::CancellationToken::new(),
        Arc::new(ContractStage),
        destination.path(),
        Duration::from_millis(1),
    )
    .await;

    assert!(matches!(result, ExecutionResult::TimedOut { .. }));
    assert_eq!(handle.cancel_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn driver_pre_cancel_interrupts_hanging_submit() {
    let adapter = ScriptAdapter {
        mode: ExecutionMode::P2p,
        behavior: AdapterBehavior::HangSubmit,
    };
    let cancellation = tokio_util::sync::CancellationToken::new();
    cancellation.cancel();
    let destination = tempfile::tempdir().unwrap();

    let result = drive_execution(
        &adapter,
        request(ExecutionDeadline::from_now(None, Duration::from_secs(5))),
        &cancellation,
        Arc::new(ContractStage),
        destination.path(),
        Duration::from_millis(1),
    )
    .await;
    assert!(matches!(result, ExecutionResult::Cancelled));
}

#[test]
fn deadline_is_absolute_and_fail_closed() {
    let deadline =
        ExecutionDeadline::from_now(Some(Duration::from_millis(10)), Duration::from_millis(20));
    assert!(deadline.soft_unix_ms.unwrap() <= deadline.hard_unix_ms);
    assert!(ExecutionDeadline::absolute(Some(20), 10).is_err());
}
