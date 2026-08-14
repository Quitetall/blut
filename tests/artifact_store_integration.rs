// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0092 A09: cross-host artifact ownership and identity contract.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use blut::backends::TrainingBackend;
use blut::framework::artifact::{Artifact, ArtifactContentId, ContentHash};
use blut::framework::artifact_store::StoredArtifact;
use blut::framework::compat::Compatible;
use blut::framework::executor::{ExecCtx, SequentialExecutor};
use blut::framework::object_store::{BlockingObjectStore, ObjectKey, ObjectNamespace};
use blut::framework::plan::Plan;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};
use blut::framework::status::{HostedEvent, StageEvent};
use blut::framework::{CacheHandle, InvocationKey, StageError};
use serde::{Deserialize, Serialize};

static RUNS: AtomicUsize = AtomicUsize::new(0);

struct TestBackend;

impl TrainingBackend for TestBackend {
    const ID: &'static str = "a09-test";
    const DESCRIPTION: &'static str = "ADR 0092 A09 integration-test backend";
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FileAndDirectory {
    file_path: PathBuf,
    directory_path: PathBuf,
    content_hash: ContentHash,
}

impl Artifact for FileAndDirectory {
    const KIND: &'static str = "test.file-and-directory";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }

    fn primary_path(&self) -> &Path {
        &self.directory_path
    }

    fn recompute_content_hash(&self) -> std::io::Result<ContentHash> {
        combined_hash(&self.file_path, &self.directory_path)
    }
}

fn combined_hash(file: &Path, directory: &Path) -> std::io::Result<ContentHash> {
    let file_hash = ContentHash::hash_file(file)?;
    let directory_hash = ContentHash::hash_dir(directory)?;
    let mut bytes = b"blut.test.file-and-directory.v1".to_vec();
    bytes.extend_from_slice(&file_hash.0);
    bytes.extend_from_slice(&directory_hash.0);
    Ok(ContentHash::of_bytes(&bytes))
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct ProduceArgs {
    label: String,
}

struct ProduceArtifacts;

fn stage_io<T>(path: &Path, result: std::io::Result<T>) -> Result<T, StageError> {
    result.map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[async_trait]
impl Stage for ProduceArtifacts {
    const NAME: &'static str = "produce_artifacts";
    const SCHEMA: u32 = 3;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = ();
    type Output = FileAndDirectory;
    type Args = ProduceArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &Self::Args,
    ) -> Result<Self::Output, StageError> {
        RUNS.fetch_add(1, Ordering::SeqCst);
        let file_path = ctx.stage_dir.join("result.txt");
        let directory_path = ctx.stage_dir.join("tree");
        stage_io(
            &directory_path,
            std::fs::create_dir_all(directory_path.join("nested")),
        )?;
        stage_io(
            &file_path,
            std::fs::write(&file_path, format!("file:{}", args.label)),
        )?;
        stage_io(
            &directory_path,
            std::fs::write(directory_path.join("root.bin"), b"root bytes"),
        )?;
        stage_io(
            &directory_path,
            std::fs::write(directory_path.join("nested/leaf.bin"), b"leaf bytes"),
        )?;
        let content_hash = stage_io(&directory_path, combined_hash(&file_path, &directory_path))?;
        Ok(FileAndDirectory {
            file_path,
            directory_path,
            content_hash,
        })
    }
}

impl Compatible<TestBackend> for ProduceArtifacts {}

fn plan() -> blut::framework::CompiledPlan {
    Plan::<(), TestBackend>::new("a09-artifact-store", serde_json::json!({}))
        .start(
            ProduceArtifacts,
            ProduceArgs {
                label: "portable".into(),
            },
        )
        .finish()
        .into_compiled()
}

fn context(job_dir: &Path, cache: Arc<CacheHandle>) -> ExecCtx {
    let mut context = ExecCtx::new(job_dir.to_path_buf());
    context.cache = cache;
    context
}

fn lifecycle_events(job_dir: &Path) -> Vec<StageEvent> {
    std::fs::read_to_string(job_dir.join("status.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<HostedEvent>(line).unwrap().event)
        .collect()
}

fn stage_end_content_id(events: &[StageEvent]) -> ArtifactContentId {
    events
        .iter()
        .find_map(|event| match event {
            StageEvent::StageEnd {
                content_id: Some(content_id),
                ..
            } => Some(*content_id),
            _ => None,
        })
        .expect("cold run emits StageEnd")
}

fn stage_skipped_content_id(events: &[StageEvent]) -> ArtifactContentId {
    events
        .iter()
        .find_map(|event| match event {
            StageEvent::StageSkipped {
                content_id: Some(content_id),
                ..
            } => Some(*content_id),
            _ => None,
        })
        .expect("warm run emits StageSkipped with content identity")
}

#[tokio::test]
async fn shared_cache_rehydrates_file_and_directory_after_producer_deletion() {
    RUNS.store(0, Ordering::SeqCst);
    let workspace = tempfile::tempdir().unwrap();
    let remote_root = workspace.path().join("shared-object-store");
    let remote = BlockingObjectStore::filesystem(remote_root);

    let host_a = workspace.path().join("host-a-job");
    let cache_a =
        Arc::new(CacheHandle::job_local(host_a.join("_cache")).with_remote(remote.clone()));
    let cold = SequentialExecutor::execute(plan(), context(&host_a, cache_a))
        .await
        .unwrap();
    assert_eq!(cold.n_cache_hits, 0);
    assert_eq!(cold.n_cache_misses, 1);
    let cold_artifact = cold.final_output.unwrap();
    let cold_typed: FileAndDirectory = cold_artifact.into_typed().unwrap();
    let expected_file = std::fs::read(&cold_typed.file_path).unwrap();
    let expected_root = std::fs::read(cold_typed.directory_path.join("root.bin")).unwrap();
    let expected_leaf = std::fs::read(cold_typed.directory_path.join("nested/leaf.bin")).unwrap();
    let cold_id = stage_end_content_id(&lifecycle_events(&host_a));
    assert_ne!(
        cold_id.digest(),
        cold_typed.content_hash,
        "portable ArtifactContentId is independent from the artifact logical hash"
    );

    std::fs::remove_dir_all(&host_a).unwrap();
    assert!(
        !host_a.exists(),
        "host A job and every producer path are gone"
    );

    let host_b = workspace.path().join("host-b-job");
    let cache_b_root = host_b.join("_cache");
    let cache_b =
        Arc::new(CacheHandle::job_local(cache_b_root.clone()).with_remote(remote.clone()));
    let warm = SequentialExecutor::execute(plan(), context(&host_b, cache_b))
        .await
        .unwrap();
    assert_eq!(warm.n_cache_hits, 1);
    assert_eq!(warm.n_cache_misses, 0);
    assert_eq!(
        RUNS.load(Ordering::SeqCst),
        1,
        "warm hit did not rerun stage"
    );

    let warm_artifact = warm.final_output.unwrap();
    let warm_typed: FileAndDirectory = warm_artifact.clone().into_typed().unwrap();
    assert!(warm_typed.file_path.starts_with(&host_b));
    assert!(warm_typed.directory_path.starts_with(&host_b));
    assert_eq!(std::fs::read(&warm_typed.file_path).unwrap(), expected_file);
    assert_eq!(
        std::fs::read(warm_typed.directory_path.join("root.bin")).unwrap(),
        expected_root
    );
    assert_eq!(
        std::fs::read(warm_typed.directory_path.join("nested/leaf.bin")).unwrap(),
        expected_leaf
    );
    assert_eq!(
        warm_typed.recompute_content_hash().unwrap(),
        cold_typed.content_hash
    );

    let warm_id = stage_skipped_content_id(&lifecycle_events(&host_b));
    assert_eq!(warm_id, cold_id, "cold and warm lineage identities match");

    let invocation = BlockingObjectStore::filesystem(&cache_b_root)
        .list_namespace(ObjectNamespace::CacheInvocation)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    assert_ne!(invocation.key.digest().to_hex(), cold_id.to_hex());

    // Build a known invocation entry for direct missing/corruption assertions.
    let fault_root = workspace.path().join("fault-cache");
    let fault_cache = CacheHandle::job_local(fault_root.clone());
    let invocation_key = CacheHandle::key_for(
        ProduceArtifacts::NAME,
        ProduceArtifacts::SCHEMA,
        ContentHash::of_bytes(b"unit-input"),
        &serde_json::json!({"label": "portable"}),
        b"test-code-v1",
    );
    assert_ne!(invocation_key.to_hex(), cold_id.to_hex());
    let source_root = host_b.join("stages/0-produce_artifacts");
    let inserted_id = fault_cache
        .insert(
            invocation_key,
            &ProduceArtifacts,
            &warm_artifact,
            &source_root,
        )
        .unwrap();
    assert_eq!(inserted_id, cold_id);

    let fault_store = BlockingObjectStore::filesystem(&fault_root);
    let object_key = ObjectKey::Artifact(cold_id);
    let object_bytes = fault_store.get(object_key).unwrap().unwrap();
    let stored: StoredArtifact = bincode::deserialize(&object_bytes).unwrap();
    let portable: FileAndDirectory = stored.manifest.erased.clone().into_typed().unwrap();
    assert!(!stored.manifest.handle_root.is_absolute());
    assert!(!portable.file_path.is_absolute());
    assert!(!portable.directory_path.is_absolute());
    assert!(!portable.file_path.to_string_lossy().contains("host-b-job"));

    assert!(fault_store.remove(object_key).unwrap());
    assert!(
        fault_cache
            .lookup(
                invocation_key,
                &ProduceArtifacts,
                &workspace.path().join("missing-consumer"),
            )
            .is_none(),
        "missing content object is a miss"
    );

    let mut modified: StoredArtifact = bincode::deserialize(&object_bytes).unwrap();
    let last = modified
        .pack
        .last_mut()
        .expect("file+dir pack is non-empty");
    *last ^= 0x80;
    fault_store
        .put(object_key, &bincode::serialize(&modified).unwrap())
        .unwrap();
    let modified_consumer = workspace.path().join("modified-consumer");
    assert!(
        fault_cache
            .lookup(invocation_key, &ProduceArtifacts, &modified_consumer)
            .is_none(),
        "modified payload is a miss"
    );
    assert!(
        !modified_consumer
            .join(".artifact-import")
            .join(cold_id.to_hex())
            .exists(),
        "failed restore removes its private materialization"
    );
}

// Compile-time domain separation is additionally locked by the compile-fail
// doctest on `InvocationKey::from_digest`; keep this typed helper in the
// integration target so both public identity types remain part of the gate.
#[allow(dead_code)]
fn invocation_only(_key: InvocationKey) {}
