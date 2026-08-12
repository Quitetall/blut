// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Production `MeshTaskRunner` over a shared content-addressed store (ADR 0079
//! A3 · the mesh↔execution wiring).
//!
//! The A3 node core made task execution pluggable ([`MeshTaskRunner`]); the CLI
//! ships a smoke runner. This is the REAL one: it executes a dispatched cookbook
//! stage, sourcing its input and sinking its output through a shared
//! [`ObjectStore`] (the cache's remote tier — a shared filesystem or qualified
//! provider adapter), keyed by
//! content hash. No per-task blob transfer is needed: the scheduler
//! [`publish_input`]s the bundle under `input_hash`, the worker reads it, runs
//! the stage, writes the output bundle under `output_hash`, and returns a
//! signed [`TaskResult`] carrying only the hash — the initiator reads the bytes
//! back from the same store.
//!
//! This is the SHARED-CACHE model, the natural fit for the k8s `BlutWorkerPool`
//! (pods share a cache) and a mesh with a common object store. It reuses the
//! bundle/unbundle/run core the legacy `peer_exec` path uses, so the same
//! fail-closed hash-binding + signature checks apply. `Restricted` data never
//! reaches here (the dispatch gate denies it before any dispatch, ADR 0061), so
//! the shared store only ever holds `Public`/`Internal` bundles — un-sealed by
//! design (the store is the cluster's trusted cache).
//!
//! [`publish_input`]: SharedCacheRunner::publish_input

use std::path::PathBuf;
use std::sync::Arc;

use crate::error::TrainError;
use crate::framework::artifact::ContentId;
use crate::framework::artifact_store::StoredArtifact;
use crate::framework::cache::CacheHandle;
use crate::framework::cookbook::Registry;
use crate::framework::object_store::{ObjectKey, ObjectStore};
use crate::framework::stage::{ErasedArtifact, StageContext};
use crate::p2p::bundle::{self, BlobDir};
use crate::p2p::crypto::{KeyPair, verify};
use crate::p2p::dispatch::DispatchPolicy;
use crate::p2p::node::MeshTaskRunner;
use crate::p2p::peer::PeerId;
use crate::p2p::task::{TaskManifest, TaskResult};
use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;

/// Executes dispatched cookbook stages, sourcing/sinking artifacts through a
/// shared content-addressed [`ObjectStore`].
pub struct SharedCacheRunner {
    registry: Arc<Registry>,
    store: ObjectStore,
    work_root: PathBuf,
    keypair: Arc<KeyPair>,
    policy: Arc<dyn DispatchPolicy>,
    /// The scheduler's Ed25519 key — every dispatched manifest must be signed by
    /// it (zero-trust; the TLS connection auth is separate, defense in depth).
    coordinator_verifying: VerifyingKey,
}

impl SharedCacheRunner {
    pub fn new(
        registry: Arc<Registry>,
        store: ObjectStore,
        work_root: PathBuf,
        keypair: Arc<KeyPair>,
        policy: Arc<dyn DispatchPolicy>,
        coordinator_verifying: VerifyingKey,
    ) -> Self {
        Self {
            registry,
            store,
            work_root,
            keypair,
            policy,
            coordinator_verifying,
        }
    }

    /// Scheduler-side: bundle `input` (rooted at `src_root`) and publish it to
    /// the shared store under its store-derived [`ContentId`], so a worker's
    /// [`run`](Self::run) can source it. The stage resolves the artifact's
    /// backing paths.
    pub async fn publish_input(
        store: &ObjectStore,
        registry: &Registry,
        stage_name: &str,
        input: ErasedArtifact,
        src_root: &std::path::Path,
    ) -> Result<ContentId, TrainError> {
        let ctor = registry.find_erased_stage(stage_name).ok_or_else(|| {
            TrainError::other(format!("publish_input: unknown stage '{stage_name}'"))
        })?;
        let stage = ctor();
        let (manifest, pack) = bundle::bundle(&*stage, input, src_root, BlobDir::Input, None)
            .map_err(|e| TrainError::other(format!("bundle input: {e}")))?;
        let input_content_id = manifest.content_id;
        let bytes = bincode::serialize(&StoredArtifact { manifest, pack })
            .map_err(|e| TrainError::other(format!("serialize input bundle: {e}")))?;
        store
            .put(ObjectKey::Artifact(input_content_id), bytes)
            .await
            .map_err(|e| TrainError::other(format!("publish input bundle: {e}")))?;
        Ok(input_content_id)
    }

    /// Initiator-side: read the output bundle a completed task wrote to the
    /// shared store, rebase it into `into_dir`, and return the verified handle.
    pub async fn fetch_output(
        &self,
        stage_name: &str,
        content_id: ContentId,
        into_dir: &std::path::Path,
    ) -> Result<ErasedArtifact, TrainError> {
        let ctor = self.registry.find_erased_stage(stage_name).ok_or_else(|| {
            TrainError::other(format!("fetch_output: unknown stage '{stage_name}'"))
        })?;
        let stage = ctor();
        let shared = self.read_bundle(content_id).await?;
        bundle::unbundle(
            &*stage,
            &shared.manifest,
            &shared.pack,
            into_dir,
            Some(content_id),
            BlobDir::Output,
        )
        .map_err(|e| TrainError::other(format!("unbundle output: {e}")))
    }

    async fn read_bundle(&self, content_id: ContentId) -> Result<StoredArtifact, TrainError> {
        let bytes = self
            .store
            .get(ObjectKey::Artifact(content_id))
            .await
            .map_err(|e| TrainError::other(format!("read shared bundle: {e}")))?
            .ok_or_else(|| {
                TrainError::other(format!("shared store has no artifact for {content_id}"))
            })?;
        bincode::deserialize(&bytes)
            .map_err(|e| TrainError::other(format!("decode shared bundle: {e}")))
    }
}

#[async_trait]
impl MeshTaskRunner for SharedCacheRunner {
    async fn run(&self, task: TaskManifest) -> Result<TaskResult, TrainError> {
        if task.protocol_version != crate::p2p::task::TASK_PROTOCOL_VERSION {
            return Err(TrainError::other(format!(
                "task protocol v{} unsupported (want v{})",
                task.protocol_version,
                crate::p2p::task::TASK_PROTOCOL_VERSION
            )));
        }
        // 1. Zero-trust manifest checks (mirror peer_exec): signature by the
        //    scheduler, args bound, safe task id.
        if !verify(
            &self.coordinator_verifying,
            &task.sign_payload(),
            &task.signature,
        ) {
            return Err(TrainError::other("task manifest signature invalid"));
        }
        task.verify_args(&task.args)?;
        if !crate::p2p::peer_exec::is_safe_task_id(&task.task_id) {
            return Err(TrainError::other(format!(
                "unsafe task_id '{}'",
                task.task_id
            )));
        }

        // 2. Policy gate (training never leaves home, etc.).
        if !self.policy.is_dispatchable(&task.stage_name) {
            return Err(TrainError::other(format!(
                "stage '{}' is not dispatchable",
                task.stage_name
            )));
        }

        // 3. Resolve the stage (the worker hosts the cookbook).
        let ctor = self
            .registry
            .find_erased_stage(&task.stage_name)
            .ok_or_else(|| TrainError::other(format!("unknown stage '{}'", task.stage_name)))?;
        let stage = ctor();

        // 4. Per-task work dir (task_id validated safe above).
        let stage_dir = self.work_root.join(&task.task_id);
        std::fs::create_dir_all(&stage_dir)
            .map_err(|e| TrainError::other(format!("create stage dir: {e}")))?;

        // 5. Source the input bundle from the shared store + unbundle (the four
        //    fail-closed gates run against task.input_hash inside unbundle).
        let shared_in = self.read_bundle(task.input_content_id).await?;
        let input = bundle::unbundle(
            &*stage,
            &shared_in.manifest,
            &shared_in.pack,
            &stage_dir,
            Some(task.input_content_id),
            BlobDir::Input,
        )
        .map_err(|e| TrainError::other(format!("unbundle input: {e}")))?;

        // 6. Run the stage under the coordinator's deadline.
        let cache = Arc::new(CacheHandle::job_local(stage_dir.join(".cache")));
        let ctx = StageContext::for_peer(
            stage_dir.clone(),
            stage_dir.clone(),
            cache,
            task.invocation_key,
        );
        let started = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(task.timeout_secs.max(1));
        let output =
            tokio::time::timeout(timeout, stage.run_erased(&ctx, input, task.args.clone()))
                .await
                .map_err(|_| {
                    TrainError::other(format!(
                        "stage '{}' exceeded timeout_secs {}",
                        task.stage_name, task.timeout_secs
                    ))
                })?
                .map_err(|e| TrainError::other(format!("stage run failed: {e}")))?;
        let wall_time_ms = started.elapsed().as_millis() as u64;

        // 7. Bundle the output + publish it to the shared store under its hash
        //    (verified == task.expected_output_hash inside bundle()).
        let (out_manifest, out_pack) = bundle::bundle(
            &*stage,
            output,
            &stage_dir,
            BlobDir::Output,
            task.expected_content_id,
        )
        .map_err(|e| TrainError::other(format!("bundle output: {e}")))?;
        let output_content_id = out_manifest.content_id;
        let out_bytes = bincode::serialize(&StoredArtifact {
            manifest: out_manifest,
            pack: out_pack,
        })
        .map_err(|e| TrainError::other(format!("serialize output bundle: {e}")))?;
        self.store
            .put(ObjectKey::Artifact(output_content_id), out_bytes)
            .await
            .map_err(|e| TrainError::other(format!("publish output bundle: {e}")))?;

        // Best-effort cleanup of the work dir (the output is on the store now).
        let _ = std::fs::remove_dir_all(&stage_dir);

        // 8. Sign + return the result (output on the shared store, not sealed).
        let mut result = TaskResult {
            protocol_version: crate::p2p::task::TASK_PROTOCOL_VERSION,
            task_id: task.task_id,
            peer_id: PeerId::from_pubkey(&self.keypair.verifying),
            content_id: output_content_id,
            encrypted_output: None,
            wall_time_ms,
            signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
        };
        result.signature = self.keypair.sign(&result.sign_payload());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::object_store::ObjectStore;
    use crate::framework::{ContentHash, InvocationKey};
    use crate::p2p::dispatch::DefaultDispatchPolicy;
    use crate::p2p::smoke::{SMOKE_STAGE, SmokeText, expected_echo_hash};
    use crate::p2p::trust::DispatchMatrix;

    /// End-to-end (single process): a scheduler publishes an echo input to a
    /// shared store, the runner executes `p2p-echo` sourcing/sinking through
    /// that store, and the signed result's output hash matches — and the output
    /// bytes are on the store for the initiator to fetch.
    #[tokio::test]
    async fn shared_cache_runner_executes_a_real_stage() {
        let store_dir = tempfile::tempdir().unwrap();
        let store = ObjectStore::filesystem(store_dir.path());

        let mut registry = Registry::new();
        crate::p2p::smoke::register(&mut registry);
        let registry = Arc::new(registry);
        let policy: Arc<dyn DispatchPolicy> =
            Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default()));

        let coordinator = Arc::new(KeyPair::generate());
        let worker = Arc::new(KeyPair::generate());

        // Scheduler side: build + publish the echo input.
        let src_root = tempfile::tempdir().unwrap();
        let in_path = src_root.path().join("p2p-echo-in.txt");
        std::fs::write(&in_path, b"hello mesh runner").unwrap();
        let input_hash = ContentHash::hash_file(&in_path).unwrap();
        let input = SmokeText {
            content_hash: input_hash,
            path: in_path,
        };
        let input_erased = ErasedArtifact::from_typed(&input).unwrap();
        let input_content_id = SharedCacheRunner::publish_input(
            &store,
            &registry,
            SMOKE_STAGE,
            input_erased,
            src_root.path(),
        )
        .await
        .unwrap();

        // The scheduler's signed manifest.
        let expected_logical = expected_echo_hash("hello mesh runner");
        let mut task = TaskManifest {
            protocol_version: crate::p2p::task::TASK_PROTOCOL_VERSION,
            task_id: "mesh-run-1".into(),
            coordinator_id: PeerId::from_pubkey(&coordinator.verifying),
            stage_name: SMOKE_STAGE.into(),
            stage_schema: 1,
            input_content_id,
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"mesh-run-1")),
            args_hash: ContentHash::of_bytes(b"{}"),
            expected_content_id: None,
            args: serde_json::json!({}),
            resources: crate::p2p::task::ResourceRequest::default(),
            data_class: crate::p2p::trust::DataClass::Public,
            timeout_secs: 30,
            deadline: crate::framework::execution::ExecutionDeadline::from_now(
                None,
                std::time::Duration::from_secs(30),
            ),
            encrypted_input: None,
            signature: coordinator.sign(b"placeholder"),
        };
        task.signature = coordinator.sign(&task.sign_payload());

        // Worker side: run it.
        let work_root = tempfile::tempdir().unwrap();
        let runner = SharedCacheRunner::new(
            registry.clone(),
            store.clone(),
            work_root.path().to_path_buf(),
            worker.clone(),
            policy,
            coordinator.verifying,
        );
        let result = runner.run(task).await.expect("stage executes");

        assert_ne!(result.content_id, input_content_id);
        assert_eq!(result.peer_id, PeerId::from_pubkey(&worker.verifying));
        assert!(
            crate::p2p::crypto::verify(
                &worker.verifying,
                &result.sign_payload(),
                &result.signature
            ),
            "worker's result signature verifies"
        );

        // The initiator fetches the output from the shared store.
        let out_dir = tempfile::tempdir().unwrap();
        let out = runner
            .fetch_output(SMOKE_STAGE, result.content_id, out_dir.path())
            .await
            .expect("fetch output from shared store");
        let out: SmokeText = out.into_typed().unwrap();
        assert_eq!(out.content_hash, expected_logical);
        let body = std::fs::read_to_string(&out.path).unwrap();
        assert_eq!(
            body, "HELLO MESH RUNNER",
            "echo stage ran + output round-trips"
        );
    }

    #[tokio::test]
    async fn forged_manifest_signature_is_rejected() {
        let store = ObjectStore::filesystem(tempfile::tempdir().unwrap().path());
        let mut registry = Registry::new();
        crate::p2p::smoke::register(&mut registry);
        let policy: Arc<dyn DispatchPolicy> =
            Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default()));
        let coordinator = KeyPair::generate();
        let impostor = KeyPair::generate();
        let worker = Arc::new(KeyPair::generate());

        // A manifest signed by an impostor, but the runner pins the coordinator.
        let mut task = TaskManifest {
            protocol_version: crate::p2p::task::TASK_PROTOCOL_VERSION,
            task_id: "x".into(),
            coordinator_id: PeerId::from_pubkey(&coordinator.verifying),
            stage_name: SMOKE_STAGE.into(),
            stage_schema: 1,
            input_content_id: ContentId::from_digest(ContentHash::of_bytes(b"i")),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"x")),
            args_hash: ContentHash::of_bytes(b"{}"),
            expected_content_id: Some(ContentId::from_digest(ContentHash::of_bytes(b"o"))),
            args: serde_json::json!({}),
            resources: crate::p2p::task::ResourceRequest::default(),
            data_class: crate::p2p::trust::DataClass::Public,
            timeout_secs: 30,
            deadline: crate::framework::execution::ExecutionDeadline::from_now(
                None,
                std::time::Duration::from_secs(30),
            ),
            encrypted_input: None,
            signature: impostor.sign(b"placeholder"),
        };
        task.signature = impostor.sign(&task.sign_payload());

        let runner = SharedCacheRunner::new(
            Arc::new(registry),
            store,
            tempfile::tempdir().unwrap().path().to_path_buf(),
            worker,
            policy,
            coordinator.verifying, // pins the real coordinator
        );
        let err = runner.run(task).await.unwrap_err();
        assert!(format!("{err}").contains("signature invalid"));
    }
}
