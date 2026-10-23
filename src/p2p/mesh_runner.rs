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
//! **Nodes that do not share a store.** Attach the node's object plane with
//! [`SharedCacheRunner::with_mesh_objects`] and a worker that lacks a task's
//! input fetches it, by content id, from the peer that dispatched the task
//! ([`crate::p2p::objects`]); the dispatcher fetches the output back the same
//! way ([`dispatch_stage`]). A shared store still works and skips the transfer:
//! an object already in the store is never fetched.
//!
//! [`publish_input`]: SharedCacheRunner::publish_input

use std::path::PathBuf;
use std::sync::Arc;

use crate::error::TrainError;
use crate::framework::artifact::ArtifactContentId;
use crate::framework::artifact_store::StoredArtifact;
use crate::framework::cache::CacheHandle;
use crate::framework::cookbook::Registry;
use crate::framework::object_store::{ObjectKey, ObjectStore};
use crate::framework::stage::{ErasedArtifact, StageContext};
use crate::p2p::bundle::{self, BlobDir};
use crate::p2p::crypto::{KeyPair, verify};
use crate::p2p::dispatch::DispatchPolicy;
use crate::p2p::node::{MeshNode, MeshTaskRunner, TaskOrigin};
use crate::p2p::objects::MeshObjects;
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
    /// When set, an input missing from `store` is fetched from the task's
    /// origin peer over the mesh.
    objects: Option<Arc<MeshObjects>>,
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
            objects: None,
        }
    }

    /// Fetch inputs that are not in the store from the dispatching peer, over
    /// `objects` (which should wrap the same store).
    pub fn with_mesh_objects(mut self, objects: Arc<MeshObjects>) -> Self {
        self.objects = Some(objects);
        self
    }

    /// Scheduler-side: bundle `input` (rooted at `src_root`) and publish it to
    /// the shared store under its store-derived [`ArtifactContentId`], so a worker's
    /// [`run`](Self::run) can source it. The stage resolves the artifact's
    /// backing paths.
    pub async fn publish_input(
        store: &ObjectStore,
        registry: &Registry,
        stage_name: &str,
        input: ErasedArtifact,
        src_root: &std::path::Path,
    ) -> Result<ArtifactContentId, TrainError> {
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
        content_id: ArtifactContentId,
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

    async fn read_bundle(
        &self,
        content_id: ArtifactContentId,
    ) -> Result<StoredArtifact, TrainError> {
        try_read_bundle(&self.store, content_id)
            .await?
            .ok_or_else(|| {
                TrainError::other(format!("shared store has no artifact for {content_id}"))
            })
    }

    async fn execute(
        &self,
        task: TaskManifest,
        origin: Option<&TaskOrigin>,
    ) -> Result<TaskResult, TrainError> {
        if task.protocol_version != crate::p2p::task::TASK_PROTOCOL_VERSION {
            return Err(TrainError::other(format!(
                "task protocol v{} unsupported (want v{})",
                task.protocol_version,
                crate::p2p::task::TASK_PROTOCOL_VERSION
            )));
        }
        // 1. Zero-trust manifest checks (mirror peer_exec): signature by the
        //    scheduler, args bound, safe task id. All of this runs BEFORE any
        //    input is fetched, so an unsigned task never makes us download.
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

        // 5. Source the input bundle: from the store, or — when it is not
        //    there and the node has an object plane — from the peer that
        //    dispatched the task. Then unbundle (the four fail-closed gates run
        //    against task.input_content_id inside unbundle). A fetched object
        //    is written to the store only after unbundle has verified it.
        let timeout = std::time::Duration::from_secs(task.timeout_secs.max(1));
        let (shared_in, fetched) = match try_read_bundle(&self.store, task.input_content_id).await?
        {
            Some(bundle) => (bundle, false),
            None => match (origin, &self.objects) {
                (Some(origin), Some(objects)) => (
                    objects
                        .fetch_artifact(&origin.conn, task.input_content_id, timeout)
                        .await
                        .map_err(|e| {
                            TrainError::other(format!(
                                "fetch input {} from {}: {e}",
                                task.input_content_id, origin.peer
                            ))
                        })?,
                    true,
                ),
                _ => {
                    return Err(TrainError::other(format!(
                        "shared store has no artifact for {}",
                        task.input_content_id
                    )));
                }
            },
        };
        let input = bundle::unbundle(
            &*stage,
            &shared_in.manifest,
            &shared_in.pack,
            &stage_dir,
            Some(task.input_content_id),
            BlobDir::Input,
        )
        .map_err(|e| TrainError::other(format!("unbundle input: {e}")))?;
        if fetched {
            let bytes = bincode::serialize(&shared_in)
                .map_err(|e| TrainError::other(format!("serialize fetched input: {e}")))?;
            self.store
                .put(ObjectKey::Artifact(task.input_content_id), bytes)
                .await
                .map_err(|e| TrainError::other(format!("store fetched input: {e}")))?;
        }
        drop(shared_in);

        // 6. Run the stage under the coordinator's deadline.
        let cache = Arc::new(CacheHandle::job_local(stage_dir.join(".cache")));
        let ctx = StageContext::for_peer(
            stage_dir.clone(),
            stage_dir.clone(),
            cache,
            task.invocation_key,
        );
        let started = std::time::Instant::now();
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

        // 8. Sign + return the result (output on the store, not sealed).
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

async fn try_read_bundle(
    store: &ObjectStore,
    content_id: ArtifactContentId,
) -> Result<Option<StoredArtifact>, TrainError> {
    let Some(bytes) = store
        .get(ObjectKey::Artifact(content_id))
        .await
        .map_err(|e| TrainError::other(format!("read shared bundle: {e}")))?
    else {
        return Ok(None);
    };
    bincode::deserialize(&bytes)
        .map(Some)
        .map_err(|e| TrainError::other(format!("decode shared bundle: {e}")))
}

/// What [`dispatch_stage`] produced.
#[derive(Debug)]
#[non_exhaustive]
pub struct MeshDispatch {
    /// The worker's signed result (verified).
    pub result: TaskResult,
    /// The output, restored under `out_dir` and verified against the result's
    /// content id.
    pub output: ErasedArtifact,
    /// The input's content id, as published and granted to the worker.
    pub input_content_id: ArtifactContentId,
    /// Bundling + publishing the input locally.
    pub publish: std::time::Duration,
    /// Dispatch to signed result: the worker's input fetch, the stage run and
    /// publishing its output.
    pub remote: std::time::Duration,
    /// Getting the output back (zero-copy when it was already in the store).
    pub output_fetch: std::time::Duration,
    /// Whether the output came over the mesh (false: it was in the store).
    pub output_fetched: bool,
}

/// Dispatch one real stage to a connected mesh peer, moving its data over the
/// mesh: publish `input` into the local store (`objects.store()`), grant it to
/// the worker, dispatch a signed task, verify the worker's signed result
/// against `worker_key`, then fetch the output from the worker by content id
/// (unless the store already holds it) and restore it under `out_dir`,
/// verified. The worker must run a [`SharedCacheRunner`] that pins this node's
/// key and has the mesh object plane attached.
#[allow(clippy::too_many_arguments)]
pub async fn dispatch_stage(
    node: &MeshNode,
    objects: &MeshObjects,
    registry: &Registry,
    keypair: &KeyPair,
    conn: &quinn::Connection,
    worker_key: &VerifyingKey,
    stage_name: &str,
    input: ErasedArtifact,
    src_root: &std::path::Path,
    args: serde_json::Value,
    timeout: std::time::Duration,
    out_dir: &std::path::Path,
) -> Result<MeshDispatch, TrainError> {
    use crate::framework::{CacheHandle, ContentHash};
    use crate::p2p::dispatch::{DefaultDispatchPolicy, DispatchPolicy};
    use crate::p2p::trust::{DataClass, DispatchMatrix};

    // The dispatcher applies the same policy the worker will: a stage that
    // must not leave this host never has its input published for a peer.
    let policy = DefaultDispatchPolicy::new(DispatchMatrix::default());
    if !policy.is_dispatchable(stage_name) {
        return Err(TrainError::other(format!(
            "stage '{stage_name}' is not dispatchable"
        )));
    }
    let data_class = policy.classify_stage(stage_name, &args);
    if data_class == DataClass::Restricted {
        return Err(TrainError::other(format!(
            "stage '{stage_name}' handles restricted data; it does not leave this host"
        )));
    }
    let ctor = registry
        .find_erased_stage(stage_name)
        .ok_or_else(|| TrainError::other(format!("unknown stage '{stage_name}'")))?;
    let stage = ctor();

    let started = std::time::Instant::now();
    let input_content_id =
        SharedCacheRunner::publish_input(objects.store(), registry, stage_name, input, src_root)
            .await?;
    let publish = started.elapsed();
    let worker_id = PeerId::from_pubkey(worker_key);
    objects.grant(&worker_id, input_content_id);

    let nonce = format!("{:016x}", rand::random::<u64>());
    let task_id = format!("mesh-{nonce}");
    let args_hash = ContentHash::of_bytes(
        &serde_json::to_vec(&args).map_err(|e| TrainError::other(format!("args: {e}")))?,
    );
    let mut task = TaskManifest {
        protocol_version: crate::p2p::task::TASK_PROTOCOL_VERSION,
        task_id: task_id.clone(),
        coordinator_id: PeerId::from_pubkey(&keypair.verifying),
        stage_name: stage_name.to_string(),
        stage_schema: stage.schema(),
        input_content_id,
        invocation_key: CacheHandle::key_for(
            stage_name,
            stage.schema(),
            input_content_id.digest(),
            &args,
            nonce.as_bytes(),
        ),
        args_hash,
        expected_content_id: None,
        args,
        resources: crate::p2p::task::ResourceRequest::default(),
        data_class,
        timeout_secs: timeout.as_secs().max(1),
        deadline: crate::framework::execution::ExecutionDeadline::from_now(None, timeout),
        encrypted_input: None,
        signature: ed25519_dalek::Signature::from_bytes(&[0u8; 64]),
    };
    task.signature = keypair.sign(&task.sign_payload());

    let started = std::time::Instant::now();
    // The worker bounds its own fetch + run by `timeout`; allow the result
    // frame a little slack on top.
    let slack = std::time::Duration::from_secs(30);
    let result = tokio::time::timeout(timeout + slack, node.dispatch(conn, task))
        .await
        .map_err(|_| {
            TrainError::other(format!(
                "no result from {worker_id} within {}s",
                (timeout + slack).as_secs()
            ))
        })??;
    let remote = started.elapsed();
    if result.peer_id != worker_id {
        return Err(TrainError::other(format!(
            "result came from {}, not the worker {worker_id} it was dispatched to",
            result.peer_id
        )));
    }
    if !verify(worker_key, &result.sign_payload(), &result.signature) {
        return Err(TrainError::other(
            "result signature does not verify against the worker's key",
        ));
    }
    if result.task_id != task_id {
        return Err(TrainError::other("result is for a different task"));
    }

    let started = std::time::Instant::now();
    let (stored, output_fetched) = match try_read_bundle(objects.store(), result.content_id).await?
    {
        Some(stored) => (stored, false),
        None => (
            objects
                .fetch_artifact(conn, result.content_id, timeout)
                .await
                .map_err(|e| TrainError::other(format!("fetch output from {worker_id}: {e}")))?,
            true,
        ),
    };
    let output = bundle::unbundle(
        &*stage,
        &stored.manifest,
        &stored.pack,
        out_dir,
        Some(result.content_id),
        BlobDir::Output,
    )
    .map_err(|e| TrainError::other(format!("unbundle output: {e}")))?;
    if output_fetched {
        let bytes = bincode::serialize(&stored)
            .map_err(|e| TrainError::other(format!("serialize fetched output: {e}")))?;
        objects
            .store()
            .put(ObjectKey::Artifact(result.content_id), bytes)
            .await
            .map_err(|e| TrainError::other(format!("store fetched output: {e}")))?;
    }
    let output_fetch = started.elapsed();
    Ok(MeshDispatch {
        result,
        output,
        input_content_id,
        publish,
        remote,
        output_fetch,
        output_fetched,
    })
}

#[async_trait]
impl MeshTaskRunner for SharedCacheRunner {
    async fn run(&self, task: TaskManifest) -> Result<TaskResult, TrainError> {
        self.execute(task, None).await
    }

    async fn run_from(
        &self,
        task: TaskManifest,
        origin: &TaskOrigin,
    ) -> Result<TaskResult, TrainError> {
        self.execute(task, Some(origin)).await
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
            input_content_id: ArtifactContentId::from_digest(ContentHash::of_bytes(b"i")),
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"x")),
            args_hash: ContentHash::of_bytes(b"{}"),
            expected_content_id: Some(ArtifactContentId::from_digest(ContentHash::of_bytes(b"o"))),
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

    /// The multi-node loopback gate: four nodes with separate stores and no
    /// shared filesystem. D seeds only A, learns B and C by gossip, then
    /// dispatches a multi-MB `p2p-echo` to C. C's store starts empty, so C
    /// fetches the input from D over the mesh; D fetches the output back.
    #[tokio::test]
    async fn four_nodes_gossip_then_move_real_stage_data() {
        use crate::p2p::node::{GossipConfig, MeshNode, NodeCapabilities};
        use crate::p2p::objects::MeshObjects;
        use crate::p2p::registry::PeerRegistry;

        fn tmp() -> std::path::PathBuf {
            let d = tempfile::tempdir().unwrap();
            let p = d.path().to_path_buf();
            std::mem::forget(d);
            p
        }
        let mut registry = Registry::new();
        crate::p2p::smoke::register(&mut registry);
        let registry = Arc::new(registry);
        let d_kp = Arc::new(KeyPair::generate());

        let cfg = GossipConfig::default()
            .with_interval(std::time::Duration::from_millis(300))
            .with_backoff(
                std::time::Duration::from_millis(200),
                std::time::Duration::from_secs(2),
            );

        // A, B, C: workers pinning D as their scheduler, each with its own store.
        let mut workers = Vec::new();
        for _ in 0..3 {
            let kp = Arc::new(KeyPair::generate());
            let store = ObjectStore::filesystem(tmp());
            let objects = Arc::new(MeshObjects::new(store.clone(), tmp()));
            let policy: Arc<dyn DispatchPolicy> =
                Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default()));
            let runner: Arc<dyn MeshTaskRunner> = Arc::new(
                SharedCacheRunner::new(
                    registry.clone(),
                    store.clone(),
                    tmp(),
                    kp.clone(),
                    policy,
                    d_kp.verifying,
                )
                .with_mesh_objects(objects.clone()),
            );
            let node = MeshNode::bind(
                "127.0.0.1:0".parse().unwrap(),
                kp.clone(),
                PeerRegistry::load(&tmp().join("peers.json")).unwrap(),
                NodeCapabilities {
                    worker: true,
                    scheduler: false,
                },
                Some(runner),
                false,
            )
            .await
            .unwrap();
            node.set_objects(objects.clone());
            node.start_gossip(cfg.clone());
            workers.push((node, kp, objects));
        }
        let (a, a_kp, _) = &workers[0];
        let a_addr = a.local_addr().unwrap();
        let a_key = a_kp.verifying.to_bytes();
        for (n, _, _) in &workers[1..] {
            n.add_seed(a_addr, a_key).await.unwrap();
            n.connect_to(a_addr, a_key).await.unwrap();
        }

        // D: scheduler only, its own store, seeded with A alone.
        let d_objects = Arc::new(MeshObjects::new(ObjectStore::filesystem(tmp()), tmp()));
        let d = MeshNode::bind(
            "127.0.0.1:0".parse().unwrap(),
            d_kp.clone(),
            PeerRegistry::load(&tmp().join("peers.json")).unwrap(),
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
            false,
        )
        .await
        .unwrap();
        d.set_objects(d_objects.clone());
        d.start_gossip(cfg);
        d.add_seed(a_addr, a_key).await.unwrap();
        d.connect_to(a_addr, a_key).await.unwrap();

        let (c, c_kp, c_objects) = &workers[2];
        let b_id = workers[1].0.node_id().clone();
        let c_id = c.node_id().clone();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let up = d.connected_peers().await;
            if up.contains(&b_id) && up.contains(&c_id) {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "D never connected to B and C: {up:?}"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // ~4 MiB of text: the input exists only in D's store.
        let text: String = (0..(4 * 1024 * 1024 / 32))
            .map(|i| format!("line {i:>10} of mesh data ...\n"))
            .collect();
        let src = tmp();
        let in_path = src.join("p2p-echo-in.txt");
        std::fs::write(&in_path, &text).unwrap();
        let input = SmokeText {
            content_hash: ContentHash::hash_file(&in_path).unwrap(),
            path: in_path,
        };
        let conn = d.connection_to(&c_id).await.unwrap();
        let out_dir = tmp();
        let done = tokio::time::timeout(
            crate::TEST_HANG_GUARD,
            dispatch_stage(
                &d,
                &d_objects,
                &registry,
                &d_kp,
                &conn,
                &c_kp.verifying,
                SMOKE_STAGE,
                ErasedArtifact::from_typed(&input).unwrap(),
                &src,
                serde_json::json!({}),
                std::time::Duration::from_secs(60),
                &out_dir,
            ),
        )
        .await
        .expect("no hang")
        .expect("C runs the stage on data fetched from D");

        assert!(done.output_fetched, "the output came back over the mesh");
        assert_eq!(done.result.peer_id, c_id);
        let out: SmokeText = done.output.into_typed().unwrap();
        let body = std::fs::read_to_string(&out.path).unwrap();
        assert_eq!(body.len(), text.len());
        assert!(body == text.to_uppercase(), "output == uppercase(input)");
        // C now holds the input it fetched (verified before it was stored).
        assert!(
            c_objects
                .store()
                .contains(ObjectKey::Artifact(done.input_content_id))
                .await
                .unwrap()
        );
        // B was never involved and never received the input.
        assert!(
            !workers[1]
                .2
                .store()
                .contains(ObjectKey::Artifact(done.input_content_id))
                .await
                .unwrap()
        );
    }

    /// A worker never fetches for a task whose signature it rejects, and a
    /// dispatcher that did not grant its input gets a clean failure.
    #[tokio::test]
    async fn an_ungranted_input_is_not_served_to_the_worker() {
        use crate::p2p::node::{MeshNode, NodeCapabilities};
        use crate::p2p::objects::MeshObjects;
        use crate::p2p::registry::PeerRegistry;

        fn tmp() -> std::path::PathBuf {
            let d = tempfile::tempdir().unwrap();
            let p = d.path().to_path_buf();
            std::mem::forget(d);
            p
        }
        let mut registry = Registry::new();
        crate::p2p::smoke::register(&mut registry);
        let registry = Arc::new(registry);
        let d_kp = Arc::new(KeyPair::generate());
        let w_kp = Arc::new(KeyPair::generate());
        let w_store = ObjectStore::filesystem(tmp());
        let w_objects = Arc::new(MeshObjects::new(w_store.clone(), tmp()));
        let policy: Arc<dyn DispatchPolicy> =
            Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default()));
        let runner: Arc<dyn MeshTaskRunner> = Arc::new(
            SharedCacheRunner::new(
                registry.clone(),
                w_store,
                tmp(),
                w_kp.clone(),
                policy,
                d_kp.verifying,
            )
            .with_mesh_objects(w_objects.clone()),
        );
        let w = MeshNode::bind(
            "127.0.0.1:0".parse().unwrap(),
            w_kp.clone(),
            PeerRegistry::load(&tmp().join("peers.json")).unwrap(),
            NodeCapabilities {
                worker: true,
                scheduler: false,
            },
            Some(runner),
            false,
        )
        .await
        .unwrap();
        w.set_objects(w_objects);
        let d_objects = Arc::new(MeshObjects::new(ObjectStore::filesystem(tmp()), tmp()));
        let d = MeshNode::bind(
            "127.0.0.1:0".parse().unwrap(),
            d_kp.clone(),
            PeerRegistry::load(&tmp().join("peers.json")).unwrap(),
            NodeCapabilities {
                worker: false,
                scheduler: true,
            },
            None,
            false,
        )
        .await
        .unwrap();
        d.set_objects(d_objects.clone());
        let conn = d
            .connect_to(w.local_addr().unwrap(), w_kp.verifying.to_bytes())
            .await
            .unwrap();

        // Publish but do NOT grant, then dispatch by hand.
        let src = tmp();
        let in_path = src.join("p2p-echo-in.txt");
        std::fs::write(&in_path, "secret-ish").unwrap();
        let input = SmokeText {
            content_hash: ContentHash::hash_file(&in_path).unwrap(),
            path: in_path,
        };
        let input_content_id = SharedCacheRunner::publish_input(
            d_objects.store(),
            &registry,
            SMOKE_STAGE,
            ErasedArtifact::from_typed(&input).unwrap(),
            &src,
        )
        .await
        .unwrap();
        let mut task = TaskManifest {
            protocol_version: crate::p2p::task::TASK_PROTOCOL_VERSION,
            task_id: "ungranted-1".into(),
            coordinator_id: PeerId::from_pubkey(&d_kp.verifying),
            stage_name: SMOKE_STAGE.into(),
            stage_schema: 1,
            input_content_id,
            invocation_key: InvocationKey::from_digest(ContentHash::of_bytes(b"u")),
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
            signature: d_kp.sign(b"placeholder"),
        };
        task.signature = d_kp.sign(&task.sign_payload());
        let err = tokio::time::timeout(crate::TEST_HANG_GUARD, d.dispatch(&conn, task))
            .await
            .expect("no hang")
            .unwrap_err();
        assert!(format!("{err}").contains("not granted"), "{err}");
    }
}
