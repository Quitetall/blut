//! Plan executor.
//!
//! Walks the topo-sorted plan one stage at a time. For each:
//!
//!   1. Build the input artifact (initial map for graph-input
//!      nodes; otherwise the predecessor's cached output).
//!   2. Compute the cache key from `(stage_name, schema,
//!      input_hash, args)`.
//!   3. Cache hit → emit `StageSkipped`, advance with the cached
//!      output.
//!   4. Cache miss → emit `StageBegin`, build `StageContext`, call
//!      `StageDyn::run_erased`, persist the output to cache, write
//!      sidecar metadata next to the artifact's primary path,
//!      emit `StageEnd`.
//!
//! v2 commit 3: sequential only. Branches/parallel land commit 6
//! when the typed `fork`/`merge` builder API + per-Resource
//! semaphores arrive together.
//!
//! Cancellation: the executor honours a `CancellationToken`
//! threaded through every `StageContext`. Cancelling between
//! stages aborts cleanly with `PlanError::Cancelled`. Cancelling
//! mid-stage is the stage's responsibility (it must observe the
//! token; `python_backend` already does).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::framework::artifact::{ArtifactMetadata, ContentHash};
use crate::framework::cache::CacheHandle;
use crate::framework::error::{PlanError, StageError};
use crate::framework::plan::{CompiledPlan, NodeId};
use crate::framework::stage::{ErasedArtifact, StageContext};
use crate::framework::status::{spawn_status_writer, StageEvent};

/// Caller-supplied execution context. Threaded through every
/// `StageContext`. Lives for the duration of one `execute` call.
pub struct ExecCtx {
    pub job_dir: PathBuf,
    pub cache: Arc<CacheHandle>,
    pub status_tx: broadcast::Sender<StageEvent>,
    pub cancel: CancellationToken,
    /// Per-resource semaphores. Stages acquire all permits in
    /// their `RESOURCES` slice before `run` is called. Default
    /// limits: Gpu=1 (single-card), Cpu=num_cpus, Network=4,
    /// Disk=2. Override via ExecCtx::with_resource_limit.
    pub resources: std::collections::HashMap<crate::framework::resource::Resource, Arc<tokio::sync::Semaphore>>,
}

impl ExecCtx {
    /// Construct an `ExecCtx` rooted at `job_dir`. The caller is
    /// responsible for creating `job_dir` if it doesn't exist.
    pub fn new(job_dir: PathBuf) -> Self {
        let cache = Arc::new(CacheHandle::job_local(job_dir.join("_cache")));
        let status_tx = crate::framework::status::make_broadcast();
        let cancel = CancellationToken::new();
        let mut resources = std::collections::HashMap::new();
        use crate::framework::resource::Resource;
        let cpu_n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        resources.insert(Resource::Gpu, Arc::new(tokio::sync::Semaphore::new(1)));
        resources.insert(Resource::Cpu, Arc::new(tokio::sync::Semaphore::new(cpu_n)));
        resources.insert(Resource::Network, Arc::new(tokio::sync::Semaphore::new(4)));
        resources.insert(Resource::Disk, Arc::new(tokio::sync::Semaphore::new(2)));
        Self {
            job_dir,
            cache,
            status_tx,
            cancel,
            resources,
        }
    }

    pub fn with_resource_limit(
        mut self,
        resource: crate::framework::resource::Resource,
        permits: usize,
    ) -> Self {
        self.resources
            .insert(resource, Arc::new(tokio::sync::Semaphore::new(permits)));
        self
    }
}

/// What the executor returns on success. Carries the final node's
/// output (when the plan has one) plus diagnostics about how the
/// run went.
#[derive(Debug)]
pub struct PlanResult {
    pub final_output: Option<ErasedArtifact>,
    pub n_stages: usize,
    pub n_cache_hits: usize,
    pub n_cache_misses: usize,
    pub elapsed: std::time::Duration,
}

/// Sequential executor. Walks the plan in topo order; every stage
/// runs in the calling task (no `tokio::spawn`) since linear
/// chains have no concurrency to exploit. Commit 6 introduces
/// `ParallelExecutor` for plans with branches.
pub struct SequentialExecutor;

impl SequentialExecutor {
    /// Execute the plan to completion.
    pub async fn execute(plan: CompiledPlan, ctx: ExecCtx) -> Result<PlanResult, PlanError> {
        // R21 precondition: ExecCtx invariants the executor relies on.
        debug_assert!(!ctx.resources.is_empty(), "ExecCtx must declare resource semaphores");
        debug_assert!(
            ctx.job_dir.is_absolute() || ctx.job_dir.starts_with("/tmp") || ctx.job_dir.is_relative(),
            "ExecCtx.job_dir must be a valid path"
        );
        let started = Instant::now();
        let order = plan.topo_order()?;
        let view = plan.exec_view();
        // R21: topo order must enumerate every node exactly once.
        debug_assert_eq!(
            order.len(),
            view.nodes.len(),
            "topo_order must cover all nodes"
        );

        // Spawn the persistent status writer. Its task ends when
        // ctx.status_tx is dropped (at end of execute).
        std::fs::create_dir_all(&ctx.job_dir)?;
        let writer_handle = spawn_status_writer(&ctx.status_tx, &ctx.job_dir)?;

        // Persist plan + recipe args at job-dir root for audit.
        let args_path = ctx.job_dir.join("args.json");
        let args_body = serde_json::to_vec_pretty(view.recipe_args)
            .map_err(|e| PlanError::Other(format!("serialize args: {e}")))?;
        std::fs::write(&args_path, args_body)?;

        // Outputs by node id. Populated as the executor walks the
        // topo order; each subsequent node looks up its
        // predecessor here.
        let mut outputs: HashMap<NodeId, ErasedArtifact> = HashMap::new();
        // Logical output hashes, parallel to `outputs`. For
        // deterministic stages this is the real content hash. For
        // nondeterministic stages (training) it's a synthesized
        // fingerprint stable across stochastic re-runs. Downstream
        // cache keys read this, NOT the real content hash, so a
        // retrained upstream doesn't force downstream re-execution
        // when the upstream identity (name+schema+args+input) is
        // unchanged.
        let mut logical_outputs: HashMap<NodeId, ContentHash> = HashMap::new();
        // Pre-seed initial inputs (graph inputs).
        for (id, art) in view.initial {
            let lh = content_hash_from_erased(art);
            outputs.insert(*id, art.clone());
            logical_outputs.insert(*id, lh);
        }

        let mut n_hits = 0usize;
        let mut n_misses = 0usize;

        for (idx, node_id) in order.iter().enumerate() {
            if ctx.cancel.is_cancelled() {
                let _ = ctx.status_tx.send(StageEvent::StageFailed {
                    node_idx: idx as u32,
                    stage_name: "<cancelled>".into(),
                    error: "plan cancelled before stage".into(),
                });
                return Err(PlanError::Cancelled);
            }

            let node = &view.nodes[*node_id as usize];
            let stage_name = node.stage.name();

            // Find input. Linear plans have at most one
            // predecessor; multi-input merge nodes (commit 6)
            // gather all predecessors into a tuple.
            let preds: Vec<NodeId> = view
                .edges
                .iter()
                .filter(|e| e.to == *node_id)
                .map(|e| e.from)
                .collect();
            let input: ErasedArtifact = match preds.as_slice() {
                [] => outputs.get(node_id).cloned().ok_or_else(|| {
                    PlanError::Other(format!(
                        "node {} has no predecessors and no initial input",
                        node_id
                    ))
                })?,
                [single] => outputs.get(single).cloned().ok_or_else(|| {
                    PlanError::Other(format!(
                        "node {} predecessor {} produced no output",
                        node_id, single
                    ))
                })?,
                multi => {
                    // Gather predecessors' outputs as a tuple
                    // ErasedArtifact. Tuple kind is "tuple<N>" where
                    // N is the arity; payload is the concatenation
                    // of child bincode bytes, preserving fork order
                    // (the order the predecessors appear in the
                    // edges Vec, which corresponds to the order the
                    // recipe author called fork/fork3).
                    //
                    // Bincode encodes a tuple `(A, B)` as
                    // `bincode(A) ++ bincode(B)` with no separator,
                    // so concatenation is wire-equivalent to
                    // `bincode::serialize(&(a, b))`. The
                    // tuple-consuming stage's blanket `into_typed`
                    // call deserializes the concatenated bytes as
                    // `(A, B)` correctly.
                    let mut payload: Vec<u8> = Vec::with_capacity(
                        multi.iter().filter_map(|p| outputs.get(p).map(|a| a.payload.len())).sum(),
                    );
                    for &pid in multi {
                        let art = outputs.get(&pid).ok_or_else(|| {
                            PlanError::Other(format!(
                                "node {} predecessor {} produced no output",
                                node_id, pid
                            ))
                        })?;
                        payload.extend_from_slice(&art.payload);
                    }
                    let tuple_kind = format!("tuple<{}>", multi.len());
                    ErasedArtifact {
                        kind: tuple_kind,
                        schema: 1,
                        payload,
                    }
                }
            };

            // Cache key derivation uses LOGICAL hashes of
            // predecessors, not real content hashes. For all-
            // deterministic chains these are identical; for chains
            // containing nondet stages, logical hashes are stable
            // across stochastic re-runs so downstream's cache key
            // doesn't drift just because an upstream retrain
            // produced different ckpt bytes.
            let input_hash = match preds.as_slice() {
                [] => *logical_outputs.get(node_id).ok_or_else(|| {
                    PlanError::Other(format!(
                        "node {} has no logical input hash",
                        node_id
                    ))
                })?,
                [single] => *logical_outputs.get(single).ok_or_else(|| {
                    PlanError::Other(format!(
                        "node {} predecessor {} missing logical hash",
                        node_id, single
                    ))
                })?,
                multi => {
                    // Tuple input: hash the concatenation of child
                    // logical hashes with arity domain separation.
                    use sha2::{Digest, Sha256};
                    let mut h = Sha256::new();
                    h.update(b"tuple");
                    h.update([multi.len() as u8]);
                    for &pid in multi {
                        let lh = logical_outputs.get(&pid).ok_or_else(|| {
                            PlanError::Other(format!(
                                "node {} predecessor {} missing logical hash",
                                node_id, pid
                            ))
                        })?;
                        h.update(lh.0);
                    }
                    let arr: [u8; 32] = h.finalize().into();
                    ContentHash(arr)
                }
            };

            // Use precomputed canonical-args bytes from the plan
            // compile pass (opt-5) instead of re-canonicalizing the
            // args Value on every stage invocation.
            let key = CacheHandle::key_for_canon_bytes(
                stage_name,
                node.stage.schema(),
                input_hash,
                &node.canon_args,
            );

            // Lookup.
            if let Some(hit) = ctx.cache.lookup(key) {
                let _ = ctx.status_tx.send(StageEvent::StageSkipped {
                    node_idx: idx as u32,
                    stage_name: stage_name.to_string(),
                    cache_key: key,
                });
                let logical = compute_logical_output_hash(
                    node.stage.as_ref(),
                    &hit.artifact,
                    node.stage.deterministic(),
                    stage_name,
                    node.stage.schema(),
                    input_hash,
                    &node.canon_args,
                );
                outputs.insert(*node_id, hit.artifact);
                logical_outputs.insert(*node_id, logical);
                n_hits += 1;
                continue;
            }

            // Miss → run.
            let _ = ctx.status_tx.send(StageEvent::StageBegin {
                node_idx: idx as u32,
                stage_name: stage_name.to_string(),
                input_hash,
            });

            // FW-2: atomic stage outputs.
            //
            // The stage runs against a private `.tmp-<key>` working
            // directory, NOT the final `stage_dir`. On Ok we promote
            // the tmp dir to the final name with an atomic rename and
            // ONLY THEN insert the cache entry; on Err/cancel we
            // `remove_dir_all` the tmp so no half-written output ever
            // survives under the final name. Because the cache entry
            // is the executor's sole resume/completion oracle (see
            // the `ctx.cache.lookup` hit path above), a partial run
            // can never look complete: no promote ⇒ no rename ⇒ no
            // cache.insert ⇒ the stage re-executes on the next run.
            let stages_root = ctx.job_dir.join("stages");
            let final_stage_dir = stages_root.join(format!("{idx}-{stage_name}"));
            // Key-scoped tmp name so a re-run with different args
            // (different key) never collides with an unrelated stale
            // tmp, and so two positions of the same stage are
            // distinct. `.`-prefixed to keep it out of any tooling
            // that lists completed stage dirs.
            let tmp_stage_dir =
                stages_root.join(format!(".tmp-{idx}-{stage_name}-{}", key.to_hex()));
            // Clear any leftover tmp from a prior crashed run, then
            // start clean. The stage always sees an empty dir — no
            // orphan file from a previous attempt can contaminate its
            // output or its `hash_dir` content address.
            let _ = std::fs::remove_dir_all(&tmp_stage_dir);
            std::fs::create_dir_all(&tmp_stage_dir)?;

            let stage_ctx = StageContext {
                job_dir: ctx.job_dir.clone(),
                stage_dir: tmp_stage_dir.clone(),
                node_idx: idx as u32,
                status_tx: ctx.status_tx.clone(),
                cancel: ctx.cancel.clone(),
                cache: ctx.cache.clone(),
            };

            // Acquire resource permits in declared order. We keep
            // the OwnedSemaphorePermits in a Vec dropped at end of
            // the loop iteration so the next stage can acquire.
            let mut permits = Vec::new();
            for resource in node.stage.resources() {
                if let Some(sem) = ctx.resources.get(resource) {
                    let _ = ctx.status_tx.send(StageEvent::StageBlocked {
                        node_idx: idx as u32,
                        stage_name: stage_name.to_string(),
                        resource: *resource,
                    });
                    match sem.clone().acquire_owned().await {
                        Ok(p) => permits.push(p),
                        Err(_) => {
                            // Semaphore closed (shouldn't happen);
                            // fail the stage cleanly rather than
                            // panicking. Drop the just-created tmp dir
                            // so no empty/partial output lingers.
                            let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                            return Err(PlanError::Other(format!(
                                "resource '{}' semaphore closed",
                                resource
                            )));
                        }
                    }
                }
            }

            let stage_started = Instant::now();
            let run_result = node
                .stage
                .run_erased(&stage_ctx, input, node.args.clone())
                .await;
            // Permits drop here, releasing the resource for the
            // next stage.
            drop(permits);
            // Drop the stage_ctx (and its status_tx clone)
            // BEFORE awaiting the writer handle on the failure
            // path — otherwise the writer's broadcast channel
            // never closes (clones are still alive) and the await
            // hangs forever.
            drop(stage_ctx);
            let output = match run_result {
                Ok(o) => {
                    // R21 postcondition: stage's declared output_kind
                    // must match the erased kind tag. StageDyn's
                    // blanket impl guarantees this; the debug_assert
                    // catches a hand-rolled StageDyn impl that
                    // violates the contract.
                    debug_assert_eq!(
                        o.kind, node.stage.output_kind(),
                        "stage '{stage_name}' produced kind '{}' but declares output_kind '{}'",
                        o.kind, node.stage.output_kind()
                    );
                    // A cancel observed during the run (the stage
                    // returned Ok but the token fired mid-work) must
                    // NOT be promoted/cached — treat it as a clean
                    // cancellation with the tmp output discarded.
                    if ctx.cancel.is_cancelled() {
                        let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                        let _ = ctx.status_tx.send(StageEvent::StageFailed {
                            node_idx: idx as u32,
                            stage_name: stage_name.to_string(),
                            error: "plan cancelled during stage".into(),
                        });
                        drop(ctx.status_tx);
                        let _ = writer_handle.await;
                        return Err(PlanError::Cancelled);
                    }
                    o
                }
                Err(e) => {
                    // FW-2: tear down the tmp working dir so the
                    // half-written output cannot survive to the next
                    // run. The cache was never written (insert is
                    // post-promote, below), so resume re-runs cleanly.
                    let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                    let _ = ctx.status_tx.send(StageEvent::StageFailed {
                        node_idx: idx as u32,
                        stage_name: stage_name.to_string(),
                        error: format!("{e}"),
                    });
                    drop(ctx.status_tx);
                    let _ = writer_handle.await;
                    return Err(PlanError::StageFailed {
                        idx: idx as u32,
                        stage: stage_name.to_string(),
                        source: e,
                    });
                }
            };

            // FW-2 promote: atomically move the completed tmp dir to
            // the final `stage_dir`. Remove any stale final dir first
            // (a prior crashed run that left a partial under the final
            // name but never reached cache.insert), then rename. On a
            // filesystem where the rename can't be atomic (cross-
            // device — shouldn't happen, both are under job_dir) the
            // io error surfaces and the stage is treated as failed.
            let _ = std::fs::remove_dir_all(&final_stage_dir);
            if let Err(e) = std::fs::rename(&tmp_stage_dir, &final_stage_dir) {
                let _ = std::fs::remove_dir_all(&tmp_stage_dir);
                let _ = ctx.status_tx.send(StageEvent::StageFailed {
                    node_idx: idx as u32,
                    stage_name: stage_name.to_string(),
                    error: format!("promote stage output: {e}"),
                });
                drop(ctx.status_tx);
                let _ = writer_handle.await;
                return Err(PlanError::StageFailed {
                    idx: idx as u32,
                    stage: stage_name.to_string(),
                    source: StageError::Io {
                        path: final_stage_dir.clone(),
                        source: e,
                    },
                });
            }

            // The stage baked tmp-rooted absolute paths into its
            // output handle; re-point them at the promoted final dir
            // so downstream stages (and a cache hit on a later run)
            // read the files where they now live.
            let output =
                node.stage
                    .rebase_output_paths(output, &tmp_stage_dir, &final_stage_dir);
            let stage_dir = final_stage_dir;

            // Persist sidecar metadata next to the artifact's
            // primary path. We don't have access to the typed
            // artifact's `primary_path()` here (it's erased), but
            // the producing stage has already written the on-disk
            // payload to its `stage_dir`. Write metadata at
            // `<stage_dir>/output.metadata.json`.
            let output_hash = content_hash_from_erased(&output);
            let metadata = ArtifactMetadata::new(
                output.kind.clone(),
                output.schema,
                output_hash,
            )
            .with_stage(stage_name.to_string());
            let _ = metadata.write_to(&stage_dir.join("output.metadata.json"));

            // Insert into cache for resumes — ONLY after the atomic
            // promote above. This ordering is the load-bearing FW-2
            // invariant: the cache entry (the resume oracle) appears
            // strictly after the output is fully in place.
            if let Err(e) = ctx.cache.insert(key, &output) {
                tracing::warn!(
                    "executor: cache insert for stage '{}' failed: {e}; continuing",
                    stage_name
                );
            }

            let _ = ctx.status_tx.send(StageEvent::StageEnd {
                node_idx: idx as u32,
                stage_name: stage_name.to_string(),
                output_hash,
                elapsed: stage_started.elapsed(),
            });

            let logical = compute_logical_output_hash(
                node.stage.as_ref(),
                &output,
                node.stage.deterministic(),
                stage_name,
                node.stage.schema(),
                input_hash,
                &node.canon_args,
            );
            outputs.insert(*node_id, output);
            logical_outputs.insert(*node_id, logical);
            n_misses += 1;
        }

        // Pick the final output (the last node in topo order; for
        // linear plans this is unambiguous).
        let final_output = order.last().and_then(|id| outputs.remove(id));

        // Drop the broadcast sender so the writer task exits, then
        // await it to flush any tail events.
        drop(ctx.status_tx);
        let _ = writer_handle.await;

        Ok(PlanResult {
            final_output,
            n_stages: order.len(),
            n_cache_hits: n_hits,
            n_cache_misses: n_misses,
            elapsed: started.elapsed(),
        })
    }
}

/// Compute the `ContentHash` of an erased artifact directly from
/// its bincode payload bytes. Bincode is canonical for any given
/// type (no key reordering, no whitespace), so the byte form is
/// already a stable digest input — no canonicalization step needed
/// (this was the prior JSON path's cost).
///
/// Why not call the typed artifact's `content_hash`? Because at
/// the executor level we've already erased the type. Trade-off
/// unchanged from before: the erased hash addresses the handle
/// bytes, not the on-disk payload. Concrete artifact impls of
/// `content_hash` already account for this by hashing the on-disk
/// bytes inside their own implementation.
/// Logical output hash for a stage — the value downstream stages
/// fold into THEIR cache key. Deterministic stages report the real
/// content address so two runs producing byte-identical content
/// (even at different absolute paths / on different machines) yield
/// the same downstream key. Nondet stages report a synthesized
/// fingerprint = hash(stage_name ‖ schema ‖ input_hash ‖ args) which
/// is byte-stable across stochastic re-runs even when the actual
/// ckpt bytes differ.
///
/// FW-1: for a deterministic stage we seed from the artifact's
/// designed `content_hash()` (via `StageDyn::output_content_hash`)
/// — the SHA-256 of the on-disk *content* — NOT from the bincode
/// handle, which embeds an absolute `path` + machine-specific
/// metadata and so was breaking `--shared-cache` across machines /
/// relocation. The handle hash (`content_hash_from_erased`) remains
/// only as the fallback for an artifact whose erased payload can't
/// be decoded to the stage's typed `Output` (e.g. a `tuple<N>`
/// merge artifact, which the typed `content_hash` path can't see).
fn compute_logical_output_hash(
    stage: &dyn crate::framework::stage::StageDyn,
    output: &ErasedArtifact,
    deterministic: bool,
    stage_name: &str,
    schema: u32,
    input_hash: ContentHash,
    canon_args: &[u8],
) -> ContentHash {
    if deterministic {
        // Prefer the typed, path-independent content address; fall
        // back to the handle hash only when the payload doesn't
        // decode as this stage's Output (tuple/merge or contract
        // violation).
        return stage
            .output_content_hash(output)
            .unwrap_or_else(|| content_hash_from_erased(output));
    }
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"blut.nondet.v1");
    h.update([0u8]);
    h.update(stage_name.as_bytes());
    h.update([0u8]);
    h.update(schema.to_le_bytes());
    h.update(input_hash.0);
    h.update(canon_args);
    let arr: [u8; 32] = h.finalize().into();
    ContentHash(arr)
}

fn content_hash_from_erased(art: &ErasedArtifact) -> ContentHash {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(art.kind.as_bytes());
    h.update(art.schema.to_le_bytes());
    h.update(&art.payload);
    let arr: [u8; 32] = h.finalize().into();
    ContentHash(arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::LamuTrainerBackend;
    use crate::framework::artifact::Artifact;
    use crate::framework::compat::Compatible;
    use crate::framework::error::StageError;
    use crate::framework::plan::Plan;
    use crate::framework::resource::Resource;
    use crate::framework::stage::Stage;
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};
    use std::path::Path;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Toy artifacts.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Counter {
        n: u32,
    }
    impl Artifact for Counter {
        const KIND: &'static str = "test.counter";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(&self.n.to_le_bytes())
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct EmptyArgs;

    /// Counter that always emits Counter { n: 1 } and counts how
    /// many times its `run` was invoked across the test process.
    /// Tests in this module touch process-global counters
    /// (MAKE_RUN_COUNT, INC_RUN_COUNT) that the parallel test
    /// runner would race on. Serialize via a module-wide mutex.
    /// Production code path uses no static state.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    static MAKE_RUN_COUNT: AtomicU32 = AtomicU32::new(0);

    struct MakeOne;
    #[async_trait]
    impl Stage for MakeOne {
        const NAME: &'static str = "make_one";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = Counter;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: (),
            _args: &EmptyArgs,
        ) -> Result<Counter, StageError> {
            MAKE_RUN_COUNT.fetch_add(1, Ordering::SeqCst);
            Ok(Counter { n: 1 })
        }
    }

    static INC_RUN_COUNT: AtomicU32 = AtomicU32::new(0);

    struct Increment;
    #[async_trait]
    impl Stage for Increment {
        const NAME: &'static str = "increment";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = Counter;
        type Output = Counter;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            input: Counter,
            _args: &EmptyArgs,
        ) -> Result<Counter, StageError> {
            INC_RUN_COUNT.fetch_add(1, Ordering::SeqCst);
            Ok(Counter { n: input.n + 1 })
        }
    }

    // Test stages need Compatible<B> for the typed Plan API.
    // Tag against LamuTrainerBackend (arbitrary; tests pick one B).
    impl Compatible<LamuTrainerBackend> for MakeOne {}
    impl Compatible<LamuTrainerBackend> for Increment {}

    /// Stage that always returns a backend error. Tests the
    /// failure-propagation path.
    struct AlwaysFail;
    #[async_trait]
    impl Stage for AlwaysFail {
        const NAME: &'static str = "always_fail";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = Counter;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: (),
            _args: &EmptyArgs,
        ) -> Result<Counter, StageError> {
            Err(StageError::BadInput("forced failure".into()))
        }
    }
    impl Compatible<LamuTrainerBackend> for AlwaysFail {}

    fn fresh_ctx() -> (tempfile::TempDir, ExecCtx) {
        let td = tempfile::tempdir().unwrap();
        let ctx = ExecCtx::new(td.path().to_path_buf());
        (td, ctx)
    }

    #[tokio::test]
    async fn linear_plan_executes_in_order() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
        INC_RUN_COUNT.store(0, Ordering::SeqCst);
        let (_td, ctx) = fresh_ctx();
        let plan = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .then(Increment, EmptyArgs)
            .then(Increment, EmptyArgs)
            .finish().into_compiled();
        let result = SequentialExecutor::execute(plan, ctx).await.unwrap();
        assert_eq!(result.n_stages, 3);
        assert_eq!(result.n_cache_misses, 3);
        assert_eq!(result.n_cache_hits, 0);
        // Final output is Counter { n: 3 } (1 → 2 → 3).
        let out = result.final_output.unwrap();
        let counter: Counter = out.into_typed().unwrap();
        assert_eq!(counter.n, 3);
    }

    #[tokio::test]
    async fn cache_hit_skips_run_on_repeat_execution() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        MAKE_RUN_COUNT.store(0, Ordering::SeqCst);
        INC_RUN_COUNT.store(0, Ordering::SeqCst);
        let (_td, ctx) = fresh_ctx();
        let cache = ctx.cache.clone();

        // First run: every stage is a miss.
        let plan = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .then(Increment, EmptyArgs)
            .finish().into_compiled();
        let r1 = SequentialExecutor::execute(plan, ctx).await.unwrap();
        assert_eq!(r1.n_cache_misses, 2);
        assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 1);

        // Second run with a fresh ExecCtx but the SAME cache dir.
        let td_keepalive_for_second_run = tempfile::tempdir().unwrap();
        let job_dir2 = td_keepalive_for_second_run.path().to_path_buf();
        let ctx2 = ExecCtx::new(job_dir2);
        let ctx2 = ExecCtx { cache, ..ctx2 };
        let plan2 = Plan::<(), LamuTrainerBackend>::new("test", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .then(Increment, EmptyArgs)
            .finish().into_compiled();
        let r2 = SequentialExecutor::execute(plan2, ctx2).await.unwrap();
        assert_eq!(r2.n_cache_hits, 2, "second run should hit cache for both stages");
        assert_eq!(r2.n_cache_misses, 0);
        // Run counters didn't increment.
        assert_eq!(MAKE_RUN_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(INC_RUN_COUNT.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stage_failure_propagates_as_plan_error() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (_td, ctx) = fresh_ctx();
        let plan = Plan::<(), LamuTrainerBackend>::new("failing", serde_json::json!({}))
            .start(AlwaysFail, EmptyArgs)
            .finish().into_compiled();
        let r = SequentialExecutor::execute(plan, ctx).await;
        match r {
            Err(PlanError::StageFailed { idx, stage, source }) => {
                assert_eq!(idx, 0);
                assert_eq!(stage, "always_fail");
                assert!(matches!(source, StageError::BadInput(_)));
            }
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[tokio::test]
    async fn cancel_before_first_stage_returns_cancelled() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (_td, ctx) = fresh_ctx();
        ctx.cancel.cancel();
        let plan = Plan::<(), LamuTrainerBackend>::new("c", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .finish().into_compiled();
        let r = SequentialExecutor::execute(plan, ctx).await;
        assert!(matches!(r, Err(PlanError::Cancelled)));
    }

    #[tokio::test]
    async fn status_jsonl_persists_to_disk() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (td, ctx) = fresh_ctx();
        let plan = Plan::<(), LamuTrainerBackend>::new("p", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .then(Increment, EmptyArgs)
            .finish().into_compiled();
        let _ = SequentialExecutor::execute(plan, ctx).await.unwrap();
        let path = td.path().join("status.jsonl");
        assert!(path.exists());
        let body = std::fs::read_to_string(&path).unwrap();
        // 2 stage_begin + 2 stage_end (one per stage).
        let n_begin = body.matches("\"kind\":\"stage_begin\"").count();
        let n_end = body.matches("\"kind\":\"stage_end\"").count();
        assert_eq!(n_begin, 2);
        assert_eq!(n_end, 2);
    }

    #[tokio::test]
    async fn args_json_persisted_at_job_root() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (td, ctx) = fresh_ctx();
        let recipe_args = serde_json::json!({"output_name": "test", "since": "30d"});
        let plan = Plan::<(), LamuTrainerBackend>::new("p", recipe_args.clone())
            .start(MakeOne, EmptyArgs)
            .finish().into_compiled();
        let _ = SequentialExecutor::execute(plan, ctx).await.unwrap();
        let body = std::fs::read_to_string(td.path().join("args.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed, recipe_args);
    }

    #[tokio::test]
    async fn sidecar_metadata_written_per_stage() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (td, ctx) = fresh_ctx();
        let plan = Plan::<(), LamuTrainerBackend>::new("p", serde_json::json!({}))
            .start(MakeOne, EmptyArgs)
            .finish().into_compiled();
        let _ = SequentialExecutor::execute(plan, ctx).await.unwrap();
        let sidecar = td
            .path()
            .join("stages/0-make_one/output.metadata.json");
        assert!(sidecar.exists(), "expected sidecar at {}", sidecar.display());
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&sidecar).unwrap()).unwrap();
        assert_eq!(parsed["kind"], "test.counter");
        assert_eq!(parsed["produced_by_stage"], "make_one");
    }

    // ════════════════════════════════════════════════════════════
    // FW-1 — cross-machine-stable cache key.
    //
    // The reproducible bug: a deterministic stage's downstream cache
    // key was seeded from `content_hash_from_erased`, which hashes
    // the FULL bincode handle — including the artifact's embedded
    // absolute `path: PathBuf`. So byte-identical CONTENT produced at
    // two different absolute paths yielded DIFFERENT downstream keys,
    // and `--shared-cache` never hit across machines / relocation.
    //
    // `PathArt` reproduces the bug precisely: its `content_hash()`
    // (the designed stable address) hashes ONLY the content byte, but
    // its serialized handle ALSO carries a machine-specific `path`.
    // ════════════════════════════════════════════════════════════

    use crate::framework::artifact::ContentHash as CH;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct PathArt {
        /// Stable content (what `content_hash` addresses).
        content: u8,
        /// Machine-specific absolute path baked into the handle.
        /// This is the field that poisoned the pre-fix cache key.
        path: PathBuf,
    }
    impl Artifact for PathArt {
        const KIND: &'static str = "test.path_art";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> CH {
            // Content address IGNORES `path` — only the bytes matter.
            CH::of_bytes(&[self.content])
        }
        fn primary_path(&self) -> &Path {
            &self.path
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct PathArtArgs {
        /// Absolute path to bake into the produced handle. Varying
        /// this between two runs (with the same `content`) is what
        /// the FW-1 test exploits.
        abs_path: String,
        content: u8,
    }

    /// Deterministic graph-input stage that emits a `PathArt` with a
    /// caller-chosen absolute path + content byte. Deterministic =
    /// true so the executor seeds the downstream logical hash from
    /// the content address (FW-1 path).
    struct MakePathArt;
    #[async_trait]
    impl Stage for MakePathArt {
        const NAME: &'static str = "make_path_art";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = PathArt;
        type Args = PathArtArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: (),
            args: &PathArtArgs,
        ) -> Result<PathArt, StageError> {
            Ok(PathArt {
                content: args.content,
                path: PathBuf::from(&args.abs_path),
            })
        }
    }
    impl Compatible<LamuTrainerBackend> for MakePathArt {}

    /// Downstream deterministic consumer of `PathArt`. Its presence
    /// is what gives FW-1 something to observe: the executor folds
    /// the upstream's logical hash into THIS stage's cache key /
    /// `input_hash`, which we capture off the status channel.
    struct ConsumePathArt;
    #[async_trait]
    impl Stage for ConsumePathArt {
        const NAME: &'static str = "consume_path_art";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = PathArt;
        type Output = Counter;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            input: PathArt,
            _args: &EmptyArgs,
        ) -> Result<Counter, StageError> {
            Ok(Counter { n: input.content as u32 })
        }
    }
    impl Compatible<LamuTrainerBackend> for ConsumePathArt {}

    /// Run a 2-stage plan (MakePathArt → ConsumePathArt) and return
    /// the `input_hash` the executor computed for the DOWNSTREAM
    /// stage — i.e. the logical hash of the upstream output, which is
    /// exactly the value FW-1 fixes. Captured off the status channel
    /// via the `StageBegin` event for node_idx 1.
    async fn downstream_input_hash(abs_path: &str, content: u8) -> CH {
        let td = tempfile::tempdir().unwrap();
        let ctx = ExecCtx::new(td.path().to_path_buf());
        let mut rx = ctx.status_tx.subscribe();
        let plan = Plan::<(), LamuTrainerBackend>::new("fw1", serde_json::json!({}))
            .start(
                MakePathArt,
                PathArtArgs { abs_path: abs_path.to_string(), content },
            )
            .then(ConsumePathArt, EmptyArgs)
            .finish()
            .into_compiled();
        SequentialExecutor::execute(plan, ctx).await.unwrap();
        // Drain events; the downstream stage is node_idx 1.
        let mut found = None;
        while let Ok(evt) = rx.try_recv() {
            if let StageEvent::StageBegin { node_idx: 1, input_hash, .. } = evt {
                found = Some(input_hash);
            }
        }
        found.expect("downstream StageBegin must carry an input_hash")
    }

    #[tokio::test]
    async fn cache_key_stable_across_abs_path() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Same CONTENT byte (7), two DIFFERENT absolute paths — the
        // kind a relocated job dir / second machine produces.
        let h1 = downstream_input_hash("/machine-a/jobs/run1/stages/0-make_path_art/out", 7).await;
        let h2 = downstream_input_hash("/totally/different/machine-b/xyz/out", 7).await;
        assert_eq!(
            h1, h2,
            "FW-1: byte-identical content at different absolute paths must yield \
             the SAME downstream cache key (pre-fix these DIFFERED because the \
             bincode handle embedded the path)"
        );
        // Sanity: different CONTENT at the same path must still differ
        // (we fixed path-sensitivity without losing content-sensitivity).
        let h3 = downstream_input_hash("/machine-a/jobs/run1/stages/0-make_path_art/out", 8).await;
        assert_ne!(h1, h3, "different content must still change the key");
    }

    #[tokio::test]
    async fn content_hash_invoked_for_deterministic_stage() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // The downstream input_hash for a deterministic upstream MUST
        // equal the upstream artifact's own `content_hash()` — NOT
        // the handle-bincode hash (`content_hash_from_erased`). This
        // asserts the executor actually consults `content_hash()`.
        let abs = "/some/abs/path/out";
        let content = 42u8;
        let observed = downstream_input_hash(abs, content).await;

        let art = PathArt { content, path: PathBuf::from(abs) };
        let want = art.content_hash();
        assert_eq!(
            observed, want,
            "FW-1: deterministic stage's logical hash must be the artifact \
             content_hash(), not the bincode handle hash"
        );

        // And it must NOT be the handle-bincode hash — prove the two
        // are genuinely different so the assertion above has teeth.
        let erased = ErasedArtifact::from_typed(&art).unwrap();
        let handle_hash = content_hash_from_erased(&erased);
        assert_ne!(
            observed, handle_hash,
            "the path-bearing handle hash must differ from the content hash \
             here (otherwise the test couldn't distinguish the bug from the fix)"
        );
    }

    // ════════════════════════════════════════════════════════════
    // FW-2 — atomic stage outputs + crash/cancel cleanup.
    //
    // A stage writes a marker file into its `stage_dir`, then either
    // succeeds, fails, or observes a cancel. The executor must:
    //   - on Err / cancel: leave NO partial under the final stage_dir
    //     AND NO cache entry (so the stage re-runs);
    //   - on Ok: promote the output into the final stage_dir AND
    //     write the cache entry.
    // ════════════════════════════════════════════════════════════

    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct WriteThenArgs {
        /// "ok" | "err" | "cancel" — what the stage does AFTER it has
        /// written a partial marker file into its stage_dir.
        mode: String,
    }

    /// Writes `partial.txt` into its `stage_dir`, then behaves per
    /// `args.mode`. The marker is the thing FW-2 must clean up on the
    /// failure / cancel paths.
    struct WriteThen;
    #[async_trait]
    impl Stage for WriteThen {
        const NAME: &'static str = "write_then";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = Counter;
        type Args = WriteThenArgs;
        async fn run(
            &self,
            ctx: &StageContext,
            _input: (),
            args: &WriteThenArgs,
        ) -> Result<Counter, StageError> {
            // Write a partial output into the (tmp) working dir the
            // executor handed us.
            let marker = ctx.stage_dir.join("partial.txt");
            std::fs::write(&marker, b"half-written").map_err(|source| StageError::Io {
                path: marker,
                source,
            })?;
            match args.mode.as_str() {
                "ok" => Ok(Counter { n: 1 }),
                "cancel" => {
                    // Simulate a mid-stage cancel: fire the token then
                    // return Ok, exactly as a stage that noticed the
                    // cancel only after producing its (now-doomed)
                    // output would. The executor must discard it.
                    ctx.cancel.cancel();
                    Ok(Counter { n: 1 })
                }
                _ => Err(StageError::BadInput("forced mid-stage failure".into())),
            }
        }
    }
    impl Compatible<LamuTrainerBackend> for WriteThen {}

    /// The final (promoted) stage dir the executor uses for node 0 of
    /// a single-stage `write_then` plan.
    fn write_then_final_dir(job_dir: &Path) -> PathBuf {
        job_dir.join("stages").join("0-write_then")
    }

    /// Any leftover `.tmp-*` working dirs under `stages/`.
    fn leftover_tmp_dirs(job_dir: &Path) -> Vec<PathBuf> {
        let stages = job_dir.join("stages");
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&stages) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with(".tmp-") {
                    out.push(e.path());
                }
            }
        }
        out
    }

    /// Count cache entries (`<keyhex>/output.bin`) under the job's
    /// `_cache`. Robust completion-oracle probe: a stage that the
    /// executor considers "done" leaves exactly one entry. Avoids
    /// hand-recomputing the exact key (which folds the `()` input's
    /// erased-handle hash, not a bare empty-bytes hash).
    fn cache_entry_count(job_dir: &Path) -> usize {
        let cache_root = job_dir.join("_cache");
        let mut n = 0;
        if let Ok(rd) = std::fs::read_dir(&cache_root) {
            for e in rd.flatten() {
                if e.path().join("output.bin").exists() {
                    n += 1;
                }
            }
        }
        n
    }

    #[tokio::test]
    async fn partial_output_cleaned_on_stage_error() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let job_dir = td.path().to_path_buf();
        let ctx = ExecCtx::new(job_dir.clone());
        let plan = Plan::<(), LamuTrainerBackend>::new("fw2-err", serde_json::json!({}))
            .start(WriteThen, WriteThenArgs { mode: "err".into() })
            .finish()
            .into_compiled();
        let r = SequentialExecutor::execute(plan, ctx).await;
        assert!(matches!(r, Err(PlanError::StageFailed { .. })), "stage must fail");

        // No partial under the FINAL stage_dir.
        let final_dir = write_then_final_dir(&job_dir);
        assert!(
            !final_dir.exists(),
            "FW-2: final stage_dir must NOT exist after a stage error (found {})",
            final_dir.display()
        );
        // No orphan tmp working dir.
        let tmp = leftover_tmp_dirs(&job_dir);
        assert!(tmp.is_empty(), "FW-2: tmp working dir must be removed on error, found {tmp:?}");

        // The cache must hold NO entry for this stage, so a resume
        // re-runs it instead of falsely skipping.
        assert_eq!(
            cache_entry_count(&job_dir),
            0,
            "FW-2: a failed stage must NOT be cached (else resume falsely skips it)"
        );
    }

    #[tokio::test]
    async fn partial_output_cleaned_on_cancel() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let job_dir = td.path().to_path_buf();
        let ctx = ExecCtx::new(job_dir.clone());
        let plan = Plan::<(), LamuTrainerBackend>::new("fw2-cancel", serde_json::json!({}))
            .start(WriteThen, WriteThenArgs { mode: "cancel".into() })
            .finish()
            .into_compiled();
        let r = SequentialExecutor::execute(plan, ctx).await;
        assert!(
            matches!(r, Err(PlanError::Cancelled)),
            "a mid-stage cancel must surface as PlanError::Cancelled, got {r:?}"
        );

        let final_dir = write_then_final_dir(&job_dir);
        assert!(
            !final_dir.exists(),
            "FW-2: cancelled stage must leave NO partial under final stage_dir"
        );
        let tmp = leftover_tmp_dirs(&job_dir);
        assert!(tmp.is_empty(), "FW-2: tmp working dir must be removed on cancel, found {tmp:?}");

        assert_eq!(
            cache_entry_count(&job_dir),
            0,
            "FW-2: a cancelled stage must NOT be cached as success"
        );
    }

    #[tokio::test]
    async fn successful_stage_output_promoted_atomically() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let job_dir = td.path().to_path_buf();
        let ctx = ExecCtx::new(job_dir.clone());
        let cache = ctx.cache.clone();
        let plan = Plan::<(), LamuTrainerBackend>::new("fw2-ok", serde_json::json!({}))
            .start(WriteThen, WriteThenArgs { mode: "ok".into() })
            .finish()
            .into_compiled();
        let res = SequentialExecutor::execute(plan, ctx).await.unwrap();
        assert_eq!(res.n_cache_misses, 1);

        // Output present under the FINAL stage_dir (promoted), and the
        // tmp working dir is gone (renamed away).
        let final_dir = write_then_final_dir(&job_dir);
        assert!(final_dir.join("partial.txt").exists(), "promoted output file must be present");
        assert!(
            final_dir.join("output.metadata.json").exists(),
            "sidecar metadata must be written into the promoted final dir"
        );
        let tmp = leftover_tmp_dirs(&job_dir);
        assert!(tmp.is_empty(), "no tmp working dir should survive a successful promote, found {tmp:?}");

        // Cached after promote: exactly one entry, and a second run
        // against the SAME cache hits (re-uses, doesn't re-run).
        assert_eq!(
            cache_entry_count(&job_dir),
            1,
            "FW-2: a successful stage must be cached after the atomic promote"
        );
        let ctx2 = ExecCtx::new(td.path().join("job2"));
        let ctx2 = ExecCtx { cache, ..ctx2 };
        let plan2 = Plan::<(), LamuTrainerBackend>::new("fw2-ok", serde_json::json!({}))
            .start(WriteThen, WriteThenArgs { mode: "ok".into() })
            .finish()
            .into_compiled();
        let res2 = SequentialExecutor::execute(plan2, ctx2).await.unwrap();
        assert_eq!(res2.n_cache_hits, 1, "second run must hit the promoted cache entry");
        assert_eq!(res2.n_cache_misses, 0);
    }

    #[tokio::test]
    async fn stale_partial_does_not_contaminate_rerun() {
        // FW-2 corollary: a partial dir left by a prior crashed run
        // (under the final stage_dir name, but with NO cache entry)
        // must NOT contaminate a fresh run. The executor runs in a
        // clean tmp dir and overwrites the stale final on promote.
        let _g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let job_dir = td.path().to_path_buf();
        // Plant a stale partial with an orphan file the stage never
        // would have written.
        let final_dir = write_then_final_dir(&job_dir);
        std::fs::create_dir_all(&final_dir).unwrap();
        std::fs::write(final_dir.join("orphan.txt"), b"stale junk").unwrap();

        let ctx = ExecCtx::new(job_dir.clone());
        let plan = Plan::<(), LamuTrainerBackend>::new("fw2-stale", serde_json::json!({}))
            .start(WriteThen, WriteThenArgs { mode: "ok".into() })
            .finish()
            .into_compiled();
        SequentialExecutor::execute(plan, ctx).await.unwrap();

        assert!(final_dir.join("partial.txt").exists(), "fresh output present");
        assert!(
            !final_dir.join("orphan.txt").exists(),
            "FW-2: stale orphan from a prior crashed run must not survive into the promoted dir"
        );
    }
}
