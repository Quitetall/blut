// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Typed Plan builder.
//!
//! `Plan<Out>` is the typed DAG. The `Out` type parameter is a
//! `PhantomData` witness for the type at the "leading edge" of the
//! plan — what would feed the next stage if you called `then`.
//! Wrong wiring fails at `cargo build` because `Plan<O>::then<S>`
//! requires `S: Stage<Input = O>`. There is no runtime check at the
//! Plan-building API; the runtime check at `StageDyn::run_erased`
//! is defense-in-depth for dynamic recipes.
//!
//! v2 commit 3 ships linear chains (`start` + `then` + `finish`).
//! `fork` / `merge` typed APIs land in commit 6 alongside the
//! parallel executor; until then plans are sequential DAGs.
//!
//! Internal storage:
//!
//! - `nodes: Vec<PlanNode>` — every stage with its erased shadow
//!   and JSON args.
//! - `edges: Vec<PlanEdge>` — `(from, to)` tuples that form the
//!   DAG. The executor topo-sorts these.
//! - `leading: Vec<NodeId>` — the leading-edge node(s) the next
//!   API call will read from. For linear chains this is always a
//!   single node; commit 6 grows it during `fork`/`merge`.
//! - `initial: HashMap<NodeId, ErasedArtifact>` — graph inputs.
//!   Populated when a stage's `Input = ()` (the empty artifact).
//! - PhantomData<fn() -> Out>: covariant type witness. We avoid
//!   `PhantomData<Out>` (invariant) so `Plan<&'a A>` compositions,
//!   if they ever arise, behave the natural way.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::backends::TrainingBackend;
use crate::framework::artifact::Artifact;
use crate::framework::cache::CacheHandle;
use crate::framework::compat::Compatible;
use crate::framework::stage::{ErasedArtifact, Stage, StageDyn};

/// 0-indexed identifier for nodes inside one plan.
pub type NodeId = u32;

#[derive(Clone)]
pub(crate) struct PlanNode {
    pub id: NodeId,
    pub stage: Arc<dyn StageDyn>,
    /// JSON-encoded `Stage::Args`. Stored canonical-ish (insertion
    /// order; the cache key path canonicalizes again before hashing).
    pub args: serde_json::Value,
    /// Precomputed canonical-JSON bytes of `args`. Filled at plan-
    /// compile time so cache-key derivation per stage doesn't walk
    /// the `Value` tree (opt-5). Kept alongside `args` because
    /// `args` is still used as the JSON payload passed to
    /// `StageDyn::run_erased` and persisted in `args.json`.
    pub canon_args: Vec<u8>,
    /// Per-node retry/timeout overrides (D1/D2). `None` = use the
    /// stage's `RETRY`/`TIMEOUT` const. Set via `Plan::with_retry` /
    /// `Plan::with_timeout`, which apply to the current leading node(s).
    pub retry: Option<crate::framework::retry::RetryPolicy>,
    pub timeout: Option<crate::framework::retry::StageTimeout>,
    /// PlanSpec v1.1 / ADR 0102 pass #4: optional user scheduling priority.
    /// `None` == 0 (neutral). Higher runs earlier among *ready* nodes — it
    /// reorders the ready queue only, never bypassing broker admission, and is
    /// scheduling metadata (NOT part of a node's cache key, like retry/timeout).
    pub priority: Option<i32>,
    /// Optional partition identity. This is data identity, not scheduling
    /// metadata: the executor appends it to the cache key after every legacy
    /// cache-key input. `None` therefore preserves the exact pre-partition key.
    pub partition: Option<blut_types::partition::PartitionKey>,
}

/// Per-node execution-control overrides applied after `from_erased_graph`
/// (PlanSpec v1.1 retry/timeout — ADR 0088; priority — ADR 0102). Each `Some`
/// REPLACES the stage's own default; `None` leaves it. Execution-control only:
/// none of these fields is ever a node cache-key input. A named struct (rather
/// than a positional tuple) so later 0102 increments (e.g. a `pure` flag for
/// speculative execution) extend it without churning every call site.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ExecutionOverrides {
    pub retry: Option<crate::framework::retry::RetryPolicy>,
    pub timeout: Option<crate::framework::retry::StageTimeout>,
    pub priority: Option<i32>,
}

impl std::fmt::Debug for PlanNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanNode")
            .field("id", &self.id)
            .field("stage_name", &self.stage.name())
            .field("input_kind", &self.stage.input_kind())
            .field("output_kind", &self.stage.output_kind())
            .field("args", &self.args)
            .field("partition", &self.partition)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanEdge {
    pub from: NodeId,
    pub to: NodeId,
}

/// Typed DAG. `Out` is the type at the leading edge; `B` is the
/// training backend the plan targets. Stages added via
/// `.then()` / `.fork*` / `.merge*` must satisfy
/// `Compatible<B>` — a wrong-backend wire becomes a cargo build
/// error.
///
/// At runtime `B` is pure PhantomData; the executor reads
/// `nodes`/`edges`/`initial` without caring which backend the
/// plan targets. Erasure at the catalog boundary happens via
/// `Plan::<(), B>::into_compiled() → CompiledPlan` (lands as part
/// of BB-3 too).
pub struct Plan<Out, B: TrainingBackend> {
    pub(crate) name: String,
    pub(crate) nodes: Vec<PlanNode>,
    pub(crate) edges: Vec<PlanEdge>,
    pub(crate) leading: Vec<NodeId>,
    pub(crate) initial: HashMap<NodeId, ErasedArtifact>,
    /// Recipe args (or top-level user args) that produced this
    /// plan. Persisted for audit; the cache uses per-stage
    /// `args` only.
    pub(crate) recipe_args: serde_json::Value,
    _phantom: PhantomData<(fn() -> Out, B)>,
}

impl<Out, B: TrainingBackend> std::fmt::Debug for Plan<Out, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plan")
            .field("name", &self.name)
            .field("n_nodes", &self.nodes.len())
            .field("n_edges", &self.edges.len())
            .field("leading", &self.leading)
            .finish()
    }
}

impl<B: TrainingBackend> Plan<(), B> {
    /// New, empty plan. `start` is the canonical entry point —
    /// it appends the first node and produces a `Plan<S::Output, B>`.
    pub fn new(name: impl Into<String>, recipe_args: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            leading: Vec::new(),
            initial: HashMap::new(),
            recipe_args,
            _phantom: PhantomData,
        }
    }

    /// Append the first stage. Compiler enforces `Input = ()` AND
    /// `Compatible<B>` — only graph-input stages whose backend
    /// matches the plan's `B` can start.
    pub fn start<S>(mut self, stage: S, args: S::Args) -> Plan<S::Output, B>
    where
        S: Stage<Input = ()> + Compatible<B> + 'static,
    {
        let id = self.nodes.len() as NodeId;
        let args_json = serde_json::to_value(&args)
            .expect("Stage::Args must serialize to JSON; verify the type's Serialize impl");
        let canon_args = CacheHandle::canonical_json_bytes(&args_json);
        self.nodes.push(PlanNode {
            id,
            stage: Arc::new(stage),
            args: args_json,
            canon_args,
            retry: None,
            timeout: None,
            priority: None,
            partition: None,
        });
        // Graph input: provide () as the input artifact.
        let unit = ErasedArtifact::from_typed(&()).expect("() always serializes");
        self.initial.insert(id, unit);
        self.leading = vec![id];
        Plan {
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            leading: self.leading,
            initial: self.initial,
            recipe_args: self.recipe_args,
            _phantom: PhantomData,
        }
    }
}

/// Output plan type of a 3-way [`Plan::fork3`]: the three siblings'
/// outputs rejoined into a single `(A, B, C)` tuple artifact.
type Fork3Plan<SA, SB, SC, B> = Plan<
    (
        <SA as Stage>::Output,
        <SB as Stage>::Output,
        <SC as Stage>::Output,
    ),
    B,
>;

impl<O: Artifact, B: TrainingBackend> Plan<O, B> {
    /// Append a stage that consumes the leading edge's output.
    /// Compiler enforces `S::Input = O` AND `S: Compatible<B>`.
    ///
    /// Wrong wiring is a compile error:
    ///
    /// ```compile_fail
    /// // Skipping a step in the middle of a typed chain must fail.
    /// // Stage A: () -> DataA. Stage C: DataB -> DataC. Chaining
    /// // C directly after A skips the required A->B step.
    /// # use blut::framework::*;
    /// # use blut::framework::plan::Plan;
    /// # use async_trait::async_trait;
    /// # use serde::{Serialize, Deserialize};
    /// # use std::path::Path;
    /// # #[derive(Clone, Serialize, Deserialize)] struct A;
    /// # #[derive(Clone, Serialize, Deserialize)] struct B;
    /// # #[derive(Clone, Serialize, Deserialize)] struct C;
    /// # impl Artifact for A { const KIND: &'static str = "a"; const SCHEMA: u32 = 1;
    /// #     fn content_hash(&self) -> ContentHash { ContentHash::of_bytes(&[]) }
    /// #     fn primary_path(&self) -> &Path { Path::new(".") } }
    /// # impl Artifact for B { const KIND: &'static str = "b"; const SCHEMA: u32 = 1;
    /// #     fn content_hash(&self) -> ContentHash { ContentHash::of_bytes(&[]) }
    /// #     fn primary_path(&self) -> &Path { Path::new(".") } }
    /// # impl Artifact for C { const KIND: &'static str = "c"; const SCHEMA: u32 = 1;
    /// #     fn content_hash(&self) -> ContentHash { ContentHash::of_bytes(&[]) }
    /// #     fn primary_path(&self) -> &Path { Path::new(".") } }
    /// # #[derive(Clone, Serialize, Deserialize, schemars::JsonSchema)] struct E;
    /// # struct MakeA; #[async_trait] impl Stage for MakeA {
    /// #   const NAME: &'static str = "a"; const SCHEMA: u32 = 1;
    /// #   const RESOURCES: &'static [Resource] = &[]; type Input = (); type Output = A; type Args = E;
    /// #   async fn run(&self, _: &StageContext, _: (), _: &E) -> Result<A, StageError> { Ok(A) } }
    /// # struct BC; #[async_trait] impl Stage for BC {
    /// #   const NAME: &'static str = "bc"; const SCHEMA: u32 = 1;
    /// #   const RESOURCES: &'static [Resource] = &[]; type Input = B; type Output = C; type Args = E;
    /// #   async fn run(&self, _: &StageContext, _: B, _: &E) -> Result<C, StageError> { Ok(C) } }
    /// let _: Plan<C> = Plan::new("bad", serde_json::json!({}))
    ///     .start(MakeA, E)
    ///     .then(BC, E);  // expected B, got A — compile error
    /// ```
    pub fn then<S>(mut self, stage: S, args: S::Args) -> Plan<S::Output, B>
    where
        S: Stage<Input = O> + Compatible<B> + 'static,
    {
        let id = self.nodes.len() as NodeId;
        let args_json = serde_json::to_value(&args)
            .expect("Stage::Args must serialize to JSON; verify the type's Serialize impl");
        let canon_args = CacheHandle::canonical_json_bytes(&args_json);
        self.nodes.push(PlanNode {
            id,
            stage: Arc::new(stage),
            args: args_json,
            canon_args,
            retry: None,
            timeout: None,
            priority: None,
            partition: None,
        });
        // Edge from each previous leading node to this one. For
        // linear chains this is always one edge; commit 6's
        // `merge` API exits a fork with multiple edges into one
        // node.
        for &from in &self.leading {
            self.edges.push(PlanEdge { from, to: id });
        }
        self.leading = vec![id];
        Plan {
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            leading: self.leading,
            initial: self.initial,
            recipe_args: self.recipe_args,
            _phantom: PhantomData,
        }
    }

    /// Terminator: erase the leading-edge type. The executor
    /// consumes `Plan<()>` (via `compile_for_execution`); a recipe's
    /// `compile` returns `Plan<()>`.
    pub fn finish(self) -> Plan<(), B> {
        Plan {
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            leading: self.leading,
            initial: self.initial,
            recipe_args: self.recipe_args,
            _phantom: PhantomData,
        }
    }

    /// Override the retry policy (D1) for the CURRENT leading node(s) —
    /// the stage(s) just added by `start`/`then`/`fork`/`merge`. Chains
    /// after any of them: `.then(Download, args).with_retry(policy)`.
    pub fn with_retry(mut self, policy: crate::framework::retry::RetryPolicy) -> Self {
        for &id in &self.leading {
            self.nodes[id as usize].retry = Some(policy);
        }
        self
    }

    /// Override the soft/hard timeout (D2) for the current leading
    /// node(s). Chains like `with_retry`.
    pub fn with_timeout(mut self, timeout: crate::framework::retry::StageTimeout) -> Self {
        for &id in &self.leading {
            self.nodes[id as usize].timeout = Some(timeout);
        }
        self
    }

    /// Branch into two siblings consuming `O`. Both stages take
    /// the leading edge as input; their outputs land in a typed
    /// tuple at the new leading edge. Rejoin via `Plan<(L::Output,
    /// R::Output)>::merge<S>` requiring `S::Input = (L::Output,
    /// R::Output)`.
    pub fn fork<L, R>(
        mut self,
        left: L,
        l_args: L::Args,
        right: R,
        r_args: R::Args,
    ) -> Plan<(L::Output, R::Output), B>
    where
        L: Stage<Input = O> + Compatible<B> + 'static,
        R: Stage<Input = O> + Compatible<B> + 'static,
    {
        let l_id = self.nodes.len() as NodeId;
        let l_args_json = serde_json::to_value(&l_args).expect("Stage::Args serialize");
        let l_canon = CacheHandle::canonical_json_bytes(&l_args_json);
        self.nodes.push(PlanNode {
            id: l_id,
            stage: Arc::new(left),
            args: l_args_json,
            canon_args: l_canon,
            retry: None,
            timeout: None,
            priority: None,
            partition: None,
        });
        let r_id = self.nodes.len() as NodeId;
        let r_args_json = serde_json::to_value(&r_args).expect("Stage::Args serialize");
        let r_canon = CacheHandle::canonical_json_bytes(&r_args_json);
        self.nodes.push(PlanNode {
            id: r_id,
            stage: Arc::new(right),
            args: r_args_json,
            canon_args: r_canon,
            retry: None,
            timeout: None,
            priority: None,
            partition: None,
        });
        for &from in &self.leading {
            self.edges.push(PlanEdge { from, to: l_id });
            self.edges.push(PlanEdge { from, to: r_id });
        }
        self.leading = vec![l_id, r_id];
        Plan {
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            leading: self.leading,
            initial: self.initial,
            recipe_args: self.recipe_args,
            _phantom: PhantomData,
        }
    }

    /// 3-way fork. Same shape as `fork` but with three siblings
    /// rejoining via `Plan<(A, B, C)>::merge`.
    pub fn fork3<SA, SB, SC>(
        mut self,
        a: SA,
        a_args: SA::Args,
        b: SB,
        b_args: SB::Args,
        c: SC,
        c_args: SC::Args,
    ) -> Fork3Plan<SA, SB, SC, B>
    where
        SA: Stage<Input = O> + Compatible<B> + 'static,
        SB: Stage<Input = O> + Compatible<B> + 'static,
        SC: Stage<Input = O> + Compatible<B> + 'static,
    {
        let mut new_ids = Vec::with_capacity(3);
        for (stage, args) in [
            (
                Arc::new(a) as Arc<dyn StageDyn>,
                serde_json::to_value(&a_args).expect("a_args"),
            ),
            (
                Arc::new(b) as Arc<dyn StageDyn>,
                serde_json::to_value(&b_args).expect("b_args"),
            ),
            (
                Arc::new(c) as Arc<dyn StageDyn>,
                serde_json::to_value(&c_args).expect("c_args"),
            ),
        ] {
            let id = self.nodes.len() as NodeId;
            let canon_args = CacheHandle::canonical_json_bytes(&args);
            self.nodes.push(PlanNode {
                id,
                stage,
                args,
                canon_args,
                retry: None,
                timeout: None,
                priority: None,
                partition: None,
            });
            for &from in &self.leading {
                self.edges.push(PlanEdge { from, to: id });
            }
            new_ids.push(id);
        }
        self.leading = new_ids;
        Plan {
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            leading: self.leading,
            initial: self.initial,
            recipe_args: self.recipe_args,
            _phantom: PhantomData,
        }
    }
}

impl<A1: Artifact, A2: Artifact, B: TrainingBackend> Plan<(A1, A2), B> {
    /// Merge a forked branch via a stage that consumes the tuple.
    pub fn merge<S>(mut self, stage: S, args: S::Args) -> Plan<S::Output, B>
    where
        S: Stage<Input = (A1, A2)> + Compatible<B> + 'static,
    {
        let id = self.nodes.len() as NodeId;
        let args_json = serde_json::to_value(&args).expect("Stage::Args serialize");
        let canon_args = CacheHandle::canonical_json_bytes(&args_json);
        self.nodes.push(PlanNode {
            id,
            stage: Arc::new(stage),
            args: args_json,
            canon_args,
            retry: None,
            timeout: None,
            priority: None,
            partition: None,
        });
        for &from in &self.leading {
            self.edges.push(PlanEdge { from, to: id });
        }
        self.leading = vec![id];
        Plan {
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            leading: self.leading,
            initial: self.initial,
            recipe_args: self.recipe_args,
            _phantom: PhantomData,
        }
    }
}

impl<A1: Artifact, A2: Artifact, A3: Artifact, B: TrainingBackend> Plan<(A1, A2, A3), B> {
    /// Merge a 3-way forked branch.
    pub fn merge3<S>(mut self, stage: S, args: S::Args) -> Plan<S::Output, B>
    where
        S: Stage<Input = (A1, A2, A3)> + Compatible<B> + 'static,
    {
        let id = self.nodes.len() as NodeId;
        let args_json = serde_json::to_value(&args).expect("Stage::Args serialize");
        let canon_args = CacheHandle::canonical_json_bytes(&args_json);
        self.nodes.push(PlanNode {
            id,
            stage: Arc::new(stage),
            args: args_json,
            canon_args,
            retry: None,
            timeout: None,
            priority: None,
            partition: None,
        });
        for &from in &self.leading {
            self.edges.push(PlanEdge { from, to: id });
        }
        self.leading = vec![id];
        Plan {
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            leading: self.leading,
            initial: self.initial,
            recipe_args: self.recipe_args,
            _phantom: PhantomData,
        }
    }
}

// Inspection methods (name/n_nodes/topo_order/render_ascii) and the
// executor-facing exec_view live on `CompiledPlan` (the erased
// post-finish form). Use `.into_compiled()` on a `Plan<(), B>` to
// reach them. Keeping both copies would diverge over time; one
// source of truth wins.

/// Borrow-only access for the executor. Avoids exposing
/// `nodes` / `edges` / `initial` as `pub` while still letting the
/// executor walk them. `pub(crate)` on the struct + fields keeps
/// the surface fully internal — no accidental external coupling.
pub(crate) struct ExecView<'a> {
    pub nodes: &'a [PlanNode],
    pub edges: &'a [PlanEdge],
    pub initial: &'a HashMap<NodeId, ErasedArtifact>,
    pub recipe_args: &'a serde_json::Value,
}

impl<B: TrainingBackend> Plan<(), B> {
    /// Erase `B` for catalog storage + heterogeneous dispatch.
    /// Recipe `DEF.compile_fn` closures call this so a single
    /// `RecipeDef` slice can hold compile_fns whose typed plans
    /// span different backends. The runtime `CompiledPlan` carries
    /// the same internal structure — `B` was only PhantomData.
    pub fn into_compiled(self) -> CompiledPlan {
        CompiledPlan {
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
            initial: self.initial,
            recipe_args: self.recipe_args,
            expansions: Vec::new(),
        }
    }
}

/// Backend-erased plan, ready for execution. Produced from a
/// `Plan<(), B>` via `into_compiled()` at the recipe catalog
/// boundary. Carries every field the executor needs; the only
/// thing dropped is the compile-time backend witness.
pub struct CompiledPlan {
    pub(crate) name: String,
    pub(crate) nodes: Vec<PlanNode>,
    pub(crate) edges: Vec<PlanEdge>,
    pub(crate) initial: HashMap<NodeId, ErasedArtifact>,
    pub(crate) recipe_args: serde_json::Value,
    /// Typed runtime fan-outs (ADR 0078 `map_output`): when node `parent`
    /// completes with a `list` output, the executor spawns one instance of
    /// `template` per element, seeded with that element. Empty for every plan
    /// that has no map (the common case).
    pub(crate) expansions: Vec<MapExpansion>,
}

/// One compiled runtime fan-out (ADR 0078). Attached to a [`CompiledPlan`];
/// the executor consults it on the completion seam.
#[derive(Clone)]
pub(crate) struct MapExpansion {
    /// The node whose `list` output drives the fan-out.
    pub parent: NodeId,
    /// The sub-plan instantiated once per list element (its single root is
    /// seeded with the element instead of the unit graph input).
    pub template: Arc<CompiledTemplate>,
    /// Optional display label; spawned instances are labelled `label[i]`.
    pub label: Option<String>,
}

/// A kind-checked map template: like a [`CompiledPlan`] but its single root
/// consumes a list ELEMENT (kind `elem_kind`) supplied at runtime, so it
/// carries no `initial` seeding. Cloned per element at expansion time.
#[derive(Clone)]
pub(crate) struct CompiledTemplate {
    /// The sole root node (0 predecessors), which takes the element.
    pub root: NodeId,
    pub nodes: Vec<PlanNode>,
    pub edges: Vec<PlanEdge>,
    /// The element `KIND` the root consumes (== the parent's element kind).
    pub elem_kind: String,
}

impl CompiledTemplate {
    /// Instantiate this template as a standalone [`CompiledPlan`] (cloning its
    /// nodes/edges). The root's input is supplied at injection time via
    /// `SpawnDelta::root_seeds`, so `initial` is empty. Cheap: `PlanNode`
    /// clones are `Arc`-backed.
    pub(crate) fn instantiate(&self, name: String) -> CompiledPlan {
        // A template always has ≥1 node (its root); the caller pairs this with
        // a `root_seeds` entry, so `initial` is intentionally empty.
        debug_assert!(!self.nodes.is_empty(), "map template must have a root node");
        CompiledPlan {
            name,
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            initial: HashMap::new(),
            recipe_args: serde_json::Value::Null,
            expansions: Vec::new(),
        }
    }
}

impl CompiledPlan {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
    pub fn recipe_args(&self) -> &serde_json::Value {
        &self.recipe_args
    }

    /// Stable identity of the executable graph excluding the logical
    /// partition key itself. Includes build code, stage-provided external-code
    /// fingerprints, node args, topology, and runtime map templates. The
    /// partition status matrix combines this with source/compiled recipe args
    /// so code drift reads stale exactly as the executor's cache would miss.
    pub fn execution_fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        fn add_node(hasher: &mut Sha256, node: &PlanNode) {
            hasher.update(node.id.to_le_bytes());
            hasher.update(node.stage.name().as_bytes());
            hasher.update([0]);
            hasher.update(node.stage.schema().to_le_bytes());
            hasher.update((node.canon_args.len() as u64).to_le_bytes());
            hasher.update(&node.canon_args);
            if let Some(code) = node.stage.code_fingerprint() {
                hasher.update((code.len() as u64).to_le_bytes());
                hasher.update(code);
            } else {
                hasher.update(0u64.to_le_bytes());
            }
        }

        let mut hasher = Sha256::new();
        hasher.update(b"blut.plan.execution.v1");
        hasher.update(env!("BLUT_GIT_HASH").as_bytes());
        for node in &self.nodes {
            add_node(&mut hasher, node);
        }
        for edge in &self.edges {
            hasher.update(edge.from.to_le_bytes());
            hasher.update(edge.to.to_le_bytes());
        }
        for expansion in &self.expansions {
            hasher.update(expansion.parent.to_le_bytes());
            for node in &expansion.template.nodes {
                add_node(&mut hasher, node);
            }
            for edge in &expansion.template.edges {
                hasher.update(edge.from.to_le_bytes());
                hasher.update(edge.to.to_le_bytes());
            }
        }
        faster_hex::hex_string(&hasher.finalize())
    }

    /// Bind this compiled run to one partition cell.
    ///
    /// Partition identity is attached to every node, including nodes inside
    /// runtime map templates, so no stage can reuse an artifact produced for a
    /// different cell. Plans default to `None`, which emits no new cache-key
    /// bytes and preserves the exact pre-partition behavior.
    pub fn with_partition(mut self, partition: blut_types::partition::PartitionKey) -> Self {
        for node in &mut self.nodes {
            node.partition = Some(partition.clone());
        }
        for expansion in &mut self.expansions {
            for node in &mut Arc::make_mut(&mut expansion.template).nodes {
                node.partition = Some(partition.clone());
            }
        }
        self
    }

    /// The plan's runtime `map_output` expansions (ADR 0078). Crate-internal
    /// (the executor + tests read it); empty for a plain DAG.
    pub(crate) fn expansions(&self) -> &[MapExpansion] {
        &self.expansions
    }

    /// Replace the recipe-args provenance blob (audit-only; the executor uses
    /// each node's own args, never this). Used by the CLI to stamp richer
    /// provenance — e.g. a `.star` script's path + fingerprint — onto a plan
    /// compiled from a `PlanSpec`.
    pub fn override_recipe_args(mut self, recipe_args: serde_json::Value) -> Self {
        self.recipe_args = recipe_args;
        self
    }

    pub fn render_ascii(&self) -> Result<String, crate::framework::error::PlanError> {
        let order = self.topo_order()?;
        let mut out = String::new();
        out.push_str(&format!(
            "plan: {} ({} nodes, {} edges)\n",
            self.name,
            self.nodes.len(),
            self.edges.len()
        ));
        for (idx, &node_id) in order.iter().enumerate() {
            let stage = &self.nodes[node_id as usize].stage;
            let preds: Vec<u32> = self
                .edges
                .iter()
                .filter(|e| e.to == node_id)
                .map(|e| e.from)
                .collect();
            let preds_str = if preds.is_empty() {
                String::from("─")
            } else {
                preds
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            };
            out.push_str(&format!(
                "  [{idx:>2}] {:<24} <- {preds_str}\n",
                stage.name()
            ));
        }
        Ok(out)
    }

    pub fn topo_order(&self) -> Result<Vec<NodeId>, crate::framework::error::PlanError> {
        let n = self.nodes.len();
        if n == 0 {
            return Err(crate::framework::error::PlanError::Empty);
        }
        let mut indeg: Vec<u32> = vec![0; n];
        for e in &self.edges {
            indeg[e.to as usize] += 1;
        }
        let mut adj: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        for e in &self.edges {
            adj[e.from as usize].push(e.to);
        }
        let mut ready: std::collections::VecDeque<NodeId> = (0..n as NodeId)
            .filter(|&i| indeg[i as usize] == 0)
            .collect();
        let mut order = Vec::with_capacity(n);
        while let Some(id) = ready.pop_front() {
            order.push(id);
            for &next in &adj[id as usize] {
                let d = &mut indeg[next as usize];
                *d -= 1;
                if *d == 0 {
                    ready.push_back(next);
                }
            }
        }
        if order.len() != n {
            let offender = (0..n as NodeId)
                .find(|&i| indeg[i as usize] > 0)
                .unwrap_or(0);
            return Err(crate::framework::error::PlanError::Cycle(offender));
        }
        Ok(order)
    }

    pub(crate) fn exec_view(&self) -> ExecView<'_> {
        ExecView {
            nodes: &self.nodes,
            edges: &self.edges,
            initial: &self.initial,
            recipe_args: &self.recipe_args,
        }
    }

    /// Move the structural pieces out for a runtime `Spawn` injection — the
    /// executor relabels these into the running graph. (`recipe_args`/`name` are
    /// dropped; a spawned sub-plan's args live in its own nodes.)
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        self,
    ) -> (
        Vec<PlanNode>,
        Vec<PlanEdge>,
        HashMap<NodeId, ErasedArtifact>,
    ) {
        (self.nodes, self.edges, self.initial)
    }

    /// Serializable plan STRUCTURE for the DAG backend (v0.20): nodes laid out
    /// in topological order so each node's `idx` equals the `node_idx` the
    /// executor stamps on its `StageEvent`s, and edges remapped to those topo
    /// indices. Persisted at launch as `<job_dir>/plan.json`; the live status is
    /// joined in by [`crate::framework::graph::graph_snapshot`]. Errors only on a
    /// cyclic/empty plan (which would also fail execution).
    pub fn graph_structure(
        &self,
    ) -> Result<crate::framework::graph::PlanGraph, crate::framework::error::PlanError> {
        use crate::framework::graph::{PlanGraph, PlanGraphEdge, PlanGraphNode};
        let order = self.topo_order()?;
        // NodeId → topo position (the inverse of `order`).
        let mut pos = vec![0usize; self.nodes.len()];
        for (p, &nid) in order.iter().enumerate() {
            pos[nid as usize] = p;
        }
        let nodes = order
            .iter()
            .enumerate()
            .map(|(p, &nid)| {
                let node = &self.nodes[nid as usize];
                PlanGraphNode {
                    idx: p,
                    stage_name: node.stage.name().to_string(),
                    args_summary: crate::framework::graph::summarize_args(&node.args),
                }
            })
            .collect();
        let edges = self
            .edges
            .iter()
            .map(|e| PlanGraphEdge {
                from: pos[e.from as usize],
                to: pos[e.to as usize],
            })
            .collect();
        Ok(PlanGraph {
            name: self.name.clone(),
            nodes,
            edges,
        })
    }

    /// Merge N independent compiled plans into one (HPO fan-out, v0.20). Each
    /// component becomes a disjoint connected sub-graph with its node ids offset
    /// by the running total, so the executor runs all N in parallel (up to the
    /// concurrency cap), gated by the GPU semaphore + never-OOM admission
    /// exactly as today. Per-component `recipe_args` are dropped (each trial's
    /// args live in its own nodes' `args`/`canon_args`); the merged
    /// `recipe_args` is the supplied `base_args` (for footprint billing /
    /// provenance). Returns `(merged, node_offsets)` where `node_offsets[i]` is
    /// the first global node id of component `i` — the caller maps trial → node
    /// range with it. Empty `components` yields an empty plan (the caller guards
    /// against launching it).
    pub fn from_components(
        name: String,
        base_args: serde_json::Value,
        components: Vec<CompiledPlan>,
    ) -> (CompiledPlan, Vec<NodeId>) {
        let mut nodes: Vec<PlanNode> = Vec::new();
        let mut edges: Vec<PlanEdge> = Vec::new();
        let mut initial: HashMap<NodeId, ErasedArtifact> = HashMap::new();
        let mut node_offsets: Vec<NodeId> = Vec::with_capacity(components.len());
        let mut offset: NodeId = 0;
        for comp in components {
            node_offsets.push(offset);
            let comp_nodes = comp.nodes;
            let n = comp_nodes.len() as NodeId;
            for mut node in comp_nodes {
                node.id += offset;
                nodes.push(node);
            }
            for e in comp.edges {
                edges.push(PlanEdge {
                    from: e.from + offset,
                    to: e.to + offset,
                });
            }
            for (id, art) in comp.initial {
                initial.insert(id + offset, art);
            }
            offset += n;
        }
        (
            CompiledPlan {
                name,
                nodes,
                edges,
                initial,
                recipe_args: base_args,
                // Merged components (HPO fan-out) carry no map expansions.
                expansions: Vec::new(),
            },
            node_offsets,
        )
    }

    /// Build a LINEAR erased plan from a chain of `(stage, args)` — the
    /// DECLARATIVE (`.toml`) path (Phase G / C3). The typed [`Plan`] builder
    /// proves stage wiring at COMPILE time; a `.toml` recipe is loaded at
    /// RUNTIME with a dynamic stage list, so this checks the same contract at
    /// runtime via `StageDyn::input_kind()`/`output_kind()`:
    ///
    ///   * the FIRST stage must be graph-input (`input_kind() == "()"`), and
    ///   * each stage's `output_kind()` must equal the next's `input_kind()`.
    ///
    /// A break is a clear [`PlanError::Other`]. Edges form the linear chain
    /// `0→1→…→n-1`; node 0 receives the unit graph-input artifact, exactly as
    /// [`Plan::start`] does. The produced [`CompiledPlan`] runs through the
    /// SAME executor as a compiled recipe (the executor only ever sees erased
    /// `Arc<dyn StageDyn>` nodes).
    pub fn from_erased_chain(
        name: impl Into<String>,
        recipe_args: serde_json::Value,
        chain: Vec<(Arc<dyn StageDyn>, serde_json::Value)>,
    ) -> Result<CompiledPlan, crate::framework::error::PlanError> {
        use crate::framework::error::PlanError;
        let name = name.into();
        if chain.is_empty() {
            return Err(PlanError::Empty);
        }
        // The first stage must take the unit graph input. (Errors name the
        // STAGES/kinds only — the caller adds recipe context.)
        let first_in = chain[0].0.input_kind();
        if first_in != <() as Artifact>::KIND {
            return Err(PlanError::Other(format!(
                "first stage '{}' must be graph-input (input_kind \"()\"), but it \
                 expects '{first_in}'",
                chain[0].0.name()
            )));
        }
        // Consecutive kind contract: out(i) == in(i+1).
        for pair in chain.windows(2) {
            let out = pair[0].0.output_kind();
            let inp = pair[1].0.input_kind();
            if out != inp {
                return Err(PlanError::Other(format!(
                    "kind-chain break — stage '{}' outputs '{out}' but the next stage \
                     '{}' expects '{inp}'",
                    pair[0].0.name(),
                    pair[1].0.name()
                )));
            }
        }
        let mut nodes: Vec<PlanNode> = Vec::with_capacity(chain.len());
        let mut edges: Vec<PlanEdge> = Vec::new();
        for (i, (stage, args)) in chain.into_iter().enumerate() {
            let id = i as NodeId;
            let canon_args = CacheHandle::canonical_json_bytes(&args);
            nodes.push(PlanNode {
                id,
                stage,
                args,
                canon_args,
                retry: None,
                timeout: None,
                priority: None,
                partition: None,
            });
            if i > 0 {
                edges.push(PlanEdge {
                    from: (i - 1) as NodeId,
                    to: id,
                });
            }
        }
        let mut initial: HashMap<NodeId, ErasedArtifact> = HashMap::new();
        let unit = ErasedArtifact::from_typed(&())
            .map_err(|e| PlanError::Other(format!("encode unit graph input: {e}")))?;
        initial.insert(0, unit);
        Ok(CompiledPlan {
            name,
            nodes,
            edges,
            initial,
            recipe_args,
            expansions: Vec::new(),
        })
    }

    /// Apply PlanSpec v1.1 per-node execution overrides (ADR 0088). `overrides`
    /// aligns with `nodes` by index; a `Some` retry/timeout REPLACES the stage's
    /// own default, a `None` leaves it. Execution-control only — node cache keys
    /// are untouched (they are over stage code + args + input, never retry/timeout).
    pub(crate) fn apply_execution_overrides(&mut self, overrides: &[ExecutionOverrides]) {
        debug_assert_eq!(
            overrides.len(),
            self.nodes.len(),
            "execution overrides must align 1:1 with nodes"
        );
        for (node, ov) in self.nodes.iter_mut().zip(overrides) {
            if ov.retry.is_some() {
                node.retry = ov.retry;
            }
            if ov.timeout.is_some() {
                node.timeout = ov.timeout;
            }
            if ov.priority.is_some() {
                node.priority = ov.priority;
            }
        }
    }

    /// Build an ARBITRARY-topology erased plan from `nodes` + `edges` — the
    /// typed dynamic-DAG path (ADR 0078). This is `from_erased_chain`
    /// generalized from a linear chain to a full DAG: it extends the same
    /// runtime kind-check guarantee to fork/merge/map topologies so a
    /// `PlanSpec` authored by a Starlark script or raw JSON is checked before
    /// it runs. Nodes get dense ids `0..n` in slice order; `edges` are
    /// `(producer_index, consumer_index)` and their **insertion order is the
    /// tuple element order** the executor's `gather_input` will assemble for a
    /// merge node.
    ///
    /// Per-node kind contract (checked before any execution):
    ///   * a node with 0 predecessors (a graph source) must be graph-input
    ///     (`input_kind() == "()"`) and is seeded the unit artifact;
    ///   * a node with 1 predecessor must have `input_kind()` equal to that
    ///     predecessor's `output_kind()`;
    ///   * a node with N ≥ 2 predecessors must take `tuple<N>` (element kinds
    ///     are re-verified at `decode_erased`, exactly as the typed `merge`
    ///     path relies on — this checks arity here).
    ///
    /// A break is a precise [`PlanError`] naming the offending stage(s)/kinds;
    /// a cycle or empty plan is rejected too (reusing [`topo_order`]). Never
    /// panics on bad input — every malformed graph is a typed `Err`.
    /// `from_erased_chain` is left untouched (the declarative TOML path keeps
    /// its own byte-equal test surface).
    pub fn from_erased_graph(
        name: impl Into<String>,
        recipe_args: serde_json::Value,
        nodes: Vec<(Arc<dyn StageDyn>, serde_json::Value)>,
        edges: Vec<(NodeId, NodeId)>,
    ) -> Result<CompiledPlan, crate::framework::error::PlanError> {
        use crate::framework::error::PlanError;
        let name = name.into();
        let n = nodes.len();
        if n == 0 {
            return Err(PlanError::Empty);
        }
        // Edge endpoints must be in range (a dangling index is caught here,
        // not by a later panic on `nodes[idx]`), and no edge may repeat (a
        // duplicate would give a merge two copies of one producer instead of
        // distinct tuple elements).
        let mut seen_edges = std::collections::HashSet::new();
        for &(from, to) in &edges {
            if from as usize >= n || to as usize >= n {
                return Err(PlanError::EdgeOutOfRange {
                    from,
                    to,
                    n_nodes: n,
                });
            }
            if !seen_edges.insert((from, to)) {
                return Err(PlanError::DuplicateEdge { from, to });
            }
        }
        // Predecessors per node, in EDGE-INSERTION order (= tuple element
        // order the executor assembles — keep this in lockstep with
        // `gather_input`).
        let mut preds: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        for &(from, to) in &edges {
            preds[to as usize].push(from);
        }
        // Kind-check every node against its predecessor set.
        for (id, (stage, _)) in nodes.iter().enumerate() {
            let in_kind = stage.input_kind();
            match preds[id].as_slice() {
                [] => {
                    if in_kind != <() as Artifact>::KIND {
                        return Err(PlanError::RootNotGraphInput {
                            stage: stage.name().to_string(),
                            got: in_kind.to_string(),
                        });
                    }
                }
                [only] => {
                    let out_kind = nodes[*only as usize].0.output_kind();
                    if out_kind != in_kind {
                        return Err(PlanError::KindBreak {
                            from_stage: nodes[*only as usize].0.name().to_string(),
                            out_kind: out_kind.to_string(),
                            to_stage: stage.name().to_string(),
                            in_kind: in_kind.to_string(),
                        });
                    }
                }
                many => {
                    let k = many.len();
                    if in_kind != format!("tuple<{k}>") {
                        return Err(PlanError::BadMergeArity {
                            stage: stage.name().to_string(),
                            expected_kind: in_kind.to_string(),
                            got: k,
                        });
                    }
                }
            }
        }
        // Materialize. Ids are dense `0..n` by slice position; each source is
        // seeded the unit graph input exactly as `from_erased_chain` seeds
        // node 0.
        let mut plan_nodes: Vec<PlanNode> = Vec::with_capacity(n);
        let mut initial: HashMap<NodeId, ErasedArtifact> = HashMap::new();
        for (i, (stage, args)) in nodes.into_iter().enumerate() {
            let id = i as NodeId;
            let canon_args = CacheHandle::canonical_json_bytes(&args);
            if preds[i].is_empty() {
                let unit = ErasedArtifact::from_typed(&())
                    .map_err(|e| PlanError::Other(format!("encode unit graph input: {e}")))?;
                initial.insert(id, unit);
            }
            plan_nodes.push(PlanNode {
                id,
                stage,
                args,
                canon_args,
                retry: None,
                timeout: None,
                priority: None,
                partition: None,
            });
        }
        let plan_edges: Vec<PlanEdge> = edges
            .into_iter()
            .map(|(from, to)| PlanEdge { from, to })
            .collect();
        let plan = CompiledPlan {
            name,
            nodes: plan_nodes,
            edges: plan_edges,
            initial,
            recipe_args,
            expansions: Vec::new(),
        };
        // Reject cycles (reuses the Kahn walk + `PlanError::Cycle`).
        plan.topo_order()?;
        Ok(plan)
    }

    /// Attach runtime `map_output` expansions (ADR 0078). Builder used by
    /// `PlanSpec::compile` after `from_erased_graph`; every other path leaves
    /// `expansions` empty.
    pub(crate) fn with_expansions(mut self, expansions: Vec<MapExpansion>) -> Self {
        self.expansions = expansions;
        self
    }

    /// Compile a map TEMPLATE (ADR 0078): an arbitrary-topology sub-plan whose
    /// SINGLE root consumes a list element of kind `elem_kind` (supplied at
    /// runtime, so no `initial` seeding). Kind-checks exactly like
    /// `from_erased_graph` except the root takes `elem_kind` instead of the
    /// unit graph input, and there must be exactly one root (the element
    /// consumer). Returns a [`CompiledTemplate`] the executor clones per list
    /// element.
    pub(crate) fn from_erased_template(
        elem_kind: &str,
        nodes: Vec<(Arc<dyn StageDyn>, serde_json::Value)>,
        edges: Vec<(NodeId, NodeId)>,
    ) -> Result<CompiledTemplate, crate::framework::error::PlanError> {
        use crate::framework::error::PlanError;
        let n = nodes.len();
        if n == 0 {
            return Err(PlanError::Empty);
        }
        let mut seen_edges = std::collections::HashSet::new();
        for &(from, to) in &edges {
            if from as usize >= n || to as usize >= n {
                return Err(PlanError::EdgeOutOfRange {
                    from,
                    to,
                    n_nodes: n,
                });
            }
            if !seen_edges.insert((from, to)) {
                return Err(PlanError::DuplicateEdge { from, to });
            }
        }
        let mut preds: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        for &(from, to) in &edges {
            preds[to as usize].push(from);
        }
        // Kind-check; collect the root(s).
        let mut roots: Vec<NodeId> = Vec::new();
        for (id, (stage, _)) in nodes.iter().enumerate() {
            let in_kind = stage.input_kind();
            match preds[id].as_slice() {
                [] => {
                    // A template root consumes the ELEMENT, not `()`.
                    if in_kind != elem_kind {
                        return Err(PlanError::KindBreak {
                            from_stage: format!("<list element '{elem_kind}'>"),
                            out_kind: elem_kind.to_string(),
                            to_stage: stage.name().to_string(),
                            in_kind: in_kind.to_string(),
                        });
                    }
                    roots.push(id as NodeId);
                }
                [only] => {
                    let out_kind = nodes[*only as usize].0.output_kind();
                    if out_kind != in_kind {
                        return Err(PlanError::KindBreak {
                            from_stage: nodes[*only as usize].0.name().to_string(),
                            out_kind: out_kind.to_string(),
                            to_stage: stage.name().to_string(),
                            in_kind: in_kind.to_string(),
                        });
                    }
                }
                many => {
                    let k = many.len();
                    if in_kind != format!("tuple<{k}>") {
                        return Err(PlanError::BadMergeArity {
                            stage: stage.name().to_string(),
                            expected_kind: in_kind.to_string(),
                            got: k,
                        });
                    }
                }
            }
        }
        // Exactly one root — the element feeds a single consumer.
        if roots.len() != 1 {
            return Err(PlanError::Other(format!(
                "a map template must have exactly one root (the element consumer), found {}",
                roots.len()
            )));
        }
        let root = roots[0];
        let plan_nodes: Vec<PlanNode> = nodes
            .into_iter()
            .enumerate()
            .map(|(i, (stage, args))| PlanNode {
                id: i as NodeId,
                canon_args: CacheHandle::canonical_json_bytes(&args),
                stage,
                args,
                retry: None,
                timeout: None,
                priority: None,
                partition: None,
            })
            .collect();
        let plan_edges: Vec<PlanEdge> = edges
            .into_iter()
            .map(|(from, to)| PlanEdge { from, to })
            .collect();
        // Acyclicity: reuse the CompiledPlan Kahn walk on a throwaway.
        let probe = CompiledPlan {
            name: String::new(),
            nodes: plan_nodes.clone(),
            edges: plan_edges.clone(),
            initial: HashMap::new(),
            recipe_args: serde_json::Value::Null,
            expansions: Vec::new(),
        };
        probe.topo_order()?;
        Ok(CompiledTemplate {
            root,
            nodes: plan_nodes,
            edges: plan_edges,
            elem_kind: elem_kind.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;
    use crate::framework::error::StageError;
    use crate::framework::resource::Resource;
    use crate::framework::stage::StageContext;
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};
    use std::path::Path;

    // Toy artifacts.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct DataA;
    impl Artifact for DataA {
        const KIND: &'static str = "test.data_a";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(b"a")
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct DataB;
    impl Artifact for DataB {
        const KIND: &'static str = "test.data_b";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(b"b")
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct DataC;
    impl Artifact for DataC {
        const KIND: &'static str = "test.data_c";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(b"c")
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }

    // Toy stages: () → A, A → B, B → C.

    use crate::backends::LamuTrainerBackend;
    use crate::framework::Compatible;

    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct EmptyArgs;

    struct MakeA;
    impl Compatible<LamuTrainerBackend> for MakeA {}
    impl Compatible<LamuTrainerBackend> for AToB {}
    impl Compatible<LamuTrainerBackend> for BToC {}
    #[async_trait]
    impl Stage for MakeA {
        const NAME: &'static str = "make_a";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = DataA;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: (),
            _args: &EmptyArgs,
        ) -> Result<DataA, StageError> {
            Ok(DataA)
        }
    }

    struct AToB;
    #[async_trait]
    impl Stage for AToB {
        const NAME: &'static str = "a_to_b";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = DataA;
        type Output = DataB;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: DataA,
            _args: &EmptyArgs,
        ) -> Result<DataB, StageError> {
            Ok(DataB)
        }
    }

    struct BToC;
    #[async_trait]
    impl Stage for BToC {
        const NAME: &'static str = "b_to_c";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = DataB;
        type Output = DataC;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: DataB,
            _args: &EmptyArgs,
        ) -> Result<DataC, StageError> {
            Ok(DataC)
        }
    }

    // Extra toy stages exercising the graph (non-linear) paths:
    //   MakeB: () -> B   (a second graph source, for a join)
    //   JoinAB: (A, B) -> C   (an N-predecessor merge node)
    //   AToA: A -> A   (kind-stable, so a cycle passes the kind-check and
    //                   reaches topo_order — used to test cycle rejection)
    struct MakeB;
    impl Compatible<LamuTrainerBackend> for MakeB {}
    #[async_trait]
    impl Stage for MakeB {
        const NAME: &'static str = "make_b";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = DataB;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: (),
            _args: &EmptyArgs,
        ) -> Result<DataB, StageError> {
            Ok(DataB)
        }
    }

    struct JoinAB;
    impl Compatible<LamuTrainerBackend> for JoinAB {}
    #[async_trait]
    impl Stage for JoinAB {
        const NAME: &'static str = "join_ab";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = (DataA, DataB);
        type Output = DataC;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: (DataA, DataB),
            _args: &EmptyArgs,
        ) -> Result<DataC, StageError> {
            Ok(DataC)
        }
    }

    struct AToA;
    impl Compatible<LamuTrainerBackend> for AToA {}
    #[async_trait]
    impl Stage for AToA {
        const NAME: &'static str = "a_to_a";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = DataA;
        type Output = DataA;
        type Args = EmptyArgs;
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: DataA,
            _args: &EmptyArgs,
        ) -> Result<DataA, StageError> {
            Ok(DataA)
        }
    }

    // Convenience: an erased node tuple for from_erased_graph tests.
    fn erased<S: StageDyn + 'static>(s: S) -> (Arc<dyn StageDyn>, serde_json::Value) {
        (Arc::new(s) as Arc<dyn StageDyn>, serde_json::json!({}))
    }

    // `CompiledPlan` has no `Debug` (deliberate), so `Result::unwrap_err`
    // (which needs the Ok type to be `Debug`) can't be used — pull the error
    // out by hand.
    fn expect_err(
        r: Result<CompiledPlan, crate::framework::error::PlanError>,
    ) -> crate::framework::error::PlanError {
        match r {
            Ok(_) => panic!("expected an Err, got Ok(CompiledPlan)"),
            Err(e) => e,
        }
    }

    #[test]
    fn empty_plan_topo_errors() {
        let p = Plan::<(), LamuTrainerBackend>::new("empty", serde_json::json!({})).into_compiled();
        let r = p.topo_order();
        assert!(matches!(r, Err(crate::framework::error::PlanError::Empty)));
    }

    #[test]
    fn linear_three_stage_plan_compiles_and_orders() {
        let plan = Plan::<(), LamuTrainerBackend>::new("linear", serde_json::json!({}))
            .start(MakeA, EmptyArgs)
            .then(AToB, EmptyArgs)
            .then(BToC, EmptyArgs)
            .finish()
            .into_compiled();
        assert_eq!(plan.n_nodes(), 3);
        assert_eq!(plan.n_edges(), 2);
        let order = plan.topo_order().unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn from_components_merges_disjoint_subplans() {
        // Two 2-node components (MakeA -> AToB) → one 4-node plan with ids,
        // edges, and graph-input initials offset, so the executor runs both
        // trials in parallel (the HPO fan-out).
        let mk = || {
            Plan::<(), LamuTrainerBackend>::new("c", serde_json::json!({}))
                .start(MakeA, EmptyArgs)
                .then(AToB, EmptyArgs)
                .finish()
                .into_compiled()
        };
        let (merged, offsets) = CompiledPlan::from_components(
            "hpo".into(),
            serde_json::json!({ "x": 1 }),
            vec![mk(), mk()],
        );
        assert_eq!(merged.n_nodes(), 4);
        assert_eq!(merged.n_edges(), 2);
        assert_eq!(offsets, vec![0, 2], "first node id per component");
        let ids: Vec<NodeId> = merged.nodes.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3], "node ids relabeled contiguously");
        assert!(merged.edges.iter().any(|e| e.from == 0 && e.to == 1));
        assert!(
            merged.edges.iter().any(|e| e.from == 2 && e.to == 3),
            "2nd edge offset"
        );
        assert!(
            merged.initial.contains_key(&0) && merged.initial.contains_key(&2),
            "both graph-input initials offset"
        );
        assert_eq!(
            merged.topo_order().unwrap().len(),
            4,
            "valid DAG, all nodes ordered"
        );
        assert_eq!(merged.recipe_args, serde_json::json!({ "x": 1 }));
        assert_eq!(merged.name(), "hpo");
    }

    #[test]
    fn first_node_has_unit_initial_input() {
        let plan = Plan::<(), LamuTrainerBackend>::new("with_unit", serde_json::json!({}))
            .start(MakeA, EmptyArgs)
            .finish()
            .into_compiled();
        assert_eq!(plan.initial.len(), 1);
        let unit = plan.initial.get(&0).unwrap();
        assert_eq!(unit.kind, "()");
    }

    #[test]
    fn from_erased_chain_builds_and_kind_checks() {
        let a = serde_json::json!({});
        let stage = |s: Arc<dyn StageDyn>| (s, a.clone());

        // Valid: () → A → B → C builds a 3-node / 2-edge linear plan with the
        // unit initial input on node 0.
        let plan = CompiledPlan::from_erased_chain(
            "decl",
            serde_json::json!({}),
            vec![
                stage(Arc::new(MakeA)),
                stage(Arc::new(AToB)),
                stage(Arc::new(BToC)),
            ],
        )
        .expect("valid kind chain");
        assert_eq!(plan.n_nodes(), 3);
        assert_eq!(plan.n_edges(), 2);
        assert_eq!(plan.topo_order().unwrap(), vec![0, 1, 2]);
        assert_eq!(plan.initial.get(&0).unwrap().kind, "()");

        // First stage not graph-input → rejected (AToB expects test.data_a).
        // (CompiledPlan has no Debug, so match rather than expect_err.)
        match CompiledPlan::from_erased_chain(
            "bad_start",
            serde_json::json!({}),
            vec![stage(Arc::new(AToB)), stage(Arc::new(BToC))],
        ) {
            Err(e) => assert!(format!("{e}").contains("graph-input")),
            Ok(_) => panic!("non-graph-input first stage must be rejected"),
        }

        // Kind-chain break: MakeA outputs test.data_a, BToC expects test.data_b.
        match CompiledPlan::from_erased_chain(
            "broken",
            serde_json::json!({}),
            vec![stage(Arc::new(MakeA)), stage(Arc::new(BToC))],
        ) {
            Err(e) => assert!(format!("{e}").contains("kind-chain")),
            Ok(_) => panic!("kind-chain break must be rejected"),
        }

        // Empty chain → Empty.
        assert!(matches!(
            CompiledPlan::from_erased_chain("empty", serde_json::json!({}), vec![]),
            Err(crate::framework::error::PlanError::Empty)
        ));
    }

    #[test]
    fn topo_order_visits_each_node_once() {
        let plan = Plan::<(), LamuTrainerBackend>::new("p", serde_json::json!({}))
            .start(MakeA, EmptyArgs)
            .then(AToB, EmptyArgs)
            .then(BToC, EmptyArgs)
            .finish()
            .into_compiled();
        let order = plan.topo_order().unwrap();
        assert_eq!(order.len(), plan.n_nodes());
        let mut seen = std::collections::HashSet::new();
        for id in order {
            assert!(seen.insert(id), "duplicate id {id}");
        }
    }

    #[test]
    fn fork_creates_two_branches_from_one_input() {
        // MakeA -> [AToB | AToB] -> tuple<2>
        let plan = Plan::<(), LamuTrainerBackend>::new("forked", serde_json::json!({}))
            .start(MakeA, EmptyArgs)
            .fork(AToB, EmptyArgs, AToB, EmptyArgs)
            .finish()
            .into_compiled();
        assert_eq!(plan.n_nodes(), 3);
        // 2 edges from MakeA → each branch.
        assert_eq!(plan.n_edges(), 2);
        let order = plan.topo_order().unwrap();
        assert_eq!(order[0], 0);
        // Branches are 1 and 2 in some order.
        assert!(order[1..].contains(&1));
        assert!(order[1..].contains(&2));
    }

    #[test]
    fn recipe_args_round_trip_through_finish() {
        let args = serde_json::json!({"output_name": "test", "since": "30d"});
        let plan = Plan::<(), LamuTrainerBackend>::new("named", args.clone())
            .start(MakeA, EmptyArgs)
            .finish()
            .into_compiled();
        assert_eq!(plan.recipe_args(), &args);
        assert_eq!(plan.name(), "named");
    }

    // Compile-time DAG enforcement is also exercised by the
    // doctest on `Plan::then` (in this file's module-level docs).
    // A `compile_fail` doctest there is what cargo test --doc
    // actually runs; tests in #[cfg(test)] modules don't process
    // doctests. The dedicated negative test below is a runtime
    // proxy: building a typed chain is checked by the compiler;
    // mismatched-tuple Plan<(A, B)>::merge requires a stage with
    // Input = (A, B) which forces the compile-time witness.

    // ── ADR 0072 A7: DAG cycle-detection soundness (property-based) ────
    //
    // `topo_order`'s Kahn's-algorithm implementation only reads
    // `nodes.len()` + `edges` — it never inspects a node's kind-chain, so a
    // `CompiledPlan` can be built directly (bypassing the typed `Plan`
    // builder's kind-chain checks) with an arbitrary edge set. That is
    // exactly what we want here: the property under test is graph-
    // structural (topological-sort soundness), not the typed-stage DSL.

    /// Minimal stand-in node — `topo_order` never inspects a node's stage
    /// content, only `nodes.len()` and `edges`, so any `Stage` impl works.
    fn dummy_node(id: NodeId) -> PlanNode {
        PlanNode {
            id,
            stage: Arc::new(MakeA),
            args: serde_json::json!({}),
            canon_args: Vec::new(),
            retry: None,
            timeout: None,
            priority: None,
            partition: None,
        }
    }

    /// Build a bare `CompiledPlan` with `n` nodes (ids `0..n`) and the
    /// given `(from, to)` edges — no typed kind-chain checks, no `initial`
    /// entries (`topo_order` doesn't read either).
    fn make_compiled_plan(n: usize, edges: Vec<(NodeId, NodeId)>) -> CompiledPlan {
        CompiledPlan {
            name: "prop_plan".to_string(),
            nodes: (0..n as NodeId).map(dummy_node).collect(),
            edges: edges
                .into_iter()
                .map(|(from, to)| PlanEdge { from, to })
                .collect(),
            initial: HashMap::new(),
            recipe_args: serde_json::json!({}),
            expansions: Vec::new(),
        }
    }

    /// Strategy: a random small DAG (5..=10 nodes) with a sparse random
    /// subset of the forward-only pairs `(i, j)` for `i < j`. Restricting
    /// edges to `i < j` guarantees the generated graph is acyclic by
    /// construction (a topological order — the identity permutation —
    /// always exists), without needing a separate acyclicity check.
    fn arb_acyclic_dag() -> impl proptest::strategy::Strategy<Value = (usize, Vec<(NodeId, NodeId)>)>
    {
        use proptest::prelude::*;
        (5usize..=10).prop_flat_map(|n| {
            let pairs: Vec<(NodeId, NodeId)> = (0..n)
                .flat_map(|i| ((i + 1)..n).map(move |j| (i as NodeId, j as NodeId)))
                .collect();
            let len = pairs.len();
            prop::collection::vec(any::<bool>(), len).prop_map(move |mask| {
                let edges: Vec<(NodeId, NodeId)> = pairs
                    .iter()
                    .zip(mask.iter())
                    .filter(|&(_, &keep)| keep)
                    .map(|(&e, _)| e)
                    .collect();
                (n, edges)
            })
        })
    }

    /// Same generator, plus a deliberately-inserted cycle: a ring edge
    /// `i -> (i+1) % n` for every node. The ring alone gives every node an
    /// incoming edge, so Kahn's algorithm can never find a 0-indegree node
    /// to start from — the whole node set is one cycle, regardless of
    /// whatever forward edges are unioned in on top.
    fn arb_dag_with_cycle()
    -> impl proptest::strategy::Strategy<Value = (usize, Vec<(NodeId, NodeId)>)> {
        use proptest::strategy::Strategy;
        arb_acyclic_dag().prop_map(|(n, mut edges)| {
            for i in 0..n {
                edges.push((i as NodeId, ((i + 1) % n) as NodeId));
            }
            (n, edges)
        })
    }

    // ---- from_erased_graph (ADR 0078) ----

    #[test]
    fn from_erased_graph_linear_matches_typed_shape() {
        // () -> A -> B -> C via the graph constructor == the typed builder.
        let plan = CompiledPlan::from_erased_graph(
            "g",
            serde_json::json!({}),
            vec![erased(MakeA), erased(AToB), erased(BToC)],
            vec![(0, 1), (1, 2)],
        )
        .expect("linear graph compiles");
        assert_eq!(plan.n_nodes(), 3);
        assert_eq!(plan.n_edges(), 2);
        assert_eq!(plan.topo_order().unwrap(), vec![0, 1, 2]);
        // The sole root is seeded the unit graph input.
        assert!(plan.initial.contains_key(&0));
        assert_eq!(plan.initial.len(), 1);
    }

    #[test]
    fn from_erased_graph_join_two_sources_into_a_merge() {
        // MakeA, MakeB -> JoinAB((A,B)->C). Edge order (0,2),(1,2) = tuple order.
        let plan = CompiledPlan::from_erased_graph(
            "j",
            serde_json::json!({}),
            vec![erased(MakeA), erased(MakeB), erased(JoinAB)],
            vec![(0, 2), (1, 2)],
        )
        .expect("join graph compiles");
        assert_eq!(plan.n_nodes(), 3);
        assert_eq!(plan.n_edges(), 2);
        // Both sources are graph-input seeded; the merge is not.
        assert!(plan.initial.contains_key(&0));
        assert!(plan.initial.contains_key(&1));
        assert!(!plan.initial.contains_key(&2));
        let order = plan.topo_order().unwrap();
        assert_eq!(*order.last().unwrap(), 2, "merge runs last");
    }

    #[test]
    fn from_erased_graph_rejects_kind_break() {
        // MakeA outputs A; BToC expects B.
        let err = expect_err(CompiledPlan::from_erased_graph(
            "k",
            serde_json::json!({}),
            vec![erased(MakeA), erased(BToC)],
            vec![(0, 1)],
        ));
        match err {
            crate::framework::error::PlanError::KindBreak {
                from_stage,
                out_kind,
                to_stage,
                in_kind,
            } => {
                assert_eq!(from_stage, "make_a");
                assert_eq!(out_kind, "test.data_a");
                assert_eq!(to_stage, "b_to_c");
                assert_eq!(in_kind, "test.data_b");
            }
            other => panic!("expected KindBreak, got {other:?}"),
        }
    }

    #[test]
    fn from_erased_graph_rejects_non_root_source() {
        // AToB expects A but has no predecessor (it is a graph source).
        let err = expect_err(CompiledPlan::from_erased_graph(
            "r",
            serde_json::json!({}),
            vec![erased(AToB)],
            vec![],
        ));
        assert!(matches!(
            err,
            crate::framework::error::PlanError::RootNotGraphInput { .. }
        ));
    }

    #[test]
    fn from_erased_graph_rejects_bad_merge_arity() {
        // AToB expects a single A (arity 1) but is fed two predecessors.
        let err = expect_err(CompiledPlan::from_erased_graph(
            "m",
            serde_json::json!({}),
            vec![erased(MakeA), erased(MakeB), erased(AToB)],
            vec![(0, 2), (1, 2)],
        ));
        match err {
            crate::framework::error::PlanError::BadMergeArity {
                expected_kind, got, ..
            } => {
                // AToB's real input kind is shown, not a misleading "1-tuple".
                assert_eq!(expected_kind, "test.data_a");
                assert_eq!(got, 2);
            }
            other => panic!("expected BadMergeArity, got {other:?}"),
        }
    }

    #[test]
    fn from_erased_graph_rejects_duplicate_edge() {
        // The same (0,1) edge twice would give node 1 two copies of node 0's
        // output instead of distinct inputs — rejected up front.
        let err = expect_err(CompiledPlan::from_erased_graph(
            "d",
            serde_json::json!({}),
            vec![erased(MakeA), erased(AToA)],
            vec![(0, 1), (0, 1)],
        ));
        assert!(matches!(
            err,
            crate::framework::error::PlanError::DuplicateEdge { from: 0, to: 1 }
        ));
    }

    #[test]
    fn from_erased_graph_rejects_out_of_range_edge() {
        let err = expect_err(CompiledPlan::from_erased_graph(
            "e",
            serde_json::json!({}),
            vec![erased(MakeA)],
            vec![(0, 5)],
        ));
        assert!(matches!(
            err,
            crate::framework::error::PlanError::EdgeOutOfRange { .. }
        ));
    }

    #[test]
    fn from_erased_graph_rejects_empty() {
        let err = expect_err(CompiledPlan::from_erased_graph(
            "z",
            serde_json::json!({}),
            vec![],
            vec![],
        ));
        assert!(matches!(err, crate::framework::error::PlanError::Empty));
    }

    #[test]
    fn from_erased_graph_rejects_cycle() {
        // Two A->A nodes wired in a cycle: kinds line up (A==A) so the
        // per-node check passes, and topo_order catches the cycle.
        let err = expect_err(CompiledPlan::from_erased_graph(
            "c",
            serde_json::json!({}),
            vec![erased(AToA), erased(AToA)],
            vec![(0, 1), (1, 0)],
        ));
        assert!(matches!(err, crate::framework::error::PlanError::Cycle(_)));
    }

    proptest::proptest! {
        // A () -> A -> A -> ... chain (MakeA then AToA×) is always kind-valid,
        // so from_erased_graph must accept it at any length.
        #[test]
        fn from_erased_graph_accepts_any_valid_atoa_chain(n in 1usize..12) {
            let mut nodes: Vec<(Arc<dyn StageDyn>, serde_json::Value)> = vec![erased(MakeA)];
            for _ in 1..n { nodes.push(erased(AToA)); }
            let edges: Vec<(NodeId, NodeId)> = (1..n as NodeId).map(|i| (i - 1, i)).collect();
            let plan = CompiledPlan::from_erased_graph("p", serde_json::json!({}), nodes, edges)
                .expect("valid chain must compile");
            proptest::prop_assert_eq!(plan.n_nodes(), n);
        }

        // Arbitrary extra edges on top of a valid chain must always return a
        // Result (Ok or a typed Err) — never panic.
        #[test]
        fn from_erased_graph_never_panics_on_arbitrary_edges(
            n in 1usize..8,
            extra in proptest::collection::vec((0u32..8, 0u32..8), 0..6),
        ) {
            let mut nodes: Vec<(Arc<dyn StageDyn>, serde_json::Value)> = vec![erased(MakeA)];
            for _ in 1..n { nodes.push(erased(AToA)); }
            let mut edges: Vec<(NodeId, NodeId)> = (1..n as NodeId).map(|i| (i - 1, i)).collect();
            edges.extend(extra);
            // Just exercise it — the assertion is "does not panic".
            let _ = CompiledPlan::from_erased_graph("p", serde_json::json!({}), nodes, edges);
        }
    }

    proptest::proptest! {
        #[test]
        fn topo_order_succeeds_on_random_acyclic_dag((n, edges) in arb_acyclic_dag()) {
            let plan = make_compiled_plan(n, edges.clone());
            let order = plan.topo_order().expect("acyclic DAG must topo-sort");
            proptest::prop_assert_eq!(order.len(), n, "must visit exactly N nodes");
            let mut seen = std::collections::HashSet::new();
            for &id in &order {
                proptest::prop_assert!(seen.insert(id), "node {id} visited twice");
            }
            // Every edge's predecessor must precede its successor in the order.
            let pos: std::collections::HashMap<NodeId, usize> =
                order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
            for (from, to) in &edges {
                proptest::prop_assert!(
                    pos[from] < pos[to],
                    "edge {from}->{to} violated: pos[{from}]={} pos[{to}]={}",
                    pos[from],
                    pos[to]
                );
            }
        }

        #[test]
        fn topo_order_always_errs_on_dag_with_cycle((n, edges) in arb_dag_with_cycle()) {
            let plan = make_compiled_plan(n, edges);
            match plan.topo_order() {
                Err(crate::framework::error::PlanError::Cycle(_)) => {}
                Err(other) => proptest::prop_assert!(
                    false,
                    "a graph containing a cycle must report PlanError::Cycle, got {other:?}"
                ),
                Ok(order) => proptest::prop_assert!(
                    false,
                    "a graph containing a cycle must report PlanError::Cycle, got Ok(len={})",
                    order.len()
                ),
            }
        }
    }
}
