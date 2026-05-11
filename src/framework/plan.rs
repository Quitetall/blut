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

use crate::framework::artifact::Artifact;
use crate::framework::stage::{ErasedArtifact, Stage, StageDyn};

/// 0-indexed identifier for nodes inside one plan.
pub type NodeId = u32;

pub(crate) struct PlanNode {
    pub id: NodeId,
    pub stage: Box<dyn StageDyn>,
    /// JSON-encoded `Stage::Args`. Stored canonical-ish (insertion
    /// order; the cache key path canonicalizes again before hashing).
    pub args: serde_json::Value,
}

impl std::fmt::Debug for PlanNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanNode")
            .field("id", &self.id)
            .field("stage_name", &self.stage.name())
            .field("input_kind", &self.stage.input_kind())
            .field("output_kind", &self.stage.output_kind())
            .field("args", &self.args)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanEdge {
    pub from: NodeId,
    pub to: NodeId,
}

/// Typed DAG. The `Out` type is the type at the leading edge.
///
/// This struct is opaque to executor code; the executor looks at
/// `nodes`, `edges`, `initial`, and the topological order via
/// `Plan::compile_for_execution`.
pub struct Plan<Out> {
    pub(crate) name: String,
    pub(crate) nodes: Vec<PlanNode>,
    pub(crate) edges: Vec<PlanEdge>,
    pub(crate) leading: Vec<NodeId>,
    pub(crate) initial: HashMap<NodeId, ErasedArtifact>,
    /// Recipe args (or top-level user args) that produced this
    /// plan. Persisted for audit; the cache uses per-stage
    /// `args` only.
    pub(crate) recipe_args: serde_json::Value,
    _phantom: PhantomData<fn() -> Out>,
}

impl<Out> std::fmt::Debug for Plan<Out> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plan")
            .field("name", &self.name)
            .field("n_nodes", &self.nodes.len())
            .field("n_edges", &self.edges.len())
            .field("leading", &self.leading)
            .finish()
    }
}

impl Plan<()> {
    /// New, empty plan. `start` is the canonical entry point —
    /// it appends the first node and produces a `Plan<S::Output>`.
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

    /// Append the first stage. Compiler enforces `Input = ()` —
    /// the empty artifact — so only graph-input stages can start.
    /// Sets up `initial[node_id] = ErasedArtifact::from_typed(&())`.
    pub fn start<S>(mut self, stage: S, args: S::Args) -> Plan<S::Output>
    where
        S: Stage<Input = ()> + 'static,
    {
        let id = self.nodes.len() as NodeId;
        let args_json = serde_json::to_value(&args)
            .expect("Stage::Args must serialize to JSON; verify the type's Serialize impl");
        self.nodes.push(PlanNode {
            id,
            stage: Box::new(stage),
            args: args_json,
        });
        // Graph input: provide () as the input artifact.
        let unit = ErasedArtifact::from_typed(&())
            .expect("() always serializes");
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

impl<O: Artifact> Plan<O> {
    /// Append a stage that consumes the leading edge's output.
    /// Compiler enforces `S::Input = O`. The new node becomes the
    /// new leading edge.
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
    pub fn then<S>(mut self, stage: S, args: S::Args) -> Plan<S::Output>
    where
        S: Stage<Input = O> + 'static,
    {
        let id = self.nodes.len() as NodeId;
        let args_json = serde_json::to_value(&args)
            .expect("Stage::Args must serialize to JSON; verify the type's Serialize impl");
        self.nodes.push(PlanNode {
            id,
            stage: Box::new(stage),
            args: args_json,
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
    pub fn finish(self) -> Plan<()> {
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
    ) -> Plan<(L::Output, R::Output)>
    where
        L: Stage<Input = O> + 'static,
        R: Stage<Input = O> + 'static,
    {
        let l_id = self.nodes.len() as NodeId;
        let l_args_json = serde_json::to_value(&l_args).expect("Stage::Args serialize");
        self.nodes.push(PlanNode {
            id: l_id,
            stage: Box::new(left),
            args: l_args_json,
        });
        let r_id = self.nodes.len() as NodeId;
        let r_args_json = serde_json::to_value(&r_args).expect("Stage::Args serialize");
        self.nodes.push(PlanNode {
            id: r_id,
            stage: Box::new(right),
            args: r_args_json,
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
    pub fn fork3<A, B, C>(
        mut self,
        a: A,
        a_args: A::Args,
        b: B,
        b_args: B::Args,
        c: C,
        c_args: C::Args,
    ) -> Plan<(A::Output, B::Output, C::Output)>
    where
        A: Stage<Input = O> + 'static,
        B: Stage<Input = O> + 'static,
        C: Stage<Input = O> + 'static,
    {
        let mut new_ids = Vec::with_capacity(3);
        for (stage, args) in [
            (Box::new(a) as Box<dyn StageDyn>, serde_json::to_value(&a_args).expect("a_args")),
            (Box::new(b) as Box<dyn StageDyn>, serde_json::to_value(&b_args).expect("b_args")),
            (Box::new(c) as Box<dyn StageDyn>, serde_json::to_value(&c_args).expect("c_args")),
        ] {
            let id = self.nodes.len() as NodeId;
            self.nodes.push(PlanNode { id, stage, args });
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

impl<A: Artifact, B: Artifact> Plan<(A, B)> {
    /// Merge a forked branch via a stage that consumes the tuple.
    pub fn merge<S>(mut self, stage: S, args: S::Args) -> Plan<S::Output>
    where
        S: Stage<Input = (A, B)> + 'static,
    {
        let id = self.nodes.len() as NodeId;
        let args_json = serde_json::to_value(&args).expect("Stage::Args serialize");
        self.nodes.push(PlanNode {
            id,
            stage: Box::new(stage),
            args: args_json,
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

impl<A: Artifact, B: Artifact, C: Artifact> Plan<(A, B, C)> {
    /// Merge a 3-way forked branch.
    pub fn merge3<S>(mut self, stage: S, args: S::Args) -> Plan<S::Output>
    where
        S: Stage<Input = (A, B, C)> + 'static,
    {
        let id = self.nodes.len() as NodeId;
        let args_json = serde_json::to_value(&args).expect("Stage::Args serialize");
        self.nodes.push(PlanNode {
            id,
            stage: Box::new(stage),
            args: args_json,
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

impl Plan<()> {
    /// Recipe / user args that built this plan. Persisted at
    /// `<job_dir>/args.json` for audit + by the cache as a
    /// secondary input to per-stage cache keys (separately —
    /// recipe-arg invalidation cascades downstream automatically
    /// because each stage's `args` field is its slice of the
    /// recipe args).
    pub fn recipe_args(&self) -> &serde_json::Value {
        &self.recipe_args
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Topologically order the nodes via Kahn's algorithm. Errors
    /// with `PlanError::Cycle(first_offender)` if there's a cycle
    /// (impossible to construct via the typed builder — defense
    /// in depth for dynamic plans).
    pub fn topo_order(&self) -> Result<Vec<NodeId>, crate::framework::error::PlanError> {
        let n = self.nodes.len();
        if n == 0 {
            return Err(crate::framework::error::PlanError::Empty);
        }
        let mut indeg: Vec<u32> = vec![0; n];
        for e in &self.edges {
            indeg[e.to as usize] += 1;
        }
        // Adjacency: outgoing edges per node.
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
            // First node still with indegree > 0 names the cycle.
            let offender = (0..n as NodeId)
                .find(|&i| indeg[i as usize] > 0)
                .unwrap_or(0);
            return Err(crate::framework::error::PlanError::Cycle(offender));
        }
        Ok(order)
    }
}

/// Borrow-only access for the executor. Avoids exposing
/// `nodes` / `edges` / `initial` as `pub` while still letting the
/// executor walk them.
pub(crate) struct ExecView<'a> {
    pub name: &'a str,
    pub nodes: &'a [PlanNode],
    pub edges: &'a [PlanEdge],
    pub initial: &'a HashMap<NodeId, ErasedArtifact>,
    pub recipe_args: &'a serde_json::Value,
}

impl Plan<()> {
    pub(crate) fn exec_view(&self) -> ExecView<'_> {
        ExecView {
            name: &self.name,
            nodes: &self.nodes,
            edges: &self.edges,
            initial: &self.initial,
            recipe_args: &self.recipe_args,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;
    use crate::framework::resource::Resource;
    use crate::framework::error::StageError;
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

    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct EmptyArgs;

    struct MakeA;
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

    #[test]
    fn empty_plan_topo_errors() {
        let p: Plan<()> = Plan::new("empty", serde_json::json!({}));
        let r = p.topo_order();
        assert!(matches!(
            r,
            Err(crate::framework::error::PlanError::Empty)
        ));
    }

    #[test]
    fn linear_three_stage_plan_compiles_and_orders() {
        let plan = Plan::new("linear", serde_json::json!({}))
            .start(MakeA, EmptyArgs)
            .then(AToB, EmptyArgs)
            .then(BToC, EmptyArgs)
            .finish();
        assert_eq!(plan.n_nodes(), 3);
        assert_eq!(plan.n_edges(), 2);
        let order = plan.topo_order().unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn first_node_has_unit_initial_input() {
        let plan = Plan::new("with_unit", serde_json::json!({}))
            .start(MakeA, EmptyArgs)
            .finish();
        assert_eq!(plan.initial.len(), 1);
        let unit = plan.initial.get(&0).unwrap();
        assert_eq!(unit.kind, "()");
    }

    #[test]
    fn topo_order_visits_each_node_once() {
        let plan = Plan::new("p", serde_json::json!({}))
            .start(MakeA, EmptyArgs)
            .then(AToB, EmptyArgs)
            .then(BToC, EmptyArgs)
            .finish();
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
        let plan = Plan::new("forked", serde_json::json!({}))
            .start(MakeA, EmptyArgs)
            .fork(AToB, EmptyArgs, AToB, EmptyArgs)
            .finish();
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
        let plan = Plan::new("named", args.clone())
            .start(MakeA, EmptyArgs)
            .finish();
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
}
