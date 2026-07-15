// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! DAG optimizer — transforms a `CompiledPlan` before execution.
//!
//! The optimizer runs as a pass between plan compilation and the
//! executor's spawn loop. It transforms the plan in-place to:
//!
//! 1. **Dead code elimination** — remove stages whose outputs are
//!    never consumed by any downstream stage.
//! 2. **Stage fusion** (ADR 0102) — record eligible maximal linear subchains
//!    as optimized-plan execution groups. The executor may conservatively
//!    refine these groups without changing semantic nodes or cache identity.
//! 3. **Critical path priority** — compute the critical path length
//!    for each node and store it as a scheduling hint.
//! 4. **Cache-aware ordering** (ADR 0102) — opt in to executor-side live cache
//!    probes once a node is ready and its exact input-dependent key is known.
//! 5. **Memory-aware scheduling** — compute peak concurrent memory
//!    for the current schedule and suggest reordering if a cheaper
//!    order exists.
//! 6. **User-priority scheduling** (ADR 0102 pass #4) — copy each node's
//!    optional `PlanNode::priority` into its hint so a latency-critical
//!    stage jumps bulk work in the executor's ready queue. Flag-gated
//!    (`priority_aware`, default off); reorders *ready* nodes only, never
//!    bypassing broker admission.
//!
//! The optimizer is conservative: it never changes the DAG's semantic
//! output, only its execution order and which nodes run at all.

use std::collections::{HashMap, HashSet, VecDeque};

use super::plan::{
    CompiledConditionGate, CompiledPlan, FusedSubchain, NodeId, PlanEdge, PlanNode,
    SpeculationCandidate,
};
use super::stage::StageExecutionBoundary;

/// Scheduling hints computed by the optimizer. Stored per-node and
/// read by the executor's spawn loop.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScheduleHint {
    /// Critical path length from this node to the terminal node.
    /// Higher = more urgent (should be scheduled first).
    pub critical_path_len: u32,
    /// Estimated peak memory (GiB) if this node and all its
    /// concurrent siblings run together.
    pub peak_concurrent_gib: u32,
    /// True only after the executor resolves this ready node's exact cache key
    /// against its live [`CacheHandle`](super::cache::CacheHandle).
    pub cache_warm: bool,
    /// ADR 0102 pass #4: user scheduling priority copied from the node's
    /// `PlanNode::priority` (0 when unset). Dominates `critical_path_len` in the
    /// executor's ready-queue ordering, so a latency-critical stage jumps bulk
    /// work — but only reorders *ready* nodes, never bypassing broker admission.
    /// Populated solely by the flag-gated `priority_aware` pass; default 0 keeps
    /// the ready-queue byte-identical to the pre-0102 `(critical_path_len, id)`.
    pub user_priority: i32,
}

/// The DAG optimizer. Runs a sequence of passes on a `CompiledPlan`.
pub struct DagOptimizer {
    /// Enable dead code elimination.
    pub eliminate_dead_code: bool,
    /// Compute critical path priorities.
    pub critical_path: bool,
    /// Prefer nodes with a live, decodable cache entry. **Off by default**;
    /// actual probing happens in the executor because only it owns resolved
    /// predecessor hashes and the run's tenant-scoped `CacheHandle`.
    pub cache_aware: bool,
    /// Compute memory-aware ordering.
    pub memory_aware: bool,
    /// ADR 0102 pass #4: honour per-node user `priority` in ready-queue ordering.
    /// **Off by default** — when off, no hint gets a non-zero `user_priority`, so
    /// the executor's ready-queue is byte-identical to the pre-0102 behaviour
    /// (each advanced 0102 pass is flag-gated, default-off, per the ADR).
    pub priority_aware: bool,
    /// ADR 0102 stage fusion. **Off by default.** The optimizer records maximal
    /// eligible deterministic linear subchains on the compiled plan; the
    /// executor may coalesce conservative runtime subsets into one task while
    /// retaining every semantic node's normal artifact and cache key.
    pub stage_fusion: bool,
    /// ADR 0102 speculative execution. **Off by default.** The optimizer emits
    /// candidates only for conditional targets with both node-requested purity
    /// and independent deterministic/speculation-safe stage certification.
    pub speculative_execution: bool,
}

impl DagOptimizer {
    pub fn new() -> Self {
        Self {
            eliminate_dead_code: true,
            critical_path: true,
            cache_aware: false,
            memory_aware: true,
            priority_aware: false,
            stage_fusion: false,
            speculative_execution: false,
        }
    }

    /// Run all enabled optimization passes on the plan.
    /// Returns the optimized plan and per-node schedule hints.
    pub fn optimize(&self, plan: CompiledPlan) -> (CompiledPlan, HashMap<NodeId, ScheduleHint>) {
        let mut plan = plan;
        // Optimization witnesses always belong to this exact post-DCE id
        // space. Clear a prior run before any transform can renumber nodes.
        plan.fused_subchains.clear();
        plan.speculation_candidates.clear();
        let mut hints: HashMap<NodeId, ScheduleHint> = HashMap::new();

        // Pass 1: Dead code elimination
        if self.eliminate_dead_code {
            plan = eliminate_dead_code(plan);
        }

        // Pass 2 (ADR 0102): emit maximal structural fusion candidates onto
        // the optimized plan. Runtime policy may conservatively split or skip
        // them for admission, cache single-flight, control, or placement, but
        // it may never fuse outside this Plan -> Plan witness. Cache-aware and
        // explicit user-priority scheduling retain per-node boundaries: an
        // atomic group would otherwise hide a ready-queue decision.
        if self.stage_fusion && !self.cache_aware && !self.priority_aware {
            plan.fused_subchains = find_fused_subchains(&plan);
        }

        // Pass 2b (ADR 0102): the optimizer is the sole owner of speculative
        // eligibility. The executor may decline these post-DCE candidates for
        // capacity, placement, or control reasons; it may never invent one.
        if self.speculative_execution {
            plan.speculation_candidates = find_speculation_candidates(&plan);
        }

        // Pass 3: Critical path computation
        if self.critical_path {
            compute_critical_paths(&plan, &mut hints);
        }

        // Pass 4 (ADR 0102): cache-aware ordering is resolved later by the
        // executor. At optimizer time downstream input hashes are not known and
        // no tenant-scoped CacheHandle exists, so any static "warm" claim would
        // be a determinism proxy rather than cache evidence.

        // Pass 5: Memory-aware scheduling
        if self.memory_aware {
            compute_memory_hints(&plan, &mut hints);
        }

        // Pass 6 (ADR 0102 pass #4): user-priority scheduling. Flag-gated /
        // default-off, so when disabled every hint keeps `user_priority = 0`
        // and the ready-queue order is unchanged.
        if self.priority_aware {
            compute_priority_hints(&plan, &mut hints);
        }

        (plan, hints)
    }
}

// ---------------------------------------------------------------------------
// Pass 2 (ADR 0102): static stage-fusion representation
// ---------------------------------------------------------------------------

fn fusion_candidate(node: &PlanNode) -> bool {
    node.stage.deterministic()
        && !node.stage.is_advisory()
        && node.stage.execution_boundary() == StageExecutionBoundary::InProcess
        && node.stage.supports_in_process_handoff()
}

/// Record maximal linear subchains in the post-DCE node-id space. Semantic
/// nodes are deliberately retained: their normal keys, artifacts, events, and
/// lineage are the equivalence witness. The executor may only refine these
/// candidates; it does not rediscover topology independently.
fn find_fused_subchains(plan: &CompiledPlan) -> Vec<FusedSubchain> {
    // A condition gate is a semantic scheduling boundary: the selector must
    // complete before the target can become ready and the target may never
    // execute. Until fused execution can preserve that boundary explicitly,
    // fail closed instead of letting a data-only subchain witness cross or
    // obscure conditional control flow.
    if !plan.expansions.is_empty() || !plan.condition_gates.is_empty() {
        return Vec::new();
    }
    let Ok(order) = plan.topo_order() else {
        return Vec::new();
    };
    let n = plan.nodes.len();
    let mut preds = vec![Vec::<NodeId>::new(); n];
    let mut succs = vec![Vec::<NodeId>::new(); n];
    for edge in &plan.edges {
        preds[edge.to as usize].push(edge.from);
        succs[edge.from as usize].push(edge.to);
    }

    let mut claimed = HashSet::new();
    let mut groups = Vec::new();
    for start in order {
        if claimed.contains(&start) || !fusion_candidate(&plan.nodes[start as usize]) {
            continue;
        }
        let joins_predecessor = if preds[start as usize].len() == 1 {
            let predecessor = preds[start as usize][0];
            succs[predecessor as usize].len() == 1
                && fusion_candidate(&plan.nodes[predecessor as usize])
        } else {
            false
        };
        if joins_predecessor {
            continue;
        }

        let mut members = vec![start];
        let mut current = start;
        while succs[current as usize].len() == 1 {
            let next = succs[current as usize][0];
            if preds[next as usize].len() != 1 || !fusion_candidate(&plan.nodes[next as usize]) {
                break;
            }
            members.push(next);
            current = next;
        }
        if members.len() >= 2 {
            claimed.extend(members.iter().copied());
            groups.push(FusedSubchain { members });
        }
    }
    groups
}

fn find_speculation_candidates(plan: &CompiledPlan) -> Vec<SpeculationCandidate> {
    if !plan.expansions.is_empty() {
        return Vec::new();
    }
    plan.condition_gates
        .iter()
        .filter_map(|gate| {
            let target = plan.nodes.get(gate.target as usize)?;
            (target.pure && target.stage.deterministic() && target.stage.speculation_safe())
                .then_some(SpeculationCandidate {
                    target: gate.target,
                })
        })
        .collect()
}

impl Default for DagOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pass 1: Dead code elimination
// ---------------------------------------------------------------------------

/// Remove stages that are unreachable from any root (node with no
/// predecessors). A disconnected node with no incoming edges is dead
/// code regardless of whether it has successors.
///
/// A node is "live" if it's reachable from at least one root via
/// forward edges. This catches both fully disconnected nodes and
/// subgraphs that nothing feeds into.
fn eliminate_dead_code(plan: CompiledPlan) -> CompiledPlan {
    let n = plan.nodes.len();
    if n == 0 {
        return plan;
    }
    // A plan with runtime `map_output` expansions (ADR 0078) is skipped: a map
    // parent's output can drive a fan-out with NO downstream plan edge, so it
    // would look "dead" here, and remapping node ids would stale the
    // expansions' `parent`. Forgoing DCE on a dynamic plan is safe.
    if !plan.expansions.is_empty() {
        return plan;
    }

    // Build the executable topology. Condition gates are control arcs rather
    // than artifact-carrying `PlanEdge`s, but they still participate in
    // reachability and root discovery: dropping a selector while retaining its
    // gated target would leave stale metadata and change execution semantics.
    let (successors, predecessors) = plan_adjacency(&plan);

    // Find roots (no predecessors) that have at least one successor.
    // A root with no successors and no predecessors is a disconnected
    // node — dead code. Exception: single-node plans are always live.
    let roots: Vec<NodeId> = if n == 1 {
        vec![0]
    } else {
        (0..n as NodeId)
            .filter(|&id| {
                predecessors[id as usize].is_empty() && !successors[id as usize].is_empty()
            })
            .collect()
    };

    // If there are no live roots, the plan is all disconnected nodes
    // — return as-is.
    if roots.is_empty() {
        return plan;
    }

    // Forward BFS from roots
    let mut live: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeId> = roots.into_iter().collect();
    while let Some(node_id) = queue.pop_front() {
        if !live.insert(node_id) {
            continue;
        }
        for &succ in &successors[node_id as usize] {
            if !live.contains(&succ) {
                queue.push_back(succ);
            }
        }
    }

    // If all nodes are live, nothing to do
    if live.len() == n {
        return plan;
    }

    // Build old→new index mapping
    let mut old_to_new: HashMap<NodeId, NodeId> = HashMap::new();
    let mut new_nodes: Vec<PlanNode> = Vec::new();
    for (old_id, mut node) in plan.nodes.into_iter().enumerate() {
        if live.contains(&(old_id as NodeId)) {
            let new_id = new_nodes.len() as NodeId;
            old_to_new.insert(old_id as NodeId, new_id);
            node.id = new_id;
            new_nodes.push(node);
        }
    }

    // Remap edges
    let new_edges: Vec<PlanEdge> = plan
        .edges
        .into_iter()
        .filter(|e| live.contains(&e.from) && live.contains(&e.to))
        .map(|e| PlanEdge {
            from: old_to_new[&e.from],
            to: old_to_new[&e.to],
        })
        .collect();

    // Remap initial artifacts
    let new_initial: HashMap<NodeId, _> = plan
        .initial
        .into_iter()
        .filter(|(id, _)| live.contains(id))
        .map(|(id, art)| (old_to_new[&id], art))
        .collect();

    // A gate's endpoints are connected by the control arc included in the
    // liveness walk above, so a live gate is remapped atomically with both
    // endpoint nodes. Filtering remains fail-safe for internally-constructed
    // plans that bypass normal PlanSpec validation.
    let new_condition_gates: Vec<CompiledConditionGate> = plan
        .condition_gates
        .into_iter()
        .filter(|gate| live.contains(&gate.condition) && live.contains(&gate.target))
        .map(|gate| CompiledConditionGate {
            condition: old_to_new[&gate.condition],
            target: old_to_new[&gate.target],
            when: gate.when,
        })
        .collect();

    let eliminated = n - new_nodes.len();
    if eliminated > 0 {
        tracing::info!(
            "DAG optimizer: eliminated {eliminated} dead nodes ({}→{})",
            n,
            new_nodes.len()
        );
    }

    CompiledPlan {
        name: plan.name,
        nodes: new_nodes,
        edges: new_edges,
        initial: new_initial,
        recipe_args: plan.recipe_args,
        // Unreachable with expansions (early-returned above); always empty here.
        expansions: Vec::new(),
        fused_subchains: Vec::new(),
        speculation_candidates: Vec::new(),
        condition_gates: new_condition_gates,
    }
}

/// Build adjacency for the executable DAG. Data edges carry artifacts;
/// condition gates carry control only, but both constrain execution order.
fn plan_adjacency(plan: &CompiledPlan) -> (Vec<Vec<NodeId>>, Vec<Vec<NodeId>>) {
    let n = plan.nodes.len();
    let mut successors = vec![Vec::new(); n];
    let mut predecessors = vec![Vec::new(); n];

    for edge in &plan.edges {
        successors[edge.from as usize].push(edge.to);
        predecessors[edge.to as usize].push(edge.from);
    }
    for gate in &plan.condition_gates {
        successors[gate.condition as usize].push(gate.target);
        predecessors[gate.target as usize].push(gate.condition);
    }

    (successors, predecessors)
}

// ---------------------------------------------------------------------------
// Pass 3: Critical path computation
// ---------------------------------------------------------------------------

/// Compute the critical path length from each node to the terminal.
/// Nodes on the longest path get the highest priority.
fn compute_critical_paths(plan: &CompiledPlan, hints: &mut HashMap<NodeId, ScheduleHint>) {
    let n = plan.nodes.len();
    if n == 0 {
        return;
    }

    // Control gates constrain the executable topology just like data edges.
    let (successors, _) = plan_adjacency(plan);

    // Dynamic programming: critical_path[node] = 1 + max(critical_path[successors])
    // Process in reverse topo order
    let topo = topo_order_from_adj(&successors, n);
    let mut cp_len: Vec<u32> = vec![0; n];

    for &node_id in topo.iter().rev() {
        let idx = node_id as usize;
        if successors[idx].is_empty() {
            cp_len[idx] = 1;
        } else {
            let max_succ = successors[idx]
                .iter()
                .map(|&s| cp_len[s as usize])
                .max()
                .unwrap_or(0);
            cp_len[idx] = 1 + max_succ;
        }
    }

    // Store in hints
    for (i, &len) in cp_len.iter().enumerate() {
        hints.entry(i as NodeId).or_default().critical_path_len = len;
    }
}

// ---------------------------------------------------------------------------
// Pass 6 (ADR 0102 pass #4): user-priority scheduling
// ---------------------------------------------------------------------------

/// Copy each node's user `priority` (from its `PlanNode`, a PlanSpec v1.1
/// additive field) into its schedule hint. Higher priority runs earlier among
/// *ready* nodes; `None`/unset stays 0 (neutral). This is the ONLY writer of
/// `user_priority`, so with the `priority_aware` flag off every hint keeps 0 and
/// the ready-queue is byte-identical to the pre-0102 order — reordering ready
/// nodes is sound by the DCE↔hint invariant (ADR 0067) and never touches a
/// node's output artifact hash (scheduling metadata, not computation, ADR 0078).
fn compute_priority_hints(plan: &CompiledPlan, hints: &mut HashMap<NodeId, ScheduleHint>) {
    for node in &plan.nodes {
        if let Some(p) = node.priority {
            hints.entry(node.id).or_default().user_priority = p;
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 5: Memory-aware scheduling
// ---------------------------------------------------------------------------

/// Compute peak concurrent memory for each node based on its
/// position in the DAG and its memory requirements.
fn compute_memory_hints(plan: &CompiledPlan, hints: &mut HashMap<NodeId, ScheduleHint>) {
    let n = plan.nodes.len();
    if n == 0 {
        return;
    }

    // Control gates constrain the executable topology just like data edges.
    let (successors, predecessors) = plan_adjacency(plan);

    // For each node, estimate peak concurrent memory as the sum of
    // memory_gib for all nodes that could run simultaneously.
    // This is a conservative estimate: nodes at the same topo level
    // could all run concurrently.
    let topo = topo_order_from_adj(&successors, n);
    let mut levels: Vec<u32> = vec![0; n]; // topo level per node

    for &node_id in &topo {
        let idx = node_id as usize;
        let max_pred_level = predecessors[idx]
            .iter()
            .map(|&p| levels[p as usize])
            .max()
            .unwrap_or(0);
        levels[idx] = if predecessors[idx].is_empty() {
            0
        } else {
            max_pred_level + 1
        };
    }

    // Group nodes by level
    let max_level = levels.iter().copied().max().unwrap_or(0);
    for level in 0..=max_level {
        let nodes_at_level: Vec<NodeId> = (0..n as NodeId)
            .filter(|&id| levels[id as usize] == level)
            .collect();
        let total_mem: u32 = nodes_at_level
            .iter()
            .map(|&id| plan.nodes[id as usize].stage.memory_gib())
            .sum();
        for &node_id in &nodes_at_level {
            hints.entry(node_id).or_default().peak_concurrent_gib = total_mem;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute topological order from adjacency list.
fn topo_order_from_adj(successors: &[Vec<NodeId>], n: usize) -> Vec<NodeId> {
    let mut in_degree: Vec<u32> = vec![0; n];
    for succs in successors {
        for &s in succs {
            in_degree[s as usize] += 1;
        }
    }

    let mut queue: VecDeque<NodeId> = (0..n as NodeId)
        .filter(|&id| in_degree[id as usize] == 0)
        .collect();

    let mut order: Vec<NodeId> = Vec::with_capacity(n);
    while let Some(node_id) = queue.pop_front() {
        order.push(node_id);
        for &succ in &successors[node_id as usize] {
            in_degree[succ as usize] -= 1;
            if in_degree[succ as usize] == 0 {
                queue.push_back(succ);
            }
        }
    }

    order
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::stage::StageDyn;
    use std::sync::Arc;

    /// Helper: create a minimal CompiledPlan for testing.
    fn make_plan(n_nodes: usize, edges: &[(NodeId, NodeId)]) -> CompiledPlan {
        use crate::framework::stage::Stage;
        use async_trait::async_trait;

        // A dummy stage for testing
        struct DummyStage;

        #[async_trait]
        impl Stage for DummyStage {
            const NAME: &'static str = "dummy";
            const SCHEMA: u32 = 1;
            const RESOURCES: &'static [crate::framework::resource::Resource] = &[];
            const MEMORY_GIB: u32 = 4; // matches the test expectation
            const DETERMINISTIC: bool = true;
            const SPECULATION_SAFE: bool = true;
            type Input = ();
            type Output = ();
            type Args = ();

            async fn run(
                &self,
                _ctx: &crate::framework::stage::StageContext,
                _input: (),
                _args: &(),
            ) -> Result<(), crate::framework::error::StageError> {
                Ok(())
            }
        }

        let nodes: Vec<PlanNode> = (0..n_nodes)
            .map(|i| PlanNode {
                id: i as NodeId,
                stage: Arc::new(DummyStage) as Arc<dyn StageDyn>,
                args: serde_json::Value::Null,
                canon_args: Vec::new(),
                admission_recipe_args: None,
                admission_scope_args: None,
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
                partition: None,
            })
            .collect();

        let edges: Vec<PlanEdge> = edges
            .iter()
            .map(|&(from, to)| PlanEdge { from, to })
            .collect();

        CompiledPlan {
            name: "test".into(),
            nodes,
            edges,
            initial: HashMap::new(),
            recipe_args: serde_json::Value::Null,
            expansions: Vec::new(),
            fused_subchains: Vec::new(),
            speculation_candidates: Vec::new(),
            condition_gates: Vec::new(),
        }
    }

    #[test]
    fn dead_code_elim_removes_unreachable() {
        // 0 → 1 → 2 (terminal), 3 is unreachable
        let plan = make_plan(4, &[(0, 1), (1, 2)]);
        let opt = DagOptimizer::new();
        let (optimized, _) = opt.optimize(plan);
        assert_eq!(optimized.n_nodes(), 3, "node 3 should be eliminated");
        assert_eq!(optimized.n_edges(), 2);
    }

    #[test]
    fn dead_code_elim_preserves_all_live() {
        // 0 → 1 → 2, all connected
        let plan = make_plan(3, &[(0, 1), (1, 2)]);
        let opt = DagOptimizer::new();
        let (optimized, _) = opt.optimize(plan);
        assert_eq!(optimized.n_nodes(), 3);
    }

    #[test]
    fn dead_code_elim_handles_diamond() {
        // 0 → 1, 0 → 2, 1 → 3, 2 → 3
        let plan = make_plan(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let opt = DagOptimizer::new();
        let (optimized, _) = opt.optimize(plan);
        assert_eq!(optimized.n_nodes(), 4, "diamond should be preserved");
    }

    #[test]
    fn dead_code_elim_remaps_condition_control_arc() {
        // Nodes 0 and 4 are disconnected. The only path keeping selector 1
        // live is its control gate to data chain 2 -> 3.
        let mut plan = make_plan(5, &[(2, 3)]);
        plan.condition_gates.push(CompiledConditionGate {
            condition: 1,
            target: 2,
            when: true,
        });

        let (optimized, _) = DagOptimizer::new().optimize(plan);

        assert_eq!(optimized.n_nodes(), 3);
        assert_eq!(optimized.n_edges(), 1);
        assert_eq!(optimized.edges[0].from, 1);
        assert_eq!(optimized.edges[0].to, 2);
        assert_eq!(optimized.condition_gates.len(), 1);
        assert_eq!(optimized.condition_gates[0].condition, 0);
        assert_eq!(optimized.condition_gates[0].target, 1);
        assert!(optimized.condition_gates[0].when);
    }

    #[test]
    fn fusion_is_disabled_for_conditional_plan() {
        let mut plan = make_plan(3, &[(0, 1)]);
        plan.condition_gates.push(CompiledConditionGate {
            condition: 2,
            target: 0,
            when: true,
        });
        let opt = DagOptimizer {
            eliminate_dead_code: false,
            critical_path: false,
            cache_aware: false,
            memory_aware: false,
            priority_aware: false,
            stage_fusion: true,
            speculative_execution: false,
        };

        let (optimized, _) = opt.optimize(plan);

        assert!(optimized.fused_subchains.is_empty());
    }

    #[test]
    fn speculation_witness_is_default_off_and_requires_node_purity() {
        let mut plan = make_plan(2, &[]);
        plan.condition_gates.push(CompiledConditionGate {
            condition: 0,
            target: 1,
            when: true,
        });
        plan.nodes[1].pure = true;

        let (default_off, _) = DagOptimizer::new().optimize(plan);
        assert!(default_off.speculation_candidates().next().is_none());

        let mut impure = make_plan(2, &[]);
        impure.condition_gates.push(CompiledConditionGate {
            condition: 0,
            target: 1,
            when: true,
        });
        let optimizer = DagOptimizer {
            speculative_execution: true,
            ..DagOptimizer::new()
        };
        let (impure, _) = optimizer.optimize(impure);
        assert!(impure.speculation_candidates().next().is_none());

        struct UnsafeStage;
        #[async_trait::async_trait]
        impl crate::framework::stage::Stage for UnsafeStage {
            const NAME: &'static str = "unsafe_speculation_dummy";
            const SCHEMA: u32 = 1;
            const RESOURCES: &'static [crate::framework::resource::Resource] = &[];
            type Input = ();
            type Output = ();
            type Args = ();

            async fn run(
                &self,
                _ctx: &crate::framework::stage::StageContext,
                _input: (),
                _args: &(),
            ) -> Result<(), crate::framework::error::StageError> {
                Ok(())
            }
        }
        let mut unsafe_stage = make_plan(2, &[]);
        unsafe_stage.nodes[1].pure = true;
        unsafe_stage.nodes[1].stage = Arc::new(UnsafeStage);
        unsafe_stage.condition_gates.push(CompiledConditionGate {
            condition: 0,
            target: 1,
            when: true,
        });
        let optimizer = DagOptimizer {
            speculative_execution: true,
            ..DagOptimizer::new()
        };
        let (unsafe_stage, _) = optimizer.optimize(unsafe_stage);
        assert!(
            unsafe_stage.speculation_candidates().next().is_none(),
            "node-requested purity cannot replace stage-owned safety certification"
        );
    }

    #[test]
    fn speculation_witness_uses_post_dce_node_ids() {
        // Disconnected nodes 0 and 4 disappear. Selector 1 and pure target 2
        // remain linked by control, and data target 2 -> tail 3 remains live.
        let mut plan = make_plan(5, &[(2, 3)]);
        plan.nodes[2].pure = true;
        plan.condition_gates.push(CompiledConditionGate {
            condition: 1,
            target: 2,
            when: true,
        });
        let optimizer = DagOptimizer {
            speculative_execution: true,
            ..DagOptimizer::new()
        };

        let (optimized, _) = optimizer.optimize(plan);

        assert_eq!(
            optimized.speculation_candidates().collect::<Vec<_>>(),
            vec![1],
            "the optimizer alone must publish the candidate in post-DCE id space"
        );
    }

    #[test]
    fn critical_path_longest_chain() {
        // 0 → 1 → 2 → 3 (linear chain)
        let plan = make_plan(4, &[(0, 1), (1, 2), (2, 3)]);
        let opt = DagOptimizer {
            eliminate_dead_code: false,
            ..DagOptimizer::new()
        };
        let (_, hints) = opt.optimize(plan);
        assert_eq!(hints[&0].critical_path_len, 4);
        assert_eq!(hints[&1].critical_path_len, 3);
        assert_eq!(hints[&2].critical_path_len, 2);
        assert_eq!(hints[&3].critical_path_len, 1);
    }

    #[test]
    fn critical_path_diamond() {
        // 0 → 1, 0 → 2, 1 → 3, 2 → 3
        // Both paths have length 3, so critical_path_len should be 3 for all
        let plan = make_plan(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let opt = DagOptimizer {
            eliminate_dead_code: false,
            ..DagOptimizer::new()
        };
        let (_, hints) = opt.optimize(plan);
        assert_eq!(hints[&0].critical_path_len, 3);
        assert_eq!(hints[&1].critical_path_len, 2);
        assert_eq!(hints[&2].critical_path_len, 2);
        assert_eq!(hints[&3].critical_path_len, 1);
    }

    #[test]
    fn memory_hints_group_by_level() {
        // 0 → 1, 0 → 2, 1 → 3, 2 → 3
        // Level 0: {0}, Level 1: {1, 2}, Level 2: {3}
        // Each node has memory_gib=4, so level 1 should have peak=8
        let plan = make_plan(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let opt = DagOptimizer {
            eliminate_dead_code: false,
            ..DagOptimizer::new()
        };
        let (_, hints) = opt.optimize(plan);
        assert_eq!(hints[&0].peak_concurrent_gib, 4); // level 0: 1 node
        assert_eq!(hints[&1].peak_concurrent_gib, 8); // level 1: 2 nodes
        assert_eq!(hints[&2].peak_concurrent_gib, 8); // level 1: 2 nodes
        assert_eq!(hints[&3].peak_concurrent_gib, 4); // level 2: 1 node
    }

    // --- ADR 0102 pass #4: user-priority scheduling ------------------------

    /// Optimizer that runs ONLY the priority pass (isolate it from DCE/critical-
    /// path/cache/memory so the assertions are unambiguous).
    fn priority_only_opt() -> DagOptimizer {
        DagOptimizer {
            eliminate_dead_code: false,
            critical_path: false,
            cache_aware: false,
            memory_aware: false,
            priority_aware: true,
            stage_fusion: false,
            speculative_execution: false,
        }
    }

    #[test]
    fn dag_opt_priority_pass_sets_user_priority() {
        // The priority pass copies each node's `PlanNode::priority` into the hint;
        // unset nodes stay at the neutral 0.
        let mut plan = make_plan(3, &[(0, 1), (1, 2)]);
        plan.nodes[2].priority = Some(50);
        plan.nodes[0].priority = Some(-5); // negative = de-prioritise
        let (_plan, hints) = priority_only_opt().optimize(plan);
        assert_eq!(hints[&2].user_priority, 50);
        assert_eq!(hints[&0].user_priority, -5);
        assert_eq!(
            hints.get(&1).map(|h| h.user_priority).unwrap_or(0),
            0,
            "an unset node stays neutral"
        );
    }

    #[test]
    fn dag_opt_priority_off_is_byte_identical() {
        // With `priority_aware` OFF (the default), even a node carrying an explicit
        // priority yields NO non-zero `user_priority`, so the ready-queue key is
        // byte-identical to the pre-0102 `(critical_path_len, id)`.
        let mut plan = make_plan(2, &[(0, 1)]);
        plan.nodes[1].priority = Some(99);
        let opt = DagOptimizer::new(); // priority_aware defaults to false
        let (_plan, hints) = opt.optimize(plan);
        assert!(
            hints.values().all(|h| h.user_priority == 0),
            "no hint may carry priority when the pass is off"
        );
    }

    #[test]
    fn dag_opt_priority_preserves_node_set() {
        // Equivalence witness: priority is scheduling metadata, so running the
        // pass leaves the node set, ids, args (== per-node cache-key inputs), and
        // edges byte-identical — the optimized plan produces the same artifact
        // hashes as the un-optimized plan (ADR 0078 preserved).
        let mut plan = make_plan(3, &[(0, 1), (1, 2)]);
        plan.nodes[1].priority = Some(7);
        let before_ids: Vec<_> = plan.nodes.iter().map(|n| n.id).collect();
        let before_args: Vec<_> = plan.nodes.iter().map(|n| n.args.clone()).collect();
        let before_canon: Vec<_> = plan.nodes.iter().map(|n| n.canon_args.clone()).collect();
        let before_edges: Vec<_> = plan.edges.iter().map(|e| (e.from, e.to)).collect();
        let (after, _hints) = priority_only_opt().optimize(plan);
        assert_eq!(
            after.nodes.iter().map(|n| n.id).collect::<Vec<_>>(),
            before_ids
        );
        assert_eq!(
            after
                .nodes
                .iter()
                .map(|n| n.args.clone())
                .collect::<Vec<_>>(),
            before_args,
            "args (a cache-key input) must be untouched"
        );
        assert_eq!(
            after
                .nodes
                .iter()
                .map(|n| n.canon_args.clone())
                .collect::<Vec<_>>(),
            before_canon,
            "canonical cache-key bytes must be untouched"
        );
        assert_eq!(
            after
                .edges
                .iter()
                .map(|e| (e.from, e.to))
                .collect::<Vec<_>>(),
            before_edges
        );
    }
}
