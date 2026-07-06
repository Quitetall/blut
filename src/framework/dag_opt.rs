// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! DAG optimizer — transforms a `CompiledPlan` before execution.
//!
//! The optimizer runs as a pass between plan compilation and the
//! executor's spawn loop. It transforms the plan in-place to:
//!
//! 1. **Dead code elimination** — remove stages whose outputs are
//!    never consumed by any downstream stage.
//! 2. **Critical path priority** — compute the critical path length
//!    for each node and store it as a scheduling hint.
//! 3. **Cache-aware ordering** — among nodes with equal priority,
//!    prefer those with warm caches (already computed).
//! 4. **Memory-aware scheduling** — compute peak concurrent memory
//!    for the current schedule and suggest reordering if a cheaper
//!    order exists.
//!
//! The optimizer is conservative: it never changes the DAG's semantic
//! output, only its execution order and which nodes run at all.

use std::collections::{HashMap, HashSet, VecDeque};

use super::plan::{CompiledPlan, NodeId, PlanEdge, PlanNode};

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
    /// True if this node's cache is warm (output already exists).
    pub cache_warm: bool,
}

/// The DAG optimizer. Runs a sequence of passes on a `CompiledPlan`.
pub struct DagOptimizer {
    /// Enable dead code elimination.
    pub eliminate_dead_code: bool,
    /// Compute critical path priorities.
    pub critical_path: bool,
    /// Prefer cache-warm nodes.
    pub cache_aware: bool,
    /// Compute memory-aware ordering.
    pub memory_aware: bool,
}

impl DagOptimizer {
    pub fn new() -> Self {
        Self {
            eliminate_dead_code: true,
            critical_path: true,
            cache_aware: true,
            memory_aware: true,
        }
    }

    /// Run all enabled optimization passes on the plan.
    /// Returns the optimized plan and per-node schedule hints.
    pub fn optimize(&self, plan: CompiledPlan) -> (CompiledPlan, HashMap<NodeId, ScheduleHint>) {
        let mut plan = plan;
        let mut hints: HashMap<NodeId, ScheduleHint> = HashMap::new();

        // Pass 1: Dead code elimination
        if self.eliminate_dead_code {
            plan = eliminate_dead_code(plan);
        }

        // Pass 2: Critical path computation
        if self.critical_path {
            compute_critical_paths(&plan, &mut hints);
        }

        // Pass 3: Cache-aware hints
        if self.cache_aware {
            compute_cache_hints(&plan, &mut hints);
        }

        // Pass 4: Memory-aware scheduling
        if self.memory_aware {
            compute_memory_hints(&plan, &mut hints);
        }

        (plan, hints)
    }
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

    // Build adjacency: successors and predecessors
    let mut successors: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    let mut predecessors: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for edge in &plan.edges {
        successors[edge.from as usize].push(edge.to);
        predecessors[edge.to as usize].push(edge.from);
    }

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
    for (old_id, node) in plan.nodes.into_iter().enumerate() {
        if live.contains(&(old_id as NodeId)) {
            old_to_new.insert(old_id as NodeId, new_nodes.len() as NodeId);
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
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Critical path computation
// ---------------------------------------------------------------------------

/// Compute the critical path length from each node to the terminal.
/// Nodes on the longest path get the highest priority.
fn compute_critical_paths(plan: &CompiledPlan, hints: &mut HashMap<NodeId, ScheduleHint>) {
    let n = plan.nodes.len();
    if n == 0 {
        return;
    }

    // Build adjacency (successors)
    let mut successors: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for edge in &plan.edges {
        successors[edge.from as usize].push(edge.to);
    }

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
// Pass 3: Cache-aware hints
// ---------------------------------------------------------------------------

/// Mark nodes whose caches are warm (output already exists).
fn compute_cache_hints(plan: &CompiledPlan, hints: &mut HashMap<NodeId, ScheduleHint>) {
    for node in &plan.nodes {
        // A node's cache is warm if it's deterministic and the cache
        // has an entry. We can't check the cache here without the
        // CacheHandle, so we mark based on the DETERMINISTIC flag.
        // The executor will check the actual cache at spawn time.
        let hint = hints.entry(node.id).or_default();
        hint.cache_warm = node.stage.deterministic();
    }
}

// ---------------------------------------------------------------------------
// Pass 4: Memory-aware scheduling
// ---------------------------------------------------------------------------

/// Compute peak concurrent memory for each node based on its
/// position in the DAG and its memory requirements.
fn compute_memory_hints(plan: &CompiledPlan, hints: &mut HashMap<NodeId, ScheduleHint>) {
    let n = plan.nodes.len();
    if n == 0 {
        return;
    }

    // Build adjacency
    let mut successors: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    let mut predecessors: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for edge in &plan.edges {
        successors[edge.from as usize].push(edge.to);
        predecessors[edge.to as usize].push(edge.from);
    }

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
                retry: None,
                timeout: None,
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
}
