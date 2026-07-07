// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `PlanDraft` — the pure accumulator a `.star` script fills via `add()`
//! (ADR 0078). Kept free of any Starlark type so the wiring logic is unit-
//! testable on its own; the Starlark glue lives in `globals.rs`.

use serde_json::Value;

use blut::framework::plan_spec::{MapSpec, PLAN_SPEC_VERSION, PlanSpec, SpecNode};

/// Nodes + edges (+ map expansions) collected while a script runs. Each
/// `add()` appends one node and wires its `after` predecessors immediately,
/// so all edges into a node are contiguous and in `after` order — exactly the
/// tuple element order `CompiledPlan::from_erased_graph` expects for a merge.
#[derive(Debug, Default)]
pub(crate) struct PlanDraft {
    pub nodes: Vec<SpecNode>,
    pub edges: Vec<(u32, u32)>,
    /// Runtime `map_output` fan-outs collected via `map_output(...)`.
    pub expansions: Vec<MapSpec>,
}

impl PlanDraft {
    /// Append a node with `args`, wire an edge from each id in `after`, and
    /// return the new node's id (its dense index).
    pub fn add(&mut self, stage: String, args: Value, after: &[u32]) -> u32 {
        // Node ids are u32; the evaluator's tick cap bounds node count far
        // below u32::MAX, so this only fires on a corrupt invariant.
        debug_assert!(
            self.nodes.len() <= u32::MAX as usize,
            "plan node count exceeds u32"
        );
        let id = self.nodes.len() as u32;
        self.nodes.push(SpecNode { stage, args });
        for &p in after {
            self.edges.push((p, id));
        }
        id
    }

    /// Finalize into a `PlanSpec` (still to be `compile`d against a
    /// `Registry` — this is pure structure, no stage resolution yet).
    pub fn into_spec(self, name: String) -> PlanSpec {
        PlanSpec {
            name,
            nodes: self.nodes,
            edges: self.edges,
            expansions: self.expansions,
            version: PLAN_SPEC_VERSION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn add_returns_dense_ids_and_wires_after_in_order() {
        let mut d = PlanDraft::default();
        let a = d.add("make".into(), json!(null), &[]);
        let b = d.add("x".into(), json!({ "k": 1 }), &[a]);
        let c = d.add("merge".into(), json!(null), &[a, b]);
        assert_eq!((a, b, c), (0, 1, 2));
        // Merge edges are (a->c),(b->c) in exactly the `after` order.
        assert_eq!(d.edges, vec![(0, 1), (0, 2), (1, 2)]);
        let spec = d.into_spec("p".into());
        assert_eq!(spec.nodes.len(), 3);
        assert_eq!(spec.nodes[1].args, json!({ "k": 1 }));
        assert_eq!(spec.version, PLAN_SPEC_VERSION);
    }
}
