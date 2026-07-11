// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Read-side lineage reports (ADR 0099) — the provenance graph over the
//! content-addressed lineage DB. Pure typed rows + a DOT/JSON export (charter:
//! no server; rendering is delegated to a sidecar). Clinical/PHI-tenant rows are
//! fail-closed excluded from any export (ADR 0061).

use serde::{Deserialize, Serialize};

/// One artifact node in a provenance graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNode {
    /// The artifact's content hash (lowercased) — its stable identity.
    pub content_hash: String,
    /// The stage that produced it, if the artifact is indexed.
    pub stage_name: Option<String>,
    /// The artifact kind, if indexed.
    pub kind: Option<String>,
    /// The job that produced it, if indexed.
    pub job_id: Option<String>,
}

/// The transitive upstream provenance DAG of an artifact: nodes + `(from → to)`
/// edges (an edge means `from` was an input that produced `to`). Node/edge order
/// is sorted so the DOT/JSON export is byte-stable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceGraph {
    pub root: String,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<(String, String)>,
}

impl ProvenanceGraph {
    /// Short 12-hex label for a content hash (commit-hash convention).
    fn short(h: &str) -> &str {
        h.get(..12).unwrap_or(h)
    }

    /// A Graphviz DOT export — deterministic (nodes/edges are pre-sorted). The
    /// root is highlighted. Feed to `dot -Tsvg` or a sidecar renderer.
    pub fn to_dot(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::from("digraph provenance {\n  rankdir=LR;\n");
        for n in &self.nodes {
            let label = match &n.stage_name {
                Some(stage) => format!("{}\\n{}", stage, Self::short(&n.content_hash)),
                None => Self::short(&n.content_hash).to_string(),
            };
            let shape = if n.content_hash == self.root {
                ", style=filled, fillcolor=lightblue"
            } else {
                ""
            };
            let _ = writeln!(s, "  \"{}\" [label=\"{label}\"{shape}];", n.content_hash);
        }
        for (from, to) in &self.edges {
            let _ = writeln!(s, "  \"{from}\" -> \"{to}\";");
        }
        s.push_str("}\n");
        s
    }

    /// The set of source artifacts (no inbound edge within the graph) — the
    /// transitive INPUTS the root was built from.
    pub fn sources(&self) -> Vec<&str> {
        let has_input: std::collections::HashSet<&str> =
            self.edges.iter().map(|(_, to)| to.as_str()).collect();
        let mut srcs: Vec<&str> = self
            .nodes
            .iter()
            .map(|n| n.content_hash.as_str())
            .filter(|h| !has_input.contains(h))
            .collect();
        srcs.sort_unstable();
        srcs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(h: &str, stage: &str) -> GraphNode {
        GraphNode {
            content_hash: h.to_string(),
            stage_name: Some(stage.to_string()),
            kind: Some("k".into()),
            job_id: Some("j".into()),
        }
    }

    #[test]
    fn dot_is_deterministic_and_marks_root() {
        let g = ProvenanceGraph {
            root: "bb".into(),
            nodes: vec![node("aa", "src"), node("bb", "train")],
            edges: vec![("aa".into(), "bb".into())],
        };
        let d1 = g.to_dot();
        let d2 = g.to_dot();
        assert_eq!(d1, d2, "DOT export is byte-stable");
        assert!(d1.contains("\"aa\" -> \"bb\""));
        assert!(d1.contains("fillcolor=lightblue")); // root highlighted
    }

    #[test]
    fn sources_are_inputless_nodes() {
        let g = ProvenanceGraph {
            root: "cc".into(),
            nodes: vec![node("aa", "src"), node("bb", "mid"), node("cc", "leaf")],
            edges: vec![("aa".into(), "bb".into()), ("bb".into(), "cc".into())],
        };
        assert_eq!(g.sources(), vec!["aa"]);
    }
}
