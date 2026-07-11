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

    /// Escape a string for a DOT double-quoted label — defense-in-depth so a
    /// stage name containing `"`/`\`/newline can't break out of the label and
    /// inject DOT attributes into an export.
    fn dot_escape(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    /// A Graphviz DOT export — deterministic (nodes/edges are pre-sorted). The
    /// root is highlighted. Feed to `dot -Tsvg` or a sidecar renderer.
    pub fn to_dot(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::from("digraph provenance {\n  rankdir=LR;\n");
        for n in &self.nodes {
            let label = match &n.stage_name {
                Some(stage) => {
                    format!(
                        "{}\\n{}",
                        Self::dot_escape(stage),
                        Self::short(&n.content_hash)
                    )
                }
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

// ── model card (ADR 0099 capability 2) ─────────────────────────────

/// The content-addressed part of a [`ModelCard`] — everything the `card_hash`
/// digests. Kept separate so the hash is over the CONTENT, not itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CardContent {
    /// The model artifact this card describes.
    pub model_hash: String,
    /// The producing run, if indexed.
    pub job_id: Option<String>,
    /// The recipe that produced it (args identity, part 1).
    pub recipe: Option<String>,
    /// The run's config fingerprint (args + input identity, part 2).
    pub config_fingerprint: Option<String>,
    /// The acceptance/gate outcome recorded for the run.
    pub gate_outcome: Option<String>,
    /// The transitive DATA source artifacts (sorted, content hashes).
    pub data_sources: Vec<String>,
    /// The run's headline metrics (sorted by name).
    pub metrics: Vec<(String, f64)>,
}

impl CardContent {
    /// Deterministic content hash over the canonical JSON of this content —
    /// re-building a card from the same lineage rows yields a byte-identical
    /// `card_hash` (ADR 0099 determinism requirement). Field order is stable
    /// (serde struct order) and every vector is pre-sorted by the builder.
    pub fn content_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(self).expect("CardContent serializes");
        let mut h = Sha256::new();
        h.update(b"blut.model-card.v1");
        h.update(&bytes);
        faster_hex::hex_string(&h.finalize())
    }
}

/// A deterministic, content-addressed model card (ADR 0099): one model's data
/// sources, args, metrics, and gate outcome, plus the `card_hash` that
/// identifies exactly this collation. Clinical/restricted data sources are
/// excluded upstream (the card is an export).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    #[serde(flatten)]
    pub content: CardContent,
    /// Content address of `content` ONLY (never of the whole envelope) — stable
    /// across rebuilds on the same rows. A verifier recomputes
    /// [`CardContent::content_hash`] and compares; it must NOT re-hash the
    /// flattened JSON (which also carries this field). See [`ModelCard::verify`].
    pub card_hash: String,
}

impl ModelCard {
    pub fn new(content: CardContent) -> Self {
        let card_hash = content.content_hash();
        Self { content, card_hash }
    }

    /// Verify the card's content address: recompute the hash over `content` and
    /// compare to the stored `card_hash`. `false` ⇒ the card was tampered with
    /// or built by an incompatible version.
    pub fn verify(&self) -> bool {
        self.content.content_hash() == self.card_hash
    }

    /// Render a human-readable card (the CLI's default, non-JSON output).
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let c = &self.content;
        let mut s = String::new();
        let _ = writeln!(s, "model     : {}", c.model_hash);
        let _ = writeln!(s, "card_hash : {}", self.card_hash);
        if let Some(j) = &c.job_id {
            let _ = writeln!(s, "run       : {j}");
        }
        let _ = writeln!(s, "recipe    : {}", c.recipe.as_deref().unwrap_or("-"));
        let _ = writeln!(
            s,
            "config_fp : {}",
            c.config_fingerprint.as_deref().unwrap_or("-")
        );
        let _ = writeln!(
            s,
            "gate      : {}",
            c.gate_outcome.as_deref().unwrap_or("-")
        );
        let _ = writeln!(s, "data ({}):", c.data_sources.len());
        for d in &c.data_sources {
            let _ = writeln!(s, "  {}", d.get(..12).unwrap_or(d));
        }
        let _ = writeln!(s, "metrics ({}):", c.metrics.len());
        for (k, v) in &c.metrics {
            let _ = writeln!(s, "  {k} = {v}");
        }
        s
    }
}

// ── run diff (ADR 0099 capability 3) ───────────────────────────────

/// The symmetric difference of two runs (ADR 0099): only the fields that
/// DIFFER. A metric present in one and absent in the other shows the missing
/// side as `None`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunDiff {
    pub run_a: String,
    pub run_b: String,
    /// `(a, b)` recipe names, present only if they differ.
    pub recipe: Option<(Option<String>, Option<String>)>,
    /// `(a, b)` config fingerprints (the args + input-artifact identity),
    /// present only if they differ.
    pub config_fingerprint: Option<(Option<String>, Option<String>)>,
    /// `(a, b)` gate outcomes, present only if they differ.
    pub gate_outcome: Option<(Option<String>, Option<String>)>,
    /// Per-metric `(name, a_value, b_value)` for every metric whose value
    /// differs or is missing on one side. Sorted by name.
    pub metric_deltas: Vec<(String, Option<f64>, Option<f64>)>,
}

impl RunDiff {
    /// Whether the two runs are identical across every compared dimension.
    pub fn is_empty(&self) -> bool {
        self.recipe.is_none()
            && self.config_fingerprint.is_none()
            && self.gate_outcome.is_none()
            && self.metric_deltas.is_empty()
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
    fn dot_label_escapes_injection() {
        // A stage name with a quote must not break out of the DOT label.
        let g = ProvenanceGraph {
            root: "aa".into(),
            nodes: vec![GraphNode {
                content_hash: "aa".into(),
                stage_name: Some("evil\" ]; hack [x=\"y".into()),
                kind: None,
                job_id: None,
            }],
            edges: vec![],
        };
        let dot = g.to_dot();
        assert!(
            dot.contains("evil\\\" ]; hack [x=\\\"y"),
            "quotes are escaped: {dot}"
        );
        // Exactly one node line (no injected statements broke the structure).
        assert_eq!(dot.matches("[label=").count(), 1);
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
