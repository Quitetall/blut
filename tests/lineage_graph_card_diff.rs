// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0099 gate (increment 1 — provenance GRAPH): against a fixture lineage db,
//! `graph_upstream` returns the full transitive producer set of a leaf artifact
//! with no missing/extra edges, the DOT export is deterministic, and a
//! clinical/`restricted`-tenant node is never present in an exported graph
//! (ADR 0061 fail-closed).
//!
//! (Card + run-diff — the other two 0099 capabilities — land in increment 2 and
//! extend this same test binary.)

use blut::lineage_db::{ArtifactRow, EdgeRow, LineageDb, RunRow};

/// A 64-hex content hash built by repeating `c`.
fn h(c: char) -> String {
    c.to_string().repeat(64)
}

fn run(job: &str, tenant: &str) -> RunRow {
    RunRow {
        job_id: job.to_string(),
        recipe: "demo".into(),
        tenant: tenant.to_string(),
        ..Default::default()
    }
}

fn artifact(job: &str, idx: i64, stage: &str, hash: &str) -> ArtifactRow {
    ArtifactRow {
        job_id: job.to_string(),
        stage_idx: idx,
        stage_name: stage.to_string(),
        content_hash: hash.to_string(),
        kind: "eeg".into(),
        schema_ver: 1,
        sidecar_path: None,
        produced_unix: Some(1000 + idx),
    }
}

fn edge(job: &str, to_idx: i64, input: &str, output: &str) -> EdgeRow {
    EdgeRow {
        job_id: job.to_string(),
        to_idx,
        input_hash: input.to_string(),
        output_hash: output.to_string(),
    }
}

/// Fixture:  src ─► mid ─► leaf  (research tenant)
///                    clin ─────►┘  (clinical tenant, feeds leaf)
fn fixture() -> (LineageDb, tempfile::TempDir, String) {
    let td = tempfile::tempdir().unwrap();
    let db = LineageDb::open_at(td.path().join("lineage.db")).unwrap();

    let (src, mid, leaf, clin) = (h('a'), h('b'), h('c'), h('d'));

    db.record_run(&run("job_r", "research/dev")).unwrap();
    db.record_run(&run("job_c", "clinical/prod")).unwrap();

    db.record_artifact(&artifact("job_r", 0, "gen", &src))
        .unwrap();
    db.record_artifact(&artifact("job_r", 1, "encode", &mid))
        .unwrap();
    db.record_artifact(&artifact("job_r", 2, "train", &leaf))
        .unwrap();
    db.record_artifact(&artifact("job_c", 0, "phi_source", &clin))
        .unwrap();

    db.record_edge(&edge("job_r", 1, &src, &mid)).unwrap();
    db.record_edge(&edge("job_r", 2, &mid, &leaf)).unwrap();
    db.record_edge(&edge("job_c", 2, &clin, &leaf)).unwrap(); // clinical input!

    (db, td, leaf)
}

#[test]
fn graph_returns_full_transitive_producer_set() {
    let (db, _td, leaf) = fixture();
    // WITHOUT the export boundary: the whole ancestry, clinical node included.
    let g = db.graph_upstream(&leaf, false).unwrap();
    let mut hashes: Vec<&str> = g.nodes.iter().map(|n| n.content_hash.as_str()).collect();
    hashes.sort_unstable();
    assert_eq!(hashes, vec![h('a'), h('b'), h('c'), h('d')]);
    // Every edge present, none extra.
    let mut edges = g.edges.clone();
    edges.sort();
    let mut want = vec![(h('a'), h('b')), (h('b'), h('c')), (h('d'), h('c'))];
    want.sort();
    assert_eq!(edges, want);
    // The leaf's transitive sources are the two inputless roots (src + clin).
    let mut sources = g.sources();
    sources.sort_unstable();
    assert_eq!(sources, vec![h('a').as_str(), h('d').as_str()]);
}

#[test]
fn tenant_is_never_downgraded_on_reingest() {
    let td = tempfile::tempdir().unwrap();
    let db = LineageDb::open_at(td.path().join("lineage.db")).unwrap();
    // Record a clinical run, then re-ingest it with NO tenant (coerced to
    // `default`) — the ADR-0061 clinical tenant must survive.
    db.record_run(&run("job_c", "clinical/prod")).unwrap();
    db.record_run(&run("job_c", "")).unwrap(); // re-ingest, tenant unset
    assert_eq!(
        db.get_run("job_c").unwrap().unwrap().tenant,
        "clinical/prod",
        "a re-ingest must not downgrade a set tenant to default"
    );
}

#[test]
fn clinical_node_never_in_exported_graph() {
    let (db, _td, leaf) = fixture();
    // WITH the export boundary (ADR 0061): the clinical node AND its edge are gone.
    let g = db.graph_upstream(&leaf, true).unwrap();
    let hashes: Vec<&str> = g.nodes.iter().map(|n| n.content_hash.as_str()).collect();
    assert!(
        !hashes.contains(&h('d').as_str()),
        "clinical artifact must be excluded from the export"
    );
    assert_eq!(hashes.len(), 3, "only the research chain src→mid→leaf");
    assert!(
        !g.edges.iter().any(|(f, _)| *f == h('d')),
        "no edge from the clinical node may survive the export"
    );
    // The DOT export contains no clinical hash and is deterministic.
    let dot = g.to_dot();
    assert!(!dot.contains(&h('d')[..12]));
    assert_eq!(dot, db.graph_upstream(&leaf, true).unwrap().to_dot());
}
