// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0099 gate (all three read-side capabilities), against a fixture lineage
//! db: (1) `graph_upstream` returns the full transitive producer set of a leaf
//! with no missing/extra edges + a deterministic DOT export; (2) `model_card` is
//! byte-identical across rebuilds (content-addressed) and carries the
//! data/args/metrics/gate-outcome sections; (3) `run_diff` reports exactly the
//! seeded recipe/config/gate/metric deltas and nothing else. Clinical/
//! `restricted`-tenant rows are fail-closed excluded from every export
//! (ADR 0061), and a set tenant is never downgraded on re-ingest.

use blut::lineage_db::{ArtifactRow, EdgeRow, LineageDb, MetricRow, RunRow};

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

fn final_metric(job: &str, name: &str, value: f64) -> MetricRow {
    MetricRow {
        job_id: job.to_string(),
        node_idx: 0,
        step: -1, // the per-(job,node) FINAL marker `final_metrics` reads
        metric: name.to_string(),
        value,
        wall_unix: None,
    }
}

#[test]
fn card_is_deterministic_and_excludes_clinical_data() {
    let (db, _td, leaf) = fixture();
    // Enrich the research run with args + gate outcome + a headline metric.
    let mut r = run("job_r", "research/dev");
    r.recipe = "train_joint".into();
    r.outcome = Some("PASS".into());
    r.config_fingerprint = Some("cfg123".into());
    db.record_run(&r).unwrap();
    db.record_metrics(&[final_metric("job_r", "val_r", 0.87)])
        .unwrap();

    let card = db.model_card(&leaf, true).unwrap().unwrap();
    // All four sections populate: data + args + metrics + gate outcome.
    assert_eq!(card.content.recipe.as_deref(), Some("train_joint"));
    assert_eq!(card.content.config_fingerprint.as_deref(), Some("cfg123"));
    assert_eq!(card.content.gate_outcome.as_deref(), Some("PASS"));
    assert_eq!(card.content.metrics, vec![("val_r".to_string(), 0.87)]);
    // The clinical data source is excluded from the export; the research one is in.
    assert!(
        card.content.data_sources.contains(&h('a')),
        "research source present"
    );
    assert!(
        !card.content.data_sources.contains(&h('d')),
        "clinical data source must be excluded from the card"
    );
    // Content-addressed determinism: a rebuild on the same rows is byte-identical.
    let again = db.model_card(&leaf, true).unwrap().unwrap();
    assert_eq!(card.card_hash, again.card_hash);
    assert!(!card.card_hash.is_empty());
    assert!(card.verify(), "card_hash must verify against its content");
}

#[test]
fn diff_reports_exactly_the_seeded_deltas() {
    let td = tempfile::tempdir().unwrap();
    let db = LineageDb::open_at(td.path().join("lineage.db")).unwrap();

    let mut a = run("run_a", "research/dev");
    a.recipe = "train".into();
    a.outcome = Some("PASS".into());
    a.config_fingerprint = Some("cfgA".into());
    let mut b = run("run_b", "research/dev");
    b.recipe = "train".into(); // SAME recipe → no recipe delta
    b.outcome = Some("FAIL".into());
    b.config_fingerprint = Some("cfgB".into());
    db.record_run(&a).unwrap();
    db.record_run(&b).unwrap();
    db.record_metrics(&[
        final_metric("run_a", "val_r", 0.8),
        final_metric("run_a", "loss", 0.1),
    ])
    .unwrap();
    db.record_metrics(&[
        final_metric("run_b", "val_r", 0.9),
        final_metric("run_b", "loss", 0.1),
    ])
    .unwrap();

    let d = db.run_diff("run_a", "run_b").unwrap();
    assert!(d.recipe.is_none(), "identical recipe ⇒ no recipe delta");
    assert_eq!(
        d.config_fingerprint,
        Some((Some("cfgA".into()), Some("cfgB".into())))
    );
    assert_eq!(
        d.gate_outcome,
        Some((Some("PASS".into()), Some("FAIL".into())))
    );
    // Only val_r differs (0.8 vs 0.9); loss is identical (0.1) so it is EXCLUDED.
    assert_eq!(
        d.metric_deltas,
        vec![("val_r".to_string(), Some(0.8), Some(0.9))]
    );
    assert!(!d.is_empty());
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
