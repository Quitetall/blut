// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0090 M2.2: dataset/experiment registry completion, recipe-argument URI
//! resolution, and a bounded asynchronous governance gate.

use std::time::Duration;

use blut::config::launcher::LaunchTarget;
use blut::datasets_db;
use blut::lineage_db::{EdgeRow, LineageDb, MetricRow, RunRow};
use blut::tenant::Tenant;
use serde_json::json;

mod registry {
    use super::*;

    const MODEL_HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CLINICAL_MODEL_HASH: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn add_dataset(
        conn: &rusqlite::Connection,
        root: &std::path::Path,
        source_name: &str,
        body: &str,
        metadata: serde_json::Value,
    ) -> datasets_db::DatasetRecord {
        let path = root.join(format!("{source_name}.jsonl"));
        std::fs::write(&path, body).unwrap();
        let record = datasets_db::record_from_jsonl(
            source_name,
            &path,
            "dataset.jsonl",
            Some(metadata.to_string()),
        )
        .unwrap();
        datasets_db::add(conn, &record).unwrap();
        record
    }

    fn run(job: &str, recipe: &str, tenant: &str, started: i64, fp: &str) -> RunRow {
        RunRow {
            job_id: job.into(),
            recipe: recipe.into(),
            config_fingerprint: Some(fp.into()),
            started_unix: Some(started),
            outcome: Some("done".into()),
            tenant: tenant.into(),
            experiment: Some("campaign-a".into()),
            args_json: Some(
                json!({
                    "lr": if job == "run-a" { 0.01 } else { 0.02 },
                    "nested": {"seed": 7}
                })
                .to_string(),
            ),
            ..RunRow::default()
        }
    }

    #[test]
    fn dataset_versions_are_immutable_tenant_scoped_and_remote_fail_closed() {
        let td = tempfile::tempdir().unwrap();
        let conn = datasets_db::open_at(&td.path().join("datasets.db")).unwrap();
        let research = Tenant::parse("research/dev").unwrap();
        let clinical = Tenant::parse("clinical/prod").unwrap();

        let tuh = add_dataset(
            &conn,
            td.path(),
            "tuh-source",
            "{\"x\":1}\n",
            json!({"modality":"eeg","fs":256}),
        );
        let pinned =
            blut::dataset_registry::pin(&conn, &tuh.name, "dataset://tuh@v3", &research, 100)
                .unwrap();
        assert_eq!(pinned.manifest_sha256, tuh.sha256);
        assert_eq!(pinned.source_path, tuh.source_path);

        let resolved = blut::dataset_registry::resolve_uri(
            &conn,
            "dataset://tuh@v3",
            &research,
            LaunchTarget::Local,
        )
        .unwrap();
        assert_eq!(resolved.manifest_sha256, tuh.sha256);
        assert!(
            blut::dataset_registry::pin(&conn, &tuh.name, "dataset://tuh@v3", &research, 101,)
                .is_ok(),
            "pinning the same immutable binding is idempotent"
        );
        assert!(
            blut::dataset_registry::pin(&conn, &tuh.name, "dataset://tuh-copy@v1", &clinical, 101,)
                .is_err(),
            "the same manifest hash must not acquire a second tenant owner"
        );

        let phi = add_dataset(
            &conn,
            td.path(),
            "phi-source",
            "{\"phi\":true}\n",
            json!({"clinical":true,"tenant":"clinical/prod"}),
        );
        blut::dataset_registry::pin(&conn, &phi.name, "dataset://phi@v1", &clinical, 102).unwrap();
        assert!(
            blut::dataset_registry::resolve_uri(
                &conn,
                "dataset://phi@v1",
                &research,
                LaunchTarget::Local,
            )
            .is_err(),
            "another tenant must not resolve the binding"
        );
        assert!(
            blut::dataset_registry::resolve_uri(
                &conn,
                "dataset://phi@v1",
                &clinical,
                LaunchTarget::Cloud,
            )
            .is_err(),
            "Restricted datasets must not resolve onto a remote launcher"
        );
        assert!(
            blut::dataset_registry::resolve_uri(
                &conn,
                "dataset://phi@v1",
                &clinical,
                LaunchTarget::Local,
            )
            .is_ok()
        );

        std::fs::write(&tuh.source_path, "{\"x\":2}\n").unwrap();
        assert!(
            blut::dataset_registry::resolve_uri(
                &conn,
                "dataset://tuh@v3",
                &research,
                LaunchTarget::Local,
            )
            .is_err(),
            "source bytes drifting from the pinned manifest hash must fail closed"
        );

        for (source_name, body, metadata) in [
            ("bad-tenant", "{\"bad_tenant\":1}\n", json!({"tenant":123})),
            (
                "bad-clinical",
                "{\"bad_clinical\":1}\n",
                json!({"clinical":"true"}),
            ),
        ] {
            let bad = add_dataset(&conn, td.path(), source_name, body, metadata);
            assert!(
                blut::dataset_registry::pin(
                    &conn,
                    &bad.name,
                    &format!("dataset://{source_name}@v1"),
                    &research,
                    103,
                )
                .is_err(),
                "mistyped classification metadata must fail closed"
            );
        }
    }

    #[test]
    fn experiment_uri_and_latest_compare_are_tenant_scoped_lineage_views() {
        let td = tempfile::tempdir().unwrap();
        let db = LineageDb::open_at(td.path().join("lineage.db")).unwrap();
        db.record_run(&run("run-a", "codec-train", "research/dev", 10, "fp-a"))
            .unwrap();
        db.record_run(&run("run-b", "codec-train", "research/dev", 20, "fp-b"))
            .unwrap();
        db.record_run(&run("run-c", "codec-train", "clinical/prod", 30, "fp-c"))
            .unwrap();
        db.record_run(&run("run-d", "codec-train", "clinical/prod", 40, "fp-d"))
            .unwrap();
        let mut relabeled = run("run-a", "codec-train", "clinical/prod", 10, "fp-a");
        relabeled.experiment = Some("campaign-a".into());
        assert!(
            db.record_run(&relabeled).is_err(),
            "a recorded run's tenant boundary must be immutable"
        );
        db.record_metrics(&[
            MetricRow {
                job_id: "run-a".into(),
                node_idx: 0,
                step: -1,
                metric: "loss".into(),
                value: 2.0,
                wall_unix: Some(10),
            },
            MetricRow {
                job_id: "run-b".into(),
                node_idx: 0,
                step: -1,
                metric: "loss".into(),
                value: 1.0,
                wall_unix: Some(20),
            },
        ])
        .unwrap();
        db.record_edge(&EdgeRow {
            job_id: "run-a".into(),
            to_idx: 0,
            input_hash: "input-a".into(),
            output_hash: "output-a".into(),
        })
        .unwrap();
        db.record_edge(&EdgeRow {
            job_id: "run-b".into(),
            to_idx: 0,
            input_hash: "input-b".into(),
            output_hash: "output-b".into(),
        })
        .unwrap();

        let research = Tenant::parse("research/dev").unwrap();
        let resolved =
            blut::experiment_registry::resolve_uri(&db, "experiment://campaign-a/run-b", &research)
                .unwrap();
        assert_eq!(resolved.job_id, "run-b");
        let comparison =
            blut::experiment_registry::compare_latest(&db, "campaign-a", &research).unwrap();
        assert_eq!(
            (comparison.run_a.as_str(), comparison.run_b.as_str()),
            ("run-a", "run-b")
        );
        assert_eq!(
            comparison
                .diff
                .config_fingerprint
                .as_ref()
                .unwrap()
                .0
                .as_deref(),
            Some("fp-a")
        );
        assert_eq!(comparison.diff.metric_deltas[0].0, "loss");
        assert_eq!(
            comparison.diff.input_hashes,
            Some((vec!["input-a".into()], vec!["input-b".into()]))
        );
        assert_eq!(comparison.diff.arg_deltas.len(), 1);
        assert_eq!(comparison.diff.arg_deltas[0].path, "$/lr");

        let clinical = Tenant::parse("clinical/prod").unwrap();
        let clinical_comparison =
            blut::experiment_registry::compare_latest(&db, "campaign-a", &clinical).unwrap();
        assert_eq!(clinical_comparison.run_a, "run-c");
        assert_eq!(clinical_comparison.run_b, "run-d");
        assert!(
        blut::experiment_registry::resolve_uri(&db, "experiment://campaign-a/run-b", &clinical,)
            .is_err(),
        "an experiment URI may not cross its run's tenant"
    );
    }

    #[test]
    fn recipe_args_resolve_registry_uris_before_typed_deserialization() {
        let td = tempfile::tempdir().unwrap();
        let datasets = datasets_db::open_at(&td.path().join("datasets.db")).unwrap();
        let source = add_dataset(
            &datasets,
            td.path(),
            "train-source",
            "{\"sample\":1}\n",
            json!({"tenant":"research/dev"}),
        );
        let tenant = Tenant::parse("research/dev").unwrap();
        blut::dataset_registry::pin(&datasets, &source.name, "dataset://train@v1", &tenant, 1)
            .unwrap();

        let mut models = blut::model_registry::open_at(&td.path().join("models.db")).unwrap();
        blut::model_registry::register(&models, MODEL_HASH, "encoder", "research/dev", None, 1)
            .unwrap();
        blut::model_registry::promote(
            &mut models,
            MODEL_HASH,
            "research/dev",
            "encoder",
            "staging",
            2,
        )
        .unwrap();

        let lineage = LineageDb::open_at(td.path().join("lineage.db")).unwrap();
        lineage
            .record_run(&run("run-a", "codec-train", "research/dev", 10, "fp-a"))
            .unwrap();

        let resolved = blut::registry_args::resolve_with(
            json!({
                "dataset_path":"dataset://train@v1",
                "model_hash":"model://encoder@staging",
                "parent_run":"experiment://campaign-a/run-a",
                "nested":["unchanged", {"again":"dataset://train@v1"}]
            }),
            &tenant,
            LaunchTarget::Local,
            &datasets,
            &models,
            &lineage,
        )
        .unwrap();
        assert_eq!(resolved["dataset_path"], json!(source.source_path));
        assert_eq!(resolved["model_hash"], MODEL_HASH);
        assert_eq!(resolved["parent_run"], "run-a");
        assert_eq!(resolved["nested"][0], "unchanged");
        assert_eq!(resolved["nested"][1]["again"], json!(source.source_path));

        blut::model_registry::register(
            &models,
            CLINICAL_MODEL_HASH,
            "clinical-encoder",
            "clinical/prod",
            None,
            3,
        )
        .unwrap();
        blut::model_registry::promote(
            &mut models,
            CLINICAL_MODEL_HASH,
            "clinical/prod",
            "clinical-encoder",
            "staging",
            4,
        )
        .unwrap();
        let clinical = Tenant::parse("clinical/prod").unwrap();
        assert!(
            blut::registry_args::resolve_with(
                json!({"model":"model://clinical-encoder@staging"}),
                &clinical,
                LaunchTarget::Cloud,
                &datasets,
                &models,
                &lineage,
            )
            .is_err(),
            "Restricted model handles must not resolve onto a remote launcher"
        );
    }

    /// A biosignal corpus is clinical and is not line-delimited JSON.
    ///
    /// Both facts were unrepresentable through the registration path: the
    /// example count came from parsing JSONL lines, and metadata was hardcoded
    /// `None` — so `classify_source`, which reads `clinical`/`tenant` from
    /// metadata, could never see a classification at all.
    #[test]
    fn a_clinical_non_jsonl_corpus_registers_pins_and_stays_node_local() {
        let td = tempfile::tempdir().unwrap();
        let conn = datasets_db::open_at(&td.path().join("datasets.db")).unwrap();
        let clinical = Tenant::parse("clinical/corpora").unwrap();
        let research = Tenant::parse("research/dev").unwrap();

        // A corpus manifest: one canonical JSON line describing 70841 entries.
        // Counting lines would report 1 — technically true, and meaningless.
        let manifest = td.path().join("tueg.json");
        std::fs::write(&manifest, "{\"corpus\":\"tueg\",\"entries\":70841}\n").unwrap();

        let record = datasets_db::record_from_file(
            "tueg-manifest",
            &manifest,
            "lamquant.corpus-manifest",
            70841,
            Some(json!({"clinical": true, "tenant": "clinical/corpora"}).to_string()),
        )
        .unwrap();
        assert_eq!(
            record.n_examples, 70841,
            "the caller's count, not a line count"
        );
        datasets_db::add(&conn, &record).unwrap();

        let pinned =
            blut::dataset_registry::pin(&conn, &record.name, "dataset://tueg@v2.0.2", &clinical, 1)
                .unwrap();
        assert!(pinned.clinical, "a clinical source must pin as clinical");

        // Restricted data resolves node-local only, and never off-tenant.
        assert!(
            blut::dataset_registry::resolve_uri(
                &conn,
                "dataset://tueg@v2.0.2",
                &clinical,
                LaunchTarget::Local,
            )
            .is_ok()
        );
        assert!(
            blut::dataset_registry::resolve_uri(
                &conn,
                "dataset://tueg@v2.0.2",
                &clinical,
                LaunchTarget::Cloud,
            )
            .is_err(),
            "clinical corpora must not resolve onto a remote launcher"
        );

        // A clinical source cannot be pinned into a non-Restricted tenant.
        let leak = td.path().join("tusz.json");
        std::fs::write(&leak, "{\"corpus\":\"tusz\"}\n").unwrap();
        let leaky = datasets_db::record_from_file(
            "tusz-manifest",
            &leak,
            "lamquant.corpus-manifest",
            24429,
            Some(json!({"clinical": true, "tenant": "clinical/corpora"}).to_string()),
        )
        .unwrap();
        datasets_db::add(&conn, &leaky).unwrap();
        assert!(
            blut::dataset_registry::pin(&conn, &leaky.name, "dataset://tusz@v2", &research, 2)
                .is_err(),
            "a clinical corpus must not enter a non-Restricted tenant"
        );

        // Malformed metadata is refused by the command that writes it.
        for bad in ["not json", "[1,2,3]", "\"a string\""] {
            assert!(
                datasets_db::record_from_file(
                    "bad-meta",
                    &manifest,
                    "lamquant.corpus-manifest",
                    1,
                    Some(bad.to_string()),
                )
                .is_err(),
                "metadata {bad:?} must be refused at registration, not at pin time"
            );
        }
        assert!(
            datasets_db::record_from_file("neg", &manifest, "k", -1, None).is_err(),
            "a negative example count is not a count"
        );
    }

    /// Resolution proves a digest; the substituted path cannot carry that proof.
    ///
    /// Before the reported form existed, a persisted run held only the path, so
    /// nothing downstream could say which pinned dataset it came from or at what
    /// bytes. These assertions are the contract that the evidence survives.
    #[test]
    fn resolution_reports_the_pinned_digest_the_substituted_path_cannot_carry() {
        let td = tempfile::tempdir().unwrap();
        let datasets = datasets_db::open_at(&td.path().join("datasets.db")).unwrap();
        let source = add_dataset(
            &datasets,
            td.path(),
            "corpus-source",
            "{\"sample\":1}\n",
            json!({"tenant":"research/dev"}),
        );
        let tenant = Tenant::parse("research/dev").unwrap();
        let pinned = blut::dataset_registry::pin(
            &datasets,
            &source.name,
            "dataset://corpus@v2.0.6",
            &tenant,
            1,
        )
        .unwrap();

        let models = blut::model_registry::open_at(&td.path().join("models.db")).unwrap();
        let lineage = LineageDb::open_at(td.path().join("lineage.db")).unwrap();

        // The same handle twice, one of them nested, plus an ordinary string.
        let (args, handles) = blut::registry_args::resolve_with_reported(
            json!({
                "train":"dataset://corpus@v2.0.6",
                "plain":"not-a-uri",
                "nested":{"val":"dataset://corpus@v2.0.6"}
            }),
            &tenant,
            LaunchTarget::Local,
            &datasets,
            &models,
            &lineage,
        )
        .unwrap();

        // Substitution is unchanged, so every existing typed arg still parses.
        assert_eq!(args["train"], json!(source.source_path));
        assert_eq!(args["nested"]["val"], json!(source.source_path));
        assert_eq!(args["plain"], "not-a-uri");

        // ...and the evidence now escapes alongside it, collapsed to one entry.
        assert_eq!(
            handles.datasets.len(),
            1,
            "one handle used twice is one row"
        );
        let bound = &handles.datasets[0];
        assert_eq!(bound.name, "corpus");
        assert_eq!(bound.version, "v2.0.6");
        assert_eq!(bound.tenant, "research/dev");
        assert!(!bound.clinical);
        assert_eq!(
            bound.manifest_sha256, pinned.manifest_sha256,
            "the reported digest must be the pinned one that was just re-verified"
        );
        assert_eq!(bound.manifest_sha256, source.sha256);

        // URI-free args stay a no-op and report nothing.
        let (untouched, none) = blut::registry_args::resolve_with_reported(
            json!({"epochs": 3, "name": "plain"}),
            &tenant,
            LaunchTarget::Local,
            &datasets,
            &models,
            &lineage,
        )
        .unwrap();
        assert_eq!(untouched, json!({"epochs": 3, "name": "plain"}));
        assert!(none.is_empty());

        // Fail-closed: drifted source bytes refuse rather than follow the path.
        std::fs::write(&source.source_path, "{\"sample\":2}\n").unwrap();
        let drifted = blut::registry_args::resolve_with_reported(
            json!({"train":"dataset://corpus@v2.0.6"}),
            &tenant,
            LaunchTarget::Local,
            &datasets,
            &models,
            &lineage,
        );
        assert!(
            drifted.is_err(),
            "a pinned name must never silently follow changed bytes"
        );
    }

    #[tokio::test]
    async fn governance_gate_is_async_bounded_and_fail_closed() {
        use blut::model_registry::{GateCmd, GateVerdict};

        let pass = GateCmd {
            program: "true".into(),
            args: vec![],
        };
        assert_eq!(
            blut::model_registry::run_gate_async(
                &pass,
                MODEL_HASH,
                "encoder",
                "prod",
                "chg-1",
                Duration::from_secs(1),
            )
            .await,
            GateVerdict::Pass
        );

        let td = tempfile::tempdir().unwrap();
        let escaped_marker = td.path().join("gate-survived");
        let hung = GateCmd {
            program: "sh".into(),
            args: vec![
                "-c".into(),
                format!("sleep 0.2; touch {}", escaped_marker.display()),
            ],
        };
        let started = std::time::Instant::now();
        let verdict = blut::model_registry::run_gate_async(
            &hung,
            MODEL_HASH,
            "encoder",
            "prod",
            "chg-2",
            Duration::from_millis(50),
        )
        .await;
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(matches!(verdict, GateVerdict::Fail(reason) if reason.contains("timed out")));
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            !escaped_marker.exists(),
            "timeout must kill the whole gate process group, not only its shell"
        );

        let noisy = GateCmd {
            program: "sh".into(),
            args: vec!["-c".into(), "head -c 1100000 /dev/zero 1>&2".into()],
        };
        let verdict = blut::model_registry::run_gate_async(
            &noisy,
            MODEL_HASH,
            "encoder",
            "prod",
            "chg-3",
            Duration::from_secs(2),
        )
        .await;
        assert!(
            matches!(verdict, GateVerdict::Fail(reason) if reason.contains("capture limit")),
            "a noisy governance process must fail before unbounded output can accumulate"
        );
    }
}
