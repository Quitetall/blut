// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0096 (increment 1): tenant-namespaced content-addressed cache + the
//! clinical boundary keyed by `Tenant`. Proves that the same graph in two
//! tenants gets DISJOINT cache roots (neither reads the other), the ADR-0078
//! fingerprint is byte-identical across tenants (namespace = path, not key), the
//! `default` tenant is the flat store, and the ADR-0085 registry refuses a
//! cross-tenant promote when keyed by `Tenant::to_string()`.

use std::sync::Arc;

use async_trait::async_trait;
use blut::broker::tenant_quota::TenantQuotaTracker;
use blut::broker::{Footprint, GIB, ResourceSnapshot};
use blut::checks::JsonlArtifact;
use blut::config::tenants::TenantQuotaPolicy;
use blut::framework::CacheHandle;
use blut::framework::ExecCtx;
use blut::framework::Registry;
use blut::framework::artifact::ContentHash;
use blut::framework::cookbook::Cookbook;
use blut::framework::error::StageError;
use blut::framework::executor::SequentialExecutor;
use blut::framework::plan_spec::{PlanSpec, SpecNode};
use blut::framework::resource::Resource;
use blut::framework::stage::{ErasedStageCtor, Stage, StageContext};
use blut::recipes::recipe::RecipeDef;
use blut::registry_db;
use blut::tenant::Tenant;

#[test]
fn tenant_quota_config_defaults_equal_shares_and_refuses_unknown_tenants() {
    let default_only = TenantQuotaPolicy::default();
    assert_eq!(default_only.fraction_for(&Tenant::default()).unwrap(), 1.0);
    assert!(
        default_only
            .fraction_for(&Tenant::parse("research/dev").unwrap())
            .is_err(),
        "without tenant config, a non-default tenant must be refused"
    );

    let equal =
        TenantQuotaPolicy::from_toml(r#"tenants = ["research/dev", "clinical/prod"]"#).unwrap();
    assert_eq!(
        equal
            .fraction_for(&Tenant::parse("research/dev").unwrap())
            .unwrap(),
        0.5
    );
    assert_eq!(
        equal
            .fraction_for(&Tenant::parse("clinical/prod").unwrap())
            .unwrap(),
        0.5
    );
    assert!(
        equal
            .fraction_for(&Tenant::parse("unknown/prod").unwrap())
            .is_err(),
        "an unknown tenant must never inherit another tenant's share"
    );

    let explicit = TenantQuotaPolicy::from_toml(
        r#"
        [fractions]
        "research/dev" = 0.75
        "clinical/prod" = 0.25
        "#,
    )
    .unwrap();
    assert_eq!(
        explicit
            .fraction_for(&Tenant::parse("research/dev").unwrap())
            .unwrap(),
        0.75
    );
    assert_eq!(
        explicit
            .fraction_for(&Tenant::parse("clinical/prod").unwrap())
            .unwrap(),
        0.25
    );

    for invalid in [
        "tenants = []",
        "tenants = [\"research/dev\", \"research/dev\"]",
        "tenants = [\"research/dev\"]\n[fractions]\n\"clinical/prod\" = 1.0",
        "[fractions]\n\"research/dev\" = 0.0",
        "[fractions]\n\"research/dev\" = 0.8\n\"clinical/prod\" = 0.8",
        "[fractions]\n\"../escape\" = 1.0",
    ] {
        assert!(
            TenantQuotaPolicy::from_toml(invalid).is_err(),
            "invalid tenant quota config must fail closed: {invalid}"
        );
    }
}

#[test]
fn tenant_quota_reservation_is_stateful_atomic_and_released_on_drop() {
    let tracker = TenantQuotaTracker::new();
    let tenant = Tenant::parse("research/dev").unwrap();
    let snap = ResourceSnapshot {
        mem_total_gb: 32.0,
        mem_avail_gb: 32.0,
        ..ResourceSnapshot::default()
    };
    let eight_gib = Footprint {
        ram_bytes: 8 * GIB,
        vram_mib: 0,
    };
    assert_eq!(
        TenantQuotaTracker::ceiling_gib(&snap, 6.0, 0.5).unwrap(),
        13.0
    );

    // 50% of (32 GiB total - 6 GiB floor) = 13 GiB. One 8 GiB job fits;
    // a concurrent second one does not.
    let first = tracker
        .reserve(&snap, &eight_gib, 6.0, &tenant, 0.5)
        .unwrap();
    assert_eq!(tracker.in_flight_gib(&tenant), 8.0);
    let refusal = tracker
        .reserve(&snap, &eight_gib, 6.0, &tenant, 0.5)
        .unwrap_err();
    assert!(format!("{refusal}").contains("tenant 'research/dev' RAM quota exceeded"));
    assert_eq!(tracker.in_flight_gib(&tenant), 8.0);

    // The permit is the accounting lifetime. Every return path releases by Drop.
    drop(first);
    assert_eq!(tracker.in_flight_gib(&tenant), 0.0);
    let second = tracker
        .reserve(&snap, &eight_gib, 6.0, &tenant, 0.5)
        .expect("released capacity must be immediately reusable");
    drop(second);

    // The enforcement seam itself is fail-closed even if a caller bypasses the
    // config parser and supplies a nonsense fraction directly.
    assert!(
        tracker
            .reserve(&snap, &eight_gib, 6.0, &tenant, f64::NAN)
            .is_err()
    );
    assert_eq!(tracker.in_flight_gib(&tenant), 0.0);

    let probe_miss = ResourceSnapshot::default();
    let parity = tracker
        .reserve(&probe_miss, &eight_gib, 6.0, &tenant, 1.0)
        .expect("a 100% single-tenant share preserves legacy probe-miss parity");
    drop(parity);
    assert!(
        tracker
            .reserve(&probe_miss, &eight_gib, 6.0, &tenant, 0.5)
            .is_err(),
        "a fractional share cannot be enforced without a capacity probe"
    );
}

#[test]
fn persisted_job_tenant_reaches_lineage_and_survives_reindex() {
    let td = tempfile::tempdir().unwrap();
    let jobs_dir = td.path().join("jobs");
    let data_dir = td.path().join("data");
    std::fs::create_dir_all(&jobs_dir).unwrap();
    std::fs::create_dir_all(&data_dir).unwrap();

    let old_jobs = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
    let old_data = std::env::var("LAMU_TRAIN_DATA_DIR").ok();
    // SAFETY: this integration binary's other tests do not read either path;
    // both variables are restored before return.
    unsafe {
        std::env::set_var("LAMU_TRAIN_JOBS_DIR", &jobs_dir);
        std::env::set_var("LAMU_TRAIN_DATA_DIR", &data_dir);
    }

    let job_id = "tenant-lineage-test";
    let clinical = Tenant::parse("clinical/prod").unwrap();
    blut::jobs::write_tenant(job_id, &clinical).unwrap();
    blut::jobs::write_experiment(job_id, "campaign-a").unwrap();
    assert_eq!(blut::jobs::read_tenant(job_id).unwrap(), clinical);
    assert_eq!(
        blut::jobs::read_experiment(job_id).unwrap().as_deref(),
        Some("campaign-a")
    );
    assert!(blut::jobs::write_experiment(job_id, "../escape").is_err());
    blut::lineage_db::ingest_job(job_id, "tenant-test", "done").unwrap();

    let db = blut::lineage_db::LineageDb::open_at(data_dir.join("lineage.db")).unwrap();
    let row = db.get_run(job_id).unwrap().unwrap();
    assert_eq!(row.tenant, "clinical/prod");
    assert_eq!(row.experiment.as_deref(), Some("campaign-a"));
    // Re-ingest reads the same canonical marker; it must never downgrade the
    // restricted row to `default`.
    blut::lineage_db::ingest_job(job_id, "tenant-test", "done").unwrap();
    assert_eq!(db.get_run(job_id).unwrap().unwrap().tenant, "clinical/prod");

    // SAFETY: restore process-global environment before leaving the test.
    unsafe {
        match old_jobs {
            Some(value) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", value),
            None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
        }
        match old_data {
            Some(value) => std::env::set_var("LAMU_TRAIN_DATA_DIR", value),
            None => std::env::remove_var("LAMU_TRAIN_DATA_DIR"),
        }
    }
}

#[test]
fn restricted_tenant_is_threaded_to_executor_and_stage_boundary() {
    let clinical = Tenant::parse("clinical/prod").unwrap();
    let ctx = ExecCtx::new(std::path::PathBuf::from("/tmp/blut-tenant-context-test"))
        .with_tenant(clinical.clone());
    assert_eq!(ctx.tenant, clinical);
    assert!(ctx.tenant.is_restricted());
}

#[test]
fn restricted_mesh_and_notify_boundaries_refuse_cross_node_delivery() {
    use blut::trust::{DataClass, DispatchMatrix, TrustLevel};
    use blut_notify::{NotificationEnvelope, NotifySink, SinkBoundary, deliver};

    let mut matrix = DispatchMatrix::default();
    matrix.set(DataClass::Restricted, TrustLevel::Trusted, true);
    assert!(
        !matrix.can_dispatch(DataClass::Restricted, TrustLevel::Trusted),
        "Restricted must stay node-local even under a custom trust matrix"
    );

    struct OffBoxSink {
        calls: usize,
    }

    impl NotifySink for OffBoxSink {
        fn boundary(&self) -> SinkBoundary {
            SinkBoundary::OffBox
        }

        fn send(&mut self, _envelope: &NotificationEnvelope) -> Result<(), String> {
            self.calls += 1;
            Ok(())
        }
    }

    let mut sink = OffBoxSink { calls: 0 };
    let envelope = NotificationEnvelope {
        tenant: Tenant::parse("clinical/prod").unwrap(),
        data_class: DataClass::Public,
        summary: "patient-name-must-not-leak".into(),
    };
    let error = deliver(&mut sink, &envelope).unwrap_err();
    assert_eq!(
        sink.calls, 0,
        "refusal must happen before the sink sees PHI"
    );
    assert!(!error.to_string().contains(&envelope.summary));
}

#[cfg(feature = "p2p")]
#[test]
fn restricted_privacy_ledger_refuses_cross_tenant_use() {
    use blut::p2p::crypto::KeyPair;
    use blut::p2p::privacy::PrivacyLedger;

    let td = tempfile::tempdir().unwrap();
    let clinical = Tenant::parse("clinical/prod").unwrap();
    let research = Tenant::parse("research/dev").unwrap();
    let kp = KeyPair::generate();
    let path = PrivacyLedger::path_for_tenant(td.path(), &clinical);
    let mut ledger = PrivacyLedger::new_for_tenant(path.clone(), &clinical);
    ledger.set_budget("corpus", 2.0);
    ledger.record("corpus", 0.5, &kp).unwrap();
    ledger.save().unwrap();
    assert!(PrivacyLedger::load_for_tenant(path, &kp.verifying, &research).is_err());
}

#[tokio::test]
async fn same_graph_executes_in_disjoint_tenant_cache_namespaces() {
    let td = tempfile::tempdir().unwrap();
    let base = td.path().join("cache");
    let job = td.path().join("job");

    let research = Tenant::parse("research/dev").unwrap();
    let clinical = Tenant::parse("clinical/prod").unwrap();

    let make = |t: &Tenant, local: &str| {
        CacheHandle::job_local(job.join(local).join("_cache"))
            .with_global(base.clone())
            .with_tenant(t)
    };
    let (h_research, h_clinical) = (make(&research, "research"), make(&clinical, "clinical"));

    // 1. Disjoint global roots, neither an ancestor of the other.
    let gr = h_research.global.clone().unwrap();
    let gc = h_clinical.global.clone().unwrap();
    assert_ne!(gr, gc);
    assert!(gr.ends_with("research/dev") && gc.ends_with("clinical/prod"));
    assert!(!gr.starts_with(&gc) && !gc.starts_with(&gr));

    // 2. The `default` tenant is the flat store — byte-identical to no prefix,
    //    so a single-tenant deployment is unchanged.
    let h_default = make(&Tenant::default(), "default");
    assert_eq!(h_default.global.clone().unwrap(), base);

    // 3. Execute the SAME graph through the real compiler/executor path under
    //    both tenant contexts. The shared cache base is identical; only the
    //    tenant namespace differs.
    let mut reg = Registry::new();
    reg.register(Box::new(TestCookbook));
    blut::checks::register(&mut reg);

    let mut research_ctx = ExecCtx::new(job.join("research")).with_tenant(research.clone());
    research_ctx.cache = Arc::new(h_research);
    let research_result = SequentialExecutor::execute(
        spec().compile(&reg).expect("research plan compiles"),
        research_ctx,
    )
    .await
    .expect("research plan executes");

    let mut clinical_ctx = ExecCtx::new(job.join("clinical")).with_tenant(clinical.clone());
    clinical_ctx.cache = Arc::new(h_clinical);
    let clinical_result = SequentialExecutor::execute(
        spec().compile(&reg).expect("clinical plan compiles"),
        clinical_ctx,
    )
    .await
    .expect("clinical plan executes");

    // If clinical could read research's namespace, its identical second run
    // would report cache hits. Both stages must instead execute in each tenant.
    assert_eq!(
        (research_result.n_cache_hits, research_result.n_cache_misses),
        (0, 2)
    );
    assert_eq!(
        (clinical_result.n_cache_hits, clinical_result.n_cache_misses),
        (0, 2)
    );

    // ADR-0078 keys remain tenant-independent: the two real executions create
    // the same key names below different tenant roots.
    let entry_names = |root: &std::path::Path| {
        std::fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<std::collections::BTreeSet<_>>()
    };
    let research_keys = entry_names(&gr);
    let clinical_keys = entry_names(&gc);
    assert_eq!(research_keys.len(), 2, "both graph stages must materialize");
    assert_eq!(
        research_keys, clinical_keys,
        "same graph must keep the same keys"
    );
}

#[cfg(feature = "p2p")]
#[tokio::test]
async fn restricted_tenant_is_refused_by_real_coordinator_submit() {
    use blut::framework::executor::{DispatchRequest, DispatchSubmitter, ResourceRequest};
    use blut::p2p::Coordinator;
    use blut::p2p::crypto::KeyPair;
    use blut::p2p::dispatch::DefaultDispatchPolicy;
    use blut::p2p::registry::PeerRegistry;
    use blut::trust::DispatchMatrix;

    let td = tempfile::tempdir().unwrap();
    let registry = PeerRegistry::load(&td.path().join("peers.json")).unwrap();
    let coordinator = Coordinator::start(
        "127.0.0.1:0".parse().unwrap(),
        Arc::new(KeyPair::generate()),
        Arc::new(DefaultDispatchPolicy::new(DispatchMatrix::default())),
        registry,
    )
    .await
    .unwrap();

    let tenant = Tenant::parse("clinical/prod").unwrap();
    let args = serde_json::json!({"payload": "patient-name-must-not-leak"});
    let request = DispatchRequest {
        stage_name: "warm_fb_cache",
        stage_schema: 1,
        input_hash: ContentHash::of_bytes(b"input"),
        args_hash: ContentHash::of_bytes(b"args"),
        args: &args,
        expected_output_hash: ContentHash::of_bytes(b"output"),
        resource_request: ResourceRequest::default(),
        data_class: 0,
        tenant: &tenant,
    };
    let refusal = match DispatchSubmitter::submit(&coordinator, request) {
        Ok(_) => panic!("clinical work reached the real coordinator dispatch path"),
        Err(error) => error,
    };
    let visible = refusal.to_string();
    assert!(visible.contains("P2P dispatch DENIED"));
    assert!(!visible.contains("patient-name-must-not-leak"));

    let research = Tenant::parse("research/dev").unwrap();
    let restricted_request = DispatchRequest {
        stage_name: "warm_fb_cache",
        stage_schema: 1,
        input_hash: ContentHash::of_bytes(b"input"),
        args_hash: ContentHash::of_bytes(b"args"),
        args: &args,
        expected_output_hash: ContentHash::of_bytes(b"output"),
        resource_request: ResourceRequest::default(),
        data_class: 2,
        tenant: &research,
    };
    let refusal = match DispatchSubmitter::submit(&coordinator, restricted_request) {
        Ok(_) => panic!("Restricted data reached the real coordinator dispatch path"),
        Err(error) => error,
    };
    let visible = refusal.to_string();
    assert!(visible.contains("P2P dispatch DENIED"));
    assert!(!visible.contains("patient-name-must-not-leak"));
    coordinator.shutdown();
}

// ── registry cross-tenant boundary keyed by Tenant (ties 0096 ↔ 0085) ──

struct TestSource;
#[async_trait]
impl Stage for TestSource {
    const NAME: &'static str = "test_source";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = JsonlArtifact;
    type Args = ();
    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        _args: &(),
    ) -> Result<JsonlArtifact, StageError> {
        std::fs::create_dir_all(&ctx.stage_dir).ok();
        let path = ctx.stage_dir.join("src.jsonl");
        std::fs::write(&path, b"{}\n")
            .map_err(|e| StageError::Backend(anyhow::anyhow!("src write: {e}")))?;
        let content_hash = ContentHash::hash_file(&path)
            .map_err(|e| StageError::Backend(anyhow::anyhow!("src hash: {e}")))?;
        Ok(JsonlArtifact { path, content_hash })
    }
}

struct TestCookbook;
impl Cookbook for TestCookbook {
    fn name(&self) -> &'static str {
        "tenant-test"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static S: &[(&str, ErasedStageCtor)] = &[("test_source", || Arc::new(TestSource))];
        S
    }
}

fn spec() -> PlanSpec {
    PlanSpec {
        name: "t".into(),
        nodes: vec![
            SpecNode {
                stage: "test_source".into(),
                args: serde_json::Value::Null,
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            },
            SpecNode {
                stage: "check_jsonl".into(),
                args: serde_json::json!({ "min_rows": 1 }),
                retry: None,
                timeout: None,
                priority: None,
                pure: false,
            },
        ],
        edges: vec![(0, 1)],
        condition_gates: Vec::new(),
        expansions: Vec::new(),
        version: 1,
    }
}

#[test]
fn clinical_fingerprint_never_promotes_onto_shared_pointer() {
    let mut reg = Registry::new();
    reg.register(Box::new(TestCookbook));
    blut::checks::register(&mut reg);

    let td = tempfile::tempdir().unwrap();
    let mut conn = registry_db::open_at(&td.path().join("registry.db")).unwrap();

    let clinical = Tenant::parse("clinical/prod").unwrap();
    let research = Tenant::parse("research/prod").unwrap();
    assert!(clinical.is_restricted() && !research.is_restricted());

    // Publish under the clinical tenant (keyed by Tenant::to_string()).
    let fp = registry_db::publish(
        &conn,
        &reg,
        &spec(),
        "tester",
        &clinical.to_string(),
        None,
        1000,
    )
    .unwrap();

    // Promoting a clinical fingerprint onto a research (shared) pointer is a
    // hard deny (ADR 0061), and the research pointer is never written.
    assert!(
        registry_db::promote(&mut conn, &fp, &research.to_string(), "prod", 1001).is_err(),
        "a clinical fingerprint must not promote onto a research pointer"
    );
    assert_eq!(
        registry_db::resolve_pointer(&conn, &research.to_string(), "prod").unwrap(),
        None
    );
    // …but it promotes fine onto its OWN clinical pointer.
    registry_db::promote(&mut conn, &fp, &clinical.to_string(), "prod", 1002).unwrap();
    assert_eq!(
        registry_db::resolve_pointer(&conn, &clinical.to_string(), "prod").unwrap(),
        Some(fp)
    );
}
