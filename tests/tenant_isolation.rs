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

#[cfg(feature = "p2p")]
#[test]
fn restricted_mesh_and_privacy_boundaries_refuse_cross_node_or_cross_tenant_use() {
    use blut::p2p::crypto::KeyPair;
    use blut::p2p::privacy::PrivacyLedger;
    use blut::p2p::trust::{DataClass, DispatchMatrix, TrustLevel};

    let mut matrix = DispatchMatrix::default();
    matrix.set(DataClass::Restricted, TrustLevel::Trusted, true);
    assert!(
        !matrix.can_dispatch(DataClass::Restricted, TrustLevel::Trusted),
        "Restricted must stay node-local even under a custom trust matrix"
    );

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

#[test]
fn tenant_cache_namespacing_is_disjoint_and_key_stable() {
    let td = tempfile::tempdir().unwrap();
    let base = td.path().join("cache");
    let job = td.path().join("job");

    let research = Tenant::parse("research/dev").unwrap();
    let clinical = Tenant::parse("clinical/prod").unwrap();

    let make = |t: &Tenant| {
        CacheHandle::job_local(job.clone())
            .with_global(base.clone())
            .with_tenant(t)
    };
    let (h_research, h_clinical) = (make(&research), make(&clinical));

    // 1. Disjoint global roots, neither an ancestor of the other.
    let gr = h_research.global.clone().unwrap();
    let gc = h_clinical.global.clone().unwrap();
    assert_ne!(gr, gc);
    assert!(gr.ends_with("research/dev") && gc.ends_with("clinical/prod"));
    assert!(!gr.starts_with(&gc) && !gc.starts_with(&gr));

    // 2. The `default` tenant is the flat store — byte-identical to no prefix,
    //    so a single-tenant deployment is unchanged.
    let h_default = make(&Tenant::default());
    assert_eq!(h_default.global.clone().unwrap(), base);

    // 3. The ADR-0078 key is tenant-INDEPENDENT: the SAME graph hashes to the
    //    SAME key under both tenants; only the parent dir differs.
    let key = CacheHandle::key_for(
        "stage",
        1,
        ContentHash([7u8; 32]),
        &serde_json::json!({ "x": 1 }),
        b"code",
    );
    let hex = key.to_hex();
    let entry_research = gr.join(&hex);
    let entry_clinical = gc.join(&hex);
    assert_ne!(entry_research, entry_clinical, "same key, disjoint roots");

    // 4. Cross-tenant read denied by construction: an entry written under
    //    research's root is not visible under clinical's.
    std::fs::create_dir_all(&entry_research).unwrap();
    std::fs::write(entry_research.join("output.bin"), b"research-only").unwrap();
    assert!(
        !entry_clinical.join("output.bin").exists(),
        "clinical must never see research's cache entry"
    );
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
            },
            SpecNode {
                stage: "check_jsonl".into(),
                args: serde_json::json!({ "min_rows": 1 }),
                retry: None,
                timeout: None,
                priority: None,
            },
        ],
        edges: vec![(0, 1)],
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
