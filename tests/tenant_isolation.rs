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
use blut::checks::JsonlArtifact;
use blut::framework::CacheHandle;
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
