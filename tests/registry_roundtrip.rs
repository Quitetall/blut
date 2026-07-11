// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0085 acceptance gate: the plan deployment registry round-trips
//! publish → promote → rollback and enforces its two fail-closed boundaries
//! (typecheck-on-publish, cross-tenant promote refusal).

use std::sync::Arc;

use async_trait::async_trait;
use blut::checks::JsonlArtifact;
use blut::framework::Registry;
use blut::framework::artifact::ContentHash;
use blut::framework::cookbook::Cookbook;
use blut::framework::error::StageError;
use blut::framework::plan_spec::{PlanSpec, SpecNode};
use blut::framework::resource::Resource;
use blut::framework::stage::{ErasedStageCtor, Stage, StageContext};
use blut::recipes::recipe::RecipeDef;
use blut::registry_db;

/// A minimal registered graph-source (`input = ()`) so a published PlanSpec is
/// kind-valid: `test_source → check_jsonl`. Emits an empty JSONL file.
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
        "registry-test"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static S: &[(&str, ErasedStageCtor)] = &[("test_source", || Arc::new(TestSource))];
        S
    }
}

/// A kind-valid spec: `test_source → check_jsonl(min_rows)`. `min_rows` varies
/// the canonical bytes so two specs get distinct fingerprints.
fn valid_spec(min_rows: u64) -> PlanSpec {
    PlanSpec {
        name: "deploy-me".into(),
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
                args: serde_json::json!({ "min_rows": min_rows }),
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

fn test_registry() -> Registry {
    let mut reg = Registry::new();
    reg.register(Box::new(TestCookbook));
    blut::checks::register(&mut reg); // brings in `check_jsonl`
    reg
}

#[test]
fn registry_roundtrip() {
    let reg = test_registry();
    let td = tempfile::tempdir().unwrap();
    let mut conn = registry_db::open_at(&td.path().join("registry.db")).unwrap();

    // Publish A, promote it onto registry://plan@test.
    let a =
        registry_db::publish(&conn, &reg, &valid_spec(1), "tester", "shared", None, 1000).unwrap();
    registry_db::promote(&mut conn, &a, "shared", "test", 1001).unwrap();
    assert_eq!(
        registry_db::resolve_pointer(&conn, "shared", "test").unwrap(),
        Some(a.clone())
    );

    // Idempotent: re-publishing identical bytes returns the same fingerprint.
    let a2 =
        registry_db::publish(&conn, &reg, &valid_spec(1), "tester", "shared", None, 1002).unwrap();
    assert_eq!(
        a, a2,
        "identical bytes ⇒ identical fingerprint (idempotent)"
    );

    // Publish B (distinct spec), promote it — pointer now resolves to B.
    let b =
        registry_db::publish(&conn, &reg, &valid_spec(2), "tester", "shared", None, 2000).unwrap();
    assert_ne!(a, b, "distinct specs ⇒ distinct fingerprints");
    registry_db::promote(&mut conn, &b, "shared", "test", 2001).unwrap();
    assert_eq!(
        registry_db::resolve_pointer(&conn, "shared", "test").unwrap(),
        Some(b.clone())
    );

    // Rollback: the pointer pops back to A, and resolve_spec loads that graph.
    let rolled = registry_db::rollback(&mut conn, "shared", "test", 3000).unwrap();
    assert_eq!(
        rolled, a,
        "rollback resolves to the exact prior fingerprint"
    );
    let spec = registry_db::resolve_spec(&conn, "shared", "test").unwrap();
    assert_eq!(registry_db::fingerprint(&spec), a);

    // History records every move (A promote, B promote, rollback→A).
    let hist = registry_db::history(&conn, "shared", "test").unwrap();
    assert_eq!(
        hist.iter()
            .map(|h| h.plan_fingerprint.clone())
            .collect::<Vec<_>>(),
        vec![a.clone(), b.clone(), a.clone()]
    );

    // Boundary 1 — typecheck-on-publish is fail-closed: a spec naming an
    // unregistered stage never enters `deployments`.
    let broken = PlanSpec {
        name: "broken".into(),
        nodes: vec![SpecNode {
            stage: "does_not_exist".into(),
            args: serde_json::Value::Null,
            retry: None,
            timeout: None,
            priority: None,
        }],
        edges: vec![],
        expansions: Vec::new(),
        version: 1,
    };
    assert!(
        registry_db::publish(&conn, &reg, &broken, "tester", "shared", None, 4000).is_err(),
        "a non-typechecking PlanSpec must be refused at publish"
    );

    // Boundary 2 — a Restricted-tenant fingerprint can never be promoted onto
    // another tenant's pointer (ADR 0061 clinical hard-block).
    let restricted = registry_db::publish(
        &conn,
        &reg,
        &valid_spec(9),
        "tester",
        registry_db::RESTRICTED_TENANT,
        None,
        5000,
    )
    .unwrap();
    assert!(
        registry_db::promote(&mut conn, &restricted, "shared", "prod", 5001).is_err(),
        "a restricted deployment must not cross onto a shared pointer"
    );
    // …and it never touched the shared pointer.
    assert_eq!(
        registry_db::resolve_pointer(&conn, "shared", "prod").unwrap(),
        None
    );

    // Boundary 3 — the SAME spec bytes cannot be re-published under a different
    // tenant (the content fingerprint would otherwise bind to one tenant's row).
    assert!(
        registry_db::publish(&conn, &reg, &valid_spec(9), "tester", "shared", None, 6000).is_err(),
        "a restricted deployment's bytes must not re-publish as shared"
    );

    // The deploy-URI parser only accepts safe identifier names.
    assert_eq!(
        registry_db::parse_pointer_uri("registry://plan@prod"),
        Some("prod")
    );
    assert_eq!(
        registry_db::parse_pointer_uri("registry://plan@a.b-c_1"),
        Some("a.b-c_1")
    );
    assert_eq!(
        registry_db::parse_pointer_uri("registry://plan@../etc"),
        None
    );
    assert_eq!(registry_db::parse_pointer_uri("registry://plan@"), None);
    assert_eq!(registry_db::parse_pointer_uri("/some/file.json"), None);
}
