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
                pure: false,
            },
            SpecNode {
                stage: "check_jsonl".into(),
                args: serde_json::json!({ "min_rows": min_rows }),
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
            pure: false,
        }],
        edges: vec![],
        condition_gates: Vec::new(),
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

/// A pointer published under the former `shared` default is not found under
/// `default` (the tenant every verb now defaults to); the error says which
/// tenant has it, and never names a restricted one.
#[test]
fn a_missing_pointer_names_the_tenant_that_has_it() {
    let reg = test_registry();
    let td = tempfile::tempdir().unwrap();
    let mut conn = registry_db::open_at(&td.path().join("registry.db")).unwrap();
    let a = registry_db::publish(&conn, &reg, &valid_spec(1), "t", "shared", None, 1).unwrap();
    registry_db::promote(&mut conn, &a, "shared", "prod", 2).unwrap();
    let err = registry_db::resolve_spec(&conn, "default", "prod")
        .unwrap_err()
        .to_string();
    assert!(err.contains("for tenant 'default'"), "{err}");
    assert!(err.contains("pass --tenant shared"), "{err}");

    let r = registry_db::publish(
        &conn,
        &reg,
        &valid_spec(2),
        "t",
        registry_db::RESTRICTED_TENANT,
        None,
        3,
    )
    .unwrap();
    registry_db::promote(&mut conn, &r, registry_db::RESTRICTED_TENANT, "clin", 4).unwrap();
    let err = registry_db::resolve_spec(&conn, "default", "clin")
        .unwrap_err()
        .to_string();
    assert!(!err.contains("restricted"), "{err}");
}

/// `plan rollback` walks the promotion history like an undo stack. It used to
/// read the two most recent trail entries, so repeated rollbacks toggled
/// between the last two deployments.
#[test]
fn rollback_walks_the_promotion_history() {
    let reg = test_registry();
    let td = tempfile::tempdir().unwrap();
    let mut conn = registry_db::open_at(&td.path().join("registry.db")).unwrap();
    let t = "default";
    let [a, b, c, d] = [1, 2, 3, 4].map(|n| {
        registry_db::publish(&conn, &reg, &valid_spec(n), "t", t, None, n as i64).unwrap()
    });
    let at = |conn: &rusqlite::Connection| registry_db::resolve_pointer(conn, t, "p").unwrap();

    // Nothing to roll back: no deployment, then a single promotion.
    let err = registry_db::rollback(&mut conn, t, "p", 10)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no deployment"), "{err}");
    registry_db::promote(&mut conn, &a, t, "p", 11).unwrap();
    let err = registry_db::rollback(&mut conn, t, "p", 12)
        .unwrap_err()
        .to_string();
    assert!(err.contains("nothing earlier"), "{err}");
    assert_eq!(
        at(&conn),
        Some(a.clone()),
        "a refused rollback moves nothing"
    );

    // A, B, C then two rollbacks: C → B → A, then refused.
    registry_db::promote(&mut conn, &b, t, "p", 13).unwrap();
    registry_db::promote(&mut conn, &c, t, "p", 14).unwrap();
    assert_eq!(registry_db::rollback(&mut conn, t, "p", 15).unwrap(), b);
    assert_eq!(registry_db::rollback(&mut conn, t, "p", 16).unwrap(), a);
    assert_eq!(at(&conn), Some(a.clone()));
    assert!(registry_db::rollback(&mut conn, t, "p", 17).is_err());

    // A promotion after a rollback continues from there: [A, D]; rolling back
    // returns to A, and B/C are not reachable by rollback any more.
    registry_db::promote(&mut conn, &d, t, "p", 18).unwrap();
    assert_eq!(registry_db::rollback(&mut conn, t, "p", 19).unwrap(), a);
    assert!(registry_db::rollback(&mut conn, t, "p", 20).is_err());

    // Re-promoting the current target is recorded but is not a new step.
    registry_db::promote(&mut conn, &b, t, "p", 21).unwrap();
    registry_db::promote(&mut conn, &b, t, "p", 22).unwrap();
    assert_eq!(registry_db::rollback(&mut conn, t, "p", 23).unwrap(), a);

    // Every move stays in the audit trail: 6 promotions and 4 rollbacks
    // (refused rollbacks record nothing).
    assert_eq!(registry_db::history(&conn, t, "p").unwrap().len(), 10);
}

/// A registry created before rollbacks were recorded as such gains the column
/// on open; its old moves count as promotions.
#[test]
fn an_old_registry_gains_the_move_kind_on_open() {
    let td = tempfile::tempdir().unwrap();
    let path = td.path().join("registry.db");
    {
        let old = rusqlite::Connection::open(&path).unwrap();
        old.execute_batch(
            "CREATE TABLE pointer_history (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, tenant TEXT NOT NULL,
                name TEXT NOT NULL, plan_fingerprint TEXT NOT NULL,
                moved_at INTEGER NOT NULL);
             INSERT INTO pointer_history (tenant, name, plan_fingerprint, moved_at)
                VALUES ('default', 'p', 'aa', 1), ('default', 'p', 'bb', 2);",
        )
        .unwrap();
    }
    let conn = registry_db::open_at(&path).unwrap();
    assert_eq!(
        registry_db::history(&conn, "default", "p").unwrap().len(),
        2
    );
    // Reopening is idempotent.
    drop(conn);
    registry_db::open_at(&path).unwrap();
}
