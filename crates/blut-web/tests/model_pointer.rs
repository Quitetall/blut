// SPDX-License-Identifier: AGPL-3.0-or-later
// `GET /api/models/{name}/{alias}` serves the tenant `blut model` writes to.
//
// E2E regression: the view only ever read the `shared` tenant, while `blut
// model register/promote` default to `default`, so every pointer created with
// the CLI's defaults was a 404 on the dashboard. A restricted tenant must stay
// unexported.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

fn state() -> blut_web::AppState {
    blut_web::AppState {
        tokens: None,
        lineage_path: None,
        cli: std::path::PathBuf::from("/bin/false"),
        audit_path: std::env::temp_dir().join("model-pointer-audit.jsonl"),
        triggers: Default::default(),
    }
}

async fn get_json(app: axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let res = app
        .oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_default())
}

#[tokio::test]
async fn model_pointer_serves_the_cli_default_tenant_and_hides_restricted() {
    let td = tempfile::tempdir().unwrap();
    // The only test in this binary, so mutating the process env is race-free.
    unsafe { std::env::set_var("BLUT_REGISTRY_DB", td.path().join("registry.db")) };

    let hash_default = "a".repeat(64);
    let hash_shared = "b".repeat(64);
    let hash_restricted = "c".repeat(64);
    let mut conn = blut::model_registry::open().unwrap();
    for (hash, tenant) in [
        (&hash_default, "default"),
        (&hash_shared, "shared"),
        (&hash_restricted, "restricted"),
    ] {
        blut::model_registry::register(&conn, hash, "demo", tenant, None, 1).unwrap();
        blut::model_registry::promote(&mut conn, hash, tenant, "demo", "staging", 2).unwrap();
    }
    drop(conn);

    let app = blut_web::build_router(state());
    let (status, body) = get_json(app.clone(), "/api/models/demo/staging").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["hash"], hash_default.as_str());
    assert_eq!(body["tenant"], "default");

    let (status, body) = get_json(app.clone(), "/api/models/demo/staging?tenant=shared").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["hash"], hash_shared.as_str());

    let (status, body) = get_json(app.clone(), "/api/models/demo/staging?tenant=restricted").await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "restricted is never exported: {body}"
    );
    assert!(!body.to_string().contains(&hash_restricted));

    let (status, _) = get_json(app, "/api/models/demo/staging?tenant=a%20b").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
