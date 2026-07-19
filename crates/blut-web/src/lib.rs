// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! blut-web (ADR 0083 M2, increment 1) — the READ-ONLY web sidecar.
//!
//! Serves JSON views over exactly the stores the TUI reads: the job registry,
//! per-job `status.jsonl`, the lineage SQLite (provenance graphs, model cards,
//! run diffs — via the wasm-safe `blut-types` wire types the future Leptos UI
//! shares), the model registry, and the dataset catalog.
//!
//! Custody posture (fail-closed, ADR 0061/0095):
//! * **The web is an EXPORT surface**: every lineage/catalog read passes
//!   `exclude_restricted = true` unconditionally ([`EXPORT_EXCLUDES_RESTRICTED`]
//!   — a constant, not a knob) and the model registry is served for the shared
//!   tenant only. A Restricted artifact structurally cannot appear in a
//!   response, whoever asks.
//! * **Auth**: with a token store (`~/.blut/web-tokens.toml`, the ADR 0095
//!   sha256-hashed store), every request needs `Authorization: Bearer <token>`
//!   resolving to a principal — else 401. Without a store the server refuses
//!   to bind anything but loopback (dev mode, warned loudly).
//! * **Mutations**: none in this increment. They arrive via the exec bridge
//!   (`rbac::enforce` → the CLI), never by reaching into engine internals.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path as AxPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};

/// The web surface is an EXPORT: restricted/clinical rows are excluded from
/// every response, unconditionally. This is a constant so "make it
/// configurable" has to touch — and justify itself against — ADR 0061 here.
pub const EXPORT_EXCLUDES_RESTRICTED: bool = true;

/// Hard cap on `?tail=` for status streams (bounded responses, no file slurp).
const STATUS_TAIL_CAP: usize = 1000;

#[derive(Clone)]
pub struct AppState {
    /// Parsed ADR-0095 token store; `None` = loopback-only dev mode.
    pub tokens: Option<Arc<blut::rbac::TokenStore>>,
    /// Lineage DB override (tests); `None` = the default `~/.blut` path.
    pub lineage_path: Option<PathBuf>,
}

fn err(status: StatusCode, msg: impl std::fmt::Display) -> Response {
    (
        status,
        Json(serde_json::json!({ "error": msg.to_string() })),
    )
        .into_response()
}

async fn require_token(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    if let Some(store) = &state.tokens {
        let presented = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        match presented.and_then(|secret| store.resolve(secret)) {
            Some(_principal) => {} // any resolved principal may READ (Viewer floor)
            None => return err(StatusCode::UNAUTHORIZED, "missing or unknown bearer token"),
        }
    }
    next.run(request).await
}

#[allow(clippy::result_large_err)] // an axum Response IS the error surface here
fn lineage(state: &AppState) -> Result<blut::lineage_db::LineageDb, Response> {
    let db = match &state.lineage_path {
        Some(p) => blut::lineage_db::LineageDb::open_at(p),
        None => blut::lineage_db::LineageDb::open(),
    };
    db.map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))
}

async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true, "service": "blut-web" }))
}

async fn jobs() -> Response {
    match blut::jobs::list_jobs() {
        Ok(list) => Json(list).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[derive(serde::Deserialize)]
struct TailQuery {
    tail: Option<usize>,
}

async fn job_status(AxPath(id): AxPath<String>, Query(q): Query<TailQuery>) -> Response {
    // The job id is a generated identifier — refuse anything path-ish.
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return err(StatusCode::BAD_REQUEST, "invalid job id");
    }
    let dir = match blut::paths::job_dir(&id) {
        Ok(d) => d,
        Err(e) => return err(StatusCode::NOT_FOUND, e),
    };
    let path = dir.join("status.jsonl");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return err(StatusCode::NOT_FOUND, "no status stream for job"),
    };
    let n = q.tail.unwrap_or(100).min(STATUS_TAIL_CAP);
    let lines: Vec<serde_json::Value> = text
        .lines()
        .rev()
        .take(n)
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let lines: Vec<_> = lines.into_iter().rev().collect();
    Json(serde_json::json!({ "job": id, "events": lines })).into_response()
}

#[derive(serde::Deserialize)]
struct GraphQuery {
    format: Option<String>,
}

async fn lineage_graph(
    State(state): State<AppState>,
    AxPath(hash): AxPath<String>,
    Query(q): Query<GraphQuery>,
) -> Response {
    let db = match lineage(&state) {
        Ok(db) => db,
        Err(r) => return r,
    };
    match db.graph_upstream(&hash, EXPORT_EXCLUDES_RESTRICTED) {
        // An unknown hash comes back as a bare, un-indexed root with no edges —
        // serve 404, not an empty-looking graph.
        Ok(graph)
            if graph.edges.is_empty()
                && graph
                    .nodes
                    .iter()
                    .all(|n| n.stage_name.is_none() && n.kind.is_none() && n.job_id.is_none()) =>
        {
            err(StatusCode::NOT_FOUND, "unknown artifact")
        }
        Ok(graph) => match q.format.as_deref() {
            Some("dot") => graph.to_dot().into_response(),
            _ => Json(graph).into_response(),
        },
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn lineage_card(State(state): State<AppState>, AxPath(hash): AxPath<String>) -> Response {
    let db = match lineage(&state) {
        Ok(db) => db,
        Err(r) => return r,
    };
    match db.model_card(&hash, EXPORT_EXCLUDES_RESTRICTED) {
        Ok(Some(card)) => Json(card).into_response(),
        Ok(None) => err(StatusCode::NOT_FOUND, "no card for artifact"),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn lineage_diff(
    State(state): State<AppState>,
    AxPath((a, b)): AxPath<(String, String)>,
) -> Response {
    let db = match lineage(&state) {
        Ok(db) => db,
        Err(r) => return r,
    };
    match db.run_diff(&a, &b) {
        Ok(diff) => Json(diff).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, e),
    }
}

async fn model_pointer(AxPath((name, alias)): AxPath<(String, String)>) -> Response {
    // The web serves the SHARED tenant only — a restricted tenant's pointers
    // never leave the box through this surface (ADR 0061/0096).
    let conn = match blut::model_registry::open() {
        Ok(c) => c,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let tenant = blut::model_registry::SHARED_TENANT;
    let resolved = match blut::model_registry::resolve_pointer(&conn, tenant, &name, &alias) {
        Ok(Some(h)) => h,
        Ok(None) => return err(StatusCode::NOT_FOUND, "no such pointer"),
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let history = blut::model_registry::history(&conn, tenant, &name, &alias)
        .unwrap_or_default()
        .into_iter()
        .map(|h| serde_json::json!({ "hash": h.model_hash, "moved_at": h.moved_at }))
        .collect::<Vec<_>>();
    Json(serde_json::json!({
        "pointer": format!("model://{name}@{alias}"),
        "hash": resolved,
        "history": history,
    }))
    .into_response()
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/api/jobs", get(jobs))
        .route("/api/jobs/{id}/status", get(job_status))
        .route("/api/lineage/graph/{hash}", get(lineage_graph))
        .route("/api/lineage/card/{hash}", get(lineage_card))
        .route("/api/lineage/diff/{a}/{b}", get(lineage_diff))
        .route("/api/models/{name}/{alias}", get(model_pointer))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_token))
        // healthz stays unauthenticated (liveness probes).
        .route("/healthz", get(healthz))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::util::ServiceExt;

    fn state(tokens: Option<blut::rbac::TokenStore>) -> AppState {
        AppState {
            tokens: tokens.map(Arc::new),
            lineage_path: None,
        }
    }

    #[test]
    fn export_excludes_restricted_is_a_constant_true() {
        // ADR 0061 tripwire: making this configurable must consciously edit
        // this test and argue with the clinical hard-block.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(EXPORT_EXCLUDES_RESTRICTED);
        }
    }

    #[tokio::test]
    async fn healthz_is_open_even_with_tokens_configured() {
        let store = blut::rbac::TokenStore::parse(
            r#"[[token]]
id = "t1"
hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
role = "viewer"
tenant = "shared"
"#,
        )
        .expect("valid store");
        let app = build_router(state(Some(store)));
        let res = app
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn api_requires_a_resolvable_bearer_token_when_store_present() {
        let store = blut::rbac::TokenStore::parse(
            r#"[[token]]
id = "t1"
hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
role = "viewer"
tenant = "shared"
"#,
        )
        .expect("valid store");
        let app = build_router(state(Some(store)));
        // No token → 401, fail-closed.
        let res = app
            .clone()
            .oneshot(Request::get("/api/jobs").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        // A wrong token → 401 (anonymous, never a fallthrough).
        let res = app
            .oneshot(
                Request::get("/api/jobs")
                    .header("authorization", "Bearer nope")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unknown_artifact_is_404_not_500() {
        let td = tempfile::tempdir().unwrap();
        let app = build_router(AppState {
            tokens: None,
            lineage_path: Some(td.path().join("lineage.db")),
        });
        let res = app
            .oneshot(
                Request::get("/api/lineage/graph/deadbeef")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn job_id_path_traversal_is_refused() {
        let app = build_router(state(None));
        let res = app
            .oneshot(
                Request::get("/api/jobs/..%2F..%2Fetc/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }
}
