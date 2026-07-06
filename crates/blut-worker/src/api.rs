// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! REST API for the BLUT cloud worker.
//!
//! Endpoints:
//!   POST /jobs          — submit a new job          (bearer-token protected)
//!   GET  /jobs          — list all jobs             (bearer-token protected)
//!   GET  /jobs/:id      — get job status            (bearer-token protected)
//!   GET  /health        — health check              (unauthenticated)
//!
//! SECURITY: `POST /jobs` executes recipes — arbitrary code. When a token
//! is configured (`BLUT_WORKER_TOKEN`), every job route requires
//! `Authorization: Bearer <token>`. Running token-less is only permitted
//! on a loopback bind (enforced at startup in `main.rs`, fail-closed).

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::sync::RwLock;

/// Shared API state.
#[derive(Clone)]
pub struct ApiState {
    pub queue_dir: PathBuf,
    pub results_dir: PathBuf,
    pub jobs: Arc<RwLock<Vec<JobStatus>>>,
    /// Worker identity reported by `/health` (the real one, not a placeholder).
    pub worker_id: String,
    /// Bearer token required on the job routes. `None` = token-less
    /// loopback-only mode (main.rs refuses non-loopback binds without it).
    pub token: Option<Arc<str>>,
}

/// Job submission request.
#[derive(Debug, Deserialize)]
pub struct SubmitJob {
    pub id: Option<String>,
    pub recipe: String,
    #[serde(default)]
    pub args: serde_json::Value,
    #[serde(default)]
    pub resources: ResourceRequest,
}

#[derive(Debug, Default, Deserialize)]
pub struct ResourceRequest {
    #[serde(default)]
    pub gpu: bool,
    #[serde(default)]
    pub memory_gib: u32,
    #[serde(default)]
    pub cpu_cores: u32,
}

/// Job status response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatus {
    pub id: String,
    pub recipe: String,
    pub status: String, // "queued", "processing", "succeeded", "failed"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
}

/// Health check response.
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub worker_id: String,
    pub queue_depth: usize,
}

/// Build the API router. Job routes sit behind the bearer-token layer;
/// `/health` stays open (it exposes only liveness + queue depth).
pub fn router(state: ApiState) -> Router {
    let protected = Router::new()
        .route("/jobs", post(submit_job).get(list_jobs))
        .route("/jobs/{id}", get(get_job))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            require_bearer,
        ));
    Router::new()
        .merge(protected)
        .route("/health", get(health))
        .with_state(state)
}

/// Middleware: when a token is configured, demand a matching
/// `Authorization: Bearer <token>` header on every protected route.
async fn require_bearer(State(state): State<ApiState>, req: Request, next: Next) -> Response {
    let Some(expected) = state.token.as_deref() else {
        // Token-less mode — main.rs only allows this on a loopback bind.
        return next.run(req).await;
    };
    let presented = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match presented {
        Some(t) if constant_time_eq(t.as_bytes(), expected.as_bytes()) => next.run(req).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "missing or invalid bearer token"})),
        )
            .into_response(),
    }
}

/// Constant-time comparison with no short-circuit on EITHER content or
/// length: iterate over the longer input (zero-padding the shorter) and
/// OR the length difference into the accumulator, so timing reveals
/// neither a matching prefix nor the token's length.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= (x ^ y) as usize;
    }
    diff == 0
}

/// POST /jobs — submit a new job.
async fn submit_job(
    State(state): State<ApiState>,
    Json(req): Json<SubmitJob>,
) -> impl IntoResponse {
    let job_id = req.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let job = serde_json::json!({
        "id": job_id,
        "recipe": req.recipe,
        "args": req.args,
        "resources": {
            "gpu": req.resources.gpu,
            "memory_gib": req.resources.memory_gib,
            "cpu_cores": req.resources.cpu_cores,
        }
    });

    let job_path = state.queue_dir.join(format!("{job_id}.json"));
    let content = match serde_json::to_string_pretty(&job) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("failed to serialize job: {e}")})),
            )
                .into_response();
        }
    };

    if let Err(e) = fs::write(&job_path, &content).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("failed to write job: {e}")})),
        )
            .into_response();
    }

    // Track in memory
    state.jobs.write().await.push(JobStatus {
        id: job_id.clone(),
        recipe: req.recipe,
        status: "queued".to_string(),
        result: None,
    });

    (
        StatusCode::CREATED,
        Json(serde_json::json!({"id": job_id, "status": "queued"})),
    )
        .into_response()
}

/// GET /jobs — list all jobs.
async fn list_jobs(State(state): State<ApiState>) -> impl IntoResponse {
    let jobs = state.jobs.read().await;
    Json(jobs.clone()).into_response()
}

/// GET /jobs/:id — get job status.
async fn get_job(State(state): State<ApiState>, Path(id): Path<String>) -> impl IntoResponse {
    // Check in-memory list first
    let jobs = state.jobs.read().await;
    if let Some(job) = jobs.iter().find(|j| j.id == id) {
        return Json(job.clone()).into_response();
    }

    // Check results directory
    let result_path = state.results_dir.join(format!("{id}.json"));
    if result_path.exists()
        && let Ok(content) = fs::read_to_string(&result_path).await
        && let Ok(result) = serde_json::from_str::<serde_json::Value>(&content)
    {
        let status = result["status"].as_str().unwrap_or("unknown");
        return Json(JobStatus {
            id: id.clone(),
            recipe: String::new(),
            status: status.to_string(),
            result: Some(result),
        })
        .into_response();
    }

    // Check queue directory
    let queue_path = state.queue_dir.join(format!("{id}.json"));
    if queue_path.exists() {
        return Json(JobStatus {
            id: id.clone(),
            recipe: String::new(),
            status: "queued".to_string(),
            result: None,
        })
        .into_response();
    }

    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "job not found"})),
    )
        .into_response()
}

/// GET /health — health check.
async fn health(State(state): State<ApiState>) -> impl IntoResponse {
    // Count pending .json job files (an unreadable dir reports depth 0).
    let queue_depth = match fs::read_dir(&state.queue_dir).await {
        Ok(mut entries) => {
            let mut n = 0usize;
            while let Ok(Some(entry)) = entries.next_entry().await {
                if entry.path().extension().is_some_and(|e| e == "json") {
                    n += 1;
                }
            }
            n
        }
        Err(_) => 0,
    };

    Json(HealthResponse {
        status: "ok".to_string(),
        worker_id: state.worker_id.clone(),
        queue_depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode, header};
    use tower::util::ServiceExt;

    fn state(token: Option<&str>, dir: &std::path::Path) -> ApiState {
        ApiState {
            queue_dir: dir.to_path_buf(),
            results_dir: dir.to_path_buf(),
            jobs: Arc::new(RwLock::new(Vec::new())),
            worker_id: "test-worker".into(),
            token: token.map(Arc::from),
        }
    }

    fn get(uri: &str, bearer: Option<&str>) -> HttpRequest<Body> {
        let mut b = HttpRequest::builder().uri(uri);
        if let Some(t) = bearer {
            b = b.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        b.body(Body::empty()).unwrap()
    }

    #[tokio::test]
    async fn job_routes_reject_without_token() {
        let td = tempfile::tempdir().unwrap();
        let app = router(state(Some("s3cret"), td.path()));
        let res = app.oneshot(get("/jobs", None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn job_routes_reject_wrong_token() {
        let td = tempfile::tempdir().unwrap();
        let app = router(state(Some("s3cret"), td.path()));
        let res = app.oneshot(get("/jobs", Some("wrong"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn job_routes_accept_correct_token() {
        let td = tempfile::tempdir().unwrap();
        let app = router(state(Some("s3cret"), td.path()));
        let res = app.oneshot(get("/jobs", Some("s3cret"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_is_open_even_with_token() {
        let td = tempfile::tempdir().unwrap();
        let app = router(state(Some("s3cret"), td.path()));
        let res = app.oneshot(get("/health", None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn tokenless_mode_allows_job_routes() {
        // Loopback-only mode (main.rs enforces the bind restriction).
        let td = tempfile::tempdir().unwrap();
        let app = router(state(None, td.path()));
        let res = app.oneshot(get("/jobs", None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    fn post_job(bearer: Option<&str>) -> HttpRequest<Body> {
        let mut b = HttpRequest::builder()
            .method("POST")
            .uri("/jobs")
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(t) = bearer {
            b = b.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        b.body(Body::from(r#"{"recipe": "noop"}"#)).unwrap()
    }

    #[tokio::test]
    async fn post_jobs_rejected_without_valid_token() {
        // POST /jobs is the recipe-execution (arbitrary-code) route — the
        // one the auth layer exists for. Both missing and wrong tokens
        // must 401 BEFORE the handler runs (no job file written).
        let td = tempfile::tempdir().unwrap();
        let app = router(state(Some("s3cret"), td.path()));
        let res = app.clone().oneshot(post_job(None)).await.unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let res = app.oneshot(post_job(Some("wrong"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        let queued = std::fs::read_dir(td.path()).unwrap().count();
        assert_eq!(queued, 0, "rejected submissions must not enqueue a job");
    }

    #[tokio::test]
    async fn post_jobs_accepted_with_correct_token() {
        let td = tempfile::tempdir().unwrap();
        let app = router(state(Some("s3cret"), td.path()));
        let res = app.oneshot(post_job(Some("s3cret"))).await.unwrap();
        assert_eq!(res.status(), StatusCode::CREATED);
        let queued = std::fs::read_dir(td.path()).unwrap().count();
        assert_eq!(queued, 1, "accepted submission writes one queue file");
    }

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
        assert!(!constant_time_eq(b"abc\0", b"abc"));
        assert!(constant_time_eq(b"", b""));
    }
}
