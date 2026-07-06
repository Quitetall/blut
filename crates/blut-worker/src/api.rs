// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! REST API for the BLUT cloud worker.
//!
//! Endpoints:
//!   POST /jobs          — submit a new job
//!   GET  /jobs          — list all jobs
//!   GET  /jobs/:id      — get job status
//!   GET  /health        — health check

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
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

/// Build the API router.
pub fn router(state: ApiState) -> Router {
    Router::new()
        .route("/jobs", post(submit_job).get(list_jobs))
        .route("/jobs/{id}", get(get_job))
        .route("/health", get(health))
        .with_state(state)
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
    let content = serde_json::to_string_pretty(&job).unwrap();

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
    // Count queue depth
    let queue_depth = fs::read_dir(&state.queue_dir)
        .await
        .map(|_entries| {
            // TODO: count entries asynchronously
            0
        })
        .unwrap_or(0);

    Json(HealthResponse {
        status: "ok".to_string(),
        worker_id: "worker-001".to_string(),
        queue_depth,
    })
}
