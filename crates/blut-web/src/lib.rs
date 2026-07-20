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
//! * **Mutations (increment 2 — the exec bridge)**: `POST /api/jobs` (run a
//!   recipe) and `POST /api/jobs/{id}/cancel` pass `rbac::enforce` (Operator
//!   floor, allow AND deny audited to `audit.jsonl` BEFORE dispatch) and then
//!   shell out to the CLI — one enforcement path for human and API, never a
//!   reach into engine internals (ADR 0083 §4). Without a token store every
//!   mutation is anonymous ⇒ denied (viewer-only), even on loopback.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path as AxPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

/// The web surface is an EXPORT: restricted/clinical rows are excluded from
/// every response, unconditionally. This is a constant so "make it
/// configurable" has to touch — and justify itself against — ADR 0061 here.
pub const EXPORT_EXCLUDES_RESTRICTED: bool = true;

/// Hard cap on `?tail=` for status streams (bounded responses, no file slurp).
const STATUS_TAIL_CAP: usize = 1000;

/// Poll cadence for the SSE live-tail of `status.jsonl` (ADR 0093). The file
/// IS the bus: we re-read it and emit only newly-appended complete lines, so a
/// writer that appends whole JSON lines can never hand a subscriber a partial
/// record. Kept modest — a dashboard tail, not a low-latency data path.
const SSE_POLL: std::time::Duration = std::time::Duration::from_millis(300);

/// The embedded dashboard (ADR 0083: one binary, zero deploy steps) — the
/// Leptos+WASM bundle staged by build.rs (`ui/dist` when built via
/// `scripts/build_ui.sh`, else a self-describing stub page). Static assets
/// carry no data, so they are served without auth; every data read/mutation
/// still goes through the token-gated `/api`.
static UI: include_dir::Dir<'_> = include_dir::include_dir!("$OUT_DIR/ui_dist");

#[derive(Clone)]
pub struct AppState {
    /// Parsed ADR-0095 token store; `None` = loopback-only dev mode.
    pub tokens: Option<Arc<blut::rbac::TokenStore>>,
    /// Lineage DB override (tests); `None` = the default `~/.blut` path.
    pub lineage_path: Option<PathBuf>,
    /// The CLI binary the exec bridge shells out to. Recipe launches need a
    /// cookbook binary (only it knows the recipes), so operators point this at
    /// theirs (`--cli lqt`); `cancel` works with the bare engine CLI too.
    pub cli: PathBuf,
    /// The ADR-0095 `audit.jsonl` every mutation is enforced against.
    pub audit_path: PathBuf,
    /// Shared ADR-0094 trigger configuration. A webhook route is closed unless
    /// its trigger has an explicit `webhook_secret` reference here.
    pub triggers: Arc<blut::trigger::TriggerConfig>,
}

/// The principal the token middleware resolved for this request (`None` =
/// anonymous — possible only in loopback dev mode, and viewer-scoped).
#[derive(Clone)]
struct MaybePrincipal(Option<blut::rbac::Principal>);

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
    mut request: axum::extract::Request,
    next: Next,
) -> Response {
    let principal = if let Some(store) = &state.tokens {
        let presented = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        match presented.and_then(|secret| store.resolve(secret)) {
            Some(p) => Some(p), // any resolved principal may READ (Viewer floor)
            None => return err(StatusCode::UNAUTHORIZED, "missing or unknown bearer token"),
        }
    } else {
        None // loopback dev mode: anonymous = viewer scope (mutations denied)
    };
    request.extensions_mut().insert(MaybePrincipal(principal));
    next.run(request).await
}

/// Authorise + audit a bridge mutation (`rbac::enforce`): the audit row is
/// written for allow AND deny BEFORE anything is dispatched, and an audit
/// write failure is itself a deny. Every bridge mutation targets the SHARED
/// tenant — the web surface structurally cannot act on a restricted tenant
/// (same posture as the read side, ADR 0061/0096).
#[allow(clippy::result_large_err)] // an axum Response IS the error surface here
fn enforce_bridge(
    state: &AppState,
    principal: Option<&blut::rbac::Principal>,
    action: blut::rbac::Action,
) -> Result<(), Response> {
    let shared = blut::tenant::Tenant::parse(blut::model_registry::SHARED_TENANT)
        .expect("the shared tenant name parses");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let decision = blut::rbac::enforce(principal, action, &shared, &state.audit_path, now);
    if decision.allowed {
        Ok(())
    } else {
        Err(err(StatusCode::FORBIDDEN, decision.reason))
    }
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

fn exportable_job_dir(id: &str) -> Result<PathBuf, Box<Response>> {
    if !valid_name(id) {
        return Err(Box::new(err(StatusCode::BAD_REQUEST, "invalid job id")));
    }
    let dir = blut::paths::jobs_dir()
        .map_err(|error| Box::new(err(StatusCode::INTERNAL_SERVER_ERROR, error)))?
        .join(id);
    if !dir.is_dir() {
        return Err(Box::new(err(StatusCode::NOT_FOUND, "no such job")));
    }
    let tenant = blut::jobs::read_tenant(id)
        .map_err(|_| Box::new(err(StatusCode::NOT_FOUND, "job is not exportable")))?;
    if tenant.is_restricted() {
        return Err(Box::new(err(
            StatusCode::NOT_FOUND,
            "job is not exportable",
        )));
    }
    Ok(dir)
}

async fn jobs() -> Response {
    match blut::jobs::list_jobs() {
        Ok(mut list) => {
            // Fail closed: malformed custody metadata and restricted tenants
            // are absent from the export rather than partially disclosed.
            list.retain(|job| exportable_job_dir(&job.id).is_ok());
            Json(list).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[derive(serde::Deserialize)]
struct TailQuery {
    tail: Option<usize>,
}

async fn job_status(AxPath(id): AxPath<String>, Query(q): Query<TailQuery>) -> Response {
    let dir = match exportable_job_dir(&id) {
        Ok(d) => d,
        Err(response) => return *response,
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

/// SSE live-tail of a job's `status.jsonl` (ADR 0093 read-only v1). Emits every
/// complete line already present, then each newly-appended line as it lands, as
/// `text/event-stream` `data:` frames. Purely a reader — it opens no write path
/// into the engine; the client dropping the connection drops the stream.
async fn job_events(AxPath(id): AxPath<String>) -> Response {
    let dir = match exportable_job_dir(&id) {
        Ok(d) => d,
        Err(response) => return *response,
    };
    let path = dir.join("status.jsonl");
    if !path.is_file() {
        return err(StatusCode::NOT_FOUND, "no status stream for job");
    }
    let stream = async_stream::stream! {
        // Track by line COUNT (not byte offset): the writer appends whole JSON
        // lines, so `lines()` on a re-read never yields a partial record, and a
        // truncation/rotation (fewer lines than emitted) resets cleanly.
        let mut emitted = 0usize;
        loop {
            if let Ok(text) = std::fs::read_to_string(&path) {
                let lines: Vec<&str> = text.lines().collect();
                if lines.len() < emitted {
                    emitted = 0; // file shrank (rotated) — re-emit from the top
                }
                for line in lines.iter().skip(emitted) {
                    if !line.is_empty() {
                        yield Ok::<_, std::convert::Infallible>(
                            axum::response::sse::Event::default().data(*line),
                        );
                    }
                }
                emitted = lines.len();
            }
            tokio::time::sleep(SSE_POLL).await;
        }
    };
    axum::response::sse::Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
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

// ── webhook ingress (ADR 0094) ─────────────────────────────────────

/// `POST /api/events/{trigger}` — signed, replay-safe webhook ingress. The HMAC
/// covers `<unix-seconds>.<raw-body>` and arrives as
/// `X-Blut-Signature: sha256=<hex>` plus `X-Blut-Timestamp`. The trigger's key
/// is a `SecretRef` in the same configuration sensord consumes.
async fn webhook_event(
    State(state): State<AppState>,
    AxPath(trigger): AxPath<String>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if !blut::trigger::safe_component(&trigger) {
        return err(StatusCode::BAD_REQUEST, "invalid trigger name");
    }
    // The body must be a JSON object (a well-formed event), bounded in size by
    // axum's default request-body limit.
    let parsed: Result<serde_json::Value, _> = serde_json::from_slice(&body);
    let payload = match parsed {
        Ok(v) if v.is_object() => v,
        _ => return err(StatusCode::BAD_REQUEST, "event body must be a JSON object"),
    };
    let Some(binding) = state.triggers.binding(&trigger) else {
        return err(StatusCode::NOT_FOUND, "webhook trigger is not configured");
    };
    let Some(secret_ref) = binding.webhook_secret.as_ref() else {
        return err(
            StatusCode::NOT_FOUND,
            "webhook ingress is disabled for trigger",
        );
    };
    let Some(tenant) = blut::tenant::Tenant::parse(&binding.tenant) else {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "invalid trigger custody configuration",
        );
    };
    if tenant.is_restricted() && binding.data_class != blut_types::trust::DataClass::Restricted {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "restricted trigger must be classified Restricted",
        );
    }

    let timestamp = match headers
        .get("x-blut-timestamp")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<i64>().ok())
    {
        Some(value) => value,
        None => {
            return err(
                StatusCode::UNAUTHORIZED,
                "missing or invalid webhook timestamp",
            );
        }
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    if now.abs_diff(timestamp) > state.triggers.webhook_max_skew_secs {
        return err(
            StatusCode::UNAUTHORIZED,
            "webhook timestamp is outside the replay window",
        );
    }
    let presented_hex = match headers
        .get("x-blut-signature")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("sha256="))
    {
        Some(value) if value.len() == 64 => value,
        _ => {
            return err(
                StatusCode::UNAUTHORIZED,
                "missing or invalid webhook signature",
            );
        }
    };
    let mut presented = [0_u8; 32];
    if faster_hex::hex_decode(presented_hex.as_bytes(), &mut presented).is_err() {
        return err(
            StatusCode::UNAUTHORIZED,
            "missing or invalid webhook signature",
        );
    }
    use blut::secrets::{EnvResolver, ResolveCtx, SecretResolver as _};
    let secret = match EnvResolver.resolve(secret_ref, ResolveCtx::remote()) {
        Ok(secret) => secret,
        Err(_) => return err(StatusCode::UNAUTHORIZED, "webhook credential unavailable"),
    };
    use hmac::{Hmac, Mac as _};
    let mut mac = match Hmac::<sha2::Sha256>::new_from_slice(secret.expose().as_bytes()) {
        Ok(mac) => mac,
        Err(_) => return err(StatusCode::UNAUTHORIZED, "webhook credential unavailable"),
    };
    mac.update(timestamp.to_string().as_bytes());
    mac.update(b".");
    mac.update(&body);
    if mac.verify_slice(&presented).is_err() {
        return err(StatusCode::UNAUTHORIZED, "webhook signature mismatch");
    }

    let dir = binding
        .dir
        .clone()
        .unwrap_or_else(|| blut::trigger::spool_dir(&trigger));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e);
    }
    #[cfg(unix)]
    if let Err(error) = std::fs::set_permissions(
        &dir,
        <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700),
    ) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, error);
    }
    // Content-address the accepted event identity so an identical replay for
    // this exact custody binding lands on the same name. Domain separation
    // prevents two triggers sharing a custom spool directory from colliding.
    let digest = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"blut.webhook.event.v1");
        let data_class = format!("{:?}", binding.data_class);
        for part in [
            trigger.as_bytes(),
            binding.tenant.as_bytes(),
            data_class.as_bytes(),
            body.as_ref(),
        ] {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part);
        }
        faster_hex::hex_string(&h.finalize())
    };
    let file = dir.join(format!("{digest}.json"));
    let event = serde_json::json!({
        "trigger": trigger,
        "tenant": tenant,
        "data_class": binding.data_class,
        "signed_unix": timestamp,
        "received_unix": now,
        "payload": payload,
    });
    let encoded = match serde_json::to_vec(&event) {
        Ok(encoded) => encoded,
        Err(error) => return err(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let replayed = match options.open(&file) {
        Ok(mut output) => {
            use std::io::Write as _;
            if let Err(error) = output.write_all(&encoded).and_then(|()| output.sync_all()) {
                let _ = std::fs::remove_file(&file);
                return err(StatusCode::INTERNAL_SERVER_ERROR, error);
            }
            false
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => true,
        Err(error) => return err(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "spooled": !replayed,
            "replayed": replayed,
            "trigger": trigger,
            "event": digest,
        })),
    )
        .into_response()
}

// ── the exec bridge (ADR 0083 §4) ──────────────────────────────────

/// A launched job is a generated identifier; a recipe name comes from a
/// cookbook catalog. Both share one conservative charset — refuse anything
/// path-ish or flag-ish before it can reach an argv.
fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('-')
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[derive(serde::Deserialize)]
struct RunRequest {
    recipe: String,
    /// Recipe args, forwarded verbatim as `--args <json>`. Must be an object.
    #[serde(default)]
    args: Option<serde_json::Value>,
}

async fn run_recipe(
    State(state): State<AppState>,
    axum::Extension(MaybePrincipal(principal)): axum::Extension<MaybePrincipal>,
    Json(req): Json<RunRequest>,
) -> Response {
    if let Err(deny) = enforce_bridge(&state, principal.as_ref(), blut::rbac::Action::Run) {
        return deny;
    }
    if !valid_name(&req.recipe) {
        return err(StatusCode::BAD_REQUEST, "invalid recipe name");
    }
    let args = req.args.unwrap_or_else(|| serde_json::json!({}));
    if !args.is_object() {
        return err(StatusCode::BAD_REQUEST, "args must be a JSON object");
    }
    // Launch DETACHED through the CLI — the run must outlive this server, and
    // args travel as one argv element (never a shell), so there is nothing to
    // inject into. Output goes to a per-launch log beside the audit trail.
    let launch_dir = state
        .audit_path
        .parent()
        .map(|d| d.join("web-launches"))
        .unwrap_or_else(|| PathBuf::from("web-launches"));
    if let Err(e) = std::fs::create_dir_all(&launch_dir) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e);
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let log_path = launch_dir.join(format!("{stamp}-{}.log", req.recipe));
    let log = match std::fs::File::create(&log_path) {
        Ok(f) => f,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let log_err = match log.try_clone() {
        Ok(f) => f,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let mut cmd = std::process::Command::new(&state.cli);
    cmd.arg("recipe")
        .arg("run")
        .arg(&req.recipe)
        .arg("--args")
        .arg(args.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(log_err);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        cmd.process_group(0); // survive this server's exit / signals
    }
    match cmd.spawn() {
        Ok(child) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "spawned": true,
                "recipe": req.recipe,
                "pid": child.id(),
                "log": log_path,
            })),
        )
            .into_response(),
        Err(e) => err(
            StatusCode::BAD_GATEWAY,
            format!("exec {} failed: {e}", state.cli.display()),
        ),
    }
}

#[derive(serde::Deserialize)]
struct CancelRequest {
    /// Grace period before SIGKILL, e.g. `"10s"` (the CLI's default).
    grace: Option<String>,
}

async fn cancel_job(
    State(state): State<AppState>,
    AxPath(id): AxPath<String>,
    axum::Extension(MaybePrincipal(principal)): axum::Extension<MaybePrincipal>,
    body: Option<Json<CancelRequest>>,
) -> Response {
    if let Err(deny) = enforce_bridge(&state, principal.as_ref(), blut::rbac::Action::Cancel) {
        return deny;
    }
    if !valid_name(&id) {
        return err(StatusCode::BAD_REQUEST, "invalid job id");
    }
    let grace = body.and_then(|Json(b)| b.grace);
    if let Some(g) = &grace
        && (g.is_empty() || !g.chars().all(|c| c.is_ascii_alphanumeric()))
    {
        return err(StatusCode::BAD_REQUEST, "invalid grace duration");
    }
    // Cancel is quick (SIGTERM + bookkeeping) — run it synchronously and
    // return the CLI's own words, one enforcement path for human and API.
    let mut cmd = tokio::process::Command::new(&state.cli);
    cmd.arg("cancel").arg(&id);
    if let Some(g) = grace {
        cmd.arg("--grace").arg(g);
    }
    match cmd.output().await {
        Ok(out) => {
            let ok = out.status.success();
            let status = if ok {
                StatusCode::OK
            } else {
                StatusCode::BAD_GATEWAY
            };
            (
                status,
                Json(serde_json::json!({
                    "ok": ok,
                    "job": id,
                    "stdout": String::from_utf8_lossy(&out.stdout),
                    "stderr": String::from_utf8_lossy(&out.stderr),
                })),
            )
                .into_response()
        }
        Err(e) => err(
            StatusCode::BAD_GATEWAY,
            format!("exec {} failed: {e}", state.cli.display()),
        ),
    }
}

// ── the embedded dashboard ─────────────────────────────────────────

async fn ui_index() -> Response {
    ui_asset(AxPath(String::from("index.html"))).await
}

async fn ui_asset(AxPath(path): AxPath<String>) -> Response {
    // The embedded dir is a closed set baked at compile time; include_dir's
    // lookup is by exact relative path, so nothing traversal-ish resolves.
    let Some(file) = UI.get_file(&path) else {
        return err(StatusCode::NOT_FOUND, "no such asset");
    };
    let mime = match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("wasm") => "application/wasm",
        Some("css") => "text/css",
        _ => "application/octet-stream",
    };
    ([(axum::http::header::CONTENT_TYPE, mime)], file.contents()).into_response()
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/api/jobs", get(jobs).post(run_recipe))
        .route("/api/jobs/{id}/status", get(job_status))
        .route("/api/jobs/{id}/events", get(job_events))
        .route("/api/jobs/{id}/cancel", post(cancel_job))
        .route("/api/events/{trigger}", post(webhook_event))
        .route("/events/{trigger}", post(webhook_event))
        .route("/api/lineage/graph/{hash}", get(lineage_graph))
        .route("/api/lineage/card/{hash}", get(lineage_card))
        .route("/api/lineage/diff/{a}/{b}", get(lineage_diff))
        .route("/api/models/{name}/{alias}", get(model_pointer))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_token))
        // Unauthenticated below: liveness + the embedded dashboard SHELL
        // (static assets carry no data; every read/mutation is `/api`).
        .route("/healthz", get(healthz))
        .route("/", get(ui_index))
        .route("/{*path}", get(ui_asset))
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
            cli: PathBuf::from("blut"),
            audit_path: std::env::temp_dir().join("blut-web-test-audit.jsonl"),
            triggers: Arc::new(blut::trigger::TriggerConfig::default()),
        }
    }

    /// A store with one operator and one viewer token whose SECRETS are the
    /// strings below (hashes precomputed via `rbac::hash_token`).
    fn two_role_store() -> (blut::rbac::TokenStore, &'static str, &'static str) {
        let op_secret = "op-secret-token";
        let view_secret = "view-secret-token";
        let toml = format!(
            "[[token]]\nid=\"op\"\nhash=\"{}\"\nrole=\"operator\"\ntenant=\"shared\"\n\
             [[token]]\nid=\"view\"\nhash=\"{}\"\nrole=\"viewer\"\ntenant=\"shared\"\n",
            blut::rbac::hash_token(op_secret),
            blut::rbac::hash_token(view_secret),
        );
        (
            blut::rbac::TokenStore::parse(&toml).expect("valid store"),
            op_secret,
            view_secret,
        )
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
        let mut st = state(None);
        st.lineage_path = Some(td.path().join("lineage.db"));
        let app = build_router(st);
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
    async fn anonymous_mutation_is_denied_and_audited() {
        // Loopback dev mode (no token store): reads are open, mutations are
        // NOT — rbac treats anonymous as viewer, and the deny is on the record.
        let td = tempfile::tempdir().unwrap();
        let audit = td.path().join("audit.jsonl");
        let mut st = state(None);
        st.audit_path = audit.clone();
        let app = build_router(st);
        let res = app
            .oneshot(
                Request::post("/api/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"recipe":"anything"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        let log = std::fs::read_to_string(&audit).expect("deny was audited");
        assert!(log.contains(r#""allowed":false"#) && log.contains(r#""action":"run""#));
    }

    #[tokio::test]
    async fn viewer_token_cannot_mutate_operator_can() {
        let (store, op, view) = two_role_store();
        let td = tempfile::tempdir().unwrap();
        let mut st = state(Some(store));
        st.audit_path = td.path().join("audit.jsonl");
        // Point the bridge CLI at /bin/echo: the cancel path runs it
        // synchronously and returns its words, proving the argv it built.
        st.cli = PathBuf::from("/bin/echo");
        let app = build_router(st);

        // Viewer: authenticated, but under the Operator floor → 403.
        let res = app
            .clone()
            .oneshot(
                Request::post("/api/jobs/j123/cancel")
                    .header("authorization", format!("Bearer {view}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::FORBIDDEN);

        // Operator: allowed; the CLI (echo) reflects `cancel j123 --grace 30s`.
        let res = app
            .oneshot(
                Request::post("/api/jobs/j123/cancel")
                    .header("authorization", format!("Bearer {op}"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"grace":"30s"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["ok"], true);
        assert!(
            v["stdout"]
                .as_str()
                .unwrap()
                .contains("cancel j123 --grace 30s")
        );
        // Both decisions are on the audit record.
        let log = std::fs::read_to_string(td.path().join("audit.jsonl")).unwrap();
        assert!(log.contains(r#""allowed":false"#) && log.contains(r#""allowed":true"#));
    }

    #[tokio::test]
    async fn bridge_refuses_flaggy_or_pathish_names() {
        let (store, op, _) = two_role_store();
        let td = tempfile::tempdir().unwrap();
        let mut st = state(Some(store));
        st.audit_path = td.path().join("audit.jsonl");
        let app = build_router(st);
        for bad in ["--force", "a/b", "", "x;y"] {
            let res = app
                .clone()
                .oneshot(
                    Request::post("/api/jobs")
                        .header("authorization", format!("Bearer {op}"))
                        .header("content-type", "application/json")
                        .body(Body::from(serde_json::json!({ "recipe": bad }).to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::BAD_REQUEST, "name {bad:?}");
        }
    }

    #[tokio::test]
    async fn dashboard_shell_is_served_at_root_without_auth() {
        // Even with tokens configured, "/" serves the embedded SHELL (static
        // assets carry no data — every read/mutation is the token-gated /api).
        let (store, _, _) = two_role_store();
        let app = build_router(state(Some(store)));
        let res = app
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert!(
            res.headers()[axum::http::header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        let body = axum::body::to_bytes(res.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        // Real bundle and stub both title themselves BLUT.
        assert!(String::from_utf8_lossy(&body).contains("BLUT"));
    }

    #[tokio::test]
    async fn unknown_ui_asset_is_404() {
        let app = build_router(state(None));
        let res = app
            .oneshot(
                Request::get("/no-such-file.js")
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
