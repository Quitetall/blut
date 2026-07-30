// SPDX-License-Identifier: AGPL-3.0-or-later
// The ADR 0093 acceptance gate: boot the read-only sidecar against fixtures and
// prove the three properties the ADR names — (1) the SSE stream reflects a live
// node-state transition in `status.jsonl`, (2) every GET is side-effect-free on
// the engine's on-disk state, and (3) a `restricted` artifact is never served
// (with a shared control that IS served, so the exclusion isn't vacuous).
//
// One test function, run serially, because it sets the process-global
// `LAMU_TRAIN_JOBS_DIR` env var that the engine's path resolver reads.

use std::path::PathBuf;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::util::ServiceExt;

fn signed_request(path: &str, body: &'static str, timestamp: i64) -> Request<Body> {
    use hmac::{Hmac, Mac as _};
    let mut mac = Hmac::<sha2::Sha256>::new_from_slice(b"test-webhook-key").unwrap();
    mac.update(timestamp.to_string().as_bytes());
    mac.update(b".");
    mac.update(body.as_bytes());
    let signature = faster_hex::hex_string(&mac.finalize().into_bytes());
    Request::post(path)
        .header("content-type", "application/json")
        .header("x-blut-timestamp", timestamp.to_string())
        .header("x-blut-signature", format!("sha256={signature}"))
        .body(Body::from(body))
        .unwrap()
}

fn state(jobs_dir: &std::path::Path, lineage: PathBuf) -> blut_web::AppState {
    // Point the engine's per-job path resolver at our fixture tree.
    unsafe { std::env::set_var("LAMU_TRAIN_JOBS_DIR", jobs_dir) };
    blut_web::AppState {
        tokens: None, // loopback dev mode: reads open, no auth needed for this gate
        lineage_path: Some(lineage),
        cli: PathBuf::from("/bin/false"), // no mutation is exercised here
        audit_path: jobs_dir.join("audit.jsonl"),
        triggers: std::sync::Arc::new(blut::trigger::TriggerConfig::default()),
    }
}

/// Read from `sock` until `marker` appears in the accumulated bytes or `budget`
/// elapses. Returns whether the marker was seen. SSE frames are written one per
/// event, so a small unique marker stays contiguous across chunk boundaries.
async fn read_until(sock: &mut tokio::net::TcpStream, marker: &str, budget: Duration) -> bool {
    let mut acc: Vec<u8> = Vec::new();
    let mut buf = [0u8; 2048];
    tokio::time::timeout(budget, async {
        loop {
            match sock.read(&mut buf).await {
                Ok(0) | Err(_) => break false,
                Ok(n) => {
                    acc.extend_from_slice(&buf[..n]);
                    if String::from_utf8_lossy(&acc).contains(marker) {
                        break true;
                    }
                }
            }
        }
    })
    .await
    .unwrap_or(false)
}

fn dir_snapshot(dir: &std::path::Path) -> Vec<(String, u64)> {
    let mut out: Vec<(String, u64)> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| {
            let len = e.metadata().map(|m| m.len()).unwrap_or(0);
            (e.file_name().to_string_lossy().into_owned(), len)
        })
        .collect();
    out.sort();
    out
}

#[tokio::test]
async fn dashboard_readonly_gate() {
    let td = tempfile::tempdir().unwrap();
    let jobs_dir = td.path().join("jobs");
    let job = "jobgate1";
    let job_dir = jobs_dir.join(job);
    std::fs::create_dir_all(&job_dir).unwrap();
    let status = job_dir.join("status.jsonl");
    // Line 1: the run starts.
    std::fs::write(&status, "{\"event\":\"start\",\"state\":\"running\"}\n").unwrap();
    std::fs::write(job_dir.join("tenant"), "shared").unwrap();

    // A restricted job must be absent from all job export surfaces, not only
    // from lineage graph/card queries.
    let restricted_job = jobs_dir.join("clinical-job");
    std::fs::create_dir_all(&restricted_job).unwrap();
    std::fs::write(restricted_job.join("tenant"), "clinical/prod").unwrap();
    std::fs::write(
        restricted_job.join("status.jsonl"),
        "{\"event\":\"patient-sensitive\"}\n",
    )
    .unwrap();

    // Seed the lineage DB: one SHARED artifact (served) and one RESTRICTED
    // artifact (excluded) — the non-vacuous control for property (3).
    let lineage_path = td.path().join("lineage.db");
    let shared_hash = "aaaa000000000000000000000000000000000000000000000000000000000001";
    let restricted_hash = "bbbb000000000000000000000000000000000000000000000000000000000002";
    {
        let db = blut::lineage_db::LineageDb::open_at(&lineage_path).unwrap();
        for (jid, tenant, hash) in [
            ("shared_job", "shared", shared_hash),
            ("restricted_job", "clinical/prod", restricted_hash),
        ] {
            db.record_run(&blut::lineage_db::RunRow {
                job_id: jid.into(),
                recipe: "demo".into(),
                tenant: tenant.into(),
                ..Default::default()
            })
            .unwrap();
            db.record_artifact(&blut::lineage_db::ArtifactRow {
                job_id: jid.into(),
                stage_idx: 0,
                stage_name: "encode".into(),
                content_hash: hash.into(),
                kind: "checkpoint".into(),
                schema_ver: 1,
                ..Default::default()
            })
            .unwrap();
        }
    }

    let mut app_state = state(&jobs_dir, lineage_path);
    let trigger_cfg = blut::trigger::TriggerConfig::parse(
        r#"
[[trigger]]
name = "nightly.v1"
plan = "registry://plan@prod"
tenant = "shared"
data_class = "Internal"
webhook_secret = { name = "BLUT_TEST_WEBHOOK_KEY" }
"#,
    )
    .unwrap();
    app_state.triggers = std::sync::Arc::new(trigger_cfg);
    // SAFETY (serial integration test): no other test in this process reads
    // this test-only credential.
    unsafe { std::env::set_var("BLUT_TEST_WEBHOOK_KEY", "test-webhook-key") };
    let app = blut_web::build_router(app_state);

    // ── property (1): SSE reflects a live transition ───────────────────
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_app = app.clone();
    tokio::spawn(async move {
        let _ = axum::serve(listener, server_app).await;
    });

    let mut sock = tokio::net::TcpStream::connect(addr).await.unwrap();
    sock.write_all(
        format!(
            "GET /api/jobs/{job}/events HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        .as_bytes(),
    )
    .await
    .unwrap();

    // Let the stream emit line 1 and enter its poll loop, then append the
    // transition line (running → succeeded) with a unique marker.
    tokio::time::sleep(Duration::from_millis(600)).await;
    let marker = "STATE_TRANSITION_ZY99";
    {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&status)
            .unwrap();
        writeln!(
            f,
            "{{\"event\":\"done\",\"state\":\"succeeded\",\"m\":\"{marker}\"}}"
        )
        .unwrap();
    }
    assert!(
        read_until(&mut sock, marker, Duration::from_secs(8)).await,
        "SSE live-tail did not deliver the appended state transition"
    );

    // The incremental tail must never hand a subscriber HALF a JSON record:
    // write a line with no terminating newline, prove it is withheld, then
    // complete it and prove it arrives whole.
    let partial_marker = "PARTIAL_RECORD_QX41";
    {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&status)
            .unwrap();
        write!(f, "{{\"event\":\"mid\",\"m\":\"{partial_marker}\"").unwrap();
    }
    assert!(
        !read_until(&mut sock, partial_marker, Duration::from_millis(1200)).await,
        "an unterminated line must NOT be emitted as an SSE event"
    );
    {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&status)
            .unwrap();
        writeln!(f, "}}").unwrap();
    }
    assert!(
        read_until(&mut sock, partial_marker, Duration::from_secs(8)).await,
        "the completed line must arrive once its newline lands"
    );

    // ── property (2): GETs are side-effect-free on engine state ────────
    let before = (dir_snapshot(&job_dir), std::fs::read(&status).unwrap());
    for path in ["/api/jobs", &format!("/api/jobs/{job}/status")] {
        let res = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK, "GET {path}");
    }
    let after = (dir_snapshot(&job_dir), std::fs::read(&status).unwrap());
    assert_eq!(before, after, "a GET mutated on-disk job state");

    // Unknown reads are also side-effect free: asking for a ghost job must not
    // create the directory via `paths::job_dir`.
    let ghost = app
        .clone()
        .oneshot(
            Request::get("/api/jobs/ghost/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ghost.status(), StatusCode::NOT_FOUND);
    assert!(!jobs_dir.join("ghost").exists());

    for path in [
        "/api/jobs/clinical-job/status",
        "/api/jobs/clinical-job/events",
    ] {
        let response = app
            .clone()
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "GET {path}");
    }
    let listed = app
        .clone()
        .oneshot(Request::get("/api/jobs").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let listed_body = axum::body::to_bytes(listed.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert!(!String::from_utf8_lossy(&listed_body).contains("clinical-job"));

    // ── property (3): restricted excluded, shared control served ───────
    let served = app
        .clone()
        .oneshot(
            Request::get(format!("/api/lineage/graph/{shared_hash}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        served.status(),
        StatusCode::OK,
        "the shared control artifact must be served"
    );
    let excluded = app
        .clone()
        .oneshot(
            Request::get(format!("/api/lineage/graph/{restricted_hash}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        excluded.status(),
        StatusCode::NOT_FOUND,
        "a restricted artifact must never be served (ADR 0061)"
    );

    // ── webhook ingress (ADR 0094): POST an event → spooled for sensord ─
    let events_dir = td.path().join("events");
    unsafe { std::env::set_var("BLUT_EVENTS_DIR", &events_dir) };
    let body = r#"{"kind":"webhook","payload":1}"#;
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let unsigned = app
        .clone()
        .oneshot(
            Request::post("/api/events/nightly.v1")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unsigned.status(), StatusCode::UNAUTHORIZED);

    let posted = app
        .clone()
        .oneshot(signed_request("/events/nightly.v1", body, timestamp))
        .await
        .unwrap();
    assert_eq!(
        posted.status(),
        StatusCode::ACCEPTED,
        "webhook event spooled"
    );
    let spooled: Vec<_> = std::fs::read_dir(events_dir.join("nightly.v1"))
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(spooled.len(), 1, "one event file written to the spool");
    let first_modified = spooled[0].metadata().unwrap().modified().unwrap();
    std::thread::sleep(Duration::from_millis(10));
    let replay = app
        .clone()
        .oneshot(signed_request("/api/events/nightly.v1", body, timestamp))
        .await
        .unwrap();
    assert_eq!(replay.status(), StatusCode::ACCEPTED);
    let after_replay: Vec<_> = std::fs::read_dir(events_dir.join("nightly.v1"))
        .unwrap()
        .filter_map(|entry| entry.ok())
        .collect();
    assert_eq!(after_replay.len(), 1, "replay must not add a spool file");
    assert_eq!(
        after_replay[0].metadata().unwrap().modified().unwrap(),
        first_modified,
        "replay must not rewrite the content-addressed event"
    );
    // A non-object body is refused.
    let bad = app
        .oneshot(signed_request("/api/events/nightly.v1", "42", timestamp))
        .await
        .unwrap();
    assert_eq!(
        bad.status(),
        StatusCode::BAD_REQUEST,
        "non-object event refused"
    );
}
