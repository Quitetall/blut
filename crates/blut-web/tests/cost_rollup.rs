// SPDX-License-Identifier: AGPL-3.0-or-later
// The ADR 0098 acceptance gate (dashboard half): per-run / per-tenant /
// per-provider sums and the burn forecast must match a fixture ledger TO THE
// CENT, and a restricted tenant's spend must never appear on this export
// surface (ADR 0061/0096).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use blut_types::cost::{CostRow, Micros};
use tower::util::ServiceExt;

fn row(run: &str, tenant: &str, provider: &str, cents: i64) -> CostRow {
    CostRow {
        run_id: run.into(),
        tenant: tenant.into(),
        provider: provider.into(),
        micros: Micros::from_cents(cents),
        unix: 1_700_000_000,
    }
}

fn state() -> blut_web::AppState {
    blut_web::AppState {
        tokens: None,
        lineage_path: None,
        cli: std::path::PathBuf::from("/bin/false"),
        audit_path: std::env::temp_dir().join("cost-rollup-audit.jsonl"),
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
async fn cost_rollup_gate() {
    let td = tempfile::tempdir().unwrap();
    let ledger = td.path().join("cost.jsonl");
    // Fixture: two runs, two tenants, two providers — plus a clinical row that
    // must be withheld from this surface but still counted locally.
    let fixture = [
        row("r1", "shared", "local", 125),        // $1.25
        row("r1", "shared", "aws", 375),          // $3.75
        row("r2", "research/dev", "aws", 1_000),  // $10.00
        row("r3", "clinical/prod", "local", 500), // withheld
    ];
    let body: String = fixture.iter().map(|r| r.to_line() + "\n").collect();
    std::fs::write(&ledger, body).unwrap();
    unsafe { std::env::set_var("BLUT_COST_LEDGER", &ledger) };

    let app = blut_web::build_router(state());
    let (status, v) = get_json(app.clone(), "/api/cost").await;
    assert_eq!(status, StatusCode::OK);

    // ── sums exact to the cent, on every dimension ─────────────────────
    let rollup = &v["rollup"];
    assert_eq!(rollup["total"], Micros::from_cents(1500).0, "$15.00");
    assert_eq!(rollup["by_run"]["r1"], Micros::from_cents(500).0, "$5.00");
    assert_eq!(rollup["by_run"]["r2"], Micros::from_cents(1000).0, "$10.00");
    assert_eq!(rollup["by_tenant"]["shared"], Micros::from_cents(500).0);
    assert_eq!(
        rollup["by_tenant"]["research/dev"],
        Micros::from_cents(1000).0
    );
    assert_eq!(
        rollup["by_provider"]["aws"],
        Micros::from_cents(1375).0,
        "$13.75"
    );
    assert_eq!(
        rollup["by_provider"]["local"],
        Micros::from_cents(125).0,
        "$1.25"
    );
    assert_eq!(rollup["rows"], 3, "the clinical row is not summed here");

    // ── the clinical row is withheld, and says so ──────────────────────
    assert_eq!(v["withheld_restricted_rows"], 1);
    assert!(
        rollup["by_tenant"].get("clinical/prod").is_none(),
        "a restricted tenant must never appear on the export surface"
    );

    // ── burn forecast: $15.00 at 25% projects $60.00, over a $20 cap ───
    let (_, f) = get_json(app.clone(), "/api/cost?progress=0.25&cap_cents=2000").await;
    let forecast = &f["forecast"];
    assert_eq!(forecast["spent"], Micros::from_cents(1500).0);
    assert_eq!(
        forecast["projected_total"],
        Micros::from_cents(6000).0,
        "$60.00"
    );
    assert_eq!(forecast["projected_over_cap"], true);

    // Under a generous cap the same projection is not flagged.
    let (_, ok) = get_json(app.clone(), "/api/cost?progress=0.25&cap_cents=10000").await;
    assert_eq!(ok["forecast"]["projected_over_cap"], false);

    // Zero progress must not fabricate a projection (which would read "free").
    let (_, zero) = get_json(app, "/api/cost?progress=0").await;
    assert!(zero["forecast"]["projected_total"].is_null());
}
