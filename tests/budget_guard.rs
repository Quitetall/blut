// SPDX-License-Identifier: AGPL-3.0-or-later
// The ADR 0098 acceptance gate (engine half). Asserts the three properties the
// ADR names explicitly:
//   1. a job whose PROJECTED spend crosses the cap is REFUSED via the broker
//      admission path — the same `AdmitDecision` a memory refusal uses, not a
//      bespoke exit that would skip admission's auditing and containment;
//   2. a missing or unreadable cap REFUSES (cap = 0), never "unlimited";
//   3. a clinical-tenant cost row is never emitted to a remote provider.

use blut::broker::admission::AdmitDecision;
use blut::cost::{
    BudgetCap, CostRow, Micros, append_row, budget_guard, exportable_rows, read_cap, read_ledger,
    rollup,
};

fn row(run: &str, tenant: &str, provider: &str, cents: i64) -> CostRow {
    CostRow {
        run_id: run.into(),
        tenant: tenant.into(),
        provider: provider.into(),
        micros: Micros::from_cents(cents),
        unix: 1_700_000_000,
    }
}

#[test]
fn budget_guard_gate() {
    let td = tempfile::tempdir().unwrap();
    let ledger = td.path().join("cost.jsonl");

    // A tenant with $7.00 of prior spend, written through the real ledger API.
    for r in [
        row("r1", "shared", "local", 300),
        row("r2", "shared", "aws", 400),
        row("r3", "clinical/prod", "local", 950),
    ] {
        append_row(&ledger, &r).unwrap();
    }
    let rows = read_ledger(&ledger);
    assert_eq!(rows.len(), 3, "ledger round-trips through the real writer");

    // ── (2) a missing cap refuses; it must NOT read as unlimited ───────
    let missing = read_cap(&td.path().join("absent.toml"), "shared");
    assert_eq!(missing, BudgetCap::REFUSE_ALL);
    match budget_guard(&rows, "shared", missing, Micros::from_cents(1)) {
        AdmitDecision::Refuse { reason } => assert!(reason.contains("budget cap exceeded")),
        other => panic!("a missing cap MUST refuse, got {other:?}"),
    }
    // Same for an unreadable/malformed cap file.
    let broken = td.path().join("broken.toml");
    std::fs::write(&broken, "}{ not toml").unwrap();
    assert_eq!(read_cap(&broken, "shared"), BudgetCap::REFUSE_ALL);

    // ── (1) crossing the cap refuses THROUGH the broker decision ───────
    let cap_file = td.path().join("budget.toml");
    std::fs::write(&cap_file, "default_cents = 1000\n").unwrap(); // $10.00
    let cap = read_cap(&cap_file, "shared");
    assert_eq!(cap.0, Micros::from_cents(1000));

    // $7.00 spent + $2.00 projected = $9.00 ≤ $10.00 → admitted.
    assert!(
        matches!(
            budget_guard(&rows, "shared", cap, Micros::from_cents(200)),
            AdmitDecision::Admit { .. }
        ),
        "a job inside the cap must not be refused"
    );
    // $7.00 + $3.01 = $10.01 > $10.00 → refused, and the refusal is the broker's
    // own type so it travels the normal admission path.
    let decision = budget_guard(&rows, "shared", cap, Micros::from_cents(301));
    match decision {
        AdmitDecision::Refuse { reason } => {
            assert!(reason.contains("budget cap exceeded"), "{reason}");
            assert!(
                reason.contains("$10.01"),
                "projected total legible: {reason}"
            );
            assert!(reason.contains("$10.00"), "cap legible: {reason}");
        }
        other => panic!("expected the broker's Refuse, got {other:?}"),
    }

    // Another tenant's spend must not consume this tenant's budget.
    assert!(
        matches!(
            budget_guard(&rows, "research/dev", cap, Micros::from_cents(900)),
            AdmitDecision::Admit { .. }
        ),
        "budgets are per-tenant"
    );

    // ── (3) clinical rows never reach a remote provider surface ────────
    let exportable = exportable_rows(&rows);
    assert_eq!(exportable.len(), 2, "the clinical row is withheld");
    assert!(
        exportable.iter().all(|r| r.tenant != "clinical/prod"),
        "a restricted tenant's spend must never leave the box (ADR 0061)"
    );
    // ...while remaining fully visible to LOCAL accounting.
    let local = rollup(&rows);
    assert_eq!(
        local.by_tenant["clinical/prod"],
        Micros::from_cents(950),
        "clinical spend is local-only, not invisible"
    );
    assert_eq!(local.total, Micros::from_cents(1650));
    assert_eq!(local.total.as_cents(), 1650, "sums are exact to the cent");
}
