// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Cost accounting + the budget guard (ADR 0098).
//!
//! A PURE lineage/ledger read: this module reads `cost.jsonl` and a declared
//! cap, and answers admission questions. It opens no socket and bills nothing
//! itself — the dashboards live in `blut-web` (ADR 0083) and the wire schema
//! plus the rollup arithmetic live in [`blut_types::cost`], so the guard, the
//! `blut cost report` CLI, and the dashboard can never disagree about a number.
//!
//! **Fail-closed by construction.** A missing, unreadable, or malformed cap is
//! [`BudgetCap::REFUSE_ALL`] — zero, which refuses — never "unlimited". A
//! budget system that opens up when its configuration breaks protects nothing
//! at exactly the moment it is needed.

pub use blut_types::cost::{
    BudgetCap, CostRow, Forecast, LOCAL_PROVIDER, Micros, Rollup, forecast, rollup,
};

/// Default ledger location (`$BLUT_COST_LEDGER` or `~/.blut/cost.jsonl`).
pub fn default_ledger_path() -> std::path::PathBuf {
    if let Some(p) = std::env::var_os("BLUT_COST_LEDGER") {
        return std::path::PathBuf::from(p);
    }
    dot_blut().join("cost.jsonl")
}

/// Default cap file (`$BLUT_BUDGET_CAP` or `~/.blut/budget.toml`).
pub fn default_cap_path() -> std::path::PathBuf {
    if let Some(p) = std::env::var_os("BLUT_BUDGET_CAP") {
        return std::path::PathBuf::from(p);
    }
    dot_blut().join("budget.toml")
}

/// `~/.blut`. NOTE: `sensord.rs` and `cli/mod.rs` each carry their own private
/// copy of this; consolidating the three is exactly the kind of ownership
/// question ADR 0092 exists to settle, so it is left alone here rather than
/// refactored mid-ADR.
fn dot_blut() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_default()
        .join(".blut")
}

/// One `[cap]` table: a per-tenant ceiling in whole cents.
#[derive(Debug, Default, serde::Deserialize)]
struct CapFile {
    /// Cap applied when no per-tenant entry matches.
    #[serde(default)]
    default_cents: Option<i64>,
    /// Per-tenant overrides, keyed by tenant string.
    #[serde(default)]
    tenant: std::collections::BTreeMap<String, i64>,
}

/// Read the declared cap for `tenant`.
///
/// Every failure path returns [`BudgetCap::REFUSE_ALL`]: an absent file, an
/// unreadable file, a malformed file, and a tenant with no entry and no default
/// all refuse. This is the ADR's explicit requirement — `cap=0`, not
/// "unlimited".
pub fn read_cap(path: &std::path::Path, tenant: &str) -> BudgetCap {
    let Ok(text) = std::fs::read_to_string(path) else {
        return BudgetCap::REFUSE_ALL;
    };
    let Ok(parsed) = toml::from_str::<CapFile>(&text) else {
        return BudgetCap::REFUSE_ALL;
    };
    let cents = parsed.tenant.get(tenant).copied().or(parsed.default_cents);
    match cents {
        Some(c) if c > 0 => BudgetCap(Micros::from_cents(c)),
        _ => BudgetCap::REFUSE_ALL,
    }
}

/// Read ledger rows. A malformed line is skipped (an append-only ledger can
/// have a torn tail); an absent ledger means no prior spend, which is honest —
/// it is the CAP, not the ledger, that fails closed.
pub fn read_ledger(path: &std::path::Path) -> Vec<CostRow> {
    match std::fs::read_to_string(path) {
        Ok(text) => text.lines().filter_map(CostRow::from_line).collect(),
        Err(_) => Vec::new(),
    }
}

/// Append one charge to the ledger.
pub fn append_row(path: &std::path::Path, row: &CostRow) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", row.to_line())
}

/// Spend already recorded against `tenant`.
pub fn spent_for_tenant(rows: &[CostRow], tenant: &str) -> Micros {
    rows.iter()
        .filter(|r| r.tenant == tenant)
        .map(|r| r.micros)
        .sum()
}

/// The budget verdict for a prospective job.
///
/// Returns a broker [`AdmitDecision`](crate::broker::admission::AdmitDecision)
/// so a budget refusal travels the SAME path as a memory refusal — the ADR is
/// explicit that this must not be a bespoke exit, because a second refusal path
/// would not inherit admission's auditing or its containment guarantees.
pub fn budget_guard(
    rows: &[CostRow],
    tenant: &str,
    cap: BudgetCap,
    projected_additional: Micros,
) -> crate::broker::admission::AdmitDecision {
    use crate::broker::admission::AdmitDecision;
    let spent = spent_for_tenant(rows, tenant);
    let projected = spent.saturating_add(projected_additional);
    if cap.allows(projected) {
        // Budget says yes. Memory admission still has its own say; this guard
        // only ever REFUSES, it never grants a memory allowance of its own.
        AdmitDecision::Admit { memmax_bytes: 0 }
    } else {
        AdmitDecision::Refuse {
            reason: format!(
                "budget cap exceeded for tenant '{tenant}': spent {} + projected {} = {} > cap {} \
                 (ADR 0098; a missing cap reads as $0.00 and refuses)",
                spent.to_dollars_string(),
                projected_additional.to_dollars_string(),
                projected.to_dollars_string(),
                cap.0.to_dollars_string(),
            ),
        }
    }
}

/// Rows that may be reported to a REMOTE provider surface. Clinical/restricted
/// tenants are filtered out here, once, so no caller has to remember to do it
/// (ADR 0061/0096 fail-closed).
pub fn exportable_rows(rows: &[CostRow]) -> Vec<&CostRow> {
    rows.iter().filter(|r| r.may_leave_box()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broker::admission::AdmitDecision;

    fn row(run: &str, tenant: &str, provider: &str, cents: i64) -> CostRow {
        CostRow {
            run_id: run.into(),
            tenant: tenant.into(),
            provider: provider.into(),
            micros: Micros::from_cents(cents),
            unix: 0,
        }
    }

    #[test]
    fn a_missing_or_broken_cap_file_refuses() {
        let td = tempfile::tempdir().unwrap();
        // Absent.
        assert_eq!(
            read_cap(&td.path().join("nope.toml"), "shared"),
            BudgetCap::REFUSE_ALL
        );
        // Malformed.
        let bad = td.path().join("bad.toml");
        std::fs::write(&bad, "this is not toml {{{").unwrap();
        assert_eq!(read_cap(&bad, "shared"), BudgetCap::REFUSE_ALL);
        // Well-formed but no entry for this tenant and no default.
        let partial = td.path().join("partial.toml");
        std::fs::write(&partial, "[tenant]\nother = 500\n").unwrap();
        assert_eq!(read_cap(&partial, "shared"), BudgetCap::REFUSE_ALL);
        // A zero or negative cap is still a refusal, not "unlimited".
        let zero = td.path().join("zero.toml");
        std::fs::write(&zero, "default_cents = 0\n").unwrap();
        assert_eq!(read_cap(&zero, "shared"), BudgetCap::REFUSE_ALL);
    }

    #[test]
    fn a_declared_cap_is_read_per_tenant_with_a_default_fallback() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("budget.toml");
        std::fs::write(
            &path,
            "default_cents = 1000\n[tenant]\n\"research/dev\" = 250\n",
        )
        .unwrap();
        assert_eq!(read_cap(&path, "shared").0, Micros::from_cents(1000));
        assert_eq!(read_cap(&path, "research/dev").0, Micros::from_cents(250));
    }

    #[test]
    fn crossing_the_cap_refuses_through_the_broker_decision_type() {
        let rows = vec![row("r1", "shared", "aws", 700)];
        let cap = BudgetCap(Micros::from_cents(1000));
        // Under cap → admit.
        assert!(matches!(
            budget_guard(&rows, "shared", cap, Micros::from_cents(200)),
            AdmitDecision::Admit { .. }
        ));
        // Exactly at cap → admit (the boundary is inclusive).
        assert!(matches!(
            budget_guard(&rows, "shared", cap, Micros::from_cents(300)),
            AdmitDecision::Admit { .. }
        ));
        // Over cap → the SAME refusal type memory admission uses.
        match budget_guard(&rows, "shared", cap, Micros::from_cents(301)) {
            AdmitDecision::Refuse { reason } => {
                assert!(reason.contains("budget cap exceeded"), "{reason}");
                assert!(reason.contains("$10.00"), "cap must be legible: {reason}");
            }
            other => panic!("expected a broker Refuse, got {other:?}"),
        }
    }

    #[test]
    fn spend_is_scoped_to_the_tenant() {
        let rows = vec![
            row("r1", "shared", "aws", 500),
            row("r2", "research/dev", "aws", 900),
        ];
        assert_eq!(spent_for_tenant(&rows, "shared"), Micros::from_cents(500));
        // Another tenant's spend must not consume this tenant's budget.
        assert!(matches!(
            budget_guard(
                &rows,
                "shared",
                BudgetCap(Micros::from_cents(600)),
                Micros::ZERO
            ),
            AdmitDecision::Admit { .. }
        ));
    }

    #[test]
    fn clinical_rows_are_filtered_before_any_remote_report() {
        let rows = vec![
            row("r1", "shared", "aws", 100),
            row("r2", "clinical/prod", "aws", 100),
        ];
        let out = exportable_rows(&rows);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].tenant, "shared");
    }

    #[test]
    fn ledger_round_trips_and_tolerates_a_torn_tail() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("sub").join("cost.jsonl");
        append_row(&path, &row("r1", "shared", "local", 125)).unwrap();
        // Simulate a torn append.
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        write!(f, "{{\"run_id\":\"partial\"").unwrap();
        drop(f);
        let rows = read_ledger(&path);
        assert_eq!(rows.len(), 1, "a torn tail must not lose the good rows");
        assert_eq!(rows[0].micros.as_cents(), 125);
    }
}
