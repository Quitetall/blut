// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Cost accounting wire schema + rollup math (ADR 0098).
//!
//! Shared by the engine's budget guard, the `blut cost report` CLI, and the
//! `blut-web` dashboards, so all three read ONE schema and — critically — run
//! ONE implementation of the rollup arithmetic. Duplicating that math in the
//! guard and the dashboard is how a system ends up refusing a job for a spend
//! figure its own dashboard disagrees with.
//!
//! **Money is integer micro-dollars, never a float.** The predecessor ledger
//! (ADR 0067 T3.1e) recorded `f64` abstract units and its own comment warned
//! that a currency ledger must be fixed-point to avoid drift; ADR 0098's gate
//! requires sums that match "to the cent", which floating point cannot promise
//! under repeated addition.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::tenant::Tenant;

/// Money, as integer micro-dollars (1e-6 USD). Signed so a credit/refund row
/// is representable without a second type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Micros(pub i64);

impl Micros {
    pub const ZERO: Micros = Micros(0);

    pub fn from_cents(cents: i64) -> Self {
        Micros(cents.saturating_mul(10_000))
    }

    /// Exact cents, truncated toward zero. Exact because the underlying unit is
    /// an integer — this is the operation the acceptance gate checks.
    pub fn as_cents(self) -> i64 {
        self.0 / 10_000
    }

    pub fn saturating_add(self, other: Micros) -> Micros {
        Micros(self.0.saturating_add(other.0))
    }

    /// Render as `$X.YY` for operator-facing output.
    pub fn to_dollars_string(self) -> String {
        // 1 dollar = 1_000_000 micros = 100 cents. Splitting on 10_000 (the
        // CENT divisor) would print cents in the dollars position.
        let negative = self.0 < 0;
        let abs = self.0.abs();
        format!(
            "{}${}.{:02}",
            if negative { "-" } else { "" },
            abs / 1_000_000,
            (abs % 1_000_000) / 10_000
        )
    }
}

impl std::iter::Sum for Micros {
    fn sum<I: Iterator<Item = Micros>>(iter: I) -> Micros {
        iter.fold(Micros::ZERO, |acc, m| acc.saturating_add(m))
    }
}

/// One ledger row: a charge attributable to a run, a tenant, and a provider.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostRow {
    pub run_id: String,
    /// Owning tenant (`project[/domain]`). Drives the clinical hard-block.
    pub tenant: String,
    /// Where the spend occurred — `local` for on-box compute, else a provider id.
    pub provider: String,
    pub micros: Micros,
    pub unix: i64,
}

/// The identifier used for on-box compute, which is never "remote".
pub const LOCAL_PROVIDER: &str = "local";

impl CostRow {
    /// May this row be reported to a REMOTE provider surface?
    ///
    /// Fail-closed on the clinical boundary (ADR 0061/0096): a restricted
    /// tenant's spend stays local-only, and an unparseable tenant string is
    /// treated as restricted rather than assumed safe.
    pub fn may_leave_box(&self) -> bool {
        Tenant::parse(&self.tenant).is_some_and(|t| !t.is_restricted())
    }

    pub fn to_line(&self) -> String {
        serde_json::to_string(self).expect("CostRow serializes")
    }

    pub fn from_line(line: &str) -> Option<Self> {
        serde_json::from_str(line).ok()
    }
}

/// A declared spend ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BudgetCap(pub Micros);

impl BudgetCap {
    /// The fail-closed default. ADR 0098 is explicit: a missing or unreadable
    /// cap means ZERO, which refuses — never "unlimited". A budget system that
    /// defaults to unlimited protects nothing precisely when its config is
    /// broken.
    pub const REFUSE_ALL: BudgetCap = BudgetCap(Micros::ZERO);

    pub fn allows(&self, projected: Micros) -> bool {
        projected <= self.0
    }
}

impl Default for BudgetCap {
    fn default() -> Self {
        Self::REFUSE_ALL
    }
}

/// Spend grouped along one dimension, plus the total. `BTreeMap` so output is
/// deterministically ordered for both the CLI and the dashboard.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rollup {
    pub by_run: BTreeMap<String, Micros>,
    pub by_tenant: BTreeMap<String, Micros>,
    pub by_provider: BTreeMap<String, Micros>,
    pub total: Micros,
    pub rows: usize,
}

/// Roll up rows per run, per tenant, and per provider. The ONE implementation
/// the guard, the CLI, and the dashboard all call.
pub fn rollup<'a>(rows: impl IntoIterator<Item = &'a CostRow>) -> Rollup {
    let mut out = Rollup::default();
    for row in rows {
        *out.by_run.entry(row.run_id.clone()).or_default() = out
            .by_run
            .get(&row.run_id)
            .copied()
            .unwrap_or_default()
            .saturating_add(row.micros);
        *out.by_tenant.entry(row.tenant.clone()).or_default() = out
            .by_tenant
            .get(&row.tenant)
            .copied()
            .unwrap_or_default()
            .saturating_add(row.micros);
        *out.by_provider.entry(row.provider.clone()).or_default() = out
            .by_provider
            .get(&row.provider)
            .copied()
            .unwrap_or_default()
            .saturating_add(row.micros);
        out.total = out.total.saturating_add(row.micros);
        out.rows += 1;
    }
    out
}

/// Linear burn-rate forecast: spend-to-date ÷ progress, and when that line
/// crosses the cap.
// No `Eq`: `progress` is an f64. Money stays integer; only the progress
// FRACTION is floating point, and it never participates in a spend total.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Forecast {
    pub spent: Micros,
    /// Projected total at completion. `None` when progress is 0 — an unknown
    /// projection must not be reported as `0`, which would read as "free".
    pub projected_total: Option<Micros>,
    /// Fraction of the projected run already spent-through, 0.0–1.0.
    pub progress: f64,
    /// True when the projection exceeds the cap.
    pub projected_over_cap: bool,
}

/// Forecast spend at completion from spend-to-date and fractional progress.
/// `progress` outside (0, 1] yields no projection rather than a fabricated one.
pub fn forecast(spent: Micros, progress: f64, cap: BudgetCap) -> Forecast {
    let projected_total = if progress.is_finite() && progress > 0.0 && progress <= 1.0 {
        // Round half away from zero so a forecast is never optimistic by a
        // truncated fraction of a cent.
        let projected = (spent.0 as f64 / progress).round();
        Some(Micros(projected as i64))
    } else {
        None
    };
    Forecast {
        spent,
        projected_total,
        progress,
        projected_over_cap: projected_total.is_some_and(|p| !cap.allows(p)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(run: &str, tenant: &str, provider: &str, micros: i64) -> CostRow {
        CostRow {
            run_id: run.into(),
            tenant: tenant.into(),
            provider: provider.into(),
            micros: Micros(micros),
            unix: 0,
        }
    }

    #[test]
    fn money_is_exact_to_the_cent() {
        // A third of a dollar, added three times, must be exact — the property
        // f64 cannot promise and the acceptance gate checks.
        let third = Micros::from_cents(33);
        let total: Micros = [third, third, third].into_iter().sum();
        assert_eq!(total.as_cents(), 99);
        assert_eq!(total.to_dollars_string(), "$0.99");
        assert_eq!(Micros::from_cents(-250).to_dollars_string(), "-$2.50");
        assert_eq!(Micros(1_234_567).as_cents(), 123);
    }

    #[test]
    fn rollup_groups_every_dimension_and_totals_exactly() {
        let rows = vec![
            row("r1", "shared", "local", 1_000_000),
            row("r1", "shared", "aws", 2_500_000),
            row("r2", "research/dev", "aws", 500_000),
        ];
        let out = rollup(&rows);
        assert_eq!(out.rows, 3);
        assert_eq!(out.total, Micros(4_000_000));
        assert_eq!(out.by_run["r1"], Micros(3_500_000));
        assert_eq!(out.by_run["r2"], Micros(500_000));
        assert_eq!(out.by_tenant["shared"], Micros(3_500_000));
        assert_eq!(out.by_provider["aws"], Micros(3_000_000));
        assert_eq!(out.by_provider["local"], Micros(1_000_000));
        assert_eq!(out.total.as_cents(), 400);
    }

    #[test]
    fn a_missing_cap_refuses_rather_than_permitting_everything() {
        let cap = BudgetCap::default();
        assert_eq!(cap, BudgetCap::REFUSE_ALL);
        assert!(!cap.allows(Micros(1)), "default cap must refuse any spend");
        assert!(cap.allows(Micros::ZERO), "a zero-spend job is not over cap");
    }

    #[test]
    fn clinical_rows_never_leave_the_box() {
        assert!(row("r", "shared", "aws", 1).may_leave_box());
        assert!(row("r", "research/dev", "aws", 1).may_leave_box());
        assert!(
            !row("r", "clinical/prod", "aws", 1).may_leave_box(),
            "a restricted tenant's spend is local-only (ADR 0061)"
        );
        assert!(
            !row("r", "../not-a-tenant", "aws", 1).may_leave_box(),
            "an unparseable tenant is treated as restricted, not assumed safe"
        );
    }

    #[test]
    fn forecast_projects_linearly_and_flags_the_cap() {
        let cap = BudgetCap(Micros::from_cents(1000)); // $10.00
        let f = forecast(Micros::from_cents(250), 0.25, cap);
        assert_eq!(f.projected_total, Some(Micros::from_cents(1000)));
        assert!(!f.projected_over_cap, "exactly at cap is allowed");

        let over = forecast(Micros::from_cents(300), 0.25, cap);
        assert_eq!(over.projected_total, Some(Micros::from_cents(1200)));
        assert!(over.projected_over_cap);
    }

    #[test]
    fn zero_progress_yields_no_projection_not_a_free_run() {
        for progress in [0.0, -1.0, 1.5, f64::NAN] {
            let f = forecast(Micros::from_cents(500), progress, BudgetCap::REFUSE_ALL);
            assert!(
                f.projected_total.is_none(),
                "progress {progress} must not fabricate a projection"
            );
            assert!(!f.projected_over_cap);
        }
    }

    #[test]
    fn rows_round_trip_through_a_ledger_line() {
        let r = row("r1", "clinical/prod", "local", 12_345);
        assert_eq!(CostRow::from_line(&r.to_line()), Some(r));
        assert!(CostRow::from_line("not json").is_none());
    }
}
