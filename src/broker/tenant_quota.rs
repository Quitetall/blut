// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Stateful per-tenant RAM reservations (ADR 0096 M2.1).
//!
//! The existing admission decision is whole-job shaped: it compares a recipe's
//! resolved peak [`Footprint`] against box capacity and a tenant sub-envelope.
//! This module owns the missing state around that pure decision. Reservation and
//! increment happen under one mutex; the returned RAII guard decrements on every
//! completion, error, cancellation, or unwind path.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::broker::admission::{AdmitDecision, TenantBudget, decide_with_tenant_quota};
use crate::broker::{Footprint, GIB, ResourceSnapshot};
use crate::error::{Result, TrainError};
use crate::tenant::Tenant;

/// Prepared tenant admission for one launch path: validated configured share +
/// one coherent resource snapshot. Recipe, HPO, and resume all use this seam so
/// policy resolution, semaphore sizing, and reservation cannot drift apart.
#[derive(Debug)]
pub struct TenantAdmission {
    tenant: Tenant,
    fraction: f64,
    snapshot: ResourceSnapshot,
}

impl TenantAdmission {
    pub fn prepare(tenant: Tenant) -> Result<Self> {
        let policy = crate::config::tenants::TenantQuotaPolicy::load()?;
        let fraction = policy.fraction_for(&tenant)?;
        Ok(Self::from_snapshot(tenant, fraction, launch_snapshot()))
    }

    fn from_snapshot(tenant: Tenant, fraction: f64, snapshot: ResourceSnapshot) -> Self {
        Self {
            tenant,
            fraction,
            snapshot,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_snapshot_for_test(
        tenant: Tenant,
        fraction: f64,
        snapshot: ResourceSnapshot,
    ) -> Self {
        Self::from_snapshot(tenant, fraction, snapshot)
    }

    /// The immutable resource snapshot captured for this launch. Worker/batch
    /// tuning, executor sizing, and quota reservation must all read this same
    /// value rather than probing independently.
    pub fn snapshot(&self) -> &ResourceSnapshot {
        &self.snapshot
    }

    /// Executor semaphore capacity for this tenant from the one launch
    /// snapshot. The bound is the smaller of the tenant's total-RAM slice and
    /// currently available RAM after the OS floor; this prevents concurrent
    /// stages/cells selected from the same snapshot from each independently
    /// consuming the full live-free envelope.
    ///
    /// `None` preserves historical single-tenant behavior when the platform
    /// cannot probe MemTotal. A fractional policy cannot be enforced without
    /// capacity data and is refused.
    pub fn executor_budget_gib(&self, floor_gib: f64) -> Result<Option<u32>> {
        if self.snapshot.mem_total_gb <= 0.0 {
            if self.fraction == 1.0 {
                return Ok(None);
            }
            return Err(TrainError::other(format!(
                "cannot enforce tenant '{}' fraction {:.6}: MemTotal probe unavailable",
                self.tenant, self.fraction
            )));
        }
        let tenant_ceiling =
            TenantQuotaTracker::ceiling_gib(&self.snapshot, floor_gib, self.fraction)?;
        let live_ceiling =
            if self.snapshot.mem_avail_gb.is_finite() && self.snapshot.mem_avail_gb > 0.0 {
                (self.snapshot.mem_avail_gb - floor_gib).max(0.0)
            } else {
                0.0
            };
        let ceiling = tenant_ceiling.min(live_ceiling);
        if ceiling < 1.0 {
            return Err(TrainError::other(format!(
                "tenant '{}' effective RAM ceiling {ceiling:.3} GiB (tenant {tenant_ceiling:.3} GiB, live {live_ceiling:.3} GiB) is below the executor's 1 GiB accounting quantum",
                self.tenant,
            )));
        }
        Ok(Some(ceiling.floor() as u32))
    }

    pub fn reserve(&self, footprint: &Footprint, floor_gib: f64) -> Result<TenantQuotaReservation> {
        process_tracker().reserve(
            &self.snapshot,
            footprint,
            floor_gib,
            &self.tenant,
            self.fraction,
        )
    }
}

#[derive(Debug, Default)]
struct TrackerInner {
    in_flight_gib: Mutex<HashMap<Tenant, f64>>,
}

/// Process-wide-capable tenant RAM accountant. Clones share the same state.
#[derive(Clone, Debug, Default)]
pub struct TenantQuotaTracker {
    inner: Arc<TrackerInner>,
}

impl TenantQuotaTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tenant's exact usable-RAM ceiling for one snapshot.
    pub fn ceiling_gib(snapshot: &ResourceSnapshot, floor_gib: f64, fraction: f64) -> Result<f64> {
        if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
            return Err(TrainError::other(format!(
                "invalid tenant quota fraction {fraction}; expected finite (0, 1]"
            )));
        }
        Ok((snapshot.mem_total_gb - floor_gib).max(0.0) * fraction)
    }

    /// Atomically decide and reserve `footprint` against the tenant's fraction
    /// of usable RAM (`MemTotal - floor_gib`). Invalid fractions fail closed.
    pub fn reserve(
        &self,
        snapshot: &ResourceSnapshot,
        footprint: &Footprint,
        floor_gib: f64,
        tenant: &Tenant,
        fraction: f64,
    ) -> Result<TenantQuotaReservation> {
        Self::ceiling_gib(snapshot, floor_gib, fraction).map_err(|_| {
            TrainError::other(format!(
                "invalid tenant '{tenant}' quota fraction {fraction}; expected finite (0, 1]"
            ))
        })?;

        let mut state = self
            .inner
            .in_flight_gib
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let reserved_gib = footprint.ram_bytes as f64 / GIB as f64;

        // Preserve the pre-tenancy gate's probe-miss behavior for a 100% share:
        // an unreadable MemTotal admitted rather than wedging every non-Linux /
        // sandboxed launch. A fractional share cannot be measured, so refuse it.
        if snapshot.mem_total_gb <= 0.0 {
            if fraction < 1.0 {
                return Err(TrainError::other(format!(
                    "cannot enforce tenant '{tenant}' fraction {fraction:.6}: MemTotal probe unavailable"
                )));
            }
            *state.entry(tenant.clone()).or_insert(0.0) += reserved_gib;
            drop(state);
            return Ok(TenantQuotaReservation {
                inner: self.inner.clone(),
                tenant: tenant.clone(),
                reserved_gib,
            });
        }

        let in_flight_gb = state.get(tenant).copied().unwrap_or(0.0);
        let budget = TenantBudget::from_fraction(
            tenant.to_string(),
            in_flight_gb,
            snapshot.mem_total_gb,
            floor_gib,
            fraction,
        );
        match decide_with_tenant_quota(snapshot, footprint, floor_gib, Some(&budget)) {
            AdmitDecision::Refuse { reason } => return Err(TrainError::other(reason)),
            AdmitDecision::Admit { .. } => {}
        }

        *state.entry(tenant.clone()).or_insert(0.0) += reserved_gib;
        drop(state);

        Ok(TenantQuotaReservation {
            inner: self.inner.clone(),
            tenant: tenant.clone(),
            reserved_gib,
        })
    }

    /// Current in-flight reservation for diagnostics/tests.
    pub fn in_flight_gib(&self, tenant: &Tenant) -> f64 {
        self.inner
            .in_flight_gib
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(tenant)
            .copied()
            .unwrap_or(0.0)
    }
}

/// Whole-job tenant RAM reservation. Drop is the release operation.
#[derive(Debug)]
pub struct TenantQuotaReservation {
    inner: Arc<TrackerInner>,
    tenant: Tenant,
    reserved_gib: f64,
}

impl Drop for TenantQuotaReservation {
    fn drop(&mut self) {
        let mut state = self
            .inner
            .in_flight_gib
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(current) = state.get_mut(&self.tenant) else {
            tracing::error!(
                tenant = %self.tenant,
                "tenant quota reservation released without matching accounting entry"
            );
            return;
        };
        *current = (*current - self.reserved_gib).max(0.0);
        if *current <= f64::EPSILON {
            state.remove(&self.tenant);
        }
    }
}

/// Shared accountant used by CLI launches in this process. Injectable
/// [`TenantQuotaTracker`] instances keep tests and embedders hermetic.
pub fn process_tracker() -> &'static TenantQuotaTracker {
    static TRACKER: OnceLock<TenantQuotaTracker> = OnceLock::new();
    TRACKER.get_or_init(TenantQuotaTracker::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(total: f64, available: f64) -> ResourceSnapshot {
        ResourceSnapshot {
            mem_total_gb: total,
            mem_avail_gb: available,
            ..ResourceSnapshot::default()
        }
    }

    #[test]
    fn executor_budget_is_exactly_bounded_or_refused_when_unrepresentable() {
        let tenant = Tenant::parse("research/dev").unwrap();
        let half = TenantAdmission::from_snapshot(tenant.clone(), 0.5, snap(32.0, 32.0));
        assert_eq!(half.executor_budget_gib(6.0).unwrap(), Some(13));

        let sub_gib = TenantAdmission::from_snapshot(tenant.clone(), 0.25, snap(8.0, 8.0));
        assert!(sub_gib.executor_budget_gib(6.0).is_err());

        let live_limited = TenantAdmission::from_snapshot(tenant.clone(), 1.0, snap(64.0, 10.0));
        assert_eq!(live_limited.executor_budget_gib(6.0).unwrap(), Some(4));
        let live_sub_gib = TenantAdmission::from_snapshot(tenant.clone(), 1.0, snap(64.0, 6.5));
        assert!(live_sub_gib.executor_budget_gib(6.0).is_err());

        let probe_miss_full =
            TenantAdmission::from_snapshot(tenant.clone(), 1.0, ResourceSnapshot::default());
        assert_eq!(probe_miss_full.executor_budget_gib(6.0).unwrap(), None);
        let probe_miss_fractional =
            TenantAdmission::from_snapshot(tenant, 0.5, ResourceSnapshot::default());
        assert!(probe_miss_fractional.executor_budget_gib(6.0).is_err());
    }
}

/// The snapshot a launch admits against.
///
/// Production probes the live machine. Under `cfg(test)` it is a FIXED
/// snapshot, because a unit test must not depend on how much RAM the host
/// happens to have free at that instant. Two `cli::partition` backfill tests
/// drove real admission and failed on CI with "effective RAM ceiling 0.101 GiB
/// ... is below the executor's 1 GiB accounting quantum" — the broker was
/// right, the runner genuinely had ~100 MiB free mid-suite, and the tests were
/// asserting on partition orchestration rather than on admission at all.
///
/// This removes NO coverage. The refusal path is tested deliberately and
/// explicitly, with hand-built snapshots, in this module's own tests
/// (`live_sub_gib.executor_budget_gib(..).is_err()`), which is where a
/// low-memory assertion belongs. A sibling partition test already injects a
/// fixed snapshot via `from_snapshot_for_test` for the same reason; this
/// extends that idiom to the tests that go through the CLI entry point and
/// therefore cannot pass one in.
fn launch_snapshot() -> ResourceSnapshot {
    #[cfg(test)]
    {
        ResourceSnapshot {
            mem_total_gb: 64.0,
            mem_avail_gb: 32.0,
            ..Default::default()
        }
    }
    #[cfg(not(test))]
    {
        ResourceSnapshot::probe()
    }
}
