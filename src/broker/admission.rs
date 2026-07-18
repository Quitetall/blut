// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Admission decision — the pure RAM-refuse gate (ADR 0046, slice-1
//! item 4). Ports `tools/blut_admit.sh` L26-35.
//!
//! NO poll-queue. The review verified `scheduler_lock` already
//! serializes blut-vs-blut GPU jobs (fail-fast on a held lock), so the
//! broker's only job here is a single-job over-subscription guard: a
//! job whose footprint can't fit free RAM (or can't fit the box at
//! all) is refused cleanly BEFORE the scheduler lock is taken — no
//! launch, no transient unit, no OOM.
//!
//! VRAM is a courtesy pre-check only (it never takes the box down — a
//! CUDA OOM is a process exception, not a kernel OOM). Slice-1 keeps
//! the VRAM branch behind an explicit footprint hint; with `vram_mib
//! == 0` (the default), VRAM never refuses.

use crate::broker::footprint::{Footprint, GIB};
use crate::broker::probe::ResourceSnapshot;

/// Default RAM headroom (GiB) to keep free for the OS / page cache.
/// Mirrors `blut_admit.sh`'s `floor=6`.
pub const DEFAULT_FLOOR_GIB: f64 = 6.0;

/// The admission verdict. Slice-1 has NO `Wait` variant — there is no
/// poll-queue (the scheduler lock is the serializer). A job either
/// proceeds or is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmitDecision {
    /// Footprint fits; proceed. Carries the cgroup `MemoryMax` cap the
    /// launcher should apply (footprint + headroom).
    Admit { memmax_bytes: u64 },
    /// Cannot launch. `reason` is an operator-facing message.
    Refuse { reason: String },
}

/// Decide admission. PURE — no I/O; the caller supplies the probed
/// snapshot. Order matches `blut_admit.sh`:
///
///   1. footprint > (box total − floor) ⇒ Refuse (never fits)
///   2. (avail − need) < floor          ⇒ Refuse (would oversubscribe)
///   3. VRAM courtesy check (only when footprint.vram_mib > 0)
///   4. otherwise Admit{ memmax = footprint + headroom }
pub fn decide(snap: &ResourceSnapshot, fp: &Footprint, floor_gib: f64) -> AdmitDecision {
    let gib = GIB as f64;
    let floor = floor_gib;
    let need_gb = fp.ram_bytes as f64 / gib;
    let total_gb = snap.mem_total_gb;
    let avail_gb = snap.mem_avail_gb;

    // (1) Impossible fit on this box, even idle. Refuse — don't spin.
    if need_gb > (total_gb - floor) {
        return AdmitDecision::Refuse {
            reason: format!(
                "RAM footprint {need_gb:.1}G exceeds box capacity \
                 ({total_gb:.1}G total − {floor:.0}G floor)"
            ),
        };
    }

    // (2) Oversubscription right now: launching would leave < floor
    // free. Refuse cleanly (no poll-queue; scheduler_lock serializes).
    if avail_gb - need_gb < floor {
        return AdmitDecision::Refuse {
            reason: format!(
                "RAM footprint {need_gb:.1}G would leave < {floor:.0}G free \
                 (only {avail_gb:.1}G available now) — run it alone, lower \
                 LMA_NUM_WORKERS, or wait for the running job to finish"
            ),
        };
    }

    // (3) VRAM courtesy pre-check (best-effort, never a box guarantee).
    if fp.vram_mib > 0 {
        if let Some(total) = snap.vram_total_mib
            && fp.vram_mib > total
        {
            return AdmitDecision::Refuse {
                reason: format!(
                    "VRAM footprint {}MiB exceeds card capacity ({}MiB)",
                    fp.vram_mib, total
                ),
            };
        }
        if let Some(free) = snap.vram_free_mib
            && fp.vram_mib > free
        {
            return AdmitDecision::Refuse {
                reason: format!(
                    "VRAM footprint {}MiB > {}MiB free — serialize big-VRAM jobs",
                    fp.vram_mib, free
                ),
            };
        }
    }

    AdmitDecision::Admit {
        memmax_bytes: fp.memmax_bytes(),
    }
}

/// A tenant's slice of the box's admission envelope (ADR 0096 per-tenant quota).
///
/// The box-level admission gate ([`decide`]) already stops any single job from
/// OOMing the box. This adds a SECOND, tenant-scoped ceiling so one tenant's
/// backlog can't monopolise the whole box even when RAM is free — the fairness
/// primitive the multi-tenancy ADR needs. It is PURE state supplied by the
/// caller (which tracks per-tenant in-flight RAM and releases on completion),
/// same dependency-injection shape as the probed [`ResourceSnapshot`].
#[derive(Clone, Debug, PartialEq)]
pub struct TenantBudget {
    /// Tenant id (`project[/domain]`) — for the refusal message only.
    pub tenant: String,
    /// RAM (GiB) already admitted for this tenant and not yet released.
    pub in_flight_gb: f64,
    /// This tenant's maximum aggregate in-flight RAM (GiB).
    pub ceiling_gb: f64,
}

impl TenantBudget {
    /// A ceiling that is `fraction` of the box's usable RAM (`total − floor`).
    /// `fraction` is clamped to `(0, 1]`; a non-finite or ≤0 fraction yields the
    /// whole usable box (no effective quota — fail-OPEN on a nonsense config, so
    /// a misconfigured quota never wedges a tenant, only the box gate binds).
    pub fn from_fraction(
        tenant: impl Into<String>,
        in_flight_gb: f64,
        box_total_gb: f64,
        floor_gib: f64,
        fraction: f64,
    ) -> Self {
        let usable = (box_total_gb - floor_gib).max(0.0);
        let frac = if fraction.is_finite() && fraction > 0.0 {
            fraction.min(1.0)
        } else {
            1.0
        };
        Self {
            tenant: tenant.into(),
            in_flight_gb,
            ceiling_gb: usable * frac,
        }
    }
}

/// Tenant-aware admission (ADR 0096): the box gate FIRST, then the tenant
/// ceiling. The tenant quota NEVER bypasses [`decide`] — a job that can't fit
/// the box is refused regardless of quota headroom (admission is never
/// weakened, ADR 0046/0047). `budget = None` is byte-identical to [`decide`],
/// so every existing un-tenanted path is unchanged.
pub fn decide_with_tenant_quota(
    snap: &ResourceSnapshot,
    fp: &Footprint,
    floor_gib: f64,
    budget: Option<&TenantBudget>,
) -> AdmitDecision {
    let box_decision = decide(snap, fp, floor_gib);
    let AdmitDecision::Admit { .. } = box_decision else {
        return box_decision; // box refusal dominates — never softened by a quota
    };
    let Some(b) = budget else {
        return box_decision;
    };
    let need_gb = fp.ram_bytes as f64 / GIB as f64;
    if b.in_flight_gb + need_gb > b.ceiling_gb {
        return AdmitDecision::Refuse {
            reason: format!(
                "tenant '{}' RAM quota exceeded: {:.1}G in-flight + {:.1}G new \
                 > {:.1}G ceiling — a tenant can't monopolise the box; wait for \
                 this tenant's running jobs to finish",
                b.tenant, b.in_flight_gb, need_gb, b.ceiling_gb
            ),
        };
    }
    box_decision
}

/// Probe the live box and decide whether `fp` may launch. `Ok(())` =
/// admit; `Err(reason)` = refuse (operator-facing). A probe miss
/// (`mem_total_gb == 0`, e.g. non-Linux / sandbox where `/proc/meminfo`
/// is unreadable) admits rather than refusing every job. `label` names
/// the job for the refusal message.
///
/// THE single admission entry point for every launch path (recipe AND
/// the legacy bare-spawn train path) so neither can launch un-gated.
pub fn gate(label: &str, fp: &Footprint) -> Result<(), String> {
    let snap = ResourceSnapshot::probe();
    if snap.mem_total_gb <= 0.0 {
        return Ok(()); // probe miss — don't refuse on missing data
    }
    match decide(&snap, fp, DEFAULT_FLOOR_GIB) {
        AdmitDecision::Admit { .. } => Ok(()),
        AdmitDecision::Refuse { reason } => {
            Err(format!("resource admission refused: {reason} ({label})"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(total_gb: f64, avail_gb: f64) -> ResourceSnapshot {
        ResourceSnapshot {
            mem_total_gb: total_gb,
            mem_avail_gb: avail_gb,
            vram_total_mib: None,
            vram_free_mib: None,
            gpus: Vec::new(),
        }
    }

    fn fp_ram(gib: u64) -> Footprint {
        Footprint {
            ram_bytes: gib * GIB,
            vram_mib: 0,
        }
    }

    #[test]
    fn fits_admits_with_memmax_headroom() {
        // 62G box, 50G free, 30G job → fits, admit with +2G memmax.
        let d = decide(&snap(62.0, 50.0), &fp_ram(30), DEFAULT_FLOOR_GIB);
        match d {
            AdmitDecision::Admit { memmax_bytes } => {
                assert_eq!(memmax_bytes, (30 + 2) * GIB);
            }
            other => panic!("expected Admit, got {other:?}"),
        }
    }

    #[test]
    fn ram_over_free_refuses() {
        // 62G box, only 20G free, 30G job → would leave < 6G floor.
        let d = decide(&snap(62.0, 20.0), &fp_ram(30), DEFAULT_FLOOR_GIB);
        assert!(matches!(d, AdmitDecision::Refuse { .. }), "got {d:?}");
    }

    #[test]
    fn ram_over_box_refuses() {
        // 62G box, even fully idle, a 60G job leaves < 6G floor → never
        // fits → Refuse (the impossible-fit branch, distinct from #2).
        let d = decide(&snap(62.0, 62.0), &fp_ram(60), DEFAULT_FLOOR_GIB);
        match d {
            AdmitDecision::Refuse { reason } => {
                assert!(reason.contains("box capacity"), "wrong branch: {reason}");
            }
            other => panic!("expected Refuse(box capacity), got {other:?}"),
        }
    }

    #[test]
    fn exactly_at_floor_boundary_admits() {
        // avail - need == floor exactly → NOT < floor → admit.
        // 62G box, 36G free, 30G job → 6G left == floor.
        let d = decide(&snap(62.0, 36.0), &fp_ram(30), DEFAULT_FLOOR_GIB);
        assert!(matches!(d, AdmitDecision::Admit { .. }), "got {d:?}");
    }

    #[test]
    fn tenant_quota_none_is_identical_to_box_decide() {
        // ADR 0096: budget=None ⇒ byte-identical to `decide` (un-tenanted parity).
        let s = snap(62.0, 50.0);
        let fp = fp_ram(30);
        assert_eq!(
            decide_with_tenant_quota(&s, &fp, DEFAULT_FLOOR_GIB, None),
            decide(&s, &fp, DEFAULT_FLOOR_GIB)
        );
    }

    #[test]
    fn tenant_over_ceiling_refuses_even_when_box_has_room() {
        // Box has 50G free (30G job fits the box), but the tenant already has
        // 40G in-flight against a 56G ceiling → 40+30 > 56 ⇒ tenant-quota refuse.
        let b = TenantBudget {
            tenant: "research/prod".into(),
            in_flight_gb: 40.0,
            ceiling_gb: 56.0,
        };
        let d =
            decide_with_tenant_quota(&snap(62.0, 50.0), &fp_ram(30), DEFAULT_FLOOR_GIB, Some(&b));
        match d {
            AdmitDecision::Refuse { reason } => {
                assert!(reason.contains("research/prod") && reason.contains("quota exceeded"));
            }
            other => panic!("expected tenant-quota Refuse, got {other:?}"),
        }
    }

    #[test]
    fn tenant_under_ceiling_admits() {
        // Same box, tenant has only 10G in-flight → 10+30 = 40 ≤ 56 ⇒ admit.
        let b = TenantBudget {
            tenant: "research/prod".into(),
            in_flight_gb: 10.0,
            ceiling_gb: 56.0,
        };
        let d =
            decide_with_tenant_quota(&snap(62.0, 50.0), &fp_ram(30), DEFAULT_FLOOR_GIB, Some(&b));
        assert!(matches!(d, AdmitDecision::Admit { .. }), "got {d:?}");
    }

    #[test]
    fn box_refusal_dominates_tenant_quota() {
        // A 60G job never fits the 62G box (< floor). Even with an enormous tenant
        // ceiling and zero in-flight, the BOX gate refuses first — the quota can
        // never soften admission (ADR 0046/0047 never bypassed).
        let b = TenantBudget {
            tenant: "research/prod".into(),
            in_flight_gb: 0.0,
            ceiling_gb: 1_000.0,
        };
        let d =
            decide_with_tenant_quota(&snap(62.0, 62.0), &fp_ram(60), DEFAULT_FLOOR_GIB, Some(&b));
        match d {
            AdmitDecision::Refuse { reason } => {
                assert!(
                    reason.contains("box capacity"),
                    "must be the BOX branch: {reason}"
                );
            }
            other => panic!("expected box Refuse, got {other:?}"),
        }
    }

    #[test]
    fn from_fraction_computes_ceiling_and_fails_open_on_nonsense() {
        // 62G box, 6G floor ⇒ 56G usable; a 0.5 fraction ⇒ 28G ceiling.
        let b = TenantBudget::from_fraction("t", 0.0, 62.0, DEFAULT_FLOOR_GIB, 0.5);
        assert!((b.ceiling_gb - 28.0).abs() < 1e-9, "got {}", b.ceiling_gb);
        // A nonsense fraction (≤0 / non-finite) fails OPEN to the whole usable box.
        let whole = TenantBudget::from_fraction("t", 0.0, 62.0, DEFAULT_FLOOR_GIB, 0.0);
        assert!(
            (whole.ceiling_gb - 56.0).abs() < 1e-9,
            "got {}",
            whole.ceiling_gb
        );
        let nan = TenantBudget::from_fraction("t", 0.0, 62.0, DEFAULT_FLOOR_GIB, f64::NAN);
        assert!(
            (nan.ceiling_gb - 56.0).abs() < 1e-9,
            "got {}",
            nan.ceiling_gb
        );
        // A >1 fraction is clamped to the whole usable box, never above it.
        let over = TenantBudget::from_fraction("t", 0.0, 62.0, DEFAULT_FLOOR_GIB, 2.0);
        assert!(
            (over.ceiling_gb - 56.0).abs() < 1e-9,
            "got {}",
            over.ceiling_gb
        );
    }

    #[test]
    fn vram_over_card_refuses_when_known() {
        let mut s = snap(62.0, 50.0);
        s.vram_total_mib = Some(24_000);
        s.vram_free_mib = Some(24_000);
        let fp = Footprint {
            ram_bytes: 30 * GIB,
            vram_mib: 30_000, // > 24G card
        };
        match decide(&s, &fp, DEFAULT_FLOOR_GIB) {
            AdmitDecision::Refuse { reason } => assert!(reason.contains("card capacity")),
            other => panic!("expected VRAM refuse, got {other:?}"),
        }
    }

    #[test]
    fn vram_over_free_refuses_when_known() {
        let mut s = snap(62.0, 50.0);
        s.vram_total_mib = Some(24_000);
        s.vram_free_mib = Some(8_000);
        let fp = Footprint {
            ram_bytes: 30 * GIB,
            vram_mib: 18_000, // fits card, > free
        };
        assert!(matches!(
            decide(&s, &fp, DEFAULT_FLOOR_GIB),
            AdmitDecision::Refuse { .. }
        ));
    }

    #[test]
    fn zero_vram_hint_never_refuses_on_vram() {
        // The slice-1 default (vram_mib == 0) must not refuse on VRAM
        // even when the card is reported full.
        let mut s = snap(62.0, 50.0);
        s.vram_total_mib = Some(24_000);
        s.vram_free_mib = Some(0);
        let d = decide(&s, &fp_ram(30), DEFAULT_FLOOR_GIB);
        assert!(matches!(d, AdmitDecision::Admit { .. }), "got {d:?}");
    }
}
