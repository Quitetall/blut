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
        if let Some(total) = snap.vram_total_mib {
            if fp.vram_mib > total {
                return AdmitDecision::Refuse {
                    reason: format!(
                        "VRAM footprint {}MiB exceeds card capacity ({}MiB)",
                        fp.vram_mib, total
                    ),
                };
            }
        }
        if let Some(free) = snap.vram_free_mib {
            if fp.vram_mib > free {
                return AdmitDecision::Refuse {
                    reason: format!(
                        "VRAM footprint {}MiB > {}MiB free — serialize big-VRAM jobs",
                        fp.vram_mib, free
                    ),
                };
            }
        }
    }

    AdmitDecision::Admit {
        memmax_bytes: fp.memmax_bytes(),
    }
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
            gpu_count: 0,
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
