// SPDX-License-Identifier: AGPL-3.0-or-later
// The ADR 0094 acceptance gate (trigger half): a file dropped in a watched
// spool dispatches EXACTLY ONE broker-admitted run of the bound plan; a second
// identical drop does NOT re-dispatch (content-hash dedupe); and the dispatch is
// genuinely admission-gated — an oversubscribed box REFUSES the triggered run
// rather than bypassing the envelope. Uses the REAL `broker::admission::decide`
// (only the launch is recorded), so "broker-admitted" is proven, not mocked.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use blut::broker::Footprint;
use blut::broker::GIB;
use blut::broker::probe::{GpuInfo, ResourceSnapshot};
use blut::trigger::{
    AdmissionDispatcher, DispatchOutcome, FileDropTrigger, SeenStore, TriggerEvent, drive_once,
};

fn snapshot(mem_total_gb: f64, mem_avail_gb: f64) -> ResourceSnapshot {
    ResourceSnapshot {
        mem_total_gb,
        mem_avail_gb,
        vram_total_mib: None,
        vram_free_mib: None,
        gpus: Vec::<GpuInfo>::new(),
    }
}

fn dispatcher(snap: ResourceSnapshot, counter: Arc<AtomicUsize>) -> AdmissionDispatcher {
    AdmissionDispatcher {
        snapshot: snap,
        footprint: Footprint {
            ram_bytes: 4 * GIB,
            vram_mib: 0,
        },
        floor_gib: 6.0,
        launch: Box::new(move |_plan: &str, _ev: &TriggerEvent| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }),
    }
}

#[test]
fn file_drop_dispatches_once_dedupes_and_is_admission_gated() {
    let td = tempfile::tempdir().unwrap();
    let spool = td.path().join("spool");
    std::fs::create_dir_all(&spool).unwrap();
    std::fs::write(spool.join("run-me.json"), "{\"plan\":\"demo\"}").unwrap();

    let trig = FileDropTrigger::new("nightly", &spool, Some("json".into()));
    let seen_path = td.path().join("seen");
    let mut seen = SeenStore::load(&seen_path).unwrap();

    // Roomy box → admit. First drive dispatches exactly once.
    let launches = Arc::new(AtomicUsize::new(0));
    let disp = dispatcher(snapshot(64.0, 50.0), launches.clone());
    let r1 = drive_once(&trig, "demo_plan", &mut seen, &disp).unwrap();
    assert_eq!(r1.len(), 1, "one fired event");
    assert_eq!(r1[0].1, DispatchOutcome::Admitted, "admitted by the broker");
    assert_eq!(launches.load(Ordering::SeqCst), 1, "launched exactly once");

    // Second drive over the SAME file → deduped, no re-dispatch.
    let r2 = drive_once(&trig, "demo_plan", &mut seen, &disp).unwrap();
    assert!(r2.is_empty(), "identical drop must not re-dispatch");
    assert_eq!(launches.load(Ordering::SeqCst), 1, "still one launch");

    // Dedupe survives a daemon restart (seen-store reloads from disk).
    let mut seen_reloaded = SeenStore::load(&seen_path).unwrap();
    let r3 = drive_once(&trig, "demo_plan", &mut seen_reloaded, &disp).unwrap();
    assert!(r3.is_empty(), "dedupe must persist across restart");
    assert_eq!(launches.load(Ordering::SeqCst), 1);

    // Admission-gated proof: a NEW event on an OVERSUBSCRIBED box is REFUSED,
    // never launched, and stays eligible (not marked seen) for a later retry.
    std::fs::write(spool.join("run-me-2.json"), "{\"plan\":\"demo\"}").unwrap();
    let tight_launches = Arc::new(AtomicUsize::new(0));
    let tight = dispatcher(snapshot(64.0, 5.0), tight_launches.clone());
    let r4 = drive_once(&trig, "demo_plan", &mut seen, &tight).unwrap();
    assert_eq!(r4.len(), 1, "the new event fired");
    assert!(
        matches!(r4[0].1, DispatchOutcome::Refused(_)),
        "an oversubscribed box refuses the triggered run (envelope not bypassed)"
    );
    assert_eq!(
        tight_launches.load(Ordering::SeqCst),
        0,
        "a refused trigger must not launch"
    );
    // The refused event retries when capacity returns (roomy box now admits it).
    let recover = dispatcher(snapshot(64.0, 50.0), launches.clone());
    let r5 = drive_once(&trig, "demo_plan", &mut seen, &recover).unwrap();
    assert_eq!(r5.len(), 1, "the previously-refused event retries");
    assert_eq!(r5[0].1, DispatchOutcome::Admitted);
    assert_eq!(
        launches.load(Ordering::SeqCst),
        2,
        "now two distinct launches"
    );
}
