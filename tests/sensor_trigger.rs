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
    AdmissionDispatcher, DispatchOutcome, FileDropTrigger, SeenStore, SpoolDirection, SpoolMetric,
    SpoolThresholdTrigger, Trigger, TriggerEvent, archive_dispatched, drive_once, observable_ids,
    prune_processed,
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

#[test]
fn spool_threshold_has_stable_snapshot_events() {
    let td = tempfile::tempdir().unwrap();
    let trigger = SpoolThresholdTrigger::new(
        "full",
        td.path(),
        SpoolMetric::Files,
        SpoolDirection::AtLeast,
        2,
    );
    std::fs::write(td.path().join("one"), "1").unwrap();
    assert!(trigger.poll().unwrap().is_empty());
    std::fs::write(td.path().join("two"), "2").unwrap();
    let first = trigger.poll().unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(trigger.poll().unwrap()[0].id, first[0].id);
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

/// Retention, the effective half: a dispatched event is archived out of the
/// watched directory, which makes its dedupe record reclaimable — and
/// reclaiming it must NOT resurrect the event. Without this the spool, the
/// `.seen` log, and the `.admitted` markers all grow for the life of the
/// daemon.
#[test]
fn archived_events_free_their_dedupe_records_without_resurrecting() {
    let td = tempfile::tempdir().unwrap();
    let spool = td.path().join("spool");
    std::fs::create_dir_all(&spool).unwrap();
    std::fs::write(spool.join("a.json"), "{\"plan\":\"demo\"}").unwrap();

    let trig = FileDropTrigger::new("retain", &spool, Some("json".into()));
    let seen_path = td.path().join(".seen");
    let mut seen = SeenStore::load(&seen_path).unwrap();
    let launches = Arc::new(AtomicUsize::new(0));
    let disp = dispatcher(snapshot(64.0, 50.0), launches.clone());

    let fired = drive_once(&trig, "demo_plan", &mut seen, &disp).unwrap();
    assert_eq!(fired.len(), 1);
    assert_eq!(fired[0].1, DispatchOutcome::Admitted);
    assert_eq!(seen.len(), 1, "one dedupe record is held while observable");

    // Archive: the event leaves the watched directory and stops being observable.
    assert!(archive_dispatched(&fired[0].0).unwrap());
    assert!(
        trig.poll().unwrap().is_empty(),
        "an archived event must not be observable"
    );

    // Now — and only now — the record is reclaimable.
    let observable = observable_ids(&trig).unwrap();
    assert_eq!(seen.retain_observable(&observable).unwrap(), 1);
    assert_eq!(seen.len(), 0, "the dedupe store shrank");

    // The load-bearing safety property: reclaiming must not re-dispatch.
    let again = drive_once(&trig, "demo_plan", &mut seen, &disp).unwrap();
    assert!(again.is_empty(), "an archived event must never re-dispatch");
    assert_eq!(
        launches.load(Ordering::SeqCst),
        1,
        "still exactly one launch after compaction"
    );

    // Compaction is durable.
    assert_eq!(SeenStore::load(&seen_path).unwrap().len(), 0);
}

/// Retention, the safe half: an event still sitting in the watched directory
/// is STILL OBSERVABLE, so its dedupe record must survive compaction — else
/// the next poll would launch it a second time.
#[test]
fn a_still_observable_event_keeps_its_dedupe_record() {
    let td = tempfile::tempdir().unwrap();
    let spool = td.path().join("spool");
    std::fs::create_dir_all(&spool).unwrap();
    std::fs::write(spool.join("stay.json"), "{}").unwrap();

    let trig = FileDropTrigger::new("stay", &spool, Some("json".into()));
    let mut seen = SeenStore::load(td.path().join(".seen")).unwrap();
    let launches = Arc::new(AtomicUsize::new(0));
    let disp = dispatcher(snapshot(64.0, 50.0), launches.clone());

    drive_once(&trig, "demo_plan", &mut seen, &disp).unwrap();
    assert_eq!(launches.load(Ordering::SeqCst), 1);

    // NOT archived — compaction must keep the record.
    let observable = observable_ids(&trig).unwrap();
    assert_eq!(
        seen.retain_observable(&observable).unwrap(),
        0,
        "an observable event's record must never be reclaimed"
    );
    assert_eq!(seen.len(), 1);

    let again = drive_once(&trig, "demo_plan", &mut seen, &disp).unwrap();
    assert!(again.is_empty(), "no re-dispatch after compaction");
    assert_eq!(launches.load(Ordering::SeqCst), 1, "still one launch");
}

/// `.processed/` is bounded by age, and the sweep never touches a fresh entry.
#[test]
fn prune_processed_removes_only_aged_archives() {
    let td = tempfile::tempdir().unwrap();
    let spool = td.path().join("spool");
    std::fs::create_dir_all(&spool).unwrap();
    std::fs::write(spool.join("old.json"), "{}").unwrap();
    std::fs::write(spool.join("new.json"), "{}").unwrap();
    let trig = FileDropTrigger::new("prune", &spool, Some("json".into()));
    for event in trig.poll().unwrap() {
        assert!(archive_dispatched(&event).unwrap());
    }
    let archive = spool.join(".processed");
    assert_eq!(std::fs::read_dir(&archive).unwrap().count(), 2);

    // Age one archived entry an hour into the past.
    let aged = archive.join("old.json");
    let handle = std::fs::File::options().write(true).open(&aged).unwrap();
    handle
        .set_times(
            std::fs::FileTimes::new()
                .set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(3600)),
        )
        .unwrap();

    assert_eq!(prune_processed(&spool, 60).unwrap(), 1, "only the aged one");
    assert!(!aged.exists());
    assert!(archive.join("new.json").exists());
    // Retention 0 disables pruning entirely.
    assert_eq!(prune_processed(&spool, 0).unwrap(), 0);
}

/// The finding this retention work closes: a long-lived daemon used to grow
/// three stores without bound (every delivered file stayed in the spool, and
/// every id stayed in `.seen` + `.admitted` forever, re-scanned at every poll
/// and every startup). Drive many events through the real dispatch → archive →
/// compact cycle and assert the STEADY STATE is O(1), not O(events), while the
/// launch count still matches exactly one per event.
#[test]
fn retention_keeps_the_daemon_steady_state_bounded() {
    let td = tempfile::tempdir().unwrap();
    let spool = td.path().join("spool");
    std::fs::create_dir_all(&spool).unwrap();
    let trig = FileDropTrigger::new("bounded", &spool, Some("json".into()));
    let seen_path = td.path().join(".seen");
    let mut seen = SeenStore::load(&seen_path).unwrap();
    let launches = Arc::new(AtomicUsize::new(0));
    let disp = dispatcher(snapshot(64.0, 50.0), launches.clone());

    const EVENTS: usize = 200;
    for n in 0..EVENTS {
        std::fs::write(spool.join(format!("ev{n}.json")), format!("{{\"n\":{n}}}")).unwrap();
        for (event, outcome) in drive_once(&trig, "demo_plan", &mut seen, &disp).unwrap() {
            assert_eq!(outcome, DispatchOutcome::Admitted);
            archive_dispatched(&event).unwrap();
        }
        let observable = observable_ids(&trig).unwrap();
        seen.retain_observable(&observable).unwrap();

        // Steady state after every single cycle — never accumulating.
        assert_eq!(seen.len(), 0, "dedupe store must not grow (cycle {n})");
        assert!(
            trig.poll().unwrap().is_empty(),
            "watched dir must be drained (cycle {n})"
        );
    }
    assert_eq!(
        launches.load(Ordering::SeqCst),
        EVENTS,
        "exactly one launch per event"
    );
    // On-disk stores are bounded too: the log is empty and the markers are gone.
    assert_eq!(SeenStore::load(&seen_path).unwrap().len(), 0);
    let admitted = spool.join(".admitted");
    let marker_count = std::fs::read_dir(&admitted).map(|d| d.count()).unwrap_or(0);
    assert_eq!(marker_count, 0, "admission markers reclaimed");
    // The archive holds the history until the age sweep bounds it.
    assert_eq!(
        std::fs::read_dir(spool.join(".processed")).unwrap().count(),
        EVENTS
    );
}
