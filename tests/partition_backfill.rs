// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Named progress gate for ADR 0101's landed partition slices.
//!
//! This does not claim the full ADR deliverable. It covers the persisted
//! missing-cell backfill list and the five-state read-side matrix; partitioned
//! cache keys, stale-target selection, lineage-derived staleness, and
//! Restricted admission enforcement remain later increments.

use std::collections::BTreeMap;

use blut::config::partition::{
    CellStatus, PartitionDim, PartitionSet, PartitionStatus, status_matrix,
};

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn partition_backfill_gate_covers_landed_persistence_and_matrix_slices() {
    let _env = ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempfile::tempdir().expect("partition tempdir");
    // SAFETY: ENV_LOCK serializes this process-global mutation with every test
    // in this integration target. Future tests in this file must take it too.
    unsafe { std::env::set_var("BLUT_PARTITIONS_DIR", temp.path()) };

    let set = PartitionSet {
        name: "by_corpus".into(),
        recipe: "codec_eval".into(),
        dims: vec![PartitionDim {
            axis: "corpus".into(),
            values: vec![
                "fresh".into(),
                "stale".into(),
                "failed".into(),
                "missing".into(),
                "phi".into(),
            ],
        }],
    };
    assert_eq!(set.validate().expect("valid partition set"), 5);

    let record = |key: &str, outcome: &str| PartitionStatus {
        key: format!("corpus={key}"),
        job_id: format!("job-{key}"),
        outcome: outcome.into(),
        recorded_at: 1,
    };
    let statuses = [
        record("fresh", "done"),
        record("stale", "done"),
        record("failed", "failed"),
        record("phi", "done"),
    ];
    for status in &statuses {
        set.record_status(status).expect("append status");
    }

    let backfill: Vec<_> = set
        .backfill_targets(false)
        .expect("derive current backfill list")
        .into_iter()
        .map(|cell| cell.key)
        .collect();
    assert_eq!(backfill, ["corpus=failed", "corpus=missing"]);
    assert_eq!(set.backfill_targets(true).expect("forced list").len(), 5);

    let latest: BTreeMap<_, _> = statuses
        .into_iter()
        .map(|status| (status.key.clone(), status))
        .collect();
    let matrix: BTreeMap<_, _> = status_matrix(
        &set.cells(),
        &latest,
        |key| key == "corpus=phi",
        |key| key == "corpus=stale",
    )
    .into_iter()
    .collect();
    assert_eq!(matrix["corpus=fresh"], CellStatus::Materialized);
    assert_eq!(matrix["corpus=stale"], CellStatus::Stale);
    assert_eq!(matrix["corpus=failed"], CellStatus::Failed);
    assert_eq!(matrix["corpus=missing"], CellStatus::Missing);
    assert_eq!(matrix["corpus=phi"], CellStatus::Restricted);

    // SAFETY: same ENV_LOCK guarantee as above.
    unsafe { std::env::remove_var("BLUT_PARTITIONS_DIR") };
}
