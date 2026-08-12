// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Named acceptance gate for ADR 0101: typed partition identity, exact legacy
//! cache parity, lineage-derived matrix states, and selector behavior. The
//! in-crate CLI test additionally executes a three-cell backfill and pins the
//! pre-launch Restricted refusal.

use std::collections::BTreeMap;

use blut::config::partition::{
    BackfillSelector, CellStatus, PartitionDim, PartitionKey, PartitionSet, PartitionSpec,
    PartitionStatus, PartitionValue, select_backfill_targets, status_matrix,
};
use blut::framework::stage::{Stage, StageContext};
use blut::framework::{CacheHandle, ContentHash, InvocationKey};
use blut::framework::{Resource, StageError};
use blut::lineage_db::{ArtifactRow, LineageDb, PartitionStatusRow, RunRow};

static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct UnitStage;

#[async_trait::async_trait]
impl Stage for UnitStage {
    const NAME: &'static str = "partition_backfill_unit";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = ();
    type Args = ();

    async fn run(&self, _ctx: &StageContext, _input: (), _args: &()) -> Result<(), StageError> {
        Ok(())
    }
}

#[test]
fn partition_key_is_a_wire_type_and_extends_cache_identity_without_moving_none() {
    let spec = PartitionSpec::Categorical {
        dimension: "corpus".into(),
        values: vec!["tuh".into(), "chbmit".into()],
    };
    let encoded = serde_json::to_string(&spec).unwrap();
    assert_eq!(
        serde_json::from_str::<PartitionSpec>(&encoded).unwrap(),
        spec
    );

    let tuh = PartitionKey::new(vec![PartitionValue::new("corpus", "tuh")]).unwrap();
    let chbmit = PartitionKey::new(vec![PartitionValue::new("corpus", "chbmit")]).unwrap();
    let input = ContentHash([0x11; 32]);
    let args = serde_json::json!({"x":1});
    let none = CacheHandle::key_for_partitioned("stage", 1, input, &args, b"code", None);
    assert_eq!(
        none.to_hex(),
        "165ede22a733784a05e136ede2cfeca4c299dd359a53787b709bba027973bb2b",
        "partition=None must remain byte-identical to the pre-ADR cache key"
    );
    assert_eq!(
        none,
        CacheHandle::key_for("stage", 1, input, &args, b"code")
    );

    let tuh_key = CacheHandle::key_for_partitioned("stage", 1, input, &args, b"code", Some(&tuh));
    let chbmit_key =
        CacheHandle::key_for_partitioned("stage", 1, input, &args, b"code", Some(&chbmit));
    assert_ne!(tuh_key, none);
    assert_ne!(tuh_key, chbmit_key);
    assert!(
        serde_json::from_str::<PartitionKey>(r#"{"values":[]}"#).is_err(),
        "wire decode must preserve the non-empty key invariant"
    );
}

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
        tenant: blut::tenant::DEFAULT_PROJECT.into(),
        key: format!("corpus={key}"),
        job_id: format!("job-{key}"),
        outcome: outcome.into(),
        recorded_at: 1,
        input_fingerprint: None,
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

#[test]
fn partition_backfill_matrix_is_lineage_derived_and_selector_exact() {
    let temp = tempfile::tempdir().expect("lineage tempdir");
    let db = LineageDb::open_at(temp.path().join("lineage.db")).unwrap();
    let record_done = |job_id: &str, key: &str, args: serde_json::Value| {
        let input_fingerprint = blut::config::partition::partition_input_fingerprint(&args, &args);
        let hash = ContentHash::of_bytes(key.as_bytes());
        let stage_dir = temp.path().join(job_id);
        let sidecar = stage_dir.join("output.metadata.json");
        blut::framework::ArtifactMetadata::new("report", 1, hash)
            .with_stage("evaluate")
            .write_to(&sidecar)
            .unwrap();
        let cache_key =
            InvocationKey::from_digest(ContentHash::of_bytes(format!("cache-{key}").as_bytes()));
        let cache_root = temp.path().join("cache");
        let cache = CacheHandle::job_local(cache_root.clone());
        let erased = blut::framework::stage::ErasedArtifact::from_typed(&()).unwrap();
        cache
            .insert(cache_key, &UnitStage, &erased, &stage_dir)
            .unwrap();
        blut::framework::cache::CacheProof {
            key: cache_key,
            entry_path: cache_root
                .join("invocations")
                .join(cache_key.to_hex())
                .join("record.bin"),
        }
        .write_to(&stage_dir.join("cache-proof.json"))
        .unwrap();
        db.record_run(&RunRow {
            job_id: job_id.into(),
            recipe: "codec_eval".into(),
            outcome: Some("done".into()),
            tenant: "research/dev".into(),
            args_json: Some(serde_json::to_string(&args).unwrap()),
            ..RunRow::default()
        })
        .unwrap();
        db.record_artifact(&ArtifactRow {
            job_id: job_id.into(),
            stage_idx: 0,
            stage_name: "evaluate".into(),
            content_hash: hash.to_hex(),
            kind: "report".into(),
            schema_ver: 1,
            sidecar_path: Some(sidecar.display().to_string()),
            produced_unix: Some(1),
        })
        .unwrap();
        db.record_partition_status(&PartitionStatusRow {
            tenant: "research/dev".into(),
            recipe: "codec_eval".into(),
            partition_set: "by_corpus".into(),
            partition_key: key.into(),
            job_id: job_id.into(),
            outcome: "done".into(),
            resolved_args_json: serde_json::to_string(&args).unwrap(),
            input_fingerprint,
            recorded_unix: 1,
        })
        .unwrap();
    };
    record_done(
        "job-fresh",
        "corpus=fresh",
        serde_json::json!({"corpus":"fresh"}),
    );
    record_done(
        "job-stale",
        "corpus=stale",
        serde_json::json!({"corpus":"old"}),
    );
    db.record_partition_status(&PartitionStatusRow {
        tenant: "research/dev".into(),
        recipe: "codec_eval".into(),
        partition_set: "by_corpus".into(),
        partition_key: "corpus=fresh".into(),
        job_id: "older-failure".into(),
        outcome: "failed".into(),
        resolved_args_json: "{}".into(),
        input_fingerprint: String::new(),
        recorded_unix: 0,
    })
    .unwrap();

    let latest = db
        .partition_statuses("research/dev", "codec_eval", "by_corpus")
        .unwrap();
    assert_eq!(
        latest["corpus=fresh"].job_id, "job-fresh",
        "an older concurrent completion cannot overwrite the latest cell row"
    );
    let status = |key: &str, args: serde_json::Value, restricted: bool| {
        let input_fingerprint = blut::config::partition::partition_input_fingerprint(&args, &args);
        db.derive_partition_status(
            latest.get(key),
            "research/dev",
            "codec_eval",
            &args,
            &input_fingerprint,
            restricted,
        )
        .unwrap()
    };
    let matrix = BTreeMap::from([
        (
            "corpus=fresh".into(),
            status("corpus=fresh", serde_json::json!({"corpus":"fresh"}), false),
        ),
        (
            "corpus=stale".into(),
            status("corpus=stale", serde_json::json!({"corpus":"stale"}), false),
        ),
        (
            "corpus=missing".into(),
            status(
                "corpus=missing",
                serde_json::json!({"corpus":"missing"}),
                false,
            ),
        ),
        (
            "corpus=phi".into(),
            status(
                "corpus=phi",
                serde_json::json!({"classification":"restricted"}),
                true,
            ),
        ),
    ]);
    assert_eq!(matrix["corpus=fresh"], CellStatus::Materialized);
    assert_eq!(matrix["corpus=stale"], CellStatus::Stale);
    assert_eq!(matrix["corpus=missing"], CellStatus::Missing);
    assert_eq!(matrix["corpus=phi"], CellStatus::Restricted);
    let moved_source = blut::config::partition::partition_input_fingerprint(
        &serde_json::json!({"dataset":"dataset://corpus@v2"}),
        &serde_json::json!({"corpus":"fresh"}),
    );
    assert_eq!(
        db.derive_partition_status(
            latest.get("corpus=fresh"),
            "research/dev",
            "codec_eval",
            &serde_json::json!({"corpus":"fresh"}),
            &moved_source,
            false,
        )
        .unwrap(),
        CellStatus::Stale,
        "a changed immutable source handle is stale even if compiled args resolve identically"
    );

    let cells = PartitionSet {
        name: "by_corpus".into(),
        recipe: "codec_eval".into(),
        dims: vec![PartitionDim {
            axis: "corpus".into(),
            values: vec![
                "fresh".into(),
                "stale".into(),
                "missing".into(),
                "phi".into(),
            ],
        }],
    }
    .cells();
    let keys = |selector| {
        select_backfill_targets(&cells, &matrix, selector)
            .into_iter()
            .map(|cell| cell.key)
            .collect::<Vec<_>>()
    };
    assert_eq!(keys(BackfillSelector::Missing), ["corpus=missing"]);
    assert_eq!(keys(BackfillSelector::Stale), ["corpus=stale"]);
    assert_eq!(
        keys(BackfillSelector::Default),
        ["corpus=stale", "corpus=missing"]
    );
    assert!(
        keys(BackfillSelector::Force).contains(&"corpus=phi".to_string()),
        "force exposes Restricted so admission can refuse it rather than silently skip it"
    );

    let fresh_cache_key = ContentHash::of_bytes(b"cache-corpus=fresh");
    std::fs::remove_file(
        temp.path()
            .join("cache")
            .join("invocations")
            .join(fresh_cache_key.to_hex())
            .join("record.bin"),
    )
    .unwrap();
    assert_eq!(
        status("corpus=fresh", serde_json::json!({"corpus":"fresh"}), false),
        CellStatus::Stale,
        "a terminal sidecar alone cannot prove materialization after cache eviction"
    );
}
