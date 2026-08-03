// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! RunLedger — BLUT's append-only record of what actually ran (ADR 0154).
//!
//! ARCHITECTURE. This file is the **source of truth**, which is the opposite of
//! [`crate::lineage_db`]'s contract and the reason it is a separate artifact
//! rather than another table. `lineage_db` is explicit that it is a rebuildable
//! INDEX recoverable by re-scanning the content-addressed filesystem; a ledger
//! recording a *judgement* (this run was promoted, by this person, for this
//! reason) is not recoverable from bytes on disk, so it cannot live only in a
//! rebuildable index. The ledger is therefore plain append-only JSONL, and
//! `lineage_db` may index it like anything else.
//!
//! WHY BLUT OWNS THIS. ADR 0034 delegated the experiment record away from BLUT
//! to `outputs/experiment_log.jsonl`. ADR 0154 amends that single row: a run is
//! a BLUT noun — BLUT already assigns `job_id` and records
//! `runs`/`artifacts`/`lineage_edges`/`git_sha` — whereas a run *dashboard* is a
//! verb and stays wandb's. The delegate target had also stopped working: 1894 of
//! its 2021 rows were pytest runs and it was last written 2026-05-28.
//!
//! THREE TIERS, ONE STREAM. Records are never moved between tiers or files.
//! Promotion APPENDS a [`Record::Promoted`]; it does not rewrite history. A
//! `Scratch` record may be cleared by `gc`, which appends a [`Record::Cleared`]
//! tombstone so a citation resolves to "this existed and was collected" rather
//! than dangling. `Recorded` and `Canonical` are never cleared.
//!
//! CONCURRENCY. The Python logger this replaces guarded appends with a
//! `threading.Lock`, which is per-PROCESS — cross-process safety rested on
//! unstated `O_APPEND` atomicity, which POSIX only guarantees for writes up to
//! `PIPE_BUF` (4096 bytes). A run record with a long argv exceeds that. So every
//! append here takes an advisory `flock(LOCK_EX)` for the duration of the write.
//! On non-unix the lock is a no-op and the guarantee degrades to O_APPEND —
//! stated rather than assumed.
//!
//! READS ARE LENIENT, AND SAY SO. A malformed line is skipped, but the count is
//! returned in [`LedgerRead::malformed`] rather than swallowed. Silently
//! skipping is how a truncated write becomes an invisible hole in the record.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};

/// Schema tag written on every record so a future reader can branch on it.
pub const SCHEMA: &str = "blut.run-ledger/v1";

/// Default objective bar: a run that held the machine this long is recorded
/// regardless of outcome or declared intent.
///
/// Four hours is "occupied the box for half a working session". The bar is on
/// DURATION rather than on any quality metric deliberately — expense is
/// measurable where significance is a judgement, and judgement is what
/// promotion to `Canonical` is for. A recipe should declare its own bar (20
/// minutes is long for an SNN probe and short for a fullband codec run); when
/// none is declared the run is still recorded and
/// [`Classification::recommendation`] carries a nudge. A missing bar must never
/// silently drop a run.
pub const DEFAULT_BAR_SECS: u64 = 4 * 60 * 60;

/// Where a run sits in the record. See the module docs for the retention rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Smoke tests, pytest, probes, and anything under the bar. Clearable.
    Scratch,
    /// Crossed the objective bar, or was declared a campaign. Never cleared.
    Recorded,
    /// Explicitly promoted, with a reason and an author. Never cleared.
    Canonical,
}

impl Tier {
    /// Whether `gc` is permitted to collect this tier. Only `Scratch`.
    pub fn is_clearable(self) -> bool {
        matches!(self, Tier::Scratch)
    }
}

/// Why the run was launched. Only the launcher knows this — it cannot be
/// recovered downstream, which is why it is recorded at the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    /// A self-test. Never reaches `Recorded` on intent alone.
    Smoke,
    /// Exploratory. Recorded only if it crosses the bar.
    Probe,
    /// Declared real work. Recorded from step 0.
    Campaign,
}

/// How a run ended. An outcome is a FIELD, not a filter: a 300-epoch divergence
/// is expensive knowledge that stops someone re-running it, so it is recorded on
/// the same bar as a success.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Completed,
    Diverged,
    Killed,
    Oom,
    Failed,
}

/// The join key. Every field is something one of the eleven existing provenance
/// surfaces already knows; the contribution is recording them in one place so a
/// `TRUTH_LEDGER` §2 citation can be resolved to a real run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunIdentity {
    /// BLUT's `job_id` (`YYYYMMDD-HHMMSS-<nanos>`), already sortable.
    pub blut_job_id: String,
    /// The trainer's own run id — `MetricLog` / `read_metric --run` /
    /// the checkpoint `provenance` dict all key on this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trainer_run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoint_sha256: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pccp_change_id: Option<String>,
    /// ADR 0144 evidence directory, referenced rather than duplicated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation_ref: Option<String>,
    /// `TRUTH_LEDGER` §2 row ids this run is cited by (e.g. `2.6`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ledger_rows: Vec<String>,
}

/// One line of the ledger. `kind` is the serde tag, matching `status.jsonl`'s
/// existing `#[serde(tag = "kind")]` convention so the two streams read alike.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    RunStarted {
        schema: String,
        run_uid: String,
        recipe: String,
        intent: Intent,
        started_unix: i64,
        /// ADR 0096 owning tenant, `project[/domain]` — e.g. `lamquant`,
        /// `lamquant/codec`, `tritium`. THE LEDGER IS GLOBAL: BLUT records runs
        /// for every project that uses it, and this is the field a project
        /// filters on to find its own. BLUT stays domain-neutral by treating the
        /// label as opaque — it never interprets "lamquant".
        ///
        /// Empty means the flat `default` namespace, matching `lineage_db`.
        #[serde(default)]
        tenant: String,
        /// The HYPOTHESIS this run tests, e.g. `E1`, `L1`, `PCCP-CHG-2026-007`.
        ///
        /// Without it the ledger records THAT something ran, not WHAT it was
        /// testing, and "has E1 been run?" stays an archaeology question. That
        /// is not hypothetical: `lineage_db` has 167 runs, three distinct recipe
        /// names, and its `experiment` column populated zero times, so the only
        /// way to answer it was to read prose in a stage card — which said
        /// "never-run" while the card's own caveat field recorded an earlier
        /// invalid attempt at R≈0.17.
        ///
        /// Free text and opaque to BLUT, like `tenant`: the engine never
        /// interprets `E1`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        experiment: Option<String>,
        /// The REPLICATE axis. Two runs sharing (experiment, config_fingerprint)
        /// and differing only here are repeats, and repeats are the only source
        /// of a run-to-run variance estimate. Without that estimate no
        /// difference between two runs can be separated from training
        /// stochasticity — see [`compare`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<i64>,
        identity: RunIdentity,
    },
    RunEnded {
        schema: String,
        run_uid: String,
        ended_unix: i64,
        duration_secs: u64,
        outcome: Outcome,
        tier: Tier,
        identity: RunIdentity,
        /// Headline metrics at end, e.g. `best_val_r`. Kept small on purpose:
        /// curves stay in wandb (ADR 0034, unchanged by 0152).
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        metrics: BTreeMap<String, f64>,
    },
    Promoted {
        schema: String,
        run_uid: String,
        promoted_unix: i64,
        /// Required. A promotion without a stated reason is not a judgement.
        reason: String,
        by: String,
    },
    Cleared {
        schema: String,
        run_uid: String,
        cleared_unix: i64,
        /// Kept so a citation to a collected run still resolves.
        recipe: String,
    },
}

/// Whether a run may appear in an EXPORT — a rendered doc, a published report,
/// anything leaving the machine.
///
/// Delegates to [`crate::tenant::Tenant::is_restricted`] rather than re-deriving
/// the rule, because that check is deliberately case-insensitive "so spelling
/// cannot bypass custody policy" and a second implementation is a second place
/// for that to rot. A `clinical`/`restricted` tenant is fail-closed excluded
/// from any export (ADR 0061/0099), and a generated doc tree is an export.
///
/// FAIL-CLOSED on an unparseable tenant: a label we cannot classify is treated
/// as restricted. The alternative — defaulting an unrecognised namespace to
/// exportable — is how PHI leaves by typo.
pub fn exportable(tenant: &str) -> bool {
    if tenant.is_empty() {
        return true; // the flat `default` namespace
    }
    match crate::tenant::Tenant::parse(tenant) {
        Some(t) => !t.is_restricted(),
        None => false,
    }
}

/// The project segment of a `project[/domain]` tenant, for "is this mine?".
/// Empty tenant reads as the `default` project.
pub fn tenant_project(tenant: &str) -> &str {
    if tenant.is_empty() {
        return crate::tenant::DEFAULT_PROJECT;
    }
    tenant.split('/').next().unwrap_or(tenant)
}

impl Record {
    /// The `run_uid` this record concerns, for grouping a stream by run.
    pub fn run_uid(&self) -> &str {
        match self {
            Record::RunStarted { run_uid, .. }
            | Record::RunEnded { run_uid, .. }
            | Record::Promoted { run_uid, .. }
            | Record::Cleared { run_uid, .. } => run_uid,
        }
    }
}

/// The verdict of classifying a finished run, plus any advice for the author.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classification {
    pub tier: Tier,
    /// Set when the recipe declared no bar of its own. Advisory — the run is
    /// recorded either way (ADR 0154: absence must not block recording).
    pub recommendation: Option<String>,
}

/// Decide the tier of a finished run.
///
/// Two independent paths reach `Recorded`, per the owner decision: a declared
/// `Campaign`, or crossing the duration bar. `Smoke` never reaches `Recorded` on
/// intent, but a smoke test that somehow ran four hours still does on duration —
/// if it held the machine that long, something is worth knowing either way.
///
/// `Canonical` is deliberately unreachable here. It is only ever entered by an
/// explicit [`Record::Promoted`], because significance is a judgement and this
/// function only measures.
pub fn classify(
    intent: Intent,
    duration_secs: u64,
    recipe_bar_secs: Option<u64>,
) -> Classification {
    let bar = recipe_bar_secs.unwrap_or(DEFAULT_BAR_SECS);
    let over_bar = duration_secs >= bar;
    let tier = if over_bar || matches!(intent, Intent::Campaign) {
        Tier::Recorded
    } else {
        Tier::Scratch
    };
    let recommendation = recipe_bar_secs.is_none().then(|| {
        format!(
            "recipe declared no duration bar; the global default of {}h was used. \
             Declare one — 4h is long for a probe and short for a fullband run.",
            DEFAULT_BAR_SECS / 3600
        )
    });
    Classification {
        tier,
        recommendation,
    }
}

/// Result of reading the whole ledger.
#[derive(Debug, Clone, Default)]
pub struct LedgerRead {
    pub records: Vec<Record>,
    /// Lines that failed to parse. Surfaced, never swallowed — a truncated
    /// write is a hole in the record and the reader must be able to say so.
    pub malformed: usize,
}

/// Append-only ledger over a JSONL file.
#[derive(Debug, Clone)]
pub struct RunLedger {
    path: PathBuf,
}

impl RunLedger {
    /// Open (without creating) a ledger at an explicit path.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The canonical location: `paths::data_dir()/run-ledger.jsonl`, which
    /// honours `$LAMU_TRAIN_DATA_DIR` like every other BLUT data path.
    pub fn default_path() -> Result<PathBuf> {
        Ok(crate::paths::data_dir()?.join("run-ledger.jsonl"))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one record. Creates the file and parent directory if absent.
    ///
    /// Holds an exclusive advisory lock across the write so concurrent
    /// appenders cannot interleave a partial line. See the module docs for why
    /// `O_APPEND` alone is not enough.
    pub fn append(&self, record: &Record) -> Result<()> {
        let mut line = serde_json::to_string(record)
            .map_err(|e| TrainError::other(format!("serialize ledger record: {e}")))?;
        line.push('\n');

        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                TrainError::other(format!("create ledger dir {}: {e}", parent.display()))
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| TrainError::other(format!("open ledger {}: {e}", self.path.display())))?;

        let mut file = lock_exclusive(file)?;
        file.write_all(line.as_bytes())
            .map_err(|e| TrainError::other(format!("append to ledger: {e}")))?;
        file.flush()
            .map_err(|e| TrainError::other(format!("flush ledger: {e}")))?;
        Ok(())
    }

    /// Read every record. A missing file is an empty ledger, not an error —
    /// the normal state before the first run.
    pub fn read(&self) -> Result<LedgerRead> {
        let file = match std::fs::File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(LedgerRead::default()),
            Err(e) => {
                return Err(TrainError::other(format!(
                    "open ledger {}: {e}",
                    self.path.display()
                )));
            }
        };
        let mut out = LedgerRead::default();
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| TrainError::other(format!("read ledger line: {e}")))?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Record>(&line) {
                Ok(rec) => out.records.push(rec),
                Err(_) => out.malformed += 1,
            }
        }
        Ok(out)
    }

    /// Effective tier per run: the highest ever reached, since promotion
    /// appends rather than rewrites. A `Cleared` tombstone does not lower the
    /// tier — it records that the detail was collected, and `gc` may only ever
    /// clear `Scratch` in the first place.
    pub fn tiers(&self) -> Result<BTreeMap<String, Tier>> {
        let read = self.read()?;
        let mut out: BTreeMap<String, Tier> = BTreeMap::new();
        for rec in &read.records {
            let uid = rec.run_uid().to_string();
            let seen = match rec {
                Record::RunStarted { .. } => Tier::Scratch,
                Record::RunEnded { tier, .. } => *tier,
                Record::Promoted { .. } => Tier::Canonical,
                Record::Cleared { .. } => continue,
            };
            let slot = out.entry(uid).or_insert(seen);
            if rank(seen) > rank(*slot) {
                *slot = seen;
            }
        }
        Ok(out)
    }
}

fn rank(t: Tier) -> u8 {
    match t {
        Tier::Scratch => 0,
        Tier::Recorded => 1,
        Tier::Canonical => 2,
    }
}

/// Take an exclusive advisory lock, returning a handle that writes through to
/// the file and unlocks on drop.
///
/// `nix::fcntl::Flock` OWNS the `File` and already is the RAII guard — it
/// `Deref`s to `File` so writes go through it, and unlocks in `Drop`. An
/// earlier draft of this module hand-rolled a borrow-based guard around the
/// free `nix::fcntl::flock`, which was wrong three ways: that function has been
/// deprecated since nix 0.28 (and this crate lints `-D warnings`), it takes a
/// `RawFd` rather than the `BorrowedFd` that was passed, and the separate guard
/// duplicated what `Flock` already provides.
///
/// Note `Flock::drop` PANICS if the unlock syscall fails and the thread is not
/// already panicking. That is upstream behaviour and it is the right trade here
/// — a ledger append that cannot release its lock has corrupted the invariant
/// every other writer depends on, and failing loudly beats leaving the file
/// wedged for every later process.
#[cfg(unix)]
fn lock_exclusive(file: std::fs::File) -> Result<nix::fcntl::Flock<std::fs::File>> {
    use nix::fcntl::{Flock, FlockArg};
    Flock::lock(file, FlockArg::LockExclusive)
        .map_err(|(_file, errno)| TrainError::other(format!("lock ledger: {errno}")))
}

/// Off unix there is no advisory lock and the guarantee degrades to `O_APPEND`
/// atomicity, which POSIX bounds at `PIPE_BUF`. Stated rather than assumed —
/// the Python logger this replaces made exactly this assumption silently.
#[cfg(not(unix))]
fn lock_exclusive(file: std::fs::File) -> Result<std::fs::File> {
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_ledger(name: &str) -> RunLedger {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "blut-run-ledger-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        RunLedger::at(p)
    }

    fn started(uid: &str) -> Record {
        Record::RunStarted {
            schema: SCHEMA.into(),
            run_uid: uid.into(),
            recipe: "lamquant_joint_codec".into(),
            intent: Intent::Campaign,
            started_unix: 1_785_000_000,
            experiment: None,
            seed: None,
            tenant: "lamquant".into(),
            identity: RunIdentity {
                blut_job_id: uid.into(),
                ..Default::default()
            },
        }
    }

    fn ended(uid: &str, tier: Tier, outcome: Outcome) -> Record {
        Record::RunEnded {
            schema: SCHEMA.into(),
            run_uid: uid.into(),
            ended_unix: 1_785_020_000,
            duration_secs: 20_000,
            outcome,
            tier,
            identity: RunIdentity {
                blut_job_id: uid.into(),
                ..Default::default()
            },
            metrics: BTreeMap::from([("best_val_r".to_string(), 0.6838)]),
        }
    }

    #[test]
    fn missing_ledger_reads_as_empty_not_error() {
        let l = tmp_ledger("missing");
        let r = l
            .read()
            .expect("missing file is the normal pre-first-run state");
        assert!(r.records.is_empty());
        assert_eq!(r.malformed, 0);
    }

    /// Frozen bytes emitted by `tools/import_experiment_log.py` (ADR 0154 §C).
    ///
    /// This is a CROSS-LANGUAGE contract: the migration writes
    /// `blut.run-ledger/v1` from Python and this reader must accept it. Pinning
    /// real emitted bytes rather than a hand-written approximation is the point
    /// — a hand-written fixture drifts silently from what the tool actually
    /// produces, and the failure then shows up as unreadable history.
    ///
    /// Note both records omit `checkpoint_sha256`: the legacy rows carry
    /// checkpoint PATHS, not digests, and omitting the field says "unknown"
    /// where an empty list would claim "we checked and found none".
    #[test]
    fn reads_the_bytes_the_python_migration_actually_emits() {
        const STARTED: &str = r#"{"kind":"run_started","schema":"blut.run-ledger/v1","run_uid":"legacy:e2e_ship_1776390243","recipe":"fast","intent":"probe","started_unix":0,"identity":{"blut_job_id":"","trainer_run_id":"e2e_ship_1776390243"}}"#;
        const ENDED: &str = r#"{"kind":"run_ended","schema":"blut.run-ledger/v1","run_uid":"legacy:e2e_ship_1776390243","ended_unix":0,"duration_secs":0,"outcome":"completed","tier":"recorded","identity":{"blut_job_id":"","trainer_run_id":"e2e_ship_1776390243"},"metrics":{"best_val_r":0.32290266334120965,"best_val_prd":116.21831108624536,"final_val_r":0.0,"final_val_prd":0.0,"best_epoch":0.0}}"#;

        let started: Record = serde_json::from_str(STARTED).expect("run_started must parse");
        let ended: Record = serde_json::from_str(ENDED).expect("run_ended must parse");

        match &started {
            Record::RunStarted {
                intent, identity, ..
            } => {
                assert_eq!(*intent, Intent::Probe);
                assert_eq!(
                    identity.trainer_run_id.as_deref(),
                    Some("e2e_ship_1776390243")
                );
                assert!(identity.checkpoint_sha256.is_empty());
            }
            other => panic!("expected RunStarted, got {other:?}"),
        }
        match &ended {
            Record::RunEnded {
                tier,
                outcome,
                metrics,
                ..
            } => {
                assert_eq!(*tier, Tier::Recorded);
                assert_eq!(*outcome, Outcome::Completed);
                assert_eq!(metrics.get("best_val_r"), Some(&0.322_902_663_341_209_65));
            }
            other => panic!("expected RunEnded, got {other:?}"),
        }
        // And the pair groups under one run, which is what `tiers()` relies on.
        assert_eq!(started.run_uid(), ended.run_uid());
    }

    #[test]
    fn append_then_read_round_trips() {
        let l = tmp_ledger("roundtrip");
        l.append(&started("a")).unwrap();
        l.append(&ended("a", Tier::Recorded, Outcome::Completed))
            .unwrap();
        let r = l.read().unwrap();
        assert_eq!(r.records.len(), 2);
        assert_eq!(r.malformed, 0);
        assert_eq!(r.records[0], started("a"));
        let _ = std::fs::remove_file(l.path());
    }

    #[test]
    fn malformed_lines_are_counted_not_swallowed() {
        // A truncated write must be visible. Silently skipping is how a hole in
        // the record becomes invisible.
        let l = tmp_ledger("malformed");
        l.append(&started("a")).unwrap();
        {
            let mut f = OpenOptions::new().append(true).open(l.path()).unwrap();
            f.write_all(b"{not json\n").unwrap();
        }
        let r = l.read().unwrap();
        assert_eq!(r.records.len(), 1);
        assert_eq!(r.malformed, 1);
        let _ = std::fs::remove_file(l.path());
    }

    #[test]
    fn campaign_is_recorded_even_when_short() {
        let c = classify(Intent::Campaign, 5, None);
        assert_eq!(c.tier, Tier::Recorded);
    }

    #[test]
    fn smoke_under_the_bar_stays_scratch() {
        let c = classify(Intent::Smoke, 60, None);
        assert_eq!(c.tier, Tier::Scratch);
    }

    #[test]
    fn anything_over_the_bar_is_recorded_regardless_of_intent() {
        // Including a smoke test: if it held the box for four hours, that is
        // worth knowing whatever it was called.
        for intent in [Intent::Smoke, Intent::Probe, Intent::Campaign] {
            let c = classify(intent, DEFAULT_BAR_SECS, None);
            assert_eq!(c.tier, Tier::Recorded, "{intent:?} over the bar");
        }
    }

    #[test]
    fn per_recipe_bar_overrides_the_global_default() {
        // 20 min bar: a 30 min probe is real work for this recipe.
        let c = classify(Intent::Probe, 1_800, Some(1_200));
        assert_eq!(c.tier, Tier::Recorded);
        assert!(c.recommendation.is_none(), "a declared bar needs no nudge");
    }

    #[test]
    fn missing_recipe_bar_records_anyway_and_recommends() {
        // ADR 0154: absence of a declared bar must not block recording.
        let c = classify(Intent::Campaign, 10, None);
        assert_eq!(c.tier, Tier::Recorded);
        let rec = c.recommendation.expect("should nudge the author");
        assert!(rec.contains("declared no duration bar"), "{rec}");
    }

    #[test]
    fn classify_never_returns_canonical() {
        // Canonical is a judgement, reachable only via an explicit Promoted
        // record. A measurement function must not be able to mint one.
        for intent in [Intent::Smoke, Intent::Probe, Intent::Campaign] {
            for secs in [0, 1, DEFAULT_BAR_SECS, u64::MAX] {
                assert_ne!(classify(intent, secs, None).tier, Tier::Canonical);
            }
        }
    }

    #[test]
    fn promotion_appends_and_raises_the_effective_tier() {
        let l = tmp_ledger("promote");
        l.append(&started("a")).unwrap();
        l.append(&ended("a", Tier::Recorded, Outcome::Completed))
            .unwrap();
        l.append(&Record::Promoted {
            schema: SCHEMA.into(),
            run_uid: "a".into(),
            promoted_unix: 1_785_030_000,
            reason: "ledger row 2.6".into(),
            by: "quitetall".into(),
        })
        .unwrap();
        let tiers = l.tiers().unwrap();
        assert_eq!(tiers.get("a"), Some(&Tier::Canonical));
        // History is intact: promotion appended, it did not rewrite.
        assert_eq!(l.read().unwrap().records.len(), 3);
        let _ = std::fs::remove_file(l.path());
    }

    #[test]
    fn a_restricted_tenant_is_never_exportable() {
        // The doc tree is an export (ADR 0061/0099). Case-insensitive, because
        // the upstream check is — spelling must not bypass custody policy.
        for t in [
            "clinical",
            "Clinical",
            "CLINICAL",
            "restricted",
            "ReStRiCtEd",
        ] {
            assert!(!exportable(t), "{t} must not be exportable");
        }
        for t in ["clinical/eeg", "restricted/phi"] {
            assert!(!exportable(t), "{t} must not be exportable");
        }
    }

    #[test]
    fn an_unparseable_tenant_fails_closed() {
        // A label we cannot classify is treated as restricted. Defaulting an
        // unrecognised namespace to exportable is how PHI leaves by typo.
        // `..` traversal, too many segments, and non-`[A-Za-z0-9_.-]` bytes are
        // all refused by Tenant::parse.
        for t in ["../escape", "a/b/c/d", "\u{0}", ".hidden", "trailing."] {
            assert!(!exportable(t), "{t:?} must fail closed");
        }
    }

    #[test]
    fn whitespace_only_tenant_is_the_default_namespace_not_a_failure() {
        // Tenant::parse TRIMS before the empty check, so "   " is the flat
        // `default` namespace and exports. Asserted explicitly because it is
        // the one input that looks like it should fail closed and does not —
        // worth pinning so a future trim change is caught here rather than by
        // a doc quietly gaining or losing rows.
        assert!(exportable("   "));
        assert_eq!(tenant_project("   "), "   ".split('/').next().unwrap());
    }

    #[test]
    fn ordinary_projects_export_and_keep_their_project_segment() {
        assert!(exportable("lamquant"));
        assert!(exportable("lamquant/codec"));
        assert!(exportable("tritium"));
        assert!(exportable("")); // flat default namespace
        assert_eq!(tenant_project("lamquant/codec"), "lamquant");
        assert_eq!(tenant_project("tritium"), "tritium");
        assert_eq!(tenant_project(""), crate::tenant::DEFAULT_PROJECT);
    }

    #[test]
    fn the_ledger_is_global_so_projects_are_distinguishable() {
        // The whole point of carrying a tenant: one ledger holds every
        // project's runs, and a consumer filters to its own.
        let l = tmp_ledger("multiproject");
        for (uid, tenant) in [("a", "lamquant"), ("b", "tritium"), ("c", "lamquant/codec")] {
            l.append(&Record::RunStarted {
                schema: SCHEMA.into(),
                run_uid: uid.into(),
                recipe: "r".into(),
                intent: Intent::Campaign,
                started_unix: 0,
                experiment: None,
                seed: None,
                tenant: tenant.into(),
                identity: RunIdentity::default(),
            })
            .unwrap();
        }
        let mine: Vec<&str> = l
            .read()
            .unwrap()
            .records
            .iter()
            .filter_map(|r| match r {
                Record::RunStarted {
                    run_uid, tenant, ..
                } if tenant_project(tenant) == "lamquant" => Some(run_uid.as_str()),
                _ => None,
            })
            .map(|s| Box::leak(s.to_string().into_boxed_str()) as &str)
            .collect();
        assert_eq!(mine, vec!["a", "c"], "tritium's run must not be mine");
        let _ = std::fs::remove_file(l.path());
    }

    #[test]
    fn only_scratch_is_clearable() {
        assert!(Tier::Scratch.is_clearable());
        assert!(!Tier::Recorded.is_clearable());
        assert!(!Tier::Canonical.is_clearable());
    }

    #[test]
    fn a_cleared_scratch_run_still_resolves() {
        // The point of the tombstone: a citation to a collected smoke run must
        // not dangle.
        let l = tmp_ledger("cleared");
        l.append(&started("s")).unwrap();
        l.append(&Record::Cleared {
            schema: SCHEMA.into(),
            run_uid: "s".into(),
            cleared_unix: 1_785_040_000,
            recipe: "lamquant_joint_codec".into(),
        })
        .unwrap();
        let r = l.read().unwrap();
        assert!(
            r.records
                .iter()
                .any(|x| matches!(x, Record::Cleared { .. }))
        );
        assert!(
            l.tiers().unwrap().contains_key("s"),
            "citation must resolve"
        );
        let _ = std::fs::remove_file(l.path());
    }

    #[test]
    fn outcome_is_a_field_not_a_filter() {
        // A long diverged run is Recorded, same as a completed one.
        let l = tmp_ledger("diverged");
        l.append(&ended("d", Tier::Recorded, Outcome::Diverged))
            .unwrap();
        let tiers = l.tiers().unwrap();
        assert_eq!(tiers.get("d"), Some(&Tier::Recorded));
        let _ = std::fs::remove_file(l.path());
    }
}

// ─────────────────────────────────────────────────────────────────
// Separating a result from run-to-run noise.
//
// The ledger's whole point is to stop a number being believed because it is
// large. A codec run reporting val_r 0.850 against a previous 0.813 looks like a
// clear win, and it may be — but training is stochastic, and without repeats
// there is NOTHING in the record that distinguishes a real 0.037 from the spread
// you would get by rerunning the SAME configuration with a different seed.
//
// So the unit of comparison is an ARM: runs sharing (experiment,
// config_fingerprint). Within an arm, runs differ only by `seed`, and their
// spread IS the noise floor. Between arms, the question is whether the observed
// difference is large relative to that floor.
//
// The test is an exact permutation test rather than a t-test, chosen for a
// property that matters more here than power: IT CANNOT LIE AT SMALL n. With one
// run per arm there are exactly two labellings, so the smallest reachable
// two-sided p is 1.0 and the answer is `Indeterminate` by construction — not by
// a threshold someone picked. No distributional assumption is made, which is
// right for n in the single digits where normality is untestable anyway.
// ─────────────────────────────────────────────────────────────────

/// One side of a comparison: every run of a single configuration.
#[derive(Debug, Clone)]
pub struct Arm {
    pub label: String,
    /// One metric value per run. Length is the replicate count.
    pub values: Vec<f64>,
}

impl Arm {
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return f64::NAN;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Sample standard deviation (n-1). `None` below two runs, because one run
    /// has no spread to report and returning 0.0 would read as "perfectly
    /// reproducible".
    pub fn sd(&self) -> Option<f64> {
        if self.values.len() < 2 {
            return None;
        }
        let mean = self.mean();
        let var = self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (self.values.len() - 1) as f64;
        Some(var.sqrt())
    }
}

/// What the record can and cannot support.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The design cannot reach the threshold no matter what the numbers are.
    /// Carries the replicate count that could, so the answer is actionable
    /// rather than a shrug.
    Indeterminate {
        reason: String,
        seeds_per_arm_needed: usize,
    },
    /// An exact p-value was computed.
    Decided { p_value: f64, significant: bool },
}

/// Result of comparing two arms on one metric.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub metric: String,
    pub baseline: String,
    pub candidate: String,
    pub baseline_n: usize,
    pub candidate_n: usize,
    pub delta: f64,
    /// Pooled within-arm SD — the noise floor. `None` when neither arm repeats.
    pub within_sd: Option<f64>,
    /// `delta` in units of that floor. `None` when the floor is unknown.
    pub effect_size: Option<f64>,
    /// Smallest two-sided p this DESIGN could produce, whatever the data. The
    /// honest headline for an underpowered comparison.
    pub min_achievable_p: f64,
    pub verdict: Verdict,
}

fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut acc = 1.0f64;
    for i in 0..k {
        acc = acc * (n - i) as f64 / (i + 1) as f64;
    }
    acc
}

/// Enumeration ceiling. C(24,12) is 2.7M; beyond that an exact test stops being
/// cheap, and a comparison with two dozen runs per arm is not the regime this
/// guard exists for.
const MAX_PERMUTATIONS: f64 = 3_000_000.0;

/// Compare two arms on one metric. `alpha` is the two-sided threshold.
///
/// Returns `Indeterminate` whenever the arms are too small for ANY arrangement
/// of the data to clear `alpha`. That is the case this exists for: it is the
/// difference between "we measured no effect" and "we could not have measured
/// one", and conflating those is how an underpowered result gets written into a
/// roadmap as fact.
pub fn compare(metric: &str, baseline: &Arm, candidate: &Arm, alpha: f64) -> Comparison {
    let (n, m) = (baseline.values.len(), candidate.values.len());
    let delta = candidate.mean() - baseline.mean();

    // Pool the within-arm variances that exist. An arm of one contributes no
    // spread and is simply absent from the pool.
    let mut ss = 0.0f64;
    let mut df = 0usize;
    for arm in [baseline, candidate] {
        if arm.values.len() >= 2 {
            let mean = arm.mean();
            ss += arm.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
            df += arm.values.len() - 1;
        }
    }
    let within_sd = if df > 0 {
        Some((ss / df as f64).sqrt())
    } else {
        None
    };
    let effect_size = within_sd.and_then(|sd| (sd > 0.0).then(|| delta / sd));

    let total = binomial(n + m, n);
    let min_achievable_p = if total > 0.0 {
        (2.0 / total).min(1.0)
    } else {
        1.0
    };

    // Smallest arm size k (per arm) with 2 / C(2k, k) <= alpha.
    let seeds_needed = (1..=32)
        .find(|k| 2.0 / binomial(2 * k, *k) <= alpha)
        .unwrap_or(32);

    if n == 0 || m == 0 {
        return Comparison {
            metric: metric.into(),
            baseline: baseline.label.clone(),
            candidate: candidate.label.clone(),
            baseline_n: n,
            candidate_n: m,
            delta,
            within_sd,
            effect_size,
            min_achievable_p: 1.0,
            verdict: Verdict::Indeterminate {
                reason: "an arm has no runs".into(),
                seeds_per_arm_needed: seeds_needed,
            },
        };
    }

    if min_achievable_p > alpha {
        return Comparison {
            metric: metric.into(),
            baseline: baseline.label.clone(),
            candidate: candidate.label.clone(),
            baseline_n: n,
            candidate_n: m,
            delta,
            within_sd,
            effect_size,
            min_achievable_p,
            verdict: Verdict::Indeterminate {
                reason: format!(
                    "{n} vs {m} runs: the smallest two-sided p this design can \
                     produce is {min_achievable_p:.3}, above alpha {alpha:.3}. \
                     No arrangement of these numbers could be significant."
                ),
                seeds_per_arm_needed: seeds_needed,
            },
        };
    }

    if total > MAX_PERMUTATIONS {
        return Comparison {
            metric: metric.into(),
            baseline: baseline.label.clone(),
            candidate: candidate.label.clone(),
            baseline_n: n,
            candidate_n: m,
            delta,
            within_sd,
            effect_size,
            min_achievable_p,
            verdict: Verdict::Indeterminate {
                reason: format!("{total:.0} permutations exceeds the exact-enumeration ceiling"),
                seeds_per_arm_needed: seeds_needed,
            },
        };
    }

    // Exact two-sided permutation test over every way to split the pooled
    // values into arms of the original sizes.
    let pooled: Vec<f64> = baseline
        .values
        .iter()
        .chain(candidate.values.iter())
        .copied()
        .collect();
    let observed = delta.abs();
    let mut at_least_as_extreme = 0u64;
    let mut seen = 0u64;
    let mut index = vec![0usize; n];
    for (slot, value) in index.iter_mut().enumerate() {
        *value = slot;
    }
    loop {
        let base_sum: f64 = index.iter().map(|&i| pooled[i]).sum();
        let total_sum: f64 = pooled.iter().sum();
        let cand_mean = (total_sum - base_sum) / m as f64;
        let base_mean = base_sum / n as f64;
        if (cand_mean - base_mean).abs() >= observed - 1e-12 {
            at_least_as_extreme += 1;
        }
        seen += 1;

        // Next combination in lexicographic order.
        let mut i = n;
        loop {
            if i == 0 {
                break;
            }
            i -= 1;
            if index[i] != i + pooled.len() - n {
                index[i] += 1;
                for j in i + 1..n {
                    index[j] = index[j - 1] + 1;
                }
                break;
            }
            if i == 0 {
                break;
            }
        }
        if index[0] > pooled.len() - n {
            break;
        }
        if seen >= total as u64 {
            break;
        }
    }

    let p_value = at_least_as_extreme as f64 / seen.max(1) as f64;
    Comparison {
        metric: metric.into(),
        baseline: baseline.label.clone(),
        candidate: candidate.label.clone(),
        baseline_n: n,
        candidate_n: m,
        delta,
        within_sd,
        effect_size,
        min_achievable_p,
        verdict: Verdict::Decided {
            p_value,
            significant: p_value <= alpha,
        },
    }
}

#[cfg(test)]
mod significance_tests {
    use super::*;

    fn arm(label: &str, values: &[f64]) -> Arm {
        Arm {
            label: label.into(),
            values: values.to_vec(),
        }
    }

    #[test]
    fn one_run_per_arm_is_indeterminate_however_large_the_gap() {
        // THE CASE FROM 2026-08-03. CHG-006 reported held-out R 0.813 and
        // CHG-007 reported 0.842 on a byte-identical validation set with one
        // variable changed. That is a well-built ablation and the delta may well
        // be real — but it is one run against one run, so nothing in the record
        // separates it from seed-to-seed spread.
        //
        // The gap is deliberately made absurd here to show the verdict does not
        // depend on effect size at all.
        let c = compare(
            "val_r",
            &arm("chg-006", &[0.813]),
            &arm("chg-007", &[0.999]),
            0.05,
        );
        assert!(
            matches!(c.verdict, Verdict::Indeterminate { .. }),
            "n=1 vs n=1 must never be reported as significant, got {:?}",
            c.verdict
        );
        assert_eq!(c.min_achievable_p, 1.0);
        assert!(c.within_sd.is_none(), "no replicates means no noise floor");
    }

    #[test]
    fn indeterminate_says_how_many_seeds_would_settle_it() {
        // A verdict the reader cannot act on is only marginally better than
        // silence. Four per arm is what an exact two-sided test needs at
        // alpha=0.05: C(8,4)=70, so 2/70 = 0.029 <= 0.05, while three per arm
        // gives C(6,3)=20 and a floor of 0.100.
        let c = compare("val_r", &arm("a", &[0.80]), &arm("b", &[0.85]), 0.05);
        match c.verdict {
            Verdict::Indeterminate {
                seeds_per_arm_needed,
                ..
            } => {
                assert_eq!(seeds_per_arm_needed, 4);
            }
            other => panic!("expected Indeterminate, got {other:?}"),
        }
    }

    #[test]
    fn three_per_arm_still_cannot_reach_alpha_05() {
        // The boundary worth pinning: 3v3 feels like "we replicated it" and is
        // still underpowered for a two-sided exact test.
        let c = compare(
            "val_r",
            &arm("a", &[0.80, 0.81, 0.79]),
            &arm("b", &[0.90, 0.91, 0.89]),
            0.05,
        );
        assert!(matches!(c.verdict, Verdict::Indeterminate { .. }));
        assert!((c.min_achievable_p - 0.1).abs() < 1e-9, "2/C(6,3) = 0.1");
        // The noise floor IS estimable here even though the test cannot fire.
        assert!(c.within_sd.is_some());
    }

    #[test]
    fn a_clean_separation_at_four_per_arm_is_significant() {
        let c = compare(
            "val_r",
            &arm("a", &[0.80, 0.81, 0.79, 0.80]),
            &arm("b", &[0.90, 0.91, 0.89, 0.90]),
            0.05,
        );
        match c.verdict {
            Verdict::Decided {
                p_value,
                significant,
            } => {
                assert!(significant, "clean separation should decide, p={p_value}");
                assert!(p_value <= 0.05);
            }
            other => panic!("expected Decided, got {other:?}"),
        }
        assert!(
            c.effect_size.unwrap() > 5.0,
            "delta should dwarf the noise floor"
        );
    }

    #[test]
    fn overlapping_arms_are_decided_but_not_significant() {
        // The other half of honesty: with enough runs, a small delta must be
        // reported as NOT significant rather than quietly dropped.
        let c = compare(
            "val_r",
            &arm("a", &[0.80, 0.84, 0.79, 0.83]),
            &arm("b", &[0.82, 0.81, 0.85, 0.80]),
            0.05,
        );
        match c.verdict {
            Verdict::Decided { significant, .. } => assert!(!significant),
            other => panic!("expected Decided, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_arm_is_indeterminate_not_a_crash() {
        let c = compare("val_r", &arm("a", &[]), &arm("b", &[0.9]), 0.05);
        assert!(matches!(c.verdict, Verdict::Indeterminate { .. }));
    }

    #[test]
    fn sd_is_none_for_a_single_run_rather_than_zero() {
        // Returning 0.0 would read as "perfectly reproducible", which is the
        // opposite of what one run tells you.
        assert!(arm("a", &[0.5]).sd().is_none());
        assert!(arm("a", &[0.5, 0.7]).sd().is_some());
    }
}

#[cfg(test)]
mod cross_language_tests {
    use super::*;

    /// THE INTERFACE THAT FAILS SILENTLY.
    ///
    /// `blut_core/run_ledger.py` hand-writes this wire format so a Python
    /// trainer can record a run without shelling out to the CLI. A drifted
    /// field name or enum spelling there would not raise: reads are lenient by
    /// design, so the records would be skipped as malformed and the ledger
    /// would look empty while every trainer believed it was logging. That is
    /// precisely the failure this whole ADR exists to end, so it gets a test
    /// that crosses the language boundary rather than two schemas maintained
    /// by eye.
    ///
    /// The fixture is written BY THE SHIM, never typed out here — a
    /// hand-written fixture would only prove this file agrees with itself.
    /// `tools/tests/test_run_ledger_shim.py` regenerates it.
    #[test]
    fn rust_reads_what_the_python_shim_writes() {
        let fixture = std::path::Path::new("/var/tmp/lamquant-gates/xlang/run-ledger.jsonl");
        if !fixture.is_file() {
            eprintln!("skipping: fixture absent (regenerate via the python shim test)");
            return;
        }
        let ledger = RunLedger::at(fixture.to_path_buf());
        let read = ledger.read().expect("a python-written ledger must parse");
        assert_eq!(
            read.malformed, 0,
            "no line written by the shim may be unparseable"
        );
        assert_eq!(read.records.len(), 4, "two runs, start + end each");

        let mut starts = 0;
        for record in &read.records {
            if let Record::RunStarted {
                experiment,
                seed,
                tenant,
                intent,
                ..
            } = record
            {
                assert_eq!(experiment.as_deref(), Some("E1"), "experiment must survive");
                assert!(
                    seed.is_some(),
                    "seed is the replicate axis; it must survive"
                );
                assert_eq!(tenant, "lamquant");
                assert_eq!(*intent, Intent::Campaign);
                starts += 1;
            }
        }
        assert_eq!(starts, 2);
    }
}

#[cfg(test)]
mod dogfood_tests {
    use super::*;

    /// The end-to-end chain on the case that prompted all of this.
    ///
    /// On 2026-08-03 a Package 16 run reported held-out R 0.842 against a prior
    /// 0.813, on a byte-identical validation set with exactly one input changed
    /// — a genuinely well-built ablation. The delta may well be real. The point
    /// is that the RECORD cannot say so, because each arm has one run, and this
    /// test pins that the tooling refuses to pretend otherwise.
    ///
    /// The fixture is written by `blut_core/run_ledger.py`, so a pass here
    /// exercises Python write -> Rust read -> verdict, not just the last step.
    #[test]
    fn the_package16_comparison_is_indeterminate_not_a_win() {
        let fixture = std::path::Path::new("/var/tmp/lamquant-gates/dogfood/run-ledger.jsonl");
        if !fixture.is_file() {
            eprintln!("skipping: dogfood fixture absent");
            return;
        }
        let read = RunLedger::at(fixture.to_path_buf()).read().expect("parse");
        assert_eq!(read.malformed, 0);

        let arm_for = |fingerprint: &str| -> Arm {
            let uids: Vec<&str> = read
                .records
                .iter()
                .filter_map(|r| match r {
                    Record::RunStarted {
                        run_uid, identity, ..
                    } if identity.config_fingerprint.as_deref() == Some(fingerprint) => {
                        Some(run_uid.as_str())
                    }
                    _ => None,
                })
                .collect();
            let values = read
                .records
                .iter()
                .filter_map(|r| match r {
                    Record::RunEnded {
                        run_uid, metrics, ..
                    } if uids.contains(&run_uid.as_str()) => metrics.get("best_val_r").copied(),
                    _ => None,
                })
                .collect();
            Arm {
                label: fingerprint.into(),
                values,
            }
        };

        let result = compare(
            "best_val_r",
            &arm_for("cohort-train9"),
            &arm_for("cohort-train56"),
            0.05,
        );
        assert_eq!(result.baseline_n, 1);
        assert_eq!(result.candidate_n, 1);
        assert!(
            (result.delta - 0.029).abs() < 1e-6,
            "delta {}",
            result.delta
        );
        assert!(
            result.within_sd.is_none(),
            "one run per arm has no noise floor"
        );
        match &result.verdict {
            Verdict::Indeterminate {
                seeds_per_arm_needed,
                ..
            } => {
                assert_eq!(*seeds_per_arm_needed, 4);
            }
            other => panic!("a 1-vs-1 comparison must not be Decided, got {other:?}"),
        }
    }
}
