//! Durable resume policy (BLUT-API Phase D) — the orchestrator HALF.
//!
//! The crash-gated DECISION: given a stable resume directory's run-state marker
//! (`state.json`), should a starting run RESUME from the recovery checkpoint
//! there, start FRESH, or REFUSE because a live run already owns the directory?
//!
//! This is deliberately the *policy* only. The trainer half — writing the
//! marker + the recovery checkpoints, and restoring optimizer/RNG on `--resume`
//! — lives in the Python trainer. The whole Rust↔Python contract is: the resume
//! directory ([`resume_dir`]), this [`ResumeState`] schema, and the
//! `--resume` / `--resume-dir` flags. The engine never reads or writes a
//! recovery checkpoint itself — it only reads the small JSON marker.
//!
//! Domain-agnostic: any cookbook train stage that writes a `state.json` of this
//! shape can call [`decide_resume`] for the crash-gated answer.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Heartbeat cadence the TRAINER must honor (wall-clock seconds): it rewrites
/// `state.json.heartbeat_unix` at least this often while alive. Time-based (NOT
/// tied to the validation/epoch cadence) so a long epoch can't masquerade as a
/// crash. Must stay ≤ [`DEFAULT_STALE_AFTER_SECS`] / 3.
pub const HEARTBEAT_INTERVAL_SECS: u64 = 60;

/// A foreign run whose heartbeat is older than this is considered CRASHED (safe
/// to take over). 3× the heartbeat cadence so a momentary stall (GC pause, a
/// slow checkpoint write) is never mistaken for a crash.
pub const DEFAULT_STALE_AFTER_SECS: u64 = 3 * HEARTBEAT_INTERVAL_SECS;

const STATUS_FINISHED: &str = "finished";

/// The run-state marker the trainer writes into the resume dir. The orchestrator
/// reads it to decide. Forward-compatible: unknown fields are ignored, and a
/// missing / unparseable marker is treated as "no marker" (⇒ [`ResumeDecision::Fresh`]).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResumeState {
    /// `"running"` while a run owns the dir; `"finished"` after a clean exit.
    /// Any other / unknown value is treated as not-finished (a run was here).
    pub status: String,
    /// The job `run_id` (job_dir basename) of the writing run. Lets the
    /// orchestrator tell an IN-PROCESS retry (same run_id — a prior attempt of
    /// THIS run died) from a CROSS-invocation resume (a different run_id).
    pub run_id: String,
    /// PID of the writing trainer. Advisory / diagnostics only — liveness is
    /// judged by the heartbeat, which works across the launcher boundary where a
    /// PID does not.
    #[serde(default)]
    pub pid: u32,
    /// Last heartbeat, Unix seconds. The trainer rewrites it every
    /// [`HEARTBEAT_INTERVAL_SECS`]. On a FOREIGN run_id: a stale heartbeat ⇒ that
    /// run crashed (resume); a fresh one ⇒ it is still alive (refuse).
    pub heartbeat_unix: u64,
}

impl ResumeState {
    /// Read + parse `<resume_dir>/state.json`. `None` on absent / unreadable /
    /// unparseable — the caller maps `None` ⇒ [`ResumeDecision::Fresh`], so a
    /// missing marker can never block a fresh run. A file that EXISTS but does
    /// not parse (a truncated / corrupt marker) is logged at `warn` so an
    /// operator can tell "never ran" from "marker corrupt" — but it still
    /// degrades to `None`/Fresh (a corrupt marker must never block a run).
    pub fn read(resume_dir: &Path) -> Option<ResumeState> {
        let path = resume_dir.join("state.json");
        let body = std::fs::read_to_string(&path).ok()?; // absent / unreadable ⇒ silent None (the common first-run case)
        match serde_json::from_str(&body) {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(
                    "resume: state.json at {} is unparseable ({e}); treating as no marker (fresh)",
                    path.display()
                );
                None
            }
        }
    }
}

/// The crash-gated decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumeDecision {
    /// Start from scratch: never ran, the prior run finished cleanly, or no
    /// usable marker.
    Fresh,
    /// Resume from the recovery checkpoint in the resume dir — either an
    /// in-process retry of THIS run (same run_id) or a crashed prior run
    /// (foreign run_id, stale heartbeat).
    Resume,
    /// A DIFFERENT run is alive and owns the dir (foreign run_id, fresh
    /// heartbeat). Resuming would race two trainers on one checkpoint dir — the
    /// caller MUST hard-error rather than silently start fresh.
    RefuseConcurrent,
}

/// Decide resume vs fresh vs refuse from the marker. Pure + total — the whole
/// policy is this one function.
///
/// * `state` — the parsed marker (`None` ⇒ absent/unparseable ⇒ `Fresh`).
/// * `this_run_id` — the CURRENT run's id (job_dir basename).
/// * `now_unix` / `stale_after_secs` — a FOREIGN run whose heartbeat is at least
///   `stale_after_secs` old is treated as crashed (`Resume`); newer ⇒ alive
///   (`RefuseConcurrent`). Use [`DEFAULT_STALE_AFTER_SECS`].
///
/// The run_id check comes BEFORE the heartbeat check, so an in-process OOM retry
/// (same run_id, whose just-killed attempt left a possibly-fresh heartbeat)
/// always `Resume`s — never falsely `RefuseConcurrent`.
///
/// Clock assumption: wall clocks are roughly monotonic within the stale window.
/// A heartbeat stamped in the FUTURE (NTP jump) makes `age` saturate to 0 ⇒
/// `RefuseConcurrent` — the SAFE direction (refuse rather than resume onto a
/// possibly-live run); the operator can force progress with the `no_resume`
/// override if a clock correction wedges a genuinely-crashed run.
pub fn decide_resume(
    state: Option<&ResumeState>,
    this_run_id: &str,
    now_unix: u64,
    stale_after_secs: u64,
) -> ResumeDecision {
    let Some(s) = state else {
        return ResumeDecision::Fresh; // never ran / no usable marker
    };
    if s.status == STATUS_FINISHED {
        return ResumeDecision::Fresh; // prior training completed — don't resume a done run
    }
    // Not finished ⇒ a run was in progress here.
    if s.run_id == this_run_id {
        // Same run: a prior ATTEMPT of THIS run died (in-process retry).
        return ResumeDecision::Resume;
    }
    // A DIFFERENT run owns the dir — crashed (stale) or alive (fresh).
    let age = now_unix.saturating_sub(s.heartbeat_unix);
    if age >= stale_after_secs {
        ResumeDecision::Resume
    } else {
        ResumeDecision::RefuseConcurrent
    }
}

/// Filesystem-safe slug: anything outside `[A-Za-z0-9_-]` collapses to `_`. Used
/// on BOTH path components below so neither the recipe name nor the key can
/// introduce a separator / `..` traversal — `resume_dir` is `pub`, so it must
/// not trust its inputs to already be path-clean.
fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The stable resume directory for a `(recipe, resume_key)`: OUTSIDE the per-run
/// job_dir, under the data root, so a cross-invocation re-run with the SAME
/// resume_key resolves the SAME dir and finds the checkpoint. `resume_key_hex`
/// is the stage's cache-key hex (the engine's canonical same-training
/// fingerprint — always `[0-9a-f]{64}` from `ContentHash`, but slugged anyway
/// since this fn is `pub`). Both components are filesystem-slugged.
pub fn resume_dir(data_root: &Path, recipe: &str, resume_key_hex: &str) -> PathBuf {
    data_root.join("Training").join("resume").join(format!(
        "{}-{}",
        slug(recipe),
        slug(resume_key_hex)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(status: &str, run_id: &str, hb: u64) -> ResumeState {
        ResumeState {
            status: status.into(),
            run_id: run_id.into(),
            pid: 1234,
            heartbeat_unix: hb,
        }
    }

    // ── the 5-row decision table ──────────────────────────────────────

    #[test]
    fn absent_marker_is_fresh() {
        assert_eq!(
            decide_resume(None, "run-A", 1000, DEFAULT_STALE_AFTER_SECS),
            ResumeDecision::Fresh
        );
    }

    #[test]
    fn finished_is_fresh_even_same_run() {
        // A cleanly-finished run is never resumed (it's done), regardless of
        // run_id or heartbeat.
        let s = state("finished", "run-A", 1000);
        assert_eq!(
            decide_resume(Some(&s), "run-A", 1000, DEFAULT_STALE_AFTER_SECS),
            ResumeDecision::Fresh
        );
    }

    #[test]
    fn same_run_running_resumes_regardless_of_heartbeat() {
        // In-process retry: the just-killed prior attempt may have left a FRESH
        // heartbeat — must still Resume (never RefuseConcurrent) because it's the
        // same run.
        let s = state("running", "run-A", 999); // heartbeat 1s old (fresh)
        assert_eq!(
            decide_resume(Some(&s), "run-A", 1000, DEFAULT_STALE_AFTER_SECS),
            ResumeDecision::Resume
        );
    }

    #[test]
    fn foreign_run_stale_heartbeat_resumes() {
        // A different run that hasn't heartbeat in > stale window crashed.
        let s = state("running", "run-OLD", 1000);
        let now = 1000 + DEFAULT_STALE_AFTER_SECS; // exactly at the threshold ⇒ stale
        assert_eq!(
            decide_resume(Some(&s), "run-NEW", now, DEFAULT_STALE_AFTER_SECS),
            ResumeDecision::Resume
        );
    }

    #[test]
    fn foreign_run_fresh_heartbeat_refuses() {
        // A different run heartbeat 10s ago is alive — refuse (concurrent).
        let s = state("running", "run-LIVE", 1000);
        assert_eq!(
            decide_resume(Some(&s), "run-NEW", 1010, DEFAULT_STALE_AFTER_SECS),
            ResumeDecision::RefuseConcurrent
        );
    }

    #[test]
    fn unknown_status_treated_as_running() {
        // A marker with a non-"finished" status means a run was here; apply the
        // run_id / heartbeat logic (here: foreign + fresh ⇒ refuse).
        let s = state("crashed_weirdly", "run-LIVE", 1000);
        assert_eq!(
            decide_resume(Some(&s), "run-NEW", 1005, DEFAULT_STALE_AFTER_SECS),
            ResumeDecision::RefuseConcurrent
        );
    }

    #[test]
    fn stale_threshold_consistency() {
        // The heartbeat cadence must be tight enough that the stale window is a
        // clean multiple — a crash is detected within ~3 missed heartbeats.
        assert!(HEARTBEAT_INTERVAL_SECS * 3 <= DEFAULT_STALE_AFTER_SECS);
    }

    // ── resume_dir derivation ─────────────────────────────────────────

    #[test]
    fn resume_dir_is_keyed_and_sanitized() {
        let d = resume_dir(Path::new("/data"), "lamquant_joint_codec", "deadbeef00");
        assert_eq!(
            d,
            PathBuf::from("/data/Training/resume/lamquant_joint_codec-deadbeef00")
        );
        // A recipe with path-unsafe chars is slugged (no traversal / separators).
        let bad = resume_dir(Path::new("/data"), "a/b ../c", "k");
        assert_eq!(bad, PathBuf::from("/data/Training/resume/a_b____c-k"));
    }

    // ── ResumeState::read ─────────────────────────────────────────────

    #[test]
    fn parses_python_written_marker_shape() {
        // CROSS-LANGUAGE CONTRACT: the exact JSON `durable_resume.py` writes
        // (json.dumps of {status, run_id, pid, heartbeat_unix}) must deserialize
        // into ResumeState. If the trainer ever renames a field, this breaks
        // here rather than silently always-Fresh in production.
        let py = r#"{"status": "running", "run_id": "run-7", "pid": 4242, "heartbeat_unix": 1700000000}"#;
        let s: ResumeState = serde_json::from_str(py).expect("python marker must parse");
        assert_eq!(s.status, "running");
        assert_eq!(s.run_id, "run-7");
        assert_eq!(s.pid, 4242);
        assert_eq!(s.heartbeat_unix, 1_700_000_000);
        // And it drives the decision as expected (foreign + fresh ⇒ refuse).
        assert_eq!(
            decide_resume(Some(&s), "run-NEW", 1_700_000_010, DEFAULT_STALE_AFTER_SECS),
            ResumeDecision::RefuseConcurrent
        );
    }

    #[test]
    fn read_roundtrips_and_degrades_to_none() {
        let td = tempfile::tempdir().unwrap();
        // Absent ⇒ None.
        assert!(ResumeState::read(td.path()).is_none());
        // Corrupt ⇒ None (never panics, never blocks a fresh run).
        std::fs::write(td.path().join("state.json"), b"{ not json").unwrap();
        assert!(ResumeState::read(td.path()).is_none());
        // Valid ⇒ Some, round-trips.
        let s = state("running", "run-A", 42);
        std::fs::write(
            td.path().join("state.json"),
            serde_json::to_string(&s).unwrap(),
        )
        .unwrap();
        assert_eq!(ResumeState::read(td.path()), Some(s));
    }
}
