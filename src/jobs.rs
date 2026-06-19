// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Per-job persistence on disk.
//!
//! One subdir per job under `paths::jobs_dir()`. Layout:
//!
//! ```text
//! <jobs_dir>/<id>/
//!     spec.json      — TrainSpec serialized at job start
//!     status.jsonl   — append-only StatusUpdate stream (one per line)
//!     pid            — child trainer.py pid; empty if foreground
//!     log.txt        — captured stderr from trainer (forwarded by
//!                      tracing in the foreground path)
//!     state          — one of: running | done | failed | cancelled
//! ```
//!
//! Job ids are timestamped + randomly suffixed so two jobs started
//! the same second don't collide. Stable sort order matches start
//! time when listing.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};
use crate::paths;
use crate::protocol::StatusUpdate;
use crate::spec::TrainSpec;

/// Compact, sortable job id: `YYYYMMDD-HHMMSS-NNNNNNNNN`.
///
/// The trailing nanoseconds field is monotonic-within-the-second
/// so lexicographic sort matches chronological order even when two
/// ids land in the same wall-clock second. Nanoseconds is 9 chars
/// of zero-padded decimal — wider than wall-clock precision but
/// keeps the format fixed-width.
pub fn new_job_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    let (y, m, d, h, mi, se) = unix_to_ymdhms(secs);
    format!("{y:04}{m:02}{d:02}-{h:02}{mi:02}{se:02}-{nanos:09}")
}

/// Lifecycle state for one job. Persisted as a single-word file so
/// `lamu-train jobs` doesn't need to parse JSON to filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Running,
    Done,
    Failed,
    Cancelled,
}

impl JobState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
    pub fn parse_label(s: &str) -> Option<Self> {
        match s.trim() {
            "running" => Some(Self::Running),
            "done" => Some(Self::Done),
            "failed" => Some(Self::Failed),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }
}

/// Lightweight summary of one job — what `lamu-train jobs` prints.
/// Built from on-disk state alone; no live process queries.
#[derive(Clone, Debug, Serialize)]
pub struct JobSummary {
    pub id: String,
    pub state: JobState,
    pub pid: Option<u32>,
    pub output_name: Option<String>,
    pub last_loss: Option<f32>,
    pub last_step: Option<u32>,
    pub final_loss: Option<f32>,
}

pub fn write_spec(job_id: &str, spec: &TrainSpec) -> Result<()> {
    let path = paths::job_dir(job_id)?.join("spec.json");
    let body = serde_json::to_vec_pretty(spec)
        .map_err(|e| TrainError::other(format!("serialize spec: {e}")))?;
    std::fs::write(&path, body).map_err(|e| TrainError::Io { path, source: e })
}

pub fn read_spec(job_id: &str) -> Result<TrainSpec> {
    let path = paths::job_dir(job_id)?.join("spec.json");
    let body = std::fs::read(&path).map_err(|e| TrainError::Io {
        path: path.clone(),
        source: e,
    })?;
    serde_json::from_slice(&body).map_err(|e| TrainError::other(format!("parse spec.json: {e}")))
}

/// `status.jsonl` rotation threshold in bytes. `LAMU_STATUS_MAX_MB`
/// overrides the 64 MiB default (0 disables rotation). A long training
/// run's per-step spam can otherwise grow the log unbounded.
pub fn status_max_bytes() -> u64 {
    std::env::var("LAMU_STATUS_MAX_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(64)
        .saturating_mul(1024 * 1024)
}

/// Roll `status.jsonl` over to `status.jsonl.1` (single generation,
/// prior `.1` overwritten) when it reaches the cap. Best-effort: a
/// rename failure is logged, never propagated — losing rotation must
/// not fail a job. Returns `true` if a rotation happened. Readers
/// (`read_status`, the TUI tail, the lineage scanner) read `.1` then
/// the current file so no history is lost across one rollover.
pub fn rotate_status_if_needed(path: &std::path::Path) -> bool {
    rotate_status_with_cap(path, status_max_bytes())
}

/// Cap-parameterized core of [`rotate_status_if_needed`] (env-free, so it
/// is unit-testable without touching the process-global `LAMU_STATUS_MAX_MB`).
fn rotate_status_with_cap(path: &std::path::Path, cap: u64) -> bool {
    if cap == 0 {
        return false;
    }
    let len = match std::fs::metadata(path) {
        Ok(m) => m.len(),
        Err(_) => return false, // not yet created
    };
    if len < cap {
        return false;
    }
    // `with_extension` replaces the last component: `status.jsonl` ->
    // `status.jsonl.1` (callers always pass the `status.jsonl` path).
    let rolled = path.with_extension("jsonl.1");
    if let Err(e) = std::fs::rename(path, &rolled) {
        tracing::warn!("status rotation rename failed for {}: {e}", path.display());
        return false;
    }
    true
}

pub fn append_status(job_id: &str, update: &StatusUpdate) -> Result<()> {
    use std::io::Write;
    let path = paths::job_dir(job_id)?.join("status.jsonl");
    rotate_status_if_needed(&path);
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| TrainError::Io {
            path: path.clone(),
            source: e,
        })?;
    let line = serde_json::to_string(update)
        .map_err(|e| TrainError::other(format!("serialize status: {e}")))?;
    writeln!(f, "{line}").map_err(|e| TrainError::Io {
        path: path.clone(),
        source: e,
    })
}

pub fn read_status(job_id: &str) -> Result<Vec<StatusUpdate>> {
    Ok(read_status_lines(job_id)?
        .iter()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}

/// Raw `status.jsonl` lines (rotation-aware: `.1` generation then the
/// current file). The base for both the legacy `StatusUpdate` reader
/// (above) and the framework `StageEvent` lineage reader — status.jsonl
/// can hold EITHER format depending on the launch path, so consumers
/// parse tolerantly.
pub fn read_status_lines(job_id: &str) -> Result<Vec<String>> {
    let path = paths::job_dir(job_id)?.join("status.jsonl");
    let mut out = Vec::new();
    for p in [path.with_extension("jsonl.1"), path] {
        if !p.exists() {
            continue;
        }
        let body = std::fs::read_to_string(&p).map_err(|e| TrainError::Io {
            path: p.clone(),
            source: e,
        })?;
        out.extend(
            body.lines()
                .filter(|l| !l.trim().is_empty())
                .map(String::from),
        );
    }
    Ok(out)
}

pub fn write_pid(job_id: &str, pid: u32) -> Result<()> {
    let path = paths::job_dir(job_id)?.join("pid");
    std::fs::write(&path, pid.to_string()).map_err(|e| TrainError::Io { path, source: e })
}

pub fn read_pid(job_id: &str) -> Result<Option<u32>> {
    let path = paths::job_dir(job_id)?.join("pid");
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(s.trim().parse().ok()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(TrainError::Io { path, source: e }),
    }
}

pub fn write_state(job_id: &str, state: JobState) -> Result<()> {
    let path = paths::job_dir(job_id)?.join("state");
    std::fs::write(&path, state.as_str()).map_err(|e| TrainError::Io { path, source: e })
}

pub fn read_state(job_id: &str) -> Result<JobState> {
    let path = paths::job_dir(job_id)?.join("state");
    let body = std::fs::read_to_string(&path).map_err(|e| TrainError::Io {
        path: path.clone(),
        source: e,
    })?;
    JobState::parse_label(&body).ok_or_else(|| {
        TrainError::other(format!(
            "unknown state '{}' at {}",
            body.trim(),
            path.display()
        ))
    })
}

pub fn list_jobs() -> Result<Vec<JobSummary>> {
    let dir = paths::jobs_dir()?;
    let mut out = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(TrainError::Io {
                path: dir,
                source: e,
            });
        }
    };
    for entry in entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let id = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        out.push(summarize(&id));
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

fn summarize(id: &str) -> JobSummary {
    let on_disk_state = read_state(id).unwrap_or(JobState::Running);
    let pid = read_pid(id).unwrap_or(None);
    let spec = read_spec(id).ok();
    let updates = read_status(id).unwrap_or_default();
    let mut last_loss = None;
    let mut last_step = None;
    let mut final_loss = None;
    for u in &updates {
        match u {
            StatusUpdate::Step { step, loss, .. } => {
                last_loss = Some(*loss);
                last_step = Some(*step);
            }
            StatusUpdate::Done { final_loss: fl, .. } => final_loss = Some(*fl),
            _ => {}
        }
    }
    // MONITOR-2: the on-disk `state` file only flips to a terminal value
    // when blut runs the cancel / completion path. A job whose process
    // crashed or was SIGKILL'd out-of-band (OOM killer, host reboot,
    // manual `kill -9`) leaves `state=running` on disk forever, so the
    // cockpit shows it Running indefinitely. Reconcile liveness against
    // the recorded pid: if a Running job has a real recorded pid that is
    // *definitively* gone, report it as Failed instead.
    let state = reconcile_liveness(on_disk_state, pid);
    JobSummary {
        id: id.to_string(),
        state,
        pid,
        output_name: spec.map(|s| s.output_name),
        last_loss,
        last_step,
        final_loss,
    }
}

/// MONITOR-2 reconciliation. Given the on-disk job state and the
/// recorded pid (the child process-group leader written to the `pid`
/// file at spawn), decide the *reported* state.
///
/// Conservative by design — only reclassifies `Running` → `Failed` when
/// **all** of the following hold, so a just-spawned job is never falsely
/// reported dead:
///   * the on-disk state is `Running` (terminal states are authoritative
///     and never second-guessed),
///   * a pid was actually recorded (`Some`) — a job that hasn't published
///     its child pid yet is left Running, and
///   * that pid is *definitively gone* (`kill(pid,0)` → `ESRCH`).
///     An `Unsignalable` (EPERM) pid still exists (owned by another user
///     / reparented to init); we treat "exists but can't signal" as
///     alive and leave it Running rather than risk a false Failed.
fn reconcile_liveness(on_disk: JobState, pid: Option<u32>) -> JobState {
    if on_disk != JobState::Running {
        return on_disk;
    }
    match pid {
        // A recorded pid that is *definitively* gone → the process tree
        // crashed; report Failed. `pid_definitely_gone` returns false for
        // both "alive" and "exists-but-unsignalable", so we never falsely
        // mark a live (or merely unsignalable) job Failed.
        Some(pid) if pid_definitely_gone(pid) => JobState::Failed,
        // Live pid, unsignalable pid, or no recorded pid yet (foreground
        // job / child not spawned): nothing definitive — leave it Running.
        _ => JobState::Running,
    }
}

/// True iff the recorded pid is *definitively* gone (`kill(pid,0)` →
/// `ESRCH`). Returns false when the process exists, when it exists but
/// is unsignalable (EPERM), or when liveness can't be determined (non-
/// Unix) — i.e. errs on the side of "still alive" so a healthy job is
/// never reported Failed.
#[cfg(unix)]
fn pid_definitely_gone(pid: u32) -> bool {
    use crate::python_kill::PidStatus;
    matches!(crate::python_kill::pid_alive(pid), PidStatus::Gone)
}

#[cfg(not(unix))]
fn pid_definitely_gone(_pid: u32) -> bool {
    // No portable cheap liveness probe off-Unix; never false-positive a
    // crash.
    false
}

/// Render a job dir layout summary as plain text — used by the
/// `jobs` and `log` subcommands. Public so tests + external tooling
/// can re-use the formatting.
pub fn render_log(updates: &[StatusUpdate]) -> String {
    let mut out = String::new();
    for u in updates {
        match u {
            StatusUpdate::Step {
                step,
                total,
                loss,
                lr,
                vram_mb,
            } => {
                out.push_str(&format!(
                    "step {step}/{total}  loss={loss:.4}  lr={lr:.2e}  vram={vram_mb}MB\n"
                ));
            }
            StatusUpdate::Eval { step, eval_loss } => {
                out.push_str(&format!("eval @{step}  loss={eval_loss:.4}\n"));
            }
            StatusUpdate::Saved { path } => {
                out.push_str(&format!("saved {}\n", path.display()));
            }
            StatusUpdate::Done {
                final_loss,
                checkpoint_dir,
            } => {
                out.push_str(&format!(
                    "done  final_loss={final_loss:.4}  ckpt={}\n",
                    checkpoint_dir.display()
                ));
            }
            StatusUpdate::Failed { error } => {
                out.push_str(&format!("FAILED: {error}\n"));
            }
            StatusUpdate::Heartbeat { phase, .. } => {
                if let Some(p) = phase {
                    out.push_str(&format!("… {p}\n"));
                }
            }
        }
    }
    out
}

/// Convert UNIX seconds to (Y,M,D,h,m,s) using the same algorithm
/// as `chrono::DateTime::from_timestamp` but without the dep.
/// Assumes UTC; lossless for any value in the i64 second range.
fn unix_to_ymdhms(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let days = secs / 86400;
    let rem = secs % 86400;
    let h = (rem / 3600) as u32;
    let mi = ((rem % 3600) / 60) as u32;
    let se = (rem % 60) as u32;

    // Civil-from-days algorithm by Howard Hinnant.
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u32, m as u32, d as u32, h, mi, se)
}

/// Public wrapper over `unix_to_ymdhms` for sibling modules (the
/// TUI views formatter reuses the same dependency-free civil-from-days
/// conversion rather than pulling in chrono just to print a date).
pub fn unix_to_ymdhms_pub(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    unix_to_ymdhms(secs)
}

/// Look up a job by exact id OR by unique prefix. Used so the user
/// can type `lamu-train cancel 20260510` instead of pasting the
/// full id. Errors when zero or multiple matches.
pub fn resolve_job_id(query: &str) -> Result<String> {
    if query.trim().is_empty() {
        return Err(TrainError::other(
            "job id is empty. Run `lamu-train jobs` to list.",
        ));
    }
    let jobs = list_jobs()?;
    let exact: Vec<_> = jobs.iter().filter(|j| j.id == query).collect();
    if exact.len() == 1 {
        return Ok(exact[0].id.clone());
    }
    let prefix: Vec<_> = jobs.iter().filter(|j| j.id.starts_with(query)).collect();
    match prefix.len() {
        0 => Err(TrainError::other(format!(
            "no job matches '{query}'. Run `lamu-train jobs` to list."
        ))),
        1 => Ok(prefix[0].id.clone()),
        n => {
            let names: Vec<&str> = prefix.iter().map(|j| j.id.as_str()).collect();
            Err(TrainError::other(format!(
                "'{query}' is ambiguous ({n} matches): {names:?}"
            )))
        }
    }
}

/// `lamu-train cancel` core. Reads pid file, sends SIGTERM, waits
/// up to `grace` for exit, falls back to SIGKILL. Marks the job
/// state Cancelled regardless. Idempotent: if no pid, no-op success.
pub async fn cancel_job(job_id: &str, grace: std::time::Duration) -> Result<()> {
    let pid = read_pid(job_id)?;
    if let Some(pid) = pid {
        crate::python_kill::graceful_kill_pid(pid, grace).await;
    }
    let _ = write_state(job_id, JobState::Cancelled);
    let _ = clear_pid(job_id);
    Ok(())
}

/// Remove a job's `pid` file (called when the job's last live child
/// exits, or on cancel) so a stale/reused pgid can't be signalled later.
pub fn clear_pid(job_id: &str) -> Result<()> {
    let path = paths::job_dir(job_id)?.join("pid");
    let _ = std::fs::remove_file(path);
    Ok(())
}

/// Public path getter for tools that want to reach into a job dir
/// directly (log tail, etc.).
pub fn job_dir_path(job_id: &str) -> Result<PathBuf> {
    paths::job_dir(job_id)
}

/// Convenience: tail the last N lines of the rendered log.
pub fn tail_log(job_id: &str, lines: usize) -> Result<String> {
    let updates = read_status(job_id)?;
    let rendered = render_log(&updates);
    Ok(rendered
        .lines()
        .rev()
        .take(lines)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn rotate_rolls_over_at_cap_and_preserves_history() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("status.jsonl");

        // Under cap: no rotation.
        std::fs::write(&path, b"a\nb\n").unwrap();
        assert!(!rotate_status_with_cap(&path, 1024));
        assert!(!path.with_extension("jsonl.1").exists());

        // At/over cap: rotate to .1, original gone.
        assert!(rotate_status_with_cap(&path, 4));
        assert!(path.with_extension("jsonl.1").exists());
        assert!(!path.exists());
        assert_eq!(
            std::fs::read_to_string(path.with_extension("jsonl.1")).unwrap(),
            "a\nb\n"
        );

        // Fresh writes land in a new current file; both generations readable.
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap();
        writeln!(f, "c").unwrap();
        // read_status order = .1 (older) then current (newer).
        let combined = {
            let mut out = String::new();
            for p in [path.with_extension("jsonl.1"), path.clone()] {
                out.push_str(&std::fs::read_to_string(&p).unwrap());
            }
            out
        };
        assert_eq!(combined, "a\nb\nc\n");
    }

    #[test]
    fn rotate_cap_zero_disables() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("status.jsonl");
        std::fs::write(&path, vec![0u8; 1_000_000]).unwrap();
        assert!(!rotate_status_with_cap(&path, 0));
        assert!(path.exists());
    }

    fn with_jobs_dir<F: FnOnce()>(f: F) {
        let _g = crate::TEST_ENV_LOCK.lock().unwrap();
        let td = tempfile::tempdir().unwrap();
        let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
        unsafe {
            std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path());
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
        if let Err(panic) = r {
            std::panic::resume_unwind(panic);
        }
    }

    fn sample_spec() -> TrainSpec {
        TrainSpec {
            base_model: "Qwen/Qwen3-7B".into(),
            output_name: "test-out".into(),
            output_dir: PathBuf::from("/tmp/lamu-train-test"),
            method: crate::spec::Method::QLora {
                rank: 16,
                alpha: 32,
            },
            dataset: crate::spec::DatasetSource::JsonlPath {
                path: PathBuf::from("/tmp/x.jsonl"),
            },
            optimizer: crate::spec::Optim::AdamW,
            lr: 2e-4,
            epochs: 1,
            batch_size: 1,
            grad_accum: 1,
            seq_len: 512,
            seed: 42,
            quant: "Q4_K_M".into(),
            skip_convert: false,
            dpo_beta: None,
        }
    }

    #[test]
    fn job_id_format_is_sortable() {
        let a = new_job_id();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = new_job_id();
        assert_ne!(a, b);
        // Lexicographic order matches chronological order — both
        // the timestamp prefix and the nanosecond suffix are
        // fixed-width zero-padded decimal.
        assert!(a < b, "a={a} b={b}");
        // Shape: YYYYMMDD-HHMMSS-NNNNNNNNN = 25 chars.
        assert_eq!(a.len(), 25, "id={a}");
    }

    #[test]
    fn write_then_read_spec_roundtrip() {
        with_jobs_dir(|| {
            let id = "test-spec-rt";
            let spec = sample_spec();
            write_spec(id, &spec).unwrap();
            let back = read_spec(id).unwrap();
            assert_eq!(back.base_model, spec.base_model);
            assert_eq!(back.output_name, spec.output_name);
        });
    }

    #[test]
    fn append_then_read_status() {
        with_jobs_dir(|| {
            let id = "test-status";
            for s in 1..=3 {
                append_status(
                    id,
                    &StatusUpdate::Step {
                        step: s,
                        total: 3,
                        loss: 1.0 / s as f32,
                        lr: 0.0001,
                        vram_mb: 1234,
                    },
                )
                .unwrap();
            }
            append_status(
                id,
                &StatusUpdate::Done {
                    final_loss: 0.123,
                    checkpoint_dir: PathBuf::from("/tmp/x"),
                },
            )
            .unwrap();
            let back = read_status(id).unwrap();
            assert_eq!(back.len(), 4);
            assert!(matches!(back[3], StatusUpdate::Done { .. }));
        });
    }

    #[test]
    fn pid_round_trip() {
        with_jobs_dir(|| {
            let id = "test-pid";
            assert!(read_pid(id).unwrap().is_none());
            write_pid(id, 12345).unwrap();
            assert_eq!(read_pid(id).unwrap(), Some(12345));
        });
    }

    #[test]
    fn state_round_trip() {
        with_jobs_dir(|| {
            let id = "test-state";
            write_state(id, JobState::Running).unwrap();
            assert_eq!(read_state(id).unwrap(), JobState::Running);
            write_state(id, JobState::Done).unwrap();
            assert_eq!(read_state(id).unwrap(), JobState::Done);
        });
    }

    #[test]
    fn list_jobs_returns_sorted() {
        with_jobs_dir(|| {
            for id in [
                "20260510-100000-100000000",
                "20260510-110000-200000000",
                "20260510-090000-300000000",
            ] {
                write_state(id, JobState::Done).unwrap();
            }
            let listed = list_jobs().unwrap();
            assert_eq!(listed.len(), 3);
            assert_eq!(listed[0].id, "20260510-090000-300000000");
            assert_eq!(listed[2].id, "20260510-110000-200000000");
        });
    }

    #[test]
    fn list_jobs_handles_missing_dir() {
        let _g = crate::TEST_ENV_LOCK.lock().unwrap();
        let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
        unsafe {
            std::env::set_var(
                "LAMU_TRAIN_JOBS_DIR",
                "/tmp/lamu-jobs-truly-nonexistent-xyz",
            );
        }
        let r = list_jobs().unwrap();
        assert!(r.is_empty());
        unsafe {
            match prev {
                Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
                None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
            }
        }
    }

    #[test]
    fn resolve_job_id_by_prefix() {
        with_jobs_dir(|| {
            for id in ["20260510-100000-000000001", "20260510-110000-000000002"] {
                write_state(id, JobState::Done).unwrap();
            }
            assert_eq!(
                resolve_job_id("20260510-1100").unwrap(),
                "20260510-110000-000000002"
            );
            assert!(resolve_job_id("20260510").is_err()); // ambiguous
            assert!(resolve_job_id("nope").is_err()); // no match
        });
    }

    #[test]
    fn render_log_renders_each_kind() {
        let updates = vec![
            StatusUpdate::Step {
                step: 1,
                total: 10,
                loss: 1.5,
                lr: 0.0002,
                vram_mb: 8000,
            },
            StatusUpdate::Eval {
                step: 5,
                eval_loss: 0.9,
            },
            StatusUpdate::Saved {
                path: PathBuf::from("/tmp/ckpt"),
            },
            StatusUpdate::Done {
                final_loss: 0.5,
                checkpoint_dir: PathBuf::from("/tmp/ckpt"),
            },
        ];
        let r = render_log(&updates);
        assert!(r.contains("step 1/10"));
        assert!(r.contains("eval @5"));
        assert!(r.contains("saved /tmp/ckpt"));
        assert!(r.contains("done"));
    }

    #[test]
    fn unix_to_ymdhms_known_dates() {
        // 1970-01-01 00:00:00 UTC
        assert_eq!(unix_to_ymdhms(0), (1970, 1, 1, 0, 0, 0));
        // 2024-01-01 00:00:00 UTC
        assert_eq!(unix_to_ymdhms(1_704_067_200), (2024, 1, 1, 0, 0, 0));
        // 2026-05-10 12:34:56 UTC = 1_778_412_896
        let (y, m, _d, h, mi, se) = unix_to_ymdhms(1_778_761_896);
        assert_eq!((y, m), (2026, 5));
        assert_eq!((h, mi, se), (12, 31, 36));
    }

    // ── MONITOR-2: JobState liveness reconciliation ─────────────────

    /// The conservative guards on `reconcile_liveness`, exercised
    /// directly (no process needed): terminal states pass through
    /// untouched, and a Running job with no recorded pid stays Running.
    #[test]
    fn reconcile_liveness_is_conservative() {
        // Terminal states are authoritative — never second-guessed even
        // with a definitely-dead pid (pid 1 is init; we pass None to keep
        // it deterministic, and also a clearly-dead-shaped pid below).
        assert_eq!(
            reconcile_liveness(JobState::Done, Some(999_999_999)),
            JobState::Done
        );
        assert_eq!(reconcile_liveness(JobState::Failed, None), JobState::Failed);
        assert_eq!(
            reconcile_liveness(JobState::Cancelled, Some(999_999_999)),
            JobState::Cancelled
        );
        // Running with NO recorded pid → left Running (don't false-fail a
        // just-spawned job that hasn't published its pid yet).
        assert_eq!(
            reconcile_liveness(JobState::Running, None),
            JobState::Running
        );
    }

    /// MONITOR-2 core: a job left in `Running` on disk whose recorded pid
    /// is definitively dead must be REPORTED as `Failed` by the listing
    /// path (`list_jobs`/`summarize`). We get a guaranteed-dead pid by
    /// spawning a trivial process, reaping it, and polling until the
    /// kernel reports it gone.
    #[cfg(unix)]
    #[test]
    fn dead_pid_running_job_reconciled_to_failed() {
        use crate::python_kill::{PidStatus, pid_alive};
        use std::process::Command;
        use std::time::{Duration, Instant};

        // Spawn + reap a short-lived process so its pid is definitively
        // gone (and reaped → no zombie keeping it "alive").
        let mut child = Command::new("true").spawn().expect("spawn `true`");
        let dead_pid = child.id();
        child.wait().expect("reap `true`");
        // Poll until the kernel reports the pid gone (ESRCH). If it never
        // goes (pid reused by an unrelated process within the window),
        // skip rather than assert a flaky result.
        let deadline = Instant::now() + Duration::from_secs(2);
        while pid_alive(dead_pid) != PidStatus::Gone && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if pid_alive(dead_pid) != PidStatus::Gone {
            eprintln!("skipping: pid {dead_pid} did not become Gone (reused?)");
            return;
        }

        with_jobs_dir(|| {
            let id = "20260528-120000-000000001";
            // On-disk state says Running, but the recorded pid is dead.
            write_state(id, JobState::Running).unwrap();
            write_pid(id, dead_pid).unwrap();

            let listed = list_jobs().unwrap();
            let job = listed.iter().find(|j| j.id == id).expect("job listed");
            assert_eq!(
                job.state,
                JobState::Failed,
                "dead-pid Running job must be reported Failed (MONITOR-2)"
            );
            // The on-disk `state` file is left untouched (reconciliation
            // is report-only here) — the pid is still surfaced.
            assert_eq!(read_state(id).unwrap(), JobState::Running);
            assert_eq!(job.pid, Some(dead_pid));
        });
    }

    /// A Running job whose recorded pid is a *live* process must stay
    /// Running — the reconciliation must not false-positive a healthy
    /// job.
    #[cfg(unix)]
    #[test]
    fn live_pid_running_job_stays_running() {
        use std::process::Command;

        // A real, live child that outlives the assertion.
        let mut child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn `sleep 30`");
        let live_pid = child.id();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_jobs_dir(|| {
                let id = "20260528-120000-000000002";
                write_state(id, JobState::Running).unwrap();
                write_pid(id, live_pid).unwrap();

                let listed = list_jobs().unwrap();
                let job = listed.iter().find(|j| j.id == id).expect("job listed");
                assert_eq!(
                    job.state,
                    JobState::Running,
                    "live-pid Running job must stay Running (no false Failed)"
                );
            });
        }));

        // Always clean up the live child, even if the assertion panicked.
        let _ = child.kill();
        let _ = child.wait();
        if let Err(p) = result {
            std::panic::resume_unwind(p);
        }
    }
}
