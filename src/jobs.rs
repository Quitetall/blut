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
    pub fn from_str(s: &str) -> Option<Self> {
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
    serde_json::from_slice(&body)
        .map_err(|e| TrainError::other(format!("parse spec.json: {e}")))
}

pub fn append_status(job_id: &str, update: &StatusUpdate) -> Result<()> {
    use std::io::Write;
    let path = paths::job_dir(job_id)?.join("status.jsonl");
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
    let path = paths::job_dir(job_id)?.join("status.jsonl");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let body = std::fs::read_to_string(&path).map_err(|e| TrainError::Io {
        path: path.clone(),
        source: e,
    })?;
    Ok(body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
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
    JobState::from_str(&body).ok_or_else(|| {
        TrainError::other(format!("unknown state '{}' at {}", body.trim(), path.display()))
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
    let state = read_state(id).unwrap_or(JobState::Running);
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
        crate::python_backend::graceful_kill_pid(pid, grace).await;
    }
    let _ = write_state(job_id, JobState::Cancelled);
    let _ = write_pid_clear(job_id);
    Ok(())
}

fn write_pid_clear(job_id: &str) -> Result<()> {
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

/// Test-friendly: ensure a job_id round-trips through the file
/// system. crate-internal — never call from production code.
#[cfg(test)]
pub(crate) fn debug_init_job(job_id: &str, spec: &TrainSpec) -> Result<PathBuf> {
    let dir = paths::job_dir(job_id)?;
    write_spec(job_id, spec)?;
    write_state(job_id, JobState::Running)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
            method: crate::spec::Method::QLora { rank: 16, alpha: 32 },
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
            for id in [
                "20260510-100000-000000001",
                "20260510-110000-000000002",
            ] {
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
        let (y, m, d, h, mi, se) = unix_to_ymdhms(1_778_761_896);
        assert_eq!((y, m), (2026, 5));
        assert_eq!((h, mi, se), (12, 31, 36));
    }
}
