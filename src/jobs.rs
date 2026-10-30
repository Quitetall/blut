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

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};
use crate::paths;
use crate::protocol::StatusUpdate;
use crate::spec::TrainSpec;
use crate::tenant::Tenant;

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

/// Persist the owning tenant beside the job's other canonical filesystem
/// records. Lineage is a rebuildable index, so it must be able to recover the
/// tenant from disk rather than from an in-memory launch argument.
pub fn write_tenant(job_id: &str, tenant: &Tenant) -> Result<()> {
    let path = paths::job_dir(job_id)?.join("tenant");
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, tenant.to_string()).map_err(|source| TrainError::Io {
        path: tmp.clone(),
        source,
    })?;
    std::fs::rename(&tmp, &path).map_err(|source| TrainError::Io { path, source })
}

/// Read a job's owning tenant. Jobs created before ADR 0096 have no marker and
/// belong to the flat `default` namespace. A present but invalid marker is
/// refused fail-closed; it is never silently coerced to default.
pub fn read_tenant(job_id: &str) -> Result<Tenant> {
    // Read paths must never materialize job state. In particular, the web
    // export sidecar calls this before serving status; using `job_dir()` here
    // used to turn a GET for an unknown id into an empty on-disk job.
    let path = paths::jobs_dir()?.join(job_id).join("tenant");
    let body = match std::fs::read_to_string(&path) {
        Ok(body) => body,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Tenant::default()),
        Err(source) => return Err(TrainError::Io { path, source }),
    };
    let label = body.trim();
    Tenant::parse(label)
        .filter(|tenant| tenant.to_string() == label)
        .ok_or_else(|| {
            TrainError::other(format!(
                "invalid tenant marker '{}' at {}",
                label,
                path.display()
            ))
        })
}

/// What a job's launch knew about itself, recorded once when execution starts.
///
/// The lineage index is rebuilt from job directories, so anything it reports
/// about a run has to be on disk in the job directory. It used to probe the
/// indexing process instead: `git rev-parse HEAD` of whatever directory
/// `blut` happened to run in, and "now" as the end time. A `lineage reindex`
/// from another checkout rewrote every historical run's git SHA, and the
/// start time was never recorded at all, so no SLA rule could ever match.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct LaunchProvenance {
    /// `git rev-parse HEAD` of the launch directory, when it is a repository.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    /// The engine build commit (`BLUT_GIT_HASH`) of the binary that launched
    /// the job.
    pub blut_build: String,
    /// The code identity folded into every stage cache key: the engine and
    /// each registered cookbook by name, version, build commit and dirty
    /// digest. Absent on jobs launched before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_identity: Option<crate::framework::code_identity::CodeIdentity>,
    /// Unix seconds when execution started.
    pub started_unix: i64,
}

const PROVENANCE_FILE: &str = "provenance.json";

/// Record [`LaunchProvenance`] for a job that is starting. Written once: a
/// `plan resume` keeps the original launch's record.
pub(crate) fn write_provenance(job_id: &str, git_sha: Option<String>) -> Result<()> {
    let path = paths::job_dir(job_id)?.join(PROVENANCE_FILE);
    if path.exists() {
        return Ok(());
    }
    let record = LaunchProvenance {
        git_sha,
        blut_build: env!("BLUT_GIT_HASH").to_string(),
        code_identity: Some(crate::framework::code_identity::current().clone()),
        started_unix: chrono::Utc::now().timestamp(),
    };
    let body = serde_json::to_vec_pretty(&record)
        .map_err(|e| TrainError::other(format!("serialize {PROVENANCE_FILE}: {e}")))?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, body).map_err(|source| TrainError::Io {
        path: tmp.clone(),
        source,
    })?;
    std::fs::rename(&tmp, &path).map_err(|source| TrainError::Io { path, source })
}

/// Read a job's [`LaunchProvenance`]. Jobs launched before it existed return
/// `None`; a present but unreadable record is an error, never a guess.
pub(crate) fn read_provenance(job_id: &str) -> Result<Option<LaunchProvenance>> {
    let path = paths::jobs_dir()?.join(job_id).join(PROVENANCE_FILE);
    let body = match std::fs::read(&path) {
        Ok(body) => body,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(TrainError::Io { path, source }),
    };
    serde_json::from_slice(&body)
        .map(Some)
        .map_err(|e| TrainError::other(format!("parse {}: {e}", path.display())))
}

/// The creation time encoded in a `YYYYMMDD-HHMMSS-NNNNNNNNN` job id (UTC), for
/// jobs that predate [`LaunchProvenance`]. `None` for any other id shape.
pub(crate) fn job_id_unix(job_id: &str) -> Option<i64> {
    let stamp = job_id.get(..15)?;
    let parsed = chrono::NaiveDateTime::parse_from_str(stamp, "%Y%m%d-%H%M%S").ok()?;
    Some(parsed.and_utc().timestamp())
}

/// When a job reached its terminal state: the modification time of its
/// `state` file, which is rewritten on every transition. `None` while the
/// job is running or when the state cannot be read.
pub(crate) fn terminal_state_unix(job_id: &str) -> Option<i64> {
    let path = paths::jobs_dir().ok()?.join(job_id).join("state");
    let body = std::fs::read_to_string(&path).ok()?;
    match JobState::parse_label(&body)? {
        JobState::Running => None,
        _ => {
            let modified = std::fs::metadata(&path).ok()?.modified().ok()?;
            let secs = modified
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_secs();
            i64::try_from(secs).ok()
        }
    }
}

/// Persist the experiment grouping key used by `experiment://<name>/<run>`.
/// Safe identifier grammar keeps the marker URI/log-safe and path-free.
pub fn write_experiment(job_id: &str, experiment: &str) -> Result<()> {
    if !crate::experiment_registry::is_safe_experiment_name(experiment) {
        return Err(TrainError::other(format!(
            "invalid experiment name '{experiment}' (expected [A-Za-z0-9_.-]+)"
        )));
    }
    let path = paths::job_dir(job_id)?.join("experiment");
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, experiment).map_err(|source| TrainError::Io {
        path: tmp.clone(),
        source,
    })?;
    std::fs::rename(&tmp, &path).map_err(|source| TrainError::Io { path, source })
}

/// Read an experiment marker. Legacy jobs return `None` and are grouped by
/// recipe name by the lineage migration/view.
pub fn read_experiment(job_id: &str) -> Result<Option<String>> {
    let path = paths::job_dir(job_id)?.join("experiment");
    let body = match std::fs::read_to_string(&path) {
        Ok(body) => body,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(TrainError::Io { path, source }),
    };
    let experiment = body.trim();
    if !crate::experiment_registry::is_safe_experiment_name(experiment) || experiment != body {
        return Err(TrainError::other(format!(
            "invalid experiment marker '{}' at {}",
            experiment,
            path.display()
        )));
    }
    Ok(Some(experiment.to_string()))
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

/// Every trainer [`StatusUpdate`] in a job's `status.jsonl`, in order: bare
/// update lines, and updates a stage forwarded as a `stage_step` event's
/// `update` payload. A plan launch only writes the second kind (a trainer
/// stage relays its trainer's stdout that way), and this read only the first,
/// so `blut log` printed nothing and `blut jobs` no step or loss for every
/// job a recipe ran.
pub fn read_status(job_id: &str) -> Result<Vec<StatusUpdate>> {
    Ok(read_progress(job_id)?
        .into_iter()
        .map(|progress| progress.update)
        .collect())
}

/// One trainer update and, when a stage forwarded it, the stage it came from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Progress {
    /// `(node_idx, stage_name)` of the forwarding stage; `None` for a bare
    /// update line.
    pub(crate) source: Option<(u32, String)>,
    pub(crate) update: StatusUpdate,
}

/// [`read_status`], keeping each update's forwarding stage.
pub(crate) fn read_progress(job_id: &str) -> Result<Vec<Progress>> {
    Ok(read_status_lines(job_id)?
        .iter()
        .filter_map(|line| progress_from_line(line))
        .collect())
}

/// The trainer update one `status.jsonl` line carries, if any.
fn progress_from_line(line: &str) -> Option<Progress> {
    // Cheap test first: most lines of a plan launch are stage events that are
    // not `stage_step`, and every bare update line parses directly.
    if !line.contains("\"stage_step\"") {
        return serde_json::from_str(line).ok().map(|update| Progress {
            source: None,
            update,
        });
    }
    let event: serde_json::Value = serde_json::from_str(line).ok()?;
    if event.get("kind").and_then(|kind| kind.as_str()) != Some("stage_step") {
        return None;
    }
    let update = serde_json::from_value(event.get("update")?.clone()).ok()?;
    let node_idx = event
        .get("node_idx")
        .and_then(|idx| idx.as_u64())
        .and_then(|idx| u32::try_from(idx).ok())?;
    let stage_name = event.get("stage_name")?.as_str()?.to_string();
    Some(Progress {
        source: Some((node_idx, stage_name)),
        update,
    })
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
    // The child pid alone cannot see a job whose stages run IN-PROCESS (no
    // trainer child, so no `pid` file): kill -9 of the blut process left it
    // `running` forever. The recorded driver closes that gap.
    let state = if on_disk_state == JobState::Running
        && read_driver(id)
            .ok()
            .flatten()
            .is_some_and(|d| driver_gone(&d))
    {
        JobState::Failed
    } else {
        reconcile_liveness(on_disk_state, pid)
    };
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
            "job id is empty. Run `blut jobs` to list.",
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
            "no job matches '{query}'. Run `blut jobs` to list."
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

/// `blut cancel` core.
///
/// A job whose blut process (its DRIVER, see [`record_driver`]) is still
/// alive is stopped through that process: SIGTERM runs its cancel handler,
/// which cancels the executor and kills every trainer group, and after
/// `grace` it is SIGKILLed. Stopping only the trainer child is not enough —
/// a job with no subprocess stage has no child at all, and a retrying stage
/// would spawn a new one. With no live driver, the recorded trainer group (a
/// child orphaned by a crashed driver) is SIGTERMed and then SIGKILLed.
///
/// A job that already finished (`done`/`failed`/`cancelled`) with nothing
/// left running is left exactly as it is; its state is never rewritten.
/// Refuses, without signalling anything, when the driver also runs other
/// jobs (a sweep or backfill), because stopping it would stop them all.
pub async fn cancel_job(job_id: &str, grace: std::time::Duration) -> Result<()> {
    let state = read_state(job_id).ok();
    let driver = read_driver(job_id)?
        .filter(|d| d.pid != std::process::id() && crate::python_kill::identity_alive(d));
    if let Some(driver) = driver {
        let siblings = jobs_sharing_driver(job_id, &driver);
        if !siblings.is_empty() {
            return Err(TrainError::other(format!(
                "job {job_id} runs in blut process {} together with {} other running \
                 job(s) ({}); cancelling it alone is not possible. Stop the whole batch \
                 with `kill -TERM {}`.",
                driver.pid,
                siblings.len(),
                siblings.join(", "),
                driver.pid
            )));
        }
        crate::python_kill::terminate_process(&driver, grace)
            .await
            .map_err(|e| TrainError::other(format!("cancel {job_id}: {e}")))?;
    } else if matches!(
        state,
        Some(JobState::Done | JobState::Failed | JobState::Cancelled)
    ) {
        // Nothing local is left, but a Slurm job its stage started can be.
        stop_slurm_jobs(job_id);
        return Ok(());
    }
    let pid = read_pid(job_id)?;
    if let Some(pid) = pid {
        crate::python_kill::graceful_kill_pid(pid, grace).await;
    }
    stop_slurm_jobs(job_id);
    let _ = write_state(job_id, JobState::Cancelled);
    let _ = clear_pid(job_id);
    let _ = clear_driver(job_id);
    Ok(())
}

/// File listing, one per line, the Slurm job names a job's stages were told
/// to use ([`slurm_job_name`]).
const SLURM_JOBS_FILE: &str = "slurm_jobs";

/// The Slurm job name for node `node_idx` of the job in `job_dir`:
/// `blut-<job id>-<node index>`.
pub fn slurm_job_name(job_dir: &Path, node_idx: u32) -> String {
    let job = job_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!("blut-{job}-{node_idx}")
}

/// Record `name` in the job directory, once.
pub fn record_slurm_job_name(job_dir: &Path, name: &str) -> Result<()> {
    let path = job_dir.join(SLURM_JOBS_FILE);
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    if existing.lines().any(|l| l == name) {
        return Ok(());
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|source| TrainError::Io {
            path: path.clone(),
            source,
        })?;
    use std::io::Write;
    writeln!(f, "{name}").map_err(|source| TrainError::Io { path, source })
}

/// The Slurm job names recorded for `job_id`.
pub fn slurm_job_names(job_id: &str) -> Result<Vec<String>> {
    let path = paths::jobs_dir()?.join(job_id).join(SLURM_JOBS_FILE);
    match std::fs::read_to_string(&path) {
        Ok(body) => Ok(body
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(source) => Err(TrainError::Io { path, source }),
    }
}

/// Slurm jobs still queued or running under a name recorded for `job_id`:
/// `(name, ids)` for each name that has any.
pub fn live_slurm_jobs(job_id: &str) -> Result<Vec<(String, Vec<String>)>> {
    let mut live = Vec::new();
    for name in slurm_job_names(job_id)? {
        let ids = crate::config::launcher::slurm_jobs_named(&name)?;
        if !ids.is_empty() {
            live.push((name, ids));
        }
    }
    Ok(live)
}

/// Why `plan resume` must not start yet: a Slurm job a stage of `job_id`
/// started is still queued or running. `None` when there is none.
pub fn slurm_resume_blocker(job_id: &str) -> Result<Option<String>> {
    let live = live_slurm_jobs(job_id)?;
    if live.is_empty() {
        return Ok(None);
    }
    let list = live
        .iter()
        .map(|(name, ids)| format!("{} ({name})", ids.join(", ")))
        .collect::<Vec<_>>()
        .join("; ");
    Ok(Some(format!(
        "job {job_id} still has Slurm job(s) running that its stages started: {list}. \
         Stop them with `blut cancel {job_id}`, or wait for them to finish, then resume"
    )))
}

/// `scancel` every Slurm job still running under a name recorded for
/// `job_id`. Returns the ids cancelled.
pub fn cancel_slurm_jobs(job_id: &str) -> Result<Vec<String>> {
    let mut cancelled = Vec::new();
    for name in slurm_job_names(job_id)? {
        cancelled.extend(crate::config::launcher::scancel_named(&name)?);
    }
    Ok(cancelled)
}

/// `scancel` the job's recorded Slurm jobs and say what happened. A Slurm job
/// keeps running when the process that started it is killed (killing `srun`
/// does not end the job), so stopping a BLUT job has to stop it too.
fn stop_slurm_jobs(job_id: &str) {
    if slurm_job_names(job_id)
        .map(|n| n.is_empty())
        .unwrap_or(true)
    {
        return;
    }
    match cancel_slurm_jobs(job_id) {
        Ok(ids) if ids.is_empty() => {}
        Ok(ids) => eprintln!("cancelled Slurm job(s) {} of {job_id}", ids.join(", ")),
        Err(e) => eprintln!("warning: could not stop the Slurm jobs of {job_id}: {e}"),
    }
}

/// File naming the blut process that executes a job in-process.
const DRIVER_FILE: &str = "driver";

/// Record THIS process as the driver of `job_id` until the returned guard
/// drops. The record is what lets a separate process tell a live job from a
/// dead one when no trainer child exists (`blut jobs`), stop it (`blut
/// cancel`), and refuse to resume it underneath itself (`blut plan resume`).
/// It survives only an abnormal exit (SIGKILL, OOM kill, power loss), which
/// is exactly the case it is there to detect. Recording is best-effort: a
/// failure is logged and the job runs as before.
pub fn record_driver(job_id: &str) -> DriverRecord {
    let me = crate::python_kill::capture_identity(std::process::id());
    let body = match me.start_time {
        Some(start) => format!("{} {start}", me.pid),
        None => me.pid.to_string(),
    };
    let written = paths::job_dir(job_id)
        .and_then(|dir| {
            let path = dir.join(DRIVER_FILE);
            std::fs::write(&path, body).map_err(|e| TrainError::Io { path, source: e })
        })
        .map_err(|e| tracing::warn!("record driver for {job_id}: {e}"))
        .is_ok();
    DriverRecord {
        job_id: job_id.to_string(),
        written,
    }
}

/// Guard returned by [`record_driver`]; removes the record on drop.
#[must_use = "the driver record is removed when this guard drops"]
pub struct DriverRecord {
    job_id: String,
    written: bool,
}

impl Drop for DriverRecord {
    fn drop(&mut self) {
        // Keep the record when the job is still marked `running`: the run
        // ended without recording a final state (the state write failed, e.g.
        // on a full disk), and the record is then the only thing that lets
        // `blut jobs` see that this process is gone.
        if self.written && read_state(&self.job_id).ok() != Some(JobState::Running) {
            let _ = clear_driver(&self.job_id);
        }
    }
}

/// The recorded driver of `job_id`, if any (`pid` and, where the platform
/// exposes it, the process start time that guards against pid reuse).
pub fn read_driver(job_id: &str) -> Result<Option<crate::python_kill::ChildIdentity>> {
    let path = paths::job_dir(job_id)?.join(DRIVER_FILE);
    let body = match std::fs::read_to_string(&path) {
        Ok(body) => body,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(TrainError::Io { path, source: e }),
    };
    let mut parts = body.split_whitespace();
    let Some(pid) = parts.next().and_then(|p| p.parse::<u32>().ok()) else {
        return Ok(None);
    };
    let start_time = parts.next().and_then(|s| s.parse::<u64>().ok());
    Ok(Some(crate::python_kill::ChildIdentity {
        pid,
        pgid: pid,
        start_time,
    }))
}

fn clear_driver(job_id: &str) -> Result<()> {
    let path = paths::job_dir(job_id)?.join(DRIVER_FILE);
    let _ = std::fs::remove_file(path);
    Ok(())
}

/// The pid of the process still executing `job_id`, or `None` when no
/// driver is recorded or the recorded one is gone (or its pid was reused).
pub fn live_driver(job_id: &str) -> Option<u32> {
    read_driver(job_id)
        .ok()
        .flatten()
        .filter(crate::python_kill::identity_alive)
        .map(|d| d.pid)
}

/// True only when the recorded driver is definitively not running. Off Unix
/// liveness cannot be probed, so this never claims a driver is gone there.
#[cfg(unix)]
fn driver_gone(d: &crate::python_kill::ChildIdentity) -> bool {
    !crate::python_kill::identity_alive(d)
}

#[cfg(not(unix))]
fn driver_gone(_d: &crate::python_kill::ChildIdentity) -> bool {
    false
}

/// Other `running` jobs whose recorded driver is the same live process.
fn jobs_sharing_driver(job_id: &str, driver: &crate::python_kill::ChildIdentity) -> Vec<String> {
    let Ok(dir) = paths::jobs_dir() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let Ok(id) = entry.file_name().into_string() else {
            continue;
        };
        if id == job_id || !entry.path().join(DRIVER_FILE).is_file() {
            continue;
        }
        if read_state(&id).ok() == Some(JobState::Running)
            && read_driver(&id).ok().flatten().is_some_and(|other| {
                other.pid == driver.pid && other.start_time == driver.start_time
            })
        {
            out.push(id);
        }
    }
    out.sort();
    out
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
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
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
            nproc_per_node: 1,
            nnodes: 1,
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

    /// A plan launch writes trainer progress only as `stage_step` events (the
    /// trainer stage forwards each `StatusUpdate`), and `read_status` parsed
    /// only bare update lines: `blut log` printed nothing and `blut jobs` had
    /// no step, loss or final loss for any job a recipe ran.
    #[test]
    fn read_status_includes_updates_a_stage_forwarded() {
        use crate::framework::status::{HostedEvent, StageEvent};
        with_jobs_dir(|| {
            let id = "20261030-000000-000000001";
            let dir = paths::job_dir(id).unwrap();
            std::fs::create_dir_all(&dir).unwrap();
            let step = |update: serde_json::Value| {
                serde_json::to_string(&HostedEvent::wrap(
                    &None,
                    StageEvent::StageStep {
                        node_idx: 2,
                        stage_name: "train".into(),
                        update,
                    },
                ))
                .unwrap()
            };
            let as_value = |u: &StatusUpdate| serde_json::to_value(u).unwrap();
            let lines = [
                "{\"kind\":\"stage_begin\",\"node_idx\":2,\"stage_name\":\"train\",\"input_hash\":\"x\"}"
                    .to_string(),
                step(as_value(&StatusUpdate::Heartbeat {
                    phase: Some("load".into()),
                    vram_mb: None,
                })),
                step(as_value(&StatusUpdate::Step {
                    step: 1,
                    total: 2,
                    loss: 2.0,
                    lr: 0.001,
                    vram_mb: 0,
                })),
                // Not trainer protocol: metrics, gauges, a cookbook's own kind.
                step(serde_json::json!({"step": 1, "val_r": 0.4})),
                step(serde_json::json!({"kind": "gpu_gauge", "gpu_util": 90.0})),
                step(serde_json::json!({"kind": "hf_step", "step": 1, "loss": 1.0})),
                // A forwarded peer's line carries a `host` tag.
                format!(
                    "{{\"host\":\"ab12\",{}",
                    &step(as_value(&StatusUpdate::Step {
                        step: 2,
                        total: 2,
                        loss: 1.0,
                        lr: 0.001,
                        vram_mb: 0,
                    }))[1..]
                ),
                step(as_value(&StatusUpdate::Done {
                    final_loss: 0.5,
                    checkpoint_dir: PathBuf::from("/ckpt"),
                })),
                "{\"kind\":\"stage_end\",\"node_idx\":2,\"stage_name\":\"train\"}".to_string(),
            ];
            std::fs::write(dir.join("status.jsonl"), lines.join("\n") + "\n").unwrap();
            // A bare update line (the single-trainer launch's format) still reads.
            append_status(
                id,
                &StatusUpdate::Eval {
                    step: 2,
                    eval_loss: 0.7,
                },
            )
            .unwrap();

            let progress = read_progress(id).unwrap();
            let kinds: Vec<(&str, Option<u32>)> = progress
                .iter()
                .map(|p| {
                    let kind = match p.update {
                        StatusUpdate::Heartbeat { .. } => "heartbeat",
                        StatusUpdate::Step { .. } => "step",
                        StatusUpdate::Done { .. } => "done",
                        StatusUpdate::Eval { .. } => "eval",
                        _ => "other",
                    };
                    (kind, p.source.as_ref().map(|(idx, _)| *idx))
                })
                .collect();
            assert_eq!(
                kinds,
                [
                    ("heartbeat", Some(2)),
                    ("step", Some(2)),
                    ("step", Some(2)),
                    ("done", Some(2)),
                    ("eval", None),
                ]
            );
            assert_eq!(read_status(id).unwrap().len(), 5);
            write_state(id, JobState::Done).unwrap();
            let summary = list_jobs().unwrap().pop().unwrap();
            assert_eq!(summary.last_step, Some(2));
            assert_eq!(summary.last_loss, Some(1.0));
            assert_eq!(summary.final_loss, Some(0.5));
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
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
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

    // ── Job driver: cancel / liveness for in-process jobs ───────────

    /// Spawn `sleep 30` as a stand-in driver and reap it on a thread, so its
    /// exit is observable (an unreaped child would linger as a zombie).
    #[cfg(target_os = "linux")]
    fn spawn_driver_stand_in() -> (
        crate::python_kill::ChildIdentity,
        std::sync::mpsc::Receiver<()>,
    ) {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn `sleep 30`");
        let identity = crate::python_kill::capture_identity(child.id());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = child.wait();
            let _ = tx.send(());
        });
        (identity, rx)
    }

    #[cfg(target_os = "linux")]
    fn write_driver_file(id: &str, d: &crate::python_kill::ChildIdentity) {
        let body = format!("{} {}", d.pid, d.start_time.expect("linux start time"));
        std::fs::write(paths::job_dir(id).unwrap().join(DRIVER_FILE), body).unwrap();
    }

    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(f)
    }

    /// `blut cancel` on a job whose stages run in-process (no trainer child,
    /// so no `pid` file) used to print "cancelled." and exit 0 while the job
    /// ran on to `done`. It must stop the driving process.
    #[cfg(target_os = "linux")]
    #[test]
    fn cancel_stops_the_driver_of_an_in_process_job() {
        let (driver, exited) = spawn_driver_stand_in();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_jobs_dir(|| {
                let id = "20261020-000000-000000001";
                write_state(id, JobState::Running).unwrap();
                write_driver_file(id, &driver);
                block_on(cancel_job(id, std::time::Duration::from_secs(5))).unwrap();
                assert!(
                    exited
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .is_ok(),
                    "the driver must have been stopped"
                );
                assert_eq!(read_state(id).unwrap(), JobState::Cancelled);
                assert!(read_driver(id).unwrap().is_none());
            })
        }));
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(driver.pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        );
        if let Err(p) = result {
            std::panic::resume_unwind(p);
        }
    }

    /// A driver that runs several jobs (a sweep or backfill) is not stopped to
    /// cancel one of them; the refusal says how to stop the batch.
    #[cfg(target_os = "linux")]
    #[test]
    fn cancel_refuses_a_driver_shared_with_other_running_jobs() {
        let (driver, exited) = spawn_driver_stand_in();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_jobs_dir(|| {
                for id in ["20261020-000000-000000002", "20261020-000000-000000003"] {
                    write_state(id, JobState::Running).unwrap();
                    write_driver_file(id, &driver);
                }
                let err = block_on(cancel_job(
                    "20261020-000000-000000002",
                    std::time::Duration::from_secs(1),
                ))
                .expect_err("must refuse");
                assert!(err.to_string().contains("kill -TERM"), "{err}");
                assert!(
                    exited
                        .recv_timeout(std::time::Duration::from_millis(300))
                        .is_err(),
                    "the shared driver must not be signalled"
                );
                assert_eq!(
                    read_state("20261020-000000-000000002").unwrap(),
                    JobState::Running
                );
            })
        }));
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(driver.pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        );
        if let Err(p) = result {
            std::panic::resume_unwind(p);
        }
    }

    /// Cancelling a job that already finished must not rewrite its state:
    /// `done` used to become `cancelled`.
    #[test]
    fn cancel_leaves_a_finished_job_alone() {
        with_jobs_dir(|| {
            let id = "20261020-000000-000000004";
            write_state(id, JobState::Done).unwrap();
            block_on(cancel_job(id, std::time::Duration::from_secs(1))).unwrap();
            assert_eq!(read_state(id).unwrap(), JobState::Done);
        });
    }

    /// A job whose driver was SIGKILLed (no trainer child, so no `pid` file)
    /// used to stay `running` forever in `blut jobs`.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_job_whose_driver_died_is_reported_failed() {
        let (driver, exited) = spawn_driver_stand_in();
        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(driver.pid as i32),
            nix::sys::signal::Signal::SIGKILL,
        )
        .unwrap();
        exited
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("driver stand-in exits");
        with_jobs_dir(|| {
            let id = "20261020-000000-000000005";
            write_state(id, JobState::Running).unwrap();
            write_driver_file(id, &driver);
            let job = list_jobs()
                .unwrap()
                .into_iter()
                .find(|j| j.id == id)
                .unwrap();
            assert_eq!(job.state, JobState::Failed);
            assert_eq!(live_driver(id), None);
        });
    }

    /// The record names this process while the guard lives and is removed
    /// when it drops (the normal-exit path).
    #[cfg(target_os = "linux")]
    #[test]
    fn driver_record_names_this_process_until_dropped() {
        with_jobs_dir(|| {
            let id = "20261020-000000-000000006";
            let guard = record_driver(id);
            assert_eq!(live_driver(id), Some(std::process::id()));
            drop(guard);
            assert!(read_driver(id).unwrap().is_none());
        });
    }

    /// A run that could not write its final state (full disk) left the job
    /// `running` with no trace of its process, so it looked live forever.
    #[test]
    fn driver_record_outlives_a_run_that_never_left_running() {
        with_jobs_dir(|| {
            let id = "20261020-000000-000000007";
            write_state(id, JobState::Running).unwrap();
            drop(record_driver(id));
            assert!(read_driver(id).unwrap().is_some());
            write_state(id, JobState::Done).unwrap();
            drop(record_driver(id));
            assert!(read_driver(id).unwrap().is_none());
        });
    }
}
