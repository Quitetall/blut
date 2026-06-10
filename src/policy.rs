//! Auto-trigger heuristic for `lamu-train auto`.
//!
//! Goal: keep a personal model fresh against accumulated
//! conversation history without the user having to manually fire
//! `lamu-train`. A cron entry runs `lamu-train auto` every ~30 min;
//! `auto` reads `train-policy.toml` + the conversation DB + the
//! scheduler lockfile and decides whether to spawn a training run.
//!
//! Decision logic (in order):
//!
//!   1. `enabled = false` — exit cleanly with "auto-trigger disabled".
//!   2. Outside `quiet_hours` — exit cleanly. Quiet hours bound when
//!      the heavy GPU load is acceptable; default 02:00–06:00.
//!   3. Within `cooldown_days` of `last_train_ts` — exit cleanly.
//!      Prevents back-to-back retraining if the cron fires during
//!      a short window after a successful run.
//!   4. Fewer than `threshold_new_turns` since `last_train_ts` —
//!      exit cleanly. Don't burn 4 GPU-hours on a 50-turn delta.
//!   5. Scheduler lock held by inference — exit cleanly. Cron will
//!      retry next tick. Never preempts inference.
//!
//! All "exit cleanly" returns are `Decision::Skip(reason)`. Only
//! `Decision::Run` triggers an actual spawn.
//!
//! Persistence: `~/.config/lamu/train-policy.toml` (override via
//! `$LAMU_TRAIN_POLICY`). Atomic writes via tmp + rename so a
//! crashed update never corrupts the file.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{Result, TrainError};

/// Maximum sane window. Mirrors `lamu-mcp::train_tool::MAX_SINCE`.
/// 10 years is well past the history any user will have.
const MAX_SINCE_SECS: u64 = 10 * 365 * 24 * 60 * 60;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainPolicy {
    /// Off by default. User opts in via `lamu-train policy enable`.
    #[serde(default)]
    pub enabled: bool,

    /// HuggingFace base model id.
    #[serde(default = "default_base")]
    pub base: String,

    /// `qlora` | `lora` | `full`.
    #[serde(default = "default_method")]
    pub method: String,

    /// Minimum new turns since `last_train_ts` before a run is
    /// triggered.
    #[serde(default = "default_threshold")]
    pub threshold_new_turns: i64,

    /// Minimum days between training runs. Even with plenty of
    /// new turns, we don't run more than once per `cooldown_days`.
    #[serde(default = "default_cooldown")]
    pub cooldown_days: u32,

    /// Allowed window expressed as ["HH:MM", "HH:MM"] in 24h local
    /// time. The first element is the start (inclusive), the
    /// second is the end (exclusive). If start > end the window
    /// wraps midnight (`["22:00", "06:00"]` = 10 PM – 6 AM).
    #[serde(default = "default_quiet_hours")]
    pub quiet_hours: [String; 2],

    /// Window of conversation history to use as the dataset.
    /// Humantime duration string ("30d", "60d", etc.).
    #[serde(default = "default_since_window")]
    pub since_window: String,

    /// UNIX seconds. Updated atomically after a successful train
    /// COMPLETION (not spawn) so a failed run never advances the
    /// cooldown. Zero on first run.
    #[serde(default)]
    pub last_train_ts: i64,

    /// Number of turns the last training run consumed. Diagnostic.
    #[serde(default)]
    pub last_train_n_turns: i64,

    /// UNIX seconds of the last auto SPAWN (success or failure).
    /// Drives the failure backoff window below. Distinct from
    /// `last_train_ts` (which only advances on success).
    #[serde(default)]
    pub last_attempt_ts: i64,

    /// Consecutive auto-run failures. Each failure doubles the backoff
    /// window (`BACKOFF_BASE_SECS * 2^failures`, capped at
    /// `cooldown_days`); a success resets it to 0. Stops a perpetually
    /// failing trainer from re-spawning every cron tick.
    #[serde(default)]
    pub consecutive_failures: u32,
}

/// Base failure-backoff window: 1 hour. Doubles per consecutive
/// failure, capped at `cooldown_days`. Internal to `decide()`.
const BACKOFF_BASE_SECS: i64 = 3600;

impl Default for TrainPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            base: default_base(),
            method: default_method(),
            threshold_new_turns: default_threshold(),
            cooldown_days: default_cooldown(),
            quiet_hours: default_quiet_hours(),
            since_window: default_since_window(),
            last_train_ts: 0,
            last_train_n_turns: 0,
            last_attempt_ts: 0,
            consecutive_failures: 0,
        }
    }
}

fn default_base() -> String {
    "Qwen/Qwen3-7B".into()
}
fn default_method() -> String {
    "qlora".into()
}
fn default_threshold() -> i64 {
    500
}
fn default_cooldown() -> u32 {
    7
}
fn default_quiet_hours() -> [String; 2] {
    ["02:00".into(), "06:00".into()]
}
fn default_since_window() -> String {
    "30d".into()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Conditions met; trigger a training spawn. Carries the
    /// resolved values the caller should pass to `lamu-train`.
    Run {
        base: String,
        method: String,
        since: String,
    },
    /// Skip with a human-readable reason. The cron-driven CLI
    /// prints this on stdout and exits 0.
    Skip(String),
}

/// Path to the policy file. Override via `$LAMU_TRAIN_POLICY` for
/// hermetic tests + bespoke installs.
pub fn policy_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LAMU_TRAIN_POLICY") {
        return Ok(PathBuf::from(p));
    }
    let dir = dirs::config_dir()
        .ok_or_else(|| TrainError::other("config_dir() unavailable; set $LAMU_TRAIN_POLICY"))?
        .join("lamu");
    Ok(dir.join("train-policy.toml"))
}

/// Read the policy from disk. Returns Default when the file
/// doesn't exist yet — first invocation of `policy show` shouldn't
/// require a manual touch.
pub fn load() -> Result<TrainPolicy> {
    load_at(&policy_path()?)
}

pub fn load_at(path: &Path) -> Result<TrainPolicy> {
    if !path.exists() {
        return Ok(TrainPolicy::default());
    }
    let body = std::fs::read_to_string(path).map_err(|e| TrainError::Io {
        path: path.into(),
        source: e,
    })?;
    toml::from_str(&body).map_err(|e| TrainError::other(format!("parse {}: {e}", path.display())))
}

/// Atomically write the policy. Tmp + rename — same pattern as
/// `lamu-core::registry::write_atomic`.
pub fn save(policy: &TrainPolicy) -> Result<()> {
    save_at(&policy_path()?, policy)
}

pub fn save_at(path: &Path, policy: &TrainPolicy) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| TrainError::Io {
            path: parent.into(),
            source: e,
        })?;
    }
    let body = toml::to_string_pretty(policy)
        .map_err(|e| TrainError::other(format!("serialize policy: {e}")))?;
    let stem = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "train-policy.toml".into());
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_file_name(format!(".{stem}.tmp.{}.{nanos}", std::process::id()));
    std::fs::write(&tmp, body).map_err(|e| TrainError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(TrainError::Io {
            path: path.into(),
            source: e,
        });
    }
    Ok(())
}

/// Pure decision function. All inputs explicit so tests can drive
/// every branch with synthetic clocks + counts.
///
/// `now_unix_secs` and `now_local_minutes_of_day` are split: the
/// first drives cooldown comparisons, the second drives quiet-hours.
/// Splitting them makes timezone shifts easy to test (a UTC clock
/// at 23:00 might be 18:00 local; quiet_hours is local).
pub fn decide(
    policy: &TrainPolicy,
    now_unix_secs: i64,
    now_local_minutes_of_day: u32,
    new_turns_since_last: i64,
    inference_lock_held: bool,
) -> Decision {
    if !policy.enabled {
        return Decision::Skip(
            "auto-trigger disabled (run `lamu-train policy enable` to opt in)".into(),
        );
    }
    let (start, end) = match parse_quiet_hours(&policy.quiet_hours) {
        Ok(p) => p,
        Err(e) => return Decision::Skip(format!("invalid quiet_hours: {e}")),
    };
    if !in_window(now_local_minutes_of_day, start, end) {
        return Decision::Skip(format!(
            "outside quiet_hours ({}–{})",
            policy.quiet_hours[0], policy.quiet_hours[1]
        ));
    }
    if policy.cooldown_days > 0 && policy.last_train_ts > 0 {
        let cooldown_secs = policy.cooldown_days as i64 * 86400;
        let since_last = now_unix_secs - policy.last_train_ts;
        if since_last < cooldown_secs {
            let days_left = (cooldown_secs - since_last + 86399) / 86400;
            return Decision::Skip(format!(
                "in cooldown ({} day(s) remaining of {})",
                days_left, policy.cooldown_days
            ));
        }
    }
    // Failure backoff: after consecutive auto-run failures, wait an
    // exponentially growing window from the last ATTEMPT (capped at the
    // cooldown) before retrying — so a perpetually failing trainer can't
    // re-spawn every cron tick.
    if policy.consecutive_failures > 0 && policy.last_attempt_ts > 0 {
        let cooldown_cap = (policy.cooldown_days.max(1) as i64) * 86400;
        let backoff = BACKOFF_BASE_SECS
            .saturating_mul(1i64 << policy.consecutive_failures.min(20))
            .min(cooldown_cap);
        let since_attempt = now_unix_secs - policy.last_attempt_ts;
        if since_attempt < backoff {
            let mins_left = (backoff - since_attempt + 59) / 60;
            return Decision::Skip(format!(
                "in failure backoff after {} consecutive failure(s); \
                 {} min remaining",
                policy.consecutive_failures, mins_left
            ));
        }
    }
    if new_turns_since_last < policy.threshold_new_turns {
        return Decision::Skip(format!(
            "only {} new turns since last train; threshold is {}",
            new_turns_since_last, policy.threshold_new_turns
        ));
    }
    if inference_lock_held {
        return Decision::Skip("GPU held by inference; will retry next tick".into());
    }
    Decision::Run {
        base: policy.base.clone(),
        method: policy.method.clone(),
        since: policy.since_window.clone(),
    }
}

/// Parse one HH:MM string to minutes-of-day. Used by both
/// quiet-hours endpoints.
fn parse_hhmm(s: &str) -> std::result::Result<u32, String> {
    let parts: Vec<&str> = s.trim().split(':').collect();
    if parts.len() != 2 {
        return Err(format!("'{s}' is not HH:MM"));
    }
    let h: u32 = parts[0]
        .parse()
        .map_err(|e| format!("hour in '{s}': {e}"))?;
    let m: u32 = parts[1]
        .parse()
        .map_err(|e| format!("minute in '{s}': {e}"))?;
    if h >= 24 || m >= 60 {
        return Err(format!("'{s}' out of range"));
    }
    Ok(h * 60 + m)
}

fn parse_quiet_hours(qh: &[String; 2]) -> std::result::Result<(u32, u32), String> {
    Ok((parse_hhmm(&qh[0])?, parse_hhmm(&qh[1])?))
}

/// True iff `t` is in the [start, end) window. Wraps midnight
/// when start > end (e.g., 22:00–06:00 covers 23:00 and 03:00 but
/// not 12:00).
fn in_window(t: u32, start: u32, end: u32) -> bool {
    if start == end {
        // Empty window — never in.
        return false;
    }
    if start < end {
        t >= start && t < end
    } else {
        // Wraps midnight.
        t >= start || t < end
    }
}

/// Validate a policy before save — caller-friendly error messages
/// for malformed user edits.
pub fn validate(policy: &TrainPolicy) -> Result<()> {
    if !matches!(policy.method.as_str(), "qlora" | "lora" | "full") {
        return Err(TrainError::other(format!(
            "method '{}' must be one of qlora|lora|full",
            policy.method
        )));
    }
    if policy.base.trim().is_empty() || !policy.base.contains('/') {
        return Err(TrainError::other(format!(
            "base '{}' must look like an HF repo id (org/name)",
            policy.base
        )));
    }
    parse_quiet_hours(&policy.quiet_hours).map_err(TrainError::other)?;
    let since = humantime::parse_duration(&policy.since_window)
        .map_err(|e| TrainError::other(format!("since_window '{}': {e}", policy.since_window)))?;
    if since.as_secs() > MAX_SINCE_SECS {
        return Err(TrainError::other(format!(
            "since_window '{}' exceeds 10-year cap",
            policy.since_window
        )));
    }
    if policy.threshold_new_turns < 0 {
        return Err(TrainError::other("threshold_new_turns must be >= 0"));
    }
    Ok(())
}

/// Record the outcome of an auto-triggered training run so the cooldown
/// and failure backoff stay honest. No-op unless `output_name` starts
/// with `auto-` (manual runs don't touch the auto policy). A success
/// advances `last_train_ts` to now and clears `consecutive_failures`; a
/// failure increments `consecutive_failures` (growing the backoff
/// window). Best-effort — a load/save failure is logged, never
/// propagated, so it can be called from a job's terminal path without
/// masking the real outcome.
pub fn record_auto_outcome(output_name: &str, success: bool) {
    if !output_name.starts_with("auto-") {
        return;
    }
    // load → modify → save is not file-locked. Concurrent auto completions
    // could lose one update (last-writer-wins), but the ≥1h failure backoff
    // makes overlapping auto runs vanishingly unlikely on a single box, and
    // a lost increment only shortens one backoff window — acceptable.
    let mut p = match load() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("record_auto_outcome: load policy failed: {e}");
            return;
        }
    };
    if success {
        let (now, _) = current_clock();
        p.last_train_ts = now;
        p.consecutive_failures = 0;
    } else {
        p.consecutive_failures = p.consecutive_failures.saturating_add(1);
    }
    if let Err(e) = save(&p) {
        tracing::warn!("record_auto_outcome: save policy failed: {e}");
    }
}

/// Helper for the production `auto` CLI: returns now() in UNIX
/// seconds + local minutes-of-day. Pure side-effect-free wrapper
/// so the decision function can be tested with synthetic clocks.
pub fn current_clock() -> (i64, u32) {
    use chrono::{Local, Timelike};
    let now = Local::now();
    let secs_unix = now.timestamp();
    // True local minutes-of-day (honours the host timezone + DST), so
    // `quiet_hours` mean what the user wrote regardless of UTC offset.
    let minutes = now.hour() * 60 + now.minute();
    (secs_unix, minutes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_policy() -> TrainPolicy {
        TrainPolicy {
            enabled: true,
            base: "Qwen/Qwen3-7B".into(),
            method: "qlora".into(),
            threshold_new_turns: 500,
            cooldown_days: 7,
            quiet_hours: ["02:00".into(), "06:00".into()],
            since_window: "30d".into(),
            last_train_ts: 0,
            last_train_n_turns: 0,
            last_attempt_ts: 0,
            consecutive_failures: 0,
        }
    }

    fn at_3am() -> u32 {
        3 * 60
    }

    #[test]
    fn disabled_policy_skips_unconditionally() {
        let mut p = run_policy();
        p.enabled = false;
        let d = decide(&p, 0, at_3am(), 9999, false);
        assert!(matches!(d, Decision::Skip(_)));
    }

    #[test]
    fn outside_quiet_hours_skips() {
        let p = run_policy();
        // 12:00 — outside [02:00, 06:00).
        let d = decide(&p, 0, 12 * 60, 9999, false);
        match d {
            Decision::Skip(reason) => assert!(reason.contains("quiet_hours")),
            _ => panic!("expected Skip"),
        }
    }

    #[test]
    fn quiet_hours_wrap_midnight() {
        let mut p = run_policy();
        p.quiet_hours = ["22:00".into(), "06:00".into()];
        // 23:30 — inside the wrap.
        let d = decide(&p, 0, 23 * 60 + 30, 9999, false);
        assert!(matches!(d, Decision::Run { .. }));
        // 12:00 — outside.
        let d = decide(&p, 0, 12 * 60, 9999, false);
        assert!(matches!(d, Decision::Skip(_)));
    }

    #[test]
    fn cooldown_skips_within_window() {
        let mut p = run_policy();
        p.last_train_ts = 1_000_000;
        // 3 days later (< 7-day cooldown).
        let now = 1_000_000 + 3 * 86400;
        let d = decide(&p, now, at_3am(), 9999, false);
        match d {
            Decision::Skip(reason) => assert!(reason.contains("cooldown")),
            _ => panic!("expected Skip"),
        }
    }

    #[test]
    fn cooldown_clears_after_window() {
        let mut p = run_policy();
        p.last_train_ts = 1_000_000;
        // 8 days later — past cooldown.
        let now = 1_000_000 + 8 * 86400;
        let d = decide(&p, now, at_3am(), 9999, false);
        assert!(matches!(d, Decision::Run { .. }));
    }

    #[test]
    fn below_threshold_skips() {
        let p = run_policy();
        let d = decide(&p, 0, at_3am(), 100, false);
        match d {
            Decision::Skip(reason) => assert!(reason.contains("new turns")),
            _ => panic!("expected Skip"),
        }
    }

    #[test]
    fn lock_held_skips_with_retry_hint() {
        let p = run_policy();
        let d = decide(&p, 0, at_3am(), 9999, true);
        match d {
            Decision::Skip(reason) => assert!(reason.contains("retry")),
            _ => panic!("expected Skip"),
        }
    }

    #[test]
    fn all_conditions_met_runs() {
        let p = run_policy();
        let d = decide(&p, 0, at_3am(), 9999, false);
        match d {
            Decision::Run {
                base,
                method,
                since,
            } => {
                assert_eq!(base, "Qwen/Qwen3-7B");
                assert_eq!(method, "qlora");
                assert_eq!(since, "30d");
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn parse_hhmm_rejects_garbage() {
        assert!(parse_hhmm("nope").is_err());
        assert!(parse_hhmm("25:00").is_err());
        assert!(parse_hhmm("12:99").is_err());
        assert!(parse_hhmm("12").is_err());
        assert_eq!(parse_hhmm("00:00"), Ok(0));
        assert_eq!(parse_hhmm("23:59"), Ok(23 * 60 + 59));
    }

    #[test]
    fn in_window_normal_range() {
        // 02:00 – 06:00
        assert!(in_window(3 * 60, 2 * 60, 6 * 60));
        assert!(!in_window(60, 2 * 60, 6 * 60));
        assert!(!in_window(7 * 60, 2 * 60, 6 * 60));
        // Endpoint inclusive at start:
        assert!(in_window(2 * 60, 2 * 60, 6 * 60));
        // Exclusive at end:
        assert!(!in_window(6 * 60, 2 * 60, 6 * 60));
    }

    #[test]
    fn in_window_wraps_midnight() {
        // 22:00 – 06:00
        assert!(in_window(23 * 60, 22 * 60, 6 * 60));
        assert!(in_window(0, 22 * 60, 6 * 60));
        assert!(in_window(5 * 60, 22 * 60, 6 * 60));
        assert!(!in_window(12 * 60, 22 * 60, 6 * 60));
        // Exclusive at end across midnight:
        assert!(!in_window(6 * 60, 22 * 60, 6 * 60));
    }

    #[test]
    fn in_window_empty_range_never_matches() {
        assert!(!in_window(0, 5 * 60, 5 * 60));
        assert!(!in_window(5 * 60, 5 * 60, 5 * 60));
    }

    #[test]
    fn save_and_load_round_trip() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("policy.toml");
        let p = TrainPolicy {
            enabled: true,
            last_train_ts: 1_700_000_000,
            ..Default::default()
        };
        save_at(&path, &p).unwrap();
        let back = load_at(&path).unwrap();
        assert!(back.enabled);
        assert_eq!(back.last_train_ts, 1_700_000_000);
        assert_eq!(back.base, "Qwen/Qwen3-7B");
    }

    #[test]
    fn load_returns_default_when_missing() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("nonexistent.toml");
        let p = load_at(&path).unwrap();
        assert!(!p.enabled);
        assert_eq!(p.threshold_new_turns, 500);
    }

    #[test]
    fn validate_rejects_bad_method() {
        let p = TrainPolicy {
            method: "rlhf".into(),
            ..Default::default()
        };
        assert!(validate(&p).is_err());
    }

    #[test]
    fn validate_rejects_bad_base() {
        let p = TrainPolicy {
            base: "no-slash".into(),
            ..Default::default()
        };
        assert!(validate(&p).is_err());
    }

    #[test]
    fn validate_rejects_oversize_window() {
        let p = TrainPolicy {
            since_window: "100y".into(),
            ..Default::default()
        };
        assert!(validate(&p).is_err());
    }

    #[test]
    fn validate_rejects_negative_threshold() {
        let p = TrainPolicy {
            threshold_new_turns: -1,
            ..Default::default()
        };
        assert!(validate(&p).is_err());
    }

    #[test]
    fn validate_accepts_default_policy() {
        validate(&TrainPolicy::default()).unwrap();
    }

    #[test]
    fn failure_backoff_skips_within_window() {
        let mut p = run_policy();
        p.last_train_ts = 0; // no cooldown
        p.consecutive_failures = 2; // backoff = 1h * 4 = 4h
        p.last_attempt_ts = 1_000_000;
        // 1h after the attempt — still inside the 4h backoff window.
        let now = p.last_attempt_ts + 3600;
        let d = decide(&p, now, at_3am(), 10_000, false);
        match d {
            Decision::Skip(r) => assert!(r.contains("failure backoff"), "{r}"),
            other => panic!("expected backoff skip, got {other:?}"),
        }
    }

    #[test]
    fn failure_backoff_clears_after_window() {
        let mut p = run_policy();
        p.last_train_ts = 0;
        p.consecutive_failures = 2; // 4h window
        p.last_attempt_ts = 1_000_000;
        // 5h later — past the 4h window → runs.
        let now = p.last_attempt_ts + 5 * 3600;
        assert!(matches!(
            decide(&p, now, at_3am(), 10_000, false),
            Decision::Run { .. }
        ));
    }

    #[test]
    fn backoff_capped_at_cooldown() {
        let mut p = run_policy();
        p.last_train_ts = 0;
        p.cooldown_days = 1; // cap = 24h
        p.consecutive_failures = 20; // raw 1h<<20 ≫ 24h → capped at 24h
        p.last_attempt_ts = 1_000_000;
        // 25h later — past the 24h cap → runs (not stuck forever).
        let now = p.last_attempt_ts + 25 * 3600;
        assert!(matches!(
            decide(&p, now, at_3am(), 10_000, false),
            Decision::Run { .. }
        ));
    }

    #[test]
    fn zero_failures_no_backoff() {
        let mut p = run_policy();
        p.last_train_ts = 0;
        p.consecutive_failures = 0;
        p.last_attempt_ts = 1_000_000;
        assert!(matches!(
            decide(&p, p.last_attempt_ts + 1, at_3am(), 10_000, false),
            Decision::Run { .. }
        ));
    }
}
