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

    /// UNIX seconds. Updated atomically after a successful spawn.
    /// Zero on first run.
    #[serde(default)]
    pub last_train_ts: i64,

    /// Number of turns the last training run consumed. Diagnostic.
    #[serde(default)]
    pub last_train_n_turns: i64,
}

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
        .ok_or_else(|| TrainError::other(
            "config_dir() unavailable; set $LAMU_TRAIN_POLICY",
        ))?
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
    toml::from_str(&body)
        .map_err(|e| TrainError::other(format!("parse {}: {e}", path.display())))
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
    let tmp = path.with_file_name(format!(
        ".{stem}.tmp.{}.{nanos}",
        std::process::id()
    ));
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
        return Decision::Skip("auto-trigger disabled (run `lamu-train policy enable` to opt in)".into());
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
    if new_turns_since_last < policy.threshold_new_turns {
        return Decision::Skip(format!(
            "only {} new turns since last train; threshold is {}",
            new_turns_since_last, policy.threshold_new_turns
        ));
    }
    if inference_lock_held {
        return Decision::Skip(
            "GPU held by inference; will retry next tick".into(),
        );
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
    parse_quiet_hours(&policy.quiet_hours).map_err(|e| TrainError::other(e))?;
    let since = humantime::parse_duration(&policy.since_window)
        .map_err(|e| TrainError::other(format!("since_window '{}': {e}", policy.since_window)))?;
    if since.as_secs() > MAX_SINCE_SECS {
        return Err(TrainError::other(format!(
            "since_window '{}' exceeds 10-year cap",
            policy.since_window
        )));
    }
    if policy.threshold_new_turns < 0 {
        return Err(TrainError::other(
            "threshold_new_turns must be >= 0",
        ));
    }
    Ok(())
}

/// Helper for the production `auto` CLI: returns now() in UNIX
/// seconds + local minutes-of-day. Pure side-effect-free wrapper
/// so the decision function can be tested with synthetic clocks.
pub fn current_clock() -> (i64, u32) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs_unix = now.as_secs() as i64;
    // Local minutes-of-day. We can't get a robust local-tz
    // conversion without `chrono::Local` and we don't want the
    // chrono dep here. Approximation: use UTC. Users can override
    // quiet_hours to UTC values explicitly.
    //
    // TODO: when chrono lands as a workspace dep, swap this for
    // chrono::Local::now().num_seconds_from_midnight() / 60.
    let secs_into_day = (secs_unix.rem_euclid(86400)) as u32;
    let minutes = secs_into_day / 60;
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
            Decision::Run { base, method, since } => {
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
        let mut p = TrainPolicy::default();
        p.enabled = true;
        p.last_train_ts = 1_700_000_000;
        save_at(&path, &p).unwrap();
        let back = load_at(&path).unwrap();
        assert_eq!(back.enabled, true);
        assert_eq!(back.last_train_ts, 1_700_000_000);
        assert_eq!(back.base, "Qwen/Qwen3-7B");
    }

    #[test]
    fn load_returns_default_when_missing() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("nonexistent.toml");
        let p = load_at(&path).unwrap();
        assert_eq!(p.enabled, false);
        assert_eq!(p.threshold_new_turns, 500);
    }

    #[test]
    fn validate_rejects_bad_method() {
        let mut p = TrainPolicy::default();
        p.method = "rlhf".into();
        assert!(validate(&p).is_err());
    }

    #[test]
    fn validate_rejects_bad_base() {
        let mut p = TrainPolicy::default();
        p.base = "no-slash".into();
        assert!(validate(&p).is_err());
    }

    #[test]
    fn validate_rejects_oversize_window() {
        let mut p = TrainPolicy::default();
        p.since_window = "100y".into();
        assert!(validate(&p).is_err());
    }

    #[test]
    fn validate_rejects_negative_threshold() {
        let mut p = TrainPolicy::default();
        p.threshold_new_turns = -1;
        assert!(validate(&p).is_err());
    }

    #[test]
    fn validate_accepts_default_policy() {
        validate(&TrainPolicy::default()).unwrap();
    }
}
