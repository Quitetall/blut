// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Diagnostic / planning views migrated into the BLUT cockpit from the
//! retired Python training cockpit (`legacy/python_cockpit/cockpit.py`)
//! and the lamquant-lossless hub `CockpitPanel`.
//!
//! Each function here is a pure data-gatherer: it scans the repo's
//! `training_logs/`, `checkpoints/`, `weights/`, and `runs/` trees and
//! returns rows the TUI renders. No side effects (the destructive Reset
//! actions live in [`reset`]). Best-effort: a missing dir / unparseable
//! CSV degrades to an empty result, never a panic.
//!
//! Parity map (Python `_screen_*` → BLUT view):
//!   * `_screen_history`      → [`run_history`]    (View::History)
//!   * `_screen_leaderboard`  → [`leaderboard`]    (View::Leaderboard)
//!   * `_screen_compare`      → [`compare`]        (View::Compare)
//!   * `_screen_checkpoints`  → [`checkpoints`]    (View::Checkpoints)
//!   * `_screen_presets`      → [`PRESETS`]        (View::Presets)
//!   * `_screen_hparams`      → [`HPARAM_GROUPS`]  (View::Presets, lower pane)
//!   * `_screen_reset`        → [`reset`]          (View::Reset)
//!   * `_screen_live_metrics` → [`metrics_tail`]   (View::Metrics)
//!   * `_screen_export`       → [`export_presets`] (View::Reset, [e])

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

/// One training-log run summary (parsed from a `training_logs/*.csv`).
#[derive(Clone, Debug)]
pub struct RunRow {
    pub name: String,
    pub best_r: f64,
    pub best_ep: usize,
    pub total_ep: usize,
    pub final_r: f64,
    pub date: String,
}

/// One checkpoint file (`.ckpt`) discovered under `checkpoints/` or
/// `weights/`.
#[derive(Clone, Debug)]
pub struct CkptRow {
    pub name: String,
    pub rel_dir: String,
    pub size_mb: f64,
    pub mtime: SystemTime,
    pub date: String,
}

/// Repo root used to resolve `training_logs/`, `checkpoints/`, etc.
/// Honours `$LAMQUANT_HOME`, else falls back to the current working
/// directory (the cockpit is normally launched from the repo root).
pub fn repo_root() -> PathBuf {
    std::env::var_os("LAMQUANT_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn fmt_date(t: SystemTime, with_time: bool) -> String {
    // Convert to a Y-M-D[ H:M] string without pulling in chrono. Reuse
    // the same civil-from-days algorithm jobs.rs uses for ids.
    let secs = t
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d, h, mi, _s) = crate::jobs::unix_to_ymdhms_pub(secs);
    if with_time {
        format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}")
    } else {
        format!("{y:04}-{m:02}-{d:02}")
    }
}

/// Strip the legacy `alpha_trajectory_` log-name prefix (Python cockpit
/// `_find_logs` / `_screen_history` did the same).
fn clean_name(stem: &str) -> String {
    stem.strip_prefix("alpha_trajectory_")
        .unwrap_or(stem)
        .to_string()
}

/// `training_logs/*.csv`, newest first, up to 20 (Python `_find_logs`).
fn find_logs(root: &Path) -> Vec<PathBuf> {
    let dir = root.join("training_logs");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut logs: Vec<(PathBuf, SystemTime)> = rd
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("csv"))
                .unwrap_or(false)
        })
        .map(|e| {
            let mt = e
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            (e.path(), mt)
        })
        .collect();
    logs.sort_by_key(|b| std::cmp::Reverse(b.1));
    logs.truncate(20);
    logs.into_iter().map(|(p, _)| p).collect()
}

/// Parse one training CSV → (best_r, best_ep, total_ep, final_r).
/// Reads the `val_r` column, falling back to `R` (Python parity).
fn parse_log_csv(path: &Path) -> (f64, usize, usize, f64) {
    let Ok(body) = std::fs::read_to_string(path) else {
        return (0.0, 0, 0, 0.0);
    };
    let mut lines = body.lines();
    let Some(header) = lines.next() else {
        return (0.0, 0, 0, 0.0);
    };
    let cols: Vec<&str> = header.split(',').map(|c| c.trim()).collect();
    let col_idx = cols
        .iter()
        .position(|c| *c == "val_r")
        .or_else(|| cols.iter().position(|c| *c == "R"));
    let Some(col_idx) = col_idx else {
        // No R column — count rows for total_ep, leave R metrics 0.
        let total = lines.filter(|l| !l.trim().is_empty()).count();
        return (0.0, 0, total, 0.0);
    };
    let mut best_r = 0.0_f64;
    let mut best_ep = 0_usize;
    let mut total = 0_usize;
    let mut final_r = 0.0_f64;
    for row in lines {
        if row.trim().is_empty() {
            continue;
        }
        total += 1;
        if let Some(v) = row
            .split(',')
            .nth(col_idx)
            .and_then(|s| s.trim().parse::<f64>().ok())
        {
            final_r = v;
            if v > best_r {
                best_r = v;
                best_ep = total;
            }
        }
    }
    (best_r, best_ep, total, final_r)
}

/// `_screen_history` + `_screen_leaderboard` data source. Returns one
/// [`RunRow`] per training-log CSV, newest first (caller re-sorts for
/// the leaderboard).
pub fn run_history(root: &Path) -> Vec<RunRow> {
    find_logs(root)
        .into_iter()
        .map(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            let name = clean_name(stem);
            let mt = p
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            let (best_r, best_ep, total_ep, final_r) = parse_log_csv(&p);
            RunRow {
                name,
                best_r,
                best_ep,
                total_ep,
                final_r,
                date: fmt_date(mt, false),
            }
        })
        .collect()
}

/// `_screen_leaderboard` — runs ranked by best validation R descending.
pub fn leaderboard(root: &Path) -> Vec<RunRow> {
    let mut rows = run_history(root);
    rows.sort_by(|a, b| {
        b.best_r
            .partial_cmp(&a.best_r)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows
}

/// `_find_checkpoints` — all `.ckpt` under `checkpoints/` + `weights/`
/// (recursive), newest first.
pub fn checkpoints(root: &Path) -> Vec<CkptRow> {
    let mut out: Vec<CkptRow> = Vec::new();
    for sub in ["checkpoints", "weights"] {
        let base = root.join(sub);
        collect_ckpts(&base, root, &mut out, 0);
    }
    out.sort_by_key(|b| std::cmp::Reverse(b.mtime));
    out
}

/// Bounded recursive `.ckpt` walk (Programming Bible Rule 19: bound
/// recursion). Depth 8 is far beyond any sane checkpoint layout.
fn collect_ckpts(dir: &Path, root: &Path, out: &mut Vec<CkptRow>, depth: u32) {
    const MAX_DEPTH: u32 = 8;
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            collect_ckpts(&p, root, out, depth + 1);
        } else if p
            .extension()
            .and_then(|x| x.to_str())
            .map(|x| x.eq_ignore_ascii_case("ckpt"))
            .unwrap_or(false)
        {
            let meta = e.metadata().ok();
            let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            let mtime = meta
                .and_then(|m| m.modified().ok())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            let rel_dir = p
                .parent()
                .and_then(|par| par.strip_prefix(root).ok())
                .map(|r| r.display().to_string())
                .unwrap_or_else(|| {
                    p.parent()
                        .map(|par| par.display().to_string())
                        .unwrap_or_default()
                });
            out.push(CkptRow {
                name: p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string(),
                rel_dir,
                size_mb: size as f64 / 1e6,
                mtime,
                date: fmt_date(mtime, true),
            });
        }
    }
}

/// Read-only preset catalog — `_screen_presets`. (preset, epochs,
/// windows-per-epoch, wall-clock, use-case).
pub const PRESETS: &[(&str, &str, &str, &str, &str)] = &[
    (
        "fast",
        "5+15 ep",
        "16K wpe",
        "~3 min",
        "smoke test — verify the pipeline runs end-to-end",
    ),
    (
        "medium",
        "10+190 ep",
        "200K wpe",
        "~3 h",
        "research iteration — enough signal for A/B decisions",
    ),
    (
        "production",
        "20+380 ep",
        "400K wpe",
        "~22 h",
        "full run — the checkpoint you ship / gate through PCCP",
    ),
];

/// `_screen_presets` "Production validated features" list.
pub const VALIDATED_FEATURES: &[&str] = &[
    "SOAP optimizer (+0.0135 R over AdamW)",
    "V2-fixed SE decoder (+0.0104 R)",
    "ParetoQ ternary quantization",
    "GAN + clinical sampling",
    "WSD infinite-LR schedule",
];

/// Decoder-tier catalog from `_screen_start` (tier, params, note).
pub const DECODER_TIERS: &[(&str, &str, &str)] = &[
    (
        "Tier 3",
        "3.8M",
        "baseline — fits in 24 GB with torch.compile",
    ),
    (
        "Tier 7",
        "844M",
        "full production — needs grad checkpointing, no compile",
    ),
];

/// `_screen_hparams` category groups (label, fields). View-only — the
/// real values live in the recipe Args JSON now (ADR 0017), so this is
/// the reference catalog of *what* each preset controls.
pub const HPARAM_GROUPS: &[(&str, &[&str])] = &[
    ("Epochs", &["warmup", "quant", "fine"]),
    ("Batch sizes", &["warmup", "quant", "fine"]),
    ("Learning rate", &["warmup", "quant", "quant_min", "fine"]),
    ("Loss weights", &["pearson_r", "spectral", "prd"]),
    (
        "Quantization",
        &[
            "alpha_clamp",
            "ceiling",
            "floor",
            "activation_bits",
            "deadzone_tau(init/final)",
        ],
    ),
    (
        "Data",
        &[
            "windows_per_epoch",
            "max_windows",
            "val_interval",
            "val_windows",
        ],
    ),
    (
        "Architecture",
        &["vocos_tier", "latent_dim", "encoder_width"],
    ),
];

/// `_screen_live_metrics` terminal-tail equivalent: tail the newest
/// `training_logs/*.csv` (or a chosen run) so the user sees live R
/// progression without leaving the TUI. Returns the last `n` lines.
pub fn metrics_tail(root: &Path, n: usize) -> Vec<String> {
    let logs = find_logs(root);
    let Some(latest) = logs.first() else {
        return vec!["(no training_logs/*.csv yet — start a run first)".into()];
    };
    let Ok(body) = std::fs::read_to_string(latest) else {
        return vec![format!("(could not read {})", latest.display())];
    };
    let mut lines: Vec<String> = vec![format!(
        "# tailing {}",
        latest.file_name().and_then(|s| s.to_str()).unwrap_or("?")
    )];
    let all: Vec<&str> = body.lines().collect();
    let start = all.len().saturating_sub(n);
    lines.extend(all[start..].iter().map(|s| s.to_string()));
    lines
}

/// One destructive maintenance action from `_screen_reset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResetAction {
    /// Kill training-related tmux sessions.
    KillTmux,
    /// Remove `.numba_cache/` to force JIT recompile.
    ClearNumba,
    /// Delete `training_logs/*.csv`.
    ClearLogs,
}

impl ResetAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::KillTmux => "Kill stale training tmux sessions",
            Self::ClearNumba => "Clear numba JIT cache (.numba_cache/)",
            Self::ClearLogs => "Clear training logs (training_logs/*.csv)",
        }
    }
}

/// `_screen_reset` action runner. Returns a human-readable result line.
/// Each action is independently guarded by the two-press confirm in the
/// TUI; this only fires once confirmed.
pub fn reset(root: &Path, action: ResetAction) -> String {
    match action {
        ResetAction::KillTmux => {
            // Find training-related sessions, mirror the Python keyword
            // filter.
            let out = Command::new("tmux").arg("ls").output();
            let Ok(out) = out else {
                return "tmux not installed; nothing killed".into();
            };
            if !out.status.success() {
                return "no tmux server running; nothing to kill".into();
            }
            let text = String::from_utf8_lossy(&out.stdout);
            let kws = [
                "teacher",
                "student",
                "decoder",
                "snn",
                "train",
                "production",
                "medium",
                "fast",
                "joint",
                "lamquant",
                "oracle",
                "encoder",
            ];
            let sessions: Vec<String> = text
                .lines()
                .filter_map(|l| l.split(':').next().map(|s| s.to_string()))
                .filter(|s| {
                    let lc = s.to_lowercase();
                    kws.iter().any(|k| lc.contains(k))
                })
                .collect();
            if sessions.is_empty() {
                return "no training tmux sessions to kill".into();
            }
            let mut killed = 0;
            for s in &sessions {
                if Command::new("tmux")
                    .args(["kill-session", "-t", s])
                    .status()
                    .map(|st| st.success())
                    .unwrap_or(false)
                {
                    killed += 1;
                }
            }
            format!(
                "killed {killed}/{} training tmux session(s)",
                sessions.len()
            )
        }
        ResetAction::ClearNumba => {
            let cache = root.join(".numba_cache");
            if !cache.exists() {
                return "no .numba_cache/ found; nothing to clear".into();
            }
            match std::fs::remove_dir_all(&cache) {
                Ok(()) => "numba JIT cache cleared (will recompile next run)".into(),
                Err(e) => format!("failed to clear .numba_cache/: {e}"),
            }
        }
        ResetAction::ClearLogs => {
            let dir = root.join("training_logs");
            let Ok(rd) = std::fs::read_dir(&dir) else {
                return "no training_logs/ dir; nothing to clear".into();
            };
            let mut removed = 0;
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("csv")
                    && std::fs::remove_file(&p).is_ok()
                {
                    removed += 1;
                }
            }
            format!("deleted {removed} training log(s) from training_logs/")
        }
    }
}

/// `_screen_export` — write each recipe's args schema to a JSON file at
/// the repo root for reproducible setup capture. Since the recipe Args
/// are the source of truth now (ADR 0017), we export the recipe-args
/// JSON schema templates for every recipe in the injected catalog
/// (domain-agnostic: the catalog is composed from the registered
/// cookbooks, so this covers whatever recipes are loaded).
pub fn export_presets(
    root: &Path,
    catalog: &[&'static crate::recipes::recipe::RecipeDef],
) -> String {
    let mut written = 0;
    let mut errs = 0;
    for r in catalog {
        let name = r.name;
        let schema = (r.args_schema_fn)();
        let out_path = root.join(format!("recipe_args_{name}.json"));
        match serde_json::to_string_pretty(&schema)
            .map_err(|e| e.to_string())
            .and_then(|body| std::fs::write(&out_path, body).map_err(|e| e.to_string()))
        {
            Ok(()) => written += 1,
            Err(_) => errs += 1,
        }
    }
    if errs == 0 {
        format!(
            "exported {written} recipe args schema(s) to {}",
            root.display()
        )
    } else {
        format!("exported {written}, {errs} failed (see {})", root.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_log(dir: &Path, name: &str, rows: &[(usize, f64)]) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        writeln!(f, "epoch,val_r,loss").unwrap();
        for (ep, r) in rows {
            writeln!(f, "{ep},{r},0.1").unwrap();
        }
        p
    }

    #[test]
    fn parse_log_picks_best_and_final() {
        let td = tempfile::tempdir().unwrap();
        let p = write_log(
            td.path(),
            "alpha_trajectory_run1.csv",
            &[(1, 0.80), (2, 0.92), (3, 0.88)],
        );
        let (best, best_ep, total, final_r) = parse_log_csv(&p);
        assert!((best - 0.92).abs() < 1e-9, "best={best}");
        assert_eq!(best_ep, 2);
        assert_eq!(total, 3);
        assert!((final_r - 0.88).abs() < 1e-9, "final={final_r}");
    }

    #[test]
    fn history_strips_prefix() {
        let td = tempfile::tempdir().unwrap();
        let logs = td.path().join("training_logs");
        write_log(&logs, "alpha_trajectory_v1.csv", &[(1, 0.5)]);
        let rows = run_history(td.path());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "v1");
    }

    #[test]
    fn leaderboard_sorts_desc() {
        let td = tempfile::tempdir().unwrap();
        let logs = td.path().join("training_logs");
        write_log(&logs, "low.csv", &[(1, 0.4)]);
        write_log(&logs, "high.csv", &[(1, 0.95)]);
        let lb = leaderboard(td.path());
        assert_eq!(lb.len(), 2);
        assert_eq!(lb[0].name, "high");
        assert!(lb[0].best_r > lb[1].best_r);
    }

    #[test]
    fn checkpoints_finds_nested_ckpt() {
        let td = tempfile::tempdir().unwrap();
        let nested = td.path().join("checkpoints").join("run1");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("epoch_1.ckpt"), b"x").unwrap();
        std::fs::write(nested.join("notes.txt"), b"y").unwrap();
        let cks = checkpoints(td.path());
        assert_eq!(cks.len(), 1);
        assert_eq!(cks[0].name, "epoch_1.ckpt");
    }

    #[test]
    fn metrics_tail_handles_empty() {
        let td = tempfile::tempdir().unwrap();
        let t = metrics_tail(td.path(), 50);
        assert_eq!(t.len(), 1);
        assert!(t[0].contains("no training_logs"));
    }

    #[test]
    fn reset_clear_numba_missing_is_noop() {
        let td = tempfile::tempdir().unwrap();
        let msg = reset(td.path(), ResetAction::ClearNumba);
        assert!(msg.contains("nothing to clear"), "msg={msg}");
    }
}
