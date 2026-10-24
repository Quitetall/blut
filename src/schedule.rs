// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Declarative schedules via systemd `--user` timers (E5).
//!
//! BLUT owns the noun (recipe → schedule mapping); systemd owns the
//! mechanism (no daemon, fire-and-forget — same shape as the cron
//! suggestion the policy path already prints). `blut schedule install`
//! writes a `oneshot` service + a `Persistent` timer under
//! `~/.config/systemd/user/` and enables it; the recipe-run admission
//! gate + scheduler lock already make a timer firing during a live run
//! refuse/fail cleanly, so no queueing logic is needed here.

use std::path::PathBuf;
use std::process::Command;

use crate::error::{Result, TrainError};

const UNIT_PREFIX: &str = "blut-";

fn user_units_dir() -> Result<PathBuf> {
    let base = dirs::config_dir().ok_or_else(|| TrainError::other("cannot resolve ~/.config"))?;
    Ok(base.join("systemd").join("user"))
}

/// Sanitize a recipe name into a unit-safe slug (systemd unit names
/// allow `[A-Za-z0-9:_.\-]`).
fn unit_stem(recipe: &str) -> String {
    let slug: String = recipe
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{UNIT_PREFIX}{slug}")
}

/// Reject recipe names that are not unit-safe. Registry recipe names are
/// compile-time `[a-z_]` identifiers and always pass; enforcing the charset
/// here guarantees the unit *file name* (`unit_stem`) and the `ExecStart`
/// *command arg* never desync (a name like `a/b` would map both `a/b` and
/// `a_b` onto the same `blut-a_b.timer` while still running `recipe run a/b`),
/// and closes any systemd/shell metacharacter injection into `ExecStart`.
fn validate_recipe_name(recipe: &str) -> Result<()> {
    let ok = !recipe.is_empty()
        && recipe
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
    if ok {
        Ok(())
    } else {
        Err(TrainError::other(format!(
            "recipe name '{recipe}' is not unit-safe (allowed chars: A-Za-z0-9 . _ -)"
        )))
    }
}

/// Quote one word for a systemd `ExecStart=` line (systemd.syntax(7)), which
/// is not a shell: `%` starts a specifier and `$` an environment
/// substitution even inside quotes, so they are doubled; backslash and the
/// quote are C-escaped. The old shell-style `'\''` escaping left an exe path
/// with a space split in two and let `%`/`$` in recipe args be rewritten.
fn systemd_quote(word: &str) -> String {
    let escaped = word
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('%', "%%")
        .replace('$', "$$");
    format!("'{escaped}'")
}

/// The `oneshot` service a timer fires.
fn render_service(exe: &str, recipe: &str, args_json: &str) -> String {
    format!(
        "[Unit]\n\
         Description=BLUT scheduled recipe: {recipe}\n\n\
         [Service]\n\
         Type=oneshot\n\
         ExecStart={} recipe run {recipe} --args {}\n",
        systemd_quote(exe),
        systemd_quote(args_json),
    )
}

/// Validate an `OnCalendar` expression via `systemd-analyze calendar`.
/// Returns the normalized form on success.
pub fn validate_calendar(expr: &str) -> Result<()> {
    let out = Command::new("systemd-analyze")
        .arg("calendar")
        .arg(expr)
        .output()
        .map_err(|e| TrainError::other(format!("run systemd-analyze: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(TrainError::other(format!(
            "invalid OnCalendar '{expr}': {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )))
    }
}

/// Install (or replace) a timer that runs `blut recipe run <recipe>` on
/// `calendar`. `args_json` is the recipe args (defaults to `{}`).
pub fn install(recipe: &str, calendar: &str, args_json: &str) -> Result<()> {
    validate_recipe_name(recipe)?;
    validate_calendar(calendar)?;
    let exe = std::env::current_exe()
        .map_err(|e| TrainError::other(format!("locate own binary: {e}")))?;
    let dir = user_units_dir()?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| TrainError::other(format!("mkdir {}: {e}", dir.display())))?;
    let stem = unit_stem(recipe);

    let service = render_service(&exe.to_string_lossy(), recipe, args_json);
    let timer = format!(
        "[Unit]\n\
         Description=BLUT timer for {recipe}\n\n\
         [Timer]\n\
         OnCalendar={calendar}\n\
         Persistent=true\n\n\
         [Install]\n\
         WantedBy=timers.target\n"
    );
    write_and_enable(&dir, &stem, &service, &timer, systemctl)
}

/// Write the unit pair and enable the timer. If systemd refuses, remove the
/// files again: a failed install used to leave them behind, and `schedule
/// list` (which reads the unit directory) then reported a timer that was
/// never enabled.
fn write_and_enable(
    dir: &std::path::Path,
    stem: &str,
    service: &str,
    timer: &str,
    systemctl: impl Fn(&[&str]) -> Result<()>,
) -> Result<()> {
    let service_path = dir.join(format!("{stem}.service"));
    let timer_path = dir.join(format!("{stem}.timer"));
    let remove = || {
        let _ = std::fs::remove_file(&timer_path);
        let _ = std::fs::remove_file(&service_path);
    };
    std::fs::write(&service_path, service)
        .map_err(|e| TrainError::other(format!("write service: {e}")))?;
    if let Err(e) = std::fs::write(&timer_path, timer) {
        remove();
        return Err(TrainError::other(format!("write timer: {e}")));
    }
    let enabled = systemctl(&["daemon-reload"])
        .and_then(|()| systemctl(&["enable", "--now", &format!("{stem}.timer")]));
    if let Err(e) = enabled {
        remove();
        let _ = systemctl(&["daemon-reload"]);
        return Err(e);
    }
    Ok(())
}

/// Disable + remove a recipe's timer/service.
pub fn uninstall(recipe: &str) -> Result<()> {
    let stem = unit_stem(recipe);
    let dir = user_units_dir()?;
    // Uninstalling a timer that was never installed (a typo) used to report
    // "removed timer" and exit 0.
    if !dir.join(format!("{stem}.timer")).exists() && !dir.join(format!("{stem}.service")).exists()
    {
        return Err(TrainError::other(format!(
            "no blut timer is installed for recipe '{recipe}' (see `blut schedule list`)"
        )));
    }
    let _ = systemctl(&["disable", "--now", &format!("{stem}.timer")]);
    let _ = std::fs::remove_file(dir.join(format!("{stem}.timer")));
    let _ = std::fs::remove_file(dir.join(format!("{stem}.service")));
    systemctl(&["daemon-reload"])?;
    Ok(())
}

/// Installed blut timers: `(recipe, has_timer_file)`.
pub fn list() -> Result<Vec<String>> {
    let dir = user_units_dir()?;
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(stem) = name.strip_suffix(".timer")
                && let Some(recipe) = stem.strip_prefix(UNIT_PREFIX)
            {
                out.push(recipe.to_string());
            }
        }
    }
    out.sort();
    Ok(out)
}

fn systemctl(args: &[&str]) -> Result<()> {
    let out = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .output()
        .map_err(|e| TrainError::other(format!("run systemctl --user {args:?}: {e}")))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(TrainError::other(format!(
            "systemctl --user {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_enable_leaves_no_unit_files() {
        let td = tempfile::tempdir().unwrap();
        let refuse = |args: &[&str]| -> Result<()> {
            if args.first() == Some(&"enable") {
                Err(TrainError::other("Unit blut-x.timer does not exist"))
            } else {
                Ok(())
            }
        };
        assert!(write_and_enable(td.path(), "blut-x", "s", "t", refuse).is_err());
        assert!(!td.path().join("blut-x.timer").exists());
        assert!(!td.path().join("blut-x.service").exists());
        assert!(write_and_enable(td.path(), "blut-x", "s", "t", |_: &[&str]| Ok(())).is_ok());
        assert!(td.path().join("blut-x.timer").exists());
    }

    /// An exe path with a space was written unquoted (the unit ran
    /// `/opt/my`), and `%`/`$` in args were left for systemd to expand.
    #[test]
    fn exec_start_quotes_the_exe_and_escapes_specifiers() {
        let unit = render_service(
            "/opt/my tools/blut",
            "count",
            r#"{"note": "50%d $HOME it's"}"#,
        );
        let exec = unit
            .lines()
            .find(|l| l.starts_with("ExecStart="))
            .expect("ExecStart line");
        assert_eq!(
            exec,
            r#"ExecStart='/opt/my tools/blut' recipe run count --args '{"note": "50%%d $$HOME it\'s"}'"#
        );
    }

    #[test]
    fn unit_stem_sanitizes() {
        assert_eq!(
            unit_stem("lamquant_joint_codec"),
            "blut-lamquant_joint_codec"
        );
        assert_eq!(unit_stem("a/b c"), "blut-a_b_c");
    }

    #[test]
    fn validate_recipe_name_rejects_unsafe() {
        assert!(validate_recipe_name("lamquant_joint_codec").is_ok());
        assert!(validate_recipe_name("eval-v2.1").is_ok());
        assert!(validate_recipe_name("").is_err());
        assert!(validate_recipe_name("a/b").is_err());
        assert!(validate_recipe_name("foo; rm -rf /").is_err());
        assert!(validate_recipe_name("a b").is_err());
    }
}
