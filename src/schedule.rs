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
    let base = dirs::config_dir()
        .ok_or_else(|| TrainError::other("cannot resolve ~/.config"))?;
    Ok(base.join("systemd").join("user"))
}

/// Sanitize a recipe name into a unit-safe slug (systemd unit names
/// allow `[A-Za-z0-9:_.\-]`).
fn unit_stem(recipe: &str) -> String {
    let slug: String = recipe
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') { c } else { '_' })
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
    std::fs::create_dir_all(&dir).map_err(|e| TrainError::other(format!("mkdir {}: {e}", dir.display())))?;
    let stem = unit_stem(recipe);

    let service = format!(
        "[Unit]\n\
         Description=BLUT scheduled recipe: {recipe}\n\n\
         [Service]\n\
         Type=oneshot\n\
         ExecStart={exe} recipe run {recipe} --args '{args}'\n",
        exe = exe.display(),
        args = args_json.replace('\'', "'\\''"),
    );
    let timer = format!(
        "[Unit]\n\
         Description=BLUT timer for {recipe}\n\n\
         [Timer]\n\
         OnCalendar={calendar}\n\
         Persistent=true\n\n\
         [Install]\n\
         WantedBy=timers.target\n"
    );
    std::fs::write(dir.join(format!("{stem}.service")), service)
        .map_err(|e| TrainError::other(format!("write service: {e}")))?;
    std::fs::write(dir.join(format!("{stem}.timer")), timer)
        .map_err(|e| TrainError::other(format!("write timer: {e}")))?;

    systemctl(&["daemon-reload"])?;
    systemctl(&["enable", "--now", &format!("{stem}.timer")])?;
    Ok(())
}

/// Disable + remove a recipe's timer/service.
pub fn uninstall(recipe: &str) -> Result<()> {
    let stem = unit_stem(recipe);
    // Best-effort disable (ignore if not enabled).
    let _ = systemctl(&["disable", "--now", &format!("{stem}.timer")]);
    let dir = user_units_dir()?;
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
            if let Some(stem) = name.strip_suffix(".timer") {
                if let Some(recipe) = stem.strip_prefix(UNIT_PREFIX) {
                    out.push(recipe.to_string());
                }
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
    fn unit_stem_sanitizes() {
        assert_eq!(unit_stem("lamquant_joint_codec"), "blut-lamquant_joint_codec");
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
