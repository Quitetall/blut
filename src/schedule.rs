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
//!
//! A timer fires in the user manager's environment, not the installer's
//! shell, so the service carries what the interactive `recipe run` would
//! have had ([`install_with`]):
//!
//! - the launch flags: `--tenant` (always written, so the unit names the
//!   tenant it was installed for), and `--shared-cache`, `--no-cache`,
//!   `--sync-io` and `--experiment` when given ([`ScheduleOptions`]);
//! - `WorkingDirectory=` set to the installer's working directory, which
//!   relative paths in the args and the run's recorded git commit resolve
//!   against;
//! - `Environment=` for each variable in [`ENVIRONMENT_ALLOWLIST`] that is set
//!   in the installer's environment: where state lives, the remote cache
//!   location, the policy files the launch is admitted under, the
//!   interpreter, and device and containment selection. Nothing else is
//!   copied. Credentials (`AWS_*`, API keys, tokens) are never written to a
//!   unit file; a schedule that needs them gets them from the user manager,
//!   for example an `EnvironmentFile=` drop-in the operator controls.

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

/// Environment variables a scheduled unit takes from the installer, by name.
/// Each names a location, a policy file, an interpreter or a device/containment
/// choice that decides what the run does and where its state goes; none is a
/// credential. A variable outside this list is never written to a unit file.
pub const ENVIRONMENT_ALLOWLIST: &[&str] = &[
    // Where state lives.
    "LAMU_TRAIN_DATA_DIR",
    "LAMU_TRAIN_JOBS_DIR",
    "LAMU_TRAIN_CACHE_DIR",
    "LAMU_CACHE_MAX_GB",
    "LAMU_MEMORY_DB",
    "LAMU_REGISTRY_PATH",
    "BLUT_CHECKPOINTS_DIR",
    "BLUT_REGISTRY_DB",
    "BLUT_REGISTRY_PATH",
    "BLUT_CATALOG_DB",
    "BLUT_FOOTPRINT_PATH",
    "BLUT_PARTITIONS_DIR",
    "BLUT_EVENTS_DIR",
    "BLUT_COST_LEDGER",
    "BLUT_HOME",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    // The cache's remote tier: a directory or `s3://bucket[/prefix]`. Its
    // credentials come from `AWS_*`, which is never copied; a value that
    // carries credentials itself is refused.
    "BLUT_REMOTE_CACHE",
    // The policy the launch is admitted under.
    "BLUT_TENANTS_CONFIG",
    "LAMU_TRAIN_POLICY",
    // What runs the stages, and on which devices.
    "LAMU_TRAIN_PYTHON",
    "LAMU_TRAINER_PY",
    "BLUT_CONTAINMENT",
    "BLUT_NO_CONTAIN",
    "BLUT_CGROUP_PARENT",
    "BLUT_SCHED_DEVICES",
    "BLUT_GPU_INVENTORY",
    "CUDA_VISIBLE_DEVICES",
];

/// The `recipe run` flags a scheduled unit passes, mirroring the interactive
/// launch. `Default` is a plain `recipe run` under the `default` tenant.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ScheduleOptions {
    /// `--tenant`; `None` is `default`. Validated with
    /// [`crate::tenant::Tenant::parse`] and written in canonical form.
    pub tenant: Option<String>,
    /// `--shared-cache`.
    pub shared_cache: bool,
    /// `--no-cache`.
    pub no_cache: bool,
    /// `--sync-io`.
    pub sync_io: bool,
    /// `--experiment`.
    pub experiment: Option<String>,
}

/// Everything a scheduled service needs, already validated.
struct ServiceSpec<'a> {
    exe: &'a str,
    recipe: &'a str,
    args_json: &'a str,
    tenant: String,
    options: &'a ScheduleOptions,
    working_dir: Option<&'a std::path::Path>,
    environment: &'a [(String, String)],
}

/// Refuse a value systemd would read as more than one line or unit syntax.
fn reject_control(what: &str, value: &str) -> Result<()> {
    if value.chars().any(char::is_control) {
        return Err(TrainError::other(format!(
            "{what} contains a control character; it cannot be written to a unit file"
        )));
    }
    Ok(())
}

/// One `Environment=` line (systemd.exec(5)): the assignment is double-quoted,
/// with `\` and `"` C-escaped and `%` doubled (specifiers expand there; `$`
/// does not).
fn environment_line(name: &str, value: &str) -> Result<String> {
    reject_control(&format!("${name}"), value)?;
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('%', "%%");
    Ok(format!("Environment=\"{name}={escaped}\"\n"))
}

/// `BLUT_REMOTE_CACHE` may name a URL. One carrying userinfo, a query or a
/// fragment could hold a credential, and the run would refuse it anyway, so it
/// is not written to a unit file.
fn check_remote_cache_value(value: &str) -> Result<()> {
    let Some((_, rest)) = value.split_once("://") else {
        return Ok(()); // a directory
    };
    let authority = rest.split('/').next().unwrap_or("");
    if authority.contains('@') || value.contains('?') || value.contains('#') {
        return Err(TrainError::other(
            "BLUT_REMOTE_CACHE carries userinfo, a query or a fragment; schedule install \
             will not write it to a unit file (credentials come from AWS_*, which a unit \
             gets from the user manager, never from blut)",
        ));
    }
    Ok(())
}

/// The allow-listed variables set in the installer's environment, by `lookup`,
/// in allow-list order. A value that is not UTF-8 is refused, naming the
/// variable, as the engine refuses such path overrides at run time.
fn captured_environment(
    lookup: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for name in ENVIRONMENT_ALLOWLIST {
        let Some(value) = lookup(name) else {
            continue;
        };
        let value = value.into_string().map_err(|_| {
            TrainError::other(format!(
                "${name} is not valid UTF-8; it cannot be written to a unit file"
            ))
        })?;
        if *name == "BLUT_REMOTE_CACHE" {
            check_remote_cache_value(&value)?;
        }
        out.push(((*name).to_string(), value));
    }
    Ok(out)
}

/// The `oneshot` service a timer fires.
fn render_service(spec: &ServiceSpec<'_>) -> Result<String> {
    let mut unit = format!(
        "[Unit]\n\
         Description=BLUT scheduled recipe: {}\n\n\
         [Service]\n\
         Type=oneshot\n",
        spec.recipe
    );
    if let Some(dir) = spec.working_dir {
        let dir = dir.to_str().ok_or_else(|| {
            TrainError::other(
                "the working directory is not valid UTF-8; it cannot be written to a unit file",
            )
        })?;
        reject_control("the working directory", dir)?;
        if !dir.starts_with('/') {
            return Err(TrainError::other(format!(
                "the working directory {dir:?} is not absolute"
            )));
        }
        unit.push_str(&format!("WorkingDirectory={}\n", dir.replace('%', "%%")));
    }
    for (name, value) in spec.environment {
        unit.push_str(&environment_line(name, value)?);
    }
    let mut exec = format!(
        "ExecStart={} recipe run {} --args {} --tenant {}",
        systemd_quote(spec.exe),
        spec.recipe,
        systemd_quote(spec.args_json),
        systemd_quote(&spec.tenant),
    );
    for (on, flag) in [
        (spec.options.shared_cache, " --shared-cache"),
        (spec.options.no_cache, " --no-cache"),
        (spec.options.sync_io, " --sync-io"),
    ] {
        if on {
            exec.push_str(flag);
        }
    }
    if let Some(experiment) = &spec.options.experiment {
        exec.push_str(&format!(" --experiment {}", systemd_quote(experiment)));
    }
    unit.push_str(&exec);
    unit.push('\n');
    Ok(unit)
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
/// `calendar`. `args_json` is the recipe args (defaults to `{}`). The same as
/// [`install_with`] with default [`ScheduleOptions`].
pub fn install(recipe: &str, calendar: &str, args_json: &str) -> Result<()> {
    install_with(recipe, calendar, args_json, &ScheduleOptions::default()).map(|_| ())
}

/// Install (or replace) a timer that runs `blut recipe run <recipe>` on
/// `calendar` with `options`' flags, from the installer's working directory
/// and with its [`ENVIRONMENT_ALLOWLIST`] variables (see the module docs).
/// Returns the names of the variables written to the unit.
pub fn install_with(
    recipe: &str,
    calendar: &str,
    args_json: &str,
    options: &ScheduleOptions,
) -> Result<Vec<String>> {
    validate_recipe_name(recipe)?;
    let tenant = match options.tenant.as_deref() {
        None => crate::tenant::Tenant::default(),
        Some(raw) => crate::tenant::Tenant::parse(raw)
            .ok_or_else(|| TrainError::other(format!("invalid tenant '{raw}'")))?,
    };
    if let Some(experiment) = &options.experiment
        && !crate::experiment_registry::is_safe_experiment_name(experiment)
    {
        return Err(TrainError::other(format!(
            "invalid experiment '{experiment}' (expected [A-Za-z0-9_.-]+)"
        )));
    }
    validate_calendar(calendar)?;
    let exe = std::env::current_exe()
        .map_err(|e| TrainError::other(format!("locate own binary: {e}")))?;
    let exe = exe.to_str().ok_or_else(|| {
        TrainError::other(
            "the blut binary's path is not valid UTF-8; it cannot be written to a unit file",
        )
    })?;
    let working_dir = std::env::current_dir()
        .map_err(|e| TrainError::other(format!("read the working directory: {e}")))?;
    let environment = captured_environment(|name| std::env::var_os(name))?;
    let service = render_service(&ServiceSpec {
        exe,
        recipe,
        args_json,
        tenant: tenant.to_string(),
        options,
        working_dir: Some(&working_dir),
        environment: &environment,
    })?;
    let dir = user_units_dir()?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| TrainError::other(format!("mkdir {}: {e}", dir.display())))?;
    let stem = unit_stem(recipe);

    let timer = format!(
        "[Unit]\n\
         Description=BLUT timer for {recipe}\n\n\
         [Timer]\n\
         OnCalendar={calendar}\n\
         Persistent=true\n\n\
         [Install]\n\
         WantedBy=timers.target\n"
    );
    write_and_enable(&dir, &stem, &service, &timer, systemctl)?;
    Ok(environment.into_iter().map(|(name, _)| name).collect())
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

    fn spec<'a>(
        exe: &'a str,
        args_json: &'a str,
        options: &'a ScheduleOptions,
        environment: &'a [(String, String)],
    ) -> ServiceSpec<'a> {
        ServiceSpec {
            exe,
            recipe: "count",
            args_json,
            tenant: "default".into(),
            options,
            working_dir: None,
            environment,
        }
    }

    /// An exe path with a space was written unquoted (the unit ran
    /// `/opt/my`), and `%`/`$` in args were left for systemd to expand.
    #[test]
    fn exec_start_quotes_the_exe_and_escapes_specifiers() {
        let options = ScheduleOptions::default();
        let unit = render_service(&spec(
            "/opt/my tools/blut",
            r#"{"note": "50%d $HOME it's"}"#,
            &options,
            &[],
        ))
        .unwrap();
        let exec = unit
            .lines()
            .find(|l| l.starts_with("ExecStart="))
            .expect("ExecStart line");
        assert_eq!(
            exec,
            r#"ExecStart='/opt/my tools/blut' recipe run count --args '{"note": "50%%d $$HOME it\'s"}' --tenant 'default'"#
        );
    }

    /// A scheduled unit carried no tenant, cache flags, working directory or
    /// environment, so a timer ran a different launch from the interactive
    /// one: another tenant, a private cache, the default data and jobs
    /// directories. It carries all of them, and only allow-listed variables.
    #[test]
    fn a_scheduled_unit_reproduces_the_interactive_launch() {
        let installer_env: std::collections::HashMap<&str, &str> = [
            ("LAMU_TRAIN_DATA_DIR", "/srv/data"),
            ("LAMU_TRAIN_JOBS_DIR", "/srv/jobs 2"),
            ("BLUT_CHECKPOINTS_DIR", "/srv/ckpt"),
            ("BLUT_REGISTRY_DB", "/srv/registry.db"),
            ("BLUT_CATALOG_DB", "/srv/catalog.db"),
            ("BLUT_REMOTE_CACHE", "s3://bucket/runs"),
            ("CUDA_VISIBLE_DEVICES", ""),
            // Never copied: credentials and anything not allow-listed.
            ("AWS_SECRET_ACCESS_KEY", "do-not-copy"),
            ("AWS_ACCESS_KEY_ID", "do-not-copy"),
            ("HF_TOKEN", "do-not-copy"),
            ("OPENAI_API_KEY", "do-not-copy"),
            ("BLUT_WEB_TOKEN", "do-not-copy"),
            ("PATH", "/do/not/copy"),
        ]
        .into_iter()
        .collect();
        let environment =
            captured_environment(|name| installer_env.get(name).map(|v| (*v).into())).unwrap();
        let names: Vec<&str> = environment.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "LAMU_TRAIN_DATA_DIR",
                "LAMU_TRAIN_JOBS_DIR",
                "BLUT_CHECKPOINTS_DIR",
                "BLUT_REGISTRY_DB",
                "BLUT_CATALOG_DB",
                "BLUT_REMOTE_CACHE",
                "CUDA_VISIBLE_DEVICES",
            ]
        );
        let options = ScheduleOptions {
            tenant: Some("lab/vision".into()),
            shared_cache: true,
            no_cache: false,
            sync_io: true,
            experiment: Some("nightly".into()),
        };
        let mut service = spec("/opt/blut", "{}", &options, &environment);
        service.tenant = "lab/vision".into();
        service.working_dir = Some(std::path::Path::new("/home/u/repo 50%"));
        let unit = render_service(&service).unwrap();
        assert!(
            !unit.contains("do-not-copy") && !unit.contains("/do/not/copy"),
            "{unit}"
        );
        let body: Vec<&str> = unit.lines().skip_while(|l| *l != "[Service]").collect();
        assert_eq!(
            body,
            [
                "[Service]",
                "Type=oneshot",
                "WorkingDirectory=/home/u/repo 50%%",
                "Environment=\"LAMU_TRAIN_DATA_DIR=/srv/data\"",
                "Environment=\"LAMU_TRAIN_JOBS_DIR=/srv/jobs 2\"",
                "Environment=\"BLUT_CHECKPOINTS_DIR=/srv/ckpt\"",
                "Environment=\"BLUT_REGISTRY_DB=/srv/registry.db\"",
                "Environment=\"BLUT_CATALOG_DB=/srv/catalog.db\"",
                "Environment=\"BLUT_REMOTE_CACHE=s3://bucket/runs\"",
                "Environment=\"CUDA_VISIBLE_DEVICES=\"",
                "ExecStart='/opt/blut' recipe run count --args '{}' --tenant 'lab/vision' \
                 --shared-cache --sync-io --experiment 'nightly'",
            ]
        );
    }

    #[test]
    fn environment_values_are_escaped_or_refused() {
        assert_eq!(
            environment_line("BLUT_HOME", r#"/a "b" \c 50% $HOME"#).unwrap(),
            "Environment=\"BLUT_HOME=/a \\\"b\\\" \\\\c 50%% $HOME\"\n"
        );
        assert!(environment_line("BLUT_HOME", "/a\nExecStartPre=/bin/evil").is_err());
        let refused = |value: &str| {
            captured_environment(|name| {
                (name == "BLUT_REMOTE_CACHE").then(|| std::ffi::OsString::from(value))
            })
        };
        for bad in [
            "s3://key:secret@bucket/p",
            "s3://bucket/p?X-Amz-Signature=abc",
            "file:///x#frag",
        ] {
            assert!(refused(bad).is_err(), "{bad}");
        }
        for ok in ["s3://bucket/prefix", "/srv/cache", "/srv/user@host/cache"] {
            assert_eq!(refused(ok).unwrap().len(), 1, "{ok}");
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt as _;
            let err = captured_environment(|name| {
                (name == "BLUT_HOME").then(|| std::ffi::OsString::from_vec(vec![0xff]))
            })
            .expect_err("not UTF-8");
            assert!(err.to_string().contains("BLUT_HOME"), "{err}");
        }
    }

    #[test]
    fn install_refuses_an_invalid_tenant_or_experiment_before_writing() {
        let bad_tenant = ScheduleOptions {
            tenant: Some("a b".into()),
            ..ScheduleOptions::default()
        };
        assert!(install_with("count", "daily", "{}", &bad_tenant).is_err());
        let bad_experiment = ScheduleOptions {
            experiment: Some("../x".into()),
            ..ScheduleOptions::default()
        };
        assert!(install_with("count", "daily", "{}", &bad_experiment).is_err());
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
