//! blut-owned `Launcher` abstraction: build the OS command that runs a
//! sweep job, optionally inside a resource-capped container.
//!
//! This trait is intentionally NOT lerna's `Launcher` (which is sweep /
//! `JobReturn`-oriented and the wrong shape here). We only import lerna by
//! explicit path elsewhere so the names do not collide.

use std::process::Command;

use crate::error::{Result, TrainError};
use crate::paths;

/// Build (and optionally spawn) the command that runs a unit of work.
pub trait Launcher {
    /// Construct the [`Command`] that runs `inner` (the program + args) under
    /// the named `unit`. Does not spawn — callers can inspect or further
    /// configure the returned command first.
    fn build_command(&self, unit: &str, inner: &[String]) -> Result<Command>;

    /// Build and spawn the command, returning the child process handle.
    fn launch(&self, unit: &str, inner: &[String]) -> Result<std::process::Child> {
        let mut c = self.build_command(unit, inner)?;
        c.spawn()
            .map_err(|e| TrainError::other(format!("launch {unit}: {e}")))
    }
}

/// Run work in a memory-capped, session-detached transient systemd `--user`
/// service by shelling `tools/run_contained.sh`. The script reads
/// `UNIT`/`MEMMAX`/`MEMHIGH`/`SWAPMAX` from the environment and execs
/// `systemd-run --user --unit=$UNIT ... -- "$@"`.
#[derive(Clone, Debug)]
pub struct LocalSystemd {
    /// Hard memory cap (cgroup-OOM if hit). `MEMMAX`.
    pub mem_max: String,
    /// Soft memory cap (reclaim pressure). `MEMHIGH`.
    pub mem_high: String,
    /// Swap ceiling (avoid thrash). `SWAPMAX`.
    pub swap_max: String,
}

impl Default for LocalSystemd {
    fn default() -> Self {
        // Matches run_contained.sh defaults.
        Self {
            mem_max: "44G".to_string(),
            mem_high: "40G".to_string(),
            swap_max: "12G".to_string(),
        }
    }
}

impl Launcher for LocalSystemd {
    fn build_command(&self, unit: &str, inner: &[String]) -> Result<Command> {
        // run_contained.sh runs under systemd-run's MINIMAL cwd, so the
        // script path MUST be absolute.
        let root = paths::meta_repo_root()?;
        let script = root.join("tools").join("run_contained.sh");
        if !script.is_file() {
            return Err(TrainError::other(format!(
                "run_contained.sh not found at {}",
                script.display()
            )));
        }
        let mut c = Command::new("bash");
        c.arg(&script);
        c.args(inner);
        c.env("UNIT", unit);
        c.env("MEMMAX", &self.mem_max);
        c.env("MEMHIGH", &self.mem_high);
        c.env("SWAPMAX", &self.swap_max);
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_command_argv_is_inspectable() {
        // Relies on meta_repo_root() default resolution (walks up from
        // CARGO_MANIFEST_DIR to the real meta-repo where tools/run_contained.sh
        // exists); does not spawn systemd.
        let cmd = LocalSystemd::default()
            .build_command(
                "unit-x",
                &["python".to_string(), "-u".to_string(), "train.py".to_string()],
            )
            .expect("build_command should resolve the script");
        assert_eq!(cmd.get_program(), "bash");

        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(
            args[0].ends_with("tools/run_contained.sh"),
            "first arg must be the script path, got {:?}",
            args[0]
        );
        assert_eq!(&args[1..], &["python", "-u", "train.py"]);

        // UNIT/MEMMAX/MEMHIGH/SWAPMAX are passed via env (not argv).
        let envs: std::collections::HashMap<String, Option<String>> = cmd
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.map(|s| s.to_string_lossy().into_owned()),
                )
            })
            .collect();
        assert_eq!(envs.get("UNIT"), Some(&Some("unit-x".to_string())));
        assert_eq!(envs.get("MEMMAX"), Some(&Some("44G".to_string())));
        assert_eq!(envs.get("MEMHIGH"), Some(&Some("40G".to_string())));
        assert_eq!(envs.get("SWAPMAX"), Some(&Some("12G".to_string())));
    }
}
