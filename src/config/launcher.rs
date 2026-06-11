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

/// Run work as a Slurm job via `srun` — synchronous (the orchestrator blocks on
/// the cluster-scheduled allocation, mirroring `LocalSystemd`'s `--pipe --wait`
/// contract), so the executor's per-stage scheduling + the broker admission stay
/// the source of truth and only the *placement* moves to the cluster. The DAG's
/// per-stage resource needs map to `srun` flags.
///
/// SHARED-FILESYSTEM CONTRACT: the content-addressed cache + job dirs must live
/// on a filesystem the compute node can see (NFS/Lustre). Cross-node SYNC of a
/// node-local cache is a separate slice (see the `#3` roadmap); this backend
/// assumes a shared mount, which every real HPC cluster provides.
#[derive(Clone, Debug, Default)]
pub struct SlurmLauncher {
    /// `--partition`.
    pub partition: Option<String>,
    /// `--mem` (e.g. `"44G"`) — the per-job RAM allocation.
    pub mem: Option<String>,
    /// `--cpus-per-task`.
    pub cpus: Option<u32>,
    /// `--gpus`.
    pub gpus: Option<u32>,
    /// `--time` (e.g. `"08:00:00"`).
    pub time: Option<String>,
    /// Verbatim passthrough flags appended before the `--` separator.
    pub extra: Vec<String>,
}

impl Launcher for SlurmLauncher {
    fn build_command(&self, unit: &str, inner: &[String]) -> Result<Command> {
        if inner.is_empty() {
            return Err(TrainError::other("slurm launcher: empty inner command"));
        }
        let mut c = Command::new("srun");
        c.arg(format!("--job-name={unit}"));
        if let Some(p) = &self.partition {
            c.arg(format!("--partition={p}"));
        }
        if let Some(m) = &self.mem {
            c.arg(format!("--mem={m}"));
        }
        if let Some(n) = self.cpus {
            c.arg(format!("--cpus-per-task={n}"));
        }
        if let Some(g) = self.gpus {
            c.arg(format!("--gpus={g}"));
        }
        if let Some(t) = &self.time {
            c.arg(format!("--time={t}"));
        }
        for x in &self.extra {
            c.arg(x);
        }
        c.arg("--");
        c.args(inner);
        Ok(c)
    }
}

/// Run work as a Ray job via `ray job submit` — the cluster head node places it.
/// `address` targets the head (else `RAY_ADDRESS` from the env). Same
/// shared-filesystem / cache-locality contract as [`SlurmLauncher`].
#[derive(Clone, Debug, Default)]
pub struct RayLauncher {
    /// `--address` of the Ray head (e.g. `"http://127.0.0.1:8265"`).
    pub address: Option<String>,
    /// `--runtime-env-json` (deps / env for the job).
    pub runtime_env: Option<String>,
    /// Verbatim passthrough flags appended before the `--` separator.
    pub extra: Vec<String>,
}

impl Launcher for RayLauncher {
    fn build_command(&self, unit: &str, inner: &[String]) -> Result<Command> {
        if inner.is_empty() {
            return Err(TrainError::other("ray launcher: empty inner command"));
        }
        let mut c = Command::new("ray");
        c.arg("job").arg("submit");
        c.arg(format!("--submission-id={unit}"));
        if let Some(a) = &self.address {
            c.arg(format!("--address={a}"));
        }
        if let Some(re) = &self.runtime_env {
            c.arg("--runtime-env-json").arg(re);
        }
        for x in &self.extra {
            c.arg(x);
        }
        c.arg("--");
        c.args(inner);
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(c: &Command) -> Vec<String> {
        c.get_args().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn slurm_build_command_maps_resources_to_flags() {
        let l = SlurmLauncher {
            partition: Some("gpu".into()),
            mem: Some("44G".into()),
            cpus: Some(8),
            gpus: Some(1),
            time: Some("08:00:00".into()),
            extra: vec!["--exclusive".into()],
        };
        let c = l.build_command("job-x", &["python".into(), "train.py".into()]).unwrap();
        assert_eq!(c.get_program(), "srun");
        let a = argv(&c);
        assert!(a.contains(&"--job-name=job-x".to_string()));
        assert!(a.contains(&"--partition=gpu".to_string()));
        assert!(a.contains(&"--mem=44G".to_string()));
        assert!(a.contains(&"--cpus-per-task=8".to_string()));
        assert!(a.contains(&"--gpus=1".to_string()));
        assert!(a.contains(&"--time=08:00:00".to_string()));
        assert!(a.contains(&"--exclusive".to_string()));
        // The inner command follows the `--` separator, verbatim + last.
        let sep = a.iter().position(|x| x == "--").unwrap();
        assert_eq!(&a[sep + 1..], &["python", "train.py"]);
    }

    #[test]
    fn slurm_omits_unset_flags_and_rejects_empty() {
        let c = SlurmLauncher::default().build_command("u", &["echo".into()]).unwrap();
        let a = argv(&c);
        assert!(!a.iter().any(|x| x.starts_with("--mem")), "unset → no flag");
        assert!(!a.iter().any(|x| x.starts_with("--partition")));
        assert!(SlurmLauncher::default().build_command("u", &[]).is_err());
    }

    #[test]
    fn ray_build_command_submits_with_id() {
        let l = RayLauncher {
            address: Some("http://h:8265".into()),
            runtime_env: Some(r#"{"pip":["torch"]}"#.into()),
            extra: vec![],
        };
        let c = l.build_command("job-y", &["python".into(), "-m".into(), "t".into()]).unwrap();
        assert_eq!(c.get_program(), "ray");
        let a = argv(&c);
        assert_eq!(&a[..2], &["job", "submit"]);
        assert!(a.contains(&"--submission-id=job-y".to_string()));
        assert!(a.contains(&"--address=http://h:8265".to_string()));
        assert!(a.contains(&"--runtime-env-json".to_string()));
        let sep = a.iter().position(|x| x == "--").unwrap();
        assert_eq!(&a[sep + 1..], &["python", "-m", "t"]);
        assert!(RayLauncher::default().build_command("u", &[]).is_err());
    }

    #[test]
    fn build_command_argv_is_inspectable() {
        // Relies on meta_repo_root() default resolution (walks up from
        // CARGO_MANIFEST_DIR to the real meta-repo where tools/run_contained.sh
        // exists); does not spawn systemd.
        let cmd = LocalSystemd::default()
            .build_command(
                "unit-x",
                &[
                    "python".to_string(),
                    "-u".to_string(),
                    "train.py".to_string(),
                ],
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
