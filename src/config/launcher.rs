//! blut-owned `Launcher` abstraction: build the OS command that runs a
//! sweep job, optionally inside a resource-capped container.
//!
//! This trait is intentionally NOT lerna's `Launcher` (which is sweep /
//! `JobReturn`-oriented and the wrong shape here). We only import lerna by
//! explicit path elsewhere so the names do not collide.

use std::process::Command;

use crate::error::{Result, TrainError};
use crate::paths;

/// Which launcher produced a [`WrappedCommand`]. Lets a backend pick the right
/// liveness / OOM-peak handling for the placement mechanism: systemd's
/// `Memory peak:` stderr line + cgroup for `Local`, exit-code / `sacct` for
/// `Slurm`, the Ray job status for `Ray`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LauncherKind {
    Local,
    Slurm,
    Ray,
}

/// Backend-AGNOSTIC launch spec: the program + argv + env a launcher prepends
/// around the inner command. A backend builds its OWN process from this — a
/// `std::process::Command` for simple spawns, or a `tokio::process::Command`
/// with `pre_exec`/piped-stdout for the async status-streaming + OOM-classify
/// path. So the launcher owns command CONSTRUCTION; the backend owns process
/// MANAGEMENT. This split is what lets the same launcher feed both the blocking
/// CLI path and the streaming trainer path without the trait knowing about
/// tokio or the broker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrappedCommand {
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub kind: LauncherKind,
}

/// Build the launch spec that runs a unit of work, optionally on a remote
/// scheduler. Implementors produce a [`WrappedCommand`]; the trait derives the
/// `std::process::Command` + spawn conveniences from it.
pub trait Launcher {
    /// The launcher prefix + env wrapping `inner` (program + args), under the
    /// named `unit`. The backend-agnostic core.
    fn wrap(&self, unit: &str, inner: &[String]) -> Result<WrappedCommand>;

    /// A `std::process::Command` from [`wrap`](Self::wrap), for callers that
    /// inspect/spawn directly. The async backends build a `tokio` command from
    /// `wrap` instead (so they can stream status + classify OOM per `kind`).
    fn build_command(&self, unit: &str, inner: &[String]) -> Result<Command> {
        let w = self.wrap(unit, inner)?;
        let mut c = Command::new(&w.program);
        c.args(&w.args);
        for (k, v) in &w.env {
            c.env(k, v);
        }
        Ok(c)
    }

    /// Build and spawn, returning the child process handle.
    fn launch(&self, unit: &str, inner: &[String]) -> Result<std::process::Child> {
        let mut c = self.build_command(unit, inner)?;
        c.spawn()
            .map_err(|e| TrainError::other(format!("launch {unit}: {e}")))
    }

    /// How many units this launcher can place CONCURRENTLY — the device
    /// parallelism a multi-cell sweep / partition scheduler packs against
    /// (Phase G). Default `1` (serialize); `Local` probes the visible GPU
    /// count, `Slurm` reports its per-job `--gpus`. ALWAYS ≥ 1 so a scheduler
    /// can always make progress (a no-GPU box still runs one cell at a time).
    fn capacity(&self) -> usize {
        1
    }
}

/// The number of GPUs VISIBLE to this process: `CUDA_VISIBLE_DEVICES` when set
/// (the authoritative visible set — empty string ⇒ no GPUs), else an
/// `nvidia-smi -L` probe, else 1 (assume the common single-device dev box when
/// we can't tell). Clamped to ≥ 1 for the scheduler-capacity use.
pub fn local_gpu_count() -> usize {
    if let Ok(v) = std::env::var("CUDA_VISIBLE_DEVICES") {
        // A device is each non-empty, comma-separated entry. "" ⇒ 0 GPUs.
        let n = v.split(',').filter(|s| !s.trim().is_empty()).count();
        return n.max(1); // ≥1: even with no GPU, a scheduler runs one cell.
    }
    if let Ok(out) = std::process::Command::new("nvidia-smi").arg("-L").output() {
        if out.status.success() {
            let n = String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| l.trim_start().starts_with("GPU "))
                .count();
            if n > 0 {
                return n;
            }
        }
    }
    1
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
    fn capacity(&self) -> usize {
        local_gpu_count()
    }

    fn wrap(&self, unit: &str, inner: &[String]) -> Result<WrappedCommand> {
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
        let mut args = vec![script.display().to_string()];
        args.extend(inner.iter().cloned());
        Ok(WrappedCommand {
            program: "bash".to_string(),
            args,
            env: vec![
                ("UNIT".into(), unit.to_string()),
                ("MEMMAX".into(), self.mem_max.clone()),
                ("MEMHIGH".into(), self.mem_high.clone()),
                ("SWAPMAX".into(), self.swap_max.clone()),
            ],
            kind: LauncherKind::Local,
        })
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
    fn capacity(&self) -> usize {
        // The per-job GPU allocation is this launcher's device parallelism.
        self.gpus.map(|g| g as usize).unwrap_or(1).max(1)
    }

    fn wrap(&self, unit: &str, inner: &[String]) -> Result<WrappedCommand> {
        if inner.is_empty() {
            return Err(TrainError::other("slurm launcher: empty inner command"));
        }
        let mut args = vec![format!("--job-name={unit}")];
        if let Some(p) = &self.partition {
            args.push(format!("--partition={p}"));
        }
        if let Some(m) = &self.mem {
            args.push(format!("--mem={m}"));
        }
        if let Some(n) = self.cpus {
            args.push(format!("--cpus-per-task={n}"));
        }
        if let Some(g) = self.gpus {
            args.push(format!("--gpus={g}"));
        }
        if let Some(t) = &self.time {
            args.push(format!("--time={t}"));
        }
        // A bare "--" in extra would prematurely close option parsing; the real
        // inner separator is appended below.
        args.extend(self.extra.iter().filter(|x| *x != "--").cloned());
        args.push("--".to_string());
        args.extend(inner.iter().cloned());
        Ok(WrappedCommand {
            program: "srun".to_string(),
            args,
            env: Vec::new(),
            kind: LauncherKind::Slurm,
        })
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
    fn wrap(&self, unit: &str, inner: &[String]) -> Result<WrappedCommand> {
        if inner.is_empty() {
            return Err(TrainError::other("ray launcher: empty inner command"));
        }
        let mut args = vec![
            "job".to_string(),
            "submit".to_string(),
            format!("--submission-id={unit}"),
        ];
        if let Some(a) = &self.address {
            args.push(format!("--address={a}"));
        }
        if let Some(re) = &self.runtime_env {
            args.push("--runtime-env-json".to_string());
            args.push(re.clone());
        }
        args.extend(self.extra.iter().filter(|x| *x != "--").cloned());
        args.push("--".to_string());
        args.extend(inner.iter().cloned());
        Ok(WrappedCommand {
            program: "ray".to_string(),
            args,
            env: Vec::new(),
            kind: LauncherKind::Ray,
        })
    }
}

/// Where to place a unit of work, selected by `--launcher`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LaunchTarget {
    /// This box, broker-gated + cgroup-contained (the never-OOM path).
    #[default]
    Local,
    /// A Slurm allocation (`srun`).
    Slurm,
    /// A Ray job (`ray job submit`).
    Ray,
}

impl std::str::FromStr for LaunchTarget {
    type Err = TrainError;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "local" | "" => Ok(Self::Local),
            "slurm" => Ok(Self::Slurm),
            "ray" => Ok(Self::Ray),
            other => Err(TrainError::other(format!(
                "unknown launcher '{other}' (expected local|slurm|ray)"
            ))),
        }
    }
}

/// The configured [`Launcher`] for a target. Slurm/Ray read their cluster
/// config from env (`BLUT_SLURM_*` / `RAY_ADDRESS` / `BLUT_RAY_RUNTIME_ENV`) so
/// the DAG itself stays portable — the same recipe runs locally or on a cluster
/// by flipping `--launcher`, no recipe edit.
pub fn launcher_for(target: LaunchTarget) -> Box<dyn Launcher> {
    let env = |k: &str| std::env::var(k).ok();
    let env_u32 = |k: &str| env(k).and_then(|s| s.parse::<u32>().ok());
    match target {
        LaunchTarget::Local => Box::new(LocalSystemd::default()),
        LaunchTarget::Slurm => Box::new(SlurmLauncher {
            partition: env("BLUT_SLURM_PARTITION"),
            mem: env("BLUT_SLURM_MEM"),
            cpus: env_u32("BLUT_SLURM_CPUS"),
            gpus: env_u32("BLUT_SLURM_GPUS"),
            time: env("BLUT_SLURM_TIME"),
            extra: Vec::new(),
        }),
        LaunchTarget::Ray => Box::new(RayLauncher {
            address: env("RAY_ADDRESS"),
            runtime_env: env("BLUT_RAY_RUNTIME_ENV"),
            extra: Vec::new(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(c: &Command) -> Vec<String> {
        c.get_args().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn capacity_reflects_cuda_visible_devices() {
        // `CUDA_VISIBLE_DEVICES` is process-global → serialize the env mutation.
        let _g = crate::TEST_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prior = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        // SAFETY: serialized by TEST_ENV_LOCK; restored below.
        unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "0,1,2") };
        assert_eq!(LocalSystemd::default().capacity(), 3, "3 visible devices");
        unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "") };
        assert_eq!(LocalSystemd::default().capacity(), 1, "no GPUs ⇒ still ≥1 (run one cell)");
        match prior {
            Some(v) => unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", v) },
            None => unsafe { std::env::remove_var("CUDA_VISIBLE_DEVICES") },
        }
        // Slurm's capacity = its per-job --gpus allocation.
        assert_eq!(SlurmLauncher { gpus: Some(4), ..Default::default() }.capacity(), 4);
        assert_eq!(SlurmLauncher::default().capacity(), 1, "unset --gpus ⇒ 1");
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
    fn launch_target_parses_and_resolves() {
        use std::str::FromStr;
        assert_eq!(LaunchTarget::from_str("slurm").unwrap(), LaunchTarget::Slurm);
        assert_eq!(LaunchTarget::from_str("RAY").unwrap(), LaunchTarget::Ray);
        assert_eq!(LaunchTarget::from_str("").unwrap(), LaunchTarget::Local);
        assert!(LaunchTarget::from_str("k8s").is_err());
        // The factory maps target → the matching launcher kind.
        assert_eq!(
            launcher_for(LaunchTarget::Slurm).wrap("u", &["x".into()]).unwrap().kind,
            LauncherKind::Slurm
        );
        assert_eq!(
            launcher_for(LaunchTarget::Ray).wrap("u", &["x".into()]).unwrap().kind,
            LauncherKind::Ray
        );
    }

    #[test]
    fn wrap_carries_kind_and_splits_env_from_argv() {
        // Slurm: all knobs are argv, env empty, kind Slurm.
        let s = SlurmLauncher { mem: Some("8G".into()), ..Default::default() }
            .wrap("u", &["echo".into()])
            .unwrap();
        assert_eq!(s.kind, LauncherKind::Slurm);
        assert_eq!(s.program, "srun");
        assert!(s.env.is_empty());
        assert!(s.args.contains(&"--mem=8G".to_string()));
        // Ray: kind Ray.
        let r = RayLauncher::default().wrap("u", &["echo".into()]).unwrap();
        assert_eq!(r.kind, LauncherKind::Ray);
        assert_eq!(r.program, "ray");
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
