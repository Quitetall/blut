// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! blut-owned `Launcher` abstraction: build the OS command that runs a
//! sweep job, optionally inside a resource-capped container.

use std::path::PathBuf;
use std::process::Command;

use crate::error::{Result, TrainError};

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

    /// The ORDERED set of device indices a multi-cell scheduler spreads cells
    /// across (Phase G). Default `0..capacity`; `LocalSystemd` honors
    /// `$BLUT_SCHED_DEVICES` to target a SUBSET (e.g. reserve some GPUs for
    /// other work). Backfill concurrency is `device_set().len()`, and cell `i`
    /// is pinned to `device_set()[i % len]`. Always ≥ 1 entry.
    fn device_set(&self) -> Vec<usize> {
        (0..self.capacity().max(1)).collect()
    }
}

/// Parse `$BLUT_SCHED_DEVICES` (comma-separated device indices → deduped,
/// first-seen order preserved) into a device set. `Err` on a non-numeric
/// token; an all-blank value is an error (unset means "all"). Indices are
/// NOT range-checked against the GPU count — targeting a subset (or a
/// specific physical GPU) is the whole point.
pub fn parse_sched_devices(s: &str) -> Result<Vec<usize>> {
    let mut out: Vec<usize> = Vec::new();
    for tok in s.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()) {
        let idx: usize = tok.parse().map_err(|_| {
            TrainError::other(format!(
                "BLUT_SCHED_DEVICES: '{tok}' is not a non-negative device index"
            ))
        })?;
        if !out.contains(&idx) {
            out.push(idx); // dedup, preserve first-seen order
        }
    }
    if out.is_empty() {
        return Err(TrainError::other(
            "BLUT_SCHED_DEVICES is set but holds no valid device indices",
        ));
    }
    Ok(out)
}

/// The scheduler device set: `$BLUT_SCHED_DEVICES` if set + valid, else
/// `0..capacity`. A malformed override falls back to `0..capacity` with a
/// warning (never panics a scheduler launch).
pub fn sched_device_set(capacity: usize) -> Vec<usize> {
    match std::env::var("BLUT_SCHED_DEVICES") {
        Ok(s) => parse_sched_devices(&s).unwrap_or_else(|e| {
            tracing::warn!("{e}; falling back to 0..{capacity}");
            (0..capacity.max(1)).collect()
        }),
        Err(_) => (0..capacity.max(1)).collect(),
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

/// The contained-launch helper, embedded into the engine so a published /
/// `cargo install`ed binary carries it with NO dependency on a surrounding
/// repo layout (the old `meta_repo_root()` walk assumed blut lived inside its
/// meta-repo). It is materialized to a cache dir at first contained launch.
const RUN_CONTAINED_SH: &str = include_str!("../../scripts/run_contained.sh");

/// Whether the never-OOM cgroup containment can be applied: it needs Linux +
/// systemd `--user`, and the user must not have opted out. Off-systemd hosts
/// (macOS / a container without systemd) and `BLUT_NO_CONTAIN` degrade to a
/// bare spawn — see [`LocalSystemd::wrap`].
fn containment_available() -> bool {
    if std::env::var_os("BLUT_NO_CONTAIN").is_some() {
        return false;
    }
    which::which("systemd-run").is_ok()
}

/// Materialize the embedded `run_contained.sh` to a stable, absolute path and
/// return it (systemd-run's `--user` cwd is minimal, so the path must be
/// absolute). Resolution order:
///   1. `$BLUT_RUN_CONTAINED` — explicit path override (a patched script).
///   2. `<base>/blut/run_contained.sh`, written from the embedded copy and
///      refreshed when content-stale, where `<base>` is `$BLUT_HOME` or the
///      XDG cache dir.
fn resolve_run_contained_script() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_RUN_CONTAINED") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
        return Err(TrainError::other(format!(
            "$BLUT_RUN_CONTAINED={} is not a file",
            p.display()
        )));
    }
    let base = match std::env::var("BLUT_HOME") {
        Ok(h) => PathBuf::from(h),
        Err(_) => dirs::cache_dir().ok_or_else(|| {
            TrainError::other("cache_dir() unavailable; set $BLUT_HOME or $BLUT_RUN_CONTAINED")
        })?,
    };
    let dir = base.join("blut");
    std::fs::create_dir_all(&dir).map_err(|e| TrainError::Io {
        path: dir.clone(),
        source: e,
    })?;
    let script = dir.join("run_contained.sh");
    // Rewrite when missing or content-stale (e.g. after an engine upgrade).
    let stale = std::fs::read_to_string(&script)
        .map(|cur| cur != RUN_CONTAINED_SH)
        .unwrap_or(true);
    if stale {
        std::fs::write(&script, RUN_CONTAINED_SH).map_err(|e| TrainError::Io {
            path: script.clone(),
            source: e,
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script)
                .map_err(|e| TrainError::Io {
                    path: script.clone(),
                    source: e,
                })?
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script, perms).map_err(|e| TrainError::Io {
                path: script.clone(),
                source: e,
            })?;
        }
    }
    Ok(script)
}

/// Run work in a memory-capped, session-detached transient systemd `--user`
/// service by shelling the embedded `run_contained.sh`. The script reads
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

    fn device_set(&self) -> Vec<usize> {
        sched_device_set(self.capacity())
    }

    fn wrap(&self, unit: &str, inner: &[String]) -> Result<WrappedCommand> {
        // Containment needs Linux + systemd `--user`. Off-systemd (macOS / a
        // container without systemd) or with `BLUT_NO_CONTAIN` set, degrade to a
        // BARE spawn with a warning — the never-OOM cgroup cap is then NOT
        // enforced (admission's box-fit refusal still gates, and the kernel OOM
        // killer is the only hard backstop).
        if !containment_available() {
            let Some((program, rest)) = inner.split_first() else {
                return Err(TrainError::other("wrap: empty inner command"));
            };
            if std::env::var_os("BLUT_NO_CONTAIN").is_none() {
                tracing::warn!(
                    "systemd-run unavailable — running unit '{unit}' UNCONTAINED (no memory cap). \
                     Containment requires Linux + systemd --user. Set BLUT_NO_CONTAIN=1 to silence."
                );
            }
            return Ok(WrappedCommand {
                program: program.clone(),
                args: rest.to_vec(),
                env: Vec::new(),
                kind: LauncherKind::Local,
            });
        }
        // run_contained.sh runs under systemd-run's MINIMAL cwd, so the script
        // path MUST be absolute. It is embedded + materialized to a cache dir,
        // so containment works from a clean install (no repo-layout assumption).
        let script = resolve_run_contained_script()?;
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
        c.get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn capacity_reflects_cuda_visible_devices() {
        // `CUDA_VISIBLE_DEVICES` is process-global → serialize the env mutation.
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prior = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        // SAFETY: serialized by TEST_ENV_LOCK; restored below.
        unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "0,1,2") };
        assert_eq!(LocalSystemd::default().capacity(), 3, "3 visible devices");
        unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "") };
        assert_eq!(
            LocalSystemd::default().capacity(),
            1,
            "no GPUs ⇒ still ≥1 (run one cell)"
        );
        match prior {
            Some(v) => unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", v) },
            None => unsafe { std::env::remove_var("CUDA_VISIBLE_DEVICES") },
        }
        // Slurm's capacity = its per-job --gpus allocation.
        assert_eq!(
            SlurmLauncher {
                gpus: Some(4),
                ..Default::default()
            }
            .capacity(),
            4
        );
        assert_eq!(SlurmLauncher::default().capacity(), 1, "unset --gpus ⇒ 1");
    }

    #[test]
    fn parse_sched_devices_dedups_and_validates() {
        assert_eq!(parse_sched_devices("0,1,2").unwrap(), vec![0, 1, 2]);
        // Order preserved, dupes removed (round-robin stays stable).
        assert_eq!(parse_sched_devices("2, 0, 2, 1, 0").unwrap(), vec![2, 0, 1]);
        // Reserve a subset (fewer than capacity) is legal.
        assert_eq!(parse_sched_devices("3").unwrap(), vec![3]);
        // Non-numeric / all-blank → error.
        assert!(parse_sched_devices("0,x,1").is_err());
        assert!(parse_sched_devices("  ,  ").is_err());
        assert!(parse_sched_devices("").is_err());
    }

    #[test]
    fn device_set_default_and_env_override() {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prior = std::env::var("BLUT_SCHED_DEVICES").ok();
        unsafe { std::env::remove_var("BLUT_SCHED_DEVICES") };
        // Default = 0..capacity.
        assert_eq!(sched_device_set(3), vec![0, 1, 2]);
        assert_eq!(
            sched_device_set(0),
            vec![0],
            "≥1 entry so a scheduler progresses"
        );
        // Override targets a subset (e.g. reserve GPU 1 on a 3-GPU box).
        unsafe { std::env::set_var("BLUT_SCHED_DEVICES", "0,2") };
        assert_eq!(sched_device_set(3), vec![0, 2]);
        // Malformed override → warn + fall back to 0..capacity (never panics).
        unsafe { std::env::set_var("BLUT_SCHED_DEVICES", "0,bad") };
        assert_eq!(sched_device_set(2), vec![0, 1]);
        match prior {
            Some(v) => unsafe { std::env::set_var("BLUT_SCHED_DEVICES", v) },
            None => unsafe { std::env::remove_var("BLUT_SCHED_DEVICES") },
        }
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
        let c = l
            .build_command("job-x", &["python".into(), "train.py".into()])
            .unwrap();
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
        let c = SlurmLauncher::default()
            .build_command("u", &["echo".into()])
            .unwrap();
        let a = argv(&c);
        assert!(!a.iter().any(|x| x.starts_with("--mem")), "unset → no flag");
        assert!(!a.iter().any(|x| x.starts_with("--partition")));
        assert!(SlurmLauncher::default().build_command("u", &[]).is_err());
    }

    #[test]
    fn launch_target_parses_and_resolves() {
        use std::str::FromStr;
        assert_eq!(
            LaunchTarget::from_str("slurm").unwrap(),
            LaunchTarget::Slurm
        );
        assert_eq!(LaunchTarget::from_str("RAY").unwrap(), LaunchTarget::Ray);
        assert_eq!(LaunchTarget::from_str("").unwrap(), LaunchTarget::Local);
        assert!(LaunchTarget::from_str("k8s").is_err());
        // The factory maps target → the matching launcher kind.
        assert_eq!(
            launcher_for(LaunchTarget::Slurm)
                .wrap("u", &["x".into()])
                .unwrap()
                .kind,
            LauncherKind::Slurm
        );
        assert_eq!(
            launcher_for(LaunchTarget::Ray)
                .wrap("u", &["x".into()])
                .unwrap()
                .kind,
            LauncherKind::Ray
        );
    }

    #[test]
    fn wrap_carries_kind_and_splits_env_from_argv() {
        // Slurm: all knobs are argv, env empty, kind Slurm.
        let s = SlurmLauncher {
            mem: Some("8G".into()),
            ..Default::default()
        }
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
        let c = l
            .build_command("job-y", &["python".into(), "-m".into(), "t".into()])
            .unwrap();
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
        // Materializes the EMBEDDED run_contained.sh to a cache dir (no repo
        // layout / meta_repo_root needed); does not spawn systemd. On a host
        // without systemd-run (or with BLUT_NO_CONTAIN) the launcher degrades to
        // a bare spawn — assert whichever path THIS host takes. Hold the env lock
        // so a concurrent BLUT_NO_CONTAIN-mutating test can't flip the branch
        // between the probe and the assertions.
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let inner = [
            "python".to_string(),
            "-u".to_string(),
            "train.py".to_string(),
        ];
        let cmd = LocalSystemd::default()
            .build_command("unit-x", &inner)
            .expect("build_command should resolve");
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();

        if containment_available() {
            assert_eq!(cmd.get_program(), "bash");
            assert!(
                args[0].ends_with("run_contained.sh") && args[0].starts_with('/'),
                "first arg must be the absolute materialized script path, got {:?}",
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
        } else {
            // Degraded (off-systemd) path: bare spawn of the inner command.
            assert_eq!(cmd.get_program(), "python");
            assert_eq!(&args[..], &["-u", "train.py"]);
        }
    }

    #[test]
    fn no_contain_env_degrades_to_bare_spawn() {
        // BLUT_NO_CONTAIN forces the uncontained bare-spawn path regardless of
        // systemd availability. Serialized on the shared env lock.
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("BLUT_NO_CONTAIN").ok();
        // SAFETY: TEST_ENV_LOCK serializes env mutation; restored below.
        unsafe {
            std::env::set_var("BLUT_NO_CONTAIN", "1");
        }
        let cmd = LocalSystemd::default()
            .build_command("u", &["python".to_string(), "x.py".to_string()])
            .expect("bare spawn builds");
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(cmd.get_program(), "python");
        assert_eq!(&args[..], &["x.py"]);
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BLUT_NO_CONTAIN", v),
                None => std::env::remove_var("BLUT_NO_CONTAIN"),
            }
        }
    }
}
