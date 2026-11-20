// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! blut-owned `Launcher` abstraction: build the OS command that runs a
//! sweep job, optionally inside a resource-capped container.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use crate::containment::{CapSpec, ContainedChild};
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

/// State of a remote job, as reported by the scheduler.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum JobState {
    /// Job is queued or running.
    Running,
    /// Job completed successfully.
    Succeeded,
    /// Job failed (non-zero exit, OOM, timeout, etc). The string carries
    /// the scheduler-reported reason (e.g. `"OutOfMemory"`, `"TIMEOUT"`).
    Failed(String),
    /// Job was cancelled by the user or the scheduler.
    Cancelled,
    /// State could not be determined from the scheduler's output.
    Unknown(String),
}

/// A handle to a running remote job. The runner calls [`poll`](Self::poll) in
/// a loop and [`stream`](Self::stream) to receive log lines for metric
/// parsing. Implementations are *synchronous* — no async runtime required — so
/// the runner can drive them from a plain thread alongside the blocking
/// orchestrator path.
pub trait RemoteJob: Send + Sync {
    /// The scheduler-assigned job ID (Slurm job-id, Ray submission-id, etc).
    fn id(&self) -> &str;

    /// Query the current job state. Called in a loop by the runner.
    fn poll(&self) -> Result<JobState>;

    /// Stream log lines to the provided sink. The sink receives each stdout
    /// line (the runner feeds it to `parse_step_update` for `BLUT_METRIC`
    /// extraction). This is a *blocking* call that returns when the job
    /// finishes or is cancelled.
    fn stream(&self, sink: &dyn Fn(&str)) -> Result<()>;

    /// Cancel the running job.
    fn cancel(&self) -> Result<()>;
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
    // GPU discovery now goes through the single consolidated one-shot probe
    // (ADR 0087) — no separate `nvidia-smi -L` here. Honors BLUT_GPU_INVENTORY.
    crate::broker::gpu::GpuInventory::probe().len().max(1)
}

/// The contained-launch helper, embedded into the engine so a published /
/// `cargo install`ed binary carries it with NO dependency on a surrounding
/// repo layout (the old `meta_repo_root()` walk assumed blut lived inside its
/// meta-repo). It is materialized to a cache dir at first contained launch.
const RUN_CONTAINED_SH: &str = include_str!("../../scripts/run_contained.sh");

/// Whether the memory-admission cgroup containment can be applied: it needs Linux +
/// systemd `--user`, and the user must not have opted out. Off-systemd hosts
/// (macOS / a container without systemd) and `BLUT_NO_CONTAIN` degrade to a
/// bare spawn — see [`LocalSystemd::wrap`].
fn containment_available() -> bool {
    if crate::config::env_flag("BLUT_NO_CONTAIN") {
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
    if let Some(p) = std::env::var_os("BLUT_RUN_CONTAINED") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
        return Err(TrainError::other(format!(
            "$BLUT_RUN_CONTAINED={} is not a file",
            p.display()
        )));
    }
    let base = match std::env::var_os("BLUT_HOME") {
        Some(h) => PathBuf::from(h),
        None => dirs::cache_dir().ok_or_else(|| {
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
///
/// [`wrap`](Launcher::wrap) and [`launch`](Launcher::launch) contain only
/// where a wrapper command can: with systemd `--user`. To contain a local
/// stage on any host, Windows included, spawn it with
/// [`spawn_contained`](Self::spawn_contained).
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

impl LocalSystemd {
    /// This launcher's memory knobs as a [`CapSpec`] in bytes. An empty knob
    /// is left unset; a knob that is not a memory size (`44G`, `512M`, bytes)
    /// is an error rather than silently uncapped.
    pub fn cap_spec(&self) -> Result<CapSpec> {
        let knob = |name: &str, value: &str| -> Result<Option<u64>> {
            let value = value.trim();
            if value.is_empty() {
                return Ok(None);
            }
            match crate::containment::memsize::parse_memsize(value) {
                Some(bytes) if bytes > 0 => Ok(Some(bytes)),
                _ => Err(TrainError::other(format!(
                    "LocalSystemd {name}={value:?} is not a memory size (e.g. 44G, 512M)"
                ))),
            }
        };
        Ok(CapSpec {
            mem_max: knob("mem_max", &self.mem_max)?,
            mem_high: knob("mem_high", &self.mem_high)?,
            swap_max: knob("swap_max", &self.swap_max)?,
        })
    }

    /// Spawn `inner` (program, then its arguments) as a local stage process
    /// under this host's containment backend
    /// ([`containment_for`](crate::containment::containment_for)), capped by
    /// this launcher's [`cap_spec`](Self::cap_spec).
    ///
    /// This is the contained counterpart of [`Launcher::launch`]. What each
    /// backend does with the stage:
    ///
    /// | Backend | Where | Cap | Peak | Tree kill |
    /// |---|---|---|---|---|
    /// | `systemd` | a transient `systemd-run --user` service | `MemoryMax`/`MemoryHigh`/`MemorySwapMax` | `Memory peak:` on stderr, when it is pumped ([`ContainedChild::pump_stderr`]) | `systemctl --user stop` |
    /// | `cgroup2` | a leaf in a delegated cgroup-v2 subtree | `memory.max` | `memory.peak` | `cgroup.kill` |
    /// | `windows-job-object` | a Job Object | tree-wide commit | `PeakJobMemoryUsed` | `TerminateJobObject` |
    /// | `rlimit` | the child itself | none (see below) | none | the caller's process group |
    /// | `bare` | the child itself | none | none | the caller's process group |
    ///
    /// The stage sees the caller's environment plus `env` under every
    /// backend; for `systemd`, whose services otherwise start from the user
    /// manager's minimal environment, it is forwarded variable by variable.
    /// That needs systemd 250 or later, which imports a value by name and so
    /// keeps values off the world-readable `systemd-run` command line; with an
    /// older systemd the stage runs under the next backend instead (an
    /// explicit `BLUT_CONTAINMENT=systemd` is refused there). Under `rlimit`
    /// no cap is applied: its cap is on virtual address space, and the
    /// launcher's default would refuse the reservations a CUDA process makes.
    ///
    /// One log line names the backend and what it applies. The unit name is
    /// `unit` (sanitized to `[A-Za-z0-9_-]`) plus this process's pid and a
    /// sequence number, so concurrent and repeated stages never collide.
    /// `configure` sets pipes, `kill_on_drop` and similar before the spawn.
    ///
    /// A `systemd` unit is stopped with SIGTERM and, 10 s later, SIGKILL
    /// (`TimeoutStopSec=10s`). A `systemd` unit and a `cgroup2` leaf outlive
    /// this process when it is SIGKILLed, so when called inside a job
    /// ([`jobs::in_job`](crate::jobs::in_job), which the executor sets up for
    /// every stage), each is recorded in the job directory before the stage's
    /// process starts ([`jobs::record_systemd_unit`](crate::jobs::record_systemd_unit),
    /// [`jobs::record_cgroup_leaf`](crate::jobs::record_cgroup_leaf)), and a
    /// unit is described as `blut job <job id>`: `blut cancel` stops either,
    /// and `plan resume` refuses while either runs, also after this process
    /// died.
    pub async fn spawn_contained(
        &self,
        unit: &str,
        inner: &[String],
        cwd: &Path,
        env: &[(String, String)],
        configure: impl FnOnce(&mut tokio::process::Command),
    ) -> Result<ContainedChild> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let Some((program, args)) = inner.split_first() else {
            return Err(TrainError::other("spawn_contained: empty inner command"));
        };
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let unit = format!("{}-{}-{seq}", sanitize_unit(unit), std::process::id());
        let (backend, note) = local_stage_backend()?;
        let mut caps = self.cap_spec()?;
        let kind = backend.kind();
        let applied = match kind {
            "bare" => "no memory cap, no tree teardown".to_string(),
            "rlimit" => {
                caps = CapSpec::default();
                "no memory cap (an address-space cap would refuse CUDA's reservations)".to_string()
            }
            _ => format!(
                "memory cap {}",
                caps.mem_max
                    .map_or_else(|| "none".to_string(), |b| format!("{b} bytes"))
            ),
        };
        #[cfg(target_os = "linux")]
        let env = if kind == "systemd" {
            crate::containment::systemd::inherited_env(env)
        } else {
            env.to_vec()
        };
        #[cfg(not(target_os = "linux"))]
        let env = env.to_vec();
        // A systemd unit outlives this process when it is SIGKILLed. Inside a
        // job the unit is recorded before it starts, so `blut cancel` and
        // `plan resume` find it, and its description names the job, so they
        // never take a later unit of the same name for it.
        #[cfg(target_os = "linux")]
        let run = match crate::jobs::current_job_dir().filter(|_| kind == "systemd") {
            Some(job_dir) => {
                if let Err(e) = crate::jobs::record_systemd_unit(&job_dir, &unit) {
                    tracing::warn!(
                        "containment: record {unit} with job {}: {e}",
                        job_dir.display()
                    );
                }
                crate::containment::systemd::SystemdCgroup.wrap_described(
                    Path::new(program),
                    args,
                    cwd,
                    &env,
                    &unit,
                    &caps,
                    Some(&crate::jobs::systemd_unit_description(&job_dir)),
                )?
            }
            None => backend.wrap_command(Path::new(program), args, cwd, &env, &unit, &caps)?,
        };
        #[cfg(not(target_os = "linux"))]
        let run = backend.wrap_command(Path::new(program), args, cwd, &env, &unit, &caps)?;
        let line =
            format!("containment: {unit} placed by LocalSystemd under {kind}, {applied}{note}");
        if matches!(kind, "bare" | "rlimit") {
            tracing::warn!("{line}");
        } else {
            tracing::info!("{line}");
        }
        ContainedChild::spawn(backend, run, unit, configure).await
    }
}

/// The backend for a local stage, and a note for the placement log line
/// saying why it is not the one [`containment_for`](crate::containment::containment_for)
/// would pick, when that is so. Refuses an explicitly named backend that is
/// not usable here, rather than running the stage under a different one.
fn local_stage_backend() -> Result<(Box<dyn crate::containment::Containment>, String)> {
    use crate::containment::Availability;
    let explicit = std::env::var("BLUT_CONTAINMENT")
        .ok()
        .filter(|v| !v.is_empty() && v != "auto");
    let backend = crate::containment::containment_for();
    if backend.available() != Availability::Present {
        return Err(TrainError::other(format!(
            "BLUT_CONTAINMENT={} names a containment backend that is not usable on this host \
             ({:?}); unset it to let BLUT pick one",
            explicit.as_deref().unwrap_or(backend.kind()),
            backend.available()
        )));
    }
    #[cfg(target_os = "linux")]
    if backend.kind() == "systemd" && !crate::containment::systemd::imports_env_by_name() {
        let version = crate::containment::systemd::systemd_run_version()
            .map_or_else(|| "unknown".to_string(), |v| v.to_string());
        if explicit.is_some() {
            return Err(TrainError::other(format!(
                "BLUT_CONTAINMENT=systemd: systemd-run {version} cannot pass a stage its \
                 environment without putting every value on its command line (needs systemd \
                 {} or later); unset BLUT_CONTAINMENT to use the next backend",
                crate::containment::systemd::SETENV_BY_NAME_SINCE
            )));
        }
        return Ok((
            crate::containment::containment_for_without_systemd(),
            format!(
                " (systemd-run {version} is older than {}, which a stage's environment needs)",
                crate::containment::systemd::SETENV_BY_NAME_SINCE
            ),
        ));
    }
    let note = if crate::config::env_flag("BLUT_NO_CONTAIN") {
        " (BLUT_NO_CONTAIN is set)".to_string()
    } else if let Some(name) = &explicit {
        format!(" (BLUT_CONTAINMENT={name})")
    } else {
        match backend.kind() {
            "bare" | "rlimit" => {
                " (no systemd user bus, no cgroup-v2 subtree with the memory controller)"
            }
            "cgroup2" => " (no systemd user bus)",
            _ => "",
        }
        .to_string()
    };
    Ok((backend, note))
}

/// A unit name every backend accepts: `[A-Za-z0-9_-]`, anything else `-`.
fn sanitize_unit(unit: &str) -> String {
    let s: String = unit
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    if s.is_empty() { "blut".to_string() } else { s }
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
        // BARE spawn with a warning — the memory-admission cgroup cap is then NOT
        // enforced (admission's box-fit refusal still gates, and the kernel OOM
        // killer is the only hard backstop).
        if !containment_available() {
            let Some((program, rest)) = inner.split_first() else {
                return Err(TrainError::other("wrap: empty inner command"));
            };
            if crate::config::env_flag("BLUT_NO_CONTAIN") {
                // Opted out: say nothing.
            } else if cfg!(windows) {
                tracing::warn!(
                    "LocalSystemd::wrap builds a bare command for unit '{unit}' on Windows: a \
                     Job Object cannot be expressed as a wrapper command, so whoever spawns it \
                     runs it UNCONTAINED (no memory cap, no tree kill). Spawn through \
                     LocalSystemd::spawn_contained instead. Set BLUT_NO_CONTAIN=1 to silence."
                );
            } else {
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
    /// `--gpus` — the job's TOTAL GPU count across all its nodes (Slurm's
    /// semantics for `--gpus`), and this launcher's [`capacity`](Launcher::capacity).
    /// For a per-node count pass `--gpus-per-node=N` in [`extra`](Self::extra).
    /// Slurm accepts both when they agree (this = nodes × N) and refuses the
    /// job at submission when they don't.
    pub gpus: Option<u32>,
    /// `--time` (e.g. `"08:00:00"`).
    pub time: Option<String>,
    /// `--nodes` — number of cluster nodes to allocate.
    pub nodes: Option<u32>,
    /// `--ntasks-per-node` — tasks (ranks) per node.
    pub ntasks_per_node: Option<u32>,
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
        if let Some(n) = self.nodes {
            args.push(format!("--nodes={n}"));
            if n > 1 && self.ntasks_per_node.is_none() {
                tracing::warn!(
                    "SlurmLauncher: --nodes={n} without --ntasks-per-node; \
                     Slurm defaults to 1 task/node — set BLUT_SLURM_NTASKS_PER_NODE \
                     if you want N ranks per node (e.g. --ntasks-per-node=gpus)."
                );
            }
        }
        if let Some(n) = self.ntasks_per_node {
            args.push(format!("--ntasks-per-node={n}"));
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

/// Slurm jobs queued or running under the job name `name`, by id, as
/// `squeue` reports them.
///
/// BLUT names the Slurm job a stage starts after the BLUT job and the stage
/// ([`crate::framework::stage::StageContext::slurm_job_name`]) and records the
/// name with the job, so a step that outlived the BLUT process that started
/// it can be found again ([`crate::jobs::live_slurm_jobs`]).
pub fn slurm_jobs_named(name: &str) -> Result<Vec<String>> {
    let out = Command::new("squeue")
        .args(["--noheader", "--format=%i", &format!("--name={name}")])
        .output()
        .map_err(|e| TrainError::other(format!("squeue invocation failed: {e}")))?;
    if !out.status.success() {
        return Err(TrainError::other(format!(
            "squeue --name={name} failed (exit {}): {}",
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(parse_squeue_ids(&String::from_utf8_lossy(&out.stdout)))
}

/// `scancel` every Slurm job named `name`. Returns the ids it found first;
/// empty when there was nothing to cancel.
pub fn scancel_named(name: &str) -> Result<Vec<String>> {
    let ids = slurm_jobs_named(name)?;
    if ids.is_empty() {
        return Ok(ids);
    }
    let out = Command::new("scancel")
        .arg(format!("--name={name}"))
        .output()
        .map_err(|e| TrainError::other(format!("scancel invocation failed: {e}")))?;
    if !out.status.success() {
        return Err(TrainError::other(format!(
            "scancel --name={name} failed (exit {}): {}",
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(ids)
}

/// One job id per non-empty line; a step listing (`123.0`) keeps its job id.
fn parse_squeue_ids(stdout: &str) -> Vec<String> {
    let mut ids: Vec<String> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| l.split('.').next().unwrap_or(l).to_string())
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

/// A handle to a submitted Slurm job, returned by
/// [`SlurmLauncher::submit_async`]. Supports polling, log streaming, and
/// cancellation via `scontrol` / `sacct` / `scancel`.
///
/// State comes from `scontrol show job`, which needs no accounting database
/// and knows a job from submission until `MinJobAge` after it ends. Only once
/// the controller has purged the record does polling fall back to `sacct`,
/// which needs `slurmdbd`.
pub struct SlurmJob {
    /// Slurm-assigned numeric job ID.
    job_id: String,
    /// Absolute path of the job's output log, or `None` when the `--output`
    /// pattern passed in `extra` cannot be resolved to one file.
    log_path: Option<PathBuf>,
    /// When `sbatch` was invoked. A log last modified before this is a
    /// previous job's (Slurm reuses job ids after its state is reset or the
    /// counter wraps) and must not be streamed as this one's.
    submitted_at: std::time::SystemTime,
}

/// How long [`SlurmJob::stream`] tolerates consecutive failed state queries (a
/// controller restart, a transient RPC error) before giving up on the job.
const SLURM_POLL_ERROR_GRACE: Duration = Duration::from_secs(300);

impl SlurmJob {
    /// Shell-escape a single argument for safe interpolation into a bash script.
    /// Wraps in single quotes and escapes any embedded single quotes.
    fn shell_escape(arg: &str) -> String {
        if arg.is_empty() {
            return "''".to_string();
        }
        // Fast path: safe characters only.
        if arg
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-+=:@/".contains(&b))
        {
            return arg.to_string();
        }
        // General path: single-quote and escape embedded single quotes.
        format!("'{}'", arg.replace('\'', "'\\''"))
    }

    /// Build an `sbatch` script body from the launcher config + inner command.
    /// The script is submitted via stdin to `sbatch`, so no temp file is
    /// needed. Inner args are shell-escaped to prevent injection.
    fn build_sbatch_script(unit: &str, launcher: &SlurmLauncher, inner: &[String]) -> String {
        let partition = launcher.partition.as_deref();
        let mem = launcher.mem.as_deref();
        let cpus = launcher.cpus;
        let gpus = launcher.gpus;
        let time = launcher.time.as_deref();
        let extra = &launcher.extra;
        let mut lines: Vec<String> = Vec::new();
        lines.push("#!/bin/bash".to_string());
        lines.push(format!("#SBATCH --job-name={unit}"));
        if let Some(p) = partition {
            lines.push(format!("#SBATCH --partition={p}"));
        }
        if let Some(m) = mem {
            lines.push(format!("#SBATCH --mem={m}"));
        }
        if let Some(n) = cpus {
            lines.push(format!("#SBATCH --cpus-per-task={n}"));
        }
        if let Some(g) = gpus {
            lines.push(format!("#SBATCH --gpus={g}"));
        }
        if let Some(t) = time {
            lines.push(format!("#SBATCH --time={t}"));
        }
        if let Some(n) = launcher.nodes {
            lines.push(format!("#SBATCH --nodes={n}"));
        }
        if let Some(n) = launcher.ntasks_per_node {
            lines.push(format!("#SBATCH --ntasks-per-node={n}"));
        }
        // Only emit default --output if extra doesn't already specify one.
        let has_output = extra.iter().any(|x| x.starts_with("--output"));
        if !has_output {
            lines.push("#SBATCH --output=slurm-%j.out".to_string());
        }
        // Append extra flags that are not bare "--".
        for x in extra.iter().filter(|x| *x != "--") {
            lines.push(format!("#SBATCH {x}"));
        }
        lines.push(String::new());
        // The inner command — shell-escape each arg to prevent injection.
        let escaped: Vec<String> = inner.iter().map(|a| Self::shell_escape(a)).collect();
        // A batch script runs once, on the allocation's first node. When the
        // job asks for more than one task, launch the inner command as a job
        // step so it runs once per task on every node; otherwise every node
        // but the first would sit idle.
        let multi_task =
            launcher.nodes.unwrap_or(1) > 1 || launcher.ntasks_per_node.unwrap_or(1) > 1;
        if multi_task {
            // Rendezvous for torchrun's c10d backend: the coordinator is the
            // allocation's first host. NODE_RANK must be read inside each
            // task — in the batch script itself SLURM_NODEID is always 0.
            lines.push(
                "# Multi-task rendezvous: the coordinator is the allocation's first host"
                    .to_string(),
            );
            lines.push(
                "export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames \"$SLURM_JOB_NODELIST\" | head -n1)}"
                    .to_string(),
            );
            lines.push("export MASTER_PORT=${MASTER_PORT:-29500}".to_string());
            lines.push(String::new());
            lines.push(format!(
                "srun --kill-on-bad-exit=1 bash -c 'export NODE_RANK=\"$SLURM_NODEID\"; exec \"$@\"' blut-task {}",
                escaped.join(" ")
            ));
        } else {
            lines.push(escaped.join(" "));
        }
        lines.join("\n")
    }

    /// Where `sbatch` will write the job's output, made absolute against the
    /// submit directory (or `--chdir`). `None` when `extra` sets an `--output`
    /// pattern this can't resolve to one file (anything but `%j`, `%x`, `%%`).
    fn resolve_log_path(
        launcher: &SlurmLauncher,
        unit: &str,
        job_id: &str,
        submit_dir: &std::path::Path,
    ) -> Option<PathBuf> {
        let flag_value = |names: &[&str]| -> Option<String> {
            launcher.extra.iter().find_map(|x| {
                names
                    .iter()
                    .find_map(|n| x.strip_prefix(n).map(str::to_string))
            })
        };
        let base = match flag_value(&["--chdir=", "-D="]) {
            Some(d) => submit_dir.join(d),
            None => submit_dir.to_path_buf(),
        };
        let pattern = flag_value(&["--output=", "-o="]).unwrap_or_else(|| "slurm-%j.out".into());
        let mut out = String::new();
        let mut chars = pattern.chars();
        while let Some(c) = chars.next() {
            if c != '%' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('j') => out.push_str(job_id),
                Some('x') => out.push_str(unit),
                Some('%') => out.push('%'),
                // %N, %t, %A, … expand per node/task/array: no single file.
                _ => return None,
            }
        }
        Some(base.join(out))
    }

    /// Map a Slurm job state name (as printed by `scontrol` or `sacct -P`) to a
    /// [`JobState`]. Accepts `sacct`'s decorations: `CANCELLED by 1000`, and
    /// the `+` it appends when a fixed-width column truncates the name.
    fn classify_state(raw: &str) -> JobState {
        let name = raw
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_end_matches('+');
        match name {
            "COMPLETED" => JobState::Succeeded,
            "CANCELLED" => JobState::Cancelled,
            "TIMEOUT" => JobState::Failed("TIMEOUT".to_string()),
            "OUT_OF_MEMORY" => JobState::Failed("OutOfMemory".to_string()),
            "FAILED" | "NODE_FAIL" | "BOOT_FAIL" | "DEADLINE" | "PREEMPTED" | "REVOKED"
            | "SPECIAL_EXIT" => JobState::Failed(raw.trim().to_string()),
            // Queued, starting, running, or between states (requeue, resize,
            // stage-out, epilog): not finished, keep waiting.
            "PENDING" | "CONFIGURING" | "RUNNING" | "COMPLETING" | "SUSPENDED" | "STOPPED"
            | "REQUEUED" | "REQUEUE_HOLD" | "REQUEUE_FED" | "RESIZING" | "SIGNALING"
            | "STAGE_OUT" | "RESV_DEL_HOLD" | "POWER_UP_NODE" | "UPDATE_DB" => JobState::Running,
            _ => JobState::Unknown(raw.trim().to_string()),
        }
    }

    /// Pull `JobState=…` out of one-line `scontrol show job -o` output.
    fn parse_scontrol_state(output: &str) -> Option<String> {
        Self::scontrol_field(output, "JobState")
    }

    /// One `Key=value` field of `scontrol show job -o` output. Values with
    /// spaces are not split by Slurm in this format (it uses `_`).
    fn scontrol_field(output: &str, key: &str) -> Option<String> {
        let prefix = format!("{key}=");
        output
            .split_whitespace()
            .find_map(|kv| kv.strip_prefix(prefix.as_str()))
            .map(str::to_string)
    }

    /// The job's raw scheduler state and, when the controller still knows the
    /// job, why it is in it.
    fn raw_state(&self) -> Result<String> {
        self.raw_status().map(|(state, _)| state)
    }

    /// `(state, reason)`. `scontrol` first (works without accounting);
    /// `sacct` only after the controller has purged the record, and it gives
    /// no reason.
    fn raw_status(&self) -> Result<(String, Option<String>)> {
        let ctl = Command::new("scontrol")
            .args(["show", "job", "-o", &self.job_id])
            .output()
            .map_err(|e| TrainError::other(format!("scontrol invocation failed: {e}")))?;
        let ctl_stdout = String::from_utf8_lossy(&ctl.stdout);
        if ctl.status.success()
            && let Some(state) = Self::parse_scontrol_state(&ctl_stdout)
        {
            let reason = Self::scontrol_field(&ctl_stdout, "Reason");
            return Ok((state, reason));
        }
        let ctl_stderr = String::from_utf8_lossy(&ctl.stderr);
        if !ctl_stderr.contains("Invalid job id") {
            return Err(TrainError::other(format!(
                "scontrol show job {} failed (exit {}): {}",
                self.job_id,
                ctl.status.code().map_or("signal".into(), |c| c.to_string()),
                ctl_stderr.trim()
            )));
        }
        // The controller no longer knows the job: ask accounting.
        let acct = Command::new("sacct")
            .args(["-X", "-P", "-n", "-o", "State", "-j", &self.job_id])
            .output()
            .map_err(|e| TrainError::other(format!("sacct invocation failed: {e}")))?;
        let acct_stdout = String::from_utf8_lossy(&acct.stdout);
        match acct_stdout.lines().map(str::trim).find(|l| !l.is_empty()) {
            Some(state) if acct.status.success() => Ok((state.to_string(), None)),
            _ => Err(TrainError::other(format!(
                "slurm job {} is unknown to the controller (purged after MinJobAge, or never \
                 existed) and sacct has no record of it: {}",
                self.job_id,
                String::from_utf8_lossy(&acct.stderr).trim()
            ))),
        }
    }
}

impl SlurmJob {
    /// A line for the log stream when the job's situation changes in a way
    /// its own output can't show: it was requeued, suspended or held, or it
    /// is stuck waiting for a reason other than ordinary queueing. `None`
    /// while it simply waits its turn or runs.
    fn status_note(
        now: &(String, Option<String>),
        before: Option<&(String, Option<String>)>,
        has_run: bool,
    ) -> Option<String> {
        if before == Some(now) {
            return None;
        }
        let (state, reason) = now;
        let reason = reason.as_deref().filter(|r| *r != "None");
        let describe = || match reason {
            Some(r) => format!("{state} ({})", r.replace('_', " ")),
            None => state.clone(),
        };
        match state.as_str() {
            "RUNNING" if has_run && before.is_some_and(|(s, _)| s != "RUNNING") => {
                Some("running again".to_string())
            }
            "RUNNING" | "COMPLETING" | "CONFIGURING" => None,
            "PENDING" if has_run => Some(format!("requeued, now {}", describe())),
            // Ordinary queueing: waiting for its turn or for free resources.
            "PENDING" if matches!(reason, None | Some("Priority" | "Resources" | "BeginTime")) => {
                None
            }
            _ if Self::classify_state(state) == JobState::Running => Some(describe()),
            _ => None,
        }
    }

    /// Append whatever is new in the log since `offset` to `sink`. A log that
    /// does not exist yet (job still queued) is not an error.
    fn drain_log(&self, offset: &mut u64, sink: &dyn Fn(&str)) -> Result<()> {
        use std::io::{Seek, SeekFrom};
        let Some(path) = self.log_path.as_ref() else {
            return Ok(());
        };
        let io_err = |source| TrainError::Io {
            path: path.clone(),
            source,
        };
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(io_err(e)),
        };
        let meta = file.metadata().map_err(io_err)?;
        if meta.modified().is_ok_and(|m| m < self.submitted_at) {
            // Still the previous owner's file: this job has not opened it yet.
            return Ok(());
        }
        if meta.len() < *offset {
            // Truncated under us (the job opened a file we had already read
            // from): everything in it now is this job's.
            *offset = 0;
        }
        let mut reader = std::io::BufReader::new(file);
        reader.seek(SeekFrom::Start(*offset)).map_err(io_err)?;
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader.read_line(&mut line).map_err(io_err)?;
            // Stop at EOF, and before a partial last line: it is re-read whole
            // on the next pass instead of being split in two.
            if n == 0 || !line.ends_with('\n') {
                return Ok(());
            }
            *offset += n as u64;
            let trimmed = line.trim_end_matches('\n');
            if !trimmed.is_empty() {
                sink(trimmed);
            }
        }
    }
}

impl RemoteJob for SlurmJob {
    fn id(&self) -> &str {
        &self.job_id
    }

    fn poll(&self) -> Result<JobState> {
        Ok(Self::classify_state(&self.raw_state()?))
    }

    fn stream(&self, sink: &dyn Fn(&str)) -> Result<()> {
        // Tail-follow the log, polling the scheduler every 2s, until the job
        // reaches a terminal state. There is deliberately no silence timeout:
        // a job may queue for hours or train quietly, and giving up here would
        // report a failure while the job kept its allocation. The job's
        // `--time` limit is what bounds it.
        let poll_interval = Duration::from_secs(2);
        let mut offset: u64 = 0;
        let mut failing_since: Option<std::time::Instant> = None;
        if self.log_path.is_none() {
            sink(&format!(
                "[blut] slurm job {}: the --output pattern names more than one file; \
                 log lines will not be streamed",
                self.job_id
            ));
        }
        let mut last_seen: Option<(String, Option<String>)> = None;
        let mut has_run = false;
        loop {
            self.drain_log(&mut offset, sink)?;
            let state = match self.raw_status() {
                Ok(status) => {
                    failing_since = None;
                    if let Some(note) = Self::status_note(&status, last_seen.as_ref(), has_run) {
                        sink(&format!("[blut] slurm job {}: {note}", self.job_id));
                    }
                    has_run |= status.0 == "RUNNING";
                    let state = Self::classify_state(&status.0);
                    last_seen = Some(status);
                    state
                }
                Err(e) => {
                    let since = *failing_since.get_or_insert_with(std::time::Instant::now);
                    if since.elapsed() > SLURM_POLL_ERROR_GRACE {
                        return Err(TrainError::other(format!(
                            "slurm job {}: could not read its state for {}s (it may still be \
                             running; check `squeue -j {0}`): {e}",
                            self.job_id,
                            SLURM_POLL_ERROR_GRACE.as_secs()
                        )));
                    }
                    JobState::Running
                }
            };
            if state == JobState::Running {
                thread::sleep(poll_interval);
                continue;
            }
            // Terminal: pick up lines written between the last read and now.
            self.drain_log(&mut offset, sink)?;
            return match state {
                JobState::Succeeded => Ok(()),
                JobState::Failed(reason) => Err(TrainError::other(format!(
                    "slurm job {} failed: {reason}",
                    self.job_id,
                ))),
                JobState::Cancelled => Err(TrainError::other(format!(
                    "slurm job {} was cancelled",
                    self.job_id,
                ))),
                other => Err(TrainError::other(format!(
                    "slurm job {} ended in a state blut does not recognise: {other:?}",
                    self.job_id,
                ))),
            };
        }
    }

    fn cancel(&self) -> Result<()> {
        let out = Command::new("scancel")
            .arg(&self.job_id)
            .output()
            .map_err(|e| TrainError::other(format!("scancel invocation failed: {e}")))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(TrainError::other(format!(
                "scancel {} failed (exit {}): {}",
                self.job_id,
                out.status.code().unwrap_or(-1),
                stderr.trim()
            )));
        }
        Ok(())
    }
}

impl SlurmLauncher {
    /// Submit a Slurm job asynchronously via `sbatch --parsable`. Returns a
    /// [`SlurmJob`] handle for polling, log streaming, and cancellation.
    ///
    /// The sbatch script is generated in-memory and piped to `sbatch` via
    /// stdin, so no temporary file is left on disk.
    pub fn submit_async(&self, unit: &str, inner: &[String]) -> Result<Box<dyn RemoteJob>> {
        if inner.is_empty() {
            return Err(TrainError::other("slurm submit_async: empty inner command"));
        }
        let script = SlurmJob::build_sbatch_script(unit, self, inner);
        // Truncate to whole seconds: some filesystems store mtimes that
        // coarsely, and the job's first write must not compare as older.
        let submitted_at = {
            let now = std::time::SystemTime::now();
            let secs = now
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            std::time::UNIX_EPOCH + Duration::from_secs(secs)
        };
        use std::io::Write;
        let mut child = Command::new("sbatch")
            .arg("--parsable")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| TrainError::other(format!("sbatch spawn failed: {e}")))?;
        if let Some(ref mut stdin) = child.stdin {
            stdin
                .write_all(script.as_bytes())
                .map_err(|e| TrainError::other(format!("sbatch stdin write failed: {e}")))?;
        }
        let output = child
            .wait_with_output()
            .map_err(|e| TrainError::other(format!("sbatch wait failed: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TrainError::other(format!(
                "sbatch failed (exit {}): {}",
                output.status.code().unwrap_or(-1),
                stderr.trim()
            )));
        }
        // --parsable output is "<job_id>" or "<job_id>;<cluster>".
        let raw = String::from_utf8_lossy(&output.stdout);
        let job_id = raw
            .trim()
            .split(';')
            .next()
            .unwrap_or(raw.trim())
            .to_string();
        if job_id.is_empty() {
            return Err(TrainError::other("sbatch --parsable returned empty job ID"));
        }
        // sbatch resolves a relative --output against the directory it was run
        // from; pin that now so a later chdir by the caller can't break streaming.
        let submit_dir = std::env::current_dir().map_err(|e| TrainError::Io {
            path: PathBuf::from("."),
            source: e,
        })?;
        let log_path = SlurmJob::resolve_log_path(self, unit, &job_id, &submit_dir);
        Ok(Box::new(SlurmJob {
            job_id,
            log_path,
            submitted_at,
        }))
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
    /// GPUs the entrypoint reserves — emitted as `--entrypoint-num-gpus` so the
    /// Ray scheduler places the job on a node with that many GPUs (ADR 0087
    /// distributed tail; mirrors [`SlurmLauncher::gpus`]). `None` ⇒ no flag, so
    /// the submitted command is byte-identical to the pre-ADR path.
    pub num_gpus: Option<u32>,
    /// Verbatim passthrough flags appended before the `--` separator.
    pub extra: Vec<String>,
}

impl RayLauncher {
    /// `ray job submit` arguments, shared by the blocking [`Launcher::wrap`]
    /// path and [`RayLauncher::submit_async`] so the two cannot drift: the
    /// async path used to omit `--entrypoint-num-gpus`, so a job submitted
    /// through it reserved no GPU and could be placed on a CPU-only node.
    fn job_submit_args(&self, unit: &str, inner: &[String], no_wait: bool) -> Vec<String> {
        let mut args = vec!["job".to_string(), "submit".to_string()];
        if no_wait {
            args.push("--no-wait".to_string());
        }
        args.push(format!("--submission-id={unit}"));
        if let Some(a) = &self.address {
            args.push(format!("--address={a}"));
        }
        if let Some(re) = &self.runtime_env {
            args.push("--runtime-env-json".to_string());
            args.push(re.clone());
        }
        // ADR 0087: a GPU-bearing unit reserves `count` GPUs on the placed node.
        if let Some(g) = self.num_gpus {
            args.push(format!("--entrypoint-num-gpus={g}"));
        }
        args.extend(self.extra.iter().filter(|x| *x != "--").cloned());
        args.push("--".to_string());
        args.extend(inner.iter().cloned());
        args
    }
}

impl Launcher for RayLauncher {
    fn wrap(&self, unit: &str, inner: &[String]) -> Result<WrappedCommand> {
        if inner.is_empty() {
            return Err(TrainError::other("ray launcher: empty inner command"));
        }
        let args = self.job_submit_args(unit, inner, false);
        Ok(WrappedCommand {
            program: "ray".to_string(),
            args,
            env: Vec::new(),
            kind: LauncherKind::Ray,
        })
    }
}

/// A handle to a submitted Ray job, returned by
/// [`RayLauncher::submit_async`]. Supports polling, log streaming, and
/// cancellation via `ray job status` / `ray job stop` / `ray job logs`.
pub struct RayJob {
    /// Ray submission ID (typically the unit name).
    submission_id: String,
    /// Optional `--address` of the Ray head node.
    address: Option<String>,
}

impl RayJob {
    /// Build the optional `--address` flag slice for Ray CLI commands.
    fn address_args(&self) -> Vec<String> {
        match &self.address {
            Some(a) => vec![format!("--address={a}")],
            None => vec![],
        }
    }

    /// Parse `ray job status <id>` output into a [`JobState`].
    ///
    /// The shapes below were CAPTURED from `ray job status` against a live
    /// two-node cluster (Ray 2.x), not inferred. This previously scanned for a
    /// `Status:` line and a bare `SUCCEEDED` token, which Ray does not emit for
    /// a terminal job -- so every poll of a finished job returned
    /// `Unknown("could not parse ray job status")`, and the launcher could
    /// never observe a job succeed or fail. The four real forms are:
    ///
    /// ```text
    /// SUCC cli.py:64 -- Job 'my-id' succeeded
    /// ERR  cli.py:72 -- Job 'my-id' failed
    /// WARN cli.py:82 -- Job 'my-id' was stopped
    /// INFO cli.py:89 -- Status for job 'my-id': RUNNING
    /// ```
    ///
    /// Scanning stops at the `Status message:` line: for a failure that message
    /// embeds the tail of the job's own logs, which can contain anything at all
    /// -- including the word `succeeded`, or a child job's own
    /// `Status for job '...': SUCCEEDED` line.
    fn parse_ray_status(output: &str) -> JobState {
        for line in output.lines() {
            if line.contains("Status message:") {
                break;
            }
            // `Status for job '<id>': <STATE>` — the non-terminal form.
            if let Some((_, rest)) = line.split_once("Status for job ")
                && let Some((_, state)) = rest.rsplit_once(": ")
            {
                return match state.trim() {
                    "SUCCEEDED" => JobState::Succeeded,
                    "FAILED" => JobState::Failed("FAILED".to_string()),
                    "RUNNING" | "PENDING" | "WAITING" | "CONSTRUCTOR" => JobState::Running,
                    "STOPPED" | "CANCELLED" => JobState::Cancelled,
                    other => JobState::Unknown(other.to_string()),
                };
            }
            // Terminal prose.
            if !line.contains("Job '") {
                continue;
            }
            if line.contains("' succeeded") {
                return JobState::Succeeded;
            }
            if line.contains("' failed") {
                return JobState::Failed("FAILED".to_string());
            }
            if line.contains("' was stopped") {
                return JobState::Cancelled;
            }
        }
        JobState::Unknown("could not parse ray job status".to_string())
    }
}

impl RemoteJob for RayJob {
    fn id(&self) -> &str {
        &self.submission_id
    }

    fn poll(&self) -> Result<JobState> {
        let mut cmd = Command::new("ray");
        cmd.args(["job", "status", &self.submission_id]);
        cmd.args(self.address_args());
        let output = cmd
            .output()
            .map_err(|e| TrainError::other(format!("ray job status failed: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TrainError::other(format!(
                "ray job status {} failed (exit {}): {}",
                self.submission_id,
                output.status.code().unwrap_or(-1),
                stderr.trim()
            )));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(Self::parse_ray_status(&stdout))
    }

    fn stream(&self, sink: &dyn Fn(&str)) -> Result<()> {
        let mut cmd = Command::new("ray");
        cmd.args(["job", "logs", &self.submission_id, "--follow"]);
        cmd.args(self.address_args());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        let mut child = cmd
            .spawn()
            .map_err(|e| TrainError::other(format!("ray job logs failed: {e}")))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| TrainError::other("ray job logs: could not capture stdout"))?;
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let line =
                line.map_err(|e| TrainError::other(format!("ray job logs read error: {e}")))?;
            sink(&line);
        }
        // After the log stream ends, check exit code.
        let status = child
            .wait()
            .map_err(|e| TrainError::other(format!("ray job logs wait failed: {e}")))?;
        if !status.success() {
            let code = match status.code() {
                Some(c) => c.to_string(),
                None => "signal".to_string(),
            };
            return Err(TrainError::other(format!(
                "ray job logs {} exited with {code}",
                self.submission_id,
            )));
        }
        Ok(())
    }

    fn cancel(&self) -> Result<()> {
        let mut cmd = Command::new("ray");
        cmd.args(["job", "stop", &self.submission_id]);
        cmd.args(self.address_args());
        let output = cmd
            .output()
            .map_err(|e| TrainError::other(format!("ray job stop failed: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TrainError::other(format!(
                "ray job stop {} failed (exit {}): {}",
                self.submission_id,
                output.status.code().unwrap_or(-1),
                stderr.trim()
            )));
        }
        Ok(())
    }
}

impl RayLauncher {
    /// Submit a Ray job asynchronously via `ray job submit --no-wait`. Returns
    /// a [`RayJob`] handle for polling, log streaming, and cancellation.
    pub fn submit_async(&self, unit: &str, inner: &[String]) -> Result<Box<dyn RemoteJob>> {
        if inner.is_empty() {
            return Err(TrainError::other("ray submit_async: empty inner command"));
        }
        let args = self.job_submit_args(unit, inner, true);
        let output = Command::new("ray")
            .args(&args)
            .output()
            .map_err(|e| TrainError::other(format!("ray job submit failed: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TrainError::other(format!(
                "ray job submit failed (exit {}): {}",
                output.status.code().unwrap_or(-1),
                stderr.trim()
            )));
        }
        Ok(Box::new(RayJob {
            submission_id: unit.to_string(),
            address: self.address.clone(),
        }))
    }
}

/// Where to place a unit of work, selected by `--launcher`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LaunchTarget {
    /// This box, broker-gated + cgroup-contained (the memory-admission path).
    #[default]
    Local,
    /// A Slurm allocation (`srun`).
    Slurm,
    /// A Ray job (`ray job submit`).
    Ray,
    /// P2P distributed compute over the `p2p node` mesh. There is no
    /// [`Launcher`] for it: a stage reaches a peer only through an
    /// `ExecutionAdapter` installed with `ExecCtx::with_execution_adapter`
    /// (feature `p2p`). `recipe run` and `recipe declare --run` install the
    /// mesh adapter (`p2p::mesh_exec`); every other command refuses
    /// `--launcher p2p`, and the executor refuses this target when no adapter
    /// is installed. A stage that declares it may leave the machine runs on a
    /// peer with the same code identity, or the run fails; the others run here.
    P2P,
    /// A Kubernetes cluster via a generated `BlutPlan` (`kubectl apply -f -`).
    /// K8s is an adapter (ADR 0037): the engine emits the manifest textually
    /// and shells `kubectl` — zero kube deps in the engine; the operator crate
    /// owns the reconcile loop.
    K8s,
    /// Cloud compute queue (object-store dispatch, billed on compute — ADR 0082).
    /// Like [`P2P`](Self::P2P) it has no [`Launcher`]: stages reach workers
    /// through an `ExecutionAdapter` (`cloud::submitter::CloudSubmitter` over
    /// the durable `cloud::object_queue::ObjectQueue`). `recipe run`, `recipe
    /// declare --run` and `plan resume` install it (feature `cloud`, queue in
    /// `BLUT_CLOUD_QUEUE`; see [`check_cloud_launch`]); every other command
    /// refuses `--launcher cloud`, and the executor refuses it without an
    /// adapter.
    Cloud,
}

impl LaunchTarget {
    /// The `--launcher` value that selects this target; `parse` round-trips it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Slurm => "slurm",
            Self::Ray => "ray",
            Self::P2P => "p2p",
            Self::K8s => "k8s",
            Self::Cloud => "cloud",
        }
    }
}

impl std::str::FromStr for LaunchTarget {
    type Err = TrainError;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "local" | "" => Ok(Self::Local),
            "slurm" => Ok(Self::Slurm),
            "ray" => Ok(Self::Ray),
            "p2p" => Ok(Self::P2P),
            "k8s" | "kubernetes" => Ok(Self::K8s),
            "cloud" => Ok(Self::Cloud),
            other => Err(TrainError::other(format!(
                "unknown launcher '{other}' (expected local|slurm|ray|p2p|k8s|kubernetes|cloud)"
            ))),
        }
    }
}

/// The numeric launcher env vars [`launcher_for`] reads for `target`.
fn numeric_launcher_env(target: LaunchTarget) -> &'static [&'static str] {
    match target {
        LaunchTarget::Slurm => &[
            "BLUT_SLURM_CPUS",
            "BLUT_SLURM_GPUS",
            "BLUT_SLURM_NODES",
            "BLUT_SLURM_NTASKS_PER_NODE",
        ],
        LaunchTarget::Ray => &["BLUT_RAY_NUM_GPUS"],
        _ => &[],
    }
}

/// Reject launcher env vars [`launcher_for`] would otherwise drop: a numeric
/// setting that is not a positive integer (`BLUT_SLURM_NODES=two`,
/// `BLUT_SLURM_GPUS=4x`) would silently fall back to the default allocation.
/// Call it before any work starts.
pub fn check_launcher_env(target: LaunchTarget) -> Result<()> {
    for key in numeric_launcher_env(target) {
        // Empty is unset, as in `launcher_for`.
        if let Ok(value) = std::env::var(key).map(|v| v.trim().to_string())
            && !value.is_empty()
        {
            match value.parse::<u32>() {
                Ok(n) if n > 0 => {}
                _ => {
                    return Err(TrainError::other(format!(
                        "{key}={value:?} is not a positive integer; fix or unset it"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Refuse a launcher whose client commands are not installed. `recipe run
/// --launcher slurm` on a host with no Slurm client used to run every stage on
/// the local machine and report success, because only launcher-aware stages
/// ever reach `sbatch`. `docs/SLURM.md` lists the client commands as a
/// requirement of the submit host.
pub(crate) fn check_launcher_tools(target: LaunchTarget) -> Result<()> {
    let required: &[&str] = match target {
        LaunchTarget::Slurm => &["sbatch", "scontrol"],
        _ => &[],
    };
    let on_path = |tool: &str| {
        std::env::var_os("PATH")
            .is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join(tool).is_file()))
    };
    let missing: Vec<&str> = required.iter().copied().filter(|t| !on_path(t)).collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(TrainError::other(format!(
            "--launcher {} needs {} on PATH, and this host has no {}; run on a submit host \
             with the client installed (see docs/SLURM.md), or use --launcher local",
            target.as_str(),
            required.join(", "),
            missing.join(", ")
        )))
    }
}

/// Refuse a target that no `blut` command can place work on. `p2p` and
/// `cloud` parse, because a recorded job or `dataset resolve` may name them,
/// but nothing hands a run's stages to their execution adapters: `recipe run
/// --launcher p2p` used to run every stage on this machine and record `p2p`
/// as the job's placement. `surface` names what was asked for, as in
/// `refuse_surface` for k8s. Call it before any work starts.
pub(crate) fn check_launcher_placeable(target: LaunchTarget, surface: &str) -> Result<()> {
    let detail = match target {
        LaunchTarget::P2P => {
            "only `recipe run` and `recipe declare --run` place a run's stages on mesh \
             peers, in a build with the `p2p` feature: `blut recipe run <name> --launcher \
             p2p --p2p-seed <addr@pubkey>` joins the mesh and fails if no peer can take an \
             eligible stage. A library caller installs `ExecCtx::with_execution_adapter`"
        }
        LaunchTarget::Cloud => {
            "only `recipe run`, `recipe declare --run` and `plan resume` submit a run's \
             stages to the cloud queue (a build with the `cloud` feature, the queue in \
             BLUT_CLOUD_QUEUE, workers started with `blut cloud worker`)"
        }
        LaunchTarget::Local | LaunchTarget::Slurm | LaunchTarget::Ray | LaunchTarget::K8s => {
            return Ok(());
        }
    };
    Err(TrainError::other(format!(
        "--launcher {} cannot place {surface}: {detail}. Nothing ran. Use --launcher local \
         to run on this machine",
        target.as_str()
    )))
}

/// Environment variable naming the cloud queue `--launcher cloud` submits to:
/// `s3://bucket[/prefix]` (a build with the `s3` feature; credentials from the
/// `AWS_*` variables only), `file:///dir`, or a directory. Workers poll the
/// same location (`blut cloud worker --queue`).
pub const CLOUD_QUEUE_ENV: &str = "BLUT_CLOUD_QUEUE";

/// Environment variable bounding how long a submitted stage may wait for a
/// worker to claim it, in seconds ([`DEFAULT_CLOUD_PICKUP_SECS`] when unset).
pub const CLOUD_PICKUP_ENV: &str = "BLUT_CLOUD_PICKUP_SECS";

/// Pickup window when [`CLOUD_PICKUP_ENV`] is unset.
pub const DEFAULT_CLOUD_PICKUP_SECS: u64 = 300;

/// The pickup window `--launcher cloud` uses: [`CLOUD_PICKUP_ENV`], or the
/// default. A value that is not a positive whole number is an error, never
/// silently the default.
pub fn cloud_pickup_window() -> Result<std::time::Duration> {
    match std::env::var(CLOUD_PICKUP_ENV)
        .map(|v| v.trim().to_string())
        .ok()
        .filter(|v| !v.is_empty())
    {
        None => Ok(std::time::Duration::from_secs(DEFAULT_CLOUD_PICKUP_SECS)),
        Some(value) => match value.parse::<u64>() {
            Ok(secs) if secs > 0 => Ok(std::time::Duration::from_secs(secs)),
            _ => Err(TrainError::other(format!(
                "{CLOUD_PICKUP_ENV}={value:?} is not a positive whole number of seconds; fix \
                 or unset it"
            ))),
        },
    }
}

/// Whether `--launcher cloud` can place `surface` (`recipe run`, `recipe
/// declare --run`, `plan resume`): this binary has the `cloud` feature, and
/// [`CLOUD_QUEUE_ENV`] names a queue. Says what is missing and that nothing
/// ran. Call it before any work starts; the queue itself is opened and probed
/// when the run starts.
pub fn check_cloud_launch(surface: &str) -> Result<()> {
    if !cfg!(feature = "cloud") {
        return Err(TrainError::other(format!(
            "--launcher cloud cannot place {surface}: this binary was built without the \
             `cloud` feature (`s3` for an s3:// queue). Nothing ran. Use --launcher local \
             to run on this machine"
        )));
    }
    let queue = std::env::var(CLOUD_QUEUE_ENV).unwrap_or_default();
    if queue.trim().is_empty() {
        return Err(TrainError::other(format!(
            "--launcher cloud cannot place {surface}: set {CLOUD_QUEUE_ENV} to the queue the \
             workers poll (s3://bucket[/prefix] or a directory, the value given to \
             `blut cloud worker --queue`). Nothing ran"
        )));
    }
    cloud_pickup_window().map(|_| ())
}

/// The configured [`Launcher`] for a target. Slurm/Ray read their cluster
/// config from env (`BLUT_SLURM_*` / `RAY_ADDRESS` / `BLUT_RAY_RUNTIME_ENV`) so
/// the DAG itself stays portable — the same recipe runs locally or on a cluster
/// by flipping `--launcher`, no recipe edit.
pub fn launcher_for(target: LaunchTarget) -> Box<dyn Launcher> {
    // An empty value is unset: `RAY_ADDRESS=` used to submit `--address=`.
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let env_u32 = |k: &str| env_count(k, env(k));
    match target {
        LaunchTarget::Local => Box::new(LocalSystemd::default()),
        LaunchTarget::Slurm => Box::new(SlurmLauncher {
            partition: env("BLUT_SLURM_PARTITION"),
            mem: env("BLUT_SLURM_MEM"),
            cpus: env_u32("BLUT_SLURM_CPUS"),
            gpus: env_u32("BLUT_SLURM_GPUS"),
            time: env("BLUT_SLURM_TIME"),
            nodes: env_u32("BLUT_SLURM_NODES"),
            ntasks_per_node: env_u32("BLUT_SLURM_NTASKS_PER_NODE"),
            extra: Vec::new(),
        }),
        LaunchTarget::Ray => Box::new(RayLauncher {
            address: env("RAY_ADDRESS"),
            runtime_env: env("BLUT_RAY_RUNTIME_ENV"),
            num_gpus: env_u32("BLUT_RAY_NUM_GPUS"),
            extra: Vec::new(),
        }),
        // P2P, k8s and cloud have no Launcher, and LocalSystemd here is not a
        // placement: this signature has no error path, so it returns the local
        // launcher. A k8s run is a BlutPlan submitted by `recipe run` (see
        // `k8s_cluster`); p2p and cloud need an ExecutionAdapter
        // (`ExecCtx::with_execution_adapter`). The cloud launch of `recipe run`,
        // `recipe declare --run` and `plan resume` and the p2p launch of `recipe
        // run` and `recipe declare --run` install one. Under p2p the stages that
        // run here are the ones that declare they must stay, contained like any
        // local stage. Every other command refuses these targets before a stage
        // could reach this (`check_launcher_placeable`,
        // `k8s_cluster::refuse_surface`), and the executor refuses them without
        // an adapter.
        LaunchTarget::P2P | LaunchTarget::K8s | LaunchTarget::Cloud => {
            Box::new(LocalSystemd::default())
        }
    }
}

/// A whole-number launcher setting (`BLUT_RAY_NUM_GPUS`, `BLUT_SLURM_GPUS`, ...).
/// An unparseable value used to vanish, so `BLUT_RAY_NUM_GPUS=two` submitted
/// the job with no GPU request at all. It is still treated as unset (the
/// signature of [`launcher_for`] has no error path), but loudly.
fn env_count(name: &str, value: Option<String>) -> Option<u32> {
    let value = value?;
    match value.trim().parse::<u32>() {
        Ok(n) => Some(n),
        Err(_) => {
            tracing::warn!(
                "{name}={value:?} is not a whole number; IGNORING it, so the job is \
                 submitted without that request. Fix or unset {name}."
            );
            eprintln!(
                "warning: {name}={value:?} is not a whole number; the job is submitted \
                 without it"
            );
            None
        }
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
            nodes: None,
            ntasks_per_node: None,
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
        assert_eq!(LaunchTarget::from_str("k8s").unwrap(), LaunchTarget::K8s);
        assert!(LaunchTarget::from_str("mesos").is_err());
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
            num_gpus: None,
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
        // num_gpus None ⇒ no GPU flag (byte-identical to the pre-ADR path).
        assert!(!a.iter().any(|x| x.starts_with("--entrypoint-num-gpus")));
        assert!(RayLauncher::default().build_command("u", &[]).is_err());
    }

    #[test]
    fn ray_emits_entrypoint_num_gpus_when_set() {
        // ADR 0087: a GPU-bearing unit reserves `count` GPUs, and the flag lands
        // BEFORE the `--` separator (it's a `ray job submit` flag, not a job arg).
        let l = RayLauncher {
            num_gpus: Some(2),
            ..Default::default()
        };
        let c = l.build_command("job-g", &["python".into()]).unwrap();
        let a = argv(&c);
        let gpu = a
            .iter()
            .position(|x| x == "--entrypoint-num-gpus=2")
            .expect("the GPU flag is emitted");
        let sep = a.iter().position(|x| x == "--").unwrap();
        assert!(gpu < sep, "the flag precedes the `--` separator");
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
    fn local_systemd_caps_parse_to_bytes() {
        const G: u64 = 1024 * 1024 * 1024;
        let caps = LocalSystemd::default().cap_spec().unwrap();
        assert_eq!(caps.mem_max, Some(44 * G));
        assert_eq!(caps.mem_high, Some(40 * G));
        assert_eq!(caps.swap_max, Some(12 * G));

        let unset = LocalSystemd {
            mem_high: String::new(),
            ..LocalSystemd::default()
        };
        assert_eq!(unset.cap_spec().unwrap().mem_high, None, "empty is unset");

        for bad in ["lots", "0", "-1G"] {
            let l = LocalSystemd {
                mem_max: bad.to_string(),
                ..LocalSystemd::default()
            };
            let err = l.cap_spec().expect_err(bad).to_string();
            assert!(err.contains("mem_max"), "{bad}: {err}");
        }
    }

    #[test]
    fn unit_names_are_sanitized_for_every_backend() {
        assert_eq!(sanitize_unit("blut-job_1"), "blut-job_1");
        assert_eq!(sanitize_unit("job 7/train.model"), "job-7-train-model");
        assert_eq!(sanitize_unit(""), "blut");
    }

    #[tokio::test]
    async fn spawn_contained_refuses_an_empty_command() {
        let err = LocalSystemd::default()
            .spawn_contained("u", &[], std::path::Path::new("."), &[], |_| {})
            .await
            .expect_err("empty inner");
        assert!(err.to_string().contains("empty inner command"), "{err}");
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

    // --- RemoteJob: Slurm state parsing ---
    //
    // Every input string below was captured from Slurm 21.08 (Ubuntu 22.04
    // packages) on a two-node test cluster, not written from the docs.

    #[test]
    fn slurm_scontrol_state_is_read_from_the_one_line_record() {
        // `scontrol show job -o <id>` (abridged; the real line is ~1.5 KB).
        let line = "JobId=2 JobName=wrap UserId=root(0) GroupId=root(0) Priority=1 \
                    JobState=FAILED Reason=NonZeroExitCode Dependency=(null) ExitCode=3:0";
        assert_eq!(
            SlurmJob::parse_scontrol_state(line).as_deref(),
            Some("FAILED")
        );
        assert_eq!(SlurmJob::parse_scontrol_state(""), None);
    }

    #[test]
    fn slurm_states_classify_as_terminal_or_waiting() {
        let cases = [
            ("COMPLETED", JobState::Succeeded),
            ("PENDING", JobState::Running),
            ("RUNNING", JobState::Running),
            // Epilog still running: not finished, exit status not final.
            ("COMPLETING", JobState::Running),
            ("REQUEUED", JobState::Running),
            ("CANCELLED", JobState::Cancelled),
            ("FAILED", JobState::Failed("FAILED".into())),
            ("TIMEOUT", JobState::Failed("TIMEOUT".into())),
            ("OUT_OF_MEMORY", JobState::Failed("OutOfMemory".into())),
            ("NODE_FAIL", JobState::Failed("NODE_FAIL".into())),
            ("PREEMPTED", JobState::Failed("PREEMPTED".into())),
        ];
        for (raw, want) in cases {
            assert_eq!(SlurmJob::classify_state(raw), want, "{raw}");
        }
    }

    #[test]
    fn slurm_sacct_decorations_do_not_hide_the_state() {
        // `sacct -X -P -o State` for a job cancelled by uid 0.
        assert_eq!(
            SlurmJob::classify_state("CANCELLED by 0"),
            JobState::Cancelled
        );
        // Default fixed-width `sacct` truncates that to "CANCELLED+".
        assert_eq!(SlurmJob::classify_state("CANCELLED+"), JobState::Cancelled);
        assert_eq!(SlurmJob::classify_state(" COMPLETED "), JobState::Succeeded);
    }

    #[test]
    fn slurm_stream_notes_a_requeue_but_not_ordinary_queueing() {
        let s = |state: &str, reason: &str| (state.to_string(), Some(reason.to_string()));
        // Waiting its turn, then running: nothing to add to the job's own log.
        assert_eq!(
            SlurmJob::status_note(&s("PENDING", "Priority"), None, false),
            None
        );
        assert_eq!(
            SlurmJob::status_note(
                &s("RUNNING", "None"),
                Some(&s("PENDING", "Resources")),
                false
            ),
            None
        );
        // Node failure: scontrol showed exactly this after a node was killed.
        let requeued = s(
            "PENDING",
            "Nodes_required_for_job_are_DOWN,_DRAINED_or_reserved_for_jobs_in_higher_priority_partitions",
        );
        let note = SlurmJob::status_note(&requeued, Some(&s("RUNNING", "None")), true).unwrap();
        assert!(
            note.starts_with("requeued, now PENDING (Nodes required for job are DOWN"),
            "{note}"
        );
        // Said once, not every poll.
        assert_eq!(
            SlurmJob::status_note(&requeued, Some(&requeued), true),
            None
        );
        assert_eq!(
            SlurmJob::status_note(&s("RUNNING", "None"), Some(&requeued), true).as_deref(),
            Some("running again")
        );
        // Never ran, but stuck on something other than its turn.
        let held = s("PENDING", "JobHeldUser");
        assert_eq!(
            SlurmJob::status_note(&held, None, false).as_deref(),
            Some("PENDING (JobHeldUser)")
        );
        assert_eq!(
            SlurmJob::status_note(&s("SUSPENDED", "None"), Some(&s("RUNNING", "None")), true)
                .as_deref(),
            Some("SUSPENDED")
        );
    }

    #[test]
    fn slurm_unrecognised_state_is_reported_verbatim() {
        assert_eq!(
            SlurmJob::classify_state("BRAND_NEW_STATE"),
            JobState::Unknown("BRAND_NEW_STATE".into())
        );
    }

    // --- sbatch output path ---

    #[test]
    fn slurm_log_path_is_absolute_against_the_submit_dir() {
        let dir = std::path::Path::new("/work/run");
        let p = SlurmJob::resolve_log_path(&SlurmLauncher::default(), "u", "42", dir);
        assert_eq!(p, Some(PathBuf::from("/work/run/slurm-42.out")));
    }

    #[test]
    fn slurm_log_path_follows_an_output_override_and_chdir() {
        let launcher = SlurmLauncher {
            extra: vec!["--chdir=sub".into(), "--output=logs/%x-%j.log".into()],
            ..Default::default()
        };
        let p = SlurmJob::resolve_log_path(&launcher, "train", "7", std::path::Path::new("/w"));
        assert_eq!(p, Some(PathBuf::from("/w/sub/logs/train-7.log")));
        let abs = SlurmLauncher {
            extra: vec!["--output=/abs/%j.out".into()],
            ..Default::default()
        };
        let p = SlurmJob::resolve_log_path(&abs, "u", "9", std::path::Path::new("/w"));
        assert_eq!(p, Some(PathBuf::from("/abs/9.out")));
    }

    #[test]
    fn slurm_log_path_per_node_pattern_has_no_single_file() {
        let launcher = SlurmLauncher {
            extra: vec!["--output=out-%N.log".into()],
            ..Default::default()
        };
        let p = SlurmJob::resolve_log_path(&launcher, "u", "1", std::path::Path::new("/w"));
        assert_eq!(p, None);
    }

    // --- launcher env validation ---

    #[test]
    fn launch_target_as_str_round_trips_through_parse() {
        for t in [
            LaunchTarget::Local,
            LaunchTarget::Slurm,
            LaunchTarget::Ray,
            LaunchTarget::P2P,
            LaunchTarget::K8s,
            LaunchTarget::Cloud,
        ] {
            assert_eq!(t.as_str().parse::<LaunchTarget>().unwrap(), t);
        }
    }

    /// `p2p` and `cloud` used to fall through `launcher_for` to the local
    /// launcher and run every stage here. They parse, but cannot be placed.
    #[test]
    fn p2p_and_cloud_are_refused_naming_what_does_dispatch() {
        for t in [
            LaunchTarget::Local,
            LaunchTarget::Slurm,
            LaunchTarget::Ray,
            LaunchTarget::K8s,
        ] {
            check_launcher_placeable(t, "`recipe run`").unwrap();
        }
        let p2p = check_launcher_placeable(LaunchTarget::P2P, "`recipe run`")
            .unwrap_err()
            .to_string();
        assert!(
            p2p.starts_with("--launcher p2p cannot place `recipe run`"),
            "{p2p}"
        );
        assert!(p2p.contains("--launcher p2p --p2p-seed"), "{p2p}");
        assert!(p2p.contains("Nothing ran"), "{p2p}");
        let cloud = check_launcher_placeable(LaunchTarget::Cloud, "`hpo` trials")
            .unwrap_err()
            .to_string();
        assert!(
            cloud.starts_with("--launcher cloud cannot place `hpo` trials"),
            "{cloud}"
        );
        assert!(cloud.contains("only `recipe run`"), "{cloud}");
        assert!(cloud.contains("Nothing ran"), "{cloud}");
    }

    /// The surfaces that do place cloud work say exactly what is missing.
    #[test]
    fn a_cloud_launch_needs_the_feature_and_a_queue() {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // SAFETY: serialized by TEST_ENV_LOCK; restored below.
        unsafe { std::env::remove_var(CLOUD_QUEUE_ENV) };
        unsafe { std::env::remove_var(CLOUD_PICKUP_ENV) };
        let err = check_cloud_launch("`recipe run`").unwrap_err().to_string();
        assert!(err.contains("Nothing ran"), "{err}");
        if cfg!(feature = "cloud") {
            assert!(err.contains(CLOUD_QUEUE_ENV), "{err}");
            unsafe { std::env::set_var(CLOUD_QUEUE_ENV, "/tmp/q") };
            check_cloud_launch("`recipe run`").unwrap();
            unsafe { std::env::set_var(CLOUD_PICKUP_ENV, "soon") };
            let err = check_cloud_launch("`recipe run`").unwrap_err().to_string();
            assert!(err.contains(CLOUD_PICKUP_ENV), "{err}");
            unsafe { std::env::set_var(CLOUD_PICKUP_ENV, "7") };
            assert_eq!(cloud_pickup_window().unwrap().as_secs(), 7);
        } else {
            assert!(err.contains("`cloud` feature"), "{err}");
        }
        unsafe { std::env::remove_var(CLOUD_QUEUE_ENV) };
        unsafe { std::env::remove_var(CLOUD_PICKUP_ENV) };
    }

    #[test]
    fn launcher_env_typo_is_an_error_not_a_default() {
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // SAFETY: serialized by TEST_ENV_LOCK; removed below.
        unsafe { std::env::set_var("BLUT_SLURM_NODES", "two") };
        let err = check_launcher_env(LaunchTarget::Slurm)
            .unwrap_err()
            .to_string();
        assert!(err.contains("BLUT_SLURM_NODES"), "{err}");
        unsafe { std::env::set_var("BLUT_SLURM_NODES", "0") };
        assert!(check_launcher_env(LaunchTarget::Slurm).is_err());
        unsafe { std::env::set_var("BLUT_SLURM_NODES", "2") };
        assert!(check_launcher_env(LaunchTarget::Slurm).is_ok());
        // Empty means unset, as `launcher_for` reads it.
        unsafe { std::env::set_var("BLUT_SLURM_NODES", "") };
        assert!(check_launcher_env(LaunchTarget::Slurm).is_ok());
        // Local runs never read it.
        unsafe { std::env::set_var("BLUT_SLURM_NODES", "two") };
        assert!(check_launcher_env(LaunchTarget::Local).is_ok());
        unsafe { std::env::remove_var("BLUT_SLURM_NODES") };
    }

    // --- RemoteJob: ray job status parsing ---
    //
    // Every fixture below is REAL `ray job status` output, captured from a live
    // two-node Ray 2.x cluster on 2026-09-19 (ANSI colour stripped, timestamps
    // kept). The tests these replaced asserted on `"Status: SUCCEEDED\n"`, a
    // string Ray does not emit for a terminal job. Parser and tests shared one
    // invented format, so they agreed with each other and with nothing else,
    // and `poll()` returned Unknown for every finished job.

    #[test]
    fn ray_parse_status_succeeded() {
        let real = "2026-09-19 09:11:33,856\tSUCC cli.py:63 -- --------------------------------\n\
                    2026-09-19 09:11:33,856\tSUCC cli.py:64 -- Job 'blut-ray-proof-1' succeeded\n\
                    2026-09-19 09:11:33,856\tSUCC cli.py:65 -- --------------------------------\n";
        assert_eq!(RayJob::parse_ray_status(real), JobState::Succeeded);
    }

    #[test]
    fn ray_parse_status_failed() {
        let real = "2026-09-19 09:11:32,014\tERR cli.py:72 -- Job 'blut-fail-1' failed\n\
                    2026-09-19 09:11:32,014\tINFO cli.py:86 -- Status message: Job entrypoint \
                    command failed with exit code 3, last available logs (truncated to 20,000 chars):\n";
        assert_eq!(
            RayJob::parse_ray_status(real),
            JobState::Failed("FAILED".to_string())
        );
    }

    #[test]
    fn ray_parse_status_running() {
        let real = "2026-09-19 09:11:43,680\tINFO cli.py:89 -- Status for job 'blut-run-2': RUNNING\n\
                    2026-09-19 09:11:43,680\tINFO cli.py:91 -- Status message: Job is currently running.\n";
        assert_eq!(RayJob::parse_ray_status(real), JobState::Running);
    }

    #[test]
    fn ray_parse_status_stopped() {
        let real = "2026-09-19 09:11:32,922\tWARN cli.py:82 -- Job 'blut-run-1' was stopped\n";
        assert_eq!(RayJob::parse_ray_status(real), JobState::Cancelled);
    }

    #[test]
    fn ray_parse_status_pending_is_running() {
        let real = "2026-09-19 09:11:43,680\tINFO cli.py:89 -- Status for job 'x': PENDING\n";
        assert_eq!(RayJob::parse_ray_status(real), JobState::Running);
    }

    #[test]
    fn a_failure_log_that_contains_the_word_succeeded_is_still_a_failure() {
        // `Status message:` for a failed job embeds the tail of the job's own
        // output, which can say anything. Scanning the whole blob for
        // "succeeded" would report a failed job as successful.
        let real = "2026-09-19 09:11:32,014\tERR cli.py:72 -- Job 'j' failed\n\
                    2026-09-19 09:11:32,014\tINFO cli.py:86 -- Status message: Job entrypoint \
                    command failed with exit code 1, last available logs:\n\
                    step 3 succeeded\n\
                    Traceback (most recent call last): ...\n";
        assert_eq!(
            RayJob::parse_ray_status(real),
            JobState::Failed("FAILED".to_string())
        );
    }

    #[test]
    fn a_failure_log_that_quotes_another_jobs_status_is_still_a_failure() {
        // A job that submits or polls child Ray jobs prints their status lines;
        // after a failure those lines sit in the `Status message:` log tail.
        for child in ["SUCCEEDED", "RUNNING"] {
            let real = format!(
                "2026-09-19 09:11:32,014\tERR cli.py:72 -- Job 'j' failed\n\
                 2026-09-19 09:11:32,014\tINFO cli.py:86 -- Status message: Job entrypoint \
                 command failed with exit code 1, last available logs:\n\
                 INFO cli.py:89 -- Status for job 'child': {child}\n"
            );
            assert_eq!(
                RayJob::parse_ray_status(&real),
                JobState::Failed("FAILED".to_string()),
                "child status {child} leaked through the log tail"
            );
        }
    }

    #[test]
    fn ray_parse_status_unknown() {
        assert!(matches!(
            RayJob::parse_ray_status("something weird"),
            JobState::Unknown(_)
        ));
    }

    #[test]
    fn squeue_ids_are_one_per_line_and_steps_fold_into_their_job() {
        assert_eq!(
            parse_squeue_ids("  41\n42.0\n\n41\n42\n"),
            vec!["41".to_string(), "42".to_string()]
        );
        assert!(parse_squeue_ids("").is_empty());
    }

    // --- sbatch script generation ---

    #[test]
    fn sbatch_script_has_all_flags() {
        let launcher = SlurmLauncher {
            partition: Some("gpu".into()),
            mem: Some("48G".into()),
            cpus: Some(16),
            gpus: Some(2),
            time: Some("12:00:00".into()),
            nodes: None,
            ntasks_per_node: None,
            extra: vec!["--exclusive".to_string(), "--".to_string()],
        };
        let script = SlurmJob::build_sbatch_script(
            "train-run1",
            &launcher,
            &["python".into(), "train.py".into()],
        );
        assert!(script.contains("#!/bin/bash"));
        assert!(script.contains("#SBATCH --job-name=train-run1"));
        assert!(script.contains("#SBATCH --partition=gpu"));
        assert!(script.contains("#SBATCH --mem=48G"));
        assert!(script.contains("#SBATCH --cpus-per-task=16"));
        assert!(script.contains("#SBATCH --gpus=2"));
        assert!(script.contains("#SBATCH --time=12:00:00"));
        assert!(script.contains("#SBATCH --exclusive"));
        assert!(script.contains("#SBATCH --output=slurm-%j.out"));
        // Bare "--" in extra should be filtered out.
        let count = script.matches("#SBATCH --\n").count();
        assert_eq!(count, 0, "bare '--' should not appear as a flag");
        // Inner command appears at the end.
        assert!(script.contains("python train.py"));
    }

    #[test]
    fn sbatch_script_multinode_has_master_addr_preamble() {
        let launcher = SlurmLauncher {
            nodes: Some(4),
            ntasks_per_node: Some(2),
            gpus: Some(2),
            ..Default::default()
        };
        let script = SlurmJob::build_sbatch_script(
            "ddp-job",
            &launcher,
            &["python".into(), "train.py".into()],
        );
        assert!(script.contains("#SBATCH --nodes=4"));
        assert!(script.contains("#SBATCH --ntasks-per-node=2"));
        assert!(script.contains("#SBATCH --gpus=2"));
        // Multi-node preamble exports rendezvous env vars.
        assert!(script.contains("MASTER_ADDR"));
        assert!(script.contains("scontrol show hostnames"));
        // The batch script runs once; the work must be a step on every node,
        // and NODE_RANK must be read inside the task, not in the batch shell.
        assert!(script.contains(
            "srun --kill-on-bad-exit=1 bash -c 'export NODE_RANK=\"$SLURM_NODEID\"; exec \"$@\"' blut-task python train.py"
        ));
        assert!(!script.contains("export NODE_RANK=${SLURM_NODEID"));
    }

    #[test]
    fn sbatch_script_single_node_omits_master_addr() {
        let launcher = SlurmLauncher {
            nodes: Some(1),
            gpus: Some(2),
            ..Default::default()
        };
        let script = SlurmJob::build_sbatch_script("single", &launcher, &["echo".into()]);
        assert!(!script.contains("MASTER_ADDR"));
        assert!(!script.contains("scontrol"));
        assert!(!script.contains("srun"), "one task needs no job step");
    }

    #[test]
    fn sbatch_script_minimal() {
        let script = SlurmJob::build_sbatch_script(
            "unit",
            &SlurmLauncher::default(),
            &["echo".into(), "hello".into()],
        );
        assert!(script.contains("#!/bin/bash"));
        assert!(script.contains("#SBATCH --job-name=unit"));
        assert!(script.contains("#SBATCH --output=slurm-%j.out"));
        assert!(!script.contains("--partition"));
        assert!(!script.contains("--mem"));
        assert!(script.contains("echo hello"));
    }

    // --- submit_async argument shape (non-exec tests) ---

    #[test]
    fn slurm_submit_async_rejects_empty_inner() {
        let l = SlurmLauncher::default();
        assert!(l.submit_async("unit", &[]).is_err());
    }

    /// `submit_async` built its own argv and left out the GPU reservation the
    /// blocking path emits.
    #[test]
    fn ray_async_submit_reserves_the_same_gpus_as_wrap() {
        let l = RayLauncher {
            num_gpus: Some(2),
            address: Some("http://h:8265".into()),
            ..Default::default()
        };
        let inner = ["python".to_string()];
        let async_args = l.job_submit_args("u", &inner, true);
        assert!(async_args.contains(&"--entrypoint-num-gpus=2".to_string()));
        assert!(async_args.contains(&"--no-wait".to_string()));
        let mut blocking = l.wrap("u", &inner).unwrap().args;
        blocking.retain(|a| a != "--no-wait");
        let mut async_args = async_args;
        async_args.retain(|a| a != "--no-wait");
        assert_eq!(async_args, blocking);
    }

    #[test]
    fn launcher_counts_are_parsed_and_empty_settings_are_unset() {
        assert_eq!(env_count("X", Some(" 2 ".into())), Some(2));
        assert_eq!(env_count("X", Some("two".into())), None);
        assert_eq!(env_count("X", None), None);
        let _g = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var_os("RAY_ADDRESS");
        // SAFETY: serialized by TEST_ENV_LOCK; restored below.
        unsafe { std::env::set_var("RAY_ADDRESS", "") };
        let cmd = launcher_for(LaunchTarget::Ray).build_command("u", &["echo".into()]);
        unsafe {
            match prev {
                Some(v) => std::env::set_var("RAY_ADDRESS", v),
                None => std::env::remove_var("RAY_ADDRESS"),
            }
        }
        assert!(
            !argv(&cmd.unwrap())
                .iter()
                .any(|a| a.starts_with("--address"))
        );
    }

    #[test]
    fn ray_submit_async_rejects_empty_inner() {
        let l = RayLauncher::default();
        assert!(l.submit_async("unit", &[]).is_err());
    }
}
