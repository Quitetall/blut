// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! blut-owned `Launcher` abstraction: build the OS command that runs a
//! sweep job, optionally inside a resource-capped container.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

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
        // BARE spawn with a warning — the memory-admission cgroup cap is then NOT
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
    /// `--gpus` (per-node GPU count).
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

/// A handle to a submitted Slurm job, returned by
/// [`SlurmLauncher::submit_async`]. Supports polling, log streaming, and
/// cancellation via `sacct` / `scancel`.
pub struct SlurmJob {
    /// Slurm-assigned numeric job ID.
    job_id: String,
    /// Path to the Slurm output log (`slurm-<id>.out`).
    log_path: PathBuf,
}

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
        // Multi-node preamble: export MASTER_ADDR from Slurm's allocation so
        // torchrun's c10d rendezvous can find the coordinator. Only needed when
        // nodes > 1; single-node torchrun uses --standalone.
        if launcher.nodes.unwrap_or(1) > 1 {
            lines.push(
                "# Multi-node rendezvous: resolve MASTER_ADDR from Slurm allocation".to_string(),
            );
            lines.push(
                "export MASTER_ADDR=$(scontrol show hostnames \"$SLURM_JOB_NODELIST\" | head -n1)"
                    .to_string(),
            );
            lines.push("export MASTER_PORT=${MASTER_PORT:-29500}".to_string());
            lines.push("export NODE_RANK=${SLURM_NODEID:-0}".to_string());
            lines.push(String::new());
        }
        // The inner command — shell-escape each arg to prevent injection.
        let escaped: Vec<String> = inner.iter().map(|a| Self::shell_escape(a)).collect();
        lines.push(escaped.join(" "));
        lines.join("\n")
    }

    /// Parse `sacct -j <id> -n -o State` output into a [`JobState`].
    fn parse_sacct_state(output: &str) -> JobState {
        // sacct may produce multiple lines (per-step entries). The first
        // non-empty token is the job-level state.
        for line in output.lines() {
            let state = line.trim();
            if state.is_empty() {
                continue;
            }
            // Strip extended info after a "+" (e.g. "FAILED+TIMEOUT" or just
            // take the first word if space-separated).
            let base = state.split('+').next().unwrap_or(state).trim();
            return match base {
                "COMPLETED" => JobState::Succeeded,
                "FAILED" => JobState::Failed(state.to_string()),
                "TIMEOUT" => JobState::Failed("TIMEOUT".to_string()),
                "OUT_OF_MEMORY" => JobState::Failed("OutOfMemory".to_string()),
                "NODE_FAIL" => JobState::Failed("NODE_FAIL".to_string()),
                "RUNNING" | "PENDING" | "CONFIGURING" | "SUSPENDED" => JobState::Running,
                "CANCELLED" => JobState::Cancelled,
                "BOOT_FAIL" | "DEADLINE" => JobState::Failed(state.to_string()),
                _ => JobState::Unknown(state.to_string()),
            };
        }
        JobState::Unknown("empty sacct output".to_string())
    }
}

impl RemoteJob for SlurmJob {
    fn id(&self) -> &str {
        &self.job_id
    }

    fn poll(&self) -> Result<JobState> {
        let out = Command::new("sacct")
            .args(["-j", &self.job_id, "-n", "-o", "State", "--noheader"])
            .output()
            .map_err(|e| TrainError::other(format!("sacct invocation failed: {e}")))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let code = match out.status.code() {
                Some(c) => c.to_string(),
                None => "signal".to_string(),
            };
            return Err(TrainError::other(format!(
                "sacct -j {} failed (exit {}): {}",
                self.job_id,
                code,
                stderr.trim()
            )));
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        Ok(Self::parse_sacct_state(&stdout))
    }

    fn stream(&self, sink: &dyn Fn(&str)) -> Result<()> {
        // Tail-follow the log file, polling sacct every 2s for terminal state.
        // Returns Ok(()) on success, Err on failure/cancel/timeout.
        let poll_interval = Duration::from_secs(2);
        let max_idle = Duration::from_secs(3600); // 1h idle → timeout
        let mut offset: u64 = 0;
        let mut idle_since = std::time::Instant::now();
        loop {
            // Stream whatever is new in the log file.
            if self.log_path.exists() {
                use std::io::{Seek, SeekFrom};
                let file = std::fs::File::open(&self.log_path).map_err(|e| TrainError::Io {
                    path: self.log_path.clone(),
                    source: e,
                })?;
                let mut reader = std::io::BufReader::new(file);
                reader
                    .seek(SeekFrom::Start(offset))
                    .map_err(|e| TrainError::Io {
                        path: self.log_path.clone(),
                        source: e,
                    })?;
                let mut line = String::new();
                loop {
                    line.clear();
                    let n = reader.read_line(&mut line).map_err(|e| TrainError::Io {
                        path: self.log_path.clone(),
                        source: e,
                    })?;
                    if n == 0 {
                        break;
                    }
                    offset += n as u64;
                    idle_since = std::time::Instant::now();
                    let trimmed = line.trim_end_matches('\n');
                    if !trimmed.is_empty() {
                        sink(trimmed);
                    }
                }
            }
            // Check if the job has finished.
            match self.poll()? {
                JobState::Running => {
                    if idle_since.elapsed() > max_idle {
                        return Err(TrainError::other(format!(
                            "slurm job {} stream timed out after no output for {}s",
                            self.job_id,
                            max_idle.as_secs()
                        )));
                    }
                    thread::sleep(poll_interval);
                    continue;
                }
                JobState::Succeeded => return Ok(()),
                JobState::Failed(ref reason) => {
                    return Err(TrainError::other(format!(
                        "slurm job {} failed: {reason}",
                        self.job_id,
                    )));
                }
                JobState::Cancelled => {
                    return Err(TrainError::other(format!(
                        "slurm job {} was cancelled",
                        self.job_id,
                    )));
                }
                JobState::Unknown(ref s) => {
                    return Err(TrainError::other(format!(
                        "slurm job {} ended in unknown state: {s}",
                        self.job_id,
                    )));
                }
            }
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
        let log_path = PathBuf::from(format!("slurm-{job_id}.out"));
        Ok(Box::new(SlurmJob { job_id, log_path }))
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
        // ADR 0087: a GPU-bearing unit reserves `count` GPUs on the placed node.
        if let Some(g) = self.num_gpus {
            args.push(format!("--entrypoint-num-gpus={g}"));
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
    fn parse_ray_status(output: &str) -> JobState {
        // Ray job status typically outputs a table like:
        //   Status: SUCCEEDED
        //   ...
        // or a JSON blob. We scan for a "Status:" line.
        for line in output.lines() {
            let trimmed = line.trim();
            if let Some(status) = trimmed.strip_prefix("Status:") {
                let status = status.trim();
                return match status {
                    "SUCCEEDED" => JobState::Succeeded,
                    "FAILED" => JobState::Failed("FAILED".to_string()),
                    "RUNNING" | "PENDING" | "WAITING" | "CONSTRUCTOR" => JobState::Running,
                    "STOPPED" | "CANCELLED" => JobState::Cancelled,
                    "UNKNOWN" => JobState::Unknown("UNKNOWN".to_string()),
                    other => JobState::Unknown(other.to_string()),
                };
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
        let mut args = vec![
            "job".to_string(),
            "submit".to_string(),
            "--no-wait".to_string(),
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
    /// P2P distributed compute (peer GPUs over QUIC).
    P2P,
    /// A Kubernetes cluster via a generated `BlutPlan` (`kubectl apply -f -`).
    /// K8s is an adapter (ADR 0037): the engine emits the manifest textually
    /// and shells `kubectl` — zero kube deps in the engine; the operator crate
    /// owns the reconcile loop.
    K8s,
    /// Cloud compute queue (object-store dispatch, billed on compute — ADR 0082).
    Cloud,
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
        // P2P dispatch is handled by the canonical ExecutionAdapter, not Launcher.
        // Fall through to LocalSystemd for local process management.
        LaunchTarget::P2P => Box::new(LocalSystemd::default()),
        // K8s submits a whole BlutPlan manifest (see `k8s::submit_plan`), not a
        // wrapped local process; fall through to LocalSystemd for any local
        // process management on the submitting side.
        LaunchTarget::K8s => Box::new(LocalSystemd::default()),
        // Cloud dispatch is handled by the cloud submit path (CloudSubmitter +
        // CloudQueue), not Launcher — same precedent as P2P. Fall through to local.
        LaunchTarget::Cloud => Box::new(LocalSystemd::default()),
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

    // --- RemoteJob: sacct parsing ---

    #[test]
    fn slurm_parse_sacct_completed() {
        assert_eq!(
            SlurmJob::parse_sacct_state("COMPLETED\n"),
            JobState::Succeeded
        );
    }

    #[test]
    fn slurm_parse_sacct_failed() {
        let s = SlurmJob::parse_sacct_state("FAILED\n");
        match s {
            JobState::Failed(reason) => assert!(reason.contains("FAILED")),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn slurm_parse_sacct_timeout() {
        assert_eq!(
            SlurmJob::parse_sacct_state("TIMEOUT\n"),
            JobState::Failed("TIMEOUT".to_string())
        );
    }

    #[test]
    fn slurm_parse_sacct_oom() {
        assert_eq!(
            SlurmJob::parse_sacct_state("OUT_OF_MEMORY\n"),
            JobState::Failed("OutOfMemory".to_string())
        );
    }

    #[test]
    fn slurm_parse_sacct_running() {
        assert_eq!(SlurmJob::parse_sacct_state("RUNNING\n"), JobState::Running);
    }

    #[test]
    fn slurm_parse_sacct_pending() {
        assert_eq!(SlurmJob::parse_sacct_state("PENDING\n"), JobState::Running);
    }

    #[test]
    fn slurm_parse_sacct_cancelled() {
        assert_eq!(
            SlurmJob::parse_sacct_state("CANCELLED+\n"),
            JobState::Cancelled
        );
    }

    #[test]
    fn slurm_parse_sacct_empty() {
        assert!(matches!(
            SlurmJob::parse_sacct_state(""),
            JobState::Unknown(_)
        ));
    }

    #[test]
    fn slurm_parse_sacct_node_fail() {
        assert_eq!(
            SlurmJob::parse_sacct_state("NODE_FAIL\n"),
            JobState::Failed("NODE_FAIL".to_string())
        );
    }

    // --- RemoteJob: ray job status parsing ---

    #[test]
    fn ray_parse_status_succeeded() {
        assert_eq!(
            RayJob::parse_ray_status("Status: SUCCEEDED\n"),
            JobState::Succeeded
        );
    }

    #[test]
    fn ray_parse_status_failed() {
        assert_eq!(
            RayJob::parse_ray_status("Status: FAILED\n"),
            JobState::Failed("FAILED".to_string())
        );
    }

    #[test]
    fn ray_parse_status_running() {
        assert_eq!(
            RayJob::parse_ray_status("Status: RUNNING\n"),
            JobState::Running
        );
    }

    #[test]
    fn ray_parse_status_pending() {
        assert_eq!(
            RayJob::parse_ray_status("Status: PENDING\n"),
            JobState::Running
        );
    }

    #[test]
    fn ray_parse_status_stopped() {
        assert_eq!(
            RayJob::parse_ray_status("Status: STOPPED\n"),
            JobState::Cancelled
        );
    }

    #[test]
    fn ray_parse_status_cancelled() {
        assert_eq!(
            RayJob::parse_ray_status("Status: CANCELLED\n"),
            JobState::Cancelled
        );
    }

    #[test]
    fn ray_parse_status_unknown() {
        assert!(matches!(
            RayJob::parse_ray_status("something weird"),
            JobState::Unknown(_)
        ));
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
        assert!(script.contains("NODE_RANK"));
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

    #[test]
    fn ray_submit_async_rejects_empty_inner() {
        let l = RayLauncher::default();
        assert!(l.submit_async("unit", &[]).is_err());
    }
}
