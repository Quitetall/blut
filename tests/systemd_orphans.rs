// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! A stage's systemd unit left running by a SIGKILLed BLUT process, found
//! and stopped from another process.
//!
//! A stage under the systemd backend runs in a transient `systemd-run --user`
//! service. The direct child of the BLUT process (its driver) is only the
//! `systemd-run` client, so when the driver is SIGKILLed, no teardown runs and
//! the service keeps the trainer running. The executor records the unit with
//! the job before it starts; `blut cancel` (`jobs::cancel_job`) stops it, and
//! `plan resume` (`jobs::resume_blocker`) refuses while it runs.
//!
//! The driver here is this test binary run again as a child process
//! ([`driver_process`]): it runs a one-stage plan through the executor whose
//! stage spawns `sleep 600` through `LocalSystemd::spawn_contained`. Needs a
//! systemd user manager (systemd 250 or later); skipped, loudly, elsewhere.
#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Set in the driver child's environment: the job id it runs.
const DRIVER_JOB_ENV: &str = "BLUT_ORPHAN_TEST_DRIVER_JOB";

fn user_bus_with_setenv_by_name() -> bool {
    let bus = Command::new("systemctl")
        .args(["--user", "show-environment"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    let version = Command::new("systemd-run")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u32>().ok())
        });
    bus && version.is_some_and(|v| v >= 250)
}

fn unit_state(service: &str) -> String {
    Command::new("systemctl")
        .args(["--user", "is-active", service])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn running(state: &str) -> bool {
    matches!(state, "active" | "activating" | "deactivating")
}

/// The last part of the driver's output, for a failure message.
fn tail(path: &std::path::Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let start = text.len().saturating_sub(4000);
    let start = (start..text.len())
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(text.len());
    text[start..].to_string()
}

/// Kills the driver and stops every unit it started when the test ends,
/// however it ends, so a failure leaves nothing running: the recorded ones,
/// and by name every `blut-orphan-test-<driver pid>-*` unit, in case the
/// record is what failed.
struct Cleanup {
    driver: Option<Child>,
    driver_pid: u32,
    job_dir: PathBuf,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(mut d) = self.driver.take() {
            let _ = d.kill();
            let _ = d.wait();
        }
        let mut units: Vec<String> = std::fs::read_to_string(self.job_dir.join("systemd_units"))
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| format!("{}.service", l.trim()))
            .collect();
        units.push(format!("blut-orphan-test-{}-*.service", self.driver_pid));
        let _ = Command::new("systemctl")
            .args(["--user", "stop", "--"])
            .args(&units)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// The driver: what `blut recipe run` does for one job, reduced to one stage
/// that spawns a contained `sleep 600` and waits for it. It never finishes on
/// its own; the test SIGKILLs it. Does nothing unless run by the test below.
#[test]
#[ignore = "helper: run as a child process by a_unit_left_by_a_killed_driver_is_stopped_by_cancel"]
fn driver_process() {
    use async_trait::async_trait;
    use blut::framework::artifact::{Artifact, ContentHash};
    use blut::framework::stage::{Stage, StageContext};

    let Ok(job_id) = std::env::var(DRIVER_JOB_ENV) else {
        return;
    };

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct Out;
    impl Artifact for Out {
        const KIND: &'static str = "test.orphan-out";
        const SCHEMA: u32 = 1;
        const INLINE: bool = true;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(b"out")
        }
        fn primary_path(&self) -> &std::path::Path {
            std::path::Path::new(".")
        }
    }
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    struct NoArgs {}
    struct RunsAUnit;
    #[async_trait]
    impl Stage for RunsAUnit {
        const NAME: &'static str = "runs_a_unit";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [blut::framework::resource::Resource] =
            &[blut::framework::resource::Resource::Cpu];
        type Input = ();
        type Output = Out;
        type Args = NoArgs;
        async fn run(
            &self,
            ctx: &StageContext,
            _: (),
            _: &NoArgs,
        ) -> Result<Out, blut::framework::StageError> {
            let inner = ["sleep".to_string(), "600".to_string()];
            let mut child = blut::config::launcher::LocalSystemd::default()
                .spawn_contained("blut-orphan-test", &inner, &ctx.job_dir, &[], |c| {
                    c.stdin(Stdio::null());
                })
                .await
                .map_err(|e| blut::framework::StageError::Backend(anyhow::anyhow!("{e}")))?;
            eprintln!("driver: {} placed under {}", child.unit(), child.kind());
            let _ = child.wait().await;
            Err(blut::framework::StageError::Backend(anyhow::anyhow!(
                "the unit ended"
            )))
        }
    }
    struct Backend;
    impl blut::backends::TrainingBackend for Backend {
        const ID: &'static str = "systemd-orphan";
        const DESCRIPTION: &'static str = "systemd orphan test";
    }
    impl blut::framework::compat::Compatible<Backend> for RunsAUnit {}

    let job_dir = blut::paths::job_dir(&job_id).unwrap();
    blut::jobs::write_state(&job_id, blut::jobs::JobState::Running).unwrap();
    let _driver = blut::jobs::record_driver(&job_id);
    let plan = blut::framework::plan::Plan::<(), Backend>::new("orphan", serde_json::json!({}))
        .start(RunsAUnit, NoArgs {})
        .finish()
        .into_compiled();
    let ctx = blut::framework::executor::ExecCtx::new(job_dir);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let r = rt.block_on(blut::framework::executor::ParallelExecutor::execute(
        plan, ctx,
    ));
    eprintln!("driver: plan ended: {:?}", r.err());
}

/// The driver is SIGKILLed while its stage's unit runs. The unit survives
/// it; the job records it; `plan resume` refuses and names it; `blut cancel`
/// stops it and says so; afterwards the record blocks nothing.
#[test]
fn a_unit_left_by_a_killed_driver_is_stopped_by_cancel() {
    if !user_bus_with_setenv_by_name() {
        eprintln!("skipped: no systemd user bus here, or systemd older than 250");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    // SAFETY: the only test this process runs (the other one runs only in
    // the driver child), so nothing reads the environment concurrently.
    unsafe {
        std::env::set_var("LAMU_TRAIN_JOBS_DIR", tmp.path());
    }
    let job_id = format!("20261030-000000-{:09}", std::process::id());
    let job_dir = blut::paths::job_dir(&job_id).unwrap();
    let log_path = tmp.path().join("driver.log");
    let log = std::fs::File::create(&log_path).unwrap();
    let driver = Command::new(std::env::current_exe().unwrap())
        .args([
            "driver_process",
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(DRIVER_JOB_ENV, &job_id)
        .env("LAMU_TRAIN_JOBS_DIR", tmp.path())
        .env("BLUT_CONTAINMENT", "systemd")
        .env_remove("BLUT_NO_CONTAIN")
        .stdin(Stdio::null())
        .stdout(log.try_clone().unwrap())
        .stderr(log)
        .spawn()
        .expect("spawn the driver");
    let mut cleanup = Cleanup {
        driver_pid: driver.id(),
        driver: Some(driver),
        job_dir: job_dir.clone(),
    };

    // The unit is recorded (before it starts) and then runs.
    let started = Instant::now();
    let (unit, service) = loop {
        if let Some(unit) = blut::jobs::systemd_unit_names(&job_id).unwrap().first() {
            let service = format!("{unit}.service");
            if unit_state(&service) == "active" {
                break (unit.clone(), service);
            }
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "no recorded unit running 60 s after the driver started; driver output:\n{}",
            tail(&log_path)
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    assert!(
        unit.starts_with(&format!("blut-orphan-test-{}-", cleanup.driver_pid)),
        "{unit}"
    );

    // SIGKILL: no destructor, no teardown.
    let mut driver = cleanup.driver.take().unwrap();
    driver.kill().unwrap();
    driver.wait().unwrap();
    assert_eq!(blut::jobs::live_driver(&job_id), None, "the driver is dead");

    // The unit outlives it: this is what has to be found and stopped.
    std::thread::sleep(Duration::from_secs(1));
    assert_eq!(
        unit_state(&service),
        "active",
        "the unit died with its driver"
    );
    assert_eq!(
        blut::jobs::live_systemd_units(&job_id).unwrap(),
        vec![service.clone()]
    );

    // `plan resume` refuses, names the unit and says how to stop it.
    let why = blut::jobs::resume_blocker(&job_id).expect("resume must refuse");
    assert!(
        why.contains(&service) && why.contains(&format!("blut cancel {job_id}")),
        "{why}"
    );

    // `blut cancel`: the driver is dead, so the unit is stopped directly.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(blut::jobs::cancel_job(&job_id, Duration::from_secs(2)))
        .expect("cancel");
    let state = unit_state(&service);
    assert!(!running(&state), "{service} still {state} after the cancel");
    assert!(blut::jobs::live_systemd_units(&job_id).unwrap().is_empty());
    assert_eq!(
        blut::jobs::read_state(&job_id).unwrap(),
        blut::jobs::JobState::Cancelled
    );

    // The record stays, and blocks nothing now that its unit is gone.
    assert_eq!(blut::jobs::systemd_unit_names(&job_id).unwrap(), vec![unit]);
    assert_eq!(blut::jobs::resume_blocker(&job_id), None);
    drop(cleanup);
}
