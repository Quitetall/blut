// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Live checks of the Slurm launcher against a real cluster. Ignored by
//! default; run on a submit host with `BLUT_SLURM_LIVE=1 cargo test --test
//! slurm_live -- --ignored --test-threads=1` from a directory every node can
//! see. The cluster needs at least two idle nodes. `docs/DISTRIBUTED_VALIDATION.md`
//! records the runs.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use blut::config::launcher::{JobState, Launcher, RemoteJob, SlurmLauncher};

fn live() -> bool {
    std::env::var("BLUT_SLURM_LIVE").as_deref() == Ok("1")
}

fn sh(script: &str) -> Vec<String> {
    vec!["bash".into(), "-c".into(), script.into()]
}

fn collect(job: &dyn RemoteJob) -> (Result<(), String>, Vec<String>) {
    let lines = Mutex::new(Vec::new());
    let r = job
        .stream(&|l| lines.lock().unwrap().push(l.to_string()))
        .map_err(|e| e.to_string());
    (r, lines.into_inner().unwrap())
}

fn wait_running(job: &dyn RemoteJob) {
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(60) {
        if std::process::Command::new("squeue")
            .args(["-h", "-j", job.id(), "-o", "%T"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "RUNNING")
            .unwrap_or(false)
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    panic!("job {} never started", job.id());
}

#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn succeeded_job_streams_its_output() {
    if !live() {
        return;
    }
    let job = SlurmLauncher::default()
        .submit_async(
            "blut-live-ok",
            &sh("sleep 2; echo first; echo 'second line'"),
        )
        .unwrap();
    let (r, lines) = collect(job.as_ref());
    assert_eq!(r, Ok(()));
    assert_eq!(lines, vec!["first", "second line"]);
    assert_eq!(job.poll().unwrap(), JobState::Succeeded);
}

#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn failed_job_is_reported_failed() {
    if !live() {
        return;
    }
    let job = SlurmLauncher::default()
        .submit_async("blut-live-fail", &sh("echo about to fail; exit 3"))
        .unwrap();
    let (r, lines) = collect(job.as_ref());
    let err = r.unwrap_err();
    assert!(err.contains("failed: FAILED"), "{err}");
    assert_eq!(lines, vec!["about to fail"]);
}

#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn cancelled_job_is_reported_cancelled() {
    if !live() {
        return;
    }
    let job = SlurmLauncher::default()
        .submit_async("blut-live-cancel", &sh("sleep 300"))
        .unwrap();
    wait_running(job.as_ref());
    job.cancel().unwrap();
    let (r, _) = collect(job.as_ref());
    assert!(r.unwrap_err().contains("was cancelled"));
}

#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn multi_node_batch_runs_on_every_node_with_its_own_rank() {
    if !live() {
        return;
    }
    let launcher = SlurmLauncher {
        nodes: Some(2),
        ntasks_per_node: Some(1),
        ..Default::default()
    };
    let job = launcher
        .submit_async(
            "blut-live-multinode",
            &sh("echo rank=$NODE_RANK host=$(hostname) master=$MASTER_ADDR"),
        )
        .unwrap();
    let (r, mut lines) = collect(job.as_ref());
    assert_eq!(r, Ok(()));
    lines.sort();
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[0].starts_with("rank=0 "), "{lines:?}");
    assert!(lines[1].starts_with("rank=1 "), "{lines:?}");
    let host = |l: &str| l.split_whitespace().nth(1).unwrap().to_string();
    assert_ne!(host(&lines[0]), host(&lines[1]), "both ranks on one node");
    // Every rank agrees on the coordinator, and it is rank 0's host.
    let master = |l: &str| l.split_whitespace().nth(2).unwrap().to_string();
    assert_eq!(master(&lines[0]), master(&lines[1]));
    assert_eq!(
        master(&lines[0]),
        format!("master={}", &host(&lines[0])[5..])
    );
}

#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn queued_job_is_waited_for_not_failed() {
    if !live() {
        return;
    }
    // Hold every node, then submit a job that needs all of them: it queues.
    let hold = SlurmLauncher {
        nodes: Some(2),
        extra: vec!["--exclusive".into()],
        ..Default::default()
    }
    .submit_async("blut-live-hold", &sh("sleep 8"))
    .unwrap();
    wait_running(hold.as_ref());
    let queued = SlurmLauncher {
        nodes: Some(2),
        extra: vec!["--exclusive".into()],
        ..Default::default()
    }
    .submit_async("blut-live-queued", &sh("echo ran after the queue"))
    .unwrap();
    assert_eq!(
        queued.poll().unwrap(),
        JobState::Running,
        "queued counts as not finished"
    );
    let (r, lines) = collect(queued.as_ref());
    assert_eq!(r, Ok(()));
    assert_eq!(lines, vec!["ran after the queue"; 2]);
    assert_eq!(collect(hold.as_ref()).0, Ok(()));
}

#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn state_of_a_purged_job_comes_from_accounting_or_a_clear_error() {
    if !live() {
        return;
    }
    let job = SlurmLauncher::default()
        .submit_async("blut-live-purge", &sh("true"))
        .unwrap();
    assert_eq!(collect(job.as_ref()).0, Ok(()));
    // Wait past MinJobAge so the controller forgets the job.
    let t = Instant::now();
    while std::process::Command::new("scontrol")
        .args(["show", "job", job.id()])
        .output()
        .unwrap()
        .status
        .success()
    {
        assert!(
            t.elapsed() < Duration::from_secs(600),
            "record never purged"
        );
        std::thread::sleep(Duration::from_secs(5));
    }
    match std::env::var("BLUT_SLURM_LIVE_ACCOUNTING").as_deref() {
        Ok("1") => assert_eq!(job.poll().unwrap(), JobState::Succeeded),
        _ => {
            let err = job.poll().unwrap_err().to_string();
            assert!(err.contains("unknown to the controller"), "{err}");
        }
    }
}

#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn srun_wrap_runs_the_inner_command_on_each_node() {
    if !live() {
        return;
    }
    let w = SlurmLauncher {
        nodes: Some(2),
        ntasks_per_node: Some(1),
        ..Default::default()
    }
    .wrap("blut-live-wrap", &sh("echo node=$SLURM_NODEID"))
    .unwrap();
    let out = std::process::Command::new(&w.program)
        .args(&w.args)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut lines: Vec<_> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_string)
        .collect();
    lines.sort();
    assert_eq!(lines, vec!["node=0", "node=1"]);
}

#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn a_previous_jobs_log_under_a_reused_id_is_not_streamed() {
    if !live() {
        return;
    }
    // Plant a stale log where the next job's output will go, as a reused job
    // id (state reset, counter wrap) would leave it.
    let cfg = std::process::Command::new("scontrol")
        .args(["show", "config"])
        .output()
        .unwrap();
    let next = String::from_utf8_lossy(&cfg.stdout)
        .lines()
        .find_map(|l| {
            l.strip_prefix("NEXT_JOB_ID")
                .map(|v| v.trim_start_matches([' ', '=']).trim().to_string())
        })
        .expect("NEXT_JOB_ID in scontrol show config");
    let stale = format!("slurm-{next}.out");
    std::fs::write(&stale, "stale line one\nstale line two from an older job\n").unwrap();
    let old = std::time::SystemTime::now() - Duration::from_secs(3600);
    std::fs::File::options()
        .write(true)
        .open(&stale)
        .unwrap()
        .set_modified(old)
        .unwrap();
    let job = SlurmLauncher::default()
        .submit_async("blut-live-reused-id", &sh("sleep 1; echo fresh"))
        .unwrap();
    assert_eq!(job.id(), next, "the planted log must match the job id");
    let (r, lines) = collect(job.as_ref());
    assert_eq!(r, Ok(()));
    assert_eq!(lines, vec!["fresh"]);
}

/// A Slurm job that outlived the BLUT process that started it: `srun` keeps
/// running when its parent dies. With its name recorded in the BLUT job,
/// `plan resume`'s check sees it and `blut cancel` stops it.
#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn a_slurm_job_left_by_a_dead_driver_is_found_and_cancelled() {
    if !live() {
        return;
    }
    let job_id = format!("live-orphan-{}", std::process::id());
    let job_dir = blut::paths::job_dir(&job_id).unwrap();
    let name = blut::jobs::slurm_job_name(&job_dir, 2);
    blut::jobs::record_slurm_job_name(&job_dir, &name).unwrap();
    blut::jobs::record_slurm_job_name(&job_dir, &name).unwrap();
    assert_eq!(
        blut::jobs::slurm_job_names(&job_id).unwrap(),
        vec![name.clone()]
    );

    // The orphan: an `srun` with no BLUT process behind it, as after
    // `kill -9` of the driver.
    let w = SlurmLauncher {
        nodes: Some(2),
        ntasks_per_node: Some(1),
        ..Default::default()
    }
    .wrap(&name, &sh("sleep 300"))
    .unwrap();
    let mut srun = std::process::Command::new(&w.program)
        .args(&w.args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let t = Instant::now();
    let live = loop {
        let live = blut::jobs::live_slurm_jobs(&job_id).unwrap();
        if !live.is_empty() || t.elapsed() > Duration::from_secs(60) {
            break live;
        }
        std::thread::sleep(Duration::from_millis(500));
    };
    assert_eq!(live.len(), 1, "the named job is found: {live:?}");
    assert_eq!(live[0].0, name);
    let why = blut::jobs::slurm_resume_blocker(&job_id)
        .unwrap()
        .expect("blocked");
    assert!(
        why.contains(&live[0].1[0]) && why.contains(&format!("blut cancel {job_id}")),
        "{why}"
    );

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(blut::jobs::cancel_job(&job_id, Duration::from_secs(2)))
        .unwrap();
    let t = Instant::now();
    while !blut::jobs::live_slurm_jobs(&job_id).unwrap().is_empty() {
        assert!(t.elapsed() < Duration::from_secs(60), "scancel took effect");
        std::thread::sleep(Duration::from_millis(500));
    }
    let status = srun.wait().unwrap();
    assert!(!status.success(), "srun ends when its job is cancelled");
    assert_eq!(blut::jobs::slurm_resume_blocker(&job_id).unwrap(), None);
    let _ = std::fs::remove_dir_all(&job_dir);
}

/// Under `--launcher slurm` the executor records the name it gives a stage
/// and, when an attempt fails, cancels the Slurm job of that name, which the
/// attempt's `srun` may have left running.
#[test]
#[ignore = "needs a live Slurm cluster (BLUT_SLURM_LIVE=1)"]
fn a_failed_attempts_slurm_job_is_cancelled_by_the_executor() {
    use async_trait::async_trait;
    use blut::framework::artifact::{Artifact, ContentHash};
    use blut::framework::stage::{Stage, StageContext};

    if !live() {
        return;
    }
    static LEFT_BEHIND: Mutex<Vec<std::process::Child>> = Mutex::new(Vec::new());
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct Out;
    impl Artifact for Out {
        const KIND: &'static str = "test.live-out";
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
    struct LeavesAJob;
    #[async_trait]
    impl Stage for LeavesAJob {
        const NAME: &'static str = "leaves_a_job";
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
            let name = ctx.slurm_job_name();
            let w = SlurmLauncher::default()
                .wrap(&name, &sh("sleep 300"))
                .unwrap();
            // Not waited for here: the attempt "dies" and leaves its Slurm
            // job behind. The test reaps the `srun` once the job is gone.
            let srun = std::process::Command::new(&w.program)
                .args(&w.args)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap();
            LEFT_BEHIND.lock().unwrap().push(srun);
            let t = Instant::now();
            while blut::config::launcher::slurm_jobs_named(&name)
                .unwrap()
                .is_empty()
            {
                assert!(t.elapsed() < Duration::from_secs(60), "job started");
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(blut::framework::StageError::Backend(anyhow::anyhow!(
                "trainer lost"
            )))
        }
    }
    struct Backend;
    impl blut::backends::TrainingBackend for Backend {
        const ID: &'static str = "slurm-live";
        const DESCRIPTION: &'static str = "slurm live";
    }
    impl blut::framework::compat::Compatible<Backend> for LeavesAJob {}

    let job_id = format!("live-failed-{}", std::process::id());
    let job_dir = blut::paths::job_dir(&job_id).unwrap();
    let plan = blut::framework::plan::Plan::<(), Backend>::new("live", serde_json::json!({}))
        .start(LeavesAJob, NoArgs {})
        .finish()
        .into_compiled();
    let ctx = blut::framework::executor::ExecCtx::new(job_dir.clone())
        .with_launch_target(blut::config::launcher::LaunchTarget::Slurm);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let r = rt.block_on(blut::framework::executor::ParallelExecutor::execute(
        plan, ctx,
    ));
    assert!(r.is_err(), "the stage failed");
    let names = blut::jobs::slurm_job_names(&job_id).unwrap();
    assert_eq!(names, vec![format!("blut-{job_id}-0")]);
    let t = Instant::now();
    while !blut::jobs::live_slurm_jobs(&job_id).unwrap().is_empty() {
        assert!(
            t.elapsed() < Duration::from_secs(60),
            "the executor cancelled the job the failed attempt left"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    for mut srun in LEFT_BEHIND.lock().unwrap().drain(..) {
        let status = srun.wait().unwrap();
        assert!(!status.success(), "srun ends when its job is cancelled");
    }
    let _ = std::fs::remove_dir_all(&job_dir);
}
