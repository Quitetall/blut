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
