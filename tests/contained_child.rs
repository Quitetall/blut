// Integration tests for `ContainedChild` and `LocalSystemd::spawn_contained`
// on Unix. The Windows Job Object path has its own suite
// (`windows_job_object.rs`); these pin the same contract on the backends a
// Linux or macOS host selects: exit status passes through, a backend with no
// peak reports none, cancellation kills the child, and the launcher's
// `mem_max` reaches the selected backend as a real cap.
#![cfg(unix)]

use std::path::Path;
use std::time::Duration;

use blut::config::launcher::LocalSystemd;
use blut::containment::{ContainedChild, Containment, PeakSource, bare::Bare};
use tokio_util::sync::CancellationToken;

fn sh(cwd: &Path, script: &str) -> blut::containment::WrappedRun {
    Bare.wrap_command(
        Path::new("sh"),
        &["-c".to_string(), script.to_string()],
        cwd,
        &[],
        "blut-contained-test",
        &Default::default(),
    )
    .expect("wrap")
}

#[tokio::test]
async fn exit_status_passes_through_and_bare_has_no_peak() {
    let tmp = tempfile::tempdir().unwrap();
    let mut child = ContainedChild::spawn(Box::new(Bare), sh(tmp.path(), "exit 3"), "exit", |_| {})
        .await
        .expect("spawn");
    assert_eq!(child.kind(), "bare");
    assert_eq!(child.peak_source(), PeakSource::None);
    let status = tokio::time::timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("exited")
        .expect("wait");
    assert_eq!(status.code(), Some(3));
    assert_eq!(child.peak_bytes(), None);
    assert_eq!(child.oom_kills(), None, "bare cannot tell");
}

#[tokio::test]
async fn wait_or_cancel_returns_the_status_when_the_child_exits_first() {
    let tmp = tempfile::tempdir().unwrap();
    let mut child = ContainedChild::spawn(Box::new(Bare), sh(tmp.path(), "exit 0"), "fast", |_| {})
        .await
        .expect("spawn");
    let cancel = CancellationToken::new();
    let outcome = tokio::time::timeout(Duration::from_secs(30), child.wait_or_cancel(&cancel))
        .await
        .expect("exited")
        .expect("wait");
    assert!(outcome.is_some_and(|s| s.success()), "got {outcome:?}");
}

#[tokio::test]
async fn wait_or_cancel_kills_a_child_that_would_outlive_the_test() {
    let tmp = tempfile::tempdir().unwrap();
    let mut child = ContainedChild::spawn(
        Box::new(Bare),
        sh(tmp.path(), "exec sleep 600"),
        "slow",
        |c| {
            c.kill_on_drop(true);
        },
    )
    .await
    .expect("spawn");
    let cancel = CancellationToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        trigger.cancel();
    });
    let outcome = tokio::time::timeout(Duration::from_secs(30), child.wait_or_cancel(&cancel))
        .await
        .expect("a 600 s sleeper was killed within 30 s")
        .expect("wait_or_cancel");
    assert!(outcome.is_none(), "reported as cancelled, got {outcome:?}");
}

#[tokio::test]
async fn spawn_failure_is_an_error_not_a_panic() {
    let tmp = tempfile::tempdir().unwrap();
    let run = Bare
        .wrap_command(
            Path::new("/nonexistent/blut-no-such-program"),
            &[],
            tmp.path(),
            &[],
            "missing",
            &Default::default(),
        )
        .expect("wrap builds; the spawn fails");
    let err = ContainedChild::spawn(Box::new(Bare), run, "missing", |_| {})
        .await
        .expect_err("spawn of a missing program");
    // The OS error kind survives, so a runner can tell a missing interpreter
    // from any other spawn failure.
    match err {
        blut::TrainError::Io { path, source } => {
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound, "{source}");
            assert!(path.ends_with("blut-no-such-program"), "{}", path.display());
        }
        other => panic!("expected TrainError::Io, got {other:?}"),
    }
}

/// Run `script` through `LocalSystemd::spawn_contained` with `mem_max`;
/// returns (succeeded, stdout, backend kind, unit).
async fn launch_python(mem_max: &str, script: &str) -> (bool, String, &'static str, String) {
    let tmp = tempfile::tempdir().unwrap();
    let launcher = LocalSystemd {
        mem_max: mem_max.to_string(),
        ..LocalSystemd::default()
    };
    let inner = ["python3".to_string(), "-c".to_string(), script.to_string()];
    let mut child = launcher
        .spawn_contained("blut launcher/test", &inner, tmp.path(), &[], |c| {
            c.stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null());
        })
        .await
        .expect("spawn_contained");
    let mut stdout = String::new();
    let mut out = child.child_mut().stdout.take().unwrap();
    tokio::io::AsyncReadExt::read_to_string(&mut out, &mut stdout)
        .await
        .unwrap();
    let status = tokio::time::timeout(Duration::from_secs(60), child.wait())
        .await
        .expect("exited")
        .expect("wait");
    (
        status.success(),
        stdout,
        child.kind(),
        child.unit().to_string(),
    )
}

/// The launcher under the `rlimit` fallback (no systemd bus, no writable
/// cgroup): the stage runs, and it runs WITHOUT the launcher's cap, because
/// `RLIMIT_AS` caps virtual address space and the launcher's default would
/// refuse a CUDA process's reservations. A 1 GiB allocation under a 512 MiB
/// `mem_max` must succeed. The unit name is the sanitized name plus the
/// driver pid and a sequence number.
#[tokio::test]
async fn local_systemd_under_rlimit_runs_the_stage_without_a_cap() {
    if which::which("python3").is_err() {
        eprintln!("skipped: python3 not on PATH");
        return;
    }
    // SAFETY: the only test in this binary that touches the environment, and
    // none of the others read these variables.
    unsafe {
        std::env::remove_var("BLUT_NO_CONTAIN");
        std::env::set_var("BLUT_CONTAINMENT", "rlimit");
    }
    let script = "x = bytearray(1024*1024*1024); print('NO_CAP')";
    let (ok, stdout, kind, unit) = launch_python("512M", script).await;
    assert_eq!(kind, "rlimit");
    let pid = std::process::id().to_string();
    let rest = unit
        .strip_prefix("blut-launcher-test-")
        .unwrap_or_else(|| panic!("unit {unit:?} not sanitized and suffixed"));
    let (unit_pid, seq) = rest.split_once('-').expect("pid-seq suffix");
    assert_eq!(unit_pid, pid);
    assert!(seq.parse::<u64>().is_ok(), "sequence in {unit:?}");
    assert!(ok, "the stage must run uncapped under rlimit");
    assert!(stdout.contains("NO_CAP"));

    // Two stages from one process never share a unit name.
    let (_, _, _, second) = launch_python("512M", "pass").await;
    assert_ne!(unit, second);
}

/// Dropping a `ContainedChild` whose backend has no tree teardown kills the
/// child's process group, so a grandchild of a group-leading child dies too
/// (a hard stage timeout drops the stage future, and with it this value).
#[tokio::test]
async fn dropping_a_bare_group_leader_kills_its_grandchild() {
    let tmp = tempfile::tempdir().unwrap();
    let pidfile = tmp.path().join("grandchild.pid");
    let run = sh(
        tmp.path(),
        &format!("sleep 600 & echo $! > {}; wait", pidfile.display()),
    );
    let child = ContainedChild::spawn(Box::new(Bare), run, "drop", |c| {
        // SAFETY: setsid is async-signal-safe.
        unsafe {
            c.pre_exec(|| {
                nix::unistd::setsid()
                    .map(|_| ())
                    .map_err(std::io::Error::from)
            });
        }
    })
    .await
    .expect("spawn");
    let grandchild = read_pid(&pidfile).await;
    assert!(alive(grandchild), "grandchild {grandchild} was not running");
    drop(child);
    wait_dead(grandchild).await;
}

async fn read_pid(path: &Path) -> i32 {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(s) = std::fs::read_to_string(path)
            && let Ok(pid) = s.trim().parse()
        {
            return pid;
        }
        assert!(std::time::Instant::now() < deadline, "no pid in {path:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Exists and is not a zombie.
fn alive(pid: i32) -> bool {
    if nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_err() {
        return false;
    }
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| {
            s.rsplit_once(')')
                .map(|(_, rest)| rest.trim_start().starts_with('Z'))
        })
        .is_none_or(|zombie| !zombie)
}

async fn wait_dead(pid: i32) {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while alive(pid) {
        assert!(
            std::time::Instant::now() < deadline,
            "pid {pid} survived the teardown"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
