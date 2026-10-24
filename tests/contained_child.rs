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

/// The launcher path: `LocalSystemd`'s `mem_max` reaches the backend the
/// factory selects. `rlimit` is selected explicitly because it needs no
/// systemd bus or cgroup delegation, so this runs on any Unix host. The same
/// 1 GiB allocation fails under a 512 MiB `mem_max` and succeeds under 8 GiB,
/// so the cap, not the host, is what refused it.
#[tokio::test]
async fn local_systemd_mem_max_caps_the_selected_backend() {
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
    assert_eq!(unit, "blut-launcher-test", "unit sanitized");
    assert!(!ok, "1 GiB fit under a 512 MiB mem_max");
    assert!(!stdout.contains("NO_CAP"));

    let (ok, stdout, _, _) = launch_python("8G", script).await;
    assert!(ok, "1 GiB must fit under an 8 GiB mem_max");
    assert!(stdout.contains("NO_CAP"));
}
