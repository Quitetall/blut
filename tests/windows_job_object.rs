// Integration tests for the Windows Job Object containment backend. Each test
// runs a real `python` child inside a real Job Object and checks the effect
// from outside: the cap fails an oversized allocation (and a larger cap lets
// the same allocation through, so the cap is what failed it), the peak is
// reported, and both cancel and dropping the last teardown handle kill a
// grandchild the child spawned. Windows-only; `python` must be on PATH (the
// CI job installs it with actions/setup-python).
#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use blut::config::launcher::LocalSystemd;
use blut::containment::{
    Availability, CapSpec, ContainedChild, Containment, PeakSource, WrappedRun, spawn_attached,
    windows::WindowsJobObject,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::sync::CancellationToken;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows_sys::Win32::System::Threading::{
    OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, WaitForSingleObject,
};

const MIB: u64 = 1024 * 1024;
/// The exit code the backend gives processes it kills (`KILLED_EXIT_CODE`).
const KILLED_EXIT_CODE: i32 = 0xB1_07;

fn python() -> PathBuf {
    PathBuf::from("python")
}

fn caps(mem_max: Option<u64>) -> CapSpec {
    CapSpec {
        mem_max,
        mem_high: None,
        swap_max: None,
    }
}

fn wrap(cwd: &Path, script: &str, caps: &CapSpec) -> WrappedRun {
    let mut wr = WindowsJobObject
        .wrap_command(
            &python(),
            &["-c".into(), script.into()],
            cwd,
            &[],
            "blut-windows-job-test",
            caps,
        )
        .expect("wrap_command");
    wr.command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    wr
}

/// Run `script` to completion inside a fresh job; returns the output and the
/// job's peak, read after exit the way a runner reads it.
async fn run_capped(script: &str, caps: &CapSpec) -> (std::process::Output, Option<u64>) {
    let tmp = tempfile::tempdir().unwrap();
    let mut wr = wrap(tmp.path(), script, caps);
    assert_eq!(wr.peak_source, PeakSource::CgroupFile);
    let child = spawn_attached(&WindowsJobObject, &mut wr.command, &wr.teardown)
        .await
        .expect("spawn + attach");
    let out = tokio::time::timeout(Duration::from_secs(120), child.wait_with_output())
        .await
        .expect("child finished within 120 s")
        .expect("wait");
    let peak = WindowsJobObject.read_peak(&wr.teardown);
    (out, peak)
}

/// A process handle that can be waited on, for checking a grandchild's fate.
struct Process(HANDLE);

impl Process {
    fn open(pid: u32) -> Self {
        // SAFETY: plain FFI call; the result is checked before use.
        let h = unsafe {
            OpenProcess(
                PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                0,
                pid,
            )
        };
        assert!(
            !h.is_null(),
            "open grandchild pid {pid}: {}",
            std::io::Error::last_os_error()
        );
        Process(h)
    }

    fn exited_within(&self, ms: u32) -> bool {
        // SAFETY: the handle is live until Drop.
        match unsafe { WaitForSingleObject(self.0, ms) } {
            WAIT_OBJECT_0 => true,
            WAIT_TIMEOUT => false,
            other => panic!("WaitForSingleObject returned {other}"),
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // SAFETY: we own the handle.
        unsafe {
            CloseHandle(self.0);
        }
    }
}

/// The child starts a grandchild that sleeps, prints its pid, then sleeps
/// itself. Returns the running child and a handle on the grandchild.
async fn spawn_tree(wr: &mut WrappedRun) -> (tokio::process::Child, Process) {
    let mut child = spawn_attached(&WindowsJobObject, &mut wr.command, &wr.teardown)
        .await
        .expect("spawn + attach");
    let stdout = child.stdout.take().expect("piped stdout");
    let mut lines = BufReader::new(stdout).lines();
    let line = tokio::time::timeout(Duration::from_secs(60), lines.next_line())
        .await
        .expect("child printed the grandchild pid within 60 s")
        .expect("read stdout")
        .expect("stdout not closed before the pid line");
    let pid: u32 = line.trim().parse().expect("grandchild pid");
    let grandchild = Process::open(pid);
    // Falsify the probe first: the grandchild must be alive before teardown,
    // or "it is dead afterwards" proves nothing.
    assert!(
        !grandchild.exited_within(500),
        "grandchild {pid} died before any teardown"
    );
    (child, grandchild)
}

const TREE_SCRIPT: &str = "import subprocess, sys, time\n\
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n\
print(p.pid, flush=True)\n\
time.sleep(600)\n";

#[test]
fn job_objects_are_available() {
    assert_eq!(WindowsJobObject.available(), Availability::Present);
    assert_eq!(WindowsJobObject.kind(), "windows-job-object");
}

#[test]
fn factory_selects_the_job_object_backend() {
    // Only meaningful when nothing overrides the auto order.
    if std::env::var_os("BLUT_NO_CONTAIN").is_some()
        || std::env::var_os("BLUT_CONTAINMENT").is_some()
    {
        eprintln!("skipped: BLUT_NO_CONTAIN / BLUT_CONTAINMENT set in the environment");
        return;
    }
    assert_eq!(
        blut::containment::containment_for().kind(),
        "windows-job-object"
    );
}

#[tokio::test]
async fn memory_cap_fails_an_allocation_past_it() {
    // 512 MiB allocation, 128 MiB tree-wide commit cap.
    let script = "x = bytearray(512*1024*1024); print('NO_CAP')";
    let (out, _peak) = run_capped(script, &caps(Some(128 * MIB))).await;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    println!(
        "JOB_CAP_TEST status={:?} stdout={stdout} stderr={stderr}",
        out.status
    );
    assert!(
        !out.status.success(),
        "the capped allocation must fail; status={:?}",
        out.status
    );
    assert!(
        !stdout.contains("NO_CAP"),
        "512 MiB fit under a 128 MiB cap"
    );
    assert!(
        stderr.contains("MemoryError"),
        "expected Python's MemoryError from the refused commit, got: {stderr}"
    );
}

#[tokio::test]
async fn the_same_allocation_succeeds_under_a_larger_cap() {
    // Control for the test above: identical program, a cap it fits under.
    let script = "x = bytearray(512*1024*1024); print('NO_CAP')";
    let (out, peak) = run_capped(script, &caps(Some(2048 * MIB))).await;
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("NO_CAP"));
    let peak = peak.expect("peak reported");
    assert!(
        peak >= 512 * MIB,
        "peak {peak} below the 512 MiB the child committed"
    );
}

#[tokio::test]
async fn peak_is_reported_and_covers_the_allocation() {
    // No cap at all: the job still accounts the tree's peak.
    let script = "x = bytearray(64*1024*1024); print(len(x))";
    let (out, peak) = run_capped(script, &caps(None)).await;
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let peak = peak.expect("PeakJobMemoryUsed reported");
    println!("JOB_PEAK_TEST peak={peak}");
    assert!(peak > 0);
    assert!(
        peak >= 64 * MIB,
        "peak {peak} below the 64 MiB the child committed"
    );
}

#[tokio::test]
async fn cancel_kills_the_whole_process_tree() {
    let tmp = tempfile::tempdir().unwrap();
    let mut wr = wrap(tmp.path(), TREE_SCRIPT, &caps(None));
    let (mut child, grandchild) = spawn_tree(&mut wr).await;

    assert!(
        WindowsJobObject
            .cancel("blut-windows-job-test", &wr.teardown)
            .is_none(),
        "the job is terminated inline; there is no command to spawn"
    );

    let status = tokio::time::timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("child died within 30 s of cancel")
        .expect("wait");
    assert_eq!(
        status.code(),
        Some(KILLED_EXIT_CODE),
        "child exit {status:?}"
    );
    assert!(
        grandchild.exited_within(30_000),
        "grandchild survived TerminateJobObject"
    );
}

#[tokio::test]
async fn dropping_the_last_teardown_handle_kills_the_tree() {
    let tmp = tempfile::tempdir().unwrap();
    let mut wr = wrap(tmp.path(), TREE_SCRIPT, &caps(None));
    let (mut child, grandchild) = spawn_tree(&mut wr).await;

    // KILL_ON_JOB_CLOSE: the job closes with its last handle.
    drop(wr);

    // No exit-code check: Windows reports processes killed by
    // KILL_ON_JOB_CLOSE with exit code 0 (observed on windows-latest). The
    // timeout proves the 600 s sleeper was killed; the grandchild check
    // below proves the kill reached the whole tree.
    tokio::time::timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("child died within 30 s of the job closing")
        .expect("wait");
    assert!(
        grandchild.exited_within(30_000),
        "grandchild survived the job closing"
    );
}

/// A caller that spawns the wrapped command itself, skipping
/// `spawn_attached`, used to get a child suspended forever: a silent hang.
/// It now runs (outside the job, which a warning reports at teardown).
#[tokio::test]
async fn a_direct_spawn_runs_instead_of_hanging() {
    let tmp = tempfile::tempdir().unwrap();
    let mut wr = wrap(tmp.path(), "print('ran')", &caps(None));
    let child = wr.command.spawn().expect("spawn");
    let out = tokio::time::timeout(Duration::from_secs(60), child.wait_with_output())
        .await
        .expect("a directly spawned child must not stay suspended")
        .expect("wait");
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "ran");
}

// ---- ContainedChild and LocalSystemd::spawn_contained ----
//
// The tests above drive the backend by hand. These drive the two entry points
// a stage runner uses, which must reach the same Job Object.

fn wrap_plain(cwd: &Path, script: &str, caps: &CapSpec) -> WrappedRun {
    WindowsJobObject
        .wrap_command(
            &python(),
            &["-c".into(), script.into()],
            cwd,
            &[],
            "blut-windows-contained-test",
            caps,
        )
        .expect("wrap_command")
}

fn piped(c: &mut tokio::process::Command) {
    c.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
}

/// Read the grandchild pid the child prints first, and falsify the probe:
/// the grandchild must be alive before any teardown.
async fn grandchild_of(child: &mut ContainedChild) -> Process {
    let stdout = child.child_mut().stdout.take().expect("piped stdout");
    let mut lines = BufReader::new(stdout).lines();
    let line = tokio::time::timeout(Duration::from_secs(60), lines.next_line())
        .await
        .expect("child printed the grandchild pid within 60 s")
        .expect("read stdout")
        .expect("stdout not closed before the pid line");
    let pid: u32 = line.trim().parse().expect("grandchild pid");
    let grandchild = Process::open(pid);
    assert!(
        !grandchild.exited_within(500),
        "grandchild {pid} died before any teardown"
    );
    grandchild
}

#[tokio::test]
async fn contained_child_reports_the_peak_after_exit() {
    let tmp = tempfile::tempdir().unwrap();
    let run = wrap_plain(
        tmp.path(),
        "x = bytearray(64*1024*1024); print(len(x))",
        &caps(None),
    );
    let mut child = ContainedChild::spawn(Box::new(WindowsJobObject), run, "peak", piped)
        .await
        .expect("spawn + attach");
    assert_eq!(child.kind(), "windows-job-object");
    assert_eq!(child.peak_bytes(), None, "no peak before exit");
    let status = tokio::time::timeout(Duration::from_secs(120), child.wait())
        .await
        .expect("exited within 120 s")
        .expect("wait");
    assert!(status.success(), "status {status:?}");
    let peak = child.peak_bytes().expect("peak after exit");
    assert!(peak >= 64 * MIB, "peak {peak} below the 64 MiB committed");
}

#[tokio::test]
async fn contained_child_cancel_kills_the_whole_tree() {
    let tmp = tempfile::tempdir().unwrap();
    let run = wrap_plain(tmp.path(), TREE_SCRIPT, &caps(None));
    let mut child = ContainedChild::spawn(Box::new(WindowsJobObject), run, "cancel", piped)
        .await
        .expect("spawn + attach");
    let grandchild = grandchild_of(&mut child).await;

    let cancel = CancellationToken::new();
    cancel.cancel();
    let outcome = tokio::time::timeout(Duration::from_secs(30), child.wait_or_cancel(&cancel))
        .await
        .expect("cancelled within 30 s")
        .expect("wait_or_cancel");
    assert!(outcome.is_none(), "reported as cancelled, got {outcome:?}");
    assert!(
        grandchild.exited_within(30_000),
        "grandchild survived the cancel"
    );
}

#[tokio::test]
async fn dropping_a_contained_child_kills_the_whole_tree() {
    let tmp = tempfile::tempdir().unwrap();
    let run = wrap_plain(tmp.path(), TREE_SCRIPT, &caps(None));
    let mut child = ContainedChild::spawn(Box::new(WindowsJobObject), run, "drop", piped)
        .await
        .expect("spawn + attach");
    let grandchild = grandchild_of(&mut child).await;
    // A hard stage timeout drops the stage future, and with it this value.
    drop(child);
    assert!(
        grandchild.exited_within(30_000),
        "grandchild survived the contained child being dropped"
    );
}

/// The launcher path end to end: `LocalSystemd`'s `mem_max` becomes the Job
/// Object cap. 512 MiB fails under a 128 MiB `mem_max` and fits under 2 GiB.
#[tokio::test]
async fn local_systemd_spawn_contained_caps_through_the_job_object() {
    if std::env::var_os("BLUT_NO_CONTAIN").is_some()
        || std::env::var_os("BLUT_CONTAINMENT").is_some()
    {
        eprintln!("skipped: BLUT_NO_CONTAIN / BLUT_CONTAINMENT set in the environment");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let inner = [
        "python".to_string(),
        "-c".to_string(),
        "x = bytearray(512*1024*1024); print('NO_CAP')".to_string(),
    ];
    let run = |mem_max: &str| {
        let launcher = LocalSystemd {
            mem_max: mem_max.to_string(),
            ..LocalSystemd::default()
        };
        let inner = inner.clone();
        let cwd = tmp.path().to_path_buf();
        async move {
            let mut child = launcher
                .spawn_contained("blut-launcher-test", &inner, &cwd, &[], piped)
                .await
                .expect("spawn_contained");
            assert_eq!(child.kind(), "windows-job-object");
            let mut stdout = String::new();
            let mut stderr = String::new();
            let mut out_pipe = child.child_mut().stdout.take().unwrap();
            let mut err_pipe = child.child_mut().stderr.take().unwrap();
            let (r1, r2) = tokio::join!(
                tokio::io::AsyncReadExt::read_to_string(&mut out_pipe, &mut stdout),
                tokio::io::AsyncReadExt::read_to_string(&mut err_pipe, &mut stderr),
            );
            r1.unwrap();
            r2.unwrap();
            let status = tokio::time::timeout(Duration::from_secs(120), child.wait())
                .await
                .expect("exited within 120 s")
                .expect("wait");
            (status, stdout, stderr, child.peak_bytes())
        }
    };

    let (status, stdout, stderr, _) = run("128M").await;
    assert!(!status.success(), "512 MiB fit under a 128 MiB mem_max");
    assert!(!stdout.contains("NO_CAP"));
    assert!(stderr.contains("MemoryError"), "stderr: {stderr}");

    let (status, stdout, stderr, peak) = run("2G").await;
    assert!(status.success(), "status {status:?} stderr {stderr}");
    assert!(stdout.contains("NO_CAP"));
    let peak = peak.expect("peak reported");
    assert!(peak >= 512 * MIB, "peak {peak} below the 512 MiB committed");
}
