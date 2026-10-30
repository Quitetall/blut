// The systemd backend through `LocalSystemd::spawn_contained`, on a host with
// a systemd user manager (systemd 250 or later). Skipped, loudly, elsewhere:
// CI runners usually have no user bus, and this is what the fallback tests in
// `contained_child.rs` cover. One test function so the environment it sets is
// never seen by a concurrent test.
//
// What it pins, each observed from outside:
//   - the stage runs in a transient `systemd-run --user` service with the
//     launcher's MemoryMax, and the exit status passes through;
//   - it sees the caller's environment, and an explicit value wins;
//   - the systemd peak is read from stderr when it is pumped, and every line
//     still reaches the sink;
//   - cancelling kills the whole tree, the grandchild included, and stops the
//     unit, although the direct child is only the systemd-run client;
//   - dropping the value stops the unit too;
//   - no unit is left behind.
#![cfg(target_os = "linux")]

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use blut::config::launcher::LocalSystemd;
use tokio_util::sync::CancellationToken;

fn user_bus_with_setenv_by_name() -> bool {
    let bus = std::process::Command::new("systemctl")
        .args(["--user", "show-environment"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    let version = std::process::Command::new("systemd-run")
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

fn unit_state(unit: &str) -> String {
    std::process::Command::new("systemctl")
        .args(["--user", "is-active", &format!("{unit}.service")])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

async fn wait_unit_gone(unit: &str) {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let state = unit_state(unit);
        if !matches!(state.as_str(), "active" | "activating" | "deactivating") {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{unit} still {state} 30 s after teardown"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn alive(pid: i32) -> bool {
    if nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_err() {
        return false;
    }
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| {
            s.rsplit_once(')')
                .map(|(_, r)| r.trim_start().starts_with('Z'))
        })
        .is_none_or(|zombie| !zombie)
}

async fn read_pids(path: &Path) -> Vec<i32> {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(s) = std::fs::read_to_string(path) {
            let pids: Vec<i32> = s
                .split_whitespace()
                .filter_map(|p| p.parse().ok())
                .collect();
            if pids.len() == 2 {
                return pids;
            }
        }
        assert!(std::time::Instant::now() < deadline, "no pids in {path:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// `sh` that records its own pid and a sleeping grandchild's, then waits.
fn tree_script(pidfile: &Path) -> Vec<String> {
    let script = format!(
        "sleep 600 & echo $$ $! > {}.tmp && mv {0}.tmp {0}; wait",
        pidfile.display()
    );
    vec!["sh".into(), "-c".into(), script]
}

#[tokio::test]
async fn the_systemd_backend_contains_reports_and_tears_down() {
    if !user_bus_with_setenv_by_name() {
        eprintln!("skipped: no systemd user bus here, or systemd older than 250");
        return;
    }
    // SAFETY: the only test in this binary.
    unsafe {
        std::env::remove_var("BLUT_NO_CONTAIN");
        std::env::set_var("BLUT_CONTAINMENT", "systemd");
        std::env::set_var("BLUT_TEST_INHERITED", "from-the-driver");
        std::env::set_var("BLUT_TEST_OVERRIDDEN", "inherited");
    }
    let tmp = tempfile::tempdir().unwrap();
    let launcher = LocalSystemd {
        mem_max: "256M".into(),
        ..LocalSystemd::default()
    };

    // 1. Placement, cap, environment, exit status, peak through the pump.
    let inner = vec![
        "sh".to_string(),
        "-c".to_string(),
        "echo \"inherited=$BLUT_TEST_INHERITED overridden=$BLUT_TEST_OVERRIDDEN\" >&2; \
         printf 'verbatim=%s\\n' \"$1\" >&2; cat /proc/self/cgroup >&2; exit 3"
            .to_string(),
        "sh".to_string(),
        // systemd expands `$` in a service's command line unless told not to.
        "a$b$$c${HOME}".to_string(),
    ];
    let mut child = launcher
        .spawn_contained(
            "blut-systemd-test",
            &inner,
            tmp.path(),
            &[("BLUT_TEST_OVERRIDDEN".into(), "explicit".into())],
            |c| {
                c.stderr(std::process::Stdio::piped());
            },
        )
        .await
        .expect("spawn_contained");
    assert_eq!(child.kind(), "systemd");
    let unit = child.unit().to_string();
    let lines = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = Arc::clone(&lines);
    assert!(child.pump_stderr(move |l| sink.lock().unwrap().push(l)));
    let status = tokio::time::timeout(Duration::from_secs(60), child.wait())
        .await
        .expect("exited")
        .expect("wait");
    assert_eq!(status.code(), Some(3), "the service's exit status");
    let text = lines.lock().unwrap().join("\n");
    assert!(
        text.contains("inherited=from-the-driver overridden=explicit"),
        "environment: {text}"
    );
    assert!(
        text.contains("verbatim=a$b$$c${HOME}"),
        "an argument was rewritten on its way through systemd-run: {text}"
    );
    assert!(
        text.contains(&format!("{unit}.service")),
        "not in its own unit's cgroup: {text}"
    );
    let peak = child.peak_bytes().expect("Memory peak scraped from stderr");
    assert!(peak > 0, "peak {peak}");
    wait_unit_gone(&unit).await;

    // The cap reached the unit: while one runs, its MemoryMax is the
    // launcher's 256 MiB.
    let pidfile = tmp.path().join("tree.pids");
    let mut child = launcher
        .spawn_contained(
            "blut-systemd-test",
            &tree_script(&pidfile),
            tmp.path(),
            &[],
            |_| {},
        )
        .await
        .expect("spawn tree");
    let unit = child.unit().to_string();
    let pids = read_pids(&pidfile).await;
    let max = std::process::Command::new("systemctl")
        .args([
            "--user",
            "show",
            "-p",
            "MemoryMax",
            "--value",
            &format!("{unit}.service"),
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap();
    assert_eq!(
        max,
        (256u64 * 1024 * 1024).to_string(),
        "MemoryMax of {unit}"
    );

    // 2. Cancel kills the whole tree and stops the unit.
    for &pid in &pids {
        assert!(alive(pid), "pid {pid} was not running before the cancel");
    }
    let cancel = CancellationToken::new();
    cancel.cancel();
    let outcome = tokio::time::timeout(Duration::from_secs(60), child.wait_or_cancel(&cancel))
        .await
        .expect("cancelled within 60 s")
        .expect("wait_or_cancel");
    assert!(outcome.is_none());
    for &pid in &pids {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while alive(pid) {
            assert!(
                std::time::Instant::now() < deadline,
                "pid {pid} survived the cancel"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    wait_unit_gone(&unit).await;

    // 3. Killing only the direct child (the systemd-run client) must not
    // orphan the service: the teardown after it exits stops the unit.
    std::fs::remove_file(&pidfile).unwrap();
    let mut child = launcher
        .spawn_contained(
            "blut-systemd-test",
            &tree_script(&pidfile),
            tmp.path(),
            &[],
            |_| {},
        )
        .await
        .expect("spawn tree");
    let unit = child.unit().to_string();
    let pids = read_pids(&pidfile).await;
    let client = child.id().expect("client pid");
    nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(client as i32),
        nix::sys::signal::Signal::SIGKILL,
    )
    .unwrap();
    let _ = tokio::time::timeout(Duration::from_secs(60), child.wait())
        .await
        .expect("client reaped");
    for &pid in &pids {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while alive(pid) {
            assert!(
                std::time::Instant::now() < deadline,
                "pid {pid} outlived its killed client"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    wait_unit_gone(&unit).await;

    // 4. Dropping the value before the child exits stops the unit too.
    std::fs::remove_file(&pidfile).unwrap();
    let child = launcher
        .spawn_contained(
            "blut-systemd-test",
            &tree_script(&pidfile),
            tmp.path(),
            &[],
            |c| {
                c.kill_on_drop(true);
            },
        )
        .await
        .expect("spawn tree");
    let unit = child.unit().to_string();
    let pids = read_pids(&pidfile).await;
    drop(child);
    for &pid in &pids {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while alive(pid) {
            assert!(
                std::time::Instant::now() < deadline,
                "pid {pid} survived the drop"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    wait_unit_gone(&unit).await;
}
