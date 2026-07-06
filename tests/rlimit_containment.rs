// Real integration test for the RLIMIT_AS containment backend — proves it
// actually caps memory (a 100 GiB allocation aborts at the cap instead of
// OOMing the box). Runs on any unix box; this is the portable cloud fallback
// validated on the Thunder k8s box (read-only cgroupfs, no systemd bus).
#![cfg(unix)]

use blut::containment::{CapSpec, Containment, PeakSource, rlimit::RlimitAddressSpace};
use std::path::PathBuf;

#[tokio::test]
async fn rlimit_caps_a_runaway_allocation() {
    let be = RlimitAddressSpace;
    // 512 MiB address-space cap; the kernel allocates 100 GiB → MemoryError.
    let caps = CapSpec {
        mem_max: Some(512 * 1024 * 1024),
        mem_high: None,
        swap_max: None,
    };
    let tmp = tempfile::tempdir().unwrap();
    let wr = be
        .wrap_command(
            &PathBuf::from("python3"),
            &[
                "-c".into(),
                "x = bytearray(100*1024*1024*1024); print('NO_CAP')".into(),
            ],
            tmp.path(),
            &[],
            "blut-rlimit-test",
            &caps,
        )
        .expect("wrap");
    assert_eq!(
        wr.peak_source,
        PeakSource::None,
        "rlimit has no peak source"
    );

    let mut cmd = wr.command;
    let out = cmd.output().await.expect("spawn python");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    println!("RLIMIT_CAP_TEST status={:?}", out.status);
    println!("RLIMIT_CAP_TEST stdout={stdout}");
    println!(
        "RLIMIT_CAP_TEST stderr(last)={}",
        stderr.lines().last().unwrap_or("")
    );
    // The allocation must FAIL (non-zero exit), and must NOT print NO_CAP.
    assert!(
        !out.status.success(),
        "the capped allocation should fail (non-zero exit)"
    );
    assert!(
        !stdout.contains("NO_CAP"),
        "the 100GiB allocation must not succeed under a 512MiB cap"
    );
    // Python surfaces the cap as MemoryError (or the process is killed).
    assert!(
        stderr.contains("MemoryError") || !out.status.success(),
        "expected MemoryError or non-zero exit, got: {stderr}"
    );
}

#[tokio::test]
async fn rlimit_allows_a_small_allocation_under_the_cap() {
    let be = RlimitAddressSpace;
    // A generous 4 GiB cap; a 64 MiB allocation succeeds. Proves the cap isn't
    // so blunt that it breaks normal work (above CUDA's VA caveat threshold a
    // real GPU job needs a much larger cap — documented in the backend).
    let caps = CapSpec {
        mem_max: Some(4 * 1024 * 1024 * 1024),
        mem_high: None,
        swap_max: None,
    };
    let tmp = tempfile::tempdir().unwrap();
    let wr = be
        .wrap_command(
            &PathBuf::from("python3"),
            &[
                "-c".into(),
                "y = bytearray(64*1024*1024); print('OK', len(y))".into(),
            ],
            tmp.path(),
            &[],
            "blut-rlimit-test2",
            &caps,
        )
        .expect("wrap");
    let mut cmd = wr.command;
    let out = cmd.output().await.expect("spawn python");
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("RLIMIT_OK_TEST status={:?} stdout={stdout}", out.status);
    assert!(out.status.success(), "small alloc under cap should succeed");
    assert!(stdout.contains("OK"), "expected OK, got: {stdout}");
}
