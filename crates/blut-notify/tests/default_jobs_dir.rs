// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam

//! The daemon's default jobs directory follows the engine's `blut::paths`
//! rule. It used to be `$HOME/.local/share/lamu/train-jobs` even when
//! `$XDG_DATA_HOME` moved the engine's jobs elsewhere, so the notifier tailed
//! an empty directory.

#[cfg(target_os = "linux")]
use std::process::Command;

// `dirs::data_local_dir` reads `$XDG_DATA_HOME` on Linux only (macOS and
// Windows have their own fixed locations).
#[cfg(target_os = "linux")]
#[test]
fn the_default_jobs_dir_honours_xdg_data_home() {
    let td = tempfile::tempdir().unwrap();
    let home = td.path().join("home");
    let xdg = td.path().join("xdg");
    let job = xdg.join("lamu/train-jobs/j1");
    std::fs::create_dir_all(&job).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(
        job.join("status.jsonl"),
        "{\"kind\":\"stage_failed\",\"stage_name\":\"train\",\"error\":\"boom\"}\n",
    )
    .unwrap();
    let config = td.path().join("notify.toml");
    std::fs::write(&config, "").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_blut-notify"))
        .arg("--config")
        .arg(&config)
        .arg("--once")
        .arg("--cursor")
        .arg(td.path().join("cursor.json"))
        .arg("--sla")
        .arg(td.path().join("sla.jsonl"))
        .env("HOME", &home)
        .env("XDG_DATA_HOME", &xdg)
        .env_remove("LAMU_TRAIN_DATA_DIR")
        .env_remove("LAMU_TRAIN_JOBS_DIR")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("lines=1 "),
        "the job under $XDG_DATA_HOME must be read: {stderr}"
    );
}
