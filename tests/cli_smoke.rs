//! End-to-end smoke for the lamu-train CLI binary.
//!
//! Invokes the compiled binary as a subprocess so flag parsing,
//! subcommand dispatch, and exit codes match what users see.
//! Uses `--help` and `jobs` (no-op when no jobs) to avoid spawning
//! the trainer.

use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn binary() -> PathBuf {
    // Cargo puts the binary at target/<profile>/<name> when running
    // integration tests; CARGO_BIN_EXE_<name> is the env var Cargo
    // sets so tests don't have to guess the profile dir.
    PathBuf::from(env!("CARGO_BIN_EXE_blut"))
}

#[test]
fn help_prints_top_level() {
    let out = Command::new(binary())
        .arg("--help")
        .output()
        .expect("spawn lamu-train");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("lamu-train"), "missing tool name in help");
    assert!(stdout.contains("Local fine-tuning"), "missing about line");
}

#[test]
fn train_help_lists_critical_flags() {
    let out = Command::new(binary())
        .arg("train")
        .arg("--help")
        .output()
        .expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    for needle in ["--base", "--dataset", "--method", "--optim", "--allow-evict", "--background"] {
        assert!(stdout.contains(needle), "train help missing {needle}");
    }
}

#[test]
fn jobs_subcommand_handles_empty() {
    let _g = ENV_LOCK.lock().unwrap();
    // Point at an empty jobs dir; subcommand must succeed + print
    // the no-jobs message rather than erroring.
    let td = tempfile::tempdir().unwrap();
    let out = Command::new(binary())
        .env("LAMU_TRAIN_JOBS_DIR", td.path())
        .arg("jobs")
        .output()
        .expect("spawn");
    assert!(out.status.success(), "jobs on empty dir must succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("no jobs"), "expected no-jobs message, got: {stdout}");
}

#[test]
fn cancel_unknown_id_errors() {
    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let out = Command::new(binary())
        .env("LAMU_TRAIN_JOBS_DIR", td.path())
        .arg("cancel")
        .arg("nonexistent-job-id")
        .output()
        .expect("spawn");
    assert!(!out.status.success(), "cancel of unknown id must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no job matches") || stderr.contains("nonexistent"),
        "stderr: {stderr}"
    );
}

#[test]
fn data_list_empty_returns_no_datasets() {
    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let db = td.path().join("test.db");
    let out = Command::new(binary())
        .env("LAMU_MEMORY_DB", &db)
        .args(["data", "list"])
        .output()
        .expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("no datasets"), "got: {stdout}");
}

#[test]
fn data_add_then_list_then_rm() {
    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let db = td.path().join("test.db");
    let jsonl = td.path().join("data.jsonl");
    std::fs::write(
        &jsonl,
        b"{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}\n\
          {\"messages\":[{\"role\":\"assistant\",\"content\":\"hello\"}]}\n",
    )
    .unwrap();

    let out = Command::new(binary())
        .env("LAMU_MEMORY_DB", &db)
        .args(["data", "add", "smoke-ds"])
        .arg(&jsonl)
        .output()
        .expect("spawn add");
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("registered 'smoke-ds'"));

    let out = Command::new(binary())
        .env("LAMU_MEMORY_DB", &db)
        .args(["data", "list"])
        .output()
        .expect("spawn list");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("smoke-ds"), "list output: {stdout}");
    assert!(stdout.contains("2"), "expected example count: {stdout}");

    let out = Command::new(binary())
        .env("LAMU_MEMORY_DB", &db)
        .args(["data", "rm", "smoke-ds"])
        .output()
        .expect("spawn rm");
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("removed 'smoke-ds'"));
}

#[test]
fn policy_show_prints_defaults_when_missing() {
    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let policy = td.path().join("policy.toml");
    let out = Command::new(binary())
        .env("LAMU_TRAIN_POLICY", &policy)
        .args(["policy", "show"])
        .output()
        .expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("enabled = false"));
    assert!(stdout.contains("Qwen/Qwen3-7B"));
}

#[test]
fn policy_enable_writes_file_and_prints_cron() {
    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let policy = td.path().join("policy.toml");
    let out = Command::new(binary())
        .env("LAMU_TRAIN_POLICY", &policy)
        .args(["policy", "enable"])
        .output()
        .expect("spawn");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("auto-trigger enabled"));
    assert!(stdout.contains("crontab") || stdout.contains("*/30 *"));
    assert!(policy.exists(), "policy file must be written");
    let body = std::fs::read_to_string(&policy).unwrap();
    assert!(body.contains("enabled = true"));
}

#[test]
fn auto_disabled_policy_skips_clean() {
    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let policy = td.path().join("policy.toml");
    // Default policy is disabled → auto skips with "disabled" reason.
    let out = Command::new(binary())
        .env("LAMU_TRAIN_POLICY", &policy)
        .arg("auto")
        .output()
        .expect("spawn");
    assert!(out.status.success(), "auto with disabled policy must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("disabled"), "stdout: {stdout}");
}

#[test]
fn data_rm_unknown_errors() {
    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();
    let db = td.path().join("test.db");
    let out = Command::new(binary())
        .env("LAMU_MEMORY_DB", &db)
        .args(["data", "rm", "definitely-not-here"])
        .output()
        .expect("spawn");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("definitely-not-here"));
}

#[test]
fn log_subcommand_renders_status() {
    use blut::jobs::{self, JobState};
    use blut::protocol::StatusUpdate;
    use std::path::PathBuf;

    let _g = ENV_LOCK.lock().unwrap();
    let td = tempfile::tempdir().unwrap();

    // Set the env var for both the test setup AND the spawned binary.
    let prev = std::env::var("LAMU_TRAIN_JOBS_DIR").ok();
    // SAFETY: ENV_LOCK serializes mutations across this test binary.
    unsafe {
        std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path());
    }

    let id = "20260510-120000-000000001";
    jobs::write_state(id, JobState::Done).unwrap();
    jobs::append_status(
        id,
        &StatusUpdate::Step {
            step: 1,
            total: 10,
            loss: 1.5,
            lr: 2e-4,
            vram_mb: 8000,
        },
    )
    .unwrap();
    jobs::append_status(
        id,
        &StatusUpdate::Done {
            final_loss: 0.5,
            checkpoint_dir: PathBuf::from("/tmp/ckpt"),
        },
    )
    .unwrap();

    let out = Command::new(binary())
        .env("LAMU_TRAIN_JOBS_DIR", td.path())
        .arg("log")
        .arg(id)
        .output()
        .expect("spawn");
    assert!(out.status.success(), "log subcommand must succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("step 1/10"), "log must render step lines: {stdout}");
    assert!(stdout.contains("done"), "log must render done line: {stdout}");

    unsafe {
        match prev {
            Some(v) => std::env::set_var("LAMU_TRAIN_JOBS_DIR", v),
            None => std::env::remove_var("LAMU_TRAIN_JOBS_DIR"),
        }
    }
}
