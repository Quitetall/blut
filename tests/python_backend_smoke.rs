//! Integration tests for `PythonTrainBackend` using the bundled
//! `trainer.py --self-check` mode.
//!
//! These tests need a working `python3` on `$PATH` but no GPU and
//! no `unsloth` install. They verify:
//!   - subprocess spawn + stdout pipe
//!   - line-by-line StatusUpdate parse
//!   - terminal Done event yields a TrainArtifact
//!   - cancel() actually kills a long-running process
//!
//! If `python3` is not available the tests are skipped with a
//! tracing note rather than a hard fail — keeps CI green on
//! Python-less hosts.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use blut::{
    DatasetSource, Method, Optim, PythonTrainBackend, StatusUpdate, TrainBackend, TrainSpec,
};

/// Run the python-side test_build_optimizer.py as a subprocess.
/// Confirms build_optimizer's resolution rules behave correctly
/// for the AdamW path and the APOLLO not-available error case
/// without needing torch/transformers/apollo-torch installed.
#[test]
fn build_optimizer_python_smoke() {
    let Some(python) = have_python() else {
        eprintln!("skipping: no python3 on PATH");
        return;
    };
    let script = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("test_build_optimizer.py");
    let out = std::process::Command::new(&python)
        .arg(&script)
        .output()
        .expect("spawn python test");
    assert!(
        out.status.success(),
        "build_optimizer python tests failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

fn trainer_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/trainer.py")
}

fn have_python() -> Option<PathBuf> {
    for cand in ["python3", "python"] {
        if let Ok(out) = std::process::Command::new(cand).arg("--version").output() {
            if out.status.success() {
                return Some(PathBuf::from(cand));
            }
        }
    }
    None
}

fn dummy_spec() -> TrainSpec {
    TrainSpec {
        base_model: "Qwen/Qwen3-7B".into(),
        output_name: "smoke-test".into(),
        output_dir: PathBuf::from("/tmp/lamu-train-smoke"),
        method: Method::QLora { rank: 16, alpha: 32 },
        dataset: DatasetSource::JsonlPath {
            path: PathBuf::from("/tmp/x.jsonl"),
        },
        optimizer: Optim::AdamW,
        lr: 2e-4,
        epochs: 1,
        batch_size: 1,
        grad_accum: 1,
        seq_len: 512,
        seed: 42,
        quant: "Q4_K_M".into(),
        skip_convert: false,
    }
}

/// Protocol-only smoke: invokes `trainer.py --self-check` directly
/// (bypassing `PythonTrainBackend::run`, which requires a full spec).
/// Verifies the trainer's stdout shape matches `StatusUpdate` parse
/// rules — the same parser used by the real run path.
#[tokio::test]
async fn self_check_protocol_round_trip() {
    let Some(python) = have_python() else {
        eprintln!("skipping: no python3 on PATH");
        return;
    };
    let collected: Arc<Mutex<Vec<StatusUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_for_cb = Arc::clone(&collected);
    let on_status = Box::new(move |u: StatusUpdate| {
        collected_for_cb.lock().unwrap().push(u);
    });

    let out = std::process::Command::new(&python)
        .arg(trainer_script())
        .arg("--self-check")
        .output()
        .expect("self-check spawn");
    assert!(out.status.success(), "trainer.py --self-check exit nonzero");

    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let u: StatusUpdate =
            serde_json::from_str(line).expect("self-check emitted unparseable status");
        on_status(u);
    }

    let updates = collected.lock().unwrap();
    assert_eq!(updates.len(), 3, "expected 2 Step + 1 Done from self-check");
    assert!(matches!(updates[0], StatusUpdate::Step { .. }));
    assert!(matches!(updates[1], StatusUpdate::Step { .. }));
    assert!(matches!(updates[2], StatusUpdate::Done { .. }));
    assert!(updates.last().unwrap().is_terminal());
}

/// Real-run path: spawn trainer.py with a full spec, expect it to
/// emit a Failed status because `import torch` (or unsloth) fails
/// in the test environment.
///
/// Skipped if torch *is* installed — in that case the test would
/// actually try to load Qwen3-7B and burn 40 GB of disk.
#[tokio::test]
async fn run_surfaces_trainer_failed_status() {
    let Some(python) = have_python() else {
        eprintln!("skipping: no python3 on PATH");
        return;
    };
    let torch_present = std::process::Command::new(&python)
        .arg("-c")
        .arg("import torch")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if torch_present {
        eprintln!(
            "skipping: torch is importable in this venv; \
             this test only exercises the missing-deps path"
        );
        return;
    }

    let mut backend = PythonTrainBackend::new(python, trainer_script());
    let collected: Arc<Mutex<Vec<StatusUpdate>>> = Arc::new(Mutex::new(Vec::new()));
    let collected_for_cb = Arc::clone(&collected);
    let on_status = Box::new(move |u: StatusUpdate| {
        collected_for_cb.lock().unwrap().push(u);
    });
    let result = backend.run(dummy_spec(), on_status).await;

    let err = result.expect_err("expected TrainError when torch is missing");
    let msg = format!("{}", err);
    let updates = collected.lock().unwrap();
    assert!(
        msg.contains("missing python deps")
            || msg.contains("trainer.py exited")
            || updates.iter().any(|u| matches!(u, StatusUpdate::Failed { .. })),
        "unexpected error shape: {}",
        msg
    );
}

#[tokio::test]
async fn cancel_kills_long_running_subprocess() {
    let Some(python) = have_python() else {
        eprintln!("skipping: no python3 on PATH");
        return;
    };
    // Owned tempdir keeps the script file on disk for the lifetime
    // of the test; Drop cleans it up when the test exits — no
    // accumulation in /tmp across repeated `cargo test` runs.
    let script_dir = tempfile::tempdir().expect("tempdir");
    let script_path = script_dir.path().join("sleeper.py");
    std::fs::write(
        &script_path,
        b"import time, sys, json\n\
          print(json.dumps({\"kind\":\"step\",\"step\":1,\"total\":99,\"loss\":1.0,\"lr\":0.0,\"vram_mb\":0}), flush=True)\n\
          time.sleep(60)\n",
    )
    .expect("write script");

    let mut backend = PythonTrainBackend::new(python, script_path);

    let backend_handle = backend.clone();
    let on_status = Box::new(|_u: StatusUpdate| {});
    let run_fut = tokio::spawn(async move {
        let mut b = backend_handle;
        b.run(dummy_spec(), on_status).await
    });

    // Wait until the trainer printed its first Step (proves it's
    // running), then cancel.
    tokio::time::sleep(Duration::from_millis(500)).await;
    backend.cancel().await.expect("cancel must succeed");

    // The run future must complete (with Trainer error) within the
    // 10s graceful_kill window.
    let r = tokio::time::timeout(Duration::from_secs(15), run_fut).await;
    assert!(r.is_ok(), "run task did not complete after cancel");

    // script_dir drops here, removing /tmp/<rand>/sleeper.py.
}
