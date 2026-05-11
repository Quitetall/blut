//! Integration test for `blut::convert::convert_to_gguf`.
//!
//! No real llama.cpp required: creates a fake `BLUT_LLAMACPP_DIR`
//! containing shell stubs for `convert_hf_to_gguf.py` (parses argv,
//! writes a fake f16 file) and `llama-quantize` (writes a fake
//! quantized file). Verifies the orchestration:
//!
//!   - both binaries are spawned with the right argv
//!   - intermediate f16 is removed after a successful quantize
//!   - skip_quantize path (quant == "f16") returns the f16 path
//!   - failure cases return TrainError::Convert with a useful message

use std::path::PathBuf;
use std::sync::Mutex;

use blut::convert::convert_to_gguf;

// All env-var mutating tests in this binary serialize on this lock
// because $BLUT_LLAMACPP_DIR is process-global.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn write_executable(path: &PathBuf, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, body).expect("write stub");
    let mut perms = std::fs::metadata(path).expect("stat").permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms).expect("chmod");
}

/// Returns (tempdir guard, llamacpp_dir, ckpt_dir).
/// `llamacpp_dir/build/bin` holds the convert + quantize stubs.
/// `ckpt_dir` is a fake HF checkpoint folder.
fn setup() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let td = tempfile::tempdir().expect("tempdir");
    let llamacpp = td.path().join("llama.cpp");
    let bin = llamacpp.join("build").join("bin");
    std::fs::create_dir_all(&bin).expect("mkdir bin");
    let ckpt = td.path().join("ckpt");
    std::fs::create_dir_all(&ckpt).expect("mkdir ckpt");
    (td, llamacpp, ckpt)
}

#[tokio::test]
async fn happy_path_writes_quantized_and_removes_f16() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, llamacpp, ckpt) = setup();
    let bin = llamacpp.join("build").join("bin");

    // convert stub: writes the --outfile path with dummy bytes.
    // (`/usr/bin/env python3 convert_hf_to_gguf.py <ckpt> --outfile X`)
    // We use a Python script the parent invokes via `python3 <stub>`.
    let convert = bin.join("convert_hf_to_gguf.py");
    write_executable(
        &convert,
        "#!/usr/bin/env python3\n\
         import sys\n\
         out = sys.argv[sys.argv.index('--outfile') + 1]\n\
         open(out, 'wb').write(b'fake-f16-gguf')\n",
    );

    // quantize stub: copies first arg to second, ignores third (quant
    // type) — just enough to satisfy the "produced a file" check.
    let quantize = bin.join("llama-quantize");
    write_executable(
        &quantize,
        "#!/bin/sh\nset -e\ncp \"$1\" \"$2\"\n",
    );

    let prev = std::env::var("BLUT_LLAMACPP_DIR").ok();
    unsafe {
        std::env::set_var("BLUT_LLAMACPP_DIR", &llamacpp);
    }

    let result = convert_to_gguf(&ckpt, "test-model", "Q4_K_M").await;
    let final_path = result.expect("convert_to_gguf must succeed");

    assert!(final_path.exists(), "final file must exist: {}", final_path.display());
    assert!(final_path.to_string_lossy().ends_with("test-model.Q4_K_M.gguf"));

    // f16 intermediate must be cleaned up.
    let f16 = ckpt.parent().unwrap().join("test-model.f16.gguf");
    assert!(!f16.exists(), "f16 intermediate must be removed");

    unsafe {
        match prev {
            Some(v) => std::env::set_var("BLUT_LLAMACPP_DIR", v),
            None => std::env::remove_var("BLUT_LLAMACPP_DIR"),
        }
    }
}

#[tokio::test]
async fn f16_quant_skips_quantize_step() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, llamacpp, ckpt) = setup();
    let bin = llamacpp.join("build").join("bin");

    let convert = bin.join("convert_hf_to_gguf.py");
    write_executable(
        &convert,
        "#!/usr/bin/env python3\n\
         import sys\n\
         out = sys.argv[sys.argv.index('--outfile') + 1]\n\
         open(out, 'wb').write(b'fake-f16-gguf')\n",
    );
    // No quantize stub at all — proves the f16 path doesn't try to
    // invoke it.

    let prev = std::env::var("BLUT_LLAMACPP_DIR").ok();
    unsafe {
        std::env::set_var("BLUT_LLAMACPP_DIR", &llamacpp);
    }

    let result = convert_to_gguf(&ckpt, "test-model", "f16").await;
    let final_path = result.expect("f16 mode must succeed without quantize");
    assert!(final_path.to_string_lossy().ends_with("test-model.f16.gguf"));
    assert!(final_path.exists());

    unsafe {
        match prev {
            Some(v) => std::env::set_var("BLUT_LLAMACPP_DIR", v),
            None => std::env::remove_var("BLUT_LLAMACPP_DIR"),
        }
    }
}

#[tokio::test]
async fn quantize_failure_keeps_f16_for_retry() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, llamacpp, ckpt) = setup();
    let bin = llamacpp.join("build").join("bin");

    let convert = bin.join("convert_hf_to_gguf.py");
    write_executable(
        &convert,
        "#!/usr/bin/env python3\n\
         import sys\n\
         out = sys.argv[sys.argv.index('--outfile') + 1]\n\
         open(out, 'wb').write(b'fake-f16-gguf')\n",
    );
    // Quantize stub that always exits non-zero.
    let quantize = bin.join("llama-quantize");
    write_executable(&quantize, "#!/bin/sh\nexit 7\n");

    let prev = std::env::var("BLUT_LLAMACPP_DIR").ok();
    unsafe {
        std::env::set_var("BLUT_LLAMACPP_DIR", &llamacpp);
    }

    let err = convert_to_gguf(&ckpt, "test-model", "Q4_K_M")
        .await
        .expect_err("quantize failure must error");
    let msg = format!("{err}");
    assert!(msg.contains("llama-quantize exited"), "msg: {msg}");
    assert!(
        msg.contains("manual retry"),
        "error must mention retry path: {msg}"
    );
    // f16 intermediate must NOT be removed when quantize fails.
    let f16 = ckpt.parent().unwrap().join("test-model.f16.gguf");
    assert!(f16.exists(), "f16 must survive a failed quantize");

    unsafe {
        match prev {
            Some(v) => std::env::set_var("BLUT_LLAMACPP_DIR", v),
            None => std::env::remove_var("BLUT_LLAMACPP_DIR"),
        }
    }
}

#[tokio::test]
async fn missing_checkpoint_dir_errors_clean() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let err = convert_to_gguf(
        &PathBuf::from("/definitely/does/not/exist/ckpt"),
        "test",
        "Q4_K_M",
    )
    .await
    .expect_err("missing ckpt must error");
    assert!(format!("{err}").contains("does not exist"));
}

#[tokio::test]
async fn missing_convert_tool_errors_with_env_var_hint() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (_td, llamacpp, ckpt) = setup();
    // No stubs created — convert_hf_to_gguf.py is missing.

    let prev = std::env::var("BLUT_LLAMACPP_DIR").ok();
    unsafe {
        std::env::set_var("BLUT_LLAMACPP_DIR", &llamacpp);
    }

    let err = convert_to_gguf(&ckpt, "test", "Q4_K_M")
        .await
        .expect_err("missing tool must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("BLUT_LLAMACPP_DIR"),
        "error must hint at env var: {msg}"
    );

    unsafe {
        match prev {
            Some(v) => std::env::set_var("BLUT_LLAMACPP_DIR", v),
            None => std::env::remove_var("BLUT_LLAMACPP_DIR"),
        }
    }
}
