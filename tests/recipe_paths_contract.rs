//! Recipe path-resolution contract tests (RCP-1 / RCP-6 / RCP-7).
//!
//! These run against the REAL on-disk submodule layout — that is the
//! point. The contract is "the wrapped python scripts EXIST at the
//! resolved roots." Pre-fix (single-root `~/Desktop/LamQuant`, which
//! no longer holds `ai_models/`, and a `scripts/` that was assumed to
//! sit next to `ai_models/`) `wrapped_scripts_exist_at_resolved_home`
//! and `multi_root_resolves_ai_models_and_scripts` both FAIL because
//! the resolved path does not exist. Post-fix they PASS.
//!
//! The env-mutating tests serialize on `ENV_LOCK` and restore every
//! variable they touch, so they are safe to run in parallel with the
//! filesystem-contract tests (which read no env override).

use std::path::Path;
use std::sync::Mutex;

use blut::paths::{DEFAULT_LABELS_DIR, LamquantRoots};

/// Serializes ALL tests in this file (cargo runs them in one binary
/// across threads). `BLUT_AI_MODELS` / `LAMQUANT_NEURAL` are
/// process-global, so even the read-only filesystem-contract tests
/// must hold the lock to avoid observing another test's override
/// mid-flight. Restores always run because each test holds the guard
/// for its whole body; the guard is taken with a poison-tolerant
/// helper so one failing test doesn't cascade-poison the rest.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the env lock, tolerating a poisoned mutex (a prior test
/// panicked while holding it). We only guard env ordering, not shared
/// data, so the poison is benign here.
fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    match ENV_LOCK.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Clear any override env so detection runs against the real layout.
/// Returns the prior values so the caller can restore them.
fn clear_overrides() -> (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
) {
    let prev = (
        std::env::var("BLUT_AI_MODELS").ok(),
        std::env::var("LAMQUANT_NEURAL").ok(),
        std::env::var("BLUT_SCRIPTS").ok(),
        std::env::var("BLUT_PCCP").ok(),
    );
    unsafe {
        std::env::remove_var("BLUT_AI_MODELS");
        std::env::remove_var("LAMQUANT_NEURAL");
        std::env::remove_var("BLUT_SCRIPTS");
        std::env::remove_var("BLUT_PCCP");
    }
    prev
}

fn restore_overrides(
    prev: (
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    ),
) {
    unsafe {
        restore("BLUT_AI_MODELS", prev.0);
        restore("LAMQUANT_NEURAL", prev.1);
        restore("BLUT_SCRIPTS", prev.2);
        restore("BLUT_PCCP", prev.3);
    }
}

/// The complete set of LamQuant `ai_models/*` scripts each
/// `lamquant_*` stage wraps, keyed by stage NAME. The relative path
/// is RELATIVE to `ai_models_root` and matches the exact components
/// the stage's `script_path` / `script` lookup hardcodes. This table
/// IS the drift detector: if a stage repoints its wrapped script, the
/// table must follow or the test fails.
fn ai_models_scripts() -> Vec<(&'static str, Vec<&'static str>)> {
    // Post MOVE-B (2026-05-29): only the two PCCP gate stages still
    // resolve under `ai_models_root` (the LamQuant-Neural submodule) —
    // PCCP governance (pccp_gate.py) stays in Neural. ALL training +
    // preprocessing scripts moved to `blut_python_root` and now live in
    // `blut_python_scripts()` below.
    vec![
        ("lamquant_pccp_gate_snn", vec!["ai_models", "pccp_gate.py"]),
        (
            "lamquant_pccp_gate_encoder",
            vec!["ai_models", "pccp_gate.py"],
        ),
    ]
}

/// The training + preprocessing scripts each `lamquant_*` stage wraps,
/// now resolved under `blut_python_root` ($BLUT_PYTHON →
/// `<blut>/python/lamquant/<area>/<script>.py`). This is the MOVE-B
/// analogue of `ai_models_scripts()`: the 14 train/preprocess stages
/// repointed here when their scripts moved from the PRIVATE Neural
/// `ai_models/` tree into the PUBLIC BLUT `python/lamquant/` tree. Each
/// rel path is RELATIVE to `blut_python_root` and MUST start with
/// `"python"`. This table IS the drift detector for the moved scripts.
fn blut_python_scripts() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "lamquant_train_mamba_snn",
            vec!["python", "lamquant", "snn", "train_mamba_snn.py"],
        ),
        (
            "lamquant_generate_snn_labels",
            vec!["python", "lamquant", "snn", "generate_activity_labels.py"],
        ),
        (
            "lamquant_build_manifest",
            vec!["python", "lamquant", "dataset", "build_manifest.py"],
        ),
        (
            "lamquant_build_split_manifest",
            vec![
                "python",
                "lamquant",
                "dataset",
                "build_seizure_split_manifest.py",
            ],
        ),
        (
            "lamquant_precompute_fullband",
            vec![
                "python",
                "lamquant",
                "dataset",
                "precompute_fullband_memmap.py",
            ],
        ),
        (
            "lamquant_precompute_l3",
            vec!["python", "lamquant", "student", "precompute_l3_fast.py"],
        ),
        (
            "lamquant_pretrain_mae",
            vec!["python", "lamquant", "student", "pretrain_mae.py"],
        ),
        (
            "lamquant_train_student",
            vec!["python", "lamquant", "student", "train_student_subband.py"],
        ),
        (
            "lamquant_train_joint",
            vec!["python", "lamquant", "student", "train_joint.py"],
        ),
        (
            "lamquant_harden_artifacts",
            vec!["python", "lamquant", "student", "harden_artifacts.py"],
        ),
        (
            "lamquant_train_teacher",
            vec!["python", "lamquant", "oracle", "train_teacher.py"],
        ),
        (
            "lamquant_train_l3_teacher",
            vec!["python", "lamquant", "oracle", "train_l3_teacher.py"],
        ),
        (
            "lamquant_train_vocos_decoder",
            vec!["python", "lamquant", "decoder", "train_vocos_decoder.py"],
        ),
        (
            "lamquant_train_combined",
            vec!["python", "lamquant", "decoder", "train_combined.py"],
        ),
    ]
}

/// `scripts/*` scripts (meta-repo root) each stage wraps.
fn scripts_scripts() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![(
        "lamquant_convert_lma",
        vec!["scripts", "bulk_lml_to_lma.py"],
    )]
}

/// Stages whose wrapped script genuinely does NOT exist under any
/// resolvable root in the current layout — documented gaps, NOT
/// fudged assertions.
///
/// * `lamquant_export_firmware` wraps `firmware/export_firmware.py`.
///   Post-split that file lives only in the *Lossless* submodule
///   (`LamQuant-Lossless/reference_implementations/c_firmware/export_firmware.py`),
///   not under `ai_models_root` (Neural) or the meta-repo. The export
///   stage is chained into NO recipe today (per STATE_REVIEW §4.1 TNN
///   gap), so it cannot break a runnable recipe. Excluded here with
///   this reason rather than inventing a path.
const KNOWN_GAPS: &[(&str, &str)] = &[(
    "lamquant_export_firmware",
    "firmware/export_firmware.py lives in the Lossless submodule \
     (reference_implementations/c_firmware/), not under ai_models_root \
     or the meta-repo; stage is unchained — see STATE_REVIEW §4.1",
)];

/// THE contract test (STATE_REVIEW §5.5 "Wrapped scripts EXIST at
/// resolved home"). For EVERY `lamquant_*` stage, resolve its wrapped
/// script via the fixed multi-root resolver and assert it exists on
/// disk against the current repo layout.
///
/// Would FAIL pre-fix: the old single-root home (`~/Desktop/LamQuant`)
/// resolves `<home>/ai_models/...` to a nonexistent path because
/// `ai_models/` moved into the `LamQuant-Neural` submodule.
#[test]
fn wrapped_scripts_exist_at_resolved_home() {
    let _g = env_guard();
    let prev = clear_overrides();

    let roots = LamquantRoots::resolve().expect(
        "LamQuant roots must resolve against the real submodule layout; \
         set BLUT_META_ROOT / BLUT_AI_MODELS / BLUT_SCRIPTS if running outside /mnt/4tb/LamQuant",
    );

    let mut failures: Vec<String> = Vec::new();

    for (stage, rel) in ai_models_scripts() {
        match roots.ai_models_script(&rel) {
            Ok(p) => assert!(
                p.exists(),
                "{stage}: resolver returned {} which does not exist",
                p.display()
            ),
            Err(e) => failures.push(format!("{stage}: {e}")),
        }
    }
    // MOVE-B: the 14 train/preprocess scripts now resolve under
    // blut_python_root (blut/python/lamquant/<area>/...). Same contract:
    // the resolved path MUST exist on disk.
    for (stage, rel) in blut_python_scripts() {
        match roots.blut_python_script(&rel) {
            Ok(p) => assert!(
                p.exists(),
                "{stage}: blut_python resolver returned {} which does not exist",
                p.display()
            ),
            Err(e) => failures.push(format!("{stage}: {e}")),
        }
    }
    for (stage, rel) in scripts_scripts() {
        match roots.scripts_script(&rel) {
            Ok(p) => assert!(
                p.exists(),
                "{stage}: resolver returned {} which does not exist",
                p.display()
            ),
            Err(e) => failures.push(format!("{stage}: {e}")),
        }
    }

    assert!(
        failures.is_empty(),
        "wrapped scripts missing at resolved roots (the RCP-1 drift):\n  {}",
        failures.join("\n  ")
    );

    // Document the gap explicitly: the excluded stage's script truly
    // does NOT resolve. `lamquant_export_firmware` joins
    // `<home>/firmware/export_firmware.py`; that file is absent under
    // BOTH the ai_models_root (Neural) and the meta-repo. If a future
    // commit DROPS it into one of those roots, this assertion flips
    // and the exclusion should be removed.
    for (stage, reason) in KNOWN_GAPS {
        assert_eq!(*stage, "lamquant_export_firmware", "unexpected gap entry");
        let fw_under_ai = roots
            .ai_models_root
            .join("firmware")
            .join("export_firmware.py");
        let fw_under_pccp = roots.pccp_root.join("firmware").join("export_firmware.py");
        let fw_under_scripts = roots
            .scripts_root
            .join("firmware")
            .join("export_firmware.py");
        assert!(
            !fw_under_ai.exists() && !fw_under_pccp.exists() && !fw_under_scripts.exists(),
            "{stage} no longer a gap ({reason}); one of {}, {}, {} now exists — \
             remove it from KNOWN_GAPS and add it to the script tables",
            fw_under_ai.display(),
            fw_under_pccp.display(),
            fw_under_scripts.display(),
        );
    }

    restore_overrides(prev);
}

/// STATE_REVIEW §5.5 "Multi-root resolution (scripts/ + ai_models/
/// under one home)". Assert both an `ai_models/` script AND a
/// `scripts/` script resolve under the detected roots — the two roots
/// that a single `lamquant_home` provably could NOT satisfy at once
/// pre-split.
///
/// Would FAIL pre-fix: a single root that holds `scripts/` (meta-repo)
/// does not hold `ai_models/` (Neural submodule), and vice-versa.
#[test]
fn multi_root_resolves_ai_models_and_scripts() {
    let _g = env_guard();
    let prev = clear_overrides();
    let roots = LamquantRoots::resolve().expect("roots resolve");

    // MOVE-B: the SNN trainer now resolves under blut_python_root
    // (blut/python/lamquant/snn/...), NOT ai_models_root. The PCCP gate
    // (governance) still resolves under ai_models_root (Neural).
    let snn = roots
        .blut_python_script(&["python", "lamquant", "snn", "train_mamba_snn.py"])
        .expect("train_mamba_snn.py resolves under blut_python_root");
    assert!(snn.starts_with(&roots.blut_python_root));
    assert!(snn.exists());

    let gate = roots
        .ai_models_script(&["ai_models", "pccp_gate.py"])
        .expect("pccp_gate.py resolves under ai_models_root (governance stays in Neural)");
    assert!(gate.starts_with(&roots.ai_models_root));
    assert!(gate.exists());

    let sc = roots
        .scripts_script(&["scripts", "bulk_lml_to_lma.py"])
        .expect("scripts script resolves under scripts_root");
    assert!(sc.starts_with(&roots.scripts_root));
    assert!(sc.exists());

    // The roots are genuinely distinct in the post-split + MOVE-B
    // layout: ai_models_root (Neural submodule) vs meta-repo scripts vs
    // blut_python_root (BLUT submodule). The SNN trainer must NOT
    // resolve under ai_models_root anymore — that is the whole point of
    // MOVE-B (training moved out of Neural into BLUT).
    assert_ne!(
        roots.ai_models_root, roots.scripts_root,
        "post-split layout should resolve ai_models_root and scripts_root to distinct dirs"
    );
    assert_ne!(
        roots.ai_models_root, roots.blut_python_root,
        "MOVE-B: blut_python_root (BLUT) must differ from ai_models_root (Neural)"
    );
    assert!(
        roots.blut_python_root.join("python").is_dir(),
        "blut_python_root {} must hold python/",
        roots.blut_python_root.display()
    );
    assert!(
        !snn.starts_with(&roots.ai_models_root),
        "MOVE-B: train_mamba_snn.py must NOT resolve under ai_models_root anymore"
    );

    // pccp_root holds pccp/ (RCP-9 gate context).
    assert!(
        roots.pccp_root.join("pccp").is_dir(),
        "pccp_root {} must hold pccp/",
        roots.pccp_root.display()
    );

    restore_overrides(prev);
}

/// STATE_REVIEW §5.7 "env set but used". `$BLUT_AI_MODELS` must take
/// precedence over detection and point resolution at the override.
#[test]
fn home_env_override_respected() {
    let _g = env_guard();
    let prev = clear_overrides();

    let td = tempfile::tempdir().unwrap();
    // Lay a stub ai_models/ tree under the temp override.
    let stub_script = td
        .path()
        .join("ai_models")
        .join("snn")
        .join("train_mamba_snn.py");
    std::fs::create_dir_all(stub_script.parent().unwrap()).unwrap();
    std::fs::write(&stub_script, "# stub\n").unwrap();

    unsafe {
        std::env::set_var("BLUT_AI_MODELS", td.path());
    }

    let roots = LamquantRoots::resolve().expect("roots resolve with override");
    assert_eq!(
        roots.ai_models_root,
        td.path(),
        "BLUT_AI_MODELS override must win over detection"
    );
    let resolved = roots
        .ai_models_script(&["ai_models", "snn", "train_mamba_snn.py"])
        .expect("stub script resolves under override");
    assert_eq!(resolved, stub_script);
    assert!(resolved.exists());

    restore_overrides(prev);
}

/// STATE_REVIEW §5.7 "LAMQUANT_HOME set but wrong root → assert clean
/// error not panic." Point `$BLUT_AI_MODELS` at a directory with no
/// `ai_models/` and assert resolution returns `Err` (clean), not a
/// panic / unwrap.
#[test]
fn missing_root_is_clean_error_not_panic() {
    let _g = env_guard();
    let prev = clear_overrides();

    let td = tempfile::tempdir().unwrap();
    // Point at a dir holding no ai_models/.
    unsafe {
        std::env::set_var("BLUT_AI_MODELS", td.path().join("does-not-exist"));
    }

    let r = LamquantRoots::resolve();
    assert!(
        r.is_err(),
        "a BLUT_AI_MODELS pointing at a dir with no ai_models/ must be a clean Err, got {r:?}"
    );
    // And the message must name the override knob so the user can fix it.
    let msg = format!("{}", r.unwrap_err());
    assert!(
        msg.contains("ai_models") && msg.contains("BLUT_AI_MODELS"),
        "error should name the missing subtree + the override env var; got: {msg}"
    );

    restore_overrides(prev);
}

/// RCP-6: the unified labels-dir default is the canonical Training
/// labels root and exists on disk.
#[test]
fn default_labels_dir_is_unified_and_exists() {
    assert_eq!(DEFAULT_LABELS_DIR, "/mnt/4tb/data/Training/labels");
    assert!(
        Path::new(DEFAULT_LABELS_DIR).is_dir(),
        "unified labels dir {DEFAULT_LABELS_DIR} must exist (RCP-6)"
    );
}

/// RCP-2 path contract for the NEW stages' wrapped binary/script:
///
///   * `build_seizure_split_manifest.py` (build_split_manifest stage)
///     is covered by `wrapped_scripts_exist_at_resolved_home` via the
///     `ai_models_scripts()` table — it MUST exist on disk and that
///     test asserts so. This test adds the explicit existence check so
///     the split-manifest script's presence is documented on its own.
///
///   * The `lml` encode binary (encode_lma stage) resolves to
///     `<lossless_root>/target/release/lml`. We assert the RESOLVED
///     PATH is correct (the resolver computes the right location), but
///     we do NOT fail the test when the binary is not built: the
///     release binary is produced by an operator `cargo build
///     --release` in the Lossless submodule and is intentionally not a
///     repo artifact. When it IS built (this machine's local layout),
///     we additionally assert it exists + is a file.
#[test]
fn new_stage_script_and_binary_paths_resolve() {
    let _g = env_guard();
    let prev = clear_overrides();
    // BLUT_LML must not leak in from the environment for the
    // computed-path assertion.
    let prev_lml = std::env::var("BLUT_LML").ok();
    unsafe {
        std::env::remove_var("BLUT_LML");
    }

    let roots = LamquantRoots::resolve().expect("roots resolve on current layout");

    // MOVE-B: build_seizure_split_manifest.py now resolves under
    // blut_python_root (blut/python/lamquant/dataset/...).
    let split_script = roots
        .blut_python_script(&[
            "python",
            "lamquant",
            "dataset",
            "build_seizure_split_manifest.py",
        ])
        .expect("build_seizure_split_manifest.py resolves under blut_python_root");
    assert!(
        split_script.exists() && split_script.is_file(),
        "build_seizure_split_manifest.py must exist on disk at {} (RCP-3 / MOVE-B)",
        split_script.display()
    );

    // lml binary: assert the RESOLVED PATH shape is correct.
    let lml = roots.lml_binary();
    let expected = roots
        .lossless_root
        .join("target")
        .join("release")
        .join("lml");
    assert_eq!(
        lml, expected,
        "lml binary should resolve to <lossless_root>/target/release/lml"
    );
    // Existence is gated: only assert when the release build is
    // present. `cargo build --release` in the Lossless submodule
    // produces it; a fresh checkout legitimately won't have it.
    if lml.exists() {
        assert!(
            lml.is_file(),
            "resolved lml path {} exists but is not a file",
            lml.display()
        );
    } else {
        eprintln!(
            "note: lml binary not built at {} — build it with `cargo build --release` \
             in the Lossless submodule (encode_lma asserts existence at run-time preflight)",
            lml.display()
        );
    }

    unsafe {
        match prev_lml {
            Some(v) => std::env::set_var("BLUT_LML", v),
            None => std::env::remove_var("BLUT_LML"),
        }
    }
    restore_overrides(prev);
}

unsafe fn restore(key: &str, prev: Option<String>) {
    match prev {
        Some(v) => unsafe { std::env::set_var(key, v) },
        None => unsafe { std::env::remove_var(key) },
    }
}
