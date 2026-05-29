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

use blut::paths::{LamquantRoots, DEFAULT_LABELS_DIR};

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
fn clear_overrides() -> (Option<String>, Option<String>, Option<String>, Option<String>) {
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

fn restore_overrides(prev: (Option<String>, Option<String>, Option<String>, Option<String>)) {
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
    vec![
        (
            "lamquant_train_mamba_snn",
            vec!["ai_models", "snn", "train_mamba_snn.py"],
        ),
        (
            "lamquant_generate_snn_labels",
            vec!["ai_models", "snn", "generate_activity_labels.py"],
        ),
        (
            "lamquant_build_manifest",
            vec!["ai_models", "dataset_sim", "build_manifest.py"],
        ),
        (
            "lamquant_precompute_fullband",
            vec!["ai_models", "dataset_sim", "precompute_fullband_memmap.py"],
        ),
        (
            "lamquant_precompute_l3",
            vec!["ai_models", "student", "precompute_l3_fast.py"],
        ),
        (
            "lamquant_pretrain_mae",
            vec!["ai_models", "student", "pretrain_mae.py"],
        ),
        (
            "lamquant_train_student",
            vec!["ai_models", "student", "train_student_subband.py"],
        ),
        (
            "lamquant_train_joint",
            vec!["ai_models", "student", "train_joint.py"],
        ),
        (
            "lamquant_harden_artifacts",
            vec!["ai_models", "student", "harden_artifacts.py"],
        ),
        (
            "lamquant_train_teacher",
            vec!["ai_models", "oracle", "train_teacher.py"],
        ),
        (
            "lamquant_train_l3_teacher",
            vec!["ai_models", "oracle", "train_l3_teacher.py"],
        ),
        (
            "lamquant_train_vocos_decoder",
            vec!["ai_models", "decoder", "train_vocos_decoder.py"],
        ),
        (
            "lamquant_train_combined",
            vec!["ai_models", "decoder", "train_combined.py"],
        ),
        (
            "lamquant_pccp_gate_snn",
            vec!["ai_models", "pccp_gate.py"],
        ),
        (
            "lamquant_pccp_gate_encoder",
            vec!["ai_models", "pccp_gate.py"],
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
        let fw_under_pccp = roots
            .pccp_root
            .join("firmware")
            .join("export_firmware.py");
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

    let ai = roots
        .ai_models_script(&["ai_models", "snn", "train_mamba_snn.py"])
        .expect("ai_models script resolves under ai_models_root");
    assert!(ai.starts_with(&roots.ai_models_root));
    assert!(ai.exists());

    let sc = roots
        .scripts_script(&["scripts", "bulk_lml_to_lma.py"])
        .expect("scripts script resolves under scripts_root");
    assert!(sc.starts_with(&roots.scripts_root));
    assert!(sc.exists());

    // The two roots are genuinely distinct in the post-split layout
    // (Neural submodule vs meta-repo) — that's the whole point of
    // multi-root resolution. (If a monorepo ever collapses them this
    // assertion would need revisiting, but the current contract is
    // the split layout.)
    assert_ne!(
        roots.ai_models_root, roots.scripts_root,
        "post-split layout should resolve ai_models_root and scripts_root to distinct dirs"
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
    let stub_script = td.path().join("ai_models").join("snn").join("train_mamba_snn.py");
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

unsafe fn restore(key: &str, prev: Option<String>) {
    match prev {
        Some(v) => unsafe { std::env::set_var(key, v) },
        None => unsafe { std::env::remove_var(key) },
    }
}
