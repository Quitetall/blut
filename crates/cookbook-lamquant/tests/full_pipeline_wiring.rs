//! Full-pipeline wiring + new-stage contract tests (RCP-2 / RCP-3 +
//! STATE_REVIEW §4.1).
//!
//! These cover the three new pieces that close the end-to-end gap:
//!
//!   * `lamquant_encode_lma` (RCP-2) — EDF→.lma via the `lml` binary.
//!   * `lamquant_build_split_manifest` (RCP-3) — patient-level split.
//!   * `lamquant_full_pipeline` — the recipe chaining encode → split →
//!     train → gate.
//!
//! Scope is compilation, wiring (n_nodes/n_edges + topo), arg
//! rejection, and path/binary resolution. The tests DO NOT run encode
//! or training (no `lml`, no GPU) — they exercise the preflight + plan
//! layers only.

use std::path::PathBuf;

use blut::framework::artifact::{Artifact, ContentHash};
use blut::framework::error::StageError;
use blut::framework::stage::{Stage, StageContext};
use cookbook_lamquant::LAMQUANT_RECIPES;
use cookbook_lamquant::artifacts::{LmaCorpus, SplitManifest};
use cookbook_lamquant::paths::LamquantRoots;
use cookbook_lamquant::stages::lamquant_build_split_manifest::{
    Args as SplitArgs, LamquantBuildSplitManifest,
};
use cookbook_lamquant::stages::lamquant_encode_lma::{Args as EncodeArgs, LamquantEncodeLma};

fn ctx(td: &std::path::Path) -> StageContext {
    std::fs::create_dir_all(td.join("stage")).unwrap();
    StageContext::for_test(td.to_path_buf(), td.join("stage"))
}

/// RCP-2: the encode stage declares NAME/SCHEMA, rejects bad args, and
/// the `lml` binary path is computed (the resolver does not panic) on
/// the current multi-root layout.
#[tokio::test]
async fn encode_lma_stage_compiles_and_args_reject() {
    // NAME / SCHEMA present + output reuses the EXISTING LmaCorpus.
    assert_eq!(LamquantEncodeLma::NAME, "lamquant_encode_lma");
    assert_eq!(LamquantEncodeLma::SCHEMA, 1);
    assert_eq!(
        <<LamquantEncodeLma as Stage>::Output as Artifact>::KIND,
        "lamquant.lma_corpus"
    );

    let td = tempfile::tempdir().unwrap();

    // Empty edf_dir → BadInput.
    let r = LamquantEncodeLma
        .run(
            &ctx(td.path()),
            (),
            &EncodeArgs {
                edf_dir: PathBuf::new(),
                out_dir: td.path().join("out"),
                corpus: String::new(),
                quiet: true,
                verify: false,
            },
        )
        .await;
    assert!(matches!(r, Err(StageError::BadInput(_))), "empty edf_dir");

    // Nonexistent edf_dir → BadInput.
    let r = LamquantEncodeLma
        .run(
            &ctx(td.path()),
            (),
            &EncodeArgs {
                edf_dir: td.path().join("no-such-corpus"),
                out_dir: td.path().join("out"),
                corpus: String::new(),
                quiet: true,
                verify: false,
            },
        )
        .await;
    assert!(
        matches!(r, Err(StageError::BadInput(_))),
        "nonexistent edf_dir"
    );

    // The `lml` binary path is computed without panic, and resolves to
    // the expected location under the Lossless submodule. We assert the
    // PATH SHAPE (not existence — the release binary may not be built).
    let roots = LamquantRoots::resolve().expect("roots resolve on current layout");
    let lml = roots.lml_binary();
    assert!(
        lml.ends_with("target/release/lml"),
        "lml binary should resolve to <lossless_root>/target/release/lml; got {}",
        lml.display()
    );
    assert!(
        lml.starts_with(&roots.lossless_root),
        "lml binary must live under lossless_root {}",
        roots.lossless_root.display()
    );
}

/// RCP-3: the split-manifest stage declares NAME/SCHEMA, produces the
/// new SplitManifest artifact, and rejects bad args.
#[tokio::test]
async fn build_split_manifest_stage_compiles() {
    assert_eq!(
        LamquantBuildSplitManifest::NAME,
        "lamquant_build_split_manifest"
    );
    assert_eq!(LamquantBuildSplitManifest::SCHEMA, 1);
    assert_eq!(
        <<LamquantBuildSplitManifest as Stage>::Output as Artifact>::KIND,
        "lamquant.split_manifest"
    );

    let td = tempfile::tempdir().unwrap();
    let corpus = LmaCorpus {
        root: td.path().to_path_buf(),
        n_archives: 1,
        content_hash: ContentHash::of_bytes(b"c"),
    };

    // Empty `out` → BadInput.
    let r = LamquantBuildSplitManifest
        .run(
            &ctx(td.path()),
            corpus.clone(),
            &SplitArgs {
                lamquant_home: td.path().display().to_string(),
                lma_root: String::new(),
                labels_dir: String::new(),
                out: PathBuf::new(),
                val_fraction: 0.10,
            },
        )
        .await;
    assert!(matches!(r, Err(StageError::BadInput(_))), "empty out");

    // Out-of-range val_fraction → BadInput.
    let r = LamquantBuildSplitManifest
        .run(
            &ctx(td.path()),
            corpus,
            &SplitArgs {
                lamquant_home: td.path().display().to_string(),
                lma_root: String::new(),
                labels_dir: String::new(),
                out: td.path().join("split.json"),
                val_fraction: 2.0,
            },
        )
        .await;
    assert!(
        matches!(r, Err(StageError::BadInput(_))),
        "out-of-range val_fraction"
    );

    // SplitManifest round-trips through serde (artifact is persisted as
    // bincode by the executor; assert the typed handle survives JSON).
    let sm = SplitManifest {
        path: PathBuf::from("/tmp/split.json"),
        content_hash: ContentHash::of_bytes(b"{}"),
        n_train_subjects: 9,
        n_val_subjects: 1,
    };
    let json = serde_json::to_string(&sm).unwrap();
    let back: SplitManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.n_train_subjects, 9);
    assert_eq!(back.n_val_subjects, 1);
    // SplitManifest is small JSON → hashes bytes (HASH_CONTENTS=true).
    const { assert!(SplitManifest::HASH_CONTENTS) };
}

/// The full-pipeline recipe compiles to the expected 5-node / 4-edge
/// linear plan, appears in RECIPES by name, and its arg schema
/// generates cleanly.
#[test]
fn full_pipeline_recipe_wires() {
    let def = LAMQUANT_RECIPES
        .iter()
        .copied()
        .find(|r| r.name == "lamquant_full_pipeline")
        .expect("lamquant_full_pipeline must be registered in RECIPES");

    // Arg schema generates (non-null).
    let schema = (def.args_schema_fn)();
    assert!(schema != serde_json::Value::Null);

    // Compile via the erased compile_fn with the minimal required args.
    let raw = serde_json::json!({
        "edf_dir": "/tmp/edf",
        "lma_output_dir": "/tmp/lma",
        "split_manifest": "/tmp/split.json",
        "eeg_dir": "/tmp/eeg",
    });
    let plan = (def.compile_fn)(raw).expect("full_pipeline compiles");
    // encode_lma → build_split_manifest → _corpus_rebind_from_split →
    // train_mamba_snn → pccp_gate_snn.
    assert_eq!(plan.n_nodes(), 5, "expected 5-node chain");
    assert_eq!(plan.n_edges(), 4, "expected 4 edges (linear chain)");
    let order = plan.topo_order().expect("topo order");
    assert_eq!(order, vec![0, 1, 2, 3, 4]);
}

/// Catalog presence: `lamquant_full_pipeline` is in RECIPES with the
/// Pipeline category + the right output kind.
#[test]
fn full_pipeline_in_catalog() {
    let def = LAMQUANT_RECIPES
        .iter()
        .copied()
        .find(|r| r.name == "lamquant_full_pipeline")
        .expect("lamquant_full_pipeline missing from RECIPES");
    assert_eq!(def.category, blut::recipes::RecipeCategory::Pipeline);
    assert_eq!(def.output_kind, "lamquant.pccp_verdict");
    assert!(def.input_kinds.is_empty(), "graph-input recipe");
    assert!(!def.description.is_empty());
}
