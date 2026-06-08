//! Integration tests for `blut::stages::register_model` input validation —
//! the fail-closed BadInput paths (empty / path-separator names) that reject
//! BEFORE any registry I/O. (BLUT L4 robustness lane.)

use blut::artifacts::GgufModel;
use blut::framework::artifact::ContentHash;
use blut::framework::stage::{Stage, StageContext};
use blut::framework::error::StageError;
use blut::stages::register_model::{Args, RegisterModel};
use std::path::PathBuf;

fn dummy_input() -> GgufModel {
    GgufModel {
        path: PathBuf::from("/tmp/blut-test-nonexistent.gguf"),
        quant: "Q4_K_M".into(),
        content_hash: ContentHash([0u8; 32]),
        registered_as: None,
    }
}

fn test_ctx() -> StageContext {
    // The validation under test returns before touching these dirs.
    StageContext::for_test(
        PathBuf::from("/tmp/blut-rmtest-job"),
        PathBuf::from("/tmp/blut-rmtest-job/stage"),
    )
}

async fn run_with_name(name: &str) -> Result<GgufModel, StageError> {
    RegisterModel
        .run(
            &test_ctx(),
            dummy_input(),
            &Args { name: name.into(), notes: String::new(), arch: "trained".into() },
        )
        .await
}

#[tokio::test]
async fn empty_name_is_bad_input() {
    assert!(matches!(run_with_name("").await, Err(StageError::BadInput(_))));
}

#[tokio::test]
async fn name_with_forward_slash_is_bad_input() {
    assert!(matches!(run_with_name("a/b").await, Err(StageError::BadInput(_))));
}

#[tokio::test]
async fn name_with_backslash_is_bad_input() {
    assert!(matches!(run_with_name("a\\b").await, Err(StageError::BadInput(_))));
}
