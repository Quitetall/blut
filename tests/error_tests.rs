// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Integration tests for `blut::error::{TrainError, Result}` — previously 0%
//! covered. (BLUT L4 robustness lane.) The stages::util write_report coverage
//! lives inline in src/stages/util.rs since write_report is pub(crate).

use blut::error::TrainError;

#[test]
fn io_error_from_conversion_has_empty_path() {
    let e: TrainError = std::io::Error::new(std::io::ErrorKind::NotFound, "x").into();
    assert!(
        matches!(&e, TrainError::Io { path, .. } if path.as_os_str().is_empty()),
        "From<io::Error> should map to Io with an empty path, got {e:?}"
    );
}

#[test]
fn serde_json_error_maps_to_other_with_json_prefix() {
    let je = serde_json::from_str::<i32>("not json").unwrap_err();
    let e: TrainError = je.into();
    assert!(matches!(e, TrainError::Other(_)), "got {e:?}");
    assert!(e.to_string().starts_with("json:"), "got {e}");
}

#[test]
fn constructors_and_display() {
    assert_eq!(TrainError::other("m").to_string(), "m");
    assert_eq!(
        TrainError::invalid_spec("s").to_string(),
        "invalid TrainSpec: s"
    );
}
