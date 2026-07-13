// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam

use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run(envelope: serde_json::Value, boundary: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_blut-notify"))
        .args(["--boundary", boundary])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&envelope).unwrap().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn restricted_cli_refuses_before_payload_reaches_output() {
    let secret = "patient-name-must-not-leak";
    let output = run(
        serde_json::json!({
            "tenant": "clinical/prod",
            "data_class": "Public",
            "summary": secret,
        }),
        "off-box",
    );
    assert!(!output.status.success());
    let visible = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(visible.contains("node-local"));
    assert!(!visible.contains(secret));
}

#[test]
fn nonrestricted_cli_delivers_serialized_envelope() {
    let output = run(
        serde_json::json!({
            "tenant": "research/dev",
            "data_class": "Internal",
            "summary": "training-complete",
        }),
        "off-box",
    );
    assert!(output.status.success());
    let body: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(body["tenant"], "research/dev");
    assert_eq!(body["data_class"], "Internal");
    assert_eq!(body["summary"], "training-complete");
}

#[test]
fn malformed_envelopes_refuse_without_echoing_wire_values() {
    let cases = [
        (
            "patient-name-in-invalid-tenant",
            serde_json::json!({
                "tenant": "../patient-name-in-invalid-tenant",
                "data_class": "Public",
                "summary": "unused",
            }),
        ),
        (
            "patient-name-in-invalid-class",
            serde_json::json!({
                "tenant": "research/dev",
                "data_class": "patient-name-in-invalid-class",
                "summary": "unused",
            }),
        ),
        (
            "patient-name-in-unknown-field",
            serde_json::json!({
                "tenant": "research/dev",
                "data_class": "Public",
                "summary": "unused",
                "patient-name-in-unknown-field": true,
            }),
        ),
    ];
    for (secret, envelope) in cases {
        let output = run(envelope, "off-box");
        assert!(!output.status.success());
        let visible = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(visible.contains("invalid notification envelope"));
        assert!(!visible.contains(secret));
    }
}
