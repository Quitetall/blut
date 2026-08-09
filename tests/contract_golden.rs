//! Trainer-contract v1 golden test (CONTRACT.md §7): the engine's own
//! `StatusUpdate` parser must accept every control line in the frozen golden
//! stream, and the non-JSON channel lines (`BLUT_CONTRACT` / `BLUT_METRIC`)
//! must be recognizable by prefix so contract-aware readers can route them.
//!
//! This is the cross-language fixture both repos' CIs validate against — the
//! Python reference implementation (`tritium.torch.contract`) emits it, this
//! parser consumes it. If this test breaks, one side drifted.

use blut::protocol::StatusUpdate;

const GOLDEN: &str = include_str!("contract/status_stream_v1.golden.jsonl");

#[test]
fn golden_stream_parses_end_to_end() {
    let mut control = Vec::new();
    let mut metric_lines = 0usize;
    let mut announce_lines = 0usize;

    for line in GOLDEN.lines().filter(|l| !l.trim().is_empty()) {
        if let Some(version) = line.strip_prefix("BLUT_CONTRACT ") {
            assert_eq!(version.trim(), "1", "golden stream announces contract v1");
            announce_lines += 1;
            continue;
        }
        if let Some(payload) = line.strip_prefix("BLUT_METRIC ") {
            let v: serde_json::Value =
                serde_json::from_str(payload).expect("BLUT_METRIC payload is JSON");
            assert!(v.get("kind").is_some(), "metric payload carries kind");
            metric_lines += 1;
            continue;
        }
        let update: StatusUpdate = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("control line failed to parse: {line} ({e})"));
        control.push(update);
    }

    assert_eq!(announce_lines, 1, "exactly one announcement line");
    assert!(metric_lines >= 1, "the golden stream exercises the metric channel");

    // Exactly one terminal event, and it is the last control line.
    let terminal_positions: Vec<usize> = control
        .iter()
        .enumerate()
        .filter(|(_, u)| u.is_terminal())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(terminal_positions, vec![control.len() - 1]);

    // Every kind in the contract's table appears at least once (failed is
    // exercised by the negative fixture semantics in the Python lint, not the
    // happy-path stream).
    let mut saw = [false; 5]; // step, eval, saved, done, heartbeat
    for u in &control {
        match u {
            StatusUpdate::Step { step, total, .. } => {
                assert!(*step >= 1 && *step <= *total, "step is 1-indexed and bounded");
                saw[0] = true;
            }
            StatusUpdate::Eval { .. } => saw[1] = true,
            StatusUpdate::Saved { .. } => saw[2] = true,
            StatusUpdate::Done { .. } => saw[3] = true,
            StatusUpdate::Heartbeat { .. } => saw[4] = true,
            StatusUpdate::Failed { .. } => {}
        }
    }
    assert!(saw.iter().all(|s| *s), "golden stream covers step/eval/saved/done/heartbeat");
}

#[test]
fn unknown_kind_is_rejected() {
    let err = serde_json::from_str::<StatusUpdate>("{\"kind\":\"telemetry\",\"x\":1}");
    assert!(err.is_err(), "unknown kinds must be protocol errors (CONTRACT.md §1)");
}
