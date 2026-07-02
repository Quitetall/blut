// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Fuzz target: `blut::framework::error_domain::FailureSummary` JSON
//! deserialization.
//!
//! ADR 0072 item A1 (already landed on this branch) made `StageFailure`/
//! `FailureSummary` derive `Serialize`/`Deserialize` and persist to
//! `status.jsonl`, which item A4 (`blut errors show`) reads back. That read
//! path is a NEW local-file untrusted-parse boundary this workflow is
//! creating: a hand-edited or corrupted `status.jsonl` line must not panic
//! the CLI. This target fuzzes that decode directly.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<blut::framework::error_domain::FailureSummary>(data);
});
