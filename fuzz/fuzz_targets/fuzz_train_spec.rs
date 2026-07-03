// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Fuzz target: `blut::spec::TrainSpec` JSON deserialization.
//!
//! `TrainSpec` is the validated, ready-to-execute training description.
//! Job specs land on disk (`status.jsonl`-adjacent job dirs) and are read
//! back by CLI/TUI commands (`blut jobs show`, resume flows) as untrusted
//! bytes -- a corrupted or hand-edited spec file must not panic the reader.
//! This target exercises `serde_json::from_slice::<TrainSpec>` directly
//! against arbitrary bytes.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<blut::spec::TrainSpec>(data);
});
