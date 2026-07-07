// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Fuzz target: `PlanSpec` JSON parse + `compile` (ADR 0078).
//!
//! `PlanSpec` is the `.json` recipe door and the wire form `blut-dsl` emits —
//! untrusted, possibly hand-edited or machine-generated text. Two robustness
//! properties this exercises:
//!   * `serde_json::from_str::<PlanSpec>` on arbitrary bytes never panics —
//!     notably a deeply NESTED `expansions[].template.expansions[]...` chain
//!     must not blow the stack during deserialization;
//!   * `PlanSpec::compile` against an (empty) registry never panics — every
//!     malformed graph/map is a typed `Err`, not an abort.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let s = String::from_utf8_lossy(data);
    if let Ok(spec) = serde_json::from_str::<blut::framework::plan_spec::PlanSpec>(&s) {
        // Empty registry: any stage name is unknown, so this returns early
        // with `UnknownStage` for most inputs — the point is that neither the
        // parse above nor this call ever panics on adversarial input.
        let _ = spec.compile(&blut::framework::Registry::new());
    }
});
