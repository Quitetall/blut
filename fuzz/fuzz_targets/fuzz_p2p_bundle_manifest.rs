// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Fuzz target: `blut::p2p::bundle::BundleManifest` bincode deserialization.
//!
//! HIGHEST PRIORITY target in this suite. `BundleManifest` rides the P2P
//! `TaskManifest.encrypted_input` / `TaskResult.encrypted_output` slots --
//! after AES-256-GCM decryption it is bincode-deserialized straight from
//! peer-supplied bytes. This is the only network-attacker-reachable parse
//! boundary in the whole system (every other target here is a local-file or
//! same-host IPC boundary). No `Arbitrary` derive needed -- bincode
//! deserializes directly from a byte slice.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = bincode::deserialize::<blut::p2p::bundle::BundleManifest>(data);
});
