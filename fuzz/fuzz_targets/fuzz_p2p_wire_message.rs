// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Fuzz target: `blut::p2p::transport::WireMessage` JSON deserialization.
//!
//! `WireMessage` is the length-prefixed JSON payload exchanged between
//! coordinator and peer over QUIC streams (`p2p/transport.rs`'s wire
//! protocol: `[4-byte LE length][JSON payload]`). The length-prefix framing
//! itself lives in the transport read loop; this target fuzzes the JSON
//! body decode in isolation, which is reachable by any peer that completes
//! (or merely attempts) the QUIC handshake.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<blut::p2p::transport::WireMessage>(data);
});
