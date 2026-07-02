// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Fuzz target: `Artifact::decode_erased` (default bincode-of-self impl).
//!
//! `ErasedArtifact { kind, schema, payload: Vec<u8> }` is the `(kind,
//! schema, bytes)` triple that crosses the `StageDyn` erased boundary
//! between stages, and rides the P2P bundle manifest across the network
//! (see `p2p/bundle.rs`). The `payload` field is bincode-deserialized by
//! `Artifact::decode_erased`'s default impl -- attacker/corruption-reachable
//! bytes decoded with a binary, non-self-describing format, which is exactly
//! the shape that tends to panic on malformed length prefixes.
//!
//! `DatasetJsonl` is a simple, already-pub `Artifact` impl that does NOT
//! override `decode_erased`, so fuzzing through it exercises the trait's
//! default decode path (the same path every non-tuple artifact uses).
#![no_main]

use libfuzzer_sys::fuzz_target;

use blut::artifacts::dataset::DatasetJsonl;
use blut::framework::artifact::Artifact;
use blut::framework::stage::ErasedArtifact;

fuzz_target!(|data: &[u8]| {
    let erased = ErasedArtifact {
        kind: DatasetJsonl::KIND.to_string(),
        schema: DatasetJsonl::SCHEMA,
        payload: data.to_vec(),
    };
    let _ = DatasetJsonl::decode_erased(erased);
});
