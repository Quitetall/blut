//! Artifact content identity: wire shape and serialization.
//!
//! Replaces `abir_artifact_identity.rs`. That file asserted `ArtifactContentId`
//! was a projection of `abir::ContentId`, which required the domain-agnostic
//! engine to link a BIOSIGNAL crate — the edge ADR 0143 says `blut` "MUST NOT"
//! have. The projection is gone; everything below is the part that was actually
//! load-bearing and is unchanged by the cut.

use blut::framework::{ArtifactContentId, ContentHash};

#[test]
fn identity_round_trips_through_its_raw_bytes() {
    let bytes = [0x5a; 32];
    let id = ArtifactContentId::from_bytes(bytes);
    assert_eq!(id.to_bytes(), bytes);
}

#[test]
fn identity_serializes_as_lower_case_64_hex() {
    // The historical serialization. A consumer that stored these as text and
    // compares them literally would break on a case or width change.
    let id = ArtifactContentId::from_bytes([0x5a; 32]);
    assert_eq!(id.to_hex(), "5a".repeat(32));
    assert_eq!(id.to_hex().len(), 64);
    assert_eq!(id.to_string(), id.to_hex());
}

#[test]
fn identity_survives_a_serde_round_trip() {
    let id = ArtifactContentId::from_bytes([0x5a; 32]);
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(
        serde_json::from_str::<ArtifactContentId>(&json).unwrap(),
        id
    );
}

#[test]
fn legacy_content_hash_and_identity_share_bincode_wire_shape() {
    // Load-bearing for readers of already-written artifacts: pre-ADR0169
    // callers stored identity bytes in a `ContentHash`, so the two types must
    // occupy the SAME bincode wire slot. If they diverge, an old manifest
    // deserializes into the wrong shape rather than failing loudly.
    let legacy = ContentHash::of_bytes(b"pre-ADR0169 object identity");
    let projected = ArtifactContentId::from_digest(legacy);

    let legacy_wire = bincode::serialize(&legacy).unwrap();
    let projected_wire = bincode::serialize(&projected).unwrap();
    assert_eq!(projected_wire, legacy_wire);
    assert_eq!(
        bincode::deserialize::<ArtifactContentId>(&legacy_wire).unwrap(),
        projected
    );
}
