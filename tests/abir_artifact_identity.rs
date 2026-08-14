use abir::ContentId as AbirContentId;
use blut::framework::{ArtifactContentId, ContentHash};

#[test]
fn blut_artifact_identity_is_an_abir_projection() {
    let semantic = AbirContentId::from_bytes([0x5a; 32]);
    let projected = ArtifactContentId::from_abir(semantic);

    assert_eq!(projected.as_abir(), semantic);
    assert_eq!(projected.to_hex(), semantic.to_string());
    assert_eq!(
        serde_json::from_str::<ArtifactContentId>(&serde_json::to_string(&projected).unwrap())
            .unwrap(),
        projected
    );
}

#[test]
fn legacy_content_hash_and_abir_projection_share_bincode_wire_shape() {
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
