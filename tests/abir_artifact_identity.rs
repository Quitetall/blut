use abir::ContentId as AbirContentId;
use blut::framework::ContentId;

#[test]
fn blut_artifact_identity_is_an_abir_projection() {
    let semantic = AbirContentId::from_bytes([0x5a; 32]);
    let projected = ContentId::from_abir(semantic);

    assert_eq!(projected.as_abir(), semantic);
    assert_eq!(projected.to_hex(), semantic.to_string());
    assert_eq!(
        serde_json::from_str::<ContentId>(&serde_json::to_string(&projected).unwrap()).unwrap(),
        projected
    );
}
