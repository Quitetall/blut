// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! ADR 0086 acceptance gate: a `SecretRef` in a stage's args leaks its plaintext
//! into NONE of the persisted provenance surfaces (args.json / cache-key input /
//! PlanSpec / lineage fingerprint), the cache key is value-INDEPENDENT (rotating
//! the secret's bytes never busts it), and a `restricted` secret refuses to
//! resolve across a mesh/sidecar boundary. Leak-prevention is STRUCTURAL — the
//! ref holds only a name — so the gate checks the serialisation surfaces (the
//! engine is lib-only; the value lives in the v1 env store, never the ref).

use blut::framework::artifact::ContentHash;
use blut::framework::cache::CacheHandle;
use blut::framework::plan_spec::{PlanSpec, SpecNode};
use blut::secrets::{EnvResolver, ResolveCtx, SecretError, SecretRef, SecretResolver};

const SENTINEL: &str = "PLAINTEXT-SENTINEL-9f3a2b";

#[derive(serde::Serialize)]
struct DbArgs {
    host: String,
    password: SecretRef,
}

fn args_with_secret(refname: &str) -> serde_json::Value {
    serde_json::to_value(DbArgs {
        host: "db.local".into(),
        password: SecretRef::new(refname),
    })
    .unwrap()
}

#[test]
fn plaintext_absent_from_all_four_surfaces() {
    // The value lives in the env (the v1 store), NEVER in the ref.
    unsafe { std::env::set_var("SECRET_LEAK_A", SENTINEL) };
    let args = args_with_secret("SECRET_LEAK_A");

    // (1) args.json / status.jsonl surface: the ref name is present, the value is
    //     structurally absent (the ref carries no value).
    let args_json = serde_json::to_string(&args).unwrap();
    assert!(
        args_json.contains("SECRET_LEAK_A"),
        "the ref name is provenance-visible"
    );
    assert!(
        !args_json.contains(SENTINEL),
        "args.json must not carry the value"
    );

    // (2) cache-key input (the cache manifest key) surface.
    let canon = CacheHandle::canonical_json_bytes(&args);
    assert!(!String::from_utf8_lossy(&canon).contains(SENTINEL));

    // (3) PlanSpec surface + (4) lineage provenance (a fingerprint OVER this spec).
    let spec = PlanSpec {
        name: "s".into(),
        nodes: vec![SpecNode {
            stage: "db".into(),
            args: args.clone(),
            retry: None,
            timeout: None,
            priority: None,
        }],
        edges: vec![],
        expansions: Vec::new(),
        version: 1,
    };
    assert!(!String::from_utf8_lossy(&spec.canonical_bytes()).contains(SENTINEL));
    let fp = spec.provenance_fingerprint("", &serde_json::Value::Null);
    assert!(!fp.to_hex().contains(SENTINEL)); // a hash reveals nothing anyway

    unsafe { std::env::remove_var("SECRET_LEAK_A") };
}

#[test]
fn cache_key_is_value_independent() {
    // The ref serialises to its NAME, so the cache-key input is identical no
    // matter what the underlying secret value is — rotating the secret's bytes
    // does not bust the cache, and the value is absent from every manifest.
    unsafe { std::env::set_var("SECRET_LEAK_B", "value-A") };
    let canon_a = CacheHandle::canonical_json_bytes(&args_with_secret("SECRET_LEAK_B"));
    unsafe { std::env::set_var("SECRET_LEAK_B", "value-B-totally-different") };
    let canon_b = CacheHandle::canonical_json_bytes(&args_with_secret("SECRET_LEAK_B"));
    assert_eq!(canon_a, canon_b, "the cache-key input is value-independent");

    // And the derived key is stable across the rotation.
    let k = |c: &[u8]| CacheHandle::key_for_canon_bytes("db", 1, ContentHash([0; 32]), c, b"sha");
    assert_eq!(k(&canon_a), k(&canon_b));
    unsafe { std::env::remove_var("SECRET_LEAK_B") };
}

#[test]
fn restricted_secret_refused_across_boundary() {
    let r = SecretRef::restricted("PHI_KEY");
    // A remote (mesh/sidecar) resolution is refused fail-closed — never a value.
    assert!(matches!(
        EnvResolver.resolve(&r, ResolveCtx::remote()),
        Err(SecretError::RestrictedCrossBoundary(_))
    ));
    // A non-restricted secret resolves remotely (present).
    unsafe { std::env::set_var("PLAIN", "v") };
    assert!(
        EnvResolver
            .resolve(&SecretRef::new("PLAIN"), ResolveCtx::remote())
            .is_ok()
    );
    unsafe { std::env::remove_var("PLAIN") };
}
