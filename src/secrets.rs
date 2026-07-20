// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Secret references (ADR 0086) — a `SecretRef` a stage accepts in place of a
//! plaintext credential. The leak-prevention is STRUCTURAL: a `SecretRef` holds
//! only a NAME, never the value, so it serialises as its name into every
//! persisted surface (PlanSpec, cache key, `args.json`, `status.jsonl`, lineage)
//! — the plaintext is never there to leak. The value is resolved lazily, held
//! only in the executing process (a zeroized-on-drop [`Secret`]), and passed to
//! the stage in-process.
//!
//! Three boundaries (ADR 0086):
//!   1. **fingerprint** — the cache key hashes the SecretRef's serialised NAME,
//!      so rotating a secret's bytes does NOT bust the cache and the value is
//!      absent from every cache manifest;
//!   2. **serialisation** — every persisted surface stores the name only; a
//!      [`redact`] pass replaces any resolved value substring before a free-text
//!      log line is written (belt-and-suspenders over the structural guarantee);
//!   3. **resolution** — the plaintext is fetched lazily; a `restricted` secret
//!      (ADR 0061/0096) is HARD-LOCAL: it refuses to resolve across a
//!      dispatch/mesh/sidecar boundary, only inside a local engine process.
//!
//! v1 store = the named ENV VAR (the ADR's stated fallback); the age-encrypted
//! `~/.blut/secrets.age` vault + `blut secret {set,ls,rm}` are the deliverable
//! upgrade behind a `secrets` crypto feature.

pub use blut_types::secrets::SecretRef;

/// A resolved plaintext secret — held ONLY in-process (as raw bytes so it can be
/// zeroised without `unsafe`), zeroised on drop, and redacting in `Debug` so it
/// can't accidentally hit a log line.
pub struct Secret(Vec<u8>);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Secret(value.into().into_bytes())
    }
    /// Borrow the plaintext (to pass to the credential consumer). Callers MUST
    /// NOT log or serialise it. The bytes are always valid UTF-8 (from a
    /// `String`), so this never observably fails.
    pub fn expose(&self) -> &str {
        std::str::from_utf8(&self.0).unwrap_or("")
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        // Zeroise the heap bytes before the Vec frees. A plain `= 0` loop to
        // about-to-be-freed memory is a DEAD STORE the optimiser may elide; no
        // `zeroize` dep + `unsafe` is denied here, so `black_box` is the safe,
        // dependency-free barrier — it forces the writes to be observed so they
        // survive optimisation.
        for b in self.0.iter_mut() {
            *b = 0;
        }
        std::hint::black_box(&self.0);
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Secret(«redacted»)")
    }
}

/// The context a resolution happens in. `remote` = the resolution is being asked
/// for on behalf of a dispatch/mesh/sidecar boundary (not a local engine
/// process) — a `restricted` secret refuses this.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResolveCtx {
    pub remote: bool,
}

impl ResolveCtx {
    pub fn local() -> Self {
        Self { remote: false }
    }
    pub fn remote() -> Self {
        Self { remote: true }
    }
}

/// A resolution failure.
#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    #[error("secret '{0}' not found (set the env var, or `blut secret set {0}`)")]
    NotFound(String),
    #[error(
        "secret '{0}' is restricted (ADR 0061): it never resolves across a \
         dispatch/mesh/sidecar boundary — only inside a local engine process on the owning box"
    )]
    RestrictedCrossBoundary(String),
}

/// Resolves a `SecretRef` to its plaintext. The clinical boundary is enforced
/// HERE, once, for every backend.
pub trait SecretResolver {
    /// Look up the raw value for `name` (store-specific). `None` = absent.
    fn lookup(&self, name: &str) -> Option<String>;

    /// Resolve a ref, enforcing the restricted hard-block first (fail-closed).
    fn resolve(&self, r: &SecretRef, ctx: ResolveCtx) -> Result<Secret, SecretError> {
        if r.restricted && ctx.remote {
            return Err(SecretError::RestrictedCrossBoundary(r.name.clone()));
        }
        self.lookup(&r.name)
            .map(Secret::new)
            .ok_or_else(|| SecretError::NotFound(r.name.clone()))
    }
}

/// The v1 store (ADR 0086 fallback): resolve a secret from the named env var.
/// The age-encrypted vault is the feature-gated upgrade over this.
pub struct EnvResolver;

impl SecretResolver for EnvResolver {
    fn lookup(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
}

/// Redact any occurrence of a resolved secret `value` in `text`, replacing it
/// with `«redacted:{name}»`. Fail-closed belt-and-suspenders over the structural
/// guarantee — if a stage's free-text output ever echoes a resolved value, the
/// value is scrubbed before the line is written/streamed. Empty values are
/// ignored (they would match everything).
pub fn redact(text: &str, secrets: &[(&str, &str)]) -> String {
    // Longest value first: if one secret's value is a substring of another's,
    // scrubbing the shorter first could break the longer's match.
    let mut ordered: Vec<&(&str, &str)> = secrets.iter().filter(|(_, v)| !v.is_empty()).collect();
    ordered.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
    let mut out = text.to_string();
    for (name, value) in ordered {
        out = out.replace(value, &format!("«redacted:{name}»"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_ref_serialises_name_only_never_a_value() {
        let r = SecretRef::new("DB_PASSWORD");
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("DB_PASSWORD"));
        // There is no value field at all — nothing to leak.
        assert!(!json.contains("value") && !json.contains("plaintext"));
    }

    #[test]
    fn restricted_refuses_remote_resolves_locally() {
        let r = SecretRef::restricted("PHI_KEY");
        // A remote/mesh/sidecar resolution is refused fail-closed.
        assert!(matches!(
            EnvResolver.resolve(&r, ResolveCtx::remote()),
            Err(SecretError::RestrictedCrossBoundary(_))
        ));
        // Locally it resolves (when present).
        // SAFETY (test): single-threaded per-test env access.
        unsafe { std::env::set_var("PHI_KEY", "local-value") };
        assert_eq!(
            EnvResolver
                .resolve(&r, ResolveCtx::local())
                .unwrap()
                .expose(),
            "local-value"
        );
        unsafe { std::env::remove_var("PHI_KEY") };
    }

    #[test]
    fn secret_debug_redacts() {
        let s = Secret::new("hunter2");
        assert_eq!(format!("{s:?}"), "Secret(«redacted»)");
        assert!(!format!("{s:?}").contains("hunter2"));
    }

    #[test]
    fn redact_scrubs_leaked_values() {
        let out = redact("connecting with hunter2 now", &[("DB", "hunter2")]);
        assert!(!out.contains("hunter2"));
        assert!(out.contains("«redacted:DB»"));
        // An empty value never matches everything.
        assert_eq!(redact("abc", &[("X", "")]), "abc");
    }
}
