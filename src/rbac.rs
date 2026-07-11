// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Capability-scoped auth (ADR 0095) — the enforcement PRIMITIVE for BLUT's
//! network mutation path (the ADR-0083/0093 `blut-web` exec-bridge). Three roles
//! ordered by capability, a `Principal` bound to a role + tenant, a fail-closed
//! `authorize`, and an append-only `audit.jsonl` of every allow/deny.
//!
//! Read-only is the default: an anonymous caller gets `viewer` scope for
//! NON-restricted tenants only; any mutation, or any action touching a
//! `restricted` tenant, requires a token. The clinical hard-block (ADR 0061/0096)
//! composes ON TOP of roles: NO role — not even `admin` — lets a shared-tenant
//! token reach a `restricted` tenant; clinical access needs a token minted in
//! that tenant. Enforcement is fail-closed: an unresolved token or an
//! under-capable role is a DENY, never a fallthrough.
//!
//! Token secrets are stored sha256-hashed (never plaintext). API tokens are
//! high-entropy random values, for which a fast cryptographic hash is the
//! correct + standard choice (bcrypt/argon2 exist for LOW-entropy human
//! passwords) — the ADR's "hashed, never plaintext" intent, dependency-free.

use serde::{Deserialize, Serialize};

use crate::tenant::Tenant;

/// A built-in role, ordered by capability: `Viewer` < `Operator` < `Admin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Viewer,
    Operator,
    Admin,
}

impl Role {
    pub fn parse(s: &str) -> Option<Role> {
        match s {
            "viewer" => Some(Role::Viewer),
            "operator" => Some(Role::Operator),
            "admin" => Some(Role::Admin),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Viewer => "viewer",
            Role::Operator => "operator",
            Role::Admin => "admin",
        }
    }
}

/// A control action subject to authorisation. `required_role` is the minimum
/// role that may perform it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    // read (viewer)
    ReadStatus,
    ReadLineage,
    // mutate (operator)
    Run,
    Cancel,
    Retry,
    // administer (admin)
    PlanPromote,
    PlanRollback,
    SecretSet,
    TokenAdmin,
    TenantAdmin,
}

impl Action {
    pub fn required_role(self) -> Role {
        match self {
            Action::ReadStatus | Action::ReadLineage => Role::Viewer,
            Action::Run | Action::Cancel | Action::Retry => Role::Operator,
            Action::PlanPromote
            | Action::PlanRollback
            | Action::SecretSet
            | Action::TokenAdmin
            | Action::TenantAdmin => Role::Admin,
        }
    }

    /// A read action can be served anonymously (viewer scope); a mutation never.
    pub fn is_mutation(self) -> bool {
        self.required_role() > Role::Viewer
    }
}

/// An authenticated caller: a token id, its role, and the tenant the token was
/// minted in (ADR 0096). `None` at an authorize call site = anonymous.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Principal {
    pub token_id: String,
    pub role: Role,
    pub tenant: Tenant,
}

/// The outcome of an authorisation check — the audit record's payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthDecision {
    pub allowed: bool,
    /// `"anonymous"` when there was no token.
    pub actor: String,
    /// The acting role (`viewer` for anonymous).
    pub role: Role,
    /// The tenant the action targeted.
    pub target_tenant: String,
    pub action: Action,
    pub reason: String,
}

/// Fail-closed authorisation (ADR 0095). Deny unless ALL hold:
///   * the clinical boundary is honoured — a `restricted` target requires a
///     principal whose OWN tenant equals the target (no role, incl. admin,
///     crosses it; anonymous can never touch a restricted tenant);
///   * the caller's role dominates the action's required role (anonymous =
///     viewer, so any mutation is denied without a token).
pub fn authorize(
    principal: Option<&Principal>,
    action: Action,
    target_tenant: &Tenant,
) -> AuthDecision {
    let (actor, role) = match principal {
        Some(p) => (p.token_id.clone(), p.role),
        None => ("anonymous".to_string(), Role::Viewer),
    };
    let deny = |reason: &str| AuthDecision {
        allowed: false,
        actor: actor.clone(),
        role,
        target_tenant: target_tenant.to_string(),
        action,
        reason: reason.to_string(),
    };

    // Clinical hard-block FIRST — it dominates role (ADR 0061/0096).
    if target_tenant.is_restricted() {
        match principal {
            None => {
                return deny("anonymous caller may not touch a restricted tenant");
            }
            Some(p) if &p.tenant != target_tenant => {
                return deny(
                    "clinical hard-block: a token minted outside the restricted tenant is refused \
                     (no role, incl. admin, crosses the clinical boundary)",
                );
            }
            Some(_) => {} // same restricted tenant — fall through to the role check
        }
    }

    // Capability: role must dominate the action's required role (Ord = the
    // Viewer < Operator < Admin capability order — one source of truth).
    if role < action.required_role() {
        return deny(if principal.is_none() && action.is_mutation() {
            "a mutation requires a token (anonymous is viewer-only)"
        } else {
            "role does not dominate the action's required capability"
        });
    }

    AuthDecision {
        allowed: true,
        actor,
        role,
        target_tenant: target_tenant.to_string(),
        action,
        reason: "authorised".to_string(),
    }
}

/// Append an audit record to `path` as one JSON line (`audit.jsonl`). Called for
/// BOTH allows and denies, BEFORE dispatch. `now_unix` is passed in so the
/// record is deterministic for tests. The whole line (incl. newline) is written
/// in ONE `write_all` under `O_APPEND`, so concurrent appends of small records
/// don't interleave (POSIX atomic append ≤ PIPE_BUF). The file is `0600` on
/// Unix — the log carries token ids / tenants / outcomes.
pub fn append_audit(
    path: &std::path::Path,
    decision: &AuthDecision,
    now_unix: i64,
) -> std::io::Result<()> {
    use std::io::Write as _;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let row = serde_json::json!({
        "ts": now_unix,
        "allowed": decision.allowed,
        "actor": decision.actor,
        "role": decision.role,
        "tenant": decision.target_tenant,
        "action": decision.action,
        "reason": decision.reason,
    });
    let mut line = row.to_string();
    line.push('\n');

    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    let mut f = opts.open(path)?;
    f.write_all(line.as_bytes())
}

/// Authorise AND audit in one step (the enforcement seam a mutating control
/// boundary calls): the audit row is written for both allow and deny, so a
/// denied action is on the record too. **Fail-closed by construction**: if the
/// audit write itself fails (disk full, permissions), this returns a DENIED
/// decision rather than propagating the error or letting the action proceed
/// unlogged — a mutation is never permitted without a durable audit row.
pub fn enforce(
    principal: Option<&Principal>,
    action: Action,
    target_tenant: &Tenant,
    audit_path: &std::path::Path,
    now_unix: i64,
) -> AuthDecision {
    let decision = authorize(principal, action, target_tenant);
    if let Err(e) = append_audit(audit_path, &decision, now_unix) {
        tracing::error!(target: "rbac", "audit write failed, denying fail-closed: {e}");
        return AuthDecision {
            allowed: false,
            reason: format!("audit write failed — fail-closed: {e}"),
            ..decision
        };
    }
    decision
}

// ── token store (`~/.blut/web-tokens.toml`) ────────────────────────

/// sha256-hex of a token secret — what the store holds (never the plaintext).
pub fn hash_token(secret: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"blut.rbac.token.v1");
    h.update(secret.as_bytes());
    faster_hex::hex_string(&h.finalize())
}

#[derive(Clone, Debug, Deserialize)]
struct TokenEntry {
    id: String,
    /// sha256-hex of the token secret.
    hash: String,
    role: String,
    tenant: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct TokenStore {
    #[serde(default)]
    token: Vec<TokenEntry>,
}

impl TokenStore {
    /// Parse a `web-tokens.toml`, validating fail-LOUD: every entry's `role` and
    /// `tenant` must parse, and no two entries may share a token hash — a
    /// misconfigured file is a hard error naming the offending id, not a token
    /// that silently never resolves.
    pub fn parse(toml_str: &str) -> Result<Self, String> {
        let store: TokenStore =
            toml::from_str(toml_str).map_err(|e| format!("parse web-tokens.toml: {e}"))?;
        let mut seen = std::collections::HashSet::new();
        for t in &store.token {
            if Role::parse(&t.role).is_none() {
                return Err(format!("token '{}': unknown role '{}'", t.id, t.role));
            }
            if Tenant::parse(&t.tenant).is_none() {
                return Err(format!("token '{}': invalid tenant '{}'", t.id, t.tenant));
            }
            if !seen.insert(t.hash.clone()) {
                return Err(format!("token '{}': duplicate token hash", t.id));
            }
        }
        Ok(store)
    }

    /// Resolve a presented token secret to its `Principal` (hash + look up).
    /// `None` = no matching token (⇒ the caller is anonymous, fail-closed).
    pub fn resolve(&self, secret: &str) -> Option<Principal> {
        let want = hash_token(secret);
        self.token.iter().find(|t| t.hash == want).and_then(|t| {
            Some(Principal {
                token_id: t.id.clone(),
                role: Role::parse(&t.role)?,
                tenant: Tenant::parse(&t.tenant)?,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_order_and_parse() {
        assert!(Role::Admin > Role::Operator && Role::Operator > Role::Viewer);
        assert_eq!(Role::parse("operator"), Some(Role::Operator));
        assert_eq!(Role::parse("root"), None);
    }

    #[test]
    fn token_store_resolves_hashed_secret() {
        let secret = "s3cr3t-random-token";
        let store_toml = format!(
            "[[token]]\nid=\"t1\"\nhash=\"{}\"\nrole=\"admin\"\ntenant=\"clinical/prod\"\n",
            hash_token(secret)
        );
        let store = TokenStore::parse(&store_toml).unwrap();
        let p = store.resolve(secret).expect("resolves");
        assert_eq!(p.role, Role::Admin);
        assert!(p.tenant.is_restricted());
        assert!(store.resolve("wrong").is_none()); // fail-closed
        // The plaintext secret never appears in the store text.
        assert!(!store_toml.contains(secret));
    }

    #[test]
    fn token_store_validation_fails_loud() {
        // Unknown role → hard error naming the id.
        let bad_role = "[[token]]\nid=\"t1\"\nhash=\"ab\"\nrole=\"root\"\ntenant=\"shared\"\n";
        assert!(TokenStore::parse(bad_role).unwrap_err().contains("t1"));
        // Invalid tenant → hard error.
        let bad_tenant = "[[token]]\nid=\"t2\"\nhash=\"cd\"\nrole=\"admin\"\ntenant=\"../etc\"\n";
        assert!(TokenStore::parse(bad_tenant).unwrap_err().contains("t2"));
        // Duplicate token hash → hard error (no silent first-match wins).
        let dup = "[[token]]\nid=\"t3\"\nhash=\"ff\"\nrole=\"viewer\"\ntenant=\"shared\"\n\
                   [[token]]\nid=\"t4\"\nhash=\"ff\"\nrole=\"admin\"\ntenant=\"shared\"\n";
        assert!(TokenStore::parse(dup).unwrap_err().contains("duplicate"));
    }
}
