// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! WASM-safe secret reference wire type (ADR 0086).

use serde::{Deserialize, Serialize};

/// A reference to a credential by name. It intentionally carries no value, so
/// plans, cache keys, configs, and sidecar rule files can persist it safely.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SecretRef {
    pub name: String,
    /// A restricted credential may resolve only inside the owning engine
    /// process, never in a network/sidecar transport.
    #[serde(default)]
    pub restricted: bool,
}

impl SecretRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            restricted: false,
        }
    }

    pub fn restricted(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            restricted: true,
        }
    }

    /// Validate the portable environment/vault lookup key. Keeping this on the
    /// wire type lets engine and sidecars reject malformed references before a
    /// daemon begins polling or a sink attempts delivery.
    pub fn validate(&self) -> Result<(), &'static str> {
        let mut chars = self.name.chars();
        let Some(first) = chars.next() else {
            return Err("secret reference name is empty");
        };
        if !(first.is_ascii_alphabetic() || first == '_')
            || !chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return Err("secret reference name must be a portable environment key");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_names_are_portable_lookup_keys() {
        assert!(SecretRef::new("BLUT_TOKEN_1").validate().is_ok());
        assert!(SecretRef::new("").validate().is_err());
        assert!(SecretRef::new("1TOKEN").validate().is_err());
        assert!(SecretRef::new("TOKEN=value").validate().is_err());
    }
}
