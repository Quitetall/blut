// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! WASM-safe tenant identity (ADR 0096).

use std::path::PathBuf;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The project name of the flat/default namespace.
pub const DEFAULT_PROJECT: &str = "default";

/// A tenant: a `project`, optionally with a `domain` (`project/domain`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Tenant {
    project: String,
    domain: Option<String>,
}

impl Default for Tenant {
    fn default() -> Self {
        Self {
            project: DEFAULT_PROJECT.to_string(),
            domain: None,
        }
    }
}

impl Tenant {
    /// Parse a `project` or `project/domain`. Empty means `default`.
    /// Segments are restricted to `[A-Za-z0-9_.-]`; empty segments, the
    /// traversal tokens `.`/`..`, and leading/trailing dots are refused.
    /// Dots remain valid inside names such as `model-v1.2`, but never at a
    /// filesystem-normalized edge where two tenant identities could alias.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Some(Self::default());
        }
        let mut parts = s.split('/');
        let project = parts.next()?;
        let domain = parts.next();
        if parts.next().is_some() {
            return None;
        }
        let ok = |segment: &str| {
            !segment.is_empty()
                && segment != "."
                && segment != ".."
                && !segment.starts_with('.')
                && !segment.ends_with('.')
                && segment
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
        };
        if !ok(project) || domain.is_some_and(|value| !ok(value)) {
            return None;
        }
        Some(Self {
            project: project.to_string(),
            domain: domain.map(str::to_string),
        })
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn domain(&self) -> Option<&str> {
        self.domain.as_deref()
    }

    /// The store path segment for this tenant.
    pub fn as_path(&self) -> PathBuf {
        let mut path = PathBuf::from(&self.project);
        if let Some(domain) = &self.domain {
            path.push(domain);
        }
        path
    }

    pub fn is_default(&self) -> bool {
        self.project == DEFAULT_PROJECT && self.domain.is_none()
    }

    /// A sealed clinical/PHI namespace. Case-insensitive so spelling cannot
    /// bypass custody policy.
    pub fn is_restricted(&self) -> bool {
        self.project.eq_ignore_ascii_case("clinical")
            || self.project.eq_ignore_ascii_case("restricted")
    }
}

impl std::fmt::Display for Tenant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.domain {
            Some(domain) => write!(f, "{}/{domain}", self.project),
            None => f.write_str(&self.project),
        }
    }
}

impl FromStr for Tenant {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or_else(|| "invalid tenant identity".to_string())
    }
}

impl Serialize for Tenant {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Tenant {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).ok_or_else(|| serde::de::Error::custom("invalid tenant identity"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_render_and_wire_round_trip() {
        let tenant = Tenant::parse("research/dev").unwrap();
        assert_eq!(tenant.to_string(), "research/dev");
        assert_eq!(serde_json::to_string(&tenant).unwrap(), r#""research/dev""#);
        assert_eq!(
            serde_json::from_str::<Tenant>(r#""research/dev""#).unwrap(),
            tenant
        );
        assert!(serde_json::from_str::<Tenant>(r#""../escape""#).is_err());
        assert_eq!(Tenant::parse("").unwrap(), Tenant::default());
        assert!(Tenant::parse("a/b/c").is_none());
        assert!(Tenant::parse("a//b").is_none());
        assert!(Tenant::parse("../etc").is_none());
        assert!(Tenant::parse(".clinical").is_none());
        assert!(Tenant::parse("clinical.").is_none());
        assert!(Tenant::parse("clinical/.prod").is_none());
        assert!(Tenant::parse("clinical/prod.").is_none());
        assert!(Tenant::parse("a b").is_none());
        assert!(Tenant::parse("model-v1.2/dev").is_some());
    }

    #[test]
    fn default_is_flat() {
        let tenant = Tenant::default();
        assert!(tenant.is_default());
        assert!(!tenant.is_restricted());
    }

    #[test]
    fn clinical_and_restricted_are_sealed() {
        assert!(Tenant::parse("clinical/prod").unwrap().is_restricted());
        assert!(Tenant::parse("restricted").unwrap().is_restricted());
        assert!(Tenant::parse("Clinical/prod").unwrap().is_restricted());
        assert!(Tenant::parse("RESTRICTED").unwrap().is_restricted());
        assert!(!Tenant::parse("research/prod").unwrap().is_restricted());
    }

    #[test]
    fn as_path_shapes() {
        assert_eq!(
            Tenant::parse("clinical/prod").unwrap().as_path(),
            PathBuf::from("clinical/prod")
        );
        assert_eq!(
            Tenant::parse("research").unwrap().as_path(),
            PathBuf::from("research")
        );
    }
}
