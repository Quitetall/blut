// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Tenants (ADR 0096) — the top-level isolation axis: a Flyte-style
//! `project[/domain]` (e.g. `research/dev`, `clinical/prod`).
//!
//! A tenant NAMESPACES the stateful stores (cache, plan registry, lineage) by a
//! path/key prefix; crucially it is **not** part of the ADR-0078 content-address
//! key, so a graph's fingerprint is byte-identical across tenants — the tenant
//! changes WHERE an artifact is stored, never WHAT its hash is. The `default`
//! tenant is the flat store (no prefix), so a single-tenant deployment behaves
//! exactly as the pre-tenancy engine.
//!
//! The clinical zone is a HARD boundary (ADR 0061): a `restricted` tenant is a
//! sealed namespace — its cache is never read by another tenant, its plan
//! fingerprints never promote onto a shared pointer (ADR 0085), and mesh /
//! sidecar paths refuse to move its artifacts off the owning box.

use std::path::PathBuf;

/// The project name of the flat/default namespace (byte-identical to the
/// pre-tenancy single store).
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
    /// Build from a `project` and optional `domain`.
    pub fn new(project: impl Into<String>, domain: Option<String>) -> Self {
        Self {
            project: project.into(),
            domain,
        }
    }

    /// Parse a `project` or `project/domain` string. An empty string is the
    /// default tenant; a trailing/leading slash or a `project//domain` is
    /// rejected (returns `None`) so a tenant can never contain an empty segment.
    /// Segments are restricted to `[A-Za-z0-9_.-]` so a tenant can't inject a
    /// `..` traversal or whitespace into a store path.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Some(Self::default());
        }
        let mut parts = s.split('/');
        let project = parts.next()?;
        let domain = parts.next();
        if parts.next().is_some() {
            return None; // more than 2 segments
        }
        let ok = |seg: &str| {
            !seg.is_empty()
                // reject `.`/`..` (path traversal) even though `.` is otherwise
                // an allowed character in a segment like `v1.2`.
                && seg != "."
                && seg != ".."
                && seg
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
        };
        if !ok(project) || domain.is_some_and(|d| !ok(d)) {
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

    /// The store path segment for this tenant: `project` or `project/domain`.
    pub fn as_path(&self) -> PathBuf {
        let mut p = PathBuf::from(&self.project);
        if let Some(d) = &self.domain {
            p.push(d);
        }
        p
    }

    /// The flat/default namespace? Then a namespace prefix is a no-op — the
    /// single-tenant store is byte-identical to the pre-tenancy engine.
    pub fn is_default(&self) -> bool {
        self.project == DEFAULT_PROJECT && self.domain.is_none()
    }

    /// A sealed clinical/PHI namespace (ADR 0061): the `clinical` project, or
    /// the `restricted` project kept for ADR-0085 compatibility. Enforced
    /// fail-closed at every cross-tenant boundary.
    pub fn is_restricted(&self) -> bool {
        self.project == "clinical" || self.project == "restricted"
    }
}

impl std::fmt::Display for Tenant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.domain {
            Some(d) => write!(f, "{}/{}", self.project, d),
            None => write!(f, "{}", self.project),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_render() {
        assert_eq!(
            Tenant::parse("research/dev").unwrap().to_string(),
            "research/dev"
        );
        assert_eq!(Tenant::parse("research").unwrap().to_string(), "research");
        assert_eq!(Tenant::parse("").unwrap(), Tenant::default());
        assert!(Tenant::parse("a/b/c").is_none()); // 3 segments
        assert!(Tenant::parse("a//b").is_none()); // empty segment
        assert!(Tenant::parse("../etc").is_none()); // traversal
        assert!(Tenant::parse("a b").is_none()); // whitespace
    }

    #[test]
    fn default_is_flat() {
        let d = Tenant::default();
        assert!(d.is_default());
        assert!(!d.is_restricted());
    }

    #[test]
    fn clinical_and_restricted_are_sealed() {
        assert!(Tenant::parse("clinical/prod").unwrap().is_restricted());
        assert!(Tenant::parse("restricted").unwrap().is_restricted());
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
