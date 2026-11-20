// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Which code produced a cache entry: the engine build and every registered
//! cookbook build, folded into every stage cache key.
//!
//! A cache key used to fold only the engine's build commit (`unknown` when
//! the engine came from crates.io) and a stage's optional
//! [`code_fingerprint`](crate::framework::stage::Stage::code_fingerprint). A
//! change to a cookbook's stage logic that did not bump the stage's `SCHEMA`
//! therefore reused artifacts the old logic had produced.
//!
//! Now every key folds a [`CodeIdentity`]: the engine's [`BuildIdentity`]
//! (crate name, version, build commit, dirty digest) and the
//! [`BuildIdentity`] of every cookbook the binary registered. A cookbook
//! supplies its own with [`crate::cookbook_identity!`] from
//! [`Cookbook::identity`](crate::framework::cookbook::Cookbook::identity):
//!
//! ```ignore
//! impl Cookbook for MyCookbook {
//!     // …
//!     fn identity(&self) -> Option<blut::framework::code_identity::BuildIdentity> {
//!         Some(blut::cookbook_identity!())
//!     }
//! }
//! ```
//!
//! The macro reads the cookbook crate's own `CARGO_PKG_NAME` and
//! `CARGO_PKG_VERSION`, plus two variables the cookbook's `build.rs` may set
//! with `cargo:rustc-env`:
//!
//! * `BLUT_COOKBOOK_COMMIT` — the git commit the cookbook was built from
//!   (empty or unset when there is no git, as in a crates.io build);
//! * `BLUT_COOKBOOK_DIRTY` — a digest of the uncommitted changes, tracked and
//!   untracked, in the source the build covers (empty when clean).
//!
//! **Dirty trees.** A build from a tree with uncommitted changes carries the
//! digest of those changes. Its keys therefore differ from the clean build of
//! the same commit and from every other dirty state, and equal only a rebuild
//! of exactly the same changes, which is the same code. The engine's own
//! `build.rs` does the same for the engine (`BLUT_GIT_DIRTY`).
//!
//! **crates.io builds** have no git: the identity is the crate name and
//! version.
//!
//! **Scope.** The identity is per binary, not per stage: every stage key folds
//! every registered cookbook's identity, so changing any cookbook in a binary
//! re-runs every stage that binary runs. That over-invalidates; it never
//! reuses an entry the current code did not produce.
//!
//! The binary installs its identity once ([`install`]) before it computes a
//! key; [`crate::cli::run`] does this from the registry it is given. A process
//! that never installs one keys on the engine identity alone.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// One crate build: what the cache key needs to tell two builds apart.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BuildIdentity {
    /// Crate name (`CARGO_PKG_NAME`).
    pub name: String,
    /// Crate version (`CARGO_PKG_VERSION`).
    pub version: String,
    /// Git commit the crate was built from, when the build had git.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Digest of the uncommitted changes the build included, when the tree
    /// was dirty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dirty: Option<String>,
}

impl BuildIdentity {
    /// A build known only by name and version (a crates.io build).
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            commit: None,
            dirty: None,
        }
    }

    /// The form [`crate::cookbook_identity!`] expands to. Empty or whitespace
    /// strings, and the engine's `unknown` placeholder, read as absent.
    pub fn from_build_env(
        name: &str,
        version: &str,
        commit: Option<&str>,
        dirty: Option<&str>,
    ) -> Self {
        let present = |v: Option<&str>| {
            v.map(str::trim)
                .filter(|s| !s.is_empty() && *s != "unknown")
                .map(str::to_string)
        };
        Self {
            name: name.to_string(),
            version: version.to_string(),
            commit: present(commit),
            dirty: present(dirty),
        }
    }

    /// This engine build.
    pub fn engine() -> Self {
        Self::from_build_env(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            Some(env!("BLUT_GIT_HASH")),
            Some(env!("BLUT_GIT_DIRTY")),
        )
    }

    /// Whether the build included uncommitted changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty.is_some()
    }

    fn hash_into(&self, h: &mut sha2::Sha256) {
        use sha2::Digest;
        for field in [
            Some(self.name.as_str()),
            Some(self.version.as_str()),
            self.commit.as_deref(),
            self.dirty.as_deref(),
        ] {
            match field {
                Some(v) => {
                    h.update([1u8]);
                    h.update((v.len() as u64).to_le_bytes());
                    h.update(v.as_bytes());
                }
                None => h.update([0u8]),
            }
        }
    }
}

impl std::fmt::Display for BuildIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.name, self.version)?;
        match (&self.commit, &self.dirty) {
            (Some(c), Some(d)) => write!(f, " ({c}, dirty {})", short(d)),
            (Some(c), None) => write!(f, " ({c})"),
            (None, Some(d)) => write!(f, " (dirty {})", short(d)),
            (None, None) => Ok(()),
        }
    }
}

fn short(s: &str) -> &str {
    s.get(..12).unwrap_or(s)
}

/// The code a binary runs: the engine build plus every registered cookbook.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CodeIdentity {
    /// The engine build.
    pub engine: BuildIdentity,
    /// Registered cookbooks that supplied an identity, sorted, without
    /// duplicates.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cookbooks: Vec<BuildIdentity>,
    /// Names of registered cookbooks that supplied none. Their code changes
    /// cannot be told apart, so `cli::run` warns about each.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unidentified: Vec<String>,
}

impl CodeIdentity {
    /// The engine alone: what a process keys on until it installs more.
    pub fn engine_only() -> Self {
        Self {
            engine: BuildIdentity::engine(),
            cookbooks: Vec::new(),
            unidentified: Vec::new(),
        }
    }

    /// Start from an explicit engine identity (tests; tools that rebuild an
    /// identity recorded elsewhere).
    pub fn with_engine(engine: BuildIdentity) -> Self {
        Self {
            engine,
            cookbooks: Vec::new(),
            unidentified: Vec::new(),
        }
    }

    /// Add a cookbook build. Kept sorted and unique, so registration order
    /// does not change the digest.
    pub fn with_cookbook(mut self, cookbook: BuildIdentity) -> Self {
        if let Err(at) = self.cookbooks.binary_search(&cookbook) {
            self.cookbooks.insert(at, cookbook);
        }
        self
    }

    /// Record a registered cookbook that supplied no identity.
    pub fn with_unidentified(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        if let Err(at) = self.unidentified.binary_search(&name) {
            self.unidentified.insert(at, name);
        }
        self
    }

    /// Whether the engine or any cookbook was built from a dirty tree.
    pub fn is_dirty(&self) -> bool {
        self.engine.is_dirty() || self.cookbooks.iter().any(BuildIdentity::is_dirty)
    }

    /// The bytes folded into every stage cache key.
    pub fn digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"blut.code-identity.v1");
        self.engine.hash_into(&mut h);
        h.update((self.cookbooks.len() as u64).to_le_bytes());
        for c in &self.cookbooks {
            c.hash_into(&mut h);
        }
        h.update((self.unidentified.len() as u64).to_le_bytes());
        for n in &self.unidentified {
            h.update((n.len() as u64).to_le_bytes());
            h.update(n.as_bytes());
        }
        h.finalize().into()
    }
}

impl std::fmt::Display for CodeIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.engine)?;
        for c in &self.cookbooks {
            write!(f, "; {c}")?;
        }
        for n in &self.unidentified {
            write!(f, "; {n} (no identity)")?;
        }
        Ok(())
    }
}

static INSTALLED: OnceLock<(CodeIdentity, [u8; 32])> = OnceLock::new();

/// Fix this process's code identity. Call once, before any plan executes.
///
/// `Err` returns the identity already in force when it differs: either an
/// earlier `install`, or the engine-only default that the first cache key
/// computed in this process fixed. Installing the same identity again is
/// `Ok`.
pub fn install(identity: CodeIdentity) -> Result<(), Box<CodeIdentity>> {
    let digest = identity.digest();
    let (current, current_digest) = INSTALLED.get_or_init(|| (identity, digest));
    if *current_digest == digest {
        Ok(())
    } else {
        Err(Box::new(current.clone()))
    }
}

/// The identity in force: the installed one, or the engine alone.
pub fn current() -> &'static CodeIdentity {
    &installed().0
}

/// [`CodeIdentity::digest`] of [`current`], computed once.
pub(crate) fn current_digest() -> &'static [u8; 32] {
    &installed().1
}

fn installed() -> &'static (CodeIdentity, [u8; 32]) {
    INSTALLED.get_or_init(|| {
        let id = CodeIdentity::engine_only();
        let digest = id.digest();
        (id, digest)
    })
}

/// The [`BuildIdentity`] of the crate that invokes it: its `CARGO_PKG_NAME`
/// and `CARGO_PKG_VERSION`, and the `BLUT_COOKBOOK_COMMIT` /
/// `BLUT_COOKBOOK_DIRTY` its `build.rs` sets (see the
/// [module docs](crate::framework::code_identity)). Use it in
/// [`Cookbook::identity`](crate::framework::cookbook::Cookbook::identity).
#[macro_export]
macro_rules! cookbook_identity {
    () => {
        $crate::framework::code_identity::BuildIdentity::from_build_env(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            option_env!("BLUT_COOKBOOK_COMMIT"),
            option_env!("BLUT_COOKBOOK_DIRTY"),
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cb(commit: Option<&str>, dirty: Option<&str>, version: &str) -> BuildIdentity {
        BuildIdentity::from_build_env("my-cookbook", version, commit, dirty)
    }

    fn with(cookbook: BuildIdentity) -> [u8; 32] {
        CodeIdentity::with_engine(BuildIdentity::new("blut", "0.2.0"))
            .with_cookbook(cookbook)
            .digest()
    }

    #[test]
    fn the_same_build_gives_the_same_digest() {
        assert_eq!(
            with(cb(Some("abc"), None, "1.0.0")),
            with(cb(Some("abc"), None, "1.0.0"))
        );
    }

    #[test]
    fn commit_version_and_dirty_state_each_change_the_digest() {
        let base = with(cb(Some("abc"), None, "1.0.0"));
        assert_ne!(base, with(cb(Some("abd"), None, "1.0.0")), "commit");
        assert_ne!(base, with(cb(Some("abc"), None, "1.0.1")), "version");
        let dirty_a = with(cb(Some("abc"), Some("d1"), "1.0.0"));
        let dirty_b = with(cb(Some("abc"), Some("d2"), "1.0.0"));
        assert_ne!(base, dirty_a, "dirty vs clean");
        assert_ne!(dirty_a, dirty_b, "two dirty states");
        assert_eq!(dirty_a, with(cb(Some("abc"), Some("d1"), "1.0.0")));
        assert_ne!(with(cb(None, None, "1.0.0")), base, "no git vs a commit");
    }

    #[test]
    fn the_engine_version_and_commit_change_the_digest() {
        let c = cb(Some("abc"), None, "1.0.0");
        let at = |e: BuildIdentity| {
            CodeIdentity::with_engine(e)
                .with_cookbook(c.clone())
                .digest()
        };
        let base = at(BuildIdentity::from_build_env(
            "blut",
            "0.2.0",
            Some("e1"),
            None,
        ));
        assert_ne!(
            base,
            at(BuildIdentity::from_build_env(
                "blut",
                "0.2.1",
                Some("e1"),
                None
            ))
        );
        assert_ne!(
            base,
            at(BuildIdentity::from_build_env(
                "blut",
                "0.2.0",
                Some("e2"),
                None
            ))
        );
    }

    #[test]
    fn fields_cannot_run_together() {
        // Length-prefixed: "ab"+"c" must not collide with "a"+"bc".
        let a = BuildIdentity::from_build_env("ab", "c", None, None);
        let b = BuildIdentity::from_build_env("a", "bc", None, None);
        assert_ne!(with(a), with(b));
    }

    #[test]
    fn registration_order_does_not_matter_and_duplicates_collapse() {
        let x = cb(Some("1"), None, "1.0.0");
        let y = BuildIdentity::from_build_env("other", "2.0.0", None, None);
        let engine = BuildIdentity::new("blut", "0.2.0");
        let a = CodeIdentity::with_engine(engine.clone())
            .with_cookbook(x.clone())
            .with_cookbook(y.clone());
        let b = CodeIdentity::with_engine(engine)
            .with_cookbook(y)
            .with_cookbook(x.clone())
            .with_cookbook(x);
        assert_eq!(a, b);
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn an_unidentified_cookbook_is_part_of_the_identity() {
        let engine = BuildIdentity::new("blut", "0.2.0");
        let plain = CodeIdentity::with_engine(engine.clone());
        let unknown = CodeIdentity::with_engine(engine).with_unidentified("legacy");
        assert_ne!(plain.digest(), unknown.digest());
        assert!(unknown.to_string().contains("legacy (no identity)"));
    }

    #[test]
    fn empty_and_unknown_build_values_read_as_absent() {
        let id = BuildIdentity::from_build_env("c", "1", Some(""), Some("  "));
        assert_eq!((id.commit, id.dirty), (None, None));
        let id = BuildIdentity::from_build_env("c", "1", Some("unknown"), None);
        assert_eq!(id.commit, None);
    }

    #[test]
    fn the_macro_reads_the_invoking_crate() {
        let id = crate::cookbook_identity!();
        assert_eq!(id.name, env!("CARGO_PKG_NAME"));
        assert_eq!(id.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn the_engine_identity_carries_its_version() {
        let e = BuildIdentity::engine();
        assert_eq!(e.name, "blut");
        assert_eq!(e.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(e.dirty.is_some(), !env!("BLUT_GIT_DIRTY").is_empty());
    }

    #[test]
    fn display_names_every_part() {
        let id = CodeIdentity::with_engine(BuildIdentity::from_build_env(
            "blut",
            "0.2.0",
            Some("e1"),
            None,
        ))
        .with_cookbook(cb(Some("abc"), Some("0123456789abcdef0123"), "1.0.0"));
        assert_eq!(
            id.to_string(),
            "blut 0.2.0 (e1); my-cookbook 1.0.0 (abc, dirty 0123456789ab)"
        );
    }
}
