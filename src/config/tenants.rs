// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Tenant quota configuration (ADR 0096).
//!
//! Operators choose exactly one static policy:
//!
//! ```toml
//! tenants = ["research/dev", "clinical/prod"] # equal shares
//! ```
//!
//! or:
//!
//! ```toml
//! [fractions]
//! "research/dev" = 0.75
//! "clinical/prod" = 0.25
//! ```
//!
//! With no file, only the flat `default` tenant exists and owns 100% of the
//! usable box. An explicit file is fail-closed: malformed tenants, duplicate
//! tenants, invalid fractions, over-allocation, an empty policy, or mixing the
//! two forms is rejected. Unknown tenants never inherit an implicit share.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Result, TrainError};
use crate::tenant::Tenant;

/// Environment override for the tenant quota policy.
pub const TENANTS_CONFIG_ENV: &str = "BLUT_TENANTS_CONFIG";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TenantQuotaFile {
    tenants: Option<Vec<String>>,
    fractions: Option<BTreeMap<String, f64>>,
}

/// Validated, immutable tenant-to-box-fraction policy.
#[derive(Clone, Debug)]
pub struct TenantQuotaPolicy {
    fractions: HashMap<Tenant, f64>,
}

impl Default for TenantQuotaPolicy {
    fn default() -> Self {
        Self {
            fractions: HashMap::from([(Tenant::default(), 1.0)]),
        }
    }
}

impl TenantQuotaPolicy {
    /// Parse and validate a tenant quota TOML document.
    pub fn from_toml(body: &str) -> Result<Self> {
        let raw: TenantQuotaFile = toml::from_str(body)
            .map_err(|e| TrainError::other(format!("parse tenant quota config: {e}")))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: TenantQuotaFile) -> Result<Self> {
        match (raw.tenants, raw.fractions) {
            (Some(tenants), None) => Self::equal_shares(tenants),
            (None, Some(fractions)) => Self::explicit_fractions(fractions),
            (Some(_), Some(_)) => Err(TrainError::other(
                "tenant quota config must use either `tenants` (equal shares) or `fractions`, not both",
            )),
            (None, None) => Err(TrainError::other(
                "tenant quota config is empty; configure `tenants` or `fractions`",
            )),
        }
    }

    fn equal_shares(names: Vec<String>) -> Result<Self> {
        if names.is_empty() {
            return Err(TrainError::other(
                "tenant quota `tenants` list must not be empty",
            ));
        }
        let fraction = 1.0 / names.len() as f64;
        let mut fractions = HashMap::with_capacity(names.len());
        for name in names {
            let tenant = parse_tenant(&name)?;
            if fractions.insert(tenant, fraction).is_some() {
                return Err(TrainError::other(format!(
                    "duplicate tenant '{name}' in tenant quota config"
                )));
            }
        }
        Ok(Self { fractions })
    }

    fn explicit_fractions(raw: BTreeMap<String, f64>) -> Result<Self> {
        if raw.is_empty() {
            return Err(TrainError::other(
                "tenant quota `fractions` table must not be empty",
            ));
        }
        let mut fractions = HashMap::with_capacity(raw.len());
        let mut total = 0.0;
        for (name, fraction) in raw {
            if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
                return Err(TrainError::other(format!(
                    "tenant '{name}' fraction must be finite and in (0, 1], got {fraction}"
                )));
            }
            let tenant = parse_tenant(&name)?;
            total += fraction;
            fractions.insert(tenant, fraction);
        }
        if total > 1.0 + 1e-12 {
            return Err(TrainError::other(format!(
                "tenant quota fractions over-allocate the box: sum {total:.6} > 1.0"
            )));
        }
        Ok(Self { fractions })
    }

    /// Load the active policy. An explicit `$BLUT_TENANTS_CONFIG` path must
    /// exist; the conventional path may be absent, which selects the safe
    /// single-tenant default.
    pub fn load() -> Result<Self> {
        if let Ok(path) = std::env::var(TENANTS_CONFIG_ENV) {
            let path = PathBuf::from(path);
            if !path.is_file() {
                return Err(TrainError::other(format!(
                    "${TENANTS_CONFIG_ENV} points to missing tenant quota config {}",
                    path.display()
                )));
            }
            return Self::load_at(&path);
        }
        let Some(base) = dirs::config_dir() else {
            return Ok(Self::default());
        };
        let path = base.join("blut").join("tenants.toml");
        if !path.exists() {
            return Ok(Self::default());
        }
        Self::load_at(&path)
    }

    /// Load and validate a policy at an explicit path.
    pub fn load_at(path: &Path) -> Result<Self> {
        let body = std::fs::read_to_string(path).map_err(|source| TrainError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml(&body).map_err(|e| {
            TrainError::other(format!(
                "invalid tenant quota config {}: {e}",
                path.display()
            ))
        })
    }

    /// Resolve a configured tenant's fraction. Unknown tenants are refused;
    /// there is no implicit equal-share or fallback-to-default path.
    pub fn fraction_for(&self, tenant: &Tenant) -> Result<f64> {
        self.fractions.get(tenant).copied().ok_or_else(|| {
            TrainError::other(format!(
                "tenant '{tenant}' is not configured in the active quota policy (fail-closed)"
            ))
        })
    }
}

fn parse_tenant(name: &str) -> Result<Tenant> {
    Tenant::parse(name)
        .filter(|tenant| tenant.to_string() == name.trim())
        .ok_or_else(|| TrainError::other(format!("invalid tenant '{name}' in quota config")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_override_path_must_exist() {
        // Poison-tolerant: this mutex guards ENV MUTATION ordering, not data
        // invariants. One test panicking while holding it must not convert a
        // single real failure into a cascade of unrelated ones — which is
        // exactly what `.unwrap()` did here (17 tests failed on CI for one
        // underlying cause). `into_inner` keeps the ordering guarantee and
        // drops the poison flag.
        let _guard = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let old = std::env::var(TENANTS_CONFIG_ENV).ok();
        // SAFETY: TEST_ENV_LOCK serializes this process-global mutation.
        unsafe { std::env::set_var(TENANTS_CONFIG_ENV, "/definitely/missing/blut-tenants.toml") };
        assert!(TenantQuotaPolicy::load().is_err());
        // SAFETY: same lock; restore prior state.
        unsafe {
            if let Some(value) = old {
                std::env::set_var(TENANTS_CONFIG_ENV, value);
            } else {
                std::env::remove_var(TENANTS_CONFIG_ENV);
            }
        }
    }
}
