// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! WASM-safe partition wire types (ADR 0101).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// One named dimension in a concrete partition key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionValue {
    pub dimension: String,
    pub value: String,
}

impl PartitionValue {
    pub fn new(dimension: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            dimension: dimension.into(),
            value: value.into(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.dimension.is_empty()
            || !self
                .dimension
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        {
            return Err("partition dimension must be non-empty [A-Za-z0-9_-]".into());
        }
        if self.value.is_empty() || self.value.chars().any(char::is_control) {
            return Err(
                "partition value must be non-empty and contain no control characters".into(),
            );
        }
        Ok(())
    }
}

/// Structured identity for one concrete cell. Dimension order is declared
/// order and therefore part of the stable cache identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionKey {
    values: Vec<PartitionValue>,
}

impl PartitionKey {
    pub fn new(values: Vec<PartitionValue>) -> Result<Self, String> {
        if values.is_empty() {
            return Err("partition key needs at least one dimension".into());
        }
        let mut seen = BTreeSet::new();
        for value in &values {
            value.validate()?;
            if !seen.insert(value.dimension.as_str()) {
                return Err(format!(
                    "partition key repeats dimension '{}'",
                    value.dimension
                ));
            }
        }
        Ok(Self { values })
    }

    pub fn values(&self) -> &[PartitionValue] {
        &self.values
    }
}

impl<'de> Deserialize<'de> for PartitionKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            values: Vec<PartitionValue>,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.values).map_err(serde::de::Error::custom)
    }
}

/// Declarative partition key-space. Additive serde fields keep this suitable
/// for engine/sidecar exchange through `blut-types`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PartitionSpec {
    Time {
        dimension: String,
        granularity: TimeGranularity,
        tz: String,
    },
    Categorical {
        dimension: String,
        values: Vec<String>,
    },
    Multi {
        dimensions: Vec<PartitionSpec>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeGranularity {
    Hour,
    Day,
    Week,
    Month,
}
