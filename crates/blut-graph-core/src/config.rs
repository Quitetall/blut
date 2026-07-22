// SPDX-License-Identifier: AGPL-3.0-or-later

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use serde::{Deserialize, Serialize};

/// Exact, architecture-independent values accepted by semantic node instances.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConfigValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    Text(String),
    Bytes(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConfigType {
    Bool,
    I64 { minimum: i64, maximum: i64 },
    U64 { minimum: u64, maximum: u64 },
    Text { max_bytes: u32 },
    Choice { values: Vec<String> },
    Bytes { max_bytes: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigField {
    pub name: String,
    pub value_type: ConfigType,
    pub required: bool,
    pub default: Option<ConfigValue>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigSchema {
    pub fields: Vec<ConfigField>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    DuplicateField(String),
    InvalidField(String),
    UnknownField(String),
    MissingField(String),
    InvalidValue(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ConfigError {}

impl ConfigSchema {
    /// Normalize set-like schema content and reject ambiguous declarations.
    pub fn normalize(&mut self) -> Result<(), ConfigError> {
        self.fields
            .sort_by(|left, right| left.name.cmp(&right.name));
        if self
            .fields
            .windows(2)
            .any(|pair| pair[0].name == pair[1].name)
        {
            return Err(ConfigError::DuplicateField(
                self.fields
                    .windows(2)
                    .find(|pair| pair[0].name == pair[1].name)
                    .expect("duplicate was detected")[0]
                    .name
                    .clone(),
            ));
        }
        for field in &mut self.fields {
            if field.name.is_empty() || (field.required && field.default.is_some()) {
                return Err(ConfigError::InvalidField(field.name.clone()));
            }
            if let ConfigType::Choice { values } = &mut field.value_type {
                values.sort_unstable();
                values.dedup();
                if values.is_empty() || values.iter().any(String::is_empty) {
                    return Err(ConfigError::InvalidField(field.name.clone()));
                }
            }
            if !valid_type(&field.value_type)
                || field
                    .default
                    .as_ref()
                    .is_some_and(|value| !valid_value(&field.value_type, value))
            {
                return Err(ConfigError::InvalidField(field.name.clone()));
            }
        }
        Ok(())
    }

    /// Validate an instance and materialize defaults into one canonical map.
    pub fn canonicalize(
        &self,
        supplied: &BTreeMap<String, ConfigValue>,
    ) -> Result<BTreeMap<String, ConfigValue>, ConfigError> {
        let fields: BTreeMap<_, _> = self
            .fields
            .iter()
            .map(|field| (field.name.as_str(), field))
            .collect();
        if let Some(unknown) = supplied
            .keys()
            .find(|name| !fields.contains_key(name.as_str()))
        {
            return Err(ConfigError::UnknownField(unknown.clone()));
        }
        let mut normalized = BTreeMap::new();
        for field in &self.fields {
            match supplied.get(&field.name).or(field.default.as_ref()) {
                Some(value) if valid_value(&field.value_type, value) => {
                    normalized.insert(field.name.clone(), value.clone());
                }
                Some(_) => return Err(ConfigError::InvalidValue(field.name.clone())),
                None if field.required => {
                    return Err(ConfigError::MissingField(field.name.clone()));
                }
                None => {}
            }
        }
        Ok(normalized)
    }
}

fn valid_type(value_type: &ConfigType) -> bool {
    match value_type {
        ConfigType::I64 { minimum, maximum } => minimum <= maximum,
        ConfigType::U64 { minimum, maximum } => minimum <= maximum,
        ConfigType::Text { max_bytes } | ConfigType::Bytes { max_bytes } => *max_bytes > 0,
        ConfigType::Choice { values } => {
            !values.is_empty()
                && values.iter().all(|value| !value.is_empty())
                && values.iter().collect::<BTreeSet<_>>().len() == values.len()
        }
        ConfigType::Bool => true,
    }
}

fn valid_value(value_type: &ConfigType, value: &ConfigValue) -> bool {
    match (value_type, value) {
        (ConfigType::Bool, ConfigValue::Bool(_)) => true,
        (ConfigType::I64 { minimum, maximum }, ConfigValue::I64(value)) => {
            value >= minimum && value <= maximum
        }
        (ConfigType::U64 { minimum, maximum }, ConfigValue::U64(value)) => {
            value >= minimum && value <= maximum
        }
        (ConfigType::Text { max_bytes }, ConfigValue::Text(value)) => {
            value.len() <= *max_bytes as usize
        }
        (ConfigType::Choice { values }, ConfigValue::Text(value)) => values.contains(value),
        (ConfigType::Bytes { max_bytes }, ConfigValue::Bytes(value)) => {
            value.len() <= *max_bytes as usize
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn schema_canonicalizes_order_choices_and_defaults() {
        let mut schema = ConfigSchema {
            fields: vec![
                ConfigField {
                    name: "mode".into(),
                    value_type: ConfigType::Choice {
                        values: vec!["safe".into(), "fast".into(), "safe".into()],
                    },
                    required: false,
                    default: Some(ConfigValue::Text("safe".into())),
                },
                ConfigField {
                    name: "channels".into(),
                    value_type: ConfigType::U64 {
                        minimum: 1,
                        maximum: 256,
                    },
                    required: true,
                    default: None,
                },
            ],
        };
        schema.normalize().unwrap();
        assert_eq!(schema.fields[0].name, "channels");
        assert_eq!(
            schema.canonicalize(&BTreeMap::from([("channels".into(), ConfigValue::U64(64))])),
            Ok(BTreeMap::from([
                ("channels".into(), ConfigValue::U64(64)),
                ("mode".into(), ConfigValue::Text("safe".into())),
            ]))
        );
    }

    #[test]
    fn instance_rejects_unknown_missing_and_out_of_range_values() {
        let schema = ConfigSchema {
            fields: vec![ConfigField {
                name: "channels".into(),
                value_type: ConfigType::U64 {
                    minimum: 1,
                    maximum: 256,
                },
                required: true,
                default: None,
            }],
        };
        assert!(matches!(
            schema.canonicalize(&BTreeMap::new()),
            Err(ConfigError::MissingField(_))
        ));
        assert!(matches!(
            schema.canonicalize(&BTreeMap::from([(
                "channels".into(),
                ConfigValue::U64(257)
            )])),
            Err(ConfigError::InvalidValue(_))
        ));
        assert!(matches!(
            schema.canonicalize(&BTreeMap::from([(
                "surprise".into(),
                ConfigValue::Bool(true)
            )])),
            Err(ConfigError::UnknownField(_))
        ));
    }
}
