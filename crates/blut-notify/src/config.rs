// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Declarative notification rules and sink configuration (ADR 0094).

use blut_types::secrets::SecretRef;
use serde::Deserialize;

use crate::SinkBoundary;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotifyConfig {
    #[serde(default)]
    pub sink: Vec<SinkSpec>,
    #[serde(default)]
    pub rule: Vec<NotifyRule>,
}

impl NotifyConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        let config: Self = toml::from_str(text).map_err(|error| error.to_string())?;
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("parse {}: {error}", path.display()),
            )
        })
    }

    fn validate(&self) -> Result<(), String> {
        let mut names = std::collections::HashSet::new();
        for sink in &self.sink {
            if !safe_name(sink.name()) {
                return Err(format!("invalid sink name {:?}", sink.name()));
            }
            if !names.insert(sink.name().to_string()) {
                return Err(format!("duplicate sink name {:?}", sink.name()));
            }
            match sink {
                SinkSpec::Slack { webhook, .. } | SinkSpec::Discord { webhook, .. } => {
                    validate_secret(sink.name(), webhook)?;
                }
                SinkSpec::Ntfy { token, .. } => {
                    if let Some(token) = token {
                        validate_secret(sink.name(), token)?;
                    }
                }
                SinkSpec::Smtp {
                    username, password, ..
                } => {
                    if let Some(username) = username {
                        validate_secret(sink.name(), username)?;
                    }
                    if let Some(password) = password {
                        validate_secret(sink.name(), password)?;
                    }
                }
                SinkSpec::Exec { .. } => {}
            }
        }
        let mut rules = std::collections::HashSet::new();
        for rule in &self.rule {
            if !safe_name(&rule.name) || !rules.insert(rule.name.clone()) {
                return Err(format!("invalid or duplicate rule name {:?}", rule.name));
            }
            if rule.sinks.is_empty() {
                return Err(format!("rule {:?} has no sinks", rule.name));
            }
            for sink in &rule.sinks {
                if !names.contains(sink) {
                    return Err(format!(
                        "rule {:?} references unknown sink {:?}",
                        rule.name, sink
                    ));
                }
            }
            if rule.field.trim().is_empty() || rule.equals.trim().is_empty() {
                return Err(format!("rule {:?} needs non-empty field/equals", rule.name));
            }
        }
        Ok(())
    }
}

fn validate_secret(sink: &str, secret: &SecretRef) -> Result<(), String> {
    secret
        .validate()
        .map_err(|error| format!("invalid credential for sink {sink:?}: {error}"))
}

fn safe_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum SinkSpec {
    Slack {
        name: String,
        webhook: SecretRef,
    },
    Discord {
        name: String,
        webhook: SecretRef,
    },
    Ntfy {
        name: String,
        server: String,
        topic: String,
        #[serde(default)]
        token: Option<SecretRef>,
    },
    Smtp {
        name: String,
        server: String,
        #[serde(default)]
        port: Option<u16>,
        from: String,
        to: Vec<String>,
        #[serde(default)]
        username: Option<SecretRef>,
        #[serde(default)]
        password: Option<SecretRef>,
    },
    Exec {
        name: String,
        program: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default = "local_boundary")]
        boundary: SinkBoundary,
    },
}

impl SinkSpec {
    pub fn name(&self) -> &str {
        match self {
            Self::Slack { name, .. }
            | Self::Discord { name, .. }
            | Self::Ntfy { name, .. }
            | Self::Smtp { name, .. }
            | Self::Exec { name, .. } => name,
        }
    }
}

const fn local_boundary() -> SinkBoundary {
    SinkBoundary::Local
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotifySource {
    #[default]
    Status,
    Sla,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NotifyRule {
    pub name: String,
    #[serde(default)]
    pub source: NotifySource,
    /// Dotted JSON field path, e.g. `kind`, `state`, or `event.kind`.
    pub field: String,
    /// String/number/bool wire representation to match exactly.
    pub equals: String,
    pub sinks: Vec<String>,
}

impl NotifyRule {
    pub fn matches(&self, source: NotifySource, value: &serde_json::Value) -> bool {
        if self.source != source {
            return false;
        }
        let mut current = value;
        for component in self.field.split('.') {
            let Some(next) = current.get(component) else {
                return false;
            };
            current = next;
        }
        match current {
            serde_json::Value::String(value) => value == &self.equals,
            serde_json::Value::Number(value) => value.to_string() == self.equals,
            serde_json::Value::Bool(value) => value.to_string() == self.equals,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_sink_kinds_and_rejects_dangling_rule() {
        let config = NotifyConfig::parse(
            r#"
[[sink]]
name = "slack"
kind = "slack"
webhook = { name = "SLACK_URL" }

[[sink]]
name = "discord"
kind = "discord"
webhook = { name = "DISCORD_URL" }

[[sink]]
name = "phone"
kind = "ntfy"
server = "https://ntfy.sh"
topic = "ops"
token = { name = "NTFY_TOKEN" }

[[sink]]
name = "mail"
kind = "smtp"
server = "smtp.example.test"
from = "blut@example.test"
to = ["ops@example.test"]
username = { name = "SMTP_USER" }
password = { name = "SMTP_PASSWORD" }

[[sink]]
name = "local"
kind = "exec"
program = "/usr/bin/logger"

[[rule]]
name = "failed"
field = "kind"
equals = "failed"
sinks = ["slack", "discord", "phone", "mail", "local"]
"#,
        )
        .unwrap();
        assert_eq!(config.sink.len(), 5);
        assert_eq!(config.rule.len(), 1);

        let dangling = r#"
[[rule]]
name = "bad"
field = "kind"
equals = "failed"
sinks = ["missing"]
"#;
        assert!(NotifyConfig::parse(dangling).is_err());
    }

    #[test]
    fn dotted_match_is_typed_and_source_scoped() {
        let rule = NotifyRule {
            name: "done".into(),
            source: NotifySource::Status,
            field: "event.state".into(),
            equals: "done".into(),
            sinks: vec!["local".into()],
        };
        let value = serde_json::json!({"event": {"state": "done"}});
        assert!(rule.matches(NotifySource::Status, &value));
        assert!(!rule.matches(NotifySource::Sla, &value));
    }
}
