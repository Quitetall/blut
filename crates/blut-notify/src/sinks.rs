// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Concrete out-of-process notification sinks (ADR 0094).

use blut_types::secrets::SecretRef;

use crate::config::SinkSpec;
use crate::{ExecSink, NotificationEnvelope, NotifySink, SinkBoundary};

pub fn build_sink(spec: &SinkSpec) -> Result<Box<dyn NotifySink>, String> {
    match spec {
        SinkSpec::Slack { webhook, .. } => Ok(Box::new(HttpSink {
            kind: HttpKind::Slack,
            endpoint: HttpEndpoint::Secret(webhook.clone()),
            token: None,
        })),
        SinkSpec::Discord { webhook, .. } => Ok(Box::new(HttpSink {
            kind: HttpKind::Discord,
            endpoint: HttpEndpoint::Secret(webhook.clone()),
            token: None,
        })),
        SinkSpec::Ntfy {
            server,
            topic,
            token,
            ..
        } => {
            let safe_topic = !topic.is_empty()
                && topic != "."
                && topic != ".."
                && topic
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'));
            let mut endpoint = reqwest::Url::parse(server)
                .map_err(|_| format!("ntfy sink {:?} has an invalid server URL", spec.name()))?;
            if endpoint.scheme() != "https"
                || endpoint.cannot_be_a_base()
                || endpoint.query().is_some()
                || endpoint.fragment().is_some()
                || !safe_topic
            {
                return Err(format!(
                    "ntfy sink {:?} requires a base https URL and one safe topic segment",
                    spec.name()
                ));
            }
            endpoint
                .path_segments_mut()
                .map_err(|()| format!("ntfy sink {:?} URL is not hierarchical", spec.name()))?
                .pop_if_empty()
                .push(topic);
            Ok(Box::new(HttpSink {
                kind: HttpKind::Ntfy,
                endpoint: HttpEndpoint::Public(endpoint.into()),
                token: token.clone(),
            }))
        }
        SinkSpec::Smtp {
            server,
            port,
            from,
            to,
            username,
            password,
            ..
        } => {
            if !matches!((username, password), (None, None) | (Some(_), Some(_))) {
                return Err(format!(
                    "smtp sink {:?} requires both username and password, or neither",
                    spec.name()
                ));
            }
            if to.is_empty() {
                return Err(format!("smtp sink {:?} has no recipients", spec.name()));
            }
            if server.trim().is_empty()
                || port.is_some_and(|port| port == 0)
                || from.parse::<lettre::message::Mailbox>().is_err()
                || to
                    .iter()
                    .any(|address| address.parse::<lettre::message::Mailbox>().is_err())
            {
                return Err(format!(
                    "smtp sink {:?} has invalid addressing",
                    spec.name()
                ));
            }
            Ok(Box::new(SmtpSink {
                server: server.clone(),
                port: *port,
                from: from.clone(),
                to: to.clone(),
                username: username.clone(),
                password: password.clone(),
            }))
        }
        SinkSpec::Exec {
            program,
            args,
            boundary,
            ..
        } => {
            if program.is_empty() {
                return Err(format!("exec sink {:?} has no program", spec.name()));
            }
            Ok(Box::new(ExecSink::new(
                program,
                args.iter().map(std::ffi::OsString::from).collect(),
                *boundary,
            )))
        }
    }
}

fn resolve_secret(reference: &SecretRef) -> Result<String, String> {
    if reference.restricted {
        return Err(format!(
            "credential reference {:?} is restricted and cannot resolve in a sidecar",
            reference.name
        ));
    }
    std::env::var(&reference.name)
        .map_err(|_| format!("credential reference {:?} is unavailable", reference.name))
}

enum HttpEndpoint {
    Public(String),
    Secret(SecretRef),
}

enum HttpKind {
    Slack,
    Discord,
    Ntfy,
}

struct HttpSink {
    kind: HttpKind,
    endpoint: HttpEndpoint,
    token: Option<SecretRef>,
}

impl NotifySink for HttpSink {
    fn boundary(&self) -> SinkBoundary {
        SinkBoundary::OffBox
    }

    fn send(&mut self, envelope: &NotificationEnvelope) -> Result<(), String> {
        let endpoint = match &self.endpoint {
            HttpEndpoint::Public(endpoint) => endpoint.clone(),
            HttpEndpoint::Secret(reference) => resolve_secret(reference)?,
        };
        if !endpoint.starts_with("https://") {
            return Err("HTTP notification endpoint must use https".to_string());
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|_| "build HTTP notification client".to_string())?;
        let request = match self.kind {
            HttpKind::Slack => client
                .post(&endpoint)
                .json(&serde_json::json!({"text": envelope.summary})),
            HttpKind::Discord => client
                .post(&endpoint)
                .json(&serde_json::json!({"content": envelope.summary})),
            HttpKind::Ntfy => {
                let mut request = client
                    .post(&endpoint)
                    .header("content-type", "text/plain; charset=utf-8")
                    .body(envelope.summary.clone());
                if let Some(reference) = &self.token {
                    let token = resolve_secret(reference)?;
                    request = request.bearer_auth(token);
                }
                request
            }
        };
        let response = request
            .send()
            .map_err(|_| "HTTP notification request failed".to_string())?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!(
                "HTTP notification endpoint returned {}",
                response.status()
            ))
        }
    }
}

struct SmtpSink {
    server: String,
    port: Option<u16>,
    from: String,
    to: Vec<String>,
    username: Option<SecretRef>,
    password: Option<SecretRef>,
}

impl NotifySink for SmtpSink {
    fn boundary(&self) -> SinkBoundary {
        SinkBoundary::OffBox
    }

    fn send(&mut self, envelope: &NotificationEnvelope) -> Result<(), String> {
        use lettre::{Message, SmtpTransport, Transport as _};

        let from = self
            .from
            .parse()
            .map_err(|_| "invalid SMTP from address".to_string())?;
        let mut builder = Message::builder().from(from).subject("BLUT notification");
        for recipient in &self.to {
            builder = builder.to(recipient
                .parse()
                .map_err(|_| "invalid SMTP recipient address".to_string())?);
        }
        let message = builder
            .body(envelope.summary.clone())
            .map_err(|_| "build SMTP notification".to_string())?;
        let mut transport = SmtpTransport::relay(&self.server)
            .map_err(|_| "build TLS SMTP transport".to_string())?;
        if let Some(port) = self.port {
            transport = transport.port(port);
        }
        if let (Some(username), Some(password)) = (&self.username, &self.password) {
            transport =
                transport.credentials(lettre::transport::smtp::authentication::Credentials::new(
                    resolve_secret(username)?,
                    resolve_secret(password)?,
                ));
        }
        transport
            .build()
            .send(&message)
            .map(|_| ())
            .map_err(|_| "SMTP notification request failed".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_network_sinks_are_off_box_and_exec_declares_its_boundary() {
        let slack = SinkSpec::Slack {
            name: "slack".into(),
            webhook: SecretRef::new("SLACK_URL"),
        };
        assert_eq!(build_sink(&slack).unwrap().boundary(), SinkBoundary::OffBox);

        let exec = SinkSpec::Exec {
            name: "local".into(),
            program: "true".into(),
            args: Vec::new(),
            boundary: SinkBoundary::Local,
        };
        assert_eq!(build_sink(&exec).unwrap().boundary(), SinkBoundary::Local);
    }

    #[test]
    fn restricted_credential_never_resolves_in_sidecar() {
        assert!(resolve_secret(&SecretRef::restricted("PHI_KEY")).is_err());
    }
}
