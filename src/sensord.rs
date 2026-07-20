// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Long-running trigger daemon wiring (ADR 0094).
//!
//! This module owns daemon/config/process choreography so the CLI stays a thin
//! command surface. Trigger implementations and durable dedupe live in
//! [`crate::trigger`]; exact resource admission remains in the normal recipe
//! launch path.

use anyhow::{Context as _, Result, anyhow, bail};

struct Bound {
    name: String,
    trigger: Box<dyn crate::trigger::Trigger>,
    plan: String,
    tenant: String,
    seen: crate::trigger::SeenStore,
}

/// Run `blut sensord`. `once` drives one deterministic poll for smoke/CI;
/// daemon mode sleeps at least one second between polls.
pub async fn run(
    triggers: Option<std::path::PathBuf>,
    once: bool,
    interval: u64,
    cli: std::path::PathBuf,
) -> Result<()> {
    let from_env = std::env::var_os("BLUT_TRIGGERS").map(std::path::PathBuf::from);
    let explicitly_configured = triggers.is_some() || from_env.is_some();
    let path = triggers
        .or(from_env)
        .unwrap_or_else(|| dot_blut().join("triggers.toml"));
    let parsed = match crate::trigger::TriggerConfig::load(&path) {
        Ok(parsed) => parsed,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !explicitly_configured => {
            crate::trigger::TriggerConfig::default()
        }
        Err(error) => {
            return Err(error).with_context(|| format!("load trigger bindings {}", path.display()));
        }
    };
    let sla_rules = std::env::var_os("BLUT_SLA_RULES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| dot_blut().join("sla.toml"));
    let sla_out = crate::sla::default_sla_path();
    if parsed.trigger.is_empty() && !sla_rules.is_file() {
        println!(
            "sensord: no triggers in {} and no SLA rules in {} — nothing to watch",
            path.display(),
            sla_rules.display()
        );
        return Ok(());
    }

    let mut bound = Vec::with_capacity(parsed.trigger.len());
    for binding in &parsed.trigger {
        let trigger = build_trigger(binding, interval)?;
        let seen = crate::trigger::SeenStore::load(crate::trigger::seen_path(&binding.name))
            .with_context(|| format!("load dedupe store for trigger '{}'", binding.name))?;
        bound.push(Bound {
            name: binding.name.clone(),
            trigger,
            plan: binding.plan.clone(),
            tenant: binding.tenant.clone(),
            seen,
        });
    }

    loop {
        for binding in &mut bound {
            let timeout = std::time::Duration::from_secs(parsed.admission_timeout_secs.max(1));
            let dispatcher = crate::trigger::ChildAdmissionDispatcher {
                launch: {
                    let cli = cli.clone();
                    let tenant = binding.tenant.clone();
                    let ack_root = crate::trigger::seen_path(&binding.name);
                    Box::new(move |plan, event| {
                        let seen = crate::trigger::SeenStore::load(&ack_root)?;
                        let ack = seen.admission_ack_path(&event.id)?;
                        launch_and_wait_for_admission(&cli, &tenant, plan, event, &ack, timeout)
                    })
                },
            };
            match crate::trigger::drive_once(
                binding.trigger.as_ref(),
                &binding.plan,
                &mut binding.seen,
                &dispatcher,
            ) {
                Ok(results) => {
                    for (event, outcome) in results {
                        match outcome {
                            crate::trigger::DispatchOutcome::Admitted => eprintln!(
                                "sensord: admitted '{}' for event {}",
                                binding.plan, event.source
                            ),
                            crate::trigger::DispatchOutcome::Refused(reason) => eprintln!(
                                "sensord: '{}' refused for event {} ({reason}) — will retry",
                                binding.plan, event.source
                            ),
                        }
                    }
                }
                Err(error) => {
                    eprintln!("sensord: poll failed for plan '{}': {error}", binding.plan)
                }
            }
        }
        let sla_report = crate::sla::check_paths(&sla_rules, &sla_out)
            .with_context(|| format!("evaluate SLA rules {}", sla_rules.display()))?;
        if sla_report.new_rows > 0 {
            eprintln!(
                "sensord: {} active SLA breach(es), {} new row(s) in {}",
                sla_report.breaches.len(),
                sla_report.new_rows,
                sla_out.display()
            );
        }
        if once {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_secs(interval.max(1))).await;
    }
}

fn build_trigger(
    binding: &crate::trigger::TriggerBinding,
    interval: u64,
) -> Result<Box<dyn crate::trigger::Trigger>> {
    let dir = binding
        .dir
        .clone()
        .unwrap_or_else(|| crate::trigger::spool_dir(&binding.name));
    match binding.kind {
        crate::trigger::TriggerKind::FileDrop => Ok(Box::new(
            crate::trigger::FileDropTrigger::new(binding.name.clone(), dir, binding.ext.clone()),
        )),
        crate::trigger::TriggerKind::Spool => {
            let metric = binding
                .spool_metric
                .ok_or_else(|| anyhow!("spool trigger {:?} requires spool_metric", binding.name))?;
            let direction = binding.spool_direction.ok_or_else(|| {
                anyhow!("spool trigger {:?} requires spool_direction", binding.name)
            })?;
            let threshold = binding
                .threshold
                .ok_or_else(|| anyhow!("spool trigger {:?} requires threshold", binding.name))?;
            Ok(Box::new(crate::trigger::SpoolThresholdTrigger::new(
                binding.name.clone(),
                dir,
                metric,
                direction,
                threshold,
            )))
        }
        crate::trigger::TriggerKind::Cron => {
            if binding.dir.is_some() || binding.ext.is_some() {
                bail!("cron trigger {:?} cannot declare dir/ext", binding.name);
            }
            let expression = binding
                .schedule
                .clone()
                .ok_or_else(|| anyhow!("cron trigger {:?} requires schedule", binding.name))?;
            let grace = std::time::Duration::from_secs(
                binding.grace_secs.unwrap_or_else(|| interval.max(1)),
            );
            Ok(Box::new(
                crate::trigger::CronTrigger::new(binding.name.clone(), expression, grace)
                    .map_err(|error| anyhow!("invalid cron for {:?}: {error}", binding.name))?,
            ))
        }
    }
}

fn launch_and_wait_for_admission(
    cli: &std::path::Path,
    tenant: &str,
    plan: &str,
    event: &crate::trigger::TriggerEvent,
    ack: &std::path::Path,
    timeout: std::time::Duration,
) -> std::io::Result<()> {
    if ack.exists() {
        return Ok(()); // recovered durable admission; never launch twice
    }

    let mut command = std::process::Command::new(cli);
    if crate::registry_db::parse_pointer_uri(plan).is_some() {
        command
            .arg("recipe")
            .arg("declare")
            .arg(plan)
            .arg("--run")
            .arg("--tenant")
            .arg(tenant);
    } else {
        command
            .arg("recipe")
            .arg("run")
            .arg(plan)
            .arg("--tenant")
            .arg(tenant);
    }
    command
        .env(crate::trigger::ADMISSION_ACK_PATH_ENV, ack)
        .env(crate::trigger::ADMISSION_EVENT_ID_ENV, &event.id)
        .stdin(std::process::Stdio::null());
    let mut child = command.spawn()?;
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if ack.is_file() {
            // `Child` has no async drop/reap. Keep a bounded waiter for each
            // admitted job so a long-running sensord cannot accumulate zombie
            // recipe processes after they finish.
            if let Err(error) = std::thread::Builder::new()
                .name("blut-trigger-reaper".to_string())
                .spawn(move || {
                    let _ = child.wait();
                })
            {
                eprintln!("sensord: could not start child reaper: {error}");
            }
            return Ok(());
        }
        if let Some(status) = child.try_wait()? {
            return Err(std::io::Error::other(format!(
                "{} exited before broker admission acknowledgement ({status})",
                cli.display()
            )));
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "{} did not acknowledge broker admission within {}s",
                    cli.display(),
                    timeout.as_secs()
                ),
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn dot_blut() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_default()
        .join(".blut")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_restricted_trigger_without_restricted_classification() {
        let config = crate::trigger::TriggerConfig::parse(
            r#"
[[trigger]]
name = "clinical"
plan = "demo"
tenant = "clinical/prod"
"#,
        )
        .unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn accepts_registry_pointer_and_exact_tenant() {
        let config = crate::trigger::TriggerConfig::parse(
            r#"
[[trigger]]
name = "nightly"
plan = "registry://plan@prod"
tenant = "research/dev"
"#,
        )
        .unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn builds_spool_and_cron_trigger_kinds() {
        let config = crate::trigger::TriggerConfig::parse(
            r#"
[[trigger]]
name = "queue"
kind = "spool"
plan = "demo"
spool_metric = "files"
spool_direction = "at_least"
threshold = 2

[[trigger]]
name = "nightly"
kind = "cron"
plan = "registry://plan@prod"
schedule = "0 0 2 * * * *"
"#,
        )
        .unwrap();
        config.validate().unwrap();
        for binding in &config.trigger {
            build_trigger(binding, 15).unwrap();
        }
    }

    #[cfg(unix)]
    #[test]
    fn child_launch_waits_for_the_admission_marker() {
        use std::os::unix::fs::PermissionsExt as _;

        let td = tempfile::tempdir().unwrap();
        let script = td.path().join("fake-blut");
        std::fs::write(
            &script,
            "#!/bin/sh\nmkdir -p \"$(dirname \"$BLUT_TRIGGER_ADMISSION_ACK\")\"\nprintf 'job-test\\n' > \"$BLUT_TRIGGER_ADMISSION_ACK\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        let id = "c".repeat(64);
        let ack = td.path().join("events/demo/.admitted").join(&id);
        let event = crate::trigger::TriggerEvent {
            id,
            source: "test".into(),
        };
        launch_and_wait_for_admission(
            &script,
            "shared",
            "demo",
            &event,
            &ack,
            std::time::Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(std::fs::read_to_string(ack).unwrap(), "job-test\n");
    }
}
