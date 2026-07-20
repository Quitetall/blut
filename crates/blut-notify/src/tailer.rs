// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Durable `status.jsonl` / `sla.jsonl` tailer and rule router (ADR 0094).

use std::io::{Read as _, Seek as _};

use blut_types::sla::SlaBreach;
use blut_types::tenant::Tenant;
use blut_types::trust::DataClass;
use serde::{Deserialize, Serialize};

use crate::config::{NotifyConfig, NotifyRule, NotifySource};
use crate::{NotificationEnvelope, NotifyError, NotifySink, deliver};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunReport {
    pub lines_read: usize,
    pub matched: usize,
    pub delivered: usize,
    pub custody_refused: usize,
    pub malformed: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CursorFile {
    #[serde(default)]
    file: std::collections::BTreeMap<String, Cursor>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Cursor {
    identity: String,
    offset: u64,
}

struct LockGuard {
    _file: nix::fcntl::Flock<std::fs::File>,
}

/// One notifier process. Holds an exclusive cursor lock for its lifetime so two
/// daemons cannot race delivery from the same durable offset ledger.
pub struct Notifier {
    rules: Vec<NotifyRule>,
    sinks: std::collections::HashMap<String, Box<dyn NotifySink>>,
    cursors: CursorFile,
    cursor_path: std::path::PathBuf,
    _lock: LockGuard,
}

impl Notifier {
    pub fn open(config: NotifyConfig, cursor_path: std::path::PathBuf) -> Result<Self, String> {
        let mut sinks = std::collections::HashMap::new();
        for spec in &config.sink {
            sinks.insert(spec.name().to_string(), crate::sinks::build_sink(spec)?);
        }
        Self::with_sinks(config.rule, sinks, cursor_path)
    }

    fn with_sinks(
        rules: Vec<NotifyRule>,
        sinks: std::collections::HashMap<String, Box<dyn NotifySink>>,
        cursor_path: std::path::PathBuf,
    ) -> Result<Self, String> {
        if let Some(parent) = cursor_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|_| "create notification state directory".to_string())?;
        }
        let lock_path = cursor_path.with_extension("lock");
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let lock_file = options.open(&lock_path).map_err(|error| {
            format!(
                "notification cursor {} lock is unavailable: {error}",
                cursor_path.display()
            )
        })?;
        let lock_file =
            nix::fcntl::Flock::lock(lock_file, nix::fcntl::FlockArg::LockExclusiveNonblock)
                .map_err(|_| {
                    format!(
                        "notification cursor {} is already locked",
                        cursor_path.display()
                    )
                })?;
        let cursors = match std::fs::read(&cursor_path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|_| "parse notification cursor state".to_string())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => CursorFile::default(),
            Err(_) => return Err("read notification cursor state".to_string()),
        };
        Ok(Self {
            rules,
            sinks,
            cursors,
            cursor_path,
            _lock: LockGuard { _file: lock_file },
        })
    }

    /// Process every newly-completed line and durably advance cursors only
    /// after all matching deliveries succeed. A crash may replay a delivered
    /// line (at-least-once); it cannot silently skip an undelivered line.
    pub fn run_once(
        &mut self,
        jobs_dir: &std::path::Path,
        sla_path: &std::path::Path,
    ) -> Result<RunReport, String> {
        let before = self.cursors.clone();
        match self.run_once_inner(jobs_dir, sla_path) {
            Ok(report) => Ok(report),
            Err(error) => {
                // A long-running process must retry the same line after a sink
                // failure just like a restarted process would from disk.
                self.cursors = before;
                Err(error)
            }
        }
    }

    fn run_once_inner(
        &mut self,
        jobs_dir: &std::path::Path,
        sla_path: &std::path::Path,
    ) -> Result<RunReport, String> {
        let mut report = RunReport::default();
        let mut jobs: Vec<std::path::PathBuf> = match std::fs::read_dir(jobs_dir) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .collect(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(_) => return Err("read jobs directory".to_string()),
        };
        jobs.sort();
        for job_dir in jobs {
            let Some(job_id) = job_dir
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
            else {
                continue;
            };
            let Some(tenant) = read_tenant(&job_dir) else {
                report.malformed += 1; // fail closed on invalid custody marker
                continue;
            };
            let data_class = if tenant.is_restricted() {
                DataClass::Restricted
            } else {
                DataClass::Internal
            };
            for status_path in [job_dir.join("status.jsonl.1"), job_dir.join("status.jsonl")] {
                for line in self.read_new_lines(&status_path)? {
                    report.lines_read += 1;
                    let value: serde_json::Value = match serde_json::from_str(&line) {
                        Ok(value) => value,
                        Err(_) => {
                            report.malformed += 1;
                            continue;
                        }
                    };
                    let class = if value
                        .get("data_class")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|value| value.eq_ignore_ascii_case("restricted"))
                    {
                        DataClass::Restricted
                    } else {
                        data_class
                    };
                    let envelope = NotificationEnvelope {
                        tenant: tenant.clone(),
                        data_class: class,
                        // Never echo status payload/error text: job id + rule
                        // are the PHI-free operational identity.
                        summary: String::new(),
                    };
                    self.route(
                        NotifySource::Status,
                        &value,
                        envelope,
                        &format!("status event for job {job_id}"),
                        &mut report,
                    )?;
                }
            }
        }

        for line in self.read_new_lines(sla_path)? {
            report.lines_read += 1;
            let breach = match SlaBreach::from_line(&line) {
                Ok(breach) => breach,
                Err(_) => {
                    report.malformed += 1;
                    continue;
                }
            };
            let value = serde_json::to_value(&breach)
                .map_err(|_| "serialize SLA breach for matching".to_string())?;
            let envelope = crate::breach_to_envelope(&breach);
            self.route(
                NotifySource::Sla,
                &value,
                envelope,
                &breach.summary,
                &mut report,
            )?;
        }

        self.save_cursors()?;
        Ok(report)
    }

    fn route(
        &mut self,
        source: NotifySource,
        value: &serde_json::Value,
        envelope: NotificationEnvelope,
        summary: &str,
        report: &mut RunReport,
    ) -> Result<(), String> {
        for rule in &self.rules {
            if !rule.matches(source, value) {
                continue;
            }
            report.matched += 1;
            let mut envelope = envelope.clone();
            envelope.summary = format!("rule '{}': {summary}", rule.name);
            for name in &rule.sinks {
                let sink = self
                    .sinks
                    .get_mut(name)
                    .ok_or_else(|| format!("rule references unavailable sink {name:?}"))?;
                match deliver(sink.as_mut(), &envelope) {
                    Ok(()) => report.delivered += 1,
                    Err(NotifyError::RestrictedOutbound { .. }) => report.custody_refused += 1,
                    Err(NotifyError::Sink(error)) => {
                        return Err(format!("sink {name:?} failed: {error}"));
                    }
                }
            }
        }
        Ok(())
    }

    fn read_new_lines(&mut self, path: &std::path::Path) -> Result<Vec<String>, String> {
        let metadata = match std::fs::metadata(path) {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => return Ok(Vec::new()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(_) => return Err(format!("stat notification source {}", path.display())),
        };
        let key = path.to_string_lossy().into_owned();
        let identity = file_identity(&metadata);
        // Rotation renames `status.jsonl` to `status.jsonl.1`. Carry the
        // inode/file-identity cursor across that path change so rotation does
        // not systematically redeliver the entire old stream.
        let inherited_offset = self
            .cursors
            .file
            .values()
            .filter(|cursor| cursor.identity == identity)
            .map(|cursor| cursor.offset)
            .max()
            .unwrap_or(0);
        let cursor = self.cursors.file.entry(key).or_default();
        if cursor.identity != identity {
            cursor.identity = identity;
            cursor.offset = inherited_offset;
        }
        if cursor.offset > metadata.len() {
            cursor.offset = 0; // copy-truncate or corrupt/stale cursor
        }
        let mut file = std::fs::File::open(path)
            .map_err(|_| format!("open notification source {}", path.display()))?;
        file.seek(std::io::SeekFrom::Start(cursor.offset))
            .map_err(|_| format!("seek notification source {}", path.display()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|_| format!("read notification source {}", path.display()))?;
        let Some(last_newline) = bytes.iter().rposition(|byte| *byte == b'\n') else {
            return Ok(Vec::new()); // keep incomplete tail for next poll
        };
        let complete = &bytes[..=last_newline];
        let text = std::str::from_utf8(complete)
            .map_err(|_| format!("notification source {} is not UTF-8", path.display()))?;
        cursor.offset += complete.len() as u64;
        Ok(text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(str::to_string)
            .collect())
    }

    fn save_cursors(&self) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(&self.cursors)
            .map_err(|_| "serialize notification cursor state".to_string())?;
        let temp = self.cursor_path.with_extension("tmp");
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temp)
            .map_err(|_| "write notification cursor state".to_string())?;
        use std::io::Write as _;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|_| "sync notification cursor state".to_string())?;
        std::fs::rename(&temp, &self.cursor_path)
            .map_err(|_| "replace notification cursor state".to_string())
    }
}

fn read_tenant(job_dir: &std::path::Path) -> Option<Tenant> {
    match std::fs::read_to_string(job_dir.join("tenant")) {
        Ok(value) => {
            let value = value.trim();
            Tenant::parse(value).filter(|tenant| tenant.to_string() == value)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Some(Tenant::default()),
        Err(_) => None,
    }
}

#[cfg(unix)]
fn file_identity(metadata: &std::fs::Metadata) -> String {
    use std::os::unix::fs::MetadataExt as _;
    format!("{}:{}", metadata.dev(), metadata.ino())
}

#[cfg(not(unix))]
fn file_identity(metadata: &std::fs::Metadata) -> String {
    let modified = metadata
        .modified()
        .ok()
        .and_then(|instant| instant.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("{}:{modified}", metadata.len())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::SinkBoundary;

    struct Capture {
        boundary: SinkBoundary,
        seen: Arc<Mutex<Vec<NotificationEnvelope>>>,
    }

    struct FailOnce {
        attempts: Arc<Mutex<usize>>,
    }

    impl NotifySink for FailOnce {
        fn boundary(&self) -> SinkBoundary {
            SinkBoundary::Local
        }

        fn send(&mut self, _envelope: &NotificationEnvelope) -> Result<(), String> {
            let mut attempts = self.attempts.lock().unwrap();
            *attempts += 1;
            if *attempts == 1 {
                Err("injected failure".into())
            } else {
                Ok(())
            }
        }
    }

    impl NotifySink for Capture {
        fn boundary(&self) -> SinkBoundary {
            self.boundary
        }

        fn send(&mut self, envelope: &NotificationEnvelope) -> Result<(), String> {
            self.seen.lock().unwrap().push(envelope.clone());
            Ok(())
        }
    }

    fn rule() -> NotifyRule {
        NotifyRule {
            name: "failed".into(),
            source: NotifySource::Status,
            field: "kind".into(),
            equals: "failed".into(),
            sinks: vec!["capture".into()],
        }
    }

    #[test]
    fn status_tail_routes_once_and_never_echoes_payload() {
        let td = tempfile::tempdir().unwrap();
        let job = td.path().join("jobs/j1");
        std::fs::create_dir_all(&job).unwrap();
        std::fs::write(job.join("tenant"), "shared").unwrap();
        std::fs::write(
            job.join("status.jsonl"),
            "{\"kind\":\"failed\",\"error\":\"patient Alice\"}\n",
        )
        .unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut sinks: std::collections::HashMap<String, Box<dyn NotifySink>> =
            std::collections::HashMap::new();
        sinks.insert(
            "capture".into(),
            Box::new(Capture {
                boundary: SinkBoundary::Local,
                seen: seen.clone(),
            }),
        );
        let mut notifier =
            Notifier::with_sinks(vec![rule()], sinks, td.path().join("cursor.json")).unwrap();
        let first = notifier
            .run_once(&td.path().join("jobs"), &td.path().join("sla.jsonl"))
            .unwrap();
        assert_eq!(first.delivered, 1);
        assert!(!seen.lock().unwrap()[0].summary.contains("Alice"));
        let second = notifier
            .run_once(&td.path().join("jobs"), &td.path().join("sla.jsonl"))
            .unwrap();
        assert_eq!(second.delivered, 0);
    }

    #[test]
    fn restricted_status_never_reaches_off_box_sink() {
        let td = tempfile::tempdir().unwrap();
        let job = td.path().join("jobs/j1");
        std::fs::create_dir_all(&job).unwrap();
        std::fs::write(job.join("tenant"), "clinical/prod").unwrap();
        std::fs::write(job.join("status.jsonl"), "{\"kind\":\"failed\"}\n").unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut sinks: std::collections::HashMap<String, Box<dyn NotifySink>> =
            std::collections::HashMap::new();
        sinks.insert(
            "capture".into(),
            Box::new(Capture {
                boundary: SinkBoundary::OffBox,
                seen: seen.clone(),
            }),
        );
        let mut notifier =
            Notifier::with_sinks(vec![rule()], sinks, td.path().join("cursor.json")).unwrap();
        let report = notifier
            .run_once(&td.path().join("jobs"), &td.path().join("sla.jsonl"))
            .unwrap();
        assert_eq!(report.custody_refused, 1);
        assert!(seen.lock().unwrap().is_empty());
    }

    #[test]
    fn status_rotation_carries_the_cursor_by_file_identity() {
        let td = tempfile::tempdir().unwrap();
        let job = td.path().join("jobs/j1");
        std::fs::create_dir_all(&job).unwrap();
        std::fs::write(job.join("tenant"), "shared").unwrap();
        let current = job.join("status.jsonl");
        std::fs::write(&current, "{\"kind\":\"failed\"}\n").unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut sinks: std::collections::HashMap<String, Box<dyn NotifySink>> =
            std::collections::HashMap::new();
        sinks.insert(
            "capture".into(),
            Box::new(Capture {
                boundary: SinkBoundary::Local,
                seen: seen.clone(),
            }),
        );
        let mut notifier =
            Notifier::with_sinks(vec![rule()], sinks, td.path().join("cursor.json")).unwrap();
        assert_eq!(
            notifier
                .run_once(&td.path().join("jobs"), &td.path().join("sla.jsonl"))
                .unwrap()
                .delivered,
            1
        );

        std::fs::rename(&current, job.join("status.jsonl.1")).unwrap();
        std::fs::write(&current, "{\"kind\":\"failed\"}\n").unwrap();
        let after_rotation = notifier
            .run_once(&td.path().join("jobs"), &td.path().join("sla.jsonl"))
            .unwrap();
        assert_eq!(after_rotation.delivered, 1);
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[test]
    fn sink_failure_retries_the_same_line_in_the_same_process() {
        let td = tempfile::tempdir().unwrap();
        let job = td.path().join("jobs/j1");
        std::fs::create_dir_all(&job).unwrap();
        std::fs::write(job.join("tenant"), "shared").unwrap();
        std::fs::write(job.join("status.jsonl"), "{\"kind\":\"failed\"}\n").unwrap();
        let attempts = Arc::new(Mutex::new(0));
        let mut sinks: std::collections::HashMap<String, Box<dyn NotifySink>> =
            std::collections::HashMap::new();
        sinks.insert(
            "capture".into(),
            Box::new(FailOnce {
                attempts: attempts.clone(),
            }),
        );
        let mut notifier =
            Notifier::with_sinks(vec![rule()], sinks, td.path().join("cursor.json")).unwrap();
        assert!(
            notifier
                .run_once(&td.path().join("jobs"), &td.path().join("sla.jsonl"))
                .is_err()
        );
        assert_eq!(
            notifier
                .run_once(&td.path().join("jobs"), &td.path().join("sla.jsonl"))
                .unwrap()
                .delivered,
            1
        );
        assert_eq!(
            notifier
                .run_once(&td.path().join("jobs"), &td.path().join("sla.jsonl"))
                .unwrap()
                .delivered,
            0
        );
        assert_eq!(*attempts.lock().unwrap(), 2);
    }

    #[test]
    fn cursor_lock_is_exclusive_and_releases_with_the_process_handle() {
        let td = tempfile::tempdir().unwrap();
        let cursor = td.path().join("cursor.json");
        let first = Notifier::with_sinks(Vec::new(), Default::default(), cursor.clone()).unwrap();
        assert!(Notifier::with_sinks(Vec::new(), Default::default(), cursor.clone()).is_err());
        drop(first);
        Notifier::with_sinks(Vec::new(), Default::default(), cursor).unwrap();
    }
}
