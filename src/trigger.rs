// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Event/data-driven TRIGGERS (ADR 0094) — the fire-on-the-world sibling of
//! [`crate::sensor`] (which answers "may I proceed?"). A [`Trigger`] observes an
//! external source (a file landing, a spool filling, a cron firing) and yields
//! zero or more [`TriggerEvent`]s, each carrying a stable content id so a
//! repeated observation of the *same* event dedupes to a single dispatch.
//!
//! Charter-clean by construction: a trigger is a pure reader (it never mutates),
//! and a fired event dispatches a registered plan only through a [`Dispatcher`]
//! that MUST route the launch through broker admission (ADR 0046/0047) — a
//! trigger flood can never stampede the box, because the same envelope throttles
//! triggered and human runs identically. This module owns the substrate + the
//! dedupe + the drive loop; the concrete admission-and-launch dispatcher and the
//! `blut sensord` daemon that schedules polls live in the CLI/daemon layer.

use std::collections::HashSet;
use std::path::PathBuf;

use blut_types::trust::DataClass;

/// One declarative trigger binding shared by `blut sensord` and the web
/// sidecar. Keeping ingress authentication and dispatch custody in one file
/// prevents the listener and daemon from silently disagreeing about tenant or
/// classification.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerBinding {
    pub name: String,
    /// Trigger implementation. Omitted for backwards-compatible file-drop.
    #[serde(default)]
    pub kind: TriggerKind,
    /// Watched directory. Default: the trigger's content-addressed spool.
    #[serde(default)]
    pub dir: Option<PathBuf>,
    /// Extension filter (no dot), e.g. `json`.
    #[serde(default)]
    pub ext: Option<String>,
    /// Registered recipe or `registry://plan@<name>` dispatched on fire.
    pub plan: String,
    /// Owning tenant. The daemon always forwards this exact identity to the
    /// launched plan; restricted tenants therefore remain same-tenant/local.
    #[serde(default = "default_trigger_tenant")]
    pub tenant: String,
    /// Explicit payload classification. A restricted tenant dominates this
    /// value at custody checks.
    #[serde(default = "default_data_class")]
    pub data_class: DataClass,
    /// HMAC credential reference for webhook ingress. No entry means this
    /// trigger has no web ingress route, even if it watches a local spool.
    #[serde(default)]
    pub webhook_secret: Option<crate::secrets::SecretRef>,
    /// Spool threshold metric/direction/value (`kind = "spool"`).
    #[serde(default)]
    pub spool_metric: Option<SpoolMetric>,
    #[serde(default)]
    pub spool_direction: Option<SpoolDirection>,
    #[serde(default)]
    pub threshold: Option<u64>,
    /// Seven-field cron expression (`sec min hour day month weekday year`).
    #[serde(default)]
    pub schedule: Option<String>,
    /// How long after a scheduled instant a daemon poll may still fire it.
    #[serde(default)]
    pub grace_secs: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TriggerKind {
    #[default]
    FileDrop,
    Spool,
    Cron,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpoolMetric {
    Files,
    Bytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpoolDirection {
    AtLeast,
    AtMost,
}

fn default_trigger_tenant() -> String {
    "default".to_string()
}

fn default_data_class() -> DataClass {
    DataClass::Internal
}

/// Shared trigger configuration (`[[trigger]]` tables).
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TriggerConfig {
    #[serde(default)]
    pub trigger: Vec<TriggerBinding>,
    /// Maximum accepted webhook clock skew in seconds.
    #[serde(default = "default_webhook_max_skew_secs")]
    pub webhook_max_skew_secs: u64,
    /// How long sensord waits for the child recipe's exact broker-admission
    /// acknowledgement before terminating it and leaving the event retryable.
    #[serde(default = "default_admission_timeout_secs")]
    pub admission_timeout_secs: u64,
    /// How long a dispatched file-drop event is kept in `<spool>/.processed/`
    /// before deletion. This bounds all three otherwise-unbounded stores: the
    /// spool itself, the `.seen` dedupe log, and the `.admitted` markers (the
    /// latter two are reclaimed once an event stops being observable). `0`
    /// disables archival — events then stay in the watched directory forever
    /// and their dedupe records can never be reclaimed.
    #[serde(default = "default_spool_retention_secs")]
    pub spool_retention_secs: u64,
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            trigger: Vec::new(),
            webhook_max_skew_secs: default_webhook_max_skew_secs(),
            admission_timeout_secs: default_admission_timeout_secs(),
            spool_retention_secs: default_spool_retention_secs(),
        }
    }
}

const fn default_webhook_max_skew_secs() -> u64 {
    300
}

const fn default_admission_timeout_secs() -> u64 {
    60
}

/// Seven days: long enough to inspect what a trigger acted on, short enough
/// that a busy webhook spool cannot grow without bound.
const fn default_spool_retention_secs() -> u64 {
    7 * 24 * 60 * 60
}

impl TriggerConfig {
    pub fn parse(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let config = Self::parse(&text).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("parse {}: {error}", path.display()),
            )
        })?;
        config.validate().map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("validate {}: {error}", path.display()),
            )
        })?;
        Ok(config)
    }

    pub fn binding(&self, name: &str) -> Option<&TriggerBinding> {
        self.trigger.iter().find(|binding| binding.name == name)
    }

    pub fn validate(&self) -> Result<(), String> {
        use std::str::FromStr as _;

        if self.webhook_max_skew_secs == 0 || self.webhook_max_skew_secs > 3600 {
            return Err("webhook_max_skew_secs must be in 1..=3600".to_string());
        }
        if self.admission_timeout_secs == 0 || self.admission_timeout_secs > 3600 {
            return Err("admission_timeout_secs must be in 1..=3600".to_string());
        }
        let mut names = HashSet::new();
        for binding in &self.trigger {
            if !safe_component(&binding.name) || !names.insert(binding.name.clone()) {
                return Err(format!(
                    "invalid or duplicate trigger name {:?}",
                    binding.name
                ));
            }
            if binding
                .ext
                .as_ref()
                .is_some_and(|extension| !safe_component(extension))
            {
                return Err(format!("invalid extension for trigger {:?}", binding.name));
            }
            let tenant = crate::tenant::Tenant::parse(&binding.tenant).ok_or_else(|| {
                format!(
                    "invalid tenant {:?} for trigger {:?}",
                    binding.tenant, binding.name
                )
            })?;
            if tenant.is_restricted() && binding.data_class != DataClass::Restricted {
                return Err(format!(
                    "restricted trigger {:?} must declare data_class = \"Restricted\"",
                    binding.name
                ));
            }
            if let Some(secret) = &binding.webhook_secret {
                secret.validate().map_err(|error| {
                    format!("invalid webhook credential for {:?}: {error}", binding.name)
                })?;
            }
            if crate::registry_db::parse_pointer_uri(&binding.plan).is_none()
                && !safe_component(&binding.plan)
            {
                return Err(format!("invalid recipe/plan target {:?}", binding.plan));
            }
            if binding.webhook_secret.is_some()
                && (binding.kind != TriggerKind::FileDrop
                    || binding.ext.as_deref().is_some_and(|ext| ext != "json"))
            {
                return Err(format!(
                    "webhook trigger {:?} must be file-drop with ext omitted or \"json\"",
                    binding.name
                ));
            }
            match binding.kind {
                TriggerKind::FileDrop => {
                    if binding.spool_metric.is_some()
                        || binding.spool_direction.is_some()
                        || binding.threshold.is_some()
                        || binding.schedule.is_some()
                        || binding.grace_secs.is_some()
                    {
                        return Err(format!(
                            "file-drop trigger {:?} has fields for another kind",
                            binding.name
                        ));
                    }
                }
                TriggerKind::Spool => {
                    if binding.spool_metric.is_none()
                        || binding.spool_direction.is_none()
                        || binding.threshold.is_none()
                        || binding.schedule.is_some()
                        || binding.grace_secs.is_some()
                    {
                        return Err(format!(
                            "spool trigger {:?} requires metric/direction/threshold only",
                            binding.name
                        ));
                    }
                }
                TriggerKind::Cron => {
                    if binding.dir.is_some()
                        || binding.ext.is_some()
                        || binding.spool_metric.is_some()
                        || binding.spool_direction.is_some()
                        || binding.threshold.is_some()
                    {
                        return Err(format!(
                            "cron trigger {:?} has non-cron fields",
                            binding.name
                        ));
                    }
                    let expression = binding.schedule.as_ref().ok_or_else(|| {
                        format!("cron trigger {:?} requires schedule", binding.name)
                    })?;
                    cron::Schedule::from_str(expression)
                        .map_err(|error| format!("invalid cron for {:?}: {error}", binding.name))?;
                    if !(1..=3600).contains(&binding.grace_secs.unwrap_or(1)) {
                        return Err(format!(
                            "cron trigger {:?} grace_secs must be in 1..=3600",
                            binding.name
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

pub fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.starts_with('-')
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
}

/// The launch callback an [`AdmissionDispatcher`] invokes only after the broker
/// admits — spawns `blut recipe run <plan>` in production, records in a test.
pub type LaunchFn = Box<dyn Fn(&str, &TriggerEvent) -> std::io::Result<()> + Send + Sync>;

/// One fired event. `id` is a stable content hash used for at-least-once →
/// exactly-once dedupe (the same underlying observation always hashes the same,
/// so a re-poll of an unchanged source does not re-dispatch); `source` is a
/// human-facing origin (a path, a webhook id) for the audit trail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TriggerEvent {
    pub id: String,
    pub source: String,
}

/// A named observation of an external event source. Pure read — a `poll` never
/// mutates on-disk state (dedupe bookkeeping is the caller's [`SeenStore`], not
/// the trigger's).
pub trait Trigger: Send + Sync {
    /// Stable identifier (the key a `triggers.toml` binding names).
    fn name(&self) -> &str;
    /// Snapshot the currently-fired events. Idempotent: polling an unchanged
    /// source yields the same event ids, so the drive loop's dedupe collapses
    /// them to a single dispatch.
    fn poll(&self) -> std::io::Result<Vec<TriggerEvent>>;
}

/// hex sha256 of the parts, domain-separated — the event id.
fn event_id(parts: &[&str]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"blut.trigger.event.v1");
    for p in parts {
        h.update((p.len() as u64).to_le_bytes()); // length-prefix ⇒ no delimiter collision
        h.update(p.as_bytes());
    }
    faster_hex::hex_string(&h.finalize())
}

/// Fire once per file present in a watched directory (optionally filtered by
/// extension) — the file-drop sensor. The event id folds path + CONTENT hash,
/// so rewriting the same bytes is a replay while genuinely new content fires.
/// A missing directory is not an error — it is an empty observation (the spool
/// simply has not been created yet).
pub struct FileDropTrigger {
    name: String,
    dir: PathBuf,
    /// Match only this extension (no leading dot), e.g. `json`. `None` = any.
    ext: Option<String>,
}

impl FileDropTrigger {
    pub fn new(name: impl Into<String>, dir: impl Into<PathBuf>, ext: Option<String>) -> Self {
        Self {
            name: name.into(),
            dir: dir.into(),
            ext,
        }
    }
}

impl Trigger for FileDropTrigger {
    fn name(&self) -> &str {
        &self.name
    }

    fn poll(&self) -> std::io::Result<Vec<TriggerEvent>> {
        let mut out = Vec::new();
        let rd = match std::fs::read_dir(&self.dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(e),
        };
        let mut entries: Vec<PathBuf> = Vec::new();
        for e in rd {
            let p = e?.path();
            if !p.is_file() {
                continue;
            }
            if p.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.'))
            {
                continue; // daemon control files are never user events
            }
            if let Some(want) = &self.ext
                && p.extension().and_then(|x| x.to_str()) != Some(want.as_str())
            {
                continue;
            }
            entries.push(p);
        }
        entries.sort(); // deterministic event order regardless of readdir order
        for p in entries {
            use sha2::{Digest as _, Sha256};
            use std::io::Read as _;
            let mut file = std::fs::File::open(&p)?;
            let mut digest = Sha256::new();
            let mut buf = [0_u8; 64 * 1024];
            loop {
                let n = file.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                digest.update(&buf[..n]);
            }
            let content = faster_hex::hex_string(&digest.finalize());
            let path_s = p.to_string_lossy();
            let id = event_id(&[&path_s, &content]);
            out.push(TriggerEvent {
                id,
                source: path_s.into_owned(),
            });
        }
        Ok(out)
    }
}

/// Fire when a direct-child spool snapshot crosses a declared file-count or
/// byte-size threshold. The event id includes the deterministic snapshot, so a
/// stable over/under-threshold state fires once while a materially changed
/// spool can fire again.
pub struct SpoolThresholdTrigger {
    name: String,
    dir: PathBuf,
    metric: SpoolMetric,
    direction: SpoolDirection,
    threshold: u64,
}

impl SpoolThresholdTrigger {
    pub fn new(
        name: impl Into<String>,
        dir: impl Into<PathBuf>,
        metric: SpoolMetric,
        direction: SpoolDirection,
        threshold: u64,
    ) -> Self {
        Self {
            name: name.into(),
            dir: dir.into(),
            metric,
            direction,
            threshold,
        }
    }
}

impl Trigger for SpoolThresholdTrigger {
    fn name(&self) -> &str {
        &self.name
    }

    fn poll(&self) -> std::io::Result<Vec<TriggerEvent>> {
        let entries = directory_snapshot(&self.dir)?;
        let observed = match self.metric {
            SpoolMetric::Files => entries.len() as u64,
            SpoolMetric::Bytes => entries.iter().map(|(_, bytes, _)| *bytes).sum(),
        };
        let fired = match self.direction {
            SpoolDirection::AtLeast => observed >= self.threshold,
            SpoolDirection::AtMost => observed <= self.threshold,
        };
        if !fired {
            return Ok(Vec::new());
        }
        let snapshot = serde_json::to_string(&entries).map_err(std::io::Error::other)?;
        let id = event_id(&[
            &self.dir.to_string_lossy(),
            &format!("{:?}", self.metric),
            &format!("{:?}", self.direction),
            &self.threshold.to_string(),
            &snapshot,
        ]);
        Ok(vec![TriggerEvent {
            id,
            source: format!(
                "{} ({:?}={observed}, {:?} {})",
                self.dir.display(),
                self.metric,
                self.direction,
                self.threshold
            ),
        }])
    }
}

fn directory_snapshot(dir: &std::path::Path) -> std::io::Result<Vec<(String, u64, u128)>> {
    let read_dir = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut snapshot = Vec::new();
    for entry in read_dir {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if !metadata.is_file() {
            continue;
        }
        if entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with('.'))
        {
            continue;
        }
        let modified = metadata
            .modified()
            .ok()
            .and_then(|instant| instant.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        snapshot.push((
            entry.file_name().to_string_lossy().into_owned(),
            metadata.len(),
            modified,
        ));
    }
    snapshot.sort();
    Ok(snapshot)
}

/// Fire on the most recent cron instant while it remains inside a bounded
/// grace window. Re-polls produce the same scheduled-instant id and therefore
/// dedupe; the next scheduled instant produces a new id.
pub struct CronTrigger {
    name: String,
    expression: String,
    schedule: cron::Schedule,
    grace: chrono::Duration,
}

impl CronTrigger {
    pub fn new(
        name: impl Into<String>,
        expression: impl Into<String>,
        grace: std::time::Duration,
    ) -> Result<Self, String> {
        use std::str::FromStr as _;
        let expression = expression.into();
        let schedule = cron::Schedule::from_str(&expression).map_err(|error| error.to_string())?;
        if grace > std::time::Duration::from_secs(3600) {
            return Err("cron grace must be <= 3600 seconds".to_string());
        }
        let grace = chrono::Duration::from_std(grace).map_err(|error| error.to_string())?;
        Ok(Self {
            name: name.into(),
            expression,
            schedule,
            grace,
        })
    }

    fn poll_at(&self, now: chrono::DateTime<chrono::Utc>) -> Vec<TriggerEvent> {
        let start = now - self.grace - chrono::Duration::seconds(1);
        let Some(scheduled) = self
            .schedule
            .after(&start)
            .take_while(|instant| *instant <= now)
            .last()
        else {
            return Vec::new();
        };
        if now.signed_duration_since(scheduled) > self.grace {
            return Vec::new();
        }
        let scheduled_unix = scheduled.timestamp().to_string();
        vec![TriggerEvent {
            id: event_id(&[&self.expression, &scheduled_unix]),
            source: format!("cron:{}@{}", self.name, scheduled.to_rfc3339()),
        }]
    }
}

impl Trigger for CronTrigger {
    fn name(&self) -> &str {
        &self.name
    }

    fn poll(&self) -> std::io::Result<Vec<TriggerEvent>> {
        Ok(self.poll_at(chrono::Utc::now()))
    }
}

/// Persistent at-least-once → exactly-once dedupe. Holds the set of already
/// -dispatched event ids and appends new ones atomically-enough for a
/// single-daemon writer (the daemon is the sole writer; the file is an append
/// log of ids, loaded whole on start).
pub struct SeenStore {
    path: PathBuf,
    seen: HashSet<String>,
}

impl SeenStore {
    /// Load the seen-id set from `path` (absent ⇒ empty).
    pub fn load(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let mut seen: HashSet<String> = match std::fs::read_to_string(&path) {
            Ok(text) => text.lines().map(str::to_string).collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => HashSet::new(),
            Err(e) => return Err(e),
        };
        // The launched CLI writes one durable marker only after exact plan
        // admission succeeds. Recover those markers before polling so a daemon
        // crash between child acknowledgement and `.seen` append cannot launch
        // the same event twice.
        let admitted_dir = admitted_dir_for(&path);
        match std::fs::read_dir(&admitted_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let id = entry.file_name().to_string_lossy().into_owned();
                    if is_event_id(&id) && entry.path().is_file() {
                        seen.insert(id);
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        Ok(Self { path, seen })
    }

    pub fn contains(&self, id: &str) -> bool {
        self.seen.contains(id)
    }

    /// Mark an id dispatched (idempotent). Persists by appending one line under
    /// `O_APPEND` so a crash mid-run cannot corrupt the earlier record.
    pub fn insert(&mut self, id: &str) -> std::io::Result<()> {
        if !self.seen.insert(id.to_string()) {
            return Ok(()); // already recorded — no duplicate line
        }
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(f, "{id}")
    }

    /// Reclaim dedupe records for events that can no longer fire.
    ///
    /// The ONLY safe reclamation rule: an id a fresh poll still observes MUST
    /// be retained — forgetting it would re-dispatch its event — while an id
    /// that is no longer observable cannot fire again, so its log line and its
    /// `.admitted` marker are both droppable. Deriving liveness from a poll
    /// (rather than an id→source map) keeps this correct for every trigger
    /// kind, including spool-threshold and cron events that own no file.
    ///
    /// Without this, `.seen` and `.admitted` grow for the life of the daemon:
    /// every id ever dispatched stays in memory, on disk, and in the linear
    /// startup scan. Pair it with [`archive_dispatched`], which is what makes
    /// ids stop being observable in the first place.
    ///
    /// The log is rewritten atomically (temp file + rename), so a crash leaves
    /// either the old log or the new one — never a truncated one.
    pub fn retain_observable(&mut self, observable: &HashSet<String>) -> std::io::Result<usize> {
        let dropped: Vec<String> = self
            .seen
            .iter()
            .filter(|id| !observable.contains(*id))
            .cloned()
            .collect();
        if dropped.is_empty() {
            return Ok(0);
        }
        self.seen.retain(|id| observable.contains(id));
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file_name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(".seen");
        let tmp = self.path.with_file_name(format!("{file_name}.compact.tmp"));
        {
            use std::io::Write as _;
            let mut file = std::fs::File::create(&tmp)?;
            for id in &self.seen {
                writeln!(file, "{id}")?;
            }
            file.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)?;
        // The durable admission markers exist only to recover a crash window
        // for events that could still fire; a reclaimed id has none.
        let admitted = admitted_dir_for(&self.path);
        for id in &dropped {
            let _ = std::fs::remove_file(admitted.join(id));
        }
        Ok(dropped.len())
    }

    /// Number of dedupe records currently held (operational visibility).
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    /// Durable child-admission marker for `id`. The path is deterministic so
    /// restart recovery can reconcile it without any in-memory daemon state.
    pub fn admission_ack_path(&self, id: &str) -> std::io::Result<PathBuf> {
        if !is_event_id(id) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid trigger event id",
            ));
        }
        let dir = admitted_dir_for(&self.path);
        std::fs::create_dir_all(&dir)?;
        Ok(dir.join(id))
    }
}

/// Where a dispatched file-drop event is moved so it can never fire again.
/// Both this and `.admitted` are DIRECTORIES inside the watched spool, and
/// every trigger's `read_dir` is non-recursive and skips non-files, so an
/// archived event is invisible to a later poll.
fn processed_dir_for(spool: &std::path::Path) -> PathBuf {
    spool.join(".processed")
}

/// Move a dispatched file-drop event out of the watched directory into
/// `.processed/`.
///
/// This is the half of retention that bounds the SPOOL, and it is what lets
/// [`SeenStore::retain_observable`] reclaim anything: while a dispatched file
/// stays in place it remains observable, so its dedupe record must be kept
/// forever. Returns `false` for events that own no single file (spool-
/// threshold and cron fire on a condition, not a document) — those are left
/// untouched.
pub fn archive_dispatched(event: &TriggerEvent) -> std::io::Result<bool> {
    let source = std::path::Path::new(&event.source);
    if !source.is_file() {
        return Ok(false);
    }
    let Some(parent) = source.parent() else {
        return Ok(false);
    };
    let Some(name) = source.file_name() else {
        return Ok(false);
    };
    let dir = processed_dir_for(parent);
    std::fs::create_dir_all(&dir)?;
    // Same-filesystem rename: atomic, and a repeated content-addressed name
    // simply replaces its identical predecessor.
    std::fs::rename(source, dir.join(name))?;
    Ok(true)
}

/// Delete archived events older than `max_age_secs`, bounding `.processed/`.
/// `0` keeps them forever (opt-out). Returns how many were removed.
pub fn prune_processed(spool: &std::path::Path, max_age_secs: u64) -> std::io::Result<usize> {
    if max_age_secs == 0 {
        return Ok(0);
    }
    let dir = processed_dir_for(spool);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let now = std::time::SystemTime::now();
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let aged = entry
            .metadata()
            .ok()
            .and_then(|meta| meta.modified().ok())
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age.as_secs() > max_age_secs);
        if aged && std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

/// The id set a fresh poll would observe — the input to
/// [`SeenStore::retain_observable`].
pub fn observable_ids(trigger: &dyn Trigger) -> std::io::Result<HashSet<String>> {
    Ok(trigger.poll()?.into_iter().map(|event| event.id).collect())
}

fn admitted_dir_for(seen_path: &std::path::Path) -> PathBuf {
    seen_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(".admitted")
}

fn is_event_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Private sensord↔recipe admission acknowledgement protocol. Sensord sets
/// both variables in the dedicated child process; the recipe writes the marker
/// only after its exact tenant reservation and scheduler lock are held.
pub const ADMISSION_ACK_PATH_ENV: &str = "BLUT_TRIGGER_ADMISSION_ACK";
pub const ADMISSION_EVENT_ID_ENV: &str = "BLUT_TRIGGER_EVENT_ID";

/// Write the durable admission marker requested by sensord, if any. Returns
/// `Ok(false)` for ordinary human/CLI launches where the protocol is absent.
/// Both variables are required together and the path must be exactly
/// `.admitted/<event-id>`; this is not a general arbitrary-file write hook.
pub fn acknowledge_admission_from_env(job_id: &str) -> std::io::Result<bool> {
    let path = std::env::var_os(ADMISSION_ACK_PATH_ENV).map(PathBuf::from);
    let event_id = std::env::var(ADMISSION_EVENT_ID_ENV).ok();
    match (path, event_id) {
        (None, None) => Ok(false),
        (Some(path), Some(event_id)) => {
            let file_name_matches =
                path.file_name().and_then(|name| name.to_str()) == Some(event_id.as_str());
            let parent_matches = path.parent().and_then(|parent| parent.file_name())
                == Some(std::ffi::OsStr::new(".admitted"));
            let trigger_name = path
                .parent()
                .and_then(std::path::Path::parent)
                .and_then(std::path::Path::file_name)
                .and_then(std::ffi::OsStr::to_str);
            let expected_path = trigger_name
                .filter(|name| safe_component(name))
                .map(|name| spool_dir(name).join(".admitted").join(&event_id));
            if !is_event_id(&event_id)
                || !file_name_matches
                || !parent_matches
                || expected_path.as_deref() != Some(path.as_path())
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "invalid sensord admission acknowledgement path",
                ));
            }
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt as _;
                options.mode(0o600);
            }
            use std::io::Write as _;
            let mut file = options.open(path)?;
            writeln!(file, "{job_id}")?;
            file.sync_all()?;
            Ok(true)
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "incomplete sensord admission acknowledgement environment",
        )),
    }
}

/// The outcome of dispatching one fired event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// Admitted by the broker and launched.
    Admitted,
    /// Refused by broker admission (box full / would oversubscribe). NOT marked
    /// seen — a refused event retries on the next poll (backpressure), it is not
    /// silently dropped.
    Refused(String),
}

/// Launch a registered plan for a fired event. The implementation MUST route
/// the launch through broker admission before spawning anything (ADR 0094: a
/// trigger never bypasses the envelope). Object-safe so the daemon can hold a
/// boxed real dispatcher and a test can inject a counting one.
pub trait Dispatcher {
    fn dispatch(&self, plan: &str, event: &TriggerEvent) -> DispatchOutcome;
}

/// Drive one poll of `trigger`: for each fired event NOT already seen, dispatch
/// the bound `plan`; mark seen only on `Admitted` (a `Refused` event stays
/// eligible so capacity freeing up retries it). Returns the per-event outcomes
/// for this poll (already-seen events are omitted — they are the dedupe no-op).
pub fn drive_once(
    trigger: &dyn Trigger,
    plan: &str,
    seen: &mut SeenStore,
    dispatcher: &dyn Dispatcher,
) -> std::io::Result<Vec<(TriggerEvent, DispatchOutcome)>> {
    let mut results = Vec::new();
    for ev in trigger.poll()? {
        if seen.contains(&ev.id) {
            continue; // dedupe: same observation already dispatched
        }
        let outcome = dispatcher.dispatch(plan, &ev);
        if outcome == DispatchOutcome::Admitted {
            seen.insert(&ev.id)?;
        }
        results.push((ev, outcome));
    }
    Ok(results)
}

/// The production dispatcher: probe the box, resolve the plan's footprint, ask
/// broker admission, and only on `Admit` launch `blut recipe run <plan>`
/// detached. Held by `blut sensord`. The launch command is injected so the
/// daemon can point it at the cookbook binary (only it knows the recipes) and a
/// test can substitute a recorder — the admission decision itself is real.
pub struct AdmissionDispatcher {
    pub snapshot: crate::broker::probe::ResourceSnapshot,
    pub footprint: crate::broker::Footprint,
    pub floor_gib: f64,
    /// Called ONLY after admission returns `Admit`. Real impl spawns the CLI;
    /// returns Ok on a successful launch.
    pub launch: LaunchFn,
}

/// Production sensord dispatcher. Its launch callback does not return until
/// the child recipe has acquired its exact tenant reservation and scheduler
/// lock and written the durable admission marker. Consequently `Admitted`
/// means the normal recipe broker path admitted the real compiled plan, not a
/// guessed daemon-side footprint.
pub struct ChildAdmissionDispatcher {
    pub launch: LaunchFn,
}

impl Dispatcher for ChildAdmissionDispatcher {
    fn dispatch(&self, plan: &str, event: &TriggerEvent) -> DispatchOutcome {
        match (self.launch)(plan, event) {
            Ok(()) => DispatchOutcome::Admitted,
            Err(error) => {
                DispatchOutcome::Refused(format!("child admission/launch failed: {error}"))
            }
        }
    }
}

impl Dispatcher for AdmissionDispatcher {
    fn dispatch(&self, plan: &str, event: &TriggerEvent) -> DispatchOutcome {
        match crate::broker::admission::decide(&self.snapshot, &self.footprint, self.floor_gib) {
            crate::broker::admission::AdmitDecision::Admit { .. } => {
                match (self.launch)(plan, event) {
                    Ok(()) => DispatchOutcome::Admitted,
                    Err(e) => DispatchOutcome::Refused(format!("launch failed: {e}")),
                }
            }
            crate::broker::admission::AdmitDecision::Refuse { reason } => {
                DispatchOutcome::Refused(reason)
            }
        }
    }
}

/// Resolve the default spool root for a trigger's file-drop directory
/// (`$BLUT_EVENTS_DIR` or `~/.blut/events/<trigger>`). The webhook ingress in
/// `blut-web` writes here; the daemon watches it.
pub fn spool_dir(trigger: &str) -> PathBuf {
    if let Some(root) = std::env::var_os("BLUT_EVENTS_DIR") {
        return PathBuf::from(root).join(trigger);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".blut").join("events").join(trigger)
}

/// Path of the daemon's dedupe log for a trigger (beside the spool root).
pub fn seen_path(trigger: &str) -> PathBuf {
    spool_dir(trigger).join(".seen")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &std::path::Path, body: &str) {
        std::fs::write(path, body).unwrap();
    }

    struct CountingDispatcher {
        calls: std::cell::RefCell<usize>,
        admit: bool,
    }
    impl Dispatcher for CountingDispatcher {
        fn dispatch(&self, _plan: &str, _event: &TriggerEvent) -> DispatchOutcome {
            *self.calls.borrow_mut() += 1;
            if self.admit {
                DispatchOutcome::Admitted
            } else {
                DispatchOutcome::Refused("test-full".into())
            }
        }
    }

    #[test]
    fn file_drop_dedupes_identical_observations() {
        let td = tempfile::tempdir().unwrap();
        let watch = td.path().join("spool");
        std::fs::create_dir_all(&watch).unwrap();
        touch(&watch.join("a.json"), "{}");
        let trig = FileDropTrigger::new("t", &watch, Some("json".into()));
        let mut seen = SeenStore::load(td.path().join(".seen")).unwrap();
        let disp = CountingDispatcher {
            calls: std::cell::RefCell::new(0),
            admit: true,
        };
        // First drive: one event, dispatched once.
        let r1 = drive_once(&trig, "plan", &mut seen, &disp).unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].1, DispatchOutcome::Admitted);
        // Second drive over the SAME file: deduped, zero new dispatches.
        let r2 = drive_once(&trig, "plan", &mut seen, &disp).unwrap();
        assert!(r2.is_empty());
        assert_eq!(*disp.calls.borrow(), 1);

        // A webhook replay may rewrite/touch its spool path. Dedupe is based
        // on content, not mutable mtime, so identical bytes remain one event.
        touch(&watch.join("a.json"), "{}");
        let r3 = drive_once(&trig, "plan", &mut seen, &disp).unwrap();
        assert!(r3.is_empty());
        assert_eq!(*disp.calls.borrow(), 1);
    }

    #[test]
    fn refused_event_is_not_marked_seen_and_retries() {
        let td = tempfile::tempdir().unwrap();
        let watch = td.path().join("spool");
        std::fs::create_dir_all(&watch).unwrap();
        touch(&watch.join("a.json"), "{}");
        let trig = FileDropTrigger::new("t", &watch, Some("json".into()));
        let mut seen = SeenStore::load(td.path().join(".seen")).unwrap();
        let disp = CountingDispatcher {
            calls: std::cell::RefCell::new(0),
            admit: false, // box full
        };
        let r1 = drive_once(&trig, "plan", &mut seen, &disp).unwrap();
        assert!(matches!(r1[0].1, DispatchOutcome::Refused(_)));
        // A refused event stays eligible — it dispatches again next poll.
        let r2 = drive_once(&trig, "plan", &mut seen, &disp).unwrap();
        assert_eq!(r2.len(), 1);
        assert_eq!(*disp.calls.borrow(), 2);
    }

    #[test]
    fn seen_store_persists_across_reload() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("s").join(".seen");
        let mut s = SeenStore::load(&p).unwrap();
        s.insert("abc").unwrap();
        s.insert("abc").unwrap(); // idempotent
        let reloaded = SeenStore::load(&p).unwrap();
        assert!(reloaded.contains("abc"));
        assert!(!reloaded.contains("xyz"));
    }

    #[test]
    fn admission_ack_recovers_crash_window() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("spool").join(".seen");
        let s = SeenStore::load(&p).unwrap();
        let id = "a".repeat(64);
        let ack = s.admission_ack_path(&id).unwrap();
        std::fs::write(ack, "job-1\n").unwrap();

        let recovered = SeenStore::load(&p).unwrap();
        assert!(recovered.contains(&id));
    }

    #[test]
    fn admission_ack_path_is_not_an_arbitrary_write_primitive() {
        let td = tempfile::tempdir().unwrap();
        let id = "b".repeat(64);
        // SAFETY (unit test): cargo runs this module's environment-mutating
        // test without any production sensord child in the same process.
        unsafe {
            std::env::set_var(ADMISSION_EVENT_ID_ENV, &id);
            std::env::set_var(ADMISSION_ACK_PATH_ENV, td.path().join("outside"));
        }
        assert!(acknowledge_admission_from_env("job-1").is_err());
        assert!(!td.path().join("outside").exists());
        let outside = td.path().join("outside").join(".admitted").join(&id);
        unsafe {
            std::env::set_var("BLUT_EVENTS_DIR", td.path().join("events"));
            std::env::set_var(ADMISSION_ACK_PATH_ENV, &outside);
        }
        assert!(acknowledge_admission_from_env("job-1").is_err());
        assert!(!outside.exists());
        unsafe {
            std::env::remove_var("BLUT_EVENTS_DIR");
            std::env::remove_var(ADMISSION_EVENT_ID_ENV);
            std::env::remove_var(ADMISSION_ACK_PATH_ENV);
        }
    }

    #[test]
    fn shared_config_carries_webhook_custody() {
        let cfg = TriggerConfig::parse(
            r#"
webhook_max_skew_secs = 45

[[trigger]]
name = "clinical-hook"
plan = "registry://plan@prod"
tenant = "clinical/prod"
data_class = "Restricted"
webhook_secret = { name = "BLUT_HOOK_KEY" }
"#,
        )
        .unwrap();
        cfg.validate().unwrap();
        let binding = cfg.binding("clinical-hook").unwrap();
        assert_eq!(binding.tenant, "clinical/prod");
        assert_eq!(binding.data_class, DataClass::Restricted);
        assert_eq!(
            binding.webhook_secret.as_ref().unwrap().name,
            "BLUT_HOOK_KEY"
        );
        assert_eq!(cfg.webhook_max_skew_secs, 45);
        assert_eq!(cfg.admission_timeout_secs, 60);
    }

    #[test]
    fn trigger_names_cannot_escape_the_event_root() {
        let cfg = TriggerConfig::parse(
            r#"
[[trigger]]
name = ".."
plan = "demo"
"#,
        )
        .unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn extension_filter_excludes_nonmatching() {
        let td = tempfile::tempdir().unwrap();
        touch(&td.path().join("keep.json"), "{}");
        touch(&td.path().join("skip.tmp"), "x");
        let trig = FileDropTrigger::new("t", td.path(), Some("json".into()));
        let evs = trig.poll().unwrap();
        assert_eq!(evs.len(), 1);
        assert!(evs[0].source.ends_with("keep.json"));
    }

    #[test]
    fn missing_spool_dir_is_empty_not_error() {
        let td = tempfile::tempdir().unwrap();
        let trig = FileDropTrigger::new("t", td.path().join("nope"), None);
        assert!(trig.poll().unwrap().is_empty());
    }

    #[test]
    fn spool_threshold_refires_only_after_snapshot_changes() {
        let td = tempfile::tempdir().unwrap();
        let trigger = SpoolThresholdTrigger::new(
            "queue-full",
            td.path(),
            SpoolMetric::Files,
            SpoolDirection::AtLeast,
            2,
        );
        touch(&td.path().join("a"), "1");
        assert!(trigger.poll().unwrap().is_empty());
        touch(&td.path().join("b"), "2");
        let first = trigger.poll().unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(trigger.poll().unwrap()[0].id, first[0].id);
        touch(&td.path().join("c"), "3");
        assert_ne!(trigger.poll().unwrap()[0].id, first[0].id);
    }

    #[test]
    fn cron_uses_scheduled_instant_as_stable_event_identity() {
        use chrono::TimeZone as _;
        let trigger = CronTrigger::new(
            "minute",
            "0 * * * * * *",
            std::time::Duration::from_secs(10),
        )
        .unwrap();
        let now = chrono::Utc
            .with_ymd_and_hms(2026, 7, 20, 12, 34, 5)
            .unwrap();
        let first = trigger.poll_at(now);
        assert_eq!(first.len(), 1);
        assert_eq!(trigger.poll_at(now)[0].id, first[0].id);
        let next = trigger.poll_at(now + chrono::Duration::minutes(1));
        assert_ne!(next[0].id, first[0].id);
        assert!(
            trigger
                .poll_at(now + chrono::Duration::seconds(11))
                .is_empty()
        );
    }
}
