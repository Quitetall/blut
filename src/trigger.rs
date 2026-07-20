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
/// extension) — the file-drop / spool sensor. The event id folds path + size +
/// mtime, so re-creating a file with new content (new mtime) re-fires while an
/// unchanged file stays deduped. A missing directory is not an error — it is an
/// empty observation (the spool simply has not been created yet).
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
            if let Some(want) = &self.ext
                && p.extension().and_then(|x| x.to_str()) != Some(want.as_str())
            {
                continue;
            }
            entries.push(p);
        }
        entries.sort(); // deterministic event order regardless of readdir order
        for p in entries {
            let md = std::fs::metadata(&p)?;
            let mtime = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path_s = p.to_string_lossy();
            let id = event_id(&[&path_s, &md.len().to_string(), &mtime.to_string()]);
            out.push(TriggerEvent {
                id,
                source: path_s.into_owned(),
            });
        }
        Ok(out)
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
        let seen = match std::fs::read_to_string(&path) {
            Ok(text) => text.lines().map(str::to_string).collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => HashSet::new(),
            Err(e) => return Err(e),
        };
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
}
