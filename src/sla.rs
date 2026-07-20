// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! SLA rules (ADR 0094) — declarative "this run should finish by T / data must
//! be no older than N", evaluated as a PURE read over run observations (the
//! jobs store + lineage timestamps). Charter-clean: `blut sla check` reads and
//! writes only `sla.jsonl`; it never mutates a run, opens a socket, or launches
//! anything. A breach is itself an event the notify sidecar (or a re-trigger)
//! consumes.
//!
//! The breach RECORD is the keystone [`blut_types::sla::SlaBreach`] so the
//! producer here and the `blut-notify` consumer share one definition. Every
//! `summary` this module writes is a PHI-FREE one-liner (job ids + seconds),
//! never patient content — a `restricted` breach can still notify a local sink.

pub use blut_types::sla::{SlaBreach, SlaKind};
use blut_types::trust::DataClass;

/// One declarative SLA rule (a row of the rule file). A rule with no bound is a
/// no-op; a rule may carry several bounds (each evaluated independently).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlaRule {
    pub name: String,
    /// Match only runs of this recipe (`None` = any recipe).
    #[serde(default)]
    pub recipe: Option<String>,
    /// Max wall-clock runtime in seconds before a `MaxRuntime` breach.
    #[serde(default)]
    pub max_runtime_secs: Option<i64>,
    /// Absolute wall-clock deadline (unix seconds); a still-running job past it
    /// is a `Deadline` breach.
    #[serde(default)]
    pub deadline_unix: Option<i64>,
    /// Freshness window in seconds; data older than this is a `Freshness`
    /// breach.
    #[serde(default)]
    pub freshness_secs: Option<i64>,
}

/// A run's observed state — what `sla check` reads from the jobs store + lineage
/// and feeds to [`evaluate`]. Kept separate from the DB rows so the evaluation
/// logic is pure and unit-testable without a live store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunObservation {
    pub job_id: String,
    pub recipe: String,
    pub tenant: String,
    pub data_class: DataClass,
    pub started_unix: i64,
    /// `None` = still running.
    pub ended_unix: Option<i64>,
    /// Age of the run's freshest data dependency in seconds (`None` = unknown ⇒
    /// freshness not evaluated, never a false breach).
    pub data_age_secs: Option<i64>,
}

fn breach(
    rule: &SlaRule,
    kind: SlaKind,
    obs: &RunObservation,
    observed: i64,
    limit: i64,
    now: i64,
) -> SlaBreach {
    // PHI-FREE summary: job id + seconds only. The tenant is a slug, not patient
    // content; no run payload is read here.
    let summary = format!(
        "{} breach on job {} (rule '{}'): {}s vs limit {}s",
        kind.as_str(),
        obs.job_id,
        rule.name,
        observed,
        limit,
    );
    SlaBreach {
        rule: rule.name.clone(),
        kind,
        job_id: obs.job_id.clone(),
        tenant: obs.tenant.clone(),
        data_class: obs.data_class,
        observed_secs: observed,
        limit_secs: limit,
        summary,
        detected_unix: now,
    }
}

/// Evaluate every rule against every observation at wall-clock `now` (unix s),
/// returning the breaches. Pure: no I/O, deterministic in its inputs.
pub fn evaluate(rules: &[SlaRule], runs: &[RunObservation], now: i64) -> Vec<SlaBreach> {
    let mut out = Vec::new();
    for obs in runs {
        for rule in rules {
            if let Some(want) = &rule.recipe
                && want != &obs.recipe
            {
                continue; // rule scoped to a different recipe
            }
            // MaxRuntime: elapsed = (end or now) − start.
            if let Some(max) = rule.max_runtime_secs {
                let elapsed = obs.ended_unix.unwrap_or(now) - obs.started_unix;
                if elapsed > max {
                    out.push(breach(rule, SlaKind::MaxRuntime, obs, elapsed, max, now));
                }
            }
            // Deadline: a run STILL running past the wall-clock deadline.
            if let Some(deadline) = rule.deadline_unix
                && obs.ended_unix.is_none()
                && now > deadline
            {
                out.push(breach(rule, SlaKind::Deadline, obs, now, deadline, now));
            }
            // Freshness: the run's data is older than the allowed window.
            if let Some(window) = rule.freshness_secs
                && let Some(age) = obs.data_age_secs
                && age > window
            {
                out.push(breach(rule, SlaKind::Freshness, obs, age, window, now));
            }
        }
    }
    out
}

/// Append breaches to `sla.jsonl` (one JSON line each, `O_APPEND`). Returns the
/// number of rows written. Creates the parent dir on demand.
pub fn append_breaches(path: &std::path::Path, breaches: &[SlaBreach]) -> std::io::Result<usize> {
    if breaches.is_empty() {
        return Ok(0);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for b in breaches {
        writeln!(f, "{}", b.to_line())?;
    }
    Ok(breaches.len())
}

/// A `sla.toml` rule file: `[[rule]]` tables → [`SlaRule`]s.
#[derive(Clone, Debug, Default, serde::Deserialize)]
pub struct RulesFile {
    #[serde(default)]
    pub rule: Vec<SlaRule>,
}

/// Load SLA rules from a TOML file (`[[rule]]` entries). Absent file ⇒ no rules
/// (SLA checking is opt-in, not a hard error).
pub fn load_rules(path: &std::path::Path) -> std::io::Result<Vec<SlaRule>> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let parsed: RulesFile = toml::from_str(&text).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("parse {path:?}: {e}"))
    })?;
    Ok(parsed.rule)
}

/// Build a [`RunObservation`] from a lineage run row (`None` if the run has no
/// start time — SLA timing is undefined without it, never a false breach). The
/// data class is derived from the tenant: a `restricted` tenant ⇒ `Restricted`,
/// else `Internal`. Freshness input is left `None` (deferred — it needs
/// per-artifact timestamps).
pub fn observation_from_run(run: &crate::lineage_db::RunRow) -> Option<RunObservation> {
    let started = run.started_unix?;
    let tenant = if run.tenant.is_empty() {
        "default"
    } else {
        run.tenant.as_str()
    };
    let data_class = if crate::tenant::Tenant::parse(tenant).is_some_and(|t| t.is_restricted()) {
        DataClass::Restricted
    } else {
        DataClass::Internal
    };
    Some(RunObservation {
        job_id: run.job_id.clone(),
        recipe: run.recipe.clone(),
        tenant: tenant.to_string(),
        data_class,
        started_unix: started,
        ended_unix: run.ended_unix,
        data_age_secs: None,
    })
}

/// Current wall-clock unix seconds (SLA evaluation's `now`).
pub fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Default sla.jsonl location (`$BLUT_SLA_PATH` or `~/.blut/sla.jsonl`).
pub fn default_sla_path() -> std::path::PathBuf {
    if let Some(p) = std::env::var_os("BLUT_SLA_PATH") {
        return std::path::PathBuf::from(p);
    }
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_default();
    home.join(".blut").join("sla.jsonl")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(
        recipe: &str,
        tenant: &str,
        dc: DataClass,
        started: i64,
        ended: Option<i64>,
    ) -> RunObservation {
        RunObservation {
            job_id: "job-abc".into(),
            recipe: recipe.into(),
            tenant: tenant.into(),
            data_class: dc,
            started_unix: started,
            ended_unix: ended,
            data_age_secs: None,
        }
    }

    #[test]
    fn max_runtime_breach_is_detected_and_redacted() {
        let rules = vec![SlaRule {
            name: "cap-1h".into(),
            recipe: Some("train".into()),
            max_runtime_secs: Some(3600),
            deadline_unix: None,
            freshness_secs: None,
        }];
        // Ran 4200s (started at 0, still running at now=4200) vs a 3600 cap.
        let runs = vec![obs(
            "train",
            "clinical/prod",
            DataClass::Restricted,
            0,
            None,
        )];
        let breaches = evaluate(&rules, &runs, 4200);
        assert_eq!(breaches.len(), 1);
        assert_eq!(breaches[0].kind, SlaKind::MaxRuntime);
        assert_eq!(breaches[0].observed_secs, 4200);
        assert_eq!(breaches[0].limit_secs, 3600);
        // PHI-free summary: no patient content, job-id + seconds only.
        assert!(breaches[0].summary.contains("job-abc"));
        assert!(!breaches[0].summary.contains("patient"));
    }

    #[test]
    fn recipe_scope_and_finished_run_within_cap_do_not_breach() {
        let rules = vec![SlaRule {
            name: "cap".into(),
            recipe: Some("train".into()),
            max_runtime_secs: Some(3600),
            deadline_unix: None,
            freshness_secs: None,
        }];
        // Different recipe → no breach.
        let other = vec![obs("eval", "shared", DataClass::Internal, 0, Some(9000))];
        assert!(evaluate(&rules, &other, 9000).is_empty());
        // Right recipe, finished within cap → no breach.
        let ok = vec![obs("train", "shared", DataClass::Internal, 0, Some(1800))];
        assert!(evaluate(&rules, &ok, 5000).is_empty());
    }

    #[test]
    fn deadline_and_freshness_breaches() {
        let rules = vec![SlaRule {
            name: "r".into(),
            recipe: None,
            max_runtime_secs: None,
            deadline_unix: Some(1000),
            freshness_secs: Some(60),
        }];
        let mut o = obs("any", "shared", DataClass::Public, 0, None);
        o.data_age_secs = Some(120);
        let breaches = evaluate(&rules, &[o], 2000);
        // Both a Deadline (running past 1000) and a Freshness (120 > 60) breach.
        assert_eq!(breaches.len(), 2);
        assert!(breaches.iter().any(|b| b.kind == SlaKind::Deadline));
        assert!(breaches.iter().any(|b| b.kind == SlaKind::Freshness));
    }

    #[test]
    fn append_writes_one_line_per_breach() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("sub").join("sla.jsonl");
        let rules = vec![SlaRule {
            name: "cap".into(),
            recipe: None,
            max_runtime_secs: Some(10),
            deadline_unix: None,
            freshness_secs: None,
        }];
        let runs = vec![obs("t", "shared", DataClass::Public, 0, Some(100))];
        let breaches = evaluate(&rules, &runs, 100);
        let n = append_breaches(&path, &breaches).unwrap();
        assert_eq!(n, 1);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 1);
        assert!(SlaBreach::from_line(text.lines().next().unwrap()).is_ok());
    }
}
