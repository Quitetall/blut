// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Structured error infrastructure for cookbook domains.
//!
//! BLUT's error library. Cookbooks register error domains (collections
//! of error codes) and produce [`StageFailure`] values that carry
//! structured context through the error chain. The executor extracts
//! `StageFailure` from `anyhow::Error` via downcasting and stores it
//! in lineage + `status.jsonl`.
//!
//! # Design
//!
//! - [`ErrorDomain`] — trait for cookbook-level error code catalogs
//! - [`StageFailure`] — structured failure context (code, severity,
//!   key-value pairs). Implements `std::error::Error` so it wraps
//!   cleanly into `anyhow::Error` → `StageError::Backend(...)`.
//! - [`Severity`] — Critical / Major / Minor classification
//!
//! Cookbooks emit `StageFailure` from their stages. The executor
//! downcasts `StageError::Backend(anyhow)` to extract the structured
//! data and stores it in `StageEvent::StageFailed`.

use std::fmt;

use serde::{Deserialize, Serialize};

// ── Severity ───────────────────────────────────────────────────────

/// Failure severity — how bad is this?
///
/// Critical = data loss, corruption, safety violation.
/// Major = wrong output, failed invariant.
/// Minor = perf regression, edge case, cosmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Critical,
    Major,
    Minor,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::Major => write!(f, "MAJOR"),
            Self::Minor => write!(f, "MINOR"),
        }
    }
}

// ── FaultOrigin ────────────────────────────────────────────────────

/// Who is at fault for this failure — engine, cookbook glue, or the
/// outside world?
///
/// `Engine` = a bug in BLUT's own executor/framework code.
/// `Cookbook` = a bug (or unhandled edge case) in cookbook glue code.
/// `External` = something outside BLUT's control (disk full, network,
/// upstream service, malformed user input).
///
/// Defaults to `Cookbook` in [`StageFailure::new`] — most failures
/// genuinely are cookbook glue; `Engine`/`External` are the minority
/// that need an explicit `.origin(...)` override.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultOrigin {
    Engine,
    Cookbook,
    External,
}

// ── ErrorDomain ────────────────────────────────────────────────────

/// A cookbook's error code catalog.
///
/// Each cookbook registers its domain with a name and a list of
/// `(code, description)` pairs. This is metadata — it doesn't affect
/// execution — but it enables `blut errors list` to show all known
/// error codes across cookbooks.
pub trait ErrorDomain: Send + Sync + 'static {
    /// Domain name (e.g. "eagle", "lamquant", "lamu").
    const NAME: &'static str;

    /// Error codes this domain defines: `(code, description)`.
    const CODES: &[(&'static str, &'static str)];
}

// ── StageFailure ───────────────────────────────────────────────────

/// Structured failure context emitted by cookbook stages.
///
/// This is the structured counterpart to `StageError::Backend(String)`.
/// Cookbooks create a `StageFailure`, convert it to `anyhow::Error`,
/// and return it as `StageError::Backend(...)`. The executor downcasts
/// to extract the structured data for lineage storage.
///
/// # Example
///
/// ```ignore
/// use blut::framework::error_domain::{StageFailure, Severity};
///
/// return Err(StageError::Backend(
///     StageFailure::new("EAGLE_ROUNDTRIP", "eagle")
///         .severity(Severity::Major)
///         .stage("eagle_decode")
///         .context("ch", "4")
///         .context("len", "4096")
///         .into_error("decode(encode(x)) != x: first diff at sample 1847"),
/// ));
///
/// // Most failures are cookbook glue (the default). Override when the
/// // fault genuinely lies outside BLUT's control:
/// use blut::framework::error_domain::FaultOrigin;
///
/// return Err(StageError::Backend(
///     StageFailure::new("EAGLE_UPSTREAM_TIMEOUT", "eagle")
///         .severity(Severity::Major)
///         .origin(FaultOrigin::External)
///         .stage("eagle_fetch")
///         .into_error("upstream service timed out after 30s"),
/// ));
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageFailure {
    /// Error code — machine-readable, greppable (e.g. "EAGLE_ROUNDTRIP").
    pub code: String,
    /// Domain that owns this code (e.g. "eagle", "lamquant").
    pub domain: String,
    /// Stage that produced this failure.
    pub stage: Option<String>,
    /// Severity classification.
    pub severity: Severity,
    /// Who's at fault — engine, cookbook glue, or external. Defaults
    /// to `Cookbook` in [`StageFailure::new`]; override with
    /// [`StageFailure::origin`] for the `Engine`/`External` minority.
    pub origin: FaultOrigin,
    /// Recipe course, if known (e.g. "train", "eval"). Plain `String`
    /// — `error_domain` sits below `recipes` in the module layering
    /// and must not depend on `recipes::recipe::Course`.
    pub course: Option<String>,
    /// Recipe name, if known (e.g. "train_joint").
    pub recipe: Option<String>,
    /// Ingredient (stage/component) implicated, if known.
    pub ingredient: Option<String>,
    /// Structured key-value context pairs.
    pub context: Vec<(String, String)>,
    /// Human-readable summary.
    pub message: String,
}

impl StageFailure {
    /// Create a new failure with code and domain.
    ///
    /// `origin` defaults to [`FaultOrigin::Cookbook`] — most failures
    /// genuinely are cookbook glue. Use [`StageFailure::origin`] to
    /// override for the `Engine`/`External` minority.
    pub fn new(code: impl Into<String>, domain: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            domain: domain.into(),
            stage: None,
            severity: Severity::Major,
            origin: FaultOrigin::Cookbook,
            course: None,
            recipe: None,
            ingredient: None,
            context: Vec::new(),
            message: String::new(),
        }
    }

    /// Set the severity.
    pub fn severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Set the stage name.
    pub fn stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    /// Override the fault origin (defaults to [`FaultOrigin::Cookbook`]).
    pub fn origin(mut self, origin: FaultOrigin) -> Self {
        self.origin = origin;
        self
    }

    /// Add a context key-value pair.
    pub fn context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }

    /// Set the human-readable message.
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Convert into an `anyhow::Error` for wrapping in `StageError::Backend(...)`.
    pub fn into_error(self, message: impl Into<String>) -> anyhow::Error {
        let mut this = self;
        this.message = message.into();
        anyhow::Error::from(this)
    }

    /// Try to extract a `StageFailure` from an `anyhow::Error`.
    ///
    /// Walks the error chain looking for a `StageFailure`. Returns
    /// `None` if the chain doesn't contain one.
    pub fn try_extract(err: &anyhow::Error) -> Option<&StageFailure> {
        err.downcast_ref::<StageFailure>()
    }
}

impl fmt::Display for StageFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.severity, self.code)?;
        if let Some(stage) = &self.stage {
            write!(f, " @{stage}")?;
        }
        if !self.context.is_empty() {
            write!(f, " [")?;
            for (i, (k, v)) in self.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{k}={v}")?;
            }
            write!(f, "]")?;
        }
        if !self.message.is_empty() {
            write!(f, ": {}", self.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for StageFailure {}

// ── Serialized form for StageEvent ─────────────────────────────────

/// Serializable failure summary stored in `StageEvent::StageFailed`.
///
/// This is a flattened view of `StageFailure` — the executor extracts
/// it from the error chain and stores it alongside the string error
/// for backwards compatibility.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailureSummary {
    pub code: String,
    pub domain: String,
    /// Stage that produced this failure. Mirrors `StageFailure::stage`
    /// (previously missing here — a pre-existing bug: the flattened
    /// summary persisted to `status.jsonl` silently dropped which
    /// stage failed).
    pub stage: Option<String>,
    pub severity: Severity,
    pub origin: FaultOrigin,
    pub course: Option<String>,
    pub recipe: Option<String>,
    pub ingredient: Option<String>,
    pub context: Vec<(String, String)>,
    pub message: String,
}

impl From<&StageFailure> for FailureSummary {
    fn from(f: &StageFailure) -> Self {
        Self {
            code: f.code.clone(),
            domain: f.domain.clone(),
            stage: f.stage.clone(),
            severity: f.severity,
            origin: f.origin,
            course: f.course.clone(),
            recipe: f.recipe.clone(),
            ingredient: f.ingredient.clone(),
            context: f.context.clone(),
            message: f.message.clone(),
        }
    }
}

/// Try to extract a `FailureSummary` from a `StageError`.
///
/// Walks the error chain looking for a `StageFailure` in `Backend(anyhow)`
/// variants. Returns `None` for non-Backend variants or if the anyhow
/// chain doesn't contain a `StageFailure`.
pub fn extract_failure_summary(err: &crate::framework::error::StageError) -> Option<FailureSummary> {
    match err {
        crate::framework::error::StageError::Backend(anyhow_err) => {
            StageFailure::try_extract(anyhow_err).map(FailureSummary::from)
        }
        _ => None,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_failure_display_includes_all_fields() {
        let f = StageFailure::new("E_ROUNDTRIP", "eagle")
            .severity(Severity::Major)
            .stage("eagle_decode")
            .context("ch", "4")
            .context("len", "4096")
            .message("first diff at sample 1847");
        let msg = format!("{f}");
        assert!(msg.contains("MAJOR"));
        assert!(msg.contains("E_ROUNDTRIP"));
        assert!(msg.contains("eagle_decode"));
        assert!(msg.contains("ch=4"));
        assert!(msg.contains("len=4096"));
        assert!(msg.contains("first diff at sample 1847"));
        // origin not overridden — defaults to Cookbook.
        assert_eq!(f.origin, FaultOrigin::Cookbook);
        assert_eq!(f.course, None);
        assert_eq!(f.recipe, None);
        assert_eq!(f.ingredient, None);
    }

    #[test]
    fn stage_failure_roundtrips_through_anyhow() {
        let f = StageFailure::new("E_REJECT", "eagle")
            .severity(Severity::Critical)
            .context("cut_at", "42");
        let err = f.into_error("accepted truncated blob");
        let extracted = StageFailure::try_extract(&err);
        assert!(extracted.is_some());
        let f = extracted.unwrap();
        assert_eq!(f.code, "E_REJECT");
        assert_eq!(f.domain, "eagle");
        assert_eq!(f.severity, Severity::Critical);
        assert_eq!(f.origin, FaultOrigin::Cookbook);
        assert_eq!(f.context, vec![("cut_at".into(), "42".into())]);
        assert_eq!(f.message, "accepted truncated blob");
    }

    #[test]
    fn try_extract_returns_none_for_plain_error() {
        let err = anyhow::anyhow!("not a stage failure");
        assert!(StageFailure::try_extract(&err).is_none());
    }

    #[test]
    fn failure_summary_from_stage_failure() {
        let f = StageFailure::new("E_THRESHOLD", "lamquant")
            .severity(Severity::Minor)
            .stage("cr_check")
            .context("cr", "0.79")
            .context("floor", "0.80")
            .message("CR below floor");
        let summary = FailureSummary::from(&f);
        assert_eq!(summary.code, "E_THRESHOLD");
        assert_eq!(summary.domain, "lamquant");
        assert_eq!(summary.stage, Some("cr_check".to_string()));
        assert_eq!(summary.severity, Severity::Minor);
        assert_eq!(summary.origin, FaultOrigin::Cookbook);
        assert_eq!(summary.course, None);
        assert_eq!(summary.recipe, None);
        assert_eq!(summary.ingredient, None);
        assert_eq!(summary.context.len(), 2);
    }

    #[test]
    fn stage_failure_new_defaults_origin_to_cookbook() {
        let f = StageFailure::new("E_DEFAULT_ORIGIN", "eagle");
        assert_eq!(f.origin, FaultOrigin::Cookbook);
    }

    #[test]
    fn stage_failure_origin_builder_overrides_default() {
        let f = StageFailure::new("E_EXTERNAL", "eagle").origin(FaultOrigin::External);
        assert_eq!(f.origin, FaultOrigin::External);

        let f = StageFailure::new("E_ENGINE", "eagle").origin(FaultOrigin::Engine);
        assert_eq!(f.origin, FaultOrigin::Engine);
    }

    /// Regression pin for the `FailureSummary` `stage`-dropping bug:
    /// `FailureSummary::from` must carry `stage` through. Reverting the
    /// `stage: f.stage.clone()` line in the `From` impl (or dropping
    /// `stage` back out of `FailureSummary`) makes this fail.
    #[test]
    fn failure_summary_from_carries_stage_through() {
        let f = StageFailure::new("E_STAGE_BUG", "eagle").stage("eagle_encode");
        let summary = FailureSummary::from(&f);
        assert_eq!(summary.stage, Some("eagle_encode".to_string()));
    }
}
