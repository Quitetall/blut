// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Built-in `checks` cookbook (ADR 0091): data-quality / assertion PASSTHROUGH
//! stages.
//!
//! A check stage takes an input artifact, validates it, and:
//!   * on PASS — returns the input BYTE-IDENTICAL (same content hash), so a
//!     downstream node sees exactly what it would without the guard, and the
//!     content-addressed cache keys (ADR 0078) are unchanged;
//!   * on a BLOCK-severity breach — returns `StageError::Backend(StageFailure)`
//!     in the `checks` domain, FAILING the node so the executor never runs the
//!     downstream (fail-closed data-quality gate — the whole point);
//!   * on a WARN-severity breach — emits a `StageStep` breach event (recorded in
//!     lineage / `status.jsonl`) and PASSES THROUGH, so the DAG continues but the
//!     breach is on the record.
//!
//! Charter: checks are ORDINARY cookbook stages — no engine change. The block
//! path rides the existing `StageError` → downstream-skip machinery; the
//! passthrough output keeps cache keys identical. This is the standard `checks`
//! cookbook the way `p2p-smoke` is a built-in cookbook.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::framework::artifact::{Artifact, ContentHash};
use crate::framework::cookbook::{Cookbook, Registry};
use crate::framework::error::StageError;
use crate::framework::error_domain::{ErrorDomain, ErrorDomainDef, Severity, StageFailure};
use crate::framework::resource::Resource;
use crate::framework::stage::{ErasedStageCtor, Stage, StageContext};
use crate::framework::status::StageEvent;
use crate::recipes::recipe::RecipeDef;

// ── error domain ───────────────────────────────────────────────────

/// The `checks` cookbook's data-quality error catalog (ADR 0091). The ADR refers
/// to this loosely as "ErrorDomain::DataQuality"; `ErrorDomain` is a per-cookbook
/// catalog (a trait, not an enum), so it materialises as the `checks` domain
/// whose codes are all data-quality breaches.
pub struct ChecksErrorDomain;

impl ErrorDomain for ChecksErrorDomain {
    const NAME: &'static str = "checks";
    const CODES: &[(&'static str, &'static str)] = &[
        (
            "DATA_QUALITY_MIN_ROWS",
            "jsonl row count below the required minimum",
        ),
        (
            "DATA_QUALITY_MAX_ROWS",
            "jsonl row count above the allowed maximum",
        ),
        (
            "ASSERT_FAILED",
            "a numeric assertion predicate did not hold",
        ),
    ];
}

crate::register_error_domain!(ChecksErrorDomain);

/// The domain string every check `StageFailure` carries.
const DOMAIN: &str = "checks";

// ── breach policy ──────────────────────────────────────────────────

/// How a check treats a breach. `Block` fails the node (fail-closed — the
/// executor skips every downstream); `Warn` records a lineage breach event and
/// passes through so the DAG continues.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum OnBreach {
    #[default]
    Block,
    Warn,
}

/// Apply the breach policy. `Block` → a `checks`-domain `StageFailure` (fails the
/// node, downstream skipped). `Warn` → emit a lineage breach event + `Ok(())` to
/// continue. Returns `Err` only for the fail-closed path.
fn handle_breach(
    ctx: &StageContext,
    stage: &'static str,
    code: &'static str,
    on_breach: OnBreach,
    detail: String,
    context: &[(&'static str, String)],
) -> Result<(), StageError> {
    match on_breach {
        OnBreach::Block => {
            let mut f = StageFailure::new(code, DOMAIN)
                .severity(Severity::Major)
                .stage(stage);
            for (k, v) in context {
                f = f.context(*k, v.clone());
            }
            Err(StageError::Backend(f.into_error(detail)))
        }
        OnBreach::Warn => {
            tracing::warn!(target: "checks", stage, code, %detail, "data-quality WARN breach (continuing)");
            // Best-effort lineage breach row; a dropped receiver is not fatal
            // (a check with no subscribers still passes through correctly).
            let _ = ctx.status_tx.send(StageEvent::StageStep {
                node_idx: ctx.node_idx,
                stage_name: stage.to_string(),
                update: serde_json::json!({
                    "checks_breach": {
                        "code": code,
                        "severity": "warn",
                        "detail": detail,
                        "context": context
                            .iter()
                            .map(|(k, v)| (k.to_string(), v.clone()))
                            .collect::<Vec<_>>(),
                    }
                }),
            });
            Ok(())
        }
    }
}

// ── artifact ───────────────────────────────────────────────────────

/// A JSONL file artifact — one UTF-8 file, one JSON value per line. File-backed
/// (the bytes live on disk; this struct is the handle), like every BLUT artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonlArtifact {
    pub path: PathBuf,
    pub content_hash: ContentHash,
}

impl Artifact for JsonlArtifact {
    const KIND: &'static str = "checks.jsonl";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
    fn primary_path(&self) -> &Path {
        &self.path
    }
}

// ── check_jsonl ────────────────────────────────────────────────────

/// Args for [`CheckJsonl`]: row-count bounds + breach policy.
#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CheckJsonlArgs {
    /// Minimum row count (inclusive). `None` = no lower bound.
    #[serde(default)]
    pub min_rows: Option<u64>,
    /// Maximum row count (inclusive). `None` = no upper bound.
    #[serde(default)]
    pub max_rows: Option<u64>,
    /// Breach policy (default: `block` — fail-closed).
    #[serde(default)]
    pub on_breach: OnBreach,
}

/// Stage name of the JSONL row-count guard.
pub const CHECK_JSONL: &str = "check_jsonl";

/// Count non-empty lines of the input JSONL and enforce `min_rows`/`max_rows`,
/// then pass the input through unchanged (same path + content hash).
pub struct CheckJsonl;

#[async_trait]
impl Stage for CheckJsonl {
    const NAME: &'static str = CHECK_JSONL;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = JsonlArtifact;
    type Output = JsonlArtifact;
    type Args = CheckJsonlArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Self::Args,
    ) -> Result<Self::Output, StageError> {
        let body = std::fs::read_to_string(&input.path).map_err(|e| {
            StageError::Backend(anyhow::anyhow!(
                "check_jsonl read {}: {e}",
                input.path.display()
            ))
        })?;
        let rows = body.lines().filter(|l| !l.trim().is_empty()).count() as u64;

        if let Some(min) = args.min_rows {
            if rows < min {
                handle_breach(
                    ctx,
                    CHECK_JSONL,
                    "DATA_QUALITY_MIN_ROWS",
                    args.on_breach,
                    format!("row count {rows} below min_rows {min}"),
                    &[("rows", rows.to_string()), ("min_rows", min.to_string())],
                )?;
            }
        }
        if let Some(max) = args.max_rows {
            if rows > max {
                handle_breach(
                    ctx,
                    CHECK_JSONL,
                    "DATA_QUALITY_MAX_ROWS",
                    args.on_breach,
                    format!("row count {rows} above max_rows {max}"),
                    &[("rows", rows.to_string()), ("max_rows", max.to_string())],
                )?;
            }
        }
        // Passthrough: same path + hash → downstream sees byte-identical input.
        Ok(input)
    }
}

// ── assert ─────────────────────────────────────────────────────────

/// Comparison operator for [`Assert`].
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum CmpOp {
    #[default]
    Ge,
    Gt,
    Le,
    Lt,
    Eq,
    Ne,
}

impl CmpOp {
    fn holds(self, actual: f64, expected: f64, tol: f64) -> bool {
        match self {
            CmpOp::Ge => actual >= expected - tol,
            CmpOp::Gt => actual > expected - tol,
            CmpOp::Le => actual <= expected + tol,
            CmpOp::Lt => actual < expected + tol,
            CmpOp::Eq => (actual - expected).abs() <= tol,
            CmpOp::Ne => (actual - expected).abs() > tol,
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            CmpOp::Ge => ">=",
            CmpOp::Gt => ">",
            CmpOp::Le => "<=",
            CmpOp::Lt => "<",
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
        }
    }
}

/// Args for [`Assert`]: a scalar predicate `actual <op> expected` (± tolerance).
#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AssertArgs {
    /// The measured value (typically threaded in from an upstream metric).
    pub actual: f64,
    /// The comparison operator.
    #[serde(default)]
    pub op: CmpOp,
    /// The bound to compare against.
    pub expected: f64,
    /// Absolute tolerance applied to the comparison (default 0).
    #[serde(default)]
    pub tolerance: f64,
    /// Breach policy (default: `block`).
    #[serde(default)]
    pub on_breach: OnBreach,
}

/// Stage name of the scalar assertion guard.
pub const ASSERT: &str = "assert";

/// Assert a scalar predicate over `args`, passing the input artifact through
/// unchanged. A guard node in a JSONL pipeline: it neither reads nor rewrites the
/// artifact, only enforces `actual <op> expected` and forwards the input.
pub struct Assert;

#[async_trait]
impl Stage for Assert {
    const NAME: &'static str = ASSERT;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = JsonlArtifact;
    type Output = JsonlArtifact;
    type Args = AssertArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Self::Args,
    ) -> Result<Self::Output, StageError> {
        if !args.op.holds(args.actual, args.expected, args.tolerance) {
            handle_breach(
                ctx,
                ASSERT,
                "ASSERT_FAILED",
                args.on_breach,
                format!(
                    "assertion failed: {} {} {} (tol {})",
                    args.actual,
                    args.op.symbol(),
                    args.expected,
                    args.tolerance
                ),
                &[
                    ("actual", args.actual.to_string()),
                    ("expected", args.expected.to_string()),
                    ("op", args.op.symbol().to_string()),
                ],
            )?;
        }
        Ok(input)
    }
}

// ── cookbook ───────────────────────────────────────────────────────

/// The built-in `checks` cookbook: `check_jsonl` + `assert` passthrough guards
/// and the `checks` error domain.
pub struct ChecksCookbook;

impl Cookbook for ChecksCookbook {
    fn name(&self) -> &'static str {
        "checks"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        &[]
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static S: &[(&str, ErasedStageCtor)] = &[
            (CHECK_JSONL, || std::sync::Arc::new(CheckJsonl)),
            (ASSERT, || std::sync::Arc::new(Assert)),
        ];
        S
    }
    fn error_domains(&self) -> &'static [&'static ErrorDomainDef] {
        static DOMAINS: &[&ErrorDomainDef] = &[&ERROR_DOMAIN_DEF];
        DOMAINS
    }
}

/// Register the built-in `checks` cookbook into `reg`.
pub fn register(reg: &mut Registry) {
    reg.register(Box::new(ChecksCookbook));
}

// ── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::error_domain::FailureSummary;

    /// Write `n` JSONL rows to a temp file and return the artifact.
    fn write_jsonl(dir: &Path, n: u64) -> JsonlArtifact {
        let path = dir.join("data.jsonl");
        let mut body = String::new();
        for i in 0..n {
            body.push_str(&format!("{{\"i\":{i}}}\n"));
        }
        std::fs::write(&path, &body).unwrap();
        let content_hash = ContentHash::hash_file(&path).unwrap();
        JsonlArtifact { path, content_hash }
    }

    fn test_ctx(dir: &Path) -> StageContext {
        StageContext::for_test(dir.to_path_buf(), dir.to_path_buf())
    }

    /// The block-severity min_rows breach fails the node with the right
    /// `checks`-domain failure — so the executor skips the downstream.
    #[tokio::test]
    async fn check_jsonl_below_min_blocks_fail_closed() {
        let tmp = std::env::temp_dir().join(format!("checks_min_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let art = write_jsonl(&tmp, 900);
        let ctx = test_ctx(&tmp);
        let args = CheckJsonlArgs {
            min_rows: Some(1000),
            max_rows: None,
            on_breach: OnBreach::Block,
        };
        let err = CheckJsonl.run(&ctx, art, &args).await.unwrap_err();
        let StageError::Backend(anyhow_err) = &err else {
            panic!("expected Backend failure, got {err:?}");
        };
        let f = StageFailure::try_extract(anyhow_err).expect("checks StageFailure in the chain");
        assert_eq!(f.domain, "checks");
        assert_eq!(f.code, "DATA_QUALITY_MIN_ROWS");
        assert_eq!(f.severity, Severity::Major);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The passing run returns the input BYTE-IDENTICAL — same path + hash, so a
    /// downstream node's cache key is unchanged.
    #[tokio::test]
    async fn check_jsonl_at_min_passes_through_byte_identical() {
        let tmp = std::env::temp_dir().join(format!("checks_pass_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let art = write_jsonl(&tmp, 1000);
        let (in_path, in_hash) = (art.path.clone(), art.content_hash);
        let ctx = test_ctx(&tmp);
        let args = CheckJsonlArgs {
            min_rows: Some(1000),
            max_rows: None,
            on_breach: OnBreach::Block,
        };
        let out = CheckJsonl.run(&ctx, art, &args).await.unwrap();
        assert_eq!(out.path, in_path, "passthrough keeps the path");
        assert_eq!(
            out.content_hash, in_hash,
            "passthrough keeps the content hash"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A warn-severity breach records a lineage breach event AND passes through
    /// (the DAG continues).
    #[tokio::test]
    async fn check_jsonl_warn_passes_through_and_records_lineage() {
        let tmp = std::env::temp_dir().join(format!("checks_warn_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let art = write_jsonl(&tmp, 3);
        let (in_hash, ctx) = (art.content_hash, test_ctx(&tmp));
        let mut rx = ctx.status_tx.subscribe();
        let args = CheckJsonlArgs {
            min_rows: Some(1000),
            max_rows: None,
            on_breach: OnBreach::Warn,
        };
        let out = CheckJsonl.run(&ctx, art, &args).await.unwrap();
        assert_eq!(out.content_hash, in_hash, "warn still passes through");
        let ev = rx.try_recv().expect("a breach event was emitted");
        match ev {
            StageEvent::StageStep { update, .. } => {
                assert_eq!(update["checks_breach"]["severity"], "warn");
                assert_eq!(update["checks_breach"]["code"], "DATA_QUALITY_MIN_ROWS");
            }
            other => panic!("expected StageStep breach, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A satisfied assertion passes through; a violated one fail-closes with an
    /// `ASSERT_FAILED` failure.
    #[tokio::test]
    async fn assert_predicate_gates_the_pipeline() {
        let tmp = std::env::temp_dir().join(format!("checks_assert_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let art = write_jsonl(&tmp, 5);
        let ctx = test_ctx(&tmp);

        // 0.86 >= 0.85 holds → passthrough.
        let ok_args = AssertArgs {
            actual: 0.86,
            op: CmpOp::Ge,
            expected: 0.85,
            tolerance: 0.0,
            on_breach: OnBreach::Block,
        };
        assert!(Assert.run(&ctx, art.clone(), &ok_args).await.is_ok());

        // 0.80 >= 0.85 fails → block.
        let bad_args = AssertArgs {
            actual: 0.80,
            expected: 0.85,
            ..ok_args.clone()
        };
        let err = Assert.run(&ctx, art, &bad_args).await.unwrap_err();
        let StageError::Backend(a) = &err else {
            panic!("expected Backend");
        };
        let summary = FailureSummary::from(StageFailure::try_extract(a).unwrap());
        assert_eq!(summary.code, "ASSERT_FAILED");
        assert_eq!(summary.domain, "checks");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The error domain publishes its data-quality codes for `blut errors list`.
    #[test]
    fn checks_error_domain_lists_codes() {
        let mut reg = Registry::new();
        register(&mut reg);
        let codes: Vec<_> = reg.all_error_domains().flat_map(|d| d.codes).collect();
        assert!(codes.iter().any(|(c, _)| *c == "DATA_QUALITY_MIN_ROWS"));
        assert!(codes.iter().any(|(c, _)| *c == "ASSERT_FAILED"));
    }
}
