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
        // Stream the row count on a blocking thread: O(1) memory (never loads
        // the whole file) and off the async runtime, so a multi-GB JSONL neither
        // spikes RAM nor stalls the executor's reactor.
        let path = input.path.clone();
        let rows = tokio::task::spawn_blocking(move || -> std::io::Result<u64> {
            use std::io::BufRead;
            let f = std::fs::File::open(&path)?;
            let mut n = 0u64;
            for line in std::io::BufReader::new(f).lines() {
                if !line?.trim().is_empty() {
                    n += 1;
                }
            }
            Ok(n)
        })
        .await
        .map_err(|e| StageError::Backend(anyhow::anyhow!("check_jsonl join: {e}")))?
        .map_err(|e| {
            StageError::Backend(anyhow::anyhow!(
                "check_jsonl read {}: {e}",
                input.path.display()
            ))
        })?;

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
    /// Absolute tolerance applied to the comparison (default 0). Band semantics:
    /// `Ge`/`Gt` relax the bound to `expected - tolerance`; `Le`/`Lt` relax it to
    /// `expected + tolerance`; `Eq`/`Ne` treat `|actual - expected| <= tolerance`
    /// as equal. Note a `NaN` `actual` fails every predicate except `Ne` (NaN is
    /// unequal to everything) — intentional: an unmeasured value never satisfies
    /// a data-quality floor.
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

// ── report (ADR 0091: `blut checks report`) ────────────────────────

/// One data-quality breach replayed from a run's `status.jsonl` — either a
/// fail-closed BLOCK (a `checks`-domain `StageFailed`) or an advisory WARN (a
/// `StageStep` `checks_breach` that the DAG continued past).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Breach {
    pub node_idx: u32,
    pub stage: String,
    pub code: String,
    /// `"block"` (the node failed, downstream skipped) or `"warn"` (advisory,
    /// the DAG continued).
    pub disposition: String,
    pub detail: String,
}

impl Breach {
    /// The breach category. Codes prefixed `DATA_QUALITY` fold into
    /// `"DataQuality"` — the tag `blut checks report | grep DataQuality` matches
    /// (ADR 0091 gate); everything else is `"Assertion"`.
    pub fn category(&self) -> &'static str {
        if self.code.starts_with("DATA_QUALITY") {
            "DataQuality"
        } else {
            "Assertion"
        }
    }
}

/// Scan `StageEvent`s for checks breaches: a `StageFailed` in the `checks`
/// domain (fail-closed BLOCK) and a `StageStep` carrying a `checks_breach`
/// payload (advisory WARN). The pure core, shared by the CLI reader and tests.
pub fn scan_breaches(events: impl IntoIterator<Item = StageEvent>) -> Vec<Breach> {
    let mut out = Vec::new();
    for ev in events {
        match ev {
            StageEvent::StageFailed {
                node_idx,
                stage_name,
                failure: Some(f),
                ..
            } if f.domain == DOMAIN => {
                out.push(Breach {
                    node_idx,
                    stage: stage_name,
                    code: f.code,
                    disposition: "block".to_string(),
                    detail: f.message,
                });
            }
            StageEvent::StageStep {
                node_idx,
                stage_name,
                update,
            } => {
                if let Some(b) = update.get("checks_breach") {
                    out.push(Breach {
                        node_idx,
                        stage: stage_name,
                        code: b
                            .get("code")
                            .and_then(|c| c.as_str())
                            .unwrap_or("")
                            .to_string(),
                        disposition: "warn".to_string(),
                        detail: b
                            .get("detail")
                            .and_then(|d| d.as_str())
                            .unwrap_or("")
                            .to_string(),
                    });
                }
            }
            _ => {}
        }
    }
    out
}

/// Replay a job's `status.jsonl` for its checks breaches (the reader
/// `blut checks report <job>` uses). Tolerant of unparseable lines, exactly like
/// [`crate::framework::lineage::job_failure`].
pub fn report_job(job: &str) -> anyhow::Result<Vec<Breach>> {
    let id = crate::jobs::resolve_job_id(job).map_err(|e| anyhow::anyhow!("{e}"))?;
    let events = crate::jobs::read_status_lines(&id)?
        .into_iter()
        .filter_map(|l| serde_json::from_str::<StageEvent>(&l).ok());
    Ok(scan_breaches(events))
}

/// Render breaches as a grep-friendly report. Each line leads with the breach
/// CATEGORY (`DataQuality` / `Assertion`) so `blut checks report | grep
/// DataQuality` finds the data-quality breaches (ADR 0091 gate).
pub fn render_report(breaches: &[Breach]) -> String {
    if breaches.is_empty() {
        return "no checks breaches recorded\n".to_string();
    }
    use std::fmt::Write as _;
    let mut s = String::new();
    for b in breaches {
        let _ = writeln!(
            s,
            "[{}] {} @{} (node {}) {} — {}",
            b.category(),
            b.disposition.to_uppercase(),
            b.stage,
            b.node_idx,
            b.code,
            b.detail
        );
    }
    s
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
        let tmp = tempfile::tempdir().unwrap();
        let art = write_jsonl(tmp.path(), 900);
        let ctx = test_ctx(tmp.path());
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
    }

    /// The passing run returns the input BYTE-IDENTICAL — same path + hash, so a
    /// downstream node's cache key is unchanged.
    #[tokio::test]
    async fn check_jsonl_at_min_passes_through_byte_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let art = write_jsonl(tmp.path(), 1000);
        let (in_path, in_hash) = (art.path.clone(), art.content_hash);
        let ctx = test_ctx(tmp.path());
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
    }

    /// A warn-severity breach records a lineage breach event AND passes through
    /// (the DAG continues).
    #[tokio::test]
    async fn check_jsonl_warn_passes_through_and_records_lineage() {
        let tmp = tempfile::tempdir().unwrap();
        let art = write_jsonl(tmp.path(), 3);
        let (in_hash, ctx) = (art.content_hash, test_ctx(tmp.path()));
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
    }

    /// A satisfied assertion passes through; a violated one fail-closes with an
    /// `ASSERT_FAILED` failure.
    #[tokio::test]
    async fn assert_predicate_gates_the_pipeline() {
        let tmp = tempfile::tempdir().unwrap();
        let art = write_jsonl(tmp.path(), 5);
        let ctx = test_ctx(tmp.path());

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

    // ── end-to-end: the `checks-demo` DAG through the real executor ─────
    //
    // gen(rows) → check_jsonl(min_rows=1000) → sink. Proves the fail-closed
    // contract THROUGH the executor: a block breach skips the downstream sink;
    // a pass runs it; a warn continues. This is the ADR 0091 gate's `blut run
    // checks-demo` behaviour — run in-crate because the executor/plan
    // constructors are `pub(crate)` and the engine ships no binary of its own.

    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
    struct DemoGenArgs {
        rows: u64,
    }

    /// Demo source: write `rows` JSONL lines and hand on a `JsonlArtifact`.
    struct DemoGen;
    #[async_trait]
    impl Stage for DemoGen {
        const NAME: &'static str = "checks_demo_gen";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = ();
        type Output = JsonlArtifact;
        type Args = DemoGenArgs;
        async fn run(
            &self,
            ctx: &StageContext,
            _input: (),
            args: &DemoGenArgs,
        ) -> Result<JsonlArtifact, StageError> {
            std::fs::create_dir_all(&ctx.stage_dir).ok();
            let path = ctx.stage_dir.join("data.jsonl");
            let mut body = String::new();
            for i in 0..args.rows {
                body.push_str(&format!("{{\"i\":{i}}}\n"));
            }
            std::fs::write(&path, &body)
                .map_err(|e| StageError::Backend(anyhow::anyhow!("demo gen write: {e}")))?;
            let content_hash = ContentHash::hash_file(&path)
                .map_err(|e| StageError::Backend(anyhow::anyhow!("demo gen hash: {e}")))?;
            Ok(JsonlArtifact { path, content_hash })
        }
    }

    /// Demo sink: bump a PER-RUN counter (its execution is the downstream we must
    /// prove is skipped on a block breach). Each run owns its own `Arc<AtomicUsize>`
    /// so the tests need no shared static or lock — they run concurrently.
    struct DemoSink(Arc<AtomicUsize>);
    #[async_trait]
    impl Stage for DemoSink {
        const NAME: &'static str = "checks_demo_sink";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = JsonlArtifact;
        type Output = ();
        type Args = ();
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: JsonlArtifact,
            _args: &(),
        ) -> Result<(), StageError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Build + run the demo DAG at `rows` with breach policy `on_breach`; return
    /// the executor result, the captured `StageEvent`s, and how many times the
    /// downstream sink ran (0 ⇒ skipped).
    async fn run_demo(
        rows: u64,
        on_breach: OnBreach,
    ) -> (
        Result<crate::framework::executor::PlanResult, crate::framework::error::PlanError>,
        Vec<StageEvent>,
        usize,
    ) {
        use crate::framework::executor::{ExecCtx, SequentialExecutor};
        use crate::framework::plan::CompiledPlan;
        use crate::framework::stage::StageDyn;

        let td = tempfile::tempdir().unwrap();
        let sink_ran = Arc::new(AtomicUsize::new(0));
        let mut check_args = serde_json::json!({ "min_rows": 1000 });
        check_args["on_breach"] = serde_json::to_value(on_breach).unwrap();
        let nodes: Vec<(Arc<dyn StageDyn>, serde_json::Value)> = vec![
            (Arc::new(DemoGen), serde_json::json!({ "rows": rows })),
            (Arc::new(CheckJsonl), check_args),
            (
                Arc::new(DemoSink(sink_ran.clone())),
                serde_json::Value::Null,
            ),
        ];
        let plan = CompiledPlan::from_erased_graph(
            "checks-demo",
            serde_json::json!({}),
            nodes,
            vec![(0, 1), (1, 2)],
        )
        .expect("demo plan compiles (kinds line up)");
        let ctx = ExecCtx::new(td.path().to_path_buf());
        let mut rx = ctx.status.subscribe();
        let res = SequentialExecutor::execute(plan, ctx).await;
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        (res, events, sink_ran.load(Ordering::SeqCst))
    }

    /// rows=900 < min_rows=1000, on_breach=block: the run FAILS and the
    /// downstream sink NEVER runs (fail-closed), and the report shows the
    /// DataQuality block breach.
    #[tokio::test]
    async fn checks_demo_block_skips_downstream() {
        let (res, events, sink_ran) = run_demo(900, OnBreach::Block).await;
        assert!(res.is_err(), "a block breach must fail the run");
        assert_eq!(sink_ran, 0, "downstream sink must be skipped (fail-closed)");
        let breaches = scan_breaches(events);
        assert!(
            breaches
                .iter()
                .any(|b| b.disposition == "block" && b.category() == "DataQuality"),
            "the report must show the blocked DataQuality breach: {breaches:?}"
        );
        assert!(render_report(&breaches).contains("DataQuality"));
    }

    /// rows=1000 >= min_rows: the run SUCCEEDS and the downstream sink runs.
    #[tokio::test]
    async fn checks_demo_pass_runs_downstream() {
        let (res, _events, sink_ran) = run_demo(1000, OnBreach::Block).await;
        assert!(
            res.is_ok(),
            "a passing check must not fail the run: {res:?}"
        );
        assert_eq!(sink_ran, 1, "downstream sink runs when the check passes");
    }

    /// rows=900 with on_breach=warn: the DAG CONTINUES (sink runs) and the
    /// advisory DataQuality breach is on the record.
    #[tokio::test]
    async fn checks_demo_warn_continues_and_records() {
        let (res, events, sink_ran) = run_demo(900, OnBreach::Warn).await;
        assert!(res.is_ok(), "a warn breach must NOT fail the run: {res:?}");
        assert_eq!(sink_ran, 1, "downstream sink runs under a warn breach");
        let breaches = scan_breaches(events);
        assert!(
            breaches
                .iter()
                .any(|b| b.disposition == "warn" && b.category() == "DataQuality"),
            "the report must show the advisory DataQuality breach: {breaches:?}"
        );
    }
}
