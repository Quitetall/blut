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
//!
//! # The input contract
//!
//! Both stages take and return [`JsonlArtifact`] (kind `checks.jsonl`), and a
//! graph kind-checks edges by exact kind. The engine is domain-agnostic and
//! cannot know a cookbook's own artifact types, so a check cannot sit directly
//! after a stage that emits another kind (for example the standard cookbook's
//! `dataset.jsonl`): `plan check` refuses that edge as a kind break. To guard a
//! cookbook's JSONL, either emit [`JsonlArtifact`] from the producing stage, or
//! add a one-line adapter stage that re-labels the file:
//!
//! ```
//! use async_trait::async_trait;
//! use blut::checks::JsonlArtifact;
//! use blut::framework::{Resource, Stage, StageContext, StageError};
//! # use blut::framework::{Artifact, ContentHash};
//! # #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
//! # struct MyJsonl { path: std::path::PathBuf }
//! # impl Artifact for MyJsonl {
//! #     const KIND: &'static str = "my.jsonl";
//! #     const SCHEMA: u32 = 1;
//! #     fn content_hash(&self) -> ContentHash { ContentHash::hash_file(&self.path).unwrap() }
//! #     fn primary_path(&self) -> &std::path::Path { &self.path }
//! # }
//!
//! /// `my.jsonl` -> `checks.jsonl`: the same file, so `check_jsonl` can guard it.
//! struct ToChecks;
//!
//! #[async_trait]
//! impl Stage for ToChecks {
//!     const NAME: &'static str = "my_jsonl_to_checks";
//!     const SCHEMA: u32 = 1;
//!     const RESOURCES: &'static [Resource] = &[Resource::Cpu];
//!     type Input = MyJsonl;
//!     type Output = JsonlArtifact;
//!     type Args = ();
//!     async fn run(&self, _: &StageContext, input: MyJsonl, _: &()) -> Result<JsonlArtifact, StageError> {
//!         JsonlArtifact::from_file(&input.path)
//!             .map_err(|e| StageError::BadInput(format!("hash {}: {e}", input.path.display())))
//!     }
//! }
//! ```
//!
//! [`Assert`] compares the literal `actual` it is given; the engine does not
//! read a stage's emitted metrics into it.
//!
//! # `check_jsonl` line checks
//!
//! A plan's `check_jsonl` resolves to [`CheckJsonlLines`], which besides the
//! row-count bounds (`min_rows`, `max_rows`) can check every row (ADR 0091's
//! `check_jsonl{schema, min_rows, max_rows, required_fields}`):
//!
//! ```json
//! { "min_rows": 1000,
//!   "required_fields": ["id", "text"],
//!   "schema": { "id": "integer",
//!               "text": "string",
//!               "label": { "type": "string", "nullable": true, "optional": true } },
//!   "max_reported": 10 }
//! ```
//!
//! **`schema`** maps a top-level field name to its rule. A rule is a type name,
//! or an object `{"type": T, "nullable": bool, "optional": bool}` (both flags
//! default to `false`; no other keys are accepted). `T` is one of:
//!
//! | type | a value matches when it is |
//! |---|---|
//! | `string` | a JSON string |
//! | `number` | any JSON number |
//! | `integer` | a JSON number with no fractional part (`3` and `3.0`, as JSON Schema counts them; not `3.5`) |
//! | `boolean` | `true` or `false` |
//! | `object` | a JSON object (its contents are not checked) |
//! | `array` | a JSON array (its elements are not checked) |
//! | `null` | `null` |
//!
//! A schema field must be present unless it is `optional`, and must not be
//! `null` unless it is `nullable` (or of type `null`). Fields the schema does
//! not name are not checked. Names are top-level keys taken literally: `a.b` is
//! the key `"a.b"`, not a path. With `schema` set, even to `{}`, every row must
//! be a JSON object.
//!
//! **`required_fields`** lists keys every row must carry, non-`null` unless
//! `schema` allows `null` for that field. Listing a field here makes it
//! required even if `schema` marks it `optional`: the two arguments are
//! separate constraints and both must hold.
//!
//! **Breaches.** Each row is checked in this order and stops at its first
//! problem, so a row is counted once: it must be valid JSON
//! (`DATA_QUALITY_INVALID_JSON`; a line that is not UTF-8 counts as invalid),
//! then a JSON object (`DATA_QUALITY_SCHEMA`), then carry each required field
//! in the order listed (`DATA_QUALITY_MISSING_FIELD`), then match each schema
//! rule in name order (absent and not `optional` ⇒ `DATA_QUALITY_MISSING_FIELD`,
//! wrong type or a disallowed `null` ⇒ `DATA_QUALITY_SCHEMA`). The whole file is
//! read: every breaching row is counted, and each breach code names its first
//! `max_reported` rows by line number and field. The default is 10 and the
//! ceiling 1000; the first breaching row is always named, so `0` acts as `1`.
//!
//! Line numbers are 1-based physical lines of the file, blank lines included,
//! so they match an editor or `sed -n '<N>p'`. Blank lines (empty or whitespace
//! only) are not rows: they are skipped by the row count and by the line checks
//! alike, so the rows counted are exactly the rows checked. An empty file has
//! no rows to break a rule, so only `min_rows` can refuse it.
//!
//! Memory is bounded by the longest line, not by the file: the scan holds one
//! line and its parsed value at a time, plus the rows named in the breach
//! message. A file with no newlines is one line, and is read whole.
//!
//! The row-count bounds are checked first, then the line checks. Under `block`
//! the first failing one is the node's failure: a line-check failure carries the
//! code of the earliest breaching line, and its message summarizes every
//! code found. Under `warn` each failing bound, and each line-breach code, is
//! its own breach event, and the input passes through. A breach message names
//! line numbers, field names from the args and JSON type names only, never a
//! row's contents.
//!
//! Without `schema` and `required_fields` the stage counts rows exactly as it
//! always has: no line is parsed, so a row that is not JSON is still a row.
//!
//! The arguments are validated when a plan is built (`plan check`, a `.toml`,
//! `.json` or `.star` recipe): an unknown type, an empty field name, a
//! repeated field in `required_fields` or an unknown key in a rule is refused
//! before any stage runs.
//!
//! `check_tensor_stats` from ADR 0091 is not here: the engine has no tensor
//! artifact type for it to check yet.
//!
//! # Two Rust types, one stage
//!
//! [`CheckJsonl`] and [`CheckJsonlArgs`] are the row-count stage as published:
//! `CheckJsonlArgs` is a struct every field of which a caller may name, so it
//! cannot gain a field without breaking that caller. They stay as they were,
//! and delegate to [`CheckJsonlLines`], the stage registered as `check_jsonl`.
//! New arguments go on [`CheckJsonlLinesArgs`], which is `#[non_exhaustive]`
//! for that reason. Both stages share a name, schema and artifact kinds, and
//! args that name only the row bounds mean the same to either. A Rust caller
//! wanting line checks uses `CheckJsonlLines`: `CheckJsonl` has no such args.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::framework::artifact::{Artifact, ContentHash};
use crate::framework::cookbook::{Cookbook, Registry};
use crate::framework::error::StageError;
use crate::framework::error_domain::{ErrorDomain, ErrorDomainDef, Severity, StageFailure};
use crate::framework::resource::Resource;
use crate::framework::stage::{ErasedStageCtor, Stage, StageContext, StageExecutionBoundary};
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
            "DATA_QUALITY_INVALID_JSON",
            "a jsonl row is not valid JSON (or not UTF-8)",
        ),
        (
            "DATA_QUALITY_MISSING_FIELD",
            "a jsonl row lacks a required field, or carries it as null",
        ),
        (
            "DATA_QUALITY_SCHEMA",
            "a jsonl row is not an object, or a field does not match its schema type",
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

impl JsonlArtifact {
    /// A handle for an existing JSONL file: its path and content hash. What an
    /// adapter stage returns to put a cookbook's JSONL in front of the checks
    /// stages (see the module docs).
    pub fn from_file(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let content_hash = ContentHash::hash_file(&path)?;
        Ok(Self { path, content_hash })
    }
}

impl Artifact for JsonlArtifact {
    const KIND: &'static str = "checks.jsonl";
    const SCHEMA: u32 = 1;
    // Check stages intentionally pass through source files owned by upstream
    // stages or datasets. Such handles remain usable but are not published as
    // portable cache objects unless their bytes live beneath producer root.
    const ALLOW_EXTERNAL_PATHS: bool = true;
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

/// Args for [`CheckJsonlLines`], the stage a plan's `check_jsonl` resolves to:
/// row-count bounds, line checks and breach policy. The module docs describe
/// the `schema` format and the order rows are checked in. Every line-check
/// argument defaults to "off" and is left out when the args are serialized,
/// so row-count-only args keep their shape. `#[non_exhaustive]`: start from
/// `CheckJsonlLinesArgs::default()` and set the fields you want.
#[non_exhaustive]
#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CheckJsonlLinesArgs {
    /// Minimum row count (inclusive). `None` = no lower bound.
    #[serde(default)]
    pub min_rows: Option<u64>,
    /// Maximum row count (inclusive). `None` = no upper bound.
    #[serde(default)]
    pub max_rows: Option<u64>,
    /// Breach policy (default: `block` — fail-closed).
    #[serde(default)]
    pub on_breach: OnBreach,
    /// Line schema: top-level field name → rule, a type name (`string`,
    /// `number`, `integer`, `boolean`, `object`, `array`, `null`) or
    /// `{"type": …, "nullable": bool, "optional": bool}`. `None` = no schema.
    /// When set, even to `{}`, every row must be a JSON object.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "de_schema"
    )]
    pub schema: Option<BTreeMap<String, FieldSpec>>,
    /// Keys every row must carry, non-null unless `schema` allows null for
    /// that field. Empty = none.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "de_required_fields"
    )]
    pub required_fields: Vec<String>,
    /// How many breaching rows each line-breach code names in its message
    /// (default 10, at most 1000). The first breaching row is always named,
    /// so `0` acts as `1`; every breaching row is counted either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_reported: Option<u32>,
}

impl From<CheckJsonlArgs> for CheckJsonlLinesArgs {
    /// The same row bounds and breach policy, with every line check off.
    fn from(args: CheckJsonlArgs) -> Self {
        let CheckJsonlArgs {
            min_rows,
            max_rows,
            on_breach,
        } = args;
        Self {
            min_rows,
            max_rows,
            on_breach,
            ..Self::default()
        }
    }
}

/// Rows each line-breach code names when `max_reported` is not set.
const DEFAULT_MAX_REPORTED: u32 = 10;

/// The most rows a line-breach code names, whatever `max_reported` says. What
/// the scan holds must not scale with a file that fails on every row.
const MAX_REPORTED_CEILING: u32 = 1000;

/// The JSON type a [`FieldSpec`] requires; the module docs say what each one
/// matches. Spelled in args as its lowercase name.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    String,
    Number,
    Integer,
    Boolean,
    Object,
    Array,
    Null,
}

impl FieldType {
    const ALL: [FieldType; 7] = [
        FieldType::String,
        FieldType::Number,
        FieldType::Integer,
        FieldType::Boolean,
        FieldType::Object,
        FieldType::Array,
        FieldType::Null,
    ];

    /// The name args spell this type with.
    pub fn name(self) -> &'static str {
        match self {
            FieldType::String => "string",
            FieldType::Number => "number",
            FieldType::Integer => "integer",
            FieldType::Boolean => "boolean",
            FieldType::Object => "object",
            FieldType::Array => "array",
            FieldType::Null => "null",
        }
    }

    /// Whether `value` is of this type. `integer` takes any number without a
    /// fractional part (`3.0` included, as JSON Schema counts it).
    pub fn matches(self, value: &serde_json::Value) -> bool {
        use serde_json::Value;
        match self {
            FieldType::String => value.is_string(),
            FieldType::Number => value.is_number(),
            FieldType::Integer => matches!(value, Value::Number(n) if is_whole(n)),
            FieldType::Boolean => value.is_boolean(),
            FieldType::Object => value.is_object(),
            FieldType::Array => value.is_array(),
            FieldType::Null => value.is_null(),
        }
    }

    fn parse(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|t| t.name() == name)
            .ok_or_else(|| format!("unknown type {name:?}; expected one of {}", type_names()))
    }
}

fn type_names() -> String {
    FieldType::ALL.map(FieldType::name).join(", ")
}

fn is_whole(n: &serde_json::Number) -> bool {
    n.is_i64()
        || n.is_u64()
        || n.as_f64()
            .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
}

/// The JSON kind of `value`, for breach messages: never the value itself.
fn json_kind(value: &serde_json::Value) -> &'static str {
    use serde_json::Value;
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) if is_whole(n) => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// One field's rule in a `check_jsonl` line schema. In args it is either a
/// type name (`"integer"`) or an object `{"type": "integer", "nullable":
/// true, "optional": true}`; both flags default to `false`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct FieldSpec {
    /// The type a present, non-null value must have.
    #[serde(rename = "type")]
    pub ty: FieldType,
    /// `null` is accepted in place of a value of `ty`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub nullable: bool,
    /// The field may be absent.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
}

impl FieldSpec {
    /// A rule requiring a present, non-null value of `ty`.
    pub fn new(ty: FieldType) -> Self {
        Self {
            ty,
            nullable: false,
            optional: false,
        }
    }

    /// The same rule, also accepting `null` (sets [`nullable`](Self::nullable)).
    pub fn allow_null(mut self) -> Self {
        self.nullable = true;
        self
    }

    /// The same rule, also accepting an absent field (sets
    /// [`optional`](Self::optional)).
    pub fn allow_missing(mut self) -> Self {
        self.optional = true;
        self
    }

    fn allows_null(&self) -> bool {
        self.nullable || self.ty == FieldType::Null
    }

    /// Parse a rule from its args form, with a message naming what is wrong.
    fn from_json(raw: &serde_json::Value) -> Result<Self, String> {
        use serde_json::Value;
        match raw {
            Value::String(name) => Ok(Self::new(FieldType::parse(name)?)),
            Value::Object(map) => {
                let flag = |key: &str, v: &Value| {
                    v.as_bool()
                        .ok_or_else(|| format!("{key:?} must be true or false"))
                };
                let (mut ty, mut nullable, mut optional) = (None, false, false);
                for (key, v) in map {
                    match key.as_str() {
                        "type" => {
                            let name = v.as_str().ok_or_else(|| {
                                format!("\"type\" must be a string, one of {}", type_names())
                            })?;
                            ty = Some(FieldType::parse(name)?);
                        }
                        "nullable" => nullable = flag(key, v)?,
                        "optional" => optional = flag(key, v)?,
                        other => {
                            return Err(format!(
                                "unknown key {other:?}; a rule takes \"type\", \"nullable\" \
                                 and \"optional\""
                            ));
                        }
                    }
                }
                let ty = ty.ok_or_else(|| "a rule object needs a \"type\"".to_string())?;
                Ok(Self {
                    ty,
                    nullable,
                    optional,
                })
            }
            other => Err(format!(
                "a rule is a type name or a {{\"type\", \"nullable\", \"optional\"}} object, \
                 not {}",
                json_kind(other)
            )),
        }
    }
}

impl<'de> Deserialize<'de> for FieldSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = serde_json::Value::deserialize(d)?;
        Self::from_json(&raw).map_err(serde::de::Error::custom)
    }
}

/// The documented shape of a [`FieldSpec`] for the args JSON schema: the
/// type-name shorthand or the rule object. Only its schema is used.
#[allow(dead_code)]
#[derive(schemars::JsonSchema)]
#[serde(untagged)]
enum FieldSpecShape {
    /// A type name: the field must be present, non-null and of this type.
    Type(FieldType),
    /// A rule object.
    Rule(FieldRuleShape),
}

#[allow(dead_code)]
#[derive(schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct FieldRuleShape {
    /// The type a present, non-null value must have.
    #[serde(rename = "type")]
    ty: FieldType,
    /// Accept `null` in place of a value (default false).
    #[serde(default)]
    nullable: bool,
    /// Accept an absent field (default false).
    #[serde(default)]
    optional: bool,
}

impl schemars::JsonSchema for FieldSpec {
    fn schema_name() -> String {
        "FieldSpec".to_string()
    }
    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        FieldSpecShape::json_schema(generator)
    }
}

/// `schema`: refuse an empty field name and parse every rule, naming the
/// field whose rule is wrong (the plan-build check runs this decode).
fn de_schema<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<BTreeMap<String, FieldSpec>>, D::Error> {
    use serde::de::Error as _;
    let Some(raw) = Option::<BTreeMap<String, serde_json::Value>>::deserialize(d)? else {
        return Ok(None);
    };
    let mut schema = BTreeMap::new();
    for (name, rule) in raw {
        if name.is_empty() {
            return Err(D::Error::custom("schema: a field name is empty"));
        }
        let spec = FieldSpec::from_json(&rule)
            .map_err(|e| D::Error::custom(format!("schema field {name:?}: {e}")))?;
        schema.insert(name, spec);
    }
    Ok(Some(schema))
}

/// `required_fields`: refuse an empty or repeated name; `null` means none.
fn de_required_fields<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    use serde::de::Error as _;
    let fields = Option::<Vec<String>>::deserialize(d)?.unwrap_or_default();
    let mut seen = std::collections::BTreeSet::new();
    for field in &fields {
        if field.is_empty() {
            return Err(D::Error::custom("required_fields: a field name is empty"));
        }
        if !seen.insert(field.as_str()) {
            return Err(D::Error::custom(format!(
                "required_fields: {field:?} is listed twice"
            )));
        }
    }
    Ok(fields)
}

// ── line checks ────────────────────────────────────────────────────

/// The three line-level breach kinds, in the order their codes are listed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineCode {
    InvalidJson,
    MissingField,
    Schema,
}

impl LineCode {
    const ALL: [LineCode; 3] = [
        LineCode::InvalidJson,
        LineCode::MissingField,
        LineCode::Schema,
    ];

    fn code(self) -> &'static str {
        match self {
            LineCode::InvalidJson => "DATA_QUALITY_INVALID_JSON",
            LineCode::MissingField => "DATA_QUALITY_MISSING_FIELD",
            LineCode::Schema => "DATA_QUALITY_SCHEMA",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Why one row failed its line checks: its first problem.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LineFault {
    code: LineCode,
    field: Option<String>,
    reason: String,
}

impl LineFault {
    fn row(code: LineCode, reason: String) -> Self {
        Self {
            code,
            field: None,
            reason,
        }
    }

    fn field(code: LineCode, field: &str, reason: String) -> Self {
        Self {
            code,
            field: Some(field.to_string()),
            reason,
        }
    }
}

/// The per-row rules `schema` and `required_fields` set. Absent when neither
/// is, and then no row is parsed.
#[derive(Clone, Debug)]
struct LineRules {
    schema: Option<BTreeMap<String, FieldSpec>>,
    required: Vec<String>,
}

impl LineRules {
    fn from_args(args: &CheckJsonlLinesArgs) -> Option<Self> {
        (args.schema.is_some() || !args.required_fields.is_empty()).then(|| Self {
            schema: args.schema.clone(),
            required: args.required_fields.clone(),
        })
    }

    /// Check one non-blank row, stopping at its first problem.
    fn check(&self, row: &str) -> Result<(), LineFault> {
        use serde_json::Value;
        let value: Value = serde_json::from_str(row).map_err(|e| {
            // serde_json ends its message with "at line 1 column N"; within a
            // single row only the column means anything.
            let msg = e.to_string();
            let msg = msg
                .rsplit_once(" at line ")
                .map_or(msg.as_str(), |(m, _)| m);
            LineFault::row(
                LineCode::InvalidJson,
                format!("not valid JSON: {msg} (column {})", e.column()),
            )
        })?;
        let Value::Object(obj) = &value else {
            return Err(LineFault::row(
                LineCode::Schema,
                format!("row is a JSON {}, not an object", json_kind(&value)),
            ));
        };
        let schema_of = |name: &str| self.schema.as_ref().and_then(|s| s.get(name));
        for name in &self.required {
            match obj.get(name) {
                None => {
                    return Err(LineFault::field(
                        LineCode::MissingField,
                        name,
                        "missing".into(),
                    ));
                }
                Some(Value::Null) if !schema_of(name).is_some_and(FieldSpec::allows_null) => {
                    return Err(LineFault::field(
                        LineCode::MissingField,
                        name,
                        "null".into(),
                    ));
                }
                Some(_) => {}
            }
        }
        for (name, spec) in self.schema.iter().flatten() {
            match obj.get(name) {
                None if spec.optional => {}
                None => {
                    return Err(LineFault::field(
                        LineCode::MissingField,
                        name,
                        "missing".into(),
                    ));
                }
                Some(Value::Null) if spec.allows_null() => {}
                Some(v) if spec.ty.matches(v) => {}
                Some(v) => {
                    return Err(LineFault::field(
                        LineCode::Schema,
                        name,
                        format!("expected {}, got {}", spec.ty.name(), json_kind(v)),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// One breaching row as a message names it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ListedBreach {
    line: u64,
    field: Option<String>,
    reason: String,
}

impl std::fmt::Display for ListedBreach {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}", self.line)?;
        if let Some(field) = &self.field {
            write!(f, " field {field:?}")?;
        }
        write!(f, ": {}", self.reason)
    }
}

/// Every breach of one code: how many rows, and the first few of them.
#[derive(Clone, Debug, Default)]
struct BreachGroup {
    count: u64,
    listed: Vec<ListedBreach>,
}

impl BreachGroup {
    /// The earliest breaching row (`listed` always holds at least one once
    /// `count > 0`).
    fn first(&self) -> &ListedBreach {
        &self.listed[0]
    }

    /// "N of R rows: line a …; line b …; K more not listed".
    fn describe(&self, rows: u64) -> String {
        let mut s = format!("{} of {rows} rows: ", self.count);
        let listed: Vec<String> = self.listed.iter().map(ToString::to_string).collect();
        s.push_str(&listed.join("; "));
        let more = self.count - self.listed.len() as u64;
        if more > 0 {
            s.push_str(&format!("; {more} more not listed"));
        }
        s
    }
}

/// What one streaming pass over a JSONL file found.
#[derive(Clone, Debug, Default)]
struct JsonlScan {
    rows: u64,
    groups: [BreachGroup; 3],
}

impl JsonlScan {
    fn record(&mut self, line: u64, fault: LineFault, cap: usize) {
        let group = &mut self.groups[fault.code.index()];
        group.count += 1;
        if group.listed.len() < cap.max(1) {
            group.listed.push(ListedBreach {
                line,
                field: fault.field,
                reason: fault.reason,
            });
        }
    }

    /// The codes that have breaches, in code order.
    fn breached(&self) -> impl Iterator<Item = (LineCode, &BreachGroup)> {
        LineCode::ALL
            .into_iter()
            .map(|c| (c, &self.groups[c.index()]))
            .filter(|(_, g)| g.count > 0)
    }
}

/// Stream `path` once: count its non-blank rows and, with `rules`, check each
/// of them. Memory is one line and its parsed value plus at most `cap` listed
/// breaches per code, whatever the file's size.
///
/// A row is a line that is not empty or whitespace once its `\n` / `\r\n` is
/// removed, which is what the row count has always counted. Without `rules`
/// a line that is not UTF-8 is a read error, as it was before line checks
/// existed; with them it is an invalid-JSON row.
fn scan_jsonl(path: &Path, rules: Option<&LineRules>, cap: usize) -> std::io::Result<JsonlScan> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::with_capacity(64 * 1024, file);
    let mut buf = Vec::new();
    let mut scan = JsonlScan::default();
    let mut line_no = 0u64;
    loop {
        buf.clear();
        if reader.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        line_no += 1;
        let mut line = buf.as_slice();
        if let Some(rest) = line.strip_suffix(b"\n") {
            line = rest.strip_suffix(b"\r").unwrap_or(rest);
        }
        let text = match (std::str::from_utf8(line), rules) {
            (Ok(text), _) => text,
            (Err(_), None) => {
                // The error `BufRead::lines` gave the row count before.
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "stream did not contain valid UTF-8",
                ));
            }
            (Err(_), Some(_)) => {
                scan.rows += 1;
                let fault = LineFault::row(LineCode::InvalidJson, "not valid UTF-8".into());
                scan.record(line_no, fault, cap);
                continue;
            }
        };
        if text.trim().is_empty() {
            continue;
        }
        scan.rows += 1;
        if let Some(rules) = rules
            && let Err(fault) = rules.check(text)
        {
            scan.record(line_no, fault, cap);
        }
    }
    Ok(scan)
}

/// The breach context naming a group's first row.
fn line_context(first: &ListedBreach, rows: u64, breached: u64) -> Vec<(&'static str, String)> {
    let mut context = vec![("line", first.line.to_string())];
    if let Some(field) = &first.field {
        context.push(("field", field.clone()));
    }
    context.push(("rows", rows.to_string()));
    context.push(("breached_rows", breached.to_string()));
    context
}

/// Apply the breach policy to the line checks' findings. `block`: one failure,
/// carrying the code of the earliest breaching line and a message covering
/// every code. `warn`: one breach event per code.
fn handle_line_breaches(
    ctx: &StageContext,
    scan: &JsonlScan,
    on_breach: OnBreach,
) -> Result<(), StageError> {
    let found: Vec<(LineCode, &BreachGroup)> = scan.breached().collect();
    let Some(&(code, group)) = found.iter().min_by_key(|(_, g)| g.first().line) else {
        return Ok(());
    };
    match on_breach {
        OnBreach::Block => {
            let mut detail = group.describe(scan.rows);
            for (other, g) in found.iter().filter(|(c, _)| *c != code) {
                detail.push_str(&format!(
                    " | also {}: {}",
                    other.code(),
                    g.describe(scan.rows)
                ));
            }
            let breached = found.iter().map(|(_, g)| g.count).sum();
            handle_breach(
                ctx,
                CHECK_JSONL,
                code.code(),
                on_breach,
                detail,
                &line_context(group.first(), scan.rows, breached),
            )
        }
        OnBreach::Warn => {
            for (code, g) in found {
                handle_breach(
                    ctx,
                    CHECK_JSONL,
                    code.code(),
                    on_breach,
                    g.describe(scan.rows),
                    &line_context(g.first(), scan.rows, g.count),
                )?;
            }
            Ok(())
        }
    }
}

/// Stage name of the JSONL guard.
pub const CHECK_JSONL: &str = "check_jsonl";

/// Count non-empty lines of the input JSONL and enforce `min_rows`/`max_rows`,
/// then pass the input through unchanged (same path + content hash). The
/// row-count form for Rust callers; see the module docs. It runs
/// [`CheckJsonlLines`] with every line check off.
pub struct CheckJsonl;

#[async_trait]
impl Stage for CheckJsonl {
    const NAME: &'static str = CHECK_JSONL;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = JsonlArtifact;
    type Output = JsonlArtifact;
    type Args = CheckJsonlArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Self::Args,
    ) -> Result<Self::Output, StageError> {
        CheckJsonlLines.run(ctx, input, &args.clone().into()).await
    }
}

/// The stage registered as `check_jsonl`: count the non-blank rows of the
/// input JSONL and enforce `min_rows` / `max_rows`; with `schema` or
/// `required_fields`, also check every row (see the module docs). Then pass
/// the input through unchanged (same path + content hash).
pub struct CheckJsonlLines;

#[async_trait]
impl Stage for CheckJsonlLines {
    const NAME: &'static str = CHECK_JSONL;
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = JsonlArtifact;
    type Output = JsonlArtifact;
    type Args = CheckJsonlLinesArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Self::Args,
    ) -> Result<Self::Output, StageError> {
        // Stream the file on a blocking thread: O(1) memory (never loads the
        // whole file) and off the async runtime, so a multi-GB JSONL neither
        // spikes RAM nor stalls the executor's reactor.
        let path = input.path.clone();
        let rules = LineRules::from_args(args);
        let cap = args
            .max_reported
            .unwrap_or(DEFAULT_MAX_REPORTED)
            .min(MAX_REPORTED_CEILING) as usize;
        let scan = tokio::task::spawn_blocking(move || scan_jsonl(&path, rules.as_ref(), cap))
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!("check_jsonl join: {e}")))?
            .map_err(|e| {
                StageError::Backend(anyhow::anyhow!(
                    "check_jsonl read {}: {e}",
                    input.path.display()
                ))
            })?;
        let rows = scan.rows;

        if let Some(min) = args.min_rows
            && rows < min
        {
            handle_breach(
                ctx,
                CHECK_JSONL,
                "DATA_QUALITY_MIN_ROWS",
                args.on_breach,
                format!("row count {rows} below min_rows {min}"),
                &[("rows", rows.to_string()), ("min_rows", min.to_string())],
            )?;
        }
        if let Some(max) = args.max_rows
            && rows > max
        {
            handle_breach(
                ctx,
                CHECK_JSONL,
                "DATA_QUALITY_MAX_ROWS",
                args.on_breach,
                format!("row count {rows} above max_rows {max}"),
                &[("rows", rows.to_string()), ("max_rows", max.to_string())],
            )?;
        }
        handle_line_breaches(ctx, &scan, args.on_breach)?;
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
    /// The value to test, as a literal arg. The engine does not read an
    /// upstream stage's emitted metrics into it: whoever builds the plan
    /// supplies the number.
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
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
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
    fn identity(&self) -> Option<crate::framework::code_identity::BuildIdentity> {
        Some(crate::framework::code_identity::BuildIdentity::engine())
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        static S: &[(&str, ErasedStageCtor)] = &[
            (CHECK_JSONL, || std::sync::Arc::new(CheckJsonlLines)),
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
mod contract_tests {
    use super::JsonlArtifact;
    use crate::framework::{Artifact, ContentHash};

    #[test]
    fn from_file_hashes_the_bytes_it_points_at() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("rows.jsonl");
        std::fs::write(&path, "{}\n{}\n").unwrap();
        let a = JsonlArtifact::from_file(&path).unwrap();
        assert_eq!(a.primary_path(), path.as_path());
        assert_eq!(a.content_hash(), ContentHash::hash_file(&path).unwrap());
        assert!(JsonlArtifact::from_file(td.path().join("missing.jsonl")).is_err());
    }
}

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
        /// 1-based row written as `{"i":"oops"}` (a string where the schema
        /// demo wants an integer). `None` = every row is `{"i":<n>}`.
        #[serde(default)]
        corrupt_row: Option<u64>,
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
                if args.corrupt_row == Some(i + 1) {
                    body.push_str("{\"i\":\"oops\"}\n");
                } else {
                    body.push_str(&format!("{{\"i\":{i}}}\n"));
                }
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
        let mut check_args = serde_json::json!({ "min_rows": 1000 });
        check_args["on_breach"] = serde_json::to_value(on_breach).unwrap();
        run_demo_with(serde_json::json!({ "rows": rows }), check_args).await
    }

    /// The stage a plan's `check_jsonl` gets: the cookbook's own registration.
    fn registered_check_jsonl() -> Arc<dyn crate::framework::stage::StageDyn> {
        let (_, ctor) = ChecksCookbook
            .stages_erased()
            .iter()
            .find(|(name, _)| *name == CHECK_JSONL)
            .expect("the checks cookbook registers check_jsonl");
        ctor()
    }

    /// The demo DAG with explicit `gen` and `check_jsonl` args.
    async fn run_demo_with(
        gen_args: serde_json::Value,
        check_args: serde_json::Value,
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
        let nodes: Vec<(Arc<dyn StageDyn>, serde_json::Value)> = vec![
            (Arc::new(DemoGen), gen_args),
            (registered_check_jsonl(), check_args),
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

    /// `gen → check_jsonl(schema) → sink` with row 500 of 1000 written as
    /// `{"i":"oops"}`: under `block` the run fails with a
    /// `DATA_QUALITY_SCHEMA` breach naming that line and field, and the sink
    /// never runs.
    #[tokio::test]
    async fn checks_demo_schema_block_skips_downstream() {
        let (res, events, sink_ran) = run_demo_with(
            serde_json::json!({ "rows": 1000, "corrupt_row": 500 }),
            serde_json::json!({ "min_rows": 1000, "schema": { "i": "integer" } }),
        )
        .await;
        assert!(
            res.is_err(),
            "a schema breach under block must fail the run"
        );
        assert_eq!(sink_ran, 0, "downstream sink must be skipped (fail-closed)");
        let breaches = scan_breaches(events);
        let b = breaches
            .iter()
            .find(|b| b.disposition == "block")
            .unwrap_or_else(|| panic!("no block breach recorded: {breaches:?}"));
        assert_eq!(b.code, "DATA_QUALITY_SCHEMA");
        assert_eq!(b.category(), "DataQuality");
        assert!(
            b.detail
                .contains("line 500 field \"i\": expected integer, got string"),
            "the breach names the line and field: {}",
            b.detail
        );
        assert!(
            !b.detail.contains("oops"),
            "row contents never reach a breach"
        );
        let report = render_report(&breaches);
        assert!(
            report.contains("[DataQuality] BLOCK @check_jsonl"),
            "{report}"
        );
        assert!(report.contains("DATA_QUALITY_SCHEMA"), "{report}");
    }

    /// The same DAG over clean rows passes the schema and the sink runs.
    #[tokio::test]
    async fn checks_demo_schema_pass_runs_downstream() {
        let (res, events, sink_ran) = run_demo_with(
            serde_json::json!({ "rows": 1000 }),
            serde_json::json!({
                "min_rows": 1000,
                "required_fields": ["i"],
                "schema": { "i": "integer" },
            }),
        )
        .await;
        assert!(
            res.is_ok(),
            "a passing schema must not fail the run: {res:?}"
        );
        assert_eq!(sink_ran, 1, "downstream sink runs when the check passes");
        assert!(scan_breaches(events).is_empty());
    }

    /// Under `warn` the schema breach is recorded and the sink still runs.
    #[tokio::test]
    async fn checks_demo_schema_warn_continues_and_records() {
        let (res, events, sink_ran) = run_demo_with(
            serde_json::json!({ "rows": 1000, "corrupt_row": 7 }),
            serde_json::json!({ "schema": { "i": "integer" }, "on_breach": "warn" }),
        )
        .await;
        assert!(res.is_ok(), "a warn breach must NOT fail the run: {res:?}");
        assert_eq!(sink_ran, 1, "downstream sink runs under a warn breach");
        let breaches = scan_breaches(events);
        assert!(
            breaches.iter().any(|b| b.disposition == "warn"
                && b.code == "DATA_QUALITY_SCHEMA"
                && b.detail.contains("line 7 field \"i\"")),
            "the advisory schema breach is on the record: {breaches:?}"
        );
    }
}

#[cfg(test)]
mod line_check_tests {
    //! `check_jsonl`'s line checks (`schema`, `required_fields`,
    //! `max_reported`), one test per breach kind, plus the guarantees the
    //! row-count-only stage already gave: pass-through byte identity, warn,
    //! and unchanged behaviour when no line check is asked for.
    use super::*;
    use crate::framework::stage::StageDyn;
    use std::sync::Arc;

    fn write(dir: &Path, body: &[u8]) -> JsonlArtifact {
        let path = dir.join("rows.jsonl");
        std::fs::write(&path, body).unwrap();
        JsonlArtifact::from_file(&path).unwrap()
    }

    fn args(v: serde_json::Value) -> CheckJsonlLinesArgs {
        serde_json::from_value(v).expect("test args decode")
    }

    /// Run `check_jsonl` over `body` and return the block failure it raised.
    async fn block_failure(body: &[u8], a: serde_json::Value) -> StageFailure {
        let tmp = tempfile::tempdir().unwrap();
        let art = write(tmp.path(), body);
        let ctx = StageContext::for_test(tmp.path().to_path_buf(), tmp.path().to_path_buf());
        let err = CheckJsonlLines.run(&ctx, art, &args(a)).await.unwrap_err();
        let StageError::Backend(e) = &err else {
            panic!("expected a Backend failure, got {err:?}");
        };
        StageFailure::try_extract(e)
            .unwrap_or_else(|| panic!("expected a checks StageFailure, got {e:#}"))
            .clone()
    }

    /// Run `check_jsonl` over `body`, which must pass; return the input
    /// handle, the output handle and the breach events emitted.
    async fn passes(
        body: &[u8],
        a: serde_json::Value,
    ) -> (JsonlArtifact, JsonlArtifact, Vec<StageEvent>) {
        let tmp = tempfile::tempdir().unwrap();
        let art = write(tmp.path(), body);
        let ctx = StageContext::for_test(tmp.path().to_path_buf(), tmp.path().to_path_buf());
        let mut rx = ctx.status_tx.subscribe();
        let out = CheckJsonlLines
            .run(&ctx, art.clone(), &args(a))
            .await
            .expect("check passes");
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        (art, out, events)
    }

    fn ctx_value<'a>(f: &'a StageFailure, key: &str) -> Option<&'a str> {
        f.context
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    #[tokio::test]
    async fn missing_required_field_blocks_naming_line_and_field() {
        let body = b"{\"id\":1,\"text\":\"a\"}\n{\"id\":2}\n{\"id\":3,\"text\":\"c\"}\n";
        let f = block_failure(
            body,
            serde_json::json!({ "required_fields": ["id", "text"] }),
        )
        .await;
        assert_eq!(f.domain, "checks");
        assert_eq!(f.code, "DATA_QUALITY_MISSING_FIELD");
        assert_eq!(f.severity, Severity::Major);
        assert_eq!(ctx_value(&f, "line"), Some("2"));
        assert_eq!(ctx_value(&f, "field"), Some("text"));
        assert_eq!(ctx_value(&f, "rows"), Some("3"));
        assert_eq!(ctx_value(&f, "breached_rows"), Some("1"));
        assert_eq!(f.message, "1 of 3 rows: line 2 field \"text\": missing");
    }

    #[tokio::test]
    async fn null_required_field_is_missing_unless_the_schema_allows_null() {
        let body = b"{\"id\":1,\"label\":null}\n";
        let f = block_failure(body, serde_json::json!({ "required_fields": ["label"] })).await;
        assert_eq!(f.code, "DATA_QUALITY_MISSING_FIELD");
        assert_eq!(f.message, "1 of 1 rows: line 1 field \"label\": null");

        // The schema says null is fine for `label`: present-but-null passes.
        let nullable = serde_json::json!({
            "required_fields": ["label"],
            "schema": { "label": { "type": "string", "nullable": true } },
        });
        passes(body, nullable).await;
        // So does a field whose type is `null`.
        passes(
            body,
            serde_json::json!({ "required_fields": ["label"], "schema": { "label": "null" } }),
        )
        .await;
    }

    #[tokio::test]
    async fn wrong_type_is_a_schema_breach_naming_line_field_and_types() {
        let body = b"{\"id\":1,\"score\":0.5}\n{\"id\":\"2\",\"score\":0.25}\n";
        let schema = serde_json::json!({ "schema": { "id": "integer", "score": "number" } });
        let f = block_failure(body, schema).await;
        assert_eq!(f.code, "DATA_QUALITY_SCHEMA");
        assert_eq!(ctx_value(&f, "line"), Some("2"));
        assert_eq!(ctx_value(&f, "field"), Some("id"));
        assert_eq!(
            f.message,
            "1 of 2 rows: line 2 field \"id\": expected integer, got string"
        );
    }

    #[tokio::test]
    async fn a_null_schema_field_is_a_type_breach_unless_nullable() {
        let body = b"{\"id\":null}\n";
        let f = block_failure(body, serde_json::json!({ "schema": { "id": "integer" } })).await;
        assert_eq!(f.code, "DATA_QUALITY_SCHEMA");
        assert_eq!(
            f.message,
            "1 of 1 rows: line 1 field \"id\": expected integer, got null"
        );
        passes(
            body,
            serde_json::json!({ "schema": { "id": { "type": "integer", "nullable": true } } }),
        )
        .await;
    }

    #[tokio::test]
    async fn an_absent_schema_field_is_missing_unless_optional() {
        let body = b"{\"id\":1}\n";
        let f = block_failure(body, serde_json::json!({ "schema": { "tag": "string" } })).await;
        assert_eq!(f.code, "DATA_QUALITY_MISSING_FIELD");
        assert_eq!(f.message, "1 of 1 rows: line 1 field \"tag\": missing");
        passes(
            body,
            serde_json::json!({ "schema": { "tag": { "type": "string", "optional": true } } }),
        )
        .await;
        // `required_fields` still requires a field the schema calls optional.
        let both = serde_json::json!({
            "required_fields": ["tag"],
            "schema": { "tag": { "type": "string", "optional": true } },
        });
        assert_eq!(
            block_failure(body, both).await.code,
            "DATA_QUALITY_MISSING_FIELD"
        );
    }

    #[tokio::test]
    async fn invalid_json_line_blocks_naming_the_line() {
        let body = b"{\"id\":1}\n{\"id\":2,\n{\"id\":3}\n";
        let f = block_failure(body, serde_json::json!({ "schema": {} })).await;
        assert_eq!(f.code, "DATA_QUALITY_INVALID_JSON");
        assert_eq!(ctx_value(&f, "line"), Some("2"));
        assert_eq!(
            ctx_value(&f, "field"),
            None,
            "a parse failure names no field"
        );
        assert!(
            f.message
                .starts_with("1 of 3 rows: line 2: not valid JSON: "),
            "{}",
            f.message
        );
        // `required_fields` alone also parses every row.
        let f = block_failure(body, serde_json::json!({ "required_fields": ["id"] })).await;
        assert_eq!(f.code, "DATA_QUALITY_INVALID_JSON");
    }

    #[tokio::test]
    async fn a_line_that_is_not_utf8_is_invalid_json_under_line_checks() {
        let body = b"{\"id\":1}\n{\"id\":\"\xff\"}\n";
        let f = block_failure(body, serde_json::json!({ "required_fields": ["id"] })).await;
        assert_eq!(f.code, "DATA_QUALITY_INVALID_JSON");
        assert_eq!(f.message, "1 of 2 rows: line 2: not valid UTF-8");
    }

    #[tokio::test]
    async fn trailing_content_after_a_row_is_invalid_json() {
        let f = block_failure(
            b"{\"id\":1} {\"id\":2}\n",
            serde_json::json!({ "schema": {} }),
        )
        .await;
        assert_eq!(f.code, "DATA_QUALITY_INVALID_JSON");
    }

    #[tokio::test]
    async fn a_row_that_is_not_an_object_is_a_schema_breach() {
        let body = b"{\"id\":1}\n[1,2]\n";
        let f = block_failure(body, serde_json::json!({ "schema": {} })).await;
        assert_eq!(f.code, "DATA_QUALITY_SCHEMA");
        assert_eq!(
            f.message,
            "1 of 2 rows: line 2: row is a JSON array, not an object"
        );
        let f = block_failure(b"42\n", serde_json::json!({ "required_fields": ["id"] })).await;
        assert_eq!(
            f.message,
            "1 of 1 rows: line 1: row is a JSON integer, not an object"
        );
    }

    /// Blank lines are skipped by the row count and the line checks alike,
    /// and line numbers stay physical (blank lines counted), CRLF included.
    #[tokio::test]
    async fn blank_lines_are_not_rows_and_line_numbers_stay_physical() {
        let body = b"{\"id\":1}\r\n\n   \t\n{\"id\":2}\r\n\n";
        let clean = serde_json::json!({
            "min_rows": 2, "max_rows": 2, "required_fields": ["id"], "schema": { "id": "integer" },
        });
        passes(body, clean).await;

        let body = b"{\"id\":1}\n\n\n{\"id\":true}\n";
        let f = block_failure(body, serde_json::json!({ "schema": { "id": "integer" } })).await;
        assert_eq!(ctx_value(&f, "line"), Some("4"), "line 4, not row 2");
        assert_eq!(ctx_value(&f, "rows"), Some("2"));

        // A file with no trailing newline still checks its last row.
        let f = block_failure(
            b"{\"id\":1}\n{}",
            serde_json::json!({ "required_fields": ["id"] }),
        )
        .await;
        assert_eq!(ctx_value(&f, "line"), Some("2"));
    }

    #[test]
    fn integer_and_number_follow_json_schema() {
        let t = |ty: FieldType, v: serde_json::Value| ty.matches(&v);
        assert!(t(FieldType::Integer, serde_json::json!(3)));
        assert!(t(FieldType::Integer, serde_json::json!(-3)));
        assert!(t(FieldType::Integer, serde_json::json!(u64::MAX)));
        assert!(t(FieldType::Integer, serde_json::json!(3.0)));
        assert!(!t(FieldType::Integer, serde_json::json!(3.5)));
        assert!(!t(FieldType::Integer, serde_json::json!("3")));
        assert!(t(FieldType::Number, serde_json::json!(3)));
        assert!(t(FieldType::Number, serde_json::json!(3.5)));
        assert!(!t(FieldType::Number, serde_json::json!(null)));
        assert!(t(FieldType::Object, serde_json::json!({})));
        assert!(t(FieldType::Array, serde_json::json!([])));
        assert!(t(FieldType::Boolean, serde_json::json!(false)));
        assert!(t(FieldType::Null, serde_json::json!(null)));
        assert!(!t(FieldType::String, serde_json::json!(null)));
    }

    /// A passing line check returns the input BYTE-IDENTICAL: same path and
    /// content hash, so the downstream cache key is unchanged.
    #[tokio::test]
    async fn a_passing_schema_check_passes_through_byte_identical() {
        let body = b"{\"id\":1,\"text\":\"a\",\"meta\":{},\"tags\":[]}\n";
        let a = serde_json::json!({
            "min_rows": 1,
            "required_fields": ["id", "text"],
            "schema": { "id": "integer", "text": "string", "meta": "object", "tags": "array" },
        });
        let tmp = tempfile::tempdir().unwrap();
        let input = write(tmp.path(), body);
        let ctx = StageContext::for_test(tmp.path().to_path_buf(), tmp.path().to_path_buf());
        let mut rx = ctx.status_tx.subscribe();
        let out = CheckJsonlLines
            .run(&ctx, input.clone(), &args(a))
            .await
            .unwrap();
        assert_eq!(out.path, input.path, "passthrough keeps the path");
        assert_eq!(out.content_hash, input.content_hash, "and the content hash");
        assert_eq!(
            std::fs::read(&out.path).unwrap(),
            body,
            "and the bytes on disk"
        );
        assert!(rx.try_recv().is_err(), "a pass records no breach");
    }

    /// Under `warn` each line-breach code is its own breach event, and the
    /// input passes through unchanged.
    #[tokio::test]
    async fn warn_records_one_event_per_code_and_passes_through() {
        let body = b"not json\n{\"id\":\"x\"}\n{}\n{\"id\":1}\n";
        let a = serde_json::json!({ "schema": { "id": "integer" }, "on_breach": "warn" });
        let (input, out, events) = passes(body, a).await;
        assert_eq!(
            out.content_hash, input.content_hash,
            "warn still passes through"
        );
        let breaches = scan_breaches(events);
        let codes: Vec<&str> = breaches.iter().map(|b| b.code.as_str()).collect();
        assert_eq!(
            codes,
            [
                "DATA_QUALITY_INVALID_JSON",
                "DATA_QUALITY_MISSING_FIELD",
                "DATA_QUALITY_SCHEMA"
            ]
        );
        assert!(breaches.iter().all(|b| b.disposition == "warn"));
        assert!(breaches[1].detail.contains("line 3 field \"id\": missing"));
        assert!(
            breaches[2]
                .detail
                .contains("line 2 field \"id\": expected integer")
        );
    }

    /// Under `block` the failure carries the earliest line's code and a
    /// message covering every code found.
    #[tokio::test]
    async fn block_reports_the_earliest_line_and_summarizes_every_code() {
        let body = b"{\"id\":1}\n{}\n{\"id\":\"x\"}\nnot json\n";
        let f = block_failure(body, serde_json::json!({ "schema": { "id": "integer" } })).await;
        assert_eq!(
            f.code, "DATA_QUALITY_MISSING_FIELD",
            "line 2 is the first breach"
        );
        assert_eq!(ctx_value(&f, "breached_rows"), Some("3"));
        assert!(
            f.message
                .starts_with("1 of 4 rows: line 2 field \"id\": missing")
        );
        assert!(
            f.message
                .contains(" | also DATA_QUALITY_INVALID_JSON: 1 of 4 rows: line 4: ")
        );
        assert!(
            f.message
                .contains(" | also DATA_QUALITY_SCHEMA: 1 of 4 rows: line 3 field")
        );
    }

    /// `max_reported` caps how many rows each code names; every breaching
    /// row is still counted, and the first is always named.
    #[tokio::test]
    async fn max_reported_caps_the_listing_not_the_count() {
        let mut body = String::new();
        for i in 1..=25 {
            body.push_str(&format!("{{\"n\":{i}}}\n"));
        }
        let f = block_failure(
            body.as_bytes(),
            serde_json::json!({ "required_fields": ["id"] }),
        )
        .await;
        assert_eq!(ctx_value(&f, "breached_rows"), Some("25"));
        assert_eq!(
            f.message.matches("field \"id\": missing").count(),
            10,
            "default 10"
        );
        assert!(f.message.ends_with("; 15 more not listed"), "{}", f.message);

        let a = serde_json::json!({ "required_fields": ["id"], "max_reported": 3 });
        let f = block_failure(body.as_bytes(), a).await;
        assert_eq!(f.message.matches("missing").count(), 3);
        assert!(f.message.ends_with("; 22 more not listed"));

        let a = serde_json::json!({ "required_fields": ["id"], "max_reported": 1 });
        assert_eq!(
            block_failure(body.as_bytes(), a).await.message,
            "25 of 25 rows: line 1 field \"id\": missing; 24 more not listed"
        );

        let a = serde_json::json!({ "required_fields": ["id"], "max_reported": 0 });
        let f = block_failure(body.as_bytes(), a).await;
        assert!(
            f.message
                .starts_with("25 of 25 rows: line 1 field \"id\": missing; 24 more")
        );
    }

    /// However large `max_reported` is, what a scan holds stays bounded, and
    /// the count still covers every row.
    #[tokio::test]
    async fn max_reported_has_a_ceiling() {
        let body = "{}\n".repeat(MAX_REPORTED_CEILING as usize + 5);
        let a = serde_json::json!({ "required_fields": ["id"], "max_reported": u32::MAX });
        let f = block_failure(body.as_bytes(), a).await;
        assert_eq!(
            f.message.matches("missing").count(),
            MAX_REPORTED_CEILING as usize
        );
        assert!(f.message.ends_with("; 5 more not listed"), "{}", f.message);
        assert_eq!(ctx_value(&f, "breached_rows"), Some("1005"));
    }

    /// An empty file has no rows to break a rule: the line checks pass it and
    /// only `min_rows` can refuse it.
    #[tokio::test]
    async fn an_empty_file_is_refused_only_by_min_rows() {
        let rules = serde_json::json!({ "required_fields": ["id"], "schema": { "id": "integer" } });
        let (_, _, events) = passes(b"", rules.clone()).await;
        assert!(events.is_empty());
        passes(b"\n  \n", rules).await;
        let a = serde_json::json!({ "min_rows": 1, "required_fields": ["id"] });
        assert_eq!(block_failure(b"", a).await.code, "DATA_QUALITY_MIN_ROWS");
    }

    /// The report shows the new codes as DataQuality breaches with their line.
    #[tokio::test]
    async fn report_renders_line_breaches() {
        let body = b"{}\n";
        let (_, _, events) = passes(
            body,
            serde_json::json!({ "required_fields": ["id"], "on_breach": "warn" }),
        )
        .await;
        let breaches = scan_breaches(events);
        assert_eq!(breaches.len(), 1);
        assert_eq!(breaches[0].category(), "DataQuality");
        let report = render_report(&breaches);
        assert_eq!(
            report,
            "[DataQuality] WARN @check_jsonl (node 0) DATA_QUALITY_MISSING_FIELD — \
             1 of 1 rows: line 1 field \"id\": missing\n"
        );
    }

    /// A breach message names line numbers, arg field names and JSON type
    /// names only: never a row's contents.
    #[tokio::test]
    async fn breach_messages_carry_no_row_content() {
        let body = b"{\"id\":\"SECRET-PATIENT-NAME\",\"note\":\"SECRET\"}\n{\"SECRET\n";
        let f = block_failure(body, serde_json::json!({ "schema": { "id": "integer" } })).await;
        assert!(!f.message.contains("SECRET"), "{}", f.message);
        assert!(f.context.iter().all(|(_, v)| !v.contains("SECRET")));
    }

    // ── row-count-only args behave exactly as before ────────────────

    /// Without `schema`/`required_fields` no row is parsed: rows that are
    /// not JSON still count, and the bounds decide alone.
    #[tokio::test]
    async fn row_count_only_args_do_not_parse_rows() {
        let body = b"not json\n[1]\n{\"a\":\n\n";
        let (_, _, events) =
            passes(body, serde_json::json!({ "min_rows": 3, "max_rows": 3 })).await;
        assert!(events.is_empty());
        let f = block_failure(body, serde_json::json!({ "min_rows": 4 })).await;
        assert_eq!(f.code, "DATA_QUALITY_MIN_ROWS");
        assert_eq!(f.message, "row count 3 below min_rows 4");
    }

    /// Row-count-only args over a file that is not UTF-8 fail as a read error,
    /// outside the checks domain, as `BufRead::lines` made them fail before.
    #[tokio::test]
    async fn row_count_only_args_still_refuse_non_utf8_as_a_read_error() {
        let tmp = tempfile::tempdir().unwrap();
        let art = write(tmp.path(), b"{}\n\xff\n");
        let ctx = StageContext::for_test(tmp.path().to_path_buf(), tmp.path().to_path_buf());
        let err = CheckJsonlLines
            .run(&ctx, art, &args(serde_json::json!({ "min_rows": 1 })))
            .await
            .unwrap_err();
        let StageError::Backend(e) = &err else {
            panic!("expected a Backend read error, got {err:?}");
        };
        assert!(
            StageFailure::try_extract(e).is_none(),
            "not a checks breach"
        );
        assert!(
            e.to_string()
                .ends_with("stream did not contain valid UTF-8"),
            "{e}"
        );
    }

    /// The published stage: an exhaustive `CheckJsonlArgs` literal (which is
    /// why that struct cannot grow) still compiles, and it counts rows and
    /// nothing else, under the same identity as the registered stage.
    #[tokio::test]
    async fn the_published_row_count_stage_is_unchanged() {
        assert_eq!(
            <CheckJsonl as Stage>::NAME,
            <CheckJsonlLines as Stage>::NAME
        );
        assert_eq!(
            <CheckJsonl as Stage>::SCHEMA,
            <CheckJsonlLines as Stage>::SCHEMA
        );
        let tmp = tempfile::tempdir().unwrap();
        let art = write(tmp.path(), b"not json\n[1]\n");
        let ctx = StageContext::for_test(tmp.path().to_path_buf(), tmp.path().to_path_buf());
        let bounds = |min_rows| CheckJsonlArgs {
            min_rows: Some(min_rows),
            max_rows: None,
            on_breach: OnBreach::Block,
        };
        let out = CheckJsonl.run(&ctx, art.clone(), &bounds(2)).await.unwrap();
        assert_eq!(out.content_hash, art.content_hash);
        let err = CheckJsonl.run(&ctx, art, &bounds(3)).await.unwrap_err();
        let StageError::Backend(e) = &err else {
            panic!("expected a Backend failure, got {err:?}");
        };
        let f = StageFailure::try_extract(e).expect("a checks failure");
        assert_eq!(f.code, "DATA_QUALITY_MIN_ROWS");
        assert_eq!(f.message, "row count 2 below min_rows 3");
        assert_eq!(
            serde_json::to_value(bounds(2)).unwrap(),
            serde_json::json!({ "min_rows": 2, "max_rows": null, "on_breach": "block" })
        );
        let lines = CheckJsonlLinesArgs::from(bounds(2));
        assert_eq!(lines.min_rows, Some(2));
        assert!(
            LineRules::from_args(&lines).is_none(),
            "no line check is armed"
        );
    }

    /// Row-count-only args decode with every line check off, and serialize
    /// to exactly the three keys they always had.
    #[test]
    fn row_count_only_args_keep_their_shape() {
        let a = args(serde_json::json!({ "min_rows": 5, "max_rows": 9, "on_breach": "warn" }));
        assert!(a.schema.is_none() && a.required_fields.is_empty() && a.max_reported.is_none());
        assert!(LineRules::from_args(&a).is_none(), "no line check is armed");
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::json!({ "min_rows": 5, "max_rows": 9, "on_breach": "warn" })
        );
        assert_eq!(
            serde_json::to_value(CheckJsonlLinesArgs::default()).unwrap(),
            serde_json::json!({ "min_rows": null, "max_rows": null, "on_breach": "block" })
        );
    }

    /// `FieldSpec`'s own `Deserialize` takes both forms and refuses as the
    /// args decode does.
    #[test]
    fn field_spec_decodes_both_forms() {
        let spec = |v| serde_json::from_value::<FieldSpec>(v);
        assert_eq!(
            spec(serde_json::json!("boolean")).unwrap(),
            FieldSpec::new(FieldType::Boolean)
        );
        assert_eq!(
            spec(serde_json::json!({ "type": "array", "optional": true })).unwrap(),
            FieldSpec::new(FieldType::Array).allow_missing()
        );
        let e = spec(serde_json::json!("bool")).unwrap_err().to_string();
        assert!(e.contains("unknown type \"bool\""), "{e}");
    }

    #[test]
    fn line_args_round_trip_in_their_long_form() {
        let a = args(serde_json::json!({
            "required_fields": ["id"],
            "schema": { "id": "integer", "label": { "type": "string", "nullable": true } },
            "max_reported": 4,
        }));
        let mut want = BTreeMap::new();
        want.insert("id".to_string(), FieldSpec::new(FieldType::Integer));
        want.insert(
            "label".to_string(),
            FieldSpec::new(FieldType::String).allow_null(),
        );
        assert_eq!(a.schema.as_ref(), Some(&want));
        let back = args(serde_json::to_value(&a).unwrap());
        assert_eq!(back.schema, a.schema);
        assert_eq!(back.required_fields, a.required_fields);
        assert_eq!(back.max_reported, Some(4));
        assert_eq!(
            serde_json::to_value(&a).unwrap()["schema"]["label"],
            serde_json::json!({ "type": "string", "nullable": true })
        );
    }

    // ── args are refused when the plan is built ─────────────────────

    /// The plan-build check (`StageDyn::check_args`, which `plan check` and
    /// every `.toml`/`.json`/`.star` compile run) refuses bad line-check args
    /// with a message naming the problem.
    #[test]
    fn bad_line_args_are_refused_at_plan_build() {
        let stage: Arc<dyn StageDyn> = Arc::new(CheckJsonlLines);
        let refused = |a: serde_json::Value| -> String {
            match stage.check_args(&a) {
                Ok(()) => panic!("args {a} were accepted"),
                Err(e) => e.to_string(),
            }
        };
        let e = refused(serde_json::json!({ "schema": { "id": "strng" } }));
        assert!(
            e.contains("schema field \"id\": unknown type \"strng\"; expected one of string, number, integer, boolean, object, array, null"),
            "{e}"
        );
        let e = refused(serde_json::json!({ "schema": { "": "string" } }));
        assert!(e.contains("schema: a field name is empty"), "{e}");
        let e = refused(serde_json::json!({ "required_fields": ["id", ""] }));
        assert!(e.contains("required_fields: a field name is empty"), "{e}");
        let e = refused(serde_json::json!({ "required_fields": ["id", "id"] }));
        assert!(e.contains("required_fields: \"id\" is listed twice"), "{e}");
        let e = refused(
            serde_json::json!({ "schema": { "id": { "type": "integer", "nulable": true } } }),
        );
        assert!(e.contains("unknown key \"nulable\""), "{e}");
        let e = refused(serde_json::json!({ "schema": { "id": { "nullable": true } } }));
        assert!(
            e.contains("schema field \"id\": a rule object needs a \"type\""),
            "{e}"
        );
        let e = refused(
            serde_json::json!({ "schema": { "id": { "type": "integer", "optional": "yes" } } }),
        );
        assert!(e.contains("\"optional\" must be true or false"), "{e}");
        let e = refused(serde_json::json!({ "schema": { "id": 3 } }));
        assert!(e.contains("not integer"), "{e}");
        let e = refused(serde_json::json!({ "schema": ["id"] }));
        assert!(e.contains("check_jsonl"), "{e}");
        let e = refused(serde_json::json!({ "required_fields": "id" }));
        assert!(e.contains("required_fields"), "{e}");
        let e = refused(serde_json::json!({ "max_reported": -1 }));
        assert!(e.contains("check_jsonl"), "{e}");
        let e = refused(serde_json::json!({ "requird_fields": ["id"] }));
        assert!(
            e.contains("requird_fields"),
            "an unknown arg is still refused: {e}"
        );

        // And good args pass the same check, as do row-count-only ones.
        for ok in [
            serde_json::json!({ "min_rows": 1000 }),
            serde_json::json!({}),
            serde_json::Value::Null,
            serde_json::json!({ "required_fields": null, "schema": null, "max_reported": null }),
            serde_json::json!({
                "min_rows": 1, "max_rows": 10, "on_breach": "warn", "max_reported": 0,
                "required_fields": ["id"],
                "schema": { "id": "integer", "x": { "type": "null" },
                            "y": { "type": "array", "optional": true, "nullable": false } },
            }),
        ] {
            stage
                .check_args(&ok)
                .unwrap_or_else(|e| panic!("{ok} refused: {e}"));
        }
    }

    /// The same refusal reaches a PlanSpec compile before any node runs.
    #[test]
    fn a_plan_spec_with_a_bad_schema_does_not_compile() {
        use crate::framework::plan_spec::{PlanSpec, SpecNode};
        let mut reg = Registry::new();
        register(&mut reg);
        let node = |args: serde_json::Value| SpecNode {
            stage: CHECK_JSONL.into(),
            args,
            retry: None,
            timeout: None,
            priority: None,
            pure: false,
        };
        let spec = |args| PlanSpec {
            name: "bad-schema".into(),
            nodes: vec![node(args)],
            edges: Vec::new(),
            condition_gates: Vec::new(),
            expansions: Vec::new(),
            version: 1,
        };
        let err = spec(serde_json::json!({ "schema": { "id": "int" } }))
            .compile(&reg)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(err.contains("node 0 stage 'check_jsonl'"), "{err}");
        assert!(
            err.contains("schema field \"id\": unknown type \"int\""),
            "{err}"
        );
    }

    /// The args JSON schema advertises the new arguments, so console and
    /// SDK tooling see them, and a rule may be a name or an object.
    #[test]
    fn args_schema_lists_the_line_arguments() {
        let (_, ctor) = ChecksCookbook
            .stages_erased()
            .iter()
            .find(|(name, _)| *name == CHECK_JSONL)
            .expect("the checks cookbook registers check_jsonl");
        let schema = ctor().args_schema();
        let defs = &schema["definitions"];
        let props = &defs["CheckJsonlLinesArgs"]["properties"];
        for key in [
            "min_rows",
            "max_rows",
            "on_breach",
            "schema",
            "required_fields",
            "max_reported",
        ] {
            assert!(props.get(key).is_some(), "args schema lacks {key}: {props}");
        }
        assert!(
            defs["CheckJsonlLinesArgs"].get("required").is_none(),
            "every argument stays optional"
        );
        let rule = defs["FieldSpec"].to_string();
        assert!(
            rule.contains("anyOf"),
            "a rule is a name or an object: {rule}"
        );
    }

    #[test]
    fn the_domain_lists_the_line_codes() {
        let mut reg = Registry::new();
        register(&mut reg);
        let codes: Vec<&str> = reg
            .all_error_domains()
            .filter(|d| d.name == "checks")
            .flat_map(|d| d.codes.iter().map(|(c, _)| *c))
            .collect();
        for code in [
            "DATA_QUALITY_MIN_ROWS",
            "DATA_QUALITY_MAX_ROWS",
            "DATA_QUALITY_INVALID_JSON",
            "DATA_QUALITY_MISSING_FIELD",
            "DATA_QUALITY_SCHEMA",
            "ASSERT_FAILED",
        ] {
            assert!(codes.contains(&code), "{code} not in {codes:?}");
        }
    }

    /// Peak memory stays flat as the file grows: the scan holds one line at a
    /// time. Run it explicitly (it writes about 140 MB to the temp dir, and
    /// removes it):
    /// `cargo test --lib checks::line_check_tests::large_file_memory_stays_flat
    /// -- --ignored --exact --nocapture`. `BLUT_CHECKS_BIG_LINES` overrides the
    /// line count (default 2,000,000).
    #[cfg(target_os = "linux")]
    #[tokio::test]
    #[ignore = "writes ~140 MB of JSONL to the temp dir; run explicitly"]
    async fn large_file_memory_stays_flat() {
        use std::io::Write as _;
        fn status_kib(key: &str) -> u64 {
            let status = std::fs::read_to_string("/proc/self/status").unwrap();
            let line = status.lines().find(|l| l.starts_with(key)).unwrap();
            line.split_whitespace().nth(1).unwrap().parse().unwrap()
        }
        let lines: u64 = std::env::var("BLUT_CHECKS_BIG_LINES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2_000_000);
        let tmp = tempfile::tempdir().unwrap();
        let ctx = StageContext::for_test(tmp.path().to_path_buf(), tmp.path().to_path_buf());
        let mut peaks = Vec::new();
        for n in [lines / 100, lines] {
            let path = tmp.path().join(format!("big-{n}.jsonl"));
            let mut w = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            for i in 0..n {
                writeln!(
                    w,
                    "{{\"id\":{i},\"text\":\"row {i} of the generated file\",\"score\":0.5}}"
                )
                .unwrap();
            }
            w.flush().unwrap();
            drop(w);
            let bytes = std::fs::metadata(&path).unwrap().len();
            let art = JsonlArtifact::from_file(&path).unwrap();
            let a = args(serde_json::json!({
                "min_rows": n,
                "max_rows": n,
                "required_fields": ["id", "text"],
                "schema": { "id": "integer", "text": "string", "score": "number" },
            }));
            // Reset the peak (VmHWM) so it measures this check alone.
            std::fs::write("/proc/self/clear_refs", "5").unwrap();
            let before = status_kib("VmRSS:");
            let started = std::time::Instant::now();
            CheckJsonlLines
                .run(&ctx, art, &a)
                .await
                .expect("clean rows pass");
            let peak = status_kib("VmHWM:");
            println!(
                "check_jsonl lines={n} bytes={bytes} rss_before={before} KiB peak={peak} KiB \
                 elapsed={:.2?}",
                started.elapsed()
            );
            peaks.push(peak);
            std::fs::remove_file(&path).unwrap();
        }
        let growth = peaks[1].saturating_sub(peaks[0]);
        assert!(
            growth < 16 * 1024,
            "peak RSS grew {growth} KiB from {} to {lines} lines",
            lines / 100
        );
    }
}
