//! `Stage` trait + `StageDyn` erased shadow + `StageContext` +
//! `ErasedArtifact`.
//!
//! Architectural keystone of the v2 framework. Reading order:
//!
//!   1. `Stage` — typed user-facing trait. Concrete stages (in
//!      `stages/`) implement this. Three associated types
//!      (`Input`, `Output`, `Args`) and three constants (`NAME`,
//!      `SCHEMA`, `RESOURCES`).
//!   2. `StageDyn` — object-safe shadow trait used by `Plan` to
//!      store stages as `Box<dyn StageDyn>`. Erases the typed
//!      I/O at JSON boundaries: `run_erased` accepts an
//!      `ErasedArtifact` (kind tag + JSON), deserializes into the
//!      typed `Input`, runs, re-serializes the typed `Output`.
//!      Type errors at this boundary surface as `StageError::
//!      KindMismatch` or `InputDeserialize`.
//!   3. `impl<S: Stage> StageDyn for S` — blanket impl that wires
//!      the conversion. Users never write this. Adding a new
//!      stage = `impl Stage for ...` and the blanket impl makes
//!      it executable through the framework.
//!   4. `StageContext` — bundle of execution context (job_dir,
//!      stage_dir, status broadcast, cancellation, cache).
//!   5. `ErasedArtifact` — `(kind, schema, json)` triple flowing
//!      across erased edges. Has typed `From`/`TryInto` helpers
//!      for the conversion symmetric.
//!
//! Why erased dispatch? The `Plan` builder produces a
//! `Vec<Box<dyn StageDyn>>` regardless of the typed lattice the
//! user composed. Walking that vector to execute the plan needs
//! a single `run_erased` shape. The cost (one JSON round-trip per
//! edge) is dwarfed by stage runtime (typically minutes-to-hours).
//! See `unified-launching-quill.md` "FP vectors" — this is item 1.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::framework::artifact::{Artifact, ContentHash};
use crate::framework::cache::CacheHandle;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::status::StageEvent;

/// Erased artifact: a kind-tagged binary blob that passes between
/// stages at the `StageDyn` boundary. The `kind` field MUST equal
/// the consuming stage's `Input::KIND` or the stage rejects with
/// `KindMismatch` before `Stage::run` is called.
///
/// `payload` holds the bincode-serialized form of the producing
/// stage's typed `Output` artifact. Bincode (length-prefixed binary)
/// beats serde_json for the erased edge because both ends know the
/// concrete `S::Input`/`S::Output` type — `deserialize_any` (the
/// reason bincode was rejected for the *cache record* in opt-2) is
/// not needed at the typed boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErasedArtifact {
    /// Kind tag from the producing artifact's `Artifact::KIND`.
    pub kind: String,
    /// Schema version from the producing artifact's `Artifact::SCHEMA`.
    pub schema: u32,
    /// Bincode-serialized form of the typed artifact struct.
    pub payload: Vec<u8>,
}

/// Wire-format version of the `tuple<N>` merge envelope. Bumped from 1
/// (the old bincode-concat form) to 2 (a length-prefixed
/// `Vec<ErasedArtifact>` carrying each child's kind+schema, validated
/// per-child on decode). The `(A, B)` / `(A, B, C)` `Artifact` impls use
/// this as their `SCHEMA`, and the executor stamps the same value on the
/// envelopes it builds — keep them equal.
pub const TUPLE_ENVELOPE_SCHEMA: u32 = 2;

impl ErasedArtifact {
    /// Wrap a concrete typed artifact for transit across the
    /// `StageDyn` boundary. Delegates to [`Artifact::encode_erased`]
    /// (default = bincode-of-self; tuples override to a per-child
    /// envelope).
    pub fn from_typed<A: Artifact>(value: &A) -> Result<Self, ErasedEncodeError> {
        value.encode_erased()
    }

    /// Reverse: typed artifact out, with strong checks. Delegates to
    /// [`Artifact::decode_erased`], which reports `Kind`/`Schema`
    /// mismatches (and, for tuples, validates each child recursively).
    pub fn into_typed<A: Artifact>(self) -> Result<A, ErasedDecodeError> {
        A::decode_erased(self)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ErasedEncodeError {
    #[error("bincode serialize: {0}")]
    Serialize(#[source] Box<bincode::ErrorKind>),
}

#[derive(Debug, thiserror::Error)]
pub enum ErasedDecodeError {
    #[error("kind mismatch: expected '{expected}', got '{got}'")]
    Kind { expected: &'static str, got: String },
    #[error("schema mismatch: expected v{expected}, got v{got}")]
    Schema { expected: u32, got: u32 },
    #[error("tuple arity mismatch: expected {expected} children, got {got}")]
    Arity { expected: usize, got: usize },
    #[error("bincode deserialize: {0}")]
    Deserialize(#[source] Box<bincode::ErrorKind>),
}

/// Per-stage execution context. Holds everything `Stage::run`
/// needs that isn't its own typed input + args.
///
/// One `StageContext` per stage invocation; the executor builds
/// it. Stages cannot create their own — that would let them
/// invent a stage_dir or status sender, which would break audit
/// integrity.
pub struct StageContext {
    /// Root job directory. Stages may read from sibling stages'
    /// dirs but should write only to their own `stage_dir`.
    pub job_dir: PathBuf,
    /// `<job_dir>/stages/<idx>-<name>/`. Stage writes its primary
    /// artifact + sidecar metadata here.
    pub stage_dir: PathBuf,
    /// 0-indexed position of this stage in the executor's topo
    /// walk. Stages emit `StageStep` events under this node_idx
    /// so dashboards can route progress correctly when the same
    /// stage appears at multiple positions in a plan.
    pub node_idx: u32,
    /// Broadcast sender for status events. Stages emit `StageStep`
    /// for fine-grained progress; framework code emits
    /// Begin/End/Failed/Skipped/Blocked.
    pub status_tx: broadcast::Sender<StageEvent>,
    /// Cooperative cancellation. Stages with long-running
    /// subprocesses (Python trainer) listen via `is_cancelled`
    /// and SIGTERM their child.
    pub cancel: CancellationToken,
    /// Cache handle for read/write.
    pub cache: Arc<CacheHandle>,
    /// Name of the RECIPE this plan was compiled from (ADR 0046
    /// slice-2). Threaded from the plan so a train stage can build the
    /// SAME `broker::FootprintKey` the cli admission gate resolves under
    /// — keying the calibration store by recipe (not stage) name keeps
    /// the engine cli from needing to know cookbook stage names. Empty
    /// for `for_test` contexts (a calibration miss → conservative hint,
    /// which is benign).
    pub recipe_name: String,
    /// Where to place this stage's work (#3). `Local` (default) = this box;
    /// a launcher-aware backend submits to Slurm/Ray when set. Threaded from
    /// the CLI `--launcher` flag via `ExecCtx`.
    pub launch_target: crate::config::launcher::LaunchTarget,
}

impl StageContext {
    /// Test-friendly constructor. `node_idx` defaults to 0 since
    /// most unit tests run a single stage at a time.
    pub fn for_test(job_dir: PathBuf, stage_dir: PathBuf) -> Self {
        Self {
            job_dir,
            stage_dir,
            node_idx: 0,
            status_tx: crate::framework::status::make_broadcast(),
            cancel: CancellationToken::new(),
            cache: Arc::new(CacheHandle::job_local(PathBuf::from("/tmp/_cache_test"))),
            recipe_name: String::new(),
            launch_target: crate::config::launcher::LaunchTarget::Local,
        }
    }
}

/// Typed user-facing trait. Implementors are concrete stages.
///
/// Constants:
///   - `NAME` — stable identifier. Used in cache keys, status
///     events, the CLI (`lamu-train stage <name>`).
///   - `SCHEMA` — bumpable version. Bump when the stage's I/O
///     contract changes; cache entries from a different schema
///     are invalidated.
///   - `RESOURCES` — what the stage holds while running. The
///     executor acquires per-Resource semaphores in this list
///     before calling `run`.
///
/// Associated types:
///   - `Input` / `Output` — the typed artifacts. `()` is allowed
///     for graph-input stages (no upstream artifact).
///   - `Args` — stage-specific configuration. Recipes assemble
///     these. Must be serde + JsonSchema for the catalog.
#[async_trait]
pub trait Stage: Send + Sync + 'static {
    const NAME: &'static str;
    const SCHEMA: u32;
    const RESOURCES: &'static [Resource];

    /// Conservative peak RAM this stage holds while running, in GiB. `0`
    /// (default) = no memory reservation. The parallel executor gates the SUM
    /// of in-flight stages' `MEMORY_GIB` against a box-fit budget (`MemTotal −
    /// floor`), so concurrent stages can't stack past the box — never-OOM-the-
    /// BOX under the parallel executor (the per-`Resource` type tags above only
    /// serialize by KIND, not by capacity). Heavy stages (training) set a
    /// conservative upper bound; light stages leave it `0`.
    const MEMORY_GIB: u32 = 0;

    /// Whether re-running this stage with the same input + args
    /// produces a byte-equal output artifact.
    ///
    /// Default `true` matches pure-function stages (filter, split,
    /// convert). Training stages must override to `false` — model
    /// weights have stochastic seeds (CUDA nondeterminism, data
    /// loader shuffle, dropout) so two runs of the same stage
    /// produce different ckpt bytes even with identical args.
    ///
    /// Executor uses this when computing downstream cache keys:
    /// for `DETERMINISTIC = true`, downstream sees this stage's
    /// output content_hash (real fingerprint). For
    /// `DETERMINISTIC = false`, downstream sees a synthesized
    /// fingerprint = hash(stage_name ‖ schema ‖ args_canon ‖
    /// input_hash) — stable across re-runs so a downstream stage
    /// doesn't re-execute just because its upstream was retrained.
    const DETERMINISTIC: bool = true;

    /// Retry policy for this stage (D1). Default: no retry. Override for
    /// stages whose failures are often transient (network downloads,
    /// OOM-prone trainers). A plan can override per-node via
    /// `Plan::with_retry`.
    const RETRY: crate::framework::retry::RetryPolicy = crate::framework::retry::RetryPolicy::NONE;

    /// Soft/hard timeout for one attempt of this stage (D2). Default:
    /// none. Override (or `Plan::with_timeout`) to bound a stage that
    /// can hang.
    const TIMEOUT: crate::framework::retry::StageTimeout =
        crate::framework::retry::StageTimeout::NONE;

    type Input: Artifact;
    type Output: Artifact;
    type Args: serde::Serialize
        + serde::de::DeserializeOwned
        + schemars::JsonSchema
        + Send
        + Sync
        + 'static;

    /// Run the stage. Pure function over `(input, args)` plus
    /// whatever side effects the stage's nature requires (reading
    /// `ctx.job_dir`, writing to `ctx.stage_dir`, etc.).
    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Self::Args,
    ) -> Result<Self::Output, StageError>;
}

/// Object-safe shadow. Implemented automatically for every
/// `Stage` via the blanket impl below. Users never write this.
#[async_trait]
pub trait StageDyn: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn schema(&self) -> u32;
    fn deterministic(&self) -> bool;
    fn resources(&self) -> &'static [Resource];
    fn memory_gib(&self) -> u32;
    fn input_kind(&self) -> &'static str;
    fn output_kind(&self) -> &'static str;
    fn args_schema(&self) -> serde_json::Value;
    fn retry(&self) -> crate::framework::retry::RetryPolicy;
    fn timeout(&self) -> crate::framework::retry::StageTimeout;

    /// Stable content address of an erased output produced by THIS
    /// stage. Deserializes the erased payload back to the typed
    /// `Output` and calls the artifact's designed `content_hash()`
    /// (FW-1). That hash addresses the on-disk *content* — never the
    /// machine-specific absolute `path` / metadata embedded in the
    /// bincode handle — so byte-identical content at two different
    /// job dirs or machines yields the same downstream cache key
    /// (`--shared-cache` hits across relocation / machines).
    ///
    /// Returns `None` only if the erased payload doesn't decode as
    /// this stage's `Output` (a contract violation the executor
    /// falls back from gracefully, see executor seam).
    fn output_content_hash(&self, art: &ErasedArtifact) -> Option<ContentHash>;

    /// Re-root any absolute paths the typed `Output` embedded under
    /// `from` so they instead live under `to`, returning the rebased
    /// erased artifact (FW-2). Used by the executor after it
    /// atomically promotes a stage's `.tmp-<key>` working directory
    /// to the final `stage_dir`: the stage baked tmp-rooted paths
    /// into its handle, and this re-points them at the final
    /// location so downstream stages read the promoted files.
    ///
    /// Generic + central: works for every artifact without
    /// per-artifact code by rebasing path-shaped strings on the
    /// JSON projection of the typed output. On any decode/encode
    /// failure it returns the input unchanged (the promote still
    /// happened; only the embedded path hint would be stale).
    fn rebase_output_paths(
        &self,
        art: ErasedArtifact,
        from: &std::path::Path,
        to: &std::path::Path,
    ) -> ErasedArtifact;

    async fn run_erased(
        &self,
        ctx: &StageContext,
        input: ErasedArtifact,
        args: serde_json::Value,
    ) -> Result<ErasedArtifact, StageError>;
}

#[async_trait]
impl<S: Stage> StageDyn for S {
    fn name(&self) -> &'static str {
        S::NAME
    }
    fn schema(&self) -> u32 {
        S::SCHEMA
    }
    fn deterministic(&self) -> bool {
        S::DETERMINISTIC
    }
    fn resources(&self) -> &'static [Resource] {
        S::RESOURCES
    }
    fn memory_gib(&self) -> u32 {
        S::MEMORY_GIB
    }
    fn retry(&self) -> crate::framework::retry::RetryPolicy {
        S::RETRY
    }
    fn timeout(&self) -> crate::framework::retry::StageTimeout {
        S::TIMEOUT
    }
    fn input_kind(&self) -> &'static str {
        <S::Input as Artifact>::KIND
    }
    fn output_kind(&self) -> &'static str {
        <S::Output as Artifact>::KIND
    }
    fn args_schema(&self) -> serde_json::Value {
        // schemars 0.8: schema_for! is a proc macro requiring a
        // type literal, so we go through `gen.subschema_for`
        // instead. The fallback here is "best-effort" — if a
        // future schemars upgrade breaks this we'll see test
        // failures, not silent wrong schemas.
        let mut schema_gen = schemars::r#gen::SchemaGenerator::default();
        let schema = schema_gen.subschema_for::<S::Args>();
        serde_json::to_value(schema).expect("schemars-derived JsonSchema must serialize cleanly")
    }

    fn output_content_hash(&self, art: &ErasedArtifact) -> Option<ContentHash> {
        // Decode the erased payload back to the concrete output type
        // and ask IT for its content address. This is the FW-1 fix:
        // `Artifact::content_hash()` hashes the on-disk content (file
        // bytes / merkle of a dir), NOT the bincode handle that
        // carries an absolute `path` + machine-specific metadata.
        //
        // `into_typed` enforces kind + schema match; a tuple/merge
        // artifact (kind `"tuple<N>"`) deliberately won't match a
        // single-output stage's `Output::KIND` and yields `None` —
        // the executor synthesizes the tuple hash itself.
        art.clone()
            .into_typed::<S::Output>()
            .ok()
            .map(|typed| typed.content_hash())
    }

    fn rebase_output_paths(
        &self,
        art: ErasedArtifact,
        from: &std::path::Path,
        to: &std::path::Path,
    ) -> ErasedArtifact {
        // Decode to the typed output, project to JSON, rewrite any
        // string whose value begins with the `from` directory prefix
        // so it instead lives under `to`, then re-encode. Generic:
        // works for `path`, `train_path`, `eval_path`, … without any
        // per-artifact code. On any failure we return the artifact
        // unchanged — the directory promote already happened; only
        // the embedded path hint would point at the (now-renamed)
        // tmp dir, which is a soft degradation, never data loss.
        let typed: S::Output = match art.clone().into_typed::<S::Output>() {
            Ok(t) => t,
            Err(_) => return art,
        };
        let mut value = match serde_json::to_value(&typed) {
            Ok(v) => v,
            Err(_) => return art,
        };
        let from_str = from.to_string_lossy();
        let to_str = to.to_string_lossy();
        rebase_path_strings(&mut value, from_str.as_ref(), to_str.as_ref());
        let rebased: S::Output = match serde_json::from_value(value) {
            Ok(t) => t,
            Err(_) => return art,
        };
        ErasedArtifact::from_typed(&rebased).unwrap_or(art)
    }

    async fn run_erased(
        &self,
        ctx: &StageContext,
        input: ErasedArtifact,
        args: serde_json::Value,
    ) -> Result<ErasedArtifact, StageError> {
        // 1. Decode input → typed S::Input. KindMismatch /
        //    InputDeserialize translate the ErasedDecodeError
        //    into the StageError variants the executor expects.
        let typed_input: S::Input = input.into_typed::<S::Input>().map_err(|e| match e {
            ErasedDecodeError::Kind { expected, got } => StageError::KindMismatch {
                stage: S::NAME,
                expected,
                got,
            },
            ErasedDecodeError::Schema { expected, got } => StageError::BadInput(format!(
                "input schema for stage '{}' expected v{expected}, got v{got}",
                S::NAME
            )),
            ErasedDecodeError::Arity { expected, got } => StageError::BadInput(format!(
                "merge input for stage '{}' expected {expected} tuple children, got {got}",
                S::NAME
            )),
            ErasedDecodeError::Deserialize(source) => StageError::InputDeserialize {
                stage: S::NAME,
                source,
            },
        })?;

        // 2. Decode args. Args validation is the stage's own
        //    concern beyond serde — we just deserialize. Distinct
        //    error variant from input deserialization so log
        //    readers can disambiguate "bad recipe args" from "bad
        //    upstream artifact".
        let typed_args: S::Args =
            serde_json::from_value(args).map_err(|source| StageError::ArgsDeserialize {
                stage: S::NAME,
                source,
            })?;

        // 3. Call the typed run. This is where the stage actually
        //    does work.
        let output: S::Output = self.run(ctx, typed_input, &typed_args).await?;

        // 4. Re-encode output for the next erased edge.
        ErasedArtifact::from_typed(&output).map_err(|e| match e {
            ErasedEncodeError::Serialize(source) => StageError::OutputSerialize {
                stage: S::NAME,
                source,
            },
        })
    }
}

/// Recursively rewrite any JSON string that begins with the `from`
/// directory prefix so it instead begins with `to`. Matches either
/// the exact prefix (the dir itself) or `from` followed by the
/// platform path separator (a child path) — so `/a/b` does NOT
/// accidentally rewrite `/a/bc`. Used by `rebase_output_paths` to
/// re-point an artifact's embedded absolute paths after the executor
/// promotes the stage's tmp working dir to its final `stage_dir`.
fn rebase_path_strings(value: &mut serde_json::Value, from: &str, to: &str) {
    use serde_json::Value;
    match value {
        Value::String(s) => {
            if s == from {
                *s = to.to_string();
            } else {
                let sep = std::path::MAIN_SEPARATOR;
                let with_sep = format!("{from}{sep}");
                if let Some(rest) = s.strip_prefix(&with_sep) {
                    *s = format!("{to}{sep}{rest}");
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                rebase_path_strings(item, from, to);
            }
        }
        Value::Object(map) => {
            for (_k, v) in map.iter_mut() {
                rebase_path_strings(v, from, to);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::artifact::ContentHash;
    use serde::{Deserialize, Serialize};
    use std::path::Path;

    // ── A toy artifact + a toy stage to exercise erased dispatch ──

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Words {
        text: String,
    }

    impl Artifact for Words {
        const KIND: &'static str = "test.words";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(self.text.as_bytes())
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Count {
        n: usize,
    }

    impl Artifact for Count {
        const KIND: &'static str = "test.count";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(&self.n.to_le_bytes())
        }
        fn primary_path(&self) -> &Path {
            Path::new(".")
        }
    }

    /// Counts words. The minimal complete stage.
    struct WordCount;

    #[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
    struct WordCountArgs {
        delimiter: String,
    }

    #[async_trait]
    impl Stage for WordCount {
        const NAME: &'static str = "word_count";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[Resource::Cpu];
        type Input = Words;
        type Output = Count;
        type Args = WordCountArgs;

        async fn run(
            &self,
            _ctx: &StageContext,
            input: Self::Input,
            args: &Self::Args,
        ) -> Result<Self::Output, StageError> {
            let n = input.text.split(args.delimiter.as_str()).count();
            Ok(Count { n })
        }
    }

    fn ctx() -> StageContext {
        let td = tempfile::tempdir().unwrap();
        let job = td.path().to_path_buf();
        let stage = job.join("stages/0-word_count");
        // Tempdir lives only for the test body; leak the guard so
        // job_dir paths stay valid for the duration. tests are
        // throwaway-process so /tmp is cleaned at next reboot.
        std::mem::forget(td);
        StageContext::for_test(job, stage)
    }

    // ── ErasedArtifact round trip ─────────────────────────────────

    #[test]
    fn erased_round_trip_preserves_kind_and_schema() {
        let w = Words {
            text: "a b c".into(),
        };
        let e = ErasedArtifact::from_typed(&w).unwrap();
        assert_eq!(e.kind, "test.words");
        assert_eq!(e.schema, 1);
        let back: Words = e.into_typed().unwrap();
        assert_eq!(back.text, "a b c");
    }

    #[test]
    fn erased_into_typed_kind_mismatch_errors() {
        let w = Words { text: "x".into() };
        let e = ErasedArtifact::from_typed(&w).unwrap();
        // Try to decode as Count.
        let r: Result<Count, _> = e.into_typed();
        match r {
            Err(ErasedDecodeError::Kind { expected, got }) => {
                assert_eq!(expected, "test.count");
                assert_eq!(got, "test.words");
            }
            other => panic!("wrong variant: {:?}", other.err()),
        }
    }

    #[test]
    fn erased_into_typed_schema_mismatch_errors() {
        // Hand-craft an ErasedArtifact with mismatched schema. The
        // payload bytes are still valid bincode of Words; the schema
        // check fires before deserialize.
        let payload = bincode::serialize(&Words { text: "x".into() }).unwrap();
        let e = ErasedArtifact {
            kind: "test.words".into(),
            schema: 99,
            payload,
        };
        let r: Result<Words, _> = e.into_typed();
        assert!(matches!(
            r,
            Err(ErasedDecodeError::Schema {
                expected: 1,
                got: 99
            })
        ));
    }

    // ── Stage typed contract ─────────────────────────────────────

    #[tokio::test]
    async fn typed_stage_run_returns_output() {
        let ctx = ctx();
        let s = WordCount;
        let out = s
            .run(
                &ctx,
                Words {
                    text: "a b c".into(),
                },
                &WordCountArgs {
                    delimiter: " ".into(),
                },
            )
            .await
            .unwrap();
        assert_eq!(out.n, 3);
    }

    // ── Erased dispatch through StageDyn ─────────────────────────

    #[tokio::test]
    async fn erased_dispatch_round_trip_via_stagedyn() {
        let ctx = ctx();
        let s: Box<dyn StageDyn> = Box::new(WordCount);

        let input = ErasedArtifact::from_typed(&Words {
            text: "alpha,beta,gamma".into(),
        })
        .unwrap();
        let args = serde_json::json!({"delimiter": ","});

        let output = s.run_erased(&ctx, input, args).await.unwrap();
        assert_eq!(output.kind, "test.count");
        assert_eq!(output.schema, 1);
        let count: Count = output.into_typed().unwrap();
        assert_eq!(count.n, 3);
    }

    #[tokio::test]
    async fn erased_dispatch_kind_mismatch_returns_kind_error() {
        let ctx = ctx();
        let s: Box<dyn StageDyn> = Box::new(WordCount);
        let wrong = ErasedArtifact::from_typed(&Count { n: 5 }).unwrap();
        let r = s
            .run_erased(&ctx, wrong, serde_json::json!({"delimiter": " "}))
            .await;
        match r {
            Err(StageError::KindMismatch {
                stage,
                expected,
                got,
            }) => {
                assert_eq!(stage, "word_count");
                assert_eq!(expected, "test.words");
                assert_eq!(got, "test.count");
            }
            other => panic!("wrong variant: {:?}", other.err()),
        }
    }

    #[tokio::test]
    async fn erased_dispatch_bad_args_returns_args_deserialize() {
        let ctx = ctx();
        let s: Box<dyn StageDyn> = Box::new(WordCount);
        let input = ErasedArtifact::from_typed(&Words { text: "x".into() }).unwrap();
        // Args expects a `delimiter` String — pass an int. Must
        // surface as ArgsDeserialize, NOT InputDeserialize, so log
        // readers can tell "bad recipe args" from "bad upstream
        // artifact".
        let bad_args = serde_json::json!({"delimiter": 42});
        let r = s.run_erased(&ctx, input, bad_args).await;
        match r {
            Err(StageError::ArgsDeserialize { stage, .. }) => {
                assert_eq!(stage, "word_count");
            }
            other => panic!("wrong variant: {:?}", other.err()),
        }
    }

    #[tokio::test]
    async fn erased_dispatch_bad_input_returns_input_deserialize() {
        let ctx = ctx();
        let s: Box<dyn StageDyn> = Box::new(WordCount);
        // Hand-craft an input whose kind matches but whose payload
        // doesn't deserialize as Words (garbage bytes).
        let bad_input = ErasedArtifact {
            kind: "test.words".into(),
            schema: 1,
            payload: vec![0xFF, 0xFF, 0xFF, 0xFF],
        };
        let good_args = serde_json::json!({"delimiter": " "});
        let r = s.run_erased(&ctx, bad_input, good_args).await;
        match r {
            Err(StageError::InputDeserialize { stage, .. }) => {
                assert_eq!(stage, "word_count");
            }
            other => panic!("wrong variant: {:?}", other.err()),
        }
    }

    // ── Constants accessible through StageDyn ────────────────────

    #[test]
    fn stagedyn_exposes_constants() {
        let s: Box<dyn StageDyn> = Box::new(WordCount);
        assert_eq!(s.name(), "word_count");
        assert_eq!(s.schema(), 1);
        assert_eq!(s.resources(), &[Resource::Cpu]);
        assert_eq!(s.input_kind(), "test.words");
        assert_eq!(s.output_kind(), "test.count");
        // args_schema returns SOMETHING valid (not Null) for a
        // type with JsonSchema.
        let schema = s.args_schema();
        assert!(
            schema != serde_json::Value::Null,
            "args_schema unexpectedly null"
        );
    }

    // ── StageContext constructible for tests ─────────────────────

    #[tokio::test]
    async fn context_carries_cancel_token_observable_to_stage() {
        let ctx = ctx();
        ctx.cancel.cancel();
        assert!(ctx.cancel.is_cancelled());
    }

    #[tokio::test]
    async fn context_status_tx_can_be_subscribed() {
        let ctx = ctx();
        let mut rx = ctx.status_tx.subscribe();
        ctx.status_tx
            .send(StageEvent::StageBegin {
                node_idx: 0,
                stage_name: "x".into(),
                input_hash: ContentHash::of_bytes(b""),
            })
            .unwrap();
        let _evt = rx.recv().await.unwrap();
    }

    // ── FW-1: StageDyn::output_content_hash ──────────────────────

    /// Artifact carrying a content byte + a path. content_hash
    /// ignores the path (content-addressed); the handle bincode does
    /// NOT — so the two hashes diverge, exactly as a real artifact.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct PathyOut {
        content: u8,
        path: std::path::PathBuf,
    }
    impl Artifact for PathyOut {
        const KIND: &'static str = "test.pathy";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(&[self.content])
        }
        fn primary_path(&self) -> &Path {
            &self.path
        }
    }

    struct MakePathy;
    #[async_trait]
    impl Stage for MakePathy {
        const NAME: &'static str = "make_pathy";
        const SCHEMA: u32 = 1;
        const RESOURCES: &'static [Resource] = &[];
        type Input = ();
        type Output = PathyOut;
        type Args = ();
        async fn run(
            &self,
            _ctx: &StageContext,
            _input: (),
            _args: &(),
        ) -> Result<PathyOut, StageError> {
            Ok(PathyOut {
                content: 9,
                path: std::path::PathBuf::from("/x"),
            })
        }
    }

    #[test]
    fn output_content_hash_uses_artifact_content_hash_not_handle() {
        let s: Box<dyn StageDyn> = Box::new(MakePathy);
        let art = PathyOut {
            content: 9,
            path: std::path::PathBuf::from("/some/abs/path"),
        };
        let erased = ErasedArtifact::from_typed(&art).unwrap();
        // The StageDyn hook must return the artifact's OWN content
        // address (path-independent), not a hash of the bincode
        // handle (which embeds the path).
        let via_dyn = s.output_content_hash(&erased).expect("decodes as PathyOut");
        assert_eq!(via_dyn, art.content_hash(), "must be the content_hash()");

        // A different path with the SAME content → same hook output.
        let art2 = PathyOut {
            content: 9,
            path: std::path::PathBuf::from("/totally/other"),
        };
        let erased2 = ErasedArtifact::from_typed(&art2).unwrap();
        assert_eq!(
            s.output_content_hash(&erased2).unwrap(),
            via_dyn,
            "content_hash must be path-independent"
        );
    }

    #[test]
    fn output_content_hash_none_on_kind_mismatch() {
        // A tuple/merge or foreign-kind payload doesn't decode as the
        // stage's Output → None (executor then synthesizes the hash).
        let s: Box<dyn StageDyn> = Box::new(MakePathy);
        let foreign = ErasedArtifact {
            kind: "tuple<2>".into(),
            schema: 1,
            payload: vec![0, 1, 2],
        };
        assert!(s.output_content_hash(&foreign).is_none());
    }

    // ── FW-2: rebase_output_paths ────────────────────────────────

    #[test]
    fn rebase_output_paths_repoints_embedded_path() {
        let s: Box<dyn StageDyn> = Box::new(MakePathy);
        let from = std::path::Path::new("/job/stages/.tmp-0-make_pathy-deadbeef");
        let to = std::path::Path::new("/job/stages/0-make_pathy");
        let art = PathyOut {
            content: 3,
            path: from.join("checkpoint/model.pt"),
        };
        let erased = ErasedArtifact::from_typed(&art).unwrap();
        let rebased = s.rebase_output_paths(erased, from, to);
        let back: PathyOut = rebased.into_typed().unwrap();
        assert_eq!(
            back.path,
            to.join("checkpoint/model.pt"),
            "FW-2: a tmp-rooted embedded path must be re-pointed at the promoted final dir"
        );
        // Content byte untouched by the rebase.
        assert_eq!(back.content, 3);
    }

    #[test]
    fn rebase_output_paths_leaves_unrelated_paths_alone() {
        let s: Box<dyn StageDyn> = Box::new(MakePathy);
        let from = std::path::Path::new("/job/stages/.tmp-0-x");
        let to = std::path::Path::new("/job/stages/0-x");
        // A path NOT under `from` (a sibling whose prefix only shares
        // the parent) must NOT be rewritten — guards against the
        // `/a/b` ⊄ `/a/bc` footgun.
        let art = PathyOut {
            content: 1,
            path: std::path::PathBuf::from("/job/stages/.tmp-0-xyz/f"),
        };
        let erased = ErasedArtifact::from_typed(&art).unwrap();
        let rebased = s.rebase_output_paths(erased, from, to);
        let back: PathyOut = rebased.into_typed().unwrap();
        assert_eq!(
            back.path,
            std::path::PathBuf::from("/job/stages/.tmp-0-xyz/f")
        );
    }

    #[test]
    fn rebase_path_strings_exact_and_child_prefix() {
        use serde_json::json;
        let mut v = json!({
            "exact": "/a/b",
            "child": "/a/b/c/d.txt",
            "sibling": "/a/bc",       // must NOT match
            "unrelated": "/x/y",
            "nested": {"p": "/a/b/inner"},
            "list": ["/a/b/k", "keep"],
        });
        let sep = std::path::MAIN_SEPARATOR;
        let from = format!("{sep}a{sep}b");
        let to = format!("{sep}new{sep}root");
        rebase_path_strings(&mut v, &from, &to);
        assert_eq!(v["exact"], json!(format!("{to}")));
        assert_eq!(v["child"], json!(format!("{to}{sep}c{sep}d.txt")));
        assert_eq!(
            v["sibling"],
            json!(format!("{sep}a{sep}bc")),
            "sibling prefix must not match"
        );
        assert_eq!(v["unrelated"], json!(format!("{sep}x{sep}y")));
        assert_eq!(v["nested"]["p"], json!(format!("{to}{sep}inner")));
        assert_eq!(v["list"][0], json!(format!("{to}{sep}k")));
        assert_eq!(v["list"][1], json!("keep"));
    }
}
