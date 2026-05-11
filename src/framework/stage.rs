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

use crate::framework::artifact::Artifact;
use crate::framework::cache::CacheHandle;
use crate::framework::error::StageError;
use crate::framework::resource::Resource;
use crate::framework::status::StageEvent;

/// Erased artifact: a kind-tagged JSON blob that passes between
/// stages at the `StageDyn` boundary. The `kind` field MUST equal
/// the consuming stage's `Input::KIND` or the stage rejects with
/// `KindMismatch` before `Stage::run` is called.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErasedArtifact {
    /// Kind tag from the producing artifact's `Artifact::KIND`.
    pub kind: String,
    /// Schema version from the producing artifact's `Artifact::SCHEMA`.
    pub schema: u32,
    /// Serialized form of the typed artifact struct.
    pub payload: serde_json::Value,
}

impl ErasedArtifact {
    /// Wrap a concrete typed artifact for transit across the
    /// `StageDyn` boundary. Cheap — a serde_json round-trip on the
    /// metadata-sized handle, not on the on-disk bytes.
    pub fn from_typed<A: Artifact>(value: &A) -> Result<Self, serde_json::Error> {
        Ok(Self {
            kind: A::KIND.to_string(),
            schema: A::SCHEMA,
            payload: serde_json::to_value(value)?,
        })
    }

    /// Reverse: typed artifact out, with strong checks. The
    /// caller (a `StageDyn::run_erased` impl) reports `KindMismatch`
    /// if the kind tag doesn't match the expected `Input::KIND`.
    pub fn into_typed<A: Artifact>(self) -> Result<A, ErasedDecodeError> {
        if self.kind != A::KIND {
            return Err(ErasedDecodeError::Kind {
                expected: A::KIND,
                got: self.kind,
            });
        }
        // Schema mismatch is downgraded to a deserialize error
        // (returns `Schema`) — the producer's SCHEMA may legitimately
        // exceed ours if a newer producer is paired with an older
        // consumer. Cache invalidation handles the common case.
        if self.schema != A::SCHEMA {
            return Err(ErasedDecodeError::Schema {
                expected: A::SCHEMA,
                got: self.schema,
            });
        }
        serde_json::from_value(self.payload).map_err(ErasedDecodeError::Deserialize)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ErasedDecodeError {
    #[error("kind mismatch: expected '{expected}', got '{got}'")]
    Kind {
        expected: &'static str,
        got: String,
    },
    #[error("schema mismatch: expected v{expected}, got v{got}")]
    Schema { expected: u32, got: u32 },
    #[error("deserialize: {0}")]
    Deserialize(#[from] serde_json::Error),
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
    /// Broadcast sender for status events. Stages emit `StageStep`
    /// for fine-grained progress; framework code emits
    /// Begin/End/Failed/Skipped/Blocked.
    pub status_tx: broadcast::Sender<StageEvent>,
    /// Cooperative cancellation. Stages with long-running
    /// subprocesses (Python trainer) listen via `is_cancelled`
    /// and SIGTERM their child.
    pub cancel: CancellationToken,
    /// Cache handle for read/write. v2-2 stub returns `None` on
    /// every lookup; v2-3 wires real lookup.
    pub cache: Arc<CacheHandle>,
}

impl StageContext {
    /// Test-friendly constructor. Production callers use the
    /// executor's builder (lands commit 3); this is the bare-bones
    /// version for unit tests in this commit.
    pub fn for_test(job_dir: PathBuf, stage_dir: PathBuf) -> Self {
        Self {
            job_dir,
            stage_dir,
            status_tx: crate::framework::status::make_broadcast(),
            cancel: CancellationToken::new(),
            cache: Arc::new(CacheHandle::job_local(PathBuf::from("/tmp/_cache_test"))),
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
    fn resources(&self) -> &'static [Resource];
    fn input_kind(&self) -> &'static str;
    fn output_kind(&self) -> &'static str;
    fn args_schema(&self) -> serde_json::Value;

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
    fn resources(&self) -> &'static [Resource] {
        S::RESOURCES
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
        serde_json::to_value(schema).unwrap_or(serde_json::Value::Null)
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
        let typed_args: S::Args = serde_json::from_value(args).map_err(|source| {
            StageError::ArgsDeserialize {
                stage: S::NAME,
                source,
            }
        })?;

        // 3. Call the typed run. This is where the stage actually
        //    does work.
        let output: S::Output = self.run(ctx, typed_input, &typed_args).await?;

        // 4. Re-encode output for the next erased edge.
        ErasedArtifact::from_typed(&output).map_err(|source| StageError::OutputSerialize {
            stage: S::NAME,
            source,
        })
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
        let w = Words { text: "a b c".into() };
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
        // Hand-craft an ErasedArtifact with mismatched schema.
        let e = ErasedArtifact {
            kind: "test.words".into(),
            schema: 99,
            payload: serde_json::json!({"text": "x"}),
        };
        let r: Result<Words, _> = e.into_typed();
        assert!(matches!(r, Err(ErasedDecodeError::Schema { expected: 1, got: 99 })));
    }

    // ── Stage typed contract ─────────────────────────────────────

    #[tokio::test]
    async fn typed_stage_run_returns_output() {
        let ctx = ctx();
        let s = WordCount;
        let out = s
            .run(&ctx, Words { text: "a b c".into() }, &WordCountArgs { delimiter: " ".into() })
            .await
            .unwrap();
        assert_eq!(out.n, 3);
    }

    // ── Erased dispatch through StageDyn ─────────────────────────

    #[tokio::test]
    async fn erased_dispatch_round_trip_via_stagedyn() {
        let ctx = ctx();
        let s: Box<dyn StageDyn> = Box::new(WordCount);

        let input = ErasedArtifact::from_typed(&Words { text: "alpha,beta,gamma".into() }).unwrap();
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
            Err(StageError::KindMismatch { stage, expected, got }) => {
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
        // doesn't deserialize as Words.
        let bad_input = ErasedArtifact {
            kind: "test.words".into(),
            schema: 1,
            payload: serde_json::json!({"text": 12345}),
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
        assert!(schema != serde_json::Value::Null, "args_schema unexpectedly null");
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
}
