//! `cargo run --example first_cookbook`
//!
//! The smallest end-to-end blut pipeline: define a backend, a typed
//! artifact, two typed stages, wire them into a `Plan`, run it, then run
//! it AGAIN to watch the content-addressed cache skip every stage.
//!
//! This is the mental model for building your own cookbook on top of the
//! engine. The four moving parts:
//!
//!   1. a **backend identity** — a unit struct implementing
//!      [`TrainingBackend`]. A `Plan<Out, B>` is parameterized over its
//!      backend `B`, and a stage can only join a plan whose backend it is
//!      `Compatible` with — so wrong wiring is a COMPILE error.
//!   2. an **artifact** — a `Serialize`/`Deserialize` struct implementing
//!      [`Artifact`] (a content-hashed handle to a stage's output).
//!   3. **stages** — typed `Input → Output` units implementing [`Stage`].
//!   4. a **plan** — `Plan::start(...).then(...).finish()`, a compile-time
//!      typed DAG run by an executor against a content-addressed cache.
//!
//! The next step up — turning a plan into a named, args-driven *recipe* and
//! grouping recipes into a `Cookbook` (via `blut::register_recipe!` and the
//! [`blut::framework::Cookbook`] trait) — is what the `blut-lamquant` /
//! `blut-lamu` cookbook crates do. See the README's "Build your own
//! cookbook" section.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

use blut::backends::TrainingBackend;
use blut::framework::{
    Artifact, Compatible, ContentHash, ExecCtx, Plan, Resource, SequentialExecutor, Stage,
    StageContext, StageError,
};

// 1. ── A backend identity ─────────────────────────────────────────────────
// The engine ships no concrete backends; you tag your own. The `ID` is a
// stable string used in cache keys, status events, and provenance.
struct DemoBackend;
impl TrainingBackend for DemoBackend {
    const ID: &'static str = "demo";
    const DESCRIPTION: &'static str = "first_cookbook example backend";
}

// 2. ── A typed artifact ───────────────────────────────────────────────────
// `content_hash` makes outputs content-addressable: identical inputs +
// identical args ⇒ identical hash ⇒ a cache hit instead of a re-run.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Counter {
    n: u32,
}
impl Artifact for Counter {
    const KIND: &'static str = "example.counter";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&self.n.to_le_bytes())
    }
    // This toy artifact lives entirely in its struct (no backing file), so
    // there is no meaningful on-disk path. A real artifact returns the path
    // to its bytes (a checkpoint dir, a dataset file, …).
    fn primary_path(&self) -> &Path {
        Path::new(".")
    }
}

// Stages take a typed `Args`. This one needs none; `schemars::JsonSchema`
// is required so a recipe could later expose it as a CLI/TUI arg schema.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct NoArgs;

// 3. ── Two typed stages ───────────────────────────────────────────────────
// `MakeOne`: () → Counter{1}. The `Input = ()` makes it a source stage.
struct MakeOne;
#[async_trait]
impl Stage for MakeOne {
    const NAME: &'static str = "make_one";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Counter;
    type Args = NoArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &NoArgs,
    ) -> Result<Counter, StageError> {
        println!("    · MakeOne ran (produced Counter {{ n: 1 }})");
        Ok(Counter { n: 1 })
    }
}
// Declare the stage compatible with our backend — this is the compile-time
// gate: a stage tagged for a different backend can't enter a DemoBackend plan.
impl Compatible<DemoBackend> for MakeOne {}

// `Increment`: Counter → Counter{n+1}. Its `Input` type MUST match the
// previous stage's `Output`, or `.then(...)` won't compile.
struct Increment;
#[async_trait]
impl Stage for Increment {
    const NAME: &'static str = "increment";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = Counter;
    type Output = Counter;
    type Args = NoArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        input: Counter,
        _args: &NoArgs,
    ) -> Result<Counter, StageError> {
        let out = Counter { n: input.n + 1 };
        println!("    · Increment ran ({} → {})", input.n, out.n);
        Ok(out)
    }
}
impl Compatible<DemoBackend> for Increment {}

#[tokio::main]
async fn main() {
    // A directory for this run's job state + the content-addressed cache
    // (`<job_dir>/_cache`). Reusing the SAME dir across runs is what lets the
    // second run hit the cache. We use a fixed temp path so re-invoking the
    // example demonstrates the cross-process cache too.
    let job_dir = std::env::temp_dir().join("blut_first_cookbook_example");
    std::fs::create_dir_all(&job_dir).expect("create job dir");

    // 4. ── The typed DAG ──────────────────────────────────────────────────
    // () → MakeOne → Counter → Increment → Counter → Increment → Counter.
    // The types line up at compile time; a mismatch is a `cargo build` error.
    let build_plan = || {
        Plan::<(), DemoBackend>::new("count_to_three", serde_json::json!({}))
            .start(MakeOne, NoArgs)
            .then(Increment, NoArgs)
            .then(Increment, NoArgs)
            .finish()
            .into_compiled()
    };

    println!("\nRun 1 (cold cache):");
    let r1 = SequentialExecutor::execute(build_plan(), ExecCtx::new(job_dir.clone()))
        .await
        .expect("plan run 1");
    println!(
        "  → {} stages, {} cache hits, {} misses, {:?}",
        r1.n_stages, r1.n_cache_hits, r1.n_cache_misses, r1.elapsed
    );

    println!("\nRun 2 (warm cache — same job dir):");
    let r2 = SequentialExecutor::execute(build_plan(), ExecCtx::new(job_dir.clone()))
        .await
        .expect("plan run 2");
    println!(
        "  → {} stages, {} cache hits, {} misses, {:?}",
        r2.n_stages, r2.n_cache_hits, r2.n_cache_misses, r2.elapsed
    );

    assert_eq!(r1.n_cache_misses, 3, "run 1 should compute every stage");
    assert_eq!(
        r2.n_cache_hits, 3,
        "run 2 should be served entirely from cache"
    );

    println!(
        "\n✓ Same inputs ⇒ content-addressed cache hit: run 2 recomputed {} of 3 stages.",
        r2.n_cache_misses
    );
    println!("  (delete {} to reset the cache.)", job_dir.display());
}
