// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `cargo run --example first_cookbook`
//!
//! A complete tour of the BLUT abstraction hierarchy in one runnable file:
//!
//! ```text
//!   BLUT  ▸  Cookbook  ▸  Course  ▸  Recipe  ▸  Ingredient
//! ```
//!
//! - **Ingredient** — an atomic primitive: a typed [`Stage`] (`Input → Output`).
//!   The smallest reusable unit of work. (A training cookbook also has finer
//!   primitives — optimizers, schedulers, losses — that a stage composes.)
//! - **Recipe** — a middle-level orchestration function: it composes ingredients
//!   into a typed [`Plan`]. A named, args-driven workflow.
//! - **Course** — the phase a recipe belongs to (`DataPrep`, `Pretrain`, `Train`,
//!   `Eval`, `Gate`, `Export`, `Pipeline`, `User`). Recipes are grouped by
//!   course; a cookbook's courses are selected and orchestrated in order.
//! - **Cookbook** — a domain pack: a set of recipes plus the backend they target.
//!   Cookbooks are *loaded into* BLUT.
//! - **BLUT** — the engine: it loads cookbooks into a [`Registry`], then compiles
//!   and runs a recipe's plan against the content-addressed cache, under
//!   per-stage resource + memory admission.
//!
//! Wrong wiring is a `cargo build` error, not a runtime panic: a stage's `Input`
//! must equal the previous stage's `Output`, and a stage can only join a plan
//! whose backend it is [`Compatible`] with.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

use blut::backends::TrainingBackend;
use blut::framework::{
    Artifact, Compatible, ContentHash, Cookbook, ExecCtx, Plan, RecipeError, Registry, Resource,
    SequentialExecutor, Stage, StageContext, StageError,
};
use blut::recipes::{Recipe, RecipeCategory, RecipeDef};

// ── Backend ────────────────────────────────────────────────────────────────
// A cookbook targets one backend. `Plan<Out, B>` is parameterized over it, and
// the engine ships none — you tag your own. The `ID` is a stable cache/audit key.
struct DemoBackend;
impl TrainingBackend for DemoBackend {
    const ID: &'static str = "demo";
    const DESCRIPTION: &'static str = "first_cookbook example backend";
}

// ── Artifact ─────────────────────────────────────────────────────────────────
// A content-hashed handle to a stage's output. Identical inputs + args ⇒ identical
// hash ⇒ a cache hit instead of a re-run.
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
    fn primary_path(&self) -> &Path {
        // A real artifact returns the path to its bytes (a checkpoint dir, a
        // dataset file, …); this toy one lives entirely in its struct.
        Path::new(".")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct NoArgs;

// ── Ingredients (atomic primitives = Stages) ─────────────────────────────────
// `MakeOne`: () → Counter{1}. `Input = ()` makes it a source stage.
struct MakeOne;
#[async_trait]
impl Stage for MakeOne {
    const NAME: &'static str = "make_one";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Counter;
    type Args = NoArgs;
    async fn run(&self, _: &StageContext, _: (), _: &NoArgs) -> Result<Counter, StageError> {
        println!("    · ingredient make_one ran → Counter {{ n: 1 }}");
        Ok(Counter { n: 1 })
    }
}
impl Compatible<DemoBackend> for MakeOne {}

// `Increment`: Counter → Counter{n+1}. Its `Input` MUST match the prior stage's
// `Output` or `.then(...)` won't compile.
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
        _: &StageContext,
        input: Counter,
        _: &NoArgs,
    ) -> Result<Counter, StageError> {
        let out = Counter { n: input.n + 1 };
        println!("    · ingredient increment ran → {} → {}", input.n, out.n);
        Ok(out)
    }
}
impl Compatible<DemoBackend> for Increment {}

// ── Recipe (middle-level orchestration) ──────────────────────────────────────
// Composes ingredients into a typed `Plan`, declares its `Course` (CATEGORY),
// its backend, and its typed `Args`. `register_recipe!` then emits the erased
// catalog entry (`DEF`) the engine lists + runs by name.
#[derive(Default)]
struct CountToThree;
impl Recipe for CountToThree {
    const NAME: &'static str = "count_to_three";
    const DESCRIPTION: &'static str = "MakeOne → Increment → Increment (demo)";
    // The Course this recipe belongs to. A real cookbook uses DataPrep / Train /
    // Eval / Gate / Export / Pipeline; `User` is the catch-all for demos.
    const CATEGORY: RecipeCategory = RecipeCategory::User;
    const OUTPUT_KIND: &'static str = Counter::KIND;
    type Backend = DemoBackend;
    type Args = NoArgs;
    fn compile(&self, _args: NoArgs) -> Result<Plan<(), DemoBackend>, RecipeError> {
        Ok(
            Plan::<(), DemoBackend>::new("count_to_three", serde_json::json!({}))
                .start(MakeOne, NoArgs)
                .then(Increment, NoArgs)
                .then(Increment, NoArgs)
                .finish(),
        )
    }
}
blut::register_recipe!(CountToThree); // → `pub static DEF: RecipeDef`

// ── Cookbook (a domain pack, loaded into BLUT) ───────────────────────────────
// Groups recipes + their backend. A real cookbook returns many recipes across
// several courses; this one has a single recipe.
struct DemoCookbook;
impl Cookbook for DemoCookbook {
    fn name(&self) -> &'static str {
        "demo"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        static RECIPES: &[&RecipeDef] = &[&DEF];
        RECIPES
    }
}

#[tokio::main]
async fn main() {
    // BLUT: load the cookbook into a Registry. This is what `blut::cli::run`
    // receives; here we drive the layers directly.
    let mut registry = Registry::new();
    registry.register(Box::new(DemoCookbook));

    println!("\nBLUT loaded cookbook 'demo'. Recipes by course:");
    for def in registry.all() {
        println!(
            "  • {:?}  {}  — {}",
            def.category, def.name, def.description
        );
    }

    // Compile the recipe (ingredients → typed Plan) and run it twice against the
    // SAME cache dir to show the content-addressed cache skip on the second run.
    let job_dir = std::env::temp_dir().join("blut_first_cookbook_example");
    std::fs::create_dir_all(&job_dir).expect("create job dir");
    let build = || {
        CountToThree
            .compile(NoArgs)
            .expect("compile recipe")
            .into_compiled()
    };

    println!("\nRun 1 (cold cache):");
    let r1 = SequentialExecutor::execute(build(), ExecCtx::new(job_dir.clone()))
        .await
        .expect("run 1");
    println!(
        "  → {} stages, {} hits, {} misses",
        r1.n_stages, r1.n_cache_hits, r1.n_cache_misses
    );

    println!("\nRun 2 (warm cache — same job dir):");
    let r2 = SequentialExecutor::execute(build(), ExecCtx::new(job_dir.clone()))
        .await
        .expect("run 2");
    println!(
        "  → {} stages, {} hits, {} misses",
        r2.n_stages, r2.n_cache_hits, r2.n_cache_misses
    );

    assert_eq!(r1.n_cache_misses, 3, "run 1 computes every ingredient");
    assert_eq!(r2.n_cache_hits, 3, "run 2 is served entirely from cache");

    println!(
        "\n✓ BLUT ▸ Cookbook ▸ Course ▸ Recipe ▸ Ingredient — run 2 recomputed {} of 3 ingredients.",
        r2.n_cache_misses
    );
    println!("  (delete {} to reset the cache.)", job_dir.display());
}
