// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Microbenchmarks for the BLUT framework hot paths.
//!
//! Runs:
//!
//!   cargo bench -p blut
//!
//! Tracks regressions in:
//!   - `ContentHash::hash_file` over a 10 MiB blob
//!   - `ContentHash::hash_dir` (parallel) vs `hash_dir_serial`
//!     across a 50-file checkpoint-shaped tree
//!   - Cache key computation
//!   - `ErasedArtifact` round-trip (post-opt-4: bincode)
//!   - Cache hit path: write a cache entry + lookup it back
//!
//! simd-json dropped after opt-4 benchmarks showed serde_json faster
//! on sub-KB payloads.
//!
//! OPT-0 (measurement plane): the second group covers the REMAINING engine
//! hot paths the OPT plan flagged — the per-run/per-step orchestration cost
//! that sits in front of HOURS of actual training:
//!   - Plan compile: `topo_order`, `from_components` (HPO fan-out merge),
//!     `from_erased_chain` (declarative `.toml` path) over linear-8 +
//!     fan-out/fan-in DAG shapes.
//!   - Status: `StatusHub::emit` for 1000 lossy `StageStep`s and 1000
//!     lossless lifecycle events (the per-step coordinator overhead).
//!   - Broker admission: the PURE `decide` box-fit math, `Drivers::estimate`,
//!     and `FootprintStore::resolve` (calibration lookup). `ResourceSnapshot::
//!     probe` is benched SEPARATELY and is syscall-bound (reads /proc/meminfo +
//!     shells out to nvidia-smi) — not a fair micro-bench, recorded for context.
//!   - Registry: `Registry::new` + `register` + `find` over a synthetic static
//!     cookbook. NOTE: blut is lib-only (recipes live in blut-lamquant), so we
//!     register a domain-free synthetic `RecipeDef` set, not the real cookbook.
//!   - Metric store: `fold_metrics` / `fold_gauges` over a synthetic
//!     status.jsonl, and `record_metrics` + `final_metrics` +
//!     `top_runs_by_metric` + `gpu_saturation` over a temp-DB populated with
//!     1000 metric rows (the `blut compare` / `hpo show` read cost).
//!
//! The live parallel coordinator loop is intentionally NOT micro-benched here:
//! it needs real async tasks + a tokio runtime + I/O, so it is profiled at
//! runtime (journald timings) rather than synthetically.

use std::sync::Arc;

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};

use blut::backends::TrainingBackend;
use blut::broker::{Drivers, FootprintSource, FootprintStore, ResourceSnapshot, decide};

/// Local backend fixture for the framework benches. The engine ships
/// no concrete backend after the v1.0 carve, and the in-crate test
/// fixture is `#[cfg(test)]`-only (invisible to benches, which compile
/// against the public non-test API). The benches need a
/// `TrainingBackend` to parameterize `Plan<Out, B>` + the toy stages'
/// `Compatible<B>` impls, so define one here.
struct LamuTrainerBackend;
impl TrainingBackend for LamuTrainerBackend {
    const ID: &'static str = "lamu";
    const DESCRIPTION: &'static str = "Bench fixture backend (engine framework benches only).";
}
use blut::framework::{
    Artifact, CacheHandle, Compatible, CompiledPlan, ContentHash, ErasedArtifact, Plan, Registry,
    Resource, Stage, StageContext, StageDyn, StageError, StageEvent, StatusHub,
};
use blut::lineage_db::{LineageDb, MetricRow};
use blut::recipes::{Course, RecipeDef};

fn bench_hash_file_10mib(c: &mut Criterion) {
    let td = tempfile::tempdir().unwrap();
    let p = td.path().join("blob.bin");
    let bytes = vec![0xAB_u8; 10 * 1024 * 1024];
    std::fs::write(&p, &bytes).unwrap();
    c.bench_function("hash_file 10 MiB", |b| {
        b.iter(|| {
            let h = ContentHash::hash_file(black_box(&p)).unwrap();
            black_box(h);
        });
    });
}

fn bench_hash_file_100mib_mmap(c: &mut Criterion) {
    // Above the 16 MiB threshold → mmap path.
    let td = tempfile::tempdir().unwrap();
    let p = td.path().join("blob.bin");
    let bytes = vec![0xEF_u8; 100 * 1024 * 1024];
    std::fs::write(&p, &bytes).unwrap();
    c.bench_function("hash_file 100 MiB (mmap)", |b| {
        b.iter(|| {
            let h = ContentHash::hash_file(black_box(&p)).unwrap();
            black_box(h);
        });
    });
}

fn bench_to_hex(c: &mut Criterion) {
    let h = ContentHash::of_bytes(b"x");
    c.bench_function("ContentHash::to_hex", |b| {
        b.iter(|| black_box(h).to_hex());
    });
}

fn bench_hash_dir_50_files(c: &mut Criterion) {
    let td = tempfile::tempdir().unwrap();
    // 50 × 1 MiB files — small-but-many shape typical of an HF
    // checkpoint after sharding.
    let blob = vec![0xCD_u8; 1024 * 1024];
    for i in 0..50 {
        std::fs::write(td.path().join(format!("shard-{i}.bin")), &blob).unwrap();
    }

    c.bench_function("hash_dir parallel (50 × 1 MiB)", |b| {
        b.iter(|| {
            let h = ContentHash::hash_dir(black_box(td.path())).unwrap();
            black_box(h);
        });
    });

    c.bench_function("hash_dir_serial (50 × 1 MiB)", |b| {
        b.iter(|| {
            let h = ContentHash::hash_dir_serial(black_box(td.path())).unwrap();
            black_box(h);
        });
    });
}

fn bench_cache_key(c: &mut Criterion) {
    let input_hash = ContentHash::of_bytes(b"x");
    let args = serde_json::json!({
        "lr": 2e-4,
        "epochs": 3,
        "batch_size": 1,
        "grad_accum": 8,
        "method": {"kind": "qlora", "rank": 16, "alpha": 32},
        "base_model": "Qwen/Qwen3-7B",
        "seq_len": 4096,
    });
    c.bench_function("cache key_for (Value, canonicalizes)", |b| {
        b.iter(|| {
            let k = CacheHandle::key_for(
                black_box("sft_train"),
                black_box(1),
                black_box(input_hash),
                black_box(&args),
                black_box(b"code-sha"),
            );
            black_box(k);
        });
    });

    // Opt-5: cache the canonical bytes once, reuse on every key
    // derivation. This is the path the executor takes per stage.
    let canon = CacheHandle::canonical_json_bytes(&args);
    c.bench_function("cache key_for_canon_bytes (precomputed)", |b| {
        b.iter(|| {
            let k = CacheHandle::key_for_canon_bytes(
                black_box("sft_train"),
                black_box(1),
                black_box(input_hash),
                black_box(&canon),
                black_box(b"code-sha"),
            );
            black_box(k);
        });
    });
}

fn bench_erased_round_trip(c: &mut Criterion) {
    use serde::{Deserialize, Serialize};
    use std::path::Path;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Toy {
        path: std::path::PathBuf,
        n: i64,
        meta: String,
    }
    impl Artifact for Toy {
        const KIND: &'static str = "test.toy";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(&self.n.to_le_bytes())
        }
        fn primary_path(&self) -> &Path {
            &self.path
        }
    }

    let toy = Toy {
        path: "/tmp/x".into(),
        n: 12345,
        meta: "lorem ipsum dolor sit amet".repeat(20),
    };
    c.bench_function("ErasedArtifact round trip", |b| {
        b.iter_batched(
            || toy.clone(),
            |toy| {
                let e = ErasedArtifact::from_typed(&toy).unwrap();
                let back: Toy = e.into_typed().unwrap();
                black_box(back);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_cache_write_then_read(c: &mut Criterion) {
    use serde::{Deserialize, Serialize};
    use std::path::Path;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Toy {
        path: std::path::PathBuf,
        n: i64,
        meta: String,
    }
    impl Artifact for Toy {
        const KIND: &'static str = "test.cache_toy";
        const SCHEMA: u32 = 1;
        fn content_hash(&self) -> ContentHash {
            ContentHash::of_bytes(&self.n.to_le_bytes())
        }
        fn primary_path(&self) -> &Path {
            &self.path
        }
    }
    let toy = Toy {
        path: "/tmp/x".into(),
        n: 12345,
        meta: "lorem ipsum dolor sit amet".repeat(20),
    };
    let art = ErasedArtifact::from_typed(&toy).unwrap();

    c.bench_function("cache insert + lookup round trip", |b| {
        b.iter_batched(
            || {
                let td = tempfile::tempdir().unwrap();
                let h = CacheHandle::job_local(td.path().to_path_buf());
                let key = ContentHash::of_bytes(b"bench");
                (td, h, key)
            },
            |(_td, h, key)| {
                h.insert(key, black_box(&art)).unwrap();
                let hit = h.lookup(key).expect("must hit");
                black_box(hit);
            },
            BatchSize::SmallInput,
        );
    });
}

// ─────────────────────────────────────────────────────────────────────────
// OPT-0: engine orchestration hot paths (plan / status / broker / registry /
// metric-store). Shared toy artifacts + stages mirror the unit-test scaffolds
// in `plan.rs` / `stage.rs` so the benched code path is the real one.
// ─────────────────────────────────────────────────────────────────────────

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ToyA;
impl Artifact for ToyA {
    const KIND: &'static str = "bench.toy_a";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(b"a")
    }
    fn primary_path(&self) -> &Path {
        Path::new(".")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ToyB;
impl Artifact for ToyB {
    const KIND: &'static str = "bench.toy_b";
    const SCHEMA: u32 = 1;
    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(b"b")
    }
    fn primary_path(&self) -> &Path {
        Path::new(".")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct BenchArgs;

/// () → A. The graph-input stage (every linear chain starts here).
struct MakeA;
impl Compatible<LamuTrainerBackend> for MakeA {}
#[async_trait]
impl Stage for MakeA {
    const NAME: &'static str = "make_a";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = ToyA;
    type Args = BenchArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: (),
        _args: &BenchArgs,
    ) -> Result<ToyA, StageError> {
        Ok(ToyA)
    }
}

/// A → A. A no-op identity link so an arbitrarily long linear chain
/// type-checks (`Output == next Input`).
struct AToA;
impl Compatible<LamuTrainerBackend> for AToA {}
#[async_trait]
impl Stage for AToA {
    const NAME: &'static str = "a_to_a";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ToyA;
    type Output = ToyA;
    type Args = BenchArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: ToyA,
        _args: &BenchArgs,
    ) -> Result<ToyA, StageError> {
        Ok(ToyA)
    }
}

/// A → B. Used as a fan-out/merge leaf so a shape has >1 distinct kind.
struct AToB;
impl Compatible<LamuTrainerBackend> for AToB {}
#[async_trait]
impl Stage for AToB {
    const NAME: &'static str = "a_to_b";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ToyA;
    type Output = ToyB;
    type Args = BenchArgs;
    async fn run(
        &self,
        _ctx: &StageContext,
        _input: ToyA,
        _args: &BenchArgs,
    ) -> Result<ToyB, StageError> {
        Ok(ToyB)
    }
}

/// Build an 8-node linear `CompiledPlan`: make_a → a_to_a ×7.
fn linear8_plan() -> CompiledPlan {
    let mut p = Plan::<(), LamuTrainerBackend>::new("linear8", serde_json::json!({}))
        .start(MakeA, BenchArgs);
    for _ in 0..7 {
        p = p.then(AToA, BenchArgs);
    }
    p.finish().into_compiled()
}

/// Build a fan-out shape: make_a forks into two parallel branches (one
/// `a_to_a`, one `a_to_b`). Exercises the topo-sort on a non-linear DAG with
/// two leading edges (the multi-root path the linear chain never hits).
fn fan_plan() -> CompiledPlan {
    Plan::<(), LamuTrainerBackend>::new("fan", serde_json::json!({}))
        .start(MakeA, BenchArgs)
        .fork(AToA, BenchArgs, AToB, BenchArgs)
        .finish()
        .into_compiled()
}

fn bench_plan_compile(c: &mut Criterion) {
    // topo_order — the per-run sort the executor + graph.json persist both call.
    let linear = linear8_plan();
    c.bench_function("plan topo_order (linear-8)", |b| {
        b.iter(|| black_box(black_box(&linear).topo_order().unwrap()));
    });

    let fan = fan_plan();
    c.bench_function("plan topo_order (fan-out/in)", |b| {
        b.iter(|| black_box(black_box(&fan).topo_order().unwrap()));
    });

    // graph_structure — the topo-sort + node/edge remap persisted at launch
    // (`plan.json`). This is the heavier per-run inspection path.
    c.bench_function("plan graph_structure (linear-8)", |b| {
        b.iter(|| black_box(black_box(&linear).graph_structure().unwrap()));
    });

    // from_components — the HPO fan-out merge of N independent trial plans into
    // one mega-plan. Bench a 16-trial × 8-node merge (128 nodes).
    c.bench_function("plan from_components (16×linear-8)", |b| {
        b.iter_batched(
            || (0..16).map(|_| linear8_plan()).collect::<Vec<_>>(),
            |comps| {
                let (merged, offs) = CompiledPlan::from_components(
                    black_box("hpo".into()),
                    serde_json::json!({}),
                    comps,
                );
                black_box((merged, offs));
            },
            BatchSize::SmallInput,
        );
    });

    // from_erased_chain — the declarative (.toml) path: build an 8-node chain
    // from boxed `Arc<dyn StageDyn>` with the runtime kind-check.
    c.bench_function("plan from_erased_chain (8 erased)", |b| {
        b.iter_batched(
            || {
                let a = serde_json::json!({});
                let mut v: Vec<(Arc<dyn StageDyn>, serde_json::Value)> =
                    vec![(Arc::new(MakeA), a.clone())];
                for _ in 0..7 {
                    v.push((Arc::new(AToA), a.clone()));
                }
                v
            },
            |chain| {
                let plan =
                    CompiledPlan::from_erased_chain("decl", serde_json::json!({}), chain).unwrap();
                black_box(plan);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_status_emit(c: &mut Criterion) {
    // Lossy class: 1000 StageStep events through the hub (broadcast only). This
    // is the per-training-step coordinator overhead — the high-volume path.
    c.bench_function("status emit 1000 StageStep (lossy)", |b| {
        b.iter_batched(
            // Fresh hub per batch; hold the lifecycle rx so the channel stays open.
            StatusHub::new,
            |(hub, _rx)| {
                for i in 0..1000u32 {
                    hub.emit(StageEvent::StageStep {
                        node_idx: 0,
                        stage_name: "train".into(),
                        update: serde_json::json!({ "step": i, "val_r": 0.42 }),
                    });
                }
                black_box(&hub);
            },
            BatchSize::SmallInput,
        );
    });

    // Lifecycle class: 1000 StageBegin events (lossless mpsc + broadcast). The
    // lower-volume audit-trail path; clones each event for the lossless channel.
    c.bench_function("status emit 1000 StageBegin (lifecycle)", |b| {
        b.iter_batched(
            StatusHub::new,
            |(hub, _rx)| {
                for i in 0..1000u32 {
                    hub.emit(StageEvent::StageBegin {
                        node_idx: i % 8,
                        stage_name: "train".into(),
                        input_hash: ContentHash::of_bytes(b"x"),
                    });
                }
                black_box(&hub);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_broker_admission(c: &mut Criterion) {
    // Drivers::estimate — the conservative-high RAM footprint formula a launch
    // computes once per run.
    let drivers = Drivers::new(4, 32, 3, 256, false, 168);
    c.bench_function("broker Drivers::estimate", |b| {
        b.iter(|| black_box(black_box(&drivers).estimate()));
    });

    // decide — the PURE box-fit / oversubscription math (NO syscalls). The
    // caller supplies the probed snapshot, so this is the admission DECISION
    // cost minus the probe.
    let snap = ResourceSnapshot {
        mem_total_gb: 62.0,
        mem_avail_gb: 50.0,
        vram_total_mib: Some(24576),
        vram_free_mib: Some(20000),
        gpus: Vec::new(),
    };
    let fp = drivers.estimate();
    c.bench_function("broker decide (pure box-fit)", |b| {
        b.iter(|| black_box(decide(black_box(&snap), black_box(&fp), 6.0)));
    });

    // FootprintStore::resolve — the calibration-store lookup that maps a key to
    // its measured/OOM-corrected peak (or falls back to the hint). Populate one
    // Measured entry so the hot branch (HashMap hit + Measured) is exercised.
    let store = {
        let td = tempfile::tempdir().unwrap();
        let mut s = FootprintStore::load_from(td.path().join("footprints.json"));
        let key = drivers.key("bench_recipe");
        s.record(&key, 20 * blut::broker::GIB, 0, FootprintSource::Measured)
            .unwrap();
        // Keep td alive for the bench duration by leaking it (the bench process
        // is short-lived; the OS reclaims on exit).
        std::mem::forget(td);
        s
    };
    let key = drivers.key("bench_recipe");
    c.bench_function("broker FootprintStore::resolve (measured hit)", |b| {
        b.iter(|| black_box(black_box(&store).resolve(black_box(&key), fp)));
    });

    // probe — SYSCALL-BOUND (reads /proc/meminfo + may shell out to nvidia-smi).
    // Benched for context only; it is NOT a fair CPU micro-bench. The admission
    // gate pays this once per launch (NOT per step).
    c.bench_function("broker ResourceSnapshot::probe (syscall-bound)", |b| {
        b.iter(|| black_box(ResourceSnapshot::probe()));
    });
}

// A synthetic, domain-free cookbook for the registry bench. blut is lib-only
// (the real recipes live in blut-lamquant), so we register a STATIC set of
// neutral `RecipeDef`s — this benches the registry machinery (Vec push + the
// `find`/`all` iteration), NOT the real cookbook's recipe count.
static BENCH_RECIPE: RecipeDef = RecipeDef {
    name: "bench_recipe",
    description: "synthetic RecipeDef for the OPT-0 registry bench",
    backend_id: "bench",
    category: Course::Train,
    input_kinds: &["bench.toy_a"],
    output_kind: "bench.toy_b",
    schedule: None,
    args_schema_fn: || serde_json::json!({"type": "object", "properties": {}}),
    compile_fn: |_raw| {
        Err(blut::framework::RecipeError::CompileFailed(
            "bench not runnable".into(),
        ))
    },
};
static BENCH_RECIPES: &[&RecipeDef] = &[&BENCH_RECIPE];

struct BenchCookbook;
impl blut::framework::Cookbook for BenchCookbook {
    fn name(&self) -> &'static str {
        "bench_cookbook"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        BENCH_RECIPES
    }
}

fn bench_registry_build(c: &mut Criterion) {
    // Cold-start: construct a Registry + register the synthetic cookbook +
    // resolve a recipe by name. The real cold-start cost in production is the
    // cookbook crate's static recipe slice (NOT reachable from lib-only blut);
    // this measures the registry's own machinery as a lower bound.
    c.bench_function("registry new + register + find", |b| {
        b.iter(|| {
            let mut r = Registry::new();
            r.register(Box::new(BenchCookbook));
            let hit = r.find(black_box("bench_recipe"));
            black_box(hit.is_some());
        });
    });
}

/// Set the jobs-dir env + write a synthetic `status.jsonl` for `job` with
/// `n_steps` training-metric steps + `n_gauges` gpu_gauge samples. Returns the
/// tempdir guard (kept alive by the caller) and the job id.
fn make_status_jsonl(n_steps: usize, n_gauges: usize) -> (tempfile::TempDir, String) {
    let td = tempfile::tempdir().unwrap();
    // SAFETY: benches run single-threaded per fn; we set this once before any
    // fold_* call and never race another thread on it.
    unsafe {
        std::env::set_var("LAMU_TRAIN_JOBS_DIR", td.path());
    }
    let job = "20260618-000000-opt0bench".to_string();
    let jdir = td.path().join(&job);
    std::fs::create_dir_all(&jdir).unwrap();
    let mut lines = String::new();
    for i in 0..n_steps {
        lines.push_str(&format!(
            r#"{{"kind":"stage_step","node_idx":1,"stage_name":"train","update":{{"step":{i},"val_r":0.42,"train_loss":1.5,"lr":0.0002,"grad_norm":3.1}}}}"#,
        ));
        lines.push('\n');
    }
    for i in 0..n_gauges {
        let wall = 1000 + i as i64;
        let util = if i % 10 == 0 { 12.0 } else { 95.0 };
        lines.push_str(&format!(
            r#"{{"kind":"stage_step","node_idx":1,"stage_name":"train","update":{{"kind":"gpu_gauge","wall_unix":{wall},"gpu_util":{util},"gpu_mem_mib":18000.0,"gpu_temp_c":70.0,"gpu_power_w":300.0}}}}"#,
        ));
        lines.push('\n');
    }
    std::fs::write(jdir.join("status.jsonl"), lines).unwrap();
    (td, job)
}

fn bench_metric_store(c: &mut Criterion) {
    // fold_metrics / fold_gauges — parse a ~1000-line status.jsonl into rows.
    // This is the `blut compare` ingest cost (status.jsonl → queryable rows).
    let (_td, job) = make_status_jsonl(1000, 200);
    c.bench_function("metric fold_metrics (1000-step status.jsonl)", |b| {
        b.iter(|| black_box(blut::framework::lineage::fold_metrics(black_box(&job)).unwrap()));
    });
    c.bench_function("metric fold_gauges (200-sample status.jsonl)", |b| {
        b.iter(|| black_box(blut::framework::lineage::fold_gauges(black_box(&job)).unwrap()));
    });

    // The SQL read cost over a populated DB. Build a temp DB with 1000 metric
    // rows (10 jobs × ~100 final-marker rows) + 200 gauge samples, then bench
    // the read queries `blut compare` / `hpo show` issue.
    let td = tempfile::tempdir().unwrap();
    let db = LineageDb::open_at(td.path().join("lineage.db")).unwrap();
    let mut rows: Vec<MetricRow> = Vec::with_capacity(1000);
    for j in 0..10 {
        let jid = format!("job-{j:02}");
        for k in 0..100 {
            rows.push(MetricRow {
                job_id: jid.clone(),
                node_idx: 1,
                // half intermediate steps, half final markers (-1) so the
                // step=-1 queries have real rows to scan.
                step: if k % 2 == 0 { k as i64 } else { -1 },
                metric: if k % 3 == 0 {
                    "val_r".into()
                } else {
                    "train_loss".into()
                },
                value: 0.40 + (j as f64) * 0.01 + (k as f64) * 0.0001,
                wall_unix: None,
            });
        }
    }
    db.record_metrics(&rows).unwrap();

    c.bench_function("metric final_metrics (1000-row DB)", |b| {
        b.iter(|| black_box(db.final_metrics(black_box("job-05")).unwrap()));
    });
    c.bench_function("metric top_runs_by_metric val_r (1000-row DB)", |b| {
        b.iter(|| black_box(db.top_runs_by_metric(black_box("val_r"), true, 10).unwrap()));
    });

    // gpu_saturation over a populated gauges table.
    let mut gauges = Vec::with_capacity(200);
    for i in 0..200i64 {
        gauges.push(blut::lineage_db::GaugeRow {
            job_id: "job-05".into(),
            node_idx: 1,
            wall_unix: 1000 + i,
            gpu_util: Some(if i % 10 == 0 { 12.0 } else { 95.0 }),
            gpu_mem_mib: Some(18000.0),
            gpu_temp_c: Some(70.0),
            gpu_power_w: Some(300.0),
            host_ram_mib: Some(40000.0),
            host_disk_free_mib: Some(500000.0),
        });
    }
    db.record_gauges(&gauges).unwrap();
    c.bench_function("metric gpu_saturation (200-sample DB)", |b| {
        b.iter(|| black_box(db.gpu_saturation(black_box("job-05"), 50.0).unwrap()));
    });
}

criterion_group!(
    benches,
    bench_hash_file_10mib,
    bench_hash_file_100mib_mmap,
    bench_hash_dir_50_files,
    bench_cache_key,
    bench_to_hex,
    bench_erased_round_trip,
    bench_cache_write_then_read,
);
criterion_group!(
    engine,
    bench_plan_compile,
    bench_status_emit,
    bench_broker_admission,
    bench_registry_build,
    bench_metric_store,
);
criterion_main!(benches, engine);
