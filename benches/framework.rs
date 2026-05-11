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
//!   - ErasedArtifact JSON round-trip

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use blut::framework::{Artifact, CacheHandle, ContentHash, ErasedArtifact};

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
    c.bench_function("cache key_for", |b| {
        b.iter(|| {
            let k = CacheHandle::key_for(
                black_box("sft_train"),
                black_box(1),
                black_box(input_hash),
                black_box(&args),
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

criterion_group!(
    benches,
    bench_hash_file_10mib,
    bench_hash_file_100mib_mmap,
    bench_hash_dir_50_files,
    bench_cache_key,
    bench_to_hex,
    bench_erased_round_trip,
);
criterion_main!(benches);
