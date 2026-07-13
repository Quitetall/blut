# BLUT Release Procedure

Public preview train: `blut-types` → `blut` → `blut-dsl`, all at the same exact
version. `blut-worker` and `blut-operator` are unpublished.

## Candidate gate

Run from a clean checkout at the proposed tag SHA:

```bash
bash scripts/scope_guard.sh
cargo +1.88 fmt --all -- --check
cargo +1.88 check --locked --workspace --all-targets --all-features
cargo +1.88 clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo +1.88 test --locked --workspace --all-targets --all-features
RUSTDOCFLAGS="-D warnings" cargo +1.88 doc --locked --workspace --all-features --no-deps
cargo +1.88 check --locked -p blut-types --target wasm32-unknown-unknown
cargo deny --locked check
cargo deny --locked --manifest-path crates/blut-dsl/Cargo.toml --config deny.toml check
cargo deny --locked --manifest-path crates/blut-worker/Cargo.toml --config deny.toml check
cargo deny --locked --manifest-path crates/blut-operator/Cargo.toml --config deny.toml check
gitleaks git --log-opts="--all"
cargo +1.88 run --locked --example first_cookbook
bash scripts/run_benchmarks.sh compare
```

Also gate detached workspaces:

```bash
(cd crates/blut-dsl && cargo +1.88 fmt --all -- --check && cargo +1.88 clippy --locked --all-targets -- -D warnings && cargo +1.88 test --locked)
(cd crates/blut-worker && cargo +1.88 fmt --all -- --check && cargo +1.88 clippy --locked --all-targets -- -D warnings && cargo +1.88 test --locked)
(cd crates/blut-operator && cargo +1.89 fmt --all -- --check && cargo +1.89 clippy --locked --all-targets -- -D warnings && cargo +1.89 test --locked)
```

Required GitHub checks must be green at this exact SHA. Do not infer release
readiness from an older successful run.

Every commit in the candidate range must also have a recorded `PASS` or
`PASS WITH NITS` verdict from `mcp__local-llm__review_commit`. A missing or
unavailable reviewer is a release blocker, not a waiver. The benchmark command
above requires the committed `release-0.2` quiet-host baseline; establish it
with `bash scripts/run_benchmarks.sh --save` only on the designated quiet host.

## Registry sequence

Each publish is irreversible. Confirm package contents before removing
`--dry-run`.

```bash
(cd crates/blut-types && cargo publish --dry-run --locked)
(cd crates/blut-types && cargo publish --locked)
# Wait until crates.io resolves blut-types at the exact preview version.
# Re-run exact-SHA CI now; root and DSL tarball verification cannot resolve
# their exact owner dependencies until the preceding package is indexed.

cargo publish --dry-run --locked
cargo publish --locked
# Wait until crates.io resolves blut at the exact preview version.

(cd crates/blut-dsl && cargo publish --dry-run --locked)
(cd crates/blut-dsl && cargo publish --locked)
```

Then verify from an empty directory and clean Cargo home:

```bash
cargo new --lib blut-consumer
cd blut-consumer
cargo add blut@=0.2.0-alpha.1
cargo check
cargo install blut-dsl --version =0.2.0-alpha.1
```

Create `v0.2.0-alpha.1` only at the reviewed SHA. Release notes must link this
changelog, exact CI run, package checksums, and supported-version statement.
