# BLUT Release Procedure

Public preview train: `blut-types` → `blut` → `blut-dsl` → `blut-notify`, all at
the same exact version. The standalone-workspace sidecars `blut-tui`, `blut-web`,
and `blut-operator` are `publish = false` — they ship as **release binaries**
(the `release.yml` `binaries` job attaches them to the GitHub Release), not as
crates, because the engine ships as the CLI + its sidecars. (`blut-worker` was
deleted at ADR 0083 M3 — superseded by `src/cloud` + the `blut-web` sidecar.)

## Automation (ADR 0083 M6)

Two workflows carry the release; both are reviewable in `.github/workflows/`:

- **`release.yml`** — on a `v*` tag it runs the full gate and builds the sidecar
  binaries; the crate publish is a SEPARATE, MANUAL `workflow_dispatch`
  (`publish_crates=true`) behind the protected `crates-io` environment and the
  `CARGO_REGISTRY_TOKEN` secret, so the irreversible publish needs a deliberate
  human trigger + approval. Without the token the publish job refuses
  (fail-closed).
- **`docs.yml`** — builds the mdBook site (`book.toml` / `book_src`, which
  `{{#include}}`s the top-level docs so there is no drift) and deploys it to
  GitHub Pages on every `main` push that touches a doc.
- **`scripts/k8s_kind_smoke.sh`** — the operator gate: installs the
  `blut-operator` CRDs on an ephemeral `kind` cluster and asserts a `BlutPlan`
  CR is accepted by the API server (schema valid); reconcile-to-Job is a
  best-effort extra when the operator image is loaded.

The step-by-step below is the AUTHORITATIVE manual procedure the automation
mirrors — run it (or the `release.yml` publish job) at the reviewed tag SHA.

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
cargo deny --locked --manifest-path crates/blut-operator/Cargo.toml --config deny.toml check
gitleaks git --log-opts="--all"
cargo +1.88 run --locked --example first_cookbook
bash scripts/run_benchmarks.sh compare
```

Also gate detached workspaces:

```bash
(cd crates/blut-dsl && cargo +1.88 fmt --all -- --check && cargo +1.88 clippy --locked --all-targets -- -D warnings && cargo +1.88 test --locked && RUSTDOCFLAGS="-D warnings" cargo +1.88 doc --locked --no-deps)
(cd crates/blut-operator && cargo +1.89 fmt --all -- --check && cargo +1.89 clippy --locked --all-targets -- -D warnings && cargo +1.89 test --locked)
```

Required GitHub checks must be green at this exact SHA. Do not infer release
readiness from an older successful run.

Every commit in the candidate range must also have a recorded `PASS` or
`PASS WITH NITS` verdict from `mcp__local-llm__review_commit`. A missing or
unavailable reviewer is a release blocker, not a waiver. The benchmark command
above requires the committed `release-0.2` quiet-host baseline; establish it
with `bash scripts/run_benchmarks.sh --save` only on the designated quiet host.

## Public source gate

Crates.io publication must not precede source availability:

```bash
test "$(gh repo view Quitetall/blut --json visibility --jq .visibility)" = PUBLIC
```

Verify the README, security policy, CLA, and immutable validation links from an
unauthenticated browser before continuing. A private repository is a hard stop.

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
# Wait until crates.io resolves blut-dsl, then the last member of the chain:

cargo publish --dry-run --locked -p blut-notify
cargo publish --locked -p blut-notify
```

Run the kind smoke before tagging (the operator's CRD schemas are part of the
public contract):

```bash
bash scripts/k8s_kind_smoke.sh   # needs kind + kubectl
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
