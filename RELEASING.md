# BLUT Release Procedure

Public preview train: `blut-types` → `blut` → `blut-dsl` → `blut-notify`.
`blut-types` is the single chain ROOT (no in-repo path deps); `blut` depends on
it, so it must be indexed on crates.io before `blut` will package — otherwise
`no matching package named 'blut-types' found`. Members share the exact preview
version.

`blut-graph-core` is NOT part of this train. The ADR 0139 adapter that used it
moved out of the engine into the unpublished `crates/blut-semantic` sidecar
(ADR 0034 — graph-core carries ABIR domain vocabulary and the engine is
domain-agnostic), so nothing in the chain depends on it. Publish it on its own
schedule once its surface settles; it is the newest and fastest-moving crate in
the repo.

`blut-tui` IS published (0.2.0-alpha.1 is on crates.io) and its manifest now
says so. The standalone-workspace sidecars `blut-web` and `blut-operator` are
`publish = false`. The `release.yml` `binaries` job
attaches those three plus the runnable `blut-notify` member binary to the GitHub
Release. (`blut-worker` was deleted at ADR 0083 M3 — superseded by `src/cloud`
and the `blut-web` sidecar.)

## Automation (ADR 0083 M6)

Two workflows carry the release; both are reviewable in `.github/workflows/`:

- **`release.yml`** — on a `v*` tag it runs the release-critical gates, including
  the real kind CRD-admission smoke, and builds all four sidecar/member binaries;
  the crate publish is a SEPARATE, MANUAL `workflow_dispatch`
  (`publish_crates=true`) behind the protected `crates-io` environment and the
  `CARGO_REGISTRY_TOKEN` secret. The operator must also enter this exact
  candidate SHA as `quiet_benchmark_sha`, proving the designated-host benchmark
  gate was run for the bytes being published. A missing/mismatched SHA or token
  refuses before publication (fail-closed).
- **`docs.yml`** — builds the mdBook site (`book.toml` / `book_src`, which
  `{{#include}}`s the top-level docs so there is no drift) and deploys it to
  GitHub Pages on every `main` push that touches a doc.
- **`scripts/k8s_kind_smoke.sh`** — the operator gate, run by `release.yml` and
  available locally: installs the
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
(cd crates/blut-web && cargo +1.88 fmt --all -- --check && cargo +1.88 clippy --locked --all-targets -- -D warnings && cargo +1.88 test --locked)
(cd crates/blut-tui && cargo +1.88 fmt --all -- --check && cargo +1.88 build --locked --all-targets && cargo +1.88 clippy --locked --all-targets -- -D warnings && cargo +1.88 test --locked && ./target/debug/blut-tui --check)
(cd crates/blut-operator && cargo +1.89 fmt --all -- --check && cargo +1.89 clippy --locked --all-targets -- -D warnings && cargo +1.89 test --locked)
```

Required GitHub checks must be green at this exact SHA. Do not infer release
readiness from an older successful run.

Every commit in the candidate range must have been reviewed, and the review
recorded. This used to name `mcp__local-llm__review_commit` as the only
acceptable reviewer and call an unavailable one a release blocker. That cannot
survive publication: it is a private, local tool that no outside contributor can
run, so as written the procedure said nobody but this machine may cut a release.

What the requirement actually protects is that no commit reaches a release
unexamined. Human review on a pull request satisfies it. So does a recorded
model review where one is available. What does not satisfy it is a commit that
went straight to `main` with nothing but CI behind it — CI checks that the code
builds and passes its gates, which is not the same as somebody having read it. The benchmark command
above requires the committed `release-0.2` quiet-host baseline; establish it
with `bash scripts/run_benchmarks.sh --save` only on the designated quiet host.
GitHub's heterogeneous hosted runners compile the benchmark harness but do not
claim comparable timings; the manual publish job requires the exact-SHA quiet
host attestation instead.

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

## Tagging

Tag at the reviewed SHA, **after** publishing, one tag per crate published:

```bash
git tag -a blut-types-v0.2.0-alpha.1 -F tagmsg.txt <sha>
git push origin blut-types-v0.2.0-alpha.1
```

**The `<crate>-v<version>` shape is load-bearing, not cosmetic.** A single
umbrella tag cannot be accurate: the 0.2.0-alpha.1 train shipped from two
commits (`blut`/`blut-types`/`blut-notify` from `e34c557`, `blut-dsl`/`blut-tui`
from `ffecee5`), and `blut-graph-core` releases on its own schedule entirely.
The prefix also keeps these tags clear of `release.yml`'s `v*` trigger, so
recording a release fires no build — which is what makes backfilling history
safe.

Verify the tag against what was actually published rather than trusting the
commit you think you cut it from. Every published `.crate` carries
`.cargo_vcs_info.json`:

```bash
curl -sL https://static.crates.io/crates/<crate>/<crate>-<version>.crate \
  | tar xzO '<crate>-<version>/.cargo_vcs_info.json'
```

Release notes must link this changelog, the exact CI run, package checksums, and
the supported-version statement. Mark alpha releases prerelease, or GitHub will
advertise one as "Latest".

## Two gates in this document do not pass today

Both are recorded here rather than quietly worked around.

**`scripts/run_benchmarks.sh compare` cannot pass.** It defaults to the baseline
name `release-0.2`, and the only baseline committed under `bench-baselines/` is
`opt0`, so it exits 2 with `missing committed benchmark baseline 'release-0.2'`.
Either establish the baseline on the designated quiet host
(`bash scripts/run_benchmarks.sh --save`) or run the comparison against the
baseline that exists (`BLUT_BENCH_BASELINE=opt0`). Do not delete the check.

**`release.yml`'s `gate` job ends BLOCKED by design.** Its last step runs
`tools/check_blut_release_state.py`, which reports BLOCKED whenever the two
external cookbook components (`blut-cookbook-standard`, `blut-cookbook-lamquant`)
are not checked out — which they never are in CI. A `v*` tag therefore produces a
red release run that is not evidence of a problem. `ci.yml` treats any verdict of
exit ≤ 2 as a report rather than a failure; until `release.yml` does the same, or
the externals are fetched, read that red with this paragraph in hand.
