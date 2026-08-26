# Contributing to blut

Thanks for your interest in `blut` (Basically Less Unsound Training;
affectionately, Brian Lam's Universal Trainer) — a
Rust-native, compile-time-typed orchestration framework for local ML
training. It's a **library crate**, not an application: you depend on it,
you don't install a tool from it.

## Before you open a PR

The crate must be clean under all four of these. CI enforces `fmt` and
`clippy -D warnings` on Ubuntu, plus a macOS compile-check job — so run
them locally first:

```bash
cargo fmt --all
cargo build --locked
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
```

Then sanity-check the core path end to end:

```bash
cargo run --locked --example first_cookbook
```

`first_cookbook` builds a tiny typed plan, runs it, and runs it again to
show the content-addressed cache skipping every stage. If it stops
behaving, you broke something load-bearing.

## What belongs here — and what doesn't

`blut` is **domain-agnostic**. It ships ZERO concrete cookbooks or
recipes. The engine knows about stages, plans, artifacts, caching,
containment, and observability — nothing about EEG, LLMs, or any other
problem domain.

Domain stages and recipes live in a **separate cookbook crate that
depends on `blut`**. See `examples/first_cookbook.rs` for the extension
pattern (a backend identity, a typed artifact, typed stages, a plan).

PRs that add domain-specific logic (EEG/LLM/codec/etc.) to the engine
will be redirected to a cookbook crate. If you think the engine is
missing a *generic* seam that your domain needs, open an issue describing
the seam, not the domain.

## API preview

The current `0.2.0-alpha.1` surface is a preview of the intended 1.0 contract —
see the "Preview surface" section in [`API.md`](API.md). Until the M6 release
gate, downstream cookbooks should pin the exact preview version.

- **Prefer additive changes.** New items, new optional arguments, and new
  variants behind `#[non_exhaustive]` reduce preview churn.
- **Breaking preview changes need discussion first.** Open an issue describing
  the break and migration before writing the PR. Once 1.0 ships, breaking
  changes require a major bump.

## Commit style

Conventional-commit-style messages preferred:

```
feat: add Network resource throttle to the executor
fix: don't double-prune the cache on resume
refactor: collapse StageEvent into StatusUpdate
docs: clarify the cookbook seam
test: cover the off-systemd fallback path
style: rustfmt the executor module
```

## Containment is Linux + systemd only

The containment features (per-stage systemd-cgroup units, `MemoryMax`)
require **Linux with systemd**. On any host without systemd, `blut`
degrades to a bare process spawn — no cgroups, no memory cap.

Do not assume cgroups in tests. A test that requires containment to be
active will fail on macOS CI and on non-systemd Linux. Gate such tests on
the runtime check, or keep them as opt-in integration tests.

## Reporting bugs / requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For bugs, the
Linux+systemd answer matters — it changes which code path ran.

## License of contributions

`blut` is licensed under **AGPL-3.0-or-later**. By submitting a contribution
you agree it is licensed under those same terms. Because the project also
offers a commercial license, contributions require signing this repository's
[`CLA.md`](CLA.md) before they can be merged.
