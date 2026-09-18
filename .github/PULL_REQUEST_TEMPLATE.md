<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
## What this changes

<!-- One or two sentences. What behaviour is different after this lands? -->

## Why

<!-- The problem, not the patch. If it fixes an issue, link it: Fixes #123 -->

## How it was verified

<!-- Commands you actually ran and what they printed. "Should work" is not
     verification; neither is a gate you did not watch fail. If you added or
     changed a gate, say how you made it fail on purpose. -->

## Checklist

These are the gates CI runs, on the pinned toolchain. Running them locally first
is faster than a round trip.

- [ ] `cargo +1.88 fmt --all -- --check`
- [ ] `cargo +1.88 clippy --locked --workspace --all-targets -- -D warnings`
- [ ] `cargo +1.88 test --locked --workspace`
- [ ] Touched a sidecar under `crates/`? It is a **separate workspace** with its
      own lockfile — run the same three in that directory.
- [ ] Changed a dependency? `bash scripts/dependency_policy.sh` (all ten
      workspaces).
- [ ] Changed public API or behaviour? `CHANGELOG.md` updated under
      `## [Unreleased]`.
- [ ] New MSRV requirement? Say so — it is a minor bump for `0.x` and belongs in
      the changelog.

## Scope

- [ ] This belongs in the **engine**, not a cookbook. The engine is
      domain-agnostic: it ships no domain stages, no recipes, no HTTP server and
      no dynamic plugins. `scripts/scope_guard.sh` enforces part of that and
      will tell you if it disagrees.

---

First contribution? The CLA bot will comment with what to do. Contributions are
AGPL-3.0-or-later; see [CONTRIBUTING.md](../CONTRIBUTING.md) and
[CLA.md](../CLA.md).
