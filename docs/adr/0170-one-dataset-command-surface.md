<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0170 — One dataset command surface

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (Proposed 2026-08-13 → In progress → Complete — frozen 2026-08-13)

Child of ADR 0167 [not federated] invariant 3.

## Context

BLUT exposes three top-level owners of one dataset noun: `catalog`, `data`, and
`dataset`. Their verbs are disjoint, so users must know implementation history
before choosing a command. A single-command-surface check `[redacted: private path]`
now demonstrates and reports all three owners.

## Decision

`blut dataset` becomes the sole top-level dataset namespace.

- Recording/import verbs from `blut data` move under `blut dataset ingest` and
  related explicit verbs.
- Pin/resolve verbs remain under `blut dataset`.
- Read-only catalog projection verbs move under `blut dataset catalog`.
- Exit codes, JSON schemas, and machine-readable diagnostics stay stable or get
  an explicit version bump.
- Old top-level nouns become fail-loud migration errors for one release, not
  hidden aliases; aliases would preserve multiple owners and defeat the gate.
- TUI, docs, shell completion, and tests consume the same command registry.

## Rationale

One namespace preserves BLUT's noun ownership while keeping implementation
verbs decomposable. A shared registry also retains current fused execution: CLI
parsing changes, not data-plane boundaries.

## Alternatives Considered

- Keep three commands with better help. Rejected: same noun still has three
  authorities.
- Make `catalog` the root. Rejected: catalog is a projection, not lifecycle
  authority.
- Compatibility aliases forever. Rejected: gate could never prove one owner.

## Consequences

Scripts need one mechanical migration. Help text becomes larger but coherent.
Read-only catalog behavior remains separable behind its existing API.

## Implementation Plan

- Inventory every current verb, flag, exit code, JSON schema, caller, and doc.
- Build one command registry and golden help/JSON fixtures.
- Move verbs without changing implementation services.
- Add fail-loud diagnostics for removed top-level nouns.
- Migrate TUI, CI, docs, and scripts; remove old enum variants.

- **Gate:** the single-command-surface check `[redacted: private path]` plus CLI
  golden tests.
- **Pass when:** exactly one top-level dataset owner exists and every former verb
  has a tested new route or intentional retirement record.
- **Rollback:** restore old parsing variants while retaining shared service APIs.

## Progress Log

- _2026-08-13 — proposed._ Parent gate reports `catalog`, `data`, and `dataset`.
- _2026-08-13 — landed._ BLUT engine commits `3603063` and `d89a153` make
  `dataset` the sole top-level owner. Mutable source verbs live under `dataset
  ingest`; immutable bindings remain at `dataset pin|resolve`; projection verbs
  live under `dataset catalog`. Removed `data` and `catalog` roots reach the
  existing external-command seam, which returns explicit ADR 0170 migration
  routes before any sidecar lookup. Table-driven tests parse all ten canonical
  verbs and prove removed roots are no longer built-ins.

## Completion / Resolution

- **verdict:** `passed`
- **gate_cmd:** the single-command-surface check `[redacted: private path]` `&& cargo
  test --locked -p blut --lib && cargo clippy --locked -p blut --lib -- -D
  warnings && cargo fmt --all -- --check`
- **result:** one owner (`dataset`); 913/913 full BLUT library tests passed;
  follow-up CLI suite 74/74 passed; clippy and formatting clean.
- **git_sha:** BLUT `3603063`, `d89a153` · **artifact sha256:** n/a

## Related Decisions

[ADR 0034](0034-blut-scope-charter.md), [ADR 0100](0100-blut-data-catalog.md), ADR 0167 [not federated].

## Validation

Watch automation compatibility, JSON schema drift, help/completion parity, and
accidental duplicate ownership in TUI or SDK wrappers.
