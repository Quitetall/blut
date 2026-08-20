# archive/

Retired material, kept rather than deleted.

Nothing in this directory describes BLUT as it is today. Treat every file here
as a dated snapshot of a tree that no longer exists — the numbers, verdicts,
paths, and version strings inside were true when written and are not
re-verified. Each file carries its own header saying what it was and what
replaced it.

Retirement here is **sequester, not delete**. This repository publishes its full
history, so removing a file does not remove it — `git log -p` recovers it either
way. What deletion *would* remove is the note explaining that it is stale.
Keeping the file plus a banner is therefore strictly more honest than an absent
one, and it keeps the audit trail intact.

## Contents

| Path | What it was | Superseded by |
|---|---|---|
| `2026-05-28-state-review-and-test-plan.md` | Six-subsystem diagnostic of the pre-split, LamQuant-first, single-crate engine | ADR 0034 (engine is domain-agnostic); current capability boundaries live in [`docs/DISTRIBUTED_VALIDATION.md`](../docs/DISTRIBUTED_VALIDATION.md) |
| `experimental-k8s/` | Early Kubernetes packaging experiment | `crates/blut-operator` (unpublished, `publish = false`, separate Rust 1.89 lane) |

## Where current truth lives

- [`README.md`](../README.md) — what the engine is, and the scaling ladder with
  per-rung status.
- [`docs/DISTRIBUTED_VALIDATION.md`](../docs/DISTRIBUTED_VALIDATION.md) — the
  evidence ledger. Every capability is graded Validated / Component-tested /
  Deferred, with the boundary of what was actually exercised.
- [`CHANGELOG.md`](../CHANGELOG.md) — what changed per release.
- [`RELEASING.md`](../RELEASING.md) — the gates a release must clear.
