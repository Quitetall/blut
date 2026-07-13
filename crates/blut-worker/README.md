# blut-worker — DEPRECATED

> **Status: deprecated (2026-07-09). Scheduled for deletion in the M3 slice of
> the premier-heavyweight roadmap.** Do not build new functionality here.

This crate was the v0 cloud-worker prototype: a file-based job queue plus a
token-guarded axum REST API. Both halves are superseded:

The retained hardening harness accepts submissions only through its loopback
REST API. Direct queue-file publication and non-loopback serving are refused.

- **Remote execution** → the `cloud` feature in the engine (`src/cloud/`,
  ADR 0082): an object-store-dispatch queue with *leases* (visibility timeout +
  `reclaim_expired`) and lease-fenced completion — the durability this crate's
  file-extension state machine lacks — reusing the P2P data plane's fail-closed
  bundle gates and the clinical-hard-block dispatch matrix.
- **API surface** → the planned `crates/blut-web` sidecar (ADR 0083): a
  read-only+control API host serving the same artifacts the TUI reads.

The one lasting contribution is the **sidecar precedent**: an HTTP server is
legal in `crates/*` but never in the engine (`blut/src`), enforced by
`scripts/scope_guard.sh` in CI. ADR 0083 documents that boundary properly.

Kept building in CI until deletion so the workspace stays green; no new
features, no new deps.
