# BLUT Distributed and Cross-Platform Validation

Evidence ledger for engine `0.2.0-alpha.1`. Status means:

- **Validated:** exercised on named runtime or exact CI gate.
- **Component-tested:** parser, command construction, or loopback component
  tested; no scale claim.
- **Deferred:** unsupported in public preview.

Last updated: 2026-07-12.

| Capability | Status | Evidence boundary |
|---|---|---|
| Linux systemd/cgroup2/rlimit containment selection | Validated | Local integration and failure-path tests |
| macOS build with degraded containment | CI-gated | Compile only; no systemd guarantee |
| `blut-types` on wasm32 | CI-gated | Rust 1.88 `wasm32-unknown-unknown` check |
| Single-node GPU admission and per-device permits | Component-tested | Deterministic scheduler and executor tests |
| Single-node `torchrun` argument construction | Component-tested | Launch contract tests; cookbook owns trainer semantics |
| Multi-node Slurm and Ray launchers | Component-tested | Command/parser tests only; no cluster-scale claim |
| P2P crypto, policy, transport, and loopback dispatch | Component-tested | Unit and loopback integration tests |
| Live multi-host P2P mesh | Deferred | Requires independent-host validation |
| Cloud queue over local object storage | Component-tested | Local-filesystem loopback smoke |
| Network object storage (S3/R2/GCS/MinIO) | Deferred | Removed from public preview pending dependency and infrastructure gates |
| `blut-worker` REST prototype | Deleted (ADR 0083 M3) | Superseded by `src/cloud` + the `blut-web` sidecar |
| Kubernetes operator | Unsupported | Unpublished prototype; separate Rust 1.89 compile/test lane |
| Windows containment | Deferred | Bare fallback only; Job Object implementation absent |

## Required evidence before stronger claims

- Record real multi-node NCCL runs with topology, versions, global-batch parity,
  failure recovery, and artifact hashes.
- Exercise P2P across independent hosts and hostile-network cases.
- Restore network object-store support only after dependency advisories, secret
  handling, TLS, and provider integration tests pass.
- Publish cookbook-specific DDP evidence in each cookbook repository. Engine
  tests prove orchestration contracts, not model-training correctness.

Exact test counts belong to CI at a commit SHA and are intentionally omitted;
counts drift while capability evidence should remain auditable.
