# BLUT Distributed and Cross-Platform Validation

Evidence ledger for engine `0.2.0-alpha.1`. Status means:

- **Validated:** exercised on named runtime or exact CI gate.
- **Component-tested:** parser, command construction, or loopback component
  tested; no scale claim.
- **Deferred:** unsupported in public preview.

Last updated: 2026-09-19.

Nothing in the table below changed when the crates were published on
2026-08-27. Publication moves code, not evidence: every row still says exactly
what was and was not exercised, and the four Deferred rows are the ones to read
first if you are deciding whether to depend on this.

| Capability | Status | Evidence boundary |
|---|---|---|
| Linux systemd/cgroup2/rlimit containment selection | Validated | Local integration and failure-path tests |
| macOS build with degraded containment | CI-gated | Compile only; no systemd guarantee |
| `blut-types` on wasm32 | CI-gated | Rust 1.88 `wasm32-unknown-unknown` check |
| Single-node GPU admission and per-device permits | Component-tested | Deterministic scheduler and executor tests |
| Single-node `torchrun` argument construction | Component-tested | Launch contract tests; cookbook owns trainer semantics |
| Multi-node Slurm and Ray launchers | Component-tested | Command/parser tests only; no cluster-scale claim |
| P2P crypto, policy, transport, and loopback dispatch | Component-tested | Unit and loopback integration tests |
| P2P dispatch between separate network namespaces | Validated | Two containers, own IPs, real QUIC over a bridge — see below |
| Live multi-host P2P mesh | Deferred | Requires independent-host validation |
| Cloud queue over local object storage | Component-tested | Local-filesystem loopback smoke |
| Network object storage (S3/R2/GCS/MinIO) | Deferred | Removed from public preview pending dependency and infrastructure gates |
| `blut-worker` REST prototype | Deleted (ADR 0083 M3) | Superseded by `src/cloud` + the `blut-web` sidecar |
| Kubernetes operator | Unsupported | Unpublished prototype; separate Rust 1.89 compile/test lane |
| Windows containment | Deferred | Bare fallback only; Job Object implementation absent |

## Recorded run: P2P dispatch across network namespaces (2026-09-19)

The first P2P dispatch in this project that was not loopback.

Two containers on a user-defined bridge, each with its own network namespace,
its own IP and its own generated Ed25519/X25519 identity. The coordinator bound
`0.0.0.0:9320` inside its namespace and waited; the worker connected to the
coordinator's container IP over real QUIC and executed the dispatched stage.

```
coordinator 172.20.13.2:9320   worker connects from a second container
INFO P2P peer connected: c18e911091a8c91c (trust: anonymous)
worker c18e911091a8c91c connected — dispatching…
✔ stage 'p2p-echo' ran on peer c18e911091a8c91c; verified output:
CROSS-CONTAINER PROOF 2026-09-18
```

The output came back transformed and verified, so this exercised the whole data
plane — identity, connection, dispatch, remote execution, result verification —
not just a handshake.

Environment: kernel 7.2.5-1-cachyos, Docker 29.8.0, `archlinux@sha256:63c7b061c0c0`,
bridge subnet 172.20.13.0/24, `blut` 0.2.0-alpha.1 from crates.io built with
`features = ["p2p"]`. The published cookbook does not enable `p2p`, so the
driver was a throwaway consumer crate calling `blut::cli::run`.

**What this does NOT prove.** One kernel, one host, one virtual bridge. No NAT,
no packet loss, no latency, no partition, no clock skew, and no second machine.
It moves the claim from "loopback only" to "crosses a real network boundary
between independent network stacks", and no further. The row for a live
multi-host mesh stays Deferred until it runs on hardware that is genuinely
separate.

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
