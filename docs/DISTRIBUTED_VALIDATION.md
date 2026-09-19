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
| Ray launcher against a live cluster | Validated | Two-node Ray 2.x, submit + stream + poll to Succeeded, 2026-09-19 |
| Multi-node Slurm launcher | Component-tested | Command/parser tests only; see the Slurm note below |
| P2P crypto, policy, transport, and loopback dispatch | Component-tested | Unit and loopback integration tests |
| P2P dispatch between separate network namespaces | Validated | Two containers, own IPs, real QUIC over a bridge — see below |
| Live multi-host P2P mesh | Deferred | Requires independent-host validation |
| Cloud queue over local object storage | Validated | `blut cloud smoke` round-trip through the shipped cookbook binary, 2026-09-19 |
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

## Recorded run: cloud queue through the shipped binary (2026-09-19)

Run with `blut-standard` built from `blut-cookbook-standard --features
distributed` — an installed-shape binary, not a test harness:

```
submitting p2p-echo to the cloud queue (store: …/cloudstore)…
running a cloud worker…
✔ cloud round-trip verified over …/cloudstore; output:
DOGFOOD CLOUD QUEUE 2026-09-19
```

Submit, worker pickup, execution and verification all ran; the output came back
transformed, so this is the queue working rather than a file being written.

**Boundary:** the object store is the local filesystem. Network object storage
(S3/R2/GCS/MinIO) is a different row and stays Deferred — it needs the
`object_store` feature set this crate does not enable, plus the advisory, secret
handling and TLS gates that removed it from the preview.

**Reachability, which was the real defect.** Until 2026-09-19 neither this nor
the P2P run was possible from anything BLUT ships. The engine is lib-only and
gates both behind cargo features; the published cookbook left them off, so the
subcommands were compiled out of the only shipped binary. `blut-cookbook-standard`
now exposes `p2p`, `cloud`, `raft` and `distributed`, and both runs above were
performed through it.

## Recorded run: Ray launcher against a live cluster (2026-09-19)

Two containers on a bridge — a Ray head and a worker, 40 CPUs — driven through
`blut::config::launcher::RayLauncher`, not by calling `ray` by hand:

```
submitted id=blut-ray-proof-fixed
  [log] BLUT_RAY_PROOF node= rayhead cpus= 40.0
terminal state: Succeeded
```

Submission, placement, log streaming and terminal-state polling all ran.

**It found a real defect on the first attempt.** `parse_ray_status` scanned for
a `Status:` line that Ray 2.x does not emit for a terminal job, so `poll()`
returned `Unknown` for every finished job — the launcher could not observe a job
succeed or fail, and never had been able to. Its seven unit tests asserted on
the same invented string, so they agreed with the parser and with nothing else.
Fixed against captured output; the run above is the same cluster after the fix.

**Slurm is not validated and this run says nothing about it.** A real `sbatch` /
`sacct` / `scancel` path needs accounting (slurmdbd + a database) and, on Debian
12's Slurm 22.05 with cgroup/v2, a systemd instance as PID 1 for `slurmd` to
create its scope over dbus. That is an environment constraint rather than
anything about BLUT, and it was not solved here. The row stays
Component-tested.

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
