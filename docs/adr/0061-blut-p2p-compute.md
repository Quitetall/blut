<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0061 — BLUT P2P distributed compute (trust model + encrypted task dispatch)

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (Proposed · Accepted 2026-08-27 · Complete — frozen 2026-08-27)

- **Extends:** [ADR 0046](0046-blut-resource-broker.md) (resource broker), [ADR 0034](0034-blut-scope-charter.md) (BLUT scope charter), [ADR 0044](0044-blut-api-standardization.md) (BLUT API standardization)
- **Follows:** the distributed-scaling work (RemoteJob trait, containment, multi-node Slurm)

## BLUT vision

BLUT is the git of ML training and DAG orchestration.

Git is a content-addressed file tracker that happens to be perfect for source control. You can use it for anything that benefits from content-addressed history. BLUT is a content-addressed DAG executor that happens to be perfect for ML training. You can use it for anything that benefits from deterministic, cached, parallel task execution.

The core feature is the ultimate DAG orchestrator. It is so well optimized for DAG orchestration that it just so happens to be able to do things that require DAGs with extreme low latency and speed — testing, data pipelines, evaluation, deployment, anything that decomposes into typed, verifiable, parallel tasks.

The architecture:

```
┌─────────────────────────────────────────────────┐
│  Applications                                   │
│  (ML training, data pipelines, evaluation,      │
│   testing, deployment — anything with a DAG)    │
├─────────────────────────────────────────────────┤
│  Features                                       │
│  (resource brokerage, containment, P2P, HPO     │
│   sweeps, lineage, TUI, multi-GPU, cluster)     │
├─────────────────────────────────────────────────┤
│  CORE: DAG Orchestrator                         │
│  (content-addressed cache, typed stages,        │
│   parallel execution, retry, verification)      │
└─────────────────────────────────────────────────┘
```

The core doesn't know about GPUs, or Python, or EEG. It knows about: **nodes, edges, hashes, and execution**. The features add: where to run, how to cap resources, what to optimize. The applications add: what to train, what to evaluate, what to test.

A one-stop shop for running training. A never-OOM tool that never catastrophically fails, never loses data, makes training deterministic and makes scaling simple. Training made simple, reliable, fast, and modular.

Works on one machine or 10. Works on an async network of 10 P2P connections or one cluster with hardware interconnects in sync. Works with a combination of async and synced machines.

The DAG doesn't care where a node runs — it cares that the output hash is correct.

## Context

BLUT's `ParallelExecutor` runs independent DAG nodes concurrently on one box. The `RemoteJob` trait dispatches to Slurm or Ray clusters. The next extension: dispatch DAG nodes to **peer GPUs** over the internet — external GPU owners assist in computing embarrassingly-parallel stages (data preprocessing, evaluation, encoding) while training stays local.

The constraint: **model weights and gradient states must never leave the coordinator**. EEG data is PHI under HIPAA. Random peers cannot be trusted with raw data.

## Decision

Add a `p2p/` module to the BLUT engine that implements:

1. **Trust model** — data classification × peer trust tiers → dispatch matrix
2. **Per-task encryption** — AES-256-GCM for data in transit, Ed25519 for authentication
3. **Content-addressed verification** — `blake3(output) == expected_hash` → zero-trust result validation
4. **QUIC transport** — encrypted-by-default networking via quinn
5. **Dispatch policy** — the executor consults the policy before spawning each DAG node; dispatchable nodes go to peers, training nodes stay local

### Trust model

**Data classification** (per-corpus, set by the user):

| Class | What | Examples |
|-------|------|----------|
| `Public` | Open datasets | TUH, Physionet, synthetic |
| `Internal` | Proprietary non-clinical | Lab recordings, dev data |
| `Restricted` | Clinical / PHI | Hospital EEG, patient records |

**Peer trust tiers** (set by the coordinator):

| Tier | Identity | Verification |
|------|----------|-------------|
| `Trusted` | Lab collaborator | Ed25519 key + coordinator whitelist |
| `Registered` | Verified external | Ed25519 key + identity record |
| `Anonymous` | Anyone | Self-generated Ed25519 key |

**Dispatch matrix** (enforced by the coordinator before every task dispatch):

```
                    Trusted    Registered    Anonymous
Public data           ✅          ✅            ✅
Internal data         ✅          ✅            ❌
Restricted data       ✅          ❌            ❌
Training (gradients)  local       local         local
```

Training stages are **always local**. Hivemind-style async gradient averaging over the internet is a future extension (requires differential privacy), not part of this ADR.

### What's protected

- **Model weights**: task-parallel stages don't load models. Peers never see weights.
- **Gradient states**: training stays local. No gradients are transmitted.
- **Data**: AES-256-GCM encrypted in transit; plaintext exists only in the peer's process memory during computation; chunked (one .lma file per task, not the whole corpus).
- **Architecture**: task manifest reveals stage names + arg shapes, not model internals.

### Threat mitigations

| Threat | Mitigation |
|--------|-----------|
| Peer copies input data | Encrypted in transit; plaintext only in memory; chunked per-task |
| Peer sends garbage output | Content-addressed verification (blake3 hash check) |
| Network eavesdropper | QUIC (TLS 1.3) for all transport |
| Peer replays old results | Cache key = `(stage, input_hash, args_hash)` — unique per run |
| Coordinator impersonation | Ed25519 signature on task manifests |
| Peer Sybil attack | Rate limit per peer; reputation scoring |
| Data exfiltration via timing | Wall-time monitoring; anomalies flagged |

### Architecture

```
Coordinator (your BLUT instance)
├── ParallelExecutor
│   ├── warm_cache         → dispatch to Peer A (Public data, Anonymous OK)
│   ├── precompute_feats   → dispatch to Peer B (Internal data, Registered+)
│   ├── train_model        → LOCAL ONLY (training)
│   ├── quality_gate       → dispatch to Peer C (Public data, Anonymous OK)
│   └── export_artifact    → LOCAL (small, fast)
│
├── QUIC server (port 9320)
│   ├── Accepts peer connections
│   ├── Verifies Ed25519 pubkey → assigns trust level
│   ├── Encrypts task input with peer's key
│   ├── Sends TaskManifest
│   ├── Receives TaskResult
│   └── Verifies output hash → inserts into cache
│
└── PeerRegistry (~/.config/blut/peers.json)
    ├── Peer A: Anonymous, RTX 4090, 24GB VRAM
    ├── Peer B: Registered, A100, 80GB VRAM
    └── Peer C: Trusted, RTX 3090, 24GB VRAM

Peer (external GPU owner)
├── QUIC client → connects to coordinator
├── Receives TaskManifest
├── Decrypts input (AES-256-GCM with sealed key)
├── Executes stage (same deterministic code as local)
├── Hashes output (blake3)
├── Encrypts output
└── Sends TaskResult back
```

### Transport: QUIC (quinn)

- TLS 1.3 built in (no separate TLS setup)
- Multiplexed streams (task submit, result return, cancel signal on one connection)
- 0-RTT resumption for known peers
- Async/tokio-native

### Module structure

```
src/p2p/
├── mod.rs           — re-exports
├── trust.rs         — DataClass, TrustLevel, dispatch matrix
├── crypto.rs        — Ed25519 sign/verify, AES-256-GCM encrypt/decrypt, key sealing
├── peer.rs          — PeerId, PeerInfo, PeerCapabilities
├── registry.rs      — PeerRegistry (persisted to JSON)
├── task.rs          — TaskManifest, TaskResult, EncryptedPayload
├── dispatch.rs      — DispatchPolicy trait, DefaultDispatchPolicy
├── job.rs           — P2pJob impl of RemoteJob trait
├── transport.rs     — QUIC server/client (quinn)
└── coordinator.rs   — advertise tasks, collect results, verify hashes
```

### Executor integration

Minimal — a new `dispatch_policy` field on `ExecCtx`:

```rust
// In ParallelExecutor::execute() spawn loop:
if let Some(peer) = dispatch_policy.should_offload(&node, &peers, data_class) {
    let task = build_task_manifest(&node, &env, &peer, data_class)?;
    let job = coordinator.submit_task(task, peer).await?;
    remote_jobs.insert(node.id, job);
} else {
    join.spawn(async move { run_node(task, env_c).await });
}
```

Existing `run_node` path is unchanged. The dispatch policy is consulted before spawning, not after.

### New crate dependencies

```toml
quinn = "0.11"          # QUIC transport
rustls = "0.23"         # TLS 1.3
ed25519-dalek = "2.1"   # Ed25519 signing
aes-gcm = "0.10"        # AES-256-GCM encryption
x25519-dalek = "2.0"    # X25519 key exchange
```

## Consequences

- DAG nodes classified as dispatchable (precompute, encode, gate, export) can run on peer GPUs
- Training stages remain local — no gradient exposure
- Data is encrypted in transit, chunked per-task, verified by hash
- The trust matrix is enforced by the coordinator — a peer cannot request data above its trust level
- Content-addressed cache already provides integrity verification — no trust in peer output required
- The `RemoteJob` trait is the integration seam — `P2pJob` sits alongside `SlurmJob` and `RayJob`

## Anti-reinvention

- Hivemind (Learning@Home) solves async gradient averaging — NOT this problem. This is task-parallel dispatch, not distributed training.
- io.net / Vast.ai / Akash solve GPU rental — NOT this problem. This is DAG-node-level dispatch with encrypted inputs, not "rent a VM."
- BOINC solves distributed scientific computing — similar model but no encryption/trust/BLUT integration.
- The content-addressed cache is the trust anchor. It already exists. The P2P module extends it to the network.

## Progress Log

- _2026-08-27_ — gave this record a concrete citation gate `[redacted: private path]` (it proves the record still points at files that exist; it does not claim the decision is implemented) and removed its closure-debt entry `[redacted: private path]`.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

## Completion / Resolution

**Complete — frozen 2026-08-27 — dispositioned under ADR 0186 [not federated] clause 8.**

- **verdict:** COMPLETE
- **evidence:** All five numbered items exist. `src/p2p/` carries `trust.rs` (1, the classification x tier dispatch matrix), `crypto.rs` (2), `dispatch.rs` (5, the executor's policy seam) and `transport.rs` (4), and the three cryptographic dependencies the Decision names are declared in `Cargo.toml`: ed25519-dalek 2.1 (:112), aes-gcm 0.10 (:113) and quinn 0.11 (:121). The module also carries `coordinator.rs`, `gossip.rs`, `mesh_runner.rs`, `fedround.rs` and `privacy.rs` beyond the scope of this ADR.
