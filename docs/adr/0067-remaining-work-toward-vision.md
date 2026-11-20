<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0067 — Remaining work toward the BLUT vision (the gated TODO)

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `in-progress` (Proposed 2026-06-30 as a living tracking ADR · Accepted 2026-08-27 · In progress 2026-08-27)

- **Extends:** [ADR 0061](0061-blut-p2p-compute.md) (P2P / distributed scaling), ADR 0065 [not federated], ADR 0066 [not federated]
- **Follows:** an FSDP/async-activation feature branch `[redacted: private repository]` (Slices A/B/C built; this ADR catalogues the tail)

> This ADR is a **map of what is not yet done**, not a single decision. Its job is
> to make the frontier legible: every outstanding item toward the vision, classified
> by what blocks it (pure-software / hardware-gated / infra-gated), each pinned to the
> **one runnable test that closes it**, so work resumes the instant a resource appears.
> It freezes only when the table is empty or superseded.

## Context

BLUT is the **git of ML training**: a typed DAG orchestrator whose every feature —
brokerage, containment, DDP/FSDP, P2P, HPO, lineage, cloud queue — is a layer that
never touches the core, scaling `1 GPU → multi-GPU → multi-node → cluster → P2P mesh
→ 100 datacenters` without a rewrite.

As of 2026-06-30 the spine is built and three slices have largely landed on
a feature branch:

- **Slice A (FSDP + async)** — FSDP2 (`fully_shard`) + `parallel_strategy` flag in the
  generic cookbook trainer; CUDA-stream prefetch + async checkpoint in
  that generic trainer `[redacted: private repository]`. The domain codec trainer carries the
  `parallel_strategy` flag but its FSDP path is **fail-closed** (raises before
  `fully_shard`) `[redacted: codec research]`.
- **Slice B (activation)** — DAG schedule hints wired (`executor.rs:1393-1401`, used at
  `:1921`), HPO+KillOnNaN composition, P2P coordinator policy-routing.
- **Slice C (P2P data plane)** — `bundle.rs`, QUIC blob side-stream, `peer_exec` loop,
  `blut p2p` CLI. **Proven on loopback** (`peer_runs_real_stage_end_to_end`); engine
  @ `70b7f55`, 648 lib + 8 integration p2p tests green.

What forced this ADR: the remaining work splits cleanly by **blocker class**, and two
classes (hardware, infra) cannot be exercised here. Without a written, test-pinned
catalogue, the unblockable-now work and the resume-the-moment-hardware-lands work blur
together and stall. The repo also carries a real hazard the catalogue must encode: the
meta-repository's main branch carries concurrent unrelated work while all Slice A/B/C
meta-level commits live only on a feature branch that has **diverged** from main
`[redacted: private repository]`.

## Decision

Adopt the catalogue below as the committed sequencing toward the vision. Three rules
govern it:

1. **Pure-software items are done now**, each behind an OFF-by-default flag with a
   CPU/no-CUDA unit test, so default behaviour is byte-identical and the feature flips
   on under hardware later.
2. **FSDP-on-the-codec-trainer stays fail-closed.** The enabling change — consolidating
   ~10 rank-0-gated save sites into a collective full-state-dict gather — is
   **implemented and validated together** when a 2-GPU box exists, never shipped blind
   into the production codec trainer (a wrong collective-gather silently corrupts the
   DDP path too, and no CPU test catches it).
3. **Hardware- and infra-gated items ship turnkey now**: the runner/recipe/runbook that
   executes them is written, committed, and documented, so the gate is "attach the
   resource and run one command," not "go build the harness."

## The Catalogue

### Tier 1 — Pure-software (no external resource; do now)

| # | Item | Where | Closing test |
|---|------|-------|--------------|
| **T1.1** | CUDA-stream double-buffered prefetcher (flag OFF) | codec trainer + its data adapter `[redacted: codec research]` | unit: same batches/order vs sync path on CPU; CUDA-overlap path behind flag |
| **T1.2** | Async checkpoint save (background thread, DDP path) | codec trainer save sites `[redacted: codec research]` | unit: two rapid saves never overlap (2nd joins 1st); reloaded value-equal to sync save |
| **T1.3** | Decoupled metric sink (off-thread `_emit`) | codec trainer `[redacted: codec research]` | unit: sink writes off-thread; training step-count unaffected by sink latency |
| **T1.4** | generic-trainer async-eval metric sink | generic cookbook trainer `[redacted: private path]` | unit: metric emit off the loop thread, same values |
| **T1.5** | *(optional)* DCE↔hint regression test | engine `dag_opt.rs` / `executor.rs` | unit: after a DCE-dropping `optimize()`, every hint key ∈ plan node ids |
| **T1.6** ✅ | **P2P dispatch trigger** (`blut p2p serve --smoke-stage`) — coordinator-side CLI driving `dispatch_to_peer`; **makes T2.3 runnable by hand**. DONE, engine `c6e648c`: built-in `p2p-echo` stage (`src/p2p/smoke.rs`) + one-shot smoke-serve. | engine `cli.rs` + `src/p2p/smoke.rs` + dispatch allowlist | **PASSING** loopback test `smoke_serve_dispatches_echo_over_loopback` (coordinator binds, worker dials, output hash-verifies) |

> **B1 is NOT a bug.** `eliminate_dead_code` (`dag_opt.rs:108-199`) already renumbers
> `node.id` to a dense index (`old_to_new`, `:160-167`), and `optimize()` runs DCE
> **first** (`:68`) then builds every hint on the renumbered plan (`:74-84`). The hint
> map and its plan share one id-space — no desync window. T1.5 only *pins* this invariant.

### Tier 2 — Hardware-gated (turnkey; run the moment a box is attached)

| # | Item | Needs | Turnkey artifact | Gate command |
|---|------|-------|------------------|--------------|
| **T2.1** | FSDP↔DDP val-parity (validates the FSDP2 path + the codec trainer save restructure) | 2+ GPU box | an FSDP parity script `[redacted: private path]` (DDP vs FSDP at equal global batch) | parity within ε on the validation metric + a clean FSDP run; the 1-GPU checkpoint reload is the recipe's own load gate, run separately |
| **T2.2** | FSDP enablement in the codec trainer (lift the fail-closed) | 2+ GPU box | the save-site → collective-gather restructure, written+validated together | T2.1 green with `--parallel-strategy fsdp` |
| **T2.3** | C2 two-machine P2P live dispatch — **now fully turnkey** (T1.6 shipped the trigger) | two hosts (a workstation + a rented cloud GPU box), both `blut --features p2p` `[redacted: internal host]` | a two-machine runbook `[redacted: private path]` §A: `serve --smoke-stage p2p-echo` + `connect` | coordinator prints content-verified `HELLO P2P WORLD` across the two boxes |

> The P2P **protocol** is already proven on loopback QUIC (`peer_runs_real_stage_end_to_end`).
> T2.3 is two-machine confirmation + NAT/trust exercise, not new protocol.

### Tier 3 — Infra-gated (not started; the economic endgame)

| # | Slice | Blocker | First deliverable |
|---|-------|---------|-------------------|
| **T3.1** | Cloud queue v1: `CloudJob` / `LaunchTarget::Cloud` / priority-match / S3 + 1 provider | object store + provider account | `LaunchTarget::Cloud` type + 1 launcher, content-addressed submit→result |
| **T3.2** | Multi-provider routing, preemption→resume, billing | T3.1 done | provider-agnostic dispatch + spot-preemption durable resume |
| **T3.3** | Cluster (K8s operator / Ray autoscale) + distributed cache | a cluster | operator reconcile loop + shared content-addressed cache |
| **T3.4** | 100-DC: Raft-HA coordinator, geo-routing, federation | T3.3 done | HA coordinator quorum + geo-aware peer selection |

### Tier 4 — Distributed spine ([ADR 0079](0079-blut-symmetric-mesh.md) mesh + [ADR 0080](0080-blut-async-dp-training.md) DP training; the "real tool" Part 2)

Pure-software slices are validatable on localhost / kind CI; the last of each
track is hardware/infra-gated. Each row pins one runnable gate. (Concretizes
T3.3 as T4.5/T4.6.)

| # | Slice | Gate (one runnable test) | Status |
|---|-------|--------------------------|--------|
| **T4.1** | Data plane: chunk store + dedup + resume (transport v2) | kill a loopback blob transfer mid-stream → reconnect sends < blob size + hash verifies; resend → ~0 chunks | pure software |
| **T4.2** | Object-store `BlobStore` trait + tiered cache (Fs + S3) | MinIO round-trip (`scripts/minio_smoke.sh`, `#[ignore]`) | CI-able (MinIO) |
| **T4.3** | Mesh A1–A2: mutual TLS pinning + bi-stream `MeshFrame` v2 | correct-cert connects, forged-Hello/no-cert refused; v2↔v2 round-trip + v2↔v1 fallback echo | pure software |
| **T4.4** | Mesh A3–A6: `blut p2p node` + signed registry + gossip + mutual reputation | 3-node loopback any-to-any dispatch (worker-only C runs the echo dispatched by scheduler A) | pure software (2-machine run network-gated) |
| **T4.5** | K8s operator + CRDs (`crates/blut-operator`: `BlutPlan`/`BlutWorkerPool`) + `K8sDispatchSubmitter` + `LaunchTarget::K8s` | `scripts/k8s_kind_smoke.sh`: end-to-end `BlutPlan` on kind + MinIO | kind CI-able (GPU cluster infra-gated) |
| **T4.6** | Shared content-addressed cache across pods (PVC / S3) | two kind Jobs, second cache-hits the first (`cache_hit` in status.jsonl) | kind CI-able |
| **T4.7** | DP training C1–C2: `PrivacyLedger` + fail-closed dispatch gate + fed-round protocol | ledger exhaustion blocks; `Restricted`+gradients denied even with DP; loopback 3-node round applies + round-2 hash differs | pure software |
| **T4.8** | DP training C3–C4: Opacus DP-SGD fed stages + dynamic-DAG rounds + resume | `test_federated.py` ε within tolerance + noise-zero rejected; 2 dynamic rounds, kill after round 1, resume without re-running it | pure software (CPU) |
| **T4.9** | DP training C5 + cross-host observability (D5) | real multi-node DP round + ε-vs-utility curve; loopback status.jsonl carries worker-host-tagged events | hardware-gated (obs pure software) |

### Tier 0 — Governance / repo hygiene (do alongside, no code risk)

| # | Item | Action |
|---|------|--------|
| **G.1** | The meta-repository's pin of the engine is pre-C (`1a7d949`) | bump → `70b7f55` on the meta feature branch (never on main) |
| **G.2** | Meta feature branch ↔ main diverged | rebase/merge the feature branch onto main before it lands; keep concurrent unrelated pins untouched `[redacted: private repository]` |

## Rationale

- **Flag-OFF + CPU unit test** is the only honest way to do GPU-overlap work without a
  GPU: default behaviour is provably unchanged, and the unit test pins every property
  that does not require CUDA (batch identity, save non-overlap, off-thread emit). The
  CUDA-overlap path is exercised under T2.1.
- **Fail-closed FSDP-on-the-codec-trainer is correct, not a gap.** That trainer has ~10
  rank-0-gated save sites; an FSDP2 full-state-dict gather is a collective *every* rank
  must enter, so a rank-0-only save writes unreadable DTensors. The generic cookbook
  trainer (one gathered save site) already proves FSDP end-to-end; the codec trainer's
  restructure is genuinely entangled with multi-GPU semantics and must be validated as
  written (T2.2), not guessed.
- **Turnkey-first for gated work** maximizes "credibility per attached hour": when the
  box arrives, the cost is one command, not a build.
- **Tier ordering** = ROI under current resources: T1 unblockable now, T2 needs the boxes
  you can get cheaply (cloud GPU / two hosts), T3 is the payoff but infra-gated, so it
  follows once T1–T2 prove the spine.

## Alternatives Considered

- **Implement FSDP-on-the-codec-trainer now (blind).** Rejected: the collective-gather
  restructure touches the common DDP save path; a subtle error corrupts real codec
  training and no CPU test catches it. Deferring to T2.2 costs nothing — the flag is
  already fail-closed.
- **Skip the ADR, track in the plan file only.** Rejected: the plan file is ephemeral
  session scratch; the vision frontier is a durable, append-only record that survives
  context resets and names the gate for each item (provenance discipline
  `[redacted: non-engine product detail]`).
- **Switch meta to the feature branch and bump pins immediately.** Deferred to G.1/G.2 as explicit,
  user-gated steps: meta is on main with concurrent work and the repo's history
  includes a meta stash+reset that wiped uncommitted work once. Branch ops stay plain
  `checkout` (no `--recurse-submodules`, no stash), add only own paths.

## Consequences

- Commits us to keeping the async wins flag-gated and OFF until T2.1 validates them —
  no silent default change to the codec trainer.
- The fail-closed FSDP raise in the codec trainer becomes load-bearing
  documentation: it must stay until T2.2 lifts it with a passing T2.1.
- The turnkey artifacts (the FSDP parity script, the two-machine runbook) become contracts;
  if the trainer/CLI interface drifts, they must be updated in lock-step.

## Implementation Plan  *(the contract)*

- **Deliverable:** this catalogue, kept current; the Tier-1 code + tests; the Tier-2
  turnkey artifacts.
- **Acceptance gate (fail-CLOSED):**
  - `pytest` on the Tier-1 unit tests (prefetch / async-ckpt / metric-sink)
    + `cargo test -p blut` green + the meta-repository doc gates `[redacted: private path]`
  - `pass when:` all exit 0; default-OFF flags leave a no-flag run byte-identical.
- **Rollback:** each Tier-1 feature is a self-contained flag; revert the flag commit. The
  ADR itself is append-only (supersede with a new ADR, never delete).

## Progress Log  *(append-only; newest last)*

- _2026-08-27_ — **In progress — 2026-08-27** under ADR 0186 [not federated] clause 8. A catalogue whose own rules keep most of it open. Rule 2 states that FSDP on the codec trainer 'stays fail-closed' until a 2-GPU box exists, because the enabling collective-gather change cannot be validated on CPU and shipping it blind would silently corrupt the DDP path too -- that box does not exist, so the item is correctly unshipped rather than late. Its gate is red for a real reason as well: the pytest target it names no longer resolves, so the record points at a path that has moved. Both are open, so the catalogue is in progress.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. Its declared gate was executed on 2026-08-27 with verdict MISSING-TOOL. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

- _2026-06-30 — opened._ Ground-truthed via two read-only audits.
  - Engine (feature branch @ `70b7f55`): B1 hints used (`executor.rs:1393-1401`, `:1921`); DCE
    renumbers + runs before hint-build → **no desync bug** (T1.5 demoted to optional test).
  - Codec trainer: `parallel_strategy` flag present; FSDP **fail-closed**; prefetch /
    async-ckpt / decoupled-metric-sink all **absent** → T1.1–T1.3 are real work
    `[redacted: codec research]`.
  - The generic-trainer async wins are **meta-level** and live on the feature line; not in
    the main working tree → T1.4 needs meta on the feature branch `[redacted: private repository]`.
- _2026-06-30 — T1.6 DONE (engine `c6e648c`)._ Shipped `blut p2p serve --smoke-stage` +
  built-in `p2p-echo` stage (`src/p2p/smoke.rs`) + `DefaultDispatchPolicy` allowlist entry.
  Drives a raw `P2pServer` (not `Coordinator`) to avoid the `handle_peer` recv race; CLI
  roles (coordinator binds / worker dials) are stream-symmetric with the loopback test's
  roles. Proven by `smoke_serve_dispatches_echo_over_loopback` (output verifies `HELLO
  SMOKE` — the stage uppercases its input, so `--smoke-input` sets the expected string;
  the runbook uses "hello p2p world" → `HELLO P2P WORLD`); 70 p2p lib + 9 integration
  tests green; default + clippy clean. An automated review passed with nits
  `[redacted: private repository]` → applied a fast unknown-stage pre-check; skipped nits
  recorded in the commit. This makes **T2.3 fully turnkey** — the two-machine run is now
  `serve --smoke-stage` + `connect`, no remaining build.
- 2026-07-07 — **Tier 4 (distributed spine) added** — the "real tool" Part 2.
  Nine slices T4.1–T4.9 mapping to the mesh ([ADR 0079](0079-blut-symmetric-mesh.md)), k8s operator, and DP
  async training ([ADR 0080](0080-blut-async-dp-training.md)); T3.3 (cluster/operator + distributed cache) is now
  concretized as T4.5/T4.6. Each row pins one runnable gate. Follows the Part 1
  typed-dynamic-DAG facility ([ADR 0078](0078-blut-starlark-typed-dynamic-dag.md), frozen 2026-07-07) — T4.8's fed-round
  loop builds on its `map_output` fan-out.
- 2026-07-07 — **Tier 4 cores LANDED** (one session, ~26 commits, each
  tested + reviewed, pushed). **T4.2** BlobStore/FsBlobStore/S3BlobStore +
  CacheHandle remote tier. **T4.1** ChunkStore/ChunkIndex (dedup+resume core;
  transport-v2 wiring still open). **T4.3/T4.4 (mesh, ADR 0079)** all six
  slices A1–A6 + `PeerRegistry::apply_introduction`: mutual TLS pinning,
  bi-stream `MeshFrame` protocol, `MeshNode` (T4.4 3-node gate proven in a
  loopback test), signed introductions (ceiling Registered), gossip, mutual
  reputation. **T4.5 (operator, ADR 0037)** `crates/blut-operator`:
  BlutPlan/BlutWorkerPool CRDs (Restricted unrepresentable), reconcilers
  (Job/StatefulSet/Service builders + apply loop), Dockerfiles, kind smoke
  (`scripts/k8s_kind_smoke.sh`), `LaunchTarget::K8s` (kubectl subprocess, zero
  kube in engine). **T4.7 (DP, ADR 0080)** PrivacyLedger (signed tamper-evident
  chain) + fail-closed gradient gate + `finalize_round`. **T4.8** real Opacus
  DP-SGD in the cookbook `federated.py` (RDP-accounted ε). **D5** cross-host
  observability (`HostedEvent` host-tagged status.jsonl + `MeshFrame::Status`
  forwarding). OPEN: transport-v2 wiring (T4.1 tail), `blut p2p node` CLI +
  production `MeshTaskRunner`, C4 fed-round Starlark helper + `fed_stages.py`,
  and the hardware-gated tails (T4.6/T4.9).
- _2026-07-09 — **T3.1 cloud queue v1 LANDED on engine `main`** (`01a3ed7`)._
  The cloud-queue feature branch was rebased across the Tier-4 spine and fast-forwarded (branch
  deleted); its ADR was salvaged from a stale branch and
  renumbered **[ADR 0082](0082-blut-cloud-queue.md)** (0070 was already taken). `blut
  cloud smoke` + `LaunchTarget::Cloud` + lease-fenced `MemQueue` + cost ledger
  all trunk-resident behind the default-OFF `cloud` feature; 850 lib + 2
  integration tests green. T3.1's remaining tail is provider-side only (S3
  account + a live worker box) — software is done. Phase 0 of the
  premier-heavyweight roadmap (an internal plan `[redacted: private repository]`).

## Completion / Resolution  *(fill at freeze)*

- **verdict:** _pending_
- **gate_cmd:** _see Implementation Plan_

## Related Decisions

[ADR 0061](0061-blut-p2p-compute.md) (P2P compute), ADR 0065 [not federated], ADR 0066 [not federated],
[ADR 0044](0044-blut-api-standardization.md) (BLUT API standardization).

## Validation  *(ongoing)*

The catalogue is *wrong* if: a Tier-1 flag changes default behaviour (no-flag run diverges
byte-wise), the fail-closed FSDP raise is lifted without a passing T2.1, or a Tier-3 slice
starts before its infra blocker clears. Any of these triggers a corrective Progress Log
entry or a superseding ADR.
