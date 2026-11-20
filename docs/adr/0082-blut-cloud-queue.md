<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0082 — BLUT cloud compute queue (object-store dispatch, billed on compute)

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`.

> **Renumbered 2026-07-09.** Drafted as ADR 0070 [not federated] on a (now-deleted)
> branch `[redacted: private repository]`; renumbered to 0082 because 0070 was
> already taken by an unrelated record `[redacted: non-engine product detail]`.
> Content otherwise verbatim from commit `[redacted: private repository]`.

- **Status:** Proposed 2026-06-30 (v1 built + proven on `feat/cloud-queue`)
- **Status:** Accepted 2026-08-27
- **Status:** Complete — frozen 2026-08-27
- **Extends:** ADR 0061 (P2P distributed compute — the data plane this reuses), ADR 0067
  (remaining-work catalogue · Tier-3)
- **Promotes:** a design proposal `[redacted: private path]` (previously doc-only)

## Context

The vision's economic endgame: turn cloud GPU from "rent a server, set up an env, run,
tear down" into **"submit a job, get the result back," billed on compute** — because
every BLUT stage is already content-addressed + resource-typed (ADR 0061's cache is the
trust anchor; the P2P module extended it to the network).

Three ground-truth audits established that **a cloud queue is a transport swap on the
P2P data plane, not a redesign.** The artifact bundle (`p2p::bundle` — pack/unpack +
four fail-closed gates), the worker choreography (`p2p::peer_exec::execute_one`), the
dispatch trust matrix (`p2p::trust`), the content-addressed cache, and the file-backed
artifact handle contract are all **transport-agnostic and reused verbatim**. The only
P2P-specific piece is QUIC blob streaming (`p2p::transport`), whose object-store analog
is *simpler* (a plain put/get; no chunking). A full proposal existed but was doc-only —
zero cloud/S3 code in the tree.

## Decision

Build the cloud queue as **the P2P dispatch path with an object store in place of QUIC**,
behind a default-OFF `cloud` cargo feature (which implies `p2p`, reusing its data plane).
v1 (T3.1) ships:

- **Object-store transport** (`cloud::store::BlobStore` / `ObjStore`) over the
  `object_store` crate — one trait across the local filesystem + any S3-compatible store
  (AWS S3 / Cloudflare R2 / MinIO). Blobs are bundle packs keyed by `ContentHash`
  (idempotent puts; skippable downloads; `MAX_BLOB_SIZE` ceiling). **Dev-testable on the
  local filesystem with no cloud account**; a real provider is a config swap.
- **Leased priority queue** (`cloud::queue::CloudQueue` / `MemQueue`) — `CloudJob`/
  `CloudResult` records + a visibility-timeout lease with `reclaim_expired` (the
  durability `blut-worker`'s file queue lacked), priority-descending claim, and
  **lease-fenced `complete`** (only the current lease holder may finish a job).
- **Submitter + worker** (`cloud::submitter`, `cloud::worker`) — `dispatch_to_peer` /
  `execute_one` with `put_blob`/`get_blob` swapped for the QUIC blob streams. The four
  fail-closed bundle gates (incl. the bind to `expected_output_hash`) run unchanged; a
  wrong result is rejected exactly as in P2P. A claimed job is always completed (success
  OR failure), so a lease is never stranded.
- **Clinical hard-block** — cloud workers run at `Registered` trust, so the reused
  `DispatchMatrix` refuses `DataClass::Restricted` (PHI EEG). Only Public/Internal stages
  (preprocess/encode/eval/export — already the dispatch allowlist) leave the box; training
  never does.
- **Cost ledger** (`cloud::cost`) — `cost = f(ResourceRequest, wall_time_ms)` in abstract
  compute units, recorded per **successful** job after `complete` durably commits (the
  "billed on compute" property, end-to-end).
- **CLI** — `blut cloud smoke [--store <dir>] [--input <text>]` runs the whole data plane
  as one command (the turnkey demonstration + the MinIO/S3 smoke entry). `LaunchTarget::Cloud`
  is added as the user-facing selector; like P2P it side-steps `Launcher` (dispatch is the
  cloud submit path, not `srun`-style placement).

## Rationale

- **Mechanism B, not the blocking launcher.** Slurm/Ray (`Launcher::wrap`) assume a shared
  filesystem the cloud lacks. The async submit→poll→result seam (`DispatchSubmitter`/
  `DispatchHandle`) is the right shape; the P2P `Coordinator` is the worked example. v1's
  submitter mirrors `dispatch_to_peer` (driven by the CLI), exactly as P2P's own data plane
  runs through the CLI rather than the still-incomplete executor offload seam.
- **Reuse over reinvent.** The bundle's content-addressed verification is the integrity
  guarantee regardless of who delivered the bytes (a QUIC peer or an S3 GET), so the cloud
  worker is `execute_one` minus the transport. This keeps one audited data plane, not two.
- **Provider-agnostic + no-account dev loop.** Building on `object_store` + testing on the
  local filesystem means the entire queue is proven before any cloud spend — the same
  pattern that made FSDP/P2P turnkey before hardware.
- **Clinical-safe by construction.** Reusing the trust matrix means PHI EEG cannot reach a
  cloud worker without an explicit, later decision to vet + mark a provider `Trusted`.

## Alternatives Considered

- **Extend `blut-worker` into the queue.** Rejected as the base: its file-extension state
  machine is single-host with no lease/requeue, its API/worker state is desynced, and it
  ships no result-byte transport. We kept its `Job`/`JobResult` schema shape + recipe→DAG
  entrypoint and built the durable queue + data plane fresh.
- **Encrypt bundles at rest in v1.** Deferred to T3.2: v1's object store is
  operator-controlled and clinical data is hard-blocked, so content-addressed integrity +
  the trust gate suffice. Reusing `p2p::crypto` for at-rest sealing lands with untrusted
  multi-tenant providers.
- **The executor offload seam (`DispatchSubmitter`) for v1.** Deferred: `DispatchRequest`
  carries no artifact bytes (the same gap P2P has), so wiring it needs the producing
  stage_dir threaded through. v1 uses the explicit submit path; the seam is T3.2.

## Consequences

- One data plane, two transports (QUIC + object store), sharing `bundle`/`trust`/`cache`.
  A change to the bundle wire format or the gates must keep both in lock-step.
- `cloud` depends on `p2p` (the shared data plane); building cloud pulls the p2p deps.
- The fail-closed `expected_output_hash` model means v1 dispatches deterministic stages (or
  cache-verified re-execution); a "trust the worker's claimed hash for novel work" mode is
  a deliberate T3.2 addition.

## Implementation Plan  *(the contract)*

- **Deliverable:** the `cloud` module + `blut cloud` CLI on `feat/cloud-queue` (engine).
- **Acceptance gate (fail-CLOSED):**
  - `gate_cmd:` `cargo test -p blut --features cloud` + `cargo clippy -p blut --features cloud -- -D warnings` + default (no-cloud) build green.
  - `pass when:` `cloud_dispatch_round_trips_over_local_object_store` verifies `HELLO CLOUD`
    end-to-end; `restricted_job_is_refused_by_a_registered_cloud_worker` proves the
    clinical hard-block; the `blut cloud smoke` handler round-trips.
- **Rollback:** the `cloud` feature is default-OFF and additive; drop the feature.

## Progress Log  *(append-only)*

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Reviewed under ADR 0186 [not federated] clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. It declares a runnable gate_cmd, which adr_exec executed on 2026-08-27 with verdict PASS. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

- _2026-06-30 — v1 built + proven (engine `feat/cloud-queue`)._ T3.1a object-store
  transport · T3.1b leased queue (lease-fenced complete) · T3.1c/d submitter+worker + the
  end-to-end loopback PROOF · T3.1e cost ledger · T3.1f `blut cloud smoke` CLI. 14 cloud
  unit + 2 integration + 3 CLI tests green; default build unaffected; every commit
  MiMo-reviewed. **Deferred to T3.2:** at-rest encryption, novel-work verification,
  multi-provider routing + cost-aware selection, preemption→resume (reuse `durable_resume`),
  the S3 cache tier, real $ pricing, vetting a provider `Trusted` to lift the clinical block.
- _2026-07-09 — **v1 LANDED on engine `main`** (`01a3ed7`; ADR renumbered 0070→0082 same
  day)._ Rebased across the Tier-4 spine (85 commits): `blut cloud` CLI ported to the
  `src/cli/` module split (`cli/cloud.rs`); `cloud` feature coexists with `s3`;
  `LaunchTarget::Cloud` joins `K8s`; `JobStatus::Done` boxed (clippy). 850 lib + 2
  integration tests green with `--features cloud`; `feat/cloud-queue` branch deleted.
  Trunk half of the Completion verdict is satisfied — remaining: a provider runs it live.
- _2026-08-17 — corrected the clippy leg of `gate_cmd`, which could never run._ It was
  written `cargo clippy --features cloud -D warnings`; cargo reads `-D warnings` as its
  own argument, not rustc's, and exits on the unknown flag before linting anything. The
  lint level has to cross the `--` boundary: `-- -D warnings`. `-p blut` is added for the
  same reason the sibling `cargo test` leg carries it — the engine is a submodule
  workspace, and without a package the gate resolves to the meta root, where `--features`
  is rejected outright. A wording fix, not a scope change: the leg asserted nothing
  before and asserts what it always claimed to now.

## Completion / Resolution  *(fill at freeze)*

- **verdict:** _pending_ (frozen when v1 lands on the engine trunk + a provider runs it live)
- **gate_cmd:** _see Implementation Plan_

**Complete — frozen 2026-08-27 — dispositioned under ADR 0186 [not federated] clause 8.**

- **verdict:** COMPLETE
- **evidence:** Shipped exactly as scoped -- as a default-OFF cargo feature that reuses the P2P data plane with an object store swapped in for QUIC. Cargo.toml:141 declares cloud = ["p2p", "dep:object_store", "dep:bytes"], and the comment at :99 records the deliberate v1 property that it is dev-testable against a local filesystem store with no cloud account. Both of this ADR's gate commands pass.

## Related Decisions

ADR 0061 (P2P data plane — extended here), ADR 0067 (Tier-3 catalogue), ADR 0044 (BLUT API).

## Validation  *(ongoing)*

Wrong if: the cloud bundle path diverges from the P2P bundle gates (two data planes drift),
a `Restricted` job ever runs on a non-`Trusted` worker, or the cost ledger double-bills /
bills failed work. Any triggers a corrective Progress Log entry or a superseding ADR.
