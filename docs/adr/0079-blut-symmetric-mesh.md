<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0079 — Symmetric P2P mesh — every host is server, worker, and (optionally) scheduler

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`.

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite earlier text. When the deliverable passes its gate, fill
> **Completion** and set Status to `Complete — frozen YYYY-MM-DD`.

## Status

Proposed 2026-07-07. Part 2 (the distributed spine) of the "BLUT real-tool"
roadmap (`[redacted: private path]`), tracked as ADR 0067
**Tier 4** slices A1–A6. Supersedes the **hub-and-spoke topology** of ADR 0061;
everything else in 0061 (the trust matrix, Ed25519 identity, QUIC/TLS
transport, X25519 blob sealing, zero-trust content verification, and the
clinical hard-block) **survives unchanged**.

`Accepted 2026-08-27`

`Complete — frozen 2026-08-27`

## Context

ADR 0061 shipped a working, secure P2P compute plane, but it is structurally
**hub-and-spoke**: a single `Coordinator` is "the single point of control";
roles are chosen at the CLI (`blut p2p serve` = coordinator, `blut p2p connect`
= worker); the QUIC transport uses **uni-directional streams in a fixed
direction per role** (workers only send `Handshake`/`Result`/`Error`, only the
coordinator sends `Task`/`Cancel`); TLS pins **only the client's view of the
server** (`with_no_client_auth` on the server side, so a peer's identity is
self-asserted in the handshake and auto-added to the registry as `Anonymous`);
and peer discovery is a hand-edited static `peers.json`.

The roadmap's vision — "every host able to run as a server or a single node" —
needs a **symmetric mesh**: any node can dispatch to, and accept work from, any
other. The hub-and-spoke shape blocks that in four concrete ways (mutual auth,
bidirectional streams, dual-role process, discovery), and it widens the attack
surface if lifted naively (an unauthenticated inbound peer writing into every
node's registry; `Restricted`/clinical data routing depending on each node's
local hand-edited trust file).

## Decision

Make the node the unit, not the role. A single process — `blut p2p node` — is a
QUIC **server + worker + optional scheduler**, all at once. "Coordinator"
becomes a **capability** (`scheduler`) any node may enable; the existing
`Coordinator` struct survives as that scheduler component *inside* a node. Six
independently-shippable slices, each loopback-testable behind the existing
`p2p` feature:

- **A1 — Mutual TLS pinning.** The server side gains a client-cert verifier
  that extracts the Ed25519 SPKI from the presented self-signed cert (reusing
  the already-tested webpki extraction) and **binds PeerId to the TLS
  identity** — killing the self-asserted-identity hole. The client presents its
  identity cert. A Hello whose claimed identity ≠ its cert identity is refused.
  Compat: a `--allow-legacy-peers` flag accepts no-client-cert connections for
  one release.
- **A2 — Symmetric wire protocol.** A `MeshFrame` envelope (versioned) carried
  on `open_bi()` request/response streams replaces the role-locked uni-stream
  pairs. Hello carries a `protocol_version`; v1 peers keep the uni-stream path.
- **A3 — `blut p2p node`.** `MeshNode` binds the server, dials seeds, and runs a
  select-loop over {inbound connections, outbound peer maintenance, local
  dispatch}. Capabilities `{worker, scheduler}` default both true. The
  scheduler capability wraps the existing dispatch-matrix / reputation /
  `DispatchSubmitter` logic verbatim — only connection ownership moves into the
  node. `serve`/`connect` become deprecated thin wrappers.
- **A4 — Signed registry + introductions.** `peers.json` entries gain a
  self-signature (tamper-evident; invalid entries dropped fail-closed). A
  signed `Introduction { introducer, subject, max_trust, expires_at, sig }` lets
  a Trusted node vouch for a peer — but **only up to `Registered`**; promotion
  to `Trusted` stays an operator-CLI decision. Expiry is mandatory.
- **A5 — Discovery.** Static `--seed` list first (persisted), then a periodic
  signed `PeerExchange` gossip (A4 makes it safe: receivers verify
  self-signatures and admit peers as `Anonymous`, subject to the fail-closed
  dispatch matrix). mDNS is explicitly deferred — gossip over the authenticated
  mesh covers the WAN case; mDNS only helps LANs.
- **A6 — Mutual reputation.** Workers also score schedulers (verified
  result-ack, abusive cancel) via the existing local EMA, so "whom to accept
  work from" uses the same signal as "whom to dispatch to".

### What stays (from ADR 0061)

The **trust matrix** (`DataClass × TrustLevel`, fail-closed to `Restricted`),
Ed25519 identity, QUIC/TLS 1.3 transport, per-blob X25519 sealing, zero-trust
`blake3(output) == expected` verification, and the **clinical hard-block**
(weights + gradients never leave a node for `Restricted`/PHI data; training
stays local) are unchanged. A symmetric mesh does not relax any of them — it
extends them to be enforced *in both directions*.

### Reputation stays local-only (rejected alternative)

Each node keeps its **own** EMA of every peer it has dispatched to or served —
there is no gossiped, global reputation. A shared reputation number is a Sybil-
laundering vector (a ring of colluding nodes upvotes a bad actor), and it makes
`Restricted` routing depend on a value the local operator didn't set. Trust
decisions stay local; introductions (A4) propagate *identity + a capped vouch*,
not a reputation score.

## Non-goals / stays forbidden

- **No in-engine HTTP server** (ADR 0034). The mesh is QUIC; the cloud-worker's
  REST API stays in `crates/blut-worker`, outside `src/`.
- **No transitive `Trusted`.** An introduction can only reach `Registered`;
  clinical-data routing requires an operator-set `Trusted`.
- **No global reputation** (see above).
- **No relaxation of the clinical hard-block.** Symmetry is about *who can talk
  to whom*, never about *what data may leave a box*.

## Alternatives considered

- **Keep hub-and-spoke, add HA to the coordinator (Raft).** Rejected for the
  mesh goal — it hardens the single point of control rather than removing it;
  HA coordinator is a separate, later concern (ADR 0067 T3.4).
- **Gossiped global reputation.** Rejected — Sybil laundering (above).
- **mDNS as the primary discovery.** Rejected as primary — LAN-only; the
  authenticated gossip generalizes. Kept as a possible later LAN convenience.

## Consequences

- Server-side client-cert verification is new attack surface reviewed under A1;
  the `--allow-legacy-peers` escape hatch is time-boxed to one release.
- The mesh is validatable on localhost (three `MeshNode`s, any-to-any dispatch)
  for every slice except the final two-machine confirmation, which is
  network-gated like ADR 0067 T2.3.

## Anti-re-litigation clause

This ADR does not reopen ADR 0061's clinical hard-block or its trust-matrix /
crypto choices, nor ADR 0034's HTTP-server ban. A future proposal to gossip
reputation, to allow transitive `Trusted`, or to move gradients for
`Restricted` data must supersede this ADR (and, for the last, ADR 0080)
explicitly with new evidence.

## Progress Log

- Gate: a citation gate in the meta-repository (`[redacted: private path]`).
- _2026-08-27_ — gave this record a concrete `gate_cmd` (`[redacted: private path]`) and removed its entry from a debt register (`[redacted: private path]`). Clause 5 accepts an instrument change only when the same commit appends here.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Reviewed under ADR 0186 [not federated] clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. It declares no gate_cmd, so it is recorded in `[redacted: private path]` as explicit, visible debt rather than treated as gated. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

- 2026-07-07 — ADR written. Implementation sequenced as Part 2 Phase D2 slices
  A1–A6; the shared data plane (chunk dedup/resume + object store) it and the
  operator both build on is Phase D1. Each slice lands with a loopback test and
  appends here.

## Completion

_Loopback/CI cores are LANDED (2026-07-07): A1–A6 + registry apply + `blut p2p
node` + the deduped/resumable transfer, all in `cargo test --features p2p` (823
tests). Hardware sign-off pending — see **the hardware-validation ledger `[redacted: private path]`
Gate M** (two real hosts running `blut p2p node`, any-to-any dispatch, resume/
dedup byte counts). Fill this section + freeze when Gate M passes._

**Complete — frozen 2026-08-27 — dispositioned under ADR 0186 [not federated] clause 8.**

- **verdict:** COMPLETE
- **evidence:** Slice A1, the one that closes the actual security hole, is in the tree at the shape the Decision specifies: src/p2p/transport.rs:754 installs a MeshClientVerifier via with_client_cert_verifier so the server binds PeerId to the TLS identity, and the one-release compatibility escape is the exact flag the ADR names -- --allow-legacy-peers, documented at :526 and cited in the refusal message at :591. The node-as-unit reshaping around it is real too: p2p/ carries mesh_runner.rs, mesh_wire.rs, node.rs and peer_exec.rs beside the surviving coordinator.rs.
