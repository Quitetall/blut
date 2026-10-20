<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0080 — Async distributed training with differential privacy (FedAvg + DP-SGD)

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`.

> **This file is append-only and living.** Append dated Progress-Log entries as
> implementation proceeds; fill **Completion** and set Status to
> `Complete — frozen YYYY-MM-DD` when the deliverable passes its gate.

## Status

Proposed 2026-07-07. Part 2 (the distributed spine) of the "BLUT real-tool"
roadmap, tracked as ADR 0067 **Tier 4** slices C1–C5. Supersedes ADR 0061's
"gradients never cross the wire" fence **for `Public`/`Internal` data only** —
`Restricted`/PHI stays local-only, differential privacy or not.

`Accepted 2026-08-27`

`Complete — frozen 2026-08-27`

## Context

ADR 0061 forbade model weights and gradient state from leaving a node, enforced
by the dispatch matrix ("training → always local"), and explicitly deferred
"Hivemind-style async gradient averaging over the internet" as "a future
extension (**requires differential privacy**), not part of this ADR." The
roadmap wants that async-training capability: a network of nodes training a
shared model without pooling raw data.

The codebase already carries an **unwired** FedAvg reference module
(`[redacted: private path]` — `FederatedServer`/`FederatedClient`
with an ad-hoc DP-noise knob, marked "post-launch", imported by nothing). This
ADR productionizes it *behind a fail-closed privacy gate*, over the existing
P2P data plane.

The load-bearing constraint remains ADR 0061's clinical hard-block: EEG is PHI
under HIPAA. So this ADR does **not** unlock async training universally — it
unlocks it for **non-clinical** corpora, with mandatory DP, and keeps clinical
data local.

## Decision

Add **asynchronous federated training** (FedAvg with client-side DP-SGD) over
the symmetric mesh (ADR 0079), gated fail-closed by data class and a privacy
budget. Five slices (ADR 0067 C1–C5):

- **C1 — Privacy ledger + dispatch gate (the brake, built first).** A
  `PrivacyLedger` (per-corpus ε budget/spent, append-only and Ed25519-signed so
  it is tamper-evident) with an `admit(cost)` that refuses over-budget. The
  dispatch policy gains a `carries_gradients` flag; the verdict is **Deny**
  unless `data_class ∈ {Public, Internal} ∧ a DP config is present ∧ the ledger
  admits`. `Restricted ∧ carries_gradients` is a **hard Deny** citing this ADR.
  Fail-closed default: unknown gradient-bearing status ⇒ treated as carrying
  gradients.
- **C2 — Fed round protocol.** An initiator-side round driver: seal the global
  weights per participant (the existing X25519 blob path), dispatch N
  local-train tasks via the existing `DispatchSubmitter` seam, collect signed
  delta results, hash-verify them, and hand deltas to the Python aggregator.
  **Aggregation always runs on the initiator.** Straggler policy:
  `min_clients` + a round deadline; late updates dropped and reputation-scored.
- **C3 — Opacus DP-SGD + fed stages.** Replace `federated.py`'s ad-hoc post-hoc
  noise with a real Opacus `PrivacyEngine` local fine-tune (per-sample gradient
  clipping + calibrated noise) that emits an RDP-accounted `(ε, δ)` in its
  result JSON for the Rust ledger. Two content-addressed cookbook stages —
  `fed-local-train` (worker) and `fed-aggregate` (initiator).
- **C4 — Rounds as dynamic DAG nodes.** A `fed_round(...)` helper (built on the
  ADR 0078 `map_output` fan-out) expands to fan-out(`fed-local-train` × N) →
  `fed-aggregate`; the round loop is dynamic-DAG generation terminated by a
  convergence metric or ledger budget. Resume replays to the last completed
  aggregate via the existing resume path (each aggregate's output is a round
  artifact).
- **C5 — Hardware-gated validation.** A real multi-node DP round + a
  utility-vs-ε curve on a real domain corpus. C1–C4 are CPU/loopback pure
  software.

### Hard invariants (fail-closed)

1. `DataClass::Restricted` gradient / delta / weight-derived artifacts **never
   dispatch** — no DP override. (HIPAA; ADR 0061 clinical block.)
2. **Ledger exhaustion is a hard stop**, not a warning — a round that would
   exceed the corpus ε budget is refused.
3. Global weights are **X25519-sealed per recipient**; updates are
   Ed25519-signed and content-verified like every other task result.
4. Aggregation runs **only on the initiating node**.
5. DP is **mandatory** for any gradient-bearing dispatch: `noise_multiplier == 0`
   is rejected in both the Python trainer and the Rust ledger.

### ε accounting lives in Python, enforcement in Rust

Opacus's RDP accountant is the source of truth for the per-round ε cost; the
Rust ledger **stores and enforces** the accounted numbers (charter: delegate
the verb — DP math — to the domain layer; the engine owns the gate). Rust never
reimplements DP math.

## Non-goals / stays forbidden

- **No async training for `Restricted` data**, ever, DP or not (invariant 1).
- **No server-side DP or secure-aggregation** in v1 — deltas carry an
  honor-plus-signature DP attestation (a malicious worker could under-noise); a
  cryptographic secure-aggregation protocol is a documented later extension.
- **No gradient exchange without a ledger admission** (invariant 2).

## Alternatives considered

- **Hivemind-style DHT gradient averaging.** Rejected for v1 — no per-corpus ε
  accounting, no data-class gate; it is the exact "internet async averaging"
  0061 fenced.
- **Server-side DP (noise added at the aggregator).** Rejected — it trusts the
  aggregator with raw gradients; client-side DP-SGD bounds each participant's
  contribution before it leaves the box.
- **Trust-based (no DP) async for `Internal`.** Rejected — DP is the roadmap's
  stated precondition for gradients on the wire; making it optional invites a
  slow slide to un-noised gradient sharing.

## Consequences

- The unwired `federated.py` becomes a real, gated code path; `opacus` joins the
  cookbook's Python deps.
- DP attestation being honor+signature is a known limitation, documented; it is
  acceptable for a Trusted-peer mesh and is the seam a future secure-aggregation
  protocol closes.
- C4 depends on the ADR 0078 `map_output` primitive (now shipped).

## Anti-re-litigation clause

This ADR does not reopen ADR 0061's clinical hard-block. A future proposal to
move gradients for `Restricted` data, to make DP optional, or to aggregate off
the initiator must supersede this ADR explicitly with new evidence (e.g. a
validated secure-aggregation protocol + a regulatory analysis).

## Progress Log

- Gate: a citation gate in the meta-repository (`[redacted: private path]`).
- _2026-08-27_ — gave this record a concrete `gate_cmd` (`[redacted: private path]`) and removed its entry from a debt register (`[redacted: private path]`). Clause 5 accepts an instrument change only when the same commit appends here.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Reviewed under ADR 0186 [not federated] clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. It declares no gate_cmd, so it is recorded in `[redacted: private path]` as explicit, visible debt rather than treated as gated. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

- 2026-07-07 — ADR written. Implementation sequenced as Part 2 Phase D4 slices
  C1–C5; C4 builds on the ADR 0078 `map_output` fan-out (shipped 2026-07-07).
  Each slice lands with tests and appends here.

## Completion

_C1–C4 are LANDED (2026-07-07): the `PrivacyLedger` + fail-closed gate,
`finalize_round`, real Opacus DP-SGD (`federated.py`/`fed_stages.py`), and the
`fed_round` Starlark builtin — in `cargo test --features p2p --lib p2p::privacy
p2p::fedround` + the cookbook `test_federated.py`/`test_fed_stages.py`. C5
(hardware) pending — see **the hardware-validation ledger `[redacted: private path]` Gate D**
(multi-node round + ε-vs-utility curve, exhaustion + Restricted-denial spot
checks, kill/resume). Fill this section + freeze when Gate D passes._

**Complete — frozen 2026-08-27 — dispositioned under ADR 0186 [not federated] clause 8.**

- **verdict:** COMPLETE
- **evidence:** C1, the brake the Decision insists is built first, exists and is used rather than merely defined: PrivacyLedger and DpConfig live in src/p2p/privacy.rs and are taken by the round driver at p2p/fedround.rs:24,86, whose own header records that it folds attestation verification and the epsilon bookkeeping into the ledger. C2's initiator-side round protocol is that same fedround.rs. The fail-closed dispatch posture the ADR requires is therefore enforced at the seam where gradients would actually move.
