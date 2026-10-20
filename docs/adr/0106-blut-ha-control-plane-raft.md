<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0106 — Make the scheduler capability highly available via Raft — leader election and replicated scheduler state that survives a coordinator crash without losing in-flight dispatch

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `in-progress` (supersedes: none)

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite earlier text. When the deliverable passes its gate, fill
> **Completion** and set Status to `Complete — frozen YYYY-MM-DD`.

## Status

Proposed 2026-07-10. Group 5 (distributed scale). **Extends [ADR 0079](0079-blut-symmetric-mesh.md) (symmetric
mesh) and [ADR 0082](0082-blut-cloud-queue.md) (cloud queue); does not supersede either.** ADR 0079 already
made "scheduler" a *capability* any node may enable and explicitly deferred HA to
"a separate, later concern ([ADR 0067](0067-remaining-work-toward-vision.md) T3.4)" — this is that concern. The mesh
transport, trust matrix, and clinical hard-block are unchanged; this ADR only
replicates the scheduler capability's *state* across a quorum of scheduler-capable
nodes.

`Accepted 2026-08-27`

`In progress — 2026-08-27`

## Context

ADR 0079 left the scheduler as a single logical decision-maker: even though every
node *can* run one, a given DAG is driven by one node's `Coordinator`/dispatch
matrix at a time. If that node crashes mid-run, the in-flight dispatch table
(which stages are RUNNING on which peers, which are QUEUED, the reputation EMAs)
is lost — surviving workers keep computing but no one collects their results, and
a restart re-dispatches already-running stages, double-billing the cloud queue
(0082) and violating content-addressed idempotency expectations. Temporal and
Airflow solve this with an HA scheduler backed by a replicated store; BLUT has
the mesh but no replicated *scheduler* state and no leader election, so there is
no principled failover. The charter forbids an in-engine HTTP server and any new
always-on network service must ride the existing mesh, so an external etcd/
ZooKeeper is off the table.

## Decision

Replicate the scheduler capability's state with an **embedded Raft group** running
over the mesh transport. Scheduler-capable nodes form a Raft cluster (odd-sized,
operator-declared members — not open gossip membership, so a random gossiped peer
cannot join the quorum); the **leader** owns dispatch, followers hot-stand-by. The
replicated state machine is exactly the scheduler's durable facts: the DAG's
per-stage status (`QUEUED/DISPATCHED/RUNNING/DONE/FAILED`), the stage→peer
assignment, the content-addressed result keys, and cloud-queue job handles (0082).
Every dispatch decision is a Raft log entry committed by quorum *before* the
`Task` frame goes out over the mesh, so a committed "stage X dispatched to peer P
with expected `blake3=…`" survives leader loss. On failover the new leader
**reconciles rather than re-dispatches**: it queries live peers for the status of
each `DISPATCHED/RUNNING` stage over the mesh, adopts any already-complete result
by its content hash (idempotent — the key is unchanged, per charter), and only
re-dispatches stages with no live owner and no committed result. Raft runs as an
**embedded library over mesh bi-streams** (charter: no HTTP, no separate service);
consensus messages are just another authenticated `MeshFrame` type. The clinical
hard-block is untouched — Raft replicates *scheduler metadata* (stage status,
content-hash keys, peer ids), never `Restricted` data or gradients, which never
enter the log.

## Rationale

- **Commit-before-dispatch is the whole trick.** If the dispatch decision is in the
  quorum-committed log before the task leaves, no failover can forget a running
  stage; reconcile-not-re-dispatch then makes recovery idempotent because result
  keys are content-addressed (unchanged, per charter).
- **Reuse the mesh; add no service.** Raft is a library, its RPCs are `MeshFrame`s
  over the already-mTLS bi-stream — no HTTP, no etcd, charter intact.
- **Closed membership for the quorum.** Gossip (0079 A5) is fine for *worker*
  discovery but a consensus quorum must not admit a gossiped stranger — operator-
  declared members keep Sybil nodes out of the vote that controls dispatch.
- **Metadata-only replication** keeps the clinical hard-block trivially true: the
  log holds status and hashes, never PHI or gradients.

## Alternatives Considered

- **External etcd / ZooKeeper for scheduler state.** Rejected — a separate
  always-on clustered service to secure and operate, reachable off-mesh, which the
  charter's "no new network service outside the mesh" spirit and the no-HTTP rule
  push back on. Embedded Raft over the mesh keeps one transport and one trust root.
- **Primary/backup with a shared object-store checkpoint (no consensus).** Rejected
  — periodic checkpointing loses every dispatch decision since the last snapshot,
  reintroducing the double-dispatch window this ADR exists to close; and split-brain
  (two nodes both think they're primary) has no principled resolution without a
  consensus term/leader.
- **Stateless re-run on failover (recompute the whole DAG).** Rejected — for a
  multi-hour training run that re-runs already-complete, cloud-billed stages;
  content-addressed keys make adoption cheap, so throwing that away is pure waste.

## Consequences

- Scheduler-capable nodes gain a Raft state machine and an operator-declared member
  list; single-node deployments run a trivial 1-member Raft (no behavior change,
  no quorum wait).
- Dispatch latency gains one quorum-commit round-trip per decision — bounded and
  worth it for exactly-once dispatch; the software gate measures it.
- The cloud queue (0082) job-handle lifecycle is now replicated, closing the
  double-bill-on-crash hole.
- A new failure mode (loss of quorum → no dispatch) is by design: fail-closed, the
  cluster refuses to dispatch rather than risk split-brain double-dispatch.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** an embedded Raft group over the mesh replicating scheduler state,
  with commit-before-dispatch and reconcile-not-re-dispatch failover.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - **Software (CI, non-HW — in-process multi-node harness):**
    - `gate_cmd:` `cargo test -p blut --features "p2p raft" ha_failover_no_double_dispatch`
    - `pass when:` `exit==0` — with 3 in-process scheduler nodes, killing the leader
      mid-DAG causes a follower to win election, adopt already-complete stages by
      content hash, re-dispatch **only** orphaned stages (assert each stage
      dispatched at most once across the whole run), and lose no committed result;
      and loss of quorum halts dispatch (fail-closed) rather than proceeding.
  - **HARDWARE-GATED (≥3 real nodes; multi-node bringup required):**
    - `gate_cmd:` a 3-node leader-kill chaos script `[redacted: private path]`, run against a live training DAG
    - `pass when:` on real hardware, a leader kill during a live DAG produces exactly
      one successful completion per stage with zero cloud double-bills (0082 ledger
      cross-check). Recorded in the tier-4 hardware validation ledger.
- **Rollback:** disable the `raft` feature; scheduler reverts to 0079's single
  driver (no HA, crash loses in-flight table as before). No data or checkpoint at
  risk; content-addressed keys make a manual re-run safe.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-08-27_ — **In progress — 2026-08-27** under ADR 0186 [not federated] clause 8. Half built. The embedded Raft group is real: src/raft/ carries consensus.rs, state.rs and ha.rs behind a declared raft cargo feature that implies p2p (lib.rs:69-71), which is the replicated state machine the Decision specifies. What is missing is the proof it survives failure -- the chaos harness its own gate invokes `[redacted: private path]` exists nowhere in the fleet. An HA control plane with no chaos test is exactly the shape of claim this corpus is not allowed to make.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Reviewed under ADR 0186 [not federated] clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. It declares a runnable gate_cmd, which the decision-log gate runner `[redacted: private path]` executed on 2026-08-27 with verdict MISSING-SCRIPT. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

- _2026-07-10 — opened._
- _2026-07-12 — M2.0 CONTROL REPAIR: planning contract only. The engine has no
  `raft` feature or named `ha_failover_no_double_dispatch` target, and
  the chaos script `[redacted: private path]` is absent. No quorum/failover measurement has run;
  the current scheduler remains single-driver as this ADR's rollback describes._
- _2026-08-25 — SOFTWARE HALF BUILT AND PASSING. The hardware half is
  untouched, so this ADR does NOT freeze._

  _`cargo test -p blut --features "p2p raft" ha_failover_no_double_dispatch`
  now exits 0 with 1 passed. The 2026-07-12 entry's three blockers are gone: the
  `raft` feature exists (default-OFF, implying `p2p`), the named target exists
  (`tests/ha_failover.rs`), and `src/raft/` implements election, replication,
  commit, commit-before-dispatch and reconcile-not-re-dispatch. Engine
  `ec0ad773`._

  _The core is DETERMINISTIC and tick-driven — no timers, no threads, no clock
  reads. Raft's randomised election timeout becomes a per-node constant the
  caller supplies, which is the same mechanism with the randomness lifted out
  where a test can pin it. Safety (at most one leader per term) does not depend
  on the randomisation; liveness becomes the caller's problem, and a real
  deployment jitters the value. A consensus gate that reproduced a bug one run
  in twenty would not be a gate._

  _TWO DESIGN CORRECTIONS the gate forced, recorded because neither is visible
  from the ADR text and both would be easy to undo:_

  _1. The guard against double dispatch also blocked the ONE legitimate
     re-dispatch. An orphan is still committed as `Dispatched`, so a new leader
     asking to re-dispatch it was told "already dispatched" and the stage
     stranded permanently — a worse failure than the one the guard prevents,
     and a quieter one. Resolved with an explicit replicated `VoidDispatch`:
     reconciliation records what it PROVED (that the assignment never took
     effect) and returns the stage to `Queued`; the re-dispatch then goes
     through the ordinary quorum-committed path. Dispatching directly from
     reconcile would have been the only dispatch in the system exempt from
     commit-before-dispatch._

  _2. A single `dispatch_count` is the wrong metric for this ADR's own
     assertion. Counting every committed decision makes "each stage dispatched
     at most once" unsatisfiable for exactly the recovery the ADR asks for. It
     is now `dispatch_count` and `voided_dispatches`, with
     `effective_dispatches()` their difference — both auditable, so a genuine
     double dispatch reads differently from a legitimate recovery. Decrementing
     a single counter would have made those two indistinguishable._

  _THE GATE WAS PROVEN ABLE TO FAIL, not merely observed passing. Planting:
  both dispatch guards disabled AND the `AlreadyDispatched` assertions
  neutralised so only the counter could catch it — it did (`train_b: 3`);
  `has_quorum` forced true — the fail-closed assertion fired. Both reverted, 0
  markers left. Worth recording that a FIRST, narrower plant did not fail: it
  disabled one of two guards and the other caught it, which is why the plants
  above disable both._

  _Also measured: full engine suite 1298 passed / 0 failed with `raft`, and
  1140 / 0 with it off — the ADR's rollback path (revert to 0079's single
  driver) is byte-untouched. clippy `-D warnings` clean for `raft` and for
  `cloud raft`; scope_guard PASS._

  _NOT DONE, and deliberately not claimed as progress toward the freeze:_
  _- the mesh `MeshFrame` variant carrying `RaftMessage` is not wired. The core
    is transport-agnostic on purpose, which is what lets the software gate run
    in CI with no network at all; ADR 0106's "Raft rides the mesh" is
    architecturally satisfied by that seam but not yet realised in the wire._
  _- no snapshotting and no log compaction. A long-lived cluster needs both._
  _- no joint-consensus membership change. ADR 0106 declares a fixed
    operator-supplied member set, so Raft's subtlest mechanism is out of scope
    by decision rather than by omission — but a later ADR that wants elastic
    membership must add it, not assume it._
  _- the HARDWARE-gated half (the 3-node leader-kill chaos script
    `[redacted: private path]`, ≥3 real nodes, 0082 double-bill ledger
    cross-check) is entirely untouched. That script does not exist._

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** —
- **gate_cmd:** —
- **result:** —
- **git_sha:** —  ·  **artifact sha256:** —

## Related Decisions

[ADR 0079](0079-blut-symmetric-mesh.md) (symmetric mesh — scheduler capability + deferred HA), [ADR 0082](0082-blut-cloud-queue.md) (cloud
queue — replicated job handles), [ADR 0034](0034-blut-scope-charter.md) (no-HTTP charter — Raft rides the mesh).

## Validation  *(ongoing — distinct from the completion gate above)*

Watch the dispatch double-fire counter and the cloud-queue double-bill ledger on
every real failover — either going non-zero means commit-before-dispatch or
reconcile has a hole and forces a fix. Watch quorum-commit latency as the member
count grows; if it dominates dispatch time the member set is too large for the
deployment and should be re-scoped.
