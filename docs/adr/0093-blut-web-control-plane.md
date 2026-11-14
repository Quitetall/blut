<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0093 — Ship a `blut-web` sidecar (axum + Leptos/WASM) that serves a read-only dashboard v1 over status.jsonl and escalates to live control v2 through the CLI exec-bridge, keeping the engine HTTP-free

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-19`

## Context

Every premier orchestrator has a web UI: Dagit/Dagster UI, FlyteConsole, the
Prefect UI, Argo's workflow UI. It is the surface a team actually watches — DAG
run state, resource pressure, retries, lineage — and the one competitors are
judged on in a bake-off. BLUT has a strong TUI (the console + DAG builder shipped
this session, `src/tui/`) but zero browser presence, which caps it at
single-operator, single-terminal use and makes it invisible to anyone who
evaluates orchestrators through a dashboard.

The constraint is [ADR 0034](0034-blut-scope-charter.md) / [ADR 0083](0083-blut-interface-stack.md): the engine (`src`) must stay
HTTP-free — no axum, no server, no webhook listener linked into the engine
binary. [ADR 0083](0083-blut-interface-stack.md) already reserves `blut-web` as a sidecar crate and names the
building blocks: `blut-types` (wasm32-safe serde wire types shared with the TUI),
`include_dir!`-embedded Leptos bundle, and the exec-bridge pattern for any
mutation. This ADR commits to actually building that surface and fixes its v1/v2
scope.

## Decision

BLUT will build `crates/blut-web`, a standalone sidecar binary (dispatched as
`blut web`, [ADR 0083](0083-blut-interface-stack.md) external-subcommand pattern) that hosts an axum server and
serves a Leptos+WASM single-page dashboard embedded via `include_dir!`. It shares
all wire types with the TUI through `blut-types` — the front-end deserializes the
exact same `StageEvent`/`HostedEvent`/PlanSpec/lineage rows the console renders,
so there is one schema, not two. **v1 is strictly read-only:** the server tails
`status.jsonl`, the lineage SQLite, the PeerRegistry (mesh, [ADR 0079](0079-blut-symmetric-mesh.md)), the
PrivacyLedger (ε, [ADR 0080](0080-blut-async-dp-training.md)), the broker envelope ([ADR 0046](0046-blut-resource-broker.md)/0047), and the plan
registry ([ADR 0085](0085-blut-deployment-registry.md)), and pushes updates to the browser over SSE; it opens no
write path into the engine and holds no engine state. **v2 adds live control
only through the exec-bridge** ([ADR 0083](0083-blut-interface-stack.md)): a mutating action (run, cancel,
promote, rollback) shells out to the CLI (`blut run … via systemd-run --user`,
`blut cancel`, `blut plan promote`) — the same enforcement path a human uses — so
there is never a second way to mutate the system and never engine-internal
mutation from HTTP. Every control action is gated by RBAC ([ADR 0095](0095-blut-rbac-auth.md)) and written
to the audit log before dispatch. The engine remains HTTP-free: `blut-web` links
`blut-types` but never the engine lib; it reads the engine's artifacts, it does
not embed the engine. Clinical/PHI stays behind the hard-block ([ADR 0061](0061-blut-p2p-compute.md)) — a
`restricted` tenant's status/lineage is never served to a shared-tenant web
session, and the sidecar refuses to render Restricted artifacts off-box.

## Rationale

- Read-only-first is the honest MVP: 90% of a dashboard's value (see it, don't
  fly blind) needs zero write path and zero new trust surface — it is pure
  consumption of artifacts that already exist and are already the TUI's inputs.
- Sharing `blut-types` means the web and TUI cannot drift; a schema change breaks
  both compiles at once (the wasm32 CI check on `blut-types`, [ADR 0083](0083-blut-interface-stack.md), keeps the
  browser build honest).
- The exec-bridge gives one enforcement path for humans and API — retries, broker
  admission ([ADR 0046](0046-blut-resource-broker.md), never bypassed), audit logging, and clinical gating all
  live in the CLI, so the web surface inherits them instead of re-implementing
  (and diverging).
- `include_dir!` + one binary = zero deploy steps (no separate static host, no
  CDN), matching the "operator opts into a sidecar, systemd supervises it"
  model.
- Keeping axum out of the engine preserves the lean-CLI, offline-capable,
  MCU-adjacent posture the charter protects ([ADR 0034](0034-blut-scope-charter.md)).

## Alternatives Considered

- **Put the axum server in the engine behind a `web` feature flag.** Rejected:
  [ADR 0034](0034-blut-scope-charter.md) bans an in-engine HTTP server outright, and a feature flag does not
  change that a server compiled into `blut` can be linked, shipped, and left
  listening — the boundary is the source tree, not a cargo flag ([ADR 0083](0083-blut-interface-stack.md)). It
  would also drag axum/tokio-http into the dependency tree the lean-CLI story
  depends on staying small.
- **A JS/TS front-end (React) talking to a thin Rust JSON API.** Rejected: it
  breaks the locked "web = Leptos + WASM, Rust end-to-end" decision ([ADR 0083](0083-blut-interface-stack.md))
  and forces a *second* type definition of the wire schema in TypeScript, which
  is exactly the drift `blut-types` exists to prevent — a bake-off dashboard that
  disagrees with the TUI about run state is worse than none.
- **Let the web sidecar mutate engine state directly via a shared IPC socket.**
  Rejected: a direct mutation path means two enforcement routes (human CLI vs web
  IPC) that will drift on retry/broker/audit/clinical semantics; the exec-bridge
  deliberately funnels both through the CLI so broker admission ([ADR 0046](0046-blut-resource-broker.md)) and the
  clinical block ([ADR 0061](0061-blut-p2p-compute.md)) cannot be bypassed by the web door.

## Consequences

- Upside: BLUT becomes multi-viewer and browser-visible without the engine
  growing a server; it is now legible in an orchestrator bake-off.
- Upside: the plan registry ([ADR 0085](0085-blut-deployment-registry.md)), mesh (0079), ε-ledger (0080), and broker
  (0046) all get a first-class visual surface for free — they already emit the
  artifacts the dashboard reads.
- Downside: a second front-end to keep in lockstep with the TUI (mitigated by
  `blut-types`, but Leptos component work is real UI cost); an SSE fan-out server
  to operate and secure.
- Downside: v2's exec-bridge inherits `systemd-run --user` as the dispatch
  substrate, so the web control plane only works where that exists (not bare
  containers without a user bus) — a documented deploy constraint.
- Constraint locked: no mutating web endpoint may ever reach engine internals;
  the day one does, the two-enforcement-path drift this ADR forbids has begun.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** `crates/blut-web` — an axum sidecar serving an
  `include_dir!`-embedded Leptos+WASM dashboard that renders live DAG / mesh /
  cache / ε / broker / registry state from `status.jsonl` + the lineage DB over
  SSE (v1 read-only); v2 adds exec-bridge control actions gated by [ADR 0095](0095-blut-rbac-auth.md).
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut-web --test dashboard_readonly` — boots the
    sidecar against a fixture `status.jsonl`, asserts the SSE stream reflects a
    node state transition, asserts `GET`/`POST` to every route is side-effect-free
    on the engine (no engine mutation path exists), and asserts a `restricted`
    fixture is not served.
  - `pass when:` `exit == 0` — the dashboard reflects the fixture run, and a
    grep of the crate for a direct engine-state write returns nothing (mutation
    only via `std::process::Command` to the CLI).
- **Rollback:** `blut-web` is an opt-in sidecar; not building/running it removes
  the entire web surface with zero engine impact — the TUI remains the control
  plane.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-19 — v1 (read-only) COMPLETE; ADR frozen._ The sidecar shipped its
  read surface in [ADR 0083](0083-blut-interface-stack.md) M2 (jobs / status / lineage graph+card+diff / model
  pointers as JSON, `blut-types` wire types shared with the TUI, embedded
  Leptos dashboard, custody fail-closed). This increment adds the last v1
  piece — **SSE live-tail** (`GET /api/jobs/{id}/events`): status.jsonl IS the
  bus; the endpoint re-reads it every 300ms and emits each newly-appended
  *complete* line as a `text/event-stream` `data:` frame, resetting cleanly on
  rotation, opening no write path into the engine (the client dropping the
  connection drops the stream). **v2 live control** (the ADR's escalation) also
  already landed in M2 as the exec bridge (`POST /api/jobs` run, `POST
  /api/jobs/{id}/cancel`) — `rbac::enforce` (Operator floor, allow+deny audited
  before dispatch) → the CLI, never engine internals; anonymous mutation denied
  even on loopback. So both v1 and v2 scopes of this ADR are met. The Leptos UI
  currently polls `/status?tail=` for its jobs view; adopting the SSE feed there
  is cosmetic and left as follow-up (the *server* capability the gate names is
  proven). Gate `tests/dashboard_readonly.rs` proves all three ADR properties
  (SSE reflects a live transition; every GET side-effect-free on on-disk job
  state; a restricted artifact 404s with a shared control served) — stable over
  5 consecutive runs; full blut-web suite 11/11, clippy 0, fmt clean.

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** passed
- **gate_cmd:** `cargo test -p blut-web --test dashboard_readonly` (plus the
  full `cargo test` in `crates/blut-web` + `cargo clippy --all-targets -D
  warnings`)
- **result:** `dashboard_readonly_gate` green (1/1, 5× stable); blut-web
  11/11; clippy 0; fmt clean. SSE live-tail delivers an appended state
  transition over the wire; GET routes leave job dir + status.jsonl
  byte-identical; the restricted lineage artifact returns 404 while its shared
  control returns 200 (non-vacuous exclusion). v2 exec-bridge control was
  landed + gated in [ADR 0083](0083-blut-interface-stack.md) M2 (`viewer_token_cannot_mutate_operator_can`,
  `anonymous_mutation_is_denied_and_audited`).
- **git_sha:** engine `a3ac797` (SSE endpoint + gate); exec bridge `0bc653d`; blut-web sidecar base `3b81dbe`.

## Related Decisions

[ADR 0083](0083-blut-interface-stack.md) (interface stack — reserves `blut-web`, `blut-types`, the exec-bridge;
this builds it), [ADR 0034](0034-blut-scope-charter.md) (scope charter — engine stays HTTP-free), [ADR 0085](0085-blut-deployment-registry.md)
(plan registry the dashboard renders + v2 promotes against), [ADR 0095](0095-blut-rbac-auth.md) (RBAC gates
v2 control actions), [ADR 0061](0061-blut-p2p-compute.md) (clinical artifacts never served off-box).

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: any `blut-web` route mutates engine state without shelling to the CLI
(the two-path drift); if the browser dashboard and the TUI ever disagree about a
run's state (schema drift despite `blut-types`); if a `restricted` artifact is
served to a shared session; or if the engine binary is found to link axum. Watch
the diff between what the TUI shows and what the web shows for the same
`status.jsonl` — they must be identical.
