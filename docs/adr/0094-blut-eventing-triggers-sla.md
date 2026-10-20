<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0094 — Turn BLUT's sensor substrate into an event/data-driven trigger system — a file/spool/cron sensor daemon, sidecar webhook ingress, a `blut-notify` sink crate, and declarative SLA rules

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-19`

## Context

Airflow and Prefect sensors, plus their alerting integrations, let a pipeline be
*triggered by the world* (a file lands, an S3 prefix fills, a cron fires, a
webhook arrives) and let operators be *notified* when a run breaks or overruns.
This is table stakes for an orchestrator: a DAG that can only be launched by a
human at a terminal is a batch script, not a control plane.

BLUT has the seed but not the system. `src/sensor.rs` (the sensor substrate,
landed) can evaluate a `SensorOutcome`, but there is no daemon that runs sensors
on a schedule, no ingress for external events, no notification sink, and no
notion of an SLA (a run that should finish by T, or data that must be no older
than N). Everything is human-initiated.

**Numbering note:** the premier-heavyweight roadmap slotted this work as "0084",
but ADR 0084 [not federated] is already taken by an unrelated codec decision. This ADR therefore
takes **0094** and supersedes the roadmap's "0084" label for eventing/triggers —
there is no codec/eventing collision; the roadmap number was simply stale.

## Decision

BLUT will build an event/data-driven trigger system in four parts, each
respecting the charter. **(1) Sensor daemon** — `blut sensord`, a long-running
engine process (charter-safe: it links no server) that extends `src/sensor.rs`
with concrete sensor kinds: `file-drop` (a path/glob appears or changes),
`spool` (a watched directory drains/fills past a threshold), and `cron` (a
schedule fires). On a sensor firing, the daemon dispatches a run of a
registered plan (`registry://plan@…`, [ADR 0085](0085-blut-deployment-registry.md)) through the normal path — broker
admission ([ADR 0046](0046-blut-resource-broker.md)/0047) is never bypassed, so a triggered run queues behind
the same envelope as a human one. **(2) Webhook ingress** — lives *only* in the
`blut-web` sidecar ([ADR 0083](0083-blut-interface-stack.md)/0093): `POST /events/:trigger` validates a signed
payload and appends an event record the sensor daemon consumes via the same
artifact-tail mechanism the dashboard uses; the engine never listens on a socket
([ADR 0034](0034-blut-scope-charter.md)). **(3) `blut-notify` crate** — a sidecar that tails `status.jsonl`
and routes rule matches to sinks (Slack / Discord / ntfy / SMTP / `exec`); sink
credentials are `SecretRef`s ([ADR 0086](0086-blut-secrets-management.md)). **(4) SLA rules** — a declarative rule
file (`max_runtime`, `deadline`, data `freshness`) that both the daemon and
`blut sla check` (a pure lineage-DB read in the engine — charter-fine) evaluate,
writing violations to `sla.jsonl`; a breach is itself an event that can notify or
re-trigger. Clinical hard-block holds at the new ingress: a webhook or sensor
tagged `restricted` ([ADR 0061](0061-blut-p2p-compute.md)/0096) may only fire a same-tenant plan on the same
box, and `blut-notify` refuses to emit any Restricted payload content off-box —
an SLA breach on a clinical run notifies with a redacted, PHI-free summary only.

## Rationale

- Extending `src/sensor.rs` rather than inventing a scheduler reuses the
  `SensorOutcome` contract and the artifact-tail transport the console/web
  already trust — one event substrate, not two.
- Putting webhook ingress in the sidecar (not the engine) is the only
  charter-legal option ([ADR 0034](0034-blut-scope-charter.md) bans an engine listener); the sensor daemon
  consuming an appended event file keeps the engine a pure file-reader.
- Routing triggered runs through broker admission ([ADR 0046](0046-blut-resource-broker.md)) means a flood of
  file-drops or a webhook storm cannot stampede the box — the envelope throttles
  triggered and human runs identically.
- SLA-as-data (a rule file both the daemon and a CLI read) gives CI, the TUI, and
  the daemon the same answer for "is this run late?" — no logic fork.
- `blut-notify` as a separate crate keeps SMTP/HTTP clients out of the engine and
  lets sinks evolve without touching the control plane.

## Alternatives Considered

- **An in-engine cron/scheduler thread with a built-in HTTP webhook listener.**
  Rejected: the webhook listener is an in-engine HTTP server, banned outright by
  [ADR 0034](0034-blut-scope-charter.md); and an in-engine scheduler that dispatches around the broker would
  let a busy cron schedule (or a webhook storm) bypass admission ([ADR 0046](0046-blut-resource-broker.md)) and
  OOM the box — the exact failure the broker exists to prevent.
- **Poll external systems from inside each stage (no sensor layer).** Rejected:
  it scatters "wait for the world" logic across every pipeline, gives no central
  place to rate-limit or audit triggers, and means a clinical stage would open
  its own outbound connections — uncontrollable at the [ADR-0061](0061-blut-p2p-compute.md) custody boundary.
  A single sensor daemon + sidecar ingress is the one place to gate all inbound
  events.
- **Reuse an off-the-shelf notifier (Alertmanager / Apprise) as a hard
  dependency.** Rejected: it adds a network daemon to operate and cannot enforce
  the clinical redaction rule (PHI-free summaries only) — `blut-notify` must own
  the redaction pass ([ADR 0086](0086-blut-secrets-management.md)) so no Restricted content reaches a sink; a
  generic notifier can be an `exec` *sink target*, not the mechanism.

## Consequences

- Upside: BLUT gains sensors, webhooks, cron, alerting, and SLAs — the operator
  feature set that lets it stand next to Airflow/Prefect in an evaluation.
- Upside: triggered runs, SLA breaches, and webhooks are all just events feeding
  the same daemon → uniform audit + broker gating + notification.
- Downside: `blut sensord` is a new long-lived process to supervise (systemd),
  monitor for its own liveness, and reason about at-least-once vs exactly-once
  trigger semantics (a file-drop seen twice must not double-dispatch — dedupe on
  the event's content hash).
- Downside: webhook ingress adds an authenticated public surface on the sidecar
  (signature verification, replay protection) — real security work (leans on ADR
  0095).
- Constraint: SLA freshness rules depend on the lineage DB carrying accurate
  data-timestamps; a stage that lies about output freshness makes SLA checks
  wrong — freshness is only as honest as the provenance rows.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** `blut sensord` (file-drop/spool/cron sensors over
  `src/sensor.rs`), `POST /events/:trigger` in `blut-web`, the `blut-notify`
  sidecar crate (Slack/Discord/ntfy/SMTP/exec sinks), and an SLA rule file +
  `blut sla check` writing `sla.jsonl`.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut --test sensor_trigger && cargo test -p
    blut-notify --test sla_notify` — drops a file the sensor watches and asserts
    exactly one broker-admitted run of the registered plan is dispatched (a
    second identical drop does NOT re-dispatch), then forces a `max_runtime`
    breach and asserts an `sla.jsonl` row + one redacted (PHI-free) notification
    for a `restricted` run.
  - `pass when:` `exit == 0` — one dispatch per unique event (dedupe holds),
    every triggered run passed through broker admission, and the clinical
    notification contains no Restricted payload content.
- **Rollback:** the sensor daemon and sidecars are opt-in processes; stopping
  `blut sensord`, `blut-notify`, and the webhook route reverts BLUT to
  human-initiated runs with zero engine change.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._ Records the numbering correction: eventing/triggers
  takes ADR **0094**, not the roadmap's stale "0084" (already a codec ADR).
- _2026-07-19 — BUILT + FROZEN in three increments (engine `88f1e12` →
  `0ae8994` → `c30d584`; cookbook `[redacted: private repository]`)._
  1. **Trigger substrate (incr 1, `88f1e12`)** — `src/trigger.rs`: the
     `Trigger` trait + `FileDropTrigger` (file-drop/spool sensor, content-hash
     event id folding path+size+mtime), a persistent `SeenStore` (at-least-once
     → exactly-once dedupe, survives a daemon restart), and the load-bearing
     `AdmissionDispatcher` — a fired event launches a plan ONLY after
     `broker::admission::decide` admits ([ADR 0046](0046-blut-resource-broker.md)/0047 never bypassed); a
     refused event stays eligible (backpressure, not a silent drop). The
     `dispatch → mark-seen-only-on-Admit` rule is the dedupe correctness pin.
  2. **SLA + notify (incr 2, `0ae8994`)** — the keystone `blut_types::sla`
     (`SlaBreach`/`SlaKind`, shared producer↔consumer, wasm32-clean);
     `src/sla.rs` (`SlaRule` max_runtime/deadline/freshness, pure `evaluate`,
     `append_breaches` → `sla.jsonl`, PHI-free summaries by construction);
     `blut-notify` grew `breach_to_envelope`/`read_breaches`/`notify_breaches`
     (custody chokepoint — a restricted breach to an off-box sink is a
     fail-closed drop) and a real `ExecSink` (payload on stdin, never argv).
  3. **Runnable wiring (incr 3, `c30d584`)** — `blut sensord`
     (`--triggers`/`--once`/`--interval`/`--cli`; loads `triggers.toml`, drives
     each binding through the real admission-gated dispatcher, launches
     `<cli> recipe run <plan>` detached); `blut sla check` (a pure lineage read
     over the new `LineageDb::all_runs`, exits 1 iff a breach was written — a
     cron/CI alert gate); and the webhook ingress `POST /api/events/{trigger}`
     in `blut-web` (validates + content-addresses the body into the trigger
     spool the daemon watches — HTTP→filesystem keeps the engine a pure reader,
     [ADR 0034](0034-blut-scope-charter.md)). The cookbook pin bump (`[redacted: private repository]`) surfaces these as
     subcommands of the domain cookbook binary.
  Charter held throughout: no engine HTTP listener (ingress is the sidecar);
  triggered runs pass broker admission; the clinical hard-block holds at the
  new ingress (a restricted breach never notifies off-box). Live smoke on the
  box: `sla check` clean; `sensord --once` (via the cookbook binary) correctly REFUSED a dropped
  event under real memory pressure (7G free < 2G+6G) — the envelope enforced on
  live state — with the event staying eligible to retry.
- _2026-07-20 — COMPLETION EVIDENCE CORRECTED (engine `df6d312`)._ The
  2026-07-19 freeze entry and completion block recorded the initial substrate
  as the whole frozen contract too early. The named commits did not yet provide
  spool-threshold or cron triggers, signed/replay-bounded webhook
  authentication, an exact child-admission acknowledgement, shared daemon/CLI
  freshness evaluation, declarative notification rules, or runnable
  Slack/Discord/ntfy/SMTP tailing. Commit `df6d312` closes those tails without
  changing the decision: shared validated `triggers.toml`; file/spool/cron
  sources; content-stable dedupe plus durable `.admitted` recovery; launch only
  after the real recipe holds its exact tenant reservation and scheduler lock;
  HMAC-SHA256 webhook ingress with timestamp window, `SecretRef`, custody, and
  content-addressed replay handling; measured fail-closed SLA freshness used by
  both `sensord` and `blut sla check`; and a locked, atomic-cursor notifier with
  Slack/Discord/ntfy/SMTP/exec sinks, redaction, rotation recovery, and
  at-least-once retry. The original gates now pass against the completed
  implementation (`sensor_trigger`, `sla_notify`, and
  `dashboard_readonly`), as do the full locked workspace, standalone web,
  notifier, clippy, rustdoc, wasm, and custody/scope gates. This dated entry is
  the authoritative completion evidence where the earlier result conflicts.

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** passed
- **gate_cmd:** `cargo test -p blut --test sensor_trigger && cargo test -p
  blut-notify --test sla_notify` (plus `cargo test -p blut-web --test
  dashboard_readonly` for the webhook ingress; full engine/notify/web suites +
  clippy)
- **result:** `sensor_trigger` green — one broker-admitted dispatch per unique
  event, identical drop does NOT re-dispatch, dedupe survives a restart, an
  oversubscribed box REFUSES (envelope not bypassed) and the refused event
  retries when capacity returns. `sla_notify` green — a max_runtime breach on a
  RESTRICTED run writes an `sla.jsonl` row (shared keystone format) and yields
  EXACTLY ONE redacted (PHI-free) notification to a local sink while an off-box
  sink is refused (0 delivered, never handed the payload). Webhook ingress
  spools a posted event (202) and refuses a non-object body (400). Engine
  18/18 suites, blut-notify 5 lib + 1 gate, blut-web 11, blut-types 17 +
  wasm32; clippy 0 across all; fmt clean.
- **git_sha:** engine `c30d584` (incr 1 `88f1e12`, incr 2 `0ae8994`, incr 3
  `c30d584`) · backends `[redacted: private repository]` · cookbook `[redacted: private repository]`.

## Related Decisions

[ADR 0083](0083-blut-interface-stack.md) (sidecars host webhook ingress + `blut-notify`; engine stays
HTTP-free), [ADR 0101](0101-blut-backfills-partitions.md) (`src/sensor.rs` substrate — this extends it, does not
supersede), [ADR 0085](0085-blut-deployment-registry.md) (triggers dispatch registered plans), [ADR 0046](0046-blut-resource-broker.md)/0047 (broker
admission never bypassed by a trigger), [ADR 0086](0086-blut-secrets-management.md) (sink creds are `SecretRef`s),
[ADR 0061](0061-blut-p2p-compute.md)/0096 (clinical triggers stay same-tenant, notifications PHI-redacted).

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: a triggered run ever bypasses broker admission; a duplicated event
double-dispatches (dedupe broke); a webhook listener appears inside the engine
binary; a clinical SLA-breach notification carries PHI off-box; or SLA freshness
checks fire on stale provenance timestamps. Watch the ratio of triggered→admitted
runs (must be 1:1 through the broker) and audit notify payloads for redaction
leaks.
