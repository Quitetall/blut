<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Architecture decisions (ADRs)

BLUT's design is recorded as Architecture Decision Records. They were written in
the project's decision log, which lives in a private meta-repository; the
engine-facing records are **federated** into this repository under
[`docs/adr/`](https://github.com/Quitetall/blut/tree/main/docs/adr), so an
"ADR NNNN" citation in the code, manifests, CI and changelog resolves to a
document you can open.

- **Index:** [`docs/adr/README.md`](https://github.com/Quitetall/blut/blob/main/docs/adr/README.md)
  lists every federated record with its number, title and status.
- **Redaction:** each record is the decision-log text with private material
  removed (machine paths, private repositories, domain research, hardware,
  business notes). Every removal is marked inline as `[redacted: <category>]`;
  the decision, its status, context and consequences are otherwise intact.
- **Drift:** the source log is frozen, so a federated copy cannot fall behind
  its original.
- **Not federated:** decisions about the LamQuant product rather than the engine
  (codec research, hardware, the ABIR graph program) stay private. Where a
  federated record mentions one, it reads `ADR NNNN [not federated]`. Two kinds
  of file still carry such numbers verbatim because they are captured data, not
  prose: the cookbook registry export used as an SDK test fixture, and recorded
  graph-runtime evidence with its receipt schema identifiers.

Two numbers moved when the decision log resolved collisions: code written
earlier cited the experiment-execution UX record as 0071 and the cookbook
taxonomy record as 0051. Those citations now read 0156 and 0055.

Load-bearing engine ADRs:

- **0034** — scope charter (the engine stays HTTP-server-free; delegated
  concerns live in sidecar crates).
- **0083** — the interface stack: `blut-types` keystone, external-subcommand
  dispatch, the `blut-tui` / `blut-web` / `blut-notify` sidecars, the exec
  bridge.
- **0093** — the read-only web dashboard (SSE over `status.jsonl`) + exec-bridge
  control.
- **0094** — eventing/triggers/SLA: `blut sensord`, webhook ingress, the
  `blut-notify` sink crate, and declarative SLA rules.
- **0133** — typed-declaration admission (`Stage::resource_envelope` is the
  admission contract).
- **0137** — the BLUT-engine M3–M6 milestone constitution and the
  engine-1.0-vs-family-1.0 boundary.
