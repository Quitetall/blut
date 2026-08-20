<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Architecture decisions (ADRs)

BLUT's design is recorded as Architecture Decision Records. The engine-facing
ADRs — the scope charter, the interface stack, typed-declaration admission, the
eventing/SLA system, and the milestone constitution — live in the LamQuant
meta-repository under `docs/decisions/`, where they are lint-gated and composed
into a single index.

They are **linked, not vendored**, on purpose: an ADR is the meta-repo's source
of truth, and copying it here would create a second copy that silently drifts.

> **That meta-repository is not public.** If you are reading this from outside
> the project, the ADR numbers cited throughout this codebase currently have no
> document you can open. There are 766 such citations across 202 files, naming
> 53 distinct ADRs — mostly Rust source comments, but also crate manifests, CI
> workflows, scripts, `CHANGELOG.md`, and `RELEASING.md`. The summaries below
> are the engine-facing extract, and they are all there is until the ADR system
> is federated so that each repository carries its own decisions.
>
> This is a known gap, not an oversight: the rationale lives somewhere you
> cannot reach, so treat an "ADR NNNN" citation as a marker that a decision was
> recorded and argued, not as a reference you are expected to follow.

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

The complete generated index is the meta-repo's `docs/topics/adr-index.md` —
reachable only from inside the project, for the reason given above.
