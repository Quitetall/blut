<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Roadmap

What stands between `0.2.0-alpha.1` and a 1.0, stated as evidence that does not
exist yet rather than as features.

The rule this file follows: a capability is only listed as done when something
ran and was recorded. Everything else is named with the measurement it is
missing. Current per-capability status lives in
[`docs/DISTRIBUTED_VALIDATION.md`](docs/DISTRIBUTED_VALIDATION.md), which grades
each row **Validated / Component-tested / Deferred** — that ledger, not this
page, is the authority.

## Where it is now

The single-node path is exercised: typed DAG compilation, the content-addressed
cache, memory admission, containment selection on Linux, durable resume, the
PlanSpec IR, the Starlark front-end and the Python SDK. That is what
`0.2.0-alpha.1` means and it is why the version says `alpha`.

## Before 1.0

**1. Distributed claims need runs, not command construction.** Multi-node
Slurm/Ray launchers, `torchrun` rendezvous and the P2P mesh are
*component-tested*: the commands and parsers have tests, nothing has been
executed across machines. Closing this means recorded runs with topology,
versions, global-batch parity, failure recovery and artifact hashes — and until
those exist the ledger keeps saying so.

**2. P2P across independent hosts, including hostile networks.** Loopback proves
the protocol, not the mesh. Needs separate machines, NAT, loss and partition.

**3. Network object storage.** The cloud queue proves itself against the local
filesystem. S3/R2/GCS/MinIO were removed from the preview pending dependency
advisories, secret handling and TLS. Restoring them means code plus provider
integration tests, not a feature flag.

**4. A benchmark baseline that exists.** `scripts/run_benchmarks.sh compare` is
in the release gate and cannot pass: only an `opt0` baseline is committed, and
the gate names `release-0.2`. Either the baseline gets established on the
designated quiet host, or the gate is honest about naming `opt0`.

**5. Windows containment.** Bare fallback only; no Job Object implementation.
Either implemented or stated as unsupported rather than "deferred".

**6. The ADR problem.** 805 citations across 208 files name decision records
this repository does not contain — they live in a private meta-repository.
Either the engine-facing ADRs are federated into this repo, or the citations
stop pretending to be references. `book_src/adrs.md` currently discloses the gap
instead of hiding it.

## Explicitly not planned

These are charter decisions (ADR 0034), not gaps:

- **No HTTP server in the engine.** The dashboard is the `blut-web` sidecar.
- **No dynamic plugin loading.** Cookbooks are compiled in.
- **No domain code.** No stages, recipes or model knowledge ship here; that is
  what a cookbook is for.
- **No in-engine metrics UI or log aggregation.** Delegated.

`scripts/scope_guard.sh` enforces part of this and will reject a PR that drifts.

## How to influence it

Open a Discussion for shape, an issue for a concrete gap. The most useful
contribution is evidence: if you run BLUT across machines and record what
happened, that moves a row in the ledger, which is the only thing that moves
this page.
