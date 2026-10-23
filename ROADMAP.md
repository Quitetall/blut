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

**1. Distributed claims need runs, not command construction.** The Slurm
launcher has run against a live two-node cluster, a two-node Ray cluster has
run jobs to completion, and cookbook training has run across two machines over
NCCL with global-batch parity. A node lost mid-job is detected and the job is
requeued and completes once the node returns. `plan resume` after a node loss
picks up from the cache on the same launcher, and DDP replicas are checked to
be bit-identical across nodes. Still missing: an N-node P2P mesh (item 2).

**2. P2P across independent hosts, including hostile networks.** Done for
dispatch. Stage data and results have crossed the internet between continents,
and a four-continent `p2p node` mesh answered 12 of 12 signed probes. Loss,
latency, partition, NAT and clock skew have run emulated, and loss and
partition have run on a real WAN; the real-WAN runs led to the opt-in BBR
controller. Still missing: stage data over the N-node mesh (it needs a shared
cache) and gossip.

**3. Network object storage.** The `s3` feature round-trips against MinIO,
over plain HTTP and over TLS with certificate verification enforced. Backblaze
B2 cannot host it: it has no conditional writes, and BLUT refuses it with an
error that says so. Still missing: a run against AWS S3 or another provider with
conditional writes on public TLS, and IAM-role credentials.

**4. A benchmark baseline that exists.** `scripts/run_benchmarks.sh compare` is
in the release gate and cannot pass: only an `opt0` baseline is committed, and
the gate names `release-0.2`. Either the baseline gets established on the
designated quiet host, or the gate is honest about naming `opt0`.

**5. Windows containment.** The Job Object backend is implemented and
CI-gated on `windows-latest` (cap, peak, tree kill; see the ledger row). Still
missing: a cookbook runner calling `Containment::after_spawn` on Windows, the
`LocalSystemd` launcher (which still spawns bare there), and a recorded
training run on a Windows host.

**6. The ADR problem.** Federated, pending owner review before publication. The
engine-facing decision records (41 of them) now live in
[`docs/adr/`](docs/adr/README.md), redacted where marked, so the engine's
citations resolve. Citations of product decisions that were not federated were
rewritten as plain rationale; two citation numbers that predate a renumbering
now point at the records' current numbers. What is left: plain-text
"ADR NNNN" mentions are not hyperlinks (the index resolves them), and captured
data — an SDK test fixture exported from a cookbook, and recorded graph-runtime
evidence — still carries non-federated numbers verbatim.

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
