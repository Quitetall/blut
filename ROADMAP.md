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
be bit-identical across nodes. Still missing: resuming a long stage from a
mid-stage checkpoint (today a lost node restarts the stage from its start).

**2. P2P across independent hosts, including hostile networks.** Done for
dispatch. Stage data and results have crossed the internet between continents,
and a four-continent `p2p node` mesh answered 12 of 12 signed probes. Loss,
latency, partition, NAT and clock skew have run emulated, and loss and
partition have run on a real WAN; the real-WAN runs led to the opt-in BBR
controller. Peers find each other by gossip, and stage data moves over the
mesh without a shared cache: on four hosts on four continents, each seeded
with at most one peer, the mesh formed by gossip and 64 MB of stage data
reached a worker with an empty cache and came back verified. Still missing:
BBR measured against competing traffic before it can become the default.

**3. Network object storage.** Done for static credentials. The `s3` feature
round-trips through Cloudflare R2 over public TLS, which enforces the
create-only writes BLUT depends on. It also round-trips through MinIO over
HTTP and over verified TLS. Backblaze B2 lacks conditional writes and is
refused with an error that says so. Still missing: IAM-role / STS credentials
and a run against AWS S3 itself.

**4. A benchmark baseline that exists.** The regression gate
(`scripts/bench_regression_gate.py`) compares confidence bounds rather than
raw medians, and `scripts/run_benchmarks.sh compare` names the `release-0.2`
baseline. That baseline is not committed yet: it is recorded on a whole
dedicated machine at the frozen release SHA, because a shared host drifted
−12% to +8% on unchanged code.

**5. Windows containment.** Local stages are contained on Windows.
`LocalSystemd::spawn_contained` places a stage process in a Job Object: a
tree-wide commit cap, the peak at exit, and a tree kill on cancel, on a stage
timeout, or when the driver dies. The standard and LamQuant cookbook runners
spawn through `spawn_attached`, and a CPU training run through BLUT on
`windows-latest` is recorded in the
[validation ledger](docs/DISTRIBUTED_VALIDATION.md). Still open: `mem_high`
and `swap_max` have no Job Object equivalent; `blut cancel` cannot stop a
driver from another process on Windows; no GPU or DDP run on Windows has been
recorded.

**6. The ADR problem.** Federated and reviewed by the owner. The
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
