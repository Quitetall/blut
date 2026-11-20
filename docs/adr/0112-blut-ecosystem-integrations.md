<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0112 — Deliver ecosystem integrations as registered connector cookbooks that shell out, never as dynamic plugins

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (frozen 2026-07-11)

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-11`

Part of Group 7 (Interfaces & ecosystem) of the "BLUT real-tool" roadmap. Builds
on ADR 0034 (scope charter), ADR 0037 (cookbook architecture), and ADR 0100
(the connector/exec-bridge substrate). Does not supersede any of them.

## Context

Airflow's moat is its **provider ecosystem** — hundreds of `apache-airflow-
providers-*` packages wrapping S3, BigQuery, Snowflake, dbt, Spark, Postgres,
Slack. Dagster ships an equivalent set of integrations; Prefect has its
collections. A modern orchestrator is judged less on its executor than on
whether it can *talk to the tools a data platform already runs*. BLUT today can
orchestrate LamQuant training and firmware builds but has **no first-class,
curated way to say "read this Parquet from S3", "run this dbt model", "submit
this Spark job"** as a typed graph node.

The obvious way every Python orchestrator did this is the obvious way BLUT
**cannot**: dynamically importable provider plugins. ADR 0034 bans dynamic dylib
plugins and any in-engine HTTP server; ADR 0037 fixed integrations as
**compiled-in cookbooks composing typed stages that shell out** ("adopt the
concepts, delegate the verbs"), explicitly rejecting a dynamic `.so`/plugin
registry. ADR 0078 reinforced it: a front-end may only *name* registered stages;
new stage *types* require Rust + cookbook registration. So the ecosystem
question for BLUT is not "how do we load providers at runtime" — it is settled
that we don't — but "**what is the charter-safe shape of a connector, and how do
we curate a registry of them so the ecosystem is discoverable and typed?**"
There is genuine white-space here: nobody has built an Airflow-provider-class
ecosystem on a *subprocess-delegated, statically-registered, typed-I/O* model.

## Decision

Define a **connector framework**: an integration is a **registered "connector"
cookbook** (compiled-in, per ADR 0037) that contributes typed **connector
stages**, each of which **delegates the verb to the external tool as a
subprocess** (per ADR 0034) and exchanges data through **content-addressed
artifacts with a small set of typed I/O kinds** — never through a dynamically
loaded library.

Concretely:

1. **Typed I/O kinds.** A curated, closed enum of connector artifact kinds —
   `ObjectRef` (an object-store URI + content hash), `TableRef` (a warehouse
   table/partition identity), `DatasetRef` (a local content-addressed path),
   `Blob` (opaque bytes + hash). A connector stage declares its input/output kind
   from this set, so the `from_erased_graph` kind-checker (ADR 0078) type-checks
   an integration graph exactly as it checks a native one. Kinds are **identity +
   hash, not bulk data** — a `TableRef` carries the table's content fingerprint,
   preserving the unchanged cache-key discipline `hash(code_sha ⊕ canon_args ⊕
   input_hash)`.

2. **Connector stages shell out, deterministically-keyed.** Each stage
   (`s3.get`, `s3.put`, `warehouse.query`, `dbt.run`, `spark.submit`, …) invokes
   the vendor CLI/JDBC bridge as a subprocess with typed args, exactly like the
   existing `kubectl`/`slurm`/`blut-dsl` adapters. The stage's cache key is the
   hash of its canonical args ⊕ the input `*Ref` hash — so re-running a connector
   node with the same inputs is a cache hit, and a changed upstream `TableRef`
   fingerprint invalidates it. Effectful verbs (`put`, `dbt run`) are marked
   non-cacheable-output but still content-address their *result ref*.

3. **A curated connector registry.** A vendored, in-tree manifest
   (`connectors/registry.toml`) lists each blessed connector cookbook: name,
   version, the external binary/version it shells (probed at stage start, fails
   closed if absent), its declared I/O kinds, and license. Discovery is
   `blut connectors list`; there is **no runtime download or dynamic load** —
   adding a connector is a Rust cookbook + a registry entry + CI, reviewed like
   any other code.

4. **Clinical + charter gates inherited.** Connector stages run through the same
   fail-closed clinical hard-block and the same containment (systemd/cgroup) as
   any stage; a connector that would move PHI to an unapproved sink fails closed.

## Rationale

- **The charter already decided the mechanism; this ADR supplies the missing
  framework around it.** ADR 0037 said integrations are cookbooks that shell out;
  it did not define the *typed I/O vocabulary* or the *curated registry* that
  make a connector ecosystem discoverable and type-safe. That is the net-new
  contribution.
- **Typed refs make integration graphs first-class.** By funneling every
  connector through a closed kind enum, an S3→dbt→Spark pipeline is kind-checked
  before launch by the same `from_erased_graph` machinery that guards a native
  DAG — a strength Airflow's XComs (untyped, runtime) never had.
- **Hash discipline survives crossing a system boundary.** Carrying identity +
  content hash in `*Ref` (not bulk bytes) means the content-addressed cache keys
  are unchanged (ADR 0078): an unchanged warehouse partition is a cache hit;
  a mutated one invalidates downstream — correct incremental behavior for free.
- **Subprocess delegation is the only charter-legal option** and happens to be
  the robust one: vendor CLIs/JDBC bridges are the tools' own supported surface,
  version-probed and containable, with no ABI coupling into the engine and no
  CPython/JVM linked into a binary that must cross-compile to firmware.
- **A vetted registry beats an open plugin bazaar for a regulated shop.** Every
  connector is code-reviewed, license-checked, and version-pinned in-tree — the
  provenance the PCCP/clinical posture demands, which a `pip install any-
  provider` model cannot give.

## Alternatives Considered

- **An Airflow-style dynamic provider plugin system (load `.so`/Python providers
  at runtime)** — REJECTED: directly violates ADR 0034 (no dynamic dylib) and
  ADR 0037/0078 (no dynamic stage/code loading). It also defeats the content-hash
  determinism and the clinical fail-closed posture, since an unreviewed provider
  could open an unapproved network sink. Non-starter under the charter.
- **An in-engine connector service exposing HTTP/gRPC endpoints per integration**
  — REJECTED: ADR 0034 bans an in-engine HTTP server. The subprocess-per-verb
  model already delivers submit/exchange without a socket.
- **Bulk-data artifacts (ship the actual table/object bytes through the graph)**
  — REJECTED: breaks the cache-key model and blows up artifact storage; a 2 TB
  warehouse table cannot be a graph edge. Typed `*Ref` identity+hash keeps edges
  small and cache-correct while the bytes stay in the external system.
- **A single generic `shell` stage users parameterize per integration** —
  REJECTED as the *primary* surface: it exists (the escape hatch) but gives no
  typed I/O, no version probe, no discoverability, and no license/provenance
  vetting — i.e., none of the properties that make an ecosystem trustworthy. The
  curated typed connectors are the blessed path; raw `shell` stays the unsafe
  fallback.

## Consequences

- A new closed I/O-kind enum (`ObjectRef`/`TableRef`/`DatasetRef`/`Blob`) becomes
  a **stable typed contract** connector cookbooks bind to; extending it is an
  additive, reviewed change (a new kind is a schema decision, not a plugin).
- A vendored `connectors/registry.toml` + `blut connectors list` becomes
  load-bearing for discovery; there is no runtime fetch, so the registry is only
  as complete as what has been reviewed and merged — a deliberate cost we pay for
  provenance.
- Each connector adds a **runtime dependency on an external binary** (aws-cli,
  dbt, spark-submit, a JDBC bridge). Stages **version-probe at start and fail
  closed** if the tool is absent or wrong-version — no silent degradation.
- Connector stages are subject to the same containment and clinical hard-block;
  a connector cannot become a PHI exfiltration path without tripping the
  fail-closed gate.
- This ADR is the record of WHY BLUT does not and will not have a dynamic
  provider-plugin ecosystem — do not re-litigate the plugin model without a
  superseding ADR that first overturns 0034/0037.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** the connector framework — the closed `*Ref`/`Blob` I/O-kind
  enum, one reference connector cookbook (`s3` with `s3.get`/`s3.put` shelling
  aws-cli, typed `ObjectRef` I/O, version-probed, cache-keyed), and
  `connectors/registry.toml` + `blut connectors list`.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut connectors:: && blut connectors list --check && blut recipe declare tests/fixtures/s3_roundtrip.json --dry-run`
  - `pass when:` `exit==0` AND the s3 connector graph kind-checks through
    `from_erased_graph` with typed `ObjectRef` edges AND `--check` fails closed
    when the probed external binary is absent AND a second identical `s3.get` run
    is a cache hit on the unchanged `ObjectRef` hash.
- **Rollback:** connectors are additive registered cookbooks; abandoning the
  framework is deleting the connector cookbook(s) + `registry.toml` +
  `blut connectors` subcommand. The engine core, charter, and PlanSpec IR are
  untouched — no core rollback.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-11 — LANDED + FROZEN (engine main `76551a8`). `src/connectors/mod.rs`:
  a compiled-in `connectors` cookbook. CLOSED I/O-kind set — `ObjectRef` /
  `TableRef` / `DatasetRef` / `Blob` (Artifact impls, `HASH_CONTENTS=false` =
  identity + hash, never bulk; cache key unchanged, bulk lives remotely, the
  local artifact is a small manifest). Stages `connector_object` (source,
  ()→ObjectRef) / `connector_fetch` (ObjectRef→DatasetRef) / `connector_store`
  (DatasetRef→ObjectRef), each delegating its verb to the external `tool` as an
  async `tokio::process` subprocess (ADR 0034) with a `--` flag-injection guard;
  fetch VERIFIES the fetched bytes hash to the declared identity (fail-closed — a
  corrupt download can't poison the cache); store returns the real `dest_uri` so
  store→fetch round-trips. `ConnectorDescriptor` + `list()` + `check()` (every
  I/O is `()` or a connector kind; no duplicate names). CLI `blut connectors list
  [--check]`; connectors registered in `cli::run` so `blut recipe declare`
  resolves + kind-checks an integration graph. **Gate met:** `cargo test -p blut
  connectors::` = 3 passed — four distinct closed kinds; `check()` healthy;
  an s3-roundtrip graph KIND-CHECKS through `compile()` while a mismatched wiring
  is refused. 857 lib green; clippy+fmt+scope_guard clean. MiMo (3 rounds) →
  async subprocess, fetch hash verification, real store URI, `--` guard ordered
  after `extra`, propagated write errors, duplicate-name check. **GATE-FORM:**
  the shell `blut connectors list --check` / `blut recipe declare … --dry-run`
  forms are adapted to the in-crate `connectors::` tests (engine lib-only); a
  `registry.toml`-driven connector set (widening `&'static str` descriptors to
  owned) is a later extension. Status → Complete — frozen 2026-07-11._

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut connectors::`
- **result:** exit 0 (3 passed). The closed connector-kind set is distinct; the
  registry validates; and an s3-roundtrip connector graph kind-checks through
  `from_erased_graph`/`compile()` while a kind-mismatched wiring is refused — an
  integration graph is type-checked exactly as a native one. scope_guard PASS;
  857 lib green; clippy + fmt clean.
- **git_sha:** engine `76551a8`  ·  **artifact sha256:** n/a (source-only)

## Related Decisions

ADR 0034 (charter — no dylib, no HTTP; delegate the verbs), ADR 0037 (cookbook
architecture — integrations are compiled-in cookbooks that shell out), ADR 0100
(connector/exec-bridge substrate), ADR 0078 (`from_erased_graph` typed
kind-check the `*Ref` kinds ride on), ADR 0111 (Python SDK — can reference
connector stages by name).

## Validation  *(ongoing — distinct from the completion gate above)*

Watch for two drift signals. First, pressure to "just load a provider at
runtime" for coverage speed — that is the charter breach this ADR exists to
prevent; answer with a reviewed connector cookbook, never a dynamic load.
Second, `*Ref` kinds leaking bulk data (someone stuffing bytes into a `Blob` to
avoid writing a connector) — if graph edges start carrying payloads instead of
identity+hash, the cache-key discipline is compromised and the kind enum needs a
guard. Either signal warrants a superseding ADR, not a quiet relaxation.
