<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0111 — Ship a Python client SDK that emits the PlanSpec IR and drives runs over the CLI

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (frozen 2026-08-17)

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite or delete earlier text; correct an earlier statement with a *new* dated
> entry beneath it, so the original stays as historical record. When the
> deliverable passes its gate, fill **Completion** and set Status to
> `Complete — frozen YYYY-MM-DD`.

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-08-17`

Part of Group 7 (Interfaces & ecosystem) of the "BLUT real-tool" roadmap. Builds
on [ADR 0078](0078-blut-starlark-typed-dynamic-dag.md) (the `PlanSpec` JSON IR + `blut recipe declare <file>.json` door)
and [ADR 0093](0093-blut-web-control-plane.md) (the exec-bridge). Does not supersede either.

## Context

ADR 0078 froze a **stable, versioned, serde IR** — `PlanSpec` — as the single
plain-data contract every DAG front-end emits, and shipped the CLI door that
consumes it: `blut recipe declare <file>.json` compiles raw PlanSpec JSON
through the feature-independent `.json` path (the one explicitly called out as
"the future Python SDK's entry point"). Two front-ends emit that IR today —
Starlark scripts (out-of-process `crates/blut-dsl`) and hand-written JSON — but
there is **no ergonomic authoring surface for a non-Rust user**. Writing
PlanSpec JSON by hand is possible but hostile: node ids, `(from,to)` edge pairs,
and typed map `expansions` are a wire format, not an API.

Every competing orchestrator won adoption through a **Python SDK**, not a config
file: Flyte's `@task`/`@workflow`, Prefect's `@flow`, Metaflow's `FlowSpec`,
Dagster's `@asset`. A data scientist expects to `pip install`, write a few
decorated Python functions, and get a graph. BLUT has the typed engine and the
IR but no such front door — the Starlark path is Python-*shaped* but is a
separate hermetic language with no `pip` story, no IDE completion, and no way to
programmatically submit/monitor/control a run from the same script that built
it.

The charter constrains the shape hard: [ADR 0034](0034-blut-scope-charter.md) forbids an in-engine HTTP server
and dynamic dylib plugins; ADR 0078's anti-re-litigation clause forbids letting
any front-end load stage *code* — a front-end may only *name* registered stages
and emit `PlanSpec`. Any SDK must live entirely **outside** the engine and talk
to it only through artifacts the engine already reads/writes.

## Decision

Ship **`blut-sdk`**, a pure-Python client package (`pip install blut-sdk`, no
compiled extension, no engine linkage) with three layers, all resting on the
existing `PlanSpec` v1 IR and CLI:

1. **A builder that emits PlanSpec JSON.** `PlanBuilder.add(stage, args=,
   after=)` mirrors the Starlark `b.add` builtin one-to-one: fork = shared
   `after`, merge = list `after`, compile-time map = a Python `for` loop, runtime
   map = `builder.map_output(parent, body)` (emits a `MapSpec` expansion). The
   builder's sole output is **canonical PlanSpec v1 JSON** — byte-identical in
   structure to what `blut-dsl` emits, so both front-ends converge on the same
   `from_erased_graph` kind-check and the same `provenance_fingerprint`. The SDK
   validates stage names and arg shapes against a registry manifest the engine
   exports (`blut registry export --json`), so an unknown stage is caught
   client-side *before* submission (the engine re-checks authoritatively).

2. **Submit/monitor/control by shelling the CLI (the exec-bridge, ADR 0093).**
   `Run.submit()` writes the PlanSpec to a temp file and invokes
   `blut recipe declare run.json` as a subprocess; `Run.status()` /
   `Run.stream()` read the engine's existing `status.jsonl` / `StatusUpdate`
   artifacts; `Run.cancel()` issues the existing control verb. The SDK is a
   **client of the CLI**, exactly like `kubectl`/`slurm` adapters — it opens **no
   socket to the engine and embeds no runtime.** No new engine surface is
   required beyond the read-only `blut registry export`.

3. **A thin decorator sugar over the builder** (`@stage_ref` / `@workflow`) that
   is *only* naming ergonomics — it resolves to `PlanBuilder.add` calls and
   cannot introduce new executable stages. Decorators name registered stages;
   they never ship stage bodies into the engine.

The SDK is **not a second runtime.** It emits the same IR Starlark does; the
engine's behavior is identical regardless of which front-end authored the JSON.

## Rationale

- **The white-space is real and the contract already exists.** ADR 0078
  deliberately froze `PlanSpec` as "the Python SDK's interface" and shipped the
  `.json` door feature-independently. This ADR fills the authoring gap the
  earlier one left open by design — it is additive, not a redesign.
- **Charter-clean by construction.** Emitting JSON + shelling the CLI touches
  zero forbidden surfaces: no in-engine HTTP (ADR 0034), no dylib (the SDK is
  never linked into the engine), no code loading (it names registered stages).
  The engine only ever *reads* JSON and *writes* status — unchanged from 0078.
- **Same IR ⇒ same guarantees for free.** Because the builder emits canonical
  PlanSpec v1, scripted and SDK-authored plans share the `from_erased_graph`
  typed kind-check, the unchanged per-stage cache keys (`hash(code_sha ⊕
  canon_args ⊕ input_hash)`), and the plan `provenance_fingerprint`. The SDK
  inherits typing and caching without re-implementing either.
- **Meets users where they are.** `pip install` + decorated functions is the
  idiom Flyte/Prefect/Metaflow trained the field on; a pure-Python client with
  no compiled extension is trivially installable in any notebook or CI, and
  needs only a `blut` binary on `PATH`.
- **Client-side validation shortens the loop** without moving authority: the
  registry manifest lets the SDK reject a typo before a subprocess launch, while
  the engine remains the single source of truth on kind-checking.

## Alternatives Considered

- **Embed a Python interpreter in the engine (PyO3), let users register
  `@task`s that ship into the executor** — REJECTED. This is exactly the dynamic
  code-loading ADR 0078's anti-re-litigation clause and ADR 0034 forbid: a task
  body is arbitrary code entering the engine. It also drags CPython into a Rust
  binary that must cross-compile to firmware targets. The SDK stays a client.
- **A gRPC/REST control server the SDK talks to** — REJECTED: ADR 0034 bans an
  in-engine HTTP server outright. The exec-bridge (subprocess + artifact files,
  ADR 0093) already gives submit/monitor/control with no socket.
- **Generate Starlark from Python and shell `blut-dsl`** — REJECTED: a lossy
  two-hop translation (Python → `.star` text → PlanSpec) with a second language's
  error messages leaking to Python users. Emitting `PlanSpec` JSON directly is
  one hop to the same contract and gives native Python tracebacks.
- **A compiled `maturin`/PyO3 SDK sharing the engine's Rust `PlanSpec` types** —
  REJECTED for v1: a compiled wheel per platform kills the "works in any
  notebook" property; the IR is simple JSON, so a pure-Python mirror of the v1
  schema is cheap and keeps install frictionless. Revisit only if schema drift
  becomes a maintenance cost.

## Consequences

- A new `blut-sdk` Python package (out-of-tree of the engine crate; ships from
  the meta-repo the same way `crates/blut-dsl` does) becomes a **versioned client
  of the `PlanSpec` v1 wire contract.** It must track the IR's additive-only
  evolution; a `PlanSpec` version bump requires an SDK compat entry.
- A new read-only engine subcommand `blut registry export --json` becomes
  load-bearing (the SDK's client-side validation source). It is pure output —
  no new control surface.
- The SDK's parity with `blut-dsl` must be tested: a golden-corpus check that the
  Python builder and the Starlark builtin emit **byte-identical canonical
  PlanSpec** for the same logical graph, so the two front-ends never diverge.
- Clinical hard-block is inherited, not weakened: the SDK submits through the
  same CLI path, so the fail-closed clinical gate fires identically regardless of
  authoring front-end.
- Documentation gains a Python quickstart; this does not make Python a *runtime* —
  the anti-re-litigation clause of ADR 0078 still stands and is referenced here.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** `blut-sdk` (pure-Python, no compiled extension) exposing
  `PlanBuilder`/`Run`, plus the read-only `blut registry export --json`
  subcommand; the builder emits canonical `PlanSpec` v1 JSON accepted unchanged
  by `blut recipe declare <file>.json`.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `python -m blut_sdk.tests.parity && blut recipe declare $(python -m blut_sdk.examples.fanout --emit /tmp/sdk_plan.json) --dry-run`
  - `pass when:` `exit==0` AND the parity test asserts the SDK-emitted
    canonical PlanSpec bytes equal the `blut-dsl` fixture for the same graph AND
    the engine compiles the SDK-authored plan through `from_erased_graph` with no
    kind-break.
- **Rollback:** the SDK is out-of-tree and additive; abandoning it is `pip
  yank blut-sdk` + reverting the `blut registry export` subcommand. The engine,
  `PlanSpec` IR, and `.json` CLI door are untouched, so no engine rollback is
  needed.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-12 — M2.0 CONTROL REPAIR: planning contract only. No `blut_sdk`
  package exists in the current repository, so the listed parity command cannot
  run. `PlanSpec` is versioned wire data, but the engine is now explicitly a
  `0.2.0-alpha.1` preview; do not call its full surface semver-stable until the
  M6 compatibility and release gates pass._
- _2026-08-17 — DELIVERED. `blut-sdk` lands at `sdk/` in the engine repo
  (not a separate repository): it MIRRORS the PlanSpec v1 IR, so co-location
  makes an IR change and its SDK compat entry land in one commit, and lets
  engine CI arm the parity gate this ADR's Validation section asks for
  (`sdk-parity` job). Both gate halves are green — see Completion._
- _2026-08-17 — CORRECTION to the Implementation Plan's `gate_cmd` spelling.
  It reads `blut recipe declare … --dry-run`, but `recipe declare` has no
  `--dry-run` flag: it compiles, kind-checks and renders the DAG, executing
  nothing, **unless** `--run` is passed. The dry run is the default, and
  `--dry-run` (which belongs to `recipe run`) is rejected with "unexpected
  argument '--dry-run' found". The gate was therefore run without that flag,
  which is the semantics the plan intended. Recorded rather than silently
  substituted, per the freeze discipline._
- _2026-08-17 — the "no new engine surface beyond the read-only `blut registry
  export`" claim held, but the surface needed three passes to be USABLE, and
  two of those fixed pre-existing engine defects rather than adding anything:_
  1. _The first export (`497469ed`) carried only the **recipe** catalog. A
     `PlanSpec` node's `stage` resolves through `find_erased_stage` over
     `Cookbook::stages_erased`, a different registry whose contents barely
     overlap the recipe list — so a client validating plan nodes against it
     would have rejected every legitimate plan while appearing to validate.
     Fixed in `2375b703` by exporting the stage palette (name, input/output/
     element kinds) alongside recipes._
  2. _`StageDyn::args_schema` returned `{"$ref": "#/definitions/Args"}` while
     dropping the `SchemaGenerator` holding the definitions — a reference to a
     map that was never emitted. Every consumer resolved it to nothing, so
     BLUT's own TUI DAG builder had been offering every stage as taking no
     arguments. Its test asserted only `schema != Value::Null`, which a
     dangling `$ref` satisfies: true and meaningless. Fixed in `b747dfec`;
     305 stage arguments now resolve where 0 did._
  3. _`plan_spec_version` added to the manifest so an SDK can DETECT the IR
     skew this ADR's Validation section warns about, instead of the skew being
     found in the field._
- _2026-08-17 — two notes for whoever maintains the canonical-bytes port.
  (a) The engine re-canonicalizes any spec it receives, so a divergence in the
  Python encoder cannot corrupt a provenance fingerprint or a cache key —
  it can only corrupt a comparison against engine-produced bytes, which is
  exactly what the gate does. The port is therefore pinned against a
  2526-value table MEASURED from serde_json via `blut-dsl`, not derived from
  reading its source. (b) `repr` and ryu differ in exactly two places, both
  found by measurement: ryu does not zero-pad exponents (`1e-7`, not `1e-07`),
  and it uses decimal notation over decimal exponents `[-5, 15]` where Python
  uses `[-4, 15]` — so they disagree on precisely the `1e-5` band._
- _2026-08-17 — the gate proves it can FAIL, not merely that it passes. Each
  run plants four drifts (reversed merge order, an int where a float belongs,
  `True` written as `1`, a dropped `map_output` label) and fails if any goes
  undetected; the live-`blut-dsl` comparison prints a loud SKIP when the binary
  is absent, and CI fails on that SKIP since it builds the binary first. An
  early revision silently skipped that check on an off-by-one path — the exact
  vacuous-gate failure recorded against ADRs 0103 and 0109._

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** passed
- **gate_cmd:**
  `cd sdk && python -m blut_sdk.tests.parity` **and**
  `<cookbook-binary> recipe declare $(python -m blut_sdk.examples.fanout --emit /tmp/sdk_plan.json)`
  (`<cookbook-binary>`: the domain cookbook's CLI `[redacted: non-engine product detail]`;
  the plan's `--dry-run` spelling corrected — `recipe declare` executes
  nothing unless `--run` is passed, and rejects `--dry-run` outright; see the
  2026-08-17 Progress Log entry)
- **result:** both halves exit 0, measured 2026-08-17 on the local box.
  - Parity: `PASS — 14 checks`. The SDK-emitted canonical PlanSpec bytes equal
    the `blut-dsl` fixture for the same graph, AND equal today's live
    `blut-dsl` output for that graph (the fixture is not merely self-consistent).
    Includes the four planted drifts, all detected.
  - Engine compile: `plan: sdk_fanout (5 nodes, 5 edges) … ✓ 'sdk_fanout'
    compiles + kind-checks (5 ingredient(s))` — an SDK-authored plan through
    `from_erased_graph` with no kind-break, over a real fork and an ordered
    2-tuple merge.
  - 62 pytest tests pass. The gate also runs from an INSTALLED wheel
    (13 checks; the live-`blut-dsl` comparison correctly SKIPs with no engine
    tree above `site-packages`, and says so).
  - Float encoding matches `serde_json` across all 2526 measured values.
  - Client-side validation resolves 305 declared stage arguments through the
    schemars `$ref`; a naive `schema["properties"]` read resolves 0, and the
    gate fails if that ever stops being true.
- **git_sha:** engine `0742e00b` (SDK + `sdk-parity` CI job), on
  `b747dfec` (stage args schemas resolve), `886ceb4c` (stage args + IR
  version exported), `2375b703` (stage palette), `497469ed` (`registry
  export`); cookbook commit `[redacted: private repository]`  ·  **artifact sha256:**
  `a9b8324986b8b78494c030510c8e8d536d0c4730add3c406a09794f66f201280`
  (`sdk/src/blut_sdk/tests/fixtures/parity_graph.json`, the `blut-dsl` golden)
- **not claimed:** the package is not published to PyPI (this repo is
  private, and RELEASING.md treats that as a hard stop), and the CI job runs
  only the parity half — the engine-compiles half needs a cookbook binary,
  since a bare `blut` registers no stages.
  *Note (2026-10): the engine repository has been public since
  0.2.0-alpha.1; "this repo is private" describes it when this was written.*

## Related Decisions

[ADR 0078](0078-blut-starlark-typed-dynamic-dag.md) (PlanSpec IR + `.json` door — the contract this SDK emits), [ADR 0093](0093-blut-web-control-plane.md)
(exec-bridge — the submit/monitor/control channel), [ADR 0034](0034-blut-scope-charter.md) (charter — no HTTP
server, no dylib), [ADR 0037](0037-blut-cookbook-architecture.md) (cookbook architecture — stages are the vocabulary),
[ADR 0112](0112-blut-ecosystem-integrations.md) (ecosystem integrations — connector cookbooks the SDK can reference).

## Validation  *(ongoing — distinct from the completion gate above)*

Watch for the SDK becoming a *second runtime*: any pressure to let it ship
executable task bodies, open a socket to the engine, or diverge from canonical
`PlanSpec` bytes is the signal that this decision drifted — respond with a
superseding ADR, not an incremental relaxation. Also watch `PlanSpec` version
skew: if the SDK and engine IR versions diverge in the field without a compat
entry, the parity gate has rotted and must be re-armed in CI.
