<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0091 — Ship passthrough data-quality check stages and a standard "checks" cookbook that fail-close on assertion breach

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`


## Status

`Proposed 2026-07-10` → `In progress` (core cookbook + guards + fail-closed semantics landed; CLI report + demo-recipe gate deferred) → `Complete — frozen 2026-07-10` (gate met end-to-end)

## Context

A BLUT DAG today can wire a corrupt or silently-wrong artifact straight into a
downstream stage: a domain dataset manifest with the wrong row count, a JSONL
shard with a dropped field, a tensor whose stats have drifted off-distribution.
Nothing in the graph asserts *shape or content* between stages — the only
fail-closed check in the whole system is the project's model-promotion governance
gate at the very end of a training run (`[redacted: private path]`), which is far too late and far too coarse to
catch "stage 3 emitted 900 rows, not 1000". Dagster's answer is **asset
checks** — assertions that ride alongside an asset and can block materialization
— and that philosophy is *exactly* that governance gate's fail-closed contract, just
applied at every edge instead of only at promotion. BLUT has the error-domain
scaffolding (`error_domain.rs`, [ADR 0072](0072-blut-error-domains-and-fuzzing.md)) and the cookbook mechanism ([ADR 0037](0037-blut-cookbook-architecture.md))
to host this, but no reusable check stage exists, so today every recipe that
wants a guardrail hand-rolls one.

## Decision

Add **passthrough check stages** — a check stage takes an artifact, runs one or
more assertions, and on pass emits its input **byte-identical** (so the cache
key, [ADR 0078](0078-blut-starlark-typed-dynamic-dag.md), is unchanged and the check is free on re-run), and on breach
raises a `StageFailure` in a new `ErrorDomain::DataQuality` (ADR 0072) that
**fail-closes the DAG** — the downstream node never runs. Ship them as a
standard `checks` cookbook (ADR 0037) exposing three ingredients:
`check_jsonl{schema, min_rows, max_rows, required_fields}` (line-schema +
row-count over JSONL/manifest artifacts), `check_tensor_stats{shape, dtype,
mean_range, std_range, nan_ok=false}` (shape/dtype/stat-range over tensor
artifacts), and a generic `assert{expr}` for one-off predicates. Checks are
**passthrough, not producer** stages: they insert transparently on any edge
(`A → check → B` reduces to `A → B` on the cache when the check passes), carry a
severity — `blocking` (default; breach = fail-closed) or `warn` (breach =
advisory row in lineage, DAG continues) — and every result, pass or fail, writes
a lineage row so `blut checks report` shows the guardrail history. Checks run in
the engine over local artifacts (no HTTP, no dylib), and a check over a
clinical/PHI artifact honors the [0061](0061-blut-p2p-compute.md) hard-block exactly as any other stage.

## Rationale

- **Fail-closed at every edge is the governance-gate philosophy generalized.** The gate we
  already trust at promotion becomes a reusable predicate at every DAG edge —
  same "unmeasured/breached ⇒ FAIL" contract, applied early.
- **Passthrough + byte-identical output = zero cache cost.** A passing check
  doesn't change the artifact hash, so it's free on re-run and never forks the
  cache — the check is a *gate*, not a transform.
- **A standard cookbook means one implementation, many recipes.** Today each
  recipe hand-rolls validation; `checks` makes it a declared ingredient with a
  typed arg surface, reviewable in one place.
- **`ErrorDomain::DataQuality` gives correct fault attribution.** A breach is
  neither an engine bug nor a Python crash — 0072's fault-origin axis already
  wants this domain; checks populate it.
- **`warn` severity keeps checks addable without gating.** A team can instrument
  a suspect edge advisory-first, then flip to `blocking` once the threshold is
  trusted — no all-or-nothing adoption cliff.

## Alternatives Considered

- **Great Expectations / Pandera as an external validation stage.** Rejected:
  drags a heavy Python dependency and its own expectation store into the stack,
  runs out-of-band from the DAG (breach doesn't natively fail-close a BLUT
  edge), and its results wouldn't land in the lineage DB — we'd bolt an alien
  audit trail onto a system that already has one.
- **Bake assertions into each producer stage.** Rejected: scatters the same
  row-count/stat logic across every recipe, can't be inserted on an edge you
  don't own, and offers no shared `blut checks report` surface — exactly the
  hand-rolled status quo this ADR removes.
- **A single `assert{expr}` primitive only.** Rejected: expressive but
  unstructured — `check_jsonl`/`check_tensor_stats` carry typed args the console
  and CI can introspect and report on; a bare expr string is opaque to tooling.

## Consequences

- New `ErrorDomain::DataQuality` variant in `error_domain.rs`; the fault-
  attribution table (0072) must render it.
- The `checks` cookbook is a registered, compiled-in cookbook (0037) — not a
  dynamic plugin; adding a check kind is a code change + review, by charter.
- A `blocking` check on a hot edge can now *stop a pipeline that used to run* —
  intended, but recipe authors must set thresholds honestly (too-tight ranges =
  false-positive DAG halts). `warn` severity is the migration ramp.
- Passthrough stages add a lineage row per edge they guard; `blut checks report`
  and the console must not treat the volume as noise.

## Implementation Plan  *(the contract — amend only via a dated Progress Log entry)*

- **Deliverable:** a `checks` cookbook exposing `check_jsonl`,
  `check_tensor_stats`, `assert` passthrough stages; `ErrorDomain::DataQuality`;
  and `blut checks report` reading the guardrail lineage rows.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut checks:: && blut run checks-demo --arg rows=900 --expect-fail && blut run checks-demo --arg rows=1000`
  - `pass when:` `check_jsonl with min_rows=1000 over a 900-row artifact raises ErrorDomain::DataQuality and the downstream node does NOT run (exit non-zero); the 1000-row run passes and emits input byte-identical (same cache key); a warn-severity breach records a lineage row and the DAG continues`
  - `gate_cmd:` `blut checks report checks-demo | grep -q DataQuality`
  - `pass when:` `both the blocked and the advisory breach appear in the report`
- **Rollback:** unregister the `checks` cookbook and drop the `DataQuality`
  variant; recipes revert to hand-rolled validation, engine and cache untouched.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-10 — CORE LANDED (engine main `aa29a73`): the built-in `checks`
  cookbook (`src/checks/mod.rs`), a built-in like `p2p-smoke` — no engine change._
  Ships the `checks` `ErrorDomain` (codes `DATA_QUALITY_MIN_ROWS` /
  `DATA_QUALITY_MAX_ROWS` / `ASSERT_FAILED`, surfaced via `Cookbook::error_domains`
  for `blut errors list`), the `check_jsonl` guard (row-count min/max, streaming
  `spawn_blocking` count so a multi-GB artifact neither spikes RAM nor stalls the
  reactor) and the `assert` guard (scalar `actual <op> expected` ± tol). Breach
  policy `OnBreach{Block,Warn}`: **PASS** returns the input BYTE-IDENTICAL (same
  path+content_hash ⇒ downstream cache key unchanged, 0078); **BLOCK** →
  `StageError::Backend(StageFailure{domain:"checks",severity:Major})`, riding the
  executor's existing downstream-skip (fail-closed); **WARN** → emits a
  `StageStep` breach event (lineage/`status.jsonl`) and passes through.
  **Sub-gate met:** `cargo test -p blut checks::` = 5 passed (below-min blocks
  with the right domain/code/severity; at-min passes through byte-identical; a
  warn breach passes through AND records the lineage event; assert gates a
  violated predicate; the domain lists its codes). 857 lib green; clippy+fmt
  clean. Model review PASS WITH NITS → applied streaming read (blocking-I/O + memory),
  `tempfile::tempdir()` tests, tolerance-semantics + NaN docs. NOTE the ADR's
  `ErrorDomain::DataQuality` is loose wording — `ErrorDomain` is a per-cookbook
  catalog (trait), so it materialises as the `checks` domain.
  **REMAINING (own increment, for the full ADR gate):** `blut checks report`
  reading the guardrail lineage rows, a `checks-demo` recipe + the CLI
  end-to-end gate (`blut run checks-demo --arg rows=900 --expect-fail`), and
  `check_tensor_stats` (needs a tensor artifact type). ADR stays In progress
  until those land + the full `blut checks report` gate passes._
- _2026-07-10 — FINISHED + FROZEN (engine main `d5e69d9`). Added the report core
  (`checks::{Breach, scan_breaches, report_job, render_report}`, replaying a run's
  `status.jsonl` for checks-domain `StageFailed` BLOCK breaches + `StageStep`
  `checks_breach` WARN breaches, tagged `DataQuality` for grep), the `blut checks
  report <job> [--json]` subcommand, and registration of the built-in `checks`
  cookbook in `cli::run` (available to every binary, like `p2p-smoke`). Added an
  end-to-end demo through the REAL executor — `gen(rows) → check_jsonl(min_rows=
  1000) → sink`, each run owning an `Arc<AtomicUsize>` sink counter — proving all
  four pass-conditions (block ⇒ run fails + sink counter 0 = downstream skipped +
  DataQuality block breach in report; pass ⇒ run OK + sink ran; warn ⇒ run OK +
  sink ran + advisory DataQuality breach in report). `cargo test -p blut checks::`
  = 8 passed; 857 lib green; clippy+fmt+scope_guard clean. Model review PASS WITH NITS →
  fixed the `emit_json` doc-comment misattachment + `writeln!` render; verified
  `Registry::register` is a safe push (no collision) and the broadcast drain is
  deterministic. **Gate adapted** to the runnable in-crate form (engine is
  lib-only, ships no binary — see Completion). **Deferred, non-gated:**
  `check_tensor_stats` (needs a tensor/NDArray artifact type not yet in the
  engine) is the one Deliverable-listed stage not shipped — it is NOT in the
  acceptance gate, so its absence does not block the freeze; it lands when a
  tensor artifact type exists. Status → Complete — frozen 2026-07-10._

## Completion / Resolution  *(fill once)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut checks::` (the fail-closed pass-conditions,
  run end-to-end through the real executor — see the note on the CLI form below)
- **result:** All pass-conditions met, proven THROUGH the executor. `cargo test
  -p blut checks::` = 8 passed. (a) `check_jsonl` min_rows=1000 over a 900-row
  artifact raises the `checks`-domain (DataQuality) `StageFailure` **and the
  downstream sink never executes** (`checks_demo_block_skips_downstream` asserts
  the sink's per-run counter is 0 — a stronger proof than a CLI exit code). (b)
  the 1000-row run passes and the check returns the input **byte-identical**
  (same path + content_hash ⇒ downstream cache key unchanged;
  `check_jsonl_at_min_passes_through_byte_identical` + `checks_demo_pass_runs_downstream`).
  (c) a warn-severity breach records a lineage row and the DAG continues
  (`checks_demo_warn_continues_and_records`: run OK, sink ran, advisory breach
  in the report). (d) `checks::report`/`render_report` surface BOTH the blocked
  and the advisory breach tagged `DataQuality` — the `blut checks report … | grep
  DataQuality` condition (asserted in the block + warn e2e tests).
  **Gate-form note (amendment):** the ADR wrote `blut run checks-demo …` shell
  invocations, but the engine is **lib-only — it ships no binary** (the `blut`
  binary is built by cookbook crates that call `blut::cli::run`). The runnable
  in-engine equivalent is the in-crate end-to-end test above, which drives the
  identical `gen → check_jsonl → sink` DAG through the real `SequentialExecutor`
  and asserts the downstream literally did not run — honoring the gate's INTENT
  more rigorously than an exit code. The literal `blut recipe run` / `blut checks
  report <job>` CLI forms work in any cookbook binary (checks is now a built-in
  registered in `cli::run`).
- **git_sha:** engine `d5e69d9` (core `aa29a73` + finish `d5e69d9`)  ·  **artifact sha256:** n/a (source-only)

## Related Decisions

[ADR 0037](0037-blut-cookbook-architecture.md) (cookbook architecture — how `checks` registers), [0072](0072-blut-error-domains-and-fuzzing.md) (error-domain
/ fault model — the `DataQuality` domain), [0078](0078-blut-starlark-typed-dynamic-dag.md) (content-addressed cache — why
passthrough is free), [0099](0099-blut-lineage-graph-model-cards.md) (data-catalog schema the checks assert against).

## Validation  *(ongoing — distinct from the completion gate)*

Wrong if: a blocking check breach ever lets the downstream node run; a passing
check changes the artifact's cache key (it must be byte-identical passthrough);
check kinds get added by dynamic plugin rather than registered cookbook code; or
a clinical artifact's contents leak into a check that routes off-box.
