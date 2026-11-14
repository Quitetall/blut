<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0090 — Give BLUT first-class experiment, model, and dataset registries addressed by content-hashed URIs

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-12`
(engine `53998a4`)

## Context

BLUT already computes a content hash for every stage output (the cache key,
[ADR 0078](0078-blut-starlark-typed-dynamic-dag.md)) and records lineage rows in
`src/lineage_db.rs`, and the project's checkpoint-promotion gate
(`[redacted: private path]`) fail-closes checkpoint promotion. What
it lacks is a **named, queryable surface** over any of it. A trained checkpoint
is a `sha256:…` cache entry with no human name, no `@prod` pointer, no
promote/rollback verb; a corpus is a path on `[redacted: private path]` with no `@v3`
handle; an experiment is a `blut run` invocation whose comparison to its
siblings lives only in a human's head (ADR 0070 [not federated] is a
*hand-maintained markdown* ledger — honest prior art, but not a programmatic
registry). MLflow/W&B ship a model registry with stage aliases
(`Staging`/`Production`) and rollback; Metaflow ships namespaced experiment/run
tracking; DVC ships `dataset@rev`. BLUT operators today reconstruct all three by
grepping lineage and remembering which `sha256` was "the good one". The
primitives exist — hashes, lineage, gates — but there is no registry noun to
bind a name to a hash and move that binding under audit.

## Decision

Add three thin, content-addressed registries as **engine-side reads over the
existing lineage DB plus one new append-only `refs` table** — no new store, no
new daemon. (1) `registry://<name>@<alias>` resolves a model name + mutable
alias (`prod`, `staging`, `@v7`) to an immutable `sha256` artifact; `blut model
promote <name> <sha> --to prod` and `blut model rollback <name>` append a new
row to the `refs` table (never mutate — the pointer's whole history is the audit
trail), and promotion to a **governed alias fail-closes on the governance gate**
(`[redacted: private path] … --change-id`) so `@prod` can never point at an
unpromoted checkpoint. (2) `dataset://<name>@<v>` resolves a dataset name+version
to the content hash of its manifest (the domain dataset manifest already hashes
its members), giving reproducible corpus pinning in a recipe's typed args. (3)
`experiment://<name>/<run>` groups `blut run` invocations under a name so
`blut exp compare <name>` diffs their input hashes, param diffs, and headline
metrics straight from lineage. All three are pure lineage-DB queries + the
`refs` table; resolution is offline, deterministic, and clinical/PHI datasets
carry the [0061](0061-blut-p2p-compute.md) hard-block flag so a `dataset://`
handle can never resolve inside a cloud-routed stage.

> **Implementation correction — 2026-07-12 (authoritative):** the landed model
> URI is `model://<name>@<alias>`, with CLI forms `model promote <hash>
> model://<name>@<alias>` and `model rollback model://<name>@<alias>`. Model
> pointers share the existing [ADR-0085](0085-blut-deployment-registry.md)
> registry SQLite file, dataset bindings share the existing datasets SQLite
> file, and experiments are lineage-only views. No new daemon or additional
> store was introduced. This appended note supersedes the stale CLI/storage
> wording immediately above; the Progress Log records the executable gate and
> migration evidence.

## Rationale

- **The hashes already exist** — a registry is a *naming layer*, not a storage
  layer. Binding `registry://enc@prod → sha256:…` is a lookup table, so the
  build cost is a schema + resolver, not a new subsystem.
- **Append-only refs = free audit + free rollback.** Rollback is "read the
  previous row"; there is no state to corrupt, matching the content-addressed,
  retry-is-relaunch philosophy of [ADR 0083](0083-blut-interface-stack.md)'s
  exec bridge.
- **The governance gate becomes reusable as a promotion predicate.** `@prod`
  promotion routes through the *same* fail-closed gate that governs checkpoint
  promotion today — one enforcement point, not two.
- **URIs are typed args, so recipes pin inputs by name.** `dataset://tuh@v3`
  in a recipe is reproducible and diffable; a bare path is neither.
- **Charter-clean:** reads run in the engine; no HTTP (a future `blut-web`
  sidecar, ADR 0083, can surface it); no dylib; broker still owns compute.

## Alternatives Considered

- **Adopt MLflow Tracking + Model Registry as the backend.** Rejected: pulls a
  Python HTTP server + its own SQL store into the stack, duplicating the
  content-addressed cache and lineage DB we already trust, and violates the
  [0034](0034-blut-scope-charter.md) no-in-engine-server charter. We'd be
  maintaining two sources of truth for "which artifact is prod".
- **Keep the ADR 0070 [not federated] markdown ledger as the only registry.**
  Rejected: a hand-edited table cannot fail-close a promotion, cannot be resolved
  by a recipe's typed args, and drifts from the lineage DB the moment someone
  forgets to update it. It stays as the *narrative* experiment log; the machine
  surface is this ADR.
- **A separate registry microservice.** Rejected: it's a daemon and a second
  store for data the lineage DB already holds; the registry is a *view*, so a
  service is strictly more moving parts with no new capability.

## Consequences

- `src/lineage_db.rs` gains a `refs` table (append-only) and three resolvers;
  the lineage schema migration must be backward-readable.
- Every `@prod`/governed-alias promotion now *requires* a green gate — a
  deliberate friction that blocks "just point prod at my new checkpoint".
- Recipes referencing `dataset://`/`registry://` URIs become non-portable to a
  machine without the resolved artifact in its cache — resolution must produce a
  clear "unresolved handle" error, never a silent path fallback.
- The markdown ledger (0070 [not federated]) and this registry must agree; the
  ledger becomes the prose annotation *of* registry rows, not an independent
  record.

## Implementation Plan  *(the contract — amend only via a dated Progress Log entry)*

- **Deliverable:** a `refs` table + `Registry` resolver in `src/lineage_db.rs`,
  the `blut model promote|rollback`, `blut dataset pin`, and `blut exp compare`
  subcommands, and URI resolution (`registry://` / `dataset://` /
  `experiment://`) wired into typed-arg parsing.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut registry:: && blut model promote demo <sha> --to prod && blut model rollback demo && blut model promote demo <unpromoted-sha> --to prod; test $? -ne 0`
  - `pass when:` `resolver round-trips a name→sha→name; rollback restores the prior ref row; promotion of a checkpoint that fails the governance gate to a governed alias exits non-zero (fail-closed); a dataset:// handle flagged clinical refuses to resolve in a cloud-routed stage`
- **Rollback:** revert the `refs` migration and the three subcommands; the
  lineage DB and cache keys (0078) are untouched, so the engine returns to
  hash-only addressing with zero data loss.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-12 — INCREMENT 1 LANDED (engine main `5a2e93f`): the MODEL registry
  `model://<name>@<alias>`, chosen first (user steer) for maximal CONSISTENCY —
  it reuses the frozen ADR-0085 plan-registry pattern verb-for-verb and shares
  its `~/.blut/registry.db`, so a user who knows `blut plan promote/rollback`
  already knows `blut model`._ `src/model_registry.rs`: `models` (IMMUTABLE, keyed
  by the checkpoint sha256 the cache/lineage already computed: name + tenant +
  source; `register` idempotent on same (hash,tenant,name), a cross-tenant OR
  rename collision refused fail-closed — a content hash = one model identity) +
  `model_pointers` `(tenant,name,alias)→hash` + append-only
  `model_pointer_history`. `promote` upserts the pointer + appends the move
  (IMMEDIATE txn); `rollback` pops to the previous target atomically (BEGIN
  IMMEDIATE, two-most-recent read + write in one snapshot — the exact 0085
  pattern). The `@alias` axis (`prod`/`staging`/`v7`) generalizes 0085's single
  pointer — each alias is an independent audited pointer under one model name.
  Clinical boundary ([ADR 0061](0061-blut-p2p-compute.md)/[0096](0096-blut-multi-tenancy.md)):
  a Restricted-tenant checkpoint can never be
  promoted onto another tenant's pointer; tenant strings validated by the frozen
  ADR-0096 `Tenant::parse` grammar; hashes validated lowercase-64-hex; names /
  aliases `[A-Za-z0-9_.-]` — nothing malformed enters a URI, an audit row, or the
  DB. CLI `blut model {register,promote,rollback,resolve,history}` mirrors `blut
  plan`, wired in `cli::run` (built-in). Gate part: `cargo test -p blut
  model_registry::` = 8 pass (hash/name/tenant validation, idempotent-register +
  collision refusals, promote/resolve/rollback roundtrip + full audit trail
  A→B→A, independent aliases, cross-tenant + wrong-name promote refusals,
  rollback-without-prior refused, garbled-tenant refused, URI grammar). Full suite
  green; clippy 0; fmt+scope clean. Model review: tenant-validation +
  promote-IMMEDIATE applied; residual nits (name re-guard in promote, rollback
  `changes()` check, multi-`@` error message) deferred to incr 2 as matching the
  frozen-0085 pattern.
  **DELIBERATELY NO Python** — this is the pure-engine registry mechanism.
  **REMAINING:** incr 2 = the governance fail-close on governed aliases (`@prod` →
  the project's gate with `--change-id`, so `@prod` can never point at an
  unpromoted checkpoint); then `dataset://<name>@<version>` +
  `experiment://<name>/<run>` registries (the latter reuses
  [0099](0099-blut-lineage-graph-model-cards.md) `run_diff`); and
  resolve-into-recipe-args wiring (`blut recipe run --model model://enc@prod`).
  ADR stays In progress._
- _2026-07-12 — INCREMENT 2 LANDED (engine main `797348f`): the governance
  FAIL-CLOSE on governed aliases._ Promotion to a GOVERNED alias (default `prod`,
  overridable via `$BLUT_MODEL_GOVERNED_ALIASES`) must clear a governance gate
  first, so `model://<name>@prod` can never point at an unvetted checkpoint.
  **Charter — the engine stays domain-AGNOSTIC:** it never names the domain gate
  script; it runs a CALLER-supplied gate command (the LamQuant cookbook / operator
  wires its regulatory gate via `--gate-cmd` / `$BLUT_MODEL_GATE_CMD`), exactly as
  [ADR-0112](0112-blut-ecosystem-integrations.md) connectors
  delegate a verb to an external tool. `src/model_registry.rs`:
  `GateVerdict{Pass|Fail(reason)|NotConfigured}` + `GateCmd{program,args}`
  (whitespace argv, NO shell) + `is_governed` + `DEFAULT_GOVERNED_ALIASES=["prod"]`.
  `run_gate` (impure, OUT of the DB path) substitutes `{hash}/{name}/{alias}/
  {change_id}` into the argv, runs with explicit program+args (no shell ⇒ the
  validated placeholders can't inject), maps exit-0 ⇒ Pass / non-zero ⇒ Fail /
  spawn-failure ⇒ Fail (fail-closed — a gate that can't run never Passes).
  `promote_governed` stays PURE + DB-only (DI: takes an already-computed verdict):
  a governed alias requires `Pass`; Fail/NotConfigured/None all refuse and never
  create the pointer; an ungoverned alias ignores the verdict = byte-identical to
  `promote`. CLI `blut model promote` gains `--change-id` + `--gate-cmd`. Gate:
  `cargo test -p blut model_registry::` = 11 pass (+3: governed-requires-passing-
  gate incl no-pointer-on-refuse, ungoverned-ignores-verdict, GateCmd parse +
  run_gate true/false/missing-binary). Full suite green; clippy 0; fmt+scope
  clean. Model review: FAIL-SAFE fix (blank `$BLUT_MODEL_GOVERNED_ALIASES` falls
  back to the default, never silently ungoverns @prod — the env can only WIDEN
  the set) + documented the GateCmd whitespace + run_gate no-timeout limits;
  placeholder-mangling was a verified FALSE POSITIVE (names/aliases are
  `[A-Za-z0-9_.-]`, no braces; change_id substitutes last; no shell).
  **REMAINING:** `dataset://` + `experiment://` (reuses 0099 `run_diff`)
  registries; resolve-into-recipe-args wiring; a bounded gate timeout + async
  run_gate for automated callers. ADR stays In progress._
- _2026-07-12 — M2.0 CONTROL REPAIR (meta status only): frontmatter corrected
  from `proposed` to `in-progress` so generated views match the Status section
  and the two landed model-registry increments. No completion claim changed:
  dataset/experiment registries, recipe-argument resolution, and bounded async
  governance remain open._
- _2026-07-12 — ACCEPTANCE-COMMAND CORRECTION (supersedes the stale CLI syntax
  in the original Implementation Plan, without rewriting it): the current
  model CLI is `register <hash> <name>`, `promote <hash>
  model://<name>@<alias>`, and `rollback model://<name>@<alias>`. The executable
  acceptance sequence is:_

  ```bash
  cargo test -p blut registry::
  export BLUT_REGISTRY_DB="$(mktemp -d)/registry.db"
  A=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  B=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  blut model register "$A" demo
  blut model register "$B" demo
  blut model promote "$A" model://demo@staging
  blut model promote "$B" model://demo@staging
  blut model rollback model://demo@staging
  test "$(blut model resolve model://demo@staging)" = "$A"
  if blut model promote "$B" model://demo@prod \
      --change-id gate-reject --gate-cmd false; then exit 1; fi
  ```

  _The `blut` executable is the cookbook-owned binary that delegates to
  `blut::cli::run`; the engine intentionally ships as a library, not a domain
  cookbook binary. The Cargo filter is required to select the four M2.2
  integration tests in addition to the model/registry unit tests; zero selected
  M2.2 tests is a failed gate._
- _2026-07-12 — M2.2 COMPLETE / FROZEN (engine main `53998a4`). Dataset
  storage-topology correction (superseding the Decision section's single
  lineage-DB/`refs` wording): model pointers share the existing ADR-0085 plan
  registry SQLite file, dataset bindings share the existing datasets SQLite
  file, and experiments remain lineage-only views. No daemon or additional
  store was introduced. Dataset
  bindings now pin `dataset://<name>@<version>` to one immutable content
  identity and exact tenant, re-hash live bytes on every resolve/resume, reject
  malformed classification metadata, and keep Restricted data node-local.
  Experiment grouping is an explicit immutable job marker backed only by
  lineage; comparisons report input hashes, JSON-leaf argument deltas, and
  headline metrics under exact tenant scope. Generic lineage diffs reject
  cross-tenant and Restricted exports; an explicit experiment tenant may
  compare its Restricted runs locally. Recursive URI resolution now precedes
  typed compilation for recipe, dry-run, sweep, HPO choices, declarative TOML,
  JSON/frozen PlanSpecs including nested expansions, and Starlark build/stage
  args. Resume retains both source handles and the resolved launch snapshot:
  datasets are re-hashed, while a moved mutable model alias is refused rather
  than silently changing the original job. Governed model gates are async,
  default to five minutes, capture at most 1 MiB per output stream, and kill the
  process group on timeout/output overflow. Model CLI defaults now match recipe
  tenant `default`; pre-change records remain addressable with `--tenant
  shared`._

  _Gate evidence: the corrected CLI sequence above passed through an external
  cookbook-style binary (`blut::cli::run`): staging A→B, rollback→A, governed
  false-gate promotion nonzero. `cargo test -p blut registry::` selected 19
  tests total including all four M2.2 integration tests; the final all-target,
  all-feature run's library target passed 938 tests with one ignored; every
  integration target, framework bench target, and cookbook example also
  passed. Strict all-feature
  clippy, fmt, wasm `blut-types`, and scope guard passed; the final human diff
  renderer test passed separately. Independent Standards and Spec reviews both
  returned PASS. Mandatory commit review returned PASS WITH NITS and recommended
  ship-as-is. Critic claims about `COALESCE(runs.value, excluded.value)` losing
  a NULL backfill and child PID not equaling process-group ID were verified
  false (`COALESCE(NULL, incoming)` selects incoming; `setsid` guarantees
  PGID=PID). The same-hash/different-dataset-name refusal is deliberate one-
  content-identity behavior; only its concurrent-conflict wording remains a
  cosmetic nit._

## Completion / Resolution  *(fill once)*

- **verdict:** `passed`
- **gate_cmd:** corrected model CLI sequence in the Progress Log; `cargo test
  -p blut registry::`; `cargo test -p blut --all-targets --all-features`;
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`;
  `cargo fmt --all -- --check`; `cargo check -p blut-types --target
  wasm32-unknown-unknown`; `bash scripts/scope_guard.sh`
- **result:** all acceptance behavior and repository gates passed; named Cargo
  filter selected the four M2.2 tests; external CLI rollback and governed
  refusal both matched the contract.
- **git_sha:** engine `53998a429352c8f5eedc1634b9adffcb05740cfa`  ·  **artifact sha256:** n/a (source-only change)

## Related Decisions

ADR 0070 [not federated] (markdown experiment ledger — prior art this promotes
to a machine surface), [0078](0078-blut-starlark-typed-dynamic-dag.md)
(content-addressed cache keys — the hashes this names),
[0083](0083-blut-interface-stack.md) (interface stack — a `blut-web` sidecar
surfaces the registry), [0085](0085-blut-deployment-registry.md),
[0099](0099-blut-lineage-graph-model-cards.md),
[0100](0100-blut-data-catalog.md) (data-catalog — shares the `dataset://`
resolver).

## Validation  *(ongoing — distinct from the completion gate)*

Wrong if: an `@prod` alias ever resolves to a checkpoint that never passed
the governance gate; the `refs` table is mutated in place (rollback would lose
history); a `dataset://` clinical handle resolves inside a cloud stage; or the
registry drifts from the lineage DB (two sources of truth for "which artifact
is prod").
