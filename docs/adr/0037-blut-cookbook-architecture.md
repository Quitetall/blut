<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0037 — BLUT cookbook architecture — domain-agnostic orchestrator + loadable cookbooks

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `in-progress` (Accepted 2026-06-07 · In progress 2026-08-27)

(Extends [ADR 0034](0034-blut-scope-charter.md) "BLUT scope charter". Does not supersede it — 0034's
own-the-nouns / delegate-the-verbs split still holds; this ADR adds the
cookbook decomposition + the steal map for the world-class build.)

## Context

BLUT today is a Rust orchestrator with LamQuant baked into its core: `src/`
holds 6 `lamquant_*` recipes + ~20 `lamquant_*` stages + ~15 LamQuant artifact
types + a `LamquantBackend`, registered in a compile-time-static `RECIPES`
array. The framework traits (`Stage`/`Artifact`/`Recipe`/`Plan`/`Executor`) are
already generic; the coupling is isolated to recipe-composition + the static
registry + a `lib.rs` export. Verified 2026-06-07: the seam for extracting
LamQuant into a loadable pack is clean (subprocess + JSON args + bincode
artifact handoff; the Python tree `[redacted: private path]` is already a natural
payload).

Owner vision (2026-06-07): **BLUT = a general orchestrator tool; a COOKBOOK =
a loadable domain pack (training recipes + data-preprocess ops + app-specific
scripts/functions); choosing a cookbook unlocks all its capabilities.** The
cookbook is the edit surface; BLUT core is not. Hierarchy: BLUT (tool) ▷
cookbook (collection/preset) ▷ recipe (one capability).

Two load-bearing requirements drove the design: (1) **bus-factor** — a second
person must launch a real codec training run from BLUT, watch it on wandb, and
stream metrics to a reviewer, with zero AI-in-a-shell; (2) **don't reinvent** —
mature tools solve parts of this; the project must steal from them rather than
hand-roll. The owner's preference is everything-in-Rust where reasonable, with
other languages only where they clearly pay.

## Decision

1. **BLUT core becomes domain-agnostic.** A `Cookbook` trait registers a pack's
   recipes/stages/artifacts/backend/paths; the runtime registry populates from
   the selected cookbook (`blut --cookbook <name>`). LamQuant is extracted into
   the first cookbook (`cookbooks/lamquant/`): its recipes + stages + artifacts
   + backend + the Python payload. BLUT-core ends with **zero `lamquant`
   symbols**.
2. **Steal formats + concepts, not runtimes.** Adopt as dependencies only the
   things that are light + Rust-native + fill a missing layer:
   - **Apache Arrow** (arrow-rs / Parquet / Arrow-IPC) for data/metrics/
     lineage/manifests/run-DB + cross-process handoff. NOT for the domain codec
     (its file formats stay bit-exact domain-specific).
   - **Lerna** (`nbprint/lerna`, Apache-2.0) as the Hydra-compatible config
     resolver — **its Rust `rust/` crate, not the Python package** (see
     Rationale: spike verdict). Vendor-pinned.
   Steal as *concepts* (thin Rust on the existing framework), reject the
   runtimes:
   - **Dagster** — code-location ≡ cookbook, asset-check ≡ PCCP gate, asset
     lineage. Reject the daemon/DB/web server (Python-only, can't span
     firmware).
   - **Temporal** — durable/resumable workflow, retries, signals. Built on the
     existing content-addressed cache + `RunManifest`. Reject the Temporal
     server.
   - **Ray** — DEFER to a rented cluster. Design the seams now (a `Launcher`
     trait `{LocalSystemd | Slurm | RayJobSubmit}` + import-callable Python
     `run(cfg)` entrypoints so Ray Train/Tune can wrap them later) and implement
     when multi-node is real. On one GPU Ray buys ~nothing.
3. **Config is Hydra-syntax, Rust-resolved, single-resolver.** BLUT ingests
   Hydra-style YAML (config groups = cookbook presets; override grammar;
   multirun sweeps) via Lerna's Rust crate, resolves to a FROZEN flat config,
   and hands that to config-dumb Python stages. BLUT is the SOLE resolver →
   zero parser drift. The sweep engine is Rust: parse → cartesian product →
   content-hash each config (⊕ code_sha ⊕ data_sha) → skip cache-hit (finished)
   runs → dispatch the remainder via the `Launcher` — a memory-safe daemon doing
   cache-aware parallel grid search, not a Python multirun loop.
4. **Split of ownership:** the AI builds the plumbing (Cookbook trait + registry,
   LamQuant extraction, Arrow metric/lineage layer, Lerna config + sweep engine,
   bus-factor recipe + systemd wiring, robustness/CI). The owner drives the
   TUI/UX (cookbook-select screen, recipe arg-forms, live run dashboard,
   artifact/lineage browser).

## Rationale

- **The seam is already clean** (verified): generic framework traits; LamQuant
  isolated to recipes/stages/artifacts/backend + the static registry. Extraction
  is a move + a registry indirection, not a framework rewrite.
- **Lerna spike verdict (2026-06-07, VERIFIED):** `nbprint/lerna` `rust/` is a
  pure **rlib** crate (`crate-type=["rlib"]`, name `lerna`) exposing
  `pub mod config, config_store, omegaconf, parser, interpolation, defaults,
  defaults_list, merge, search_path, sweep, sweeper, launcher, job, job_runner,
  package, glob, env, callback, core`. The PyO3 cdylib (`lerna_py`,
  `rust/python/lib.rs`) is a SEPARATE wrapper depending on that rlib. So BLUT
  can depend on the rlib directly — native Hydra resolution + sweep in Rust, no
  Python in the path, and the *same engine* is dual-bindable to Python (zero
  drift) if ever needed. This turns "reimplement Hydra in Rust" into "adopt a
  vendored rlib" — and validates the cookbook=config-group mental model (Lerna's
  `config_store`/`defaults_list` are exactly that).
- **Steal-formats-not-runtimes** keeps it from becoming a franken-stack: Arrow +
  Lerna are libs/formats (cheap, Rust-native); Dagster/Temporal/Ray are heavy
  runtimes whose *concepts* fit but whose *infra* (daemons, DBs, K8s,
  Python-only) does not span the project's ML+embedded+regulatory surface.
- **BLUT's moat is real** and worth keeping bespoke: Rust + unified ML *and*
  firmware/embedded orchestration (cross-compile, probe-rs, Verilator) +
  single-workstation systemd containment + fail-closed ship gates
  `[redacted: non-engine product detail]` + the Rust `lml`
  codec. No Python orchestrator (Dagster/Metaflow/Flyte) spans this.
- **Single-resolver** (BLUT resolves → frozen config → dumb Python) is what
  makes Hydra-syntax safe: two interpreters (Rust BLUT + real Python Hydra) on
  the same YAML would drift. One owner kills that.
- **Everything-in-Rust core** = one binary, one edit surface (the cookbook),
  the owner's stated preference; achievable because the one layer that tempted
  Python (config) is covered by Lerna's rlib.

## Alternatives Considered

- **Hydra (Python) directly** — REJECTED as a dependency: not an orchestrator
  (no DAG/cache/gate/containment), Python in the config path = a second edit
  surface fighting "cookbook is THE edit surface". Its config-group concept is
  adopted via Lerna's Rust port instead.
- **Hand-roll a Hydra parser in Rust** — REJECTED: the Lerna rlib already does
  it (~100% Hydra compat, 2854 tests). Adopt-vendored, don't reimplement.
- **Dagster as the orchestrator** — REJECTED: closest existing match (code
  locations ≡ cookbooks, asset checks ≡ gates) but Python-only (no firmware),
  daemon+DB ops model wrong for a single-workstation Rust+embedded shop. Steal
  the concepts.
- **Ray now** — REJECTED (deferred): on one workstation GPU it buys ~nothing; heavy dep,
  zero current payoff. Seams designed now, implemented at cluster-rent.
- **Declarative-only / runtime cookbook (TOML + dyn stage registry)** —
  REJECTED as the primary: loses the compile-time typed Plan DAG. Hybrid kept:
  typed Rust stages for heavy work + a declarative TOML layer for simple
  preprocess/script recipes (the "drop in scripts/functions" part of the
  vision).

## Consequences

- New deps: arrow-rs (+ Parquet), a vendor-pinned `lerna` rlib (Apache-2.0;
  22-star/1-maintainer ⇒ own a frozen in-tree copy, do not float).
- Migration phases (each fail-closed; LamQuant cookbook stays byte-runnable
  throughout): C0 Arrow adoption · C1 Cookbook trait+registry · C2 extract
  LamQuant cookbook · C3 declarative recipe layer · C4 Lerna config + sweep
  engine · C5 durable-resume + lineage · C6 (deferred) cluster launchers + Ray.
- Bus-factor is unblocked first by a minimal standalone codec training recipe +
  wandb in the trainer + Arrow metric-stream + `run_contained`/systemd wiring,
  in the CURRENT architecture — it does NOT wait for the full cookbook refactor
  (which runs as a parallel track and must not gate the product critical
  path `[redacted: hardware]`).
- The TUI/UX is owner-driven; this ADR fixes the plumbing contracts the UI binds
  to (cookbook selection, recipe arg-schema, run/metric/lineage artifacts).
- Anti-reinvention: this ADR is the record of WHY Hydra/Dagster/Temporal/Ray
  runtimes were rejected — do not re-litigate without a superseding ADR.

## Progress Log *(append-only; newest last)*

- _2026-08-27_ — **In progress — 2026-08-27** under ADR 0186 [not federated] clause 8. Half the Decision landed, half was never taken up. BLUT core did become domain-agnostic behind a Cookbook trait -- `src/framework/cookbook.rs:88` defines it (plus CookbookTui at :34) -- and LamQuant is extracted into its own cookbook `[redacted: private path]`. The 'steal formats' half did not: neither Apache Arrow, Parquet, Arrow-IPC nor the Lerna Rust crate appears as a dependency in `Cargo.toml` or any crate manifest under `crates/`, so the run-DB/lineage/manifest layer this ADR planned to build on Arrow is still on the pre-existing format.

- _2026-08-27_ — gave this record a runnable citation gate `[redacted: private path]` (it proves the record still points at files that exist; it does not claim the decision is implemented) and removed its closure-debt entry `[redacted: private path]`.
