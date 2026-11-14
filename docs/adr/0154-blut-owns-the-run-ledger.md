<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0154 — BLUT owns an append-only run ledger; a run is a noun, a dashboard is a verb

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `in-progress` (accepted 2026-08-02; in progress 2026-08-27)

## Status

`Proposed 2026-08-02` → `Accepted 2026-08-02`

`In progress — 2026-08-27`

## Context

Training-run provenance is recorded in **eleven** independent places and joined in
none of them. Measured on 2026-08-02:

| surface | state |
|---|---|
| `outputs/experiment_log.jsonl` | 2021 rows, **1894 (94%) pytest runs**, last real write **2026-05-28** |
| BLUT per-job directories `[redacted: private path]` | live — 132 job dirs, per-stage `output.metadata.json` receipts |
| `lineage.db` | live — 167 runs, 247 artifacts, 44,110 metrics, 76,185 gauges |
| `MetricLog` CSV/Parquet | live, read via `read_metric --run` |
| `RUN_MANIFEST.json` | code exists; **0 files on disk** at the documented path |
| checkpoint `provenance` dict + `resume_key` | live |
| a hand-maintained results ledger `[redacted: private path]` | live, hand-maintained |
| regulatory change-control records `[redacted: non-engine product detail]` | live |
| signed training attestations (ADR 0144 [not federated]) | live, fail-closed |
| an evidence graph `[redacted: non-engine product detail]` | live but per-value, not per-run |
| wandb | live, per-step curves |

The data collection is not the problem. Two things are missing:

1. **A join key.** Nothing ties a BLUT `job_id` to a trainer `run_id` to a
   results-ledger row to a change-control id to a checkpoint SHA. Each surface is
   individually sound and mutually unreachable.

2. **A classification.** No field anywhere distinguishes a smoke test from a
   campaign. The only way to tell one from the other in `experiment_log.jsonl` is to
   grep `encoder_ckpt` for `/tmp/` — which is how 1894 pytest rows came to sit in the
   production experiment record indistinguishable from real science.

Two further facts about that file: its two test modules are import-errors
(`from ai_models import …` against a path that has not existed since the repo split),
so its append-only contract is currently **untested**; and its writer now resolves to
a file inside the cookbook tree [redacted: private path] (4 records)
while every reader still points at the 2.6 MB meta-root file. Writer and readers have
been looking at different files since the module moved.

Separately, the project's results ledger "+PP+" states *"a number with no provenance is a rumor,
not an entry"*, yet nothing machine-validates that a cited run id exists anywhere.
Two of its rows cite throwaway scripts "+PP+" — provenance that cannot be recovered.

### The conflict this ADR must resolve

[ADR 0034](0034-blut-scope-charter.md) (line 97) and a scope-guard document [redacted: private path] both carry:

> `| experiment record | outputs/experiment_log.jsonl + wandb | append to it |`

That row delegates the experiment record **away from BLUT**, to the file described
above. It cannot stand alongside a BLUT-owned ledger.

Worth stating plainly: ADR 0034 is still `status: proposed` — it was never accepted,
though three area audits were run against it and the scope-guard document treats it as
live. This ADR therefore amends an unratified proposal, not a ratified charter.

## Decision

The BLUT **engine core** owns an append-only run ledger at
`paths::data_dir()/run-ledger.jsonl`. The engine owns the writer; cookbooks call it
and never implement their own. Every run is written once with a `tier` — `scratch`,
`recorded`, or `canonical` — and a stable `run_uid` that binds the existing surfaces.
Records are never moved between tiers or files: promotion **appends** a promotion
record. `scratch` records are clearable; clearing leaves a tombstone so a citation
never dangles. LamQuant ingests the ledger and renders it into its composed doc tree
as a training log [redacted: private path].

This replaces **one row** of ADR 0034's DELEGATE table — the `experiment record` row —
and nothing else. Every other delegation in 0034 stands: process launch to
`systemd-run`, liveness to systemd cgroups, log tailing to journald, and in
particular **metric curves, training-dynamics plots and run comparison remain
wandb's**. ADR 0034 as a whole is not superseded and its status is unchanged.

## Rationale

- **A run is a BLUT noun by 0034's own test.** The charter's rule is "own the nouns,
  delegate the verbs". BLUT already assigns `job_id`, and already records `runs`,
  `artifacts`, `lineage_edges`, `git_sha` and hardware in `lineage.db`. It is not
  acquiring a new responsibility; it is being allowed to write down what it already
  knows. What 0034 was protecting against is BLUT reimplementing dashboards — a run
  *ledger* is the noun, a run *dashboard* is the verb, and the verb stays delegated.

- **The delegate target is not viable.** Delegating the experiment record to a file
  that is dead, 94% test noise, untested, and whose writer and readers disagree on
  which file they mean is a delegation in name only. The row describes an arrangement
  that has not functioned since May.

- **Only the launcher knows intent.** Whether a run is a smoke test cannot be
  recovered downstream — it is a property of why the run was started. It has to be
  recorded at launch, and BLUT is what launches.

- **Duration is the honest default bar.** A run that occupied the machine for four
  hours is expensive regardless of outcome, and expense is measurable where
  significance is not. Per-recipe bars are better and are encouraged by a
  recommendation, but a missing bar must not silently drop a run.

- **Outcome is a field, not a filter.** A 300-epoch divergence is knowledge that
  stops someone re-running it. Filtering failures out of the record is how the same
  dead end gets explored twice.

## Alternatives Considered

- **Extend `experiment_log.jsonl` in place.** Rejected: it would inherit 1894 pytest
  rows, a per-process-only lock whose cross-process safety rests on unstated
  `O_APPEND` atomicity, a duplicate-`run_id` history (2021 lines / 454 ids), and two
  import-error test files. It also leaves the writer/reader path split unfixed.

- **Put the ledger in `lineage.db`.** Rejected on the DB's own stated contract —
  `lineage_db.rs:3-25`: *"the DB is a rebuildable INDEX, never the source of truth."*
  A canonical record that exists only in SQLite violates the invariant that a deleted
  `lineage.db` is fully recoverable by `blut lineage reindex`.

- **Let the LamQuant cookbook own the writer.** Rejected by the owner: diagnostics
  are core to the engine, and log work is an extension, not a cookbook. A
  cookbook-owned writer would also mean every future cookbook reimplements it.

- **A single tier with a `is_smoke` boolean.** Rejected: it collapses "not worth
  keeping" and "kept but not promoted" into one flag, and gives no place for the
  explicit-promotion decision the owner asked for.

- **Auto-promote to canonical on threshold.** Rejected: canonical is a claim about
  significance, and significance is a judgement. Thresholds decide *recording*;
  people decide *canonical*.

## Consequences

- ADR 0034's `experiment record` row becomes historical. The scope-guard document [redacted: private path]
  must carry the same amendment or the guard doc and the charter will disagree.
- `outputs/experiment_log.jsonl` is sequestered to `legacy/` — readable forever,
  never appended to again. Its 454 unique real `run_id`s are imported as historical
  `recorded` rows so the history survives the retirement.
- Experiment-watching scripts [redacted: private path] and the `list_experiments`/`best_by`/`compare`
  helpers point at the retired file and will need repointing or retiring with it.
- BLUT gains a durable artifact it must not lose: the ledger becomes a thing backups
  and `gc` have to respect, and `gc` must be incapable of touching a non-`scratch`
  row.
- Once every results-ledger citation is required to resolve, the two rows citing
  throwaway scripts will fail — correctly. That is a pre-existing provenance hole this
  makes visible, not a regression introduced here.
- A canonical log makes two standing change-control facts permanently visible
  [redacted: codec research]. Both are true today and merely unlit.
- The evidence graph's `Action` and `Outcome` classes are reserved and unused (ADR 0114
  [not federated]). The ledger is the substrate they were reserved for; the record schema must
  let an `Outcome` node cite a `run_uid` later without a schema break.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** `src/run_ledger.rs` writing an append-only
  `run-ledger.jsonl` of `run_started` / `run_ended` / `promoted` / `cleared` records,
  a `blut ledger {list,show,promote,gc,verify}` CLI, a `blut_core` shim so Python
  stages call the engine writer, and a training log composed by a meta-repository
  tool [redacted: private path].
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut --lib run_ledger && <training-log check [redacted: private path]> && blut ledger verify --require-ledger-rows-resolve`
  - `pass when:` unit suite exits 0; the training-log check exits 0 against a
    freshly built tree and exits 2 when stale; `ledger verify` resolves every
    results-ledger "Run id / source" citation to a ledger record, or names the
    rows that fail with the reason.
- **Rollback:** revert the `run_ledger` module and the training-log
  composition block [redacted: private path]; restore `outputs/experiment_log.jsonl` from `legacy/` and
  reinstate the 0034 row. The ledger file itself is append-only data and is retained
  either way — a rollback of the code does not destroy records already written.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-08-27_ — **In progress — 2026-08-27** under ADR 0186 [not federated] clause 8. The ledger itself is built: src/run_ledger.rs is a real module (declared at lib.rs:77) and resolves the canonical path paths::data_dir()/run-ledger.jsonl exactly as the Decision specifies (:318-321). What is not demonstrable is the gate. It runs cargo test -p blut from the engine directory, then invokes the training-log tool, which lives at the meta-repository root [redacted: private path] -- so the command dies before the ledger assertions are reached. A path defect in the instrument, not a missing deliverable; repairing it is grouped with the other gate repairs.

- _2026-08-02 — opened._ Counts in Context measured this date: `lineage.db`
  167 runs / 44,110 metrics / 76,185 gauges; 132 job dirs; `experiment_log.jsonl`
  2021 rows / 454 unique run ids / 1894 with `/tmp/` checkpoint paths; last real
  write 2026-05-28. ADR number 0152 chosen over the free 0145–0149 because
  a meta-repository document [redacted: private path] already cites a nonexistent "ADR 0146" — reusing a
  cited-but-missing number would resolve that dangling reference to the wrong
  document.
- _2026-08-02 — accepted by the owner._ Implementation of §B (the engine writer)
  begins against this contract.

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** _pending_
- **gate_cmd:** _see Implementation Plan_
- **result:** _pending_
- **git_sha:** _pending_  ·  **artifact sha256:** _pending_

## Related Decisions

- **ADR 0034** (BLUT scope charter) — amended in one row only; see Decision.
- **ADR 0038 [not federated]** — verify numbers against raw sources, never a prose digest. The
  ledger is a raw source; it must never be summarized into the record it is checked
  against.
- **ADR 0090** (`experiment://` URIs) — the ledger supplies the runs those URIs
  resolve; deliberately not a second store.
- **ADR 0114 / 0115 [not federated]** — reserved `Action`/`Outcome`; this is
  their substrate.
- **ADR 0144 [not federated]** (training acceptance attestations) — the most rigorous per-run
  artifact; the ledger references it by `attestation_ref` rather than duplicating it.

- _2026-08-02 — renumbered 0152 -> 0154._ The number collided. `main` carries
  its own ADR 0152 [not federated], added
  in a meta-repository commit [redacted: private repository] — one of 43 commits on `main` that this branch does not have —
  so both files would have landed as ADR 0152 on merge. 0154 and 0155 are free
  across every ref in the repository, checked with
  `git log --all --diff-filter=A` over the decision log. The title line
  and every ADR-sense reference moved with the file; the opening entry above is
  left exactly as written, including its now-superseded reasoning for choosing
  0152, because that is what the append-only rule is for.

  Two references were deliberately NOT rewritten in this commit:
  `src/{run_ledger.rs,cli/ledger.rs,cli/mod.rs}` still say
  "ADR 0152" in comments. They live in a submodule, and touching them costs an
  engine commit plus a gitlink bump, which re-opens the cross-repo pin cascade
  for three comment strings. They are folded into the engine's next commit
  instead. Nothing functional reads the number.
