<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0156 — BLUT experiment-execution UX (advisory stages, admission auto-tune, result retrieval)

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `accepted` (proposed 2026-06-30; accepted 2026-08-27).

> **Numbering note.** Engine source written before the renumbering cited this record as
> **ADR 0071**, its original number; it was renumbered (via 0093) to 0156 after a
> collision, and those citations now read ADR 0156.
>
> **Recovery provenance (2026-07-10):** restored verbatim from a meta-repository commit
> [redacted: private repository] after the original ADR 0071 number was reused by another decision (ADR 0071 [not federated]).
> Only the ADR number, canonical metadata, the stale cloud-queue reference, and
> this recovery entry changed.

- **Status:** Proposed 2026-06-30
- **Status:** Accepted 2026-08-27
- **Extends:** ADR 0034 (BLUT scope charter — train↔deploy parity is the real gap),
  ADR 0046 (broker live), ADR 0047 (never-OOM-the-RUN), and the plan/executor model.
- **Driven by:** running the real codec ablation grid (the ADR 0070 [not federated] program) — 8 concrete
  frictions, three of which (Tier 1) cost the most time + confusion.

## Context

The codec program is **ablation-driven**: dozens of train→evaluate runs sweeping schedule,
capacity, and data. That program lives or dies on **fast, legible, contained experiment
execution + clean result retrieval.** Running it surfaced frictions where BLUT's behaviour
actively *cost* time:

- **Success was mislabeled as failure.** Every codec ablation run [redacted: codec research] reported
  `exit=1 / FAILURE` though training fully succeeded — only a downstream PCCP dry-run gate
  failed (on a stale path). A stage failure maps 1:1 to a plan failure
  (`NodeFailure::Stage → PlanError::StageFailed → exit 1`, `executor.rs:1487` / `cli.rs:3527`);
  `PlanResult` is success-only with no per-stage status, so the live run surfaced no ckpt or
  metrics — the operator had to dig through lineage to confirm a *good* experiment.
- **Admission hard-refused instead of fitting.** A joint run was refused ("40.6G would leave
  <6G free") at the worker count it was launched with; manually lowering it fit instantly.
  Every contained train stage runs fixed at `UNCALIBRATED_WORKER_CAP = 2`
  (`footprint.rs:178` + `helpers.rs:273`) — the broker neither lowers it to fit nor raises it
  to saturate.
- **Decode was GPU-starved by default.** That same cap left the GPU at ~0–10% (decode-bound):
  bps 0.3 → 45 (an ~80× speedup; a 26 h estimate → ~15 min) once workers were raised by hand.

The narrow fixes are mechanical, but two of them **change load-bearing semantics** — what
"plan failure" means, and how the never-OOM broker decides — so they need a decision record.
(Ground-truth also corrected two premises: the GPU-starved detector already exists
(`gpu_sampler.rs`, <25% util) and env auto-detection already exists; those asks shrink to
surfacing + 3 stale hardcoded paths.)

## Decision

Adopt four changes to BLUT's experiment-execution surface. Each is **additive / opt-in**, so
the default behaviour for any stage that does not opt in is unchanged.

1. **Advisory stage outcome.** A stage may declare itself *advisory* (`StageDyn::is_advisory()`,
   default `false`). An advisory stage's failure is recorded as a **non-fatal** outcome — a new
   sibling to `NodeFailure::Killed` ("NOT a plan failure", `executor.rs:463-468`): it WARNS,
   the plan continues/completes, and the process exits 0. `PlanResult` gains a per-stage
   `warnings` list and **preserves `final_output`** (the upstream train ckpt). A plain
   `train → gate` chain therefore reports `train: OK · gate: skipped (advisory): <reason>` and
   surfaces the **ckpt path + headline R** live, instead of `plan execution failed`. A
   *dry-run / verdict* gate is the canonical advisory stage; a **real promotion gate stays
   fatal** (advisory is opt-in, not the default).

   Two contracts pin the semantics (resolving the review):
   - **Descendant data contract** — an advisory failure **prunes its descendants** (they do
     NOT run), exactly as `Killed` prunes a killed node's subtree. So a fatal stage can never
     execute on the missing/garbage output of a failed advisory upstream. (In the canonical
     `train → advisory-gate` chain the gate is terminal, so nothing is pruned and the train
     output is `final_output`.)
   - **Machine-parseable signal, not just the exit code** — clean success and
     success-with-advisory-warnings both exit 0, so they are distinguished by
     `PlanResult.warnings` (non-empty ⇒ an advisory tripped), surfaced in `--json`. A
     `--strict-advisory` flag re-promotes advisory failures to fatal for CI that wants the old
     fail-hard behaviour. Tooling MUST check `warnings` (not exit 0 alone) before acting on a
     surfaced ckpt — e.g. a promoter never promotes when `warnings` is non-empty.

2. **Admission auto-tune (fit-AND-saturate).** The broker auto-tunes the decode worker count
   from `(cpu_count, available-RAM headroom)` rather than running fixed at the cap. A single
   shared function `workers_to_fit_and_saturate(cpu_count, avail, floor, drivers)` (mirroring
   `warm_workers_for_budget`, `footprint.rs:618-628`) starts from a throughput target
   (~`cpu_count`) and steps **down** until `estimate_ram_bytes(W, …)` fits `avail − floor`. At
   admission, instead of hard-refusing, the broker **admits with a note**
   (`admitted: workers N→W to fit Xg`). Precise contract (resolving the review):
   - **Output range:** `W ∈ [1, cpu_count]` and `estimate_ram_bytes(W) ≤ avail − floor`,
     maximising `W`. **If even `W=1` does not fit, REFUSE** (the existing box-capacity error) —
     there is **no fall-back to a fixed cap that would OOM**. So "lower to fit *before* raise to
     saturate" with a hard refuse floor; never-OOM-the-box (ADR 0047) is preserved.
   - **Parity via a single snapshot:** the inputs (`avail`, `cpu_count`, `drivers`) are
     **snapshotted ONCE at admission** and the chosen `W` is **cached with the reservation**,
     then reused verbatim on the launch side — so transient memory pressure between resolve and
     launch cannot yield two different `W` and miss the `FootprintKey`. The shared function
     (`footprint.rs:44-51` parity contract) computes `W` from that one snapshot, not a fresh
     syscall each call.

3. **First-class result retrieval + contained measurement.** `blut results <job>` exposes the
   existing SQLite metric store as `{best_val_r, trajectory, per_band_prd, ckpt_path}` (new
   `series` + `best_metric` getters + an artifacts join; the store keeps only the `step=-1`
   *final* today, which can collapse after the peak). A `lamquant_measure` recipe runs an
   arbitrary measurement script under the **same contained env + venv training uses**
   (`<ai_models_root>/.venv`, resolved by the same plumbing). **Decisions:** the metric store
   stays a *derived, rebuildable index* (never the source of truth — that is `status.jsonl` +
   the sidecars); and ad-hoc measurements **reuse the training env**, never a hand-picked venv.
   Because the store is folded at job ingest, `blut results` on a job whose metrics have not
   yet been flushed returns an explicit **"no data yet"**, never an empty result that reads as
   success (resolving the review).

4. **Additive follow-ons (no semantic change):** an opt-in staleness *action*
   (`BLUT_AUTO_REBUILD=1` → rebuild + re-exec the cookbook crate, default off), and the
   `decoder_init` / `encoder_lr_scale` recipe knobs that unblock the schedule ablations.

## Rationale

- **An advisory gate failing a good experiment is a category error.** A dry-run verdict is
  *information*, not a build gate; the gate's own doc already claims "never short-circuits the
  plan." Making the outcome non-fatal + legible aligns behaviour with intent and stops
  contributors mis-reading success as failure. Opt-in (not default-non-fatal) keeps real gates
  protective.
- **Auto-tune beats refuse OR a fixed cap.** The footprint is already modeled as a function of
  workers; the *only* place worker count adapts to RAM today is the warm stage
  (`warm_workers_for_budget`). Generalising that one precedent fixes *both* the refuse (lower
  to fit) and the starvation (raise to saturate) with one knob — and never trades away
  never-OOM, which is the broker's whole reason to exist.
- **The metric store already exists; it's just not exposed.** `blut results` is queries over
  existing tables, not new infrastructure — the cheapest possible fix for the
  "grep BLUT_METRIC + `ls -t` a stale CSV" friction.
- **Measurements must inherit the training env.** The "3 venv misses" were the resolved venv
  diverging from the one with the full stack; reusing the training invocation plumbing makes
  the env+containment free.

## Alternatives Considered

- **A `--continue-on-error` plan flag instead of advisory stages.** Rejected: too blunt — it
  would swallow *real* failures too. Advisory is a property of the *stage* (a verdict gate),
  not of the run.
- **Bigger box / let the user always set `LMA_NUM_WORKERS`.** Rejected: that is exactly the
  manual guesswork the broker exists to remove; the box is fixed at 62 G.
- **A new metrics store / dashboard.** Rejected: the SQLite lineage DB already folds every
  `BLUT_METRIC`; a query + a CLI command is sufficient.

## Consequences

- The plan failure → exit-code contract gains a third state (completed-with-advisory-warnings).
  Any tooling that treats non-zero exit as "training failed" must learn the new exit-0 success.
- The broker's admitted worker count becomes data-dependent (cpu_count, RAM) — the
  `admitted: N→W` note is the audit trail; calibration keys must use the shared function.
- `is_advisory()` defaulting false means the change is invisible until a stage opts in (the
  PCCP gate, wired separately).

## Implementation Plan  *(the contract)*

- **Deliverable:** the engine changes (advisory outcome + `PlanResult.warnings`; the auto-tune
  fn + admission seam; `blut results`; opt-in auto-rebuild) and the cookbook changes (consume
  the auto-tuned workers; `lamquant_measure`; stale-path fixes; the two recipe knobs), each its
  own commit + review.
- **Acceptance gate (fail-CLOSED):**
  - `gate_cmd:` `cargo test -p blut` + `cargo clippy -- -D warnings` (engine) + `pytest`
    (cookbook) + the three doc gates.
  - `pass when:` (a) a `train → failing-advisory-gate` plan exits **0**, surfaces the ckpt, and
    its `warnings` is non-empty; (b) a fatal stage downstream of a *failed* advisory stage is
    **pruned, not run** (descendant contract); (c) a recipe that would refuse at `W=cpu_count` is
    **admitted** with `N→W` and `estimate(W) ≤ avail − floor`; (d) **never-OOM regression** — a
    recipe whose single-worker footprint exceeds `avail − floor` is still **refused** with the
    existing error, NOT admitted; (e) `blut results <job>` matches lineage + the run CSV, and an
    un-flushed job returns "no data yet".
- **Rollback:** each behaviour is opt-in / default-off — revert the specific commit; no wire
  format or stored-artifact change.

## Progress Log  *(append-only)*

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Reviewed under ADR 0186 clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. It declares a runnable gate_cmd, which a decision-log gate runner [redacted: private path] executed on 2026-08-27 with verdict TIMEOUT. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.

- _2026-06-30 — opened._ Scoped from 8 field frictions; PCCP gate path + advisory-wiring +
  a cookbook config file [redacted: private path] are handled separately by the project owner. Engine work branches
  off the engine trunk (independent of the unmerged `feat/cloud-queue`); cookbook off its trunk.
- _2026-06-30 — review refinements (MiMo PASS WITH NITS)._ Pinned three contracts the reviewer
  flagged as under-specified: (1) advisory failure **prunes descendants** so no fatal stage runs
  on bad data; (2) success-with-warnings is distinguished by `PlanResult.warnings` (+ a
  `--strict-advisory` CI flag), not exit code alone; (3) auto-tune **refuses if even `W=1`
  won't fit** (no OOM-cap fallback) and snapshots `avail`/caches `W` **once at admission** for
  resolve↔launch parity. Acceptance gate gained the never-OOM-still-refuses + advisory-doesn't-
  promote + `blut results` "no data yet" cases.
- _2026-07-10 — recovered as ADR 0093._ The source record from a meta-repository commit [redacted: private repository] had
  been stranded after ADR 0071 was assigned to BCS. ADR 0072 now points here;
  the cloud-queue relationship below now points to its canonical ADR 0082.
- _2026-08-04 — renumbered 0093 -> 0156 before first publication._ The number
  0093 had been taken on `main` by "blut-web read-only dashboard v1" (frozen at
  a meta-repository commit [redacted: private repository]) while this decision sat uncommitted in a side worktree. The entry
  above is left as written; it records what was true when written. This is the
  fifth ADR-number collision in this repository (0058, 0071, 0093, 0145, 0146),
  and a decision-log lint [redacted: private path] — landing in this same commit — is the check that
  makes the sixth impossible.

## Completion / Resolution  *(fill at freeze)*

- **verdict:** _pending_
- **gate_cmd:** _see Implementation Plan_

## Related Decisions

ADR 0034 (scope charter), ADR 0046/0047 (broker / never-OOM), ADR 0082 (cloud queue — the
program these experiments feed).

## Validation  *(ongoing)*

Wrong if: an advisory outcome ever swallows a *real* (non-opt-in) stage failure; the auto-tune
admits a run that then OOMs the box (never-OOM violated); or `blut results` is treated as the
source of truth instead of the rebuildable index. Any triggers a corrective Progress Log entry
or a superseding ADR.
