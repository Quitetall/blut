<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0110 — Ship auto batch-size, LR-finder, AMP, grad-accum, and loss-spike tuning as opt-in control policies that only ever propose — the broker still decides

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete` (frozen 2026-07-11)

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite or delete earlier text; correct an earlier statement with a *new* dated
> entry beneath it. When the deliverable passes its gate, fill **Completion** and
> set Status to `Complete — frozen YYYY-MM-DD`.

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-11`

- **Extends, does not supersede:** [ADR 0088](0088-blut-durable-execution-planspec-1-1.md), ADR 0104 [not federated]; reuses the never-OOM
  broker ([ADR 0046](0046-blut-resource-broker.md)/[0047](0047-broker-measured-envelope-and-codec-dataloader.md)) as the sole admission authority.

## Context

Practitioners expect the conveniences PyTorch Lightning and Metaflow bundle:
auto batch-size ("scale until just below OOM"), an LR range finder
(Smith/`lr_find`), automatic mixed precision, automatic gradient accumulation to
hit a target effective batch, and a loss-spike guard that backs off LR when the
loss diverges. Today a BLUT trainer stage takes these as fixed typed args; a
human hand-picks them per corpus, and a bad guess either OOMs mid-run or wastes a
warm-up.

Two things make the Lightning/Metaflow implementations unsafe to copy directly:

1. **They mutate the trainer silently.** Lightning's `auto_scale_batch_size`
   binary-searches by *actually launching* larger batches and catching the OOM —
   i.e. it deliberately OOMs the box to find the ceiling, and it rewrites
   `trainer.batch_size` in place. In BLUT that is a double charter violation: it
   bypasses broker admission (0046/0047 exist precisely so a stage never OOMs the
   box) and it makes an untracked edit to a stage's typed inputs (breaks the
   content-addressed cache key — the run would no longer be reproducible from its
   recorded config).
2. **They live inside the training loop**, so there's no audit record of *what
   was changed and why* — the effective config drifts from the declared one.

We want the ergonomics without either failure. The broker already knows the
memory budget analytically (ADR 0046/0047 admission is a *prediction*, not a
try-and-catch), so "auto batch-size" can be answered *without ever launching an
over-budget batch*.

## Decision

Add five **auto-tuner control policies**, all **opt-in** (off by default; a run
with no tuner flags is byte-identical to today) and all obeying one rule:
**a tuner PROPOSES a config change; it never edits a running trainer, and every
proposal is re-admitted by the broker before it takes effect.** The tuner emits a
typed `TuneProposal { field, old, new, reason }`; the scheduler applies it by
**spawning a new stage-run with the amended (still content-addressed) config**,
not by hot-patching the live process. The proposal + the broker's accept/refuse
verdict are appended to the run's `tuning.jsonl` audit trail.

The five:

- **`AutoBatch`** — asks the **broker's analytic memory model** for the largest
  batch that *admits* under the current budget (no launch-and-catch, no OOM). The
  broker's number is the ceiling by construction; the tuner just proposes it. If
  the operator's floor batch already exceeds budget, the tuner proposes nothing
  and the stage fails admission as it would today (fail-closed).
- **`GradAccum`** — given a target *effective* batch and the `AutoBatch`
  micro-batch the broker admitted, proposes `accum_steps = ceil(target /
  micro)`. Pure arithmetic on already-admitted numbers; never raises the
  micro-batch above what the broker allowed.
- **`LrFinder`** — runs a short **bounded** LR-range sweep as its **own DAG
  node** (a mini exponential-LR probe over N cache-keyed steps), reads the
  loss-vs-LR curve, proposes the min-gradient LR. It is a node, so it is
  broker-admitted like any node; results are cached.
- **`AutoAmp`** — proposes an AMP dtype (bf16 where supported, else fp16 w/ loss
  scaler, else off) from a capability probe; refuses to propose AMP for a stage
  the numerics-guard marks fp32-only (e.g. ternary-QAT calibration paths).
- **`LossSpikeTuner`** — a *reactive* policy: on a loss spike past a z-score
  threshold it proposes an LR back-off (× factor) or a rollback-to-last-good
  checkpoint. It composes with, and never overrides, the landed
  `KillOnNaN` — NaN still kills; a finite spike gets the graduated response.

All five are `ControlPolicy` impls feeding the same scheduler seam ADR 0067
Slice B built; none is a daemon, none links HTTP. Clinical/PHI runs: tuners are
allowed (they touch config, not PHI), but an `LrFinder` probe over a PHI corpus
inherits the existing fail-closed hard-block on that corpus.

## Rationale

- **The broker already has the answer to "how big can this batch be?"** —
  analytically, before launch. Copying Lightning's launch-until-OOM would throw
  that away *and* violate the never-OOM invariant. Proposing the broker's own
  admission ceiling is both safer and cheaper (zero failed launches).
- **Propose-then-readmit keeps the cache honest.** Because the change lands as a
  new content-addressed config rather than an in-place mutation, the run is still
  reproducible from its recorded inputs, and the tuning decision is auditable in
  `tuning.jsonl` — the effective config can never silently diverge from the
  declared one.
- **Opt-in / off-by-default** preserves determinism: a run without tuner flags
  is byte-identical to today, so the numerics and byte-equal gates elsewhere are
  untouched.
- **Reactive spike handling composes with `KillOnNaN`** rather than replacing it —
  NaN stays a hard kill; only finite divergence gets the softer LR back-off.

## Alternatives Considered

- **Port PyTorch Lightning's `auto_scale_batch_size` / `Tuner` verbatim.**
  Rejected on two counts: it finds the batch ceiling by *launching* over-budget
  batches and catching the OOM (directly violates the never-OOM broker), and it
  mutates `trainer` state in place (breaks the content-addressed cache key /
  reproducibility). We take the *feature* and invert the mechanism: analytic
  broker query instead of launch-and-catch, new-config-spawn instead of in-place
  edit.
- **In-loop tuning callbacks (the Lightning/Metaflow decorator style).**
  Rejected: hides the effective config inside the training process with no
  lineage record, so the declared config and the run drift silently — the exact
  provenance failure ADR 0104 [not federated] / the audit trail exist to prevent.
- **Let AutoBatch launch a tiny probe batch to measure real memory, then scale.**
  Rejected as unnecessary and risky: the broker's analytic model is the budget
  authority already; a measured probe that disagreed with the broker would just
  be a second, launch-based memory oracle competing with the admission gate.

## Consequences

- New typed artifact `TuneProposal` + a per-run `tuning.jsonl` audit trail;
  lineage tooling records proposal→verdict pairs.
- The broker's memory-estimation surface (ADR 0088 lineage) gains a second
  caller: `AutoBatch` queries "largest admissible batch" — the estimator must
  expose that query, not just yes/no admission.
- `LrFinder` adds a short probe node to studies that enable it (cheap, cached);
  it is broker-admitted like any node.
- A tuner can make a stage *fail to start* (if even the floor config is
  over-budget) — that is the correct fail-closed behavior, but it means enabling
  a tuner can surface a latent budget problem that a hand-tuned run masked.
- Off-by-default means zero impact on existing runs and gates; the surface is
  strictly additive.

## Implementation Plan  *(the contract — amend only by a dated Progress Log entry)*

- **Deliverable:** five `ControlPolicy` impls (`AutoBatch`, `GradAccum`,
  `LrFinder`, `AutoAmp`, `LossSpikeTuner`), a `TuneProposal` typed record, a
  broker "largest-admissible-batch" query, and a `tuning.jsonl` audit writer;
  each tuner opt-in via a study/stage flag, all off by default.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut auto_tune:: && blut run
    tests/fixtures/train_autobatch.toml --auto-batch && ` a tuning-audit
    checker `[redacted: private path]` over `runs/<id>/tuning.jsonl`
  - `pass when:` exit==0 AND `AutoBatch`'s proposed batch equals the broker's
    admission ceiling (no over-budget launch occurred — box did not OOM) AND
    `tuning.jsonl` records every proposal with a broker accept/refuse verdict AND
    a run with no tuner flag produces a byte-identical config hash to the
    pre-ADR baseline.
  - `pass when (spike):` an injected finite loss spike triggers an `LrFinder`/
    back-off proposal while a NaN still hits `KillOnNaN` (composition holds).
- **Rollback:** feature-gate `auto-tune`; disabled ⇒ all five policies
  unavailable and stages take their fixed typed args exactly as today
  (byte-identical config hashes).

## Progress Log  *(append-only; newest last)*

- _2026-07-10 — opened._
- _2026-07-11 — LANDED + FROZEN (engine main `64a94de`). `src/auto_tune.rs`: the
  five tuner PROPOSAL policies + `TuneProposal` + `BrokerVerdict` + `tuning.jsonl`
  writer. `auto_batch` proposes the broker's largest-ADMISSIBLE batch (the caller
  passes the ceiling from `broker::footprint::batch_size_to_fit` — the analytic
  memory model, never a launch-and-catch) and is fail-closed: a ceiling below the
  operator floor (or 0) proposes NOTHING. `grad_accum` = ceil(target/micro) on
  already-admitted numbers. `lr_finder` = the min-gradient LR (steepest descent
  of loss vs log lr) from a bounded probe curve. `auto_amp` = bf16-else-fp16-else
  -off, records the actual `current` as `old`, refuses fp32-only stages + no-op
  proposals. `loss_spike` = a reactive z-score spike → graduated LR back-off,
  while a NaN defers to the landed `KillOnNaN` (composition holds, never
  overrides). **Gate met:** `cargo test -p blut auto_tune::` = 6 passed — AutoBatch
  proposes EXACTLY the broker ceiling (and nothing when ceiling < floor / 0);
  GradAccum ceil-division; LrFinder steepest-descent pick; AutoAmp bf16-pref +
  fp32-only refusal; a finite spike backs off while a NaN hits KillOnNaN; and
  every proposal + verdict lands in `tuning.jsonl`. 857 lib green; clippy+fmt+
  scope_guard clean. MiMo (2 rounds) → `auto_amp(current)` (accurate audit `old`
  + no no-op rows), `ceiling==0` guard, doc softening. **TAIL (own increment):**
  wiring the five as `ControlPolicy` impls into the ADR-0067 scheduler seam (the
  `--auto-batch` CLI flag / spawn-amended-config path) + the tuning-audit checker `[redacted: private path]` — the PROPOSAL logic + audit are the gated primitive.
  The broker is never bypassed (auto_batch reuses the admission ceiling) and the
  config-hash-unchanged-when-off invariant is structural (no proposal ⇒ no config
  change). Status → Complete — frozen 2026-07-11._

## Completion / Resolution  *(fill once)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut auto_tune::`
- **result:** exit 0 (6 passed). `AutoBatch`'s proposed batch equals the broker's
  admission ceiling exactly (no over-budget launch); a NaN still hits `KillOnNaN`
  while a finite spike gets a back-off (composition holds); `tuning.jsonl` records
  every proposal with an accept/refuse verdict; and — structurally — a run with no
  tuner flag proposes nothing, so its config hash is byte-identical to today.
  scope_guard PASS; 857 lib green; clippy + fmt clean.
- **git_sha:** engine `64a94de`  ·  **artifact sha256:** n/a (source-only)

## Related Decisions

[ADR 0088](0088-blut-durable-execution-planspec-1-1.md) (memory estimation — `AutoBatch` queries its admissible-batch surface),
ADR 0104 [not federated] (provenance/audit — `tuning.jsonl` extends it), [ADR 0109](0109-blut-hpo-advanced.md) (advanced HPO —
a tuner-proposed config can seed the surrogate search; sibling), [ADR 0067](0067-remaining-work-toward-vision.md)
(control-policy seam + `KillOnNaN` that `LossSpikeTuner` composes with),
[ADR 0046](0046-blut-resource-broker.md)/[0047](0047-broker-measured-envelope-and-codec-dataloader.md) (never-OOM broker — the sole admission authority these tuners defer to).

## Validation  *(ongoing)*

Wrong if: any tuner ever mutates a running trainer in place instead of spawning a
re-admitted config; `AutoBatch` launches a batch the broker did not admit (a
single box OOM falsifies the whole ADR); a tuner-changed run's effective config
stops matching its declared/cached config hash; or a tuner override lets a NaN
survive `KillOnNaN`.
