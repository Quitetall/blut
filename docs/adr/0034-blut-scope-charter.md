<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0034 — BLUT is the domain control plane — own the nouns, delegate the verbs

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `in-progress` (Proposed 2026-06-04 · Accepted 2026-08-27 · In progress 2026-08-27)

## Context

BLUT has accreted a TUI cockpit, a Rust job-lifecycle layer
(`jobs.rs`, `python_kill.rs`, `backends/lamquant/runner.rs`), and a
Python experiment-coordination layer (`experiment_runner.py`,
`experiment_log.py`). Four area audits (tui-views, orchestration,
recipes-artifacts, pccp-parity-coupling) were run against a single
charter: *BLUT = LamQuant's domain control plane — it OWNS the
project's nouns and DELEGATES the generic verbs.*

The audits converged on a clean two-bucket result:

- **recipes-artifacts and pccp-parity-coupling found ~zero bloat.**
  The recipe catalog (`RecipeDef`), the typed `Artifact` boundary,
  the content-addressed `cache`, the typed `Plan`/`executor` DAG, the
  PCCP gate stages, multi-root `paths.rs`, and the firmware-export
  parity point are all genuinely LamQuant-specific. They are KEEP.

- **All SHED/REPLACE findings come from exactly two places:**
  1. **TUI generic monitoring** — `tui/system.rs` (nvidia-smi /
     /proc / df probes), `draw_system`, and the `metrics_tail` /
     `refresh_jobs` polling loop (`REFRESH_TICK` = 1500 ms).
  2. **Process-lifecycle reimplementation** — `jobs.rs` MONITOR-2
     dead-pid reconciliation via `kill(pid,0)`, `python_kill.rs`
     setsid/pgid/graceful-kill (~502 LOC), and `runner.rs`
     tqdm-from-pipe parsing + process-group setup.

The forcing function is concrete and dated: on **2026-06-04 03:48** a
DataLoader RAM blowup triggered a *global* kernel OOM that swept the
entire `user@1000` slice. The fix already written —
`run_contained.sh` (now `scripts/run_contained.sh` in this repository) —
wraps jobs in memory-capped, session-immune
`systemd-run --user` transient units. That tool is in active manual
use but **no BLUT Rust source calls it yet**. systemd already owns
OOM-isolation, liveness, cleanup, and graceful kill; BLUT's
hand-rolled process supervision is a redundant reimplementation of a
mechanism the OS provides.

Three modules received *conflicting* verdicts across audits because
auditors looked at different layers of the same file: `jobs.rs`
(SHED vs KEEP-bookkeeping vs REPLACE-polling), `runner.rs` (REPLACE
vs KEEP-delegated), `python_kill.rs` (SPLIT). The charter resolves
these by layer, not by file (see Decision).

## Decision

BLUT is the **domain control plane** for LamQuant. It OWNS the
project's nouns and DELEGATES the generic verbs to named,
already-deployed systems.

**OWN (KEEP — nothing off-the-shelf knows these LamQuant nouns):**

- **Recipes** — declarative typed `RecipeDef` train/distill/eval/
  codec-bench definitions; `category` / `input_kinds` / `output_kind`
  KIND validation; the launch CONTRACT (paths, manifests, quant
  config correct). `src/recipes/`, `src/stages/lamquant_*.rs`.
- **Artifact + provenance graph** — typed `Artifact` boundary
  (`KIND`/`SCHEMA`/`content_hash`), `ArtifactMetadata` lineage
  sidecars, navigation along the domain's artifact chain
  `[redacted: codec research]`; the typed
  `Plan`/`executor`/`cache` DAG; `RunManifest` (git_sha, config_hash,
  dataset SHAs). `src/framework/`, `src/artifacts/`,
  `[redacted: private path]`.
- **PCCP / ship-gate front door** — classify change, run the
  domain's ship gate `[redacted: private path]`, registry/changelog surface. `src/registry.rs`,
  `src/stages/lamquant_pccp_gate_*.rs`,
  `src/recipes/lamquant_full_pipeline.rs`.
- **train↔deploy PARITY contract** — `HardenedCkpt` →
  `FirmwareBundle` typed-input enforcement that the deployable bundle is built only
  from hardened, matched training outputs, bit-exact `[redacted: non-engine product detail]`.
  `src/stages/lamquant_export_firmware.rs`,
  `src/stages/lamquant_harden_artifacts.rs`. (Currently enforced
  *implicitly* by types; see Consequences.)
- **Experiment coordination** — recipe-based A/B/grid sweeps, the
  experiment catalog, domain quality-gate semantics `[redacted: codec research]`.
  `[redacted: private path]` (cookbook experiment runner and log modules).

**DELEGATE (SHED or REPLACE → named tool):**

| Generic verb | Delegate to |
|---|---|
| compute orchestration / job launch / OOM containment | `systemd-run --user` via `run_contained.sh` |
| job liveness / dead-pid reconcile / graceful kill | systemd cgroup unit state + `systemctl --user stop` |
| log tailing / aggregation / stdout capture | journald (`journalctl --user -u <unit>`) |
| metrics + training-dynamics viz / run comparison | wandb (offline) |
| GPU/CPU/MEM/DISK monitoring | wandb offline dashboard or `systemctl status` |
| experiment record | `outputs/experiment_log.jsonl` + wandb |

> **AMENDMENT 2026-08-02 — the `experiment record` row above is superseded by
> [ADR 0154](0154-blut-owns-the-run-ledger.md).** The row is left in place as
> historical record; it no longer describes the system.
>
> Every other delegation in this table stands unchanged — in particular *metrics +
> training-dynamics viz / run comparison* remains wandb's. The distinction 0154
> (then numbered 0152) draws is that a run **ledger** is a BLUT noun (BLUT already assigns `job_id` and
> records `runs`/`artifacts`/`lineage_edges`/`git_sha`), whereas a run **dashboard**
> is a verb and stays delegated.
>
> The delegate target had also stopped functioning: `outputs/experiment_log.jsonl`
> was last written 2026-05-28, 1894 of its 2021 rows are pytest runs, its two test
> modules are import-errors, and its writer and its readers had come to point at
> different files. See ADR 0154 Context for the measurements.

**Layered resolution of the conflicting-verdict modules** — split
each file along the line auditors actually drew. KEEP the *domain
schema/interface*, SHED/REPLACE the *generic mechanism*, because once
jobs run as systemd transient units that mechanism is dead code:

- `jobs.rs` — KEEP the domain schema (job-dir layout, `JobState`
  enum, `JobSummary`, `new_job_id`); SHED MONITOR-2 `kill(pid,0)`
  liveness + `summarize()` reconciliation. systemd owns liveness.
- `python_kill.rs` — KEEP the pid-reuse *concept*; SHED the
  setsid/pgid registry + SIGTERM→SIGKILL graceful-kill machinery.
  systemd cgroups own grandchild cleanup and `systemctl stop` owns
  graceful kill.
- `runner.rs` — KEEP the `LamquantBackend` trait boundary +
  expected-output verification; REPLACE bare-subprocess spawn with
  `systemd-run`, and the tqdm-from-pipe parser with journald
  structured output.

## Rationale

- **The charter test is binary and the audits agree on which side
  each module falls.** A module is in scope iff it manipulates a
  LamQuant domain noun (recipe / artifact / lineage / PCCP / parity)
  AND no off-the-shelf system owns the mechanism. `system.rs` probes
  `nvidia-smi`/`/proc`/`df` — pure generic monitoring, zero domain
  nouns → out of scope. `RecipeDef.input_kinds` validation — pure
  LamQuant noun → in scope.
- **systemd is the correct primitive and is already proven.**
  `run_contained.sh` was written in response to a real global-OOM
  incident; its cgroup `MemoryMax`/`MemoryHigh`/`MemorySwapMax` and
  slice immunity are exactly the isolation/liveness/kill primitives
  `python_kill.rs` reinvents in user space. Reusing the OS beats
  maintaining a fragile reimplementation.
- **Two append-only status logs is one too many.** `jobs.rs` writes
  `status.jsonl` per job (journal-like event semantics for a single
  domain) while journald provides generic event streaming for free.
  Domain results belong in `experiment_log.jsonl` (domain-semantic
  quality metrics `[redacted: codec research]`); process events belong in
  journald.
- **The two-bucket result makes the charter falsifiable in practice.**
  Core (recipes/artifacts/cache/plan/PCCP/paths/parity) audited clean
  KEEP across two independent audits; every SHED/REPLACE traces to
  TUI monitoring or process-lifecycle. If a future audit finds bloat
  *inside* the core bucket, the charter was mis-drawn.

## Alternatives Considered

- **Keep BLUT's hand-rolled job supervision** (status quo). Rejected:
  it caused the 2026-06-04 global OOM (no cgroup isolation), requires
  `kill(pid,0)` liveness polling that races out-of-band death, and
  maintains ~502 LOC of setsid/pgid plumbing that systemd does
  natively and correctly.
- **Build a richer in-TUI metrics dashboard** (extend `system.rs`,
  add charts). Rejected: directly violates the charter — metrics
  visualization is a generic verb owned by wandb. Extending it is the
  exact scope-bloat the audit flagged.
- **Delete `jobs.rs`/`runner.rs` wholesale** (orchestration audit's
  SHED-everything reading). Rejected: it throws away the domain schema
  (`JobState`, `JobSummary`, the `LamquantBackend` trait boundary) that
  the recipes audit correctly KEEPs. The right cut is SPLIT-by-layer,
  not delete-the-file.

## Consequences

- **Commits us to wiring BLUT's launch path through `run_contained.sh`
  /systemd-run.** This is the keystone migration (see a migration plan
  `[redacted: private path]`): cancel → `systemctl --user stop`,
  status capture → journald, liveness → unit state. It ripples into
  the TUI job/log views, so it ships behind a flag and is A/B'd
  against the current path.
- **Dead code on completion:** MONITOR-2 reconciliation,
  `python_kill.rs` pgid/graceful-kill machinery, `runner.rs` tqdm
  parser + process-group setup, `tui/system.rs`, `draw_system`,
  `metrics_tail`.
- **Load-bearing on completion:** `run_contained.sh` (and the systemd
  --user manager) becomes a hard runtime dependency of BLUT job
  launch. journald becomes the log source of truth for process
  events. wandb-offline becomes the metrics surface (currently wired
  in only one trainer `[redacted: private path]` — full coverage is a
  prerequisite, not done).
- **PARITY is OWNED but currently only IMPLICIT** (pccp audit
  SIGNAL-3): `FirmwareBundle`'s typed inputs enforce that the deployable
  bundle comes from `HardenedCkpt` + matched ckpts, but there is no written
  "FirmwareBundle SHALL be byte-exact reproducible from training
  outputs" contract. Follow-up: add a written parity contract
  `[redacted: private path]`. This
  ADR claims parity as owned-but-implicit, not as a written contract.
- **Anti-scope-bloat is enforced, not aspirational** — see the scope-guard
  note `[redacted: private path]` (implemented here as `scripts/scope_guard.sh`):
  a CI grep job over `src/**.rs`
  fails on new generic-concern imports (nvidia-smi/proc/df, plotting
  crates, http servers, setsid/sysinfo), plus a PR-checklist line.

## Related Decisions

- ADR 0017 [not federated].
- ADR 0025 [not federated].
- ADR 0026 [not federated].
- Follow-up: a written parity contract `[redacted: private path]` (SIGNAL-3 gap).

## Validation

The charter is right if, after migration:

- `src/**.rs` contains zero `nvidia-smi`/`/proc/(meminfo|loadavg)`
  /`\bdf\b`/`setsid`/`kill(.*,0)` matches (the scope-guard CI job is
  green). Falsifier: any of those reappear in core.
- BLUT job launch survives a session-compaction SIGTERM and a
  DataLoader RAM blowup is cgroup-OOM-killed in isolation (the
  2026-06-04 incident does not recur).
- A future audit of the OWN bucket
  (recipes/artifacts/cache/plan/PCCP/paths/parity) finds no
  SHED/REPLACE rows. If it finds bloat inside core, revisit the
  bucket boundary.

The falsifiable **"in scope?" test for any new feature**:

> A feature belongs in BLUT iff it (1) manipulates a LamQuant domain
> noun — recipe, artifact, lineage, PCCP, or parity — AND (2) no
> off-the-shelf system already owns the mechanism. If systemd /
> journald / wandb already does it, BLUT may keep only the thin domain
> adapter (the noun-to-tool mapping), never the mechanism itself.

Worked failures: `system.rs` nvidia-smi probe → fails (2): systemd/
wandb own monitoring → out. `metrics_tail` CSV polling → fails (2):
wandb owns metrics viz → out. `python_kill.rs` setsid → fails (2):
systemd owns process groups → out. `RecipeDef` KIND validation →
passes both → in.

## Progress Log

- _2026-08-27_ — **In progress — 2026-08-27** under ADR 0186 [not federated] clause 8. The charter's DELEGATE half landed and its OWN half is largely built, but the one thing it flags as an open consequence is still open. Delegation is real: `run_contained.sh` exists and the engine actually shells to systemd-run (`src/config/launcher.rs:194`, `src/containment/memsize.rs:103`). What is missing is the deliverable the charter itself calls out -- the train-to-deploy PARITY contract, 'currently enforced implicitly by types' -- and its citation gate names exactly that gap: the written parity contract does not exist. The cited `tui/system.rs` did survive, at `crates/blut-tui/src/system.rs`, and is now aliased so the gate reports the real gap rather than a stale path.

- A citation gate `[redacted: private path]` was attached to this record on
  2026-08-27 (it proves the record still points at files that exist; it does
  not claim the decision is implemented), replacing its entry in a closure-debt
  list `[redacted: private path]`.

- _2026-08-27_ — **Accepted 2026-08-27** under ADR 0186 [not federated] clause 8. Its '## Decision' section states a decision this project follows, so `proposed` understated it: the record was never withdrawn and later work builds on it. Accepted is non-terminal and asserts only that the decision stands, not that it is finished.
