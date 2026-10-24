<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# End-to-end pass: registry, data and distributed commands

A functional pass over one half of the CLI for `0.2.0-alpha.2`, run through
built binaries against real on-disk state rather than through the test suite.
The other half (jobs, runs, lineage, plans, recipes, cache, schedules and the
rest) is covered by a separate pass.

Run on 2026-10-23 and 2026-10-24, Linux x86_64, Rust 1.88, from the release
candidate `d24dd2c` plus the fix commits named below. The cookbook binary was
built from `Quitetall/blut-cookbooks` at `6918883`.

## Method

- **Binaries.** Three builds of the engine at this branch:
  - a plain engine driver (`blut::cli::run` over an empty registry plus one toy
    cookbook, built with `--features s3`);
  - the standard cookbook binary `blut-standard` from
    `Quitetall/blut-cookbooks` (branch `agent/slurm-launch-placement`), with its
    `blut` dependency pointed at this tree;
  - the sidecars `blut-tui`, `blut-web`, `blut-notify` and `blut-dsl`, on
    `PATH` so `blut <cmd>` external dispatch reaches them.
- **Toy cookbook.** `blut-standard` registers stages but no recipes, and the
  `blut-core` recipes start a Python trainer. HPO and studies therefore ran
  against a one-stage recipe, `toy_quad`, whose stage emits a deterministic
  objective per "epoch" (`val_r` peaks at `lr = 1e-3, wd = 0.5`; `cost` grows
  with `lr` and `wd`, so the two compete). No model was trained and no GPU was
  used.
- **Isolation.** Every command ran with `HOME`, `XDG_CONFIG_HOME`,
  `XDG_DATA_HOME` and `LAMU_TRAIN_DATA_DIR` pointed into a scratch directory.
  `LAMU_TRAIN_DATA_DIR` alone does not isolate a run: the job directories, the
  registry and catalog databases, the dataset database and the policy file
  each have their own location and override variable (see open items).
- **State checks.** Outputs were compared with the state they describe:
  `model resolve` after each `promote` and `rollback`, `dataset resolve` after
  editing, moving and restoring the pinned file, `hpo show` values recomputed
  from the objective formula and from `status.jsonl`, `pareto.json` checked for
  dominance by hand, `blut-notify` delivery counts checked against the number
  of matching lines in every `status.jsonl`.
- **`update`.** Run only with a scratch `CARGO_HOME` holding a hand-written
  install ledger, and a `cargo` shim first on `PATH` that forwards `cargo info`
  (read-only) to the real cargo and records every other call instead of
  running it. Nothing was installed.
- **Object storage.** `cloud smoke` ran against local directories, `file://`
  URLs, and `s3://` on a throwaway MinIO container (removed afterwards).

## Results

Status: **pass** (worked as documented), **fixed** (a bug or footgun found and
fixed on this branch, with a regression test), **doc** (documentation
corrected), **open** (recorded below, not fixed), **n/a** (does not exist).

| Subcommand | What was run | Result | Evidence | Fix |
|---|---|---|---|---|
| `hpo run` | `--algo` random, asha, median, percentile, pbt, tpe, surrogate; `--space` YAML and `--param`; `--metric`/`--mode min`; `--args`; `--seed`; `--eta`/`--min-budget`/`--max-budget`; `--experiment`; failing trials; bad dist, bounds, algo, mode, JSON, unknown arg, `--max-trials 0`, bad tenant/launcher, missing space file | fixed; open | Before the fix every early-stopping algorithm failed the job on its first prune ("training diverged"); TPE refused its first suggestion ("runtime HPO child needs 2147483648 RAM bytes, above the held 67108864-byte tenant reservation") and replayed 4 of 6 initial trials; `--metric nosuch` exited 0; pbt on a recipe without `resume_from` only logged warnings. After: asha 9 trials (3 done, 6 stopped), tpe 6 + 6 suggestions, all exit 0; `--metric nosuch` fails naming the reported keys | `2414f09` |
| `hpo show` | latest and explicit job, `--json`, no HPO jobs, failed and stopped trials | pass; open | Values match the objective formula and the last `stage_step` per node in `status.jsonl`. Suggested (runtime) trials are not listed | — |
| `hpo best` | latest and explicit job, `--json`, metric never reported | fixed | The reproduce line printed only the winning overlay, dropping the job's base `--args '{"epochs":1,"sleep_ms":1}'`; it now prints the overlay merged over `args.json` | `2414f09` |
| `registry export` | default, `--pretty`, `--json`; plain engine (5 stages, 1 recipe) and `blut-standard` (25 stages, 0 recipes); Python SDK `RegistryManifest.from_engine` | fixed | `--json`, as written in ADR 0111, was rejected; now accepted | `f8d3a9a` |
| `study run` | two competing objectives, `sampler = "surrogate"`, 6 trials; one objective; `sampler = "grid"`; bad launcher and tenant | pass; open | `pareto.json`: 5 non-dominated of 6, checked by hand. Refusals are clear. Suggested trials are not in the front | — |
| `study report` | latest, explicit job, unknown job, missing spec | pass | Re-derives the same front | — |
| `model register` | new, idempotent repeat, other name, other tenant, uppercase and short hash, unsafe name | pass | Refusals name the existing identity | — |
| `model promote` | ungoverned alias; `@prod` with no gate, gate only, change id only, both; failing, missing, slow (`--gate-timeout 1s`) and quoted gates; `BLUT_MODEL_GATE_CMD`; `BLUT_MODEL_GOVERNED_ALIASES`; unregistered hash; wrong name and tenant; bad URI; 12 concurrent promotes | fixed | Before: 11 of 12 concurrent promotes failed "database is locked"; the gate ran for an unregistered hash; "none configured" for a gate without `--change-id`; quoted gate args arrived with literal quotes. After: 12 of 12 succeed and history has 12 rows; each refusal names its cause before any gate runs | `f8d3a9a` |
| `model rollback` | no prior target, repeated rollbacks, 6 concurrent | doc | A second rollback undoes the first (toggles between the last two). Help now says so | `f8d3a9a` |
| `model resolve` / `model history` | after every promote and rollback; `--tenant shared`; unknown pointer | pass | Hash matches the last history row | — |
| `exp compare` | toy recipe, `--json`, `--tenant`, experiment with one run, unknown name | pass; open | Arg and metric deltas match `args.json` and the last steps | — |
| `exp resolve` | known run, `--json`, unknown run, wrong recipe, other tenant, missing scheme | pass | Returns the lineage row | — |
| `dataset ingest list/add/rm/show` | JSONL and binary sources, `--kind`, `--n-examples`, `--metadata` (object, non-object, invalid), relative path, directory, missing file, duplicate name, unknown name | fixed | Before: a WARN on every command, relative paths stored as typed, the duplicate hint named `lamu-train`, and `rm` of a pinned source made `dataset://ds_a@v1` permanently unresolvable (re-pinning reported "concurrent rebind"). After: absolute paths, no warning, `rm` refused naming the pinned versions | `f8d3a9a` |
| `dataset pin` | new, idempotent, rebind, same bytes under another name, unknown source, bad URI, bad tenant, clinical source into `default` and `restricted` | pass | | — |
| `dataset resolve` | `--json`, `--tenant`, `--launcher` local/slurm/bogus, edited bytes, moved file, restored file, clinical on slurm | pass | Drift refused by hash; restricted refused off-node | — |
| `dataset catalog rebuild/search/show/tag` | rebuild, every query key, bad key, non-numeric `fs`, `--cloud`, `--json`, `name@version`, unknown name, tag survives rebuild | fixed | Before: `show ds_c@v99` printed `ds_c`; `tag nope sleep` succeeded. After: both refused | `f8d3a9a` |
| `dataset compare` | — | n/a | No such subcommand; comparison of runs is `exp compare` | — |
| `connectors list` | plain, `--check`, `--json` | pass | 3 connectors, kinds match | — |
| `policy show/enable/disable` | default file, `$LAMU_TRAIN_POLICY`, invalid value, unknown key; printed cron line checked against `sensor eval auto-train-policy` | fixed | Before: unknown or misspelt keys were silently ignored and dropped on the next `enable`; with `$LAMU_TRAIN_POLICY` the cron line evaluated the default policy. After: unknown keys warned; the cron line carries the variable | `f8d3a9a` |
| `cloud smoke` | local dir, relative dir, existing and missing `file://` dir, `s3://` on MinIO (twice, idempotent), missing bucket, wrong secret, `http://` without `AWS_ALLOW_HTTP`, unknown scheme, store path is a file, empty input; engine driver and `blut-standard` | fixed; open | Before: `file:///missing/dir` failed "Unable to canonicalize filesystem root". After: created like a bare path. Round-trip output verified each time | `f8d3a9a` |
| `tui` | `--check`; bare engine exec of `blut-tui`; driven in a pty (view keys `J L Y H B K P M X`, then `q`); sidecar missing from `PATH`; unknown flag | fixed | Before: on a machine with no jobs the home view showed a running plan `train_joint · tier7`, 4 mesh peers, `73% hit` and `412 MiB/s`, all made up. After: an idle model with only loader data. Exits 0 and leaves the alternate screen | `d91aff2` |
| `update` | `--check`, default, `--yes`, `--check --yes`; ledger says behind, ahead, prerelease, path install with `%20`, non-semver, no entry, no ledger; scratch `CARGO_HOME`, `cargo` shim | fixed | Only `cargo info` and one `cargo install …` were recorded by the shim, never run. `--check --yes` ignored `--yes`; now an error | `f8d3a9a` |
| External passthrough | unknown command, invalid name, removed `data` and `catalog`, argument passing, sidecar exit 3, sidecar killed by signal 9; `blut web`, `blut notify`, `blut dsl` | fixed | Before: exit 3 became 1 behind a second error line. After: 3 passes through, a signal maps to 128 + n | `f8d3a9a` |
| `p2p keys generate/show` | default path, `--key` (mode 0600, parent dirs created), refuse overwrite, `--force`, missing and malformed key | pass | | — |
| `p2p peers list/trust/remove` | empty, `--json`, after real connections, prefix match, unknown level, unknown id, repeated remove | pass; open | Reputation and task counts stay 0 after successful smoke dispatches | — |
| `p2p serve` / `p2p connect` | `--smoke-stage p2p-echo` with `--smoke-input`; long-running serve stopped with SIGINT; connect to an absent coordinator; short pubkey; bad listen address | pass; open | Echo output verified. One handshake timed out under heavy host load and did not recur in five retries; connect to an absent coordinator prints nothing for 30 s before a clear error | — |
| `blut-dsl` | `examples/demo.star` with and without `tiers`, invalid `--args`, missing file, a script that adds nothing, via `blut dsl` | doc | The README example omitted `tiers`, which the shipped demo requires | `196f4e9` |
| `blut-notify` | stdin envelope (internal, restricted, malformed, empty); `--config --once` with an `exec` sink; cursor resume | doc | The README rule matched `kind = "failed"`, which no status line has (0 of 667 lines); `stage_failed` delivered 42, equal to the `stage_failed` count across all jobs | `196f4e9` |
| `blut-web` | `/healthz`, `/api/jobs`, job status (known, unknown), `/api/cost`, `/api/models/{name}/{alias}`, lineage card, mutation without a token, non-loopback bind without tokens, UI index | fixed | Before: every pointer created with `blut model` defaults was 404, because the view read only the `shared` tenant. After: defaults to `default`, `?tenant=` selects another, restricted stays unexported | `000681e` |
| `blut-tui` binary | `--check`, `--help`, `--version`, a typo, run without a terminal | fixed | Before: every argument but `--check` was ignored, so `--help` tried to enter raw mode ("No such device or address"). After: help and version print, unknown arguments exit 2, no terminal is a clear error | `d91aff2` |

## Summary

30 rows: 13 pass (5 of them with an open remainder), 13 fixed (2 with an
open remainder), 3 documentation corrections, 1 not applicable. Fix commits on `agent/e2e-registry`:

| Commit | What it fixes |
|---|---|
| `2414f09` | HPO early stopping no longer fails the job; TPE / surrogate / PBT children billed their own envelope and no longer replay the initial population; `hpo best` reproduce line keeps the base args; unreported `--metric` and PBT without `resume_from` refused |
| `f8d3a9a` | SQLite busy timeout on the registry, model, dataset and catalog stores; pinned dataset sources cannot be removed; absolute dataset paths; catalog `name@version` and unknown names; governed `model promote` checks before the gate and names what is missing; `update --check --yes`; policy unknown keys and cron override; `file://` cloud stores; `registry export --json`; external exit codes |
| `d91aff2` | `blut tui` console shows only loaded state; `blut-tui` parses `--help` / `--version` / typos and needs a terminal |
| `000681e` | `blut-web` model view reads the `default` tenant |
| `196f4e9` | `blut-notify` and `blut-dsl` README examples |

Gates for each fix commit, Rust 1.88: `cargo fmt --check`; `cargo clippy
--locked --all-targets -D warnings` (default, `p2p`, `s3`); `cargo test
--locked` (after `f8d3a9a`: 1203 default, 1400 `p2p`, 1435 `s3`, 0 failed);
`cargo doc` with `-D warnings`; `scripts/scope_guard.sh`;
`scripts/semver_checks.sh` (all six published crates compatible). The
`blut-tui` and `blut-web` crates: fmt, clippy, tests (79 and 13 passed).

## Open items

Not fixed on this branch, by severity.

**High**

1. **One failing trial fails the whole HPO job.** Trials are nodes of one
   plan, and the executor fails a plan fast: with `fail_above` set so three of
   six trials error, `hpo run` exits 1 ("hpo plan execution failed") and a
   healthy running trial is killed. `hpo show` and `hpo best` still read the
   finished trials. Search tools usually record a failed trial and continue;
   whether HPO plans should is a design decision (per-trial failure isolation
   in the executor).
2. **Runtime-spawned trials are invisible.** TPE and surrogate suggestions
   and PBT clones run (12 nodes for a 6-trial TPE job) but are not in
   `hpo.json`, so `hpo show`, `hpo best` and a study's `pareto.json` only rank
   the random initial population: the model-based part of the search never
   reaches the result.

**Medium**

3. **`exp compare` on HPO jobs** compares the last step of whichever trial
   finished last, which says nothing about either search.
4. **`LAMU_TRAIN_DATA_DIR` isolates less than its name suggests.** Job
   directories (`LAMU_TRAIN_JOBS_DIR`), `~/.blut/registry.db`
   (`BLUT_REGISTRY_DB`), `~/.blut/catalog.db` (`BLUT_CATALOG_DB`), the dataset
   database (`LAMU_MEMORY_DB`, a file named `conversations.db`) and the policy
   file (`LAMU_TRAIN_POLICY`) each have their own default under `$HOME`.
5. **Timing-sensitive tests on a loaded host.** With load average 60 to 227
   from other work on the same machine, the suites failed on deadlines that
   are not about the code under test: p2p tests whose QUIC handshake reply
   missed its 10 s window, `dag_opt_advanced` 5 s event deadlines, and once a
   `conditional_execution` run that waited indefinitely at 0% CPU (also seen
   at the same moment in two other worktrees of this commit). Each passed when
   re-run. A test that waits on a signal with no hang guard turns a lost race
   into a hang instead of a failure.

**Low**

6. `hpo show` labels an early-stopped trial `killed`, the same word used for a
   trial killed because a sibling failed.
7. `model history` prints raw Unix seconds and does not say whether a row came
   from a promote or a rollback.
8. `cloud smoke` errors repeat the provider's message once per error layer
   (five copies of the S3 XML body), and a plain-HTTP endpoint without
   `AWS_ALLOW_HTTP=true` ends in "URL scheme is not allowed" with no hint.
9. `cloud smoke` defaults `--store` to `/tmp/blut-cloud-smoke`, a fixed path
   in a shared directory.
10. `p2p connect` to an address with no coordinator prints nothing until its
    30 s timeout.
11. `p2p peers list` shows reputation 0.50 and zero completed tasks for peers
    that ran smoke dispatches successfully.
12. `blut-notify` summaries say only "status event for job X", not which
    stage or what error.
13. A Starlark script that never calls `add()` emits an empty PlanSpec and
    exits 0.
14. `README.md` ("v0.2.0-alpha.1 — current development preview") and
    `ROADMAP.md` ("between `0.2.0-alpha.1` and a 1.0") still name alpha.1,
    while the crate and the README install line are alpha.2. Left to the
    release commit.
15. `blut-standard` registers stages but no recipes, so `hpo` and `study`
    cannot be tried with the standard cookbook alone; the `blut-core` recipes
    start a Python trainer and were not run here.

