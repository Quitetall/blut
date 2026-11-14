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
  At the time `LAMU_TRAIN_DATA_DIR` alone did not isolate a run: the job
  directories, the registry and catalog databases, the dataset database and
  the policy file each had their own location (fixed since, see below).
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

## Follow-ups

Fixed after the pass, on `agent/fu-hpo`:

| Item | Commit | What changed |
|---|---|---|
| 1. A failing trial failed the whole HPO job | `a5d6f4e` | Trials run with stage-failure isolation: a failed trial is recorded with its error and its siblings keep running; the job fails only when no trial completes with the objective |
| 2. Runtime-spawned trials were invisible | `a5d6f4e` | `ControlPolicy::on_spawned` and `hpo::TrialBook` record TPE / surrogate suggestions and PBT clones in `hpo.json` as they start; they are ranked and fed back to the policy |
| 3. `exp compare` on HPO jobs | `36b4041` | Compared through the best completed trial's nodes, which the output names |
| 4. `LAMU_TRAIN_DATA_DIR` isolated little | `31d0443` | It is the root of every store of user state; explicit per-store variables still win; unset is unchanged |
| 6. `killed` meant two things | `a5d6f4e` | `stopped` (early stop), `cancelled` (plan cancel), `failed` |
| 7. `model history` raw seconds | `4f8d24c` | UTC timestamps, current row marked; a promote and a rollback still look alike (no kind column in the table) |
| 8. Repeated S3 errors, no `AWS_ALLOW_HTTP` hint | `4f8d24c` | One-line errors; a plain-HTTP endpoint is refused up front with the hint |
| 10. `p2p connect` silent for 30 s | `4f8d24c` | Prints that it is connecting |

E2E after the fixes (toy CPU objective): 6 trials with 3 failing → exit 0,
3 completed and 3 failed with their errors; 3 of 3 failing → exit 1 naming
the first error; asha with failures → 2 completed, 5 stopped, 2 failed; tpe
→ 12 ranked trials (6 suggested); pbt → 11 (5 clones); a surrogate study's
front over 12 measured trials; with `LAMU_TRAIN_DATA_DIR` set and a fresh
`HOME`, a recipe run, an HPO search, model, dataset, catalog and policy
commands wrote nothing under `HOME`.

Fixed after that, on `agent/fu-smalls` (open items below are marked with
their commit):

| Item | Commit | What changed |
|---|---|---|
| `model rollback` toggled; `model history` had no kind (follow-up 7) | `9f57d1e` | Rollback walks the promotion history like `plan rollback` (refused at the first promotion; a promotion after a rollback continues from there; re-promoting the current target is not a step); `model history` prints `promote` / `rollback`; old registries gain the column on open, old rows count as promotions |
| 16. `LAMU_TRAIN_DATA_DIR` migration | `dc4c837` | When it is set, each store whose own variable is unset, that is empty under the root, and whose old default holds data is named in one startup warning with the variable to set (`paths::stranded_stores`) |
| Temp directories left by test runs | `94524a7` | The lineage, stage-context, run-ledger, executor-partition and p2p-CLI test helpers and one bench hold their guards. Their tests leave 0 entries in a fresh `TMPDIR`; the old lineage helper alone left 12. Whole-suite counts below |
| 17. TPE suggested a uniform dim's exclusive bound | `0e0d934` | TPE and multivariate TPE keep suggestions in `[low, high)` |
| 18. `blut-notify` jobs dir ignored `$XDG_DATA_HOME` | `17c48ea` | Follows `blut::paths::jobs_dir`; the SLA log honours `$BLUT_SLA_PATH` |
| `blut log` showed `.tmp-…` attempt paths | `bea69a0` | A committed attempt's `saved` / `done` paths are shown under `stages/<idx>-<stage>`; a path still in an attempt directory is marked |
| `recipe run --dry-run` said 2.0G for a 256 MiB envelope | `dad25e4` | 2.0G was the legacy whole-GiB estimate with the floor applied to every plan, not what a launch bills; the dry run now reports the launch's footprint and its basis (declared envelope or compatibility floor) |
| 13. Empty Starlark plan exited 0 | `10bcbe2` | `blut-dsl` refuses a `build(args)` that adds no stage; the engine already refused an empty PlanSpec at compile (now pinned by a test) |
| 12. `blut-notify` summaries | `d634b51` | Summaries name the event kind, the stage and the error (200 characters); the error is withheld for restricted data on every sink |
| 9. `cloud smoke` fixed `/tmp` store | `61efa9a` | Default store is a fresh private temp directory, removed afterwards |
| Calibration claim in `examples/first_cookbook.rs` | `234d3d4` | Corrected (the engine reads the measured-peak store; a cookbook runner fills it). Recording peaks from the executor was not a small change: the stage owns its child, the key needs the launch context |
| cgroup-v2 orphan | `7f7f6c5` | Leaves are recorded with the job (path and inode); `blut cancel` stops them (SIGTERM, then `cgroup.kill`) and `plan resume` refuses while one is populated. Live-tested in a delegated user service |

Temp directories, measured with `TMPDIR` set to a fresh empty directory: the
default suite leaves 0 entries. The `p2p` suite leaves 113, all from test
helpers under `src/p2p/`, which was being changed on the mesh dispatch branch
and was left alone (item 19 below): `transport` 10, `node` 39, `mesh_runner`
25, `objects` 39, and 0 from everything else, integration tests and
`cli::p2p` included.

## Open items

Still open, by severity.

**Medium**

5. **Timing-sensitive tests on a loaded host.** With load average 45 to 227
   from other work on the same machine, suites fail on deadlines that are not
   about the code under test: p2p tests whose QUIC handshake reply missed its
   10 s window, `dag_opt_advanced` 5 s event deadlines, and
   `conditional_execution`, whose 500 ms plan-deadline tests wait on signals
   with no hang guard: when the deadline passes before the stages start, the
   test waits forever. Seen in base `d297e11` too (2 of 4 runs at load ~100).
   Each passed when re-run.
16. *(Resolved, `dc4c837`: a startup warning names each stranded store and
    its variable.)* **Migration of an existing `LAMU_TRAIN_DATA_DIR`.** An install that
    already set it (it used to hold only materialized training data) now sees
    empty jobs and registries until the per-store variables point back at the
    old ones. Nothing warns about it; whether to detect and warn is an owner
    decision.

**Low**

9. *(Resolved, `61efa9a`.)* `cloud smoke` defaults `--store` to `/tmp/blut-cloud-smoke`, a fixed path
   in a shared directory.
11. `p2p peers list` shows reputation 0.50 and zero completed tasks for peers
    that ran smoke dispatches successfully.
12. *(Resolved, `d634b51`.)* `blut-notify` summaries say only "status event for job X", not which
    stage or what error.
13. *(Resolved, `10bcbe2`.)* A Starlark script that never calls `add()` emits an empty PlanSpec and
    exits 0.
14. `README.md` ("v0.2.0-alpha.1 — current development preview") and
    `ROADMAP.md` ("between `0.2.0-alpha.1` and a 1.0") still name alpha.1,
    while the crate and the README install line are alpha.2. Left to the
    release commit.
15. `blut-standard` registers stages but no recipes, so `hpo` and `study`
    cannot be tried with the standard cookbook alone; the `blut-core` recipes
    start a Python trainer and were not run here.
17. *(Resolved, `0e0d934`.)* TPE can suggest a value equal to a `uniform` dim's exclusive upper bound
    (`wd = 1.0` for `uniform(0, 1)`): its candidates are clamped to `high`
    inclusive.
18. *(Resolved, `17c48ea`.)* `blut-notify`'s default jobs directory is `$HOME/.local/share/lamu/
    train-jobs` even when `$XDG_DATA_HOME` moves the engine's elsewhere.
19. Test helpers under `src/p2p/` still leak their temp directories
    (`mem::forget`): `transport.rs` `bind_loopback_server`, `node.rs`
    `spawn_node`, `gossip_node` and two inline uses, `mesh_runner.rs` two
    `tmp()`, `objects.rs` `tmp()`. A `p2p` or `s3` suite run leaves 113
    entries. The fix is the one `cli::p2p`'s `mesh_node` got: hand the guard
    back to the test.
