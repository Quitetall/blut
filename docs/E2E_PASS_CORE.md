<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# End-to-end pass: core commands (0.2.0-alpha.2 candidate)

What this records: every subcommand of the core half of the CLI was run
through a shipped binary against real state, and its output was checked
against that state, not only its exit code. Candidate: `d24dd2c`. The other
half (hpo, registry, study, model, experiment, dataset, connectors, policy,
cloud, tui, update) was covered separately.

## Setup

- **Binaries.** `blut-core` and `blut-standard` from the public cookbooks
  repository (branch `agent/slurm-launch-placement`), with `blut` pointed at
  the candidate by path; a bare engine driver calling
  `blut::cli::run(Registry::new())`; and a throwaway test cookbook whose
  stages write real files, emit metrics through `StageStep`, sleep (to be
  cancelled), fail on demand (to be resumed), and emit `checks.jsonl` (to feed
  the built-in checks). No model was trained and no GPU was used.
- **State.** An isolated `HOME`, `XDG_*` and `LAMU_TRAIN_DATA_DIR`, so jobs,
  cache, lineage, ledger, partitions, footprints, triggers and SLA files lived
  in a scratch tree. The run ledger was filled by the cookbook's own Python
  writer (`blut_core.run_ledger`).
- **Toolchain.** Rust 1.88. The host was heavily loaded by other work during
  the pass (load average about 95 on 32 cores, I/O pressure about 80 %), so
  durations below are not performance data.

Results: **P** passed as documented, **F** a bug or footgun fixed here (commit
listed), **O** an open item (see the end), **N** not exercisable here.

## Matrix

| Command | What was run | Result | Evidence | Fix |
|---|---|---|---|---|
| `jobs` / `--json` | empty store; after done, failed, cancelled, running jobs | F | pid showed `-` for a running in-process job; columns overflowed 25-char ids | a740a3b |
| `cancel` | unknown id; running job with `--grace 2s` and default grace; already-finished job | F | cancelled run was indexed `outcome=failed` beside state `cancelled` | 07e3c7d |
| `log` / `--tail` / `--json` | unknown id; in-process jobs (no trainer stream) | P (N for trainer output) | prints a hint naming `dag`; trainer `StatusUpdate` rendering needs a Python trainer | — |
| `runs diff` / `--all` / `--json` | two jobs with different args; same job twice | F | `--json` outcome was `"done       -"` | a740a3b |
| `lineage show` / `--json` | done, failed, resumed, cache-hit jobs | P | per-node hashes match `dag --json` and the sidecars | — |
| `lineage trace` / `--json` | full hash and prefix of a reused artifact | P | chain names producing jobs, git SHA, host | — |
| `lineage graph` / `--json` | prefix and full hash | F | a prefix was "unknown artifact" | 85c73c0 |
| `lineage card` / `--json` | prefix and full hash | F (prefix), O (run choice) | see open item 4 | 85c73c0 |
| `lineage diff` / `--json` | two runs differing in args | P | arg and metric deltas match `args.json` and the metric store | — |
| `lineage reindex` | fresh install; after 27 jobs incl. plan-file jobs; from another git checkout | F | failed on a fresh install; skipped plan-file jobs; rewrote every run's git SHA to the indexing directory's HEAD | a740a3b, 5ca80ba |
| `lineage freshness` / `--data-version` / `--json` | from three directories; re-run afterwards | F | said `STALE — a re-run re-executes`, then the re-run hit the cache 2/2; `--data-version` with nothing to compare said `FRESH` | 5ca80ba |
| `ledger list` / `--tier` / `--json` | 9 runs written by the Python writer; bad tier | F | a collected run listed as live `Scratch`; tiers Debug-cased | ac15aa5 |
| `ledger show` / `--json` | exact, prefix, ambiguous prefix | P (O: text form) | text output is Rust `Debug` | — |
| `ledger promote` | blank reason; open run; completed run | P | only one completed, uncurated run promotable | — |
| `ledger gc` / `--commit` | dry run, commit, rerun | P | tombstone appended once | — |
| `ledger compare` / `--metric` / `--alpha` / `--json` | 3v3 arms; 3v1; unknown, empty, ambiguous prefix; unknown metric; `--alpha 2` | F | `--alpha 2` printed `SIGNIFICANT`; `""` pooled every run; `--json` printed `NaN` | ac15aa5 |
| `ledger verify` / flags | clean ledger; `--require-ledger-rows-resolve` | F (help), O | no help text; see open item 6 | ac15aa5 |
| `results` / `--json` / `--metric` / `--maximize` / `--minimize` | done, failed, resumed jobs; unknown metric | F | path labelled `ckpt` was the output sidecar | a740a3b |
| `errors list` / `--json` | engine-only registry | F (help) | help said no domain is registered | a740a3b |
| `errors show` / `--json` | done; failed; failed then resumed; blocked check | F | resumed job showed the recovered failure as terminal | c0ff982 |
| `checks report` / `--json` | warn and block breaches through `check_jsonl` | P | both breaches reported with node and code | — |
| `compare` | two runs with metrics; same run twice | F | `git=` came from the indexing directory | 5ca80ba |
| `dag` / `--json` | latest; done, failed, running, killed, resumed | F | a node that failed and then succeeded on resume stayed `failed` | c0ff982 |
| `schedule list` | empty; after a refused install | F | listed a timer systemd had refused | 55102ed |
| `schedule install` | unknown recipe; no calendar; bad calendar; bad args; valid | F, N | failed install left unit files; enabling against the real user manager was not run (see open item 9) | 55102ed |
| `schedule uninstall` | never-installed recipe; installed files | F | reported "removed timer" for nothing | 55102ed |
| `partition define` | `--dim` sets; re-define; unknown recipe; non-arg axis; repeated values; malformed `--dim` | F | `bogus=1,2` accepted then every cell failed; `seed=1,1` made two cells; re-define silent | 1c26552 |
| `partition list` | empty; three sets | P | — | — |
| `partition status` / `--json` | missing, materialized, failed cells; from another directory | F | a fresh cell read `stale` from another checkout | 5ca80ba |
| `partition backfill` (`--partitions`, `--missing`, `--stale`) | explicit cell; missing; stale; undeclared key | P | cells run as ordinary jobs; undeclared key refused | — |
| `sensor list` / `--json` | idle; while a job holds the lock | F (text) | detail cited a nonexistent `--allow-evict` | a740a3b |
| `sensor eval` / `--json` | `scheduler-lock` idle and held; unknown sensor | P | exit 0 READY, 1 WAIT | — |
| `sensord` (`--once`, `--triggers`, `--cli`, `--interval`) | file-drop and spool triggers; missing and invalid config; bad `--cli` | F | started with a missing `--cli` and retried forever; `--once` exited 0 with every dispatch failing | 55102ed |
| `sla check` (`--rules`, `--out`, `--json`) | runtime and freshness rules over 27 runs; missing rule file | F | evaluated `0 run(s)` always; explicit missing file meant "0 rules"; after the fix, freshness judged per run breached forever | 5ca80ba, a740a3b |
| `footprint list` / `--json` | after runs | P (empty) | no test stage declares an envelope, so nothing is calibrated | — |
| `footprint forget` | no args; key and `--recipe`; unknown key | P | — | — |
| `plan resume` (`--shared-cache`, `--sync-io`) | failed job (still failing, then fixed); done job; running job; plan-file job; unknown id | P | cached stages hit, failed stage re-ran; running job refused; plan-file job refused with a reason | — |
| `plan inspect` | with and without `--args`; unknown recipe; bad args | F | required `--args` | a740a3b |
| `plan check` / `--json` | valid spec; unknown stage; kind break; unknown field; bad args; broken JSON; missing file | P | refusal names the cause | — |
| `plan run` (`--dry-run`, `--no-cache`, `--tenant`, `--shared-cache`) | test cookbook spec; a real cookbook stage (`load_dataset`); bad stage; unknown tenant | P | job id printed; `dag`, `lineage`, `artifact` work on it | — |
| `plan publish` / `--tenant` | new; republish; other tenant | P (O) | see open item 1 | — |
| `plan promote` | full fingerprint; prefix; cross-tenant; bad pointer | P | prefix refused (needs the full fingerprint) | — |
| `plan rollback` | with and without history | O | see open item 2 | — |
| `plan history` / `--tenant` | after promotes and rollbacks | P | — | — |
| `recipe list` / `--json` | test cookbook; `blut-core`; `blut-standard` | F | empty catalog printed a bare header; category column overflowed | a740a3b |
| `recipe show` | test and real cookbook recipes; unknown | P | — | — |
| `recipe declare` (file, `--run`, `--args`, flags, `registry://`) | `.toml` and `.json`; discovery dir; `--run`; resume of the result; registry pointer | F, O | launch-only flags without `--run` were ignored; registry pointer: open item 1 | a740a3b |
| `recipe run` (`--args`, `--shared-cache`, `--no-cache`, `--launcher`, `--tenant`, `--experiment`, `--dry-run`) | cold, warm shared cache, forced; lock held; slurm with no client; bad launcher and tenant | F | lock-refused launch left a `failed` job; `--launcher slurm` without Slurm ran locally and reported success | 07e3c7d, 55102ed |
| `recipe run` sweep (`--config-dir`, `--config-name`, `--config-key`, `--set`, `--sweep`) | flat and nested configs, dry run and real | F | `--set lr=…` (the flag's own example) and `--sweep seed=1,2` changed nothing: two identical runs, then "already complete" | 2e0e9ef |
| `cache show` | before and after shared-cache runs | F | 3.5 KiB printed as `0.00 GiB` | a740a3b |
| `cache prune` / `--max-gb` / `$LAMU_CACHE_MAX_GB` | under and over cap; invalid env | F | "pruned 0.00 GiB" after deleting the whole cache | a740a3b |
| `cache stats` / `--json` | cold, warm, resumed jobs | P | hit counts match `stage_skipped` events | — |
| `artifact ls` / `--json` | done and failed jobs | P | matches the sidecars on disk | — |
| `artifact inspect` / `--json` | reused artifact by prefix and by full hash; unknown | F | could not inspect any artifact a cache hit reused, even by full hash | 85c73c0 |

53 subcommands (flags counted with their command): 21 passed as documented,
30 had at least one bug or footgun fixed here, and 2 (`plan publish`, `plan
rollback`) work as coded but rest on owner decisions. Some fixed rows also
carry an open item.

## Open items

Items 1-5 and 8 were resolved after this pass (owner decisions for 1-3):
1 by `efca0f2` (one default tenant, `default`), 2 by `c80cf82` (rollback walks
the promotion history), 3 by `f332d70` (runtime and freshness judge the newest
run per tenant), 4 by `6dcb2c3`, 5 by `934b929` (the contract is documented;
`JsonlArtifact::from_file` makes the adapter one call) and 8 by `b391db8`
(every wait in `tests/conditional_execution.rs` is bounded). They are kept
below as found.

1. **Owner decision — deployment tenant vs run tenant.** `plan publish` and
   `plan promote` default to tenant `shared`; `recipe declare` defaults to
   `default`, so `recipe declare registry://plan@prod --run` reports "no
   deployment pointer". With `--tenant shared` the pointer resolves, and the
   launch is then refused because `shared` is not a quota tenant. The
   documented publish → promote → run path does not work with default
   configuration. Sensord dispatches registry pointers the same way. High.
2. **Owner decision — rollback history.** Repeated `plan rollback` toggles
   between the last two targets instead of walking back the history. Medium.
3. **Owner decision — SLA freshness semantics.** Once runs became visible to
   SLA evaluation, a freshness rule judged per run breached for every run
   older than the window, forever. This pass changed it to judge the newest
   measured run per tenant (5ca80ba); confirm that is the intended contract.
   `max_runtime_secs` stays per run, so one historical slow run keeps `sla
   check` non-zero permanently. Medium.
4. **`lineage card` and `lineage graph` attribute an artifact to the newest
   job holding it**, often a full cache hit with no metrics of its own; the
   card's metrics then describe that job, not the producing run. The index
   does not record whether a job's artifact was produced or reused. Medium.
5. **The engine's checks stages accept only `checks.jsonl`**, which no shipped
   cookbook stage emits (the standard cookbook emits `dataset.jsonl`), so they
   cannot be placed after a real data stage. Medium.
6. **`ledger verify --require-ledger-rows-resolve`** reads one product's
   `docs/TRUTH_LEDGER.md` (found from the working directory or
   `$LAMQUANT_META_ROOT`) and reports "LamQuant ledger run" failures: domain
   logic in the engine. Owner decision. Low.
7. **Cache identity does not include the cookbook's code version.** A stage
   cache key folds in the engine's build commit (`BLUT_GIT_HASH`, stamped
   `unknown` when the engine is built from crates.io without git) plus an
   optional per-stage code fingerprint. A cookbook commit that changes a
   stage's logic without bumping its `SCHEMA` or providing a fingerprint is
   served the old cached output. Medium. (Addressed after this pass: keys now
   fold every registered cookbook's name, version, commit and dirty digest;
   see `framework::code_identity`.)
8. **Test reliability on a loaded host.** Under load (average about 95),
   `tests/conditional_execution.rs` hung twice (16 tests idle behind
   `SPECULATION_TEST_LOCK`, which has no hang guard) and several executor and
   p2p tests exceeded their 60 s / 10 s guards; each passed when rerun alone,
   and the final default run passed in full. The p2p test
   `gossip_lets_a_node_seeded_with_one_peer_reach_the_rest` failed once with
   `peer_trust == Some(Anonymous)` where it expects `None` (an assertion, not
   a timeout) and passed three times alone; it looks like state shared
   between concurrent tests. Medium for CI.
9. **Not exercised:** `schedule install` enabling a timer in the real user
   systemd manager (the isolated `HOME` cannot reach it, and inspecting the
   real manager was not permitted); `log` rendering of trainer progress and
   `jobs` output/loss columns (need a Python trainer stage); footprint
   calibration entries (no test stage declares an envelope); scheduled units
   carry no environment, tenant or cache flags from the installer.
10. **Smaller:** `ledger show` text is Rust `Debug`; `plan promote` needs the
    full fingerprint while other commands take prefixes; `plan publish
    --tenant` accepts any string; `check_jsonl` and `load_dataset` outputs
    warn "not portable and will not be cached" on every run.
