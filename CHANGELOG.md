# Changelog

All notable changes to the **blut** engine crate are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`blut` is the domain-agnostic engine (Stage → Plan → Recipe → Registry + CLI +
resource broker). The user-facing binaries live in separate cookbook crates that
depend on this crate.

## [Unreleased]

### Added
- `docs/adr/`: the 41 engine-facing architecture decision records, federated
  from the project's decision log with private material redacted inline as
  `[redacted: <category>]`, plus an index. "ADR NNNN" citations in the engine
  now resolve to a document in this repository; `book_src/adrs.md` points at
  the index instead of describing the gap.

### Changed
- Citations of decisions that were not federated (product, codec and platform
  records) are rewritten as the rationale they stood for. Citations written
  before the decision log renumbered two records now use the current numbers:
  experiment-execution UX is ADR 0156 (was 0071), cookbook taxonomy is ADR 0055
  (was 0051).

## [0.2.0-alpha.2] — 2026-10-09

`blut-types`, `blut`, `blut-dsl`, `blut-notify` and `blut-tui`, all at
0.2.0-alpha.2. `blut-types` carries no code change; it moves because the
train shares one exact version.

### Added
- `BLUT_P2P_CONGESTION=bbr` selects BBR congestion control for P2P QUIC
  connections (default `cubic`; any other value is refused). On a real
  Australia-Korea-Germany path at 5% random loss, 1 MiB each way took 14.2 s
  with BBR and did not finish in 300 s with Cubic. quinn marks its BBR
  experimental, hence opt-in.
- `blut p2p node --probe`: dial every `--seed`, dispatch one signed task to
  each, verify each signed reply against the seed's pinned key, print one line
  per seed, and exit 0 only if all answered. Until now no shipped command could
  dispatch over the mesh: `p2p node` connected to its seeds and waited.
- `s3` cargo feature: network object storage. `ObjectStore::from_url` opens
  `s3://bucket/prefix` (AWS S3, or an S3-compatible endpoint that supports
  conditional writes) and `file://` stores, and `blut cloud --store` accepts a
  URL as well as a directory. Credentials come only from the standard AWS
  environment variables, never from flags. Validated end to end against MinIO.
  An endpoint without conditional writes (Backblaze B2) is refused with an error
  that says so, instead of a generic 501 after ten retries; objects are never
  written without the create-only guarantee.
- `blut update` reports whether the binary cargo installed (per its install
  ledger) is behind the newest version on crates.io, and with `--yes` runs the
  pinned `cargo install --locked`. A stable install is never moved onto a
  prerelease.

- Windows Job Object containment (`containment::windows::WindowsJobObject`),
  selected by `containment_for()` on Windows or `BLUT_CONTAINMENT=windows-job-object`.
  One job per run: `mem_max` (else `$MEMMAX`) becomes a tree-wide committed-memory
  cap (`JOB_OBJECT_LIMIT_JOB_MEMORY`; an allocation past it fails rather than
  killing the tree), `cancel` is `TerminateJobObject`, the peak is
  `PeakJobMemoryUsed`, and closing the job kills whatever is left in it.
  `mem_high` and `swap_max` are not mapped; `CapSpec` has no CPU field, so no
  CPU rate limit is set. CI-gated on `windows-latest`.
- `Containment::after_spawn` (provided, default no-op) and
  `containment::spawn_attached`. The Windows backend spawns the child suspended
  and assigns it to the job here, so a Windows caller must use one of them; a
  skipped call leaves the child suspended, never running uncapped.

### Changed
- **Breaking (P2P API):** `p2p::transport::WireMessage` gains `Goodbye`, the
  orderly end of a session, and is now `#[non_exhaustive]` so later messages
  are not breaking. Code outside `blut` that matches on it needs a wildcard
  arm. This is the only finding `scripts/semver_checks.sh` accepts against
  0.2.0-alpha.1, recorded in `scripts/semver_exemptions.tsv`.

### Fixed
- The Slurm launcher's `sbatch` path (`SlurmLauncher::submit_async` and its
  `RemoteJob`) now works on a real cluster. Against a two-node Slurm 21.08
  cluster the previous code failed 7 of 8 live checks without accounting and 6
  of 8 with it:
  - Job state came only from `sacct`, which fails outright on a cluster
    without `slurmdbd` and prints nothing for a job it has not recorded yet, so
    `stream` gave up on every job straight after submitting it. State now comes
    from `scontrol show job`, with `sacct -X -P` only for a job the controller
    has purged; a transient query failure is retried for five minutes.
  - A multi-node batch job ran its command once, on the first node, with
    `NODE_RANK=0`. The command now runs as a job step on every task, each
    reading its own `NODE_RANK`.
  - A job that waited in the queue for an hour was reported as timed out while
    it was still queued. Streaming now waits until the job ends; the job's
    `--time` limit bounds it.
  - When Slurm reused a job id, the previous job's `slurm-<id>.out` was
    streamed as the new job's output. The log is now made absolute at submit
    time, honours `--output` and `--chdir` passed in `extra`, and is ignored
    until this job writes it.
  - `CANCELLED by <uid>`, `PREEMPTED`, `COMPLETING`, `REQUEUED` and the other
    states Slurm reports are recognised rather than ending the stream as
    unknown.
  - A job requeued after a node failure went back to waiting without a word.
    The stream now notes a requeue, suspension or hold, and anything other
    than ordinary queueing, once each with Slurm's reason, and notes when the
    job is running again.
  `tests/slurm_live.rs` holds the live checks.
- A `BLUT_SLURM_*` or `BLUT_RAY_NUM_GPUS` value that is not a positive integer
  (`BLUT_SLURM_NODES=two`) was silently ignored, so the job ran with the default
  allocation. `blut recipe run --launcher` now refuses it before any work starts.
- `SlurmLauncher::gpus` was documented as a per-node count; it is Slurm's
  `--gpus`, the job's total. The documentation now says so and points to
  `--gpus-per-node` for a per-node count.
- `EvalReport` artifacts can cross the stage boundary. Their free-form
  `metrics` (a `serde_json::Value`) could not be decoded from the engine's
  bincode transport, so every stage that produced one — any evaluation stage in
  any cookbook — failed at finalization with "artifact handle did not decode as
  the selected stage role", after its evaluation had already run. Binary
  encodings now carry the metrics as a JSON string; JSON is unchanged.
  `EvalReport::SCHEMA` is 2.
- The Ray launcher's job-status parser matched a line format Ray never emits,
  so it could never report success or failure. It now parses Ray 2.x output,
  and stops at `Status message:` so a failed job's log tail containing
  "succeeded", or a child job's `Status for job …: SUCCEEDED` line, cannot
  read as success.
- Enabling `s3` alongside `p2p` made rustls panic on the first QUIC
  connection ("Could not automatically determine the process-level
  CryptoProvider"), because `object_store/aws` pulls in a second provider.
  The P2P transport now builds its TLS configs with `ring` passed explicitly
  and never installs a process-wide provider, so an embedder that installs
  its own later is unaffected.
- `blut cancel` on a job whose stages run in the blut process (no trainer
  subprocess) printed "cancelled." and exited 0 while the job ran on to
  `done`. Each job now records the blut process executing it, and `cancel`
  stops that process through its own SIGTERM handler, escalating to SIGKILL
  after `--grace`. A process running several jobs (a sweep or backfill) is
  not stopped to cancel one of them; the refusal names the batch.
- `blut cancel` on a finished job rewrote `done` or `failed` as `cancelled`.
  It now reports that the job already finished and changes nothing.
- A job whose blut process was killed (SIGKILL, OOM killer, power loss), or
  that exited without being able to write its final state (a full disk),
  stayed `running` in `blut jobs` forever when no trainer subprocess had
  recorded a pid. It is now reported `failed`.
- `blut plan resume` on a job that is still running started a second executor
  in the same job directory, and when the scheduler lock refused it marked the
  live job `failed`. It now refuses with the running pid. A resume refused by
  admission or the scheduler lock no longer rewrites the job's state.
- Ctrl-C or SIGTERM during `blut plan resume` killed blut outright, leaving
  the job `running` and any trainer subprocess (which runs in its own session)
  orphaned. Resume now installs the same cancel handler as `recipe run`.
- `blut update` printed `cargo install --path /my%20dir` for a checkout whose
  path contains a space (cargo records it as a URL), and reinstalled without
  the features the binary was installed with, so `--yes` on a
  `--features cloud` install silently removed `blut cloud`. The path is now
  decoded and shell-quoted, and the feature selection recorded in
  `.crates2.json` is passed back to `cargo install`.
- A misspelled recipe or stage argument was dropped without a word: serde
  ignores unknown keys unless the args type opts out, so
  `--args '{"epoch": 50}'` against an `epochs` field trained on the default.
  Recipe args (`recipe run`, `--set`, sweeps, HPO) and stage args in
  `.toml`, `.json` and `.star` plans now refuse a key the args schema does not
  declare, naming the accepted keys. Schemas that admit extra keys (a
  flattened map) are unaffected. A `.toml` recipe also refuses unknown keys
  of its own, so `arg = {…}` for `args` is an error instead of a stage run
  with no arguments.
- Stage args in `.toml`, `.json` and `.star` plans were checked only when the
  executor reached each node, so `plan check` and `recipe declare` reported
  success for a plan whose last stage could not decode its args, and the
  earlier stages ran (for hours, in a training plan) before it failed. They
  are now checked when the plan is built.
- A plan node that omits `args` passed JSON `null`, which an args struct
  whose fields all have defaults rejects at run time. `null` and `{}` are
  now each tried in place of the other, as recipe args already were.
- `blut cache prune --max-gb=-1` (or `NaN`) cast the cap to 0 bytes and
  emptied the whole shared cache, then printed the cap it was given. A
  negative or non-finite cap is now refused, and so is an unparseable
  `LAMU_CACHE_MAX_GB`, which silently meant 50 GiB. `--max-gb 0` still
  empties the cache on purpose.
- `blut policy enable` printed a crontab line running `blut auto`, a verb
  `blut` does not have, so the advertised auto-trigger failed every 30
  minutes into its log. It now prints a line that gates on
  `blut sensor eval auto-train-policy` and runs a recipe the operator names,
  with paths shell-quoted.
- Error messages named commands that do not exist: `lamu-train jobs` and
  `lamu-train policy enable` (now `blut jobs`, `blut policy enable`), and
  `--allow-evict` when the scheduler lock is held (now: wait, or
  `blut cancel <job>`). "recipe not in catalog" now points at
  `blut recipe list`.
- On/off environment switches parsed inconsistently. `BLUT_NO_CONTAIN` and
  `BLUT_AUTO_REBUILD` were on whenever set, so `BLUT_NO_CONTAIN=0` switched
  memory containment off and `BLUT_AUTO_REBUILD=0` switched rebuild-and-re-exec
  on; `BLUT_KILL_ON_NAN` and `BLUT_STRICT_DECLARATIONS` accepted only `1`. All
  of them, with `BLUT_STRICT_ADVISORY`, now read `1/true/yes/on` as on and
  `0/false/no/off` or empty as off, and warn on anything else.
- Path variables (`LAMU_TRAIN_JOBS_DIR`, `BLUT_HOME`, `BLUT_DSL_BIN` and the
  other `*_DIR`/`*_PATH`/`*_DB` overrides) were read as UTF-8 strings, so a
  non-UTF-8 value was ignored without a word and jobs, caches and registries
  went to the default location instead. They are read as OS strings now,
  except `LAMU_TRAIN_JOBS_DIR` and `LAMU_TRAIN_DATA_DIR`, whose paths are
  recorded in JSON: a non-UTF-8 value there is refused, naming the variable.
- `blut log` printed nothing, with exit 0, for a job whose stages run
  in-process: it renders trainer progress only, and such a job records stage
  events instead. It now says so on stderr and points at `blut dag <job>`.
- `blut schedule install` checked only that `--args` was JSON, so args the
  recipe rejects installed a timer that failed every time it fired, visible
  only in the journal. The args are now compiled at install. The generated
  `ExecStart=` also left the binary path unquoted (a path with a space ran
  the wrong program) and used shell escaping, which systemd does not
  understand; `%` and `$` in args were open to systemd's specifier and
  variable expansion. Words are now quoted and escaped per systemd.syntax(7).
- `RayLauncher::submit_async` built its own `ray job submit` argv without
  `--entrypoint-num-gpus`, so an asynchronously submitted job reserved no GPU.
  Both paths now share one argument builder. A non-numeric
  `BLUT_RAY_NUM_GPUS` (or any `BLUT_SLURM_*` count) was dropped without a
  word; it is still ignored but now warns, and an empty `RAY_ADDRESS` (or
  other launcher setting) counts as unset instead of producing `--address=`.
- `recipe run --sweep … --dry-run` listed the expanded combos without
  compiling them, so a sweep whose every combo would fail (a misspelled
  `--sweep` key) dry-ran clean. Each combo is now resolved and compiled, as
  the single-run `--dry-run` already did, and the dry run fails if any would.
- `blut plan resume` on a job launched from a plan file (`recipe declare
  --run`, `plan run`) failed with a bare "No such file or directory". It now
  says such a job has no recipe to re-compile and must be launched again.
- The crate builds on Windows. `python_kill::graceful_kill_group` had no
  non-Unix definition, the run ledger's non-Unix lock fallback did not
  type-check, and `blut-notify` took its cursor lock with `flock`
  unconditionally (on Windows it now holds the lock file open with share
  mode 0).
- On Windows a checked run-ledger append was refused outright, so a promotion
  or clear could never win; it now takes an exclusive `LockFileEx` lock (on one
  byte far past the end, so concurrent readers are not blocked).
- On Windows a scheduler lock left by a dead process could never be reclaimed:
  liveness always answered "alive". It now asks the OS whether the pid has
  exited; a pid it cannot open for any other reason still counts as alive.
- `blut plan resume` dropped the job's `--launcher`, so a Slurm job resumed
  on the local machine; a multi-node one failed at once ("multi-node training
  needs MASTER_ADDR"). The launcher is now recorded with the job and restored
  on resume. Jobs recorded before this resume locally, as before.
- `blut p2p node --seed addr@key` accepted only the 64-hex Ed25519 key, and
  the whole `pubkey` line that `blut p2p keys show` prints (128 hex) failed with
  "Invalid input length". Both are accepted now. A pin mismatch no longer
  names `--coordinator-pubkey` when the key came from `--seed`.
- `blut p2p connect` checked only the identity (Ed25519) half of
  `--coordinator-pubkey`. With the right first half and a wrong second (X25519)
  half, a worker connected, ran the task, sealed the result to a key nobody
  holds, and exited 0; only the coordinator saw "failed to unseal". The
  coordinator's TLS certificate now also publishes its X25519 key, and the
  worker refuses a mismatch before any work, naming both keys. A coordinator
  from an earlier release publishes none and is not checked. Found on a real
  Nevada-to-France run.
- P2P sessions under hostile network conditions (found on emulated latency,
  loss, partition, NAT and clock skew between containers, and on a real
  Oregon-to-France run):
  - A worker exited about 30 s after the smoke coordinator. The coordinator
    exited without a close the worker reliably received, so the worker waited
    out the QUIC idle timeout. Coordinators now end each session with a
    `Goodbye` on a reliable stream, close with an application code, and wait
    (at most 5 s) for the close to go out. `p2p serve` does the same on Ctrl-C.
    The worker acknowledges the `Goodbye` and lets the coordinator close
    first, so neither side waits on a close the other never sent.
  - `p2p serve` never exited on Ctrl-C: once the endpoint closed, its accept
    loop spun without yielding, flooding the log with "QUIC endpoint closed".
  - An idle worker was disconnected after 30 s, and a session did not survive
    a 20 s partition even after it healed, because nothing was sent between the
    heal and the idle timeout. Every P2P connection now sends a keep-alive
    every 5 s; the idle timeout stays 30 s.
  - A worker that lost its coordinator (partition, crash) printed "coordinator
    connection closed." and exited 0. It now exits nonzero and says why, for
    example "timed out: nothing heard from the other side for 30s".
  - A worker whose clock ran ahead of the coordinator's read the task's
    absolute deadline as already passed, so any stage that yielded was killed
    with a hard-deadline failure. When the clocks disagree by more than 5 s the
    worker now uses the coordinator's relative budget and logs the skew.
  - A transfer abandoned on its deadline kept sending: `send_blob` returned
    once quinn had buffered a chunk, so the data went on being transmitted for
    nobody, and under congestion it held the connection's CLOSE frame back
    (quinn gates it behind pending stream data), so the worker again waited
    out the idle timeout. Each chunk now waits for the peer's acknowledgement
    and is reset if the transfer is abandoned; the worker sees "stream reset
    by peer" at once.
  - The coordinator's connect hint named its bind address (`0.0.0.0:9320`); it
    now prints `<this-host>:9320` for a wildcard bind.
  - Dial errors say what to check: an unanswered dial names the 30 s bound and
    the UDP port; a wrong `--coordinator-pubkey` names the presented and the
    pinned key instead of "ApplicationVerificationFailure"; a pubkey of the
    wrong length says 128 hex characters are expected; a lost transfer names
    the connection's close reason. `p2p connect` accepts a host name.
  - One failed handshake (a worker with the wrong pin, a port scanner) made
    `p2p serve --smoke-stage` exit; it now reports it and keeps waiting. The
    application handshake after QUIC connects is bounded to 10 s, so a silent
    client cannot hold the accept loop.

### Security
- rustls-webpki 0.103.16 and webpki-roots / webpki-root-certs 1.0.10 in
  every lockfile, closing RUSTSEC-2026-0381, -0382 and -0383 (distrusted
  emSign intermediates could be trusted). `blut` now requires rustls-webpki
  0.103.16 or later.
- rustls 0.23.45 everywhere, closing RUSTSEC-2026-0285 (TLS 1.3 handshake
  messages accepted across encryption-level boundaries). `blut-tui`
  0.2.0-alpha.1 shipped a lockfile pinning 0.23.43, which `cargo install
  --locked blut-tui` would use; 0.2.0-alpha.2 does not.
- `faster-hex` floor raised to 0.10.1 in `blut` and `blut-types`, closing
  RUSTSEC-2026-0306 (the safe `hex_decode_unchecked` AVX2 path read past its
  input). Every committed lockfile now resolves 0.10.1.

## [0.2.0-alpha.1] — published 2026-08-27

Published to crates.io from two commits, recorded here because the tarballs are
the only place that pairing is visible: `blut`, `blut-types` and `blut-notify`
from `e34c557`; `blut-dsl` from `ffecee5` on 2026-09-06. `blut-graph-core` is
not part of this train and released separately as 0.3.0 from `e01f32f`.

### Changed
- **Version truth repair.** The package family is reset to
  `0.2.0-alpha.1`. The local 1.x tags are retained as internal milestone
  history; they were not crates.io releases. A real 1.0 remains gated on the
  M6 release train.
- **Public preview boundary.** `blut-types`, `blut`, and `blut-dsl` form the
  publishable preview. Deprecated worker and experimental operator crates are
  explicitly unpublished. Network object-store support is deferred; the cloud
  queue keeps its local-filesystem proof path.
- **MSRV.** Public preview crates now require and test Rust 1.88. The
  unpublished operator requires Rust 1.89.
- **Package hygiene.** Crate contents use a fail-closed allowlist; internal
  agent memory, deployment fixtures, and private host paths are excluded.

### Added
- **Typed dynamic DAGs (ADR 0078).** `blut recipe declare` now accepts a
  `.json` [`PlanSpec`](API.md) (arbitrary kind-checked DAG) and a `.star`
  Starlark script alongside the existing linear `.toml`. `PlanSpec` is the
  engine-native, versioned plan IR — also the door for a future Python SDK.
  - `CompiledPlan::from_erased_graph` extends the before-execution kind check
    from linear chains to arbitrary fork/merge DAGs.
  - **`map_output`** typed runtime fan-out: a stage whose output is a
    `ListOf<E>` fans out one sub-plan (template) per element, executed on the
    parallel executor's completion seam with cache-stable shard keys.
  - The Starlark front-end ships as a SEPARATE binary, `crates/blut-dsl`
    (`blut-dsl <script.star> --args <json>` → `PlanSpec` JSON), which the
    engine shells out to. `starlark` is kept out of the engine binary because
    it forces `serde_json/arbitrary_precision` — incompatible with the
    engine's internally-tagged enums under Cargo feature unification.
- **Runtime governance.** Tenant-namespaced cache and admission, capability
  RBAC, audited `SecretRef` handling, deployment/model registries, built-in
  fail-closed checks, typed connectors, model cards, provenance graphs, and
  run-diff reporting.
- **Scheduling and recovery.** GPU-aware per-device admission, per-node retry
  and timeout overrides, step-granular resume state, partition status cells,
  user-priority DAG scheduling, and opt-in auto-tune proposals.
- **Engine console.** Live run, broker, cache, lineage, mesh, privacy, recipe,
  and plan-builder views plus cookbook TUI extension hooks.
- **Public release gates.** Rustdoc with warnings denied, declared-MSRV builds,
  package-content checks, dependency/advisory/license policy, secret scanning,
  and in-tree public-example coverage.
- ADR 0094 eventing is complete: file-drop, threshold-spool, and cron triggers;
  signed/replay-safe webhooks; durable exact-admission acknowledgement; measured
  SLA freshness; and a rule-driven notification daemon with Slack, Discord,
  ntfy, SMTP, and exec sinks.

### Fixed
- P2P certificate pinning, signed-task argument coverage, authenticated receive
  ordering, encrypted blob transfer, dispatch races, and peer-registry
  persistence now fail closed.
- Cache atomic writes preserve `fsync` failures; containment cleans failed
  cgroup setup; lineage, tenant, registry, RBAC, connector, and auto-tune edge
  cases now preserve their safety invariants.
- Public documentation, packaged links, version claims, and rustdoc links match
  the library-only preview surface.
- Web job/status/SSE exports now fail closed for restricted tenants and unknown
  GETs no longer create job directories.
- SLA breach rows and webhook events deduplicate durably across repeated checks
  and daemon restarts.

### Security
- Deprecated worker job IDs are validated as flat ASCII identifiers at API and
  queue boundaries; duplicate submissions cannot overwrite files. Its
  plaintext prototype now refuses every non-loopback bind.
- Absolute source paths are no longer embedded by default. Local stale-source
  detection requires explicit `BLUT_EMBED_SRC_DIR=1` opt-in.
- Locked dependencies were updated for active RustSec findings. `bincode 1.3`
  remains temporarily under a documented unmaintained-only exception because
  replacement changes cache and P2P wire formats; migration is required before
  1.0.

## [1.0.0] — 2026-06-19 (historical internal milestone; never published)

This local milestone recorded the engine carve. It was not a public crates.io
release; the package version was later corrected to the `0.2.0-alpha.1`
development line. The architectural changes below remain historical facts.

### Changed
- **License: AGPL-3.0-or-later** (with a commercial license offered separately).
  Per-file SPDX headers on every source file; the canonical AGPL text in `LICENSE`.
- **Interactive TUI cockpit ships (default-on `tui` feature).** Bare `blut`
  opens a ratatui cockpit whose panels read the engine's OWN state — jobs, log,
  run DAG (`graph_snapshot`), lineage/provenance, artifacts, run history,
  metric leaderboard, compare, run metrics, and a registry-driven recipe
  catalog — instead of any domain file convention. `--no-default-features`
  drops `ratatui`/`crossterm`/`fuzzy-matcher` for a lean CLI-only binary (bare
  `blut` then prints help). The cockpit is fully domain-clean and covered by a
  headless render/interaction test harness (`tui --check` + 57 unit tests).
- **Proposed stable surface contract.** This milestone drafted the future 1.0
  additive-only contract; the current preview terms live in `API.md`.

### Removed
- The bundled generic-LLM cookbook (concrete stages + lamu/hf backends + the
  Python payload) — extracted to a separate `blut-backends` crate. The engine
  ships zero concrete stages/recipes/backends and zero Python.
- The `lerna` git-dependency (the sole `cargo publish` blocker) — replaced by a
  native, Hydra-compatible `config::hydra` compose module (serde_yaml only).
- The `meta_repo_root()` meta-repo assumption — the engine no longer assumes it
  lives inside a parent repo; meta-repo detection moved to the cookbook.

### Added
- `examples/first_cookbook.rs` — a runnable end-to-end demo (typed stages → plan
  → executor → content-addressed cache hit), doubling as an integration test.
- Off-systemd graceful degrade: containment falls back to a bare spawn (with a
  warning) when `systemd-run` is absent or `BLUT_NO_CONTAIN` is set.
- `run_contained.sh` embedded via `include_str!` so containment works from a
  clean install with no repo-layout dependency.
- A `macos-latest` CI compile-check; a `tui`-feature CI lint/test pass.
- crates.io publish hygiene: a trimmed package (`exclude` of dev artifacts) plus
  `homepage`/`documentation` metadata.

## [0.11.0] — 2026-06-17

**Feature-complete milestone (pre-1.0, ADR 0044).** This release lands the last
permitted breaking core edits before the optimization gate and 1.0.0 — all of
the BLUT-API phases A–G plus the cookbook taxonomy rebuild.

*Version note.* A forward minor bump for breaking pre-1.0 changes (semver 0.x).
The "feature-complete, pre-optimization" milestone semantics live in the 1.0
charter (ADR 0044), not in the version integer — `v0.10.0` was already
published, so the milestone moves forward rather than down to a `0.9.0` marker.

### Added

#### Cookbook taxonomy + ingredient registry (ADR 0055)

- **Course** — `RecipeCategory` renamed to `Course`, the culinary layer in
  BLUT ▸ Cookbook ▸ Course ▸ Recipe ▸ Ingredient, with new `Pretraining` and
  `Gate` variants and a compiler-forced exhaustive `order()` driving the TUI menu
  sort. `pub type RecipeCategory = Course` is kept as a transitional alias.
- **Ingredient registry** — `IngredientSpec` + a fail-closed
  `build_ingredient(kind, name, cfg, **extra)` over 13 kinds (data / sampler /
  preprocess / model / forward / loss / optimizer / scheduler / step / ema /
  eval / checkpoint / logging), generalizing the lamquant cookbook's optimizer registry. Each
  spec coerces a frozen dataclass config and declares whether its selection
  changes the cached artifact (`cache_relevant`).
- **Trainer decomposition** — the canonical trainers' sub-stages are extracted
  into byte-identical ingredients (optimizer construction, the WSD scheduler,
  EMA, the QAT generator step, the shared SNN SSM step, the joint / 4-state / MAE
  / teacher losses, the joint + 4-state evals, atomic-save / manager /
  durable-resume checkpointing, and the LMA data adapters), and all five canon
  trainers (`train_l3_teacher`, `pretrain_mae`, `pretrain_ssl_tueg`,
  `train_4state_controller`, `train_joint`) call `build_ingredient` instead of
  inlining the primitive.

#### Durable + auto resume (Phase D)

- **Durable resume.** A killed training run no longer restarts from scratch. The
  orchestrator owns a crash-gated *policy* (`framework::resume`): a stable
  per-config resume directory keyed on the stage cache key, a `state.json`
  run-state marker with a heartbeat, and a 5-row decision (resume an in-process
  retry or a crashed prior run; refuse a live concurrent run; start fresh
  otherwise). The trainer owns the *mechanics* (`durable_resume.py`): atomic,
  prev-rotated recovery checkpoints at each validation embedding the optimizer +
  RNG state, and a `--resume` that restores model + **optimizer** + RNG (a
  continuous loss curve, no cold-optimizer dip). Covers in-process OOM retry,
  cross-invocation re-run after a crash, and clean optimizer resume. A
  `no_resume` recipe arg forces a fresh start. (Epoch-boundary granularity;
  mid-epoch dataloader-position resume remains out of scope.)
- **Auto-resume on retry** — a retried stage reuses `decide_resume`; on
  attempt ≥ 2 with the same run id the decision is deterministically `Resume` and
  the trainer is relaunched with `--resume`.
- **`StageError::Diverged`** — a new transient, retryable error; a `KillOnNaN`
  divergence is reclassified as `Diverged`, so a NaN kill becomes a bounded
  retry → auto-resume instead of a hard failure.
- **Stage hooks** — defaulted `preflight` / `resume_handle` / `divergence_check`
  methods on the `Stage` trait, mirrored on the dyn shim (zero ripple to existing
  `impl Stage`).

#### Hyperparameter optimization

- Search space + samplers, `blut hpo run` fan-out, a median/percentile early-stop
  scheduler, **ASHA** async successive halving, **PBT**, a **TPE** Parzen-window
  sampler, trial tracking with `blut hpo show` / `best`, `Control::Spawn` runtime
  sub-plan injection, and a queryable graph backend behind `blut dag`.

#### Metric store + GPU saturation (Phase E)

- A queryable metric store — additive `metrics` / `gauges` tables folded from
  `status.jsonl` at run-end — with `blut compare A B`; a live GPU-saturation
  sampler recording `gpu_util / mem / temp / power` per run, deriving the
  `gpu_saturation` / `gpu_wasted` headline numbers with a starvation sentinel; and
  a per-epoch `BLUT_METRIC` val_r line surfaced as a `StageStep`.

#### Partitions, declarative recipes, sensors, scheduler (Phase G)

- A Dagster-class partition primitive with `blut partition {define,list,status,
  backfill}`; a `.toml` declarative-recipe engine over an erased stage registry;
  lineage **freshness** (flag outputs stale vs code/data drift); **named sensors**
  (the resume policy reconciled as a `Sensor`); and a per-device parallel backfill
  scheduler with multi-GPU locks (one cell per GPU).

#### Schema, TUI, launcher

- Schema-preflight args validation (`recipe show` lists typed fields; bad-type
  args rejected pre-dispatch); a real `blut tui --check` flag; in-TUI arg
  validation + auto-focus on the launched job's log; a `Launcher::capacity()`
  device probe.

### Changed

- **`code_sha` in the cache key (one-time global cache bust).** Stage cache keys
  and the nondeterministic-output fingerprint now fold a `code_sha` (cookbook git
  SHA ‖ resolved script content hash), so editing a trainer re-runs the stage
  instead of serving a stale checkpoint. Key tag bumped `v1 → v2` — a one-time
  global invalidation; pre-1.0, no migration is owed.
- Build hash re-stamped on commits, not only branch switches.
- Sequestered 15 dead scripts + 5 tests to `deprecated/`, relocated the optimizer
  modules under `lamquant/ingredients/optimizers/`, and retired the archived
  mamba-SNN + per-arch student trainers.

### Fixed

- `blut footprint list|forget` (audited calibration heal); the warm fork pool
  sized to the measured ~8 G/worker peak; per-worker peak-RSS logging
  (`LAMQUANT_RSS_DEBUG`); recovery checkpoints load with `weights_only=False`; and
  assorted HPO / executor / partition / scheduler review hardening.

## [0.10.0] — 2026-06-12

First tagged release. The engine has graduated from the early `0.1` prototype
to a memory-admitting, parallel DAG orchestrator with a distributed launch
seam, run lineage, and declarative scheduling.

### Highlights

- **Memory-admission resource broker (ADR 0046).** Every memory-spending
  stage runs inside a cgroup-capped systemd unit sized from a *scaling*
  footprint model, gated by a live admission probe. A wrong (too-small)
  estimate is caught by the cgroup as a clean unit kill — never the OS OOM
  killer taking the host.
- **Parallel DAG execution.** The `ParallelExecutor` runs independent stages
  concurrently (the long-missing piece), with capacity-aware memory admission
  so concurrency never overcommits the box.

### Added

- **Resource broker / footprint model** — system probe, conservative-high
  scaling RAM estimate (`workers × prefetch + tier + latent + batch + in_ch`),
  and a fail-closed admission gate that refuses or right-sizes a run against
  live free RAM (slice-1).
- **Calibration store** — measured cgroup peaks are recorded and MAX-merged per
  config key so the broker stops over-refusing the conservative estimate
  (slice-2); an OOM records an `OomCorrected` lower bound that the next
  admission **escalates** (self-heal), box-fit clamped so an escalation can
  never push the cgroup past the host.
- **Self-healing OOM retry** — `RetryOn::OutOfMemoryOnly`: an OOM-killed stage
  re-admits at the escalated cap on retry instead of re-sitting on the same
  failing cap.
- **Capacity-aware parallel admission** — the parallel executor reserves each
  stage's footprint before scheduling it concurrently.
- **`in_ch` (input-channel) footprint term** — a 168-channel fullband run is no
  longer billed like a 21-channel L3 run (the under-bill that caused
  admit-then-cgroup-kill on every fullband launch).
- **Distributed launch seam** — `blut … --launcher local|slurm|ray`. A
  backend-agnostic `WrappedCommand` + `LaunchTarget` routes a train stage to a
  cluster (which owns its own memory); `Local` keeps the cgroup-contained
  this-box path byte-identical.
- **LineageDB** — an embedded, rebuildable SQLite index of every run with git +
  hardware provenance captured on completion; `blut lineage trace <hash>` /
  `reindex`.
- **Declarative recipe schedules** — `blut`-managed systemd timers install
  recurring recipe runs (charset-validated unit names).
- **`register_recipe!` macro** + `schema_of` / `compile_erased` helpers for
  registering cookbook recipes with typed args.
- **Runtime DAG mutation** — `KILL-on-NaN` branch pruning: a stage emitting NaN
  cancels its branch + descendants and frees the GPU while sibling branches
  continue (a no-op policy path is byte-identical to static execution).
- **Subprocess liveness watchdog + heartbeat**, per-stage **retry policy** and
  **soft/hard timeouts**.
- **CLI** — `--json` output flags, `blut runs diff`, cache-stats + artifact
  inspection, and a **build-commit stamp** (`blut --version` →
  `0.10.0+<commit>[-dirty]`) with a **stale-binary warning** at startup when the
  source tree has moved past the running binary (the "git pull, forgot
  `cargo install --force`" trap).

### Fixed

- Footprint drivers are now the single cross-crate source of truth, fixing a
  calibration key-parity bug where the CLI (RESOLVE) and the stage (RECORD)
  keyed differently and the calibration never hit.
- The warm fullband-cache precompute stage (`lamquant_warm_fb_cache`, cookbook)
  was the last uncontained stage — it forked a worker pool under a flat
  reservation and could OOM the **box**. It now runs contained, sized from a
  dedicated fork-pool model, box-fit clamped, with its worker count pinned to
  the billed cap; a warm OOM is non-fatal (the contained trainer re-decodes).
- Footprint right-sized to the measured envelope (tier-3 ~35G → ~23G) so the
  broker stops over-refusing; honest per-worker term + conservative worker cap.
- Numerous review-driven hardening fixes across the broker, executor, launcher,
  and lineage paths (see commit history).

### Known limitations

- **No mid-epoch durable resume yet** (BLUT-API Phase D). Stages retry on OOM,
  but a killed run restarts from scratch rather than resuming a checkpoint. This
  does not block normal runs.
- **Warm fork-pool sizing is clamped against total box RAM, not live-free
  RAM.** On a heavily loaded box the warm default may not complete (the run
  falls back to a slower decode-bound path and warns you to lower the worker
  count). Tune it down per run where needed.
- The LamQuant python payload still lives under this crate's `python/`
  transitionally; it moves to `blut-lamquant` in a later migration.

[Unreleased]: https://github.com/Quitetall/blut/commits/main
