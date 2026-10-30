# Changelog

All notable changes to the **blut** engine crate are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`blut` is the domain-agnostic engine (Stage → Plan → Recipe → Registry + CLI +
resource broker). The user-facing binaries live in separate cookbook crates that
depend on this crate.

## [Unreleased]

### Added
- `schedule install` takes `--tenant`, `--shared-cache`, `--no-cache`,
  `--sync-io` and `--experiment`, passed to the scheduled `recipe run`; the
  tenant is checked against the quota policy at install, as `recipe run`
  checks it. Library: `schedule::install_with` with `ScheduleOptions`, and
  `schedule::ENVIRONMENT_ALLOWLIST`; `schedule::install` is unchanged in
  signature.
- `plan promote` takes an unambiguous prefix of a deployment fingerprint, as
  other commands take job ids and content hashes; a prefix matching no
  deployment or several is refused. `registry_db::resolve_fingerprint` does the
  resolution.
- `checks::JsonlArtifact::from_file(path)`, and documentation of the checks
  stages' input contract: `check_jsonl` and `assert` take only `checks.jsonl`,
  so a cookbook guards its own JSONL by emitting `JsonlArtifact` or through a
  one-line adapter stage (a compiled example is in the `checks` module docs).
  `assert` compares the literal `actual` it is given.
- `--launcher k8s` submits to a cluster. `recipe run <name> --launcher k8s` and
  `recipe declare <file> --run --launcher k8s` check the recipe locally (a
  named recipe's args against its schema, a file by compiling it), create a
  `BlutPlan` for blut-operator (`planSource.recipe` or an inline
  file), follow its status, and exit with its result; Ctrl-C, SIGTERM and
  `blut cancel <job>` delete the plan. Settings come from `BLUT_K8S_*`
  (image, namespace, store and the name of its credentials Secret, resources,
  retries); credentials never enter the manifest. Without a kubeconfig, or on a
  cluster without the operator's CRD, the command refuses before submitting.
  The operator gains `planSource.recipe` (a recipe run by name).
- `BLUT_REMOTE_CACHE` for `recipe run` and `recipe declare --run`: a directory,
  or with the `s3` feature an `s3://bucket[/prefix]` URL, used as the cache's
  content-addressed remote tier. Stage outputs are written through to it and a
  cold machine or pod reads them back instead of recomputing. The store is
  probed once at launch, so a wrong endpoint or rejected credentials fail the
  run before any stage starts. Credentials come from `AWS_*` only.
- `BLUT_RUN_SUMMARY` for the same commands: a file that receives a one-object
  JSON summary when the run ends (state, per-stage state, cache counts, the
  error), at most 4096 bytes, including for failures before any stage starts.
  Without the variable nothing changes.
- `blut-operator` is supported: it runs `BlutPlan` resources as Jobs with
  status (phase, per-stage state, cache counts, retries, lost pods), store
  credentials from a Secret, immutable specs, schema and operator-side
  validation, and a disruption policy that retries drained or evicted pods
  without spending a retry. Shipped manifests in `crates/blut-operator/deploy`
  (CRDs, least-privilege RBAC, Deployment), `docker/Dockerfile.operator` and
  `docker/Dockerfile.cookbook`, `scripts/k8s_kind_e2e.sh` and a CI job running
  it, and [docs/OPERATOR.md](docs/OPERATOR.md).
- `docs/adr/`: the 41 engine-facing architecture decision records, federated
  from the project's decision log with private material redacted inline as
  `[redacted: <category>]`, plus an index. "ADR NNNN" citations in the engine
  now resolve to a document in this repository; `book_src/adrs.md` points at
  the index instead of describing the gap.
- Mid-stage checkpoint slots (CONTRACT.md §8, trainer contract revision 1.1).
  A stage that returns `true` from the new `Stage::checkpoint_slot` gets a
  durable directory keyed by its cache key and an engine decision, in
  `StageContext::checkpoint`, on whether the attempt resumes. The trainer
  receives `BLUT_CHECKPOINT_DIR`, `BLUT_RESUME`, `BLUT_RUN_ID` and
  `BLUT_ATTEMPT_ID`. A retry, a `plan resume` and a Slurm requeue that re-runs
  the recipe all resume from the newest checkpoint instead of step 0; a trainer
  that may still be running is refused (`CheckpointBusy`) rather than raced.
  Each decision is a `stage_checkpoint` event in `status.jsonl`, and the slot
  is deleted once the stage's output is in the cache. New module
  `framework::stage_checkpoint`; `ExecCtx::with_checkpoint_root` and
  `$BLUT_CHECKPOINTS_DIR` choose where slots live.
- Cache keys identify the cookbook code that produced an entry
  (`framework::code_identity`). A cookbook returns
  `Some(blut::cookbook_identity!())` from the new `Cookbook::identity`, and
  every stage key folds the engine's and each registered cookbook's crate
  name, version, git commit and a digest of uncommitted changes. A cookbook
  change re-runs its stages without a `SCHEMA` bump; a build from a dirty
  tree never shares entries with the clean build or with another dirty
  state; a crates.io build falls back to name and version. `cli::run`
  installs the identity, warns about a cookbook that supplies none, and job
  provenance records it. The engine's `build.rs` now stamps
  `BLUT_GIT_DIRTY`. API.md, "Cache identity", has the `build.rs` contract.
- `cache prune` also deletes checkpoint slots unused for
  `--checkpoint-max-age-hours` (default 168), never one whose stage is
  running or whose trainer may still be alive
  (`stage_checkpoint::collect_slots`).
- Slurm jobs a stage starts are named `blut-<job id>-<node index>`
  (`StageContext::slurm_job_name`) and recorded with the BLUT job
  (`jobs::record_slurm_job_name`). Killing `srun` does not end its Slurm job,
  so: `blut cancel` now `scancel`s the job's recorded Slurm jobs, also after
  the BLUT process that started them died; `plan resume` refuses while any of
  them still runs and says to cancel them; and when an attempt fails under
  `--launcher slurm`, the executor cancels the Slurm job of that attempt.
- `ContainedChild::pump_stderr`: relays a contained child's piped stderr to a
  sink line by line and, for the systemd backend, records the `Memory peak:`
  line, so `peak_bytes` reports it. `ContainedChild::tears_down_tree` says
  whether the backend can kill the whole tree itself (systemd, cgroup v2, Job
  Object) or the caller must signal the process group (`rlimit`, `bare`).
- The systemd unit a stage runs in is recorded with the BLUT job, the way
  Slurm job names are. A transient `systemd-run --user` service keeps running
  when the BLUT process that started it is SIGKILLed (only its `systemd-run`
  client dies), and `blut cancel` used to reach nothing but that client. Now
  `LocalSystemd::spawn_contained`, called inside a job, writes the unit name
  to `<job dir>/systemd_units` before the process starts and describes the
  unit as `blut job <job id>`; the executor runs every local stage attempt
  inside its job (`jobs::in_job`). `blut cancel` stops the recorded units that
  still run for the job (`systemctl --user stop`), also when its BLUT process
  is dead or the job already finished, and prints each one it stopped;
  `plan resume` refuses while one runs, names it and says to cancel. A unit
  that has stopped, or that now belongs to another job (same name, other
  description), is ignored, so an old record never blocks a resume. New:
  `jobs::{in_job, record_systemd_unit, systemd_unit_names,
  live_systemd_units, systemd_resume_blocker, cancel_systemd_units,
  resume_blocker}`.

### Changed
- `blut jobs`' `output` column is now `recipe`: the recipe, deploy pointer or
  plan name the job ran (from its resume marker or `plan.json`), else a legacy
  single-trainer job's output name. The output name is set only by that legacy
  launch, so the column read `-` for every job a recipe ran. `--json` rows gain
  `recipe`; `output_name` is unchanged.
- A stage output whose artifact type declares external paths
  (`ALLOW_EXTERNAL_PATHS`, such as `checks.jsonl` and `dataset.jsonl`) is
  logged as "not portable and will not be cached" at debug instead of as a
  warning: it is the type's declared contract, and the warning was printed on
  every run of `check_jsonl`, `load_dataset` and the like. The output's sidecar
  still records `persisted: false`.
- **Behaviour change:** `plan publish`, `plan promote`, `plan rollback` and
  `plan history` default to tenant `default` (was `shared`), the tenant
  `recipe declare`/`recipe run`, `plan run`, the model registry and sensord
  already used. Publish → promote → `recipe declare registry://plan@prod --run`
  now works with default configuration; before, the pointer was written under
  `shared`, looked up under `default`, and `shared` was not an admitted quota
  tenant. Deployments and pointers made under `shared` stay where they are:
  address them with `--tenant shared`, and a lookup under `default` that misses
  names the tenant holding the pointer. blut-web authorises run and cancel
  against `default`, the tenant the launched run uses.
- `plan rollback` walks the promotion history like an undo stack: each
  rollback moves one promotion further back (promote A, B, C, then two
  rollbacks land on A), a promotion after a rollback continues from there, and
  a rollback at the first promotion is refused. It used to toggle between the
  last two targets. The trail records each move's kind, shown by `plan
  history`; registries created earlier gain the column on open, and their old
  moves count as promotions.
- SLA `max_runtime_secs` judges the newest matching run per tenant (on its
  runtime so far if still running), like `freshness_secs`, so an old breach
  stops failing `sla check` once a newer run meets the rule. It was judged per
  run: one historical slow run kept `sla check` non-zero forever.
  `deadline_unix` still judges every run still running.
- Every cache key changes once: the engine build commit alone is replaced by
  the code identity above. Existing entries miss on the first run.
- A refused checkpoint slot reads `resume checkpoint busy: stage '<name>':
  <holder>` instead of saying "busy" twice.
- A `ContainedChild` no longer outlives its tree. Under the systemd backend
  the direct child is the `systemd-run` client, and killing it left the
  transient service running. The backend's teardown now runs whenever the
  direct child exits, and again, best-effort, when the value is dropped before
  that (a hard stage timeout drops the stage future); for `rlimit` and `bare`
  a drop kills the child's process group.
- `LocalSystemd::spawn_contained` serves every local stage, not only Windows
  ones:
  - Under systemd the stage now gets the caller's environment (a
    `systemd-run --user` service otherwise starts from the user manager's
    minimal one), forwarded by name. That needs systemd 250; with an older
    systemd the stage runs under the next backend, and an explicit
    `BLUT_CONTAINMENT=systemd` is refused.
  - Under `rlimit` no cap is applied: an address-space cap at the launcher's
    default would refuse the reservations a CUDA process makes.
  - Unit names are `<unit>-<pid>-<seq>`, so concurrent and repeated stages
    never collide.
  - One log line names the backend, what it applies, and why it is not
    systemd when it is not (`BLUT_NO_CONTAIN`, no user bus, an old systemd).
  - A systemd unit is stopped with SIGTERM and, 10 s later, SIGKILL
    (`TimeoutStopSec=10s`), the grace `rlimit`, `bare` and `blut cancel`
    give a process group. It used to inherit the user manager's
    `DefaultTimeoutStopSec=` (90 s by default), so a trainer that ignored
    SIGTERM lingered that long after a cancel. `ContainedChild` waits up to
    15 s (was 10 s) for the backend's teardown command, so it no longer gives
    up on `systemctl stop` just before the SIGKILL.
- `blut-operator` moves to the engine's MSRV, Rust 1.88 (was 1.89), and from
  kube 0.96 to kube 4.0 / k8s-openapi 0.28 (Kubernetes 1.32 API). It is
  publishable (`publish = ["crates-io"]`, 0.2.0-alpha.2) but not yet published.
  CRD changes since the prototype: `planSource.inline.filename`, `command`,
  `store`, `resources`, `gpus`, `maxRetries`, `timeoutSeconds`; status gains
  `attempts`, `pod`, `lastPodFailure`, `cacheHits`, `cacheMisses`,
  `observedGeneration`; a `configMapRef` is mounted whole; the spec is
  immutable; `BlutWorkerPool` gains `coordinatorPubkey`. The prototype's Job
  could not run (it read the recipe from an empty stdin, never mounted the
  ConfigMap, and set environment variables the engine does not read), a PVC
  cache was mounted without `--shared-cache`, pool pods crash-looped for want
  of an identity key, and the pool's mesh port was TCP instead of UDP.
- Citations of decisions that were not federated (product, codec and platform
  records) are rewritten as the rationale they stood for. Citations written
  before the decision log renumbered two records now use the current numbers:
  experiment-execution UX is ADR 0156 (was 0071), cookbook taxonomy is ADR 0055
  (was 0051).

### Fixed
- A scheduled unit ran a different launch from the interactive one: it carried
  no tenant, no cache flags, no working directory and none of the installer's
  environment, so a timer ran under `default`, with a private cache, from the
  user manager's directory, against the default data, jobs, registry and
  catalog locations. The service now passes `--tenant` (always) and the
  launch flags given, sets `WorkingDirectory=` to the installer's directory,
  and sets `Environment=` for each variable of
  `schedule::ENVIRONMENT_ALLOWLIST` that is set (state locations such as
  `LAMU_TRAIN_DATA_DIR`, `LAMU_TRAIN_JOBS_DIR`, `BLUT_CHECKPOINTS_DIR`,
  `BLUT_REGISTRY_DB` and `BLUT_CATALOG_DB`, the remote cache location, the
  policy files, the interpreter, device and containment selection). Nothing
  outside the list is copied, so credentials (`AWS_*`, API keys, tokens) never
  reach a unit file, and a `BLUT_REMOTE_CACHE` URL with userinfo, a query or a
  fragment is refused.
- `blut log` printed only "no trainer progress" and `blut jobs` no step, loss or
  final loss for every job a recipe ran. A plan launch records trainer progress
  as `stage_step` events (a trainer stage forwards each `StatusUpdate` its
  trainer prints), and `jobs::read_status` read only bare update lines. It now
  also returns the updates stages forwarded, which the TUI's log pane reads
  too. When more than one stage reported progress, `blut log` prefixes each
  line with the stage (`[1 train] step 3/10 …`), and `blut log --json` lines
  carry `node_idx` and `stage_name`.
- `plan publish --tenant` stored any string, so a deployment could be published
  under a tenant no run can resolve (`a b`, `../x`). `plan publish`, `promote`,
  `rollback` and `history` parse `--tenant` like every other tenant input and
  use its canonical form (` lab/vision ` is `lab/vision`).
- `ledger show` printed each record as Rust `Debug`
  (`RunStarted { schema: "blut.run-ledger/v1", … }`). It prints one line per
  record (time in UTC, recipe, intent, tenant, outcome, tier, duration,
  promotion reason) with the run identity fields each record adds indented
  below it; `--json` is unchanged.
- An SLA rule could not select runs launched from a deploy pointer: such a run
  is recorded under the recipe `registry://plan@<name>`, and a rule's `recipe`
  refused that URI, so only a rule with no recipe (matching every run) judged
  them. A rule's `recipe` may now be the pointer URI. `lineage reindex` also
  rebuilt such a run under the published spec's name (refusing it as a recipe
  change when the run was already indexed); a pointer launch's `plan.json` now
  carries the URI it ran under, so `blut dag` shows it and a rebuilt index
  matches the live one.
- `lineage card` and `lineage graph` attributed an artifact to the newest job
  holding it, often a cache-hit job that never produced it, so a model card
  showed none of the training run's metrics. The index records whether each
  job produced or reused an artifact and prefers the producing job (existing
  indexes treat every row as produced until `lineage reindex`).
- `recipe declare registry://plan@<name> --run` defaulted the run's experiment
  to the URI, which is not a valid experiment name, and failed after creating
  the job directory but before its state, leaving a job that read `running`
  forever. Pointer launches group as experiment `plan.<name>`, any launch name
  becomes a valid label, the label is checked before anything is written, and
  a failure while writing a starting job's records removes the partial job.
- The systemd containment backend rewrote arguments that contain `$`: a
  `systemd-run` service's command line is expanded like `ExecStart=`, so
  `$VAR`, `${VAR}` and `$$` changed on the way to the stage (a JSON spec, a
  password, a regex). It now passes `--expand-environment=no` (systemd 254 and
  later) or doubles each `$`.
- The cgroup-v2 containment backend reported itself usable wherever it could
  create a directory in its own cgroup. That cgroup is usually a terminal's
  scope or a service with processes in it, where cgroup v2 will not enable the
  memory controller for children, so the first `memory.max` write failed and
  the stage with it. It now probes with a throwaway leaf and is usable only if
  the leaf gets `memory.max`; otherwise the factory falls through to `rlimit`.
- The systemd containment backend put every environment value it passed on
  `systemd-run`'s command line, which any user on the host can read. With
  systemd 250 and later only the names go there; the values travel in the
  client's own environment.
- `--launcher k8s` used to fall through to the local launcher and run the
  recipe on the submitting machine. `hpo`, `partition` and `plan resume` now
  refuse it, and the local launch path refuses any k8s placement that reaches
  it.
- `--launcher p2p` and `--launcher cloud` were accepted by `recipe run`
  (including `--dry-run` and config sweeps), `hpo run`, `study run`,
  `partition` and `plan resume`, and every stage ran on the submitting machine
  while the job recorded the remote placement: `launcher_for` mapped both to
  the local launcher, and no command installs the execution adapter that
  places stages on peers or on the cloud queue. Each of these commands now
  refuses both before writing a job, naming what does dispatch there
  (`blut p2p node --dispatch-stage`, `blut cloud smoke`, or for library callers
  `ExecCtx::with_execution_adapter`); `recipe declare --run` no longer points
  them at `recipe run`, and `plan resume` refuses a job recorded with either.
  The local launch path refuses them too, and the executor refuses a p2p or
  cloud placement with no execution adapter installed, and any k8s placement,
  before a stage runs.
- `blut dag` showed a node that failed and was then re-run to completion by
  `plan resume` as `failed`, because the first terminal event in the job's
  status log won. A later attempt (`stage_begin` or a cache hit) now reopens
  a failed or killed node. `blut errors show` likewise reported the recovered
  failure as the job's terminal failure next to `outcome : done`.
- The lineage index took a run's git SHA, end time and host from the process
  that indexed it: `git rev-parse HEAD` of wherever `blut` ran and "now". A
  `lineage reindex` from another directory rewrote every run's SHA, and no run
  ever had a start time. A job now records `provenance.json` at launch (launch
  directory HEAD, engine build commit, start time); the index reads it, and the
  end time is when the job's state became terminal.
- `blut sla check` evaluated "0 run(s)" whatever the lineage held, because no
  run row had a start time, so no SLA rule could breach. It now sees every
  indexed run. An explicit `--rules` or `$BLUT_SLA_RULES` file that does not
  exist is an error instead of "0 rules". A freshness rule now judges the
  newest measured run per tenant: judged per run, every run older than the
  window breached forever, and one run that failed before producing anything
  made the rule unmeasured.
- `blut lineage freshness` (and partition staleness) compared the launch
  directory's git HEAD with the current directory's and reported `STALE — a
  re-run re-executes` for runs a re-run then served from cache. It now
  compares the engine build commit the cache key folds in. `--data-version`
  with no artifact recording a data version reports `UNKNOWN`, not `FRESH`.
- `blut ledger compare` accepted `--alpha 2` (an underpowered comparison then
  read `SIGNIFICANT`), compared an empty arm for an unknown fingerprint
  prefix, pooled every run for the empty prefix, and printed `NaN` in
  `--json`. It now refuses an alpha outside (0, 1), a prefix that matches no
  arm or several, the same arm twice, and a metric no run recorded.
  `ledger list` marks collected runs and prints tiers as `--tier` spells them;
  `ledger verify` has help text (it had been attached to `compare`).
- `recipe run --config-dir … --set lr=1e-3` (the flag's own example) and
  `--sweep seed=1,2` changed nothing: every dot-less key was taken as a config
  group selection, so a sweep ran N identical combos. As in Hydra, a dot-less
  key is a group selection only when it names a config group. A sweep whose
  combos give the recipe identical args is refused before anything runs.
- `blut artifact inspect` refused an artifact reused by a cache hit as
  ambiguous, even given its full hash. `lineage graph` and `lineage card`
  accept a unique hash prefix, as `lineage trace` does.
- `blut lineage reindex` failed on a fresh install, and skipped every job
  launched from a plan file (`plan run`, `recipe declare --run`).
- A launch refused by tenant admission or the scheduler lock left a `failed`
  job behind. A job stopped by `blut cancel` was recorded as `failed` in the
  lineage index beside a `cancelled` job state.
- `recipe run --launcher slurm` on a host without the Slurm client ran every
  stage locally and reported success; it now requires `sbatch` and `scontrol`
  on PATH.
- `partition define` accepted an axis that is not a recipe arg (every cell
  then failed at backfill) and repeated values (duplicate cells), and replaced
  an existing set without a word.
- `blut sensord` started with a `--cli` that does not exist and retried every
  event against it forever; it now refuses to start. `--once` exits non-zero
  when a dispatch was refused or a poll failed.
- `schedule install` left its unit files behind when systemd refused to enable
  the timer, so `schedule list` reported it installed; `schedule uninstall` of
  a recipe with no timer reported "removed timer".
- Smaller: `plan inspect` defaults `--args` to `{}`; `recipe declare` refuses
  `--shared-cache`, `--no-cache`, `--sync-io` and `--experiment` without
  `--run` instead of ignoring them; `cache show`/`prune` print small sizes in
  bytes or KiB instead of `0.00 GiB`; `runs diff --json` outcomes carry no
  column padding; `blut jobs` shows the driving `blut` pid of a running
  in-process job; `results` labels its path as the output sidecar; `recipe
  list` says when a binary registers no recipes; the scheduler-lock sensor no
  longer names a nonexistent `--allow-evict` flag.

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
- Peer exchange (gossip) in `blut p2p node`, on by default (`--no-gossip` turns
  it off). A node sends its neighbours a signed list of the peers it has
  reached, on every new connection and every `--gossip-interval` seconds
  (default 30). It dials the peers it learns, pinning the key each was
  advertised with. Dials are bounded by `--max-peers` (default 32), a 10 s
  timeout, eight in flight, and a backoff that doubles from 5 s to 10 min.
  Seeds are redialed after a drop. A receiver refuses an exchange in these
  cases:
  - its `from` is not the TLS identity of the connection it arrived on;
  - it does not verify against that connection's key;
  - it is stale, oversized, or older than the last one from that peer;
  - it exceeds a per-peer rate limit.

  Gossip never writes the peer registry, so it cannot change any peer's trust.
  `--advertise host:port` sets the address a node advertises (needed behind
  NAT or a port forward). With a wildcard listen address and no `--advertise`,
  receivers use the IP they observe on the connection. Library:
  `MeshNode::start_gossip`, `GossipConfig`, `mesh_view`, `connected_peers`,
  `connection_to`, `add_seed`, `set_advertise_addr`.
- Stage data over the mesh without a shared cache (`p2p::objects`). A worker
  that lacks a task's input fetches it by content id from the node that
  dispatched the task, and the dispatcher fetches the output back the same
  way. The fetch reuses the deduplicated chunk transfer.
  - **Access:** a node serves an object only to a peer it was granted to (the
    task's worker for an input, the dispatching peer for an output), and only
    after the task's signature has been checked.
  - **Verification:** every chunk is hash-checked; the object must name the
    requested content id and match its manifest; the full content-id check
    runs before the object is used or stored.
  - **Limits:** size is capped before any chunk is requested
    (`--max-object-mb`, default 2048); every read stops after 60 s of silence;
    the whole fetch is bounded by the task timeout.
  - **Shared stores:** `--shared-store <dir|file://…|s3://…>` still lets
    nodes share a store; an object already present is not fetched.

  Library: `MeshObjects`, `SharedCacheRunner::with_mesh_objects`,
  `mesh_runner::dispatch_stage`, and `MeshTaskRunner::run_from`, a provided
  method, so existing runners are unaffected.
- `blut p2p node --dispatch-stage p2p-echo --input FILE --to PEER`: dispatch a
  real stage with real data over the mesh from the shipped CLI. PEER is a
  learned peer's id prefix or `addr@pubkey`. The command verifies the signed
  result and the output, then prints the timings. `--probe` now probes every
  connected peer, learned ones included, after a `--discover-secs` window.
- `s3` cargo feature: network object storage. `ObjectStore::from_url` opens
  `s3://bucket/prefix` (AWS S3, or an S3-compatible endpoint that supports
  conditional writes) and `file://` stores, and `blut cloud --store` accepts a
  URL as well as a directory. Credentials come only from the standard AWS
  environment variables, never from flags. Validated end to end against
  Cloudflare R2 over public TLS (which enforces the create-only writes) and
  MinIO.
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
- `Containment::before_spawn` and `Containment::after_spawn` (provided,
  default no-op) and `containment::spawn_attached`, which calls both. On
  Windows, `before_spawn` starts the child suspended and `after_spawn` assigns
  it to the job before it runs, so use `spawn_attached`. A caller that spawns
  the wrapped command itself gets a running child outside the job (uncapped,
  not killed by cancel), and a warning naming `spawn_attached` when the run's
  teardown handle is dropped; it is never left suspended. A failed `spawn()`
  is returned as `TrainError::Io` (program as `path`, the OS error as
  `source`), so a caller can still tell a missing program from other failures.
- `containment::ContainedChild`: a spawned child together with its
  containment. It spawns through `spawn_attached`, keeps the backend and
  teardown handle until the child exits, reads the peak after exit
  (`peak_bytes`), and kills the whole tree on `kill_tree`, on
  `wait_or_cancel` when the token fires, or (Windows) when dropped.
- `LocalSystemd::spawn_contained` and `LocalSystemd::cap_spec`: spawn a local
  stage under `containment_for()` with the launcher's `mem_max`/`mem_high`/
  `swap_max` as the cap. On Windows this is the Job Object, so a local stage
  gets the memory cap, peak reporting and tree kill; `LocalSystemd::wrap`
  cannot express a Job Object as a command line and now says so, naming
  `spawn_contained`, instead of blaming a missing `systemd-run`.

### Changed
- **Breaking (P2P API):** `p2p::transport::WireMessage` gains `Goodbye`, the
  orderly end of a session, and is now `#[non_exhaustive]` so later messages
  are not breaking. Code outside `blut` that matches on it needs a wildcard
  arm. This is the only finding `scripts/semver_checks.sh` accepts against
  0.2.0-alpha.1, recorded in `scripts/semver_exemptions.tsv`.

### Fixed
- A mesh node served only the connections it accepted, so a peer it dialed
  could not send it anything back over that connection. It now serves every
  connection in both directions. It also handled requests one at a time per
  connection with no read timeout: a long task blocked every other request,
  and a stream opened and left silent blocked the connection forever. Each
  request now runs in its own task, 16 at a time per connection, with 30 s for
  the first byte and 300 s for the frame.
- A mesh node completed inbound handshakes one at a time. While one peer's
  handshake was slow, stalled or deliberately silent (up to the 10 s handshake
  bound), every other peer trying to connect waited behind it, and under load
  dials timed out. Handshakes now run concurrently, at most 64 at once.
- One malformed or too-new frame ended service of the whole connection. It
  now fails only its own stream, with an error reply.
- `p2p node --coordinator-pubkey` accepted only the 64-hex Ed25519 key; it now
  also takes the whole 128-hex `pubkey` line that `keys show` prints, as
  `--seed` does.
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
