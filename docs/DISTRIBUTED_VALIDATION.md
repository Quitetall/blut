# BLUT Distributed and Cross-Platform Validation

Evidence ledger for engine `0.2.0-alpha.2`. Status means:

- **Validated:** exercised on named runtime or exact CI gate.
- **Component-tested:** parser, command construction, or loopback component
  tested; no scale claim.
- **Deferred:** unsupported in public preview.

Last updated: 2026-11-05.

Nothing in the table below changed when the crates were published on
2026-08-27. Publication moves code, not evidence: every row still says exactly
what was and was not exercised, and the Deferred rows are the ones to read
first if you are deciding whether to depend on this.

| Capability | Status | Evidence boundary |
|---|---|---|
| Linux systemd/cgroup2/rlimit containment selection | Validated | Local integration and failure-path tests |
| Linux systemd unit left by a killed BLUT process | Validated | One host, systemd 262, `sleep` stubs: recorded, found by `plan resume`'s check, stopped by `blut cancel`; SIGKILL 10 s into a stop, 2026-10-30 |
| macOS build with degraded containment | CI-gated | Compile only; no systemd guarantee |
| `blut-types` on wasm32 | CI-gated | Rust 1.88 `wasm32-unknown-unknown` check |
| Single-node GPU admission and per-device permits | Component-tested | Deterministic scheduler and executor tests |
| Single-node `torchrun` argument construction | Component-tested | Launch contract tests; cookbook owns trainer semantics |
| Ray launcher against a live cluster | Validated | Two-node Ray 2.x, submit + stream + poll to Succeeded, 2026-09-19 |
| Slurm launcher: `sbatch` submit, stream, poll, cancel; `srun` steps | Validated | Two-node Slurm 21.08 cluster, with and without accounting, 2026-10-20; Slurm jobs left by a dead driver found and cancelled by name, 2026-10-30 — see below |
| Mid-stage checkpoint resume (Slurm requeue, `plan resume`, stage retry) | Validated (CPU) | Two vast.ai VMs on different hosts, Slurm 21.08 over WireGuard, CPU DDP over gloo; each case resumed from its newest checkpoint and recomputed steps matched their losses exactly, 2026-10-24 — see below. No GPU, NCCL or FSDP run |
| Multi-node NCCL training through Slurm | Validated (cookbook) | Two physical machines, 2× RTX 5090, NCCL over a private network, 2026-10-20 — see below |
| P2P crypto, policy, transport, and loopback dispatch | Component-tested | Unit and loopback integration tests |
| P2P dispatch between separate network namespaces | Validated | Two containers, own IPs, real QUIC over a bridge — see below |
| P2P dispatch between independent hosts over the internet | Validated | Oregon ↔ France, both behind NAT, real QUIC over the public internet, 2026-10-20 — see below |
| P2P under latency, loss, partition, NAT and clock skew | Validated (emulated) | Containers on one host; impairment emulated, not a real WAN, 2026-10-20 — see below |
| P2P mesh across independent hosts (`p2p node`) | Validated (connectivity + signed dispatch) | Four hosts on four continents, every node probing every other, 12/12, 2026-10-20 — see below. Gossip and stage data over the mesh were not part of that run |
| P2P mesh gossip and stage data without a shared cache | Validated (independent hosts) | Four vast.ai hosts on four continents, each seeded with at most one peer, formed a full mesh by gossip; a fresh identity seeded with one node probed all four, 4/4; 64,640,000 bytes of stage data moved from the US to an empty-cache worker in New Zealand and back, verified, 2026-10-23 — see below. Earlier the same day across network namespaces on one host |
| Cloud queue over local object storage | Validated | `blut cloud smoke` round-trip through the shipped cookbook binary, 2026-09-19 |
| `--launcher cloud` on run commands | Validated (R2 over public TLS; MinIO; one host) | `recipe run`, `recipe declare --run` and `plan resume` place every stage on `blut cloud worker` processes through a durable queue on object storage. A four-stage standard-cookbook recipe on Cloudflare R2 with two workers as separate processes; claim races, lease expiry, cancellation, code-identity refusal, SIGTERM release, SIGKILL requeue and resume on R2 and MinIO, 2026-11-05 — see below. Workers and submitter on one host; no GPU stage; `hpo`, `study` and `partition` still refuse it |
| `--launcher p2p` on `recipe run` and `recipe declare --run` | Validated (independent hosts, CPU) | A five-stage cookbook plan from a workstation behind NAT: its four eligible stages ran on vast.ai hosts in Norway, New Zealand and Wisconsin found by gossip from one seed, the local-only stage stayed home, 4.8 MB of stage data moved over the mesh, and the output bytes equalled a local run; an anonymous peer and a peer running other code each failed the run with nothing placed, 2026-11-05 — see below. Cancel and a lost submitter are loopback tests only; no GPU stage, no contained process on a worker. `hpo`, `study`, `partition` and `plan resume` refuse the launcher |
| Network object storage (S3-compatible) | Validated (R2 over public TLS; MinIO) | `s3` feature; cloud round-trip against Cloudflare R2 over public TLS with create-only writes enforced, 2026-10-22; MinIO over HTTP and TLS; Backblaze B2 refused (no conditional writes) — see below |
| `blut-worker` REST prototype | Deleted (ADR 0083 M3) | Superseded by `src/cloud` + the `blut-web` sidecar |
| Kubernetes operator | Validated (kind) | `blut-operator` on a three-node kind cluster, Kubernetes 1.35.0, with MinIO in the cluster: a real CPU plan to Succeeded with per-stage status, outputs in S3, a rerun and a retry served from the store, bad specs refused, deletion cleaning up, a deleted pod and a drained node retried, an operator restart without duplicate work, least-privilege RBAC, 2026-10-24; `--launcher k8s` from the CLI submits, follows, cancels and refuses, 2026-10-30 — see below. One host; no GPUs, no managed Kubernetes, no mesh traffic between pool pods |
| Windows containment | Validated (CI) | Engine job on `windows-latest`, Rust 1.88, per push: a 512 MiB allocation fails under a 128 MiB Job Object cap and succeeds under 2 GiB, directly and through `LocalSystemd::spawn_contained`; the reported peak covers the commit; `TerminateJobObject`, closing the job, and `ContainedChild` cancel and drop each kill a grandchild that was alive a moment before. A cookbook run through BLUT on `windows-latest`, 2026-10-24: a CPU `train_from_dataset` with trainer and evaluator each in a Job Object (cap and peak reported), and a stage timeout and a killed driver each taking the training tree down — see below. `mem_high`/`swap_max` are not mapped, and the cap bounds committed memory, not RSS |

## Recorded run: P2P dispatch across network namespaces (2026-09-19)

The first P2P dispatch in this project that was not loopback.

Two containers on a user-defined bridge, each with its own network namespace,
its own IP and its own generated Ed25519/X25519 identity. The coordinator bound
`0.0.0.0:9320` inside its namespace and waited; the worker connected to the
coordinator's container IP over real QUIC and executed the dispatched stage.

```
coordinator 172.20.13.2:9320   worker connects from a second container
INFO P2P peer connected: c18e911091a8c91c (trust: anonymous)
worker c18e911091a8c91c connected — dispatching…
✔ stage 'p2p-echo' ran on peer c18e911091a8c91c; verified output:
CROSS-CONTAINER PROOF 2026-09-18
```

The output came back transformed and verified, so this exercised the whole data
plane — identity, connection, dispatch, remote execution, result verification —
not just a handshake.

Environment: kernel 7.2.5-1-cachyos, Docker 29.8.0, `archlinux@sha256:63c7b061c0c0`,
bridge subnet 172.20.13.0/24, `blut` 0.2.0-alpha.1 from crates.io built with
`features = ["p2p"]`. The published cookbook does not enable `p2p`, so the
driver was a throwaway consumer crate calling `blut::cli::run`.

**What this does NOT prove.** One kernel, one host, one virtual bridge. No NAT,
no packet loss, no latency, no partition, no clock skew, and no second machine.
It moves the claim from "loopback only" to "crosses a real network boundary
between independent network stacks", and no further. The row for a live
multi-host mesh stays Deferred until it runs on hardware that is genuinely
separate.

## Recorded run: cloud queue through the shipped binary (2026-09-19)

Run with `blut-standard` built from `blut-cookbook-standard --features
distributed` — an installed-shape binary, not a test harness:

```
submitting p2p-echo to the cloud queue (store: …/cloudstore)…
running a cloud worker…
✔ cloud round-trip verified over …/cloudstore; output:
DOGFOOD CLOUD QUEUE 2026-09-19
```

Submit, worker pickup, execution and verification all ran; the output came back
transformed, so this is the queue working rather than a file being written.

**Boundary:** the object store here is the local filesystem. Network object
storage is a separate row, validated against MinIO with the `s3` feature (see
below).

**Reachability, which was the real defect.** Until 2026-09-19 neither this nor
the P2P run was possible from anything BLUT ships. The engine is lib-only and
gates both behind cargo features; the published cookbook left them off, so the
subcommands were compiled out of the only shipped binary. `blut-cookbook-standard`
now exposes `p2p`, `cloud`, `raft` and `distributed`, and both runs above were
performed through it.

## Recorded run: Ray launcher against a live cluster (2026-09-19)

Two containers on a bridge — a Ray head and a worker, 40 CPUs — driven through
`blut::config::launcher::RayLauncher`, not by calling `ray` by hand:

```
submitted id=blut-ray-proof-fixed
  [log] BLUT_RAY_PROOF node= rayhead cpus= 40.0
terminal state: Succeeded
```

Submission, placement, log streaming and terminal-state polling all ran.

**It found a real defect on the first attempt.** `parse_ray_status` scanned for
a `Status:` line that Ray 2.x does not emit for a terminal job, so `poll()`
returned `Unknown` for every finished job — the launcher could not observe a job
succeed or fail, and never had been able to. Its seven unit tests asserted on
the same invented string, so they agreed with the parser and with nothing else.
Fixed against captured output; the run above is the same cluster after the fix.

This run says nothing about Slurm; the Slurm runs are recorded below.

## Recorded run: P2P dispatch across the internet (2026-10-20)

Two rented machines on different continents, sharing no network: a coordinator
in Oregon (US) and a worker in France. Each was a container behind its
provider's NAT. The coordinator was reachable only through a port forward
(public UDP 45026 → container 9320); the worker dialled out through ordinary
outbound NAT. Each side ran the same statically linked binary
(`blut::cli::run` with `features = ["p2p"]`) and had its own generated identity.

```
coordinator (Oregon)                          worker (France)
blut p2p serve --addr 0.0.0.0:9320 \           blut p2p connect 15.218.195.24:45026 \
  --smoke-stage p2p-echo                          --coordinator-pubkey eb5c597a…
P2P peer connected: a34788959fecd3cb (trust: anonymous)
✔ stage 'p2p-echo' ran on peer a34788959fecd3cb; verified output:
WAN OREGON TO FRANCE, RUN TWO
```

The input was lowercase (`wan oregon to france, run two`) and came back
uppercased and verified. So the identity exchange, the transatlantic QUIC
connection, dispatch, remote execution and result verification all crossed a
real WAN, not an emulated one. It ran twice; both passed.

**What it found.** The coordinator finished in 4.9 s, but the worker took
30.7 s to exit. The coordinator ended without a graceful QUIC close, so the
worker only noticed after the idle timeout. The coordinator's printed hint
also told users to connect to `0.0.0.0:9320`, its bind address, which no peer
can reach.

**Boundaries.**
- Two hosts and one dispatch, with the coordinator/worker roles, not an N-node
  `p2p node` mesh.
- No induced loss or partition here; the loss, latency, partition and NAT
  cases are a separate emulated matrix.
- NAT traversal relied on a provider port forward on the coordinator; there
  was no hole punching.

## Using the mesh (`blut p2p node`)

Each machine runs `blut p2p keys generate` once and then `blut p2p node`. Pass
peers as `addr@pubkey`, where `pubkey` is the `pubkey` line printed by
`blut p2p keys show` on that peer. The command needs a build with the `p2p`
feature.

**Finding peers.** One `--seed` is enough. Nodes exchange signed peer lists
(gossip) on connect and every `--gossip-interval` seconds (30 by default), and
dial what they learn. Each dial pins the key the peer was advertised with, so
a learned address can only reach that identity. Gossip never changes a peer's
trust, and `--no-gossip` turns it off.

**Addresses.** Normally nothing needs setting. A node advertises its listen
address; with a wildcard listen address such as `0.0.0.0:9320`, each peer
fills in the IP it actually sees the node on. Behind NAT or a port forward
whose public port differs from the listen port, give the public address:
`--advertise 203.0.113.9:40001`.

**Checking the mesh.** `blut p2p node --seed <one peer> --probe` dials the
seed, waits `--discover-secs` (5 by default) for gossip, then sends a signed
task to every connected peer, learned ones included. It exits 0 only if all
of them answer. A long-running node logs a `mesh: N connected, M known: …`
line whenever its view changes.

**Moving stage data.**
- **Worker:** start it with `--coordinator-pubkey <dispatcher pubkey>`. It
  runs stages only for that dispatcher.
- **Dispatcher:** run

  ```
  blut p2p node --seed <one peer> --dispatch-stage p2p-echo \
      --input big.txt --to <worker id prefix> --out result.txt
  ```
- **What happens:** the worker fetches the input from the dispatcher by
  content id, runs the stage and returns a signed result. The dispatcher then
  fetches and verifies the output.
- **Access:** an object is served only to the peer it was granted to (the
  task's worker, or the dispatching peer for an output).
- **Verification:** every chunk and the whole object are hash-checked before
  anything is used or stored.
- **Limits:** `--max-object-mb` (default 2048) caps what a node fetches.
- **Shared storage:** nodes with shared storage can pass the same
  `--shared-store <dir|file://…|s3://…>`, and an object already there is not
  transferred. `s3://` needs the `s3` feature.

`--dispatch-stage` builds its input from a file, which only the built-in
`p2p-echo` stage (UTF-8 text in, uppercase out) accepts. Library callers
dispatch any stage with `blut::p2p::dispatch_stage`.

**Running a recipe on the mesh.** `recipe run <name> --launcher p2p` and
`recipe declare <file> --run --launcher p2p` hand a run's stages to peers:

```
# each worker
blut p2p keys generate
blut p2p node --coordinator-pubkey <submitter pubkey> --seed <one peer>

# the submitting machine (its own identity)
blut p2p keys generate
blut p2p peers trust <worker pubkey> registered   # only for `internal` stages
blut recipe run <name> --args '...' --launcher p2p --p2p-seed <one peer>
```

- **What leaves:** a stage leaves only if it declares so, with
  `Stage::placement` and the class of its data. The other stages run on the
  submitting machine in the same plan, and the log says so per stage. A
  cookbook's stages stay home until its author opts them in.
- **Where it goes:** to a connected peer that runs the same code identity
  (the same engine build and the same cookbook builds), accepts this
  submitter, holds the trust the data class needs (`public` any peer,
  `internal` registered or trusted, `restricted` none), and has the declared
  cores and memory. Stages spread: fewest in flight, then least used. A stage
  that needs a GPU is not placed: workers have no GPU admission yet.
- **No peer:** if no peer can take an eligible stage within `--p2p-peer-wait`
  seconds (default 60), the run fails and names each peer it saw and why it was
  passed over. It never runs the stage on the submitting machine instead. A
  peer that is only busy is waited for, up to the stage's deadline.
- **Stopping it:** `blut cancel <job>` and Ctrl-C stop the stages running on
  peers. A worker whose submitter closes its connection, or goes silent for
  the QUIC idle timeout (30 s), stops the stage itself.
- **Trust from the command line:** `blut p2p peers trust` takes a whole
  `pubkey` line, which also adds a peer this machine has only dialed or
  learned by gossip.
- **Other run commands:** `hpo`, `study`, `partition` and `plan resume` still
  refuse `--launcher p2p`.

## Recorded run: a four-continent P2P mesh (2026-10-20)

Four rented machines in Nevada (US), Germany, South Korea and Australia. Each
was a container behind its provider's NAT, reachable only through a UDP port
forward. Each ran `blut p2p node` with its own identity. Then each node ran
`blut p2p node --probe` with the other three as `--seed`, given as the full
`pubkey` line from `keys show`:

```
from au:  de ok 360 ms   kr ok 166 ms   nv ok 241 ms   all 3 seeds answered
from de:  au ok 411 ms   kr ok 297 ms   nv ok 203 ms   all 3 seeds answered
from kr:  au ok 173 ms   de ok 303 ms   nv ok 203 ms   all 3 seeds answered
from nv:  au ok 219 ms   de ok 197 ms   kr ok 192 ms   all 3 seeds answered
```

Each line is a mutual-TLS connection pinned to the seed's key, then one task
dispatched. The probe verifies that the reply comes from that peer, carries a
valid signature from the seed's key, and names the same task and output. A seed
with the wrong key and an unroutable seed both failed within 11 s, with exit 1
and a message naming the presented and pinned keys.

**What it found.**
- Nothing could dispatch over the mesh from a shipped binary: `p2p node`
  connected to its seeds and waited. `--probe` closes that.
- `--seed` refused the 128-hex `pubkey` line that `keys show` prints.
- A pin mismatch blamed `--coordinator-pubkey` even when the key came from
  `--seed`.

**Boundaries.**
- The mesh's built-in worker answers a connectivity task; it does not move
  stage data. At the time, real stages over the mesh (`--coordinator-pubkey`
  on a worker node) read and wrote a content-addressed cache that every node
  had to share, and none was shared here. Since 2026-10-23 objects move over
  the mesh instead (recorded below, not yet between independent hosts).
- Stage data across the internet is covered by the coordinator/worker path
  above.
- Gossip (peer exchange) was not exercised.

## Recorded run: mesh gossip and stage data across network namespaces (2026-10-23)

**Setup.**
- **Nodes:** four containers (A, B, C, D) on one user-defined bridge, each with
  its own network namespace, IP address, identity and data directory. There is
  no shared filesystem.
- **Binary:** a static musl build of `blut` with the `p2p` feature, through a
  throwaway driver crate that calls `blut::cli::run`.
- **Topology:** every node listened on `0.0.0.0:9320` with
  `--gossip-interval 5`. B, C and D were each given only A as `--seed`; nobody
  was told about anyone else.

**Phase 1: gossip and probe.** All workers were smoke workers. D ran
`p2p node --seed A --probe`:

```
mesh: 3 connected, 3 known: 0fef…@172.29.77.2:9320(seed,up) 298d…@172.29.77.3:9320(gossip,up) e1ab…@172.29.77.4:9320(gossip,up)
probe 172.29.77.2:9320 (seed): ok, peer 0fefcf03eb92249f ran a signed task in 1 ms
probe 172.29.77.3:9320 (gossip): ok, peer 298d1e6e0faaf514 ran a signed task in 0 ms
probe 172.29.77.4:9320 (gossip): ok, peer e1abae2bdcb9418f ran a signed task in 0 ms
probe: all 3 connected peers answered (1 seeds, 2 learned, 0 known but not connected)
```

D learned B and C from A's signed exchange, dialed them with the keys A
advertised, and each answered a signed task. Exit 0 in 1.1 s wall. Nodes
advertised a wildcard address, and each receiver filled in the IP it saw.

**Phase 2: stage data.**
- **Workers:** B and C were restarted with `--coordinator-pubkey <D>`, and C's
  store was emptied.
- **Dispatcher:** D, again seeded only with A, ran
  `p2p node --dispatch-stage p2p-echo --input in.txt --to <C id prefix>`.
- **Input:** 50,331,648 bytes of base64 text, which spans several 12 MiB
  transfer chunks.

| Run | State | Publish | Dispatch → signed result | Output back | Total |
|---|---|---|---|---|---|
| 1 | C and D cold | 154 ms | 835 ms (C fetched the input from D: 50,332,230 bytes in 299 ms; stage 89 ms) | fetched from C, 852 ms | 1.85 s |
| 2 | both warm | 196 ms | 453 ms (input already in C's store, not fetched) | already in D's store, 1,249 ms | 1.90 s |
| 3 | C warm, D's store emptied | 132 ms | 420 ms (not fetched) | fetched from C, 2,859 ms | 3.41 s |

Each run printed `verified: output == uppercase(input), signed by
e1abae2bdcb9418f`. Outside BLUT, `sha256(uppercase(in.txt))` and
`sha256(out.txt)` were equal (`7c963444…be1ba`). B's store never received the
input.

**Environment.** Kernel 7.3.0-3-cachyos, Docker 29.9.2, `archlinux@sha256:8b15f4618d79`,
bridge 172.29.77.0/24, 2 GiB memory per container. Data directories were on
`tmpfs`.

**Why tmpfs.** The host disk was saturated by another job: IO pressure "full"
was about 45%, and one synchronous 12 MiB write took 1.2–7.9 s. On the
disk-backed first attempt, the same cold dispatch took 75 s (C's input fetch
alone took 72 s). Some dials also timed out, because the peer registry is
fsynced while an inbound handshake is in progress. The tmpfs run measures the
protocol; the disk-backed numbers are what a node on a disk that slow will
see.

**What it found.** The node completed inbound handshakes one at a time, so a
slow or silent handshake held up every other peer. It also served only the
connections it accepted, and a malformed frame ended service of the whole
connection. All three are fixed (see the changelog). The gossip loop could
also dial a seed a second time while the first explicit dial was still in
flight; a dial in progress now blocks a duplicate.

**Boundaries.** One host and one bridge: no NAT, no loss, no latency, and no
separate machines. `--advertise` was not needed and was not exercised.
`--shared-store s3://` was not exercised over the mesh.

## Recorded run: mesh gossip and stage data between independent hosts (2026-10-23)

**Setup.**
- **Hosts:** four vast.ai instances (`ubuntu:22.04`, no GPU used) in
  Wisconsin (US), Sweden, South Korea and New Zealand. Each had its own
  identity and data directory, and only a UDP port forward to `9320`; there was
  no shared filesystem and no overlay network.
- **Binary:** the static musl build described above, from the alpha.2
  candidate with the mesh work merged; the same sha256 on every host.
- **Topology:** every node ran `p2p node --listen 0.0.0.0:9320 --advertise
  <public ip>:<forwarded port> --gossip-interval 10` with
  `BLUT_P2P_CONGESTION=bbr`. Wisconsin had no seed; Sweden seeded only
  Wisconsin, South Korea only Sweden, New Zealand only South Korea.

**Phase 1: gossip and probe.** Within a minute every node logged
`mesh: 3 connected, 3 known`, with the peers it was not seeded with marked
`gossip` or `inbound`. A fifth identity on the Wisconsin host, seeded only
with New Zealand, then ran `--probe --discover-secs 20`:

```
mesh: 4 connected, 4 known: 4111…@114.23.254.176:52964(seed,up) d6c4…@212.100.111.103:20029(gossip,up) d6eb…@1.208.108.242:60782(gossip,up) ee6d…@167.237.110.62:48179(gossip,up)
probe 114.23.254.176:52964 (seed): ok, peer 4111366410e76a45 ran a signed task in 187 ms
probe 212.100.111.103:20029 (gossip): ok, peer d6c423ede22621be ran a signed task in 124 ms
probe 1.208.108.242:60782 (gossip): ok, peer d6eb34491e2a692b ran a signed task in 176 ms
probe 167.237.110.62:48179 (gossip): ok, peer ee6d170056d7527d ran a signed task in 2 ms
probe: all 4 connected peers answered (1 seeds, 3 learned, 0 known but not connected)
```

Exit 0. The advertised public addresses were the ones gossip carried; no node
was told the forwarded port of any peer except its one seed.

**Phase 2: stage data.**
- **Worker:** New Zealand, restarted with `--coordinator-pubkey <dispatcher>`
  and its `shared-cache`, `chunks` and `work` directories removed.
- **Dispatcher:** a separate identity on the Wisconsin host, seeded only with
  Sweden, ran `p2p node --dispatch-stage p2p-echo --input big.txt --to
  4111366410e76a45`. It had no inbound port forward; the worker fetched over
  the connection the task arrived on.
- **Input:** 64,640,000 bytes of base64 text.

| Run | State | Publish | Dispatch → signed result | Output back | Total |
|---|---|---|---|---|---|
| 1 | worker cold | 1,165 ms | 23,165 ms (worker fetched 64,640,582 bytes in 17,980 ms, verified in 586 ms; stage 622 ms) | fetched over the mesh, 18,075 ms | 42.4 s |
| 2 | both warm | 1,619 ms | 4,978 ms (not fetched; stage 557 ms) | read from the dispatcher's store, 1,586 ms | 8.2 s |

Both runs printed `verified: output == uppercase(input), signed by
4111366410e76a45`. On the dispatcher host, outside BLUT,
`sha256(uppercase(big.txt))`, `sha256(out1.txt)` and `sha256(out2.txt)` were
all `995373a1eda0c400…`. The worker logged exactly one `fetched object`. The
Wisconsin, Sweden and South Korea nodes had no `shared-cache` or `chunks`
directory afterwards: the data went only to the worker it was dispatched to.
No node logged a warning or an error.

**Rate.** About 3.6 MB/s (29 Mbit/s) each way between Wisconsin and New
Zealand with BBR. That is one sample on one path; it is not a throughput
claim.

**Boundaries.** No injected loss or partition on this run (those are recorded
separately below, for dispatch). `--shared-store s3://` over the mesh was not
exercised. The probe and the real-stage worker cannot be the same process: a
worker started with `--coordinator-pubkey` refuses the probe task, so the
worker was restarted between phases.

## Recorded run: a recipe's stages across independent hosts (2026-11-05)

**Setup.**
- **Workers:** three vast.ai instances (`ubuntu:22.04`, no GPU used) in Norway,
  New Zealand and Wisconsin (US), each with its own identity and data directory
  and a UDP port forward to `9320`. No shared filesystem, no overlay network.
  Norway had no seed; the other two seeded only Norway.
- **Submitter:** a workstation behind NAT with no inbound port, running
  `recipe declare plan.json --run --launcher p2p`, seeded only with Norway. It
  learned the other two workers by gossip.
- **Binary:** one static musl `blut-core` from the standard cookbook (candidate
  `7261d4e`), built against engine commit `68a8797` with the `p2p` feature,
  sha256 `6a3cdf2e…147e`, the same file everywhere. `BLUT_P2P_CONGESTION=bbr`.
- **Cookbook patch:** the cookbook's stages do not declare placement yet, so
  this build carried a local patch (kept with the evidence) declaring
  `filter_dataset`, `split_train_eval`, `take_train` and `take_eval` remote with
  `internal` data.
- **Plan:** a PlanSpec: `materialize_dataset_path` (declares nothing, reads a
  local path) → `filter_dataset` → `split_train_eval` → `take_train` and
  `take_eval`, over 4,802,910 bytes of synthetic conversation JSONL.
- **Workers' launch:** `p2p node --listen 0.0.0.0:9320 --advertise <public
  ip>:<forwarded port> --gossip-interval 10 --coordinator-pubkey <submitter>`.

**Runs.**

| Run | Conditions | Result |
|---|---|---|
| 0 | workers `anonymous` to the submitter, `--p2p-peer-wait 20` | Exit 1: `no mesh peer could take stage 'filter_dataset' within 20s`, naming each worker, its trust and the `peers trust` command. The stage ran nowhere |
| 1 | workers `registered`, cold, sequential executor | Exit 0, 41.9 s, each worker ran at least one stage |
| 2 | warm, `--no-cache`, `BLUT_EXECUTOR=parallel` | Exit 0; no worker fetched anything; `take_eval` and `take_train` started 109 ms apart on different workers |
| 3 | one worker running the previous build (`129688b`), `--p2p-peer-wait 15` | Exit 1: the error names both builds. The stage ran nowhere |

Run 1 stage by stage. "There" is the stage on the worker; "dispatch" runs from
the worker accepting to the output being back.

| Stage | Ran on | Input fetched by the worker | There | Output back | Dispatch |
|---|---|---|---|---|---|
| `materialize_dataset_path` | submitter | | | | |
| `filter_dataset` | Wisconsin | 4,803,492 bytes, 2,960 ms | 85 ms | 4,422,730 bytes, 1,402 ms | 4,827 ms |
| `split_train_eval` | New Zealand | 4,423,289 bytes, 5,407 ms | 45 ms | 4,422,749 bytes, 3,042 ms | 8,855 ms |
| `take_train` | Norway | 4,423,520 bytes, 1,885 ms | 0 ms | 3,549,135 bytes, 3,436 ms | 5,818 ms |
| `take_eval` | Wisconsin | 4,423,520 bytes, 1,733 ms | 0 ms | 873,614 bytes, 149 ms | 2,135 ms |

**Verification.** Outside BLUT, the sha256 of every output file from runs 1 and
2 equals a `--launcher local` run of the same plan: `filtered.jsonl`
`c61c2300…`, `train.jsonl` `784f1133…`, `eval.jsonl` `76ed1e08…`. The content
ids of `filter_dataset` and `split_train_eval` equal the local run's as well.
`take_train` and `take_eval` have different ids with the same bytes: locally
they return a file of the split stage, which is not persisted and gets a
non-portable id; on a worker the file arrives inside the stage's own directory
and is packed. Worker logs show one `running stage` line per dispatched stage,
fetches only in run 1, and no warning or error. After run 2 `p2p peers list` on
the submitter showed each worker's completed tasks (`ok=4`, `ok=2`, `ok=2`) with
reputations above 0.50.

**What it found.** An earlier round from commit `129688b` on Korea, Quebec and
Norway ran the same plan with the same result, but the parallel executor sent
`take_train` and `take_eval` to one worker: a stage counted its peer as in
flight only once the worker accepted, a round trip after choosing it. Choosing
and reserving are now one step. Run 2 above is the rerun.

**Hosts that did not work.** Of the nine instances rented, six were usable:
a Polish host's UDP forward never answered from three other machines, a
Korean instance exited on start, and a Vietnamese one never accepted SSH. Each
was destroyed and replaced. Offers also disappeared between search and create.

**Timing caveat.** Disk pressure on the workstation from other jobs was
20–30% during these runs. The gaps between stages in run 2, which took 140 s
end to end, were the workstation's own disk; the stages and transfers inside
them took seconds. These numbers are one sample each and are not a throughput
claim.

**Boundaries.**
- The stages are in-process Rust. No stage started a process on a worker, so
  containment on a worker was not exercised here, and no GPU stage can reach a
  worker.
- `blut cancel` and a lost submitter are covered by the loopback tests
  (`tests/p2p_mesh_dispatch.rs`), not by this run: these stages finish too fast
  to interrupt over a WAN.
- A worker with other code was exercised on the submitter's side. The worker's
  own refusal, for a submitter that skips that check, is a loopback test.
- No injected loss or partition. `--shared-store s3://` was not used.
- The submitting machine's memory admission still bills the whole plan,
  including stages that run elsewhere.
- Worker progress events do not reach the submitter's status log.
- The published cookbook has not declared placement for its stages.

## Recorded run: hostile conditions on a real WAN (2026-10-20)

- **Path.** The worker in Australia dialled a relay in South Korea, which
  forwarded to the coordinator in Germany. That is two real intercontinental
  legs, with loss and partitions injected at the relay.
- **Payload.** 1 MiB each way, verified as uppercase(input).
- **Variants.** Cubic is quinn's default controller. BBR is opt-in, via
  `BLUT_P2P_CONGESTION=bbr`.

| Case | Cubic | BBR |
|---|---|---|
| Direct, no relay | 9.6 s | — |
| Through the relay, no loss | 12.1 s | 10.4 s |
| 1% random loss | 87–145 s | 11.3 s |
| 5% random loss | did not finish in 300 s (twice) | 14.2 s |
| 20 s partition mid-transfer, then heal | completes, 44 s | completes at 5% loss, 38 s |
| Permanent partition | both sides exit 1 within 36–42 s ("nothing heard from the other side for 30s") | same |

**What it found.** Random loss on a long path collapses loss-based congestion
control. At 5% loss, Cubic could not move 1 MiB in five minutes, which the
emulated matrix had predicted on one host. BBR does not read random loss as
congestion and stays close to the clean-path time. quinn marks its BBR
implementation experimental, so it is an opt-in rather than the default.
`BLUT_P2P_CONGESTION` accepts `cubic` (default) or `bbr`, and refuses anything
else before any connection is made.

**Boundaries.** Loss and partitions were injected at one relay. The
underlying paths were real but healthy. Payloads were 1 MiB. Fairness against
competing traffic was not measured.

## Recorded run: the Slurm launcher on a live cluster (2026-10-20)

A two-node cluster: two containers on one bridge, Slurm 21.08.5 from Ubuntu
22.04, `slurmctld` and `slurmd` on `n0`, `slurmd` on `n1`, and a volume both
nodes mount as the shared filesystem. It was run twice: once with
`AccountingStorageType=accounting_storage/none` and once with `slurmdbd` backed
by MariaDB. `cgroup` and `TaskPlugin` were off (`proctrack/linuxproc`), so this
exercises scheduling and process launch, not resource enforcement.

`tests/slurm_live.rs` drives `SlurmLauncher` through its public API, checking
eight things:

- a job's output streams in order and it ends `Succeeded`;
- `exit 3` ends `Failed`;
- `scancel` ends `Cancelled`;
- a two-node batch job runs on both nodes, each with its own `NODE_RANK` and
  one agreed `MASTER_ADDR`;
- a job queued behind an exclusive one is waited for, not failed;
- a job the controller has purged is answered by accounting, or refused with a
  clear message without it;
- `srun` wrapping runs on every node;
- a previous job's log under a reused job id is not streamed.

```
                                   accounting off   accounting on
code before this run (8d7eb4d)     1/8 pass         2/8 pass
code after                         8/8 pass         8/8 pass
```

**What it found.**
- Job state came only from `sacct`. Without accounting, `sacct` exits 1 ("Slurm
  accounting storage is disabled"). With accounting, it prints nothing for a
  job it has not recorded yet. Either way, streaming gave up immediately after
  submission.
- A multi-node batch script ran its command once, on node 0, and set
  `NODE_RANK` from the batch shell, where it is always 0.
- Slurm reuses job ids after its state is reset. The previous job's
  `slurm-<id>.out` was then read as the new job's output, then seeked past the
  end when the new job truncated it.
- The parser's tests asserted `CANCELLED+`. That is `sacct` truncating
  `CANCELLED by 0` to fit its column, not a state; the parser handled it only
  by accident.

All of the parser inputs in the unit tests are now captured from this cluster.

**Boundaries.** One host and one kernel, with Slurm 21.08 only. No cgroup
enforcement, no GPU binding (GPUs were declared with placeholder device files to
check how Slurm treats `--gpus` with `--gpus-per-node`), no preemption, no node
failure, and no federation.

## Recorded run: multi-node training through Slurm (2026-10-17, 2026-10-20)

The engine places stages; the cookbook owns training. These runs used the
standard cookbook's `train_from_dataset` recipe:

```
blut-core recipe run train_from_dataset --launcher slurm
```

It trained SmolLM2-135M on `Salesforce/wikitext` with `nnodes=2` and one rank
per node.

| | 2026-10-17 | 2026-10-20 |
|---|---|---|
| Machines | one host, two `slurmd` | **two physical machines** |
| GPUs | 2× RTX 3060 | 2× RTX 5090 |
| Network | loopback | private overlay network between the machines |
| NCCL transport | — | `NET/Socket` over the overlay, `eth0` |
| World size | 2 | 2 |
| Train loss (epoch 1) | 3.5554 | 3.5554 |
| Eval loss | 3.36 | 3.3595 |
| Exit | 0 | 0 |

Both runs used Slurm 21.08, torch 2.8.0+cu128 and transformers 5.7.0. The
engine was `8d7eb4d`.

The train loss is identical across the two topologies: same seed, same global
batch of 16, different hardware. That is the global-batch parity check.

**Caveats.**
- The containers could not mount NFS or FUSE. On 2026-10-20 a wrapper around
  `srun` copied the job directory to the second machine before each step,
  standing in for the shared filesystem real clusters provide.
- On 2026-10-17 a task prolog pinned each node's GPU, because there was no
  cgroup GPU confinement.
- Neither run exercised failure recovery: no node was lost mid-run.

## Recorded run: losing a node mid-job (2026-10-20)

Same two-node cluster, with `SlurmdTimeout=30`. A two-node `sbatch` job
streamed through `SlurmLauncher`; 12 s in, node `n1`'s container was killed.

```
  | rank=0 tick=1
  | rank=1 tick=1
  | srun: error: Node failure on n1
  | slurmstepd-n0: error: *** JOB 1 ON n0 CANCELLED AT … DUE TO NODE FAILURE ***
  | [blut] slurm job 1: requeued, now PENDING (BeginTime)
                                  (n1 restarted here)
  | rank=0 tick=1
  | rank=1 tick=1
  | [blut] slurm job 1: running again
  | rank=1 tick=30
  | rank=0 tick=30
stream returned after 227s: Ok(())      poll: Succeeded
```

**What it found.** Slurm requeues a batch job whose node fails. The job went
back to `PENDING`, and while its node stayed down it waited there
indefinitely. The stream waited too, which is correct, but it said nothing, so
the operator could not tell that the job had been killed and was waiting for
hardware. The stream now notes a requeue, a suspension, a hold, or a wait for
anything other than the job's turn, each once and with Slurm's reason, and
notes when the job is running again. The rerun's output streamed from the start
of its truncated log.

**Boundaries.** Slurm-level recovery only: the job restarted from the
beginning. Resuming BLUT's own DAG from its last completed stage after a node
loss (`plan resume`) was not exercised here.

## Recorded run: durable resume after losing a node (2026-10-20)

A rented 16-core machine ran two `slurmd` nodes (`n0`, `n1`) in one container
under Slurm 21.08. It ran the standard cookbook's `train_from_dataset` with
`--launcher slurm`, `nnodes=2`, CPU ranks over gloo, and SmolLM2-135M on 1200
wikitext rows.

1. **Node loss.** Mid-epoch, `n1`'s `slurmd` and its training rank were killed
   with SIGKILL. Seconds later the job failed with `plan stage
   'train_model_on_split' (idx 2) failed`, exit 1. Nothing hung.
2. **Resume.** With `n1` back, `blut-core plan resume <job>` ran:

```
resuming 20261020-204247-001040720 with --launcher slurm (as it was run)
[trainer] epoch 1/1: loss=3.1975
[trainer] replicas agree across 2 ranks: params sha256 b2f7f10d3f19f932…
[evaluator] metrics: {'loss': 3.0972, 'n_rows': 120, 'n_items': 61}
done — 5 ingredients, 2 cache hits, 3 misses        exit 0
```

The two stages that had finished before the loss came from the cache. Training
ran again on both nodes and the job completed.

**What it found.**
- `plan resume` dropped the job's launcher: every resume ran locally. A
  two-node Slurm job resumed that way failed at once ("multi-node training
  needs MASTER_ADDR"). The launcher is now recorded with the job, and resume
  places stages where the run did. Jobs recorded before this resume locally,
  as they did before.
- The trainer now checks that DDP replicas agree. Every rank hashes its
  parameters, and the run fails before saving if any rank differs. That gives
  an artifact hash compared across nodes, from the training process itself.
  Both runs above agreed.

**Boundaries.**
- One host, CPU ranks, a short run.
- Resume restarts the interrupted stage from its beginning. There is no
  mid-stage checkpoint resume.

## Recorded run: resuming a training stage from its checkpoint (2026-10-24)

Two vast.ai virtual machines on different physical hosts at one site, each an
Intel Xeon E5-2699 v4 with 21 vCPUs, joined by WireGuard (about 2 ms round
trip). Slurm 21.08.5 from Ubuntu 22.04: `slurmctld` and `slurmd`
on `n0`, `slurmd` on `n1`, `SlurmdTimeout=30`, `JobRequeue=1`. The standard
cookbook's `train_from_dataset` trained SmolLM2-135M on wikitext
(`max_samples` 1200, or 700 for the second and third cases) with
`--launcher slurm`, `nnodes=2`, one CPU rank per node over gloo, torch 2.8.0,
checkpointing every 10 steps (1.6 GB each: weights and AdamW state) and
printing the loss every step. Engine `917e7b4`, cookbook `a6bcc11`.

There was no shared filesystem. As in the earlier cross-host run, a wrapper
around `srun` copied the jobs directory to `n1` before each step. Checkpoints
were not copied: only rank 0, on `n0`, reads or writes them.

"Node loss" below means SIGKILL to `n1`'s `slurmd`, its step daemon and its
training rank.

**1. Slurm requeue.** The recipe ran inside `sbatch --requeue --nodes=2`. At
step 33 of 71, with the step-30 checkpoint written, `n1` was lost.

```
[rig] batch start: slurm job 5, restart 0, on n0
 INFO checkpoint: node 2 (train_model_on_split) attempt 1 starts fresh in …/train_model_on_split-058b5a1f…
[trainer] step 33/71 loss=3.3921
                                  (n1 lost; 40 s later Slurm requeues job 5; n1 returns)
slurmstepd-n0: error: *** JOB 5 ON n0 CANCELLED AT 07:10:40 DUE TO NODE FAILURE ***
[rig] batch start: slurm job 5, restart 1, on n0
 INFO checkpoint: node 2 (train_model_on_split) attempt 1 resumes from step 30 in …/train_model_on_split-058b5a1f…
[trainer] resumed from step-0000000030.pt: step 30/71, epoch 1, batch 30/71
[trainer] step 31/71 loss=3.3021
[trainer] epoch 1/1: loss=3.1975
[trainer] replicas agree across 2 ranks: params sha256 33ea5793…
 INFO checkpoint: node 2 (train_model_on_split) attempt 1 resumed from step 30
done — 5 ingredients, 0 cache hits, 5 misses        exit 0
```

The requeued run is a new BLUT job, so the dataset stages ran again; the
training stage got the same cache key and therefore the same slot. The engine
treated the slot as abandoned because the previous attempt had run in the same
Slurm job with a lower `SLURM_RESTART_COUNT`, and resumed at once rather than
waiting for the old heartbeat to go stale.

**2. `plan resume` after the BLUT process dies.** Run directly on `n0`
(seed 43, 39 steps). At step 23, with the step-20 checkpoint written, the
`blut-core` process got SIGKILL.

- Its `srun` step kept running: `srun`, both step daemons and both training
  ranks were still alive 30 s later, and step 24 was logged after the kill.
- The first `blut-core plan resume` refused before running the stage:
  `checkpoint slot for 'train_model_on_split' is busy: trainer pid 215242 on
  n0 (run 20261024-073108-…) wrote a heartbeat 27s ago; it may still be
  running`.
- After `scancel` of that step, the second `plan resume` resumed:
  `2 cache hits, 3 misses`, `resumed from step-0000000020.pt: step 20/39`,
  replicas agreed, exit 0.

**3. Stage retry.** Run directly on `n0` with `train_attempts: 2` (seed 44,
38 steps). At step 23, with the step-20 checkpoint written, `n1` was lost.
The first attempt failed about 35 s later (gloo: connection closed by peer;
`srun: error: Node failure on n1`). After the 10 s backoff the engine started
attempt 2 in the same process, which waited in `srun` until `n1` was back:

```
 INFO checkpoint: node 2 (train_model_on_split) attempt 1 starts fresh in …/train_model_on_split-4d493d04…
{"kind":"stage_retrying",…,"attempt":1,"max_attempts":2,"error":"backend: blut_core.trainer exited with exit status: 1","backoff_ms":10000}
 INFO checkpoint: node 2 (train_model_on_split) attempt 2 resumes from step 20 in …/train_model_on_split-4d493d04…
[trainer] resumed from step-0000000020.pt: step 20/38, epoch 1, batch 20/38
[trainer] replicas agree across 2 ranks: params sha256 0680252c…
done — 5 ingredients, 0 cache hits, 5 misses        exit 0
```

**Loss continuity.** In each case the steps between the newest checkpoint and
the interruption ran twice, before and after it, and gave the same loss to the
last printed digit of the float:

```
                  before the interruption        after resuming
requeue, step 31   3.302097797393799              3.302097797393799
requeue, step 33   3.3920702934265137             3.3920702934265137
plan resume, 24    3.051217794418335              3.051217794418335
retry, step 23      3.0574417114257812             3.0574417114257812
```

The requeue case was then run again with the same arguments and nothing
interrupted. All 71 step losses matched the requeued run's at full printed
precision, and so did the epoch loss (3.1975), the final parameter hash
(`33ea5793…`, agreed by both ranks in both runs) and the held-out evaluation
loss (3.0949). The resumed run ended with the weights an uninterrupted run
produces.

Each resume appears in the job's `status.jsonl`: `resume_offered` with the
marker's step, `resumed` with the step the trainer loaded, then `cleaned` once
the output was in the cache. No slot remained after any run.

**What it found.**
- Killing the BLUT process does not stop its training step, so a naive
  `plan resume` would have put a second trainer on the same checkpoints. The
  slot refuses that, and says which process to stop. At the time BLUT did not
  record the step's Slurm job with the job, so stopping it was `scancel` by
  hand; the record below closes that.
- DDP replicas on different CPU models drift apart bit by bit. An earlier pair
  of these VMs had an Intel Xeon E5-2699 v4 and an AMD Ryzen 9 3900X; after 12
  steps the trainer's replica check failed with two different parameter
  hashes and the stage failed. The run above used two machines with the same
  CPU model, and the hashes agreed.
- A local two-rank test showed why saves must be atomic: when rank 1 died
  while rank 0 was writing its first checkpoint, `torchrun` killed rank 0
  mid-write. Only a temporary file was left, so the next attempt found no
  checkpoint and started at step 0 instead of loading half a file.

**Boundaries.**
- CPU ranks over gloo, one rank per node, two nodes; no GPU, NCCL or FSDP run.
  FSDP stages do not checkpoint mid-stage.
- No shared filesystem; the slot sat on rank 0's node.
- Same site, low latency. The requeue case relied on Slurm's restart count;
  a run whose previous owner is on another host and not in the same Slurm job
  waits for the 180 s heartbeat timeout.

## Recorded run: Slurm jobs left behind by a dead driver (2026-10-30)

The containerized two-node cluster from the launcher record above (Slurm
21.08.5, `slurmctld` and `slurmd` on `n0`, `slurmd` on `n1`, a shared volume).
`tests/slurm_live.rs` gained two checks, and the suite ran 10 of 10:

- **A Slurm job outlives the process that started it.** A two-node
  `srun --job-name=blut-<job>-2 sleep 300` was started with no BLUT process
  behind it, and its name recorded in a BLUT job directory. `plan resume`'s
  check (`jobs::slurm_resume_blocker`) named the running job and pointed at
  `blut cancel`; `jobs::cancel_job` `scancel`ed it; the `srun` exited and the
  check cleared.
- **A failed attempt does not leave its job running.** A stage under
  `--launcher slurm` started a detached `srun --job-name=<slurm_job_name>
  sleep 300` and then failed. The executor recorded the name and cancelled
  the job by name. With that cancel removed, the same test failed after 60 s
  with the job still running.

**Boundaries.** One host, containers, no training: the jobs are `sleep`. A
stage inside a batch allocation names a job step, which `squeue --name`
does not list; there the batch job ends its steps when the driver exits.

## Recorded run: a systemd unit left behind by a dead driver (2026-10-30)

The developer box (systemd 262, a user manager, Rust 1.88), the engine at the
`agent/fu-contain-orphans` branch. Under the systemd backend a stage runs in a
transient `systemd-run --user` service whose only link to the BLUT process is
the `systemd-run` client; SIGKILLing the BLUT process (or SIGTERMing the
client) leaves the service running, which was checked by hand on this systemd
before the change.

- **Found and stopped after the driver is SIGKILLed**
  (`tests/systemd_orphans.rs`). The test binary, run again as a child, is the
  driver: it records itself as the job's driver and runs a one-stage plan
  through the parallel executor whose stage spawns `sleep 600` through
  `LocalSystemd::spawn_contained`. The unit appeared in
  `<job dir>/systemd_units` and ran; the driver was SIGKILLed; one second
  later the unit was still active. `jobs::resume_blocker` (what `plan resume`
  runs) named the unit and `blut cancel <job>`; `jobs::cancel_job` printed
  `stopped systemd unit(s) blut-orphan-test-<pid>-1.service of <job>`, the
  unit was gone, the job read `cancelled`, and with the record still on disk
  the resume check returned nothing.
- **Escalation** (`tests/systemd_containment.rs`, part 5). A `sh` and its
  `sleep` grandchild, both ignoring SIGTERM, were cancelled through
  `ContainedChild::wait_or_cancel`; the unit's transient file carried
  `TimeoutStopSec=10s`, and both processes were gone between 9 s and 25 s
  into the cancel (the whole test took 10.4 s). This host's user manager
  already defaults to `DefaultTimeoutStopSec=10s`, so the timing alone does
  not tell the change apart here; the transient-file check does, and on a
  stock manager (90 s) the timing check would too.
- **Planted regressions.** Each was planted alone, run, and restored: the
  record write removed; the executor's `jobs::in_job` scope removed; the
  unit's job description removed; the stop in `cancel_job` removed; the
  `TimeoutStopSec` argument removed. The first two failed after 60 s with no
  unit recorded, the third and fourth at once (the unit not recognised as the
  job's; the unit still active after the cancel), and the last failed both
  the argv unit test and the live test.

**Boundaries.** One host; stubs only, no training. The check runs on the host
that runs `blut cancel` or `plan resume`: a job directory on a shared
filesystem, resumed from another host, does not see a unit on the first one.
A stage that spawns from a task of its own sees no job unless it wraps that
task in `jobs::in_job`. Units of `LocalSystemd::wrap`/`launch`
(`run_contained.sh`) and cgroup-v2 leaves are not recorded.

## Required evidence (Slurm, still open)

- A cluster with cgroup enforcement and GPU binding, rather than placeholder
  devices.
- Mid-stage resume on GPUs over NCCL, with a shared filesystem holding the
  checkpoint slots.

## Recorded run: cloud queue over S3 (2026-09-20)

Built with `--features s3`, pointed at a MinIO endpoint over HTTP:

```
submitting p2p-echo to the cloud queue (store: s3://blut-cloud/runs)…
✔ cloud round-trip verified over s3://blut-cloud/runs; output:
S3 AFTER PROVIDER FIX 2026-09-20
```

The bytes are in the bucket, under canonical addressing with the URL's own path
applied as a prefix:

```
runs/cloud/v1/dispatch-bundles/3d121ea13b4f…
```

Credentials and endpoint came from the environment (`AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ENDPOINT`, `AWS_ALLOW_HTTP`), which
is the only way the CLI accepts them.

**The gates that deferred this were re-measured rather than assumed.** Enabling
`s3` adds 50 packages — aws-lc-rs, hyper, h2, reqwest and their trees. With
`all-features = true` in `deny.toml`: advisories ok, bans ok, licenses ok,
sources ok.

**It also broke P2P, and that is fixed here.** `object_store/aws` pulls
`aws-lc-rs`, so rustls saw two crypto providers and refused to choose; eleven
p2p tests panicked under `s3` while passing under `cloud`. The transport now
installs `ring` explicitly. Feature matrix after the fix, each run separately:
default 1000, p2p 1140, cloud 1159, s3 1163, all-features 1163 — zero failures.

**Boundaries.** MinIO over plain HTTP on a container network: this exercises the
S3 API, the signing path and the provider wiring. It is not a test of TLS to a
public endpoint, of IAM roles, of regional latency, or of any commercial
provider's quirks. And it is not yet reachable from a shipped binary —
`blut-cookbook-standard` cannot enable `s3` until an engine release carrying the
feature is published.

## Recorded run: S3 over TLS, and a provider that cannot host it (2026-10-20)

Built with `--features s3` and driven through `blut cloud smoke`.

**TLS.** MinIO served HTTPS with a certificate from a private CA created for the
run:

```
untrusted CA:              failed to verify TLS certificate: invalid peer certificate: UnknownIssuer   (exit 1)
SSL_CERT_FILE=<that CA>:   ✔ cloud round-trip verified over s3://blut-tls/run1; output: TLS ROUND TRIP 2026-10-20
```

So certificate verification is enforced, and the platform trust store, or
`SSL_CERT_FILE`, is what decides.

**Backblaze B2, a commercial S3-compatible endpoint, over public TLS.** The TLS
handshake and request signing succeeded, then every create-only `PUT` was
refused:

```
501 Not Implemented  <Code>NotImplemented</Code>
A header you provided implies functionality that is not implemented
```

B2 does not implement `If-None-Match: *` on `PUT`. BLUT publishes every object
that way, so a second writer can never replace the first. Cache invocations are
keyed by their inputs, not their bytes, so without the guarantee two workers
could silently overwrite each other's cached result. There is therefore no
fallback. BLUT now refuses such an endpoint with an error naming the missing
capability; before, it surfaced a generic 501 after ten retries.

**Boundaries.**
- TLS was proven against MinIO with a private CA, not against a public CA chain
  on a commercial provider that also supports conditional writes.
- IAM roles, STS and instance credentials were not exercised. Only the static
  key variables were.
- AWS S3 itself has not been run against.

## Recorded run: P2P under hostile network conditions (2026-10-20)

The P2P smoke path, `p2p serve --smoke-stage p2p-echo` on a coordinator and
`p2p connect` on a worker, run between containers while the network between
them was made hostile. Every process ran under a timeout, and a hang counted as
a failure.

**Rig.** Arch Linux containers on one host (kernel 7.2.9-2-cachyos, Docker),
each with its own network namespace, IP and Ed25519/X25519 identity. The driver
was a throwaway crate calling `blut::cli::run`, built from this branch with
`features = ["p2p"]`. The rig used three impairment mechanisms:

- *Kernel*, for NAT and idle partitions. The worker sat on a second network
  behind a router container doing `MASQUERADE`, with `iptables` `DROP` inside the
  worker's namespace.
- *A userspace UDP relay* on the router, for delay, jitter, loss and timed
  partitions. `tc netem` could not be used because the running kernel's modules
  directory had been removed by an upgrade, so `sch_netem` would not load. The
  relay applies independent per-direction random loss and per-packet delay plus
  uniform jitter, delivered in order. A partition starts after N forwarded
  packets so it lands mid-transfer. The relay can also rebind to a new source
  port, as a NAT does.
- *`libfaketime`* for clock skew.

The 4 MiB payloads could not go through `--smoke-input`, because Linux caps one
argument at 128 KiB. For those the driver read the file and called the same
library path: `P2pServer::bind`, `accept_peer`, `smoke_dispatch_once`, then
`end_session`. The worker was the stock `p2p connect`.

Results with the fixed engine. Wall times are per process. The coordinator's
includes up to 5 s spent closing the session.

| # | Scenario | Expected | Observed | Wall | Result |
|---|---|---|---|---|---|
| 1 | Baseline, bridge, ×3 | verified output, both exit 0 | as expected | coord 1.1 s, worker ≤0.1 s | pass |
| 2 | Delay 200 ms ± 50 ms each way, ×3 | completes | as expected | worker 2.6–2.7 s, coord 5.3–5.4 s | pass |
| 2b | Same delay, 4 MiB payload | completes | as expected | 21.5 s | pass |
| 3 | 5 % loss each way, ×10 | completes | 10/10 | 1.1–1.4 s | pass |
| 3b | 30 % loss each way, ×10 | completes | 10/10 | 1.3–8.4 s | pass |
| 4 | 20 s partition mid-transfer (4 MiB, 50 ms each way), ×2 | completes after the heal, or a bounded error | completes after the heal | 37.7–38.5 s | pass |
| 4b | Permanent partition mid-transfer | bounded error, both exit nonzero | both exit 1: "timed out: nothing heard from the other side for 30s (QUIC idle timeout)" | 32 s | pass |
| 4c | Idle worker, 60 s, no traffic | session stays up | stays up; `serve` exits on SIGINT, worker exits 0 | 60 s | pass |
| 4d | Idle worker, kernel `DROP` for 20 s, then heal | session survives | survives | 60 s | pass |
| 4e | Idle worker, kernel `DROP` permanently | worker fails, bounded | worker exits 1 after 30 s idle | 45 s (15 + 30) | pass |
| 5 | Worker behind `MASQUERADE`, dialing out | completes | completes. The coordinator sees the router's address (conntrack) | 1.1 s | pass |
| 5b | NAT rebinding mid-transfer: new masquerade port range and conntrack flush | QUIC migrates, or a clean failure | migrates and completes (mapping moved to a new port) | 1.4 s | pass |
| 5c | Relay rebinds to a new source port mid-transfer (4 MiB, 50 ms), ×2 | QUIC migrates | migrates and completes | 4.9 s | pass |
| 6 | Dial a host that does not exist / a closed port | bounded, clear, nonzero | exit 1 after 30.0 s: "no QUIC response from … within 30s … UDP port 9320 is reachable" | 30.0 s | pass |
| 6b | Wrong `--coordinator-pubkey` | clear, nonzero | exit 1 at once, naming the presented and the pinned key. The coordinator logs the attempt and keeps waiting | 0.0 s | pass |
| 6c | Truncated pubkey, unparsable address, host name | clear errors; names resolve | as expected | 0.0 s | pass |
| 7 | Clock skew ±10 min, on the worker or the coordinator | no false rejection | all four complete. The worker logs the skew and uses the relative budget | 1.1 s | pass |
| 8 | 4 MiB, 5 % loss, ×3 | completes | 3/3 | 10.3–11.1 s | pass |
| 8b | 4 MiB, 5 % loss plus 100 ms each way | completes or meets its deadline | misses the 120 s deadline; coordinator exits 1 at 152 s, worker exits 1.6 s later ("stream reset by peer") | 152 s | bounded failure |
| x | 4 MiB, 200 ± 50 ms with reordering, as `netem` jitter does | — | misses its deadline the same way | 154 s | bounded failure |

Rows 8b and x are throughput, not hangs. Loss-based congestion control at
5 % loss and 200 ms RTT sustains about 30 KB/s. Heavy reordering reads to QUIC
as loss. Both runs ended at the deadline with clear errors on both sides.

**What it found.** The same matrix against the base commit:

- **A worker exited about 30 s late.** In 4 of 13 delayed or lossy smoke runs,
  and on a real Oregon-to-France run, the coordinator exited before its close reached
  the worker. The worker then waited out the 30 s idle timeout. The
  coordinator now sends `Goodbye` on a reliable stream, waits up to 5 s for the
  acknowledgement, closes with an application code, and drains before exiting.
  The worker lets the coordinator close first.
- **An idle worker was dropped after 30 s, and a 20 s partition was fatal.**
  The transport set no keep-alive. Mid-transfer, PTO backoff left nothing sent
  between the heal at 20 s and the idle timeout at 30 s, so the session died
  even though the network had come back. Every connection now sends a
  keep-alive every 5 s. Row 4 completes; before, both sides failed at about
  36 s, the worker with exit 0.
- **A worker that lost its coordinator exited 0** and printed "coordinator
  connection closed.". It now exits 1 and gives the reason.
- **`p2p serve` never exited on Ctrl-C.** After the endpoint closed, the accept
  loop spun without yielding: 7.9 million "QUIC endpoint closed" log lines in
  5 s, and then the process had to be SIGKILLed.
- **An abandoned transfer kept sending.** `send_blob` returned once quinn had
  buffered the data. After a dispatch deadline, the coordinator kept
  transmitting the input for nobody. Under congestion, quinn also held the
  CONNECTION_CLOSE frame behind that pending data (seen in a quinn trace), so
  the worker learned of the end only by idle timeout. Each chunk now waits for
  the acknowledgement and is reset if abandoned.
- **Clock skew.** The worker read the coordinator's absolute deadline on its
  own clock. A worker 10 minutes ahead therefore saw every task as already
  expired. `p2p-echo` finishes within one poll, so the smoke run passed anyway;
  any stage that yields would have been killed. Unit tests pin the corrected
  budget. No handshake, certificate or replay check depends on wall time:
  certificate validity is not checked, because identity is the pinned key.
- **Errors and hints.** The coordinator's hint printed its bind address
  (`0.0.0.0:9320`). A wrong pin surfaced as "ApplicationVerificationFailure". A
  short pubkey reported "Invalid input length 128". `p2p connect` rejected host
  names. One stray handshake made the smoke coordinator exit.

**Timeouts in effect.** QUIC idle timeout 30 s, which also bounds an
unanswered dial. Keep-alive every 5 s. The application handshake after QUIC
connects: 10 s. Orderly close: up to 5 s for `Goodbye` to be acknowledged, plus
up to 5 s to drain. Smoke dispatch: 60 s stage deadline, and the coordinator
gives up on the reply at 90 s (`timeout_secs + 30`). Clock-skew tolerance
before the worker switches to the relative budget: 5 s. Host-name resolution:
10 s. A smoke coordinator waits for its first worker without a time limit, like
any server.

**Boundaries.** One host, one kernel, one Docker bridge. Delay, jitter, loss
and partitions were emulated, in userspace for all but NAT and the idle
partitions, and are not a real WAN. Real paths differ in queueing, bursty loss,
middleboxes and MTU. NAT was Linux `MASQUERADE` only: no symmetric or
carrier-grade NAT, and no NAT on both sides. IPv4 only. Only the
coordinator/worker smoke path was exercised, not the symmetric `p2p node` mesh
or multi-peer dispatch. Evidence: `artifacts/BLUT/keep/2026-10-20-p2p-hostile/`
on the validation host.

## Recorded run: S3 against Cloudflare R2 over public TLS (2026-10-22)

The run used a bucket created for it and an API token scoped to that bucket
alone, with a two-day expiry. Both were deleted, and the token revoked,
afterwards. Built with `--features s3`, endpoint
`https://<account>.r2.cloudflarestorage.com`, region `auto`:

```
✔ cloud round-trip verified over s3://blut-qa-2026-10-22/run1; output: R2 PUBLIC TLS ROUND TRIP 1
✔ cloud round-trip verified over s3://blut-qa-2026-10-22/run2; output: R2 PUBLIC TLS ROUND TRIP 2
```

Create-only writes are enforced, not just accepted. A driver over
`ObjectStore::from_url` wrote one cache key three times:

```
first put: Stored
same bytes again: AlreadyPresent
different bytes: refused as Conflict (create-only enforced)
stored bytes unchanged: "first writer"
```

The R2 account endpoint failed TLS (`handshake failure`) for a while after R2
was first enabled on the account; it worked once Cloudflare had issued its
certificate. A brand-new account can hit this, and it clears by itself.

**Boundaries.** Static key credentials only; IAM roles and STS were not
exercised. AWS S3 itself has not been run against.

## Recorded run: `--launcher cloud` against Cloudflare R2 (2026-11-05)

Engine `afa6bd8`, `blut-core` from the standard cookbook (`2e7a77f`) with
`blut/s3`, Rust 1.88. The bucket was created for the run, with an R2 API token
scoped to that bucket alone and a six-hour expiry; the derived S3 credentials
reached the processes through their environment only. Endpoint
`https://<account>.r2.cloudflarestorage.com`, region `auto`, public TLS.

A four-stage recipe of CPU stages, `materialize_dataset_path` →
`filter_dataset` → `split_train_eval` → `take_train`, over a 400-line JSONL
dataset, through `recipe declare chain.toml --run --launcher cloud`. The
submitter and two `blut cloud worker` processes ran as separate processes with
separate state roots, on one machine:

```
cloud  s3://blut-qa-cloud-queue-2026-11-05/… (workers: `blut cloud worker --queue <the same location>`)
done — 4 ingredients, 0 cache hits, 4 misses, elapsed 23.918921028s

worker r2-w2: claimed materialize_dataset_path, take_train; 2 succeeded
worker r2-w1: claimed filter_dataset, split_train_eval; 2 succeeded
0 job(s) on s3://blut-qa-cloud-queue-2026-11-05/…
```

Nothing ran in the submitter. A local run of the same recipe from the same
binary produced byte-identical files (`dataset.jsonl`, `filtered.jsonl`,
`train.jsonl`, `eval.jsonl`, compared by SHA-256). `take_train`'s recorded
content id differs between the two, by design of portability: locally its
output is the previous stage's file, outside its stage directory, so it gets the
unpersisted identity; on a worker the file is inside the stage directory and is
shipped. The bytes are the same.

The whole test suite then ran on the same bucket:

- queue semantics, 12 of 12: lifecycle, eight workers racing for each job with
  exactly one winner, lease expiry and fencing of the old holder, renewal,
  release, cancellation, idempotent completion, the code-identity and GPU
  refusals, priority order;
- the process-level tests, 10 of 10: every stage in a worker's pid; no worker
  fails after the pickup window and withdraws the stage; a worker of other code
  leaves the job and the submitter reports why; `blut cancel` stops a running
  stage on a worker after the submitter was SIGKILLed, and withdraws a queued
  one; `plan resume` refuses while a stage runs and finishes the job afterwards
  with the first stage from cache; SIGTERM hands a stage to another worker;
  a SIGKILLed worker's stage is requeued when its lease runs out; two runs
  overlap on one submitting machine.

On a directory all of them pass. On MinIO the queue semantics and nine of the
process tests passed; the tenth was added after that pass. Three planted
regressions were caught: claims written without create-only (six workers won one job), leases
that never expire, and no code-identity check (both the queue test and the
process test failed).

R2's latency showed a one-second lease in a test expiring during a single claim
attempt; the test's lease was too short, and `--lease-secs` defaults to 60. Host
memory pressure during the same pass found a real defect: a cloud run took local
RAM admission and the training lock, though nothing runs on the submitter. It no
longer does.

**Secrets.** All 353 bucket objects, the job directories and the logs kept with
this record were scanned for the access key, the secret, the Cloudflare API
token and the account id: 425 files, no hit.

**Cleanup.** Bucket emptied (353 objects, then 0), bucket deleted (lookup 404),
token revoked (lookup 404), and the old credentials then get `Unauthorized`.

**Boundaries.** One machine, so one clock and no second network path; the
workers shared the dataset path with the submitter, because a stage reads paths
on the worker. No GPU stage, no subprocess stage on a worker, nothing large:
bundles here are kilobytes. Static key credentials only. AWS S3 itself has not
been run against. Evidence: `artifacts/BLUT/keep/2026-10-30-cloud-queue/`.

## Recorded run: training through BLUT on Windows, in Job Objects (2026-10-24)

GitHub Actions `windows-latest` (a Windows Server VM), Rust 1.88, Python 3.12,
CPU torch 2.14.1, transformers 5.20.2. The standard cookbook
([blut-cookbooks](https://github.com/Quitetall/blut-cookbooks) `fa1baec`,
branch `agent/windows-runner`) was built against engine `0c869b9` (branch
`agent/windows-local`), patched in for crates.io's `blut`. Workflow
`windows-containment`,
[run 40053013881](https://github.com/Quitetall/blut-cookbooks/actions/runs/40053013881):
6 min 38 s in all, of which 1 min 46 s building (warm cache), 1 min 28 s
installing torch and transformers, and 140 s of checks.

Every local stage process in that cookbook spawns through
`LocalSystemd::spawn_contained`. `tests/windows_e2e/run.py` in that
repository drives `blut-core` and checks from outside, confirming every
process it expects to die is alive first:

1. **Train.** `train_from_dataset` on 64 text rows, `sshleifer/tiny-gpt2`, one
   epoch on the CPU, then the held-out evaluation:

   ```
   containment: blut-train_model placed by LocalSystemd under windows-job-object, memory cap 47244640256 bytes
   containment: blut-train_model attached to windows-job-object (pid 7668)
   [trainer] epoch 1/1: loss=10.8087
   containment: blut-train_model exited; peak memory 618266624 bytes (windows-job-object)
   containment: blut-evaluator placed by LocalSystemd under windows-job-object, memory cap 47244640256 bytes
   containment: blut-evaluator attached to windows-job-object (pid 7792)
   containment: blut-evaluator exited; peak memory 601137152 bytes (windows-job-object)
   ```

   Exit 0, a 421,241-byte checkpoint with its HuggingFace export, and an
   evaluation report with a held-out loss of 10.7845. The cap is
   `LocalSystemd`'s default `mem_max` (44G). No "never assigned" and no
   "UNCONTAINED" warning.
2. **Stage timeout.** A PlanSpec, `load_dataset` then `train_model`, with a
   120 s soft timeout on `train_model` and a model ingredient that starts a
   grandchild process and then hangs. With trainer and grandchild confirmed
   running:

   ```
   containment: blut-train_model cancelled; killing its process tree (windows-job-object)
   plan stage 'train_model' (idx 1) failed: stage timed out after 120.0336577s (limit 120s)
   ```

   The plan exited 1, and the trainer and the grandchild were gone within
   30 s.
3. **Driver killed.** The same plan without a timeout. With trainer and
   grandchild running, the `blut-core` process was terminated outright
   (`TerminateProcess`, so no cleanup code ran). The trainer and the
   grandchild were gone within 30 s: the run's Job Object closed with the
   driver's handle, and `KILL_ON_JOB_CLOSE` took the tree.

**What it found.** The first run
([40052362399](https://github.com/Quitetall/blut-cookbooks/actions/runs/40052362399),
engine `ce98a58`) trained, evaluated and attached both stages, then failed its
own warning check: the backend's availability probe built a run's job handle
and dropped it, which logged "no child was ever assigned" once per process.
Engine `0c869b9` probes with a bare handle.

**Boundaries.** One VM image, the CPU only, a tiny model, one process per
stage: no GPU, no DDP. The 44G cap was never approached here; that a cap
refuses an allocation is the engine's Windows tests above. `mem_high` and
`swap_max` are not mapped. Ctrl-C and `blut cancel` were not exercised, and
`blut cancel` cannot yet stop a driver from another process on Windows. The
LamQuant cookbook's runner was not run on Windows.

## Recorded run: the Kubernetes operator on kind (2026-10-24)

`scripts/k8s_kind_e2e.sh`, the script the `operator · kind end-to-end` CI job
runs, against a kind cluster of one control plane and two workers, Kubernetes
v1.35.0 (kind 0.31.0, kubectl 1.36.2), each node container capped at 3 GiB.
The operator image was built at `6d8e1d0` and the cookbook image (the
standard cookbook, `blut-cookbooks` at `6918883`, compiled against the engine
at `5d44f77` with `p2p` and `s3`) at the same engine source; the script is the
one committed with this record. MinIO ran in the cluster with a throwaway root
password generated by the script, and the plans read their S3 credentials from
a Secret. Everything was installed from `crates/blut-operator/deploy`.

The plan is three CPU stages from the standard cookbook: materialize a
400-line JSONL dataset carried in the plan's ConfigMap, filter it, split it.

```
[  81s] check 1: plan runs to Succeeded
[  90s]   succeeded: 3 stage(s), 0 from cache, 3 computed
[  90s] check 2: stage outputs are in the store
[  94s]   s3://blut-store/run1: 3 artifact objects, 3 invocation records
[  94s] check 3: a rerun is served from the store
[  98s]   succeeded: 3 stage(s), 3 from cache, 0 computed
[  98s] check 4: bad specs are refused
[  99s]   5 refused by the API server, 3 by the operator, no Job for any
[  99s] check 5: deleting a running plan removes its Job, pod and ConfigMap
[ 103s]   Job, pod doomed-run-b8jw9 and ConfigMap gone
[ 103s] check 6: a deleted pod is retried and the retry reuses the store
[ 119s]   retry lose-pod-run-6z6qv: done — 3 ingredients, 3 cache hits, 0 misses, elapsed 2.65962632s
[ 119s]   status while retrying: attempt 2, running; previous pod lose-pod-run-rh84g was deleted: Error, exit 143
[ 183s] check 7: draining the node under a running plan
[ 198s]   drain-me-run-968ps on blut-op-e2e-worker evicted; drain-me-run-hbr7z on blut-op-e2e-worker2; status: attempt 2, running; previous pod drain-me-run-968ps was disrupted (not counted against maxRetries): Error, exit 143
[ 261s]   maxRetries 0, drained once, Succeeded
[ 261s] check 8: restarting the operator mid-run
[ 327s]   one Job (uid unchanged), one pod (restart-me-run-mk2nk), Succeeded
[ 338s]   finished plan with its Job deleted: not re-run
[ 338s] check 9: a BlutWorkerPool's StatefulSet comes up and its status follows it
[ 350s]   2/2 ready, a deleted pod replaced, StatefulSet and Service gone with the pool
[ 350s] check 10: the operator's permissions
[ 350s]   allowed 4/4, denied 12/12, no forbidden calls in the operator log
[ 351s] PASS: all checks
```

The refusals, verbatim from the API server and the plan status:

```
spec.dataClassCeiling: Unsupported value: "Restricted": supported values: "Public", "Internal"
spec.cookbookImage: Invalid value: "": cookbookImage must not be empty
spec.maxRetries: Invalid value: 50: maxRetries must be at most 20
spec.store.url: Invalid value: "relative/dir": store.url must be s3://bucket[/prefix] or an absolute path
spec: Invalid value: BlutPlan spec is immutable; delete the plan and create a new one
invalid spec: store.url must be s3://bucket[/prefix]; credentials and the endpoint belong in store.credentialsSecret, never in the URL
invalid spec: recipe file 'plan.yaml' must end in .json (PlanSpec), .toml (chain) or .star (Starlark); the extension selects the format
invalid spec: args is only for .star recipes; 'plan.json' carries its own args
```

The URL in the first operator refusal carried a fake key and secret; neither
appears in the status.

Checks 6 to 8 use a test-only wrapper image, `blut-hold`, that runs the plan and
then keeps the pod alive for 60 s, so there is a running pod whose stages have
all finished and been written to the store. The operator does not know about
it.

**What it found.** Before this run the operator could not run a plan at all:
the Job read the recipe from an empty stdin with no file extension, never
mounted the ConfigMap it named, and set environment variables the engine does
not read. Nothing wrote a status. A PVC cache was mounted but `--shared-cache`
was never passed. The pool's Service and container port were TCP; the mesh is
QUIC. The `gpus` field the engine's `LaunchTarget::K8s` renders was not in the
CRD, so the API server pruned it. Getting the run to pass also found:

- a deleted or evicted pod leaves the API before the next reconcile can read
  it, so the status now carries the last lost pod forward (`lastPodFailure`)
  and counts attempts by pod name;
- an S3 error on a remote-cache lookup is a cache miss, not a failure. While
  MinIO's drive monitor took its disk offline on a saturated host, a rerun
  recomputed one stage instead of failing. The e2e now runs MinIO on a
  memory-backed volume;
- every `BlutWorkerPool` pod crash-looped: `p2p node` will not start without
  an identity key and nothing created one. An init container now generates
  it. A pool also had no way to pass `--coordinator-pubkey`, so its workers
  could only run the mesh's connectivity smoke stage; `coordinatorPubkey`
  passes it;
- `kubectl auth can-i get pods/log` asks about a resource literally named
  `pods/log`, so a denial check written that way passes whatever the role
  grants. The RBAC check names subresources with `--subresource`, and one of
  its allowed probes (`patch blutplans --subresource=status`) proves the probe
  can say yes;
- on a host with a load average near 100 the defaults were too short: the
  control plane's controller manager lost its lease and restarted, pod starts
  took two minutes, and a drain waited on a Job finalizer the stalled
  controller manager had not removed. The script's waits are now up to ten
  minutes.

From cluster creation to PASS took 351 s with the images already built.
Building both images took 19 minutes with a warm build cache on the loaded
host; the CI job's time on a hosted runner has not been measured yet.

**Boundaries.** One host: three kind nodes are containers on one kernel, so a
"drained node" is a cordon and eviction, not a machine failing, and no network
was impaired. etcd ran with `--unsafe-no-fsync` (`KIND_ETCD_NO_FSYNC=1`) because
the host's disk was shared with other heavy jobs; CI does not set it. A run
without it on the same host, at a load average near 50, failed at check 1
with an API-server timeout, so the pass above is not evidence for an
fsync-bound etcd under contention. MinIO over
plain HTTP inside the cluster, static keys only. No GPUs, no `ReadWriteMany`
PVC cache, no managed Kubernetes, no IAM roles for service accounts, no cluster
larger than three nodes. A `BlutWorkerPool` was brought up and kept up; mesh
traffic between its pods was not exercised. Operator restart was a pod deletion
with the Deployment replacing it, not a crash during a status write.

## Recorded run: `--launcher k8s` from the CLI (2026-10-30)

Until this run, `recipe run --launcher k8s` ran every stage on the submitting
machine: the k8s target fell through to the local launcher. It now submits a
`BlutPlan` and follows it (docs/OPERATOR.md, "Submitting from the CLI").

`scripts/k8s_kind_e2e.sh` as committed, all fifteen checks, on a fresh kind
cluster (kind 0.31.0, Kubernetes 1.35.0, one control plane and two workers),
459 s with the images prebuilt. Checks 1 to 10 are the operator checks
recorded above; they passed again with this operator build. Checks 11 to 15
run the CLI from the cookbook image, the binary a user installs, on the host
network with the script's kubeconfig and kubectl mounted in:

```
check 11: the CLI submits a recipe file and reports Succeeded
  exit 0; done — succeeded: 3 stage(s), 0 from cache, 3 computed
  (BlutPlan blut-e2e/split-20261030-163324-773990601);
  s3://blut-store/run11: 3 artifact objects
check 12: SIGTERM to the following CLI deletes the plan
  exit 1; BlutPlan, Job and pods of split-20261030-163335-714309565 gone
check 13: blut cancel <job> from another process deletes the plan
  blut cancel 20261030-163342-233697062: follower exited 1;
  BlutPlan split-20261030-163342-233697062 and its pods gone
check 14: recipe run <name> --launcher k8s submits a registry recipe
  BlutPlan eval-only-20261030-163352-404603962 ran recipe run eval_only
  in the pod: Failed, CLI exit 1
check 15: the CLI refuses without a kubeconfig, and without the CRDs
  no kubeconfig: exit 1, kubectl has no current context
  no CRD: exit 1, names blut-operator and kubectl apply
PASS: all checks
```

What the CLI printed in check 11, and the two refusals in check 15:

```
recipe split
job    20261030-163324-773990601
plan   blutplan/split-20261030-163324-773990601 in namespace blut-e2e (blut-operator runs it)
       Ctrl-C or `blut cancel 20261030-163324-773990601` deletes it
[k8s] Pending waiting for a pod to start
[k8s] Running running
[k8s] Running finishing
[k8s] Succeeded succeeded: 3 stage(s), 0 from cache, 3 computed
  filter_dataset                   ok
  materialize_dataset_path         ok
  split_train_eval                 ok
done — succeeded: 3 stage(s), 0 from cache, 3 computed (BlutPlan blut-e2e/split-20261030-163324-773990601)

Error: --launcher k8s: kubectl has no current context (error: current-context is not set).
Point KUBECONFIG at the cluster that runs blut-operator, or set BLUT_K8S_CONTEXT. Nothing
was submitted and nothing ran locally

Error: --launcher k8s: the cluster has no BlutPlan resource, so blut-operator is not
installed there; install blut-operator with `kubectl apply -f crates/blut-operator/deploy/crds.yaml
-f ... operator.yaml -f ... rbac.yaml` (docs/OPERATOR.md), or submit a BlutPlan yourself
with `kubectl apply`. Nothing was submitted and nothing ran locally
```

Check 14 ends `Failed` on purpose. The standard cookbook has no registry
recipes, and the core cookbook's are training and evaluation recipes that
need a model and a Python trainer. `eval_only` with a checkpoint path that
does not exist is submitted, the pod runs `recipe run eval_only`, the pod's
compile refuses the path, and the CLI reports the plan's failure with the
pod's error and exits 1:

```
Error: BlutPlan blut-e2e/eval-only-20261030-163352-404603962 failed: out of retries: last pod
eval-only-20261030-163352-404603962-run-gd6s6 failed: Error, exit 1, recipe compile failed:
recipe args invalid (checkpoint_path): not found: /nonexistent/ckpt
```

**What it found.** The first run of check 14 was refused before submission:
the CLI compiled the named recipe locally, and the cookbook's compile checks
that `checkpoint_path` exists, which it does in the pod and not on the
submitting machine. A named recipe's args are now checked against its schema
on the submitting machine and compiled only in the pod. The follower also
printed `Pending` between the pod exiting 0 and the Job being marked
Complete; the operator now reports that as `Running, finishing` (the
`[k8s] Running finishing` line above).

**Boundaries.** One host, as above, with etcd run with `--unsafe-no-fsync`.
No named recipe ran to `Succeeded` through the cluster: none that is CPU-only
and self-contained exists in the cookbooks. The CLI ran in a container on the
host network, not on a separate workstation; there was no network between it
and the API server to lose. Two failure paths of the follower were not
exercised, here or by a test: no operator reporting within
`BLUT_K8S_PICKUP_SECS`, and losing the API server while following.

## Required evidence before stronger claims

- Mid-stage checkpoint resume on GPUs, over NCCL, and for FSDP, whose
  sharded state is not checkpointed mid-stage. Resume after a requeue, a
  `plan resume` and a retry is recorded above for CPU ranks, with node loss,
  replica hashes, topology, versions and global-batch parity.
- Fairness of the BBR opt-in against competing traffic, and gossip and mesh
  stage data under injected loss and partition. Two-host dispatch, the
  four-continent mesh, gossip and stage data between independent hosts, the
  emulated matrix and real-WAN impairment of dispatch are recorded above.
- Run the S3 path with IAM-role (STS) credentials, and against AWS S3 itself.
- Run the Kubernetes operator on a managed cluster (EKS, GKE or AKS) with IAM
  roles for service accounts, a `ReadWriteMany` PVC cache, GPUs, and mesh
  dispatch between `BlutWorkerPool` pods.
- Publish cookbook-specific DDP evidence in each cookbook repository. Engine
  tests prove orchestration contracts, not model-training correctness.

Exact test counts belong to CI at a commit SHA and are intentionally omitted;
counts drift while capability evidence should remain auditable.
