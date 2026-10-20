# BLUT Distributed and Cross-Platform Validation

Evidence ledger for engine `0.2.0-alpha.2`. Status means:

- **Validated:** exercised on named runtime or exact CI gate.
- **Component-tested:** parser, command construction, or loopback component
  tested; no scale claim.
- **Deferred:** unsupported in public preview.

Last updated: 2026-10-20.

Nothing in the table below changed when the crates were published on
2026-08-27. Publication moves code, not evidence: every row still says exactly
what was and was not exercised, and the four Deferred rows are the ones to read
first if you are deciding whether to depend on this.

| Capability | Status | Evidence boundary |
|---|---|---|
| Linux systemd/cgroup2/rlimit containment selection | Validated | Local integration and failure-path tests |
| macOS build with degraded containment | CI-gated | Compile only; no systemd guarantee |
| `blut-types` on wasm32 | CI-gated | Rust 1.88 `wasm32-unknown-unknown` check |
| Single-node GPU admission and per-device permits | Component-tested | Deterministic scheduler and executor tests |
| Single-node `torchrun` argument construction | Component-tested | Launch contract tests; cookbook owns trainer semantics |
| Ray launcher against a live cluster | Validated | Two-node Ray 2.x, submit + stream + poll to Succeeded, 2026-09-19 |
| Slurm launcher: `sbatch` submit, stream, poll, cancel; `srun` steps | Validated | Two-node Slurm 21.08 cluster, with and without accounting, 2026-10-20 — see below |
| Multi-node NCCL training through Slurm | Validated (cookbook) | Two physical machines, 2× RTX 5090, NCCL over a private network, 2026-10-20 — see below |
| P2P crypto, policy, transport, and loopback dispatch | Component-tested | Unit and loopback integration tests |
| P2P dispatch between separate network namespaces | Validated | Two containers, own IPs, real QUIC over a bridge — see below |
| P2P dispatch between independent hosts over the internet | Validated | Oregon ↔ France, both behind NAT, real QUIC over the public internet, 2026-10-20 — see below |
| Live multi-host P2P mesh (N nodes, `p2p node`) | Deferred | Two-host coordinator/worker dispatch only so far |
| Cloud queue over local object storage | Validated | `blut cloud smoke` round-trip through the shipped cookbook binary, 2026-09-19 |
| Network object storage (S3-compatible) | Validated (MinIO) | `s3` feature; full cloud round-trip against MinIO over HTTP, 2026-09-20. Backblaze B2 over TLS refused: no conditional writes, 2026-10-20 — see below |
| `blut-worker` REST prototype | Deleted (ADR 0083 M3) | Superseded by `src/cloud` + the `blut-web` sidecar |
| Kubernetes operator | Unsupported | Unpublished prototype; separate Rust 1.89 compile/test lane |
| Windows containment | Deferred | Bare fallback only; Job Object implementation absent |

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

## Required evidence (Slurm, still open)

- Node loss mid-run and recovery through durable resume.
- A cluster with cgroup enforcement and GPU binding, rather than placeholder
  devices.

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

## Required evidence before stronger claims

- Record multi-node failure recovery, and artifact hashes across nodes. Topology,
  versions and global-batch parity are recorded above.
- Exercise P2P across independent hosts and hostile-network cases.
- Run the S3 path against AWS S3 (or another provider with conditional writes)
  on public TLS, and with IAM-role credentials.
- Publish cookbook-specific DDP evidence in each cookbook repository. Engine
  tests prove orchestration contracts, not model-training correctness.

Exact test counts belong to CI at a commit SHA and are intentionally omitted;
counts drift while capability evidence should remain auditable.
