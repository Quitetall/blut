# BLUT Kubernetes operator

`blut-operator` runs BLUT plans on a Kubernetes cluster. It watches two custom
resources:

- **`BlutPlan`** runs one declarative recipe to completion as a Kubernetes
  `Job`, and reports what happened in the resource's status.
- **`BlutWorkerPool`** keeps a `StatefulSet` of `blut p2p node --no-scheduler`
  pods running behind a headless Service, for the P2P mesh.

Kubernetes is an adapter here, not the control plane. Inside each pod the
engine still compiles the recipe, checks stage kinds, consults the cache and
runs the stages. Kubernetes places the pod, restarts it when it is lost, and
garbage-collects it. The engine has no Kubernetes dependency; all kube-rs code
is in `crates/blut-operator`.

What has been exercised, and on what, is recorded in
[DISTRIBUTED_VALIDATION.md](DISTRIBUTED_VALIDATION.md#recorded-run-the-kubernetes-operator-on-kind-2026-10-24).

## Install

You need two images:

- **the operator**: `docker build -f docker/Dockerfile.operator -t <registry>/blut-operator:<tag> .`
- **a cookbook image** that the plan pods run. It must contain a cookbook
  binary (the standard cookbook's `blut-standard`, LamQuant's `lqt`, or your
  own) built against the engine. A plan runs `blut` unless its spec names
  another `command`. `docker/Dockerfile.cookbook` builds the standard cookbook
  against an engine checkout, with the `p2p` and `s3` features, and installs it
  as `blut`.

No images are published to a registry yet. Build them and push them to one
your cluster can pull from.

Then apply the manifests in `crates/blut-operator/deploy/`:

```bash
kubectl apply -f crates/blut-operator/deploy/crds.yaml
kubectl apply -f crates/blut-operator/deploy/operator.yaml -f crates/blut-operator/deploy/rbac.yaml
kubectl -n blut-system set image deploy/blut-operator operator=<registry>/blut-operator:<tag>
kubectl -n blut-system rollout status deploy/blut-operator
```

or `kubectl apply -k crates/blut-operator/deploy` with an `images:` override in
an overlay. The manifests create the `blut-system` namespace (Pod Security
`restricted`), a service account, a ClusterRole and its binding, and a
one-replica Deployment. `blut-operator crds` prints the same CRDs as
`deploy/crds.yaml`; a unit test fails if the two differ.

To watch a single namespace instead of the whole cluster, run the operator with
`--namespace <ns>` (or `BLUT_OPERATOR_NAMESPACE`) and bind the same rules with a
Role and RoleBinding in that namespace.

Requirements: the operator is built against the Kubernetes 1.32 API and has
been run on 1.35.0 (kind). The Job `podFailurePolicy` that tells disruptions
from failures, and the CEL validation rules in the CRDs, are GA from 1.31 and
1.29 respectively; older clusters are not supported.

## Running a plan

A plan names a recipe and an image:

```yaml
apiVersion: blut.lamquant.dev/v1alpha1
kind: BlutPlan
metadata:
  name: split-example
spec:
  cookbookImage: <registry>/blut-cookbook:<tag>
  planSource:
    configMapRef: { name: split-recipe, key: split.toml }
  store:
    url: s3://blut-store/examples
    credentialsSecret: blut-store-credentials
  resources: { cpuMillis: 500, memoryMib: 512 }
  maxRetries: 2
  timeoutSeconds: 900
```

A complete example, with the ConfigMap holding the recipe and its input, is
`crates/blut-operator/deploy/examples/blutplan.yaml`.

```console
$ kubectl get blutplans
NAME            PHASE       ATTEMPTS   JOB                 AGE
split-example   Succeeded   1          split-example-run   41s
$ kubectl get blutplan split-example -o jsonpath='{.status.message}'
succeeded: 3 stage(s), 0 from cache, 3 computed
```

The operator creates one Job, `<plan>-run`. Its pod runs

```
<command> recipe declare /blut/plan/<file> --run
```

directly as PID 1, so the engine's own SIGTERM handler sees a pod deletion. The
engine writes a JSON run summary to the container's termination-message file
(`BLUT_RUN_SUMMARY=/dev/termination-log`), and the kubelet hands it back in the
pod status. That is how per-stage state and cache counts reach the plan's
status without the operator reading logs or files.

## CRD reference

### `BlutPlan` spec

| Field | Default | Meaning |
|---|---|---|
| `planSource.inline.content` | | The recipe text. The operator copies it into a ConfigMap `<plan>-plan` that the plan owns. At most about 900 KiB. |
| `planSource.inline.filename` | `plan.json` | The file name the engine sees. The extension selects the format: `.json` (PlanSpec), `.toml` (linear chain) or `.star` (Starlark; the image must also carry `blut-dsl`). |
| `planSource.configMapRef.name` | | An existing ConfigMap in the plan's namespace, mounted whole at `/blut/plan`. Other keys (small inputs the recipe reads) are mounted beside the recipe. |
| `planSource.configMapRef.key` | `plan.json` | The key holding the recipe; also its file name. |
| `planSource.recipe.name` | | A recipe compiled into the cookbook binary (`recipe list`). The pod runs `<command> recipe run <name> --args <args>`. |
| `planSource.recipe.args` | `{}` | The recipe's args, as a JSON object in a string. |
| `cookbookImage` | | Image with the cookbook binary. Required, non-empty. |
| `command` | `blut` | The cookbook binary inside the image. One executable, no arguments. |
| `dataClassCeiling` | `Internal` | `Public` or `Internal`. `Restricted` is not a value of this type, so a manifest naming it is refused by the API server (ADR 0061). Stamped on the Job and pod as the label `blut.lamquant.dev/data-class` for admission policies; the engine does not read it. |
| `cache.pvc.claimName` | | Mount a PersistentVolumeClaim as the engine's global cache (`--shared-cache`). Needs `ReadWriteMany` for concurrent pods. |
| `cache.none` | yes | Job-local cache only. |
| `store.url` | | `s3://bucket[/prefix]`, or an absolute directory inside the container. Stage outputs are written through to it; a later pod reads them back instead of recomputing. The image must be built with the engine's `s3` feature for `s3://`. |
| `store.credentialsSecret` | | Secret whose keys become the container's environment (below). |
| `args` | | Args JSON for a `.star` recipe. Refused for `.json` and `.toml`. |
| `resources.cpuMillis`, `resources.memoryMib` | unbounded | Requests and limits, set equal. |
| `gpus` | `0` | `nvidia.com/gpu` request and limit. Not validated on a GPU cluster. |
| `maxRetries` | `2` | Pod failures Kubernetes retries before the plan is `Failed` (`backoffLimit`). At most 20. Disruptions do not count (below). |
| `timeoutSeconds` | none | Wall-clock limit for the whole plan, across retries (`activeDeadlineSeconds`). |

The spec is **immutable**: the API server refuses an edit with "BlutPlan spec is
immutable; delete the plan and create a new one". A plan is the record of one
run; to run again, create a new plan. With the same `store`, the new plan's
stages that already ran are served from it.

### `BlutPlan` status

| Field | Meaning |
|---|---|
| `phase` | `Pending`, `Running`, `Succeeded` or `Failed`. |
| `message` | One line: the run summary on success, the failure on `Failed`, the previous pod's fate while a retry runs. |
| `stages` | Last state of each stage in the final pod's run: `ok`, `cached`, `failed`, `pruned`, `running`, `blocked`, `retrying`. |
| `cacheHits`, `cacheMisses` | Stages served from the cache and computed, in that run. |
| `jobName`, `pod` | The Job, and the latest pod. |
| `attempts` | Pods started, including ones lost to disruptions. |
| `lastPodFailure` | What happened to the most recent pod that failed, was deleted or was evicted. Kept after the pod is gone. |
| `observedGeneration` | The `metadata.generation` the status describes. |

### `BlutWorkerPool` spec

| Field | Default | Meaning |
|---|---|---|
| `replicas` | | Worker pods, at most 1000. |
| `image` | | Image with a cookbook binary built with the `p2p` feature. |
| `command` | `blut` | The binary inside the image. |
| `coordinatorPubkey` | | Hex public key of the scheduler whose signed tasks the workers run (`--coordinator-pubkey`). Without it a worker runs only the mesh's connectivity smoke stage. |
| `cache.pvc.claimName` | | Mounted at `/blut-cache` and passed as `--shared-store`. |
| `resources.cpuCores`, `memoryGib`, `gpus` | 2, 8, 0 | Requests and limits per pod. |

Each pod generates an identity key in an init container
(`p2p keys generate --force`) and runs `blut p2p node --no-scheduler --listen
0.0.0.0:9320`. The key lives on the pod's scratch volume, so a replaced pod has
a new identity; schedulers that pinned the old key must learn the new one. The
node prints its public key at start (`kubectl logs`). The Service
`<pool>-workers` is headless and publishes UDP 9320 (QUIC). `status.readyReplicas`
mirrors the StatefulSet.

## Submitting from the CLI (`--launcher k8s`)

`recipe run` and `recipe declare --run` submit to the cluster themselves when
given `--launcher k8s`:

```bash
export BLUT_K8S_IMAGE=<registry>/blut-cookbook:<tag>
export BLUT_K8S_STORE=s3://blut-store/runs BLUT_K8S_STORE_SECRET=blut-store-credentials
blut recipe declare split.toml --run --launcher k8s     # a recipe file, shipped inline
blut recipe run <name> --args '{...}' --launcher k8s     # a recipe compiled into the cookbook
```

The CLI checks the recipe first, so a mistake fails before anything is
submitted: a named recipe's args against its schema (unknown or missing keys,
wrong types), and a recipe file by compiling it (stage names, kinds). It does
not compile a named recipe locally, because a cookbook's compile can check
that the paths in its args exist, and those are paths inside the pod. It then
creates a `BlutPlan` (`kubectl create`,
never `apply`, so it cannot overwrite another run), records a local job, and
follows the plan's status until it is `Succeeded` or `Failed`, printing each
change. It exits 0 only on `Succeeded`. Ctrl-C, SIGTERM, or `blut cancel <job>`
from another shell deletes the plan, and owner references take its Job and
pods with it. Nothing runs on the submitting machine.

| Variable | Meaning |
|---|---|
| `BLUT_K8S_IMAGE` | The cookbook image the pod runs. Required. |
| `BLUT_K8S_COMMAND` | The cookbook binary in the image (default `blut`). |
| `BLUT_K8S_NAMESPACE` | Namespace (default: the kubectl context's, else `default`). |
| `BLUT_K8S_CONTEXT` | kubectl context (default: the current one). |
| `BLUT_K8S_STORE` | `store.url`. Without it, outputs stay in the pod and the CLI warns. |
| `BLUT_K8S_STORE_SECRET` | `store.credentialsSecret`. Only the Secret's name is sent. |
| `BLUT_K8S_CPU_MILLIS`, `BLUT_K8S_MEMORY_MIB` | `resources` (both or neither). |
| `BLUT_K8S_GPUS`, `BLUT_K8S_MAX_RETRIES`, `BLUT_K8S_TIMEOUT_SECS` | `gpus`, `maxRetries`, `timeoutSeconds`. |
| `BLUT_K8S_PICKUP_SECS` | How long to wait for the operator's first status (default 120). If it never comes, the plan is deleted and the run fails, naming the operator. |
| `BLUT_KUBECTL` | The kubectl binary (default `kubectl` on PATH). |

The CLI refuses, before submitting anything and without running anything
locally, when: kubectl is missing; kubectl has no current context; the
cluster does not answer; the cluster has no `BlutPlan` CRD (the message names
the install and the `kubectl apply` route); the args name `dataset://` or other
registry handles, which resolve against the submitting machine's registries;
the file is a `registry://plan@…` pointer; or a flag the cluster run cannot
honour is set (config mode and sweeps, `--shared-cache`, `--no-cache`,
`--sync-io`, a non-default `--tenant`, `--experiment`). `hpo`, `partition` and
`plan resume` refuse `--launcher k8s` outright. Every local launch path also
refuses a k8s placement, so none can fall back to running on the submitting
machine.

The pod runs the same command against the same cookbook, so it compiles the
same plan, provided the image's cookbook matches the submitting binary's.
Paths in the recipe (datasets) are paths inside the pod.

## Credentials

Store credentials live in a Secret you create. The plan names the Secret; the
operator writes `secretKeyRef` entries for the keys below into the pod spec,
each `optional`, and the kubelet resolves them. The operator never reads a
Secret and its RBAC has no access to Secrets.

| Key | Use |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Static credentials. |
| `AWS_SESSION_TOKEN` | Temporary credentials. |
| `AWS_REGION` | Region (MinIO accepts `us-east-1`). |
| `AWS_ENDPOINT` | An S3-compatible endpoint, e.g. `http://minio.store.svc:9000`. |
| `AWS_ALLOW_HTTP` | `true` for a plain-HTTP endpoint. |

A `store.url` that carries userinfo (`s3://key:secret@bucket`) is refused, and
the refusal does not repeat it. The engine probes the store once before any
stage runs, so wrong credentials or an unreachable endpoint fail the plan
before any work is done (an endpoint refusing connections took about five
seconds, through the S3 client's ten retries) instead of producing a run whose
outputs never left the pod.

The store must support conditional writes (`If-None-Match: *`); see the S3 rows
of DISTRIBUTED_VALIDATION.md for which providers were checked.

## Failures, retries and deletion

- **A stage fails.** The engine exits non-zero, the pod fails, and Kubernetes
  starts another pod, up to `maxRetries`. Then the plan is `Failed` with
  `out of retries: last pod <name> failed: <the engine's error>`.
- **A pod is deleted** (`kubectl delete pod`). Counted as a failure and retried.
  The status names the deleted pod.
- **A node is drained, a pod is preempted or evicted.** Kubernetes marks the pod
  with the `DisruptionTarget` condition, and the Job's `podFailurePolicy` retries
  it without spending one of `maxRetries`. The status says
  `was disrupted (not counted against maxRetries)`.
- **A retry does not redo finished work** when the plan has a `store` (or a
  shared PVC cache): each finished stage's output was written through, and the
  next pod reads it back. Without one, a retry starts from scratch.
- **Deleting a plan** deletes its Job, pods and ConfigMap through owner
  references. A running pod gets SIGTERM; the engine cancels the stage.
- **The operator restarts.** It finds the existing Job and carries on. It never
  creates a second Job for a plan, and it never re-runs a plan whose status is
  already `Succeeded` or `Failed`, even if that plan's Job has been deleted.
- **A plan's Job is deleted while the plan is still running.** The operator
  creates a new one; with a `store`, finished stages are not recomputed.

## Containment inside a pod

Pods have no systemd user bus, so the engine's cgroup containment cannot start
transient units there. The operator sets `BLUT_NO_CONTAIN=1`, and the pod's own
cgroup, sized by `resources`, is the containment. Exceeding `memoryMib` is an
OOM kill of the pod, which then counts as a failed attempt.

The engine's memory admission reads the node's `/proc/meminfo`, not the pod's
cgroup limit, so it can admit a stage the pod limit will then kill. Size
`resources.memoryMib` for the plan's largest stage.

Each plan container runs with privilege escalation off, all capabilities
dropped, the runtime's default seccomp profile, no service-account token and
no service links. The cookbook image built from `docker/Dockerfile.cookbook`
runs as UID 65532. Every path the engine writes (`HOME`, XDG directories, job
directories, temp files) is on an `emptyDir`.

## RBAC

`deploy/rbac.yaml`, cluster-wide:

| Resource | Verbs | Why |
|---|---|---|
| `blutplans`, `blutworkerpools` | get, list, watch | Watch the resources. |
| `blutplans/status`, `blutworkerpools/status` | get, patch | Write status. Nothing else of the resource. |
| `jobs` | get, list, watch, create | One Job per plan. |
| `configmaps` | get, create | An inline plan's recipe, read back to check it is the plan's own. |
| `pods` | list, watch | Read the run summary and notice lost pods. |
| `statefulsets` | get, list, watch, create, patch | Server-side apply of a pool. |
| `services` | get, create, patch | Server-side apply of a pool's Service. |

It has no `delete` on anything (owner references do the cleanup), no access to
Secrets, no `pods/exec` or `pods/log`, and cannot create pods directly.

## Limits

- **No published images.** Build both images yourself.
- **Starlark recipes** need the `blut-dsl` binary in the cookbook image; the
  image from `docker/Dockerfile.cookbook` does not include it.
- **One operator replica.** There is no leader election. Two replicas would not
  run a plan twice (the Job name is fixed and the second create is refused) but
  would race on status. The Deployment uses `Recreate`.
- **Not exercised:** GPUs, multi-node training inside one plan, `ReadWriteMany`
  PVC caches, managed Kubernetes (EKS, GKE, AKS), IAM roles for service
  accounts, clusters larger than three nodes, and `BlutWorkerPool` mesh traffic
  between pods. A pool's StatefulSet and Service are built and unit-tested;
  mesh dispatch across pool pods has not been run.
- **Engine CLI surface.** The operator relies on `recipe declare <file> --run`,
  `BLUT_REMOTE_CACHE`, `BLUT_RUN_SUMMARY` and `BLUT_NO_CONTAIN`. A cookbook
  binary built against an engine older than 0.2.0-alpha.2 has the first and the
  last only: plans run, but outputs stay in the pod and the status has no
  per-stage detail.
- **Status granularity.** Per-stage state arrives when a pod finishes, not
  while it runs.
