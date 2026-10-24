#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
#
# End-to-end test of blut-operator on a real kind cluster (ADR 0067 T4.5).
#
# Builds the operator image and a cookbook image (the standard cookbook built
# against this engine checkout), creates a kind cluster of one control plane
# and two workers, installs the shipped manifests from crates/blut-operator/deploy,
# runs MinIO in the cluster with credentials in a Secret, and checks:
#
#   1. a real CPU plan (materialize -> filter -> split) runs to Succeeded and
#      the status carries every stage's state and the cache counts;
#   2. every stage's output is in the S3 store;
#   3. a second plan with the same recipe and store computes nothing (3/3 from
#      cache);
#   4. bad specs are refused: by the API server's schema (Restricted ceiling,
#      empty image, spec edits) and by the operator (credentials in a URL, an
#      unknown recipe extension, args on a .json recipe), with no Job created;
#   5. deleting a running plan removes its Job, pod and ConfigMap;
#   6. a pod deleted mid-run is retried, the retry reuses the finished stages
#      from the store, and the status names the lost pod;
#   7. a drained node's pod is retried elsewhere without spending a retry;
#   8. restarting the operator mid-run creates no second Job or pod and the
#      plan still finishes;
#   9. a BlutWorkerPool's StatefulSet comes up, its ready count reaches the
#      pool's status, a deleted pod is replaced, and deleting the pool removes
#      its StatefulSet and Service (mesh traffic between the pods is not tested);
#  10. the operator's service account can do exactly what deploy/rbac.yaml
#      says and nothing else (no Secrets, no deletes, no pods/exec).
#
# Requires: docker (with buildx), kind, kubectl, and a checkout of
# https://github.com/Quitetall/blut-cookbooks in BLUT_COOKBOOK_DIR.
#
# Environment:
#   BLUT_COOKBOOK_DIR   cookbook checkout (required unless SKIP_BUILD=1)
#   OPERATOR_IMAGE      default blut-operator:e2e
#   COOKBOOK_IMAGE      default blut-cookbook:e2e
#   SKIP_BUILD=1        use OPERATOR_IMAGE / COOKBOOK_IMAGE as already built
#   KIND_CLUSTER        default blut-e2e
#   KIND_NODE_IMAGE     node image for `kind create cluster --image`
#   NODE_MEMORY         per-node memory cap applied with `docker update` (3g)
#   KIND_ETCD_NO_FSYNC=1  run etcd with --unsafe-no-fsync, for a host whose disk
#                       is too busy for etcd's fsync latency (a test cluster only)
#   EVIDENCE_DIR        if set, outputs and logs are saved there
#   KEEP_CLUSTER=1      leave the cluster up afterwards
#   REUSE_CLUSTER=1     reuse a kept cluster of the same name: its BLUT
#                       namespaces and CRDs are deleted first (for iterating)
#   DOCKER_BUILDER      buildx builder (default: the docker driver, `default`)
#
# Exits non-zero on the first failed check.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="${KIND_CLUSTER:-blut-e2e}"
OPERATOR_IMAGE="${OPERATOR_IMAGE:-blut-operator:e2e}"
COOKBOOK_IMAGE="${COOKBOOK_IMAGE:-blut-cookbook:e2e}"
HOLD_IMAGE="${COOKBOOK_IMAGE%%:*}-hold:e2e"
# MinIO no longer publishes tagged images publicly; this digest is
# RELEASE.2025-09-07T16-13-09Z, which also ships `mc`.
MINIO_IMAGE_REF="minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e"
MINIO_IMAGE="blut-e2e/minio:2025-09-07"
NODE_MEMORY="${NODE_MEMORY:-3g}"
NS=blut-e2e
OPNS=blut-system
DEPLOY="$REPO/crates/blut-operator/deploy"
BUILDER="${DOCKER_BUILDER:-default}"
EVIDENCE="${EVIDENCE_DIR:-}"
WORK="$(mktemp -d)"
T0=$(date +%s)

log() { printf '[%4ds] %s\n' "$(($(date +%s) - T0))" "$*"; }
CLUSTER_UP=0
fail() {
  log "FAIL: $*" >&2
  [ "$CLUSTER_UP" = 1 ] || exit 1
  {
    kubectl get blutplans,jobs,pods -A -o wide || true
    kubectl -n "$OPNS" logs deploy/blut-operator --tail=80 || true
  } >&2 2>&1
  exit 1
}
save() { [ -n "$EVIDENCE" ] && cat > "$EVIDENCE/$1" || cat > /dev/null; }

cleanup() {
  status=$?
  if [ -n "$EVIDENCE" ] && kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
    kubectl -n "$OPNS" logs deploy/blut-operator --tail=-1 > "$EVIDENCE/operator.log" 2>&1 || true
    kubectl get events -A --sort-by=.lastTimestamp > "$EVIDENCE/events.txt" 2>&1 || true
  fi
  if [ "${KEEP_CLUSTER:-0}" != 1 ]; then
    kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK"
  log "exit $status"
}
trap cleanup EXIT

for tool in docker kind kubectl; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 2; }
done
[ -n "$EVIDENCE" ] && mkdir -p "$EVIDENCE"

# ── images ──────────────────────────────────────────────────────────────────
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  : "${BLUT_COOKBOOK_DIR:?set BLUT_COOKBOOK_DIR to a blut-cookbooks checkout}"
  log "building $OPERATOR_IMAGE and $COOKBOOK_IMAGE"
  docker buildx build --builder "$BUILDER" --load -f "$REPO/docker/Dockerfile.operator" \
    -t "$OPERATOR_IMAGE" "$REPO" > "$WORK/build-operator.log" 2>&1 &
  op_pid=$!
  docker buildx build --builder "$BUILDER" --load -f "$REPO/docker/Dockerfile.cookbook" \
    --build-context engine="$REPO" --build-context cookbook="$BLUT_COOKBOOK_DIR" \
    -t "$COOKBOOK_IMAGE" "$REPO" > "$WORK/build-cookbook.log" 2>&1 &
  cb_pid=$!
  wait "$op_pid" || { tail -40 "$WORK/build-operator.log"; fail "operator image build"; }
  wait "$cb_pid" || { tail -40 "$WORK/build-cookbook.log"; fail "cookbook image build"; }
  log "images built"
fi

# Test-only image: the cookbook image plus `blut-hold`, which runs the plan
# and then keeps the pod alive for HOLD seconds. It gives the disruption
# checks a running pod whose stages have all finished and been written to the
# store; nothing in the operator knows about it.
cat > "$WORK/Dockerfile.hold" <<EOF
FROM $COOKBOOK_IMAGE
USER 0
RUN printf '%s\n' '#!/bin/sh' \
  'trap "exit 143" TERM INT' \
  'blut "\$@"; rc=\$?' \
  'sleep "\${BLUT_E2E_HOLD:-60}" & wait \$!' \
  'exit \$rc' > /usr/local/bin/blut-hold && chmod 0755 /usr/local/bin/blut-hold
USER 65532:65532
EOF
docker buildx build --builder "$BUILDER" --load -q -t "$HOLD_IMAGE" -f "$WORK/Dockerfile.hold" "$WORK" > /dev/null
docker image inspect "$MINIO_IMAGE" > /dev/null 2>&1 || {
  docker pull -q "$MINIO_IMAGE_REF" > /dev/null
  docker tag "$MINIO_IMAGE_REF" "$MINIO_IMAGE"
}

# ── cluster ─────────────────────────────────────────────────────────────────
cat > "$WORK/kind.yaml" <<'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
EOF
if [ "${KIND_ETCD_NO_FSYNC:-0}" = 1 ]; then
  cat >> "$WORK/kind.yaml" <<'EOF'
kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    etcd:
      local:
        extraArgs:
          unsafe-no-fsync: "true"
EOF
fi
if [ "${REUSE_CLUSTER:-0}" = 1 ] && kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  log "reusing kind cluster $CLUSTER"
  kubectl config use-context "kind-$CLUSTER" > /dev/null
  CLUSTER_UP=1
  kubectl delete namespace "$NS" "$OPNS" --ignore-not-found --wait=true --timeout=300s > /dev/null
  kubectl delete -f "$DEPLOY/crds.yaml" --ignore-not-found --wait=true > /dev/null
  kubectl delete clusterrolebinding,clusterrole blut-operator --ignore-not-found > /dev/null
  for node in $(kubectl get nodes -o name); do kubectl uncordon "$node" > /dev/null; done
else
  log "creating kind cluster $CLUSTER"
  kind create cluster --name "$CLUSTER" --config "$WORK/kind.yaml" --wait 180s \
    ${KIND_NODE_IMAGE:+--image "$KIND_NODE_IMAGE"} > "$WORK/kind-create.log" 2>&1 \
    || { cat "$WORK/kind-create.log"; fail "kind create cluster"; }
  CLUSTER_UP=1
fi
for node in $(kind get nodes --name "$CLUSTER"); do
  docker update --memory "$NODE_MEMORY" --memory-swap "$NODE_MEMORY" "$node" > /dev/null
done
kubectl version -o json | save k8s-version.json
kubectl get nodes -o wide | save nodes.txt
log "loading images"
kind load docker-image --name "$CLUSTER" "$OPERATOR_IMAGE" "$COOKBOOK_IMAGE" "$HOLD_IMAGE" "$MINIO_IMAGE" > /dev/null

# ── install the operator from the shipped manifests ──────────────────────────
log "installing CRDs, RBAC and the operator"
kubectl apply -f "$DEPLOY/crds.yaml" > /dev/null
kubectl wait --for=condition=Established --timeout=60s \
  crd/blutplans.blut.lamquant.dev crd/blutworkerpools.blut.lamquant.dev > /dev/null
kubectl apply -f "$DEPLOY/operator.yaml" -f "$DEPLOY/rbac.yaml" > /dev/null
kubectl -n "$OPNS" set image deploy/blut-operator operator="$OPERATOR_IMAGE" > /dev/null
kubectl -n "$OPNS" rollout status deploy/blut-operator --timeout=300s > /dev/null \
  || fail "operator did not become ready"

# ── MinIO with throwaway credentials, kept in Secrets only ───────────────────
kubectl create namespace "$NS" > /dev/null
MINIO_USER="e2e-$(od -An -N4 -tx4 /dev/urandom | tr -d ' ')"
MINIO_PASSWORD="$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')"
kubectl -n "$NS" create secret generic minio-root \
  --from-literal=MINIO_ROOT_USER="$MINIO_USER" \
  --from-literal=MINIO_ROOT_PASSWORD="$MINIO_PASSWORD" > /dev/null
kubectl -n "$NS" create secret generic blut-store-credentials \
  --from-literal=AWS_ACCESS_KEY_ID="$MINIO_USER" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$MINIO_PASSWORD" \
  --from-literal=AWS_REGION=us-east-1 \
  --from-literal=AWS_ENDPOINT="http://minio.$NS.svc.cluster.local:9000" \
  --from-literal=AWS_ALLOW_HTTP=true > /dev/null
unset MINIO_PASSWORD
kubectl -n "$NS" apply -f - > /dev/null <<EOF
apiVersion: apps/v1
kind: Deployment
metadata: { name: minio }
spec:
  replicas: 1
  selector: { matchLabels: { app: minio } }
  template:
    metadata: { labels: { app: minio } }
    spec:
      # Pinned to the control plane, so draining a worker never takes the
      # store (an emptyDir) with it.
      nodeSelector: { node-role.kubernetes.io/control-plane: "" }
      tolerations: [{ key: node-role.kubernetes.io/control-plane, effect: NoSchedule }]
      containers:
        - name: minio
          image: $MINIO_IMAGE
          imagePullPolicy: Never
          args: ["server", "/data"]
          envFrom: [{ secretRef: { name: minio-root } }]
          ports: [{ containerPort: 9000 }]
          readinessProbe: { httpGet: { path: /minio/health/ready, port: 9000 } }
          resources: { limits: { memory: 768Mi } }
          volumeMounts: [{ name: data, mountPath: /data }]
      # In memory: MinIO takes a drive offline (HTTP 503) when a write stalls,
      # which a busy host disk makes happen; the e2e tests BLUT, not the disk.
      volumes: [{ name: data, emptyDir: { medium: Memory, sizeLimit: 256Mi } }]
---
apiVersion: v1
kind: Service
metadata: { name: minio }
spec:
  selector: { app: minio }
  ports: [{ port: 9000 }]
EOF
kubectl -n "$NS" rollout status deploy/minio --timeout=300s > /dev/null || fail "minio not ready"

# Run `mc` against the in-cluster MinIO; the alias is built from the Secret
# inside the pod, so the password never reaches this script's output. The
# alias is retried: on a loaded node the Service can lag the pod's readiness.
mc_run() {
  local name="mc-$(od -An -N3 -tx1 /dev/urandom | tr -d ' ')"
  kubectl -n "$NS" run "$name" --quiet --restart=Never --image="$MINIO_IMAGE" \
    --image-pull-policy=Never --command \
    --overrides='{"spec":{"containers":[{"name":"'"$name"'","image":"'"$MINIO_IMAGE"'","imagePullPolicy":"Never","envFrom":[{"secretRef":{"name":"minio-root"}}],"command":["sh","-c","for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do mc alias set s http://minio:9000 \"$MINIO_ROOT_USER\" \"$MINIO_ROOT_PASSWORD\" >/dev/null 2>&1 \u0026\u0026 break; sleep 4; done; '"$1"'"]}]}}' \
    -- true > /dev/null
  kubectl -n "$NS" wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$name" --timeout=240s > /dev/null \
    || { kubectl -n "$NS" logs "$name" >&2 || true; fail "mc: $1"; }
  kubectl -n "$NS" logs "$name"
  kubectl -n "$NS" delete pod "$name" --wait=false > /dev/null
}
mc_run "mc mb --ignore-existing s/blut-store" > /dev/null
log "MinIO up, bucket created"

# ── the plan: a recipe and its input in one ConfigMap ─────────────────────────
python3 - "$WORK/train.jsonl" <<'PY'
import json, sys
with open(sys.argv[1], "w") as f:
    for i in range(400):
        f.write(json.dumps({"messages": [
            {"role": "user", "content": f"question {i}"},
            {"role": "assistant", "content": f"answer {i}"}]}) + "\n")
PY
cat > "$WORK/split.toml" <<'EOF'
name = "k8s_e2e_split"

[[stages]]
stage = "materialize_dataset_path"
args = { path = "/blut/plan/train.jsonl" }

[[stages]]
stage = "filter_dataset"
args = { min_turns = 1 }

[[stages]]
stage = "split_train_eval"
args = { eval_ratio = 0.25, seed = 7 }
EOF
kubectl -n "$NS" create configmap split-recipe \
  --from-file=split.toml="$WORK/split.toml" --from-file=train.jsonl="$WORK/train.jsonl" > /dev/null

# plan NAME IMAGE COMMAND PREFIX [MAX_RETRIES]
plan() {
  kubectl -n "$NS" apply -f - > /dev/null <<EOF
apiVersion: blut.lamquant.dev/v1alpha1
kind: BlutPlan
metadata: { name: $1 }
spec:
  cookbookImage: $2
  command: $3
  dataClassCeiling: Internal
  maxRetries: ${5:-2}
  timeoutSeconds: 900
  resources: { cpuMillis: 500, memoryMib: 512 }
  store: { url: "s3://blut-store/$4", credentialsSecret: blut-store-credentials }
  planSource: { configMapRef: { name: split-recipe, key: split.toml } }
EOF
}
# Reads tolerate a slow API server (a timed-out GET is retried by the caller's
# loop, not fatal under `set -e`).
phase() { kubectl -n "$NS" get blutplan "$1" -o jsonpath='{.status.phase}' 2>/dev/null || true; }
field() { kubectl -n "$NS" get blutplan "$1" -o jsonpath="{.status.$2}" 2>/dev/null || true; }
wait_phase() { # NAME PHASE TIMEOUT
  local deadline=$(( $(date +%s) + $3 ))
  until [ "$(phase "$1")" = "$2" ]; do
    [ "$(phase "$1")" = Failed ] && [ "$2" != Failed ] && fail "$1 Failed: $(field "$1" message)"
    [ "$(date +%s)" -lt "$deadline" ] || fail "$1 not $2 within $3 s (phase '$(phase "$1")')"
    sleep 2
  done
}
pods_of() { kubectl -n "$NS" get pods -l "blut.lamquant.dev/plan=$1" -o name | wc -l; }
running_pod() { # NAME -> pod name once Running
  local deadline=$(( $(date +%s) + 600 )) p
  while :; do
    p=$(kubectl -n "$NS" get pods -l "blut.lamquant.dev/plan=$1" --field-selector=status.phase=Running -o name | head -n1)
    [ -n "$p" ] && { echo "${p#pod/}"; return; }
    [ "$(date +%s)" -lt "$deadline" ] || fail "no running pod for $1"
    sleep 1
  done
}
next_pod() { # NAME PREVIOUS-POD -> a running pod other than PREVIOUS-POD
  local deadline=$(( $(date +%s) + 600 )) p
  while :; do
    p=$(kubectl -n "$NS" get pods -l "blut.lamquant.dev/plan=$1" --field-selector=status.phase=Running \
          --no-headers -o custom-columns=NAME:.metadata.name,DELETING:.metadata.deletionTimestamp \
        | awk -v prev="$2" '$2 == "<none>" && $1 != prev { print $1; exit }')
    [ -n "$p" ] && { echo "$p"; return; }
    [ "$(date +%s)" -lt "$deadline" ] || fail "no replacement pod for $1"
    sleep 1
  done
}
# Wait until the run inside a held pod has finished (its summary is written).
wait_held_run_done() { # POD
  local deadline=$(( $(date +%s) + 600 ))
  until kubectl -n "$NS" logs "$1" 2>/dev/null | grep -q '^done — '; do
    [ "$(date +%s)" -lt "$deadline" ] || fail "run in $1 did not finish"
    sleep 2
  done
}
expect_stages() { # NAME STATE HITS MISSES
  local stages; stages=$(field "$1" stages)
  for s in materialize_dataset_path filter_dataset split_train_eval; do
    echo "$stages" | grep -q "\"$s\":\"$2\"" || fail "$1: stage $s not $2 in $stages"
  done
  [ "$(field "$1" cacheHits)" = "$3" ] || fail "$1: cacheHits $(field "$1" cacheHits), want $3"
  [ "$(field "$1" cacheMisses)" = "$4" ] || fail "$1: cacheMisses $(field "$1" cacheMisses), want $4"
}

# 1-2. a real plan, status, outputs in the store ─────────────────────────────
log "check 1: plan runs to Succeeded"
plan split "$COOKBOOK_IMAGE" blut run1
wait_phase split Succeeded 600
expect_stages split ok 0 3
kubectl -n "$NS" get blutplan split -o yaml | save plan-split.yaml
kubectl -n "$NS" get blutplan split
log "  $(field split message)"
log "check 2: stage outputs are in the store"
objects=$(mc_run "mc ls --recursive s/blut-store/run1/")
echo "$objects" | save store-run1.txt
n_art=$(echo "$objects" | grep -c ' cache/v1/artifacts/' || true)
n_inv=$(echo "$objects" | grep -c ' cache/v1/cache-invocations/' || true)
[ "$n_art" -ge 3 ] && [ "$n_inv" -ge 3 ] || fail "store holds $n_art artifacts, $n_inv invocations; want >= 3 each"
log "  s3://blut-store/run1: $n_art artifact objects, $n_inv invocation records"

# 3. same recipe, same store: nothing recomputed ─────────────────────────────
log "check 3: a rerun is served from the store"
plan split-again "$COOKBOOK_IMAGE" blut run1
wait_phase split-again Succeeded 600
expect_stages split-again cached 3 0
log "  $(field split-again message)"

# 4. bad specs ────────────────────────────────────────────────────────────────
log "check 4: bad specs are refused"
reject_by_api() { # NAME YAML-SPEC-BODY EXPECTED-SUBSTRING
  if out=$(printf 'apiVersion: blut.lamquant.dev/v1alpha1\nkind: BlutPlan\nmetadata: { name: %s }\nspec:\n%s\n' "$1" "$2" \
      | kubectl -n "$NS" apply -f - 2>&1); then
    fail "$1 was accepted by the API server"
  fi
  echo "$out" | grep -qF "$3" || fail "$1 refused, but not with '$3': $out"
  echo "$1: $out"
}
spec_ok='  cookbookImage: x
  planSource: { inline: { content: "{}" } }'
{
  reject_by_api restricted "$spec_ok
  dataClassCeiling: Restricted" 'Unsupported value: "Restricted"'
  reject_by_api empty-image '  cookbookImage: ""
  planSource: { inline: { content: "{}" } }' 'cookbookImage must not be empty'
  reject_by_api many-retries "$spec_ok
  maxRetries: 50" 'maxRetries must be at most 20'
  reject_by_api file-store "$spec_ok
  store: { url: \"relative/dir\" }" 'store.url must be s3://bucket[/prefix] or an absolute path'
  out=$(kubectl -n "$NS" patch blutplan split --type=merge -p '{"spec":{"maxRetries":5}}' 2>&1) \
    && fail "editing a plan's spec was accepted"
  echo "$out" | grep -qF 'BlutPlan spec is immutable' || fail "spec edit refused without the immutability message: $out"
  echo "edit split: $out"
} | save rejected-by-api.txt
reject_by_operator() { # NAME SPEC-BODY EXPECTED-SUBSTRING
  printf 'apiVersion: blut.lamquant.dev/v1alpha1\nkind: BlutPlan\nmetadata: { name: %s }\nspec:\n%s\n' "$1" "$2" \
    | kubectl -n "$NS" apply -f - > /dev/null
  wait_phase "$1" Failed 60
  msg=$(field "$1" message)
  echo "$msg" | grep -qF "$3" || fail "$1 failed without '$3': $msg"
  kubectl -n "$NS" get job "$1-run" > /dev/null 2>&1 && fail "$1 got a Job despite an invalid spec"
  echo "$1: $msg"
}
{
  reject_by_operator creds-in-url "$spec_ok
  store: { url: \"s3://AKIAEXAMPLE:notasecret@bucket/x\" }" 'never in the URL'
  reject_by_operator yaml-recipe '  cookbookImage: x
  planSource: { inline: { content: "a: b", filename: plan.yaml } }' 'must end in .json (PlanSpec), .toml (chain) or .star'
  reject_by_operator json-args "$spec_ok
  args: '{}'" 'args is only for .star recipes'
} | save rejected-by-operator.txt
kubectl -n "$NS" get blutplan creds-in-url -o jsonpath='{.status.message}' | grep -q notasecret \
  && fail "a credential from the URL was echoed into the status"
log "  5 refused by the API server, 3 by the operator, no Job for any"

# 5. delete cleans up ─────────────────────────────────────────────────────────
log "check 5: deleting a running plan removes its Job, pod and ConfigMap"
# An inline recipe this time, so the operator's own ConfigMap is exercised.
# Its one stage reads a file that does not exist; blut-hold keeps the pod up.
kubectl -n "$NS" apply -f - > /dev/null <<EOF
apiVersion: blut.lamquant.dev/v1alpha1
kind: BlutPlan
metadata: { name: doomed }
spec:
  cookbookImage: $HOLD_IMAGE
  command: blut-hold
  planSource:
    inline:
      filename: one.toml
      content: |
        name = "doomed"
        [[stages]]
        stage = "materialize_dataset_path"
        args = { path = "/nonexistent.jsonl" }
EOF
pod=$(running_pod doomed)
kubectl -n "$NS" get configmap doomed-plan > /dev/null || fail "inline plan has no ConfigMap"
deadline=$(( $(date +%s) + 300 ))
until kubectl -n "$NS" logs "$pod" 2>/dev/null | grep -q "'doomed' compiles + kind-checks"; do
  [ "$(date +%s)" -lt "$deadline" ] || fail "the inline recipe was not read in $pod"
  sleep 1
done
kubectl -n "$NS" delete blutplan doomed --wait=true --timeout=60s > /dev/null
deadline=$(( $(date +%s) + 300 ))
while kubectl -n "$NS" get job doomed-run > /dev/null 2>&1 || kubectl -n "$NS" get pod "$pod" > /dev/null 2>&1 \
    || kubectl -n "$NS" get configmap doomed-plan > /dev/null 2>&1; do
  [ "$(date +%s)" -lt "$deadline" ] || fail "children of a deleted plan are still present"
  sleep 1
done
log "  Job, pod $pod and ConfigMap gone"

# 6. a pod deleted mid-run: retried, finished stages reused ──────────────────
log "check 6: a deleted pod is retried and the retry reuses the store"
plan lose-pod "$HOLD_IMAGE" blut-hold run6
first=$(running_pod lose-pod)
wait_held_run_done "$first"
kubectl -n "$NS" delete pod "$first" --wait=false > /dev/null
second=$(next_pod lose-pod "$first")
wait_held_run_done "$second"
kubectl -n "$NS" logs "$second" | grep '^done — ' | save lose-pod-retry-done.txt
grep -q '3 cache hits, 0 misses' <(kubectl -n "$NS" logs "$second") \
  || fail "the retry recomputed stages: $(kubectl -n "$NS" logs "$second" | grep '^done')"
msg=$(field lose-pod message)
echo "$msg" | grep -q "previous pod $first" || fail "status does not name the lost pod: $msg"
log "  retry $second: $(kubectl -n "$NS" logs "$second" | grep '^done — ')"
log "  status while retrying: $msg"
wait_phase lose-pod Succeeded 600
[ "$(field lose-pod attempts)" = 2 ] || fail "lose-pod attempts $(field lose-pod attempts), want 2"
expect_stages lose-pod cached 3 0
kubectl -n "$NS" get blutplan lose-pod -o yaml | save plan-lose-pod.yaml

# 7. a drained node: retried elsewhere, not counted ─────────────────────────
log "check 7: draining the node under a running plan"
plan drain-me "$HOLD_IMAGE" blut-hold run7 0
first=$(running_pod drain-me)
node=$(kubectl -n "$NS" get pod "$first" -o jsonpath='{.spec.nodeName}')
wait_held_run_done "$first"
kubectl drain "$node" --ignore-daemonsets --delete-emptydir-data --timeout=300s > "$WORK/drain.log" 2>&1 \
  || { cat "$WORK/drain.log"; fail "drain $node"; }
second=$(next_pod drain-me "$first")
other=$(kubectl -n "$NS" get pod "$second" -o jsonpath='{.spec.nodeName}')
[ "$other" != "$node" ] || fail "replacement pod landed on the drained node"
msg=$(field drain-me message)
log "  $first on $node evicted; $second on $other; status: $msg"
echo "$msg" | grep -q 'disrupted (not counted against maxRetries)' || fail "drain not reported as a disruption: $msg"
wait_phase drain-me Succeeded 600
expect_stages drain-me cached 3 0
[ "$(field drain-me attempts)" = 2 ] || fail "drain-me attempts $(field drain-me attempts), want 2"
kubectl uncordon "$node" > /dev/null
kubectl -n "$NS" get blutplan drain-me -o yaml | save plan-drain-me.yaml
log "  maxRetries 0, drained once, Succeeded"

# 8. operator restart mid-run ─────────────────────────────────────────────────
log "check 8: restarting the operator mid-run"
plan restart-me "$HOLD_IMAGE" blut-hold run8
pod=$(running_pod restart-me)
job_uid=$(kubectl -n "$NS" get job restart-me-run -o jsonpath='{.metadata.uid}')
kubectl -n "$OPNS" delete pod -l app.kubernetes.io/name=blut-operator --wait=true > /dev/null
kubectl -n "$OPNS" rollout status deploy/blut-operator --timeout=300s > /dev/null
wait_phase restart-me Succeeded 600
[ "$(kubectl -n "$NS" get jobs -l blut.lamquant.dev/plan=restart-me -o name | wc -l)" = 1 ] \
  || fail "more than one Job for restart-me"
[ "$(kubectl -n "$NS" get job restart-me-run -o jsonpath='{.metadata.uid}')" = "$job_uid" ] \
  || fail "the Job was replaced"
[ "$(pods_of restart-me)" = 1 ] || fail "restart-me ran $(pods_of restart-me) pods"
[ "$(field restart-me attempts)" = 1 ] || fail "restart-me attempts $(field restart-me attempts)"
log "  one Job (uid unchanged), one pod ($pod), Succeeded"
# The operator never re-runs a finished plan, even when its Job is gone.
kubectl -n "$NS" delete job split-run --wait=true > /dev/null
kubectl -n "$OPNS" delete pod -l app.kubernetes.io/name=blut-operator --wait=true > /dev/null
kubectl -n "$OPNS" rollout status deploy/blut-operator --timeout=300s > /dev/null
sleep 10
kubectl -n "$NS" get job split-run > /dev/null 2>&1 && fail "a finished plan was run again"
[ "$(phase split)" = Succeeded ] || fail "split lost its status"
log "  finished plan with its Job deleted: not re-run"

# 9. a worker pool is kept running ───────────────────────────────────────────
log "check 9: a BlutWorkerPool's StatefulSet comes up and its status follows it"
kubectl -n "$NS" apply -f - > /dev/null <<EOF
apiVersion: blut.lamquant.dev/v1alpha1
kind: BlutWorkerPool
metadata: { name: mesh }
spec:
  replicas: 2
  image: $COOKBOOK_IMAGE
  resources: { cpuCores: 1, memoryGib: 1 }
EOF
deadline=$(( $(date +%s) + 600 ))
until [ "$(kubectl -n "$NS" get blutworkerpool mesh -o jsonpath='{.status.readyReplicas}' 2>/dev/null)" = 2 ]; do
  [ "$(date +%s)" -lt "$deadline" ] || fail "pool mesh not 2/2 ready"
  sleep 3
done
kubectl -n "$NS" get svc mesh-workers -o jsonpath='{.spec.clusterIP} {.spec.ports[0].protocol}/{.spec.ports[0].port}' \
  | grep -qx 'None UDP/9320' || fail "pool Service is not headless UDP 9320"
pod0=$(kubectl -n "$NS" get pod mesh-workers-0 -o jsonpath='{.status.phase}')
[ "$pod0" = Running ] || fail "mesh-workers-0 is $pod0"
kubectl -n "$NS" delete pod mesh-workers-1 --wait=false > /dev/null
deadline=$(( $(date +%s) + 600 ))
sleep 5
until [ "$(kubectl -n "$NS" get blutworkerpool mesh -o jsonpath='{.status.readyReplicas}' 2>/dev/null)" = 2 ]; do
  [ "$(date +%s)" -lt "$deadline" ] || fail "pool mesh did not recover a deleted pod"
  sleep 3
done
kubectl -n "$NS" get blutworkerpool mesh -o yaml | save pool-mesh.yaml
kubectl -n "$NS" delete blutworkerpool mesh --wait=true > /dev/null
deadline=$(( $(date +%s) + 300 ))
while kubectl -n "$NS" get statefulset mesh-workers > /dev/null 2>&1 || kubectl -n "$NS" get svc mesh-workers > /dev/null 2>&1; do
  [ "$(date +%s)" -lt "$deadline" ] || fail "a deleted pool's StatefulSet or Service is still present"
  sleep 2
done
log "  2/2 ready, a deleted pod replaced, StatefulSet and Service gone with the pool"

# 10. RBAC ────────────────────────────────────────────────────────────────────
log "check 10: the operator's permissions"
SA="system:serviceaccount:$OPNS:blut-operator"
kubectl auth can-i --list --as="$SA" | save rbac-can-i.txt
# Subresources are named with --subresource: `kubectl auth can-i get pods/log`
# asks about a resource called "pods/log", which nobody has, so a denial check
# written that way passes whatever the role grants.
for allowed in "create jobs.batch -n $NS" "patch blutplans.blut.lamquant.dev --subresource=status -n $NS" \
    "watch pods -n $NS" "create configmaps -n $NS"; do
  kubectl auth can-i ${allowed} --as="$SA" > /dev/null || fail "operator cannot $allowed"
done
for denied in "get secrets -n $NS" "list secrets -A" "delete jobs.batch -n $NS" "delete pods -n $NS" \
    "create pods -n $NS" "create pods --subresource=exec -n $NS" "get pods --subresource=log -n $NS" \
    "update blutplans.blut.lamquant.dev -n $NS" "patch blutworkerpools.blut.lamquant.dev -n $NS" \
    "patch blutplans.blut.lamquant.dev -n $NS" "create deployments.apps -n $NS" "create clusterroles.rbac.authorization.k8s.io"; do
  if kubectl auth can-i ${denied} --as="$SA" > /dev/null 2>&1; then fail "operator can $denied"; fi
  echo "denied: $denied"
done | save rbac-denied.txt
kubectl -n "$OPNS" logs deploy/blut-operator --tail=-1 | grep -i forbidden && fail "operator hit a forbidden call"
log "  allowed 4/4, denied 12/12, no forbidden calls in the operator log"

kubectl -n "$NS" get blutplans | save plans-final.txt
kubectl -n "$NS" get blutplans
log "PASS: all checks"
