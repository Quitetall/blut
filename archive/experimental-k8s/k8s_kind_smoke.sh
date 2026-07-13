#!/usr/bin/env bash
# ARCHIVED: depends on the retired Dockerfile.blut prototype and cannot run
# against the library-only engine. Retained only as historical design evidence.
# End-to-end kind smoke for the BLUT operator (ADR 0067 T4.5 · B5).
#
# Stands up a throwaway kind cluster + MinIO, installs the CRDs, runs the
# operator, applies a trivial BlutPlan, and asserts the operator materialized
# its dispatcher Job. CPU-only — CI-able on any runner with docker + kind
# (GPU / device-plugin / RWX / cross-pod NCCL are infra-gated, T4.6).
#
#   ./scripts/k8s_kind_smoke.sh
#
# Exits non-zero on any failure; tears the cluster down on exit.
set -euo pipefail

CLUSTER="blut-smoke-$$"
NS="blut"
OPERATOR_IMAGE="blut-operator:smoke"
BLUT_IMAGE="blut:smoke"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

need() { command -v "$1" >/dev/null 2>&1 || { echo ">> missing required tool: $1" >&2; exit 1; }; }
need kind
need kubectl
need docker

cleanup() { kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo ">> creating kind cluster $CLUSTER"
kind create cluster --name "$CLUSTER" --wait 120s

echo ">> building + loading images (this is the slow part)"
docker build -f "$REPO_ROOT/docker/Dockerfile.operator" -t "$OPERATOR_IMAGE" "$REPO_ROOT"
docker build -f "$REPO_ROOT/docker/Dockerfile.blut" -t "$BLUT_IMAGE" "$REPO_ROOT"
kind load docker-image "$OPERATOR_IMAGE" "$BLUT_IMAGE" --name "$CLUSTER"

echo ">> installing CRDs"
docker run --rm "$OPERATOR_IMAGE" crds | kubectl apply -f -
kubectl create namespace "$NS"

echo ">> deploying the operator"
kubectl -n "$NS" create serviceaccount blut-operator
# Cluster-admin for the smoke; a tight Role is the production posture.
kubectl create clusterrolebinding "blut-operator-$CLUSTER" \
  --clusterrole=cluster-admin --serviceaccount="$NS:blut-operator"
kubectl -n "$NS" apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata: { name: blut-operator }
spec:
  replicas: 1
  selector: { matchLabels: { app: blut-operator } }
  template:
    metadata: { labels: { app: blut-operator } }
    spec:
      serviceAccountName: blut-operator
      containers:
        - name: operator
          image: $OPERATOR_IMAGE
          imagePullPolicy: Never
          args: ["run"]
YAML
kubectl -n "$NS" rollout status deploy/blut-operator --timeout=120s

echo ">> applying a trivial BlutPlan"
kubectl -n "$NS" apply -f - <<YAML
apiVersion: blut.lamquant.dev/v1alpha1
kind: BlutPlan
metadata: { name: smoke }
spec:
  planSource: { inline: { content: "{}" } }
  cookbookImage: $BLUT_IMAGE
  dataClassCeiling: Internal
YAML

echo ">> waiting for the operator to create the dispatcher Job (smoke-run)"
for _ in $(seq 1 30); do
  if kubectl -n "$NS" get job smoke-run >/dev/null 2>&1; then
    echo ">> dispatcher Job created — operator reconcile works"
    kubectl -n "$NS" get job smoke-run -o jsonpath='{.metadata.ownerReferences[0].kind}' \
      | grep -q BlutPlan && echo ">> Job is owner-ref'd to the BlutPlan (GC wired)"
    echo ">> KIND SMOKE PASSED"
    exit 0
  fi
  sleep 2
done

echo ">> FAILED: operator did not create the dispatcher Job in time" >&2
kubectl -n "$NS" describe blutplan smoke >&2 || true
kubectl -n "$NS" logs deploy/blut-operator >&2 || true
exit 1
