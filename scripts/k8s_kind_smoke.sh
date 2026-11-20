#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# k8s operator kind smoke (ADR 0083 M6 / ADR 0137): the cheap gate that keeps
# the shipped CRDs (crates/blut-operator/deploy/crds.yaml) installable and their
# schema doing its job on a real API server.
#
# Asserted: a fresh kind cluster comes up, both CRDs apply and become
# Established, a sample BlutPlan is accepted, a Restricted ceiling and a spec
# edit are refused. It does not run the operator; scripts/k8s_kind_e2e.sh does
# (reconcile, status, store, retries, drain, restart, RBAC).
#
# That deploy/crds.yaml matches the Rust types is a unit test in the operator
# crate (`the_committed_crd_manifest_matches_the_types`).
#
# Requires: kind, kubectl. Usage: bash scripts/k8s_kind_smoke.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="blut-smoke"
NS="blut-smoke"
cleanup() { kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

for tool in kind kubectl; do
  command -v "$tool" >/dev/null 2>&1 || { echo "::error::missing required tool: $tool"; exit 1; }
done

echo "==> creating kind cluster '$CLUSTER'"
kind create cluster --name "$CLUSTER" --wait 120s

echo "==> installing operator CRDs"
kubectl apply -f "$REPO/crates/blut-operator/deploy/crds.yaml"
kubectl wait --for=condition=Established --timeout=60s \
  crd/blutplans.blut.lamquant.dev crd/blutworkerpools.blut.lamquant.dev

kubectl create namespace "$NS"

echo "==> applying a sample BlutPlan (schema must validate server-side)"
kubectl apply -n "$NS" -f - <<'YAML'
apiVersion: blut.lamquant.dev/v1alpha1
kind: BlutPlan
metadata:
  name: smoke-plan
spec:
  planSource:
    inline:
      filename: plan.toml
      content: |
        name = "smoke"
        [[stages]]
        stage = "materialize_dataset_path"
        args = { path = "/data/train.jsonl" }
  cookbookImage: "blut-cookbook:smoke"
YAML
kubectl get blutplan -n "$NS" smoke-plan -o name >/dev/null

echo "==> a Restricted ceiling must be refused"
if kubectl apply -n "$NS" -f - 2>/dev/null <<'YAML'
apiVersion: blut.lamquant.dev/v1alpha1
kind: BlutPlan
metadata:
  name: restricted-plan
spec:
  planSource: { inline: { content: "{}" } }
  cookbookImage: "blut-cookbook:smoke"
  dataClassCeiling: Restricted
YAML
then
  echo "::error::a Restricted BlutPlan was accepted"; exit 1
fi

echo "==> a spec edit must be refused"
if kubectl patch blutplan -n "$NS" smoke-plan --type=merge -p '{"spec":{"maxRetries":5}}' 2>/dev/null; then
  echo "::error::a BlutPlan spec edit was accepted"; exit 1
fi

echo "==> PASS: CRDs installed; schema accepts a valid plan and refuses Restricted and edits"
