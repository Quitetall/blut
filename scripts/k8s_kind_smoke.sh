#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# k8s operator kind smoke (ADR 0083 M6 / ADR 0137) — the gate that keeps the
# `blut-operator` CRD schemas installable and a `BlutPlan` custom resource
# accepted by a real API server, on an ephemeral `kind` cluster.
#
# Reliable core (always asserted): a fresh kind cluster comes up, the operator's
# emitted CRDs apply cleanly, and a sample BlutPlan CR is ACCEPTED (its schema
# validates server-side — a malformed CRD or a breaking schema change fails
# here). Best-effort (asserted only if the operator image is available):
# the reconcile loop turns the plan into a Job.
#
# Requires: kind, kubectl, cargo. Usage: bash scripts/k8s_kind_smoke.sh
set -euo pipefail

CLUSTER="blut-smoke"
NS="blut-smoke"
cleanup() { kind delete cluster --name "$CLUSTER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

for tool in kind kubectl cargo; do
  command -v "$tool" >/dev/null 2>&1 || { echo "::error::missing required tool: $tool"; exit 1; }
done

echo "==> building blut-operator"
# blut-operator is a STANDALONE workspace (it path-depends on the engine but is
# not a member of the root workspace), so it builds from its own directory.
(cd crates/blut-operator && cargo build --release --locked)
OPERATOR="crates/blut-operator/target/release/blut-operator"

echo "==> creating kind cluster '$CLUSTER'"
kind create cluster --name "$CLUSTER" --wait 120s

echo "==> installing operator CRDs"
"$OPERATOR" crds | kubectl apply -f -
# Wait for the API server to register both CRDs.
kubectl wait --for=condition=Established --timeout=60s \
  crd/blutplans.blut.lamquant.dev crd/blutworkerpools.blut.lamquant.dev

kubectl create namespace "$NS"

echo "==> applying a sample BlutPlan (schema must validate server-side)"
# Fields mirror crds.rs::BlutPlanSpec (camelCase): an inline PlanSource + the
# cookbook image are the required fields; dataClassCeiling defaults to Internal
# (Restricted is unrepresentable, ADR 0061).
cat <<'YAML' | kubectl apply -n "$NS" -f -
apiVersion: blut.lamquant.dev/v1alpha1
kind: BlutPlan
metadata:
  name: smoke-plan
spec:
  planSource:
    inline:
      content: '{"version":1,"nodes":[]}'
  cookbookImage: "ghcr.io/quitetall/blut-lamquant:smoke"
YAML

# The CR must exist and be readable back (accepted, not rejected by the schema).
kubectl get blutplan -n "$NS" smoke-plan -o name >/dev/null
echo "==> PASS: CRDs installed and a BlutPlan CR was accepted by the API server"

# Best-effort reconcile check: only if an operator image was loaded into the
# cluster (out of scope for the schema smoke; documented, never a silent skip).
if kubectl get deployment -n "$NS" blut-operator >/dev/null 2>&1; then
  echo "==> operator deployment present — waiting for it to reconcile a Job"
  if kubectl wait --for=condition=Complete --timeout=120s \
       -n "$NS" job -l blut.lamquant.dev/plan=smoke-plan 2>/dev/null; then
    echo "==> PASS: operator reconciled the BlutPlan into a Job"
  else
    echo "==> NOTE: no reconciled Job within the timeout (operator image/RBAC not fully wired in this smoke)"
  fi
else
  echo "==> NOTE: reconcile assertion SKIPPED — no in-cluster operator deployment"
  echo "         (this smoke gates the CRD schemas + CR admission; full reconcile"
  echo "          needs the operator image + RBAC loaded, a heavier CI lane)."
fi

echo "==> kind smoke complete"
