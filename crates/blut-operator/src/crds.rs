// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT Custom Resource Definitions (ADR 0067 T4.5, ADR 0037).
//!
//! Two resources model "run BLUT on a cluster" while keeping k8s an ADAPTER,
//! not the control plane (ADR 0037):
//!
//! - [`BlutWorkerPool`] → a StatefulSet of `blut p2p node --no-scheduler` pods.
//!   The mesh (ADR 0079) is the runtime; k8s just keeps the pods alive. Node-0's
//!   public key is published in the pool status so an out-of-cluster scheduler
//!   can `--seed` to it.
//! - [`BlutPlan`] → run one declarative plan (a Starlark/JSON PlanSpec) to
//!   completion. Its `dataClassCeiling` CANNOT be `Restricted` — the type
//!   simply has no such variant, so clinical/PHI data can never enter a
//!   cluster through this CRD (ADR 0061 clinical hard-block, enforced at the
//!   schema level, fail-closed).

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The highest data sensitivity a cluster-run plan may touch. There is
/// deliberately NO `Restricted` variant: clinical/PHI data never enters a
/// cluster through a `BlutPlan` (ADR 0061). A manifest that names anything
/// else fails to deserialize — fail-closed at the API boundary.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, JsonSchema)]
#[serde(rename_all = "PascalCase")]
pub enum DataClassCeiling {
    /// Openly shareable data.
    Public,
    /// Organization-internal data (the default ceiling).
    #[default]
    Internal,
}

/// Where a plan's PlanSpec comes from.
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum PlanSource {
    /// A ConfigMap key holding the PlanSpec JSON (or a `.star` script).
    ConfigMapRef {
        name: String,
        #[serde(default = "default_plan_key")]
        key: String,
    },
    /// The PlanSpec / Starlark inline. Content-hashable for provenance.
    Inline { content: String },
}

fn default_plan_key() -> String {
    "plan.json".to_string()
}

/// Where the content-addressed cache lives so pods and reruns share it.
/// Externally tagged, and EVERY variant is an object (including `None {}`), so
/// each becomes a distinct object property in a `oneOf` — the shape Kubernetes
/// structural schemas accept. (A bare unit variant would serialize to a string
/// and a per-variant internal discriminator confuses the CRD flattener; both
/// were tried and rejected by kube.)
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CacheConfig {
    /// A ReadWriteMany PVC mounted at the global-cache path (FsBlobStore).
    Pvc { claim_name: String },
    /// An S3-compatible bucket (S3BlobStore via the engine's `s3` feature).
    S3 {
        bucket: String,
        #[serde(default)]
        prefix: String,
    },
    /// No shared cache — job-local only (default).
    None {},
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig::None {}
    }
}

/// A declarative BLUT plan to run to completion on the cluster.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "blut.lamquant.dev",
    version = "v1alpha1",
    kind = "BlutPlan",
    namespaced,
    status = "BlutPlanStatus",
    shortname = "bp",
    printcolumn = r#"{"name":"Phase","type":"string","jsonPath":".status.phase"}"#,
    printcolumn = r#"{"name":"Ceiling","type":"string","jsonPath":".spec.dataClassCeiling"}"#
)]
#[serde(rename_all = "camelCase")]
pub struct BlutPlanSpec {
    /// Where the plan comes from.
    pub plan_source: PlanSource,
    /// Container image carrying the cookbook + engine to run the plan.
    pub cookbook_image: String,
    /// The highest data class the plan may touch. Defaults to `Internal`;
    /// `Restricted` is unrepresentable (ADR 0061).
    #[serde(default)]
    pub data_class_ceiling: DataClassCeiling,
    /// Shared cache backing (PVC / S3 / none).
    #[serde(default)]
    pub cache: CacheConfig,
    /// Optional args JSON handed to the plan/recipe.
    #[serde(default)]
    pub args: Option<String>,
}

/// Coarse lifecycle phase of a `BlutPlan`. An enum (not a bare string) so the
/// contract is compile-time enforced; changing the set post-ship is a
/// CRD-schema break, so it's pinned now.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, JsonSchema)]
#[serde(rename_all = "PascalCase")]
pub enum PlanPhase {
    /// Accepted, dispatcher Job not yet created.
    #[default]
    Pending,
    /// The dispatcher Job is running the plan.
    Running,
    /// The plan completed successfully.
    Succeeded,
    /// The plan failed (see `message`).
    Failed,
}

/// Live status of a `BlutPlan`, mirrored from the run's `status.jsonl`.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct BlutPlanStatus {
    /// Coarse lifecycle phase.
    #[serde(default)]
    pub phase: PlanPhase,
    /// Per-stage states (name → state string), mirrored from status.jsonl.
    #[serde(default)]
    pub stages: std::collections::BTreeMap<String, String>,
    /// Human-readable last message (e.g. a failure summary).
    #[serde(default)]
    pub message: Option<String>,
    /// The dispatcher Job that runs the plan, once created.
    #[serde(default)]
    pub job_name: Option<String>,
}

/// A pool of worker nodes (`blut p2p node --no-scheduler`) kept alive by k8s.
/// The mesh is the runtime; the operator just maintains the StatefulSet.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "blut.lamquant.dev",
    version = "v1alpha1",
    kind = "BlutWorkerPool",
    namespaced,
    status = "BlutWorkerPoolStatus",
    shortname = "bwp",
    printcolumn = r#"{"name":"Replicas","type":"integer","jsonPath":".spec.replicas"}"#,
    printcolumn = r#"{"name":"Ready","type":"integer","jsonPath":".status.readyReplicas"}"#
)]
#[serde(rename_all = "camelCase")]
pub struct BlutWorkerPoolSpec {
    /// Number of worker node pods.
    pub replicas: u32,
    /// The blut node image.
    pub image: String,
    /// Shared cache backing for the pool.
    #[serde(default)]
    pub cache: CacheConfig,
    /// Per-pod resource requests (CPU cores / memory GiB / GPUs).
    #[serde(default)]
    pub resources: PoolResources,
}

/// Per-pod resource requests for a worker pool. Every field defaults so a
/// partial `resources:` block (e.g. only `gpus`) still deserializes.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PoolResources {
    #[serde(default = "default_cpu_cores")]
    pub cpu_cores: u32,
    #[serde(default = "default_memory_gib")]
    pub memory_gib: u32,
    #[serde(default)]
    pub gpus: u32,
}

fn default_cpu_cores() -> u32 {
    2
}
fn default_memory_gib() -> u32 {
    8
}

impl Default for PoolResources {
    fn default() -> Self {
        Self {
            cpu_cores: default_cpu_cores(),
            memory_gib: default_memory_gib(),
            gpus: 0,
        }
    }
}

/// Status of a worker pool.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct BlutWorkerPoolStatus {
    /// Ready worker pods.
    #[serde(default)]
    pub ready_replicas: u32,
    /// Node-0's public key (hex) — the `--seed` an out-of-cluster scheduler
    /// dials to reach the pool over the mesh.
    #[serde(default)]
    pub seed_pubkey: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restricted_ceiling_is_unrepresentable() {
        // The clinical hard-block is enforced at the schema: no manifest can
        // name Restricted, because the enum has no such variant.
        let err = serde_json::from_str::<DataClassCeiling>("\"Restricted\"");
        assert!(err.is_err(), "Restricted must not deserialize");
        assert_eq!(
            serde_json::from_str::<DataClassCeiling>("\"Internal\"").unwrap(),
            DataClassCeiling::Internal
        );
        assert_eq!(
            serde_json::from_str::<DataClassCeiling>("\"Public\"").unwrap(),
            DataClassCeiling::Public
        );
    }

    #[test]
    fn ceiling_defaults_to_internal() {
        assert_eq!(DataClassCeiling::default(), DataClassCeiling::Internal);
    }

    #[test]
    fn blutplan_spec_round_trips_with_defaults() {
        let json = serde_json::json!({
            "planSource": { "inline": { "content": "{}" } },
            "cookbookImage": "ghcr.io/lamquant/blut:latest"
        });
        let spec: BlutPlanSpec = serde_json::from_value(json).unwrap();
        assert_eq!(spec.data_class_ceiling, DataClassCeiling::Internal);
        assert!(matches!(spec.cache, CacheConfig::None {}));
        assert_eq!(spec.cookbook_image, "ghcr.io/lamquant/blut:latest");
    }

    #[test]
    fn crds_generate_valid_schemas() {
        // The derive must produce installable CRDs (schema generation doesn't
        // panic and yields the expected kinds).
        use kube::CustomResourceExt;
        let plan_crd = BlutPlan::crd();
        assert_eq!(plan_crd.spec.names.kind, "BlutPlan");
        let pool_crd = BlutWorkerPool::crd();
        assert_eq!(pool_crd.spec.names.kind, "BlutWorkerPool");
    }
}
