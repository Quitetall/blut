// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! BLUT Custom Resource Definitions (ADR 0067 T4.5).
//!
//! Two resources model "run BLUT on a cluster" while keeping Kubernetes an
//! adapter, not the control plane: the engine's executor, cache and admission
//! still decide what runs; Kubernetes schedules the pod and keeps it alive.
//!
//! - [`BlutPlan`] runs one declarative plan (a `.json` PlanSpec, a `.toml`
//!   chain, or a `.star` script) to completion as a Kubernetes `Job`. Its
//!   `dataClassCeiling` cannot be `Restricted`: the type has no such variant,
//!   so a manifest naming it is rejected by the API server (ADR 0061).
//! - [`BlutWorkerPool`] keeps a `StatefulSet` of `blut p2p node --no-scheduler`
//!   pods alive behind a headless Service. The mesh (ADR 0079) is the runtime.
//!
//! A plan's spec is immutable after creation. A plan is a record of one run;
//! editing it in place would leave the Job, its pods and the status describing
//! different things. Delete and recreate instead.

// Each `printcolumn(...)` repeats `type_ = ...`; clippy reads the repeated
// key inside one `#[kube(...)]` as a duplicated attribute. It is not one.
#![allow(clippy::duplicated_attributes)]

use kube::{CustomResource, KubeSchema};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The highest data sensitivity a cluster-run plan may touch. There is
/// deliberately NO `Restricted` variant: clinical/PHI data never enters a
/// cluster through a `BlutPlan` (ADR 0061). A manifest that names anything
/// else fails schema validation at the API server.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, JsonSchema)]
#[serde(rename_all = "PascalCase")]
pub enum DataClassCeiling {
    /// Openly shareable data.
    Public,
    /// Organization-internal data (the default ceiling).
    #[default]
    Internal,
}

impl DataClassCeiling {
    /// The value as it appears in a manifest.
    pub fn as_str(self) -> &'static str {
        match self {
            DataClassCeiling::Public => "Public",
            DataClassCeiling::Internal => "Internal",
        }
    }
}

/// Where a plan's recipe file comes from.
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum PlanSource {
    /// An existing ConfigMap in the plan's namespace, mounted whole at
    /// `/blut/plan`. `key` names the recipe; its extension selects the format.
    /// Other keys (small inputs the recipe reads) are mounted beside it.
    ConfigMapRef {
        /// ConfigMap name.
        name: String,
        /// Key holding the recipe; also its file name (`plan.json` by default).
        #[serde(default = "default_plan_file")]
        key: String,
    },
    /// The recipe inline. The operator copies it into a ConfigMap it owns.
    Inline {
        /// The recipe text.
        content: String,
        /// File name the engine sees; its extension (`.json`, `.toml`,
        /// `.star`) selects the format. `plan.json` by default.
        #[serde(default = "default_plan_file")]
        filename: String,
    },
}

/// The default recipe file name.
pub fn default_plan_file() -> String {
    "plan.json".to_string()
}

/// Where the engine's global cache lives so pods and reruns share it.
/// Externally tagged, and EVERY variant is an object (including `None {}`), so
/// each becomes a distinct object property in a `oneOf` — the shape Kubernetes
/// structural schemas accept.
#[derive(Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum CacheConfig {
    /// A PersistentVolumeClaim mounted as the engine's global cache. Needs
    /// `ReadWriteMany` for more than one pod at a time.
    Pvc {
        /// Claim name, in the resource's namespace.
        claim_name: String,
    },
    /// No shared cache: job-local only (default).
    None {},
}

impl Default for CacheConfig {
    fn default() -> Self {
        CacheConfig::None {}
    }
}

/// A content-addressed object store that stage outputs are written through to,
/// and that a later pod (a retry, or a rerun of the same plan) reads them back
/// from instead of recomputing.
#[derive(Serialize, Deserialize, Clone, Debug, KubeSchema)]
#[serde(rename_all = "camelCase")]
pub struct StoreConfig {
    /// `s3://bucket[/prefix]` (the cookbook image must be built with the
    /// engine's `s3` feature), or an absolute directory inside the container.
    /// Never put credentials here; use `credentialsSecret`.
    #[x_kube(validation = Rule::new("self.startsWith('s3://') || self.startsWith('/')").message("store.url must be s3://bucket[/prefix] or an absolute path"))]
    pub url: String,
    /// Name of a Secret in the plan's namespace holding the store credentials.
    /// The operator only names it in the pod spec; it never reads it. Keys
    /// passed to the plan container when present: `AWS_ACCESS_KEY_ID`,
    /// `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`,
    /// `AWS_ENDPOINT`, `AWS_ALLOW_HTTP`.
    #[serde(default)]
    pub credentials_secret: Option<String>,
}

/// Per-pod CPU and memory. Requests equal limits, so the pod's cgroup is the
/// memory containment (pods have no systemd user bus).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, KubeSchema)]
#[serde(rename_all = "camelCase")]
pub struct PlanResources {
    /// CPU, in millicores.
    #[x_kube(validation = Rule::new("self >= 1").message("cpuMillis must be at least 1"))]
    pub cpu_millis: u32,
    /// Memory, in MiB.
    #[x_kube(validation = Rule::new("self >= 16").message("memoryMib must be at least 16"))]
    pub memory_mib: u32,
}

/// A declarative BLUT plan to run to completion on the cluster.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, KubeSchema)]
#[kube(
    group = "blut.lamquant.dev",
    version = "v1alpha1",
    kind = "BlutPlan",
    namespaced,
    status = "BlutPlanStatus",
    shortname = "bp",
    printcolumn(name = "Phase", type_ = "string", json_path = ".status.phase"),
    printcolumn(name = "Attempts", type_ = "integer", json_path = ".status.attempts"),
    printcolumn(name = "Job", type_ = "string", json_path = ".status.jobName"),
    printcolumn(
        name = "Age",
        type_ = "date",
        json_path = ".metadata.creationTimestamp"
    )
)]
#[x_kube(validation = Rule::new("self == oldSelf").message("BlutPlan spec is immutable; delete the plan and create a new one"))]
#[serde(rename_all = "camelCase")]
pub struct BlutPlanSpec {
    /// Where the recipe comes from.
    pub plan_source: PlanSource,
    /// Container image carrying the cookbook binary and the engine.
    #[x_kube(validation = Rule::new("self.size() > 0").message("cookbookImage must not be empty"))]
    pub cookbook_image: String,
    /// The cookbook binary inside the image (default `blut`).
    #[serde(default)]
    pub command: Option<String>,
    /// The highest data class the plan may touch. Defaults to `Internal`;
    /// `Restricted` is unrepresentable (ADR 0061).
    #[serde(default)]
    pub data_class_ceiling: DataClassCeiling,
    /// Shared cache backing (PVC or job-local only).
    #[serde(default)]
    pub cache: CacheConfig,
    /// Object store that stage outputs are written through to.
    #[serde(default)]
    pub store: Option<StoreConfig>,
    /// Args JSON for a `.star` recipe's `build(args)`. Refused for `.json` and
    /// `.toml`, which carry their own args.
    #[serde(default)]
    pub args: Option<String>,
    /// CPU and memory for the plan pod. Without it the pod is unbounded and
    /// the node's capacity is the only limit.
    #[serde(default)]
    pub resources: Option<PlanResources>,
    /// GPUs to request (`nvidia.com/gpu`). Not validated on a GPU cluster.
    #[serde(default)]
    pub gpus: u32,
    /// Pod failures (an error exit, a deleted or evicted pod, a lost node)
    /// Kubernetes retries before the plan is `Failed`. Default 2.
    #[serde(default = "default_max_retries")]
    #[x_kube(validation = Rule::new("self <= 20").message("maxRetries must be at most 20"))]
    pub max_retries: u32,
    /// Wall-clock limit for the whole plan, across retries, in seconds.
    #[serde(default)]
    pub timeout_seconds: Option<u32>,
}

fn default_max_retries() -> u32 {
    2
}

/// Coarse lifecycle phase of a `BlutPlan`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, JsonSchema)]
#[serde(rename_all = "PascalCase")]
pub enum PlanPhase {
    /// Accepted; the Job has no running pod yet.
    #[default]
    Pending,
    /// A pod is running the plan.
    Running,
    /// The plan completed successfully.
    Succeeded,
    /// The plan failed, ran out of retries, or was rejected (see `message`).
    Failed,
}

impl PlanPhase {
    /// No further transitions happen from this phase.
    pub fn is_terminal(self) -> bool {
        matches!(self, PlanPhase::Succeeded | PlanPhase::Failed)
    }
}

/// Live status of a `BlutPlan`.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, KubeSchema)]
#[serde(rename_all = "camelCase")]
pub struct BlutPlanStatus {
    /// Coarse lifecycle phase.
    #[serde(default)]
    pub phase: PlanPhase,
    /// Last state of each stage, from the final pod's run summary. Empty until
    /// a pod has finished.
    #[serde(default)]
    pub stages: std::collections::BTreeMap<String, String>,
    /// Human-readable last message (a failure reason, or the run summary).
    #[serde(default)]
    pub message: Option<String>,
    /// The Job that runs the plan, once created.
    #[serde(default)]
    pub job_name: Option<String>,
    /// Pods started so far (1 + retries).
    #[serde(default)]
    pub attempts: u32,
    /// Stages served from the cache in the final pod's run.
    #[serde(default)]
    pub cache_hits: Option<u32>,
    /// Stages computed in the final pod's run.
    #[serde(default)]
    pub cache_misses: Option<u32>,
    /// The most recent pod of the plan's Job.
    #[serde(default)]
    pub pod: Option<String>,
    /// What happened to the most recent pod that failed, was deleted, or was
    /// evicted. Kept after the pod itself is gone.
    #[serde(default)]
    pub last_pod_failure: Option<String>,
    /// `metadata.generation` this status describes.
    #[serde(default)]
    pub observed_generation: Option<i64>,
}

/// A pool of worker nodes (`blut p2p node --no-scheduler`) kept alive by k8s.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, KubeSchema)]
#[kube(
    group = "blut.lamquant.dev",
    version = "v1alpha1",
    kind = "BlutWorkerPool",
    namespaced,
    status = "BlutWorkerPoolStatus",
    shortname = "bwp",
    printcolumn(name = "Replicas", type_ = "integer", json_path = ".spec.replicas"),
    printcolumn(name = "Ready", type_ = "integer", json_path = ".status.readyReplicas")
)]
#[serde(rename_all = "camelCase")]
pub struct BlutWorkerPoolSpec {
    /// Number of worker node pods.
    #[x_kube(validation = Rule::new("self <= 1000").message("replicas must be at most 1000"))]
    pub replicas: u32,
    /// The node image; it must be built with the engine's `p2p` feature.
    #[x_kube(validation = Rule::new("self.size() > 0").message("image must not be empty"))]
    pub image: String,
    /// The cookbook binary inside the image (default `blut`).
    #[serde(default)]
    pub command: Option<String>,
    /// The public key (hex) of the scheduler whose signed tasks the workers
    /// run (`--coordinator-pubkey`). Without it a worker runs only the mesh's
    /// connectivity smoke stage.
    #[serde(default)]
    #[x_kube(validation = Rule::new("self.matches('^[0-9a-f]+$')").message("coordinatorPubkey must be lowercase hex"))]
    pub coordinator_pubkey: Option<String>,
    /// Shared cache backing for the pool.
    #[serde(default)]
    pub cache: CacheConfig,
    /// Per-pod resource requests (CPU cores / memory GiB / GPUs).
    #[serde(default)]
    pub resources: PoolResources,
}

/// Per-pod resource requests for a worker pool. Every field defaults so a
/// partial `resources:` block (e.g. only `gpus`) still deserializes.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, KubeSchema)]
#[serde(rename_all = "camelCase")]
pub struct PoolResources {
    /// CPU cores per pod.
    #[serde(default = "default_cpu_cores")]
    pub cpu_cores: u32,
    /// Memory per pod, GiB.
    #[serde(default = "default_memory_gib")]
    pub memory_gib: u32,
    /// GPUs per pod (`nvidia.com/gpu`).
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
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, KubeSchema)]
#[serde(rename_all = "camelCase")]
pub struct BlutWorkerPoolStatus {
    /// Ready worker pods, mirrored from the StatefulSet.
    #[serde(default)]
    pub ready_replicas: u32,
    /// `metadata.generation` this status describes.
    #[serde(default)]
    pub observed_generation: Option<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kube::CustomResourceExt;

    #[test]
    fn restricted_ceiling_is_unrepresentable() {
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
        assert_eq!(spec.max_retries, 2);
        assert_eq!(spec.gpus, 0);
        match spec.plan_source {
            PlanSource::Inline { filename, .. } => assert_eq!(filename, "plan.json"),
            other => panic!("unexpected source {other:?}"),
        }
    }

    /// The manifest the engine's `LaunchTarget::K8s` renders
    /// (`src/config/k8s.rs`) must deserialize field for field. A field the
    /// engine emits and this spec lacks would be pruned by the API server
    /// without a word.
    #[test]
    fn the_engine_rendered_manifest_deserializes_without_loss() {
        let yaml = "planSource:\n  inline:\n    content: \"{\\\"nodes\\\":[]}\"\n\
                    cookbookImage: \"img:1\"\ndataClassCeiling: \"Public\"\n\
                    args: \"{}\"\ngpus: 2\n";
        let value: serde_json::Value = serde_yaml::from_str(yaml).unwrap();
        let spec: BlutPlanSpec = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(spec.gpus, 2);
        assert_eq!(spec.data_class_ceiling, DataClassCeiling::Public);
        let back = serde_json::to_value(&spec).unwrap();
        for key in value.as_object().unwrap().keys() {
            assert!(back.get(key).is_some(), "field {key} would be pruned");
        }
    }

    fn rules(v: &serde_json::Value, out: &mut Vec<String>) {
        match v {
            serde_json::Value::Object(m) => {
                if let Some(serde_json::Value::Array(rs)) = m.get("x-kubernetes-validations") {
                    for r in rs {
                        out.push(r["rule"].as_str().unwrap_or_default().to_string());
                    }
                }
                m.values().for_each(|x| rules(x, out));
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| rules(x, out)),
            _ => {}
        }
    }

    #[test]
    fn plan_crd_carries_the_immutability_and_field_rules() {
        let crd = serde_json::to_value(BlutPlan::crd()).unwrap();
        let mut found = Vec::new();
        rules(&crd, &mut found);
        for want in [
            "self == oldSelf",
            "self.size() > 0",
            "self <= 20",
            "self.startsWith('s3://') || self.startsWith('/')",
        ] {
            assert!(
                found.iter().any(|r| r == want),
                "missing rule {want}: {found:?}"
            );
        }
        let spec = &crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]["spec"];
        let ceiling = &spec["properties"]["dataClassCeiling"]["enum"];
        assert_eq!(ceiling, &serde_json::json!(["Public", "Internal"]));
    }

    #[test]
    fn crds_generate_valid_schemas() {
        let plan_crd = BlutPlan::crd();
        assert_eq!(plan_crd.spec.names.kind, "BlutPlan");
        let pool_crd = BlutWorkerPool::crd();
        assert_eq!(pool_crd.spec.names.kind, "BlutWorkerPool");
        assert!(
            plan_crd.spec.versions[0]
                .subresources
                .as_ref()
                .and_then(|s| s.status.as_ref())
                .is_some(),
            "status is a subresource, so users cannot write it"
        );
    }
}
