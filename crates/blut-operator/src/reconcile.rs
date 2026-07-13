// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Reconcilers: desired-state builders + apply loops for the BLUT CRDs.
//!
//! The pure BUILDERS ([`build_plan_job`], [`build_pool_statefulset`],
//! [`build_pool_service`]) turn a CR into the Kubernetes objects it implies —
//! testable without a cluster. The reconcile actions server-side-apply them and
//! own them via owner references (so deleting the CR garbage-collects its
//! children). K8s only keeps things alive; the mesh + engine broker stay
//! authoritative (ADR 0037).

use std::collections::BTreeMap;

use k8s_openapi::api::apps::v1::{StatefulSet, StatefulSetSpec};
use k8s_openapi::api::batch::v1::{Job, JobSpec};
use k8s_openapi::api::core::v1::{
    Container, ContainerPort, EnvVar, PersistentVolumeClaimVolumeSource, PodSpec, PodTemplateSpec,
    ResourceRequirements, Service, ServicePort, ServiceSpec, Volume, VolumeMount,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{LabelSelector, ObjectMeta};
use kube::{Resource, ResourceExt};

use crate::crds::{BlutPlan, BlutWorkerPool, CacheConfig, DataClassCeiling, PlanSource};

/// Common labels stamped on every owned object.
pub fn labels(instance: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "app.kubernetes.io/managed-by".to_string(),
            "blut-operator".to_string(),
        ),
        (
            "app.kubernetes.io/instance".to_string(),
            instance.to_string(),
        ),
    ])
}

fn ceiling_str(c: DataClassCeiling) -> &'static str {
    match c {
        DataClassCeiling::Public => "Public",
        DataClassCeiling::Internal => "Internal",
    }
}

/// The mount path a PVC-backed cache is exposed at inside the container.
const CACHE_MOUNT_PATH: &str = "/blut-cache";
const CACHE_VOLUME_NAME: &str = "blut-cache";

/// Env + volume + mount wiring for a cache backing, shared by plan Jobs and
/// pool pods. A PVC backing emits BOTH the env pointing the engine's cache at
/// the mount AND the Volume/VolumeMount that actually mounts it (without the
/// latter the container reads an empty ephemeral dir — a silent data-loss trap).
#[derive(Default)]
struct CacheWiring {
    env: Vec<EnvVar>,
    volumes: Vec<Volume>,
    mounts: Vec<VolumeMount>,
}

fn cache_wiring(cache: &CacheConfig) -> CacheWiring {
    match cache {
        CacheConfig::Pvc { claim_name } => CacheWiring {
            env: vec![EnvVar {
                name: "LAMU_TRAIN_CACHE_DIR".into(),
                value: Some(CACHE_MOUNT_PATH.into()),
                ..Default::default()
            }],
            volumes: vec![Volume {
                name: CACHE_VOLUME_NAME.into(),
                persistent_volume_claim: Some(PersistentVolumeClaimVolumeSource {
                    claim_name: claim_name.clone(),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            mounts: vec![VolumeMount {
                name: CACHE_VOLUME_NAME.into(),
                mount_path: CACHE_MOUNT_PATH.into(),
                ..Default::default()
            }],
        },
        CacheConfig::None {} => CacheWiring::default(),
    }
}

/// `Some(v)` if `v` is non-empty, else `None` — for optional k8s list fields.
fn non_empty<T>(v: Vec<T>) -> Option<Vec<T>> {
    if v.is_empty() { None } else { Some(v) }
}

fn resource_requirements(cpu: u32, mem_gib: u32, gpus: u32) -> ResourceRequirements {
    let mut requests = BTreeMap::new();
    requests.insert("cpu".to_string(), Quantity(cpu.to_string()));
    requests.insert("memory".to_string(), Quantity(format!("{mem_gib}Gi")));
    if gpus > 0 {
        requests.insert("nvidia.com/gpu".to_string(), Quantity(gpus.to_string()));
    }
    ResourceRequirements {
        // Requests == limits for the GPU (k8s requires it) and a stable QoS.
        limits: Some(requests.clone()),
        requests: Some(requests),
        ..Default::default()
    }
}

/// Build the dispatcher `Job` that runs a `BlutPlan` to completion.
///
/// Defense in depth on the clinical block: `DataClassCeiling` already can't be
/// `Restricted` by type, and the ceiling is passed to the engine so any stage
/// classified above it fail-closes at dispatch (ADR 0061).
pub fn build_plan_job(plan: &BlutPlan) -> Job {
    let instance = plan.name_any();
    let job_name = format!("{instance}-run");

    // How the plan reaches the container.
    let (plan_arg, plan_env) = match &plan.spec.plan_source {
        PlanSource::Inline { content } => (
            "/dev/stdin".to_string(),
            vec![EnvVar {
                name: "BLUT_PLAN_INLINE".into(),
                value: Some(content.clone()),
                ..Default::default()
            }],
        ),
        PlanSource::ConfigMapRef { name, key } => (
            format!("/plan/{key}"),
            vec![EnvVar {
                name: "BLUT_PLAN_CONFIGMAP".into(),
                value: Some(name.clone()),
                ..Default::default()
            }],
        ),
    };

    let cache = cache_wiring(&plan.spec.cache);
    let mut env = vec![EnvVar {
        name: "BLUT_DATA_CLASS_CEILING".into(),
        value: Some(ceiling_str(plan.spec.data_class_ceiling).into()),
        ..Default::default()
    }];
    env.extend(plan_env);
    env.extend(cache.env);
    if let Some(args) = &plan.spec.args {
        env.push(EnvVar {
            name: "BLUT_PLAN_ARGS".into(),
            value: Some(args.clone()),
            ..Default::default()
        });
    }

    let container = Container {
        name: "blut-plan".into(),
        image: Some(plan.spec.cookbook_image.clone()),
        command: Some(vec!["blut".into()]),
        args: Some(vec![
            "recipe".into(),
            "declare".into(),
            plan_arg,
            "--run".into(),
        ]),
        env: Some(env),
        volume_mounts: non_empty(cache.mounts),
        ..Default::default()
    };

    let pod_spec = PodSpec {
        containers: vec![container],
        restart_policy: Some("Never".into()),
        volumes: non_empty(cache.volumes),
        ..Default::default()
    };

    Job {
        metadata: ObjectMeta {
            name: Some(job_name),
            namespace: plan.namespace(),
            labels: Some(labels(&instance)),
            owner_references: plan.controller_owner_ref(&()).map(|r| vec![r]),
            ..Default::default()
        },
        spec: Some(JobSpec {
            backoff_limit: Some(2),
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels(&instance)),
                    ..Default::default()
                }),
                spec: Some(pod_spec),
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Build the headless `Service` fronting a worker pool's StatefulSet (stable
/// per-pod DNS so a scheduler can reach node-0 as the mesh `--seed`).
pub fn build_pool_service(pool: &BlutWorkerPool) -> Service {
    let instance = pool.name_any();
    Service {
        metadata: ObjectMeta {
            name: Some(format!("{instance}-workers")),
            namespace: pool.namespace(),
            labels: Some(labels(&instance)),
            owner_references: pool.controller_owner_ref(&()).map(|r| vec![r]),
            ..Default::default()
        },
        spec: Some(ServiceSpec {
            cluster_ip: Some("None".into()), // headless
            selector: Some(labels(&instance)),
            ports: Some(vec![ServicePort {
                name: Some("mesh".into()),
                port: 9320,
                ..Default::default()
            }]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Build the `StatefulSet` of worker node pods for a pool. Pods run
/// `blut p2p node --no-scheduler` — the mesh is the runtime; k8s keeps them
/// alive (ADR 0037).
pub fn build_pool_statefulset(pool: &BlutWorkerPool) -> StatefulSet {
    let instance = pool.name_any();
    let res = pool.spec.resources;

    let cache = cache_wiring(&pool.spec.cache);
    let mut env = vec![EnvVar {
        name: "BLUT_NODE_ROLE".into(),
        value: Some("worker".into()),
        ..Default::default()
    }];
    env.extend(cache.env);

    let container = Container {
        name: "blut-node".into(),
        image: Some(pool.spec.image.clone()),
        command: Some(vec!["blut".into()]),
        args: Some(vec![
            "p2p".into(),
            "node".into(),
            "--no-scheduler".into(),
            "--listen".into(),
            "0.0.0.0:9320".into(),
        ]),
        env: Some(env),
        ports: Some(vec![ContainerPort {
            name: Some("mesh".into()),
            container_port: 9320,
            ..Default::default()
        }]),
        resources: Some(resource_requirements(
            res.cpu_cores,
            res.memory_gib,
            res.gpus,
        )),
        volume_mounts: non_empty(cache.mounts),
        ..Default::default()
    };

    let pod_spec = PodSpec {
        containers: vec![container],
        volumes: non_empty(cache.volumes),
        ..Default::default()
    };

    StatefulSet {
        metadata: ObjectMeta {
            name: Some(format!("{instance}-workers")),
            namespace: pool.namespace(),
            labels: Some(labels(&instance)),
            owner_references: pool.controller_owner_ref(&()).map(|r| vec![r]),
            ..Default::default()
        },
        spec: Some(StatefulSetSpec {
            replicas: Some(i32::try_from(pool.spec.replicas).unwrap_or(i32::MAX)),
            service_name: format!("{instance}-workers"),
            selector: LabelSelector {
                match_labels: Some(labels(&instance)),
                ..Default::default()
            },
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels(&instance)),
                    ..Default::default()
                }),
                spec: Some(pod_spec),
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

// ── Live controller glue (needs a cluster; compiled + gated, run infra-side) ──

use std::sync::Arc;
use std::time::Duration;

use kube::api::{Patch, PatchParams};
use kube::runtime::controller::Action;
use kube::{Api, Client};

/// Shared reconcile context.
pub struct Context {
    pub client: Client,
}

/// Field manager for server-side apply — makes the operator own the fields it
/// sets, so applies are idempotent and don't fight other managers.
const FIELD_MANAGER: &str = "blut-operator";

/// On a successful apply, wait for the next watch event rather than polling —
/// the Controller already re-reconciles when the CR or its owned children
/// change, so a fixed requeue would just add idle API load.
fn settled() -> Action {
    Action::await_change()
}

/// Reconcile a `BlutPlan`: server-side-apply its dispatcher Job (owner-ref'd,
/// so deleting the plan GCs the Job). Status mirroring is the B2 sidecar's job.
pub async fn reconcile_plan(plan: Arc<BlutPlan>, ctx: Arc<Context>) -> Result<Action, kube::Error> {
    let ns = plan.namespace().unwrap_or_else(|| "default".into());
    let jobs: Api<Job> = Api::namespaced(ctx.client.clone(), &ns);
    let job = build_plan_job(&plan);
    let name = job.metadata.name.clone().unwrap_or_default();
    jobs.patch(
        &name,
        &PatchParams::apply(FIELD_MANAGER).force(),
        &Patch::Apply(&job),
    )
    .await?;
    Ok(settled())
}

/// Reconcile a `BlutWorkerPool`: apply its headless Service + StatefulSet.
pub async fn reconcile_pool(
    pool: Arc<BlutWorkerPool>,
    ctx: Arc<Context>,
) -> Result<Action, kube::Error> {
    let ns = pool.namespace().unwrap_or_else(|| "default".into());
    let svcs: Api<Service> = Api::namespaced(ctx.client.clone(), &ns);
    let sets: Api<StatefulSet> = Api::namespaced(ctx.client.clone(), &ns);
    let pp = PatchParams::apply(FIELD_MANAGER).force();

    let svc = build_pool_service(&pool);
    svcs.patch(
        &svc.metadata.name.clone().unwrap_or_default(),
        &pp,
        &Patch::Apply(&svc),
    )
    .await?;

    let sts = build_pool_statefulset(&pool);
    sets.patch(
        &sts.metadata.name.clone().unwrap_or_default(),
        &pp,
        &Patch::Apply(&sts),
    )
    .await?;
    Ok(settled())
}

/// Error policy: retry after a short back-off on any reconcile error.
pub fn error_policy<K>(_obj: Arc<K>, err: &kube::Error, _ctx: Arc<Context>) -> Action {
    tracing::warn!("reconcile error: {err}");
    Action::requeue(Duration::from_secs(10))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crds::{BlutPlanSpec, BlutWorkerPoolSpec, PoolResources};

    fn plan(source: PlanSource, ceiling: DataClassCeiling, cache: CacheConfig) -> BlutPlan {
        let mut p = BlutPlan::new(
            "demo",
            BlutPlanSpec {
                plan_source: source,
                cookbook_image: "ghcr.io/lamquant/blut:latest".into(),
                data_class_ceiling: ceiling,
                cache,
                args: None,
            },
        );
        p.metadata.namespace = Some("blut".into());
        p.metadata.uid = Some("uid-123".into());
        p
    }

    #[test]
    fn plan_job_names_owns_and_sets_ceiling() {
        let p = plan(
            PlanSource::Inline {
                content: "{}".into(),
            },
            DataClassCeiling::Internal,
            CacheConfig::None {},
        );
        let job = build_plan_job(&p);
        assert_eq!(job.metadata.name.as_deref(), Some("demo-run"));
        assert_eq!(job.metadata.namespace.as_deref(), Some("blut"));
        // Owner reference for GC.
        let owners = job.metadata.owner_references.unwrap();
        assert_eq!(owners[0].kind, "BlutPlan");
        assert_eq!(owners[0].uid, "uid-123");
        // Ceiling passed to the engine + restart-never.
        let spec = job.spec.unwrap();
        let pod = spec.template.spec.unwrap();
        assert_eq!(pod.restart_policy.as_deref(), Some("Never"));
        let env = pod.containers[0].env.as_ref().unwrap();
        assert!(
            env.iter()
                .any(|e| e.name == "BLUT_DATA_CLASS_CEILING"
                    && e.value.as_deref() == Some("Internal"))
        );
    }

    fn pool(replicas: u32, res: PoolResources) -> BlutWorkerPool {
        let mut p = BlutWorkerPool::new(
            "gpu",
            BlutWorkerPoolSpec {
                replicas,
                image: "ghcr.io/lamquant/blut:latest".into(),
                cache: CacheConfig::None {},
                resources: res,
            },
        );
        p.metadata.namespace = Some("blut".into());
        p.metadata.uid = Some("uid-pool".into());
        p
    }

    #[test]
    fn pool_statefulset_is_no_scheduler_worker() {
        let p = pool(
            3,
            PoolResources {
                cpu_cores: 4,
                memory_gib: 16,
                gpus: 2,
            },
        );
        let sts = build_pool_statefulset(&p);
        let spec = sts.spec.unwrap();
        assert_eq!(spec.replicas, Some(3));
        assert_eq!(spec.service_name, "gpu-workers");
        let pod = spec.template.spec.unwrap();
        let args = pod.containers[0].args.as_ref().unwrap();
        assert!(
            args.contains(&"--no-scheduler".to_string()),
            "worker pool runs --no-scheduler"
        );
        // GPU request wired.
        let req = pod.containers[0].resources.as_ref().unwrap();
        let limits = req.limits.as_ref().unwrap();
        assert_eq!(limits.get("nvidia.com/gpu").unwrap().0, "2");
        assert_eq!(limits.get("memory").unwrap().0, "16Gi");
    }

    #[test]
    fn pvc_cache_mounts_the_volume_not_just_env() {
        // Regression: PVC mode used to set the cache env but never mount the
        // volume, so the container read an empty ephemeral dir.
        let p = plan(
            PlanSource::Inline {
                content: "{}".into(),
            },
            DataClassCeiling::Internal,
            CacheConfig::Pvc {
                claim_name: "cache-pvc".into(),
            },
        );
        let job = build_plan_job(&p);
        let pod = job.spec.unwrap().template.spec.unwrap();
        // Volume references the PVC.
        let vol = &pod.volumes.as_ref().unwrap()[0];
        assert_eq!(
            vol.persistent_volume_claim.as_ref().unwrap().claim_name,
            "cache-pvc"
        );
        // Container mounts it at the path the env points to.
        let mount = &pod.containers[0].volume_mounts.as_ref().unwrap()[0];
        assert_eq!(mount.mount_path, "/blut-cache");
        assert_eq!(mount.name, vol.name);
        let env = pod.containers[0].env.as_ref().unwrap();
        assert!(
            env.iter()
                .any(|e| e.name == "LAMU_TRAIN_CACHE_DIR"
                    && e.value.as_deref() == Some("/blut-cache"))
        );
    }

    #[test]
    fn pool_service_is_headless() {
        let p = pool(1, PoolResources::default());
        let svc = build_pool_service(&p);
        let spec = svc.spec.unwrap();
        assert_eq!(spec.cluster_ip.as_deref(), Some("None"));
        assert_eq!(svc.metadata.name.as_deref(), Some("gpu-workers"));
    }

    #[test]
    fn no_gpu_omits_gpu_request() {
        let p = pool(
            1,
            PoolResources {
                cpu_cores: 1,
                memory_gib: 2,
                gpus: 0,
            },
        );
        let sts = build_pool_statefulset(&p);
        let pod = sts.spec.unwrap().template.spec.unwrap();
        let limits = pod.containers[0]
            .resources
            .as_ref()
            .unwrap()
            .limits
            .as_ref()
            .unwrap();
        assert!(!limits.contains_key("nvidia.com/gpu"));
    }
}
