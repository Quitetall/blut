// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Reconcilers for the BLUT CRDs.
//!
//! Everything that decides WHAT the cluster should look like is a pure
//! function of the custom resource and the objects observed for it, and is
//! unit-tested without a cluster:
//!
//! - [`validate_plan`] — semantic checks the CRD schema cannot express;
//! - [`build_plan_configmap`], [`build_plan_job`] — the objects a plan implies;
//! - [`plan_status`] — the status a plan should show, given its Job and pods;
//! - [`plan_action`] — whether to create the Job, report, or stand still;
//! - [`build_pool_service`], [`build_pool_statefulset`] — a pool's objects.
//!
//! The async glue below them only reads, creates and patches. Children carry a
//! controller owner reference, so deleting a `BlutPlan` or `BlutWorkerPool`
//! garbage-collects everything it created; the operator holds no delete
//! permission at all.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use k8s_openapi::api::apps::v1::{StatefulSet, StatefulSetSpec};
use k8s_openapi::api::batch::v1::{
    Job, JobSpec, PodFailurePolicy, PodFailurePolicyOnPodConditionsPattern, PodFailurePolicyRule,
};
use k8s_openapi::api::core::v1::{
    Capabilities, ConfigMap, ConfigMapVolumeSource, Container, ContainerPort, EmptyDirVolumeSource,
    EnvVar, EnvVarSource, KeyToPath, PersistentVolumeClaimVolumeSource, Pod, PodSecurityContext,
    PodSpec, PodTemplateSpec, ResourceRequirements, SeccompProfile, SecretKeySelector,
    SecurityContext, Service, ServicePort, ServiceSpec, Volume, VolumeMount,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{LabelSelector, ObjectMeta};
use kube::api::{ListParams, Patch, PatchParams, PostParams};
use kube::runtime::controller::Action;
use kube::{Api, Client, Resource, ResourceExt};

use crate::crds::{
    BlutPlan, BlutPlanSpec, BlutPlanStatus, BlutWorkerPool, BlutWorkerPoolStatus, CacheConfig,
    PlanPhase, PlanSource,
};

/// Label naming the `BlutPlan` an object belongs to.
pub const PLAN_LABEL: &str = "blut.lamquant.dev/plan";
/// Label naming the `BlutWorkerPool` an object belongs to.
pub const POOL_LABEL: &str = "blut.lamquant.dev/pool";
/// Label carrying a plan's data-class ceiling, for admission policies.
pub const DATA_CLASS_LABEL: &str = "blut.lamquant.dev/data-class";

/// Where the recipe file is mounted in the plan container.
pub const PLAN_DIR: &str = "/blut/plan";
/// Writable scratch for the engine (HOME, job directories, temp files).
pub const WORK_DIR: &str = "/blut/work";
/// The mount path of a PVC-backed cache.
pub const CACHE_MOUNT_PATH: &str = "/blut-cache";
/// The file the kubelet reads as the termination message; the engine writes
/// its run summary there (`BLUT_RUN_SUMMARY`).
pub const TERMINATION_MESSAGE_PATH: &str = "/dev/termination-log";
/// The cookbook binary used when a spec names none.
pub const DEFAULT_COMMAND: &str = "blut";

/// Credential keys copied from `store.credentialsSecret` into the plan
/// container's environment when present. The operator names them in the pod
/// spec; the kubelet resolves them. The operator never reads a Secret.
pub const STORE_CREDENTIAL_KEYS: &[&str] = &[
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_ENDPOINT",
    "AWS_ALLOW_HTTP",
];

const PLAN_VOLUME: &str = "plan";
const WORK_VOLUME: &str = "work";
const TMP_VOLUME: &str = "tmp";
const CACHE_VOLUME: &str = "blut-cache";
const MESH_PORT: i32 = 9320;

/// Longest plan name whose Job name (`<name>-run`) still fits the 63-character
/// label value Kubernetes stamps on every Job pod.
pub const MAX_PLAN_NAME: usize = 63 - "-run".len();

/// Labels stamped on every object owned by a plan.
pub fn plan_labels(plan: &BlutPlan) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "app.kubernetes.io/managed-by".to_string(),
            "blut-operator".to_string(),
        ),
        (PLAN_LABEL.to_string(), plan.name_any()),
    ])
}

fn pool_labels(instance: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "app.kubernetes.io/managed-by".to_string(),
            "blut-operator".to_string(),
        ),
        (POOL_LABEL.to_string(), instance.to_string()),
    ])
}

/// The Job a plan runs as.
pub fn plan_job_name(plan: &BlutPlan) -> String {
    format!("{}-run", plan.name_any())
}

/// The ConfigMap holding an inline plan's recipe.
pub fn plan_configmap_name(plan: &BlutPlan) -> String {
    format!("{}-plan", plan.name_any())
}

/// The recipe's file name inside the container (its extension picks the
/// format).
pub fn plan_file(spec: &BlutPlanSpec) -> &str {
    match &spec.plan_source {
        PlanSource::Inline { filename, .. } => filename,
        PlanSource::ConfigMapRef { key, .. } => key,
    }
}

fn is_dns_label(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 63
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !s.starts_with('-')
        && !s.ends_with('-')
}

fn is_dns_subdomain(s: &str) -> bool {
    !s.is_empty() && s.len() <= 253 && s.split('.').all(is_dns_label)
}

/// Checks the CRD schema cannot express. Each error names the field and says
/// what would be accepted. A plan that fails here never gets a Job; its status
/// becomes `Failed` with the message.
pub fn validate_plan(plan: &BlutPlan) -> Result<(), String> {
    let name = plan.name_any();
    if name.len() > MAX_PLAN_NAME {
        return Err(format!(
            "metadata.name is {} characters; at most {MAX_PLAN_NAME} are allowed so the \
             Job name '<name>-run' fits a Kubernetes label",
            name.len()
        ));
    }
    let spec = &plan.spec;
    let file = plan_file(spec);
    if file.is_empty()
        || file.contains('/')
        || file.starts_with('.')
        || file.len() > 253
        || !file
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return Err(format!(
            "recipe file name '{file}' must be a plain file name of letters, digits, \
             '.', '_' or '-' (no directories)"
        ));
    }
    let extension = file.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    if !matches!(extension, "json" | "toml" | "star") {
        return Err(format!(
            "recipe file '{file}' must end in .json (PlanSpec), .toml (chain) or .star \
             (Starlark); the extension selects the format"
        ));
    }
    match &spec.plan_source {
        PlanSource::Inline { content, .. } if content.trim().is_empty() => {
            return Err("planSource.inline.content is empty".into());
        }
        PlanSource::Inline { content, .. } if content.len() > 900 * 1024 => {
            return Err(format!(
                "planSource.inline.content is {} bytes; a ConfigMap holds at most ~1 MiB. \
                 Put the recipe in your own ConfigMap and use planSource.configMapRef",
                content.len()
            ));
        }
        PlanSource::ConfigMapRef { name, .. } if !is_dns_subdomain(name) => {
            return Err(format!(
                "planSource.configMapRef.name '{name}' is not a valid ConfigMap name"
            ));
        }
        _ => {}
    }
    if let Some(args) = &spec.args {
        if extension != "star" {
            return Err(format!(
                "args is only for .star recipes; '{file}' carries its own args"
            ));
        }
        if serde_json::from_str::<serde_json::Value>(args).is_err() {
            return Err("args is not valid JSON".into());
        }
    }
    if let Some(command) = &spec.command
        && (command.is_empty() || command.chars().any(char::is_whitespace))
    {
        return Err(format!(
            "command '{command}' must be one executable name or path, without arguments"
        ));
    }
    if let CacheConfig::Pvc { claim_name } = &spec.cache
        && !is_dns_subdomain(claim_name)
    {
        return Err(format!(
            "cache.pvc.claimName '{claim_name}' is not a valid claim name"
        ));
    }
    if let Some(store) = &spec.store {
        let url = store.url.as_str();
        if let Some(rest) = url.strip_prefix("s3://") {
            let bucket = rest.split('/').next().unwrap_or("");
            if bucket.is_empty() || bucket.contains('@') || bucket.contains(':') {
                return Err(
                    "store.url must be s3://bucket[/prefix]; credentials and the endpoint \
                     belong in store.credentialsSecret, never in the URL"
                        .into(),
                );
            }
            if url.contains('?') || url.contains('#') {
                return Err("store.url must not carry a query or fragment".into());
            }
        } else if !url.starts_with('/') {
            return Err("store.url must be s3://bucket[/prefix] or an absolute path".into());
        }
        if let Some(secret) = &store.credentials_secret
            && !is_dns_subdomain(secret)
        {
            return Err(format!(
                "store.credentialsSecret '{secret}' is not a valid Secret name"
            ));
        }
    }
    if spec.timeout_seconds == Some(0) {
        return Err("timeoutSeconds must be positive when set".into());
    }
    Ok(())
}

fn env(name: &str, value: impl Into<String>) -> EnvVar {
    EnvVar {
        name: name.into(),
        value: Some(value.into()),
        ..Default::default()
    }
}

fn non_empty<T>(v: Vec<T>) -> Option<Vec<T>> {
    if v.is_empty() { None } else { Some(v) }
}

fn empty_dir(name: &str) -> Volume {
    Volume {
        name: name.into(),
        empty_dir: Some(EmptyDirVolumeSource::default()),
        ..Default::default()
    }
}

fn mount(name: &str, path: &str, read_only: bool) -> VolumeMount {
    VolumeMount {
        name: name.into(),
        mount_path: path.into(),
        read_only: read_only.then_some(true),
        ..Default::default()
    }
}

fn pvc_volume(claim_name: &str) -> Volume {
    Volume {
        name: CACHE_VOLUME.into(),
        persistent_volume_claim: Some(PersistentVolumeClaimVolumeSource {
            claim_name: claim_name.to_string(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Environment that makes the engine run inside a pod: no systemd user bus
/// exists, so containment is the pod's own cgroup (`BLUT_NO_CONTAIN=1`), and
/// every path the engine writes lives on the writable scratch volume.
fn engine_env() -> Vec<EnvVar> {
    vec![
        env("BLUT_NO_CONTAIN", "1"),
        env("HOME", WORK_DIR),
        env("XDG_DATA_HOME", format!("{WORK_DIR}/data")),
        env("XDG_CONFIG_HOME", format!("{WORK_DIR}/config")),
        env("XDG_CACHE_HOME", format!("{WORK_DIR}/cache")),
        env("XDG_STATE_HOME", format!("{WORK_DIR}/state")),
        env("XDG_RUNTIME_DIR", format!("{WORK_DIR}/run")),
        env("TMPDIR", "/tmp"),
    ]
}

fn restricted_security_context() -> SecurityContext {
    SecurityContext {
        allow_privilege_escalation: Some(false),
        capabilities: Some(Capabilities {
            drop: Some(vec!["ALL".into()]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn pod_security_context() -> PodSecurityContext {
    PodSecurityContext {
        seccomp_profile: Some(SeccompProfile {
            type_: "RuntimeDefault".into(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// The ConfigMap an inline plan's recipe is copied into. `None` for a
/// `configMapRef` plan, whose ConfigMap the user owns.
pub fn build_plan_configmap(plan: &BlutPlan) -> Option<ConfigMap> {
    let PlanSource::Inline { content, filename } = &plan.spec.plan_source else {
        return None;
    };
    Some(ConfigMap {
        metadata: ObjectMeta {
            name: Some(plan_configmap_name(plan)),
            namespace: plan.namespace(),
            labels: Some(plan_labels(plan)),
            owner_references: plan.controller_owner_ref(&()).map(|r| vec![r]),
            ..Default::default()
        },
        data: Some(BTreeMap::from([(filename.clone(), content.clone())])),
        immutable: Some(true),
        ..Default::default()
    })
}

/// Build the `Job` that runs a `BlutPlan` to completion.
///
/// The container runs `<command> recipe declare /blut/plan/<file> --run`
/// directly as PID 1, so a pod deletion's SIGTERM reaches the engine's cancel
/// handler. The engine writes its run summary to the termination-message file,
/// which is how per-stage state and cache counts reach the plan's status.
pub fn build_plan_job(plan: &BlutPlan) -> Job {
    let spec = &plan.spec;
    let labels = {
        let mut l = plan_labels(plan);
        l.insert(
            DATA_CLASS_LABEL.into(),
            spec.data_class_ceiling.as_str().into(),
        );
        l
    };
    let file = plan_file(spec).to_string();
    // An inline recipe is the only key of the operator's ConfigMap. A
    // referenced ConfigMap is mounted whole, so small inputs the recipe names
    // (`/blut/plan/<key>`) can travel with it.
    let (configmap, items) = match &spec.plan_source {
        PlanSource::Inline { filename, .. } => (
            plan_configmap_name(plan),
            Some(vec![KeyToPath {
                key: filename.clone(),
                path: file.clone(),
                ..Default::default()
            }]),
        ),
        PlanSource::ConfigMapRef { name, .. } => (name.clone(), None),
    };

    let mut args = vec![
        "recipe".to_string(),
        "declare".to_string(),
        format!("{PLAN_DIR}/{file}"),
        "--run".to_string(),
    ];
    if let Some(a) = &spec.args {
        args.push("--args".into());
        args.push(a.clone());
    }

    let mut env = engine_env();
    env.push(env_var_summary());
    let mut volumes = vec![
        Volume {
            name: PLAN_VOLUME.into(),
            config_map: Some(ConfigMapVolumeSource {
                name: configmap,
                items,
                ..Default::default()
            }),
            ..Default::default()
        },
        empty_dir(WORK_VOLUME),
        empty_dir(TMP_VOLUME),
    ];
    let mut mounts = vec![
        mount(PLAN_VOLUME, PLAN_DIR, true),
        mount(WORK_VOLUME, WORK_DIR, false),
        mount(TMP_VOLUME, "/tmp", false),
    ];
    if let CacheConfig::Pvc { claim_name } = &spec.cache {
        // The engine's global cache tier, used with --shared-cache.
        env.push(self::env("LAMU_TRAIN_CACHE_DIR", CACHE_MOUNT_PATH));
        volumes.push(pvc_volume(claim_name));
        mounts.push(mount(CACHE_VOLUME, CACHE_MOUNT_PATH, false));
        args.push("--shared-cache".into());
    }
    if let Some(store) = &spec.store {
        env.push(self::env("BLUT_REMOTE_CACHE", store.url.clone()));
        if let Some(secret) = &store.credentials_secret {
            for key in STORE_CREDENTIAL_KEYS {
                env.push(EnvVar {
                    name: (*key).into(),
                    value_from: Some(EnvVarSource {
                        secret_key_ref: Some(SecretKeySelector {
                            name: secret.clone(),
                            key: (*key).into(),
                            optional: Some(true),
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                });
            }
        }
    }

    let container = Container {
        name: "blut-plan".into(),
        image: Some(spec.cookbook_image.clone()),
        command: Some(vec![
            spec.command
                .clone()
                .unwrap_or_else(|| DEFAULT_COMMAND.to_string()),
        ]),
        args: Some(args),
        env: Some(env),
        volume_mounts: Some(mounts),
        resources: plan_resources(spec),
        termination_message_path: Some(TERMINATION_MESSAGE_PATH.into()),
        termination_message_policy: Some("FallbackToLogsOnError".into()),
        security_context: Some(restricted_security_context()),
        ..Default::default()
    };

    let pod_spec = PodSpec {
        containers: vec![container],
        restart_policy: Some("Never".into()),
        volumes: Some(volumes),
        automount_service_account_token: Some(false),
        enable_service_links: Some(false),
        security_context: Some(pod_security_context()),
        ..Default::default()
    };

    Job {
        metadata: ObjectMeta {
            name: Some(plan_job_name(plan)),
            namespace: plan.namespace(),
            labels: Some(labels.clone()),
            owner_references: plan.controller_owner_ref(&()).map(|r| vec![r]),
            ..Default::default()
        },
        spec: Some(JobSpec {
            backoff_limit: Some(i32::try_from(spec.max_retries).unwrap_or(i32::MAX)),
            active_deadline_seconds: spec.timeout_seconds.map(i64::from),
            // A pod lost to a drain, a preemption or a node going away is the
            // cluster's doing, not the plan's: retry it without spending one
            // of the plan's retries. Kubernetes marks such pods with the
            // DisruptionTarget condition.
            pod_failure_policy: Some(PodFailurePolicy {
                rules: vec![PodFailurePolicyRule {
                    action: "Ignore".into(),
                    on_pod_conditions: Some(vec![PodFailurePolicyOnPodConditionsPattern {
                        type_: "DisruptionTarget".into(),
                        status: "True".into(),
                    }]),
                    ..Default::default()
                }],
            }),
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(labels),
                    ..Default::default()
                }),
                spec: Some(pod_spec),
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn env_var_summary() -> EnvVar {
    env("BLUT_RUN_SUMMARY", TERMINATION_MESSAGE_PATH)
}

fn plan_resources(spec: &BlutPlanSpec) -> Option<ResourceRequirements> {
    let mut limits = BTreeMap::new();
    if let Some(r) = spec.resources {
        limits.insert("cpu".to_string(), Quantity(format!("{}m", r.cpu_millis)));
        limits.insert(
            "memory".to_string(),
            Quantity(format!("{}Mi", r.memory_mib)),
        );
    }
    if spec.gpus > 0 {
        limits.insert(
            "nvidia.com/gpu".to_string(),
            Quantity(spec.gpus.to_string()),
        );
    }
    if limits.is_empty() {
        return None;
    }
    Some(ResourceRequirements {
        // Requests equal limits: a stable QoS class, and the GPU request
        // Kubernetes requires to equal its limit.
        requests: Some(limits.clone()),
        limits: Some(limits),
        ..Default::default()
    })
}

/// The run summary the engine writes to the termination-message file
/// (`BLUT_RUN_SUMMARY`, engine `src/cli/run_summary.rs`).
#[derive(serde::Deserialize, Debug, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RunSummary {
    /// Summary format version (1).
    pub version: u32,
    /// `succeeded` or `failed`.
    pub state: String,
    /// The engine's job id inside the pod.
    #[serde(default)]
    pub job: Option<String>,
    /// Stages in the plan.
    #[serde(default)]
    pub stages_total: Option<u32>,
    /// Stages served from the cache.
    #[serde(default)]
    pub cache_hits: Option<u32>,
    /// Stages computed.
    #[serde(default)]
    pub cache_misses: Option<u32>,
    /// Last state of each stage.
    #[serde(default)]
    pub stages: BTreeMap<String, String>,
    /// The failure, when `state` is `failed`.
    #[serde(default)]
    pub error: Option<String>,
}

/// Parse a termination message as a run summary. Anything else (the log tail
/// the kubelet substitutes when the engine wrote nothing) is `None`.
pub fn parse_run_summary(message: &str) -> Option<RunSummary> {
    let summary: RunSummary = serde_json::from_str(message.trim()).ok()?;
    (summary.version == 1).then_some(summary)
}

/// What became of one pod of the plan's Job.
#[derive(Debug, Default)]
struct PodOutcome {
    name: String,
    created: Option<k8s_openapi::jiff::Timestamp>,
    terminated: bool,
    exit_code: Option<i32>,
    reason: Option<String>,
    message: Option<String>,
    disrupted: bool,
    deleting: bool,
    running: bool,
}

impl PodOutcome {
    /// The pod is gone or going without having succeeded.
    fn lost(&self) -> bool {
        self.disrupted || self.deleting || (self.terminated && self.exit_code != Some(0))
    }

    /// One line on what happened to this pod, e.g. `pod x was deleted (exit 143)`.
    fn describe(&self) -> String {
        let mut detail = Vec::new();
        if let Some(r) = &self.reason {
            detail.push(r.clone());
        }
        if let Some(code) = self.exit_code {
            detail.push(format!("exit {code}"));
        }
        if let Some(m) = &self.message {
            match parse_run_summary(m) {
                Some(s) if s.state == "failed" => {
                    detail.push(first_line(s.error.as_deref().unwrap_or("plan failed"), 400));
                }
                Some(_) => {}
                None => detail.push(first_line(m, 400)),
            }
        }
        let what = if self.disrupted {
            "was disrupted (not counted against maxRetries)"
        } else if self.deleting {
            "was deleted"
        } else {
            "failed"
        };
        if detail.is_empty() {
            format!("pod {} {what}", self.name)
        } else {
            format!("pod {} {what}: {}", self.name, detail.join(", "))
        }
    }
}

fn pod_outcome(pod: &Pod) -> PodOutcome {
    let mut out = PodOutcome {
        name: pod.name_any(),
        created: pod.metadata.creation_timestamp.as_ref().map(|t| t.0),
        deleting: pod.metadata.deletion_timestamp.is_some(),
        ..Default::default()
    };
    let Some(status) = &pod.status else {
        return out;
    };
    out.running = status.phase.as_deref() == Some("Running") && !out.deleting;
    out.disrupted = status
        .conditions
        .iter()
        .flatten()
        .any(|c| c.type_ == "DisruptionTarget" && c.status == "True");
    if matches!(status.phase.as_deref(), Some("Failed")) {
        // Pod-level failure (Evicted, DeadlineExceeded, NodeLost …).
        out.terminated = true;
        out.reason = status.reason.clone();
        out.message = status.message.clone();
    }
    let container = status
        .container_statuses
        .iter()
        .flatten()
        .find(|c| c.name == "blut-plan");
    if let Some(term) = container
        .and_then(|c| c.state.as_ref())
        .and_then(|s| s.terminated.as_ref())
    {
        out.terminated = true;
        out.exit_code = Some(term.exit_code);
        out.reason = term.reason.clone().or(out.reason);
        out.message = term.message.clone().or(out.message);
    }
    out
}

fn first_line(text: &str, max: usize) -> String {
    let line = text
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("");
    let mut out: String = line.chars().take(max).collect();
    if line.chars().count() > max {
        out.push('…');
    }
    out
}

fn job_condition<'a>(
    job: &'a Job,
    type_: &str,
) -> Option<&'a k8s_openapi::api::batch::v1::JobCondition> {
    job.status
        .as_ref()?
        .conditions
        .iter()
        .flatten()
        .find(|c| c.type_ == type_ && c.status == "True")
}

/// The status a plan should show, given its Job (if any), that Job's pods, and
/// the status it showed before (`plan.status`).
///
/// A deleted or evicted pod disappears from the API soon after it stops, so
/// what happened to it is carried forward in `lastPodFailure`, and `attempts`
/// counts every pod the operator has seen (`pod` holds the latest), not only
/// the ones still listed.
pub fn plan_status(plan: &BlutPlan, job: Option<&Job>, pods: &[Pod]) -> BlutPlanStatus {
    let prev = plan.status.clone().unwrap_or_default();
    let mut status = BlutPlanStatus {
        observed_generation: plan.metadata.generation,
        attempts: prev.attempts,
        pod: prev.pod.clone(),
        last_pod_failure: prev.last_pod_failure.clone(),
        ..Default::default()
    };
    let Some(job) = job else {
        status.phase = PlanPhase::Pending;
        status.message = Some("creating the Job".into());
        return status;
    };
    status.job_name = job.metadata.name.clone();

    let mut outcomes: Vec<PodOutcome> = pods.iter().map(pod_outcome).collect();
    outcomes.sort_by_key(|o| (o.created, o.name.clone()));
    if let Some(latest) = outcomes.last()
        && status.pod.as_deref() != Some(latest.name.as_str())
    {
        status.attempts += 1;
        status.pod = Some(latest.name.clone());
    }
    if let Some(js) = &job.status {
        let n = |v: Option<i32>| u32::try_from(v.unwrap_or(0)).unwrap_or(0);
        status.attempts = status
            .attempts
            .max(n(js.active) + n(js.succeeded) + n(js.failed));
    }
    status.attempts = status
        .attempts
        .max(u32::try_from(outcomes.len()).unwrap_or(u32::MAX));
    if let Some(lost) = outcomes.iter().rev().find(|o| o.lost()) {
        status.last_pod_failure = Some(lost.describe());
    }

    let last_terminated = outcomes.iter().rev().find(|o| o.terminated);
    let summary = last_terminated
        .and_then(|o| o.message.as_deref())
        .and_then(parse_run_summary);
    if let Some(s) = &summary {
        status.stages = s.stages.clone();
        status.cache_hits = s.cache_hits;
        status.cache_misses = s.cache_misses;
    } else {
        status.stages = prev.stages.clone();
        status.cache_hits = prev.cache_hits;
        status.cache_misses = prev.cache_misses;
    }

    if job_condition(job, "Complete").is_some() {
        status.phase = PlanPhase::Succeeded;
        status.message = Some(match &summary {
            Some(s) => format!(
                "succeeded: {} stage(s), {} from cache, {} computed",
                s.stages_total.unwrap_or(0),
                s.cache_hits.unwrap_or(0),
                s.cache_misses.unwrap_or(0)
            ),
            None => "succeeded".into(),
        });
        return status;
    }
    if let Some(cond) = job_condition(job, "Failed") {
        status.phase = PlanPhase::Failed;
        let why = match cond.reason.as_deref() {
            Some("BackoffLimitExceeded") => "out of retries".to_string(),
            Some("DeadlineExceeded") => "timeoutSeconds exceeded".to_string(),
            Some(other) => other.to_string(),
            None => "failed".to_string(),
        };
        status.message = Some(match &status.last_pod_failure {
            Some(cause) => format!("{why}: last {cause}"),
            None => why,
        });
        return status;
    }

    let running = outcomes.iter().any(|o| o.running);
    status.phase = if running {
        PlanPhase::Running
    } else {
        PlanPhase::Pending
    };
    let state = if running {
        "running"
    } else {
        "waiting for a pod to start"
    };
    status.message = Some(match &status.last_pod_failure {
        Some(cause) => format!("attempt {}, {state}; previous {cause}", status.attempts),
        None => state.to_string(),
    });
    status
}

/// What the plan reconciler does next.
#[derive(Debug, PartialEq, Eq)]
pub enum PlanAction {
    /// Mark the plan `Failed` with this message; create nothing.
    Reject(String),
    /// Create the Job (and, for an inline plan, its ConfigMap).
    CreateJob,
    /// The Job exists; mirror its state into the status.
    Observe,
    /// The plan already finished and its Job is gone; leave it alone. A
    /// finished plan is never run a second time.
    KeepFinished,
}

/// `meta` carries a controller owner reference to `plan`.
pub fn owned_by(meta: &ObjectMeta, plan: &BlutPlan) -> bool {
    let uid = plan.metadata.uid.as_deref();
    meta.owner_references
        .iter()
        .flatten()
        .any(|r| Some(r.uid.as_str()) == uid && r.controller == Some(true))
}

/// Decide the next step from the plan and whether its Job exists (and is
/// owned by this plan).
pub fn plan_action(plan: &BlutPlan, job: Option<&Job>) -> PlanAction {
    if let Err(e) = validate_plan(plan) {
        return PlanAction::Reject(format!("invalid spec: {e}"));
    }
    match job {
        Some(job) => {
            if owned_by(&job.metadata, plan) {
                PlanAction::Observe
            } else {
                PlanAction::Reject(format!(
                    "a Job named {} already exists and does not belong to this plan",
                    plan_job_name(plan)
                ))
            }
        }
        None => {
            let finished = plan
                .status
                .as_ref()
                .is_some_and(|s| s.phase.is_terminal() && s.job_name.is_some());
            if finished {
                PlanAction::KeepFinished
            } else {
                PlanAction::CreateJob
            }
        }
    }
}

/// Build the headless `Service` fronting a worker pool's StatefulSet (stable
/// per-pod DNS, so a scheduler can reach node-0 as the mesh `--seed`).
pub fn build_pool_service(pool: &BlutWorkerPool) -> Service {
    let instance = pool.name_any();
    Service {
        metadata: ObjectMeta {
            name: Some(format!("{instance}-workers")),
            namespace: pool.namespace(),
            labels: Some(pool_labels(&instance)),
            owner_references: pool.controller_owner_ref(&()).map(|r| vec![r]),
            ..Default::default()
        },
        spec: Some(ServiceSpec {
            cluster_ip: Some("None".into()),
            selector: Some(pool_labels(&instance)),
            // The mesh is QUIC, which is UDP.
            ports: Some(vec![ServicePort {
                name: Some("mesh".into()),
                port: MESH_PORT,
                protocol: Some("UDP".into()),
                ..Default::default()
            }]),
            publish_not_ready_addresses: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn pool_resource_requirements(cpu: u32, mem_gib: u32, gpus: u32) -> ResourceRequirements {
    let mut requests = BTreeMap::new();
    requests.insert("cpu".to_string(), Quantity(cpu.to_string()));
    requests.insert("memory".to_string(), Quantity(format!("{mem_gib}Gi")));
    if gpus > 0 {
        requests.insert("nvidia.com/gpu".to_string(), Quantity(gpus.to_string()));
    }
    ResourceRequirements {
        limits: Some(requests.clone()),
        requests: Some(requests),
        ..Default::default()
    }
}

/// Build the `StatefulSet` of worker node pods for a pool. Pods run
/// `<command> p2p node --no-scheduler --listen 0.0.0.0:9320`.
pub fn build_pool_statefulset(pool: &BlutWorkerPool) -> StatefulSet {
    let instance = pool.name_any();
    let res = pool.spec.resources;

    let mut env = engine_env();
    env.push(self::env("BLUT_NODE_ROLE", "worker"));
    let mut args = vec![
        "p2p".to_string(),
        "node".to_string(),
        "--no-scheduler".to_string(),
        "--listen".to_string(),
        format!("0.0.0.0:{MESH_PORT}"),
    ];
    let mut volumes = vec![empty_dir(WORK_VOLUME), empty_dir(TMP_VOLUME)];
    let mut mounts = vec![
        mount(WORK_VOLUME, WORK_DIR, false),
        mount(TMP_VOLUME, "/tmp", false),
    ];
    if let CacheConfig::Pvc { claim_name } = &pool.spec.cache {
        volumes.push(pvc_volume(claim_name));
        mounts.push(mount(CACHE_VOLUME, CACHE_MOUNT_PATH, false));
        args.push("--shared-store".into());
        args.push(CACHE_MOUNT_PATH.into());
    }

    let container = Container {
        name: "blut-node".into(),
        image: Some(pool.spec.image.clone()),
        command: Some(vec![
            pool.spec
                .command
                .clone()
                .unwrap_or_else(|| DEFAULT_COMMAND.to_string()),
        ]),
        args: Some(args),
        env: Some(env),
        ports: Some(vec![ContainerPort {
            name: Some("mesh".into()),
            container_port: MESH_PORT,
            protocol: Some("UDP".into()),
            ..Default::default()
        }]),
        resources: Some(pool_resource_requirements(
            res.cpu_cores,
            res.memory_gib,
            res.gpus,
        )),
        volume_mounts: non_empty(mounts),
        security_context: Some(restricted_security_context()),
        ..Default::default()
    };

    let pod_spec = PodSpec {
        containers: vec![container],
        volumes: non_empty(volumes),
        automount_service_account_token: Some(false),
        enable_service_links: Some(false),
        security_context: Some(pod_security_context()),
        ..Default::default()
    };

    StatefulSet {
        metadata: ObjectMeta {
            name: Some(format!("{instance}-workers")),
            namespace: pool.namespace(),
            labels: Some(pool_labels(&instance)),
            owner_references: pool.controller_owner_ref(&()).map(|r| vec![r]),
            ..Default::default()
        },
        spec: Some(StatefulSetSpec {
            replicas: Some(i32::try_from(pool.spec.replicas).unwrap_or(i32::MAX)),
            service_name: Some(format!("{instance}-workers")),
            selector: LabelSelector {
                match_labels: Some(pool_labels(&instance)),
                ..Default::default()
            },
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(pool_labels(&instance)),
                    ..Default::default()
                }),
                spec: Some(pod_spec),
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

// ── Live controller glue ────────────────────────────────────────────────────

/// Shared reconcile context.
pub struct Context {
    /// The API client.
    pub client: Client,
}

/// Field manager for server-side apply and status patches.
pub const FIELD_MANAGER: &str = "blut-operator";

/// How often a plan that is still running is looked at even without a watch
/// event. Job and pod changes normally trigger a reconcile first; this bounds
/// how stale a status can get if an event is missed.
const RUNNING_RESYNC: Duration = Duration::from_secs(30);

/// Drop `null`s: in a server-side apply an explicit null is an instruction,
/// not an absence.
fn strip_nulls(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => serde_json::Value::Object(
            m.into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, strip_nulls(v)))
                .collect(),
        ),
        other => other,
    }
}

/// Server-side apply of a status: the operator owns every status field, so a
/// stage name or message that disappears from the new status is removed
/// rather than merged into the old one.
pub fn status_apply_body<K: Resource<DynamicType = ()>>(
    status: &impl serde::Serialize,
) -> serde_json::Value {
    serde_json::json!({
        "apiVersion": K::api_version(&()),
        "kind": K::kind(&()),
        "status": strip_nulls(serde_json::to_value(status).unwrap_or_default()),
    })
}

async fn patch_plan_status(
    plans: &Api<BlutPlan>,
    plan: &BlutPlan,
    status: BlutPlanStatus,
) -> Result<(), kube::Error> {
    if plan.status.as_ref() == Some(&status) {
        return Ok(());
    }
    plans
        .patch_status(
            &plan.name_any(),
            &PatchParams::apply(FIELD_MANAGER).force(),
            &Patch::Apply(status_apply_body::<BlutPlan>(&status)),
        )
        .await?;
    Ok(())
}

fn is_conflict(e: &kube::Error) -> bool {
    matches!(e, kube::Error::Api(s) if s.code == 409)
}

/// Reconcile a `BlutPlan`: validate it, create its Job once, and mirror the
/// Job's progress into the status.
pub async fn reconcile_plan(plan: Arc<BlutPlan>, ctx: Arc<Context>) -> Result<Action, kube::Error> {
    if plan.metadata.deletion_timestamp.is_some() {
        // Owner references garbage-collect the Job, its pods and the ConfigMap.
        return Ok(Action::await_change());
    }
    let ns = plan.namespace().unwrap_or_else(|| "default".into());
    let plans: Api<BlutPlan> = Api::namespaced(ctx.client.clone(), &ns);
    let jobs: Api<Job> = Api::namespaced(ctx.client.clone(), &ns);
    let pods: Api<Pod> = Api::namespaced(ctx.client.clone(), &ns);

    let job_name = plan_job_name(&plan);
    let mut job = jobs.get_opt(&job_name).await?;
    match plan_action(&plan, job.as_ref()) {
        PlanAction::Reject(message) => {
            tracing::info!(plan = %plan.name_any(), %ns, "rejected: {message}");
            let status = BlutPlanStatus {
                phase: PlanPhase::Failed,
                message: Some(message),
                observed_generation: plan.metadata.generation,
                ..plan.status.clone().unwrap_or_default()
            };
            patch_plan_status(&plans, &plan, status).await?;
            return Ok(Action::await_change());
        }
        PlanAction::KeepFinished => return Ok(Action::await_change()),
        PlanAction::CreateJob => {
            if let Some(cm) = build_plan_configmap(&plan) {
                let cms: Api<ConfigMap> = Api::namespaced(ctx.client.clone(), &ns);
                match cms.create(&PostParams::default(), &cm).await {
                    Ok(_) => {}
                    // Ours from an earlier reconcile, or someone else's: only
                    // the first may be mounted.
                    Err(e) if is_conflict(&e) => {
                        let existing = cms.get(&plan_configmap_name(&plan)).await?;
                        if !owned_by(&existing.metadata, &plan) {
                            let status = BlutPlanStatus {
                                phase: PlanPhase::Failed,
                                message: Some(format!(
                                    "a ConfigMap named {} already exists and does not belong \
                                     to this plan",
                                    plan_configmap_name(&plan)
                                )),
                                observed_generation: plan.metadata.generation,
                                ..plan.status.clone().unwrap_or_default()
                            };
                            patch_plan_status(&plans, &plan, status).await?;
                            return Ok(Action::await_change());
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            match jobs
                .create(&PostParams::default(), &build_plan_job(&plan))
                .await
            {
                Ok(created) => {
                    tracing::info!(plan = %plan.name_any(), %ns, job = %job_name, "created Job");
                    job = Some(created);
                }
                // Created by an earlier reconcile whose result we did not see.
                Err(e) if is_conflict(&e) => job = jobs.get_opt(&job_name).await?,
                Err(e) => return Err(e),
            }
        }
        PlanAction::Observe => {}
    }

    let pod_list = match &job {
        Some(j) => {
            let uid = j.metadata.uid.clone().unwrap_or_default();
            let lp = ListParams::default().labels(&format!("{PLAN_LABEL}={}", plan.name_any()));
            pods.list(&lp)
                .await?
                .items
                .into_iter()
                .filter(|p| {
                    p.owner_references()
                        .iter()
                        .any(|r| r.uid == uid && r.kind == "Job")
                })
                .collect::<Vec<_>>()
        }
        None => Vec::new(),
    };
    let status = plan_status(&plan, job.as_ref(), &pod_list);
    let terminal = status.phase.is_terminal();
    patch_plan_status(&plans, &plan, status).await?;
    Ok(if terminal {
        Action::await_change()
    } else {
        Action::requeue(RUNNING_RESYNC)
    })
}

/// Reconcile a `BlutWorkerPool`: apply its headless Service and StatefulSet
/// and mirror the ready count.
pub async fn reconcile_pool(
    pool: Arc<BlutWorkerPool>,
    ctx: Arc<Context>,
) -> Result<Action, kube::Error> {
    if pool.metadata.deletion_timestamp.is_some() {
        return Ok(Action::await_change());
    }
    let ns = pool.namespace().unwrap_or_else(|| "default".into());
    let svcs: Api<Service> = Api::namespaced(ctx.client.clone(), &ns);
    let sets: Api<StatefulSet> = Api::namespaced(ctx.client.clone(), &ns);
    let pools: Api<BlutWorkerPool> = Api::namespaced(ctx.client.clone(), &ns);
    let pp = PatchParams::apply(FIELD_MANAGER).force();

    let svc = build_pool_service(&pool);
    svcs.patch(
        &svc.metadata.name.clone().unwrap_or_default(),
        &pp,
        &Patch::Apply(&svc),
    )
    .await?;

    let sts = build_pool_statefulset(&pool);
    let applied = sets
        .patch(
            &sts.metadata.name.clone().unwrap_or_default(),
            &pp,
            &Patch::Apply(&sts),
        )
        .await?;
    let ready = applied
        .status
        .as_ref()
        .and_then(|s| s.ready_replicas)
        .unwrap_or(0);
    let status = BlutWorkerPoolStatus {
        ready_replicas: u32::try_from(ready).unwrap_or(0),
        observed_generation: pool.metadata.generation,
    };
    if pool.status.as_ref() != Some(&status) {
        pools
            .patch_status(
                &pool.name_any(),
                &PatchParams::apply(FIELD_MANAGER).force(),
                &Patch::Apply(status_apply_body::<BlutWorkerPool>(&status)),
            )
            .await?;
    }
    Ok(Action::await_change())
}

/// Error policy: retry after a short back-off on any reconcile error.
pub fn error_policy<K>(_obj: Arc<K>, err: &kube::Error, _ctx: Arc<Context>) -> Action {
    tracing::warn!("reconcile error: {err}");
    Action::requeue(Duration::from_secs(10))
}

#[cfg(test)]
mod tests;
