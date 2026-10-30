// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Reconcile-logic tests: every decision the operator makes, without a cluster.

use super::*;
use crate::crds::{
    BlutPlanSpec, BlutWorkerPoolSpec, DataClassCeiling, PlanResources, PoolResources, StoreConfig,
};
use k8s_openapi::api::batch::v1::{JobCondition, JobStatus};
use k8s_openapi::api::core::v1::{
    ContainerState, ContainerStateTerminated, ContainerStatus, PodCondition, PodStatus,
};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{OwnerReference, Time};

fn spec(source: PlanSource) -> BlutPlanSpec {
    BlutPlanSpec {
        plan_source: source,
        cookbook_image: "ghcr.io/example/blut:1".into(),
        command: None,
        data_class_ceiling: DataClassCeiling::Internal,
        cache: CacheConfig::None {},
        store: None,
        args: None,
        resources: None,
        gpus: 0,
        max_retries: 2,
        timeout_seconds: None,
    }
}

fn inline(content: &str) -> PlanSource {
    PlanSource::Inline {
        content: content.into(),
        filename: "plan.json".into(),
    }
}

fn plan_with(spec: BlutPlanSpec) -> BlutPlan {
    let mut p = BlutPlan::new("demo", spec);
    p.metadata.namespace = Some("blut".into());
    p.metadata.uid = Some("uid-123".into());
    p.metadata.generation = Some(1);
    p
}

fn plan() -> BlutPlan {
    plan_with(spec(inline(r#"{"version":1,"nodes":[]}"#)))
}

fn container(job: &Job) -> &Container {
    &job.spec
        .as_ref()
        .unwrap()
        .template
        .spec
        .as_ref()
        .unwrap()
        .containers[0]
}

fn env_value<'a>(c: &'a Container, name: &str) -> Option<&'a str> {
    c.env
        .as_ref()?
        .iter()
        .find(|e| e.name == name)
        .and_then(|e| e.value.as_deref())
}

// ── validation ─────────────────────────────────────────────────────────────

#[test]
fn a_well_formed_plan_validates() {
    assert_eq!(validate_plan(&plan()), Ok(()));
}

#[test]
fn bad_specs_are_rejected_with_a_message_naming_the_field() {
    let cases: Vec<(BlutPlanSpec, &str)> = vec![
        (
            spec(PlanSource::Inline {
                content: "x".into(),
                filename: "plan.yaml".into(),
            }),
            ".json (PlanSpec), .toml (chain) or .star",
        ),
        (
            spec(PlanSource::Inline {
                content: "x".into(),
                filename: "../etc/plan.json".into(),
            }),
            "plain file name",
        ),
        (spec(inline("   ")), "content is empty"),
        (
            BlutPlanSpec {
                args: Some("{}".into()),
                ..spec(inline("{}"))
            },
            "args is only for .star",
        ),
        (
            BlutPlanSpec {
                args: Some("{not json".into()),
                ..spec(PlanSource::Inline {
                    content: "def build(a): pass".into(),
                    filename: "p.star".into(),
                })
            },
            "args is not valid JSON",
        ),
        (
            BlutPlanSpec {
                command: Some("blut --verbose".into()),
                ..spec(inline("{}"))
            },
            "without arguments",
        ),
        (
            BlutPlanSpec {
                store: Some(StoreConfig {
                    url: "s3://AKIA:secret@bucket/x".into(),
                    credentials_secret: None,
                }),
                ..spec(inline("{}"))
            },
            "never in the URL",
        ),
        (
            BlutPlanSpec {
                store: Some(StoreConfig {
                    url: "relative/dir".into(),
                    credentials_secret: None,
                }),
                ..spec(inline("{}"))
            },
            "absolute path",
        ),
        (
            BlutPlanSpec {
                store: Some(StoreConfig {
                    url: "s3://bucket".into(),
                    credentials_secret: Some("Not_A_Name".into()),
                }),
                ..spec(inline("{}"))
            },
            "credentialsSecret",
        ),
        (
            BlutPlanSpec {
                cache: CacheConfig::Pvc {
                    claim_name: "".into(),
                },
                ..spec(inline("{}"))
            },
            "claimName",
        ),
        (
            spec(PlanSource::ConfigMapRef {
                name: "UPPER".into(),
                key: "plan.json".into(),
            }),
            "configMapRef.name",
        ),
        (
            BlutPlanSpec {
                timeout_seconds: Some(0),
                ..spec(inline("{}"))
            },
            "timeoutSeconds",
        ),
    ];
    for (s, needle) in cases {
        let err = validate_plan(&plan_with(s.clone())).expect_err(needle);
        assert!(err.contains(needle), "{err:?} should mention {needle:?}");
        assert!(
            !err.contains("secret@"),
            "a credential must not be echoed: {err}"
        );
        assert!(matches!(
            plan_action(&plan_with(s), None),
            PlanAction::Reject(m) if m.starts_with("invalid spec: ")
        ));
    }
}

#[test]
fn a_name_too_long_for_the_job_label_is_rejected() {
    let mut p = plan();
    p.metadata.name = Some("a".repeat(MAX_PLAN_NAME + 1));
    assert!(validate_plan(&p).unwrap_err().contains("metadata.name"));
    p.metadata.name = Some("a".repeat(MAX_PLAN_NAME));
    assert_eq!(validate_plan(&p), Ok(()));
}

// ── the Job ────────────────────────────────────────────────────────────────

#[test]
fn the_job_runs_the_mounted_recipe_directly_with_the_engine_env() {
    let job = build_plan_job(&plan());
    assert_eq!(job.metadata.name.as_deref(), Some("demo-run"));
    assert_eq!(job.metadata.namespace.as_deref(), Some("blut"));
    let owners = job.metadata.owner_references.as_ref().unwrap();
    assert_eq!(owners[0].kind, "BlutPlan");
    assert_eq!(owners[0].uid, "uid-123");
    assert_eq!(owners[0].controller, Some(true));

    let c = container(&job);
    assert_eq!(c.command.as_deref(), Some(&["blut".to_string()][..]));
    assert_eq!(
        c.args.as_deref().unwrap(),
        ["recipe", "declare", "/blut/plan/plan.json", "--run"]
    );
    assert_eq!(env_value(c, "BLUT_NO_CONTAIN"), Some("1"));
    assert_eq!(env_value(c, "HOME"), Some(WORK_DIR));
    assert_eq!(
        env_value(c, "BLUT_RUN_SUMMARY"),
        Some(TERMINATION_MESSAGE_PATH)
    );
    assert_eq!(
        c.termination_message_policy.as_deref(),
        Some("FallbackToLogsOnError")
    );
    let pod = job.spec.as_ref().unwrap().template.spec.as_ref().unwrap();
    assert_eq!(pod.restart_policy.as_deref(), Some("Never"));
    assert_eq!(pod.automount_service_account_token, Some(false));
    let sc = c.security_context.as_ref().unwrap();
    assert_eq!(sc.allow_privilege_escalation, Some(false));

    // The inline recipe is mounted from the operator's ConfigMap under its
    // file name, read-only.
    let vol = pod
        .volumes
        .as_ref()
        .unwrap()
        .iter()
        .find(|v| v.name == "plan")
        .unwrap();
    let cm = vol.config_map.as_ref().unwrap();
    assert_eq!(cm.name, "demo-plan");
    assert_eq!(cm.items.as_ref().unwrap()[0].path, "plan.json");
    let m = c
        .volume_mounts
        .as_ref()
        .unwrap()
        .iter()
        .find(|m| m.name == "plan")
        .unwrap();
    assert_eq!(m.mount_path, PLAN_DIR);
    assert_eq!(m.read_only, Some(true));
}

#[test]
fn retries_deadline_and_disruption_policy_come_from_the_spec() {
    let job = build_plan_job(&plan_with(BlutPlanSpec {
        max_retries: 5,
        timeout_seconds: Some(600),
        ..spec(inline("{}"))
    }));
    let js = job.spec.unwrap();
    assert_eq!(js.backoff_limit, Some(5));
    assert_eq!(js.active_deadline_seconds, Some(600));
    let rule = &js.pod_failure_policy.unwrap().rules[0];
    assert_eq!(rule.action, "Ignore");
    assert_eq!(
        rule.on_pod_conditions.as_ref().unwrap()[0].type_,
        "DisruptionTarget"
    );
}

#[test]
fn the_inline_configmap_is_owned_and_immutable() {
    let cm = build_plan_configmap(&plan()).unwrap();
    assert_eq!(cm.metadata.name.as_deref(), Some("demo-plan"));
    assert_eq!(cm.immutable, Some(true));
    assert_eq!(
        cm.metadata.owner_references.unwrap()[0].uid,
        "uid-123".to_string()
    );
    assert!(cm.data.unwrap().contains_key("plan.json"));
    let referenced = plan_with(spec(PlanSource::ConfigMapRef {
        name: "mine".into(),
        key: "chain.toml".into(),
    }));
    assert!(build_plan_configmap(&referenced).is_none());
    let job = build_plan_job(&referenced);
    assert_eq!(
        container(&job).args.as_deref().unwrap()[2],
        "/blut/plan/chain.toml"
    );
    // A referenced ConfigMap is mounted whole, so data keys travel with it.
    let pod = job.spec.as_ref().unwrap().template.spec.as_ref().unwrap();
    let cm = pod.volumes.as_ref().unwrap()[0]
        .config_map
        .as_ref()
        .unwrap();
    assert_eq!(cm.name, "mine");
    assert!(cm.items.is_none());
}

#[test]
fn store_credentials_come_from_the_secret_by_reference_only() {
    let job = build_plan_job(&plan_with(BlutPlanSpec {
        store: Some(StoreConfig {
            url: "s3://bucket/runs".into(),
            credentials_secret: Some("store-creds".into()),
        }),
        ..spec(inline("{}"))
    }));
    let c = container(&job);
    assert_eq!(env_value(c, "BLUT_REMOTE_CACHE"), Some("s3://bucket/runs"));
    let envs = c.env.as_ref().unwrap();
    for key in STORE_CREDENTIAL_KEYS {
        let e = envs.iter().find(|e| e.name == *key).unwrap();
        assert!(e.value.is_none(), "{key} must not be a literal value");
        let r = e
            .value_from
            .as_ref()
            .unwrap()
            .secret_key_ref
            .as_ref()
            .unwrap();
        assert_eq!(r.name, "store-creds");
        assert_eq!(r.key, *key);
        assert_eq!(r.optional, Some(true));
    }
    // Nothing secret-shaped is anywhere in the rendered Job.
    let rendered = serde_json::to_string(&job).unwrap();
    assert!(!rendered.contains("secretKeyRef\":{\"key\":\"\""));
}

#[test]
fn resources_and_gpus_become_equal_requests_and_limits() {
    let job = build_plan_job(&plan_with(BlutPlanSpec {
        resources: Some(PlanResources {
            cpu_millis: 500,
            memory_mib: 256,
        }),
        gpus: 1,
        ..spec(inline("{}"))
    }));
    let r = container(&job).resources.clone().unwrap();
    let limits = r.limits.unwrap();
    assert_eq!(limits["cpu"].0, "500m");
    assert_eq!(limits["memory"].0, "256Mi");
    assert_eq!(limits["nvidia.com/gpu"].0, "1");
    assert_eq!(r.requests.unwrap(), limits);
    assert!(container(&build_plan_job(&plan())).resources.is_none());
}

#[test]
fn a_pvc_cache_mounts_the_volume_and_turns_on_the_shared_cache() {
    // Regression: PVC mode used to set the cache env but neither mounted the
    // volume nor passed --shared-cache, so the engine never used it.
    let job = build_plan_job(&plan_with(BlutPlanSpec {
        cache: CacheConfig::Pvc {
            claim_name: "cache-pvc".into(),
        },
        ..spec(inline("{}"))
    }));
    let pod = job.spec.as_ref().unwrap().template.spec.as_ref().unwrap();
    let vol = pod
        .volumes
        .as_ref()
        .unwrap()
        .iter()
        .find(|v| v.persistent_volume_claim.is_some())
        .unwrap();
    assert_eq!(
        vol.persistent_volume_claim.as_ref().unwrap().claim_name,
        "cache-pvc"
    );
    let c = container(&job);
    assert!(
        c.volume_mounts
            .as_ref()
            .unwrap()
            .iter()
            .any(|m| m.name == vol.name && m.mount_path == CACHE_MOUNT_PATH)
    );
    assert_eq!(env_value(c, "LAMU_TRAIN_CACHE_DIR"), Some(CACHE_MOUNT_PATH));
    assert!(c.args.as_ref().unwrap().contains(&"--shared-cache".into()));
}

#[test]
fn star_args_and_a_custom_command_reach_the_container() {
    let job = build_plan_job(&plan_with(BlutPlanSpec {
        command: Some("blut-standard".into()),
        args: Some(r#"{"n":2}"#.into()),
        ..spec(PlanSource::Inline {
            content: "def build(a): pass".into(),
            filename: "fan.star".into(),
        })
    }));
    let c = container(&job);
    assert_eq!(c.command.as_ref().unwrap()[0], "blut-standard");
    let args = c.args.as_ref().unwrap();
    assert_eq!(&args[4..], ["--args", r#"{"n":2}"#]);
}

#[test]
fn the_data_class_ceiling_is_a_label_on_job_and_pod() {
    let job = build_plan_job(&plan_with(BlutPlanSpec {
        data_class_ceiling: DataClassCeiling::Public,
        ..spec(inline("{}"))
    }));
    assert_eq!(
        job.metadata.labels.as_ref().unwrap()[DATA_CLASS_LABEL],
        "Public"
    );
    let pod_labels = job.spec.unwrap().template.metadata.unwrap().labels.unwrap();
    assert_eq!(pod_labels[DATA_CLASS_LABEL], "Public");
    assert_eq!(pod_labels[PLAN_LABEL], "demo");
}

// ── decisions ──────────────────────────────────────────────────────────────

fn owned_job(uid: &str) -> Job {
    let mut job = build_plan_job(&plan());
    job.metadata.uid = Some("job-uid".into());
    job.metadata.owner_references = Some(vec![OwnerReference {
        api_version: "blut.lamquant.dev/v1alpha1".into(),
        kind: "BlutPlan".into(),
        name: "demo".into(),
        uid: uid.into(),
        controller: Some(true),
        block_owner_deletion: Some(true),
    }]);
    job
}

#[test]
fn a_new_plan_creates_its_job_and_a_running_one_is_only_observed() {
    assert_eq!(plan_action(&plan(), None), PlanAction::CreateJob);
    assert_eq!(
        plan_action(&plan(), Some(&owned_job("uid-123"))),
        PlanAction::Observe
    );
}

#[test]
fn a_job_owned_by_someone_else_is_not_adopted() {
    match plan_action(&plan(), Some(&owned_job("someone-else"))) {
        PlanAction::Reject(m) => assert!(m.contains("does not belong"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_finished_plan_whose_job_is_gone_is_never_run_again() {
    for phase in [PlanPhase::Succeeded, PlanPhase::Failed] {
        let mut p = plan();
        p.status = Some(BlutPlanStatus {
            phase,
            job_name: Some("demo-run".into()),
            ..Default::default()
        });
        assert_eq!(plan_action(&p, None), PlanAction::KeepFinished);
    }
    // A plan still in flight whose Job vanished gets a new one.
    let mut p = plan();
    p.status = Some(BlutPlanStatus {
        phase: PlanPhase::Running,
        job_name: Some("demo-run".into()),
        ..Default::default()
    });
    assert_eq!(plan_action(&p, None), PlanAction::CreateJob);
}

// ── status ─────────────────────────────────────────────────────────────────

fn job_with(active: i32, succeeded: i32, failed: i32, cond: Option<(&str, &str)>) -> Job {
    let mut job = owned_job("uid-123");
    job.status = Some(JobStatus {
        active: Some(active),
        succeeded: Some(succeeded),
        failed: Some(failed),
        conditions: cond.map(|(t, reason)| {
            vec![JobCondition {
                type_: t.into(),
                status: "True".into(),
                reason: Some(reason.into()),
                ..Default::default()
            }]
        }),
        ..Default::default()
    });
    job
}

fn pod(name: &str, secs: i64, phase: &str, term: Option<(i32, &str, &str)>) -> Pod {
    let mut p = Pod::default();
    p.metadata.name = Some(name.into());
    p.metadata.creation_timestamp = Some(Time(
        k8s_openapi::jiff::Timestamp::from_second(1_800_000_000 + secs).unwrap(),
    ));
    p.status = Some(PodStatus {
        phase: Some(phase.into()),
        container_statuses: Some(vec![ContainerStatus {
            name: "blut-plan".into(),
            state: Some(ContainerState {
                terminated: term.map(|(code, reason, message)| ContainerStateTerminated {
                    exit_code: code,
                    reason: Some(reason.into()),
                    message: Some(message.into()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }]),
        ..Default::default()
    });
    p
}

const OK_SUMMARY: &str = r#"{"version":1,"state":"succeeded","job":"j","stagesTotal":3,"cacheHits":1,"cacheMisses":2,"stages":{"materialize_dataset_path":"cached","filter_dataset":"ok","split_train_eval":"ok"}}"#;

#[test]
fn no_job_yet_is_pending() {
    let s = plan_status(&plan(), None, &[]);
    assert_eq!(s.phase, PlanPhase::Pending);
    assert_eq!(s.observed_generation, Some(1));
}

#[test]
fn a_running_pod_is_running() {
    let s = plan_status(
        &plan(),
        Some(&job_with(1, 0, 0, None)),
        &[pod("demo-run-a", 0, "Running", None)],
    );
    assert_eq!(s.phase, PlanPhase::Running);
    assert_eq!(s.job_name.as_deref(), Some("demo-run"));
    assert_eq!(s.attempts, 1);
}

#[test]
fn success_carries_the_stage_states_and_cache_counts() {
    let s = plan_status(
        &plan(),
        Some(&job_with(0, 1, 0, Some(("Complete", "CompletionsReached")))),
        &[pod(
            "demo-run-a",
            0,
            "Succeeded",
            Some((0, "Completed", OK_SUMMARY)),
        )],
    );
    assert_eq!(s.phase, PlanPhase::Succeeded);
    assert_eq!(s.stages["materialize_dataset_path"], "cached");
    assert_eq!(s.stages.len(), 3);
    assert_eq!(s.cache_hits, Some(1));
    assert_eq!(s.cache_misses, Some(2));
    assert_eq!(
        s.message.as_deref(),
        Some("succeeded: 3 stage(s), 1 from cache, 2 computed")
    );
}

#[test]
fn running_out_of_retries_reports_the_engine_error() {
    let failed = r#"{"version":1,"state":"failed","error":"plan execution failed: stage 'split_train_eval' failed: bad ratio","stages":{"split_train_eval":"failed"}}"#;
    let s = plan_status(
        &plan(),
        Some(&job_with(0, 0, 3, Some(("Failed", "BackoffLimitExceeded")))),
        &[
            pod("demo-run-a", 0, "Failed", Some((1, "Error", failed))),
            pod("demo-run-b", 10, "Failed", Some((1, "Error", failed))),
            pod("demo-run-c", 20, "Failed", Some((1, "Error", failed))),
        ],
    );
    assert_eq!(s.phase, PlanPhase::Failed);
    assert_eq!(s.attempts, 3);
    let m = s.message.unwrap();
    assert!(m.starts_with("out of retries: "), "{m}");
    assert!(m.contains("bad ratio"), "{m}");
    assert_eq!(s.stages["split_train_eval"], "failed");
}

#[test]
fn a_timeout_is_named_as_such() {
    let s = plan_status(
        &plan(),
        Some(&job_with(0, 0, 1, Some(("Failed", "DeadlineExceeded")))),
        &[],
    );
    assert_eq!(s.phase, PlanPhase::Failed);
    assert_eq!(s.message.as_deref(), Some("timeoutSeconds exceeded"));
}

#[test]
fn a_retry_after_an_oom_kill_names_the_previous_pod_and_why() {
    // The kubelet substitutes the log tail when the engine wrote no summary.
    let s = plan_status(
        &plan(),
        Some(&job_with(1, 0, 1, None)),
        &[
            pod(
                "demo-run-a",
                0,
                "Failed",
                Some((137, "OOMKilled", "loading…\nkilled")),
            ),
            pod("demo-run-b", 30, "Running", None),
        ],
    );
    assert_eq!(s.phase, PlanPhase::Running);
    assert_eq!(s.attempts, 2);
    let m = s.message.unwrap();
    assert!(
        m.starts_with("attempt 2, running; previous pod demo-run-a failed: "),
        "{m}"
    );
    assert!(m.contains("OOMKilled") && m.contains("exit 137"), "{m}");
}

#[test]
fn a_drained_pod_is_reported_as_a_disruption() {
    let mut drained = pod("demo-run-a", 0, "Failed", None);
    let st = drained.status.as_mut().unwrap();
    st.reason = Some("Evicted".into());
    st.conditions = Some(vec![PodCondition {
        type_: "DisruptionTarget".into(),
        status: "True".into(),
        ..Default::default()
    }]);
    let s = plan_status(
        &plan(),
        Some(&job_with(1, 0, 0, None)),
        &[drained, pod("demo-run-b", 30, "Pending", None)],
    );
    assert_eq!(s.phase, PlanPhase::Pending);
    let m = s.message.unwrap();
    assert!(
        m.contains("disrupted (not counted against maxRetries)"),
        "{m}"
    );
    assert!(m.contains("Evicted"), "{m}");
}

#[test]
fn only_a_version_1_summary_is_trusted() {
    assert!(parse_run_summary(OK_SUMMARY).is_some());
    assert!(parse_run_summary(r#"{"version":2,"state":"succeeded"}"#).is_none());
    assert!(parse_run_summary("Error: plan execution failed").is_none());
}

// ── pools ──────────────────────────────────────────────────────────────────

fn pool(replicas: u32, res: PoolResources, cache: CacheConfig) -> BlutWorkerPool {
    let mut p = BlutWorkerPool::new(
        "gpu",
        BlutWorkerPoolSpec {
            replicas,
            image: "ghcr.io/example/blut:1".into(),
            command: None,
            coordinator_pubkey: None,
            cache,
            resources: res,
        },
    );
    p.metadata.namespace = Some("blut".into());
    p.metadata.uid = Some("uid-pool".into());
    p
}

#[test]
fn pool_statefulset_is_a_no_scheduler_worker_on_udp() {
    let p = pool(
        3,
        PoolResources {
            cpu_cores: 4,
            memory_gib: 16,
            gpus: 2,
        },
        CacheConfig::None {},
    );
    let sts = build_pool_statefulset(&p);
    let spec = sts.spec.unwrap();
    assert_eq!(spec.replicas, Some(3));
    assert_eq!(spec.service_name.as_deref(), Some("gpu-workers"));
    let pod = spec.template.spec.unwrap();
    let c = &pod.containers[0];
    let args = c.args.as_ref().unwrap();
    assert!(args.contains(&"--no-scheduler".to_string()));
    assert_eq!(
        c.ports.as_ref().unwrap()[0].protocol.as_deref(),
        Some("UDP"),
        "the mesh is QUIC"
    );
    assert_eq!(env_value(c, "BLUT_NO_CONTAIN"), Some("1"));
    // Regression: `p2p node` exits at once without an identity key.
    let init = &pod.init_containers.as_ref().unwrap()[0];
    assert_eq!(
        init.args.as_deref().unwrap(),
        ["p2p", "keys", "generate", "--force"]
    );
    assert_eq!(
        init.volume_mounts, c.volume_mounts,
        "the key lands where the node reads it"
    );
    assert_eq!(env_value(init, "HOME"), env_value(c, "HOME"));
    let limits = c.resources.as_ref().unwrap().limits.as_ref().unwrap();
    assert_eq!(limits.get("nvidia.com/gpu").unwrap().0, "2");
    assert_eq!(limits.get("memory").unwrap().0, "16Gi");
}

#[test]
fn pool_service_is_headless_udp() {
    let svc = build_pool_service(&pool(1, PoolResources::default(), CacheConfig::None {}));
    let spec = svc.spec.unwrap();
    assert_eq!(spec.cluster_ip.as_deref(), Some("None"));
    assert_eq!(spec.ports.unwrap()[0].protocol.as_deref(), Some("UDP"));
    assert_eq!(svc.metadata.name.as_deref(), Some("gpu-workers"));
}

#[test]
fn a_pool_pvc_becomes_the_shared_store() {
    let sts = build_pool_statefulset(&pool(
        1,
        PoolResources::default(),
        CacheConfig::Pvc {
            claim_name: "rwx".into(),
        },
    ));
    let pod = sts.spec.unwrap().template.spec.unwrap();
    let args = pod.containers[0].args.as_ref().unwrap();
    let i = args.iter().position(|a| a == "--shared-store").unwrap();
    assert_eq!(args[i + 1], CACHE_MOUNT_PATH);
    assert!(pod.volumes.unwrap().iter().any(|v| {
        v.persistent_volume_claim
            .as_ref()
            .map(|c| c.claim_name.as_str())
            == Some("rwx")
    }));
}

#[test]
fn no_gpu_omits_the_gpu_request() {
    let sts = build_pool_statefulset(&pool(
        1,
        PoolResources {
            cpu_cores: 1,
            memory_gib: 2,
            gpus: 0,
        },
        CacheConfig::None {},
    ));
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

#[test]
fn the_committed_crd_manifest_matches_the_types() {
    let committed = include_str!("../../deploy/crds.yaml");
    assert_eq!(
        committed,
        crate::crds_yaml().unwrap(),
        "deploy/crds.yaml is stale: run `cargo run -- crds > deploy/crds.yaml` in crates/blut-operator"
    );
}

#[test]
fn a_vanished_pod_is_still_reported_and_counted() {
    // Reconcile 1: the first pod is being deleted.
    let mut deleting = pod("demo-run-a", 0, "Running", None);
    deleting.metadata.deletion_timestamp = deleting.metadata.creation_timestamp.clone();
    let mut p = plan();
    let s1 = plan_status(&p, Some(&job_with(1, 0, 0, None)), &[deleting]);
    assert_eq!(s1.attempts, 1);
    assert_eq!(s1.pod.as_deref(), Some("demo-run-a"));
    assert_eq!(
        s1.last_pod_failure.as_deref(),
        Some("pod demo-run-a was deleted")
    );
    // Reconcile 2: it is gone from the API and its replacement runs. The Job
    // counted nothing yet, but the status remembers both.
    p.status = Some(s1);
    let s2 = plan_status(
        &p,
        Some(&job_with(1, 0, 0, None)),
        &[pod("demo-run-b", 30, "Running", None)],
    );
    assert_eq!(s2.attempts, 2);
    assert_eq!(s2.pod.as_deref(), Some("demo-run-b"));
    assert_eq!(
        s2.message.as_deref(),
        Some("attempt 2, running; previous pod demo-run-a was deleted")
    );
    // Reconcile 3 (operator restarted): the same pod is not counted again.
    p.status = Some(s2);
    let s3 = plan_status(
        &p,
        Some(&job_with(1, 0, 0, None)),
        &[pod("demo-run-b", 30, "Running", None)],
    );
    assert_eq!(s3.attempts, 2);
}

#[test]
fn the_status_apply_body_names_the_kind_and_drops_nulls() {
    let status = BlutPlanStatus {
        phase: PlanPhase::Running,
        message: None,
        ..Default::default()
    };
    let body = status_apply_body::<BlutPlan>(&status);
    assert_eq!(body["apiVersion"], "blut.lamquant.dev/v1alpha1");
    assert_eq!(body["kind"], "BlutPlan");
    assert_eq!(body["status"]["phase"], "Running");
    assert!(body["status"].get("message").is_none(), "{body}");
}

#[test]
fn a_pool_coordinator_key_makes_workers_run_real_stages() {
    let mut p = pool(1, PoolResources::default(), CacheConfig::None {});
    assert!(
        !build_pool_statefulset(&p)
            .spec
            .unwrap()
            .template
            .spec
            .unwrap()
            .containers[0]
            .args
            .as_ref()
            .unwrap()
            .contains(&"--coordinator-pubkey".to_string())
    );
    p.spec.coordinator_pubkey = Some("ab01".into());
    let sts = build_pool_statefulset(&p);
    let args = sts.spec.unwrap().template.spec.unwrap().containers[0]
        .args
        .clone()
        .unwrap();
    let i = args
        .iter()
        .position(|a| a == "--coordinator-pubkey")
        .unwrap();
    assert_eq!(args[i + 1], "ab01");
}

// ── named recipes (what `recipe run --launcher k8s` submits) ──────────────

fn named(name: &str, args: &str) -> PlanSource {
    PlanSource::Recipe {
        name: name.into(),
        args: args.into(),
    }
}

#[test]
fn a_named_recipe_runs_recipe_run_without_a_plan_volume() {
    let p = plan_with(spec(named("split_dataset", r#"{"eval_ratio":0.25}"#)));
    assert_eq!(validate_plan(&p), Ok(()));
    assert!(build_plan_configmap(&p).is_none(), "nothing to copy");
    let job = build_plan_job(&p);
    let c = container(&job);
    assert_eq!(
        c.args.as_deref().unwrap(),
        [
            "recipe",
            "run",
            "split_dataset",
            "--args",
            r#"{"eval_ratio":0.25}"#
        ]
    );
    let pod = job.spec.as_ref().unwrap().template.spec.as_ref().unwrap();
    assert!(
        pod.volumes
            .as_ref()
            .unwrap()
            .iter()
            .all(|v| v.name != "plan"),
        "no ConfigMap volume for a named recipe"
    );
    assert!(
        c.volume_mounts
            .as_ref()
            .unwrap()
            .iter()
            .all(|m| m.name != "plan")
    );
    assert_eq!(
        env_value(c, "BLUT_RUN_SUMMARY"),
        Some(TERMINATION_MESSAGE_PATH)
    );
}

#[test]
fn a_named_recipe_with_a_bad_name_or_args_is_refused() {
    for (source, needle) in [
        (named("", "{}"), "planSource.recipe.name"),
        (named("a b", "{}"), "planSource.recipe.name"),
        (named("../x", "{}"), "planSource.recipe.name"),
        (named("ok", "[1]"), "must be a JSON object"),
        (named("ok", "{not json"), "must be a JSON object"),
    ] {
        let err = validate_plan(&plan_with(spec(source))).expect_err(needle);
        assert!(err.contains(needle), "{err:?} should mention {needle:?}");
    }
    let both = BlutPlanSpec {
        args: Some("{}".into()),
        ..spec(named("ok", "{}"))
    };
    let err = validate_plan(&plan_with(both)).unwrap_err();
    assert!(err.contains("planSource.recipe.args"), "{err}");
}

#[test]
fn a_named_recipe_defaults_to_empty_args() {
    let json = serde_json::json!({
        "planSource": { "recipe": { "name": "train" } },
        "cookbookImage": "img:1"
    });
    let s: BlutPlanSpec = serde_json::from_value(json).unwrap();
    match s.plan_source {
        PlanSource::Recipe { name, args } => {
            assert_eq!(name, "train");
            assert_eq!(args, "{}");
        }
        other => panic!("{other:?}"),
    }
}
