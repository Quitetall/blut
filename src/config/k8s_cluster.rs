// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `--launcher k8s`: submit a run to a Kubernetes cluster as a `BlutPlan` and
//! follow it through `blut-operator` (ADR 0067 T4.5).
//!
//! The engine has no Kubernetes dependency, here as elsewhere: every call is
//! a `kubectl` subprocess, and the manifest is plain JSON. What runs in the
//! cluster is the same cookbook command the user typed, executed by the
//! operator in a pod (`recipe run <name> --args <json>`), so the cluster run
//! and a local run compile the same plan.
//!
//! Everything the submission needs comes from the environment, like the
//! Slurm and Ray launchers (`BLUT_SLURM_*`, `RAY_ADDRESS`):
//!
//! | variable | meaning |
//! |---|---|
//! | `BLUT_K8S_IMAGE` | cookbook image the plan pod runs (required) |
//! | `BLUT_K8S_COMMAND` | cookbook binary inside the image (default `blut`) |
//! | `BLUT_K8S_NAMESPACE` | namespace (default: the kubectl context's) |
//! | `BLUT_K8S_CONTEXT` | kubectl context (default: the current one) |
//! | `BLUT_K8S_STORE` | `s3://bucket[/prefix]` the pod writes stage outputs to |
//! | `BLUT_K8S_STORE_SECRET` | Secret holding the store's `AWS_*` keys |
//! | `BLUT_K8S_CPU_MILLIS`, `BLUT_K8S_MEMORY_MIB` | pod CPU and memory (both or neither) |
//! | `BLUT_K8S_GPUS` | `nvidia.com/gpu` per pod |
//! | `BLUT_K8S_MAX_RETRIES`, `BLUT_K8S_TIMEOUT_SECS` | the plan's retry and wall-clock limits |
//! | `BLUT_K8S_PICKUP_SECS` | how long to wait for the operator to report (default 120) |
//! | `BLUT_KUBECTL` | the kubectl binary (default `kubectl` on PATH) |
//!
//! Credentials never enter the manifest: the plan names the Secret, and the
//! kubelet resolves its keys inside the pod.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::{Result, TrainError};

/// API group and version of the operator's CRDs.
pub(crate) const API_VERSION: &str = "blut.lamquant.dev/v1alpha1";
/// The CRD `kubectl api-resources` must list for a submission to work.
pub(crate) const PLAN_RESOURCE: &str = "blutplans.blut.lamquant.dev";
/// Where an install is described, named in every refusal.
pub(crate) const INSTALL_HINT: &str = "install blut-operator with `kubectl apply -f \
     crates/blut-operator/deploy/crds.yaml -f crates/blut-operator/deploy/operator.yaml \
     -f crates/blut-operator/deploy/rbac.yaml` (docs/OPERATOR.md), or submit a BlutPlan \
     yourself with `kubectl apply`";
/// File in the local job directory naming the submitted plan, so `blut
/// cancel` can delete it even when the following process is gone.
pub(crate) const MARKER_FILE: &str = "k8s-plan.json";

/// Longest plan name: the operator names the Job `<plan>-run`, and Job names
/// become 63-character label values.
const MAX_PLAN_NAME: usize = 59;

/// Submission settings, read from `BLUT_K8S_*`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ClusterSettings {
    pub image: String,
    pub command: Option<String>,
    pub namespace: Option<String>,
    pub context: Option<String>,
    pub store_url: Option<String>,
    pub store_secret: Option<String>,
    pub resources: Option<(u32, u32)>,
    pub gpus: u32,
    pub max_retries: Option<u32>,
    pub timeout_secs: Option<u32>,
    pub pickup: Duration,
    pub kubectl: PathBuf,
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

impl ClusterSettings {
    /// Read the settings from the process environment.
    pub(crate) fn from_env() -> Result<Self> {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// Read the settings through `get`; an empty value counts as unset.
    pub(crate) fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let get = |k: &str| {
            get(k)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let number = |k: &str| -> Result<Option<u32>> {
            match get(k) {
                None => Ok(None),
                Some(v) => v.parse::<u32>().map(Some).map_err(|_| {
                    TrainError::other(format!("{k}={v:?} is not a non-negative integer"))
                }),
            }
        };
        let image = get("BLUT_K8S_IMAGE").ok_or_else(|| {
            TrainError::other(
                "--launcher k8s needs BLUT_K8S_IMAGE: the cookbook image the plan's pod \
                 runs, built with the engine's `s3` feature if BLUT_K8S_STORE is s3:// \
                 (docker/Dockerfile.cookbook; docs/OPERATOR.md)",
            )
        })?;
        let command = get("BLUT_K8S_COMMAND");
        if let Some(c) = &command
            && c.chars().any(char::is_whitespace)
        {
            return Err(TrainError::other(format!(
                "BLUT_K8S_COMMAND={c:?} must be one executable, without arguments"
            )));
        }
        let namespace = get("BLUT_K8S_NAMESPACE");
        if let Some(ns) = &namespace
            && !is_dns_label(ns)
        {
            return Err(TrainError::other(format!(
                "BLUT_K8S_NAMESPACE={ns:?} is not a valid namespace name"
            )));
        }
        let store_url = get("BLUT_K8S_STORE");
        if let Some(url) = &store_url {
            match url.strip_prefix("s3://") {
                Some(rest) => {
                    let bucket = rest.split('/').next().unwrap_or("");
                    if bucket.is_empty()
                        || bucket.contains('@')
                        || bucket.contains(':')
                        || url.contains('?')
                        || url.contains('#')
                    {
                        return Err(TrainError::other(
                            "BLUT_K8S_STORE must be s3://bucket[/prefix]: credentials and \
                             the endpoint belong in the Secret named by \
                             BLUT_K8S_STORE_SECRET, never in the URL",
                        ));
                    }
                }
                None if url.starts_with('/') => {}
                None => {
                    return Err(TrainError::other(
                        "BLUT_K8S_STORE must be s3://bucket[/prefix] or an absolute path \
                         inside the pod",
                    ));
                }
            }
        }
        let store_secret = get("BLUT_K8S_STORE_SECRET");
        if let Some(secret) = &store_secret {
            if !is_dns_subdomain(secret) {
                return Err(TrainError::other(format!(
                    "BLUT_K8S_STORE_SECRET={secret:?} is not a valid Secret name"
                )));
            }
            if store_url.is_none() {
                return Err(TrainError::other(
                    "BLUT_K8S_STORE_SECRET is set but BLUT_K8S_STORE is not; the Secret \
                     only carries the store's credentials",
                ));
            }
        }
        let resources = match (
            number("BLUT_K8S_CPU_MILLIS")?,
            number("BLUT_K8S_MEMORY_MIB")?,
        ) {
            (None, None) => None,
            (Some(cpu), Some(mem)) if cpu >= 1 && mem >= 16 => Some((cpu, mem)),
            (Some(_), Some(_)) => {
                return Err(TrainError::other(
                    "BLUT_K8S_CPU_MILLIS must be at least 1 and BLUT_K8S_MEMORY_MIB at \
                     least 16",
                ));
            }
            _ => {
                return Err(TrainError::other(
                    "set both BLUT_K8S_CPU_MILLIS and BLUT_K8S_MEMORY_MIB, or neither",
                ));
            }
        };
        let max_retries = number("BLUT_K8S_MAX_RETRIES")?;
        if max_retries.is_some_and(|n| n > 20) {
            return Err(TrainError::other("BLUT_K8S_MAX_RETRIES must be at most 20"));
        }
        let timeout_secs = number("BLUT_K8S_TIMEOUT_SECS")?;
        if timeout_secs == Some(0) {
            return Err(TrainError::other(
                "BLUT_K8S_TIMEOUT_SECS must be positive when set",
            ));
        }
        let pickup = match number("BLUT_K8S_PICKUP_SECS")? {
            None => Duration::from_secs(120),
            Some(0) => {
                return Err(TrainError::other("BLUT_K8S_PICKUP_SECS must be positive"));
            }
            Some(n) => Duration::from_secs(u64::from(n)),
        };
        Ok(Self {
            image,
            command,
            namespace,
            context: get("BLUT_K8S_CONTEXT"),
            store_url,
            store_secret,
            resources,
            gpus: number("BLUT_K8S_GPUS")?.unwrap_or(0),
            max_retries,
            timeout_secs,
            pickup,
            kubectl: get("BLUT_KUBECTL")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("kubectl")),
        })
    }
}

/// A plan name for a recipe run: `<recipe>-<job id>`, lowercased, at most 59
/// characters, a valid DNS label. Unique per local job id.
pub(crate) fn plan_name(recipe: &str, job_id: &str) -> String {
    let clean = |s: &str| {
        let mut out = String::new();
        for c in s.chars() {
            let c = c.to_ascii_lowercase();
            if c.is_ascii_lowercase() || c.is_ascii_digit() {
                out.push(c);
            } else if !out.ends_with('-') {
                out.push('-');
            }
        }
        out.trim_matches('-').to_string()
    };
    let suffix = clean(job_id);
    let room = MAX_PLAN_NAME.saturating_sub(suffix.len() + 1);
    let mut prefix: String = clean(recipe).chars().take(room).collect();
    prefix = prefix.trim_matches('-').to_string();
    if prefix.is_empty() {
        prefix = "blut".into();
    }
    let name = format!("{prefix}-{suffix}");
    name.chars().take(MAX_PLAN_NAME).collect::<String>()
}

/// What the plan's pod runs.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ClusterSource {
    /// A recipe compiled into the cookbook binary: `recipe run <name> --args`.
    Recipe { name: String, args: Value },
    /// A recipe file (`.json`, `.toml`, `.star`) shipped inline:
    /// `recipe declare /blut/plan/<filename> --run [--args]`.
    File {
        filename: String,
        content: String,
        args: Option<String>,
    },
}

/// The `BlutPlan` manifest for `source`.
pub(crate) fn render_plan(
    settings: &ClusterSettings,
    name: &str,
    namespace: &str,
    source: &ClusterSource,
    job_id: &str,
) -> Value {
    let plan_source = match source {
        ClusterSource::Recipe { name, args } => {
            json!({ "recipe": { "name": name, "args": args.to_string() } })
        }
        ClusterSource::File {
            filename, content, ..
        } => json!({ "inline": { "filename": filename, "content": content } }),
    };
    let mut spec = json!({
        "planSource": plan_source,
        "cookbookImage": settings.image,
        "dataClassCeiling": "Internal",
    });
    if let ClusterSource::File {
        args: Some(args), ..
    } = source
    {
        spec["args"] = json!(args);
    }
    if let Some(command) = &settings.command {
        spec["command"] = json!(command);
    }
    if let Some(url) = &settings.store_url {
        let mut store = json!({ "url": url });
        if let Some(secret) = &settings.store_secret {
            store["credentialsSecret"] = json!(secret);
        }
        spec["store"] = store;
    }
    if let Some((cpu, mem)) = settings.resources {
        spec["resources"] = json!({ "cpuMillis": cpu, "memoryMib": mem });
    }
    if settings.gpus > 0 {
        spec["gpus"] = json!(settings.gpus);
    }
    if let Some(n) = settings.max_retries {
        spec["maxRetries"] = json!(n);
    }
    if let Some(n) = settings.timeout_secs {
        spec["timeoutSeconds"] = json!(n);
    }
    json!({
        "apiVersion": API_VERSION,
        "kind": "BlutPlan",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": { "app.kubernetes.io/created-by": "blut" },
            "annotations": { "blut.lamquant.dev/job": job_id },
        },
        "spec": spec,
    })
}

/// A recipe file the operator accepts inline: a plain name ending in `.json`,
/// `.toml` or `.star`, under the ConfigMap size limit.
pub(crate) fn check_inline_file(filename: &str, content: &str) -> Result<()> {
    let plain = !filename.is_empty()
        && !filename.starts_with('.')
        && filename
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    let ext = filename.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    if !plain || !matches!(ext, "json" | "toml" | "star") {
        return Err(TrainError::other(format!(
            "--launcher k8s ships the recipe file inline, and '{filename}' is not a plain \
             .json, .toml or .star file name; rename it"
        )));
    }
    if content.len() > 900 * 1024 {
        return Err(TrainError::other(format!(
            "--launcher k8s ships the recipe file inline in a ConfigMap (at most ~900 KiB), \
             and '{filename}' is {} bytes; put it in a ConfigMap yourself and submit a \
             BlutPlan with planSource.configMapRef",
            content.len()
        )));
    }
    Ok(())
}

/// What the follower reads from a plan's status.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ClusterStatus {
    pub phase: Option<String>,
    pub message: Option<String>,
    pub attempts: u64,
    pub stages: BTreeMap<String, String>,
    pub cache_hits: Option<u64>,
    pub cache_misses: Option<u64>,
    pub pod: Option<String>,
}

impl ClusterStatus {
    /// `Succeeded` or `Failed`: the plan will not change again.
    pub(crate) fn is_terminal(&self) -> bool {
        matches!(self.phase.as_deref(), Some("Succeeded" | "Failed"))
    }
}

/// Read the status out of a `kubectl get blutplan -o json` document.
pub(crate) fn parse_status(plan: &Value) -> ClusterStatus {
    let s = &plan["status"];
    let text = |k: &str| s[k].as_str().map(str::to_string);
    ClusterStatus {
        phase: text("phase"),
        message: text("message"),
        attempts: s["attempts"].as_u64().unwrap_or(0),
        stages: s["stages"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        cache_hits: s["cacheHits"].as_u64(),
        cache_misses: s["cacheMisses"].as_u64(),
        pod: text("pod"),
    }
}

/// `kubectl`, with the context the settings name and a request timeout so no
/// call can hang.
#[derive(Clone, Debug)]
pub(crate) struct Kubectl {
    bin: PathBuf,
    context: Option<String>,
}

struct Ran {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn first_line(s: &str) -> String {
    s.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("")
        .to_string()
}

impl Kubectl {
    pub(crate) fn new(bin: PathBuf, context: Option<String>) -> Self {
        Self { bin, context }
    }

    pub(crate) fn from_settings(settings: &ClusterSettings) -> Self {
        Self::new(settings.kubectl.clone(), settings.context.clone())
    }

    fn run(&self, args: &[&str], stdin: Option<&str>) -> Result<Ran> {
        let mut cmd = Command::new(&self.bin);
        if let Some(ctx) = &self.context {
            cmd.arg("--context").arg(ctx);
        }
        cmd.arg("--request-timeout=30s").args(args);
        cmd.stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| {
            TrainError::other(format!(
                "--launcher k8s submits a BlutPlan with kubectl, and `{}` could not be \
                 run: {e}. Install kubectl or point BLUT_KUBECTL at it",
                self.bin.display()
            ))
        })?;
        let write = match (stdin, child.stdin.take()) {
            (Some(body), Some(mut pipe)) => pipe.write_all(body.as_bytes()),
            _ => Ok(()),
        };
        let out = child
            .wait_with_output()
            .map_err(|e| TrainError::other(format!("kubectl: {e}")))?;
        write.map_err(|e| TrainError::other(format!("kubectl stdin: {e}")))?;
        Ok(Ran {
            ok: out.status.success(),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        })
    }

    /// Refuse before anything is submitted unless kubectl has a context, the
    /// cluster answers, and the operator's CRD is installed.
    pub(crate) fn preflight(&self) -> Result<()> {
        let ctx = self.run(&["config", "current-context"], None)?;
        if !ctx.ok && self.context.is_none() {
            return Err(TrainError::other(format!(
                "--launcher k8s: kubectl has no current context ({}). Point KUBECONFIG at \
                 the cluster that runs blut-operator, or set BLUT_K8S_CONTEXT. Nothing \
                 was submitted and nothing ran locally",
                first_line(&ctx.stderr)
            )));
        }
        let found = self.run(
            &[
                "api-resources",
                "--api-group=blut.lamquant.dev",
                "-o",
                "name",
            ],
            None,
        )?;
        if !found.ok {
            return Err(TrainError::other(format!(
                "--launcher k8s: the cluster did not answer kubectl ({}). Nothing was \
                 submitted and nothing ran locally",
                first_line(&found.stderr)
            )));
        }
        if !found.stdout.lines().any(|l| l.trim() == PLAN_RESOURCE) {
            return Err(TrainError::other(format!(
                "--launcher k8s: the cluster has no BlutPlan resource, so blut-operator is \
                 not installed there; {INSTALL_HINT}. Nothing was submitted and nothing \
                 ran locally"
            )));
        }
        Ok(())
    }

    /// The namespace to use: the configured one, else the context's, else
    /// `default`.
    pub(crate) fn namespace(&self, configured: Option<&str>) -> Result<String> {
        if let Some(ns) = configured {
            return Ok(ns.to_string());
        }
        let out = self.run(
            &["config", "view", "--minify", "-o", "jsonpath={..namespace}"],
            None,
        )?;
        let ns = out.stdout.trim();
        Ok(if out.ok && is_dns_label(ns) {
            ns.to_string()
        } else {
            "default".to_string()
        })
    }

    /// Create the plan. `create`, not `apply`: an existing plan of the same
    /// name is someone else's run and is never overwritten.
    pub(crate) fn create(&self, manifest: &Value) -> Result<()> {
        let body = manifest.to_string();
        let out = self.run(&["create", "-f", "-", "-o", "name"], Some(&body))?;
        if out.ok {
            Ok(())
        } else {
            Err(TrainError::other(format!(
                "kubectl create BlutPlan failed: {}",
                first_line(&out.stderr)
            )))
        }
    }

    /// The plan as JSON, or `None` once it is gone.
    pub(crate) fn get(&self, namespace: &str, name: &str) -> Result<Option<Value>> {
        let out = self.run(
            &[
                "get",
                "blutplan",
                name,
                "-n",
                namespace,
                "-o",
                "json",
                "--ignore-not-found",
            ],
            None,
        )?;
        if !out.ok {
            return Err(TrainError::other(format!(
                "kubectl get blutplan {namespace}/{name}: {}",
                first_line(&out.stderr)
            )));
        }
        if out.stdout.trim().is_empty() {
            return Ok(None);
        }
        serde_json::from_str(&out.stdout)
            .map(Some)
            .map_err(|e| TrainError::other(format!("kubectl get blutplan: bad JSON: {e}")))
    }

    /// Delete the plan; owner references remove its Job and pods.
    pub(crate) fn delete(&self, namespace: &str, name: &str) -> Result<()> {
        let out = self.run(
            &[
                "delete",
                "blutplan",
                name,
                "-n",
                namespace,
                "--ignore-not-found",
                "--wait=false",
            ],
            None,
        )?;
        if out.ok {
            Ok(())
        } else {
            Err(TrainError::other(format!(
                "kubectl delete blutplan {namespace}/{name}: {}",
                first_line(&out.stderr)
            )))
        }
    }
}

/// What the local job directory records about a submitted plan.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PlanMarker {
    pub namespace: String,
    pub name: String,
    #[serde(default)]
    pub context: Option<String>,
    pub kubectl: PathBuf,
}

pub(crate) fn write_marker(job_dir: &Path, marker: &PlanMarker) -> Result<()> {
    let path = job_dir.join(MARKER_FILE);
    let body = serde_json::to_vec_pretty(marker)
        .map_err(|e| TrainError::other(format!("serialize {MARKER_FILE}: {e}")))?;
    std::fs::write(&path, body).map_err(|source| TrainError::Io { path, source })
}

pub(crate) fn read_marker(job_dir: &Path) -> Result<Option<PlanMarker>> {
    let path = job_dir.join(MARKER_FILE);
    match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| TrainError::other(format!("{}: {e}", path.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(TrainError::Io { path, source }),
    }
}

/// Delete the plan a job submitted, if it submitted one. Returns whether a
/// plan was recorded.
pub(crate) fn cancel_recorded(job_dir: &Path) -> Result<bool> {
    let Some(m) = read_marker(job_dir)? else {
        return Ok(false);
    };
    Kubectl::new(m.kubectl.clone(), m.context.clone()).delete(&m.namespace, &m.name)?;
    Ok(true)
}

/// The refusal for a surface that takes `--launcher` but cannot submit to a
/// cluster. Running locally instead would claim a placement that did not
/// happen.
pub(crate) fn refuse_surface(surface: &str) -> TrainError {
    TrainError::other(format!(
        "--launcher k8s is supported by `recipe run` (one recipe, no sweep) and \
         `recipe declare <file> --run` only: they submit a BlutPlan to blut-operator \
         and follow it. {surface} cannot be submitted that way; {INSTALL_HINT}. \
         Nothing ran locally"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: BTreeMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| map.get(k).cloned()
    }

    fn settings() -> ClusterSettings {
        ClusterSettings::from_lookup(env(&[
            ("BLUT_K8S_IMAGE", "img:1"),
            ("BLUT_K8S_STORE", "s3://bucket/runs"),
            ("BLUT_K8S_STORE_SECRET", "store-creds"),
            ("BLUT_K8S_CPU_MILLIS", "500"),
            ("BLUT_K8S_MEMORY_MIB", "512"),
            ("BLUT_K8S_MAX_RETRIES", "1"),
            ("BLUT_K8S_TIMEOUT_SECS", "600"),
        ]))
        .unwrap()
    }

    #[test]
    fn the_image_is_required_and_named_in_the_refusal() {
        let err = ClusterSettings::from_lookup(env(&[])).unwrap_err();
        assert!(err.to_string().contains("BLUT_K8S_IMAGE"), "{err}");
    }

    #[test]
    fn bad_settings_are_refused_with_the_variable_named() {
        for (pairs, needle) in [
            (
                vec![("BLUT_K8S_STORE", "s3://key:secret@b/x")],
                "never in the URL",
            ),
            (vec![("BLUT_K8S_STORE", "relative")], "absolute path"),
            (
                vec![("BLUT_K8S_STORE_SECRET", "s")],
                "BLUT_K8S_STORE is not",
            ),
            (vec![("BLUT_K8S_CPU_MILLIS", "500")], "or neither"),
            (vec![("BLUT_K8S_MAX_RETRIES", "50")], "at most 20"),
            (vec![("BLUT_K8S_NAMESPACE", "Bad_NS")], "namespace"),
            (vec![("BLUT_K8S_GPUS", "two")], "BLUT_K8S_GPUS"),
            (vec![("BLUT_K8S_COMMAND", "blut --x")], "one executable"),
        ] {
            let mut all = vec![("BLUT_K8S_IMAGE", "img:1")];
            all.extend(pairs);
            let err = ClusterSettings::from_lookup(env(&all))
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{err:?} should mention {needle:?}");
            assert!(!err.contains("secret@"), "a credential was echoed: {err}");
        }
    }

    #[test]
    fn the_manifest_names_the_recipe_and_only_references_the_secret() {
        let m = render_plan(
            &settings(),
            "train-20261029-1",
            "blut",
            &ClusterSource::Recipe {
                name: "train".into(),
                args: json!({"epochs": 2}),
            },
            "20261029-1",
        );
        assert_eq!(m["kind"], "BlutPlan");
        assert_eq!(m["apiVersion"], API_VERSION);
        assert_eq!(m["metadata"]["namespace"], "blut");
        let spec = &m["spec"];
        assert_eq!(spec["planSource"]["recipe"]["name"], "train");
        assert_eq!(spec["planSource"]["recipe"]["args"], r#"{"epochs":2}"#);
        assert_eq!(spec["cookbookImage"], "img:1");
        assert_eq!(spec["dataClassCeiling"], "Internal");
        assert_eq!(spec["store"]["url"], "s3://bucket/runs");
        assert_eq!(spec["store"]["credentialsSecret"], "store-creds");
        assert_eq!(spec["resources"]["cpuMillis"], 500);
        assert_eq!(spec["maxRetries"], 1);
        assert_eq!(spec["timeoutSeconds"], 600);
        assert!(spec.get("gpus").is_none(), "gpus only when asked for");
        assert!(spec.get("command").is_none());
        let text = m.to_string();
        assert!(!text.contains("AWS_"), "no credential keys in the manifest");
    }

    #[test]
    fn a_recipe_file_travels_inline_with_its_name_and_star_args() {
        let m = render_plan(
            &settings(),
            "fan-1",
            "blut",
            &ClusterSource::File {
                filename: "fan.star".into(),
                content: "def build(a): pass".into(),
                args: Some(r#"{"n":2}"#.into()),
            },
            "1",
        );
        assert_eq!(m["spec"]["planSource"]["inline"]["filename"], "fan.star");
        assert_eq!(
            m["spec"]["planSource"]["inline"]["content"],
            "def build(a): pass"
        );
        assert_eq!(m["spec"]["args"], r#"{"n":2}"#);
        check_inline_file("fan.star", "x").unwrap();
        for bad in ["../x.toml", "x.yaml", ".hidden.json", "a b.toml"] {
            assert!(check_inline_file(bad, "x").is_err(), "{bad}");
        }
        let big = "x".repeat(901 * 1024);
        assert!(
            check_inline_file("big.json", &big)
                .unwrap_err()
                .to_string()
                .contains("configMapRef")
        );
    }

    #[test]
    fn plan_names_are_dns_labels_that_fit_the_job_name() {
        let n = plan_name("split_train_eval", "20261029-140501-123456789");
        assert_eq!(n, "split-train-eval-20261029-140501-123456789");
        assert!(is_dns_label(&n));
        let long = plan_name(&"x".repeat(200), "20261029-140501-123456789");
        assert!(long.len() <= MAX_PLAN_NAME, "{}", long.len());
        assert!(is_dns_label(&long), "{long}");
        assert!(long.ends_with("20261029-140501-123456789"));
        assert!(plan_name("__", "1").starts_with("blut-"));
    }

    #[test]
    fn status_is_read_from_the_plan_document() {
        let doc = json!({"status": {
            "phase": "Succeeded", "message": "succeeded: 3 stage(s)", "attempts": 2,
            "stages": {"a": "ok", "b": "cached"}, "cacheHits": 1, "cacheMisses": 1,
            "pod": "p-1"}});
        let s = parse_status(&doc);
        assert!(s.is_terminal());
        assert_eq!(s.stages["b"], "cached");
        assert_eq!(s.attempts, 2);
        assert_eq!(s.cache_hits, Some(1));
        let fresh = parse_status(&json!({"spec": {}}));
        assert_eq!(fresh.phase, None);
        assert!(!fresh.is_terminal());
    }

    /// A fake kubectl: a shell script that answers by subcommand, so the
    /// refusals are tested without a cluster.
    #[cfg(unix)]
    fn fake_kubectl(dir: &Path, script: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("kubectl");
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn preflight_refuses_without_a_context_and_names_kubeconfig() {
        let td = tempfile::tempdir().unwrap();
        let bin = fake_kubectl(
            td.path(),
            r#"case "$*" in *current-context*) echo "error: current-context is not set" >&2; exit 1;; esac; exit 0"#,
        );
        let err = Kubectl::new(bin, None).preflight().unwrap_err().to_string();
        assert!(err.contains("KUBECONFIG"), "{err}");
        assert!(err.contains("nothing ran locally"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn preflight_refuses_without_the_crd_and_names_the_operator() {
        let td = tempfile::tempdir().unwrap();
        let bin = fake_kubectl(
            td.path(),
            r#"case "$*" in *current-context*) echo kind-x;; *api-resources*) echo "";; esac; exit 0"#,
        );
        let err = Kubectl::new(bin, None).preflight().unwrap_err().to_string();
        assert!(err.contains("blut-operator is not installed"), "{err}");
        assert!(err.contains("kubectl apply"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn preflight_passes_when_the_crd_is_listed() {
        let td = tempfile::tempdir().unwrap();
        let bin = fake_kubectl(
            td.path(),
            r#"case "$*" in *current-context*) echo kind-x;; *api-resources*) printf 'blutplans.blut.lamquant.dev\nblutworkerpools.blut.lamquant.dev\n';; esac; exit 0"#,
        );
        Kubectl::new(bin, None).preflight().unwrap();
    }

    #[test]
    fn a_missing_kubectl_is_named() {
        let err = Kubectl::new(PathBuf::from("/nonexistent/kubectl"), None)
            .preflight()
            .unwrap_err()
            .to_string();
        assert!(err.contains("BLUT_KUBECTL"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn cancel_deletes_the_recorded_plan() {
        let td = tempfile::tempdir().unwrap();
        let log = td.path().join("calls");
        let bin = fake_kubectl(td.path(), &format!("echo \"$*\" >> {}", log.display()));
        let job = td.path().join("job");
        std::fs::create_dir_all(&job).unwrap();
        assert!(!cancel_recorded(&job).unwrap(), "no marker, nothing to do");
        write_marker(
            &job,
            &PlanMarker {
                namespace: "blut".into(),
                name: "train-1".into(),
                context: Some("kind-x".into()),
                kubectl: bin,
            },
        )
        .unwrap();
        assert!(cancel_recorded(&job).unwrap());
        let calls = std::fs::read_to_string(&log).unwrap();
        assert!(
            calls.contains("--context kind-x")
                && calls.contains("delete blutplan train-1 -n blut --ignore-not-found"),
            "{calls}"
        );
    }
}
