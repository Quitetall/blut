// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `recipe run --launcher k8s`: submit the recipe to a cluster as a
//! `BlutPlan`, follow it through `blut-operator`, and exit with its result.
//!
//! Mounted as a child of the `cli` module and glob-imported back.
//!
//! Nothing here executes a stage locally. The recipe is compiled here only to
//! refuse bad args before anything is submitted; the run itself is the
//! operator's pod executing `recipe run <name> --args <json>` with the same
//! cookbook. Ctrl-C, SIGTERM and `blut cancel <job>` delete the plan.

use super::*;

use crate::config::k8s_cluster::{
    ClusterSettings, ClusterSource, ClusterStatus, Kubectl, PlanMarker, check_inline_file,
    parse_status, plan_name, render_plan, write_marker,
};

/// Flags of `recipe run` that a cluster run cannot honour yet. Each one that
/// is set is refused rather than silently dropped.
pub(super) struct K8sUnsupported {
    pub config_mode: bool,
    pub shared_cache: bool,
    pub no_cache: bool,
    pub sync_io: bool,
    pub tenant_is_default: bool,
    pub experiment: bool,
}

impl K8sUnsupported {
    fn refuse(&self) -> Result<()> {
        let mut named = Vec::new();
        if self.config_mode {
            named.push("config mode (--config-dir/--config-name/--set/--sweep)");
        }
        if self.shared_cache {
            named.push("--shared-cache (give the plan a store: BLUT_K8S_STORE)");
        }
        if self.no_cache {
            named.push("--no-cache");
        }
        if self.sync_io {
            named.push("--sync-io");
        }
        if !self.tenant_is_default {
            named.push("--tenant other than `default`");
        }
        if self.experiment {
            named.push("--experiment");
        }
        if named.is_empty() {
            return Ok(());
        }
        Err(anyhow!(
            "--launcher k8s cannot carry {} to the cluster; drop them or run locally. \
             Nothing was submitted and nothing ran locally",
            named.join(", ")
        ))
    }
}

/// How often the plan's status is read.
const POLL: Duration = Duration::from_secs(2);
/// Consecutive failed status reads before the follower gives up (the plan
/// keeps running; the error says how to see or delete it).
const MAX_READ_FAILURES: u32 = 30;

/// `recipe run <name> --launcher k8s`: submit the named recipe with
/// `args_json` and follow it to a terminal phase. With `dry_run`, validate
/// and print the manifest without touching a cluster.
pub(super) async fn run_recipe_on_k8s(
    reg: &crate::framework::Registry,
    name: &str,
    args_json: &str,
    dry_run: bool,
    unsupported: K8sUnsupported,
) -> Result<()> {
    unsupported.refuse()?;
    let settings = ClusterSettings::from_env().map_err(|e| anyhow!("{e}"))?;

    let raw: serde_json::Value = serde_json::from_str(args_json)
        .with_context(|| format!("parse --args as JSON: {args_json}"))?;
    if !raw.is_object() {
        return Err(anyhow!("--args must be a JSON object for --launcher k8s"));
    }
    refuse_registry_handles(&raw)?;
    let def = reg
        .find(name)
        .ok_or_else(|| anyhow!("recipe '{name}' not in catalog (see `blut recipe list`)"))?;
    validate_for_cluster(def, &raw)?;
    let source = ClusterSource::Recipe {
        name: name.to_string(),
        args: raw,
    };
    submit_and_follow(settings, name, source, dry_run).await
}

/// `recipe declare <file> --run --launcher k8s`: ship the recipe file inline
/// and follow the plan. The file is compiled here first, against this
/// binary's cookbooks, so a wrong stage name fails before submission.
pub(super) async fn run_file_on_k8s(
    reg: &crate::framework::Registry,
    path: &std::path::Path,
    star_args: &str,
    unsupported: K8sUnsupported,
) -> Result<()> {
    unsupported.refuse()?;
    let settings = ClusterSettings::from_env().map_err(|e| anyhow!("{e}"))?;
    let shown = path.to_string_lossy();
    if crate::registry_db::parse_pointer_uri(&shown).is_some() {
        return Err(anyhow!(
            "--launcher k8s: {shown} resolves against this machine's deployment registry, \
             which the cluster cannot see; export the PlanSpec to a .json file and submit \
             that. Nothing was submitted"
        ));
    }
    let filename = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("read recipe file {}", path.display()))?;
    check_inline_file(&filename, &content).map_err(|e| anyhow!("{e}"))?;
    let is_star = filename.ends_with(".star");
    let parsed_args: serde_json::Value = serde_json::from_str(star_args)
        .with_context(|| format!("parse --args as JSON: {star_args}"))?;
    refuse_registry_handles(&parsed_args)?;
    if !is_star && star_args.trim() != "{}" {
        return Err(anyhow!(
            "--args is only for .star recipes; '{filename}' carries its own args"
        ));
    }
    let tenant = crate::tenant::Tenant::default();
    compile_declared_recipe(
        reg,
        path,
        star_args,
        &tenant,
        crate::config::launcher::LaunchTarget::Local,
    )
    .map_err(|e| anyhow!("{e:#}"))?;
    let source = ClusterSource::File {
        filename: filename.clone(),
        content,
        args: is_star.then(|| star_args.to_string()),
    };
    let label = filename
        .rsplit_once('.')
        .map(|(stem, _)| stem.to_string())
        .unwrap_or(filename);
    submit_and_follow(settings, &label, source, false).await
}

/// Check a named recipe's args here, before anything reaches the cluster:
/// a misspelled key or a wrong type fails now. The recipe is NOT compiled
/// here: compiling can check that paths in the args exist, and those are
/// paths inside the pod. The pod compiles it, and a failure there is the
/// plan's failure, reported by the follower.
fn validate_for_cluster(
    def: &crate::recipes::recipe::RecipeDef,
    raw: &serde_json::Value,
) -> Result<()> {
    crate::recipes::recipe::validate_args_against_schema(def.name, &(def.args_schema_fn)(), raw)
        .map_err(|e| anyhow!("recipe '{}': {e}", def.name))
}

fn refuse_registry_handles(args: &serde_json::Value) -> Result<()> {
    if crate::registry_args::names_registry_handles(args) {
        return Err(anyhow!(
            "--launcher k8s: the args name registry handles (dataset://, model or \
             experiment references), which resolve against this machine's registries; \
             the cluster pod cannot see them. Pass the resolved values instead. Nothing \
             was submitted"
        ));
    }
    Ok(())
}

/// Preflight the cluster, record a local job, create the plan, follow it.
async fn submit_and_follow(
    settings: ClusterSettings,
    label: &str,
    source: ClusterSource,
    dry_run: bool,
) -> Result<()> {
    let job_id = crate::jobs::new_job_id();
    let plan = plan_name(label, &job_id);

    if dry_run {
        let ns = settings
            .namespace
            .as_deref()
            .unwrap_or("<context namespace>");
        let manifest = render_plan(&settings, &plan, ns, &source, &job_id);
        println!(
            "{}",
            serde_json::to_string_pretty(&manifest).unwrap_or_default()
        );
        eprintln!("[dry-run] '{label}' compiles; the BlutPlan above was not submitted.");
        return Ok(());
    }

    let kubectl = Kubectl::from_settings(&settings);
    {
        let k = kubectl.clone();
        tokio::task::spawn_blocking(move || k.preflight())
            .await
            .map_err(|e| anyhow!("{e}"))?
            .map_err(|e| anyhow!("{e}"))?;
    }
    let namespace = {
        let k = kubectl.clone();
        let configured = settings.namespace.clone();
        tokio::task::spawn_blocking(move || k.namespace(configured.as_deref()))
            .await
            .map_err(|e| anyhow!("{e}"))?
            .map_err(|e| anyhow!("{e}"))?
    };
    if settings.store_url.is_none() {
        eprintln!(
            "warning: BLUT_K8S_STORE is not set, so stage outputs stay in the plan's pod \
             and are gone when it is deleted"
        );
    }

    let manifest = render_plan(&settings, &plan, &namespace, &source, &job_id);
    let job_dir = crate::paths::job_dir(&job_id)?;
    crate::jobs::write_state(&job_id, JobState::Running)
        .with_context(|| format!("write Running state for {job_id}"))?;
    let _driver = crate::jobs::record_driver(&job_id);
    write_marker(
        &job_dir,
        &PlanMarker {
            namespace: namespace.clone(),
            name: plan.clone(),
            context: settings.context.clone(),
            kubectl: settings.kubectl.clone(),
        },
    )
    .map_err(|e| anyhow!("{e}"))?;

    let created = {
        let k = kubectl.clone();
        tokio::task::spawn_blocking(move || k.create(&manifest))
            .await
            .map_err(|e| anyhow!("{e}"))?
    };
    if let Err(e) = created {
        let _ = crate::jobs::write_state(&job_id, JobState::Failed);
        return Err(anyhow!("{e}"));
    }
    eprintln!("recipe {label}");
    eprintln!("job    {job_id}");
    eprintln!("plan   blutplan/{plan} in namespace {namespace} (blut-operator runs it)");
    eprintln!("       Ctrl-C or `blut cancel {job_id}` deletes it");

    let outcome = follow(&kubectl, &namespace, &plan, settings.pickup).await;
    match outcome {
        Followed::Finished(status) if status.phase.as_deref() == Some("Succeeded") => {
            crate::jobs::write_state(&job_id, JobState::Done)
                .with_context(|| format!("write Done state for {job_id}"))?;
            print_stages(&status);
            eprintln!(
                "done — {} (BlutPlan {namespace}/{plan})",
                status.message.as_deref().unwrap_or("succeeded")
            );
            Ok(())
        }
        Followed::Finished(status) => {
            let _ = crate::jobs::write_state(&job_id, JobState::Failed);
            print_stages(&status);
            Err(anyhow!(
                "BlutPlan {namespace}/{plan} failed: {}",
                status.message.as_deref().unwrap_or("no message")
            ))
        }
        Followed::Interrupted => {
            let deleted = {
                let k = kubectl.clone();
                let (ns, p) = (namespace.clone(), plan.clone());
                tokio::task::spawn_blocking(move || k.delete(&ns, &p)).await
            };
            let _ = crate::jobs::write_state(&job_id, JobState::Cancelled);
            match deleted {
                Ok(Ok(())) => Err(anyhow!(
                    "cancelled: BlutPlan {namespace}/{plan} deleted (its Job and pods go with it)"
                )),
                Ok(Err(e)) => Err(anyhow!(
                    "cancelled, but deleting BlutPlan {namespace}/{plan} failed: {e}; delete \
                     it with `kubectl delete blutplan {plan} -n {namespace}`"
                )),
                Err(e) => Err(anyhow!("cancelled; delete task failed: {e}")),
            }
        }
        Followed::NotPickedUp(waited) => {
            let k = kubectl.clone();
            let (ns, p) = (namespace.clone(), plan.clone());
            let _ = tokio::task::spawn_blocking(move || k.delete(&ns, &p)).await;
            let _ = crate::jobs::write_state(&job_id, JobState::Failed);
            Err(anyhow!(
                "no blut-operator reported on BlutPlan {namespace}/{plan} within {}s, so it \
                 was deleted. The CRD is installed but the operator is not running or \
                 cannot see namespace {namespace} (kubectl -n blut-system get pods); \
                 BLUT_K8S_PICKUP_SECS sets the wait",
                waited.as_secs()
            ))
        }
        Followed::Vanished => {
            let _ = crate::jobs::write_state(&job_id, JobState::Cancelled);
            Err(anyhow!(
                "BlutPlan {namespace}/{plan} was deleted before it finished"
            ))
        }
        Followed::LostContact(e) => {
            let _ = crate::jobs::write_state(&job_id, JobState::Failed);
            Err(anyhow!(
                "lost contact with the cluster ({e}); BlutPlan {namespace}/{plan} may still \
                 be running: `kubectl get blutplan {plan} -n {namespace}`; delete it with \
                 `blut cancel {job_id}`"
            ))
        }
    }
}

enum Followed {
    Finished(ClusterStatus),
    Interrupted,
    NotPickedUp(Duration),
    Vanished,
    LostContact(String),
}

/// Resolves on the first SIGTERM or Ctrl-C.
async fn interrupted() {
    let term = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            if let Ok(mut s) = signal(SignalKind::terminate()) {
                s.recv().await;
            }
        }
        #[cfg(not(unix))]
        {
            std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term => {}
    }
}

async fn follow(kubectl: &Kubectl, namespace: &str, plan: &str, pickup: Duration) -> Followed {
    let started = std::time::Instant::now();
    let stop = interrupted();
    tokio::pin!(stop);
    let mut last_line = String::new();
    let mut failures = 0u32;
    loop {
        let read = {
            let k = kubectl.clone();
            let (ns, p) = (namespace.to_string(), plan.to_string());
            tokio::task::spawn_blocking(move || k.get(&ns, &p))
        };
        let doc = tokio::select! {
            _ = &mut stop => return Followed::Interrupted,
            r = read => r,
        };
        match doc {
            Ok(Ok(Some(doc))) => {
                failures = 0;
                let status = parse_status(&doc);
                let line = format!(
                    "{} {}",
                    status.phase.as_deref().unwrap_or("Submitted"),
                    status.message.as_deref().unwrap_or("")
                );
                if line != last_line {
                    eprintln!("[k8s] {}", line.trim_end());
                    last_line = line;
                }
                if status.is_terminal() {
                    return Followed::Finished(status);
                }
                if status.phase.is_none() && started.elapsed() > pickup {
                    return Followed::NotPickedUp(pickup);
                }
            }
            Ok(Ok(None)) => return Followed::Vanished,
            Ok(Err(e)) => {
                failures += 1;
                if failures >= MAX_READ_FAILURES {
                    return Followed::LostContact(e.to_string());
                }
            }
            Err(e) => return Followed::LostContact(e.to_string()),
        }
        tokio::select! {
            _ = &mut stop => return Followed::Interrupted,
            _ = tokio::time::sleep(POLL) => {}
        }
    }
}

fn print_stages(status: &ClusterStatus) {
    for (stage, state) in &status.stages {
        eprintln!("  {stage:<32} {state}");
    }
}

#[cfg(test)]
mod tests {
    use super::{K8sUnsupported, validate_for_cluster};
    use crate::recipes::recipe::{RecipeCategory, RecipeDef};

    #[derive(schemars::JsonSchema)]
    #[allow(dead_code)]
    struct PathArgs {
        checkpoint_path: std::path::PathBuf,
    }

    /// A recipe whose compile checks a path, as cookbook recipes do.
    static CHECKS_PATHS: RecipeDef = RecipeDef {
        name: "checks_paths",
        description: "d",
        backend_id: "b",
        category: RecipeCategory::User,
        input_kinds: &[],
        output_kind: "k",
        schedule: None,
        args_schema_fn: crate::recipes::recipe::schema_of::<PathArgs>,
        compile_fn: |_| {
            Err(crate::framework::RecipeError::CompileFailed(
                "checkpoint_path not found on this machine".into(),
            ))
        },
    };

    #[test]
    fn args_are_checked_against_the_schema_not_by_compiling_here() {
        // A path that exists only in the pod must not be refused here.
        validate_for_cluster(
            &CHECKS_PATHS,
            &serde_json::json!({"checkpoint_path": "/only/in/the/pod"}),
        )
        .unwrap();
        // A misspelled or missing key still fails before submission.
        for (args, needle) in [
            (
                serde_json::json!({"checkpoint_pth": "/x"}),
                "checkpoint_path",
            ),
            (
                serde_json::json!({"checkpoint_path": "/x", "epochz": 2}),
                "epochz",
            ),
        ] {
            let err = validate_for_cluster(&CHECKS_PATHS, &args)
                .unwrap_err()
                .to_string();
            assert!(err.contains(needle), "{err}");
        }
    }

    fn none() -> K8sUnsupported {
        K8sUnsupported {
            config_mode: false,
            shared_cache: false,
            no_cache: false,
            sync_io: false,
            tenant_is_default: true,
            experiment: false,
        }
    }

    #[test]
    fn plain_flags_pass_and_each_unsupported_one_is_named() {
        none().refuse().unwrap();
        let err = K8sUnsupported {
            config_mode: true,
            no_cache: true,
            tenant_is_default: false,
            ..none()
        }
        .refuse()
        .unwrap_err()
        .to_string();
        for needle in [
            "config mode",
            "--no-cache",
            "--tenant",
            "nothing ran locally",
        ] {
            assert!(err.contains(needle), "{err:?} should name {needle:?}");
        }
    }
}
