// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `--launcher p2p` and `--launcher cloud` used to run every stage on the
//! submitting machine: `launcher_for` mapped both to the local launcher and no
//! command installed an execution adapter. Each test drives a real command
//! handler with a recipe whose only stage writes a marker file, and checks
//! that the command refuses, names what does dispatch there, writes no job,
//! and that the stage never ran.
//!
//! `recipe run`, `recipe declare --run` and `plan resume` now place cloud
//! work (a build with the `cloud` feature and `BLUT_CLOUD_QUEUE` set); here,
//! with no queue configured, they refuse and say what is missing. The other
//! commands still refuse cloud outright.

use super::{Cli, Command, Parser};
use serde_json::json;

/// Lives entirely in its struct; the stage's observable effect is the marker
/// file it writes as a side effect, outside the stage directory.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ProbeArtifact {
    marker: String,
}

impl crate::framework::Artifact for ProbeArtifact {
    const KIND: &'static str = "test.launcher_probe";
    const SCHEMA: u32 = 1;
    const INLINE: bool = true;

    fn content_hash(&self) -> crate::framework::ContentHash {
        crate::framework::ContentHash::of_bytes(self.marker.as_bytes())
    }

    fn primary_path(&self) -> &std::path::Path {
        std::path::Path::new(".")
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct ProbeArgs {
    /// File the stage writes when it runs, on whatever machine runs it.
    marker: String,
    /// An HPO / study search dimension.
    #[serde(default)]
    x: Option<f64>,
    /// A partition axis.
    #[serde(default)]
    tag: Option<String>,
}

struct ProbeStage;

#[async_trait::async_trait]
impl crate::framework::Stage for ProbeStage {
    const NAME: &'static str = "launcher_probe_stage";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [crate::framework::Resource] = &[crate::framework::Resource::Cpu];
    type Input = ();
    type Output = ProbeArtifact;
    type Args = ProbeArgs;

    async fn run(
        &self,
        _ctx: &crate::framework::StageContext,
        _input: (),
        args: &Self::Args,
    ) -> std::result::Result<Self::Output, crate::framework::StageError> {
        let path = std::path::PathBuf::from(&args.marker);
        std::fs::write(&path, b"ran on this machine").map_err(|source| {
            crate::framework::StageError::Io {
                path: path.clone(),
                source,
            }
        })?;
        Ok(ProbeArtifact {
            marker: args.marker.clone(),
        })
    }
}

impl crate::framework::Compatible<crate::backends::LamuTrainerBackend> for ProbeStage {}

fn compile_probe(
    raw: serde_json::Value,
) -> std::result::Result<crate::framework::plan::CompiledPlan, crate::framework::RecipeError> {
    let args: ProbeArgs =
        serde_json::from_value(raw).map_err(|e| crate::framework::RecipeError::InvalidArgs {
            field: None,
            message: e.to_string(),
        })?;
    Ok(
        crate::framework::Plan::<(), crate::backends::LamuTrainerBackend>::new(
            "launcher_probe",
            json!({}),
        )
        .start(ProbeStage, args)
        .finish()
        .into_compiled(),
    )
}

static PROBE_DEF: crate::recipes::recipe::RecipeDef = crate::recipes::recipe::RecipeDef {
    name: "launcher_probe",
    description: "launcher refusal fixture: writes a marker file where it runs",
    backend_id: "lamu_trainer",
    category: crate::recipes::recipe::RecipeCategory::User,
    input_kinds: &[],
    output_kind: "test.launcher_probe",
    schedule: None,
    args_schema_fn: || crate::recipes::recipe::schema_of::<ProbeArgs>(),
    compile_fn: compile_probe,
};

struct ProbeCookbook;

impl crate::framework::Cookbook for ProbeCookbook {
    fn name(&self) -> &'static str {
        "launcher_probe"
    }

    fn recipes(&self) -> &'static [&'static crate::recipes::recipe::RecipeDef] {
        static RECIPES: &[&crate::recipes::recipe::RecipeDef] = &[&PROBE_DEF];
        RECIPES
    }
}

fn registry() -> crate::framework::Registry {
    let mut registry = crate::framework::Registry::new();
    registry.register(Box::new(ProbeCookbook));
    registry
}

/// Points every job, data, cache and partition directory into a temp dir for
/// the test's lifetime, and restores the environment on drop. Hold
/// `TEST_ENV_LOCK` for the guard's whole life.
struct Sandbox {
    dir: tempfile::TempDir,
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl Sandbox {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let vars: [(&'static str, std::ffi::OsString); 10] = [
            // No cloud queue configured: the cloud cases below are refusals.
            ("BLUT_CLOUD_QUEUE", "".into()),
            ("LAMU_TRAIN_JOBS_DIR", dir.path().join("jobs").into()),
            ("LAMU_TRAIN_DATA_DIR", dir.path().join("data").into()),
            ("XDG_DATA_HOME", dir.path().join("xdg-data").into()),
            ("XDG_CACHE_HOME", dir.path().join("xdg-cache").into()),
            ("XDG_CONFIG_HOME", dir.path().join("xdg-config").into()),
            ("BLUT_PARTITIONS_DIR", dir.path().join("partitions").into()),
            ("BLUT_SCHED_DEVICES", "0".into()),
            ("CUDA_VISIBLE_DEVICES", "".into()),
            ("BLUT_NO_CONTAIN", "1".into()),
        ];
        let saved = vars
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect();
        for (key, value) in vars {
            // SAFETY: the caller holds TEST_ENV_LOCK for this guard's lifetime.
            unsafe { std::env::set_var(key, value) };
        }
        Self { dir, saved }
    }

    fn marker(&self) -> std::path::PathBuf {
        self.dir.path().join("stage-ran")
    }

    fn args(&self) -> String {
        json!({ "marker": self.marker() }).to_string()
    }

    fn assert_nothing_ran(&self, what: &str) {
        assert!(
            !self.marker().exists(),
            "{what}: the stage ran on this machine"
        );
        let jobs = crate::jobs::list_jobs().unwrap();
        assert!(jobs.is_empty(), "{what}: a refused command wrote {jobs:?}");
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..) {
            // SAFETY: TEST_ENV_LOCK is still held by the caller.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn lock() -> std::sync::MutexGuard<'static, ()> {
    crate::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Launchers no command here can place, each with what its refusal names.
/// `recipe run`, `recipe declare --run` and `plan resume` can place cloud
/// work, so for them [`CLOUD_LAUNCH`] says what is missing instead.
const REMOTE: [(&str, &str); 2] = [
    ("p2p", "--launcher p2p --p2p-seed"),
    ("cloud", "only `recipe run`"),
];

/// What a cloud-capable command names when it cannot launch: the queue
/// variable in a build that has the feature, the feature in one that has not.
const CLOUD_LAUNCH: &str = if cfg!(feature = "cloud") {
    "BLUT_CLOUD_QUEUE"
} else {
    "`cloud` feature"
};

/// [`REMOTE`] as a command that can place cloud work refuses it.
fn launch_refusals() -> [(&'static str, &'static str); 2] {
    [REMOTE[0], ("cloud", CLOUD_LAUNCH)]
}

fn recipe_cmd(argv: &[&str]) -> super::RecipeCommand {
    match Cli::try_parse_from(argv).expect("parse").command {
        Some(Command::Recipe { cmd }) => cmd,
        other => panic!("expected a recipe command, got {other:?}"),
    }
}

// The env lock must span the awaited command: the handlers read these vars.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn recipe_run_refuses_p2p_and_cloud_before_anything_runs() {
    let _lock = lock();
    let reg = registry();
    for (launcher, names) in launch_refusals() {
        let sandbox = Sandbox::new();
        let args = sandbox.args();
        let config_dir = sandbox.dir.path().join("conf").display().to_string();
        let runs: [Vec<&str>; 3] = [
            vec!["--args", args.as_str()],
            vec!["--args", args.as_str(), "--dry-run"],
            vec![
                "--config-dir",
                config_dir.as_str(),
                "--config-name",
                "c",
                "--sweep",
                "x=1,2",
            ],
        ];
        for extra in runs {
            let mut argv = vec!["blut", "recipe", "run", "launcher_probe"];
            argv.extend(extra.iter().copied());
            argv.extend(["--launcher", launcher]);
            let err = super::run_recipe(&reg, recipe_cmd(&argv))
                .await
                .expect_err("a placement no command can make must be refused")
                .to_string();
            if placed_on_a_mesh(launcher) {
                // A p2p build places it, on a mesh this machine has not
                // joined: no identity yet, so nothing runs.
                assert!(
                    err.contains("--launcher p2p needs this machine's mesh identity"),
                    "{argv:?}: {err}"
                );
                assert!(err.contains("Nothing ran"), "{argv:?}: {err}");
            } else {
                assert!(
                    err.contains(&format!("--launcher {launcher} cannot place `recipe run`")),
                    "{argv:?}: {err}"
                );
                assert!(err.contains(names), "{argv:?}: {err}");
            }
            sandbox.assert_nothing_ran(&format!("{argv:?}"));
        }
    }
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn recipe_declare_run_refuses_p2p_and_cloud_without_pointing_at_recipe_run() {
    let _lock = lock();
    let reg = registry();
    for (launcher, names) in launch_refusals() {
        let sandbox = Sandbox::new();
        let file = sandbox.dir.path().join("r.toml").display().to_string();
        let argv = [
            "blut",
            "recipe",
            "declare",
            file.as_str(),
            "--run",
            "--launcher",
            launcher,
        ];
        let err = super::run_recipe(&reg, recipe_cmd(&argv))
            .await
            .expect_err("refused")
            .to_string();
        if placed_on_a_mesh(launcher) {
            assert!(
                err.contains("--launcher p2p needs this machine's mesh identity"),
                "{err}"
            );
        } else {
            assert!(
                err.contains(&format!(
                    "--launcher {launcher} cannot place `recipe declare --run`"
                )),
                "{err}"
            );
            assert!(err.contains(names), "{err}");
        }
        sandbox.assert_nothing_ran(launcher);
    }
}

/// `recipe run` and `recipe declare --run` place `--launcher p2p` on the mesh
/// in a build with the `p2p` feature; every other command refuses it.
fn placed_on_a_mesh(launcher: &str) -> bool {
    cfg!(feature = "p2p") && launcher == "p2p"
}

/// `--p2p-*` flags with another launcher would be ignored; they are refused.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn mesh_flags_with_another_launcher_are_refused() {
    let _lock = lock();
    let reg = registry();
    let sandbox = Sandbox::new();
    let args = sandbox.args();
    let argv = [
        "blut",
        "recipe",
        "run",
        "launcher_probe",
        "--args",
        args.as_str(),
        "--p2p-seed",
        "127.0.0.1:9@00",
    ];
    let err = super::run_recipe(&reg, recipe_cmd(&argv))
        .await
        .expect_err("refused")
        .to_string();
    assert!(
        err.contains("--p2p-seed only apply to --launcher p2p"),
        "{err}"
    );
    sandbox.assert_nothing_ran("stray --p2p-seed");
}

/// With an identity but a seed that does not answer, `--launcher p2p` stops
/// before a job exists: there is no mesh to place anything on.
#[cfg(feature = "p2p")]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn recipe_run_on_an_unreachable_mesh_runs_nothing() {
    let _lock = lock();
    let reg = registry();
    let sandbox = Sandbox::new();
    let key = sandbox.dir.path().join("identity.key");
    std::fs::write(&key, crate::p2p::crypto::KeyPair::generate().to_bytes()).unwrap();
    let peer = crate::p2p::crypto::KeyPair::generate();
    // A port nothing listens on.
    let port = std::net::UdpSocket::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let seed = format!(
        "127.0.0.1:{port}@{}",
        faster_hex::hex_string(peer.verifying.as_bytes())
    );
    let args = sandbox.args();
    let key_arg = key.display().to_string();
    let argv = [
        "blut",
        "recipe",
        "run",
        "launcher_probe",
        "--args",
        args.as_str(),
        "--launcher",
        "p2p",
        "--p2p-key",
        key_arg.as_str(),
        "--p2p-seed",
        seed.as_str(),
    ];
    let err = super::run_recipe(&reg, recipe_cmd(&argv))
        .await
        .expect_err("refused")
        .to_string();
    assert!(err.contains("could not reach any seed"), "{err}");
    assert!(err.contains("Nothing ran"), "{err}");
    sandbox.assert_nothing_ran("unreachable mesh");
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn hpo_run_refuses_p2p_and_cloud_before_any_trial() {
    let _lock = lock();
    let reg = registry();
    for (launcher, names) in REMOTE {
        let sandbox = Sandbox::new();
        let args = sandbox.args();
        let argv = [
            "blut",
            "hpo",
            "run",
            "launcher_probe",
            "--args",
            args.as_str(),
            "--param",
            "x=uniform(0,1)",
            "--algo",
            "random",
            "--max-trials",
            "1",
            "--launcher",
            launcher,
        ];
        let cmd = match Cli::try_parse_from(argv).expect("parse").command {
            Some(Command::Hpo { cmd }) => cmd,
            other => panic!("expected hpo, got {other:?}"),
        };
        let err = super::run_hpo(&reg, cmd)
            .await
            .expect_err("refused")
            .to_string();
        assert!(
            err.contains(&format!(
                "--launcher {launcher} cannot place `hpo` and `study` trials"
            )),
            "{err}"
        );
        assert!(err.contains(names), "{err}");
        sandbox.assert_nothing_ran(launcher);
    }
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn study_run_refuses_p2p_and_cloud_before_any_trial() {
    let _lock = lock();
    let reg = registry();
    for (launcher, names) in REMOTE {
        let sandbox = Sandbox::new();
        let spec = sandbox.dir.path().join("study.toml");
        std::fs::write(
            &spec,
            "recipe = \"launcher_probe\"\nsampler = \"random\"\n\
             [[objectives]]\nname = \"a\"\nmetric = \"m.a\"\ndirection = \"minimize\"\n\
             [[objectives]]\nname = \"b\"\nmetric = \"m.b\"\ndirection = \"maximize\"\n\
             [space.dims.x]\ndist = \"uniform\"\nlow = 0.0\nhigh = 1.0\n",
        )
        .unwrap();
        let err = super::study::run_study(
            &reg,
            super::study::StudyCommand::Run {
                spec: spec.display().to_string(),
                args: sandbox.args(),
                max_trials: 1,
                tenant: "default".into(),
                launcher: launcher.into(),
            },
        )
        .await
        .expect_err("refused")
        .to_string();
        assert!(
            err.contains(&format!(
                "--launcher {launcher} cannot place `hpo` and `study` trials"
            )),
            "{err}"
        );
        assert!(err.contains(names), "{err}");
        sandbox.assert_nothing_ran(launcher);
    }
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn partition_backfill_and_status_refuse_p2p_and_cloud() {
    use crate::config::partition::{PartitionDim, PartitionSet};
    let _lock = lock();
    let reg = registry();
    for (launcher, names) in REMOTE {
        let sandbox = Sandbox::new();
        let set = PartitionSet {
            name: "tags".into(),
            recipe: "launcher_probe".into(),
            dims: vec![PartitionDim {
                axis: "tag".into(),
                values: vec!["a".into()],
            }],
        };
        set.save().unwrap();
        let commands = [
            super::PartitionCommand::Backfill {
                recipe: set.recipe.clone(),
                name: set.name.clone(),
                force: false,
                missing: false,
                stale: false,
                partitions: None,
                args: sandbox.args(),
                launcher: launcher.into(),
                tenant: "default".into(),
                sync_io: false,
            },
            super::PartitionCommand::Status {
                recipe: set.recipe.clone(),
                name: set.name.clone(),
                args: sandbox.args(),
                launcher: launcher.into(),
                tenant: "default".into(),
                json: false,
            },
        ];
        for cmd in commands {
            let shown = format!("{cmd:?}");
            let err = super::run_partition(&reg, cmd)
                .await
                .expect_err("refused")
                .to_string();
            assert!(
                err.contains(&format!(
                    "--launcher {launcher} cannot place `partition` runs"
                )),
                "{shown}: {err}"
            );
            assert!(err.contains(names), "{err}");
            sandbox.assert_nothing_ran(&shown);
        }
    }
}

/// A job recorded with `--launcher p2p` before the refusal ran here; resuming
/// it "as it was run" would repeat the false placement. A cloud job resumes
/// on the cloud queue, so without one it is refused for that reason.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn plan_resume_refuses_a_job_recorded_with_p2p_or_cloud() {
    let _lock = lock();
    let reg = registry();
    for (launcher, names) in launch_refusals() {
        let sandbox = Sandbox::new();
        let job_id = crate::jobs::new_job_id();
        let job_dir = crate::paths::job_dir(&job_id).unwrap();
        let args = json!({ "marker": sandbox.marker() });
        super::RecipeMarker {
            name: "launcher_probe".into(),
            args: args.clone(),
            source_args: Some(args),
            resolved_handles: Default::default(),
            launcher: Some(launcher.into()),
        }
        .write_to(&job_dir)
        .unwrap();
        crate::jobs::write_state(&job_id, crate::jobs::JobState::Failed).unwrap();

        let err = super::run_plan_cmd(
            &reg,
            super::PlanCommand::Resume {
                id: job_id.clone(),
                shared_cache: false,
                sync_io: false,
            },
        )
        .await
        .expect_err("refused")
        .to_string();
        assert!(
            err.contains(&format!(
                "--launcher {launcher} cannot place `plan resume` of {job_id}"
            )),
            "{err}"
        );
        if launcher == "p2p" {
            assert!(err.contains("recipe run launcher_probe"), "{err}");
        }
        assert!(err.contains(names), "{err}");
        assert!(!sandbox.marker().exists(), "the resumed stage ran here");
        assert_eq!(
            crate::jobs::read_state(&job_id).unwrap(),
            crate::jobs::JobState::Failed,
            "a refused resume must leave the job as it was"
        );
    }
}

/// The local launch path itself refuses, before a job directory exists, when
/// a caller reaches it with a placement it cannot make: p2p always, cloud
/// without a queue (or without the `cloud` feature).
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn the_local_launch_path_refuses_p2p_and_cloud_before_writing_a_job() {
    use crate::config::launcher::LaunchTarget;
    let _lock = lock();
    // SAFETY: serialized by TEST_ENV_LOCK (`lock`).
    unsafe { std::env::remove_var(crate::config::launcher::CLOUD_QUEUE_ENV) };
    for target in [LaunchTarget::P2P, LaunchTarget::Cloud] {
        let sandbox = Sandbox::new();
        let plan = compile_probe(json!({ "marker": sandbox.marker() })).unwrap();
        let err = super::launch_compiled_plan(
            &crate::framework::Registry::new(),
            "launcher_probe",
            plan,
            None,
            None,
            false,
            target,
            None,
            false,
            false,
            crate::tenant::Tenant::default(),
            None,
        )
        .await
        .expect_err("refused")
        .to_string();
        assert!(
            err.contains(&format!(
                "--launcher {} cannot place 'launcher_probe' on this path",
                target.as_str()
            )),
            "{err}"
        );
        sandbox.assert_nothing_ran(target.as_str());
    }
}

/// The fixture is not vacuous: run locally, the same recipe writes its marker.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn the_probe_recipe_runs_and_writes_its_marker_locally() {
    let _lock = lock();
    let reg = registry();
    let sandbox = Sandbox::new();
    let args = sandbox.args();
    let argv = [
        "blut",
        "recipe",
        "run",
        "launcher_probe",
        "--args",
        args.as_str(),
        "--launcher",
        "local",
    ];
    super::run_recipe(&reg, recipe_cmd(&argv))
        .await
        .expect("a local run of the probe succeeds");
    assert!(sandbox.marker().exists(), "the local run wrote its marker");
}
