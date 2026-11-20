// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Every stage cache key folds the code identity (engine + cookbooks). A
//! binary fixes its identity once per process, so each case runs in a child
//! process: this test binary re-executes itself with `CHILD_ENV` set, the
//! child installs the identity it is given, runs one plan against a shared
//! cache, and prints what happened. That is what two builds of a cookbook
//! sharing one cache look like.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use blut::backends::TrainingBackend;
use blut::framework::artifact::{Artifact, ContentHash};
use blut::framework::code_identity::{BuildIdentity, CodeIdentity};
use blut::framework::compat::Compatible;
use blut::framework::executor::{ExecCtx, ParallelExecutor};
use blut::framework::plan::Plan;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};
use blut::framework::{CacheHandle, StageError};
use serde::{Deserialize, Serialize};

const CHILD_ENV: &str = "BLUT_CODE_IDENTITY_TEST_CHILD";

struct TestBackend;
impl TrainingBackend for TestBackend {
    const ID: &'static str = "code-identity-test";
    const DESCRIPTION: &'static str = "code identity tests";
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Number {
    n: u32,
}
impl Artifact for Number {
    const KIND: &'static str = "test.number";
    const SCHEMA: u32 = 1;
    const INLINE: bool = true;
    fn content_hash(&self) -> ContentHash {
        ContentHash::of_bytes(&self.n.to_le_bytes())
    }
    fn primary_path(&self) -> &Path {
        Path::new(".")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct WorkArgs {
    n: u32,
}

/// A stage that opts into checkpoint slots and records where its slot was.
struct Work;
#[async_trait]
impl Stage for Work {
    const NAME: &'static str = "work";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = Number;
    type Args = WorkArgs;
    fn checkpoint_slot(&self, _: &WorkArgs) -> bool {
        true
    }
    async fn run(&self, ctx: &StageContext, _: (), a: &WorkArgs) -> Result<Number, StageError> {
        let slot = ctx.checkpoint.as_ref().map(|s| s.dir.clone());
        println!(
            "SLOT {}",
            slot.map(|p| p.display().to_string()).unwrap_or_default()
        );
        Ok(Number { n: a.n * 2 })
    }
}
impl Compatible<TestBackend> for Work {}

#[derive(Debug, Serialize, Deserialize)]
struct Child {
    identity: Option<CodeIdentity>,
    cache: PathBuf,
    checkpoints: PathBuf,
    job: PathBuf,
}

/// The child half. A no-op unless this process was started as a child.
#[tokio::test]
async fn child_entry() {
    let Ok(spec) = std::env::var(CHILD_ENV) else {
        return;
    };
    let child: Child = serde_json::from_str(&spec).unwrap();
    if let Some(identity) = child.identity {
        blut::framework::code_identity::install(identity).expect("first install");
    }
    let plan = Plan::<(), TestBackend>::new("code-identity", serde_json::json!({}))
        .start(Work, WorkArgs { n: 21 })
        .finish()
        .into_compiled();
    std::fs::create_dir_all(&child.job).unwrap();
    let mut ctx = ExecCtx::new(child.job.clone()).with_checkpoint_root(child.checkpoints);
    ctx.cache = std::sync::Arc::new(
        CacheHandle::job_local(child.job.join("_cache")).with_global(child.cache),
    );
    let r = ParallelExecutor::execute(plan, ctx).await.unwrap();
    println!("RESULT {} {}", r.n_cache_hits, r.n_cache_misses);
}

struct Outcome {
    hit: bool,
    slot: Option<String>,
}

fn run_child(root: &Path, job: &str, identity: Option<CodeIdentity>) -> Outcome {
    let spec = Child {
        identity,
        cache: root.join("shared-cache"),
        checkpoints: root.join("checkpoints"),
        job: root.join("jobs").join(job),
    };
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, serde_json::to_string(&spec).unwrap())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child {job} failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The harness prints "test child_entry ... " before the child's own
    // output, on the same line.
    let after = |tag: &'static str| {
        move |l: &str| l.split_once(tag).map(|(_, rest)| rest.trim().to_string())
    };
    let result = stdout
        .lines()
        .find_map(after("RESULT "))
        .unwrap_or_else(|| panic!("no RESULT from child {job}:\n{stdout}"));
    let mut parts = result
        .split_whitespace()
        .map(|n| n.parse::<usize>().unwrap());
    let (hits, misses) = (parts.next().unwrap(), parts.next().unwrap());
    assert_eq!(hits + misses, 1, "one stage");
    Outcome {
        hit: hits == 1,
        slot: stdout.lines().find_map(after("SLOT ")),
    }
}

fn identity(commit: &str, version: &str, dirty: Option<&str>) -> CodeIdentity {
    CodeIdentity::with_engine(BuildIdentity::engine()).with_cookbook(BuildIdentity::from_build_env(
        "test-cookbook",
        version,
        Some(commit),
        dirty,
    ))
}

#[test]
fn a_cache_entry_is_reused_only_by_the_same_build() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let td = tempfile::tempdir().unwrap();
    let root = td.path();
    let c1 = identity("c1", "1.0.0", None);

    let first = run_child(root, "a", Some(c1.clone()));
    assert!(!first.hit, "cold cache");
    assert!(
        run_child(root, "b", Some(c1.clone())).hit,
        "same commit: hit"
    );

    let other_commit = run_child(root, "c", Some(identity("c2", "1.0.0", None)));
    assert!(!other_commit.hit, "different commit: miss");
    assert!(
        !run_child(root, "d", Some(identity("c1", "1.0.1", None))).hit,
        "version bump: miss"
    );

    let dirty = run_child(root, "e", Some(identity("c1", "1.0.0", Some("d1"))));
    assert!(!dirty.hit, "dirty tree vs the clean build: miss");
    assert!(
        !run_child(root, "f", Some(identity("c1", "1.0.0", Some("d2")))).hit,
        "another dirty state: miss"
    );
    assert!(
        run_child(root, "g", Some(identity("c1", "1.0.0", Some("d1")))).hit,
        "the same dirty state again: hit"
    );
    assert!(
        !run_child(root, "h", None).hit,
        "a binary that installs no cookbook identity: miss"
    );

    // Checkpoint slots are keyed by the cache key, so they follow.
    let slots = [&first, &other_commit, &dirty].map(|o| o.slot.clone().expect("slot"));
    assert_ne!(slots[0], slots[1], "a slot per commit");
    assert_ne!(slots[0], slots[2], "a slot per dirty state");
}

#[test]
fn installing_a_second_different_identity_is_refused() {
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    // In this process (not a child): nothing installed yet unless another
    // test here computed a key, which none does.
    let a = identity("x1", "1.0.0", None);
    let b = identity("x2", "1.0.0", None);
    assert!(blut::framework::code_identity::install(a.clone()).is_ok());
    assert!(
        blut::framework::code_identity::install(a.clone()).is_ok(),
        "same again"
    );
    assert_eq!(
        blut::framework::code_identity::install(b),
        Err(Box::new(a.clone()))
    );
    assert_eq!(blut::framework::code_identity::current(), &a);
}
