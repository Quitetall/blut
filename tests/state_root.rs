// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `$LAMU_TRAIN_DATA_DIR` is the root of a separate install: every store of
//! user state resolves under it, an explicit per-store variable still wins,
//! and with it unset every path is the historical default.
//!
//! One test in its own binary: it rewrites the process environment, which
//! no other test may observe.

use std::path::{Path, PathBuf};

const STORE_VARS: &[&str] = &[
    "LAMU_TRAIN_DATA_DIR",
    "LAMU_TRAIN_JOBS_DIR",
    "LAMU_TRAIN_CACHE_DIR",
    "BLUT_REGISTRY_DB",
    "BLUT_CATALOG_DB",
    "LAMU_MEMORY_DB",
    "LAMU_TRAIN_POLICY",
    "BLUT_FOOTPRINT_PATH",
    "BLUT_REGISTRY_PATH",
    "LAMU_REGISTRY_PATH",
    "BLUT_PARTITIONS_DIR",
    "BLUT_USER_RECIPES_DIR",
    "BLUT_SLA_PATH",
    "BLUT_COST_LEDGER",
    "BLUT_BUDGET_CAP",
    "BLUT_EVENTS_DIR",
];

fn set(key: &str, value: &Path) {
    // SAFETY: the only test in this binary; no other thread reads the env.
    unsafe { std::env::set_var(key, value) };
}

fn unset(key: &str) {
    // SAFETY: as above.
    unsafe { std::env::remove_var(key) };
}

/// Every store this test covers, as `(name, resolved path)`.
fn stores() -> Vec<(&'static str, PathBuf)> {
    vec![
        ("jobs", blut::paths::jobs_dir().unwrap()),
        (
            "cache",
            blut::framework::CacheHandle::default_global_path().unwrap(),
        ),
        ("registry", blut::registry_db::registry_db_path().unwrap()),
        ("catalog", blut::catalog::catalog_db_path().unwrap()),
        ("datasets", blut::datasets_db::registry_path().unwrap()),
        ("policy", blut::policy::policy_path().unwrap()),
        ("footprints", blut::config::footprint_store_path()),
        ("backend registry", blut::config::registry_path()),
        (
            "recipes",
            blut::recipes::declarative::user_recipes_dir().unwrap(),
        ),
        ("sla", blut::sla::default_sla_path()),
        ("cost", blut::cost::default_ledger_path()),
        ("budget", blut::cost::default_cap_path()),
        ("events", blut::trigger::spool_dir("t")),
    ]
}

#[test]
fn the_data_dir_isolates_every_store_and_unset_keeps_the_old_paths() {
    let td = tempfile::tempdir().unwrap();
    let home = td.path().join("home");
    for key in STORE_VARS {
        unset(key);
    }
    set("HOME", &home);
    set("XDG_DATA_HOME", &home.join(".local/share"));
    set("XDG_CONFIG_HOME", &home.join(".config"));

    // Unset: the historical defaults, byte for byte.
    let legacy = stores();
    let expected_legacy: Vec<(&str, PathBuf)> = vec![
        ("jobs", home.join(".local/share/lamu/train-jobs")),
        ("cache", home.join(".local/share/lamu/train-cache")),
        ("registry", home.join(".blut/registry.db")),
        ("catalog", home.join(".blut/catalog.db")),
        ("datasets", home.join(".local/share/lamu/conversations.db")),
        ("policy", home.join(".config/lamu/train-policy.toml")),
        ("footprints", home.join(".config/blut/footprints.json")),
        ("backend registry", home.join(".config/blut/registry.yaml")),
        ("recipes", home.join(".config/blut/recipes")),
        ("sla", home.join(".blut/sla.jsonl")),
        ("cost", home.join(".blut/cost.jsonl")),
        ("budget", home.join(".blut/budget.toml")),
        ("events", home.join(".blut/events/t")),
    ];
    assert_eq!(legacy, expected_legacy);

    // Set: everything under the root, nothing under HOME.
    let root = td.path().join("install-a");
    set("LAMU_TRAIN_DATA_DIR", &root);
    for (name, path) in stores() {
        assert!(
            path.starts_with(&root),
            "{name} resolved to {} outside the data dir",
            path.display()
        );
    }
    assert_eq!(blut::paths::data_dir().unwrap(), root);

    // An explicit per-store variable still wins over the root.
    let explicit = td.path().join("elsewhere/registry.db");
    set("BLUT_REGISTRY_DB", &explicit);
    assert_eq!(blut::registry_db::registry_db_path().unwrap(), explicit);
    unset("BLUT_REGISTRY_DB");

    // A second root is a second install: no path is shared.
    let first: Vec<PathBuf> = stores().into_iter().map(|(_, p)| p).collect();
    set("LAMU_TRAIN_DATA_DIR", &td.path().join("install-b"));
    let second: Vec<PathBuf> = stores().into_iter().map(|(_, p)| p).collect();
    assert!(first.iter().all(|p| !second.contains(p)));
}
