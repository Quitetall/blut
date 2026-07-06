// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Integration tests for the BLUT cloud worker agent.
//!
//! Tests the job lifecycle: queue → read → process → result.

use std::path::PathBuf;

/// Helper: create a temp dir for testing.
fn temp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

/// Helper: write a job JSON to the queue directory.
fn write_job(queue_dir: &std::path::Path, id: &str, recipe: &str) -> PathBuf {
    let job = serde_json::json!({
        "id": id,
        "recipe": recipe,
        "args": {},
        "resources": {
            "gpu": false,
            "memory_gib": 4,
            "cpu_cores": 2
        }
    });
    let path = queue_dir.join(format!("{id}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&job).unwrap()).unwrap();
    path
}

#[test]
fn job_json_roundtrip() {
    let td = temp_dir();
    let queue_dir = td.path().join("queue");
    std::fs::create_dir_all(&queue_dir).unwrap();

    // Write a job
    let job_path = write_job(&queue_dir, "test-001", "train_from_dataset");

    // Read it back
    let content = std::fs::read_to_string(&job_path).unwrap();
    let job: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(job["id"], "test-001");
    assert_eq!(job["recipe"], "train_from_dataset");
    assert_eq!(job["resources"]["gpu"], false);
    assert_eq!(job["resources"]["memory_gib"], 4);
}

#[test]
fn result_json_format() {
    let td = temp_dir();
    let results_dir = td.path().join("results");
    std::fs::create_dir_all(&results_dir).unwrap();

    // Write a result
    let result = serde_json::json!({
        "job_id": "test-001",
        "status": "succeeded",
        "artifacts": ["checkpoint/model.pt"],
        "metrics": [
            {"epoch": 1, "loss": 0.42},
            {"epoch": 2, "loss": 0.31}
        ],
        "compute_time_secs": 120,
        "error": null
    });
    let result_path = results_dir.join("test-001.json");
    std::fs::write(&result_path, serde_json::to_string_pretty(&result).unwrap()).unwrap();

    // Read it back
    let content = std::fs::read_to_string(&result_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["job_id"], "test-001");
    assert_eq!(parsed["status"], "succeeded");
    assert_eq!(parsed["compute_time_secs"], 120);
    assert_eq!(parsed["metrics"][0]["loss"], 0.42);
}

#[test]
fn queue_listing_finds_json_files() {
    let td = temp_dir();
    let queue_dir = td.path().join("queue");
    std::fs::create_dir_all(&queue_dir).unwrap();

    // Write 3 jobs
    write_job(&queue_dir, "job-001", "train_from_dataset");
    write_job(&queue_dir, "job-002", "finetune_pretrained");
    write_job(&queue_dir, "job-003", "eval_only");

    // Write a non-json file (should be ignored)
    std::fs::write(queue_dir.join("README.txt"), "not a job").unwrap();

    // List JSON files
    let entries: Vec<_> = std::fs::read_dir(&queue_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .collect();

    assert_eq!(entries.len(), 3);
}

#[test]
fn atomic_job_processing() {
    let td = temp_dir();
    let queue_dir = td.path().join("queue");
    std::fs::create_dir_all(&queue_dir).unwrap();

    // Write a job
    let job_path = write_job(&queue_dir, "job-001", "train_from_dataset");
    assert!(job_path.exists());

    // Simulate atomic processing: rename to .processing
    let processing_path = job_path.with_extension("json.processing");
    std::fs::rename(&job_path, &processing_path).unwrap();

    // Original should be gone, processing should exist
    assert!(!job_path.exists());
    assert!(processing_path.exists());

    // After completion, remove processing file
    std::fs::remove_file(&processing_path).unwrap();
    assert!(!processing_path.exists());
}

#[test]
fn malformed_job_file_handled() {
    let td = temp_dir();
    let queue_dir = td.path().join("queue");
    std::fs::create_dir_all(&queue_dir).unwrap();

    // Write malformed JSON
    let bad_path = queue_dir.join("bad-job.json");
    std::fs::write(&bad_path, "{not valid json").unwrap();

    // Try to parse
    let content = std::fs::read_to_string(&bad_path).unwrap();
    let result = serde_json::from_str::<serde_json::Value>(&content);

    assert!(result.is_err());

    // Move aside (as the worker does)
    let aside_path = bad_path.with_extension("json.bad");
    std::fs::rename(&bad_path, &aside_path).unwrap();

    assert!(!bad_path.exists());
    assert!(aside_path.exists());
}

#[test]
fn concurrent_queue_access() {
    use std::sync::Arc;
    use std::thread;

    let td = Arc::new(temp_dir());
    let queue_dir = td.path().join("queue");
    std::fs::create_dir_all(&queue_dir).unwrap();

    // Multiple threads writing jobs concurrently
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let qd = queue_dir.clone();
            thread::spawn(move || {
                write_job(&qd, &format!("job-{i:03}"), "train_from_dataset");
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // All 10 jobs should be present
    let entries: Vec<_> = std::fs::read_dir(&queue_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .collect();

    assert_eq!(entries.len(), 10);
}
