// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! TEST FIXTURE ONLY: execute a local CPU PlanSpec with a public, synthetic test
//! key. Never use this producer/key for authority or production collection.
//! Usage: openwarrant_fixture_producer CHECKED_BINDING_JSON NEW_OUTPUT_DIR

#[cfg(feature = "openwarrant-receipts")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use blut::framework::openwarrant_receipt::{
        DispatchBinding, execute_with_receipt, registry_digest,
    };
    use blut::framework::{ContentHash, ExecCtx, PlanSpec, Registry};
    use ed25519_dalek::SigningKey;
    use std::{fs, io::Read, path::PathBuf};

    let args: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(
        args.len() == 2,
        "expected binding path and new output directory"
    );
    let binding_path = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    anyhow::ensure!(
        binding_path.is_absolute() && output.is_absolute(),
        "absolute paths required"
    );
    let mut binding_bytes = Vec::new();
    fs::File::open(binding_path)?
        .take(32769)
        .read_to_end(&mut binding_bytes)?;
    anyhow::ensure!(binding_bytes.len() <= 32768, "binding byte budget");
    let binding: DispatchBinding = serde_json::from_slice(&binding_bytes)?;
    let mut registry = Registry::new();
    // Match the explicit registry assembled by openwarrant_verifier/CLI.
    blut::checks::register(&mut registry);
    blut::connectors::register(&mut registry);
    let expected_catalog = registry_digest(&registry)?;
    let plan: PlanSpec = serde_json::from_value(serde_json::json!({
        "name":"openwarrant-native-fixture", "nodes":[{
            "stage":"connector_object", "args":{
                "uri":"fixture://declared-object-not-fetched",
                "content_hash":ContentHash::of_bytes(b"fixture declared object").to_hex()
            }
        }], "edges":[]
    }))?;
    // Exclusive creation refuses replacing prior evidence.
    fs::create_dir(&output)?;
    fs::write(output.join("plan.json"), serde_json::to_vec(&plan)?)?;
    fs::write(output.join("binding.json"), binding_bytes)?;
    // This seed is deliberately public test material, never an operator secret.
    // The key remains in memory; only its PUBLIC half is written.
    let key = SigningKey::from_bytes(&[7; 32]);
    fs::write(
        output.join("public-key.bin"),
        key.verifying_key().as_bytes(),
    )?;
    let producer = std::env::current_exe()?;
    fs::copy(&producer, output.join("producer.original"))?;
    let run = execute_with_receipt(
        &plan,
        &registry,
        ExecCtx::new(output.join("job")),
        binding,
        &key,
    )
    .await?;
    run.result?;
    fs::write(
        output.join("receipt.json"),
        serde_json::to_vec(&run.envelope)?,
    )?;
    fs::write(
        output.join("fixture.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema":"blut/openwarrant-fixture/v1", "test_key_only":true,
            "expected_catalog_digest":expected_catalog, "assurance":"not-established"
        }))?,
    )?;
    println!("local CPU fixture executed; no authority or assurance");
    Ok(())
}

#[cfg(not(feature = "openwarrant-receipts"))]
fn main() {
    eprintln!("requires --features openwarrant-receipts; test fixture only");
    std::process::exit(2);
}
