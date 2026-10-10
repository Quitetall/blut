// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Read-only verifier CLI example. This empty cookbook registry supports only
//! BLUT's built-ins. Real cookbook binaries use their own registered catalog.
#[cfg(feature = "openwarrant-receipts")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    blut::cli::run(blut::framework::Registry::new()).await
}

#[cfg(not(feature = "openwarrant-receipts"))]
fn main() {
    eprintln!("Enable --features openwarrant-receipts for this example.");
    std::process::exit(2);
}
