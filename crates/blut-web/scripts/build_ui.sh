#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Build the Leptos dashboard (ui/) to wasm32 and stage it in ui/dist/, where
# blut-web's build.rs embeds it into the binary (ADR 0083: one binary, zero
# deploy steps). Needs: the wasm32-unknown-unknown target + wasm-bindgen-cli
# matching the ui crate's pinned wasm-bindgen version (the pin in
# ui/Cargo.toml exists precisely so this pairing can't drift).
#
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version <pinned>
#   bash scripts/build_ui.sh && cargo build --release
set -euo pipefail
cd "$(dirname "$0")/../ui"

cargo build --release --target wasm32-unknown-unknown
rm -rf dist && mkdir -p dist
wasm-bindgen --target web --no-typescript --out-dir dist --out-name blut_web_ui \
  target/wasm32-unknown-unknown/release/blut-web-ui.wasm
cp index.html dist/index.html
echo "ui bundle staged: $(du -sh dist | cut -f1) in ui/dist/"
