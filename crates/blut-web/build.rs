// SPDX-License-Identifier: AGPL-3.0-or-later
// Stage the dashboard bundle for embedding (ADR 0083: one binary, zero deploy
// steps). If `ui/dist/` exists (built by `scripts/build_ui.sh`), it is copied
// into OUT_DIR and embedded verbatim; otherwise a self-describing stub page is
// embedded instead, so an API-only build (CI without a wasm toolchain, a quick
// `cargo build`) still serves "/" with instructions rather than a 404 — and
// the crate compiles either way (include_dir! needs the dir to exist).

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=ui/dist");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let staged = Path::new(&out).join("ui_dist");
    if staged.exists() {
        std::fs::remove_dir_all(&staged).expect("clear staged ui");
    }
    std::fs::create_dir_all(&staged).expect("create staged ui dir");

    let dist = Path::new("ui/dist");
    if dist.join("index.html").exists() {
        copy_tree(dist, &staged);
    } else {
        std::fs::write(
            staged.join("index.html"),
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>BLUT</title></head>\
             <body style=\"font-family:monospace;background:#101418;color:#cfd8dc;padding:2em\">\
             <h1>blut-web (API-only build)</h1>\
             <p>The dashboard bundle was not built into this binary. Build it with:</p>\
             <pre>bash scripts/build_ui.sh &amp;&amp; cargo build --release</pre>\
             <p>The JSON API is live under <code>/api</code> (see API docs).</p>\
             </body></html>",
        )
        .expect("write stub index");
    }
}

fn copy_tree(from: &Path, to: &Path) {
    for entry in std::fs::read_dir(from).expect("read ui/dist") {
        let entry = entry.expect("dir entry");
        let dest = to.join(entry.file_name());
        if entry.file_type().expect("file type").is_dir() {
            std::fs::create_dir_all(&dest).expect("mkdir");
            copy_tree(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), &dest).expect("copy asset");
        }
    }
}
