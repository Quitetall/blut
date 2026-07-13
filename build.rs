//! Build-time git stamp for the blut binary.
//!
//! Embeds the source tree's commit + dirty flag at compile time so the
//! running binary can (a) report exactly which commit it was built from
//! (`blut --version`) and, for opted-in local builds, (b) WARN at startup if
//! the source tree has since
//! moved past it — the "git pull, forgot to rebuild/reinstall, silently ran the
//! stale binary" trap (the in_ch fix needed `cargo install --force` to go live,
//! and a human who just `git pull`s would not have noticed). The runtime stale
//! check (cli.rs `warn_if_stale_binary`) compares the EMBEDDED hash here to the
//! source tree's LIVE HEAD; this script only stamps the embedded values.
//!
//! Always emits the three vars (falling back to `unknown` / empty) so the
//! `env!` reads in cli.rs compile even with no git / a tarball build.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());

    // Run a git subcommand inside the crate's source dir. `None` on any failure
    // (git absent, not a repo, non-zero exit, empty output) — a tarball build
    // or a CI sandbox without git degrades to `unknown`, never a build break.
    let git = |args: &[&str]| -> Option<String> {
        let mut full = vec!["-C", manifest.as_str()];
        full.extend_from_slice(args);
        let out = Command::new("git").args(&full).output().ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
        (!s.is_empty()).then_some(s)
    };

    let hash = git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    // Dirty = uncommitted changes to TRACKED files (ignore untracked so a fresh
    // build artifact or a new-but-unstaged scratch file does not flip the flag).
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);

    // Compose the display version here (folding in the dirty marker) so the flag
    // is actually CONSUMED — `--version` shows e.g. `0.1.0+a1b2c3d4e5f6` or
    // `…-dirty` for a non-reproducible build. BLUT_GIT_HASH stays separate
    // because the runtime stale check compares it to the live HEAD (no marker).
    let pkg = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0".into());
    let version = format!("{pkg}+{hash}{}", if dirty { "-dirty" } else { "" });

    // Absolute build paths are private host data and make registry artifacts
    // less reproducible. Local path installs may explicitly opt into the
    // stale-source probe; crates.io and ordinary source builds embed no path.
    println!("cargo:rerun-if-env-changed=BLUT_EMBED_SRC_DIR");
    let embedded_src = match env::var("BLUT_EMBED_SRC_DIR").as_deref() {
        Ok("1") => manifest.as_str(),
        _ => "",
    };
    println!("cargo:rustc-env=BLUT_GIT_HASH={hash}");
    println!("cargo:rustc-env=BLUT_VERSION={version}");
    println!("cargo:rustc-env=BLUT_SRC_DIR={embedded_src}");

    // Re-stamp the embedded hash when HEAD moves (a new commit / checkout).
    // Resolve the REAL HEAD + index paths via `git rev-parse --git-path` so this
    // works for a SUBMODULE or worktree (where `.git` is a file pointer, not a
    // dir) — blut is a submodule, so a naive `.git/HEAD` would never exist.
    let watch = |rel: Option<String>| {
        if let Some(p) = rel {
            let path = Path::new(&p);
            let abs = if path.is_absolute() {
                path.to_path_buf()
            } else {
                PathBuf::from(&manifest).join(path)
            };
            println!("cargo:rerun-if-changed={}", abs.display());
        }
    };
    // Re-stamp on any HEAD movement. SUBTLE (the stamp-lag bug): a `git commit`
    // on the current branch does NOT change HEAD's content — it stays
    // `ref: refs/heads/<branch>`; the commit rewrites the BRANCH ref + appends
    // the reflog. So watching HEAD alone catches only branch SWITCHES, not
    // commits, and the embedded hash silently lags the source. Watch every file
    // a HEAD movement can touch, resolved via `git rev-parse --git-path`
    // (submodule/worktree-correct, where `.git` is a file pointer):
    //   * HEAD                — a branch switch / detached-HEAD move
    //   * refs/heads/<branch> — a commit on the current branch (LOOSE ref)
    //   * packed-refs         — a commit when that branch ref is PACKED
    //   * logs/HEAD           — the reflog appends on EVERY HEAD move (the
    //                           belt-and-suspenders catch-all when enabled)
    // Over-triggering (e.g. a `git gc` repack) only re-stamps harmlessly; the
    // goal is to never MISS a move. Deliberately NOT watching `index`: staging
    // (`git add`) would recompile the crate on every staged change, and the
    // reflog already does not move on a bare `git add` (only on ref updates).
    // Also watch build.rs itself — emitting any rerun-if-changed overrides
    // cargo's default "re-run when the script changes".
    println!("cargo:rerun-if-changed=build.rs");
    watch(git(&["rev-parse", "--git-path", "HEAD"]));
    watch(git(&["rev-parse", "--git-path", "logs/HEAD"]));
    watch(git(&["rev-parse", "--git-path", "packed-refs"]));
    // The loose branch ref HEAD points at (None on a detached HEAD — the HEAD
    // watch above covers that case). `symbolic-ref --quiet` prints the full ref
    // name (`refs/heads/<branch>`) or exits non-zero ⇒ git() returns None.
    if let Some(branch_ref) = git(&["symbolic-ref", "--quiet", "HEAD"]) {
        watch(git(&["rev-parse", "--git-path", branch_ref.as_str()]));
    }
}
