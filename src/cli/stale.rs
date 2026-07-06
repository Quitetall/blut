// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! Stale-binary detection + auto-rebuild for the installed `blut` bin.
//!
//! Split out of the former single-file `cli.rs`; mounted as a child of
//! the `cli` module and glob-imported back, so `super::*` (the shared
//! imports, the other submodules' items, and the mod.rs helpers)
//! resolves exactly as it did inline.

/// Warn (once, at startup) if the running binary was built from a DIFFERENT
/// commit than its source tree's CURRENT HEAD — the "git pull, forgot to
/// rebuild/reinstall, silently ran the stale binary" trap. The in_ch /
/// warm-containment never-OOM fixes only go live after a rebuild; a human who
/// `git pull`s and runs the old `~/.cargo/bin/blut` would otherwise get the
/// stale admission/footprint/containment logic with no signal.
///
/// build.rs stamps the build-time hash (`BLUT_GIT_HASH`) + the source dir
/// (`BLUT_SRC_DIR`); this re-resolves that dir's live HEAD at RUNTIME and warns
/// on mismatch. SILENT when up to date, when the source tree is gone (binary
/// copied off the build box), when git is unavailable, or when the build was
/// not stamped (`unknown`) — a missing signal must never become noise or a
/// false alarm.
/// The result of the staleness probe: the binary's build-time hash, its source
/// tree's CURRENT short HEAD, and the stamped source dir. Present only when
/// there IS a trustworthy mismatch.
pub(super) struct StaleInfo {
    built: String,
    live: String,
    src: String,
}

/// Probe whether the running binary is stale (built from a different commit
/// than its stamped source tree's live HEAD). Returns `None` — stay silent — on
/// version/help, an unstamped build, a vanished source tree, or no git: a
/// missing signal must never become noise or a false alarm.
pub(super) fn detect_stale_binary() -> Option<StaleInfo> {
    // `--version` / `--help` should be fast and clean: skip the git probe (clap
    // exits during parse, so a stale notice would just be stderr noise).
    if std::env::args().any(|a| matches!(a.as_str(), "--version" | "-V" | "--help" | "-h")) {
        return None;
    }
    let built = env!("BLUT_GIT_HASH");
    let src = env!("BLUT_SRC_DIR");
    if built == "unknown" || src.is_empty() {
        return None;
    }
    let live = std::process::Command::new("git")
        .args(["-C", src, "rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())?;
    if live == built {
        return None;
    }
    Some(StaleInfo {
        built: built.to_string(),
        live,
        src: src.to_string(),
    })
}

/// Warn (once, at startup) if the running binary was built from a DIFFERENT
/// commit than its source tree's CURRENT HEAD — the "git pull, forgot to
/// rebuild/reinstall, silently ran the stale binary" trap. The in_ch /
/// warm-containment never-OOM fixes only go live after a rebuild; a human who
/// `git pull`s and runs the old `~/.cargo/bin/blut` would otherwise get the
/// stale admission/footprint/containment logic with no signal.
///
/// build.rs stamps the build-time hash (`BLUT_GIT_HASH`) + the source dir
/// (`BLUT_SRC_DIR`); [`detect_stale_binary`] re-resolves that dir's live HEAD at
/// RUNTIME. When `BLUT_AUTO_REBUILD=1` (opt-in, ADR 0071 A4) a stale binary is
/// rebuilt + re-exec'd instead of merely warned; default OFF (warn-only) so
/// there are no surprise rebuilds.
pub(super) fn warn_if_stale_binary() {
    let Some(info) = detect_stale_binary() else {
        return;
    };
    // Opt-in auto-rebuild. On a successful rebuild this re-execs the fresh
    // binary and never returns; otherwise it falls through to the warning.
    maybe_auto_rebuild(&info);
    // Deliberately NOT a `--path` hint: the `blut` binary is built from the
    // cookbook crate (blut-lamquant), not this engine crate (BLUT_SRC_DIR), so a
    // specific `--path` would point at the wrong directory. Keep it generic —
    // the operator knows how they installed.
    tracing::warn!(
        "blut binary is STALE: built from {} but its source tree ({}) is now \
         at {} — this run uses OLD code (admission / footprint / containment logic \
         may predate the source). Rebuild + reinstall (`cargo install --force`, or \
         `cargo build` for a local checkout), or set BLUT_AUTO_REBUILD=1 to do it \
         automatically.",
        info.built,
        info.src,
        info.live
    );
}

/// Parse a cargo `.crates.toml` for the package that installed binary `bin`,
/// returning the local crate DIR if it was a `path+file://` install (the only
/// case we can rebuild from). Lines look like:
///   "blut-lamquant 1.0.0 (path+file:///abs/dir)" = ["blut", ...]
/// Returns `None` for a registry/git install (no local dir to `--path` at) or
/// when `bin` isn't an installed binary. Pure (testable) — no IO. Assumes
/// cargo's single-line `.crates.toml` format; an unexpected/evolved format
/// falls through to `None` (warn-only), never a wrong dir.
pub(super) fn crate_dir_for_installed_bin(crates_toml: &str, bin: &str) -> Option<String> {
    let needle = format!("\"{bin}\"");
    for line in crates_toml.lines() {
        let line = line.trim();
        // The value side lists the binaries this package installed.
        let Some((key, vals)) = line.split_once('=') else {
            continue;
        };
        // Match the bin as a quoted list element (avoid a substring false-hit on
        // e.g. "blutx" when looking for "blut").
        let lists_bin = vals.split(',').any(|tok| {
            tok.trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim()
                == needle
        });
        if !lists_bin {
            continue;
        }
        // Extract the `path+file://DIR` source from the key's `(...)`.
        let src = key.split_once("(path+file://")?.1;
        let dir = src.split(')').next()?.trim();
        if !dir.is_empty() {
            return Some(dir.to_string());
        }
    }
    None
}

/// Resolve how to rebuild the installed `blut` binary. Prefers an explicit
/// `BLUT_REBUILD_CMD` (run via `sh -c` — covers a local `cargo build` checkout,
/// pipes, `&&` chains); otherwise detects a `cargo install --path <cookbook-dir>`
/// from `$CARGO_HOME/.crates.toml`. `None` ⇒ can't determine the target → warn
/// only.
///
/// TRUST MODEL: `BLUT_REBUILD_CMD` is executed verbatim by a shell, so it is an
/// arbitrary-command surface. It is opt-in (only consulted when both it AND
/// `BLUT_AUTO_REBUILD=1` are set) and the value comes from the invoking user's
/// OWN environment — a user who can set it can already run any command, so this
/// is not a privilege escalation in normal (non-setuid) use. The `sh -c` form
/// is deliberate: the value must support shell features (a local checkout often
/// needs `cargo build --release && cp …`). Do NOT run blut setuid / as another
/// user with an attacker-controlled environment.
pub(super) fn resolve_rebuild_command() -> Option<Vec<String>> {
    if let Some(cmd) = std::env::var_os("BLUT_REBUILD_CMD") {
        let cmd = cmd.to_string_lossy().to_string();
        if !cmd.trim().is_empty() {
            // Shell-exec surface — see the TRUST MODEL note above.
            return Some(vec!["sh".into(), "-c".into(), cmd]);
        }
    }
    // Detect the path-install source. The bin name is this exe's file stem.
    let exe = std::env::current_exe().ok()?;
    let bin = exe.file_stem()?.to_string_lossy().to_string();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cargo")))?;
    let crates_toml = std::fs::read_to_string(cargo_home.join(".crates.toml")).ok()?;
    let dir = crate_dir_for_installed_bin(&crates_toml, &bin)?;
    Some(vec![
        "cargo".into(),
        "install".into(),
        "--path".into(),
        dir,
        "--force".into(),
    ])
}

/// Opt-in (`BLUT_AUTO_REBUILD=1`) rebuild + re-exec of a stale binary (ADR 0071
/// A4). On a successful rebuild this re-execs the fresh binary with the same
/// args and DOES NOT RETURN. Returns (falling through to the warning) when:
/// the opt-in is off, a rebuild already ran this chain (loop guard), the target
/// can't be resolved, or the rebuild/exec failed.
pub(super) fn maybe_auto_rebuild(info: &StaleInfo) {
    if std::env::var_os("BLUT_AUTO_REBUILD").is_none() {
        return;
    }
    // Loop guard: we rebuild+re-exec at most once per invocation chain. If the
    // child still reads stale (rebuild didn't move the hash — uncommitted work,
    // src moved again), don't spin.
    if std::env::var_os("BLUT_AUTO_REBUILD_DONE").is_some() {
        tracing::warn!(
            "auto-rebuild already ran but blut is still stale ({} ≠ {}); not retrying — \
             rebuild manually (uncommitted changes in {}?).",
            info.built,
            info.live,
            info.src
        );
        return;
    }
    let Some(cmd) = resolve_rebuild_command() else {
        tracing::warn!(
            "BLUT_AUTO_REBUILD=1 but the install source can't be determined; set \
             BLUT_REBUILD_CMD='<rebuild command>' or rebuild manually."
        );
        return;
    };
    tracing::warn!(
        "blut stale ({} → {}); BLUT_AUTO_REBUILD=1 → rebuilding via `{}` (a from-source \
         `cargo install` may take a few minutes — not a hang) ...",
        info.built,
        info.live,
        cmd.join(" ")
    );
    // cmd is ["cargo","install",...] (detected) or ["sh","-c",<BLUT_REBUILD_CMD>].
    let status = std::process::Command::new(&cmd[0]).args(&cmd[1..]).status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            tracing::error!("auto-rebuild failed (exit {s}); continuing on the STALE binary.");
            return;
        }
        Err(e) => {
            tracing::error!(
                "auto-rebuild could not start (`{}`: {e}); continuing STALE.",
                cmd[0]
            );
            return;
        }
    }
    // Re-exec the freshly-installed binary with the original args. current_exe()
    // returns the PATH (not a pinned inode/vnode) on both Linux (/proc/self/exe)
    // and macOS, so exec() re-resolves it to the NEW content cargo wrote in
    // place under --force. The DONE sentinel arms the loop guard in the child.
    let Ok(exe) = std::env::current_exe() else {
        tracing::warn!("rebuilt OK but current_exe() unknown; re-run blut to use fresh code.");
        return;
    };
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    tracing::info!(
        "auto-rebuild OK — re-exec {} with fresh code.",
        exe.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // exec replaces this process image; on success it never returns. If it
        // DOES return, it failed — fall through to the warning.
        let err = std::process::Command::new(&exe)
            .args(&args)
            .env("BLUT_AUTO_REBUILD_DONE", "1")
            .exec();
        tracing::error!("re-exec failed ({err}); continuing on the STALE (pre-rebuild) process.");
    }
    #[cfg(not(unix))]
    {
        // No exec(); spawn the fresh binary, forward its exit, and stop this one.
        match std::process::Command::new(&exe)
            .args(&args)
            .env("BLUT_AUTO_REBUILD_DONE", "1")
            .status()
        {
            // exit() skips Drop/atexit — fine here: we're mid-startup (this runs
            // before the async runtime does real work), nothing to flush.
            Ok(s) => std::process::exit(s.code().unwrap_or(0)),
            Err(e) => tracing::error!("re-spawn failed ({e}); continuing STALE."),
        }
    }
}

#[cfg(test)]
mod stale_rebuild_tests {
    use super::crate_dir_for_installed_bin;

    const SAMPLE: &str = r#"[v1]
"blut-lamquant 1.0.0 (path+file:///mnt/4tb/LamQuant/training/cookbooks/lamquant)" = ["blut"]
"ripgrep 14.0.0 (registry+https://github.com/rust-lang/crates.io-index)" = ["rg"]
"some-multi 0.1.0 (path+file:///home/u/multi)" = ["foo", "blutx", "bar"]
"#;

    #[test]
    fn finds_path_install_dir_for_bin() {
        assert_eq!(
            crate_dir_for_installed_bin(SAMPLE, "blut").as_deref(),
            Some("/mnt/4tb/LamQuant/training/cookbooks/lamquant"),
        );
    }

    #[test]
    fn ignores_registry_install_and_substring_binaries() {
        // `rg` is a registry install → no local dir to --path at.
        assert_eq!(crate_dir_for_installed_bin(SAMPLE, "rg"), None);
        // "blut" must NOT match the "blutx" element (quoted-token compare).
        assert_eq!(
            crate_dir_for_installed_bin(SAMPLE, "blutx").as_deref(),
            Some("/home/u/multi"),
        );
        // an unknown bin → None.
        assert_eq!(crate_dir_for_installed_bin(SAMPLE, "nope"), None);
    }

    // ── ADR 0072 A7: never-panics on adversarial input (property-based) ──
    //
    // The two tests above pin correctness on well-formed `.crates.toml`
    // input. This is the complementary property: the doc comment on
    // `crate_dir_for_installed_bin` promises "an unexpected/evolved format
    // falls through to `None`... never a wrong dir" — the proptest below
    // widens that to "never PANICS", fed input that is emphatically NOT
    // well-formed `.crates.toml` (arbitrary Unicode, and separately, random
    // garbage built only from the characters the real format uses — quotes,
    // parens, brackets, `=`, `,`, newlines — to bias toward the parser's
    // internal branches without being valid).
    proptest::proptest! {
        #[test]
        fn crate_dir_for_installed_bin_never_panics(
            crates_toml in proptest::prop_oneof![
                proptest::prelude::any::<String>(),
                "[a-zA-Z0-9_.:/()\\[\\],=+\"'\n -]{0,400}",
            ],
            bin in proptest::prop_oneof![
                proptest::prelude::any::<String>(),
                "[a-zA-Z0-9_-]{0,40}",
            ],
        ) {
            // Only claim: this must run to completion (no panic — no OOB
            // index/slice, no unwrap/expect, no arithmetic overflow). The
            // return value is intentionally not asserted on here; happy-path
            // shape is already pinned above.
            let _ = crate_dir_for_installed_bin(&crates_toml, &bin);
        }
    }
}
