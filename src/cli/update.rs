// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Brian Lam
//! `blut update` — is the installed binary behind what is published?
//!
//! The engine does not speak HTTP (ADR 0034 keeps network clients out of this
//! crate), so this asks the tool that already owns registry access: it shells
//! out to `cargo info <pkg>` and reads the version line. No new dependency, no
//! socket, and the user's existing cargo auth/proxy/mirror configuration is
//! honoured for free.
//!
//! What it will NOT do: reach out on its own. A CLI that phones a registry
//! during unrelated commands is a surprise in an operator tool, so the check
//! happens only when someone types `blut update`, and an actual install only
//! with `--yes`. Everything else prints the exact command and stops.

use super::*;
use std::process::Command as Proc;

/// Where the running binary came from, as recorded by cargo itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Origin {
    /// `cargo install <pkg>` — upgradeable from the registry.
    Registry { pkg: String, version: String },
    /// `cargo install --path <dir>` — a local checkout; the registry says
    /// nothing useful about it, so the advice is to rebuild from source.
    Path { pkg: String, dir: String },
}

/// Parse `$CARGO_HOME/.crates.toml` for the entry that installed binary `bin`.
///
/// Lines are one per installed package and look like:
///   `"blut-lamquant 1.0.0 (registry+https://github.com/rust-lang/…)" = ["blut"]`
///   `"blut-lamquant 1.0.0 (path+file:///abs/dir)" = ["blut", "lqt"]`
///
/// Pure, so the format assumptions are testable without a cargo home. An
/// unrecognised shape yields `None` (advice-only) rather than a guess: telling
/// someone to reinstall the wrong package is worse than saying nothing.
pub(super) fn parse_installed_origin(crates_toml: &str, bin: &str) -> Option<Origin> {
    let needle = format!("\"{bin}\"");
    for line in crates_toml.lines() {
        let line = line.trim();
        let Some((key, vals)) = line.split_once('=') else {
            continue;
        };
        // Match `bin` as a whole quoted list element: "blut" must not hit "blutx".
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
        let inner = key.trim().trim_start_matches('"');
        let mut parts = inner.split_whitespace();
        let pkg = parts.next()?.to_string();
        let version = parts.next()?.to_string();
        let source = key.split_once('(')?.1;
        if let Some(rest) = source.split_once("path+file://") {
            // Cargo records the path as a URL: a space is `%20`. Handing that
            // back verbatim printed a `cargo install --path` that cannot work.
            let dir = percent_decode(rest.1.split(')').next()?.trim())?;
            if dir.is_empty() {
                return None;
            }
            return Some(Origin::Path { pkg, dir });
        }
        if source.starts_with("registry+") {
            return Some(Origin::Registry { pkg, version });
        }
        // git+ installs: neither a registry upgrade nor a local rebuild.
        return None;
    }
    None
}

/// Decode the `%XX` escapes of a `file://` URL path. `None` on a malformed
/// escape or a result that is not UTF-8, so a mangled ledger yields no advice
/// rather than a wrong path.
fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// The feature selection the binary `bin` was installed with, read from
/// cargo's `.crates2.json`, as `cargo install` arguments. Reinstalling without
/// them silently drops the features: a `--features cloud` install would come
/// back from `blut update --yes` without `blut cloud`. Unknown or unreadable
/// input yields no flags.
pub(super) fn parse_install_flags(crates2_json: &str, bin: &str) -> Vec<String> {
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(crates2_json) else {
        return Vec::new();
    };
    let Some(installs) = doc.get("installs").and_then(|i| i.as_object()) else {
        return Vec::new();
    };
    let Some(entry) = installs.values().find(|entry| {
        entry
            .get("bins")
            .and_then(|b| b.as_array())
            .is_some_and(|bins| bins.iter().any(|b| b.as_str() == Some(bin)))
    }) else {
        return Vec::new();
    };
    let mut flags = Vec::new();
    if entry.get("all_features").and_then(|v| v.as_bool()) == Some(true) {
        flags.push("--all-features".to_string());
    }
    if entry.get("no_default_features").and_then(|v| v.as_bool()) == Some(true) {
        flags.push("--no-default-features".to_string());
    }
    let features: Vec<&str> = entry
        .get("features")
        .and_then(|f| f.as_array())
        .map(|f| f.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if !features.is_empty() {
        flags.push("--features".to_string());
        flags.push(features.join(","));
    }
    flags
}

/// Render an argv for a POSIX shell, quoting only the words that need it, so
/// a printed command can be pasted as-is.
pub(super) fn shell_join(argv: &[String]) -> String {
    argv.iter()
        .map(|word| {
            let plain = !word.is_empty()
                && word
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_./=:,+@%".contains(c));
            if plain {
                word.clone()
            } else {
                format!("'{}'", word.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Pull the newest published version out of `cargo info` output. The command
/// prints a block of `key: value` lines; only the first bare `version:` counts
/// (later lines can carry `rust-version:`, which must not match).
///
/// When the newest release needs a newer Rust than the local toolchain, cargo
/// shows the newest one the toolchain can build and appends the true newest:
/// `version: 0.17.3 (latest 0.20.0)`. The question here is "what is
/// published", so the `latest` value wins.
pub(super) fn parse_cargo_info_version(stdout: &str) -> Option<String> {
    stdout.lines().find_map(|l| {
        let v = l.trim().strip_prefix("version:")?.trim();
        let v = match v.split_once("(latest ") {
            Some((_, rest)) => rest.trim_end().trim_end_matches(')'),
            None => v,
        };
        Some(v.trim().to_string())
    })
}

/// What the user should do, decided from two version strings.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    UpToDate,
    Behind {
        latest: String,
    },
    /// The installed build is NEWER than anything published — normal while a
    /// release is being prepared. Saying "up to date" would be a lie and
    /// "behind" would be wrong, so it gets its own answer.
    Ahead {
        latest: String,
    },
    /// The installed build is a stable release and the only newer version is
    /// a prerelease. A stable install is never moved onto an alpha.
    NewerPrereleaseOnly {
        latest: String,
    },
}

/// Compare with real semver ordering, because string comparison gets
/// prereleases exactly backwards: `"0.2.0-alpha.1" < "0.2.0"` is true in semver
/// and false lexically, and this crate ships alpha versions today.
pub(super) fn verdict(installed: &str, latest: &str) -> Result<Verdict> {
    let cur = semver::Version::parse(installed.trim())
        .with_context(|| format!("installed version {installed:?} is not semver"))?;
    let new = semver::Version::parse(latest.trim())
        .with_context(|| format!("published version {latest:?} is not semver"))?;
    Ok(match cur.cmp(&new) {
        std::cmp::Ordering::Less if cur.pre.is_empty() && !new.pre.is_empty() => {
            Verdict::NewerPrereleaseOnly {
                latest: new.to_string(),
            }
        }
        std::cmp::Ordering::Less => Verdict::Behind {
            latest: new.to_string(),
        },
        std::cmp::Ordering::Equal => Verdict::UpToDate,
        std::cmp::Ordering::Greater => Verdict::Ahead {
            latest: new.to_string(),
        },
    })
}

/// The command that performs the upgrade. `--locked` is not optional here: it
/// installs the dependency versions the release was tested with instead of
/// whatever resolves today.
pub(super) fn upgrade_argv(pkg: &str, version: &str, install_flags: &[String]) -> Vec<String> {
    let mut argv: Vec<String> = vec![
        "cargo".into(),
        "install".into(),
        pkg.into(),
        "--version".into(),
        version.into(),
        "--locked".into(),
        "--force".into(),
    ];
    argv.extend(install_flags.iter().cloned());
    argv
}

/// The command that rebuilds a `cargo install --path` binary from its checkout.
pub(super) fn path_reinstall_argv(dir: &str, install_flags: &[String]) -> Vec<String> {
    let mut argv: Vec<String> = vec![
        "cargo".into(),
        "install".into(),
        "--path".into(),
        dir.into(),
        "--force".into(),
    ];
    argv.extend(install_flags.iter().cloned());
    argv
}

/// Ask the registry what the newest version of `pkg` is.
///
/// Runs from a scratch directory ON PURPOSE. Inside a workspace that contains
/// `pkg` as a path member, `cargo info pkg` answers from the local member and
/// reports a version that was never published — the exact defect that made a
/// release guard in this repository's CI unable to take its false branch.
fn published_version(pkg: &str) -> Result<String> {
    // A fresh, unpredictable directory: a reused or guessable one could be
    // pre-seeded with a `.cargo/config.toml` that points at another registry.
    let scratch = tempfile::Builder::new()
        .prefix("blut-update-")
        .tempdir()
        .context("create scratch dir for `cargo info`")?;
    let out = Proc::new("cargo")
        .args(["info", pkg])
        .current_dir(scratch.path())
        .output()
        .context("run `cargo info` (is cargo on PATH?)")?;
    drop(scratch);
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow!(
            "`cargo info {pkg}` failed ({}): {}",
            out.status,
            err.trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    parse_cargo_info_version(&stdout)
        .ok_or_else(|| anyhow!("`cargo info {pkg}` printed no version line"))
}

/// Resolve the origin of the running binary from cargo's own install ledger,
/// with the feature flags it was installed with.
fn installed_origin() -> Result<(Origin, Vec<String>)> {
    let exe = std::env::current_exe().context("resolve current executable")?;
    let bin = exe
        .file_stem()
        .ok_or_else(|| anyhow!("executable has no file stem"))?
        .to_string_lossy()
        .to_string();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")))
        .ok_or_else(|| anyhow!("neither CARGO_HOME nor HOME is set"))?;
    let ledger = cargo_home.join(".crates.toml");
    let text =
        std::fs::read_to_string(&ledger).with_context(|| format!("read {}", ledger.display()))?;
    let origin = parse_installed_origin(&text, &bin).ok_or_else(|| {
        anyhow!(
            "{} lists no registry or path install for the binary {bin:?}; \
             nothing to compare against",
            ledger.display()
        )
    })?;
    let flags = std::fs::read_to_string(cargo_home.join(".crates2.json"))
        .map(|json| parse_install_flags(&json, &bin))
        .unwrap_or_default();
    Ok((origin, flags))
}

/// `blut update` — report, and on `--yes` perform, an upgrade of the binary
/// cargo's install ledger records (not necessarily the one running). Exit
/// status 0 means a verdict was reached, "behind" included; non-zero means no
/// verdict could be reached or the upgrade failed.
pub(super) fn run_update(check_only: bool, assume_yes: bool) -> Result<()> {
    let (origin, flags) = installed_origin()?;
    match &origin {
        Origin::Path { pkg, dir } => {
            println!("{pkg} was installed from a local checkout ({dir}).");
            println!(
                "The registry cannot speak for it. Update the checkout and reinstall:\n  {}",
                shell_join(&path_reinstall_argv(dir, &flags))
            );
            Ok(())
        }
        Origin::Registry { pkg, version } => {
            let latest = published_version(pkg)?;
            match verdict(version, &latest)? {
                Verdict::UpToDate => {
                    println!("{pkg} {version} is the newest published version.");
                    Ok(())
                }
                Verdict::Ahead { latest } => {
                    println!(
                        "{pkg} {version} is installed; the registry's newest is {latest}. \
                         Nothing to update."
                    );
                    Ok(())
                }
                Verdict::NewerPrereleaseOnly { latest } => {
                    println!(
                        "{pkg} {version} is installed; the registry's newest is the \
                         prerelease {latest}. A stable install is not moved onto a \
                         prerelease; install it explicitly with:\n  {}",
                        shell_join(&upgrade_argv(pkg, &latest, &flags))
                    );
                    Ok(())
                }
                Verdict::Behind { latest } => {
                    let argv = upgrade_argv(pkg, &latest, &flags);
                    println!("{pkg} {version} is installed; {latest} is published.");
                    if check_only || !assume_yes {
                        println!("Update with:\n  {}", shell_join(&argv));
                        if !check_only {
                            println!("Or re-run with --yes to do it now.");
                        }
                        return Ok(());
                    }
                    println!("Running: {}", shell_join(&argv));
                    let status = Proc::new(&argv[0])
                        .args(&argv[1..])
                        .status()
                        .context("spawn cargo install")?;
                    if !status.success() {
                        return Err(anyhow!("cargo install failed ({status})"));
                    }
                    println!("{pkg} updated to {latest}.");
                    Ok(())
                }
            }
        }
    }
}

#[cfg(test)]
mod update_tests {
    use super::*;

    const LEDGER: &str = r#"[v1]
"blut-lamquant 0.2.0-alpha.1 (registry+https://github.com/rust-lang/crates.io-index)" = ["blut"]
"some-multi 0.1.0 (path+file:///home/u/multi)" = ["foo", "blutx", "bar"]
"local-cook 1.2.3 (path+file:///home/u/cookbooks/lamquant)" = ["lqt"]
"gitpkg 0.1.0 (git+https://example.invalid/x)" = ["gitbin"]
"#;

    #[test]
    fn reads_a_registry_install() {
        assert_eq!(
            parse_installed_origin(LEDGER, "blut"),
            Some(Origin::Registry {
                pkg: "blut-lamquant".into(),
                version: "0.2.0-alpha.1".into()
            })
        );
    }

    #[test]
    fn reads_a_path_install() {
        assert_eq!(
            parse_installed_origin(LEDGER, "lqt"),
            Some(Origin::Path {
                pkg: "local-cook".into(),
                dir: "/home/u/cookbooks/lamquant".into()
            })
        );
    }

    #[test]
    fn a_git_install_is_neither() {
        assert_eq!(parse_installed_origin(LEDGER, "gitbin"), None);
    }

    #[test]
    fn does_not_match_a_binary_by_substring() {
        // "blut" must not resolve through the entry that installs "blutx".
        let Some(Origin::Registry { pkg, .. }) = parse_installed_origin(LEDGER, "blut") else {
            panic!("blut resolves to the registry entry, not the one installing blutx");
        };
        assert_eq!(pkg, "blut-lamquant");
        assert_eq!(parse_installed_origin(LEDGER, "nope"), None);
    }

    #[test]
    fn version_line_is_the_package_version_not_the_rust_version() {
        let out = "blut #training\ndescription: engine\nversion: 0.3.1\nlicense: AGPL\nrust-version: 1.88\n";
        assert_eq!(parse_cargo_info_version(out).as_deref(), Some("0.3.1"));
    }

    #[test]
    fn a_toolchain_limited_version_reports_the_true_latest() {
        let out = "bevy\nversion: 0.17.3 (latest 0.20.0)\nrust-version: 1.85\n";
        assert_eq!(parse_cargo_info_version(out).as_deref(), Some("0.20.0"));
        let out = "wasmtime\nversion: 38.0.4 (latest 50.0.0-rc.1)\n";
        assert_eq!(
            parse_cargo_info_version(out).as_deref(),
            Some("50.0.0-rc.1")
        );
    }

    #[test]
    fn a_stable_install_is_not_offered_a_prerelease_but_a_prerelease_is() {
        assert_eq!(
            verdict("0.2.0", "0.3.0-alpha.1").unwrap(),
            Verdict::NewerPrereleaseOnly {
                latest: "0.3.0-alpha.1".into()
            }
        );
        assert_eq!(
            verdict("0.2.0-alpha.2", "0.2.0-alpha.10").unwrap(),
            Verdict::Behind {
                latest: "0.2.0-alpha.10".into()
            }
        );
    }

    #[test]
    fn missing_version_line_is_none() {
        assert_eq!(parse_cargo_info_version("no versions here\n"), None);
    }

    #[test]
    fn prerelease_ordering_follows_semver_not_string_compare() {
        // The string comparison this replaces gets exactly this case wrong:
        // lexically "0.2.0-alpha.1" > "0.2.0".
        assert!("0.2.0-alpha.1" > "0.2.0");
        assert_eq!(
            verdict("0.2.0-alpha.1", "0.2.0").unwrap(),
            Verdict::Behind {
                latest: "0.2.0".into()
            }
        );
    }

    #[test]
    fn equal_versions_are_up_to_date_and_newer_local_is_ahead() {
        assert_eq!(verdict("1.0.0", "1.0.0").unwrap(), Verdict::UpToDate);
        assert_eq!(
            verdict("1.1.0", "1.0.0").unwrap(),
            Verdict::Ahead {
                latest: "1.0.0".into()
            }
        );
    }

    #[test]
    fn a_non_semver_version_is_an_error_not_a_silent_up_to_date() {
        assert!(verdict("not-a-version", "1.0.0").is_err());
        assert!(verdict("1.0.0", "").is_err());
    }

    #[test]
    fn the_verb_is_built_in_and_not_swallowed_by_the_external_dispatch() {
        // `Update` sits before the `external_subcommand` catch-all, which would
        // otherwise try to exec a `blut-update` binary from PATH.
        let cli = Cli::try_parse_from(["blut", "update", "--check"]).expect("update parses");
        assert!(matches!(
            cli.command,
            Some(Command::Update {
                check: true,
                yes: false
            })
        ));
        let cli = Cli::try_parse_from(["blut", "update", "--yes"]).expect("update --yes parses");
        assert!(matches!(
            cli.command,
            Some(Command::Update {
                check: false,
                yes: true
            })
        ));
    }

    #[test]
    fn the_upgrade_command_pins_the_version_and_the_lockfile() {
        assert_eq!(
            upgrade_argv("blut-lamquant", "0.3.0", &[]),
            vec![
                "cargo",
                "install",
                "blut-lamquant",
                "--version",
                "0.3.0",
                "--locked",
                "--force"
            ]
        );
    }

    /// Cargo writes a path install as a URL (`my%20dir`). The advice printed
    /// that verbatim, so the suggested `cargo install --path` failed.
    #[test]
    fn a_path_install_with_a_space_prints_a_command_that_works() {
        let ledger = "[v1]\n\"tinyx 0.1.0 (path+file:///home/u/my%20dir/tiny)\" = [\"tinyx\"]\n";
        let Some(Origin::Path { dir, .. }) = parse_installed_origin(ledger, "tinyx") else {
            panic!("expected a path install");
        };
        assert_eq!(dir, "/home/u/my dir/tiny");
        assert_eq!(
            shell_join(&path_reinstall_argv(&dir, &[])),
            "cargo install --path '/home/u/my dir/tiny' --force"
        );
    }

    /// `blut update --yes` reinstalled without the features the binary was
    /// installed with, silently dropping e.g. `blut cloud`.
    #[test]
    fn the_upgrade_keeps_the_installed_feature_selection() {
        let crates2 = r#"{"installs":{"blut-lamquant 0.2.0-alpha.1 (registry+https://github.com/rust-lang/crates.io-index)":{"version_req":null,"bins":["blut"],"features":["cloud","p2p"],"all_features":false,"no_default_features":true,"profile":"release"}}}"#;
        let flags = parse_install_flags(crates2, "blut");
        assert_eq!(
            upgrade_argv("blut-lamquant", "0.2.0-alpha.2", &flags),
            vec![
                "cargo",
                "install",
                "blut-lamquant",
                "--version",
                "0.2.0-alpha.2",
                "--locked",
                "--force",
                "--no-default-features",
                "--features",
                "cloud,p2p"
            ]
        );
        assert!(parse_install_flags(crates2, "other").is_empty());
        assert!(parse_install_flags("not json", "blut").is_empty());
    }

    #[test]
    fn a_malformed_percent_escape_yields_no_advice() {
        let ledger = "[v1]\n\"x 0.1.0 (path+file:///home/u/bad%2)\" = [\"x\"]\n";
        assert_eq!(parse_installed_origin(ledger, "x"), None);
    }
}
