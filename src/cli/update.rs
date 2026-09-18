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
            let dir = rest.1.split(')').next()?.trim().to_string();
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

/// Pull the version out of `cargo info` output. The command prints a block of
/// `key: value` lines; only the first bare `version:` is the published one
/// (later lines can carry `rust-version:`, which must not match).
pub(super) fn parse_cargo_info_version(stdout: &str) -> Option<String> {
    stdout.lines().find_map(|l| {
        let l = l.trim();
        l.strip_prefix("version:").map(|v| v.trim().to_string())
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
pub(super) fn upgrade_argv(pkg: &str, version: &str) -> Vec<String> {
    vec![
        "cargo".into(),
        "install".into(),
        pkg.into(),
        "--version".into(),
        version.into(),
        "--locked".into(),
        "--force".into(),
    ]
}

/// Ask the registry what the newest version of `pkg` is.
///
/// Runs from a scratch directory ON PURPOSE. Inside a workspace that contains
/// `pkg` as a path member, `cargo info pkg` answers from the local member and
/// reports a version that was never published — the exact defect that made a
/// release guard in this repository's CI unable to take its false branch.
fn published_version(pkg: &str) -> Result<String> {
    let scratch = std::env::temp_dir().join(format!("blut-update-{}", std::process::id()));
    std::fs::create_dir_all(&scratch)
        .with_context(|| format!("create scratch dir {}", scratch.display()))?;
    let out = Proc::new("cargo")
        .args(["info", pkg])
        .current_dir(&scratch)
        .output()
        .context("run `cargo info` (is cargo on PATH?)")?;
    let _ = std::fs::remove_dir_all(&scratch);
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

/// Resolve the origin of the running binary from cargo's own install ledger.
fn installed_origin() -> Result<Origin> {
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
    parse_installed_origin(&text, &bin).ok_or_else(|| {
        anyhow!(
            "{} lists no registry or path install for the binary {bin:?}; \
             nothing to compare against",
            ledger.display()
        )
    })
}

/// `blut update` — report, and on `--yes` perform, an upgrade of the installed
/// binary. Exit status is a verdict, never a crash: 0 when nothing is owed.
pub(super) fn run_update(check_only: bool, assume_yes: bool) -> Result<()> {
    let origin = installed_origin()?;
    match &origin {
        Origin::Path { pkg, dir } => {
            println!("{pkg} was installed from a local checkout ({dir}).");
            println!(
                "The registry cannot speak for it. Update the checkout and reinstall:\n  \
                 cargo install --path {dir} --force"
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
                Verdict::Behind { latest } => {
                    let argv = upgrade_argv(pkg, &latest);
                    println!("{pkg} {version} is installed; {latest} is published.");
                    if check_only || !assume_yes {
                        println!("Update with:\n  {}", argv.join(" "));
                        if !check_only {
                            println!("Or re-run with --yes to do it now.");
                        }
                        return Ok(());
                    }
                    println!("Running: {}", argv.join(" "));
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
            upgrade_argv("blut-lamquant", "0.3.0"),
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
}
