#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
#
# Run the dependency policy (advisories, bans, licenses, sources) over EVERY
# cargo workspace in this repository.
#
# WHY THIS EXISTS AS A SCRIPT, AND NOT AS THREE LINES IN ci.yml.
#
# The CI job listed three manifests by hand: the root, blut-dsl and
# blut-operator. There are ten workspaces, each committing its own Cargo.lock,
# so seven were never audited. That is not hypothetical: RUSTSEC-2026-0285
# (rustls accepting TLS 1.3 handshake messages across encryption level
# boundaries) sat in crates/blut-operator/Cargo.lock at 0.23.41,
# crates/blut-tui/Cargo.lock at 0.23.43 and fuzz/Cargo.lock at 0.23.41 while the
# "advisories · licenses · secrets" job reported success. blut-tui is a
# PUBLISHED crate.
#
# A hand-maintained list silently under-covers the moment someone adds a
# workspace, and nothing tells you. So the list is DERIVED: every committed
# Cargo.lock is a workspace root, and every one of them is checked. Adding a
# workspace adds coverage automatically; the only way to escape the gate is to
# delete your lockfile, which breaks other things loudly.
#
# One policy file (the root deny.toml) is used everywhere, so an exception
# granted to a sidecar is visible in the same place as every other exception.
#
# Usage: scripts/dependency_policy.sh [check-subcommand]   (default: all checks)
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
root="$PWD"
config="$root/deny.toml"

if [[ ! -f "$config" ]]; then
  echo "dependency_policy: missing $config" >&2
  exit 2
fi

# Derive the workspace set from committed lockfiles. `git ls-files` (not `find`)
# so an untracked scratch workspace under target/ is never audited as if it
# shipped.
mapfile -t locks < <(git ls-files '*Cargo.lock' | sort)
if [[ ${#locks[@]} -eq 0 ]]; then
  echo "dependency_policy: no committed Cargo.lock found — refusing to report success" >&2
  exit 2
fi

failed=()
checked=0

for lock in "${locks[@]}"; do
  dir=$(dirname "$lock")
  manifest="$dir/Cargo.toml"
  [[ "$dir" == "." ]] && manifest="Cargo.toml"
  if [[ ! -f "$manifest" ]]; then
    echo "dependency_policy: $lock has no sibling Cargo.toml — refusing to skip silently" >&2
    failed+=("$dir (no manifest)")
    continue
  fi
  echo "=== cargo deny: $manifest"
  if cargo deny --locked --manifest-path "$manifest" --config "$config" check "$@"; then
    checked=$((checked + 1))
  else
    failed+=("$manifest")
  fi
done

echo
echo "dependency_policy: ${checked}/${#locks[@]} workspaces clean"
if [[ ${#failed[@]} -gt 0 ]]; then
  printf 'dependency_policy: FAILED in %s\n' "${failed[@]}" >&2
  exit 1
fi
