#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Brian Lam
#
# API-compatibility gate: has anything already published been broken?
#
# Each published crate is compared against its own newest release on crates.io.
#
# WHY `--release-type minor`, AND WHY IT IS LOAD-BEARING.
#
# Without it, cargo-semver-checks compares the current version string against
# the baseline version string. On `main` between releases those are IDENTICAL,
# so it reports "no change; assume major" and runs ZERO checks -- 0 pass, 254
# skip, exit 0. That is a gate that cannot fail, which is not a gate.
#
# Naming the release type forces the comparison to happen: 196 real checks, and
# the question becomes "could the code on main ship as a minor bump?" -- i.e.
# has anything published been removed or broken. Measured on this repository:
# removing `pub` from one method in blut-types produces
# `inherent_method_missing` and exit 100.
#
# When a crate IS deliberately getting a major bump, this gate is the wrong
# question and the version bump commit should say so.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2

# crate:baseline-version:manifest ("" manifest = root workspace member)
TARGETS=(
  "blut:0.2.0-alpha.1:"
  "blut-types:0.2.0-alpha.1:"
  "blut-notify:0.2.0-alpha.1:"
  "blut-dsl:0.2.0-alpha.1:crates/blut-dsl/Cargo.toml"
  "blut-tui:0.2.0-alpha.1:crates/blut-tui/Cargo.toml"
  "blut-graph-core:0.3.0:crates/blut-graph-core/Cargo.toml"
)

failed=()
for entry in "${TARGETS[@]}"; do
  crate="${entry%%:*}"
  rest="${entry#*:}"
  baseline="${rest%%:*}"
  manifest="${rest#*:}"

  echo "=== semver-checks: $crate (baseline $baseline)"
  if [[ -n "$manifest" ]]; then
    args=(--manifest-path "$manifest")
  else
    args=(-p "$crate")
  fi
  if ! cargo semver-checks check-release "${args[@]}" \
        --baseline-version "$baseline" --release-type minor; then
    failed+=("$crate")
  fi
done

echo
if [[ ${#failed[@]} -gt 0 ]]; then
  printf 'semver_checks: FAILED for %s\n' "${failed[@]}" >&2
  echo "A published API was removed or changed incompatibly. Either restore it," >&2
  echo "or make the next release a major bump and say so in CHANGELOG.md." >&2
  exit 1
fi
echo "semver_checks: ${#TARGETS[@]} published crates still compatible with their releases"
