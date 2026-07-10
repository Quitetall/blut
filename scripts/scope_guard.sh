#!/usr/bin/env bash
# scope_guard.sh — fail if BLUT Rust source reimplements a GENERIC concern
# that the charter (ADR 0034) says BLUT must DELEGATE.
#
# CANONICAL COPY. This guard lives with the source it guards (the engine
# repo); the meta-repo's tools/blut_scope_guard.sh is a thin wrapper that
# delegates here. CI runs it in the build-test-lint job.
#
# BLUT OWNS the orchestration nouns (recipes / artifacts / lineage / parity).
# It must NOT reimplement: system/GPU monitoring, metrics plotting, log
# aggregation, process supervision, or HTTP servers. Those are delegated to
# systemd-run / journald / wandb-offline / sidecar crates (crates/* are
# OUTSIDE this guard's scope by design — the ADR 0079/0083 sidecar boundary).
#
# Mechanism: grep src/**.rs for the fingerprints the four scope audits
# flagged as bloat_signals. Any match => non-zero exit => CI red.
#
# Usage:  scripts/scope_guard.sh [BLUT_SRC_DIR]
#   default BLUT_SRC_DIR = src relative to the engine repo root (cwd).
# Allowlist: append a path suffix to scripts/scope_guard.allow to exempt a
#   file (permanent by-design exemptions cite their ADR; staged migrations
#   cite the ADR-0034 step that will remove them).
set -euo pipefail

SRC="${1:-src}"
ALLOW="$(dirname "$0")/scope_guard.allow"

if [[ ! -d "$SRC" ]]; then
  echo "scope_guard: source dir not found: $SRC" >&2
  exit 2
fi

# (label, extended-regex) pairs. Keep these mined from audit bloat_signals.
# Matched against CODE only (Rust // line-comments stripped first) so that
# comments referencing a concept are not false positives. The HTTP pattern is
# scoped to `use`/dependency lines so log-filter strings (e.g. "hyper=warn")
# do not trip it.
PATTERNS=(
  "system/GPU monitoring|nvidia-smi|/proc/(meminfo|loadavg)|Command::new\\(\"df\"|\\bsysinfo\\b"
  "metrics plotting/charts|\\b(plotters|ratatui::widgets::(Chart|BarChart|Sparkline)|charming|kuva|asciigraph)\\b"
  "log aggregation (reimpl)|\\b(slog|fern|flexi_logger)\\b"
  "process supervision|\\b(setsid|pre_exec|process_group|libc::kill)\\b|kill\\([^)]*,[[:space:]]*0[[:space:]]*\\)"
  "HTTP/web server|[0-9]+:[[:space:]]*use[[:space:]]+(axum|hyper|warp|tower_http|actix_web|rocket|tiny_http)\\b"
)

fail=0
while IFS= read -r -d '' f; do
  # honor allowlist — SUFFIX match (not substring) so 'runner.rs' can't exempt
  # a sibling like 'runner.rs_extra.rs'; entries are repo-relative path tails.
  if [[ -f "$ALLOW" ]]; then
    _skip=0
    while IFS= read -r pat; do
      [[ -z "$pat" ]] && continue
      [[ "$f" == *"$pat" ]] && { _skip=1; break; }
    done < <(sed 's/#.*//; s/[[:space:]]*$//' "$ALLOW" | sed '/^[[:space:]]*$/d')
    [[ "$_skip" == 1 ]] && continue
  fi
  # strip whole-line // comments, trailing `//`, and inline /* */ block comments,
  # keep line nums (grep -n -> 'N:line', which the patterns account for).
  code="$(grep -nv '^[[:space:]]*//' "$f" | sed 's://[^"]*$::' | sed 's:/\*[^*]*\*/::g')"
  for entry in "${PATTERNS[@]}"; do
    label="${entry%%|*}"
    rx="${entry#*|}"
    hits="$(printf '%s\n' "$code" | grep -E "$rx" || true)"
    if [[ -n "$hits" ]]; then
      printf '%s\n' "$hits" | sed "s|^|$f:|"
      echo "  ^-- SCOPE-BLOAT [$label] in $f  (ADR 0034: delegate, do not reimplement)" >&2
      fail=1
    fi
  done
done < <(find "$SRC" -name '*.rs' -type f -print0)

if [[ "$fail" -ne 0 ]]; then
  echo "" >&2
  echo "scope_guard: FAIL — BLUT reimplements a delegated concern." >&2
  echo "  Resolve: delegate to systemd-run/journald/wandb or a sidecar crate" >&2
  echo "  (ADR 0034/0083), or add the file suffix to scripts/scope_guard.allow" >&2
  echo "  citing the ADR that justifies it." >&2
  exit 1
fi
echo "scope_guard: PASS — no delegated-concern reimplementation in $SRC"
