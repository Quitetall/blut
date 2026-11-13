#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

baseline=${BLUT_BENCH_BASELINE:-release-0.2}
# A comparison fails when a benchmark's median is slower than the baseline by
# more than this fraction even at its confidence interval's lower bound.
max_regression=${BLUT_BENCH_MAX_REGRESSION:-0.10}
# A measurement starts only once the 1-minute load average is at or below
# this, and the script gives up after waiting this many seconds for that.
max_load=${BLUT_BENCH_MAX_LOAD:-1.0}
quiet_wait=${BLUT_BENCH_QUIET_WAIT:-900}
mode=${1:-compare}
cargo_cmd=(cargo +1.99.0)

# Build first, so compiling is not what the first samples are measured beside.
build_benches() {
  "${cargo_cmd[@]}" bench -p blut --bench framework --no-run
}

wait_quiet() {
  local deadline=$((SECONDS + quiet_wait)) load
  while :; do
    read -r load _ < /proc/loadavg
    if awk -v l="$load" -v m="$max_load" 'BEGIN { exit !(l <= m) }'; then
      echo "host is quiet: load $load (limit $max_load)" >&2
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "host is not quiet: load $load is above $max_load after ${quiet_wait}s" >&2
      exit 3
    fi
    sleep 10
  done
}

compare_pass() {
  wait_quiet
  # Drop change estimates from earlier comparisons so only this pass is judged.
  find target/criterion -type d -name change -prune -exec rm -rf -- {} +
  "${cargo_cmd[@]}" bench -p blut --bench framework -- --baseline "$baseline"
}

restore_baseline() {
  local found=0 src rel dst
  while IFS= read -r -d '' src; do
    found=1
    rel=${src#bench-baselines/framework/}
    dst="target/criterion/$rel"
    mkdir -p "$(dirname "$dst")"
    rm -rf -- "$dst"
    cp -R -- "$src" "$dst"
  done < <(find bench-baselines/framework -type d -name "$baseline" -print0)
  if [[ $found -eq 0 ]]; then
    echo "missing committed benchmark baseline '$baseline'" >&2
    exit 2
  fi
}

save_baseline() {
  local src rel bench dst
  build_benches
  wait_quiet
  "${cargo_cmd[@]}" bench -p blut --bench framework -- --save-baseline "$baseline"
  while IFS= read -r -d '' src; do
    rel=${src#target/criterion/}
    bench=${rel%/$baseline}
    dst="bench-baselines/framework/${bench}/${baseline}"
    mkdir -p "$(dirname "$dst")"
    rm -rf -- "$dst"
    cp -R -- "$src" "$dst"
  done < <(find target/criterion -type d -name "$baseline" -print0)
}

case "$mode" in
  compare)
    restore_baseline
    build_benches
    suspects=$(mktemp)
    trap 'rm -f -- "$suspects"' EXIT
    compare_pass
    # Criterion reports a regression but still exits 0; this is the gate. A
    # slowdown fails it only when a second pass, on a quiet host again, shows
    # the same benchmark over the limit: one pass alone flags unchanged code.
    if python3 scripts/bench_regression_gate.py target/criterion "$max_regression" \
        --suspects-out "$suspects"; then
      exit 0
    elif [[ ! -s $suspects ]]; then
      exit 2
    fi
    echo "first pass flagged $(wc -l < "$suspects") benchmark(s); confirming with a second pass" >&2
    compare_pass
    python3 scripts/bench_regression_gate.py target/criterion "$max_regression" \
      --confirm "$suspects"
    ;;
  --save)
    save_baseline
    ;;
  *)
    echo "usage: $0 [compare|--save]" >&2
    exit 2
    ;;
esac
