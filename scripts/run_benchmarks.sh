#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

baseline=${BLUT_BENCH_BASELINE:-release-0.2}
mode=${1:-compare}
cargo_cmd=(cargo +1.88)

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
    "${cargo_cmd[@]}" bench -p blut --bench framework -- --baseline "$baseline"
    ;;
  --save)
    save_baseline
    ;;
  *)
    echo "usage: $0 [compare|--save]" >&2
    exit 2
    ;;
esac
