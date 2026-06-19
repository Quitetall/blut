---
name: Bug report
about: Something in the blut engine doesn't behave as documented
title: "[bug] "
labels: bug
---

## Summary
<!-- One or two sentences: what's wrong. -->

## Reproduction
<!-- Smallest steps / code that triggers it. A failing snippet against the
     public API (Stage / Plan / executor / cache) is ideal. -->

```rust
// ...
```

## Expected vs actual
- **Expected:**
- **Actual:**

## Environment
- OS / distro:
- `rustc --version`:
- `blut` version (crates.io version or git SHA):
- **Linux + systemd?** (yes / no) — containment (cgroup memory caps) only runs
  on Linux + systemd; off-systemd blut degrades to a bare spawn, so this changes
  which code path executed:

## Logs / output
<!-- Relevant stderr, the status.jsonl tail, a backtrace (RUST_BACKTRACE=1), etc. -->
