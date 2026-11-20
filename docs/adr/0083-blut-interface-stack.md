<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0083 — BLUT interface stack: CLI engine, sidecar TUI/web crates, and the charter boundary

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`.

> **This file is append-only and living.** Append dated Progress-Log entries as
> implementation proceeds; fill **Completion** and set Status to
> `Complete — frozen YYYY-MM-DD` when the deliverable passes its gate.

- **Status:** `Complete — frozen 2026-07-19` (M1 2026-07-12 → M2/M3 2026-07-19)
- **Extends:** ADR 0034 (scope charter — this ADR documents its sidecar boundary
  properly, replacing the one-liner in ADR 0079's Non-goals), ADR 0037 (cookbook
  architecture), ADR 0078 (out-of-process front-end precedent: `blut-dsl`)
- **Context:** the premier-heavyweight roadmap (a planning note
  `[redacted: private path]`, M1–M6). Locked user decisions: TUI-first;
  web = Leptos + WASM, Rust end-to-end; **UI and TUI in separate crates; the
  engine ships as the CLI tool**.

## Context

ADR 0034 forbids HTTP servers, GPU monitoring, plotting, log aggregation, and
process supervision inside the engine (`src/`) — but until today the ban was
enforced by convention (the guard script existed in the meta-repo and ran in no
CI), and the one legal exception (`crates/blut-worker`'s axum server) was
documented only as a one-line Non-goal in ADR 0079. Meanwhile the roadmap adds
two interface surfaces (a web dashboard, a notifier daemon) and splits the TUI
out of the engine — all of which need the boundary stated precisely, once.

## Decision

### 1. The charter boundary, stated precisely

**The unit of the charter is the engine source tree (`src/`), not the git
repository.** A *sidecar crate* — any crate under `crates/*` — may host servers,
HTTP clients, SMTP, or other delegated concerns, because it is a separate
process the operator opts into, supervised by systemd, that the engine never
links. The engine may `exec` a sidecar (dispatch, below); it must never embed
one.

Enforcement is now mechanical: `scripts/scope_guard.sh` (canonical copy beside
the source it guards; a meta-repo copy `[redacted: private path]` is a delegating
wrapper) runs as the first step of the engine CI's build-test-lint job. The
allowlist beside it distinguishes **permanent by-design exemptions** (the broker
admission probe, the containment ladder's `pre_exec` — each citing its ADR) from
**staged migrations** (citing the step that removes them).

### 2. Crate topology (end state, built across M1–M3)

```
blut-types      NEW   pure-serde wire types; wasm32-safe (serde + serde_json only)
   ↑        ↑
   |        |
 blut     blut-web/ui  NEW  Leptos+WASM CSR front-end
   ↑   ↑      ↑
   |   |      | (static bundle embedded via include_dir!)
   |  blut-tui  NEW  crates/blut-tui — console + cookbook-cockpit binary
   |
 blut-web    NEW  crates/blut-web — axum API host + webhook ingress, serves the UI
 blut-notify NEW  crates/blut-notify — status.jsonl tail → sinks + SLA rules
 blut-dsl, blut-operator   existing sidecars, unchanged
 blut-worker DEPRECATED    → deleted in M3 (superseded by src/cloud + blut-web)
```

- **`blut-types` is the keystone**: `StageEvent`/`HostedEvent`, the PlanSpec IR,
  `SensorOutcome`, lineage rows — data types only, `serde`+`serde_json` as the
  only mandatory deps. The engine re-exports them at their old paths (no
  downstream churn). CI carries a required
  `cargo check --target wasm32-unknown-unknown -p blut-types` job from day one —
  the Rust-end-to-end premise of the web UI depends on this staying green.
- **`blut-tui`** takes `src/tui/` (~6.5k LOC) as the `blut-tui` binary. The
  `CookbookTui` trait STAYS in the engine (cookbooks compile against `blut`),
  behind a `tui-api` feature carrying only the trait definitions + ratatui
  types. Ratatui is pinned `=x.y` in lockstep between the two crates; the trait
  surface is frozen by this ADR (changes require a Progress-Log entry).
- **`blut-web`** hosts the axum API (read paths serve exactly the artifacts the
  TUI reads: status.jsonl, lineage SQLite, PeerRegistry, PrivacyLedger) and the
  embedded Leptos bundle. One binary, zero deploy steps.
- **`blut-notify`** tails status.jsonl and routes rule matches to sinks
  (webhook / SMTP / exec); SLA rules live in its engine-readable rule file so
  `blut sla check` (a pure lineage-db read in the engine — charter-fine) gives
  the TUI and CI the same answers.

### 3. CLI dispatch: cargo-style external subcommands

`blut <cmd>` falls through to `exec`ing `blut-<cmd>` from `PATH` when `<cmd>` is
not built in: `blut tui` → `blut-tui`, `blut web` → `blut-web`,
`blut notify` → `blut-notify`. This is `exec`, not process supervision
(charter-safe); it keeps "the main engine ships as the CLI tool" literally true;
and it makes every future sidecar free to add. (~80 LOC in `src/cli/mod.rs`.)

### 4. Live control: the exec bridge

Mutating web endpoints never reach into engine internals — they shell out to the
CLI (`blut run …` spawned detached via `systemd-run --user`; the existing
`blut cancel <id> --grace`). One enforcement path for human and API; the
user-action audit log (`~/.blut/audit.jsonl`, M4) wraps the bridge in one place.
Retry = re-launch — the content-addressed cache makes completed nodes free, which
is the honest semantics.

- _2026-07-19 — M2 INCREMENT 1 LANDED (engine `3b81dbe`): `crates/blut-web`,
  the read-only web sidecar._ Standalone-workspace axum binary (engine
  path-dep; the engine never links it; reached as `blut web` via the M1
  dispatch). Serves the TUI's stores as JSON: jobs, bounded status.jsonl
  tails, lineage graphs (JSON/DOT), model cards, run diffs, model-registry
  pointers+history. Custody fail-closed: `EXPORT_EXCLUDES_RESTRICTED` is a
  constant `true` with a tripwire test (ADR 0061); shared-tenant-only registry;
  ADR-0095 bearer-token auth; a non-loopback bind WITHOUT a token store is
  refused. CI gate added. Live-smoked on the box (real jobs served, 404s,
  0.0.0.0 refusal). 5/5 tests, clippy 0. REMAINING M2/M3: the exec bridge
  (mutations via rbac::enforce → the CLI), the embedded Leptos+WASM bundle
  (shares the blut-types wire types these endpoints serve), the blut-tui
  extraction, blut-worker deletion._

- _2026-07-19 — M2 COMPLETE + M3 DELETION (engine `0bc653d` → `2724ec9`;
  backends and cookbook commits `[redacted: private repository]`). Four landings:_
  1. **The exec bridge (incr 2, `0bc653d`)** — `POST /api/jobs` (run) +
     `POST /api/jobs/{id}/cancel`: `rbac::enforce` (Operator floor, shared
     tenant only, allow AND deny audited to `audit.jsonl` BEFORE dispatch,
     audit-write failure = deny) → shell out to the CLI (`--cli` names the
     cookbook binary; args travel as ONE argv element, spawn detached with a
     launch log). No token store ⇒ mutations denied even on loopback. One
     enforcement path for human and API — §4 exactly as written.
  2. **The blut-tui extraction (`5f43ce8`)** — `src/tui/` (~7.5k LOC) →
     `crates/blut-tui` (standalone workspace; lib + `blut-tui` binary). The
     engine ships ZERO terminal-UI deps (ratatui/crossterm/fuzzy-matcher gone,
     `tui` feature deleted, default = []). Seam: `cli::TuiHook` (two plain fn
     pointers) + `run_with_tui` — a cookbook binary passes `blut_tui::hook()`
     and bare invocation opens the console over its live registry; a hookless
     binary's `tui` subcommand execs `blut-tui` from PATH via the M1 dispatch.
     Per the 2026-07-12 correction, NO ratatui type crosses the engine
     boundary (`CookbookTui` unchanged), so the ratatui `=x.y` lockstep pin is
     moot — the engine has no ratatui to lock against. Cookbook flipped
     (`[redacted: private repository]`): `lqt tui --check` renders 14 views + 8 tabs + 4 overlays
     through the hook; live dry-run resolves the identical 40.0G (ADR 0133
     store continuity intact across the move).
  3. **The embedded Leptos+WASM dashboard (incr 3, `198d7ef`)** — the gate
     line is TRUE: `blut web` serves the dashboard from one binary. Leptos 0.8
     CSR app (`crates/blut-web/ui`): jobs table → status tail + cancel, a
     recipe launcher through the exec bridge, lineage graph/card panels —
     deserializing the wasm-safe `blut-types` wire types the endpoints emit
     (the keystone premise, closed). Bundled by `scripts/build_ui.sh` (cargo
     wasm32 + wasm-bindgen pinned `=0.2.126`; trunk rejected — its
     libdeflate-sys build breaks here), staged by build.rs into the binary via
     `include_dir!` (428K; a self-describing stub page when unbuilt, so
     API-only builds still serve "/"). Static shell unauthenticated; ALL data
     via the token-gated `/api`. CI: wasm32 fmt+clippy gate for the ui crate.
  4. **blut-worker DELETED (M3, `2724ec9`)** — superseded by `src/cloud`
     (ADR 0082) + this stack; CI gate, deny line, release steps removed. Its
     sidecar precedent survives as this ADR.

## Consequences

- The engine's dependency tree gets *smaller* (ratatui/crossterm leave the
  default build in M1); the lean-CLI story becomes real.
- Two lockstep constraints appear: the ratatui `=x.y` pin (blut ↔ blut-tui) and
  the wasm32-cleanliness of `blut-types` (blut ↔ blut-web/ui). Both are CI-gated.
- `blut-worker` is deprecated now, deleted in M3 — its sidecar precedent is the
  only part that survives, as this ADR.

## Progress Log  *(append-only; newest last)*

- _2026-07-09 — opened._ Same day: the scope guard moved to its canonical home
  (`scripts/scope_guard.sh` + allowlist) and became the first
  step of the engine CI build-test-lint job — the charter is now enforced, not
  conventional. Four accumulated violations triaged into the allowlist with
  justifications (containment `pre_exec` ×2 → permanent by design, ADR 0046;
  nvidia-smi one-shot probes ×2 → consolidating into the ADR 0087 GPU-inventory
  seam). `crates/blut-worker` marked deprecated (README), deletion scheduled M3.
- _2026-07-12 — M1 LANDED (engine main `d66e87f`): the wasm32 keystone crate +
  the external-subcommand dispatch seam._ Two of this ADR's load-bearing pieces:
  1. **`crates/blut-types`** — the wasm32-SAFE keystone. The read-side lineage
     wire types (`ProvenanceGraph`/`ModelCard`/`RunDiff`/`CardContent`/`GraphNode`,
     ADR 0099) move here from `src/lineage_report.rs`; serde + serde_json are the
     only structural deps (sha2/faster-hex are pure-Rust hash helpers) — NO tokio,
     rusqlite, filesystem, or engine dep, so the Leptos+WASM `blut-web` sidecar can
     depend on it. **Zero downstream churn:** `src/lineage_report.rs` is now an
     EXPLICIT re-export (`pub use blut_types::report::{…}` — named, not a glob, so
     the keystone can't silently widen the engine's public API), so
     `crate::lineage_report::{…}` still resolves for `lineage_db` + the CLI.
  2. **External-subcommand dispatch** — `#[command(external_subcommand)]
     External(Vec<String>)`: `blut <cmd> <args…>` with no built-in match execs
     `blut-<cmd>` from PATH, forwarding args + propagating exit status/signal. This
     is the seam by which the sidecar binaries the engine deliberately never links
     (`blut-web`, `blut-notify`, eventually `blut-tui`) become first-class `blut`
     subcommands — no in-process server, no dylib loader (ADR 0034). Fail-closed:
     the name is ALLOWLISTED (`[A-Za-z0-9_-]`, cargo's `cargo-<cmd>` convention) so
     no path/escape/separator can reach outside the `blut-*` family; a missing
     sidecar is a hard error naming the binary, never a silent no-op.
  3. **CI gate `wasm32-keystone`** — `cargo check -p blut-types --target
     wasm32-unknown-unknown` + host tests. Fail-closed guard: a PR that adds a
     non-wasm dep to `blut-types` breaks here before it can reach main.
  Gate part: `cargo test -p blut-types` = 4 passed (incl DOT node-ID/edge escape),
  `cargo test -p blut external_subcommand` = 3 passed (traversal-reject, empty-argv,
  missing-sidecar), `--test lineage_graph_card_diff` = 5 passed (re-export intact);
  full suite green (`-j 2`); clippy 0; fmt clean; scope_guard PASS; blut-types
  wasm32 check green. MiMo: DOT-injection fix (escape node IDs + edge endpoints, not
  just labels) + allowlist-over-blocklist applied; `CardContent`-missing finding was
  a verified FALSE POSITIVE (it is the first item in the re-export list; suite
  compiled green). **REMAINING (M2/M3):** the `blut-tui` / `blut-web` (Leptos+WASM
  dashboard reading the lineage/partition tables) / `blut-notify` sidecar crates;
  deleting `crates/blut-worker`. ADR stays In progress until `blut web` serves the
  dashboard from one binary.
- _2026-07-12 — INTERFACE CONTRACT CORRECTION (owner-locked)._ A cookbook's
  bespoke TUI is **not** a BLUT widget framework and is not required to use
  BLUT-owned Ratatui components. The current `CookbookTui` surface is deliberately
  small, object-safe discovery/lifecycle glue (`label`, `about`, and `run` with
  the live `Arc<Registry>`); it exposes no Ratatui widget or event-loop types.
  A cookbook may implement a normal Ratatui application with its own component
  model, or launch an arbitrary sidecar binary, own the terminal while active,
  and return control when it exits. The generic BLUT console is an optional
  fallback for cookbooks without a bespoke TUI. Therefore the earlier proposed
  `tui-api` feature carrying Ratatui types and a lockstep Ratatui API surface is
  not the controlling contract; keep launch/status/control messages stable, not
  cookbook UI implementation internals. Execution order is also corrected:
  finish engine software before starting the web tier. ADR remains In progress._

## Completion / Resolution  *(fill at freeze)*

- **verdict:** passed
- **gate_cmd:** engine CI green with the scope-guard step + the wasm32
  `blut-types` check; `blut tui` opens via external-subcommand dispatch;
  `blut web` serves the dashboard from one binary.
- **result:** all three hold — scope_guard PASS + wasm32 gates in CI (now
  incl. the blut-web-ui wasm32 lint + the blut-tui sidecar gate); the `tui`
  subcommand routes to the sidecar (in-process via `TuiHook` for cookbook
  binaries — `lqt tui --check` 14 views/8 tabs/4 overlays; PATH exec for
  hookless binaries via the M1 dispatch, dispatch tests green); the embedded
  dashboard live-smoked from one binary (shell 200, 392K wasm
  `application/wasm`, `/api` coexisting, 404s). Engine 762 + web 10 + tui 76
  + backends 93 + cookbook suites all green; clippy 0 everywhere.
- **shas:** engine `2724ec9` (bridge `0bc653d`, extraction `5f43ce8`,
  dashboard `198d7ef`, worker deletion `2724ec9`) · backends and cookbook
  `[redacted: private repository]`.

## Related Decisions

ADR 0034 (charter), 0037 (cookbooks), 0078 (out-of-process front-ends), 0079
(the superseded one-liner), 0082 (cloud queue — retires blut-worker's queue),
0084–0091 (0084 and 0089 [not federated]; the roadmap ADRs that build on this stack).

## Validation  *(ongoing)*

Wrong if: any `crates/*` code gets linked into the engine binary (the boundary
is process, not repo), the scope guard is removed/skipped in CI, `blut-types`
grows a non-serde mandatory dep, or a mutating web endpoint bypasses the exec
bridge (two enforcement paths would drift).
