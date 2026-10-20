<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0086 — Give BLUT a SecretRef indirection backed by an age-encrypted store, so credentials reach stages by reference and never land in logs or the content-addressed cache

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`.

> **This file is append-only and living.** Write the decision once. As
> implementation proceeds, **append** dated entries to the Progress Log — never
> rewrite or delete earlier text; correct an earlier statement with a *new* dated
> entry beneath it, so the original stays as historical record. When the
> deliverable passes its gate, fill **Completion** and set Status to
> `Complete — frozen YYYY-MM-DD`. A frozen ADR is permanent. To reverse a frozen
> decision later, write a *new* ADR that supersedes it (updating this one's
> Status pointer is the only edit allowed after freeze).

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-11`

## Context

Prefect, Dagster+, and Flyte all ship a first-class secrets abstraction: a task
declares a *reference* (a block name / secret key), the platform resolves it at
run time from a backend (env, vault, cloud KMS), and the plaintext is kept out of
the serialized flow, out of the UI, and out of logs.

BLUT has nothing here. A stage that needs an S3 token, a Slack webhook URL, or a
wandb API key today gets it from a raw environment variable or a plaintext arg —
which means the value can land in three places it must never be: (1)
`status.jsonl` and the lineage DB (stage args are recorded for provenance); (2)
the ADR-0078 content-addressed cache key (a secret in a stage arg becomes part of
the fingerprint, so rotating the secret silently invalidates every cached node,
*and* the secret bytes sit in the on-disk cache manifest); (3) the console / web
dashboard (ADR 0093) which renders stage args verbatim. For a system that also
handles clinical/PHI workloads (ADR 0061), plaintext credentials flowing into
content-addressed, mesh-replicable artifacts (ADR 0079) is an unacceptable
custody leak.

## Decision

BLUT will introduce a `SecretRef(name)` newtype that stages accept in place of a
plaintext credential, plus an **age/rage-encrypted local secret store** at
`~/.blut/secrets.age` (keyed by an age identity in `~/.blut/keys/`, never in the
repo). A stage argument typed `SecretRef` is treated specially at three
boundaries: **(1) fingerprinting** — the cache key (ADR 0078) hashes the
*reference name*, never the resolved value, so rotating a secret's bytes does not
bust the cache and the secret is absent from every cache manifest; **(2)
serialization** — `status.jsonl`, the lineage DB, and PlanSpec store the
`SecretRef` name only; a redaction pass replaces any resolved secret substring
with `«redacted:name»` before any line is written or streamed; **(3) resolution**
— the plaintext is decrypted lazily, held only in the executing stage's process
memory, zeroized on drop, and passed to the compiled stage via an in-process
handle, never an env var that a child process or crash dump could capture. CLI:
`blut secret set <name>` (reads the value from stdin/`$EDITOR`, never argv),
`blut secret ls` (names + metadata only), `blut secret rm <name>`. Clinical
secrets are hard-local: a secret tagged `restricted` (ADR 0061 tenant, ADR 0096)
is refused by the mesh replication path and by any sidecar (`blut-notify`,
`blut-web`) resolution — it decrypts only inside an engine process on the box
that owns it and never crosses a dispatch/tenant boundary. Redaction is
fail-closed: if the redactor cannot confirm a log line is secret-free (e.g. a
value it could not bound), it drops the line's free-text field rather than emit
it.

## Rationale

- Hashing the *name* not the value is the load-bearing invariant: it keeps ADR
  0078 cache keys stable across rotation AND keeps secret bytes out of the
  content-addressed store the mesh replicates — one design choice fixes both the
  correctness and the custody problem.
- age/rage is a single static binary with a tiny, audited format and no daemon —
  it fits the "engine ships as a CLI tool, no server" charter (ADR 0034) far
  better than running or depending on a Vault process.
- Lazy decrypt + zeroize-on-drop + in-process handle (not env) minimises the
  plaintext's lifetime and blast radius; a crash dump or `/proc/<pid>/environ`
  read does not expose it.
- Reference-by-name is what lets the web/TUI render "uses secret: s3-token"
  safely — the UI never has to be trusted with plaintext.
- Fail-closed redaction matches the house rule everywhere else (gate, broker,
  clinical): an unprovable line is dropped, not emitted.

## Alternatives Considered

- **Plain environment variables + a `.env` file (the status quo, formalised).**
  Rejected: env values are inherited by every child process the stage spawns
  (including `systemd-run` dispatch and any exec'd sidecar), appear in
  `/proc/<pid>/environ`, and give no path to keep the value out of the
  content-addressed cache key — the exact three leaks this ADR exists to close.
- **A network secrets backend (HashiCorp Vault / cloud KMS) as the store.**
  Rejected: it requires the engine to make outbound network calls to resolve a
  secret, which breaks offline `blut run` and — more seriously — means a clinical
  workload's credential resolution depends on, and could exfiltrate to, an
  external service, violating the ADR-0061 "Restricted data never leaves the box"
  hard-block. A network backend can be added later *as a sidecar-mediated,
  non-clinical option*, not as the base mechanism.
- **Encrypt the whole `status.jsonl` / cache at rest instead of per-secret.**
  Rejected: at-rest encryption still materializes plaintext into the cache *key*
  (so rotation busts the cache and the value is in the fingerprint's preimage),
  and it does nothing for the live SSE stream the dashboard reads — it addresses
  storage, not the reference-semantics problem.

## Consequences

- Upside: secrets stop invalidating the cache on rotation; stage args become
  safe to log, replicate, and render; clinical credentials gain an enforced
  local-only custody boundary.
- Upside: `blut-notify` (ADR 0083) can hold its Slack/SMTP creds as
  `SecretRef`s resolved in the sidecar's own process, not baked into rule files.
- Downside: every stage that consumes a credential must migrate its arg type from
  `String` to `SecretRef`; a plaintext-credential-in-arg lint is needed to stop
  regressions.
- Downside: an age identity in `~/.blut/keys/` becomes a new operator
  responsibility — lose it and the store is unrecoverable (by design); this needs
  a documented key-backup procedure.
- Constraint: the redaction pass is now on the hot path of every log/status
  write; it must be cheap (bounded substring set) or it throttles throughput.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** a `SecretRef` newtype + age-backed `~/.blut/secrets.age`
  store, a redaction filter on the status/lineage/log write path, fingerprint
  logic that hashes the ref name, and `blut secret {set,ls,rm}`.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut --test secrets_leak` — runs a stage that
    consumes a `SecretRef`, then greps the produced `status.jsonl`, lineage DB
    rows, cache manifest, and PlanSpec for the known plaintext sentinel;
    additionally asserts a `restricted`-tagged secret is refused by the mesh
    replication and sidecar-resolution paths.
  - `pass when:` `exit == 0` — the plaintext sentinel appears in **zero** of the
    four persisted surfaces, the cache key is byte-identical before/after
    rotating the secret's value, and the restricted-secret cross-boundary attempt
    returns non-zero.
- **Rollback:** `SecretRef` resolution falls back to reading the named env var
  when the store is absent (feature-gated), so removing the store degrades to the
  status quo without breaking stages that already migrated their arg types.

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-11 — LANDED + FROZEN (engine main `78ebf7c`). `src/secrets.rs`:
  `SecretRef {name, restricted}` — a VALUE-FREE reference (serde/JsonSchema) whose
  leak-prevention is STRUCTURAL: it serialises as its name into every provenance
  surface, so the plaintext is never present to leak. `Secret` holds the resolved
  value as `Vec<u8>`, zeroised on drop (a `black_box` barrier defeats dead-store
  elision — no `zeroize` dep, `unsafe` denied) and redacting in `Debug`.
  `SecretResolver::resolve` enforces the restricted hard-block ONCE for every
  backend (a `restricted` secret refuses a remote/mesh/sidecar resolve — ADR 0061
  hard-local, fail-closed); `EnvResolver` = the v1 env-var store (the ADR's stated
  fallback); `redact()` = a longest-first fail-closed scrub belt-and-suspenders.
  **Gate met:** `cargo test -p blut --test secrets_leak` = 3 passed — a SecretRef
  in a stage's args leaves the plaintext sentinel ABSENT from all four surfaces
  (args.json / cache-key input / PlanSpec / lineage fingerprint) while the ref
  name stays visible; the cache-key input is value-INDEPENDENT (rotating the
  secret's bytes yields byte-identical canon + key ⇒ a rotation never busts the
  cache, value absent from every manifest); a restricted secret is refused across
  a boundary while a non-restricted one resolves. Plus 4 unit tests. 857 lib
  green; clippy+fmt+scope_guard clean. MiMo (2 rounds) → the `black_box` zeroize
  barrier + longest-first redact. **DELIVERABLE TAILS (own increments):** the
  age-encrypted `~/.blut/secrets.age` vault + `blut secret {set,ls,rm}` ride a
  `secrets` crypto feature (aes-gcm already a dep) — the gate is store-independent
  (env fallback); wiring `redact()` onto the status/lineage/log writer is a small
  follow-up over the structural guarantee. Status → Complete — frozen 2026-07-11._

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut --test secrets_leak`
- **result:** exit 0 (3 passed). The plaintext sentinel appears in ZERO of the
  four persisted surfaces (args.json / cache-key input / PlanSpec / lineage
  fingerprint); the cache key is byte-identical before/after rotating the
  secret's value; and the restricted-secret cross-boundary attempt returns an
  error (non-zero). scope_guard PASS; 857 lib green; clippy + fmt clean.
- **git_sha:** engine `78ebf7c`

## Related Decisions

ADR 0061 (clinical hard-block — restricted secrets are local-only), ADR 0083
(sidecar crates resolve their own secrets), ADR 0078 (cache key hashes the ref
name, not the value — extends, does not supersede), ADR 0095 (RBAC — who may
`secret set`), ADR 0096 (tenant-scoped secret namespaces).

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: a secret plaintext is ever found in `status.jsonl`, the lineage DB, a
cache manifest, the SSE stream, or a crash dump; if rotating a secret busts the
content-addressed cache (proves the value leaked into the key); or if a
`restricted` secret is observed on a peer node. Add a periodic sentinel-grep over
the persisted stores to CI so a regression surfaces as a red build, not an
incident.
