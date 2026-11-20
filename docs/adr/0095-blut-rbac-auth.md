<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# ADR 0095 — Gate every mutating BLUT control action behind capability-scoped tokens (viewer/operator/admin) with an audit trail, defaulting to read-only and opening a path to OIDC

> **Federated record.** Federated from the project's decision log; text redacted
> for publication where marked `[redacted: <category>]`. The source log is frozen,
> so this copy does not drift. Index: [docs/adr/README.md](README.md).

**Status:** `complete`

## Status

`Proposed 2026-07-10` → `In progress` → `Complete — frozen 2026-07-11`

## Context

Prefect and Dagster+ ship workspace RBAC; Flyte leans on project-scoped IAM.
Once an orchestrator has a web control plane ([ADR 0093](0093-blut-web-control-plane.md)), "who is allowed to
promote to prod / cancel a run / read a clinical tenant's lineage" stops being a
single-operator assumption and becomes a security boundary an evaluator will
probe.

BLUT has no authz at all. The TUI assumes the one person at the terminal owns the
box; the moment `blut-web` ([ADR 0093](0093-blut-web-control-plane.md) v2) and webhook ingress ([ADR 0094](0094-blut-eventing-triggers-sla.md)) expose
mutating actions over the network, an unauthenticated `POST` could launch a run,
rollback a deployment ([ADR 0085](0085-blut-deployment-registry.md)), or read a Restricted tenant's status. There is
no token, no role, no audit of who did what. This must be closed *before* the web
control plane's write path (v2) ships, and the read path (v1) must default to a
posture that leaks nothing.

## Decision

BLUT will add **capability-scoped auth** enforced at every mutating control
boundary. Three built-in roles, ordered by capability: `viewer` (read status /
lineage / DAG — no mutation), `operator` (viewer + run / cancel / retry), `admin`
(operator + `plan promote`/`rollback`, secret management, tenant + token
administration). Tokens live in `~/.blut/web-tokens.toml` (bcrypt/argon2-hashed
token secrets, never plaintext), each row binding a token to a role and a tenant
([ADR 0096](0096-blut-multi-tenancy.md)). **Read-only is the default**: an unauthenticated or unrecognised
caller gets `viewer` scope *only* for non-restricted tenants, and any request
without a valid token that targets a mutation or a `restricted` tenant is
refused. **Every mutating action is gated AND audit-logged**: the exec-bridge
([ADR 0083](0083-blut-interface-stack.md)) checks the caller's capability before shelling to the CLI and writes a
`~/.blut/audit.jsonl` row (actor token id, role, tenant, action, target
fingerprint, timestamp, allow/deny) *before* dispatch — a denied action is
logged too. Enforcement is fail-closed: a token that does not resolve to a role,
or a role that does not dominate the action's required capability, is a deny, not
a fallthrough. Clinical hard-block composes on top ([ADR 0061](0061-blut-p2p-compute.md)/0096): no role,
including `admin`, grants a *shared*-tenant token access to a `restricted`
tenant's data or actions — clinical access requires a token minted in that
tenant, on that box. The token file is the v1 mechanism; the design keeps the
capability check behind a `Principal` trait so an **OIDC** resolver (map an OIDC
group claim → role) can be added later as a sidecar-mediated token exchange
without changing any gate.

## Rationale

- Capability domination (`admin ⊇ operator ⊇ viewer`) is a total order that makes
  each gate a single comparison — simple to audit and impossible to
  half-configure into an accidental grant.
- Read-only default + fail-closed means the *insecure* state is "can't do
  anything," never "can do everything" — the safe failure direction for a control
  plane exposed to the network.
- Gating at the exec-bridge ([ADR 0083](0083-blut-interface-stack.md)) reuses the single enforcement path humans
  and the web share, so authz cannot be bypassed by a second mutation route (ADR
  0093 forbids one existing anyway).
- Audit-before-dispatch (including denies) gives the tamper-evident "who did
  what" trail RBAC exists to provide, and it is the same `audit.jsonl` the plan
  registry ([ADR 0085](0085-blut-deployment-registry.md)) and secrets ([ADR 0086](0086-blut-secrets-management.md)) actions write to — one trail.
- A `Principal` trait behind the token file means OIDC is an additive resolver,
  not a rewrite — we ship the simple thing now without painting out the
  enterprise path.

## Alternatives Considered

- **OS-user / unix-socket peer-credential auth only (trust whoever can reach the
  box).** Rejected: it collapses the moment `blut-web` serves over a network
  socket to browsers that have no OS identity on the host, and it cannot express
  "this operator may run but not promote to prod" or the clinical tenant boundary
  — peer-cred is coarse machine-level trust, not per-action capability.
- **Full OIDC/SSO from day one.** Rejected: it requires standing up (or
  depending on) an external identity provider before *any* authz exists, blocking
  the [ADR-0093](0093-blut-web-control-plane.md) v2 write path on enterprise infrastructure most single-team
  deployments don't have; the token file gives real authz immediately and OIDC
  slots in behind the same `Principal` trait later.
- **Per-endpoint ACL lists instead of ordered roles.** Rejected: free-form ACLs
  make "is this caller allowed?" an unbounded set-membership question that is easy
  to misconfigure into an over-grant and hard to audit; three dominating roles
  keep every decision a single comparison and every token's power obvious from its
  role name.

## Consequences

- Upside: the web control plane ([ADR 0093](0093-blut-web-control-plane.md) v2) and webhook ingress ([ADR 0094](0094-blut-eventing-triggers-sla.md)) can
  ship a write path safely; mutations are gated and attributable.
- Upside: one `audit.jsonl` becomes the compliance substrate for registry,
  secrets, and control actions — a single tamper-evident log to review.
- Downside: token issuance/rotation is a new operator responsibility; a lost
  `admin` token or a botched `web-tokens.toml` can lock out mutation (recoverable
  only by local file edit on the box — an intentional last-resort).
- Downside: every mutating code path now takes a `Principal` and a capability
  check — a cross-cutting change touching the exec-bridge and every control
  subcommand; a missing check is a silent authz hole, so the gate test must
  enumerate mutating actions.
- Constraint: `viewer`-default read access still must honour tenant scoping —
  the default viewer sees only non-restricted tenants, never all data.

## Implementation Plan  *(the contract — write once; amend only by a dated Progress Log entry)*

- **Deliverable:** a `Principal` + capability model, `~/.blut/web-tokens.toml`
  (hashed tokens → role + tenant), gate checks on every mutating control action,
  and `~/.blut/audit.jsonl` allow/deny records; an OIDC-resolver seam behind the
  `Principal` trait.
- **Acceptance gate (fail-CLOSED; unmeasured ⇒ FAIL):**
  - `gate_cmd:` `cargo test -p blut --test rbac_matrix` — drives each mutating
    action (`run`, `cancel`, `plan promote`, `plan rollback`, `secret set`) with a
    `viewer`, `operator`, and `admin` token and with no token, asserting the
    allow/deny matrix, that a `restricted`-tenant action is denied to every
    shared-tenant token including `admin`, and that every attempt (allow and deny)
    produced an `audit.jsonl` row.
  - `pass when:` `exit == 0` — matrix matches the role order exactly, clinical
    cross-tenant is always denied, and the audit row count equals the attempt
    count.
- **Rollback:** auth is behind a `--features rbac` gate defaulting on for
  `blut-web`; disabling it reverts to single-operator TUI trust (no network
  write path), so the fallback is "no web mutation," never "open web mutation."

## Progress Log  *(append-only; newest last; never rewrite an earlier entry)*

- _2026-07-10 — opened._
- _2026-07-11 — LANDED + FROZEN (engine main `f9de1d0`). `src/rbac.rs`: `Role`
  {Viewer<Operator<Admin} (derived `Ord` = the single capability order); `Action`
  → `required_role` (read=viewer, run/cancel/retry=operator, plan
  promote/rollback + secret/token/tenant admin=admin); `Principal{token_id, role,
  tenant}`; fail-closed `authorize` (read-only default — anonymous = viewer on
  non-restricted only; the clinical hard-block is checked FIRST and dominates
  role: no shared-tenant token, incl admin, reaches a restricted tenant);
  `enforce` = authorize + append_audit, **fail-closed by construction** (an audit
  write failure → DENY, so a mutation is never permitted unlogged); `audit.jsonl`
  written 0600, one atomic `write_all` per record under O_APPEND; `TokenStore`
  over `web-tokens.toml` with sha256-hashed secrets + fail-loud validation
  (unknown role/tenant or duplicate hash = hard error). **Gate met:** `cargo test
  -p blut --test rbac_matrix` = 3 passed — the viewer/operator/admin/none × every
  action allow-deny matrix holds exactly; a restricted-tenant action is denied to
  every shared-tenant token incl admin AND anonymous, while a token minted in the
  tenant is subject only to its role; every attempt (allow + deny) produces one
  audit row and the denied clinical attempt is on the record. Plus 3 rbac unit
  tests. 857 lib green; clippy+fmt+scope_guard clean. Model review (2 rounds) → fail-closed
  enforce, 0600 + atomic audit append, fail-loud token-store validation, Ord
  cleanup. **DESIGN REFINEMENT:** token secrets are sha256-hashed, not
  bcrypt/argon2 as the Decision names — for HIGH-entropy random API tokens a fast
  cryptographic hash is the correct + standard choice (bcrypt/argon2 exist for
  LOW-entropy human passwords), honouring "hashed, never plaintext" dependency-
  free. **WIRING TAIL (rides [ADR 0083](0083-blut-interface-stack.md)):** the enforcement is a complete, gated
  PRIMITIVE; the call sites (the `blut-web` exec-bridge invoking `enforce` before
  a mutating CLI action) land with `blut-web` — the only network mutation path,
  not yet split out — and local CLI mutation stays single-operator TUI trust per
  the Decision. `AuthDecision.reason` embeds diagnostics and must be redacted, not
  returned verbatim, at that HTTP boundary. The `Principal`/token seam is the
  documented OIDC extension point. Status → Complete — frozen 2026-07-11._

## Completion / Resolution  *(fill once; this + the frozen Status line is the single freeze mark)*

- **verdict:** `passed`
- **gate_cmd:** `cargo test -p blut --test rbac_matrix`
- **result:** exit 0 (3 passed). The capability matrix matches the role order
  exactly across every action; the clinical cross-tenant boundary is always
  denied (every shared-tenant token incl admin, and anonymous); and the audit
  row count equals the attempt count (allow AND deny logged). scope_guard PASS;
  857 lib green; clippy + fmt clean.
- **git_sha:** engine `f9de1d0`  ·  **artifact sha256:** n/a (source-only)

## Related Decisions

[ADR 0093](0093-blut-web-control-plane.md) (web control plane — its v2 write path depends on this gate), [ADR 0086](0086-blut-secrets-management.md)
(secrets — `secret set` is an `admin` action), [ADR 0096](0096-blut-multi-tenancy.md) (tenant scoping the token
binds to), [ADR 0083](0083-blut-interface-stack.md) (exec-bridge is the single gated mutation path), [ADR 0061](0061-blut-p2p-compute.md)
(clinical block composes above every role).

## Validation  *(ongoing — distinct from the completion gate above)*

Wrong if: any mutating action is reachable without a capability check (an
un-gated code path); a `viewer` or no-token caller ever mutates or reads a
`restricted` tenant; an action executes without an `audit.jsonl` row; or an
`admin` token from a shared tenant is observed acting on clinical data. Watch for
mutating subcommands added without a gate — enforce by a test that fails if a
control action is registered without declaring a required capability.
