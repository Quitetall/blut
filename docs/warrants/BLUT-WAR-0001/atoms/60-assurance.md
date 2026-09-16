---
schema: oh.war/atom/v1
warrant_uuid: 01a0abce-b93f-7071-b331-ccb57756af04
role: assurance
jurisdiction: authored
order: 60
classification: internal
---

# Assurance

## Acceptance Obligations

### OBL-001 — the SAS is accepted and pinned
- **scope:** `docs/sas/revisions/0.1.0.toml` and this Warrant's compiled basis.
- **gate:** `gate://software.repo.war-check@1.0.0`
- **evidence:** `war sas status` reports 0.1.0 accepted by a named human; `war check` reports no `sas.unrecorded`.

### OBL-002 — an independent verification is recorded
- **scope:** this Warrant's `verifications/` before resolution.
- **gate:** `gate://software.repo.war-check@1.0.0`
- **evidence:** a verification whose verifier is not the performer, with
  dispositions backed by the cited gate receipt. Human resolution is the
  subsequent closure act, not a prerequisite for verifying this obligation.

## Gate Adequacy

Required at `basic`.

**Adversarial question:** can this Warrant be resolved without a human ever
signing anything? `war` refuses an agent-kind authorizer or resolver, but this
scaffold has not tested human key custody or an independent verifier. A green
record check cannot settle either question.

- **outcome:** unresolved

## Residual Risk

- The key custody question (`ssh-add -c`) is the operator's; `war` cannot verify it.
