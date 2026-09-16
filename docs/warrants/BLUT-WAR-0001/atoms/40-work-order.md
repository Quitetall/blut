---
schema: oh.war/atom/v1
warrant_uuid: 01a0abce-b93f-7071-b331-ccb57756af04
role: work_order
jurisdiction: authored
order: 40
classification: internal
---

# Work Order

## Deliverables

1. The SAS accepted at 0.1.0 (`war sas propose 0.1.0`, then a human's `war sign 0.1.0 --ssh-sign`).
2. This Warrant authorized (`war authorize BLUT-WAR-0001`, then `war sign BLUT-WAR-0001 --ssh-sign`).
3. Gate evidence recorded (`war evidence record BLUT-WAR-0001`).
4. An independent verification ingested (`war verify BLUT-WAR-0001 --performer <you>` handed to a separate context; `war verify BLUT-WAR-0001 --response <file>`).
5. This Warrant resolved (`war resolve BLUT-WAR-0001`, then `war sign BLUT-WAR-0001 --ssh-sign`).

## Frozen Surfaces

`docs/authority/roles.toml` and `docs/authority/allowed_signers` are
human-written. The committed `.example` files confer no authority.

## Premade Instructions

- `war next` before every step; it names whose act comes next.
- Never edit a record to make `war check` go green.

## Autonomy and Escalation

Tier T2: the agent drafts and records; every signature is a human's.

## Rollback

Before authorization, revise the draft atoms and recompile. After
authorization, use a new contract revision; never delete a signed or resolved
record to simulate rollback. If a gate or signature fails, leave the refusal
visible and request a correction through OpenWarrant's governed path.
