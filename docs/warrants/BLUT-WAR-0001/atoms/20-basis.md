---
schema: oh.war/atom/v1
warrant_uuid: 01a0abce-b93f-7071-b331-ccb57756af04
role: basis
jurisdiction: authored
order: 20
classification: internal
---

# Basis

## Governing text

- `docs/sas/BLUT_SAS.md` at the revision `war sas propose 0.1.0`
  records; §6.10 (levels), §98 Phase 1 (this phase), §106 rows RQ-001–003.
- OpenWarrant's own SAS and CLI at the pinned `v1.0.0` source tag; check the
  active binary with `war --version` before using it.

## Prerequisites

- `docs/authority/roles.toml` written by a human from `roles.toml.example`,
  naming at least one human with the authorizer and resolver roles.
- `docs/authority/allowed_signers` written by a human from its `.example`,
  with the key loaded under `ssh-add -c`.

## Unknowns

- Whether a second eligible signer and an independently controlled verifier
  workspace will be configured. The committed independence flags are false;
  `war check` must keep reporting that limit until controls actually exist.
