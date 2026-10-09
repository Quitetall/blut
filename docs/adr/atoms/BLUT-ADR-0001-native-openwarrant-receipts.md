---
schema: oh.war/atom/v1
adr_uuid: ea1c3c27-5fd3-4747-a203-7a1b8a584584
local_alias: BLUT-ADR-0001
status: proposed
governs:
  - "war://01a122ab-ee43-7108-85fb-baffadf67560"
role: adr
jurisdiction: bound
order: 30
classification: internal
---

# Proposed native receipt protocol

Status: draft, unaccepted. This is a proposed provider protocol, not an established receipt or placeholder canonicalization.

## Producer boundary

An additive BLUT entry point compiles the supplied PlanSpec against the actual Registry and executes through execute_plan. It captures exact canonical plan bytes, the live registry export, executable identity, native job directory, start/end observations and executor outcome. The caller supplies the checked Dispatch identity before execution; the resulting receipt binds Warrant UUID, contract digest, Dispatch digest, stage and attempt. A caller-supplied status cannot produce a completed receipt.

## Signed bytes

Candidate envelope version 1 retains an exact UTF-8 JSON payload as bytes and a detached Ed25519 signature. Verification signs/verifies the exact message `blut-openwarrant-receipt-v1` followed by the payload length as unsigned 64-bit little endian and the payload bytes. No verifier reserializes JSON. Payload parsing rejects duplicate fields, unknown fields, unsupported versions, missing binding fields and contradictory outcome fields. Existing BLUT ContentHash identifies exact retained bytes; its BLAKE3 identity must be labeled separately from OpenWarrant raw SHA-256 capture identity.

Trusted verification keys come from the caller's independently configured provider trust, never from a key embedded in the receipt. Signing keys stay in memory or an operator-provided signer; no key is written to source or scratch. Signing the payload authenticates the collector assertion, not sandbox confinement or human approval.

## Verification and references

A verifier checks signature, exact expected Dispatch binding, exact plan/registry/binary identities and retained native artifact/lineage bytes. Wrong key, altered payload/signature, mixed job, changed plan/catalog, wrong stage/attempt or missing required native references cannot establish completion. Unavailable native evidence remains UNKNOWN. A successful signature alone is insufficient.

Completed, completed-with-warnings, failed, cancelled and deadline-exceeded remain distinct. A run with warnings cannot silently become clean completion. Resource accounting and confinement remain explicit unknown unless the provider observed them. No retroactive conversion of historical OpenWarrant transcripts into native receipts.

## Implementation order

First prove exact-byte signing and refusal through a public provider API. Then wire producer to actual execution and capture native references. Then connect the OpenWarrant ReceiptVerifier seam and run an actual bounded CPU plan. Only independently qualified, authorized integration may clear formal assurance.

## Candidate implementation details

The optional `openwarrant-receipts` feature exposes `framework::openwarrant_receipt`. It enables Ed25519 only; it adds no P2P transport or cloud dependency. `execute_with_receipt` accepts a PlanSpec, the actual Registry, an ExecCtx, a checked caller-supplied Dispatch binding and an in-memory signing key. It refuses any existing job directory and compiles before creating a new one.

The envelope contains exact payload bytes and a 64-byte signature. The version-1 typed payload contains the complete Dispatch binding, generated native run UUID, plan/catalog/executable BLAKE3 identities, start/end milliseconds, terminal outcome, warning/stage counts and a sorted table of native job file paths, lengths, BLAKE3 hashes and modes. Native input records retain the exact canonical plan, live catalogue and run/Dispatch binding. The executor's persisted status stream is captured after its writer is drained. The signing message is the one defined above. Provider receipt identity is BLAKE3 over that signing message followed by the exact 64-byte signature; OpenWarrant raw-envelope SHA-256 remains a separate identity.

The offline `verify` API takes the envelope, independently trusted public key, expected Dispatch, PlanSpec, Registry, retained native job and exact retained producer executable. It rechecks the complete native file table and input records, plus plan/catalog/executable identities. It does not choose trust from a receipt field. Missing native evidence is unavailable; an observed mismatch is rejected. A clean-completion claim requires nonzero stage count, zero warnings and a nonempty native lifecycle stream. The producer does not accept an asserted terminal status.

Resource bounds: payload at most 2 MiB, 4096 job files, 64 MiB per job file, 256 MiB total, 32 directory levels, producer executable at most 1 GiB. Symbolic links and nonregular job files refuse capture. Native paths use UTF-8 normal components joined by `/`; components containing backslashes or control characters refuse capture instead of being renamed in the receipt. This preserves distinct file identities on Unix and Windows. Signing keys are never persisted by this API. Capture can fail after execution; callers must retain the job and must not automatically repeat execution just because receipt capture failed.

This first prototype has no operator key-custody qualification, OpenWarrant callback/CLI integration, confinement proof or metered cost proof. CPU fixture keys are synthetic and confer no project authority. Positive execution, cancellation and refusal tests are implementation evidence. Authenticated duplicate/unknown fields, unsupported versions and contradictory outcome payloads are refused by public-seam controls. Warnings, failed execution, deadline expiry, public transport and broader integration still need explicit controls before qualification.
