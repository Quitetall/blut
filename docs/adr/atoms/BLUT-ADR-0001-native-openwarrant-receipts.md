---
schema: oh.war/atom/v1
warrant_uuid: 01a122ab-ee43-7108-85fb-baffadf67560
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
