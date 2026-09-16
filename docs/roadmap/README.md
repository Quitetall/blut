# Governance roadmap

The controlled phase definitions and exits are in
[`../sas/BLUT_SAS.md`](../sas/BLUT_SAS.md) §98. This directory is for
phase-specific records, not a second status ledger. `war status` reports
Warrant state; source and runtime readiness remain bounded by their evidence.

Initial lanes: governance adoption, single-box semantic/runtime proof,
sidecar boundary proof, distributed rung proof, and release train. Each lane
needs its own Warrant and independent verification before its exit can be
claimed.

Known OpenWarrant v1.0.0 defect: its status compiler uses the adopter's §98
phases only when their count equals OpenWarrant's own eleven phases. BLUT has
five, so `war status`, `war next`, and generated corpus objective views show
OpenWarrant phase titles under `BLUT-PHASE-*`. These labels and exits are
invalid for BLUT, and `war next` phase priority may be wrong. Read this repo's
SAS §98 directly; leave generated files untouched. This limits status-dashboard
use, not the validity of the authored
SAS, Warrant atoms, or `war check`'s record-integrity findings.
