# Working in BLUT

BLUT is a domain-neutral semantic compiler and execution engine. `src/` owns
engine behavior; `crates/blut-graph-core/` owns the portable graph contract;
`crates/blut-*` hold sidecars and adapters. Cookbooks live outside this repo.
A cookbook may ship its own Ratatui TUI: BLUT's TUI is optional, not a widget
API that cookbook authors must adopt.

## Governed work

Read `docs/sas/BLUT_SAS.md` before changing architecture or claiming a phase
complete. Read `war next` and `war pins --resolved-only` before changing a
Warrant or a file named by one. `docs/roadmap/README.md` maps governance lanes;
the SAS and individual Warrant atoms carry the actual contracts.

OpenWarrant v1.0.0 has an adopter-phase projection defect: with BLUT's five
§98 phases, `war status`, `war next`, and generated corpus status pages show
OpenWarrant's eleven internal phase titles under a BLUT prefix. Use BLUT's SAS
§98 for objective names and exits; do not rely on `war next` phase priority
until an upstream fix is pinned. Leave generated pages untouched and keep
BLUT's phases independent of the tool's count.

An agent may draft, implement, test, and report. A human owns SAS acceptance,
Warrant authorization, resolution, signer roles, and correction of pinned
deliverables. An independent verifier owns verification. Emit requests for
those acts; never create signatures, role assignments, signer lists, or a
verification of your own work.

Edit authored atoms under `docs/warrants/`, then run `war compile` to refresh
their projections. Do not hand-edit `generated/`. Authorized contract revisions
and resolved deliverables are pinned; amend through the tool's governed path.
Record an unavailable observation as UNKNOWN, never PASS.

For Warrant work, run `war check`, `war compile`, and `war check --generated`.
`war check` validates records; it does not run acceptance gates. Use
`war gate --run` for a probe and `war evidence record <alias>` only when an
authorized Warrant calls for a receipt. State which ran and what remains.

## Code boundaries and gates

Keep domain logic in cookbook repos. Keep network/UI sidecars outside the
portable graph core. Use `CONTRIBUTING.md` for Rust gates and `.github/workflows/ci.yml`
for CI coverage. Local tests and generated-view checks are bounded evidence,
not release or distributed-runtime qualification.
